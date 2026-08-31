#!/usr/bin/env python3
"""
基于 Step2 已保存的 RFE 结果（特征子集 + 最优超参），重新跑 10 折 CV，
落盘折级 AUC / PR-AUC / Sensitivity / Specificity，并写回带 95% CI 的汇总表。

不重跑 Optuna，保证与现有 RFE 路径一致。Sensitivity/Specificity 阈值固定 0.5
（与原 pipeline_common.run_rfe_optuna 中指标计算一致）。
"""
from __future__ import annotations

import argparse
import ast
import gc
import json
import os
import sys
import traceback
from datetime import datetime

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PIPELINE_DIR)

from pipeline_common import (  # noqa: E402
    STEP2_OUT,
    _apply_device_to_params,
    _fit_predict_proba,
    _lgb_boost_params,
    build_tabpfn_classifier,
    calculate_all_metrics,
    get_model_configs,
    load_cv_folds,
    load_train_data,
    resolve_device,
    scale_pos_weight,
    summarize_fold_metrics,
)

CSV_MAP = {
    "TabPFN": "tabpfn_feature_selection_results.csv",
    "CatBoost": "catboost_feature_selection_results.csv",
    "LightGBM": "lightgbm_feature_selection_results.csv",
    "XGBoost": "xgboost_feature_selection_results.csv",
    "RandomForest": "randomforest_feature_selection_results.csv",
    "SVM": "svm_feature_selection_results.csv",
    "LogisticRegression": "logisticregression_feature_selection_results.csv",
    "KNeighborsClassifier": "kneighborsclassifier_feature_selection_results.csv",
    "QuadraticDiscriminantAnalysis": "quadraticdiscriminantanalysis_feature_selection_results.csv",
    "GaussianNB": "gaussiannb_feature_selection_results.csv",
}

THRESHOLD = 0.5


def _parse_features(s: str) -> list[str]:
    return [x.strip() for x in str(s).split(",") if x.strip()]


def _parse_params(s) -> dict:
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return {}
    text = str(s).strip()
    if not text or text.lower() in {"nan", "none"}:
        return {}
    return ast.literal_eval(text)


def _fit_predict_proba_reeval(
    model_name: str,
    params: dict,
    train_x: pd.DataFrame,
    train_y: pd.Series,
    valid_x: pd.DataFrame,
    valid_y: pd.Series,
    spw: float,
    device: str,
) -> np.ndarray:
    """重评专用：LightGBM 去掉 early_stopping，限制线程，避免卡死/过慢。"""
    if model_name == "LightGBM":
        import lightgbm as lgb

        lgb_params = _lgb_boost_params(params, device)
        lgb_params["num_threads"] = 4
        train_data = lgb.Dataset(train_x, label=train_y)
        model = lgb.train(lgb_params, train_data, num_boost_round=200)
        return model.predict(valid_x)

    if model_name == "CatBoost":
        params = dict(params)
        params["thread_count"] = 4
        params.setdefault("allow_writing_files", False)
    if model_name == "RandomForest":
        params = dict(params)
        params["n_jobs"] = 4
    if model_name == "XGBoost":
        params = dict(params)
        params["n_jobs"] = 4

    return _fit_predict_proba(model_name, params, train_x, train_y, valid_x, valid_y, spw, device)


def reeval_model(model_name: str, output_dir: str, device: str) -> None:
    device = resolve_device(device)
    csv_name = CSV_MAP[model_name]
    model_dir = os.path.join(output_dir, model_name)
    src_csv = os.path.join(model_dir, csv_name)
    if not os.path.isfile(src_csv):
        raise FileNotFoundError(src_csv)

    X_all, y, _, _, _ = load_train_data()
    spw = scale_pos_weight(y)
    folds = load_cv_folds()
    # 若上次中断留下半成品列，优先用 bak 恢复干净输入
    bak_candidates = sorted(
        [p for p in os.listdir(model_dir) if p.startswith(csv_name + ".bak_")],
        reverse=True,
    )
    read_csv = src_csv
    if bak_candidates:
        # 若当前文件已含 fold 汇总列但仍缺 fold_metrics，继续用当前；否则用 bak
        pass
    df = pd.read_csv(read_csv)
    # 去掉上次重评追加列，避免重复
    drop_cols = [c for c in df.columns if c.endswith(("_median", "_mean", "_std", "_ci_lower", "_ci_upper"))
                 or c in ("Sensitivity", "Specificity", "OOF_PR_AUC_reeval", "OOF_AUC_reeval")]
    if drop_cols:
        # 若有 bak，用 bak 作为干净源
        if bak_candidates:
            df = pd.read_csv(os.path.join(model_dir, bak_candidates[0]))
        else:
            df = df.drop(columns=drop_cols, errors="ignore")
    has_params = "最佳参数" in df.columns

    print(f"\n===== {model_name} | device={device} | rows={len(df)} =====", flush=True)

    fold_rows = []
    updated_rows = []

    for _, row in df.iterrows():
        n_feat = int(row["特征数量"])
        feats = _parse_features(row["使用特征"])
        data_x = X_all[feats]
        params = {}
        if model_name != "TabPFN":
            params = dict(get_model_configs(spw)[model_name]["fixed_params"])
            if has_params:
                params.update(_parse_params(row.get("最佳参数")))
            params = _apply_device_to_params(model_name, params, device)
            if model_name == "KNeighborsClassifier" and params.get("metric") == "minkowski" and "p" not in params:
                params["p"] = 2

        fold_metrics = []
        oof = np.full(len(y), np.nan)

        for fold_i, (train_idx, valid_idx) in enumerate(folds):
            train_x = data_x.iloc[train_idx]
            train_y = y.iloc[train_idx]
            valid_x = data_x.iloc[valid_idx]
            valid_y = y.iloc[valid_idx]
            try:
                if model_name == "TabPFN":
                    clf = build_tabpfn_classifier(device)
                    clf.fit(train_x.values, train_y.values)
                    proba = clf.predict_proba(valid_x.values)[:, 1]
                    del clf
                else:
                    proba = _fit_predict_proba_reeval(
                        model_name, params, train_x, train_y, valid_x, valid_y, spw, device
                    )
                oof[valid_idx] = proba
                pred = (proba >= THRESHOLD).astype(int)
                m = calculate_all_metrics(valid_y, pred, proba)
            except Exception as exc:
                print(f"  [warn] n={n_feat} fold={fold_i+1}: {exc}", flush=True)
                m = {k: float("nan") for k in ("auc", "pr_auc", "sensitivity", "specificity")}
            fold_metrics.append(m)
            fold_rows.append({
                "model": model_name,
                "n_features": n_feat,
                "fold": fold_i + 1,
                "auc": m.get("auc", np.nan),
                "pr_auc": m.get("pr_auc", np.nan),
                "sensitivity": m.get("sensitivity", np.nan),
                "specificity": m.get("specificity", np.nan),
                "threshold": THRESHOLD,
            })
            gc.collect()

        fold_dict = {
            k: [m[k] for m in fold_metrics]
            for k in ("auc", "pr_auc", "sensitivity", "specificity")
        }
        summary = summarize_fold_metrics(fold_dict)

        # 保留原 OOF PR-AUC / 原列，并补充折级汇总
        new_row = row.to_dict()
        new_row["Sensitivity"] = summary.get("sensitivity_mean", np.nan)
        new_row["Specificity"] = summary.get("specificity_mean", np.nan)
        # 折均值 AUC（与原逻辑一致）；若原 AUC 为空则填入
        if pd.isna(new_row.get("AUC")):
            new_row["AUC"] = summary.get("auc_mean", np.nan)
        for metric in ("auc", "pr_auc", "sensitivity", "specificity"):
            for suffix in ("median", "mean", "std", "ci_lower", "ci_upper"):
                new_row[f"{metric}_{suffix}"] = summary.get(f"{metric}_{suffix}", np.nan)
        # 另存一份基于折级的 OOF PR-AUC（与 Optuna 目标可比）
        if np.isfinite(oof).all():
            from sklearn.metrics import average_precision_score, roc_auc_score

            new_row["OOF_PR_AUC_reeval"] = float(average_precision_score(y, oof))
            new_row["OOF_AUC_reeval"] = float(roc_auc_score(y, oof))
        updated_rows.append(new_row)
        print(
            f"  n={n_feat:2d}  sens={summary.get('sensitivity_median', float('nan')):.3f} "
            f"[{summary.get('sensitivity_ci_lower', float('nan')):.3f}, "
            f"{summary.get('sensitivity_ci_upper', float('nan')):.3f}]  "
            f"auc_med={summary.get('auc_median', float('nan')):.3f}",
            flush=True,
        )

    out_df = pd.DataFrame(updated_rows)
    # 备份旧结果一次
    bak = src_csv + f".bak_{datetime.now():%Y%m%d_%H%M%S}"
    if not os.path.isfile(bak):
        pd.read_csv(src_csv).to_csv(bak, index=False, encoding="utf-8-sig")
    out_df.to_csv(src_csv, index=False, encoding="utf-8-sig")

    fold_csv = os.path.join(model_dir, f"{model_name.lower()}_fold_metrics.csv")
    pd.DataFrame(fold_rows).to_csv(fold_csv, index=False, encoding="utf-8-sig")

    # 更新 best_summary
    best_path = os.path.join(model_dir, "best_summary.json")
    if os.path.isfile(best_path):
        with open(best_path, encoding="utf-8") as f:
            best = json.load(f)
    else:
        best = {"model": model_name}
    best_n = int(best.get("best_feature_count") or out_df.loc[out_df["PR-AUC"].idxmax(), "特征数量"])
    best_sub = out_df[out_df["特征数量"] == best_n].iloc[0]
    best.update({
        "threshold_for_sens_spec": THRESHOLD,
        "fold_metrics_csv": os.path.basename(fold_csv),
        "best_n_features_metrics": {
            "n_features": best_n,
            "sensitivity_median": float(best_sub.get("sensitivity_median", np.nan)),
            "sensitivity_ci_lower": float(best_sub.get("sensitivity_ci_lower", np.nan)),
            "sensitivity_ci_upper": float(best_sub.get("sensitivity_ci_upper", np.nan)),
            "specificity_median": float(best_sub.get("specificity_median", np.nan)),
            "specificity_ci_lower": float(best_sub.get("specificity_ci_lower", np.nan)),
            "specificity_ci_upper": float(best_sub.get("specificity_ci_upper", np.nan)),
            "auc_median": float(best_sub.get("auc_median", np.nan)),
            "auc_ci_lower": float(best_sub.get("auc_ci_lower", np.nan)),
            "auc_ci_upper": float(best_sub.get("auc_ci_upper", np.nan)),
            "pr_auc_median": float(best_sub.get("pr_auc_median", np.nan)),
            "pr_auc_ci_lower": float(best_sub.get("pr_auc_ci_lower", np.nan)),
            "pr_auc_ci_upper": float(best_sub.get("pr_auc_ci_upper", np.nan)),
        },
        "reeval_at": datetime.now().isoformat(timespec="seconds"),
    })
    with open(best_path, "w", encoding="utf-8") as f:
        json.dump(best, f, ensure_ascii=False, indent=2)

    print(f"  saved: {src_csv}", flush=True)
    print(f"  saved: {fold_csv}", flush=True)


def main():
    p = argparse.ArgumentParser(description="Re-evaluate Step2 RFE configs with fold-level metrics")
    p.add_argument("--models", nargs="*", default=list(CSV_MAP.keys()))
    p.add_argument("--output-dir", default=STEP2_OUT)
    p.add_argument("--device", default="cpu", help="cpu/cuda；TabPFN 建议 cuda")
    p.add_argument("--tabpfn-device", default="cuda")
    p.add_argument("--gpu-id", type=int, default=1, help="CUDA_VISIBLE_DEVICES 用的物理 GPU")
    args = p.parse_args()

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu_id))
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    ok, failed = [], []
    for model in args.models:
        if model not in CSV_MAP:
            print(f"跳过未知模型: {model}")
            continue
        try:
            dev = args.tabpfn_device if model == "TabPFN" else args.device
            # 树模型也可选 cuda；默认 cpu 更稳
            if model in ("CatBoost", "XGBoost", "LightGBM") and args.device == "cuda":
                dev = "cuda"
            reeval_model(model, args.output_dir, dev)
            ok.append(model)
        except Exception:
            failed.append(model)
            traceback.print_exc()

    print("\n===== DONE =====")
    print("ok:", ok)
    print("failed:", failed)
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
