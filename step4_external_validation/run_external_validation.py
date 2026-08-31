"""
Step4 — 外部验证：用 Step2 特征/超参 + Step3 阈值，在宜昌/重庆上评估。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

PIPELINE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PIPELINE_DIR not in sys.path:
    sys.path.insert(0, PIPELINE_DIR)

from pipeline_common import scale_pos_weight  # noqa: E402
from step3_model_training_testing.threshold_common import (  # noqa: E402
    MODEL_DEVICE_DEFAULTS,
    STEP3_OUT,
    THRESHOLD_METHODS,
    _metrics_at_threshold,
    build_model_params,
    list_models_with_best_summary,
    load_best_summary,
    plot_confusion_matrix_heatmap,
    predict_proba,
    save_confusion_matrix_csv,
)

STEP4_DIR = os.path.dirname(os.path.abspath(__file__))
STEP4_DATA = os.path.join(STEP4_DIR, "data")
STEP4_OUT = os.path.join(STEP4_DIR, "outputs")
CENTERS = ("Yichang", "Chongqing")


def load_train_pack():
    X = pd.read_csv(os.path.join(STEP4_DATA, "X_train_rfimput_scaled.csv"))
    y = pd.read_csv(os.path.join(STEP4_DATA, "y_train.csv")).squeeze().astype(int)
    if len(X) != len(y):
        raise RuntimeError(f"train X/y 行数不一致: {len(X)} vs {len(y)}")
    return X, y


def load_center(center: str):
    X = pd.read_csv(os.path.join(STEP4_DATA, f"X_{center}_rfimput_scaled.csv"))
    y = pd.read_csv(os.path.join(STEP4_DATA, f"y_{center}.csv")).squeeze()
    meta = pd.read_csv(os.path.join(STEP4_DATA, f"meta_{center}.csv"))
    mask = pd.read_csv(os.path.join(STEP4_DATA, f"labeled_mask_{center}.csv")).squeeze().astype(bool)
    return X, y, meta, mask


def load_thresholds(model_name: str) -> dict:
    path = os.path.join(STEP3_OUT, model_name, "threshold_summary.json")
    with open(path, encoding="utf-8") as f:
        saved = json.load(f)
    return saved["final_thresholds"], saved


def evaluate_model(model_name: str, device: str | None, output_dir: str) -> list[dict]:
    step2 = load_best_summary(model_name)
    features = step2["best_features"]
    best_params = step2.get("best_params") or {}
    thresholds, step3 = load_thresholds(model_name)
    device = device or step3.get("device") or MODEL_DEVICE_DEFAULTS.get(model_name, "cpu")

    X_train_all, y_train = load_train_pack()
    missing = [f for f in features if f not in X_train_all.columns]
    if missing:
        raise RuntimeError(f"{model_name} 缺少特征: {missing}")
    X_train = X_train_all[features]
    spw = scale_pos_weight(y_train)
    params = build_model_params(model_name, best_params, spw=spw, device=device)

    model_dir = os.path.join(output_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)
    log_path = os.path.join(
        model_dir, f"{model_name.lower()}_external_{datetime.now():%Y%m%d_%H%M%S}.log"
    )
    log_f = open(log_path, "w", encoding="utf-8")

    def log(*a, **k):
        print(*a, **k)
        print(*a, **k, file=log_f)
        log_f.flush()

    log(f"=== {model_name} Step4 外部验证 ===")
    log(f"features={len(features)}, device={device}")
    log(f"thresholds={ {m: round(thresholds[m], 4) for m in THRESHOLD_METHODS} }")

    rows = []
    for center in CENTERS:
        X_all, y_all, meta, mask = load_center(center)
        X_ext = X_all.loc[mask, features]
        y_ext = y_all.loc[mask].astype(int)
        meta_ext = meta.loc[mask].reset_index(drop=True)

        log(f"\n--- {center}: labeled={len(y_ext)}, pos={(y_ext==1).sum()}, neg={(y_ext==0).sum()} ---")
        y_proba = predict_proba(
            model_name, params, X_train, y_train, X_ext, spw=spw, device=device
        )

        center_dir = os.path.join(model_dir, center)
        cm_dir = os.path.join(center_dir, "confusion_matrices")
        plot_dir = os.path.join(center_dir, "confusion_plots")
        pred_dir = os.path.join(center_dir, "patient_predictions")
        os.makedirs(cm_dir, exist_ok=True)
        os.makedirs(plot_dir, exist_ok=True)
        os.makedirs(pred_dir, exist_ok=True)

        for method in THRESHOLD_METHODS:
            thr = float(thresholds[method])
            metrics = _metrics_at_threshold(y_ext.values, y_proba, thr)
            cm_csv = os.path.join(cm_dir, f"{model_name}_{center}_{method}_confusion_matrix.csv")
            save_confusion_matrix_csv(metrics, f"{model_name}/{center}", method, thr, cm_csv)
            plot_path = os.path.join(plot_dir, f"{model_name}_{center}_{method}_confusion_matrix.png")
            plot_confusion_matrix_heatmap(
                metrics, f"{model_name} | {center}", method, thr, plot_path
            )

            y_pred = (y_proba >= thr).astype(int)
            pred_df = meta_ext.copy()
            pred_df["真实标签"] = y_ext.values
            pred_df["预测概率"] = y_proba
            pred_df["预测标签"] = y_pred
            pred_df["阈值"] = thr
            pred_df["阈值方法"] = method
            pred_df["模型"] = model_name
            pred_df["中心"] = center
            pred_csv = os.path.join(
                pred_dir, f"{model_name}_{center}_{method}_patient_predictions.csv"
            )
            pred_df.to_csv(pred_csv, index=False, encoding="utf-8-sig")

            row = {
                "model": model_name,
                "center": center,
                "threshold_method": method,
                "threshold": thr,
                "n_labeled": int(len(y_ext)),
                "n_pos": int((y_ext == 1).sum()),
                "n_neg": int((y_ext == 0).sum()),
                "TN": metrics["TN"],
                "FP": metrics["FP"],
                "FN": metrics["FN"],
                "TP": metrics["TP"],
                "sensitivity": metrics["sensitivity"],
                "specificity": metrics["specificity"],
                "PPV": metrics["ppv"],
                "NPV": metrics["npv"],
                "PR-AUC": metrics["pr_auc"],
                "AUC": metrics["auc"],
                "accuracy": metrics["accuracy"],
                "F1_macro": metrics["f1_macro"],
                "confusion_matrix_csv": cm_csv,
                "confusion_plot_png": plot_path,
                "patient_predictions_csv": pred_csv,
            }
            rows.append(row)
            log(
                f"  {method}: thr={thr:.4f} PR-AUC={metrics['pr_auc']:.4f} "
                f"AUC={metrics['auc']:.4f} sens={metrics['sensitivity']:.4f} "
                f"spec={metrics['specificity']:.4f} "
                f"CM=[TN={metrics['TN']},FP={metrics['FP']},FN={metrics['FN']},TP={metrics['TP']}]"
            )

        # 每个中心保存该模型全部方法汇总
        pd.DataFrame([r for r in rows if r["center"] == center and r["model"] == model_name]).to_csv(
            os.path.join(center_dir, "metrics_summary.csv"), index=False, encoding="utf-8-sig"
        )

    pd.DataFrame(rows).to_csv(
        os.path.join(model_dir, "external_metrics_summary.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    log_f.close()
    return rows


def write_report(output_dir: str, all_rows: list[dict]) -> None:
    df = pd.DataFrame(all_rows)
    lines = [
        "# Step 4 — 外部验证结果总结",
        "",
        f"> 更新时间：{datetime.now():%Y-%m-%d %H:%M}",
        "",
        "## 设置",
        "",
        "- 训练：中南 train（与 Step3 相同预处理：RFimput + Scaled，仅在 train 上 fit）",
        "- 特征/超参：Step2 `best_summary.json`",
        "- 阈值：Step3 `final_thresholds`（不在外部集上重调）",
        "- 重庆 / 宜昌：本实验 `data/external/` 外部验证数据",
        "",
    ]
    for center in CENTERS:
        sub = df[df["center"] == center]
        f2 = sub[sub["threshold_method"] == "F2_max"].sort_values("PR-AUC", ascending=False)
        if f2.empty:
            continue
        n_lab = int(f2.iloc[0]["n_labeled"])
        n_pos = int(f2.iloc[0]["n_pos"])
        n_neg = int(f2.iloc[0]["n_neg"])
        lines.extend([
            f"## {center}（有标签 {n_lab}：阳 {n_pos} / 阴 {n_neg}）— F2_max 排名",
            "",
            "| 排名 | 模型 | PR-AUC | AUC | Sens | Spec | TN | FP | FN | TP | 阈值 |",
            "|------|------|--------|-----|------|------|----|----|----|-----|------|",
        ])
        for i, (_, r) in enumerate(f2.iterrows(), 1):
            lines.append(
                f"| {i} | {r['model']} | {r['PR-AUC']:.4f} | {r['AUC']:.4f} | "
                f"{r['sensitivity']:.4f} | {r['specificity']:.4f} | "
                f"{int(r['TN'])} | {int(r['FP'])} | {int(r['FN'])} | {int(r['TP'])} | "
                f"{r['threshold']:.4f} |"
            )
        lines.append("")
        # top3 confusion
        for _, r in f2.head(3).iterrows():
            lines.extend([
                f"### {r['model']} @ {center}（F2_max, thr={r['threshold']:.4f}）",
                "",
                "| | 预测阴性 | 预测阳性 |",
                "|---|---------|---------|",
                f"| **实际阴性** | {int(r['TN'])} | {int(r['FP'])} |",
                f"| **实际阳性** | {int(r['FN'])} | {int(r['TP'])} |",
                "",
                f"- 热力图：`{r['confusion_plot_png']}`",
                "",
            ])

    lines.extend([
        "## 输出",
        "",
        "```",
        "step4_external_validation/outputs/",
        "├── <Model>/<Center>/confusion_matrices|plots|patient_predictions/",
        "├── external_metrics_all_models.csv",
        "└── 总结.md",
        "```",
        "",
    ])
    path = os.path.join(output_dir, "总结.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"报告已写: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Step4 外部验证")
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--tabpfn-gpu-id", type=int, default=1)
    parser.add_argument("--output-dir", default=STEP4_OUT)
    args = parser.parse_args()

    if not os.path.isfile(os.path.join(STEP4_DATA, "X_Yichang_rfimput_scaled.csv")):
        raise SystemExit("请先运行 prepare_external_data.py")

    models = args.models or [
        m for m in list_models_with_best_summary()
        if os.path.isfile(os.path.join(STEP3_OUT, m, "threshold_summary.json"))
    ]
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"将评估 {len(models)} 个模型: {models}")

    all_rows = []
    failed = []
    for model_name in models:
        device = args.device or MODEL_DEVICE_DEFAULTS.get(model_name, "cpu")
        if model_name == "TabPFN" and device == "cuda":
            os.environ["CUDA_VISIBLE_DEVICES"] = str(args.tabpfn_gpu_id)
        try:
            rows = evaluate_model(model_name, device, args.output_dir)
            all_rows.extend(rows)
        except Exception as exc:
            print(f"[失败] {model_name}: {exc}")
            failed.append({"model": model_name, "error": str(exc)})

    if all_rows:
        out_csv = os.path.join(args.output_dir, "external_metrics_all_models.csv")
        pd.DataFrame(all_rows).to_csv(out_csv, index=False, encoding="utf-8-sig")
        write_report(args.output_dir, all_rows)
        print(f"汇总: {out_csv}")

    with open(os.path.join(args.output_dir, "run_status.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "completed": sorted({r["model"] for r in all_rows}),
                "failed": failed,
                "n_rows": len(all_rows),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )


if __name__ == "__main__":
    main()
