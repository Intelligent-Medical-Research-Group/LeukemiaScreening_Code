"""
Step 5 — TabPFN 可解释性（SHAP）

使用 PriorLabs tabpfn-extensions 提供的 TabPFN 专用 SHAP 解释器
（PermutationExplainer + NaN masking，见 TabPFN 官方 interpretability 扩展）。

解释对象：各评估队列中，F2 阈值下预测标签与真实标签一致的全部记录（论文方法）。
训练集：中南 train 的锁定 19 特征 TabPFN。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

STEP5_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = STEP5_DIR.parent
DATA_DIR = PIPELINE_DIR / "data"
EXTERNAL_DIR = DATA_DIR / "external"
OUT_DIR = STEP5_DIR / "outputs"

sys.path.insert(0, str(PIPELINE_DIR))
from pipeline_common import TABPFN_MODEL_PATH, build_tabpfn_classifier  # noqa: E402

CENTERS = ("Zhongnan_test", "Chongqing", "Yichang")
RANDOM_STATE = 42
F2_THRESHOLD = 0.47680647671222687  # Step3 TabPFN F2_max（论文主工作点）
# Locked TabPFN model: Step1 dropped MCV; TabPFN RFE dropped UA and sex.
TABPFN_FEATURES_19 = [
    "A/G",
    "ALB",
    "APTT",
    "BASO#",
    "DBIL",
    "GLB",
    "HCT",
    "LDH",
    "MCH",
    "MCHC",
    "MPV",
    "NEUT#",
    "PLT",
    "PTTA",
    "RBC",
    "RDW",
    "TBIL",
    "WBC",
    "年龄",
]


def load_features() -> list[str]:
    return list(TABPFN_FEATURES_19)


def load_train() -> tuple[pd.DataFrame, pd.Series]:
    X = pd.read_csv(DATA_DIR / "X_train_rfimput_scaled.csv")
    y = pd.read_csv(DATA_DIR / "y_train.csv").squeeze().astype(int)
    return X, y


def load_center_table(center: str) -> pd.DataFrame:
    path = EXTERNAL_DIR / f"{center}.csv"
    if not path.exists():
        raise FileNotFoundError(f"未找到外部数据: {path}")
    return pd.read_csv(path)


def select_f2_matching_records(
    df: pd.DataFrame,
    clf,
    features: list[str],
    threshold: float = F2_THRESHOLD,
) -> pd.DataFrame:
    """保留 F2 阈值下预测标签与真实标签一致的全部记录。"""
    y_true = df["is_leukemia"].astype(int).to_numpy()
    y_proba = clf.predict_proba(df[features].to_numpy(dtype=float))[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    mask = y_pred == y_true
    out = df.loc[mask].copy().reset_index(drop=True)
    out["proba_TabPFN"] = y_proba[mask]
    out["pred_TabPFN"] = y_pred[mask]
    return out


def fit_tabpfn(X_train: pd.DataFrame, y_train: pd.Series, device: str):
    clf = build_tabpfn_classifier(
        device,
        fit_mode="fit_with_cache",
        ignore_pretraining_limits=(device == "cpu"),
    )
    clf.fit(X_train.values, y_train.values)
    return clf


def compute_shap(clf, X_explain: pd.DataFrame, features: list[str]):
    import shap
    from tabpfn_extensions.interpretability.shap import get_shap_values

    raw = get_shap_values(clf, X_explain[features], attribute_names=features)
    vals = np.asarray(raw.values if hasattr(raw, "values") else raw)
    if vals.ndim == 3 and vals.shape[-1] == 2:
        vals = vals[..., 1]
    return shap.Explanation(
        values=vals,
        data=X_explain[features].to_numpy(dtype=float),
        feature_names=features,
    )


def shap_to_long(exp, sample_df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows = []
    for i in range(len(sample_df)):
        pid = sample_df.loc[i, "patient_id"]
        label = int(sample_df.loc[i, "is_leukemia"])
        for j, feat in enumerate(features):
            rows.append({
                "patient_id": pid,
                "is_leukemia": label,
                "feature": feat,
                "feature_value": float(sample_df.loc[i, feat]),
                "shap_value": float(exp.values[i, j]),
            })
    return pd.DataFrame(rows)


def mean_abs_importance(long_df: pd.DataFrame) -> pd.DataFrame:
    imp = (
        long_df.groupby("feature")["shap_value"]
        .apply(lambda s: float(np.mean(np.abs(s))))
        .sort_values(ascending=False)
        .reset_index()
    )
    imp.columns = ["feature", "mean_abs_shap"]
    imp["rank"] = np.arange(1, len(imp) + 1)
    return imp


def save_plots(center: str, exp, sample_df: pd.DataFrame, features: list[str], out_center: Path):
    import shap

    out_center.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    shap.summary_plot(exp, sample_df[features], show=False, max_display=len(features))
    plt.tight_layout()
    plt.savefig(out_center / "beeswarm.png", dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 5))
    shap.plots.bar(exp, show=False, max_display=min(15, len(features)))
    plt.tight_layout()
    plt.savefig(out_center / "bar_mean_abs.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 选一个 TP / FP 做 waterfall（若存在 proba_TabPFN）
    proba_col = "proba_TabPFN" if "proba_TabPFN" in sample_df.columns else None
    thr = F2_THRESHOLD
    if proba_col:
        y_true = sample_df["is_leukemia"].astype(int).to_numpy()
        y_pred = (sample_df[proba_col].astype(float) >= thr).astype(int)
        tp_idx = np.where((y_true == 1) & (y_pred == 1))[0]
        fp_idx = np.where((y_true == 0) & (y_pred == 1))[0]
        for tag, idxs in [("tp", tp_idx), ("fp", fp_idx)]:
            if len(idxs):
                i = int(idxs[0])
                plt.figure(figsize=(9, 5))
                shap.plots.waterfall(exp[i], show=False, max_display=min(15, len(features)))
                plt.tight_layout()
                plt.savefig(out_center / f"waterfall_{tag}.png", dpi=150, bbox_inches="tight")
                plt.close()


def run_center(
    center: str,
    clf,
    features: list[str],
) -> dict:
    df = load_center_table(center)
    sample_df = select_f2_matching_records(df, clf, features)
    X_ex = sample_df[features]

    print(
        f"  [{center}] F2-matching {len(sample_df)}/{len(df)} samples; computing SHAP ..."
    )
    exp = compute_shap(clf, X_ex, features)
    long_df = shap_to_long(exp, sample_df, features)
    imp = mean_abs_importance(long_df)

    out_center = OUT_DIR / center
    out_center.mkdir(parents=True, exist_ok=True)
    long_df.to_csv(out_center / "shap_values_long.csv", index=False, encoding="utf-8-sig")
    imp.to_csv(out_center / "mean_abs_importance.csv", index=False, encoding="utf-8-sig")
    sample_df.to_csv(out_center / "explained_samples.csv", index=False, encoding="utf-8-sig")
    try:
        save_plots(center, exp, sample_df, features, out_center)
    except Exception as exc:
        print(f"  [{center}] skipped pipeline plots ({exc})")
        plt.close("all")

    return {
        "center": center,
        "n_samples": int(len(df)),
        "n_explained": int(len(sample_df)),
        "n_pos_explained": int(sample_df["is_leukemia"].sum()),
        "n_neg_explained": int((sample_df["is_leukemia"] == 0).sum()),
        "top5_features": imp.head(5)[["feature", "mean_abs_shap"]].to_dict("records"),
    }


def write_report(summaries: list[dict], meta: dict) -> None:
    lines = [
        "# Step 5 — TabPFN 可解释性（SHAP）",
        "",
        f"> 更新时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## 1. 方法",
        "",
        "| 项目 | 说明 |",
        "|------|------|",
        "| 模型 | TabPFN（与 Step3 相同 checkpoint，`fit_mode=fit_with_cache`） |",
        "| 解释方法 | `tabpfn-extensions` 官方 SHAP（TabPFN 专用 PermutationExplainer + NaN masking） |",
        "| 训练集 | 中南 train 的锁定 19 特征（RFimput + StandardScaler） |",
        "| 解释对象 | F2 阈值下预测与标签一致的全部记录 |",
        "| 输出类别 | 白血病阳性类（`predict_proba[:, 1]`）的 SHAP 值 |",
        "",
        "参考：[PriorLabs/TabPFN](https://github.com/PriorLabs/TabPFN) — Interpretability 扩展（`tabpfn-extensions[interpretability]`）。",
        "",
        "---",
        "",
        "## 2. 各中心 Top 特征（mean |SHAP|）",
        "",
    ]
    for s in summaries:
        lines += [
            f"### {s['center']}",
            "",
            f"- 该中心 {s['n_samples']} 例；本次解释 {s['n_explained']} 例（阳 {s['n_pos_explained']} / 阴 {s['n_neg_explained']}）",
            "",
            "| 排名 | 特征 | mean \\|SHAP\\| |",
            "|------|------|-------------|",
        ]
        for i, row in enumerate(s["top5_features"], 1):
            lines.append(f"| {i} | {row['feature']} | {row['mean_abs_shap']:.4f} |")
        lines += [
            "",
            f"![{s['center']} beeswarm]({s['center']}/beeswarm.png)",
            "",
            f"![{s['center']} bar]({s['center']}/bar_mean_abs.png)",
            "",
        ]

    lines += [
        "---",
        "",
        "## 3. 输出文件",
        "",
        "- 各中心目录：`{center}/shap_values_long.csv`（逐样本逐特征 SHAP）",
        "- `{center}/mean_abs_importance.csv` — 特征重要性排序",
        "- `{center}/explained_samples.csv` — 被解释的样本明细",
        "- `{center}/beeswarm.png`, `bar_mean_abs.png`, `waterfall_tp.png`, `waterfall_fp.png`",
        "",
        f"- 运行参数见 `run_meta.json`",
        "",
    ]
    (OUT_DIR / "总结.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Step5 TabPFN SHAP interpretability")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--tabpfn-gpu-id", type=int, default=1)
    parser.add_argument("--centers", nargs="+", default=list(CENTERS))
    args = parser.parse_args()

    if args.device == "cpu":
        os.environ["TABPFN_ALLOW_CPU_LARGE_DATASET"] = "1"
    elif args.device == "cuda":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.tabpfn_gpu_id)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    features = load_features()
    X_train, y_train = load_train()
    missing = [f for f in features if f not in X_train.columns]
    if missing:
        raise KeyError(f"Training matrix missing locked TabPFN features: {missing}")
    X_train = X_train[features]

    print(f"Train: {len(X_train)} samples, {len(features)} features (locked TabPFN set)")
    print("Fitting TabPFN ...")
    clf = fit_tabpfn(X_train, y_train, args.device)
    print("TabPFN fit done.")

    summaries = []
    for center in args.centers:
        print(f"\n===== {center} =====")
        summaries.append(run_center(center, clf, features))

    meta = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "method": "tabpfn_extensions.get_shap_values; all test-set records whose prediction matched the label at F2 cutoff",
        "model_path": TABPFN_MODEL_PATH,
        "device": args.device,
        "gpu": str(args.tabpfn_gpu_id) if args.device == "cuda" else None,
        "n_features": len(features),
        "features": features,
        "selection": "all predicted-matching records from each test set",
        "threshold": F2_THRESHOLD,
        "centers": summaries,
    }
    with open(OUT_DIR / "run_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    write_report(summaries, meta)
    print(f"\nWrote {OUT_DIR / '总结.md'}")
    print("DONE")


if __name__ == "__main__":
    main()
