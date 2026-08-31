"""根据 external_metrics_all_models.csv 生成同阈值多模型混淆矩阵拼图，并重写 总结.md。"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

OUT = Path(__file__).resolve().parent / "outputs"
FIG = OUT / "report_figures"
METHODS = ["F1_max", "F2_max", "Sensitivity_0.9", "Specificity_0.9"]
CENTERS = ["Yichang", "Chongqing"]
SHORT = {
    "QuadraticDiscriminantAnalysis": "QDA",
    "KNeighborsClassifier": "KNN",
    "LogisticRegression": "LogReg",
    "RandomForest": "RF",
    "GaussianNB": "GNB",
}


def short(m: str) -> str:
    return SHORT.get(m, m)


def plot_grid(df: pd.DataFrame, center: str, method: str) -> Path:
    sub = df[(df.center == center) & (df.threshold_method == method)].sort_values(
        "PR-AUC", ascending=False
    )
    models = sub["model"].tolist()
    n, ncols = len(models), 5
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.0 * nrows))
    axes = np.atleast_2d(axes)
    for i, model in enumerate(models):
        r, c = divmod(i, ncols)
        ax = axes[r, c]
        row = sub[sub.model == model].iloc[0]
        cm = np.array([[row.TN, row.FP], [row.FN, row.TP]], dtype=float)
        rs = cm.sum(axis=1, keepdims=True)
        rs[rs == 0] = 1
        sns.heatmap(
            cm / rs,
            annot=np.array([[f"{int(row.TN)}", f"{int(row.FP)}"], [f"{int(row.FN)}", f"{int(row.TP)}"]]),
            fmt="",
            cmap="Blues",
            vmin=0,
            vmax=1,
            cbar=False,
            ax=ax,
            xticklabels=["Pred−", "Pred+"],
            yticklabels=["True−", "True+"],
            linewidths=0.5,
            linecolor="white",
            annot_kws={"size": 11, "weight": "bold"},
        )
        ax.set_title(
            f"{short(model)}\nPR={row['PR-AUC']:.3f} AUC={row['AUC']:.3f}\n"
            f"Sen={row.sensitivity:.3f} Spe={row.specificity:.3f}\nthr={row.threshold:.3f}",
            fontsize=9,
            pad=6,
        )
        ax.tick_params(labelsize=8)
    for j in range(n, nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r, c].axis("off")
    fig.suptitle(f"{center}  |  {method}  —  10 models", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    path = FIG / f"{center}_{method}_all_models_cm.png"
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def plot_top4(df: pd.DataFrame, center: str, method: str = "F2_max") -> Path:
    sub = (
        df[(df.center == center) & (df.threshold_method == method)]
        .sort_values("PR-AUC", ascending=False)
        .head(4)
    )
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6))
    for ax, (_, row) in zip(axes, sub.iterrows()):
        cm = np.array([[row.TN, row.FP], [row.FN, row.TP]], dtype=float)
        rs = cm.sum(axis=1, keepdims=True)
        rs[rs == 0] = 1
        sns.heatmap(
            cm / rs,
            annot=np.array(
                [[f"TN {int(row.TN)}", f"FP {int(row.FP)}"], [f"FN {int(row.FN)}", f"TP {int(row.TP)}"]]
            ),
            fmt="",
            cmap="Blues",
            vmin=0,
            vmax=1,
            cbar=False,
            ax=ax,
            xticklabels=["Pred−", "Pred+"],
            yticklabels=["True−", "True+"],
            annot_kws={"size": 12, "weight": "bold"},
        )
        ax.set_title(
            f"{short(row.model)}\nPR-AUC={row['PR-AUC']:.3f}  AUC={row['AUC']:.3f}\n"
            f"Sens={row.sensitivity:.3f}  Spec={row.specificity:.3f}",
            fontsize=10,
        )
    fig.suptitle(f"{center}  |  {method}  —  Top-4 by PR-AUC", fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = FIG / f"{center}_{method}_top4_cm.png"
    fig.savefig(path, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def main() -> None:
    FIG.mkdir(exist_ok=True)
    df = pd.read_csv(OUT / "external_metrics_all_models.csv")
    paths = {}
    for center in CENTERS:
        for method in METHODS:
            paths[(center, method)] = plot_grid(df, center, method)
        paths[(center, "F2_top4")] = plot_top4(df, center, "F2_max")

    def rel(p: Path) -> str:
        return str(p.relative_to(OUT)).replace("\\", "/")

    lines = [
        "# Step 4 — 外部验证结果总结",
        "",
        f"> 更新时间：{datetime.now():%Y-%m-%d %H:%M}",
        "",
        "## 1. 实验设置",
        "",
        "| 项目 | 说明 |",
        "|------|------|",
        "| 训练数据 | 中南 train（RFimput + Scaled，仅在 train 上 fit） |",
        "| 特征 / 超参 | Step2 `best_summary.json` |",
        "| 阈值 | Step3 `final_thresholds`（外部集不重调） |",
        "| 宜昌 | `data/external`（2320：阳 520 / 阴 1800） |",
        "| 重庆 | `data/external`（1260：阳 360 / 阴 900） |",
        "",
        "---",
        "",
        "## 2. 主结果：F2_max（推荐阈值）",
        "",
    ]
    for center in CENTERS:
        sub = df[(df.center == center) & (df.threshold_method == "F2_max")].sort_values(
            "PR-AUC", ascending=False
        )
        n_lab, n_pos, n_neg = int(sub.iloc[0].n_labeled), int(sub.iloc[0].n_pos), int(sub.iloc[0].n_neg)
        lines += [
            f'### {"2.1" if center == "Yichang" else "2.2"} {center}（有标签 {n_lab}：阳 {n_pos} / 阴 {n_neg}）',
            "",
            "| 排名 | 模型 | PR-AUC | AUC | Sens | Spec | TN | FP | FN | TP | 阈值 |",
            "|------|------|--------|-----|------|------|----|----|----|-----|------|",
        ]
        for i, (_, r) in enumerate(sub.iterrows(), 1):
            lines.append(
                f"| {i} | {r['model']} | {r['PR-AUC']:.4f} | {r['AUC']:.4f} | "
                f"{r['sensitivity']:.4f} | {r['specificity']:.4f} | "
                f"{int(r['TN'])} | {int(r['FP'])} | {int(r['FN'])} | {int(r['TP'])} | {r['threshold']:.4f} |"
            )
        lines += [
            "",
            "**Top-4 混淆矩阵对比（F2_max）**",
            "",
            f"![Top4 {center} F2]({rel(paths[(center, 'F2_top4')])})",
            "",
            "**全部 10 模型混淆矩阵（F2_max）**",
            "",
            f"![{center} F2 all]({rel(paths[(center, 'F2_max')])})",
            "",
        ]

    lines += ["---", "", "## 3. 其他阈值：同阈值多模型对比", ""]
    for method in METHODS:
        if method == "F2_max":
            continue
        lines += [f"### {method}", ""]
        for center in CENTERS:
            sub = df[(df.center == center) & (df.threshold_method == method)].sort_values(
                "PR-AUC", ascending=False
            )
            lines += [
                f"#### {center}",
                "",
                "| 排名 | 模型 | PR-AUC | AUC | Sens | Spec | TN | FP | FN | TP |",
                "|------|------|--------|-----|------|------|----|----|----|-----|",
            ]
            for i, (_, r) in enumerate(sub.iterrows(), 1):
                lines.append(
                    f"| {i} | {short(r['model'])} | {r['PR-AUC']:.4f} | {r['AUC']:.4f} | "
                    f"{r['sensitivity']:.4f} | {r['specificity']:.4f} | "
                    f"{int(r['TN'])} | {int(r['FP'])} | {int(r['FN'])} | {int(r['TP'])} |"
                )
            lines += ["", f"![{center} {method}]({rel(paths[(center, method)])})", ""]

    yc = df[(df.center == "Yichang") & (df.threshold_method == "F2_max")].sort_values(
        "PR-AUC", ascending=False
    )
    cq = df[(df.center == "Chongqing") & (df.threshold_method == "F2_max")].sort_values(
        "PR-AUC", ascending=False
    )
    lines += [
        "---",
        "",
        "## 4. 简要结论",
        "",
        f"- **宜昌 F2_max**：最佳 {yc.iloc[0].model}（PR-AUC={yc.iloc[0]['PR-AUC']:.4f}）",
        f"- **重庆 F2_max**：最佳 {cq.iloc[0].model}（PR-AUC={cq.iloc[0]['PR-AUC']:.4f}）",
        "- 拼图目录：`report_figures/`；数值表：`external_metrics_all_models.csv`",
        "",
    ]
    (OUT / "总结.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT / '总结.md'} and {len(paths)} figures in {FIG}")


if __name__ == "__main__":
    main()
