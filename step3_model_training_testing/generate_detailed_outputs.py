"""
基于已有 Step3 结果，生成混淆矩阵 CSV、热力图、病人级预测表，并更新总结报告。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime

import pandas as pd

PIPELINE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PIPELINE_DIR not in sys.path:
    sys.path.insert(0, PIPELINE_DIR)

from threshold_common import (  # noqa: E402
    MODEL_DEVICE_DEFAULTS,
    STEP3_OUT,
    THRESHOLD_METHODS,
    _metrics_at_threshold,
    build_model_params,
    build_patient_predictions,
    evaluate_split_with_proba,
    list_models_with_best_summary,
    load_best_summary,
    plot_confusion_matrix_heatmap,
    save_confusion_matrix_csv,
)


def process_model(model_name: str, output_dir: str, device: str | None) -> dict:
    summary_path = os.path.join(output_dir, model_name, "threshold_summary.json")
    if os.path.isfile(summary_path):
        with open(summary_path, encoding="utf-8") as f:
            saved = json.load(f)
        features = saved["best_features"]
        best_params = saved.get("best_params") or {}
        thresholds = saved["final_thresholds"]
        device = device or saved.get("device") or MODEL_DEVICE_DEFAULTS.get(model_name, "cpu")
    else:
        step2 = load_best_summary(model_name)
        features = step2["best_features"]
        best_params = step2.get("best_params") or {}
        thresholds = {m: 0.5 for m in THRESHOLD_METHODS}
        device = device or MODEL_DEVICE_DEFAULTS.get(model_name, "cpu")

    from pipeline_common import load_train_data, scale_pos_weight

    _, y_train, _, _, _ = load_train_data()
    params = build_model_params(
        model_name, best_params, spw=scale_pos_weight(y_train), device=device
    )

    model_dir = os.path.join(output_dir, model_name)
    cm_dir = os.path.join(model_dir, "confusion_matrices")
    plot_dir = os.path.join(model_dir, "confusion_plots")
    pred_dir = os.path.join(model_dir, "patient_predictions")
    os.makedirs(cm_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(pred_dir, exist_ok=True)

    print(f"=== {model_name}: 生成详细输出 (device={device}) ===")
    eval_pack = evaluate_split_with_proba(
        model_name, features, params, device=device, split="test"
    )

    cm_rows = []
    for method in THRESHOLD_METHODS:
        thr = float(thresholds[method])
        metrics = _metrics_at_threshold(eval_pack["y_true"], eval_pack["y_proba"], thr)

        cm_csv = os.path.join(cm_dir, f"{model_name}_{method}_confusion_matrix.csv")
        save_confusion_matrix_csv(metrics, model_name, method, thr, cm_csv)

        plot_path = os.path.join(plot_dir, f"{model_name}_{method}_confusion_matrix.png")
        plot_confusion_matrix_heatmap(metrics, model_name, method, thr, plot_path)

        pred_df = build_patient_predictions(eval_pack, model_name, method, thr)
        pred_csv = os.path.join(pred_dir, f"{model_name}_{method}_patient_predictions.csv")
        pred_df.to_csv(pred_csv, index=False, encoding="utf-8-sig")

        cm_rows.append({
            "model": model_name,
            "threshold_method": method,
            "threshold": thr,
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
            "confusion_matrix_csv": cm_csv,
            "confusion_plot_png": plot_path,
            "patient_predictions_csv": pred_csv,
        })
        print(
            f"  {method}: TN={metrics['TN']} FP={metrics['FP']} "
            f"FN={metrics['FN']} TP={metrics['TP']}"
        )

    summary_df = pd.DataFrame(cm_rows)
    summary_df.to_csv(
        os.path.join(model_dir, "confusion_matrix_summary.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    return {"model": model_name, "rows": cm_rows}


def write_report(output_dir: str, all_rows: list) -> str:
    df = pd.DataFrame(all_rows)
    f2 = df[df["threshold_method"] == "F2_max"].sort_values("PR-AUC", ascending=False)

    lines = [
        "# Step 3 — 模型训练与测试（§2.5 阈值选择）结果总结",
        "",
        f"> 更新时间：{datetime.now():%Y-%m-%d %H:%M}",
        "",
        "## 新增输出（混淆矩阵 & 病人预测）",
        "",
        "每个模型目录下新增：",
        "",
        "```",
        "<Model>/",
        "├── confusion_matrices/     # 2×2 混淆矩阵 CSV",
        "├── confusion_plots/        # 混淆矩阵热力图 PNG",
        "├── patient_predictions/    # 病人级预测表 CSV",
        "└── confusion_matrix_summary.csv",
        "```",
        "",
        "病人预测表字段：`住院号, 真实标签, 白血病类型, 预测概率, 预测标签, 阈值, 阈值方法, 模型`",
        "",
        "## test 集 F2_max 混淆矩阵",
        "",
    ]

    for _, row in f2.iterrows():
        lines.extend([
            f"### {row['model']}（阈值={row['threshold']:.4f}）",
            "",
            "| | 预测阴性 | 预测阳性 |",
            "|---|---------|---------|",
            f"| **实际阴性** | {int(row['TN'])} | {int(row['FP'])} |",
            f"| **实际阳性** | {int(row['FN'])} | {int(row['TP'])} |",
            "",
            f"- Sensitivity={row['sensitivity']:.4f}, Specificity={row['specificity']:.4f}",
            f"- PPV={row['PPV']:.4f}, NPV={row['NPV']:.4f}",
            f"- PR-AUC={row['PR-AUC']:.4f}, AUC={row['AUC']:.4f}",
            f"- 热力图：`{row['confusion_plot_png']}`",
            f"- 病人预测：`{row['patient_predictions_csv']}`",
            "",
        ])

    lines.extend([
        "## test 集排名（F2_max）",
        "",
        "| 排名 | 模型 | PR-AUC | AUC | Sensitivity | Specificity | TP | TN | FP | FN |",
        "|------|------|--------|-----|-------------|-------------|----|----|----|----|",
    ])
    for i, (_, row) in enumerate(f2.iterrows(), 1):
        lines.append(
            f"| {i} | {row['model']} | {row['PR-AUC']:.4f} | {row['AUC']:.4f} | "
            f"{row['sensitivity']:.4f} | {row['specificity']:.4f} | "
            f"{int(row['TP'])} | {int(row['TN'])} | {int(row['FP'])} | {int(row['FN'])} |"
        )

    lines.extend([
        "",
        "## 运行方式",
        "",
        "```bash",
        "cd LeukemiaScreening_Code",
        "python step3_model_training_testing/generate_detailed_outputs.py",
        "```",
        "",
    ])

    report_path = os.path.join(output_dir, "总结.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="生成混淆矩阵/热力图/病人预测表")
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--tabpfn-gpu-id", type=int, default=1)
    parser.add_argument("--output-dir", default=STEP3_OUT)
    args = parser.parse_args()

    models = args.models or [
        m for m in list_models_with_best_summary()
        if os.path.isdir(os.path.join(args.output_dir, m))
    ]
    if not models:
        models = list_models_with_best_summary()

    all_rows = []
    for model_name in models:
        device = args.device or MODEL_DEVICE_DEFAULTS.get(model_name, "cpu")
        if model_name == "TabPFN" and device == "cuda":
            os.environ["CUDA_VISIBLE_DEVICES"] = str(args.tabpfn_gpu_id)
        result = process_model(model_name, args.output_dir, device)
        all_rows.extend(result["rows"])

    if all_rows:
        pd.DataFrame(all_rows).to_csv(
            os.path.join(args.output_dir, "confusion_matrix_all_models.csv"),
            index=False,
            encoding="utf-8-sig",
        )
        report = write_report(args.output_dir, all_rows)
        print(f"\n报告已更新: {report}")


if __name__ == "__main__":
    main()
