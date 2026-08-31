"""
Step 3 — §2.5 模型训练与测试：10折CV学阈值 + 全训练集重训 + test评估。

四种阈值方法：F1_max / F2_max / Sensitivity_0.9 / Specificity_0.9
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
    evaluate_on_split,
    evaluate_split_with_proba,
    learn_thresholds_cv,
    list_models_with_best_summary,
    load_best_summary,
    metrics_to_result_row,
    plot_confusion_matrix_heatmap,
    save_confusion_matrix_csv,
)


def run_model(model_name: str, device: str | None, output_dir: str) -> dict:
    summary = load_best_summary(model_name)
    features = summary["best_features"]
    feature_count = int(summary.get("best_feature_count", len(features)))
    best_params = summary.get("best_params") or {}
    device = device or MODEL_DEVICE_DEFAULTS.get(model_name, "cpu")

    model_dir = os.path.join(output_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)
    log_path = os.path.join(
        model_dir,
        f"{model_name.lower()}_threshold_eval_{datetime.now():%Y%m%d_%H%M%S}.log",
    )
    log_file = open(log_path, "w", encoding="utf-8")

    def log_print(*args, **kwargs):
        print(*args, **kwargs)
        print(*args, **kwargs, file=log_file)
        log_file.flush()

    log_print(f"=== {model_name} Step3 阈值评估 ===")
    log_print(f"特征数: {feature_count}, device: {device}")
    log_print(f"特征: {features}")

    from pipeline_common import load_train_data, scale_pos_weight

    _, y_train, _, _, _ = load_train_data()
    params = build_model_params(
        model_name, best_params, spw=scale_pos_weight(y_train), device=device
    )

    log_print("\n--- 10折CV 学习阈值 ---")
    final_thresholds, fold_thresholds, cv_summary = learn_thresholds_cv(
        model_name, features, params, device, log_print=log_print
    )
    for method in THRESHOLD_METHODS:
        log_print(
            f"  {method}: median={final_thresholds[method]:.4f}, "
            f"folds={[round(x, 4) for x in fold_thresholds[method]]}"
        )

    log_print("\n--- 全训练集重训 + test 评估 ---")
    eval_pack = evaluate_split_with_proba(
        model_name, features, params, device=device, split="test"
    )
    test_results = {
        method: _metrics_at_threshold(eval_pack["y_true"], eval_pack["y_proba"], thr)
        for method, thr in final_thresholds.items()
    }

    cm_dir = os.path.join(model_dir, "confusion_matrices")
    plot_dir = os.path.join(model_dir, "confusion_plots")
    pred_dir = os.path.join(model_dir, "patient_predictions")
    os.makedirs(cm_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(pred_dir, exist_ok=True)

    cv_rows = []
    cm_rows = []
    for method in THRESHOLD_METHODS:
        row = {
            "model": model_name,
            "threshold_method": method,
            "feature_count": feature_count,
            **cv_summary[method],
        }
        cv_rows.append(row)

        test_row = metrics_to_result_row(
            feature_count, final_thresholds[method], test_results[method]
        )
        test_df = pd.DataFrame([test_row])
        out_csv = os.path.join(model_dir, f"{model_name}_{method}_test.csv")
        test_df.to_csv(out_csv, index=False, encoding="utf-8-sig")

        thr = final_thresholds[method]
        metrics = test_results[method]
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
            "confusion_matrix_csv": cm_csv,
            "confusion_plot_png": plot_path,
            "patient_predictions_csv": pred_csv,
        })

        log_print(
            f"  {method}: thr={thr:.4f}, "
            f"test PR-AUC={metrics['pr_auc']:.4f}, "
            f"sens={metrics['sensitivity']:.4f}, "
            f"spec={metrics['specificity']:.4f}, "
            f"CM=[TN={metrics['TN']}, FP={metrics['FP']}, FN={metrics['FN']}, TP={metrics['TP']}]"
        )

    if cm_rows:
        pd.DataFrame(cm_rows).to_csv(
            os.path.join(model_dir, "confusion_matrix_summary.csv"),
            index=False,
            encoding="utf-8-sig",
        )

    with open(os.path.join(model_dir, "threshold_summary.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": model_name,
                "feature_count": feature_count,
                "best_features": features,
                "best_params": best_params,
                "device": device,
                "final_thresholds": final_thresholds,
                "fold_thresholds": fold_thresholds,
                "cv_summary": cv_summary,
                "test_results": test_results,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    log_file.close()
    return {"model": model_name, "cv_rows": cv_rows, "test_results": test_results}


def main() -> None:
    parser = argparse.ArgumentParser(description="Step3 阈值选择与 test 评估")
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="指定模型名；默认处理所有存在 best_summary.json 的模型",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="覆盖所有模型的 device（TabPFN 建议 cuda）",
    )
    parser.add_argument(
        "--tabpfn-gpu-id",
        type=int,
        default=1,
        help="TabPFN 使用的 GPU 编号（通过 CUDA_VISIBLE_DEVICES 设置）",
    )
    parser.add_argument(
        "--output-dir",
        default=STEP3_OUT,
        help="输出目录",
    )
    args = parser.parse_args()

    models = args.models or list_models_with_best_summary()
    if not models:
        raise SystemExit("未找到任何 Step2 best_summary.json，请先完成 step2。")

    os.makedirs(args.output_dir, exist_ok=True)
    all_cv_rows = []
    completed = []
    failed = []

    print(f"将处理 {len(models)} 个模型: {models}")
    for model_name in models:
        device = args.device or MODEL_DEVICE_DEFAULTS.get(model_name, "cpu")
        if model_name == "TabPFN" and device == "cuda":
            os.environ["CUDA_VISIBLE_DEVICES"] = str(args.tabpfn_gpu_id)
        try:
            result = run_model(model_name, device, args.output_dir)
            all_cv_rows.extend(result["cv_rows"])
            completed.append(model_name)
        except Exception as exc:
            print(f"[失败] {model_name}: {exc}")
            failed.append({"model": model_name, "error": str(exc)})

    if all_cv_rows:
        cv_df = pd.DataFrame(all_cv_rows)
        cv_path = os.path.join(args.output_dir, "threshold_cv_summary.csv")
        cv_df.to_csv(cv_path, index=False, encoding="utf-8-sig")
        print(f"\nCV 汇总已保存: {cv_path}")

    status = {
        "completed": completed,
        "failed": failed,
        "n_completed": len(completed),
        "n_failed": len(failed),
    }
    with open(os.path.join(args.output_dir, "run_status.json"), "w", encoding="utf-8") as f:
        json.dump(status, f, ensure_ascii=False, indent=2)

    print(f"\n完成: {len(completed)}/{len(models)}")
    if failed:
        print("失败模型:")
        for item in failed:
            print(f"  - {item['model']}: {item['error']}")


if __name__ == "__main__":
    main()
