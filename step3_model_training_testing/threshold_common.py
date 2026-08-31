"""
Step 3 阈值选择与模型评估共享工具。
"""
from __future__ import annotations

import gc
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import confusion_matrix, precision_recall_curve

PIPELINE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PIPELINE_DIR not in sys.path:
    sys.path.insert(0, PIPELINE_DIR)

from pipeline_common import (  # noqa: E402
    DATA_DIR,
    STEP2_OUT,
    _apply_device_to_params,
    _fit_balanced_if_supported,
    _lgb_boost_params,
    _xgb_params_for_y,
    build_tabpfn_classifier,
    calculate_all_metrics,
    get_model_configs,
    load_cv_folds,
    load_train_data,
    scale_pos_weight,
)

STEP3_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")

THRESHOLD_METHODS = (
    "F1_max",
    "F2_max",
    "Sensitivity_0.9",
    "Specificity_0.9",
)

MODEL_DEVICE_DEFAULTS = {
    "TabPFN": "cuda",
    "CatBoost": "cpu",
    "XGBoost": "cpu",
    "LightGBM": "cpu",
}


def load_test_data() -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    X = pd.read_csv(os.path.join(DATA_DIR, "X_test_rfimput_scaled.csv"))
    y = pd.read_csv(os.path.join(DATA_DIR, "y_test.csv")).squeeze().astype(int)
    groups = pd.read_csv(os.path.join(DATA_DIR, "groups_test.csv")).squeeze().astype(str)
    return X, y, groups


def list_models_with_best_summary(step2_out: str = STEP2_OUT) -> List[str]:
    models = []
    if not os.path.isdir(step2_out):
        return models
    for name in sorted(os.listdir(step2_out)):
        summary_path = os.path.join(step2_out, name, "best_summary.json")
        if os.path.isfile(summary_path):
            models.append(name)
    return models


def load_best_summary(model_name: str, step2_out: str = STEP2_OUT) -> dict:
    path = os.path.join(step2_out, model_name, "best_summary.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"未找到 {path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_model_params(model_name: str, best_params: dict, spw: float, device: str = "cpu") -> dict:
    if model_name == "TabPFN":
        return {}
    config = get_model_configs(spw)[model_name]
    params = dict(config["fixed_params"])
    params.update(best_params or {})
    params = _apply_device_to_params(model_name, params, device)
    if model_name == "KNeighborsClassifier" and params.get("metric") == "minkowski" and "p" not in params:
        params["p"] = 2
    return params


def predict_proba(
    model_name: str,
    params: dict,
    train_x: pd.DataFrame,
    train_y: pd.Series,
    pred_x: pd.DataFrame,
    spw: float,
    device: str = "cpu",
) -> np.ndarray:
    import lightgbm as lgb

    if model_name == "TabPFN":
        clf = build_tabpfn_classifier(device)
        clf.fit(train_x.values, train_y.values)
        return clf.predict_proba(pred_x.values)[:, 1]

    if model_name == "LightGBM":
        lgb_params = _lgb_boost_params(params, device)
        train_data = lgb.Dataset(train_x, label=train_y)
        model = lgb.train(lgb_params, train_data, num_boost_round=200)
        return model.predict(pred_x, num_iteration=model.best_iteration or 200)

    if model_name == "KNeighborsClassifier":
        from sklearn.neighbors import KNeighborsClassifier

        base = KNeighborsClassifier(**params)
        clf = CalibratedClassifierCV(base, cv=5, method="isotonic")
        _fit_balanced_if_supported(clf, train_x, train_y)
        return clf.predict_proba(pred_x)[:, 1]

    if model_name == "GaussianNB":
        from sklearn.naive_bayes import GaussianNB

        clf = GaussianNB(**params)
        _fit_balanced_if_supported(clf, train_x, train_y)
        return clf.predict_proba(pred_x)[:, 1]

    if model_name == "QuadraticDiscriminantAnalysis":
        from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

        clf = QuadraticDiscriminantAnalysis(**params)
        _fit_balanced_if_supported(clf, train_x, train_y)
        return clf.predict_proba(pred_x)[:, 1]

    if model_name == "XGBoost":
        from xgboost import XGBClassifier

        clf = XGBClassifier(**_xgb_params_for_y(params, train_y))
        clf.fit(train_x, train_y)
        return clf.predict_proba(pred_x)[:, 1]

    config = get_model_configs(spw)[model_name]
    clf = config["class"](**params)
    clf.fit(train_x, train_y)
    return clf.predict_proba(pred_x)[:, 1]


def _metrics_at_threshold(y_true: np.ndarray, y_proba: np.ndarray, threshold: float) -> dict:
    y_pred = (y_proba >= threshold).astype(int)
    base = calculate_all_metrics(y_true, y_pred, y_proba)
    cm = confusion_matrix(y_true, y_pred)
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = fp = fn = tp = 0
    return {
        "threshold": float(threshold),
        "accuracy": base["accuracy"],
        "sensitivity": base["sensitivity"],
        "specificity": base["specificity"],
        "ppv": base["ppv"],
        "npv": base["npv"],
        "f1_micro": base["f1_micro"],
        "f1_macro": base["f1_macro"],
        "auc": base["auc"],
        "pr_auc": base["pr_auc"],
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
    }


def _candidate_thresholds(y_true: np.ndarray, y_proba: np.ndarray) -> np.ndarray:
    _, _, pr_thresholds = precision_recall_curve(y_true, y_proba)
    candidates = np.unique(np.concatenate([pr_thresholds, np.linspace(0.0, 1.0, 201)]))
    return np.sort(candidates)


def find_thresholds(y_true: np.ndarray, y_proba: np.ndarray) -> Dict[str, float]:
    """四种阈值策略；Sensitivity/Specificity 约束使用真实混淆矩阵指标。"""
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    thresholds = _candidate_thresholds(y_true, y_proba)

    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_true, y_proba)
    f1_scores = 2 * (precision_curve * recall_curve) / (precision_curve + recall_curve + 1e-8)
    f2_scores = 5 * (precision_curve * recall_curve) / (4 * precision_curve + recall_curve + 1e-8)

    best_f1_idx = int(np.argmax(f1_scores))
    best_f2_idx = int(np.argmax(f2_scores))
    f1_threshold = float(pr_thresholds[best_f1_idx]) if best_f1_idx < len(pr_thresholds) else 0.5
    f2_threshold = float(pr_thresholds[best_f2_idx]) if best_f2_idx < len(pr_thresholds) else 0.5

    sens_spec_pairs = []
    spec_sens_pairs = []
    for thr in thresholds:
        m = _metrics_at_threshold(y_true, y_proba, float(thr))
        sens_spec_pairs.append((m["sensitivity"], m["specificity"], float(thr)))
        spec_sens_pairs.append((m["specificity"], m["sensitivity"], float(thr)))

    sens_ge_09 = [x for x in sens_spec_pairs if x[0] >= 0.9]
    if sens_ge_09:
        sensitivity_09_threshold = max(sens_ge_09, key=lambda x: (x[1], -x[2]))[2]
    else:
        sensitivity_09_threshold = max(sens_spec_pairs, key=lambda x: (x[0], x[1], -x[2]))[2]

    spec_ge_09 = [x for x in spec_sens_pairs if x[0] >= 0.9]
    if spec_ge_09:
        specificity_09_threshold = max(spec_ge_09, key=lambda x: (x[1], -x[2]))[2]
    else:
        specificity_09_threshold = max(spec_sens_pairs, key=lambda x: (x[0], x[1], -x[2]))[2]

    return {
        "F1_max": f1_threshold,
        "F2_max": f2_threshold,
        "Sensitivity_0.9": float(sensitivity_09_threshold),
        "Specificity_0.9": float(specificity_09_threshold),
    }


def aggregate_thresholds(fold_thresholds: Dict[str, List[float]]) -> Dict[str, float]:
    return {method: float(np.median(vals)) for method, vals in fold_thresholds.items() if vals}


def metrics_to_result_row(
    feature_count: int,
    threshold: float,
    metrics: dict,
) -> dict:
    return {
        "特征数量": feature_count,
        "最优阈值": threshold,
        "PR-AUC": metrics["pr_auc"],
        "TP": metrics["TP"],
        "TN": metrics["TN"],
        "FP": metrics["FP"],
        "FN": metrics["FN"],
        "accuracy": metrics["accuracy"],
        "sensitivity": metrics["sensitivity"],
        "specificity": metrics["specificity"],
        "PPV": metrics["ppv"],
        "NPV": metrics["npv"],
        "F1_micro": metrics["f1_micro"],
        "F1_macro": metrics["f1_macro"],
        "AUC": metrics["auc"],
    }


def learn_thresholds_cv(
    model_name: str,
    features: List[str],
    params: dict,
    device: str,
    log_print=print,
) -> Tuple[Dict[str, float], Dict[str, List[float]], Dict[str, dict]]:
    X, y, _, _, _ = load_train_data()
    X = X[features]
    spw = scale_pos_weight(y)
    fold_indices = load_cv_folds()

    fold_thresholds = {m: [] for m in THRESHOLD_METHODS}
    fold_metrics = {m: [] for m in THRESHOLD_METHODS}

    for fold_idx, (train_idx, valid_idx) in enumerate(fold_indices):
        log_print(f"  CV 折 {fold_idx + 1}/10: 训练并学习阈值...")
        train_x = X.iloc[train_idx]
        train_y = y.iloc[train_idx]
        valid_x = X.iloc[valid_idx]
        valid_y = y.iloc[valid_idx]

        valid_proba = predict_proba(
            model_name, params, train_x, train_y, valid_x, spw, device=device
        )
        thr_map = find_thresholds(valid_y.values, valid_proba)
        for method, thr in thr_map.items():
            fold_thresholds[method].append(thr)
            fold_metrics[method].append(
                _metrics_at_threshold(valid_y.values, valid_proba, thr)
            )
        gc.collect()

    final_thresholds = aggregate_thresholds(fold_thresholds)
    cv_summary = {}
    for method in THRESHOLD_METHODS:
        vals = fold_metrics[method]
        cv_summary[method] = {
            "threshold_median": final_thresholds[method],
            "threshold_mean": float(np.mean(fold_thresholds[method])),
            "threshold_std": float(np.std(fold_thresholds[method])),
        }
        if vals:
            for key in ("accuracy", "sensitivity", "specificity", "ppv", "npv", "f1_micro", "f1_macro", "auc", "pr_auc"):
                cv_summary[method][f"cv_{key}_mean"] = float(np.mean([v[key] for v in vals]))
                cv_summary[method][f"cv_{key}_std"] = float(np.std([v[key] for v in vals]))

    return final_thresholds, fold_thresholds, cv_summary


def evaluate_on_split(
    model_name: str,
    features: List[str],
    params: dict,
    thresholds: Dict[str, float],
    device: str,
    split: str = "test",
) -> Dict[str, dict]:
    eval_pack = evaluate_split_with_proba(
        model_name, features, params, device=device, split=split
    )
    y_true = eval_pack["y_true"]
    y_proba = eval_pack["y_proba"]

    results = {}
    for method, thr in thresholds.items():
        results[method] = _metrics_at_threshold(y_true, y_proba, thr)
    return results


def evaluate_split_with_proba(
    model_name: str,
    features: List[str],
    params: dict,
    device: str,
    split: str = "test",
) -> dict:
    X_train, y_train, _, _, _ = load_train_data()
    meta = None
    groups = None
    if split == "test":
        X_eval, y_eval, groups = load_test_data()
        meta = pd.read_csv(os.path.join(DATA_DIR, "meta_test.csv"))
    else:
        X_eval, y_eval = X_train, y_train
        groups = pd.read_csv(os.path.join(DATA_DIR, "groups_train.csv")).squeeze().astype(str)
        meta = pd.read_csv(os.path.join(DATA_DIR, "meta_train.csv"))

    X_train = X_train[features]
    X_eval = X_eval[features]
    spw = scale_pos_weight(y_train)

    y_proba = predict_proba(
        model_name, params, X_train, y_train, X_eval, spw, device=device
    )
    return {
        "y_true": y_eval.values,
        "y_proba": y_proba,
        "groups": groups.values,
        "meta": meta,
        "split": split,
    }


def confusion_matrix_table(metrics: dict) -> pd.DataFrame:
    """返回标准 2x2 混淆矩阵表。"""
    return pd.DataFrame(
        [
            ["实际阴性", metrics["TN"], metrics["FP"]],
            ["实际阳性", metrics["FN"], metrics["TP"]],
        ],
        columns=["", "预测阴性", "预测阳性"],
    )


def save_confusion_matrix_csv(
    metrics: dict,
    model_name: str,
    method: str,
    threshold: float,
    out_path: str,
) -> None:
    cm_df = confusion_matrix_table(metrics)
    summary = pd.DataFrame(
        [{
            "model": model_name,
            "threshold_method": method,
            "threshold": threshold,
            "TN": metrics["TN"],
            "FP": metrics["FP"],
            "FN": metrics["FN"],
            "TP": metrics["TP"],
            "accuracy": metrics["accuracy"],
            "sensitivity": metrics["sensitivity"],
            "specificity": metrics["specificity"],
            "PPV": metrics["ppv"],
            "NPV": metrics["npv"],
            "PR-AUC": metrics["pr_auc"],
            "AUC": metrics["auc"],
        }]
    )
    with open(out_path, "w", encoding="utf-8-sig") as f:
        summary.to_csv(f, index=False)
        f.write("\n")
        cm_df.to_csv(f, index=False)


def plot_confusion_matrix_heatmap(
    metrics: dict,
    model_name: str,
    method: str,
    threshold: float,
    out_path: str,
) -> None:
    import matplotlib.pyplot as plt

    cm = np.array([[metrics["TN"], metrics["FP"]], [metrics["FN"], metrics["TP"]]])
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Predicted Negative", "Predicted Positive"])
    ax.set_yticklabels(["Actual Negative", "Actual Positive"])
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Actual label")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center", color="black", fontsize=14)
    ax.set_title(f"{model_name} | {method}\nThreshold = {threshold:.4f}")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def build_patient_predictions(
    eval_pack: dict,
    model_name: str,
    method: str,
    threshold: float,
) -> pd.DataFrame:
    meta = eval_pack["meta"].copy()
    y_true = eval_pack["y_true"]
    y_proba = eval_pack["y_proba"]
    y_pred = (y_proba >= threshold).astype(int)

    out = pd.DataFrame({
        "住院号": meta["patient_id"].astype(str).values,
        "真实标签": y_true,
        "白血病类型": meta["leukemia_type"].values,
        "预测概率": y_proba,
        "预测标签": y_pred,
        "阈值": threshold,
        "阈值方法": method,
        "模型": model_name,
    })
    return out
