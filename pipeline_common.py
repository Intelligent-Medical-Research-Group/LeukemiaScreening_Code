"""
中南 21feat pipeline 共享工具：数据加载、Optuna 目标函数、RFE 主循环。
"""
from __future__ import annotations

import gc
import json
import os
import pickle
import warnings
from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight

warnings.filterwarnings("ignore")

PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PIPELINE_DIR, "data")
STEP1_OUT = os.path.join(PIPELINE_DIR, "step1_preliminary_feature_selection", "outputs")
STEP2_OUT = os.path.join(PIPELINE_DIR, "step2_model_rfe_optuna", "outputs")
TABPFN_MODEL_PATH = os.environ.get(
    "TABPFN_MODEL_PATH",
    os.path.join(PIPELINE_DIR, "models", "tabpfn-v2-classifier.ckpt"),
)

N_TRIALS_DEFAULT = 100  # 每个特征数 100 次 Optuna TPE（论文设定；TabPFN 不做超参搜索）
N_JOBS = 4
RANDOM_STATE = 42
DEFAULT_DEVICE = "cpu"
GPU_MODELS = {"CatBoost", "LightGBM", "XGBoost", "TabPFN"}
_LGB_CUDA_AVAILABLE: Optional[bool] = None


def resolve_device(device: str = DEFAULT_DEVICE) -> str:
    """默认 cpu；仅当显式传入 cuda 时才用 GPU（小数据集上 GPU 往往更慢）。"""
    if device not in ("auto", "cuda", "cpu"):
        raise ValueError(f"device 必须是 auto/cuda/cpu，收到: {device}")
    if device in ("cpu", "auto"):
        return "cpu"
    return "cuda"


def _apply_device_to_params(model_name: str, params: dict, device: str) -> dict:
    params = dict(params)
    if device != "cuda" or model_name not in GPU_MODELS:
        if model_name == "CatBoost":
            params.setdefault("task_type", "CPU")
        return params
    if model_name == "CatBoost":
        params["task_type"] = "GPU"
    elif model_name == "XGBoost":
        params["device"] = "cuda"
        params["tree_method"] = "hist"
    return params


def _lgb_supports_cuda() -> bool:
    """检测当前 LightGBM 是否为 CUDA 编译（device=cuda）；否则回退 OpenCL（device=gpu）。"""
    global _LGB_CUDA_AVAILABLE
    if _LGB_CUDA_AVAILABLE is not None:
        return _LGB_CUDA_AVAILABLE
    try:
        import lightgbm as lgb

        x = np.random.rand(16, 2)
        y = np.random.randint(0, 2, 16)
        lgb.train(
            {"objective": "binary", "device": "cuda", "verbose": -1, "num_leaves": 4},
            lgb.Dataset(x, label=y),
            num_boost_round=1,
        )
        _LGB_CUDA_AVAILABLE = True
    except Exception:
        _LGB_CUDA_AVAILABLE = False
    return _LGB_CUDA_AVAILABLE


def _apply_device_to_lgb_params(lgb_params: dict, device: str) -> dict:
    lgb_params = dict(lgb_params)
    if device == "cuda":
        gpu_device_id = int(os.environ.get("LGB_GPU_DEVICE_ID", "1"))
        lgb_params["device"] = "cuda" if _lgb_supports_cuda() else "gpu"
        lgb_params["gpu_platform_id"] = 0
        lgb_params["gpu_device_id"] = gpu_device_id
    return lgb_params


def load_train_data() -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series, List[str]]:
    X = pd.read_csv(os.path.join(DATA_DIR, "X_train_rfimput_scaled.csv"))
    y = pd.read_csv(os.path.join(DATA_DIR, "y_train.csv")).squeeze().astype(int)
    groups = pd.read_csv(os.path.join(DATA_DIR, "groups_train.csv")).squeeze().astype(str)
    stratify = pd.read_csv(os.path.join(DATA_DIR, "stratify_train.csv")).squeeze().astype(str)
    with open(os.path.join(DATA_DIR, "features.txt"), encoding="utf-8") as f:
        features = [x.strip() for x in f.read().split(",") if x.strip()]
    return X, y, groups, stratify, features


def load_cv_folds() -> List[Tuple[np.ndarray, np.ndarray]]:
    with open(os.path.join(DATA_DIR, "cv_folds.pkl"), "rb") as f:
        return pickle.load(f)


def load_selected_features() -> List[str]:
    path = os.path.join(STEP1_OUT, "selected_features.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"未找到 {path}，请先运行 step1_preliminary_feature_selection"
        )
    df = pd.read_csv(path)
    return df["feature_name"].tolist()


def scale_pos_weight(y: pd.Series) -> float:
    n_pos = max(int((pd.Series(y) == 1).sum()), 1)
    return float((pd.Series(y) == 0).sum()) / n_pos


def cv_stratify_labels(leukemia_type: pd.Series, y: pd.Series) -> pd.Series:
    """StratifiedGroupKFold 分层标签：详细白血病类别 × 二分类结局。"""
    return leukemia_type.astype(str).str.strip() + "|" + pd.Series(y).astype(int).astype(str)


def build_tabpfn_classifier(
    device: str,
    *,
    fit_mode: Optional[str] = None,
    ignore_pretraining_limits: bool = False,
):
    """TabPFN v2：论文设定为默认 8 estimators、温度 0.9、softmax 后平均、概率平衡。"""
    from tabpfn import TabPFNClassifier

    kwargs = {
        "device": device,
        "model_path": TABPFN_MODEL_PATH,
        "balance_probabilities": True,
        "random_state": RANDOM_STATE,
        "n_estimators": 8,
        "softmax_temperature": 0.9,
        "average_before_softmax": False,
        "ignore_pretraining_limits": ignore_pretraining_limits,
    }
    if fit_mode is not None:
        kwargs["fit_mode"] = fit_mode
    return TabPFNClassifier(**kwargs)


def _fit_balanced_if_supported(clf, X, y):
    """折级 balanced sample_weight；估计器不支持时退回普通 fit。"""
    sw = compute_sample_weight("balanced", y)
    try:
        clf.fit(X, y, sample_weight=sw)
    except (TypeError, ValueError):
        clf.fit(X, y)
    return clf


def _lgb_boost_params(params: dict, device: str) -> dict:
    """LightGBM：论文为 balanced class weights（is_unbalance），不用 scale_pos_weight。"""
    grid = get_model_configs(1.0)["LightGBM"]["param_grid"]
    return _apply_device_to_lgb_params(
        {
            "objective": "binary",
            "metric": "binary_logloss",
            "boosting_type": "gbdt",
            "verbose": -1,
            "random_state": RANDOM_STATE,
            **{k: params[k] for k in params if k in grid},
            "is_unbalance": True,
        },
        device,
    )


def _xgb_params_for_y(params: dict, y) -> dict:
    """XGBoost：每折（或当前训练集）单独计算 scale_pos_weight。"""
    out = dict(params)
    out["scale_pos_weight"] = scale_pos_weight(pd.Series(y))
    return out


def setup_logging(model_name: str, output_dir: str):
    model_dir = os.path.join(output_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)
    log_path = os.path.join(
        model_dir,
        f"{model_name.lower()}_feature_selection_log_{datetime.now():%Y%m%d_%H%M%S}.log",
    )
    log_file = open(log_path, "w", encoding="utf-8")

    def log_print(*args, **kwargs):
        print(*args, **kwargs)
        print(*args, **kwargs, file=log_file)
        log_file.flush()

    return log_print, log_file, log_path


def calculate_all_metrics(y_true, y_pred, y_pred_proba) -> Dict[str, float]:
    try:
        accuracy = accuracy_score(y_true, y_pred)
        f1_micro = f1_score(y_true, y_pred, average="micro")
        f1_macro = f1_score(y_true, y_pred, average="macro")
        auc = roc_auc_score(y_true, y_pred_proba)
        pr_auc = average_precision_score(y_true, y_pred_proba)
        cm = confusion_matrix(y_true, y_pred)
        if cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
        else:
            tn = fp = fn = tp = 0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        return {
            "accuracy": accuracy,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "ppv": ppv,
            "npv": npv,
            "f1_micro": f1_micro,
            "f1_macro": f1_macro,
            "auc": auc,
            "pr_auc": pr_auc,
        }
    except Exception:
        return {
            "accuracy": 0.0,
            "sensitivity": 0.0,
            "specificity": 0.0,
            "ppv": 0.0,
            "npv": 0.0,
            "f1_micro": 0.0,
            "f1_macro": 0.0,
            "auc": 0.0,
            "pr_auc": 0.0,
        }


def summarize_fold_metrics(fold_dict: Dict[str, List[float]]) -> Dict[str, float]:
    """对折级指标计算 mean/median/std 与 95% CI（2.5–97.5 分位）。"""
    out: Dict[str, float] = {}
    for metric, vals in fold_dict.items():
        arr = np.asarray(vals, dtype=float)
        arr = arr[~np.isnan(arr)]
        if len(arr) == 0:
            for suffix in ("median", "mean", "std", "ci_lower", "ci_upper"):
                out[f"{metric}_{suffix}"] = float("nan")
            continue
        out[f"{metric}_median"] = float(np.median(arr))
        out[f"{metric}_mean"] = float(np.mean(arr))
        out[f"{metric}_std"] = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
        out[f"{metric}_ci_lower"] = float(np.percentile(arr, 2.5))
        out[f"{metric}_ci_upper"] = float(np.percentile(arr, 97.5))
    return out


def get_model_configs(spw: float) -> Dict:
    return {
        "CatBoost": {
            "class": __import__("catboost").CatBoostClassifier,
            "param_grid": {
                "iterations": (100, 500),
                "learning_rate": (0.01, 0.3),
                "depth": (4, 10),
                "l2_leaf_reg": (1, 10),
                "border_count": (32, 255),
                "bagging_temperature": (0, 1),
                "random_strength": (0, 1),
            },
            "fixed_params": {
                "auto_class_weights": "Balanced",
                "random_state": RANDOM_STATE,
                "verbose": False,
            },
        },
        "LightGBM": {
            "class": None,
            "param_grid": {
                "num_leaves": (20, 100),
                "learning_rate": (0.01, 0.3),
                "feature_fraction": (0.2, 1.0),
                "bagging_fraction": (0.2, 1.0),
                "bagging_freq": (1, 10),
                "min_child_samples": (10, 50),
                "reg_alpha": (0.0, 1.0),
                "reg_lambda": (0.0, 1.0),
            },
            "fixed_params": {
                "objective": "binary",
                "metric": "binary_logloss",
                "boosting_type": "gbdt",
                "random_state": RANDOM_STATE,
                "verbose": -1,
            },
        },
        "XGBoost": {
            "class": None,  # lazy import
            "param_grid": {
                "n_estimators": (20, 200),
                "learning_rate": (0.01, 0.3),
                "max_depth": (5, 10),
                "subsample": (0.2, 1.0),
                "colsample_bytree": (0.2, 1.0),
            },
            "fixed_params": {
                "objective": "binary:logistic",
                "booster": "gbtree",
                "random_state": RANDOM_STATE,
                "eval_metric": "logloss",
                "verbosity": 0,
            },
        },
        "RandomForest": {
            "class": RandomForestClassifier,
            "param_grid": {
                "n_estimators": (100, 300),
                "max_depth": (5, 20),
                "min_samples_split": (2, 10),
                "min_samples_leaf": (1, 5),
                "max_features": ["sqrt", "log2"],
                "bootstrap": [True, False],
            },
            "fixed_params": {
                "class_weight": "balanced",
                "random_state": RANDOM_STATE,
                "n_jobs": N_JOBS,
            },
        },
        "SVM": {
            "class": SVC,
            "param_grid": {
                "C": (0.1, 10.0),
                "gamma": ["scale", "auto"],
                "kernel": ["rbf", "linear", "poly"],
            },
            "fixed_params": {
                "class_weight": "balanced",
                "probability": True,
                "random_state": RANDOM_STATE,
            },
        },
        "GaussianNB": {
            "class": GaussianNB,
            "param_grid": {"var_smoothing": (1e-9, 1e-3)},
            "fixed_params": {},
        },
        "KNeighborsClassifier": {
            "class": KNeighborsClassifier,
            "param_grid": {
                "n_neighbors": (3, 20),
                "weights": ["uniform", "distance"],
                "metric": ["euclidean", "manhattan", "minkowski"],
            },
            "fixed_params": {},
        },
        "QuadraticDiscriminantAnalysis": {
            "class": QuadraticDiscriminantAnalysis,
            "param_grid": {
                "reg_param": (0.001, 1.0),
                "store_covariance": [True, False],
                "tol": (1e-4, 1e-2),
            },
            "fixed_params": {},
        },
        "LogisticRegression": {
            "class": LogisticRegression,
            "param_grid": {
                "C": (0.001, 100.0),
                "penalty": ["l1", "l2"],
                "max_iter": (500, 2000),
            },
            "fixed_params": {
                "solver": "liblinear",
                "class_weight": "balanced",
                "random_state": RANDOM_STATE,
            },
        },
    }


def _fit_predict_proba(
    model_name: str,
    params: dict,
    train_x: pd.DataFrame,
    train_y: pd.Series,
    valid_x: pd.DataFrame,
    valid_y: pd.Series,
    spw: float,
    device: str = "cpu",
) -> np.ndarray:
    import lightgbm as lgb

    params = _apply_device_to_params(model_name, params, device)

    if model_name == "LightGBM":
        lgb_params = _lgb_boost_params(params, device)
        train_data = lgb.Dataset(train_x, label=train_y)
        valid_data = lgb.Dataset(valid_x, label=valid_y, reference=train_data)
        callbacks = [lgb.early_stopping(stopping_rounds=50, verbose=False)]
        model = lgb.train(
            lgb_params, train_data, valid_sets=[valid_data],
            num_boost_round=200, callbacks=callbacks,
        )
        return model.predict(valid_x, num_iteration=model.best_iteration)

    if model_name == "KNeighborsClassifier":
        base = KNeighborsClassifier(**params)
        clf = CalibratedClassifierCV(base, cv=5, method="isotonic")
        _fit_balanced_if_supported(clf, train_x, train_y)
        return clf.predict_proba(valid_x)[:, 1]

    if model_name == "GaussianNB":
        clf = GaussianNB(**params)
        _fit_balanced_if_supported(clf, train_x, train_y)
        return clf.predict_proba(valid_x)[:, 1]

    if model_name == "QuadraticDiscriminantAnalysis":
        clf = QuadraticDiscriminantAnalysis(**params)
        _fit_balanced_if_supported(clf, train_x, train_y)
        return clf.predict_proba(valid_x)[:, 1]

    if model_name == "XGBoost":
        from xgboost import XGBClassifier
        clf = XGBClassifier(**_xgb_params_for_y(params, train_y))
        clf.fit(train_x, train_y)
        return clf.predict_proba(valid_x)[:, 1]

    config = get_model_configs(spw)[model_name]
    clf = config["class"](**params)
    clf.fit(train_x, train_y)
    return clf.predict_proba(valid_x)[:, 1]


def create_objective_function(
    model_name: str,
    data_x: pd.DataFrame,
    data_y: pd.Series,
    groups: pd.Series,
    spw: float,
    fold_indices: Optional[List] = None,
    device: str = "cpu",
) -> Callable:
    import optuna

    config = get_model_configs(spw)[model_name]
    folds = fold_indices or load_cv_folds()

    def objective(trial):
        params = dict(config["fixed_params"])
        for param_name, param_range in config["param_grid"].items():
            if isinstance(param_range, tuple):
                if isinstance(param_range[0], int):
                    params[param_name] = trial.suggest_int(param_name, param_range[0], param_range[1])
                elif len(param_range) == 3:
                    params[param_name] = trial.suggest_float(
                        param_name, param_range[0], param_range[1], step=param_range[2]
                    )
                else:
                    params[param_name] = trial.suggest_float(param_name, param_range[0], param_range[1])
            elif isinstance(param_range, list):
                params[param_name] = trial.suggest_categorical(param_name, param_range)

        if model_name == "KNeighborsClassifier" and params.get("metric") == "minkowski":
            params["p"] = trial.suggest_int("p", 1, 3)

        total_preds = np.zeros(len(data_y))
        for train_idx, valid_idx in folds:
            train_x = data_x.iloc[train_idx]
            train_y = data_y.iloc[train_idx]
            valid_x = data_x.iloc[valid_idx]
            valid_y = data_y.iloc[valid_idx]
            try:
                total_preds[valid_idx] = _fit_predict_proba(
                    model_name, params, train_x, train_y, valid_x, valid_y, spw, device
                )
            except Exception as exc:
                print(f"训练失败: {exc}")
                return 0.0
            gc.collect()

        try:
            return average_precision_score(data_y, total_preds)
        except Exception:
            return 0.0

    return objective


def compute_feature_importance(
    model_name: str,
    data_x: pd.DataFrame,
    y: pd.Series,
    features_list: List[str],
    best_params: dict,
    spw: float,
    device: str = "cpu",
) -> Dict[str, float]:
    """RFE 删特征：树模型用自带 importance；其余用 PR-AUC permutation importance。"""
    import lightgbm as lgb

    params = dict(get_model_configs(spw)[model_name]["fixed_params"])
    params.update(best_params)
    params = _apply_device_to_params(model_name, params, device)

    if model_name == "LightGBM":
        lgb_params = _lgb_boost_params(params, device)
        model = lgb.train(lgb_params, lgb.Dataset(data_x, label=y), num_boost_round=200)
        importance = model.feature_importance(importance_type="gain")
        return dict(zip(features_list, importance))

    if model_name == "XGBoost":
        from xgboost import XGBClassifier

        clf = XGBClassifier(**_xgb_params_for_y(params, y))
        clf.fit(data_x, y)
        return dict(zip(features_list, clf.feature_importances_))

    config = get_model_configs(spw)[model_name]
    if model_name == "KNeighborsClassifier" and params.get("metric") == "minkowski" and "p" not in params:
        params["p"] = 2
    if model_name == "KNeighborsClassifier":
        clf = CalibratedClassifierCV(config["class"](**params), cv=5, method="isotonic")
    else:
        clf = config["class"](**params)
    if model_name in ("GaussianNB", "QuadraticDiscriminantAnalysis", "KNeighborsClassifier"):
        _fit_balanced_if_supported(clf, data_x, y)
    else:
        clf.fit(data_x, y)

    if hasattr(clf, "feature_importances_"):
        return dict(zip(features_list, clf.feature_importances_))

    perm = permutation_importance(
        clf, data_x, y, n_repeats=5, random_state=RANDOM_STATE, scoring="average_precision"
    )
    return dict(zip(features_list, perm.importances_mean))


def run_tabpfn_rfe(
    selected_features: List[str],
    output_dir: str = STEP2_OUT,
    device: str = DEFAULT_DEVICE,
) -> Tuple[int, float]:
    """TabPFN 无 Optuna 超参搜索，按特征数递减 + 排列重要性。"""
    device = resolve_device(device)

    X, y, groups, _, _ = load_train_data()
    X = X[selected_features]
    fold_indices = load_cv_folds()

    log_print, log_file, log_path = setup_logging("TabPFN", output_dir)
    log_print(f"TabPFN RFE 开始，初始特征数: {len(selected_features)}")
    log_print(f"device={device}, model={TABPFN_MODEL_PATH}")

    features_list = list(selected_features)
    results_rows = []
    fold_rows = []

    for n_features in range(len(selected_features), 0, -1):
        log_print(f"\n--- 当前特征数: {n_features} ---")
        data_x = X[features_list]
        total_preds = np.zeros(len(y))
        fold_metrics = []

        for fold_i, (train_idx, valid_idx) in enumerate(fold_indices):
            train_x = data_x.iloc[train_idx].values
            train_y = y.iloc[train_idx].values
            valid_x = data_x.iloc[valid_idx].values
            valid_y = y.iloc[valid_idx]
            clf = build_tabpfn_classifier(device)
            clf.fit(train_x, train_y)
            proba = clf.predict_proba(valid_x)[:, 1]
            total_preds[valid_idx] = proba
            pred = (proba >= 0.5).astype(int)
            m = calculate_all_metrics(valid_y, pred, proba)
            fold_metrics.append(m)
            fold_rows.append({
                "model": "TabPFN",
                "n_features": n_features,
                "fold": fold_i + 1,
                "auc": m["auc"],
                "pr_auc": m["pr_auc"],
                "sensitivity": m["sensitivity"],
                "specificity": m["specificity"],
                "threshold": 0.5,
            })
            del clf
            gc.collect()

        pr_auc = average_precision_score(y, total_preds)
        auc = roc_auc_score(y, total_preds)
        fold_dict = {k: [m[k] for m in fold_metrics] for k in fold_metrics[0]}
        summary = summarize_fold_metrics(fold_dict)
        avg_sens = float(np.mean(fold_dict["sensitivity"]))
        avg_spec = float(np.mean(fold_dict["specificity"]))
        log_print(f"PR-AUC={pr_auc:.4f}, AUC={auc:.4f}, Sens={avg_sens:.4f}, Spec={avg_spec:.4f}")

        row = {
            "特征数量": n_features,
            "PR-AUC": pr_auc,
            "AUC": auc,
            "Sensitivity": avg_sens,
            "Specificity": avg_spec,
            "使用特征": ", ".join(features_list),
        }
        for metric in ("auc", "pr_auc", "sensitivity", "specificity"):
            for suffix in ("median", "mean", "std", "ci_lower", "ci_upper"):
                row[f"{metric}_{suffix}"] = summary.get(f"{metric}_{suffix}", np.nan)
        results_rows.append(row)

        if n_features == 1:
            break

        # 全数据 permutation importance（n_repeats=3，PR-AUC）
        clf = build_tabpfn_classifier(device)
        clf.fit(data_x.values, y.values)
        perm = permutation_importance(
            clf, data_x.values, y.values, n_repeats=3,
            random_state=RANDOM_STATE, scoring="average_precision",
        )
        imp = dict(zip(features_list, perm.importances_mean))
        worst = min(imp, key=imp.get)
        features_list.remove(worst)
        log_print(f"移除特征: {worst}")
        del clf
        gc.collect()

    model_dir = os.path.join(output_dir, "TabPFN")
    os.makedirs(model_dir, exist_ok=True)
    results_df = pd.DataFrame(results_rows)
    csv_path = os.path.join(model_dir, "tabpfn_feature_selection_results.csv")
    results_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    fold_csv = os.path.join(model_dir, "tabpfn_fold_metrics.csv")
    pd.DataFrame(fold_rows).to_csv(fold_csv, index=False, encoding="utf-8-sig")

    best_idx = results_df["PR-AUC"].idxmax()
    best_row = results_df.loc[best_idx]
    with open(os.path.join(model_dir, "best_summary.json"), "w", encoding="utf-8") as f:
        json.dump({
            "model": "TabPFN",
            "best_feature_count": int(best_row["特征数量"]),
            "best_pr_auc": float(best_row["PR-AUC"]),
            "best_auc": float(best_row["AUC"]),
            "best_features": [x.strip() for x in str(best_row["使用特征"]).split(",") if x.strip()],
            "threshold_for_sens_spec": 0.5,
            "fold_metrics_csv": os.path.basename(fold_csv),
        }, f, ensure_ascii=False, indent=2)

    log_print(f"\n最佳特征数: {best_row['特征数量']}, PR-AUC: {best_row['PR-AUC']:.4f}")
    log_print(f"结果已保存: {csv_path}")
    log_print(f"折级指标: {fold_csv}")
    log_file.close()
    return int(best_row["特征数量"]), float(best_row["PR-AUC"])


def run_rfe_optuna(
    model_name: str,
    selected_features: Optional[List[str]] = None,
    n_trials: int = N_TRIALS_DEFAULT,
    output_dir: str = STEP2_OUT,
    n_jobs: int = N_JOBS,
    device: str = DEFAULT_DEVICE,
) -> Tuple[int, float]:
    import optuna

    device = resolve_device(device)

    if model_name == "TabPFN":
        return run_tabpfn_rfe(selected_features or load_selected_features(), output_dir, device=device)

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    X, y, groups, _, all_features = load_train_data()
    selected = selected_features or load_selected_features()
    X = X[selected]
    spw = scale_pos_weight(y)
    fold_indices = load_cv_folds()

    if model_name not in get_model_configs(spw):
        raise ValueError(f"未知模型: {model_name}")

    log_print, log_file, log_path = setup_logging(model_name, output_dir)
    log_print(f"=== {model_name} RFE + Optuna 开始 ===")
    log_print(f"初始特征数: {len(selected)}, Optuna trials: {n_trials}, device: {device}")
    if device == "cuda" and model_name in GPU_MODELS:
        log_print(f"GPU 加速: 已启用 ({device})")
        if model_name == "LightGBM":
            lgb_dev = _apply_device_to_lgb_params({}, device)
            log_print(
                f"LightGBM 后端: {lgb_dev.get('device')}, "
                f"gpu_device_id={lgb_dev.get('gpu_device_id')}"
            )
    else:
        log_print(f"训练设备: CPU")

    features_list = list(selected)
    best_trial_list = []
    best_params_list = []
    feature_list_list = []
    all_metrics_list = []
    fold_metric_lists: List[Dict[str, List[float]]] = []

    for n_features in range(len(selected), 0, -1):
        log_print(f"\n--- 当前特征数: {n_features} ---")
        data_x = X[features_list]

        objective = create_objective_function(
            model_name, data_x, y, groups, spw, fold_indices, device
        )
        study = optuna.create_study(
            direction="maximize",
            study_name=model_name.lower(),
            sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
        )
        log_print(f"开始 Optuna 优化 ({n_trials} trials)...")
        study.optimize(objective, n_trials=n_trials, n_jobs=1)

        best_trial_list.append(study.best_trial)
        best_params_list.append(study.best_params)
        feature_list_list.append(features_list.copy())
        log_print(f"最佳 PR-AUC: {study.best_trial.value:.4f}")
        log_print(f"最佳参数: {study.best_params}")

        # 用最佳参数重算各折指标
        params = dict(get_model_configs(spw)[model_name]["fixed_params"])
        params.update(study.best_params)
        params = _apply_device_to_params(model_name, params, device)
        if model_name == "KNeighborsClassifier" and params.get("metric") == "minkowski" and "p" not in params:
            params["p"] = 2

        fold_metrics = []
        for train_idx, valid_idx in fold_indices:
            train_x = data_x.iloc[train_idx]
            train_y = y.iloc[train_idx]
            valid_x = data_x.iloc[valid_idx]
            valid_y = y.iloc[valid_idx]
            try:
                proba = _fit_predict_proba(model_name, params, train_x, train_y, valid_x, valid_y, spw, device)
                pred = (proba >= 0.5).astype(int)
                fold_metrics.append(calculate_all_metrics(valid_y, pred, proba))
            except Exception as exc:
                log_print(f"指标计算失败: {exc}")
            gc.collect()

        avg_metrics = {k: float(np.mean([m[k] for m in fold_metrics])) for k in fold_metrics[0]} if fold_metrics else {}
        fold_metric_lists.append({k: [m[k] for m in fold_metrics] for k in fold_metrics[0]} if fold_metrics else {})
        all_metrics_list.append(avg_metrics)
        log_print(
            f"10折平均 AUC={avg_metrics.get('auc', 0):.4f}, "
            f"Sens={avg_metrics.get('sensitivity', 0):.4f}, "
            f"Spec={avg_metrics.get('specificity', 0):.4f}"
        )

        if n_features == 1:
            del study
            break

        try:
            imp = compute_feature_importance(
                model_name, data_x, y, features_list, study.best_params, spw, device
            )
            worst = min(imp, key=imp.get)
            features_list.remove(worst)
            log_print(f"移除特征: {worst} (importance={imp[worst]:.4f})")
        except Exception as exc:
            removed = features_list.pop()
            log_print(f"重要性计算失败 ({exc})，移除: {removed}")

        del study
        gc.collect()

    # 保存结果（含折级指标与 95% CI）
    model_dir = os.path.join(output_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)
    pr_aucs = [t.value for t in best_trial_list if t.value is not None]
    best_idx = int(np.argmax(pr_aucs)) if pr_aucs else 0
    best_feature_count = len(selected) - best_idx

    rows = []
    fold_rows = []
    for i, n_feat in enumerate(range(len(selected), len(selected) - len(best_trial_list), -1)):
        fold_dict = fold_metric_lists[i] if i < len(fold_metric_lists) else {}
        summary = summarize_fold_metrics(fold_dict) if fold_dict else {}
        row = {
            "特征数量": n_feat,
            "PR-AUC": best_trial_list[i].value,
            "AUC": all_metrics_list[i].get("auc", np.nan) if i < len(all_metrics_list) else np.nan,
            "Sensitivity": all_metrics_list[i].get("sensitivity", np.nan) if i < len(all_metrics_list) else np.nan,
            "Specificity": all_metrics_list[i].get("specificity", np.nan) if i < len(all_metrics_list) else np.nan,
            "最佳参数": str(best_params_list[i]),
            "使用特征": ", ".join(feature_list_list[i]),
        }
        for metric in ("auc", "pr_auc", "sensitivity", "specificity"):
            for suffix in ("median", "mean", "std", "ci_lower", "ci_upper"):
                key = f"{metric}_{suffix}"
                row[key] = summary.get(key, np.nan)
        rows.append(row)
        for fold_i in range(10):
            fold_rows.append({
                "model": model_name,
                "n_features": n_feat,
                "fold": fold_i + 1,
                "auc": fold_dict.get("auc", [np.nan] * 10)[fold_i] if fold_dict.get("auc") and fold_i < len(fold_dict["auc"]) else np.nan,
                "pr_auc": fold_dict.get("pr_auc", [np.nan] * 10)[fold_i] if fold_dict.get("pr_auc") and fold_i < len(fold_dict["pr_auc"]) else np.nan,
                "sensitivity": fold_dict.get("sensitivity", [np.nan] * 10)[fold_i] if fold_dict.get("sensitivity") and fold_i < len(fold_dict["sensitivity"]) else np.nan,
                "specificity": fold_dict.get("specificity", [np.nan] * 10)[fold_i] if fold_dict.get("specificity") and fold_i < len(fold_dict["specificity"]) else np.nan,
                "threshold": 0.5,
            })

    results_df = pd.DataFrame(rows)
    csv_path = os.path.join(model_dir, f"{model_name.lower()}_feature_selection_results.csv")
    results_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    fold_csv = os.path.join(model_dir, f"{model_name.lower()}_fold_metrics.csv")
    pd.DataFrame(fold_rows).to_csv(fold_csv, index=False, encoding="utf-8-sig")

    with open(os.path.join(model_dir, "best_summary.json"), "w", encoding="utf-8") as f:
        json.dump({
            "model": model_name,
            "best_feature_count": best_feature_count,
            "best_pr_auc": float(pr_aucs[best_idx]) if pr_aucs else None,
            "best_params": best_params_list[best_idx] if best_params_list else {},
            "best_features": feature_list_list[best_idx] if feature_list_list else [],
            "threshold_for_sens_spec": 0.5,
            "fold_metrics_csv": os.path.basename(fold_csv),
        }, f, ensure_ascii=False, indent=2)

    log_print(f"\n最佳特征数: {best_feature_count}, PR-AUC: {pr_aucs[best_idx]:.4f}")
    log_print(f"结果: {csv_path}")
    log_print(f"折级指标: {fold_csv}")
    log_file.close()
    return best_feature_count, float(pr_aucs[best_idx])
