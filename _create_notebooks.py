"""生成 step2 各模型 Jupyter notebook（一次性脚本）。"""
import json
import os

NOTEBOOK_DIR = os.path.join(os.path.dirname(__file__), "step2_model_rfe_optuna")
os.makedirs(NOTEBOOK_DIR, exist_ok=True)

MODELS = [
    ("lightgbm", "LightGBM"),
    ("xgboost", "XGBoost"),
    ("catboost", "CatBoost"),
    ("random_forest", "RandomForest"),
    ("svm", "SVM"),
    ("logistic_regression", "LogisticRegression"),
    ("gaussian_nb", "GaussianNB"),
    ("knn", "KNeighborsClassifier"),
    ("qda", "QuadraticDiscriminantAnalysis"),
    ("tabpfn", "TabPFN"),
]


def make_nb(slug: str, model_name: str) -> None:
    run_cell_lines = [
        f'MODEL_NAME = "{model_name}"',
        "N_TRIALS = 100",
        'DEVICE = "cpu"  # 默认 CPU；TabPFN 可改为 "cuda"',
    ]
    if model_name == "TabPFN":
        run_cell_lines.extend([
            "from pipeline_common import run_tabpfn_rfe",
            "best_n, best_pr = run_tabpfn_rfe(selected, device=DEVICE)",
        ])
    else:
        run_cell_lines.append(
            "best_n, best_pr = run_rfe_optuna(MODEL_NAME, selected, n_trials=N_TRIALS, device=DEVICE)"
        )
    run_cell_lines.append('print(f"\\n完成: 最佳特征数={best_n}, PR-AUC={best_pr:.4f}")')

    nb = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    f"# {model_name} — RFE + Optuna\n",
                    "\n",
                    "中南 21+1 特征流水线 — Step 2\n",
                    "\n",
                    "- 预处理：RFimput + Scaled（`../data/`）\n",
                    "- 输入：Step1 并集特征\n",
                    "- Optuna trials: **100**（TabPFN 仅 RFE）\n",
                    "- 训练设备：**CPU**（默认；小数据 GPU 通常更慢）\n",
                ],
            },
            {
                "cell_type": "code",
                "metadata": {},
                "source": [
                    "import os\n",
                    "import sys\n",
                    "\n",
                    "PIPELINE_DIR = os.path.abspath('..')\n",
                    "if not os.path.isfile(os.path.join(PIPELINE_DIR, 'pipeline_common.py')):\n",
                    "    PIPELINE_DIR = os.path.abspath('.')\n",
                    "if PIPELINE_DIR not in sys.path:\n",
                    "    sys.path.insert(0, PIPELINE_DIR)\n",
                    "\n",
                    "import importlib\n",
                    "import pipeline_common\n",
                    "importlib.reload(pipeline_common)\n",
                    "from pipeline_common import run_rfe_optuna, load_selected_features, N_TRIALS_DEFAULT, DEFAULT_DEVICE\n",
                    "\n",
                    "selected = load_selected_features()\n",
                    "print(f'pipeline_common: {pipeline_common.__file__}')\n",
                    "print(f'Step1 并集特征数: {len(selected)}')\n",
                    "print(f'N_TRIALS_DEFAULT: {N_TRIALS_DEFAULT}, DEFAULT_DEVICE: {DEFAULT_DEVICE}')\n",
                    "print(selected)\n",
                ],
                "outputs": [],
                "execution_count": None,
            },
            {
                "cell_type": "code",
                "metadata": {},
                "source": [line + "\n" for line in run_cell_lines],
                "outputs": [],
                "execution_count": None,
            },
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.9.0"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    path = os.path.join(NOTEBOOK_DIR, f"{slug}_rfe_optuna.ipynb")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print("wrote", path)


if __name__ == "__main__":
    for slug, model in MODELS:
        make_nb(slug, model)
