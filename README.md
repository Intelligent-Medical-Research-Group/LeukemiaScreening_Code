# Leukemia screening from routine laboratory tests

TabPFN pipeline for early identification of leukemia versus non-leukemia using routinely ordered laboratory variables. Development used Zhongnan Hospital records; Chongqing and Yichang were held out for external validation.

**Interactive demo (research use only, not a diagnostic device):**  
https://waves-jul-presidential-finances.trycloudflare.com

The locked TabPFN model uses **19 features** (step 2 dropped uric acid and sex). The TabPFN v2 classifier checkpoint is not stored here; set `TABPFN_MODEL_PATH` to a local `tabpfn-v2-classifier.ckpt`, or place that file at `models/tabpfn-v2-classifier.ckpt`.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

GPU is optional. TabPFN, CatBoost, LightGBM, and XGBoost can use CUDA when available.

## Pipeline

Run from the repository root. Intermediate files are written under each step’s `outputs/` directory.

1. **Prepare Zhongnan tables** (only if you have raw files in `raw_data/`): `python prepare_data.py`
2. **Penalized screening:** `python step1_preliminary_feature_selection/run_lasso_elasticnet.py`  
   Or skip and use the committed `selected_features.csv`.
3. **RFE + Optuna** (one notebook per classifier in `step2_model_rfe_optuna/`). TabPFN does not run hyperparameter search. Nested search used 100 Optuna trials per remaining feature count.
4. **Train / test and thresholds:** `python step3_model_training_testing/run_threshold_evaluation.py`
5. **External validation:** `python step4_external_validation/prepare_external_data.py` then `python step4_external_validation/run_external_validation.py`
6. **SHAP:** `python step5_tabpfn_shap/run_step5_tabpfn_shap.py`

Primary operating point: F2-maximizing cutoff **0.4768**.

## Website

The demo above scores 19 laboratory variables with the locked TabPFN model. It is a research illustration only and does not replace morphology, immunophenotyping, or molecular diagnosis.
