"""
生成中南 22 特征（21 共有 + 性别）预处理数据：RFimput + Scaled。

- 插补器与缩放器仅在 train 上 fit，再 transform train 与 test
- 保存 10 折 CV 索引（StratifiedGroupKFold, random_state=42），供后续步骤共用
- 外部验证重庆/宜昌不在此生成，见 data/external/ 与 step4 prepare_external_data.py
"""
from __future__ import annotations

import json
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
RAW_DIR = os.environ.get("LEUKEMIA_RAW_DATA_DIR", os.path.join(SCRIPT_DIR, "raw_data"))

BASELINE_CSV = os.path.join(RAW_DIR, "zhongnan_baseline.csv")
FEATURES_TXT = os.path.join(RAW_DIR, "common_features.txt")
TRAIN_RAW = os.path.join(RAW_DIR, "train_set_clean.csv")
TEST_RAW = os.path.join(RAW_DIR, "test_set_clean.csv")

GENDER_MAP = {"男": 1.0, "女": 0.0}


def load_feature_names() -> list[str]:
    with open(FEATURES_TXT, encoding="utf-8") as f:
        features = [x.strip() for x in f.read().split(",") if x.strip()]
    if "性别" not in features:
        features.append("性别")
    return features


def load_gender_lookup() -> pd.Series:
    train_g = pd.read_csv(TRAIN_RAW, usecols=["住院号", "性别"])
    test_g = pd.read_csv(TEST_RAW, usecols=["住院号", "性别"])
    gender_df = pd.concat([train_g, test_g], ignore_index=True)
    gender_df["住院号"] = gender_df["住院号"].astype(str)
    gender_df["性别"] = gender_df["性别"].astype(str).str.strip().map(GENDER_MAP)
    return gender_df.set_index("住院号")["性别"]


def main() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    features = load_feature_names()
    gender_lookup = load_gender_lookup()

    df = pd.read_csv(BASELINE_CSV)
    df["patient_id"] = df["patient_id"].astype(str)
    df["性别"] = df["patient_id"].map(gender_lookup)
    missing_gender = df["性别"].isna().sum()
    if missing_gender:
        print(f"警告: {missing_gender} 行性别无法匹配，将交由 RF 插补处理")

    train_df = df[df["split"] == "train"].copy()
    test_df = df[df["split"] == "test"].copy()

    meta_cols = ["patient_id", "is_leukemia", "leukemia_type", "label_raw", "split"]
    for split_name, split_df in [("train", train_df), ("test", test_df)]:
        X_raw = split_df[features].apply(pd.to_numeric, errors="coerce")
        if split_name == "train":
            imputer = IterativeImputer(random_state=42, max_iter=20, tol=1e-2)
            X_imputed = pd.DataFrame(
                imputer.fit_transform(X_raw),
                columns=features,
                index=X_raw.index,
            )
            scaler = StandardScaler()
            X_scaled = pd.DataFrame(
                scaler.fit_transform(X_imputed),
                columns=features,
                index=X_imputed.index,
            )
        else:
            X_imputed = pd.DataFrame(
                imputer.transform(X_raw),
                columns=features,
                index=X_raw.index,
            )
            X_scaled = pd.DataFrame(
                scaler.transform(X_imputed),
                columns=features,
                index=X_imputed.index,
            )

        prefix = split_name
        X_scaled.to_csv(os.path.join(DATA_DIR, f"X_{prefix}_rfimput_scaled.csv"), index=False)
        split_df[["is_leukemia"]].to_csv(
            os.path.join(DATA_DIR, f"y_{prefix}.csv"), index=False
        )
        split_df[["patient_id"]].rename(columns={"patient_id": "patient_id"}).to_csv(
            os.path.join(DATA_DIR, f"groups_{prefix}.csv"), index=False
        )
        split_df[["leukemia_type"]].to_csv(
            os.path.join(DATA_DIR, f"stratify_{prefix}.csv"), index=False
        )
        split_df[meta_cols].to_csv(
            os.path.join(DATA_DIR, f"meta_{prefix}.csv"), index=False, encoding="utf-8-sig"
        )

    y_train = train_df["is_leukemia"].astype(int)
    groups_train = train_df["patient_id"]
    stratify_train = train_df["leukemia_type"].astype(str)
    stratify_cv = stratify_train.str.strip() + "|" + y_train.astype(str)

    cv = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42)
    fold_indices = list(cv.split(train_df[features], stratify_cv, groups_train))
    with open(os.path.join(DATA_DIR, "cv_folds.pkl"), "wb") as f:
        pickle.dump(fold_indices, f)

    with open(os.path.join(DATA_DIR, "features.txt"), "w", encoding="utf-8") as f:
        f.write(", ".join(features))

    metadata = {
        "preprocessing": {"imputer": "RFimputer (IterativeImputer)", "scaler": "StandardScaler"},
        "n_features": len(features),
        "features": features,
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "n_pos_train": int((y_train == 1).sum()),
        "n_neg_train": int((y_train == 0).sum()),
        "scale_pos_weight": float((y_train == 0).sum() / max((y_train == 1).sum(), 1)),
        "cv": {
            "method": "StratifiedGroupKFold",
            "n_splits": 10,
            "shuffle": True,
            "random_state": 42,
            "groups": "patient_id",
            "stratify": "leukemia_type | is_leukemia",
        },
        "gender_encoding": {"男": 1.0, "女": 0.0, "不详/缺失": "NaN → RF impute"},
        "note": "建模步骤仅使用 train  split；test 已预处理供最终评估",
    }
    with open(os.path.join(DATA_DIR, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("预处理完成")
    print(f"  特征数: {len(features)} ({features[-1]} 已加入)")
    print(f"  train: {len(train_df)}, test: {len(test_df)}")
    print(f"  输出目录: {DATA_DIR}")


if __name__ == "__main__":
    main()
