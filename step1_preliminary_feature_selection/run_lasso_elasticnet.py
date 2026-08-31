"""
§2.3 特征初步删减：LASSO + Elastic Net，10 折 CV，输出并集特征。
"""
from __future__ import annotations

import json
import os
import sys

import pandas as pd
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression

PIPELINE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PIPELINE_DIR not in sys.path:
    sys.path.insert(0, PIPELINE_DIR)

from pipeline_common import STEP1_OUT, load_cv_folds, load_train_data

OUT_DIR = STEP1_OUT


def _fold_sets(per_fold_features) -> tuple[set, set]:
    intersection = set(per_fold_features[0])
    union = set()
    for feats in per_fold_features:
        intersection &= set(feats)
        union |= set(feats)
    return intersection, union


def run_method(name: str, fold_indices, X, y) -> tuple[list, set, set]:
    print(f"\n=== {name} 特征选择 ===")
    per_fold = []
    per_fold_removed = []

    for fold_idx, (train_idx, _) in enumerate(fold_indices):
        X_fold = X.iloc[train_idx]
        y_fold = y.iloc[train_idx]

        if name == "LASSO":
            model = LogisticRegression(
                penalty="l1",
                solver="liblinear",
                C=0.1,
                class_weight="balanced",
                random_state=42,
                max_iter=1000,
            )
        else:
            model = LogisticRegression(
                penalty="elasticnet",
                solver="saga",
                l1_ratio=0.5,
                C=0.1,
                class_weight="balanced",
                random_state=42,
                max_iter=1000,
            )

        model.fit(X_fold, y_fold)
        selector = SelectFromModel(model, prefit=True)
        selected = X_fold.columns[selector.get_support()].tolist()
        removed = [c for c in X_fold.columns if c not in selected]
        per_fold.append(selected)
        per_fold_removed.append(removed)
        print(f"  折 {fold_idx + 1}: {len(selected)}/{X.shape[1]} 特征保留")

    intersection, union = _fold_sets(per_fold)
    print(f"  交集: {len(intersection)}, 并集: {len(union)}")
    return per_fold, intersection, union


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    X, y, groups, stratify, features = load_train_data()
    fold_indices = load_cv_folds()

    print(f"原始特征数: {X.shape[1]}")
    print(f"样本数: {len(y)}")

    lasso_per_fold, lasso_inter, lasso_union = run_method("LASSO", fold_indices, X, y)
    enet_per_fold, enet_inter, enet_union = run_method("Elastic Net", fold_indices, X, y)

    final_union = lasso_union | enet_union
    final_inter = lasso_inter & enet_inter
    removed = sorted(set(features) - final_union)

    print(f"\n=== 汇总 ===")
    print(f"LASSO 并集: {len(lasso_union)}, Elastic Net 并集: {len(enet_union)}")
    print(f"最终并集特征数: {len(final_union)}")
    print(f"删除特征 ({len(removed)}): {removed}")

    pd.DataFrame({"feature_name": sorted(final_union), "selection_method": "LASSO_ElasticNet_Union"}).to_csv(
        os.path.join(OUT_DIR, "selected_features.csv"), index=False, encoding="utf-8-sig"
    )
    pd.DataFrame({"feature_name": removed, "reason": "coefficient_zero_all_methods_union"}).to_csv(
        os.path.join(OUT_DIR, "removed_features.csv"), index=False, encoding="utf-8-sig"
    )

    summary = {
        "n_original": len(features),
        "n_lasso_union": len(lasso_union),
        "n_elastic_net_union": len(enet_union),
        "n_final_union": len(final_union),
        "n_final_intersection": len(final_inter),
        "lasso_union": sorted(lasso_union),
        "elastic_net_union": sorted(enet_union),
        "final_selected": sorted(final_union),
        "removed": removed,
    }
    with open(os.path.join(OUT_DIR, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    fold_detail = []
    for i in range(len(fold_indices)):
        fold_detail.append({
            "fold": i + 1,
            "lasso_selected": lasso_per_fold[i],
            "elastic_net_selected": enet_per_fold[i],
        })
    with open(os.path.join(OUT_DIR, "per_fold_selection.json"), "w", encoding="utf-8") as f:
        json.dump(fold_detail, f, ensure_ascii=False, indent=2)

    print(f"\n结果已保存到 {OUT_DIR}")


if __name__ == "__main__":
    main()
