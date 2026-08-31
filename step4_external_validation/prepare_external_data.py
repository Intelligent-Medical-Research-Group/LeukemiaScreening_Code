"""
Step4：准备外部验证数据。

外部集使用本实验 `data/external/` 中的重庆 / 宜昌数据。
特征矩阵与中南 train 处于同一套 RFimput + Scaled 预处理空间。

中南 train 从本实验 `data/` 复制，供外部验证时重训模型使用。
"""
from __future__ import annotations

import json
import os
import shutil

import pandas as pd

PIPELINE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PIPELINE_DIR, "data")
EXTERNAL_DIR = os.path.join(DATA_DIR, "external")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

CENTERS = ("Chongqing", "Yichang")


def load_features() -> list[str]:
    with open(os.path.join(DATA_DIR, "features.txt"), encoding="utf-8") as f:
        return [x.strip() for x in f.read().split(",") if x.strip()]


def materialize_center(center: str, features: list[str]) -> dict:
    x_path = os.path.join(EXTERNAL_DIR, f"{center}_X.csv")
    meta_path = os.path.join(EXTERNAL_DIR, f"{center}_meta.csv")
    if not os.path.isfile(x_path) or not os.path.isfile(meta_path):
        raise FileNotFoundError(f"缺少外部数据: {x_path} / {meta_path}")

    X = pd.read_csv(x_path)
    meta = pd.read_csv(meta_path)
    missing = [c for c in features if c not in X.columns]
    if missing:
        raise RuntimeError(f"{center} 特征缺失: {missing}")
    if len(X) != len(meta):
        raise RuntimeError(f"{center} X/meta 行数不一致: {len(X)} vs {len(meta)}")

    X = X[features]
    y = pd.to_numeric(meta["is_leukemia"], errors="coerce")
    if y.isna().any():
        raise RuntimeError(f"{center} 存在未知标签，不应出现")
    labeled_mask = pd.Series(True, index=y.index)

    X.to_csv(os.path.join(OUT_DIR, f"X_{center}_rfimput_scaled.csv"), index=False)
    y.to_csv(os.path.join(OUT_DIR, f"y_{center}.csv"), index=False)
    meta.to_csv(os.path.join(OUT_DIR, f"meta_{center}.csv"), index=False, encoding="utf-8-sig")
    labeled_mask.astype(int).to_csv(
        os.path.join(OUT_DIR, f"labeled_mask_{center}.csv"), index=False
    )

    return {
        "source": os.path.relpath(x_path, PIPELINE_DIR),
        "n_total": int(len(X)),
        "n_labeled": int(len(X)),
        "n_pos": int((y == 1).sum()),
        "n_neg": int((y == 0).sum()),
        "n_unknown": 0,
    }


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    features = load_features()
    print(f"特征数: {len(features)}")

    for name in ("X_train_rfimput_scaled.csv", "y_train.csv"):
        src = os.path.join(DATA_DIR, name)
        dst = os.path.join(OUT_DIR, name)
        shutil.copy2(src, dst)
    n_train = len(pd.read_csv(os.path.join(OUT_DIR, "y_train.csv")))

    summary = {
        "features": features,
        "n_train_fit": int(n_train),
        "external_source": "data/external (Chongqing/Yichang)",
        "note": "外部验证使用本实验 data/external 中的重庆与宜昌数据",
        "centers": {},
    }

    for center in CENTERS:
        info = materialize_center(center, features)
        summary["centers"][center] = info
        print(
            f"{center}: total={info['n_total']}, labeled={info['n_labeled']}, "
            f"pos={info['n_pos']}, neg={info['n_neg']}"
        )

    with open(os.path.join(OUT_DIR, "external_data_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"输出目录: {OUT_DIR}")


if __name__ == "__main__":
    main()
