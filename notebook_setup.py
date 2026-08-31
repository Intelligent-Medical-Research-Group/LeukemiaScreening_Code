"""Jupyter notebook 公共初始化：定位 pipeline 目录并 reload pipeline_common。"""
from __future__ import annotations

import importlib
import os
import sys


def setup_pipeline():
    here = os.path.dirname(os.path.abspath(__file__))
    cwd = os.getcwd()
    candidates = [
        here,
        cwd,
        os.path.abspath(os.path.join(cwd, "..")),
        os.path.join(cwd, "LeukemiaScreening_Code"),
    ]
    pipeline_dir = None
    for p in candidates:
        p = os.path.abspath(p)
        if os.path.isfile(os.path.join(p, "pipeline_common.py")):
            pipeline_dir = p
            break
    if pipeline_dir is None:
        raise FileNotFoundError(
            "找不到 pipeline_common.py，请确认 notebook 工作目录在 "
            "LeukemiaScreening_Code/ 或其 step2 子目录下。"
        )
    if pipeline_dir not in sys.path:
        sys.path.insert(0, pipeline_dir)
    import pipeline_common

    importlib.reload(pipeline_common)
    return pipeline_dir, pipeline_common
