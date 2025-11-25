# monkey3d/tracking_io.py

from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple
import pandas as pd
import numpy as np

from .roi import compute_box_center


def load_tracking_csv(path: Path) -> pd.DataFrame:
    """
    Load a tracking CSV consisting of bounding boxes OR xc/yc.
    Adds xc/yc columns if not present.
    """
    df = pd.read_csv(path)

    # If center exists, use directly
    if {"xc", "yc"}.issubset(df.columns):
        return df

    # Otherwise compute center from bounding box
    required = {"x1", "y1", "x2", "y2"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV must contain either xc/yc or {required}")

    centers = []
    for _, row in df.iterrows():
        centers.append(compute_box_center(row["x1"], row["y1"], row["x2"], row["y2"]))

    centers = np.array(centers)
    df["xc"] = centers[:, 0]
    df["yc"] = centers[:, 1]

    return df


def truncate_to_shortest(*dfs: pd.DataFrame) -> Tuple[pd.DataFrame, ...]:
    """
    Ensure multiple dataframes have the same number of rows (frames).
    This is useful for syncing front & top views.
    """
    min_len = min(len(df) for df in dfs)
    return tuple(df.iloc[:min_len].reset_index(drop=True) for df in dfs)

