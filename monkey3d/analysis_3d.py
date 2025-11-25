# monkey3d/analysis_3d.py

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Tuple

from .tracking_io import truncate_to_shortest


def combine_front_top(
    df_front: pd.DataFrame,
    df_top: pd.DataFrame,
) -> pd.DataFrame:
    """
    Combine front-view and top-view 2D coordinates into a pseudo-3D track.
    Assumes:
        - top gives (x, y)
        - front gives z (mapped from y pixel coordinate)
    Note: This is a placeholder. Replace with your calibrated model.

    """
    df_front, df_top = truncate_to_shortest(df_front, df_top)

    x = df_top["xc"].to_numpy()
    y = df_top["yc"].to_numpy()
    z = df_front["yc"].to_numpy()  # TODO: convert pixel → real distance

    return pd.DataFrame({"x": x, "y": y, "z": z})


# ----------------------------
# geometry utilities
# ----------------------------

def line_from_points(p1: np.ndarray, p2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return line direction + point."""
    return p1, p2 - p1


def line_intersection_ext(
    p1: np.ndarray, d1: np.ndarray,
    p2: np.ndarray, d2: np.ndarray,
) -> Optional[np.ndarray]:
    """
    Compute intersection of two lines in 2D.
    Returns None if parallel.
    """
    A = np.array([d1, -d2]).T
    if np.linalg.matrix_rank(A) < 2:
        return None

    b = p2 - p1
    t = np.linalg.solve(A, b)
    return p1 + d1 * t[0]

