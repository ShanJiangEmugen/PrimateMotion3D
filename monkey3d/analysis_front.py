# monkey3d/analysis_front.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from .roi import ROI, Surface, compute_box_center, assign_surface_for_points


@dataclass
class FrontAnalysisConfig:
    """Configuration for front-view analysis."""
    fps: float
    rois: List[ROI]
    video_path: Optional[Path] = None


def load_coordinates_csv(path: Path) -> pd.DataFrame:
    """Load tracking coordinates.

    Expected columns (adapt as needed):
    - x1, y1, x2, y2: bounding box corners
    - or directly xc, yc if centers are pre-computed.
    """
    df = pd.read_csv(path)

    if {"xc", "yc"}.issubset(df.columns):
        # centers already available
        centers = df[["xc", "yc"]].to_numpy()
    else:
        required = {"x1", "y1", "x2", "y2"}
        if not required.issubset(df.columns):
            raise ValueError(f"CSV must contain either (xc,yc) or {required}")
        centers = []
        for _, row in df.iterrows():
            xc, yc = compute_box_center(row["x1"], row["y1"], row["x2"], row["y2"])
            centers.append((xc, yc))
        centers = np.array(centers)
        df["xc"] = centers[:, 0]
        df["yc"] = centers[:, 1]

    return df


def classify_surfaces_for_dataframe(df: pd.DataFrame, rois: List[ROI]) -> pd.DataFrame:
    """Add a 'surface' column to df based on ROI membership of (xc, yc)."""
    points = list(zip(df["xc"].to_numpy(), df["yc"].to_numpy()))
    surfaces = assign_surface_for_points(points, rois)
    df = df.copy()
    df["surface"] = [s.value for s in surfaces]
    return df


# ----------------- plotting helpers ----------------- #

def _plot_rois_on_axes(ax: plt.Axes, rois: List[ROI]) -> None:
    for roi in rois:
        pts = np.array(roi.vertices)
        pts_closed = np.vstack([pts, pts[0]])
        ax.plot(pts_closed[:, 0], pts_closed[:, 1], linestyle="--", linewidth=1.0)


def plot_trace(
    frame: np.ndarray,
    df: pd.DataFrame,
    rois: List[ROI],
    title: str = "Front view trace",
) -> plt.Figure:
    """Plot full 2D trace on a single frame background."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(frame[..., ::-1])  # BGR -> RGB
    ax.plot(df["xc"], df["yc"], ".", markersize=1)
    _plot_rois_on_axes(ax, rois)
    ax.set_title(title)
    ax.set_axis_off()
    fig.tight_layout()
    return fig


def plot_trace_by_surface(
    frame: np.ndarray,
    df: pd.DataFrame,
    rois: List[ROI],
    title: str = "Front view trace by surface",
) -> plt.Figure:
    surfaces_unique = sorted(df["surface"].unique())
    n = len(surfaces_unique)
    cols = min(n, 3)
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
    axes = np.atleast_1d(axes).reshape(-1)

    for ax, surface in zip(axes, surfaces_unique):
        sub = df[df["surface"] == surface]
        ax.imshow(frame[..., ::-1])
        ax.plot(sub["xc"], sub["yc"], ".", markersize=1)
        _plot_rois_on_axes(ax, rois)
        ax.set_title(surface)
        ax.set_axis_off()

    # hide extra axes
    for ax in axes[len(surfaces_unique) :]:
        ax.set_visible(False)

    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_duration_bar(
    df: pd.DataFrame,
    fps: float,
    title: str = "Time spent per surface",
) -> Tuple[plt.Figure, Dict[str, float]]:
    """Plot bar chart of total time spent on each surface.

    Returns the figure and a dict {surface: seconds}.
    """
    counts = df["surface"].value_counts().to_dict()
    durations_sec = {k: v / fps for k, v in counts.items()}

    surfaces = list(durations_sec.keys())
    values = [durations_sec[s] for s in surfaces]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(surfaces, values)
    ax.set_ylabel("Time (s)")
    ax.set_title(title)
    fig.tight_layout()
    return fig, durations_sec


# ----------------- high-level API ----------------- #

def analyze_front_view(
    cfg: FrontAnalysisConfig,
    coord_csv: Path,
    output_dir: Path,
    max_frames: Optional[int] = None,
) -> Dict[str, float]:
    """Run the full front-view analysis pipeline for a single video.

    Steps:
    1. Load tracking CSV, compute centers if needed.
    2. Classify each frame's center to a cage surface.
    3. Grab a background frame from the video (first frame by default).
    4. Plot full trace, trace by surface, and time-per-surface bar chart.
    5. Return durations in seconds for each surface.

    Parameters
    ----------
    cfg:
        FrontAnalysisConfig object.
    coord_csv:
        Path to CSV file containing tracking results.
    output_dir:
        Where to save plots.
    max_frames:
        If set, truncate to this number of frames (for debugging).

    """
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_coordinates_csv(coord_csv)
    if max_frames is not None:
        df = df.iloc[:max_frames].reset_index(drop=True)

    df = classify_surfaces_for_dataframe(df, cfg.rois)

    if cfg.video_path is None:
        raise ValueError("cfg.video_path must be set to draw background frame.")

    cap = cv2.VideoCapture(str(cfg.video_path))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Failed to read first frame from {cfg.video_path}")

    video_stem = cfg.video_path.stem

    # 1) full trace
    fig_trace = plot_trace(frame, df, cfg.rois, title=f"{video_stem} — trace")
    fig_trace.savefig(output_dir / f"{video_stem}_trace.png", dpi=200)
    plt.close(fig_trace)

    # 2) per-surface trace
    fig_trace_surf = plot_trace_by_surface(
        frame, df, cfg.rois, title=f"{video_stem} — trace by surface"
    )
    fig_trace_surf.savefig(output_dir / f"{video_stem}_trace_by_surface.png", dpi=200)
    plt.close(fig_trace_surf)

    # 3) duration bar
    fig_dur, durations = plot_duration_bar(
        df, cfg.fps, title=f"{video_stem} — time per surface"
    )
    fig_dur.savefig(output_dir / f"{video_stem}_duration.png", dpi=200)
    plt.close(fig_dur)

    return durations

