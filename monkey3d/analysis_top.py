# monkey3d/analysis_top.py

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Tuple

import cv2
import pandas as pd

from .roi import ROI, assign_surface_for_points
from .tracking_io import load_tracking_csv
from .plotting import (
    plot_trace_image,
    plot_trace_by_surface_image,
    plot_duration_bar,
)


@dataclass
class TopAnalysisConfig:
    fps: float
    rois: List[ROI]
    video_path: Optional[Path] = None


def analyze_top_view(
    cfg: TopAnalysisConfig,
    coord_csv: Path,
    output_dir: Path,
    max_frames: Optional[int] = None,
) -> Dict[str, float]:

    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_tracking_csv(coord_csv)
    if max_frames is not None:
        df = df.iloc[:max_frames].reset_index(drop=True)

    # classify surfaces
    points = list(zip(df["xc"], df["yc"]))
    surfaces = assign_surface_for_points(points, cfg.rois)
    df["surface"] = [s.value for s in surfaces]

    # Load first frame for drawing
    cap = cv2.VideoCapture(str(cfg.video_path))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Failed to read first frame from {cfg.video_path}")

    video_stem = cfg.video_path.stem

    # 1) Full trace
    fig_trace = plot_trace_image(frame, df, cfg.rois, title=f"{video_stem} — top trace")
    fig_trace.savefig(output_dir / f"{video_stem}_top_trace.png", dpi=200)
    fig_trace.close()

    # 2) Per-surface
    fig_surf = plot_trace_by_surface_image(frame, df, cfg.rois, title=f"{video_stem} — top by surface")
    fig_surf.savefig(output_dir / f"{video_stem}_top_by_surface.png", dpi=200)
    fig_surf.close()

    # 3) Duration
    fig_dur, durations = plot_duration_bar(df, cfg.fps, title=f"{video_stem} — top durations")
    fig_dur.savefig(output_dir / f"{video_stem}_top_duration.png", dpi=200)
    fig_dur.close()

    return durations

