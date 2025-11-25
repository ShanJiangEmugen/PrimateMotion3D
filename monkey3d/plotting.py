# monkey3d/plotting.py

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .roi import ROI


def _draw_rois(ax, rois):
    for roi in rois:
        pts = np.array(roi.vertices)
        pts = np.vstack([pts, pts[0]])
        ax.plot(pts[:, 0], pts[:, 1], "--", linewidth=1)


def plot_trace_image(frame, df, rois, title="Trace"):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.imshow(frame[:, :, ::-1])  # BGR→RGB
    ax.plot(df["xc"], df["yc"], ".", markersize=1)
    _draw_rois(ax, rois)
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    return fig


def plot_trace_by_surface_image(frame, df, rois, title="Trace by surface"):
    surfaces = df["surface"].unique()
    n = len(surfaces)

    fig, axes = plt.subplots(1, n, figsize=(4 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, surface in zip(axes, surfaces):
        sub = df[df["surface"] == surface]
        ax.imshow(frame[:, :, ::-1])
        ax.plot(sub["xc"], sub["yc"], ".", markersize=1)
        _draw_rois(ax, rois)
        ax.set_title(surface)
        ax.axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_duration_bar(df, fps, title="Surface durations"):
    counts = df["surface"].value_counts().to_dict()
    durations = {k: v / fps for k, v in counts.items()}

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(list(durations.keys()), list(durations.values()))
    ax.set_ylabel("Time (s)")
    ax.set_title(title)
    fig.tight_layout()

    return fig, durations

