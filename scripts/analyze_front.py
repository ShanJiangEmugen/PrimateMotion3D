# scripts/analyze_front.py

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from monkey3d.analysis_front import FrontAnalysisConfig, analyze_front_view
from monkey3d.roi import rois_from_dict


def main():
    parser = argparse.ArgumentParser(description="Run front-view analysis for one video.")
    parser.add_argument("--coords", type=str, required=True, help="Path to CSV of tracking coords")
    parser.add_argument("--video", type=str, required=True, help="Path to front-view video file")
    parser.add_argument("--config", type=str, required=True, help="YAML config with ROI and fps")
    parser.add_argument(
        "--outdir",
        type=str,
        default="plots",
        help="Directory to save plots (default: ./plots)",
    )
    args = parser.parse_args()

    cfg_path = Path(args.config)
    cfg_data = yaml.safe_load(cfg_path.read_text())

    fps = float(cfg_data["fps"])
    rois = rois_from_dict(cfg_data)

    cfg = FrontAnalysisConfig(
        fps=fps,
        rois=rois,
        video_path=Path(args.video),
    )

    durations = analyze_front_view(
        cfg=cfg,
        coord_csv=Path(args.coords),
        output_dir=Path(args.outdir),
    )

    print("Durations (seconds):")
    print(json.dumps(durations, indent=2))


if __name__ == "__main__":
    main()

