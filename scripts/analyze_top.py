# scripts/analyze_top.py

import argparse
from pathlib import Path
import yaml
import json

from monkey3d.analysis_top import TopAnalysisConfig, analyze_top_view
from monkey3d.roi import rois_from_dict


def main():
    parser = argparse.ArgumentParser(description="Run top-view analysis")
    parser.add_argument("--coords", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--outdir", default="plots")
    args = parser.parse_args()

    cfg_data = yaml.safe_load(open(args.config, "r"))
    fps = cfg_data["fps"]
    rois = rois_from_dict(cfg_data)

    cfg = TopAnalysisConfig(
        fps=fps,
        rois=rois,
        video_path=Path(args.video),
    )

    durations = analyze_top_view(cfg, Path(args.coords), Path(args.outdir))

    print(json.dumps(durations, indent=2))


if __name__ == "__main__":
    main()

