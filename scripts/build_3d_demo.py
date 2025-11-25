# scripts/build_3d_demo.py

from __future__ import annotations
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from monkey3d.analysis_3d import combine_front_top
from monkey3d.tracking_io import load_tracking_csv


def main():
    parser = argparse.ArgumentParser(description="Build a simple 3D trajectory demo")
    parser.add_argument("--front", required=True, help="Front-view CSV")
    parser.add_argument("--top", required=True, help="Top-view CSV")
    parser.add_argument("--out", default="3d_demo.png")
    args = parser.parse_args()

    df_front = load_tracking_csv(Path(args.front))
    df_top = load_tracking_csv(Path(args.top))

    df3d = combine_front_top(df_front, df_top)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(projection="3d")
    ax.plot(df3d["x"], df3d["y"], df3d["z"], linewidth=1)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D Trajectory Demo")

    fig.tight_layout()
    fig.savefig(args.out, dpi=200)
    print("Saved:", args.out)


if __name__ == "__main__":
    main()

