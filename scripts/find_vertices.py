# scripts/find_vertices.py

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np


Point = Tuple[int, int]


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Interactively pick polygon vertices on an image. "
            "Left-click to add a point, 'n' to start a new ROI, 's' to print all ROIs."
        )
    )
    parser.add_argument("image", type=str, help="Path to image file")
    args = parser.parse_args()

    img_path = Path(args.image)
    if not img_path.is_file():
        raise FileNotFoundError(img_path)

    img = cv2.imread(str(img_path))
    if img is None:
        raise RuntimeError(f"Failed to read image: {img_path}")

    rois: List[List[Point]] = [[]]  # List of ROIs, each ROI is a list of points
    current_idx = 0
    window_name = "ROI picker"

    def mouse_callback(event, x, y, flags, param):
        nonlocal rois, current_idx, img

        if event == cv2.EVENT_LBUTTONDOWN:
            # add point to current ROI
            rois[current_idx].append((x, y))
            print(f"ROI {current_idx}, point: ({x}, {y})")

            # draw point
            cv2.circle(img, (x, y), 4, (0, 0, 255), -1)
            cv2.putText(
                img,
                f"{len(rois[current_idx])}",
                (x + 5, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_callback)

    print("Instructions:")
    print("  - Left-click to add a point to current ROI.")
    print("  - Press 'n' to start a new ROI.")
    print("  - Press 's' to print all ROIs and continue.")
    print("  - Press ESC to exit.")

    while True:
        cv2.imshow(window_name, img)
        key = cv2.waitKey(50) & 0xFF

        if key == 27:  # ESC
            break
        elif key == ord("n"):
            # start new ROI
            if rois[current_idx]:
                print(f"Starting new ROI {current_idx + 1}")
                rois.append([])
                current_idx += 1
        elif key == ord("s"):
            # print all ROIs in a YAML-friendly way
            print("\nROIs (copy into config YAML):")
            for i, roi in enumerate(rois):
                if not roi:
                    continue
                print(f"- name: roi_{i}")
                print("  surface: ???  # TODO: left/right/back/bottom/top")
                print("  vertices:")
                for x, y in roi:
                    print(f"    - [{x}, {y}]")
            print("---- end ----\n")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

