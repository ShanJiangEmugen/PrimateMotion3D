# monkey3d/roi.py

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Iterable, List, Tuple, Optional

import numpy as np
import cv2


Point = Tuple[float, float]


class Surface(str, Enum):
    """Logical cage surfaces."""
    LEFT = "left"
    RIGHT = "right"
    BACK = "back"
    FRONT = "front"
    BOTTOM = "bottom"
    TOP = "top"
    UNKNOWN = "unknown"


@dataclass
class ROI:
    """Polygonal ROI for a cage surface in image coordinates.

    Attributes
    ----------
    name:
        Human-readable name (e.g. 'left_wall').
    surface:
        Logical surface label, e.g. Surface.LEFT or Surface.BOTTOM.
    vertices:
        List of (x, y) points in image pixel coordinates, in order.
    """
    name: str
    surface: Surface
    vertices: List[Point]

    def contains(self, point: Point) -> bool:
        """Return True if the point lies inside this ROI polygon."""
        # cv2 expects Nx1x2 array of float32
        contour = np.array(self.vertices, dtype=np.float32).reshape((-1, 1, 2))
        pt = np.array(point, dtype=np.float32)
        # pointPolygonTest > 0: strictly inside, =0: on edge, <0: outside
        res = cv2.pointPolygonTest(contour, pt, False)
        return res >= 0.0


def compute_box_center(x1: float, y1: float, x2: float, y2: float) -> Point:
    """Compute center of a bounding box given top-left and bottom-right corners."""
    xc = (x1 + x2) / 2.0
    yc = (y1 + y2) / 2.0
    return float(xc), float(yc)


def assign_surface_for_points(
    points: Iterable[Point],
    rois: List[ROI],
    default_surface: Surface = Surface.UNKNOWN,
) -> List[Surface]:
    """Assign a logical surface to each point based on which ROI contains it.

    Parameters
    ----------
    points:
        Iterable of (x, y) pixel coordinates (e.g. tracked centers).
    rois:
        List of ROI objects defining all cage surfaces.
    default_surface:
        Label used when a point is not inside any ROI.

    Returns
    -------
    surfaces:
        List of Surface enums, one per input point.
    """
    surfaces: List[Surface] = []

    for x, y in points:
        surface = default_surface
        for roi in rois:
            if roi.contains((x, y)):
                surface = roi.surface
                break
        surfaces.append(surface)

    return surfaces


def rois_from_dict(config: Dict) -> List[ROI]:
    """Create ROI objects from a config dict, e.g. loaded from YAML.

    Example config section:
    rois:
      - name: left_wall
        surface: left
        vertices: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
      - name: bottom
        surface: bottom
        vertices: ...

    """
    rois_cfg = config.get("rois", [])
    rois: List[ROI] = []
    for item in rois_cfg:
        name = item["name"]
        surface_str = item["surface"].upper()
        surface = Surface[surface_str]
        vertices = [tuple(v) for v in item["vertices"]]
        rois.append(ROI(name=name, surface=surface, vertices=vertices))
    return rois

