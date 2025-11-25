# Monkey3D-TrackingToolkit

A modular multi-view locomotion tracking toolkit for primate behavioral research.

<p align="left"> <!-- Release --> <img src="https://img.shields.io/github/v/release/ShanJiangEmugen/PrimateMotion3D?color=blue&label=Release&style=flat-square" /> 
  <!-- License --> <img src="https://img.shields.io/github/license/ShanJiangEmugen/PrimateMotion3D?style=flat-square" /> 
  <!-- Issues --> <img src="https://img.shields.io/github/issues/ShanJiangEmugen/PrimateMotion3D?style=flat-square" /> 
  <!-- Python Version --> <img src="https://img.shields.io/badge/Python-3.8%20|%203.9%20|%203.10-blue?style=flat-square" /> 
  <!-- Platform --> <img src="https://img.shields.io/badge/Platform-Linux%20%7C%20macOS%20%7C%20Windows-lightgrey?style=flat-square" /> 
</p>



## Overview
**Monkey3D-TrackingToolkit** is a lightweight, modular, and production-ready toolkit for **multi-view primate locomotion tracking** in caged environments.
It provides:

- **Front & top view 2D trajectory extraction**
- **ROI-based surface classification** (left / right / back / bottom / top)
- **Time-on-surface quantification**
- **Trajectory visualization & summary plots**
- **Interactive ROI annotation tools**
- **Foundational utilities for 3D reconstruction** (front + top fusion)


The toolkit is designed for researchers in neuroscience, behavior analysis, and non-human primate studies, providing clean APIs and modular components for integration into larger pipelines.

## Features
- ROI Annotation       
  - Interactive tool (find_vertices.py) for clicking cage corners & surfaces       
  - YAML-based configuration, reusable across experiments       

- Front View Analysis
  - Load tracking CSV (xc, yc or bounding box)
  - Assign each frame to surfaces via polygon-in-ROI
  - Output:
    - 2D trajectory
    - Surface-wise trajectories
    - Time-on-surface bar charts

- Top View Analysis
  - Same pipeline as front view
  - Supports multi-view synchronization

- 3D Reconstruction (Prototype)
  - Combines front z-axis + top x-y
  - Geometry utilities for extending to calibration-based reconstruction

- Clean, Modular Codebase
  - Easy to extend to:
    - Additional sensors
    - Depth cameras
    - Multi-view triangulation
    - ML-based behavioral classification


## Installation
```bash
git clone https://github.com/ShanJiangEmugen/Monkey3D-TrackingToolkit.git
cd Monkey3D-TrackingToolkit
pip install -r requirements.txt
```

## Quick Start
#### 1. Create ROIs for Your Cage
Use the interactive point-picking tool:       
```bash
python scripts/find_vertices.py path/to/frame.png
```

Click your cage surfaces and copy the output into config.yaml.

#### 2. Run Front-View Analysis
```bash
python scripts/analyze_front.py \
    --coords data/examples/front_coords.csv \
    --video data/examples/front_video.mp4 \
    --config monkey3d/config_example.yaml \
    --outdir outputs/front/
```

This will automatically generate:
- `*_trace.png`
- `*_trace_by_surface.png`
- `*_duration.png`

#### 3. Run Top-View Analysis
```bash
python scripts/analyze_top.py \
    --coords data/examples/top_coords.csv \
    --video data/examples/top_video.mp4 \
    --config monkey3d/config_example.yaml \
    --outdir outputs/top/
```

#### 4. Simple 3D Reconstruction Demo
```bash
python scripts/build_3d_demo.py \
    --front data/examples/front_coords.csv \
    --top data/examples/top_coords.csv \
    --out outputs/trajectory3d.png
```

## Configuration (YAML)
Example:
```yaml
fps: 30.0

rois:
  - name: left_wall
    surface: left
    vertices:
      - [120, 590]
      - [210, 100]
      - [330, 110]
      - [300, 600]

  - name: right_wall
    surface: right
    vertices:
      - [600, 580]
      - [680, 120]
      - [790, 130]
      - [770, 590]

  - name: bottom
    surface: bottom
    vertices:
      - [250, 690]
      - [550, 695]
      - [540, 720]
      - [260, 725]
```

## Example Output
- Front view trajectory
- Surface-wise subplots
- Duration statistics
- 3D trajectory line plot

## Roadmap
- Add calibration module for real-world coordinates
- True 3D triangulation from dual-view geometry
- Behavioral state classification (optional ML module)
- Real-time tracking support
- GUI for ROI editing

