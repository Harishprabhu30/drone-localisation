#!/usr/bin/env bash

set -euo pipefail

# Usage:
#   ./create_structure.sh gnss_denied_uav_localization
#   ./create_structure.sh .
#
# If no argument is given, use current directory.
ROOT="${1:-.}"

# Directory list
DIRS=(
    "configs"
    "data/raw/zurich_mav_sample"
    "data/raw/alto"
#    "data/raw/company"
    "data/processed"

    "maps/orthophotos"
    "maps/tiles"
    "maps/metadata"

    "src/uavloc/data"
    "src/uavloc/geometry"
    "src/uavloc/inspection"
    "src/uavloc/visualization"
    "src/uavloc/relative"
    "src/uavloc/evaluation"
    "src/uavloc/utils"

    "scripts"
    "notebooks"

    "outputs/reports"
    "outputs/figures"
    "outputs/maps"
    "outputs/trajectories"
    "outputs/logs"

    "docs"
)

# File list
FILES=(
#   "README.md"
    "requirements.txt"
    "pyproject.toml"
    ".gitignore"

    "configs/dataset_zurich.yaml"
#   "configs/dataset_alto.yaml"
#   "configs/dataset_company_template.yaml"
    "configs/camera.yaml"
    "configs/sync.yaml"
    "configs/map.yaml"

    "data/README.md"

    "src/uavloc/data/dataset_base.py"
    "src/uavloc/data/zurich_loader.py"
#   "src/uavloc/data/alto_loader.py"
#   "src/uavloc/data/company_loader.py"
    "src/uavloc/data/video_loader.py"
    "src/uavloc/data/telemetry_loader.py"
    "src/uavloc/data/frame_sync.py"

    "src/uavloc/geometry/gps_enu.py"
    "src/uavloc/geometry/camera_model.py"
    "src/uavloc/geometry/transforms.py"
    "src/uavloc/geometry/attitude.py"

    "src/uavloc/inspection/dataset_report.py"
    "src/uavloc/inspection/image_stats.py"
    "src/uavloc/inspection/telemetry_stats.py"

    "src/uavloc/visualization/trajectory_plot.py"
    "src/uavloc/visualization/altitude_speed_plot.py"
    "src/uavloc/visualization/folium_map.py"
    "src/uavloc/visualization/sample_frames.py"

    "src/uavloc/relative/orb_motion.py"
    "src/uavloc/relative/optical_flow.py"
    "src/uavloc/relative/trajectory_builder.py"

    "src/uavloc/evaluation/metrics.py"
    "src/uavloc/evaluation/error_analysis.py"

    "src/uavloc/utils/io.py"
    "src/uavloc/utils/logging.py"
    "src/uavloc/utils/config.py"

    "scripts/inspect_dataset.py"
    "scripts/prepare_dataset.py"
    "scripts/run_reference_visualization.py"
    "scripts/run_frame_sync_check.py"
    "scripts/run_week2_demo.py"

    "notebooks/01_dataset_inspection.ipynb"
#   "notebooks/02_reference_trajectory.ipynb"
#   "notebooks/03_frame_sync_check.ipynb"

    "docs/week2_plan.md"
    "docs/dataset_format.md"
#   "docs/company_dataset_questions.md"
    "docs/limitations.md"
)

echo "Creating structure under: $ROOT"

mkdir -p "$ROOT"

# Create directories
for dir in "${DIRS[@]}"; do
    mkdir -p "$ROOT/$dir"
done

# Create files
for file in "${FILES[@]}"; do
    touch "$ROOT/$file"
done

# Create __init__.py files for all Python packages
find "$ROOT/src" -type d -exec touch {}/__init__.py \; 2>/dev/null || true

echo "Project structure created successfully."