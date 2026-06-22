# Week 2 Progress — Zurich MAV Dataset Loader

## Phase

Week 2 — Development environment and baseline data pipeline.

## Current Goal & Current Status - 19th June 2026

Build a reusable dataset engine before starting visual localization. The current focus is:

1. Load image folders and telemetry logs.
2. Inspect dataset structure.
3. Generate dataset report.
4. Prepare reference trajectory.
5. Visualize GNSS/reference path.
6. Only then move to visual odometry.

## Dataset Used

Zurich Urban MAV sample dataset.

## Implemented Files

```text
configs/dataset_zurich.yaml
src/uavloc/utils/config.py
src/uavloc/data/zurich_loader.py
scripts/inspect_dataset.py
```

## Current Working Command

```bash
export PYTHONPATH=$PWD/src
python scripts/inspect_dataset.py --config configs/dataset_zurich.yaml
```

## Successful Dataset Summary

```text
mav_images: 350 images
calib_images: 30 images
streetview_images: 113 images
onboard_gps_file: 81169 rows
groundtruth_agl_file: 2708 rows
onboard_pose_file: 135098 rows
barometer_file: 27052 rows
raw_accel_file: 27050 rows
raw_gyro_file: 27050 rows
streetview_gps_file: 113 rows
calibration_available: True
```

## Debugging Issues Fixed

### Issue 1 — Header Row Treated as Data

Initial loader used `pd.read_csv(..., header=None)`, so header rows such as `Timestamp` and `imgid` were treated as numeric data.

Fix:

* Implemented robust CSV reader.
* Header-like rows are detected and removed.
* Numeric columns are converted safely.

### Issue 2 — Extra Trailing Commas

`StreetViewGPS.csv` contained rows with trailing commas, causing:

```text
ParserError: Expected 6 fields, saw 7
```

Fix:

* Reader now uses expected `usecols`.
* Extra trailing columns are ignored.
* Raw dataset files are not manually modified.

### Issue 3 — GPS File Path Mismatch

The onboard GPS file was not appearing in the report because the YAML path/key had a naming mismatch.

Fix:

* Corrected YAML path.
* `onboard_gps_file` now loads successfully.

## Important Dataset Observations

* The sample has only 350 MAV images, but GPS contains 81169 rows.
* Therefore, future synchronization must use `imgid` or nearest timestamp, not row index.
* Camera calibration file is available.
* The dataset contains GPS, ground truth, onboard pose, barometer, accelerometer, and gyro logs.
* This makes it suitable for Week 2 pipeline development and later baseline testing.

## Next Implementation Step

Create reference trajectory generation:

```text
scripts/build_reference_trajectory.py
src/uavloc/geometry/gps_enu.py
```

Expected generated files:

```text
outputs/zurich_mav_sample/trajectories/reference_trajectory.csv
outputs/zurich_mav_sample/reports/reference_origin.json
outputs/zurich_mav_sample/reports/reference_trajectory_summary.json
```

## Rule Going Forward

Do not edit files inside `data/raw/` unless absolutely necessary. Dataset-specific problems should be handled inside loader code.
