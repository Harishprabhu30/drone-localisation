# Week 2 Progress — Zurich MAV Dataset Loader

## Phase

Week 2 — Development environment and baseline data pipeline.

## Achieved Goal & Status - 19th June 2026

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

## Update (22nd June 2026) — Reference Trajectory and Visualization Completed

Reference trajectory generation is now working for Zurich MAV. The project is currently tested on the Zurich Urban MAV sample dataset.

### Completed

* Python virtual environment created.
* Repository structure created.
* YAML-based dataset configuration added.
* Zurich MAV dataset loader implemented.
* Robust CSV reader added for messy dataset logs.
* Dataset inspection script added.
* Dataset report generation working.
* Camera calibration file detection working.
* Onboard GPS, ground truth, onboard pose, barometer, accelerometer, gyroscope, MAV images, calibration images, and Street View images are loaded and summarized.
* Reference trajectory generation implemented.
* GPS scaling issue fixed.
* UTM to local ENU conversion issue fixed.
* Reference trajectory visualization implemented.
* Folium interactive map visualization working.

### Current Working Commands

```bash
export PYTHONPATH=$PWD/src
python scripts/inspect_dataset.py --config configs/dataset_zurich.yaml
python scripts/build_reference_trajectory.py --config configs/dataset_zurich.yaml
python scripts/run_reference_visualization.py --config configs/dataset_zurich.yaml
```

Implemented files:

```text
src/uavloc/geometry/gps_enu.py
scripts/build_reference_trajectory.py
src/uavloc/visualization/trajectory_plot.py
src/uavloc/visualization/folium_map.py
scripts/run_reference_visualization.py
```

Generated outputs:

```text
outputs/zurich_mav_sample/trajectories/reference_trajectory.csv
outputs/zurich_mav_sample/reports/reference_origin.json
outputs/zurich_mav_sample/reports/reference_trajectory_summary.json
outputs/zurich_mav_sample/figures/trajectory_xy.png
outputs/zurich_mav_sample/figures/altitude_profile.png
outputs/zurich_mav_sample/figures/speed_profile.png
outputs/zurich_mav_sample/maps/trajectory_map.html
```

Important fixes:

* GPS scaling was corrected from raw scale to direct decimal values.
* ENU conversion bug was fixed by subtracting the UTM northing origin correctly.
* The Folium map now works and shows the travelled path interactively.


### Reference Trajectory Summary

```text
Rows: 81169
X range [m]: -193.25 to 165.09
Y range [m]: -234.27 to 331.58
Z range [m]: -15.95 to 54.39
```

### Current Interpretation

The Zurich reference trajectory now looks valid. The XY plot shows a realistic flight path in meters. The altitude profile shows realistic altitude variation around 449–520 m. The Folium HTML map opens interactively and shows the travelled path on the map.


Next planned block:

```text
Frame-to-telemetry synchronization
```

## Update (25th June 2026)

## 1. Output Organization Decision

From the next implementation block onward, outputs will be grouped by pipeline stage to avoid mixing unrelated plots, reports, and metadata files.

The main canonical files will remain in their simple locations when other scripts depend on them, for example:

```text
outputs/zurich_mav_sample/trajectories/reference_trajectory.csv
outputs/zurich_mav_sample/metadata/synchronized_frames.csv
outputs/zurich_mav_sample/metadata/frame_quality.csv
```

New diagnostic outputs will use stage folders such as:

```text
outputs/zurich_mav_sample/metadata/05_orb_pair_diagnostics/
outputs/zurich_mav_sample/reports/05_orb_pair_diagnostics/
outputs/zurich_mav_sample/figures/05_orb_pair_diagnostics/
```

This keeps the pipeline easier to inspect, debug, and explain during Week 3 relative localization work.


## 2. ORB Pair and Stride Diagnostics Completed

The ORB frame-pair diagnostics block has been completed successfully on the Zurich MAV sample dataset.

### ORB Consecutive Pair Diagnostics

Implemented files:

```text
src/uavloc/relative/orb_pair_diagnostics.py
scripts/run_orb_pair_diagnostics.py
```

Generated outputs:

```text
outputs/zurich_mav_sample/metadata/05_orb_pair_diagnostics/orb_pair_quality.csv
outputs/zurich_mav_sample/reports/05_orb_pair_diagnostics/orb_pair_summary.json
outputs/zurich_mav_sample/figures/05_orb_pair_diagnostics/orb_pair_quality_plots.png
outputs/zurich_mav_sample/figures/05_orb_pair_diagnostics/orb_matches_good_pair.png
outputs/zurich_mav_sample/figures/05_orb_pair_diagnostics/orb_matches_weak_pair.png
```

Successful summary:

```text
Total pairs: 349
Good pairs: 349
Medium pairs: 0
Weak pairs: 0
Homography successes: 349
Good matches median: 1347
RANSAC inliers median: 1202
Inlier ratio median: 0.903
```

Interpretation:

The Zurich MAV sample segment is highly suitable for ORB-based relative localization. Consecutive frames have strong visual overlap, many valid ORB matches, and very high RANSAC inlier ratios. The earlier image-quality weak candidates are therefore diagnostic warnings only, not confirmed localization failures.

### ORB Stride Diagnostics

Implemented file:

```text
scripts/run_orb_stride_diagnostics.py
```

Generated outputs:

```text
outputs/zurich_mav_sample/metadata/06_orb_stride_diagnostics/orb_stride_pair_quality.csv
outputs/zurich_mav_sample/metadata/06_orb_stride_diagnostics/orb_stride_summary.csv
outputs/zurich_mav_sample/reports/06_orb_stride_diagnostics/orb_stride_summary.json
outputs/zurich_mav_sample/figures/06_orb_stride_diagnostics/orb_stride_quality_summary.png
outputs/zurich_mav_sample/figures/06_orb_stride_diagnostics/orb_stride_reference_distance.png
```

Successful stride summary:

```text
stride 1:  good pairs 349/349, median inlier ratio 0.903
stride 2:  good pairs 348/348, median inlier ratio 0.860
stride 3:  good pairs 347/347, median inlier ratio 0.830
stride 5:  good pairs 345/345, median inlier ratio 0.781
stride 10: good pairs 340/340, median inlier ratio 0.692
```

Interpretation:

ORB matching remains stable even when skipping frames. For the first relative localization baseline, stride 1 will be used because it is safest and easiest to debug. Stride 5 can be tested later as a faster/efficient version. Stride 10 works on this sample but should not be selected first because harder datasets may have less overlap.

### Important Note

The reference frame-to-frame displacement is often very small or zero for short strides. This likely comes from low-rate, quantized, or repeated GNSS/reference positions relative to the image sequence. Therefore, GNSS should be used mainly for accumulated trajectory evaluation, not for judging every tiny frame-to-frame visual step.

### Next Planned Block

```text
Block: First ORB relative motion estimation
```

Goal:

```text
matched ORB points + homography
↓
estimate frame-to-frame image motion
↓
accumulate relative motion
↓
produce estimated local XY trajectory
↓
compare estimated trajectory shape against reference trajectory
```
