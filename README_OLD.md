# drone-localisation
Relative and Absolute Localisation of the drone using a map

## Workflow steps to be followed:
1. go to project directory, cd drone_localisation then, 

2. activate pyenv version: pyenv local 3.10.13, then check python --version

3. activate virtual environment: source .drone_venv/bin/activate

4. Verify again: which python; python --version

## Current Status — Week 2 Data Pipeline - 22th June 2026

The current focus is Week 2: dataset loading, telemetry inspection, coordinate conversion, reference trajectory generation, and visualization before starting visual odometry.

The project is currently tested on the Zurich Urban MAV sample dataset.

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

### Current Successful Outputs

```text
outputs/zurich_mav_sample/reports/dataset_report.json
outputs/zurich_mav_sample/trajectories/reference_trajectory.csv
outputs/zurich_mav_sample/reports/reference_origin.json
outputs/zurich_mav_sample/reports/reference_trajectory_summary.json
outputs/zurich_mav_sample/figures/trajectory_xy.png
outputs/zurich_mav_sample/figures/altitude_profile.png
outputs/zurich_mav_sample/figures/speed_profile.png
outputs/zurich_mav_sample/maps/trajectory_map.html
```

### Reference Trajectory Summary

```text
Rows: 81169
X range [m]: -193.25 to 165.09
Y range [m]: -234.27 to 331.58
Z range [m]: -15.95 to 54.39
```

### Current Interpretation

The Zurich reference trajectory now looks valid. The XY plot shows a realistic flight path in meters. The altitude profile shows realistic altitude variation around 449–520 m. The Folium HTML map opens interactively and shows the travelled path on the map.

### Next Step

Implement frame-to-telemetry synchronization for the Zurich MAV sample.

Planned output:

```text
outputs/zurich_mav_sample/metadata/synchronized_frames.csv
```

