# drone-localisation
Relative and Absolute Localisation of the drone using a map

## Workflow steps to be followed:
1. go to project directory, cd drone_localisation then, 

2. activate pyenv version: pyenv local 3.10.13, then check python --version

3. activate virtual environment: source .drone_venv/bin/activate

4. Verify again: which python; python --version

## Current Status - 19th June 2026

This repository implements a modular GNSS-denied / weak-GNSS UAV visual localization pipeline.

The current focus is Week 2: dataset loading, telemetry inspection, coordinate preparation, and reference trajectory visualization before starting visual odometry.

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
* MAV image, calibration image, Street View image, GPS, pose, barometer, accelerometer, and gyro logs are loaded and summarized.

### Current Working Command

```bash
export PYTHONPATH=$PWD/src
python scripts/inspect_dataset.py --config configs/dataset_zurich.yaml
```

### Current Successful Output Summary

```text
Dataset name: zurich_mav_sample
Dataset type: zurich_mav
MAV images: 350
Calibration images: 30
Street View images: 113
Onboard GPS: 81169 rows
GroundTruthAGL: 2708 rows
OnboardPose: 135098 rows
Barometer: 27052 rows
RawAccel: 27050 rows
RawGyro: 27050 rows
StreetViewGPS: 113 rows
Calibration available: True
```

### Important Dataset Notes

* The Zurich sample image folder contains 350 MAV images.
* The onboard GPS log contains 81169 rows, so image-to-telemetry matching must not be done by row number.
* Synchronization should be done using `imgid` and/or timestamps.
* Some CSV files contain header rows.
* Some CSV rows contain trailing commas, especially in `StreetViewGPS.csv`.
* The loader handles these formatting issues without modifying the raw dataset.
* Raw dataset files should remain unchanged whenever possible.

### Next Step

Build the reference trajectory generation script:

```bash
python scripts/build_reference_trajectory.py --config configs/dataset_zurich.yaml
```

Expected output:

```text
outputs/zurich_mav_sample/trajectories/reference_trajectory.csv
outputs/zurich_mav_sample/reports/reference_origin.json
outputs/zurich_mav_sample/reports/reference_trajectory_summary.json
```
