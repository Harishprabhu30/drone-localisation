# Zurich Urban MAV Dataset Notes

## Purpose in This Project

The Zurich Urban MAV sample dataset is used as the first Week 2 development dataset for the GNSS-denied UAV localization project.

Its purpose is to test:

* dataset folder parsing,
* image and telemetry loading,
* GPS/reference trajectory preparation,
* timestamp and `imgid` understanding,
* camera calibration availability,
* future frame-to-telemetry synchronization.

This dataset is used before moving to ALTO or company-provided drone data.

---

## Current Dataset Location

Current local project path:

```text
data/raw/zurich_mav_sample/
```

Current config file:

```text
configs/dataset_zurich.yaml
```

Current output folder:

```text
outputs/zurich_mav_sample/
```

---

## Available Data

The sample dataset currently contains:

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

---

## Important Files

### MAV Images

```text
MAV Images/
```

Contains aerial MAV images.

Current sample count:

```text
350 images
```

### Calibration Images

```text
MAV Images Calib/
```

Contains calibration images used to estimate intrinsic camera parameters.

Current sample count:

```text
30 images
```

### Street View Images

```text
Street View Images/
```

Contains Google Street View reference/database images.

Current sample count:

```text
113 images
```

### Camera Calibration

```text
calibration_data.npz
```

Contains internal camera calibration parameters.

The loader currently detects this file successfully.

---

## Log Files

### Onboard GPS

```text
OnboardGPS.csv
```

Original dataset spelling may be:

```text
OnbordGPS.csv
```

In this local project, the file was renamed to:

```text
OnboardGPS.csv
```

Expected columns:

```text
timestamp
imgid
lat_raw
lon_raw
alt_raw
s_variance_m_s
c_variance_rad
fix_type
eph_m
epv_m
vel_n_m_s
vel_e_m_s
vel_d_m_s
num_sat
```

Important scaling:

```text
latitude  = lat_raw * 1e-7
longitude = lon_raw * 1e-7
altitude  = alt_raw * 1e-3
```

Notes:

* GPS has 81169 rows.
* MAV image sample has only 350 images.
* Therefore, future synchronization must not use row index.
* Use `imgid` and/or nearest timestamp matching.

---

### Ground Truth AGL

```text
GroundTruthAGL.csv
```

Expected columns:

```text
imgid
x_gt
y_gt
z_gt
omega_yaw_deg
phi_pitch_deg
kappa_roll_deg
x_gps
y_gps
z_gps
```

Notes:

* Position values are given in WGS84 / UTM zone 32N.
* This file is useful for reference trajectory comparison.
* It has fewer rows than onboard GPS.

---

### Onboard Pose

```text
OnboardPose.csv
```

Expected columns:

```text
timestamp
omega_x
omega_y
omega_z
accel_x
accel_y
accel_z
vel_x
vel_y
vel_z
acc_bias_x
acc_bias_y
acc_bias_z
azimuth
attitude_w
attitude_x
attitude_y
attitude_z
height
altitude
veh_pitch
tether_angle
tether_angle_dot
tether_force
gps_on
```

Notes:

* Contains raw sensor data and onboard pose estimate from Pixhawk/autopilot.
* Useful later for IMU, attitude, velocity, altitude, and possible fusion experiments.

---

### Barometer

```text
BarometricPressure.csv
```

Expected columns:

```text
timestamp
pressure
altitude
temperature
```

Notes:

* Useful for altitude profile and altitude-based visual motion scaling.

---

### Raw Accelerometer

```text
RawAccel.csv
```

Expected columns:

```text
timestamp
error_count
x
y
z
temperature
range_rad_s
scaling
x_raw
y_raw
z_raw
temperature_raw
```

---

### Raw Gyroscope

```text
RawGyro.csv
```

Expected columns:

```text
timestamp
error_count
x
y
z
temperature
range_rad_s
scaling
x_raw
y_raw
z_raw
temperature_raw
```

---

### Street View GPS

```text
StreetViewGPS.csv
```

Expected columns:

```text
latitude
longitude
yaw_degree
tilt_yaw_degree
tilt_pitch_degree
auxiliary
```

Known issue:

Some rows contain a trailing comma, for example:

```text
47.386938, 8.542639, 336.61, -12.47, 1.92, -1,
```

This creates an extra empty column.

The current loader handles this by reading only the expected number of columns.

---

## Known Dataset Quirks

### 1. Header Rows

Some CSV files contain header rows such as:

```text
Timestamp
imgid
```

The loader handles this by detecting and removing header-like rows.

### 2. Trailing Commas

Some CSV rows contain trailing commas, especially in `StreetViewGPS.csv`.

The loader ignores extra trailing fields.

### 3. File Name Mismatch

The original dataset uses:

```text
OnbordGPS.csv
```

The local project currently uses:

```text
OnboardGPS.csv
```

The YAML config must match the local file name exactly.

### 4. Image Count and GPS Count Differ

The sample has:

```text
350 MAV images
81169 GPS rows
```

So future synchronization must use:

```text
imgid
timestamp
nearest-neighbor matching
```

not row number.

---

## Current Loader Files

Implemented files:

```text
configs/dataset_zurich.yaml
src/uavloc/data/zurich_loader.py
src/uavloc/utils/config.py
scripts/inspect_dataset.py
```

---

## Current Working Command

Run from project root:

```bash
export PYTHONPATH=$PWD/src
python scripts/inspect_dataset.py --config configs/dataset_zurich.yaml
```

Successful output:

```text
Dataset name: zurich_mav_sample
Dataset type: zurich_mav
Raw data dir: data/raw/zurich_mav_sample
Output dir: outputs/zurich_mav_sample
Found 513 files.

Zurich MAV dataset summary
--------------------------
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
Calibration available: True

Saved dataset report to: outputs/zurich_mav_sample/reports/dataset_report.json
```

---

## Generated Output

Current generated report:

```text
outputs/zurich_mav_sample/reports/dataset_report.json
```

This report summarizes:

* dataset name,
* dataset type,
* image folders,
* image counts,
* log file row counts,
* timestamp ranges,
* calibration availability,
* missing files if any.

---

## Next Planned Step

Build reference trajectory generation.

Planned files:

```text
src/uavloc/geometry/gps_enu.py
scripts/build_reference_trajectory.py
```

Expected outputs:

```text
outputs/zurich_mav_sample/trajectories/reference_trajectory.csv
outputs/zurich_mav_sample/reports/reference_origin.json
outputs/zurich_mav_sample/reports/reference_trajectory_summary.json
```

Expected reference trajectory format:

```text
timestamp
imgid
lat
lon
alt
x_enu_m
y_enu_m
z_enu_m
reference_type
fix_type
num_sat
eph_m
epv_m
vel_n_m_s
vel_e_m_s
vel_d_m_s
```

---

## Rule Going Forward

Do not manually edit files inside:

```text
data/raw/
```

Raw dataset files should stay unchanged whenever possible.

Dataset-specific problems should be handled inside loader code.
