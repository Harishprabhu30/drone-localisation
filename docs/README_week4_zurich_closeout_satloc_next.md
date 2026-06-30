# Week 4 Closeout — Zurich MAV Diagnostics and SatLoc Transition

This section closes the Zurich MAV relative-localization and sensor-diagnostic phase and defines the next phase using SatLoc for map-based localization.

---

## 1. Current Phase Status - 30th June 2026

The Zurich MAV work has completed the following blocks:

```text
10A.1–10A.4  Sensor stream timing, IMU/barometer signal checks
10A.5        Attitude/quaternion diagnostic
10A.6        Raw IMU ↔ OnboardPose axis/sign mapping
10A.7        Camera motion vs gyro yaw diagnostic on ORB windows
10B          Sensor-assisted ORB metric scaling experiment
```

The purpose of these blocks was not to build final VIO/EKF fusion. The purpose was to decide whether Zurich MAV is suitable for reliable metric visual localization and whether the available sensors can support the next estimator stage.

---

## 2. Zurich MAV: What Worked

Zurich MAV remains useful for:

```text
- dataset loading and synchronization
- reference trajectory generation
- camera calibration loading
- ORB feature tracking
- stride diagnostics
- visual relative-localization baseline
- metric-scaling failure analysis
- IMU/barometer/quaternion signal diagnostics
```

The important result is that ORB image-to-image tracking did not fail. On the official windows, ORB pair tracking remained successful:

```text
full_00001_01000_stride1: 999 / 999 ORB pairs OK
full_00001_01000_stride5: 199 / 199 ORB pairs OK
full_40000_41000_stride1: 1000 / 1000 ORB pairs OK
full_40000_41000_stride5: 200 / 200 ORB pairs OK
```

The issue is not feature tracking. The issue is the conversion of image motion into physically valid metric ENU/local displacement.

---

## 3. Zurich MAV Sensor Diagnostics Summary

### 3.1 Timing and stream availability

The camera, OnboardPose, raw IMU, barometer, and GNSS/reference streams are time-overlapping and usable for diagnostics.

Approximate stream rates found during diagnostics:

```text
camera / synchronized frames: ~30 Hz
onboard GPS:                 ~30 Hz
OnboardPose:                 ~50 Hz
raw accelerometer:           ~10 Hz
raw gyroscope:               ~10 Hz
barometer:                   ~9 Hz
```

The raw IMU is lower rate than the camera, but the weak visual-inertial relation is not mainly a timestamp problem. The bigger problem is camera geometry and projection.

### 3.2 Attitude / quaternion diagnostic

OnboardPose quaternion is structurally valid:

```text
quaternion norm median: 1.000000035
roll range:  -9.47° to 3.91°
pitch range: -10.52° to 7.49°
yaw range:   -130.58° to 178.84°
```

This means the quaternion is meaningful as an attitude candidate. However, camera-to-body extrinsics are still not confirmed.

### 3.3 Raw IMU ↔ OnboardPose axis/sign mapping

Best accelerometer mapping:

```text
pose_accel_x ≈ -raw_accel_y
pose_accel_y ≈ -raw_accel_x
pose_accel_z ≈ -raw_accel_z
mean_abs_corr ≈ 0.719
```

Best gyroscope mapping:

```text
pose_omega_x ≈ -raw_gyro_x
pose_omega_y ≈ -raw_gyro_y
pose_omega_z ≈ -raw_gyro_z
mean_abs_corr ≈ 0.734
```

The strongest finding is the z/yaw-rate relation:

```text
pose_omega_z ≈ -raw_gyro_z
corr_z ≈ 0.999
```

So raw gyro z is meaningful after sign correction. Raw accelerometer is physically meaningful but harder to use because it includes gravity, vibration, platform acceleration, and frame convention differences.

---

## 4. Camera Motion vs Gyro Yaw Result

The 10A.7 diagnostic compared ORB affine image-plane rotation with:

```text
- OnboardPose omega_z integrated yaw
- sign-corrected raw gyro z integrated yaw
- quaternion yaw delta
```

Results:

```text
full_00001_01000_stride1: best_corr =  0.084
full_00001_01000_stride5: best_corr =  0.189
full_40000_41000_stride1: best_corr =  0.029
full_40000_41000_stride5: best_corr = -0.029
```

Interpretation:

```text
ORB affine image rotation is not a reliable direct proxy for drone yaw on Zurich MAV.
```

This does not mean the gyro is bad. It means 2D image-plane motion is a mixed projection effect, especially with an oblique camera and urban 3D structure.

---

## 5. Why Oblique Camera Geometry Causes the Problem

A gyro measures body angular velocity. An accelerometer measures body acceleration. A camera measures projected image motion.

For a clean nadir-looking camera over approximately flat ground:

```text
image x/y motion + altitude + camera intrinsics
≈ ground x/y motion
```

For Zurich MAV, the camera view is oblique and the scene is urban/3D. ORB feature motion contains a mixture of:

```text
- drone translation
- yaw
- roll/pitch
- oblique perspective
- parallax
- buildings/roads/trees at different depths
- local feature layout
- uncertain camera-to-body orientation
```

Therefore:

```text
ORB image dx/dy/rotation ≠ direct drone-body dx/dy/yaw
```

A correct oblique-camera solution would require:

```text
1. camera intrinsics
2. camera-to-body extrinsics
3. drone attitude
4. true AGL/depth or ground/map model
5. flat-ground, DEM, 3D map, stereo depth, or reliable map intersection
```

Zurich gives camera intrinsics and likely usable attitude, but it does not give enough reliable true AGL/depth or camera-to-body/map geometry for stable metric conversion.

---

## 6. Sensor-Assisted ORB Scaling Result

10B tested a separate experimental sensor-assisted scaling sweep. It did not overwrite the original camera-only ORB metric baseline.

### 6.1 With debug height candidate

```text
Run: full_00001_01000_stride5
best_variant: h-height_agl_column__yaw-pose_omega_integrated__noswap__sx-1__sy1
estimated path: 8.07 m
reference path: 19.06 m
RMSE: 2.96 m
final error: 6.02 m
drift: 31.59 m / 100 m
```

```text
Run: full_40000_41000_stride5
best_variant: h-height_agl_column__yaw-pose_omega_integrated__noswap__sx1__sy-1
estimated path: 28.54 m
reference path: 41.04 m
RMSE: 6.81 m
final error: 3.87 m
drift: 9.42 m / 100 m
```

These results improved some windows, but they used `height_agl_column`, which is only an approximate/debug height candidate and not verified true AGL.

Also, the best image-axis signs changed between windows:

```text
early segment:  sx=-1, sy=1
middle segment: sx=1, sy=-1
```

A real physical calibration should not require different sign conventions per flight segment.

### 6.2 Without debug height candidate

```text
Run: full_00001_01000_stride5_no_debug_height
best_variant: h-fixed__yaw-raw_gyro_z_corrected_integrated__noswap__sx-1__sy1
estimated path: 85.28 m
reference path: 19.06 m
RMSE: 9.05 m
final error: 6.93 m
drift: 36.34 m / 100 m
```

```text
Run: full_40000_41000_stride5_no_debug_height
best_variant: h-pose_altitude_relative__yaw-pose_omega_integrated__noswap__sx1__sy-1
estimated path: 76.98 m
reference path: 41.04 m
RMSE: 15.77 m
final error: 33.63 m
drift: 81.94 m / 100 m
```

Without debug height, the scaling becomes unstable and overestimates the path length.

---

## 7. Zurich Phase Closeout Commands / Output Locations

Important generated outputs:

```text
outputs/zurich_mav_full/reports/10a7_camera_motion_vs_gyro_yaw/camera_motion_vs_gyro_yaw_all_runs_summary.json
outputs/zurich_mav_full/reports/10b_sensor_assisted_orb_scaling/sensor_assisted_scaling_all_runs_summary.json
outputs/zurich_mav_full/reports/10b_sensor_assisted_orb_scaling/sensor_assisted_scaling_best_variants_all_runs.csv
```


## Key figures:

![Accelerometer axis mapping](docs/assets/week4_zurich_closeout/accel_best_axis_mapping.png)

![Gyroscope axis mapping](docs/assets/week4_zurich_closeout/gyro_best_axis_mapping.png)

![Early stride-5 sensor-assisted trajectory](docs/assets/week4_zurich_closeout/full_00001_01000_stride5_sensor_assisted_trajectory.png)

![Middle stride-5 sensor-assisted trajectory](docs/assets/week4_zurich_closeout/full_40000_41000_stride5_sensor_assisted_trajectory.png)


## 8. Final Zurich MAV Decision

Zurich MAV should be retained as a diagnostic and baseline dataset, but it should not be continued as the main dataset for final metric-localization accuracy or map-to-camera localization.

Final decision:

```text
Zurich MAV is useful for ORB tracking, synchronization, reference evaluation, and sensor diagnostics.
Zurich MAV is not advisable as the main dataset for final metric localization or absolute frame-to-map localization.
```

Reason:

```text
- ORB tracking works, but metric conversion does not generalize.
- The camera is oblique, not clean nadir.
- The scene has strong 3D urban structure and parallax.
- True AGL/depth is unavailable or not reliable.
- OnboardPose height and azimuth are unusable.
- height_agl_m is approximate/debug only.
- The dataset does not include a clean georeferenced orthophoto/satellite map package for frame-to-map matching.
- Sensor-assisted scaling improves only under debug height/sign sweeps and does not form a stable physical calibration.
```

Report-ready conclusion:

```text
Zurich MAV was successful for dataset loading, synchronization, reference trajectory generation, ORB feature tracking, stride diagnostics, and failure analysis. However, it is not suitable as the main dataset for final metric-localization accuracy. The limitation is not ORB tracking failure; the visual tracking layer works. The limitation is the conversion from oblique image motion to local metric displacement due to missing true AGL/depth, uncertain camera-to-body geometry, and lack of a georeferenced map/orthophoto package. Zurich MAV is therefore frozen as a diagnostic/baseline dataset, and the next main development phase moves to SatLoc for map-based localization.
```

---

## 9. Next Phase — SatLoc Map-Based Localization

The next dataset phase will start fresh with SatLoc.

Current assumption from dataset inspection/download notes:

```text
SatLoc does not provide full IMU/barometer sensor logs for VIO-style fusion.
SatLoc provides drone/UAV frames and georeferenced satellite/orthophoto/map imagery.
```

Therefore, the SatLoc phase should focus on map-based visual localization, especially fast localization within a known map area.

### SatLoc development goal

```text
Drone nadir frame
↓
fast retrieval of candidate map tile / area
↓
local feature matching or homography refinement
↓
candidate GPS/map position
↓
evaluation against available frame coordinate labels / references
```

### SatLoc block plan

```text
S1 — SatLoc dataset inspection and config
S2 — SatLoc loader: UAV frames, satellite tiles, GeoTIFF/map metadata
S3 — Build map tile index and coordinate parser
S4 — Fast area identification / image retrieval baseline
S5 — Local frame-to-map matching refinement
S6 — Candidate GPS/local coordinate output
S7 — Evaluation and visualization
```

### First methods to try on SatLoc

```text
Classical fast retrieval:
- ORB/SIFT/AKAZE descriptors on map tiles
- bag-of-visual-words / descriptor matching
- image hash or global descriptors as a quick coarse filter

Refinement:
- ORB/SIFT matching between UAV frame and candidate tile
- RANSAC homography
- map-coordinate projection from matched tile

Later/stretch:
- SuperPoint + LightGlue
- LoFTR
- RoMa/RoMaV2
- learned aerial image retrieval descriptors
```

---