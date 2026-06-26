# Week 3 Progress — ORB Relative Localization Baseline

## Phase

Week 3 (26th June 2026) — Relative localization baseline.

## Goal

Estimate drone movement from camera frames without using GNSS during localization. GNSS/reference data is used only for evaluation.

The current baseline uses ORB feature matching between consecutive Zurich MAV frames, RANSAC/homography filtering, accumulated image motion, approximate metric scaling using camera calibration and telemetry-derived height/yaw candidates, and comparison against the reference trajectory.

## Current Dataset

Zurich Urban MAV sample dataset.

Current sample size:

```text
MAV frames: 350
Synchronized frames: 350
Reference rows: 81169
```

## Completed Blocks

### Block 07 — ORB Relative Image-Motion Baseline

Implemented files:

```text
src/uavloc/relative/orb_relative_motion.py
scripts/run_orb_relative_motion.py
```

Generated outputs:

```text
outputs/zurich_mav_sample/trajectories/07_orb_relative_motion/orb_relative_trajectory.csv
outputs/zurich_mav_sample/reports/07_orb_relative_motion/orb_relative_motion_summary.json
outputs/zurich_mav_sample/figures/07_orb_relative_motion/orb_relative_xy.png
outputs/zurich_mav_sample/figures/07_orb_relative_motion/orb_reference_comparison_xy.png
```

Successful run summary:

```text
Frames used:         350
Attempted pairs:     349
OK pairs:            349
Failed pairs:        0
Median matches:      1347.0
Median inliers:      1202.0
Median inlier ratio: 0.903
Aligned RMSE [m]:    0.656
Scale [m/px]:        0.011581
Rotation [deg]:      145.88
```

Interpretation:

ORB tracking is strong on the Zurich MAV sample. All consecutive frame pairs were processed successfully. The high match count and high RANSAC inlier ratio show that the sample has sufficient visual overlap for feature-based relative localization.

The raw ORB trajectory is in image-motion pixel units. The similarity-aligned comparison is only a diagnostic shape comparison and does not use GNSS inside the localization algorithm.

### Block 08 — Approximate ORB Metric Scaling

Implemented files:

```text
src/uavloc/relative/orb_metric_scaling.py
scripts/run_orb_metric_scaling.py
```

Initial run used fixed height and fixed heading:

```text
Height: fixed 50 m
Yaw: fixed 0 deg
Estimated path: 19.918 m
Reference path: 6.095 m
RMSE: 7.090 m
Drift per 100 m: 156.99
```

This showed that the metric-scaling infrastructure worked, but the guessed height/yaw values were not physically correct for the sample.

### Block 08B — Zurich Telemetry Enrichment

Implemented files:

```text
src/uavloc/data/enrich_zurich_sync.py
scripts/enrich_zurich_synchronized_frames.py
scripts/inspect_zurich_metric_inputs.py
```

Generated outputs:

```text
outputs/zurich_mav_sample/metadata/synchronized_frames_enriched.csv
outputs/zurich_mav_sample/reports/08b_sync_enrichment/sync_enrichment_summary.json
outputs/zurich_mav_sample/metadata/08c_metric_input_inspection/metric_input_candidate_columns.csv
outputs/zurich_mav_sample/reports/08c_metric_input_inspection/metric_input_inspection_summary.json
```

Final enrichment result:

```text
Height source:   abs_gt_z_minus_gps_z_m
Height quality:  approximate_debug_not_true_agl
Yaw source:      gt_omega_deg
Yaw quality:     groundtruth_orientation_candidate
Height median:   4.051 m
Height range:    2.387 to 4.911 m
Yaw median:      98.349 deg
Yaw range:       95.539 to 100.267 deg
```

Important warnings:

```text
OnboardPose Height is constant zero and unusable.
OnboardPose Azimuth is constant zero and unusable.
Selected height is approximate/debug only, not verified true AGL.
Selected yaw comes from GroundTruthAGL omega_gt because OnboardPose Azimuth is unusable.
```

Final Block 08 metric run using enriched columns:

```text
Height column:       height_agl_m
Yaw column:          yaw_deg
Estimated path:      1.489 m
Reference path:      6.095 m
RMSE:                1.703 m
Mean error:          1.629 m
Max error:           2.193 m
Final error:         1.323 m
Drift per 100 m:     21.71
Shape RMSE:          0.684 m
Shape scale:         2.901
```

## Current Interpretation

The visual tracking part is working well. ORB produces stable frame-to-frame matches and reliable RANSAC inliers.

The remaining limitation is metric scale. The Zurich sample does not provide a reliable true AGL height in the synchronized telemetry. `OnboardPose Height` is all zero, and absolute altitude/barometer altitude are not true AGL. Therefore, `height_agl_m` is currently an approximate/debug source based on `abs(gt_z - z_gps)`.

Yaw improved after rejecting the zero-valued OnboardPose Azimuth and using `gt_omega_deg`.

The final result is a valid Week 3 baseline: relative image-motion trajectory, approximate metric trajectory, reference comparison, error graphs, and documented metadata limitations.

## Current Status

Block 08B is complete and committed.

## Next Planned Step

Test the same pipeline on the full Zurich MAV dataset.

Planned sequence:

```text
1. Unzip full Zurich dataset.
2. Inspect folder structure.
3. Create a new YAML config for the full dataset.
4. Run dataset inspection.
5. Build reference trajectory.
6. Synchronize frames.
7. Run frame quality inspection.
8. Run ORB pair/stride diagnostics on a manageable subset if needed.
9. Run ORB relative motion.
10. Run telemetry enrichment.
11. Run metric scaling.
12. Compare sample vs full dataset behavior.
```

