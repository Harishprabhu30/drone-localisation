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

```bash
export PYTHONPATH=$PWD/src
python scripts/run_orb_relative_motion.py --config configs/dataset_zurich.yaml
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

```bash
export PYTHONPATH=$PWD/src
python scripts/run_orb_metric_scaling.py --config configs/dataset_zurich.yaml
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

```bash
export PYTHONPATH=$PWD/src
python scripts/inspect_zurich_metric_inputs.py --config configs/dataset_zurich.yaml
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

Update 26th June 2026 : 

## Block 09A — Full Dataset ORB Stride Subset Diagnostics

### Goal

Before running ORB relative motion on the full Zurich MAV dataset, a stride diagnostic test was performed to understand how much frame skipping can be used without losing reliable feature matching.

The purpose was to evaluate whether larger frame strides can speed up future trajectory estimation while maintaining enough ORB feature matches and RANSAC inliers.

### Dataset

Zurich MAV full dataset.

```text
Total synchronized frames: 81169
Dataset config: configs/dataset_zurich_full.yaml
```

### Implemented Script

```text
scripts/run_orb_stride_subset_diagnostics.py
```

The script supports selected full-dataset windows using:

```text
--start-imgid
--end-imgid
--strides
--run-name
```

For each selected segment, it evaluates ORB matching quality for multiple frame strides and saves pair-level diagnostics, summary CSV/JSON files, and quality plots.

### Tested Segments

Five representative full-dataset segments were tested:

```text
full_00001_01000
full_20000_21000
full_40000_41000
full_60000_61000
full_80000_81169
```

Each segment was tested with:

```text
strides = 1, 2, 3, 5, 10
```

### Results Summary

All tested segments showed strong ORB matching performance.

Across all tested windows:

```text
stride 1  → ok_ratio 1.000, strongest inlier ratio
stride 2  → ok_ratio 1.000, still very strong
stride 3  → ok_ratio 1.000, strong
stride 5  → ok_ratio 1.000, strong fast-mode candidate
stride 10 → ok_ratio 1.000, usable but lower inlier ratio
```

Median inlier ratio range across tested segments:

```text
stride 1:  0.982 to 0.992
stride 2:  0.960 to 0.985
stride 3:  0.936 to 0.974
stride 5:  0.901 to 0.944
stride 10: 0.828 to 0.916
```

This shows that ORB feature matching is robust across the full Zurich MAV dataset, not only on the small sample subset.

### Runtime Observation

Each approximately 1000-frame segment with five tested strides required several minutes to process.

Example runtimes:

```text
full_00001_01000: 471.02 s
full_20000_21000: 473.35 s
full_40000_41000: 454.24 s
full_60000_61000: 462.35 s
full_80000_81169: 773.86 s
```

This confirms that running all strides across all 81169 frames would be computationally heavy. Future full-dataset experiments should use selected strides and run-specific output folders.

### Interpretation

The stride test confirms that ORB tracking is stable across early, middle, late, and end sections of the full Zurich MAV sequence.

The main conclusion is:

```text
stride 1  = safest and most accurate baseline
stride 5  = practical fast-mode candidate
stride 10 = possible coarse-mode experiment, but not first baseline
```

The decrease in inlier ratio with increasing stride is expected because larger stride means larger viewpoint/motion difference between compared frames.

### Important Note on Reference Distance

For small strides, the median reference displacement was often close to `0.000 m`. This does not mean the drone was not moving. It means that GNSS/reference displacement between very close image frames can be too small or quantized for frame-to-frame interpretation.

Therefore, reference evaluation should focus on accumulated trajectory error over longer windows instead of individual tiny frame-pair displacements.

### Decision

Block 09A is successful.

The next step should be Block 09B:

```text
ORB Relative Motion Subset Runs
```

In Block 09B, actual accumulated ORB trajectories should be generated on selected full-dataset windows using controlled frame ranges and run names.

Planned first tests:

```text
full_00001_01000_stride1
full_00001_01000_stride5
full_40000_41000_stride1
full_40000_41000_stride5
```

The goal is to compare actual accumulated trajectory drift, not only pairwise feature-matching quality.
