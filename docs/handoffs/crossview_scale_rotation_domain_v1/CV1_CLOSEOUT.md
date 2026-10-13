# CV1 closeout — causal scale / rotation study

Status: **PASS / CLOSED**

Branch:
`research/crossview-scale-rotation-domain-v1`

## Decision

CV1 is closed with a positive fixed-scale finding and a negative coarse-rotation
finding.

The promoted development control for subsequent cross-view frontend studies is:

`ORB + fixed query scale_m2 (0.6944444444444444)`

This is a research control, not yet a deployment rule or calibrated GSD
normalization.

## Frozen evidence

### CV1.1 uniform40 reference-free freeze

Evidence SHA256:
`4abfacb77cf9d1d4b09ea917e048b0f2191659907dc4391f7d8fc3aadf020666`

Baseline parity:
`PASS_CV1_1_BASELINE_PARITY`

### CV1.2 uniform40 result

Baseline:
- Top1 containment 28/40
- Top1 <=40 m 13/40
- Top1 <=80 m 28/40
- projection <=40 m 29/40
- projection <=80 m 29/40
- projection >200 m 10/40

scale_m2:
- Top1 containment 32/40
- Top1 <=40 m 15/40
- Top1 <=80 m 31/40
- projection <=40 m 32/40
- projection <=80 m 34/40
- projection >200 m 6/40

scale_m1 also passed the continuation gate, but the predeclared promotion rule
selected scale_m2.  No scale retuning was performed on full403.

Quarter-turn rotations did not materially change locality metrics or the
catastrophic tail on uniform40.

### CV1.3 full403 reference-free scale_m2 freeze

Evidence SHA256:
`94353f763725f44f278192bcedcfd139ab136ecebf4f1dcc6e07487380964810`

Reference loaded: false.
SRT loaded: false.
Baseline regenerated: false.

### CV1.4 full403 post-freeze validation

Baseline -> scale_m2:
- Top1 containment: 292 -> 305
- Top1 <=40 m: 150 -> 163
- Top1 <=80 m: 287 -> 293
- selected projection median: 6.092 -> 5.840 m
- RMSE: 323.219 -> 282.573 m
- selected projection <=40 m: 293 -> 319
- selected projection <=80 m: 305 -> 331
- selected projection >200 m: 88 -> 63
- median Top1-Top2 inlier margin: 2 -> 3

Winner changed:
`170 / 403`

Rescue / damage:
- containment 37 / 24
- center <=80 m 35 / 29
- catastrophic projection 37 / 12

Replication gate:
`PASS`

### CV1.5 mechanism / camera audit

Status:
`PASS_CV1_5_SCALE_MECHANISM_CAMERA_GSD_AUDIT`

Catastrophic rescue pathway:
- rescues total: 37
- rescues with winner changed: 36
- rescues with same winner: 1
- damages total: 12
- damages with winner changed: 12
- damages with same winner: 0

Therefore the catastrophic-tail gain is overwhelmingly a **candidate/locality
selection effect**, not a same-selected-tile projection correction.

Median ORB keypoint count:
- baseline 1800
- scale_m2 1800

The detector remains at its configured feature cap, so the gain is not explained
by a simple reduction in the number of retained ORB keypoints.

Camera metadata on 403 aligned samples:
- focal_len_mm: constant 24.0
- dzoom_ratio: constant 1.0
- rel_alt_m: 118.775 .. 120.122 m, median 119.989 m
- gimbal pitch: -90.0 / -89.9 deg

The fixed scale factor would imply, **only if it were exactly a physical
map-GSD match**:
- baseline processed query GSD ~0.138889 m/px
- processed fx ~864 px at H=120 m
- horizontal FOV ~61.30 deg for width 1024

These values are diagnostic implications, not calibration.

Physical-GSD claim remains blocked by:
- camera model not identified in repository;
- calibrated intrinsics unavailable;
- SRT focal-length semantics unverified.

GPS was not emitted into the sanitized camera artifact.

## Scientific conclusions

1. **Scale handling matters.**
   A fixed, predeclared query downsampling intervention robustly improves ORB
   candidate discrimination on uniform40 and full403.

2. **The benefit is primarily locality selection.**
   36/37 catastrophic rescues change the selected candidate.

3. **Coarse quarter-turn orientation is not the dominant problem.**
   Fixed 90/180/270-degree rotations did not materially improve the screen.
   This does not rule out arbitrary heading canonicalization.

4. **Physical GSD is unresolved.**
   The available SRT fields are stable enough to be useful metadata, but the
   repository lacks the calibrated camera model needed to call scale_m2 a
   physical GSD correction.

5. **No more scale sweep on this development trajectory.**
   Do not tune a better factor on full403.  Treat scale_m2 as a frozen research
   control until validation on another trajectory/dataset exists.

## Next stage

CV2:
SIFT + LightGlue diagnostic on the same frozen QV1.4 uniform40 Top20 screen.

Purpose:
test whether a local frontend with explicit SIFT scale/orientation estimates and
the official SIFT-specific LightGlue model can approach or exceed the stronger
scale_m2 ORB control.

Do not apply scale_m2 to SIFT in the first CV2 screen; that would confound the
explicit-scale/orientation hypothesis with the CV1 preprocessing intervention.
