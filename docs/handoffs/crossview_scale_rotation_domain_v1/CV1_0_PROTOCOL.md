# CV1.0 protocol freeze — cross-view scale / rotation diagnostics

Status: **FROZEN PROTOCOL / IMPLEMENTATION MAY BEGIN**

Branch:
`research/crossview-scale-rotation-domain-v1`

Protocol parent:
`bd576840f5fb9ddfae6f9c7d63351708b87ccae0`

## 1. Research question

CV1 asks:

> Under the frozen QV1.4 Top20 candidate pool, how much of ORB's candidate-discrimination behaviour can be causally attributed to query/map scale mismatch and relative image-plane orientation, and where do those effects interact with already-observed VILLOC visual-content regimes?

This is still the existing absolute-localization problem. Retrieval, state estimation, temporal fusion and deployment architecture are not changed in CV1.

## 2. Frozen upstream

Development trajectory:
`villoc_traj01_90deg_stable120m`

Frozen retrieval:
`qv1_4_center_unique_allview_fill20`

Map variant:
`768_s256`

Map geometry:
- CRS: EPSG:3346
- orthophoto source GSD: approximately 0.20 m/px
- tile footprint: approximately 153.6 m
- stride: approximately 51.2 m

Frozen development Top20:
`outputs/research_runs/local_map_neighborhood_subtile_consensus_v1/rg2_1a/development/rg2_1a_frozen_qv_top20.csv`

Expected SHA256:
`0a7832c6f0997a33851343fd7df537d2f6410211ac0e38e7ec4325899d8df47e`

Frozen ORB geometry contract:
`src/uavloc/geometry/orb_subtile.py`

Frozen richer geometry layer:
`src/uavloc/geometry/top20_geometric_evidence.py`

GD1.0 development evidence:
`outputs/research_runs/top20_geometric_discrimination_v1/gd1_0/development/gd1_0_top20_geometric_evidence.csv`

Expected GD1.0 development evidence SHA256:
`91ba3381e463bd083f9355b517dbb93953e4423a4b366638c1a97419ea0aea3f`

## 3. Exact screening subset

Use the already-frozen deterministic uniform40 IDs:

`1,11,22,32,42,53,63,73,83,94,104,114,125,135,145,156,166,176,187,197,207,217,228,238,248,259,269,279,290,300,310,321,331,341,351,362,372,382,393,403`

This gives:

`40 queries x 20 candidates = 800 candidate pairs per transform condition`

Do not create a new hand-picked hard/easy subset for CV1.

## 4. Historical VILLOC evidence that CV1 must not repeat

### 4.1 S8.12C.0 was context-scale, not UAV-GSD normalization

Historical `feature/s8` work tested center crops of the same satellite candidate at 100%, 75%, 60%, 45% and 30%.

Conclusion:
- scale/context hypothesis partly supported;
- no single center-only crop generalized over Q23/Q33/Q52.

That experiment changed **map context/FOV inside a candidate tile**. It did not resample the UAV query to a physically matched UAV/map ground sampling distance.

Therefore CV1 scale resampling is a different experiment.

### 4.2 S8.12C.1 was directional spatial crop search, not image rotation

Historical `feature/s8` work tested:
- scales 60% and 45%;
- crop positions north/south/east/west/centre;
- 600 LightGlue pairs for Q23/Q33/Q52.

It showed that extra spatial hypotheses can reveal the oracle but also create stronger repeated-structure false maxima.

CV1 must not recreate a larger crop bank.

### 4.3 S8.13A heading rotation was planned, not closed

`docs/README_s8_13A_heading_rotation_and_temporal_plan.md` proposed a heading-aligned dynamic map canvas, but the repository contains a planning document rather than a completed S8.13A rotation result.

Important unresolved questions in that plan included:
- whether gimbal yaw follows aircraft heading;
- which image edge corresponds to yaw=0;
- whether yaw is true-north referenced.

Therefore a controlled rotation diagnostic remains scientifically open.

### 4.4 Existing traj01 visual-factor evidence

The curated traj01 factor story already found, for DINO Top1 error:
- yaw: weak relationship, Spearman approximately +0.023;
- relative altitude: weak relationship, approximately -0.067;
- sharpness/Laplacian: moderate relationship, approximately +0.457;
- brightness: moderate relationship, approximately -0.417;
- contrast: weak relationship, approximately +0.104;
- edge density: moderate relationship, approximately +0.445.

Interpretation already frozen:
sharpness and edge density may represent scene complexity/repetition rather than simply image quality.

CV1 therefore does **not** restart a broad visual-quality correlation study. Later post-freeze CV1 analysis may stratify *transform benefit/damage* by these existing variables.

## 5. Blind/reference boundary

### 5.1 Allowed in CV1.1 reference-free evidence

- UAV query pixels from the canonical blind manifest;
- frozen QV1.4 Top20 candidate identities/ranks;
- map tile pixels and metric tile geometry;
- frozen ORB settings;
- fixed predeclared scale factors;
- fixed predeclared rotations;
- image-only geometric evidence;
- query/tile identifiers and provenance.

### 5.2 Forbidden until the reference-free evidence is frozen

- latitude / longitude;
- reference ENU coordinates;
- oracle tile IDs;
- containment labels;
- <=40 m / <=80 m labels;
- candidate geographic error;
- selected geographic error;
- RMSE/p95/final error;
- SRT-derived yaw/altitude/camera fields for choosing a transform;
- any best-transform-per-query or best-transform-per-candidate selection.

### 5.3 Current blind research contract

`configs/trajectories/villoc_traj01_90deg_stable120m_v1.yaml` currently exposes:
- assumed relative altitude = 120.0 m;
- assumed gimbal pitch = -90 deg;
- body orientation = unavailable;
- IMU = unavailable.

Therefore SRT `gb_yaw_deg` is not currently a deployable blind input. It may be attached after a reference-free freeze for explanatory analysis only.

## 6. SRT camera metadata decision for physical GSD

The VILLOC SRT parser preserves:
- `focal_len_mm`;
- `dzoom_ratio`;
- `fnum`;
- `iso`;
- `shutter`;
- `ev`;
- `rel_alt_m`;
- `abs_alt_m`;
- gimbal yaw/pitch/roll.

### 6.1 Geometrically useful candidates

Potential GSD inputs:
- relative altitude / known AGL height;
- focal length or calibrated focal length in pixels;
- digital zoom/FOV state;
- camera pitch/roll when not exactly nadir;
- static camera intrinsics or verified sensor geometry.

### 6.2 Not GSD inputs

Do not use these to calculate GSD:
- f-number: aperture, not field of view;
- ISO: exposure/noise;
- shutter: exposure/motion-blur risk;
- EV: exposure;
- absolute altitude: not camera height above local ground by itself.

These may later be appearance/quality covariates only.

### 6.3 Critical limitation

`focal_len_mm + rel_alt_m + dzoom_ratio` are **not sufficient by themselves** for a trustworthy absolute GSD unless the focal-length semantics and camera projection are verified.

A physical GSD claim additionally needs one of:
- calibrated `fx/fy` at the relevant image resolution, or
- verified physical focal length + sensor dimensions + digital-zoom semantics, or
- a verified horizontal/vertical FOV model.

Do not assume that the SRT focal length is physical rather than 35-mm-equivalent without validation.

### 6.4 CV1 consequence

Physical-GSD normalization is conditional.

CV1.1 may proceed immediately with fixed controlled scale perturbations.

A later CV1 camera-metadata audit may promote a physically justified scale factor only if:
1. camera-model/intrinsic semantics are verified;
2. zoom semantics are verified or zoom is constant/non-active;
3. the resulting metadata can be represented as a legitimate static/onboard deployment input rather than hidden GT.

For the stable120m trajectory, the blind contract's fixed 120 m altitude is preferred over per-frame SRT altitude for the first deployable-style calculation. SRT altitude may later validate that approximation post-freeze.

## 7. CV1-S1 fixed scale-sensitivity conditions

Apply scale to the **already frozen ORB-preprocessed query image**, not to the raw image before the frozen long-edge downsize.

Reason:
resizing the raw image first and then applying the frozen 1024-long-edge downsize could cancel the intended perturbation.

Preserve query FOV by resizing the processed query canvas itself; do not center-crop/pad merely to force the original dimensions.

Conditions are tied to ORB's approximately 1.2 scale pyramid:

- `scale_m2 = 1 / 1.2^2 = 0.6944444444`
- `scale_m1 = 1 / 1.2   = 0.8333333333`
- `baseline = 1.0`
- `scale_p1 = 1.2`
- `scale_p2 = 1.2^2 = 1.44`

Run each condition independently.

Forbidden:
- best scale per query;
- best scale per candidate;
- combining scale conditions before truth attachment.

## 8. CV1-R1 fixed rotation-sensitivity conditions

Reference-free conditions:
- baseline 0 degrees;
- 90 degrees;
- 180 degrees;
- 270 degrees.

Use exact quarter-turn image rotation so full query content is preserved and no interpolation is introduced.

Run each orientation independently.

Forbidden:
- best rotation per query;
- best rotation per candidate;
- using SRT yaw to choose a rotation during the reference-free stage.

The fixed rotations are diagnostics, not a proposed deployment oracle.

## 9. Baseline parity gate

The `baseline` condition must reproduce the frozen GD1.0 uniform40 evidence for the unchanged ORB contract.

At minimum require exact equality for discrete evidence:
- good_matches;
- inliers;
- homography_ok.

Require numerical equality within deterministic floating tolerance for:
- projected tile pixels;
- projected easting/northing;
- inlier ratio;
- coverage/residual/Jacobian evidence where finite.

If baseline parity fails, stop CV1.1 before interpreting transformed conditions.

## 10. Reference-free CV1.1 output contract

Freeze one deterministic pair-evidence table containing:
- query ID;
- pool rank;
- tile ID;
- source view/rank;
- transform family;
- transform ID;
- scale factor or rotation degrees;
- processed query dimensions;
- all frozen/richer ORB geometric evidence;
- continuous projected map coordinate when finite.

The table must not contain truth/evaluation labels.

Freeze:
- evidence CSV SHA256;
- report JSON SHA256/provenance;
- input Top20 SHA256;
- baseline GD1.0 SHA256;
- ORB module SHA256;
- richer-evidence module SHA256;
- transform module SHA256.

## 11. Post-freeze evaluation metrics

Only after the CV1.1 evidence is frozen:

Locality discrimination:
- Top1 containment;
- Top1 center <=40 m;
- Top1 center <=80 m;
- first useful candidate rank;
- rescue / damage counts versus baseline;
- Top1-Top2 raw-inlier margin.

Continuous projection:
- selected projection median;
- RMSE;
- p95;
- <=40 m;
- <=80 m;
- >200 m catastrophic count;
- oracle-finite projection metrics.

Primary selector stays:
`raw ORB inliers descending -> pool_rank ascending -> tile_id ascending`.

Do not introduce another score formula in CV1.

## 12. Uniform40 continuation gates

A transform condition justifies an unchanged full403 run if it achieves at least one screening improvement versus frozen ORB:
- +4/40 Top1 containment; or
- +4/40 Top1 <=80 m; or
- at least 3/40 fewer selected-projection errors >200 m while losing no more than 2/40 baseline <=40 m projections.

These are resource/continuation gates, not claims of statistical significance.

No retuning is allowed between uniform40 and full403.

## 13. CV1 content-regime use

Do not repeat the old broad factor analysis.

After truth attachment, optional interaction analysis may ask whether **benefit or damage caused by a frozen transform** varies with:
- existing sharpness/Laplacian;
- brightness;
- contrast;
- edge density;
- retrieval rank;
- post-freeze SRT yaw/altitude where explicitly labeled development-only.

No content variable may choose a transform in CV1.

## 14. CV1 stop conditions

Stop CV1 and move to the next predeclared family if:
- baseline parity fails and cannot be explained without changing the frozen ORB contract;
- no fixed scale/rotation condition passes the uniform40 continuation gate;
- physical GSD cannot be justified from a valid camera model and the fixed scale diagnostic shows no material effect.

If scale/rotation evidence is useful, freeze the winning diagnostic condition before any full403 run.

Do not start SIFT, AerialMegaDepth/RDD, LoFTR/RoMa, triangulation, networking, or state redesign inside CV1.0/CV1.1.
