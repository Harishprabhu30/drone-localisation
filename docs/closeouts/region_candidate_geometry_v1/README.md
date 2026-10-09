# Region-candidate geometry v1 closeout — 2026-10-09

Status: **CLOSED / RESEARCH QUESTION ANSWERED**

Branch:

```text
research/region-candidate-geometry-v1
```

Starting point:

```text
4baa5166c82a1bebfc69f3cd2d12332c06d15bbd
docs/closeouts/trajectory_adapter_v1/README.md
```

## 1. Why RG1 existed

QV1.4 had already improved fixed-budget region-level candidate availability:

```text
center_unique_allview_fill20
768_s256
fixed Top20

contain R20 384 / 403
<=80 R20   380 / 403
<=40 R20   277 / 403
```

RG1 asked:

> Once a stronger region candidate pool exists, should geometry act mainly as
> local candidate validation/refinement rather than as a global rescue reranker?

The answer is now sufficiently clear to stop this branch.

## 2. Frozen inputs and research boundaries

Frozen upstream:

```text
Trajectory Adapter v1
  -> villoc_traj01_90deg_stable120m
  -> QV1.4 center_unique_allview_fill20
  -> 768_s256
  -> fixed Top20
```

Geometry baseline:

```text
CLAHE luma
long side 1024
ORB nfeatures 1800
fastThreshold 7
edgeThreshold 15
patchSize 31
Hamming KNN
Lowe ratio 0.80
RANSAC homography 5 px
```

Excluded throughout RG1:

```text
bootstrap
state
temporal fusion
learned confidence
GT before freeze
silent Top20 changes
```

## 3. RG1.0 — geometry contract preflight

Status:

```text
PASS_RG1_GEOMETRY_CONTRACT_PREFLIGHT
```

Measured:

```text
403 queries
20 candidates/query
8060 query-candidate pairs
310 unique selected tiles

source rows:
  center_square 5401
  right_square  1328
  left_square    937
  resize_square  394

center Top1 preserved
all query images resolved
all selected tile images resolved
reference not loaded
selection disabled
```

## 4. RG1.1 — all-candidate geometry evidence audit

Status:

```text
PASS_RG1_1_ALL_CANDIDATE_GEOMETRY_AUDIT
```

Across all 8060 pairs:

```text
homography_ok              8060 / 8060
median inliers             6
median inlier ratio        0.20
```

The historical >=4-inlier homography flag is therefore not useful as a
confidence signal by itself.

Geometry discrimination:

```text
contains:
  inliers       AUC* 0.8558
  good matches  AUC* 0.8235
  inlier ratio  AUC* 0.7569

<=80:
  inliers       AUC* 0.8611
  good matches  AUC* 0.8278
  inlier ratio  AUC* 0.7665

<=40:
  inliers       AUC* 0.8565
  inlier ratio  AUC* 0.8037
  good matches  AUC* 0.8035
```

For valid containing candidates:

```text
count                         1098
tile-center error median      52.7231 m
projected-point error median   6.4977 m
projection improvement rate    0.7951
```

Result:

> ORB evidence is materially discriminative after candidate generation is
> stronger, and query-center homography projection is highly useful for
> within-region continuous refinement.

## 5. RG1.2 — full-pool geometry gate calibration

Status:

```text
PASS_RG1_2_LOCAL_GEOMETRY_GATE_CALIBRATION_AUDIT
```

Representative <=80 results:

```text
gate                              precision  query retention  median candidates/query
inliers >= 6                     0.1729     0.9553           13
inliers >= 6 + inside            0.1832     0.9500           12
inliers >= 5 + ratio >= 0.20     0.1909     0.8895           10
inliers >= 6 + ratio >= 0.20     0.2521     0.8579            7
+ inside                         0.2714     0.8526             7
```

As gates become stricter, candidate volume decreases, but wrong-only accepted
queries increase and useful-query retention falls.

Decision:

```text
DO NOT promote geometry as a hard Top20 pruning gate.
```

## 6. RG1.3 — retrieval-anchor trust / abstain / refinement

Status:

```text
PASS_RG1_3_RETRIEVAL_ANCHOR_TRUST_REFINEMENT_AUDIT
```

Frozen retrieval anchor:

```text
pool_rank   1
source_view center_square
source_rank 1
```

Always-trust baseline:

```text
contain 174 / 403 = 0.4318
<=80    171 / 403 = 0.4243
<=40    101 / 403 = 0.2506
```

Simple `inliers >= 6`:

```text
contains:
  precision 0.5608
  recall    0.9540
  coverage  0.7345

<=80:
  precision 0.5507
  recall    0.9532
  coverage  0.7345

<=40:
  precision 0.3311
  recall    0.9703
  coverage  0.7345
```

Accepted useful anchors retain approximately 4.5--4.6 m projected-error medians
in the shown Pareto region.

Result:

> Geometry is much better supported as retrieval-anchor trust/abstain +
> continuous refinement than as global candidate selection.

## 7. RG1.4 — provisional simple gate transferred to blind stress

Status:

```text
PASS_RG1_4_PROVISIONAL_GATE_BLIND_STRESS_TRANSFER
```

Predeclared development-only freeze rule:

```text
family: inliers_only
require recall >= 0.90 for contain / <=80 / <=40
choose highest inlier threshold satisfying all
```

Frozen result:

```text
inliers >= 6
```

Development:

```text
accepted fraction 0.734491
recall contain     0.954023
recall <=80        0.953216
recall <=40        0.970297
```

Blind stress:

```text
123 queries
77 accepted
accepted fraction          0.626016
median inliers             6
accepted median inliers    6
accepted inside-tile rate  0.909091
all-successive accepted
projection jump median    72.4855 m
```

No blind GT/reference or accuracy metric was used.

## 8. RG1.5 — temporal transfer diagnosis

Status:

```text
PASS_RG1_5_CROSS_TRAJECTORY_TEMPORAL_BEHAVIOR_AUDIT
```

Development:

```text
accepted fraction                         0.734491
accepted gap median                       1.0 s
accepted run median                       2 queries
all accepted jump median                 25.1883 m
contiguous accepted pairs               251
contiguous jump median                   13.2451 m
contiguous speed median                  13.2451 m/s
contiguous same-tile fraction             0.764940
changed-tile center-jump median          184.6042 m
changed-tile projected-jump median       190.8585 m
```

Blind stress:

```text
accepted fraction                         0.626016
accepted gap median                       1.0 s
accepted run median                       2 queries
all accepted jump median                 72.4855 m
contiguous accepted pairs                53
contiguous jump median                   62.8039 m
contiguous speed median                  62.8039 m/s
contiguous same-tile fraction             0.754717
changed-tile center-jump median          323.8172 m
changed-tile projected-jump median       330.6987 m
```

Map:

```text
stride          51.2 m
diagonal stride 72.4077 m
```

The blind all-accepted jump median is numerically close to one diagonal stride,
but the detailed evidence does not support reducing the problem to simple
adjacent-tile switching:

- same-tile fractions are almost the same on development and blind stress;
- blind contiguous projections jump much farther;
- changed-tile transitions are typically hundreds of metres;
- therefore global retrieval aliases remain a separate upstream failure mode.

## 9. Final RG1 conclusions

### Supported

```text
1. QV1.4 is a strong region-level candidate generator on development.

2. ORB inlier evidence is useful once retrieval has narrowed the problem.

3. ORB query-center homography projection can convert a useful tile observation
   into a much more precise continuous EPSG:3346 map point.

4. Geometry is useful as:
     retrieval-anchor trust / abstain
     +
     continuous sub-tile refinement.

5. A provisional development-derived trust hypothesis is:
     inliers >= 6
   but it is NOT independently accuracy-validated.

6. The no-GT blind flight shows no gross acceptance collapse, but much poorer
   temporal projection continuity.
```

### Not supported

```text
1. homography_ok as a useful confidence gate;
2. geometry as a hard Top20 pruning mechanism;
3. global strongest-ORB-candidate selection;
4. using sub-tile projection to repair distant retrieval aliases;
5. claiming blind-flight geographic accuracy.
```

## 10. Important connection to earlier bootstrap research

The repository already recorded the precursor to the next research question.

The root research README notes that:

```text
correct consecutive observations can map to the same tile center,
reducing motion information available to bootstrap geometry,

while incorrect neighboring tile selections can create artificial
spatial diversity.
```

It explicitly motivated:

```text
stronger sub-tile localization
candidate-region geometry
multi-frame map evidence
```

Related historical artifacts:

```text
docs/demo/blind_recorded_flight_final_001/README.md
configs/bootstrap/minimum_confident_v2/architecture_contract.json
scripts/villoc/research/minimum_confident_bootstrap/diagnostics/
  r4_11_blind_subtile_projection_recompute.py
  r4_18_acquisition_to_tracking_counterfactual.py
```

The architecture contract also records:

```text
tile_center_as_primary_measurement = false
minimum_map_span_m = 50
maturity support from consecutive compatible sub-tile innovations
```

Therefore the next locality/sub-tile stage is a continuation of the prior
bootstrap research thread, not a new unrelated direction.

## 11. Why stride is NOT changed yet

Current map:

```text
tile footprint approx 153.6 m
stride approx          51.2 m
overlap per axis approx 66.7%
```

A physical UAV location can legitimately be represented by several overlapping
tiles.

However, changing stride now would confound:

```text
local-neighborhood representation
multi-tile geometry consensus
map sampling density
candidate redundancy
```

The next stage must first test locality / overlapping-tile consensus with the
current map frozen.

Stride ablation is deferred until after the locality representation is
understood.

## 12. Next stage

Next research question:

> Can overlapping local map tiles that describe the same physical locality be
> converted into a tile-boundary-invariant continuous position hypothesis, and
> can repeated consecutive observations be represented as support for one
> locality rather than falsely independent tile-center observations?

Proposed next branch:

```text
research/local-map-neighborhood-subtile-consensus-v1
```

Proposed stage family:

```text
RG2 — Local Map Neighborhood / Sub-Tile Consensus
```

Start from this RG1 closeout.

Do not change stride in RG2.0.
Do not modify QV1.4.
Do not reintroduce bootstrap/state yet.
