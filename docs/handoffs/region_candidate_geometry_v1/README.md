# Region-candidate geometry v1 — handoff

Status: **RG1.0 CLOSED / RG1.1 IMPLEMENTED — VALIDATION READY**

Branch:

```text
research/region-candidate-geometry-v1
```

Starting point:

```text
4baa5166c82a1bebfc69f3cd2d12332c06d15bbd
docs/closeouts/trajectory_adapter_v1/README.md
```

## Research question

QV1.4 now supplies a fixed Top20 candidate pool with strong region availability
on the development trajectory.

The new question is:

> Can geometry operate mainly as local candidate validation and sub-tile
> refinement, instead of acting as a global rescue reranker?

## Frozen upstream input

```text
trajectory adapter v1
  -> villoc_traj01_90deg_stable120m
  -> QV1.4 center_unique_allview_fill20
  -> 768_s256
  -> fixed Top20
```

The candidate pool is immutable inside RG1.

Development reference:

```text
contain R20 384 / 403
<=80 R20   380 / 403
<=40 R20   277 / 403
```

## Geometry baseline

RG1 begins with the already-reproducible ORB geometry frontend:

```text
preprocess:       CLAHE luma
resize long side: 1024
ORB nfeatures:    1800
fastThreshold:    7
edgeThreshold:    15
patchSize:        31
BFMatcher:        Hamming KNN k=2
Lowe ratio:       0.80
homography:       RANSAC
RANSAC threshold: 5 px
minimum matches:  4
minimum inliers:  4
```

This is a baseline evidence generator, not a promoted selector.

## Critical separation from historical ORB usage

Historical work used ORB as a Top-K reranker, often with a DINO-rank prior.

RG1 initially forbids that.

```text
allowed:
  ORB match evidence
  homography validity
  inlier statistics
  inlier spatial coverage
  query-center projection into a candidate tile
  continuous EPSG:3346 projection

not allowed in RG1.0/RG1.1:
  hybrid DINO+ORB reranking
  rank-prior geometry score
  bootstrap
  state acceptance
  temporal fusion
```

## RG1.0 — contract preflight

RG1.0 verifies before any geometry compute:

- 403 canonical development queries;
- exactly 20 frozen candidates/query;
- query-ID parity with the canonical trajectory manifest;
- center-square rank-1 authority preserved;
- only expected query-view sources;
- every selected tile exists in the 768_s256 index;
- every query image exists;
- every selected map-tile image exists;
- no reference/evaluation columns in the blind inputs;
- no reference is loaded.

Run:

```bash
source .drone_venv/bin/activate
export PYTHONPATH=$PWD/src

python -m unittest discover -s tests -v

python scripts/villoc/geometry/rg1_0_contract_preflight.py
```

Expected:

```text
PASS_RG1_GEOMETRY_CONTRACT_PREFLIGHT
```

## RG1.1 — planned first geometry experiment

After RG1.0 passes:

```text
for every query:
  for every frozen Top20 candidate:
    compute ORB evidence
    compute homography if valid
    project processed query center into tile
    convert to EPSG:3346
```

All 403 x 20 candidate pairs are processed.

The blind evidence table is written and hashed before reference is attached.

No candidate is selected by geometry in RG1.1.

Post-freeze only, evaluate whether:

- inliers / ratio / coverage discriminate containing or <=80 candidates;
- valid homographies occur more often on useful candidates than aliases;
- local projected positions improve upon tile-center precision;
- geometry failure modes differ by QV candidate source;
- named q57/q99/q228/q390 cases agree with the full-dataset pattern.

The blind-stress trajectory remains withheld from algorithm tuning until the
development geometry audit is frozen.


## RG1.0 result

Status:

```text
PASS_RG1_GEOMETRY_CONTRACT_PREFLIGHT
```

Measured:

```text
trajectory:          villoc_traj01_90deg_stable120m
queries:             403
candidate rows:      8060
candidate budget:    20
unique selected tiles: 310

source counts:
  center_square  5401
  right_square   1328
  left_square     937
  resize_square   394

center Top1 preserved: true
query images resolved: 403
tile images resolved:  310
reference loaded:      false
selection enabled:     false
```

## RG1.1 — all-candidate geometry evidence audit

RG1.1 processes every one of the 8060 frozen candidate pairs.

Blind phase:

```text
query image -> exact historical ORB frontend
candidate tile -> exact historical ORB frontend
query->tile homography
processed query-center projection
tile-pixel -> EPSG:3346 conversion
freeze/hash
```

Blind outputs include:

```text
good matches
inliers
inlier ratio
query inlier coverage
tile inlier coverage
homography validity
reprojection RMSE
projected tile pixel
projected EPSG:3346 point
projected-inside-tile flag
```

There is deliberately no:

```text
geometry rank
geometry-selected tile
hybrid score
DINO rank prior
bootstrap
state
temporal fusion
```

Only after the blind geometry table is written and hashed is the prepared
development reference attachment read.

Post-freeze diagnostics measure:

- geometry-feature AUC* for containment, <=80 m and <=40 m candidate classes;
- valid projection rates for useful versus false candidates;
- projected-point error versus tile-center error;
- projection improvement rate;
- behavior by candidate source view;
- q57/q99/q228/q390 diagnostics as examples only.

### RG1.1 command

```bash
source .drone_venv/bin/activate
export PYTHONPATH=$PWD/src

python -m unittest discover -s tests -v

python scripts/villoc/geometry/rg1_1_all_candidate_geometry_audit.py
```

Expected:

```text
PASS_RG1_1_ALL_CANDIDATE_GEOMETRY_AUDIT
```

This is a heavier run: 8060 ORB matching/homography candidate pairs are
processed, but feature extraction is cached across 403 query images and 310
unique tile images.

Do not design RG1.2 from named cases alone. The aggregate discrimination and
projection diagnostics are the gate.
