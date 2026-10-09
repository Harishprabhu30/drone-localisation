# Region-candidate geometry v1 — handoff

Status: **RG1.0 IMPLEMENTED — VALIDATION READY**

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
