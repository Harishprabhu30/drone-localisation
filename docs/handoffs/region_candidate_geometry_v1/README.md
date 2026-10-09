# Region-candidate geometry v1 — handoff

Status: **RG1.0/RG1.1/RG1.2/RG1.3 CLOSED / RG1.4 IMPLEMENTED — VALIDATION READY**

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


## RG1.1 result

Status:

```text
PASS_RG1_1_ALL_CANDIDATE_GEOMETRY_AUDIT
```

Measured across all 8060 frozen candidate pairs:

```text
homography success          8060 / 8060
median inliers              6
median inlier ratio         0.20
```

Discrimination:

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

The historical binary homography flag is not useful as confidence here because
every candidate pair passes its current >=4-inlier definition.

For valid containing candidates:

```text
count                         1098
tile-center error median      52.7231 m
projected-point error median   6.4977 m
projection improvement rate    0.7951
```

Interpretation:

ORB evidence is materially discriminative once candidate generation is stronger,
and the query-center homography projection has strong within-region refinement
value on true containing candidates.

This still does not justify selecting the globally strongest ORB candidate.

## RG1.2 — local geometry gate calibration audit

RG1.2 reuses the frozen RG1.1 evidence and does no new feature matching.

It sweeps simple blind-safe gate families using only:

```text
inlier count
inlier ratio
projected-inside-tile
```

Families:

```text
inliers only
inliers + ratio
inliers + inside
inliers + ratio + inside
```

For each gate, measure:

```text
accepted candidate fraction
accepted candidates/query
candidate precision
candidate recall
positive-query retention
wrong-only accepted queries
conditional projected-error median
conditional projection-improvement rate
```

for:

```text
containment
<=80 m
<=40 m
```

A Pareto frontier is reported over:

```text
higher candidate precision
higher positive-query retention
lower accepted candidate fraction
```

No gate is promoted by RG1.2.

Reason:

```text
the same development trajectory is being used for calibration,
and no second labeled validation trajectory is available
```

### RG1.2 command

```bash
source .drone_venv/bin/activate
export PYTHONPATH=$PWD/src

python -m unittest discover -s tests -v

python scripts/villoc/geometry/rg1_2_local_geometry_gate_calibration.py
```

Expected:

```text
PASS_RG1_2_LOCAL_GEOMETRY_GATE_CALIBRATION_AUDIT
```

Send the printed Pareto frontiers. They determine whether a simple geometry
gate is worth freezing provisionally for no-GT stress testing.


## RG1.2 result

Status:

```text
PASS_RG1_2_LOCAL_GEOMETRY_GATE_CALIBRATION_AUDIT
```

The all-candidate gate does not justify promotion.

Representative <=80 tradeoffs:

```text
gate                              precision  query retention  candidates/query
inliers >= 6                     0.1729     0.9553           13
inliers >= 6 + inside            0.1832     0.9500           12
inliers >= 5 + ratio >= 0.20     0.1909     0.8895           10
inliers >= 6 + ratio >= 0.20     0.2521     0.8579            7
+ inside                         0.2714     0.8526            7
```

Stricter gates reduce candidate volume, but the surviving set is still too
impure to act as a reliable hard Top20-pruning stage, and wrong-only accepted
queries increase.

Conclusion:

```text
DO NOT promote an RG1.2 all-candidate geometry gate.
```

The evidence instead supports testing geometry as selective confidence and
continuous refinement for an already-chosen retrieval anchor.

## RG1.3 — retrieval-anchor geometry trust/refinement audit

RG1.3 uses only the frozen candidate-pool rank-1 row:

```text
pool_rank 1
source_view center_square
source_rank 1
403 queries
```

Geometry may:

```text
accept anchor for continuous projection
or
abstain
```

Geometry may NOT:

```text
switch to another Top20 candidate
rerank the pool
invoke bootstrap/state/temporal logic
```

This directly tests the architectural role:

```text
retrieval chooses region
geometry asks whether that region is trustworthy
geometry refines to a continuous map point when trusted
```

RG1.3 sweeps the same blind-safe inlier/ratio/inside evidence families and
reports selective precision, recall, coverage and projected-error quality for
containment, <=80 m and <=40 m.

No threshold is promoted in RG1.3 because it is still development-trajectory
calibration.

### RG1.3 command

```bash
source .drone_venv/bin/activate
export PYTHONPATH=$PWD/src

python -m unittest discover -s tests -v

python scripts/villoc/geometry/rg1_3_retrieval_anchor_trust_refinement.py
```

Expected:

```text
PASS_RG1_3_RETRIEVAL_ANCHOR_TRUST_REFINEMENT_AUDIT
```

Send the three selective-trust Pareto frontiers. The key question is whether
geometry can substantially improve anchor precision at useful coverage while
retaining the strong ~metre-scale projection behavior seen in RG1.1.


## RG1.3 result

Status:

```text
PASS_RG1_3_RETRIEVAL_ANCHOR_TRUST_REFINEMENT_AUDIT
```

The retrieval anchor is the frozen center-square Top1.

Baseline always-trust:

```text
contain 174 / 403 = 0.4318
<=80    171 / 403 = 0.4243
<=40    101 / 403 = 0.2506
```

Simple inlier evidence materially improves selective trust.

Representative gate:

```text
inliers >= 6

contain:
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

Accepted useful anchors retain strong sub-tile refinement, with projected
medians around 4.5--4.6 m in the shown frontier.

Interpretation:

```text
geometry is materially more convincing as:
  trust / abstain on the retrieval anchor
  + continuous refinement

than as:
  hard pruning of the full Top20
  or global candidate selection
```

## RG1.4 — provisional gate freeze + blind-stress transfer

RG1.4 freezes a simple development-derived rule before touching the blind
trajectory.

Predeclared freeze rule:

```text
family:
  inliers_only

constraints:
  contains recall >= 0.90
  <=80 recall       >= 0.90
  <=40 recall       >= 0.90

selection:
  choose the highest inlier threshold satisfying all three
```

The threshold is selected exclusively from the RG1.3 development table and
written to a freeze artifact before blind-stress geometry is evaluated.

The 123-query blind flight is then evaluated with:

```text
same frozen retrieval anchor
same ORB geometry contract
same frozen inlier threshold
no threshold retuning
no GT
no accuracy metrics
no candidate switching
```

Blind-stress outputs are behavioral only:

```text
acceptance fraction
inlier distributions
inside-tile projection rate
reprojection RMSE
accepted projected-point jump statistics
```

### RG1.4 command

```bash
source .drone_venv/bin/activate
export PYTHONPATH=$PWD/src

python -m unittest discover -s tests -v

python scripts/villoc/geometry/rg1_4_provisional_gate_blind_stress_transfer.py
```

Expected:

```text
PASS_RG1_4_PROVISIONAL_GATE_BLIND_STRESS_TRANSFER
```

Do not interpret blind acceptance as accuracy. RG1.4 only asks whether the
development-frozen trust mechanism behaves coherently on the second flight.
