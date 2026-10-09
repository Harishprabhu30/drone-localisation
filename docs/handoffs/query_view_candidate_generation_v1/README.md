# Handoff — query-view candidate generation v1

Status: **QV1.0–QV1.5 CLOSED — BRANCH FROZEN**

Branch:

```text
research/query-view-candidate-generation-v1
```

Starting point:

```text
8fc7424be01ccde122c15befa701fb66652880d4
docs/closeouts/absolute_frontend_decoupling_v1/README.md
```

## Research question

> Does the historical center-square query preprocessing discard horizontal
> visual context that is useful for map candidate generation?

Keep frozen initially:

```text
map:        768_s256
backbone:   DINOv2 ViT-S/14
input:      518
pooling:    avgpatch
similarity: cosine/dot on normalized descriptors
```

No ORB, bootstrap or state logic is allowed in QV1.

## QV1.0 — geometry/provenance preflight

Before new DINO inference:

- verify all 403 blind image paths;
- verify frame dimensions;
- freeze deterministic left/center/right square crop geometry;
- prove L/C/R covers the full horizontal field;
- prove historical center-square parity;
- load no coordinates, GPS, reference or oracle columns.

For 3840 x 2160:

```text
left:    [0,    0, 2160, 2160]
center:  [840,  0, 3000, 2160]
right:   [1680, 0, 3840, 2160]
```

Historical center-square retains 56.25% of the horizontal extent and discards
43.75%.

## QV1.1 — bounded representation comparison

After QV1.0 passes:

```text
A. center_square
   historical control

B. resize_square
   full-frame distortion ablation

C. left / center / right square descriptors
   independent DINO descriptors
   neutral reciprocal-rank fusion
```

No learned crop weighting in the first experiment.

Evaluation reference is attached only after ranking is frozen.


## QV1.0 result

Status:

```text
PASS_QV1_QUERY_VIEW_PREFLIGHT
```

Measured:

```text
queries:                       403
dimensions:                    3840 x 2160
center retained horizontal:    0.5625
center discarded horizontal:   0.4375

left:    [0,    0, 2160, 2160]
center:  [840,  0, 3000, 2160]
right:   [1680, 0, 3840, 2160]

L/C/R horizontal union:        1.0000
adjacent crop overlap:         1320 px
```

The geometry/provenance gate passes.

## QV1.1 implementation

QV1.1 changes query representation only.

Historical control:

```text
center_square:
  reuse the strict-blind frozen img518 DINO query cache
```

New query descriptors to encode:

```text
left_square
right_square
resize_square
```

All use:

```text
DINOv2 ViT-S/14
img518
avgpatch
ImageNet normalization
FP32
L2-normalized descriptor
```

Map descriptors remain exactly the frozen `768_s256` cache.

Rankings:

```text
center_square
left_square
right_square
resize_square
LCR_RRF(left, center, right)
```

LCR RRF is neutral:

```text
RRF k = 60
per-view depth = 50
final depth = 20
equal view weight
```

The ranking CSVs are written and hashed before reference evaluation.

Historical center parity gate:

```text
Top1 containing 174 / 403
Top1 <=40       101 / 403
Top1 <=80       171 / 403
```

If center parity fails, do not interpret new crop results.


## QV1.1 result — center wins standalone, views are complementary

Status:

```text
PASS_QV1_QUERY_VIEW_RETRIEVAL_COMPARISON
```

Historical center parity passed.

Measured:

```text
center_square:
  Top1 contain 174 / 403
  Top1 <=40    101 / 403
  Top1 <=80    171 / 403
  R20 contain 355 / 403
  R20 <=40    284 / 403
  R20 <=80    341 / 403

left_square:
  Top1 contain  81 / 403
  R20 contain 338 / 403

right_square:
  Top1 contain 125 / 403
  R20 contain 334 / 403

resize_square:
  Top1 contain 125 / 403
  R20 contain 357 / 403

L/C/R neutral RRF:
  Top1 contain 144 / 403
  R20 contain 343 / 403
```

Conclusion:

- center-square remains the best standalone query representation;
- naive equal-weight L/C/R RRF is worse than center;
- alternate views contain complementary candidates that center can miss.

Examples:

```text
q57:
  center misses containment within Top20
  right crop finds a containing/<=80 candidate at rank 11

q99:
  center Top1 is correct (~21.5 m)
  other views are poor
  naive LCR fusion damages the correct center result

q390:
  center Top1 is wrong (~463.6 m)
  right Top1 is containing and ~63.8 m
  resize Top1 is containing and ~80.3 m
```

Therefore the crop axis is not closed, but direct multi-view fusion is not
promoted.

## QV1.2 — candidate complementarity / rescue audit

Status: **STARTING**

QV1.2 is post-freeze diagnostic only. It adds no selector and runs no new DINO.

Question:

> How much candidate availability do the alternate query views add beyond the
> center-square control?

Measure for Top20:

- center miss -> left rescue;
- center miss -> right rescue;
- center miss -> resize rescue;
- any L/C/R availability ceiling;
- any center/left/right/resize availability ceiling;
- Top1 rescue opportunities;
- distinct candidate union size and pairwise Top20 overlap/Jaccard.

Important interpretation rule:

```text
multi-view union metrics use a larger effective candidate budget
and are an availability ceiling, NOT a deployable fixed-budget result
```

If extra views provide meaningful unique rescues, preserve their descriptors as
candidate-generation assets for later work. If gains are negligible, close the
query-view axis.


## QV1.2 result — alternate views are valuable candidate generators

Status:

```text
PASS_QV1_QUERY_VIEW_COMPLEMENTARITY_AUDIT
```

Candidate-availability ceiling at Top20:

```text
center only:
  contain 355 / 403
  <=40    284 / 403
  <=80    341 / 403

L/C/R any-view:
  contain 381 / 403
  <=40    331 / 403
  <=80    378 / 403

all independent views:
  contain 386 / 403
  <=40    335 / 403
  <=80    384 / 403
```

Average distinct candidate count when taking unions:

```text
L/C/R:      ~38.7
all views:  ~41.5
```

Unique Top20 rescues beyond center include:

```text
right crop:
  contain 21
  <=40    42
  <=80    33

left crop:
  contain 18
  <=40    15
  <=80    29

resize:
  contain 16
  <=40     8
  <=80    18
```

Therefore alternate query views are retained as candidate-generation assets.

The union metrics are not deployable results because they exceed the fixed
Top20 budget.

## QV1.3 — fixed-budget multi-view candidate pool

Status: **STARTING**

Question:

> Can the multi-view complementarity be compressed back into one fair Top20
> candidate pool while preserving the strongest center-square evidence?

Frozen conservative contract:

```text
final budget:               20
protected center candidates: 10
alternate source depth:      20
spatial exclusion radius:    51.2 m
```

Policies:

```text
center_top20
  historical control

center10_lcr_diverse20
  keep center ranks 1..10
  fill remaining slots with spatially distinct left/right candidates

center10_allview_diverse20
  keep center ranks 1..10
  fill remaining slots with spatially distinct left/right/resize candidates
```

Alternate candidates are added blind in deterministic round-robin source order.
A candidate is considered spatially redundant if its tile center lies within
one map stride (51.2 m) of an already selected tile.

If the diversity pass cannot fill 20 slots, deterministic refill is used so
every query still has exactly 20 candidates.

This is still candidate-pool generation:

```text
NO ORB
NO bootstrap
NO state
NO GT before pool freeze
```

Top1 remains the historical center Top1 by construction.


## QV1.3 result — fixed-budget multi-view pool is promising

Status:

```text
PASS_QV1_FIXED_BUDGET_MULTIVIEW_POOL
```

Measured fixed Top20:

```text
center_top20:
  contain 355 / 403
  <=40    284 / 403
  <=80    341 / 403
  Top1 contain 174 / 403

center10_lcr_diverse20:
  contain 363 / 403   (+8)
  <=40    298 / 403  (+14)
  <=80    363 / 403  (+22)
  Top1 contain unchanged

center10_allview_diverse20:
  contain 376 / 403  (+21)
  <=40    297 / 403  (+13)
  <=80    374 / 403  (+33)
  Top1 contain unchanged
```

Named cases:

```text
q57:
  rescued at fixed Top20

q99:
  historical center Top1 preserved and remains correct

q228:
  remains healthy

q390:
  containment/<=80 retained,
  but center's deeper <=40 candidate is lost
```

Therefore multi-view candidate generation is now positive even under a fair
Top20 budget.

However, the QV1.3 center-depth split at rank 10 is arbitrary and can discard
useful unique center candidates in ranks 11..20.

## QV1.4 — redundancy-aware fixed-budget pool

Status: **STARTING**

Question:

> Can alternate views replace only spatially redundant center candidates,
> rather than replacing every center candidate after an arbitrary rank cutoff?

Frozen blind rule:

1. scan center Top20 in rank order;
2. retain every center candidate whose tile center is more than 51.2 m from
   every already-retained center candidate;
3. center Top1 is always retained;
4. freed slots are filled from alternate query views using the same 51.2 m
   spatial-diversity rule;
5. if diversity cannot fill the Top20, deterministically reinsert dropped
   center candidates before any redundant alternate refill;
6. final candidate budget remains exactly 20.

Policies:

```text
center_top20

center_unique_lcr_fill20
  redundant center slots -> left/right candidates

center_unique_allview_fill20
  redundant center slots -> left/right/resize candidates
```

This removes the arbitrary protected-depth parameter while preserving unique
deep center evidence.

Post-freeze evaluation must report both:

```text
net metric delta
AND
per-query rescue / loss counts versus center_top20
```

No ORB, bootstrap or state logic is involved.

## QV1.4 result — region coverage improves, <=40 precision trades off

Status: PASS_QV1_REDUNDANCY_AWARE_MULTIVIEW_POOL

Measured center_unique_allview_fill20 versus center_top20:

- containment: 384/403 vs 355/403; rescue/loss +29/-0
- <=80 m: 380/403 vs 341/403; rescue/loss +40/-1
- <=40 m: 277/403 vs 284/403; rescue/loss +36/-43
- Top1 containment remains 174/403.

Interpretation: the redundancy-aware pool is extremely strong at preserving and expanding
the correct physical region, but the 51.2 m redundancy rule is too coarse to preserve
every tile-center <=40 candidate.

Named cases: q57 is rescued, q99 remains protected, q228 keeps <=40 in the all-view
pool, and q390 regains its deep-center <=40 evidence.

## QV1.5 — precision-versus-region attribution

Status: STARTING

QV1.5 is diagnostic only. It builds no new pool and runs no new descriptors.

For every center-good <=40 query lost by QV1.3 or QV1.4, classify whether the new
pool still preserves containment and/or <=80 evidence, and measure the nearest selected
tile to the lost precise center tile.

Categories:

- precision_only_region_safe: <=40 lost, but containment and <=80 remain.
- precision_only_le80_safe: <=40 lost, <=80 remains but containment does not.
- true_region_regression: <=40 and <=80 are both lost.

This decides whether QV1.4 is already a strong region-candidate generator for later
geometric refinement, or whether a local member-preservation mechanism is still needed.


## QV1.5 result and closeout

QV1.5 status:

```text
PASS_QV1_PRECISION_REGION_ATTRIBUTION
```

For `center_unique_allview_fill20`:

```text
containment rescue/loss  +29 / -0
<=80 rescue/loss         +40 / -1
<=40 rescue/loss         +36 / -43

all 43 <=40 losses:
  precision_only_region_safe = 43
  true_region_regression     = 0
```

QV1 is now frozen to avoid further research-design overfitting on the same
403-query flight.

Closeout:

```text
docs/closeouts/query_view_candidate_generation_v1/README.md
```

Next:

```text
research/trajectory-adapter-v1
```
