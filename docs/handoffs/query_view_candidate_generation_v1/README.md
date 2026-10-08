# Handoff — query-view candidate generation v1

Status: **QV1.0/QV1.1 CLOSED — QV1.2 STARTED**

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
