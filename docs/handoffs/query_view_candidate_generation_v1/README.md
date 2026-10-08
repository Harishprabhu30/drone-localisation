# Handoff — query-view candidate generation v1

Status: **QV1.0 CLOSED / QV1.1 STARTED**

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
