# Query-view candidate generation v1 closeout — 2026-10-09

Status: **CLOSED / FROZEN DEVELOPMENT HYPOTHESIS**

Branch:

```text
research/query-view-candidate-generation-v1
```

## Purpose

QV1 tested whether changing only the UAV query representation could improve
absolute candidate generation against the frozen 768_s256 map representation.

No ORB, bootstrap or state logic was used to construct the QV1 candidate pools.

## Final development-trajectory finding

Historical center Top20:

```text
contain 355 / 403
<=40    284 / 403
<=80    341 / 403
Top1 contain 174 / 403
```

Frozen QV1 research candidate:

```text
center_unique_allview_fill20

contain 384 / 403
<=40    277 / 403
<=80    380 / 403
Top1 contain 174 / 403
```

Rescue/loss versus center:

```text
containment +29 / -0
<=80       +40 / -1
<=40       +36 / -43
```

QV1.5 attributed all 43 <=40 losses in the redundancy-aware all-view pool as:

```text
precision_only_region_safe  43
precision_only_le80_safe     0
true_region_regression       0
```

For all 43 precision losses, the nearest retained selected tile was within one
51.2 m map stride of the lost precise center tile.

Interpretation:

The QV1.4 all-view construction is a strong **region-level fixed-budget
candidate generator** on the development trajectory. It should not be described
as a universally validated localization frontend yet.

## Why QV1 stops here

The same 403-query trajectory has been used repeatedly for discovery,
diagnostics and architecture refinement. Even with strict pre-freeze leakage
rules, continued design changes on the same flight risk research-design
overfitting.

Therefore QV1 is frozen here rather than tuned further.

## Frozen hypothesis for future validation

```text
map representation:
  768_s256

query views:
  center_square
  left_square
  right_square
  resize_square

candidate budget:
  20

center policy:
  preserve center Top1
  preserve spatially unique center Top20 regions

alternate policy:
  replace only center candidates redundant within 51.2 m
  fill with spatially distinct left/right/resize candidates

name:
  center_unique_allview_fill20
```

## Validation policy

Use complete trajectories as experimental units.

Do not randomly split adjacent frames from one flight into train/test sets.

For a trajectory with reference:

```text
blind candidate generation
  -> freeze/hash
  -> attach reference
  -> absolute accuracy evaluation
```

For a trajectory without reference:

```text
blind candidate generation
  -> freeze/hash
  -> reference-free behavioral/stress diagnostics
```

No absolute-accuracy claim may be made from a trajectory with no GT/reference.

## Next architecture step

Introduce a reusable trajectory adapter so research code consumes a canonical
trajectory package rather than hard-coded dataset-specific paths.

Next branch:

```text
research/trajectory-adapter-v1
```
