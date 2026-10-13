# CV2.0 protocol freeze — SIFT + LightGlue diagnostic

Status: **FROZEN PROTOCOL / PREFLIGHT MAY BEGIN**

## Research question

> Inside the frozen QV1.4 Top20 pool, does native SIFT + the official
> SIFT-specific LightGlue matcher provide stronger candidate discrimination than
> generic learned sparse frontends, and can it approach or exceed the frozen
> scale_m2 ORB control?

CV2 tests the hypothesis that explicit local feature scale/orientation support
helps cross-view UAV-to-orthophoto matching.

## Why this is not a repeat of LF2

The previously closed LightGlue frontends were:
- SuperPoint + LightGlue;
- ALIKED + LightGlue.

The pinned official LightGlue implementation has a dedicated SIFT checkpoint.
For `features='sift'`, LightGlue explicitly adds the extractor's SIFT scale and
orientation values to its positional encoding.

CV2 therefore tests a different representation contract rather than reopening
generic learned sparse threshold tuning.

## Pinned dependency

Official repository:
`https://github.com/cvg/LightGlue.git`

Pinned commit:
`eb42fee2d71449efb0aa5c10549752b5d75384d8`

Pinned source evidence:
- `lightglue/sift.py` Git blob:
  `802fc1c2eb9ee852691e0e4dd67455f822f8405f`
- `lightglue/lightglue.py` Git blob:
  `485833515eb0a80866b3d078d3e43d39ddcd84ae`

## SIFT extractor contract

Use the pinned official `lightglue.SIFT` extractor:

- backend: `opencv`
- RootSIFT: true
- max_num_keypoints: 4096
- internal resize long edge: 1024
- NMS radius: 0
- detection threshold: 0.0066667
- edge threshold: 10
- number of octave layers: 4

Do not tune these values from VILLOC truth.

The official extractor maps internally resized keypoints back into original
image coordinates.  Homography/query-center projection therefore remains in the
same original pixel-coordinate convention used by the existing absolute
frontends.

## LightGlue matcher contract

- feature type: `sift`
- checkpoint family: `sift_lightglue`
- filter threshold: 0.1
- depth confidence: 0.95
- width confidence: 0.99
- device: CPU for parity with prior LF2 screening
- RANSAC threshold: 5 px
- minimum homography inliers: 4

Keep the default adaptive profile for the first diagnostic.  Do not perform a
second maximum-accuracy threshold sweep unless the default profile exposes a
specific blocker.

## Inputs

Frozen development trajectory:
`villoc_traj01_90deg_stable120m`

Frozen QV1.4 Top20 SHA256:
`0a7832c6f0997a33851343fd7df537d2f6410211ac0e38e7ec4325899d8df47e`

Use the exact existing uniform40 query IDs:
`1,11,22,32,42,53,63,73,83,94,104,114,125,135,145,156,166,176,187,197,207,217,228,238,248,259,269,279,290,300,310,321,331,341,351,362,372,382,393,403`

Screen size:
`40 x 20 = 800 pairs`.

## Controls

Post-freeze CV2 comparison must include:

1. original frozen ORB control;
2. frozen CV1 `scale_m2` ORB control;
3. native SIFT + LightGlue.

CV1 uniform40 evidence SHA256:
`4abfacb77cf9d1d4b09ea917e048b0f2191659907dc4391f7d8fc3aadf020666`

The new primary control is scale_m2 ORB because it is the strongest current
frozen ORB intervention.

## Critical causal boundary

Do **not** apply scale_m2 to SIFT in the first CV2 screen.

Reason:
the first question is whether native SIFT scale/orientation support itself
closes the gap.  Applying scale_m2 simultaneously would mix the CV1
preprocessing intervention with the SIFT representation hypothesis.

## Stage sequence

### CV2.1
Small reference-free SIFT+LightGlue smoke:
- query IDs 1, 207, 403;
- pool ranks 1, 10, 20;
- 9 pairs total.

Purpose:
- verify pinned source;
- download/hash SIFT-LightGlue weights;
- verify SIFT scale/orientation fields are present;
- verify homography/projection plumbing;
- measure rough runtime only.

No accuracy claim.

### CV2.2
If smoke passes:
freeze native SIFT+LightGlue evidence on exact uniform40 x Top20.

No reference, reranking, threshold tuning, or full403.

### CV2.3
Only after CV2.2 evidence is frozen:
attach development truth and compare original ORB, scale_m2 ORB, and
SIFT+LightGlue.

## Continuation gate relative to scale_m2 ORB

A SIFT+LightGlue full403 run is justified only if the uniform40 post-freeze
comparison satisfies both:

Ranking:
- no regression greater than 2 queries on each of Top1 containment, Top1 <=40,
  and Top1 <=80 relative to scale_m2 ORB;
- at least +2 queries on at least one of those targets.

Projection:
- selected projection <=40 loses no more than 2 queries;
- selected projection >200 increases by no more than 2 queries.

This is a resource gate, not a deployment threshold or significance test.

If SIFT+LightGlue fails, close CV2 and proceed to the aerial-domain upper-bound
family rather than tuning SIFT/LightGlue thresholds on VILLOC.
