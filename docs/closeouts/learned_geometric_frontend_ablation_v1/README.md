# Learned geometric frontend ablation v1 — branch closeout

Status: **CLOSED / DIRECTION CHANGE**.

This branch tested whether generic learned local matching could raise the geometric-discrimination ceiling over the frozen ORB Top20 verifier. It did not. The branch is closed so the next work can explicitly study cross-view scale/rotation/domain effects instead of continuing incremental sparse-frontend swaps.

## Main problem remains unchanged

The project is still solving the same core absolute-localization problem:

> Given a UAV query and the frozen QV1.4 Top20 map candidates, identify the correct physical locality reliably enough to produce a precise continuous map position without catastrophic false corrections.

The direction change is a subproblem change, not a project change. Retrieval stays frozen unless later evidence shows that it is again the bottleneck.

## What this branch established

### ORB control

ORB remains the strongest tested local frontend. On the full 403-query development set, raw ORB inlier ranking reached:
- contains_query Top1: 292/403;
- center <=40 m Top1: 150/403;
- center <=80 m Top1: 287/403;
- selected projection median: 6.092 m;
- selected projection <=40 m: 293/403;
- selected projection >200 m: 88/403.

### XFeat

Sparse XFeat and XFeat* semi-dense were both rejected.
- Sparse XFeat Top1: 184/403 containment, 91/403 <=40 m, 182/403 <=80 m.
- XFeat* Top1: 168/403 containment, 79/403 <=40 m, 162/403 <=80 m.
- XFeat* increased correspondence density but reduced discrimination.

### SuperPoint + LightGlue

The official pinned LightGlue integration was tested with SuperPoint.
- default adaptive smoke was viable;
- official maximum-accuracy profile (`max_num_keypoints=None`, `depth_confidence=-1`, `width_confidence=-1`) was much slower and weaker on this workload;
- deterministic uniform40 truth audit: 18/40 containment and 18/40 <=80 m versus ORB 28/40 and 28/40;
- decision: close SuperPoint+LightGlue.

### ALIKED + LightGlue

ALIKED improved the reference-free smoke and improved over SuperPoint in the uniform40 audit, but still failed the ORB gate:
- containment 21/40 versus ORB 28/40;
- <=40 m 13/40 versus ORB 13/40;
- <=80 m 20/40 versus ORB 28/40;
- selected projection median 107.885 m versus ORB 5.416 m;
- >200 m 17/40 versus ORB 10/40.

Final LF2.1-C decision: `CLOSE_SPARSE_LIGHTGLUE_FAMILY`.

## New working hypotheses

These are hypotheses, not yet causal conclusions:

1. ORB may benefit from explicit local orientation canonicalization and its multi-scale pyramid under UAV heading changes and query/map scale mismatch.
2. Generic learned sparse features are being used zero-shot outside their dominant training distribution; cross-view/aerial-specific supervision may matter more than matcher complexity.
3. ALIKED's partial improvement suggests feature support geometry/representation matters.
4. Correct-candidate geometry often exists somewhere in Top20, but ranking/alias discrimination is the dominant failure.

## What was NOT completed

A controlled per-query UAV-GSD-to-orthophoto-GSD normalization experiment was not closed in the current VILLOC pipeline. Earlier retrieval work changed physical map footprint while keeping orthophoto source GSD fixed; that is not the same experiment.

## Transition

Continue on branch:
`research/crossview-scale-rotation-domain-v1`

Canonical handoff:
`docs/handoffs/crossview_scale_rotation_domain_v1/README.md`

Do not resume generic XFeat/SuperPoint/ALIKED threshold tuning on this closed branch unless new evidence explicitly reopens it.
