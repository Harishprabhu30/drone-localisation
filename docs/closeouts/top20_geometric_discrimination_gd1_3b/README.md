# GD1.3-B closeout — ORB-side selector formulation gate

Status: **PASS / CLOSED** on user-reported local execution.

Status:
PASS_GD1_3B_OOF_QUERY_RELATIVE_LOCALITY_AUDIT

OOF score SHA256:
6948f6d463fb3116a4d176a721cad6a6b5375c60cd6bfa6898fae7c62f8951fa

## Decision

Query-relative normalization and physical-locality features do not materially outperform the raw ORB inlier baseline across the three query-level objectives. Further ORB-side selector formulation is therefore stopped here; the next stage changes the local feature frontend under the same frozen Top20 candidate pool and downstream evaluation contract.

## Key Top1 results

### contains_query
- retrieval rank: 174/403
- inliers only: 296/403
- inliers query-relative: 292/403
- all query-relative geometry: 303/403
- best locality-combined model: 298/403

### center <=40 m
- retrieval rank: 101/403
- inliers only: 158/403
- inliers query-relative: 150/403
- all query-relative geometry: 138/403
- best locality-combined model: 150/403

### center <=80 m
- retrieval rank: 171/403
- inliers only: 288/403
- inliers query-relative: 287/403
- all query-relative geometry: 290/403
- best locality-combined model: 292/403

## Interpretation

1. ORB inlier count remains the strongest simple Top20 ranking cue.
2. Query-relative normalization improves pair AUC but does not materially improve Top1 ranking.
3. Locality-component features alone are weaker than candidate-level geometry.
4. Combining locality support with query-relative geometry does not unlock a large hidden gain.
5. The strict <=40 m objective is still best with raw inliers (158/403).
6. The evidence now supports moving to a controlled frontend ablation rather than continuing to add selector complexity around ORB.

No blind-stress labels were loaded. No deployable model was exported.

## Next stage

Create a separate learned-frontend ablation branch from this closeout. Keep QV1.4 Top20, map variant, query images, homography/projection semantics, evaluation labels, and OOF protocol fixed. Change only the correspondence frontend.

First learned frontend: official VERLab XFeat at the historical pinned commit e92685f57f8318b18725c5c8c0bd28c7fe188d9a. Begin with environment/API/weights preflight and a small reference-free absolute query-to-map smoke test before any full Top20 run.
