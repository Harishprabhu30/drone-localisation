# GD1.0 — Reference-free Top20 geometric evidence freeze

Status: **CODE PREPARED / LOCAL FREEZE PENDING**.

Branch:
research/top20-geometric-discrimination-v1

Base:
RG2.4 closeout commit 1d7c4a801d6e735645244997e02e594c8b84bb3a

## Purpose

Freeze richer blind geometric evidence for every candidate in the already-frozen
QV1.4 Top20 pool. This stage does not rerank, select, fuse, threshold, or use
reference labels.

The research question is:

> Given a strong coarse Top20 retrieval pool, which geometric measurements later
> discriminate correct physical candidates from visually plausible aliases?

## Frozen upstream

- QV1.4 center_unique_allview_fill20
- fixed budget 20
- map variant 768_s256
- frozen query manifests
- frozen 768_s256 map index
- baseline ORB/RANSAC contract from RG1/RG2

Development:
403 queries, 8060 candidate pairs.

Blind stress:
123 queries, 2460 candidate pairs.

## New evidence families

### Correspondence strength
- good matches
- inliers
- inlier ratio
- descriptor-distance summaries
- actual Lowe-ratio summaries

### Spatial support
- bounding-box coverage
- convex-hull coverage
- 4x4 grid occupancy
- normalized grid entropy
- PCA minor/major spread ratio

### Residual geometry
- forward reprojection median/p90/RMSE
- backward reprojection median/p90/RMSE
- symmetric transfer-error median/p90/RMSE

### Local feature consistency
- orientation-delta circular mean
- orientation resultant
- orientation MAD
- log keypoint-scale ratio median/MAD

### Homography shape
- matrix condition number
- local center Jacobian determinant
- Jacobian singular values
- anisotropy
- local rotation
- reflection flag

### Projective stability / footprint plausibility
- homography denominator minimum/maximum magnitude over query footprint
- denominator ratio and sign consistency
- projected query-footprint finiteness/convexity
- footprint area relative to satellite tile
- projected-corner inside fraction
- footprint edge-length ratio

These are measurements only. No variable is promoted as confidence evidence in
GD1.0.

## Scientific boundary

- no reference / GT / SRT / GPS
- no candidate selection
- no reranking
- no score fusion
- no threshold tuning
- no learned matcher
- no temporal fusion
- no bootstrap/state change

The next post-freeze block will attach development reference to ask which of
these blind variables actually discriminate containment / <=40 / <=80 candidates
and whether they can identify a useful Top20 locality.

## Run

    git pull --ff-only origin research/top20-geometric-discrimination-v1
    source .drone_venv/bin/activate

    python -m unittest tests.test_gd1_0_top20_geometric_evidence -v

    python scripts/villoc/geometry/gd1_0_freeze_top20_geometric_evidence.py \
      --config configs/research/top20_geometric_discrimination_v1.yaml

Expected unit tests:
6 tests.

Expected stage status:
PASS_GD1_0_REFERENCE_FREE_TOP20_GEOMETRIC_EVIDENCE_FREEZE

Return the full output including:
- query/pair counts
- unique tiles
- homography-ok / finite-projection / inside counts
- all20-homography query count
- failed-homography query count
- evidence SHA256 for both trajectories

Do not attach reference or build a reranker before this freeze is inspected.
