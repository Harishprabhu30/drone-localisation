# CV3.2 — uniform40 x six-candidate reference-free aerial-MASt3R screen

Status: **IMPLEMENTATION PREPARED**

## Purpose

Freeze the first candidate-separation evidence for aerial-MASt3R without using
development truth.

Exact query set:
the frozen uniform40 used by CV1/CV2.

Exact candidate slice:
pool ranks \`1,2,3,5,10,20\`.

Total:
40 x 6 = 240 pairs.

## Inputs and model contract

Unchanged from CV3.1B1:
- QV1.4 Top20 SHA256
  \`0a7832c6f0997a33851343fd7df537d2f6410211ac0e38e7ec4325899d8df47e\`;
- pinned AerialMegaDepth source commit
  \`cdf478ff4fdaf888fb91c940bc48d55095e2cdbd\`;
- pinned aerial-MASt3R checkpoint revision;
- model SHA256
  \`093c3b4b3ad8294673774aae106309d5f6a1a975976bae574075fb7c5f94a67d\`;
- FP32;
- load size 512;
- square_ok=false;
- extract_correspondences_nonsym;
- subsample 4;
- pixel_tol 3;
- confidence >=0.3;
- 3-pixel processed-image border;
- canonical-coordinate restoration;
- 5 px canonical-space homography RANSAC;
- minimum 4 inliers.

No tuning is introduced after CV3.1.

## Bundle

A Mac-side reference-free bundle contains:
- 40 canonical query images;
- the candidate tile images needed by the exact six-rank slice;
- source rank/view;
- metric tile bounds;
- SHA256 for each copied asset.

It contains no SRT/reference coordinates.

## Frozen selector

After all 240 pair measurements are generated, CV3.2 freezes the predeclared
selector:

\`inliers DESC -> pool_rank ASC -> tile_id ASC\`.

Two outputs are frozen before truth:
1. pair-level geometry evidence;
2. one selected candidate/projection per query, including inlier margin.

No learned score combination is permitted.

## Runtime / memory

The Jetson runs one pair at a time and empties transient CUDA cache between
pairs. A progress JSONL is written after every pair.

CV3.1B1 observed only ~0.733 GiB minimum free shared/system memory, so CV3.2
keeps the exact same FP32/batch-1 contract and treats any OOM as evidence rather
than silently changing precision.

## Stage boundary

CV3.2 makes no accuracy claim.

Only after both 240-pair evidence files and the selector output are frozen may
CV3.3 attach development truth and compare aerial-MASt3R against the same
six-candidate SIFT+LightGlue control.

CV3.3 continuation gate remains the protocol freeze:
- +3/40 Top1 containment; or
- +3/40 Top1 <=80 m; or
- >=3 unsafe-tail reduction without >2 loss in selected <=40 m.
