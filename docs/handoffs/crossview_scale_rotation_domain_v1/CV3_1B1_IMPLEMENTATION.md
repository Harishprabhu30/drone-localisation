# CV3.1B1 — reference-free 9-pair aerial-MASt3R smoke

Status: **IMPLEMENTATION PREPARED**

CV3.1B0 result:
- PASS FP32 model-load gate;
- 688,638,856 parameters;
- 2.565 GiB FP32 parameters;
- 2.662 GiB CUDA/shared memory free after model load;
- 2.695 GiB system memory available after model load;
- no image inference in B0.

## Purpose

CV3.1B1 is the first image inference stage.

Frozen pairs:
- query IDs: 1, 207, 403;
- QV1.4 pool ranks: 1, 10, 20;
- total: 9.

No truth is attached and no candidate is reranked.

## Why a tiny transfer bundle is used

The Jetson's historical benchmark assets are based on a different map variant.
CV3 is locked to QV1.4 / 768_s256.

The Mac therefore prepares a reference-free bundle containing only:
- 3 canonical query images;
- the 9 selected 768_s256 candidate tile files (deduplicated on disk);
- tile metric bounds;
- source ranks/views;
- SHA256 for every image;
- a manifest tied to the frozen QV1.4 Top20 SHA.

No SRT/reference coordinates are copied.

## MASt3R image geometry

The pinned upstream loader resizes the long edge to 512 and center-crops to
multiples of 16. With square_ok=false, square map tiles become 512x384.

CV3.1B1 mirrors that geometry and maps dense correspondence coordinates back
into canonical original image pixels before homography estimation.

For the canonical expected sizes:
- 3840x2160 query -> 512x288;
- 768x768 tile -> resize 512x512 -> centered 512x384 crop.

Homography:
- query canonical pixels -> tile canonical pixels;
- RANSAC threshold 5 px in canonical space;
- minimum 4 inliers;
- canonical query center projection;
- existing raster pixel-center map convention.

## Runtime/memory safety

Inference runs in a child process. The parent writes a BLOCKED report if the
worker exits or is killed.

The worker:
- loads the verified FP32 model once;
- processes the nine pairs sequentially at batch size 1;
- records inference/matching/geometry runtime;
- records CUDA and system memory per pair;
- empties transient CUDA cache after each pair;
- writes a progress JSONL after every completed pair.

Runtime and memory fields are diagnostic and are excluded from frozen geometric
evidence.

## Correspondence contract

From the pinned AerialMegaDepth demo:
- load size: 512;
- extract_correspondences_nonsym;
- subsample 4;
- pixel tolerance 3;
- descriptor-confidence threshold 0.3;
- 3-pixel processed-image border exclusion.

No tuning is allowed in CV3.1B1.

## Stage result meaning

PASS means only:
- model inference is feasible on this Jetson for the smoke;
- dense correspondences can be extracted;
- coordinates can be converted to canonical image space;
- homography/projection plumbing operates on the nine frozen pairs.

It is not an accuracy result.

Post-freeze truth is still forbidden in CV3.1B1.


## First Jetson inference attempt diagnostic

The first CV3.1B1 Jetson attempt reached the worker's final result-writing step
after the nine-pair processing loop, but strict JSON serialization failed
because unavailable projection fields were represented internally as NaN.

This is a serialization defect, not evidence of model/inference failure.
Non-finite geometry values are legitimate when a pair does not produce a valid
homography/projection.

The corrected worker converts non-finite JSON-facing values to `null` while
preserving the existing CSV evidence behavior and the frozen matching/geometry
contract. Progress trace JSON is also written strictly after this fix.

No model, dtype, threshold, resize, matching, RANSAC, or asset identity changed.
