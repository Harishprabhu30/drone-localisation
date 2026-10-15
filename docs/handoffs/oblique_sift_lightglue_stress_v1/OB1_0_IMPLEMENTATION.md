# OB1.0 — 45-degree oblique stress of the current promoted frontend

Purpose: test whether the current VILLOC absolute-localization stack transfers
from the 90-degree stable120m development trajectory to the existing 112-query
45-degree oblique dataset without tuning.

## Why this is a new run

The historical 45-degree experiments used:
- 1024_s512 tiles;
- DINOv2 img224;
- older ORB / SuperPoint-LightGlue-style reranking policies.

The current promoted stack uses:
- 768_s256;
- DINOv2 ViT-S/14 img518 center-square avg-patch;
- SIFT + official SIFT-LightGlue;
- query-center homography projection;
- inliers-descending selector.

Therefore OB1.0 rebuilds the 45-degree retrieval candidates under the current
map/descriptor architecture instead of reusing the historical candidate pool.

## One combined execution

Reference-free phase:
1. load exactly 112 existing 45-degree query images;
2. build/reuse an img518 DINO query cache;
3. reuse the current frozen 768_s256 map descriptor cache;
4. compute a new DINO Top20 (2,240 pairs);
5. run the promoted SIFT+LightGlue frontend unchanged;
6. freeze candidate, pair and selected evidence;
7. verify hashes.

Only after the reference-free freeze:
8. attach the existing 45-degree SRT/GPS development reference;
9. evaluate tile/locality and projected-center stress metrics.

## Oblique evaluation caveat

At 45 degrees, the center of the UAV image can view ground significantly ahead
of the UAV body. A correct query-center homography projection therefore need
not land on the body point.

For this reason OB1.0 reports separately:
- selected tile contains-body;
- selected tile-center-to-body error;
- DINO Top-K body-containment availability;
- projected query-center-to-body error.

The final quantity is explicitly a viewpoint stress diagnostic and must not be
presented as a pure body-localization metric unless a valid camera-footprint
model is introduced later.

No footprint offset correction is applied in OB1.0 because the historical
simple footprint-offset experiment was negative.

## Scientific boundary

No SIFT/LightGlue parameter changes.
No threshold tuning.
No selector changes.
No reference before the frozen evidence is written.
No attempt to tune specifically for oblique imagery.

The historical 45-degree LightGlue result is contextual only, not a clean
single-variable ablation, because map geometry, DINO representation and
frontend/policy all differ.
