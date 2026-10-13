# CV2.3 — post-freeze SIFT+LightGlue vs ORB controls

Status: **IMPLEMENTATION PREPARED / LOCAL EXECUTION REQUIRED**

## Frozen inputs

CV1 uniform40 ORB evidence:
`4abfacb77cf9d1d4b09ea917e048b0f2191659907dc4391f7d8fc3aadf020666`

CV2.2 SIFT+LightGlue evidence:
`c2d8f510db9a0fcc3740b06a898861371cea16da492d194326ab7a37a7494d36`

Both hashes are verified before development truth is opened.

## Comparison

The same raw-inlier selector is used for all three frontends:

`inliers DESC -> pool_rank ASC -> tile_id ASC`

Frontends:
1. original ORB;
2. CV1 scale_m2 ORB;
3. native SIFT + LightGlue.

The frozen CV2 continuation gate is evaluated **against scale_m2 ORB**, because
that is the strongest current classical control.

## Metrics

- Top1 containment / center <=40 / center <=80;
- Top3/5/10/20 hits;
- selected projection finite count;
- selected projection median/RMSE/p95;
- selected projection <=40 / <=80 / >200;
- selected non-finite projection count;
- diagnostic unsafe tail = non-finite OR >200;
- Top1-Top2 inlier margin;
- oracle-best finite projection.

The non-finite/unsafe-tail metrics are diagnostic only and do not modify the
already-frozen CV2 continuation gate.

## Decision

Proceed to SIFT+LightGlue full403 only if the frozen gate relative to scale_m2
ORB passes.

If it fails:
close CV2 and move to the predeclared aerial-domain upper-bound family rather
than tuning SIFT thresholds, LightGlue thresholds, or applying scale_m2 to SIFT
on this development screen.

## Local run

```bash
python -m unittest tests.test_cv2_3_sift_lightglue_postfreeze_comparison -v

python scripts/villoc/geometry/cv2_3_postfreeze_sift_lightglue_comparison.py \
  --config configs/research/crossview_scale_rotation_domain_cv2_3.yaml
```
