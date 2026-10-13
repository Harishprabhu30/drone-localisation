# CV2.4 — SIFT + LightGlue full403 reference-free freeze

Status: **IMPLEMENTATION PREPARED / LOCAL EXECUTION REQUIRED**

Promotion source:
CV2.3 uniform40 post-freeze comparison.

Frozen frontend:
- official pinned SIFT extractor;
- official SIFT-specific LightGlue checkpoint;
- unchanged CV2.0 parameters;
- CPU device;
- no CV1 scale_m2 preprocessing.

## Purpose

Measure the promoted SIFT+LightGlue frontend on:

`403 queries x 20 frozen QV1.4 candidates = 8060 pairs`

without opening development truth.

## Boundary

CV2.4:
- verifies the CV2.2 evidence hash and CV2.3 promotion decision;
- verifies the frozen QV1.4 candidate-pool hash;
- processes every candidate pair in deterministic query/rank order;
- retains failed homography pairs;
- excludes runtime fields from the frozen evidence bytes;
- does not select/rerank a winner;
- does not load reference truth;
- does not tune anything.

The existing frozen controls are not regenerated:
- GD1.0 original ORB full403;
- CV1.3 scale_m2 ORB full403.

Those are used only after CV2.4 freezes, in CV2.5.

## Local run

```bash
python -m unittest tests.test_cv2_4_sift_lightglue_full403_freeze -v

python scripts/villoc/geometry/cv2_4_freeze_sift_lightglue_full403.py \
  --config configs/research/crossview_scale_rotation_domain_cv2_4.yaml
```

Expected evidence rows:
`8060`.

Do not attach reference truth before the CV2.4 evidence SHA256 is frozen.
