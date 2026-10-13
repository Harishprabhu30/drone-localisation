# CV2.2 — SIFT + LightGlue uniform40 reference-free freeze

Status: **IMPLEMENTATION PREPARED / LOCAL EXECUTION REQUIRED**

CV2.1 passed with the pinned SIFT-specific LightGlue frontend.

CV2.2 now freezes evidence for the exact existing uniform40 screen:

`40 queries x 20 frozen QV1.4 candidates = 800 pairs`.

## Boundary

CV2.2:
- uses native pinned SIFT + LightGlue;
- does not apply CV1 scale_m2 to SIFT;
- does not load reference truth;
- does not select/rerank candidates;
- does not tune thresholds;
- does not run full403.

Runtime is measured and printed, but runtime fields are excluded from frozen
evidence bytes.

## Local run

```bash
python -m unittest tests.test_cv2_2_sift_lightglue_uniform40 -v

python scripts/villoc/geometry/cv2_2_freeze_sift_lightglue_uniform40.py \
  --config configs/research/crossview_scale_rotation_domain_cv2_2.yaml
```

Do not run CV2.3 accuracy comparison until the CV2.2 evidence SHA256 is frozen.
