# CV1.3 — full403 promoted scale freeze

Status: **IMPLEMENTATION PREPARED / LOCAL EXECUTION REQUIRED**

Promoted condition:
`scale_m2 = 0.6944444444444444`

Promotion source:
CV1.2 uniform40 post-freeze evaluation.

## Why only scale_m2

Both `scale_m2` and `scale_m1` passed the screening continuation gate, but
the predeclared promotion rule selected `scale_m2` because its selected
projection catastrophic tail was smaller.

No second scale is carried into full403 and no factor is retuned.

## Boundary

CV1.3:
- processes all 403 development queries x frozen QV1.4 Top20;
- applies only the frozen scale_m2 transform;
- does not load reference truth;
- does not load SRT;
- does not select a candidate winner;
- does not evaluate accuracy;
- does not regenerate the baseline.

The frozen GD1.0 full403 evidence remains the baseline for the later CV1.4
post-freeze comparison.

## Local run

```bash
python -m unittest tests.test_cv1_3_full403_scale_freeze -v

python scripts/villoc/geometry/cv1_3_freeze_full403_scale_m2_evidence.py \
  --config configs/research/crossview_scale_rotation_domain_cv1_3.yaml
```

Expected evidence rows:
`403 x 20 = 8060`.

Do not run a reference/accuracy comparison before the CV1.3 evidence SHA256 has
been frozen.
