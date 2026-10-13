# CV1.4 — full403 post-freeze scale_m2 validation

Status: **IMPLEMENTATION PREPARED / LOCAL EXECUTION REQUIRED**

Inputs:
- frozen GD1.0 full403 ORB baseline;
- frozen CV1.3 full403 `scale_m2` evidence.

## Purpose

Test whether the uniform40 scale effect replicates over all 403 development
queries without introducing a second scale factor or changing the selector.

Primary question:

> Does fixed query downsampling reduce catastrophic wrong-locality selection on
> full403 while preserving the existing precise-projection behavior?

## Evaluation

Both evidence sets are verified before development truth is opened.

The selector remains:
`inliers DESC -> pool_rank ASC -> tile_id ASC`.

Report:
- Top1 containment / <=40 / <=80;
- Top3/5/10/20 availability after inlier ranking;
- selected projection median / RMSE / p95 / <=40 / <=80 / >200;
- oracle-best finite projection;
- selected-tile changes;
- containment, <=80 and catastrophic rescue/damage counts;
- Top1-Top2 inlier margin.

## Predeclared replication gate

Continue to a camera/GSD/mechanism audit only if full403 satisfies all of:

- at least 10 fewer selected projections >200 m;
- no more than 5 fewer selected projections <=40 m;
- no more than 5 Top1 containment losses;
- no more than 5 Top1 <=80 m losses.

This is a continuation/resource gate, not a significance test and not a
deployment acceptance rule.

If it fails, the uniform40 scale result is treated as non-robust for the current
fixed factor and CV1 scale work should close rather than retuning on full403.

## Local execution

```bash
python -m unittest tests.test_cv1_4_full403_postfreeze_scale_validation -v

python scripts/villoc/geometry/cv1_4_postfreeze_full403_scale_validation.py \
  --config configs/research/crossview_scale_rotation_domain_cv1_4.yaml
```
