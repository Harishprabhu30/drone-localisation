# CV2.5 — full403 post-freeze SIFT+LightGlue validation

Status: **IMPLEMENTATION PREPARED / LOCAL EXECUTION REQUIRED**

## Purpose

CV2.4 completed the expensive CPU run, so the next step is not Jetson migration.
It is the inexpensive post-freeze truth attachment needed to determine whether
the SIFT+LightGlue uniform40 gain survives all 403 queries.

Frozen inputs:
- original ORB full403;
- CV1 scale_m2 ORB full403;
- CV2.4 native SIFT+LightGlue full403.

All evidence hashes are checked before reference truth is opened.

## Common selector

`inliers DESC -> pool_rank ASC -> tile_id ASC`

No frontend-specific selector is introduced.

## Metrics

For all three frontends:
- Top1 containment / center <=40 / center <=80;
- Top3/5/10/20 ranking hits;
- selected finite/non-finite projection counts;
- median/RMSE/p95;
- selected <=40 / <=80 / >200;
- unsafe tail = non-finite selected projection OR >200 m;
- oracle-best finite projection;
- median Top1-Top2 inlier margin.

Pairwise SIFT+LightGlue vs scale_m2 ORB:
- winner changes;
- containment rescues/damages;
- <=80 rescues/damages;
- unsafe-tail rescues/damages.

## Predeclared validation decision

The full403 SIFT+LightGlue result is considered **robustly promotable over
scale_m2 ORB** only if:

Safety/preservation:
- no Top1 target regresses by more than 5 queries;
- selected <=40 projections regress by no more than 5;
- unsafe tail increases by no more than 5.

And at least one meaningful gain is present:
- +5 on any Top1 target; or
- +10 selected <=40 projections; or
- -10 unsafe-tail queries.

If preservation passes but no meaningful gain is present, SIFT+LightGlue is a
valid research alternative but scale_m2 ORB remains the simpler primary control.

If preservation fails, close CV2 as non-robust full403.

This is a research continuation gate, not a deployment acceptance threshold.

## Jetson status

Jetson/CUDA migration is parked until CV2.5 is known.

Reason:
the CPU full403 evidence has already completed and is the cleanest continuation
of the frozen CV2 protocol.  If SIFT+LightGlue survives CV2.5, Jetson becomes
high-value because runtime is then the major blocker.  If it fails CV2.5,
spending effort porting this frontend to CUDA is lower priority.

## Local execution

```bash
python -m unittest tests.test_cv2_5_sift_lightglue_full403_validation -v

python scripts/villoc/geometry/cv2_5_postfreeze_sift_lightglue_full403_validation.py \
  --config configs/research/crossview_scale_rotation_domain_cv2_5.yaml
```
