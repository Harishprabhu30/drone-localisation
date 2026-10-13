# CV2.3 post-freeze SIFT + LightGlue uniform40 comparison

Status: **PASS_CV2_3_POSTFREEZE_SIFT_LIGHTGLUE_COMPARISON**

Decision:
`PROCEED_SIFT_LIGHTGLUE_FULL403`

## Frozen comparison

| frontend | Top1 contains | Top1 <=40 | Top1 <=80 | finite | nonfinite | proj median m | RMSE m | p95 m | proj <=40 | proj <=80 | proj >200 | unsafe nonfinite/>200 | oracle median m | oracle <=40 | oracle <=80 | median Top1-Top2 inlier margin |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| original ORB | 28 | 13 | 28 | 40 | 0 | 5.416024 | 366.297768 | 872.632052 | 29 | 29 | 10 | 10 | 5.071893 | 35 | 36 | 2.0 |
| scale_m2 ORB | 32 | 15 | 31 | 40 | 0 | 5.227080 | 300.000234 | 849.432714 | 32 | 34 | 6 | 6 | 5.030855 | 33 | 37 | 3.0 |
| SIFT + LightGlue | 33 | 18 | 33 | 40 | 0 | 4.888097 | 239.519438 | 706.720011 | 36 | 36 | 4 | 4 | 4.652173 | 39 | 40 | 43.5 |

## Frozen continuation gate versus scale_m2 ORB

Result:
`PASS`

SIFT+LightGlue minus scale_m2 ORB:
- Top1 containment: +1
- Top1 center <=40: +3
- Top1 center <=80: +2
- selected projection <=40: +4
- selected projection >200: -2

No CV1 scale preprocessing was applied to SIFT.
No threshold tuning was performed.
No full403 SIFT+LightGlue run was executed before this gate.

## Interpretation

The native SIFT + SIFT-specific LightGlue representation passes the predeclared
uniform40 continuation gate against the stronger scale_m2 ORB control.

The large increase in median Top1-Top2 inlier margin (3.0 -> 43.5) is
particularly notable: on this screen, SIFT+LightGlue is not merely changing
near-ties but often creates substantially larger separation between its first
and second candidate under the common raw-inlier selector.

This remains a screening result. The exact unchanged frontend must now be
measured on all 403 development queries before any full-trajectory accuracy
claim.

## Next stage

CV2.4:
freeze native SIFT+LightGlue evidence for all 403 development queries x frozen
QV1.4 Top20, reference-free.

Do not:
- change SIFT parameters;
- change LightGlue parameters;
- apply scale_m2 to SIFT;
- tune the selector;
- attach truth before the CV2.4 evidence hash is frozen.
