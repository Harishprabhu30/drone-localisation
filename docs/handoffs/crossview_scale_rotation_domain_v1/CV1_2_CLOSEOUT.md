# CV1.2 post-freeze uniform40 scale / rotation evaluation

Status: **PASS_CV1_2_POSTFREEZE_UNIFORM40_EVALUATION**

Input CV1.1 evidence SHA256:
`4abfacb77cf9d1d4b09ea917e048b0f2191659907dc4391f7d8fc3aadf020666`

## Result table

| condition | Top1 contains | Top1 <=40 | Top1 <=80 | proj median m | proj RMSE m | proj p95 m | proj <=40 | proj <=80 | proj >200 | gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| baseline | 28 | 13 | 28 | 5.416 | 366.298 | 872.632 | 29 | 29 | 10 | FAIL/control |
| scale_m2 (0.694444...) | 32 | 15 | 31 | 5.227 | 300.000 | 849.433 | 32 | 34 | 6 | PASS |
| scale_m1 (0.833333...) | 32 | 17 | 32 | 5.463 | 323.843 | 865.232 | 30 | 32 | 8 | PASS |
| scale_p1 (1.2) | 29 | 13 | 29 | 7.121 | 306.223 | 739.351 | 26 | 29 | 11 | FAIL |
| scale_p2 (1.44) | 25 | 11 | 25 | 8.870 | 347.741 | 806.219 | 24 | 27 | 12 | FAIL |
| rot_90 | 28 | 14 | 28 | 5.914 | 364.157 | 869.865 | 29 | 29 | 10 | FAIL |
| rot_180 | 28 | 12 | 28 | 5.917 | 364.530 | 874.439 | 29 | 29 | 10 | FAIL |
| rot_270 | 28 | 14 | 28 | 5.499 | 370.801 | 871.348 | 29 | 29 | 10 | FAIL |

## Predeclared gate outcome

Eligible conditions:
- `scale_m2`
- `scale_m1`

Predeclared promotion rule selected:
`scale_m2`

The full-403 continuation is therefore **scale_m2 only**.  Do not carry both
eligible conditions forward and do not retune the scale after seeing uniform40.

## Main interpretation

### Scale

The downsampled query conditions materially change candidate discrimination.
`scale_m2` improves over baseline on the screening set:

- Top1 containment: 28 -> 32
- Top1 <=80 m: 28 -> 31
- selected projection <=40 m: 29 -> 32
- selected projection <=80 m: 29 -> 34
- selected projection >200 m: 10 -> 6
- RMSE: 366.30 -> 300.00 m

Its catastrophic tail improvement is not merely a count artifact:
5 baseline catastrophic cases are rescued and 1 new catastrophic case is
introduced.

`scale_m1` also passes, but the predeclared promotion rule favors
`scale_m2` because it leaves fewer >200 m selected projections.

### Rotation

The exact quarter-turn conditions produce essentially no locality change:

- Top1 containment stays 28/40 for all rotations;
- Top1 <=80 m stays 28/40;
- selected projection >200 m stays 10/40;
- projection <=40 stays 29/40.

This is evidence against **coarse 90-degree orientation mismatch** being the
dominant explanation for current ORB performance on this screen.  It does not
prove that arbitrary heading/yaw canonicalization is useless.

### Physical-GSD boundary

Do not call `scale_m2` a physical GSD correction yet.

The observed effect may come from one or more of:
- genuine query/map scale alignment;
- better alignment with the ORB pyramid/support scale;
- suppression of high-frequency/repetitive detail;
- changed keypoint density/support geometry after downsampling.

Camera/intrinsic validation is still required before assigning a physical GSD
meaning to the factor.

## Next stage

CV1.3:
run only the frozen `scale_m2 = 0.6944444444444444` transform on all 403
development queries x frozen QV1.4 Top20, **reference-free**.

The existing frozen GD1.0 full403 evidence remains the baseline control and is
not regenerated or retuned.
