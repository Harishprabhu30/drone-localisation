# CV1.4 full403 post-freeze scale_m2 validation closeout

Status: **PASS_CV1_4_POSTFREEZE_FULL403_SCALE_VALIDATION**

Decision:
`CONTINUE_SCALE_MECHANISM_AND_CAMERA_GSD_AUDIT`

## Frozen comparison

| condition | Top1 contains | Top1 <=40 | Top1 <=80 | projection median m | RMSE m | p95 m | projection <=40 | projection <=80 | projection >200 | oracle median m | oracle <=40 | oracle <=80 | median Top1-Top2 inlier margin |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 292 | 150 | 287 | 6.092135 | 323.218571 | 843.408949 | 293 | 305 | 88 | 5.298163 | 336 | 370 | 2.0 |
| scale_m2 | 305 | 163 | 293 | 5.839836 | 282.572974 | 841.930929 | 319 | 331 | 63 | 5.394470 | 347 | 371 | 3.0 |

Winner changed:
`170 / 403`

Rescue / damage counts:
- containment: 37 / 24
- center <=80 m: 35 / 29
- selected projection >200 m: 37 / 12

Replication gate:
`PASS`

Scientific boundary:
- SRT loaded: false
- transform retuned: false
- threshold tuned: false

CV1.3 scale evidence SHA256:
`94353f763725f44f278192bcedcfd139ab136ecebf4f1dcc6e07487380964810`

## Interpretation

The uniform40 scale effect replicated on the complete 403-query development
trajectory.

Relative to frozen ORB, the fixed `scale_m2` treatment:
- raises Top1 containment by 13 queries;
- raises Top1 center <=40 by 13;
- raises Top1 center <=80 by 6;
- raises selected projection <=40 by 26;
- raises selected projection <=80 by 26;
- reduces selected projection >200 by 25;
- lowers RMSE by about 40.65 m;
- increases median Top1-Top2 inlier margin from 2 to 3.

The selected winner changes for 170/403 queries, so the transform is materially
changing candidate discrimination rather than merely perturbing a handful of
homographies.

The oracle-best finite projection changes much less than selected projection.
This suggests the main gain is likely in candidate/locality discrimination,
although CV1.5 must explicitly separate winner-selection effects from
same-winner geometric effects.

Do not interpret `0.694444...` as calibrated UAV/map GSD yet.  CV1.5 audits:
1. pair/query-level ORB mechanism changes;
2. same-winner versus changed-winner rescue pathways;
3. camera/SRT metadata constancy and physical-GSD feasibility.
