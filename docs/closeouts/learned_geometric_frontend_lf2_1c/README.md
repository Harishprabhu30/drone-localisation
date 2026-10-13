# LF2.1-C closeout — ALIKED + LightGlue screening audit

Status: **PASS / CLOSED** on user-reported local execution.

Final stage status:
`PASS_LF2_1C_POSTFREEZE_ALIKED_LIGHTGLUE_VS_SUPERPOINT_AND_ORB_AUDIT`

Decision:
`CLOSE_SPARSE_LIGHTGLUE_FAMILY`

Audit diagnostics SHA256:
`67881e6b3d4fe841b3985f4c64a9c24d20b9a2779e596efd6833713db6c547a6`

## Frozen 40-query result

All methods were evaluated on the same deterministic query IDs:
`1,11,22,32,42,53,63,73,83,94,104,114,125,135,145,156,166,176,187,197,207,217,228,238,248,259,269,279,290,300,310,321,331,341,351,362,372,382,393,403`.

Top1 candidate ranking:

| target | ORB | SuperPoint+LightGlue | ALIKED+LightGlue |
|---|---:|---:|---:|
| contains_query | 28/40 | 18/40 | 21/40 |
| center <=40 m | 13/40 | 13/40 | 13/40 |
| center <=80 m | 28/40 | 18/40 | 20/40 |

ALIKED improved over SuperPoint on containment (+3) and <=80 m (+2), but remained substantially below ORB (-7 and -8 respectively).

Selected continuous projection:

| frontend | median m | RMSE m | p95 m | <=40 | <=80 | >200 |
|---|---:|---:|---:|---:|---:|---:|
| ORB | 5.416 | 366.298 | 872.632 | 29/40 | 29/40 | 10/40 |
| SuperPoint+LightGlue | 106.198 | 471.489 | 778.732 | 16/40 | 19/40 | 19/40 |
| ALIKED+LightGlue | 107.885 | 378.212 | 780.820 | 18/40 | 19/40 | 17/40 |

ALIKED native LightGlue scores did not pass the secondary formulation gate. Raw inliers remained the best simple ALIKED selector among the tested native signals, but were still materially weaker than ORB.

## Interpretation

1. The ALIKED extractor partially repairs the SuperPoint failure, so extractor/domain representation matters.
2. Pretrained sparse LightGlue with the tested generic feature families does not close the absolute candidate-discrimination gap.
3. Homography existence is nearly saturated for these frontends and is not a useful discriminator by itself.
4. ORB remains the strongest tested local geometric discriminator for the frozen QV1.4 Top20 problem.
5. The next research question is no longer 'which generic sparse learned frontend beats ORB?'. It is 'which cross-view invariances and training-domain properties make ORB strong, and can aerial-specific or detector-free matching exploit them better?'.

## Boundary

No blind-stress reference was loaded. No full 403-query ALIKED+LightGlue run was executed. No learned frontend was promoted into the deployment pipeline.
