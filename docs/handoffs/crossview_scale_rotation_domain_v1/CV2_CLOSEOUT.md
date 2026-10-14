# CV2 closeout — SIFT + LightGlue explicit scale/orientation frontend

Status: **PASS / PROMOTED AS STRONGER DEVELOPMENT FRONTEND**

Decision:
\`PROMOTE_SIFT_LIGHTGLUE_AS_STRONGER_DEVELOPMENT_FRONTEND\`

## Frozen full403 result

CV2.4 reference-free evidence SHA256:
\`c0031242f0b359f1bfa3c87ea47ac428393ed3e8e03d7b5c6d0822e8822cb8a0\`

CV2.5 post-freeze comparison:

| frontend | Top1 contains | Top1 <=40 | Top1 <=80 | projection median m | RMSE m | p95 m | projection <=40 | projection <=80 | unsafe tail | oracle <=40 | oracle <=80 | median Top1-Top2 margin |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| original ORB | 292 | 150 | 287 | 6.092135 | 323.218571 | 843.408949 | 293 | 305 | 88 | 336 | 370 | 2.0 |
| scale_m2 ORB | 305 | 163 | 293 | 5.839836 | 282.572974 | 841.930929 | 319 | 331 | 63 | 347 | 371 | 3.0 |
| SIFT + LightGlue | 349 | 205 | 346 | 4.988110 | 224.799107 | 635.898445 | 367 | 368 | 30 | 386 | 392 | 34.0 |

SIFT+LightGlue vs scale_m2 ORB:
- winner changed: 193 / 403
- containment rescues / damages: 60 / 16
- center <=80 rescues / damages: 68 / 15
- unsafe-tail rescues / damages: 43 / 10

Validation gate:
\`PASS\`

Meaningful-gain details:
- Top1 containment: +44
- Top1 <=40 m: +42
- Top1 <=80 m: +53
- selected projection <=40 m: +48
- unsafe tail: 63 -> 30 (-33)

Scientific boundary:
- no scale_m2 preprocessing applied to SIFT;
- no threshold tuning;
- no Jetson/CUDA evidence used;
- frozen CPU evidence only;
- development truth attached only after reference-free freeze.

## Interpretation

CV2 establishes that the earlier SuperPoint/ALIKED LightGlue failures did not
close the entire sparse-LightGlue family. The SIFT-specific representation is
materially different because it provides explicit local scale/orientation to the
official SIFT LightGlue model.

The improvement is large enough to promote SIFT+LightGlue as the strongest
current **development** absolute frontend.

However, it is not yet a safe deployment endpoint:
- 30 / 403 selected queries remain in the unsafe tail (>200 m or non-finite);
- full403 remains one development trajectory;
- no blind-stress / independent-dataset confirmation exists for this frontend;
- CPU runtime is approximately 20,993 s wall time for the 8,060-pair evidence run.

Therefore:
1. do not return to deployment authority yet;
2. do not tune SIFT/LightGlue further on this trajectory;
3. proceed to the predeclared aerial-domain upper-bound experiment;
4. park Jetson/CUDA migration as a high-value runtime task if this or the next
   frontend remains a leading accuracy candidate.

## Next stage

CV3:
pretrained aerial-domain MASt3R upper bound from AerialMegaDepth.

The goal is not to replace QV1.4 retrieval. It is to ask whether aerial-domain
training further reduces wrong-locality selection inside the same frozen Top20.

Do not download the full AerialMegaDepth training dataset for CV3.


## Deep-dive record

For Top-k behavior, oracle ceiling, selected-error bins, residual selector gap,
runtime implications and future confidence/abstention clues, read:

\`docs/handoffs/crossview_scale_rotation_domain_v1/CV2_SIFT_LIGHTGLUE_DEEP_DIVE.md\`

This note is derived from the frozen CV2.5 report and does not reopen tuning.
