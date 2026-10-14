# CV2 SIFT + LightGlue full403 deep-dive

Status: **FROZEN INTERPRETATION NOTE**

Primary source:
\`outputs/research_runs/crossview_scale_rotation_domain_v1/cv2/cv2_5/cv2_5_validation_report.json\`

Report status:
\`PASS_CV2_5_POSTFREEZE_SIFT_LIGHTGLUE_FULL403_VALIDATION\`

Decision:
\`PROMOTE_SIFT_LIGHTGLUE_AS_STRONGER_DEVELOPMENT_FRONTEND\`

## Provenance and boundary

Frozen evidence SHA256:
- original ORB:
  \`91ba3381e463bd083f9355b517dbb93953e4423a4b366638c1a97419ea0aea3f\`
- scale_m2 ORB:
  \`94353f763725f44f278192bcedcfd139ab136ecebf4f1dcc6e07487380964810\`
- SIFT+LightGlue:
  \`c0031242f0b359f1bfa3c87ea47ac428393ed3e8e03d7b5c6d0822e8822cb8a0\`
- reference attachment:
  \`0021dfef67083360bc2b555466dc524afe9e4b7a5e4de72cc5f361c3caf26e59\`
- tile geometry:
  \`01b7aa6f47a1e95dff7da2fc34baa2c6bda660ead436fc0ad9abb48acdf3cb5c\`
- CV2.5 config:
  \`91a146e7581709aeac29aa8635a431228d325d9a2bc99e77d90d511a6c130932\`

Scientific boundary:
- reference attached only after reference-free freezes;
- no scale_m2 preprocessing applied to SIFT;
- no threshold tuning;
- no selector formula change;
- no Jetson/CUDA evidence;
- no state/fusion change.

## Full403 primary comparison

| frontend | Top1 contains | Top1 <=40 | Top1 <=80 | selected <=40 | selected <=80 | unsafe tail | selected median m | RMSE m | p95 m | median Top1-Top2 margin |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| original ORB | 292 | 150 | 287 | 293 | 305 | 88 | 6.092 | 323.219 | 843.409 | 2.0 |
| scale_m2 ORB | 305 | 163 | 293 | 319 | 331 | 63 | 5.840 | 282.573 | 841.931 | 3.0 |
| SIFT+LightGlue | 349 | 205 | 346 | 367 | 368 | 30 | 4.988 | 224.799 | 635.898 | 34.0 |

Versus scale_m2 ORB, SIFT+LightGlue changes the selected winner on 193/403
queries and yields:
- containment rescues/damages: 60/16;
- center<=80 rescues/damages: 68/15;
- unsafe-tail rescues/damages: 43/10.

This is not a marginal tie-break effect.

## Top-k behavior

### Original ORB

| target | Top1 | Top3 | Top5 | Top10 | Top20 |
|---|---:|---:|---:|---:|---:|
| contains query | 292 | 335 | 349 | 370 | 384 |
| center <=40 | 150 | 226 | 249 | 272 | 277 |
| center <=80 | 287 | 333 | 346 | 368 | 380 |

### scale_m2 ORB

| target | Top1 | Top3 | Top5 | Top10 | Top20 |
|---|---:|---:|---:|---:|---:|
| contains query | 305 | 350 | 360 | 375 | 384 |
| center <=40 | 163 | 248 | 259 | 267 | 277 |
| center <=80 | 293 | 347 | 357 | 368 | 380 |

### SIFT+LightGlue

| target | Top1 | Top3 | Top5 | Top10 | Top20 |
|---|---:|---:|---:|---:|---:|
| contains query | 349 | 362 | 368 | 371 | 384 |
| center <=40 | 205 | 267 | 271 | 273 | 277 |
| center <=80 | 346 | 359 | 364 | 367 | 380 |

Important interpretation:
- SIFT+LightGlue is strongest at **top-end concentration**, especially Top1.
- It is not strictly superior at every larger k; e.g. containment Top10 is
  371 versus 375 for scale_m2 ORB.
- center<=40 Top3 is 267/277 available Top20 cases (~96.4%).

The Top3 headroom is a future research clue only. It does not authorize a new
selector without a separate reference-free/OOF protocol.

## Oracle geometric ceiling

| frontend | oracle median m | oracle RMSE m | oracle p95 m | oracle <=40 | oracle <=80 | oracle unsafe |
|---|---:|---:|---:|---:|---:|---:|
| original ORB | 5.298 | 47.277 | 114.816 | 336 | 370 | 4 |
| scale_m2 ORB | 5.394 | 43.910 | 97.600 | 347 | 371 | 3 |
| SIFT+LightGlue | 4.655 | 37.436 | 34.446 | 386 | 392 | 5 |

This is a major second strength of SIFT+LightGlue: it improves both candidate
ranking and the candidate-level geometric ceiling.

The oracle is truth-based and non-deployable.

## Residual selection gap

SIFT+LightGlue:
- selected unsafe tail: 30/403;
- oracle unsafe tail: 5/403.

There is therefore a 25-query selected-vs-oracle unsafe gap.

Do not claim those 25 are recoverable by a simple threshold. The correct
conclusion is that the frozen pair evidence contains more useful geometric
information than the current raw-inlier selector extracts.

## Strongly bimodal selected projection behavior

SIFT+LightGlue selected projection:
- <=40 m: 367/403;
- (40,80] m: 1/403;
- (80,200] m: 5/403;
- >200 m: 30/403;
- non-finite: 0/403.

The system is usually very accurate, but residual failures are often
catastrophic. Future safety work should prioritize wrong-locality recognition,
abstention and independent/temporal evidence rather than optimizing the already
good ~5 m median.

## Homography completeness

CV2.4 reference-free pair evidence:
- homography OK: 7382/8060;
- finite projection: 7382/8060;
- queries with all 20 homographies: 230/403;
- queries with no homography: 0/403.

CV2.5 selected result:
- finite selected projection: 403/403;
- selected non-finite: 0/403.

Thus failed candidate-pair homographies do not create selected-output dropout,
but homography existence still does not guarantee correctness.

## Inlier-margin interpretation

Median Top1-Top2 raw-inlier margin:
- original ORB: 2;
- scale_m2 ORB: 3;
- SIFT+LightGlue: 34.

This is evidence of stronger SIFT candidate separation. It is **not** a
cross-frontend calibrated confidence score. Any future inlier-margin threshold
must be calibrated under a separate OOF/blind-safe protocol.

## Runtime record

CV2.4 CPU full403:
- 8060 pairs;
- median pair runtime: 1881.012 ms;
- wall time: 20993.051 s (~5.83 h);
- effective wall time per candidate pair: ~2.60 s including extraction/cache/
  orchestration overhead.

GPU migration is high-value if SIFT+LightGlue remains the final accuracy
frontend, but CPU/CUDA parity must be proven before mixing CUDA results with the
frozen CPU accuracy claim.

## Future SIFT+LightGlue questions worth reopening later

These are research ideas, not currently authorized stages:

- confidence/abstention audit using SIFT inlier margin, inlier ratio, LightGlue
  confidence, spatial support, reprojection residual, retrieval agreement,
  scale_m2 ORB disagreement and temporal consistency;
- a separately frozen Top3 secondary selector/refinement study;
- CPU-SIFT + CUDA-LightGlue parity and Jetson runtime benchmarking;
- blind-stress / independent-trajectory validation.

## Current scientific takeaway

CV2 changes the project interpretation:

- the earlier learned sparse failures did not mean learned matching is
  intrinsically weak for this problem;
- explicit SIFT scale/orientation plus the SIFT-specific LightGlue checkpoint is
  a strong cross-view representation;
- SIFT materially improves both locality discrimination and oracle geometric
  quality;
- the residual bottleneck is increasingly unsafe locality selection/confidence;
- aerial-domain training remains worth testing because 30 catastrophic selected
  cases remain and the SIFT oracle shows substantial unused geometric headroom.
