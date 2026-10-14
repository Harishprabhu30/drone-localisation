# Canonical handoff — cross-view scale / rotation / domain study v1

Status: **ACTIVE DIRECTION / START HERE IN A NEW CHAT**

Active branch:
`research/crossview-scale-rotation-domain-v1`

Branch base / previous closeout commit:
`b769a8d6b62ae08696aef5d43e03ff6df4a735cf`

Previous branch:
`research/learned-geometric-frontend-ablation-v1`

Previous branch closeout:
`docs/closeouts/learned_geometric_frontend_ablation_v1/README.md`

---

## 1. Main problem has NOT changed

The project is still solving the same absolute localization problem:

> Given a UAV query frame and the frozen QV1.4 Top20 orthophoto candidates, select the correct physical locality reliably enough to produce a precise continuous map position without catastrophic false corrections.

This branch does **not** replace that problem with a new project. It investigates why ORB remains unusually strong and whether explicit cross-view scale/orientation handling or aerial-domain training can raise the verifier/reranker ceiling.

After this branch finishes, results must return to the same frozen Top20 localization stack and then feed the later safe/deployable architecture work.

Do not skip directly to communications, deployment networking, fusion redesign, or onboard/offboard authority while candidate selection/locality remains unresolved.

---

## 2. Frozen system context

Development trajectory:
`villoc_traj01_90deg_stable120m`

Development query count:
403

Frozen candidate pool:
QV1.4 `center_unique_allview_fill20`

Map:
- variant: `768_s256`
- orthophoto source GSD approximately 0.20 m/px
- tile size 768 px
- physical tile footprint approximately 153.6 m
- nominal stride approximately 51.2 m
- EPSG:3346

Frozen QV1.4 development Top20 SHA256:
`0a7832c6f0997a33851343fd7df537d2f6410211ac0e38e7ec4325899d8df47e`

Frozen QV1.4 blind Top20 SHA256:
`68bbf59d3b839cfd90aef50dfca30316cf02296218dd33db236450bbe8b62bd8`

Retrieval availability on development:
- containment in Top20: 384/403
- center <=40 m in Top20: 277/403
- center <=80 m in Top20: 380/403

Therefore coarse retrieval remains sufficiently strong for the current research question. The near-term bottleneck is correct locality/candidate selection inside Top20.

---

## 3. Strongest current local-geometry control: ORB

Frozen ORB:
- CLAHE luma
- resize long edge 1024, downsize only
- nfeatures 1800
- fastThreshold 7
- edgeThreshold 15
- patchSize 31
- OpenCV ORB defaults for scale pyramid retained, including scaleFactor 1.2 and nlevels 8
- Hamming KNN + Lowe ratio 0.80
- query->tile homography, RANSAC 5 px
- minimum 4 inliers
- query-center map projection
- same raster pixel-center convention used throughout the project

GD1.0 development evidence SHA256:
`91ba3381e463bd083f9355b517dbb93953e4423a4b366638c1a97419ea0aea3f`

Full-403 ORB raw-inlier ranking:
- contains_query Top1: 292/403
- center <=40 m Top1: 150/403
- center <=80 m Top1: 287/403

Selected projection:
- median: 6.092 m
- RMSE: 323.219 m
- p95: 843.409 m
- <=40 m: 293/403
- <=80 m: 305/403
- >200 m: 88/403

Oracle finite projection:
- median: 5.298 m
- <=40 m: 336/403
- <=80 m: 370/403

Interpretation:
ORB often estimates precise local geometry once the right locality is selected, but catastrophic wrong-locality aliases remain.

---

## 4. Learned frontend branch — closed evidence

Read:
`docs/closeouts/learned_geometric_frontend_ablation_v1/README.md`

### Sparse XFeat
Rejected.

Full-403 Top1:
- containment 184/403
- <=40 m 91/403
- <=80 m 182/403

### XFeat* semi-dense
Rejected.

Full-403 Top1:
- containment 168/403
- <=40 m 79/403
- <=80 m 162/403

Increasing correspondence density did not improve candidate discrimination.

### SuperPoint + LightGlue
Official cvg/LightGlue checkout pinned to:
`eb42fee2d71449efb0aa5c10549752b5d75384d8`

Uniform40 audit:
- containment: ORB 28/40 vs SuperPoint+LG 18/40
- <=40 m: 13/40 vs 13/40
- <=80 m: 28/40 vs 18/40

Selected projection:
- ORB median 5.416 m, >200 m 10/40
- SuperPoint+LG median 106.198 m, >200 m 19/40

Official maximum-accuracy LightGlue profile (all SuperPoint keypoints, depth/width adaptivity disabled) was slower and weaker on this workload.

### ALIKED + LightGlue
ALIKED improved over SuperPoint but still failed the ORB gate.

LF2.1-C status:
`PASS_LF2_1C_POSTFREEZE_ALIKED_LIGHTGLUE_VS_SUPERPOINT_AND_ORB_AUDIT`

LF2.1-C decision:
`CLOSE_SPARSE_LIGHTGLUE_FAMILY`

LF2.1-C audit diagnostics SHA256:
`67881e6b3d4fe841b3985f4c64a9c24d20b9a2779e596efd6833713db6c547a6`

Uniform40 Top1:
- containment: ORB 28/40, SuperPoint+LG 18/40, ALIKED+LG 21/40
- <=40 m: ORB 13/40, SuperPoint+LG 13/40, ALIKED+LG 13/40
- <=80 m: ORB 28/40, SuperPoint+LG 18/40, ALIKED+LG 20/40

ALIKED selected projection:
- median 107.885 m
- <=40 m 18/40
- <=80 m 19/40
- >200 m 17/40

No ALIKED-native LightGlue score passed the secondary formulation gate.

Do not resume generic sparse XFeat/SuperPoint/ALIKED threshold tuning unless new evidence explicitly reopens it.

---

## 4A. Current branch result after CV1/CV2

CV1 and CV2 have now been completed on this branch.

Strongest current **development** frontend:
\`SIFT + LightGlue\` using the pinned official SIFT-specific LightGlue model.

Full403 result:
- Top1 containment: 349/403
- Top1 center <=40 m: 205/403
- Top1 center <=80 m: 346/403
- selected projection median: 4.988 m
- RMSE: 224.799 m
- p95: 635.898 m
- selected projection <=40 m: 367/403
- selected projection <=80 m: 368/403
- unsafe tail (>200 m or non-finite): 30/403
- median Top1-Top2 inlier margin: 34.0

Frozen SIFT+LightGlue full403 evidence SHA256:
\`c0031242f0b359f1bfa3c87ea47ac428393ed3e8e03d7b5c6d0822e8822cb8a0\`

Decision:
\`PROMOTE_SIFT_LIGHTGLUE_AS_STRONGER_DEVELOPMENT_FRONTEND\`

This supersedes Section 3's wording that ORB is the strongest current
local-geometry control. ORB remains the historical frozen control;
SIFT+LightGlue is now the strongest development frontend.

Next predeclared family:
CV3 aerial-domain model upper bound.


Read the frozen SIFT+LightGlue deep-dive before reopening confidence, Top3
selection, or backend migration:
\`CV2_SIFT_LIGHTGLUE_DEEP_DIVE.md\`

Sections 5 onward preserve original branch hypotheses/planning history. CV1 and
CV2 are complete; do not restart them. The authoritative next-stage contract is
\`CV3_0_PROTOCOL.md\`.

---

## 5. Current causal hypotheses

These are **hypotheses**, not proven explanations.

### H1 — explicit orientation handling
ORB may benefit from explicit local orientation canonicalization when UAV heading changes relative to the north-up orthophoto.

Important wording:
ORB does not understand forward/backward flight. The relevant variable is image-plane orientation/yaw between query and map.

### H2 — explicit multi-scale support
ORB's 8-level image pyramid may compensate for query/map ground-scale mismatch.

### H3 — learned sparse domain gap
Generic pretrained local features/matchers are being applied zero-shot outside their dominant training distribution. The partial ALIKED rescue suggests feature representation/support geometry matters.

### H4 — alias discrimination, not homography existence
ORB, XFeat*, SuperPoint+LG and ALIKED+LG often produce valid homographies for wrong candidates. Homography existence alone is nearly saturated and is not a useful selector.

---

## 6. Critical unfinished experiment: true UAV-to-map GSD normalization

Earlier retrieval research changed **map physical footprint represented per network input while keeping orthophoto source GSD fixed**.

That is NOT the same as a controlled per-query experiment where the UAV frame is resampled so that its effective ground sampling distance matches the orthophoto/map representation.

Therefore do not write that GSD normalization was already solved.

New branch must explicitly determine:
1. what UAV GSD can be estimated from available non-reference camera/altitude information;
2. whether scale mismatch explains part of ORB's advantage;
3. whether physically justified resampling improves learned or classical verification;
4. whether any scale normalization remains causal/deployable under the blind boundary.

For the stable120m development trajectory, altitude is nearly constant, which may allow a simple controlled first experiment before per-frame scaling.

---

## 7. New research direction

The branch should answer:

> Which invariances and training-domain properties are actually required to discriminate the correct UAV-to-orthophoto locality inside frozen Top20?

Recommended sequence:

### CV1 — causal scale/orientation diagnostics
Do this before another long matcher benchmark.

Measure performance versus:
- relative image-plane rotation / available yaw proxy
- estimated UAV/map ground-scale ratio
- scene/content regime
- retrieval rank

Run scale and rotation corrections separately first.

Do not combine them initially; otherwise causal interpretation is lost.

Potential controlled transformations:
- physically justified UAV/map scale normalization
- north-up/yaw canonicalization where the input is deployably available
- fixed 0/90/180/270 rotation diagnostic only if framed explicitly as an offline diagnostic, not a deployable oracle

Main output:
determine whether ORB's advantage grows with rotation and/or scale mismatch.

### CV2 — SIFT + LightGlue diagnostic
Reason:
SIFT has explicit scale and orientation estimates, and the pinned LightGlue implementation has a SIFT-specific model that consumes scale/orientation information.

Purpose:
test the hypothesis that explicit scale+orientation support, rather than 'classical vs learned', explains part of ORB's advantage.

Use the same frozen uniform40 screen before any full403 run.

### CV3 — aerial-domain model upper bound
AerialMegaDepth is relevant here.

Project:
https://aerial-megadepth.github.io/

Paper:
https://arxiv.org/abs/2504.13157

The paper argues that lack of co-registered aerial-ground training data is a key reason generic learning-based reconstruction/matching systems fail under extreme cross-view variation, and reports large gains after aerial-ground fine-tuning.

Our UAV-nadir-to-orthophoto task is NOT identical to aerial-ground reconstruction, so do not assume direct transfer.

First use:
- test a published aerial-specific MASt3R checkpoint/model as an expensive pairwise verifier/locality upper bound on the same frozen Top20 or uniform40 screen;
- ask whether aerial-domain training changes candidate discrimination where generic pretrained sparse methods failed.

Do NOT download or ingest the full AerialMegaDepth training dataset before the pretrained-model upper-bound experiment justifies it.

Potential later use:
- domain-specific fine-tuning/training resource for a lighter verifier if the aerial-specific upper bound materially beats ORB.

### CV4 — aerial-specific local features
Candidate:
RDD / RDD+LightGlue or another feature explicitly trained for air-to-ground / large scale-viewpoint changes.

Treat this as a separate family; do not blend it with generic LF2 conclusions.

### CV5 — detector-free/semi-dense upper bounds
Priority:
1. EfficientLoFTR / LoFTR-class matcher
2. RoMa as a stronger dense robustness upper bound

Question:
can detector-free matching resolve correct locality even when repeatable sparse keypoints fail?

RoMa should initially be treated as an accuracy upper bound, not an edge-deployment choice.

---

## 8. AerialMegaDepth placement in the architecture

AerialMegaDepth should NOT replace QV1.4 retrieval at this stage.

Preferred placement:

```
UAV query
   |
   v
Frozen QV1.4 Top20 retrieval
   |
   +--> ORB verifier / control
   |
   +--> aerial-domain verifier upper bound
        (e.g. aerial-MASt3R / model inspired by AerialMegaDepth)
   |
   v
candidate/locality score
   |
   v
continuous geometric projection
   |
   v
later temporal/state authority
```

Possible roles:
1. pairwise verifier score
2. candidate reranker
3. candidate rejection/uncertainty signal
4. dense correspondences for continuous map projection
5. training source for a future lighter verifier

Do not use it as a learned selector trained on development labels until a predeclared split/OOF protocol exists.

---

## 9. ORB + LightGlue clarification

Do not feed ORB BRIEF descriptors into the official pretrained LightGlue models.

The official pretrained LightGlue checkpoints are tied to supported descriptor families/dimensions. ORB's binary BRIEF descriptor space is incompatible without retraining/adaptation.

Possible later hybrid:
- keep ORB as one independent verifier
- combine ORB evidence with an aerial-specific/detector-free verifier through a separately frozen selection/fusion stage

Do not call such a score fusion 'ORB+LightGlue' unless LightGlue is actually retrained for ORB descriptors.

---

## 10. Geometry / triangulation boundary

Do not jump to triangulation as the next fix.

Current bottleneck:
wrong locality/candidate selection inside Top20.

For a single near-nadir UAV frame against an orthophoto, homography/query-center projection remains a reasonable local planar model.

Triangulation / bundle adjustment / PnP become relevant later when:
- multiple UAV frames are jointly optimized,
- real 3D map/DEM/landmark geometry is used,
- or selected correspondences are already trustworthy enough to justify multi-view pose optimization.

Fix locality selection first.

---

## 11. Deployment work remains downstream, not forgotten

After the cross-view direction is closed and a reliable absolute frontend is selected, return to the main pipeline and address:

- confidence / acceptance authority
- temporal independence and persistence
- relative-to-absolute fusion
- fallback behavior when absolute localization disappears
- onboard vs offboard compute split
- Jetson runtime
- map caching
- Ethernet / Wi-Fi / 4G / 5G communication choices
- latency / bandwidth budgets
- telemetry / message schemas
- watchdog / safety state machine
- offline/no-network behavior
- deployable debugging and evidence logging

Do not design these around a matcher that has not yet passed the localization gate.

---

## 12. Required reading order for a new model/session

Read these before proposing code:

1. `docs/handoffs/crossview_scale_rotation_domain_v1/README.md` — this file
2. `docs/closeouts/learned_geometric_frontend_ablation_v1/README.md`
3. `docs/closeouts/learned_geometric_frontend_lf2_1c/README.md`
4. `docs/closeouts/top20_geometric_discrimination_gd1_3b/README.md`
5. `docs/handoffs/retrieval_candidate_pool_v2/README.md`
6. `docs/README_villoc_traj01_s8_full_pipeline_fusion_closeout.md`
7. `docs/demo/blind_recorded_flight_final_001/README.md`

Then inspect exact configs/scripts only for the stage being modified.

Do not reconstruct numbers from memory when a frozen report/hash exists.

---

## 13. Immediate next action

Do NOT start CV2/SIFT, AerialMegaDepth, LoFTR, RoMa, communications or deployment immediately.

First stage on this branch should be a small **CV1.0 direction/protocol freeze** that defines:
- exact diagnostic subset
- allowed/non-allowed metadata
- GSD estimation method
- rotation/yaw source and blind/development boundary
- scale normalization transforms
- rotation normalization transforms
- ORB control metrics
- no-reference freeze before truth attachment
- explicit stop/continue gates

Only after that protocol is frozen should implementation begin.

