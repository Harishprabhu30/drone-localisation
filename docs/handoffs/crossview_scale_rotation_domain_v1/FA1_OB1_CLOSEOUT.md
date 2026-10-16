# FA1 + OB1 closeout — residual failure analysis and oblique stress

**Project:** GNSS-denied / weak-GNSS UAV visual localization  
**Branch:** research/crossview-scale-rotation-domain-v1  
**Status:** CLOSED FOR THIS RESEARCH BLOCK  
**Date:** 2026-10-16

This document closes the research block that began after the cross-view scale /
rotation investigation and ended with:

- promotion of SIFT + LightGlue as the strongest practical absolute geometric
  frontend on the 403-query nadir development trajectory;
- compact aerial-MASt3R upper-bound screening;
- blind transfer of SIFT + LightGlue into the recorded-flight bootstrap;
- explicit analysis of the remaining SIFT failure tail;
- a current-stack 45-degree oblique stress run.

It is a decision record. It explains what was tested, why it was tested, what
the evidence showed, what hypotheses were rejected, and which questions are
still open.

---

## 1. Scientific problem that remained after the earlier pipeline

The main absolute-localization problem has remained consistent:

> Given a UAV query image and a shortlist of georeferenced orthophoto candidates,
> identify the correct physical locality reliably enough to produce an absolute
> map observation without catastrophic false correction.

By the start of this branch, coarse DINOv2 retrieval was already strong enough
that the main problem had shifted away from "is the correct area anywhere in
Top20?" toward:

> "Can the geometric frontend and authority logic correctly discriminate the
> useful locality inside Top20 and know when not to trust itself?"

That distinction is important throughout this closeout.

---

## 2. Current development dataset and map contract

Primary development trajectory:

- villoc_traj01_90deg_stable120m
- 403 queries at 1 Hz
- near-nadir
- approximately 120 m relative altitude
- approximately 1.96 km trajectory

Current research map:

- ORT10LT orthophoto
- CRS EPSG:3346 / LKS-94 Lithuania TM
- variant 768_s256
- source GSD approximately 0.20 m/px
- tile side approximately 153.6 m
- map-tile stride approximately 51.2 m

Frozen QV1.4 candidate pool:

- center_unique_allview_fill20
- 20 candidates/query
- 8,060 development query-candidate pairs
- development Top20 SHA256:
  0a7832c6f0997a33851343fd7df537d2f6410211ac0e38e7ec4325899d8df47e

Development candidate availability:

- containment in Top20: 384 / 403
- center <=40 m in Top20: 277 / 403
- center <=80 m in Top20: 380 / 403

The current research conclusion is therefore not that retrieval is perfect.
Rather, retrieval is sufficiently strong that candidate discrimination and
trust authority are the dominant next problems for many frames.

---

## 3. Geometry frontend progression

### 3.1 Original ORB

Frozen ORB control:

- CLAHE luma
- long edge 1024, downsize only
- nfeatures 1800
- fastThreshold 7
- edgeThreshold 15
- patchSize 31
- scaleFactor 1.2
- nlevels 8
- Hamming KNN
- Lowe ratio 0.80
- query-to-tile homography
- RANSAC 5 px
- minimum 4 inliers
- query-center projection

Full403 result:

| Metric | Original ORB |
|---|---:|
| Top1 containment | 292 / 403 |
| Top1 center <=40 m | 150 / 403 |
| Top1 center <=80 m | 287 / 403 |
| Selected projection <=40 m | 293 / 403 |
| Selected projection <=80 m | 305 / 403 |
| Unsafe tail | 88 / 403 |
| Median selected projection | 6.092 m |
| RMSE | 323.219 m |
| p95 | 843.409 m |

Evidence SHA256:

91ba3381e463bd083f9355b517dbb93953e4423a4b366638c1a97419ea0aea3f

### 3.2 scale_m2 ORB

The best frozen scale control used scale_m2 = 0.694444...

Full403 result:

| Metric | scale_m2 ORB |
|---|---:|
| Top1 containment | 305 / 403 |
| Top1 center <=40 m | 163 / 403 |
| Top1 center <=80 m | 293 / 403 |
| Selected projection <=40 m | 319 / 403 |
| Selected projection <=80 m | 331 / 403 |
| Unsafe tail | 63 / 403 |
| Median selected projection | 5.840 m |
| RMSE | 282.573 m |
| p95 | 841.931 m |

Evidence SHA256:

94353f763725f44f278192bcedcfd139ab136ecebf4f1dcc6e07487380964810

Interpretation:

Scale matters materially. The improvement was driven more by selecting the
correct locality than by tiny sub-tile projection refinements.

### 3.3 Earlier learned sparse frontend screens

Before SIFT-specific LightGlue, several learned sparse frontends were screened.

- XFeat full403 Top1 containment: 184 / 403
- XFeat full403 Top1 center <=40 m: 91 / 403
- XFeat* semi-dense Top1 containment: 168 / 403
- XFeat* Top1 center <=40 m: 79 / 403

Uniform40 screens also showed SuperPoint + LightGlue and ALIKED + LightGlue
below the ORB control. Those experiments closed the generic sparse-LightGlue
family for that representation, but did **not** prove that every LightGlue
feature family was unsuitable.

### 3.4 SIFT + LightGlue

The key representation change was explicit SIFT scale and orientation support.

Pinned implementation:

- official LightGlue commit:
  eb42fee2d71449efb0aa5c10549752b5d75384d8
- SIFT source blob:
  802fc1c2eb9ee852691e0e4dd67455f822f8405f
- LightGlue source blob:
  485833515eb0a80866b3d078d3e43d39ddcd84ae
- weights SHA256:
  5b52b8d9982d43532dc042606b346bb9594c9f5a4bd6f64362c63866287b4ac0
- OpenCV SIFT
- RootSIFT enabled
- max keypoints 4096
- resize long edge 1024
- NMS radius 0
- detection threshold 0.0066667
- edge threshold 10
- octaves 4
- LightGlue filter threshold 0.1
- depth confidence 0.95
- width confidence 0.99
- homography RANSAC 5 px
- minimum 4 inliers
- CPU frozen evidence

Frozen full403 evidence SHA256:

c0031242f0b359f1bfa3c87ea47ac428393ed3e8e03d7b5c6d0822e8822cb8a0

Full403 result:

| Metric | SIFT + LightGlue |
|---|---:|
| Top1 containment | 349 / 403 |
| Top1 center <=40 m | 205 / 403 |
| Top1 center <=80 m | 346 / 403 |
| Selected projection <=40 m | 367 / 403 |
| Selected projection <=80 m | 368 / 403 |
| Unsafe tail | 30 / 403 |
| Median selected projection | 4.988 m |
| RMSE | 224.799 m |
| p95 | 635.898 m |
| Oracle projection <=40 m | 386 / 403 |
| Oracle projection <=80 m | 392 / 403 |

Versus scale_m2 ORB:

- containment: +44
- Top1 center <=40 m: +42
- Top1 center <=80 m: +53
- selected projection <=40 m: +48
- unsafe tail: 63 -> 30
- catastrophic rescues: 43
- catastrophic damages: 10

Decision:

**PROMOTE_SIFT_LIGHTGLUE_AS_STRONGER_DEVELOPMENT_FRONTEND**

### 3.5 Aerial-MASt3R compact upper bound

The full 688,638,856-parameter FP32 Aerial-MASt3R model was tested on the exact
same compact six-candidate screen as SIFT.

On 40 queries x 6 candidate identities:

- SIFT containment: 26 / 40
- MASt3R containment: 27 / 40
- SIFT selected projection <=40 m: 26 / 40
- MASt3R selected projection <=40 m: 28 / 40
- SIFT unsafe: 13
- MASt3R unsafe: 12

The gain was directionally positive but below the predeclared promotion gate.

Decision:

**CLOSE_CV3_AERIAL_MAST3R_COMPACT_SCREEN**

Do not describe this as "MASt3R failed." The correct statement is that it did
not justify full Top20 expansion relative to complexity/runtime.

---

## 4. Visualization correction before failure analysis

An early frontend-comparison figure connected every independent absolute
estimate in query order. A single wrong-locality estimate therefore created a
long line to a distant map region and back, producing a misleading
"spiderweb."

The corrected visualization:

- keeps the reference path connected;
- plots absolute frontend outputs as independent points;
- does not imply that the framewise selected positions are a fused trajectory.

Corrected local output:

outputs/research_runs/crossview_scale_rotation_domain_v1/cv2/cv2_5/visual_diagnostics/traj01_90deg_stable120m_frontend_selected_scatter_ortho.png

The ORB->SIFT comparison was also changed from overlapping rescue/damage labels
to mutually exclusive error transitions.

Exact transition matrix:

| scale_m2 ORB -> SIFT | SIFT <=40 | SIFT 40-200 | SIFT unsafe |
|---|---:|---:|---:|
| ORB <=40 | 308 | 4 | 7 |
| ORB 40-200 | 18 | 0 | 3 |
| ORB unsafe | 41 | 2 | 20 |

Therefore:

- ORB-good cases lost by SIFT: 11
- severe ORB<=40 -> SIFT unsafe: 7
- catastrophic SIFT damage total: 10
- catastrophic ORB-unsafe rescues: 43

Corrected transition visual:

outputs/research_runs/crossview_scale_rotation_domain_v1/cv2/cv2_5/visual_diagnostics/traj01_90deg_stable120m_sift_vs_scale_m2_selected_points_ortho.png

---

## 5. FA1.0 — frozen residual-failure inventory

Status:

**PASS_FA1_0_FROZEN_SIFT_FAILURE_AUDIT**

The failure-analysis stage deliberately did not tune the frontend.

Cases:

- oracle-rescuable SIFT failures: 19
- catastrophic SIFT damages: 10
- severe ORB<=40 -> SIFT unsafe: 7
- union: 20 unique queries
- intersection of oracle-rescuable and catastrophic-damage: 9
- oracle-rescuable only: 10
- catastrophic-damage only: 1

For the 19 oracle-rescuable cases:

- useful candidate selector rank median: 11
- useful candidate selector rank max: 17
- useful candidate DINO pool-rank median: 15
- selected minus oracle inliers median: +1
- Top1-Top2 inlier margin median: 0

This was the first strong clue that the remaining failures were often
near-ties rather than obviously stronger wrong matches.

Outputs:

- fa1_0_case_inventory.csv
- fa1_0_case_top20_candidate_detail.csv
- fa1_0_temporal_context_pm2.csv
- fa1_0_report.json

under:

outputs/research_runs/crossview_scale_rotation_domain_v1/fa1_0/

---

## 6. FA1.1 — targeted geometric diagnostics

Status:

**PASS_FA1_1_TARGETED_PAIR_GEOMETRY_DIAGNOSTICS**

Scope:

- 20 union queries
- 40 SIFT selected/oracle diagnostic pairs
- 7 scale_m2 ORB severe-regression pairs
- 198 exported SIFT inlier correspondences
- 40 correspondence overlays

Frozen parity:

- SIFT parity: true
- scale_m2 ORB parity: true

The targeted rerun reproduced frozen match counts, inlier counts, homography
state, and projected position before accepting any diagnostic evidence.

Selected minus oracle median differences across the 19 oracle-rescuable cases:

| Diagnostic | Median selected - oracle |
|---|---:|
| Query hull coverage | +0.0403 |
| Tile hull coverage | +0.0220 |
| Query grid entropy | -0.0195 |
| Tile grid entropy | -0.0195 |
| Query PCA minor/major | +0.0176 |
| Tile PCA minor/major | +0.0347 |
| Inlier ratio | **-0.1494** |
| Symmetric reprojection median | **+3.505 px** |
| Jacobian anisotropy | -0.0212 |
| Footprint-area fraction | +0.0712 |
| Orientation resultant | -0.00034 |
| log-scale-ratio MAD | +0.0446 |

Important negative result:

> Spatially broader matching is not sufficient evidence of correctness.

Wrong vegetation matches can be distributed across a large fraction of the
image and still correspond to the wrong physical locality.

---

## 7. FA1.2 — visual / semantic review

FA1.2 generated:

- one 6-panel montage for each of the 20 residual failure queries;
- a 20-case contact sheet;
- a structured manual review worksheet.

Local contact sheet:

outputs/research_runs/crossview_scale_rotation_domain_v1/fa1_2/fa1_2_failure_contact_sheet.png

Representative reviewed cases:

- q280
- q289
- q370
- q384

Each montage contains:

1. UAV query;
2. SIFT-selected tile;
3. SIFT oracle tile;
4. scale_m2 ORB selected tile;
5. selected SIFT inlier overlay;
6. oracle SIFT inlier overlay.

Manual observation:

> The residual failure set is predominantly vegetation-heavy or mixed
> vegetation with limited distinctive man-made structure. Dense street-heavy or
> dense-urban failures were not prominent in the reviewed set.

This remains a qualitative observation. It is **not** an automated
scene-classification result and must not be generalized to "SIFT fails on
vegetation."

Observed mechanism:

~~~text
vegetation-dominant query
        |
        v
many visually similar canopy regions
        |
        v
wrong candidate obtains 5-6 plausible correspondences
        |
        +--> sometimes broad spatial coverage
        |
correct candidate may have only 4-5 correspondences
but better geometric efficiency / locality
        |
        v
raw-inlier selector wins by 0-2 inliers
        |
        v
wrong-global-locality selection
~~~

---

## 8. FA1.3 — combined context / separability audit

Status:

**PASS_FA1_3_COMBINED_CONTEXT_SEPARABILITY_AUDIT**

All403 SIFT error bins:

- <=40 m: 367
- 40-200 m: 6
- unsafe: 30

For the 19 oracle-rescuable cases:

| Context question | Oracle favored |
|---|---:|
| More temporally consistent with +/-2 query context | **14 / 19** |
| scale_m2 ORB closer to oracle than selected SIFT | **14 / 19** |
| More Top20 support within 51.2 m | 1 / 19 |
| More Top20 support within 102.4 m | 1 / 19 |
| More reciprocal-rank support within 51.2 m | 1 / 19 |
| More reciprocal-rank support within 102.4 m | 1 / 19 |

Selected / oracle medians:

- DINO pool rank: 7 / 15
- oracle SIFT selector rank: 11
- inlier ratio: 0.357 / 0.462
- symmetric reprojection error: 6.371 px / 0.000 px

Reference-free feature medians, successful <=40 / unsafe / FA1 union:

| Feature | <=40 | unsafe | FA1 union |
|---|---:|---:|---:|
| Selected pool rank | 5 | 9 | 8 |
| Selected inliers | 203 | 5 | 5 |
| Selected inlier ratio | 0.482 | 0.355 | 0.345 |
| Top1-Top2 inlier margin | 42 | 0 | 0 |
| Max-inlier tie count | 1 | 2 | 2 |
| Selected minus DINO-rank1 inliers | 48 | 1 | 2 |
| SIFT vs scale_m2 projected distance | **1.18 m** | **664.53 m** | **609.79 m** |
| Temporal +/-2 median distance | **6.46 m** | **625.46 m** | **625.46 m** |
| Top20 support within 102.4 m | 2 | 1 | 1.5 |

The strongest warning signals found were:

1. temporal inconsistency;
2. SIFT-vs-ORB disagreement;
3. weak/tied SIFT evidence;
4. lower inlier ratio;
5. retrieval-vs-geometry conflict.

Map-neighborhood candidate density was **not** the dominant missing signal.

FA1.3 was descriptive only. No confidence threshold was selected and no
classifier was trained.

---

## 9. ORB's role after FA1

The evidence does **not** justify:

> vegetation => use ORB

Some vegetation-heavy cases are rescued by ORB; other vegetation-heavy cases
defeat both frontends.

ORB's stronger future role is:

> **secondary challenger / consistency sensor**

Proposed future authority logic:

~~~mermaid
flowchart TD
    Q[UAV query] --> R[DINOv2 Top-K retrieval]
    R --> S[SIFT + LightGlue primary]
    R --> O[scale_m2 ORB challenger]

    S --> C{SIFT strong and causally consistent?}
    O --> D[Cross-frontend disagreement]
    D --> C

    C -- yes --> A[Accept SIFT absolute observation]
    C -- no --> T{Does challenger agree with past trajectory?}
    T -- yes --> B[Use challenger-supported hypothesis]
    T -- no --> H[HOLD / abstain]
    H --> P[Relative odometry propagates]
~~~

This is a proposed architecture, not a frozen deployment policy.

A future confidence policy must use **past-only causal temporal evidence**.
FA1.3's +/-2 diagnostic used future neighbors only for explanation and must not
be copied directly into runtime logic.

---

## 10. Blind SIFT + LightGlue transfer

The historical recorded-flight blind demo used:

- 123 queries
- 512_s256
- DINO Top20
- ORB
- XFeat relative motion
- minimum_confident_v2 bootstrap

The current transfer deliberately changed the absolute observation stack.

### BV1.0

- map: 768_s256
- 123 queries
- 2,460 SIFT+LightGlue pairs
- selected homography: 123 / 123
- finite selected projection: 123 / 123
- median selected matches / inliers: 74 / 22
- median selected DINO rank: 5
- median inlier margin: 5
- no SRT / GPS / reference during execution

Candidate-pool SHA256:

d067432aa63c060fe61673c136127fc6f823ce0dfc9976234584ac8f60b20454

Pair-evidence SHA256:

71ca0ed935bda71d010327799aec11dfb92f6e6825ac76de468edd93030285fb

Selection SHA256:

adbb714b479d1a1d67881d3d99a29cce3f976c235a27b81398217c8973c6e3b1

### BV1.2

The minimum_confident_v2 state logic was preserved while the absolute
observation source changed to SIFT+LightGlue.

Preserved:

- Top4 observations
- four evidence frames
- minimum three unique tiles
- visual span >=100 px
- map span >=50 m
- median residual <=51.2 m
- max residual <=102.4 m
- acquisition threshold 12.8 m
- tracking threshold 12.8 m
- maturity support 3

Result:

- state: PROVISIONAL_ABSOLUTE_LOCK
- first leader: q8
- maturity: q11
- final active source: q104
- leader updates: 94

Historical ORB blind lock matured around q30.

This is a blind orchestration/authority result. Do not claim geographic
accuracy from lock timing alone.

---

## 11. OB1.0 — current-stack 45-degree oblique stress

The old 45-degree experiments were not reused as a direct current-stack test
because they used:

- 1024_s512
- DINO img224
- older LightGlue/SuperPoint-style reranking
- a different selection policy

OB1.0 rebuilt the oblique experiment under the current stack:

- 112 queries
- 768_s256
- DINOv2 ViT-S/14 img518 center-square avg-patch
- Top20
- 2,240 query-candidate pairs
- promoted SIFT + LightGlue unchanged
- inliers-descending selector
- reference-free freeze before SRT/GPS attachment

Status:

**PASS_OB1_0_POSTFREEZE_OBLIQUE_EVALUATION**

Reference-free SIFT evidence:

- selected homography / finite: 112 / 112
- median selected matches / inliers: 81.5 / 25
- median selected DINO pool rank: 8
- median inlier margin: 6

Current-stack DINO candidate availability:

| K | Contains body | center <=40 m available |
|---:|---:|---:|
| Top1 | 35 / 112 | 8 / 112 |
| Top5 | 73 / 112 | 20 / 112 |
| Top10 | 97 / 112 | 57 / 112 |
| Top20 | **112 / 112** | **86 / 112** |

SIFT-selected:

- contains body: 47 / 112
- tile-center <=40: 4 / 112
- tile-center <=80: 44 / 112
- median tile-center error: 86.58 m
- projected query-center vs body <=40: 2 / 112
- projected query-center vs body <=80: 60 / 112
- projected query-center vs body median: 75.94 m
- projected query-center unsafe >200/non-finite: 2 / 112

Top20 oracle:

- tile-center <=40: 86 / 112
- tile-center <=80: 112 / 112
- projected query-center vs body <=40: 67 / 112
- projected query-center vs body <=80: 103 / 112

### Correct interpretation

Oblique view degrades the current stack materially compared with the nadir
development trajectory.

However, it is **not** a total collapse:

- candidate availability remains high;
- all selected pairs still produce valid homographies;
- catastrophic projected failures are rare;
- the larger limitation is selecting the correct physical locality and mapping
  oblique image-center geometry to UAV body position.

At 45 degrees, image center may correspond to ground ahead of the UAV body.
Therefore projected-query-center vs body error is a **stress diagnostic**, not a
pure body-position metric.

The old simple global footprint-offset experiment was already negative, so the
future oblique solution should not be a constant offset. It requires calibrated
camera rays, pose/extrinsics, and ground/range geometry.

---

## 12. Thermal future extension

Thermal imagery is **not validated** in the current stack.

The current RGB map is visible-spectrum ORT10LT. Using thermal UAV imagery would
therefore create a cross-spectral thermal-to-RGB localization problem.

The current software can technically process a thermal frame converted to an
8-bit image representation, but this does not imply useful performance:

- DINOv2 is not validated here for thermal-to-RGB retrieval;
- SIFT can detect intensity-gradient structure;
- LightGlue can process the resulting SIFT descriptors;
- thermal and RGB appearance can differ substantially, especially in vegetation
  and temperature-dependent surfaces.

Future hypothesis:

> thermal may be useful as an auxiliary challenger / confidence source fused
> with RGB rather than as an immediate replacement for the RGB stack.

Recommended first thermal experiment, if reopened:

- 10-20 representative thermal queries;
- Top5 only;
- 50-100 pairs;
- inspect retrieval rank, SIFT matches/inliers, homography success, and whether
  correct locality becomes competitive.

Do not spend a full 112x20 or 403x20 run until that smoke is promising.

---

## 13. Rejected / weakened hypotheses

The current evidence weakens or rejects the following simple explanations:

- **More inliers always means correct locality** — false in residual near-ties.
- **Broader match coverage means correct locality** — false in canopy aliases.
- **Any valid homography is strong evidence** — false; wrong candidates often
  have valid homographies.
- **Top20 neighborhood density will solve the residual selector problem** —
  not supported by FA1.3.
- **Vegetation should automatically trigger ORB** — not supported.
- **One constant oblique footprint correction solves 45 degrees** — historical
  test was negative.
- **A heavier aerial model automatically justifies deployment** — MASt3R gain
  did not pass its promotion gate.

---

## 14. Claims ledger

| Claim | Evidence class | Status |
|---|---|---|
| SIFT+LightGlue is stronger than scale_m2 ORB on full403 nadir | Frozen measured evidence | Supported |
| Residual SIFT failures are often weak near-ties | Post-freeze diagnostic | Supported |
| Failure set is predominantly vegetation-heavy | Manual visual observation | Supported qualitatively only |
| ORB disagreement is a useful warning signal | Post-freeze diagnostic | Strong clue, not a frozen rule |
| Temporal inconsistency is a useful warning signal | Post-freeze diagnostic | Strong clue, not a frozen rule |
| SIFT-primary + ORB-challenger is the next authority architecture | Proposed architecture | Not yet validated |
| 45-degree viewpoint degrades current-stack accuracy | Frozen + post-freeze stress evidence | Supported |
| Thermal can strengthen RGB verification | Future hypothesis | Unvalidated |
| True UAV-to-map GSD normalization improves matching | Future hypothesis | Unfinished |

---

## 15. Canonical recent commands

Environment:

~~~bash
cd /Users/harishprabhu/Documents/drone-localisation
source .drone_venv/bin/activate
export PYTHONPATH=$PWD/src
~~~

FA1.0:

~~~bash
python scripts/villoc/geometry/fa1_0_sift_residual_failure_audit.py   --config configs/research/crossview_scale_rotation_domain_fa1.yaml
~~~

FA1.1:

~~~bash
python scripts/villoc/geometry/fa1_1_sift_targeted_pair_diagnostics.py   --config configs/research/crossview_scale_rotation_domain_fa1_1.yaml
~~~

FA1.2:

~~~bash
python scripts/villoc/geometry/fa1_2_visual_semantic_review.py   --config configs/research/crossview_scale_rotation_domain_fa1_2.yaml
~~~

FA1.3:

~~~bash
python scripts/villoc/geometry/fa1_3_combined_context_separability_audit.py   --config configs/research/crossview_scale_rotation_domain_fa1_3.yaml
~~~

OB1.0:

~~~bash
python scripts/villoc/geometry/ob1_0_45deg_current_stack_sift_lightglue.py   --config configs/research/villoc_45deg_sift_lightglue_ob1.yaml
~~~

---

## 16. Closeout decision

This research block has produced a significantly stronger absolute frontend than
the earlier ORB stack and enough diagnostic evidence to identify the next
bottleneck.

Current research state:

~~~text
retrieval coverage             strong enough for continued work
SIFT+LightGlue absolute frontend  promoted
Aerial-MASt3R compact screen      closed
blind SIFT transfer               demonstrated
residual failure mechanism        analyzed
45-degree current-stack stress    completed
confidence / abstention           NOT yet frozen
true physical UAV GSD matching    NOT yet completed
physical drone integration        next hardware phase
~~~

The correct next move is to **park algorithmic matcher expansion temporarily**
and inspect the physical drone/camera/autopilot system properly so that the
unfinished physical-GSD and causal deployment experiments can be designed from
real sensor/calibration data.
