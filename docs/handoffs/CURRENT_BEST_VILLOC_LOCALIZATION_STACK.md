# Current best VILLOC localization stack

**Purpose:** canonical description of the strongest current research stack  
**Status:** strong research baseline; not yet final deployment architecture  
**Branch:** research/crossview-scale-rotation-domain-v1  
**Date:** 2026-10-16

This document answers:

> What is the strongest system currently available in this repository, how does
> it work, what evidence supports it, where does it fail, and what should the
> next implementation build on?

For experiment history and decision reasoning, read:

docs/handoffs/crossview_scale_rotation_domain_v1/FA1_OB1_CLOSEOUT.md

---

## 1. System objective

The system estimates UAV motion and intermittent absolute map position in
GNSS-denied or weak-GNSS conditions using:

- recorded/onboard camera imagery;
- a prepared georeferenced orthophoto;
- relative visual motion;
- global visual retrieval;
- local geometric verification;
- temporal map-state authority.

GNSS/SRT reference is isolated from localization decisions and attached only
for evaluation where the stage contract permits it.

---

## 2. Current architecture

### 2.1 Implemented research architecture

~~~mermaid
flowchart TD
    A[UAV RGB frame] --> B[Relative frontend<br/>XFeat in current blind pipeline]
    B --> C[Continuous relative trajectory]

    A --> D[DINOv2 retrieval<br/>ViT-S/14 img518<br/>center-square avg-patch]
    D --> E[Top20 map candidates<br/>768_s256 EPSG:3346]
    E --> F[SIFT + LightGlue<br/>primary geometric frontend]
    F --> G[Homography<br/>query-center map projection]
    G --> H[Current frozen selector<br/>inliers desc<br/>DINO rank asc<br/>tile id asc]

    H --> I[Absolute observation candidates]
    I --> J[minimum_confident_v2<br/>Top4 / four-frame evidence]
    C --> J
    J --> K{Map evidence mature?}
    K -- no --> L[Relative-only / HOLD]
    K -- yes --> M[PROVISIONAL_ABSOLUTE_LOCK]
    M --> N[Causal map alignment]
    L --> O[Trajectory output]
    N --> O
~~~

### 2.2 Proposed next authority architecture

The frontend itself is no longer the only bottleneck. FA1 indicates that the
next authority should use complementary evidence.

~~~mermaid
flowchart TD
    A[UAV frame] --> R[DINOv2 shortlist]
    R --> S[SIFT + LightGlue primary]
    R --> O[scale_m2 ORB challenger]

    S --> Q[SIFT strength<br/>inliers, margin, ratio]
    S --> T[Past-only temporal innovation]
    O --> X[SIFT vs ORB disagreement]

    Q --> D{Authority decision}
    T --> D
    X --> D

    D -- strong SIFT + causal --> AS[Accept SIFT]
    D -- challenger fits history --> AO[Use challenger-supported hypothesis]
    D -- unresolved conflict --> H[Abstain / HOLD]

    H --> P[Relative odometry propagates]
~~~

This second diagram is a **proposal**, not a frozen implemented policy.

---

## 3. Dataset roles

| Dataset / run | Role | Important boundary |
|---|---|---|
| villoc_traj01_90deg_stable120m | Primary development / quantitative frontend evaluation | Post-freeze truth allowed only after reference-free evidence |
| recorded-flight 123-query blind demo | Blind orchestration / causal bootstrap validation | No SRT/GPS/reference during localization |
| VILLOC 45-degree 112-query dataset | Viewpoint stress test | Oblique center/body geometry differs from nadir |
| Historical 90°/45° VILLOC sets | Earlier retrieval/verifier diagnostics | Different map variants/frontends; contextual comparison only |
| SATLOC traj01 | Earlier algorithm-development benchmark | Pre-VILLOC development history |

The 123-query blind demo is related to the same project area and should not be
described as fully independent dataset generalization.

---

## 4. Map and retrieval layer

Current map:

- source: Geoportal ORT10LT 2024-2026
- CRS: EPSG:3346
- variant: 768_s256
- tile side: 768 px
- physical footprint: approximately 153.6 m
- nominal stride: approximately 51.2 m
- source GSD: approximately 0.20 m/px

Current retrieval:

- DINOv2 ViT-S/14
- image size 518
- center_square
- avgpatch
- L2-normalized cached descriptors
- Top20 shortlist

Development Top20 SHA256:

0a7832c6f0997a33851343fd7df537d2f6410211ac0e38e7ec4325899d8df47e

Development Top20 availability:

- containment: 384 / 403
- center <=40 m: 277 / 403
- center <=80 m: 380 / 403

Interpretation:

DINOv2 is a **candidate generator**, not a final GPS estimator.

---

## 5. Primary absolute geometric frontend

Current primary:

**SIFT + official SIFT-specific LightGlue**

Configuration:

| Parameter | Value |
|---|---|
| Feature backend | OpenCV SIFT |
| RootSIFT | true |
| Max keypoints | 4096 |
| Resize long edge | 1024 |
| NMS radius | 0 |
| Detection threshold | 0.0066667 |
| Edge threshold | 10 |
| Octaves | 4 |
| LightGlue filter threshold | 0.1 |
| Depth confidence | 0.95 |
| Width confidence | 0.99 |
| RANSAC | 5 px |
| Min homography inliers | 4 |
| Projection | query center -> tile -> EPSG:3346 |
| Frozen device | CPU |

LightGlue provenance:

- commit:
  eb42fee2d71449efb0aa5c10549752b5d75384d8
- SIFT blob:
  802fc1c2eb9ee852691e0e4dd67455f822f8405f
- LightGlue blob:
  485833515eb0a80866b3d078d3e43d39ddcd84ae
- weights:
  5b52b8d9982d43532dc042606b346bb9594c9f5a4bd6f64362c63866287b4ac0

Full403 evidence:

c0031242f0b359f1bfa3c87ea47ac428393ed3e8e03d7b5c6d0822e8822cb8a0

---

## 6. Geometry-method benchmark

### 6.1 Full403 comparable controls

| Frontend | Top1 contains | Top1 <=40 center | Top1 <=80 center | Projection <=40 | Projection <=80 | Unsafe | Median projection | RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Original ORB | 292 | 150 | 287 | 293 | 305 | 88 | 6.09 m | 323.22 m |
| scale_m2 ORB | 305 | 163 | 293 | 319 | 331 | 63 | 5.84 m | 282.57 m |
| **SIFT + LightGlue** | **349** | **205** | **346** | **367** | **368** | **30** | **4.99 m** | **224.80 m** |

SIFT's main gain is not only sub-metre/local projection quality. It strongly
reduces wrong-locality selection.

### 6.2 Other screened frontends

| Family | Scope | Outcome |
|---|---|---|
| XFeat sparse absolute frontend | full403 | rejected versus ORB |
| XFeat* semi-dense | full403 | rejected versus ORB |
| SuperPoint + LightGlue | uniform40 | below ORB |
| ALIKED + LightGlue | uniform40 | improved over SuperPoint but below ORB |
| Aerial-MASt3R | 40 x 6 compact exact-pair screen | directionally better than SIFT but below promotion gate |

Do not mix n=40 screening numbers into full403 result tables without explicitly
showing their denominator.

---

## 7. Why SIFT + LightGlue was promoted

Versus scale_m2 ORB:

- +44 Top1 containment
- +42 Top1 center <=40
- +53 Top1 center <=80
- +48 selected projection <=40
- unsafe tail 63 -> 30
- 43 catastrophic rescues
- 10 catastrophic damages

The evidence supports:

> explicit local scale/orientation representation is materially valuable for
> UAV-to-orthophoto matching.

It does **not** support:

> SIFT+LightGlue is always safer than ORB.

The 10 catastrophic damages are exactly why ORB remains useful as a challenger.

---

## 8. Known failure envelope

The strongest residual failure pattern is:

- vegetation-heavy / canopy-dominant or mixed vegetation scenes;
- only limited distinctive urban structure;
- weak raw inlier counts;
- Top1-Top2 inlier ties or near-ties;
- large SIFT-vs-ORB disagreement;
- large temporal innovation.

FA1 key medians:

| Feature | SIFT <=40 success | SIFT unsafe |
|---|---:|---:|
| Selected inliers | 203 | 5 |
| Inlier ratio | 0.482 | 0.355 |
| Top1-Top2 margin | 42 | 0 |
| Max-inlier tie count | 1 | 2 |
| SIFT-vs-ORB map disagreement | 1.18 m | 664.53 m |
| +/-2 temporal median distance | 6.46 m | 625.46 m |

The useful oracle candidate was often deep in the current SIFT ranking:

- median SIFT selector rank: 11
- max: 17
- median DINO pool rank: 15

The selected wrong candidate exceeded the oracle by only +1 inlier median.

This is a confidence/authority problem.

---

## 9. Blind map-state bootstrap

Current transferred blind absolute observation source:

- map: 768_s256
- DINO Top20
- SIFT+LightGlue
- selector: inliers desc -> DINO rank asc -> tile ID asc

minimum_confident_v2 preserves:

- Top4 candidate observations
- four evidence frames
- at least 3 unique tile IDs
- visual span >=100 px
- map span >=50 m
- median residual <=51.2 m
- max residual <=102.4 m
- acquisition innovation <=12.8 m
- tracking innovation <=12.8 m
- maturity support = 3

The 12.8 m threshold is one quarter of the 51.2 m frozen map stride.

BV1.2 result:

- PROVISIONAL_ABSOLUTE_LOCK
- first leader q8
- matured q11
- 94 leader updates
- final active source q104

Historical ORB demo matured near q30.

Do not use lock timing alone as an accuracy claim.

---

## 10. Relative frontend and fusion position

The absolute frontend is only one part of the full localization system.

The project has already demonstrated:

- continuous relative motion;
- map-state bootstrap;
- map alignment;
- sparse absolute correction/fusion;
- estimated map XY / lat-lon export;
- orthophoto and interactive map visualization;
- reference-free run orchestration;
- post-freeze evaluation separation.

Current deployment concept:

~~~mermaid
flowchart LR
    C[Camera stream] --> R[High-rate relative frontend]
    R --> P[Continuous local pose]

    C --> A[Low-rate / event-driven absolute localization]
    A --> V[Confidence authority]
    P --> V

    V -- correction accepted --> F[Map alignment / fusion]
    V -- abstain --> P

    F --> O[Map-coordinate trajectory]
    P --> O
~~~

The absolute matcher should **not** run Top20 SIFT+LightGlue at camera FPS.

---

## 11. Runtime implications

Frozen CV2.4 full403 CPU evidence:

- 8,060 pairs
- wall time approximately 20,993 s
- approximately 5.83 h
- median pair total around 1.88 s

Naive Top20-per-frame verification is therefore not a deployment architecture.

Future runtime path:

1. CUDA parity;
2. precompute map-side SIFT features;
3. compute query SIFT once;
4. restrict expensive verification to a smaller candidate subset when safe;
5. run absolute localization asynchronously;
6. use confidence/abstention instead of forcing every query into a map update.

Jetson optimization is downstream of frozen accuracy/authority design.

---

## 12. 45-degree stress result

OB1.0 used the current stack unchanged on 112 oblique queries.

Current-stack DINO Top20:

- contains body: 112 / 112
- center <=40 available: 86 / 112
- center <=80 available: 112 / 112

SIFT selected:

- contains body: 47 / 112
- tile-center <=40: 4 / 112
- tile-center <=80: 44 / 112
- median tile-center error: 86.58 m
- projected query-center/body median: 75.94 m
- projected unsafe: 2 / 112

This demonstrates substantial viewpoint degradation.

It does **not** demonstrate retrieval collapse. The good locality is frequently
still available inside Top20.

Oblique imagery also changes what image center means geometrically. The image
center can see ground ahead of the UAV body, so body error and query-center
projection error cannot be treated as identical.

Future oblique geometry needs calibrated ray/ground intersection rather than a
constant offset.

---

## 13. True UAV-to-orthophoto GSD normalization remains open

Earlier map-footprint experiments changed how much physical map area was
represented to retrieval. That is **not** the same as a physically controlled
per-query UAV-to-map GSD normalization.

The unfinished experiment should estimate UAV image ground scale using real
camera geometry.

For a nadir rectified pinhole camera over approximately planar ground:

~~~text
GSD_x approximately AGL / fx_pixels
GSD_y approximately AGL / fy_pixels
~~~

in metres/pixel when fx/fy are calibrated pixel focal lengths and AGL is in
metres.

Therefore physical sensor width is not strictly required if calibrated camera
intrinsics in pixels are available.

For robust real use, capture:

- camera model;
- resolution;
- fx, fy, cx, cy;
- distortion coefficients;
- camera-to-body or camera-to-gimbal extrinsics;
- camera mounting/gimbal angle;
- synchronized frame timestamps;
- AGL / range / ground-height estimate;
- vehicle attitude;
- rolling/global shutter information.

For oblique imagery, project camera rays to a ground plane / DEM / range model
instead of using one scalar GSD.

This is a major reason to inspect the physical drone next.

---

## 14. Thermal / multispectral future extension

Thermal is a future cross-spectral extension.

Possible architecture:

~~~mermaid
flowchart TD
    RGB[RGB frame] --> RGBR[RGB retrieval + SIFT/LG]
    TH[Thermal frame] --> THV[Thermal structural / cross-spectral verifier]

    RGBR --> F[Multimodal confidence]
    THV --> F
    F --> A[Absolute authority]
~~~

Current status:

- RGB path: validated experimentally;
- thermal path: not validated;
- thermal-to-RGB DINO retrieval: unknown;
- thermal SIFT/LightGlue structural correspondence: possible but unproven.

Thermal should first be tested as a small challenger/confidence experiment, not
as a full replacement pipeline.

---

## 15. Visual evidence index

Tracked repository visuals from the evaluated pipeline:

- docs/assets/villoc_traj01_final/figures/01_final_mission_story_orthophoto_map_clean.png
- docs/assets/villoc_traj01_final/figures/04_fusion_error_vs_distance_with_events.png
- docs/assets/villoc_traj01_final/figures/07_absolute_dino_recall_funnel.png
- docs/assets/villoc_traj01_final/figures/08_absolute_orb_verifier_funnel.png

Recent generated research visuals:

- outputs/research_runs/crossview_scale_rotation_domain_v1/cv2/cv2_5/visual_diagnostics/traj01_90deg_stable120m_frontend_selected_scatter_ortho.png
- outputs/research_runs/crossview_scale_rotation_domain_v1/cv2/cv2_5/visual_diagnostics/traj01_90deg_stable120m_sift_vs_scale_m2_selected_points_ortho.png
- outputs/research_runs/crossview_scale_rotation_domain_v1/fa1_2/fa1_2_failure_contact_sheet.png
- outputs/research_runs/crossview_scale_rotation_domain_v1/fa1_2/montages/
- BV1.2 blind comparison visuals under:
  outputs/research_runs/blind_demo_sift_lightglue_transfer_v1/

The generated-output paths are part of the local evidence tree; not every binary
figure is checked into Git.

---

## 16. Provenance / frozen artifacts

Important frozen hashes:

| Artifact | SHA256 |
|---|---|
| QV1.4 development Top20 | 0a7832c6f0997a33851343fd7df537d2f6410211ac0e38e7ec4325899d8df47e |
| Original ORB full403 | 91ba3381e463bd083f9355b517dbb93953e4423a4b366638c1a97419ea0aea3f |
| scale_m2 ORB full403 | 94353f763725f44f278192bcedcfd139ab136ecebf4f1dcc6e07487380964810 |
| SIFT+LightGlue full403 | c0031242f0b359f1bfa3c87ea47ac428393ed3e8e03d7b5c6d0822e8822cb8a0 |
| LightGlue weights | 5b52b8d9982d43532dc042606b346bb9594c9f5a4bd6f64362c63866287b4ac0 |
| BV1.0 candidate pool | d067432aa63c060fe61673c136127fc6f823ce0dfc9976234584ac8f60b20454 |
| BV1.0 pair evidence | 71ca0ed935bda71d010327799aec11dfb92f6e6825ac76de468edd93030285fb |
| BV1.0 selection | adbb714b479d1a1d67881d3d99a29cce3f976c235a27b81398217c8973c6e3b1 |

OB1.0 freeze/evaluation hashes are recorded in the generated local
ob1_0_reference_free_freeze_report.json and ob1_0_postfreeze_evaluation_report.json.

Do not reconstruct missing hashes from memory.

---

## 17. Claims and readiness

| Component | Current status |
|---|---|
| DINOv2 Top20 retrieval | Strong research component |
| 768_s256 map geometry | Current promoted research map |
| SIFT+LightGlue absolute frontend | **Promoted** |
| scale_m2 ORB | Frozen control + proposed challenger |
| Raw-inlier selector | Implemented but known weak point |
| minimum_confident_v2 | Demonstrated causal blind bootstrap |
| SIFT-primary / ORB-challenger authority | Proposed, not frozen |
| Aerial-MASt3R | Compact screen closed |
| 45-degree handling | Stress-tested, not solved |
| Thermal fusion | Future hypothesis |
| True physical GSD normalization | Pending |
| Jetson optimized runtime | Pending final architecture |
| Physical drone integration | Next hardware phase |

---

## 18. Immediate development direction

Do **not** immediately add another matcher.

The next practical step is to inspect and instrument the physical drone so that
the following can be measured rather than guessed:

- camera intrinsics;
- camera/body geometry;
- synchronized attitude;
- altitude/AGL;
- actual camera streams;
- Pixhawk telemetry;
- compute/network topology.

That hardware information enables the unfinished physical GSD normalization,
oblique ground-ray geometry, and later onboard runtime integration.

Read next:

docs/handoffs/PHYSICAL_DRONE_TRANSITION_HANDOFF_20261016.md
