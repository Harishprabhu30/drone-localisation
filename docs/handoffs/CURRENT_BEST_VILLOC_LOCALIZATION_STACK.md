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

## 18. New dataset execution runbook — team demo and scientific benchmark

This section is the practical re-entry path when a **new recorded flight or
dataset** is provided in the future.

The objective is to prevent a new session from reconstructing the pipeline from
memory or accidentally evaluating a different frontend.

### 18.1 First decide what kind of run is being requested

There are four independent questions.

| Question | Option A | Option B |
|---|---|---|
| Geography | same mapped AOI | new geographic area |
| Evaluation | hidden reference available | blind/no reference |
| Camera | near-nadir RGB | oblique / changing attitude |
| Modality | RGB | thermal / other |

The standard validated path is:

> **RGB + known/prepared orthophoto + DINOv2 img518 Top20 + 768_s256 +
> SIFT+LightGlue**

Thermal is not part of the standard run. Use the small thermal smoke described
earlier before attempting a full benchmark.

For oblique imagery, report tile/locality metrics separately from
query-center-vs-UAV-body projection error.

---

### 18.2 Two reporting modes

#### Mode A — team / operational demo

Use this when no trustworthy hidden reference is available.

Allowed claims:

- dataset processed successfully;
- number of frames/queries;
- retrieval and matcher execution success;
- selected candidate/map observations;
- map-state lock/HOLD behavior;
- relative/map-aligned trajectory when available;
- runtime/resource use;
- qualitative map visualization.

Do **not** report RMSE, <=40 m accuracy, or "correct localization percentage"
without an isolated reference.

#### Mode B — scientific benchmark

Use this when SRT/GNSS/RTK/reference is available.

Rule:

~~~text
prepare reference
        |
        +--> keep out of retrieval / verifier / acceptance
        |
reference-free localization output
        |
        v
freeze output + hashes
        |
        v
attach reference
        |
        v
metrics / plots / failure analysis
~~~

Reference may be used for:

- synchronization;
- predeclared dataset/map preparation;
- post-freeze evaluation;
- sanity checks.

Reference must not be used to:

- choose DINO candidates;
- choose SIFT winner;
- choose correction;
- tune an error threshold after seeing answers;
- cherry-pick "good" frames for the headline result.

---

### 18.3 Create a dataset identity before running anything

Use one stable name everywhere.

Example:

~~~bash
export DATASET=client_flight_20261016
export CFG=configs/dataset_${DATASET}.yaml
export OUT=outputs/villoc/${DATASET}
export STREAM=V
export SAMPLE_FPS=1
~~~

Create the new dataset config from the closest existing config:

~~~bash
cp configs/dataset_villoc_traj01_90deg_stable120m.yaml "$CFG"
~~~

Then edit, at minimum:

~~~text
dataset.name
dataset.sequence_name
dataset.raw_root
dataset.processed_root
dataset.output_root

streams.V.video
streams.V.srt
streams.V.fps_expected
streams.V.resolution_expected

dataset.view_angle_group
reference.source
map/reuse policy
~~~

Do not leave old dataset names or paths in the copied config.

Recommended raw layout:

~~~text
data/raw/villoc/<dataset>/
    <video>.MP4
    <telemetry/reference>.SRT
    optional additional streams
~~~

Recommended output layout:

~~~text
outputs/villoc/<dataset>/
    metadata/
    reports/
    descriptors/
    retrieval/
    figures/
    logs/
~~~

Before the run, record:

~~~bash
mkdir -p "$OUT/logs"

git rev-parse HEAD | tee "$OUT/logs/git_commit.txt"
git status --short | tee "$OUT/logs/git_status.txt"
python --version | tee "$OUT/logs/python_version.txt"
~~~

A team result should always be traceable to one Git commit.

---

### 18.4 DJI/VILLOC-style MP4 + SRT ingestion

These commands apply when the new dataset follows the current Villoc/DJI
MP4+SRT format.

Activate the environment:

~~~bash
cd /Users/harishprabhu/Documents/drone-localisation
source .drone_venv/bin/activate
export PYTHONPATH=$PWD/src
~~~

Parse SRT:

~~~bash
python scripts/villoc/s8_1_parse_villoc_srt.py   --config "$CFG"   2>&1 | tee "$OUT/logs/s8_1_parse_srt.log"
~~~

Extract 1 Hz benchmark frames:

~~~bash
python scripts/villoc/s8_2_extract_sampled_frames.py   --config "$CFG"   --stream "$STREAM"   --sample-rate-fps "$SAMPLE_FPS"   --overwrite   2>&1 | tee "$OUT/logs/s8_2_extract_frames.log"
~~~

Build the reference trajectory:

~~~bash
python scripts/villoc/s8_3_build_reference_trajectory.py   --config "$CFG"   --stream "$STREAM"   --sample-rate-fps "$SAMPLE_FPS"   2>&1 | tee "$OUT/logs/s8_3_reference_trajectory.log"
~~~

The reference trajectory exists for later evaluation. Do not feed its position
columns into retrieval or verification.

Audit image quality:

~~~bash
python scripts/villoc/s8_4_visual_frame_audit.py   --config "$CFG"   --stream "$STREAM"   --sample-rate-fps "$SAMPLE_FPS"   2>&1 | tee "$OUT/logs/s8_4_visual_audit.log"
~~~

Build the query index:

~~~bash
python scripts/villoc/s8_5_build_uav_frame_index.py   --config "$CFG"   --stream "$STREAM"   --sample-rate-fps "$SAMPLE_FPS"   --golden-n 20   2>&1 | tee "$OUT/logs/s8_5_query_index.log"
~~~

For a different recorder format, do **not** fabricate an SRT file merely to
satisfy these scripts. Instead export a canonical query manifest from the
recorder.

At minimum the absolute stack needs stable:

~~~text
query_id
token0_id
image_path
timestamp
~~~

Reference position/attitude fields should be stored separately or clearly
marked evaluation-only.

---

### 18.5 Same AOI — reuse the promoted current map

If the new flight is inside the already prepared Vilnius AOI, reuse these
current assets instead of regenerating the map:

~~~text
768_s256 tiles:
data/processed/villoc/research/retrieval_candidate_pool_v2/r1/tiles_768_s256

768_s256 tile index:
outputs/research_runs/retrieval_candidate_pool_v2/r1_map_pyramid/metadata/
s8_9_satellite_tile_index_768_s256.csv

DINOv2 img518 map cache:
outputs/research_runs/retrieval_candidate_pool_v2/r1_map_pyramid/descriptors/
s8_11b_dinov2_map_768_s256_dinov2_vits14_img518_center_square_avgpatch_cpu.npz
~~~

The new dataset config should expose a 768_s256 tile variant pointing to that
tile index.

Do not silently fall back to 512_s256 simply because an older config already
contains it.

---

### 18.6 New geographic area — prepare a new map first

If the flight is outside the current AOI, obtain a georeferenced RGB orthophoto
with known CRS and pixel resolution.

Prefer a mission/predeclared AOI. Avoid using a very tight GT-derived crop that
artificially tells retrieval where the vehicle must be.

Define:

~~~bash
export MASTER_TIF=/path/to/master_orthophoto.tif
export AOI_TIF=data/processed/villoc/$DATASET/maps/aoi.tif
export TILES_DIR=data/processed/villoc/$DATASET/maps/tiles_768_s256
export TILE_INDEX=$OUT/metadata/s8_9_satellite_tile_index_768_s256.csv
~~~

If reference data is available for dataset-preparation bounds, create the bbox
plan:

~~~bash
python scripts/villoc/s8_6_build_map_bbox_plan.py   --config "$CFG"   --stream "$STREAM"   --sample-rate-fps "$SAMPLE_FPS"   2>&1 | tee "$OUT/logs/s8_6_map_bbox.log"
~~~

Validate the source GeoTIFF:

~~~bash
python scripts/villoc/s8_7a_validate_map_geotiff.py   --config "$CFG"   --tif "$MASTER_TIF"   --name master_orthophoto   --tile-size-px 768   --stride-px 256   2>&1 | tee "$OUT/logs/s8_7a_validate_map.log"
~~~

Crop the predeclared AOI when required:

~~~bash
python scripts/villoc/s8_8_crop_master_aoi.py   --config "$CFG"   --src-tif "$MASTER_TIF"   --output-name "$(basename "$AOI_TIF")"   2>&1 | tee "$OUT/logs/s8_8_crop_aoi.log"
~~~

Generate the current map geometry:

~~~bash
python scripts/villoc/s8_9_generate_reference_tiles.py   --config "$CFG"   --src-tif "$AOI_TIF"   --tiles-dir "$TILES_DIR"   --tile-size-px 768   --stride-px 256   --image-format jpg   --jpeg-quality 95   --overwrite   2>&1 | tee "$OUT/logs/s8_9_tiles_768_s256.log"
~~~

Audit the tile index:

~~~bash
python scripts/villoc/s8_10a_audit_tile_index_integrity.py   --config "$CFG"   --variant "768_s256:768:256:$TILES_DIR:$TILE_INDEX"   2>&1 | tee "$OUT/logs/s8_10a_tile_integrity.log"
~~~

Before continuing, verify:

- CRS is correct;
- physical pixel size is plausible;
- tile footprint is known;
- tile count is non-zero;
- there are no duplicate tile IDs;
- the intended mission area is covered.

For a new map whose source GSD is not approximately 0.20 m/px, **do not claim
that 768 px still represents 153.6 m**. Recompute the physical footprint and
record it.

---

### 18.7 Build the canonical benchmark manifest when reference exists

For the current Villoc benchmark tooling, S8.10B can create the canonical
manifest and oracle relation for evaluation.

For a single 768_s256 variant:

~~~bash
python scripts/villoc/s8_10b_uav_tile_oracle_audit.py   --config "$CFG"   --variant "768_s256:$TILE_INDEX"   2>&1 | tee "$OUT/logs/s8_10b_oracle_audit.log"
~~~

Expected canonical manifest:

~~~text
$OUT/metadata/s8_10b_canonical_uav_query_manifest.csv
~~~

The manifest contains reference fields for evaluation, but descriptor/ranking
code must consume only model-safe columns.

For a genuinely blind demonstration, use a blind/reference-free manifest
instead of an oracle-bearing benchmark manifest.

---

### 18.8 Build DINOv2 img518 query descriptors

Current descriptor protocol:

~~~text
dinov2_vits14
img518
center_square
avgpatch
L2 normalized
~~~

For the same AOI, reuse the frozen map descriptor cache and build only the new
query cache.

Example:

~~~bash
export QUERY_CSV=$OUT/metadata/s8_10b_canonical_uav_query_manifest.csv
export MAP_CACHE_ROOT=outputs/research_runs/retrieval_candidate_pool_v2/r1_map_pyramid/descriptors
export MAP_CACHE_TAG=dinov2_vits14_img518_center_square_avgpatch_cpu

python scripts/villoc/s8_11bc_build_dinov2_caches.py   --config "$CFG"   --query-csv "$QUERY_CSV"   --image-size 518   --crop-mode center_square   --pooling avgpatch   --device cpu   --reuse-map-caches   --map-cache-root "$MAP_CACHE_ROOT"   --map-cache-tag "$MAP_CACHE_TAG"   2>&1 | tee "$OUT/logs/s8_11bc_dino_cache.log"
~~~

For a new geographic area, build the 768_s256 map cache once under that
dataset's map-cache directory rather than reusing the Vilnius cache.

Do not compare a new dataset using img224 if the claim is intended to represent
the current promoted stack.

---

### 18.9 DINO retrieval / candidate availability

For a reference-backed benchmark, the existing independent retrieval evaluator
can rank candidates and report oracle availability:

~~~bash
python scripts/villoc/s8_11d_independent_dinov2_retrieval.py   --config "$CFG"   --map-cache-root "$MAP_CACHE_ROOT"   --image-size 518   --crop-mode center_square   --pooling avgpatch   --device-tag cpu   2>&1 | tee "$OUT/logs/s8_11d_retrieval.log"
~~~

Record at least:

- Top1;
- Top5;
- Top10;
- Top20;
- containment availability;
- center <=40 m availability;
- center <=80 m availability.

This separates:

> retrieval failure

from:

> correct candidate present but verifier/selector failure.

That distinction must be preserved in every new-dataset report.

For blind/no-reference use, use the blind retrieval path rather than an
evaluation script that loads oracle data.

---

### 18.10 Current promoted SIFT+LightGlue execution

**Important operational note:**

The repository now exposes a canonical one-command runner for the promoted
current SIFT+LightGlue stack:

~~~bash
python scripts/villoc/run_localization_pipeline.py \
  --config configs/localization/current_stack_reference_free.yaml \
  --video <recorded_flight.mp4> \
  --run-id <unique_run_id>
~~~

The runner reuses the proven reference-free orchestration stages, but its
absolute frontend is now explicitly:

~~~text
DINOv2 img518 Top20
    -> SIFT+LightGlue on all Top20
    -> inliers DESC / DINO rank ASC / tile_id ASC
    -> Top4 projected observations
    -> minimum_confident_v2
~~~

Do **not** replace this current-stack verifier with:

~~~text
s8_12e1b_lightglue_top20_verifier_reranker.py
~~~

for a "current stack" claim. That script belongs to an older LightGlue
frontend/policy and is not equivalent to the promoted SIFT-specific frontend.

The promoted implementation is:

~~~text
src/uavloc/geometry/sift_lightglue_absolute_frontend.py
~~~

and its frozen contracts are demonstrated by:

~~~text
scripts/villoc/geometry/cv2_4_freeze_sift_lightglue_full403.py
scripts/villoc/geometry/ob1_0_45deg_current_stack_sift_lightglue.py
~~~

When a new dataset arrives, use the canonical runner above for the
reference-free execution. It consumes the prepared map assets from the run YAML
and writes neutral project-facing artifacts while preserving compatibility
files needed by the historical downstream helpers.

The historical recorded-flight orchestrator remains untouched for exact replay.
Do not edit completed frozen experiments in place.

The new stage must preserve:

~~~text
SIFT/OpenCV + RootSIFT
max keypoints 4096
long edge 1024
LightGlue SIFT model
filter threshold 0.1
depth confidence 0.95
width confidence 0.99
RANSAC 5 px
minimum 4 inliers
query-center projection
selector:
    inliers DESC
    DINO pool rank ASC
    tile_id ASC
Top20 candidate budget
~~~

It must also:

1. write pair-level reference-free evidence;
2. write one selected observation/query;
3. freeze hashes before attaching GT;
4. preserve failed pairs rather than silently dropping them;
5. report matcher/runtime counts;
6. never tune SIFT based on the new dataset's answer labels during the same
   benchmark.

This generic runner is a small **operationalization task**, not a new research
question.

---

### 18.11 Optional relative + blind bootstrap demonstration

If the goal is to show the **whole localization system** rather than absolute
frontend accuracy only, also run:

~~~text
relative frontend
        +
DINO Top20
        +
SIFT+LightGlue absolute observations
        +
minimum_confident_v2
        +
map alignment / temporal authority
~~~

For a new flight, preserve:

- reference unavailable to localization;
- Top4 observation hypothesis interface;
- causal evidence only;
- HOLD/relative-only behavior if no lock is trustworthy;
- output freeze before post-run evaluation.

Do not force a map lock simply to make the demonstration look successful.

A safe no-lock result is preferable to a confident wrong locality.

---

### 18.12 Post-freeze evaluation bundle

When hidden reference is available, attach it only after the reference-free
output is frozen.

Every new dataset report should include at minimum:

#### Dataset / geometry

- query count;
- duration;
- path length if reference permits;
- camera orientation regime;
- altitude/AGL range;
- image resolution;
- map source;
- map date;
- CRS;
- source map GSD;
- tile physical footprint;
- tile stride.

#### Retrieval

- Top1/5/10/20 useful-candidate availability;
- containment availability;
- center <=40 / <=80 availability.

#### SIFT+LightGlue

- selected containment;
- selected tile-center <=40 / <=80;
- selected projected <=40 / <=80;
- median;
- RMSE;
- p95;
- unsafe >200/non-finite;
- Top20 oracle values;
- selected matches/inliers;
- Top1-Top2 inlier margin.

#### Full system when executed

- relative-only drift;
- lock state;
- first leader;
- maturity query;
- accepted/HOLD updates;
- final trajectory RMSE/p95/final error/drift;
- percentage of map-aligned poses.

#### Runtime

- descriptor-cache time;
- retrieval/query;
- SIFT extraction/query;
- LightGlue/pair;
- candidate count/query;
- total absolute-localization latency;
- CPU/GPU/device.

---

### 18.13 Required visuals for a team presentation

Produce a compact evidence set rather than screenshots chosen after seeing the
answer.

Recommended:

1. dataset contact sheet;
2. map AOI + reference path if evaluation is allowed;
3. DINO Top1/Top5/Top20 retrieval examples;
4. SIFT selected candidate + correspondence overlay;
5. estimated vs reference path;
6. position error versus distance/time;
7. failure examples;
8. runtime summary;
9. one architecture diagram;
10. comparison table against the frozen stable120m baseline.

For oblique data also show the camera-view geometry caveat.

For vegetation-heavy failures include both selected and alternative/oracle
candidate evidence when post-freeze truth is authorized.

---

### 18.14 Stop / sanity gates

Do not proceed blindly through the whole pipeline.

Stop and investigate if:

- frame/SRT timestamps are badly misaligned;
- image read failures occur;
- map CRS is missing/wrong;
- map does not cover the flight;
- DINO query/map descriptor protocol differs;
- Top20 availability collapses;
- SIFT produces widespread non-finite homographies;
- the new dataset uses a materially different camera modality;
- a reference-bearing column accidentally enters ranking;
- outputs from an older run would be overwritten.

A weak result is still scientifically useful if the run contract is clean.

---

### 18.15 Fast smoke before a full expensive run

For a completely new camera/domain, do **not** immediately spend hours on
Top20 for every frame.

First run approximately:

~~~text
20 representative queries
x
Top5 candidates
=
100 pairs
~~~

Use uniformly sampled queries, not hand-picked easy frames.

Continue to full Top20 only if:

- images decode correctly;
- DINO retrieval is not obviously collapsed;
- SIFT produces usable correspondences/homographies;
- runtime is understood;
- the map/query modality is compatible.

This rule is especially important for:

- thermal;
- night imagery;
- strongly oblique imagery;
- different sensor/FOV;
- very different altitude;
- different map season/source.

---

### 18.16 What should be frozen for every future team dataset

Create one run root containing:

~~~text
config snapshot
Git commit
environment summary
query manifest
map/tile metadata
DINO query-cache metadata
Top20 candidate CSV
SIFT pair evidence
selected-observation CSV
reference-free freeze report + SHA256s
post-freeze evaluation report
runtime/resource report
figures
notes / known limitations
~~~

The point is that a future team member should be able to answer:

> Which code, map, model, parameters and data produced this number?

without depending on chat history.

---

### 18.17 Canonical one-command current-stack runner

The former orchestration gap is now closed by:

~~~text
scripts/villoc/run_localization_pipeline.py
configs/localization/current_stack_reference_free.yaml
~~~

The historical ORB/demo orchestrator is intentionally preserved unchanged for
reproducibility. The new runner reuses its proven ingestion, relative-motion,
DINO retrieval, map-state, temporal-authority, export, visualization and freeze
stages, but replaces the hard-coded ORB verifier with the promoted
SIFT+LightGlue projected-candidate interface.

~~~mermaid
flowchart LR
    V[Recorded RGB video] --> Q[Reference-free query manifest]
    Q --> X[XFeat relative trajectory]
    Q --> D[DINO img518 query cache]
    M[768_s256 map cache] --> R[DINO Top20]
    D --> R
    R --> S[SIFT+LightGlue Top20 verification]
    S --> C[Canonical projected-candidate evidence]
    C --> T[Top4 minimum_confident_v2]
    X --> T
    T --> A[Causal map-state timeline]
    A --> O[Estimated trajectory]
    O --> F[Reference-free freeze]
    F --> P[Neutral canonical run package]
~~~

Dry-run first:

~~~bash
source .drone_venv/bin/activate
export PYTHONPATH=$PWD/src

python scripts/villoc/run_localization_pipeline.py \
  --config configs/localization/current_stack_reference_free.yaml \
  --video data/raw/villoc/<dataset>/<video>.MP4 \
  --run-id <unique_run_id> \
  --dry-run
~~~

Then execute:

~~~bash
python scripts/villoc/run_localization_pipeline.py \
  --config configs/localization/current_stack_reference_free.yaml \
  --video data/raw/villoc/<dataset>/<video>.MP4 \
  --run-id <unique_run_id>
~~~

Default execution devices preserve the frozen CPU absolute-frontend contract.
The runner can select CPU/CUDA for DINO query encoding and XFeat, but
SIFT+LightGlue remains CPU-only in this canonical command until CUDA parity is
separately demonstrated.

Canonical project-facing outputs are:

~~~text
metadata/localization_query_manifest.csv
absolute/geometric_candidate_evidence.csv
absolute/selected_absolute_observations.csv
absolute/absolute_query_summary.csv
map_state/map_state.json
map_state/map_state_timeline.json
map_state/temporal_authority_timeline.csv
trajectories/estimated_trajectory.csv
provenance/output_freeze.json
provenance/run_manifest.json
summary/localization_run_summary.md
~~~

A few historical filenames such as blind_query_manifest.csv and
submission_estimated_trajectory.csv are still created internally because frozen
downstream helpers consume them. They are compatibility artifacts, not the
preferred project-facing names.

The bootstrap no longer recomputes ORB geometry under the SIFT stack.
SIFT+LightGlue owns candidate homography/projection; minimum_confident_v2
consumes those already-projected candidates and owns only the four-frame
similarity hypothesis plus causal ACQUISITION/TRACKING authority.

For a scientific benchmark with hidden GT/SRT, run and freeze this pipeline
first. Attach the reference only afterward using a separate evaluation stage.
The reference-free runner intentionally has no GT/SRT/GPS input path.

Before a completely new sensor/domain, still use the 20-query x Top5 smoke
described in Section 18.15 rather than immediately spending a full Top20 run.

---

## 19. Immediate development direction

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
