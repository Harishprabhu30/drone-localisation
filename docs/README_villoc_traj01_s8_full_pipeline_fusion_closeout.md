# Villoc traj01 S8 Full Pipeline and Relative–Absolute Fusion Closeout

**Project:** GNSS-denied / weak-GNSS UAV visual localization  
**Dataset:** Villoc `traj01_90deg_stable120m`  
**Config:** `configs/dataset_villoc_traj01_90deg_stable120m.yaml`  
**Primary output root:** `outputs/villoc/traj01_90deg_stable120m/`  
**Stage covered:** S8.1 → S8.12E, S8.R relative frontend, S8.F relative–absolute fusion  
**Final experimental status:** Complete for Villoc traj01  
**Recommended final method:** XFeat relative odometry + temporal-consistency-gated ORB-reranked absolute correction, soft correction weight `alpha=0.25`  
**Final report label:** **Temporal-consistency fusion**

---

## 0. Executive conclusion

Villoc traj01 is now a complete end-to-end recorded-data GNSS-denied localization experiment.

The working chain is:

```text
Raw DJI video + SRT telemetry
        ↓
1 fps query extraction + reference trajectory
        ↓
orthophoto AOI/tile reuse + geometric oracle audit
        ↓
DINOv2 satellite retrieval
        ↓
ORB Top-20 absolute reranking
        ↓
XFeat relative visual odometry
        ↓
relative + absolute fusion
        ↓
temporal-consistency-gated soft correction
```

The most important final results are:

| Method | Correction style | RMSE | p95 | Final error | Drift |
|---|---:|---:|---:|---:|---:|
| XFeat relative-only | no map correction | 35.76 m | 46.24 m | 39.57 m | 2.32 m/100 m |
| Best RMSE fusion | periodic 50 m + strict gate, `alpha=0.25` | **13.84 m** | **25.24 m** | 15.69 m | 0.92 m/100 m |
| Recommended realistic fusion | temporal-consistency gate, `alpha=0.25` | 14.02 m | 30.16 m | **9.09 m** | **0.53 m/100 m** |
| Oracle soft upper bound | eval-only true anchors, `alpha=0.25` | 14.28 m | 25.25 m | 13.97 m | 0.82 m/100 m |

Recommended report conclusion:

```text
Relative odometry provides smooth short-term motion but drifts over distance.
Absolute image-to-map localization provides occasional global anchors but is not
safe on every frame. The final system therefore uses absolute localization as a
sparse, confidence-gated drift correction source. Temporal consistency further
checks whether a proposed map correction agrees with recent relative motion
before applying a soft correction.
```

---

## 1. Locked rule: what is online vs evaluation-only

This rule must stay explicit in code comments, README, report, and presentation.

### Online-usable signals

The online/casual system may use:

```text
UAV image
satellite tile image
DINO descriptor similarity / original rank
ORB verifier score
ORB good matches / inliers / inlier ratio / query coverage
XFeat relative displacement
distance travelled according to relative trajectory
previous accepted correction state
```

### Evaluation-only signals

The online system must not use:

```text
reference_x_m / reference_y_m
latitude / longitude as correction truth
ground-truth error
oracle flags
hit_le_40m labels
body containment labels
dangerous false labels
```

Those are used only after the run to evaluate whether the online decision was good.

---

## 2. Canonical paths

```bash
cd /Users/harishprabhu/Documents/drone-localisation
source .drone_venv/bin/activate
export PYTHONPATH=$PWD/src
```

```text
CFG=configs/dataset_villoc_traj01_90deg_stable120m.yaml
ROOT=outputs/villoc/traj01_90deg_stable120m
RAW=data/raw/villoc/traj01_90deg_stable120m
PROC=data/processed/villoc/traj01_90deg_stable120m
OLD90=outputs/villoc/90_deg
TAG=dinov2_vits14_img518_center_square_avgpatch_cpu
PRIMARY_VARIANT=512_s256
```

Primary current dataset files:

```text
Raw video:
data/raw/villoc/traj01_90deg_stable120m/
  villoc_traj01_90deg_stable120m_V_merged.MP4
  villoc_traj01_90deg_stable120m_V_merged.SRT

Frames:
data/processed/villoc/traj01_90deg_stable120m/frames_v_1fps/

Canonical query manifest:
outputs/villoc/traj01_90deg_stable120m/metadata/s8_10b_canonical_uav_query_manifest.csv

DINO query cache:
outputs/villoc/traj01_90deg_stable120m/descriptors/s8_11c_dinov2_queries_v_1fps_dinov2_vits14_img518_center_square_avgpatch_cpu.npz

Reused map/tile assets:
outputs/villoc/90_deg/metadata/s8_9_satellite_tile_index_512_s256.csv
outputs/villoc/90_deg/metadata/s8_9_satellite_tile_index_1024_s512.csv
outputs/villoc/90_deg/metadata/s8_9_satellite_tile_index_1024_s256.csv
outputs/villoc/90_deg/descriptors/
```

---

## 3. End-to-end execution order

Use this as the master order.

```text
S8.1–S8.5   Dataset parse, frames, trajectory, visual audit, UAV index
S8.6        AOI reuse sanity
S8.10B      geometric oracle / tile coverage audit
S8.11BC     DINOv2 cache build
S8.11D      independent DINO retrieval
S8.12D      retrieval diagnostics
S8.12E.1    ORB Top-20 absolute reranker
S8.12E.1B   LightGlue Top-20 diagnostic only
S8.R1–R2    ORB relative baseline
S8.R3       KLT comparison baseline
S8.R4       XFeat relative frontend
S8.F1       absolute correction manifest
S8.F2       controlled fusion replay
S8.F3       temporal-agreement gating
S8.F3B      temporal-gated fusion replay
S8.F4       closeout, asset copy, report figures
```

Important branch decision:

```text
If AOI reuse passes:
    skip S8.7/S8.8/S8.9/S8.10A and reuse old 90_deg map/tile assets.

If AOI reuse fails:
    rebuild AOI crop, tile databases, tile index, and map descriptors.

If DINO map cache with requested TAG already exists:
    reuse map descriptors.

If image size / crop / pooling changes:
    rebuild map descriptors for that TAG.

If LightGlue full underperforms:
    keep LightGlue as diagnostic only, do not promote it for fusion.

If XFeat fails or becomes unstable:
    use ORB relative trajectory as fallback relative frontend.

If temporal fusion does not improve RMSE:
    still keep it as a safety/realism comparison if it reduces false corrections or final drift.
```

---

# Part A — Dataset and geometric setup

## 4. S8.1 — SRT parse and cleaning

### Purpose

Convert DJI SRT telemetry into clean time-indexed records.

### Final result

```text
Raw rows: 12062
Clean rows: 12061
Video time: 0.000–402.423 s
Relative altitude: 118.74–120.153 m
Pitch median: -90.0°
Yaw range: -83.1° to 31.4°
Status: PASS_S8_1_CLEAN
```

### Interpretation

The dataset is a stable near-nadir 120 m altitude flight. This is suitable for visual map localization because scale and viewpoint are more stable than in the earlier 45° oblique experiment.

---

## 5. S8.2 — 1 fps frame extraction

### Purpose

Extract a manageable set of query frames while preserving the trajectory.

### Final result

```text
Samples: 403
Extracted: 403 / 403
Median alignment: 8 ms
Max alignment: 17 ms
Status: PASS_S8_2
```

### Output

```text
data/processed/villoc/traj01_90deg_stable120m/frames_v_1fps/
```

---

## 6. S8.3 — Reference trajectory

### Purpose

Build the evaluation trajectory in local metric coordinates.

### Final result

```text
Rows: 403
2D path length: 1962.73 m
Start–end displacement: 145.66 m
Relative altitude range: 118.78–120.12 m
Status: PASS_S8_3
```

### Note

This trajectory is used only for evaluation and plotting. It must not be used for online correction decisions.

---

## 7. S8.4 — Visual audit

### Purpose

Check image readability, sharpness, brightness, contrast, edge density, yaw, pitch, and altitude.

### Final result

```text
Images read: 403 / 403
Laplacian median: 1443.93
Brightness median: 104.45
Contrast median: 66.97
Edge density median: 0.1832
Status: PASS_S8_4
```

### Key figure

```text
outputs/villoc/traj01_90deg_stable120m/figures/s8_4_visual_audit/s8_4_combined_quality_yaw_V_1fps.png
```

### Interpretation

The sequence is visually strong. The localization difficulty is not mainly blur; it comes from repeated map structures, vegetation/urban texture changes, visually similar streets/roofs/fields, and global retrieval ambiguity.

---

## 8. S8.5 — UAV frame index

### Purpose

Build a canonical index connecting query frames, `token0_id`, SRT timing, image paths, and reference/evaluation coordinates.

### Known patch

The script needed a DataFrame assignment fix:

```python
out = pd.DataFrame(index=df.index)
out["dataset_name"] = [dataset_name] * len(df)
out["sequence_name"] = [sequence_name] * len(df)
out["view_angle_group"] = [view_angle_group] * len(df)
```

### Expected output

```text
outputs/villoc/traj01_90deg_stable120m/metadata/s8_5_uav_frames_index_v_1fps.csv
```

### Status

```text
PASS_S8_5
```

---

## 9. S8.6 — AOI reuse sanity

### Purpose

Check if this new trajectory lies inside the earlier Villoc 90° AOI and can reuse the old orthophoto crop and tile database.

### Result

```text
Queries inside existing AOI: 403 / 403
Outside: 0
West margin: 39.59 m
Status: PASS_S8_6_EXISTING_AOI_REUSE_SANITY
```

### Branch decision

```text
AOI reuse passed.
Do not rebuild S8.7/S8.8/S8.9/S8.10A for traj01.
Reuse outputs/villoc/90_deg map tiles and map descriptors.
```

If AOI reuse had failed, the branch would be:

```text
S8.7 crop new orthophoto AOI
S8.8 generate tile grids
S8.9 build satellite tile index
S8.10A validate tile integrity
S8.11BC rebuild map descriptors
```

---

## 10. S8.10B — Canonical query manifest and oracle tile audit

### Purpose

Audit geometric map coverage for every query and every tile variant before retrieval.

### Run command

```bash
cd /Users/harishprabhu/Documents/drone-localisation
source .drone_venv/bin/activate
export PYTHONPATH=$PWD/src

python scripts/villoc/s8_10b_uav_tile_oracle_audit.py \
  --config configs/dataset_villoc_traj01_90deg_stable120m.yaml \
  --src-tif data/processed/villoc/90_deg/maps/ort10lt_2024_2026/ort10lt_2024_2026_aoi300m.tif \
  --uav-index-csv outputs/villoc/traj01_90deg_stable120m/metadata/s8_5_uav_frames_index_v_1fps.csv \
  --trajectory-csv outputs/villoc/traj01_90deg_stable120m/trajectories/s8_3_reference_trajectory_V_1fps.csv \
  --trajectory-crs EPSG:3346 \
  --variant "512_s256:outputs/villoc/90_deg/metadata/s8_9_satellite_tile_index_512_s256.csv" \
  --variant "1024_s512:outputs/villoc/90_deg/metadata/s8_9_satellite_tile_index_1024_s512.csv" \
  --variant "1024_s256:outputs/villoc/90_deg/metadata/s8_9_satellite_tile_index_1024_s256.csv"
```

### Outputs

```text
outputs/villoc/traj01_90deg_stable120m/metadata/s8_10b_canonical_uav_query_manifest.csv
outputs/villoc/traj01_90deg_stable120m/metadata/s8_10b_uav_tile_oracle_summary.csv
outputs/villoc/traj01_90deg_stable120m/reports/s8_10b_uav_tile_oracle_audit.json
```

### Final result

```text
Status: PASS_UAV_TILE_ORACLE_AUDIT
Queries: 403
```

### Variant audit

| Variant | Tiles | Median nearest-center error | p95 nearest-center error | Interpretation |
|---|---:|---:|---:|---|
| `512_s256` | 475 | 20.94 m | 30.74 m | best strict 40 m center geometry |
| `1024_s512` | 108 | 40.61 m | 62.88 m | large tile / strong candidate generator |
| `1024_s256` | 391 | 21.32 m | 34.49 m | dense overlap, not selected later |

---

# Part B — Absolute localization branch

## 11. S8.11BC — DINOv2 cache build

### Purpose

Build query descriptors for the 403 UAV frames and map descriptors for reused old Villoc 90° tiles.

### Run command

```bash
cd /Users/harishprabhu/Documents/drone-localisation
source .drone_venv/bin/activate
export PYTHONPATH=$PWD/src

CFG=configs/dataset_villoc_traj01_90deg_stable120m.yaml
ROOT=outputs/villoc/traj01_90deg_stable120m

mkdir -p "$ROOT/logs/s8_11bc_dinov2_caches"

python scripts/villoc/s8_11bc_build_dinov2_caches.py \
  --config "$CFG" \
  --reuse-map-caches \
  --map-cache-root outputs/villoc/90_deg/descriptors \
  --batch-size 1 \
  --image-size 518 \
  --crop-mode center_square \
  --pooling avgpatch \
  2>&1 | tee "$ROOT/logs/s8_11bc_dinov2_caches/s8_11bc_build_query_cache_reuse_map.log"
```

### Final result

```text
Status: PASS_DESCRIPTOR_CACHES_BUILT
Query descriptors: 403 × 384
Map descriptors:
  512_s256: 475 × 384
  1024_s512: 108 × 384
  1024_s256: 391 × 384
```

### Note

Because the run used `image_size=518`, new img518 map descriptors were built under the old `outputs/villoc/90_deg/descriptors` cache root. This is expected and valid.

---

## 12. S8.11D — Independent DINO retrieval

### Purpose

Evaluate pure descriptor retrieval before any geometric reranking.

### Run command

```bash
cd /Users/harishprabhu/Documents/drone-localisation
source .drone_venv/bin/activate
export PYTHONPATH=$PWD/src

CFG=configs/dataset_villoc_traj01_90deg_stable120m.yaml
ROOT=outputs/villoc/traj01_90deg_stable120m
TAG=dinov2_vits14_img518_center_square_avgpatch_cpu

python scripts/villoc/s8_11d_independent_dinov2_retrieval.py \
  --config "$CFG" \
  --image-size 518 \
  --crop-mode center_square \
  --pooling avgpatch \
  --device-tag cpu \
  --map-cache-root outputs/villoc/90_deg/descriptors \
  2>&1 | tee "$ROOT/logs/s8_11d_independent_retrieval_img518.log"
```

### Key outputs

```text
outputs/villoc/traj01_90deg_stable120m/retrieval/s8_11d/
outputs/villoc/traj01_90deg_stable120m/reports/s8_11d/
```

### Final result

| Variant | R@1 | R@5 | R@20 | Top1 <=40 m | Median error |
|---|---:|---:|---:|---:|---:|
| `512_s256` | 0.362 | 0.625 | 0.831 | 114 / 403 | 332.04 m |
| `1024_s512` | 0.251 | 0.578 | 0.940 | 13 / 403 | 421.24 m |
| `1024_s256` | 0.270 | 0.459 | 0.821 | 33 / 403 | 504.10 m |

### Interpretation

`512_s256` is the best direct Top-1 localizer.  
`1024_s512` is useful as a broad candidate generator but not best for strict center error.  
Pure DINO retrieval is not enough as final localization because Top-1 is often globally wrong.

---

## 13. S8.12D — Retrieval diagnostics

### Purpose

Analyze why Top-1 fails, where the oracle is located inside Top-K, and whether reranking has enough candidate coverage.

### Run command pattern

```bash
ROOT=outputs/villoc/traj01_90deg_stable120m
TAG=dinov2_vits14_img518_center_square_avgpatch_cpu

python scripts/villoc/s8_retrieval_diagnostics/s8_retrieval_diagnostics.py \
  --query-manifest "$ROOT/metadata/s8_10b_canonical_uav_query_manifest.csv" \
  --topk-csv "$ROOT/retrieval/s8_11d/s8_11d_topk_512_s256_${TAG}.csv" \
  --out-root "$ROOT/reports/s8_12d_retrieval_diagnostics/512_s256_img518" \
  --variant 512_s256 \
  --hit-threshold-m 40
```

Repeat for:

```text
1024_s512
1024_s256
```

### Result summary

| Variant | good Top-1 | bad Top-1 | oracle in Top20 not Top1 | oracle missing Top20 | high-conf wrong Top1 |
|---|---:|---:|---:|---:|---:|
| `512_s256` | 146 | 257 | 189 | 68 | 15 |
| `1024_s512` | 101 | 302 | 278 | 24 | 28 |
| `1024_s256` | 109 | 294 | 222 | 72 | 19 |

### Interpretation

The correct or near-correct tile is often inside the candidate list, especially for `512_s256` and `1024_s512`, but DINO does not rank it first reliably. This motivates local geometric reranking.

---

## 14. S8.12E.1 — ORB Top-20 verifier/reranker

### Purpose

Use ORB + RANSAC as a local geometric verifier over DINO Top-20 candidates.

This is still not “ORB alone.”  
The method is:

```text
DINOv2 retrieves candidate tiles
        ↓
ORB checks local geometric consistency inside Top-20
        ↓
hybrid score combines ORB verifier score + DINO rank prior
```

### Run command: primary `512_s256`

```bash
cd /Users/harishprabhu/Documents/drone-localisation
source .drone_venv/bin/activate
export PYTHONPATH=$PWD/src

ROOT=outputs/villoc/traj01_90deg_stable120m
TAG=dinov2_vits14_img518_center_square_avgpatch_cpu
VARIANT=512_s256

python scripts/villoc/s8_12e1_top20_verifier_reranker.py \
  --config configs/dataset_villoc_traj01_90deg_stable120m.yaml \
  --variant "$VARIANT" \
  --tag "$TAG" \
  --top-n 20 \
  --policy hybrid \
  --hit-threshold-m 40 \
  --query-csv "$ROOT/metadata/s8_10b_canonical_uav_query_manifest.csv" \
  --topk-csv "$ROOT/retrieval/s8_11d/s8_11d_topk_${VARIANT}_${TAG}.csv" \
  --tile-index-csv "outputs/villoc/90_deg/metadata/s8_9_satellite_tile_index_${VARIANT}.csv" \
  --out-root "$ROOT/reports/s8_12e1_top20_verifier_reranker/${VARIANT}_orb_hybrid_top20_img518" \
  2>&1 | tee "$ROOT/logs/s8_12e1_${VARIANT}_orb_hybrid_top20_img518.log"
```

Repeat with:

```text
VARIANT=1024_s512
```

### Final result

| Variant | DINO Top1 <=40 m | ORB-reranked Top1 <=40 m | DINO median | ORB-reranked median |
|---|---:|---:|---:|---:|
| `512_s256` | 114 / 403 | **173 / 403** | 332.04 m | **45.46 m** |
| `1024_s512` | 13 / 403 | 69 / 403 | 421.24 m | 88.04 m |

### Interpretation

The best absolute correction source is:

```text
512_s256 + ORB hybrid Top-20 reranking
```

This is selected for fusion.

---

## 15. S8.12E.1B — LightGlue diagnostic

### Purpose

Test whether LightGlue is a better learned geometric verifier over DINO Top-20 candidates.

### Smoke result

The first 20 frames looked excellent for `512_s256`:

```text
LG <=40 m: 19 / 20
improved: 19
worsened: 0
```

### Full run command

```bash
cd /Users/harishprabhu/Documents/drone-localisation
source .drone_venv/bin/activate
export PYTHONPATH=$PWD/src

CFG=configs/dataset_villoc_traj01_90deg_stable120m.yaml
ROOT=outputs/villoc/traj01_90deg_stable120m
TAG=dinov2_vits14_img518_center_square_avgpatch_cpu
VARIANT=512_s256

mkdir -p "$ROOT/logs/s8_12e1b_lightglue_top20"

caffeinate -dimsu python scripts/villoc/s8_12e1b_lightglue_top20_verifier_reranker.py \
  --config "$CFG" \
  --variant "$VARIANT" \
  --tag "$TAG" \
  --top-n 20 \
  --policy hybrid \
  --hit-threshold-m 40 \
  --query-csv "$ROOT/metadata/s8_10b_canonical_uav_query_manifest.csv" \
  --topk-csv "$ROOT/retrieval/s8_11d/s8_11d_topk_${VARIANT}_${TAG}.csv" \
  --tile-index-csv "outputs/villoc/90_deg/metadata/s8_9_satellite_tile_index_${VARIANT}.csv" \
  --out-root "$ROOT/reports/s8_12e1b_lightglue_top20/${VARIANT}_lg_hybrid_top20_img518_full403" \
  --device cpu \
  --max-keypoints 780 \
  --resize-long 1024 \
  --ransac-thresh 5.0 \
  --dino-prior-weight 2.0 \
  --checkpoint-every 200 \
  --progress-every 40 \
  --num-threads 1 \
  2>&1 | tee "$ROOT/logs/s8_12e1b_lightglue_top20/s8_12e1b_${VARIANT}_lg_hybrid_top20_img518_full403.log"
```

If `--dino-prior-weight` is not recognized:

```bash
# replace:
--dino-prior-weight 2.0
# with:
--rank-prior-weight 2.0
```

### Full result

```text
DINO Top-1 <=40m: 114 / 403
LG Top-1 <=40m: 101 / 403
DINO median error: 332.04 m
LG median error: 164.02 m
Best Top20 median error: 22.40 m
improved / worsened / equal: 148 / 181 / 74
Status: PASS_LIGHTGLUE_TOPK_VERIFIER_RERANKER
```

### Interpretation

LightGlue improved the median error compared with raw DINO, but reduced strict Top-1 <=40 m accuracy and worsened more rows than it improved. It is therefore kept as a diagnostic verifier, not promoted for final fusion.

---

# Part C — Relative localization branch

## 16. S8.R1 — ORB affine stride diagnostics

### Purpose

Evaluate whether frame-to-frame ORB motion is reliable enough for visual odometry.

### Script

```text
scripts/satloc/s6a/s6a_1_orb_affine_stride_diagnostics.py
```

### Result

```text
Stride 1:
  success: 402 / 402
  good-quality: 1.000
  good matches median: 707.5
  inliers median: 640.5
  inlier ratio median: 0.900
  center motion median: 24.114 px
  reference step median: 4.415 m

Status: PASS_S8R1_ORB_AFFINE
```

### Interpretation

ORB is a very strong relative frontend baseline for this stable nadir sequence.

---

## 17. S8.R2 — ORB relative trajectory and drift

### Purpose

Integrate frame-to-frame ORB motion into a relative trajectory and evaluate drift after a prefix alignment.

### Script

```text
scripts/satloc/s6a/s6a_2_orb_relative_trajectory_and_drift.py
```

### Result

| Variant | RMSE | p95 | Final error | Drift |
|---|---:|---:|---:|---:|
| `se2_scale_normalized` | 38.97 m | 51.99 m | 56.11 m | 3.29 m/100 m |
| `sim2_local_step` | **38.46 m** | **50.70 m** | **55.18 m** | **3.24 m/100 m** |

### Interpretation

ORB is a reliable classical relative baseline. It is not the final selected relative frontend because XFeat later improves drift.

---

## 18. S8.R3 — KLT comparison

### Purpose

Test whether dense-ish KLT tracking produces better integrated drift than ORB.

### Script

```text
scripts/satloc/s6a/s6a_4a_klt_vs_orb_relative_comparison_v2.py
```

### Result

```text
KLT affine success: 1.000
KLT good rate: 0.965
KLT inliers median: 1084
KLT inlier ratio median: 0.957

But:
KLT RMSE: 323.97 m
KLT drift: 17.97 m/100 m
```

### Interpretation

KLT has excellent pairwise statistics but poor accumulated trajectory drift. It is rejected as a fusion frontend.

Report insight:

```text
More pairwise matches do not guarantee better long-horizon odometry.
```

---

## 19. S8.R4 — XFeat relative frontend

### Purpose

Evaluate learned local features for relative visual odometry.

### Run command

```bash
cd /Users/harishprabhu/Documents/drone-localisation
source .drone_venv/bin/activate
export PYTHONPATH=$PWD/src

ROOT=outputs/villoc/traj01_90deg_stable120m

python scripts/villoc/s8_r4_xfeat_relative_frontend.py \
  --manifest "$ROOT/metadata/s6a_relative_motion/s6a_sequence_manifest.csv" \
  --output-root "$ROOT" \
  --xfeat-repo third_party/accelerated_features \
  --orb-pairs "$ROOT/metadata/s6a_relative_motion/s6a1_orb_affine_pair_diagnostics.csv" \
  --orb-aligned "$ROOT/metadata/s6a_relative_motion/s6a2_orb_relative_trajectory_aligned_eval_only.csv" \
  --orb-summary "$ROOT/metadata/s6a_relative_motion/s6a2_orb_relative_trajectory_summary.csv" \
  --sequence traj01 \
  --device cpu \
  --resize-long 960 \
  --top-k 1200 \
  --detection-threshold 0.05 \
  --min-cossim 0.82 \
  --ransac-threshold 3.0 \
  --alignment-prefix-frames 50 \
  --thresholds-m 10,20,40,80,120 \
  --sustain-frames 5 \
  2>&1 | tee "$ROOT/logs/s6a_relative_motion/s8r4_xfeat_full403.log"
```

### Outputs

```text
outputs/villoc/traj01_90deg_stable120m/metadata/s8_xfeat_relative_frontend/s8r4_xfeat_frame_features.csv
outputs/villoc/traj01_90deg_stable120m/metadata/s8_xfeat_relative_frontend/s8r4_xfeat_pair_diagnostics.csv
outputs/villoc/traj01_90deg_stable120m/metadata/s8_xfeat_relative_frontend/s8r4_xfeat_relative_trajectory_aligned_eval_only.csv
outputs/villoc/traj01_90deg_stable120m/metadata/s8_xfeat_relative_frontend/s8r4_xfeat_orb_comparison.csv
outputs/villoc/traj01_90deg_stable120m/reports/s8_xfeat_relative_frontend/s8r4_xfeat_relative_frontend_report.json
outputs/villoc/traj01_90deg_stable120m/figures/s8_xfeat_relative_frontend/
```

### Final result

```text
Status: PASS_XFEAT_RELATIVE_FRONTEND
Frames: 403
Pairs: 402
Affine success: 1.0
Good quality: 0.910
Inliers median: 358
RMSE: 35.76 m
p95: 46.24 m
Final error: 39.57 m
Drift: 2.32 m/100 m
```

### Interpretation

XFeat is selected as the main relative trajectory because it improves long-horizon drift compared with ORB.

---

# Part D — Relative + absolute fusion

## 20. S8.F0 — Fusion input audit

### Purpose

Check that the relative trajectory, absolute reranker output, candidate score table, and query manifest align.

### Files confirmed

```text
XFeat relative:
outputs/villoc/traj01_90deg_stable120m/metadata/s8_xfeat_relative_frontend/s8r4_xfeat_relative_trajectory_aligned_eval_only.csv
Rows: 403

ORB absolute query summary:
outputs/villoc/traj01_90deg_stable120m/reports/s8_12e1_top20_verifier_reranker/512_s256_orb_hybrid_top20_img518/s8_12e1_query_summary.csv
Rows: 403

ORB candidate scores:
outputs/villoc/traj01_90deg_stable120m/reports/s8_12e1_top20_verifier_reranker/512_s256_orb_hybrid_top20_img518/s8_12e1_all_candidate_verifier_scores.csv
Rows: 8060

Canonical query manifest:
outputs/villoc/traj01_90deg_stable120m/metadata/s8_10b_canonical_uav_query_manifest.csv
Rows: 403
```

### Interpretation

All required fusion inputs are valid.

---

## 21. S8.F1 — Absolute correction manifest

### Purpose

Build a single correction manifest joining:

```text
XFeat relative state
query/frame identity
ORB-reranked absolute tile center
online confidence evidence
evaluation-only correctness labels
candidate correction policies
```

### Script

```text
scripts/villoc/s8_fusion/s8_f1_build_absolute_correction_manifest.py
```

### Run command

```bash
cd /Users/harishprabhu/Documents/drone-localisation
source .drone_venv/bin/activate
export PYTHONPATH=$PWD/src

ROOT=outputs/villoc/traj01_90deg_stable120m

python scripts/villoc/s8_fusion/s8_f1_build_absolute_correction_manifest.py \
  --root "$ROOT" \
  --intervals-m 50,100,200,400 \
  2>&1 | tee "$ROOT/logs/s8_relative_absolute_fusion/s8_f1_absolute_correction_manifest.log"
```

### Key outputs

```text
outputs/villoc/traj01_90deg_stable120m/metadata/s8_relative_absolute_fusion/s8_f1_absolute_correction_manifest.csv
outputs/villoc/traj01_90deg_stable120m/metadata/s8_relative_absolute_fusion/s8_f1_policy_summary.csv
outputs/villoc/traj01_90deg_stable120m/metadata/s8_relative_absolute_fusion/s8_f1_policy_gaps.csv
outputs/villoc/traj01_90deg_stable120m/reports/s8_relative_absolute_fusion/s8_f1_absolute_correction_manifest_report.json
outputs/villoc/traj01_90deg_stable120m/figures/s8_relative_absolute_fusion/s8_f1_policy_correction_distribution.png
```

### Result summary

| Policy | Accepted | True <=40 m | False | Dangerous >100 m | Precision |
|---|---:|---:|---:|---:|---:|
| all reranked ablation | 403 | 173 | 230 | 133 | 0.429 |
| strict A | 64 | 40 | 24 | 0 | 0.625 |
| strict B | 16 | 14 | 2 | 0 | 0.875 |
| interval 50 m + strict A | 8 | 6 | 2 | 0 | 0.750 |
| interval 100 m + strict A | 4 | 4 | 0 | 0 | 1.000 |
| oracle hit40 eval-only | 173 | 173 | 0 | 0 | 1.000 |

### Explanation of policy names

#### `all_reranked_accept_ablation`

Accept every ORB-reranked absolute result at every frame.  
This is not a real system. It is an ablation to prove that blind absolute correction is unsafe.

#### `strict_a_accept_online`

Accept only if ORB evidence passes a moderate confidence gate.

#### `strict_b_accept_online`

Accept only if ORB evidence passes a stronger confidence gate.

#### `interval_50m_strict_a_accept_online`

Attempt corrections roughly every 50 m and accept only if `strict_a` passes.

#### `oracle_hit40_accept_eval_only`

Accept only corrections known after evaluation to be within 40 m. This is an upper-bound, not an online method.

---

## 22. S8.F2 — Controlled fusion replay

### Purpose

Replay XFeat relative trajectory with absolute corrections applied as hard or soft resets.

Correction formula:

```text
fused_position_after_correction =
    (1 - alpha) * current_relative_position + alpha * absolute_position
```

Interpretation:

```text
alpha = 1.00  hard reset
alpha = 0.75  strong correction
alpha = 0.50  half correction
alpha = 0.25  soft drift anchor
```

### Script

```text
scripts/villoc/s8_fusion/s8_f2_controlled_fusion_replay.py
```

### Run command

```bash
cd /Users/harishprabhu/Documents/drone-localisation
source .drone_venv/bin/activate
export PYTHONPATH=$PWD/src

ROOT=outputs/villoc/traj01_90deg_stable120m

python scripts/villoc/s8_fusion/s8_f2_controlled_fusion_replay.py \
  --root "$ROOT" \
  --alphas 1.0,0.75,0.5,0.25 \
  --eval-start-index 49 \
  --thresholds-m 10,20,40,80,120 \
  --sustain-frames 5 \
  2>&1 | tee "$ROOT/logs/s8_relative_absolute_fusion/s8_f2_controlled_fusion_replay.log"
```

### Key outputs

```text
outputs/villoc/traj01_90deg_stable120m/metadata/s8_relative_absolute_fusion/s8_f2_fusion_replay_trajectory.csv
outputs/villoc/traj01_90deg_stable120m/metadata/s8_relative_absolute_fusion/s8_f2_fusion_replay_summary.csv
outputs/villoc/traj01_90deg_stable120m/metadata/s8_relative_absolute_fusion/s8_f2_fusion_threshold_crossings.csv
outputs/villoc/traj01_90deg_stable120m/reports/s8_relative_absolute_fusion/s8_f2_controlled_fusion_replay_report.json
outputs/villoc/traj01_90deg_stable120m/figures/s8_relative_absolute_fusion/s8_f2_fusion_error_vs_distance.png
outputs/villoc/traj01_90deg_stable120m/figures/s8_relative_absolute_fusion/s8_f2_fusion_xy.png
outputs/villoc/traj01_90deg_stable120m/figures/s8_relative_absolute_fusion/s8_f2_fusion_rmse_summary.png
```

### Main result

```text
Best S8.F2:
interval_50m_strict_a_accept_online, alpha=0.25
8 corrections: 6 true, 2 false
RMSE: 13.84 m
p95: 25.24 m
Final error: 15.69 m
Drift: 0.92 m/100 m
```

### Important insight

Hard reset is often worse than soft reset because even correct tile-center anchors can still have 20–40 m error. Soft correction preserves the smooth relative path while gently reducing drift.

---

## 23. S8.F3 — Temporal-agreement gating

### Purpose

Build online policies that accept an absolute correction only if it is:

```text
visually confident
and
consistent with recent XFeat relative motion
```

This is closer to a real system than fixed interval-only correction.

### Concept

If an anchor was accepted at frame A and a new candidate appears at frame B:

```text
XFeat relative displacement = relative_position[B] - relative_position[A]
Absolute map displacement   = absolute_position[B] - absolute_position[A]

Temporal residual = difference between those two displacement arrows
```

If the residual is small enough, accept the candidate.  
If the residual is too large, reject it.

### Script

```text
scripts/villoc/s8_fusion/s8_f3_temporal_agreement_gating.py
```

### Run command

```bash
cd /Users/harishprabhu/Documents/drone-localisation
source .drone_venv/bin/activate
export PYTHONPATH=$PWD/src

ROOT=outputs/villoc/traj01_90deg_stable120m

python scripts/villoc/s8_fusion/s8_f3_temporal_agreement_gating.py \
  --root "$ROOT" \
  2>&1 | tee "$ROOT/logs/s8_relative_absolute_fusion/s8_f3_temporal_agreement_gating.log"
```

### Outputs

```text
outputs/villoc/traj01_90deg_stable120m/metadata/s8_relative_absolute_fusion/s8_f3_temporal_agreement_manifest.csv
outputs/villoc/traj01_90deg_stable120m/metadata/s8_relative_absolute_fusion/s8_f3_temporal_policy_summary.csv
outputs/villoc/traj01_90deg_stable120m/metadata/s8_relative_absolute_fusion/s8_f3_temporal_policy_gaps.csv
outputs/villoc/traj01_90deg_stable120m/reports/s8_relative_absolute_fusion/s8_f3_temporal_agreement_gating_report.json
outputs/villoc/traj01_90deg_stable120m/figures/s8_relative_absolute_fusion/s8_f3_temporal_policy_correction_distribution.png
outputs/villoc/traj01_90deg_stable120m/figures/s8_relative_absolute_fusion/s8_f3_temporal_residuals.png
```

### Decoding temporal policy names

Example:

```text
f3_temporal_a_bootb_res50_r030_gap30_accept_online
```

| Part | Meaning |
|---|---|
| `f3` | S8.F3 stage |
| `temporal` | use motion-consistency checking |
| `a` | possible candidates come from `strict_a_accept_online` |
| `bootb` | first anchor must come from stricter `strict_b_accept_online` |
| `res50` | allow at least 50 m residual tolerance |
| `r030` | also allow residual up to 30% of relative travelled distance |
| `gap30` | require at least 30 m since previous accepted correction |
| `accept_online` | online-usable accept/reject decision |

Report-friendly name:

```text
Temporal-consistency fusion
```

### S8.F3 policy summary

| Policy | Accepted | True | False | Dangerous | Precision |
|---|---:|---:|---:|---:|---:|
| temporal B, res30, gap30 | 5 | 5 | 0 | 0 | 1.000 |
| temporal A bootB, res50, gap50 | 8 | 8 | 0 | 0 | 1.000 |
| temporal A bootB, res50, gap100 | 5 | 5 | 0 | 0 | 1.000 |
| temporal A bootB, res50, gap30 | 11 | 10 | 1 | 0 | 0.909 |

### Interpretation

Temporal agreement reduced false accepted corrections while keeping enough useful anchors for fusion.

---

## 24. S8.F3B — Temporal-gated fusion replay

### Purpose

Replay the S8.F3 temporal policies through the same fusion system and compare with S8.F2.

### Script

```text
scripts/villoc/s8_fusion/s8_f3b_temporal_policy_fusion_replay.py
```

### Run command

```bash
cd /Users/harishprabhu/Documents/drone-localisation
source .drone_venv/bin/activate
export PYTHONPATH=$PWD/src

ROOT=outputs/villoc/traj01_90deg_stable120m

python scripts/villoc/s8_fusion/s8_f3b_temporal_policy_fusion_replay.py \
  --root "$ROOT" \
  --manifest "$ROOT/metadata/s8_relative_absolute_fusion/s8_f3_temporal_agreement_manifest.csv" \
  --alphas 1.0,0.75,0.5,0.25 \
  --eval-start-index 49 \
  --thresholds-m 10,20,40,80,120 \
  --sustain-frames 5 \
  2>&1 | tee "$ROOT/logs/s8_relative_absolute_fusion/s8_f3b_temporal_policy_fusion_replay.log"
```

### Outputs

```text
outputs/villoc/traj01_90deg_stable120m/metadata/s8_relative_absolute_fusion/s8_f3b_temporal_fusion_replay_trajectory.csv
outputs/villoc/traj01_90deg_stable120m/metadata/s8_relative_absolute_fusion/s8_f3b_temporal_fusion_replay_summary.csv
outputs/villoc/traj01_90deg_stable120m/metadata/s8_relative_absolute_fusion/s8_f3b_temporal_fusion_threshold_crossings.csv
outputs/villoc/traj01_90deg_stable120m/reports/s8_relative_absolute_fusion/s8_f3b_temporal_policy_fusion_replay_report.json
outputs/villoc/traj01_90deg_stable120m/figures/s8_relative_absolute_fusion/s8_f3b_temporal_fusion_error_vs_distance.png
outputs/villoc/traj01_90deg_stable120m/figures/s8_relative_absolute_fusion/s8_f3b_temporal_fusion_xy.png
outputs/villoc/traj01_90deg_stable120m/figures/s8_relative_absolute_fusion/s8_f3b_temporal_fusion_rmse_summary.png
```

### Final S8.F3B result

| Policy | alpha | Corrections | True | False | RMSE | p95 | Final | Drift |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Relative-only baseline | 0.00 | 0 | 0 | 0 | 35.76 | 46.24 | 39.57 | 2.32 |
| Periodic 50 m strict A | 0.25 | 8 | 6 | 2 | **13.84** | **25.24** | 15.69 | 0.92 |
| Temporal A bootB res50 gap30 | 0.25 | 11 | 10 | 1 | 14.02 | 30.16 | **9.09** | **0.53** |
| Temporal B conservative | 0.25 | 5 | 5 | 0 | 19.91 | 40.67 | 3.74 | 0.22 |
| Oracle hit40 | 0.25 | 173 | 173 | 0 | 14.28 | 25.25 | 13.97 | 0.82 |

### Final choice

Use two labels in the report:

```text
Best RMSE fusion:
Periodic 50 m strict correction, alpha=0.25

Recommended realistic online fusion:
Temporal-consistency fusion, alpha=0.25
```

---

# Part E — Report-friendly explanation

## 25. Simple explanation of the final system

### Relative-only localization

The drone estimates how it moves from one frame to the next.  
This gives a smooth path, but errors slowly accumulate.

### Absolute localization

The drone image is matched against an orthophoto map.  
This gives a global map position, but some map matches are wrong.

### Fusion

The system keeps the smooth relative trajectory and occasionally pulls it toward a trusted map match.

### Why soft correction

The map position is based on a satellite tile center. Even a correct tile can still be off by 20–40 m.  
So the correction should not fully replace the relative position. It should gently pull the trajectory toward the map anchor.

### Why temporal gating

A visually confident map match can still be wrong.  
Temporal gating asks whether the new map correction agrees with the drone’s recent relative motion. If it disagrees, the correction is rejected.

---

## 26. Human-readable method labels for the report

Do not show raw policy names in the main report.

| Internal name | Report label |
|---|---|
| `no_correction_baseline` | XFeat relative-only |
| `512_s256_orb_hybrid_top20_img518` | ORB-reranked absolute localization |
| `interval_50m_strict_a_accept_online` | Periodic 50 m confidence-gated fusion |
| `strict_a_accept_online` | Confidence-gated fusion |
| `strict_b_accept_online` | High-confidence sparse fusion |
| `f3_temporal_a_bootb_res50_r030_gap30_accept_online` | Temporal-consistency fusion |
| `oracle_hit40_accept_eval_only` | Oracle upper-bound fusion |
| `all_reranked_accept_ablation` | Blind absolute-correction ablation |

---

## 27. report wording

```text
The final fusion strategy uses XFeat relative odometry as the continuous motion
estimate and ORB-reranked image-to-map localization as a sparse correction
source. Corrections are not applied at every frame because absolute map matches
can be wrong. Instead, candidate corrections are accepted only when the local
geometric evidence is strong and, in the final temporal policy, when the implied
map displacement agrees with the recent XFeat relative displacement. The
accepted correction is applied softly using a 25% correction weight, which
reduces drift while avoiding abrupt jumps caused by noisy or false absolute
matches.
```

---
