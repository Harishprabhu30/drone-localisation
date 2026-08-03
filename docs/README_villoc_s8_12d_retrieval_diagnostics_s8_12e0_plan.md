# Villoc S8.12D Retrieval Diagnostics and S8.12E.0 Footprint-Aware Transition

**Project:** GNSS-denied / weak-GNSS UAV visual localization  
**Dataset:** Villoc Vilnius, 90° nadir and 45° oblique RGB streams  
**Stage covered in this chat:** S8.12D reusable DINO retrieval diagnostics, 45° vs 90° comparison, and S8.12E.0 footprint-aware next experiment  
**Primary variant discussed:** `1024_s512` with `dinov2_vits14_img224_center_square_avgpatch_cpu`  
**Status:** `PASS_RETRIEVAL_DIAGNOSTICS` for 45° and 90°; `PASS_RETRIEVAL_DIAGNOSTICS_COMPARISON`

---

## 1. Executive conclusion

This chat converted the Villoc 45°/90° DINO retrieval analysis from ad-hoc scripts into a reusable diagnostic workflow.

The final reusable workflow is:

```text
DINO retrieval outputs
        ↓
s8_retrieval_diagnostics.py
        ↓
per-run diagnostic CSV buckets + visual panels
        ↓
s8_retrieval_compare_reports.py
        ↓
comparison table/plots across 2 or more runs
        ↓
S8.12E.0 footprint-aware diagnostic for 45° oblique imagery
```

Main result:

```text
45° and 90° have similar Top-1 success counts.
45° has better Top-20 coverage and many more cases where the oracle is already inside Top-5.
90° has two Top-20 candidate-pool failures.
45° errors may partly be footprint-offset errors because the camera is oblique.
```

The most important interpretation is:

```text
45° retrieval should not be evaluated exactly like 90° nadir retrieval.
For 45° oblique imagery, DINO may retrieve the forward visible ground footprint rather than the UAV body ground point.
```

Therefore, the next quick experiment should be:

```text
S8.12E.0 — Footprint-center-ray diagnostic
```

This experiment checks whether the apparent 45° retrieval error decreases when the oracle is shifted from the UAV body position to the estimated forward camera footprint center.

---

## 2. What changed in code design

### 2.1 New reusable diagnostic layer

The new reusable script is:

```text
scripts/villoc/s8_retrieval_diagnostics/s8_retrieval_diagnostics.py
```

It takes one retrieval run and creates:

```text
csv/good_top1.csv
csv/bad_top1.csv
csv/oracle_in_top5_but_not_top1.csv
csv/oracle_in_top20_but_not_top1.csv
csv/oracle_missing_top20.csv
csv/high_conf_wrong_top1.csv
csv/easy_correct.csv
csv/hard_correct.csv
csv/all_queries_with_diagnostic_flags.csv
figures/<bucket_name>/*.png
report.json
```

Each panel shows:

```text
UAV query | Top-1 tile | Top-2 tile | Top-3 tile | First oracle tile
```

Coordinates/oracles are used only after descriptor ranking for diagnostics/evaluation. They are not used to rank candidates.

### 2.2 New general comparison layer

The new comparison script is:

```text
scripts/villoc/s8_retrieval_diagnostics/s8_retrieval_compare_reports.py
```

It compares any number of run reports:

```text
45° vs 90°
new angle vs old angle
new AOI vs old AOI
same dataset with different tile variants
same dataset with different descriptor tags
```

It reads `report.json` from each diagnostic run and writes:

```text
run_summary.csv
run_summary.md
comparison_report.json
figures/bucket_counts.png
figures/top1_vs_recall_at_20.png
pairwise_<runA>_vs_<runB>_by_query_id.csv
bucket_transition_<runA>_vs_<runB>_by_query_id.csv
```

Important note:

```text
Pairwise query_id comparison is meaningful only when query IDs represent aligned samples.
For separate recordings, aggregate comparison is safer unless query IDs are explicitly aligned.
```

---

## 3. Completed command chain in this chat

### 3.1 45° retrieval diagnostics

```bash
mkdir -p outputs/villoc/45_deg/logs/s8_12d_45deg_diagnostics

python scripts/villoc/s8_retrieval_diagnostics/s8_retrieval_diagnostics.py \
  --config configs/dataset_villoc_45deg.yaml \
  --variant 1024_s512 \
  --tag dinov2_vits14_img224_center_square_avgpatch_cpu \
  --oracle-k 20 \
  --max-panels 40 \
  --top-n-tiles 3 \
  2>&1 | tee outputs/villoc/45_deg/logs/s8_12d_45deg_diagnostics/s8_retrieval_diagnostics_1024_s512.log
```

Result:

```text
STATUS: PASS_RETRIEVAL_DIAGNOSTICS
Report: outputs/villoc/45_deg/diagnostics/s8_retrieval_diagnostics/1024_s512/report.json
```

### 3.2 90° retrieval diagnostics

Run:

```bash
mkdir -p outputs/villoc/90_deg/logs/s8_retrieval_diagnostics

python scripts/villoc/s8_retrieval_diagnostics/s8_retrieval_diagnostics.py \
  --config configs/dataset_villoc_90deg.yaml \
  --variant 1024_s512 \
  --tag dinov2_vits14_img224_center_square_avgpatch_cpu \
  --oracle-k 20 \
  --max-panels 40 \
  --top-n-tiles 3 \
  2>&1 | tee outputs/villoc/90_deg/logs/s8_retrieval_diagnostics/s8_retrieval_diagnostics_1024_s512.log
```

Result:

```text
STATUS: PASS_RETRIEVAL_DIAGNOSTICS
Report: outputs/villoc/90_deg/diagnostics/s8_retrieval_diagnostics/1024_s512/report.json
```

### 3.3 45° vs 90° comparison

```bash
mkdir -p outputs/villoc/reports/s8_retrieval_comparisons/villoc_45_vs_90_1024_s512

python scripts/villoc/s8_retrieval_diagnostics/s8_retrieval_compare_reports.py \
  --run 45deg:outputs/villoc/45_deg/diagnostics/s8_retrieval_diagnostics/1024_s512/report.json \
  --run 90deg:outputs/villoc/90_deg/diagnostics/s8_retrieval_diagnostics/1024_s512/report.json \
  --oracle-k 20 \
  --out-root outputs/villoc/reports/s8_retrieval_comparisons/villoc_45_vs_90_1024_s512 \
  2>&1 | tee outputs/villoc/reports/s8_retrieval_comparisons/villoc_45_vs_90_1024_s512/compare.log
```

Result:

```text
STATUS: PASS_RETRIEVAL_DIAGNOSTICS_COMPARISON
Summary CSV: outputs/villoc/reports/s8_retrieval_comparisons/villoc_45_vs_90_1024_s512/run_summary.csv
Summary MD:  outputs/villoc/reports/s8_retrieval_comparisons/villoc_45_vs_90_1024_s512/run_summary.md
Report:      outputs/villoc/reports/s8_retrieval_comparisons/villoc_45_vs_90_1024_s512/comparison_report.json
Figures:     outputs/villoc/reports/s8_retrieval_comparisons/villoc_45_vs_90_1024_s512/figures
```

---

## 4. Main comparison table

| label | variant | query_count | top1_hits | top1_hit_rate | recall_at_20_hits | recall_at_20 | good_top1 | bad_top1 | oracle_in_top5_but_not_top1 | oracle_in_top20_but_not_top1 | oracle_missing_top20 | high_conf_wrong_top1 | easy_correct | hard_correct | first_oracle_rank_median | top1_center_error_median_m |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 45deg | 1024_s512 | 112 | 23 | 0.2054 | 112 | 1.0000 | 23 | 89 | 61 | 89 | 0 | 12 | 4 | 19 | 3.5 | 192.729 |
| 90deg | 1024_s512 | 115 | 22 | 0.1913 | 113 | 0.9826 | 22 | 93 | 41 | 91 | 2 | 12 | 8 | 14 | 3.0 | 212.871 |

### Interpretation

```text
45° and 90° have similar Top-1 success.
45° is stronger as a candidate pool: all 112 queries have an oracle inside Top-20.
45° has many more cases where the oracle is already inside Top-5 but not Top-1.
90° has two Top-20 candidate-pool failures.
Both views have 12 high-confidence wrong Top-1 candidates.
```

Practical meaning:

```text
Most current failures are not global-search failures.
They are ranking/refinement failures.
```

For 45° specifically:

```text
Some apparent errors may be because the retrieved tile corresponds to the forward visible footprint, not the UAV body position.
```

---

README image references:

![45° vs 90° bucket counts](assets/villoc_s8_12d_retrieval_diagnostics/comparison/figures/bucket_counts.png)

![Top-1 vs Recall@20](assets/villoc_s8_12d_retrieval_diagnostics/comparison/figures/top1_vs_recall_at_20.png)

Interpretation for `bucket_counts.png`:

```text
The plot shows that both views have many bad Top-1 results, but most bad cases still contain an oracle inside Top-20. 45° has zero Top-20 candidate-pool failures; 90° has two.
```

Interpretation for `top1_vs_recall_at_20.png`:

```text
Top-1 accuracy is low for both views, but Recall@20 is almost saturated. This means DINO candidate generation is strong enough for a verifier/reranker stage.
```

Useful folders to inspect manually:

```text
docs/assets/villoc_s8_12d_retrieval_diagnostics/45deg_1024_s512/figures/high_conf_wrong_top1/
docs/assets/villoc_s8_12d_retrieval_diagnostics/45deg_1024_s512/figures/oracle_in_top5_but_not_top1/
docs/assets/villoc_s8_12d_retrieval_diagnostics/45deg_1024_s512/figures/bad_top1/
```

Interpretation:

```text
The 45° diagnostic panels show that many consecutive frames view similar forward ground areas. DINO often retrieves a stable candidate for several frames. This suggests temporal redundancy and footprint offset must both be handled before judging 45° retrieval as incorrect.
```

Useful folders to inspect manually:

```text
docs/assets/villoc_s8_12d_retrieval_diagnostics/90deg_1024_s512/figures/oracle_missing_top20/
docs/assets/villoc_s8_12d_retrieval_diagnostics/90deg_1024_s512/figures/high_conf_wrong_top1/
docs/assets/villoc_s8_12d_retrieval_diagnostics/90deg_1024_s512/figures/bad_top1/
```

Interpretation:

```text
The two 90° Top-20 candidate-pool failures appear related to low-altitude limited context and repeated highway/roadside vegetation patterns. Nadir imagery is geometrically cleaner, but when altitude is low the visible context can be too small for robust global retrieval.
```

---

## 6. Correct understanding: map/AOI cases

### Case A — new dataset lies inside existing AOI crop

This is the 45° case.

Use existing:

```text
orthophoto crop
satellite tile indexes
map descriptor caches
```

Run order:

```text
1. configs/dataset_new.yaml
2. s8_1_parse_villoc_srt.py
3. s8_2_extract_sampled_frames.py
4. s8_3_build_reference_trajectory.py
5. s8_4_visual_frame_audit.py
6. s8_5_build_uav_frame_index.py
7. s8_6_build_map_bbox_plan.py
8. s8_10b_uav_tile_oracle_audit.py using existing AOI/tile indexes
9. s8_11bc_build_dinov2_caches.py with --reuse-map-caches
10. s8_11d_independent_dinov2_retrieval.py with --map-cache-root pointing to reused map descriptors
11. s8_retrieval_diagnostics.py
12. s8_retrieval_compare_reports.py if comparing with another run
```

No need to rerun:

```text
s8_7, s8_8, s8_9, s8_10a, map descriptor cache rebuild unless the existing map/tile database is invalid.
```

### Case B — new trajectory is outside current AOI crop but inside the larger master GeoTIFF

Corrected understanding:

```text
Do not replace the old AOI.
Crop a new AOI from the same master GeoTIFF.
Generate new tiles and new map descriptors for that AOI.
```

Run order:

```text
1. configs/dataset_new.yaml
2. S8.1–S8.6 to get trajectory bbox
3. confirm bbox is inside the available master GeoTIFF
4. s8_8_crop_master_aoi.py for the new AOI
5. s8_9_generate_reference_tiles.py
6. s8_10a_audit_tile_index_integrity.py
7. s8_10b_uav_tile_oracle_audit.py
8. s8_11bc_build_dinov2_caches.py to build new map + query caches
9. s8_11d_independent_dinov2_retrieval.py
10. s8_retrieval_diagnostics.py
```

### Case C — new trajectory is outside the whole available GeoTIFF

Corrected understanding:

```text
Download/export a new GeoTIFF.
Do not overwrite the old validated GeoTIFF or AOI.
```

Good structure:

```text
data/processed/villoc/<site_or_dataset>/maps/<map_source>/<aoi_name>.tif
outputs/villoc/<site_or_dataset>/metadata/s8_9_satellite_tile_index_<variant>.csv
outputs/villoc/<site_or_dataset>/descriptors/s8_11b_dinov2_map_<variant>_<tag>.npz
```

Run order:

```text
1. S8.1–S8.6 to define bbox
2. export/download new GeoTIFF for that bbox/map source
3. s8_7_validate_map_geotiff.py
4. optional s8_7b_compare_map_exports.py
5. s8_8_crop_master_aoi.py
6. s8_9_generate_reference_tiles.py
7. s8_10a_audit_tile_index_integrity.py
8. s8_10b_uav_tile_oracle_audit.py
9. s8_11bc_build_dinov2_caches.py
10. s8_11d_independent_dinov2_retrieval.py
11. s8_retrieval_diagnostics.py
```

---

## 7. User visual observations recorded

### 7.1 45° oblique observations

User observation:

```text
The 45° camera often makes DINO fetch candidates in the forward visible direction, sometimes around 60–100 m ahead. Multiple consecutive frames can show nearly the same front-view area, so the retrieved candidate remains stable over many frames.
```

Technical interpretation:

```text
This is physically plausible. For a 45° depression angle, the camera center ray intersects the ground approximately one altitude ahead of the UAV body. At 60–70 m relative altitude, a 60–70 m forward offset is expected even before considering the full image FOV.
```

Implication:

```text
45° body-position oracle may underestimate retrieval quality.
A footprint-aware oracle is needed before deciding 45° retrieval is bad.
```

### 7.2 90° nadir observations

User observation:

```text
90° query images have less context at lower altitude. At higher altitude, some tiles match but not always accurately. The two candidate-pool failures involve repeated highway/roadside forest/bush patterns.
```

Technical interpretation:

```text
Nadir view is geometrically cleaner because the camera center is close to the UAV body ground projection. However, low altitude reduces spatial context, and repeated road/vegetation patterns can still produce candidate-pool failures.
```

---

## 8. S8.12E.0 — Footprint-center-ray diagnostic

### 8.1 Why this experiment is needed

The current 45° evaluation asks:

```text
Is the retrieved tile near the UAV body position?
```

But a 45° oblique image may visually describe:

```text
the forward ground footprint
```

So S8.12E.0 asks:

```text
Is the retrieved tile closer to the estimated camera footprint center than to the UAV body point?
```

And:

```text
If we shift the retrieved visible-ground position backward by the camera look offset, does body-position error improve?
```

### 8.2 Geometry

Side-view model:

```text
          camera / UAV body
              *
              |\
              | \
 height h     |  \  slant distance
              |   \
              |    \
              |_____\
         body ground  footprint center

          horizontal offset d
```

Formulas:

```text
forward_offset_m = rel_alt_m / tan(abs(gimbal_pitch_deg))
slant_m          = sqrt(rel_alt_m^2 + forward_offset_m^2)
```

For pitch near `-45°`:

```text
forward_offset_m ≈ rel_alt_m
```

Yaw projection, assuming yaw is clockwise from north:

```text
offset_east_m  = forward_offset_m * sin(yaw_rad)
offset_north_m = forward_offset_m * cos(yaw_rad)

footprint_x_m = body_x_m + offset_east_m
footprint_y_m = body_y_m + offset_north_m
```

Body correction from retrieved tile center:

```text
corrected_body_x_m = retrieved_tile_center_x_m - offset_east_m
corrected_body_y_m = retrieved_tile_center_y_m - offset_north_m
```

### 8.3 Add the quick experiment script

Place the provided script at:

```text
scripts/villoc/s8_12e0_footprint_center_ray_diagnostic.py
```

Make executable:

```bash
chmod +x scripts/villoc/s8_12e0_footprint_center_ray_diagnostic.py
python -m py_compile scripts/villoc/s8_12e0_footprint_center_ray_diagnostic.py
```

### 8.4 Run default 45° footprint diagnostic

```bash
python scripts/villoc/s8_12e0_footprint_center_ray_diagnostic.py \
  --config configs/dataset_villoc_45deg.yaml \
  --variant 1024_s512 \
  --tag dinov2_vits14_img224_center_square_avgpatch_cpu \
  --oracle-k 20 \
  2>&1 | tee outputs/villoc/45_deg/logs/s8_12d_45deg_diagnostics/s8_12e0_footprint_center_ray_1024_s512.log
```

Expected outputs:

```text
outputs/villoc/45_deg/reports/s8_12e0_footprint_center_ray/1024_s512/
  s8_12e0_all_ranked_body_vs_footprint.csv
  s8_12e0_top1_body_vs_footprint.csv
  s8_12e0_footprint_center_ray_summary.json
  figures/
    s8_12e0_top1_error_boxplot.png
    s8_12e0_offset_vs_footprint_improvement.png
    s8_12e0_corrected_body_scatter.png
```

### 8.5 Optional yaw-convention audit

If the default yaw convention looks wrong, run a small diagnostic sweep:

```bash
for OFF in 0 90 -90 180; do
  python scripts/villoc/s8_12e0_footprint_center_ray_diagnostic.py \
    --config configs/dataset_villoc_45deg.yaml \
    --variant 1024_s512 \
    --tag dinov2_vits14_img224_center_square_avgpatch_cpu \
    --oracle-k 20 \
    --yaw-offset-deg "$OFF" \
    --out-root outputs/villoc/45_deg/reports/s8_12e0_footprint_center_ray/1024_s512_yawoff_${OFF} \
    2>&1 | tee outputs/villoc/45_deg/logs/s8_12d_45deg_diagnostics/s8_12e0_yawoff_${OFF}.log
done
```

## 9. Important corrections to the current mental model

### 9.1 Pythagoras is only part of it

Correct idea:

```text
The geometry is a right triangle.
```

More precise correction:

```text
The forward ground offset uses trigonometry: d = h / tan(theta).
Pythagoras gives the slant distance after d is known.
```

### 9.2 Do not select “1–2 candidates before the fetched tile”

This is not reliable because candidate rank order is not spatial order.

Better:

```text
Use pitch + yaw + relative altitude to estimate the footprint offset.
Then evaluate or correct in map coordinates.
```

### 9.3 Do not overwrite old GeoTIFFs

If a new dataset is outside the old AOI, create a new AOI/map database. Do not replace validated files silently.

### 9.4 45° is not simply worse than 90°

45° has:

```text
larger forward context
strong candidate-pool coverage
more temporal redundancy
body-vs-footprint evaluation mismatch
```

90° has:

```text
cleaner body-position geometry
less context at low altitude
repeated-road/vegetation candidate-pool failures
```

---

### 10.1 Immediate next experiment

Run:

```text
S8.12E.0 footprint-center-ray diagnostic
```

Decision after S8.12E.0:

```text
If footprint-aware error improves:
  keep 45° as a useful forward-footprint localization signal.

If corrected-body error improves:
  test 45° body correction using geometric back-shift.

If neither improves:
  prioritize 90° nadir + LightGlue / temporal voting first.
```

### 10.2 Better query sampling for 45°

Fixed 1 fps is visually redundant for oblique view.

Future sampler options:

```text
A. distance-based sampling:
   keep one query every 20–30 m of trajectory movement

B. descriptor-novelty sampling:
   keep frame only if DINO descriptor distance from previous selected frame is large enough

C. yaw/heading-change sampling:
   keep frame if yaw changes meaningfully

D. hybrid:
   keep frame if displacement >= 20 m OR descriptor novelty >= threshold OR yaw change >= threshold
```

This can reduce repeated front-view frames and make diagnostics less redundant.

### 10.3 Multi-view fusion idea

Potential future policy:

```text
90° nadir:
  body-position localization signal

45° oblique:
  forward-footprint localization signal

Fusion:
  use 45° as preview / look-ahead map evidence
  use 90° as current-body correction evidence
```

This is interesting because 45° may see a place before the nadir camera reaches it.

### 10.4 Innovation idea: footprint-aware LightGlue

Instead of LightGlue only against body-centered oracle tiles, build candidate pools around:

```text
body-centered tiles
footprint-centered tiles
intermediate tiles between body and footprint
```

Then test whether LightGlue selects a geometrically consistent visible-ground match.

---

## 11. Final current stage status

```text
S8.12D.8  Generalized DINO scripts                  PASS
S8.12D.9  45° independent DINO retrieval             PASS
S8.12D.10 General retrieval diagnostics              PASS
S8.12D.11 45° vs 90° general comparison              PASS
```

