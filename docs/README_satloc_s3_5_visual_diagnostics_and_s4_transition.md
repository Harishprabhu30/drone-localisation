# SatLoc S3.5 Visual-Domain Diagnostics and S4 Transition

Project: `drone-localisation`  
Branch context: `feature/satloc-loader-map-localization` and `feature/satloc-visual-domain-diagnostics`
Dataset: SatLoc `part_1`, currently focusing on `traj01`  
Status: SatLoc loader and visual diagnostics are working. Next step is S4A ORB top-k tile retrieval.

---

## 1. Why this block was added

After Zurich MAV closeout, the project shifted from oblique-camera relative localization to SatLoc map-based localization. SatLoc provides UAV images, satellite/map tiles, and coordinate labels encoded in UAV filenames. This makes it suitable for the internship task direction: UAV-to-satellite / UAV-to-map localization, candidate coordinate output, and evaluation against withheld/reference labels.

Before implementing full tile retrieval, we added a short visual-domain diagnostics block because UAV-to-satellite matching has a strong visual gap:

- UAV frames are sharper, closer, and sometimes view-dependent.
- Satellite tiles are blurrier, lower-detail, and map-like.
- Vegetation, shadows, roads, ponds, roof grids, field patterns, and buildings behave differently under feature extraction.
- A preprocessing method that looks visually cleaner may not necessarily improve ORB matching.

---

## 2. Current SatLoc dataset inspection result

SatLoc `part_1` structure discovered:

```text
data/raw/satloc/part_1/
├── UAV Data/
│   ├── traj01/
│   ├── traj03/
│   └── traj04/
└── Satellite Data/
    ├── CS_Area1.tif
    ├── ref_sample.csv
    └── sat_image_ref/
```

Inspection summary:

```text
UAV images:                    2959
UAV parsed coords:             2959
UAV sequences:                 3
Satellite tile rows:           8625
Satellite tile files matched:  8625
GeoTIFF:                       8192 × 4650, EPSG:4326
Reference CSV rows:            8625
```

Important interpretation:

- UAV filenames encode coordinate labels, e.g. `1@0@112.816130@28.297316.png`.
- The safe parse is `token0_id`, `token1_order`, `longitude`, `latitude`.
- `ref_sample.csv` maps satellite tile filenames to pixel boxes in the GeoTIFF/DOM map.
- Its columns named `Loc1 lon`, `Loc1 lat`, `Loc2 lon`, `Loc2 lat` are pixel coordinates, not direct geographic lon/lat.
- GeoTIFF transform converts tile pixel boxes into lon/lat bounding boxes and tile centers.

Estimator rule:

```text
UAV filename lon/lat is reference/evaluation data only.
It must not be used inside the localization estimator.
```

---

## 3. Key files added or used

### Config

```text
configs/dataset_satloc.yaml
```

### Dataset loaders / indexing

```text
src/uavloc/data/satloc_loader.py
src/uavloc/data/satloc_coordinate_index.py
```

### Visual-domain analysis module

```text
src/uavloc/analysis/__init__.py
src/uavloc/analysis/visual_domain.py
```

### SatLoc scripts

```text
scripts/satloc/inspect_satloc_dataset.py
scripts/satloc/build_satloc_coordinate_index.py
scripts/satloc/run_traj01_visual_domain_diagnostics.py
scripts/satloc/build_traj01_decomposition_panels.py
scripts/satloc/build_traj01_edge_shadow_variant_panels.py
scripts/satloc/build_traj01_focused_sobel_panels.py
scripts/satloc/build_traj01_preprocessing_variant_panels.py
scripts/satloc/build_traj01_orb_preprocessing_comparison.py
scripts/satloc/run_traj01_true_tile_orb_matching.py
```

---

## 4. Outputs generated

### Coordinate indexing

```text
outputs/satloc/metadata/uav_frames_index_enriched.csv
outputs/satloc/metadata/satellite_tiles_index_enriched.csv
outputs/satloc/trajectories/uav_reference_trajectory.csv
outputs/satloc/reports/satloc_coordinate_summary.json
outputs/satloc/figures/reference_trajectories/
outputs/satloc/maps/reference_trajectories/
```

### Visual-domain diagnostics

```text
outputs/satloc/metadata/s3_5_visual_domain_traj01/traj01_image_stats.csv
outputs/satloc/reports/s3_5_visual_domain_traj01/traj01_visual_domain_summary.json
outputs/satloc/figures/s3_5_visual_domain_traj01/
```

### Focused Sobel / preprocessing diagnostics

```text
outputs/satloc/figures/s3_5_visual_domain_traj01/focused_sobel_panels/
outputs/satloc/metadata/s3_5_visual_domain_traj01/focused_sobel_panel_manifest.csv
```

### ORB preprocessing comparison

```text
outputs/satloc/figures/s3_5_visual_domain_traj01/orb_preprocessing_comparison/
outputs/satloc/metadata/s3_5_visual_domain_traj01/orb_preprocessing_comparison_stats.csv
outputs/satloc/metadata/s3_5_visual_domain_traj01/orb_preprocessing_comparison_summary.csv
```

### True-tile ORB matching diagnostics

```text
outputs/satloc/figures/s3_5_visual_domain_traj01/true_tile_orb_matching/
outputs/satloc/metadata/s3_5_visual_domain_traj01/true_tile_orb_matching_results.csv
outputs/satloc/reports/s3_5_visual_domain_traj01/true_tile_orb_matching_summary.csv
outputs/satloc/metadata/s3_5_visual_domain_traj01/true_tile_orb_matching_manifest.csv
```

---

## 5. S3.5A numeric visual diagnostics

Script:

```bash
export PYTHONPATH=$PWD/src
python scripts/satloc/run_traj01_visual_domain_diagnostics.py --config configs/dataset_satloc.yaml
```

Result summary for `traj01`:

```text
rows: 1034
read_ok: 1034
luma_mean median:          124.395
luma_std median:           46.279
laplacian_variance median: 1433.427
edge_density median:       0.235
entropy_gray median:       7.477
ORB keypoints median:      2500.0
AKAZE keypoints median:    2405.5
Hough lines median:        1442.0
```

Interpretation:

- `traj01` has rich visual structure.
- ORB count saturated at `nfeatures=2500`, so raw keypoint count is not useful alone.
- AKAZE count, edge density, sharpness, and luma contrast are more informative for frame quality.
- Numeric diagnostics are useful for selecting frames, but visual decomposition is needed to understand what the algorithms actually see.

---

## 6. S3.5B decomposition panels

Script:

```bash
python scripts/satloc/build_traj01_decomposition_panels.py \
  --config configs/dataset_satloc.yaml \
  --ranges 1-150:12,250-350:10,400-500:10
```

Also supports exact frame selection:

```bash
python scripts/satloc/build_traj01_decomposition_panels.py \
  --config configs/dataset_satloc.yaml \
  --frames 1,50,120,260,320,420,470
```

Findings:

- Canny creates many edges and is visually confusing in vegetation/shadow regions.
- Canny is useful near ponds, roads, and clear boundaries.
- Hough lines follow many structural directions but also respond to repeated vegetation/crop/tree patterns.
- AKAZE spreads widely and is not easy to interpret visually.
- ORB focuses on many useful corners and motion-stable features, but it can also follow repeated textures.
- Sobel reveals roads, buildings, roof grids, pond boundaries, and shadow boundaries clearly.
- CLAHE makes local contrast clearer but may amplify shadow/vegetation clutter.

---

## 7. S3.5D focused Sobel and preprocessing study

The focused panel was narrowed to compare:

```text
RGB original
Sobel on luma
Sobel on CLAHE-luma
Bilateral(CLAHE-luma)
Bilateral(alt-CLAHE-luma)
Abs difference maps
```

Key parameter meaning:

```text
CLAHE clipLimit:
  lower = milder contrast boost
  higher = stronger/aggressive contrast boost

CLAHE tileGridSize:
  smaller tile = more local contrast, more detail/noise
  larger tile = broader smoother contrast normalization

Bilateral d:
  larger = stronger spatial smoothing neighborhood
  smaller = weaker/more local smoothing

Bilateral sigmaColor:
  lower = stronger edge preservation
  higher = more blending across intensity differences

Bilateral sigmaSpace:
  higher = smoothing over larger spatial range
```

Candidate settings tested:

```bash
--clahe-clip-limit 2.0 \
--alt-clahe-clip-limit 1.0 \
--alt-clahe-tile-size 8 \
--bilateral-d 13 \
--bilateral-sigma-color 30 \
--bilateral-sigma-space 55
```

Visual finding:

- Sobel on luma / CLAHE-luma is rich but cluttered.
- Bilateral after mild CLAHE smooths vegetation and preserves many road/building/mud-road boundaries.
- Stronger CLAHE can reveal more structure but also amplifies shadow/tree texture.
- Mild CLAHE + bilateral is visually safer, but visual quality alone does not prove matching improvement.

---

## 8. S3.5D.2 ORB preprocessing comparison

Script:

```bash
python scripts/satloc/build_traj01_orb_preprocessing_comparison.py \
  --config configs/dataset_satloc.yaml \
  --ranges 1-100:20 \
  --nfeatures 1200 \
  --clahe-clip-limit 2.0 \
  --clahe-tile-size 8 \
  --alt-clahe-clip-limit 1.0 \
  --alt-clahe-tile-size 8 \
  --bilateral-d 13 \
  --bilateral-sigma-color 30 \
  --bilateral-sigma-space 55
```

Variants:

```text
V0_gray
V1_luma
V2_clahe_luma
V3_bilateral_clahe
V4_bilateral_alt_clahe
```

Summary at `nfeatures=1200`:

```text
variant                    grid_occ   concentration   green_ratio   mean_response   overlap_luma
V0_gray                    0.771875   0.120375        0.002000      0.000721        0.995250
V1_luma                    0.770833   0.119792        0.002042      0.000721        1.000000
V2_clahe_luma              0.811458   0.113292        0.002583      0.002532        0.798042
V3_bilateral_clahe         0.786458   0.113708        0.002250      0.001947        0.816875
V4_bilateral_alt_clahe     0.773958   0.109500        0.002167      0.001045        0.924500
```

Interpretation:

- Grayscale and luma are almost identical for ORB.
- ORB count is saturated, so count alone is not useful.
- V2 produces stronger and more distributed features, but also more vegetation/texture activation.
- V3 changes ORB behavior meaningfully and enhances structure.
- V4 is conservative: it smooths while staying close to luma.
- For visual analysis use `nfeatures=800` or `1200`; `2500` is too cluttered.

---

## 9. S3.5D.3 true-tile ORB matching

Script:

```bash
python scripts/satloc/run_traj01_true_tile_orb_matching.py \
  --config configs/dataset_satloc.yaml \
  --ranges 1-100:10 \
  --nfeatures 1200 \
  --ratio 0.75 \
  --ransac-thresh 5.0 \
  --clahe-clip-limit 2.0 \
  --clahe-tile-size 8 \
  --alt-clahe-clip-limit 1.0 \
  --alt-clahe-tile-size 8 \
  --bilateral-d 13 \
  --bilateral-sigma-color 30 \
  --bilateral-sigma-space 55
```

Summary:

```text
variant                    good_mean   inliers_mean   inlier_ratio   homography_success   score_mean
V1_luma                    7.0         1.9            0.188889       0.4                  2.60
V2_clahe_luma              7.3         2.5            0.229182       0.5                  3.23
V3_bilateral_clahe         6.8         1.0            0.091667       0.2                  1.68
V4_bilateral_alt_clahe     7.8         2.3            0.231288       0.5                  3.08
```

Interpretation:

- V2 and V4 slightly improve over raw luma.
- V3 looks visually useful in some structured frames, but is weaker on average.
- Direct ORB matching to the true satellite tile is weak overall: match counts and RANSAC inliers are low.
- Some structured frames work better; forest-heavy / vegetation-heavy frames fail.
- Preprocessing alone is not enough to solve UAV-to-satellite localization.

### Important Figures:
![Satloc Traj01 reference trajectory [ENU]](docs/assets/week4_satloc_visual_diagnostics/satloc_all_global_enu.png)

![Satloc Traj01 reference trajectory [Lon/Lat]](docs/assets/week4_satloc_visual_diagnostics/satloc_all_lonlat.png)

![Feature Preprocessing Trials](docs/assets/week4_satloc_visual_diagnostics/traj01_frame_0100_focused_sobel.png)

![ORB preprocessing Comparision on different Feature Extraction methods](docs/assets/week4_satloc_visual_diagnostics/traj01_frame_0100_orb_preprocessing_comparison.png)

![True tile matching with satelite ref images](docs/assets/week4_satloc_visual_diagnostics/traj01_frame_0100_true_tile_orb_matches.png)


## Conclusion:

```text
A visual-domain preprocessing study was conducted on SatLoc traj01. Luma, CLAHE-luma, bilateral+CLAHE, and mild-CLAHE+bilateral variants were compared using ORB keypoint distribution and true-tile UAV-to-satellite matching. Although CLAHE and mild bilateral preprocessing slightly improved average true-tile matching over raw luma, all classical ORB variants produced low match counts and low RANSAC inlier counts. This suggests that direct ORB homography to a single ground-truth tile is unstable under UAV-to-satellite appearance, scale, blur, texture, and viewpoint mismatch. The next step is therefore to test top-k satellite tile retrieval and candidate localization, not to continue tuning visual preprocessing.
```

---

## 10. Current technical decision

Choosing variants for further:

```text
V1_luma                    baseline
V2_clahe_luma              best average true-tile score
V4_bilateral_alt_clahe     conservative smoothing candidate
```

Keep V3 as optional for structured urban/road-only tests:

```text
V3_bilateral_clahe         useful in some structured frames, not default
```

Default parameters:

```text
nfeatures = 1200
ratio = 0.75
ransac_thresh = 5.0
```

---

## 11. Next: S4A ORB top-k tile retrieval baseline

Goal:

```text
Input: one UAV frame from traj01
Database: satellite tiles from sat_image_ref
Output: top-k predicted satellite tiles
Evaluation: does GT lon/lat fall inside/near top-k predicted tile bboxes?
```

Metrics:

```text
Recall@1
Recall@5
Recall@10
rank of first tile containing GT
center error in meters
good matches
RANSAC inliers
inlier ratio
homography success
query time
```

Planned initial implementation is small, then expanded:

```text
10–20 UAV query frames from traj01
all 8625 satellite tiles if feasible, otherwise local/test subset first
variants: V1, V2, V4
```

Expected output files:

```text
outputs/satloc/metadata/s4a_orb_tile_retrieval/orb_retrieval_results.csv
outputs/satloc/reports/s4a_orb_tile_retrieval/orb_retrieval_metrics.json
outputs/satloc/figures/s4a_orb_tile_retrieval/query_topk_panels/
```

---
