# SatLoc S4B — Structural Retrieval Diagnostics and Texture-Suppressed HOG Closeout

## 1. Purpose

This README documents the SatLoc **S4B structural retrieval chain** after the ORB full-map baseline failed.

The main question tested here was:

> Can a global structural descriptor, based on Sobel/HOG edge layout, retrieve the correct UAV-to-satellite map neighborhood better than ORB?

Ground-truth longitude/latitude from UAV filenames was used **only for evaluation, visualization, and diagnostics**. It was not used inside retrieval ranking.

---

## 2. High-level result

The S4B HOG+edge structural descriptor was clearly better than ORB for visual/structural retrieval. It sometimes retrieved near-correct satellite neighborhoods within tens of meters.

However, the diagnostics showed a strong limitation:

```text
HOG+edge compares gradient layout and texture statistics.
It does not understand whether an edge belongs to road, pond, roof, forest, or construction land.
```

So dense vegetation and structured false positives still ranked highly. S4B.1g cell-level texture suppression helped, but did not fully solve this limitation.

Final conclusion:

```text
S4B.1 should be closed as a strong classical structural baseline and diagnostic chain.
The next candidate method should move beyond pure HOG/Sobel.
```

## 4. Execution chain

All commands below assume execution from repository root.

```bash
export PYTHONPATH=$PWD/src
```

### 4.1 S4B.1 — Structural full-map retrieval

Run this first for each token that will be diagnosed later. The diagnostic scripts depend on its ranked CSV output.

Token 1:

```bash
python scripts/satloc/s4b_1_structural_global_retrieval.py \
  --sequence traj01 \
  --token 1 \ # 100, 166 or other token id for different query img selection
  --top-k 10 \
  --preprocess luma \
  --descriptor-type hog_edge \
  --resize-mode crop \
  --resize-size 512 \
  --cells 8 \
  --bins 9 \
  --edge-pool-size 32 \
  --rebuild-cache
```

Core settings used:

```text
preprocess:       luma
descriptor:       hog_edge
resize mode:      crop
resize size:      512
HOG cells:        8 x 8
orientation bins: 9
edge pool size:   32
```

Interpretation:

- `crop` gave better UAV-to-satellite FOV alignment than stretch/pad.
- `512` preserved more road/building/pond structure than 256.
- `hog_edge` worked better than ORB because it compares coarse global structure rather than local binary patch matches.

---

### 4.2 S4B.1b — GT-neighborhood failure diagnostics

Token 166:

```bash
python scripts/satloc/s4b_1b_gt_neighborhood_failure_panel.py \
  --sequence traj01 \
  --token 166 \
  --top-k 10 \
  --structure-top-k 5 \
  --preprocess luma \
  --descriptor-type hog_edge \
  --resize-mode crop \
  --resize-size 512 \
  --cells 8 \
  --bins 9 \
  --edge-pool-size 32
```

Token 1:

```bash
python scripts/satloc/s4b_1b_gt_neighborhood_failure_panel.py \
  --sequence traj01 \
  --token 1 \
  --top-k 10 \
  --structure-top-k 5 \
  --preprocess luma \
  --descriptor-type hog_edge \
  --resize-mode crop \
  --resize-size 512 \
  --cells 8 \
  --bins 9 \
  --edge-pool-size 32
```

#### Figure: token 166 GT 3x3 neighborhood

![S4B.1b token 166 GT 3x3](assets/satloc/s4b_structural_retrieval/s4b1b_token0166_luma_hog_edge_modecrop_r512_c8_b9_e32_gt3x3_rgb.png)

Interpretation:

- The UAV query and GT-neighborhood tiles share visible road/building/vegetation layout.
- Top-1 tile 1149 had approximately 25.7 m error, meaning this token was a good structural retrieval example.
- Some GT-neighborhood labels show `GT False` because the script used nearest-center matching when exact bbox containment was not found. This does not mean the neighborhood is visually wrong.

#### Figure: token 166 top-k retrieved tiles

![S4B.1b token 166 top-k](assets/satloc/s4b_structural_retrieval/s4b1b_token0166_luma_hog_edge_modecrop_r512_c8_b9_e32_top10_rgb.png)

Interpretation:

- Top-1 is near the correct neighborhood.
- But several top-k tiles are mostly forest/vegetation.
- This revealed the first major weakness: vegetation texture can score highly even when the semantic scene does not match.

#### Figure: token 166 structural decomposition

![S4B.1b token 166 structure](assets/satloc/s4b_structural_retrieval/s4b1b_token0166_luma_hog_edge_modecrop_r512_c8_b9_e32_structure_gt3x3_top5.png)

Interpretation:

- The descriptor sees luma, Sobel magnitude, gradient orientation, and HOG cell layout.
- It does not see object classes.
- Forest texture creates dense gradients that may resemble useful structure numerically.

#### Figure: token 1 GT 3x3 neighborhood

![S4B.1b token 1 GT 3x3](assets/satloc/s4b_structural_retrieval/s4b1b_token0001_luma_hog_edge_modecrop_r512_c8_b9_e32_gt3x3_rgb.png)

Interpretation:

- The GT-neighborhood has pond/road/building layout similar to the UAV query.
- However, many GT-neighbor ranks were low.
- This motivated testing whether the issue was spatial phase shift or descriptor ambiguity.

---

### 4.3 S4B.1c — Shift-tolerance diagnostic

Token 166:

```bash
python scripts/satloc/s4b_1c_shift_tolerance_diagnostic.py \
  --sequence traj01 \
  --token 166 \
  --top-k 10 \
  --preprocess luma \
  --descriptor-type hog_edge \
  --resize-mode crop \
  --resize-size 512 \
  --cells 8 \
  --bins 9 \
  --edge-pool-size 32 \
  --max-shift 2
```

Token 1:

```bash
python scripts/satloc/s4b_1c_shift_tolerance_diagnostic.py \
  --sequence traj01 \
  --token 1 \
  --top-k 10 \
  --preprocess luma \
  --descriptor-type hog_edge \
  --resize-mode crop \
  --resize-size 512 \
  --cells 8 \
  --bins 9 \
  --edge-pool-size 32 \
  --max-shift 2
```

#### Figure: token 166 shift heatmap

![S4B.1c token 166 shift heatmap](assets/satloc/s4b_structural_retrieval/s4b1c_token0166_luma_modecrop_r512_c8_b9_shift2_shift_heatmaps.png)

Interpretation:

- Each heatmap shows HOG similarity after shifting the satellite HOG grid by cell offsets.
- One cell is 64 px because `512 / 8 = 64`.
- The best tile was still tile 1149 without needing meaningful shift improvement.
- Conclusion: for token 166, spatial phase shift was not the main problem.

#### Figure: token 1 shift heatmap

![S4B.1c token 1 shift heatmap](assets/satloc/s4b_structural_retrieval/s4b1c_token0001_luma_modecrop_r512_c8_b9_shift2_shift_heatmaps.png)

Interpretation:

- The best GT-neighborhood tile was still not rescued strongly by shifting.
- The false best tile only slightly outscored the correct neighborhood.
- This suggested the bigger problem was not cell shift, but descriptor ambiguity.

---

### 4.4 S4B.1d — Score decomposition panel

Token 1:

```bash
python scripts/satloc/s4b_1d_score_decomposition_panel.py \
  --sequence traj01 \
  --token 1 \
  --top-k 10 \
  --panel-top-k 5 \
  --preprocess luma \
  --descriptor-type hog_edge \
  --resize-mode crop \
  --resize-size 512 \
  --cells 8 \
  --bins 9 \
  --edge-pool-size 32
```

#### Figure: token 1 score decomposition panel

![S4B.1d token 1 decomposition](assets/satloc/s4b_structural_retrieval/s4b1d_token0001_luma_hog_edge_modecrop_r512_c8_b9_e32_score_decomposition_panel.png)

How to read:

- Column 1: RGB candidate tile.
- Column 2: candidate Sobel map.
- Column 3: absolute Sobel difference against UAV query; white means different.
- Column 4: 8x8 per-cell HOG similarity.
- Column 5: numeric score details.

Interpretation:

- The 8x8 grid is not pixel-level. Each square represents a 64x64 image region.
- Yellow means the local gradient-direction statistics are similar.
- Forest false positives can have many yellow cells because dense vegetation creates gradients everywhere.
- HOG does not know if the edge came from a pond boundary, road, roof, or forest texture.

#### Figure: token 1 score barplot

![S4B.1d token 1 score barplot](assets/satloc/s4b_structural_retrieval/s4b1d_token0001_luma_hog_edge_modecrop_r512_c8_b9_e32_score_barplot.png)

Interpretation:

- False positives had high HOG/edge similarity.
- GT center was present but not ranked first.
- This confirmed that the method was retrieving structural similarity, not reliable place identity.

---

### 4.5 S4B.1e — Texture entropy diagnostic

Run metric extraction:

```bash
python scripts/satloc/s4b_1e_texture_entropy_diagnostic.py \
  --token 1 \
  --preprocess luma \
  --resize-size 512 \
  --cells 8 \
  --bins 9
```

Run plots:

```bash
python scripts/satloc/s4b_1e_plot_texture_entropy.py --token 1
```

Important note:

```text
S4B.1e needs S4B.1d CSV first.
If another token says file not found, run S4B.1d for that token before S4B.1e.
```

#### Figure: similarity vs edge density

![S4B.1e edge density](assets/satloc/s4b_structural_retrieval/s4b1e_token0001_similarity_vs_edge_density.png)

Interpretation:

- False positives sit in the high-similarity, high-edge-density region.
- This means the descriptor rewards tiles with dense edge texture.
- Vegetation/forest can therefore win even when visually unrelated.

#### Figure: similarity vs structural sparsity

![S4B.1e sparsity](assets/satloc/s4b_structural_retrieval/s4b1e_token0001_similarity_vs_sparsity.png)

Interpretation:

- False positives tend to have low sparsity because their texture fills many cells uniformly.
- GT tiles tend to have higher sparsity because road/pond/building edges are cleaner and more localized.
- Current HOG+edge score undervalues that cleaner sparse structure.

#### Figure: sorted texture table

![S4B.1e sorted table](assets/satloc/s4b_structural_retrieval/s4b1e_token0001_sorted_similarity_texture_table.png)

Interpretation:

- Top wrong tiles have high similarity and high edge density.
- GT center has lower edge density and higher sparsity.
- This directly supports the hypothesis that vegetation texture is a major false-positive source.

Key numerical observation from token 1:

```text
False positives:
  edge density ≈ 93–94%
  HOG entropy ≈ 3.14
  sparsity CV ≈ 0.10–0.17

GT center:
  edge density ≈ 89.7%
  HOG entropy ≈ 3.10
  sparsity CV ≈ 0.26
```

---

### 4.6 S4B.1f — Global texture-penalty rerank

Token 1:

```bash
python scripts/satloc/s4b_1f_texture_penalty_rerank.py \
  --token 1 \
  --edge-penalty 0.050 \
  --sparsity-reward 0.045 \
  --entropy-penalty 0.015
```

#### Figure: texture penalty rerank

![S4B.1f rerank](assets/satloc/s4b_structural_retrieval/s4b1f_token0001_texture_penalty_rerank.png)

Interpretation:

- False rank-1 tile 3002 dropped from local rank 1 to local rank 9.
- GT center tile 3545 improved from local rank 9 to local rank 3.
- This proved that texture penalty helped suppress dense vegetation false positives.
- But other structured false positives still remained above GT.

Conclusion:

```text
Global texture penalty helps, but is too coarse.
The score must be controlled at cell level, not only whole-image level.
```

---

### 4.7 S4B.1g — Cell-level texture-suppressed HOG rerank

Token 1:

```bash
python scripts/satloc/s4b_1g_cell_texture_suppressed_rerank.py \
  --sequence traj01 \
  --token 1 \
  --preprocess luma \
  --resize-size 512 \
  --cells 8 \
  --bins 9 \
  --edge-threshold 0.05 \
  --blend-original 0.30
```

Token 100:

```bash
python scripts/satloc/s4b_1g_cell_texture_suppressed_rerank.py \
  --sequence traj01 \
  --token 100 \
  --preprocess luma \
  --resize-size 512 \
  --cells 8 \
  --bins 9 \
  --edge-threshold 0.05 \
  --blend-original 0.30
```

Important note:

```text
S4B.1g needs S4B.1d CSV first.
Run S4B.1d for a token before running S4B.1g for that token.
```

#### Figure: token 1 rerank barplot

![S4B.1g token 1 barplot](assets/satloc/s4b_structural_retrieval/s4b1g_token0001_luma_r512_c8_b9_eth0.05_blend0.3_barplot.png)

Interpretation:

- GT center tile 3545 moved from original local rank 9 to new local rank 1.
- Error was approximately 15.0 m.
- This shows that cell-level texture suppression can fix some vegetation-dominated failure cases.

#### Figure: token 1 inspection panel

![S4B.1g token 1 inspection](assets/satloc/s4b_structural_retrieval/s4b1g_token0001_luma_r512_c8_b9_eth0.05_blend0.3_inspection_panel.png)

How to read:

- `texture badness`: high entropy × high edge density cells. Bright means likely noisy/vegetation-like texture.
- `candidate cell weight`: dark cells are suppressed, bright cells are trusted more.
- `raw per-cell HOG sim`: original HOG cell similarity.
- `weighted contribution`: final cells that actually drive the score after suppression.

Interpretation:

- Dense chaotic cells are suppressed.
- Cleaner structural cells are weighted more.
- For token 1, this was enough to move the GT center to rank 1 locally.

#### Figure: token 100 rerank barplot

![S4B.1g token 100 barplot](assets/satloc/s4b_structural_retrieval/s4b1g_token0100_luma_r512_c8_b9_eth0.05_blend0.3_barplot.png)

Interpretation:

- A GT-neighbor tile remains rank 1, but many false positives still remain high.
- This shows cell-level texture suppression helps, but it does not fully solve the false-positive issue.

#### Figure: token 100 inspection panel

![S4B.1g token 100 inspection](assets/satloc/s4b_structural_retrieval/s4b1g_token0100_luma_r512_c8_b9_eth0.05_blend0.3_inspection_panel.png)

Interpretation:

- Structured false positives remain difficult.
- Even after vegetation suppression, HOG+edge cannot reliably identify whether a clean edge belongs to road, roof, pond boundary, or unrelated built-up/construction area.
- This is the key stopping point for pure classical HOG/Sobel tuning.

---

## 5. Final interpretation of S4B.1

### What improved over ORB

```text
ORB:
  local binary patch matching
  failed full-map retrieval
  top-1 errors often > 1 km

HOG+edge:
  coarse structural layout matching
  often retrieves visually meaningful/nearby neighborhoods
  sometimes reaches tens-of-meters local candidates
```

### What still fails

```text
HOG+edge cannot understand semantic identity.
It sees gradients, not objects.
```

Failure examples:

- forest edge texture can resemble query structure
- pond edge and vegetation edge are both just gradients
- road boundary and roof boundary are both strong oriented edges
- construction land and building layouts can produce similar HOG grids
- repeated map patterns create high-scoring false positives

### Why tuning stops here?

S4B.1g showed that texture suppression helps. But token 100 still showed false positives after suppression. This means the bottleneck is not just a parameter issue.

The bottleneck is descriptor representation.

```text
Sobel/HOG is not enough for robust absolute map localization.
```

---

## 6. Recommended next candidate method

The next phase should be treated as a new method, not another HOG tuning block.

Recommended directions:

1. Semantic/class-aware masking  
   Separate vegetation, water, road, building, bare land before matching.

2. Learned global retrieval descriptors  
   Test DINO/CLIP/NetVLAD-style image embeddings for UAV-to-satellite retrieval.

3. Hybrid retrieval  
   Use HOG+edge for coarse top-k, then rerank with semantic/learned descriptors.

4. Larger context/mosaic matching  
   Compare UAV frame against a wider satellite context instead of single isolated tiles.

5. Temporal consistency  
   Use multiple consecutive UAV frames and enforce motion continuity instead of single-frame retrieval.

---

## 8. Status

S4B.1 is complete.

This phase should be presented as:

```text
A classical structural retrieval baseline that improved over ORB,
identified vegetation/texture false positives,
and showed why pure HOG/Sobel is not enough for reliable full-map localization.
```

Next step: start a new candidate method, likely semantic/learned/hybrid retrieval.
