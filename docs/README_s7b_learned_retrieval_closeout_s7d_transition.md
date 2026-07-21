# S7B Learned Retrieval Closeout — DINOv2 / DINOv2-VLAD / FOV Rescue

**Project:** GNSS-denied UAV localization / SatLoc absolute retrieval  
**Stage closed:** S7B learned candidate generation  
**Next block prepared:** S7D.0 budget-aware candidate merge for verifier/fusion input  
**Status:** Freeze DINOv2/VLAD experimentation for now; continue pipeline integration

> **Locked rule:** GT/reference coordinates are used only after retrieval ranking for offline evaluation. They are never used for descriptor extraction, codebook fitting, ranking, candidate generation, acceptance, or correction.

---

## 1. Why S7B was needed

Earlier absolute retrieval showed that the bottleneck was the candidate pool. The verifier cannot recover a correct match if the correct satellite tile is not inside the candidate set.

Previous classical retrieval ceiling:

```text
S5C PHOG/LSD union oracle@50: 102/263 = 38.8%
S5C LightGlue selected hits:   68/263 = 25.9%
```

S7B therefore tested learned retrieval to raise this candidate-pool ceiling before LightGlue and relative/absolute fusion.

---

## 2. Blocks completed

```text
S7B.1  DINOv2 global descriptor baseline
S7B.2  DINOv2 patch-token VLAD / AnyLoc-style retrieval
S7B.3  FOV / crop-mode diversification and RRF union
S7B.4  Failure-token diagnostic and budget-aware rescue analysis
```

---

## 3. S7B.1 — DINOv2 global descriptor baseline

### Method

DINOv2 ViT-S/14 was used off-the-shelf. Each image was represented by an average-pooled DINO patch-token descriptor.

Two sizes were tested:

```text
img224: fast baseline
img518: DINO patch-aligned higher-resolution baseline
```

### Results

| Variant | Recall@1 | Recall@5 | Recall@10 | Recall@20 | Recall@50 | Recall@100 | Median oracle error | Median oracle rank |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DINOv2 global img224 | 32/263 = 12.2% | 81/263 = 30.8% | 103/263 = 39.2% | 115/263 = 43.7% | 138/263 = 52.5% | 166/263 = 63.1% | 24.535 m | 32 |
| DINOv2 global img518 | 37/263 = 14.1% | 84/263 = 31.9% | 101/263 = 38.4% | 119/263 = 45.2% | 136/263 = 51.7% | 158/263 = 60.1% | 25.423 m | 33 |

### Interpretation

DINOv2 global descriptors already beat the engineered PHOG/LSD candidate pool, but global average pooling is not strong enough for final top-rank localization. It remains a useful comparison baseline and possible future auxiliary stream.

---

## 4. S7B.2 — DINOv2-VLAD / AnyLoc-style retrieval

### Method

S7B.2 used local DINO patch tokens and VLAD aggregation:

```text
image
  -> DINOv2 ViT patch tokens
  -> MiniBatchKMeans visual codebook
  -> VLAD residual aggregation
  -> cosine retrieval
```

Frozen promoted settings:

```text
Model:             DINOv2 ViT-S/14
Image size:        224
Crop mode:         center_square
VLAD clusters:     32
Codebook:          500 satellite tiles
Patches/image:     32
Candidate Top-K:   100
Device:            CPU
```

### Full result

```text
Top-1 hits:          72/263 = 27.4%
Top-5 hits:         113/263 = 43.0%
Top-10 hits:        132/263 = 50.2%
Top-20 hits:        151/263 = 57.4%
Top-50 hits:        182/263 = 69.2%
Top-100 hits:       206/263 = 78.3%
Median top-1 error: 307.102 m
Median oracle error: 17.647 m
Median oracle rank: 21
Retrieval runtime:  1.610 s
```

### Scene-wise result

| Scene | Queries | Top-1 hit rate | Recall@50 | Recall@100 | Median oracle error |
|---|---:|---:|---:|---:|---:|
| mixed_urban_natural | 125 | 32.8% | 67.2% | 79.2% | 18.673 m |
| water_wetland | 53 | 24.5% | 81.1% | 84.9% | 14.953 m |
| forest_canopy | 50 | 22.0% | 74.0% | 84.0% | 16.014 m |
| urban | 24 | 29.2% | 75.0% | 83.3% | 16.168 m |
| agricultural_open_field | 11 | 0.0% | 0.0% | 0.0% | 170.084 m |

### Interpretation

DINOv2-VLAD is the strongest candidate generator so far. It increases candidate-pool recall from the old 38.8% classical ceiling to 78.3% at Top-100. The remaining hard scene is agricultural_open_field.

---

## 5. S7B.3 — FOV / crop-mode diversification

### Why union was tested

Two FOV treatments were compared:

```text
center_square:
  crops to central square
  stronger geometry
  may lose border / wider-FOV context

resize_square:
  preserves full UAV image content
  keeps border / wider-FOV context
  may distort geometry and scale
```

The union asked whether resize_square recovers candidates missed by center_square. This was a coverage-diversity test, not final localization.

### Results

| Stream | Recall@1 | Recall@5 | Recall@10 | Recall@20 | Recall@50 | Recall@100 |
|---|---:|---:|---:|---:|---:|---:|
| center_square | 72/263 = 27.4% | 113/263 = 43.0% | 132/263 = 50.2% | 151/263 = 57.4% | 182/263 = 69.2% | 206/263 = 78.3% |
| resize_square | 40/263 = 15.2% | 85/263 = 32.3% | 116/263 = 44.1% | 137/263 = 52.1% | 170/263 = 64.6% | 188/263 = 71.5% |
| RRF union | 59/263 = 22.4% | 109/263 = 41.4% | 134/263 = 51.0% | 148/263 = 56.3% | 182/263 = 69.2% | 206/263 = 78.3% |

Rescue breakdown:

```text
union_hits_at_100:              206
center_hits_at_100:             206
resize_hits_at_100:             188
union_new_vs_any_single_at_100:   0
center_unique_hits_at_100:       30
resize_unique_hits_at_100:       12
```

### Interpretation

center_square remains the main stream. resize_square is weaker overall, but it contains 12 unique hits that center misses. Simple RRF Top-100 does not exploit those 12 rescues, so resize should be kept as an auxiliary rescue source for budget-aware merging.

---

## 6. S7B.4 — Failure-token and budget-aware rescue diagnostic

### Result

```text
Anchor stream:         center
Failure budget:        Top-100
Anchor hits@100:       206/263
Any-stream hits@100:   218/263
Non-anchor rescues:    12
Anchor misses@100:     57
```

Failure groups:

| Failure group | Count | Interpretation |
|---|---:|---|
| anchor_success | 206 | solved by center-square DINOv2-VLAD |
| near_pool_miss_40_100m | 37 | close but not within 40 m; possible soft/geographic/tile-neighbor rescue |
| non_anchor_rescue | 12 | missed by center but rescued by resize |
| pool_failure_or_far_ambiguity | 8 | hardest failures / likely ambiguous or absent pool |

Anchor misses by scene:

| Scene | Anchor misses |
|---|---:|
| mixed_urban_natural | 26 |
| agricultural_open_field | 11 |
| forest_canopy | 8 |
| water_wetland | 8 |
| urban | 4 |

### Interpretation

The problem is no longer broadly poor candidate generation. The remaining problem is how to exploit auxiliary streams, near-pool misses, and scene-specific ambiguities.

S7B.4 shows that the potential candidate ceiling improves from:

```text
center-only Top-100: 206/263 = 78.3%
any-stream Top-100:  218/263 = 82.9%
```

---

## 7. Visual diagnostic interpretation

S7B.4 generated visual panels for selected failure tokens.

Panel meanings:

```text
Original UAV image:
  raw query frame

DINO input:
  exact crop/resize passed to DINOv2

ViT patch grid:
  16x16 patch layout for image size 224 and patch size 14

ORB keypoints:
  classical sparse-feature contrast only

DINO token PCA:
  color projection of learned patch embeddings

DINO token deviation:
  patches that differ strongly from image-average representation

VLAD cluster assignment:
  nearest visual codeword per patch

Top-1 satellite:
  highest-ranked retrieved satellite candidate

Best available candidate:
  lowest-error candidate inside the available ranked streams
```

Important note:

```text
DINOv2 is not detecting sparse keypoints like ORB/SIFT.
It represents dense image patches using learned ViT token vectors.
The pixelated diagnostic maps are expected because ViT works on patch tokens.
```

---

## 8. Main comparison summary

| Method | Role | Best result |
|---|---|---|
| PHOG/LSD classical retrieval | old handcrafted candidate generator | 102/263 oracle@50 |
| LightGlue selected output | verifier output from old pool | 68/263 |
| DINOv2 global img224 | learned global baseline | 166/263 Recall@100 |
| DINOv2 global img518 | high-resolution global baseline | 158/263 Recall@100 |
| DINOv2-VLAD center img224 | promoted main generator | 206/263 Recall@100 |
| DINOv2-VLAD resize img224 | auxiliary rescue stream | 188/263 Recall@100; 12 unique rescues |
| Any-stream availability | potential merge ceiling | 218/263 Recall@100 |

---

## 9. Frozen decisions

```text
S7B.1 COMPLETE:
DINOv2 global retrieval beats classical retrieval but is not promoted as the main stream.

S7B.2 COMPLETE_PROMOTE:
Promote DINOv2-VLAD center_square img224 K32 cb500 as the main candidate generator.

S7B.3 COMPLETE_DIAGNOSTIC:
Do not use simple RRF union as final policy.
Keep resize_square only as auxiliary rescue.

S7B.4 COMPLETE:
Failure analysis identifies 12 non-anchor rescues, 37 near-pool misses, and 8 hard failures.

DINOv2/VLAD tuning is frozen for now.
```

---

## 10. Future stronger learned methods: DINOv3, teacher-student, SimCLR-style augmentation

This stage used a base off-the-shelf DINOv2 ViT-S/14. It was not domain-adapted, distilled, fine-tuned, or trained on UAV/satellite imagery.

Future stronger routes:

```text
DINOv3 feature extractor swap
DINOv3 + VLAD / DINOv3 + GeM / DINOv3 + learned projection
teacher-student distillation from a larger retrieval teacher to a smaller deployable student
SimCLR-style contrastive learning with UAV/satellite augmentations
domain-specific contrastive or distillation training
larger or scene-balanced VLAD codebooks
K64 / K128 VLAD clusters
multi-crop or multi-scale DINO token aggregation
GPU acceleration for larger models
```

Recommended future order:

```text
1. Swap DINOv2 -> DINOv3 without changing the retrieval protocol.
2. Rerun global + VLAD baselines.
3. Compare against frozen S7B.2 center result.
4. Only then consider teacher-student/domain distillation.
5. For SimCLR-style work, design augmentations carefully because too much invariance can remove location-critical cues.
```

---

## 11. Next block prepared — S7D.0 budget-aware candidate merge

### Goal

Convert S7B.4 findings into a practical candidate list for LightGlue / absolute correction fusion.

Instead of simple RRF Top-100, create merged candidates using GT-free stream metadata:

```text
center_square candidates as primary stream
resize_square unique candidates as rescue stream
deduplicate tile_id
preserve source/rank/score metadata
evaluate budgets 100, 125, 150, 200
```

### Online-safe inputs

Allowed for ranking/merge:

```text
stream name
stream rank
stream score
tile_id
duplicate presence across streams
candidate source count
retrieval metadata
```

Not allowed for ranking/merge:

```text
GT lon/lat
reference trajectory
eval_error_m
whether a tile is within 40 m
oracle rank
```

### Candidate merge policies

#### Policy A — center-first append resize-unique

```text
1. take center candidates in rank order
2. append resize candidates not already present
3. deduplicate tile_id
4. cut to budget K
```

#### Policy B — balanced source budget

```text
merge@100:
  center top80 + resize top40 unique -> dedupe -> source-priority ranking

merge@150:
  center top100 + resize top75 unique -> dedupe -> source-priority ranking
```

#### Policy C — RRF with larger budget

```text
Use reciprocal-rank fusion, but evaluate K=125, 150, 200.
```

### Success criteria

```text
Baseline center Recall@100: 206/263
Any-stream ceiling@100:     218/263

S7D.0 should aim for:
Recall@150 >= 218/263
or
clear rescue gain with acceptable verifier budget.
```

The most important scene target:

```text
agricultural_open_field: currently 0/11 at center Top-100
```

---