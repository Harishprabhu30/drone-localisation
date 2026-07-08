# SatLoc S4C.6 Failure Analysis Closeout

**Project:** GNSS-denied UAV localization using visual map matching  
**Dataset:** SatLoc `traj01` diagnostic subset  
**Stage:** S4C.6 — parameter sensitivity, rotation-aware luma-LSD, and failure-group analysis  
**Status:** S4C classical structural retrieval closeout; prepare transition to next method branch.

---

## 1. Purpose

This closeout documents the current SatLoc map-matching pipeline after ORB, HOG+edge, macro-contour PHOG, luma-LSD verification, junction diagnostics, parameter sensitivity, rotation-aware alignment, and failure-group analysis.


## 2. Current Pipeline Summary

Current strongest classical pipeline:

```text
UAV luma image
├── Sobel macro-contour → PHOG descriptor → full-map PHOG top-50 candidate retrieval
└── LSD on luma image → distance-transform line alignment → rerank PHOG top-50
```

Important clarification:

```text
Sobel/PHOG and LSD are not pixel-level fused.
Sobel/PHOG = candidate generator.
luma-LSD = candidate verifier/reranker.
```

---

## 3. Key Results So Far

### 3.1 S4A ORB Full-Map Retrieval

ORB was retained as a reproducible baseline, but it failed as global image-to-map retrieval.

| Metric | Result |
|---|---:|
| Query count | 10 |
| Satellite tiles/query | 8625 |
| Recall@1 / @5 / @10 | 0 / 0 / 0 |
| Median first-correct rank | 271 |
| Median top-1 error | 1488.65 m |
| Mean runtime/query | 191.26 s |

Conclusion:

```text
ORB local binary features are not robust enough for UAV-to-satellite full-map retrieval under scale, view, season, and texture mismatch.
```

---

### 3.2 S4B HOG+Edge Structural Retrieval

HOG+edge improved over ORB by capturing broader layout, but failed in vegetation/forest and repeated rural structures.

Main finding:

```text
HOG+edge captures gradient layout, but not object identity.
Vegetation, shadows, road edges, pond borders, fields, and roofs can produce similar gradient statistics.
```

---

### 3.3 S4C PHOG Candidate Generation

Macro-contour PHOG became a useful candidate generator.

On the 73-frame selected diagnostic subset:

| Method | Top-1 <=20m | Top-1 <=40m | Top-10 <=40m | Median Top-1 Error | Median Best Top-10 Error |
|---|---:|---:|---:|---:|---:|
| PHOG only | 8.22% | 17.81% | 30.14% | 1449.88 m | 485.04 m |

Interpretation:

```text
PHOG often places a near-correct candidate inside top-50, but not reliably at rank 1.
```

---

### 3.4 S4C.4C luma-LSD Reranking

luma-LSD distance-transform verification was the strongest classical reranking signal.

On the 73-frame selected subset:

| Profile | Top-1 <=20m | Top-1 <=40m | Top-10 <=40m | Median Top-1 Error | Median Best Top-10 Error |
|---|---:|---:|---:|---:|---:|
| lsd_only | 13.70% | 30.14% | 42.47% | 706.55 m | 223.11 m |
| lsd_strong | 15.07% | 30.14% | 41.10% | 1007.58 m | 223.11 m |
| phog_only | 8.22% | 17.81% | 30.14% | 1449.88 m | 485.04 m |

Main conclusion:

```text
luma-LSD reranking improved PHOG top-1 <=40m from 17.8% to 30.1% and reduced median top-1 error from 1449.9 m to 706.6 m.
```

---

### 3.5 S4C.4D Confidence-Gated Reranker

Confidence gating used unsupervised scores only:

```text
PHOG margin
LSD score gap
LSD shift magnitude
LSD shift-boundary flag
LSD basin area
```

It did **not** include Peak Salience Fusion, Entropy Gating, or Structural Sparsity Gating.

Best gated result:

| Profile | Top-1 <=20m | Top-1 <=40m | Median Error | Mean Error |
|---|---:|---:|---:|---:|
| gate_phog_anchor_else_lsd | 13.70% | 30.14% | 624.58 m | 883.10 m |
| lsd_only | 13.70% | 30.14% | 706.55 m | 859.02 m |
| phog_only | 8.22% | 17.81% | 1449.88 m | 1359.25 m |

Conclusion:

```text
Confidence gating did not improve <=40m hit rate beyond luma-LSD, but reduced median error from 706.6 m to 624.6 m.
```

---

### 3.6 S4C.5 Junction-Aware luma-LSD Frontend

Junction diagnostics detected L/T/X junctions from luma-LSD line intersections.

73-frame role averages:

| Role | Junction Count | L Count | T Count | X Count | Junction Salience | luma-LSD Lines | Mean Error |
|---|---:|---:|---:|---:|---:|---:|---:|
| UAV query | 20.25 | 14.49 | 5.73 | 0.03 | 2092.91 | 89.59 | 0.00 m |
| GT / nearest | 11.40 | 8.67 | 2.73 | 0.00 | 1036.82 | 70.22 | 13.07 m |
| LSD top1 | 11.59 | 8.90 | 2.68 | 0.00 | 1087.73 | 71.62 | 859.02 m |
| PHOG top1 | 6.05 | 4.71 | 1.34 | 0.00 | 590.50 | 44.05 | 1359.25 m |

Conclusion:

```text
Junctions separate PHOG false positives from more structural candidates, but raw junction count/salience is not enough because line-rich false positives also exist.
Junctions should be used through spatial alignment or graph matching, not standalone count-based gating.
```

---

### 3.7 S4C.6A Parameter Sensitivity Sweep

A YAML-based parameter sweep was added for controlled sensitivity testing.

Tested parameter families:

```text
Macro/Sobel frontend:
  threshold_percentile, blur_ksize, min_component_area

LSD frontend:
  min_line_length, max_lines

Alignment:
  line_thickness, max_shift_px, symmetric_weight
```

The 10-token diagnostic subset was saturated: many settings achieved the same top-level result.

Top result pattern:

| Setting Family | Top-1 <=40m | Median Top-1 Error |
|---|---:|---:|
| baseline / thick / dense / blur variants with lsd_strong | 90% | 17.26 m |

Conclusion:

```text
The 10-token subset is too easy for final hyperparameter selection.
Parameter changes did not break the method, but the real failures are in rotated, agricultural, pond, and low-structure frames.
```

---

### 3.8 S4C.6B Rotation-Aware luma-LSD Alignment

Rotation-aware luma-LSD alignment was tested by rotating only the UAV luma-LSD canvas inside PHOG top-N candidates.

Modes tested:

```text
fixed angles: [-30, -20, -10, 0, 10, 20, 30]
orientation_prior: estimate likely angles from LSD orientation histograms
```

Observed result on hard-token run:

| Profile | Top-1 <=40m | Median Top-1 Error |
|---|---:|---:|
| rot_phog_protected | 6.25% | 1514.26 m |
| rot_lsd_only | 6.25% | 1655.20 m |
| rot_lsd_strong | 6.25% | 1883.34 m |
| phog_only | 0.00% | 2082.13 m |

Orientation-prior mode was similar or worse.

Important note:

```text
The run displayed num_queries = 16 even though 6 hard tokens were expected, so token inclusion should be checked before final reporting.
```

Conclusion:

```text
Brute rotation-aware luma-LSD did not solve the hard failures. Extra rotation freedom often allows false line-rich candidates to align better, so rotation search alone is unsafe without stronger structural/semantic constraints.
```

---

## 4. S4C.6C Failure-Group Analysis

S4C.6C grouped all 73 selected frames using:

```text
PHOG top1 error
luma-LSD top1 error
oracle best PHOG top50 error
```

Threshold: **40 m**.

### 4.1 Failure Group Summary

| Failure Group | Count | Rate | Median PHOG Error | Median LSD Error | Median Oracle Top50 Error | Main Meaning |
|---|---:|---:|---:|---:|---:|---|
| candidate_pool_failure | 37 | 50.68% | 1889.03 m | 1234.38 m | 295.22 m | Correct candidate often not close enough in PHOG top50. Candidate generation limitation. |
| lsd_rescue | 11 | 15.07% | 1449.88 m | 23.20 m | 19.47 m | luma-LSD successfully rescued PHOG failures. |
| stable_success | 11 | 15.07% | 20.69 m | 19.85 m | 19.85 m | Both PHOG and LSD work. |
| selection_failure_correct_in_pool | 9 | 12.33% | 725.42 m | 960.89 m | 23.23 m | Correct candidate exists in top50 but selection/reranking fails. |
| weak_pool_near_candidate | 3 | 4.11% | 1606.13 m | 588.57 m | 78.63 m | Some near candidate exists, but not within 40m. |
| lsd_destroyed_phog_success | 2 | 2.74% | 18.18 m | 1417.56 m | 18.18 m | LSD destroys already-good PHOG result. |

### 4.2 Representative Tokens

| Group | Tokens | Key Interpretation |
|---|---|---|
| candidate_pool_failure | 387, 573, 577, 366 | PHOG top50 candidate pool itself is weak. Better candidate generator or learned/global retrieval needed. |
| lsd_destroyed_phog_success | 494, 269 | LSD line-rich false positives can override correct PHOG. Need confidence protection / semantic guard. |
| lsd_rescue | 90, 1, 310, 990 | luma-LSD is valuable; it can recover correct location when PHOG top1 fails. |
| selection_failure_correct_in_pool | 276, 694, 1015, 107 | Correct candidate exists, but reranking chooses wrong tile. Need better verifier. |
| stable_success | 516, 40, 905, 874 | Good structural overlap; current method works. |
| weak_pool_near_candidate | 434, 564, 937 | Candidate pool is partially close but not precise enough. |

---

## 5. Visual Failure Causes From Inspection

Main observed limitations:

```text
1. Satellite image is older, lower-zoom, blurred, and seasonally different.
2. UAV images are sharper, newer, and sometimes rotated.
3. Roadside vegetation and pond boundaries differ strongly between UAV and satellite.
4. Rural/agricultural areas lack stable long lines and junctions.
5. Rotation-aware brute search adds too much freedom and can reward wrong line-rich candidates.
6. Correct candidates often exist inside PHOG top50, but selection remains weak.
7. In more than half of frames, the candidate pool itself is not strong enough.
```

---

## 6. Figures and Assets

After running the copy script, README figures are expected under:

```text
docs/assets/
```

### 6.1 Failure Group Counts

![S4C.6 failure group counts](docs/assets/s4c6_figures/s4c6_failure_group_counts.png)

### 6.2 PHOG vs luma-LSD Error

![S4C.6 PHOG vs LSD error](docs/assets/s4c6_figures/s4c6_phog_vs_lsd_error.png)

### 6.3 Selection Gap to Oracle

![S4C.6 selection gap to oracle](docs/assets/s4c6_figures/s4c6_selection_gap_to_oracle.png)

### 6.4 Structure Availability vs Error

![S4C.6 structure vs error](docs/assets/s4c6_figures/s4c6_structure_vs_error.png)

### 6.5 Representative Failure Panels

Copied representative panels are placed under:

```text
docs/assets/s4c6_representatives/
```

Suggested manual inspection order:

```text
lsd_rescue/token0001 or token0090
stable_success/token0040 or token0516
lsd_destroyed_phog_success/token0269 or token0494
selection_failure_correct_in_pool/token0276 or token0694
candidate_pool_failure/token0387 or token0573
weak_pool_near_candidate/token0434 or token0937
```

---

## 7. Interpretation and Method Decision

### What worked

```text
1. PHOG is useful as a broad structural candidate generator.
2. luma-LSD is useful as a verifier/reranker inside PHOG top50.
3. Confidence gating stabilizes median error slightly.
4. Junction features are meaningful but need spatial/graph matching, not count-only gating.
```

### What did not work

```text
1. ORB full-map retrieval.
2. Pure macro-contour Chamfer reranking.
3. Simple temporal smoothing on sparse frames.
4. Raw junction count/salience as direct score.
5. Brute rotation-aware luma-LSD alignment.
```

### Current strongest classical result

```text
S4C.4D gate_phog_anchor_else_lsd:
  Top1 <=40m: 30.1%
  Median top1 error: 624.6 m
```

This is better than PHOG-only but not sufficient as a final absolute localizer.

---

## 8. Recommended Next Branch

The next branch should not continue random classical tuning. S4C has reached a clear diagnostic ceiling.

Recommended next direction:

```text
S5A — learned/local matcher comparison inside PHOG top-K
```

Controlled use:

```text
PHOG / structural retrieval gives top50 candidates
↓
learned matcher or stronger local verifier runs only inside top-K
↓
evaluate on S4C.6C failure groups
```

Candidate methods to compare later:

```text
LightGlue / SuperPoint-style feature matching
LoFTR-style detector-free matching
RoMA / dense matching style methods
DINO/CLIP/NetVLAD/CosPlace-style retrieval features only if compute permits
```

But do not run learned methods blindly over the full map. Use them as top-K verification first.

Alternative classical continuation:

```text
S5B — semantic/region-aware structural matching
```

Examples:

```text
water/pond mask
road/field/vegetation region cues
junction graph matching with spatial constraints
rotation-normalized structural descriptor
```
---

## 09. Final Closeout Statement

```text
S4C established a reproducible classical structural map-matching pipeline. ORB failed as global retrieval, HOG+edge and PHOG improved candidate generation, and luma-LSD provided the strongest verification signal. However, failure-group analysis shows that more than half of selected frames still suffer from candidate-pool failure, while a smaller subset contains correct candidates that current reranking cannot select. Rotation-aware brute alignment did not solve this because it increased false-positive flexibility. Therefore, the next stage should evaluate learned or semantic/region-aware verification inside the PHOG top-K pool, using S4C.6C failure groups as the benchmark split.
```
