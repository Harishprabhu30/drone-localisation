# SatLoc S4A — ORB Full-Map Retrieval Baseline

**Project:** GNSS-denied drone localization / UAV-to-map visual localization  
**Dataset:** SatLoc `part_1`, sequence `traj01`  
**Baseline method:** ORB feature matching + Lowe ratio test + RANSAC homography scoring  
**Stage covered:** S4A.1 to S4A.4

---

## 1. Purpose

This document records the completed SatLoc S4A ORB baseline for map-based UAV localization.

The goal was to test whether a UAV query frame can retrieve the correct satellite/orthophoto tile from a map tile database using classical ORB feature matching.

The pipeline used:

```text
UAV query image
      ↓
preprocess image: V2 CLAHE-luma
      ↓
ORB keypoints + descriptors
      ↓
match against satellite tile descriptors
      ↓
Lowe ratio filtering
      ↓
RANSAC homography verification
      ↓
rank satellite tiles by score
      ↓
evaluate using GT lon/lat only after ranking
```

GT lon/lat from the UAV filename was used only for evaluation/debugging, not for retrieval scoring.

---

## 2. Implementation summary

### S4A.1 — Pair match debug

Tested one UAV query against its debug/GT-containing satellite tile.

Result: the pair matcher ran successfully, but true-tile ORB evidence was weak. Even the correct tile produced only a small number of good matches and RANSAC inliers.

### S4A.2 — One query vs local subset

Tested one UAV frame against a small candidate window around the GT tile.

Key finding:

- Query token `1`: correct/near tile appeared in top-10.
- Query token `100`: correct tile did not appear in top-10.
- Local ORB retrieval sometimes finds the right neighborhood, but top-1 is unstable.

Example local subset figures:

![S4A.2 token 1 top-10 panel](assets/s4a2_token0001_subset_top10_panel.png)

![S4A.2 token 1 score by rank](assets/s4a2_token0001_subset_score_by_rank.png)

![S4A.2 token 100 top-10 panel](assets/s4a2_token0100_subset_top10_panel.png)

![S4A.2 token 100 score by rank](assets/s4a2_token0100_subset_score_by_rank.png)

### S4A.2B — Retrieval evidence diagnostics

Compared top false positives with first GT-containing tiles.

Observation:

- ORB matched many vegetation, shadow, and local corner-like patches to unrelated structures.
- Building/road/pond structure was not preserved as a global shape.
- False positives could have more RANSAC inliers than the correct tile.

### S4A.3A — Satellite ORB feature cache

Precomputed ORB descriptors for the full satellite tile database.

Configuration:

```text
variant: V2_clahe_luma
nfeatures: 1200
satellite tiles: 8625
```

300-tile sanity run showed all satellite tiles saturated at 1200 ORB keypoints. This confirmed that keypoint count alone is not useful; ranking must depend on match/RANSAC evidence.

### S4A.3B — One query vs full cached database

Ran full 8625-tile retrieval for single queries.

Important result for token `1`:

| Tile | Search mode | Rank | Score | Good matches | RANSAC inliers | Contains GT | Center error |
|---:|---|---:|---:|---:|---:|---|---:|
| 3544 | local subset | 7 | 6.1 | 11 | 5 | true | 44.0 m |
| 3544 | full map | 528 | 6.1 | 11 | 5 | true | 44.0 m |
| 5471 | full map | 1 | 8.9 | 19 | 7 | false | 1855.8 m |

The cached pipeline was correct: scores were identical between local and full modes. The full map simply introduced many stronger false positives.

### S4A.3C — Full retrieval evidence diagnostics

For full retrieval, the top-ranked tiles were usually far away false positives.

Example for query token `1`:

| Selection | Rank | Tile | GT | Error | Score | Good | Inliers |
|---|---:|---:|---|---:|---:|---:|---:|
| top1 | 1 | 5471 | false | 1855.8 m | 8.90 | 19 | 7 |
| top2 | 2 | 1287 | false | 676.9 m | 8.80 | 18 | 7 |
| top3 | 3 | 830 | false | 1430.9 m | 8.40 | 14 | 7 |
| first_correct | 528 | 3544 | true | 44.0 m | 6.10 | 11 | 5 |
| closest_correct | 7279 | 3545 | true | 15.0 m | 0.60 | 6 | 0 |

Example for query token `100`:

| Selection | Rank | Tile | GT | Error | Score | Good | Inliers |
|---|---:|---:|---|---:|---:|---:|---:|
| top1 | 1 | 2109 | false | 2458.7 m | 10.50 | 15 | 9 |
| top2 | 2 | 4674 | false | 1211.4 m | 10.30 | 13 | 9 |
| top3 | 3 | 1869 | false | 2793.5 m | 9.80 | 18 | 8 |
| first_correct | 377 | 538 | true | 66.7 m | 6.10 | 11 | 5 |
| closest_correct | 2120 | 415 | true | 16.3 m | 4.80 | 8 | 4 |

---

## 3. S4A.4 official multi-query benchmark

### Benchmark command

```bash
export PYTHONPATH=$PWD/src

python scripts/satloc/s4a_4_multi_query_orb_full_benchmark.py \
  --sequence traj01 \
  --max-queries 10 \
  --variant V2_clahe_luma \
  --nfeatures 1200 \
  --ratio 0.75 \
  --ransac-thresh 5.0 \
  --top-k 10 \
  --save-topk-panels
```

### Benchmark configuration

| Item | Value |
|---|---:|
| Sequence | `traj01` |
| Number of query frames | 10 |
| Satellite tiles ranked per query | 8625 |
| Preprocessing variant | `V2_clahe_luma` |
| ORB features | 1200 |
| Ratio test | 0.75 |
| RANSAC threshold | 5.0 px |
| Score | `ransac_inliers + 0.1 * good_matches` |

### Per-query results

| Query | Token | Top-1 tile | Top-1 error (m) | First correct rank | R@1 | R@5 | R@10 | Runtime (s) |
|---:|---:|---:|---:|---:|---|---|---|---:|
| 1 | 1 | 5471 | 1855.8 | 529 | false | false | false | 168.9 |
| 2 | 115 | 5523 | 1467.4 | 229 | false | false | false | 147.7 |
| 3 | 230 | 4722 | 2626.4 | 434 | false | false | false | 233.4 |
| 4 | 345 | 4488 | 3026.7 | 313 | false | false | false | 317.5 |
| 5 | 460 | 4791 | 1010.8 | 495 | false | false | false | 178.3 |
| 6 | 574 | 3817 | 260.5 | 207 | false | false | false | 143.4 |
| 7 | 689 | 2887 | 1509.9 | 122 | false | false | false | 147.8 |
| 8 | 804 | 3274 | 298.5 | 223 | false | false | false | 174.9 |
| 9 | 919 | 4220 | 1896.4 | 155 | false | false | false | 175.9 |
| 10 | 1034 | 5177 | 421.2 | 537 | false | false | false | 224.7 |

CSV copy of this table: [`assets/s4a4_query_summary.csv`](assets/s4a4_query_summary.csv)

### Aggregate metrics

| Metric | Value |
|---|---:|
| Recall@1 | 0.0 |
| Recall@5 | 0.0 |
| Recall@10 | 0.0 |
| First correct rank median | 271.0 |
| First correct rank mean | 324.4 |
| Top-1 error median | 1488.65 m |
| Top-1 error mean | 1437.37 m |
| Closest correct error median | 15.85 m |
| Runtime per query mean | 191.26 s |
| Total benchmark runtime | 1966.00 s |

JSON copy of aggregate metrics: [`assets/s4a4_aggregate_metrics.json`](assets/s4a4_aggregate_metrics.json)

---

## 4. Important figures

### Recall summary

![S4A.4 recall summary](assets/s4a4_recall_summary.png)

### First correct rank per query

![S4A.4 first correct rank](assets/s4a4_first_correct_rank.png)

### Top-1 error per query

![S4A.4 top1 error](assets/s4a4_top1_error.png)

### Runtime per query

![S4A.4 runtime](assets/s4a4_runtime.png)

---

## 5. Interpretation

The S4A ORB full-map baseline is successful as an implementation baseline, but weak as a localization method.

Main finding:

```text
ORB-only global UAV-to-satellite retrieval failed on the 10-query benchmark.
Recall@1, Recall@5, and Recall@10 were all 0.0.
```

This does not mean the code failed. It means the classical ORB signal is not distinctive enough for global map retrieval in this SatLoc setting.

Failure pattern:

1. ORB matches local binary corner-like patches.
2. Vegetation, shadows, field boundaries, roads, and building corners create repeated local structures.
3. False positives often produce more good matches/RANSAC inliers than the true tile.
4. The correct tile can have low center error but poor ORB score.
5. The full 8625-tile database introduces many unrelated but locally similar candidates.

The closest correct tile often had low physical error, around 15 m median, but it ranked far below false positives.

---

## 6. Baseline conclusion for report

Use this paragraph directly in the progress report:

> A classical ORB feature-matching baseline was implemented for SatLoc UAV-to-satellite tile retrieval. Satellite ORB descriptors were precomputed for all 8625 map tiles and UAV query frames were ranked using Lowe-ratio descriptor matching followed by RANSAC homography scoring. While local-subset experiments sometimes placed the correct tile within the top candidates, full-map retrieval over all 8625 tiles failed to retrieve the GT-containing tile within top-10 for 10 sampled `traj01` queries. The benchmark produced Recall@1/5/10 = 0.0, median first-correct rank 271, and median top-1 error 1488.65 m. Evidence diagnostics showed that vegetation, shadows, and repetitive local textures created stronger false-positive ORB matches than the correct map tile. Therefore, ORB is retained as a reproducible classical baseline, but a more global/structural or learned retrieval method is required for the next stage.

## 8. Next stage

Next method branch:

```text
S4B — Global structural retrieval baseline
```

Initial candidate:

```text
Sobel/HOG-style global descriptor on UAV and satellite tiles
+
cosine/L2 retrieval
+
top-k evaluation
```

Reason:

ORB proved that isolated local patches are too ambiguous. The next baseline should explicitly test larger-scale structure such as road/pond/field/building layout before moving to heavier learned retrieval methods.
