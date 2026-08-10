# Blind Visual Localization Run Summary

- **Dataset:** `blind_recorded_flight_final_001`
- **Run ID:** `blind_recorded_flight_final_001`
- **Primary mode:** blind / no-reference localization
- **Evaluation attachment:** not included
- **Status:** PASS

> The localization result is produced without SRT/GPS/ground truth. Reference unavailable: accuracy metrics not computed.

## 1. Dataset summary

- Blind query frames: **123**
- Video-time range: **0.000 s** to **122.000 s**
- Source video: `data/raw/villoc/demonstration/DJI_20260807171627_0003_V_trimmed.MP4`
- Assumed relative altitude: **122.00 m**
- Assumed gimbal pitch: **-90.00°**

## 2. Run mode

- Localization inputs: camera imagery + prepared georeferenced map.
- SRT/GPS/reference used by localization: **No**.
- Post-run evaluation enabled: **No**.
- Localization state: **PROVISIONAL_ABSOLUTE_LOCK**.
- Geographic/map coordinates are available only after causal map initialization.

## 3. Input files

- [Blind query manifest](../metadata/blind_query_manifest.csv)
- [Blind map-aligned relative trajectory](../trajectories/blind_map_aligned_relative_trajectory.csv)
- [Blind temporal correction manifest](../metadata/blind_temporal_fusion/blind_temporal_correction_manifest.csv)
- [Frozen estimated trajectory](../trajectories/submission_estimated_trajectory.csv)

## 4. Map/AOI/cache reuse status

- Map alignment source: `blind map bootstrap`
- Prepared map root: `/Users/harishprabhu/Documents/drone-localisation/outputs/villoc/90_deg`
- Runtime registry: **17 measured**, **1 pending**.
- Map/cache reuse is treated separately from per-flight localization computation.

## 5. Relative localization summary

- Map-aligned relative poses: **94/123**.
- Pre-lock poses without geographic output: **29**.
- Map lock: frame **30**, time **29.000 s**.
- Post-lock relative path length: **1062.56 m**.

## 6. Absolute retrieval/reranking summary

- Absolute retrieval evidence is produced by DINOv2 and geometrically checked/reranked by ORB.
- Queries with DINO/ORB output: **123/123**.
- Hybrid score median: **15.982** (ranking/verifier score; not a calibrated probability).
- Top-K correctness thresholds are not used during blind localization; they are evaluation-only.

## 7. Fusion/correction summary

- Blind correction candidates: **1**.
- Accepted corrections: **1**.
- Rejected candidates: **0**.
- Fusion state: continuous relative propagation with sparse confidence/temporal-gated absolute corrections.
- Decision reasons: `R3V2_CAUSAL_MAP_STATE_EVENT`=1.

## 8. Estimated latitude/longitude export

- Estimated geographic poses: **94/123**.
- Latitude range: **54.74334271** to **54.74561748**.
- Longitude range: **25.26034244** to **25.27053415**.
- **Estimated latitude/longitude are visual map-matching outputs, not GPS inputs.**

## 9. Core runtime and resource summary

The table below intentionally contains only **localization-critical computation**. Plot generation, Folium rendering, CSV writing, and summary-generation overhead are excluded from this headline view.

| Core stage | Measured runtime | Normalized cost | Scope |
|---|---:|---:|---|
| Relative frontend — XFeat | 35.242 s | 288.869 ms/frame_pair | per_new_flight_localization |
| DINO query encoding | 193.883 s | 1576.288 ms/query_image | per_new_flight_localization |
| DINO cached retrieval | 0.006 s | 0.053 ms/query | per_new_flight_localization |
| ORB Top-20 verification/reranking | 182.289 s | 1482.026 ms/query | per_new_flight_localization |
| Blind map bootstrap | 65.295 s | 530.857 ms/query_frame | per_new_flight_localization |
| Map-alignment continuation | 0.033 s | 0.272 ms/trajectory_row | per_new_flight_localization |
| Temporal fusion/control | 0.029 s | 0.240 ms/trajectory_row | per_new_flight_localization |

### Resource context

- Execution devices: `{'dino': 'cpu', 'xfeat': 'cpu', 'orb': 'cpu'}`.
- CUDA available: **False**.
- Total RAM: **8.00 GiB**.

**Interpretation:** the runtime/resource section is for deployment feasibility. Supporting visualization and report-generation costs are measured in the registry for completeness but are not treated as localization bottlenecks.

## 10. Blind-run visual diagnostics

### Blind map-aligned relative trajectory

![Blind map-aligned relative trajectory](../figures/estimated_relative_xy.png)

### Blind fused map trajectory

![Blind fused map trajectory](../figures/estimated_fused_xy.png)

### Blind absolute-localization confidence

![Blind absolute-localization confidence](../figures/confidence_vs_time.png)

### Blind correction decisions

![Blind correction decisions](../figures/accepted_corrections_timeline.png)

- [Open interactive blind estimated trajectory map](../maps/estimated_fused_map.html)

> Reference unavailable: accuracy metrics not computed.

## 11. Post-freeze evaluation

**Reference unavailable: accuracy metrics not computed.**

Run this summary generator with `--include-evaluation` only after a frozen blind submission has been evaluated.

## 12. Generated files

- [Estimated trajectory CSV](../trajectories/submission_estimated_trajectory.csv)
- [Blind relative XY figure](../figures/estimated_relative_xy.png)
- [Blind fused XY figure](../figures/estimated_fused_xy.png)
- [Confidence timeline](../figures/confidence_vs_time.png)
- [Correction-decision timeline](../figures/accepted_corrections_timeline.png)
- [Interactive estimated trajectory map](../maps/estimated_fused_map.html)

## 13. Known limitations and next improvements

- Absolute positioning currently relies on coarse map-tile evidence; sub-tile camera localization is a major accuracy improvement opportunity.
- Relative visual motion remains continuous but accumulates error between trustworthy absolute anchors.
- Absolute corrections are sparse and repeated nearby views may not provide independent map evidence.
- Fixed soft-fusion weighting is a baseline; future work should investigate confidence/uncertainty-aware filtering or graph-based fusion.
- DINO query encoding and ORB verification dominate the tested CPU computation. Higher absolute-update rates require scheduling/optimization and/or faster compute.
- Retrieval ambiguity remains challenging in repeated structures, roads, vegetation, and appearance changes.
- Evaluation metrics must remain separated from blind localization decisions and attached only after output freeze.
