# Villoc traj01 final report assets

This folder contains the curated Villoc traj01 assets selected for the internship report.

## Main report candidates

- `figures/01_final_mission_story_orthophoto_map_clean.png` — **Primary final result figure**. Shows final localization story on the actual orthophoto: reference, relative-only drift, temporal fusion, accepted corrections.
- `figures/02_final_mission_story_strip.png` — **Route-order explanation**. Explains retrieval outcomes, DINO confidence, accepted corrections, and error evolution along distance.
- `figures/03_final_trajectory_overlay_xy.png` — **Clean metric trajectory comparison**. Shows reference, XFeat relative-only, periodic fusion, temporal fusion without orthophoto clutter.
- `figures/04_fusion_error_vs_distance_with_events.png` — **Quantitative fusion improvement**. Shows how relative-only and fusion errors change over route distance with correction events.
- `tables/01_final_system_summary.csv` — **Main metrics table**. Contains RMSE, MAE, median, p95, max, and final error for selected systems.
- `figures/05_relative_frontend_xy_comparison.png` — **Frontend comparison figure**. Compares ORB, KLT optical flow, and XFeat relative trajectories.
- `tables/02_relative_frontend_comparison.csv` — **Frontend comparison metrics**. Numerical summary for ORB/KLT/XFeat relative localization.
- `figures/07_absolute_dino_recall_funnel.png` — **Absolute retrieval candidate funnel**. Shows DINO candidate availability improves from Top-1 to Top-20/Top-100.
- `figures/08_absolute_orb_verifier_funnel.png` — **Verifier/reranking funnel**. Shows DINO candidate pool to ORB-selected correction source.
- `tables/03_absolute_localization_comparison.csv` — **Absolute localization metrics**. Compares DINO Top-1, ORB rerank, and LightGlue full403.
- `tables/04_absolute_funnel_counts.csv` — **Funnel counts table**. Exact counts behind the funnel plots.
- `tables/05_spatial_retrieval_failure_summary.csv` — **Spatial retrieval summary counts**. Counts for Good Top-1, Top-20 yes/Top-1 no, high-confidence wrong, and Top-20 missing.
- `figures/12_factor_confidence_calibration.png` — **Confidence calibration**. Shows DINO score is the strongest simple confidence signal.
- `figures/13_factor_verdict_chart.png` — **Factor verdict chart**. Summarizes which factors are strong, weak, misleading, or not applicable.
- `tables/06_factor_confidence_calibration.csv` — **Confidence calibration table**. Exact bin counts/success rates behind the confidence plot.
- `tables/07_factor_verdict.csv` — **Factor verdict table**. Exact correlation values behind the verdict chart.
- `tables/08_correction_evidence_events.csv` — **Accepted correction event table**. Evidence table with selected tiles, verifier scores, inliers, gaps, and errors.

## Appendix / demo assets

- `figures/06_relative_frontend_error_vs_distance.png` — **Frontend error diagnostic**. Supports why XFeat was selected over weaker classical relative frontend options.
- `figures/09_absolute_orb_vs_lightglue_funnel.png` — **ORB vs LightGlue comparison**. Shows full-run ORB and LightGlue selection differences from the shared DINO Top-20 pool.
- `figures/10_spatial_retrieval_failure_xy.png` — **Spatial failure diagnostic**. Maps good Top-1, reranking opportunities, high-confidence wrong, and Top-20-missing cases.
- `figures/11_spatial_retrieval_failure_distance_strip.png` — **Failure route-order diagnostic**. Shows retrieval outcome by traveled distance.
- `figures/14_correction_evidence_xy_with_gap_labels.png` — **Correction evidence map**. Shows accepted correction events and important gap distances.
- `figures/15_correction_significant_gap_bars.png` — **Correction gap summary**. Simplified visual for long sparse-correction gaps.
- `maps/map_final_mission_story_orthophoto_interactive.html` — **Final interactive orthophoto map**. Interactive version of final mission story map.
- `maps/map_final_trajectory_overlay.html` — **Interactive final trajectory overlay**. Folium version of reference/relative/fusion trajectories.
- `maps/map_spatial_retrieval_failure_512_s256.html` — **Interactive spatial failure map**. Clickable retrieval/failure classes along the route.
- `maps/map_correction_evidence_interactive.html` — **Interactive correction evidence map**. Clickable correction-event diagnostics.
- `manifests/report_08_orthophoto_mission_story_interpretation.md` — **Block 8 interpretation**. Report-ready explanation for the final orthophoto story map.
- `manifests/report_07b_simplified_factor_story_512_s256.md` — **Block 7B interpretation**. Report-ready factor-analysis explanation.
