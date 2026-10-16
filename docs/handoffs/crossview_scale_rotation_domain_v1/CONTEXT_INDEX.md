# Context index — current canonical navigation

**Branch:** research/crossview-scale-rotation-domain-v1  
**Updated:** 2026-10-16

This index replaces the older CV3-first navigation order.

## 1. Read first

1. ../CURRENT_BEST_VILLOC_LOCALIZATION_STACK.md
2. FA1_OB1_CLOSEOUT.md
3. ../PHYSICAL_DRONE_TRANSITION_HANDOFF_20261016.md
4. README.md
5. NEW_CHAT_BOOTSTRAP.md

## 2. Current frozen absolute-localization evidence

- CV1_CLOSEOUT.md — scale/rotation study
- CV2_CLOSEOUT.md — SIFT+LightGlue promotion
- CV2_SIFT_LIGHTGLUE_DEEP_DIVE.md — Top-k/oracle/error-tail interpretation
- CV3_CLOSEOUT.md — aerial-MASt3R compact-screen closeout
- FA1_FAILURE_ANALYSIS_PLAN.md — failure-analysis contract
- FA1_1_IMPLEMENTATION.md
- FA1_2_IMPLEMENTATION.md
- FA1_3_IMPLEMENTATION.md
- FA1_OB1_CLOSEOUT.md — consolidated failure-analysis + oblique closeout

## 3. Blind SIFT transfer

Read:

- ../blind_demo_sift_lightglue_transfer_v1/BV1_0_IMPLEMENTATION.md
- ../blind_demo_sift_lightglue_transfer_v1/BV1_2_IMPLEMENTATION.md
- ../blind_demo_sift_lightglue_transfer_v1/BV1_2_VISUAL_DIAGNOSTICS.md
- ../blind_demo_sift_lightglue_transfer_v1/EXPANDED_MAP_AND_TRAJ01_VISUALS.md

## 4. Earlier candidate/retrieval lineage

Read only when the question depends on why the current Top20/map exists:

- ../retrieval_candidate_pool_v2/README.md
- ../query_view_candidate_generation_v1/README.md
- ../local_map_neighborhood_subtile_consensus_v1/README.md
- ../top20_geometric_discrimination_v1/GD1_0_REFERENCE_FREE_EVIDENCE.md
- ../../closeouts/top20_geometric_discrimination_gd1_3b/README.md
- ../../closeouts/learned_geometric_frontend_ablation_v1/README.md

## 5. Older full-pipeline / internship lineage

- ../../README_satloc_s3_5_visual_diagnostics_and_s4_transition.md
- ../../README_satloc_s4a_orb_fullmap_baseline.md
- ../../README_satloc_s4b_structural_retrieval_detailed.md
- ../../README_satloc_s4c6_failure_analysis_closeout.md
- ../../README_satloc_s5_absolute_lightglue_results.md
- ../../README_satloc_s6_relative_absolute_fusion.md
- ../../README_s7_relative_frontend_closeout_s7c_transition.md
- ../../README_s7b_learned_retrieval_closeout_s7d_transition.md
- ../../README_villoc_s8_1_to_s8_10b_geometric_oracle_closeout.md
- ../../README_villoc_S8_11A0_to_S8_12C1_closeout.md
- ../../README_villoc_s8_12d_retrieval_diagnostics_s8_12e0_plan.md
- ../../README_villoc_traj01_s8_full_pipeline_fusion_closeout.md
- ../../demo/blind_recorded_flight_final_001/README.md

## 6. Important frozen numbers

Current map:
768_s256, EPSG:3346, approximate 153.6 m tile footprint, 51.2 m stride.

SIFT+LightGlue full403:
349 containment, 205 center<=40, 346 center<=80, 367 projected<=40,
368 projected<=80, unsafe 30.

SIFT full403 evidence:
c0031242f0b359f1bfa3c87ea47ac428393ed3e8e03d7b5c6d0822e8822cb8a0

scale_m2 ORB evidence:
94353f763725f44f278192bcedcfd139ab136ecebf4f1dcc6e07487380964810

## 7. Do not restart closed work

Do not restart generic:
- XFeat absolute tuning;
- SuperPoint+LightGlue tuning;
- ALIKED+LightGlue tuning;
- CV1 scale_m2 sweep;
- CV3 MASt3R full expansion;
- FA1 threshold tuning.

Do not select a confidence threshold from the 403 development answers.

## 8. Current next action

The next project phase is **physical-drone instrumentation**, not another
absolute matcher.

Read:

../PHYSICAL_DRONE_TRANSITION_HANDOFF_20261016.md

Reopen localization research only after camera calibration/geometry, Pixhawk
telemetry, AGL/range, and synchronization are documented sufficiently to define
a causal physical-GSD protocol.
