# Context index — crossview scale / rotation / domain v1

Use this file as the navigation map for a new session.

## Start here
1. `README.md` in this directory — canonical current state.
2. `NEW_CHAT_BOOTSTRAP.md` — copy/paste prompt for a new chat.

## Previous closeouts
- `../../closeouts/learned_geometric_frontend_ablation_v1/README.md`
- `../../closeouts/learned_geometric_frontend_lf2_1c/README.md`
- `../../closeouts/top20_geometric_discrimination_gd1_3b/README.md`

## Retrieval / candidate-pool context
- `../retrieval_candidate_pool_v2/README.md`

## Main VILLOC pipeline context
- `../../README_villoc_traj01_s8_full_pipeline_fusion_closeout.md`
- `../../demo/blind_recorded_flight_final_001/README.md`

## Reporting
- `../../../configs/reporting/villoc_method_benchmark_registry.yaml`
- Do not merge n=40 screening results into n=403 tables without an explicit denominator-aware reporting stage.

## External research queue
- AerialMegaDepth project: https://aerial-megadepth.github.io/
- AerialMegaDepth paper: https://arxiv.org/abs/2504.13157
- SIFT+LightGlue: inspect the pinned cvg/LightGlue checkout before use
- RDD / aerial-domain features: research only after CV1 causal diagnostic
- EfficientLoFTR / LoFTR: detector-free candidate family
- RoMa: dense robustness upper bound

## Branch policy
Active:
`research/crossview-scale-rotation-domain-v1`

Base closeout commit:
`b769a8d6b62ae08696aef5d43e03ff6df4a735cf`

Do not continue feature work on:
`research/learned-geometric-frontend-ablation-v1`

unless explicitly reopening that closed experiment.
