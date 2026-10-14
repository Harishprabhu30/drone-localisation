# Context index — crossview scale / rotation / domain v1

Use this file as the navigation map for a new session.

## Start here
1. \`README.md\` — canonical current state.
2. \`NEW_CHAT_BOOTSTRAP.md\` — current new-chat prompt.
3. \`CV2_CLOSEOUT.md\` — frozen SIFT+LightGlue promotion result.
4. \`CV2_SIFT_LIGHTGLUE_DEEP_DIVE.md\` — Top-k/oracle/error-tail interpretation.
5. \`CV3_0_PROTOCOL.md\` — authoritative next-stage contract.

## Completed branch stages
- \`CV1_CLOSEOUT.md\` — scale/rotation causal study.
- \`CV2_CLOSEOUT.md\` — SIFT+LightGlue promotion.
- \`CV2_SIFT_LIGHTGLUE_DEEP_DIVE.md\` — future-facing SIFT evidence record.

## Previous closeouts
- \`../../closeouts/learned_geometric_frontend_ablation_v1/README.md\`
- \`../../closeouts/learned_geometric_frontend_lf2_1c/README.md\`
- \`../../closeouts/top20_geometric_discrimination_gd1_3b/README.md\`

## Retrieval / candidate-pool context
- \`../retrieval_candidate_pool_v2/README.md\`

## Main VILLOC pipeline context
- \`../../README_villoc_traj01_s8_full_pipeline_fusion_closeout.md\`
- \`../../demo/blind_recorded_flight_final_001/README.md\`

## Reporting
- \`../../../configs/reporting/villoc_method_benchmark_registry.yaml\`
- Do not merge n=40 screening results into n=403 tables without an explicit
  denominator-aware reporting stage.

## External research queue
- AerialMegaDepth project: https://aerial-megadepth.github.io/
- AerialMegaDepth paper: https://arxiv.org/abs/2504.13157
- SIFT+LightGlue: frozen current development frontend; do not tune on full403
  truth
- RDD / aerial-domain features: after the CV3 aerial-domain upper bound
- EfficientLoFTR / LoFTR: detector-free candidate family
- RoMa: dense robustness upper bound

## Branch policy
Active:
\`research/crossview-scale-rotation-domain-v1\`

Base closeout commit:
\`b769a8d6b62ae08696aef5d43e03ff6df4a735cf\`

Do not continue feature work on:
\`research/learned-geometric-frontend-ablation-v1\`

unless explicitly reopening that closed experiment.
