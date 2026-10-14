# New-chat bootstrap prompt

Copy the text below into a new ChatGPT chat for this project.

---

We are continuing the existing \`Harishprabhu30/drone-localisation\` VILLOC
research project. Do not treat this as a new localization project and do not
reset the scientific context.

Active branch:
\`research/crossview-scale-rotation-domain-v1\`

Before proposing code, use the GitHub connector and read these files in order:

1. \`docs/handoffs/crossview_scale_rotation_domain_v1/README.md\`
2. \`docs/handoffs/crossview_scale_rotation_domain_v1/CV2_CLOSEOUT.md\`
3. \`docs/handoffs/crossview_scale_rotation_domain_v1/CV2_SIFT_LIGHTGLUE_DEEP_DIVE.md\`
4. \`docs/handoffs/crossview_scale_rotation_domain_v1/CV3_0_PROTOCOL.md\`
5. \`docs/closeouts/learned_geometric_frontend_ablation_v1/README.md\`
6. \`docs/closeouts/top20_geometric_discrimination_gd1_3b/README.md\`
7. \`docs/handoffs/retrieval_candidate_pool_v2/README.md\`
8. \`docs/README_villoc_traj01_s8_full_pipeline_fusion_closeout.md\`
9. \`docs/demo/blind_recorded_flight_final_001/README.md\`

Then summarize the current frozen state before changing the repo.

Critical current state:
- the main problem remains reliable absolute UAV localization inside frozen
  QV1.4 Top20;
- QV1.4 retrieval remains frozen;
- SIFT+LightGlue is the strongest development frontend;
- frozen full403 SIFT+LightGlue evidence SHA256:
  \`c0031242f0b359f1bfa3c87ea47ac428393ed3e8e03d7b5c6d0822e8822cb8a0\`;
- full403 SIFT+LightGlue:
  Top1 containment 349/403, Top1 <=40 205/403, Top1 <=80 346/403,
  selected <=40 367/403, selected <=80 368/403, unsafe tail 30/403;
- SIFT oracle projection <=40 is 386/403, <=80 is 392/403, unsafe tail 5/403;
- median SIFT Top1-Top2 inlier margin is 34;
- the residual bottleneck is increasingly wrong-locality selection/confidence,
  while SIFT also materially improves the candidate-level geometric ceiling;
- CV1 scale_m2 remains a frozen causal control; do not apply it to SIFT unless
  a future protocol explicitly reopens that question;
- do not tune SIFT/LightGlue further on full403 development truth;
- Jetson/CUDA migration is parked until cross-view accuracy-family selection is
  complete;
- CV3 is the current next stage: pretrained aerial-MASt3R upper bound using the
  official AerialMegaDepth checkpoint;
- do not download the full AerialMegaDepth training dataset;
- do not jump to deployment/networking/triangulation yet;
- keep development truth detached until each reference-free evidence stage is
  frozen;
- make small auditable GitHub commits;
- the user runs local commands and pastes outputs; never fabricate execution.

Immediate next action:
follow \`CV3_0_PROTOCOL.md\` and begin only with CV3.1 environment/model smoke.
Do not jump directly to uniform40/full403 Aerial-MASt3R and do not fine-tune.

---
