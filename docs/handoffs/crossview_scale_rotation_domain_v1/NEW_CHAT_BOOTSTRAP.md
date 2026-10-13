# New-chat bootstrap prompt

Copy the text below into a new ChatGPT chat for this project.

---

We are continuing the existing `Harishprabhu30/drone-localisation` VILLOC research project. Do not treat this as a new localization project and do not reset the scientific context.

Active branch:
`research/crossview-scale-rotation-domain-v1`

Before proposing any code, use the GitHub connector and read these files in order:

1. `docs/handoffs/crossview_scale_rotation_domain_v1/README.md`
2. `docs/closeouts/learned_geometric_frontend_ablation_v1/README.md`
3. `docs/closeouts/learned_geometric_frontend_lf2_1c/README.md`
4. `docs/closeouts/top20_geometric_discrimination_gd1_3b/README.md`
5. `docs/handoffs/retrieval_candidate_pool_v2/README.md`
6. `docs/README_villoc_traj01_s8_full_pipeline_fusion_closeout.md`
7. `docs/demo/blind_recorded_flight_final_001/README.md`

Then summarize the current frozen state in your own words before changing the repo.

Critical constraints:
- The main problem is still reliable absolute UAV localization inside frozen QV1.4 Top20.
- QV1.4 retrieval stays frozen unless evidence reopens retrieval.
- ORB is the strongest current local geometric discriminator.
- XFeat sparse, XFeat*, SuperPoint+LightGlue and ALIKED+LightGlue have been closed under their documented gates.
- Do not resume generic learned sparse threshold tuning.
- Do not jump to deployment/networking/triangulation yet.
- The new branch tests the causal hypotheses around rotation, GSD/scale mismatch and aerial-domain training.
- AerialMegaDepth is a candidate aerial-specific verifier/fine-tuning resource, not a drop-in replacement for retrieval.
- Do not download the full AerialMegaDepth dataset before testing a pretrained aerial-specific model/upper bound.
- The first new stage must be CV1.0: a protocol freeze for scale/rotation diagnostics, including allowed metadata, blind/reference boundaries, exact subset, transforms, metrics and stop gates.
- Keep development truth detached until the reference-free evidence for a stage is frozen.
- Make small auditable GitHub commits.
- I will run local commands and paste outputs; do not fabricate execution.

When you have read the files, propose CV1.0 only. Do not implement CV2/SIFT, AerialMegaDepth, LoFTR, RoMa, deployment communications or architecture changes yet.

---
