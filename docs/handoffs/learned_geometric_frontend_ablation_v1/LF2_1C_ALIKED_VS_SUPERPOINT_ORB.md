# LF2.1-C — Post-freeze ALIKED+LightGlue vs SuperPoint+LightGlue vs ORB audit

Status: **CODE PREPARED / LOCAL AUDIT PENDING**.

Purpose: determine whether the ALIKED extractor improvement survives development truth attachment strongly enough to justify a full 403-query learned-frontend run.

Inputs are frozen before reference:
- LF2.1-B ALIKED+LightGlue uniform40 evidence SHA256 `d87cab4cf37b3f4f51aa96846e0aef7a475077f0ec80f62c4d1dcabd470cec7d`;
- LF2.0-B SuperPoint+LightGlue uniform40 evidence SHA256 `6896dc66c725f2505a7319a2f5f0b08f9e63f5e5e5eff539fc58aac5d51ada3b`;
- ORB GD1.0 development evidence SHA256 `91ba3381e463bd083f9355b517dbb93953e4423a4b366638c1a97419ea0aea3f`;
- exact same 40 query IDs and exact same 800 candidate rows required for all three.

Primary ranking:
- raw inlier count for ORB, SuperPoint+LightGlue and ALIKED+LightGlue;
- Top1/Top3/Top5/Top10 for contains_query, center<=40 m and center<=80 m;
- selected continuous projection metrics.

The full-run gate against ORB is intentionally identical to LF2.0-C:
- ALIKED may trail ORB by at most 2 Top1 queries on every target;
- ALIKED must improve at least one Top1 target by >=2 queries;
- selected projection <=40 may trail by at most 2;
- >200 m selected projection failures may increase by at most 2.

ALIKED-native score diagnostics are secondary only:
- match_score_sum;
- match_score_mean;
- match_score_median;
- inlier_ratio.

Possible decisions:
- `PROCEED_FULL_ALIKED_LIGHTGLUE`;
- `PROCEED_ALIKED_LIGHTGLUE_SCORE_FORMULATION_STAGE`;
- `CLOSE_SPARSE_LIGHTGLUE_FAMILY`.

Run:

    git pull --ff-only origin research/learned-geometric-frontend-ablation-v1
    source .drone_venv/bin/activate

    python -m unittest tests.test_lf2_1c_aliked_vs_superpoint_orb -v

    python scripts/villoc/geometry/lf2_1c_postfreeze_aliked_vs_superpoint_orb.py \
      --config configs/research/learned_geometric_frontend_lf2_1c.yaml

Expected tests: 4.

Expected status:
`PASS_LF2_1C_POSTFREEZE_ALIKED_LIGHTGLUE_VS_SUPERPOINT_AND_ORB_AUDIT`

Return the complete output, especially the final decision, the three Top1 comparisons, projection summaries and ALIKED-native score diagnostics.