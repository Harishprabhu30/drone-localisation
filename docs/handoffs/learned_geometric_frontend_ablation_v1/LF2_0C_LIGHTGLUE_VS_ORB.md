# LF2.0-C — Post-freeze LightGlue uniform40 vs ORB audit

Status: **CODE PREPARED / LOCAL AUDIT PENDING**.

Purpose: attach development reference only after LF2.0-B is frozen, then compare the exact same 40-query × Top20 subset against frozen ORB evidence.

## Frozen inputs

- LightGlue LF2.0-B evidence SHA256: `6896dc66c725f2505a7319a2f5f0b08f9e63f5e5e5eff539fc58aac5d51ada3b`
- ORB GD1.0 development evidence SHA256: `91ba3381e463bd083f9355b517dbb93953e4423a4b366638c1a97419ea0aea3f`
- exact selected query IDs are read from LF2.0-B's freeze report
- only those ORB rows are admitted into the comparison

## Primary comparison

Same selectors and tie-break logic:

- frozen retrieval rank
- raw inlier count
- homography-valid then inlier count

Targets:

- contains_query
- center <=40 m
- center <=80 m

Projection diagnostics:

- selected valid-then-inliers projection
- median / RMSE / p95
- <=40 / <=80
- >200 m catastrophic tail
- oracle best finite projection

## Predeclared full-run gate

A full 403-query LightGlue run is allowed only if all primary conditions pass:

1. LightGlue Top1 may trail ORB by at most 2 queries on **each** target.
2. LightGlue must improve at least one Top1 target by at least 2 queries.
3. LightGlue selected-projection <=40 count may trail ORB by at most 2 queries.
4. LightGlue >200 m selected-projection count may exceed ORB by at most 2 queries.

If raw inliers fail this gate, LightGlue-native scores are audited secondarily:

- match_score_sum
- match_score_mean
- match_score_median
- inlier_ratio

A secondary score does **not** get promoted here. It can only trigger a new frozen formulation stage if it meets the separate secondary gate.

Possible decisions:

- `PROCEED_FULL_LIGHTGLUE`
- `PROCEED_LIGHTGLUE_SCORE_FORMULATION_STAGE`
- `CLOSE_SUPERPOINT_LIGHTGLUE`

## Scientific boundary

- development reference only after LF2.0-B freeze verification
- no blind-stress loading
- no threshold tuning
- no trained selector
- no frontend parameter changes
- no full 403-query LightGlue evidence loaded
- no bootstrap/state changes

## Run

```bash
git pull --ff-only origin research/learned-geometric-frontend-ablation-v1
source .drone_venv/bin/activate

python -m unittest tests.test_lf2_0c_lightglue_vs_orb -v

python scripts/villoc/geometry/lf2_0c_postfreeze_lightglue_uniform40_vs_orb.py \
  --config configs/research/learned_geometric_frontend_lf2_0c.yaml
```

Expected tests: 4.

Expected status:

`PASS_LF2_0C_POSTFREEZE_LIGHTGLUE_UNIFORM40_VS_ORB_AUDIT`

Return the complete output, especially the final decision and all primary/secondary Top1 numbers.
