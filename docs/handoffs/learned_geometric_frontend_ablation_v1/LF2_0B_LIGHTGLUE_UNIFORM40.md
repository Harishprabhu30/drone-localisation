# LF2.0-B — Reference-free LightGlue uniform40 Top20 screening freeze

Status: **CODE PREPARED / LOCAL SCREENING PENDING**.

Purpose: screen the default adaptive SuperPoint+LightGlue frontend on a deterministic development subset before committing to the full 403-query × Top20 workload.

## Why the default profile

LF2.0-AH tested the official maximum-accuracy configuration (`max_num_keypoints=None`, `depth_confidence=-1`, `width_confidence=-1`) on the same 32 smoke pairs. In this UAV↔satellite domain it was worse than the default adaptive profile and much more expensive on CPU:

- development homography: 20/20 default → 14/20 hard profile;
- development inside projection: 14/20 → 11/20;
- development median matches: 32.5 → 7.0;
- development median inliers: 5.0 → 4.0;
- development median pair runtime: 1995.750 ms → 10301.919 ms;
- blind homography: 12/12 → 9/12;
- blind inside projection: 8/12 → 7/12;
- blind median matches: 36.5 → 10.5;
- blind median inliers: 6.0 → 5.0;
- blind median pair runtime: 2043.937 ms → 8525.530 ms.

Therefore LF2.0-B returns to the frozen default adaptive profile. This does not claim the upstream maximum-accuracy guidance is wrong; it records that the profile is not beneficial for this specific cross-view CPU workload.

## Screening subset

- full development manifest: 403 sorted query IDs;
- select 40 queries by endpoint-inclusive uniform indices using `round(linspace(0, 402, 40))`;
- use all 20 frozen QV1.4 candidates for every selected query;
- total 800 candidate pairs;
- selected indices and query IDs are frozen into the report.

No reference/GT/oracle is loaded during LF2.0-B.

## Frozen frontend

- official cvg/LightGlue pinned commit `eb42fee2d71449efb0aa5c10549752b5d75384d8`;
- SuperPoint max keypoints 780;
- resize-long 1024;
- LightGlue filter threshold 0.1;
- depth confidence 0.95;
- width confidence 0.99;
- CPU;
- homography RANSAC 5 px;
- minimum 4 homography inliers.

## Run

    git pull --ff-only origin research/learned-geometric-frontend-ablation-v1
    source .drone_venv/bin/activate

    python -m unittest tests.test_lf2_0b_lightglue_uniform40 -v

    python scripts/villoc/geometry/lf2_0b_freeze_lightglue_uniform40.py \
      --config configs/research/learned_geometric_frontend_lf2_0b.yaml

Expected tests: 3.

Expected status:
`PASS_LF2_0B_REFERENCE_FREE_LIGHTGLUE_UNIFORM40_SCREENING_FREEZE`

Return the complete output including the selected query IDs, homography/finite/inside counts, median matches/inliers, evidence SHA256 and observed runtime.

LF2.0-C will attach development reference only after this freeze and compare the exact same 800 pairs against frozen ORB evidence before deciding whether a full 403-query LightGlue run is justified.
