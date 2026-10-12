# LF2.1-A — ALIKED + LightGlue extractor-swap smoke

Status: **CODE PREPARED / LOCAL SMOKE PENDING**.

Purpose: determine whether the LF2.0-C failure is primarily caused by SuperPoint features or by the LightGlue sparse matching family itself.

Only the local feature extractor changes:

- SuperPoint -> ALIKED-N16;
- LightGlue remains the same pinned matcher and adaptive configuration;
- same resize-long 1024;
- same max keypoint budget 780;
- same filter threshold 0.1;
- same depth confidence 0.95;
- same width confidence 0.99;
- same 5 px homography RANSAC;
- same 4-inlier validity threshold;
- same 32 smoke pairs.

ALIKED contract:

- model: `aliked-n16`;
- detection threshold: 0.2;
- NMS radius: 2;
- pinned ALIKED source blob from the pinned LightGlue checkout;
- locally cached ALIKED and ALIKED-LightGlue weight SHA256s recorded in the report.

LF2.1-A verifies the frozen LF2.0-A SuperPoint baseline first and prints direct SuperPoint -> ALIKED deltas for inside projection, median matches, median inliers, and pair runtime.

No reference, GT, oracle, reranking, threshold tuning, training, temporal fusion, bootstrap, or state changes.

Run:

    git pull --ff-only origin research/learned-geometric-frontend-ablation-v1
    source .drone_venv/bin/activate

    python -m unittest tests.test_lf2_1a_aliked_lightglue_smoke -v

    python scripts/villoc/geometry/lf2_1a_aliked_lightglue_smoke.py \
      --config configs/research/learned_geometric_frontend_lf2_1a.yaml

Expected tests: 2.

Expected status:
`PASS_LF2_1A_ALIKED_LIGHTGLUE_EXTRACTOR_SWAP_SMOKE`

Decision gate:

- if ALIKED produces materially stronger match/inlier/projection support than SuperPoint, proceed to the same deterministic uniform40 screening protocol;
- if ALIKED is similarly weak or worse, close sparse LightGlue as a family and move to detector-free matching.
