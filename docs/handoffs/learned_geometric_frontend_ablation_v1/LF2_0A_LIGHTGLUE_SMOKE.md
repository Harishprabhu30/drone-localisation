# LF2.0-A — SuperPoint + LightGlue absolute preflight/smoke

Status: **CODE PREPARED / LOCAL SMOKE PENDING**.

Purpose: begin the next learned matcher family after sparse XFeat and XFeat* were both rejected by LF1.2-C.

Frozen LightGlue contract:
- official repository: cvg/LightGlue;
- pinned commit: eb42fee2d71449efb0aa5c10549752b5d75384d8;
- feature extractor: SuperPoint;
- max keypoints: 780;
- resize-long: 1024;
- LightGlue filter threshold: 0.1;
- depth confidence: 0.95;
- width confidence: 0.99;
- CPU;
- homography RANSAC threshold: 5 px;
- minimum homography inliers: 4.

The stage records the resolved repository commit plus SHA256 hashes of the locally cached SuperPoint and LightGlue weights.

Smoke set is identical to prior learned frontend preflights:
- development q=[1,101,201,301,403];
- blind q=[1,61,123];
- pool ranks [1,5,10,20];
- 32 total pairs.

Evidence includes:
- match count and LightGlue match-score statistics;
- homography/inlier count and ratio;
- query/tile spatial support;
- forward/backward/symmetric reprojection errors;
- homography condition/Jacobian;
- projective denominator and footprint stability;
- continuous query-center map projection;
- feature/match/geometry/total runtime.

Scientific boundary:
- no reference/GT/oracle;
- no full Top20 benchmark;
- no reranking/selector;
- no threshold tuning;
- no model training;
- no bootstrap/state changes.

Run:

    git pull --ff-only origin research/learned-geometric-frontend-ablation-v1
    source .drone_venv/bin/activate

    python -m unittest tests.test_lf2_0a_lightglue_smoke -v

    python scripts/villoc/geometry/lf2_0a_lightglue_absolute_preflight_smoke.py \
      --config configs/research/learned_geometric_frontend_lf2_0a.yaml

Expected tests: 2.

Expected status:
PASS_LF2_0A_SUPERPOINT_LIGHTGLUE_ABSOLUTE_PREFLIGHT_SMOKE

Decision gate:
- if LightGlue produces viable homographies and useful correspondence support at acceptable smoke cost, proceed to the full reference-free Top20 freeze;
- if the local pinned checkout differs, fail closed and resolve the repository version before running.
