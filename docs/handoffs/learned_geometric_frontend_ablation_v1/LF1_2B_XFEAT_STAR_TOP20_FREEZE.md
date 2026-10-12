# LF1.2-B — Full reference-free XFeat* Top20 freeze

Status: **CODE PREPARED / LOCAL FREEZE PENDING**.

Purpose: run the pinned XFeat* semi-dense frontend over the exact frozen QV1.4 Top20 pools for development and blind stress before any reference is attached.

Workload:
- development: 403 × 20 = 8060 pairs;
- blind stress: 123 × 20 = 2460 pairs;
- total: 10520 pairs.

Frozen contract:
- XFeat commit e92685f57f8318b18725c5c8c0bd28c7fe188d9a;
- weights SHA256 0f5187fd7bedd26c7fe6acc9685444493a165a35ecc087b33c2db3627f3ea10b;
- resize-long 960;
- top_k 1200;
- dual-scale dense extraction;
- mutual coarse matching;
- subpixel refinement fine_confidence 0.25;
- homography RANSAC 5 px;
- minimum 4 homography inliers;
- same QV1.4 Top20 and 768_s256 map geometry.

Evidence frozen:
- matches/inliers/projection;
- spatial hull/grid support;
- forward/backward/symmetric reprojection diagnostics;
- homography condition/Jacobian;
- projective denominator/footprint stability.

Runtime is printed as an observation but not included in frozen evidence hashes.

Scientific boundary:
- no reference/GT/oracle;
- no reranking/selector;
- no threshold tuning;
- no model training;
- no temporal/bootstrap/state changes.

Run:

    git pull --ff-only origin research/learned-geometric-frontend-ablation-v1
    source .drone_venv/bin/activate

    python -m unittest tests.test_lf1_2b_xfeat_star_top20_freeze -v

    python scripts/villoc/geometry/lf1_2b_freeze_xfeat_star_top20.py \
      --config configs/research/learned_geometric_frontend_lf1_2b.yaml

Expected tests: 4.

Expected status:
PASS_LF1_2B_REFERENCE_FREE_XFEAT_STAR_TOP20_FREEZE

Return the full output including homography/finite/inside counts, median matches/inliers, observed runtime, and both evidence SHA256s.
