# LF1.1-A — Full reference-free XFeat Top20 freeze

Status: **CODE PREPARED / LOCAL FREEZE PENDING**.

Purpose: run the pinned XFeat absolute frontend over the exact frozen QV1.4 Top20 pools for development and blind stress, preserving the same query-center homography projection semantics used by ORB.

Frozen workload:
- development: 403 queries × 20 = 8060 pairs;
- blind stress: 123 queries × 20 = 2460 pairs;
- total: 10520 candidate pairs.

XFeat contract:
- official VERLab checkout;
- commit e92685f57f8318b18725c5c8c0bd28c7fe188d9a;
- weights SHA256 0f5187fd7bedd26c7fe6acc9685444493a165a35ecc087b33c2db3627f3ea10b;
- CPU;
- resize-long 960;
- top_k 1200;
- detection threshold 0.05;
- MNN cosine threshold 0.82;
- homography RANSAC 5 px;
- minimum homography inliers 4.

Frozen evidence includes sparse match/inlier counts plus frontend-independent spatial support, forward/backward/symmetric reprojection diagnostics, homography conditioning/Jacobian and projective-footprint stability.

Runtime is observed and printed but deliberately excluded from frozen evidence hashes.

Scientific boundary:
- no reference/GT/oracle;
- no candidate reranking/selection;
- no threshold tuning;
- no model training;
- no temporal/bootstrap/state changes.

Run:

    git pull --ff-only origin research/learned-geometric-frontend-ablation-v1
    source .drone_venv/bin/activate

    python -m unittest tests.test_lf1_1a_xfeat_top20_freeze -v

    python scripts/villoc/geometry/lf1_1a_freeze_xfeat_top20.py \
      --config configs/research/learned_geometric_frontend_lf1_1a.yaml

Expected tests: 4.

Expected status:
PASS_LF1_1A_REFERENCE_FREE_XFEAT_TOP20_FREEZE

Return the complete status output including homography/finite/inside counts, query-level homography coverage, median matches/inliers, observed runtime, and both evidence SHA256s.
