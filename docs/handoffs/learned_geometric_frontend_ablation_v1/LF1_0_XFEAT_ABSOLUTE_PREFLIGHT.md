# LF1.0 — XFeat absolute frontend preflight/smoke

Status: **CODE PREPARED / LOCAL SMOKE PENDING**.

Branch:
research/learned-geometric-frontend-ablation-v1

Base:
GD1.3-B closeout commit 83c7e4f78aaf47e57a73401c03d2dff71cd24039

Purpose: verify the already-pinned official VERLab XFeat implementation on the absolute UAV↔satellite matching problem before running the full frozen Top20 benchmark.

Historical XFeat contract reused:
- official VERLab accelerated_features repository;
- pinned commit e92685f57f8318b18725c5c8c0bd28c7fe188d9a;
- resize long side 960;
- top_k 1200;
- detection threshold 0.05;
- MNN cosine threshold 0.82.

For this absolute-localization experiment the old relative-motion affine model is replaced by the same query→satellite homography semantics used by ORB GD1.0:
- RANSAC homography threshold 5 px;
- minimum homography inliers 4;
- query image center projected into satellite tile;
- raster outer-edge pixel-center map conversion.

Smoke sample:
- development queries: 1, 101, 201, 301, 403;
- blind-stress queries: 1, 61, 123;
- frozen pool ranks: 1, 5, 10, 20;
- total pairs: 32.

The smoke records:
- exact XFeat commit;
- weights SHA256;
- device;
- feature counts;
- match counts;
- homography inliers;
- finite/inside projections;
- feature/match/geometry/total runtime.

Scientific boundary:
- no reference / GT / oracle labels;
- no full Top20 benchmark;
- no reranking/selector;
- no threshold tuning;
- no model training;
- no bootstrap/state changes.

Run:

    git fetch origin
    git checkout research/learned-geometric-frontend-ablation-v1
    git pull --ff-only origin research/learned-geometric-frontend-ablation-v1
    source .drone_venv/bin/activate

    python -m unittest tests.test_lf1_0_xfeat_absolute_preflight -v

    python scripts/villoc/geometry/lf1_0_xfeat_absolute_preflight_smoke.py \
      --config configs/research/learned_geometric_frontend_lf1_0.yaml

Expected tests: 4.

Expected status:
PASS_LF1_0_XFEAT_ABSOLUTE_FRONTEND_PREFLIGHT_SMOKE

Return the complete output including XFeat commit, weight hash, device, per-trajectory homography/finite/inside counts, median matches/inliers/runtime, and smoke SHA256.
