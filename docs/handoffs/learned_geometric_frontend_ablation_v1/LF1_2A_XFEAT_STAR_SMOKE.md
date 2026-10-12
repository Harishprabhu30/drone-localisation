# LF1.2-A — XFeat* semi-dense absolute preflight/smoke

Status: **CODE PREPARED / LOCAL SMOKE PENDING**.

Purpose: test the pinned XFeat implementation's distinct semi-dense XFeat* matching mode after sparse XFeat failed the LF1.1-B ORB comparison.

Only the correspondence regime changes. Frozen candidate pool, query/map images, resize, top_k, homography and map conversion remain unchanged.

XFeat* contract:
- pinned commit e92685f57f8318b18725c5c8c0bd28c7fe188d9a;
- resize-long 960;
- top_k 1200;
- dual-scale dense extraction enabled;
- batch mutual matching;
- subpixel refinement with fine confidence 0.25;
- homography RANSAC 5 px;
- minimum 4 homography inliers.

Smoke set is identical to LF1.0:
- development q=[1,101,201,301,403];
- blind q=[1,61,123];
- pool ranks [1,5,10,20];
- 32 total pairs.

Decision gate:
- if XFeat* clearly increases match/inlier support and stable homography coverage, proceed to a full reference-free Top20 freeze;
- if it remains weak/fragile or prohibitively expensive even on smoke, stop XFeat and move to LightGlue.

Run:

    git pull --ff-only origin research/learned-geometric-frontend-ablation-v1
    source .drone_venv/bin/activate

    python -m unittest tests.test_lf1_2a_xfeat_star_smoke -v

    python scripts/villoc/geometry/lf1_2a_xfeat_star_preflight_smoke.py \
      --config configs/research/learned_geometric_frontend_lf1_2a.yaml

Expected tests: 2.

Expected status:
PASS_LF1_2A_XFEAT_STAR_ABSOLUTE_PREFLIGHT_SMOKE
