# LF2.0-AH — Official LightGlue maximum-accuracy hard-pair smoke

Status: **CODE PREPARED / LOCAL SMOKE PENDING**.

Motivation: the official LightGlue guide recommends a distinct maximum-accuracy configuration for difficult pairs. The default LF2.0-A smoke therefore must not be used alone to judge LightGlue on UAV↔satellite matching.

Official maximum-accuracy changes relative to LF2.0-A:
- SuperPoint `max_num_keypoints=None` (all detected keypoints);
- LightGlue `depth_confidence=-1` (disable adaptive early stopping);
- LightGlue `width_confidence=-1` (disable point pruning).

Frozen unchanged:
- pinned official LightGlue commit;
- SuperPoint + LightGlue weights;
- resize-long 1024;
- LightGlue filter threshold 0.1;
- CPU device;
- homography RANSAC 5 px;
- minimum 4 homography inliers;
- exact same 32 smoke pairs;
- exact same QV1.4 Top20 candidates;
- no reference/GT/oracle.

The stage verifies the frozen LF2.0-A smoke report and prints direct default→maximum-accuracy comparisons for inside-tile projections, median matches, median inliers, and median pair runtime.

Run:

    git pull --ff-only origin research/learned-geometric-frontend-ablation-v1
    source .drone_venv/bin/activate

    python -m unittest tests.test_lf2_0ah_lightglue_max_accuracy_smoke -v

    python scripts/villoc/geometry/lf2_0ah_lightglue_max_accuracy_smoke.py \
      --config configs/research/learned_geometric_frontend_lf2_0ah.yaml

Expected tests: 2.

Expected status:
PASS_LF2_0AH_SUPERPOINT_LIGHTGLUE_MAX_ACCURACY_SMOKE

Decision after LF2.0-AH:
- if the official max-accuracy profile materially strengthens match/inlier support and/or inside-tile projection coverage, use this profile for the fixed uniform screening subset;
- if it gives negligible geometric improvement at much higher CPU cost, retain the default profile for screening;
- do not tune `filter_threshold` yet. Any threshold study must be a separate predeclared stage after a profile is selected.
