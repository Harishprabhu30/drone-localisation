# LF1.2-C — Post-freeze ORB vs sparse XFeat vs XFeat* comparison

Status: **CODE PREPARED / LOCAL AUDIT PENDING**.

Purpose: compare all three frozen absolute-geometry frontends on the exact same QV1.4 Top20 development candidate pairs.

Frozen inputs:
- ORB GD1.0 evidence SHA256: 91ba3381e463bd083f9355b517dbb93953e4423a4b366638c1a97419ea0aea3f
- sparse XFeat LF1.1-A evidence SHA256: 73ed73dade7820072fb16b3df029be69d280dea18489f63cd66595e600ddf3b1
- XFeat* LF1.2-B development evidence SHA256: 07e8c6dfa674c0a94a0ddc566456eb880edc75cdcbcb13e4ebda6b5a3bdd6b87
- XFeat* LF1.2-B blind evidence SHA256 is verified but blind evidence is not loaded: 3571c5c8e7abaaee55631e91c18ff3fd9575fbfa01d23b0463c4d8d8b814a5c4

Comparison:
- retrieval-rank baseline;
- raw inlier ranking;
- homography-valid then inlier ranking;
- useful-pair retention / wrong-pair rejection;
- selected continuous projection median/RMSE/p95/<=40/<=80/>200;
- oracle best finite projection;
- common geometry feature AUC* and Top-k diagnostic ranking.

Scientific boundary:
- all three frontend freezes verified before development reference is opened;
- no blind-stress evidence/reference loaded;
- no parameter tuning;
- no trained selector;
- no frontend parameter changes;
- no winner promotion.

Run:

    git pull --ff-only origin research/learned-geometric-frontend-ablation-v1
    source .drone_venv/bin/activate

    python -m unittest tests.test_lf1_2c_three_frontend_comparison -v

    python scripts/villoc/geometry/lf1_2c_postfreeze_three_frontend_comparison.py \
      --config configs/research/learned_geometric_frontend_lf1_2c.yaml

Expected tests: 1.

Expected status:
PASS_LF1_2C_POSTFREEZE_THREE_FRONTEND_COMPARISON

Decision gate:
- if XFeat* materially closes or beats the ORB ranking/projection gap, keep XFeat* and proceed to external/blind generalization;
- if it remains clearly behind ORB, close XFeat and move to LightGlue as the next learned matcher family.
