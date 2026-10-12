# LF1.1-B — Post-freeze ORB vs XFeat absolute comparison

Status: **CODE PREPARED / LOCAL AUDIT PENDING**.

Purpose: compare frozen ORB and frozen XFeat evidence on the exact same QV1.4 Top20 development candidate pairs after both frontends have been generated reference-free.

Inputs are pinned:
- ORB GD1.0 development evidence SHA256: 91ba3381e463bd083f9355b517dbb93953e4423a4b366638c1a97419ea0aea3f
- XFeat LF1.1-A development evidence SHA256: 73ed73dade7820072fb16b3df029be69d280dea18489f63cd66595e600ddf3b1
- XFeat blind evidence SHA256 is verified but blind evidence is not loaded in this audit: 1f764ea12064c8f9257fbbb9e6120a1051da8958f8d14575776526c39191d929

LF1.1-B asks four questions:
1. Does XFeat homography validity retain useful candidates while rejecting aliases?
2. Does XFeat rank useful Top20 candidates better than ORB using the same inlier-style selector?
3. Does the selected XFeat candidate yield better continuous sub-tile projection error?
4. For evidence available from both frontends, which frontend produces more discriminative geometry?

Rankings compared for each label:
- retrieval rank;
- raw inlier count;
- homography-valid first, then inlier count.

Labels:
- tile contains query reference point;
- tile-center error <=40 m;
- tile-center error <=80 m.

Projection diagnostics:
- selected candidate under valid-then-inliers;
- oracle best finite projection as a diagnostic upper bound;
- median / RMSE / p95;
- <=40 m / <=80 m counts;
- >200 m catastrophic tail count.

Common ORB↔XFeat evidence diagnostics include:
- inliers and inlier ratio;
- match count;
- hull/grid spatial support;
- symmetric reprojection error;
- homography conditioning;
- Jacobian anisotropy;
- projective denominator stability;
- projected footprint area.

Scientific boundary:
- both frontend freezes are verified before development reference is opened;
- no blind-stress evidence or reference is loaded;
- no threshold tuning;
- no trained selector;
- no frontend parameter changes;
- no frontend winner is promoted;
- no bootstrap/state changes.

Run:

    git pull --ff-only origin research/learned-geometric-frontend-ablation-v1
    source .drone_venv/bin/activate

    python -m unittest tests.test_lf1_1b_orb_xfeat_comparison -v

    python scripts/villoc/geometry/lf1_1b_postfreeze_orb_xfeat_comparison.py \
      --config configs/research/learned_geometric_frontend_lf1_1b.yaml

Expected tests: 6.

Expected status:
PASS_LF1_1B_POSTFREEZE_ORB_XFEAT_COMPARISON

Return the complete output. The decision gate after this stage is whether XFeat raises Top1/Top3/Top5 useful-candidate recovery and/or improves projection quality enough to justify deeper learned-frontend integration.
