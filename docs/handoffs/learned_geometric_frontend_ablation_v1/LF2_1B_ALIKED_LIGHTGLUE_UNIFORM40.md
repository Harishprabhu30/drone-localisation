# LF2.1-B — Reference-free ALIKED+LightGlue uniform40 screening

Status: **CODE PREPARED / LOCAL SCREENING PENDING**.

Purpose: determine whether the ALIKED extractor improvement seen in LF2.1-A survives a broader, directly paired candidate-discrimination screen.

Subset contract:
- reuse exactly the 40 frozen query IDs from LF2.0-B;
- independently recompute the endpoint-inclusive uniform40 rule and require exact equality;
- all 20 QV1.4 candidates for each query;
- total 800 pairs;
- no reference/GT/oracle.

Frozen frontend:
- ALIKED-N16;
- max keypoints 780;
- detection threshold 0.2;
- NMS radius 2;
- resize-long 1024;
- LightGlue filter threshold 0.1;
- depth confidence 0.95;
- width confidence 0.99;
- CPU;
- homography RANSAC 5 px;
- minimum 4 inliers.

Provenance checks:
- LF2.1-A smoke evidence SHA256 `28e77284895c19f0e666a9c699431d4b91d0517cbdfa638c5e3eff470d99de9b`;
- LF2.0-B SuperPoint screening evidence SHA256 `6896dc66c725f2505a7319a2f5f0b08f9e63f5e5e5eff539fc58aac5d51ada3b`;
- same exact query ID list as LF2.0-B;
- ALIKED and ALIKED-LightGlue weight hashes reverified.

Run:

    git pull --ff-only origin research/learned-geometric-frontend-ablation-v1
    source .drone_venv/bin/activate

    python -m unittest tests.test_lf2_1b_aliked_lightglue_uniform40 -v

    python scripts/villoc/geometry/lf2_1b_freeze_aliked_lightglue_uniform40.py \
      --config configs/research/learned_geometric_frontend_lf2_1b.yaml

Expected tests: 3.

Expected status:
`PASS_LF2_1B_REFERENCE_FREE_ALIKED_LIGHTGLUE_UNIFORM40_SCREENING_FREEZE`

Return the complete output including homography/finite/inside counts, median matches/inliers, evidence SHA256, and observed runtime.

LF2.1-C will then attach development reference on this exact same subset and compare ALIKED+LightGlue directly with both ORB and SuperPoint+LightGlue.