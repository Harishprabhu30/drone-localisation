# GD1.3-B — OOF query-relative and locality selector audit

Status: **CODE PREPARED / LOCAL AUDIT PENDING**.

Purpose: test whether within-query normalization and Top20 physical-locality support materially outperform the frozen raw-inlier baseline under the same 5-fold contiguous-query OOF protocol used in GD1.2.

Frozen GD1.3-A hashes:
- development: 9e4778223dd3ae7498b44faf95cf9dc5e8b20decfb700c5957ea4569fcc7c840
- blind stress: 24bedf575a4d6624e01dbff2cdf8b6571e7ca273beb40070559fda63481810fb

Model families:
- retrieval rank only
- inliers only
- inliers + within-query z/percentile
- all query-relative geometry
- nearby locality only
- overlap locality only
- query-relative + nearby locality
- query-relative + overlap locality
- query-relative + both localities
- query-relative + both localities + retrieval rank

No hyperparameter tuning, no frontend change, no candidate-pool change, no blind labels, and no deployable model export are allowed.

Decision gate:
If query-relative/locality models do not materially outperform the raw-inlier Top1 baselines (containment 296/403, <=80 288/403, <=40 158/403), stop further ORB-side selector formulation and activate the learned-frontend adapter benchmark.

Run:

    git pull --ff-only origin research/top20-geometric-discrimination-v1
    source .drone_venv/bin/activate

    python -m unittest tests.test_gd1_3b_oof_query_relative_locality -v

    python scripts/villoc/geometry/gd1_3b_oof_query_relative_locality_audit.py \
      --config configs/research/top20_geometric_discrimination_gd1_3b.yaml

Expected tests: 6.

Expected status:
    PASS_GD1_3B_OOF_QUERY_RELATIVE_LOCALITY_AUDIT

Return all model-family Top1/Top3/Top5/Top10 results for contains_query, center_le40 and center_le80, plus the OOF score SHA256.
