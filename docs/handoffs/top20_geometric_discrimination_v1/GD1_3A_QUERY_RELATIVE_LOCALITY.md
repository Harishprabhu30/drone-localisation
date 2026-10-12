# GD1.3-A — Reference-free query-relative and physical-locality feature freeze

Status: **CODE PREPARED / LOCAL FREEZE PENDING**.

Purpose: reformulate the frozen GD1.0 Top20 evidence as a within-query ranking problem and as physical locality hypotheses, without rerunning ORB and without loading reference.

Inputs:
- frozen GD1.0 development evidence SHA256: 91ba3381e463bd083f9355b517dbb93953e4423a4b366638c1a97419ea0aea3f
- frozen GD1.0 blind evidence SHA256: 302bd878381f5befb26652a207f459e809f303b28f424c4fc0f784c75450cf74
- frozen RG2.0 map topology SHA256: 1a9456cea5c951af30c63d5f1adaca3d82c3b8f1a275a841bf2d0a7d5fd71272

Query-relative transforms are generated independently inside each query's Top20:
- robust z-score using median and IQR, with standard-deviation fallback;
- within-query percentile;
- explicit missing-value indicator.

Physical locality features are computed from Top20-induced connected components under the frozen RG2.0 nearby and overlaps graphs. Each candidate inherits component statistics such as:
- component size;
- best and mean retrieval rank;
- reciprocal-rank support;
- source-view diversity;
- max/median inliers;
- max good matches;
- median orientation consistency;
- median Jacobian anisotropy.

Scientific boundary:
- no reference / oracle labels;
- no reranking or selector;
- no threshold tuning;
- no ORB rerun;
- no learned frontend;
- no temporal fusion;
- no bootstrap/state change.

Run:

    git pull --ff-only origin research/top20-geometric-discrimination-v1
    source .drone_venv/bin/activate

    python -m unittest tests.test_gd1_3a_query_relative_locality -v

    python scripts/villoc/geometry/gd1_3a_freeze_query_relative_locality.py \
      --config configs/research/top20_geometric_discrimination_gd1_3a.yaml

Expected unit tests: 4.

Expected status:
    PASS_GD1_3A_REFERENCE_FREE_QUERY_RELATIVE_LOCALITY_FREEZE

Return both trajectories' row/query counts, component summaries, and SHA256s. Stop before reference attachment.
