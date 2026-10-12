# RG2.4-A closeout and RG2.4-B transition

RG2.4-A reference-free locality persistence freeze passed.

Development:
- 403 queries
- 4030 primary all-finite consensus rows
- 8060 transition rows
- 4657 locality runs
- 20 tracks
- transitions SHA256: 4b77653ee6626789a5a1a4be81eff40e5820cbb8062aefcfa58bb3e88886e0ba
- runs SHA256: 75174aa46a60ac716283e2ef5d0a77911bfff43bb7d03417ef006ab12bec5601

Blind stress:
- 123 queries
- 1230 primary all-finite consensus rows
- 2460 transition rows
- 1886 locality runs
- 20 tracks
- transitions SHA256: 3ec55cc852ef878a4dcac4efa312c344c2daa193535fd14d71f6101a5fced4d6
- runs SHA256: 6461e747925a8cdcf374b6fdf3a7de5382c94c3b989a8ce5313f8db5f38e59e2

Reference was not loaded. RG2.3-B accuracy was not loaded. Bootstrap/state were unchanged.

Key blind interpretation:
- half-stride (25.6 m) links produce substantially more multi-query persistence than quarter-stride (12.8 m);
- full-overlap inlier-Huber half-stride covers 82.4% of development queries and 78.9% of blind-stress queries in multi-query localities;
- full-overlap geometric-median half-stride covers 75.7% development and 66.7% blind;
- median locality run remains one query for every track, so isolated observations coexist with long repeated localities;
- maximum runs reach 36--46 queries on development and 13 queries on blind stress.

These results prove that frame-level absolute observations are often correlated repeated support. They do not prove correctness: a stable visual alias can also persist.

RG2.4-B therefore attaches development reference only after verifying the RG2.4-A hashes and asks whether blind-available persistence features actually predict correctness.

RG2.4-B evaluates all 20 frozen tracks; it does not select a consensus method or tune a threshold.

Primary questions:
- Are queries inside runs of length >=2, >=3, >=5 or >=10 more accurate?
- How many long runs remain wrong (>80 m median error)?
- How long can a persistent wrong locality survive?
- Do duration, path/span, support-window diversity/redundancy and projection dispersion correlate with run error?
- Is persistence useful as confidence, or can it reinforce stable aliases?

Scientific boundary:
- development reference only, post-freeze;
- blind-stress tables are not loaded;
- no threshold tuning;
- no method selection;
- no relative-frontend fusion;
- no bootstrap/state changes.

Run:

    git pull --ff-only origin research/local-map-neighborhood-subtile-consensus-v1
    source .drone_venv/bin/activate

    python -m unittest \
      tests.test_rg2_0_topology \
      tests.test_rg2_1a_freeze_neighborhoods \
      tests.test_rg2_1b_postfreeze_availability_oracle \
      tests.test_rg2_2_freeze_multi_tile_projections \
      tests.test_rg2_3a_projection_consensus \
      tests.test_rg2_3b_postfreeze_consensus_accuracy \
      tests.test_rg2_4a_locality_persistence \
      tests.test_rg2_4b_persistence_correctness -v

    python scripts/villoc/geometry/rg2_4b_postfreeze_persistence_correctness.py \
      --config configs/research/local_map_neighborhood_rg2_4b.yaml

Expected test total: 65.

Expected status:
    PASS_RG2_4B_POSTFREEZE_PERSISTENCE_CORRECTNESS_AUDIT

Return the complete output, particularly persistent_wrong_runs, longest_wrong,
and the run>=3 accuracy line for every track. Do not modify bootstrap/state yet.
