# RG2.3-A — Reference-free projection consensus freeze

Status: **CODE PREPARED / LOCAL FREEZE PENDING**.

Purpose: transform the already frozen RG2.2 EPSG:3346 projection clouds into deterministic consensus coordinates without any reference trajectory.

## Frozen RG2.2 inputs

Development pair evidence SHA256:
    c75372f9db9670f7e339f886b8dadcb850512902559aa00f9b9b3c2dae3dc98e
Development query summary SHA256:
    9ffe8ba9a30e53cf6afaa56f419f8672280d8b305120da90d6fa1d1fd6eb4041

Blind-stress pair evidence SHA256:
    d053d48dbd94184326e4b6bc42a5889536a34b66620ce033bd359cfea17a7b37
Blind-stress query summary SHA256:
    559f61716122f27c1ed7045647cc24e220709f5713765e31abe0640b2a6f440a

## Frozen consensus contract

Subsets:
    immediate
    full_overlap

Projection views:
    all_finite      PRIMARY
    inside_only     PREDECLARED DIAGNOSTIC

Methods:
    coordinate_median
    geometric_median
    medoid
    radius_cluster_medoid
    inlier_huber_centroid

Cluster radius: 25.6 m = half the frozen nominal 51.2 m map stride. No reference tuning.
Huber k: 1.345.
Huber blind weight: raw ORB inlier count, clipped only to minimum positive 1; no inlier gate.

Anchor projection is carried as a baseline coordinate for every output row but is not recomputed.

Each row also freezes projection-cloud dispersion metrics: radial median/p90/max, pairwise median/p90/max, and bounding-box diagonal. The radius-cluster method records support count.

## Scientific boundary

    reference loaded                 false
    SRT/GPS loaded                   false
    oracle labels loaded             false
    ORB rerun                        false
    projection rerun                 false
    accuracy-based method selection  false
    accuracy evaluation              false

RG2.3-B may evaluate only these already-frozen coordinates on development reference.

## Local command

    git pull --ff-only origin research/local-map-neighborhood-subtile-consensus-v1
    source .drone_venv/bin/activate

    python -m unittest \
      tests.test_rg2_0_topology \
      tests.test_rg2_1a_freeze_neighborhoods \
      tests.test_rg2_1b_postfreeze_availability_oracle \
      tests.test_rg2_2_freeze_multi_tile_projections \
      tests.test_rg2_3a_projection_consensus -v

    python scripts/villoc/geometry/rg2_3a_freeze_projection_consensus.py \
      --config configs/research/local_map_neighborhood_rg2_3a.yaml

Expected test total: **46**.

Expected status:
    PASS_RG2_3A_REFERENCE_FREE_CONSENSUS_FREEZE

Return the complete output including development/blind consensus row counts, subset/view counts, and consensus SHA256s. Stop before RG2.3-B.
