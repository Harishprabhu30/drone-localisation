# RG2.2 — Reference-free multi-tile sub-tile projection generation

Status: **CODE PREPARED / LOCAL FREEZE PENDING**.

## Question

Can the frozen local map windows around each retrieval anchor independently produce continuous query-center map projections that can later be compared in a common EPSG:3346 coordinate frame?

RG2.2 does **not** ask which projection is correct and does not form a consensus.

## Frozen input set

Use the RG2.1-A full-overlap neighborhood exactly as frozen:

    development: 403 queries, 9315 query/tile pairs
    local-neighborhood SHA256:
    ee708fc261b2833f663fb51a4fc0fd88c926a1a1234ae4d841a93269e8faa100

    blind stress: 123 queries, 2955 query/tile pairs
    local-neighborhood SHA256:
    1d2fa9cabf7a69609bc94a32f8ff2c8c5282f76d47e0e231ccb8fc3d19ddd3b7

RG2.0 topology SHA256:

    1a9456cea5c951af30c63d5f1adaca3d82c3b8f1a275a841bf2d0a7d5fd71272

The in_immediate flag is preserved on every tested row so RG2.3 can compare the smaller immediate set against the wider full-overlap set without rerunning ORB after seeing reference.

## Geometry contract

The exact historical R4.11 / RG1 geometry implementation has been placed in src/uavloc/geometry/orb_subtile.py to satisfy the repository import contract. The executable RG2.2 script does not import another script.

    preprocess       clahe_luma
    resize_long      1024, downsize only
    ORB              1800 features
    fastThreshold    7
    edgeThreshold    15
    patchSize        31
    matcher          Hamming KNN
    Lowe             0.80
    RANSAC           5 px
    H validity       H exists and >=4 inliers
    H direction      query -> satellite tile
    query point      processed image centre
    map conversion   raster outer bounds + pixel-centre 0.5 offset
    map CRS          EPSG:3346

The prior development-derived inliers >= 6 trust hypothesis is **not used** in RG2.2. That would be a selection policy. RG2.2 freezes raw pair evidence.

## Frozen pair evidence

Every frozen full-overlap query/tile pair emits a row, whether geometry succeeds or fails. Evidence includes query/anchor/tile identity, immediate membership, relative tile offset, match/inlier metrics, verifier score, homography validity, processed dimensions, query center pixel, projected tile pixel, projected EPSG:3346 coordinate, finite_projection, projected_inside_tile, projective denominator, reprojection RMSE, and geometry error.

The homography matrix itself is not serialized because RG2.3 needs the frozen continuous projection/evidence, not replaying or modifying H. Reproduction is controlled by input hashes, geometry-module hash, and the frozen contract.

## Outputs

    outputs/research_runs/local_map_neighborhood_subtile_consensus_v1/rg2_2/
      development/
        rg2_2_multi_tile_projection_evidence.csv
        rg2_2_query_projection_summary.csv
      blind_stress/
        rg2_2_multi_tile_projection_evidence.csv
        rg2_2_query_projection_summary.csv
      rg2_2_freeze_report.json

The report excludes wall-clock runtime from its frozen bytes. Runtime is printed only to the console.

## Scientific boundary

    reference loaded             false
    SRT/GPS loaded               false
    oracle labels loaded         false
    retrieval reranked           false
    candidate selection          false
    inlier confidence gate       false
    consensus                    false
    bootstrap/state              false
    accuracy claim               false

homography_ok is only the historical minimum geometry-validity condition. It is not an accuracy/trust decision.

## Local execution

    git pull --ff-only origin research/local-map-neighborhood-subtile-consensus-v1
    source .drone_venv/bin/activate

    python -m unittest \
      tests.test_rg2_0_topology \
      tests.test_rg2_1a_freeze_neighborhoods \
      tests.test_rg2_1b_postfreeze_availability_oracle \
      tests.test_rg2_2_freeze_multi_tile_projections -v

    python scripts/villoc/geometry/rg2_2_freeze_multi_tile_projections.py \
      --config configs/research/local_map_neighborhood_rg2_2.yaml

The script self-bootstraps repo/src; PYTHONPATH is optional.

Expected total test count: **39**.

Expected freeze status:

    PASS_RG2_2_REFERENCE_FREE_MULTI_TILE_PROJECTION_FREEZE

## Stop gate

Return the full console result, especially for both trajectories: pair/query counts, unique tile count, homography-ok pair count, finite-projection pair count, inside-tile projection count, queries with any full-overlap homography, queries with immediate-neighborhood homography, queries with anchor homography, homography-count histogram, and both SHA256 outputs.

Do not attach development reference or implement projection consensus until this freeze has been inspected. RG2.3 will consume this evidence post-freeze.


## RG2.2 local freeze closeout

Status: **PASS / CLOSED** on user-reported local execution evidence.

Development: 403 queries, 9315 pairs, 358 unique tiles. Homography OK 9315/9315; finite projections 9315/9315; inside-tile projections 8407/9315 (90.25%). Runtime 537.565 s. Pair SHA256 `c75372f9db9670f7e339f886b8dadcb850512902559aa00f9b9b3c2dae3dc98e`; summary SHA256 `9ffe8ba9a30e53cf6afaa56f419f8672280d8b305120da90d6fa1d1fd6eb4041`.

Blind stress: 123 queries, 2955 pairs, 258 unique tiles. Homography OK 2955/2955; finite projections 2955/2955; inside-tile projections 2736/2955 (92.59%). Runtime 186.443 s. Pair SHA256 `d053d48dbd94184326e4b6bc42a5889536a34b66620ce033bd359cfea17a7b37`; summary SHA256 `559f61716122f27c1ed7045647cc24e220709f5713765e31abe0640b2a6f440a`.

Scientific boundary remained intact: reference not loaded; consensus not executed; no accuracy claim.

Interpretation: the historical minimum homography-validity condition is non-discriminative inside RG2.2 local neighborhoods because it passed every pair on both trajectories. `finite_projection` also passed every pair. Therefore RG2.3 must focus on cross-tile global-coordinate agreement, spread, clustering and robust consensus rather than homography availability. Outside-tile projections remain frozen evidence and must not be discarded before RG2.3.

Decision: RG2.2 is closed. RG2.3 is the next stage and must consume only the hash-verified frozen development evidence. Blind-stress evidence remains reference-free.
