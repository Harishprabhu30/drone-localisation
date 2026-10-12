# RG2.4-A — Reference-free locality persistence / evidence-independence

Status: **CODE PREPARED / LOCAL FREEZE PENDING**.

Purpose: measure how often consecutive absolute/locality observations represent
new spatial evidence versus repeated support for the same physical locality.

Inputs are the frozen RG2.3-A all-finite consensus tracks plus frozen RG2.2
supporting tile-window identities. RG2.3-B accuracy outputs are explicitly
forbidden.

Tracks are generated for all five frozen consensus methods and both immediate
and full-overlap subsets. No method is selected from development accuracy.

Predeclared locality-link thresholds:
- 12.8 m = quarter of frozen 51.2 m map stride
- 25.6 m = half of frozen 51.2 m map stride

These match the historical R4.18 blind acquisition/tracking scales. Temporal
continuity uses the RG1.5 rule: at most 1.5 × nominal sampling interval.

A locality run ends whenever either:
- the time gap is non-contiguous; or
- successive consensus coordinates differ by more than the frozen link threshold.

Each run reports duration, query count, path length, maximum spatial span,
start/end displacement, anchor-tile diversity/change count, distinct supporting
overlap windows, total support-window observations, support redundancy ratio,
and projection-cloud dispersion summaries.

Support redundancy ratio:
    1 - distinct_support_windows / support_window_observations

This is a diagnostic measure only; locality IDs are not truth labels.

Scientific boundary:
- no reference/SRT/GPS/oracle labels
- no RG2.3-B accuracy inputs
- no method selection or threshold tuning
- no ORB/projection/consensus rerun
- no bootstrap/state modification
- no relative-frontend fusion in RG2.4-A

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
      tests.test_rg2_4a_locality_persistence -v

    python scripts/villoc/geometry/rg2_4a_freeze_locality_persistence.py \
      --config configs/research/local_map_neighborhood_rg2_4a.yaml

Expected test total: **59**.

Expected stage status:
    PASS_RG2_4A_REFERENCE_FREE_LOCALITY_PERSISTENCE_FREEZE

Stop after the freeze and inspect per-track locality counts, multi-query
fractions, median/max run lengths and hashes before adding relative-motion
support or modifying bootstrap/state.
