# Continuation prompt — begin RG2 exactly from RG1 closeout

Use this prompt in a new ChatGPT chat inside the same SLAM project.

---

We are continuing the `Harishprabhu30/drone-localisation` VILLOC research
chain. Do not restart from generic SLAM/localization advice and do not redesign
the pipeline from memory alone.

## First recover repository context

Use the GitHub connector and inspect the current repository before changing
anything.

The previous branch is:

```text
research/region-candidate-geometry-v1
```

Its final closeout is:

```text
docs/closeouts/region_candidate_geometry_v1/README.md
```

The next-stage handoff is:

```text
docs/handoffs/local_map_neighborhood_subtile_consensus_v1/README.md
```

Also read, in order:

```text
docs/closeouts/query_view_candidate_generation_v1/README.md
docs/closeouts/trajectory_adapter_v1/README.md
docs/handoffs/region_candidate_geometry_v1/README.md
README.md
docs/demo/blind_recorded_flight_final_001/README.md
configs/bootstrap/minimum_confident_v2/architecture_contract.json
```

For the historical sub-tile/bootstrap implementation context, inspect:

```text
scripts/villoc/research/minimum_confident_bootstrap/diagnostics/
  r4_11_blind_subtile_projection_recompute.py
  r4_18_acquisition_to_tracking_counterfactual.py
```

Do not assume chat memory is sufficient; the committed closeouts/handoffs are
the source of truth.

## Research lineage to preserve

The chain is:

```text
retrieval / absolute frontend diagnostics
  -> QV1 query-view candidate generation
  -> Trajectory Adapter v1
  -> RG1 region-candidate geometry
  -> NOW: RG2 local map neighborhood / sub-tile consensus
```

QV1.4 is frozen:

```text
center_unique_allview_fill20
768_s256
fixed Top20
center Top1 preserved
```

Development trajectory:

```text
villoc_traj01_90deg_stable120m
403 queries
```

Blind-stress trajectory:

```text
villoc_blind_recorded_flight_final_001
123 queries
no GT/reference
```

Trajectory Adapter v1 is frozen and should be used rather than adding
dataset-specific paths to new research code.

## Important RG1 conclusions

RG1 processed all 8060 development query-candidate pairs.

Geometry evidence:

```text
inlier AUC*:
  contain ~0.856
  <=80   ~0.861
  <=40   ~0.856
```

For valid containing candidates:

```text
tile-center error median      52.72 m
projected-point error median   6.50 m
projection improves           79.5%
```

Do NOT promote geometry as a hard Top20 pruning/reranking stage.

Geometry is best supported as:

```text
retrieval-anchor trust / abstain
+
continuous sub-tile refinement
```

A provisional development-derived trust hypothesis is:

```text
ORB inliers >= 6
```

It is NOT independently accuracy-validated.

RG1.5 showed that the blind trajectory has much worse temporal projection
continuity:

```text
development contiguous projected jump median ~13.25 m
blind contiguous projected jump median       ~62.80 m

development changed-tile center jump median ~184.60 m
blind changed-tile center jump median       ~323.82 m
```

Therefore local-neighborhood work must not be presented as a solution to
distant global retrieval aliases.

## Why RG2 starts now

Earlier bootstrap research already identified that:

```text
correct consecutive observations can collapse to the same tile center,
reducing useful motion/map-span information,

while wrong neighboring tile choices can create artificial diversity.
```

The bootstrap architecture already records:

```text
tile_center_as_primary_measurement = false
minimum_map_span_m = 50
maturity based on consecutive compatible sub-tile innovations
```

RG2 should build the missing locality representation cleanly before bootstrap
is revisited.

## RG2 research question

Can overlapping map tiles that describe the same physical locality be converted
into one tile-boundary-invariant continuous map-position hypothesis, and can
repeated consecutive observations be represented as support for one locality
rather than falsely independent tile-center observations?

## Frozen for RG2 initially

Do NOT change:

```text
768_s256 map footprint
51.2 m stride
QV1.4 candidate generation
DINO representations
ORB frontend
trajectory adapter
blind/reference boundary
```

Do NOT begin with a stride sweep.

Stride becomes a later controlled ablation only after locality/sub-tile
consensus is understood.

## Begin with RG2.0 only

Create the new branch from the final RG1 closeout commit:

```text
research/local-map-neighborhood-subtile-consensus-v1
```

RG2.0 is a neighborhood/topology contract and preflight.

Use the actual 768_s256 tile index geometry to derive, deterministically:

```text
tile bounds
tile centers
overlap relationships
local neighboring tiles
relative center offsets
shared physical area / overlap
AOI edge behavior
```

Do not infer neighbors from filename order.

No GT/reference should be loaded in RG2.0.

Produce:

```text
config
preflight script
unit tests
handoff update
deterministic report
small commit
```

Then stop and give me the exact local command to run.

Expected scientific boundary:

```text
RG2.0 does NOT:
  run ORB
  project query points
  use GT
  change stride
  change candidate generation
  invoke bootstrap/state
```

After I paste the RG2.0 result, continue stage-by-stage using the handoff
contract.

Preserve strict research discipline:
- distinguish blind evidence from post-freeze evaluation;
- use full trajectories, not named cases, for aggregate conclusions;
- named cases are diagnostics only;
- no silent parameter changes;
- no GT/SRT/GPS before freeze;
- record results in README/closeout/handoff documents;
- use GitHub connector for repository modifications;
- make small auditable commits;
- do not tune on the blind-stress trajectory.

---

End of continuation prompt.
