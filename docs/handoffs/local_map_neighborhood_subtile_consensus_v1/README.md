# Handoff — RG2 local map neighborhood / sub-tile consensus v1

Status: **PLANNED / START IN NEXT CHAT**

Proposed branch:

```text
research/local-map-neighborhood-subtile-consensus-v1
```

Base this branch on the final closeout commit of:

```text
research/region-candidate-geometry-v1
```

Primary predecessor:

```text
docs/closeouts/region_candidate_geometry_v1/README.md
```

## 1. Continuation chain

Before implementing RG2, recover the research lineage in this order:

```text
absolute frontend / retrieval diagnostics
        ↓
docs/closeouts/query_view_candidate_generation_v1/README.md
        ↓
QV1.4 frozen fixed-budget region candidate generator
        ↓
docs/closeouts/trajectory_adapter_v1/README.md
        ↓
trajectory-independent development + blind-stress interface
        ↓
docs/closeouts/region_candidate_geometry_v1/README.md
        ↓
RG1 geometry role:
  trust / abstain
  + continuous sub-tile refinement
        ↓
RG2 local map neighborhood / multi-tile consensus
```

Also reconnect the earlier bootstrap thread:

```text
README.md
docs/demo/blind_recorded_flight_final_001/README.md
configs/bootstrap/minimum_confident_v2/architecture_contract.json
scripts/villoc/research/minimum_confident_bootstrap/diagnostics/
  r4_11_blind_subtile_projection_recompute.py
  r4_18_acquisition_to_tracking_counterfactual.py
```

These earlier artifacts contain the motivation that repeated consecutive
observations can collapse to the same tile center and that wrong neighboring
tile selections can create artificial map-span diversity.

## 2. Frozen research context entering RG2

### Development trajectory

```text
villoc_traj01_90deg_stable120m
403 queries
reference available only post-freeze
```

### Blind stress trajectory

```text
villoc_blind_recorded_flight_final_001
123 queries
no GT/SRT/reference
```

### Trajectory adapter

Use the canonical trajectory packages created by Trajectory Adapter v1.

### Frozen global candidate generation

```text
QV1.4
center_unique_allview_fill20
768_s256
fixed Top20
center Top1 preserved
```

Development reference:

```text
contain R20 384 / 403
<=80 R20   380 / 403
<=40 R20   277 / 403
```

### Frozen geometry knowledge

Do not treat the following as independently validated deployment thresholds.

Provisional development-derived hypothesis:

```text
retrieval-anchor trust:
  ORB inliers >= 6
```

Development behavior:

```text
contains precision 0.5608
contains recall    0.9540

<=80 precision     0.5507
<=80 recall        0.9532

<=40 precision     0.3311
<=40 recall        0.9703
```

Useful containing candidates:

```text
tile-center median error     52.7231 m
projected-point median error  6.4977 m
projection improves           79.5%
```

Blind stress:

```text
inliers >= 6 accepted 77 / 123
accepted fraction 0.6260
no accuracy claim
```

## 3. Key conceptual change for RG2

Do not use tile identity as the final absolute observation.

A tile is an overlapping image window onto the physical map.

The desired representation is:

```text
retrieved physical region
       ↓
anchor tile + local overlapping neighbors
       ↓
independent geometric projections
       ↓
convert all valid projections to EPSG:3346
       ↓
cluster / agree in global coordinates
       ↓
one locality hypothesis
       ↓
continuous position estimate
```

If multiple neighboring tiles observe the same physical location, their
continuous global projections should ideally agree even though their tile IDs
differ.

## 4. Important bootstrap connection

Do not count consecutive frames as independent absolute confirmations merely
because they produce separate accepted tile observations.

Example:

```text
q20 -> tile A
q21 -> tile A
q22 -> overlapping tile B
q23 -> overlapping tile B
q24 -> tile C
```

may represent one persistent locality, not five independent map hypotheses.

RG2 should produce a locality representation that a later bootstrap/state stage
can reason about using:

```text
temporal span
relative-motion span
map-position span
viewpoint diversity
distinct locality support
projection consistency
```

Do NOT modify bootstrap in RG2.0--RG2.3.

## 5. Keep frozen initially

```text
map footprint: 768 px ~= 153.6 m
stride:        256 px ~= 51.2 m
map variant:   768_s256
QV1.4 candidate generation
DINO query-view caches
ORB frontend contract
trajectory adapter
strict blind/reference boundary
```

Do NOT start with a stride sweep.

## 6. Proposed RG2 stages

### RG2.0 — neighborhood/topology contract

Goal:

Define map neighbors geometrically from EPSG:3346 bounds/centers, not filename
ordering.

Build for every 768_s256 tile:

```text
tile ID
center
bounds
overlap relationships
near neighbors
relative offset
shared physical area
```

Preflight only.

No GT.

Questions:

```text
How many neighboring/overlapping tiles does each tile have?
What is the exact overlap topology of the current 51.2 m stride grid?
Can all neighborhoods be generated deterministically at AOI edges?
```

### RG2.1 — local-region availability oracle audit

Development only, reference attached after blind neighborhood generation.

Question:

> When the retrieval anchor is near the correct physical locality, does its
> frozen local neighborhood contain a useful/containing map window even when
> the anchor tile itself is imperfect?

Compare:

```text
anchor only
anchor + fixed local neighborhood
existing QV1.4 Top20 availability
```

This is an availability/oracle diagnostic, not a deployable selector.

### RG2.2 — blind multi-tile sub-tile projection generation

For each query:

```text
frozen retrieval anchor
  -> local overlapping neighborhood
  -> ORB geometry per local tile
  -> query-center projection per valid tile
  -> EPSG:3346 projected point
  -> freeze/hash
```

No GT before freeze.

### RG2.3 — global-coordinate projection consensus audit

After freeze on development:

Compare:

```text
anchor tile center
single-anchor sub-tile projection
best local valid projection [oracle diagnostic only]
multi-tile robust projection consensus
```

Consensus candidates to evaluate should remain simple first:

```text
coordinate-wise median
geometric median
small-radius spatial cluster / medoid
inlier-weighted robust centroid
```

Do not fit a learned model initially.

Key outputs:

```text
number of local tiles tested
number with geometry
number of projected points
projection cluster size
projection spread
cross-tile agreement
continuous consensus coordinate
post-freeze position error
```

### RG2.4 — locality persistence / evidence-independence audit

Before bootstrap changes, ask:

> How many consecutive accepted observations represent genuinely new map
> evidence versus repeated support for one persistent locality?

Measure:

```text
locality ID / cluster persistence
time span
relative-motion span
continuous map-position span
number of distinct supporting tile windows
projection dispersion
locality transitions
```

This stage should diagnose how a future bootstrap ought to count support.

Do not implement bootstrap/state in RG2.4.

## 7. Deferred stage — stride research

Only after the locality representation is understood should a later map
sampling study vary stride while holding footprint and algorithm fixed.

Potential future family:

```text
RG3 — Map Sampling / Stride Ablation
```

Possible strides:

```text
25.6 m
51.2 m current control
76.8 m
102.4 m
```

Evaluate:

```text
candidate redundancy
locality coverage
overlapping useful tiles
projection consensus quality
alias behavior
runtime
memory
bootstrap evidence independence
```

Do not start RG2 by changing stride.

## 8. Critical interpretation rule

RG1.5 showed:

```text
development changed-tile center jump median 184.6 m
blind changed-tile center jump median       323.8 m
```

Therefore local-neighborhood consensus must not be described as a solution to
all global aliases.

Keep the architectural distinction:

```text
global retrieval:
  get into the right physical region

local neighborhood geometry:
  become tile-boundary invariant
  refine continuous position

future temporal/state logic:
  reject implausible absolute-region transitions
```

## 9. First action in the next chat

Do not immediately code RG2.1.

First:

1. verify the current branch/commit and read the predecessor closeouts;
2. create `research/local-map-neighborhood-subtile-consensus-v1` from the RG1
   closeout commit;
3. implement RG2.0 only;
4. inspect the actual 768_s256 tile index and derive neighborhood topology from
   map geometry;
5. keep reference unavailable during RG2.0.

The first gate should be a deterministic topology/preflight report.

## 10. Stop condition

If RG2.0 reveals that the map index/bounds are inconsistent or that the
neighborhood cannot be defined unambiguously, stop and repair the map contract
before running projection consensus experiments.
