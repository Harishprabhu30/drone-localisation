# RG2.1-A — Reference-free anchor neighborhoods / freeze

Status: **CODE COMMITTED; LOCAL TEST + FREEZE REQUIRED BEFORE RG2.1-B**.

## Scientific contract

RG2.0 local preflight: 432 tiles, 4,772 undirected overlap pairs,
1,624 undirected near-neighbor pairs; 7/7 unit tests reportedly passed.
The 768_s256 index has compressed boundary intervals (3.6 m and 33.2 m)
which are **not** evidence of independent spatial support.

RG2.1-A freezes exactly one anchor per query: QV1.4 Top1,
`source_view=center_square`, `source_rank=1`. The two geometry-only sets are:

- **immediate:** anchor plus RG2.0 `nearby=true` directed neighbors
  (center distance at most sqrt(2) times 51.2 m).
- **full_overlap:** anchor plus RG2.0 `overlaps=true` directed neighbors
  (positive intersection area of georeferenced tile rectangles).

They are *not* subsets of the QV1.4 Top20 by construction. The original
20 ranked tiles are frozen separately as an **unchanged comparator**.
The emitted union table records both memberships and has deterministic
anchor-first, distance-then-ID ordering. Multiple nearly identical AOI-edge
windows remain distinct tile records; no evidence-independence assumption is
made.

Both trajectories are processed and partitioned into distinct output folders:
403-query development, 123-query blind-stress. Both explicitly reuse
`768_s256`, even though the older blind recorded-flight specification's
primary map variant is `512_s256`.

## Run locally on new branch

```bash
git pull --ff-only origin research/local-map-neighborhood-subtile-consensus-v1
source .drone_venv/bin/activate
export PYTHONPATH="$PWD/src"

python -m unittest tests.test_rg2_0_topology tests.test_rg2_1a_freeze_neighborhoods -v

python scripts/villoc/geometry/rg2_1a_freeze_neighborhoods.py \
  --config configs/research/local_map_neighborhood_rg2_1a.yaml
```

Expected status:

```text
PASS_RG2_1A_REFERENCE_FREE_NEIGHBORHOOD_FREEZE
```

Output root: `outputs/research_runs/local_map_neighborhood_subtile_consensus_v1/rg2_1a`.

- Each trajectory: `rg2_1a_local_neighborhoods.csv`
- Each trajectory: `rg2_1a_query_summaries.csv`
- Each trajectory: `rg2_1a_frozen_qv_top20.csv`
- Root: `rg2_1a_freeze_report.json`

SHA256s are recorded for the script/config, unchanged RG2.0 geometry,
actual tile index, both canonical manifests, both QV pools, and all emitted
tables. Frozen artifacts are never overwritten with different bytes.

The stage rejects unexpected oracle/reference-derived CSV columns, manifest
`reference_available=true`, missing/duplicated query IDs, wrong budget,
duplicate tile IDs per Top20, a changed center-square Top1 contract,
unknown map tile IDs, a mutated RG2.0 topology, or missing input files.

**No GT/SRT/reference attachment, image, descriptor, ORB, learned
inference, reranking, bootstrap, or state runs here.** Spatial proximity
does not assert correctness.

## Stop gate

Inspect unit-test summary, full command log, counts for both trajectories,
near/full-overlap count histograms, SHA256 report and any errors.
Do **not** attach reference or run an oracle audit until the frozen manifests
have passed inspection. The next stage is RG2.1-B development-only,
post-freeze availability labels comparing anchor vs immediate vs full-overlap
vs original Top20. The blind-stress trajectory has no reference and must
receive no accuracy claims.
