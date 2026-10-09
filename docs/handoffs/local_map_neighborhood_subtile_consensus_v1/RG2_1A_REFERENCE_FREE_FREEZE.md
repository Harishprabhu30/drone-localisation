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

## Schema guard corrective patch (2026-10-09)

Initial local execution stopped before writing the freeze:

```text
ValueError: Reference/evaluation columns forbidden:
['sampling_alignment_error_ms']
```

Cause: the blind-column guard matched `error_m` as a substring of
`sampling_alignment_error_ms`. This is a benign timestamp-alignment
diagnostic, not positional ground truth. Narrow correction: `error_m` is
rejected as an **exact column name**, while reference/GT/oracle positional
tokens and exact forbidden evaluation columns remain prohibited. Added tests
for the permitted alignment column and for forbidden `error_m`,
`ground_truth_error`, `reference_x_m`, `oracle_tile_identity`,
and `gps_lat`.

The source code and test-suite changed; frozen stage config, RG2.0 map topology
and input manifest are unchanged. Re-run the complete unit suite and RG2.1-A
freeze. Do not delete/change any pre-existing frozen outputs to bypass the
immutability check.


## RG2.1-A real-data freeze acceptance (2026-10-09)

**Status: PASSED AND CLOSED on user-reported local execution evidence.**
This is a reproducibility/evidence checkpoint; the generated CSVs themselves
remain local, are not uploaded to GitHub, and were not independently read in
this closeout.

- Unit tests: **17 passed** (RG2.0 and RG2.1-A).
- Preflight: \`PASS_RG2_1A_REFERENCE_FREE_NEIGHBORHOOD_FREEZE\`.
- Frozen RG2.0 relationships SHA256:
  \`1a9456cea5c951af30c63d5f1adaca3d82c3b8f1a275a841bf2d0a7d5fd71272\`.
- Reported reference loaded: **False**.
- Reported oracle/reference evaluation executed: **False**.

### Development, 403 queries

\`\`\`text
frozen QV Top20 rows    8060
local union rows        9315
immediate count histogram  {6:71, 7:8, 9:319, 10:5}
overlap count histogram    {15:27, 20:106, 25:262, 30:8}
\`\`\`

Verified arithmetic: each histogram sums to 403 queries; the
immediate histogram accounts for 3,403 membership rows and the overlap
histogram for 9,315 membership rows.

\`\`\`text
local_neighborhoods SHA256
ee708fc261b2833f663fb51a4fc0fd88c926a1a1234ae4d841a93269e8faa100

query_summaries SHA256
15459ad6a5814a08264e15522627165975769d075a98ca5b731375d3c6436b6c

frozen_qv_top20 SHA256
0a7832c6f0997a33851343fd7df537d2f6410211ac0e38e7ec4325899d8df47e
\`\`\`

### Blind stress, 123 queries

\`\`\`text
frozen QV Top20 rows    2460
local union rows        2955
immediate count histogram  {6:8, 9:114, 10:1}
overlap count histogram    {15:4, 20:16, 25:103}
\`\`\`

Verified arithmetic: each histogram sums to 123 queries; the
immediate histogram accounts for 1,084 membership rows and the overlap
histogram for 2,955 membership rows.

\`\`\`text
local_neighborhoods SHA256
1d2fa9cabf7a69609bc94a32f8ff2c8c5282f76d47e0e231ccb8fc3d19ddd3b7

query_summaries SHA256
ace22b4ddf67605d48137087ddfe4c7ce52dae27c0e641e3696d0f2bcbd285d3

frozen_qv_top20 SHA256
68bbf59d3b839cfd90aef50dfca30316cf02296218dd33db236450bbe8b62bd8
\`\`\`

### Interpretation and next gate

For these queries, union row totals equal full-overlap row totals, so
the immediate-neighbor set added **no tile beyond** the full-overlap set.
This is a property of the frozen sets for these trajectories, not a
statement about localization accuracy.

The full-overlap candidate budget differs from and is often greater than
20; the next availability audit **must report per-query candidate budget**
and acknowledge this as a cost/coverage tradeoff, not a controlled
same-budget superiority claim. Do not treat several overlapping tile
windows as statistically independent pieces of evidence.

**RG2.1-B is not yet implemented here.** Its planned task is a
development-only post-freeze oracle-availability audit comparing:
frozen Top1 anchor vs immediate neighborhood vs full overlap vs frozen
Top20, holding the frozen RG2.1-A manifests and SHA256s immutable.
Blind-stress data stay reference-free and cannot support accuracy claims.
