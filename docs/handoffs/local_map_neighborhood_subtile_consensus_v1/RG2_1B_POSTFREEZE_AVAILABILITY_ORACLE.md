# RG2.1-B — Post-freeze local-region availability oracle audit

Status: **CODE PREPARED / LOCAL DEVELOPMENT EVALUATION PENDING**.

## Purpose

Measure whether a frozen local map neighborhood makes a useful map window
*available* when the frozen retrieval anchor is imperfect. This is an oracle
availability diagnostic, not a deployable candidate selector.

The candidate sets were already produced and hashed in RG2.1-A without
reference. RG2.1-B must verify those bytes before opening reference.

## Frozen inputs

Development queries: 403.

RG2.1-A pinned SHA256:

\`\`\`text
local_neighborhoods
ee708fc261b2833f663fb51a4fc0fd88c926a1a1234ae4d841a93269e8faa100

query_summaries
15459ad6a5814a08264e15522627165975769d075a98ca5b731375d3c6436b6c

frozen_qv_top20
0a7832c6f0997a33851343fd7df537d2f6410211ac0e38e7ec4325899d8df47e
\`\`\`

RG2.0 relationship topology SHA256:

\`\`\`text
1a9456cea5c951af30c63d5f1adaca3d82c3b8f1a275a841bf2d0a7d5fd71272
\`\`\`

No blind-stress CSV is an RG2.1-B input.

## Oracle semantics

Reuse existing project semantics exactly:

\`\`\`text
containment:
  gt_x within [left_easting, right_easting]
  gt_y within [bottom_northing, top_northing]

<=40:
  tile-center Euclidean error <= 40 m

<=80:
  tile-center Euclidean error <= 80 m
\`\`\`

Reference is loaded via the existing QV1 reference attachment loader
(\`eval_ref_lon/lat -> EPSG:3346\`).

## Sets

\`\`\`text
anchor
immediate
full_overlap
frozen_top20
\`\`\`

The first three are nested by construction:

\`\`\`text
anchor ⊆ immediate ⊆ full_overlap
\`\`\`

Top20 includes the same anchor but otherwise is an independently frozen
retrieval candidate set.

## Required historical parity

Before interpreting new results:

\`\`\`text
frozen Top20:
  contain 384 / 403
  <=40    277 / 403
  <=80    380 / 403

anchor:
  contain 174 / 403
  <=40    101 / 403
  <=80    171 / 403
\`\`\`

Any mismatch is a hard stop.

## Outputs

Output root:

\`\`\`text
outputs/research_runs/local_map_neighborhood_subtile_consensus_v1/rg2_1b/
\`\`\`

Files:

\`\`\`text
rg2_1b_local_candidate_oracle_labels.csv
rg2_1b_frozen_top20_oracle_labels.csv
rg2_1b_query_availability_comparison.csv
rg2_1b_availability_oracle_report.json
\`\`\`

These are **REFERENCE-CONTAMINATED EVALUATION ARTIFACTS**. Never use them as
blind inputs to RG2.2, bootstrap, state, retrieval or parameter fitting.

The report includes candidate-budget distributions, absolute availability,
best center-error summaries, and paired outcomes (both / A-only / B-only /
neither). The most important paired comparison is full-overlap vs frozen
Top20, because it distinguishes local-neighborhood-only rescues from
retrieval-Top20-only rescues.

Candidate count is part of the scientific result. Full-overlap often exceeds
20 windows, so this is a coverage/cost comparison, **not same-budget
superiority**.

## Local command

\`\`\`bash
git pull --ff-only origin research/local-map-neighborhood-subtile-consensus-v1
source .drone_venv/bin/activate
export PYTHONPATH="$PWD/src"

python -m unittest \
  tests.test_rg2_0_topology \
  tests.test_rg2_1a_freeze_neighborhoods \
  tests.test_rg2_1b_postfreeze_availability_oracle -v

python scripts/villoc/geometry/rg2_1b_postfreeze_availability_oracle.py \
  --config configs/research/local_map_neighborhood_rg2_1b.yaml
\`\`\`

Expected test total after this commit: **27**.

Expected audit status:

\`\`\`text
PASS_RG2_1B_POSTFREEZE_AVAILABILITY_ORACLE_AUDIT
\`\`\`

## Stop gate

Return the complete terminal output, especially:

- four set availability counts;
- mean/median/min/max candidate budgets;
- full-overlap vs Top20 paired outcomes for containment, <=40 and <=80;
- historical parity result;
- query comparison SHA256.

Do not implement RG2.2 until these results are interpreted.


## Stable import-path fix (2026-10-09)

Initial direct execution failed before the audit began:

\`\`\`text
ModuleNotFoundError: No module named 'scripts'
\`\`\`

Cause: RG2.1-B imported \`load_reference_xy\` from another executable file under
\`scripts/\`. When Python executes a nested script directly, its containing
directory becomes the import root; repository-root \`scripts\` is therefore not
a stable import target. \`PYTHONPATH=$PWD/src\` correctly exposes \`uavloc\`,
but it does not expose repo-root \`scripts\`.

Stable correction:

- reusable reference loading moved to
  \`src/uavloc/evaluation/reference.py\`;
- RG2.1-B now bootstraps \`<repo>/src\` from its own \`__file__\` before
  importing \`uavloc\`;
- no RG2.1-B import from \`scripts.*\` remains;
- test coverage includes the shared loader and a subprocess
  \`--help\` smoke test with \`PYTHONPATH\` removed;
- repository convention documented in
  \`docs/development/python_import_contract.md\`.

This changes import structure only. The historical oracle semantics remain
\`eval_ref_lon/lat -> EPSG:3346\`; RG2.1-A hashes and candidate sets are
unchanged. The failed execution occurred during module import, before reference
loading or RG2.1-B output generation.
