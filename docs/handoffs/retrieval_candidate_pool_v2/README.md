# Handoff — retrieval candidate pool v2

Status: **A1 CLOSED / PASS — R1 READY / STARTING**

A1 closeout:

```text
docs/closeouts/retrieval_candidate_pool_v2_a1/README.md
```

A1 proved exact canonical 403-query DINO ranking parity through the retrieval
backend and tie-equivalent historical blind-artifact parity under the documented
numerical comparator. The retrieval abstraction is now the required path for R1.

Intended branch:

```text
research/retrieval-candidate-pool-v2
```

Create it from the current documentation-updated `main` in the next session. Do **not** create it from the historical `research/minimum-confident-bootstrap` branch and do not modify the frozen benchmark tag.

The frozen deployment/benchmark reference remains:

```text
jetson-native-baseline-20261007
0f71a2d27ef299d78417904ca381815e9e7c9fd8
```

Documentation-only commits after that tag do not change the frozen algorithm.

---

## 1. Why this branch exists

The next research question is upstream candidate quality:

> Can the map/query representation produce a more reliable and spatially useful Top-K candidate pool before geometric verification and state estimation?

The immediate target is the class of failures represented by q57, where the useful geographic region was absent from Top-20.

q228 is a control case: useful retrieval evidence was already available and the harmful outcome occurred downstream in transform/state logic. Retrieval research must not claim to solve q228 unless downstream evidence changes.

Do not modify `minimum_confident_v2`, temporal authority, ORB acceptance policy, blindness rules or the frozen map-state semantics during the initial retrieval study.

---

## 2. Frozen control

Control pipeline:

```text
video / query frames
  -> XFeat relative frontend
  -> DINOv2 ViT-S/14
       img518
       center_square
       avgpatch
  -> 512_s256 ORT10LT map cache
  -> Top-20
  -> ORB reranking
  -> minimum_confident_v2
  -> causal map alignment / temporal authority
  -> frozen blind output
```

Map source:

- Geoportal ORT10LT 2024–2026;
- EPSG:3346;
- source resolution approximately 0.20 m/pixel;
- promoted `512_s256` footprint approximately 102.4 m with approximately 51.2 m center spacing.

Existing historical map variants already demonstrate that physical footprint matters:

- `512_s256`: about 102.4 m footprint;
- `1024_s512`: about 204.8 m footprint;
- `1024_s256`: about 204.8 m footprint with denser overlap.

CRS is not a retrieval hyperparameter. Keep EPSG:3346 as the metric map geometry unless a separate geospatial correctness issue is demonstrated.

---

## 3. Research ordering

### R0 — reproduce the frozen retrieval control

Before architectural or algorithmic changes:

- load the canonical 403-query cache and frozen `512_s256` map cache;
- reproduce current Top-K results;
- establish the evaluation harness;
- retain q57 and q228 as named diagnostics.

Gate: the control implementation must reproduce the frozen candidate ordering or an explicitly documented equivalent before any new backend is trusted.

### R1 — ground-footprint map pyramid

Status: **STARTING after A1 closeout.**

Keep the DINO backbone and query preprocessing frozen.

Test map windows approximately:

```text
76.8 m   = 384 px at 0.20 m/px
102.4 m  = 512 px   [frozen control]
153.6 m  = 768 px
204.8 m  = 1024 px
```

All levels are encoded at the same network input size so that the experiment isolates represented physical context.

The main question is not "which source raster GSD is best?" The ORT10LT source GSD stays fixed. The experiment changes **ground footprint represented per network input**.

#### R1 primary ablation — fixed center spacing

Use a fixed source-raster stride of 256 px (approximately 51.2 m) for the primary
footprint study:

```text
384_s256
512_s256   [frozen control]
768_s256
1024_s256  [existing fixed-stride large-footprint endpoint]
```

Reason: keeping center spacing fixed reduces candidate-density/overlap as a
confound while changing the physical context represented by each network input.
The overlap ratio is therefore *not* held fixed in the primary ablation.

Do not compare `384_s192`, `768_s384` and `1024_s512` as if that were a pure
footprint experiment: those variants change footprint and center spacing
simultaneously. A constant-50%-overlap study may be run later as a secondary
density/overlap ablation if R1 evidence warrants it.

The primary R1 experiment should reuse existing `512_s256` and `1024_s256`
assets and generate only the missing `384_s256` and `768_s256` map levels.

### R2 — cross-scale candidate fusion

Do not send Top-20 from every scale to ORB.

Use:

```text
per-scale DINO ranks
      -> rank fusion (start with RRF)
      -> spatial duplicate suppression
      -> one merged Top-20
      -> unchanged ORB Top-20 budget
```

Candidate records must carry physical provenance:

- pyramid level;
- source window size;
- ground footprint;
- grid row/column where applicable;
- metric bounds;
- center easting/northing;
- DINO score/rank;
- fused rank.

Spatial deduplication should prevent multiple overlapping representations of the same physical region from consuming the Top-20 budget.

### R3 — query-view/crop study

The 3840x2160 UAV frames are currently center-square cropped before resizing, discarding a large amount of horizontal context.

Compare the control against bounded alternatives such as:

- center-square;
- left/center/right overlapping square crops with rank fusion;
- full-frame resize only as an ablation;
- later aspect-preserving/rectangular-token approaches if justified.

Query-side multi-crop adds online inference cost, so it comes after the cheap offline map-pyramid study.

### R4 — geometry-aware scale selection

Only after R1/R2 prove scale value, use camera intrinsics plus altitude/pitch assumptions to estimate query ground footprint and restrict the searched pyramid levels.

This should be treated as an efficiency policy, not assumed useful before the all-scale experiment.

### R5 — descriptor aggregation / retrieval backend study

Only after physical-scale representation is understood, compare aggregation/backbone alternatives.

Candidates include:

- frozen DINOv2 average-patch control;
- lightweight DINO patch-token spatial pyramid;
- deployment-sized DINOv3 candidate;
- SALAD / AnyLoc-style aggregation;
- optional FlatVPR or another compact VPR backend if justified.

### R6 — downstream replay

Run ORB/bootstrap/state replay only for representations that materially improve candidate availability or spatial diversity.

Do not repeatedly run the full estimator for losing retrieval variants.

---

## 4. Lightweight descriptor spatial pyramid

This is different from the orthomap ground-footprint pyramid.

At DINOv2 ViT-S/14 with 518x518 input, the patch grid is 37x37. The current `avgpatch` representation averages all patch tokens into one 384-D vector.

The first lightweight spatial-pyramid experiment should reuse **one backbone forward pass** and pool:

```text
1x1 global pool        -> 1 x 384-D
2x2 regional pools     -> 4 x 384-D
                         -----------
                         5 descriptors/image
```

Store the result conceptually as `[5, 384]`, not as one opaque 1920-D vector.

Advantages:

- no extra backbone inference;
- preserves some local/partial-overlap evidence lost by global averaging;
- cheap for the current 475-tile control map;
- compatible with larger physical map windows.

Initial scoring should remain simple and testable. For example:

```text
global similarity
  +
small weighted local-overlap term
```

where the local term can start from top regional cross-similarities. Do not assume direct quadrant-to-quadrant correspondence because yaw and partial overlap can invalidate fixed spatial alignment.

A later experiment may add rotation-aware or spatially consistent regional matching, but only if the simple version demonstrates value.

The descriptor-spatial pyramid and the map-footprint pyramid are independent axes:

```text
physical map footprint
        x
descriptor spatial aggregation
```

Do not change both in the first ablation.

---

## 5. Evaluation metrics

Candidate-pool research should report more than center-distance Top-1.

Required metrics:

- Recall@1 / @5 / @20;
- first useful candidate rank;
- useful candidate available in Top-20;
- center error thresholds such as <=40 m and <=80 m;
- geometric tile containment / overlap relevance;
- number of distinct spatial regions represented in Top-20;
- candidate spatial dispersion;
- scene-conditioned results where labels exist;
- q57 and q228 diagnostics;
- descriptor cache size;
- query encoding latency;
- retrieval/fusion latency;
- Jetson memory/latency for finalists.

Large tiles must not be judged only by a strict center-distance threshold; a query can be geometrically contained by a larger tile even when its tile center is more than 40 m away.

Coordinates/oracle information remain evaluation-only and must not participate in blind ranking.

---

## 6. Architecture evolution — when to change what

The project should evolve toward a real-time, pluggable runtime, but not through one large rewrite.

### A0 — now: documentation-only closeout on main

Completed.

No baseline algorithm, dependency file or runtime code is changed here.

### A1 — first on the research branch: thin retrieval abstraction

Do this **before adding several new retrieval methods**, because otherwise every experiment will hard-code another path.

Introduce a narrow interface around the existing behavior, for example:

```text
RetrievalBackend
  build/load map representation
  encode query
  rank candidates
  report backend metadata
```

Likely implementation layout:

```text
retrieval/
  contracts
  dinov2_backend
  future_dinov3_backend
  future_salad_backend

map/
  tile_index
  map_pyramid
  spatial_dedup

verification/
  orb_backend

state/
  minimum_confident_v2

runtime/
  profiling

orchestrator/
  thin wiring
```

The current DINOv2 backend must pass a parity gate before research changes continue.

### A2 — immediately after A1 parity: reproducibility foundation

This is the right time to modernize dependency metadata, **before many new backends multiply the environment complexity**, but after the migrated baseline is safely frozen.

Planned work:

- populate `pyproject.toml`;
- set an explicit tested Python range initially centered on Python 3.10;
- declare currently implicit runtime dependencies, including `psutil`;
- introduce `uv.lock` only after it can recreate the supported environment;
- keep `.drone_venv` as the local convention unless deliberately changed;
- capture model/checkpoint and critical asset SHA256 values;
- record board/runtime profiles.

Do not assume one identical binary environment can be forced onto every board. Jetson uses NVIDIA-specific CUDA/PyTorch builds while macOS uses CPU PyPI wheels.

Use a two-layer reproducibility contract:

```text
common project dependency contract
        +
platform/runtime profile
        +
model/data asset hashes
        +
parity tests
```

The desired end state may use one multi-platform `uv.lock` with markers/sources if it is proven reliable. If NVIDIA Jetson wheels cannot be represented cleanly, keep the common lock plus an explicit Jetson overlay/profile rather than pretending the environments are identical.

Gate: new Mac recreation must pass; Jetson must pass its short parity smoke before dependency modernization is considered complete.

### A3 — R1 to R5: research through stable interfaces

Implement map pyramids, rank fusion, spatial deduplication, query crops, descriptor spatial pyramid and alternative backends behind the contracts from A1.

Do not redesign the whole runtime while the research question is still moving.

### A4 — after a retrieval winner exists: frame-source abstraction

The current video/file workflow is valuable because it is deterministic and reproducible. Keep it as a test adapter.

Then separate ingestion from localization:

```text
FrameSource
  VideoFrameSource
  ImageDirectoryFrameSource
  LiveCamera/GStreamer/ROS2 source later

FramePacket
  image
  monotonic timestamp
  source sequence id
  optional camera/altitude/pitch metadata
```

The online localization API should process a frame/query incrementally instead of requiring a prebuilt flight manifest.

This is the correct point to transition toward real-time input: after the retrieval representation is selected, before hardware flight integration.

### A5 — real-time runtime

After FrameSource parity:

- bounded queues;
- explicit frame-drop/backpressure policy;
- stage timestamps and latency budgets;
- offline map precomputation;
- warm model/cache initialization;
- high-rate relative frontend separated from lower-rate absolute retrieval;
- incremental persistent state rather than replaying whole history;
- resource telemetry;
- deterministic recorded-video replay through the same online API.

Recorded video must remain a first-class regression source even after live cameras are added.

### A6 — deployment/board optimization

Only after the online architecture exists:

- profile the selected pipeline on Jetson;
- define latency/memory/power budgets from measured stage costs;
- optimize the actual bottleneck;
- consider descriptor precision/compression, approximate search or GPU/CPU placement only with evidence;
- retain CPU fallback where practical.

Resume Docker/container work **after** the native runtime and dependency contracts are stable. Do not merge the historical unfinished Docker branch wholesale.

### A7 — CI, GitHub automation and agent readiness

Once `pyproject.toml`, stable CLIs and small parity fixtures exist:

- add fast unit/contract tests;
- add a tiny retrieval parity fixture that does not require the private/full dataset;
- add import/environment checks;
- add config/schema validation;
- add lint/format tooling only if it reduces maintenance rather than creating churn;
- add GitHub Actions for fast checks;
- use manual/self-hosted workflows for heavy Jetson benchmarks;
- protect frozen tags and prefer PRs for automated/agent changes.

Do not put multi-minute/full-dataset research benchmarks in the default CI path.

### A8 — coding/AI agents

When module boundaries and tests are stable, add a concise root `AGENTS.md`.

It should point agents to:

- current architecture;
- handoff document;
- allowed branch;
- commands/tests;
- frozen contracts;
- data/blindness rules;
- files they must not modify;
- definition of PASS.

Whether the agent is OpenAI-based or another coding-agent system, stable machine-readable interfaces matter more than vendor-specific prompts.

Agent automation should initially be limited to well-bounded work such as:

- test generation;
- report/metric extraction;
- cache/schema validation;
- documentation updates;
- controlled backend implementation behind an interface;
- PR creation.

Do not grant an agent an unconstrained path to rewrite the estimator, modify frozen outputs, change blind/evaluation boundaries or push directly to protected `main`.

---

## 7. Reproducibility / maintainability target

A future contributor should be able to answer these questions without reading months of logs:

```text
What algorithm is frozen?
What branch am I on?
What exact model/cache generated this result?
What platform profile am I using?
What command reproduces the small parity test?
What output schema should appear?
Which metrics changed because of research?
Which changes are only infrastructure?
```

Move toward:

- typed/shared configuration rather than duplicated constants;
- explicit artifact schemas/versioning;
- provenance in every cache/report;
- content hashes for model/data contracts;
- deterministic seeds where applicable;
- stage-level timing;
- small fixtures;
- no hidden absolute developer paths;
- no network download during a frozen run unless explicitly allowed;
- README closeouts instead of conversational state as the sole source of truth.

---

## 8. Git workflow for the next session

In the next chat:

1. update the new-Mac `main` with the documentation commits;
2. verify a clean tree;
3. create `research/retrieval-candidate-pool-v2` from current `main`;
4. record its starting commit;
5. implement **A1 only**: the minimal retrieval abstraction/parity harness;
6. prove frozen DINOv2 behavior through the abstraction;
7. then begin R1 map-footprint pyramid.

Do not start with DINOv3, live camera ingestion, Docker, `uv.lock`, state-policy changes or a large source-tree rewrite in the same first commit.

The first branch milestone is:

> Existing DINOv2 + frozen map queried through the new retrieval interface produces the same candidate ordering as the current control.

Only after that gate passes should the branch become an experimental retrieval platform.

---

## 9. Closeout condition for this research line

The retrieval-candidate-pool-v2 phase can close when:

- one representation/fusion strategy clearly improves useful candidate availability or spatial diversity over the frozen control;
- gains are characterized by scene/failure type;
- q57 is explicitly re-evaluated;
- q228 remains correctly classified as downstream unless new evidence says otherwise;
- Jetson cost is measured for finalists;
- the winning retrieval backend fits the pluggable contract;
- recorded replay remains reproducible;
- the next state/realtime integration question is separately scoped.
