# Jetson Orin NX full403 native benchmark closeout and research transition — 2026-10-06

## Start here

This document is the **current native Jetson benchmark closeout and handoff** for
the GNSS-denied UAV visual-localization project.

It supersedes the "full traj01 GPU run pending" status in:

- `docs/benchmarks/jetson_native_blind_baseline/README.md`
- `docs/benchmarks/jetson_orin_nx_20261005/README.md`

The full `traj01_90deg_stable120m` blind pipeline has now been executed
natively on Jetson Orin NX with CUDA-enabled XFeat and DINO query inference,
frozen before reference attachment, transferred to the Mac without the 1.6 GB
extracted-frame directory, and evaluated post-freeze against the isolated
403-row reference trajectory.

This is both:

1. a deployment/runtime benchmark closeout; and
2. the evidence bridge back into the absolute-localization research program.

Do **not** restart the earlier project stages or reinterpret the historical
controlled-fusion result as the same estimator used here. Continue from the
frozen evidence and unresolved questions recorded below.

---

# 1. Executive summary

## 1.1 What has been proven

The promoted recorded-flight orchestrator completed the full 403-query Villoc
trajectory natively on Jetson Orin NX:

- 17 / 17 orchestrator stages passed;
- no GPS, SRT, reference trajectory, oracle label or evaluation metric was used
  before output freeze;
- XFeat relative motion executed on CUDA;
- DINO query descriptor inference executed on CUDA;
- the existing frozen Mac-built CPU DINO map cache was reused unchanged;
- cached descriptor retrieval remained CPU / NumPy;
- ORB Top-20 verification remained CPU because the installed OpenCV build
  exposes no CUDA ORB device;
- `minimum_confident_v2` produced a provisional absolute map state;
- a 403-row submission was frozen and hashed;
- reference was attached only afterward for evaluation.

The native execution path is therefore operationally proven.

## 1.2 What has *not* been proven

The result does not establish production-ready localization accuracy or real-time
performance.

The full frozen trajectory has high map-coordinate coverage but poor absolute
accuracy overall. The dominant accuracy failures are now localized to a small
number of harmful map-state replacements and scene-dependent retrieval /
sub-tile reliability problems.

The complete pipeline also remains well below the 1 Hz deployment target.

---

# 2. Repository, branches, worktrees and machines

Repository:

`Harishprabhu30/drone-localisation`

## 2.1 Mac main repository

Path:

`/Users/harishprabhu/Documents/drone-localisation`

Parked branch:

`feature/dockerized-blind-eval-tool`

Parked Docker checkpoint:

`74ed70ed2c267a44c79dcd2cd3562f2212c5fd32`

Purpose:

- retains the unfinished Docker wrapper / compose work;
- also holds the large local canonical assets used through worktree symlinks;
- Dockerization is intentionally parked until the native benchmark is fully
  documented and the remaining native experiments are complete.

Do not casually mix benchmark changes into this worktree.

## 2.2 Mac benchmark worktree

Path:

`/Users/harishprabhu/Documents/drone-localisation-jetson-native`

Branch:

`benchmark/jetson-native-blind-baseline`

Source commit for the successful full403 run:

`cc81e8bd11e10d75cb749b8fbc21cc00a2a6dc14`

Important branch commits:

- `3dd33d0` — initial frozen native Jetson baseline documentation;
- `73cf50d` — retain native branch / handoff work;
- `05ed1f6` — canonical traj01 blind-v2 configuration contract;
- `548bbd5` — DINO CUDA query execution support;
- `8557120` — independent DINO/XFeat GPU execution-device plumbing;
- `cc81e8b` — decouple current-flight query-cache path from frozen map tag in
  runtime/resource registry stages.

The Mac benchmark worktree reuses the main-repository Python environment:

`/Users/harishprabhu/Documents/drone-localisation/.drone_venv/bin/activate`

Large ignored assets are bridged from the main repository where required.

## 2.3 Jetson repository

Path:

`/home/rodu/projects/drone-localisation-jetson-native`

Branch:

`benchmark/jetson-native-blind-baseline`

Successful run source commit:

`cc81e8bd11e10d75cb749b8fbc21cc00a2a6dc14`

Environment:

`/home/rodu/projects/venvs/drone-localisation-jetson-native/bin/activate`

Always run from repository root with:

`export PYTHONPATH=$PWD/src`

---

# 3. Hardware and runtime environment

Jetson device:

- NVIDIA Jetson Orin NX Engineering Reference Developer Kit;
- Ubuntu 22.04.5 LTS;
- L4T R36.4.4;
- kernel 5.15.148-tegra;
- power mode during full benchmark: MAXN;
- Python 3.10.12;
- torch 2.8.0;
- CUDA build 12.6;
- CUDA device name: `Orin`;
- CUDA capability: `(8, 7)`;
- OpenCV 4.10.0;
- NumPy 1.26.4;
- pandas 2.2.2;
- scipy 1.11.4;
- PyYAML 6.0.2.

XFeat source revision:

`e92685f57f8318b18725c5c8c0bd28c7fe188d9a`

OpenCV reports no CUDA-enabled device for its CUDA module in this environment,
therefore the promoted ORB verifier remains CPU.

---

# 4. Canonical full-flight blind contract

Configuration:

`configs/demo_villoc_traj01_90deg_stable120m_blind_v2.yaml`

Configuration SHA256 at Stage 5:

`dd3a6a80c835b9336aea3f1adad29b1365e7d7c09723b222997d3b20eed65a18`

Dataset:

`traj01_90deg_stable120m`

Canonical video:

`data/raw/villoc/traj01_90deg_stable120m/villoc_traj01_90deg_stable120m_V_merged.MP4`

Video SHA256:

`c2c8f512ed04277124e71da14794f1b5335d3ee6683ffcbb50594b525e6fab2d`

Video metadata used by the sampler:

- FPS: 29.968385707914084;
- frame count: 12,090;
- duration: 403.4251333333333 s;
- blind sampling: 1.0 FPS;
- manifest: 403 queries;
- first timestamp: 0.0 s;
- last timestamp: 402.0 s.

Map:

- source: Geoportal ORT10LT 2024–2026;
- CRS: EPSG:3346;
- promoted variant: `512_s256`;
- tile size: 512 px;
- stride: 256 px;
- overlap: 50%.

Frozen map descriptor protocol:

`dinov2_vits14_img518_center_square_avgpatch_cpu`

Frozen map cache:

`outputs/villoc/90_deg/descriptors/s8_11b_dinov2_map_512_s256_dinov2_vits14_img518_center_square_avgpatch_cpu.npz`

Map-cache SHA256:

`55f49fd03ba438fe4a4d3597d825ca1bde03fd5c7d4844e22eacdd31bf08bcbe`

Map-index SHA256:

`894369e5800a5ae9b7976d530905b591271db1a8577d8058414ca67a54d9aa2c`

The map cache was hashed before and after the full run and remained unchanged.

Blind execution:

- relative frontend: XFeat;
- XFeat execution device: CUDA;
- query global descriptor: DINOv2 ViT-S/14, img518,
  center_square, avgpatch;
- DINO query execution device: CUDA;
- map descriptor cache: existing CPU-built cache;
- retrieval depth: Top-20;
- verifier: ORB Top-20;
- bootstrap/state backend: `minimum_confident_v2`;
- selected policy: `activate_quarter_track_quarter`;
- activation threshold: 12.8 m;
- tracking threshold: 12.8 m;
- maturity support: 3;
- temporal authority remains owned by the v2 causal map-state timeline;
- output state is provisional, not certified `ABSOLUTE_LOCKED`.

Forbidden before freeze:

- SRT;
- GPS/GNSS;
- latitude/longitude reference;
- reference ENU;
- GT trajectory;
- oracle tile identity;
- GT error;
- RMSE / p95 / final-error feedback;
- evaluation-derived alignment.

---

# 5. Stage-by-stage benchmark history

## Stage 0 — branch/worktree freeze

Goal:

Preserve unfinished Docker work and create an isolated native-benchmark path.

Outcome:

- Docker work parked on `feature/dockerized-blind-eval-tool`;
- benchmark worktree created on
  `benchmark/jetson-native-blind-baseline`;
- algorithm behavior intentionally frozen while deployment/runtime work
  proceeded.

## Stage 1 — canonical traj01 blind-v2 contract

A dedicated configuration was created for the 403-query stable-120m trajectory.

Critical requirements:

- `minimum_confident_v2` explicitly selected;
- blind/reference boundary explicit;
- DINO protocol and map cache frozen;
- ORB Top-20 retained;
- XFeat retained;
- no Stage-18D / q390 research fix promoted into the benchmark.

## Stage 2 — native Jetson runtime preflight

Verified:

- torch CUDA available;
- DINO weights/cache available;
- XFeat revision/weights available;
- production video/map/tile/cache assets present;
- OpenCV ORB remains CPU;
- full runtime/device execution matrix established.

Execution matrix:

- MP4/JPEG decode: CPU;
- XFeat network: CUDA;
- affine/RANSAC: CPU OpenCV;
- DINO preprocessing: CPU;
- DINO neural inference: CUDA;
- DINO output / Top-K search: CPU NumPy;
- ORB verification: CPU;
- bootstrap/state/alignment/export/reporting: CPU.

This pipeline should be described as **mixed CPU/GPU native execution**, not
"all-GPU".

## Stage 3 — GPU plumbing

DINO builder was extended with explicit CPU/CUDA query-device support while
preserving CPU defaults and keeping map-cache identity separate from query-cache
identity.

Controlled 4-frame CPU/CUDA test:

- descriptor shape: 4 x 384;
- max absolute CPU/CUDA difference approximately 3.48e-7;
- cosine similarity essentially 1;
- Top-1 identical 4/4;
- Top-20 membership identical 20/20;
- exact Top-20 order identical 4/4;
- map cache reused without modification.

The orchestrator was then extended with:

- `--dino-device {cpu,cuda}`;
- `--xfeat-device {cpu,cuda}`;
- runtime provenance for map tag, query tag and execution devices.

## Stage 4 — short full-chain mixed CPU/GPU smoke

A stream-copy prefix of the canonical MP4 was created.

Initial 1 FPS random-seek validation exposed an OpenCV tail-seek problem in the
truncated GOP. This was treated as an input-smoke-design issue, not an
algorithmic failure.

A 0.8 FPS smoke-only config sampled nine positions:

0.00, 1.25, 2.50, 3.75, 5.00, 6.25, 7.50, 8.75, 10.00 s.

Those sampled frames were pixel-identical to the corresponding full-video
frames.

First real smoke run:

`stage4_short_mixed_gpu_smoke_001`

Stages 1–11 passed, but Stage 12 failed because the runtime registry derived the
current-flight DINO query-cache filename from the frozen CPU map tag and looked
for a non-existent CPU query cache.

The failure was intentionally preserved.

Fix:

`cc81e8b fix(benchmark): decouple query cache from map tag`

Changes:

- runtime benchmark accepts explicit `--query-cache`;
- resource report accepts explicit `--query-cache`;
- orchestrator passes the resolved current-flight CUDA query cache to
  registry/resource stages;
- old CPU/tag-derived fallback remains available.

The fix was regression-tested against a disposable copy of the failed run.

Clean smoke run:

`stage4_short_mixed_gpu_smoke_002`

Result:

- 17/17 stages passed;
- CUDA DINO query cache;
- CUDA XFeat provenance;
- frozen CPU map cache unchanged;
- 9-row frozen submission.

Stage 4 closed.

## Stage 5 — full 403-query native Jetson run

Run ID:

`traj01_90deg_stable120m_blind_v2_jetson_001`

Start condition:

- date: 2026-10-06;
- source commit: `cc81e8b`;
- disk free before run: approximately 20 GB;
- memory available before run: approximately 6.2 GiB;
- MAXN power mode;
- tegrastats logging started before orchestrator execution.

Execution:

- shell wall time: 1500 s;
- orchestrator wall time: 1499.237 s;
- 17/17 stages passed;
- no fail-fast marker;
- disk free after run: approximately 18 GB.

Raw whole-pipeline throughput:

- 403 queries / 1499.237 s ≈ 0.269 queries/s;
- ≈ 3.72 s/query.

Do **not** infer the dominant bottleneck only from this number. Per-stage timing
breakdown remains a pending benchmark task.

XFeat frontend report:

- 403 frames / 402 pairs;
- affine success rate: 1.0;
- good-quality rate: 0.895522;
- median inlier ratio: 0.55146;
- p05 inlier ratio: 0.31078;
- median inliers: 355;
- feature time: 16.503 s;
- matching/RANSAC time: 1.683 s;
- XFeat-stage wall time: 65.934 s;
- mean pair wall time: 0.1640 s.

Hardware telemetry:

- tegrastats log captured 1,758 samples;
- visible start/end samples show RAM around 1.3–1.45 GiB used by the system;
- swap remained low, roughly 133–156 MB in the shown samples;
- endpoint temperatures were around 48–51 C;
- no OOM or thermal failure occurred.

These are **sample observations**, not yet parsed peak/mean telemetry statistics.
Parse the full log before reporting peak RAM, average power, peak temperature or
GPU utilization.

Frozen submission:

- status: `PASS_BLIND_SUBMISSION_FROZEN`;
- state: `PROVISIONAL_ABSOLUTE_LOCK`;
- rows: 403;
- map positions available: 393;
- unavailable: 10;
- estimated lat/lon available: 393;
- accepted corrections/map-state events: 36;
- blind maturity query: 11;
- frozen submission SHA256:
  `09b6e40f0b3da9cb84bf8db05f9492431a11c1627fc29b4f51ae0284cc152199`.

The freeze record reports 359/393 available positions inside the prepared AOI
and 34 outside. This is a geometric plausibility check, not an accuracy metric.

## Stage 6 — post-freeze evaluation and failure localization

Reference source on Mac:

`/Users/harishprabhu/Documents/drone-localisation/outputs/villoc/traj01_90deg_stable120m/trajectories/s8_3_reference_trajectory_V_1fps.csv`

Reference rows:

403.

Submission/reference timestamp alignment:

- 403/403 exact;
- max timestamp difference: 0 s.

GT latitude/longitude was projected directly to EPSG:3346.

No fitted GT scale, rotation or translation was applied to the frozen absolute
trajectory.

### Stage 6B — absolute accuracy

Available map positions:

393/403 = 97.52%.

Absolute error over available map positions:

- RMSE: 79.020 m;
- mean: 73.356 m;
- median: 70.124 m;
- p75: 95.646 m;
- p90: 109.340 m;
- p95: 114.946 m;
- max: 121.754 m;
- final available error: 65.676 m.

Threshold success:

- <=10 m: 44/393 = 11.20%;
- <=20 m: 46/393 = 11.70%;
- <=40 m: 46/393 = 11.70%;
- <=80 m: 242/393 = 61.58%;
- <=120 m: 385/393 = 97.96%.

Accepted corrections:

36.

Accepted-event error:

- RMSE: 18.260 m;
- mean: 10.839 m;
- median: 7.857 m;
- p95: 22.818 m;
- max: 84.329 m.

Key interpretation:

High coordinate coverage is **not** high localization accuracy. Many accepted
absolute observations are individually useful, but a small number of harmful
state-transform replacements dominate the final trajectory error.

### Stage 6C — causal event audit

35 events have a previous-state counterfactual.

Most state updates are benign.

Three updates worsen event-frame error by more than 5 m:

- q36: +5.944 m;
- q57: +48.547 m;
- q228: +17.351 m.

Dominant failures:

q57:

- transform switch distance: 50.838 m;
- resulting segment RMSE: 67.464 m.

q228:

- transform switch distance: 38.830 m;
- resulting segment RMSE: 97.493 m.

### Stage 6D — Jetson CUDA XFeat relative-only evaluation

Historical-style 50-frame prefix similarity evaluation:

- reference path: 1962.727 m;
- prefix distance: 257.744 m;
- prefix scale: 0.184399236 m/px;
- prefix rotation: -30.564214 deg;
- RMSE: 42.816 m;
- mean: 38.930 m;
- median: 47.038 m;
- p95: 59.593 m;
- max: 61.754 m;
- final error: 39.171 m;
- evaluation distance: 1704.983 m;
- final drift: 2.297 m/100 m;
- global-fit diagnostic RMSE: 19.129 m.

Mac-vs-Jetson raw integrated XFeat trajectories are not numerically identical:

- median XY delta: 43.10 px;
- p95: 130.59 px;
- max: 137.73 px;
- exact rows: 1/403.

However, final-error/drift scale remains broadly comparable to the historical
Mac result. Treat platform parity as a diagnostic, not the primary research
objective.

### Stage 6E — q57/q228 scene and candidate analysis

q54–q59:

- dense forest canopy / low-context natural imagery.

q56/q57:

- no GT-containing tile in DINO Top-20;
- DINO Top-1 approximately 519–528 m from GT;
- ORB promotes the best available candidate, roughly 56–65 m from GT.

q58:

- useful candidate still absent;
- selected ORB candidate approximately 455 m from GT;
- state manager correctly holds rather than installing another state.

q227–q229:

- urban/building/road scene;
- useful candidate available;
- q227 DINO/ORB selected tile roughly 20.8 m from GT;
- q228 selected DINO/ORB tile roughly 21.2 m from GT;
- q229 selected tile roughly 23.0 m from GT.

Therefore q57 is primarily a **candidate-availability / low-observability**
failure, while q228 is a **downstream map-observation/hypothesis/state**
failure.

### Stage 6E transform-construction audit

q57 accepted leader:

- evidence: q1, q33, q41, q57;
- fitted scale: 0.202495 m/visual-px;
- rotation: -36.554 deg;
- q1 projected GT error: 3.017 m;
- q33: 3.036 m;
- q41: 128.562 m;
- q57: 8.553 m;
- q41 Jacobian condition: 25.327;
- q57 projected point outside tile;
- median evidence GT error: 5.794 m;
- max evidence GT error: 128.562 m.

q228 accepted leader:

- evidence: q41, q100, q146, q228;
- fitted scale: 0.206728 m/visual-px;
- rotation: -25.274 deg;
- q41 projected GT error: 128.562 m;
- q100: 5.147 m;
- q146: 6.135 m;
- q228 chosen rank-4 observation: 69.145 m;
- median evidence GT error: 37.640 m;
- max: 128.562 m.

The same bad q41 sub-tile projection contaminates both dominant harmful
transforms.

### Stage 6F — broad structural event audit

Canonical events:

36.

Counterfactual-comparable events:

35.

Evidence age:

- median oldest evidence age: 27.5 queries;
- p95: 55.2;
- events with evidence >100 queries old: 1;
- >200: 0.

Therefore generic "whole-prefix stale evidence is always the problem" is **not**
supported.

Projection reliability:

- events containing any >50 m GT-error observation: 2;
- events containing any >100 m GT-error observation: 2;
- events containing Jacobian condition >10: 2;
- events containing outside-tile observation: 10;
- events whose current candidate rank !=1: 16.

Descriptive harmful-update rates (>5 m):

- Jacobian condition >10: 2/2 harmful;
- outside-tile evidence: 1/10 harmful;
- current rank !=1: 1/15 comparable harmful;
- current observation GT error >40 m: 1/1 harmful.

The two major failures are the only events containing the severe q41-style
homography-conditioning / >100 m projection error.

This is strong evidence that homography reliability deserves further study, but
2/2 is **not enough evidence to freeze a universal hard Jacobian threshold**.

---

# 6. What the full run says about the architecture

## 6.1 Relative frontend

XFeat is imperfect but not the dominant source of the 79 m absolute error.

The relative drift remains roughly 2.3 m per 100 m and final relative error is
about 39 m after the historical prefix evaluation.

Keep relative-frontend work separate from absolute-state research.

## 6.2 Retrieval

Dense forest can be genuinely unobservable for the current global descriptor.

At q57 the useful geographic region is absent from Top-20. ORB cannot recover a
candidate that retrieval never supplies.

This motivates **abstention / observability detection** in addition to better
retrieval models.

## 6.3 ORB verifier and continuous sub-tile projection

ORB sub-tile measurements are often valuable.

The accepted-event median GT error is about 7.86 m.

The failure is not "ORB never works".

However, rare homography projections can be badly conditioned and can poison a
multi-observation similarity state.

## 6.4 minimum_confident_v2 state manager

The policy gates the current absolute observation against the active state's
current-position prediction.

But when accepted, the system installs an entire four-observation similarity
leader.

The **object being gated** and the **object being installed** are therefore not
identical.

q57:

- minimum current innovation: 4.061 m;
- tracking gate: 12.8 m;
- therefore accepted;
- installed transform moves the map prediction by about 50.8 m.

q228:

- minimum innovation: 2.512 m;
- active state is already wrong;
- a wrong observation can therefore be self-consistent with the wrong state.

This exposes two research questions:

1. **transform-transition innovation** — should the proposed replacement
   transform itself be tested against the active transform over current/recent
   visual probe points?
2. **state trust / recovery** — how can the system detect that the state it uses
   as the innovation reference may itself be wrong?

These are research questions, not benchmark-baseline patches.

---

# 7. Stage 7A research reconciliation

Existing research on `research/minimum-confident-bootstrap` remains highly
relevant.

Current evidence-based interpretation:

- candidate ambiguity / availability: strongly supported;
- transform-family persistence: partially supported;
- blind discriminator research: strongly supported;
- DINO-only discrimination: insufficient by itself;
- sub-tile localization: strongly supported;
- late evidence failure localization: strongly supported;
- generic Top-M expansion: not a universal solution;
- homography support/Jacobian reliability: strongly supported as a diagnostic,
  not yet a frozen hard gate;
- causal innovation: useful but incomplete;
- defer/HOLD: strongly supported;
- separate ACQUISITION and TRACKING: strongly supported;
- safe `NO_PROVISIONAL_LOCK`: still required;
- generic sliding-window evidence age: not justified by this run alone;
- altitude-conditioned scale: not the primary issue on this stable-120m
  trajectory, although it remains relevant to the previous altitude-varying
  validation sequence.

Current priority order:

1. state-transition safety;
2. absolute-observation / homography reliability;
3. low-observability detection and abstention;
4. robust multi-evidence similarity construction;
5. state recovery / re-acquisition;
6. candidate-generation improvement;
7. deployment/runtime optimization.

---

# 8. Important map-indexing clarification and future spatial-search direction

The current map database **already has a matrix topology**.

The tile generator:

`scripts/villoc/s8_9_generate_reference_tiles.py`

creates independent X/Y start arrays and iterates:

`for grid_row in y_starts: for grid_col in x_starts`

Each tile-index record stores:

- `grid_row`;
- `grid_col`;
- pixel offsets;
- EPSG:3346 bounds;
- center easting/northing;
- geographic center;
- tile width/height;
- edge flags.

Tile IDs are assigned in row-major order as `sat_000001`, `sat_000002`, ...

Therefore a new TMS/XYZ conversion is **not required merely to obtain
neighbourhood relationships**.

The missing capability is to use the existing `(grid_row, grid_col)` topology
during retrieval/tracking.

Potential future architecture:

GLOBAL ACQUISITION
    full descriptor index
        ->
provisional map state
        ->
LOCAL TRACKING
    cache/search tiles around predicted grid cell
        ->
confidence weakens
        ->
expand 3x3 / 5x5 / physical-radius neighbourhood
        ->
regional recovery
        ->
global re-localization if state is lost.

Critical warning:

A local spatial prior must never become a permanent prison. A wrong map state
would otherwise restrict the search to the wrong geographic region and prevent
recovery.

The spatial-search subsystem must therefore support:

- local search;
- expanding/ring/spiral search;
- regional search;
- global fallback/re-acquisition.

This is primarily a **map-search/runtime/recovery architecture** improvement,
not a substitute for global VPR quality.

TMS/XYZ may still be useful later for multi-resolution pyramids, standardized
web-map addressing or deployment interoperability, but it is not a prerequisite
for matrix-neighbour retrieval in the present EPSG:3346 tile database.

---

# 9. Multi-agent / multi-expert concept parking lot

A future architecture may benefit from multiple adaptive expert modules, but do
not introduce an LLM-style agent framework into the localization loop without a
clear experimental need.

Useful decomposition:

- relative-motion expert — XFeat;
- global-place expert — DINO/SALAD/other VPR;
- geometric-verification expert — ORB / learned matcher;
- observability expert — decides whether current imagery is informative for
  absolute localization;
- spatial-search expert — local/regional/global map search;
- map-state expert — acquisition/tracking/HOLD/recovery;
- resource manager — CPU/GPU/cache scheduling.

A deterministic state manager can arbitrate these experts.

Possible future learned adaptation:

- scene-conditioned retrieval policy;
- altitude/viewpoint-conditioned descriptor or thresholds;
- environment/season confidence;
- online state-confidence estimation;
- controlled learning from repeated missions.

Do not interpret "self-improving agent" as permission to mutate localization
logic or thresholds online without bounded training, validation, rollback and
safety constraints.

For the current research, "multi-expert adaptive state manager" is a more useful
engineering description than "AI agents".

---

# 10. Retrieval-backbone research parking lot

Do not swap backbones inside the native benchmark baseline.

Use a separate controlled retrieval study with the same 403 queries and same map
asset.

Current baseline:

- DINOv2 ViT-S/14 img518 center_square avgpatch.

Candidate future retrieval systems to benchmark separately:

- DINOv3 small / deployment-appropriate backbone;
- satellite-domain DINOv3 if protocol/license/deployment fit is confirmed;
- AnyLoc-style DINO aggregation;
- SALAD;
- FlatVPR as a research experiment if its training/adaptation requirements fit
  the project;
- classic NetVLAD mainly as a control rather than the first replacement.

Evaluate:

- Top-1 / Top-5 / Top-20;
- GT-containing-tile availability;
- <=40 m and <=80 m best-candidate availability;
- forest/canopy rescue;
- urban retention;
- descriptor-cache size;
- query latency;
- Jetson memory;
- downstream ORB candidate changes.

SuperPoint/LightGlue belongs primarily to local matching/verification rather
than global retrieval. It cannot recover q57 if the useful map region is absent
from the retrieved candidate set, and it does not directly fix q228's state
transition failure.

---

# 11. Planned native map-cache A/B/C experiment

Do not overwrite the canonical copied Mac cache.

Run controlled caches with distinct identity:

A. existing Mac CPU-built map cache — canonical benchmark baseline;

B. native Jetson CPU map-cache rebuild — isolates platform / decode /
preprocessing / CPU-architecture effects;

C. native Jetson CUDA map-cache rebuild — isolates execution-device effects
after B.

Compare:

- descriptor norms;
- descriptor max/mean absolute differences;
- cosine similarities;
- Top-1 identity;
- Top-20 membership;
- Top-20 order;
- candidate geographic availability;
- ORB reranker input;
- downstream selected tiles;
- map-state events;
- final frozen trajectory if a full replay is justified.

Because map descriptors start from the already-materialized orthophoto tiles,
MP4 video-decoder differences are not directly involved in the map-cache
experiment. Image decode/resize/library/platform effects can still matter.

---

# 12. Runtime work still pending

Before declaring deployment benchmarking complete, parse:

- per-stage orchestrator wall times;
- DINO query encoding contribution;
- cached Top-K retrieval cost;
- ORB Top-20 verification cost;
- XFeat frontend cost;
- bootstrap/map alignment/temporal/export/reporting cost;
- tegrastats average/peak RAM;
- swap;
- GPU utilization;
- temperatures;
- input/CPU-GPU/SOC power rails where available.

Do not state "ORB is the bottleneck" until the full stage breakdown proves it.

Known whole-run result:

- 1499.237 s;
- 403 queries;
- 0.269 queries/s;
- 3.72 s/query.

Known XFeat-stage wall time:

- 65.934 s for 403 frames / 402 pairs.

---

# 13. Evidence locations

## Jetson source-of-truth run

`/home/rodu/projects/drone-localisation-jetson-native/outputs/demo_runs/traj01_90deg_stable120m_blind_v2_jetson_001`

Benchmark evidence:

`/home/rodu/projects/drone-localisation-jetson-native/outputs/benchmarks/stage5_full403_jetson_001`

The Jetson run including extracted frames was approximately 1.6 GB, of which
the frame directory accounted for approximately 1.6 GB.

## Mac transferred non-frame run

`/Users/harishprabhu/Documents/drone-localisation-jetson-native/outputs/demo_runs/traj01_90deg_stable120m_blind_v2_jetson_001`

Transferred size without frames:

approximately 20 MB.

Transfer integrity:

70 non-frame files were hashed on the Jetson before transfer.

Post-freeze evaluation:

`/Users/harishprabhu/Documents/drone-localisation-jetson-native/outputs/benchmarks/stage6_postfreeze_gt_jetson_001`

Important Stage-6 files:

- `stage6b_postfreeze_position_errors.csv`;
- `stage6b_postfreeze_summary.json`;
- `stage6b_absolute_trajectory_vs_gt.png`;
- `stage6b_absolute_error_vs_time.png`;
- `stage6c_map_state_event_audit.csv`;
- `stage6d_jetson_cuda_xfeat_prefix_aligned_eval.csv`;
- `stage6d_jetson_cuda_xfeat_summary.json`;
- `stage6e_scene_context.png`;
- `stage6f_structural_event_audit.csv`.

---

# 14. Do not conflate these results

## Historical controlled fusion

Earlier Villoc experiments reported approximately 14.02 m fused RMSE under a
different controlled fusion/evaluation contract.

That is **not** the same estimator as the strict-blind
`minimum_confident_v2` causal map-state timeline evaluated here.

Do not interpret:

14.02 m vs 79.02 m

as a Jetson-vs-Mac hardware regression.

## Mac-vs-Jetson parity

Exact Mac/Jetson equality is useful only for isolating platform-sensitive
boundaries.

The research objective is not byte-identical cross-platform VPR.

Prioritize failure mechanisms that survive across platforms and use parity only
where it helps identify hidden sensitivity.

## Coverage vs accuracy

393/403 map-aligned rows means 97.5% coordinate availability.

It does **not** mean 97.5% correct absolute localization.

---

# 15. Parked Docker work

Docker branch:

`feature/dockerized-blind-eval-tool`

Checkpoint:

`74ed70ed2c267a44c79dcd2cd3562f2212c5fd32`

Known parked changes include:

- `docker/compose.runtime.yaml`;
- `docker/README.md`;
- `scripts/demo/docker_tool.py`.

Native Jetson benchmarking was deliberately performed before returning to
Docker.

Recommended return-to-Docker gate:

1. benchmark closeout committed;
2. full timing breakdown documented;
3. native map-cache A/B/C experiment documented;
4. no active native-baseline ambiguity remains.

Dockerization should package a known native baseline; it should not be used to
hide unresolved localization or performance questions.

---

# 16. Recommended next execution order

## Track A — finish native benchmark

1. parse full stage/runtime/tegrastats evidence;
2. run native map-cache A/B/C experiment;
3. freeze/document those results;
4. close the native benchmark branch or retain it as reference.

## Track B — retrieval research

Use a separate controlled experiment/branch:

1. current DINOv2 control;
2. DINOv3 candidate;
3. AnyLoc/SALAD candidate;
4. optional FlatVPR research experiment;
5. compare retrieval availability, runtime and memory before touching state
   logic.

## Track C — absolute-state research

Return to `research/minimum-confident-bootstrap` after benchmark closure.

Priority questions:

1. transform-transition innovation;
2. observation/homography reliability;
3. observability / abstention;
4. robust multi-evidence fitting;
5. state trust / recovery / re-acquisition;
6. interaction with spatial local/regional/global search.

Keep these experiments separate from retrieval-backbone changes so attribution
remains possible.

## Track D — deployment packaging

Resume Docker only after the native baseline and remaining native controls are
frozen, unless an external delivery deadline makes containerization the current
team priority.

---

# 17. Safe continuation protocol

Before any future work:

```bash
# Mac benchmark worktree
cd /Users/harishprabhu/Documents/drone-localisation-jetson-native
git status --short --branch
git log -5 --oneline

# Jetson
cd /home/rodu/projects/drone-localisation-jetson-native
source /home/rodu/projects/venvs/drone-localisation-jetson-native/bin/activate
export PYTHONPATH=$PWD/src
git status --short --branch
git log -5 --oneline
```

Never:

- delete a failed benchmark before recording why it failed;
- reuse an existing full-run ID;
- overwrite the canonical frozen map cache during A/B/C testing;
- attach GT before the blind freeze;
- promote post-freeze diagnostic thresholds directly into the benchmark
  baseline;
- change retrieval backbone and state-policy mathematics in the same experiment;
- resume Docker work in the benchmark worktree.

---

# 18. Resume prompt for a future ChatGPT session

Paste:

> Continue the GNSS-denied UAV localization project from the native Jetson
> full403 closeout on branch `benchmark/jetson-native-blind-baseline`.
> Read
> `docs/benchmarks/jetson_orin_nx_20261006_full403_closeout/README.md`
> before acting. The canonical full run is
> `traj01_90deg_stable120m_blind_v2_jetson_001`, produced from commit
> `cc81e8bd11e10d75cb749b8fbc21cc00a2a6dc14`.
> It passed 17/17 stages with CUDA XFeat/DINO queries, frozen CPU map cache,
> 403 queries and strict pre-freeze blindness. Post-freeze absolute RMSE is
> 79.02 m despite 393/403 coordinate coverage. Relative XFeat drift is
> 2.297 m/100 m. Dominant harmful state updates are q57 and q228; both reuse the
> bad q41 sub-tile observation (128.6 m GT error, Jacobian condition 25.3).
> q57 is a forest candidate-availability failure; q228 has useful DINO/ORB
> candidates but a bad downstream four-evidence state transform. Stage 6F
> shows severe Jacobian/projection failure is rare rather than universal.
> Do not patch the frozen baseline. Next native tasks are full timing/tegrastats
> parsing and the A/B/C map-cache experiment (Mac CPU vs Jetson CPU vs Jetson
> CUDA). Then reconcile retrieval-backbone research and
> `minimum_confident_v2` state-manager research separately. Docker work remains
> parked on `feature/dockerized-blind-eval-tool` at `74ed70e`.

---

# 19. Current conclusion

The native Jetson effort achieved its primary engineering objective:

**the frozen strict-blind pipeline can execute end-to-end natively on Jetson
with supported CUDA model inference and reproducible output freeze.**

The benchmark also produced a more useful research result than simple parity:

- relative visual motion remains usable but drifts;
- global retrieval becomes weak in dense canopy;
- many continuous ORB sub-tile observations are geographically useful;
- rare bad homography projections can poison multi-frame similarity states;
- current-observation innovation does not guarantee that the proposed
  replacement transform is safe;
- a wrong active state can make wrong observations appear self-consistent;
- spatial map topology is already available through `grid_row/grid_col` and
  can support future local/regional/global retrieval without redesigning the
  tile database.

The next phase should improve **evidence reliability, observability and
state-transition safety** while preserving the benchmark as an immutable
reference.
