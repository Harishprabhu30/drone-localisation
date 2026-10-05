# Jetson Orin NX benchmark handoff — 2026-10-05

## Start here

**Next task:** run the complete `traj01_90deg_stable120m` blind pipeline with
`minimum_confident_v2` natively on Jetson Orin NX, enable supported CUDA model
execution, measure speed, freeze the blind submission, and only then evaluate
against the sequence's SRT/GT. Start a fresh run; retain the existing baselines.

Repository: `Harishprabhu30/drone-localisation`.
Handoff/next-work branch: `benchmark/jetson-native-blind-baseline`.
Base checkpoint on this branch:
`3dd33d0415cbeef75781bc37663d227cb4548c18`.
That commit's parent/source baseline is
`05ef9dd2c9bcf5bc68bdea946ee3d1900d155ebe`.

This handoff adds documentation, archived probes, and a read-only environment
inventory. It does not implement GPU support in the orchestrator or change
the bootstrap algorithm. Actual videos, caches, feature NPZs, and run outputs
remain on the user's machines; they are not included in this commit.

### Answer to the scope correction

We **did** run the recorded-flight CPU pipeline on Jetson: it reported
`NO_PROVISIONAL_LOCK`. We also ran controlled 30-query feature experiments on
Jetson that restored the Mac q30 lock. We **have not** run the full 403-query
traj01 v2 GPU orchestrator on Jetson. The later traj01 drift/guard experiments
were Mac research replays and do not explain the recorded-flight machine gap.

Continue from the evidence below; there is no need to restart project research.
Pause bootstrap redesign while establishing the native Jetson benchmark.

## Evidence and confidence

Run values below come from terminal output and reports the user supplied in
this conversation. GitHub code/configuration and the attached closeout READMEs
were inspected separately. The assistant has not executed these runs on the
user's Jetson or independently read every referenced local output file.
Before relying on an old result, reopen its report and verify its freeze hashes.

`PASS` in an implementation/contract report does not establish geographic
accuracy. A provisional lock is a state transition, not an accuracy certificate.
Post-freeze saved-transform replay metrics are not fused orchestrator metrics.

## Completed run matrix

| Dataset / test | Machine and scope | Result | Interpretation |
| --- | --- | --- | --- |
| Hidden-GT recorded flight, 123 queries | Mac, complete blind orchestrator | Provisional lock q30; one causal map-state event | Original comparison baseline; accuracy unestablished |
| Same recorded flight, 123 queries | Jetson, native CPU complete orchestrator | 17/17 stages passed; no provisional lock; maximum consistency streak 2 (requires 3) | Real Jetson no-lock result, despite successful retrieval and verification |
| Native Jetson recorded-flight queries | Jetson, DINO-only CUDA FP32 probe, 123 queries | All Top-1 identities equal to Jetson CPU; all Top-20 memberships equal | GPU model feasibility/parity probe, not a complete GPU pipeline |
| Copied Mac JPEGs + Mac Top-20 | Jetson, native ORB + either relative trajectory, 30 queries | No provisional lock | Copying JPEGs alone did not reproduce Mac query features |
| Mac query ORB features + native Jetson tile features | Jetson, 30-query rerank and bootstrap, Mac relative trajectory | Provisional lock q30 | Controlled input substitution; not deployment mode |
| Same Mac query features + native Jetson tile features | Jetson, 30-query bootstrap, Jetson relative trajectory | Provisional lock q30 | Relative-motion differences did not prevent this controlled lock |
| traj01, first 38 queries | Mac, R5.1 v2 research equivalence | All four policies lock q9; exact historical action/mode/source equivalence | Short-prefix research baseline |
| traj01, all 403 queries | Mac, R5.1 v2 research execution + post-freeze replay | All four policies lock q9; bad state replacement at q390 | Algorithm diagnostic; not Jetson benchmarking |
| traj01, all 403 queries | Jetson, complete native v2 GPU orchestrator | **Pending** | Next task |

### Recorded-flight CPU baseline

Jetson run:
`outputs/demo_runs/jetson_native_cpu_baseline_20261002_002`.
Reported total wall time: `1748.965 s` (about 29.15 min).
Bootstrap backend: `minimum_confident_v2`; state: `NO_PROVISIONAL_LOCK`;
map-state availability false; trust `NONE`; causal events 0.

Mac run: `outputs/demo_runs/blind_recorded_flight_final_001`.
Frozen submission: 123 rows, 94 with map positions, 29 without, one accepted
correction. The freeze's coverage check reports 5 inside prepared map bounds
and 89 outside. This does not supply hidden flight GT or quantify its error.
Submission SHA256:
`734780591aa6438329f8992b362605f478b38c71d80ab800318afe95b46420a5`.

### CUDA DINO probe

Output: `outputs/benchmark_probes/jetson_native_dino_cuda_123_20261005_001`.
Jetson-native frames; ViT-S/14; image 518; center-square; average patch tokens;
FP32; TF32 disabled; batch size 1; 475 CPU map descriptors reused.

| Measurement | Reported value |
| --- | ---: |
| Model load | 1.104 s |
| 123-query encoding wall time | 42.845 s |
| Median query time including preprocessing | 0.361 s |
| Mean query time including preprocessing | 0.348 s |
| Maximum descriptor absolute difference vs Jetson CPU | 4.321e-7 |
| Minimum descriptor cosine vs Jetson CPU | 0.9999999999968123 |
| Equal Top-1 identities | 123 / 123 |
| Top-20 overlap | 20 / 20 for every query |
| Torch peak allocated memory | 121.194 MiB |

The probe's `changed_rankings` list tests Top-1 identity and Top-20 membership;
it does not prove identical ordering of ranks 2–20. Torch allocated memory is
not total device/shared memory. These are DINO query timings, not whole-flight
runtime or validated real-time throughput. No full GPU XFeat timing is recorded.

## Recorded-flight reproducibility diagnosis

Two differences were isolated:

1. Native video extraction produced different pixels across machines despite
   the same MP4 and source-frame filenames. Both used OpenCV 4.10.0, with
   different FFmpeg builds. Decoding repeats were stable on each machine.
2. On identical copied Mac JPEGs, sampled query hashes match through decoded
   BGR, resize, YCrCb, and luma, then first differ at **CLAHE luma**. ORB query
   feature hashes differ downstream. This is a concrete preprocessing boundary
   to investigate later, not a proven cause within CLAHE's implementation.

The two sampled tile CLAHE images match, but tile ORB feature hashes still
differ. Swapping tile features did not alter those two pair results; that does
not establish global tile-feature parity.
Seed changes and optimization disabling left the shown results unchanged.
Jetson's one-thread run also matched its default. Mac's report labeled
`one_thread` still showed 4 threads; do not claim a valid Mac one-thread test.

| 600 fixed query/tile pairs | Native Jetson ORB on copied Mac JPEGs | Mac query features + native Jetson tiles |
| --- | ---: | ---: |
| Different good-match counts vs Mac | 20 | 0 |
| Different inlier counts | 68 | 2 |
| Different hybrid ranks | 472 | 13 |
| Maximum hybrid score difference | 3.624656 | 1.070336 |

With Mac query features, all four bootstrap policies locked at q30 on Jetson
using either relative trajectory. With Jetson relative motion, q28–30
innovations were approximately 7.256, 6.109, 6.785 m; streaks 1, 2, 3.
This strongly implicates query preprocessing/features in the recorded-flight
gap. A portable native preprocessing/ORB fix remains pending. Shipping Mac
feature files is an experimental control, not a solution for new flights.

The fixed-input DINO CPU/CUDA probe does not support TF32/MPS precision as the
cause of this ORB gap. It also does not prove every stage is portable.

## Machines, assets, and contracts

Jetson repository: `/home/rodu/projects/drone-localisation-jetson-native`.
Activate:
`source /home/rodu/projects/venvs/drone-localisation-jetson-native/bin/activate`.
Mac repository: `/Users/harishprabhu/Documents/drone-localisation`.
Mac environment: `.drone_venv`. Do not copy the Mac environment to Jetson.

Reported Jetson runtime: torch 2.8.0, CUDA build 12.6, CUDA available,
GPU `Orin`, capability `(8, 7)`; OpenCV 4.10.0, CUDA devices 0.
PyTorch CUDA availability does not give this OpenCV build CUDA support.
Record actual module paths: some torch warnings came from `~/.local` while
OpenCV-related dependencies were in the venv. The inventory script captures
import provenance; an environment reinstall has not been established as needed.
`rg` was unavailable on Jetson; use `grep` there if necessary.

traj01 video:
`data/raw/villoc/traj01_90deg_stable120m/villoc_traj01_90deg_stable120m_V_merged.MP4`.
Confirm its SRT's exact filename on the Jetson filesystem; do not infer or
attach it to blind stages. Sampling is 1 fps; prior Mac manifests have 403
queries. Confirm counts/time ranges in the fresh Jetson manifest.

Map: EPSG:3346, `ort10lt_2024_2026`, tiles `512_s256`;
tile index: `outputs/villoc/90_deg/metadata/s8_9_satellite_tile_index_512_s256.csv`;
map cache:
`outputs/villoc/90_deg/descriptors/s8_11b_dinov2_map_512_s256_dinov2_vits14_img518_center_square_avgpatch_cpu.npz`.
Model checkpoint path is defined by `CHECKPOINT` in the DINO cache builder;
inspect and hash the resolved weight file, not just its Python source.
Historical XFeat revision: `e92685f57f8318b18725c5c8c0bd28c7fe188d9a`;
verify the installed checkout and weights.

| Contract / source | SHA256 reported in this work |
| --- | --- |
| v2 architecture contract | `c47acc6313cd8d32a59b81d9457e29047192e7b9d6ef56524f39c4ce3208f93e` |
| v2 final policy contract | `ab074e4f63e126eefe34639beec4703edffb9fbef59f726f48d8c7a5759a4ab6` |
| ORB reranker source | `c2ed7e289719de457831f52d54b1d58ebf9c79b4cbb14f27105313af9e88fb3b` |
| v2 backend source | `ee29deb6ba432dbe5d22a1ae8460c7bea4de8851d8f3671f1c1c1e1c576eaa83` |
| R4.11 projection source | `393a93f30190e0e8e5a6ae7041e650478428ac4b79547ce7dec507c3c1426180` |
| DINO builder source reported by probes | `e02237f71b3b66baff9d388264a5a14a71112c14a26ff1d75de802fb8f287419` |

## Next-session execution order

1. Inspect the Jetson working tree and environment through its terminal/SSH.
   Preserve local changes. Record commit, dirty status, OS/JetPack/L4T, power
   mode, thermal state, module paths, model/map/video hashes, and storage.
   This handoff did not establish an SSH session or execute on Jetson.
2. Prepare a **new traj01 config and unique output/run ID**. Explicitly copy
   the complete `bootstrap` stanza from `configs/demo_villoc_blind_recorded.yaml`.
   The existing `configs/demo_villoc_traj01_blind_regression.yaml` lacks that
   stanza; Stage 07 defaults to `legacy_strict` when omitted. Its description
   also names the wrong recorded-flight clip. Correct it in the new config.
3. Inspect GPU device plumbing before running. The DINO cache builder exposes
   no `--device` option and explicitly rejects non-CPU protocols. Implement an
   isolated, reviewable CUDA path with device/dtype/precision in cache metadata
   and distinct cache identity; preserve the frozen CPU path. Do not relabel
   CUDA-produced descriptors with an unqualified `_cpu` provenance. Inspect
   XFeat's frontend and orchestrator device resolution too. ORB stays on CPU
   with the currently installed OpenCV.
4. Run a short blind device smoke test, then the full 403-query native GPU
   pipeline. Existing DINO parity evidence avoids repeating the whole
   recorded-flight CPU study. A traj01 CPU comparison can be added if needed
   to isolate a concrete discrepancy; it need not block the GPU benchmark.
5. Save each stage log and timing, final resolved config, manifest, intermediate
   outputs, submission, and freeze. Report no-lock honestly if it occurs;
   do not change thresholds to manufacture a lock. Diagnose the first
   divergent stage on Jetson after the baseline is captured.
6. Attach traj01 SRT/GT **only after freeze**. Verify source-frame/time alignment
   and coordinate conversion. Evaluate the actual frozen exported trajectory:
   available rows, lock latency, RMSE/median/p95/final error, accepted updates,
   AOI consistency. Do not use reference to regenerate localization.
7. Benchmark cold model load, warm query latency, synchronized CUDA time,
   per-stage wall time, total flight wall time, throughput, shared memory,
   utilization, power mode, temperature, and reporting costs. Separate reusable
   map preparation from per-flight inference. State whether map descriptors
   were reused from CPU or rebuilt on GPU. Sampling at 1 fps is not a runtime
   performance result.
8. Check repeatability on the same Jetson and compare with Mac using identical
   code/config/assets where possible. Exact hash agreement is useful for fixed
   input boundaries; numerical and decision parity need stated tolerances.
   Only then prioritize portability fixes or algorithm changes.

### Safe first commands on Jetson

```bash
cd /home/rodu/projects/drone-localisation-jetson-native
source /home/rodu/projects/venvs/drone-localisation-jetson-native/bin/activate
git status --short
git rev-parse HEAD
python scripts/benchmark_probes/jetson_orin_nx_20261005/environment_inventory.py
python scripts/demo/run_recorded_flight_demo.py --help
python scripts/villoc/s8_11bc_build_dinov2_caches.py --help
```

Fetch the handoff branch first if its inventory file is not installed; inspect
local changes before switching branches. Do not execute an old config assuming
it is v2 or assuming `--device cuda` is supported by the DINO builder.

## Parked traj01 accuracy investigation

Keep this evidence for later algorithm work; do not promote the tested guards
into the Jetson benchmark baseline.

Mac 38-query research root:
`outputs/research_runs/minimum_confident_bootstrap/traj01_blind_r5_1_v2_equivalence_001`;
freeze SHA256:
`300933c4e0d2f67d8b8ffadc702830b5e356e51f6f7695d8f998a788b13dca41`.
All four policies matched the frozen historical equivalence timeline.

Mac full403 root:
`outputs/research_runs/minimum_confident_bootstrap/traj01_blind_r5_1_v2_full403_20261005_001`;
freeze SHA256:
`0983aa281f0d8aa52f9caefec3d7be4b52eb15975b7a19bc51fc5a165a5ba792`.
403 queries, 1612 valid Top-4 projections, 235 leader updates, q9 maturity.
Reference attachment used by post-freeze diagnostics:
`outputs/demo_runs/traj01_blind_regression_001/evaluation/reference_attachment.csv`.

For `activate_quarter_track_quarter`, the saved-state replay had q9–38 RMSE
6.910 m; q9–403 RMSE 77.227 m; final error 396.759 m, with 42 tracking accepts.
At q390, the minimum-innovation gate matched `sat_000271` at 11.048 m, but the
replacement leader used `sat_000404` at the current query, with 357.948 m
endpoint innovation. The state jumped 334.928 m and rotation changed about
174.02 degrees. This identifies a gate/selected-leader consistency failure:
the observation satisfying the gate need not belong to the leader installed.

Exploratory 51.2 m state-jump and/or 90-degree rotation guards blocked only
q390 in this replay, preserved the helpful q134 update, and yielded RMSE
31.789 m/final 31.918 m. The earlier tight leader-endpoint guard blocked both
q134 and q390 (RMSE 31.643 m), sacrificing an otherwise helpful q134 correction.
These variants were developed after viewing this sequence's evaluation.
Blind replay decisions did not read GT, but the variants are development
diagnostics and need independent validation. No production policy was changed.

## Historical project context

Read the repository's current root README and relevant closeout READMEs before
changing protocol. Earlier work progressed through full-map ORB/structural
retrieval, learned matching, relative/absolute fusion, XFeat, Villoc map
preparation, DINO retrieval, ORB verification, and finally blind orchestration.
The evaluated Villoc full-pipeline/fusion results and blind v2 research replay
have different inputs and contracts; their metrics are not interchangeable.

Historical branches mentioned in the closeout sources:
`demo/blind-villoc-recorded-flight`, `research/minimum-confident-bootstrap`,
and parked `feature/dockerized-blind-eval-tool` (checkpoint
`74ed70ed2c267a44c79dcd2cd3562f2212c5fd32`). This is a context index, not a
claim that every historical commit/branch has been audited in this handoff.
Keep orchestrator entry points thin; operational blind code lives under
`scripts/villoc/blind_demo`; reference diagnostics belong outside online stages.

## Archived scripts and new-chat prompt

See [the probe archive](../../../scripts/benchmark_probes/jetson_orin_nx_20261005/README.md)
and its SHA256 manifest. The historical probes have fixed paths and dependencies;
read their archive instructions before executing. They are not a generic
traj01 GPU runner. Available scripts are preserved; terminal reports are
summarized here, not uploaded as independently verified raw artifacts.

Paste into the next chat:

> Continue the Jetson Orin NX native benchmark in
> Harishprabhu30/drone-localisation, branch benchmark/jetson-native-blind-baseline.
> Read docs/benchmarks/jetson_orin_nx_20261005/README.md at the handoff commit
> before acting. The next task is a fresh full traj01_90deg_stable120m blind
> minimum_confident_v2 run on Jetson with supported GPU execution and stage/full
> runtime measurements, followed by evaluation only after freeze. Inspect the
> current Jetson checkout/config/device plumbing first. Recorded-flight Jetson
> CPU no-lock is already established; DINO-only CUDA parity passed. Full traj01
> v2 GPU execution is pending. Keep the exploratory continuity guards parked.
