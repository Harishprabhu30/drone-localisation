# Native Jetson blind-pipeline baseline

Status: PREPARATION — Jetson execution and benchmarking pending.
Branch: benchmark/jetson-native-blind-baseline
Selected source baseline:
05ef9dd2c9bcf5bc68bdea946ee3d1900d155ebe

Parked Docker checkpoint:
74ed70ed2c267a44c79dcd2cd3562f2212c5fd32
Branch: feature/dockerized-blind-eval-tool

## Objective

Reproduce the frozen blind pipeline natively on Jetson before changing
localization behavior or resuming Dockerization. Verify functional and
numerical parity, measure deployment costs, and identify the bottleneck.

## Verified comparison evidence

Original Mac run: outputs/demo_runs/blind_recorded_flight_final_001
Execution: PASS; 17 planned and 17 completed stages.
Localization state: PROVISIONAL_ABSOLUTE_LOCK.
Documented maturity query: 30.
Submission: 123 rows; 29 without map positions; 94 with map positions.
Estimated lat/lon available: 94 rows.
Freeze record accepted_corrections: 1.
Secondary temporal fusion: disabled for this backend.
No pre-lock coordinate backfill.

Verified original submission SHA256:
734780591aa6438329f8992b362605f478b38c71d80ab800318afe95b46420a5

## Known geographic consistency limitation

The original freeze record reports 5 estimated map positions inside the
prepared AOI bounds and 89 outside, out of 94 available map positions.

This check uses prepared map geometry, not flight reference/GT.
It identifies a consistency concern but does not measure geographic error
or establish its cause. Provisional initialization is not certified accuracy.

Execution success and original artifact integrity are verified.
Geographic accuracy is not established by this verification.
Preserve the original behavior during deployment reproduction.

## Locked pipeline and blind boundary

Use configs/demo_villoc_blind_recorded.yaml.
Preserve minimum_confident_v2 and its temporal authority.
Preserve both hashed bootstrap contracts.
Preserve XFeat, DINOv2 ViT-S/14 img518 center_square avgpatch,
512_s256 map assets, ORB Top-20, and existing selection policies.
No new Stage 18D research changes or algorithm/threshold tuning.
No GPS/SRT/reference/oracle input before blind output freeze.
Never overwrite original comparison artifacts; use fresh run IDs.

## Deployment and parity protocol

Inspect Jetson model, RAM, OS, JetPack/L4T and runtime versions first.
Use Git for code and selectively transfer verified assets over SSH.
Build the Python environment natively; do not copy the Mac environment.
Record model weights, third-party revisions and asset hashes.
Perform CPU-path reproduction first; label CUDA measurements separately.
Define numerical comparison tolerances before Jetson execution.
Compare intermediate decisions and trajectories, not only summary counts.
The original SHA verifies original integrity; cross-platform numerical
parity must be assessed separately from byte-identical serialization.

## Benchmark scope

Measure DINO query encoding, cached retrieval, XFeat, ORB reranking,
bootstrap/map alignment/temporal routing, and complete execution.
Distinguish reusable offline work, per-flight work, and output/reporting.
Record cold/warm timing, repeated-run latency, CPU/GPU use, shared memory,
temperature, power mode and storage conditions.
Synchronize accelerator measurements where required.
Return to Docker only after the native results and bottleneck are documented.
