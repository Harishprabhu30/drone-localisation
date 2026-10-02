# Docker deployment checkpoint — intentionally paused

Status: INCOMPLETE / PAUSED
Pause date: 2026-10-02
Branch: feature/dockerized-blind-eval-tool
Pre-checkpoint HEAD: 08e8d47cd4efa9f4806d9b4e2dae2284d440a89e

## Reason for pausing

First reproduce and benchmark the frozen blind pipeline natively on Jetson.
Establish functional parity, stage runtime, resource use and the deployment
bottleneck before returning to Jetson-aware Dockerization.

This checkpoint preserves unfinished deployment work. It does not declare
Docker end-to-end execution or Jetson compatibility complete.

## Existing committed work

- 63cad07: CPU container baseline.
- 08e8d47: offline runtime asset contract.
- Dockerfile targets a CPU environment; Compose specifies linux/amd64.
- Compose includes both project and XFeat directories in PYTHONPATH.
- Video and prepared map/model assets are mounted read-only.
- Compose disables runtime networking and uses separate writable output paths.

These execution properties have not been revalidated during this checkpoint.

## Pending work preserved here

- Compose selects drone-localisation:stage4-cpu.
- BLIND_VIDEO_PATH optionally overrides the mounted input video.
- The container sees the video at /workspace/inputs/flight.mp4.
- scripts/demo/docker_tool.py adds check and blind commands.
- The wrapper stages the video in container-local temporary storage.
- Existing output run directories are protected from overwrite.

## Validation limits and resume work

- Python syntax checking is not a container runtime test.
- Verify that stage4-cpu exists and contains this checkpoint's wrapper.
- Revalidate model loading, imports, mounts and the asset contract.
- The wrapper check rejects CUDA availability; it is CPU-container-specific.
- The blind command does not automatically invoke the check command.
- Validate all 17 stages and freeze contents beyond file-existence checks.
- Review temporary-video integrity, failure cleanup and recovery behavior.
- Full Docker parity with the frozen run remains to be established.
- ARM64, JetPack/CUDA compatibility and accelerator use are future work.

## Frozen native-baseline candidate

Candidate commit: 05ef9dd2c9bcf5bc68bdea946ee3d1900d155ebe
Evidence: docs/demo/blind_recorded_flight_final_001/README.md
Run: blind_recorded_flight_final_001
Recorded result: 17/17 PASS; PROVISIONAL_ABSOLUTE_LOCK at query 30;
123 queries; 94 map-aligned poses; no secondary temporal fusion.
Frozen submission SHA256:
734780591aa6438329f8992b362605f478b38c71d80ab800318afe95b46420a5

Verify this candidate against the configuration, contracts and original
frozen artifacts before creating the separate Jetson benchmark branch.
Do not derive that branch from this Docker checkpoint.

## Locked development rules

Preserve minimum_confident_v2, its contract hashes, temporal authority,
XFeat, DINOv2 preprocessing/model, ORB Top-20 and prepared map assets.
No new Stage 18D research changes or localization-policy tuning.
No GPS/SRT/reference/oracle information before blind output freeze.
Use fresh run IDs; never overwrite the original frozen comparison run.
Keep unrelated reporting/diagnostic WIP and existing stashes untouched.

## Conditions for resuming Dockerization

First document native Jetson functional parity, environment and asset
provenance, stage timings, resource measurements and the actual bottleneck.
Then resume this branch and implement the required Jetson deployment changes.
