# Trajectory Adapter v1 — TA1 handoff

Status: **TA1 CLOSED / TA2 IMPLEMENTED — VALIDATION READY**

Branch:

```text
research/trajectory-adapter-v1
```

Base:

```text
3aa3ca33ec3c6bdb8df2922ea5bc3e2d8eff8a87
```

QV1 is frozen before this infrastructure stage.

## Purpose

Trajectory Adapter v1 separates dataset-specific ingestion from research
algorithms.

Research stages should eventually consume a canonical trajectory contract
instead of hard-coded traj01/demo paths.

Reference/GT is optional and is never part of the blind runtime contract.

## TA1 scope

TA1 implements only:

```text
canonical YAML contract
loader
validator
capability summary
blind runtime contract
path validation hook
unit tests
example trajectory spec
```

TA1 does NOT yet:

```text
extract frames
build canonical manifests
adapt traj01
adapt the blind demo
parse SRT
run QV1
run retrieval
run ORB
run state
```

Those are TA2+.

## Signal source contract

Signals are generic and future-facing.

Supported source modes:

```text
assumed
constant_metadata
per_frame_file
provider
unavailable
```

Current use:

```yaml
signals:
  relative_altitude_m:
    source: assumed
    value: 120.0

  gimbal_pitch_deg:
    source: assumed
    value: -90.0
```

Future IMU/VIO-style use can retain the same semantic signal name while
changing only its provider:

```yaml
signals:
  gimbal_pitch_deg:
    source: provider
    provider: vio_state
    field: camera_pitch_deg
```

TA1 validates the declaration only. It does not implement the provider.

This keeps the trajectory contract independent of whether a signal comes from
an assumption, metadata, an IMU/VIO frontend, or another future estimator.

## Reference contract

Supported reference modes:

```text
unavailable
postfreeze_optional
postfreeze_required
```

A reference path/provider is stored in the trajectory specification, but
`blind_runtime_contract()` exposes only:

```text
reference:
  available_to_localization: false
  mode: ...
```

It intentionally omits the reference path and payload.

This preserves the existing blind-demo rule:

```text
localize first
freeze/hash
attach reference later
```

## Trajectory roles

```text
development
validation
blind_stress
held_out
```

A held-out trajectory cannot declare
`allow_method_changes_after_run: true`.

## Canonical blind-manifest minimum fields

```text
trajectory_id
query_id
frame_index
timestamp_s
image_path
image_width
image_height
reference_available
```

TA2 will make the existing labeled traj01 dataset produce this canonical
contract and prove parity with the current 403-query research manifest.

TA3 will adapt the recorded blind demo with `reference.mode=unavailable`.

## TA1 gate

Run:

```bash
source .drone_venv/bin/activate
export PYTHONPATH=$PWD/src

python -m unittest discover -s tests -v

python scripts/trajectory_adapter/ta1_validate_trajectory_spec.py \
  --trajectory configs/trajectories/example_trajectory_v1.yaml
```

Expected:

```text
PASS_TA1_TRAJECTORY_CONTRACT
```

Do not use `--check-paths` with the example file; its data paths are
illustrative.

## Next

TA2:

```text
traj01_90deg_stable120m adapter
  -> canonical blind package
  -> exact query-count/time/image-path parity checks
  -> optional post-freeze SRT/reference capability
```


## TA1 result

Status:

```text
PASS_TA1_TRAJECTORY_CONTRACT
```

The example contract validates successfully, including:

- assumed altitude and gimbal-pitch signals;
- unavailable IMU/body-orientation signals;
- optional post-freeze reference capability;
- canonical blind-column contract;
- forbidden reference-column contract.

## TA2 — current labeled development trajectory adapter

TA2 adds:

```text
configs/trajectories/villoc_traj01_90deg_stable120m_v1.yaml
scripts/trajectory_adapter/ta2_adapt_traj01.py
tests/test_ta2_traj01_adapter.py
```

TA2 intentionally reuses the already-established 403-query blind manifest as
the parity source. It does not re-extract the video.

The trajectory spec still declares:

```text
raw video
1 Hz sampling policy
assumed relative altitude = 120 m
assumed gimbal pitch = -90 deg
768_s256 map contract
optional post-freeze SRT provider
```

TA2 reads no SRT/reference data.

It produces:

```text
blind_package/
  canonical_blind_manifest.csv
  trajectory_runtime_contract.json
  capabilities.json
  provenance.json

reports/
  ta2_traj01_adapter_report.json
```

Parity gate:

```text
403 queries
query IDs 1..403
frame indices 0..402
timestamps 0..402 s
3840 x 2160
exact legacy image-path parity
exact query-ID/frame/timestamp parity
reference_available=false in every canonical row
no forbidden GT/reference columns
```

The SRT path is declared in the trajectory YAML but is not read and is not
exposed in the blind package.

### TA2 validation command

```bash
source .drone_venv/bin/activate
export PYTHONPATH=$PWD/src

python -m unittest discover -s tests -v

python scripts/trajectory_adapter/ta2_adapt_traj01.py
```

Expected:

```text
PASS_TA2_TRAJ01_CANONICAL_ADAPTER
```

If TA2 passes, proceed to TA3: adapt the recorded blind demonstration trajectory
with `reference.mode=unavailable`.
