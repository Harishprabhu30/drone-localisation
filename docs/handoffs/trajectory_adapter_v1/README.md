# Trajectory Adapter v1 — TA1 handoff

Status: **TA1/TA2/TA3 CLOSED / TA4 IMPLEMENTED — VALIDATION READY**

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


## TA2 result

Status:

```text
PASS_TA2_TRAJ01_CANONICAL_ADAPTER
```

Measured parity:

```text
trajectory:       villoc_traj01_90deg_stable120m
role:             development
queries:          403
query IDs:        1..403
frame indices:    0..402
timestamps:       0..402 s
image-path parity true
timestamp parity  true
reference declared true
reference read     false
```

The first real dataset therefore satisfies the canonical adapter contract.

## TA3 — recorded blind demonstration adapter

TA3 adds:

```text
configs/trajectories/villoc_blind_recorded_flight_final_001_v1.yaml
scripts/trajectory_adapter/ta3_adapt_blind_demo.py
tests/test_ta3_blind_demo_adapter.py
```

Trajectory role:

```text
blind_stress
```

Reference contract:

```text
reference.mode: unavailable
```

The frozen successful demo manifest is reused:

```text
outputs/demo_runs/blind_recorded_flight_final_001/metadata/blind_query_manifest.csv
```

Expected parity:

```text
123 queries
query IDs 1..123
frame indices 0..122
timestamps 0..122 s
3840 x 2160
assumed relative altitude 122 m
assumed gimbal pitch -90 deg
reference_available=false
```

No SRT/GPS/reference path exists in the TA3 trajectory spec, and TA3 requires
no ground truth.

The research policy for this trajectory freezes method changes after the run:

```yaml
research_policy:
  inspect_individual_failures: true
  allow_method_changes_after_run: false
```

This lets the flight serve as behavioral/stress evidence without pretending it
provides absolute-accuracy validation.

### TA3 validation command

```bash
source .drone_venv/bin/activate
export PYTHONPATH=$PWD/src

python -m unittest discover -s tests -v

python scripts/trajectory_adapter/ta3_adapt_blind_demo.py
```

Expected:

```text
PASS_TA3_BLIND_DEMO_CANONICAL_ADAPTER
```

If TA3 passes, TA4 will prove the next abstraction boundary: one
trajectory-independent frozen research harness consuming both TA2 and TA3
canonical packages, with accuracy metrics enabled only when post-freeze
reference exists.


## TA3 result

Status:

```text
PASS_TA3_BLIND_DEMO_CANONICAL_ADAPTER
```

Measured:

```text
trajectory:          villoc_blind_recorded_flight_final_001
role:                blind_stress
queries:             123
query IDs:           1..123
frame indices:       0..122
timestamps:          0..122 s
image-path parity:   true
timestamp parity:    true
reference declared:  false
reference read:      false
GT required:         false
```

TA3 proves that the canonical trajectory package does not depend on SRT/GT.

## TA4 — trajectory-independent frozen QV runner

TA4 is the first research harness that consumes the trajectory adapter rather
than hard-coded traj01/demo paths.

Frozen algorithm:

```text
QV1.4 center_unique_allview_fill20
map variant: 768_s256
query views: center / left / right / resize
DINOv2 ViT-S/14 img518 avgpatch
ranking depth: 20
final candidate budget: 20
center spatial-redundancy radius: 51.2 m
no ORB / bootstrap / state
```

For the development trajectory, TA4 reuses the frozen QV1 query descriptor
caches and requires exact pool parity with the previous QV1.4 result.

Expected development accuracy after blind pool freeze:

```text
contain R20 384 / 403
<=40 R20   277 / 403
<=80 R20   380 / 403
Top1 contain 174 / 403
```

For the blind demo, TA4 generates any missing query-view descriptors and runs
the same frozen candidate generator against the same 768_s256 map variant.

No absolute-accuracy evaluation is allowed for the blind demo.

Blind diagnostics include:

```text
center-vs-view Top20 Jaccard
final-pool source composition
center spatial-core size
alternate candidates added
consecutive final-pool Jaccard
center-Top1 map-space jump statistics
```

The blind-demo trajectory retains its historical 512_s256 primary map
declaration, but TA4 explicitly requests the additional 768_s256 variant so
that the frozen QV method is identical across both trajectories.

### TA4 development parity run

```bash
python scripts/trajectory_adapter/ta4_run_frozen_qv.py \
  --trajectory configs/trajectories/villoc_traj01_90deg_stable120m_v1.yaml
```

Expected:

```text
PASS_TA4_FROZEN_QV_WITH_REFERENCE
exact frozen QV1.4 pool parity: True
```

### TA4 blind-stress run

```bash
python scripts/trajectory_adapter/ta4_run_frozen_qv.py \
  --trajectory configs/trajectories/villoc_blind_recorded_flight_final_001_v1.yaml \
  --device cpu \
  --batch-size 1
```

Expected:

```text
PASS_TA4_FROZEN_QV_BLIND_STRESS
reference evaluation: False
```

The blind run will require DINO inference for missing left/right/resize/full
query-view caches, so it is expected to be materially heavier than TA2/TA3.
