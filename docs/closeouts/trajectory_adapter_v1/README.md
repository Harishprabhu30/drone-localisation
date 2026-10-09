# Trajectory Adapter v1 closeout — 2026-10-09

Status: **CLOSED / INFRASTRUCTURE MILESTONE COMPLETE**

Branch:

```text
research/trajectory-adapter-v1
```

Starting point:

```text
3aa3ca33ec3c6bdb8df2922ea5bc3e2d8eff8a87
docs/closeouts/query_view_candidate_generation_v1/README.md
```

## 1. Purpose

Trajectory Adapter v1 removes dataset-specific path/reference assumptions from
research algorithms.

The adapter introduces one canonical trajectory contract that supports:

```text
development
validation
blind_stress
held_out
```

and treats reference/GT as an optional post-freeze capability rather than a
localization requirement.

## 2. TA1 — canonical schema and loader

Status:

```text
PASS_TA1_TRAJECTORY_CONTRACT
```

Signal-source modes:

```text
assumed
constant_metadata
per_frame_file
provider
unavailable
```

This allows today's blind assumptions:

```yaml
relative_altitude_m:
  source: assumed
  value: 120.0

gimbal_pitch_deg:
  source: assumed
  value: -90.0
```

to become future IMU/VIO/provider-backed signals without changing the
trajectory schema.

Reference modes:

```text
unavailable
postfreeze_optional
postfreeze_required
```

The blind runtime contract never exposes the reference path/payload.

## 3. TA2 — labeled development trajectory

Status:

```text
PASS_TA2_TRAJ01_CANONICAL_ADAPTER
```

Canonicalized:

```text
villoc_traj01_90deg_stable120m

403 queries
query IDs      1..403
frame indices  0..402
timestamps     0..402 s
3840 x 2160
```

Exact image/query/frame/timestamp parity is preserved.

The SRT/reference is declared as post-freeze optional and is not read during
blind package creation.

## 4. TA3 — no-GT blind-stress trajectory

Status:

```text
PASS_TA3_BLIND_DEMO_CANONICAL_ADAPTER
```

Canonicalized:

```text
villoc_blind_recorded_flight_final_001

123 queries
query IDs      1..123
frame indices  0..122
timestamps     0..122 s
3840 x 2160
```

Reference:

```text
mode      unavailable
declared  false
read      false
required  false
```

TA3 proves that the adapter and blind research flow do not depend on SRT/GPS/GT.

## 5. TA4 — trajectory-independent frozen research runner

Frozen method:

```text
QV1.4 center_unique_allview_fill20
768_s256
DINOv2 ViT-S/14 img518 avgpatch
center / left / right / resize query views
fixed Top20
51.2 m center redundancy rule
no ORB
no bootstrap
no state
```

### Development trajectory

Status:

```text
PASS_TA4_FROZEN_QV_WITH_REFERENCE
```

Blind structural behavior:

```text
mean center unique         12.5087
mean alternates added       6.5980
median final-pool Jaccard   0.7391
median center-Top1 jump     0.0 m
```

Exact frozen QV1.4 pool parity:

```text
true
```

Post-freeze accuracy parity:

```text
contain R20   384 / 403
<=40 R20      277 / 403
<=80 R20      380 / 403
Top1 contain  174 / 403
```

### Blind-stress trajectory

Status:

```text
PASS_TA4_FROZEN_QV_BLIND_STRESS
```

Blind structural behavior:

```text
mean center unique         11.9756
mean alternates added       6.9350
median final-pool Jaccard   0.6667
median center-Top1 jump     0.0 m
```

No reference evaluation is performed.

## 6. TA4 cross-trajectory behavior comparison

Status:

```text
PASS_TA4_CROSS_TRAJECTORY_BEHAVIOR_COMPARISON
```

Blind-stress minus development:

```text
mean center unique         -0.5331
mean alternates added      +0.3369
median pool Jaccard        -0.0725
median Top1 map jump        0.0 m

center-vs-view Top20 Jaccard median:
  left                     -0.0529
  right                    +0.0000
  resize                   +0.0615
```

Interpretation:

The second flight is not absolute-accuracy validation because GT is unavailable.

It is valid behavioral/stress evidence that the frozen QV candidate generator
does not immediately enter a radically different construction regime on a
different real flight.

Do not convert this into an accuracy/generalization percentage.

## 7. Architectural result

Research code can now be organized as:

```text
trajectory YAML
  -> canonical blind package
  -> frozen research algorithm
  -> freeze/hash
  -> blind-safe diagnostics

if reference capability exists:
  -> post-freeze reference attachment
  -> absolute evaluation
else:
  -> no accuracy claim
```

Future trajectories should require primarily a new trajectory YAML plus adapter
execution rather than dataset-specific algorithm rewrites.

## 8. Future IMU/VIO compatibility

The trajectory contract already separates semantic signals from their source.

Therefore future work can change:

```text
gimbal_pitch_deg:
  assumed
```

to:

```text
gimbal_pitch_deg:
  provider -> VIO / IMU-derived state
```

without redesigning the dataset interface.

TA v1 does not implement VIO. It preserves the plug boundary.

## 9. Next algorithm stage

Return to localization research from the frozen candidate-generation interface.

Frozen input hypothesis:

```text
QV1.4 center_unique_allview_fill20
fixed Top20
region-level candidate generator
```

Next question:

> Given a candidate pool with strong region availability, can geometric
> evidence operate mainly as local candidate validation/refinement rather than
> as a global rescue reranker?

Next branch:

```text
research/region-candidate-geometry-v1
```

Keep state/bootstrap out of the first geometry experiments.
