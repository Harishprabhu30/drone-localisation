# New-chat bootstrap prompt — current state

We are continuing Harishprabhu30/drone-localisation.

Do not treat this as a new localization project.

Active research branch:

research/crossview-scale-rotation-domain-v1

Before proposing code, read in this order:

1. docs/handoffs/CURRENT_BEST_VILLOC_LOCALIZATION_STACK.md
2. docs/handoffs/crossview_scale_rotation_domain_v1/FA1_OB1_CLOSEOUT.md
3. docs/handoffs/PHYSICAL_DRONE_TRANSITION_HANDOFF_20261016.md
4. docs/handoffs/crossview_scale_rotation_domain_v1/README.md
5. docs/handoffs/crossview_scale_rotation_domain_v1/CONTEXT_INDEX.md

Then inspect only the exact configs/scripts needed by the requested stage.

## Current frozen state

Primary development dataset:
villoc_traj01_90deg_stable120m, 403 queries.

Map:
768_s256, EPSG:3346, approximately 153.6 m footprint and 51.2 m stride.

Candidate generator:
DINOv2 ViT-S/14 img518 center_square avgpatch Top20.

Strongest development absolute frontend:
SIFT + LightGlue.

Frozen SIFT evidence SHA256:
c0031242f0b359f1bfa3c87ea47ac428393ed3e8e03d7b5c6d0822e8822cb8a0

Full403:
- containment 349/403
- center<=40 205/403
- center<=80 346/403
- selected projection<=40 367/403
- selected projection<=80 368/403
- unsafe 30/403

scale_m2 ORB remains the frozen classical control and a proposed challenger.

FA1:
- 19 oracle-rescuable SIFT failures
- 10 catastrophic SIFT damages
- 7 severe ORB<=40 -> SIFT unsafe
- temporal consistency and SIFT/ORB disagreement are the strongest diagnostic
  warning signals
- neighborhood candidate density is not the main missing signal
- manual residual failures are mostly vegetation-heavy/mixed vegetation

OB1.0:
- 112 oblique queries
- current 768_s256 + img518 stack
- Top20 contains body 112/112
- SIFT selected contains body 47/112
- selected tile-center <=40/<=80 = 4/44
- median tile-center 86.58 m
- substantial viewpoint degradation, but no retrieval collapse

Blind SIFT transfer:
- BV1.2 state PROVISIONAL_ABSOLUTE_LOCK
- first leader q8
- matured q11
- no SRT/GPS/reference during localization

## Current scientific boundary

- Do not tune SIFT/LightGlue on full403 truth.
- Do not create a confidence threshold from FA1 labels.
- Do not treat +/-2 FA1 temporal diagnostics as causal runtime logic.
- Do not implement "vegetation => ORB."
- Do not claim 45-degree projected image-center error is identical to UAV-body
  localization error.
- Do not claim thermal support; it is a future cross-spectral hypothesis.
- Do not claim true UAV-to-map physical GSD normalization is completed.

## Current project direction

Localization research is parked temporarily.

The next phase is the physical work drone.

Known prior bench context that must be revalidated and saved:
- Jetson Orin NX
- Pixhawk 6X
- Pixhawk link on enP8p1s0
- Jetson target 192.168.144.15/24
- Pixhawk 192.168.144.14
- ping previously passed
- MAVLink heartbeat previously passed
- system/component 1/1
- ArduPilot quadcopter
- firmware 4.7.1
- normal telemetry streams were not yet confirmed
- next planned test was requesting ATTITUDE at 2 Hz

The earlier hardware scripts/commands were not saved properly.

Immediate next action:
create and commit reproducible read-only platform/network/MAVLink/camera probes,
save their logs, then recover camera intrinsics/extrinsics/AGL/timestamps needed
for the physical UAV-to-orthophoto GSD experiment.

The user runs commands and pastes outputs. Never fabricate execution.
