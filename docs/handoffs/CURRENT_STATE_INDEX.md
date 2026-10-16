# Current state index

Use this file to decide what to read.

## If you need the strongest current architecture

Read:

1. CURRENT_BEST_VILLOC_LOCALIZATION_STACK.md
2. crossview_scale_rotation_domain_v1/FA1_OB1_CLOSEOUT.md

## If you need the experiment / decision trail

Read:

1. crossview_scale_rotation_domain_v1/README.md
2. crossview_scale_rotation_domain_v1/CONTEXT_INDEX.md
3. crossview_scale_rotation_domain_v1/FA1_OB1_CLOSEOUT.md

## If you are resuming the physical drone

Read:

1. CURRENT_BEST_VILLOC_LOCALIZATION_STACK.md
2. PHYSICAL_DRONE_TRANSITION_HANDOFF_20261016.md
3. crossview_scale_rotation_domain_v1/NEW_CHAT_BOOTSTRAP.md

## Current branch

research/crossview-scale-rotation-domain-v1

## Current research state

- SIFT+LightGlue is the strongest promoted development absolute frontend.
- scale_m2 ORB remains the strongest classical control and proposed challenger.
- Aerial-MASt3R compact upper bound is closed.
- FA1 residual failure analysis is closed.
- OB1.0 current-stack 45-degree stress is complete.
- confidence/abstention is not yet frozen.
- true physical UAV-to-orthophoto GSD normalization remains unfinished.
- localization research is parked while the physical drone is instrumented.


## Operationalization update

The current stack now has a canonical one-command reference-free runner:

~~~text
scripts/villoc/run_localization_pipeline.py
configs/localization/current_stack_reference_free.yaml
~~~

It preserves the historical ORB/demo orchestrator for reproducibility while
running DINOv2 Top20 -> SIFT+LightGlue -> Top4 minimum_confident_v2 on 768_s256.
Neutral project-facing outputs are written under metadata/, absolute/,
map_state/, trajectories/, provenance/ and summary/.

The next active development branch should be:
`hardware/work-drone-instrumentation-v1`.

Known Jetson MAVLink tooling already exists in a local `mavtools` workspace
and must be inspected/migrated rather than recreated. The immediate hardware
focus is camera/telemetry/network timing characterization followed by a
stationary/hover relative-estimation dataset.
