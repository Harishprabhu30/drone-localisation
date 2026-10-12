# Geometric frontend roadmap after GD1.0

## Decision

Do not replace the frozen retrieval frontend yet.

QV1.4 remains the coarse candidate generator. The immediate research objective is
to discriminate the correct physical locality inside the frozen Top20.

GD1.0 shows that the historical condition "a homography exists" is saturated:
every development and blind-stress Top20 pair produced a homography and finite
projection. Therefore frontend comparison must be judged by candidate
discrimination, projection quality, robustness, latency, memory, and failure
behavior—not by homography success rate alone.

## Ordered plan

1. GD1.1: post-freeze analysis of the richer ORB evidence already frozen.
2. Keep the common geometric frontend adapter stable.
3. Benchmark learned frontends under the same Top20, map, query images, labels,
   and downstream evaluation.
4. Only then decide whether the frontend should be changed.

## Adapter contract

All future frontends implement a common measurement interface and expose:

- query/tile identity
- number of correspondences
- number of geometric inliers
- geometry-valid flag
- continuous projected coordinate when available
- projected-inside-tile flag
- frontend-specific evidence dictionary
- feature extraction time
- matching time
- geometry time
- total pair time
- device/backend metadata

The downstream reference attachment, candidate labels, and ranking evaluation
must remain frontend-independent.

## Learned frontend ladder

### ORB
Control. Lowest dependency/compute baseline and current historical contract.

### XFeat
Recommended first learned comparison because it targets efficient matching and
offers sparse/semi-dense operation. It is especially relevant to the eventual
edge-compute target.

### Sparse learned features + LightGlue
Use a stable detector/descriptor such as SuperPoint or ALIKED with LightGlue.
This tests whether learned correspondence reasoning improves Top20 candidate
discrimination without moving to a fully dense matcher.

### Detector-free / dense upper bound
LoFTR, EfficientLoFTR, RoMa, or a comparable robust dense method can be used as
an accuracy-oriented upper bound. These methods are likely more expensive but
can reveal whether sparse local features are fundamentally limiting the
cross-view UAV-to-satellite problem.

### OmniGlue
Optional domain-generalization baseline. Relevant because UAV imagery and
satellite orthophotos form a cross-domain matching problem.

Do not promote a frontend from one development trajectory alone.

## Benchmark dimensions

For every frontend:

- pair-level correct-vs-alias discrimination
- query-level Top1/Top3/Top5 locality recovery
- continuous projection error when the candidate is useful
- catastrophic projection tail
- correspondence spatial support
- robustness to repetitive/low-texture scenes
- cold/warm runtime
- feature-cache memory
- model weights
- CPU/GPU utilization
- Jetson feasibility
- deterministic/reproducible output behavior

## Public GPS-denied autonomy lesson

Public defense/autonomy descriptions should be treated as architectural
inspiration, not evidence of a specific proprietary matcher.

The relevant pattern is layered estimation:

visual/inertial odometry or inertial navigation
  -> short-term relative motion

terrain/map/scene matching
  -> intermittent absolute correction

confidence/health logic
  -> reject inconsistent absolute fixes

temporal/state estimator
  -> fuse corrections while preserving continuity

fallback/reacquisition
  -> recover when absolute matching becomes unavailable or ambiguous

This reinforces the current project direction: improve the quality and
confidence of absolute visual-map measurements, but do not design the system to
trust any single matcher or map candidate unconditionally.

## Current research hypothesis

A strong absolute correction should eventually require agreement between
several independent dimensions:

- global retrieval locality
- local correspondence geometry
- physically plausible projective transform
- cross-window/global-coordinate agreement
- temporal locality persistence
- relative-motion consistency

The next experiments should determine which of these dimensions are actually
independent and discriminative before they are fused into one acceptance score.
