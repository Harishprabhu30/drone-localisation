# SatLoc S7 — Relative Frontend and Candidate-Generation Upgrade Plan

**Project:** GNSS-denied UAV localization from onboard imagery and georeferenced satellite maps  
**Previous frozen stage:** S6 — Relative + Absolute Localization and Causal Drift Correction  
**New stage:** S7 — Relative frontend comparison, learned/structural retrieval, candidate fusion, and efficient verification  
**Development dataset:** SatLoc `traj01` for method development; later trajectories remain validation targets  
**Primary objective:** Increase correct-region coverage in Top-20 and Top-50 while reducing expensive LightGlue verification work  
**Status:** Planned — begin from **S7.0**

---

## 1. Why S7 is the correct next stage

S6 already satisfied and exceeded the minimum internship requirement:

```text
relative visual trajectory
+ measured drift
+ absolute satellite localization
+ confidence gating
+ temporal consistency
+ causal map-frame bootstrap
+ relative–absolute correction
```

The strongest remaining bottleneck is no longer whether fusion works. It is whether the correct satellite region enters the candidate pool early enough and often enough.

Frozen S5C/S6 evidence:

```text
Temporal absolute queries:       263
Union Oracle@50 availability:    102 / 263 = 38.8%
LightGlue selected hits:          68 / 263 = 25.9%
Balanced accepted corrections:    33
Causal post-lock hybrid fixes:     50
```

A local verifier cannot recover a correct map region that is absent from Top-50. Therefore, the next high-value question is:

> Can a better retrieval system place the correct region inside Top-20 or Top-50 more often, so fewer candidates need expensive geometric verification?

S7 also includes one bounded relative-front-end comparison because ORB produced strong local motion but substantial accumulated drift.

---

## 2. S7 is not another open-ended method collection

The stage is organized by system role.

### Relative motion frontend

```text
ORB
XFeat
optional SuperPoint ablation
```

Question:

> Which local feature system produces the lowest full-trajectory drift at acceptable runtime?

### Global candidate generation

```text
DINOv2 global descriptor
DINOv2 patch descriptors + VLAD / AnyLoc-style aggregation
structural PHLO stream
natural macro-structure stream
multi-scale satellite views
```

Question:

> Which descriptor places the correct map region inside Top-20 and Top-50?

### Candidate fusion and diversification

```text
rank fusion
score fusion
geographic non-maximum suppression
multi-stream union
```

Question:

> Can complementary streams improve recall without filling Top-K with redundant neighbouring tiles?

### Candidate verification

```text
LightGlue Top-50
LightGlue Top-20
LightGlue Top-10
```

Question:

> Can improved retrieval preserve or increase correction quality while reducing verification cost?

---

# Part I — Knowledge corrections and design principles

## 3. Vegetation is not simply noise

The goal is not to erase vegetation.

The correct objective is:

```text
suppress unstable fine natural texture
while preserving repeatable natural macro-structure
```

Potentially unstable information:

```text
individual leaves
small branches
wind-driven texture
seasonal colour
fine canopy shadows
high-frequency grass texture
```

Potentially stable natural anchors:

```text
forest outlines
tree-canopy mass and density layout
clearings
field boundaries
crop-row direction at suitable scale
riverbanks
roads crossing vegetation
isolated tree groups
natural/man-made interfaces
```

The system should therefore distinguish:

```text
fine transient texture
versus
persistent spatial organization
```

---

## 4. Proposed PHLO descriptor

The internal project name is:

# Pyramid Histogram of Line Orientations — PHLO

PHLO is a planned project descriptor rather than a claim that a standard published descriptor with exactly this name already exists.

The intended representation is:

```text
multi-scale image decomposition
        ↓
LSD or compatible line extraction at each scale
        ↓
cross-scale line consolidation
        ↓
line weighting by length, persistence, support and scale
        ↓
spatial pyramid / region layout encoding
        ↓
orientation histogram representation
        ↓
rotation-aware comparison without warping the full image
```

### Intended strengths

1. **Efficient rotation handling**

   Lines have explicit coordinates and orientations. Instead of rotating the complete image for every yaw hypothesis, the line set can be transformed mathematically and re-binned.

2. **Persistent natural anchors**

   Large forest boundaries, field edges and roads that survive across several scales can receive more weight than fine transient vegetation texture.

3. **Clutter rejection**

   Very short, low-support, single-scale segments can be downweighted or removed.

4. **Interpretability**

   Retrieved candidates can be explained through line overlays, orientation histograms, persistence maps and spatial-pyramid contributions.

---

## 5. Important rotation correction: shifting orientation bins alone is not always sufficient

A global line-orientation histogram can be compared under yaw by circularly shifting angular bins.

However, PHLO is planned as a **spatial pyramid**. When an image rotates:

```text
line orientation changes
and
the line midpoint moves to another spatial cell
```

Therefore:

> Circularly shifting only the orientation bins corrects direction but not spatial-cell assignment.

The efficient correct options are:

### Option A — Rotate the line set, not the image

For each yaw hypothesis:

```text
rotate line endpoints or midpoints around the image centre
rotate line orientation
reassign the transformed line to PHLO spatial cells
rebuild or update the descriptor
```

This remains much cheaper than image warping and repeated edge/LSD extraction.

### Option B — Dominant-orientation normalization

Estimate a stable dominant orientation from persistent long lines and rotate the line coordinate system once before PHLO construction.

Risk:

```text
natural scenes may have several equally strong directions
or no stable dominant direction
```

### Option C — Rotation-invariant polar spatial partition

Use radial rings and angular sectors around the image centre, then compare under circular sector shifts.

This better matches image rotation than a Cartesian spatial grid, but it changes the descriptor geometry.

### Recommended S7 implementation

Begin with:

```text
line-set yaw hypotheses
+ Cartesian PHLO
```

Then compare with:

```text
orientation-bin-only shift
```

This directly proves whether spatial reassignment matters.

---

## 6. Line and boundary weighting

A large line should not receive unlimited weight merely because it is long. Similar forest boundaries can cause perceptual aliasing.

Recommended weight:

```text
W =
    normalized length
  × cross-scale persistence
  × gradient/edge support
  × orientation stability
  × optional semantic-region support
```

Recommended safeguards:

```text
cap maximum single-line contribution
normalize contribution per pyramid cell
downweight repeated common orientations
retain junction/topology information
combine line evidence with learned appearance evidence
```

Possible additional structural features:

```text
junction count and type
parallel-line groups
line intersection topology
closed or near-closed boundary support
relative orientation pairs
region adjacency
```

---

## 7. Multi-scale natural structure

The scale-space branch should retain different information at different levels.

```text
Level 0 — fine/original:
    local texture and smaller boundaries
    low structural weight unless repeated

Level 1 — medium smoothing:
    roads, roof outlines, field edges, canopy clusters

Level 2 — coarse smoothing:
    forest outlines, clearings, large terrain partitions
```

A feature becomes more credible when it:

```text
persists across levels
maintains similar orientation
remains near the same projected location
has stable supporting gradients
```

The goal is not to let coarse forest boundaries dominate every query. They should become one stable stream alongside structural and learned evidence.

---

## 8. Preprocessing order

Recommended controlled ablations:

```text
P0 — original image
P1 — Gaussian scale pyramid control
P2 — Rolling Guidance Filter scale separation
P3 — anisotropic diffusion
P4 — post-extraction morphological consolidation
```

Priority:

```text
Gaussian control
then RGF
then morphological consolidation
then anisotropic diffusion only if needed
```

Morphological closing should generally operate on extracted edge/boundary maps rather than indiscriminately altering the original image.

No large blind parameter sweep should be started before the fixed diagnostic subset is understood.

---

# Part II — Primary technical directions

## 9. S7A — Relative frontend benchmark

### Purpose

Test whether XFeat reduces full-trajectory drift relative to ORB.

### S7A.0 — Freeze comparison protocol

Use the same:

```text
traj01 frame order
image resolution
stride-1 pairs
geometric model
RANSAC thresholds
trajectory integration
alignment/evaluation protocol
failure handling
metric definitions
```

### S7A.1 — XFeat sparse smoke test

Run a small continuous sequence first.

Collect:

```text
pair success
matches
RANSAC inliers
inlier ratio
forward/backward consistency if available
runtime
memory
failure examples
```

### S7A.2 — Full traj01 XFeat trajectory

Measure:

```text
RMSE
p95
maximum error
final error
drift per 100 m
failure rate
safe horizon
pair latency
full-run latency
```

### S7A.3 — ORB versus XFeat closeout

Decision rule:

```text
retain XFeat only if it gives materially lower drift,
better robustness,
or a strong accuracy/runtime deployment advantage
```

Do not spend several days tuning relative features if candidate generation remains the dominant fusion bottleneck.

---

## 10. S7B — Learned global retrieval

### S7B.0 — Freeze retrieval benchmark

Use the frozen 263 S5C temporal queries and all 8,625 satellite tiles.

Metrics:

```text
Recall@1
Recall@5
Recall@10
Recall@20
Recall@50
median correct-region rank
mean reciprocal rank
queries with no correct candidate in Top-50
query descriptor time
satellite cache time
descriptor memory
retrieval search time
```

Ground-truth coordinates remain evaluation-only.

### S7B.1 — DINOv2 global baseline

Test a simple global descriptor:

```text
CLS token
global average of patch tokens
optional GeM pooling
```

This establishes the learned global baseline.

### S7B.2 — DINOv2 patch-VLAD / AnyLoc-style retrieval

```text
DINOv2 patch descriptors
        ↓
visual vocabulary
        ↓
residual aggregation
        ↓
normalized global descriptor
```

This is the primary learned-retrieval experiment.

Required diagnostics:

```text
urban
mixed
natural/forest
agricultural/open field
low-structure
rotation-challenging
```

### S7B.3 — Multi-scale/FOV retrieval

Test whether the UAV and satellite views differ in effective field of view.

Candidate satellite cache variants:

```text
1.0× nominal crop
1.5× context crop
2.0× context crop
```

Possible retrieval approaches:

```text
separate scale caches
max score across scales
scale-aware rank fusion
coarse context retrieval followed by finer local tiles
```

### S7B.4 — Rotation augmentation for learned patches

Do not rotate all satellite images online.

Possible query-only strategies:

```text
rotate UAV query image for a small yaw set
or
rotate/reindex patch coordinates if the descriptor permits
```

This remains secondary to the PHLO line-set rotation experiment.

---

## 11. S7C — PHLO and natural macro-structure retrieval

### S7C.0 — Freeze structural diagnostic subset

Include examples from:

```text
urban successes
vegetation false positives
natural-scene failures
candidate-pool failures
selection failures
strong yaw/rotation mismatch
field-of-view mismatch
```

No method is promoted based only on visually pleasing decompositions.

### S7C.1 — Gaussian pyramid LSD baseline

At each scale:

```text
extract LSD
record length, orientation, midpoint and support
normalize coordinates
```

This is the control for later RGF experiments.

### S7C.2 — RGF scale-separated lines and boundaries

Extract:

```text
persistent long lines
macro-boundaries
road/forest interfaces
field boundaries
large canopy-region edges
```

Compare the number, length and repeatability of lines against the Gaussian control.

### S7C.3 — Cross-scale consolidation

Merge segments when they have:

```text
similar orientation
small perpendicular distance
overlapping projected extent
consistent position after scale normalization
```

Record:

```text
number of raw lines
number of consolidated lines
persistence level
retained total length
removed clutter
```

### S7C.4 — PHLO descriptor

Suggested initial configuration:

```text
orientation bins:       9 or 12
spatial levels:         global + 2×2 + 4×4
scale levels:           3
line weight:            length × persistence × support
normalization:          per-cell L1 or L2, then global normalization
```

The exact values are experimental and must be frozen in S7C.0 before the full benchmark.

### S7C.5 — Rotation-aware PHLO comparison

Compare:

```text
R0 — no rotation handling
R1 — circular orientation-bin shifts only
R2 — rotate line coordinates + orientations, then re-bin
R3 — optional dominant-orientation normalization
```

Recommended yaw hypotheses for the first controlled test:

```text
0°, 30°, 60°, ..., 330°
```

Later reduce the yaw set if recall is preserved.

### S7C.6 — Natural macro-texture companion descriptor

PHLO is strongest for explicit lines and boundaries. Add a companion descriptor for natural regions:

```text
coarse gradient layout
regional texture energy
multi-scale patch statistics
forest/field boundary occupancy
learned DINO patch distribution
```

Do not force all natural information into LSD.

### S7C.7 — Full 263-query promotion test

Only the best diagnostic variants are run across the full temporal benchmark.

---

## 12. S7D — Multi-cue candidate fusion

Candidate streams:

```text
DINOv2 global
DINO-VLAD / AnyLoc-style
PHLO
natural macro-texture
existing structural/domain-normalized PHOG variants
```

### S7D.1 — Score calibration

Normalize scores without using retrieval ground truth in the online calculation:

```text
z-score per descriptor stream
robust percentile normalization
temperature-scaled cosine similarity
rank normalization
```

Ground truth may be used afterward to compare calibration choices.

### S7D.2 — Rank fusion

Start with:

```text
reciprocal rank fusion
weighted reciprocal rank fusion
```

This is safer than immediate descriptor concatenation because streams have different dimensions and score distributions.

### S7D.3 — Score fusion

After calibration:

```text
learned appearance score
+ man-made structural score
+ natural macro-structure score
```

Begin with fixed weights. Scene-adaptive weights are later work.

### S7D.4 — Geographic candidate diversification

Problem:

```text
many adjacent tiles from one wrong region can occupy most of Top-K
```

Online-safe solution:

```text
take highest-scoring tile
group or suppress nearby overlapping tiles
retain limited representatives per map region
continue until Top-K contains geographically diverse hypotheses
```

Test:

```text
different suppression radii
1, 2 or 3 representatives per local region
Top-20 and Top-50 coverage
```

### S7D.5 — Best candidate-pool closeout

Primary target:

```text
new Recall@20 >= old Recall@50
```

Secondary target:

```text
improve Recall@50 without reducing natural-scene performance
```

---

## 13. S7E — Efficient LightGlue verification cascade

Compare:

```text
new retrieval Top-10
new retrieval Top-20
new retrieval Top-50
```

Measure:

```text
LightGlue selected hit rate
confidence-gated accepted corrections
precision, evaluation only
dangerous false corrections, evaluation only
verification time
candidates verified per query
memory
```

Candidate cascade:

```text
global retrieval
        ↓
cheap PHLO / structural rerank or fusion
        ↓
geographic diversification
        ↓
LightGlue Top-20
        ↓
confidence gate
        ↓
temporal confirmation
```

Do not claim time efficiency only from a smaller K. Measure actual wall-clock time.

---

## 14. S7F — Frozen fusion impact

Only after candidate generation clearly improves:

```text
rebuild S5C candidate/verification outputs
rebuild S6B.0 correction manifest
retain frozen causal fusion logic
rerun the causal post-lock replay
```

Do not tune fusion thresholds first.

Compare against the frozen S6 result:

```text
candidate Recall@20 / Recall@50
accepted corrections
correction blackout distance
post-lock RMSE
p95
maximum error
failure rate
runtime
```

This creates the clean causal story:

> Better retrieval produced more useful absolute corrections, which reduced relative trajectory drift.

---

# Part III — Stage blocks and stop rules

## 15. Full S7 block structure

```text
S7.0   Stage preflight and frozen benchmark specification

S7A    Relative frontend comparison
S7A.0  Protocol freeze
S7A.1  XFeat smoke
S7A.2  XFeat full trajectory
S7A.3  ORB/XFeat closeout

S7B    Learned global retrieval
S7B.0  Retrieval benchmark freeze
S7B.1  DINOv2 global
S7B.2  DINO-VLAD / AnyLoc-style
S7B.3  Multi-scale/FOV cache
S7B.4  Learned retrieval diagnostics

S7C    PHLO and natural macro-structure
S7C.0  Diagnostic subset
S7C.1  Gaussian pyramid LSD
S7C.2  RGF line/boundary extraction
S7C.3  Cross-scale consolidation
S7C.4  PHLO descriptor
S7C.5  Rotation handling
S7C.6  Natural macro-texture stream
S7C.7  Full benchmark promotion

S7D    Candidate fusion
S7D.0  Fusion manifest
S7D.1  Score calibration
S7D.2  Rank fusion
S7D.3  Score fusion
S7D.4  Geographic diversification
S7D.5  Top-K closeout

S7E    Efficient verification
S7E.0  Verification protocol
S7E.1  Top-10
S7E.2  Top-20
S7E.3  Top-50 reference
S7E.4  Accuracy/runtime closeout

S7F    End-to-end impact
S7F.0  Rebuild temporal absolute results
S7F.1  Frozen causal fusion replay
S7F.2  Final comparison and documentation
```

---

## 16. Priority order

1. **S7.0 benchmark freeze**
2. **S7A bounded XFeat comparison**
3. **S7B DINOv2 and DINO-VLAD**
4. **S7C PHLO rotation-aware structural stream**
5. **S7D rank fusion and geographic diversification**
6. **S7E Top-20 verification cascade**
7. **S7F frozen fusion impact**

Optional only after the primary chain works:

```text
SALAD
SuperPoint relative trajectory
anisotropic-diffusion sweep
scene-adaptive fusion weights
dynamic event-wise correction alpha
custom cross-view model training
```

---

## 17. Stop rules

### Relative frontend stop

Stop after ORB/XFeat full comparison unless there is a clear benefit.

### Structural preprocessing stop

Do not continue a filter because the image looks cleaner. Promote it only if:

```text
Recall@20 or Recall@50 improves
natural-scene retrieval does not collapse
rotation failures reduce
runtime remains acceptable
```

### PHLO stop

If line-set rotation does not improve recall over no rotation or DINO-VLAD already dominates every group, retain PHLO as an interpretable diagnostic stream rather than forcing it into the final pipeline.

### Learned retrieval stop

Do not custom-train a model during this stage unless zero-shot/frozen methods fail and adequate labelled cross-view data already exist.

### Fusion stop

Do not rerun S6 fusion for every retrieval ablation. Rerun only after a candidate system is clearly better.

---

# Part IV — Required visual outputs

## 18. Mesmerizing but technically meaningful visuals

For each selected UAV query, generate a story panel:

```text
A. UAV image
B. scale-space decomposition
C. raw LSD lines per level
D. consolidated persistent lines
E. PHLO spatial cells and orientation histograms
F. DINO patch-similarity or patch-cluster view
G. Top-20 map candidates
H. geographically diversified candidates
I. LightGlue correspondences for selected candidate
J. final map correction and trajectory effect
```

Required plots:

```text
Recall@K curves
correct-tile rank distribution
urban/mixed/natural group bars
Top-20 versus Top-50 coverage
retrieval time versus recall
verification candidates versus selected-hit rate
correction opportunity timeline
trajectory error before and after upgraded retrieval
candidate map scatter before/after diversification
PHLO yaw-response curve
cross-scale line persistence plot
```

The visuals must explain why a method works, not merely decorate the report.

---

# Part V — Company-facing story

## 19. Strong narrative

```text
1. Established reliable local visual motion.

2. Quantified accumulated drift over a long UAV trajectory.

3. Built full-map absolute retrieval and learned local verification.

4. Demonstrated confidence-gated relative–absolute correction.

5. Removed reference-derived map alignment through causal bootstrap.

6. Identified candidate-pool recall as the dominant remaining bottleneck.

7. Developed complementary learned, structural and natural-scene retrieval streams.

8. Reduced candidate verification cost through stronger Top-20 retrieval.

9. Re-ran the frozen fusion logic to show end-to-end trajectory impact.

10. Documented runtime, memory, failure cases and embedded recommendations.
```

This is stronger than presenting a list of tried methods.

---

## 20. Expected final technical contribution

The desired S7 result is:

```text
DINO-VLAD learned retrieval
        +
rotation-aware PHLO structural retrieval
        +
natural macro-structure retention
        +
geographic candidate diversification
        ↓
higher correct-region Recall@20 / Recall@50
        ↓
fewer LightGlue candidates
        ↓
more frequent trustworthy absolute corrections
        ↓
lower causal fused trajectory drift
```

---

# Part VI — S7.0 immediate starting block

## 21. S7.0 — Stage preflight and benchmark specification

Before coding a new descriptor, confirm:

```text
S6 README is present and treated as frozen
S5C 263-query manifest is unchanged
satellite index contains 8,625 tiles
GT fields remain evaluation-only
existing PHOG union and LightGlue results are preserved
hardware available for DINO/XFeat is recorded
all new methods write into new S7 output directories
```

Recommended directories:

```text
scripts/satloc/s7/
outputs/satloc/metadata/s7_retrieval_upgrade/
outputs/satloc/reports/s7_retrieval_upgrade/
outputs/satloc/figures/s7_retrieval_upgrade/
outputs/satloc/cache/s7_retrieval_upgrade/
docs/assets/s7_retrieval_upgrade/
```

S7.0 should produce:

```text
s7_0_stage_manifest.json
s7_0_query_manifest.csv
s7_0_method_registry.csv
s7_0_environment.json
s7_0_metric_specification.json
s7_0_preflight_report.md
```

No feature method should begin until S7.0 confirms the frozen inputs and evaluation rules.
