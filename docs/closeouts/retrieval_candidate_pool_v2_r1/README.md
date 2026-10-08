# R1 ground-footprint map-pyramid closeout — 2026-10-08

Status: **CLOSED / PASS**

Branch:

```text
research/retrieval-candidate-pool-v2
```

A1 prerequisite:

```text
docs/closeouts/retrieval_candidate_pool_v2_a1/README.md
```

Frozen deployment reference remains unchanged:

```text
jetson-native-baseline-20261007
0f71a2d27ef299d78417904ca381815e9e7c9fd8
```

R1 asked one controlled question:

> With ORT10LT source GSD, query representation, DINOv2 protocol, retrieval
> backend and map-center spacing held fixed, how does represented map ground
> footprint affect candidate-pool quality and downstream blind localization?

R1 did not tune ORB, bootstrap, temporal authority, relative localization or
query preprocessing.

---

## 1. Experimental protocol

Primary fixed-stride map pyramid:

```text
variant       source window     footprint      stride
384_s256      384 px             76.8 m        256 px ≈ 51.2 m
512_s256      512 px            102.4 m        256 px ≈ 51.2 m   control
768_s256      768 px            153.6 m        256 px ≈ 51.2 m
1024_s256    1024 px            204.8 m        256 px ≈ 51.2 m
```

All levels use the same orthophoto source and are encoded at the same DINO
network input size:

```text
DINOv2 ViT-S/14
img518
center_square
avgpatch
ImageNet normalization
float32
L2 normalization
cosine / dot-product retrieval
NumPy descending argsort
```

The primary ablation intentionally holds source-raster center spacing fixed
instead of holding percentage overlap fixed.

---

## 2. R1.0 — geospatial preflight

Status:

```text
PASS_R1_MAP_PYRAMID_PREFLIGHT
```

Measured source raster:

```text
ORT10LT 2024–2026
raw CRS metadata:
  PROJCS["LKS94 / Lithuania TM", ... unnamed datum ...]
expected coordinate contract:
  EPSG:3346

source GSD:
  0.199999999999376 x 0.19999999999940396 m/px

fixed stride:
  ~51.2 m
```

Important reproducibility lesson:

The historical GeoTIFF does not retain enough datum authority metadata for
`pyproj` to declare formal EPSG:3346 equality. R1 therefore validates the
projected-coordinate contract instead:

- Transverse Mercator;
- central meridian 24°;
- latitude of origin 0°;
- scale 0.9998;
- false easting 500000 m;
- false northing 0 m;
- GRS80 ellipsoid;
- metric axis semantics;
- identity behavior at representative AOI points.

Measured AOI identity delta:

```text
0.0 m
```

Do not regress this to literal CRS-string comparison.

Predicted/validated pyramid counts:

```text
384_s256     500
512_s256     475
768_s256     432
1024_s256    391
```

---

## 3. R1.1 — missing map levels

Only the missing primary levels were generated:

```text
384_s256
  grid             20 x 25
  tiles            500
  footprint        76.80 x 76.80 m
  stride           51.20 x 51.20 m
  right anchored   true
  bottom anchored  true
  index SHA prefix d9faafbcf800

768_s256
  grid             18 x 24
  tiles            432
  footprint        153.60 x 153.60 m
  stride           51.20 x 51.20 m
  right anchored   true
  bottom anchored  true
  index SHA prefix d313af8eb182
```

S8.10A audit:

```text
variants audited       2
variants passed        2
variants failed        0
total failures         0

PASS_TILE_INDEX_INTEGRITY
PASS_R1_MISSING_LEVELS_GENERATED_AND_AUDITED
```

No 512 or 1024 control asset was regenerated.

---

## 4. R1.2 — DINO map caches

Status:

```text
PASS_R1_MAP_DESCRIPTOR_CACHES_BUILT_AND_VALIDATED
```

Measured:

```text
384_s256   shape [500, 384]   norm [1.000000, 1.000000]
            cache SHA prefix 3b9146bdb78d

768_s256   shape [432, 384]   norm [1.000000, 1.000000]
            cache SHA prefix 247a83143927
```

The canonical query cache SHA was checked before and after and remained
unchanged.

The R1.2 implementation reuses the promoted S8.11BC DINO builder rather than
duplicating preprocessing/model code.

Infrastructure lesson:

A manually created `importlib` module must be registered in `sys.modules`
before executing a dataclass-bearing module. The R1.2 loader now has regression
coverage for the real S8.11BC builder.

---

## 5. R1.3 — independent retrieval

Status:

```text
PASS_R1_INDEPENDENT_FOOTPRINT_RETRIEVAL
```

Canonical 403-query retrieval result:

| Variant | Contain R@1 | Contain R@5 | Contain R@20 | <=40 m @20 | <=80 m @20 | spatial NMS @51.2 m |
|---|---:|---:|---:|---:|---:|---:|
| 384_s256 | 0.277916 | 0.528536 | 0.729529 | 0.727047 | 0.821340 | 14 |
| 512_s256 | 0.362283 | **0.625310** | 0.831266 | **0.813896** | 0.846154 | 12 |
| 768_s256 | **0.429280** | 0.600496 | **0.883375** | 0.707196 | **0.848635** | 11 |
| 1024_s256 | 0.270471 | 0.459057 | 0.821340 | 0.598015 | 0.791563 | 10 |

Interpretation:

- footprint quality is non-monotonic;
- 512 remains strongest for tight <=40 m candidate availability and Top-5;
- 768 provides the strongest overall containment coverage and best <=80 m
  Top-20 availability;
- 1024 does not win globally but exposes context unavailable to smaller scales;
- 384 provides more spatial diversity but is dominated by 512 on the primary
  accuracy metrics.

### q57 retrieval diagnostic

Canonical first containing rank:

```text
384_s256     38
512_s256     29
768_s256     23
1024_s256    14
```

Increasing map context moves a geographically relevant q57 candidate toward the
front of the list.

### q228 downstream-control diagnostic

The 512 control already has:

```text
first containing rank   1
center error            21.22 m
```

Therefore q228 must not be described as primarily a retrieval-availability
failure.

---

## 6. Why R1.4 was added

R1.3 could not answer whether geographically useful larger-footprint candidates
survive ORB verification and the state estimator.

R1.4 therefore replayed the absolute-localization/state chain from one common
blind pre-retrieval checkpoint.

Common frozen inputs for every variant:

```text
blind query manifest
blind DINO query cache
blind XFeat raw relative trajectory
```

Only the map representation changed.

Replayed chain:

```text
DINO Top-20
 -> unchanged ORB hybrid verification
 -> unchanged minimum_confident_v2
 -> canonical causal map-state timeline
 -> unchanged map alignment
 -> unchanged temporal authority
 -> estimated output
 -> blind submission freeze
 -> post-freeze reference attachment/evaluation
```

No GT/reference information entered before the submission freeze.

---

## 7. R1.4 downstream result

Status:

```text
PASS_R1_DOWNSTREAM_TRANSLATION_REPLAY
```

| Metric | 512_s256 | 768_s256 | 1024_s256 |
|---|---:|---:|---:|
| maturity query | 9 | 11 | 11 |
| map-state events | 43 | 50 | 30 |
| map-aligned rows | 395 | 393 | 393 |
| accepted corrections | 43 | 50 | 30 |
| trajectory RMSE m | 77.209 | **27.368** | 68.503 |
| median m | 29.202 | **25.809** | 66.786 |
| p95 m | 51.023 | **43.853** | 93.586 |
| max m | 396.737 | **47.852** | 101.176 |
| final m | 396.737 | **29.299** | 61.512 |
| ORB-selected containing hits | 241 | **256** | 215 |
| ORB-selected <=40 m hits | **170** | 146 | 89 |
| ORB-selected <=80 m hits | **266** | 252 | 175 |

Relative to the 512 control, 768 reduces:

```text
RMSE   ~64.6%
p95    ~14.1%
max    ~87.9%
final  ~92.6%
```

The most important result is not merely lower average error: the 768 run avoids
the catastrophic state-family failure observed in the 512 run.

---

## 8. q390 — localized 512 catastrophic failure

Immediately before q390:

```text
q389 trajectory error   29.128901 m
active source query     q187
active scale            ~0.178876537
active rotation         ~-32.897072 deg
```

At q390 the frozen final policy
`activate_quarter_track_quarter` sees:

```text
tracking threshold      12.8 m
minimum innovation      11.047744 m
innovation tile         sat_000271
innovation choice rank  3

action                   TRACKING_ACCEPT
source update            q187 -> q390
active hypothesis        98785
```

The newly installed transform is:

```text
scale       0.102654643
rotation   +141.123018 deg
```

This is an approximately 43% scale collapse plus an approximately 174° rotation
family jump relative to the preceding active transform.

Trajectory error changes at the same effective query:

```text
q389    29.128901 m
q390   363.108192 m
q391   366.314389 m
...
q403   396.736551 m
```

No later map-state event recovers the trajectory.

R1 conclusion:

The frozen innovation gate can accept a transform that is locally/self
consistent with the current blind hypothesis machinery but belongs to a
globally wrong transform family.

This is a downstream state-safety limitation. It is **not** to be patched inside
R2 candidate-fusion experiments; changing candidate fusion and state policy at
the same time would destroy attribution.

---

## 9. q57 — availability versus geometric selection

### 512

R1.4:

```text
ORB-selected tile       sat_000100
DINO rank               1
inliers                 7
post-freeze center err  528.12 m
contains query          false
became state update     true
trajectory err at q57   28.23 m
```

The q57 state update is false absolute evidence, though the trajectory later
recovers to a plausible transform family before the q390 failure.

### 768

Blind q57 Top-20:

- no GT-containing candidate is present;
- closest candidate is `sat_000275`, DINO rank 16;
- ORB/hybrid promotes it to rank 1;
- 7 inliers;
- post-freeze center error ~81.01 m;
- q57 is not installed as a state update;
- trajectory error at q57 ~4.66 m.

So the 768 run is safe at q57 primarily because the state layer refuses to
install the imperfect absolute evidence.

### 1024

The blind q57 Top-20 contains a genuine rescue opportunity.

Post-freeze reconstruction identifies:

```text
sat_000223
  DINO rank       14
  hybrid rank     10
  verifier rank    7
  inliers          5
  contains GT      true
  center err      ~80.36 m

sat_000241
  DINO rank       15
  hybrid rank     16
  verifier rank   17
  inliers          5
  contains GT      true
  center err      ~13.70 m
```

But ORB/hybrid selects instead:

```text
sat_000352
  DINO rank        3
  hybrid rank      1
  verifier rank    1
  inliers          6
  center err     ~549.31 m
  contains GT      false
```

Therefore the 1024 q57 rescue is **not lost at retrieval**. It is lost at
geometric/reranking selection.

This is one of R1's strongest architectural findings:

```text
candidate available
    !=
candidate selected
    !=
candidate accepted into state
    !=
safe trajectory
```

All four boundaries must remain separately measurable.

---

## 10. q228 remains a downstream control

R1.3 / R1.4 continue to show useful q228 absolute evidence without q228 becoming
a state update.

Representative 512 evidence:

```text
containing candidate at rank 1
center error ~21.22 m
q228 state update false
trajectory error at q228 ~46.30 m
```

Do not count q228 as a retrieval rescue problem.

---

## 11. Final R1 decision

R1 does **not** promote one map footprint as universally optimal.

It assigns roles:

### Primary single-scale research reference — 768_s256

Why:

- best containment R@1 and R@20;
- best <=80 m Top-20 availability;
- highest ORB-selected containment count;
- best end-to-end RMSE;
- best p95;
- dramatically smaller maximum/final error;
- continues plausible map-state updates late in the flight;
- avoids the q390 catastrophic transform-family failure seen with 512.

### Precision/control expert — 512_s256

Retain because:

- frozen historical control;
- best Top-5 containment;
- best <=40 m Top-20 availability;
- highest ORB-selected <=40/<=80 m counts.

Do not promote as the new research default because the same downstream policy
can accept a catastrophic late transform from its candidate stream.

### Rescue/context expert — 1024_s256

Retain selectively because:

- global standalone performance is weak;
- ORB-selected candidate quality is substantially worse;
- state updates stop after q62 in this replay;
- but it exposes q57 rescue candidates that smaller scales do not bring into
  the operational Top-20.

### 384_s256

Keep as an experimental/diversity control but exclude from the first R2
production-budget fusion unless a unique-rescue analysis proves incremental
value.

---

## 12. What R2 must test

R2 remains **candidate fusion**, not state-policy redesign.

First fusion family:

```text
512_s256 ranks
768_s256 ranks
1024_s256 ranks
      ->
rank fusion (start with RRF)
      ->
physical-region / overlap-aware duplicate suppression
      ->
one merged Top-20
      ->
unchanged ORB Top-20 budget
```

Candidate records must preserve:

- scale / pyramid level;
- DINO per-scale rank and score;
- source window size;
- physical footprint;
- metric tile bounds;
- center easting/northing;
- fused score/rank;
- spatial-dedup provenance.

Primary R2 questions:

1. Can fused Top-20 preserve 512's tight candidates while gaining 768's broad
   containment?
2. Can 1024 unique rescue regions enter the merged budget without flooding it
   with coarse duplicates?
3. Does physical-region deduplication increase distinct useful regions per
   ORB budget?
4. Does fused ranking improve q57 candidate selection before ORB?
5. Does unchanged ORB preserve or discard the new rescue candidates?
6. Does downstream replay avoid the q390-class catastrophic state transition?

Question 6 is an evaluation requirement only. R2 must not change the frozen
state policy to force the answer.

---

## 13. Separate future state-safety research

R1 exposes a later research line that must remain separate from R2:

```text
transform-family discontinuity
cross-scale state consensus
rotation/scale jump sanity
multiple-hypothesis state persistence
recovery / rollback after late bad update
```

The q390 event is the canonical diagnostic for that future stage.

Do not silently add a q390-specific threshold or hand-tuned guard inside R2.

---

## 14. Final status

```text
R1_GROUND_FOOTPRINT_PYRAMID = PASS / CLOSED

primary research reference:
  768_s256

retained experts:
  512_s256 precision/control
  1024_s256 rescue/context

next stage:
  R2 cross-scale candidate fusion
```
