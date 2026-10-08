# R2 cross-scale candidate fusion closeout — 2026-10-08

Status: **CLOSED**

Branch:

```text
research/retrieval-candidate-pool-v2
```

R1 closeout:

```text
docs/closeouts/retrieval_candidate_pool_v2_r1/README.md
```

Frozen deployment reference remains unchanged:

```text
jetson-native-baseline-20261007
0f71a2d27ef299d78417904ca381815e9e7c9fd8
```

R2 asked:

> Can a fixed-budget cross-scale candidate pool preserve the precision of the
> smaller map footprint while gaining useful context from larger footprints,
> before changing ORB or state policy?

---

## 1. R2.0 — fusion contract

R2 froze a neutral physical-region RRF contract:

```text
experts:
  512_s256
  768_s256
  1024_s256

per-scale depth:
  50

RRF:
  k = 60

physical duplicate rule:
  cross-scale only
  center distance <= 51.2 m
  bbox IoU >= 0.20
  at most one member per scale per region

final budget:
  one merged Top-20
```

Same-scale neighbors were deliberately not collapsed.

---

## 2. R2.1 — candidate-pool result

The pair `512+768` did not justify promotion.

The triple `512+768+1024` did.

Measured triple candidate-pool metrics:

```text
region containment R@20          0.908189
representative containment R@20  0.893300
region <=40 m @20                0.873449
representative <=40 m @20        0.754342
region <=80 m @20                0.900744
representative <=80 m @20        0.890819
```

Compared with the R1 768 single-scale reference:

```text
768 containment R@20             0.883375
768 <=40 m @20                   0.707196
768 <=80 m @20                   0.848635
```

R2 therefore established that cross-scale fusion can improve useful physical
region availability under the same Top-20 budget.

The gap between region metrics and representative metrics is itself important:
finding the correct physical region does not guarantee that the one concrete
tile forwarded downstream is the best member.

### q57

The triple fusion recovers the coarse-context q57 rescue and places
`1024_s256::sat_000241` at fused rank 19, with post-freeze center error about
13.7 m and GT containment true in the canonical-cache diagnostic.

---

## 3. R2.2 — unchanged ORB translation

R2.2 recomputed the promoted triple on the strict blind query cache and passed
one merged Top-20 through the unchanged historical ORB/hybrid verifier.

Aggregate selected-candidate result:

```text
triple fused+ORB:
  containing selections   253 / 403
  <=40 m                  133 / 403
  <=80 m                  252 / 403

R1.4 768 control:
  containing selections   256 / 403
  <=40 m                  146 / 403
  <=80 m                  252 / 403
```

The triple therefore did not beat 768 on aggregate ORB-selected precision.

Scale contribution remained genuinely mixed:

```text
512 representatives selected by ORB    163
768 representatives selected by ORB    159
1024 representatives selected by ORB    81
```

### q57

ORB still selected the known false 512 candidate:

```text
512_s256::sat_000100
~528.1 m center error
7 inliers
hybrid rank 1
```

So q57 retains a geometric-selection failure mode.

### q390

Fusion materially improved the ORB-selected evidence at the historical 512
catastrophic query:

```text
1024_s256::sat_000226
contains GT
~44.6 m center error
20 inliers
hybrid/verifier rank 1
```

This justified one unchanged downstream replay.

---

## 4. R2.3 — unchanged state replay

R2.3 reused the exact R2.2 fused Top-20 and ORB score artifacts by hash and
replayed only:

```text
minimum_confident_v2
 -> map alignment
 -> temporal authority
 -> export
 -> freeze
 -> post-freeze evaluation
```

No retrieval, ORB, bootstrap or state policy was changed.

Result:

```text
triple fused+ORB:
  maturity query       10
  map-state events     43
  RMSE                 356.668 m
  median               297.250 m
  p95                  592.625 m
  max                  623.187 m
  final                335.099 m
  first >100 m         q99
```

Single-scale controls:

```text
512_s256:
  RMSE 77.209 m
  p95  51.023 m
  max 396.737 m
  final 396.737 m

768_s256:
  RMSE 27.368 m
  p95  43.853 m
  max 47.852 m
  final 29.299 m

1024_s256:
  RMSE 68.503 m
  p95  93.586 m
  max 101.176 m
  final 61.512 m
```

Thus the triple must **not** be promoted end-to-end through the frozen state
stack.

---

## 5. R2.4 — q99 poisoning attribution

The first trajectory error above 100, 200 and 300 m all occurs at q99.

Immediately before q99:

```text
trajectory error ~3.771 m
```

At q99:

```text
TRACKING_ACCEPT
hypothesis 24244

scale                 0.173950509
rotation             -100.794662 deg
scale change          -3.07%
rotation jump         -69.17 deg
translation jump      343.39 m

trajectory error      480.448 m
error jump             476.677 m
```

Critically, the q99 fused+ORB candidate stream itself contains a good selected
observation:

```text
768_s256::sat_000309
contains GT
center error ~21.535 m
15 inliers
hybrid rank 1
verifier rank 1
bootstrap choice rank 1
```

The stable 768 run has the same q99 tile as its hybrid/verifier rank-1
candidate.

Therefore the q99 catastrophe is not explained by losing the correct tile at
retrieval or ORB selection.

---

## 6. R2.5 — innovation-gate / accepted-leader decoupling

R2.5 proved the state-policy mechanism at q99.

Current innovation gate:

```text
minimum innovation       2.0159 m
tracking threshold      12.8 m
innovation-best tile    768_s256::sat_000309
choice rank             1
```

But accepted leader hypothesis 24244 is supported by:

```text
q1    768_s256::sat_000195   choice rank 3
q39   512_s256::sat_000144   choice rank 4
q60   768_s256::sat_000154   choice rank 1
q99   512_s256::sat_000160   choice rank 4
```

The good q99 gate tile:

```text
768_s256::sat_000309
```

is **not** in the accepted leader hypothesis.

Measured:

```text
gate_tile_in_leader                    false
gate_choice_matches_leader_current_q   false
```

The frozen code uses:

```text
gate:
  minimum innovation over ANY valid current Top-4 observation

accepted object:
  independently selected blind Pareto leader hypothesis
```

The gate observation and accepted hypothesis are not required to be the same
object.

The stable 768 run at q99 instead has no current leader and performs:

```text
TRACKING_HOLD_NO_LEADER
```

preserving its safe existing transform.

---

## 7. Scientific interpretation

R2 has three different outcomes that must not be collapsed into one label.

### Candidate-pool fusion — PASS

The triple improves useful physical-region availability under a fixed Top-20
budget and exposes complementary large-footprint candidates.

### Existing ORB/hybrid selection — not improved globally

The unchanged ORB/hybrid layer does not convert the richer candidate pool into
better aggregate selected-candidate accuracy than the 768 single-scale
reference.

It also retains specific failures such as q57.

### Existing bootstrap/state integration — FAIL for the triple

The frozen `minimum_confident_v2` contract was developed around the historical
single-scale frontend. Feeding the richer mixed-scale stream creates additional
admissible hypotheses and exposes a gate/leader coupling failure.

The R2.3 trajectory failure must therefore not be interpreted as evidence that
cross-scale candidate fusion itself is bad.

---

## 8. What remains promoted

### Stable reference

```text
768_s256
```

Use it as the current single-scale operational/research control.

It remains the best demonstrated end-to-end stream under the existing
ORB/bootstrap/state stack.

### Experimental frontend asset

```text
512_s256 + 768_s256 + 1024_s256
physical-region RRF
fixed merged Top-20
```

Keep it for later frontend research.

Do not promote it unchanged into the historical bootstrap/state stack.

### Historical 512 control

Retain for reproducibility and regression.

### 1024 expert

Retain as a coarse-context/rescue expert rather than a standalone default.

---

## 9. Next research stage

Start a separate branch:

```text
research/absolute-frontend-decoupling-v1
```

Objective:

> Strengthen and measure absolute candidate selection before state estimation,
> and reduce the amount of candidate-rescue responsibility placed on the
> geometric reranker.

First controls:

```text
768 DINO Top-1
triple fused rank-1 representative
current ORB/hybrid selected candidate
```

Keep these boundaries distinct:

```text
candidate availability
 -> candidate selection
 -> geometric projection/refinement
 -> state acceptance
```

ORB reranking and ORB geometric projection are different responsibilities and
must be evaluated separately.

The frozen bootstrap remains a reproducible baseline, not a restriction on
future research. Do not modify it in-place.

---

## 10. Final status

```text
R2_CANDIDATE_FUSION = PASS / CLOSED
R2_END_TO_END_PROMOTION = FAIL

stable reference:
  768_s256

experimental richer frontend:
  triple 512+768+1024 RRF

next:
  absolute frontend decoupling / verifier-independence research
```
