# A1 retrieval abstraction and parity closeout — 2026-10-07

Status: **CLOSED / PASS**

Branch:

```text
research/retrieval-candidate-pool-v2
```

Starting baseline:

```text
main @ 256595bed684c3bb6fbfccf7e0faaaeaec3d02f5
docs: hand off retrieval candidate pool v2
```

Frozen deployment reference remains unchanged:

```text
jetson-native-baseline-20261007
0f71a2d27ef299d78417904ca381815e9e7c9fd8
```

A1 changed retrieval infrastructure only. It did **not** change the DINOv2
representation, query preprocessing, map geometry, ORB verifier policy,
`minimum_confident_v2`, temporal authority, blindness rules, or frozen map-state
semantics.

---

## 1. Why A1 existed

The retrieval-candidate-pool-v2 research line needs to compare map footprints,
fusion strategies and eventually other representations without hard-coding a new
retrieval path for each experiment.

A1 therefore introduced the smallest stable boundary around the already-promoted
behavior:

```text
precomputed query/map representations
            ↓
RetrievalBackend
  load representation
  validate compatibility
  rank Top-K
  report backend metadata
            ↓
DinoV2CachedRetrievalBackend
```

The frozen DINO descriptor producer remains the existing S8.11BC implementation.
A1 did not move image preprocessing/model inference behind a new interface because
that would have enlarged the parity surface before it was needed.

---

## 2. A1 commits

```text
5b2f38a  refactor(retrieval): add A1 backend contract and parity gate
f95a865  refactor(retrieval): route DINO consumers through A1 backend
86dba06  test(retrieval): define A1 tie-equivalent artifact parity
64af233  test(retrieval): require schema parity in A1 comparator
```

Primary new implementation:

```text
src/uavloc/retrieval/
  __init__.py
  contracts.py
  dinov2_backend.py

scripts/villoc/retrieval/
  a1_dinov2_retrieval_parity.py
  a1_compare_candidate_artifacts.py

tests/
  test_retrieval_backend.py
  test_retrieval_artifact_parity.py
```

Maintained consumers routed through the backend:

```text
scripts/villoc/s8_11d_independent_dinov2_retrieval.py
scripts/villoc/blind_demo/stage6_blind_dino_topk_retrieval.py
```

---

## 3. Canonical mathematical parity gate

The canonical 403-query traj01 cache and frozen `512_s256` map cache were compared
using the historical NumPy ranking formula and the new backend.

Measured result:

```text
status                      PASS_A1_RETRIEVAL_PARITY
queries                     403
map tiles                   475
Top-K                       20
candidate ordering exact    true
scores exact                true
max absolute score delta    0.0
mismatched queries          0
q57 ordering preserved      true
q228 ordering preserved     true
```

This is exact parity at the ranking boundary, not approximate metric parity.

Local report:

```text
outputs/research_runs/retrieval_candidate_pool_v2/
  a1_parity/
    a1_dinov2_retrieval_parity.json
```

---

## 4. Maintained blind-consumer integration smoke

The migrated Stage-6 blind retrieval consumer was run through the new backend on
the full canonical 403-query cache.

Measured result:

```text
status              PASS_BLIND_DINO_TOPK_RETRIEVAL
queries             403
map tiles           475
descriptor dim      384
Top-K               20
candidate rows      8060

coordinates used    false
oracle used         false
GPS used            false
SRT used            false

Top-1 mean          0.8658517553256109
Top-1 median        0.8904029130935669
median Top1-Top2    0.004701375961303711
```

The blind output schema and leakage guards remained intact.

---

## 5. Important provenance discovery

An initial historical-output comparison appeared to fail:

```text
historical traj01 blind regression Top-K
        versus
new Stage-6 output from canonical dataset query cache
```

This was not a valid parity comparison because the two artifacts used different
query descriptor caches.

### Canonical dataset cache

```text
outputs/villoc/traj01_90deg_stable120m/descriptors/
s8_11c_dinov2_queries_v_1fps_dinov2_vits14_img518_center_square_avgpatch_cpu.npz
```

Source manifest SHA256:

```text
9284bb0c7a8c1710a8878a4f7e6a843297d730025d7494d7cccf519914e34485
```

Paths hash:

```text
f80b2b66714d9bbdf0d9470f4671920dabbcb3ebe227fd0a626341864223eac2
```

Protocol hash:

```text
4f945492686682bf8ee63ce23f7a0c8432f9227f233b2f2ced12194d6e5d700e
```

### Historical blind-run cache

```text
outputs/demo_runs/traj01_blind_regression_001/descriptors/
s8_11c_dinov2_queries_v_1fps_dinov2_vits14_img518_center_square_avgpatch_cpu.npz
```

Source manifest SHA256:

```text
f99fbe574328b5c408be46b3a45c3f3e162979fd84644ba420da1b94a4157f33
```

Paths hash:

```text
ecc4000cc771b507efce9f104d22c13aac06a6a9dbdc3d6147d4fcff8deefaaf
```

Protocol hash:

```text
056aac8e2a341811c7a74e30a8402813280e9a87abcd0fa4bae9957749280f88
```

The two caches had:

```text
descriptor shape exact       403 x 384 for both
query IDs exact              true
paths exact                  false
descriptor arrays exact      false
descriptor-exact rows        284 / 403
max descriptor abs delta     0.018493808805942535
mean descriptor abs delta    0.0004974890616722405
checkpoint SHA256 exact      true
DINO representation protocol same
```

The canonical S8.2 dataset path selects sampled frames through the parsed SRT
timeline/frame counters. The blind manifest cannot use SRT and samples directly
from video timing. Therefore identical DINO protocol tags do not imply identical
source images or descriptor arrays.

### Lesson

A descriptor tag identifies the **representation protocol**, not the complete
artifact provenance.

Future reproducibility metadata should bind at least:

```text
representation protocol
+ model/checkpoint hash
+ source manifest hash
+ ordered query IDs
+ ordered source paths / frame identity
+ ideally source-image/content identity
```

Do not compare retrieval artifacts as if they share the same inputs merely because
their descriptor tag strings are equal.

This is an A2 reproducibility lesson. It does not change the A1 algorithm.

---

## 6. Historical blind-artifact parity

The correct operational comparison reused the historical blind-run query cache
and the same frozen map cache, then ran the migrated Stage-6 consumer.

The resulting artifact matched the historical output except for one
floating-point near-tie:

```text
query                     q188
historical rank 3         sat_000314
historical rank 4         sat_000295
new rank 3                sat_000295
new rank 4                sat_000314
reported scores           approximately 0.914341 for both
maximum score delta       4.76837158203125e-07
Top-20 membership         unchanged
output schema             unchanged
```

The A1 handoff explicitly allows either frozen candidate ordering **or an
explicitly documented equivalent**.

A reusable artifact comparator was therefore added. It does not broadly relax
parity. A result is tie-equivalent only when:

1. output schema is identical;
2. query set is identical;
3. Top-K candidate membership is identical;
4. tile-matched scores differ by no more than `1e-6`;
5. moved candidates remain inside a contiguous reference-score tie bucket with
   adjacent score gaps no greater than `1e-6`;
6. no candidate crosses a meaningful score gap.

The historical replay passed this gate and its report was saved locally under the
A1 parity output tree.

The comparator intentionally fails:

- schema changes;
- candidate additions/removals;
- meaningful rank swaps;
- score changes beyond tolerance.

### Why no artificial deterministic tie-break was added

Changing production ranking solely to recreate one historical rank-3/rank-4
floating-point tie would itself change the frozen algorithm.

The correct action is to preserve the existing ranking formula and document
numerical equivalence at an explicitly bounded tie.

---

## 7. What A1 proved

A1 establishes all of the following:

```text
existing cached DINOv2 behavior can be expressed behind a stable backend
canonical 403-query candidate ordering is reproduced exactly
canonical scores are reproduced exactly
maintained retrieval consumers can use the same backend
blind retrieval boundaries remain intact
historical output differences can be classified without hiding them
the abstraction is ready for controlled retrieval research
```

A1 does **not** prove:

```text
better retrieval accuracy
better q57 observability
better q228 localization
map-scale improvement
backend/model superiority
real-time readiness
dependency portability
```

Those belong to later gated stages.

---

## 8. Development lessons to carry forward

### 8.1 Compare like with like

Before treating output differences as algorithm regressions, verify:

```text
query cache identity
map cache identity
source manifest identity
model/checkpoint identity
representation protocol
candidate evaluation definition
```

### 8.2 Separate exact parity from scientific equivalence

Use exact equality where deterministic equality is expected.

When crossing platforms, BLAS implementations, rebuilt descriptors or historical
serialization boundaries, define a tolerance **before** interpreting the result
and separately test decision equivalence.

Never silently convert approximate equality into "exact parity."

### 8.3 A stable interface should preserve semantics

The retrieval backend validates and transports the existing behavior; it does not
silently:

- normalize a previously unnormalized representation;
- change sort semantics;
- add tie-break rules;
- add geographic priors;
- reinterpret trust;
- use evaluation coordinates.

### 8.4 Evaluation remains outside retrieval

Coordinates, oracle labels and GT-derived errors remain downstream evaluation
inputs. Candidate generation stays blind-safe.

### 8.5 Keep infrastructure and research commits separate

A1 infrastructure is now closed before R1 changes the map representation. This is
important because any later metric change can be attributed to the research
variable rather than to the abstraction refactor.

---

## 9. Final A1 decision

```text
A1_RETRIEVAL_ABSTRACTION = PASS / CLOSED
```

The branch may proceed to:

```text
R1 — ground-footprint map pyramid
```

with these frozen controls unchanged:

```text
DINOv2 ViT-S/14
img518
center_square
avgpatch
same canonical query cache for the R1 scientific benchmark
same ORT10LT source raster / GSD
same cosine-dot-product ranking backend
same evaluation-only coordinate/oracle boundary
q57 named retrieval-failure diagnostic
q228 named downstream-control diagnostic
```

R1 must change map ground footprint deliberately and must not mix in query-crop,
descriptor-aggregation, DINOv3/SALAD, ORB-policy, bootstrap or state changes.
