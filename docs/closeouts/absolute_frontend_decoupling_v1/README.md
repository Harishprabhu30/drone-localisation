# Absolute frontend decoupling v1 closeout — 2026-10-08

Status: **CLOSED**

Branch:

```text
research/absolute-frontend-decoupling-v1
```

Starting R2 closeout:

```text
docs/closeouts/retrieval_candidate_pool_v2_r2/README.md
```

Stable operational/research reference remains:

```text
768_s256
```

Experimental richer candidate source retained:

```text
512_s256 + 768_s256 + 1024_s256
physical-region RRF
```

## 1. Research question

This branch asked:

> Given the stronger R2 cross-scale candidate pool, can absolute candidate
> selection be made reliable before ORB reranking and state estimation?

It deliberately separated:

```text
candidate availability
  -> candidate selection/confidence
  -> geometric projection/refinement
  -> state acceptance
```

No new ORB or bootstrap/state policy was promoted.

## 2. R3.1 — direct retrieval-only selection

Direct rank-1 result:

```text
768 DINO Top-1:
  contains       174 / 403
  <=40 m         101 / 403
  <=80 m         171 / 403
  median error   256.837 m
  p95 error      863.389 m

triple fused rank-1 representative:
  contains       127 / 403
  <=40 m          80 / 403
  <=80 m         125 / 403
  median error   461.397 m
  p95 error      862.965 m
```

Conclusion:

The triple's value is deeper candidate availability, not raw fused rank-1.

## 3. R3.2 — support-count rescue

Cross-scale support count was tested as a blind rescue signal around 768.

Result:

```text
768 strict-support rescue:
  switches        38
  improved         4
  worsened        34
  contains       145 / 403
  <=40            83 / 403
  <=80           142 / 403
```

Three-scale-only rescue was identical.

Conclusion:

```text
cross-scale support count alone is not a trustworthy confidence signal
```

Correlated map aliases can receive multi-scale support.

## 4. R3.3 — static retrieval confidence audit

Post-freeze diagnostic only.

Discordant cases:

```text
containment:
  anchor-only good   63
  fused-only good    16

<=80 m:
  anchor-only good   65
  fused-only good    19
```

Best static diagnostic:

```text
anchor_fused_rrf_score

containment AUC* ~0.795
<=80 m     AUC* ~0.791
```

Other useful but weaker static signals included fused member ranks and the 768
Top-1/Top-2 DINO margin.

No threshold was fitted or promoted.

## 5. R3.4 — causal temporal retrieval audit

Causal windows:

```text
previous 3 queries
previous 5 queries
```

Physical radii:

```text
51.2 m
102.4 m
```

Best temporal diagnostics:

```text
containment:
  fused nearest-distance mean, w3
  AUC* ~0.729

<=80 m:
  fused nearest-distance mean, w3
  AUC* ~0.751
```

These are weaker than the best static R3.3 diagnostic:

```text
containment: 0.729 < 0.795
<=80 m:     0.751 < 0.791
```

Named cases show why temporal persistence is unsafe as a rescue rule:

```text
q57:
  wrong location has perfect recent persistence

q99:
  wrong fused region has stronger recent rank-weighted support
  than the correct 768 anchor

q390:
  useful fused region is temporally weaker than the wrong 768 anchor
```

Therefore short-horizon retrieval persistence can reinforce aliases and must not
be promoted as a selector from this evidence.

## 6. Closeout interpretation

This branch is closed with:

```text
direct 768 selection baseline:
  retained

triple fused rank-1 replacement:
  rejected

support-count rescue:
  rejected

static confidence threshold:
  not promoted

short-horizon temporal rescue:
  not promoted
```

The R2 triple remains scientifically useful because it improved candidate-pool
availability. The failure here is the inability of simple static/temporal blind
signals to identify the right physical region reliably enough.

This is a stopping condition for selector tuning on the current representation.

## 7. Next stage

Return to the originally planned candidate-generation / query-representation
axis.

Next branch:

```text
research/query-view-candidate-generation-v1
```

Question:

> Does the historical center-square query preprocessing discard visual context
> that is important for map retrieval?

Keep frozen initially:

```text
DINOv2 ViT-S/14
img518
avgpatch
768_s256 map representation
cosine retrieval
Top-K evaluation contract
```

Change only query representation.

Initial bounded study:

```text
center_square              [historical control]
resize_square/full-frame   [distortion ablation]
left/center/right square crops
  -> independent descriptors/ranks
  -> neutral query-view rank fusion
```

Do not change map footprint, descriptor backend, ORB, bootstrap or state in the
same experiment.

## 8. Final status

```text
ABSOLUTE_FRONTEND_DECOUPLING_V1 = CLOSED

stable selection reference:
  768_s256

triple physical-region candidate pool:
  retained as experimental evidence source

next:
  query-view / crop candidate-generation research
```
