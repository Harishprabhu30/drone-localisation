# Handoff — absolute frontend decoupling v1

Status: **R3.1/R3.2/R3.3 CLOSED — R3.4 STARTED**

Branch:

```text
research/absolute-frontend-decoupling-v1
```

Starting point:

```text
413ec336a7d7038c17adad6fd9fc6b81b36df653
docs/closeouts/retrieval_candidate_pool_v2_r2/README.md
```

## Why this stage exists

R2 established two facts that must now be treated separately:

1. the triple 512/768/1024 physical-region RRF improves useful candidate-region
   availability under the same Top-20 budget;
2. the historical ORB/bootstrap/state chain does not safely convert that richer
   stream into a better end-to-end trajectory.

The next research question is therefore:

> How strong can the absolute frontend become before geometric reranking and
> state estimation are allowed to influence candidate selection?

## Current references

Stable single-scale control:

```text
768_s256
```

Experimental richer frontend:

```text
512_s256 + 768_s256 + 1024_s256
physical-region RRF
one merged Top-20
```

Do not call the triple the current deployable winner. It is a research asset.

## Stage boundaries

Keep four responsibilities separately measurable:

```text
candidate availability
  -> candidate selection
  -> geometric projection/refinement
  -> state acceptance
```

ORB reranking and ORB sub-tile projection are different responsibilities.

The frozen minimum_confident_v2 remains a reproducible baseline only. It is not
a restriction on future research, but it must not be modified in place.

## R3.0 — contract

R3.0 freezes the frontend-decoupling principle:

- strict-blind candidate artifacts only before selection;
- no ORB reranking;
- no ORB projection;
- no bootstrap/state;
- reference attached only after candidate choice is frozen;
- keep 768_s256 as the stable selection reference;
- keep the R2 triple as the richer experimental candidate source.

## R3.1 — retrieval-only selection baseline

Status:

```text
PASS_R3_RETRIEVAL_ONLY_BASELINE
```

Measured direct rank-1 selection:

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

Interpretation:

The triple's R2 value does **not** come from its raw fused rank-1
representative. The stable 768 DINO Top-1 is substantially stronger as a direct
selector.

Named examples reinforce the complementarity:

```text
q99:
  768 Top-1     ~21.5 m / containing
  triple Top-1 ~523.3 m / false

q228:
  768 Top-1     ~56.8 m / containing
  triple Top-1  ~21.2 m / containing

q390:
  768 Top-1    ~463.6 m / false
  triple Top-1 ~83.6 m / containing
```

Therefore R3 must use the triple as a **consensus/rescue source**, not replace
768 blindly with fused rank-1.

## R3.2 — retrieval-native consensus / rescue selector

Status: **STARTING**

No learned weights and no GT-tuned thresholds are allowed in the first selector
family.

The main policy is:

```text
768 DINO Top-1
      |
      +-- find its physical-region support inside triple fused Top-20
      |
      +-- find strongest cross-scale region by:
            support-scale count
            then fused RRF
            then fused rank
      |
      +-- switch only if another region has STRICTLY MORE
          cross-scale support than the 768 anchor
      |
      +-- when a selected region contains a 768 member,
          use that 768 member as the concrete tile;
          otherwise use the region representative
```

This is `768_strict_support_rescue`.

A more conservative control, `768_three_scale_rescue`, switches only when a
three-scale region exists and the 768 anchor itself is not three-scale
supported.

Two diagnostics isolate the role of region/member choice:

```text
triple_support_first_representative
triple_support_first_768_preferred
```

All policies are frozen before post-selection reference evaluation.

Required outputs:

- direct candidate accuracy for every policy;
- switch count away from 768;
- post-freeze switch gains/losses;
- containment and <=40/<=80 changes;
- q57, q99, q228, q390;
- no ORB/state execution.

## R3.2 result — support count is not sufficient confidence

Status:

```text
PASS_R3_RETRIEVAL_NATIVE_CONSENSUS_SELECTOR
```

Measured:

```text
768_dino_top1:
  contains 174 / 403
  <=40     101 / 403
  <=80     171 / 403

triple_support_first_representative:
  contains 127 / 403
  <=40      80 / 403
  <=80     125 / 403
  switches 269

triple_support_first_768_preferred:
  contains 137 / 403
  <=40      68 / 403
  <=80     132 / 403
  switches 161

768_strict_support_rescue:
  contains 145 / 403
  <=40      83 / 403
  <=80     142 / 403
  switches 38
  improved switches 4
  worsened switches 34

768_three_scale_rescue:
  identical to strict-support rescue
```

Therefore cross-scale support count is not a valid standalone confidence signal.

Named cases explain the ambiguity:

```text
q99:
  correct 768 anchor region has three-scale support, fused rank 7
  wrong fused rank-1 region also has three-scale support
  -> support count cannot say "hold"

q390:
  wrong 768 anchor region has three-scale support, fused rank 5
  useful fused rank-1 region also has three-scale support
  -> support count cannot say "rescue"

q57:
  wrong region itself has three-scale support
  -> correlated multi-scale aliasing can reinforce a false location
```

R3.2 therefore does not promote any new selector.

## R3.3 — retrieval confidence-feature audit

Status: **STARTING**

R3.3 is diagnostic only. It does not choose a new candidate.

Blind features are frozen first from retrieval artifacts:

- 768 DINO Top-1 score and Top-1/Top-2 margin;
- 768 anchor fused rank/support/RRF score;
- fused rank-1 support and RRF margin;
- fused rank-1 per-scale member ranks;
- distance between the 768 anchor and fused rank-1 physical regions;
- count of three-scale and multi-scale regions in the fused Top-K.

Only after the feature table is frozen, reference is attached to label:

```text
anchor-only good
fused-only good
both good
both bad
```

for containment and <=80 m.

The purpose is to determine whether any **static blind retrieval confidence
signal** can distinguish q99-type "hold 768" cases from q390-type "rescue"
cases. No thresholds or selector policy are tuned in R3.3.

If no static feature separates those cases credibly, the next step should be
causal temporal retrieval consistency rather than more static score tuning.

Do not reopen ORB/bootstrap/state research inside R3.


## R3.3 result — static confidence is informative but not decisive

Status:

```text
PASS_R3_RETRIEVAL_CONFIDENCE_AUDIT
```

Measured discordant direct-selection cases:

```text
containment:
  anchor-only good   63
  fused-only good    16

<=80 m:
  anchor-only good   65
  fused-only good    19
```

Best static diagnostic separation is informative but not sufficient to promote
a selector:

```text
containment:
  anchor_fused_rrf_score   AUC* ~0.795
  member-rank statistics  AUC* ~0.746
  768 Top1/Top2 margin     AUC* ~0.723

<=80 m:
  anchor_fused_rrf_score   AUC* ~0.791
  anchor fused rank        AUC* ~0.729
  fused RRF/member ranks   AUC* ~0.72
```

Named cases still require opposite decisions under similar static support:
q99 should hold 768, while q390 should prefer the fused region. q57 shows that
three-scale agreement can reinforce the wrong place.

No threshold is promoted from R3.3.

## R3.4 — causal temporal retrieval audit

Status: **STARTING**

R3.4 is diagnostic only.

Question:

> Does a candidate physical region have a stronger causal track through recent
> retrieval candidate sets than its static rank/score alone suggests?

For the 768 anchor and fused Top-1, compute only from previous queries:

- nearby-region hit count in the previous 3 and 5 queries;
- rank-weighted support;
- nearest historical-region distance;
- matched historical ranks;
- displacement from the previous anchor/fused Top-1.

Fixed physical radii:

```text
51.2 m
102.4 m
```

The blind temporal feature table is frozen before R3.3 labels are attached.

No selector threshold is fitted in R3.4.

Decision:

- if causal temporal features clearly improve separation and are mechanistically
  interpretable, freeze a simple causal policy next;
- otherwise stop selector tuning on this trajectory and return to
  candidate-generation / representation research.

This keeps the original research objective intact while following measured
bottlenecks.
