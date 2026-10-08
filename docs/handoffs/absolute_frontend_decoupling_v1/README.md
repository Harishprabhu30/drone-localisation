# Handoff — absolute frontend decoupling v1

Status: **R3.1 CLOSED / R3.2 STARTED**

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

## Decision after R3.2

Promote a retrieval-native selector only if it improves or preserves 768's
direct-selection behavior without relying on GT, ORB or state logic.

If consensus switching is too aggressive, keep 768 as the selection baseline
and study confidence/temporal evidence next rather than tune thresholds on this
trajectory.

Do not reopen bootstrap/state research inside R3.
