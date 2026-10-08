# Handoff — absolute frontend decoupling v1

Status: **R3.0 STARTED**

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

R3.0 freezes the first frontend-decoupling experiment:

- strict-blind candidate artifacts only before selection;
- no ORB reranking;
- no ORB projection;
- no bootstrap/state;
- reference attached only after candidate choice is frozen;
- compare direct 768 DINO Top-1 with direct triple fused rank-1 representative;
- keep current ORB-selected results beside them as downstream comparators only.

## R3.1 — retrieval-only selection baseline

Question:

> Without ORB reranking or state logic, how good is the direct rank-1 absolute
> candidate produced by the stable 768 frontend versus the triple fused
> frontend?

Required metrics:

- containment Top-1;
- center error <=40 m;
- center error <=80 m;
- center-error median/p95;
- named diagnostics q57, q99, q228, q390;
- direct comparison with existing ORB-selected results, without using those
  results to choose the retrieval candidate.

No new selector is introduced in R3.1.

## Decision after R3.1

If direct triple rank-1 is already competitive or complementary, proceed to
blind retrieval-level confidence/consensus selection.

If direct triple rank-1 is weak while useful members are often available deeper
in the fused region/Top-K, then R3.2 should focus on a retrieval-native
selection rule rather than asking ORB to rescue the list.

Do not reopen bootstrap/state research inside R3.
