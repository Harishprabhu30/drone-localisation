# GD1.2 closeout — global pair classifier reaches diminishing returns

Status: **PASS / CLOSED** on user-reported local execution.

Status:
PASS_GD1_2_CONTIGUOUS_OOF_MULTIVARIATE_SELECTOR_DIAGNOSTIC

OOF score SHA256:
f6bd590905242f017984202c2c3d850c44e3d11fe20256f6bf422b780531628c

## Main result

The richer ORB evidence is strongly useful, but a fixed linear global pair classifier adds only modest query-level ranking gains beyond the strongest single cue (inliers).

### contains_query Top1
- retrieval rank: 174/403
- inliers only: 296/403
- all geometry: 301/403
- all geometry + retrieval rank: 304/403

### center <=80 m Top1
- retrieval rank: 171/403
- inliers only: 288/403
- all geometry: 290/403
- all geometry + retrieval rank: 290/403

### center <=40 m Top1
- retrieval rank: 101/403
- inliers only: 158/403
- all geometry: 130/403
- all geometry + retrieval rank: 140/403

The strict <=40 objective is especially important: adding globally trained multivariate evidence can reduce query-level Top1 performance even while pair AUC improves. Therefore pair classification and within-query candidate ranking must not be treated as the same problem.

## Interpretation

1. Frozen ORB geometry already dominates original retrieval rank for Top1 locality selection.
2. Adding retrieval rank after geometry provides little additional Top1 benefit for containment/<=80.
3. A simple global linear combination does not unlock a large hidden gain over inliers and can hurt precise <=40 selection.
4. This does not yet establish the information ceiling of ORB because the model uses absolute pair features across all queries rather than query-relative evidence and physical locality support.
5. One final ORB-side formulation experiment is justified before learned frontend ablation: normalize evidence within each query and represent spatially neighboring Top20 tiles as physical locality hypotheses.

No deployable selector was exported. No blind-stress labels were loaded.
