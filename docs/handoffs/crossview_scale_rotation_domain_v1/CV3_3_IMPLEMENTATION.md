# CV3.3 — compact post-freeze aerial-MASt3R truth evaluation

Status: **IMPLEMENTATION PREPARED**

## Purpose

Evaluate the already-frozen CV3.2 compact aerial-MASt3R evidence against the
already-frozen SIFT+LightGlue uniform40 evidence, using development truth only
after all reference-free freezes and candidate identities are verified.

No model inference occurs in CV3.3.

## Pre-truth verification

The evaluator must verify before loading reference:

1. CV3.2 pair evidence SHA256:
   \`a9c9f8e422f297d77d1c00c8416d74f6b54f9f47691597db5763c9634aaa3dd4\`
2. CV3.2 selection evidence SHA256:
   \`e5b7d036acb4d0d11daaff708886cd18fe85b51d4c6dea9921c8ee1dc5d6357f\`
3. CV3.2 report status and embedded hashes;
4. recomputation of the frozen MASt3R selector reproduces all frozen winner
   identities/inlier scores;
5. CV2.2 SIFT+LightGlue evidence SHA256:
   \`c2d8f510db9a0fcc3740b06a898861371cea16da492d194326ab7a37a7494d36\`;
6. SIFT is restricted to ranks 1,2,3,5,10,20;
7. all 240 query/rank/tile identities are identical between frontends.

Only then may the development reference attachment be loaded.

## Fair comparison

Both frontends use:
- exact same 40 queries;
- exact same six candidates per query;
- selector: inliers DESC -> pool_rank ASC -> tile_id ASC;
- same tile geometry;
- same post-freeze reference.

No threshold or selector tuning is allowed.

## Metrics

For each frontend:
- selected Top1 containment;
- selected candidate center <=40 m;
- selected candidate center <=80 m;
- selected projection finite/nonfinite;
- projection median/RMSE/p95;
- projection <=40 m / <=80 m / >200 m;
- unsafe tail = nonfinite + >200 m;
- oracle best finite projection within the same six candidates;
- Top1-Top2 inlier margin.

The evaluator also reports per-query rescues/damages and winner changes.

## Frozen continuation gate

Proceed to CV3.4 full uniform40 x Top20 if at least one is true:

1. MASt3R Top1 containment improves by >=3/40 over six-candidate SIFT;
2. MASt3R Top1 center<=80 improves by >=3/40;
3. MASt3R reduces unsafe selected projections by >=3/40 and loses no more
   than 2/40 selected projection<=40 cases.

Otherwise close CV3 aerial-MASt3R as the compact candidate-selection upper
bound for this protocol.

No full Top20 or full403 computation occurs in CV3.3.
