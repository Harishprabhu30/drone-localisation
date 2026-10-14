# BV1.0 — combined blind 768_s256 SIFT+LightGlue transfer

BV1.0 combines preflight and the first blind verifier execution.

It uses the existing 123-query no-SRT demonstration but intentionally changes
the absolute map stack from historical 512_s256 + ORB to:

- 768_s256 map geometry;
- corresponding frozen DINO map descriptor cache;
- a newly recomputed blind DINO Top20 for all 123 queries;
- frozen SIFT+LightGlue verification on all 2460 pairs.

This is therefore not described as a verifier-only A/B replay. Both map
geometry/candidate pool and verifier are declared changes.

Historical demo outputs remain immutable.

Blind boundary: no SRT, GPS, reference coordinates, oracle labels, or evaluation
metrics. BV1.0 freezes candidate pool, pair evidence, and one selected map
observation per query.

No bootstrap or fusion is run in BV1.0.

Next: BV1.2 adapts minimum_confident_v2 to consume frozen SIFT+LightGlue
projected observations directly, because the historical backend internally
recomputes ORB geometry. The 45-degree oblique test remains a separate later
stress experiment.
