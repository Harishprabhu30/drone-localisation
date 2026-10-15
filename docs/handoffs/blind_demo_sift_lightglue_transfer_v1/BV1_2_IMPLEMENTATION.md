# BV1.2 — minimum-confident-v2 transfer to SIFT+LightGlue observations

BV1.2 consumes the frozen BV1.0 768_s256 SIFT+LightGlue pair evidence directly.

Preserved from the historical minimum-confident-v2 policy:
- Top-4 candidate observations;
- four visual evidence frames;
- minimum three unique tile IDs;
- minimum visual span 100 px;
- minimum map span 50 m;
- median residual limit 51.2 m;
- maximum residual limit 102.4 m;
- maturity support 3;
- acquisition threshold 12.8 m;
- tracking threshold 12.8 m;
- ACQUISITION to TRACKING causal accept/hold behavior.

Transferred observation source:
- 768_s256 map geometry;
- SIFT+LightGlue projected query-center positions;
- candidate order: inliers descending, then DINO rank ascending, then tile ID ascending;
- no ORB homography recomputation.

This is a new blind transfer experiment. It does not relabel the historical
ORB/512_s256 architecture contract as though it were unchanged.

No GT, SRT, GPS, oracle, fusion, or post-freeze evaluation is used.
The strongest allowed positive state remains PROVISIONAL_ABSOLUTE_LOCK.
