# RG2.4 closeout — locality persistence and evidence independence

Status: **PASS / CLOSED**.

## RG2.4-A reference-free freeze

Development:
- 403 queries
- 8060 transition rows
- 4657 locality runs
- transitions SHA256: 4b77653ee6626789a5a1a4be81eff40e5820cbb8062aefcfa58bb3e88886e0ba
- runs SHA256: 75174aa46a60ac716283e2ef5d0a77911bfff43bb7d03417ef006ab12bec5601

Blind stress:
- 123 queries
- 2460 transition rows
- 1886 locality runs
- transitions SHA256: 3ec55cc852ef878a4dcac4efa312c344c2daa193535fd14d71f6101a5fced4d6
- runs SHA256: 6461e747925a8cdcf374b6fdf3a7de5382c94c3b989a8ce5313f8db5f38e59e2

Reference unavailable; bootstrap/state unchanged.

## RG2.4-B post-freeze development result

Status:
PASS_RG2_4B_POSTFREEZE_PERSISTENCE_CORRECTNESS_AUDIT

Run accuracy SHA256:
3efd919580e13e2ea783ff46fb74f35962532fe550848156916fbc5164cee398

### Main result

Persistence is useful only when interpreted together with spatial tightness and
the underlying consensus geometry. It is not a universal correctness signal.

The 25.6 m half-stride rule covers more queries in persistent runs but can
preserve wrong aliases for many consecutive frames. The 12.8 m quarter-stride
rule is more selective and, for several medoid-style tracks, yields much higher
conditional accuracy once a run reaches at least three frames.

Examples for query observations belonging to runs of length >=3:

- full-overlap medoid, quarter-stride:
  112 queries; <=40 m 109/112; <=80 m 109/112; median 4.735 m;
  one persistent-wrong run; longest wrong run 3 queries.
- immediate medoid, quarter-stride:
  123 queries; <=40 m 117/123; <=80 m 117/123; median 4.271 m;
  two persistent-wrong runs; longest wrong run 3 queries.
- immediate radius-cluster medoid, quarter-stride:
  154 queries; <=40 m 145/154; <=80 m 146/154; median 4.274 m;
  two persistent-wrong runs; longest wrong run 5 queries.
- full-overlap radius-cluster medoid, quarter-stride:
  162 queries; <=40 m 144/162; <=80 m 145/162; median 4.377 m;
  four persistent-wrong runs; longest wrong run 8 queries.

Counterexample showing that loose persistence can reinforce a wrong alias:

- full-overlap inlier-Huber, half-stride:
  290 queries in runs >=3; <=40 m 130/290; <=80 m 162/290;
  median 45.999 m; 24 persistent-wrong runs; longest wrong run 16 queries.

Therefore:
1. frame-level absolute observations are not independent;
2. persistent locality can be highly informative when spatially tight;
3. persistence alone is unsafe because stable wrong aliases exist;
4. no RG2.4 method/threshold is promoted as a deployment rule because all
   correctness analysis used one development reference trajectory;
5. bootstrap/state remains unchanged.

## Research priority after RG2.4

Freeze QV1.4 retrieval for now. It already places a useful region inside Top20
for most development queries. The next research axis should address the harder
problem:

> Given the frozen Top20, which candidate or physical locality is geometrically
> consistent with the query, and how can false visual aliases be rejected?

This is broader than switching ORB to a stronger matcher. The next stage should
first enrich the geometric evidence available for every Top20 pair, including:
- symmetric forward/backward reprojection residuals;
- spatial support beyond bounding-box coverage;
- inlier grid occupancy / convex-hull support;
- keypoint orientation consistency;
- keypoint scale-ratio consistency;
- local homography Jacobian / singular-value anisotropy;
- projective denominator stability over the query footprint;
- projected query-footprint area / convexity / extrapolation diagnostics.

Only after this evidence is frozen should development reference be attached to
determine which blind quantities discriminate correct candidates from aliases.
The matcher frontend can then be ablated while holding the evidence/selector
framework fixed.

Recommended next branch:
research/top20-geometric-discrimination-v1
