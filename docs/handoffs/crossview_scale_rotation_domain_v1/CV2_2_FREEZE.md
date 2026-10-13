# CV2.2 SIFT + LightGlue uniform40 reference-free freeze

Status: **PASS_CV2_2_REFERENCE_FREE_SIFT_LIGHTGLUE_UNIFORM40_FREEZE**

Evidence SHA256:
`c2d8f510db9a0fcc3740b06a898861371cea16da492d194326ab7a37a7494d36`

Frozen screen:
- 40 queries
- 20 frozen QV1.4 candidates/query
- 800 candidate pairs
- native SIFT + SIFT-specific LightGlue
- no CV1 scale_m2 preprocessing

Reference-free diagnostics:
- homography OK: 732/800
- finite projection: 732/800
- queries with all 20 candidates producing homographies: 21/40
- median matches: 15
- median inliers: 5
- median pair runtime: 2275.249 ms (not frozen)
- observed wall time: 2463.900 s (not frozen)

Scientific boundary:
- reference loaded: false
- candidate selected/reranked: false
- accuracy claim: false
- threshold tuning: false
- full403: false

## Interpretation

The SIFT-specific frontend is less homography-saturated than the 9-pair smoke:
68/800 candidate pairs fail the minimum homography contract.

This is neither positive nor negative evidence for locality selection until truth
is attached. A lower wrong-candidate homography rate may help discrimination,
but correct candidates may also be among the failed pairs.

Proceed to CV2.3 post-freeze comparison against:
1. frozen original ORB;
2. frozen CV1 scale_m2 ORB;
3. frozen native SIFT + LightGlue.

No frontend parameter may change before that comparison.
