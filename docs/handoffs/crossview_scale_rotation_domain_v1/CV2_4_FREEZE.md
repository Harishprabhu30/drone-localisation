# CV2.4 full403 SIFT + LightGlue reference-free freeze

Status: **PASS_CV2_4_REFERENCE_FREE_SIFT_LIGHTGLUE_FULL403_FREEZE**

Evidence SHA256:
`c0031242f0b359f1bfa3c87ea47ac428393ed3e8e03d7b5c6d0822e8822cb8a0`

Frozen run:
- queries: 403
- candidates/query: 20
- pairs: 8060
- homography OK: 7382 / 8060
- finite projection: 7382 / 8060
- queries with all 20 homographies: 230 / 403
- queries with no homography: 0 / 403
- median matches/inliers: 15 / 5
- median total pair runtime: 1881.012 ms (not frozen)
- observed wall time: 20993.051 s (not frozen)

Scientific boundary:
- reference loaded: false
- scale_m2 applied to SIFT: false
- frontend parameters changed: false
- candidate selected/reranked: false
- accuracy claim: false

## Interpretation

The promoted SIFT + LightGlue frontend has now been measured across the complete
403-query development candidate pool without opening truth.

The 7382/8060 homography retention confirms that the full trajectory preserves
the same qualitative behavior seen on uniform40: most candidate pairs support a
homography, but the frontend is not fully homography-saturated.

No geographic-accuracy conclusion may be drawn until CV2.5 attaches development
truth to this frozen evidence.

Jetson/CUDA execution is not required to answer CV2 accuracy because the CPU
full403 evidence has completed.  GPU migration remains a separate runtime /
deployment experiment and should not replace this frozen CPU reference.
