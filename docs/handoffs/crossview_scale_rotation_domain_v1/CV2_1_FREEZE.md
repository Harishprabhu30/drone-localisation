# CV2.1 SIFT + LightGlue smoke freeze

Status: **PASS_CV2_1_REFERENCE_FREE_SIFT_LIGHTGLUE_SMOKE**

## Frozen smoke

- LightGlue pinned commit:
  `eb42fee2d71449efb0aa5c10549752b5d75384d8`
- SIFT source blob:
  `802fc1c2eb9ee852691e0e4dd67455f822f8405f`
- LightGlue source blob:
  `485833515eb0a80866b3d078d3e43d39ddcd84ae`
- SIFT-LightGlue weights SHA256:
  `5b52b8d9982d43532dc042606b346bb9594c9f5a4bd6f64362c63866287b4ac0`
- scale/orientation fields present: true
- homography OK: 9/9
- finite projection: 9/9
- median query/tile keypoints: 3271 / 3368
- median matches/inliers: 23 / 6
- median total runtime per pair: 3637.861 ms (diagnostic only; not frozen)
- reference loaded: false
- scale_m2 applied to SIFT: false

Smoke evidence SHA256:
`1eedef502722230ea85012625d5323792d41d651a8a4c1e65043b4661f1c50d1`

## Interpretation

The CV2 SIFT-specific frontend plumbing is valid:
- pinned source and checkpoint provenance are fixed;
- SIFT scale/orientation tensors are present;
- the SIFT-specific LightGlue model executes;
- homography and query-center projection are functional.

The 9/9 homography rate is **not** an accuracy result.  It reinforces the
project's existing observation that homography existence alone is not a useful
locality discriminator.

CV2.2 may now run the exact frozen uniform40 x Top20 reference-free screen.
