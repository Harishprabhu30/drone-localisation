# CV3.2 closeout — compact reference-free aerial-MASt3R screen

Status: **PASS_CV3_2_REFERENCE_FREE_COMPACT_FREEZE**

Frozen scope:
- exact uniform40;
- pool ranks 1,2,3,5,10,20;
- 40 queries;
- 240 pairs;
- no reference truth.

Frozen evidence:
- pair SHA256:
  \`a9c9f8e422f297d77d1c00c8416d74f6b54f9f47691597db5763c9634aaa3dd4\`
- selection SHA256:
  \`e5b7d036acb4d0d11daaff708886cd18fe85b51d4c6dea9921c8ee1dc5d6357f\`

Reference-free pair summary:
- homography ok: 140 / 240;
- finite projection: 140 / 240;
- queries with any/all6/no homography: 39 / 2 / 1;
- selected homography/finite projection: 39 / 40;
- median raw/filtered correspondences: 143 / 9.5;
- median pair inliers: 6;
- median selected inliers: 5422;
- median selected inlier margin: 1565.5.

Reference-free selection-integrity audit:
- selected filtered correspondences median/max: 9730 / 14289;
- selected inliers median/max: 5422 / 12472;
- selected inlier-ratio median: ~0.582;
- every selected inlier count was <= that selected pair's filtered correspondence count;
- all 40 frozen selection rows matched the corresponding pair evidence;
- audit ended with \`PASS_CV3_2_REFERENCE_FREE_SELECTION_INTEGRITY\`.

Jetson runtime/memory diagnostics (not frozen geometric evidence):
- median inference: 1559.846 ms/pair;
- median dense matching: 2053.041 ms/pair;
- median total: 3858.946 ms/pair;
- max Torch peak allocated: 3.119 GiB;
- minimum CUDA/shared free: 0.772 GiB;
- minimum system available: 0.775 GiB.

Interpretation boundary:
the large selected-inlier counts are valid dense-MASt3R correspondence counts.
They are not an accuracy claim.

Decision:
CV3.2 evidence is frozen and copied back to the Mac with matching SHA256.
Proceed to CV3.3 post-freeze truth attachment against SIFT+LightGlue restricted
to the exact same six candidate identities.
