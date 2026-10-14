# CV3.1B1 closeout — aerial-MASt3R nine-pair smoke

Status: **PASS_CV3_1B1_REFERENCE_FREE_AERIAL_MAST3R_SMOKE**

Frozen geometric evidence SHA256:
\`6b44199c32e6d54a13c4e47b1aa58c9b4fdd69cc2c2c05be44eba514790a1d44\`

Frozen sample:
- queries: 1, 207, 403;
- pool ranks: 1, 10, 20;
- pairs: 9.

Reference-free outcomes:
- homography ok: 7 / 9;
- finite projection: 7 / 9;
- median raw correspondences: 187;
- median filtered correspondences: 20;
- median inliers: 12.

Jetson runtime/memory diagnostics (not frozen evidence):
- median inference: 1561.025 ms/pair;
- median dense matching: 2056.270 ms/pair;
- median total: 3834.688 ms/pair;
- max Torch peak allocated: 3.119 GiB;
- minimum CUDA/shared free memory after a pair: 0.733 GiB;
- minimum system available memory after a pair: 0.733 GiB.

Model:
- full FP32 aerial-MASt3R;
- no FP16/autocast promotion;
- checkpoint SHA256 \`093c3b4b3ad8294673774aae106309d5f6a1a975976bae574075fb7c5f94a67d\`.

Boundary:
- no reference truth;
- no accuracy claim;
- no threshold tuning;
- no candidate reranking;
- no AerialMegaDepth dataset.

Interpretation:
CV3.1B1 proves end-to-end feasibility/plumbing on the Jetson. The 7/9
homography count is not an accuracy continuation gate.

Decision:
proceed to frozen CV3.2 compact reference-free screen on exact uniform40 with
pool ranks 1,2,3,5,10,20 (240 pairs).
