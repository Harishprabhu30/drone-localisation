# FA1.2 — visual / semantic residual-failure review

FA1.1 passed frozen-parity checks for all targeted SIFT and scale_m2 ORB pairs.

FA1.2 is therefore a visualization/review stage only.

For each of the 20 unique FA1 failure queries it creates one montage containing:

1. UAV query image;
2. SIFT-selected map tile;
3. SIFT oracle-best map tile;
4. scale_m2 ORB selected map tile;
5. SIFT-selected inlier overlay;
6. SIFT-oracle inlier overlay.

Titles expose only already-authorized post-freeze development metadata:
projection errors, selector/DINO ranks, inlier counts, inlier ratio, symmetric
reprojection error, and hull coverage.

Outputs also include:
- one 20-case contact sheet;
- a manual-review worksheet;
- a review guide with fixed qualitative vocabulary.

The semantic columns are deliberately blank. No automated semantic labeling,
threshold tuning, selector modification, confidence training, matcher rerun, or
blind-data use occurs in FA1.2.

After manual review, FA1.3 should combine the human failure labels with the
already-frozen +/-2 temporal context and map-neighborhood evidence to identify
candidate reference-free warning signals.
