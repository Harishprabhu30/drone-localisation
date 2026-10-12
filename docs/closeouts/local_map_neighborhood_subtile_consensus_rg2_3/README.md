# RG2.3 closeout — projection consensus audit

Status: **PASS / CLOSED** on development post-freeze evaluation.

## RG2.3-A freeze

Development consensus SHA256:
c4189a61d56dbedea90fa67a17184ecee05ce92ad222cfb6f83fd4edd3c1ec09

Blind-stress consensus SHA256:
ba577e907b268ee5a323491763ab653e1d7079d0457b3e05b8e1c2868fff1e07

Reference was not loaded during consensus generation.

## RG2.3-B development audit

Status:
PASS_RG2_3B_POSTFREEZE_CONSENSUS_ACCURACY_AUDIT

Consensus accuracy SHA256:
d864bb7f694485499e84b898d1302b22968002eb6021668669662726cd9e12b2

### Baselines

anchor tile center:
- median 256.837 m
- RMSE 441.435 m
- <=40 m 101/403
- <=80 m 171/403

anchor sub-tile projection:
- median 255.951 m
- RMSE 2884.510 m
- <=40 m 146/403
- <=80 m 162/403

Interpretation: sub-tile projection increases close-range successes but creates very large outliers. A valid homography is therefore not a trustworthy absolute-position measurement by itself.

### Primary all-finite consensus

full overlap:
- coordinate median: median 247.010, RMSE 424.097, <=40 144, <=80 174
- geometric median: median 249.273, RMSE 422.664, <=40 148, <=80 173
- inlier-Huber centroid: median 240.679, RMSE 422.451, <=40 137, <=80 181
- medoid: median 246.869, RMSE 422.772, <=40 148, <=80 169
- radius-cluster medoid: median 301.319, RMSE 417.274, <=40 168, <=80 177

immediate:
- coordinate median: median 249.146, RMSE 442.706, <=40 160, <=80 172
- geometric median: median 243.520, RMSE 439.900, <=40 160, <=80 171
- inlier-Huber centroid: median 251.141, RMSE 438.055, <=40 166, <=80 183
- medoid: median 230.522, RMSE 438.949, <=40 159, <=80 170
- radius-cluster medoid: median 303.724, RMSE 446.814, <=40 168, <=80 178

No single consensus estimator dominates across RMSE, median, <=40 and <=80. No deployment method is selected from this development audit.

### Oracle diagnostic

best full-overlap all-finite frozen projection:
- median 34.045 m
- <=40 m 203/403
- <=80 m 206/403

best full-overlap inside-only:
- median 94.739 m
- <=40 m 193/403
- <=80 m 200/403

best immediate all-finite:
- median 182.383 m
- <=40 m 183/403
- <=80 m 187/403

best immediate inside-only:
- median 196.600 m
- <=40 m 179/403
- <=80 m 184/403

The oracle gap is the key result: useful local projections frequently exist, especially in the full-overlap neighborhood, but the current blind consensus variables do not reliably identify the correct mode.

Inside-only filtering is not automatically beneficial. The full-overlap oracle median degrades from 34.045 m to 94.739 m when outside-tile projections are excluded. Outside-tile projections must therefore remain evidence, not be discarded solely because they extrapolate beyond tile bounds.

## Scientific conclusion

RG2 establishes that:
1. overlapping map windows can produce continuous global-coordinate projections;
2. raw homography validity is saturated and non-discriminative;
3. a single projected coordinate can be highly accurate or catastrophically wrong;
4. simple robust consensus reduces catastrophic RMSE relative to raw anchor projection, but does not solve global retrieval aliasing;
5. the large oracle-best versus blind-consensus gap identifies projection-mode selection / agreement quality as the remaining local problem;
6. local consensus cannot replace frozen global retrieval.

## Next stage

Proceed to RG2.4 locality persistence / evidence-independence audit before any bootstrap or state-machine modification.

RG2.4 should study repeated locality evidence over time using only blind-available quantities such as:
- consensus coordinate persistence;
- projection-cloud dispersion;
- supporting tile-window diversity;
- temporal span;
- relative motion span where available;
- locality transitions.

Do not tune a new confidence threshold from RG2.3-B reference results before freezing the RG2.4 measurements.
