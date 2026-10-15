# FA1.3 — combined context and separability audit

FA1.3 intentionally combines the remaining short residual-failure audits into
one execution.

It performs four descriptive analyses:

1. **Temporal consistency**
   - distance from the current SIFT-selected projection to selected projections
     at q-2, q-1, q+1 and q+2;
   - for FA1 cases, the same measurement is computed for the oracle candidate.

2. **Geographic Top20 neighborhood consensus**
   - number of other DINO Top20 tile centers within one map stride (51.2 m);
   - number within two strides (102.4 m);
   - reciprocal-DINO-rank-weighted support at both radii.
   These radii are derived from frozen map geometry, not tuned on failures.

3. **Cross-frontend disagreement**
   - map distance between SIFT+LightGlue selected projection and scale_m2 ORB
     selected projection;
   - for FA1 cases, whether scale_m2 ORB lies closer to the oracle than to the
     SIFT-selected position.

4. **Retrieval-versus-geometry conflict / descriptive separability**
   - selected DINO pool rank;
   - raw inliers and inlier ratio;
   - Top1-Top2 inlier margin;
   - number of candidates tied at the maximum inlier count;
   - gain in inliers over DINO rank 1;
   - symmetric reprojection residual;
   - neighborhood support;
   - temporal distance;
   - cross-frontend disagreement.

The script computes these reference-free features for all 403 development
queries, then uses already-authorized development truth only to label groups:
<=40 m, 40-200 m, unsafe, FA1 union, oracle-rescuable, catastrophic-damage and
severe-regression.

No threshold is selected. No classifier/confidence rule is trained. No matcher
is rerun. No blind data is used.

## FA1.2 qualitative observation

Manual review of the generated contact sheet and representative montages found
the residual failures to be predominantly vegetation-heavy or mixed
vegetation/limited-urban scenes. Dense street-heavy or urban-heavy failures were
not prominent in the reviewed set. This remains a qualitative diagnostic
observation, not an automated scene classification.

## Intended closeout

FA1.3 should be the last short audit before an FA1 closeout. If the combined
evidence identifies plausible reference-free warning signals, the next new
research stage should freeze an out-of-fold/blind-safe confidence or abstention
protocol rather than tuning a rule directly on these 403 development queries.
