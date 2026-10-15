# FA1.1 — targeted pair geometry and match-support diagnostics

FA1.0 established:
- 19 oracle-rescuable SIFT failures;
- 10 catastrophic SIFT damages;
- 7 severe scale_m2 ORB<=40 m -> SIFT unsafe regressions;
- 20 unique queries in the union.

The striking FA1.0 clue is that, for oracle-rescuable cases, the useful
candidate is typically deep in the SIFT selector order (median rank 11, max 17)
while the selected candidate exceeds the oracle by only one inlier median and
the Top1-Top2 inlier margin median is zero.

FA1.1 therefore tests whether the raw inlier count is hiding differences in
correspondence geometry.

For each union query it reruns:
- the frozen SIFT selected pair;
- the frozen oracle-best pair.

For the seven severe regressions it additionally reruns the exact frozen
scale_m2 ORB selected pair.

Every rerun must reproduce frozen match/inlier counts and projected position
within 0.01 m before diagnostic evidence is accepted.

New SIFT diagnostics:
- query/tile convex-hull coverage;
- query/tile bounding-box coverage;
- 4x4 grid cells, occupancy and entropy;
- PCA minor/major ratio of inlier support;
- inlier LightGlue score statistics;
- symmetric reprojection residual;
- homography condition number;
- Jacobian anisotropy/rotation/reflection;
- footprint shape/area diagnostics;
- orientation consistency;
- scale-ratio consistency;
- complete inlier correspondence coordinates.

The script also exports selected/oracle inlier-overlay PNGs. These overlays are
diagnostic only; semantic scene classification remains FA1.2.

No thresholds are tuned, no selector is changed, no confidence rule is trained,
and no blind data is loaded.
