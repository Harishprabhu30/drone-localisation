# FA1 — residual SIFT+LightGlue failure analysis plan

Purpose: explain why the promoted SIFT+LightGlue selector still makes residual
wrong-locality choices on traj01_90deg_stable120m without tuning on the answers.

## FA1.0 frozen-evidence audit

Cases:
- 19 oracle-rescuable queries: selected projection >40 m while a <=40 m
  candidate exists in the same frozen Top20;
- 10 catastrophic SIFT damages relative to scale_m2 ORB;
- 7 severe regressions: scale_m2 ORB <=40 m -> SIFT unsafe (>200 m/nonfinite).

The audit measures:
- selected vs oracle candidate selector rank and DINO pool rank;
- inlier/match differences and Top1-Top2 inlier margin;
- LightGlue match-confidence summaries;
- inlier ratio;
- symmetric reprojection residual;
- Jacobian scale/anisotropy/rotation/reflection;
- projective denominator and footprint diagnostics;
- projected-inside-tile state;
- ORB selected candidate/error;
- +/-2 query temporal context.

No matcher rerun, threshold tuning, selector change or confidence training occurs.

## FA1.1 targeted geometry/match-support rerun

The frozen CV2.4 SIFT evidence does not retain individual inlier coordinates.
For the union of FA1.0 cases, rerun only diagnostic pairs:
- selected SIFT candidate;
- oracle-best <=40 candidate where available;
- scale_m2 ORB selected candidate for severe regressions.

Recover:
- query/tile inlier convex-hull coverage;
- 4x4 grid occupancy/entropy;
- PCA minor/major ratio;
- correspondence visualizations;
- optional scale/orientation distributions.

This is diagnostic evidence only. It must not alter the frozen CV2 result.

## FA1.2 visual/semantic case review

Create per-query montages:
- UAV query;
- SIFT selected tile;
- SIFT oracle tile;
- scale_m2 ORB selected tile;
- correspondence overlays.

Manual labels should describe visible ambiguity rather than change thresholds:
repetitive apartments/roofs, road geometry, forest canopy, shadows, low-texture
areas, repeated parking patterns, partial overlap, etc.

## FA1.3 causal/context audit

For each failure ask:
- Is the wrong winner an isolated temporal jump?
- Do q-1/q+1 select the correct locality?
- Is the oracle candidate within SIFT selector Top3?
- Is the selected tile geographically isolated from other strong candidates?
- Do nearby/overlapping map tiles support a competing locality?
- Does ORB/SIFT disagreement provide a reference-free warning signal?

## Decision objective

The objective is not to invent a threshold from these 403 development queries.
It is to identify reference-free signals that plausibly separate correct from
wrong winners. Any new confidence/abstention or secondary selector must later
be frozen under an OOF/blind-safe protocol.

## Documentation after FA1

Once failure mechanisms are understood, prepare a consolidated technical report
covering:
- problem/architecture;
- 768_s256 map and QV1.4 DINO Top20 retrieval;
- original ORB and scale_m2;
- promoted SIFT+LightGlue;
- aerial-MASt3R upper bound;
- full403 quantitative evidence;
- FA1 residual failure mechanisms;
- blind 123-query no-SRT transfer and minimum-confident bootstrap;
- runtime/Jetson implications;
- limitations and deployment architecture;
- frozen provenance/hashes and scientific boundaries.
