# Expanded orthophoto + traj01 frontend visual diagnostics

## Blind BV1.2 update

The blind visualizer now prefers the original ORT10LT master raster when it is
available. It reads only the raster window needed to cover all plotted ORB and
SIFT+LightGlue points plus a configurable metric margin, so the historical ORB
trajectory is no longer clipped by the old AOI300m visualization extent.

Default margin: 200 m.

The blind visualization remains reference-free.

## traj01_90deg_stable120m

New script:

`scripts/villoc/geometry/traj01_90deg_stable120m_visualize_frontend_comparison.py`

It loads only the frozen evidence already validated in CV2.5:
- original ORB;
- scale_m2 ORB;
- SIFT+LightGlue;
- the same post-freeze development reference attachment already authorized in
  CV2.5.

Outputs:

1. `traj01_90deg_stable120m_frontend_trajectory_comparison_ortho.png`
   - reference path;
   - original ORB selected absolute positions;
   - promoted scale_m2 ORB selected positions;
   - SIFT+LightGlue selected positions.

2. `traj01_90deg_stable120m_sift_vs_scale_m2_selected_points_ortho.png`
   - reference path;
   - scale_m2 ORB selections;
   - SIFT+LightGlue selections;
   - SIFT rescues to <=40 m;
   - SIFT rescues from ORB catastrophic (>200 m/nonfinite) selections;
   - SIFT losses from ORB <=40 m;
   - SIFT catastrophic damages.

The script also exports the selected-point tables and a per-query visual audit
CSV for later failure analysis.
