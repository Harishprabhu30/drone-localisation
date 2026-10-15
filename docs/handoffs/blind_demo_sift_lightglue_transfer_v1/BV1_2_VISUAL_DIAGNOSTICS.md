# BV1.2 visual diagnostics

This diagnostic is intentionally reference-free.

It uses only:
- the frozen historical ORB blind submission;
- the frozen BV1.2 SIFT+LightGlue bootstrap outputs;
- the blind XFeat relative trajectory;
- the prepared EPSG:3346 orthophoto.

It does not load SRT, GT, reference attachments, oracle labels, or error metrics.

Outputs:

1. `bv1_2_trajectory_vs_historical_orb_ortho.png`
   - historical ORB map-aligned blind trajectory;
   - BV1.2 SIFT+LightGlue trajectory reconstructed causally from the active
     hypothesis at every query;
   - each frontend's first provisional/map-aligned query is marked.

2. `bv1_2_accepted_points_vs_historical_orb_ortho.png`
   - historical ORB accepted corrections;
   - BV1.2 SIFT observations that satisfy the active 12.8 m innovation gate;
   - BV1.2 tracking observations held because innovation exceeds 12.8 m.

The BV1.2 trajectory starts only at its frozen maturity query. The final map
transform is never retroactively applied to earlier frames.

Run:

```bash
source .drone_venv/bin/activate
export PYTHONPATH=$PWD/src

python scripts/villoc/geometry/bv1_2_visualize_blind_comparison.py
```

Default output directory:

`outputs/research_runs/blind_demo_sift_lightglue_transfer_v1/bv1_2/visual_diagnostics`
