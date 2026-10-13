# CV1.5 — scale mechanism and camera/GSD audit

Status: **IMPLEMENTATION PREPARED / LOCAL EXECUTION REQUIRED**

## Purpose

CV1.4 established that fixed query downsampling improves full403 candidate
selection and catastrophic-tail behavior.  CV1.5 does not test another scale.

It asks:

1. Does `scale_m2` mainly improve locality selection or same-tile projection?
2. How do ORB match/support/scale-consistency signals change on true versus
   false candidate pairs?
3. Does downsampling alter query ORB octave/support allocation?
4. Are the SRT camera fields sufficiently stable and sufficiently interpretable
   to justify a physical UAV-to-map GSD model?

## Camera metadata boundary

CV1.5 parses a **camera-only** subset from the SRT.  Latitude/longitude are not
emitted into the sanitized table and are not used by the camera audit.

Geometry candidates:
- relative altitude;
- focal length;
- digital zoom ratio;
- gimbal pitch/roll.

Appearance/explanatory fields:
- f-number;
- ISO;
- shutter;
- EV.

Absolute altitude is recorded only as explanatory metadata; it is not treated as
AGL height.

The repository does not currently identify the camera model, calibrated
`fx/fy`, sensor dimensions, or the exact semantic contract of the SRT
`focal_len` field.  Therefore physical GSD remains an audit hypothesis.

## Implied-scale diagnostic

If, purely as a diagnostic hypothesis, `scale_m2` were exactly the factor
required to match the 0.20 m/px map at the nominal 120 m altitude, then:

`baseline_query_gsd = scale_m2 * map_gsd`

and an implied pinhole focal length in processed pixels can be computed from:

`f_px = H / baseline_query_gsd`.

This is an **implied value**, not calibration.

## Local execution

```bash
python -m unittest tests.test_cv1_5_scale_mechanism_camera_audit -v

python scripts/villoc/geometry/cv1_5_scale_mechanism_camera_gsd_audit.py \
  --config configs/research/crossview_scale_rotation_domain_cv1_5.yaml
```

The stage will print:
- mechanism summary;
- same-winner/changed-winner catastrophic rescue split;
- ORB keypoint/octave changes;
- SRT camera-field ranges/unique counts;
- implied GSD/focal/FOV diagnostic;
- whether a physical-GSD claim remains blocked.
