# S8.13A Planning README --- Heading-Aligned Dynamic Map & Temporal Context Roadmap

## Confirmed conclusions

-   Evaluate heading-aligned dynamic rotation of the cached satellite
    neighbourhood first.
-   Do not store multiple rotated cache copies.
-   Keep the north-up map immutable; rotate only the runtime working
    canvas.
-   Treat the first few frames as startup orientation/calibration rather
    than stable navigation.
-   Defer 45° multi-view integration until heading alignment is
    evaluated.
-   Next experiment: previous + current nadir frame context for DINO
    retrieval.
-   Record relative altitude together with retrieval performance to
    determine the most useful operational altitude.

## Planned stages

  Stage    Objective
  -------- -------------------------------------------
  S8.13A   Heading-aligned dynamic cached map
  S8.13B   Previous + current frame temporal context
  S8.13C   45° multi-view context

## Assets

Copy current figures:

``` bash
mkdir -p docs/assets/s8_13a_heading_rotation
cp outputs/villoc/90_deg/figures/s8_3_reference_xy_trajectory*.png docs/assets/s8_13a_heading_rotation/s8_13a_xy_trajectory.png 2>/dev/null || true
cp outputs/villoc/90_deg/figures/s8_3_reference_yaw*.png docs/assets/s8_13a_heading_rotation/s8_13a_yaw_profile.png 2>/dev/null || true
cp outputs/villoc/90_deg/figures/s8_4_contact_sheet*.png docs/assets/s8_13a_heading_rotation/s8_13a_contact_sheet.png 2>/dev/null || true
```

Generate during S8.13A: - Unwrapped yaw vs time - Yaw rate vs time -
Relative altitude vs frame - Relative altitude vs retrieval/oracle
rank - Motion-state timeline - Heading-aligned map illustration

## Team questions

-   Does gimbal yaw follow aircraft heading?
-   Which image edge corresponds to yaw=0°?
-   Is yaw true-north referenced?
-   Expected operational flight altitude?
