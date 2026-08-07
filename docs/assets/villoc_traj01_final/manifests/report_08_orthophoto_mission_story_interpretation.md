# Orthophoto mission story map interpretation

## Purpose

This figure is the final visual summary of the Villoc traj01 experiment. It places the localization results directly on the orthophoto/map source, which is more realistic than a plain OSM/Folium background.

## How to read the map

- Black line: reference route used only for evaluation.
- Orange dashed line: XFeat relative-only trajectory, showing continuous visual odometry drift.
- Cyan line: temporal relative-absolute fusion trajectory.
- Small colored points: DINO retrieval outcome classes along the route.
- Large rings: accepted absolute correction events.

## Report-ready conclusion

The figure shows the main project idea: relative odometry provides continuous motion, but drift is reduced when trustworthy absolute image-to-map corrections are accepted. The retrieval outcome colors explain why absolute localization cannot be used blindly: some points are correct, some contain the correct tile only inside the Top-20 candidate pool, and some are high-confidence or Top-20 failures. Therefore, the final pipeline uses sparse gated map anchors rather than replacing relative odometry with frame-by-frame absolute localization.

## Recommended report usage

- Use the clean orthophoto map as the main report figure.
- Use the rich orthophoto map as a detailed technical figure or appendix.
- Use the mission strip when explaining route order, correction timing, and error reduction.
