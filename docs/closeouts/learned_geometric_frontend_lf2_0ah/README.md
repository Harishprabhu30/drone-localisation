# LF2.0-AH closeout — official maximum-accuracy LightGlue profile rejected for this workload

Status: **PASS / CLOSED** on user-reported local execution.

Status:
`PASS_LF2_0AH_SUPERPOINT_LIGHTGLUE_MAX_ACCURACY_SMOKE`

Smoke evidence SHA256:
`3a274e8b6a557d926a087436ef0f5056e08907092c677ea46a96897d01f751ca`

The official maximum-accuracy configuration was tested exactly as recommended upstream: all SuperPoint keypoints plus disabled depth and width adaptivity. On this CPU UAV↔satellite smoke it reduced match/homography support and increased pair runtime roughly four-to-fivefold relative to LF2.0-A.

Decision: do not use this profile for the next screening stage. Retain the default adaptive LF2.0-A profile and evaluate its actual candidate discrimination on a deterministic reference-free subset.
