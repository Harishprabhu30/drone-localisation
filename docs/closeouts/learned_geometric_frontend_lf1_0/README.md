# LF1.0 closeout — XFeat absolute preflight/smoke

Status: **PASS / CLOSED** on user-reported local execution.

Status:
PASS_LF1_0_XFEAT_ABSOLUTE_FRONTEND_PREFLIGHT_SMOKE

Official VERLab XFeat commit:
e92685f57f8318b18725c5c8c0bd28c7fe188d9a

Weights SHA256:
0f5187fd7bedd26c7fe6acc9685444493a165a35ecc087b33c2db3627f3ea10b

Smoke evidence SHA256:
72e65e92417a3abfaab161b319ed70a58fdf0d6780aeef0cac7e1929640b6ca3

CPU smoke results:
- development: 5 queries / 20 pairs; homography 20/20; finite 20/20; inside 18/20; median matches 19.5; median inliers 5.0; median total pair time 221.208 ms.
- blind stress: 3 queries / 12 pairs; homography 11/12; finite 11/12; inside 10/12; median matches 13.5; median inliers 4.5; median total pair time 208.074 ms.

Reference was not loaded and no accuracy claim was made.

Interpretation:
- pinned XFeat installation, weights and sparse matching API are viable for absolute UAV-to-map matching;
- unlike ORB GD1.0, homography success is not saturated in the smoke sample;
- XFeat produces much fewer sparse correspondences/inliers than ORB, so raw match counts cannot be compared by magnitude across frontends;
- smoke runtime is only a small-sample observation and is not a final performance benchmark;
- proceed to a full reference-free Top20 evidence freeze before attaching development truth.
