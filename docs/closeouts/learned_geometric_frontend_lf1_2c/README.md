# LF1.2-C closeout — XFeat family rejected for absolute Top20 discrimination

Status: **PASS / CLOSED** on user-reported local execution.

Status:
PASS_LF1_2C_POSTFREEZE_THREE_FRONTEND_COMPARISON

Comparison diagnostics SHA256:
ad819015a71263b01f7163908573dad36264ff79935aa2c861bb14b5a304f367

## Decision

Close both sparse XFeat and XFeat* semi-dense as candidate-ranking frontends for the current QV1.4 Top20 absolute-localization problem. ORB remains the control frontend.

### Raw inlier Top1
- contains_query: ORB 292/403; XFeat sparse 184/403; XFeat* 168/403
- center <=40 m: ORB 150/403; XFeat sparse 91/403; XFeat* 79/403
- center <=80 m: ORB 287/403; XFeat sparse 182/403; XFeat* 162/403

XFeat* inlier ranking is below the frozen retrieval baseline on all three Top1 targets.

### Selected continuous projection
- ORB: median 6.092 m; RMSE 323.219 m; p95 843.409 m; <=40 293/403; <=80 305/403; >200 88/403
- XFeat sparse: median 152.493 m; RMSE 434.665 m; p95 872.378 m; <=40 134/403; <=80 160/403; >200 190/403
- XFeat*: median 256.158 m; RMSE 436.591 m; p95 811.758 m; <=40 131/403; <=80 157/403; >200 207/403

### Oracle finite projection
- ORB: median 5.298 m; <=40 336/403; <=80 370/403
- XFeat sparse: median 30.282 m; <=40 250/403; <=80 367/403
- XFeat*: median 29.886 m; <=40 238/403; <=80 362/403

## Interpretation

1. Increasing XFeat correspondence density from sparse to XFeat* does not solve candidate discrimination.
2. XFeat* saturates homography validity, so homography existence provides no rejection value.
3. Useful XFeat/XFeat* geometry often exists somewhere in Top20, especially at <=80 m oracle scale, but simple learned correspondence density does not identify the correct locality.
4. ORB remains substantially more discriminative across inliers, Jacobian anisotropy, match count, denominator stability and selected projection.
5. Do not spend another stage tuning XFeat thresholds. The next learned family should change the matcher reasoning itself.

## Next stage

Move to SuperPoint + LightGlue, reusing the existing VILLOC LightGlue lineage but under the new frozen QV1.4 Top20 and common frontend-adapter/geometry contract.

No blind-stress reference was loaded and no frontend winner was promoted.
