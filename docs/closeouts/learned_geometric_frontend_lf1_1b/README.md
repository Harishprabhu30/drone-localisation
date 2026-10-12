# LF1.1-B closeout — sparse XFeat does not beat ORB

Status: **PASS / CLOSED** on user-reported local execution.

Status:
PASS_LF1_1B_POSTFREEZE_ORB_XFEAT_COMPARISON

Comparison diagnostics SHA256:
3c0590625f7b772d69bbe92fb9a1ffddaf0414c25d76e1d9552c59cde6ca31aa

## Decision

The pinned sparse XFeat frontend is rejected as a replacement for ORB for the current QV1.4 Top20 absolute-localization pipeline.

### Raw inlier Top1 ranking
- contains_query: ORB 292/403, XFeat 184/403
- center <=40 m: ORB 150/403, XFeat 91/403
- center <=80 m: ORB 287/403, XFeat 182/403

### Selected continuous projection (valid-then-inliers)
- ORB median 6.092 m, RMSE 323.219 m, p95 843.409 m, <=40 293/403, <=80 305/403, >200 88/403
- XFeat median 152.493 m, RMSE 434.665 m, p95 872.378 m, <=40 134/403, <=80 160/403, >200 190/403

### Homography validity as rejection

XFeat homography validity does not become a useful alias gate on development:
- useful-pair retention is approximately 99–100%;
- negative-pair rejection is only approximately 0.8%;
- query-level useful-candidate retention remains approximately 99–100%.

ORB homography validity is completely saturated and rejects nothing.

### Common geometric evidence

ORB remains much more discriminative. Examples:
- contains_query inlier AUC*: ORB 0.8558, XFeat 0.6788;
- <=40 inlier AUC*: ORB 0.8565, XFeat 0.6808;
- <=80 inlier AUC*: ORB 0.8611, XFeat 0.6848.

## Important nuance

XFeat is not devoid of useful absolute geometry. Its oracle best finite projection reaches <=80 m for 367/403 queries, close to ORB's 370/403. However its oracle median is 30.282 m versus ORB 5.298 m and its <=40 oracle count is 250/403 versus ORB 336/403.

This means the sparse XFeat failure is primarily a poor candidate-discrimination/ranking regime plus weaker precise projection, not complete absence of useful correspondences.

## Next gate

Do not tune sparse XFeat thresholds. Test the pinned implementation's distinct semi-dense XFeat* mode once, using the same frozen Top20 and homography semantics. If semi-dense XFeat still fails to materially close the ORB ranking/projection gap, stop XFeat and move to a different learned matcher family (LightGlue next).

No blind-stress reference was loaded and no frontend winner was promoted.
