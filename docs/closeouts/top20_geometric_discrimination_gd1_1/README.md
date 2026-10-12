# GD1.1 closeout — richer ORB geometry is strongly discriminative

Status: **PASS / CLOSED**.

GD1.1 post-freeze development audit status:
PASS_GD1_1_POSTFREEZE_GEOMETRIC_DISCRIMINATION_AUDIT

Feature discrimination SHA256:
8ae2e25da535b448b4f0f3782c8afb5ecfc42afb4aa243d26896d52cffc0d409

## Retrieval baseline

contains_query:
Top1 174/403, Top3 238/403, Top5 258/403, Top10 325/403, Top20 384/403

center_le40:
Top1 101/403, Top3 153/403, Top5 178/403, Top10 215/403, Top20 277/403

center_le80:
Top1 171/403, Top3 235/403, Top5 252/403, Top10 320/403, Top20 380/403

## Main result

Richer geometry derived from the existing ORB frontend already carries strong
candidate-discrimination signal. The current bottleneck is therefore not simply
"ORB cannot match the imagery".

For containment:
- inliers: AUC* 0.8558, Top1 292/403, Top5 349/403
- orientation MAD: AUC* 0.8539, Top1 251/403, Top5 338/403
- orientation resultant: AUC* 0.8483, Top1 274/403, Top5 329/403
- Jacobian anisotropy: AUC* 0.8261, Top1 281/403, Top5 333/403

For center <=40 m:
- Jacobian anisotropy: AUC* 0.8571, Top1 128/403, Top5 230/403
- inliers: AUC* 0.8565, Top1 150/403, Top5 249/403
- orientation resultant: AUC* 0.8513, Top1 117/403, Top5 233/403
- orientation MAD: AUC* 0.8477, Top1 105/403, Top5 233/403

For center <=80 m:
- inliers: AUC* 0.8611, Top1 287/403, Top5 346/403
- orientation MAD: AUC* 0.8599, Top1 242/403, Top5 335/403
- orientation resultant: AUC* 0.8559, Top1 264/403, Top5 328/403
- Jacobian anisotropy: AUC* 0.8423, Top1 271/403, Top5 328/403

Single-feature inlier ranking raises Top1 containment from 174 to 292 and Top1
<=80 from 171 to 287. It recovers roughly three quarters of the queries where
the useful region is available anywhere in Top20.

Top5 geometry ranking is also close to the original Top20 availability ceiling:
- containment 349/384 available
- <=40 249/277 available
- <=80 346/380 available

## Scientific interpretation

Several different frozen evidence families are independently informative:
- correspondence strength;
- orientation consistency;
- local projective shape;
- scale consistency;
- projective denominator stability;
- footprint plausibility.

This makes a multivariate geometric selector scientifically justified before
replacing the matcher frontend.

Pair-level AUC and within-query ranking are different diagnostics. Some features
have strong pair AUC but weak Top1 ranking, so candidate selection must be
evaluated query-by-query, not by AUC alone.

No reranker/selector was promoted from GD1.1. The direction of each feature was
learned from development labels and remains diagnostic only.

## Next stage

GD1.2 should test whether the *predeclared GD1.0 evidence families* combine into
a stronger query-level selector than:
1. original retrieval rank;
2. the best single ORB evidence variable.

To reduce same-query overfitting, use contiguous query-block out-of-fold
evaluation. Do not tune hyperparameters. Compare:
- each frozen evidence family separately;
- all geometry;
- all geometry + frozen retrieval rank.

No deployment model should be exported from the single development trajectory.
