# CV3.0 protocol freeze — AerialMegaDepth / aerial-MASt3R upper bound

Status: **FROZEN PROTOCOL / ENVIRONMENT PREFLIGHT NEXT**

## Research question

> Does an aerial-domain-trained dense matching model materially reduce residual
> wrong-locality selection inside the frozen QV1.4 Top20 compared with the new
> SIFT+LightGlue development control?

CV3 is an accuracy upper-bound experiment. It is not initially a deployment
runtime candidate.

## Why CV3 is justified now

SIFT+LightGlue is the strongest current development frontend but still leaves:
- 30 / 403 unsafe selected projections;
- Top1 containment 349 / 403;
- Top1 center <=80 m 346 / 403;
- selected projection <=80 m 368 / 403.

The frozen retrieval pool contains center <=80 m candidates for 380 / 403
queries, so residual candidate-selection headroom remains.

## External model

Project:
\`AerialMegaDepth: Learning Aerial-Ground Reconstruction and View Synthesis\`
(CVPR 2025)

Official repository:
\`https://github.com/kvuong2711/aerial-megadepth\`

Repository commit pinned for CV3 source inspection:
\`cdf478ff4fdaf888fb91c940bc48d55095e2cdbd\`

Primary checkpoint:
\`kvuong2711/checkpoint-aerial-mast3r\`

Reason for MASt3R first:
the official AerialMegaDepth demo exposes dense local descriptor/confidence maps
and extracts image-to-image correspondences, which maps naturally onto the
current verifier/locality question.

Do not use the full AerialMegaDepth dataset in CV3.

## Official demo-derived correspondence defaults

The pinned official demo uses:
- image load size: 512
- correspondence extraction: \`extract_correspondences_nonsym\`
- subsample: 4
- pixel tolerance: 3
- match confidence threshold: 0.3

These are frozen as the first CV3 diagnostic settings because they come from
the authors' demo, not VILLOC truth.

Do not tune them on uniform40/full403.

## Geometry contract

CV3 must convert Aerial-MASt3R correspondence coordinates back into the
canonical query/tile image coordinate systems before map projection.

Then:
- query -> tile homography;
- RANSAC threshold: 5 px in canonical coordinate space;
- minimum 4 inliers;
- query-center projection;
- same raster pixel-center map convention as ORB/SIFT+LightGlue.

Primary candidate selector:
\`homography inliers DESC -> pool_rank ASC -> tile_id ASC\`.

Secondary reference-free evidence may include:
- extracted correspondence count;
- correspondence confidence sum/mean/median;
- inlier ratio;
- spatial support/coverage;
- symmetric reprojection residual;
- homography Jacobian/footprint diagnostics.

Do not promote a learned score combination in CV3 without a separate freeze.

## Frozen inputs

Development trajectory:
\`villoc_traj01_90deg_stable120m\`

QV1.4 Top20 SHA256:
\`0a7832c6f0997a33851343fd7df537d2f6410211ac0e38e7ec4325899d8df47e\`

Current SIFT+LightGlue full403 evidence SHA256:
\`c0031242f0b359f1bfa3c87ea47ac428393ed3e8e03d7b5c6d0822e8822cb8a0\`

Current SIFT+LightGlue development control:
- Top1 containment 349/403
- Top1 <=40 205/403
- Top1 <=80 346/403
- selected <=40 367/403
- selected <=80 368/403
- unsafe tail 30/403

## Stage sequence

### CV3.1 — environment/model smoke

Do not begin with uniform40.

Use the same 9 pair identities as CV2.1:
- query IDs: 1, 207, 403
- pool ranks: 1, 10, 20

Purpose:
- verify pinned AerialMegaDepth source/submodules;
- resolve and hash the aerial-MASt3R checkpoint;
- record model architecture/device/dtype;
- record peak memory and pair runtime;
- verify correspondence coordinates and canonical rescaling;
- verify homography/projection plumbing;
- confirm no reference truth is loaded.

No accuracy claim.

### CV3.2 — compact reference-free ranking screen

Only if CV3.1 passes and runtime/memory are feasible.

Use exact uniform40 but begin with a reduced frozen candidate slice:
\`pool ranks 1, 2, 3, 5, 10, 20\` = 240 pairs.

Reason:
aerial-MASt3R is an expensive accuracy upper bound. This compact screen first
tests whether its raw geometry has any useful candidate-separation signal before
committing to 800 dense-model pairs.

No truth until the 240-pair evidence is frozen.

### CV3.3 — post-freeze compact evaluation

Attach development truth only after CV3.2 freezes.

Continue to full uniform40 x Top20 only if aerial-MASt3R shows at least one of:
- +3/40 Top1 containment relative to the same six-candidate SIFT+LightGlue control;
- +3/40 Top1 <=80 m;
- >=3/40 unsafe-tail reduction without >2/40 loss in selected <=40 m.

If the compact screen fails, close aerial-MASt3R as a candidate-selection
upper bound for this protocol.

### CV3.4 — full uniform40 Top20

Only if CV3.3 passes:
freeze 40 x 20 = 800 aerial-MASt3R pairs, then evaluate post-freeze against
SIFT+LightGlue.

Do not jump directly to full403.

## Device policy

CV3 requires a GPU-capable environment for practical execution.

Jetson Orin may be tested, but is **not assumed** to be the CV3 device because
Aerial-MASt3R uses a large MASt3R architecture and shared-memory limits may be
binding.

Before model download/execution, record:
- device model;
- available RAM;
- CUDA/PyTorch versions;
- free storage;
- power mode / thermal state where relevant.

If Jetson cannot load the checkpoint safely, stop the Jetson attempt rather
than modifying the model/precision contract ad hoc. Use another CUDA-capable
environment for CV3 if available.

Initial dtype:
FP32 unless the upstream checkpoint/runtime requires otherwise.

Do not introduce FP16/autocast merely to force model fit without a separately
documented parity stage.

## Dataset/download boundary

Allowed for CV3:
- official AerialMegaDepth source repository/submodules;
- pretrained aerial-MASt3R checkpoint;
- existing VILLOC query/map assets.

Not allowed:
- full AerialMegaDepth training dataset;
- AerialMegaDepth fine-tuning;
- VILLOC-supervised fine-tuning;
- hidden/oracle candidate selection.

## Stop / continue logic

CV3 is an upper-bound experiment, not a commitment to deploy MASt3R.

If aerial-MASt3R materially beats SIFT+LightGlue:
- preserve it as an aerial-domain accuracy ceiling;
- then decide whether to distill/replace with a lighter aerial-specific
  frontend (CV4) or use it selectively/offboard.

If it does not materially beat SIFT+LightGlue:
- close CV3 without training;
- proceed to the next predeclared aerial-specific local/detector-free family.

## Jetson / deployment boundary

The earlier Jetson SIFT+LightGlue CUDA migration idea is parked, not cancelled.

Decision order:
1. finish cross-view accuracy family selection;
2. then benchmark/promote the chosen frontend on Jetson;
3. then return to acceptance authority, state/fusion and deployment work.

This avoids optimizing the runtime of a frontend that may immediately be
superseded by the next accuracy experiment.
