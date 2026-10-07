# New Mac migration closeout — 2026-10-07

Status: **CLOSED / PASS**

This note closes the migration of the frozen Villoc blind-localization development environment from the old Intel Mac to the newer Intel i7 Mac. It is intentionally short; the benchmark and algorithm closeouts remain authoritative for scientific results.

## Frozen reference

- Repository baseline before this documentation-only closeout: `0f71a2d27ef299d78417904ca381815e9e7c9fd8`.
- Tag: `jetson-native-baseline-20261007`.
- Promoted blind configuration: `configs/demo_villoc_traj01_90deg_stable120m_blind_v2.yaml`.
- Bootstrap backend: `minimum_confident_v2`.
- Relative frontend: XFeat.
- Absolute retrieval: DINOv2 ViT-S/14, img518, center-square, average-patch.
- Map: ORT10LT 2024–2026, EPSG:3346, promoted `512_s256` cache.
- Geometric verification: ORB Top-20.
- Strict blind rule remains unchanged: no SRT/GPS/GNSS/reference/oracle/evaluation error before freeze.

## New Mac environment

- Intel x86_64.
- Python 3.10.22 in `.drone_venv`.
- Core parity stack restored around NumPy 1.26.4, PyTorch 2.2.2 and TorchVision 0.17.2.
- CUDA unavailable and MPS unavailable on this Intel Mac; CPU execution is expected.

## Migration evidence

All critical transferred assets and Git topology were verified before runtime testing.

DINOv2:
- checkpoint SHA256 matched the frozen checkpoint;
- real UAV inference passed;
- recomputed descriptors for q1, q57 and q228 differed from old-Mac CPU cache only at about 1e-8 to 1e-7 absolute scale;
- cosine similarity was effectively 1.0;
- Top-1, Top-20 membership and exact Top-20 ordering were identical for all three probes.

XFeat:
- pinned checkout `e92685f57f8318b18725c5c8c0bd28c7fe188d9a`;
- real two-frame inference/matching passed;
- old/new first-pair matches = 711, inliers = 479, inlier ratio = 0.6736990154711674;
- affine differences were floating-point rounding only.

Orchestration:
- promoted 17-stage dry-run passed;
- a 45-query re-encoded smoke video was used only as an orchestration test;
- the complete promoted blind chain passed with `PASS_DEMO_STAGE9B_ORCHESTRATED_BLIND_RUN`;
- post-run certification passed with `STAGE4EB3_NEW_MAC_MIGRATION_CERTIFICATION=PASS`;
- blind freeze/hash contract passed;
- repository working tree remained clean.

The re-encoded 45-query smoke result is **not** a scientific accuracy baseline. Scientific migration parity is supported by the canonical DINO/XFeat comparisons above.

## Dependency gap discovered

Stage-01 environment preflight requires `psutil`, but `psutil` was not present in the dependency set restored during the initial migration. Installing it resolved the only orchestration preflight failure.

Do not rewrite the frozen benchmark/tag for this. The dependency declaration should be corrected as part of the next infrastructure/reproducibility work on a development branch.

## Decision

No full 403-query Mac rerun is required solely for migration.

The old-Mac -> new-Mac migration is technically closed. The next development line is `research/retrieval-candidate-pool-v2`, created from the current documentation-updated `main`, while `jetson-native-baseline-20261007` remains the immutable frozen reference.
