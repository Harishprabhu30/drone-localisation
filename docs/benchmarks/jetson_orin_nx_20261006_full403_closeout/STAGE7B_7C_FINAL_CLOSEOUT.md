# Stage 7B / 7C final Jetson benchmark closeout — 2026-10-07

Status: COMPLETE

Branch: `benchmark/jetson-native-blind-baseline`

Canonical run:
`traj01_90deg_stable120m_blind_v2_jetson_001`

Canonical run source:
`cc81e8bd11e10d75cb749b8fbc21cc00a2a6dc14`

## Stage 7B runtime and telemetry

Whole run:
- 403 queries
- 1499.237 s wall
- 3.720 s/query
- 0.269 query/s
- measured runtime registry coverage 97.86%

Major wall-time contributors:
- ORB Top-20 verifier/reranker: 568.989 s (37.952%)
- recorded-video frame extraction: 505.864 s (33.741%)
- blind map bootstrap: 195.411 s (13.034%)
- DINO query encoding: 122.234 s (8.153%)
- XFeat relative frontend: 65.934 s (4.398%)
- cached DINO Top-K ranking: 0.029 s total

Recorded-video extraction is an offline benchmark-path cost. Excluding that
preprocessing, measured localization time is approximately 952.685 s for 403
queries = 2.364 s/query = 0.423 query/s.

The primary measured localization bottleneck is ORB Top-20 verification.

Telemetry over 1,758 tegrastats samples:
- RAM mean 1628.9 MB, p95 1963.3 MB, max 2240 MB
- swap mean 153.1 MB, max 162 MB
- GR3D mean 3.88%, max 99%
- GPU nonzero in 4.72% of samples
- GPU >=50% in 4.04% of samples
- CPU max 59.25 C
- GPU max 58.06 C
- TJ max 59.25 C
- VDD_IN mean 8.838 W, p95 11.702 W, max 13.391 W

No successful live per-stage RSS files were captured, so per-stage RAM peaks
remain unmeasured. Full-run telemetry shows no OOM or thermal-pressure issue.
The pipeline is not GPU-saturated; most runtime is CPU/preprocessing/state work.

## Stage 7C native CUDA map-cache study

Final comparison:
A. existing canonical Mac/CPU-built map cache
B. native Jetson/CUDA-built map cache

The Jetson CPU rebuild was intentionally omitted because the deployment target
uses the GPU path. This means the comparison is deployment-oriented and does not
separate machine effects from CPU-vs-CUDA effects.

Native CUDA map cache:
`outputs/benchmarks/stage7c_gpu_map_cache_jetson_001/map_descriptors/s8_11b_dinov2_map_512_s256_dinov2_vits14_img518_center_square_avgpatch_cuda.npz`

SHA256:
`6af11c5a380a03dcec6206896e71f3b6064d644cc99a533f4111711e30069bc8`

Index SHA256:
`894369e5800a5ae9b7976d530905b591271db1a8577d8058414ca67a54d9aa2c`

Protocol:
- 475 x 384 descriptors
- DINOv2 ViT-S/14
- img518
- center_square
- avgpatch
- CUDA
- checkpoint SHA256
  `b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9`

Map descriptor runtime:
- historical CPU metadata: 955.130 s
- Jetson CUDA metadata: 91.662 s
- speedup: 10.42x
- complete builder wall: 97.29 s

Descriptor parity:
- tile IDs identical
- mean abs delta 8.141e-08
- p95 abs delta 2.068e-07
- max abs delta 7.097e-07
- minimum same-tile cosine 0.9999999999935485

Retrieval parity using the frozen CUDA query cache:
- Top-1 identical: 403/403
- Top-20 exact ordering: 403/403
- Top-20 membership: 20/20 for every query
- q57 unchanged
- q228 unchanged

Conclusion:
the native Jetson CUDA map cache is retrieval-decision equivalent to the
canonical CPU-built cache for this dataset/protocol while being about 10.42x
faster to generate. No downstream ORB or full-pipeline replay is justified for
map-cache parity because every query presents the same ordered Top-20 tile list
to the verifier.

## Native benchmark final status

The native Jetson benchmark is now closed.

Proven:
- full 403-query strict-blind pipeline executes natively
- 17/17 stages pass
- XFeat and DINO queries run on CUDA
- map generation can run natively on CUDA
- runtime/resource bottlenecks are measured
- post-freeze accuracy/failure modes are documented

Still unresolved research problems:
- 79.02 m absolute RMSE
- low-observability forest retrieval
- rare bad sub-tile/homography observations
- harmful multi-evidence state replacement
- state trust/recovery
- runtime below 1 Hz

Keep the benchmark branch as the reference baseline. Start future localization,
retrieval, spatial-search or adaptive-state research from a deliberate new
research branch rather than editing this closed benchmark directly.
