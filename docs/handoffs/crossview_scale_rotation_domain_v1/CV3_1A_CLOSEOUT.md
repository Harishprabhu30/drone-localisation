# CV3.1A Jetson environment/source preflight closeout

Status: **PASS_CV3_1A_ENVIRONMENT_SOURCE_PREFLIGHT**

Compute host:
NVIDIA Jetson Orin NX Engineering Reference Developer Kit / aarch64.

Frozen local report SHA256:
\`eddabfb304cd57290a27d21d025cc20c630f154a694844c28052d64dce6a8400\`

Observed environment:
- Python 3.10.12
- PyTorch 2.8.0
- PyTorch CUDA build 12.6
- CUDA available: true
- GPU: Orin
- compute capability: 8.7
- CUDA/shared free memory at preflight: ~5.40 GiB / 7.44 GiB total
- system available RAM before CV3 model load: ~5.9 GiB
- swap: ~3.7 GiB total
- disk free on root/NVMe: ~18 GiB
- power mode: MAXN

Isolated dependency overlay:
\`/home/rodu/projects/cv3_pydeps\`

Verified additions:
- einops 0.8.2
- huggingface_hub 0.36.2
- safetensors 0.8.0

The NVIDIA-specific Torch install was not replaced.

Pinned AerialMegaDepth source:
\`third_party/aerial-megadepth\`

Pinned source commit:
\`cdf478ff4fdaf888fb91c940bc48d55095e2cdbd\`

Tracked source modified:
false.

Scientific boundary:
- checkpoint not downloaded in CV3.1A;
- model not loaded;
- inference not executed;
- AerialMegaDepth dataset not downloaded;
- reference truth not loaded.

## Decision

Proceed to CV3.1B0:
download/hash only the pinned pretrained aerial-MASt3R model files, then perform
an FP32 model-load/unload memory gate in a child process.

Do not run an image pair until CV3.1B0 completes.
