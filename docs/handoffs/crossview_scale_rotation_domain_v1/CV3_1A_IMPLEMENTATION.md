# CV3.1A — CUDA/source environment preflight

Status: **IMPLEMENTATION PREPARED / RUN ON TARGET CUDA HOST**

Parent protocol:
\`CV3_0_PROTOCOL.md\`

## Purpose

CV3 uses the pretrained aerial-MASt3R model as an accuracy upper bound. Before
downloading or loading the checkpoint, CV3.1A determines whether the intended
CUDA host is suitable.

This stage performs **no model download and no inference**.

It records:
- CUDA/PyTorch availability;
- GPU identity and free/total CUDA memory;
- Python dependency availability;
- disk free space;
- AerialMegaDepth source presence and pinned Git commit;
- whether required MASt3R/DUSt3R source files are present;
- whether the frozen QV1.4/SIFT evidence happens to be present on the compute
  host.

Frozen-lineage assets are informative but are not required to pass the hardware
preflight; they may be transferred after the host is accepted.

## Source preparation

The intended source location is:

\`third_party/aerial-megadepth\`

Pinned commit:

\`cdf478ff4fdaf888fb91c940bc48d55095e2cdbd\`

Do not download the AerialMegaDepth dataset.

If the source is not present, CV3.1A should report:

\`BLOCKED_CV3_1A_ENVIRONMENT_SOURCE_PREFLIGHT\`

with blocker \`aerial_megadepth_source_missing\`.

That is expected and is not a scientific failure. After hardware suitability is
confirmed, source setup can be performed before CV3.1B.

## Run

On the intended CUDA machine:

\`\`\`bash
python -m unittest tests.test_cv3_1a_aerial_mast3r_environment_preflight -v

python scripts/villoc/geometry/cv3_1a_aerial_mast3r_environment_preflight.py \\
  --config configs/research/crossview_scale_rotation_domain_cv3_1a.yaml
\`\`\`

Do not install FP16/autocast changes to make the model fit. CV3.1B begins in
FP32, matching the frozen protocol.

## Continue condition

Proceed to CV3.1B model/checkpoint smoke only after:
- CUDA is available;
- required Python imports are available;
- AerialMegaDepth source is at the pinned commit with clean tracked files.

Memory sufficiency is deliberately not guessed from device capacity alone. The
actual model-load/inference peak is measured in CV3.1B.
