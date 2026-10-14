# CV3.1B0 — aerial-MASt3R checkpoint + FP32 model-load gate

Status: **IMPLEMENTATION PREPARED / JETSON EXECUTION REQUIRED**

Parent:
CV3.1A Jetson preflight PASS.

CV3.1A report SHA256:
\`eddabfb304cd57290a27d21d025cc20c630f154a694844c28052d64dce6a8400\`

## Checkpoint contract

Hugging Face:
\`kvuong2711/checkpoint-aerial-mast3r\`

Pinned checkpoint revision:
\`5ee8e004c8795e0c299b5f43acda27d0e02bcd97\`

Expected \`model.safetensors\` SHA256:
\`093c3b4b3ad8294673774aae106309d5f6a1a975976bae574075fb7c5f94a67d\`

Expected file size:
approximately 2.75 GB.

The stage downloads only:
- \`config.json\`
- \`model.safetensors\`

No AerialMegaDepth dataset is downloaded.

## Why the model load runs in a child process

Jetson Orin has ~7.44 GiB unified memory and the checkpoint contains about 0.7B
FP32 parameters. A load/move failure may trigger a CUDA OOM or Linux OOM kill.

The parent process therefore:
1. downloads and hashes the checkpoint;
2. launches a local-only child worker;
3. records the worker exit code/stdout/stderr;
4. writes a BLOCKED report even when the worker dies before producing its own
   report.

The child:
- imports the pinned source;
- loads the model in FP32;
- verifies all parameters are FP32;
- moves the model to CUDA;
- records system/CUDA memory;
- performs **zero image inference**;
- unloads the model and empties the CUDA cache.

## Run

On the CV3 Jetson detached worktree, with the existing dependency overlay active:

\`\`\`bash
export PYTHONPATH=/home/rodu/projects/cv3_pydeps:$PWD/src\${PYTHONPATH:+:\$PYTHONPATH}

python -m unittest tests.test_cv3_1b0_aerial_mast3r_model_load_gate -v

python scripts/villoc/geometry/cv3_1b0_aerial_mast3r_model_load_gate.py \\
  --config configs/research/crossview_scale_rotation_domain_cv3_1b0.yaml
\`\`\`

The first run downloads ~2.75 GB and therefore needs network access and several
GiB of free disk.

## Continue condition

CV3.1B1 image smoke is allowed only if:
\`PASS_CV3_1B0_FP32_MODEL_LOAD_GATE\`.

A low-memory-headroom warning is informative. It does not automatically change
dtype or authorize FP16/autocast.

If FP32 model load is blocked:
stop the Jetson MASt3R attempt and reassess compute hardware. Do not silently
change precision/model architecture.


## First Jetson attempt diagnostic

The first CV3.1B0 attempt downloaded and verified the correct pinned checkpoint,
but the worker failed before model loading because the parent script called
`Path.resolve()` on Hugging Face snapshot symlinks. This converted paths such
as:

`snapshots/<commit>/config.json`

into their backing:

`blobs/<hash>`

and then passed the `blobs/` directory to
`AsymmetricMASt3R.from_pretrained()`.

That directory has no named `config.json`, so the failure was a loader-path
defect, **not evidence of Jetson FP32 memory failure**.

The corrected implementation uses `snapshot_download()`, preserves the named
snapshot directory, and includes a symlink-layout regression test.

Do not change precision/model settings because of the first blocked attempt.
