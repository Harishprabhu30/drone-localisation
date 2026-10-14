#!/usr/bin/env python3
"""Child worker for CV3.1B0 MASt3R FP32 model-load memory measurement."""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


def memavailable_bytes() -> int:
    with open("/proc/meminfo", "r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    return -1


def rss_bytes() -> int:
    with open("/proc/self/status", "r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    return -1


def snapshot(torch, label: str) -> dict[str, Any]:
    free_b, total_b = torch.cuda.mem_get_info()
    return {
        "label": label,
        "system_mem_available_bytes": memavailable_bytes(),
        "process_rss_bytes": rss_bytes(),
        "cuda_free_bytes": int(free_b),
        "cuda_total_bytes": int(total_b),
        "torch_allocated_bytes": int(torch.cuda.memory_allocated()),
        "torch_reserved_bytes": int(torch.cuda.memory_reserved()),
        "torch_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "torch_peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mast3r-root", required=True, type=Path)
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    mast3r_root = args.mast3r_root.resolve()
    checkpoint_dir = args.checkpoint_dir.resolve()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    # Enforce local-only model loading in the worker.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    if str(mast3r_root) not in sys.path:
        sys.path.insert(0, str(mast3r_root))

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable in CV3.1B0 worker")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    before_import = snapshot(torch, "before_mast3r_import")

    from mast3r.model import AsymmetricMASt3R

    after_import = snapshot(torch, "after_mast3r_import")

    started = time.perf_counter()
    model = AsymmetricMASt3R.from_pretrained(str(checkpoint_dir))
    cpu_load_seconds = float(time.perf_counter() - started)
    after_cpu_load = snapshot(torch, "after_cpu_model_load")

    # Explicitly verify the frozen FP32 contract before moving to CUDA.
    dtypes = {str(p.dtype) for p in model.parameters()}
    parameter_count = int(sum(p.numel() for p in model.parameters()))
    parameter_bytes = int(sum(p.numel() * p.element_size() for p in model.parameters()))
    if dtypes != {"torch.float32"}:
        raise RuntimeError(f"Unexpected parameter dtypes before CUDA move: {sorted(dtypes)}")

    started = time.perf_counter()
    model = model.to("cuda")
    model.eval()
    torch.cuda.synchronize()
    cuda_move_seconds = float(time.perf_counter() - started)
    after_cuda_load = snapshot(torch, "after_cuda_model_load")

    result: dict[str, Any] = {
        "status": "PASS_WORKER_CV3_1B0_FP32_MODEL_LOAD",
        "parameter_count": parameter_count,
        "parameter_bytes": parameter_bytes,
        "parameter_dtypes": sorted(dtypes),
        "cpu_load_seconds": cpu_load_seconds,
        "cuda_move_seconds": cuda_move_seconds,
        "snapshots": [
            before_import,
            after_import,
            after_cpu_load,
            after_cuda_load,
        ],
        "model_training_flag_after_eval": bool(model.training),
        "inference_executed": False,
    }

    del model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    result["snapshots"].append(snapshot(torch, "after_model_unload"))
    result["unload_attempted"] = True

    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("Worker status:", result["status"])
    print("Parameter count:", parameter_count)
    print("Parameter GiB:", f"{parameter_bytes / 2**30:.3f}")
    print(
        "CUDA free GiB after load:",
        f"{after_cuda_load['cuda_free_bytes'] / 2**30:.3f}",
    )
    print(
        "System available GiB after load:",
        f"{after_cuda_load['system_mem_available_bytes'] / 2**30:.3f}",
    )
    print("Inference executed: False")


if __name__ == "__main__":
    main()
