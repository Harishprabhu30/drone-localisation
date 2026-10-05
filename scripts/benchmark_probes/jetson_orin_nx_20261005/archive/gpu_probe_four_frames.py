"""Isolated DINOv2 CUDA parity probe for the four fixed Mac frames."""

import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch


ROOT = Path.cwd().resolve()
SOURCE = ROOT / "scripts/villoc/s8_11bc_build_dinov2_caches.py"
CPU_CACHE = ROOT / (
    "outputs/benchmark_probes/fixed_mac_frames_dino_cpu_20261002_001/"
    "descriptors/s8_11c_dinov2_queries_v_1fps_"
    "dinov2_vits14_img518_center_square_avgpatch_cpu.npz"
)
OUT = ROOT / "outputs/benchmark_probes/fixed_mac_frames_dino_gpu_20261005_001"

spec = importlib.util.spec_from_file_location("frozen_dino_cache_builder", SOURCE)
assert spec is not None and spec.loader is not None
builder = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = builder
spec.loader.exec_module(builder)

assert torch.cuda.is_available(), "CUDA is unavailable in this Python environment"
assert builder.CHECKPOINT.exists(), f"Missing checkpoint: {builder.CHECKPOINT}"
assert CPU_CACHE.exists(), f"Missing four-frame CPU cache: {CPU_CACHE}"
os.environ["TORCH_HOME"] = str(Path.home() / ".cache/torch")
torch.manual_seed(7)
torch.cuda.manual_seed_all(7)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = False

with np.load(CPU_CACHE, allow_pickle=False) as cache:
    ids = cache["ids"].astype(str)
    saved_paths = cache["paths"].astype(str)
    cpu = cache["descriptors"].astype(np.float32)
assert len(ids) == 4 and cpu.shape == (4, 384), (ids, cpu.shape)
paths = [Path(p) if Path(p).is_absolute() else ROOT / p for p in saved_paths]
assert all(p.is_file() for p in paths), paths

t0 = time.perf_counter()
model = torch.hub.load(
    str(builder.TORCH_HUB_REPO), "dinov2_vits14", source="local", pretrained=True
).eval().to("cuda")
torch.cuda.synchronize()
load_s = time.perf_counter() - t0

rows = []
descs = []
with torch.inference_mode():
    for qid, path in zip(ids, paths):
        t1 = time.perf_counter()
        arr = builder.preprocess_image(path, image_size=518, crop_mode="center_square")
        x = torch.from_numpy(arr[None, ...]).to("cuda")
        patch = model.forward_features(x)["x_norm_patchtokens"]
        desc = patch.mean(dim=1).detach().float().cpu().numpy().astype(np.float32)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t1
        descs.append(desc)
        rows.append({"id": str(qid), "seconds_including_preprocess": elapsed})
        print(f"encoded id={qid} elapsed_s={elapsed:.3f}", flush=True)

gpu = builder.l2_normalize_np(np.vstack(descs).astype(np.float32)).astype(np.float32)
diff = np.abs(gpu - cpu)
cos = np.sum(gpu.astype(np.float64) * cpu.astype(np.float64), axis=1) / (
    np.linalg.norm(gpu.astype(np.float64), axis=1)
    * np.linalg.norm(cpu.astype(np.float64), axis=1)
)
for row, maxdiff, cosine in zip(rows, diff.max(axis=1), cos):
    row["max_abs_difference_vs_cpu"] = float(maxdiff)
    row["cosine_vs_cpu"] = float(cosine)

OUT.mkdir(parents=True, exist_ok=True)
report = {
    "status": "FOUR_FIXED_MAC_FRAMES_CUDA_PROBE",
    "device": torch.cuda.get_device_name(0),
    "torch": torch.__version__,
    "cuda_build": torch.version.cuda,
    "tf32": False,
    "dtype": "float32",
    "model_load_s": load_s,
    "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    "cpu_cache": str(CPU_CACHE),
    "rows": rows,
}
(OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
np.savez_compressed(
    OUT / "gpu_descriptors.npz", ids=ids, paths=saved_paths, descriptors=gpu
)
print(json.dumps(report, indent=2))
print("saved:", OUT)
