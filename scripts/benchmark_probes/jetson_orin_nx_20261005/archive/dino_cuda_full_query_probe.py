"""Isolated 123-query DINOv2 CUDA encoding, CPU parity, and Top-20 audit."""

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
CPU_QUERY = ROOT / (
    "outputs/demo_runs/jetson_native_cpu_baseline_20261002_002/descriptors/"
    "s8_11c_dinov2_queries_v_1fps_dinov2_vits14_img518_center_square_avgpatch_cpu.npz"
)
CPU_MAP = ROOT / (
    "outputs/villoc/90_deg/descriptors/"
    "s8_11b_dinov2_map_512_s256_dinov2_vits14_img518_center_square_avgpatch_cpu.npz"
)
OUT = ROOT / "outputs/benchmark_probes/jetson_native_dino_cuda_123_20261005_001"

spec = importlib.util.spec_from_file_location("frozen_dino_cache_builder", SOURCE)
assert spec is not None and spec.loader is not None
builder = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = builder
spec.loader.exec_module(builder)

assert torch.cuda.is_available(), "CUDA unavailable"
for path in (CPU_QUERY, CPU_MAP, builder.CHECKPOINT, builder.TORCH_HUB_REPO):
    assert path.exists(), f"Missing baseline asset: {path}"
os.environ["TORCH_HOME"] = str(Path.home() / ".cache/torch")
torch.manual_seed(7)
torch.cuda.manual_seed_all(7)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = False

with np.load(CPU_QUERY, allow_pickle=False) as cache:
    ids = cache["ids"].astype(str)
    saved_paths = cache["paths"].astype(str)
    cpu = cache["descriptors"].astype(np.float32)
with np.load(CPU_MAP, allow_pickle=False) as cache:
    map_ids = cache["ids"].astype(str)
    map_desc = cache["descriptors"].astype(np.float32)
assert len(ids) == 123 and cpu.shape == (123, 384), (len(ids), cpu.shape)
assert map_desc.ndim == 2 and map_desc.shape[1] == 384
assert len(map_ids) == len(map_desc)
paths = [Path(p) if Path(p).is_absolute() else ROOT / p for p in saved_paths]
assert all(p.is_file() for p in paths), "Missing query image in CPU cache"

start = time.perf_counter()
model = torch.hub.load(
    str(builder.TORCH_HUB_REPO), "dinov2_vits14", source="local", pretrained=True
).eval().to("cuda")
torch.cuda.synchronize()
load_s = time.perf_counter() - start

raw = []
seconds = []
encode_start = time.perf_counter()
with torch.inference_mode():
    for index, (query_id, path) in enumerate(zip(ids, paths), 1):
        start = time.perf_counter()
        image = builder.preprocess_image(path, image_size=518, crop_mode="center_square")
        x = torch.from_numpy(image[None, ...]).to("cuda")
        patches = model.forward_features(x)["x_norm_patchtokens"]
        desc = patches.mean(dim=1).detach().float().cpu().numpy().astype(np.float32)
        torch.cuda.synchronize()
        seconds.append(time.perf_counter() - start)
        raw.append(desc)
        if index % 25 == 0 or index == len(ids):
            print(f"encoded {index}/{len(ids)} | elapsed_s={time.perf_counter()-encode_start:.2f}", flush=True)
encode_s = time.perf_counter() - encode_start

gpu = builder.l2_normalize_np(np.vstack(raw).astype(np.float32)).astype(np.float32)
maxdiff = np.max(np.abs(gpu - cpu), axis=1)
cpu64, gpu64 = cpu.astype(np.float64), gpu.astype(np.float64)
cosines = np.sum(cpu64 * gpu64, axis=1) / (
    np.linalg.norm(cpu64, axis=1) * np.linalg.norm(gpu64, axis=1)
)
# Both rankings use the same map matrix and stable NumPy ordering; no GT is read.
cpu_score = cpu @ map_desc.T
gpu_score = gpu @ map_desc.T
cpu_order = np.argsort(-cpu_score, axis=1, kind="stable")[:, :20]
gpu_order = np.argsort(-gpu_score, axis=1, kind="stable")[:, :20]
overlap = np.array([len(set(a) & set(b)) for a, b in zip(cpu_order, gpu_order)])
top1_equal = cpu_order[:, 0] == gpu_order[:, 0]
changed = [
    {"id": str(ids[i]), "cpu_top1": str(map_ids[cpu_order[i, 0]]),
     "gpu_top1": str(map_ids[gpu_order[i, 0]]), "top20_overlap": int(overlap[i])}
    for i in range(len(ids)) if not top1_equal[i] or overlap[i] < 20
]

OUT.mkdir(parents=True, exist_ok=True)
report = {
    "stage": "ISOLATED_JETSON_CUDA_DINO_QUERY_PROBE",
    "blind_reference_used": False,
    "device": torch.cuda.get_device_name(0),
    "torch": torch.__version__, "cuda_build": torch.version.cuda,
    "tf32": False, "dtype": "float32", "batch_size": 1,
    "query_rows": len(ids), "map_rows": len(map_ids),
    "model_load_s": load_s, "encode_wall_s": encode_s,
    "median_query_s_including_preprocess": float(np.median(seconds)),
    "mean_query_s_including_preprocess": float(np.mean(seconds)),
    "max_abs_descriptor_difference": float(maxdiff.max()),
    "min_descriptor_cosine": float(cosines.min()),
    "top1_equal_queries": int(top1_equal.sum()),
    "top20_overlap_min": int(overlap.min()),
    "top20_overlap_mean": float(overlap.mean()),
    "changed_rankings": changed,
    "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    "cpu_query_cache": str(CPU_QUERY), "cpu_map_cache": str(CPU_MAP),
    "gpu_peak_allocated_mib": torch.cuda.max_memory_allocated() / (1024 ** 2),
}
(OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
np.savez_compressed(OUT / "gpu_query_descriptors_probe_only.npz",
                    ids=ids, paths=saved_paths, descriptors=gpu,
                    meta_json=np.asarray(json.dumps({"probe_only": True, "device": "cuda",
                                                 "tf32": False})))
print(json.dumps(report, indent=2), flush=True)
print("saved:", OUT, flush=True)
