#!/usr/bin/env python3
"""CV3.1B0: acquire/hash aerial-MASt3R and test FP32 model load with zero images."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
ROOT = REPO_ROOT
DEFAULT_CONFIG = ROOT / "configs/research/crossview_scale_rotation_domain_cv3_1b0.yaml"


def resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def git_value(repo: Path, *args: str) -> str:
    p = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=False,
    )
    return p.stdout.strip() if p.returncode == 0 else ""


def verify_config(cfg: dict[str, Any]) -> None:
    if cfg.get("stage") != "CV3.1B0":
        raise RuntimeError("Expected CV3.1B0 config")
    m = cfg["model"]
    if m["device"] != "cuda" or m["dtype"] != "float32":
        raise RuntimeError("CV3.1B0 device/dtype contract changed")
    for key in ("inference_allowed", "autocast_allowed", "fp16_allowed"):
        if m[key] is not False:
            raise RuntimeError(f"{key} must remain false")
    for key, value in cfg["scientific_boundary"].items():
        if value is not False:
            raise RuntimeError(f"{key} must remain false in CV3.1B0")


def memavailable_bytes() -> int:
    with open("/proc/meminfo", "r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    return -1


def checkpoint_files_from_snapshot(
    snapshot_dir: Path,
    filenames: list[str],
) -> dict[str, Path]:
    """Return named snapshot paths without dereferencing Hub-cache symlinks.

    Hugging Face snapshots contain filenames (for example config.json and
    model.safetensors) that may be symlinks into the cache's blobs directory.
    Calling Path.resolve() on those files destroys the snapshot layout needed
    by PyTorchModelHubMixin.from_pretrained(local_directory).
    """
    snapshot_dir = snapshot_dir.expanduser()
    if not snapshot_dir.is_dir():
        raise RuntimeError(f"Checkpoint snapshot directory missing: {snapshot_dir}")

    files: dict[str, Path] = {}
    for filename in filenames:
        path = snapshot_dir / filename
        if not path.is_file():
            raise RuntimeError(
                f"Checkpoint snapshot missing required file {filename}: {path}"
            )
        files[filename] = path
    return files


def main() -> None:
    import yaml
    import torch
    from huggingface_hub import snapshot_download

    parser = argparse.ArgumentParser(
        description="CV3.1B0 checkpoint acquisition and FP32 model-load gate"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()

    cfg_path = resolve(args.config)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    verify_config(cfg)

    a = cfg["cv3_1a"]
    a_report_path = resolve(a["report"])
    if sha256_file(a_report_path) != a["expected_report_sha256"]:
        raise RuntimeError("CV3.1A report hash changed")
    a_report = json.loads(a_report_path.read_text(encoding="utf-8"))
    if a_report.get("status") != a["expected_status"]:
        raise RuntimeError("CV3.1A was not PASS")

    source_root = resolve(cfg["upstream"]["source_root"])
    mast3r_root = resolve(cfg["upstream"]["mast3r_root"])
    if git_value(source_root, "rev-parse", "HEAD") != cfg["upstream"]["expected_source_commit"]:
        raise RuntimeError("AerialMegaDepth source commit changed")
    if git_value(source_root, "status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("AerialMegaDepth tracked source is modified")
    if not mast3r_root.is_dir():
        raise FileNotFoundError(mast3r_root)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    if torch.get_default_dtype() != torch.float32:
        raise RuntimeError("Torch default dtype is not float32")

    checkpoint = cfg["checkpoint"]
    repo_id = str(checkpoint["repo_id"])
    revision = str(checkpoint["revision"])

    print("CV3.1B0 checkpoint acquisition:")
    print("Repo:", repo_id)
    print("Revision:", revision)
    print("Dataset download: False")
    print("Image inference: False")

    download_started = time.perf_counter()
    snapshot_dir = Path(
        snapshot_download(
            repo_id=repo_id,
            revision=revision,
            allow_patterns=[str(x) for x in checkpoint["files"]],
        )
    ).expanduser()
    download_seconds = float(time.perf_counter() - download_started)

    # Critical: do NOT resolve() snapshot file paths. The Hub cache intentionally
    # exposes named files in snapshots/<commit>/ as symlinks into blobs/. The
    # named snapshot directory is what from_pretrained(local_directory) needs.
    downloaded = checkpoint_files_from_snapshot(
        snapshot_dir,
        [str(x) for x in checkpoint["files"]],
    )
    if snapshot_dir.name != revision:
        raise RuntimeError(
            f"Unexpected checkpoint snapshot revision directory: {snapshot_dir}"
        )

    model_path = downloaded[str(checkpoint["model_filename"])]
    observed_model_sha = sha256_file(model_path)
    if observed_model_sha != checkpoint["expected_model_sha256"]:
        raise RuntimeError(
            f"Model SHA mismatch: {observed_model_sha} != {checkpoint['expected_model_sha256']}"
        )
    model_size = model_path.stat().st_size
    if not (
        int(checkpoint["expected_model_size_bytes_min"])
        <= model_size
        <= int(checkpoint["expected_model_size_bytes_max"])
    ):
        raise RuntimeError(f"Unexpected model size: {model_size}")

    parents = {path.parent for path in downloaded.values()}
    if parents != {snapshot_dir}:
        raise RuntimeError(
            "Checkpoint files are not preserved inside the pinned snapshot directory"
        )
    checkpoint_dir = snapshot_dir

    output_root = resolve(cfg["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    worker_path = ROOT / "scripts/villoc/geometry/cv3_1b0_aerial_mast3r_model_load_worker.py"
    worker_report_path = output_root / "cv3_1b0_worker_report.json"
    if worker_report_path.exists():
        worker_report_path.unlink()

    system_available_before = memavailable_bytes()
    free_before, total_before = torch.cuda.mem_get_info()

    cmd = [
        sys.executable,
        str(worker_path),
        "--mast3r-root",
        str(mast3r_root),
        "--checkpoint-dir",
        str(checkpoint_dir),
        "--output",
        str(worker_report_path),
    ]
    env = os.environ.copy()
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"

    started = time.perf_counter()
    proc = subprocess.run(
        cmd,
        env=env,
        capture_output=True,
        text=True,
        timeout=int(cfg["memory"]["worker_timeout_seconds"]),
        check=False,
    )
    worker_seconds = float(time.perf_counter() - started)

    worker_report = None
    if worker_report_path.is_file():
        worker_report = json.loads(worker_report_path.read_text(encoding="utf-8"))

    blockers: list[str] = []
    if proc.returncode != 0:
        blockers.append(f"model_load_worker_returncode:{proc.returncode}")
    if worker_report is None:
        blockers.append("model_load_worker_report_missing")
    elif worker_report.get("status") != "PASS_WORKER_CV3_1B0_FP32_MODEL_LOAD":
        blockers.append("model_load_worker_not_pass")

    warnings: list[str] = []
    if worker_report is not None:
        loaded = next(
            x for x in worker_report["snapshots"]
            if x["label"] == "after_cuda_model_load"
        )
        threshold = float(cfg["memory"]["low_headroom_warning_gib"]) * 2**30
        if loaded["cuda_free_bytes"] < threshold:
            warnings.append("low_cuda_headroom_after_model_load")
        if loaded["system_mem_available_bytes"] < threshold:
            warnings.append("low_system_headroom_after_model_load")

    status = (
        "PASS_CV3_1B0_FP32_MODEL_LOAD_GATE"
        if not blockers
        else "BLOCKED_CV3_1B0_FP32_MODEL_LOAD_GATE"
    )

    report = {
        "stage": "CV3.1B0",
        "status": status,
        "blockers": blockers,
        "warnings": warnings,
        "checkpoint": {
            "repo_id": repo_id,
            "revision": revision,
            "snapshot_dir": str(checkpoint_dir),
            "files": {
                name: {
                    "path": str(path),
                    "sha256": sha256_file(path),
                    "size_bytes": int(path.stat().st_size),
                }
                for name, path in downloaded.items()
            },
            "download_seconds": download_seconds,
        },
        "host_before_worker": {
            "system_mem_available_bytes": system_available_before,
            "cuda_free_bytes": int(free_before),
            "cuda_total_bytes": int(total_before),
        },
        "worker": {
            "returncode": int(proc.returncode),
            "elapsed_seconds": worker_seconds,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "report": worker_report,
        },
        "scientific_boundary": {
            "reference_loaded": False,
            "oracle_labels_loaded": False,
            "checkpoint_downloaded": True,
            "model_load_attempted": True,
            "image_loaded": False,
            "pair_inference_executed": False,
            "correspondence_extraction_executed": False,
            "aerialmegadepth_dataset_downloaded": False,
            "threshold_tuned": False,
            "uniform40_executed": False,
            "full403_executed": False,
        },
        "interpretation": {
            "load_gate_only": True,
            "fp32_only": True,
            "no_accuracy_claim": True,
            "pass_does_not_guarantee_inference_memory_fit": True,
        },
        "provenance": {
            "config_sha256": sha256_file(cfg_path),
            "cv3_1a_report_sha256": sha256_file(a_report_path),
            "aerialmegadepth_source_commit": git_value(source_root, "rev-parse", "HEAD"),
            "model_safetensors_sha256": observed_model_sha,
        },
    }

    report_path = output_root / "cv3_1b0_model_load_gate_report.json"
    payload = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf-8")
    if report_path.exists() and report_path.read_bytes() != payload:
        raise RuntimeError("Refusing to overwrite differing CV3.1B0 report")
    if not report_path.exists():
        report_path.write_bytes(payload)

    print("Status:", status)
    print("Blockers:", ",".join(blockers) if blockers else "NONE")
    print("Warnings:", ",".join(warnings) if warnings else "NONE")
    print("Checkpoint revision:", revision)
    print("Model SHA256:", observed_model_sha)
    print("Model size GiB:", f"{model_size / 2**30:.3f}")
    print("Download s:", f"{download_seconds:.3f}")
    if worker_report is not None:
        print("Worker status:", worker_report.get("status"))
        print("Parameter count:", worker_report.get("parameter_count"))
        print("Parameter GiB:", f"{worker_report.get('parameter_bytes', 0)/2**30:.3f}")
        loaded = next(
            x for x in worker_report["snapshots"]
            if x["label"] == "after_cuda_model_load"
        )
        unloaded = next(
            x for x in worker_report["snapshots"]
            if x["label"] == "after_model_unload"
        )
        print("CUDA free GiB after load:", f"{loaded['cuda_free_bytes']/2**30:.3f}")
        print("System available GiB after load:", f"{loaded['system_mem_available_bytes']/2**30:.3f}")
        print("Torch peak allocated GiB:", f"{loaded['torch_peak_allocated_bytes']/2**30:.3f}")
        print("CUDA free GiB after unload:", f"{unloaded['cuda_free_bytes']/2**30:.3f}")
        print("System available GiB after unload:", f"{unloaded['system_mem_available_bytes']/2**30:.3f}")
    print("Image inference executed: False")
    print("Dataset downloaded: False")
    print("Reference loaded: False")
    print("Report:", report_path)


if __name__ == "__main__":
    main()
