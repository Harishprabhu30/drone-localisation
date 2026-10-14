#!/usr/bin/env python3
"""CV3.1A: CUDA/source preflight for the aerial-MASt3R upper-bound experiment."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
ROOT = REPO_ROOT
DEFAULT_CONFIG = ROOT / "configs/research/crossview_scale_rotation_domain_cv3_1a.yaml"


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
    if p.returncode != 0:
        return ""
    return p.stdout.strip()


def disk_info(path: Path) -> dict[str, int]:
    anchor = path
    while not anchor.exists() and anchor != anchor.parent:
        anchor = anchor.parent
    usage = shutil.disk_usage(anchor)
    return {
        "total_bytes": int(usage.total),
        "used_bytes": int(usage.used),
        "free_bytes": int(usage.free),
    }


def import_status(name: str) -> dict[str, Any]:
    try:
        module = importlib.import_module(name)
        return {
            "available": True,
            "version": str(getattr(module, "__version__", "unknown")),
            "path": str(getattr(module, "__file__", "")),
        }
    except Exception as exc:
        return {
            "available": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


def verify_config(cfg: dict[str, Any]) -> None:
    if cfg.get("stage") != "CV3.1A":
        raise RuntimeError("Expected CV3.1A config")
    p = cfg["protocol"]
    if int(p["image_load_size"]) != 512 or str(p["dtype"]) != "float32":
        raise RuntimeError("CV3.1A model/input contract changed")
    if p["require_cuda"] is not True:
        raise RuntimeError("CV3.1A requires CUDA")
    for key in ("allow_autocast", "allow_fp16", "allow_dataset_download", "allow_model_download"):
        if p[key] is not False:
            raise RuntimeError(f"{key} must remain false in CV3.1A")
    for key, value in cfg["scientific_boundary"].items():
        if value is not False:
            raise RuntimeError(f"{key} must remain false in CV3.1A")


def main() -> None:
    import yaml

    parser = argparse.ArgumentParser(
        description="CV3.1A CUDA/source preflight; no model or dataset download"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()

    cfg_path = resolve(args.config)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    verify_config(cfg)

    imports = {
        name: import_status(str(name))
        for name in cfg["required_python_imports"]
    }

    torch_info: dict[str, Any] = {}
    torch_module = None
    if imports["torch"]["available"]:
        torch_module = importlib.import_module("torch")
        cuda_available = bool(torch_module.cuda.is_available())
        torch_info = {
            "version": str(torch_module.__version__),
            "cuda_build": str(torch_module.version.cuda),
            "cuda_available": cuda_available,
        }
        if cuda_available:
            free_b, total_b = torch_module.cuda.mem_get_info()
            prop = torch_module.cuda.get_device_properties(0)
            torch_info.update({
                "device_count": int(torch_module.cuda.device_count()),
                "device_name": str(torch_module.cuda.get_device_name(0)),
                "compute_capability": list(torch_module.cuda.get_device_capability(0)),
                "device_total_memory_bytes": int(prop.total_memory),
                "cuda_free_memory_bytes_at_preflight": int(free_b),
                "cuda_total_memory_bytes_at_preflight": int(total_b),
            })

    upstream_root = resolve(cfg["upstream"]["local_repo"])
    source = {
        "present": upstream_root.is_dir(),
        "path": str(upstream_root),
        "head": None,
        "dirty_tracked": None,
        "pinned_commit": cfg["upstream"]["pinned_commit"],
        "pinned_match": False,
        "required_files": {},
    }
    if upstream_root.is_dir():
        source["head"] = git_value(upstream_root, "rev-parse", "HEAD") or None
        source["dirty_tracked"] = bool(
            git_value(upstream_root, "status", "--porcelain", "--untracked-files=no")
        )
        source["pinned_match"] = source["head"] == cfg["upstream"]["pinned_commit"]
        for rel in cfg["required_source_files"]:
            path = upstream_root / str(rel)
            source["required_files"][str(rel)] = path.is_file()
    else:
        for rel in cfg["required_source_files"]:
            source["required_files"][str(rel)] = False

    lineage = {}
    for name, path_key, hash_key in (
        ("qv1_4_top20", "qv1_4_top20", "expected_top20_sha256"),
        ("sift_lightglue_full403", "sift_lightglue_full403", "expected_sift_lightglue_sha256"),
    ):
        path = resolve(cfg["frozen_lineage"][path_key])
        lineage[name] = {
            "path": str(path),
            "present": path.is_file(),
            "sha256": sha256_file(path) if path.is_file() else None,
            "expected_sha256": cfg["frozen_lineage"][hash_key],
            "hash_match": (
                path.is_file()
                and sha256_file(path) == cfg["frozen_lineage"][hash_key]
            ),
        }

    repo_head = git_value(ROOT, "rev-parse", "HEAD")
    branch = git_value(ROOT, "branch", "--show-current")

    blockers: list[str] = []
    if not imports["torch"]["available"]:
        blockers.append("torch_unavailable")
    elif not torch_info.get("cuda_available", False):
        blockers.append("cuda_unavailable")

    for name, status in imports.items():
        if not status["available"]:
            blockers.append(f"python_import_missing:{name}")

    if not source["present"]:
        blockers.append("aerial_megadepth_source_missing")
    else:
        if not source["pinned_match"]:
            blockers.append("aerial_megadepth_source_not_at_pinned_commit")
        missing_files = [k for k, v in source["required_files"].items() if not v]
        if missing_files:
            blockers.append("aerial_megadepth_required_source_files_missing")
        if source["dirty_tracked"]:
            blockers.append("aerial_megadepth_tracked_files_modified")

    # Frozen lineage is informative on a compute host. It is not required for
    # the environment preflight because assets may be transferred only after
    # the host is judged suitable.
    status = (
        "PASS_CV3_1A_ENVIRONMENT_SOURCE_PREFLIGHT"
        if not blockers
        else "BLOCKED_CV3_1A_ENVIRONMENT_SOURCE_PREFLIGHT"
    )

    report = {
        "stage": "CV3.1A",
        "status": status,
        "blockers": blockers,
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python_version": sys.version,
            "hostname": platform.node(),
            "repo_branch": branch,
            "repo_head": repo_head,
        },
        "protocol": {
            "checkpoint_id": cfg["upstream"]["checkpoint_id"],
            "checkpoint_base_model": cfg["upstream"]["checkpoint_base_model"],
            "image_load_size": int(cfg["protocol"]["image_load_size"]),
            "dtype": cfg["protocol"]["dtype"],
            "model_download_executed": False,
            "dataset_download_executed": False,
            "inference_executed": False,
        },
        "python_imports": imports,
        "torch": torch_info,
        "source": source,
        "disk": {
            "repo": disk_info(ROOT),
            "torch_cache": disk_info(Path(os.environ.get("TORCH_HOME", Path.home() / ".cache/torch"))),
            "huggingface_cache": disk_info(Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface"))),
        },
        "frozen_lineage": lineage,
        "scientific_boundary": {
            "reference_loaded": False,
            "oracle_labels_loaded": False,
            "checkpoint_downloaded": False,
            "model_loaded": False,
            "inference_executed": False,
            "aerialmegadepth_dataset_downloaded": False,
            "uniform40_executed": False,
            "full403_executed": False,
        },
    }

    output_root = resolve(cfg["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    report_path = output_root / "cv3_1a_environment_report.json"
    payload = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf-8")
    if report_path.exists() and report_path.read_bytes() != payload:
        raise RuntimeError(
            "Refusing to overwrite differing CV3.1A report: " + str(report_path)
        )
    if not report_path.exists():
        report_path.write_bytes(payload)

    print("Status:", status)
    print("Blockers:", ",".join(blockers) if blockers else "NONE")
    print("Host:", platform.node(), platform.machine())
    print("Python:", sys.version.split()[0])
    print("Torch:", torch_info.get("version", "UNAVAILABLE"))
    print("Torch CUDA build:", torch_info.get("cuda_build", "UNAVAILABLE"))
    print("CUDA available:", torch_info.get("cuda_available", False))
    if torch_info.get("cuda_available", False):
        print("GPU:", torch_info.get("device_name"))
        print(
            "CUDA free/total GiB:",
            f"{torch_info['cuda_free_memory_bytes_at_preflight']/2**30:.2f}/"
            f"{torch_info['cuda_total_memory_bytes_at_preflight']/2**30:.2f}",
        )
    print("AerialMegaDepth source present:", source["present"])
    print("AerialMegaDepth HEAD:", source["head"] or "NONE")
    print("Pinned source match:", source["pinned_match"])
    print("Tracked source modified:", source["dirty_tracked"])
    print("Checkpoint downloaded: False")
    print("Model loaded: False")
    print("Inference executed: False")
    print("Dataset downloaded: False")
    print("Reference loaded: False")
    print("Report:", report_path)


if __name__ == "__main__":
    main()
