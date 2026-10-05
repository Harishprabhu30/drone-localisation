#!/usr/bin/env python3
"""Read-only environment/source inventory; no inference or reference reads."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def command(argv: list[str], root: Path) -> dict:
    try:
        result = subprocess.run(argv, cwd=root, capture_output=True, text=True,
                                timeout=15, check=False)
        return {"returncode": result.returncode, "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"error": str(exc)}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--asset", action="append", type=Path, default=[],
                        help="Explicit video/map/weight file to hash; repeatable.")
    parser.add_argument("--skip-runtime-imports", action="store_true")
    args = parser.parse_args()
    root = args.repo_root.resolve()
    report = {
        "stage": "JETSON_NATIVE_ENVIRONMENT_INVENTORY",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(root), "python": sys.version,
        "executable": sys.executable, "prefix": sys.prefix,
        "base_prefix": sys.base_prefix, "platform": platform.platform(),
        "machine": platform.machine(),
        "git": {
            "head": command(["git", "rev-parse", "HEAD"], root),
            "branch": command(["git", "branch", "--show-current"], root),
            "status": command(["git", "status", "--short"], root),
        },
        "system_files": {}, "modules": {}, "files": {},
    }
    for name in ("/etc/os-release", "/etc/nv_tegra_release",
                 "/proc/device-tree/model"):
        path = Path(name)
        if path.is_file():
            report["system_files"][name] = path.read_bytes().decode(
                "utf-8", errors="replace").strip("\x00\n ")
    if not args.skip_runtime_imports:
        for name in ("numpy", "PIL", "cv2", "torch"):
            try:
                module = importlib.import_module(name)
                info = {"version": getattr(module, "__version__", None),
                        "path": getattr(module, "__file__", None)}
                report["modules"][name] = info
                if name == "cv2":
                    info.update(threads=module.getNumThreads(),
                                optimized=module.useOptimized(),
                                build_information=module.getBuildInformation())
                    try:
                        info["cuda_devices"] = module.cuda.getCudaEnabledDeviceCount()
                    except Exception as exc:
                        info["cuda_error"] = str(exc)
                if name == "torch":
                    available = module.cuda.is_available()
                    info.update(cuda_build=module.version.cuda,
                                cuda_available=available)
                    if available:
                        info["cuda_devices"] = [
                            {"index": i, "name": module.cuda.get_device_name(i),
                             "capability": list(module.cuda.get_device_capability(i)),
                             "total_memory_bytes": module.cuda.get_device_properties(i).total_memory}
                            for i in range(module.cuda.device_count())
                        ]
            except Exception as exc:
                report["modules"][name] = {"error": str(exc)}
    sources = [
        "scripts/demo/run_recorded_flight_demo.py",
        "scripts/villoc/s8_11bc_build_dinov2_caches.py",
        "scripts/villoc/s8_12e1_top20_verifier_reranker.py",
        "scripts/villoc/blind_demo/bootstrap/minimum_confident_v2_backend.py",
        "scripts/villoc/research/minimum_confident_bootstrap/diagnostics/r4_11_blind_subtile_projection_recompute.py",
        "configs/bootstrap/minimum_confident_v2/architecture_contract.json",
        "configs/bootstrap/minimum_confident_v2/final_policy_contract.json",
    ]
    for given in [Path(s) for s in sources] + args.asset:
        path = given if given.is_absolute() else root / given
        if path.is_file():
            report["files"][str(given)] = {"path": str(path.resolve()),
                                          "bytes": path.stat().st_size,
                                          "sha256": digest(path)}
        else:
            report["files"][str(given)] = {"path": str(path), "missing": True}
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
