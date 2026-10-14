#!/usr/bin/env python3
"""Run and freeze the reference-free CV3.1B1 nine-pair aerial-MASt3R smoke."""
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
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import pandas as pd
import yaml
from huggingface_hub import snapshot_download

from uavloc.geometry.aerial_mast3r_smoke import sha256_file

ROOT = REPO_ROOT
DEFAULT_CONFIG = ROOT / "configs/research/crossview_scale_rotation_domain_cv3_1b1.yaml"


def resolve(v: str | Path) -> Path:
    p = Path(v).expanduser()
    return p if p.is_absolute() else ROOT / p


def git_value(repo: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.stdout.strip() if proc.returncode == 0 else ""


def canonical_csv(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, lineterminator="\n", float_format="%.9f").encode()


def canonical_json(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def freeze(files: dict[Path, bytes]) -> None:
    bad = [str(p) for p, d in files.items() if p.exists() and p.read_bytes() != d]
    if bad:
        raise RuntimeError("Refusing to overwrite differing CV3.1B1 outputs: " + ", ".join(bad))
    for p, d in files.items():
        if not p.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
            with p.open("xb") as handle:
                handle.write(d)


def verify_config(cfg: dict[str, Any]) -> None:
    if cfg.get("stage") != "CV3.1B1":
        raise RuntimeError("Expected CV3.1B1 config")
    if int(cfg["sample"]["expected_pairs"]) != 9:
        raise RuntimeError("CV3.1B1 pair count changed")
    if cfg["model"]["dtype"] != "float32" or cfg["model"]["device"] != "cuda":
        raise RuntimeError("CV3.1B1 model contract changed")
    if cfg["matching"]["method"] != "extract_correspondences_nonsym":
        raise RuntimeError("CV3.1B1 matching method changed")
    for key, value in cfg["scientific_boundary"].items():
        if value is not False:
            raise RuntimeError(f"{key} must remain false")


def verify_bundle(cfg: dict[str, Any]) -> tuple[Path, pd.DataFrame, dict[str, Any]]:
    root = resolve(cfg["bundle"]["root"])
    manifest_path = root / cfg["bundle"]["manifest_filename"]
    report_path = root / cfg["bundle"]["report_filename"]
    if not manifest_path.is_file() or not report_path.is_file():
        raise RuntimeError("CV3.1B1 smoke bundle is missing")
    frame = pd.read_csv(manifest_path)
    report = json.loads(report_path.read_text())
    if report.get("status") != "PASS_CV3_1B1_REFERENCE_FREE_SMOKE_BUNDLE":
        raise RuntimeError("Smoke bundle report is not PASS")
    if report["provenance"]["top20_sha256"] != cfg["lineage"]["expected_top20_sha256"]:
        raise RuntimeError("Smoke bundle Top20 lineage changed")
    if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != report["provenance"]["bundle_manifest_sha256"]:
        raise RuntimeError("Smoke bundle manifest hash mismatch")
    if len(frame) != int(cfg["sample"]["expected_pairs"]):
        raise RuntimeError("Smoke bundle pair count changed")
    expected_ids = set(int(x) for x in cfg["sample"]["query_ids"])
    expected_ranks = set(int(x) for x in cfg["sample"]["pool_ranks"])
    if set(pd.to_numeric(frame["query_id"]).astype(int)) != expected_ids:
        raise RuntimeError("Smoke bundle query IDs changed")
    if set(pd.to_numeric(frame["pool_rank"]).astype(int)) != expected_ranks:
        raise RuntimeError("Smoke bundle pool ranks changed")
    for row in frame.itertuples(index=False):
        q = root / str(row.query_relpath)
        t = root / str(row.tile_relpath)
        if not q.is_file() or not t.is_file():
            raise RuntimeError("Bundled smoke image missing")
        if sha256_file(q) != str(row.query_sha256):
            raise RuntimeError(f"Bundled query hash mismatch: {q}")
        if sha256_file(t) != str(row.tile_sha256):
            raise RuntimeError(f"Bundled tile hash mismatch: {t}")
    return root, frame, report


def main() -> None:
    parser = argparse.ArgumentParser(description="CV3.1B1 9-pair aerial-MASt3R smoke")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()

    cfg_path = resolve(args.config)
    cfg = yaml.safe_load(cfg_path.read_text())
    verify_config(cfg)

    b0_path = resolve(cfg["cv3_1b0"]["report"])
    b0 = json.loads(b0_path.read_text())
    if b0.get("status") != cfg["cv3_1b0"]["expected_status"]:
        raise RuntimeError("CV3.1B0 is not PASS")
    if b0["provenance"]["model_safetensors_sha256"] != cfg["checkpoint"]["expected_model_sha256"]:
        raise RuntimeError("CV3.1B0 checkpoint SHA changed")

    source_root = resolve(cfg["upstream"]["source_root"])
    if git_value(source_root, "rev-parse", "HEAD") != cfg["upstream"]["expected_source_commit"]:
        raise RuntimeError("AerialMegaDepth source commit changed")
    if git_value(source_root, "status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("AerialMegaDepth tracked source modified")

    bundle_root, bundle_frame, bundle_report = verify_bundle(cfg)

    snapshot_dir = Path(snapshot_download(
        repo_id=cfg["checkpoint"]["repo_id"],
        revision=cfg["checkpoint"]["revision"],
        allow_patterns=[str(x) for x in cfg["checkpoint"]["files"]],
        local_files_only=True,
    )).expanduser()
    model_path = snapshot_dir / cfg["checkpoint"]["model_filename"]
    if sha256_file(model_path) != cfg["checkpoint"]["expected_model_sha256"]:
        raise RuntimeError("Cached aerial-MASt3R model SHA mismatch")

    outroot = resolve(cfg["output_root"])
    outroot.mkdir(parents=True, exist_ok=True)
    worker_result = outroot / "cv3_1b1_worker_result.json"
    trace_path = outroot / "cv3_1b1_worker_trace.jsonl"
    for path in (worker_result, trace_path):
        if path.exists():
            path.unlink()

    worker = ROOT / "scripts/villoc/geometry/cv3_1b1_aerial_mast3r_smoke_worker.py"
    cmd = [
        sys.executable, str(worker),
        "--config", str(cfg_path),
        "--bundle-root", str(bundle_root),
        "--snapshot-dir", str(snapshot_dir),
        "--output", str(worker_result),
        "--trace", str(trace_path),
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
        timeout=int(cfg["worker"]["timeout_seconds"]),
        check=False,
    )
    elapsed = float(time.perf_counter() - started)

    worker_payload = json.loads(worker_result.read_text()) if worker_result.is_file() else None
    blockers: list[str] = []
    if proc.returncode != 0:
        blockers.append(f"smoke_worker_returncode:{proc.returncode}")
    if worker_payload is None:
        blockers.append("smoke_worker_result_missing")
    elif worker_payload.get("status") != "PASS_WORKER_CV3_1B1_NINE_PAIR_SMOKE":
        blockers.append("smoke_worker_not_pass")

    status = (
        "PASS_CV3_1B1_REFERENCE_FREE_AERIAL_MAST3R_SMOKE"
        if not blockers
        else "BLOCKED_CV3_1B1_REFERENCE_FREE_AERIAL_MAST3R_SMOKE"
    )

    evidence_bytes = None
    evidence_sha = None
    summary = None
    if worker_payload is not None:
        frame = pd.DataFrame(worker_payload["rows"]).sort_values(
            ["query_id","pool_rank"], kind="mergesort"
        ).reset_index(drop=True)
        if len(frame) == int(cfg["sample"]["expected_pairs"]):
            runtime_columns = [
                c for c in frame.columns
                if c.endswith("_ms") or "_gib_" in c or c.startswith("torch_peak_")
            ]
            frozen = frame.drop(columns=runtime_columns)
            evidence_bytes = canonical_csv(frozen)
            evidence_sha = hashlib.sha256(evidence_bytes).hexdigest()
            summary = {
                "pairs": int(len(frame)),
                "homography_ok": int(frame["homography_ok"].astype(bool).sum()),
                "finite_projection": int(frame["finite_projection"].astype(bool).sum()),
                "inside_projection": int(frame["projected_inside_tile"].astype(bool).sum()),
                "median_raw_correspondences": float(pd.to_numeric(frame["raw_correspondences"]).median()),
                "median_filtered_correspondences": float(pd.to_numeric(frame["filtered_correspondences"]).median()),
                "median_inliers": float(pd.to_numeric(frame["inliers"]).median()),
                "median_inference_ms_not_frozen": float(pd.to_numeric(frame["inference_ms"]).median()),
                "median_matching_ms_not_frozen": float(pd.to_numeric(frame["matching_ms"]).median()),
                "median_total_ms_not_frozen": float(pd.to_numeric(frame["total_ms"]).median()),
                "max_torch_peak_allocated_gib_not_frozen": float(pd.to_numeric(frame["torch_peak_allocated_gib_pair"]).max()),
                "min_cuda_free_gib_after_pair_before_cleanup_not_frozen": float(pd.to_numeric(frame["cuda_free_gib_after_pair_before_cleanup"]).min()),
                "min_system_available_gib_after_pair_before_cleanup_not_frozen": float(pd.to_numeric(frame["system_available_gib_after_pair_before_cleanup"]).min()),
            }

    report = {
        "stage": "CV3.1B1",
        "status": status,
        "blockers": blockers,
        "sample": {
            "query_ids": [int(x) for x in cfg["sample"]["query_ids"]],
            "pool_ranks": [int(x) for x in cfg["sample"]["pool_ranks"]],
            "pairs": int(cfg["sample"]["expected_pairs"]),
        },
        "summary": summary,
        "worker": {
            "returncode": int(proc.returncode),
            "elapsed_seconds": elapsed,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "trace_path": str(trace_path),
        },
        "scientific_boundary": {
            "reference_loaded": False,
            "oracle_labels_loaded": False,
            "aerialmegadepth_dataset_downloaded": False,
            "threshold_tuned": False,
            "candidate_reranked": False,
            "uniform40_executed": False,
            "full403_executed": False,
            "accuracy_claim": False,
        },
        "provenance": {
            "config_sha256": sha256_file(cfg_path),
            "cv3_1b0_report_sha256": sha256_file(b0_path),
            "bundle_report_sha256": sha256_file(bundle_root / cfg["bundle"]["report_filename"]),
            "bundle_manifest_sha256": hashlib.sha256((bundle_root / cfg["bundle"]["manifest_filename"]).read_bytes()).hexdigest(),
            "top20_sha256": bundle_report["provenance"]["top20_sha256"],
            "aerialmegadepth_source_commit": git_value(source_root, "rev-parse", "HEAD"),
            "checkpoint_revision": cfg["checkpoint"]["revision"],
            "model_safetensors_sha256": sha256_file(model_path),
            "evidence_sha256": evidence_sha,
        },
    }

    files = {outroot / "cv3_1b1_smoke_report.json": canonical_json(report)}
    if evidence_bytes is not None and status.startswith("PASS_"):
        files[outroot / "cv3_1b1_reference_free_evidence.csv"] = evidence_bytes
    freeze(files)

    print("Status:", status)
    print("Blockers:", ",".join(blockers) if blockers else "NONE")
    if summary is not None:
        print("Pairs:", summary["pairs"], "/ 9")
        print("Homography ok:", summary["homography_ok"], "/ 9")
        print("Finite projection:", summary["finite_projection"], "/ 9")
        print("Median raw/filtered correspondences:", f"{summary['median_raw_correspondences']:.1f}/{summary['median_filtered_correspondences']:.1f}")
        print("Median inliers:", f"{summary['median_inliers']:.1f}")
        print("Median inference ms (not frozen):", f"{summary['median_inference_ms_not_frozen']:.3f}")
        print("Median matching ms (not frozen):", f"{summary['median_matching_ms_not_frozen']:.3f}")
        print("Median total ms (not frozen):", f"{summary['median_total_ms_not_frozen']:.3f}")
        print("Max Torch peak allocated GiB (not frozen):", f"{summary['max_torch_peak_allocated_gib_not_frozen']:.3f}")
        print("Min CUDA free GiB after pair (not frozen):", f"{summary['min_cuda_free_gib_after_pair_before_cleanup_not_frozen']:.3f}")
        print("Min system available GiB after pair (not frozen):", f"{summary['min_system_available_gib_after_pair_before_cleanup_not_frozen']:.3f}")
    print("Evidence SHA256:", evidence_sha or "NONE")
    print("Reference loaded: False")
    print("Accuracy claim: False")
    print("Worker stdout:")
    print(proc.stdout.rstrip() or "<empty>")
    if proc.stderr:
        print("Worker stderr:")
        print(proc.stderr.rstrip())
    print("Report:", outroot / "cv3_1b1_smoke_report.json")


if __name__ == "__main__":
    main()
