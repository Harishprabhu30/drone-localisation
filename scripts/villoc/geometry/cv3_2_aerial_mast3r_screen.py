#!/usr/bin/env python3
"""Run and freeze CV3.2 uniform40 x six-candidate aerial-MASt3R evidence."""
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
DEFAULT_CONFIG = ROOT / "configs/research/crossview_scale_rotation_domain_cv3_2.yaml"


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
        raise RuntimeError("Refusing to overwrite differing CV3.2 outputs: " + ", ".join(bad))
    for p, d in files.items():
        if not p.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
            with p.open("xb") as handle:
                handle.write(d)


def verify_config(cfg: dict[str, Any]) -> None:
    if cfg.get("stage") != "CV3.2":
        raise RuntimeError("Expected CV3.2 config")
    u = cfg["uniform40"]
    if int(u["expected_queries"]) != 40 or int(u["expected_pairs"]) != 240:
        raise RuntimeError("CV3.2 size contract changed")
    if [int(x) for x in u["pool_ranks"]] != [1, 2, 3, 5, 10, 20]:
        raise RuntimeError("CV3.2 rank slice changed")
    if cfg["model"]["dtype"] != "float32" or cfg["model"]["device"] != "cuda":
        raise RuntimeError("CV3.2 model contract changed")
    if cfg["selector"]["primary"] != "homography_inliers_desc":
        raise RuntimeError("CV3.2 selector changed")
    if cfg["selector"]["tiebreak"] != ["pool_rank_asc", "tile_id_asc"]:
        raise RuntimeError("CV3.2 selector tiebreak changed")
    for key, value in cfg["scientific_boundary"].items():
        if value is not False:
            raise RuntimeError(f"{key} must remain false")


def verify_bundle(cfg: dict[str, Any]) -> tuple[Path, pd.DataFrame, dict[str, Any]]:
    root = resolve(cfg["bundle"]["root"])
    manifest_path = root / cfg["bundle"]["manifest_filename"]
    report_path = root / cfg["bundle"]["report_filename"]
    if not manifest_path.is_file() or not report_path.is_file():
        raise RuntimeError("CV3.2 compact bundle is missing")
    frame = pd.read_csv(manifest_path)
    report = json.loads(report_path.read_text())
    if report.get("status") != "PASS_CV3_2_REFERENCE_FREE_COMPACT_BUNDLE":
        raise RuntimeError("CV3.2 compact bundle report is not PASS")
    if report["provenance"]["top20_sha256"] != cfg["lineage"]["expected_top20_sha256"]:
        raise RuntimeError("CV3.2 compact bundle Top20 lineage changed")
    if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != report["provenance"]["bundle_manifest_sha256"]:
        raise RuntimeError("CV3.2 compact bundle manifest hash mismatch")
    if len(frame) != int(cfg["uniform40"]["expected_pairs"]):
        raise RuntimeError("CV3.2 compact bundle pair count changed")
    if frame["query_id"].nunique() != int(cfg["uniform40"]["expected_queries"]):
        raise RuntimeError("CV3.2 compact bundle query count changed")
    for row in frame.itertuples(index=False):
        q = root / str(row.query_relpath)
        t = root / str(row.tile_relpath)
        if not q.is_file() or not t.is_file():
            raise RuntimeError("CV3.2 bundled image missing")
        if sha256_file(q) != str(row.query_sha256):
            raise RuntimeError(f"CV3.2 bundled query hash mismatch: {q}")
        if sha256_file(t) != str(row.tile_sha256):
            raise RuntimeError(f"CV3.2 bundled tile hash mismatch: {t}")
    return root, frame, report


def freeze_reference_free_selection(frame: pd.DataFrame) -> pd.DataFrame:
    ranked = frame.copy()
    ranked["inliers"] = pd.to_numeric(ranked["inliers"], errors="raise").astype(int)
    ranked["pool_rank"] = pd.to_numeric(ranked["pool_rank"], errors="raise").astype(int)
    ranked["tile_id"] = ranked["tile_id"].astype(str)
    ranked = ranked.sort_values(
        ["query_id", "inliers", "pool_rank", "tile_id"],
        ascending=[True, False, True, True],
        kind="mergesort",
    )

    rows = []
    for qid, group in ranked.groupby("query_id", sort=True):
        ordered = group.reset_index(drop=True)
        winner = ordered.iloc[0]
        runner_up = ordered.iloc[1]
        rows.append({
            "query_id": int(qid),
            "selected_pool_rank": int(winner["pool_rank"]),
            "selected_tile_id": str(winner["tile_id"]),
            "selected_inliers": int(winner["inliers"]),
            "runner_up_inliers": int(runner_up["inliers"]),
            "inlier_margin": int(winner["inliers"] - runner_up["inliers"]),
            "selected_homography_ok": bool(winner["homography_ok"]),
            "selected_finite_projection": bool(winner["finite_projection"]),
            "selected_projected_inside_tile": bool(winner["projected_inside_tile"]),
            "selected_projected_easting": winner["projected_easting"],
            "selected_projected_northing": winner["projected_northing"],
        })
    return pd.DataFrame(rows).sort_values("query_id", kind="mergesort").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="CV3.2 compact reference-free aerial-MASt3R screen")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()

    cfg_path = resolve(args.config)
    cfg = yaml.safe_load(cfg_path.read_text())
    verify_config(cfg)

    b1_report_path = resolve(cfg["cv3_1b1"]["report"])
    b1_evidence_path = resolve(cfg["cv3_1b1"]["evidence"])
    b1 = json.loads(b1_report_path.read_text())
    if b1.get("status") != cfg["cv3_1b1"]["expected_status"]:
        raise RuntimeError("CV3.1B1 is not PASS")
    if sha256_file(b1_evidence_path) != cfg["cv3_1b1"]["expected_evidence_sha256"]:
        raise RuntimeError("CV3.1B1 frozen evidence bytes changed")
    if b1["provenance"]["evidence_sha256"] != cfg["cv3_1b1"]["expected_evidence_sha256"]:
        raise RuntimeError("CV3.1B1 report evidence hash changed")
    if b1["scientific_boundary"]["reference_loaded"] is not False:
        raise RuntimeError("CV3.1B1 reference boundary violated")

    source_root = resolve(cfg["upstream"]["source_root"])
    if git_value(source_root, "rev-parse", "HEAD") != cfg["upstream"]["expected_source_commit"]:
        raise RuntimeError("AerialMegaDepth source commit changed")
    if git_value(source_root, "status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("AerialMegaDepth tracked source modified")

    bundle_root, _, bundle_report = verify_bundle(cfg)

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
    worker_result = outroot / "cv3_2_worker_result.json"
    trace_path = outroot / "cv3_2_worker_trace.jsonl"
    for path in (worker_result, trace_path):
        if path.exists():
            path.unlink()

    worker = ROOT / "scripts/villoc/geometry/cv3_2_aerial_mast3r_worker.py"
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
        blockers.append(f"compact_worker_returncode:{proc.returncode}")
    if worker_payload is None:
        blockers.append("compact_worker_result_missing")
    elif worker_payload.get("status") != "PASS_WORKER_CV3_2_REFERENCE_FREE_COMPACT_SCREEN":
        blockers.append("compact_worker_not_pass")

    status = (
        "PASS_CV3_2_REFERENCE_FREE_COMPACT_FREEZE"
        if not blockers
        else "BLOCKED_CV3_2_REFERENCE_FREE_COMPACT_FREEZE"
    )

    pair_bytes = selection_bytes = None
    pair_sha = selection_sha = None
    summary = None
    if worker_payload is not None:
        frame = pd.DataFrame(worker_payload["rows"]).sort_values(
            ["query_id", "pool_rank"], kind="mergesort"
        ).reset_index(drop=True)
        if len(frame) == int(cfg["uniform40"]["expected_pairs"]):
            runtime_columns = [
                c for c in frame.columns
                if c.endswith("_ms") or "_gib_" in c or c.startswith("torch_peak_")
            ]
            frozen_pairs = frame.drop(columns=runtime_columns)
            pair_bytes = canonical_csv(frozen_pairs)
            pair_sha = hashlib.sha256(pair_bytes).hexdigest()

            selected = freeze_reference_free_selection(frozen_pairs)
            selection_bytes = canonical_csv(selected)
            selection_sha = hashlib.sha256(selection_bytes).hexdigest()

            homography = frame["homography_ok"].astype(bool)
            per_query_h = homography.groupby(frame["query_id"]).sum()
            summary = {
                "queries": int(frame["query_id"].nunique()),
                "pairs": int(len(frame)),
                "homography_ok_pairs": int(homography.sum()),
                "finite_projection_pairs": int(frame["finite_projection"].astype(bool).sum()),
                "queries_any_homography": int((per_query_h > 0).sum()),
                "queries_all6_homographies": int((per_query_h == 6).sum()),
                "queries_no_homography": int((per_query_h == 0).sum()),
                "selected_homography_ok": int(selected["selected_homography_ok"].astype(bool).sum()),
                "selected_finite_projection": int(selected["selected_finite_projection"].astype(bool).sum()),
                "median_raw_correspondences": float(pd.to_numeric(frame["raw_correspondences"]).median()),
                "median_filtered_correspondences": float(pd.to_numeric(frame["filtered_correspondences"]).median()),
                "median_inliers": float(pd.to_numeric(frame["inliers"]).median()),
                "median_selected_inliers": float(pd.to_numeric(selected["selected_inliers"]).median()),
                "median_inlier_margin": float(pd.to_numeric(selected["inlier_margin"]).median()),
                "median_inference_ms_not_frozen": float(pd.to_numeric(frame["inference_ms"]).median()),
                "median_matching_ms_not_frozen": float(pd.to_numeric(frame["matching_ms"]).median()),
                "median_total_ms_not_frozen": float(pd.to_numeric(frame["total_ms"]).median()),
                "max_torch_peak_allocated_gib_not_frozen": float(pd.to_numeric(frame["torch_peak_allocated_gib_pair"]).max()),
                "min_cuda_free_gib_after_pair_before_cleanup_not_frozen": float(pd.to_numeric(frame["cuda_free_gib_after_pair_before_cleanup"]).min()),
                "min_system_available_gib_after_pair_before_cleanup_not_frozen": float(pd.to_numeric(frame["system_available_gib_after_pair_before_cleanup"]).min()),
            }

    report = {
        "stage": "CV3.2",
        "status": status,
        "blockers": blockers,
        "sample": {
            "query_ids": [int(x) for x in cfg["uniform40"]["query_ids"]],
            "pool_ranks": [int(x) for x in cfg["uniform40"]["pool_ranks"]],
            "queries": int(cfg["uniform40"]["expected_queries"]),
            "pairs": int(cfg["uniform40"]["expected_pairs"]),
        },
        "selector": cfg["selector"],
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
            "learned_score_fusion_executed": False,
            "selector_frozen_reference_free": bool(status.startswith("PASS_")),
            "full_top20_executed": False,
            "full403_executed": False,
            "accuracy_claim": False,
        },
        "provenance": {
            "config_sha256": sha256_file(cfg_path),
            "cv3_1b1_report_sha256": sha256_file(b1_report_path),
            "cv3_1b1_evidence_sha256": sha256_file(b1_evidence_path),
            "bundle_report_sha256": sha256_file(bundle_root / cfg["bundle"]["report_filename"]),
            "bundle_manifest_sha256": hashlib.sha256((bundle_root / cfg["bundle"]["manifest_filename"]).read_bytes()).hexdigest(),
            "top20_sha256": bundle_report["provenance"]["top20_sha256"],
            "aerialmegadepth_source_commit": git_value(source_root, "rev-parse", "HEAD"),
            "checkpoint_revision": cfg["checkpoint"]["revision"],
            "model_safetensors_sha256": sha256_file(model_path),
            "pair_evidence_sha256": pair_sha,
            "selection_evidence_sha256": selection_sha,
        },
    }

    files = {outroot / "cv3_2_freeze_report.json": canonical_json(report)}
    if status.startswith("PASS_") and pair_bytes is not None and selection_bytes is not None:
        files[outroot / "cv3_2_pair_reference_free_evidence.csv"] = pair_bytes
        files[outroot / "cv3_2_selection_reference_free_evidence.csv"] = selection_bytes
    freeze(files)

    print("Status:", status)
    print("Blockers:", ",".join(blockers) if blockers else "NONE")
    if summary is not None:
        print("Queries/pairs:", f"{summary['queries']}/{summary['pairs']}")
        print("Homography ok pairs:", summary["homography_ok_pairs"], "/ 240")
        print("Finite projection pairs:", summary["finite_projection_pairs"], "/ 240")
        print("Queries any/all6/no homography:", f"{summary['queries_any_homography']}/{summary['queries_all6_homographies']}/{summary['queries_no_homography']}")
        print("Selected homography/finite projection:", f"{summary['selected_homography_ok']}/{summary['selected_finite_projection']}", "/ 40")
        print("Median raw/filtered correspondences:", f"{summary['median_raw_correspondences']:.1f}/{summary['median_filtered_correspondences']:.1f}")
        print("Median pair/selected inliers:", f"{summary['median_inliers']:.1f}/{summary['median_selected_inliers']:.1f}")
        print("Median inlier margin:", f"{summary['median_inlier_margin']:.1f}")
        print("Median inference ms (not frozen):", f"{summary['median_inference_ms_not_frozen']:.3f}")
        print("Median matching ms (not frozen):", f"{summary['median_matching_ms_not_frozen']:.3f}")
        print("Median total ms (not frozen):", f"{summary['median_total_ms_not_frozen']:.3f}")
        print("Max Torch peak allocated GiB (not frozen):", f"{summary['max_torch_peak_allocated_gib_not_frozen']:.3f}")
        print("Min CUDA free GiB after pair (not frozen):", f"{summary['min_cuda_free_gib_after_pair_before_cleanup_not_frozen']:.3f}")
        print("Min system available GiB after pair (not frozen):", f"{summary['min_system_available_gib_after_pair_before_cleanup_not_frozen']:.3f}")
    print("Pair evidence SHA256:", pair_sha or "NONE")
    print("Selection evidence SHA256:", selection_sha or "NONE")
    print("Reference loaded: False")
    print("Accuracy claim: False")
    print("Worker stdout:")
    print(proc.stdout.rstrip() or "<empty>")
    if proc.stderr:
        print("Worker stderr:")
        print(proc.stderr.rstrip())
    print("Report:", outroot / "cv3_2_freeze_report.json")


if __name__ == "__main__":
    main()
