#!/usr/bin/env python3
"""CV1.3: freeze full403 reference-free ORB evidence for promoted scale_m2."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
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

from uavloc.data.trajectory_adapter import FORBIDDEN_BLIND_REFERENCE_COLUMNS
from uavloc.geometry import orb_subtile
from uavloc.geometry.crossview_query_transforms import (
    feature_pack_from_gray,
    read_frozen_processed_query,
    scale_processed_query,
)
from uavloc.geometry.top20_geometric_evidence import analyze_pair

ROOT = REPO_ROOT
DEFAULT_CONFIG = ROOT / "configs/research/crossview_scale_rotation_domain_cv1_3.yaml"

MANIFEST_REQUIRED = {"trajectory_id", "query_id", "image_path", "reference_available"}
POOL_REQUIRED = {
    "trajectory_id", "query_id", "pool_rank", "tile_id",
    "source_view", "source_rank",
}
TILE_REQUIRED = {
    "tile_id", "tile_path", "left_easting", "right_easting",
    "bottom_northing", "top_northing",
}
FORBIDDEN_PARTIAL = (
    "ground_truth", "oracle", "eval_ref_", "reference_x", "reference_y",
    "chosen_error", "projected_error", "tile_center_error", "gt_error",
)


def resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_csv(rows: list[dict[str, Any]]) -> bytes:
    if not rows:
        raise RuntimeError("Cannot freeze empty CV1.3 evidence")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    for row in rows:
        cooked: dict[str, Any] = {}
        for key, value in row.items():
            if isinstance(value, bool):
                cooked[key] = "true" if value else "false"
            elif isinstance(value, float):
                cooked[key] = "" if not math.isfinite(value) else f"{value:.9f}"
            elif value is None:
                cooked[key] = ""
            else:
                cooked[key] = value
        writer.writerow(cooked)
    return stream.getvalue().encode("utf-8")


def canonical_json(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def freeze_files(files: dict[Path, bytes]) -> None:
    changed = [str(p) for p, data in files.items() if p.exists() and p.read_bytes() != data]
    if changed:
        raise RuntimeError("Refusing to overwrite differing CV1.3 freeze: " + ", ".join(changed))
    for path, data in files.items():
        if path.exists():
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as handle:
            handle.write(data)


def assert_blind_safe_columns(frame: pd.DataFrame, label: str) -> None:
    lower = {str(c).strip().lower() for c in frame.columns}
    exact = {
        str(c).strip().lower()
        for c in FORBIDDEN_BLIND_REFERENCE_COLUMNS
        if str(c).strip().lower() != "reference_available"
    }
    leaked = sorted(
        (lower & exact)
        | {
            c for c in lower
            if c != "reference_available"
            and any(token in c for token in FORBIDDEN_PARTIAL)
        }
    )
    if leaked:
        raise RuntimeError(f"{label} contains forbidden reference/eval columns: {leaked}")


def require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"{label} missing columns: {missing}")


def resolve_existing_file(value: Any, *, extra_roots: list[Path] | None = None) -> Path:
    raw = str(value).strip()
    if not raw or raw.lower() in {"nan", "none"}:
        raise FileNotFoundError(f"Invalid file path: {value!r}")
    path = Path(raw).expanduser()
    candidates = [path] if path.is_absolute() else [ROOT / path]
    if not path.is_absolute():
        candidates.extend(root / path for root in (extra_roots or []))
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError("Could not resolve file; tried: " + ", ".join(map(str, candidates)))


def verify_config(cfg: dict[str, Any]) -> None:
    if cfg.get("stage") != "CV1.3":
        raise RuntimeError("Expected CV1.3 config")
    p = cfg["promotion"]
    if p.get("promoted_condition") != "scale_m2":
        raise RuntimeError("CV1.3 promoted condition changed")
    if not math.isclose(float(p.get("scale_factor")), 0.6944444444444444, rel_tol=0.0, abs_tol=1e-15):
        raise RuntimeError("CV1.3 scale factor changed")
    if p.get("no_retuning_after_uniform40") is not True:
        raise RuntimeError("CV1.3 no-retuning contract missing")
    if cfg["lineage"]["frozen_retrieval"] != "qv1_4_center_unique_allview_fill20":
        raise RuntimeError("Frozen retrieval changed")
    for key, value in cfg["scientific_boundary"].items():
        if value is not False:
            raise RuntimeError(f"Scientific boundary {key} must remain false in CV1.3")


def load_inputs(cfg: dict[str, Any]) -> dict[str, Any]:
    tc = cfg["trajectory"]
    spec_path = resolve(tc["spec"])
    manifest_path = resolve(tc["manifest"])
    pool_path = resolve(tc["frozen_top20"])
    baseline_path = resolve(cfg["lineage"]["baseline_gd1_0_evidence"])
    for path in (spec_path, manifest_path, pool_path, baseline_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    if sha256_file(pool_path) != cfg["lineage"]["expected_top20_sha256"]:
        raise RuntimeError("Frozen QV1.4 development Top20 SHA256 changed")
    if sha256_file(baseline_path) != cfg["lineage"]["expected_baseline_gd1_0_sha256"]:
        raise RuntimeError("Frozen GD1.0 baseline evidence SHA256 changed")

    spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    if spec["trajectory"]["id"] != "villoc_traj01_90deg_stable120m":
        raise RuntimeError("Development trajectory changed")
    if spec["map"]["crs"] != "EPSG:3346" or "768_s256" not in spec["map"]["variants"]:
        raise RuntimeError("Trajectory map contract changed")

    tile_index_path = resolve(spec["map"]["variants"]["768_s256"]["tile_index"])
    manifest = pd.read_csv(manifest_path).copy()
    pool = pd.read_csv(pool_path).copy()
    tile = pd.read_csv(tile_index_path).copy()

    for frame, label in (
        (manifest, "CV1.3 development manifest"),
        (pool, "CV1.3 frozen Top20"),
        (tile, "CV1.3 tile index"),
    ):
        assert_blind_safe_columns(frame, label)

    require_columns(manifest, MANIFEST_REQUIRED, "CV1.3 development manifest")
    require_columns(pool, POOL_REQUIRED, "CV1.3 frozen Top20")
    require_columns(tile, TILE_REQUIRED, "CV1.3 tile index")

    manifest["query_id"] = pd.to_numeric(manifest["query_id"], errors="raise").astype(int)
    pool["query_id"] = pd.to_numeric(pool["query_id"], errors="raise").astype(int)
    pool["pool_rank"] = pd.to_numeric(pool["pool_rank"], errors="raise").astype(int)
    pool["source_rank"] = pd.to_numeric(pool["source_rank"], errors="raise").astype(int)
    pool["tile_id"] = pool["tile_id"].astype(str)
    tile["tile_id"] = tile["tile_id"].astype(str)

    refs = manifest["reference_available"].astype(str).str.strip().str.lower()
    if not refs.isin(["false", "0", "no", "n", "f"]).all():
        raise RuntimeError("CV1.3 manifest exposes reference")
    if len(manifest) != int(tc["expected_queries"]) or manifest["query_id"].duplicated().any():
        raise RuntimeError("CV1.3 development query contract changed")
    if len(pool) != int(tc["expected_pairs"]):
        raise RuntimeError("CV1.3 frozen Top20 pair count changed")
    if pool[["query_id", "tile_id"]].duplicated().any():
        raise RuntimeError("CV1.3 duplicate query/tile candidate")
    if set(pool["query_id"]) != set(manifest["query_id"]):
        raise RuntimeError("CV1.3 manifest/Top20 query sets differ")

    for qid, group in pool.groupby("query_id", sort=True):
        ordered = group.sort_values("pool_rank")
        if len(ordered) != 20 or ordered["pool_rank"].tolist() != list(range(1, 21)):
            raise RuntimeError(f"CV1.3 q{qid} Top20 rank contract changed")
        top1 = ordered.iloc[0]
        if str(top1["source_view"]) != "center_square" or int(top1["source_rank"]) != 1:
            raise RuntimeError(f"CV1.3 q{qid} center Top1 contract changed")

    return {
        "spec_path": spec_path,
        "manifest_path": manifest_path,
        "pool_path": pool_path,
        "baseline_path": baseline_path,
        "tile_index_path": tile_index_path,
        "manifest": manifest,
        "pool": pool.sort_values(["query_id", "pool_rank"], kind="mergesort").reset_index(drop=True),
        "tile": tile,
        "trajectory_id": spec["trajectory"]["id"],
    }


def preflight_paths(inputs: dict[str, Any]) -> tuple[dict[int, Path], dict[str, Path]]:
    q_paths: dict[int, Path] = {}
    for row in inputs["manifest"].itertuples(index=False):
        q_paths[int(row.query_id)] = resolve_existing_file(row.image_path)

    tile_lookup = inputs["tile"].set_index("tile_id")
    t_paths: dict[str, Path] = {}
    for tile_id in sorted(set(inputs["pool"]["tile_id"])):
        t_paths[tile_id] = resolve_existing_file(
            tile_lookup.loc[tile_id, "tile_path"],
            extra_roots=[inputs["tile_index_path"].parent],
        )
    return q_paths, t_paths


def append_projection(
    out: dict[str, Any],
    evidence: dict[str, Any],
    q,
    t,
    tile_row: pd.Series,
) -> None:
    projected_e = math.nan
    projected_n = math.nan
    finite_projection = False
    inside = False
    if bool(evidence["homography_ok"]):
        u = float(evidence["projected_tile_u_px"])
        v = float(evidence["projected_tile_v_px"])
        projected_e, projected_n = orb_subtile.tile_pixel_to_map(
            u, v, t.image_shape,
            float(tile_row["left_easting"]), float(tile_row["right_easting"]),
            float(tile_row["bottom_northing"]), float(tile_row["top_northing"]),
        )
        finite_projection = bool(math.isfinite(projected_e) and math.isfinite(projected_n))
        th, tw = t.image_shape
        inside = bool(
            finite_projection
            and 0.0 <= u < float(tw)
            and 0.0 <= v < float(th)
        )

    out.update({
        "query_processed_height_px": int(q.image_shape[0]),
        "query_processed_width_px": int(q.image_shape[1]),
        "tile_processed_height_px": int(t.image_shape[0]),
        "tile_processed_width_px": int(t.image_shape[1]),
        "projected_easting": float(projected_e),
        "projected_northing": float(projected_n),
        "finite_projection": finite_projection,
        "projected_inside_tile": inside,
    })


def generate(
    cfg: dict[str, Any],
    inputs: dict[str, Any],
    q_paths: dict[int, Path],
    t_paths: dict[str, Path],
) -> tuple[list[dict[str, Any]], dict[str, Any], float]:
    detector = orb_subtile.create_detector()
    tile_lookup = inputs["tile"].set_index("tile_id")
    q_cache: dict[int, Any] = {}
    t_cache: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    scale_factor = float(cfg["promotion"]["scale_factor"])
    progress_every = int(cfg["generation"]["progress_every_pairs"])

    def qfeat(qid: int):
        if qid not in q_cache:
            gray = read_frozen_processed_query(q_paths[qid])
            gray = scale_processed_query(gray, scale_factor)
            q_cache[qid] = feature_pack_from_gray(gray, detector)
        return q_cache[qid]

    def tfeat(tile_id: str):
        if tile_id not in t_cache:
            t_cache[tile_id] = orb_subtile.compute_features(t_paths[tile_id], detector)
        return t_cache[tile_id]

    started = time.perf_counter()
    pool = inputs["pool"]
    for idx, row in enumerate(pool.itertuples(index=False), start=1):
        if idx == 1 or idx % max(1, progress_every) == 0 or idx == len(pool):
            print(f"CV1.3 full403 scale_m2 pair {idx}/{len(pool)}", flush=True)

        qid = int(row.query_id)
        tile_id = str(row.tile_id)
        q = qfeat(qid)
        t = tfeat(tile_id)
        evidence = analyze_pair(q, t)

        out = {
            "trajectory_id": inputs["trajectory_id"],
            "role": "development",
            "query_id": qid,
            "pool_rank": int(row.pool_rank),
            "tile_id": tile_id,
            "source_view": str(row.source_view),
            "source_rank": int(row.source_rank),
            "transform_family": "scale",
            "transform_id": "scale_m2",
            "scale_factor": scale_factor,
            "rotation_deg": 0,
        }
        out.update(evidence)
        append_projection(out, evidence, q, t, tile_lookup.loc[tile_id])
        rows.append(out)

    runtime = float(time.perf_counter() - started)
    frame = pd.DataFrame(rows)
    stats = {
        "queries": int(frame["query_id"].nunique()),
        "candidate_pairs": int(len(frame)),
        "unique_tiles": int(frame["tile_id"].nunique()),
        "homography_ok_pairs": int(frame["homography_ok"].astype(bool).sum()),
        "finite_projection_pairs": int(frame["finite_projection"].astype(bool).sum()),
        "inside_projection_pairs": int(frame["projected_inside_tile"].astype(bool).sum()),
    }
    return rows, stats, runtime


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CV1.3 full403 reference-free promoted scale_m2 evidence freeze"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()

    cfg_path = resolve(args.config)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    verify_config(cfg)
    inputs = load_inputs(cfg)
    q_paths, t_paths = preflight_paths(inputs)

    print(
        f"CV1.3 preflight: queries={len(inputs['manifest'])} "
        f"pairs={len(inputs['pool'])} unique_tiles={len(t_paths)} "
        f"condition=scale_m2 factor={cfg['promotion']['scale_factor']}"
    )
    print("Reference loaded: False")
    print("SRT loaded: False")
    print("Baseline regenerated: False")

    rows, stats, runtime = generate(cfg, inputs, q_paths, t_paths)
    if len(rows) != int(cfg["trajectory"]["expected_pairs"]):
        raise RuntimeError("CV1.3 generated row count changed")

    evidence_path = resolve(cfg["output_root"]) / "cv1_3_full403_scale_m2_reference_free_evidence.csv"
    evidence_payload = canonical_csv(rows)

    report = {
        "stage": "CV1.3",
        "status": "PASS_CV1_3_REFERENCE_FREE_FULL403_SCALE_M2_FREEZE",
        "promoted_condition": cfg["promotion"],
        "stats": stats,
        "scientific_boundary": {
            "reference_loaded": False,
            "srt_loaded": False,
            "oracle_labels_loaded": False,
            "candidate_selected": False,
            "reranking_executed": False,
            "threshold_tuned": False,
            "transform_retuned": False,
            "baseline_regenerated": False,
            "state_modified": False,
            "temporal_fusion_executed": False,
        },
        "interpretation": {
            "measurement_stage_only": True,
            "no_accuracy_claim": True,
            "physical_gsd_not_claimed": True,
            "baseline_is_existing_frozen_gd1_0_full403": True,
        },
        "provenance": {
            "config_sha256": sha256_file(cfg_path),
            "protocol_config_sha256": sha256_file(resolve(cfg["lineage"]["protocol_config"])),
            "trajectory_spec_sha256": sha256_file(inputs["spec_path"]),
            "blind_manifest_sha256": sha256_file(inputs["manifest_path"]),
            "frozen_top20_sha256": sha256_file(inputs["pool_path"]),
            "baseline_gd1_0_sha256": sha256_file(inputs["baseline_path"]),
            "map_index_sha256": sha256_file(inputs["tile_index_path"]),
            "baseline_geometry_module_sha256": sha256_file(Path(orb_subtile.__file__).resolve()),
            "transform_module_sha256": sha256_file(
                ROOT / "src/uavloc/geometry/crossview_query_transforms.py"
            ),
            "evidence_module_sha256": sha256_file(
                ROOT / "src/uavloc/geometry/top20_geometric_evidence.py"
            ),
            "evidence_output_sha256": sha256_bytes(evidence_payload),
        },
    }

    report_path = resolve(cfg["output_root"]) / "cv1_3_freeze_report.json"
    freeze_files({
        evidence_path: evidence_payload,
        report_path: canonical_json(report),
    })

    print("Status:", report["status"])
    print("Evidence SHA256:", report["provenance"]["evidence_output_sha256"])
    print(f"Observed runtime s (not frozen): {runtime:.3f}")
    print("Reference loaded: False")
    print("SRT loaded: False")
    print("Baseline regenerated: False")
    print("Candidate selected/reranked: False")
    print("Accuracy claim: False")
    print("Freeze report:", report_path)


if __name__ == "__main__":
    main()
