#!/usr/bin/env python3
"""GD1.0: freeze richer blind geometric evidence for every frozen QV1.4 Top20 pair."""
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
from uavloc.geometry.top20_geometric_evidence import GRID_SIZE, analyze_pair

ROOT = REPO_ROOT
DEFAULT_CONFIG = ROOT / "configs/research/top20_geometric_discrimination_v1.yaml"

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


def resolve_existing_file(
    value: Any,
    *,
    extra_roots: list[Path] | None = None,
) -> Path:
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


def canonical_csv(rows: list[dict[str, Any]]) -> bytes:
    if not rows:
        raise RuntimeError("Cannot freeze empty GD1.0 evidence")
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
        raise RuntimeError("Refusing to overwrite differing GD1.0 freeze: " + ", ".join(changed))
    for path, data in files.items():
        if path.exists():
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as handle:
            handle.write(data)


def verify_config(cfg: dict[str, Any]) -> None:
    if cfg.get("stage") != "GD1.0":
        raise RuntimeError("Expected GD1.0 config")
    if cfg.get("map_variant") != "768_s256" or cfg.get("map_crs") != "EPSG:3346":
        raise RuntimeError("Map contract changed")
    if cfg.get("lineage", {}).get("frozen_retrieval") != "qv1_4_center_unique_allview_fill20":
        raise RuntimeError("Frozen retrieval contract changed")
    if int(cfg.get("lineage", {}).get("final_budget")) != 20:
        raise RuntimeError("Top20 budget changed")

    gc = cfg["geometry_contract"]
    baseline = orb_subtile.contract()
    expected_baseline = {
        "preprocess": gc["preprocess"],
        "resize_long": gc["resize_long"],
        "orb_nfeatures": gc["orb_nfeatures"],
        "fast_threshold": gc["fast_threshold"],
        "edge_threshold": gc["edge_threshold"],
        "patch_size": gc["patch_size"],
        "lowe_ratio": gc["lowe_ratio"],
        "ransac_threshold_px": gc["ransac_threshold_px"],
        "min_homography_inliers": gc["min_homography_inliers"],
        "homography_direction": baseline["homography_direction"],
        "query_point": baseline["query_point"],
        "map_conversion": baseline["map_conversion"],
    }
    if baseline != expected_baseline:
        raise RuntimeError(
            "GD1.0 baseline ORB geometry differs from frozen RG1/RG2 contract: "
            + json.dumps({"baseline": baseline, "expected": expected_baseline}, sort_keys=True)
        )
    if int(gc.get("grid_size")) != int(GRID_SIZE):
        raise RuntimeError("Spatial support grid size changed")

    for key, value in cfg["scientific_boundary"].items():
        if value is not False:
            raise RuntimeError(f"{key} must be false in GD1.0")
    gen = cfg["generation"]
    if gen.get("process_all_top20_candidates") is not True:
        raise RuntimeError("GD1.0 must process all Top20 candidates")
    if gen.get("freeze_failed_pairs") is not True:
        raise RuntimeError("GD1.0 must retain failed geometry pairs")


def verify_map_lineage(cfg: dict[str, Any]) -> tuple[str, str]:
    lineage = cfg["lineage"]
    report_path = resolve(lineage["rg2_0_report"])
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "PASS_RG2_0_MAP_TOPOLOGY_PREFLIGHT":
        raise RuntimeError("RG2.0 map topology report not PASS")
    if report["provenance"]["relationships_sha256"] != lineage["expected_rg2_0_relationships_sha256"]:
        raise RuntimeError("RG2.0 topology hash changed")
    return report["provenance"]["tile_index_sha256"], sha256_file(report_path)


def load_trajectory_inputs(
    cfg: dict[str, Any],
    key: str,
    expected_map_index_sha: str,
) -> dict[str, Any]:
    tc = cfg["trajectories"][key]
    spec_path = resolve(tc["spec"])
    manifest_path = resolve(tc["manifest"])
    pool_path = resolve(tc["frozen_top20"])
    for path in (spec_path, manifest_path, pool_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    if sha256_file(pool_path) != tc["expected_top20_sha256"]:
        raise RuntimeError(f"{key} frozen Top20 bytes changed")

    spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    if spec["trajectory"]["role"] != tc["role"]:
        raise RuntimeError(f"{key} trajectory role changed")
    if spec["map"]["crs"] != "EPSG:3346" or "768_s256" not in spec["map"]["variants"]:
        raise RuntimeError(f"{key} map contract changed")
    tile_index_path = resolve(spec["map"]["variants"]["768_s256"]["tile_index"])
    if sha256_file(tile_index_path) != expected_map_index_sha:
        raise RuntimeError(f"{key} map index differs from frozen RG2.0 map")

    manifest = pd.read_csv(manifest_path).copy()
    pool = pd.read_csv(pool_path).copy()
    tile = pd.read_csv(tile_index_path).copy()
    for frame, label in (
        (manifest, f"{key} manifest"),
        (pool, f"{key} Top20"),
        (tile, f"{key} tile index"),
    ):
        assert_blind_safe_columns(frame, label)
    require_columns(manifest, MANIFEST_REQUIRED, f"{key} manifest")
    require_columns(pool, POOL_REQUIRED, f"{key} Top20")
    require_columns(tile, TILE_REQUIRED, f"{key} tile index")

    manifest["query_id"] = pd.to_numeric(manifest["query_id"], errors="raise").astype(int)
    pool["query_id"] = pd.to_numeric(pool["query_id"], errors="raise").astype(int)
    pool["pool_rank"] = pd.to_numeric(pool["pool_rank"], errors="raise").astype(int)
    pool["source_rank"] = pd.to_numeric(pool["source_rank"], errors="raise").astype(int)
    pool["tile_id"] = pool["tile_id"].astype(str)
    tile["tile_id"] = tile["tile_id"].astype(str)

    refs = manifest["reference_available"].astype(str).str.strip().str.lower()
    if not refs.isin(["false", "0", "no", "n", "f"]).all():
        raise RuntimeError(f"{key} manifest exposes reference")

    if len(manifest) != int(tc["expected_queries"]) or manifest["query_id"].duplicated().any():
        raise RuntimeError(f"{key} manifest count/uniqueness mismatch")
    if len(pool) != int(tc["expected_pairs"]):
        raise RuntimeError(f"{key} frozen pair count mismatch")
    if pool[["query_id", "tile_id"]].duplicated().any():
        raise RuntimeError(f"{key} duplicate query/tile candidate")

    for qid, group in pool.groupby("query_id", sort=True):
        ordered = group.sort_values("pool_rank")
        if len(ordered) != 20 or ordered["pool_rank"].tolist() != list(range(1, 21)):
            raise RuntimeError(f"{key} q{qid} Top20 rank contract changed")
        top1 = ordered.iloc[0]
        if str(top1["source_view"]) != "center_square" or int(top1["source_rank"]) != 1:
            raise RuntimeError(f"{key} q{qid} center Top1 contract changed")
        if ordered["tile_id"].duplicated().any():
            raise RuntimeError(f"{key} q{qid} repeated tile ID in Top20")

    if set(pool["query_id"]) != set(manifest["query_id"]):
        raise RuntimeError(f"{key} Top20/manifest query sets differ")
    missing_tiles = sorted(set(pool["tile_id"]) - set(tile["tile_id"]))
    if missing_tiles:
        raise RuntimeError(f"{key} Top20 references missing map tiles: {missing_tiles[:20]}")

    return {
        "spec_path": spec_path,
        "manifest_path": manifest_path,
        "pool_path": pool_path,
        "tile_index_path": tile_index_path,
        "manifest": manifest,
        "pool": pool.sort_values(["query_id", "pool_rank"], kind="mergesort").reset_index(drop=True),
        "tile": tile,
        "trajectory_id": spec["trajectory"]["id"],
        "role": spec["trajectory"]["role"],
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


def generate(
    inputs: dict[str, Any],
    q_paths: dict[int, Path],
    t_paths: dict[str, Path],
    *,
    progress_every: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], float]:
    detector = orb_subtile.create_detector()
    q_cache: dict[int, Any] = {}
    t_cache: dict[str, Any] = {}
    tile_lookup = inputs["tile"].set_index("tile_id")
    rows: list[dict[str, Any]] = []

    def qfeat(qid: int):
        if qid not in q_cache:
            q_cache[qid] = orb_subtile.compute_features(q_paths[qid], detector)
        return q_cache[qid]

    def tfeat(tile_id: str):
        if tile_id not in t_cache:
            t_cache[tile_id] = orb_subtile.compute_features(t_paths[tile_id], detector)
        return t_cache[tile_id]

    started = time.perf_counter()
    pool = inputs["pool"]
    for idx, row in enumerate(pool.itertuples(index=False), start=1):
        if idx == 1 or idx % max(1, progress_every) == 0 or idx == len(pool):
            print(f"{inputs['role']} Top20 geometry pair {idx}/{len(pool)}", flush=True)
        qid = int(row.query_id)
        tile_id = str(row.tile_id)
        q = qfeat(qid)
        t = tfeat(tile_id)
        evidence = analyze_pair(q, t)

        projected_e = math.nan
        projected_n = math.nan
        finite_projection = False
        inside = False
        if bool(evidence["homography_ok"]):
            u = float(evidence["projected_tile_u_px"])
            v = float(evidence["projected_tile_v_px"])
            tile = tile_lookup.loc[tile_id]
            projected_e, projected_n = orb_subtile.tile_pixel_to_map(
                u, v, t.image_shape,
                float(tile["left_easting"]), float(tile["right_easting"]),
                float(tile["bottom_northing"]), float(tile["top_northing"]),
            )
            finite_projection = bool(math.isfinite(projected_e) and math.isfinite(projected_n))
            th, tw = t.image_shape
            inside = bool(
                finite_projection
                and 0.0 <= u < float(tw)
                and 0.0 <= v < float(th)
            )

        out = {
            "trajectory_id": inputs["trajectory_id"],
            "role": inputs["role"],
            "query_id": qid,
            "pool_rank": int(row.pool_rank),
            "tile_id": tile_id,
            "source_view": str(row.source_view),
            "source_rank": int(row.source_rank),
        }
        out.update(evidence)
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
        rows.append(out)

    runtime = float(time.perf_counter() - started)
    frame = pd.DataFrame(rows)
    summary = {
        "queries": int(frame["query_id"].nunique()),
        "candidate_pairs": int(len(frame)),
        "unique_tiles": int(frame["tile_id"].nunique()),
        "homography_ok_pairs": int(frame["homography_ok"].astype(bool).sum()),
        "finite_projection_pairs": int(frame["finite_projection"].astype(bool).sum()),
        "inside_projection_pairs": int(frame["projected_inside_tile"].astype(bool).sum()),
        "queries_with_all_20_homographies": int(
            (frame.groupby("query_id")["homography_ok"].sum() == 20).sum()
        ),
        "queries_with_any_failed_homography": int(
            (frame.groupby("query_id")["homography_ok"].sum() < 20).sum()
        ),
        "feature_cache_query_images": int(len(q_cache)),
        "feature_cache_tile_images": int(len(t_cache)),
    }
    return rows, summary, runtime


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GD1.0 reference-free richer geometric evidence freeze for frozen Top20"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()

    cfg_path = resolve(args.config)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    verify_config(cfg)
    map_index_sha, rg20_report_sha = verify_map_lineage(cfg)

    prepared: dict[str, Any] = {}
    for key in ("development", "blind_stress"):
        inputs = load_trajectory_inputs(cfg, key, map_index_sha)
        q_paths, t_paths = preflight_paths(inputs)
        prepared[key] = (inputs, q_paths, t_paths)
        print(
            f"{key} preflight: {len(inputs['manifest'])} queries, "
            f"{len(inputs['pool'])} frozen Top20 pairs, {len(t_paths)} unique map tiles; "
            "reference loaded=False"
        )

    files: dict[Path, bytes] = {}
    trajectory_reports: dict[str, Any] = {}
    runtime_only: dict[str, float] = {}
    progress = int(cfg["generation"]["progress_every_pairs"])

    for key in ("development", "blind_stress"):
        inputs, q_paths, t_paths = prepared[key]
        rows, stats, runtime = generate(
            inputs, q_paths, t_paths, progress_every=progress
        )
        runtime_only[key] = runtime
        out = resolve(cfg["output_root"]) / key / "gd1_0_top20_geometric_evidence.csv"
        payload = canonical_csv(rows)
        files[out] = payload
        trajectory_reports[key] = {
            **stats,
            "trajectory_id": inputs["trajectory_id"],
            "role": inputs["role"],
            "provenance": {
                "trajectory_spec_sha256": sha256_file(inputs["spec_path"]),
                "blind_manifest_sha256": sha256_file(inputs["manifest_path"]),
                "frozen_top20_sha256": sha256_file(inputs["pool_path"]),
                "map_index_sha256": sha256_file(inputs["tile_index_path"]),
            },
            "output": {
                "path": str(out.relative_to(resolve(cfg["output_root"]))),
                "sha256": sha256_bytes(payload),
                "rows": len(rows),
            },
        }

    report = {
        "stage": "GD1.0",
        "status": "PASS_GD1_0_REFERENCE_FREE_TOP20_GEOMETRIC_EVIDENCE_FREEZE",
        "map_variant": "768_s256",
        "map_crs": "EPSG:3346",
        "frozen_retrieval": cfg["lineage"]["frozen_retrieval"],
        "geometry_contract": cfg["geometry_contract"],
        "evidence_groups": cfg["evidence_groups"],
        "trajectories": trajectory_reports,
        "scientific_boundary": {
            "reference_loaded": False,
            "srt_gps_loaded": False,
            "oracle_labels_loaded": False,
            "candidate_selected": False,
            "reranking_executed": False,
            "score_fusion_executed": False,
            "threshold_tuned": False,
            "learned_frontend_executed": False,
            "temporal_fusion_executed": False,
            "bootstrap_state_modified": False,
        },
        "interpretation": {
            "measurement_stage_only": True,
            "no_candidate_winner": True,
            "no_accuracy_claim": True,
            "failed_geometry_pairs_retained": True,
        },
        "provenance": {
            "config_sha256": sha256_file(cfg_path),
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "baseline_geometry_module_sha256": sha256_file(Path(orb_subtile.__file__).resolve()),
            "evidence_module_sha256": sha256_file(
                Path(sys.modules[analyze_pair.__module__].__file__).resolve()
            ),
            "rg2_0_report_sha256": rg20_report_sha,
            "map_index_sha256": map_index_sha,
        },
    }

    report_path = resolve(cfg["output_root"]) / "gd1_0_freeze_report.json"
    files[report_path] = canonical_json(report)
    freeze_files(files)

    print("Status:", report["status"])
    for key, tr in trajectory_reports.items():
        print(
            f"{key}: queries={tr['queries']} pairs={tr['candidate_pairs']} "
            f"unique_tiles={tr['unique_tiles']} homography_ok={tr['homography_ok_pairs']} "
            f"finite_projection={tr['finite_projection_pairs']} "
            f"inside={tr['inside_projection_pairs']} "
            f"all20_h_queries={tr['queries_with_all_20_homographies']} "
            f"any_failed_h_queries={tr['queries_with_any_failed_homography']}"
        )
        print("  evidence SHA256:", tr["output"]["sha256"])
        print(f"  observed runtime s (not frozen): {runtime_only[key]:.3f}")
    print("Reference loaded: False")
    print("Candidate selected/reranked: False")
    print("Accuracy claim: False")
    print("Freeze report:", report_path)


if __name__ == "__main__":
    main()
