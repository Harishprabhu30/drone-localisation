#!/usr/bin/env python3
"""LF2.0-B: reference-free uniform40 SuperPoint+LightGlue Top20 screening freeze."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import numpy as np
import pandas as pd
import yaml

from uavloc.geometry.lightglue_absolute_frontend import (
    FEATURE_TYPE,
    LIGHTGLUE_DEPTH_CONFIDENCE,
    LIGHTGLUE_FILTER_THRESHOLD,
    LIGHTGLUE_WIDTH_CONFIDENCE,
    MAX_KEYPOINTS,
    MIN_HOMOGRAPHY_INLIERS,
    PINNED_COMMIT,
    RANSAC_THRESHOLD_PX,
    RESIZE_LONG,
    SuperPointLightGlueAbsoluteFrontend,
)

ROOT = REPO_ROOT
DEFAULT_CONFIG = ROOT / "configs/research/learned_geometric_frontend_lf2_0b.yaml"

MANIFEST_REQUIRED = {"query_id", "image_path", "reference_available"}
POOL_REQUIRED = {"query_id", "pool_rank", "tile_id", "source_view", "source_rank"}
TILE_REQUIRED = {
    "tile_id", "tile_path", "left_easting", "right_easting",
    "bottom_northing", "top_northing",
}


def resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_csv(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(
        index=False, lineterminator="\n", float_format="%.9f"
    ).encode("utf-8")


def canonical_json(value: dict[str, Any]) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def freeze(files: dict[Path, bytes]) -> None:
    changed = [
        str(path)
        for path, data in files.items()
        if path.exists() and path.read_bytes() != data
    ]
    if changed:
        raise RuntimeError(
            "Refusing to overwrite differing LF2.0-B freeze: " + ", ".join(changed)
        )
    for path, data in files.items():
        if path.exists():
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as handle:
            handle.write(data)


def require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"{label} missing columns: {missing}")


def resolve_existing(value: Any, *roots: Path) -> Path:
    raw = Path(str(value)).expanduser()
    candidates = [raw] if raw.is_absolute() else [ROOT / raw, *[r / raw for r in roots]]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "Could not resolve file; tried: " + ", ".join(map(str, candidates))
    )


def bool_value(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes", "y", "t"}


def finite_median(values: pd.Series) -> float | None:
    array = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    array = array[np.isfinite(array)]
    return float(np.median(array)) if len(array) else None


def uniform_query_selection(query_ids: list[int], count: int) -> tuple[list[int], list[int]]:
    ordered = sorted(int(q) for q in query_ids)
    if count < 2:
        raise ValueError("uniform screening count must be at least 2")
    if count > len(ordered):
        raise ValueError("uniform screening count exceeds available queries")
    indices = np.rint(
        np.linspace(0, len(ordered) - 1, num=count, endpoint=True)
    ).astype(int).tolist()
    if len(set(indices)) != count:
        raise RuntimeError("uniform index selection produced duplicate indices")
    selected = [ordered[index] for index in indices]
    return indices, selected


def verify_config(cfg: dict[str, Any]) -> None:
    if cfg.get("stage") != "LF2.0-B":
        raise RuntimeError("Expected LF2.0-B config")
    x = cfg["lightglue"]
    expected = {
        "pinned_commit": PINNED_COMMIT,
        "feature_type": FEATURE_TYPE,
        "max_keypoints": MAX_KEYPOINTS,
        "resize_long": RESIZE_LONG,
        "filter_threshold": LIGHTGLUE_FILTER_THRESHOLD,
        "depth_confidence": LIGHTGLUE_DEPTH_CONFIDENCE,
        "width_confidence": LIGHTGLUE_WIDTH_CONFIDENCE,
        "ransac_threshold_px": RANSAC_THRESHOLD_PX,
        "min_homography_inliers": MIN_HOMOGRAPHY_INLIERS,
        "profile": "default_adaptive",
    }
    for key, value in expected.items():
        if x.get(key) != value:
            raise RuntimeError(f"LightGlue contract changed: {key}")
    screening = cfg["screening"]
    if screening.get("selection_method") != "uniform_index_over_sorted_query_ids_v1":
        raise RuntimeError("Screening query selection method changed")
    if int(screening.get("query_count", 0)) != 40:
        raise RuntimeError("Screening query count changed")
    if int(screening.get("candidates_per_query", 0)) != 20:
        raise RuntimeError("Screening candidate count changed")
    if int(screening.get("expected_pairs", 0)) != 800:
        raise RuntimeError("Screening pair count changed")
    if screening.get("include_endpoints") is not True:
        raise RuntimeError("Uniform screening must include endpoints")
    generation = cfg["generation"]
    if generation.get("pair_order") != "query_id_then_pool_rank":
        raise RuntimeError("Pair ordering changed")
    if generation.get("runtime_fields_frozen") is not False:
        raise RuntimeError("Runtime fields must not enter frozen evidence")
    if generation.get("retain_failed_pairs") is not True:
        raise RuntimeError("Failed pairs must be retained")
    for key, value in cfg["scientific_boundary"].items():
        if value is not False:
            raise RuntimeError(f"{key} must be false")


def verify_prior_smokes(cfg: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    base_block = cfg["lf2_0a"]
    base_path = resolve(base_block["report"])
    base = json.loads(base_path.read_text(encoding="utf-8"))
    if base.get("status") != "PASS_LF2_0A_SUPERPOINT_LIGHTGLUE_ABSOLUTE_PREFLIGHT_SMOKE":
        raise RuntimeError("LF2.0-A baseline report not PASS")
    if base["outputs"]["smoke_evidence"]["sha256"] != base_block["expected_smoke_sha256"]:
        raise RuntimeError("LF2.0-A smoke hash mismatch")
    if base["frontend"]["superpoint_weights_sha256"] != base_block["expected_superpoint_weights_sha256"]:
        raise RuntimeError("LF2.0-A SuperPoint weights hash mismatch")
    if base["frontend"]["lightglue_weights_sha256"] != base_block["expected_lightglue_weights_sha256"]:
        raise RuntimeError("LF2.0-A LightGlue weights hash mismatch")
    if base["frontend"]["pinned_commit"] != PINNED_COMMIT:
        raise RuntimeError("LF2.0-A pinned commit mismatch")

    hard_block = cfg["lf2_0ah"]
    hard_path = resolve(hard_block["report"])
    hard = json.loads(hard_path.read_text(encoding="utf-8"))
    if hard.get("status") != "PASS_LF2_0AH_SUPERPOINT_LIGHTGLUE_MAX_ACCURACY_SMOKE":
        raise RuntimeError("LF2.0-AH hard-profile report not PASS")
    if hard["outputs"]["smoke_evidence"]["sha256"] != hard_block["expected_smoke_sha256"]:
        raise RuntimeError("LF2.0-AH smoke hash mismatch")
    return base, hard


def load_development(cfg: dict[str, Any]) -> dict[str, Any]:
    block = cfg["development"]
    spec_path = resolve(block["spec"])
    manifest_path = resolve(block["manifest"])
    pool_path = resolve(block["top20"])
    if sha256_file(pool_path) != block["expected_top20_sha256"]:
        raise RuntimeError("development Top20 hash changed")

    spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    tile_index_path = resolve(spec["map"]["variants"]["768_s256"]["tile_index"])
    manifest = pd.read_csv(manifest_path).copy()
    pool = pd.read_csv(pool_path).copy()
    tiles = pd.read_csv(tile_index_path).copy()

    require_columns(manifest, MANIFEST_REQUIRED, "development manifest")
    require_columns(pool, POOL_REQUIRED, "development Top20")
    require_columns(tiles, TILE_REQUIRED, "development tile index")

    manifest["query_id"] = pd.to_numeric(manifest["query_id"], errors="raise").astype(int)
    pool["query_id"] = pd.to_numeric(pool["query_id"], errors="raise").astype(int)
    pool["pool_rank"] = pd.to_numeric(pool["pool_rank"], errors="raise").astype(int)
    pool["source_rank"] = pd.to_numeric(pool["source_rank"], errors="raise").astype(int)
    pool["tile_id"] = pool["tile_id"].astype(str)
    tiles["tile_id"] = tiles["tile_id"].astype(str)

    refs = manifest["reference_available"].astype(str).str.strip().str.lower()
    if not refs.isin(["false", "0", "no", "n", "f"]).all():
        raise RuntimeError("development manifest exposes reference")
    if len(manifest) != int(block["expected_full_queries"]):
        raise RuntimeError("development manifest count mismatch")
    if manifest["query_id"].duplicated().any():
        raise RuntimeError("development manifest query IDs are not unique")
    if len(pool) != int(block["expected_full_pairs"]):
        raise RuntimeError("development Top20 pair count mismatch")
    if pool[["query_id", "tile_id"]].duplicated().any():
        raise RuntimeError("development duplicate query/tile candidate")
    if set(pool["query_id"]) != set(manifest["query_id"]):
        raise RuntimeError("development query set mismatch")

    for qid, group in pool.groupby("query_id", sort=True):
        ordered = group.sort_values("pool_rank")
        if len(ordered) != 20 or ordered["pool_rank"].tolist() != list(range(1, 21)):
            raise RuntimeError(f"development q{qid} Top20 rank contract changed")
        top1 = ordered.iloc[0]
        if str(top1["source_view"]) != "center_square" or int(top1["source_rank"]) != 1:
            raise RuntimeError(f"development q{qid} Top1 contract changed")

    selected_indices, selected_ids = uniform_query_selection(
        manifest["query_id"].tolist(), int(cfg["screening"]["query_count"])
    )
    selected_manifest = manifest[manifest["query_id"].isin(selected_ids)].copy()
    selected_pool = pool[pool["query_id"].isin(selected_ids)].copy()
    selected_pool = selected_pool.sort_values(
        ["query_id", "pool_rank"], kind="mergesort"
    ).reset_index(drop=True)
    if len(selected_manifest) != int(cfg["screening"]["query_count"]):
        raise RuntimeError("selected manifest query count mismatch")
    if len(selected_pool) != int(cfg["screening"]["expected_pairs"]):
        raise RuntimeError("selected screening pair count mismatch")

    missing_tiles = sorted(set(selected_pool["tile_id"]) - set(tiles["tile_id"]))
    if missing_tiles:
        raise RuntimeError(f"screening missing tile IDs: {missing_tiles[:20]}")

    return {
        "spec_path": spec_path,
        "manifest_path": manifest_path,
        "pool_path": pool_path,
        "tile_index_path": tile_index_path,
        "manifest": selected_manifest,
        "pool": selected_pool,
        "tiles": tiles,
        "trajectory_id": str(spec["trajectory"]["id"]),
        "selected_indices": selected_indices,
        "selected_query_ids": selected_ids,
    }


def preflight_paths(inputs: dict[str, Any]) -> tuple[dict[int, Path], dict[str, Path]]:
    query_paths: dict[int, Path] = {}
    for row in inputs["manifest"].itertuples(index=False):
        query_paths[int(row.query_id)] = resolve_existing(row.image_path)
    tile_lookup = inputs["tiles"].set_index("tile_id")
    tile_paths: dict[str, Path] = {}
    for tile_id in sorted(set(inputs["pool"]["tile_id"])):
        tile_paths[tile_id] = resolve_existing(
            tile_lookup.loc[tile_id, "tile_path"], inputs["tile_index_path"].parent
        )
    return query_paths, tile_paths


def freeze_value(value: Any) -> Any:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value)
    return value


def generate(
    frontend: SuperPointLightGlueAbsoluteFrontend,
    inputs: dict[str, Any],
    query_paths: dict[int, Path],
    tile_paths: dict[str, Path],
    *,
    progress_every: int,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, float]]:
    tile_lookup = inputs["tiles"].set_index("tile_id")
    rows: list[dict[str, Any]] = []
    feature_ms: list[float] = []
    match_ms: list[float] = []
    geometry_ms: list[float] = []
    total_ms: list[float] = []
    wall_start = time.perf_counter()

    pool = inputs["pool"]
    for index, row in enumerate(pool.itertuples(index=False), start=1):
        if index == 1 or index % max(1, progress_every) == 0 or index == len(pool):
            print(
                f"{inputs['trajectory_id']} LightGlue screening pair {index}/{len(pool)}",
                flush=True,
            )
        qid = int(row.query_id)
        tile_id = str(row.tile_id)
        tile = tile_lookup.loc[tile_id]
        measurement = frontend.measure_pair(
            query_paths[qid],
            tile_paths[tile_id],
            tile_bounds=(
                float(tile["left_easting"]),
                float(tile["right_easting"]),
                float(tile["bottom_northing"]),
                float(tile["top_northing"]),
            ),
        )
        feature_ms.append(float(measurement.runtime.feature_ms))
        match_ms.append(float(measurement.runtime.match_ms))
        geometry_ms.append(float(measurement.runtime.geometry_ms))
        total_ms.append(float(measurement.runtime.total_ms))

        out = {
            "trajectory_id": inputs["trajectory_id"],
            "query_id": qid,
            "pool_rank": int(row.pool_rank),
            "tile_id": tile_id,
            "source_view": str(row.source_view),
            "source_rank": int(row.source_rank),
            "frontend_name": measurement.frontend_name,
            "good_matches": int(measurement.good_matches),
            "inliers": int(measurement.inliers),
            "homography_ok": bool(measurement.homography_ok),
            "finite_projection": bool(measurement.finite_projection),
            "projected_easting": float(measurement.projected_easting),
            "projected_northing": float(measurement.projected_northing),
            "projected_inside_tile": bool(measurement.projected_inside_tile),
        }
        for name, value in sorted(measurement.evidence.items()):
            if name in {"query_feature_cache_hit", "tile_feature_cache_hit"}:
                continue
            out[f"evidence_{name}"] = freeze_value(value)
        rows.append(out)

    wall_s = float(time.perf_counter() - wall_start)
    frame = pd.DataFrame(rows).sort_values(
        ["query_id", "pool_rank"], kind="mergesort"
    ).reset_index(drop=True)

    h = frame["homography_ok"].map(bool_value)
    finite = frame["finite_projection"].map(bool_value)
    inside = frame["projected_inside_tile"].map(bool_value)
    h_per_query = h.groupby(frame["query_id"]).sum()
    summary = {
        "queries": int(frame["query_id"].nunique()),
        "pairs": int(len(frame)),
        "unique_tiles": int(frame["tile_id"].nunique()),
        "homography_ok": int(h.sum()),
        "finite_projection": int(finite.sum()),
        "inside_projection": int(inside.sum()),
        "queries_any_homography": int((h_per_query > 0).sum()),
        "queries_all20_homographies": int((h_per_query == 20).sum()),
        "queries_no_homography": int((h_per_query == 0).sum()),
        "median_matches": finite_median(frame["good_matches"]),
        "median_inliers": finite_median(frame["inliers"]),
    }
    runtime = {
        "wall_s": wall_s,
        "feature_compute_ms_sum": float(np.sum(feature_ms)),
        "match_ms_sum": float(np.sum(match_ms)),
        "geometry_ms_sum": float(np.sum(geometry_ms)),
        "median_total_ms_pair": float(np.median(total_ms)),
        "mean_total_ms_pair": float(np.mean(total_ms)),
    }
    return frame, summary, runtime


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LF2.0-B reference-free SuperPoint+LightGlue uniform40 Top20 screening freeze"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()

    cfg_path = resolve(args.config)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    verify_config(cfg)
    baseline_smoke, hard_smoke = verify_prior_smokes(cfg)

    inputs = load_development(cfg)
    query_paths, tile_paths = preflight_paths(inputs)
    print(
        f"development screening preflight: queries={len(inputs['selected_query_ids'])} "
        f"pairs={len(inputs['pool'])} unique_tiles={len(tile_paths)} "
        "reference_loaded=False"
    )
    print("selected query IDs:", ",".join(map(str, inputs["selected_query_ids"])))

    frontend = SuperPointLightGlueAbsoluteFrontend(
        resolve(cfg["lightglue"]["repo"]),
        device_name=str(cfg["lightglue"]["device"]),
        max_keypoints=int(cfg["lightglue"]["max_keypoints"]),
        resize_long=int(cfg["lightglue"]["resize_long"]),
        filter_threshold=float(cfg["lightglue"]["filter_threshold"]),
        depth_confidence=float(cfg["lightglue"]["depth_confidence"]),
        width_confidence=float(cfg["lightglue"]["width_confidence"]),
    )
    meta = frontend.describe()
    if meta["pinned_commit"] != PINNED_COMMIT:
        raise RuntimeError("LightGlue pinned commit changed")
    if meta["superpoint_weights_sha256"] != cfg["lf2_0a"]["expected_superpoint_weights_sha256"]:
        raise RuntimeError("SuperPoint weights changed since LF2.0-A")
    if meta["lightglue_weights_sha256"] != cfg["lf2_0a"]["expected_lightglue_weights_sha256"]:
        raise RuntimeError("LightGlue weights changed since LF2.0-A")

    frame, summary, runtime = generate(
        frontend,
        inputs,
        query_paths,
        tile_paths,
        progress_every=int(cfg["generation"]["progress_every_pairs"]),
    )

    out_root = resolve(cfg["output_root"])
    evidence_path = out_root / "development" / "lf2_0b_lightglue_uniform40_top20_evidence.csv"
    evidence_bytes = canonical_csv(frame)
    report = {
        "stage": "LF2.0-B",
        "status": "PASS_LF2_0B_REFERENCE_FREE_LIGHTGLUE_UNIFORM40_SCREENING_FREEZE",
        "frontend": meta,
        "screening_contract": {
            "selection_method": cfg["screening"]["selection_method"],
            "full_query_count": int(cfg["development"]["expected_full_queries"]),
            "selected_query_count": int(cfg["screening"]["query_count"]),
            "candidates_per_query": int(cfg["screening"]["candidates_per_query"]),
            "selected_indices_zero_based": inputs["selected_indices"],
            "selected_query_ids": inputs["selected_query_ids"],
        },
        "development": {
            **summary,
            "output": {
                "path": str(evidence_path.relative_to(ROOT)),
                "sha256": hashlib.sha256(evidence_bytes).hexdigest(),
                "rows": int(len(frame)),
            },
            "provenance": {
                "spec_sha256": sha256_file(inputs["spec_path"]),
                "manifest_sha256": sha256_file(inputs["manifest_path"]),
                "top20_sha256": sha256_file(inputs["pool_path"]),
                "tile_index_sha256": sha256_file(inputs["tile_index_path"]),
            },
        },
        "scientific_boundary": {
            "reference_loaded": False,
            "oracle_labels_loaded": False,
            "full_403_benchmark_executed": False,
            "candidate_reranked": False,
            "selector_executed": False,
            "threshold_tuned": False,
            "model_trained": False,
            "temporal_fusion_executed": False,
            "bootstrap_state_modified": False,
            "runtime_fields_frozen": False,
        },
        "interpretation": {
            "default_adaptive_profile_selected_for_screening": True,
            "official_max_accuracy_profile_rejected_by_smoke": True,
            "failed_pairs_retained": True,
            "no_accuracy_claim": True,
            "runtime_printed_not_hashed": True,
        },
        "provenance": {
            "config_sha256": sha256_file(cfg_path),
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "lf2_0a_report_sha256": sha256_file(resolve(cfg["lf2_0a"]["report"])),
            "lf2_0ah_report_sha256": sha256_file(resolve(cfg["lf2_0ah"]["report"])),
            "lf2_0a_smoke_sha256": baseline_smoke["outputs"]["smoke_evidence"]["sha256"],
            "lf2_0ah_smoke_sha256": hard_smoke["outputs"]["smoke_evidence"]["sha256"],
            "superpoint_weights_sha256": meta["superpoint_weights_sha256"],
            "lightglue_weights_sha256": meta["lightglue_weights_sha256"],
            "lightglue_pinned_commit": meta["pinned_commit"],
            "lightglue_source_mode": meta["source_mode"],
        },
    }
    report_path = out_root / "lf2_0b_freeze_report.json"
    freeze({
        evidence_path: evidence_bytes,
        report_path: canonical_json(report),
    })

    print("Status:", report["status"])
    print("LightGlue pinned commit:", meta["pinned_commit"])
    print("LightGlue source mode:", meta["source_mode"])
    print("SuperPoint weights SHA256:", meta["superpoint_weights_sha256"])
    print("LightGlue weights SHA256:", meta["lightglue_weights_sha256"])
    print("Device:", meta["device"])
    print(
        f"development: queries={summary['queries']} pairs={summary['pairs']} "
        f"unique_tiles={summary['unique_tiles']} homography_ok={summary['homography_ok']} "
        f"finite={summary['finite_projection']} inside={summary['inside_projection']} "
        f"queries_any_h={summary['queries_any_homography']} "
        f"queries_all20_h={summary['queries_all20_homographies']} "
        f"queries_no_h={summary['queries_no_homography']} "
        f"median_matches={summary['median_matches']:.3f} "
        f"median_inliers={summary['median_inliers']:.3f}"
    )
    print("  evidence SHA256:", report["development"]["output"]["sha256"])
    print(
        f"  observed runtime (not frozen): wall_s={runtime['wall_s']:.3f} "
        f"feature_compute_s={runtime['feature_compute_ms_sum']/1000.0:.3f} "
        f"match_s={runtime['match_ms_sum']/1000.0:.3f} "
        f"geometry_s={runtime['geometry_ms_sum']/1000.0:.3f} "
        f"median_total_ms_pair={runtime['median_total_ms_pair']:.3f}"
    )
    print("Reference loaded: False")
    print("Full 403-query benchmark executed: False")
    print("Candidate reranked/selected: False")
    print("Accuracy claim: False")
    print("Freeze report:", report_path)


if __name__ == "__main__":
    main()
