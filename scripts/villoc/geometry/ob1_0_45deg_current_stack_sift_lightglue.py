#!/usr/bin/env python3
"""OB1.0 — 45-degree oblique stress test of the current VILLOC stack.

Reference-free phase:
  112 oblique queries
  -> DINOv2 img518 against current 768_s256 map cache
  -> Top20
  -> promoted SIFT+LightGlue
  -> query-center homography projection
  -> frozen selector: inliers desc, pool-rank asc, tile-id asc
  -> immutable evidence freeze

Only after that freeze is complete does this script attach the existing
development SRT/GPS reference for evaluation.

Important: for a 45-degree camera the projected query-image center need not
coincide with the UAV body ground point. Therefore projected-center-vs-body
error is reported as a stress diagnostic, not as a pure body-localization
metric. Tile containment / tile-center metrics are reported separately.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
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

import numpy as np
import pandas as pd
import yaml
from pyproj import Transformer

from uavloc.geometry.sift_lightglue_absolute_frontend import (
    PINNED_COMMIT,
    PINNED_LIGHTGLUE_BLOB,
    PINNED_SIFT_BLOB,
    SIFTLightGlueAbsoluteFrontend,
)

ROOT = REPO_ROOT
DEFAULT_CONFIG = ROOT / "configs/research/villoc_45deg_sift_lightglue_ob1.yaml"
DINO_BUILDER = ROOT / "scripts/villoc/s8_11bc_build_dinov2_caches.py"

PROTOCOL_KEYS = (
    "model_name", "image_size", "crop_mode", "pooling",
    "normalization", "l2_normalize", "descriptor_dtype",
)


def resolve(v):
    p = Path(v).expanduser()
    return p if p.is_absolute() else ROOT / p


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False, lineterminator="\n", float_format="%.9f").encode()


def canonical_json(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def freeze(files: dict[Path, bytes]) -> None:
    differing = [
        str(path)
        for path, payload in files.items()
        if path.exists() and path.read_bytes() != payload
    ]
    if differing:
        raise RuntimeError(
            "Refusing to overwrite differing OB1.0 frozen outputs: "
            + ", ".join(differing)
        )
    for path, payload in files.items():
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as f:
                f.write(payload)


def dynamic_import(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def normalize_id(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)


def find_col(df: pd.DataFrame, names: list[str], label: str) -> str:
    lower = {str(c).lower(): c for c in df.columns}
    for name in names:
        if name.lower() in lower:
            return lower[name.lower()]
    raise KeyError(f"{label}: none of {names} found; columns={list(df.columns)}")


def resolve_existing(value, *roots: Path) -> Path:
    p = Path(str(value)).expanduser()
    candidates = [p] if p.is_absolute() else [ROOT / p, *[r / p for r in roots]]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(str(value))


def load_cache(path: Path) -> dict[str, Any]:
    data = np.load(path, allow_pickle=False)
    required = {"descriptors", "ids", "meta_json"}
    missing = required - set(data.files)
    if missing:
        raise RuntimeError(f"{path}: missing cache keys {sorted(missing)}")
    meta_raw = data["meta_json"]
    if hasattr(meta_raw, "item"):
        meta_raw = meta_raw.item()
    return {
        "descriptors": data["descriptors"].astype(np.float32),
        "ids": data["ids"].astype(str),
        "meta": json.loads(str(meta_raw)),
    }


def protocol_signature(meta: dict[str, Any]) -> dict[str, Any]:
    protocol = meta.get("protocol", {})
    return {key: protocol.get(key) for key in PROTOCOL_KEYS}


def verify_config(cfg: dict[str, Any]) -> None:
    if cfg.get("stage") != "OB1.0":
        raise RuntimeError("Expected OB1.0 config")
    if cfg["map"]["variant"] != "768_s256":
        raise RuntimeError("OB1.0 must use 768_s256")
    if int(cfg["retrieval"]["image_size"]) != 518:
        raise RuntimeError("OB1.0 must use DINO img518")
    if int(cfg["retrieval"]["top_k"]) != 20:
        raise RuntimeError("OB1.0 must use Top20")
    if int(cfg["retrieval"]["expected_pairs"]) != 2240:
        raise RuntimeError("OB1.0 expected 112 x 20 = 2240 pairs")
    if cfg["sift_lightglue"]["device"] != "cpu":
        raise RuntimeError("OB1.0 frozen frontend must use CPU")
    if cfg["sift_lightglue"]["selector"]["primary"] != "inliers_desc":
        raise RuntimeError("OB1.0 selector changed")
    if cfg["sift_lightglue"]["selector"]["tiebreak"] != [
        "pool_rank_asc", "tile_id_asc"
    ]:
        raise RuntimeError("OB1.0 selector tie-break changed")
    b = cfg["scientific_boundary"]
    for key in (
        "rebuild_oblique_retrieval_for_current_stack",
        "use_768_s256",
        "use_img518_dino",
        "truth_attached_only_after_freeze",
        "projected_query_center_vs_body_is_stress_metric",
        "oblique_footprint_offset_not_corrected",
    ):
        if b[key] is not True:
            raise RuntimeError(f"{key} must remain true")
    for key in (
        "sift_parameters_changed",
        "threshold_tuning_allowed",
        "selector_change_allowed",
        "reference_used_before_freeze",
    ):
        if b[key] is not False:
            raise RuntimeError(f"{key} must remain false")


def build_or_load_query_cache(
    cfg: dict[str, Any],
    query_ids: np.ndarray,
    query_paths: list[Path],
    source_manifest_path: Path,
    map_cache: dict[str, Any],
    out_path: Path,
) -> dict[str, Any]:
    expected_sig = protocol_signature(map_cache["meta"])
    expected = {
        "model_name": "dinov2_vits14",
        "image_size": int(cfg["retrieval"]["image_size"]),
        "crop_mode": cfg["retrieval"]["crop_mode"],
        "pooling": cfg["retrieval"]["pooling"],
        "normalization": "imagenet",
        "l2_normalize": True,
        "descriptor_dtype": "float32",
    }
    if expected_sig != expected:
        raise RuntimeError(
            f"Current 768 map cache protocol mismatch: {expected_sig} != {expected}"
        )

    if out_path.is_file():
        cache = load_cache(out_path)
        if protocol_signature(cache["meta"]) != expected_sig:
            raise RuntimeError("Existing OB1.0 query cache protocol mismatch")
        if cache["ids"].tolist() != query_ids.astype(str).tolist():
            raise RuntimeError("Existing OB1.0 query cache IDs mismatch")
        return cache

    builder = dynamic_import("ob1_dino_builder", DINO_BUILDER)
    protocol = builder.Protocol(
        model_name=expected["model_name"],
        device=str(cfg["retrieval"]["device"]),
        image_size=int(expected["image_size"]),
        crop_mode=str(expected["crop_mode"]),
        pooling=str(expected["pooling"]),
        normalization=str(expected["normalization"]),
        l2_normalize=bool(expected["l2_normalize"]),
        descriptor_dtype=str(expected["descriptor_dtype"]),
        batch_size=1,
    )

    print("Building 45deg DINO img518 query cache...", flush=True)
    model, torch = builder.load_model(protocol)
    started = time.perf_counter()
    descriptors = builder.encode_paths(
        query_paths, model, torch, protocol, "OB1.0 45deg queries"
    ).astype(np.float32)
    runtime_s = time.perf_counter() - started

    meta = {
        "stage": "OB1.0",
        "cache_kind": "query",
        "protocol": expected,
        "source_manifest": str(source_manifest_path),
        "source_manifest_sha256": sha256_file(source_manifest_path),
        "ordered_ids_sha256": hashlib.sha256(
            "\n".join(query_ids.astype(str)).encode()
        ).hexdigest(),
        "runtime_s_not_frozen_claim": runtime_s,
        "reference_fields_used": False,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        descriptors=descriptors,
        ids=query_ids.astype(str),
        meta_json=np.array(json.dumps(meta, sort_keys=True)),
    )
    return load_cache(out_path)


def build_candidate_pool(qcache, mcache, top_k: int) -> pd.DataFrame:
    qd = qcache["descriptors"]
    md = mcache["descriptors"]
    if qd.ndim != 2 or md.ndim != 2 or qd.shape[1] != md.shape[1]:
        raise RuntimeError("DINO descriptor dimensions mismatch")
    if not np.isfinite(qd).all() or not np.isfinite(md).all():
        raise RuntimeError("Non-finite DINO descriptors")
    similarity = qd @ md.T
    order = np.argsort(-similarity, axis=1)[:, :top_k]
    rows = []
    for qi, qid in enumerate(qcache["ids"]):
        for rank, mi in enumerate(order[qi], start=1):
            rows.append({
                "query_id": str(qid),
                "pool_rank": int(rank),
                "tile_id": str(mcache["ids"][int(mi)]),
                "dino_score": float(similarity[qi, int(mi)]),
            })
    return pd.DataFrame(rows)


def select_winners(pair: pd.DataFrame) -> pd.DataFrame:
    x = pair.copy()
    x["query_id"] = normalize_id(x["query_id"])
    x["tile_id"] = normalize_id(x["tile_id"])
    x["pool_rank"] = pd.to_numeric(x["pool_rank"], errors="raise").astype(int)
    x["inliers"] = pd.to_numeric(x["inliers"], errors="raise").astype(int)
    x = x.sort_values(
        ["query_id", "inliers", "pool_rank", "tile_id"],
        ascending=[True, False, True, True],
        kind="mergesort",
    )
    rows = []
    for qid, group in x.groupby("query_id", sort=False):
        g = group.reset_index(drop=True)
        winner = g.iloc[0]
        runner = g.iloc[1]
        rows.append({
            "query_id": str(qid),
            "selected_pool_rank": int(winner["pool_rank"]),
            "selected_tile_id": str(winner["tile_id"]),
            "selected_dino_score": float(winner["dino_score"]),
            "selected_matches": int(winner["matches"]),
            "selected_inliers": int(winner["inliers"]),
            "runner_up_inliers": int(runner["inliers"]),
            "inlier_margin": int(winner["inliers"] - runner["inliers"]),
            "selected_homography_ok": bool(winner["homography_ok"]),
            "selected_finite_projection": bool(winner["finite_projection"]),
            "selected_projected_inside_tile": bool(winner["projected_inside_tile"]),
            "selected_projected_easting": float(winner["projected_easting"]),
            "selected_projected_northing": float(winner["projected_northing"]),
        })
    return pd.DataFrame(rows)


def ensure_ref_free_frozen(cfg, manifest_path, manifest, tile_index_path, tiles, map_cache_path, map_cache):
    outroot = resolve(cfg["output_root"])
    candidate_path = outroot / "ob1_0_dino_top20_768_s256_reference_free.csv"
    pair_path = outroot / "ob1_0_sift_lightglue_pair_reference_free_evidence.csv"
    selection_path = outroot / "ob1_0_sift_lightglue_selection_reference_free_evidence.csv"
    freeze_report_path = outroot / "ob1_0_reference_free_freeze_report.json"
    query_cache_path = outroot / "ob1_0_dinov2_45deg_query_img518.npz"

    if all(p.is_file() for p in (candidate_path, pair_path, selection_path, freeze_report_path)):
        report = json.loads(freeze_report_path.read_text())
        if report.get("status") != "PASS_OB1_0_REFERENCE_FREE_FREEZE":
            raise RuntimeError("Existing OB1.0 freeze report is not PASS")
        if sha256_file(candidate_path) != report["provenance"]["candidate_pool_sha256"]:
            raise RuntimeError("Frozen candidate-pool bytes changed")
        if sha256_file(pair_path) != report["provenance"]["pair_evidence_sha256"]:
            raise RuntimeError("Frozen pair-evidence bytes changed")
        if sha256_file(selection_path) != report["provenance"]["selection_evidence_sha256"]:
            raise RuntimeError("Frozen selection-evidence bytes changed")
        print("Reusing existing PASS_OB1_0_REFERENCE_FREE_FREEZE")
        return candidate_path, pair_path, selection_path, freeze_report_path

    qid_col = find_col(manifest, ["token0_id", "query_id"], "45deg manifest")
    image_col = find_col(manifest, ["image_path", "query_image"], "45deg manifest")
    ids = normalize_id(manifest[qid_col]).to_numpy()
    if len(ids) != int(cfg["source"]["expected_queries"]):
        raise RuntimeError("45deg query count changed")
    if len(set(ids.tolist())) != len(ids):
        raise RuntimeError("45deg query IDs are not unique")

    query_paths = [resolve_existing(v, manifest_path.parent) for v in manifest[image_col]]
    qcache = build_or_load_query_cache(
        cfg, ids, query_paths, manifest_path, map_cache, query_cache_path
    )
    if protocol_signature(qcache["meta"]) != protocol_signature(map_cache["meta"]):
        raise RuntimeError("Query/map DINO protocol mismatch")

    candidate = build_candidate_pool(qcache, map_cache, int(cfg["retrieval"]["top_k"]))
    if len(candidate) != int(cfg["retrieval"]["expected_pairs"]):
        raise RuntimeError("OB1.0 candidate-pool size changed")

    tiles = tiles.copy()
    tiles["tile_id"] = normalize_id(tiles["tile_id"])
    required_tile = {
        "tile_id", "tile_path", "left_easting", "right_easting",
        "bottom_northing", "top_northing", "center_easting", "center_northing",
    }
    missing = required_tile - set(tiles.columns)
    if missing:
        raise RuntimeError(f"768 tile index missing {sorted(missing)}")
    if set(tiles["tile_id"]) != set(normalize_id(pd.Series(map_cache["ids"]))):
        raise RuntimeError("768 map cache / tile-index identity mismatch")

    query_table = pd.DataFrame({
        "query_id": ids.astype(str),
        "query_path": [str(p) for p in query_paths],
    })
    work = (
        candidate.merge(query_table, on="query_id", validate="many_to_one")
        .merge(tiles[list(required_tile)], on="tile_id", validate="many_to_one")
    )

    frontend = SIFTLightGlueAbsoluteFrontend(
        resolve(cfg["sift_lightglue"]["repo"]),
        device_name=cfg["sift_lightglue"]["device"],
    )
    desc = frontend.describe()
    if desc["pinned_commit"] != cfg["sift_lightglue"]["expected_commit"]:
        raise RuntimeError("LightGlue pinned commit mismatch")
    if PINNED_COMMIT != cfg["sift_lightglue"]["expected_commit"]:
        raise RuntimeError("Adapter pinned commit mismatch")
    if PINNED_SIFT_BLOB != cfg["sift_lightglue"]["expected_sift_blob"]:
        raise RuntimeError("SIFT source blob mismatch")
    if PINNED_LIGHTGLUE_BLOB != cfg["sift_lightglue"]["expected_lightglue_blob"]:
        raise RuntimeError("LightGlue source blob mismatch")
    if desc["lightglue_weights_sha256"] != cfg["sift_lightglue"]["expected_weights_sha256"]:
        raise RuntimeError("LightGlue weight SHA mismatch")

    rows = []
    runtimes = []
    total = len(work)
    for i, row in enumerate(work.itertuples(index=False), start=1):
        if i == 1 or i % 100 == 0 or i == total:
            print(f"OB1.0 SIFT+LightGlue pair {i}/{total}", flush=True)
        measurement = frontend.measure_pair(
            Path(row.query_path),
            resolve_existing(row.tile_path, tile_index_path.parent),
            tile_bounds=(
                float(row.left_easting), float(row.right_easting),
                float(row.bottom_northing), float(row.top_northing),
            ),
        )
        ev = measurement.evidence
        runtimes.append(float(measurement.runtime.total_ms))
        rows.append({
            "query_id": str(row.query_id),
            "pool_rank": int(row.pool_rank),
            "tile_id": str(row.tile_id),
            "dino_score": float(row.dino_score),
            "query_image": str(row.query_path),
            "tile_path": str(resolve_existing(row.tile_path, tile_index_path.parent)),
            "center_easting": float(row.center_easting),
            "center_northing": float(row.center_northing),
            "left_easting": float(row.left_easting),
            "right_easting": float(row.right_easting),
            "bottom_northing": float(row.bottom_northing),
            "top_northing": float(row.top_northing),
            "matches": int(measurement.good_matches),
            "inliers": int(measurement.inliers),
            "inlier_ratio": float(ev["inlier_ratio"]),
            "homography_ok": bool(measurement.homography_ok),
            "finite_projection": bool(measurement.finite_projection),
            "projected_inside_tile": bool(measurement.projected_inside_tile),
            "projected_easting": float(measurement.projected_easting),
            "projected_northing": float(measurement.projected_northing),
            "symmetric_error_median_px": float(ev["symmetric_error_median_px"]),
            "jacobian_anisotropy": float(ev["jacobian_anisotropy"]),
            "footprint_area_fraction_tile": float(ev["footprint_area_fraction_tile"]),
            "match_score_mean": float(ev["match_score_mean"]),
            "match_score_median": float(ev["match_score_median"]),
        })

    pair = pd.DataFrame(rows).sort_values(
        ["query_id", "pool_rank"], kind="mergesort"
    ).reset_index(drop=True)
    selection = select_winners(pair)

    candidate_bytes = canonical_csv(candidate)
    pair_bytes = canonical_csv(pair)
    selection_bytes = canonical_csv(selection)

    freeze_report = {
        "stage": "OB1.0",
        "status": "PASS_OB1_0_REFERENCE_FREE_FREEZE",
        "counts": {
            "queries": int(selection["query_id"].nunique()),
            "pairs": int(len(pair)),
            "candidate_homography_ok": int(pair["homography_ok"].astype(bool).sum()),
            "candidate_finite_projection": int(pair["finite_projection"].astype(bool).sum()),
            "selected_homography_ok": int(selection["selected_homography_ok"].astype(bool).sum()),
            "selected_finite_projection": int(selection["selected_finite_projection"].astype(bool).sum()),
        },
        "reference_free_diagnostics": {
            "median_pair_matches": float(pair["matches"].median()),
            "median_pair_inliers": float(pair["inliers"].median()),
            "median_selected_matches": float(selection["selected_matches"].median()),
            "median_selected_inliers": float(selection["selected_inliers"].median()),
            "median_selected_pool_rank": float(selection["selected_pool_rank"].median()),
            "median_inlier_margin": float(selection["inlier_margin"].median()),
            "median_pair_runtime_ms_not_frozen": float(np.median(runtimes)),
        },
        "scientific_boundary": {
            "reference_loaded": False,
            "srt_gps_used": False,
            "oracle_used": False,
            "accuracy_computed": False,
            "threshold_tuned": False,
            "selector_changed": False,
        },
        "provenance": {
            "source_manifest_sha256": sha256_file(manifest_path),
            "tile_index_sha256": sha256_file(tile_index_path),
            "map_cache_sha256": sha256_file(map_cache_path),
            "query_cache_sha256": sha256_file(query_cache_path),
            "candidate_pool_sha256": sha256_bytes(candidate_bytes),
            "pair_evidence_sha256": sha256_bytes(pair_bytes),
            "selection_evidence_sha256": sha256_bytes(selection_bytes),
            "lightglue": desc,
            "dino_protocol": protocol_signature(map_cache["meta"]),
        },
    }

    freeze({
        candidate_path: candidate_bytes,
        pair_path: pair_bytes,
        selection_path: selection_bytes,
        freeze_report_path: canonical_json(json_safe(freeze_report)),
    })
    return candidate_path, pair_path, selection_path, freeze_report_path


def attach_reference(manifest: pd.DataFrame, selection: pd.DataFrame, pair: pd.DataFrame, cfg):
    qid_col = find_col(manifest, ["token0_id", "query_id"], "45deg manifest")

    # Evaluation is intentionally loaded only here, after the reference-free freeze.
    lower = {str(c).lower(): c for c in manifest.columns}
    if "easting" in lower and "northing" in lower:
        gt_e = pd.to_numeric(manifest[lower["easting"]], errors="raise").to_numpy(float)
        gt_n = pd.to_numeric(manifest[lower["northing"]], errors="raise").to_numpy(float)
    else:
        lat_col = find_col(
            manifest,
            ["latitude", "lat", "gps_lat", "srt_lat"],
            "45deg reference latitude",
        )
        lon_col = find_col(
            manifest,
            ["longitude", "lon", "gps_lon", "srt_lon"],
            "45deg reference longitude",
        )
        lat = pd.to_numeric(manifest[lat_col], errors="raise").to_numpy(float)
        lon = pd.to_numeric(manifest[lon_col], errors="raise").to_numpy(float)
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:3346", always_xy=True)
        gt_e, gt_n = transformer.transform(lon, lat)
        gt_e = np.asarray(gt_e, float)
        gt_n = np.asarray(gt_n, float)

    truth = pd.DataFrame({
        "query_id": normalize_id(manifest[qid_col]),
        "gt_easting": gt_e,
        "gt_northing": gt_n,
    })

    pair_eval = pair.merge(truth, on="query_id", validate="many_to_one")
    pair_eval["contains_body"] = (
        (pair_eval["gt_easting"] >= pair_eval["left_easting"])
        & (pair_eval["gt_easting"] <= pair_eval["right_easting"])
        & (pair_eval["gt_northing"] >= pair_eval["bottom_northing"])
        & (pair_eval["gt_northing"] <= pair_eval["top_northing"])
    )
    pair_eval["tile_center_error_m"] = np.hypot(
        pair_eval["center_easting"] - pair_eval["gt_easting"],
        pair_eval["center_northing"] - pair_eval["gt_northing"],
    )
    pair_eval["projection_body_error_m"] = np.hypot(
        pair_eval["projected_easting"] - pair_eval["gt_easting"],
        pair_eval["projected_northing"] - pair_eval["gt_northing"],
    )

    selected_eval = selection.merge(truth, on="query_id", validate="one_to_one")
    selected_pair = pair_eval.merge(
        selection[["query_id", "selected_tile_id"]],
        left_on=["query_id", "tile_id"],
        right_on=["query_id", "selected_tile_id"],
        how="inner",
        validate="one_to_one",
    )
    return truth, pair_eval, selected_pair


def metric_summary(values: pd.Series) -> dict[str, Any]:
    v = pd.to_numeric(values, errors="coerce")
    finite = v[np.isfinite(v)]
    if not len(finite):
        return {
            "finite": 0, "nonfinite": int(len(v)),
            "median_m": None, "rmse_m": None, "p95_m": None,
            "le40": 0, "le80": 0, "gt200_or_nonfinite": int(len(v)),
        }
    return {
        "finite": int(len(finite)),
        "nonfinite": int(len(v) - len(finite)),
        "median_m": float(finite.median()),
        "rmse_m": float(np.sqrt(np.mean(np.square(finite.to_numpy(float))))),
        "p95_m": float(np.percentile(finite.to_numpy(float), 95)),
        "le40": int((finite <= 40.0).sum()),
        "le80": int((finite <= 80.0).sum()),
        "gt200_or_nonfinite": int((finite > 200.0).sum() + (len(v) - len(finite))),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = ap.parse_args()

    cfg_path = resolve(args.config)
    cfg = yaml.safe_load(cfg_path.read_text())
    verify_config(cfg)

    manifest_path = resolve(cfg["source"]["query_manifest"])
    tile_index_path = resolve(cfg["map"]["tile_index"])
    map_cache_path = resolve(cfg["map"]["descriptor_cache"])
    for path in (manifest_path, tile_index_path, map_cache_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    manifest = pd.read_csv(manifest_path)
    tiles = pd.read_csv(tile_index_path)
    map_cache = load_cache(map_cache_path)

    if len(manifest) != int(cfg["source"]["expected_queries"]):
        raise RuntimeError("45deg query count changed")

    candidate_path, pair_path, selection_path, freeze_report_path = ensure_ref_free_frozen(
        cfg, manifest_path, manifest, tile_index_path, tiles, map_cache_path, map_cache
    )

    print()
    print("Reference-free freeze complete and verified.")
    print("Attaching 45deg development reference only now...", flush=True)

    candidate = pd.read_csv(candidate_path)
    pair = pd.read_csv(pair_path)
    selection = pd.read_csv(selection_path)
    for df in (candidate, pair, selection):
        df["query_id"] = normalize_id(df["query_id"])
    pair["tile_id"] = normalize_id(pair["tile_id"])
    selection["selected_tile_id"] = normalize_id(selection["selected_tile_id"])

    truth, pair_eval, selected_eval = attach_reference(
        manifest, selection, pair, cfg
    )

    qids = sorted(pair_eval["query_id"].unique())
    retrieval_counts = {}
    for k in cfg["evaluation"]["retrieval_topk"]:
        subset = pair_eval[pd.to_numeric(pair_eval["pool_rank"]) <= int(k)]
        retrieval_counts[f"contains_body_top{k}"] = int(
            subset.groupby("query_id")["contains_body"].any().sum()
        )
        retrieval_counts[f"center_le40_top{k}"] = int(
            subset.assign(hit=subset["tile_center_error_m"] <= 40.0)
            .groupby("query_id")["hit"].any().sum()
        )

    selected_projection = metric_summary(selected_eval["projection_body_error_m"])
    selected_center = metric_summary(selected_eval["tile_center_error_m"])

    oracle_projection = (
        pair_eval.groupby("query_id")["projection_body_error_m"].min()
        .reindex(qids)
    )
    oracle_center = (
        pair_eval.groupby("query_id")["tile_center_error_m"].min()
        .reindex(qids)
    )
    oracle_projection_summary = metric_summary(oracle_projection)
    oracle_center_summary = metric_summary(oracle_center)

    selected_contains = int(selected_eval["contains_body"].sum())

    eval_table = selected_eval[
        [
            "query_id", "tile_id", "pool_rank", "inliers", "inlier_ratio",
            "contains_body", "tile_center_error_m", "projection_body_error_m",
            "projected_easting", "projected_northing", "gt_easting", "gt_northing",
        ]
    ].copy()

    outroot = resolve(cfg["output_root"])
    eval_table_path = outroot / "ob1_0_postfreeze_selected_evaluation.csv"
    report_path = outroot / "ob1_0_postfreeze_evaluation_report.json"
    eval_table.to_csv(eval_table_path, index=False)

    report = {
        "stage": "OB1.0",
        "status": "PASS_OB1_0_POSTFREEZE_OBLIQUE_EVALUATION",
        "counts": {
            "queries": int(len(qids)),
            "pairs": int(len(pair_eval)),
            "selected_contains_body": selected_contains,
            **retrieval_counts,
        },
        "selected_tile_center_error": selected_center,
        "selected_projected_query_center_vs_body_error": selected_projection,
        "oracle_tile_center_error_within_top20": oracle_center_summary,
        "oracle_projected_query_center_vs_body_error_within_top20": oracle_projection_summary,
        "interpretation": {
            "primary_viewpoint_stress_question": (
                "Does the current 768_s256 + img518 + SIFT/LightGlue stack "
                "retain useful locality discrimination under 45-degree oblique imagery?"
            ),
            "projected_center_caveat": (
                "For oblique imagery the query-image center can correspond to a "
                "forward visible ground footprint rather than the UAV body point. "
                "Projected-center-vs-body error is therefore a stress diagnostic, "
                "not a pure body-localization error."
            ),
            "historical_45deg_run_not_direct_ablation": (
                "The historical 45deg LightGlue run used 1024_s512, DINO img224, "
                "SuperPoint/LightGlue-style reranking and a different policy. "
                "Comparison is contextual, not single-variable."
            ),
        },
        "scientific_boundary": {
            "reference_free_evidence_frozen_before_truth": True,
            "threshold_tuned": False,
            "selector_changed": False,
            "sift_parameters_changed": False,
            "footprint_offset_corrected": False,
        },
        "provenance": {
            "config_sha256": sha256_file(cfg_path),
            "reference_free_freeze_report_sha256": sha256_file(freeze_report_path),
            "candidate_pool_sha256": sha256_file(candidate_path),
            "pair_evidence_sha256": sha256_file(pair_path),
            "selection_evidence_sha256": sha256_file(selection_path),
            "evaluation_csv_sha256": sha256_file(eval_table_path),
        },
    }
    report_path.write_text(json.dumps(json_safe(report), indent=2, allow_nan=False))

    freeze_report = json.loads(freeze_report_path.read_text())
    rd = freeze_report["reference_free_diagnostics"]
    counts = freeze_report["counts"]

    print()
    print("Status:", report["status"])
    print("Queries/pairs:", f"{len(qids)}/{len(pair_eval)}")
    print("Reference-free selected homography/finite:",
          f"{counts['selected_homography_ok']}/{counts['selected_finite_projection']}")
    print("Reference-free median selected matches/inliers:",
          f"{rd['median_selected_matches']:.1f}/{rd['median_selected_inliers']:.1f}")
    print("Reference-free median selected pool rank:",
          f"{rd['median_selected_pool_rank']:.1f}")
    print("Reference-free median inlier margin:",
          f"{rd['median_inlier_margin']:.1f}")
    print()
    print("DINO current-stack candidate availability:")
    for k in cfg["evaluation"]["retrieval_topk"]:
        print(
            f"  Top{k} contains body / center<=40:",
            f"{retrieval_counts[f'contains_body_top{k}']}/"
            f"{retrieval_counts[f'center_le40_top{k}']}",
        )
    print()
    print("SIFT selected:")
    print("  contains body:", selected_contains)
    print("  tile-center <=40 / <=80:",
          f"{selected_center['le40']}/{selected_center['le80']}")
    print("  tile-center median:", selected_center["median_m"])
    print("  projected query-center vs body <=40 / <=80:",
          f"{selected_projection['le40']}/{selected_projection['le80']}")
    print("  projected query-center vs body median:",
          selected_projection["median_m"])
    print("  projected unsafe >200/nonfinite:",
          selected_projection["gt200_or_nonfinite"])
    print()
    print("Top20 oracle:")
    print("  tile-center <=40 / <=80:",
          f"{oracle_center_summary['le40']}/{oracle_center_summary['le80']}")
    print("  projected query-center vs body <=40 / <=80:",
          f"{oracle_projection_summary['le40']}/{oracle_projection_summary['le80']}")
    print()
    print("IMPORTANT: projected query-center vs body is an oblique stress metric,")
    print("not a pure body-position metric; no footprint offset correction was applied.")
    print("Freeze report:", freeze_report_path)
    print("Evaluation report:", report_path)


if __name__ == "__main__":
    main()
