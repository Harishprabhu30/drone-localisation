#!/usr/bin/env python3
"""RG2.2 reference-free multi-tile sub-tile projection generation and freeze."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import sys
import time
from collections import Counter
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


ROOT = REPO_ROOT
DEFAULT_CONFIG = ROOT / "configs/research/local_map_neighborhood_rg2_2.yaml"

LOCAL_REQUIRED = {
    "trajectory_id", "query_id", "anchor_tile_id", "tile_id", "union_rank",
    "immediate_rank", "overlap_rank", "is_anchor", "in_immediate",
    "in_full_overlap", "center_distance_m", "delta_easting_m",
    "delta_northing_m",
}
MANIFEST_REQUIRED = {
    "trajectory_id", "query_id", "image_path", "reference_available",
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


def require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"{label} missing columns: {missing}")


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


def bool_series(series: pd.Series, label: str) -> pd.Series:
    mapping = {
        "true": True, "false": False, "1": True, "0": False,
        "yes": True, "no": False, "y": True, "n": False,
    }
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    out = series.astype(str).str.strip().str.lower().map(mapping)
    if out.isna().any():
        bad = sorted(series[out.isna()].astype(str).unique().tolist())
        raise RuntimeError(f"{label} has non-boolean values: {bad[:10]}")
    return out.astype(bool)


def resolve_existing_file(value: Any, *, extra_roots: list[Path] | None = None) -> Path:
    raw = str(value).strip()
    if not raw or raw.lower() in {"nan", "none"}:
        raise FileNotFoundError(f"Invalid path value: {value!r}")
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
        raise RuntimeError("Cannot freeze empty RG2.2 table")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    for row in rows:
        cooked = {}
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
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def freeze_files(files: dict[Path, bytes]) -> None:
    changed = [str(p) for p, data in files.items() if p.exists() and p.read_bytes() != data]
    if changed:
        raise RuntimeError("Refusing to overwrite differing RG2.2 freeze: " + ", ".join(changed))
    for path, data in files.items():
        if path.exists():
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as handle:
            handle.write(data)


def verify_config(cfg: dict[str, Any]) -> None:
    if cfg.get("stage") != "RG2.2":
        raise RuntimeError("Expected RG2.2 config")
    if cfg.get("map_variant") != "768_s256" or cfg.get("map_crs") != "EPSG:3346":
        raise RuntimeError("Frozen map contract changed")
    boundary = cfg.get("scientific_boundary", {})
    for key in (
        "reference_allowed", "reference_attachment_enabled", "srt_gps_allowed",
        "oracle_labels_allowed", "retrieval_reranking_enabled",
        "candidate_selection_enabled", "inlier_gate_enabled", "consensus_enabled",
        "bootstrap_enabled", "state_enabled", "learned_model_enabled",
    ):
        if boundary.get(key) is not False:
            raise RuntimeError(f"{key} must be false in RG2.2")
    gen = cfg.get("generation", {})
    if gen.get("candidate_set") != "full_overlap":
        raise RuntimeError("RG2.2 candidate set changed")
    if gen.get("selection_threshold") != "none" or gen.get("consensus_enabled") is not False:
        raise RuntimeError("RG2.2 must freeze raw pair evidence without selection/consensus")
    if gen.get("freeze_failed_pairs") is not True:
        raise RuntimeError("RG2.2 must freeze failed pairs")
    expected = cfg["geometry_contract"]
    actual = orb_subtile.contract()
    if expected != actual:
        raise RuntimeError(
            "Packaged ORB/sub-tile contract differs from frozen RG2.2 config: "
            + json.dumps({"expected": expected, "actual": actual}, sort_keys=True)
        )


def verify_rg2_lineage(cfg: dict[str, Any]) -> tuple[dict[str, Any], str]:
    rg20_path = resolve(cfg["rg2_0"]["report"])
    rg20 = json.loads(rg20_path.read_text(encoding="utf-8"))
    if rg20.get("status") != "PASS_RG2_0_MAP_TOPOLOGY_PREFLIGHT":
        raise RuntimeError("RG2.0 preflight not PASS")
    topology_sha = rg20["provenance"]["relationships_sha256"]
    if topology_sha != cfg["rg2_0"]["expected_relationships_sha256"]:
        raise RuntimeError("RG2.0 topology hash changed")

    rg21a_path = resolve(cfg["rg2_1a"]["report"])
    rg21a = json.loads(rg21a_path.read_text(encoding="utf-8"))
    if rg21a.get("status") != "PASS_RG2_1A_REFERENCE_FREE_NEIGHBORHOOD_FREEZE":
        raise RuntimeError("RG2.1-A freeze not PASS")
    for key in (
        "reference_loaded", "srt_gps_loaded", "image_or_descriptor_loaded",
        "orb_executed", "candidate_generator_executed", "top20_reranked",
        "bootstrap_state_modified", "oracle_evaluation_executed",
    ):
        if rg21a.get("boundaries", {}).get(key) is not False:
            raise RuntimeError(f"RG2.1-A blind boundary changed: {key}")
    return rg21a, topology_sha


def load_inputs(
    cfg: dict[str, Any],
    rg21a: dict[str, Any],
    trajectory_key: str,
) -> dict[str, Any]:
    tc = cfg["trajectories"][trajectory_key]
    fc = cfg["rg2_1a"][trajectory_key]
    spec_path = resolve(tc["spec"])
    manifest_path = resolve(tc["manifest"])
    local_path = resolve(fc["local_neighborhoods"])
    summary_path = resolve(fc["query_summaries"])
    for path in (spec_path, manifest_path, local_path, summary_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    if sha256_file(local_path) != fc["expected_local_neighborhoods_sha256"]:
        raise RuntimeError(f"{trajectory_key} RG2.1-A local freeze hash changed")
    if sha256_file(summary_path) != fc["expected_query_summaries_sha256"]:
        raise RuntimeError(f"{trajectory_key} RG2.1-A summary hash changed")
    report_tr = rg21a["trajectories"][trajectory_key]
    if report_tr["outputs"]["local_neighborhoods"]["sha256"] != fc["expected_local_neighborhoods_sha256"]:
        raise RuntimeError(f"{trajectory_key} RG2.1-A report/local hash mismatch")

    spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    if spec["trajectory"]["role"] != tc["role"]:
        raise RuntimeError(f"{trajectory_key} role mismatch")
    if spec["map"]["crs"] != "EPSG:3346" or "768_s256" not in spec["map"]["variants"]:
        raise RuntimeError(f"{trajectory_key} map contract mismatch")
    tile_index_path = resolve(spec["map"]["variants"]["768_s256"]["tile_index"])

    manifest = pd.read_csv(manifest_path).copy()
    local = pd.read_csv(local_path).copy()
    summaries = pd.read_csv(summary_path).copy()
    assert_blind_safe_columns(manifest, f"{trajectory_key} manifest")
    assert_blind_safe_columns(local, f"{trajectory_key} local freeze")
    assert_blind_safe_columns(summaries, f"{trajectory_key} summaries")
    require_columns(manifest, MANIFEST_REQUIRED, f"{trajectory_key} manifest")
    require_columns(local, LOCAL_REQUIRED, f"{trajectory_key} local freeze")

    manifest["query_id"] = pd.to_numeric(manifest["query_id"], errors="raise").astype(int)
    local["query_id"] = pd.to_numeric(local["query_id"], errors="raise").astype(int)
    local["union_rank"] = pd.to_numeric(local["union_rank"], errors="raise").astype(int)
    for col in ("is_anchor", "in_immediate", "in_full_overlap"):
        local[col] = bool_series(local[col], col)
    refs = manifest["reference_available"].astype(str).str.strip().str.lower()
    if not refs.isin(["false", "0", "no", "n", "f"]).all():
        raise RuntimeError(f"{trajectory_key} manifest exposes reference")
    if len(manifest) != int(tc["expected_queries"]) or manifest["query_id"].duplicated().any():
        raise RuntimeError(f"{trajectory_key} manifest query count/uniqueness mismatch")

    # Generate only the pre-frozen full-overlap membership; never add/remove by geometry.
    pairs = local[local["in_full_overlap"]].copy()
    pairs = pairs.sort_values(["query_id", "union_rank", "tile_id"], kind="mergesort").reset_index(drop=True)
    if len(pairs) != int(tc["expected_pair_rows"]):
        raise RuntimeError(
            f"{trajectory_key} expected {tc['expected_pair_rows']} frozen full-overlap pairs, got {len(pairs)}"
        )
    if pairs[["query_id", "tile_id"]].duplicated().any():
        raise RuntimeError(f"{trajectory_key} duplicate frozen query/tile pair")
    if set(pairs["query_id"]) != set(manifest["query_id"]):
        raise RuntimeError(f"{trajectory_key} frozen pairs/manifests query IDs differ")

    tile = pd.read_csv(tile_index_path).copy()
    require_columns(tile, TILE_REQUIRED, f"{trajectory_key} tile index")
    tile["tile_id"] = tile["tile_id"].astype(str)
    if tile["tile_id"].duplicated().any():
        raise RuntimeError("Tile index duplicate IDs")
    missing_tiles = sorted(set(pairs["tile_id"].astype(str)) - set(tile["tile_id"]))
    if missing_tiles:
        raise RuntimeError(f"{trajectory_key} missing tile IDs: {missing_tiles[:20]}")

    return {
        "spec_path": spec_path,
        "manifest_path": manifest_path,
        "local_path": local_path,
        "summary_path": summary_path,
        "tile_index_path": tile_index_path,
        "manifest": manifest,
        "pairs": pairs,
        "tile": tile,
        "trajectory_id": spec["trajectory"]["id"],
        "role": spec["trajectory"]["role"],
    }


def preflight_paths(inputs: dict[str, Any]) -> tuple[dict[int, Path], dict[str, Path]]:
    manifest = inputs["manifest"]
    pairs = inputs["pairs"]
    tile = inputs["tile"]
    tile_index_path = inputs["tile_index_path"]

    q_paths = {}
    for row in manifest.itertuples(index=False):
        q_paths[int(row.query_id)] = resolve_existing_file(row.image_path)

    tile_lookup = tile.set_index("tile_id")
    t_paths = {}
    for tile_id in sorted(set(pairs["tile_id"].astype(str))):
        raw = tile_lookup.loc[tile_id, "tile_path"]
        t_paths[tile_id] = resolve_existing_file(
            raw, extra_roots=[tile_index_path.parent]
        )
    return q_paths, t_paths


def generate_geometry(
    inputs: dict[str, Any],
    q_paths: dict[int, Path],
    t_paths: dict[str, Path],
    progress_every: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    manifest = inputs["manifest"].set_index("query_id")
    pairs = inputs["pairs"]
    tile_lookup = inputs["tile"].set_index("tile_id")
    detector = orb_subtile.create_detector()
    q_cache: dict[int, Any] = {}
    t_cache: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    start = time.perf_counter()

    def qfeat(qid: int):
        if qid not in q_cache:
            q_cache[qid] = orb_subtile.compute_features(q_paths[qid], detector)
        return q_cache[qid]

    def tfeat(tile_id: str):
        if tile_id not in t_cache:
            t_cache[tile_id] = orb_subtile.compute_features(t_paths[tile_id], detector)
        return t_cache[tile_id]

    for idx, row in enumerate(pairs.itertuples(index=False), start=1):
        if idx == 1 or idx % max(1, progress_every) == 0 or idx == len(pairs):
            print(f"{inputs['role']} geometry pair {idx}/{len(pairs)}", flush=True)
        qid = int(row.query_id)
        tile_id = str(row.tile_id)
        qf = qfeat(qid)
        tf = tfeat(tile_id)
        result = orb_subtile.verify_pair_with_geometry(qf, tf)

        projected_e = math.nan
        projected_n = math.nan
        inside = False
        if bool(result["homography_ok"]):
            u = float(result["projected_tile_u_px"])
            v = float(result["projected_tile_v_px"])
            tile = tile_lookup.loc[tile_id]
            projected_e, projected_n = orb_subtile.tile_pixel_to_map(
                u, v, tf.image_shape,
                float(tile["left_easting"]), float(tile["right_easting"]),
                float(tile["bottom_northing"]), float(tile["top_northing"]),
            )
            th, tw = tf.image_shape
            inside = bool(0.0 <= u < float(tw) and 0.0 <= v < float(th))

        rows.append({
            "trajectory_id": inputs["trajectory_id"],
            "role": inputs["role"],
            "query_id": qid,
            "anchor_tile_id": str(row.anchor_tile_id),
            "tile_id": tile_id,
            "union_rank": int(row.union_rank),
            "immediate_rank": "" if pd.isna(row.immediate_rank) else int(row.immediate_rank),
            "overlap_rank": "" if pd.isna(row.overlap_rank) else int(row.overlap_rank),
            "is_anchor": bool(row.is_anchor),
            "in_immediate": bool(row.in_immediate),
            "center_distance_m": float(row.center_distance_m),
            "delta_easting_m": float(row.delta_easting_m),
            "delta_northing_m": float(row.delta_northing_m),
            "good_matches": int(result["good_matches"]),
            "inliers": int(result["inliers"]),
            "inlier_ratio": float(result["inlier_ratio"]),
            "query_inlier_coverage": float(result["query_inlier_coverage"]),
            "tile_inlier_coverage": float(result["sat_inlier_coverage"]),
            "verifier_score": float(result["verifier_score"]),
            "homography_ok": bool(result["homography_ok"]),
            "query_processed_height_px": int(qf.image_shape[0]),
            "query_processed_width_px": int(qf.image_shape[1]),
            "tile_processed_height_px": int(tf.image_shape[0]),
            "tile_processed_width_px": int(tf.image_shape[1]),
            "query_center_u_px": float(result["query_center_u_px"]),
            "query_center_v_px": float(result["query_center_v_px"]),
            "projected_tile_u_px": float(result["projected_tile_u_px"]),
            "projected_tile_v_px": float(result["projected_tile_v_px"]),
            "projected_easting": float(projected_e),
            "projected_northing": float(projected_n),
            "finite_projection": bool(
                bool(result["homography_ok"])
                and math.isfinite(float(projected_e))
                and math.isfinite(float(projected_n))
            ),
            "projected_inside_tile": inside,
            "projective_denominator": float(result["projective_denominator"]),
            "inlier_reprojection_rmse_px": float(result["inlier_reprojection_rmse_px"]),
            "geometry_error": result["error"] or "",
        })

    elapsed = time.perf_counter() - start
    frame = pd.DataFrame(rows)
    summaries: list[dict[str, Any]] = []
    for qid, group in frame.groupby("query_id", sort=True):
        hom = group["homography_ok"].astype(bool)
        finite = group["finite_projection"].astype(bool)
        inside = group["projected_inside_tile"].astype(bool)
        immediate = group["in_immediate"].astype(bool)
        anchor = group["is_anchor"].astype(bool)
        summaries.append({
            "trajectory_id": inputs["trajectory_id"],
            "role": inputs["role"],
            "query_id": int(qid),
            "tested_pairs": int(len(group)),
            "immediate_pairs": int(immediate.sum()),
            "homography_ok_count": int(hom.sum()),
            "finite_projection_count": int(finite.sum()),
            "inside_projection_count": int((finite & inside).sum()),
            "immediate_homography_ok_count": int((hom & immediate).sum()),
            "immediate_finite_projection_count": int((finite & immediate).sum()),
            "immediate_inside_projection_count": int((finite & inside & immediate).sum()),
            "anchor_homography_ok": bool((hom & anchor).any()),
            "anchor_inside_projection": bool((hom & inside & anchor).any()),
        })
    stats = {
        "pair_rows": int(len(frame)),
        "query_count": int(frame["query_id"].nunique()),
        "unique_tiles_processed": int(frame["tile_id"].nunique()),
        "homography_ok_pairs": int(frame["homography_ok"].astype(bool).sum()),
        "finite_projection_pairs": int(frame["finite_projection"].astype(bool).sum()),
        "inside_projection_pairs": int((
            frame["finite_projection"].astype(bool)
            & frame["projected_inside_tile"].astype(bool)
        ).sum()),
        "queries_with_any_homography": int(sum(r["homography_ok_count"] > 0 for r in summaries)),
        "queries_with_any_finite_projection": int(sum(r["finite_projection_count"] > 0 for r in summaries)),
        "queries_with_any_inside_projection": int(sum(r["inside_projection_count"] > 0 for r in summaries)),
        "queries_with_immediate_homography": int(sum(r["immediate_homography_ok_count"] > 0 for r in summaries)),
        "queries_with_anchor_homography": int(sum(r["anchor_homography_ok"] for r in summaries)),
        "homography_count_histogram": dict(sorted(
            Counter(str(r["homography_ok_count"]) for r in summaries).items(),
            key=lambda kv: int(kv[0]),
        )),
        "runtime_observed_s": float(elapsed),
    }
    return rows, summaries, stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RG2.2 reference-free multi-tile sub-tile projection generation"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()

    cfg_path = resolve(args.config)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    verify_config(cfg)
    rg21a, topology_sha = verify_rg2_lineage(cfg)

    prepared = {}
    map_index_sha = None
    for key in ("development", "blind_stress"):
        inputs = load_inputs(cfg, rg21a, key)
        rg20 = json.loads(resolve(cfg["rg2_0"]["report"]).read_text(encoding="utf-8"))
        this_map_sha = sha256_file(inputs["tile_index_path"])
        if this_map_sha != rg20["provenance"]["tile_index_sha256"]:
            raise RuntimeError(f"{key} map index differs from RG2.0 frozen map")
        if map_index_sha is None:
            map_index_sha = this_map_sha
        elif map_index_sha != this_map_sha:
            raise RuntimeError("Development and blind-stress map indices differ")
        q_paths, t_paths = preflight_paths(inputs)
        prepared[key] = (inputs, q_paths, t_paths)
        print(
            f"{key} preflight: {len(inputs['manifest'])} queries, "
            f"{len(inputs['pairs'])} frozen pairs, "
            f"{len(t_paths)} unique tile images; reference loaded=False"
        )

    all_files: dict[Path, bytes] = {}
    trajectory_reports = {}
    runtime_only = {}
    progress = int(cfg["generation"]["progress_every_pairs"])
    for key in ("development", "blind_stress"):
        inputs, q_paths, t_paths = prepared[key]
        rows, summaries, stats = generate_geometry(inputs, q_paths, t_paths, progress)
        # Runtime is observational and deliberately excluded from frozen report bytes.
        runtime_only[key] = stats.pop("runtime_observed_s")
        root = resolve(cfg["output_root"]) / key
        pair_path = root / "rg2_2_multi_tile_projection_evidence.csv"
        summary_path = root / "rg2_2_query_projection_summary.csv"
        pair_bytes = canonical_csv(rows)
        summary_bytes = canonical_csv(summaries)
        all_files[pair_path] = pair_bytes
        all_files[summary_path] = summary_bytes
        trajectory_reports[key] = {
            **stats,
            "role": inputs["role"],
            "trajectory_id": inputs["trajectory_id"],
            "provenance": {
                "trajectory_spec_sha256": sha256_file(inputs["spec_path"]),
                "blind_manifest_sha256": sha256_file(inputs["manifest_path"]),
                "rg2_1a_local_neighborhoods_sha256": sha256_file(inputs["local_path"]),
                "rg2_1a_query_summaries_sha256": sha256_file(inputs["summary_path"]),
                "map_index_sha256": sha256_file(inputs["tile_index_path"]),
            },
            "outputs": {
                "pair_evidence": {
                    "path": pair_path.relative_to(resolve(cfg["output_root"])).as_posix(),
                    "sha256": sha256_bytes(pair_bytes),
                    "rows": len(rows),
                },
                "query_summary": {
                    "path": summary_path.relative_to(resolve(cfg["output_root"])).as_posix(),
                    "sha256": sha256_bytes(summary_bytes),
                    "rows": len(summaries),
                },
            },
        }

    report = {
        "stage": "RG2.2",
        "status": "PASS_RG2_2_REFERENCE_FREE_MULTI_TILE_PROJECTION_FREEZE",
        "map_variant": "768_s256",
        "map_crs": "EPSG:3346",
        "geometry_contract": orb_subtile.contract(),
        "topology_sha256": topology_sha,
        "trajectories": trajectory_reports,
        "scientific_boundary": {
            "reference_loaded": False,
            "srt_gps_loaded": False,
            "oracle_labels_loaded": False,
            "retrieval_reranked": False,
            "candidate_selection_executed": False,
            "inlier_gate_executed": False,
            "consensus_executed": False,
            "bootstrap_state_modified": False,
        },
        "interpretation": {
            "raw_geometry_evidence_only": True,
            "failed_pairs_retained": True,
            "homography_ok_is_historical_minimum_geometry_validity_not_confidence_gate": True,
            "no_accuracy_claim": True,
        },
        "provenance": {
            "config_sha256": sha256_file(cfg_path),
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "geometry_module_sha256": sha256_file(Path(orb_subtile.__file__).resolve()),
            "rg2_1a_report_sha256": sha256_file(resolve(cfg["rg2_1a"]["report"])),
            "rg2_0_report_sha256": sha256_file(resolve(cfg["rg2_0"]["report"])),
            "map_index_sha256": map_index_sha,
        },
    }
    report_path = resolve(cfg["output_root"]) / "rg2_2_freeze_report.json"
    all_files[report_path] = canonical_json(report)
    freeze_files(all_files)

    print("Status:", report["status"])
    print("Geometry contract:", json.dumps(orb_subtile.contract(), sort_keys=True))
    for key, tr in trajectory_reports.items():
        print(
            f"{key}: pairs={tr['pair_rows']} queries={tr['query_count']} "
            f"unique_tiles={tr['unique_tiles_processed']} "
            f"homography_ok={tr['homography_ok_pairs']} "
            f"finite_projection={tr['finite_projection_pairs']} "
            f"inside={tr['inside_projection_pairs']} "
            f"queries_any_h={tr['queries_with_any_homography']} "
            f"queries_any_finite={tr['queries_with_any_finite_projection']} "
            f"queries_immediate_h={tr['queries_with_immediate_homography']} "
            f"queries_anchor_h={tr['queries_with_anchor_homography']}"
        )
        print("  homography-count hist:", tr["homography_count_histogram"])
        print("  pair evidence SHA256:", tr["outputs"]["pair_evidence"]["sha256"])
        print("  query summary SHA256:", tr["outputs"]["query_summary"]["sha256"])
        print(f"  observed runtime s (not frozen): {runtime_only[key]:.3f}")
    print("Reference loaded: False")
    print("Consensus executed: False")
    print("Accuracy claim: False")
    print("Freeze report:", report_path)


if __name__ == "__main__":
    main()
