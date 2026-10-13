#!/usr/bin/env python3
"""CV1.1: freeze reference-free ORB evidence under fixed scale/rotation transforms."""
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
    rotate_processed_query,
    scale_processed_query,
)
from uavloc.geometry.top20_geometric_evidence import analyze_pair

ROOT = REPO_ROOT
DEFAULT_CONFIG = ROOT / "configs/research/crossview_scale_rotation_domain_cv1.yaml"

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
PARITY_EXACT = (
    "good_matches", "inliers", "homography_ok",
)
PARITY_FLOAT = (
    "inlier_ratio",
    "projected_tile_u_px", "projected_tile_v_px",
    "query_bbox_coverage", "tile_bbox_coverage",
    "query_hull_coverage", "tile_hull_coverage",
    "orientation_delta_mean_deg", "orientation_resultant", "orientation_mad_deg",
    "log_scale_ratio_median", "log_scale_ratio_mad",
    "jacobian_det", "jacobian_s1", "jacobian_s2",
    "jacobian_anisotropy", "jacobian_rotation_deg",
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
        raise RuntimeError("Cannot freeze empty CV1.1 evidence")
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
        raise RuntimeError("Refusing to overwrite differing CV1.1 freeze: " + ", ".join(changed))
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
    if cfg.get("stage") != "CV1.0":
        raise RuntimeError("Expected frozen CV1.0 protocol config")
    if cfg["map"]["variant"] != "768_s256" or cfg["map"]["crs"] != "EPSG:3346":
        raise RuntimeError("CV1 map contract changed")
    if cfg["lineage"]["frozen_retrieval"] != "qv1_4_center_unique_allview_fill20":
        raise RuntimeError("CV1 frozen retrieval changed")
    if cfg["geometry_contract"]["selector"] != "raw_inliers":
        raise RuntimeError("CV1 selector changed")
    for key, value in cfg["scientific_boundary"].items():
        if value is not False:
            raise RuntimeError(f"Scientific boundary {key} must remain false in CV1.1")


def condition_table(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    scales = {str(x["id"]): x for x in cfg["transforms"]["scale"]["conditions"]}
    rots = {str(x["id"]): x for x in cfg["transforms"]["rotation"]["conditions"]}
    rows: list[dict[str, Any]] = []
    for condition_id in cfg["cv1_1"]["unique_conditions"]:
        cid = str(condition_id)
        if cid == "baseline":
            rows.append({
                "transform_family": "baseline",
                "transform_id": "baseline",
                "scale_factor": 1.0,
                "rotation_deg": 0,
            })
        elif cid in scales:
            rows.append({
                "transform_family": "scale",
                "transform_id": cid,
                "scale_factor": float(scales[cid]["factor"]),
                "rotation_deg": 0,
            })
        elif cid in rots:
            rows.append({
                "transform_family": "rotation",
                "transform_id": cid,
                "scale_factor": 1.0,
                "rotation_deg": int(rots[cid]["degrees"]),
            })
        else:
            raise RuntimeError(f"Unknown CV1.1 transform ID: {cid}")
    return rows


def load_inputs(cfg: dict[str, Any]) -> dict[str, Any]:
    tc = cfg["trajectory"]
    spec_path = resolve(tc["spec"])
    manifest_path = resolve(tc["manifest"])
    pool_path = resolve(tc["frozen_top20"])
    gd1_path = resolve(tc["gd1_0_evidence"])
    for path in (spec_path, manifest_path, pool_path, gd1_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    if sha256_file(pool_path) != cfg["lineage"]["expected_top20_sha256"]:
        raise RuntimeError("Frozen QV1.4 development Top20 SHA256 changed")
    if sha256_file(gd1_path) != cfg["lineage"]["expected_gd1_0_development_sha256"]:
        raise RuntimeError("Frozen GD1.0 development evidence SHA256 changed")

    spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    if spec["trajectory"]["id"] != "villoc_traj01_90deg_stable120m":
        raise RuntimeError("Development trajectory changed")
    if spec["map"]["crs"] != "EPSG:3346" or "768_s256" not in spec["map"]["variants"]:
        raise RuntimeError("Trajectory map contract changed")

    tile_index_path = resolve(spec["map"]["variants"]["768_s256"]["tile_index"])
    manifest = pd.read_csv(manifest_path).copy()
    pool = pd.read_csv(pool_path).copy()
    tile = pd.read_csv(tile_index_path).copy()
    gd1 = pd.read_csv(gd1_path).copy()

    for frame, label in (
        (manifest, "CV1 development manifest"),
        (pool, "CV1 frozen Top20"),
        (tile, "CV1 tile index"),
    ):
        assert_blind_safe_columns(frame, label)

    require_columns(manifest, MANIFEST_REQUIRED, "CV1 development manifest")
    require_columns(pool, POOL_REQUIRED, "CV1 frozen Top20")
    require_columns(tile, TILE_REQUIRED, "CV1 tile index")

    manifest["query_id"] = pd.to_numeric(manifest["query_id"], errors="raise").astype(int)
    pool["query_id"] = pd.to_numeric(pool["query_id"], errors="raise").astype(int)
    pool["pool_rank"] = pd.to_numeric(pool["pool_rank"], errors="raise").astype(int)
    pool["source_rank"] = pd.to_numeric(pool["source_rank"], errors="raise").astype(int)
    pool["tile_id"] = pool["tile_id"].astype(str)
    tile["tile_id"] = tile["tile_id"].astype(str)
    gd1["query_id"] = pd.to_numeric(gd1["query_id"], errors="raise").astype(int)
    gd1["pool_rank"] = pd.to_numeric(gd1["pool_rank"], errors="raise").astype(int)
    gd1["tile_id"] = gd1["tile_id"].astype(str)

    refs = manifest["reference_available"].astype(str).str.strip().str.lower()
    if not refs.isin(["false", "0", "no", "n", "f"]).all():
        raise RuntimeError("CV1 manifest exposes reference")

    if len(manifest) != int(tc["expected_queries"]):
        raise RuntimeError("Development query count changed")
    if len(pool) != int(tc["expected_pairs"]):
        raise RuntimeError("Development Top20 pair count changed")
    if pool[["query_id", "tile_id"]].duplicated().any():
        raise RuntimeError("Duplicate query/tile pair in frozen Top20")

    query_ids = [int(x) for x in cfg["screen"]["query_ids"]]
    if len(query_ids) != int(cfg["screen"]["expected_queries"]) or len(set(query_ids)) != len(query_ids):
        raise RuntimeError("CV1 uniform40 query list count/uniqueness mismatch")
    if not set(query_ids).issubset(set(manifest["query_id"])):
        raise RuntimeError("CV1 uniform40 contains missing query IDs")

    pool = pool[pool["query_id"].isin(query_ids)].copy()
    if len(pool) != int(cfg["screen"]["expected_pairs_per_condition"]):
        raise RuntimeError("CV1 uniform40 Top20 pair count changed")

    for qid, group in pool.groupby("query_id", sort=True):
        ordered = group.sort_values("pool_rank")
        if len(ordered) != 20 or ordered["pool_rank"].tolist() != list(range(1, 21)):
            raise RuntimeError(f"CV1 q{qid} Top20 rank contract changed")

    return {
        "spec_path": spec_path,
        "manifest_path": manifest_path,
        "pool_path": pool_path,
        "gd1_path": gd1_path,
        "tile_index_path": tile_index_path,
        "manifest": manifest[manifest["query_id"].isin(query_ids)].copy(),
        "pool": pool.sort_values(["query_id", "pool_rank"], kind="mergesort").reset_index(drop=True),
        "tile": tile,
        "gd1": gd1[gd1["query_id"].isin(query_ids)].copy(),
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


def transformed_query_features(
    path: Path,
    condition: dict[str, Any],
    detector,
):
    if condition["transform_family"] == "baseline":
        return orb_subtile.compute_features(path, detector)

    gray = read_frozen_processed_query(path)
    if condition["transform_family"] == "scale":
        gray = scale_processed_query(gray, float(condition["scale_factor"]))
    elif condition["transform_family"] == "rotation":
        gray = rotate_processed_query(gray, int(condition["rotation_deg"]))
    else:
        raise RuntimeError(f"Unsupported transform family: {condition['transform_family']}")
    return feature_pack_from_gray(gray, detector)


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
    conditions = condition_table(cfg)
    tile_cache: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    progress_every = int(cfg["cv1_1"]["progress_every_pairs"])
    total = len(inputs["pool"]) * len(conditions)
    processed = 0

    def tfeat(tile_id: str):
        if tile_id not in tile_cache:
            tile_cache[tile_id] = orb_subtile.compute_features(t_paths[tile_id], detector)
        return tile_cache[tile_id]

    started = time.perf_counter()
    for condition in conditions:
        q_cache: dict[int, Any] = {}
        for row in inputs["pool"].itertuples(index=False):
            processed += 1
            if processed == 1 or processed % max(1, progress_every) == 0 or processed == total:
                print(
                    f"CV1.1 pair {processed}/{total} condition={condition['transform_id']}",
                    flush=True,
                )

            qid = int(row.query_id)
            tile_id = str(row.tile_id)
            if qid not in q_cache:
                q_cache[qid] = transformed_query_features(
                    q_paths[qid], condition, detector
                )
            q = q_cache[qid]
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
                "transform_family": str(condition["transform_family"]),
                "transform_id": str(condition["transform_id"]),
                "scale_factor": float(condition["scale_factor"]),
                "rotation_deg": int(condition["rotation_deg"]),
            }
            out.update(evidence)
            append_projection(out, evidence, q, t, tile_lookup.loc[tile_id])
            rows.append(out)

    runtime = float(time.perf_counter() - started)
    frame = pd.DataFrame(rows)
    stats = {
        "queries": int(frame["query_id"].nunique()),
        "conditions": int(frame["transform_id"].nunique()),
        "candidate_pairs": int(len(frame)),
        "unique_tiles": int(frame["tile_id"].nunique()),
        "homography_ok_pairs": int(frame["homography_ok"].astype(bool).sum()),
        "finite_projection_pairs": int(frame["finite_projection"].astype(bool).sum()),
        "inside_projection_pairs": int(frame["projected_inside_tile"].astype(bool).sum()),
    }
    return rows, stats, runtime


def values_equal(a: Any, b: Any, *, atol: float = 1e-7) -> bool:
    if pd.isna(a) and pd.isna(b):
        return True
    try:
        af = float(a)
        bf = float(b)
    except Exception:
        return a == b
    if math.isnan(af) and math.isnan(bf):
        return True
    if math.isinf(af) or math.isinf(bf):
        return af == bf
    return math.isclose(af, bf, rel_tol=0.0, abs_tol=atol)


def verify_baseline_parity(
    generated_rows: list[dict[str, Any]],
    gd1: pd.DataFrame,
) -> dict[str, Any]:
    baseline = pd.DataFrame(
        [r for r in generated_rows if r["transform_id"] == "baseline"]
    ).copy()
    expected = gd1.copy()

    key = ["query_id", "pool_rank", "tile_id"]
    merged = baseline.merge(
        expected,
        on=key,
        how="outer",
        suffixes=("_cv1", "_gd1"),
        indicator=True,
    )
    if not (merged["_merge"] == "both").all():
        raise RuntimeError("CV1.1 baseline/GD1.0 row identities differ")

    mismatches: list[dict[str, Any]] = []
    for row in merged.itertuples(index=False):
        rd = row._asdict()
        for field in PARITY_EXACT:
            a = rd[f"{field}_cv1"]
            b = rd[f"{field}_gd1"]
            if field == "homography_ok":
                equal = str(a).strip().lower() == str(b).strip().lower()
            else:
                equal = values_equal(a, b, atol=0.0)
            if not equal:
                mismatches.append({
                    "query_id": int(rd["query_id"]),
                    "pool_rank": int(rd["pool_rank"]),
                    "tile_id": str(rd["tile_id"]),
                    "field": field,
                    "cv1": a,
                    "gd1": b,
                })
        for field in PARITY_FLOAT:
            a = rd[f"{field}_cv1"]
            b = rd[f"{field}_gd1"]
            if not values_equal(a, b, atol=1e-6):
                mismatches.append({
                    "query_id": int(rd["query_id"]),
                    "pool_rank": int(rd["pool_rank"]),
                    "tile_id": str(rd["tile_id"]),
                    "field": field,
                    "cv1": a,
                    "gd1": b,
                })

    if mismatches:
        sample = mismatches[:10]
        raise RuntimeError(
            "CV1.1 baseline parity failed against frozen GD1.0; sample="
            + json.dumps(sample, default=str)
        )

    return {
        "status": "PASS_CV1_1_BASELINE_PARITY",
        "rows_compared": int(len(merged)),
        "exact_fields": list(PARITY_EXACT),
        "float_fields": list(PARITY_FLOAT),
        "float_atol": 1e-6,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CV1.1 reference-free fixed scale/rotation ORB evidence freeze"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()

    cfg_path = resolve(args.config)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    verify_config(cfg)
    inputs = load_inputs(cfg)
    q_paths, t_paths = preflight_paths(inputs)

    conditions = condition_table(cfg)
    expected_total = int(cfg["cv1_1"]["expected_total_pairs"])
    actual_total = len(inputs["pool"]) * len(conditions)
    if actual_total != expected_total:
        raise RuntimeError(
            f"CV1.1 total pair contract changed: expected {expected_total}, got {actual_total}"
        )

    print(
        f"CV1.1 preflight: queries={len(inputs['manifest'])} "
        f"pairs/condition={len(inputs['pool'])} conditions={len(conditions)} "
        f"total_pairs={actual_total} unique_tiles={len(t_paths)}"
    )
    print("Reference loaded: False")
    print("SRT loaded: False")

    rows, stats, runtime = generate(cfg, inputs, q_paths, t_paths)
    parity = verify_baseline_parity(rows, inputs["gd1"])

    output_root = resolve(cfg["output_root"]) / "cv1_1"
    evidence_path = output_root / "cv1_1_uniform40_reference_free_evidence.csv"
    evidence_payload = canonical_csv(rows)

    report = {
        "stage": "CV1.1",
        "status": "PASS_CV1_1_REFERENCE_FREE_TRANSFORM_EVIDENCE_FREEZE",
        "screen": "uniform40",
        "transform_conditions": conditions,
        "stats": stats,
        "baseline_parity": parity,
        "physical_gsd_status": cfg["physical_gsd"]["status"],
        "scientific_boundary": {
            "reference_loaded": False,
            "srt_loaded": False,
            "oracle_labels_loaded": False,
            "best_transform_per_query_executed": False,
            "best_transform_per_candidate_executed": False,
            "candidate_selected": False,
            "reranking_executed": False,
            "threshold_tuned": False,
            "retrieval_modified": False,
            "state_modified": False,
            "temporal_fusion_executed": False,
        },
        "interpretation": {
            "measurement_stage_only": True,
            "no_accuracy_claim": True,
            "fixed_conditions_only": True,
            "physical_gsd_not_claimed": True,
        },
        "provenance": {
            "config_sha256": sha256_file(cfg_path),
            "trajectory_spec_sha256": sha256_file(inputs["spec_path"]),
            "blind_manifest_sha256": sha256_file(inputs["manifest_path"]),
            "frozen_top20_sha256": sha256_file(inputs["pool_path"]),
            "gd1_0_development_sha256": sha256_file(inputs["gd1_path"]),
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

    report_path = output_root / "cv1_1_freeze_report.json"
    files = {
        evidence_path: evidence_payload,
        report_path: canonical_json(report),
    }
    freeze_files(files)

    print("Status:", report["status"])
    print("Baseline parity:", parity["status"])
    print("Evidence SHA256:", report["provenance"]["evidence_output_sha256"])
    print(f"Observed runtime s (not frozen): {runtime:.3f}")
    print("Reference loaded: False")
    print("SRT loaded: False")
    print("Candidate selected/reranked: False")
    print("Accuracy claim: False")
    print("Freeze report:", report_path)


if __name__ == "__main__":
    main()
