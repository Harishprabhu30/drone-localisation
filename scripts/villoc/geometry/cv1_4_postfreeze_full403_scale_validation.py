#!/usr/bin/env python3
"""CV1.4: post-freeze full403 validation of frozen scale_m2 vs frozen ORB baseline."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import numpy as np
import pandas as pd
import yaml

from uavloc.evaluation.reference import load_reference_xy

ROOT = REPO_ROOT
DEFAULT_CONFIG = ROOT / "configs/research/crossview_scale_rotation_domain_cv1_4.yaml"

BASE_REQUIRED = {
    "query_id", "pool_rank", "tile_id", "inliers",
    "homography_ok", "finite_projection",
    "projected_easting", "projected_northing",
}
TILE_REQUIRED = {
    "tile_id", "left_easting", "right_easting",
    "bottom_northing", "top_northing",
    "center_easting", "center_northing",
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


def deterministic_csv(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, lineterminator="\n", float_format="%.9f").encode("utf-8")


def deterministic_json(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def write_immutable(files: dict[Path, bytes]) -> None:
    changed = [str(path) for path, data in files.items() if path.exists() and path.read_bytes() != data]
    if changed:
        raise RuntimeError("Refusing to overwrite differing CV1.4 outputs: " + ", ".join(changed))
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


def bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    mapping = {
        "true": True, "false": False, "1": True, "0": False,
        "yes": True, "no": False, "y": True, "n": False,
        "t": True, "f": False,
    }
    out = series.astype(str).str.strip().str.lower().map(mapping)
    if out.isna().any():
        bad = series[out.isna()].astype(str).unique().tolist()
        raise RuntimeError(f"Could not parse booleans: {bad[:10]}")
    return out.astype(bool)


def verify_config(cfg: dict[str, Any]) -> None:
    if cfg.get("stage") != "CV1.4":
        raise RuntimeError("Expected CV1.4 config")
    if cfg["map"]["variant"] != "768_s256" or cfg["map"]["crs"] != "EPSG:3346":
        raise RuntimeError("CV1.4 map contract changed")
    if cfg["ranking"]["selector"] != "inliers":
        raise RuntimeError("CV1.4 selector changed")
    if cfg["ranking"]["tie_break"] != ["pool_rank_asc", "tile_id_asc"]:
        raise RuntimeError("CV1.4 tie-break changed")
    boundary = cfg["scientific_boundary"]
    if boundary.get("development_reference_allowed") is not True:
        raise RuntimeError("Development reference must be explicitly allowed")
    if boundary.get("reference_after_both_freezes_only") is not True:
        raise RuntimeError("Reference-after-freeze contract missing")
    for key in (
        "baseline_evidence_mutable", "scale_m2_evidence_mutable",
        "transform_retuning_allowed", "additional_scale_conditions_allowed",
        "best_transform_per_query_allowed", "selector_formula_mutable",
        "threshold_tuning_enabled", "srt_loaded", "blind_stress_loaded",
        "state_enabled", "temporal_fusion_enabled",
    ):
        if boundary.get(key) is not False:
            raise RuntimeError(f"{key} must be false")


def verify_inputs(cfg: dict[str, Any]) -> tuple[Path, Path]:
    b = cfg["baseline"]
    s = cfg["scale_m2"]
    baseline_path = resolve(b["evidence"])
    scale_path = resolve(s["evidence"])
    report_path = resolve(s["report"])

    if sha256_file(baseline_path) != b["expected_evidence_sha256"]:
        raise RuntimeError("Frozen GD1.0 baseline bytes changed")
    if sha256_file(scale_path) != s["expected_evidence_sha256"]:
        raise RuntimeError("Frozen CV1.3 scale_m2 bytes changed")

    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != s["expected_status"]:
        raise RuntimeError("CV1.3 report not PASS")
    if report.get("provenance", {}).get("evidence_output_sha256") != s["expected_evidence_sha256"]:
        raise RuntimeError("CV1.3 report evidence hash mismatch")
    if report.get("scientific_boundary", {}).get("reference_loaded") is not False:
        raise RuntimeError("CV1.3 reference boundary violated")
    if report.get("scientific_boundary", {}).get("srt_loaded") is not False:
        raise RuntimeError("CV1.3 SRT boundary violated")
    if not math.isclose(
        float(report.get("promoted_condition", {}).get("scale_factor")),
        float(s["expected_scale_factor"]),
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise RuntimeError("CV1.3 scale factor mismatch")
    return baseline_path, scale_path


def verify_tile_geometry(cfg: dict[str, Any]) -> Path:
    report_path = resolve(cfg["map"]["rg2_0_report"])
    tile_path = resolve(cfg["map"]["tile_geometry"])
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "PASS_RG2_0_MAP_TOPOLOGY_PREFLIGHT":
        raise RuntimeError("RG2.0 topology report not PASS")
    if sha256_file(tile_path) != report["provenance"]["tile_geometry_sha256"]:
        raise RuntimeError("RG2.0 tile geometry changed")
    return tile_path


def attach_reference(
    evidence: pd.DataFrame,
    tile: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    condition: str,
) -> pd.DataFrame:
    require_columns(evidence, BASE_REQUIRED, condition)
    require_columns(tile, TILE_REQUIRED, "tile geometry")

    frame = evidence.copy()
    frame["query_id"] = pd.to_numeric(frame["query_id"], errors="raise").astype(int)
    frame["pool_rank"] = pd.to_numeric(frame["pool_rank"], errors="raise").astype(int)
    frame["tile_id"] = frame["tile_id"].astype(str)
    frame["homography_ok"] = bool_series(frame["homography_ok"])
    frame["finite_projection"] = bool_series(frame["finite_projection"])
    frame["condition"] = condition

    tiles = tile[list(TILE_REQUIRED)].copy()
    tiles["tile_id"] = tiles["tile_id"].astype(str)
    if tiles["tile_id"].duplicated().any():
        raise RuntimeError("Duplicate tile IDs")

    refs = reference[["gt_x", "gt_y"]].reset_index()
    out = (
        frame.merge(tiles, on="tile_id", how="left", validate="many_to_one")
        .merge(refs, on="query_id", how="left", validate="many_to_one")
    )
    needed = [
        "left_easting", "right_easting", "bottom_northing", "top_northing",
        "center_easting", "center_northing", "gt_x", "gt_y",
    ]
    if out[needed].isna().any().any():
        raise RuntimeError(f"{condition} reference/tile join produced missing values")

    out["contains_query"] = (
        (out["gt_x"] >= out["left_easting"])
        & (out["gt_x"] <= out["right_easting"])
        & (out["gt_y"] >= out["bottom_northing"])
        & (out["gt_y"] <= out["top_northing"])
    )
    out["center_error_m"] = np.hypot(
        out["center_easting"] - out["gt_x"],
        out["center_northing"] - out["gt_y"],
    )
    out["center_le40"] = out["center_error_m"] <= 40.0
    out["center_le80"] = out["center_error_m"] <= 80.0

    px = pd.to_numeric(out["projected_easting"], errors="coerce")
    py = pd.to_numeric(out["projected_northing"], errors="coerce")
    finite = np.isfinite(px) & np.isfinite(py)
    out["projection_error_m"] = np.nan
    out.loc[finite, "projection_error_m"] = np.hypot(
        px[finite] - out.loc[finite, "gt_x"],
        py[finite] - out.loc[finite, "gt_y"],
    )
    return out


def rank_candidates(group: pd.DataFrame) -> pd.DataFrame:
    work = group.copy()
    work["_inliers"] = pd.to_numeric(work["inliers"], errors="coerce").fillna(-1.0)
    return work.sort_values(
        ["_inliers", "pool_rank", "tile_id"],
        ascending=[False, True, True],
        kind="mergesort",
    )


def projection_summary(values: pd.Series) -> dict[str, Any]:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    if not len(arr):
        return {"count": 0, "median_m": None, "rmse_m": None, "p95_m": None, "le40": 0, "le80": 0, "gt200": 0}
    return {
        "count": int(len(arr)),
        "median_m": float(np.median(arr)),
        "rmse_m": float(np.sqrt(np.mean(arr * arr))),
        "p95_m": float(np.quantile(arr, 0.95)),
        "le40": int((arr <= 40.0).sum()),
        "le80": int((arr <= 80.0).sum()),
        "gt200": int((arr > 200.0).sum()),
    }


def oracle_projection_summary(frame: pd.DataFrame) -> dict[str, Any]:
    best = (
        frame.replace([np.inf, -np.inf], np.nan)
        .dropna(subset=["projection_error_m"])
        .groupby("query_id")["projection_error_m"]
        .min()
    )
    return projection_summary(best)


def summarize_condition(frame: pd.DataFrame, topks: list[int]) -> tuple[dict[str, Any], pd.DataFrame]:
    label_hits = {
        label: {f"top{k}": 0 for k in topks}
        for label in ("contains_query", "center_le40", "center_le80")
    }
    selected_rows: list[dict[str, Any]] = []

    for qid, group in frame.groupby("query_id", sort=True):
        ordered = rank_candidates(group).reset_index(drop=True)
        selected = ordered.iloc[0]
        for label in label_hits:
            for k in topks:
                label_hits[label][f"top{k}"] += int(
                    ordered.head(k)[label].astype(bool).any()
                )
        selected_rows.append({
            "condition": str(selected["condition"]),
            "query_id": int(qid),
            "pool_rank": int(selected["pool_rank"]),
            "tile_id": str(selected["tile_id"]),
            "inliers": float(selected["inliers"]),
            "contains_query": bool(selected["contains_query"]),
            "center_le40": bool(selected["center_le40"]),
            "center_le80": bool(selected["center_le80"]),
            "projection_error_m": float(selected["projection_error_m"]) if pd.notna(selected["projection_error_m"]) else np.nan,
            "top1_top2_inlier_margin": (
                float(pd.to_numeric(ordered.iloc[0]["inliers"], errors="coerce"))
                - float(pd.to_numeric(ordered.iloc[1]["inliers"], errors="coerce"))
                if len(ordered) >= 2 else np.nan
            ),
        })

    selected = pd.DataFrame(selected_rows)
    return {
        "queries": int(len(selected)),
        "ranking": label_hits,
        "selected_top1": {
            "contains_query": int(selected["contains_query"].sum()),
            "center_le40": int(selected["center_le40"].sum()),
            "center_le80": int(selected["center_le80"].sum()),
        },
        "selected_projection": projection_summary(selected["projection_error_m"]),
        "oracle_best_finite_projection": oracle_projection_summary(frame),
        "median_top1_top2_inlier_margin": float(
            pd.to_numeric(selected["top1_top2_inlier_margin"], errors="coerce").median()
        ),
    }, selected


def compare_selected(baseline: pd.DataFrame, scale: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    b = baseline.add_prefix("baseline_").rename(columns={"baseline_query_id": "query_id"})
    s = scale.add_prefix("scale_m2_").rename(columns={"scale_m2_query_id": "query_id"})
    merged = b.merge(s, on="query_id", how="inner", validate="one_to_one")

    merged["winner_changed"] = merged["baseline_tile_id"] != merged["scale_m2_tile_id"]
    merged["containment_rescue"] = (
        ~merged["baseline_contains_query"] & merged["scale_m2_contains_query"]
    )
    merged["containment_damage"] = (
        merged["baseline_contains_query"] & ~merged["scale_m2_contains_query"]
    )
    merged["le80_rescue"] = (
        ~merged["baseline_center_le80"] & merged["scale_m2_center_le80"]
    )
    merged["le80_damage"] = (
        merged["baseline_center_le80"] & ~merged["scale_m2_center_le80"]
    )

    b_err = pd.to_numeric(merged["baseline_projection_error_m"], errors="coerce")
    s_err = pd.to_numeric(merged["scale_m2_projection_error_m"], errors="coerce")
    merged["baseline_gt200"] = b_err > 200.0
    merged["scale_m2_gt200"] = s_err > 200.0
    merged["catastrophic_rescue"] = merged["baseline_gt200"] & ~merged["scale_m2_gt200"]
    merged["catastrophic_damage"] = ~merged["baseline_gt200"] & merged["scale_m2_gt200"]

    summary = {
        "queries": int(len(merged)),
        "winner_changed": int(merged["winner_changed"].sum()),
        "containment_rescues": int(merged["containment_rescue"].sum()),
        "containment_damages": int(merged["containment_damage"].sum()),
        "center_le80_rescues": int(merged["le80_rescue"].sum()),
        "center_le80_damages": int(merged["le80_damage"].sum()),
        "catastrophic_rescues": int(merged["catastrophic_rescue"].sum()),
        "catastrophic_damages": int(merged["catastrophic_damage"].sum()),
    }
    return summary, merged


def evaluate_replication_gate(
    baseline: dict[str, Any],
    scale: dict[str, Any],
    cfg: dict[str, Any],
) -> dict[str, Any]:
    gate = cfg["replication_gate"]
    gt200_reduction = (
        int(baseline["selected_projection"]["gt200"])
        - int(scale["selected_projection"]["gt200"])
    )
    le40_delta = (
        int(scale["selected_projection"]["le40"])
        - int(baseline["selected_projection"]["le40"])
    )
    containment_delta = (
        int(scale["selected_top1"]["contains_query"])
        - int(baseline["selected_top1"]["contains_query"])
    )
    le80_delta = (
        int(scale["selected_top1"]["center_le80"])
        - int(baseline["selected_top1"]["center_le80"])
    )
    checks = {
        "gt200_reduction": gt200_reduction,
        "gt200_pass": gt200_reduction >= int(gate["min_selected_projection_gt200_reduction"]),
        "projection_le40_delta": le40_delta,
        "projection_le40_pass": le40_delta >= -int(gate["max_selected_projection_le40_loss"]),
        "top1_containment_delta": containment_delta,
        "top1_containment_pass": containment_delta >= -int(gate["max_top1_containment_loss"]),
        "top1_center_le80_delta": le80_delta,
        "top1_center_le80_pass": le80_delta >= -int(gate["max_top1_center_le80_loss"]),
    }
    checks["pass"] = bool(
        checks["gt200_pass"]
        and checks["projection_le40_pass"]
        and checks["top1_containment_pass"]
        and checks["top1_center_le80_pass"]
    )
    return checks


def verify_baseline_control(summary: dict[str, Any], cfg: dict[str, Any]) -> None:
    expected = cfg["baseline"]["expected_control"]
    observed = {
        "top1_contains_query": summary["selected_top1"]["contains_query"],
        "top1_center_le40": summary["selected_top1"]["center_le40"],
        "top1_center_le80": summary["selected_top1"]["center_le80"],
        "selected_projection_le40": summary["selected_projection"]["le40"],
        "selected_projection_le80": summary["selected_projection"]["le80"],
        "selected_projection_gt200": summary["selected_projection"]["gt200"],
    }
    if observed != expected:
        raise RuntimeError(
            "Frozen GD1.0 baseline control metrics changed: "
            + json.dumps({"expected": expected, "observed": observed}, sort_keys=True)
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CV1.4 full403 post-freeze scale_m2 vs frozen ORB evaluation"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()

    cfg_path = resolve(args.config)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    verify_config(cfg)
    baseline_path, scale_path = verify_inputs(cfg)
    tile_path = verify_tile_geometry(cfg)

    baseline_raw = pd.read_csv(baseline_path)
    scale_raw = pd.read_csv(scale_path)
    for name, frame, block in (
        ("baseline", baseline_raw, cfg["baseline"]),
        ("scale_m2", scale_raw, cfg["scale_m2"]),
    ):
        if len(frame) != int(block["expected_pairs"]):
            raise RuntimeError(f"{name} pair count mismatch")
        if frame["query_id"].nunique() != int(block["expected_queries"]):
            raise RuntimeError(f"{name} query count mismatch")

    key_cols = ["query_id", "pool_rank", "tile_id"]
    if not baseline_raw[key_cols].equals(scale_raw[key_cols]):
        raise RuntimeError("Baseline and scale_m2 frozen pair ordering differs")

    print("Both full403 evidence freezes verified before reference attachment.")
    print("Baseline SHA256:", sha256_file(baseline_path))
    print("Scale_m2 SHA256:", sha256_file(scale_path))
    print("Attaching development reference for CV1.4 diagnostics only.")

    reference_path = resolve(cfg["development"]["reference_attachment"])
    reference = load_reference_xy(reference_path)
    tile = pd.read_csv(tile_path)

    baseline = attach_reference(baseline_raw, tile, reference, condition="baseline")
    scale = attach_reference(scale_raw, tile, reference, condition="scale_m2")

    topks = [int(x) for x in cfg["ranking"]["topk"]]
    baseline_summary, baseline_selected = summarize_condition(baseline, topks)
    scale_summary, scale_selected = summarize_condition(scale, topks)
    verify_baseline_control(baseline_summary, cfg)

    change_summary, selected_comparison = compare_selected(
        baseline_selected, scale_selected
    )
    gate = evaluate_replication_gate(baseline_summary, scale_summary, cfg)

    report = {
        "stage": "CV1.4",
        "status": "PASS_CV1_4_POSTFREEZE_FULL403_SCALE_VALIDATION",
        "baseline": baseline_summary,
        "scale_m2": scale_summary,
        "selected_winner_changes": change_summary,
        "replication_gate": gate,
        "decision": (
            "CONTINUE_SCALE_MECHANISM_AND_CAMERA_GSD_AUDIT"
            if gate["pass"]
            else "CLOSE_FIXED_SCALE_M2_AS_NONROBUST_FULL403"
        ),
        "scientific_boundary": {
            "both_freezes_verified_before_reference": True,
            "reference_loaded_after_freeze": True,
            "srt_loaded": False,
            "blind_stress_loaded": False,
            "transform_retuned": False,
            "additional_scale_conditions_tested": False,
            "best_transform_per_query_executed": False,
            "selector_formula_changed": False,
            "threshold_tuned": False,
            "state_modified": False,
            "temporal_fusion_executed": False,
        },
        "interpretation": {
            "development_full403_only": True,
            "physical_gsd_claim": False,
            "deployment_promotion": False,
            "replication_gate_is_resource_gate_not_significance_test": True,
        },
        "provenance": {
            "config_sha256": sha256_file(cfg_path),
            "baseline_evidence_sha256": sha256_file(baseline_path),
            "scale_m2_evidence_sha256": sha256_file(scale_path),
            "scale_m2_report_sha256": sha256_file(resolve(cfg["scale_m2"]["report"])),
            "tile_geometry_sha256": sha256_file(tile_path),
            "reference_attachment_sha256": sha256_file(reference_path),
        },
    }

    summary_rows = []
    for name, s in (("baseline", baseline_summary), ("scale_m2", scale_summary)):
        summary_rows.append({
            "condition": name,
            "top1_contains_query": s["selected_top1"]["contains_query"],
            "top1_center_le40": s["selected_top1"]["center_le40"],
            "top1_center_le80": s["selected_top1"]["center_le80"],
            "projection_median_m": s["selected_projection"]["median_m"],
            "projection_rmse_m": s["selected_projection"]["rmse_m"],
            "projection_p95_m": s["selected_projection"]["p95_m"],
            "projection_le40": s["selected_projection"]["le40"],
            "projection_le80": s["selected_projection"]["le80"],
            "projection_gt200": s["selected_projection"]["gt200"],
            "oracle_projection_median_m": s["oracle_best_finite_projection"]["median_m"],
            "oracle_projection_le40": s["oracle_best_finite_projection"]["le40"],
            "oracle_projection_le80": s["oracle_best_finite_projection"]["le80"],
            "median_top1_top2_inlier_margin": s["median_top1_top2_inlier_margin"],
        })
    summary_frame = pd.DataFrame(summary_rows)

    output_root = resolve(cfg["output_root"])
    summary_path = output_root / "cv1_4_full403_summary.csv"
    selected_path = output_root / "cv1_4_selected_query_comparison.csv"
    report_path = output_root / "cv1_4_validation_report.json"
    write_immutable({
        summary_path: deterministic_csv(summary_frame),
        selected_path: deterministic_csv(selected_comparison),
        report_path: deterministic_json(report),
    })

    print("Status:", report["status"])
    print()
    print(summary_frame.to_string(index=False))
    print()
    print("Winner changed:", change_summary["winner_changed"])
    print(
        "Containment rescues/damages:",
        f"{change_summary['containment_rescues']}/{change_summary['containment_damages']}",
    )
    print(
        "Center<=80 rescues/damages:",
        f"{change_summary['center_le80_rescues']}/{change_summary['center_le80_damages']}",
    )
    print(
        "Catastrophic rescues/damages:",
        f"{change_summary['catastrophic_rescues']}/{change_summary['catastrophic_damages']}",
    )
    print("Replication gate:", "PASS" if gate["pass"] else "FAIL")
    print("Decision:", report["decision"])
    print("SRT loaded: False")
    print("Transform retuned: False")
    print("Threshold tuned: False")
    print("Report:", report_path)


if __name__ == "__main__":
    main()
