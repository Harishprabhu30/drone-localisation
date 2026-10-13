#!/usr/bin/env python3
"""CV1.2: post-freeze evaluation of fixed CV1.1 scale/rotation conditions."""
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
DEFAULT_CONFIG = ROOT / "configs/research/crossview_scale_rotation_domain_cv1_2.yaml"

BASE_REQUIRED = {
    "query_id", "pool_rank", "tile_id", "transform_id",
    "inliers", "homography_ok", "finite_projection",
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
    return frame.to_csv(
        index=False, lineterminator="\n", float_format="%.9f"
    ).encode("utf-8")


def deterministic_json(value: dict[str, Any]) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def write_immutable(files: dict[Path, bytes]) -> None:
    changed = [
        str(path)
        for path, data in files.items()
        if path.exists() and path.read_bytes() != data
    ]
    if changed:
        raise RuntimeError(
            "Refusing to overwrite differing CV1.2 outputs: "
            + ", ".join(changed)
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
    if cfg.get("stage") != "CV1.2":
        raise RuntimeError("Expected CV1.2 config")
    if cfg["map"]["variant"] != "768_s256" or cfg["map"]["crs"] != "EPSG:3346":
        raise RuntimeError("CV1.2 map contract changed")
    if cfg["labels"] != ["contains_query", "center_le40", "center_le80"]:
        raise RuntimeError("CV1.2 label contract changed")
    if cfg["ranking"]["selector"] != "inliers":
        raise RuntimeError("CV1.2 selector changed")
    if cfg["ranking"]["tie_break"] != ["pool_rank_asc", "tile_id_asc"]:
        raise RuntimeError("CV1.2 tie-break changed")
    boundary = cfg["scientific_boundary"]
    if boundary.get("development_reference_allowed") is not True:
        raise RuntimeError("Development reference must be explicitly allowed")
    if boundary.get("reference_after_cv1_1_freeze_only") is not True:
        raise RuntimeError("Reference-after-freeze boundary missing")
    for key in (
        "cv1_1_evidence_mutable",
        "transform_parameters_mutable",
        "best_transform_per_query_allowed",
        "best_transform_per_candidate_allowed",
        "selector_formula_mutable",
        "threshold_tuning_enabled",
        "full403_loaded",
        "blind_stress_loaded",
        "srt_loaded",
        "state_enabled",
        "temporal_fusion_enabled",
    ):
        if boundary.get(key) is not False:
            raise RuntimeError(f"{key} must be false")


def verify_cv11(cfg: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    block = cfg["cv1_1"]
    report_path = resolve(block["report"])
    evidence_path = resolve(block["evidence"])
    if not report_path.is_file() or not evidence_path.is_file():
        raise FileNotFoundError("Missing CV1.1 freeze inputs")

    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != block["expected_status"]:
        raise RuntimeError("CV1.1 report status mismatch")
    if report.get("baseline_parity", {}).get("status") != block["expected_baseline_parity_status"]:
        raise RuntimeError("CV1.1 baseline parity status mismatch")
    if report.get("scientific_boundary", {}).get("reference_loaded") is not False:
        raise RuntimeError("CV1.1 reference boundary violated")
    if report.get("scientific_boundary", {}).get("srt_loaded") is not False:
        raise RuntimeError("CV1.1 SRT boundary violated")
    if report.get("provenance", {}).get("evidence_output_sha256") != block["expected_evidence_sha256"]:
        raise RuntimeError("CV1.1 report evidence hash mismatch")
    if sha256_file(evidence_path) != block["expected_evidence_sha256"]:
        raise RuntimeError("CV1.1 evidence bytes changed")
    return evidence_path, report


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
) -> pd.DataFrame:
    require_columns(evidence, BASE_REQUIRED, "CV1.1 evidence")
    require_columns(tile, TILE_REQUIRED, "tile geometry")

    frame = evidence.copy()
    frame["query_id"] = pd.to_numeric(frame["query_id"], errors="raise").astype(int)
    frame["pool_rank"] = pd.to_numeric(frame["pool_rank"], errors="raise").astype(int)
    frame["tile_id"] = frame["tile_id"].astype(str)
    frame["transform_id"] = frame["transform_id"].astype(str)
    frame["homography_ok"] = bool_series(frame["homography_ok"])
    frame["finite_projection"] = bool_series(frame["finite_projection"])

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
        raise RuntimeError("CV1.2 reference/tile join produced missing values")

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


def first_positive_rank(group: pd.DataFrame, label: str) -> int | None:
    ordered = rank_candidates(group).reset_index(drop=True)
    hits = np.flatnonzero(ordered[label].astype(bool).to_numpy())
    return int(hits[0] + 1) if len(hits) else None


def projection_summary(values: pd.Series) -> dict[str, Any]:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    if not len(arr):
        return {
            "count": 0, "median_m": None, "rmse_m": None, "p95_m": None,
            "le40": 0, "le80": 0, "gt200": 0,
        }
    return {
        "count": int(len(arr)),
        "median_m": float(np.median(arr)),
        "rmse_m": float(np.sqrt(np.mean(arr * arr))),
        "p95_m": float(np.quantile(arr, 0.95)),
        "le40": int((arr <= 40.0).sum()),
        "le80": int((arr <= 80.0).sum()),
        "gt200": int((arr > 200.0).sum()),
    }


def summarize_condition(
    frame: pd.DataFrame,
    *,
    topks: list[int],
) -> tuple[dict[str, Any], pd.DataFrame]:
    query_rows: list[dict[str, Any]] = []
    label_hits = {
        label: {f"top{k}": 0 for k in topks}
        for label in ("contains_query", "center_le40", "center_le80")
    }

    for qid, group in frame.groupby("query_id", sort=True):
        ordered = rank_candidates(group).reset_index(drop=True)
        selected = ordered.iloc[0]
        row = {
            "query_id": int(qid),
            "pool_rank": int(selected["pool_rank"]),
            "tile_id": str(selected["tile_id"]),
            "inliers": float(selected["inliers"]),
            "contains_query": bool(selected["contains_query"]),
            "center_le40": bool(selected["center_le40"]),
            "center_le80": bool(selected["center_le80"]),
            "projection_error_m": (
                float(selected["projection_error_m"])
                if pd.notna(selected["projection_error_m"]) else np.nan
            ),
            "top1_top2_inlier_margin": (
                float(pd.to_numeric(ordered.iloc[0]["inliers"], errors="coerce"))
                - float(pd.to_numeric(ordered.iloc[1]["inliers"], errors="coerce"))
                if len(ordered) >= 2 else np.nan
            ),
        }
        for label in label_hits:
            row[f"first_{label}_rank"] = first_positive_rank(group, label)
            for k in topks:
                hit = bool(ordered.head(k)[label].astype(bool).any())
                label_hits[label][f"top{k}"] += int(hit)
        query_rows.append(row)

    selected = pd.DataFrame(query_rows)
    proj = projection_summary(selected["projection_error_m"])
    summary = {
        "queries": int(len(selected)),
        "ranking": label_hits,
        "selected_projection": proj,
        "selected_top1": {
            "contains_query": int(selected["contains_query"].sum()),
            "center_le40": int(selected["center_le40"].sum()),
            "center_le80": int(selected["center_le80"].sum()),
        },
        "median_top1_top2_inlier_margin": float(
            pd.to_numeric(selected["top1_top2_inlier_margin"], errors="coerce").median()
        ),
    }
    return summary, selected


def evaluate_gate(
    baseline: dict[str, Any],
    condition: dict[str, Any],
    gate: dict[str, Any],
) -> dict[str, Any]:
    containment_gain = (
        int(condition["selected_top1"]["contains_query"])
        - int(baseline["selected_top1"]["contains_query"])
    )
    le80_gain = (
        int(condition["selected_top1"]["center_le80"])
        - int(baseline["selected_top1"]["center_le80"])
    )
    gt200_reduction = (
        int(baseline["selected_projection"]["gt200"])
        - int(condition["selected_projection"]["gt200"])
    )
    le40_projection_delta = (
        int(condition["selected_projection"]["le40"])
        - int(baseline["selected_projection"]["le40"])
    )

    containment_pass = containment_gain >= int(gate["containment_gain_min"])
    le80_pass = le80_gain >= int(gate["le80_gain_min"])
    catastrophic_pass = (
        gt200_reduction >= int(gate["catastrophic_gt200_reduction_min"])
        and le40_projection_delta
        >= -int(gate["max_le40_projection_loss_if_using_catastrophic_gate"])
    )
    return {
        "containment_gain": containment_gain,
        "center_le80_gain": le80_gain,
        "selected_projection_gt200_reduction": gt200_reduction,
        "selected_projection_le40_delta": le40_projection_delta,
        "containment_gate_pass": bool(containment_pass),
        "center_le80_gate_pass": bool(le80_pass),
        "catastrophic_tail_gate_pass": bool(catastrophic_pass),
        "pass": bool(containment_pass or le80_pass or catastrophic_pass),
    }


def promotion_key(
    condition_id: str,
    summary: dict[str, Any],
    fixed_order: list[str],
) -> tuple[Any, ...]:
    proj = summary["selected_projection"]
    p95 = proj["p95_m"]
    return (
        int(proj["gt200"]),
        -int(summary["selected_top1"]["contains_query"]),
        -int(summary["selected_top1"]["center_le80"]),
        -int(proj["le40"]),
        float(p95) if p95 is not None else math.inf,
        fixed_order.index(condition_id),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CV1.2 post-freeze fixed scale/rotation uniform40 evaluation"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()

    cfg_path = resolve(args.config)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    verify_config(cfg)

    evidence_path, cv11_report = verify_cv11(cfg)
    tile_path = verify_tile_geometry(cfg)

    evidence = pd.read_csv(evidence_path)
    if len(evidence) != int(cfg["cv1_1"]["expected_pairs"]):
        raise RuntimeError("CV1.1 evidence pair count mismatch")
    if evidence["query_id"].nunique() != int(cfg["cv1_1"]["expected_queries"]):
        raise RuntimeError("CV1.1 query count mismatch")
    if evidence["transform_id"].nunique() != int(cfg["cv1_1"]["expected_conditions"]):
        raise RuntimeError("CV1.1 condition count mismatch")

    fixed_order = [str(x) for x in cfg["conditions"]["fixed_order"]]
    actual_conditions = evidence["transform_id"].astype(str).drop_duplicates().tolist()
    if sorted(actual_conditions) != sorted(fixed_order):
        raise RuntimeError(
            f"CV1.1 condition IDs differ: expected={fixed_order} actual={actual_conditions}"
        )

    print("Frozen CV1.1 evidence verified before reference attachment.")
    print("Evidence SHA256:", sha256_file(evidence_path))
    print("Attaching development reference for CV1.2 diagnostics only.")

    reference = load_reference_xy(resolve(cfg["development"]["reference_attachment"]))
    tile = pd.read_csv(tile_path)
    labeled = attach_reference(evidence, tile, reference)

    summaries: dict[str, Any] = {}
    selected_tables: list[pd.DataFrame] = []
    topks = [int(x) for x in cfg["ranking"]["topk"]]

    for condition_id in fixed_order:
        frame = labeled[labeled["transform_id"].astype(str) == condition_id].copy()
        summary, selected = summarize_condition(frame, topks=topks)
        selected.insert(0, "transform_id", condition_id)
        summaries[condition_id] = summary
        selected_tables.append(selected)

    baseline_id = str(cfg["conditions"]["baseline"])
    baseline = summaries[baseline_id]
    gates: dict[str, Any] = {}
    for condition_id in fixed_order:
        if condition_id == baseline_id:
            gates[condition_id] = {
                "pass": False,
                "reason": "baseline_control",
            }
        else:
            gates[condition_id] = evaluate_gate(
                baseline,
                summaries[condition_id],
                cfg["continuation_gate_uniform40"],
            )

    eligible = [
        cid for cid in fixed_order
        if cid != baseline_id and gates[cid].get("pass") is True
    ]
    promoted = (
        min(eligible, key=lambda cid: promotion_key(cid, summaries[cid], fixed_order))
        if eligible else None
    )

    selected_all = pd.concat(selected_tables, ignore_index=True)
    baseline_selected = selected_all[
        selected_all["transform_id"] == baseline_id
    ][["query_id", "contains_query", "center_le40", "center_le80", "projection_error_m"]].copy()
    baseline_selected = baseline_selected.rename(columns={
        "contains_query": "baseline_contains_query",
        "center_le40": "baseline_center_le40",
        "center_le80": "baseline_center_le80",
        "projection_error_m": "baseline_projection_error_m",
    })
    selected_all = selected_all.merge(
        baseline_selected, on="query_id", how="left", validate="many_to_one"
    )
    selected_all["containment_rescue"] = (
        selected_all["contains_query"] & ~selected_all["baseline_contains_query"]
    )
    selected_all["containment_damage"] = (
        ~selected_all["contains_query"] & selected_all["baseline_contains_query"]
    )
    selected_all["center_le80_rescue"] = (
        selected_all["center_le80"] & ~selected_all["baseline_center_le80"]
    )
    selected_all["center_le80_damage"] = (
        ~selected_all["center_le80"] & selected_all["baseline_center_le80"]
    )
    selected_all["projection_gt200"] = (
        pd.to_numeric(selected_all["projection_error_m"], errors="coerce") > 200.0
    )
    selected_all["baseline_projection_gt200"] = (
        pd.to_numeric(selected_all["baseline_projection_error_m"], errors="coerce") > 200.0
    )
    selected_all["catastrophic_rescue"] = (
        ~selected_all["projection_gt200"] & selected_all["baseline_projection_gt200"]
    )
    selected_all["catastrophic_damage"] = (
        selected_all["projection_gt200"] & ~selected_all["baseline_projection_gt200"]
    )

    delta_summary: dict[str, Any] = {}
    for cid in fixed_order:
        rows = selected_all[selected_all["transform_id"] == cid]
        delta_summary[cid] = {
            "containment_rescues": int(rows["containment_rescue"].sum()),
            "containment_damages": int(rows["containment_damage"].sum()),
            "center_le80_rescues": int(rows["center_le80_rescue"].sum()),
            "center_le80_damages": int(rows["center_le80_damage"].sum()),
            "catastrophic_rescues": int(rows["catastrophic_rescue"].sum()),
            "catastrophic_damages": int(rows["catastrophic_damage"].sum()),
        }

    report = {
        "stage": "CV1.2",
        "status": "PASS_CV1_2_POSTFREEZE_UNIFORM40_EVALUATION",
        "input_evidence_sha256": sha256_file(evidence_path),
        "conditions": summaries,
        "baseline_id": baseline_id,
        "continuation_gates": gates,
        "delta_summary": delta_summary,
        "eligible_conditions": eligible,
        "promoted_condition_for_full403": promoted,
        "promotion_rule": cfg["promotion_rule_if_multiple_pass"],
        "scientific_boundary": {
            "reference_loaded_after_freeze": True,
            "srt_loaded": False,
            "full403_loaded": False,
            "blind_stress_loaded": False,
            "transform_parameters_changed": False,
            "best_transform_per_query_executed": False,
            "best_transform_per_candidate_executed": False,
            "selector_formula_changed": False,
            "threshold_tuned": False,
            "state_modified": False,
            "temporal_fusion_executed": False,
        },
        "interpretation": {
            "screening_only": True,
            "full403_accuracy_claim": False,
            "physical_gsd_claim": False,
            "promotion_means_full403_continuation_only": True,
        },
        "provenance": {
            "config_sha256": sha256_file(cfg_path),
            "cv1_1_report_sha256": sha256_file(resolve(cfg["cv1_1"]["report"])),
            "cv1_1_evidence_sha256": sha256_file(evidence_path),
            "tile_geometry_sha256": sha256_file(tile_path),
            "reference_attachment_sha256": sha256_file(resolve(cfg["development"]["reference_attachment"])),
            "cv1_1_report_status": cv11_report["status"],
        },
    }

    output_root = resolve(cfg["output_root"])
    selected_path = output_root / "cv1_2_selected_query_audit.csv"
    summary_rows = []
    for cid in fixed_order:
        s = summaries[cid]
        g = gates[cid]
        d = delta_summary[cid]
        summary_rows.append({
            "transform_id": cid,
            "top1_contains_query": s["selected_top1"]["contains_query"],
            "top1_center_le40": s["selected_top1"]["center_le40"],
            "top1_center_le80": s["selected_top1"]["center_le80"],
            "projection_median_m": s["selected_projection"]["median_m"],
            "projection_rmse_m": s["selected_projection"]["rmse_m"],
            "projection_p95_m": s["selected_projection"]["p95_m"],
            "projection_le40": s["selected_projection"]["le40"],
            "projection_le80": s["selected_projection"]["le80"],
            "projection_gt200": s["selected_projection"]["gt200"],
            "containment_rescues": d["containment_rescues"],
            "containment_damages": d["containment_damages"],
            "center_le80_rescues": d["center_le80_rescues"],
            "center_le80_damages": d["center_le80_damages"],
            "catastrophic_rescues": d["catastrophic_rescues"],
            "catastrophic_damages": d["catastrophic_damages"],
            "continuation_gate_pass": bool(g.get("pass", False)),
        })
    summary_frame = pd.DataFrame(summary_rows)
    summary_path = output_root / "cv1_2_condition_summary.csv"
    report_path = output_root / "cv1_2_evaluation_report.json"

    write_immutable({
        selected_path: deterministic_csv(selected_all),
        summary_path: deterministic_csv(summary_frame),
        report_path: deterministic_json(report),
    })

    print("Status:", report["status"])
    print("Input evidence SHA256:", report["input_evidence_sha256"])
    print()
    print(summary_frame.to_string(index=False))
    print()
    print("Eligible conditions:", ",".join(eligible) if eligible else "NONE")
    print("Promoted condition for full403:", promoted or "NONE")
    print("SRT loaded: False")
    print("Full403 loaded: False")
    print("Threshold tuned: False")
    print("Report:", report_path)


if __name__ == "__main__":
    main()
