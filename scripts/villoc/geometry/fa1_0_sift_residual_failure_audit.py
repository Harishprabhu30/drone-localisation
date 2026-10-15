#!/usr/bin/env python3
"""FA1.0: post-freeze audit of residual SIFT+LightGlue failures.

Uses only frozen CV2.4/CV2.5 evidence and already-authorized development truth.
No matcher rerun and no selector/threshold tuning.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = ROOT / "configs/research/crossview_scale_rotation_domain_fa1.yaml"
CV25_SCRIPT = ROOT / "scripts/villoc/geometry/cv2_5_postfreeze_sift_lightglue_full403_validation.py"


def resolve(v):
    p = Path(v).expanduser()
    return p if p.is_absolute() else ROOT / p


def dynamic_import(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def error_bin(value: float) -> str:
    if not math.isfinite(value) or value > 200.0:
        return "unsafe"
    if value <= 40.0:
        return "le40"
    return "mid40_200"


def rank_with_selector(cv25, group: pd.DataFrame) -> pd.DataFrame:
    ranked = cv25.rank(group).reset_index(drop=True).copy()
    ranked["selector_rank"] = np.arange(1, len(ranked) + 1)
    return ranked


def finite_or_nan(value):
    try:
        v = float(value)
    except Exception:
        return math.nan
    return v if math.isfinite(v) else math.nan


def prefixed_pair(row: pd.Series, prefix: str, evidence_cols: list[str]) -> dict[str, Any]:
    keys = [
        "tile_id", "pool_rank", "source_view", "source_rank",
        "good_matches", "inliers", "homography_ok", "finite_projection",
        "projected_inside_tile", "projected_easting", "projected_northing",
        "projection_error_m", "center_error_m", "contains_query",
        "center_le40", "center_le80",
    ]
    out = {}
    for key in keys:
        if key in row.index:
            out[f"{prefix}_{key}"] = row[key]
    for key in evidence_cols:
        if key in row.index:
            out[f"{prefix}_{key}"] = row[key]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = ap.parse_args()

    cfg_path = resolve(args.config)
    cfg = yaml.safe_load(cfg_path.read_text())
    if cfg.get("stage") != "FA1.0":
        raise RuntimeError("Expected FA1.0 config")

    b = cfg["scientific_boundary"]
    if b["postfreeze_development_truth_allowed"] is not True:
        raise RuntimeError("Post-freeze development truth must be explicitly allowed")
    for key in (
        "frozen_evidence_mutable", "matcher_rerun_allowed", "threshold_tuning_allowed",
        "selector_change_allowed", "confidence_rule_training_allowed",
        "blind_data_loaded", "temporal_fusion_changed",
    ):
        if b[key] is not False:
            raise RuntimeError(f"{key} must remain false")

    cv25 = dynamic_import("fa1_cv25", CV25_SCRIPT)
    cv25_cfg_path = resolve(cfg["source"]["cv2_5_config"])
    cv25_cfg = yaml.safe_load(cv25_cfg_path.read_text())
    cv25.verify_config(cv25_cfg)

    orb_raw, scale_raw, sift_raw = cv25.verify_inputs(cv25_cfg)
    if cv25.sha256_file(resolve(cv25_cfg["sift_lightglue"]["evidence"])) != cfg["source"]["expected_sift_sha256"]:
        raise RuntimeError("SIFT frozen evidence SHA mismatch")
    if cv25.sha256_file(resolve(cv25_cfg["scale_m2_orb"]["evidence"])) != cfg["source"]["expected_scale_m2_sha256"]:
        raise RuntimeError("scale_m2 frozen evidence SHA mismatch")

    tile_path = cv25.verify_tiles(cv25_cfg)
    tiles = pd.read_csv(tile_path)
    ref_path = cv25.resolve(cv25_cfg["development"]["reference_attachment"])
    reference = cv25.load_reference_xy(ref_path)

    scale = cv25.attach(scale_raw, tiles, reference, "scale_m2_orb")
    sift = cv25.attach(sift_raw, tiles, reference, "sift_lightglue")

    evidence_cols = sorted(c for c in sift.columns if str(c).startswith("evidence_"))
    qids = sorted(pd.to_numeric(sift["query_id"], errors="raise").astype(int).unique())
    if len(qids) != int(cfg["source"]["expected_queries"]):
        raise RuntimeError("FA1.0 query count changed")

    case_rows = []
    selected_rows = {}
    oracle_rows = {}
    scale_selected_rows = {}

    for qid in qids:
        sg = sift[sift["query_id"] == qid].copy()
        rg = rank_with_selector(cv25, sg)
        selected = rg.iloc[0]
        finite = rg[np.isfinite(pd.to_numeric(rg["projection_error_m"], errors="coerce"))].copy()
        if finite.empty:
            oracle = rg.iloc[0]
        else:
            oracle = finite.sort_values(
                ["projection_error_m", "pool_rank", "tile_id"],
                ascending=[True, True, True],
                kind="mergesort",
            ).iloc[0]

        scale_ranked = rank_with_selector(
            cv25,
            scale[scale["query_id"] == qid].copy(),
        )
        scale_selected = scale_ranked.iloc[0]

        selected_rows[qid] = selected
        oracle_rows[qid] = oracle
        scale_selected_rows[qid] = scale_selected

        sel_err = finite_or_nan(selected["projection_error_m"])
        ora_err = finite_or_nan(oracle["projection_error_m"])
        scale_err = finite_or_nan(scale_selected["projection_error_m"])

        oracle_rescuable = (
            (not math.isfinite(sel_err) or sel_err > float(cfg["case_contract"]["oracle_rescuable"]["selected_error_gt_m"]))
            and math.isfinite(ora_err)
            and ora_err <= float(cfg["case_contract"]["oracle_rescuable"]["oracle_error_le_m"])
        )
        sift_unsafe = (not math.isfinite(sel_err)) or sel_err > float(cfg["case_contract"]["catastrophic_damage"]["sift_error_gt_m"])
        scale_safe200 = math.isfinite(scale_err) and scale_err <= float(cfg["case_contract"]["catastrophic_damage"]["scale_m2_error_le_m"])
        catastrophic_damage = bool(scale_safe200 and sift_unsafe)
        severe = bool(
            math.isfinite(scale_err)
            and scale_err <= float(cfg["case_contract"]["severe_regression"]["scale_m2_error_le_m"])
            and sift_unsafe
        )

        if not (oracle_rescuable or catastrophic_damage):
            continue

        oracle_selector_rank = int(
            rg.loc[rg["tile_id"].astype(str) == str(oracle["tile_id"]), "selector_rank"].iloc[0]
        )
        second = rg.iloc[1]

        row = {
            "query_id": int(qid),
            "oracle_rescuable": bool(oracle_rescuable),
            "catastrophic_damage": bool(catastrophic_damage),
            "severe_orb_le40_to_sift_unsafe": bool(severe),
            "scale_error_bin": error_bin(scale_err),
            "sift_selected_error_bin": error_bin(sel_err),
            "sift_selected_error_m": sel_err,
            "sift_oracle_error_m": ora_err,
            "scale_selected_error_m": scale_err,
            "sift_selected_selector_rank": 1,
            "sift_oracle_selector_rank": oracle_selector_rank,
            "sift_oracle_pool_rank": int(oracle["pool_rank"]),
            "sift_selected_pool_rank": int(selected["pool_rank"]),
            "sift_selected_inliers": float(selected["inliers"]),
            "sift_second_inliers": float(second["inliers"]),
            "sift_top1_top2_inlier_margin": float(selected["inliers"]) - float(second["inliers"]),
            "sift_oracle_inliers": float(oracle["inliers"]),
            "sift_selected_minus_oracle_inliers": float(selected["inliers"]) - float(oracle["inliers"]),
            "sift_selected_good_matches": float(selected["good_matches"]),
            "sift_oracle_good_matches": float(oracle["good_matches"]),
            "sift_selected_oracle_projected_distance_m": (
                float(np.hypot(
                    float(selected["projected_easting"]) - float(oracle["projected_easting"]),
                    float(selected["projected_northing"]) - float(oracle["projected_northing"]),
                ))
                if all(math.isfinite(float(v)) for v in (
                    selected["projected_easting"], selected["projected_northing"],
                    oracle["projected_easting"], oracle["projected_northing"],
                )) else math.nan
            ),
            "selected_equals_oracle_tile": str(selected["tile_id"]) == str(oracle["tile_id"]),
        }
        row.update(prefixed_pair(selected, "sift_selected", evidence_cols))
        row.update(prefixed_pair(oracle, "sift_oracle", evidence_cols))
        row.update(prefixed_pair(scale_selected, "scale_selected", []))
        case_rows.append(row)

    cases = pd.DataFrame(case_rows).sort_values(
        ["oracle_rescuable", "catastrophic_damage", "query_id"],
        ascending=[False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)

    expected_oracle = int(cfg["case_contract"]["oracle_rescuable"]["expected_count"])
    expected_cat = int(cfg["case_contract"]["catastrophic_damage"]["expected_count"])
    expected_severe = int(cfg["case_contract"]["severe_regression"]["expected_count"])
    observed_oracle = int(cases["oracle_rescuable"].sum())
    observed_cat = int(cases["catastrophic_damage"].sum())
    observed_severe = int(cases["severe_orb_le40_to_sift_unsafe"].sum())
    if observed_oracle != expected_oracle:
        raise RuntimeError(f"Oracle-rescuable count changed: {observed_oracle} != {expected_oracle}")
    if observed_cat != expected_cat:
        raise RuntimeError(f"Catastrophic-damage count changed: {observed_cat} != {expected_cat}")
    if observed_severe != expected_severe:
        raise RuntimeError(f"Severe regression count changed: {observed_severe} != {expected_severe}")

    case_qids = sorted(cases["query_id"].astype(int).unique())

    detail_rows = []
    for qid in case_qids:
        rg = rank_with_selector(cv25, sift[sift["query_id"] == qid].copy())
        oracle_tile = str(oracle_rows[qid]["tile_id"])
        selected_tile = str(selected_rows[qid]["tile_id"])
        for _, row in rg.iterrows():
            out = row.to_dict()
            out["selector_rank"] = int(row["selector_rank"])
            out["is_selected"] = str(row["tile_id"]) == selected_tile
            out["is_oracle_best"] = str(row["tile_id"]) == oracle_tile
            detail_rows.append(out)
    detail = pd.DataFrame(detail_rows)

    radius = int(cfg["case_contract"]["temporal_radius_queries"])
    temporal_rows = []
    for case_q in case_qids:
        for qid in range(max(min(qids), case_q - radius), min(max(qids), case_q + radius) + 1):
            if qid not in selected_rows:
                continue
            s = selected_rows[qid]
            r = scale_selected_rows[qid]
            temporal_rows.append({
                "case_query_id": int(case_q),
                "neighbor_query_id": int(qid),
                "offset": int(qid - case_q),
                "sift_tile_id": str(s["tile_id"]),
                "sift_pool_rank": int(s["pool_rank"]),
                "sift_inliers": float(s["inliers"]),
                "sift_projection_error_m": finite_or_nan(s["projection_error_m"]),
                "sift_projected_easting": finite_or_nan(s["projected_easting"]),
                "sift_projected_northing": finite_or_nan(s["projected_northing"]),
                "scale_tile_id": str(r["tile_id"]),
                "scale_pool_rank": int(r["pool_rank"]),
                "scale_inliers": float(r["inliers"]),
                "scale_projection_error_m": finite_or_nan(r["projection_error_m"]),
                "scale_projected_easting": finite_or_nan(r["projected_easting"]),
                "scale_projected_northing": finite_or_nan(r["projected_northing"]),
            })
    temporal = pd.DataFrame(temporal_rows)

    overlap = int((cases["oracle_rescuable"] & cases["catastrophic_damage"]).sum())
    oracle_only = int((cases["oracle_rescuable"] & ~cases["catastrophic_damage"]).sum())
    cat_only = int((~cases["oracle_rescuable"] & cases["catastrophic_damage"]).sum())

    oracle_cases = cases[cases["oracle_rescuable"]]
    summary = {
        "stage": "FA1.0",
        "status": "PASS_FA1_0_FROZEN_SIFT_FAILURE_AUDIT",
        "counts": {
            "oracle_rescuable": observed_oracle,
            "catastrophic_damage": observed_cat,
            "severe_orb_le40_to_sift_unsafe": observed_severe,
            "union_unique_queries": int(len(case_qids)),
            "intersection_oracle_rescuable_and_catastrophic_damage": overlap,
            "oracle_rescuable_only": oracle_only,
            "catastrophic_damage_only": cat_only,
        },
        "oracle_rescuable_diagnostics": {
            "oracle_selector_rank_median": float(oracle_cases["sift_oracle_selector_rank"].median()),
            "oracle_selector_rank_max": int(oracle_cases["sift_oracle_selector_rank"].max()),
            "oracle_pool_rank_median": float(oracle_cases["sift_oracle_pool_rank"].median()),
            "selected_minus_oracle_inliers_median": float(
                oracle_cases["sift_selected_minus_oracle_inliers"].median()
            ),
            "top1_top2_inlier_margin_median": float(
                oracle_cases["sift_top1_top2_inlier_margin"].median()
            ),
        },
        "scientific_boundary": {
            "frozen_evidence_only": True,
            "matcher_rerun": False,
            "threshold_tuned": False,
            "selector_changed": False,
            "confidence_rule_trained": False,
            "blind_data_loaded": False,
        },
        "next_diagnostic_need": {
            "targeted_pair_rerun_required_for_spatial_inlier_support": True,
            "reason": (
                "Frozen SIFT evidence retains match/inlier counts, LightGlue score summaries, "
                "reprojection/Jacobian/footprint diagnostics, but not inlier coordinate distributions."
            ),
        },
    }

    outroot = resolve(cfg["output_root"])
    outroot.mkdir(parents=True, exist_ok=True)
    cases_path = outroot / "fa1_0_case_inventory.csv"
    detail_path = outroot / "fa1_0_case_top20_candidate_detail.csv"
    temporal_path = outroot / "fa1_0_temporal_context_pm2.csv"
    report_path = outroot / "fa1_0_report.json"

    cases.to_csv(cases_path, index=False)
    detail.to_csv(detail_path, index=False)
    temporal.to_csv(temporal_path, index=False)
    report_path.write_text(json.dumps(summary, indent=2, allow_nan=False))

    print("Status:", summary["status"])
    print("Oracle-rescuable SIFT failures:", observed_oracle)
    print("Catastrophic SIFT damages:", observed_cat)
    print("Severe ORB<=40 -> SIFT unsafe:", observed_severe)
    print("Unique union queries:", len(case_qids))
    print("Intersection oracle-rescuable & catastrophic-damage:", overlap)
    print("Oracle-rescuable only:", oracle_only)
    print("Catastrophic-damage only:", cat_only)
    print()
    print("Oracle-rescuable good-candidate selector rank median/max:",
          f"{summary['oracle_rescuable_diagnostics']['oracle_selector_rank_median']:.1f}/"
          f"{summary['oracle_rescuable_diagnostics']['oracle_selector_rank_max']}")
    print("Oracle-rescuable good-candidate pool-rank median:",
          f"{summary['oracle_rescuable_diagnostics']['oracle_pool_rank_median']:.1f}")
    print("Selected - oracle inlier deficit median:",
          f"{summary['oracle_rescuable_diagnostics']['selected_minus_oracle_inliers_median']:.1f}")
    print("Selected Top1-Top2 inlier margin median:",
          f"{summary['oracle_rescuable_diagnostics']['top1_top2_inlier_margin_median']:.1f}")
    print()
    print("Case inventory:", cases_path)
    print("Top20 detail:", detail_path)
    print("Temporal +/-2 context:", temporal_path)
    print("Report:", report_path)
    print("Matcher rerun / tuning / selector change: False")


if __name__ == "__main__":
    main()
