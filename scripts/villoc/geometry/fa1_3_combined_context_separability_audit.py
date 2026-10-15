#!/usr/bin/env python3
"""FA1.3 combined temporal, neighborhood, cross-frontend and separability audit.

All context features are computable without GT. Development truth is used only
to label outcome groups and identify the frozen FA1 selected/oracle cases.
No thresholds are tuned and no confidence rule is trained.
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
DEFAULT_CONFIG = ROOT / "configs/research/crossview_scale_rotation_domain_fa1_3.yaml"
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


def bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y", "t"})


def finite_xy(row: pd.Series, x: str = "projected_easting", y: str = "projected_northing"):
    try:
        a, b = float(row[x]), float(row[y])
    except Exception:
        return None
    return (a, b) if math.isfinite(a) and math.isfinite(b) else None


def distance(a, b):
    if a is None or b is None:
        return math.nan
    return float(math.hypot(a[0] - b[0], a[1] - b[1]))


def error_bin(value) -> str:
    try:
        v = float(value)
    except Exception:
        return "unsafe"
    if not math.isfinite(v) or v > 200.0:
        return "unsafe"
    if v <= 40.0:
        return "le40"
    return "mid40_200"


def numeric(row: pd.Series, names: list[str], default=math.nan):
    for name in names:
        if name in row.index:
            try:
                value = float(row[name])
            except Exception:
                continue
            return value
    return default


def rank_group(cv25, group: pd.DataFrame) -> pd.DataFrame:
    out = cv25.rank(group).reset_index(drop=True).copy()
    out["selector_rank"] = np.arange(1, len(out) + 1)
    return out


def candidate_support(group: pd.DataFrame, target: pd.Series, radii: list[float]) -> dict[str, Any]:
    tx = float(target["center_easting"])
    ty = float(target["center_northing"])
    centers = group[["center_easting", "center_northing"]].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(float)
    ranks = pd.to_numeric(group["pool_rank"], errors="coerce").to_numpy(float)
    finite = np.isfinite(centers).all(axis=1)
    d = np.full(len(group), np.nan)
    d[finite] = np.hypot(centers[finite, 0] - tx, centers[finite, 1] - ty)

    out = {}
    for radius in radii:
        tag = str(radius).replace(".", "p")
        inside = np.isfinite(d) & (d <= radius + 1e-9)
        self_mask = group["tile_id"].astype(str).to_numpy() == str(target["tile_id"])
        others = inside & ~self_mask
        top5 = others & (ranks <= 5)
        rr = np.where(others, 1.0 / np.maximum(ranks, 1.0), 0.0)
        out[f"support_{tag}m_other_tiles"] = int(others.sum())
        out[f"support_{tag}m_top5_other_tiles"] = int(top5.sum())
        out[f"support_{tag}m_reciprocal_rank_sum"] = float(rr.sum())
    return out


def temporal_context(target_xy, qid: int, selected_lookup: dict[int, tuple[float, float] | None], radius: int):
    distances = []
    offsets = []
    for offset in range(-radius, radius + 1):
        if offset == 0:
            continue
        neighbor = selected_lookup.get(qid + offset)
        d = distance(target_xy, neighbor)
        if math.isfinite(d):
            distances.append(d)
            offsets.append(offset)
    if not distances:
        return {
            "temporal_pm2_count": 0,
            "temporal_pm2_min_m": math.nan,
            "temporal_pm2_median_m": math.nan,
            "temporal_pm2_max_m": math.nan,
        }
    arr = np.asarray(distances, float)
    return {
        "temporal_pm2_count": int(len(arr)),
        "temporal_pm2_min_m": float(np.min(arr)),
        "temporal_pm2_median_m": float(np.median(arr)),
        "temporal_pm2_max_m": float(np.max(arr)),
    }


def group_summary(features: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    rows = []
    groups = {
        "all403": pd.Series(True, index=features.index),
        "sift_le40": features["sift_error_bin"].eq("le40"),
        "sift_mid40_200": features["sift_error_bin"].eq("mid40_200"),
        "sift_unsafe": features["sift_error_bin"].eq("unsafe"),
        "fa1_union": features["fa1_union"].astype(bool),
        "oracle_rescuable": features["oracle_rescuable"].astype(bool),
        "catastrophic_damage": features["catastrophic_damage"].astype(bool),
        "severe_regression": features["severe_regression"].astype(bool),
    }
    for group_name, mask in groups.items():
        block = features.loc[mask]
        for metric in metrics:
            vals = pd.to_numeric(block[metric], errors="coerce")
            vals = vals[np.isfinite(vals)]
            rows.append({
                "group": group_name,
                "metric": metric,
                "group_queries": int(len(block)),
                "finite_count": int(len(vals)),
                "median": float(vals.median()) if len(vals) else math.nan,
                "q25": float(vals.quantile(0.25)) if len(vals) else math.nan,
                "q75": float(vals.quantile(0.75)) if len(vals) else math.nan,
            })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = ap.parse_args()

    cfg_path = resolve(args.config)
    cfg = yaml.safe_load(cfg_path.read_text())
    if cfg.get("stage") != "FA1.3":
        raise RuntimeError("Expected FA1.3 config")

    boundary = cfg["scientific_boundary"]
    if boundary["postfreeze_development_truth_allowed_for_labels"] is not True:
        raise RuntimeError("Truth-label boundary missing")
    if boundary["context_features_reference_free"] is not True:
        raise RuntimeError("Context features must remain reference-free")
    for key in (
        "frozen_evidence_mutable", "matcher_rerun_allowed", "selector_change_allowed",
        "threshold_tuning_allowed", "confidence_rule_training_allowed",
        "blind_data_loaded", "temporal_fusion_changed",
    ):
        if boundary[key] is not False:
            raise RuntimeError(f"{key} must remain false")

    cv25 = dynamic_import("fa13_cv25", CV25_SCRIPT)
    cv25_cfg_path = resolve(cfg["cv2_5"]["config"])
    cv25_cfg = yaml.safe_load(cv25_cfg_path.read_text())
    cv25.verify_config(cv25_cfg)
    _, scale_raw, sift_raw = cv25.verify_inputs(cv25_cfg)

    sift_path = cv25.resolve(cv25_cfg["sift_lightglue"]["evidence"])
    scale_path = cv25.resolve(cv25_cfg["scale_m2_orb"]["evidence"])
    if cv25.sha256_file(sift_path) != cfg["cv2_5"]["expected_sift_sha256"]:
        raise RuntimeError("SIFT frozen SHA mismatch")
    if cv25.sha256_file(scale_path) != cfg["cv2_5"]["expected_scale_m2_sha256"]:
        raise RuntimeError("scale_m2 frozen SHA mismatch")

    fa0_report = json.loads(resolve(cfg["fa1_0"]["report"]).read_text())
    if fa0_report.get("status") != cfg["fa1_0"]["expected_status"]:
        raise RuntimeError("FA1.0 report not PASS")
    expected_counts = {
        "union_unique_queries": cfg["fa1_0"]["expected_union_queries"],
        "oracle_rescuable": cfg["fa1_0"]["expected_oracle_rescuable"],
        "catastrophic_damage": cfg["fa1_0"]["expected_catastrophic_damage"],
        "severe_orb_le40_to_sift_unsafe": cfg["fa1_0"]["expected_severe_regression"],
    }
    for key, expected in expected_counts.items():
        if int(fa0_report["counts"][key]) != int(expected):
            raise RuntimeError(f"FA1.0 count changed: {key}")

    fa11_report = json.loads(resolve(cfg["fa1_1"]["report"]).read_text())
    if fa11_report.get("status") != cfg["fa1_1"]["expected_status"]:
        raise RuntimeError("FA1.1 report not PASS")
    if not fa11_report["parity"]["all_sift_pairs_match_frozen_evidence"]:
        raise RuntimeError("FA1.1 SIFT parity not PASS")
    if not fa11_report["parity"]["all_scale_m2_pairs_match_frozen_evidence"]:
        raise RuntimeError("FA1.1 scale_m2 parity not PASS")

    cases = pd.read_csv(resolve(cfg["fa1_0"]["case_inventory"]))
    cases["query_id"] = pd.to_numeric(cases["query_id"], errors="raise").astype(int)
    for col in ("oracle_rescuable", "catastrophic_damage", "severe_orb_le40_to_sift_unsafe"):
        cases[col] = bool_series(cases[col])
    case_lookup = cases.set_index("query_id")

    tile_path = cv25.verify_tiles(cv25_cfg)
    tiles = pd.read_csv(tile_path)
    ref_path = cv25.resolve(cv25_cfg["development"]["reference_attachment"])
    reference = cv25.load_reference_xy(ref_path)

    sift = cv25.attach(sift_raw, tiles, reference, "sift_lightglue")
    scale = cv25.attach(scale_raw, tiles, reference, "scale_m2_orb")

    radii = [float(x) for x in cfg["context"]["neighborhood_radii_m"]]
    temporal_radius = int(cfg["context"]["temporal_radius_queries"])

    # Freeze the current selector outputs for both frontends.
    sift_selected = {}
    scale_selected = {}
    sift_ranked = {}
    for qid, group in sift.groupby("query_id", sort=True):
        ranked = rank_group(cv25, group)
        sift_ranked[int(qid)] = ranked
        sift_selected[int(qid)] = ranked.iloc[0]
    for qid, group in scale.groupby("query_id", sort=True):
        ranked = rank_group(cv25, group)
        scale_selected[int(qid)] = ranked.iloc[0]

    selected_xy = {
        qid: finite_xy(row)
        for qid, row in sift_selected.items()
    }

    all_rows = []
    for qid in sorted(sift_selected):
        selected = sift_selected[qid]
        ranked = sift_ranked[qid]
        scale_row = scale_selected[qid]
        selected_point = finite_xy(selected)
        scale_point = finite_xy(scale_row)

        max_inliers = float(selected["inliers"])
        tie_count = int(
            (pd.to_numeric(ranked["inliers"], errors="coerce") == max_inliers).sum()
        )
        second_inliers = float(ranked.iloc[1]["inliers"]) if len(ranked) > 1 else math.nan
        rank1_row = ranked[pd.to_numeric(ranked["pool_rank"], errors="coerce") == 1]
        rank1_inliers = (
            float(rank1_row.iloc[0]["inliers"]) if len(rank1_row) == 1 else math.nan
        )

        support = candidate_support(ranked, selected, radii)
        temporal = temporal_context(selected_point, qid, selected_xy, temporal_radius)

        projection_error = numeric(selected, ["projection_error_m"])
        case = case_lookup.loc[qid] if qid in case_lookup.index else None

        all_rows.append({
            "query_id": int(qid),
            "sift_error_bin": error_bin(projection_error),
            "sift_projection_error_m_truth_label_only": projection_error,
            "fa1_union": bool(case is not None),
            "oracle_rescuable": bool(case["oracle_rescuable"]) if case is not None else False,
            "catastrophic_damage": bool(case["catastrophic_damage"]) if case is not None else False,
            "severe_regression": bool(case["severe_orb_le40_to_sift_unsafe"]) if case is not None else False,

            # Reference-free selected-pair / retrieval conflict features.
            "sift_selected_tile_id": str(selected["tile_id"]),
            "sift_selected_pool_rank": int(selected["pool_rank"]),
            "sift_selected_inliers": max_inliers,
            "sift_selected_good_matches": numeric(selected, ["good_matches", "evidence_matches"]),
            "sift_selected_inlier_ratio": numeric(selected, ["evidence_inlier_ratio"]),
            "sift_selected_symmetric_error_median_px": numeric(
                selected, ["evidence_symmetric_error_median_px"]
            ),
            "sift_selected_jacobian_anisotropy": numeric(
                selected, ["evidence_jacobian_anisotropy"]
            ),
            "sift_selected_footprint_area_fraction_tile": numeric(
                selected, ["evidence_footprint_area_fraction_tile"]
            ),
            "sift_top1_top2_inlier_margin": max_inliers - second_inliers,
            "sift_max_inlier_tie_count": tie_count,
            "sift_dino_rank1_inliers": rank1_inliers,
            "sift_selected_minus_dino_rank1_inliers": (
                max_inliers - rank1_inliers
                if math.isfinite(rank1_inliers) else math.nan
            ),

            # Cross-frontend disagreement is reference-free.
            "scale_m2_selected_tile_id": str(scale_row["tile_id"]),
            "cross_frontend_same_tile": str(selected["tile_id"]) == str(scale_row["tile_id"]),
            "cross_frontend_projected_distance_m": distance(selected_point, scale_point),

            **support,
            **temporal,
        })

    features = pd.DataFrame(all_rows).sort_values("query_id").reset_index(drop=True)
    if len(features) != int(cfg["cv2_5"]["expected_queries"]):
        raise RuntimeError("FA1.3 all403 feature count changed")

    # Selected vs oracle context for the 20 frozen cases.
    case_context_rows = []
    for case in cases.sort_values("query_id").itertuples(index=False):
        qid = int(case.query_id)
        ranked = sift_ranked[qid]
        selected = sift_selected[qid]
        oracle_tile = str(case.sift_oracle_tile_id)
        oracle_rows = ranked[ranked["tile_id"].astype(str) == oracle_tile]
        if len(oracle_rows) != 1:
            raise RuntimeError(f"Oracle candidate missing for q{qid}: {oracle_tile}")
        oracle = oracle_rows.iloc[0]
        scale_row = scale_selected[qid]

        selected_point = finite_xy(selected)
        oracle_point = finite_xy(oracle)
        scale_point = finite_xy(scale_row)

        selected_support = candidate_support(ranked, selected, radii)
        oracle_support = candidate_support(ranked, oracle, radii)
        selected_temporal = temporal_context(selected_point, qid, selected_xy, temporal_radius)
        oracle_temporal = temporal_context(oracle_point, qid, selected_xy, temporal_radius)

        row = {
            "query_id": qid,
            "oracle_rescuable": bool(case.oracle_rescuable),
            "catastrophic_damage": bool(case.catastrophic_damage),
            "severe_regression": bool(case.severe_orb_le40_to_sift_unsafe),

            "selected_tile_id": str(selected["tile_id"]),
            "oracle_tile_id": oracle_tile,
            "scale_m2_tile_id": str(scale_row["tile_id"]),

            "selected_pool_rank": int(selected["pool_rank"]),
            "oracle_pool_rank": int(oracle["pool_rank"]),
            "oracle_selector_rank": int(oracle["selector_rank"]),
            "selected_inliers": float(selected["inliers"]),
            "oracle_inliers": float(oracle["inliers"]),
            "selected_inlier_ratio": numeric(selected, ["evidence_inlier_ratio"]),
            "oracle_inlier_ratio": numeric(oracle, ["evidence_inlier_ratio"]),
            "selected_symmetric_error_px": numeric(
                selected, ["evidence_symmetric_error_median_px"]
            ),
            "oracle_symmetric_error_px": numeric(
                oracle, ["evidence_symmetric_error_median_px"]
            ),

            "selected_to_scale_m2_distance_m": distance(selected_point, scale_point),
            "oracle_to_scale_m2_distance_m": distance(oracle_point, scale_point),
            "orb_closer_to_oracle_than_selected": (
                distance(oracle_point, scale_point) < distance(selected_point, scale_point)
                if math.isfinite(distance(oracle_point, scale_point))
                and math.isfinite(distance(selected_point, scale_point))
                else False
            ),

            "selected_temporal_pm2_median_m": selected_temporal["temporal_pm2_median_m"],
            "oracle_temporal_pm2_median_m": oracle_temporal["temporal_pm2_median_m"],
            "oracle_more_temporally_consistent": (
                oracle_temporal["temporal_pm2_median_m"] < selected_temporal["temporal_pm2_median_m"]
                if math.isfinite(oracle_temporal["temporal_pm2_median_m"])
                and math.isfinite(selected_temporal["temporal_pm2_median_m"])
                else False
            ),
        }

        for radius in radii:
            tag = str(radius).replace(".", "p")
            s_count = selected_support[f"support_{tag}m_other_tiles"]
            o_count = oracle_support[f"support_{tag}m_other_tiles"]
            s_rr = selected_support[f"support_{tag}m_reciprocal_rank_sum"]
            o_rr = oracle_support[f"support_{tag}m_reciprocal_rank_sum"]
            row[f"selected_support_{tag}m_other_tiles"] = s_count
            row[f"oracle_support_{tag}m_other_tiles"] = o_count
            row[f"oracle_more_support_{tag}m"] = bool(o_count > s_count)
            row[f"selected_support_{tag}m_reciprocal_rank_sum"] = s_rr
            row[f"oracle_support_{tag}m_reciprocal_rank_sum"] = o_rr
            row[f"oracle_more_rr_support_{tag}m"] = bool(o_rr > s_rr)

        case_context_rows.append(row)

    case_context = pd.DataFrame(case_context_rows)

    metrics = [
        "sift_selected_pool_rank",
        "sift_selected_inliers",
        "sift_selected_inlier_ratio",
        "sift_selected_symmetric_error_median_px",
        "sift_top1_top2_inlier_margin",
        "sift_max_inlier_tie_count",
        "sift_selected_minus_dino_rank1_inliers",
        "cross_frontend_projected_distance_m",
        "support_51p2m_other_tiles",
        "support_102p4m_other_tiles",
        "support_51p2m_reciprocal_rank_sum",
        "support_102p4m_reciprocal_rank_sum",
        "temporal_pm2_median_m",
    ]
    summary_table = group_summary(features, metrics)

    oracle_cases = case_context[case_context["oracle_rescuable"]].copy()
    context_summary = {
        "oracle_rescuable_cases": int(len(oracle_cases)),
        "oracle_more_temporally_consistent": int(
            oracle_cases["oracle_more_temporally_consistent"].sum()
        ),
        "orb_closer_to_oracle_than_selected": int(
            oracle_cases["orb_closer_to_oracle_than_selected"].sum()
        ),
        "oracle_more_support_51p2m": int(
            oracle_cases["oracle_more_support_51p2m"].sum()
        ),
        "oracle_more_support_102p4m": int(
            oracle_cases["oracle_more_support_102p4m"].sum()
        ),
        "oracle_more_rr_support_51p2m": int(
            oracle_cases["oracle_more_rr_support_51p2m"].sum()
        ),
        "oracle_more_rr_support_102p4m": int(
            oracle_cases["oracle_more_rr_support_102p4m"].sum()
        ),
        "selected_pool_rank_median": float(oracle_cases["selected_pool_rank"].median()),
        "oracle_pool_rank_median": float(oracle_cases["oracle_pool_rank"].median()),
        "oracle_selector_rank_median": float(oracle_cases["oracle_selector_rank"].median()),
        "selected_inlier_ratio_median": float(oracle_cases["selected_inlier_ratio"].median()),
        "oracle_inlier_ratio_median": float(oracle_cases["oracle_inlier_ratio"].median()),
        "selected_symmetric_error_median_px": float(
            oracle_cases["selected_symmetric_error_px"].median()
        ),
        "oracle_symmetric_error_median_px": float(
            oracle_cases["oracle_symmetric_error_px"].median()
        ),
    }

    # Selected rows from group summary for concise console/report comparison.
    def med(group, metric):
        row = summary_table[
            (summary_table["group"] == group)
            & (summary_table["metric"] == metric)
        ]
        return float(row.iloc[0]["median"]) if len(row) else math.nan

    key_group_comparison = {}
    for metric in metrics:
        key_group_comparison[metric] = {
            "sift_le40_median": med("sift_le40", metric),
            "sift_unsafe_median": med("sift_unsafe", metric),
            "fa1_union_median": med("fa1_union", metric),
        }

    report = {
        "stage": "FA1.3",
        "status": "PASS_FA1_3_COMBINED_CONTEXT_SEPARABILITY_AUDIT",
        "counts": {
            "all_queries": int(len(features)),
            "fa1_union_queries": int(features["fa1_union"].sum()),
            "oracle_rescuable": int(features["oracle_rescuable"].sum()),
            "catastrophic_damage": int(features["catastrophic_damage"].sum()),
            "severe_regression": int(features["severe_regression"].sum()),
            "sift_le40": int(features["sift_error_bin"].eq("le40").sum()),
            "sift_mid40_200": int(features["sift_error_bin"].eq("mid40_200").sum()),
            "sift_unsafe": int(features["sift_error_bin"].eq("unsafe").sum()),
        },
        "oracle_rescuable_context": context_summary,
        "group_median_comparison": key_group_comparison,
        "manual_visual_observation_from_fa1_2": (
            "User review reports the failure contact sheet is predominantly "
            "vegetation-heavy or mixed vegetation/limited-urban coverage, with "
            "no prominent street-heavy or dense-urban failure cluster."
        ),
        "scientific_boundary": {
            "context_features_reference_free": True,
            "development_truth_used_for_labels_only": True,
            "matcher_rerun": False,
            "selector_changed": False,
            "threshold_tuned": False,
            "confidence_rule_trained": False,
            "blind_data_loaded": False,
            "temporal_fusion_changed": False,
        },
        "interpretation_boundary": (
            "Feature differences are descriptive failure-analysis evidence only. "
            "They do not define a deployable confidence threshold or selector."
        ),
    }

    outroot = resolve(cfg["output_root"])
    outroot.mkdir(parents=True, exist_ok=True)
    features_path = outroot / "fa1_3_all403_reference_free_context_features.csv"
    cases_path = outroot / "fa1_3_failure_selected_vs_oracle_context.csv"
    summary_path = outroot / "fa1_3_group_descriptive_summary.csv"
    report_path = outroot / "fa1_3_report.json"

    features.to_csv(features_path, index=False)
    case_context.to_csv(cases_path, index=False)
    summary_table.to_csv(summary_path, index=False)
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False))

    print("Status:", report["status"])
    print("All403 / FA1 union:", f"{len(features)}/{int(features['fa1_union'].sum())}")
    print("SIFT error bins <=40 / 40-200 / unsafe:",
          f"{report['counts']['sift_le40']}/"
          f"{report['counts']['sift_mid40_200']}/"
          f"{report['counts']['sift_unsafe']}")
    print()
    print("Oracle-rescuable context (of 19):")
    for key in (
        "oracle_more_temporally_consistent",
        "orb_closer_to_oracle_than_selected",
        "oracle_more_support_51p2m",
        "oracle_more_support_102p4m",
        "oracle_more_rr_support_51p2m",
        "oracle_more_rr_support_102p4m",
    ):
        print(f"  {key}: {context_summary[key]}/19")
    print()
    print("Oracle-rescuable selected/oracle medians:")
    print("  DINO pool rank:",
          f"{context_summary['selected_pool_rank_median']:.1f} / "
          f"{context_summary['oracle_pool_rank_median']:.1f}")
    print("  oracle selector rank:",
          f"{context_summary['oracle_selector_rank_median']:.1f}")
    print("  inlier ratio selected/oracle:",
          f"{context_summary['selected_inlier_ratio_median']:.3f} / "
          f"{context_summary['oracle_inlier_ratio_median']:.3f}")
    print("  symmetric error px selected/oracle:",
          f"{context_summary['selected_symmetric_error_median_px']:.3f} / "
          f"{context_summary['oracle_symmetric_error_median_px']:.3f}")
    print()
    print("Key reference-free feature medians: <=40 / unsafe / FA1-union")
    for metric in metrics:
        item = key_group_comparison[metric]
        print(
            f"  {metric}: "
            f"{item['sift_le40_median']} / "
            f"{item['sift_unsafe_median']} / "
            f"{item['fa1_union_median']}"
        )
    print()
    print("All403 context features:", features_path)
    print("Failure selected-vs-oracle context:", cases_path)
    print("Group descriptive summary:", summary_path)
    print("Report:", report_path)
    print("Matcher rerun / tuning / selector change / confidence training: False")


if __name__ == "__main__":
    main()
