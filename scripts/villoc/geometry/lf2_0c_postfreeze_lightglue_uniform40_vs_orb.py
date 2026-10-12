#!/usr/bin/env python3
"""LF2.0-C: post-freeze LightGlue uniform40 vs ORB screening audit."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
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

from uavloc.evaluation.reference import load_reference_xy
from scripts.villoc.geometry.lf1_1b_postfreeze_orb_xfeat_comparison import (
    attach_reference,
    diagnostic_feature_ranking,
    homography_retention,
    projection_summary,
    verify_tile_geometry,
)

ROOT = REPO_ROOT
DEFAULT_CONFIG = ROOT / "configs/research/learned_geometric_frontend_lf2_0c.yaml"


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
        index=False,
        lineterminator="\n",
        float_format="%.9f",
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
            "Refusing to overwrite differing LF2.0-C outputs: "
            + ", ".join(changed)
        )
    for path, data in files.items():
        if path.exists():
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as handle:
            handle.write(data)


def verify_config(cfg: dict[str, Any]) -> None:
    if cfg.get("stage") != "LF2.0-C":
        raise RuntimeError("Expected LF2.0-C config")
    if cfg.get("labels") != [
        "contains_query",
        "center_le40",
        "center_le80",
    ]:
        raise RuntimeError("Label contract changed")
    if cfg["ranking"].get("orb_primary_selector") != "inliers":
        raise RuntimeError("ORB primary selector changed")
    if cfg["ranking"].get("lightglue_primary_selector") != "inliers":
        raise RuntimeError("LightGlue primary selector changed")
    expected_secondary = [
        "evidence_match_score_sum",
        "evidence_match_score_mean",
        "evidence_match_score_median",
        "evidence_inlier_ratio",
    ]
    if cfg["ranking"].get("lightglue_secondary_scores") != expected_secondary:
        raise RuntimeError("LightGlue secondary score contract changed")

    gate = cfg["primary_full_run_gate"]
    expected_gate = {
        "max_top1_regression_queries_each_target": 2,
        "min_top1_improvement_queries_any_target": 2,
        "max_selected_projection_le40_regression_queries": 2,
        "max_selected_projection_gt200_increase_queries": 2,
    }
    for key, value in expected_gate.items():
        if int(gate.get(key, -999)) != value:
            raise RuntimeError(f"Primary gate changed: {key}")

    secondary = cfg["secondary_formulation_gate"]
    expected_secondary_gate = {
        "min_top1_gain_over_lightglue_inliers_on_two_targets": 4,
        "max_top1_gap_to_orb_each_target": 2,
    }
    for key, value in expected_secondary_gate.items():
        if int(secondary.get(key, -999)) != value:
            raise RuntimeError(f"Secondary gate changed: {key}")

    boundary = cfg["scientific_boundary"]
    if boundary.get("development_reference_allowed") is not True:
        raise RuntimeError("Development reference must be explicitly allowed")
    if boundary.get("reference_after_screening_freeze_only") is not True:
        raise RuntimeError("Reference-after-freeze contract missing")
    for key in (
        "blind_stress_loaded",
        "blind_stress_reference_allowed",
        "full_403_lightglue_loaded",
        "threshold_tuning_enabled",
        "selector_training_enabled",
        "frontend_parameters_mutable",
        "bootstrap_enabled",
        "state_enabled",
    ):
        if boundary.get(key) is not False:
            raise RuntimeError(f"{key} must be false")


def verify_inputs(
    cfg: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, list[int], dict[str, Any]]:
    lg_block = cfg["lightglue"]
    lg_report_path = resolve(lg_block["report"])
    lg_evidence_path = resolve(lg_block["evidence"])
    lg_report = json.loads(lg_report_path.read_text(encoding="utf-8"))
    if lg_report.get("status") != (
        "PASS_LF2_0B_REFERENCE_FREE_LIGHTGLUE_UNIFORM40_SCREENING_FREEZE"
    ):
        raise RuntimeError("LF2.0-B report not PASS")
    if lg_report["development"]["output"]["sha256"] != (
        lg_block["expected_evidence_sha256"]
    ):
        raise RuntimeError("LF2.0-B report evidence hash mismatch")
    if sha256_file(lg_evidence_path) != lg_block["expected_evidence_sha256"]:
        raise RuntimeError("LF2.0-B evidence bytes changed")
    if lg_report["scientific_boundary"].get("reference_loaded") is not False:
        raise RuntimeError("LF2.0-B reference boundary violated")

    selected_ids = [
        int(value)
        for value in lg_report["screening_contract"]["selected_query_ids"]
    ]
    if len(selected_ids) != int(lg_block["expected_queries"]):
        raise RuntimeError("LF2.0-B selected query count mismatch")
    if len(set(selected_ids)) != len(selected_ids):
        raise RuntimeError("LF2.0-B selected query IDs are not unique")

    lightglue = pd.read_csv(lg_evidence_path)
    if len(lightglue) != int(lg_block["expected_pairs"]):
        raise RuntimeError("LF2.0-B pair count mismatch")
    if lightglue["query_id"].nunique() != int(lg_block["expected_queries"]):
        raise RuntimeError("LF2.0-B query count mismatch")
    if sorted(lightglue["query_id"].astype(int).unique().tolist()) != sorted(selected_ids):
        raise RuntimeError("LF2.0-B evidence query set mismatch")

    orb_block = cfg["orb"]
    orb_report_path = resolve(orb_block["report"])
    orb_evidence_path = resolve(orb_block["evidence"])
    orb_report = json.loads(orb_report_path.read_text(encoding="utf-8"))
    if orb_report.get("status") != (
        "PASS_GD1_0_REFERENCE_FREE_TOP20_GEOMETRIC_EVIDENCE_FREEZE"
    ):
        raise RuntimeError("ORB/GD1.0 report not PASS")
    if orb_report["trajectories"]["development"]["output"]["sha256"] != (
        orb_block["expected_evidence_sha256"]
    ):
        raise RuntimeError("ORB report evidence hash mismatch")
    if sha256_file(orb_evidence_path) != orb_block["expected_evidence_sha256"]:
        raise RuntimeError("ORB evidence bytes changed")
    if orb_report["scientific_boundary"].get("reference_loaded") is not False:
        raise RuntimeError("ORB/GD1.0 reference boundary violated")

    orb_full = pd.read_csv(orb_evidence_path)
    orb = orb_full[orb_full["query_id"].astype(int).isin(selected_ids)].copy()
    orb = orb.sort_values(["query_id", "pool_rank"], kind="mergesort").reset_index(drop=True)
    lightglue = lightglue.sort_values(
        ["query_id", "pool_rank"], kind="mergesort"
    ).reset_index(drop=True)

    key_cols = ["query_id", "pool_rank", "tile_id"]
    if not orb[key_cols].equals(lightglue[key_cols]):
        raise RuntimeError("ORB and LightGlue screening pair ordering differs")

    return orb, lightglue, selected_ids, lg_report


def rank_by_score(
    group: pd.DataFrame,
    score_column: str,
) -> pd.DataFrame:
    work = group.copy()
    work["_score"] = pd.to_numeric(
        work[score_column], errors="coerce"
    ).replace([np.inf, -np.inf], np.nan).fillna(-np.inf)
    return work.sort_values(
        ["_score", "pool_rank", "tile_id"],
        ascending=[False, True, True],
        kind="mergesort",
    )


def ranking_summary(
    frame: pd.DataFrame,
    label: str,
    score_column: str,
    topks: list[int],
) -> dict[str, Any]:
    hits = {f"top{k}": 0 for k in topks}
    for _, group in frame.groupby("query_id", sort=True):
        ordered = rank_by_score(group, score_column)
        for k in topks:
            hits[f"top{k}"] += int(
                ordered.head(k)[label].astype(bool).any()
            )
    return {
        "queries": int(frame["query_id"].nunique()),
        "hits": hits,
    }


def selected_projection_rows(
    frame: pd.DataFrame,
    score_column: str,
    selector_name: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for qid, group in frame.groupby("query_id", sort=True):
        selected = rank_by_score(group, score_column).iloc[0]
        rows.append(
            {
                "frontend": str(selected["frontend"]),
                "query_id": int(qid),
                "selector": selector_name,
                "pool_rank": int(selected["pool_rank"]),
                "tile_id": str(selected["tile_id"]),
                "homography_ok": bool(selected["homography_ok"]),
                "score": float(
                    pd.to_numeric(
                        pd.Series([selected[score_column]]),
                        errors="coerce",
                    ).iloc[0]
                )
                if pd.notna(selected[score_column])
                else np.nan,
                "inliers": float(selected["inliers"]),
                "projection_error_m": float(
                    selected["projection_error_m"]
                )
                if pd.notna(selected["projection_error_m"])
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def evaluate_primary_gate(
    orb_top1: dict[str, int],
    lg_top1: dict[str, int],
    orb_projection: dict[str, Any],
    lg_projection: dict[str, Any],
    gate: dict[str, Any],
) -> dict[str, Any]:
    deltas = {
        label: int(lg_top1[label] - orb_top1[label])
        for label in orb_top1
    }
    ranking_pass = (
        all(
            delta >= -int(gate["max_top1_regression_queries_each_target"])
            for delta in deltas.values()
        )
        and max(deltas.values())
        >= int(gate["min_top1_improvement_queries_any_target"])
    )

    le40_delta = int(lg_projection["le40"] - orb_projection["le40"])
    gt200_delta = int(lg_projection["gt200"] - orb_projection["gt200"])
    projection_pass = (
        le40_delta
        >= -int(gate["max_selected_projection_le40_regression_queries"])
        and gt200_delta
        <= int(gate["max_selected_projection_gt200_increase_queries"])
    )

    return {
        "top1_deltas_lightglue_minus_orb": deltas,
        "ranking_pass": bool(ranking_pass),
        "selected_projection_le40_delta": le40_delta,
        "selected_projection_gt200_delta": gt200_delta,
        "projection_pass": bool(projection_pass),
        "pass": bool(ranking_pass and projection_pass),
    }


def evaluate_secondary_gate(
    orb_top1: dict[str, int],
    lg_inlier_top1: dict[str, int],
    secondary_top1: dict[str, int],
    gate: dict[str, Any],
) -> dict[str, Any]:
    gain_threshold = int(
        gate["min_top1_gain_over_lightglue_inliers_on_two_targets"]
    )
    max_gap = int(gate["max_top1_gap_to_orb_each_target"])
    gains = {
        label: int(secondary_top1[label] - lg_inlier_top1[label])
        for label in orb_top1
    }
    gaps_to_orb = {
        label: int(orb_top1[label] - secondary_top1[label])
        for label in orb_top1
    }
    gain_targets = [
        label for label, gain in gains.items()
        if gain >= gain_threshold
    ]
    pass_gate = (
        len(gain_targets) >= 2
        and all(gap <= max_gap for gap in gaps_to_orb.values())
    )
    return {
        "top1_gains_over_lightglue_inliers": gains,
        "targets_with_required_gain": gain_targets,
        "top1_gaps_to_orb": gaps_to_orb,
        "pass": bool(pass_gate),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LF2.0-C post-freeze LightGlue uniform40 vs ORB screening audit"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()

    cfg_path = resolve(args.config)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    verify_config(cfg)

    orb_raw, lg_raw, selected_ids, lg_freeze_report = verify_inputs(cfg)
    tile_path = verify_tile_geometry(
        {
            "map": {
                "rg2_0_report": cfg["map"]["rg2_0_report"],
                "tile_geometry": cfg["map"]["tile_geometry"],
            }
        }
    )

    for spec in cfg["common_features"]:
        if spec["orb"] not in orb_raw.columns:
            raise RuntimeError(f"ORB missing common feature {spec['orb']}")
        if spec["lightglue"] not in lg_raw.columns:
            raise RuntimeError(
                f"LightGlue missing common feature {spec['lightglue']}"
            )
    for column in cfg["ranking"]["lightglue_secondary_scores"]:
        if column not in lg_raw.columns:
            raise RuntimeError(f"LightGlue missing native score {column}")

    # Critical boundary: both frozen evidence sets are verified before truth is opened.
    reference_path = resolve(cfg["development"]["reference_attachment"])
    print("ORB and LightGlue screening freezes verified before reference attachment.")
    print("Selected screening query IDs:", ",".join(map(str, selected_ids)))
    print("Attaching development reference for LF2.0-C diagnostics only.")
    reference = load_reference_xy(reference_path)
    tile = pd.read_csv(tile_path)

    orb = attach_reference(orb_raw, tile, reference, frontend="orb")
    lightglue = attach_reference(
        lg_raw, tile, reference, frontend="lightglue"
    )

    labels = list(cfg["labels"])
    topks = [int(value) for value in cfg["ranking"]["topk"]]

    rankings: dict[str, Any] = {
        "orb_inliers": {},
        "lightglue_inliers": {},
        "lightglue_secondary": {},
    }
    for label in labels:
        rankings["orb_inliers"][label] = ranking_summary(
            orb, label, "inliers", topks
        )
        rankings["lightglue_inliers"][label] = ranking_summary(
            lightglue, label, "inliers", topks
        )

    for score in cfg["ranking"]["lightglue_secondary_scores"]:
        rankings["lightglue_secondary"][score] = {}
        for label in labels:
            rankings["lightglue_secondary"][score][label] = (
                ranking_summary(lightglue, label, score, topks)
            )

    selected_frames: list[pd.DataFrame] = []
    orb_selected = selected_projection_rows(
        orb, "inliers", "orb_inliers"
    )
    lg_selected = selected_projection_rows(
        lightglue, "inliers", "lightglue_inliers"
    )
    selected_frames.extend([orb_selected, lg_selected])

    projection_summaries: dict[str, Any] = {
        "orb_inliers": projection_summary(
            orb_selected["projection_error_m"]
        ),
        "lightglue_inliers": projection_summary(
            lg_selected["projection_error_m"]
        ),
        "lightglue_secondary": {},
    }
    for score in cfg["ranking"]["lightglue_secondary_scores"]:
        selected = selected_projection_rows(
            lightglue, score, f"lightglue_{score}"
        )
        selected_frames.append(selected)
        projection_summaries["lightglue_secondary"][score] = (
            projection_summary(selected["projection_error_m"])
        )

    orb_top1 = {
        label: int(rankings["orb_inliers"][label]["hits"]["top1"])
        for label in labels
    }
    lg_top1 = {
        label: int(rankings["lightglue_inliers"][label]["hits"]["top1"])
        for label in labels
    }
    primary_gate = evaluate_primary_gate(
        orb_top1,
        lg_top1,
        projection_summaries["orb_inliers"],
        projection_summaries["lightglue_inliers"],
        cfg["primary_full_run_gate"],
    )

    secondary_gates: dict[str, Any] = {}
    for score in cfg["ranking"]["lightglue_secondary_scores"]:
        secondary_top1 = {
            label: int(
                rankings["lightglue_secondary"][score][label]["hits"]["top1"]
            )
            for label in labels
        }
        secondary_gates[score] = evaluate_secondary_gate(
            orb_top1,
            lg_top1,
            secondary_top1,
            cfg["secondary_formulation_gate"],
        )

    secondary_passers = [
        score for score, result in secondary_gates.items()
        if result["pass"]
    ]
    if primary_gate["pass"]:
        decision = "PROCEED_FULL_LIGHTGLUE"
    elif secondary_passers:
        decision = "PROCEED_LIGHTGLUE_SCORE_FORMULATION_STAGE"
    else:
        decision = "CLOSE_SUPERPOINT_LIGHTGLUE"

    feature_rows: list[dict[str, Any]] = []
    common_feature_report: dict[str, Any] = {}
    for spec in cfg["common_features"]:
        name = str(spec["name"])
        common_feature_report[name] = {}
        for frontend_name, frame, column in (
            ("orb", orb, str(spec["orb"])),
            ("lightglue", lightglue, str(spec["lightglue"])),
        ):
            common_feature_report[name][frontend_name] = {}
            for label in labels:
                diagnostic = diagnostic_feature_ranking(
                    frame, column, label, topks
                )
                common_feature_report[name][frontend_name][label] = diagnostic
                auc = diagnostic["pair_auc"]
                hits = diagnostic["query_hits"] or {}
                feature_rows.append(
                    {
                        "feature": name,
                        "frontend": frontend_name,
                        "source_column": column,
                        "label": label,
                        "auc": auc["auc"],
                        "auc_star": auc["auc_star"],
                        "direction": auc["direction"],
                        "top1": hits.get("top1"),
                        "top3": hits.get("top3"),
                        "top5": hits.get("top5"),
                        "top10": hits.get("top10"),
                    }
                )

    native_rows: list[dict[str, Any]] = []
    for score in cfg["ranking"]["lightglue_secondary_scores"]:
        for label in labels:
            diagnostic = diagnostic_feature_ranking(
                lightglue, score, label, topks
            )
            auc = diagnostic["pair_auc"]
            hits = diagnostic["query_hits"] or {}
            native_rows.append(
                {
                    "score": score,
                    "label": label,
                    "auc": auc["auc"],
                    "auc_star": auc["auc_star"],
                    "direction": auc["direction"],
                    "top1": hits.get("top1"),
                    "top3": hits.get("top3"),
                    "top5": hits.get("top5"),
                    "top10": hits.get("top10"),
                    "secondary_gate_pass": bool(
                        secondary_gates[score]["pass"]
                    ),
                }
            )

    labeled = pd.concat([orb, lightglue], ignore_index=True).sort_values(
        ["frontend", "query_id", "pool_rank"],
        kind="mergesort",
    ).reset_index(drop=True)
    selected = pd.concat(selected_frames, ignore_index=True).sort_values(
        ["selector", "query_id"], kind="mergesort"
    ).reset_index(drop=True)
    feature_diag = pd.DataFrame(feature_rows).sort_values(
        ["label", "feature", "frontend"], kind="mergesort"
    ).reset_index(drop=True)
    native_diag = pd.DataFrame(native_rows).sort_values(
        ["label", "score"], kind="mergesort"
    ).reset_index(drop=True)

    report: dict[str, Any] = {
        "stage": "LF2.0-C",
        "status": "PASS_LF2_0C_POSTFREEZE_LIGHTGLUE_UNIFORM40_VS_ORB_AUDIT",
        "decision": decision,
        "selected_query_ids": selected_ids,
        "rankings": rankings,
        "projection_summaries": projection_summaries,
        "homography_retention": {
            "orb": {
                label: homography_retention(orb, label)
                for label in labels
            },
            "lightglue": {
                label: homography_retention(lightglue, label)
                for label in labels
            },
        },
        "primary_full_run_gate": {
            "contract": cfg["primary_full_run_gate"],
            "result": primary_gate,
        },
        "secondary_formulation_gate": {
            "contract": cfg["secondary_formulation_gate"],
            "results": secondary_gates,
            "passing_scores": secondary_passers,
        },
        "common_feature_diagnostics": common_feature_report,
        "scientific_boundary": {
            "screening_freeze_verified_before_reference": True,
            "orb_freeze_verified_before_reference": True,
            "blind_stress_loaded": False,
            "blind_stress_reference_used": False,
            "full_403_lightglue_loaded": False,
            "threshold_tuning_executed": False,
            "selector_training_executed": False,
            "frontend_parameters_changed": False,
            "bootstrap_state_modified": False,
        },
        "provenance": {
            "config_sha256": sha256_file(cfg_path),
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "lf2_0b_report_sha256": sha256_file(
                resolve(cfg["lightglue"]["report"])
            ),
            "lightglue_evidence_sha256": sha256_file(
                resolve(cfg["lightglue"]["evidence"])
            ),
            "orb_report_sha256": sha256_file(resolve(cfg["orb"]["report"])),
            "orb_evidence_sha256": sha256_file(resolve(cfg["orb"]["evidence"])),
            "reference_attachment_sha256": sha256_file(reference_path),
            "tile_geometry_sha256": sha256_file(tile_path),
            "lf2_0b_selected_query_ids": (
                lg_freeze_report["screening_contract"]["selected_query_ids"]
            ),
        },
        "outputs": {},
    }

    out_root = resolve(cfg["output_root"])
    output_frames = {
        "labeled_pairs": (
            out_root / "lf2_0c_labeled_pairs.csv",
            labeled,
        ),
        "selected_projection": (
            out_root / "lf2_0c_selected_projection.csv",
            selected,
        ),
        "common_feature_diagnostics": (
            out_root / "lf2_0c_common_feature_diagnostics.csv",
            feature_diag,
        ),
        "lightglue_native_score_diagnostics": (
            out_root / "lf2_0c_lightglue_native_score_diagnostics.csv",
            native_diag,
        ),
    }
    files: dict[Path, bytes] = {}
    for name, (path, frame) in output_frames.items():
        payload = deterministic_csv(frame)
        files[path] = payload
        report["outputs"][name] = {
            "path": str(path.relative_to(ROOT)),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "rows": int(len(frame)),
        }

    report_path = out_root / "lf2_0c_audit_report.json"
    files[report_path] = deterministic_json(report)
    write_immutable(files)

    print("Status:", report["status"])
    print("Decision:", decision)
    print("Selected screening query IDs:", ",".join(map(str, selected_ids)))
    print("Primary Top1 comparison (LightGlue - ORB):")
    for label in labels:
        orb_hits = rankings["orb_inliers"][label]["hits"]
        lg_hits = rankings["lightglue_inliers"][label]["hits"]
        print(
            f"  {label}: "
            f"ORB={orb_hits['top1']}/40 "
            f"LightGlue={lg_hits['top1']}/40 "
            f"delta={lg_hits['top1']-orb_hits['top1']:+d} "
            f"LG_top3={lg_hits['top3']}/40 "
            f"LG_top5={lg_hits['top5']}/40"
        )

    print("Primary selected projection:")
    for name in ("orb_inliers", "lightglue_inliers"):
        p = projection_summaries[name]
        print(
            f"  {name}: median={p['median_m']:.3f} "
            f"rmse={p['rmse_m']:.3f} p95={p['p95_m']:.3f} "
            f"le40={p['le40']}/{p['count']} "
            f"le80={p['le80']}/{p['count']} "
            f"gt200={p['gt200']}/{p['count']}"
        )
    print("Primary full-run gate:", primary_gate["pass"])

    print("LightGlue native score diagnostics:")
    for score in cfg["ranking"]["lightglue_secondary_scores"]:
        gate_result = secondary_gates[score]
        tops = []
        for label in labels:
            hits = rankings["lightglue_secondary"][score][label]["hits"]
            tops.append(f"{label}={hits['top1']}/40")
        print(
            f"  {score}: {' '.join(tops)} "
            f"secondary_gate={gate_result['pass']}"
        )
    print("Secondary passing scores:", ",".join(secondary_passers) or "none")

    for label in labels:
        print(f"Top common features for {label}:")
        subset = feature_diag[feature_diag["label"] == label]
        for frontend in ("orb", "lightglue"):
            top = subset[subset["frontend"] == frontend].sort_values(
                ["auc_star", "feature"],
                ascending=[False, True],
                kind="mergesort",
            ).head(5)
            print(f"  {frontend}:")
            for row in top.itertuples(index=False):
                print(
                    f"    {row.feature}: auc*={row.auc_star:.4f} "
                    f"dir={row.direction} top1={int(row.top1)}/40 "
                    f"top5={int(row.top5)}/40"
                )

    print("Blind stress loaded: False")
    print("Full 403-query LightGlue loaded: False")
    print("Report:", report_path)
    print(
        "Audit diagnostics SHA256:",
        report["outputs"]["lightglue_native_score_diagnostics"]["sha256"],
    )


if __name__ == "__main__":
    main()
