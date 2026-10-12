#!/usr/bin/env python3
"""LF1.1-B: post-freeze ORB vs XFeat absolute-geometry comparison."""
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
DEFAULT_CONFIG = ROOT / "configs/research/learned_geometric_frontend_lf1_1b.yaml"

BASE_REQUIRED = {
    "query_id",
    "pool_rank",
    "tile_id",
    "homography_ok",
    "finite_projection",
    "projected_easting",
    "projected_northing",
    "inliers",
    "good_matches",
}
TILE_REQUIRED = {
    "tile_id",
    "left_easting",
    "right_easting",
    "bottom_northing",
    "top_northing",
    "center_easting",
    "center_northing",
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
            "Refusing to overwrite differing LF1.1-B outputs: "
            + ", ".join(changed)
        )
    for path, data in files.items():
        if path.exists():
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as handle:
            handle.write(data)


def require_columns(
    frame: pd.DataFrame,
    required: set[str],
    label: str,
) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"{label} missing columns: {missing}")


def bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    mapped = (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map(
            {
                "true": True,
                "false": False,
                "1": True,
                "0": False,
                "yes": True,
                "no": False,
                "y": True,
                "n": False,
                "t": True,
                "f": False,
            }
        )
    )
    if mapped.isna().any():
        bad = series[mapped.isna()].astype(str).unique().tolist()
        raise RuntimeError(f"Could not parse boolean values: {bad[:10]}")
    return mapped.astype(bool)


def numeric_feature(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(float)
    text = series.astype(str).str.strip().str.lower()
    boolish = text.map(
        {
            "true": 1.0,
            "false": 0.0,
            "yes": 1.0,
            "no": 0.0,
            "y": 1.0,
            "n": 0.0,
        }
    )
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.where(numeric.notna(), boolish).astype(float)


def verify_config(cfg: dict[str, Any]) -> None:
    if cfg.get("stage") != "LF1.1-B":
        raise RuntimeError("Expected LF1.1-B config")
    if cfg.get("labels") != [
        "contains_query",
        "center_le40",
        "center_le80",
    ]:
        raise RuntimeError("Label contract changed")
    if cfg["ranking"].get("selectors") != [
        "retrieval_rank",
        "inliers",
        "valid_then_inliers",
    ]:
        raise RuntimeError("Ranking selector contract changed")

    boundary = cfg["scientific_boundary"]
    if boundary.get("development_reference_allowed") is not True:
        raise RuntimeError("Development reference must be explicitly allowed")
    if boundary.get("reference_after_both_freezes_only") is not True:
        raise RuntimeError("Reference-after-freeze contract missing")
    for key in (
        "blind_stress_loaded",
        "blind_stress_reference_allowed",
        "evidence_mutable",
        "threshold_tuning_enabled",
        "selector_training_enabled",
        "frontend_parameters_mutable",
        "bootstrap_enabled",
        "state_enabled",
    ):
        if boundary.get(key) is not False:
            raise RuntimeError(f"{key} must be false")


def verify_orb(cfg: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    block = cfg["orb"]
    report_path = resolve(block["report"])
    evidence_path = resolve(block["development_evidence"])
    report = json.loads(report_path.read_text(encoding="utf-8"))

    if report.get("status") != (
        "PASS_GD1_0_REFERENCE_FREE_TOP20_GEOMETRIC_EVIDENCE_FREEZE"
    ):
        raise RuntimeError("ORB/GD1.0 report not PASS")
    boundary = report.get("scientific_boundary", {})
    if boundary.get("reference_loaded") is not False:
        raise RuntimeError("ORB/GD1.0 reference boundary violated")
    if report["trajectories"]["development"]["output"]["sha256"] != (
        block["expected_development_sha256"]
    ):
        raise RuntimeError("ORB development report hash mismatch")
    if sha256_file(evidence_path) != block["expected_development_sha256"]:
        raise RuntimeError("ORB development evidence bytes changed")
    return evidence_path, report


def verify_xfeat(cfg: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    block = cfg["xfeat"]
    report_path = resolve(block["report"])
    evidence_path = resolve(block["development_evidence"])
    report = json.loads(report_path.read_text(encoding="utf-8"))

    if report.get("status") != (
        "PASS_LF1_1A_REFERENCE_FREE_XFEAT_TOP20_FREEZE"
    ):
        raise RuntimeError("XFeat/LF1.1-A report not PASS")
    boundary = report.get("scientific_boundary", {})
    if boundary.get("reference_loaded") is not False:
        raise RuntimeError("XFeat/LF1.1-A reference boundary violated")
    if report["trajectories"]["development"]["output"]["sha256"] != (
        block["expected_development_sha256"]
    ):
        raise RuntimeError("XFeat development report hash mismatch")
    if report["trajectories"]["blind_stress"]["output"]["sha256"] != (
        block["expected_blind_sha256"]
    ):
        raise RuntimeError("XFeat blind report hash mismatch")
    if sha256_file(evidence_path) != block["expected_development_sha256"]:
        raise RuntimeError("XFeat development evidence bytes changed")
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
    *,
    frontend: str,
) -> pd.DataFrame:
    require_columns(evidence, BASE_REQUIRED, f"{frontend} evidence")
    require_columns(tile, TILE_REQUIRED, "tile geometry")

    frame = evidence.copy()
    frame["query_id"] = pd.to_numeric(
        frame["query_id"], errors="raise"
    ).astype(int)
    frame["pool_rank"] = pd.to_numeric(
        frame["pool_rank"], errors="raise"
    ).astype(int)
    frame["tile_id"] = frame["tile_id"].astype(str)
    frame["homography_ok"] = bool_series(frame["homography_ok"])
    frame["finite_projection"] = bool_series(frame["finite_projection"])
    frame["frontend"] = frontend

    tiles = tile[list(TILE_REQUIRED)].copy()
    tiles["tile_id"] = tiles["tile_id"].astype(str)
    if tiles["tile_id"].duplicated().any():
        raise RuntimeError("Duplicate tile IDs")

    refs = reference[["gt_x", "gt_y"]].reset_index()
    out = (
        frame.merge(tiles, on="tile_id", how="left", validate="many_to_one")
        .merge(refs, on="query_id", how="left", validate="many_to_one")
    )
    required_join = [
        "left_easting",
        "right_easting",
        "bottom_northing",
        "top_northing",
        "center_easting",
        "center_northing",
        "gt_x",
        "gt_y",
    ]
    if out[required_join].isna().any().any():
        raise RuntimeError(f"{frontend} reference/tile join has missing values")

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


def binary_auc(
    labels: pd.Series,
    scores: pd.Series,
) -> dict[str, Any]:
    frame = pd.DataFrame(
        {
            "label": labels.astype(bool),
            "score": pd.to_numeric(scores, errors="coerce"),
        }
    ).replace([np.inf, -np.inf], np.nan).dropna()
    positives = int(frame["label"].sum())
    negatives = int((~frame["label"]).sum())
    if positives == 0 or negatives == 0:
        return {
            "count": int(len(frame)),
            "positives": positives,
            "negatives": negatives,
            "auc": None,
            "auc_star": None,
            "direction": None,
        }

    ranks = frame["score"].rank(method="average", ascending=True)
    rank_sum = float(ranks[frame["label"]].sum())
    auc = (
        rank_sum - positives * (positives + 1) / 2.0
    ) / float(positives * negatives)
    if auc >= 0.5:
        return {
            "count": int(len(frame)),
            "positives": positives,
            "negatives": negatives,
            "auc": float(auc),
            "auc_star": float(auc),
            "direction": "higher",
        }
    return {
        "count": int(len(frame)),
        "positives": positives,
        "negatives": negatives,
        "auc": float(auc),
        "auc_star": float(1.0 - auc),
        "direction": "lower",
    }


def sort_candidates(
    group: pd.DataFrame,
    selector: str,
) -> pd.DataFrame:
    work = group.copy()
    if selector == "retrieval_rank":
        return work.sort_values(
            ["pool_rank", "tile_id"],
            ascending=[True, True],
            kind="mergesort",
        )
    if selector == "inliers":
        work["_inliers"] = pd.to_numeric(
            work["inliers"], errors="coerce"
        ).fillna(-1.0)
        return work.sort_values(
            ["_inliers", "pool_rank", "tile_id"],
            ascending=[False, True, True],
            kind="mergesort",
        )
    if selector == "valid_then_inliers":
        work["_valid"] = work["homography_ok"].astype(bool).astype(int)
        work["_inliers"] = pd.to_numeric(
            work["inliers"], errors="coerce"
        ).fillna(-1.0)
        return work.sort_values(
            ["_valid", "_inliers", "pool_rank", "tile_id"],
            ascending=[False, False, True, True],
            kind="mergesort",
        )
    raise ValueError(selector)


def ranking_summary(
    frame: pd.DataFrame,
    label: str,
    selector: str,
    topks: list[int],
) -> dict[str, Any]:
    hits = {f"top{k}": 0 for k in topks}
    for _, group in frame.groupby("query_id", sort=True):
        ordered = sort_candidates(group, selector)
        for k in topks:
            hits[f"top{k}"] += int(
                ordered.head(k)[label].astype(bool).any()
            )
    return {
        "queries": int(frame["query_id"].nunique()),
        "hits": hits,
    }


def homography_retention(
    frame: pd.DataFrame,
    label: str,
) -> dict[str, Any]:
    valid = frame["homography_ok"].astype(bool)
    positive = frame[label].astype(bool)
    negative = ~positive

    positive_pairs = int(positive.sum())
    negative_pairs = int(negative.sum())
    positive_valid = int((positive & valid).sum())
    negative_rejected = int((negative & ~valid).sum())

    original_query_hits = int(
        frame.groupby("query_id")[label].any().sum()
    )
    valid_query_hits = int(
        frame.assign(_valid_positive=positive & valid)
        .groupby("query_id")["_valid_positive"]
        .any()
        .sum()
    )
    return {
        "positive_pairs": positive_pairs,
        "positive_valid_pairs": positive_valid,
        "positive_pair_retention": (
            float(positive_valid / positive_pairs)
            if positive_pairs else None
        ),
        "negative_pairs": negative_pairs,
        "negative_rejected_pairs": negative_rejected,
        "negative_pair_rejection": (
            float(negative_rejected / negative_pairs)
            if negative_pairs else None
        ),
        "queries_with_positive_original": original_query_hits,
        "queries_with_positive_after_homography": valid_query_hits,
        "query_positive_retention": (
            float(valid_query_hits / original_query_hits)
            if original_query_hits else None
        ),
    }


def projection_summary(values: pd.Series) -> dict[str, Any]:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    if not len(arr):
        return {
            "count": 0,
            "median_m": None,
            "rmse_m": None,
            "p95_m": None,
            "le40": 0,
            "le80": 0,
            "gt200": 0,
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


def selected_projection_rows(
    frame: pd.DataFrame,
    selector: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for qid, group in frame.groupby("query_id", sort=True):
        selected = sort_candidates(group, selector).iloc[0]
        rows.append(
            {
                "frontend": str(selected["frontend"]),
                "query_id": int(qid),
                "selector": selector,
                "pool_rank": int(selected["pool_rank"]),
                "tile_id": str(selected["tile_id"]),
                "homography_ok": bool(selected["homography_ok"]),
                "inliers": float(selected["inliers"]),
                "projection_error_m": float(
                    selected["projection_error_m"]
                )
                if pd.notna(selected["projection_error_m"])
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def oracle_projection_summary(frame: pd.DataFrame) -> dict[str, Any]:
    best = (
        frame.replace([np.inf, -np.inf], np.nan)
        .dropna(subset=["projection_error_m"])
        .groupby("query_id")["projection_error_m"]
        .min()
    )
    return projection_summary(best)


def diagnostic_feature_ranking(
    frame: pd.DataFrame,
    feature: str,
    label: str,
    topks: list[int],
) -> dict[str, Any]:
    scores = numeric_feature(frame[feature])
    auc = binary_auc(frame[label], scores)
    direction = auc["direction"]
    if direction is None:
        return {"pair_auc": auc, "query_hits": None}

    hits = {f"top{k}": 0 for k in topks}
    finite_queries = 0
    for _, group in frame.groupby("query_id", sort=True):
        g = group.copy()
        g["_score"] = numeric_feature(g[feature])
        g["_finite"] = np.isfinite(g["_score"]).astype(int)
        finite_queries += int(bool(g["_finite"].any()))
        ascending_score = direction == "lower"
        g = g.sort_values(
            ["_finite", "_score", "pool_rank", "tile_id"],
            ascending=[False, ascending_score, True, True],
            kind="mergesort",
        )
        for k in topks:
            hits[f"top{k}"] += int(
                g.head(k)[label].astype(bool).any()
            )
    return {
        "pair_auc": auc,
        "query_hits": hits,
        "queries_with_finite_feature": int(finite_queries),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LF1.1-B post-freeze ORB vs XFeat absolute comparison"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()

    cfg_path = resolve(args.config)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    verify_config(cfg)

    orb_path, orb_report = verify_orb(cfg)
    xfeat_path, xfeat_report = verify_xfeat(cfg)
    tile_path = verify_tile_geometry(cfg)

    orb_raw = pd.read_csv(orb_path)
    xfeat_raw = pd.read_csv(xfeat_path)
    tile = pd.read_csv(tile_path)

    for name, frame, block in (
        ("orb", orb_raw, cfg["orb"]),
        ("xfeat", xfeat_raw, cfg["xfeat"]),
    ):
        if len(frame) != int(block["expected_pairs"]):
            raise RuntimeError(f"{name} pair count mismatch")
        if frame["query_id"].nunique() != int(block["expected_queries"]):
            raise RuntimeError(f"{name} query count mismatch")

    key_cols = ["query_id", "pool_rank", "tile_id"]
    if not orb_raw[key_cols].equals(xfeat_raw[key_cols]):
        raise RuntimeError("ORB and XFeat frozen Top20 pair ordering differs")

    for spec in cfg["common_features"]:
        if spec["orb"] not in orb_raw.columns:
            raise RuntimeError(
                f"ORB missing common feature {spec['orb']}"
            )
        if spec["xfeat"] not in xfeat_raw.columns:
            raise RuntimeError(
                f"XFeat missing common feature {spec['xfeat']}"
            )

    # Critical boundary: verify both frontends before opening development truth.
    reference_path = resolve(cfg["development"]["reference_attachment"])
    print("ORB and XFeat frozen evidence verified before reference attachment.")
    print("Attaching development reference for LF1.1-B diagnostics only.")
    reference = load_reference_xy(reference_path)

    orb = attach_reference(
        orb_raw, tile, reference, frontend="orb"
    )
    xfeat = attach_reference(
        xfeat_raw, tile, reference, frontend="xfeat"
    )

    labels = list(cfg["labels"])
    topks = [int(k) for k in cfg["ranking"]["topk"]]
    selectors = list(cfg["ranking"]["selectors"])

    report: dict[str, Any] = {
        "stage": "LF1.1-B",
        "status": "PASS_LF1_1B_POSTFREEZE_ORB_XFEAT_COMPARISON",
        "scope": "development_only_postfreeze",
        "frontends": {},
        "common_feature_diagnostics": {},
        "scientific_boundary": {
            "orb_freeze_verified_before_reference": True,
            "xfeat_freeze_verified_before_reference": True,
            "blind_stress_loaded": False,
            "blind_stress_reference_used": False,
            "threshold_tuning_executed": False,
            "selector_training_executed": False,
            "frontend_parameters_changed": False,
            "frontend_winner_promoted": False,
            "bootstrap_state_modified": False,
        },
        "interpretation": {
            "feature_auc_direction_is_development_diagnostic_only": True,
            "oracle_projection_is_diagnostic_upper_bound_only": True,
            "no_deployment_rule_selected": True,
        },
        "provenance": {
            "config_sha256": sha256_file(cfg_path),
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "orb_report_sha256": sha256_file(resolve(cfg["orb"]["report"])),
            "orb_evidence_sha256": sha256_file(orb_path),
            "xfeat_report_sha256": sha256_file(resolve(cfg["xfeat"]["report"])),
            "xfeat_evidence_sha256": sha256_file(xfeat_path),
            "tile_geometry_sha256": sha256_file(tile_path),
            "reference_attachment_sha256": sha256_file(reference_path),
        },
        "outputs": {},
    }

    labeled_frames = []
    selected_rows = []

    for frontend_name, frame in (("orb", orb), ("xfeat", xfeat)):
        frontend_block: dict[str, Any] = {
            "pair_counts": {
                "pairs": int(len(frame)),
                "queries": int(frame["query_id"].nunique()),
                "homography_ok": int(
                    frame["homography_ok"].astype(bool).sum()
                ),
                "finite_projection": int(
                    frame["finite_projection"].astype(bool).sum()
                ),
            },
            "labels": {},
            "projection": {},
        }

        for label in labels:
            frontend_block["labels"][label] = {
                "homography_retention": homography_retention(
                    frame, label
                ),
                "rankings": {
                    selector: ranking_summary(
                        frame, label, selector, topks
                    )
                    for selector in selectors
                },
            }

        selected = selected_projection_rows(
            frame,
            str(cfg["projection"]["selected_selector"]),
        )
        selected_rows.append(selected)
        frontend_block["projection"] = {
            "selected_valid_then_inliers": projection_summary(
                selected["projection_error_m"]
            ),
            "oracle_best_finite_projection": oracle_projection_summary(frame),
        }

        report["frontends"][frontend_name] = frontend_block

        labeled_copy = frame.copy()
        labeled_copy["frontend"] = frontend_name
        labeled_frames.append(labeled_copy)

    feature_rows = []
    for feature_spec in cfg["common_features"]:
        common_name = str(feature_spec["name"])
        report["common_feature_diagnostics"][common_name] = {}

        for frontend_name, frame, column in (
            ("orb", orb, str(feature_spec["orb"])),
            ("xfeat", xfeat, str(feature_spec["xfeat"])),
        ):
            fblock: dict[str, Any] = {}
            for label in labels:
                diagnostic = diagnostic_feature_ranking(
                    frame, column, label, topks
                )
                fblock[label] = diagnostic
                auc = diagnostic["pair_auc"]
                hits = diagnostic["query_hits"] or {}
                feature_rows.append(
                    {
                        "feature": common_name,
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
                        "top20": hits.get("top20"),
                        "queries_with_finite_feature": (
                            diagnostic["queries_with_finite_feature"]
                            if diagnostic["query_hits"] is not None
                            else 0
                        ),
                    }
                )
            report["common_feature_diagnostics"][common_name][
                frontend_name
            ] = fblock

    labeled = pd.concat(
        labeled_frames,
        ignore_index=True,
    ).sort_values(
        ["frontend", "query_id", "pool_rank"],
        kind="mergesort",
    )
    selected = pd.concat(
        selected_rows,
        ignore_index=True,
    ).sort_values(
        ["frontend", "query_id"],
        kind="mergesort",
    )
    feature_diag = pd.DataFrame(feature_rows).sort_values(
        ["label", "feature", "frontend"],
        kind="mergesort",
    )

    out_root = resolve(cfg["output_root"])
    labeled_path = out_root / "lf1_1b_labeled_pairs.csv"
    selected_path = out_root / "lf1_1b_selected_projection.csv"
    feature_path = out_root / "lf1_1b_common_feature_diagnostics.csv"

    files = {
        labeled_path: deterministic_csv(labeled),
        selected_path: deterministic_csv(selected),
        feature_path: deterministic_csv(feature_diag),
    }

    for name, path, frame in (
        ("labeled_pairs", labeled_path, labeled),
        ("selected_projection", selected_path, selected),
        ("common_feature_diagnostics", feature_path, feature_diag),
    ):
        report["outputs"][name] = {
            "path": str(path.relative_to(ROOT)),
            "sha256": hashlib.sha256(files[path]).hexdigest(),
            "rows": int(len(frame)),
        }

    report_path = out_root / "lf1_1b_comparison_report.json"
    files[report_path] = deterministic_json(report)
    write_immutable(files)

    print("Status:", report["status"])
    for frontend_name in ("orb", "xfeat"):
        fb = report["frontends"][frontend_name]
        print(
            f"{frontend_name}: pairs={fb['pair_counts']['pairs']} "
            f"homography_ok={fb['pair_counts']['homography_ok']} "
            f"finite={fb['pair_counts']['finite_projection']}"
        )
        for label in labels:
            retention = fb["labels"][label]["homography_retention"]
            ranks = fb["labels"][label]["rankings"]
            print(
                f"  {label}: "
                f"positive_pair_retention="
                f"{retention['positive_pair_retention']:.4f} "
                f"negative_pair_rejection="
                f"{retention['negative_pair_rejection']:.4f} "
                f"query_positive_retention="
                f"{retention['query_positive_retention']:.4f}"
            )
            for selector in selectors:
                hits = ranks[selector]["hits"]
                print(
                    f"    {selector}: "
                    f"top1={hits['top1']}/403 "
                    f"top3={hits['top3']}/403 "
                    f"top5={hits['top5']}/403 "
                    f"top10={hits['top10']}/403"
                )
        projection = fb["projection"]
        selected_summary = projection["selected_valid_then_inliers"]
        oracle_summary = projection["oracle_best_finite_projection"]
        print(
            "  selected projection: "
            f"median={selected_summary['median_m']:.3f} "
            f"rmse={selected_summary['rmse_m']:.3f} "
            f"p95={selected_summary['p95_m']:.3f} "
            f"le40={selected_summary['le40']}/"
            f"{selected_summary['count']} "
            f"le80={selected_summary['le80']}/"
            f"{selected_summary['count']} "
            f"gt200={selected_summary['gt200']}/"
            f"{selected_summary['count']}"
        )
        print(
            "  oracle projection: "
            f"median={oracle_summary['median_m']:.3f} "
            f"le40={oracle_summary['le40']}/"
            f"{oracle_summary['count']} "
            f"le80={oracle_summary['le80']}/"
            f"{oracle_summary['count']}"
        )

    for label in labels:
        subset = feature_diag[feature_diag["label"] == label].copy()
        subset = subset.sort_values(
            ["frontend", "auc_star"],
            ascending=[True, False],
            kind="mergesort",
        )
        print(f"Top common features for {label}:")
        for frontend_name in ("orb", "xfeat"):
            top = subset[
                subset["frontend"] == frontend_name
            ].head(5)
            print(f"  {frontend_name}:")
            for row in top.itertuples(index=False):
                print(
                    f"    {row.feature}: auc*={row.auc_star:.4f} "
                    f"dir={row.direction} "
                    f"top1={int(row.top1)}/403 "
                    f"top5={int(row.top5)}/403"
                )

    print("Blind stress loaded: False")
    print("Frontend winner promoted: False")
    print("Report:", report_path)
    print(
        "Comparison diagnostics SHA256:",
        report["outputs"]["common_feature_diagnostics"]["sha256"],
    )


if __name__ == "__main__":
    main()
