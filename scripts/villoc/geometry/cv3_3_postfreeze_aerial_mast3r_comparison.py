#!/usr/bin/env python3
"""CV3.3: post-freeze truth evaluation of compact aerial-MASt3R vs SIFT+LightGlue."""
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
DEFAULT_CONFIG = ROOT / "configs/research/crossview_scale_rotation_domain_cv3_3.yaml"

REQ = {
    "query_id", "pool_rank", "tile_id", "inliers",
    "homography_ok", "finite_projection",
    "projected_easting", "projected_northing",
}
TILE_REQ = {
    "tile_id", "left_easting", "right_easting",
    "bottom_northing", "top_northing",
    "center_easting", "center_northing",
}


def resolve(v: str | Path) -> Path:
    p = Path(v).expanduser()
    return p if p.is_absolute() else ROOT / p


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_csv(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, lineterminator="\n", float_format="%.9f").encode()


def canonical_json(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def freeze(files: dict[Path, bytes]) -> None:
    bad = [str(p) for p, payload in files.items() if p.exists() and p.read_bytes() != payload]
    if bad:
        raise RuntimeError("Refusing to overwrite differing CV3.3 outputs: " + ", ".join(bad))
    for path, payload in files.items():
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as handle:
                handle.write(payload)


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
        raise RuntimeError("Unparseable boolean values")
    return out.astype(bool)


def verify_config(cfg: dict[str, Any]) -> None:
    if cfg.get("stage") != "CV3.3":
        raise RuntimeError("Expected CV3.3 config")
    if cfg["ranking"]["selector"] != "inliers":
        raise RuntimeError("CV3.3 selector changed")
    if cfg["ranking"]["tie_break"] != ["pool_rank_asc", "tile_id_asc"]:
        raise RuntimeError("CV3.3 selector tie-break changed")
    if [int(x) for x in cfg["candidate_slice"]["pool_ranks"]] != [1,2,3,5,10,20]:
        raise RuntimeError("CV3.3 candidate slice changed")
    boundary = cfg["scientific_boundary"]
    for key in (
        "both_reference_free_freezes_verified_before_truth",
        "frozen_mast3r_selection_verified_before_truth",
        "same_candidate_identities_required",
        "development_reference_allowed",
    ):
        if boundary[key] is not True:
            raise RuntimeError(f"{key} must remain true")
    for key in (
        "mast3r_mutable", "sift_lightglue_mutable", "threshold_tuning_allowed",
        "selector_formula_mutable", "learned_score_fusion_allowed",
        "full_top20_allowed", "full403_allowed", "blind_stress_loaded",
        "state_changes_allowed", "temporal_fusion_allowed",
    ):
        if boundary[key] is not False:
            raise RuntimeError(f"{key} must remain false")


def rank(group: pd.DataFrame) -> pd.DataFrame:
    x = group.copy()
    x["_score"] = pd.to_numeric(x["inliers"], errors="coerce").fillna(-1.0)
    x["pool_rank"] = pd.to_numeric(x["pool_rank"], errors="raise").astype(int)
    x["tile_id"] = x["tile_id"].astype(str)
    return x.sort_values(
        ["_score", "pool_rank", "tile_id"],
        ascending=[False, True, True],
        kind="mergesort",
    )


def recompute_selection(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for qid, group in frame.groupby("query_id", sort=True):
        ordered = rank(group).reset_index(drop=True)
        winner = ordered.iloc[0]
        runner_up = ordered.iloc[1]
        rows.append({
            "query_id": int(qid),
            "selected_pool_rank": int(winner["pool_rank"]),
            "selected_tile_id": str(winner["tile_id"]),
            "selected_inliers": int(winner["inliers"]),
            "runner_up_inliers": int(runner_up["inliers"]),
            "inlier_margin": int(winner["inliers"] - runner_up["inliers"]),
            "selected_homography_ok": bool(winner["homography_ok"]),
            "selected_finite_projection": bool(winner["finite_projection"]),
            "selected_projected_inside_tile": bool(winner["projected_inside_tile"]),
            "selected_projected_easting": winner["projected_easting"],
            "selected_projected_northing": winner["projected_northing"],
        })
    return pd.DataFrame(rows).sort_values("query_id", kind="mergesort").reset_index(drop=True)


def verify_mast3r_freeze(cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    block = cfg["cv3_2"]
    report_path = resolve(block["report"])
    pair_path = resolve(block["pair_evidence"])
    selection_path = resolve(block["selection_evidence"])

    if sha256_file(pair_path) != block["expected_pair_sha256"]:
        raise RuntimeError("Frozen CV3.2 pair evidence bytes changed")
    if sha256_file(selection_path) != block["expected_selection_sha256"]:
        raise RuntimeError("Frozen CV3.2 selection evidence bytes changed")

    report = json.loads(report_path.read_text())
    if report.get("status") != block["expected_status"]:
        raise RuntimeError("CV3.2 report status mismatch")
    if report["provenance"]["pair_evidence_sha256"] != block["expected_pair_sha256"]:
        raise RuntimeError("CV3.2 report pair hash mismatch")
    if report["provenance"]["selection_evidence_sha256"] != block["expected_selection_sha256"]:
        raise RuntimeError("CV3.2 report selection hash mismatch")
    if report["scientific_boundary"]["reference_loaded"] is not False:
        raise RuntimeError("CV3.2 reference boundary violated")
    if report["scientific_boundary"]["selector_frozen_reference_free"] is not True:
        raise RuntimeError("CV3.2 selector was not frozen reference-free")

    pairs = pd.read_csv(pair_path)
    frozen_selection = pd.read_csv(selection_path)
    for frame in (pairs, frozen_selection):
        frame["query_id"] = pd.to_numeric(frame["query_id"], errors="raise").astype(int)
    pairs["pool_rank"] = pd.to_numeric(pairs["pool_rank"], errors="raise").astype(int)
    pairs["inliers"] = pd.to_numeric(pairs["inliers"], errors="raise").astype(int)
    pairs["tile_id"] = pairs["tile_id"].astype(str)
    pairs["homography_ok"] = bool_series(pairs["homography_ok"])
    pairs["finite_projection"] = bool_series(pairs["finite_projection"])
    pairs["projected_inside_tile"] = bool_series(pairs["projected_inside_tile"])

    if len(pairs) != int(block["expected_pairs"]) or pairs["query_id"].nunique() != int(block["expected_queries"]):
        raise RuntimeError("CV3.2 pair count changed")
    if len(frozen_selection) != int(block["expected_queries"]):
        raise RuntimeError("CV3.2 selection count changed")

    recomputed = recompute_selection(pairs)
    key_cols = ["query_id", "selected_pool_rank", "selected_tile_id", "selected_inliers", "runner_up_inliers", "inlier_margin"]
    left = recomputed[key_cols].copy()
    right = frozen_selection[key_cols].copy()
    right["selected_pool_rank"] = pd.to_numeric(right["selected_pool_rank"], errors="raise").astype(int)
    right["selected_inliers"] = pd.to_numeric(right["selected_inliers"], errors="raise").astype(int)
    right["runner_up_inliers"] = pd.to_numeric(right["runner_up_inliers"], errors="raise").astype(int)
    right["inlier_margin"] = pd.to_numeric(right["inlier_margin"], errors="raise").astype(int)
    right["selected_tile_id"] = right["selected_tile_id"].astype(str)
    if not left.equals(right):
        raise RuntimeError("Recomputed MASt3R selection differs from frozen CV3.2 selection")

    for col in ("selected_projected_easting", "selected_projected_northing"):
        a = pd.to_numeric(recomputed[col], errors="coerce").to_numpy(float)
        b = pd.to_numeric(frozen_selection[col], errors="coerce").to_numpy(float)
        if not np.allclose(a, b, equal_nan=True, rtol=0.0, atol=1e-8):
            raise RuntimeError(f"Frozen MASt3R selection projection differs: {col}")

    return pairs, frozen_selection, report


def verify_sift_freeze(cfg: dict[str, Any], mast_pairs: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    block = cfg["sift_lightglue"]
    report_path = resolve(block["report"])
    evidence_path = resolve(block["evidence"])
    if sha256_file(evidence_path) != block["expected_sha256"]:
        raise RuntimeError("Frozen SIFT+LightGlue evidence bytes changed")

    report = json.loads(report_path.read_text())
    if report.get("status") != block["expected_status"]:
        raise RuntimeError("CV2.2 report status mismatch")
    if report["provenance"]["evidence_sha256"] != block["expected_sha256"]:
        raise RuntimeError("CV2.2 report evidence hash mismatch")
    if report["scientific_boundary"]["reference_loaded"] is not False:
        raise RuntimeError("CV2.2 reference boundary violated")

    sift = pd.read_csv(evidence_path)
    sift["query_id"] = pd.to_numeric(sift["query_id"], errors="raise").astype(int)
    sift["pool_rank"] = pd.to_numeric(sift["pool_rank"], errors="raise").astype(int)
    sift["tile_id"] = sift["tile_id"].astype(str)
    sift["inliers"] = pd.to_numeric(sift["inliers"], errors="raise").astype(int)
    sift["homography_ok"] = bool_series(sift["homography_ok"])
    sift["finite_projection"] = bool_series(sift["finite_projection"])

    if len(sift) != int(block["expected_pairs_full_top20"]) or sift["query_id"].nunique() != int(block["expected_queries"]):
        raise RuntimeError("CV2.2 evidence count changed")

    ranks = [int(x) for x in cfg["candidate_slice"]["pool_ranks"]]
    sift = sift[sift["pool_rank"].isin(ranks)].copy()
    sift = sift.sort_values(["query_id", "pool_rank"], kind="mergesort").reset_index(drop=True)
    mast = mast_pairs.sort_values(["query_id", "pool_rank"], kind="mergesort").reset_index(drop=True)

    if len(sift) != int(cfg["candidate_slice"]["expected_pairs"]):
        raise RuntimeError("SIFT six-candidate subset size changed")

    ids = ["query_id", "pool_rank", "tile_id"]
    if not mast[ids].equals(sift[ids]):
        raise RuntimeError("MASt3R and SIFT compact candidate identities differ")

    return sift, report


def verify_tiles(cfg: dict[str, Any]) -> Path:
    report_path = resolve(cfg["map"]["rg2_0_report"])
    tile_path = resolve(cfg["map"]["tile_geometry"])
    report = json.loads(report_path.read_text())
    if report.get("status") != "PASS_RG2_0_MAP_TOPOLOGY_PREFLIGHT":
        raise RuntimeError("RG2.0 topology preflight not PASS")
    if sha256_file(tile_path) != report["provenance"]["tile_geometry_sha256"]:
        raise RuntimeError("Frozen tile geometry changed")
    return tile_path


def attach_truth(frame: pd.DataFrame, tiles: pd.DataFrame, ref: pd.DataFrame, frontend: str) -> pd.DataFrame:
    missing = sorted(REQ - set(frame.columns))
    if missing:
        raise RuntimeError(f"{frontend} missing columns: {missing}")
    missing_tiles = sorted(TILE_REQ - set(tiles.columns))
    if missing_tiles:
        raise RuntimeError(f"Tile geometry missing columns: {missing_tiles}")

    x = frame.copy()
    x["frontend"] = frontend
    x["query_id"] = pd.to_numeric(x["query_id"], errors="raise").astype(int)
    x["pool_rank"] = pd.to_numeric(x["pool_rank"], errors="raise").astype(int)
    x["tile_id"] = x["tile_id"].astype(str)
    x["inliers"] = pd.to_numeric(x["inliers"], errors="raise").astype(int)
    x["homography_ok"] = bool_series(x["homography_ok"])
    x["finite_projection"] = bool_series(x["finite_projection"])

    t = tiles[list(TILE_REQ)].copy()
    t["tile_id"] = t["tile_id"].astype(str)
    r = ref[["gt_x", "gt_y"]].reset_index()
    out = x.merge(t, on="tile_id", how="left", validate="many_to_one").merge(
        r, on="query_id", how="left", validate="many_to_one"
    )
    required = list(TILE_REQ - {"tile_id"}) + ["gt_x", "gt_y"]
    if out[required].isna().any().any():
        raise RuntimeError(f"{frontend} truth/tile join incomplete")

    out["contains_query"] = (
        (out["gt_x"] >= out["left_easting"]) &
        (out["gt_x"] <= out["right_easting"]) &
        (out["gt_y"] >= out["bottom_northing"]) &
        (out["gt_y"] <= out["top_northing"])
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


def projection_summary(values: pd.Series, total: int) -> dict[str, Any]:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    finite = arr[np.isfinite(arr)]
    nonfinite = int(total - len(finite))
    return {
        "finite_count": int(len(finite)),
        "nonfinite_count": nonfinite,
        "median_m": float(np.median(finite)) if len(finite) else None,
        "rmse_m": float(np.sqrt(np.mean(finite * finite))) if len(finite) else None,
        "p95_m": float(np.quantile(finite, 0.95)) if len(finite) else None,
        "le40": int((finite <= 40.0).sum()),
        "le80": int((finite <= 80.0).sum()),
        "gt200": int((finite > 200.0).sum()),
        "unsafe_tail": int(nonfinite + (finite > 200.0).sum()),
    }


def summarize(frame: pd.DataFrame, topks: list[int]) -> tuple[dict[str, Any], pd.DataFrame]:
    hits = {
        label: {f"top{k}": 0 for k in topks}
        for label in ("contains_query", "center_le40", "center_le80")
    }
    selected_rows = []

    for qid, group in frame.groupby("query_id", sort=True):
        ordered = rank(group).reset_index(drop=True)
        selected = ordered.iloc[0]
        for label in hits:
            for k in topks:
                hits[label][f"top{k}"] += int(ordered.head(k)[label].astype(bool).any())

        selected_rows.append({
            "query_id": int(qid),
            "tile_id": str(selected["tile_id"]),
            "pool_rank": int(selected["pool_rank"]),
            "inliers": int(selected["inliers"]),
            "contains_query": bool(selected["contains_query"]),
            "center_le40": bool(selected["center_le40"]),
            "center_le80": bool(selected["center_le80"]),
            "projection_error_m": (
                float(selected["projection_error_m"])
                if pd.notna(selected["projection_error_m"]) else np.nan
            ),
            "top1_top2_inlier_margin": (
                int(selected["inliers"]) - int(ordered.iloc[1]["inliers"])
                if len(ordered) > 1 else np.nan
            ),
        })

    selected = pd.DataFrame(selected_rows)
    total = len(selected)
    oracle = (
        frame.replace([np.inf, -np.inf], np.nan)
        .dropna(subset=["projection_error_m"])
        .groupby("query_id")["projection_error_m"]
        .min()
    )

    return {
        "queries": int(total),
        "ranking": hits,
        "selected_top1": {
            "contains_query": int(selected["contains_query"].sum()),
            "center_le40": int(selected["center_le40"].sum()),
            "center_le80": int(selected["center_le80"].sum()),
        },
        "selected_projection": projection_summary(selected["projection_error_m"], total),
        "oracle_best_finite_projection": projection_summary(oracle, total),
        "median_top1_top2_inlier_margin": float(
            pd.to_numeric(selected["top1_top2_inlier_margin"], errors="coerce").median()
        ),
    }, selected


def compare_selected(sift: pd.DataFrame, mast: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    a = sift.add_prefix("sift_").rename(columns={"sift_query_id": "query_id"})
    b = mast.add_prefix("mast_").rename(columns={"mast_query_id": "query_id"})
    merged = a.merge(b, on="query_id", validate="one_to_one")

    merged["winner_changed"] = merged["sift_tile_id"] != merged["mast_tile_id"]
    merged["containment_rescue"] = ~merged["sift_contains_query"] & merged["mast_contains_query"]
    merged["containment_damage"] = merged["sift_contains_query"] & ~merged["mast_contains_query"]
    merged["center80_rescue"] = ~merged["sift_center_le80"] & merged["mast_center_le80"]
    merged["center80_damage"] = merged["sift_center_le80"] & ~merged["mast_center_le80"]

    se = pd.to_numeric(merged["sift_projection_error_m"], errors="coerce")
    me = pd.to_numeric(merged["mast_projection_error_m"], errors="coerce")
    sift_unsafe = (~np.isfinite(se)) | (se > 200.0)
    mast_unsafe = (~np.isfinite(me)) | (me > 200.0)
    merged["unsafe_rescue"] = sift_unsafe & ~mast_unsafe
    merged["unsafe_damage"] = ~sift_unsafe & mast_unsafe

    return {
        "winner_changed": int(merged["winner_changed"].sum()),
        "containment_rescues": int(merged["containment_rescue"].sum()),
        "containment_damages": int(merged["containment_damage"].sum()),
        "center_le80_rescues": int(merged["center80_rescue"].sum()),
        "center_le80_damages": int(merged["center80_damage"].sum()),
        "unsafe_tail_rescues": int(merged["unsafe_rescue"].sum()),
        "unsafe_tail_damages": int(merged["unsafe_damage"].sum()),
    }, merged


def evaluate_gate(sift: dict[str, Any], mast: dict[str, Any], contract: dict[str, Any]) -> dict[str, Any]:
    containment_gain = int(
        mast["selected_top1"]["contains_query"] - sift["selected_top1"]["contains_query"]
    )
    center80_gain = int(
        mast["selected_top1"]["center_le80"] - sift["selected_top1"]["center_le80"]
    )
    center40_gain = int(
        mast["selected_top1"]["center_le40"] - sift["selected_top1"]["center_le40"]
    )
    selected_le40_gain = int(
        mast["selected_projection"]["le40"] - sift["selected_projection"]["le40"]
    )
    unsafe_reduction = int(
        sift["selected_projection"]["unsafe_tail"] - mast["selected_projection"]["unsafe_tail"]
    )

    containment_pass = containment_gain >= int(contract["min_top1_containment_gain"])
    center80_pass = center80_gain >= int(contract["min_top1_center_le80_gain"])
    unsafe_pass = (
        unsafe_reduction >= int(contract["min_unsafe_tail_reduction"])
        and selected_le40_gain >= -int(contract["max_selected_projection_le40_loss_for_unsafe_gate"])
    )

    return {
        "top1_containment_gain_mast3r_minus_sift": containment_gain,
        "top1_center_le40_gain_mast3r_minus_sift": center40_gain,
        "top1_center_le80_gain_mast3r_minus_sift": center80_gain,
        "selected_projection_le40_gain_mast3r_minus_sift": selected_le40_gain,
        "unsafe_tail_reduction_sift_minus_mast3r": unsafe_reduction,
        "containment_gate_pass": bool(containment_pass),
        "center_le80_gate_pass": bool(center80_pass),
        "unsafe_tail_gate_pass": bool(unsafe_pass),
        "pass": bool(containment_pass or center80_pass or unsafe_pass),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="CV3.3 post-freeze compact MASt3R vs SIFT evaluation")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()

    cfg_path = resolve(args.config)
    cfg = yaml.safe_load(cfg_path.read_text())
    verify_config(cfg)

    mast_raw, frozen_selection, cv32_report = verify_mast3r_freeze(cfg)
    sift_raw, cv22_report = verify_sift_freeze(cfg, mast_raw)
    tile_path = verify_tiles(cfg)

    print("CV3.2 pair and selection freezes verified.")
    print("Frozen MASt3R selector reproduced exactly.")
    print("SIFT+LightGlue six-candidate identities verified identical.")
    print("All reference-free evidence verified before truth attachment.")
    print("Attaching development reference for CV3.3 only.")

    ref_path = resolve(cfg["development"]["reference_attachment"])
    ref = load_reference_xy(ref_path)
    tiles = pd.read_csv(tile_path)

    frames = {
        "sift_lightglue_six": attach_truth(sift_raw, tiles, ref, "sift_lightglue_six"),
        "aerial_mast3r_six": attach_truth(mast_raw, tiles, ref, "aerial_mast3r_six"),
    }

    topks = [int(x) for x in cfg["ranking"]["topk"]]
    summaries: dict[str, Any] = {}
    selected: dict[str, pd.DataFrame] = {}
    for name, frame in frames.items():
        summaries[name], selected[name] = summarize(frame, topks)

    comparison, audit = compare_selected(
        selected["sift_lightglue_six"],
        selected["aerial_mast3r_six"],
    )
    gate = evaluate_gate(
        summaries["sift_lightglue_six"],
        summaries["aerial_mast3r_six"],
        cfg["continuation_gate_vs_sift_six_candidate"],
    )

    decision = (
        "PROCEED_CV3_4_FULL_UNIFORM40_TOP20"
        if gate["pass"]
        else "CLOSE_CV3_AERIAL_MAST3R_COMPACT_SCREEN"
    )

    summary_rows = []
    for name in ("sift_lightglue_six", "aerial_mast3r_six"):
        s = summaries[name]
        p = s["selected_projection"]
        o = s["oracle_best_finite_projection"]
        summary_rows.append({
            "frontend": name,
            "top1_contains_query": s["selected_top1"]["contains_query"],
            "top1_center_le40": s["selected_top1"]["center_le40"],
            "top1_center_le80": s["selected_top1"]["center_le80"],
            "projection_finite": p["finite_count"],
            "projection_nonfinite": p["nonfinite_count"],
            "projection_median_m": p["median_m"],
            "projection_rmse_m": p["rmse_m"],
            "projection_p95_m": p["p95_m"],
            "projection_le40": p["le40"],
            "projection_le80": p["le80"],
            "projection_gt200": p["gt200"],
            "projection_unsafe_tail": p["unsafe_tail"],
            "oracle_projection_median_m": o["median_m"],
            "oracle_projection_le40": o["le40"],
            "oracle_projection_le80": o["le80"],
            "median_top1_top2_inlier_margin": s["median_top1_top2_inlier_margin"],
        })
    summary = pd.DataFrame(summary_rows)

    report = {
        "stage": "CV3.3",
        "status": "PASS_CV3_3_POSTFREEZE_COMPACT_TRUTH_EVALUATION",
        "summaries": summaries,
        "mast3r_vs_sift_selected_comparison": comparison,
        "continuation_gate": {
            "contract": cfg["continuation_gate_vs_sift_six_candidate"],
            "result": gate,
        },
        "decision": decision,
        "scientific_boundary": {
            "cv3_2_pair_freeze_verified_before_reference": True,
            "cv3_2_selection_freeze_verified_before_reference": True,
            "mast3r_selection_recomputed_and_matched_freeze_before_reference": True,
            "sift_freeze_verified_before_reference": True,
            "same_six_candidate_identities_verified": True,
            "development_reference_used_postfreeze": True,
            "threshold_tuned": False,
            "selector_formula_changed": False,
            "learned_score_fusion_executed": False,
            "full_top20_executed": False,
            "full403_executed": False,
            "blind_stress_loaded": False,
            "state_modified": False,
        },
        "provenance": {
            "config_sha256": sha256_file(cfg_path),
            "cv3_2_report_sha256": sha256_file(resolve(cfg["cv3_2"]["report"])),
            "cv3_2_pair_evidence_sha256": sha256_file(resolve(cfg["cv3_2"]["pair_evidence"])),
            "cv3_2_selection_evidence_sha256": sha256_file(resolve(cfg["cv3_2"]["selection_evidence"])),
            "cv2_2_report_sha256": sha256_file(resolve(cfg["sift_lightglue"]["report"])),
            "sift_lightglue_uniform40_sha256": sha256_file(resolve(cfg["sift_lightglue"]["evidence"])),
            "reference_attachment_sha256": sha256_file(ref_path),
            "tile_geometry_sha256": sha256_file(tile_path),
            "cv3_2_status": cv32_report["status"],
            "cv2_2_status": cv22_report["status"],
        },
    }

    outroot = resolve(cfg["output_root"])
    summary_path = outroot / "cv3_3_frontend_summary.csv"
    audit_path = outroot / "cv3_3_selected_query_audit.csv"
    report_path = outroot / "cv3_3_report.json"
    freeze({
        summary_path: canonical_csv(summary),
        audit_path: canonical_csv(audit.sort_values("query_id", kind="mergesort")),
        report_path: canonical_json(report),
    })

    print("Status:", report["status"])
    print()
    print(summary.to_string(index=False))
    print()
    print("Winner changed:", comparison["winner_changed"])
    print("Containment rescues/damages:", f"{comparison['containment_rescues']}/{comparison['containment_damages']}")
    print("Center<=80 rescues/damages:", f"{comparison['center_le80_rescues']}/{comparison['center_le80_damages']}")
    print("Unsafe-tail rescues/damages:", f"{comparison['unsafe_tail_rescues']}/{comparison['unsafe_tail_damages']}")
    print("Continuation gate:", "PASS" if gate["pass"] else "FAIL")
    print("Gate result:", json.dumps(gate, sort_keys=True))
    print("Decision:", decision)
    print("Threshold tuned: False")
    print("Selector formula changed: False")
    print("Full Top20 executed: False")
    print("Report:", report_path)


if __name__ == "__main__":
    main()
