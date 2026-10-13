#!/usr/bin/env python3
"""CV2.3: post-freeze comparison of original ORB, scale_m2 ORB, and SIFT+LightGlue."""
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
DEFAULT_CONFIG = ROOT / "configs/research/crossview_scale_rotation_domain_cv2_3.yaml"

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
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def deterministic_csv(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, lineterminator="\n", float_format="%.9f").encode()


def deterministic_json(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def write_immutable(files: dict[Path, bytes]) -> None:
    bad = [str(p) for p, d in files.items() if p.exists() and p.read_bytes() != d]
    if bad:
        raise RuntimeError("Refusing to overwrite differing CV2.3 outputs: " + ", ".join(bad))
    for p, d in files.items():
        if not p.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
            with p.open("xb") as f:
                f.write(d)


def require(frame: pd.DataFrame, cols: set[str], label: str) -> None:
    missing = sorted(cols - set(frame.columns))
    if missing:
        raise RuntimeError(f"{label} missing columns: {missing}")


def bool_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.astype(bool)
    m = {"true": True, "false": False, "1": True, "0": False,
         "yes": True, "no": False, "y": True, "n": False, "t": True, "f": False}
    out = s.astype(str).str.strip().str.lower().map(m)
    if out.isna().any():
        raise RuntimeError("Unparseable boolean")
    return out.astype(bool)


def verify_config(cfg: dict[str, Any]) -> None:
    if cfg.get("stage") != "CV2.3":
        raise RuntimeError("Expected CV2.3 config")
    if cfg["ranking"]["selector"] != "inliers":
        raise RuntimeError("Selector changed")
    if cfg["ranking"]["tie_break"] != ["pool_rank_asc", "tile_id_asc"]:
        raise RuntimeError("Tie-break changed")
    b = cfg["scientific_boundary"]
    if b["both_reference_free_freezes_verified_before_truth"] is not True or b["development_reference_allowed"] is not True:
        raise RuntimeError("Reference boundary missing")
    for key in (
        "original_orb_mutable", "scale_m2_orb_mutable", "sift_lightglue_mutable",
        "scale_m2_applied_to_sift", "threshold_tuning_allowed",
        "selector_formula_mutable", "full403_allowed", "blind_stress_loaded",
        "state_changes_allowed", "temporal_fusion_allowed",
    ):
        if b[key] is not False:
            raise RuntimeError(f"{key} must be false")


def verify_inputs(cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cv1 = cfg["cv1_controls"]
    cv1_path = resolve(cv1["evidence"])
    if sha256_file(cv1_path) != cv1["expected_sha256"]:
        raise RuntimeError("Frozen CV1 uniform40 evidence changed")
    all_cv1 = pd.read_csv(cv1_path)
    orig = all_cv1[all_cv1["transform_id"].astype(str) == cv1["original_orb_transform_id"]].copy()
    scale = all_cv1[all_cv1["transform_id"].astype(str) == cv1["scale_m2_transform_id"]].copy()
    for name, frame in (("orb_original", orig), ("orb_scale_m2", scale)):
        if len(frame) != int(cv1["expected_pairs_per_condition"]) or frame["query_id"].nunique() != int(cv1["expected_queries"]):
            raise RuntimeError(f"{name} CV1 control count changed")

    s = cfg["sift_lightglue"]
    report = json.loads(resolve(s["report"]).read_text())
    if report.get("status") != s["expected_status"]:
        raise RuntimeError("CV2.2 report status mismatch")
    if report["provenance"]["evidence_sha256"] != s["expected_sha256"]:
        raise RuntimeError("CV2.2 report hash mismatch")
    sift_path = resolve(s["evidence"])
    if sha256_file(sift_path) != s["expected_sha256"]:
        raise RuntimeError("CV2.2 evidence bytes changed")
    sift = pd.read_csv(sift_path)
    if len(sift) != int(s["expected_pairs"]) or sift["query_id"].nunique() != int(s["expected_queries"]):
        raise RuntimeError("CV2.2 evidence count changed")
    if report["scientific_boundary"]["reference_loaded"] is not False:
        raise RuntimeError("CV2.2 reference boundary violated")

    key = ["query_id", "pool_rank", "tile_id"]
    orig = orig.sort_values(key, kind="mergesort").reset_index(drop=True)
    scale = scale.sort_values(key, kind="mergesort").reset_index(drop=True)
    sift = sift.sort_values(key, kind="mergesort").reset_index(drop=True)
    if not orig[key].equals(scale[key]) or not orig[key].equals(sift[key]):
        raise RuntimeError("Frozen frontend pair identities differ")
    return orig, scale, sift


def verify_tiles(cfg: dict[str, Any]) -> Path:
    rp = resolve(cfg["map"]["rg2_0_report"])
    tp = resolve(cfg["map"]["tile_geometry"])
    report = json.loads(rp.read_text())
    if report.get("status") != "PASS_RG2_0_MAP_TOPOLOGY_PREFLIGHT":
        raise RuntimeError("RG2.0 topology not PASS")
    if sha256_file(tp) != report["provenance"]["tile_geometry_sha256"]:
        raise RuntimeError("Tile geometry changed")
    return tp


def attach(frame: pd.DataFrame, tiles: pd.DataFrame, ref: pd.DataFrame, frontend: str) -> pd.DataFrame:
    require(frame, REQ, frontend)
    require(tiles, TILE_REQ, "tiles")
    x = frame.copy()
    x["query_id"] = pd.to_numeric(x["query_id"], errors="raise").astype(int)
    x["pool_rank"] = pd.to_numeric(x["pool_rank"], errors="raise").astype(int)
    x["tile_id"] = x["tile_id"].astype(str)
    x["homography_ok"] = bool_series(x["homography_ok"])
    x["finite_projection"] = bool_series(x["finite_projection"])
    x["frontend"] = frontend

    t = tiles[list(TILE_REQ)].copy()
    t["tile_id"] = t["tile_id"].astype(str)
    r = ref[["gt_x", "gt_y"]].reset_index()
    out = x.merge(t, on="tile_id", how="left", validate="many_to_one").merge(
        r, on="query_id", how="left", validate="many_to_one"
    )
    needed = list(TILE_REQ - {"tile_id"}) + ["gt_x", "gt_y"]
    if out[needed].isna().any().any():
        raise RuntimeError(f"{frontend} reference join incomplete")
    out["contains_query"] = (
        (out["gt_x"] >= out["left_easting"]) & (out["gt_x"] <= out["right_easting"]) &
        (out["gt_y"] >= out["bottom_northing"]) & (out["gt_y"] <= out["top_northing"])
    )
    out["center_error_m"] = np.hypot(out["center_easting"] - out["gt_x"], out["center_northing"] - out["gt_y"])
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


def rank(group: pd.DataFrame) -> pd.DataFrame:
    x = group.copy()
    x["_score"] = pd.to_numeric(x["inliers"], errors="coerce").fillna(-1.0)
    return x.sort_values(["_score", "pool_rank", "tile_id"], ascending=[False, True, True], kind="mergesort")


def proj_summary(values: pd.Series, total_queries: int) -> dict[str, Any]:
    a = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    finite = a[np.isfinite(a)]
    return {
        "finite_count": int(len(finite)),
        "nonfinite_count": int(total_queries - len(finite)),
        "median_m": float(np.median(finite)) if len(finite) else None,
        "rmse_m": float(np.sqrt(np.mean(finite * finite))) if len(finite) else None,
        "p95_m": float(np.quantile(finite, .95)) if len(finite) else None,
        "le40": int((finite <= 40).sum()),
        "le80": int((finite <= 80).sum()),
        "gt200": int((finite > 200).sum()),
        "unsafe_nonfinite_or_gt200": int((total_queries - len(finite)) + (finite > 200).sum()),
    }


def summarize(frame: pd.DataFrame, topks: list[int]) -> tuple[dict[str, Any], pd.DataFrame]:
    hits = {label: {f"top{k}": 0 for k in topks} for label in ("contains_query", "center_le40", "center_le80")}
    rows = []
    for qid, g in frame.groupby("query_id", sort=True):
        o = rank(g).reset_index(drop=True)
        s = o.iloc[0]
        for label in hits:
            for k in topks:
                hits[label][f"top{k}"] += int(o.head(k)[label].astype(bool).any())
        rows.append({
            "query_id": int(qid),
            "tile_id": str(s["tile_id"]),
            "pool_rank": int(s["pool_rank"]),
            "inliers": float(s["inliers"]),
            "contains_query": bool(s["contains_query"]),
            "center_le40": bool(s["center_le40"]),
            "center_le80": bool(s["center_le80"]),
            "projection_error_m": float(s["projection_error_m"]) if pd.notna(s["projection_error_m"]) else np.nan,
            "top1_top2_inlier_margin": (
                float(o.iloc[0]["inliers"]) - float(o.iloc[1]["inliers"]) if len(o) > 1 else np.nan
            ),
        })
    sel = pd.DataFrame(rows)
    qn = len(sel)
    oracle = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=["projection_error_m"]).groupby("query_id")["projection_error_m"].min()
    return {
        "queries": qn,
        "ranking": hits,
        "selected_top1": {
            "contains_query": int(sel["contains_query"].sum()),
            "center_le40": int(sel["center_le40"].sum()),
            "center_le80": int(sel["center_le80"].sum()),
        },
        "selected_projection": proj_summary(sel["projection_error_m"], qn),
        "oracle_best_finite_projection": proj_summary(oracle, qn),
        "median_top1_top2_inlier_margin": float(pd.to_numeric(sel["top1_top2_inlier_margin"], errors="coerce").median()),
    }, sel


def gate(scale: dict[str, Any], sift: dict[str, Any], contract: dict[str, Any]) -> dict[str, Any]:
    labels = ("contains_query", "center_le40", "center_le80")
    d = {k: int(sift["selected_top1"][k] - scale["selected_top1"][k]) for k in labels}
    ranking_pass = (
        all(v >= -int(contract["max_top1_regression_queries_each_target"]) for v in d.values())
        and max(d.values()) >= int(contract["min_top1_improvement_queries_any_target"])
    )
    le40d = int(sift["selected_projection"]["le40"] - scale["selected_projection"]["le40"])
    gt200d = int(sift["selected_projection"]["gt200"] - scale["selected_projection"]["gt200"])
    projection_pass = (
        le40d >= -int(contract["max_selected_projection_le40_regression_queries"])
        and gt200d <= int(contract["max_selected_projection_gt200_increase_queries"])
    )
    return {
        "top1_deltas_sift_minus_scale_m2_orb": d,
        "ranking_pass": bool(ranking_pass),
        "selected_projection_le40_delta": le40d,
        "selected_projection_gt200_delta": gt200d,
        "projection_pass": bool(projection_pass),
        "pass": bool(ranking_pass and projection_pass),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="CV2.3 post-freeze SIFT+LightGlue comparison")
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = ap.parse_args()
    cfg_path = resolve(args.config)
    cfg = yaml.safe_load(cfg_path.read_text())
    verify_config(cfg)

    orig_raw, scale_raw, sift_raw = verify_inputs(cfg)
    tile_path = verify_tiles(cfg)
    ref_path = resolve(cfg["development"]["reference_attachment"])
    print("All three frozen frontend evidence sets verified before reference attachment.")
    print("Attaching development reference for CV2.3 only.")
    ref = load_reference_xy(ref_path)
    tiles = pd.read_csv(tile_path)

    frames = {
        "orb_original": attach(orig_raw, tiles, ref, "orb_original"),
        "orb_scale_m2": attach(scale_raw, tiles, ref, "orb_scale_m2"),
        "sift_lightglue": attach(sift_raw, tiles, ref, "sift_lightglue"),
    }
    topks = [int(x) for x in cfg["ranking"]["topk"]]
    summaries, selected = {}, {}
    for name, frame in frames.items():
        summaries[name], selected[name] = summarize(frame, topks)

    result_gate = gate(
        summaries["orb_scale_m2"],
        summaries["sift_lightglue"],
        cfg["continuation_gate_vs_scale_m2"],
    )
    decision = "PROCEED_SIFT_LIGHTGLUE_FULL403" if result_gate["pass"] else "CLOSE_CV2_SIFT_LIGHTGLUE"

    summary_rows = []
    for name in ("orb_original", "orb_scale_m2", "sift_lightglue"):
        s = summaries[name]
        p = s["selected_projection"]
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
            "projection_unsafe_nonfinite_or_gt200": p["unsafe_nonfinite_or_gt200"],
            "oracle_projection_median_m": s["oracle_best_finite_projection"]["median_m"],
            "oracle_projection_le40": s["oracle_best_finite_projection"]["le40"],
            "oracle_projection_le80": s["oracle_best_finite_projection"]["le80"],
            "median_top1_top2_inlier_margin": s["median_top1_top2_inlier_margin"],
        })
    summary = pd.DataFrame(summary_rows)

    audit = []
    for name, sel in selected.items():
        x = sel.copy()
        x.insert(0, "frontend", name)
        audit.append(x)
    audit = pd.concat(audit, ignore_index=True).sort_values(["frontend", "query_id"], kind="mergesort")

    report = {
        "stage": "CV2.3",
        "status": "PASS_CV2_3_POSTFREEZE_SIFT_LIGHTGLUE_COMPARISON",
        "summaries": summaries,
        "continuation_gate": {
            "contract": cfg["continuation_gate_vs_scale_m2"],
            "result": result_gate,
        },
        "decision": decision,
        "scientific_boundary": {
            "freezes_verified_before_reference": True,
            "development_reference_used_postfreeze": True,
            "scale_m2_applied_to_sift": False,
            "threshold_tuned": False,
            "selector_formula_changed": False,
            "full403_executed": False,
            "blind_stress_loaded": False,
            "state_modified": False,
        },
        "provenance": {
            "config_sha256": sha256_file(cfg_path),
            "cv1_uniform40_sha256": sha256_file(resolve(cfg["cv1_controls"]["evidence"])),
            "sift_lightglue_sha256": sha256_file(resolve(cfg["sift_lightglue"]["evidence"])),
            "reference_attachment_sha256": sha256_file(ref_path),
            "tile_geometry_sha256": sha256_file(tile_path),
        },
    }

    outroot = resolve(cfg["output_root"])
    summary_path = outroot / "cv2_3_frontend_summary.csv"
    audit_path = outroot / "cv2_3_selected_query_audit.csv"
    report_path = outroot / "cv2_3_report.json"
    write_immutable({
        summary_path: deterministic_csv(summary),
        audit_path: deterministic_csv(audit),
        report_path: deterministic_json(report),
    })

    print("Status:", report["status"])
    print()
    print(summary.to_string(index=False))
    print()
    print("Continuation gate:", "PASS" if result_gate["pass"] else "FAIL")
    print("Gate result:", json.dumps(result_gate, sort_keys=True))
    print("Decision:", decision)
    print("scale_m2 applied to SIFT: False")
    print("Threshold tuned: False")
    print("Full403 executed: False")
    print("Report:", report_path)


if __name__ == "__main__":
    main()
