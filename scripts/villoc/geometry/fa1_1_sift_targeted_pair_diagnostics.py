#!/usr/bin/env python3
"""FA1.1 targeted geometry/match-support diagnostics for frozen SIFT failures.

The case set is selected by FA1.0. This script reruns only selected/oracle SIFT
pairs and scale_m2 ORB pairs for severe regressions. Every rerun must reproduce
the frozen pair evidence before any new spatial-support diagnostic is accepted.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import cv2
import numpy as np
import pandas as pd
import yaml

from uavloc.geometry.sift_lightglue_absolute_frontend import (
    RANSAC_THRESHOLD_PX,
    MIN_HOMOGRAPHY_INLIERS,
    SIFTLightGlueAbsoluteFrontend,
)
from uavloc.geometry.lightglue_absolute_frontend import _tensor_tree_to_device
from uavloc.geometry.top20_geometric_evidence import (
    _convex_hull_coverage,
    _grid_support,
    _pca_minor_major_ratio,
    _orientation_stats,
    _scale_stats,
    _project_points,
    _finite_summary,
    _jacobian_metrics,
    _denominator_and_footprint,
    analyze_pair as analyze_orb_pair,
)
from uavloc.geometry import orb_subtile
from uavloc.geometry.crossview_query_transforms import (
    feature_pack_from_gray,
    read_frozen_processed_query,
    scale_processed_query,
)
from uavloc.geometry.xfeat_absolute_frontend import tile_pixel_to_map

ROOT = REPO_ROOT
DEFAULT_CONFIG = ROOT / "configs/research/crossview_scale_rotation_domain_fa1_1.yaml"


def resolve(v):
    p = Path(v).expanduser()
    return p if p.is_absolute() else ROOT / p


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def dynamic_import(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def json_safe(v):
    if isinstance(v, dict):
        return {str(k): json_safe(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [json_safe(x) for x in v]
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        v = float(v)
    if isinstance(v, float):
        return v if math.isfinite(v) else None
    return v


def bool_value(v) -> bool:
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    return str(v).strip().lower() in {"true", "1", "yes", "y", "t"}


def resolve_existing(value: Any, *roots: Path) -> Path:
    p = Path(str(value)).expanduser()
    candidates = [p] if p.is_absolute() else [ROOT / p, *[r / p for r in roots]]
    for c in candidates:
        if c.is_file():
            return c.resolve()
    raise FileNotFoundError("Could not resolve: " + str(value))


def verify_config(cfg: dict[str, Any]) -> None:
    if cfg.get("stage") != "FA1.1":
        raise RuntimeError("Expected FA1.1 config")
    d = cfg["diagnostics"]
    if not all(d[k] is True for k in (
        "rerun_sift_selected", "rerun_sift_oracle",
        "rerun_scale_m2_orb_for_severe",
        "export_inlier_correspondences", "export_pair_overlays",
    )):
        raise RuntimeError("FA1.1 diagnostic scope changed")
    b = cfg["scientific_boundary"]
    if b["postfreeze_development_truth_used_only_for_case_selection"] is not True:
        raise RuntimeError("Case-selection truth boundary missing")
    if b["targeted_matcher_rerun_allowed"] is not True:
        raise RuntimeError("Targeted rerun must be explicitly allowed")
    for key in (
        "frozen_evidence_mutable", "frontend_parameters_mutable",
        "threshold_tuning_allowed", "selector_change_allowed",
        "confidence_rule_training_allowed", "blind_data_loaded",
        "accuracy_recomputed",
    ):
        if b[key] is not False:
            raise RuntimeError(f"{key} must remain false")


def flatten_feature(feats: dict[str, Any], key: str) -> np.ndarray:
    value = feats.get(key)
    if value is None:
        return np.empty((0,), dtype=float)
    arr = value[0].detach().cpu().numpy()
    return np.asarray(arr).reshape(-1)


def draw_inlier_overlay(
    query_path: Path,
    tile_path: Path,
    qpts: np.ndarray,
    tpts: np.ndarray,
    scores: np.ndarray,
    out_path: Path,
    *,
    title: str,
    max_side: int,
) -> None:
    qimg = cv2.imread(str(query_path), cv2.IMREAD_COLOR)
    timg = cv2.imread(str(tile_path), cv2.IMREAD_COLOR)
    if qimg is None or timg is None:
        raise RuntimeError("Could not read overlay input image")

    def prep(img):
        h, w = img.shape[:2]
        scale = min(1.0, float(max_side) / float(max(h, w)))
        if scale < 1.0:
            img2 = cv2.resize(
                img,
                (int(round(w * scale)), int(round(h * scale))),
                interpolation=cv2.INTER_AREA,
            )
        else:
            img2 = img.copy()
        return img2, scale

    qv, qs = prep(qimg)
    tv, ts = prep(timg)
    h = max(qv.shape[0], tv.shape[0])
    canvas = np.zeros((h + 48, qv.shape[1] + tv.shape[1], 3), dtype=np.uint8)
    canvas[48:48 + qv.shape[0], :qv.shape[1]] = qv
    canvas[48:48 + tv.shape[0], qv.shape[1]:qv.shape[1] + tv.shape[1]] = tv

    cv2.putText(
        canvas, title[:150], (12, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA,
    )

    order = np.argsort(-scores) if len(scores) else np.arange(len(qpts))
    for idx in order:
        qx, qy = qpts[int(idx)]
        tx, ty = tpts[int(idx)]
        p0 = (int(round(qx * qs)), int(round(qy * qs)) + 48)
        p1 = (
            qv.shape[1] + int(round(tx * ts)),
            int(round(ty * ts)) + 48,
        )
        cv2.circle(canvas, p0, 3, (0, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(canvas, p1, 3, (0, 255, 255), -1, cv2.LINE_AA)
        cv2.line(canvas, p0, p1, (0, 255, 0), 1, cv2.LINE_AA)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out_path), canvas):
        raise RuntimeError(f"Could not write {out_path}")


def rerun_sift_pair(
    frontend: SIFTLightGlueAbsoluteFrontend,
    query_path: Path,
    tile_path: Path,
    tile_bounds: tuple[float, float, float, float],
) -> tuple[dict[str, Any], list[dict[str, Any]], np.ndarray, np.ndarray, np.ndarray]:
    q_cpu, q_hw, _, _ = frontend._extract(query_path)
    t_cpu, t_hw, _, _ = frontend._extract(tile_path)
    q = _tensor_tree_to_device(q_cpu, frontend.device)
    t = _tensor_tree_to_device(t_cpu, frontend.device)

    frontend._sync()
    output = frontend.matcher({"image0": q, "image1": t})
    frontend._sync()

    matches = output["matches"][0].detach().cpu()
    scores = output["scores"][0].detach().cpu().numpy().astype(float)
    qk = q_cpu["keypoints"][0].numpy().astype(np.float32)
    tk = t_cpu["keypoints"][0].numpy().astype(np.float32)

    if len(matches):
        mi = matches.numpy().astype(int)
        qpts = qk[mi[:, 0]]
        tpts = tk[mi[:, 1]]
    else:
        mi = np.empty((0, 2), dtype=int)
        qpts = np.empty((0, 2), dtype=np.float32)
        tpts = np.empty((0, 2), dtype=np.float32)

    H = mask = None
    if len(qpts) >= 4:
        H, mask = cv2.findHomography(qpts, tpts, cv2.RANSAC, RANSAC_THRESHOLD_PX)
    inlier_mask = (
        mask.ravel().astype(bool)
        if mask is not None
        else np.zeros(len(qpts), dtype=bool)
    )
    inliers = int(inlier_mask.sum())
    homography_ok = bool(H is not None and inliers >= MIN_HOMOGRAPHY_INLIERS)

    qi = qpts[inlier_mask].astype(float)
    ti = tpts[inlier_mask].astype(float)
    inlier_scores = scores[inlier_mask] if len(scores) else np.empty((0,), float)

    q_scales = flatten_feature(q_cpu, "scales")
    t_scales = flatten_feature(t_cpu, "scales")
    q_oris = flatten_feature(q_cpu, "oris")
    t_oris = flatten_feature(t_cpu, "oris")

    q_match_scales = q_scales[mi[:, 0]] if len(mi) and len(q_scales) else np.empty((0,))
    t_match_scales = t_scales[mi[:, 1]] if len(mi) and len(t_scales) else np.empty((0,))
    q_match_oris = q_oris[mi[:, 0]] if len(mi) and len(q_oris) else np.empty((0,))
    t_match_oris = t_oris[mi[:, 1]] if len(mi) and len(t_oris) else np.empty((0,))

    q_inlier_scales = q_match_scales[inlier_mask] if len(q_match_scales) else np.empty((0,))
    t_inlier_scales = t_match_scales[inlier_mask] if len(t_match_scales) else np.empty((0,))
    q_inlier_oris = q_match_oris[inlier_mask] if len(q_match_oris) else np.empty((0,))
    t_inlier_oris = t_match_oris[inlier_mask] if len(t_match_oris) else np.empty((0,))

    u = v = e = n = math.nan
    finite_projection = inside = False

    diag: dict[str, Any] = {
        "matches": int(len(qpts)),
        "inliers": inliers,
        "inlier_ratio": float(inliers / max(1, len(qpts))),
        "homography_ok": homography_ok,
        "finite_projection": False,
        "projected_inside_tile": False,
        "projected_easting": math.nan,
        "projected_northing": math.nan,
        "query_hull_coverage": _convex_hull_coverage(qi, q_hw),
        "tile_hull_coverage": _convex_hull_coverage(ti, t_hw),
        "query_bbox_coverage": orb_subtile.bbox_coverage(qi, q_hw),
        "tile_bbox_coverage": orb_subtile.bbox_coverage(ti, t_hw),
        "query_pca_minor_major_ratio": _pca_minor_major_ratio(qi),
        "tile_pca_minor_major_ratio": _pca_minor_major_ratio(ti),
        "inlier_score_mean": float(np.mean(inlier_scores)) if len(inlier_scores) else 0.0,
        "inlier_score_median": float(np.median(inlier_scores)) if len(inlier_scores) else 0.0,
        "homography_condition_number": math.nan,
        "symmetric_error_median_px": math.nan,
        "symmetric_error_p90_px": math.nan,
        "symmetric_error_rmse_px": math.nan,
    }

    q_cells, q_occ, q_ent = _grid_support(qi, q_hw)
    t_cells, t_occ, t_ent = _grid_support(ti, t_hw)
    diag.update({
        "query_grid_cells": q_cells,
        "query_grid_occupancy": q_occ,
        "query_grid_entropy": q_ent,
        "tile_grid_cells": t_cells,
        "tile_grid_occupancy": t_occ,
        "tile_grid_entropy": t_ent,
    })
    diag.update(_orientation_stats(q_inlier_oris, t_inlier_oris))
    diag.update(_scale_stats(q_inlier_scales, t_inlier_scales))

    if homography_ok:
        Hn = np.asarray(H, dtype=float)
        diag["homography_condition_number"] = float(np.linalg.cond(Hn))
        forward = _project_points(Hn, qi)
        ferr = np.linalg.norm(forward - ti, axis=1)
        try:
            backward = _project_points(np.linalg.inv(Hn), ti)
            berr = np.linalg.norm(backward - qi, axis=1)
        except (np.linalg.LinAlgError, cv2.error):
            berr = np.full(len(qi), np.nan)
        sym = np.sqrt(ferr * ferr + berr * berr)
        ss = _finite_summary(sym)
        diag["symmetric_error_median_px"] = ss["median"]
        diag["symmetric_error_p90_px"] = ss["p90"]
        diag["symmetric_error_rmse_px"] = ss["rmse"]

        center_xy = (float(q_hw[1]) / 2.0, float(q_hw[0]) / 2.0)
        diag.update(_jacobian_metrics(Hn, center_xy))
        diag.update(_denominator_and_footprint(Hn, q_hw, t_hw))

        center = np.float32([[[q_hw[1] / 2.0, q_hw[0] / 2.0]]])
        projected = cv2.perspectiveTransform(center, Hn).reshape(2)
        u, v = float(projected[0]), float(projected[1])
        e, n = tile_pixel_to_map(u, v, t_hw, tile_bounds)
        finite_projection = bool(math.isfinite(e) and math.isfinite(n))
        inside = bool(
            finite_projection
            and 0 <= u < t_hw[1]
            and 0 <= v < t_hw[0]
        )
        diag.update({
            "projected_tile_u_px": u,
            "projected_tile_v_px": v,
            "finite_projection": finite_projection,
            "projected_inside_tile": inside,
            "projected_easting": e,
            "projected_northing": n,
        })

    corr_rows = []
    for j, match_index in enumerate(np.flatnonzero(inlier_mask)):
        q_idx, t_idx = mi[int(match_index)]
        corr_rows.append({
            "inlier_index": int(j),
            "query_keypoint_index": int(q_idx),
            "tile_keypoint_index": int(t_idx),
            "query_x": float(qpts[int(match_index), 0]),
            "query_y": float(qpts[int(match_index), 1]),
            "tile_x": float(tpts[int(match_index), 0]),
            "tile_y": float(tpts[int(match_index), 1]),
            "lightglue_score": float(scores[int(match_index)]),
            "query_scale": float(q_scales[q_idx]) if len(q_scales) else math.nan,
            "tile_scale": float(t_scales[t_idx]) if len(t_scales) else math.nan,
            "query_orientation": float(q_oris[q_idx]) if len(q_oris) else math.nan,
            "tile_orientation": float(t_oris[t_idx]) if len(t_oris) else math.nan,
        })

    return diag, corr_rows, qi, ti, inlier_scores


def assert_pair_parity(
    label: str,
    diagnostic: dict[str, Any],
    frozen: pd.Series,
    tolerance_m: float,
) -> None:
    frozen_matches_col = "good_matches" if "good_matches" in frozen.index else "matches"
    if int(diagnostic["matches"]) != int(frozen[frozen_matches_col]):
        raise RuntimeError(
            f"{label}: match-count parity failed "
            f"{diagnostic['matches']} != {frozen[frozen_matches_col]}"
        )
    if int(diagnostic["inliers"]) != int(frozen["inliers"]):
        raise RuntimeError(
            f"{label}: inlier parity failed "
            f"{diagnostic['inliers']} != {frozen['inliers']}"
        )
    if bool(diagnostic["homography_ok"]) != bool_value(frozen["homography_ok"]):
        raise RuntimeError(f"{label}: homography parity failed")

    fe = float(frozen["projected_easting"])
    fn = float(frozen["projected_northing"])
    de = float(diagnostic["projected_easting"])
    dn = float(diagnostic["projected_northing"])
    frozen_finite = math.isfinite(fe) and math.isfinite(fn)
    diag_finite = math.isfinite(de) and math.isfinite(dn)
    if frozen_finite != diag_finite:
        raise RuntimeError(f"{label}: finite-projection parity failed")
    if frozen_finite:
        delta = float(math.hypot(fe - de, fn - dn))
        if delta > tolerance_m:
            raise RuntimeError(
                f"{label}: projection parity failed by {delta:.6f} m "
                f"(tolerance {tolerance_m})"
            )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = ap.parse_args()

    cfg_path = resolve(args.config)
    cfg = yaml.safe_load(cfg_path.read_text())
    verify_config(cfg)

    fa0 = cfg["fa1_0"]
    fa0_report_path = resolve(fa0["report"])
    fa0_inventory_path = resolve(fa0["case_inventory"])
    fa0_report = json.loads(fa0_report_path.read_text())
    if fa0_report.get("status") != fa0["expected_status"]:
        raise RuntimeError("FA1.0 report is not the frozen PASS")

    counts = fa0_report["counts"]
    expected = {
        "oracle_rescuable": int(fa0["expected_oracle_rescuable"]),
        "catastrophic_damage": int(fa0["expected_catastrophic_damage"]),
        "severe_orb_le40_to_sift_unsafe": int(fa0["expected_severe_regression"]),
        "union_unique_queries": int(fa0["expected_union_queries"]),
    }
    for key, value in expected.items():
        if int(counts[key]) != value:
            raise RuntimeError(f"FA1.0 count changed for {key}")

    cases = pd.read_csv(fa0_inventory_path)
    if cases["query_id"].nunique() != int(fa0["expected_union_queries"]):
        raise RuntimeError("FA1.0 case inventory union count changed")
    cases["query_id"] = pd.to_numeric(cases["query_id"], errors="raise").astype(int)
    for col in (
        "oracle_rescuable", "catastrophic_damage",
        "severe_orb_le40_to_sift_unsafe",
    ):
        cases[col] = cases[col].astype(str).str.lower().isin({"true", "1", "yes"})

    frozen = cfg["frozen_frontends"]
    sift_path = resolve(frozen["sift_evidence"])
    scale_path = resolve(frozen["scale_m2_evidence"])
    if sha256_file(sift_path) != frozen["expected_sift_sha256"]:
        raise RuntimeError("Frozen SIFT evidence SHA mismatch")
    if sha256_file(scale_path) != frozen["expected_scale_m2_sha256"]:
        raise RuntimeError("Frozen scale_m2 evidence SHA mismatch")

    cv24_cfg_path = resolve(frozen["cv2_4_config"])
    cv24_cfg = yaml.safe_load(cv24_cfg_path.read_text())
    protocol = yaml.safe_load(resolve(cv24_cfg["protocol"]["config"]).read_text())
    dev = protocol["development"]
    manifest_path = resolve(dev["manifest"])
    spec_path = resolve(dev["spec"])
    manifest = pd.read_csv(manifest_path)
    spec = yaml.safe_load(spec_path.read_text())
    tile_index_path = resolve(spec["map"]["variants"]["768_s256"]["tile_index"])
    tiles = pd.read_csv(tile_index_path)

    manifest["query_id"] = pd.to_numeric(manifest["query_id"], errors="raise").astype(int)
    tiles["tile_id"] = tiles["tile_id"].astype(str)
    qlookup = manifest.set_index("query_id")
    tlookup = tiles.set_index("tile_id")

    sift_frozen = pd.read_csv(sift_path)
    sift_frozen["query_id"] = pd.to_numeric(sift_frozen["query_id"], errors="raise").astype(int)
    sift_frozen["tile_id"] = sift_frozen["tile_id"].astype(str)
    scale_frozen = pd.read_csv(scale_path)
    scale_frozen["query_id"] = pd.to_numeric(scale_frozen["query_id"], errors="raise").astype(int)
    scale_frozen["tile_id"] = scale_frozen["tile_id"].astype(str)

    frontend = SIFTLightGlueAbsoluteFrontend(
        resolve(protocol["lightglue"]["repo"]),
        device_name=str(protocol["lightglue"]["device"]),
    )
    meta = frontend.describe()
    if meta["lightglue_weights_sha256"] != frozen["lightglue_weights_sha256"]:
        raise RuntimeError("LightGlue weights changed")

    outroot = resolve(cfg["output_root"])
    overlay_root = outroot / "overlays"
    overlay_root.mkdir(parents=True, exist_ok=True)

    tolerance = float(cfg["diagnostics"]["parity_projection_tolerance_m"])
    max_side = int(cfg["diagnostics"]["overlay_max_side_px"])

    sift_diag_rows = []
    corr_rows_all = []

    requested_pairs = []
    for row in cases.itertuples(index=False):
        qid = int(row.query_id)
        requested_pairs.append((qid, "selected", str(row.sift_selected_tile_id)))
        requested_pairs.append((qid, "oracle", str(row.sift_oracle_tile_id)))
    requested_pairs = sorted(set(requested_pairs))

    for index, (qid, role, tile_id) in enumerate(requested_pairs, start=1):
        print(f"FA1.1 SIFT diagnostic pair {index}/{len(requested_pairs)}: q{qid} {role} {tile_id}", flush=True)
        qpath = resolve_existing(qlookup.loc[qid, "image_path"])
        tile_row = tlookup.loc[tile_id]
        tpath = resolve_existing(tile_row["tile_path"], tile_index_path.parent)
        bounds = (
            float(tile_row["left_easting"]),
            float(tile_row["right_easting"]),
            float(tile_row["bottom_northing"]),
            float(tile_row["top_northing"]),
        )

        diag, corr_rows, qi, ti, inlier_scores = rerun_sift_pair(
            frontend, qpath, tpath, bounds
        )
        frozen_row = sift_frozen[
            (sift_frozen["query_id"] == qid)
            & (sift_frozen["tile_id"] == tile_id)
        ]
        if len(frozen_row) != 1:
            raise RuntimeError(f"Frozen SIFT pair not unique: q{qid} {tile_id}")
        frozen_row = frozen_row.iloc[0]
        assert_pair_parity(
            f"SIFT q{qid} {role} {tile_id}",
            diag,
            frozen_row,
            tolerance,
        )

        case = cases[cases["query_id"] == qid].iloc[0]
        out = {
            "query_id": qid,
            "role": role,
            "tile_id": tile_id,
            "pool_rank": int(frozen_row["pool_rank"]),
            "oracle_rescuable_case": bool(case["oracle_rescuable"]),
            "catastrophic_damage_case": bool(case["catastrophic_damage"]),
            "severe_regression_case": bool(case["severe_orb_le40_to_sift_unsafe"]),
            "frozen_projection_error_m": float(
                case["sift_selected_error_m"]
                if role == "selected"
                else case["sift_oracle_error_m"]
            ),
            **diag,
            "frozen_parity_pass": True,
        }
        sift_diag_rows.append(out)

        for c in corr_rows:
            corr_rows_all.append({
                "query_id": qid,
                "role": role,
                "tile_id": tile_id,
                **c,
            })

        if cfg["diagnostics"]["export_pair_overlays"]:
            draw_inlier_overlay(
                qpath,
                tpath,
                qi,
                ti,
                inlier_scores,
                overlay_root / f"q{qid:03d}_{role}_{tile_id}_sift_inliers.png",
                title=(
                    f"q{qid} {role} {tile_id} | "
                    f"matches={diag['matches']} inliers={diag['inliers']} "
                    f"qHull={diag['query_hull_coverage']:.3f} "
                    f"tHull={diag['tile_hull_coverage']:.3f}"
                ),
                max_side=max_side,
            )

    sift_diag = pd.DataFrame(sift_diag_rows)
    corr = pd.DataFrame(corr_rows_all)

    # Pair selected and oracle diagnostics for the 19 oracle-rescuable cases.
    selected = sift_diag[sift_diag["role"] == "selected"].add_prefix("selected_")
    oracle = sift_diag[sift_diag["role"] == "oracle"].add_prefix("oracle_")
    paired = selected.merge(
        oracle,
        left_on="selected_query_id",
        right_on="oracle_query_id",
        how="inner",
        validate="one_to_one",
    ).rename(columns={"selected_query_id": "query_id"})
    paired = paired[paired["selected_oracle_rescuable_case"]].copy()

    delta_metrics = [
        "inliers", "inlier_ratio",
        "query_hull_coverage", "tile_hull_coverage",
        "query_bbox_coverage", "tile_bbox_coverage",
        "query_grid_occupancy", "tile_grid_occupancy",
        "query_grid_entropy", "tile_grid_entropy",
        "query_pca_minor_major_ratio", "tile_pca_minor_major_ratio",
        "inlier_score_mean", "inlier_score_median",
        "symmetric_error_median_px", "symmetric_error_p90_px",
        "homography_condition_number", "jacobian_anisotropy",
        "footprint_area_fraction_tile", "footprint_inside_corner_fraction",
        "orientation_resultant", "orientation_mad_deg",
        "log_scale_ratio_median", "log_scale_ratio_mad",
    ]
    for metric in delta_metrics:
        a = pd.to_numeric(paired[f"selected_{metric}"], errors="coerce")
        b = pd.to_numeric(paired[f"oracle_{metric}"], errors="coerce")
        paired[f"delta_selected_minus_oracle_{metric}"] = a - b

    # Reproduce the exact scale_m2 ORB selected pair for the seven severe regressions.
    detector = orb_subtile.create_detector()
    orb_rows = []
    scale_factor = float(frozen["scale_m2_factor"])
    severe_cases = cases[cases["severe_orb_le40_to_sift_unsafe"]].copy()
    for index, row in enumerate(severe_cases.itertuples(index=False), start=1):
        qid = int(row.query_id)
        tile_id = str(row.scale_selected_tile_id)
        print(f"FA1.1 scale_m2 ORB severe pair {index}/{len(severe_cases)}: q{qid} {tile_id}", flush=True)
        qpath = resolve_existing(qlookup.loc[qid, "image_path"])
        tile_row = tlookup.loc[tile_id]
        tpath = resolve_existing(tile_row["tile_path"], tile_index_path.parent)

        gray = read_frozen_processed_query(qpath)
        gray = scale_processed_query(gray, scale_factor)
        qfeat = feature_pack_from_gray(gray, detector)
        tfeat = orb_subtile.compute_features(tpath, detector)
        diag = analyze_orb_pair(qfeat, tfeat)

        u = float(diag["projected_tile_u_px"])
        v = float(diag["projected_tile_v_px"])
        e = n = math.nan
        if bool(diag["homography_ok"]):
            e, n = orb_subtile.tile_pixel_to_map(
                u, v, tfeat.image_shape,
                float(tile_row["left_easting"]),
                float(tile_row["right_easting"]),
                float(tile_row["bottom_northing"]),
                float(tile_row["top_northing"]),
            )

        frozen_row = scale_frozen[
            (scale_frozen["query_id"] == qid)
            & (scale_frozen["tile_id"] == tile_id)
        ]
        if len(frozen_row) != 1:
            raise RuntimeError(f"Frozen scale_m2 pair not unique: q{qid} {tile_id}")
        frozen_row = frozen_row.iloc[0]

        if int(diag["good_matches"]) != int(frozen_row["good_matches"]):
            raise RuntimeError(f"scale_m2 q{qid}: good-match parity failed")
        if int(diag["inliers"]) != int(frozen_row["inliers"]):
            raise RuntimeError(f"scale_m2 q{qid}: inlier parity failed")
        fe = float(frozen_row["projected_easting"])
        fn = float(frozen_row["projected_northing"])
        if math.isfinite(fe) and math.isfinite(fn):
            delta = math.hypot(float(e) - fe, float(n) - fn)
            if delta > tolerance:
                raise RuntimeError(
                    f"scale_m2 q{qid}: projection parity failed by {delta:.6f} m"
                )

        orb_rows.append({
            "query_id": qid,
            "tile_id": tile_id,
            "scale_factor": scale_factor,
            **{k: v for k, v in diag.items() if k != "error"},
            "projected_easting": e,
            "projected_northing": n,
            "frozen_parity_pass": True,
        })

    orb_diag = pd.DataFrame(orb_rows)

    # Compact aggregate: only descriptive, no thresholds or learned rule.
    aggregate = {}
    for metric in delta_metrics:
        col = f"delta_selected_minus_oracle_{metric}"
        vals = pd.to_numeric(paired[col], errors="coerce")
        finite = vals[np.isfinite(vals)]
        aggregate[metric] = {
            "median_selected_minus_oracle": float(finite.median()) if len(finite) else None,
            "selected_greater_count": int((finite > 0).sum()),
            "selected_lower_count": int((finite < 0).sum()),
            "equal_count": int((finite == 0).sum()),
            "finite_cases": int(len(finite)),
        }

    report = {
        "stage": "FA1.1",
        "status": "PASS_FA1_1_TARGETED_PAIR_GEOMETRY_DIAGNOSTICS",
        "counts": {
            "union_queries": int(cases["query_id"].nunique()),
            "sift_diagnostic_pairs": int(len(sift_diag)),
            "sift_inlier_correspondences": int(len(corr)),
            "oracle_rescuable_paired_cases": int(len(paired)),
            "scale_m2_orb_severe_pairs": int(len(orb_diag)),
            "overlay_pngs": int(len(list(overlay_root.glob("*.png")))),
        },
        "parity": {
            "all_sift_pairs_match_frozen_evidence": bool(
                len(sift_diag) > 0 and sift_diag["frozen_parity_pass"].all()
            ),
            "all_scale_m2_pairs_match_frozen_evidence": bool(
                len(orb_diag) == int(fa0["expected_severe_regression"])
                and orb_diag["frozen_parity_pass"].all()
            ),
            "projection_tolerance_m": tolerance,
        },
        "selected_vs_oracle_descriptive_deltas": aggregate,
        "scientific_boundary": {
            "case_membership_from_postfreeze_truth": True,
            "targeted_rerun_only": True,
            "frozen_frontends_changed": False,
            "threshold_tuned": False,
            "selector_changed": False,
            "confidence_rule_trained": False,
            "blind_data_loaded": False,
            "accuracy_recomputed": False,
        },
        "provenance": {
            "config_sha256": sha256_file(cfg_path),
            "fa1_0_report_sha256": sha256_file(fa0_report_path),
            "fa1_0_case_inventory_sha256": sha256_file(fa0_inventory_path),
            "sift_evidence_sha256": sha256_file(sift_path),
            "scale_m2_evidence_sha256": sha256_file(scale_path),
            "manifest_sha256": sha256_file(manifest_path),
            "tile_index_sha256": sha256_file(tile_index_path),
            "lightglue": meta,
        },
    }

    outroot.mkdir(parents=True, exist_ok=True)
    sift_path_out = outroot / "fa1_1_sift_pair_diagnostics.csv"
    corr_path_out = outroot / "fa1_1_sift_inlier_correspondences.csv"
    paired_path_out = outroot / "fa1_1_selected_vs_oracle_deltas.csv"
    orb_path_out = outroot / "fa1_1_scale_m2_severe_pair_diagnostics.csv"
    report_path_out = outroot / "fa1_1_report.json"

    sift_diag.to_csv(sift_path_out, index=False)
    corr.to_csv(corr_path_out, index=False)
    paired.to_csv(paired_path_out, index=False)
    orb_diag.to_csv(orb_path_out, index=False)
    report_path_out.write_text(json.dumps(json_safe(report), indent=2, allow_nan=False))

    print("Status:", report["status"])
    print("Union failure queries:", report["counts"]["union_queries"])
    print("SIFT diagnostic pairs:", report["counts"]["sift_diagnostic_pairs"])
    print("Oracle-rescuable paired cases:", report["counts"]["oracle_rescuable_paired_cases"])
    print("scale_m2 ORB severe diagnostic pairs:", report["counts"]["scale_m2_orb_severe_pairs"])
    print("SIFT inlier correspondences exported:", report["counts"]["sift_inlier_correspondences"])
    print("Overlay PNGs:", report["counts"]["overlay_pngs"])
    print("SIFT frozen parity:", report["parity"]["all_sift_pairs_match_frozen_evidence"])
    print("scale_m2 frozen parity:", report["parity"]["all_scale_m2_pairs_match_frozen_evidence"])
    print()
    for metric in (
        "query_hull_coverage", "tile_hull_coverage",
        "query_grid_entropy", "tile_grid_entropy",
        "query_pca_minor_major_ratio", "tile_pca_minor_major_ratio",
        "inlier_ratio", "symmetric_error_median_px",
        "homography_condition_number", "jacobian_anisotropy",
        "footprint_area_fraction_tile", "orientation_resultant",
        "log_scale_ratio_mad",
    ):
        item = aggregate[metric]
        print(
            f"{metric}: median selected-oracle="
            f"{item['median_selected_minus_oracle']} "
            f"(selected>oracle {item['selected_greater_count']}, "
            f"selected<oracle {item['selected_lower_count']}, "
            f"finite {item['finite_cases']})"
        )
    print()
    print("SIFT pair diagnostics:", sift_path_out)
    print("Inlier correspondences:", corr_path_out)
    print("Selected-vs-oracle deltas:", paired_path_out)
    print("scale_m2 severe diagnostics:", orb_path_out)
    print("Overlays:", overlay_root)
    print("Report:", report_path_out)
    print("Tuning / selector change / confidence training: False")


if __name__ == "__main__":
    main()
