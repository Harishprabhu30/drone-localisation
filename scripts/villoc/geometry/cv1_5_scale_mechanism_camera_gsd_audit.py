#!/usr/bin/env python3
"""CV1.5: explain the frozen scale_m2 gain and audit physical-GSD feasibility."""
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

import cv2
import numpy as np
import pandas as pd
import yaml

from uavloc.data.villoc_srt import parse_villoc_srt
from uavloc.evaluation.reference import load_reference_xy
from uavloc.geometry import orb_subtile
from uavloc.geometry.crossview_query_transforms import (
    feature_pack_from_gray,
    read_frozen_processed_query,
    scale_processed_query,
)

ROOT = REPO_ROOT
DEFAULT_CONFIG = ROOT / "configs/research/crossview_scale_rotation_domain_cv1_5.yaml"

PAIR_REQUIRED = {
    "query_id", "pool_rank", "tile_id", "inliers", "good_matches",
    "inlier_ratio", "homography_ok", "finite_projection",
    "projected_easting", "projected_northing",
}
TILE_REQUIRED = {
    "tile_id", "left_easting", "right_easting",
    "bottom_northing", "top_northing",
    "center_easting", "center_northing",
}
CAMERA_SAFE_FIELDS = [
    "query_id", "timestamp_s", "alignment_error_ms",
    "focal_len_mm", "dzoom_ratio", "iso", "shutter", "fnum", "ev",
    "rel_alt_m", "abs_alt_m",
    "gb_yaw_deg", "gb_pitch_deg", "gb_roll_deg",
]
FORBIDDEN_CAMERA_FIELDS = {"lat", "lon", "latitude", "longitude"}


def resolve(value: str | Path) -> Path:
    p = Path(value).expanduser()
    return p if p.is_absolute() else ROOT / p


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def deterministic_csv(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, lineterminator="\n", float_format="%.9f").encode("utf-8")


def deterministic_json(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def write_immutable(files: dict[Path, bytes]) -> None:
    changed = [str(p) for p, data in files.items() if p.exists() and p.read_bytes() != data]
    if changed:
        raise RuntimeError("Refusing to overwrite differing CV1.5 outputs: " + ", ".join(changed))
    for p, data in files.items():
        if p.exists():
            continue
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("xb") as f:
            f.write(data)


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
    if cfg.get("stage") != "CV1.5":
        raise RuntimeError("Expected CV1.5 config")
    if not math.isclose(
        float(cfg["scale_m2"]["scale_factor"]),
        0.6944444444444444,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise RuntimeError("scale_m2 changed")
    boundary = cfg["scientific_boundary"]
    if boundary.get("development_reference_allowed_for_explanation") is not True:
        raise RuntimeError("Development explanation reference not explicitly allowed")
    if boundary.get("srt_camera_metadata_allowed_postfreeze") is not True:
        raise RuntimeError("SRT camera audit not explicitly allowed")
    for key in (
        "srt_gps_allowed_for_analysis", "new_transform_conditions_allowed",
        "transform_retuning_allowed", "selector_changes_allowed",
        "threshold_tuning_allowed", "deployment_promotion_allowed",
        "physical_gsd_claim_allowed", "state_changes_allowed",
    ):
        if boundary.get(key) is not False:
            raise RuntimeError(f"{key} must be false")


def require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"{label} missing columns: {missing}")


def attach_labels(
    evidence: pd.DataFrame,
    tile: pd.DataFrame,
    reference: pd.DataFrame,
) -> pd.DataFrame:
    require_columns(evidence, PAIR_REQUIRED, "mechanism evidence")
    require_columns(tile, TILE_REQUIRED, "tile geometry")
    e = evidence.copy()
    e["query_id"] = pd.to_numeric(e["query_id"], errors="raise").astype(int)
    e["pool_rank"] = pd.to_numeric(e["pool_rank"], errors="raise").astype(int)
    e["tile_id"] = e["tile_id"].astype(str)
    e["homography_ok"] = bool_series(e["homography_ok"])
    e["finite_projection"] = bool_series(e["finite_projection"])

    t = tile[list(TILE_REQUIRED)].copy()
    t["tile_id"] = t["tile_id"].astype(str)
    ref = reference[["gt_x", "gt_y"]].reset_index()
    out = (
        e.merge(t, on="tile_id", how="left", validate="many_to_one")
        .merge(ref, on="query_id", how="left", validate="many_to_one")
    )
    if out[list(TILE_REQUIRED - {"tile_id"}) + ["gt_x", "gt_y"]].isna().any().any():
        raise RuntimeError("Reference/tile join produced missing values")
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


def rank_selected(frame: pd.DataFrame, condition: str) -> pd.DataFrame:
    rows = []
    for qid, group in frame.groupby("query_id", sort=True):
        g = group.copy()
        g["_inliers"] = pd.to_numeric(g["inliers"], errors="coerce").fillna(-1.0)
        g = g.sort_values(
            ["_inliers", "pool_rank", "tile_id"],
            ascending=[False, True, True],
            kind="mergesort",
        )
        r = g.iloc[0]
        rows.append({
            "query_id": int(qid),
            "condition": condition,
            "tile_id": str(r["tile_id"]),
            "pool_rank": int(r["pool_rank"]),
            "inliers": float(r["inliers"]),
            "contains_query": bool(r["contains_query"]),
            "center_le80": bool(r["center_le80"]),
            "projection_error_m": float(r["projection_error_m"]) if pd.notna(r["projection_error_m"]) else np.nan,
        })
    return pd.DataFrame(rows)


def numeric_stats(series: pd.Series) -> dict[str, Any]:
    v = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    v = v[np.isfinite(v)]
    if not len(v):
        return {"count": 0, "unique": 0, "min": None, "median": None, "max": None}
    return {
        "count": int(len(v)),
        "unique": int(len(np.unique(v))),
        "min": float(np.min(v)),
        "median": float(np.median(v)),
        "max": float(np.max(v)),
    }


def sanitize_camera_metadata(
    manifest: pd.DataFrame,
    srt: pd.DataFrame,
    max_error_ms: float,
) -> pd.DataFrame:
    if any(c in manifest.columns for c in FORBIDDEN_CAMERA_FIELDS):
        raise RuntimeError("Canonical blind manifest unexpectedly exposes GPS fields")
    m = manifest[["query_id", "timestamp_s"]].copy()
    m["query_id"] = pd.to_numeric(m["query_id"], errors="raise").astype(int)
    m["timestamp_s"] = pd.to_numeric(m["timestamp_s"], errors="raise").astype(float)

    src = srt.copy().sort_values("video_time_s")
    keep = [
        "video_time_s", "focal_len_mm", "dzoom_ratio", "iso", "shutter",
        "fnum", "ev", "rel_alt_m", "abs_alt_m",
        "gb_yaw_deg", "gb_pitch_deg", "gb_roll_deg",
    ]
    src = src[keep].copy()
    out = pd.merge_asof(
        m.sort_values("timestamp_s"),
        src,
        left_on="timestamp_s",
        right_on="video_time_s",
        direction="nearest",
    )
    out["alignment_error_ms"] = (
        (out["timestamp_s"] - out["video_time_s"]).abs() * 1000.0
    )
    if out["alignment_error_ms"].max() > float(max_error_ms):
        raise RuntimeError(
            f"SRT alignment exceeds {max_error_ms} ms: "
            f"{out['alignment_error_ms'].max():.3f} ms"
        )
    out = out.drop(columns=["video_time_s"])
    out = out[CAMERA_SAFE_FIELDS]
    leaked = FORBIDDEN_CAMERA_FIELDS & set(out.columns)
    if leaked:
        raise RuntimeError(f"GPS leaked into sanitized camera metadata: {sorted(leaked)}")
    return out.sort_values("query_id").reset_index(drop=True)


def keypoint_octave_row(query_id: int, image_path: Path, detector, scale_factor: float) -> dict[str, Any]:
    gray = read_frozen_processed_query(image_path)
    base = feature_pack_from_gray(gray, detector)
    scaled_gray = scale_processed_query(gray, scale_factor)
    scaled = feature_pack_from_gray(scaled_gray, detector)

    row: dict[str, Any] = {
        "query_id": int(query_id),
        "baseline_height_px": int(gray.shape[0]),
        "baseline_width_px": int(gray.shape[1]),
        "scale_m2_height_px": int(scaled_gray.shape[0]),
        "scale_m2_width_px": int(scaled_gray.shape[1]),
        "baseline_keypoints": int(len(base.keypoints)),
        "scale_m2_keypoints": int(len(scaled.keypoints)),
    }
    for name, pack in (("baseline", base), ("scale_m2", scaled)):
        octaves = np.array([int(k.octave) for k in pack.keypoints], dtype=int)
        sizes = np.array([float(k.size) for k in pack.keypoints], dtype=float)
        row[f"{name}_keypoint_size_median_px"] = float(np.median(sizes)) if len(sizes) else np.nan
        for octave in range(8):
            row[f"{name}_octave_{octave}_count"] = int((octaves == octave).sum())
            row[f"{name}_octave_{octave}_fraction"] = (
                float((octaves == octave).sum() / len(octaves)) if len(octaves) else 0.0
            )
    return row


def pair_mechanism_summary(
    baseline: pd.DataFrame,
    scale: pd.DataFrame,
    metrics: list[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    key = ["query_id", "pool_rank", "tile_id"]
    missing = [m for m in metrics if m not in baseline.columns or m not in scale.columns]
    if missing:
        raise RuntimeError(f"Mechanism metrics missing: {missing}")
    joined = baseline[key + ["contains_query", "center_le80"] + metrics].merge(
        scale[key + metrics],
        on=key,
        suffixes=("_baseline", "_scale_m2"),
        validate="one_to_one",
    )
    rows = []
    summary: dict[str, Any] = {}
    for metric in metrics:
        b = pd.to_numeric(joined[f"{metric}_baseline"], errors="coerce")
        s = pd.to_numeric(joined[f"{metric}_scale_m2"], errors="coerce")
        delta = s - b
        joined[f"{metric}_delta"] = delta
        block: dict[str, Any] = {}
        for label_name, mask in (
            ("all_pairs", pd.Series(True, index=joined.index)),
            ("contains_true", joined["contains_query"].astype(bool)),
            ("contains_false", ~joined["contains_query"].astype(bool)),
            ("center_le80_true", joined["center_le80"].astype(bool)),
            ("center_le80_false", ~joined["center_le80"].astype(bool)),
        ):
            vals = delta[mask]
            vals = vals[np.isfinite(vals)]
            block[label_name] = {
                "count": int(len(vals)),
                "median_delta": float(np.median(vals)) if len(vals) else None,
                "mean_delta": float(np.mean(vals)) if len(vals) else None,
                "positive_delta_fraction": float((vals > 0).mean()) if len(vals) else None,
            }
        summary[metric] = block
        rows.append({
            "metric": metric,
            "all_median_delta": block["all_pairs"]["median_delta"],
            "contains_true_median_delta": block["contains_true"]["median_delta"],
            "contains_false_median_delta": block["contains_false"]["median_delta"],
            "center_le80_true_median_delta": block["center_le80_true"]["median_delta"],
            "center_le80_false_median_delta": block["center_le80_false"]["median_delta"],
        })
    return pd.DataFrame(rows), summary


def camera_report(camera: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, Any]:
    fields = [
        "focal_len_mm", "dzoom_ratio", "iso", "fnum", "ev",
        "rel_alt_m", "abs_alt_m", "gb_yaw_deg", "gb_pitch_deg", "gb_roll_deg",
        "alignment_error_ms",
    ]
    stats = {field: numeric_stats(camera[field]) for field in fields}
    scale = float(cfg["scale_m2"]["scale_factor"])
    map_gsd = float(cfg["map"]["source_gsd_m_per_px"])
    H = float(cfg["camera_audit"]["nominal_blind_altitude_m"])
    base_gsd_implied = scale * map_gsd
    implied_fx = H / base_gsd_implied
    processed_w = float(cfg["camera_audit"]["baseline_processed_long_edge_px"])
    implied_hfov_deg = math.degrees(2.0 * math.atan(processed_w / (2.0 * implied_fx)))

    blockers = []
    if not cfg["camera_audit"]["camera_model_identified_in_repo"]:
        blockers.append("camera_model_not_identified_in_repo")
    if not cfg["camera_audit"]["calibrated_intrinsics_available"]:
        blockers.append("calibrated_intrinsics_unavailable")
    if not cfg["camera_audit"]["focal_length_semantics_verified"]:
        blockers.append("srt_focal_length_semantics_unverified")
    if stats["dzoom_ratio"]["unique"] > 1:
        blockers.append("digital_zoom_varies_across_sequence")

    return {
        "field_stats": stats,
        "implied_scale_diagnostic": {
            "map_gsd_m_per_px": map_gsd,
            "scale_m2_factor": scale,
            "implied_baseline_query_gsd_m_per_px_if_scale_m2_is_exact_gsd_match": base_gsd_implied,
            "nominal_altitude_m": H,
            "implied_processed_fx_px": implied_fx,
            "implied_processed_horizontal_fov_deg_assuming_width_1024": implied_hfov_deg,
            "claim_type": "diagnostic_implication_only_not_calibration",
        },
        "physical_gsd_claim_blocked": bool(blockers),
        "blockers": blockers,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CV1.5 frozen scale mechanism and camera/GSD feasibility audit"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()

    cfg_path = resolve(args.config)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    verify_config(cfg)

    baseline_path = resolve(cfg["baseline"]["evidence"])
    scale_path = resolve(cfg["scale_m2"]["evidence"])
    if sha256_file(baseline_path) != cfg["baseline"]["expected_sha256"]:
        raise RuntimeError("Baseline evidence hash changed")
    if sha256_file(scale_path) != cfg["scale_m2"]["expected_sha256"]:
        raise RuntimeError("scale_m2 evidence hash changed")

    baseline_raw = pd.read_csv(baseline_path)
    scale_raw = pd.read_csv(scale_path)
    if len(baseline_raw) != int(cfg["trajectory"]["expected_pairs"]) or len(scale_raw) != int(cfg["trajectory"]["expected_pairs"]):
        raise RuntimeError("Pair count changed")
    key = ["query_id", "pool_rank", "tile_id"]
    if not baseline_raw[key].equals(scale_raw[key]):
        raise RuntimeError("Baseline and scale pair ordering differs")

    manifest_path = resolve(cfg["trajectory"]["manifest"])
    manifest = pd.read_csv(manifest_path)
    if len(manifest) != int(cfg["trajectory"]["expected_queries"]):
        raise RuntimeError("Manifest query count changed")

    spec = yaml.safe_load(resolve(cfg["trajectory"]["spec"]).read_text(encoding="utf-8"))
    srt_path = resolve(spec["reference"]["path"])
    srt = parse_villoc_srt(
        srt_path,
        video_id=spec["trajectory"]["id"],
        modality="V",
        role="development_camera_audit_only",
    )
    camera = sanitize_camera_metadata(
        manifest,
        srt,
        float(cfg["trajectory"]["srt_alignment_max_error_ms"]),
    )
    if len(camera) != int(cfg["trajectory"]["expected_queries"]):
        raise RuntimeError("Sanitized camera metadata query count changed")

    reference = load_reference_xy(resolve(cfg["map"]["reference_attachment"]))
    tile = pd.read_csv(resolve(cfg["map"]["tile_geometry"]))
    baseline = attach_labels(baseline_raw, tile, reference)
    scale = attach_labels(scale_raw, tile, reference)

    baseline_selected = rank_selected(baseline, "baseline")
    scale_selected = rank_selected(scale, "scale_m2")
    selected = baseline_selected.merge(
        scale_selected,
        on="query_id",
        suffixes=("_baseline", "_scale_m2"),
        validate="one_to_one",
    )
    selected["winner_changed"] = selected["tile_id_baseline"] != selected["tile_id_scale_m2"]
    b_err = pd.to_numeric(selected["projection_error_m_baseline"], errors="coerce")
    s_err = pd.to_numeric(selected["projection_error_m_scale_m2"], errors="coerce")
    selected["catastrophic_rescue"] = (b_err > 200.0) & ~(s_err > 200.0)
    selected["catastrophic_damage"] = ~(b_err > 200.0) & (s_err > 200.0)

    rescue_split = {
        "catastrophic_rescues_total": int(selected["catastrophic_rescue"].sum()),
        "catastrophic_rescues_winner_changed": int(
            (selected["catastrophic_rescue"] & selected["winner_changed"]).sum()
        ),
        "catastrophic_rescues_same_winner": int(
            (selected["catastrophic_rescue"] & ~selected["winner_changed"]).sum()
        ),
        "catastrophic_damages_total": int(selected["catastrophic_damage"].sum()),
        "catastrophic_damages_winner_changed": int(
            (selected["catastrophic_damage"] & selected["winner_changed"]).sum()
        ),
        "catastrophic_damages_same_winner": int(
            (selected["catastrophic_damage"] & ~selected["winner_changed"]).sum()
        ),
    }

    mechanism_table, mechanism = pair_mechanism_summary(
        baseline, scale, [str(x) for x in cfg["mechanism"]["paired_metrics"]]
    )

    detector = orb_subtile.create_detector()
    octave_rows = []
    for row in manifest.sort_values("query_id").itertuples(index=False):
        image_path = resolve(str(row.image_path))
        octave_rows.append(
            keypoint_octave_row(
                int(row.query_id),
                image_path,
                detector,
                float(cfg["scale_m2"]["scale_factor"]),
            )
        )
    octave = pd.DataFrame(octave_rows)

    octave_summary = {
        "baseline_keypoints_median": float(octave["baseline_keypoints"].median()),
        "scale_m2_keypoints_median": float(octave["scale_m2_keypoints"].median()),
        "baseline_keypoint_size_median_px": float(octave["baseline_keypoint_size_median_px"].median()),
        "scale_m2_keypoint_size_median_px": float(octave["scale_m2_keypoint_size_median_px"].median()),
        "octave_fraction_medians": {},
    }
    for o in range(8):
        octave_summary["octave_fraction_medians"][str(o)] = {
            "baseline": float(octave[f"baseline_octave_{o}_fraction"].median()),
            "scale_m2": float(octave[f"scale_m2_octave_{o}_fraction"].median()),
        }

    cam_report = camera_report(camera, cfg)

    report = {
        "stage": "CV1.5",
        "status": "PASS_CV1_5_SCALE_MECHANISM_CAMERA_GSD_AUDIT",
        "rescue_pathway": rescue_split,
        "pair_mechanism": mechanism,
        "keypoint_octave_summary": octave_summary,
        "camera_metadata": cam_report,
        "decision": (
            "PHYSICAL_GSD_CLAIM_BLOCKED_CONTINUE_MECHANISM_ONLY"
            if cam_report["physical_gsd_claim_blocked"]
            else "CAMERA_MODEL_SUFFICIENT_FOR_PHYSICAL_GSD_FOLLOWUP"
        ),
        "scientific_boundary": {
            "reference_used_postfreeze_for_explanation": True,
            "srt_camera_metadata_used_postfreeze": True,
            "srt_gps_emitted": False,
            "new_transform_tested": False,
            "transform_retuned": False,
            "selector_changed": False,
            "threshold_tuned": False,
            "deployment_promoted": False,
            "physical_gsd_claim_made": False,
            "state_modified": False,
        },
        "provenance": {
            "config_sha256": sha256_file(cfg_path),
            "baseline_evidence_sha256": sha256_file(baseline_path),
            "scale_m2_evidence_sha256": sha256_file(scale_path),
            "manifest_sha256": sha256_file(manifest_path),
            "srt_sha256": sha256_file(srt_path),
        },
    }

    out_root = resolve(cfg["output_root"])
    camera_path = out_root / "cv1_5_sanitized_camera_metadata.csv"
    octave_path = out_root / "cv1_5_query_orb_octave_audit.csv"
    mechanism_path = out_root / "cv1_5_pair_mechanism_summary.csv"
    selected_path = out_root / "cv1_5_selected_rescue_pathways.csv"
    report_path = out_root / "cv1_5_audit_report.json"
    write_immutable({
        camera_path: deterministic_csv(camera),
        octave_path: deterministic_csv(octave),
        mechanism_path: deterministic_csv(mechanism_table),
        selected_path: deterministic_csv(selected),
        report_path: deterministic_json(report),
    })

    print("Status:", report["status"])
    print("Catastrophic rescue pathway:", json.dumps(rescue_split, sort_keys=True))
    print(
        "ORB keypoints median baseline/scale_m2:",
        f"{octave_summary['baseline_keypoints_median']:.1f}/"
        f"{octave_summary['scale_m2_keypoints_median']:.1f}",
    )
    print("Camera focal_len_mm:", json.dumps(cam_report["field_stats"]["focal_len_mm"], sort_keys=True))
    print("Camera dzoom_ratio:", json.dumps(cam_report["field_stats"]["dzoom_ratio"], sort_keys=True))
    print("Camera rel_alt_m:", json.dumps(cam_report["field_stats"]["rel_alt_m"], sort_keys=True))
    print("Camera pitch_deg:", json.dumps(cam_report["field_stats"]["gb_pitch_deg"], sort_keys=True))
    print("Implied scale diagnostic:", json.dumps(cam_report["implied_scale_diagnostic"], sort_keys=True))
    print("Physical GSD claim blocked:", cam_report["physical_gsd_claim_blocked"])
    print("GSD blockers:", ",".join(cam_report["blockers"]) if cam_report["blockers"] else "NONE")
    print("SRT GPS emitted: False")
    print("Transform retuned: False")
    print("Report:", report_path)


if __name__ == "__main__":
    main()
