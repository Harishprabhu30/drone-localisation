from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from uavloc.relative.orb_relative_motion import resolve_project_paths


STAGE_NAME = "08b_sync_enrichment"


def read_clean_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, engine="python")
    df.columns = [str(c).strip() for c in df.columns]

    keep_cols = []
    for c in df.columns:
        c_clean = str(c).strip()
        if c_clean == "" or c_clean.lower().startswith("unnamed"):
            continue
        keep_cols.append(c)

    df = df[keep_cols].copy()

    for c in df.columns:
        converted = pd.to_numeric(df[c], errors="coerce")
        if converted.notna().sum() > 0:
            df[c] = converted

    return df


def find_raw_file(raw_dir: Path, filename: str) -> Optional[Path]:
    filename_lower = filename.lower()

    search_roots = [
        raw_dir,
        raw_dir / "Log_Files",
        raw_dir / "Log Files",
        raw_dir.parent,
        Path.cwd() / "data" / "raw" / "zurich_mav_sample",
        Path.cwd() / "data" / "raw" / "zurich_mav_sample" / "Log_Files",
        Path.cwd() / "data" / "raw" / "zurich_mav_sample" / "Log Files",
    ]

    checked = []

    for root in search_roots:
        root = Path(root)
        checked.append(str(root))

        direct = root / filename
        if direct.exists():
            return direct

        if not root.exists():
            continue

        for candidate in root.rglob("*"):
            if candidate.is_file() and candidate.name.lower() == filename_lower:
                return candidate

    print(f"[WARN] Could not find {filename}. Checked roots:")
    for item in checked:
        print(f"  - {item}")

    return None


def numeric_series(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce")


def numeric_array(df: pd.DataFrame, col: str) -> np.ndarray:
    return numeric_series(df, col).to_numpy(dtype=float)


def is_usable_numeric(
    values: pd.Series | np.ndarray,
    min_valid: int = 5,
    reject_mostly_zero: bool = True,
    mostly_zero_threshold: float = 0.95,
) -> bool:
    s = pd.to_numeric(pd.Series(values), errors="coerce").dropna()

    if len(s) < min_valid:
        return False

    if s.nunique(dropna=True) <= 1:
        return False

    if float(s.std()) < 1e-12:
        return False

    if reject_mostly_zero:
        zero_ratio = float((s == 0).sum() / len(s))
        if zero_ratio >= mostly_zero_threshold:
            return False

    return True


def maybe_angle_to_deg(values: pd.Series | np.ndarray) -> np.ndarray:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    finite = arr[np.isfinite(arr)]

    if len(finite) == 0:
        return arr

    max_abs = float(np.nanmax(np.abs(finite)))

    if max_abs <= 2 * math.pi + 0.5:
        return np.rad2deg(arr)

    return arr


def merge_nearest_timestamp(
    sync_df: pd.DataFrame,
    telemetry_df: pd.DataFrame,
    rename_map: Dict[str, str],
    tolerance: Optional[float],
) -> Tuple[pd.DataFrame, int]:
    if "timestamp" not in sync_df.columns or "timestamp" not in telemetry_df.columns:
        return sync_df, 0

    left = sync_df.copy()
    right = telemetry_df.copy()

    left["_row_order"] = np.arange(len(left))

    left["timestamp"] = pd.to_numeric(left["timestamp"], errors="coerce")
    right["timestamp"] = pd.to_numeric(right["timestamp"], errors="coerce")

    cols = ["timestamp"] + [c for c in rename_map.keys() if c in right.columns]
    right = right[cols].copy()
    right = right.rename(columns=rename_map)

    left_valid = left[left["timestamp"].notna()].copy()
    left_invalid = left[left["timestamp"].isna()].copy()
    right = right[right["timestamp"].notna()].copy()

    if left_valid.empty or right.empty:
        return sync_df, 0

    left_valid["timestamp"] = left_valid["timestamp"].round().astype("int64")
    right["timestamp"] = right["timestamp"].round().astype("int64")

    tolerance_value = None if tolerance is None else int(round(float(tolerance)))

    left_valid = left_valid.sort_values("timestamp")
    right = right.sort_values("timestamp")

    merged_valid = pd.merge_asof(
        left_valid,
        right,
        on="timestamp",
        direction="nearest",
        tolerance=tolerance_value,
    )

    merged = pd.concat([merged_valid, left_invalid], axis=0, ignore_index=True)
    merged = merged.sort_values("_row_order").drop(columns=["_row_order"])

    matched_col = next((v for v in rename_map.values() if v in merged.columns), None)
    matched_rows = int(pd.to_numeric(merged[matched_col], errors="coerce").notna().sum()) if matched_col else 0

    return merged, matched_rows


def interpolate_groundtruth_by_imgid(sync_df: pd.DataFrame, gt_df: pd.DataFrame) -> pd.DataFrame:
    if "imgid" not in sync_df.columns or "imgid" not in gt_df.columns:
        return sync_df

    out = sync_df.copy()

    sync_imgid = numeric_array(out, "imgid")
    gt_df = gt_df.sort_values("imgid").copy()
    gt_imgid = numeric_array(gt_df, "imgid")

    rename_map = {
        "x_gt": "gt_x_utm_m",
        "y_gt": "gt_y_utm_m",
        "z_gt": "gt_z_m",
        "omega_gt": "gt_omega_deg",
        "phi_gt": "gt_phi_deg",
        "kappa_gt": "gt_kappa_deg",
        "x_gps": "gt_x_gps_utm_m",
        "y_gps": "gt_y_gps_utm_m",
        "z_gps": "gt_z_gps_m",
    }

    for src_col, dst_col in rename_map.items():
        if src_col not in gt_df.columns:
            continue

        src_values = numeric_array(gt_df, src_col)
        interp = np.interp(sync_imgid, gt_imgid, src_values)

        outside = (sync_imgid < np.nanmin(gt_imgid)) | (sync_imgid > np.nanmax(gt_imgid))
        interp[outside] = np.nan

        out[dst_col] = interp

    if "gt_z_m" in out.columns and "gt_z_gps_m" in out.columns:
        out["gt_z_minus_gps_z_m"] = out["gt_z_m"] - out["gt_z_gps_m"]
        out["abs_gt_z_minus_gps_z_m"] = out["gt_z_minus_gps_z_m"].abs()

    out["gt_interpolated_by_imgid"] = True

    return out


def select_height_and_yaw(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    out = df.copy()
    summary: Dict[str, Any] = {}

    height_source = None
    height_quality = None

    if "pose_height_m" in out.columns and is_usable_numeric(out["pose_height_m"]):
        h = numeric_series(out, "pose_height_m")
        if 0.1 <= float(h.median()) <= 200.0:
            out["height_agl_m"] = h
            out["height_agl_source"] = "pose_height_m"
            out["height_agl_quality"] = "preferred_if_verified"
            height_source = "pose_height_m"
            height_quality = "preferred_if_verified"

    if height_source is None and "abs_gt_z_minus_gps_z_m" in out.columns and is_usable_numeric(out["abs_gt_z_minus_gps_z_m"], reject_mostly_zero=False):
        h = numeric_series(out, "abs_gt_z_minus_gps_z_m")
        if 0.05 <= float(h.median()) <= 200.0:
            out["height_agl_m"] = h
            out["height_agl_source"] = "abs_gt_z_minus_gps_z_m"
            out["height_agl_quality"] = "approximate_debug_not_true_agl"
            height_source = "abs_gt_z_minus_gps_z_m"
            height_quality = "approximate_debug_not_true_agl"

    if height_source is None and "barometer_altitude_m" in out.columns and is_usable_numeric(out["barometer_altitude_m"]):
        alt = numeric_series(out, "barometer_altitude_m")
        first_valid = alt.dropna().iloc[0]
        out["height_agl_m"] = (alt - first_valid).abs()
        out["height_agl_source"] = "relative_barometer_altitude_from_start"
        out["height_agl_quality"] = "debug_relative_only_not_true_agl"
        height_source = "relative_barometer_altitude_from_start"
        height_quality = "debug_relative_only_not_true_agl"

    yaw_source = None
    yaw_quality = None

    if "pose_azimuth_raw" in out.columns:
        pose_azimuth_deg = maybe_angle_to_deg(out["pose_azimuth_raw"])
        out["pose_azimuth_deg"] = pose_azimuth_deg

        if is_usable_numeric(pd.Series(pose_azimuth_deg)):
            out["yaw_deg"] = pose_azimuth_deg
            out["yaw_source"] = "pose_azimuth_deg"
            out["yaw_quality"] = "preferred_if_verified"
            yaw_source = "pose_azimuth_deg"
            yaw_quality = "preferred_if_verified"

    if yaw_source is None and "gt_omega_deg" in out.columns and is_usable_numeric(out["gt_omega_deg"]):
        out["yaw_deg"] = numeric_series(out, "gt_omega_deg")
        out["yaw_source"] = "gt_omega_deg"
        out["yaw_quality"] = "groundtruth_orientation_candidate"
        yaw_source = "gt_omega_deg"
        yaw_quality = "groundtruth_orientation_candidate"

    if "pose_veh_pitch_raw" in out.columns:
        out["pose_veh_pitch_deg"] = maybe_angle_to_deg(out["pose_veh_pitch_raw"])

    summary["selected_height_source"] = height_source
    summary["selected_height_quality"] = height_quality
    summary["selected_yaw_source"] = yaw_source
    summary["selected_yaw_quality"] = yaw_quality

    if "height_agl_m" in out.columns:
        h = numeric_series(out, "height_agl_m")
        summary["height_agl_m"] = {
            "valid_count": int(h.notna().sum()),
            "median": float(h.median()),
            "min": float(h.min()),
            "max": float(h.max()),
            "source": height_source,
            "quality": height_quality,
        }

    if "yaw_deg" in out.columns:
        y = numeric_series(out, "yaw_deg")
        summary["yaw_deg"] = {
            "valid_count": int(y.notna().sum()),
            "median": float(y.median()),
            "min": float(y.min()),
            "max": float(y.max()),
            "source": yaw_source,
            "quality": yaw_quality,
        }

    warnings = []

    if "pose_height_m" in out.columns and not is_usable_numeric(out["pose_height_m"]):
        warnings.append("OnboardPose Height is not usable because it is constant/zero or invalid.")

    if "pose_azimuth_deg" in out.columns and not is_usable_numeric(out["pose_azimuth_deg"]):
        warnings.append("OnboardPose Azimuth is not usable because it is constant/zero or invalid.")

    if height_quality and "not_true_agl" in height_quality:
        warnings.append("Selected height_agl_m is approximate/debug only, not verified true AGL.")

    if yaw_source == "gt_omega_deg":
        warnings.append("Selected yaw_deg comes from GroundTruthAGL omega_gt because OnboardPose Azimuth is unusable.")

    summary["warnings"] = warnings

    return out, summary


def run_enrich_zurich_sync(
    config_path: str | Path,
    pose_tolerance: float = 200000.0,
    barometer_tolerance: float = 500000.0,
) -> Dict[str, Any]:
    _, _, raw_dir, output_dir, dataset_name = resolve_project_paths(config_path)

    sync_csv = output_dir / "metadata" / "synchronized_frames.csv"

    if not sync_csv.exists():
        raise FileNotFoundError(
            f"Missing synchronized frames file: {sync_csv}\n"
            "Run sync_zurich_frames.py first."
        )

    sync_df = pd.read_csv(sync_csv)
    sync_df.columns = [str(c).strip() for c in sync_df.columns]

    enriched = sync_df.copy()
    source_status: Dict[str, Any] = {}

    gt_path = find_raw_file(raw_dir, "GroundTruthAGL.csv")
    pose_path = find_raw_file(raw_dir, "OnboardPose.csv")
    baro_path = find_raw_file(raw_dir, "BarometricPressure.csv")

    if gt_path is not None and gt_path.exists():
        gt_df = read_clean_csv(gt_path)
        enriched = interpolate_groundtruth_by_imgid(enriched, gt_df)
        source_status["GroundTruthAGL.csv"] = {
            "available": True,
            "path": str(gt_path),
            "rows": int(len(gt_df)),
            "columns": list(gt_df.columns),
        }
    else:
        source_status["GroundTruthAGL.csv"] = {"available": False, "path": str(gt_path) if gt_path else None}

    if pose_path is not None and pose_path.exists():
        pose_df = read_clean_csv(pose_path)

        pose_map = {
            "Azimuth": "pose_azimuth_raw",
            "Height": "pose_height_m",
            "Altitude": "pose_altitude_m",
            "veh_pitch": "pose_veh_pitch_raw",
            "Attitude_w": "pose_attitude_w",
            "Attitude_x": "pose_attitude_x",
            "Attitude_y": "pose_attitude_y",
            "Attitude_z": "pose_attitude_z",
            "Vel_x": "pose_vel_x",
            "Vel_y": "pose_vel_y",
            "Vel_z": "pose_vel_z",
            "GPS_on": "pose_gps_on",
        }

        enriched, matched = merge_nearest_timestamp(
            enriched,
            pose_df,
            rename_map=pose_map,
            tolerance=pose_tolerance,
        )

        source_status["OnboardPose.csv"] = {
            "available": True,
            "path": str(pose_path),
            "rows": int(len(pose_df)),
            "columns": list(pose_df.columns),
            "nearest_timestamp_tolerance": float(pose_tolerance),
            "matched_rows": int(matched),
        }
    else:
        source_status["OnboardPose.csv"] = {"available": False, "path": str(pose_path) if pose_path else None}

    if baro_path is not None and baro_path.exists():
        baro_df = read_clean_csv(baro_path)

        baro_map = {
            "Pressure": "barometer_pressure",
            "Altitude": "barometer_altitude_m",
            "Temperature": "barometer_temperature",
            "Error_count": "barometer_error_count",
        }

        enriched, matched = merge_nearest_timestamp(
            enriched,
            baro_df,
            rename_map=baro_map,
            tolerance=barometer_tolerance,
        )

        source_status["BarometricPressure.csv"] = {
            "available": True,
            "path": str(baro_path),
            "rows": int(len(baro_df)),
            "columns": list(baro_df.columns),
            "nearest_timestamp_tolerance": float(barometer_tolerance),
            "matched_rows": int(matched),
        }
    else:
        source_status["BarometricPressure.csv"] = {"available": False, "path": str(baro_path) if baro_path else None}

    enriched, selection_summary = select_height_and_yaw(enriched)

    metadata_dir = output_dir / "metadata"
    report_dir = output_dir / "reports" / STAGE_NAME
    metadata_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    out_csv = metadata_dir / "synchronized_frames_enriched.csv"
    summary_json = report_dir / "sync_enrichment_summary.json"

    enriched.to_csv(out_csv, index=False)

    summary = {
        "dataset_name": dataset_name,
        "stage": STAGE_NAME,
        "input_sync_csv": str(sync_csv),
        "output_enriched_csv": str(out_csv),
        "frames": int(len(enriched)),
        "source_status": source_status,
        "selection_summary": selection_summary,
        "columns": list(enriched.columns),
        "important_note": (
            "height_agl_m and yaw_deg are added for metric visual-motion scaling. "
            "For Zurich sample, height may be approximate/debug because true AGL is not clearly available. "
            "GNSS/reference remains for evaluation only."
        ),
    }

    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary
