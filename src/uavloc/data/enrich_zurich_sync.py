from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from uavloc.relative.orb_relative_motion import resolve_project_paths


STAGE_NAME = "08b_sync_enrichment"


def read_clean_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, engine="python")
    df.columns = [str(c).strip() for c in df.columns]

    # Drop empty columns caused by trailing commas.
    keep_cols = []
    for c in df.columns:
        c_clean = str(c).strip()
        if c_clean == "" or c_clean.lower().startswith("unnamed"):
            continue
        keep_cols.append(c)

    df = df[keep_cols].copy()

    # Convert numeric-looking columns, but keep text columns safely.
    for c in df.columns:
        try:
            df[c] = pd.to_numeric(df[c], errors="raise")
        except Exception:
            pass

    return df


def find_raw_file(raw_dir: Path, filename: str) -> Optional[Path]:
    """Find Zurich MAV raw log files robustly, including Log_Files subfolder."""
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


def numeric_array(df: pd.DataFrame, col: str) -> np.ndarray:
    return pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)


def maybe_angle_to_deg(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)

    finite = values[np.isfinite(values)]

    if len(finite) == 0:
        return values

    max_abs = float(np.nanmax(np.abs(finite)))

    # If values look like radians, convert to degrees.
    if max_abs <= 7.0:
        return np.rad2deg(values)

    return values


def merge_nearest_timestamp(
    sync_df: pd.DataFrame,
    telemetry_df: pd.DataFrame,
    rename_map: Dict[str, str],
    tolerance: Optional[float],
) -> pd.DataFrame:
    if "timestamp" not in sync_df.columns or "timestamp" not in telemetry_df.columns:
        return sync_df

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

    # Zurich timestamps are integer-like microsecond values.
    # Pandas merge_asof requires tolerance type to match timestamp dtype.
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

    return merged


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

    out["gt_interpolated_by_imgid"] = True

    return out


def select_height_and_yaw(df: pd.DataFrame) -> tuple[pd.DataFrame, Dict[str, Any]]:
    out = df.copy()
    summary: Dict[str, Any] = {}

    height_source = None

    if "pose_height_m" in out.columns:
        h = pd.to_numeric(out["pose_height_m"], errors="coerce").to_numpy(dtype=float)
        finite = h[np.isfinite(h)]

        if len(finite) > 0 and 0.1 <= np.nanmedian(finite) <= 200.0:
            out["height_agl_m"] = h
            height_source = "pose_height_m"

    if height_source is None and "gt_z_minus_gps_z_m" in out.columns:
        h = pd.to_numeric(out["gt_z_minus_gps_z_m"], errors="coerce").to_numpy(dtype=float)
        finite = h[np.isfinite(h)]

        if len(finite) > 0 and 0.1 <= np.nanmedian(np.abs(finite)) <= 200.0:
            out["height_agl_m"] = np.abs(h)
            height_source = "abs(gt_z_minus_gps_z_m)_approx"

    if height_source is None and "barometer_altitude_m" in out.columns:
        # Do not use absolute barometer altitude as AGL directly.
        # Convert it to relative height from the first valid value as a fallback diagnostic.
        alt = pd.to_numeric(out["barometer_altitude_m"], errors="coerce").to_numpy(dtype=float)
        finite = alt[np.isfinite(alt)]

        if len(finite) > 0:
            out["height_agl_m"] = np.abs(alt - finite[0])
            height_source = "relative_barometer_altitude_from_start_debug"

    yaw_source = None

    if "pose_azimuth_raw" in out.columns:
        yaw_raw = pd.to_numeric(out["pose_azimuth_raw"], errors="coerce").to_numpy(dtype=float)
        out["pose_azimuth_deg"] = maybe_angle_to_deg(yaw_raw)
        out["yaw_deg"] = out["pose_azimuth_deg"]
        yaw_source = "pose_azimuth_deg"

    if yaw_source is None and "gt_omega_deg" in out.columns:
        out["yaw_deg"] = pd.to_numeric(out["gt_omega_deg"], errors="coerce")
        yaw_source = "gt_omega_deg_interpolated"

    if "pose_veh_pitch_raw" in out.columns:
        pitch_raw = pd.to_numeric(out["pose_veh_pitch_raw"], errors="coerce").to_numpy(dtype=float)
        out["pose_veh_pitch_deg"] = maybe_angle_to_deg(pitch_raw)

    summary["selected_height_source"] = height_source
    summary["selected_yaw_source"] = yaw_source

    if "height_agl_m" in out.columns:
        h = pd.to_numeric(out["height_agl_m"], errors="coerce")
        summary["height_agl_m"] = {
            "valid_count": int(h.notna().sum()),
            "median": float(h.median()),
            "min": float(h.min()),
            "max": float(h.max()),
        }

    if "yaw_deg" in out.columns:
        y = pd.to_numeric(out["yaw_deg"], errors="coerce")
        summary["yaw_deg"] = {
            "valid_count": int(y.notna().sum()),
            "median": float(y.median()),
            "min": float(y.min()),
            "max": float(y.max()),
        }

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
            "rows": int(len(gt_df)),
            "columns": list(gt_df.columns),
        }
    else:
        source_status["GroundTruthAGL.csv"] = {"available": False}

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

        enriched = merge_nearest_timestamp(
            enriched,
            pose_df,
            rename_map=pose_map,
            tolerance=pose_tolerance,
        )

        matched = int(pd.to_numeric(enriched.get("pose_height_m"), errors="coerce").notna().sum()) if "pose_height_m" in enriched.columns else 0

        source_status["OnboardPose.csv"] = {
            "available": True,
            "rows": int(len(pose_df)),
            "columns": list(pose_df.columns),
            "nearest_timestamp_tolerance": pose_tolerance,
            "matched_rows": matched,
        }
    else:
        source_status["OnboardPose.csv"] = {"available": False}

    if baro_path is not None and baro_path.exists():
        baro_df = read_clean_csv(baro_path)

        baro_map = {
            "Pressure": "barometer_pressure",
            "Altitude": "barometer_altitude_m",
            "Temperature": "barometer_temperature",
            "Error_count": "barometer_error_count",
        }

        enriched = merge_nearest_timestamp(
            enriched,
            baro_df,
            rename_map=baro_map,
            tolerance=barometer_tolerance,
        )

        matched = int(pd.to_numeric(enriched.get("barometer_altitude_m"), errors="coerce").notna().sum()) if "barometer_altitude_m" in enriched.columns else 0

        source_status["BarometricPressure.csv"] = {
            "available": True,
            "rows": int(len(baro_df)),
            "columns": list(baro_df.columns),
            "nearest_timestamp_tolerance": barometer_tolerance,
            "matched_rows": matched,
        }
    else:
        source_status["BarometricPressure.csv"] = {"available": False}

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
            "GNSS/reference remains for evaluation only."
        ),
    }

    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary
