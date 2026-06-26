from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from uavloc.relative.orb_relative_motion import resolve_project_paths


STAGE_NAME = "08c_metric_input_inspection"


def find_raw_file(raw_dir: Path, filename: str) -> Optional[Path]:
    filename_lower = filename.lower()

    search_roots = [
        raw_dir,
        raw_dir / "Log_Files",
        raw_dir / "Log Files",
        Path.cwd() / "data" / "raw" / "zurich_mav_sample",
        Path.cwd() / "data" / "raw" / "zurich_mav_sample" / "Log_Files",
        Path.cwd() / "data" / "raw" / "zurich_mav_sample" / "Log Files",
    ]

    for root in search_roots:
        root = Path(root)

        direct = root / filename
        if direct.exists():
            return direct

        if not root.exists():
            continue

        for candidate in root.rglob("*"):
            if candidate.is_file() and candidate.name.lower() == filename_lower:
                return candidate

    return None


def read_clean_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, engine="python")
    df.columns = [str(c).strip() for c in df.columns]

    # Normalize known Zurich timestamp column spellings.
    timestamp_aliases = {
        "Timpstemp": "timestamp",
        "Timestamp": "timestamp",
        "timeStamp": "timestamp",
    }
    df = df.rename(columns={
        c: timestamp_aliases.get(str(c).strip(), str(c).strip())
        for c in df.columns
    })


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


def safe_numeric(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce")


def summarize_numeric(series: pd.Series) -> Dict[str, Any]:
    s = pd.to_numeric(series, errors="coerce")
    valid = s.dropna()

    out: Dict[str, Any] = {
        "rows": int(len(series)),
        "valid_count": int(valid.shape[0]),
        "nan_count": int(s.isna().sum()),
    }

    if valid.empty:
        out.update(
            {
                "min": None,
                "max": None,
                "mean": None,
                "median": None,
                "std": None,
                "first": None,
                "last": None,
                "unique_count": 0,
                "zero_count": 0,
                "zero_ratio": None,
                "is_constant_like": None,
            }
        )
        return out

    zero_count = int((valid == 0).sum())
    unique_count = int(valid.nunique(dropna=True))
    std = float(valid.std()) if len(valid) > 1 else 0.0

    out.update(
        {
            "min": float(valid.min()),
            "max": float(valid.max()),
            "mean": float(valid.mean()),
            "median": float(valid.median()),
            "std": std,
            "first": float(valid.iloc[0]),
            "last": float(valid.iloc[-1]),
            "unique_count": unique_count,
            "zero_count": zero_count,
            "zero_ratio": float(zero_count / len(valid)),
            "is_constant_like": bool(unique_count <= 1 or abs(std) < 1e-12),
            "delta_last_minus_first": float(valid.iloc[-1] - valid.iloc[0]),
            "q05": float(valid.quantile(0.05)),
            "q95": float(valid.quantile(0.95)),
        }
    )

    return out


def guess_angle_unit(series: pd.Series) -> str:
    s = pd.to_numeric(series, errors="coerce").dropna()

    if s.empty:
        return "unknown_empty"

    max_abs = float(np.nanmax(np.abs(s)))

    if max_abs <= 2 * math.pi + 0.5:
        return "likely_radians"

    if max_abs <= 360.0:
        return "likely_degrees"

    return "unknown_large_angle_values"


def nearest_timestamp_stats(
    sync_df: pd.DataFrame,
    telemetry_df: pd.DataFrame,
    label: str,
    tolerance: Optional[int] = None,
) -> Dict[str, Any]:
    if "timestamp" not in sync_df.columns or "timestamp" not in telemetry_df.columns:
        return {"available": False, "reason": "timestamp column missing"}

    left = sync_df[["timestamp"]].copy()
    right = telemetry_df[["timestamp"]].copy()

    left["timestamp"] = pd.to_numeric(left["timestamp"], errors="coerce")
    right["timestamp"] = pd.to_numeric(right["timestamp"], errors="coerce")

    left = left[left["timestamp"].notna()].copy()
    right = right[right["timestamp"].notna()].copy()

    if left.empty or right.empty:
        return {"available": False, "reason": "empty valid timestamps"}

    left["_left_ts"] = left["timestamp"].round().astype("int64")
    right["_right_ts"] = right["timestamp"].round().astype("int64")

    left = left.rename(columns={"timestamp": "timestamp_left"})
    right = right.rename(columns={"timestamp": "timestamp_right"})

    left = left.sort_values("_left_ts")
    right = right.sort_values("_right_ts")

    merged = pd.merge_asof(
        left,
        right,
        left_on="_left_ts",
        right_on="_right_ts",
        direction="nearest",
        tolerance=tolerance,
    )

    dt = merged["timestamp_right"] - merged["timestamp_left"]
    valid_dt = dt.dropna()

    if valid_dt.empty:
        return {
            "available": True,
            "label": label,
            "matched_count": 0,
            "reason": "no matches inside tolerance",
        }

    return {
        "available": True,
        "label": label,
        "matched_count": int(valid_dt.shape[0]),
        "sync_rows": int(sync_df.shape[0]),
        "median_abs_dt": float(valid_dt.abs().median()),
        "max_abs_dt": float(valid_dt.abs().max()),
        "mean_abs_dt": float(valid_dt.abs().mean()),
    }


def add_column_report(
    rows: List[Dict[str, Any]],
    source: str,
    df: pd.DataFrame,
    col: str,
    candidate_meaning: str,
    unit_guess: str = "",
    note: str = "",
) -> None:
    if col not in df.columns:
        rows.append(
            {
                "source": source,
                "column": col,
                "exists": False,
                "candidate_meaning": candidate_meaning,
                "unit_guess": unit_guess,
                "note": "missing",
            }
        )
        return

    summary = summarize_numeric(df[col])

    rows.append(
        {
            "source": source,
            "column": col,
            "exists": True,
            "candidate_meaning": candidate_meaning,
            "unit_guess": unit_guess,
            "note": note,
            **summary,
        }
    )


def build_decisions(candidate_df: pd.DataFrame) -> Dict[str, Any]:
    decisions: Dict[str, Any] = {
        "height_candidates": [],
        "yaw_candidates": [],
        "warnings": [],
        "recommendation": {},
    }

    for _, row in candidate_df.iterrows():
        col = row.get("column")
        exists = bool(row.get("exists", False))
        if not exists:
            continue

        candidate_meaning = str(row.get("candidate_meaning", "")).lower()
        median = row.get("median")
        zero_ratio = row.get("zero_ratio")
        is_constant = row.get("is_constant_like")

        item = row.to_dict()

        if "height" in candidate_meaning or "altitude" in candidate_meaning or "agl" in candidate_meaning:
            decisions["height_candidates"].append(item)

        if "yaw" in candidate_meaning or "heading" in candidate_meaning or "azimuth" in candidate_meaning:
            decisions["yaw_candidates"].append(item)

        if zero_ratio is not None and zero_ratio > 0.95:
            decisions["warnings"].append(f"{row['source']}::{col} is mostly/all zero.")

        if is_constant is True:
            decisions["warnings"].append(f"{row['source']}::{col} is constant-like.")

        if isinstance(median, (int, float)) and "absolute altitude" in candidate_meaning and median > 100:
            decisions["warnings"].append(
                f"{row['source']}::{col} looks like absolute altitude, not AGL height."
            )

    # Simple automatic recommendations for this project stage.
    height_recommendation = None
    yaw_recommendation = None

    for _, row in candidate_df.iterrows():
        if not bool(row.get("exists", False)):
            continue

        col = str(row.get("column"))
        median = row.get("median")
        zero_ratio = row.get("zero_ratio")
        meaning = str(row.get("candidate_meaning", "")).lower()

        if col == "Height" and isinstance(median, (int, float)) and 0.1 <= median <= 200 and (zero_ratio is None or zero_ratio < 0.95):
            height_recommendation = "OnboardPose::Height"
            break

    if height_recommendation is None:
        for _, row in candidate_df.iterrows():
            if str(row.get("column")) == "gt_z_minus_gps_z_m":
                height_recommendation = "Derived::abs(gt_z_minus_gps_z_m) only as approximate/debug height"
                break

    for _, row in candidate_df.iterrows():
        if str(row.get("column")) == "Azimuth":
            zero_ratio = row.get("zero_ratio")
            if zero_ratio is not None and zero_ratio < 0.95:
                yaw_recommendation = "OnboardPose::Azimuth"
                break

    if yaw_recommendation is None:
        for _, row in candidate_df.iterrows():
            if str(row.get("column")) == "omega_gt":
                yaw_recommendation = "GroundTruthAGL::omega_gt"
                break

    decisions["recommendation"] = {
        "height": height_recommendation,
        "yaw_or_heading": yaw_recommendation,
        "note": (
            "Recommendations are diagnostic. Verify dataset documentation before treating any column as true AGL or yaw."
        ),
    }

    return decisions


def run_inspection(config_path: str | Path) -> Dict[str, Any]:
    _, _, raw_dir, output_dir, dataset_name = resolve_project_paths(config_path)

    report_dir = output_dir / "reports" / STAGE_NAME
    metadata_dir = output_dir / "metadata" / STAGE_NAME
    report_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    sync_path = output_dir / "metadata" / "synchronized_frames.csv"
    enriched_path = output_dir / "metadata" / "synchronized_frames_enriched.csv"

    gt_path = find_raw_file(raw_dir, "GroundTruthAGL.csv")
    pose_path = find_raw_file(raw_dir, "OnboardPose.csv")
    baro_path = find_raw_file(raw_dir, "BarometricPressure.csv")

    loaded: Dict[str, pd.DataFrame] = {}
    files: Dict[str, Any] = {}

    for label, path in [
        ("synchronized_frames", sync_path if sync_path.exists() else None),
        ("synchronized_frames_enriched", enriched_path if enriched_path.exists() else None),
        ("GroundTruthAGL", gt_path),
        ("OnboardPose", pose_path),
        ("BarometricPressure", baro_path),
    ]:
        if path is None or not Path(path).exists():
            files[label] = {"available": False, "path": str(path) if path else None}
            continue

        df = read_clean_csv(Path(path))
        loaded[label] = df
        files[label] = {
            "available": True,
            "path": str(path),
            "rows": int(len(df)),
            "columns": list(df.columns),
        }

    candidate_rows: List[Dict[str, Any]] = []

    if "GroundTruthAGL" in loaded:
        gt = loaded["GroundTruthAGL"]

        if "z_gt" in gt.columns and "z_gps" in gt.columns:
            gt["gt_z_minus_gps_z_m"] = safe_numeric(gt, "z_gt") - safe_numeric(gt, "z_gps")
            gt["abs_gt_z_minus_gps_z_m"] = gt["gt_z_minus_gps_z_m"].abs()
            loaded["GroundTruthAGL"] = gt

        add_column_report(candidate_rows, "GroundTruthAGL", gt, "omega_gt", "yaw/heading candidate", guess_angle_unit(gt["omega_gt"]) if "omega_gt" in gt.columns else "")
        add_column_report(candidate_rows, "GroundTruthAGL", gt, "phi_gt", "pitch/attitude candidate", guess_angle_unit(gt["phi_gt"]) if "phi_gt" in gt.columns else "")
        add_column_report(candidate_rows, "GroundTruthAGL", gt, "kappa_gt", "roll/attitude candidate", guess_angle_unit(gt["kappa_gt"]) if "kappa_gt" in gt.columns else "")
        add_column_report(candidate_rows, "GroundTruthAGL", gt, "z_gt", "absolute altitude / z candidate", "meters")
        add_column_report(candidate_rows, "GroundTruthAGL", gt, "z_gps", "absolute altitude / gps z candidate", "meters")
        add_column_report(candidate_rows, "GroundTruthAGL", gt, "gt_z_minus_gps_z_m", "derived height/altitude-difference candidate", "meters", "not guaranteed true AGL")
        add_column_report(candidate_rows, "GroundTruthAGL", gt, "abs_gt_z_minus_gps_z_m", "derived positive height/altitude-difference candidate", "meters", "debug only unless verified")

    if "OnboardPose" in loaded:
        pose = loaded["OnboardPose"]

        add_column_report(candidate_rows, "OnboardPose", pose, "Azimuth", "yaw/heading/azimuth candidate", guess_angle_unit(pose["Azimuth"]) if "Azimuth" in pose.columns else "")
        add_column_report(candidate_rows, "OnboardPose", pose, "Height", "AGL height candidate", "meters")
        add_column_report(candidate_rows, "OnboardPose", pose, "Altitude", "absolute altitude candidate", "meters")
        add_column_report(candidate_rows, "OnboardPose", pose, "veh_pitch", "vehicle pitch candidate", guess_angle_unit(pose["veh_pitch"]) if "veh_pitch" in pose.columns else "")
        add_column_report(candidate_rows, "OnboardPose", pose, "Vel_x", "velocity candidate", "m/s maybe")
        add_column_report(candidate_rows, "OnboardPose", pose, "Vel_y", "velocity candidate", "m/s maybe")
        add_column_report(candidate_rows, "OnboardPose", pose, "Vel_z", "velocity candidate", "m/s maybe")
        add_column_report(candidate_rows, "OnboardPose", pose, "GPS_on", "GPS status flag candidate", "")

    if "BarometricPressure" in loaded:
        baro = loaded["BarometricPressure"]

        add_column_report(candidate_rows, "BarometricPressure", baro, "Altitude", "barometer absolute altitude candidate", "meters")
        add_column_report(candidate_rows, "BarometricPressure", baro, "Pressure", "barometric pressure", "")
        add_column_report(candidate_rows, "BarometricPressure", baro, "Temperature", "temperature", "")
        add_column_report(candidate_rows, "BarometricPressure", baro, "Error_count", "barometer error flag", "")

    if "synchronized_frames_enriched" in loaded:
        enriched = loaded["synchronized_frames_enriched"]
        add_column_report(candidate_rows, "EnrichedSync", enriched, "height_agl_m", "selected height/AGL used by pipeline", "meters")
        add_column_report(candidate_rows, "EnrichedSync", enriched, "yaw_deg", "selected yaw used by pipeline", "degrees")
        add_column_report(candidate_rows, "EnrichedSync", enriched, "pose_height_m", "merged pose height", "meters")
        add_column_report(candidate_rows, "EnrichedSync", enriched, "pose_azimuth_deg", "merged pose azimuth", "degrees")
        add_column_report(candidate_rows, "EnrichedSync", enriched, "gt_omega_deg", "interpolated gt heading/yaw", "degrees")
        add_column_report(candidate_rows, "EnrichedSync", enriched, "barometer_altitude_m", "merged barometer altitude", "meters")

    candidate_df = pd.DataFrame(candidate_rows)

    timestamp_checks: Dict[str, Any] = {}

    if "synchronized_frames" in loaded and "OnboardPose" in loaded:
        timestamp_checks["sync_to_onboard_pose"] = nearest_timestamp_stats(
            loaded["synchronized_frames"],
            loaded["OnboardPose"],
            "sync_to_onboard_pose",
            tolerance=200000,
        )

    if "synchronized_frames" in loaded and "BarometricPressure" in loaded:
        timestamp_checks["sync_to_barometer"] = nearest_timestamp_stats(
            loaded["synchronized_frames"],
            loaded["BarometricPressure"],
            "sync_to_barometer",
            tolerance=500000,
        )

    decisions = build_decisions(candidate_df)

    candidate_csv = metadata_dir / "metric_input_candidate_columns.csv"
    summary_json = report_dir / "metric_input_inspection_summary.json"

    candidate_df.to_csv(candidate_csv, index=False)

    summary = {
        "dataset_name": dataset_name,
        "stage": STAGE_NAME,
        "raw_dir": str(raw_dir),
        "files": files,
        "timestamp_checks": timestamp_checks,
        "decisions": decisions,
        "outputs": {
            "candidate_csv": str(candidate_csv),
            "summary_json": str(summary_json),
        },
    }

    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary, candidate_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect Zurich MAV metric-scaling input columns.")
    parser.add_argument("--config", required=True, help="Path to dataset YAML config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    summary, candidate_df = run_inspection(args.config)

    print("Zurich metric input inspection complete")
    print("---------------------------------------")
    print(f"Dataset:      {summary['dataset_name']}")
    print(f"Raw dir:      {summary['raw_dir']}")

    print("\nFiles")
    print("-----")
    for label, item in summary["files"].items():
        print(f"{label}: available={item['available']} rows={item.get('rows')} path={item.get('path')}")

    print("\nTimestamp checks")
    print("----------------")
    for label, item in summary["timestamp_checks"].items():
        print(label, item)

    print("\nMain candidate columns")
    print("----------------------")
    display_cols = [
        "source",
        "column",
        "exists",
        "candidate_meaning",
        "unit_guess",
        "valid_count",
        "zero_ratio",
        "is_constant_like",
        "median",
        "min",
        "max",
        "std",
        "note",
    ]
    existing_display_cols = [c for c in display_cols if c in candidate_df.columns]

    important = candidate_df[
        candidate_df["candidate_meaning"].astype(str).str.contains(
            "height|altitude|yaw|heading|azimuth|pitch|roll", case=False, na=False
        )
    ].copy()

    print(important[existing_display_cols].to_string(index=False))

    print("\nWarnings")
    print("--------")
    for w in summary["decisions"]["warnings"]:
        print("-", w)

    print("\nRecommendation")
    print("--------------")
    for k, v in summary["decisions"]["recommendation"].items():
        print(f"{k}: {v}")

    print("\nOutputs")
    print("-------")
    for k, v in summary["outputs"].items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
