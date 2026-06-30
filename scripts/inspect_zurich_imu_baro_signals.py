#!/usr/bin/env python3
"""
Block 10A.3 / 10A.4 — Zurich IMU and barometer signal diagnostics.

This script NOT implement fusion.
It checks whether accelerometer, gyroscope, barometer, and OnboardPose signals are
physically plausible and useful for future EKF/ESKF/OpenVINS/VINS work.

Outputs:
- imu_baro_signal_stats.csv
- camera_nearest_sensor_values.csv
- imu_baro_signal_summary.json
- accel_axes_and_norm.png
- gyro_axes_and_norm.png
- onboard_pose_signal_checks.png
- barometer_altitude_signal.png
- barometer_vs_onboard_pose_altitude.png
- raw_vs_onboard_pose_imu_norms.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


LOG_SPECS: Dict[str, Dict[str, Any]] = {
    "accelerometer": {
        "filenames": ["RawAccel.csv", "RawAccelerometer.csv", "RawAccelerometerData.csv"],
        "columns": [
            "timestamp",
            "error_count",
            "x",
            "y",
            "z",
            "temperature",
            "range_rad_s",
            "scaling",
            "x_raw",
            "y_raw",
            "z_raw",
            "temperature_raw",
        ],
    },
    "gyroscope": {
        "filenames": ["RawGyro.csv", "RawGyroscope.csv", "RawGyroscopeData.csv"],
        "columns": [
            "timestamp",
            "error_count",
            "x",
            "y",
            "z",
            "temperature",
            "range_rad_s",
            "scaling",
            "x_raw",
            "y_raw",
            "z_raw",
            "temperature_raw",
        ],
    },
    "barometer": {
        "filenames": ["BarometricPressure.csv", "Barometer.csv"],
        "columns": ["timestamp", "pressure", "altitude", "temperature"],
    },
    "onboard_pose": {
        "filenames": ["OnboardPose.csv"],
        "columns": [
            "timestamp",
            "omega_x",
            "omega_y",
            "omega_z",
            "accel_x",
            "accel_y",
            "accel_z",
            "vel_x",
            "vel_y",
            "vel_z",
            "acc_bias_x",
            "acc_bias_y",
            "acc_bias_z",
            "azimuth",
            "attitude_w",
            "attitude_x",
            "attitude_y",
            "attitude_z",
            "height",
            "altitude",
            "veh_pitch",
            "tether_angle",
            "tether_angle_dot",
            "tether_force",
            "gps_on",
        ],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f)
    return loaded if isinstance(loaded, dict) else {}


def deep_find_first(obj: Any, key_candidates: List[str]) -> Optional[Any]:
    if isinstance(obj, dict):
        for key, value in obj.items():
            if str(key) in key_candidates:
                return value
        for value in obj.values():
            found = deep_find_first(value, key_candidates)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for item in obj:
            found = deep_find_first(item, key_candidates)
            if found is not None:
                return found
    return None


def resolve_path(project_root: Path, value: Any) -> Optional[Path]:
    if value is None:
        return None
    p = Path(str(value))
    return p if p.is_absolute() else project_root / p


def infer_dataset_name(cfg: Dict[str, Any], config_path: Path) -> str:
    for key in ["dataset_name", "name"]:
        found = deep_find_first(cfg, [key])
        if found is not None and str(found).strip():
            return str(found)
    return config_path.stem.replace("dataset_", "")


def infer_dirs(
    cfg: Dict[str, Any],
    config_path: Path,
    project_root: Path,
) -> Tuple[Path, Path, str]:
    dataset_name = infer_dataset_name(cfg, config_path)

    raw_value = deep_find_first(
        cfg,
        ["raw_data_dir", "raw_dir", "data_raw_dir", "dataset_dir", "raw_dataset_dir"],
    )
    output_value = deep_find_first(
        cfg,
        ["output_dir", "outputs_dir", "output_root"],
    )

    raw_dir = resolve_path(project_root, raw_value)
    output_dir = resolve_path(project_root, output_value)

    if raw_dir is None:
        raw_dir = project_root / "data" / "raw" / dataset_name
    if output_dir is None:
        output_dir = project_root / "outputs" / dataset_name

    return raw_dir, output_dir, dataset_name


def find_first_existing(base_dir: Path, filenames: List[str]) -> Optional[Path]:
    for name in filenames:
        p = base_dir / name
        if p.exists():
            return p

    for name in filenames:
        matches = list(base_dir.rglob(name))
        if matches:
            return matches[0]

    return None


def read_expected_csv(path: Path, columns: List[str]) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        header=None,
        names=columns,
        usecols=list(range(len(columns))),
        engine="python",
        skipinitialspace=True,
    )

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "timestamp" in df.columns:
        df = df[df["timestamp"].notna()].copy()

    return df.reset_index(drop=True)


def read_sync_csv(output_dir: Path) -> Tuple[Path, pd.DataFrame]:
    candidates = [
        output_dir / "metadata" / "synchronized_frames_enriched.csv",
        output_dir / "metadata" / "synchronized_frames.csv",
    ]

    for p in candidates:
        if p.exists():
            return p, pd.read_csv(p)

    raise FileNotFoundError(
        "Missing synchronized frames CSV. Expected synchronized_frames_enriched.csv or synchronized_frames.csv"
    )


def numeric_array(series: pd.Series) -> np.ndarray:
    arr = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    return arr[np.isfinite(arr)]


def infer_timestamp_scale(ts_raw: np.ndarray) -> Tuple[float, str]:
    finite = ts_raw[np.isfinite(ts_raw)]
    if finite.size < 2:
        return 1.0, "unknown"

    diffs = np.diff(finite)
    pos_diffs = diffs[diffs > 0]

    abs_median = float(np.nanmedian(np.abs(finite)))
    dt_median = float(np.nanmedian(pos_diffs)) if pos_diffs.size else np.nan

    if abs_median > 1e14:
        return 1e-6, "microseconds_inferred_from_large_epoch"
    if abs_median > 1e11:
        return 1e-3, "milliseconds_inferred_from_large_epoch"
    if np.isfinite(dt_median) and dt_median > 1000:
        return 1e-6, "microseconds_inferred_from_delta"
    if np.isfinite(dt_median) and dt_median > 10 and abs_median > 1e8:
        return 1e-3, "milliseconds_inferred_from_delta"

    return 1.0, "seconds_or_normalized"


def add_time_seconds(df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    if "timestamp" not in df.columns:
        out = df.copy()
        out["time_s"] = np.nan
        return out, "missing_timestamp"

    ts_raw = numeric_array(df["timestamp"])
    scale, label = infer_timestamp_scale(ts_raw)

    out = df.copy()
    out["time_s"] = pd.to_numeric(out["timestamp"], errors="coerce") * scale
    return out, label


def vector_norm(df: pd.DataFrame, cols: List[str]) -> np.ndarray:
    if not all(col in df.columns for col in cols):
        return np.array([], dtype=float)

    arr = df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    valid = np.all(np.isfinite(arr), axis=1)
    arr = arr[valid]

    if arr.size == 0:
        return np.array([], dtype=float)

    return np.linalg.norm(arr, axis=1)


def stats(arr: np.ndarray, prefix: str = "") -> Dict[str, Optional[float]]:
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            f"{prefix}count": 0,
            f"{prefix}min": None,
            f"{prefix}p05": None,
            f"{prefix}median": None,
            f"{prefix}mean": None,
            f"{prefix}p95": None,
            f"{prefix}max": None,
            f"{prefix}std": None,
        }

    return {
        f"{prefix}count": int(arr.size),
        f"{prefix}min": float(np.min(arr)),
        f"{prefix}p05": float(np.percentile(arr, 5)),
        f"{prefix}median": float(np.median(arr)),
        f"{prefix}mean": float(np.mean(arr)),
        f"{prefix}p95": float(np.percentile(arr, 95)),
        f"{prefix}max": float(np.max(arr)),
        f"{prefix}std": float(np.std(arr)),
    }


def series_stats(df: pd.DataFrame, col: str) -> Dict[str, Any]:
    if col not in df.columns:
        return {"column": col, "available": False}

    arr = numeric_array(df[col])
    out = {"column": col, "available": True}
    out.update(stats(arr))

    if arr.size:
        rounded = np.round(arr, 9)
        out["unique_count_rounded_1e9"] = int(pd.Series(rounded).nunique())
        out["is_constant_or_nearly_constant"] = bool(np.nanstd(arr) < 1e-9)
    else:
        out["unique_count_rounded_1e9"] = 0
        out["is_constant_or_nearly_constant"] = True

    return out


def accel_unit_guess(median_norm: Optional[float]) -> str:
    if median_norm is None:
        return "missing"
    if 7.0 <= median_norm <= 12.5:
        return "likely_m_per_s2"
    if 0.7 <= median_norm <= 1.3:
        return "likely_g_units"
    if 600.0 <= median_norm <= 1300.0:
        return "possibly_milli_g_or_raw_scaled"
    return "unclear"


def gyro_unit_guess(median_norm: Optional[float]) -> str:
    if median_norm is None:
        return "missing"
    if median_norm < 0.5:
        return "small_motion_or_rad_per_s_possible"
    if median_norm < 10.0:
        return "rad_per_s_possible_but_not_confirmed"
    if median_norm < 500.0:
        return "deg_per_s_or_raw_scaled_possible"
    return "unclear_or_raw_units"


def correlation(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    mask = np.isfinite(a) & np.isfinite(b)
    a = a[mask]
    b = b[mask]
    if a.size < 3 or np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def nearest_rows(
    query_time_s: np.ndarray,
    ref_df: pd.DataFrame,
    value_cols: List[str],
    prefix: str,
) -> pd.DataFrame:
    out = pd.DataFrame(index=np.arange(len(query_time_s)))

    if "time_s" not in ref_df.columns or len(ref_df) == 0:
        for col in value_cols:
            out[f"{prefix}_{col}"] = np.nan
        out[f"{prefix}_nearest_dt_s"] = np.nan
        return out

    ref = ref_df.copy()
    ref = ref[np.isfinite(ref["time_s"])].copy()
    ref = ref.sort_values("time_s").reset_index(drop=True)

    if len(ref) == 0:
        for col in value_cols:
            out[f"{prefix}_{col}"] = np.nan
        out[f"{prefix}_nearest_dt_s"] = np.nan
        return out

    ref_t = ref["time_s"].to_numpy(dtype=float)
    q = query_time_s.astype(float)

    idx = np.searchsorted(ref_t, q)
    left = np.clip(idx - 1, 0, len(ref_t) - 1)
    right = np.clip(idx, 0, len(ref_t) - 1)

    left_dt = ref_t[left] - q
    right_dt = ref_t[right] - q
    choose_right = np.abs(right_dt) < np.abs(left_dt)
    best_idx = np.where(choose_right, right, left)
    best_dt = ref_t[best_idx] - q

    for col in value_cols:
        if col in ref.columns:
            out[f"{prefix}_{col}"] = ref[col].to_numpy()[best_idx]
        else:
            out[f"{prefix}_{col}"] = np.nan

    out[f"{prefix}_nearest_dt_s"] = best_dt
    return out


def downsample(df: pd.DataFrame, max_points: int = 8000) -> pd.DataFrame:
    if len(df) <= max_points:
        return df
    idx = np.linspace(0, len(df) - 1, max_points).astype(int)
    return df.iloc[idx].copy()


def plot_axes_and_norm(
    df: pd.DataFrame,
    cols: List[str],
    title: str,
    out_path: Path,
    norm_label: str,
) -> None:
    plt.figure(figsize=(11, 5))

    if "time_s" not in df.columns or not all(c in df.columns for c in cols):
        plt.text(0.5, 0.5, "Required columns missing", ha="center", va="center")
        plt.axis("off")
    else:
        d = df[np.isfinite(df["time_s"])].copy()
        d = downsample(d)

        t = d["time_s"].to_numpy(dtype=float)
        t = t - np.nanmin(t)

        for col in cols:
            plt.plot(t, d[col].to_numpy(dtype=float), label=col, linewidth=0.8)

        n = vector_norm(d, cols)
        if n.size == len(d):
            plt.plot(t, n, label=norm_label, linewidth=1.1)

        plt.xlabel("Time from stream start [s]")
        plt.ylabel("Signal value")
        plt.title(title)
        plt.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_onboard_pose_checks(pose_df: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(11, 5))

    if "time_s" not in pose_df.columns or len(pose_df) == 0:
        plt.text(0.5, 0.5, "OnboardPose missing", ha="center", va="center")
        plt.axis("off")
    else:
        d = pose_df[np.isfinite(pose_df["time_s"])].copy()
        d = downsample(d)
        t = d["time_s"].to_numpy(dtype=float)
        t = t - np.nanmin(t)

        plotted = False

        for col in ["height", "altitude", "azimuth", "veh_pitch"]:
            if col in d.columns:
                plt.plot(t, d[col].to_numpy(dtype=float), label=col, linewidth=0.9)
                plotted = True

        if plotted:
            plt.xlabel("Time from OnboardPose start [s]")
            plt.ylabel("Signal value")
            plt.title("OnboardPose key signal checks")
            plt.legend()
        else:
            plt.text(0.5, 0.5, "No key OnboardPose columns found", ha="center", va="center")
            plt.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_barometer_signal(baro_df: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(11, 5))

    if "time_s" not in baro_df.columns or len(baro_df) == 0:
        plt.text(0.5, 0.5, "Barometer missing", ha="center", va="center")
        plt.axis("off")
    else:
        d = baro_df[np.isfinite(baro_df["time_s"])].copy()
        d = downsample(d)
        t = d["time_s"].to_numpy(dtype=float)
        t = t - np.nanmin(t)

        plotted = False

        for col in ["altitude", "pressure", "temperature"]:
            if col in d.columns:
                vals = d[col].to_numpy(dtype=float)
                vals_norm = vals - np.nanmedian(vals)
                plt.plot(t, vals_norm, label=f"{col} minus median", linewidth=0.9)
                plotted = True

        if plotted:
            plt.xlabel("Time from barometer start [s]")
            plt.ylabel("Median-centered value")
            plt.title("Barometer altitude / pressure / temperature signal")
            plt.legend()
        else:
            plt.text(0.5, 0.5, "No barometer columns found", ha="center", va="center")
            plt.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_baro_vs_pose_altitude(baro_pose_df: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(11, 5))

    required = ["baro_time_s", "baro_altitude", "pose_altitude"]
    if not all(c in baro_pose_df.columns for c in required) or len(baro_pose_df) == 0:
        plt.text(0.5, 0.5, "Comparison data missing", ha="center", va="center")
        plt.axis("off")
    else:
        d = baro_pose_df.dropna(subset=required).copy()
        d = downsample(d)

        if len(d) == 0:
            plt.text(0.5, 0.5, "No valid comparison rows", ha="center", va="center")
            plt.axis("off")
        else:
            t = d["baro_time_s"].to_numpy(dtype=float)
            t = t - np.nanmin(t)

            baro = d["baro_altitude"].to_numpy(dtype=float)
            pose = d["pose_altitude"].to_numpy(dtype=float)

            plt.plot(t, baro - np.nanmedian(baro), label="barometer altitude minus median", linewidth=0.9)
            plt.plot(t, pose - np.nanmedian(pose), label="pose altitude minus median", linewidth=0.9)
            plt.xlabel("Time from comparison start [s]")
            plt.ylabel("Relative altitude signal [m-like units]")
            plt.title("Barometer altitude versus OnboardPose altitude")
            plt.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_raw_vs_pose_norms(compare_df: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(11, 5))

    cols = [
        "raw_accel_norm",
        "pose_accel_norm",
        "raw_gyro_norm",
        "pose_omega_norm",
    ]

    if not any(c in compare_df.columns for c in cols):
        plt.text(0.5, 0.5, "No IMU comparison columns", ha="center", va="center")
        plt.axis("off")
    else:
        d = compare_df.copy()
        d = downsample(d)
        t = d["raw_time_s"].to_numpy(dtype=float)
        t = t - np.nanmin(t)

        plotted = False
        for col in cols:
            if col in d.columns:
                vals = d[col].to_numpy(dtype=float)
                if np.isfinite(vals).any():
                    plt.plot(t, vals, label=col, linewidth=0.9)
                    plotted = True

        if plotted:
            plt.xlabel("Time from raw IMU start [s]")
            plt.ylabel("Vector norm")
            plt.title("Raw IMU norms versus nearest OnboardPose IMU-like fields")
            plt.legend()
        else:
            plt.text(0.5, 0.5, "No valid norm data", ha="center", va="center")
            plt.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_safe(v) for v in obj]
    if isinstance(obj, tuple):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return json_safe(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        value = float(obj)
        return value if np.isfinite(value) else None
    if isinstance(obj, float):
        return obj if np.isfinite(obj) else None
    return obj


def main() -> None:
    args = parse_args()

    project_root = Path.cwd()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = project_root / config_path

    cfg = load_yaml(config_path)
    raw_dir, output_dir, dataset_name = infer_dirs(cfg, config_path, project_root)

    metadata_dir = output_dir / "metadata" / "10a_sensor_signal_diagnostics"
    report_dir = output_dir / "reports" / "10a_sensor_signal_diagnostics"
    fig_dir = output_dir / "figures" / "10a_sensor_signal_diagnostics"

    metadata_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    streams: Dict[str, Dict[str, Any]] = {}

    for name, spec in LOG_SPECS.items():
        path = find_first_existing(raw_dir, spec["filenames"])
        if path is None:
            streams[name] = {
                "path": None,
                "df": pd.DataFrame(),
                "timestamp_unit": "file_missing",
            }
            continue

        df = read_expected_csv(path, spec["columns"])
        df, unit = add_time_seconds(df)

        streams[name] = {
            "path": path,
            "df": df,
            "timestamp_unit": unit,
        }

    sync_path, sync_df = read_sync_csv(output_dir)
    sync_df = sync_df.copy()

    if "timestamp" not in sync_df.columns:
        raise RuntimeError("synchronized frame CSV does not contain timestamp column")

    camera_ts = pd.to_numeric(sync_df["timestamp"], errors="coerce").to_numpy(dtype=float)
    cam_scale, cam_unit = infer_timestamp_scale(camera_ts[np.isfinite(camera_ts)])
    sync_df["time_s"] = camera_ts * cam_scale

    accel_df = streams["accelerometer"]["df"]
    gyro_df = streams["gyroscope"]["df"]
    baro_df = streams["barometer"]["df"]
    pose_df = streams["onboard_pose"]["df"]

    accel_norm = vector_norm(accel_df, ["x", "y", "z"])
    gyro_norm = vector_norm(gyro_df, ["x", "y", "z"])
    pose_accel_norm = vector_norm(pose_df, ["accel_x", "accel_y", "accel_z"])
    pose_omega_norm = vector_norm(pose_df, ["omega_x", "omega_y", "omega_z"])
    pose_quat_norm = vector_norm(pose_df, ["attitude_w", "attitude_x", "attitude_y", "attitude_z"])

    signal_rows: List[Dict[str, Any]] = []

    for stream_name, df in [
        ("accelerometer", accel_df),
        ("gyroscope", gyro_df),
        ("barometer", baro_df),
        ("onboard_pose", pose_df),
    ]:
        row: Dict[str, Any] = {
            "stream": stream_name,
            "path": str(streams[stream_name]["path"]) if streams[stream_name]["path"] else None,
            "rows": int(len(df)),
            "timestamp_unit": streams[stream_name]["timestamp_unit"],
            "time_min_s": float(np.nanmin(df["time_s"])) if "time_s" in df.columns and len(df) else None,
            "time_max_s": float(np.nanmax(df["time_s"])) if "time_s" in df.columns and len(df) else None,
        }

        if stream_name == "accelerometer":
            row.update(stats(accel_norm, prefix="norm_"))
            row["unit_guess"] = accel_unit_guess(row.get("norm_median"))
        elif stream_name == "gyroscope":
            row.update(stats(gyro_norm, prefix="norm_"))
            row["unit_guess"] = gyro_unit_guess(row.get("norm_median"))
        elif stream_name == "barometer":
            for col in ["altitude", "pressure", "temperature"]:
                if col in df.columns:
                    col_stats = stats(numeric_array(df[col]), prefix=f"{col}_")
                    row.update(col_stats)
        elif stream_name == "onboard_pose":
            row.update(stats(pose_accel_norm, prefix="pose_accel_norm_"))
            row.update(stats(pose_omega_norm, prefix="pose_omega_norm_"))
            row.update(stats(pose_quat_norm, prefix="pose_quat_norm_"))

        signal_rows.append(row)

    signal_stats_csv = metadata_dir / "imu_baro_signal_stats.csv"
    pd.DataFrame(signal_rows).to_csv(signal_stats_csv, index=False)

    # Build camera-nearest values for future fusion/debugging.
    camera_nearest = pd.DataFrame()
    camera_nearest["camera_row"] = np.arange(len(sync_df))
    camera_nearest["camera_time_s"] = sync_df["time_s"].to_numpy(dtype=float)

    if "imgid" in sync_df.columns:
        camera_nearest["imgid"] = pd.to_numeric(sync_df["imgid"], errors="coerce").to_numpy()

    camera_nearest = pd.concat(
        [
            camera_nearest,
            nearest_rows(sync_df["time_s"].to_numpy(dtype=float), accel_df, ["x", "y", "z"], "accel"),
            nearest_rows(sync_df["time_s"].to_numpy(dtype=float), gyro_df, ["x", "y", "z"], "gyro"),
            nearest_rows(sync_df["time_s"].to_numpy(dtype=float), baro_df, ["pressure", "altitude", "temperature"], "baro"),
            nearest_rows(
                sync_df["time_s"].to_numpy(dtype=float),
                pose_df,
                [
                    "omega_x",
                    "omega_y",
                    "omega_z",
                    "accel_x",
                    "accel_y",
                    "accel_z",
                    "vel_x",
                    "vel_y",
                    "vel_z",
                    "azimuth",
                    "attitude_w",
                    "attitude_x",
                    "attitude_y",
                    "attitude_z",
                    "height",
                    "altitude",
                    "veh_pitch",
                ],
                "pose",
            ),
        ],
        axis=1,
    )

    camera_nearest_csv = metadata_dir / "camera_nearest_sensor_values.csv"
    camera_nearest.to_csv(camera_nearest_csv, index=False)

    # Compare raw accelerometer and raw gyro against nearest OnboardPose IMU-like fields.
    compare_base = pd.DataFrame()
    if len(accel_df) and "time_s" in accel_df.columns:
        compare_base["raw_time_s"] = accel_df["time_s"].to_numpy(dtype=float)
        compare_base["raw_accel_x"] = accel_df["x"].to_numpy(dtype=float)
        compare_base["raw_accel_y"] = accel_df["y"].to_numpy(dtype=float)
        compare_base["raw_accel_z"] = accel_df["z"].to_numpy(dtype=float)
        compare_base["raw_accel_norm"] = vector_norm(accel_df, ["x", "y", "z"])

        nearest_pose_for_accel = nearest_rows(
            compare_base["raw_time_s"].to_numpy(dtype=float),
            pose_df,
            ["accel_x", "accel_y", "accel_z", "omega_x", "omega_y", "omega_z"],
            "pose",
        )
        compare_base = pd.concat([compare_base, nearest_pose_for_accel], axis=1)
        compare_base["pose_accel_norm"] = np.linalg.norm(
            compare_base[["pose_accel_x", "pose_accel_y", "pose_accel_z"]].to_numpy(dtype=float),
            axis=1,
        )
        compare_base["pose_omega_norm"] = np.linalg.norm(
            compare_base[["pose_omega_x", "pose_omega_y", "pose_omega_z"]].to_numpy(dtype=float),
            axis=1,
        )

    if len(gyro_df) and "time_s" in gyro_df.columns:
        gyro_nearest = nearest_rows(
            compare_base["raw_time_s"].to_numpy(dtype=float) if len(compare_base) else gyro_df["time_s"].to_numpy(dtype=float),
            gyro_df,
            ["x", "y", "z"],
            "raw_gyro",
        )
        if len(compare_base):
            compare_base = pd.concat([compare_base, gyro_nearest], axis=1)
            compare_base["raw_gyro_norm"] = np.linalg.norm(
                compare_base[["raw_gyro_x", "raw_gyro_y", "raw_gyro_z"]].to_numpy(dtype=float),
                axis=1,
            )

    raw_pose_compare_csv = metadata_dir / "raw_vs_onboard_pose_imu_comparison.csv"
    compare_base.to_csv(raw_pose_compare_csv, index=False)

    # Compare barometer altitude against nearest OnboardPose altitude.
    baro_pose_df = pd.DataFrame()
    if len(baro_df) and len(pose_df):
        baro_pose_df["baro_time_s"] = baro_df["time_s"].to_numpy(dtype=float)
        baro_pose_df["baro_altitude"] = baro_df["altitude"].to_numpy(dtype=float) if "altitude" in baro_df else np.nan
        nearest_pose_for_baro = nearest_rows(
            baro_pose_df["baro_time_s"].to_numpy(dtype=float),
            pose_df,
            ["altitude", "height"],
            "pose",
        )
        baro_pose_df = pd.concat([baro_pose_df, nearest_pose_for_baro], axis=1)
        baro_pose_df = baro_pose_df.rename(
            columns={
                "pose_altitude": "pose_altitude",
                "pose_height": "pose_height",
            }
        )

    baro_pose_csv = metadata_dir / "barometer_vs_onboard_pose_altitude.csv"
    baro_pose_df.to_csv(baro_pose_csv, index=False)

    # Correlations / relation checks.
    relation_summary: Dict[str, Any] = {}

    if len(compare_base):
        relation_summary["raw_accel_vs_pose_accel_corr"] = {
            "x": correlation(compare_base["raw_accel_x"].to_numpy(dtype=float), compare_base["pose_accel_x"].to_numpy(dtype=float))
            if "pose_accel_x" in compare_base else None,
            "y": correlation(compare_base["raw_accel_y"].to_numpy(dtype=float), compare_base["pose_accel_y"].to_numpy(dtype=float))
            if "pose_accel_y" in compare_base else None,
            "z": correlation(compare_base["raw_accel_z"].to_numpy(dtype=float), compare_base["pose_accel_z"].to_numpy(dtype=float))
            if "pose_accel_z" in compare_base else None,
            "norm": correlation(compare_base["raw_accel_norm"].to_numpy(dtype=float), compare_base["pose_accel_norm"].to_numpy(dtype=float))
            if "pose_accel_norm" in compare_base else None,
        }

        relation_summary["raw_gyro_vs_pose_omega_corr"] = {
            "x": correlation(compare_base["raw_gyro_x"].to_numpy(dtype=float), compare_base["pose_omega_x"].to_numpy(dtype=float))
            if "raw_gyro_x" in compare_base and "pose_omega_x" in compare_base else None,
            "y": correlation(compare_base["raw_gyro_y"].to_numpy(dtype=float), compare_base["pose_omega_y"].to_numpy(dtype=float))
            if "raw_gyro_y" in compare_base and "pose_omega_y" in compare_base else None,
            "z": correlation(compare_base["raw_gyro_z"].to_numpy(dtype=float), compare_base["pose_omega_z"].to_numpy(dtype=float))
            if "raw_gyro_z" in compare_base and "pose_omega_z" in compare_base else None,
            "norm": correlation(compare_base["raw_gyro_norm"].to_numpy(dtype=float), compare_base["pose_omega_norm"].to_numpy(dtype=float))
            if "raw_gyro_norm" in compare_base and "pose_omega_norm" in compare_base else None,
        }

    if len(baro_pose_df) and "pose_altitude" in baro_pose_df.columns:
        relation_summary["barometer_altitude_vs_pose_altitude_corr"] = correlation(
            baro_pose_df["baro_altitude"].to_numpy(dtype=float),
            baro_pose_df["pose_altitude"].to_numpy(dtype=float),
        )

    onboard_pose_checks = {
        "height": series_stats(pose_df, "height"),
        "altitude": series_stats(pose_df, "altitude"),
        "azimuth": series_stats(pose_df, "azimuth"),
        "veh_pitch": series_stats(pose_df, "veh_pitch"),
        "attitude_quaternion_norm": stats(pose_quat_norm),
    }

    accel_summary = stats(accel_norm)
    gyro_summary = stats(gyro_norm)

    decision_notes: List[str] = []

    accel_guess = accel_unit_guess(accel_summary.get("median"))
    gyro_guess = gyro_unit_guess(gyro_summary.get("median"))

    if accel_guess == "likely_m_per_s2":
        decision_notes.append("Raw accelerometer norm is consistent with m/s^2 gravity-scale values.")
    elif accel_guess == "likely_g_units":
        decision_notes.append("Raw accelerometer norm appears closer to g-units; conversion may be required before EKF.")
    else:
        decision_notes.append("Raw accelerometer unit is unclear; inspect generated plot and CSV before fusion.")

    if gyro_guess in ["small_motion_or_rad_per_s_possible", "rad_per_s_possible_but_not_confirmed"]:
        decision_notes.append("Raw gyroscope magnitude may be usable, but axis/sign convention still needs validation.")
    else:
        decision_notes.append("Raw gyroscope unit is unclear or may be degree/raw-scaled; conversion may be required.")

    if onboard_pose_checks["height"].get("is_constant_or_nearly_constant"):
        decision_notes.append("OnboardPose height is constant/nearly constant, so do not use it as true AGL.")
    if onboard_pose_checks["azimuth"].get("is_constant_or_nearly_constant"):
        decision_notes.append("OnboardPose azimuth is constant/nearly constant, so do not use it as reliable heading.")

    if onboard_pose_checks["attitude_quaternion_norm"].get("median") is not None:
        q_med = onboard_pose_checks["attitude_quaternion_norm"]["median"]
        if 0.95 <= q_med <= 1.05:
            decision_notes.append("OnboardPose quaternion norm is close to 1, so attitude quaternion may be structurally valid.")
        else:
            decision_notes.append("OnboardPose quaternion norm is not close to 1; verify attitude fields before use.")

    # Plots.
    plot_axes_and_norm(
        accel_df,
        ["x", "y", "z"],
        "Raw accelerometer axes and norm",
        fig_dir / "accel_axes_and_norm.png",
        "accel_norm",
    )
    plot_axes_and_norm(
        gyro_df,
        ["x", "y", "z"],
        "Raw gyroscope axes and norm",
        fig_dir / "gyro_axes_and_norm.png",
        "gyro_norm",
    )
    plot_onboard_pose_checks(pose_df, fig_dir / "onboard_pose_signal_checks.png")
    plot_barometer_signal(baro_df, fig_dir / "barometer_altitude_signal.png")
    plot_baro_vs_pose_altitude(baro_pose_df, fig_dir / "barometer_vs_onboard_pose_altitude.png")
    plot_raw_vs_pose_norms(compare_base, fig_dir / "raw_vs_onboard_pose_imu_norms.png")

    summary = {
        "block": "10A_imu_baro_signal_diagnostics",
        "dataset_name": dataset_name,
        "config_path": str(config_path),
        "raw_dir": str(raw_dir),
        "output_dir": str(output_dir),
        "sync_file": str(sync_path),
        "camera_timestamp_unit": cam_unit,
        "generated_outputs": {
            "imu_baro_signal_stats_csv": str(signal_stats_csv),
            "camera_nearest_sensor_values_csv": str(camera_nearest_csv),
            "raw_vs_onboard_pose_imu_comparison_csv": str(raw_pose_compare_csv),
            "barometer_vs_onboard_pose_altitude_csv": str(baro_pose_csv),
            "summary_json": str(report_dir / "imu_baro_signal_summary.json"),
            "accel_axes_and_norm_png": str(fig_dir / "accel_axes_and_norm.png"),
            "gyro_axes_and_norm_png": str(fig_dir / "gyro_axes_and_norm.png"),
            "onboard_pose_signal_checks_png": str(fig_dir / "onboard_pose_signal_checks.png"),
            "barometer_altitude_signal_png": str(fig_dir / "barometer_altitude_signal.png"),
            "barometer_vs_onboard_pose_altitude_png": str(fig_dir / "barometer_vs_onboard_pose_altitude.png"),
            "raw_vs_onboard_pose_imu_norms_png": str(fig_dir / "raw_vs_onboard_pose_imu_norms.png"),
        },
        "signal_rows": signal_rows,
        "accelerometer_norm_summary": {
            **accel_summary,
            "unit_guess": accel_guess,
        },
        "gyroscope_norm_summary": {
            **gyro_summary,
            "unit_guess": gyro_guess,
        },
        "pose_accel_norm_summary": stats(pose_accel_norm),
        "pose_omega_norm_summary": stats(pose_omega_norm),
        "onboard_pose_checks": onboard_pose_checks,
        "relation_summary": relation_summary,
        "decision_notes": decision_notes,
        "important_limits": [
            "This diagnostic does not prove camera-to-IMU extrinsics.",
            "This diagnostic does not estimate IMU noise density or random walk.",
            "This diagnostic does not validate IMU axis convention against camera/body frame.",
            "GNSS/reference remains evaluation-only and is not used as estimator input.",
        ],
    }

    summary_json = report_dir / "imu_baro_signal_summary.json"
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(json_safe(summary), f, indent=2)

    print("Block 10A IMU / barometer signal diagnostics generated")
    print("------------------------------------------------------")
    print(f"Dataset:        {dataset_name}")
    print(f"Raw dir:        {raw_dir}")
    print(f"Output dir:     {output_dir}")
    print("")
    print("Generated:")
    print(f"- {signal_stats_csv}")
    print(f"- {camera_nearest_csv}")
    print(f"- {raw_pose_compare_csv}")
    print(f"- {baro_pose_csv}")
    print(f"- {summary_json}")
    print(f"- {fig_dir / 'accel_axes_and_norm.png'}")
    print(f"- {fig_dir / 'gyro_axes_and_norm.png'}")
    print(f"- {fig_dir / 'onboard_pose_signal_checks.png'}")
    print(f"- {fig_dir / 'barometer_altitude_signal.png'}")
    print(f"- {fig_dir / 'barometer_vs_onboard_pose_altitude.png'}")
    print(f"- {fig_dir / 'raw_vs_onboard_pose_imu_norms.png'}")
    print("")
    print("Key signal checks:")
    print(f"- accelerometer norm median: {accel_summary.get('median')} | unit guess: {accel_guess}")
    print(f"- gyroscope norm median:     {gyro_summary.get('median')} | unit guess: {gyro_guess}")
    print(f"- pose quaternion norm median: {onboard_pose_checks['attitude_quaternion_norm'].get('median')}")
    print(f"- OnboardPose height constant: {onboard_pose_checks['height'].get('is_constant_or_nearly_constant')}")
    print(f"- OnboardPose azimuth constant: {onboard_pose_checks['azimuth'].get('is_constant_or_nearly_constant')}")
    print("")
    print("Relations:")
    for key, value in relation_summary.items():
        print(f"- {key}: {value}")
    print("")
    print("Decision notes:")
    for note in decision_notes:
        print(f"- {note}")


if __name__ == "__main__":
    main()
