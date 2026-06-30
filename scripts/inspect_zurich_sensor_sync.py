#!/usr/bin/env python3
"""
10A — Zurich IMU / Barometer / Camera synchronization diagnostics.

This script NOT perform sensor fusion.
It checks whether camera frames, accelerometer, gyroscope, barometer,
onboard pose, and GPS/reference timestamps are usable for future EKF/ESKF/VIO work.

Outputs:
- sensor_stream_stats.csv
- camera_sensor_nearest_timestamps.csv
- sensor_sync_summary.json
- sensor_timestamp_ranges.png
- imu_sampling_intervals.png
- camera_sensor_time_offsets.png
- imu_magnitude_checks.png
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
        "filenames": [
            "RawAccel.csv",
            "RawAccelerometer.csv",
            "RawAccelerometerData.csv",
        ],
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
        "filenames": [
            "RawGyro.csv",
            "RawGyroscope.csv",
            "RawGyroscopeData.csv",
        ],
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
        "filenames": [
            "BarometricPressure.csv",
            "Barometer.csv",
        ],
        "columns": [
            "timestamp",
            "pressure",
            "altitude",
            "temperature",
        ],
    },
    "onboard_pose": {
        "filenames": [
            "OnboardPose.csv",
        ],
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
    "onboard_gps": {
        "filenames": [
            "OnboardGPS.csv",
            "OnbordGPS.csv",
        ],
        "columns": [
            "timestamp",
            "imgid",
            "lat_raw",
            "lon_raw",
            "alt_raw",
            "s_variance_m_s",
            "c_variance_rad",
            "fix_type",
            "eph_m",
            "epv_m",
            "vel_n_m_s",
            "vel_e_m_s",
            "vel_d_m_s",
            "num_sat",
        ],
    },
    "groundtruth_agl": {
        "filenames": [
            "GroundTruthAGL.csv",
        ],
        "columns": [
            "imgid",
            "x_gt",
            "y_gt",
            "z_gt",
            "omega_yaw_deg",
            "phi_pitch_deg",
            "kappa_roll_deg",
            "x_gps",
            "y_gps",
            "z_gps",
        ],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        required=True,
        help="Dataset config path, e.g. configs/dataset_zurich_full.yaml",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f)
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Config is not a dictionary: {path}")
    return loaded


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
    if p.is_absolute():
        return p
    return project_root / p


def infer_dataset_name(cfg: Dict[str, Any], config_path: Path) -> str:
    for key in ["dataset_name", "name"]:
        found = deep_find_first(cfg, [key])
        if found is not None and str(found).strip():
            return str(found)
    return config_path.stem.replace("dataset_", "")


def infer_raw_and_output_dirs(
    cfg: Dict[str, Any],
    config_path: Path,
    project_root: Path,
) -> Tuple[Path, Path, str]:
    dataset_name = infer_dataset_name(cfg, config_path)

    raw_value = deep_find_first(
        cfg,
        [
            "raw_data_dir",
            "raw_dir",
            "data_raw_dir",
            "dataset_dir",
            "raw_dataset_dir",
        ],
    )
    output_value = deep_find_first(
        cfg,
        [
            "output_dir",
            "outputs_dir",
            "output_root",
        ],
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
        candidate = base_dir / name
        if candidate.exists():
            return candidate

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
    elif "imgid" in df.columns:
        df = df[df["imgid"].notna()].copy()

    return df.reset_index(drop=True)


def read_sync_csv(output_dir: Path) -> Tuple[Path, pd.DataFrame]:
    candidates = [
        output_dir / "metadata" / "synchronized_frames_enriched.csv",
        output_dir / "metadata" / "synchronized_frames.csv",
    ]

    for path in candidates:
        if path.exists():
            return path, pd.read_csv(path)

    raise FileNotFoundError(
        "Could not find synchronized frames CSV. Expected one of:\n"
        + "\n".join(str(p) for p in candidates)
    )


def numeric_values(series: pd.Series) -> np.ndarray:
    vals = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    return vals[np.isfinite(vals)]


def choose_time_column(df: pd.DataFrame, stream_name: str) -> Tuple[Optional[str], str]:
    preferred = [
        "timestamp",
        "Timestamp",
        "time",
        "frame_timestamp",
        "camera_timestamp",
        "image_timestamp",
        "gps_timestamp",
    ]

    for col in preferred:
        if col in df.columns and numeric_values(df[col]).size > 0:
            return col, "timestamp"

    for col in df.columns:
        if "timestamp" in str(col).lower() and numeric_values(df[col]).size > 0:
            return col, "timestamp"

    if "imgid" in df.columns and numeric_values(df["imgid"]).size > 0:
        return "imgid", "imgid_fallback_not_true_time"

    return None, "missing"


def infer_timestamp_scale(ts_raw: np.ndarray) -> Tuple[float, str]:
    if ts_raw.size < 2:
        return 1.0, "unknown_single_value"

    finite = ts_raw[np.isfinite(ts_raw)]
    if finite.size < 2:
        return 1.0, "unknown_no_finite_values"

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

    return 1.0, "seconds_or_already_normalized"


def safe_percentile(vals: np.ndarray, q: float) -> Optional[float]:
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return None
    return float(np.percentile(vals, q))


def summarize_array(vals: np.ndarray, prefix: str = "") -> Dict[str, Optional[float]]:
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {
            f"{prefix}min": None,
            f"{prefix}p05": None,
            f"{prefix}median": None,
            f"{prefix}mean": None,
            f"{prefix}p95": None,
            f"{prefix}max": None,
        }

    return {
        f"{prefix}min": float(np.min(vals)),
        f"{prefix}p05": safe_percentile(vals, 5),
        f"{prefix}median": float(np.median(vals)),
        f"{prefix}mean": float(np.mean(vals)),
        f"{prefix}p95": safe_percentile(vals, 95),
        f"{prefix}max": float(np.max(vals)),
    }


def stream_stats(
    name: str,
    path: Optional[Path],
    df: pd.DataFrame,
    time_col: Optional[str],
    time_col_status: str,
) -> Tuple[Dict[str, Any], Optional[np.ndarray]]:
    row: Dict[str, Any] = {
        "stream": name,
        "path": str(path) if path is not None else None,
        "rows": int(len(df)),
        "time_column": time_col,
        "time_column_status": time_col_status,
        "timestamp_unit_assumption": None,
        "timestamp_scale_to_seconds": None,
        "timestamp_min_raw": None,
        "timestamp_max_raw": None,
        "time_min_s": None,
        "time_max_s": None,
        "duration_s": None,
        "is_monotonic_in_file_order": None,
        "duplicate_timestamp_count": None,
        "nonpositive_dt_count": None,
        "median_dt_s": None,
        "mean_dt_s": None,
        "min_dt_s": None,
        "p95_dt_s": None,
        "max_dt_s": None,
        "approx_rate_hz_from_median_dt": None,
    }

    if time_col is None or time_col not in df.columns:
        return row, None

    ts_raw = numeric_values(df[time_col])
    if ts_raw.size == 0:
        return row, None

    scale, unit_label = infer_timestamp_scale(ts_raw)
    time_s = ts_raw * scale

    dt_s = np.diff(time_s)
    pos_dt_s = dt_s[dt_s > 0]

    row["timestamp_unit_assumption"] = unit_label
    row["timestamp_scale_to_seconds"] = scale
    row["timestamp_min_raw"] = float(np.min(ts_raw))
    row["timestamp_max_raw"] = float(np.max(ts_raw))
    row["time_min_s"] = float(np.min(time_s))
    row["time_max_s"] = float(np.max(time_s))
    row["duration_s"] = float(np.max(time_s) - np.min(time_s))
    row["is_monotonic_in_file_order"] = bool(np.all(dt_s >= 0)) if dt_s.size else True
    row["duplicate_timestamp_count"] = int(pd.Series(ts_raw).duplicated().sum())
    row["nonpositive_dt_count"] = int(np.sum(dt_s <= 0)) if dt_s.size else 0

    if pos_dt_s.size:
        row["median_dt_s"] = float(np.median(pos_dt_s))
        row["mean_dt_s"] = float(np.mean(pos_dt_s))
        row["min_dt_s"] = float(np.min(pos_dt_s))
        row["p95_dt_s"] = safe_percentile(pos_dt_s, 95)
        row["max_dt_s"] = float(np.max(pos_dt_s))
        row["approx_rate_hz_from_median_dt"] = (
            float(1.0 / np.median(pos_dt_s)) if np.median(pos_dt_s) > 0 else None
        )

    return row, time_s


def nearest_signed_delta(query_s: np.ndarray, reference_s: np.ndarray) -> np.ndarray:
    reference_s = reference_s[np.isfinite(reference_s)]
    reference_s = np.sort(reference_s)

    out = np.full(query_s.shape, np.nan, dtype=float)

    if reference_s.size == 0:
        return out

    valid = np.isfinite(query_s)
    q = query_s[valid]

    idx = np.searchsorted(reference_s, q)

    left_idx = np.clip(idx - 1, 0, reference_s.size - 1)
    right_idx = np.clip(idx, 0, reference_s.size - 1)

    left_delta = reference_s[left_idx] - q
    right_delta = reference_s[right_idx] - q

    choose_right = np.abs(right_delta) < np.abs(left_delta)
    best = np.where(choose_right, right_delta, left_delta)

    out[valid] = best
    return out


def offset_quality(stream_name: str, p95_abs_ms: Optional[float]) -> str:
    if p95_abs_ms is None:
        return "missing"

    thresholds_ms = {
        "accelerometer": 20.0,
        "gyroscope": 20.0,
        "barometer": 200.0,
        "onboard_pose": 50.0,
        "onboard_gps": 200.0,
        "groundtruth_agl": 500.0,
    }

    threshold = thresholds_ms.get(stream_name, 100.0)

    if p95_abs_ms <= threshold:
        return "good"
    if p95_abs_ms <= threshold * 10:
        return "check"
    return "poor_or_timestamp_mismatch"


def vector_magnitude(df: pd.DataFrame, cols: List[str]) -> Optional[np.ndarray]:
    if not all(col in df.columns for col in cols):
        return None

    arr = df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    arr = arr[np.all(np.isfinite(arr), axis=1)]

    if arr.size == 0:
        return None

    return np.linalg.norm(arr, axis=1)


def accel_unit_guess(median_norm: Optional[float]) -> str:
    if median_norm is None:
        return "missing"
    if 7.0 <= median_norm <= 12.5:
        return "likely_m_per_s2"
    if 0.7 <= median_norm <= 1.3:
        return "likely_g_units"
    return "unclear"


def gyro_unit_guess(median_norm: Optional[float]) -> str:
    if median_norm is None:
        return "missing"
    if median_norm < 0.5:
        return "small_motion_or_rad_per_s_possible"
    if median_norm < 10.0:
        return "rad_per_s_possible_but_not_confirmed"
    return "deg_per_s_or_raw_units_possible"


def plot_timestamp_ranges(stats_rows: List[Dict[str, Any]], fig_dir: Path) -> None:
    rows = [
        row
        for row in stats_rows
        if row.get("time_min_s") is not None and row.get("time_max_s") is not None
    ]

    if not rows:
        return

    global_start = min(float(row["time_min_s"]) for row in rows)

    names = [str(row["stream"]) for row in rows]
    starts = [float(row["time_min_s"]) - global_start for row in rows]
    widths = [float(row["time_max_s"]) - float(row["time_min_s"]) for row in rows]

    plt.figure(figsize=(11, max(4, 0.55 * len(rows) + 1.5)))
    plt.barh(names, widths, left=starts)
    plt.xlabel("Seconds from earliest available timestamp")
    plt.ylabel("Sensor stream")
    plt.title("Sensor timestamp coverage")
    plt.tight_layout()
    plt.savefig(fig_dir / "sensor_timestamp_ranges.png", dpi=160)
    plt.close()


def plot_sampling_intervals(streams: Dict[str, Dict[str, Any]], fig_dir: Path) -> None:
    plt.figure(figsize=(10, 5))

    plotted = False
    for name in ["accelerometer", "gyroscope", "barometer", "onboard_pose"]:
        time_s = streams.get(name, {}).get("time_s")
        if time_s is None:
            continue

        dt_ms = np.diff(time_s)
        dt_ms = dt_ms[np.isfinite(dt_ms) & (dt_ms > 0)] * 1000.0

        if dt_ms.size == 0:
            continue

        cutoff = np.percentile(dt_ms, 99)
        dt_ms_plot = dt_ms[dt_ms <= cutoff]
        plt.hist(dt_ms_plot, bins=80, alpha=0.45, label=name)
        plotted = True

    if plotted:
        plt.xlabel("Sampling interval [ms]")
        plt.ylabel("Count")
        plt.title("Sampling interval distribution")
        plt.legend()
    else:
        plt.text(0.5, 0.5, "No sampling interval data available", ha="center", va="center")
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(fig_dir / "imu_sampling_intervals.png", dpi=160)
    plt.close()


def plot_camera_sensor_offsets(offset_df: pd.DataFrame, fig_dir: Path) -> None:
    cols = [col for col in offset_df.columns if col.endswith("_abs_dt_ms")]

    plt.figure(figsize=(10, 5))

    if cols:
        data = [
            offset_df[col].dropna().to_numpy(dtype=float)
            for col in cols
            if offset_df[col].dropna().size > 0
        ]
        labels = [
            col.replace("_abs_dt_ms", "")
            for col in cols
            if offset_df[col].dropna().size > 0
        ]

        if data:
            plt.boxplot(data, labels=labels, showfliers=False)
            plt.ylabel("Absolute nearest timestamp offset [ms]")
            plt.title("Camera-to-sensor nearest timestamp offsets")
            plt.xticks(rotation=25, ha="right")
        else:
            plt.text(0.5, 0.5, "No offset data available", ha="center", va="center")
            plt.axis("off")
    else:
        plt.text(0.5, 0.5, "No offset columns available", ha="center", va="center")
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(fig_dir / "camera_sensor_time_offsets.png", dpi=160)
    plt.close()


def plot_imu_magnitudes(
    accel_mag: Optional[np.ndarray],
    gyro_mag: Optional[np.ndarray],
    fig_dir: Path,
) -> None:
    plt.figure(figsize=(10, 5))

    plotted = False

    if accel_mag is not None and accel_mag.size:
        idx = np.linspace(0, accel_mag.size - 1, min(accel_mag.size, 5000)).astype(int)
        plt.plot(idx, accel_mag[idx], label="accelerometer norm")
        plotted = True

    if gyro_mag is not None and gyro_mag.size:
        idx = np.linspace(0, gyro_mag.size - 1, min(gyro_mag.size, 5000)).astype(int)
        plt.plot(idx, gyro_mag[idx], label="gyroscope norm")
        plotted = True

    if plotted:
        plt.xlabel("Sample index, downsampled for display")
        plt.ylabel("Vector magnitude")
        plt.title("IMU magnitude sanity checks")
        plt.legend()
    else:
        plt.text(0.5, 0.5, "No IMU magnitude data available", ha="center", va="center")
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(fig_dir / "imu_magnitude_checks.png", dpi=160)
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
        if not np.isfinite(value):
            return None
        return value
    if isinstance(obj, float):
        if not np.isfinite(obj):
            return None
        return obj
    return obj


def main() -> None:
    args = parse_args()

    project_root = Path.cwd()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = project_root / config_path

    cfg = load_yaml(config_path)
    raw_dir, output_dir, dataset_name = infer_raw_and_output_dirs(
        cfg=cfg,
        config_path=config_path,
        project_root=project_root,
    )

    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw data directory not found: {raw_dir}")

    sync_path, sync_df = read_sync_csv(output_dir)

    metadata_dir = output_dir / "metadata" / "10a_sensor_sync_diagnostics"
    report_dir = output_dir / "reports" / "10a_sensor_sync_diagnostics"
    fig_dir = output_dir / "figures" / "10a_sensor_sync_diagnostics"

    metadata_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    streams: Dict[str, Dict[str, Any]] = {}

    camera_time_col, camera_time_status = choose_time_column(sync_df, "camera")
    camera_stats, camera_time_s = stream_stats(
        name="camera_sync",
        path=sync_path,
        df=sync_df,
        time_col=camera_time_col,
        time_col_status=camera_time_status,
    )

    streams["camera_sync"] = {
        "path": sync_path,
        "df": sync_df,
        "time_col": camera_time_col,
        "time_col_status": camera_time_status,
        "stats": camera_stats,
        "time_s": camera_time_s,
    }

    for stream_name, spec in LOG_SPECS.items():
        path = find_first_existing(raw_dir, spec["filenames"])

        if path is None:
            empty_df = pd.DataFrame()
            stats, time_s = stream_stats(
                name=stream_name,
                path=None,
                df=empty_df,
                time_col=None,
                time_col_status="file_missing",
            )
            streams[stream_name] = {
                "path": None,
                "df": empty_df,
                "time_col": None,
                "time_col_status": "file_missing",
                "stats": stats,
                "time_s": time_s,
            }
            continue

        df = read_expected_csv(path, spec["columns"])
        time_col, time_status = choose_time_column(df, stream_name)
        stats, time_s = stream_stats(
            name=stream_name,
            path=path,
            df=df,
            time_col=time_col,
            time_col_status=time_status,
        )

        streams[stream_name] = {
            "path": path,
            "df": df,
            "time_col": time_col,
            "time_col_status": time_status,
            "stats": stats,
            "time_s": time_s,
        }

    stats_rows = [streams[name]["stats"] for name in streams]
    stats_df = pd.DataFrame(stats_rows)
    stats_csv = metadata_dir / "sensor_stream_stats.csv"
    stats_df.to_csv(stats_csv, index=False)

    camera_time_s = streams["camera_sync"]["time_s"]
    if camera_time_s is None:
        raise RuntimeError(
            "Camera synchronized frame file has no usable timestamp or imgid column."
        )

    offset_df = pd.DataFrame()
    offset_df["camera_row"] = np.arange(len(camera_time_s))
    offset_df["camera_time_s"] = camera_time_s

    if "imgid" in sync_df.columns:
        offset_df["imgid"] = pd.to_numeric(sync_df["imgid"], errors="coerce").to_numpy()

    offset_summary: Dict[str, Any] = {}

    for sensor_name in [
        "accelerometer",
        "gyroscope",
        "barometer",
        "onboard_pose",
        "onboard_gps",
        "groundtruth_agl",
    ]:
        sensor_time_s = streams.get(sensor_name, {}).get("time_s")
        if sensor_time_s is None:
            offset_summary[sensor_name] = {
                "available": False,
                "quality": "missing",
            }
            continue

        dt_s = nearest_signed_delta(camera_time_s, sensor_time_s)
        abs_dt_ms = np.abs(dt_s) * 1000.0

        offset_df[f"{sensor_name}_dt_s"] = dt_s
        offset_df[f"{sensor_name}_abs_dt_ms"] = abs_dt_ms

        sensor_stats = summarize_array(abs_dt_ms, prefix="abs_offset_ms_")
        quality = offset_quality(sensor_name, sensor_stats.get("abs_offset_ms_p95"))

        overlap_start = max(
            float(np.nanmin(camera_time_s)),
            float(np.nanmin(sensor_time_s)),
        )
        overlap_end = min(
            float(np.nanmax(camera_time_s)),
            float(np.nanmax(sensor_time_s)),
        )
        overlap_s = max(0.0, overlap_end - overlap_start)

        offset_summary[sensor_name] = {
            "available": True,
            "quality": quality,
            "overlap_s": overlap_s,
            **sensor_stats,
        }

    offsets_csv = metadata_dir / "camera_sensor_nearest_timestamps.csv"
    offset_df.to_csv(offsets_csv, index=False)

    accel_mag = vector_magnitude(streams["accelerometer"]["df"], ["x", "y", "z"])
    gyro_mag = vector_magnitude(streams["gyroscope"]["df"], ["x", "y", "z"])

    accel_mag_summary = summarize_array(accel_mag if accel_mag is not None else np.array([]))
    gyro_mag_summary = summarize_array(gyro_mag if gyro_mag is not None else np.array([]))

    imu_summary = {
        "accelerometer_norm": {
            **accel_mag_summary,
            "unit_guess": accel_unit_guess(accel_mag_summary.get("median")),
        },
        "gyroscope_norm": {
            **gyro_mag_summary,
            "unit_guess": gyro_unit_guess(gyro_mag_summary.get("median")),
        },
    }

    camera_imu_ready = (
        offset_summary.get("accelerometer", {}).get("quality") in ["good", "check"]
        and offset_summary.get("gyroscope", {}).get("quality") in ["good", "check"]
        and streams["accelerometer"]["stats"].get("rows", 0) > 0
        and streams["gyroscope"]["stats"].get("rows", 0) > 0
    )

    if camera_imu_ready:
        preliminary_decision = (
            "camera_imu_timestamps_exist_and_are_potentially_usable_for_next_diagnostics"
        )
    else:
        preliminary_decision = (
            "camera_imu_fusion_not_ready_until_timestamp_or_file_issues_are_resolved"
        )

    plot_timestamp_ranges(stats_rows, fig_dir)
    plot_sampling_intervals(streams, fig_dir)
    plot_camera_sensor_offsets(offset_df, fig_dir)
    plot_imu_magnitudes(accel_mag, gyro_mag, fig_dir)

    summary = {
        "block": "10A_sensor_sync_diagnostics",
        "dataset_name": dataset_name,
        "config_path": str(config_path),
        "raw_dir": str(raw_dir),
        "output_dir": str(output_dir),
        "sync_file": str(sync_path),
        "generated_outputs": {
            "sensor_stream_stats_csv": str(stats_csv),
            "camera_sensor_nearest_timestamps_csv": str(offsets_csv),
            "summary_json": str(report_dir / "sensor_sync_summary.json"),
            "sensor_timestamp_ranges_png": str(fig_dir / "sensor_timestamp_ranges.png"),
            "imu_sampling_intervals_png": str(fig_dir / "imu_sampling_intervals.png"),
            "camera_sensor_time_offsets_png": str(fig_dir / "camera_sensor_time_offsets.png"),
            "imu_magnitude_checks_png": str(fig_dir / "imu_magnitude_checks.png"),
        },
        "stream_stats": stats_rows,
        "camera_sensor_offset_summary": offset_summary,
        "imu_magnitude_summary": imu_summary,
        "preliminary_decision": preliminary_decision,
        "notes": [
            "This is a diagnostic block only. No GNSS/reference is used for estimation.",
            "Timestamp unit assumptions are inferred and must be checked from terminal/report output.",
            "Good timestamp matching does not prove VIO readiness by itself.",
            "VIO still requires camera intrinsics, distortion, camera-to-IMU extrinsics, IMU noise parameters, and axis convention validation.",
        ],
    }

    summary_json = report_dir / "sensor_sync_summary.json"
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(json_safe(summary), f, indent=2)

    print("Block 10A sensor synchronization diagnostics generated")
    print("------------------------------------------------------")
    print(f"Dataset:        {dataset_name}")
    print(f"Raw dir:        {raw_dir}")
    print(f"Output dir:     {output_dir}")
    print(f"Sync file:      {sync_path}")
    print("")
    print("Generated:")
    print(f"- {stats_csv}")
    print(f"- {offsets_csv}")
    print(f"- {summary_json}")
    print(f"- {fig_dir / 'sensor_timestamp_ranges.png'}")
    print(f"- {fig_dir / 'imu_sampling_intervals.png'}")
    print(f"- {fig_dir / 'camera_sensor_time_offsets.png'}")
    print(f"- {fig_dir / 'imu_magnitude_checks.png'}")
    print("")
    print("Stream summary:")
    for row in stats_rows:
        print(
            f"- {row['stream']}: rows={row['rows']}, "
            f"time_col={row['time_column']}, "
            f"unit={row['timestamp_unit_assumption']}, "
            f"median_dt_s={row['median_dt_s']}, "
            f"rate_hz={row['approx_rate_hz_from_median_dt']}"
        )
    print("")
    print("Camera-to-sensor offset quality:")
    for name, item in offset_summary.items():
        print(
            f"- {name}: available={item.get('available')}, "
            f"quality={item.get('quality')}, "
            f"p95_abs_ms={item.get('abs_offset_ms_p95')}"
        )
    print("")
    print(f"Preliminary decision: {preliminary_decision}")


if __name__ == "__main__":
    main()
