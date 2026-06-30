#!/usr/bin/env python3
"""
Block 10A.5 / Week 4A.1 — Zurich attitude and gyro axis mapping diagnostics.

It checks:
- whether OnboardPose quaternion gives meaningful roll/pitch/yaw,
- whether yaw-rate from quaternion matches pose omega axes,
- whether yaw-rate from quaternion matches raw gyro axes,
- which axis/sign mapping is likely.

Outputs:
- attitude_by_pose_timestamp.csv
- yaw_rate_gyro_alignment.csv
- attitude_axis_summary.json
- attitude_rpy_over_time.png
- yaw_rate_vs_best_gyro.png
- gyro_axis_correlation_bars.png
- quaternion_norm.png
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


ONBOARD_POSE_COLUMNS = [
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
]

RAW_GYRO_COLUMNS = [
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
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--smooth-window", type=int, default=21)
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


def find_first_existing(base_dir: Path, names: List[str]) -> Optional[Path]:
    for name in names:
        p = base_dir / name
        if p.exists():
            return p

    for name in names:
        matches = list(base_dir.rglob(name))
        if matches:
            return matches[0]

    return None


def read_csv_expected(path: Path, columns: List[str]) -> pd.DataFrame:
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


def add_time_s(df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    out = df.copy()
    ts = pd.to_numeric(out["timestamp"], errors="coerce").to_numpy(dtype=float)
    scale, label = infer_timestamp_scale(ts)
    out["time_s"] = ts * scale
    return out, label


def quaternion_to_euler_wxyz(
    w: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert quaternion w,x,y,z to roll,pitch,yaw radians.
    Convention: standard aerospace-like intrinsic XYZ output.
    This is diagnostic only until the dataset convention is confirmed.
    """

    norm = np.sqrt(w * w + x * x + y * y + z * z)
    norm = np.where(norm == 0, np.nan, norm)

    w = w / norm
    x = x / norm
    y = y / norm
    z = z / norm

    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    sinp = np.clip(sinp, -1.0, 1.0)
    pitch = np.arcsin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


def rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return arr

    s = pd.Series(arr)
    return (
        s.rolling(window=window, center=True, min_periods=max(3, window // 3))
        .mean()
        .bfill()
        .ffill()
        .to_numpy(dtype=float)
    )


def finite_corr(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    mask = np.isfinite(a) & np.isfinite(b)
    a = a[mask]
    b = b[mask]

    if a.size < 10:
        return None
    if np.std(a) == 0 or np.std(b) == 0:
        return None

    return float(np.corrcoef(a, b)[0, 1])


def nearest_values(query_t: np.ndarray, ref_t: np.ndarray, ref_vals: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    query_t = np.asarray(query_t, dtype=float)
    ref_t = np.asarray(ref_t, dtype=float)
    ref_vals = np.asarray(ref_vals, dtype=float)

    valid = np.isfinite(ref_t) & np.isfinite(ref_vals)
    ref_t = ref_t[valid]
    ref_vals = ref_vals[valid]

    order = np.argsort(ref_t)
    ref_t = ref_t[order]
    ref_vals = ref_vals[order]

    out_vals = np.full(query_t.shape, np.nan, dtype=float)
    out_dt = np.full(query_t.shape, np.nan, dtype=float)

    if ref_t.size == 0:
        return out_vals, out_dt

    idx = np.searchsorted(ref_t, query_t)
    left = np.clip(idx - 1, 0, ref_t.size - 1)
    right = np.clip(idx, 0, ref_t.size - 1)

    left_dt = ref_t[left] - query_t
    right_dt = ref_t[right] - query_t

    choose_right = np.abs(right_dt) < np.abs(left_dt)
    best = np.where(choose_right, right, left)

    out_vals = ref_vals[best]
    out_dt = ref_t[best] - query_t

    return out_vals, out_dt


def stats(arr: np.ndarray) -> Dict[str, Optional[float]]:
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]

    if arr.size == 0:
        return {
            "count": 0,
            "min": None,
            "median": None,
            "mean": None,
            "max": None,
            "std": None,
        }

    return {
        "count": int(arr.size),
        "min": float(np.min(arr)),
        "median": float(np.median(arr)),
        "mean": float(np.mean(arr)),
        "max": float(np.max(arr)),
        "std": float(np.std(arr)),
    }


def angle_stats_deg(arr_rad: np.ndarray) -> Dict[str, Optional[float]]:
    s = stats(np.rad2deg(arr_rad))
    return {f"deg_{k}": v for k, v in s.items()}


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


def downsample(df: pd.DataFrame, max_points: int = 8000) -> pd.DataFrame:
    if len(df) <= max_points:
        return df
    idx = np.linspace(0, len(df) - 1, max_points).astype(int)
    return df.iloc[idx].copy()


def plot_attitude(att_df: pd.DataFrame, out_path: Path) -> None:
    d = downsample(att_df.dropna(subset=["time_s"]))

    plt.figure(figsize=(11, 5))
    if len(d) == 0:
        plt.text(0.5, 0.5, "No attitude data", ha="center", va="center")
        plt.axis("off")
    else:
        t = d["time_s"].to_numpy(dtype=float)
        t = t - np.nanmin(t)

        plt.plot(t, d["roll_deg"], label="roll_deg", linewidth=0.9)
        plt.plot(t, d["pitch_deg"], label="pitch_deg", linewidth=0.9)
        plt.plot(t, d["yaw_deg_unwrapped"], label="yaw_deg_unwrapped", linewidth=0.9)

        plt.xlabel("Time from OnboardPose start [s]")
        plt.ylabel("Angle [deg]")
        plt.title("OnboardPose quaternion converted to roll / pitch / yaw")
        plt.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_quaternion_norm(att_df: pd.DataFrame, out_path: Path) -> None:
    d = downsample(att_df.dropna(subset=["time_s"]))

    plt.figure(figsize=(10, 4))
    if len(d) == 0:
        plt.text(0.5, 0.5, "No quaternion norm data", ha="center", va="center")
        plt.axis("off")
    else:
        t = d["time_s"].to_numpy(dtype=float)
        t = t - np.nanmin(t)

        plt.plot(t, d["quat_norm"], linewidth=0.9)
        plt.xlabel("Time from OnboardPose start [s]")
        plt.ylabel("Quaternion norm")
        plt.title("OnboardPose quaternion norm")

    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_yaw_rate_best(att_df: pd.DataFrame, best_name: str, out_path: Path) -> None:
    d = downsample(att_df.dropna(subset=["time_s", "yaw_rate_smooth_rad_s", best_name]))

    plt.figure(figsize=(11, 5))
    if len(d) == 0:
        plt.text(0.5, 0.5, "No yaw-rate comparison data", ha="center", va="center")
        plt.axis("off")
    else:
        t = d["time_s"].to_numpy(dtype=float)
        t = t - np.nanmin(t)

        plt.plot(t, d["yaw_rate_smooth_rad_s"], label="quaternion yaw-rate", linewidth=0.9)
        plt.plot(t, d[best_name], label=best_name, linewidth=0.9)

        plt.xlabel("Time from OnboardPose start [s]")
        plt.ylabel("Angular rate [rad/s]")
        plt.title("Quaternion yaw-rate versus best gyro/omega candidate")
        plt.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_correlation_bars(score_df: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(10, 5))

    if len(score_df) == 0:
        plt.text(0.5, 0.5, "No correlation scores", ha="center", va="center")
        plt.axis("off")
    else:
        d = score_df.sort_values("abs_corr", ascending=False).head(12).copy()
        labels = d["candidate"].tolist()
        values = d["corr"].to_numpy(dtype=float)

        plt.bar(labels, values)
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Correlation with quaternion yaw-rate")
        plt.title("Gyro / omega axis candidates for yaw-rate")
        plt.ylim(-1.05, 1.05)

    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def main() -> None:
    args = parse_args()

    project_root = Path.cwd()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = project_root / config_path

    cfg = load_yaml(config_path)
    raw_dir, output_dir, dataset_name = infer_dirs(cfg, config_path, project_root)

    pose_path = find_first_existing(raw_dir, ["OnboardPose.csv"])
    gyro_path = find_first_existing(raw_dir, ["RawGyro.csv", "RawGyroscope.csv", "RawGyroscopeData.csv"])

    if pose_path is None:
        raise FileNotFoundError(f"Could not find OnboardPose.csv inside {raw_dir}")
    if gyro_path is None:
        raise FileNotFoundError(f"Could not find RawGyro/RawGyroscope CSV inside {raw_dir}")

    pose_df = read_csv_expected(pose_path, ONBOARD_POSE_COLUMNS)
    gyro_df = read_csv_expected(gyro_path, RAW_GYRO_COLUMNS)

    pose_df, pose_time_unit = add_time_s(pose_df)
    gyro_df, gyro_time_unit = add_time_s(gyro_df)

    metadata_dir = output_dir / "metadata" / "10a_attitude_axis_diagnostics"
    report_dir = output_dir / "reports" / "10a_attitude_axis_diagnostics"
    fig_dir = output_dir / "figures" / "10a_attitude_axis_diagnostics"

    metadata_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    q_cols = ["attitude_w", "attitude_x", "attitude_y", "attitude_z"]
    q = pose_df[q_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    quat_norm = np.sqrt(w * w + x * x + y * y + z * z)

    roll, pitch, yaw = quaternion_to_euler_wxyz(w, x, y, z)
    yaw_unwrapped = np.unwrap(yaw)

    time_s = pose_df["time_s"].to_numpy(dtype=float)

    smooth_window = max(1, int(args.smooth_window))
    if smooth_window % 2 == 0:
        smooth_window += 1

    yaw_unwrapped_smooth = rolling_mean(yaw_unwrapped, smooth_window)
    roll_smooth = rolling_mean(roll, smooth_window)
    pitch_smooth = rolling_mean(pitch, smooth_window)

    yaw_rate = np.gradient(yaw_unwrapped, time_s)
    yaw_rate_smooth = np.gradient(yaw_unwrapped_smooth, time_s)
    roll_rate_smooth = np.gradient(roll_smooth, time_s)
    pitch_rate_smooth = np.gradient(pitch_smooth, time_s)

    att_df = pd.DataFrame()
    att_df["timestamp"] = pose_df["timestamp"]
    att_df["time_s"] = time_s
    att_df["quat_norm"] = quat_norm

    att_df["roll_rad"] = roll
    att_df["pitch_rad"] = pitch
    att_df["yaw_rad"] = yaw
    att_df["yaw_rad_unwrapped"] = yaw_unwrapped
    att_df["roll_deg"] = np.rad2deg(roll)
    att_df["pitch_deg"] = np.rad2deg(pitch)
    att_df["yaw_deg"] = np.rad2deg(yaw)
    att_df["yaw_deg_unwrapped"] = np.rad2deg(yaw_unwrapped)

    att_df["yaw_rate_rad_s"] = yaw_rate
    att_df["yaw_rate_smooth_rad_s"] = yaw_rate_smooth
    att_df["roll_rate_smooth_rad_s"] = roll_rate_smooth
    att_df["pitch_rate_smooth_rad_s"] = pitch_rate_smooth

    for col in ["omega_x", "omega_y", "omega_z"]:
        att_df[f"pose_{col}"] = pose_df[col].to_numpy(dtype=float)

    for axis in ["x", "y", "z"]:
        vals, dt = nearest_values(
            query_t=time_s,
            ref_t=gyro_df["time_s"].to_numpy(dtype=float),
            ref_vals=gyro_df[axis].to_numpy(dtype=float),
        )
        att_df[f"raw_gyro_{axis}"] = vals
        att_df[f"raw_gyro_{axis}_nearest_dt_s"] = dt

    candidate_cols = [
        "pose_omega_x",
        "pose_omega_y",
        "pose_omega_z",
        "raw_gyro_x",
        "raw_gyro_y",
        "raw_gyro_z",
    ]

    score_rows: List[Dict[str, Any]] = []
    target = att_df["yaw_rate_smooth_rad_s"].to_numpy(dtype=float)

    for col in candidate_cols:
        vals = att_df[col].to_numpy(dtype=float)
        corr_pos = finite_corr(target, vals)
        corr_neg = finite_corr(target, -vals)

        score_rows.append(
            {
                "candidate": col,
                "sign": "+",
                "corr": corr_pos,
                "abs_corr": abs(corr_pos) if corr_pos is not None else None,
            }
        )
        score_rows.append(
            {
                "candidate": f"-{col}",
                "sign": "-",
                "source_column": col,
                "corr": corr_neg,
                "abs_corr": abs(corr_neg) if corr_neg is not None else None,
            }
        )

    score_df = pd.DataFrame(score_rows)
    score_df = score_df.sort_values("abs_corr", ascending=False, na_position="last").reset_index(drop=True)

    best_candidate = None
    best_corr = None
    best_abs_corr = None

    if len(score_df) and pd.notna(score_df.loc[0, "abs_corr"]):
        best_candidate = str(score_df.loc[0, "candidate"])
        best_corr = float(score_df.loc[0, "corr"])
        best_abs_corr = float(score_df.loc[0, "abs_corr"])

        if best_candidate.startswith("-"):
            source = best_candidate[1:]
            att_df[best_candidate] = -att_df[source]
        else:
            source = best_candidate
            att_df[best_candidate] = att_df[source]

    attitude_csv = metadata_dir / "attitude_by_pose_timestamp.csv"
    scores_csv = metadata_dir / "yaw_rate_gyro_alignment.csv"

    att_df.to_csv(attitude_csv, index=False)
    score_df.to_csv(scores_csv, index=False)

    plot_attitude(att_df, fig_dir / "attitude_rpy_over_time.png")
    plot_quaternion_norm(att_df, fig_dir / "quaternion_norm.png")
    plot_correlation_bars(score_df, fig_dir / "gyro_axis_correlation_bars.png")

    if best_candidate is not None:
        plot_yaw_rate_best(att_df, best_candidate, fig_dir / "yaw_rate_vs_best_gyro.png")
    else:
        plot_yaw_rate_best(att_df, "pose_omega_z", fig_dir / "yaw_rate_vs_best_gyro.png")

    pose_omega_norm = np.linalg.norm(
        pose_df[["omega_x", "omega_y", "omega_z"]].to_numpy(dtype=float),
        axis=1,
    )

    raw_gyro_norm = np.linalg.norm(
        att_df[["raw_gyro_x", "raw_gyro_y", "raw_gyro_z"]].to_numpy(dtype=float),
        axis=1,
    )

    likely_decision = "unknown"
    if best_abs_corr is not None and best_abs_corr >= 0.8:
        likely_decision = "strong_axis_sign_candidate_found"
    elif best_abs_corr is not None and best_abs_corr >= 0.5:
        likely_decision = "moderate_axis_sign_candidate_found"
    else:
        likely_decision = "weak_or_ambiguous_axis_mapping"

    decision_notes = [
        "Quaternion was converted as w,x,y,z. This convention is diagnostic until dataset documentation confirms it.",
        "Yaw-rate from unwrapped quaternion yaw is compared against pose omega and raw gyro axes.",
        "A high absolute correlation suggests an axis/sign candidate, not a final calibration.",
        "Camera-to-body extrinsics are still not solved by this diagnostic.",
    ]

    summary = {
        "block": "10A5_week4A1_attitude_axis_mapping",
        "dataset_name": dataset_name,
        "config_path": str(config_path),
        "raw_dir": str(raw_dir),
        "output_dir": str(output_dir),
        "pose_path": str(pose_path),
        "gyro_path": str(gyro_path),
        "pose_rows": int(len(pose_df)),
        "gyro_rows": int(len(gyro_df)),
        "pose_time_unit": pose_time_unit,
        "gyro_time_unit": gyro_time_unit,
        "smooth_window": smooth_window,
        "generated_outputs": {
            "attitude_by_pose_timestamp_csv": str(attitude_csv),
            "yaw_rate_gyro_alignment_csv": str(scores_csv),
            "summary_json": str(report_dir / "attitude_axis_summary.json"),
            "attitude_rpy_over_time_png": str(fig_dir / "attitude_rpy_over_time.png"),
            "yaw_rate_vs_best_gyro_png": str(fig_dir / "yaw_rate_vs_best_gyro.png"),
            "gyro_axis_correlation_bars_png": str(fig_dir / "gyro_axis_correlation_bars.png"),
            "quaternion_norm_png": str(fig_dir / "quaternion_norm.png"),
        },
        "quaternion_norm_summary": stats(quat_norm),
        "roll_summary": angle_stats_deg(roll),
        "pitch_summary": angle_stats_deg(pitch),
        "yaw_unwrapped_summary": angle_stats_deg(yaw_unwrapped),
        "yaw_rate_smooth_rad_s_summary": stats(yaw_rate_smooth),
        "pose_omega_norm_summary": stats(pose_omega_norm),
        "raw_gyro_norm_summary": stats(raw_gyro_norm),
        "best_yaw_rate_axis_candidate": {
            "candidate": best_candidate,
            "corr": best_corr,
            "abs_corr": best_abs_corr,
        },
        "likely_decision": likely_decision,
        "top_alignment_scores": score_df.head(12).to_dict(orient="records"),
        "decision_notes": decision_notes,
    }

    summary_json = report_dir / "attitude_axis_summary.json"
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(json_safe(summary), f, indent=2)

    print("Block 10A.5 / Week 4A.1 attitude-axis diagnostics generated")
    print("-----------------------------------------------------------")
    print(f"Dataset:        {dataset_name}")
    print(f"Pose rows:      {len(pose_df)}")
    print(f"Gyro rows:      {len(gyro_df)}")
    print(f"Pose time unit: {pose_time_unit}")
    print(f"Gyro time unit: {gyro_time_unit}")
    print("")
    print("Generated:")
    print(f"- {attitude_csv}")
    print(f"- {scores_csv}")
    print(f"- {summary_json}")
    print(f"- {fig_dir / 'attitude_rpy_over_time.png'}")
    print(f"- {fig_dir / 'yaw_rate_vs_best_gyro.png'}")
    print(f"- {fig_dir / 'gyro_axis_correlation_bars.png'}")
    print(f"- {fig_dir / 'quaternion_norm.png'}")
    print("")
    print("Key checks:")
    print(f"- quaternion norm median: {stats(quat_norm).get('median')}")
    print(f"- roll range deg:         {angle_stats_deg(roll).get('deg_min')} to {angle_stats_deg(roll).get('deg_max')}")
    print(f"- pitch range deg:        {angle_stats_deg(pitch).get('deg_min')} to {angle_stats_deg(pitch).get('deg_max')}")
    print(f"- yaw unwrapped range deg:{angle_stats_deg(yaw_unwrapped).get('deg_min')} to {angle_stats_deg(yaw_unwrapped).get('deg_max')}")
    print("")
    print("Best yaw-rate axis/sign candidate:")
    print(f"- candidate: {best_candidate}")
    print(f"- corr:      {best_corr}")
    print(f"- decision:  {likely_decision}")
    print("")
    print("Top alignment scores:")
    for _, row in score_df.head(8).iterrows():
        print(f"- {row['candidate']}: corr={row['corr']}, abs_corr={row['abs_corr']}")


if __name__ == "__main__":
    main()
