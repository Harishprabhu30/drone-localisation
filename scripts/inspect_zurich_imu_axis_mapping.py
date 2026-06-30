#!/usr/bin/env python3
"""
Block 10A.6 — Zurich raw IMU ↔ OnboardPose axis/sign mapping.

This script does NOT perform fusion.
It compares:
- RawAccel x/y/z against OnboardPose accel_x/y/z
- RawGyro x/y/z against OnboardPose omega_x/y/z

It finds the best axis/sign correspondence by correlation.
"""

from __future__ import annotations

import argparse
import json
from itertools import permutations, product
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


RAW_ACCEL_COLUMNS = [
    "timestamp", "error_count", "x", "y", "z", "temperature",
    "range_rad_s", "scaling", "x_raw", "y_raw", "z_raw", "temperature_raw",
]

RAW_GYRO_COLUMNS = [
    "timestamp", "error_count", "x", "y", "z", "temperature",
    "range_rad_s", "scaling", "x_raw", "y_raw", "z_raw", "temperature_raw",
]

ONBOARD_POSE_COLUMNS = [
    "timestamp",
    "omega_x", "omega_y", "omega_z",
    "accel_x", "accel_y", "accel_z",
    "vel_x", "vel_y", "vel_z",
    "acc_bias_x", "acc_bias_y", "acc_bias_z",
    "azimuth",
    "attitude_w", "attitude_x", "attitude_y", "attitude_z",
    "height", "altitude", "veh_pitch",
    "tether_angle", "tether_angle_dot", "tether_force", "gps_on",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--smooth-window", type=int, default=11)
    return parser.parse_args()


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f)
    return loaded if isinstance(loaded, dict) else {}


def deep_find_first(obj: Any, keys: List[str]) -> Optional[Any]:
    if isinstance(obj, dict):
        for k, v in obj.items():
            if str(k) in keys:
                return v
        for v in obj.values():
            found = deep_find_first(v, keys)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for item in obj:
            found = deep_find_first(item, keys)
            if found is not None:
                return found
    return None


def resolve_path(root: Path, value: Any) -> Optional[Path]:
    if value is None:
        return None
    p = Path(str(value))
    return p if p.is_absolute() else root / p


def infer_dataset_name(cfg: Dict[str, Any], config_path: Path) -> str:
    found = deep_find_first(cfg, ["dataset_name", "name"])
    if found is not None and str(found).strip():
        return str(found)
    return config_path.stem.replace("dataset_", "")


def infer_dirs(cfg: Dict[str, Any], config_path: Path, root: Path) -> Tuple[Path, Path, str]:
    dataset_name = infer_dataset_name(cfg, config_path)

    raw_value = deep_find_first(
        cfg,
        ["raw_data_dir", "raw_dir", "data_raw_dir", "dataset_dir", "raw_dataset_dir"],
    )
    output_value = deep_find_first(cfg, ["output_dir", "outputs_dir", "output_root"])

    raw_dir = resolve_path(root, raw_value)
    output_dir = resolve_path(root, output_value)

    if raw_dir is None:
        raw_dir = root / "data" / "raw" / dataset_name
    if output_dir is None:
        output_dir = root / "outputs" / dataset_name

    return raw_dir, output_dir, dataset_name


def find_file(base: Path, names: List[str]) -> Optional[Path]:
    for name in names:
        p = base / name
        if p.exists():
            return p
    for name in names:
        matches = list(base.rglob(name))
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
    df = df[df["timestamp"].notna()].copy()
    return df.reset_index(drop=True)


def infer_timestamp_scale(ts: np.ndarray) -> Tuple[float, str]:
    finite = ts[np.isfinite(ts)]
    if finite.size < 2:
        return 1.0, "unknown"

    diffs = np.diff(finite)
    pos = diffs[diffs > 0]

    abs_med = float(np.nanmedian(np.abs(finite)))
    dt_med = float(np.nanmedian(pos)) if pos.size else np.nan

    if abs_med > 1e14:
        return 1e-6, "microseconds_inferred_from_large_epoch"
    if abs_med > 1e11:
        return 1e-3, "milliseconds_inferred_from_large_epoch"
    if np.isfinite(dt_med) and dt_med > 1000:
        return 1e-6, "microseconds_inferred_from_delta"
    if np.isfinite(dt_med) and dt_med > 10 and abs_med > 1e8:
        return 1e-3, "milliseconds_inferred_from_delta"

    return 1.0, "seconds_or_normalized"


def add_time_s(df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    out = df.copy()
    ts = pd.to_numeric(out["timestamp"], errors="coerce").to_numpy(dtype=float)
    scale, label = infer_timestamp_scale(ts)
    out["time_s"] = ts * scale
    return out, label


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values
    if window % 2 == 0:
        window += 1
    return (
        pd.Series(values)
        .rolling(window=window, center=True, min_periods=max(3, window // 3))
        .mean()
        .bfill()
        .ffill()
        .to_numpy(dtype=float)
    )


def nearest_dataframe(query_t: np.ndarray, ref_df: pd.DataFrame, cols: List[str], prefix: str) -> pd.DataFrame:
    ref = ref_df.copy()
    ref = ref[np.isfinite(ref["time_s"])].sort_values("time_s").reset_index(drop=True)

    out = pd.DataFrame(index=np.arange(len(query_t)))
    if len(ref) == 0:
        for col in cols:
            out[f"{prefix}_{col}"] = np.nan
        out[f"{prefix}_dt_s"] = np.nan
        return out

    ref_t = ref["time_s"].to_numpy(dtype=float)
    idx = np.searchsorted(ref_t, query_t)

    left = np.clip(idx - 1, 0, len(ref_t) - 1)
    right = np.clip(idx, 0, len(ref_t) - 1)

    left_dt = ref_t[left] - query_t
    right_dt = ref_t[right] - query_t

    choose_right = np.abs(right_dt) < np.abs(left_dt)
    best = np.where(choose_right, right, left)

    for col in cols:
        out[f"{prefix}_{col}"] = ref[col].to_numpy(dtype=float)[best]

    out[f"{prefix}_dt_s"] = ref_t[best] - query_t
    return out


def corr(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    mask = np.isfinite(a) & np.isfinite(b)
    a = a[mask]
    b = b[mask]
    if a.size < 20:
        return None
    if np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def evaluate_axis_mapping(raw: np.ndarray, pose: np.ndarray) -> pd.DataFrame:
    """
    raw shape: N x 3
    pose shape: N x 3

    Test all permutations and sign flips:
    mapped_raw[:, i] = sign[i] * raw[:, perm[i]]
    compare mapped_raw[:, i] to pose[:, i]
    """

    axis_names = ["x", "y", "z"]
    rows: List[Dict[str, Any]] = []

    for perm in permutations([0, 1, 2]):
        for signs in product([-1, 1], repeat=3):
            mapped = np.zeros_like(raw, dtype=float)
            per_axis_corrs: List[Optional[float]] = []

            for target_i in range(3):
                mapped[:, target_i] = signs[target_i] * raw[:, perm[target_i]]
                c = corr(mapped[:, target_i], pose[:, target_i])
                per_axis_corrs.append(c)

            valid_corrs = [abs(c) for c in per_axis_corrs if c is not None]
            mean_abs_corr = float(np.mean(valid_corrs)) if valid_corrs else None
            min_abs_corr = float(np.min(valid_corrs)) if valid_corrs else None

            mapping_text = {
                axis_names[target_i]: f"{'+' if signs[target_i] > 0 else '-'}raw_{axis_names[perm[target_i]]}"
                for target_i in range(3)
            }

            rows.append(
                {
                    "pose_x_from": mapping_text["x"],
                    "pose_y_from": mapping_text["y"],
                    "pose_z_from": mapping_text["z"],
                    "corr_x": per_axis_corrs[0],
                    "corr_y": per_axis_corrs[1],
                    "corr_z": per_axis_corrs[2],
                    "mean_abs_corr": mean_abs_corr,
                    "min_abs_corr": min_abs_corr,
                }
            )

    df = pd.DataFrame(rows)
    df = df.sort_values(["mean_abs_corr", "min_abs_corr"], ascending=False, na_position="last")
    return df.reset_index(drop=True)


def vector_norm(arr: np.ndarray) -> np.ndarray:
    return np.linalg.norm(arr, axis=1)


def stats(values: np.ndarray) -> Dict[str, Optional[float]]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"count": 0, "min": None, "median": None, "mean": None, "max": None, "std": None}
    return {
        "count": int(values.size),
        "min": float(np.min(values)),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "max": float(np.max(values)),
        "std": float(np.std(values)),
    }


def downsample(df: pd.DataFrame, max_points: int = 8000) -> pd.DataFrame:
    if len(df) <= max_points:
        return df
    idx = np.linspace(0, len(df) - 1, max_points).astype(int)
    return df.iloc[idx].copy()


def plot_best_mapping(compare_df: pd.DataFrame, best_row: pd.Series, sensor: str, out_path: Path) -> None:
    d = compare_df.copy()
    d = downsample(d)

    t = d["time_s"].to_numpy(dtype=float)
    t = t - np.nanmin(t)

    plt.figure(figsize=(11, 6))

    axes = ["x", "y", "z"]
    for axis in axes:
        source_text = best_row[f"pose_{axis}_from"]
        sign = -1.0 if source_text.startswith("-") else 1.0
        raw_axis = source_text.split("raw_")[-1]

        pose_col = f"pose_{axis}"
        raw_col = f"raw_{raw_axis}"
        mapped = sign * d[raw_col].to_numpy(dtype=float)

        plt.plot(t, d[pose_col].to_numpy(dtype=float), label=f"{sensor} pose_{axis}", linewidth=0.8)
        plt.plot(t, mapped, label=f"{sensor} mapped {source_text} → pose_{axis}", linewidth=0.8, linestyle="--")

    plt.xlabel("Time from raw stream start [s]")
    plt.ylabel("Signal value")
    plt.title(f"{sensor} best raw-to-OnboardPose axis/sign mapping")
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_mapping_scores(score_df: pd.DataFrame, title: str, out_path: Path) -> None:
    d = score_df.head(12).copy()
    labels = [
        f"x:{r.pose_x_from}\ny:{r.pose_y_from}\nz:{r.pose_z_from}"
        for r in d.itertuples()
    ]
    values = d["mean_abs_corr"].to_numpy(dtype=float)

    plt.figure(figsize=(12, 5))
    plt.bar(np.arange(len(values)), values)
    plt.xticks(np.arange(len(values)), labels, rotation=45, ha="right", fontsize=8)
    plt.ylabel("Mean absolute correlation")
    plt.title(title)
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

    root = Path.cwd()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = root / config_path

    cfg = load_yaml(config_path)
    raw_dir, output_dir, dataset_name = infer_dirs(cfg, config_path, root)

    accel_path = find_file(raw_dir, ["RawAccel.csv", "RawAccelerometer.csv", "RawAccelerometerData.csv"])
    gyro_path = find_file(raw_dir, ["RawGyro.csv", "RawGyroscope.csv", "RawGyroscopeData.csv"])
    pose_path = find_file(raw_dir, ["OnboardPose.csv"])

    if accel_path is None:
        raise FileNotFoundError(f"Raw accelerometer file not found in {raw_dir}")
    if gyro_path is None:
        raise FileNotFoundError(f"Raw gyroscope file not found in {raw_dir}")
    if pose_path is None:
        raise FileNotFoundError(f"OnboardPose.csv not found in {raw_dir}")

    accel_df, accel_unit = add_time_s(read_csv_expected(accel_path, RAW_ACCEL_COLUMNS))
    gyro_df, gyro_unit = add_time_s(read_csv_expected(gyro_path, RAW_GYRO_COLUMNS))
    pose_df, pose_unit = add_time_s(read_csv_expected(pose_path, ONBOARD_POSE_COLUMNS))

    metadata_dir = output_dir / "metadata" / "10a_imu_axis_mapping"
    report_dir = output_dir / "reports" / "10a_imu_axis_mapping"
    fig_dir = output_dir / "figures" / "10a_imu_axis_mapping"

    metadata_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    smooth_window = int(args.smooth_window)
    if smooth_window < 1:
        smooth_window = 1

    # Compare raw accel to nearest pose accel at raw accel timestamps.
    accel_query_t = accel_df["time_s"].to_numpy(dtype=float)
    pose_accel_nearest = nearest_dataframe(
        accel_query_t,
        pose_df,
        ["accel_x", "accel_y", "accel_z"],
        "pose",
    )

    accel_compare = pd.DataFrame()
    accel_compare["time_s"] = accel_query_t
    accel_compare["raw_x"] = rolling_mean(accel_df["x"].to_numpy(dtype=float), smooth_window)
    accel_compare["raw_y"] = rolling_mean(accel_df["y"].to_numpy(dtype=float), smooth_window)
    accel_compare["raw_z"] = rolling_mean(accel_df["z"].to_numpy(dtype=float), smooth_window)
    accel_compare["pose_x"] = rolling_mean(pose_accel_nearest["pose_accel_x"].to_numpy(dtype=float), smooth_window)
    accel_compare["pose_y"] = rolling_mean(pose_accel_nearest["pose_accel_y"].to_numpy(dtype=float), smooth_window)
    accel_compare["pose_z"] = rolling_mean(pose_accel_nearest["pose_accel_z"].to_numpy(dtype=float), smooth_window)
    accel_compare["pose_dt_s"] = pose_accel_nearest["pose_dt_s"]

    # Compare raw gyro to nearest pose omega at raw gyro timestamps.
    gyro_query_t = gyro_df["time_s"].to_numpy(dtype=float)
    pose_gyro_nearest = nearest_dataframe(
        gyro_query_t,
        pose_df,
        ["omega_x", "omega_y", "omega_z"],
        "pose",
    )

    gyro_compare = pd.DataFrame()
    gyro_compare["time_s"] = gyro_query_t
    gyro_compare["raw_x"] = rolling_mean(gyro_df["x"].to_numpy(dtype=float), smooth_window)
    gyro_compare["raw_y"] = rolling_mean(gyro_df["y"].to_numpy(dtype=float), smooth_window)
    gyro_compare["raw_z"] = rolling_mean(gyro_df["z"].to_numpy(dtype=float), smooth_window)
    gyro_compare["pose_x"] = rolling_mean(pose_gyro_nearest["pose_omega_x"].to_numpy(dtype=float), smooth_window)
    gyro_compare["pose_y"] = rolling_mean(pose_gyro_nearest["pose_omega_y"].to_numpy(dtype=float), smooth_window)
    gyro_compare["pose_z"] = rolling_mean(pose_gyro_nearest["pose_omega_z"].to_numpy(dtype=float), smooth_window)
    gyro_compare["pose_dt_s"] = pose_gyro_nearest["pose_dt_s"]

    accel_raw = accel_compare[["raw_x", "raw_y", "raw_z"]].to_numpy(dtype=float)
    accel_pose = accel_compare[["pose_x", "pose_y", "pose_z"]].to_numpy(dtype=float)

    gyro_raw = gyro_compare[["raw_x", "raw_y", "raw_z"]].to_numpy(dtype=float)
    gyro_pose = gyro_compare[["pose_x", "pose_y", "pose_z"]].to_numpy(dtype=float)

    accel_scores = evaluate_axis_mapping(accel_raw, accel_pose)
    gyro_scores = evaluate_axis_mapping(gyro_raw, gyro_pose)

    accel_compare_csv = metadata_dir / "raw_accel_vs_pose_accel_timeseries.csv"
    gyro_compare_csv = metadata_dir / "raw_gyro_vs_pose_omega_timeseries.csv"
    accel_scores_csv = metadata_dir / "accel_axis_mapping_scores.csv"
    gyro_scores_csv = metadata_dir / "gyro_axis_mapping_scores.csv"

    accel_compare.to_csv(accel_compare_csv, index=False)
    gyro_compare.to_csv(gyro_compare_csv, index=False)
    accel_scores.to_csv(accel_scores_csv, index=False)
    gyro_scores.to_csv(gyro_scores_csv, index=False)

    plot_best_mapping(
        accel_compare,
        accel_scores.iloc[0],
        "accelerometer",
        fig_dir / "accel_best_axis_mapping.png",
    )
    plot_best_mapping(
        gyro_compare,
        gyro_scores.iloc[0],
        "gyroscope",
        fig_dir / "gyro_best_axis_mapping.png",
    )
    plot_mapping_scores(
        accel_scores,
        "Top accelerometer raw-to-pose axis/sign mappings",
        fig_dir / "accel_axis_mapping_scores.png",
    )
    plot_mapping_scores(
        gyro_scores,
        "Top gyroscope raw-to-pose axis/sign mappings",
        fig_dir / "gyro_axis_mapping_scores.png",
    )

    accel_best = accel_scores.iloc[0].to_dict()
    gyro_best = gyro_scores.iloc[0].to_dict()

    accel_norm_corr = corr(vector_norm(accel_raw), vector_norm(accel_pose))
    gyro_norm_corr = corr(vector_norm(gyro_raw), vector_norm(gyro_pose))

    if gyro_best["mean_abs_corr"] is not None and gyro_best["mean_abs_corr"] >= 0.75:
        gyro_decision = "strong_raw_gyro_to_pose_omega_mapping"
    elif gyro_best["mean_abs_corr"] is not None and gyro_best["mean_abs_corr"] >= 0.5:
        gyro_decision = "moderate_raw_gyro_to_pose_omega_mapping"
    else:
        gyro_decision = "weak_or_unclear_raw_gyro_to_pose_omega_mapping"

    if accel_best["mean_abs_corr"] is not None and accel_best["mean_abs_corr"] >= 0.75:
        accel_decision = "strong_raw_accel_to_pose_accel_mapping"
    elif accel_best["mean_abs_corr"] is not None and accel_best["mean_abs_corr"] >= 0.5:
        accel_decision = "moderate_raw_accel_to_pose_accel_mapping"
    else:
        accel_decision = "weak_or_unclear_raw_accel_to_pose_accel_mapping"

    summary = {
        "block": "10A6_raw_imu_to_onboardpose_axis_mapping",
        "dataset_name": dataset_name,
        "config_path": str(config_path),
        "raw_dir": str(raw_dir),
        "output_dir": str(output_dir),
        "accel_path": str(accel_path),
        "gyro_path": str(gyro_path),
        "pose_path": str(pose_path),
        "timestamp_units": {
            "raw_accel": accel_unit,
            "raw_gyro": gyro_unit,
            "onboard_pose": pose_unit,
        },
        "smooth_window": smooth_window,
        "generated_outputs": {
            "raw_accel_vs_pose_accel_timeseries_csv": str(accel_compare_csv),
            "raw_gyro_vs_pose_omega_timeseries_csv": str(gyro_compare_csv),
            "accel_axis_mapping_scores_csv": str(accel_scores_csv),
            "gyro_axis_mapping_scores_csv": str(gyro_scores_csv),
            "summary_json": str(report_dir / "imu_axis_mapping_summary.json"),
            "accel_best_axis_mapping_png": str(fig_dir / "accel_best_axis_mapping.png"),
            "gyro_best_axis_mapping_png": str(fig_dir / "gyro_best_axis_mapping.png"),
            "accel_axis_mapping_scores_png": str(fig_dir / "accel_axis_mapping_scores.png"),
            "gyro_axis_mapping_scores_png": str(fig_dir / "gyro_axis_mapping_scores.png"),
        },
        "accel_best_mapping": accel_best,
        "gyro_best_mapping": gyro_best,
        "accel_norm_corr": accel_norm_corr,
        "gyro_norm_corr": gyro_norm_corr,
        "accel_decision": accel_decision,
        "gyro_decision": gyro_decision,
        "notes": [
            "This block compares raw IMU with OnboardPose IMU-like fields only.",
            "A high correlation indicates likely axis/sign relation, not final physical calibration.",
            "Camera-to-IMU extrinsics are still unknown.",
            "The mapping should be validated again using camera motion in Block 10A.7.",
        ],
    }

    summary_json = report_dir / "imu_axis_mapping_summary.json"
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(json_safe(summary), f, indent=2)

    print("Block 10A.6 raw IMU ↔ OnboardPose axis/sign mapping generated")
    print("------------------------------------------------------------")
    print(f"Dataset:        {dataset_name}")
    print(f"Raw dir:        {raw_dir}")
    print(f"Output dir:     {output_dir}")
    print("")
    print("Generated:")
    print(f"- {accel_compare_csv}")
    print(f"- {gyro_compare_csv}")
    print(f"- {accel_scores_csv}")
    print(f"- {gyro_scores_csv}")
    print(f"- {summary_json}")
    print(f"- {fig_dir / 'accel_best_axis_mapping.png'}")
    print(f"- {fig_dir / 'gyro_best_axis_mapping.png'}")
    print(f"- {fig_dir / 'accel_axis_mapping_scores.png'}")
    print(f"- {fig_dir / 'gyro_axis_mapping_scores.png'}")
    print("")
    print("Best accelerometer mapping:")
    print(f"- pose_x_from:     {accel_best['pose_x_from']}")
    print(f"- pose_y_from:     {accel_best['pose_y_from']}")
    print(f"- pose_z_from:     {accel_best['pose_z_from']}")
    print(f"- corr_x/y/z:      {accel_best['corr_x']}, {accel_best['corr_y']}, {accel_best['corr_z']}")
    print(f"- mean_abs_corr:   {accel_best['mean_abs_corr']}")
    print(f"- decision:        {accel_decision}")
    print("")
    print("Best gyroscope mapping:")
    print(f"- pose_x_from:     {gyro_best['pose_x_from']}")
    print(f"- pose_y_from:     {gyro_best['pose_y_from']}")
    print(f"- pose_z_from:     {gyro_best['pose_z_from']}")
    print(f"- corr_x/y/z:      {gyro_best['corr_x']}, {gyro_best['corr_y']}, {gyro_best['corr_z']}")
    print(f"- mean_abs_corr:   {gyro_best['mean_abs_corr']}")
    print(f"- decision:        {gyro_decision}")
    print("")
    print("Norm correlations:")
    print(f"- accel_norm_corr: {accel_norm_corr}")
    print(f"- gyro_norm_corr:  {gyro_norm_corr}")


if __name__ == "__main__":
    main()
