#!/usr/bin/env python3
"""
Block 10B — Sensor-assisted ORB metric scaling experiment.

This script is EXPERIMENTAL and separate from the camera-only ORB metric baseline.

It recomputes ORB pair motion on selected Zurich windows and tests:
- height candidates:
    fixed
    pose_altitude_relative
    baro_relative
    height_agl_column, if available in synchronized_frames_enriched.csv
- yaw candidates:
    fixed_zero
    quat_yaw
    pose_omega_integrated
    raw_gyro_z_corrected_integrated
- image-axis variants:
    normal / swapped
    x/y sign combinations

The goal is not to claim final fusion.
The goal is to determine whether sensor-assisted metric conversion improves over the
current simple ORB metric scaling layer.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


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

BAROMETER_COLUMNS = [
    "timestamp",
    "pressure",
    "altitude",
    "temperature",
]

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


DEFAULT_RUNS = [
    ("full_00001_01000_stride1", 1, 1000, 1),
    ("full_00001_01000_stride5", 1, 1000, 5),
    ("full_40000_41000_stride1", 40000, 41000, 1),
    ("full_40000_41000_stride5", 40000, 41000, 5),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--official-runs", action="store_true")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--start-imgid", type=int, default=None)
    parser.add_argument("--end-imgid", type=int, default=None)
    parser.add_argument("--stride", type=int, default=None)

    parser.add_argument("--base-height-m", type=float, default=50.0)
    parser.add_argument("--focal-px", type=float, default=None)
    parser.add_argument("--max-features", type=int, default=4000)
    parser.add_argument("--ratio", type=float, default=0.75)
    parser.add_argument("--ransac-threshold", type=float, default=3.0)
    parser.add_argument("--max-pairs", type=int, default=None)

    parser.add_argument(
        "--height-modes",
        default="fixed,pose_altitude_relative,baro_relative,height_agl_column",
    )
    parser.add_argument(
        "--yaw-modes",
        default="fixed_zero,quat_yaw,pose_omega_integrated,raw_gyro_z_corrected_integrated",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f)
    return loaded if isinstance(loaded, dict) else {}


def deep_find_first(obj: Any, keys: List[str]) -> Optional[Any]:
    if isinstance(obj, dict):
        for key, value in obj.items():
            if str(key) in keys:
                return value
        for value in obj.values():
            found = deep_find_first(value, keys)
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
    if "timestamp" in df.columns:
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


def read_sync_csv(output_dir: Path) -> Tuple[Path, pd.DataFrame]:
    candidates = [
        output_dir / "metadata" / "synchronized_frames_enriched.csv",
        output_dir / "metadata" / "synchronized_frames.csv",
    ]
    for p in candidates:
        if p.exists():
            return p, pd.read_csv(p)
    raise FileNotFoundError("Missing synchronized_frames_enriched.csv or synchronized_frames.csv")


def normalize_sync_time(sync_df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    if "timestamp" not in sync_df.columns:
        raise RuntimeError("Synchronized frames CSV must contain timestamp.")
    out = sync_df.copy()
    ts = pd.to_numeric(out["timestamp"], errors="coerce").to_numpy(dtype=float)
    scale, label = infer_timestamp_scale(ts)
    out["time_s"] = ts * scale
    return out, label


def quaternion_to_yaw(w: np.ndarray, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    norm = np.sqrt(w * w + x * x + y * y + z * z)
    norm = np.where(norm == 0, np.nan, norm)

    w = w / norm
    x = x / norm
    y = y / norm
    z = z / norm

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return np.unwrap(np.arctan2(siny_cosp, cosy_cosp))


def nearest_value(query_t: np.ndarray, ref_t: np.ndarray, ref_v: np.ndarray) -> np.ndarray:
    ref_t = np.asarray(ref_t, dtype=float)
    ref_v = np.asarray(ref_v, dtype=float)

    valid = np.isfinite(ref_t) & np.isfinite(ref_v)
    ref_t = ref_t[valid]
    ref_v = ref_v[valid]

    if ref_t.size == 0:
        return np.full(query_t.shape, np.nan)

    order = np.argsort(ref_t)
    ref_t = ref_t[order]
    ref_v = ref_v[order]

    idx = np.searchsorted(ref_t, query_t)
    left = np.clip(idx - 1, 0, ref_t.size - 1)
    right = np.clip(idx, 0, ref_t.size - 1)

    left_dt = ref_t[left] - query_t
    right_dt = ref_t[right] - query_t

    choose_right = np.abs(right_dt) < np.abs(left_dt)
    best = np.where(choose_right, right, left)

    return ref_v[best]


def integrate_rate(t: np.ndarray, rate: np.ndarray, t0: float, t1: float) -> Optional[float]:
    if not np.isfinite(t0) or not np.isfinite(t1) or t1 <= t0:
        return None

    t = np.asarray(t, dtype=float)
    rate = np.asarray(rate, dtype=float)
    valid = np.isfinite(t) & np.isfinite(rate)
    t = t[valid]
    rate = rate[valid]

    if t.size < 2 or t0 < np.min(t) or t1 > np.max(t):
        return None

    order = np.argsort(t)
    t = t[order]
    rate = rate[order]

    mask = (t >= t0) & (t <= t1)
    ti = t[mask]
    ri = rate[mask]

    r0 = np.interp(t0, t, rate)
    r1 = np.interp(t1, t, rate)

    ti = np.concatenate([[t0], ti, [t1]])
    ri = np.concatenate([[r0], ri, [r1]])

    return float(np.trapz(ri, ti))


def find_image_column(df: pd.DataFrame) -> Optional[str]:
    candidates = [
        "image_path",
        "frame_path",
        "mav_image_path",
        "path",
        "filename",
        "file_name",
        "image_filename",
        "image_file",
    ]
    for col in candidates:
        if col in df.columns:
            return col
    for col in df.columns:
        name = str(col).lower()
        if "image" in name and ("path" in name or "file" in name or "name" in name):
            return col
    return None


def build_imgid_to_path(sync_df: pd.DataFrame, raw_dir: Path) -> Dict[int, Path]:
    image_col = find_image_column(sync_df)
    mapping: Dict[int, Path] = {}

    if image_col is not None:
        for _, row in sync_df.iterrows():
            if pd.isna(row.get("imgid")) or pd.isna(row.get(image_col)):
                continue
            imgid = int(row["imgid"])
            p = Path(str(row[image_col]))
            if not p.is_absolute():
                p1 = Path.cwd() / p
                p2 = raw_dir / p
                p = p1 if p1.exists() else p2
            mapping[imgid] = p

    if mapping:
        return mapping

    image_files: List[Path] = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff"]:
        image_files.extend(raw_dir.rglob(ext))

    by_number: Dict[int, Path] = {}
    for p in image_files:
        groups = "".join(ch if ch.isdigit() else " " for ch in p.stem).split()
        nums = [int(g) for g in groups]
        if nums:
            by_number[nums[-1]] = p

    for imgid in pd.to_numeric(sync_df["imgid"], errors="coerce").dropna().astype(int):
        if imgid in by_number:
            mapping[int(imgid)] = by_number[int(imgid)]

    return mapping


def load_gray(path: Optional[Path]) -> Optional[np.ndarray]:
    if path is None or not path.exists():
        return None
    return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)


def estimate_affine_motion(
    img0: np.ndarray,
    img1: np.ndarray,
    max_features: int,
    ratio: float,
    ransac_threshold: float,
) -> Dict[str, Any]:
    orb = cv2.ORB_create(nfeatures=max_features)

    k0, d0 = orb.detectAndCompute(img0, None)
    k1, d1 = orb.detectAndCompute(img1, None)

    result = {
        "ok": False,
        "num_kp0": 0 if k0 is None else len(k0),
        "num_kp1": 0 if k1 is None else len(k1),
        "num_good_matches": 0,
        "num_inliers": 0,
        "inlier_ratio": np.nan,
        "dx_px": np.nan,
        "dy_px": np.nan,
        "rot_rad": np.nan,
        "scale_affine": np.nan,
        "reason": None,
    }

    if d0 is None or d1 is None or len(k0) < 8 or len(k1) < 8:
        result["reason"] = "not_enough_keypoints"
        return result

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn = matcher.knnMatch(d0, d1, k=2)

    good = []
    for pair in knn:
        if len(pair) != 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            good.append(m)

    result["num_good_matches"] = len(good)

    if len(good) < 8:
        result["reason"] = "not_enough_good_matches"
        return result

    pts0 = np.float32([k0[m.queryIdx].pt for m in good])
    pts1 = np.float32([k1[m.trainIdx].pt for m in good])

    A, mask = cv2.estimateAffinePartial2D(
        pts0,
        pts1,
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_threshold,
        maxIters=2000,
        confidence=0.99,
    )

    if A is None or mask is None:
        result["reason"] = "affine_failed"
        return result

    mask_bool = mask.ravel().astype(bool)
    inliers = int(np.sum(mask_bool))

    result["num_inliers"] = inliers
    result["inlier_ratio"] = float(inliers / max(1, len(good)))

    if inliers < 8:
        result["reason"] = "not_enough_inliers"
        return result

    a, b, tx = float(A[0, 0]), float(A[0, 1]), float(A[0, 2])
    c, d, ty = float(A[1, 0]), float(A[1, 1]), float(A[1, 2])

    theta = math.atan2(c, a)
    scale = math.sqrt(a * a + c * c)

    result.update(
        {
            "ok": True,
            "dx_px": tx,
            "dy_px": ty,
            "rot_rad": theta,
            "scale_affine": scale,
            "reason": "ok",
        }
    )
    return result


def try_load_focal_px(raw_dir: Path, fallback_focal_px: Optional[float]) -> Tuple[float, str]:
    if fallback_focal_px is not None:
        return float(fallback_focal_px), "argument"

    npz_files = list(raw_dir.rglob("calibration_data.npz")) + list(raw_dir.rglob("*.npz"))

    for p in npz_files:
        try:
            data = np.load(p)
            for key in ["K", "camera_matrix", "mtx", "intrinsic_matrix"]:
                if key in data:
                    K = np.asarray(data[key], dtype=float)
                    if K.shape == (3, 3):
                        fx = float(K[0, 0])
                        fy = float(K[1, 1])
                        if np.isfinite(fx) and np.isfinite(fy) and fx > 0 and fy > 0:
                            return float((fx + fy) / 2.0), f"calibration_npz:{p.name}:{key}"
        except Exception:
            continue

    return 800.0, "fallback_800px_no_calibration_key_found"


def pick_reference_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    x_candidates = [
        "ref_x_enu_m",
        "reference_x_enu_m",
        "x_enu_m",
        "x_m",
        "x",
    ]
    y_candidates = [
        "ref_y_enu_m",
        "reference_y_enu_m",
        "y_enu_m",
        "y_m",
        "y",
    ]

    x_col = None
    y_col = None

    for col in x_candidates:
        if col in df.columns:
            x_col = col
            break
    for col in y_candidates:
        if col in df.columns:
            y_col = col
            break

    return x_col, y_col


def evaluate_trajectory(df: pd.DataFrame) -> Dict[str, Any]:
    x_col, y_col = pick_reference_columns(df)

    if x_col is None or y_col is None:
        return {
            "has_reference": False,
            "reason": "reference_xy_columns_missing",
        }

    est = df[["est_x_m", "est_y_m"]].to_numpy(dtype=float)
    ref = df[[x_col, y_col]].to_numpy(dtype=float)

    valid = np.all(np.isfinite(est), axis=1) & np.all(np.isfinite(ref), axis=1)
    est = est[valid]
    ref = ref[valid]

    if len(est) < 2:
        return {
            "has_reference": False,
            "reason": "not_enough_valid_reference_rows",
        }

    est = est - est[0]
    ref = ref - ref[0]

    errors = np.linalg.norm(est - ref, axis=1)

    est_steps = np.linalg.norm(np.diff(est, axis=0), axis=1)
    ref_steps = np.linalg.norm(np.diff(ref, axis=0), axis=1)

    est_path = float(np.sum(est_steps))
    ref_path = float(np.sum(ref_steps))
    final_error = float(errors[-1])
    rmse = float(np.sqrt(np.mean(errors**2)))
    mean_error = float(np.mean(errors))
    max_error = float(np.max(errors))
    drift_per_100m = float(final_error / ref_path * 100.0) if ref_path > 1e-9 else None

    return {
        "has_reference": True,
        "reference_x_col": x_col,
        "reference_y_col": y_col,
        "estimated_path_m": est_path,
        "reference_path_m": ref_path,
        "rmse_m": rmse,
        "mean_error_m": mean_error,
        "max_error_m": max_error,
        "final_error_m": final_error,
        "drift_per_100m": drift_per_100m,
    }


def stats(values: np.ndarray) -> Dict[str, Optional[float]]:
    values = np.asarray(values, dtype=float)
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
        val = float(obj)
        return val if np.isfinite(val) else None
    if isinstance(obj, float):
        return obj if np.isfinite(obj) else None
    return obj


def plot_trajectory(df: pd.DataFrame, out_path: Path, title: str) -> None:
    plt.figure(figsize=(7, 7))

    plt.plot(df["est_x_m"], df["est_y_m"], label="estimated", linewidth=1.2)

    x_col, y_col = pick_reference_columns(df)
    if x_col is not None and y_col is not None:
        ref_x = pd.to_numeric(df[x_col], errors="coerce").to_numpy(dtype=float)
        ref_y = pd.to_numeric(df[y_col], errors="coerce").to_numpy(dtype=float)

        valid = np.isfinite(ref_x) & np.isfinite(ref_y)
        if np.any(valid):
            first_idx = int(np.where(valid)[0][0])
            ref_x = ref_x - ref_x[first_idx]
            ref_y = ref_y - ref_y[first_idx]
            plt.plot(ref_x[valid], ref_y[valid], label="reference", linewidth=1.2)
        else:
            print(f"Warning: reference columns found but no finite reference values: {x_col}, {y_col}")
    else:
        print(f"Warning: reference columns not found for plot: available columns = {list(df.columns)}")

    plt.axis("equal")
    plt.xlabel("x / east-like [m]")
    plt.ylabel("y / north-like [m]")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_error(df: pd.DataFrame, out_path: Path, title: str) -> None:
    x_col, y_col = pick_reference_columns(df)

    plt.figure(figsize=(10, 4))

    if x_col is None or y_col is None:
        plt.text(0.5, 0.5, "Reference columns missing", ha="center", va="center")
        plt.axis("off")
    else:
        est = df[["est_x_m", "est_y_m"]].to_numpy(dtype=float)
        ref = df[[x_col, y_col]].to_numpy(dtype=float)

        valid = np.all(np.isfinite(est), axis=1) & np.all(np.isfinite(ref), axis=1)

        if np.sum(valid) < 2:
            plt.text(0.5, 0.5, "Not enough finite reference values", ha="center", va="center")
            plt.axis("off")
        else:
            est_valid = est[valid]
            ref_valid = ref[valid]

            est_valid = est_valid - est_valid[0]
            ref_valid = ref_valid - ref_valid[0]

            err = np.linalg.norm(est_valid - ref_valid, axis=1)
            plt.plot(np.arange(len(err)), err, linewidth=1.0)
            plt.xlabel("Valid frame index in run")
            plt.ylabel("Position error [m]")
            plt.title(title)

    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def compute_pair_motion_table(
    run_name: str,
    start_imgid: int,
    end_imgid: int,
    stride: int,
    sync_df: pd.DataFrame,
    imgid_to_path: Dict[int, Path],
    max_features: int,
    ratio: float,
    ransac_threshold: float,
    max_pairs: Optional[int],
) -> pd.DataFrame:
    window = sync_df[
        (pd.to_numeric(sync_df["imgid"], errors="coerce") >= start_imgid)
        & (pd.to_numeric(sync_df["imgid"], errors="coerce") <= end_imgid)
    ].copy()
    window = window.sort_values("imgid").reset_index(drop=True)
    window = window.iloc[::stride].copy().reset_index(drop=True)

    if max_pairs is not None and len(window) > max_pairs + 1:
        window = window.iloc[: max_pairs + 1].copy().reset_index(drop=True)

    rows: List[Dict[str, Any]] = []

    for i in range(len(window) - 1):
        r0 = window.iloc[i]
        r1 = window.iloc[i + 1]

        imgid0 = int(r0["imgid"])
        imgid1 = int(r1["imgid"])

        p0 = imgid_to_path.get(imgid0)
        p1 = imgid_to_path.get(imgid1)

        img0 = load_gray(p0)
        img1 = load_gray(p1)

        row: Dict[str, Any] = {
            "run_name": run_name,
            "pair_index": i,
            "imgid0": imgid0,
            "imgid1": imgid1,
            "t0_s": float(r0["time_s"]),
            "t1_s": float(r1["time_s"]),
            "dt_s": float(r1["time_s"] - r0["time_s"]),
            "frame0_index": i,
            "frame1_index": i + 1,
        }

        # Carry reference columns from frame1.
        for col in window.columns:
            if col not in row:
                row[f"frame1_{col}"] = r1[col]

        if img0 is None or img1 is None:
            row.update(
                {
                    "ok": False,
                    "reason": "image_missing",
                    "dx_px": np.nan,
                    "dy_px": np.nan,
                    "rot_rad": np.nan,
                    "scale_affine": np.nan,
                    "num_good_matches": 0,
                    "num_inliers": 0,
                    "inlier_ratio": np.nan,
                }
            )
        else:
            row.update(
                estimate_affine_motion(
                    img0=img0,
                    img1=img1,
                    max_features=max_features,
                    ratio=ratio,
                    ransac_threshold=ransac_threshold,
                )
            )

        rows.append(row)

    return pd.DataFrame(rows)


def build_frame_output_from_pairs(pair_df: pd.DataFrame) -> pd.DataFrame:
    if len(pair_df) == 0:
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []

    first = pair_df.iloc[0]
    first_row = {
        "run_name": first["run_name"],
        "imgid": int(first["imgid0"]),
        "time_s": float(first["t0_s"]),
        "est_x_m": 0.0,
        "est_y_m": 0.0,
    }
    rows.append(first_row)

    for _, pair in pair_df.iterrows():
        row = {
            "run_name": pair["run_name"],
            "imgid": int(pair["imgid1"]),
            "time_s": float(pair["t1_s"]),
        }

        for col in pair_df.columns:
            if col.startswith("frame1_"):
                clean_col = col.replace("frame1_", "", 1)
                row[clean_col] = pair[col]

        rows.append(row)

    return pd.DataFrame(rows)


def apply_sensor_assisted_scaling(
    pair_df: pd.DataFrame,
    frame_df: pd.DataFrame,
    pose_df: pd.DataFrame,
    gyro_df: pd.DataFrame,
    baro_df: pd.DataFrame,
    focal_px: float,
    base_height_m: float,
    height_mode: str,
    yaw_mode: str,
    swap_xy: bool,
    sx: float,
    sy: float,
) -> pd.DataFrame:
    out = frame_df.copy()

    est_x = [0.0]
    est_y = [0.0]

    pose_time = pose_df["time_s"].to_numpy(dtype=float)
    gyro_time = gyro_df["time_s"].to_numpy(dtype=float)
    baro_time = baro_df["time_s"].to_numpy(dtype=float)

    pose_alt = pose_df["altitude"].to_numpy(dtype=float) if "altitude" in pose_df else np.full(len(pose_df), np.nan)
    baro_alt = baro_df["altitude"].to_numpy(dtype=float) if "altitude" in baro_df else np.full(len(baro_df), np.nan)

    quat_yaw = quaternion_to_yaw(
        pose_df["attitude_w"].to_numpy(dtype=float),
        pose_df["attitude_x"].to_numpy(dtype=float),
        pose_df["attitude_y"].to_numpy(dtype=float),
        pose_df["attitude_z"].to_numpy(dtype=float),
    )

    pose_omega_z = pose_df["omega_z"].to_numpy(dtype=float)
    raw_gyro_z_corrected = -gyro_df["z"].to_numpy(dtype=float)

    pose_alt0 = nearest_value(np.array([pair_df.iloc[0]["t0_s"]]), pose_time, pose_alt)[0]
    baro_alt0 = nearest_value(np.array([pair_df.iloc[0]["t0_s"]]), baro_time, baro_alt)[0]
    yaw_accum_pose = 0.0
    yaw_accum_raw = 0.0

    for _, pair in pair_df.iterrows():
        dx_px = float(pair["dx_px"]) if np.isfinite(pair["dx_px"]) else 0.0
        dy_px = float(pair["dy_px"]) if np.isfinite(pair["dy_px"]) else 0.0

        t0 = float(pair["t0_s"])
        t1 = float(pair["t1_s"])

        if height_mode == "fixed":
            height_m = base_height_m
        elif height_mode == "pose_altitude_relative":
            alt = nearest_value(np.array([t1]), pose_time, pose_alt)[0]
            height_m = base_height_m + float(alt - pose_alt0) if np.isfinite(alt) and np.isfinite(pose_alt0) else base_height_m
        elif height_mode == "baro_relative":
            alt = nearest_value(np.array([t1]), baro_time, baro_alt)[0]
            height_m = base_height_m + float(alt - baro_alt0) if np.isfinite(alt) and np.isfinite(baro_alt0) else base_height_m
        elif height_mode == "height_agl_column":
            # This is debug only if present.
            val = pair.get("frame1_height_agl_m", np.nan)
            height_m = float(val) if np.isfinite(val) and float(val) > 0 else base_height_m
        else:
            height_m = base_height_m

        height_m = max(0.5, float(height_m))
        meters_per_px = height_m / focal_px

        mx = dx_px * meters_per_px
        my = dy_px * meters_per_px

        if swap_xy:
            mx, my = my, mx

        mx *= sx
        my *= sy

        if yaw_mode == "fixed_zero":
            yaw = 0.0
        elif yaw_mode == "quat_yaw":
            yaw = nearest_value(np.array([t1]), pose_time, quat_yaw)[0]
            yaw = float(yaw) if np.isfinite(yaw) else 0.0
        elif yaw_mode == "pose_omega_integrated":
            dyaw = integrate_rate(pose_time, pose_omega_z, t0, t1)
            yaw_accum_pose += float(dyaw) if dyaw is not None else 0.0
            yaw = yaw_accum_pose
        elif yaw_mode == "raw_gyro_z_corrected_integrated":
            dyaw = integrate_rate(gyro_time, raw_gyro_z_corrected, t0, t1)
            yaw_accum_raw += float(dyaw) if dyaw is not None else 0.0
            yaw = yaw_accum_raw
        else:
            yaw = 0.0

        c = math.cos(yaw)
        s = math.sin(yaw)

        # Rotate image-plane displacement candidate into an ENU-like local frame.
        de = c * mx - s * my
        dn = s * mx + c * my

        est_x.append(est_x[-1] + de)
        est_y.append(est_y[-1] + dn)

    out["est_x_m"] = est_x[: len(out)]
    out["est_y_m"] = est_y[: len(out)]

    return out


def run_one_window(
    run_name: str,
    start_imgid: int,
    end_imgid: int,
    stride: int,
    sync_df: pd.DataFrame,
    imgid_to_path: Dict[int, Path],
    pose_df: pd.DataFrame,
    gyro_df: pd.DataFrame,
    baro_df: pd.DataFrame,
    output_dir: Path,
    focal_px: float,
    focal_source: str,
    base_height_m: float,
    height_modes: List[str],
    yaw_modes: List[str],
    max_features: int,
    ratio: float,
    ransac_threshold: float,
    max_pairs: Optional[int],
) -> Dict[str, Any]:
    metadata_dir = output_dir / "metadata" / "10b_sensor_assisted_orb_scaling" / run_name
    traj_dir = output_dir / "trajectories" / "10b_sensor_assisted_orb_scaling" / run_name
    report_dir = output_dir / "reports" / "10b_sensor_assisted_orb_scaling" / run_name
    fig_dir = output_dir / "figures" / "10b_sensor_assisted_orb_scaling" / run_name

    for d in [metadata_dir, traj_dir, report_dir, fig_dir]:
        d.mkdir(parents=True, exist_ok=True)

    pair_df = compute_pair_motion_table(
        run_name=run_name,
        start_imgid=start_imgid,
        end_imgid=end_imgid,
        stride=stride,
        sync_df=sync_df,
        imgid_to_path=imgid_to_path,
        max_features=max_features,
        ratio=ratio,
        ransac_threshold=ransac_threshold,
        max_pairs=max_pairs,
    )

    pair_csv = metadata_dir / "orb_sensor_assisted_pair_motion.csv"
    pair_df.to_csv(pair_csv, index=False)

    frame_template = build_frame_output_from_pairs(pair_df)

    # Rename carried reference columns into direct columns where possible.
    # If synchronized frame columns already include x_enu_m/y_enu_m, they will be present.
    summaries: List[Dict[str, Any]] = []

    height_modes_available: List[str] = []
    for hm in height_modes:
        if hm == "height_agl_column" and "frame1_height_agl_m" not in pair_df.columns:
            continue
        height_modes_available.append(hm)

    axis_variants = []
    for swap_xy in [False, True]:
        for sx in [-1.0, 1.0]:
            for sy in [-1.0, 1.0]:
                axis_variants.append((swap_xy, sx, sy))

    for height_mode in height_modes_available:
        for yaw_mode in yaw_modes:
            for swap_xy, sx, sy in axis_variants:
                variant_name = (
                    f"h-{height_mode}__yaw-{yaw_mode}__"
                    f"{'swap' if swap_xy else 'noswap'}__sx{int(sx)}__sy{int(sy)}"
                )

                traj_df = apply_sensor_assisted_scaling(
                    pair_df=pair_df,
                    frame_df=frame_template,
                    pose_df=pose_df,
                    gyro_df=gyro_df,
                    baro_df=baro_df,
                    focal_px=focal_px,
                    base_height_m=base_height_m,
                    height_mode=height_mode,
                    yaw_mode=yaw_mode,
                    swap_xy=swap_xy,
                    sx=sx,
                    sy=sy,
                )

                traj_csv = traj_dir / f"{variant_name}.csv"
                traj_df.to_csv(traj_csv, index=False)

                eval_summary = evaluate_trajectory(traj_df)

                summary = {
                    "run_name": run_name,
                    "variant_name": variant_name,
                    "height_mode": height_mode,
                    "yaw_mode": yaw_mode,
                    "swap_xy": swap_xy,
                    "sx": sx,
                    "sy": sy,
                    "trajectory_csv": str(traj_csv),
                    **eval_summary,
                }
                summaries.append(summary)

    summary_df = pd.DataFrame(summaries)
    summary_df = summary_df.sort_values("rmse_m", ascending=True, na_position="last")

    summary_csv = report_dir / "sensor_assisted_scaling_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    best = summary_df.iloc[0].to_dict() if len(summary_df) else {}

    if best and isinstance(best.get("trajectory_csv"), str):
        best_traj = pd.read_csv(best["trajectory_csv"])
        plot_trajectory(
            best_traj,
            fig_dir / "best_sensor_assisted_trajectory_vs_reference.png",
            f"{run_name}: best sensor-assisted ORB trajectory",
        )
        plot_error(
            best_traj,
            fig_dir / "best_sensor_assisted_error_over_frame.png",
            f"{run_name}: best sensor-assisted ORB error",
        )

    ok_pairs = int(pair_df["ok"].sum()) if "ok" in pair_df.columns else 0

    run_summary = {
        "run_name": run_name,
        "start_imgid": start_imgid,
        "end_imgid": end_imgid,
        "stride": stride,
        "attempted_pairs": int(len(pair_df)),
        "ok_pairs": ok_pairs,
        "failed_pairs": int(len(pair_df) - ok_pairs),
        "median_inlier_ratio": float(np.nanmedian(pair_df["inlier_ratio"])) if len(pair_df) else None,
        "focal_px": focal_px,
        "focal_source": focal_source,
        "base_height_m": base_height_m,
        "height_modes_tested": height_modes_available,
        "yaw_modes_tested": yaw_modes,
        "variant_count": int(len(summary_df)),
        "best_variant": best,
        "generated_outputs": {
            "pair_csv": str(pair_csv),
            "summary_csv": str(summary_csv),
            "summary_json": str(report_dir / "sensor_assisted_scaling_summary.json"),
            "best_trajectory_png": str(fig_dir / "best_sensor_assisted_trajectory_vs_reference.png"),
            "best_error_png": str(fig_dir / "best_sensor_assisted_error_over_frame.png"),
        },
        "notes": [
            "This is an experimental sensor-assisted scaling sweep.",
            "The best variant is selected using reference only for evaluation, not as estimator input.",
            "Do not treat the best variant as final calibration unless it generalizes across windows.",
            "height_agl_column remains approximate/debug only if used.",
        ],
    }

    summary_json = report_dir / "sensor_assisted_scaling_summary.json"
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(json_safe(run_summary), f, indent=2)

    return run_summary


def main() -> None:
    args = parse_args()

    root = Path.cwd()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = root / config_path

    cfg = load_yaml(config_path)
    raw_dir, output_dir, dataset_name = infer_dirs(cfg, config_path, root)

    sync_path, sync_df = read_sync_csv(output_dir)
    sync_df, sync_time_unit = normalize_sync_time(sync_df)

    imgid_to_path = build_imgid_to_path(sync_df, raw_dir)

    pose_path = find_file(raw_dir, ["OnboardPose.csv"])
    gyro_path = find_file(raw_dir, ["RawGyro.csv", "RawGyroscope.csv", "RawGyroscopeData.csv"])
    baro_path = find_file(raw_dir, ["BarometricPressure.csv", "Barometer.csv"])

    if pose_path is None:
        raise FileNotFoundError("OnboardPose.csv not found")
    if gyro_path is None:
        raise FileNotFoundError("RawGyro/RawGyroscope CSV not found")
    if baro_path is None:
        raise FileNotFoundError("BarometricPressure/Barometer CSV not found")

    pose_df, pose_time_unit = add_time_s(read_csv_expected(pose_path, ONBOARD_POSE_COLUMNS))
    gyro_df, gyro_time_unit = add_time_s(read_csv_expected(gyro_path, RAW_GYRO_COLUMNS))
    baro_df, baro_time_unit = add_time_s(read_csv_expected(baro_path, BAROMETER_COLUMNS))

    focal_px, focal_source = try_load_focal_px(raw_dir, args.focal_px)

    height_modes = [x.strip() for x in args.height_modes.split(",") if x.strip()]
    yaw_modes = [x.strip() for x in args.yaw_modes.split(",") if x.strip()]

    if args.official_runs:
        runs = DEFAULT_RUNS
    else:
        if args.start_imgid is None or args.end_imgid is None or args.stride is None:
            raise RuntimeError(
                "Use --official-runs or provide --start-imgid --end-imgid --stride."
            )
        run_name = args.run_name or f"imgid_{args.start_imgid}_{args.end_imgid}_stride{args.stride}"
        runs = [(run_name, args.start_imgid, args.end_imgid, args.stride)]

    all_summaries = []

    for run_name, start_imgid, end_imgid, stride in runs:
        print(f"Running {run_name}...")
        summary = run_one_window(
            run_name=run_name,
            start_imgid=start_imgid,
            end_imgid=end_imgid,
            stride=stride,
            sync_df=sync_df,
            imgid_to_path=imgid_to_path,
            pose_df=pose_df,
            gyro_df=gyro_df,
            baro_df=baro_df,
            output_dir=output_dir,
            focal_px=focal_px,
            focal_source=focal_source,
            base_height_m=args.base_height_m,
            height_modes=height_modes,
            yaw_modes=yaw_modes,
            max_features=args.max_features,
            ratio=args.ratio,
            ransac_threshold=args.ransac_threshold,
            max_pairs=args.max_pairs,
        )
        all_summaries.append(summary)

    report_dir = output_dir / "reports" / "10b_sensor_assisted_orb_scaling"
    report_dir.mkdir(parents=True, exist_ok=True)

    combined_rows = []
    for run_summary in all_summaries:
        best = run_summary.get("best_variant", {})
        combined_rows.append(
            {
                "run_name": run_summary.get("run_name"),
                "attempted_pairs": run_summary.get("attempted_pairs"),
                "ok_pairs": run_summary.get("ok_pairs"),
                "median_inlier_ratio": run_summary.get("median_inlier_ratio"),
                "best_variant_name": best.get("variant_name"),
                "best_height_mode": best.get("height_mode"),
                "best_yaw_mode": best.get("yaw_mode"),
                "best_swap_xy": best.get("swap_xy"),
                "best_sx": best.get("sx"),
                "best_sy": best.get("sy"),
                "best_estimated_path_m": best.get("estimated_path_m"),
                "best_reference_path_m": best.get("reference_path_m"),
                "best_rmse_m": best.get("rmse_m"),
                "best_final_error_m": best.get("final_error_m"),
                "best_drift_per_100m": best.get("drift_per_100m"),
            }
        )

    combined_df = pd.DataFrame(combined_rows)
    combined_csv = report_dir / "sensor_assisted_scaling_best_variants_all_runs.csv"
    combined_df.to_csv(combined_csv, index=False)

    combined_summary = {
        "block": "10B_sensor_assisted_orb_metric_scaling",
        "dataset_name": dataset_name,
        "config_path": str(config_path),
        "raw_dir": str(raw_dir),
        "output_dir": str(output_dir),
        "sync_path": str(sync_path),
        "image_paths_found": len(imgid_to_path),
        "timestamp_units": {
            "sync": sync_time_unit,
            "pose": pose_time_unit,
            "gyro": gyro_time_unit,
            "barometer": baro_time_unit,
        },
        "focal_px": focal_px,
        "focal_source": focal_source,
        "base_height_m": args.base_height_m,
        "combined_csv": str(combined_csv),
        "runs": all_summaries,
    }

    combined_json = report_dir / "sensor_assisted_scaling_all_runs_summary.json"
    with combined_json.open("w", encoding="utf-8") as f:
        json.dump(json_safe(combined_summary), f, indent=2)

    print("")
    print("Block 10B sensor-assisted ORB scaling generated")
    print("-----------------------------------------------")
    print(f"Dataset:          {dataset_name}")
    print(f"Image paths found:{len(imgid_to_path)}")
    print(f"Focal px:         {focal_px}")
    print(f"Focal source:     {focal_source}")
    print(f"Combined CSV:     {combined_csv}")
    print(f"Combined JSON:    {combined_json}")
    print("")
    print("Best variants:")
    for row in combined_rows:
        print(f"Run: {row['run_name']}")
        print(f"- ok_pairs:       {row['ok_pairs']} / {row['attempted_pairs']}")
        print(f"- best_variant:   {row['best_variant_name']}")
        print(f"- est/ref path:   {row['best_estimated_path_m']} / {row['best_reference_path_m']}")
        print(f"- rmse_m:         {row['best_rmse_m']}")
        print(f"- final_error_m:  {row['best_final_error_m']}")
        print(f"- drift_per_100m: {row['best_drift_per_100m']}")
        print("")


if __name__ == "__main__":
    main()
