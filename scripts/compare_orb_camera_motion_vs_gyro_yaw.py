#!/usr/bin/env python3
"""
Block 10A.7 — Camera motion vs gyro yaw diagnostic on ORB windows.

Comparing:
- image-plane rotation estimated from ORB matches,
- OnboardPose omega_z integrated yaw,
- raw gyro z integrated yaw after sign correction,
- optional quaternion yaw delta from OnboardPose.

Purpose:
Check whether visual frame-to-frame rotation is consistent with inertial yaw.
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
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--start-imgid", type=int, default=None)
    parser.add_argument("--end-imgid", type=int, default=None)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument(
        "--official-runs",
        action="store_true",
        help="Run the four official Week 3 windows.",
    )
    parser.add_argument("--max-features", type=int, default=4000)
    parser.add_argument("--ratio", type=float, default=0.75)
    parser.add_argument("--ransac-threshold", type=float, default=3.0)
    parser.add_argument("--max-pairs", type=int, default=None)
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


def normalize_time_column(sync_df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    if "timestamp" not in sync_df.columns:
        raise RuntimeError("Synchronized frames CSV must contain timestamp column.")

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
    return np.arctan2(siny_cosp, cosy_cosp)


def nearest_value(query_t: np.ndarray, ref_t: np.ndarray, ref_v: np.ndarray) -> np.ndarray:
    ref_t = np.asarray(ref_t, dtype=float)
    ref_v = np.asarray(ref_v, dtype=float)

    valid = np.isfinite(ref_t) & np.isfinite(ref_v)
    ref_t = ref_t[valid]
    ref_v = ref_v[valid]

    order = np.argsort(ref_t)
    ref_t = ref_t[order]
    ref_v = ref_v[order]

    out = np.full(query_t.shape, np.nan, dtype=float)

    if ref_t.size == 0:
        return out

    idx = np.searchsorted(ref_t, query_t)
    left = np.clip(idx - 1, 0, ref_t.size - 1)
    right = np.clip(idx, 0, ref_t.size - 1)

    left_dt = ref_t[left] - query_t
    right_dt = ref_t[right] - query_t

    choose_right = np.abs(right_dt) < np.abs(left_dt)
    best = np.where(choose_right, right, left)

    out = ref_v[best]
    return out


def integrate_rate(t: np.ndarray, rate: np.ndarray, t0: float, t1: float) -> Optional[float]:
    if not np.isfinite(t0) or not np.isfinite(t1):
        return None
    if t1 <= t0:
        return None

    t = np.asarray(t, dtype=float)
    rate = np.asarray(rate, dtype=float)

    valid = np.isfinite(t) & np.isfinite(rate)
    t = t[valid]
    rate = rate[valid]

    if t.size == 0:
        return None

    order = np.argsort(t)
    t = t[order]
    rate = rate[order]

    mask = (t >= t0) & (t <= t1)
    ti = t[mask]
    ri = rate[mask]

    # Add boundary interpolations so short intervals still get a value.
    if t0 < t[0] or t1 > t[-1]:
        return None

    r0 = np.interp(t0, t, rate)
    r1 = np.interp(t1, t, rate)

    ti = np.concatenate([[t0], ti, [t1]])
    ri = np.concatenate([[r0], ri, [r1]])

    if ti.size < 2:
        return None

    return float(np.trapz(ri, ti))


def angle_diff(a: float, b: float) -> Optional[float]:
    if not np.isfinite(a) or not np.isfinite(b):
        return None
    return float(np.arctan2(np.sin(b - a), np.cos(b - a)))


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
    if "imgid" not in sync_df.columns:
        raise RuntimeError("Synchronized CSV must contain imgid column.")

    image_col = find_image_column(sync_df)

    mapping: Dict[int, Path] = {}

    if image_col is not None:
        for _, row in sync_df.iterrows():
            imgid = int(row["imgid"])
            value = row[image_col]
            if pd.isna(value):
                continue
            p = Path(str(value))
            if not p.is_absolute():
                # Try project-relative path first, then raw-dir-relative.
                p1 = Path.cwd() / p
                p2 = raw_dir / p
                if p1.exists():
                    p = p1
                else:
                    p = p2
            mapping[imgid] = p

    # Fallback: search image files and match numeric stem or contained imgid.
    if not mapping:
        image_exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff"]
        image_files: List[Path] = []
        for ext in image_exts:
            image_files.extend(raw_dir.rglob(ext))

        by_number: Dict[int, Path] = {}
        for p in image_files:
            digits = "".join(ch if ch.isdigit() else " " for ch in p.stem).split()
            nums = [int(d) for d in digits if d.isdigit()]
            if not nums:
                continue

            # Prefer the last number because names often include prefixes.
            imgid = nums[-1]
            if imgid not in by_number:
                by_number[imgid] = p

        for imgid in pd.to_numeric(sync_df["imgid"], errors="coerce").dropna().astype(int).tolist():
            if imgid in by_number:
                mapping[imgid] = by_number[imgid]

    return mapping


def load_gray(path: Path) -> Optional[np.ndarray]:
    if path is None or not path.exists():
        return None
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return img


def estimate_orb_pair_rotation(
    img0: np.ndarray,
    img1: np.ndarray,
    max_features: int,
    ratio: float,
    ransac_threshold: float,
) -> Dict[str, Any]:
    orb = cv2.ORB_create(nfeatures=max_features)

    k0, d0 = orb.detectAndCompute(img0, None)
    k1, d1 = orb.detectAndCompute(img1, None)

    result: Dict[str, Any] = {
        "ok": False,
        "num_kp0": 0 if k0 is None else len(k0),
        "num_kp1": 0 if k1 is None else len(k1),
        "num_good_matches": 0,
        "num_inliers": 0,
        "inlier_ratio": np.nan,
        "affine_rotation_rad": np.nan,
        "affine_rotation_deg": np.nan,
        "homography_rotation_rad": np.nan,
        "homography_rotation_deg": np.nan,
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

    H, hmask = cv2.findHomography(pts0, pts1, cv2.RANSAC, ransac_threshold)

    if hmask is not None:
        hmask_bool = hmask.ravel().astype(bool)
    else:
        hmask_bool = np.ones(len(good), dtype=bool)

    inlier_count = int(np.sum(hmask_bool))
    result["num_inliers"] = inlier_count
    result["inlier_ratio"] = float(inlier_count / max(1, len(good)))

    if inlier_count < 8:
        result["reason"] = "not_enough_inliers"
        return result

    pts0_in = pts0[hmask_bool]
    pts1_in = pts1[hmask_bool]

    A, amask = cv2.estimateAffinePartial2D(
        pts0_in,
        pts1_in,
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_threshold,
        maxIters=2000,
        confidence=0.99,
    )

    if A is not None:
        theta = math.atan2(float(A[1, 0]), float(A[0, 0]))
        result["affine_rotation_rad"] = theta
        result["affine_rotation_deg"] = math.degrees(theta)

    if H is not None:
        theta_h = math.atan2(float(H[1, 0]), float(H[0, 0]))
        result["homography_rotation_rad"] = theta_h
        result["homography_rotation_deg"] = math.degrees(theta_h)

    result["ok"] = np.isfinite(result["affine_rotation_rad"]) or np.isfinite(result["homography_rotation_rad"])
    result["reason"] = "ok" if result["ok"] else "rotation_estimate_failed"
    return result


def corr(a: np.ndarray, b: np.ndarray) -> Optional[float]:
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
        value = float(obj)
        return value if np.isfinite(value) else None
    if isinstance(obj, float):
        return obj if np.isfinite(obj) else None
    return obj


def plot_timeseries(df: pd.DataFrame, out_path: Path, title: str) -> None:
    plt.figure(figsize=(11, 5))

    if len(df) == 0:
        plt.text(0.5, 0.5, "No pair data", ha="center", va="center")
        plt.axis("off")
    else:
        x = np.arange(len(df))

        plt.plot(x, df["affine_rotation_rad"], label="ORB affine image rotation", linewidth=0.9)
        plt.plot(x, df["pose_omega_z_delta_rad"], label="pose omega_z integrated yaw", linewidth=0.9)
        plt.plot(x, df["raw_gyro_z_sign_corrected_delta_rad"], label="-raw gyro_z integrated yaw", linewidth=0.9)
        plt.plot(x, df["quat_yaw_delta_rad"], label="quaternion yaw delta", linewidth=0.9)

        plt.xlabel("Pair index")
        plt.ylabel("Rotation / yaw delta [rad]")
        plt.title(title)
        plt.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_scatter(df: pd.DataFrame, out_path: Path, title: str) -> None:
    plt.figure(figsize=(6, 6))

    if len(df) == 0:
        plt.text(0.5, 0.5, "No pair data", ha="center", va="center")
        plt.axis("off")
    else:
        x = df["affine_rotation_rad"].to_numpy(dtype=float)
        y = df["pose_omega_z_delta_rad"].to_numpy(dtype=float)

        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask]
        y = y[mask]

        plt.scatter(x, y, s=10, alpha=0.5)

        if x.size:
            lim = float(max(np.max(np.abs(x)), np.max(np.abs(y)), 1e-6))
            plt.plot([-lim, lim], [-lim, lim], linestyle="--", linewidth=0.8)
            plt.plot([-lim, lim], [lim, -lim], linestyle=":", linewidth=0.8)
            plt.xlim(-lim, lim)
            plt.ylim(-lim, lim)

        plt.xlabel("ORB affine image rotation [rad]")
        plt.ylabel("Pose omega_z integrated yaw [rad]")
        plt.title(title)

    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_correlation_bars(corr_rows: List[Dict[str, Any]], out_path: Path, title: str) -> None:
    df = pd.DataFrame(corr_rows)
    plt.figure(figsize=(10, 5))

    if len(df) == 0:
        plt.text(0.5, 0.5, "No correlations", ha="center", va="center")
        plt.axis("off")
    else:
        labels = df["comparison"].tolist()
        vals = df["corr"].to_numpy(dtype=float)

        plt.bar(labels, vals)
        plt.xticks(rotation=35, ha="right")
        plt.ylim(-1.05, 1.05)
        plt.ylabel("Correlation")
        plt.title(title)

    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def run_one_window(
    run_name: str,
    start_imgid: int,
    end_imgid: int,
    stride: int,
    sync_df: pd.DataFrame,
    imgid_to_path: Dict[int, Path],
    pose_df: pd.DataFrame,
    gyro_df: pd.DataFrame,
    output_dir: Path,
    max_features: int,
    ratio: float,
    ransac_threshold: float,
    max_pairs: Optional[int],
) -> Dict[str, Any]:
    metadata_dir = output_dir / "metadata" / "10a7_camera_motion_vs_gyro_yaw" / run_name
    report_dir = output_dir / "reports" / "10a7_camera_motion_vs_gyro_yaw" / run_name
    fig_dir = output_dir / "figures" / "10a7_camera_motion_vs_gyro_yaw" / run_name

    metadata_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    window = sync_df[
        (pd.to_numeric(sync_df["imgid"], errors="coerce") >= start_imgid)
        & (pd.to_numeric(sync_df["imgid"], errors="coerce") <= end_imgid)
    ].copy()

    window = window.sort_values("imgid").reset_index(drop=True)
    window = window.iloc[::stride].copy().reset_index(drop=True)

    if max_pairs is not None and len(window) > max_pairs + 1:
        window = window.iloc[: max_pairs + 1].copy().reset_index(drop=True)

    pose_time = pose_df["time_s"].to_numpy(dtype=float)
    pose_omega_z = pose_df["omega_z"].to_numpy(dtype=float)

    raw_time = gyro_df["time_s"].to_numpy(dtype=float)
    raw_gyro_z = gyro_df["z"].to_numpy(dtype=float)

    q_yaw = quaternion_to_yaw(
        pose_df["attitude_w"].to_numpy(dtype=float),
        pose_df["attitude_x"].to_numpy(dtype=float),
        pose_df["attitude_y"].to_numpy(dtype=float),
        pose_df["attitude_z"].to_numpy(dtype=float),
    )
    q_yaw_unwrapped = np.unwrap(q_yaw)

    rows: List[Dict[str, Any]] = []

    for i in range(len(window) - 1):
        row0 = window.iloc[i]
        row1 = window.iloc[i + 1]

        imgid0 = int(row0["imgid"])
        imgid1 = int(row1["imgid"])
        t0 = float(row0["time_s"])
        t1 = float(row1["time_s"])

        p0 = imgid_to_path.get(imgid0)
        p1 = imgid_to_path.get(imgid1)

        img0 = load_gray(p0) if p0 is not None else None
        img1 = load_gray(p1) if p1 is not None else None

        pair: Dict[str, Any] = {
            "run_name": run_name,
            "pair_index": i,
            "imgid0": imgid0,
            "imgid1": imgid1,
            "timestamp0_s": t0,
            "timestamp1_s": t1,
            "dt_s": t1 - t0,
            "image0_path": str(p0) if p0 is not None else None,
            "image1_path": str(p1) if p1 is not None else None,
        }

        if img0 is None or img1 is None:
            pair.update(
                {
                    "ok": False,
                    "reason": "image_missing",
                    "affine_rotation_rad": np.nan,
                    "affine_rotation_deg": np.nan,
                    "homography_rotation_rad": np.nan,
                    "homography_rotation_deg": np.nan,
                    "num_good_matches": 0,
                    "num_inliers": 0,
                    "inlier_ratio": np.nan,
                }
            )
        else:
            orb_result = estimate_orb_pair_rotation(
                img0=img0,
                img1=img1,
                max_features=max_features,
                ratio=ratio,
                ransac_threshold=ransac_threshold,
            )
            pair.update(orb_result)

        pose_delta = integrate_rate(pose_time, pose_omega_z, t0, t1)

        # From previous diagnostics: pose_omega_z ≈ -raw_gyro_z.
        raw_delta_corrected = integrate_rate(raw_time, -raw_gyro_z, t0, t1)
        raw_delta_uncorrected = integrate_rate(raw_time, raw_gyro_z, t0, t1)

        q0 = nearest_value(np.array([t0]), pose_time, q_yaw_unwrapped)[0]
        q1 = nearest_value(np.array([t1]), pose_time, q_yaw_unwrapped)[0]
        q_delta = float(q1 - q0) if np.isfinite(q0) and np.isfinite(q1) else np.nan

        pair["pose_omega_z_delta_rad"] = pose_delta if pose_delta is not None else np.nan
        pair["raw_gyro_z_sign_corrected_delta_rad"] = (
            raw_delta_corrected if raw_delta_corrected is not None else np.nan
        )
        pair["raw_gyro_z_uncorrected_delta_rad"] = (
            raw_delta_uncorrected if raw_delta_uncorrected is not None else np.nan
        )
        pair["quat_yaw_delta_rad"] = q_delta

        rows.append(pair)

    pair_df = pd.DataFrame(rows)

    pair_csv = metadata_dir / "camera_motion_vs_gyro_yaw_pairs.csv"
    pair_df.to_csv(pair_csv, index=False)

    ok_df = pair_df[pair_df["ok"] == True].copy()

    comparisons = [
        ("orb_affine_vs_pose_omega_z", "affine_rotation_rad", "pose_omega_z_delta_rad"),
        ("orb_affine_vs_neg_pose_omega_z", "affine_rotation_rad", "pose_omega_z_delta_rad_neg"),
        ("orb_affine_vs_raw_gyro_z_corrected", "affine_rotation_rad", "raw_gyro_z_sign_corrected_delta_rad"),
        ("orb_affine_vs_neg_raw_gyro_z_corrected", "affine_rotation_rad", "raw_gyro_z_sign_corrected_delta_rad_neg"),
        ("orb_affine_vs_quat_yaw", "affine_rotation_rad", "quat_yaw_delta_rad"),
        ("orb_affine_vs_neg_quat_yaw", "affine_rotation_rad", "quat_yaw_delta_rad_neg"),
    ]

    ok_df["pose_omega_z_delta_rad_neg"] = -ok_df["pose_omega_z_delta_rad"]
    ok_df["raw_gyro_z_sign_corrected_delta_rad_neg"] = -ok_df["raw_gyro_z_sign_corrected_delta_rad"]
    ok_df["quat_yaw_delta_rad_neg"] = -ok_df["quat_yaw_delta_rad"]

    corr_rows: List[Dict[str, Any]] = []
    for name, a_col, b_col in comparisons:
        c = corr(ok_df[a_col].to_numpy(dtype=float), ok_df[b_col].to_numpy(dtype=float))
        corr_rows.append(
            {
                "comparison": name,
                "corr": c,
                "abs_corr": abs(c) if c is not None else None,
            }
        )

    corr_df = pd.DataFrame(corr_rows).sort_values("abs_corr", ascending=False, na_position="last")
    corr_csv = metadata_dir / "camera_motion_vs_gyro_yaw_correlations.csv"
    corr_df.to_csv(corr_csv, index=False)

    plot_timeseries(
        ok_df,
        fig_dir / "camera_rotation_vs_gyro_yaw_timeseries.png",
        f"{run_name}: ORB image rotation vs gyro/pose yaw",
    )
    plot_scatter(
        ok_df,
        fig_dir / "orb_rotation_vs_pose_yaw_scatter.png",
        f"{run_name}: ORB image rotation vs pose yaw",
    )
    plot_correlation_bars(
        corr_rows,
        fig_dir / "camera_motion_vs_gyro_yaw_correlations.png",
        f"{run_name}: correlation signs",
    )

    best = corr_df.iloc[0].to_dict() if len(corr_df) else {}

    if best.get("abs_corr") is not None and best["abs_corr"] >= 0.75:
        decision = "strong_visual_inertial_yaw_relation"
    elif best.get("abs_corr") is not None and best["abs_corr"] >= 0.45:
        decision = "moderate_visual_inertial_yaw_relation"
    else:
        decision = "weak_or_unclear_visual_inertial_yaw_relation"

    summary = {
        "run_name": run_name,
        "start_imgid": start_imgid,
        "end_imgid": end_imgid,
        "stride": stride,
        "frames_in_window_after_stride": int(len(window)),
        "attempted_pairs": int(len(pair_df)),
        "ok_pairs": int(ok_df["ok"].sum()) if len(ok_df) else 0,
        "failed_pairs": int(len(pair_df) - len(ok_df)),
        "median_inlier_ratio": float(np.nanmedian(ok_df["inlier_ratio"])) if len(ok_df) else None,
        "affine_rotation_rad_summary": stats(ok_df["affine_rotation_rad"].to_numpy(dtype=float)) if len(ok_df) else {},
        "pose_omega_z_delta_rad_summary": stats(ok_df["pose_omega_z_delta_rad"].to_numpy(dtype=float)) if len(ok_df) else {},
        "raw_gyro_z_sign_corrected_delta_rad_summary": stats(ok_df["raw_gyro_z_sign_corrected_delta_rad"].to_numpy(dtype=float)) if len(ok_df) else {},
        "quat_yaw_delta_rad_summary": stats(ok_df["quat_yaw_delta_rad"].to_numpy(dtype=float)) if len(ok_df) else {},
        "best_comparison": best,
        "decision": decision,
        "generated_outputs": {
            "pair_csv": str(pair_csv),
            "correlation_csv": str(corr_csv),
            "summary_json": str(report_dir / "camera_motion_vs_gyro_yaw_summary.json"),
            "timeseries_png": str(fig_dir / "camera_rotation_vs_gyro_yaw_timeseries.png"),
            "scatter_png": str(fig_dir / "orb_rotation_vs_pose_yaw_scatter.png"),
            "correlation_png": str(fig_dir / "camera_motion_vs_gyro_yaw_correlations.png"),
        },
        "notes": [
            "ORB image rotation is image-plane rotation, not pure drone yaw.",
            "For a camera looking at the world, image rotation can have opposite sign to drone yaw depending on camera frame convention.",
            "Raw gyro z is sign-corrected using previous diagnostics: corrected yaw-rate = -raw_gyro_z.",
            "A weak correlation does not mean gyro is bad; it can also mean translation/perspective dominates the image transform.",
        ],
    }

    summary_json = report_dir / "camera_motion_vs_gyro_yaw_summary.json"
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(json_safe(summary), f, indent=2)

    return summary


def main() -> None:
    args = parse_args()

    root = Path.cwd()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = root / config_path

    cfg = load_yaml(config_path)
    raw_dir, output_dir, dataset_name = infer_dirs(cfg, config_path, root)

    sync_path, sync_df = read_sync_csv(output_dir)
    sync_df, sync_time_unit = normalize_time_column(sync_df)

    if "imgid" not in sync_df.columns:
        raise RuntimeError("Synchronized frames CSV must contain imgid column.")

    imgid_to_path = build_imgid_to_path(sync_df, raw_dir)

    gyro_path = find_file(raw_dir, ["RawGyro.csv", "RawGyroscope.csv", "RawGyroscopeData.csv"])
    pose_path = find_file(raw_dir, ["OnboardPose.csv"])

    if gyro_path is None:
        raise FileNotFoundError(f"RawGyro/RawGyroscope CSV not found inside {raw_dir}")
    if pose_path is None:
        raise FileNotFoundError(f"OnboardPose.csv not found inside {raw_dir}")

    gyro_df, gyro_time_unit = add_time_s(read_csv_expected(gyro_path, RAW_GYRO_COLUMNS))
    pose_df, pose_time_unit = add_time_s(read_csv_expected(pose_path, ONBOARD_POSE_COLUMNS))

    if args.official_runs:
        runs = DEFAULT_RUNS
    else:
        if args.start_imgid is None or args.end_imgid is None or args.stride is None:
            raise RuntimeError(
                "Provide --official-runs or provide --start-imgid, --end-imgid, and --stride."
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
            output_dir=output_dir,
            max_features=args.max_features,
            ratio=args.ratio,
            ransac_threshold=args.ransac_threshold,
            max_pairs=args.max_pairs,
        )
        all_summaries.append(summary)

    report_dir = output_dir / "reports" / "10a7_camera_motion_vs_gyro_yaw"
    report_dir.mkdir(parents=True, exist_ok=True)

    combined_summary = {
        "block": "10A7_camera_motion_vs_gyro_yaw",
        "dataset_name": dataset_name,
        "config_path": str(config_path),
        "raw_dir": str(raw_dir),
        "output_dir": str(output_dir),
        "sync_path": str(sync_path),
        "gyro_path": str(gyro_path),
        "pose_path": str(pose_path),
        "timestamp_units": {
            "sync": sync_time_unit,
            "gyro": gyro_time_unit,
            "pose": pose_time_unit,
        },
        "runs": all_summaries,
        "important_interpretation": [
            "This diagnostic checks consistency between camera image rotation and inertial yaw signals.",
            "It is not yet an EKF/VIO fusion result.",
            "Image-plane rotation may be opposite sign to drone yaw depending on camera coordinate convention.",
            "If correlation is moderate/strong, gyro yaw can be tested as an ORB stabilization cue.",
        ],
    }

    combined_json = report_dir / "camera_motion_vs_gyro_yaw_all_runs_summary.json"
    with combined_json.open("w", encoding="utf-8") as f:
        json.dump(json_safe(combined_summary), f, indent=2)

    print("")
    print("Block 10A.7 camera motion vs gyro yaw diagnostics generated")
    print("----------------------------------------------------------")
    print(f"Dataset:        {dataset_name}")
    print(f"Sync file:      {sync_path}")
    print(f"Image paths found: {len(imgid_to_path)}")
    print(f"Combined summary: {combined_json}")
    print("")
    for summary in all_summaries:
        best = summary.get("best_comparison", {})
        print(f"Run: {summary['run_name']}")
        print(f"- attempted_pairs:     {summary['attempted_pairs']}")
        print(f"- ok_pairs:            {summary['ok_pairs']}")
        print(f"- median_inlier_ratio: {summary['median_inlier_ratio']}")
        print(f"- best_comparison:     {best.get('comparison')}")
        print(f"- best_corr:           {best.get('corr')}")
        print(f"- decision:            {summary['decision']}")
        print("")


if __name__ == "__main__":
    main()
