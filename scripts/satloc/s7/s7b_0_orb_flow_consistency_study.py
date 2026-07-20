#!/usr/bin/env python3
"""S7B.0 — ORB + Optical-Flow Relative Motion Consistency Study.

Purpose:
- Keep ORB as the frozen relative baseline after S7A.2 KEEP_ORB.
- Add camera-only sparse Lucas-Kanade optical flow as a consistency and smoothing layer.
- Compare ORB, XFeat, KLT-flow-only, and ORB+flow guarded fusion on the same traj01 chain.
- Use no sensor logs. Optical flow is computed directly from consecutive UAV images.

Bounded design:
- one KLT configuration only;
- all 1,033 consecutive traj01 pairs;
- same 960-pixel long side and partial-affine RANSAC geometry;
- same SE(2) scale-normalized trajectory integration;
- same 50-frame prefix evaluation;
- reference ENU only after trajectories are built.

The study does not replace the absolute retrieval benchmark and does not start a
new optical-flow parameter sweep.

Command Used:

export PYTHONPATH=$PWD/src

mkdir -p outputs/satloc/reports/s7_relative_frontend
set -o pipefail

python scripts/satloc/s7/s7b_0_orb_flow_consistency_study.py \
  2>&1 | tee \
  outputs/satloc/reports/s7_relative_frontend/s7b_0_orb_flow_consistency_study.log

"""

from __future__ import annotations

import argparse
import json
import math
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EXPECTED_FRAMES = 1034
EXPECTED_PAIRS = 1033
ALIGNMENT_PREFIX_FRAMES = 50
ERROR_THRESHOLDS_M = [10.0, 20.0, 40.0, 80.0]
SUSTAIN_FRAMES = 5

DEFAULT_OUTPUT_ROOT = Path("outputs/satloc")
DEFAULT_SEQUENCE_MANIFEST = Path(
    "outputs/satloc/metadata/s6a_relative_motion/s6a_sequence_manifest.csv"
)
DEFAULT_ORB_PAIRS = Path(
    "outputs/satloc/metadata/s6a_relative_motion/"
    "s6a1_orb_affine_pair_diagnostics.csv"
)
DEFAULT_ORB_ALIGNED = Path(
    "outputs/satloc/metadata/s6a_relative_motion/"
    "s6a2_orb_relative_trajectory_aligned_eval_only.csv"
)
DEFAULT_ORB_SUMMARY = Path(
    "outputs/satloc/metadata/s6a_relative_motion/"
    "s6a2_orb_relative_trajectory_summary.csv"
)
DEFAULT_ORB_RUNTIME = Path(
    "outputs/satloc/reports/s6a_relative_motion/"
    "s6a1_orb_affine_stride_summary.json"
)
DEFAULT_FULL_PAIR_MANIFEST = Path(
    "outputs/satloc/metadata/s7_relative_frontend/"
    "s7a_0_full_pair_manifest.csv"
)
DEFAULT_XFEAT_COMPARISON = Path(
    "outputs/satloc/metadata/s7_relative_frontend/"
    "s7a_2_full_method_comparison.csv"
)
DEFAULT_XFEAT_ALIGNED = Path(
    "outputs/satloc/metadata/s7_relative_frontend/"
    "s7a_2_xfeat_relative_trajectory_aligned_eval_only.csv"
)
DEFAULT_XFEAT_DECISION = Path(
    "outputs/satloc/metadata/s7_relative_frontend/"
    "s7a_2_promotion_decision.json"
)


def utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_from_repo(repo_root: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def git_value(repo: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return ""


def peak_process_memory_mb() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if platform.system() == "Darwin":
        return value / (1024.0 * 1024.0)
    return value / 1024.0


def bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes", "y", "t"})
    )


def finite_stat(values: pd.Series | np.ndarray, function, default: float = float("nan")) -> float:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return default
    return float(function(array))


def wrap_angle_deg(angle: float) -> float:
    return float((angle + 180.0) % 360.0 - 180.0)


def resize_long_side(image: np.ndarray, target_long_side: int) -> np.ndarray:
    height, width = image.shape[:2]
    long_side = max(height, width)
    if target_long_side <= 0 or long_side == target_long_side:
        return image
    scale = target_long_side / float(long_side)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(image, (new_width, new_height), interpolation=interpolation)


def resolve_image_path(repo_root: Path, value: Any) -> Path:
    raw = Path(str(value)).expanduser()
    for candidate in (raw, repo_root / raw):
        if candidate.exists():
            return candidate.resolve()
    return raw


def load_sequence_manifest(repo_root: Path, path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"sequence_frame_id", "token0_id", "x_enu_m", "y_enu_m"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise RuntimeError(f"Sequence manifest missing columns: {missing}")

    image_column = None
    for candidate in ("image_path_resolved", "image_path", "resolved_image_path"):
        if candidate in frame.columns:
            image_column = candidate
            break
    if image_column is None:
        raise RuntimeError("Sequence manifest has no usable image-path column.")

    frame["sequence_frame_id"] = pd.to_numeric(frame["sequence_frame_id"], errors="raise").astype(int)
    frame["token0_id"] = pd.to_numeric(frame["token0_id"], errors="raise").astype(int)
    frame["x_enu_m"] = pd.to_numeric(frame["x_enu_m"], errors="raise")
    frame["y_enu_m"] = pd.to_numeric(frame["y_enu_m"], errors="raise")
    frame = frame.sort_values("sequence_frame_id", kind="mergesort").reset_index(drop=True)

    if len(frame) != EXPECTED_FRAMES:
        raise RuntimeError(f"Expected {EXPECTED_FRAMES} frames, found {len(frame)}.")
    if not np.array_equal(frame["sequence_frame_id"].to_numpy(dtype=int), np.arange(EXPECTED_FRAMES)):
        raise RuntimeError("sequence_frame_id is not contiguous 0..1033.")
    if not np.array_equal(frame["token0_id"].to_numpy(dtype=int), np.arange(1, EXPECTED_FRAMES + 1)):
        raise RuntimeError("token0_id is not the canonical range 1..1034.")

    frame["image_path_flow_resolved"] = frame[image_column].map(
        lambda value: str(resolve_image_path(repo_root, value))
    )
    missing_paths = [
        Path(value) for value in frame["image_path_flow_resolved"] if not Path(value).exists()
    ]
    if missing_paths:
        raise FileNotFoundError(
            f"{len(missing_paths)} UAV images are missing. First missing: {missing_paths[0]}"
        )
    return frame


def load_orb_pairs(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {
        "stride",
        "pair_number",
        "frame_index_a",
        "frame_index_b",
        "token0_a",
        "token0_b",
        "status",
        "affine_ok",
        "good_matches",
        "inliers",
        "inlier_ratio",
        "affine_a00",
        "affine_a01",
        "affine_a10",
        "affine_a11",
        "affine_tx_px",
        "affine_ty_px",
        "affine_scale",
        "affine_rotation_deg",
        "elapsed_ms",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise RuntimeError(f"ORB pair CSV missing columns: {missing}")

    numeric = [
        "stride",
        "pair_number",
        "frame_index_a",
        "frame_index_b",
        "token0_a",
        "token0_b",
        "good_matches",
        "inliers",
        "inlier_ratio",
        "affine_a00",
        "affine_a01",
        "affine_a10",
        "affine_a11",
        "affine_tx_px",
        "affine_ty_px",
        "affine_scale",
        "affine_rotation_deg",
        "elapsed_ms",
    ]
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="raise")

    frame["affine_ok"] = bool_series(frame["affine_ok"])
    frame = (
        frame[frame["stride"] == 1]
        .copy()
        .sort_values("frame_index_a", kind="mergesort")
        .reset_index(drop=True)
    )

    if len(frame) != EXPECTED_PAIRS:
        raise RuntimeError(f"Expected {EXPECTED_PAIRS} ORB stride-1 pairs, found {len(frame)}.")
    return frame


def load_pair_manifest(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {
        "comparison_pair_id",
        "pair_number",
        "frame_index_a",
        "frame_index_b",
        "token0_a",
        "token0_b",
        "in_stratified_diagnostic_subset",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise RuntimeError(f"Full pair manifest missing columns: {missing}")

    for column in [
        "comparison_pair_id",
        "pair_number",
        "frame_index_a",
        "frame_index_b",
        "token0_a",
        "token0_b",
    ]:
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype(int)

    frame["in_stratified_diagnostic_subset"] = bool_series(
        frame["in_stratified_diagnostic_subset"]
    )
    frame = frame.sort_values("comparison_pair_id", kind="mergesort").reset_index(drop=True)

    if len(frame) != EXPECTED_PAIRS:
        raise RuntimeError(f"Expected {EXPECTED_PAIRS} pair-manifest rows, found {len(frame)}.")

    # Fill non-diagnostic labels so the full pair table stays easy to group.
    for column in ["primary_scene", "secondary_scene", "selection_role", "range_id"]:
        if column not in frame.columns:
            frame[column] = ""
        frame[column] = frame[column].fillna("").astype(str)

    return frame


def read_gray_resized(path: Path, resize_long: int) -> np.ndarray:
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise RuntimeError(f"OpenCV could not read: {path}")
    return resize_long_side(gray, resize_long)


def affine_metrics_from_points(
    points_a: np.ndarray,
    points_b: np.ndarray,
    width: int,
    height: int,
    ransac_threshold: float,
) -> dict[str, Any]:
    if len(points_a) < 3:
        return {
            "status": "too_few_tracks",
            "affine_ok": False,
            "inliers": 0,
            "inlier_ratio": 0.0,
        }

    affine, inlier_mask = cv2.estimateAffinePartial2D(
        points_a.astype(np.float32),
        points_b.astype(np.float32),
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_threshold,
        maxIters=3000,
        confidence=0.995,
        refineIters=10,
    )

    if affine is None or inlier_mask is None:
        return {
            "status": "affine_failed",
            "affine_ok": False,
            "inliers": 0,
            "inlier_ratio": 0.0,
        }

    mask = np.asarray(inlier_mask).ravel().astype(bool)
    inliers = int(mask.sum())
    inlier_ratio = inliers / max(len(points_a), 1)

    a00, a01, tx = [float(value) for value in affine[0]]
    a10, a11, ty = [float(value) for value in affine[1]]
    scale_x = math.hypot(a00, a10)
    scale_y = math.hypot(a01, a11)
    scale = 0.5 * (scale_x + scale_y)
    rotation_deg = math.degrees(math.atan2(a10, a00))

    center = np.array([width / 2.0, height / 2.0, 1.0], dtype=float)
    mapped = affine @ center
    center_dx = float(mapped[0] - center[0])
    center_dy = float(mapped[1] - center[1])

    good = (
        len(points_a) >= 30
        and inliers >= 20
        and inlier_ratio >= 0.35
        and 0.70 <= scale <= 1.40
    )

    return {
        "status": "good" if good else "weak",
        "affine_ok": True,
        "inliers": inliers,
        "inlier_ratio": float(inlier_ratio),
        "affine_a00": a00,
        "affine_a01": a01,
        "affine_a10": a10,
        "affine_a11": a11,
        "affine_tx_px": tx,
        "affine_ty_px": ty,
        "affine_scale": float(scale),
        "affine_rotation_deg": float(rotation_deg),
        "center_content_dx_px": center_dx,
        "center_content_dy_px": center_dy,
        "center_content_motion_px": float(math.hypot(center_dx, center_dy)),
    }


def compute_klt_pair(
    gray_a: np.ndarray,
    gray_b: np.ndarray,
    max_corners: int,
    quality_level: float,
    min_distance: float,
    block_size: int,
    lk_win_size: int,
    lk_max_level: int,
    fb_error_thresh: float,
    ransac_threshold: float,
) -> dict[str, Any]:
    pair_start = time.perf_counter()

    corners = cv2.goodFeaturesToTrack(
        gray_a,
        maxCorners=max_corners,
        qualityLevel=quality_level,
        minDistance=min_distance,
        blockSize=block_size,
        useHarrisDetector=False,
    )

    if corners is None or len(corners) < 3:
        return {
            "status": "no_corners",
            "corners_detected": 0 if corners is None else int(len(corners)),
            "lk_tracks": 0,
            "fb_good_tracks": 0,
            "fb_error_median_px": float("nan"),
            "affine_ok": False,
            "inliers": 0,
            "inlier_ratio": 0.0,
            "elapsed_ms": (time.perf_counter() - pair_start) * 1000.0,
        }

    p0 = corners.reshape(-1, 1, 2).astype(np.float32)
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        30,
        0.01,
    )
    p1, st1, _ = cv2.calcOpticalFlowPyrLK(
        gray_a,
        gray_b,
        p0,
        None,
        winSize=(lk_win_size, lk_win_size),
        maxLevel=lk_max_level,
        criteria=criteria,
    )
    if p1 is None or st1 is None:
        return {
            "status": "lk_failed",
            "corners_detected": int(len(p0)),
            "lk_tracks": 0,
            "fb_good_tracks": 0,
            "fb_error_median_px": float("nan"),
            "affine_ok": False,
            "inliers": 0,
            "inlier_ratio": 0.0,
            "elapsed_ms": (time.perf_counter() - pair_start) * 1000.0,
        }

    p0_back, st2, _ = cv2.calcOpticalFlowPyrLK(
        gray_b,
        gray_a,
        p1,
        None,
        winSize=(lk_win_size, lk_win_size),
        maxLevel=lk_max_level,
        criteria=criteria,
    )

    st1 = st1.reshape(-1).astype(bool)
    st2_valid = np.zeros_like(st1, dtype=bool) if st2 is None else st2.reshape(-1).astype(bool)
    p0_flat = p0.reshape(-1, 2)
    p1_flat = p1.reshape(-1, 2)
    p0_back_flat = p0_back.reshape(-1, 2) if p0_back is not None else np.full_like(p0_flat, np.nan)

    fb_error = np.linalg.norm(p0_flat - p0_back_flat, axis=1)
    valid = (
        st1
        & st2_valid
        & np.isfinite(p1_flat).all(axis=1)
        & np.isfinite(p0_back_flat).all(axis=1)
        & (fb_error <= fb_error_thresh)
    )

    tracked = int(st1.sum())
    points_a = p0_flat[valid]
    points_b = p1_flat[valid]

    metrics = affine_metrics_from_points(
        points_a,
        points_b,
        width=int(gray_a.shape[1]),
        height=int(gray_a.shape[0]),
        ransac_threshold=ransac_threshold,
    )
    elapsed_ms = (time.perf_counter() - pair_start) * 1000.0

    return {
        "corners_detected": int(len(p0)),
        "lk_tracks": tracked,
        "fb_good_tracks": int(valid.sum()),
        "fb_error_median_px": finite_stat(fb_error[valid], np.median),
        **metrics,
        "elapsed_ms": float(elapsed_ms),
    }


def run_klt_flow(
    sequence: pd.DataFrame,
    pair_manifest: pd.DataFrame,
    resize_long: int,
    max_corners: int,
    quality_level: float,
    min_distance: float,
    block_size: int,
    lk_win_size: int,
    lk_max_level: int,
    fb_error_thresh: float,
    ransac_threshold: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    wall_start = time.perf_counter()

    gray_prev = read_gray_resized(Path(sequence.iloc[0]["image_path_flow_resolved"]), resize_long)
    width = int(gray_prev.shape[1])
    height = int(gray_prev.shape[0])

    for frame_index in range(1, len(sequence)):
        gray_current = read_gray_resized(
            Path(sequence.iloc[frame_index]["image_path_flow_resolved"]),
            resize_long,
        )
        if gray_current.shape[:2] != gray_prev.shape[:2]:
            raise RuntimeError("Resized image dimensions changed during the sequence.")

        pair = pair_manifest.iloc[frame_index - 1]
        metrics = compute_klt_pair(
            gray_prev,
            gray_current,
            max_corners=max_corners,
            quality_level=quality_level,
            min_distance=min_distance,
            block_size=block_size,
            lk_win_size=lk_win_size,
            lk_max_level=lk_max_level,
            fb_error_thresh=fb_error_thresh,
            ransac_threshold=ransac_threshold,
        )

        rows.append(
            {
                "comparison_pair_id": int(pair["comparison_pair_id"]),
                "pair_number": int(pair["pair_number"]),
                "frame_index_a": int(pair["frame_index_a"]),
                "frame_index_b": int(pair["frame_index_b"]),
                "token0_a": int(pair["token0_a"]),
                "token0_b": int(pair["token0_b"]),
                "in_stratified_diagnostic_subset": bool(pair["in_stratified_diagnostic_subset"]),
                "primary_scene": str(pair.get("primary_scene", "")),
                "secondary_scene": str(pair.get("secondary_scene", "")),
                "selection_role": str(pair.get("selection_role", "")),
                "range_id": str(pair.get("range_id", "")),
                "width": width,
                "height": height,
                **metrics,
            }
        )

        gray_prev = gray_current

        if frame_index % 100 == 0 or frame_index == len(sequence) - 1:
            print(
                f"KLT flow full chain: frame {frame_index + 1}/{len(sequence)}, "
                f"pairs {frame_index}/{EXPECTED_PAIRS}"
            )

    pair_df = pd.DataFrame(rows)
    pair_df.attrs["wall_seconds"] = time.perf_counter() - wall_start
    return pair_df


def affine_from_row(row: pd.Series, prefix: str = "") -> np.ndarray:
    return np.array(
        [
            [
                float(row[f"{prefix}affine_a00"]),
                float(row[f"{prefix}affine_a01"]),
                float(row[f"{prefix}affine_tx_px"]),
            ],
            [
                float(row[f"{prefix}affine_a10"]),
                float(row[f"{prefix}affine_a11"]),
                float(row[f"{prefix}affine_ty_px"]),
            ],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def normalize_affine_scale_about_center(matrix: np.ndarray, center: np.ndarray) -> np.ndarray:
    linear = matrix[:2, :2]
    determinant = float(np.linalg.det(linear))
    scale = math.sqrt(abs(determinant))
    if not np.isfinite(scale) or scale <= 1e-12:
        return matrix.copy()

    rotation = linear / scale
    u, _, vt = np.linalg.svd(rotation)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1.0
        rotation = u @ vt

    mapped_center = matrix @ center
    translation = mapped_center[:2] - rotation @ center[:2]

    normalized = np.eye(3, dtype=float)
    normalized[:2, :2] = rotation
    normalized[:2, 2] = translation
    return normalized


def local_camera_step_from_scene_affine(scene_affine: np.ndarray, center: np.ndarray) -> np.ndarray:
    inverse = np.linalg.inv(scene_affine)
    camera_b_center_in_a = inverse @ center
    camera_b_center_in_a /= camera_b_center_in_a[2]
    step_image = camera_b_center_in_a[:2] - center[:2]
    return np.array([step_image[0], -step_image[1]], dtype=float)


def rotation_matrix(angle_rad: float) -> np.ndarray:
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return np.array([[c, -s], [s, c]], dtype=float)


def step_yaw_from_pair(row: pd.Series, prefix: str = "") -> tuple[np.ndarray, float]:
    width = int(row.get("width", row.get("orb_width", 0)))
    height = int(row.get("height", row.get("orb_height", 0)))
    if width <= 0 or height <= 0:
        # S6A ORB features used a constant 960 long-side size; when not stored in
        # the ORB pair row, use the flow row dimensions after merging.
        width = int(row["flow_width"]) if "flow_width" in row else 960
        height = int(row["flow_height"]) if "flow_height" in row else 540

    center = np.array([width / 2.0, height / 2.0, 1.0], dtype=float)
    matrix = normalize_affine_scale_about_center(
        affine_from_row(row, prefix=prefix),
        center,
    )
    step = local_camera_step_from_scene_affine(matrix, center)
    yaw_deg = float(row[f"{prefix}affine_rotation_deg"])
    return step, yaw_deg


def build_consistency_table(
    orb_pairs: pd.DataFrame,
    flow_pairs: pd.DataFrame,
) -> pd.DataFrame:
    orb = orb_pairs.copy()
    orb = orb.rename(
        columns={
            "status": "orb_status",
            "affine_ok": "orb_affine_ok",
            "good_matches": "orb_good_matches",
            "inliers": "orb_inliers",
            "inlier_ratio": "orb_inlier_ratio",
            "affine_a00": "orb_affine_a00",
            "affine_a01": "orb_affine_a01",
            "affine_a10": "orb_affine_a10",
            "affine_a11": "orb_affine_a11",
            "affine_tx_px": "orb_affine_tx_px",
            "affine_ty_px": "orb_affine_ty_px",
            "affine_scale": "orb_affine_scale",
            "affine_rotation_deg": "orb_affine_rotation_deg",
            "center_content_dx_px": "orb_center_content_dx_px",
            "center_content_dy_px": "orb_center_content_dy_px",
            "center_content_motion_px": "orb_center_content_motion_px",
            "elapsed_ms": "orb_elapsed_ms",
        }
    )

    flow = flow_pairs.copy()
    flow = flow.rename(
        columns={
            "status": "flow_status",
            "affine_ok": "flow_affine_ok",
            "inliers": "flow_inliers",
            "inlier_ratio": "flow_inlier_ratio",
            "affine_a00": "flow_affine_a00",
            "affine_a01": "flow_affine_a01",
            "affine_a10": "flow_affine_a10",
            "affine_a11": "flow_affine_a11",
            "affine_tx_px": "flow_affine_tx_px",
            "affine_ty_px": "flow_affine_ty_px",
            "affine_scale": "flow_affine_scale",
            "affine_rotation_deg": "flow_affine_rotation_deg",
            "center_content_dx_px": "flow_center_content_dx_px",
            "center_content_dy_px": "flow_center_content_dy_px",
            "center_content_motion_px": "flow_center_content_motion_px",
            "elapsed_ms": "flow_elapsed_ms",
            "width": "flow_width",
            "height": "flow_height",
        }
    )

    keep_flow = [
        "pair_number",
        "flow_status",
        "flow_affine_ok",
        "corners_detected",
        "lk_tracks",
        "fb_good_tracks",
        "fb_error_median_px",
        "flow_inliers",
        "flow_inlier_ratio",
        "flow_affine_a00",
        "flow_affine_a01",
        "flow_affine_a10",
        "flow_affine_a11",
        "flow_affine_tx_px",
        "flow_affine_ty_px",
        "flow_affine_scale",
        "flow_affine_rotation_deg",
        "flow_center_content_dx_px",
        "flow_center_content_dy_px",
        "flow_center_content_motion_px",
        "flow_elapsed_ms",
        "flow_width",
        "flow_height",
        "in_stratified_diagnostic_subset",
        "primary_scene",
        "secondary_scene",
        "selection_role",
        "range_id",
    ]
    merged = orb.merge(flow[keep_flow], on="pair_number", how="inner", validate="one_to_one")
    if len(merged) != EXPECTED_PAIRS:
        raise RuntimeError("ORB/flow consistency merge did not cover all pairs.")

    step_rows: list[dict[str, Any]] = []
    for _, row in merged.iterrows():
        orb_step, orb_yaw = step_yaw_from_pair(row, prefix="orb_")

        if bool(row["flow_affine_ok"]):
            flow_step, flow_yaw = step_yaw_from_pair(row, prefix="flow_")
            step_delta = float(np.linalg.norm(orb_step - flow_step))
            yaw_delta = abs(wrap_angle_deg(flow_yaw - orb_yaw))
            scale_delta = abs(float(row["flow_affine_scale"]) - float(row["orb_affine_scale"]))
            motion_ratio = (
                float(np.linalg.norm(flow_step)) / max(float(np.linalg.norm(orb_step)), 1e-9)
            )
        else:
            flow_step = np.array([np.nan, np.nan], dtype=float)
            flow_yaw = float("nan")
            step_delta = float("nan")
            yaw_delta = float("nan")
            scale_delta = float("nan")
            motion_ratio = float("nan")

        flow_good = (
            bool(row["flow_affine_ok"])
            and str(row["flow_status"]) == "good"
            and float(row["flow_inlier_ratio"]) >= 0.35
        )
        orb_good = bool(row["orb_affine_ok"]) and str(row["orb_status"]) == "good"
        agreement_ok = bool(
            flow_good
            and orb_good
            and np.isfinite(step_delta)
            and step_delta <= 15.0
            and yaw_delta <= 5.0
            and 0.70 <= motion_ratio <= 1.30
        )
        suspicious_orb_step = bool(
            orb_good
            and flow_good
            and (
                step_delta > 30.0
                or yaw_delta > 10.0
                or motion_ratio < 0.50
                or motion_ratio > 1.80
            )
        )

        step_rows.append(
            {
                "pair_number": int(row["pair_number"]),
                "orb_step_x_local_px": float(orb_step[0]),
                "orb_step_y_local_px": float(orb_step[1]),
                "orb_yaw_increment_deg": float(orb_yaw),
                "flow_step_x_local_px": float(flow_step[0]),
                "flow_step_y_local_px": float(flow_step[1]),
                "flow_yaw_increment_deg": float(flow_yaw),
                "orb_flow_step_delta_px": step_delta,
                "orb_flow_yaw_delta_deg": yaw_delta,
                "orb_flow_scale_delta": scale_delta,
                "flow_orb_motion_ratio": motion_ratio,
                "flow_good_image_only": flow_good,
                "orb_good_image_only": orb_good,
                "orb_flow_agreement_ok": agreement_ok,
                "suspicious_orb_step_by_flow": suspicious_orb_step,
            }
        )

    return merged.merge(pd.DataFrame(step_rows), on="pair_number", how="inner", validate="one_to_one")


def integrate_steps(step_df: pd.DataFrame, variant: str) -> pd.DataFrame:
    position = np.zeros(2, dtype=float)
    yaw_rad = 0.0
    rows: list[dict[str, Any]] = [
        {
            "variant": variant,
            "sequence_frame_id": 0,
            "visual_x_px": 0.0,
            "visual_y_px": 0.0,
            "visual_yaw_rad": 0.0,
            "visual_yaw_deg_unwrapped": 0.0,
            "step_x_local_px": 0.0,
            "step_y_local_px": 0.0,
            "step_x_global_px": 0.0,
            "step_y_global_px": 0.0,
            "step_motion_px": 0.0,
            "pair_safe_image_only": True,
            "fusion_mode": "start",
        }
    ]

    for _, row in step_df.iterrows():
        orb_step = np.array(
            [row["orb_step_x_local_px"], row["orb_step_y_local_px"]],
            dtype=float,
        )
        flow_step = np.array(
            [row["flow_step_x_local_px"], row["flow_step_y_local_px"]],
            dtype=float,
        )
        orb_yaw = float(row["orb_yaw_increment_deg"])
        flow_yaw = float(row["flow_yaw_increment_deg"])

        if variant == "orb_only_recomputed":
            step = orb_step
            yaw_deg = orb_yaw
            safe = bool(row["orb_good_image_only"])
            fusion_mode = "orb"
        elif variant == "klt_flow_only":
            if bool(row["flow_good_image_only"]):
                step = flow_step
                yaw_deg = flow_yaw
                safe = True
                fusion_mode = "flow"
            else:
                step = np.array([0.0, 0.0], dtype=float)
                yaw_deg = 0.0
                safe = False
                fusion_mode = "flow_failed_zero_step"
        elif variant == "orb_flow_fallback":
            if bool(row["orb_good_image_only"]):
                step = orb_step
                yaw_deg = orb_yaw
                safe = True
                fusion_mode = "orb_primary"
            elif bool(row["flow_good_image_only"]):
                step = flow_step
                yaw_deg = flow_yaw
                safe = True
                fusion_mode = "flow_fallback"
            else:
                step = np.array([0.0, 0.0], dtype=float)
                yaw_deg = 0.0
                safe = False
                fusion_mode = "both_failed_zero_step"
        elif variant == "orb_flow_guarded_blend":
            if bool(row["orb_flow_agreement_ok"]):
                alpha = 0.80
                step = alpha * orb_step + (1.0 - alpha) * flow_step
                yaw_delta = wrap_angle_deg(flow_yaw - orb_yaw)
                yaw_deg = orb_yaw + (1.0 - alpha) * yaw_delta
                safe = True
                fusion_mode = "orb80_flow20_agree"
            elif bool(row["orb_good_image_only"]):
                step = orb_step
                yaw_deg = orb_yaw
                safe = True
                fusion_mode = "orb_due_to_disagreement"
            elif bool(row["flow_good_image_only"]):
                step = flow_step
                yaw_deg = flow_yaw
                safe = True
                fusion_mode = "flow_fallback"
            else:
                step = np.array([0.0, 0.0], dtype=float)
                yaw_deg = 0.0
                safe = False
                fusion_mode = "both_failed_zero_step"
        else:
            raise RuntimeError(f"Unknown trajectory variant: {variant}")

        global_step = rotation_matrix(yaw_rad) @ step
        position = position + global_step
        yaw_rad += math.radians(yaw_deg)

        rows.append(
            {
                "variant": variant,
                "sequence_frame_id": int(row["frame_index_b"]),
                "visual_x_px": float(position[0]),
                "visual_y_px": float(position[1]),
                "visual_yaw_rad": float(yaw_rad),
                "visual_yaw_deg_unwrapped": float(math.degrees(yaw_rad)),
                "step_x_local_px": float(step[0]),
                "step_y_local_px": float(step[1]),
                "step_x_global_px": float(global_step[0]),
                "step_y_global_px": float(global_step[1]),
                "step_motion_px": float(np.linalg.norm(global_step)),
                "pair_safe_image_only": safe,
                "fusion_mode": fusion_mode,
            }
        )

    return pd.DataFrame(rows)


def fit_similarity(source: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    valid = np.isfinite(source).all(axis=1) & np.isfinite(target).all(axis=1)
    source = source[valid]
    target = target[valid]
    if len(source) < 3:
        raise RuntimeError("At least three finite points are required for alignment.")

    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean

    covariance = source_centered.T @ target_centered
    u, singular_values, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1.0
        rotation = vt.T @ u.T
        singular_values[-1] *= -1.0

    denominator = float(np.sum(source_centered * source_centered))
    if denominator <= 1e-12:
        raise RuntimeError("Visual trajectory has insufficient spread.")

    scale = float(np.sum(singular_values) / denominator)
    translation = target_mean - scale * (rotation @ source_mean)
    return {
        "scale": scale,
        "rotation": rotation,
        "translation": translation,
        "rotation_deg": float(math.degrees(math.atan2(rotation[1, 0], rotation[0, 0]))),
        "fit_points": int(len(source)),
    }


def apply_similarity(points: np.ndarray, transform: dict[str, Any]) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    return transform["scale"] * (points @ transform["rotation"].T) + transform["translation"]


def trajectory_path_length(points: np.ndarray) -> float:
    diffs = np.diff(np.asarray(points, dtype=float), axis=0)
    return float(np.sum(np.linalg.norm(diffs, axis=1)))


def error_summary(errors: np.ndarray, reference_distance: np.ndarray, evaluation_start_index: int) -> dict[str, Any]:
    tail_errors = np.asarray(errors[evaluation_start_index:], dtype=float)
    tail_distance = np.asarray(reference_distance[evaluation_start_index:], dtype=float)
    valid = np.isfinite(tail_errors)
    tail_errors = tail_errors[valid]
    tail_distance = tail_distance[valid]
    if len(tail_errors) == 0:
        return {}

    distance_from_start = tail_distance - tail_distance[0]
    final_distance = float(distance_from_start[-1])
    final_error = float(tail_errors[-1])
    return {
        "evaluation_points": int(len(tail_errors)),
        "rmse_m": float(math.sqrt(np.mean(tail_errors * tail_errors))),
        "mean_error_m": float(np.mean(tail_errors)),
        "median_error_m": float(np.median(tail_errors)),
        "p95_error_m": float(np.percentile(tail_errors, 95)),
        "max_error_m": float(np.max(tail_errors)),
        "final_error_m": final_error,
        "evaluation_distance_m": final_distance,
        "final_drift_per_100m": (
            100.0 * final_error / final_distance if final_distance > 1e-9 else float("nan")
        ),
        "failure_rate_gt40m": float(np.mean(tail_errors > 40.0)),
    }


def first_sustained_crossing(
    errors: np.ndarray,
    reference_distance: np.ndarray,
    threshold_m: float,
    start_index: int,
    sustain_frames: int,
) -> dict[str, Any]:
    errors = np.asarray(errors, dtype=float)
    reference_distance = np.asarray(reference_distance, dtype=float)
    start_distance = float(reference_distance[start_index])
    above = np.isfinite(errors) & (errors >= threshold_m)

    for index in range(start_index, len(errors) - sustain_frames + 1):
        if bool(np.all(above[index : index + sustain_frames])):
            return {
                "threshold_m": float(threshold_m),
                "crossed": True,
                "frame_index": int(index),
                "frames_after_alignment_prefix": int(index - start_index),
                "distance_after_alignment_prefix_m": float(reference_distance[index] - start_distance),
                "error_at_crossing_m": float(errors[index]),
                "sustain_frames": int(sustain_frames),
            }

    return {
        "threshold_m": float(threshold_m),
        "crossed": False,
        "frame_index": None,
        "frames_after_alignment_prefix": None,
        "distance_after_alignment_prefix_m": None,
        "error_at_crossing_m": None,
        "sustain_frames": int(sustain_frames),
    }


def evaluate_trajectory(trajectory: pd.DataFrame, sequence: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any], list[dict[str, Any]]]:
    variant = str(trajectory["variant"].iloc[0])
    merged = sequence[
        ["sequence_frame_id", "token0_id", "x_enu_m", "y_enu_m"]
    ].merge(
        trajectory,
        on="sequence_frame_id",
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != EXPECTED_FRAMES:
        raise RuntimeError(f"{variant}: trajectory did not merge to all frames.")

    reference = merged[["x_enu_m", "y_enu_m"]].to_numpy(dtype=float)
    reference = reference - reference[0]
    visual = merged[["visual_x_px", "visual_y_px"]].to_numpy(dtype=float)

    reference_steps = np.linalg.norm(np.diff(reference, axis=0), axis=1)
    cumulative_reference_distance = np.concatenate([[0.0], np.cumsum(reference_steps)])

    global_transform = fit_similarity(visual, reference)
    global_aligned = apply_similarity(visual, global_transform)
    global_error = np.linalg.norm(global_aligned - reference, axis=1)

    prefix_count = ALIGNMENT_PREFIX_FRAMES
    prefix_transform = fit_similarity(visual[:prefix_count], reference[:prefix_count])
    prefix_aligned = apply_similarity(visual, prefix_transform)
    prefix_error = np.linalg.norm(prefix_aligned - reference, axis=1)

    output = merged.copy()
    output["reference_x_m"] = reference[:, 0]
    output["reference_y_m"] = reference[:, 1]
    output["reference_cumulative_distance_m"] = cumulative_reference_distance
    output["global_aligned_x_m"] = global_aligned[:, 0]
    output["global_aligned_y_m"] = global_aligned[:, 1]
    output["global_alignment_error_m"] = global_error
    output["prefix_aligned_x_m"] = prefix_aligned[:, 0]
    output["prefix_aligned_y_m"] = prefix_aligned[:, 1]
    output["prefix_locked_error_m"] = prefix_error

    distance_after_prefix = cumulative_reference_distance - cumulative_reference_distance[prefix_count - 1]
    output["distance_after_alignment_prefix_m"] = distance_after_prefix
    output["prefix_locked_drift_per_100m"] = np.divide(
        100.0 * prefix_error,
        distance_after_prefix,
        out=np.full_like(prefix_error, np.nan, dtype=float),
        where=distance_after_prefix > 1e-9,
    )

    global_summary = error_summary(global_error, cumulative_reference_distance, evaluation_start_index=0)
    prefix_summary = error_summary(prefix_error, cumulative_reference_distance, evaluation_start_index=prefix_count - 1)

    crossings = [
        first_sustained_crossing(
            prefix_error,
            cumulative_reference_distance,
            threshold,
            start_index=prefix_count - 1,
            sustain_frames=SUSTAIN_FRAMES,
        )
        for threshold in ERROR_THRESHOLDS_M
    ]

    summary = {
        "variant": variant,
        "frames": int(len(output)),
        "reference_path_m": trajectory_path_length(reference),
        "visual_path_px": trajectory_path_length(visual),
        "safe_pair_rate": float(output["pair_safe_image_only"].iloc[1:].astype(bool).mean()),
        "global_alignment": {
            "scale_m_per_px": float(global_transform["scale"]),
            "rotation_deg": float(global_transform["rotation_deg"]),
            **global_summary,
        },
        "prefix_locked_alignment": {
            "prefix_frames": int(prefix_count),
            "prefix_last_frame_index": int(prefix_count - 1),
            "prefix_reference_distance_m": float(cumulative_reference_distance[prefix_count - 1]),
            "scale_m_per_px": float(prefix_transform["scale"]),
            "rotation_deg": float(prefix_transform["rotation_deg"]),
            **prefix_summary,
        },
    }
    return output, summary, crossings


def load_orb_baseline(
    orb_summary_path: Path,
    orb_aligned_path: Path,
    orb_runtime_path: Path,
    orb_pairs: pd.DataFrame,
) -> dict[str, Any]:
    summary = pd.read_csv(orb_summary_path)
    selected = summary[summary["variant"].astype(str) == "se2_scale_normalized"]
    if selected.empty:
        raise RuntimeError("ORB trajectory summary lacks se2_scale_normalized.")
    row = selected.iloc[0]

    aligned = pd.read_csv(orb_aligned_path)
    aligned = aligned[aligned["variant"].astype(str) == "se2_scale_normalized"].sort_values("sequence_frame_id")
    prefix_tail = aligned.iloc[ALIGNMENT_PREFIX_FRAMES - 1 :]
    failure_rate = float((pd.to_numeric(prefix_tail["prefix_locked_error_m"], errors="coerce") > 40.0).mean())

    runtime = json.loads(orb_runtime_path.read_text(encoding="utf-8"))
    orb_feature_seconds = float(runtime.get("feature_cache_seconds", 0.0))
    orb_match_seconds = float(pd.to_numeric(orb_pairs["elapsed_ms"], errors="coerce").sum() / 1000.0)
    orb_total_seconds = orb_feature_seconds + orb_match_seconds

    return {
        "method": "orb",
        "source": "frozen_s6a",
        "frames": EXPECTED_FRAMES,
        "pairs": EXPECTED_PAIRS,
        "affine_success_rate": float(orb_pairs["affine_ok"].mean()),
        "good_quality_rate": float((orb_pairs["status"].astype(str) == "good").mean()),
        "inlier_ratio_median": finite_stat(orb_pairs["inlier_ratio"], np.median),
        "inlier_ratio_p05": finite_stat(orb_pairs["inlier_ratio"], lambda values: np.percentile(values, 5)),
        "feature_seconds": orb_feature_seconds,
        "matching_ransac_seconds": orb_match_seconds,
        "total_frontend_seconds": orb_total_seconds,
        "total_frontend_ms_per_pair": 1000.0 * orb_total_seconds / EXPECTED_PAIRS,
        "prefix_locked_rmse_m": float(row["prefix_locked_alignment.rmse_m"]),
        "prefix_locked_p95_m": float(row["prefix_locked_alignment.p95_error_m"]),
        "prefix_locked_max_m": float(row["prefix_locked_alignment.max_error_m"]),
        "final_error_m": float(row["prefix_locked_alignment.final_error_m"]),
        "final_drift_per_100m": float(row["prefix_locked_alignment.final_drift_per_100m"]),
        "failure_rate_gt40m": failure_rate,
    }


def method_summary_from_trajectory(
    variant: str,
    trajectory_summary: dict[str, Any],
    pair_df: pd.DataFrame,
    flow_wall_seconds: float,
) -> dict[str, Any]:
    prefix = trajectory_summary["prefix_locked_alignment"]
    if variant == "klt_flow_only":
        affine_success = float(pair_df["flow_affine_ok"].mean())
        good_rate = float(pair_df["flow_good_image_only"].mean())
        inlier_ratio = pair_df["flow_inlier_ratio"]
        frontend_seconds = float(flow_wall_seconds)
    elif variant in {"orb_flow_fallback", "orb_flow_guarded_blend", "orb_only_recomputed"}:
        affine_success = 1.0
        good_rate = float(pair_df["pair_safe_image_only_for_" + variant].mean()) if "pair_safe_image_only_for_" + variant in pair_df else 1.0
        inlier_ratio = pair_df["orb_inlier_ratio"]
        # ORB+flow requires both frozen ORB and newly computed flow in this study.
        frontend_seconds = float(flow_wall_seconds)
    else:
        affine_success = float("nan")
        good_rate = float("nan")
        inlier_ratio = pd.Series(dtype=float)
        frontend_seconds = float("nan")

    return {
        "method": variant,
        "source": "s7b0_recomputed",
        "frames": EXPECTED_FRAMES,
        "pairs": EXPECTED_PAIRS,
        "affine_success_rate": affine_success,
        "good_quality_rate": good_rate,
        "inlier_ratio_median": finite_stat(inlier_ratio, np.median),
        "inlier_ratio_p05": finite_stat(inlier_ratio, lambda values: np.percentile(values, 5)),
        "feature_seconds": float("nan"),
        "matching_ransac_seconds": float("nan"),
        "total_frontend_seconds": frontend_seconds,
        "total_frontend_ms_per_pair": 1000.0 * frontend_seconds / EXPECTED_PAIRS,
        "prefix_locked_rmse_m": float(prefix["rmse_m"]),
        "prefix_locked_p95_m": float(prefix["p95_error_m"]),
        "prefix_locked_max_m": float(prefix["max_error_m"]),
        "final_error_m": float(prefix["final_error_m"]),
        "final_drift_per_100m": float(prefix["final_drift_per_100m"]),
        "failure_rate_gt40m": float(prefix["failure_rate_gt40m"]),
    }


def append_xfeat_if_available(comparison_rows: list[dict[str, Any]], xfeat_comparison_path: Path) -> None:
    if not xfeat_comparison_path.exists():
        return
    try:
        frame = pd.read_csv(xfeat_comparison_path)
    except Exception:
        return
    if "method" not in frame.columns:
        return
    selected = frame[frame["method"].astype(str) == "xfeat"]
    if selected.empty:
        return
    row = selected.iloc[0].to_dict()
    row["source"] = "s7a2_full"
    comparison_rows.append(row)


def build_scene_summary(consistency: pd.DataFrame) -> pd.DataFrame:
    subset = consistency[consistency["in_stratified_diagnostic_subset"]].copy()
    if subset.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for scene, group in subset.groupby("primary_scene", sort=True):
        rows.append(
            {
                "primary_scene": scene,
                "pairs": int(len(group)),
                "orb_affine_success_rate": float(group["orb_affine_ok"].mean()),
                "flow_affine_success_rate": float(group["flow_affine_ok"].mean()),
                "orb_good_rate": float(group["orb_good_image_only"].mean()),
                "flow_good_rate": float(group["flow_good_image_only"].mean()),
                "agreement_rate": float(group["orb_flow_agreement_ok"].mean()),
                "suspicious_orb_step_rate": float(group["suspicious_orb_step_by_flow"].mean()),
                "orb_inlier_ratio_median": finite_stat(group["orb_inlier_ratio"], np.median),
                "flow_inlier_ratio_median": finite_stat(group["flow_inlier_ratio"], np.median),
                "step_delta_px_median": finite_stat(group["orb_flow_step_delta_px"], np.median),
                "yaw_delta_deg_median": finite_stat(group["orb_flow_yaw_delta_deg"], np.median),
                "flow_elapsed_ms_mean": finite_stat(group["flow_elapsed_ms"], np.mean),
            }
        )
    return pd.DataFrame(rows)


def decide_orb_flow(orb: dict[str, Any], best: dict[str, Any], consistency: pd.DataFrame) -> dict[str, Any]:
    rmse_improvement = (
        orb["prefix_locked_rmse_m"] - best["prefix_locked_rmse_m"]
    ) / max(orb["prefix_locked_rmse_m"], 1e-12)
    p95_ratio = best["prefix_locked_p95_m"] / max(orb["prefix_locked_p95_m"], 1e-12)
    max_ratio = best["prefix_locked_max_m"] / max(orb["prefix_locked_max_m"], 1e-12)
    failure_reduction = (
        orb["failure_rate_gt40m"] - best["failure_rate_gt40m"]
    ) / max(orb["failure_rate_gt40m"], 1e-12)
    runtime_ratio = best["total_frontend_ms_per_pair"] / max(orb["total_frontend_ms_per_pair"], 1e-12)

    # Bounded study guard: ORB+flow must preserve ORB robustness and offer a material trajectory benefit.
    guard_pass = bool(
        best["method"] in {"orb_flow_guarded_blend", "orb_flow_fallback"}
        and best["prefix_locked_p95_m"] <= orb["prefix_locked_p95_m"] * 1.02
        and best["prefix_locked_max_m"] <= orb["prefix_locked_max_m"] * 1.02
        and runtime_ratio <= 2.0
    )
    material_benefit = bool(
        rmse_improvement >= 0.05
        or failure_reduction >= 0.20
    )
    promote = bool(guard_pass and material_benefit)

    return {
        "decision": "PROMOTE_ORB_FLOW" if promote else "KEEP_ORB",
        "promote_orb_flow": promote,
        "selected_orb_flow_variant": best["method"],
        "comparison_values": {
            "prefix_rmse_improvement_fraction": float(rmse_improvement),
            "p95_ratio_variant_over_orb": float(p95_ratio),
            "max_ratio_variant_over_orb": float(max_ratio),
            "failure_rate_reduction_fraction": float(failure_reduction),
            "runtime_ratio_variant_over_orb": float(runtime_ratio),
            "suspicious_orb_step_rate_by_flow": float(consistency["suspicious_orb_step_by_flow"].mean()),
            "orb_flow_agreement_rate": float(consistency["orb_flow_agreement_ok"].mean()),
        },
        "guards": {
            "robustness_guard_pass": guard_pass,
            "material_benefit_pass": material_benefit,
            "runtime_guard": "variant runtime must be <= 2x ORB",
            "accuracy_guard": "p95 and max error must not exceed ORB by more than 2%",
        },
        "closeout_rule": (
            "S7B.0 is a bounded consistency study. No optical-flow parameter sweep follows."
        ),
    }


def save_plots(
    aligned_all: pd.DataFrame,
    consistency: pd.DataFrame,
    orb_aligned_path: Path,
    xfeat_aligned_path: Path,
    figures_dir: Path,
) -> None:
    ensure_dir(figures_dir)

    orb_aligned = pd.read_csv(orb_aligned_path)
    orb_aligned = orb_aligned[
        orb_aligned["variant"].astype(str) == "se2_scale_normalized"
    ].sort_values("sequence_frame_id")

    plt.figure(figsize=(11, 6))
    plt.plot(
        orb_aligned["reference_cumulative_distance_m"],
        orb_aligned["prefix_locked_error_m"],
        label="ORB",
    )
    if xfeat_aligned_path.exists():
        xfeat = pd.read_csv(xfeat_aligned_path).sort_values("sequence_frame_id")
        plt.plot(
            xfeat["reference_cumulative_distance_m"],
            xfeat["prefix_locked_error_m"],
            label="XFeat",
        )
    for variant, group in aligned_all.groupby("variant", sort=True):
        if variant in {"orb_flow_guarded_blend", "klt_flow_only"}:
            plt.plot(
                group["reference_cumulative_distance_m"],
                group["prefix_locked_error_m"],
                label=variant,
            )
    for threshold in ERROR_THRESHOLDS_M:
        plt.axhline(threshold, linestyle="--", linewidth=1.0, label=f"{threshold:g} m")
    plt.xlabel("Reference cumulative distance [m] — evaluation only")
    plt.ylabel("Prefix-locked error [m]")
    plt.title("S7B.0 ORB, XFeat and ORB+flow error growth")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "s7b0_error_vs_distance.png", dpi=180)
    plt.close()

    plt.figure(figsize=(9, 8))
    plt.plot(orb_aligned["reference_x_m"], orb_aligned["reference_y_m"], label="Reference — evaluation only")
    plt.plot(orb_aligned["prefix_aligned_x_m"], orb_aligned["prefix_aligned_y_m"], label="ORB")
    selected = aligned_all[aligned_all["variant"] == "orb_flow_guarded_blend"]
    if not selected.empty:
        plt.plot(selected["prefix_aligned_x_m"], selected["prefix_aligned_y_m"], label="ORB+flow guarded")
    plt.axis("equal")
    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")
    plt.title("S7B.0 prefix-locked trajectory")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "s7b0_prefix_locked_xy.png", dpi=180)
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.plot(consistency["pair_number"], consistency["orb_flow_step_delta_px"], label="step delta [px]")
    plt.plot(consistency["pair_number"], consistency["orb_flow_yaw_delta_deg"], label="yaw delta [deg]")
    plt.xlabel("Stride-1 pair number")
    plt.ylabel("ORB-flow disagreement")
    plt.title("S7B.0 ORB versus KLT-flow consistency")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "s7b0_orb_flow_disagreement.png", dpi=180)
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.plot(consistency["pair_number"], consistency["orb_inlier_ratio"], label="ORB")
    plt.plot(consistency["pair_number"], consistency["flow_inlier_ratio"], label="KLT flow")
    plt.xlabel("Stride-1 pair number")
    plt.ylabel("RANSAC inlier ratio")
    plt.title("S7B.0 ORB versus KLT-flow inlier ratio")
    plt.ylim(0.0, 1.02)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "s7b0_orb_flow_inlier_ratio.png", dpi=180)
    plt.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--resize-long", type=int, default=960)
    parser.add_argument("--max-corners", type=int, default=1200)
    parser.add_argument("--quality-level", type=float, default=0.01)
    parser.add_argument("--min-distance", type=float, default=8.0)
    parser.add_argument("--block-size", type=int, default=7)
    parser.add_argument("--lk-win-size", type=int, default=21)
    parser.add_argument("--lk-max-level", type=int, default=3)
    parser.add_argument("--fb-error-thresh", type=float, default=1.5)
    parser.add_argument("--ransac-threshold", type=float, default=3.0)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    output_root = resolve_from_repo(repo_root, args.output_root)

    if args.resize_long != 960:
        raise RuntimeError("S7B.0 is frozen at resize-long 960.")
    if args.max_corners != 1200:
        raise RuntimeError("S7B.0 is frozen at max-corners 1200.")
    if abs(args.ransac_threshold - 3.0) > 1e-12:
        raise RuntimeError("S7B.0 is frozen at RANSAC threshold 3.0 px.")

    paths = {
        "sequence": resolve_from_repo(repo_root, DEFAULT_SEQUENCE_MANIFEST),
        "orb_pairs": resolve_from_repo(repo_root, DEFAULT_ORB_PAIRS),
        "orb_aligned": resolve_from_repo(repo_root, DEFAULT_ORB_ALIGNED),
        "orb_summary": resolve_from_repo(repo_root, DEFAULT_ORB_SUMMARY),
        "orb_runtime": resolve_from_repo(repo_root, DEFAULT_ORB_RUNTIME),
        "full_pair_manifest": resolve_from_repo(repo_root, DEFAULT_FULL_PAIR_MANIFEST),
        "xfeat_comparison": resolve_from_repo(repo_root, DEFAULT_XFEAT_COMPARISON),
        "xfeat_aligned": resolve_from_repo(repo_root, DEFAULT_XFEAT_ALIGNED),
        "xfeat_decision": resolve_from_repo(repo_root, DEFAULT_XFEAT_DECISION),
    }
    for required_name in ["sequence", "orb_pairs", "orb_aligned", "orb_summary", "orb_runtime", "full_pair_manifest"]:
        if not paths[required_name].exists():
            raise FileNotFoundError(f"Missing {required_name}: {paths[required_name]}")

    sequence = load_sequence_manifest(repo_root, paths["sequence"])
    orb_pairs = load_orb_pairs(paths["orb_pairs"])
    pair_manifest = load_pair_manifest(paths["full_pair_manifest"])

    metadata_dir = ensure_dir(output_root / "metadata" / "s7_relative_frontend")
    reports_dir = ensure_dir(output_root / "reports" / "s7_relative_frontend")
    figures_dir = ensure_dir(output_root / "figures" / "s7_relative_frontend" / "s7b0_orb_flow")

    flow_wall_start = time.perf_counter()
    flow_pairs = run_klt_flow(
        sequence=sequence,
        pair_manifest=pair_manifest,
        resize_long=args.resize_long,
        max_corners=args.max_corners,
        quality_level=args.quality_level,
        min_distance=args.min_distance,
        block_size=args.block_size,
        lk_win_size=args.lk_win_size,
        lk_max_level=args.lk_max_level,
        fb_error_thresh=args.fb_error_thresh,
        ransac_threshold=args.ransac_threshold,
    )
    flow_wall_seconds = time.perf_counter() - flow_wall_start
    peak_memory = peak_process_memory_mb()

    consistency = build_consistency_table(orb_pairs, flow_pairs)

    raw_trajectories: list[pd.DataFrame] = []
    aligned_trajectories: list[pd.DataFrame] = []
    trajectory_summaries: list[dict[str, Any]] = []
    crossing_rows: list[dict[str, Any]] = []

    for variant in [
        "orb_only_recomputed",
        "klt_flow_only",
        "orb_flow_fallback",
        "orb_flow_guarded_blend",
    ]:
        raw = integrate_steps(consistency, variant)
        raw_trajectories.append(raw)
        aligned, summary, crossings = evaluate_trajectory(raw, sequence)
        aligned_trajectories.append(aligned)
        trajectory_summaries.append(summary)
        crossing_rows.extend([{"variant": variant, **row} for row in crossings])

    raw_all = pd.concat(raw_trajectories, ignore_index=True)
    aligned_all = pd.concat(aligned_trajectories, ignore_index=True)
    summary_norm = pd.json_normalize(trajectory_summaries, sep=".")

    orb_baseline = load_orb_baseline(
        paths["orb_summary"],
        paths["orb_aligned"],
        paths["orb_runtime"],
        orb_pairs,
    )

    comparison_rows: list[dict[str, Any]] = [orb_baseline]
    append_xfeat_if_available(comparison_rows, paths["xfeat_comparison"])

    for summary in trajectory_summaries:
        comparison_rows.append(
            method_summary_from_trajectory(
                summary["variant"],
                summary,
                consistency,
                flow_wall_seconds=flow_wall_seconds,
            )
        )

    comparison_df = pd.DataFrame(comparison_rows)

    candidate_rows = comparison_df[
        comparison_df["method"].isin(["orb_flow_fallback", "orb_flow_guarded_blend"])
    ].copy()
    candidate_rows["rmse_improvement_vs_orb"] = (
        orb_baseline["prefix_locked_rmse_m"] - candidate_rows["prefix_locked_rmse_m"]
    ) / max(orb_baseline["prefix_locked_rmse_m"], 1e-12)
    candidate_rows = candidate_rows.sort_values(
        ["rmse_improvement_vs_orb", "failure_rate_gt40m"],
        ascending=[False, True],
        kind="mergesort",
    )
    best_orb_flow = candidate_rows.iloc[0].to_dict()
    decision = decide_orb_flow(orb_baseline, best_orb_flow, consistency)

    scene_summary = build_scene_summary(consistency)

    flow_pair_path = metadata_dir / "s7b_0_klt_flow_pair_diagnostics.csv"
    consistency_path = metadata_dir / "s7b_0_orb_flow_consistency_pairs.csv"
    raw_path = metadata_dir / "s7b_0_relative_trajectory_pixels.csv"
    aligned_path = metadata_dir / "s7b_0_relative_trajectory_aligned_eval_only.csv"
    trajectory_summary_path = metadata_dir / "s7b_0_trajectory_summary.csv"
    method_comparison_path = metadata_dir / "s7b_0_method_comparison.csv"
    scene_summary_path = metadata_dir / "s7b_0_scene_consistency_summary.csv"
    crossing_path = metadata_dir / "s7b_0_drift_threshold_crossings.csv"
    decision_path = metadata_dir / "s7b_0_orb_flow_decision.json"
    json_report_path = reports_dir / "s7b_0_orb_flow_consistency_study.json"
    md_report_path = reports_dir / "s7b_0_orb_flow_consistency_study_report.md"

    flow_pairs.to_csv(flow_pair_path, index=False)
    consistency.to_csv(consistency_path, index=False)
    raw_all.to_csv(raw_path, index=False)
    aligned_all.to_csv(aligned_path, index=False)
    summary_norm.to_csv(trajectory_summary_path, index=False)
    comparison_df.to_csv(method_comparison_path, index=False)
    scene_summary.to_csv(scene_summary_path, index=False)
    pd.DataFrame(crossing_rows).to_csv(crossing_path, index=False)
    decision_path.write_text(json.dumps(decision, indent=2), encoding="utf-8")

    save_plots(
        aligned_all,
        consistency,
        paths["orb_aligned"],
        paths["xfeat_aligned"],
        figures_dir,
    )

    payload = {
        "generated_utc": utc_now(),
        "stage": "S7B.0",
        "status": "COMPLETE_" + decision["decision"],
        "scientific_scope": (
            "Camera-only sparse KLT optical-flow consistency study and bounded "
            "ORB+flow fusion check on traj01."
        ),
        "configuration": {
            "resize_long": args.resize_long,
            "max_corners": args.max_corners,
            "quality_level": args.quality_level,
            "min_distance": args.min_distance,
            "block_size": args.block_size,
            "lk_win_size": args.lk_win_size,
            "lk_max_level": args.lk_max_level,
            "fb_error_thresh": args.fb_error_thresh,
            "ransac_threshold": args.ransac_threshold,
            "agreement_step_delta_px": 15.0,
            "agreement_yaw_delta_deg": 5.0,
            "agreement_motion_ratio_range": [0.70, 1.30],
            "blend_rule": "80% ORB + 20% KLT flow only when image-only agreement passes",
        },
        "counts": {
            "frames": EXPECTED_FRAMES,
            "pairs": EXPECTED_PAIRS,
            "flow_affine_successes": int(flow_pairs["affine_ok"].sum()),
            "flow_good_pairs": int((flow_pairs["status"].astype(str) == "good").sum()),
            "orb_flow_agreement_pairs": int(consistency["orb_flow_agreement_ok"].sum()),
            "suspicious_orb_steps_by_flow": int(consistency["suspicious_orb_step_by_flow"].sum()),
        },
        "timing": {
            "flow_wall_seconds": float(flow_wall_seconds),
            "flow_ms_per_pair": float(1000.0 * flow_wall_seconds / EXPECTED_PAIRS),
            "peak_process_memory_mb": float(peak_memory),
        },
        "orb_baseline": orb_baseline,
        "method_comparison": comparison_df.to_dict(orient="records"),
        "decision": decision,
        "ground_truth_rule": (
            "Reference ENU is used only after image-only trajectories are built, "
            "for S6A-compatible alignment and error metrics."
        ),
        "git": {
            "branch": git_value(repo_root, "branch", "--show-current"),
            "commit": git_value(repo_root, "rev-parse", "HEAD"),
            "working_tree_porcelain": git_value(repo_root, "status", "--porcelain"),
        },
        "environment": {
            "python": sys.version,
            "opencv": cv2.__version__,
            "platform": platform.platform(),
        },
        "outputs": {
            "flow_pair_diagnostics": str(flow_pair_path.relative_to(repo_root)),
            "consistency_pairs": str(consistency_path.relative_to(repo_root)),
            "raw_trajectories": str(raw_path.relative_to(repo_root)),
            "aligned_trajectories": str(aligned_path.relative_to(repo_root)),
            "trajectory_summary": str(trajectory_summary_path.relative_to(repo_root)),
            "method_comparison": str(method_comparison_path.relative_to(repo_root)),
            "scene_summary": str(scene_summary_path.relative_to(repo_root)),
            "crossings": str(crossing_path.relative_to(repo_root)),
            "decision": str(decision_path.relative_to(repo_root)),
            "figures": str(figures_dir.relative_to(repo_root)),
            "report": str(md_report_path.relative_to(repo_root)),
        },
    }
    json_report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    selected_cols = [
        "method",
        "prefix_locked_rmse_m",
        "prefix_locked_p95_m",
        "prefix_locked_max_m",
        "final_error_m",
        "final_drift_per_100m",
        "failure_rate_gt40m",
        "total_frontend_ms_per_pair",
    ]
    present_cols = [col for col in selected_cols if col in comparison_df.columns]
    method_table = comparison_df[present_cols].to_markdown(index=False)

    scene_table = (
        scene_summary.to_markdown(index=False)
        if not scene_summary.empty
        else "No stratified scene rows available."
    )

    md_report = f"""# S7B.0 — ORB + Optical-Flow Relative Motion Consistency Study

Generated: `{payload["generated_utc"]}`

## Status

```text
{payload["status"]}
```

## What was tested

This block computes sparse Lucas-Kanade optical flow from consecutive UAV frames
only. No IMU, GNSS, timestamps, velocity logs, reference trajectory, retrieval
rank, or oracle label is used by the optical-flow frontend.

```text
Frames:                 {EXPECTED_FRAMES}
Consecutive pairs:      {EXPECTED_PAIRS}
Resolution:             960 px long side
KLT corners:            1200
Forward-backward gate:  {args.fb_error_thresh} px
RANSAC threshold:       {args.ransac_threshold} px
Fusion variant:         ORB 80% + flow 20% only when image-only agreement passes
```

## Main method comparison

{method_table}

## Scene consistency summary

{scene_table}

## Decision

```text
{decision["decision"]}
```

This is a bounded consistency study. If ORB+flow does not clearly reduce drift or
failure rate while preserving ORB p95/max robustness, ORB remains the frozen
relative frontend.

## Outputs

```text
{flow_pair_path.relative_to(repo_root)}
{consistency_path.relative_to(repo_root)}
{method_comparison_path.relative_to(repo_root)}
{scene_summary_path.relative_to(repo_root)}
{decision_path.relative_to(repo_root)}
{figures_dir.relative_to(repo_root)}
```
"""
    md_report_path.write_text(md_report, encoding="utf-8")

    best_flow_method = decision["selected_orb_flow_variant"]
    best_row = comparison_df[comparison_df["method"] == best_flow_method].iloc[0]

    print("S7B.0 ORB + Optical-Flow Consistency Study")
    print("------------------------------------------")
    print(f"Status:                     {payload['status']}")
    print(f"Frames:                     {EXPECTED_FRAMES}")
    print(f"Pairs:                      {EXPECTED_PAIRS}")
    print(f"Flow affine success:        {float(flow_pairs['affine_ok'].mean()):.4f}")
    print(f"Flow good-quality rate:     {float((flow_pairs['status'].astype(str) == 'good').mean()):.4f}")
    print(f"ORB-flow agreement rate:    {float(consistency['orb_flow_agreement_ok'].mean()):.4f}")
    print(f"Suspicious ORB step rate:   {float(consistency['suspicious_orb_step_by_flow'].mean()):.4f}")
    print(f"ORB prefix RMSE m:          {orb_baseline['prefix_locked_rmse_m']:.3f}")
    print(f"{best_flow_method} RMSE m:  {float(best_row['prefix_locked_rmse_m']):.3f}")
    print(f"ORB prefix p95 m:           {orb_baseline['prefix_locked_p95_m']:.3f}")
    print(f"{best_flow_method} p95 m:   {float(best_row['prefix_locked_p95_m']):.3f}")
    print(f"ORB final drift m/100m:     {orb_baseline['final_drift_per_100m']:.3f}")
    print(f"{best_flow_method} drift:   {float(best_row['final_drift_per_100m']):.3f}")
    print(f"Flow wall ms/pair:          {1000.0 * flow_wall_seconds / EXPECTED_PAIRS:.2f}")
    print(f"Peak process memory MB:     {peak_memory:.1f}")
    print(f"Decision:                   {decision['decision']}")
    print(f"Flow diagnostics:           {flow_pair_path.relative_to(repo_root)}")
    print(f"Consistency pairs:          {consistency_path.relative_to(repo_root)}")
    print(f"Method comparison:          {method_comparison_path.relative_to(repo_root)}")
    print(f"Scene summary:              {scene_summary_path.relative_to(repo_root)}")
    print(f"Decision JSON:              {decision_path.relative_to(repo_root)}")
    print(f"Figures:                    {figures_dir.relative_to(repo_root)}")
    print(f"Report:                     {md_report_path.relative_to(repo_root)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print("S7B.0 ORB + Optical-Flow Consistency Study", file=sys.stderr)
        print("------------------------------------------", file=sys.stderr)
        print("Status: BLOCKED", file=sys.stderr)
        print(f"Reason: {exc}", file=sys.stderr)
        raise
