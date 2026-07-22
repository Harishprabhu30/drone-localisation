#!/usr/bin/env python3
"""S7B.1 — ORB Sliding-Window SE(2) Pose Smoothing.

Backend-only relative-motion study.

Inputs:
- frozen S6A ORB stride-1 affine pair diagnostics;
- frozen S6A sequence manifest and feature metadata;
- optional S7A/S7B comparison files for report tables.

No ORB/XFeat/flow frontend is rerun. This script starts from frozen ORB affine
transforms, converts them to SE(2) local increments, applies fixed image-only
smoothing policies, integrates trajectories, and evaluates them with the same
S6A prefix-alignment protocol.

Variants:
- orb_only_recomputed:
    Direct integration of frozen ORB affine increments. This is a sanity check.
- orb_confidence_ewma_causal:
    Deployable causal smoother. Each local step/yaw increment is blended with
    previous smoothed velocity using confidence from ORB inliers only.
- orb_window9_weighted_smooth:
    Centered 9-pair weighted smoother. Diagnostic/offline unless latency is
    allowed because it uses future increments.

Reference ENU is used only after each complete image-only trajectory is built.

Command USed:

export PYTHONPATH=$PWD/src

mkdir -p outputs/satloc/reports/s7_relative_frontend
set -o pipefail

python scripts/satloc/s7/s7b_1_orb_sliding_window_se2_smoothing.py \
  2>&1 | tee \
  outputs/satloc/reports/s7_relative_frontend/s7b_1_orb_sliding_window_se2_smoothing.log

"""

from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

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
DEFAULT_ORB_FEATURES = Path(
    "outputs/satloc/metadata/s6a_relative_motion/s6a1_orb_frame_features.csv"
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
DEFAULT_FLOW_COMPARISON = Path(
    "outputs/satloc/metadata/s7_relative_frontend/"
    "s7b_0_method_comparison.csv"
)
DEFAULT_FLOW_DECISION = Path(
    "outputs/satloc/metadata/s7_relative_frontend/"
    "s7b_0_orb_flow_decision.json"
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


def angle_lerp_deg(previous: float, current: float, alpha: float) -> float:
    delta = wrap_angle_deg(current - previous)
    return float(previous + alpha * delta)


def circular_weighted_mean_deg(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(valid):
        return float("nan")
    angles = np.deg2rad(values[valid])
    w = weights[valid]
    vector = np.sum(w * np.exp(1j * angles)) / np.sum(w)
    return float(np.rad2deg(np.angle(vector)))


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
    # OpenCV image y-axis points down. Convert to y-up visual frame.
    return np.array([step_image[0], -step_image[1]], dtype=float)


def rotation_matrix(angle_rad: float) -> np.ndarray:
    cosine = math.cos(angle_rad)
    sine = math.sin(angle_rad)
    return np.array([[cosine, -sine], [sine, cosine]], dtype=float)


def load_sequence_manifest(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"sequence_frame_id", "token0_id", "x_enu_m", "y_enu_m"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise RuntimeError(f"Sequence manifest missing columns: {missing}")

    frame["sequence_frame_id"] = pd.to_numeric(frame["sequence_frame_id"], errors="raise").astype(int)
    frame["token0_id"] = pd.to_numeric(frame["token0_id"], errors="raise").astype(int)
    frame["x_enu_m"] = pd.to_numeric(frame["x_enu_m"], errors="raise")
    frame["y_enu_m"] = pd.to_numeric(frame["y_enu_m"], errors="raise")
    frame = frame.sort_values("sequence_frame_id", kind="mergesort").reset_index(drop=True)

    if len(frame) != EXPECTED_FRAMES:
        raise RuntimeError(f"Expected {EXPECTED_FRAMES} sequence rows, found {len(frame)}.")
    if not np.array_equal(frame["sequence_frame_id"].to_numpy(dtype=int), np.arange(EXPECTED_FRAMES)):
        raise RuntimeError("sequence_frame_id must be contiguous 0..1033.")
    if not np.array_equal(frame["token0_id"].to_numpy(dtype=int), np.arange(1, EXPECTED_FRAMES + 1)):
        raise RuntimeError("token0_id must be canonical 1..1034.")
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
        raise RuntimeError(f"ORB pair diagnostics missing columns: {missing}")

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
        raise RuntimeError(f"Expected {EXPECTED_PAIRS} stride-1 ORB pairs, found {len(frame)}.")
    if not bool(frame["affine_ok"].all()):
        raise RuntimeError("Frozen ORB chain has failed affine rows; this S7B.1 backend assumes complete ORB.")
    return frame


def load_feature_dimensions(path: Path) -> tuple[int, int]:
    frame = pd.read_csv(path)
    required = {"width", "height"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise RuntimeError(f"Feature metadata missing columns: {missing}")
    width = int(pd.to_numeric(frame["width"], errors="raise").iloc[0])
    height = int(pd.to_numeric(frame["height"], errors="raise").iloc[0])
    return width, height


def load_full_pair_labels(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    if "pair_number" not in frame.columns:
        return None
    frame["pair_number"] = pd.to_numeric(frame["pair_number"], errors="coerce").astype("Int64")
    for column in ["in_stratified_diagnostic_subset", "primary_scene", "secondary_scene", "selection_role", "range_id"]:
        if column not in frame.columns:
            frame[column] = ""
    frame["in_stratified_diagnostic_subset"] = bool_series(frame["in_stratified_diagnostic_subset"])
    return frame[
        [
            "pair_number",
            "in_stratified_diagnostic_subset",
            "primary_scene",
            "secondary_scene",
            "selection_role",
            "range_id",
        ]
    ].dropna(subset=["pair_number"]).astype({"pair_number": int})


def compute_image_only_confidence(pair_df: pd.DataFrame) -> pd.Series:
    ratio = pd.to_numeric(pair_df["inlier_ratio"], errors="coerce").fillna(0.0)
    inliers = pd.to_numeric(pair_df["inliers"], errors="coerce").fillna(0.0)
    matches = pd.to_numeric(pair_df["good_matches"], errors="coerce").fillna(0.0)
    scale = pd.to_numeric(pair_df["affine_scale"], errors="coerce").fillna(1.0)

    ratio_score = np.clip((ratio - 0.35) / (0.95 - 0.35), 0.0, 1.0)
    inlier_score = np.clip(inliers / 400.0, 0.0, 1.0)
    match_score = np.clip(matches / 700.0, 0.0, 1.0)
    scale_score = np.exp(-np.abs(scale - 1.0) / 0.10)
    status_factor = np.where(pair_df["status"].astype(str) == "good", 1.0, 0.50)

    confidence = (
        0.50 * ratio_score
        + 0.25 * inlier_score
        + 0.15 * match_score
        + 0.10 * scale_score
    ) * status_factor
    return pd.Series(np.clip(confidence, 0.0, 1.0), index=pair_df.index)


def build_orb_increment_table(
    orb_pairs: pd.DataFrame,
    width: int,
    height: int,
    full_pair_labels: pd.DataFrame | None,
) -> pd.DataFrame:
    center = np.array([width / 2.0, height / 2.0, 1.0], dtype=float)
    rows: list[dict[str, Any]] = []

    for _, row in orb_pairs.iterrows():
        matrix = normalize_affine_scale_about_center(affine_from_row(row), center)
        step = local_camera_step_from_scene_affine(matrix, center)
        rows.append(
            {
                "pair_number": int(row["pair_number"]),
                "frame_index_a": int(row["frame_index_a"]),
                "frame_index_b": int(row["frame_index_b"]),
                "token0_a": int(row["token0_a"]),
                "token0_b": int(row["token0_b"]),
                "status": str(row["status"]),
                "affine_ok": bool(row["affine_ok"]),
                "good_matches": int(row["good_matches"]),
                "inliers": int(row["inliers"]),
                "inlier_ratio": float(row["inlier_ratio"]),
                "affine_scale": float(row["affine_scale"]),
                "affine_rotation_deg": float(row["affine_rotation_deg"]),
                "elapsed_ms": float(row["elapsed_ms"]),
                "raw_step_x_local_px": float(step[0]),
                "raw_step_y_local_px": float(step[1]),
                "raw_step_motion_px": float(np.linalg.norm(step)),
                "raw_yaw_increment_deg": float(row["affine_rotation_deg"]),
                "image_only_confidence": 0.0,
            }
        )

    increments = pd.DataFrame(rows)
    increments["image_only_confidence"] = compute_image_only_confidence(orb_pairs).to_numpy(dtype=float)

    if full_pair_labels is not None:
        increments = increments.merge(
            full_pair_labels,
            on="pair_number",
            how="left",
            validate="one_to_one",
        )
    else:
        increments["in_stratified_diagnostic_subset"] = False
        increments["primary_scene"] = ""
        increments["secondary_scene"] = ""
        increments["selection_role"] = ""
        increments["range_id"] = ""

    for column in ["primary_scene", "secondary_scene", "selection_role", "range_id"]:
        increments[column] = increments[column].fillna("").astype(str)
    increments["in_stratified_diagnostic_subset"] = bool_series(
        increments["in_stratified_diagnostic_subset"]
    )
    return increments


def apply_causal_confidence_ewma(increments: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    previous_step: np.ndarray | None = None
    previous_yaw: float | None = None

    for _, row in increments.iterrows():
        raw_step = np.array(
            [row["raw_step_x_local_px"], row["raw_step_y_local_px"]],
            dtype=float,
        )
        raw_yaw = float(row["raw_yaw_increment_deg"])
        confidence = float(row["image_only_confidence"])
        alpha = 0.55 + 0.40 * confidence

        if previous_step is None or previous_yaw is None:
            smoothed_step = raw_step
            smoothed_yaw = raw_yaw
        else:
            smoothed_step = alpha * raw_step + (1.0 - alpha) * previous_step
            smoothed_yaw = angle_lerp_deg(previous_yaw, raw_yaw, alpha)

        rows.append(
            {
                "pair_number": int(row["pair_number"]),
                "causal_alpha": float(alpha),
                "causal_step_x_local_px": float(smoothed_step[0]),
                "causal_step_y_local_px": float(smoothed_step[1]),
                "causal_step_motion_px": float(np.linalg.norm(smoothed_step)),
                "causal_yaw_increment_deg": float(smoothed_yaw),
                "causal_delta_from_raw_step_px": float(np.linalg.norm(smoothed_step - raw_step)),
                "causal_delta_from_raw_yaw_deg": abs(wrap_angle_deg(smoothed_yaw - raw_yaw)),
            }
        )
        previous_step = smoothed_step
        previous_yaw = smoothed_yaw

    return increments.merge(pd.DataFrame(rows), on="pair_number", how="inner", validate="one_to_one")


def apply_window9_weighted_smooth(increments: pd.DataFrame, window: int) -> pd.DataFrame:
    if window % 2 != 1 or window < 3:
        raise RuntimeError("Sliding window must be odd and at least 3.")

    half = window // 2
    steps = increments[["raw_step_x_local_px", "raw_step_y_local_px"]].to_numpy(dtype=float)
    yaws = increments["raw_yaw_increment_deg"].to_numpy(dtype=float)
    confidence = increments["image_only_confidence"].to_numpy(dtype=float)

    rows: list[dict[str, Any]] = []
    n = len(increments)

    for index, row in increments.reset_index(drop=True).iterrows():
        lo = max(0, index - half)
        hi = min(n, index + half + 1)
        neighbor_indices = np.arange(lo, hi)
        distance = np.abs(neighbor_indices - index)
        triangular = (half + 1 - distance).astype(float)
        weights = triangular * np.clip(confidence[lo:hi], 0.05, 1.0)

        raw_step = steps[index]
        if weights.sum() <= 1e-12:
            smoothed_step = raw_step
            smoothed_yaw = yaws[index]
        else:
            smoothed_step = np.sum(steps[lo:hi] * weights[:, None], axis=0) / np.sum(weights)
            smoothed_yaw = circular_weighted_mean_deg(yaws[lo:hi], weights)

        rows.append(
            {
                "pair_number": int(row["pair_number"]),
                "window9_step_x_local_px": float(smoothed_step[0]),
                "window9_step_y_local_px": float(smoothed_step[1]),
                "window9_step_motion_px": float(np.linalg.norm(smoothed_step)),
                "window9_yaw_increment_deg": float(smoothed_yaw),
                "window9_delta_from_raw_step_px": float(np.linalg.norm(smoothed_step - raw_step)),
                "window9_delta_from_raw_yaw_deg": abs(wrap_angle_deg(smoothed_yaw - yaws[index])),
                "window9_effective_neighbors": int(hi - lo),
            }
        )

    return increments.merge(pd.DataFrame(rows), on="pair_number", how="inner", validate="one_to_one")


def integrate_increment_variant(increments: pd.DataFrame, variant: str) -> pd.DataFrame:
    if variant == "orb_only_recomputed":
        step_x = "raw_step_x_local_px"
        step_y = "raw_step_y_local_px"
        yaw_col = "raw_yaw_increment_deg"
        delta_col = None
        deployable = True
    elif variant == "orb_confidence_ewma_causal":
        step_x = "causal_step_x_local_px"
        step_y = "causal_step_y_local_px"
        yaw_col = "causal_yaw_increment_deg"
        delta_col = "causal_delta_from_raw_step_px"
        deployable = True
    elif variant == "orb_window9_weighted_smooth":
        step_x = "window9_step_x_local_px"
        step_y = "window9_step_y_local_px"
        yaw_col = "window9_yaw_increment_deg"
        delta_col = "window9_delta_from_raw_step_px"
        deployable = False
    else:
        raise RuntimeError(f"Unknown variant: {variant}")

    position = np.zeros(2, dtype=float)
    yaw_rad = 0.0
    rows: list[dict[str, Any]] = [
        {
            "variant": variant,
            "deployable_online": deployable,
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
            "yaw_increment_deg": 0.0,
            "image_only_confidence": 1.0,
            "pair_safe_image_only": True,
            "delta_from_raw_step_px": 0.0,
        }
    ]

    for _, row in increments.iterrows():
        local_step = np.array([row[step_x], row[step_y]], dtype=float)
        global_step = rotation_matrix(yaw_rad) @ local_step
        position = position + global_step
        yaw_increment = float(row[yaw_col])
        yaw_rad += math.radians(yaw_increment)

        rows.append(
            {
                "variant": variant,
                "deployable_online": deployable,
                "sequence_frame_id": int(row["frame_index_b"]),
                "visual_x_px": float(position[0]),
                "visual_y_px": float(position[1]),
                "visual_yaw_rad": float(yaw_rad),
                "visual_yaw_deg_unwrapped": float(math.degrees(yaw_rad)),
                "step_x_local_px": float(local_step[0]),
                "step_y_local_px": float(local_step[1]),
                "step_x_global_px": float(global_step[0]),
                "step_y_global_px": float(global_step[1]),
                "step_motion_px": float(np.linalg.norm(global_step)),
                "yaw_increment_deg": yaw_increment,
                "image_only_confidence": float(row["image_only_confidence"]),
                "pair_safe_image_only": str(row["status"]) == "good",
                "delta_from_raw_step_px": (
                    0.0 if delta_col is None else float(row[delta_col])
                ),
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
    differences = np.diff(np.asarray(points, dtype=float), axis=0)
    return float(np.sum(np.linalg.norm(differences, axis=1)))


def error_summary(errors: np.ndarray, reference_distance: np.ndarray, evaluation_start_index: int) -> dict[str, Any]:
    tail_errors = np.asarray(errors[evaluation_start_index:], dtype=float)
    tail_distance = np.asarray(reference_distance[evaluation_start_index:], dtype=float)
    valid = np.isfinite(tail_errors)
    tail_errors = tail_errors[valid]
    tail_distance = tail_distance[valid]

    if len(tail_errors) == 0:
        return {}

    distance_from_evaluation_start = tail_distance - tail_distance[0]
    final_distance = float(distance_from_evaluation_start[-1])
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
            100.0 * final_error / final_distance
            if final_distance > 1e-9
            else float("nan")
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
        raise RuntimeError(f"{variant}: trajectory did not merge to all 1,034 frames.")

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
        "deployable_online": bool(trajectory["deployable_online"].iloc[0]),
        "frames": int(len(output)),
        "reference_path_m": trajectory_path_length(reference),
        "visual_path_px": trajectory_path_length(visual),
        "safe_pair_rate": float(output["pair_safe_image_only"].iloc[1:].astype(bool).mean()),
        "mean_delta_from_raw_step_px": finite_stat(output["delta_from_raw_step_px"].iloc[1:], np.mean),
        "median_delta_from_raw_step_px": finite_stat(output["delta_from_raw_step_px"].iloc[1:], np.median),
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
    summary_path: Path,
    aligned_path: Path,
    runtime_path: Path,
    orb_pairs: pd.DataFrame,
) -> dict[str, Any]:
    summary = pd.read_csv(summary_path)
    selected = summary[summary["variant"].astype(str) == "se2_scale_normalized"]
    if selected.empty:
        raise RuntimeError("Frozen ORB summary has no se2_scale_normalized row.")
    row = selected.iloc[0]

    aligned = pd.read_csv(aligned_path)
    aligned = aligned[aligned["variant"].astype(str) == "se2_scale_normalized"].sort_values("sequence_frame_id")
    prefix_tail = aligned.iloc[ALIGNMENT_PREFIX_FRAMES - 1 :]
    failure_rate = float((pd.to_numeric(prefix_tail["prefix_locked_error_m"], errors="coerce") > 40.0).mean())

    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    feature_seconds = float(runtime.get("feature_cache_seconds", 0.0))
    matching_seconds = float(pd.to_numeric(orb_pairs["elapsed_ms"], errors="coerce").sum() / 1000.0)
    total_seconds = feature_seconds + matching_seconds

    return {
        "method": "orb_frozen_s6a",
        "source": "frozen_s6a",
        "deployable_online": True,
        "frames": EXPECTED_FRAMES,
        "pairs": EXPECTED_PAIRS,
        "affine_success_rate": float(orb_pairs["affine_ok"].mean()),
        "good_quality_rate": float((orb_pairs["status"].astype(str) == "good").mean()),
        "inlier_ratio_median": finite_stat(orb_pairs["inlier_ratio"], np.median),
        "inlier_ratio_p05": finite_stat(orb_pairs["inlier_ratio"], lambda values: np.percentile(values, 5)),
        "feature_seconds": feature_seconds,
        "matching_ransac_seconds": matching_seconds,
        "total_frontend_seconds": total_seconds,
        "total_frontend_ms_per_pair": 1000.0 * total_seconds / EXPECTED_PAIRS,
        "prefix_locked_rmse_m": float(row["prefix_locked_alignment.rmse_m"]),
        "prefix_locked_p95_m": float(row["prefix_locked_alignment.p95_error_m"]),
        "prefix_locked_max_m": float(row["prefix_locked_alignment.max_error_m"]),
        "final_error_m": float(row["prefix_locked_alignment.final_error_m"]),
        "final_drift_per_100m": float(row["prefix_locked_alignment.final_drift_per_100m"]),
        "failure_rate_gt40m": failure_rate,
    }


def method_summary_from_variant(
    summary: dict[str, Any],
    increments: pd.DataFrame,
    backend_runtime_seconds: float,
) -> dict[str, Any]:
    prefix = summary["prefix_locked_alignment"]
    return {
        "method": summary["variant"],
        "source": "s7b1_backend",
        "deployable_online": bool(summary["deployable_online"]),
        "frames": EXPECTED_FRAMES,
        "pairs": EXPECTED_PAIRS,
        "affine_success_rate": float(increments["affine_ok"].mean()),
        "good_quality_rate": float((increments["status"].astype(str) == "good").mean()),
        "inlier_ratio_median": finite_stat(increments["inlier_ratio"], np.median),
        "inlier_ratio_p05": finite_stat(increments["inlier_ratio"], lambda values: np.percentile(values, 5)),
        "feature_seconds": 0.0,
        "matching_ransac_seconds": 0.0,
        "total_frontend_seconds": float(backend_runtime_seconds),
        "total_frontend_ms_per_pair": 1000.0 * backend_runtime_seconds / EXPECTED_PAIRS,
        "prefix_locked_rmse_m": float(prefix["rmse_m"]),
        "prefix_locked_p95_m": float(prefix["p95_error_m"]),
        "prefix_locked_max_m": float(prefix["max_error_m"]),
        "final_error_m": float(prefix["final_error_m"]),
        "final_drift_per_100m": float(prefix["final_drift_per_100m"]),
        "failure_rate_gt40m": float(prefix["failure_rate_gt40m"]),
        "mean_delta_from_raw_step_px": float(summary["mean_delta_from_raw_step_px"]),
        "median_delta_from_raw_step_px": float(summary["median_delta_from_raw_step_px"]),
    }


def append_prior_method_rows(rows: list[dict[str, Any]], path: Path, methods: set[str], source: str) -> None:
    if not path.exists():
        return
    try:
        frame = pd.read_csv(path)
    except Exception:
        return
    if "method" not in frame.columns:
        return
    for _, row in frame.iterrows():
        method = str(row["method"])
        if method not in methods:
            continue
        payload = row.to_dict()
        payload["source"] = source
        rows.append(payload)


def build_scene_backend_summary(increments: pd.DataFrame) -> pd.DataFrame:
    subset = increments[increments["in_stratified_diagnostic_subset"]].copy()
    if subset.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for scene, group in subset.groupby("primary_scene", sort=True):
        rows.append(
            {
                "primary_scene": scene,
                "pairs": int(len(group)),
                "confidence_median": finite_stat(group["image_only_confidence"], np.median),
                "confidence_p05": finite_stat(group["image_only_confidence"], lambda values: np.percentile(values, 5)),
                "raw_step_motion_median_px": finite_stat(group["raw_step_motion_px"], np.median),
                "raw_abs_yaw_median_deg": finite_stat(np.abs(group["raw_yaw_increment_deg"]), np.median),
                "causal_step_delta_median_px": finite_stat(group["causal_delta_from_raw_step_px"], np.median),
                "window9_step_delta_median_px": finite_stat(group["window9_delta_from_raw_step_px"], np.median),
                "inlier_ratio_median": finite_stat(group["inlier_ratio"], np.median),
            }
        )
    return pd.DataFrame(rows)


def decide_backend(
    orb_baseline: dict[str, Any],
    comparison_df: pd.DataFrame,
) -> dict[str, Any]:
    candidate = comparison_df[
        comparison_df["method"].astype(str) == "orb_confidence_ewma_causal"
    ].iloc[0].to_dict()
    diagnostic = comparison_df[
        comparison_df["method"].astype(str) == "orb_window9_weighted_smooth"
    ].iloc[0].to_dict()

    rmse_improvement = (
        orb_baseline["prefix_locked_rmse_m"] - float(candidate["prefix_locked_rmse_m"])
    ) / max(orb_baseline["prefix_locked_rmse_m"], 1e-12)
    p95_ratio = float(candidate["prefix_locked_p95_m"]) / max(orb_baseline["prefix_locked_p95_m"], 1e-12)
    max_ratio = float(candidate["prefix_locked_max_m"]) / max(orb_baseline["prefix_locked_max_m"], 1e-12)
    failure_reduction = (
        orb_baseline["failure_rate_gt40m"] - float(candidate["failure_rate_gt40m"])
    ) / max(orb_baseline["failure_rate_gt40m"], 1e-12)

    diagnostic_rmse_improvement = (
        orb_baseline["prefix_locked_rmse_m"] - float(diagnostic["prefix_locked_rmse_m"])
    ) / max(orb_baseline["prefix_locked_rmse_m"], 1e-12)

    robustness_guard = bool(p95_ratio <= 1.02 and max_ratio <= 1.02)
    material_benefit = bool(rmse_improvement >= 0.05 or failure_reduction >= 0.20)
    promote = bool(robustness_guard and material_benefit)

    return {
        "decision": "PROMOTE_ORB_EWMA_BACKEND" if promote else "KEEP_ORB",
        "promote_orb_backend": promote,
        "deployable_candidate": "orb_confidence_ewma_causal",
        "diagnostic_offline_variant": "orb_window9_weighted_smooth",
        "comparison_values": {
            "causal_rmse_improvement_fraction": float(rmse_improvement),
            "causal_p95_ratio_over_orb": float(p95_ratio),
            "causal_max_ratio_over_orb": float(max_ratio),
            "causal_failure_rate_reduction_fraction": float(failure_reduction),
            "diagnostic_window9_rmse_improvement_fraction": float(diagnostic_rmse_improvement),
        },
        "guards": {
            "robustness_guard_pass": robustness_guard,
            "material_benefit_pass": material_benefit,
            "promotion_rule": (
                "Promote only the causal online smoother when it improves RMSE by "
                "at least 5% or failure rate by at least 20%, while keeping p95 "
                "and max error within 2% of frozen ORB."
            ),
        },
        "interpretation_rule": (
            "If the centered window helps but causal smoothing does not, treat it "
            "as evidence that multi-frame optimization may help, not as a deployable "
            "promotion."
        ),
        "closeout_rule": (
            "S7B.1 is backend-only and bounded. No smoothing hyperparameter sweep follows."
        ),
    }


def save_plots(
    aligned_all: pd.DataFrame,
    increments: pd.DataFrame,
    orb_aligned_path: Path,
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
        label="frozen ORB",
    )
    for variant, group in aligned_all.groupby("variant", sort=True):
        plt.plot(
            group["reference_cumulative_distance_m"],
            group["prefix_locked_error_m"],
            label=variant,
        )
    for threshold in ERROR_THRESHOLDS_M:
        plt.axhline(threshold, linestyle="--", linewidth=1.0, label=f"{threshold:g} m")
    plt.xlabel("Reference cumulative distance [m] — evaluation only")
    plt.ylabel("Prefix-locked error [m]")
    plt.title("S7B.1 ORB backend smoothing error growth")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "s7b1_error_vs_distance.png", dpi=180)
    plt.close()

    plt.figure(figsize=(9, 8))
    plt.plot(
        orb_aligned["reference_x_m"],
        orb_aligned["reference_y_m"],
        label="Reference — evaluation only",
    )
    plt.plot(
        orb_aligned["prefix_aligned_x_m"],
        orb_aligned["prefix_aligned_y_m"],
        label="frozen ORB",
    )
    for variant in ["orb_confidence_ewma_causal", "orb_window9_weighted_smooth"]:
        selected = aligned_all[aligned_all["variant"] == variant]
        if not selected.empty:
            plt.plot(
                selected["prefix_aligned_x_m"],
                selected["prefix_aligned_y_m"],
                label=variant,
            )
    plt.axis("equal")
    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")
    plt.title("S7B.1 prefix-locked XY trajectory")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "s7b1_prefix_locked_xy.png", dpi=180)
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.plot(
        increments["pair_number"],
        increments["image_only_confidence"],
        label="image-only confidence",
    )
    plt.plot(
        increments["pair_number"],
        increments["inlier_ratio"],
        label="ORB inlier ratio",
    )
    plt.xlabel("Stride-1 pair number")
    plt.ylabel("Score")
    plt.title("S7B.1 ORB confidence used for smoothing")
    plt.ylim(0.0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "s7b1_confidence_signal.png", dpi=180)
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.plot(
        increments["pair_number"],
        increments["causal_delta_from_raw_step_px"],
        label="causal EWMA delta",
    )
    plt.plot(
        increments["pair_number"],
        increments["window9_delta_from_raw_step_px"],
        label="window-9 delta",
    )
    plt.xlabel("Stride-1 pair number")
    plt.ylabel("Change from raw ORB step [px]")
    plt.title("S7B.1 smoothing magnitude")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "s7b1_smoothing_delta.png", dpi=180)
    plt.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--window", type=int, default=9)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    output_root = resolve_from_repo(repo_root, args.output_root)

    if args.window != 9:
        raise RuntimeError("S7B.1 is frozen to a centered 9-pair window.")

    paths = {
        "sequence": resolve_from_repo(repo_root, DEFAULT_SEQUENCE_MANIFEST),
        "orb_pairs": resolve_from_repo(repo_root, DEFAULT_ORB_PAIRS),
        "orb_features": resolve_from_repo(repo_root, DEFAULT_ORB_FEATURES),
        "orb_aligned": resolve_from_repo(repo_root, DEFAULT_ORB_ALIGNED),
        "orb_summary": resolve_from_repo(repo_root, DEFAULT_ORB_SUMMARY),
        "orb_runtime": resolve_from_repo(repo_root, DEFAULT_ORB_RUNTIME),
        "full_pair_manifest": resolve_from_repo(repo_root, DEFAULT_FULL_PAIR_MANIFEST),
        "xfeat_comparison": resolve_from_repo(repo_root, DEFAULT_XFEAT_COMPARISON),
        "flow_comparison": resolve_from_repo(repo_root, DEFAULT_FLOW_COMPARISON),
        "flow_decision": resolve_from_repo(repo_root, DEFAULT_FLOW_DECISION),
    }

    for label in ["sequence", "orb_pairs", "orb_features", "orb_aligned", "orb_summary", "orb_runtime"]:
        if not paths[label].exists():
            raise FileNotFoundError(f"Missing required {label}: {paths[label]}")

    stage_start = time.perf_counter()

    sequence = load_sequence_manifest(paths["sequence"])
    orb_pairs = load_orb_pairs(paths["orb_pairs"])
    width, height = load_feature_dimensions(paths["orb_features"])
    labels = load_full_pair_labels(paths["full_pair_manifest"])

    increments = build_orb_increment_table(
        orb_pairs=orb_pairs,
        width=width,
        height=height,
        full_pair_labels=labels,
    )
    increments = apply_causal_confidence_ewma(increments)
    increments = apply_window9_weighted_smooth(increments, window=args.window)

    raw_trajectories: list[pd.DataFrame] = []
    aligned_trajectories: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []
    crossing_rows: list[dict[str, Any]] = []

    for variant in [
        "orb_only_recomputed",
        "orb_confidence_ewma_causal",
        "orb_window9_weighted_smooth",
    ]:
        raw = integrate_increment_variant(increments, variant)
        aligned, summary, crossings = evaluate_trajectory(raw, sequence)
        raw_trajectories.append(raw)
        aligned_trajectories.append(aligned)
        summaries.append(summary)
        crossing_rows.extend([{"variant": variant, **row} for row in crossings])

    backend_runtime_seconds = time.perf_counter() - stage_start

    raw_all = pd.concat(raw_trajectories, ignore_index=True)
    aligned_all = pd.concat(aligned_trajectories, ignore_index=True)
    summary_df = pd.json_normalize(summaries, sep=".")

    orb_baseline = load_orb_baseline(
        paths["orb_summary"],
        paths["orb_aligned"],
        paths["orb_runtime"],
        orb_pairs,
    )

    comparison_rows: list[dict[str, Any]] = [orb_baseline]
    append_prior_method_rows(comparison_rows, paths["xfeat_comparison"], {"xfeat"}, "s7a2_full")
    append_prior_method_rows(
        comparison_rows,
        paths["flow_comparison"],
        {"klt_flow_only", "orb_flow_fallback", "orb_flow_guarded_blend"},
        "s7b0_orb_flow",
    )
    for summary in summaries:
        comparison_rows.append(
            method_summary_from_variant(
                summary,
                increments,
                backend_runtime_seconds=backend_runtime_seconds,
            )
        )

    comparison_df = pd.DataFrame(comparison_rows)
    decision = decide_backend(orb_baseline, comparison_df)
    scene_summary = build_scene_backend_summary(increments)

    metadata_dir = ensure_dir(output_root / "metadata" / "s7_relative_frontend")
    reports_dir = ensure_dir(output_root / "reports" / "s7_relative_frontend")
    figures_dir = ensure_dir(output_root / "figures" / "s7_relative_frontend" / "s7b1_orb_backend")

    increment_path = metadata_dir / "s7b_1_orb_increment_smoothing_table.csv"
    raw_path = metadata_dir / "s7b_1_relative_trajectory_pixels.csv"
    aligned_path = metadata_dir / "s7b_1_relative_trajectory_aligned_eval_only.csv"
    summary_path = metadata_dir / "s7b_1_trajectory_summary.csv"
    method_comparison_path = metadata_dir / "s7b_1_method_comparison.csv"
    scene_summary_path = metadata_dir / "s7b_1_scene_backend_summary.csv"
    crossing_path = metadata_dir / "s7b_1_drift_threshold_crossings.csv"
    decision_path = metadata_dir / "s7b_1_backend_decision.json"
    json_report_path = reports_dir / "s7b_1_orb_sliding_window_se2_smoothing.json"
    md_report_path = reports_dir / "s7b_1_orb_sliding_window_se2_smoothing_report.md"

    increments.to_csv(increment_path, index=False)
    raw_all.to_csv(raw_path, index=False)
    aligned_all.to_csv(aligned_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    comparison_df.to_csv(method_comparison_path, index=False)
    scene_summary.to_csv(scene_summary_path, index=False)
    pd.DataFrame(crossing_rows).to_csv(crossing_path, index=False)
    decision_path.write_text(json.dumps(decision, indent=2), encoding="utf-8")

    save_plots(aligned_all, increments, paths["orb_aligned"], figures_dir)

    candidate = comparison_df[
        comparison_df["method"].astype(str) == "orb_confidence_ewma_causal"
    ].iloc[0].to_dict()
    diagnostic = comparison_df[
        comparison_df["method"].astype(str) == "orb_window9_weighted_smooth"
    ].iloc[0].to_dict()

    payload = {
        "generated_utc": utc_now(),
        "stage": "S7B.1",
        "status": "COMPLETE_" + decision["decision"],
        "scientific_scope": (
            "Backend-only ORB SE(2) increment smoothing study. Frozen ORB "
            "pairwise affines are reused; no frontend matching is rerun."
        ),
        "configuration": {
            "variants": [
                "orb_only_recomputed",
                "orb_confidence_ewma_causal",
                "orb_window9_weighted_smooth",
            ],
            "causal_alpha_rule": "alpha = 0.55 + 0.40 * image_only_confidence",
            "image_only_confidence": (
                "Weighted ORB ratio/inlier/match/scale/status score; no reference error."
            ),
            "centered_window": args.window,
            "alignment_prefix_frames": ALIGNMENT_PREFIX_FRAMES,
            "error_thresholds_m": ERROR_THRESHOLDS_M,
            "sustain_frames": SUSTAIN_FRAMES,
        },
        "counts": {
            "frames": EXPECTED_FRAMES,
            "pairs": EXPECTED_PAIRS,
            "diagnostic_scene_pairs": int(increments["in_stratified_diagnostic_subset"].sum()),
        },
        "timing": {
            "backend_runtime_seconds": float(backend_runtime_seconds),
            "backend_ms_per_pair": float(1000.0 * backend_runtime_seconds / EXPECTED_PAIRS),
        },
        "orb_baseline": orb_baseline,
        "candidate_causal": candidate,
        "diagnostic_window9": diagnostic,
        "decision": decision,
        "ground_truth_rule": (
            "Reference ENU is used only after each image-only trajectory is integrated."
        ),
        "git": {
            "branch": git_value(repo_root, "branch", "--show-current"),
            "commit": git_value(repo_root, "rev-parse", "HEAD"),
            "working_tree_porcelain": git_value(repo_root, "status", "--porcelain"),
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
        },
        "outputs": {
            "increment_smoothing_table": str(increment_path.relative_to(repo_root)),
            "raw_trajectories": str(raw_path.relative_to(repo_root)),
            "aligned_trajectories": str(aligned_path.relative_to(repo_root)),
            "trajectory_summary": str(summary_path.relative_to(repo_root)),
            "method_comparison": str(method_comparison_path.relative_to(repo_root)),
            "scene_summary": str(scene_summary_path.relative_to(repo_root)),
            "crossings": str(crossing_path.relative_to(repo_root)),
            "decision": str(decision_path.relative_to(repo_root)),
            "figures": str(figures_dir.relative_to(repo_root)),
            "report": str(md_report_path.relative_to(repo_root)),
        },
    }
    json_report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    selected_columns = [
        "method",
        "source",
        "deployable_online",
        "prefix_locked_rmse_m",
        "prefix_locked_p95_m",
        "prefix_locked_max_m",
        "final_error_m",
        "final_drift_per_100m",
        "failure_rate_gt40m",
        "total_frontend_ms_per_pair",
    ]
    present_columns = [column for column in selected_columns if column in comparison_df.columns]
    method_table = comparison_df[present_columns].to_markdown(index=False)
    scene_table = (
        scene_summary.to_markdown(index=False)
        if not scene_summary.empty
        else "No stratified scene labels were available."
    )

    md_report = f"""# S7B.1 — ORB Sliding-Window SE(2) Pose Smoothing

Generated: `{payload["generated_utc"]}`

## Status

```text
{payload["status"]}
```

## What was tested

This is a backend-only experiment. The frozen ORB pair transforms from S6A are
converted to local SE(2) increments and smoothed without rerunning the frontend.

```text
Frames:                {EXPECTED_FRAMES}
Pairs:                 {EXPECTED_PAIRS}
Online candidate:      orb_confidence_ewma_causal
Diagnostic variant:    orb_window9_weighted_smooth
Centered window:       {args.window} pairs
Alignment prefix:      {ALIGNMENT_PREFIX_FRAMES} frames
```

## Main comparison

{method_table}

## Scene/backend diagnostic summary

{scene_table}

## Decision

```text
{decision["decision"]}
```

Only the causal EWMA smoother is eligible for promotion. The centered window is
reported as diagnostic evidence because it uses future increments.

## Outputs

```text
{increment_path.relative_to(repo_root)}
{method_comparison_path.relative_to(repo_root)}
{decision_path.relative_to(repo_root)}
{figures_dir.relative_to(repo_root)}
```
"""
    md_report_path.write_text(md_report, encoding="utf-8")

    print("S7B.1 ORB Sliding-Window SE(2) Pose Smoothing")
    print("---------------------------------------------")
    print(f"Status:                         {payload['status']}")
    print(f"Frames:                         {EXPECTED_FRAMES}")
    print(f"Pairs:                          {EXPECTED_PAIRS}")
    print(f"Diagnostic scene pairs:         {int(increments['in_stratified_diagnostic_subset'].sum())}")
    print(f"ORB prefix RMSE m:              {orb_baseline['prefix_locked_rmse_m']:.3f}")
    print(f"Causal EWMA prefix RMSE m:      {float(candidate['prefix_locked_rmse_m']):.3f}")
    print(f"Window9 prefix RMSE m:          {float(diagnostic['prefix_locked_rmse_m']):.3f}")
    print(f"ORB prefix p95 m:               {orb_baseline['prefix_locked_p95_m']:.3f}")
    print(f"Causal EWMA prefix p95 m:       {float(candidate['prefix_locked_p95_m']):.3f}")
    print(f"Window9 prefix p95 m:           {float(diagnostic['prefix_locked_p95_m']):.3f}")
    print(f"ORB final drift m/100m:         {orb_baseline['final_drift_per_100m']:.3f}")
    print(f"Causal EWMA final drift m/100m: {float(candidate['final_drift_per_100m']):.3f}")
    print(f"Window9 final drift m/100m:     {float(diagnostic['final_drift_per_100m']):.3f}")
    print(f"Backend ms/pair:                {1000.0 * backend_runtime_seconds / EXPECTED_PAIRS:.3f}")
    print(f"Decision:                       {decision['decision']}")
    print(f"Increment table:                {increment_path.relative_to(repo_root)}")
    print(f"Method comparison:              {method_comparison_path.relative_to(repo_root)}")
    print(f"Scene summary:                  {scene_summary_path.relative_to(repo_root)}")
    print(f"Decision JSON:                  {decision_path.relative_to(repo_root)}")
    print(f"Figures:                        {figures_dir.relative_to(repo_root)}")
    print(f"Report:                         {md_report_path.relative_to(repo_root)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print("S7B.1 ORB Sliding-Window SE(2) Pose Smoothing", file=sys.stderr)
        print("---------------------------------------------", file=sys.stderr)
        print("Status: BLOCKED", file=sys.stderr)
        print(f"Reason: {exc}", file=sys.stderr)
        raise
