#!/usr/bin/env python3
"""S7A.2 — full traj01 XFeat relative-frontend comparison and closeout.

Run exactly one full-sequence XFeat experiment after S7A.1 passes.

Frozen rules:
- official VERLab XFeat checkout at the pinned commit;
- sparse extraction, 960-pixel long side, top_k=1200;
- detection threshold 0.05 and MNN cosine threshold 0.82;
- all 1,033 consecutive traj01 pairs;
- partial-affine RANSAC at 3 px;
- SE(2) scale-normalized trajectory integration;
- 50-frame prefix evaluation alignment;
- reference ENU used only after trajectory construction. 

The script streams through the sequence so each frame is extracted once. It writes
the full XFeat diagnostics, trajectory, ORB/XFeat comparison, scene diagnostics,
plots, and a final PROMOTE_XFEAT or KEEP_ORB decision. No parameter sweep follows.

Command Used:

source .drone_venv/bin/activate
export PYTHONPATH=$PWD/src

mkdir -p outputs/satloc/reports/s7_relative_frontend
set -o pipefail

python scripts/satloc/s7/s7a_2_xfeat_full_trajectory_comparison.py \
  --device cpu \
  2>&1 | tee \
  outputs/satloc/reports/s7_relative_frontend/s7a_2_xfeat_full_comparison.log
  
"""

from __future__ import annotations

import argparse
import importlib
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


OFFICIAL_XFEAT_REPOSITORY = "https://github.com/verlab/accelerated_features.git"
PINNED_XFEAT_COMMIT = "e92685f57f8318b18725c5c8c0bd28c7fe188d9a"

EXPECTED_FRAMES = 1034
EXPECTED_PAIRS = 1033
ALIGNMENT_PREFIX_FRAMES = 50
ERROR_THRESHOLDS_M = [10.0, 20.0, 40.0, 80.0]
SUSTAIN_FRAMES = 5

DEFAULT_OUTPUT_ROOT = Path("outputs/satloc")
DEFAULT_XFEAT_REPO = Path("third_party/accelerated_features")
DEFAULT_SEQUENCE_MANIFEST = Path(
    "outputs/satloc/metadata/s6a_relative_motion/s6a_sequence_manifest.csv"
)
DEFAULT_ORB_PAIRS = Path(
    "outputs/satloc/metadata/s6a_relative_motion/"
    "s6a1_orb_affine_pair_diagnostics.csv"
)
DEFAULT_ORB_TRAJECTORY_SUMMARY = Path(
    "outputs/satloc/metadata/s6a_relative_motion/"
    "s6a2_orb_relative_trajectory_summary.csv"
)
DEFAULT_ORB_ALIGNED_TRAJECTORY = Path(
    "outputs/satloc/metadata/s6a_relative_motion/"
    "s6a2_orb_relative_trajectory_aligned_eval_only.csv"
)
DEFAULT_ORB_RUNTIME_REPORT = Path(
    "outputs/satloc/reports/s6a_relative_motion/"
    "s6a1_orb_affine_stride_summary.json"
)
DEFAULT_FULL_PAIR_MANIFEST = Path(
    "outputs/satloc/metadata/s7_relative_frontend/"
    "s7a_0_full_pair_manifest.csv"
)
DEFAULT_PROTOCOL = Path(
    "outputs/satloc/metadata/s7_relative_frontend/"
    "s7a_0_protocol_manifest.json"
)
DEFAULT_SMOKE_SUMMARY = Path(
    "outputs/satloc/reports/s7_relative_frontend/"
    "s7a_1_xfeat_sparse_smoke.json"
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


def synchronize(torch_module: Any, device: Any) -> None:
    if str(device).startswith("cuda") and torch_module.cuda.is_available():
        torch_module.cuda.synchronize(device)


def peak_process_memory_mb() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if platform.system() == "Darwin":
        return value / (1024.0 * 1024.0)
    return value / 1024.0


def resize_long_side(image: np.ndarray, target: int) -> np.ndarray:
    height, width = image.shape[:2]
    long_side = max(height, width)
    if target <= 0 or long_side == target:
        return image
    scale = target / float(long_side)
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


def bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes", "y", "t"})
    )


def finite_stat(values: pd.Series | np.ndarray, function) -> float:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    return float(function(array)) if len(array) else float("nan")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_sequence_manifest(repo_root: Path, path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {
        "sequence_frame_id",
        "token0_id",
        "x_enu_m",
        "y_enu_m",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"Sequence manifest missing columns: {missing}")

    image_column = None
    for candidate in ("image_path_resolved", "image_path", "resolved_image_path"):
        if candidate in frame.columns:
            image_column = candidate
            break
    if image_column is None:
        raise RuntimeError("Sequence manifest has no usable image-path column.")

    frame["sequence_frame_id"] = pd.to_numeric(
        frame["sequence_frame_id"], errors="raise"
    ).astype(int)
    frame["token0_id"] = pd.to_numeric(
        frame["token0_id"], errors="raise"
    ).astype(int)
    frame["x_enu_m"] = pd.to_numeric(frame["x_enu_m"], errors="raise")
    frame["y_enu_m"] = pd.to_numeric(frame["y_enu_m"], errors="raise")
    frame = frame.sort_values(
        "sequence_frame_id", kind="mergesort"
    ).reset_index(drop=True)

    if len(frame) != EXPECTED_FRAMES:
        raise RuntimeError(
            f"Expected {EXPECTED_FRAMES} frames, found {len(frame)}."
        )
    if not np.array_equal(
        frame["sequence_frame_id"].to_numpy(dtype=int),
        np.arange(EXPECTED_FRAMES),
    ):
        raise RuntimeError("sequence_frame_id is not contiguous 0..1033.")
    if not np.array_equal(
        frame["token0_id"].to_numpy(dtype=int),
        np.arange(1, EXPECTED_FRAMES + 1),
    ):
        raise RuntimeError("token0_id is not the canonical range 1..1034.")

    frame["image_path_full_resolved"] = frame[image_column].map(
        lambda value: str(resolve_image_path(repo_root, value))
    )
    missing_paths = [
        Path(value)
        for value in frame["image_path_full_resolved"]
        if not Path(value).exists()
    ]
    if missing_paths:
        raise FileNotFoundError(
            f"{len(missing_paths)} UAV images are missing. "
            f"First missing: {missing_paths[0]}"
        )
    return frame


def load_full_pair_manifest(path: Path) -> pd.DataFrame:
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
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"Full pair manifest missing columns: {missing}")

    numeric = [
        "comparison_pair_id",
        "pair_number",
        "frame_index_a",
        "frame_index_b",
        "token0_a",
        "token0_b",
    ]
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype(int)
    frame["in_stratified_diagnostic_subset"] = bool_series(
        frame["in_stratified_diagnostic_subset"]
    )
    frame = frame.sort_values(
        "comparison_pair_id", kind="mergesort"
    ).reset_index(drop=True)

    if len(frame) != EXPECTED_PAIRS:
        raise RuntimeError(
            f"Expected {EXPECTED_PAIRS} full pairs, found {len(frame)}."
        )
    if not np.array_equal(
        frame["frame_index_a"].to_numpy(dtype=int),
        np.arange(EXPECTED_PAIRS),
    ):
        raise RuntimeError("Full pair frame_index_a chain is invalid.")
    if not np.array_equal(
        frame["frame_index_b"].to_numpy(dtype=int),
        np.arange(1, EXPECTED_FRAMES),
    ):
        raise RuntimeError("Full pair frame_index_b chain is invalid.")
    return frame


def load_official_xfeat(
    xfeat_repo: Path,
    device_name: str,
    top_k: int,
    detection_threshold: float,
) -> tuple[Any, Any, Any, str]:
    module_path = xfeat_repo / "modules" / "xfeat.py"
    weights_path = xfeat_repo / "weights" / "xfeat.pt"
    if not module_path.exists():
        raise FileNotFoundError(f"Official XFeat module not found: {module_path}")
    if not weights_path.exists():
        raise FileNotFoundError(f"Official XFeat weights not found: {weights_path}")

    sys.path.insert(0, str(xfeat_repo))
    for module_name in list(sys.modules):
        if module_name == "modules" or module_name.startswith("modules."):
            del sys.modules[module_name]

    try:
        torch = importlib.import_module("torch")
        XFeat = getattr(importlib.import_module("modules.xfeat"), "XFeat")
    except Exception as exc:
        raise RuntimeError(
            f"Could not import official XFeat from {xfeat_repo}: {exc}"
        ) from exc

    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is unavailable.")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = XFeat(
        weights=str(weights_path),
        top_k=top_k,
        detection_threshold=detection_threshold,
    )
    model.dev = device
    model.net = model.net.to(device).eval()
    model.eval()

    return torch, model, device, git_value(xfeat_repo, "rev-parse", "HEAD")


def extract_one(
    torch: Any,
    model: Any,
    device: Any,
    image_path: Path,
    resize_long: int,
    top_k: int,
    detection_threshold: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise RuntimeError(f"OpenCV could not read: {image_path}")
    gray = resize_long_side(gray, resize_long)

    tensor = (
        torch.from_numpy(gray)
        .to(device=device, dtype=torch.float32)
        .unsqueeze(0)
        .unsqueeze(0)
        / 255.0
    )

    synchronize(torch, device)
    start = time.perf_counter()
    output = model.detectAndCompute(
        tensor,
        top_k=top_k,
        detection_threshold=detection_threshold,
    )[0]
    synchronize(torch, device)
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    # Store on CPU so the full sequence never accumulates device memory.
    keypoints = output["keypoints"].detach().cpu()
    descriptors = output["descriptors"].detach().cpu()
    scores = output["scores"].detach().cpu()

    feature = {
        "keypoints": keypoints,
        "descriptors": descriptors,
        "scores": scores,
        "width": int(gray.shape[1]),
        "height": int(gray.shape[0]),
    }
    metadata = {
        "keypoints": int(len(keypoints)),
        "score_median": (
            float(scores.median().item()) if len(scores) else float("nan")
        ),
        "width": int(gray.shape[1]),
        "height": int(gray.shape[0]),
        "feature_time_ms": float(elapsed_ms),
        "read_ok": True,
    }
    return feature, metadata


def match_and_estimate(
    torch: Any,
    model: Any,
    device: Any,
    feature_a: dict[str, Any],
    feature_b: dict[str, Any],
    min_cossim: float,
    ransac_threshold: float,
) -> dict[str, Any]:
    descriptors_a = feature_a["descriptors"].to(device)
    descriptors_b = feature_b["descriptors"].to(device)

    if len(descriptors_a) == 0 or len(descriptors_b) == 0:
        return {
            "status": "no_descriptors",
            "matches": 0,
            "inliers": 0,
            "inlier_ratio": 0.0,
            "affine_ok": False,
            "matching_ransac_time_ms": 0.0,
        }

    synchronize(torch, device)
    start = time.perf_counter()
    idx_a, idx_b = model.match(
        descriptors_a,
        descriptors_b,
        min_cossim=min_cossim,
    )
    synchronize(torch, device)

    points_a = (
        feature_a["keypoints"][idx_a.detach().cpu()]
        .numpy()
        .astype(np.float32)
        if len(idx_a)
        else np.empty((0, 2), dtype=np.float32)
    )
    points_b = (
        feature_b["keypoints"][idx_b.detach().cpu()]
        .numpy()
        .astype(np.float32)
        if len(idx_b)
        else np.empty((0, 2), dtype=np.float32)
    )

    if len(points_a) < 3:
        return {
            "status": "too_few_matches",
            "matches": int(len(points_a)),
            "inliers": 0,
            "inlier_ratio": 0.0,
            "affine_ok": False,
            "matching_ransac_time_ms": float(
                (time.perf_counter() - start) * 1000.0
            ),
        }

    affine, inlier_mask = cv2.estimateAffinePartial2D(
        points_a,
        points_b,
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_threshold,
        maxIters=3000,
        confidence=0.995,
        refineIters=10,
    )
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    if affine is None or inlier_mask is None:
        return {
            "status": "affine_failed",
            "matches": int(len(points_a)),
            "inliers": 0,
            "inlier_ratio": 0.0,
            "affine_ok": False,
            "matching_ransac_time_ms": float(elapsed_ms),
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

    center = np.array(
        [
            feature_a["width"] / 2.0,
            feature_a["height"] / 2.0,
            1.0,
        ],
        dtype=float,
    )
    mapped_center = affine @ center
    center_dx = float(mapped_center[0] - center[0])
    center_dy = float(mapped_center[1] - center[1])

    good = (
        len(points_a) >= 30
        and inliers >= 20
        and inlier_ratio >= 0.35
        and 0.70 <= scale <= 1.40
    )

    return {
        "status": "good" if good else "weak",
        "matches": int(len(points_a)),
        "inliers": inliers,
        "inlier_ratio": float(inlier_ratio),
        "affine_ok": True,
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
        "center_content_motion_px": float(
            math.hypot(center_dx, center_dy)
        ),
        "matching_ransac_time_ms": float(elapsed_ms),
    }


def run_streaming_frontend(
    torch: Any,
    model: Any,
    device: Any,
    sequence: pd.DataFrame,
    pair_manifest: pd.DataFrame,
    resize_long: int,
    top_k: int,
    detection_threshold: float,
    min_cossim: float,
    ransac_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame, float, float]:
    feature_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []

    wall_start = time.perf_counter()

    first_row = sequence.iloc[0]
    previous, previous_metadata = extract_one(
        torch,
        model,
        device,
        Path(first_row["image_path_full_resolved"]),
        resize_long,
        top_k,
        detection_threshold,
    )
    feature_rows.append(
        {
            "sequence_frame_id": 0,
            "token0_id": int(first_row["token0_id"]),
            **previous_metadata,
        }
    )

    for frame_index in range(1, len(sequence)):
        current_row = sequence.iloc[frame_index]
        current, current_metadata = extract_one(
            torch,
            model,
            device,
            Path(current_row["image_path_full_resolved"]),
            resize_long,
            top_k,
            detection_threshold,
        )
        feature_rows.append(
            {
                "sequence_frame_id": frame_index,
                "token0_id": int(current_row["token0_id"]),
                **current_metadata,
            }
        )

        pair = pair_manifest.iloc[frame_index - 1]
        result = match_and_estimate(
            torch,
            model,
            device,
            previous,
            current,
            min_cossim,
            ransac_threshold,
        )

        pair_rows.append(
            {
                "comparison_pair_id": int(pair["comparison_pair_id"]),
                "pair_number": int(pair["pair_number"]),
                "frame_index_a": int(pair["frame_index_a"]),
                "frame_index_b": int(pair["frame_index_b"]),
                "token0_a": int(pair["token0_a"]),
                "token0_b": int(pair["token0_b"]),
                "in_stratified_diagnostic_subset": bool(
                    pair["in_stratified_diagnostic_subset"]
                ),
                "primary_scene": (
                    str(pair["primary_scene"])
                    if pd.notna(pair.get("primary_scene"))
                    else ""
                ),
                "secondary_scene": (
                    str(pair["secondary_scene"])
                    if pd.notna(pair.get("secondary_scene"))
                    else ""
                ),
                "selection_role": (
                    str(pair["selection_role"])
                    if pd.notna(pair.get("selection_role"))
                    else ""
                ),
                "range_id": (
                    str(pair["range_id"])
                    if pd.notna(pair.get("range_id"))
                    else ""
                ),
                **result,
            }
        )

        previous = current

        if frame_index % 100 == 0 or frame_index == len(sequence) - 1:
            print(
                f"XFeat full chain: frame {frame_index + 1}/"
                f"{len(sequence)}, pairs {frame_index}/{EXPECTED_PAIRS}"
            )

    wall_seconds = time.perf_counter() - wall_start
    feature_df = pd.DataFrame(feature_rows)
    pair_df = pd.DataFrame(pair_rows)
    feature_seconds = float(feature_df["feature_time_ms"].sum() / 1000.0)
    return feature_df, pair_df, feature_seconds, wall_seconds


def affine_from_row(row: pd.Series) -> np.ndarray:
    return np.array(
        [
            [
                float(row["affine_a00"]),
                float(row["affine_a01"]),
                float(row["affine_tx_px"]),
            ],
            [
                float(row["affine_a10"]),
                float(row["affine_a11"]),
                float(row["affine_ty_px"]),
            ],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def normalize_affine_scale_about_center(
    matrix: np.ndarray,
    center: np.ndarray,
) -> np.ndarray:
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


def local_camera_step_from_scene_affine(
    scene_affine: np.ndarray,
    center: np.ndarray,
) -> np.ndarray:
    inverse = np.linalg.inv(scene_affine)
    camera_b_center_in_a = inverse @ center
    camera_b_center_in_a /= camera_b_center_in_a[2]
    step_image = camera_b_center_in_a[:2] - center[:2]
    return np.array([step_image[0], -step_image[1]], dtype=float)


def rotation_matrix(angle_rad: float) -> np.ndarray:
    cosine = math.cos(angle_rad)
    sine = math.sin(angle_rad)
    return np.array(
        [[cosine, -sine], [sine, cosine]],
        dtype=float,
    )


def integrate_se2_scale_normalized(
    pair_df: pd.DataFrame,
    width: int,
    height: int,
) -> pd.DataFrame:
    center = np.array([width / 2.0, height / 2.0, 1.0], dtype=float)
    position = np.zeros(2, dtype=float)
    yaw_rad = 0.0

    rows: list[dict[str, Any]] = [
        {
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
        }
    ]

    for _, pair in pair_df.iterrows():
        matrix = normalize_affine_scale_about_center(
            affine_from_row(pair),
            center,
        )
        local_step = local_camera_step_from_scene_affine(matrix, center)
        global_step = rotation_matrix(yaw_rad) @ local_step
        position = position + global_step

        yaw_rad += math.radians(float(pair["affine_rotation_deg"]))
        pair_safe = str(pair["status"]) == "good"

        rows.append(
            {
                "sequence_frame_id": int(pair["frame_index_b"]),
                "visual_x_px": float(position[0]),
                "visual_y_px": float(position[1]),
                "visual_yaw_rad": float(yaw_rad),
                "visual_yaw_deg_unwrapped": float(math.degrees(yaw_rad)),
                "step_x_local_px": float(local_step[0]),
                "step_y_local_px": float(local_step[1]),
                "step_x_global_px": float(global_step[0]),
                "step_y_global_px": float(global_step[1]),
                "step_motion_px": float(np.linalg.norm(global_step)),
                "pair_safe_image_only": bool(pair_safe),
            }
        )

    return pd.DataFrame(rows)


def fit_similarity(source: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    valid = (
        np.isfinite(source).all(axis=1)
        & np.isfinite(target).all(axis=1)
    )
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
        "rotation_deg": float(
            math.degrees(math.atan2(rotation[1, 0], rotation[0, 0]))
        ),
        "fit_points": int(len(source)),
    }


def apply_similarity(points: np.ndarray, transform: dict[str, Any]) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    return (
        transform["scale"] * (points @ transform["rotation"].T)
        + transform["translation"]
    )


def trajectory_path_length(points: np.ndarray) -> float:
    differences = np.diff(np.asarray(points, dtype=float), axis=0)
    return float(np.sum(np.linalg.norm(differences, axis=1)))


def error_summary(
    errors: np.ndarray,
    reference_distance: np.ndarray,
    evaluation_start_index: int,
) -> dict[str, Any]:
    tail_errors = np.asarray(errors[evaluation_start_index:], dtype=float)
    tail_distance = np.asarray(
        reference_distance[evaluation_start_index:], dtype=float
    )
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

    for index in range(
        start_index,
        len(errors) - sustain_frames + 1,
    ):
        if bool(np.all(above[index : index + sustain_frames])):
            return {
                "threshold_m": float(threshold_m),
                "crossed": True,
                "frame_index": int(index),
                "frames_after_alignment_prefix": int(index - start_index),
                "distance_after_alignment_prefix_m": float(
                    reference_distance[index] - start_distance
                ),
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


def evaluate_trajectory(
    trajectory: pd.DataFrame,
    sequence: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any], list[dict[str, Any]]]:
    merged = sequence[
        ["sequence_frame_id", "token0_id", "x_enu_m", "y_enu_m"]
    ].merge(
        trajectory,
        on="sequence_frame_id",
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != EXPECTED_FRAMES:
        raise RuntimeError("Trajectory did not merge to all 1,034 frames.")

    reference = merged[["x_enu_m", "y_enu_m"]].to_numpy(dtype=float)
    reference = reference - reference[0]
    visual = merged[["visual_x_px", "visual_y_px"]].to_numpy(dtype=float)

    reference_steps = np.linalg.norm(np.diff(reference, axis=0), axis=1)
    cumulative_distance = np.concatenate([[0.0], np.cumsum(reference_steps)])

    global_transform = fit_similarity(visual, reference)
    global_aligned = apply_similarity(visual, global_transform)
    global_error = np.linalg.norm(global_aligned - reference, axis=1)

    prefix_count = ALIGNMENT_PREFIX_FRAMES
    prefix_transform = fit_similarity(
        visual[:prefix_count],
        reference[:prefix_count],
    )
    prefix_aligned = apply_similarity(visual, prefix_transform)
    prefix_error = np.linalg.norm(prefix_aligned - reference, axis=1)

    output = merged.copy()
    output["variant"] = "se2_scale_normalized"
    output["reference_x_m"] = reference[:, 0]
    output["reference_y_m"] = reference[:, 1]
    output["reference_cumulative_distance_m"] = cumulative_distance
    output["global_aligned_x_m"] = global_aligned[:, 0]
    output["global_aligned_y_m"] = global_aligned[:, 1]
    output["global_alignment_error_m"] = global_error
    output["prefix_aligned_x_m"] = prefix_aligned[:, 0]
    output["prefix_aligned_y_m"] = prefix_aligned[:, 1]
    output["prefix_locked_error_m"] = prefix_error

    distance_after_prefix = (
        cumulative_distance - cumulative_distance[prefix_count - 1]
    )
    output["distance_after_alignment_prefix_m"] = distance_after_prefix
    output["prefix_locked_drift_per_100m"] = np.where(
        distance_after_prefix > 1e-9,
        100.0 * prefix_error / distance_after_prefix,
        np.nan,
    )

    global_summary = error_summary(
        global_error, cumulative_distance, evaluation_start_index=0
    )
    prefix_summary = error_summary(
        prefix_error,
        cumulative_distance,
        evaluation_start_index=prefix_count - 1,
    )

    crossings = [
        first_sustained_crossing(
            prefix_error,
            cumulative_distance,
            threshold,
            start_index=prefix_count - 1,
            sustain_frames=SUSTAIN_FRAMES,
        )
        for threshold in ERROR_THRESHOLDS_M
    ]

    summary = {
        "variant": "se2_scale_normalized",
        "frames": int(len(output)),
        "reference_path_m": trajectory_path_length(reference),
        "visual_path_px": trajectory_path_length(visual),
        "global_alignment": {
            "scale_m_per_px": float(global_transform["scale"]),
            "rotation_deg": float(global_transform["rotation_deg"]),
            **global_summary,
        },
        "prefix_locked_alignment": {
            "prefix_frames": prefix_count,
            "prefix_last_frame_index": prefix_count - 1,
            "prefix_reference_distance_m": float(
                cumulative_distance[prefix_count - 1]
            ),
            "scale_m_per_px": float(prefix_transform["scale"]),
            "rotation_deg": float(prefix_transform["rotation_deg"]),
            **prefix_summary,
        },
    }
    return output, summary, crossings


def load_orb_baseline(
    summary_path: Path,
    aligned_path: Path,
    runtime_report_path: Path,
    orb_pairs_path: Path,
) -> dict[str, Any]:
    summary_df = pd.read_csv(summary_path)
    selected = summary_df[
        summary_df["variant"].astype(str) == "se2_scale_normalized"
    ]
    if selected.empty:
        raise RuntimeError("ORB summary lacks se2_scale_normalized.")
    row = selected.iloc[0]

    aligned = pd.read_csv(aligned_path)
    aligned = aligned[
        aligned["variant"].astype(str) == "se2_scale_normalized"
    ].sort_values("sequence_frame_id")
    prefix_tail = aligned.iloc[ALIGNMENT_PREFIX_FRAMES - 1 :]
    failure_rate = float(
        (
            pd.to_numeric(
                prefix_tail["prefix_locked_error_m"],
                errors="coerce",
            )
            > 40.0
        ).mean()
    )

    runtime_report = load_json(runtime_report_path)
    orb_feature_seconds = float(runtime_report["feature_cache_seconds"])

    orb_pairs = pd.read_csv(orb_pairs_path)
    orb_pairs["stride"] = pd.to_numeric(
        orb_pairs["stride"], errors="raise"
    ).astype(int)
    stride1 = orb_pairs[orb_pairs["stride"] == 1].copy()
    orb_matching_seconds = float(
        pd.to_numeric(stride1["elapsed_ms"], errors="coerce").sum()
        / 1000.0
    )
    orb_total_seconds = orb_feature_seconds + orb_matching_seconds

    return {
        "method": "orb",
        "frames": EXPECTED_FRAMES,
        "pairs": EXPECTED_PAIRS,
        "affine_success_rate": float(
            bool_series(stride1["affine_ok"]).mean()
        ),
        "good_quality_rate": float(
            (stride1["status"].astype(str) == "good").mean()
        ),
        "inlier_ratio_median": finite_stat(
            pd.to_numeric(stride1["inlier_ratio"], errors="coerce"),
            np.median,
        ),
        "inlier_ratio_p05": finite_stat(
            pd.to_numeric(stride1["inlier_ratio"], errors="coerce"),
            lambda values: np.percentile(values, 5),
        ),
        "feature_seconds": orb_feature_seconds,
        "matching_ransac_seconds": orb_matching_seconds,
        "total_frontend_seconds": orb_total_seconds,
        "total_frontend_ms_per_pair": (
            1000.0 * orb_total_seconds / EXPECTED_PAIRS
        ),
        "global_shape_rmse_m": float(row["global_alignment.rmse_m"]),
        "prefix_locked_rmse_m": float(
            row["prefix_locked_alignment.rmse_m"]
        ),
        "prefix_locked_p95_m": float(
            row["prefix_locked_alignment.p95_error_m"]
        ),
        "prefix_locked_max_m": float(
            row["prefix_locked_alignment.max_error_m"]
        ),
        "final_error_m": float(
            row["prefix_locked_alignment.final_error_m"]
        ),
        "final_drift_per_100m": float(
            row["prefix_locked_alignment.final_drift_per_100m"]
        ),
        "failure_rate_gt40m": failure_rate,
    }


def build_xfeat_method_summary(
    feature_df: pd.DataFrame,
    pair_df: pd.DataFrame,
    trajectory_summary: dict[str, Any],
    feature_seconds: float,
    wall_seconds: float,
    peak_memory_mb: float,
) -> dict[str, Any]:
    matching_seconds = float(
        pair_df["matching_ransac_time_ms"].sum() / 1000.0
    )
    # `wall_seconds` covers image read, resize, tensor preparation, feature
    # inference, descriptor matching and RANSAC, matching the scope of the
    # frozen ORB feature-cache plus pair-matching timing.
    total_seconds = float(wall_seconds)
    prefix = trajectory_summary["prefix_locked_alignment"]
    global_summary = trajectory_summary["global_alignment"]

    return {
        "method": "xfeat",
        "frames": int(len(feature_df)),
        "pairs": int(len(pair_df)),
        "affine_success_rate": float(pair_df["affine_ok"].mean()),
        "good_quality_rate": float(
            (pair_df["status"].astype(str) == "good").mean()
        ),
        "inlier_ratio_median": finite_stat(
            pair_df["inlier_ratio"], np.median
        ),
        "inlier_ratio_p05": finite_stat(
            pair_df["inlier_ratio"],
            lambda values: np.percentile(values, 5),
        ),
        "feature_seconds": float(feature_seconds),
        "matching_ransac_seconds": matching_seconds,
        "total_frontend_seconds": total_seconds,
        "full_run_wall_clock_seconds": float(wall_seconds),
        "total_frontend_ms_per_pair": (
            1000.0 * total_seconds / EXPECTED_PAIRS
        ),
        "peak_process_memory_mb": float(peak_memory_mb),
        "global_shape_rmse_m": float(global_summary["rmse_m"]),
        "prefix_locked_rmse_m": float(prefix["rmse_m"]),
        "prefix_locked_p95_m": float(prefix["p95_error_m"]),
        "prefix_locked_max_m": float(prefix["max_error_m"]),
        "final_error_m": float(prefix["final_error_m"]),
        "final_drift_per_100m": float(prefix["final_drift_per_100m"]),
        "failure_rate_gt40m": float(prefix["failure_rate_gt40m"]),
    }


def summarize_scene_method(
    method: str,
    frame: pd.DataFrame,
    match_column: str,
    time_column: str,
    amortized_feature_ms_per_pair: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for scene, group in frame.groupby("primary_scene", sort=True):
        affine_ok = bool_series(group["affine_ok"])
        successful = group[affine_ok]
        rows.append(
            {
                "primary_scene": scene,
                "method": method,
                "pairs": int(len(group)),
                "affine_success_rate": float(affine_ok.mean()),
                "good_quality_rate": float(
                    (group["status"].astype(str) == "good").mean()
                ),
                "matches_median": finite_stat(
                    successful[match_column], np.median
                ),
                "inliers_median": finite_stat(
                    successful["inliers"], np.median
                ),
                "inlier_ratio_median": finite_stat(
                    successful["inlier_ratio"], np.median
                ),
                "inlier_ratio_p05": finite_stat(
                    successful["inlier_ratio"],
                    lambda values: np.percentile(values, 5),
                ),
                "matching_ransac_time_ms_mean": finite_stat(
                    successful[time_column], np.mean
                ),
                "amortized_total_frontend_ms_mean": float(
                    finite_stat(successful[time_column], np.mean)
                    + amortized_feature_ms_per_pair
                ),
            }
        )
    return pd.DataFrame(rows)


def build_scene_comparison(
    full_manifest: pd.DataFrame,
    xfeat_pairs: pd.DataFrame,
    orb_pairs_path: Path,
    xfeat_feature_seconds: float,
    orb_feature_seconds: float,
) -> pd.DataFrame:
    labels = full_manifest[
        full_manifest["in_stratified_diagnostic_subset"]
    ][
        [
            "pair_number",
            "primary_scene",
            "secondary_scene",
            "selection_role",
            "range_id",
        ]
    ].copy()

    xfeat_metric_columns = [
        column
        for column in xfeat_pairs.columns
        if column
        not in {
            "primary_scene",
            "secondary_scene",
            "selection_role",
            "range_id",
        }
    ]
    xfeat_subset = labels.merge(
        xfeat_pairs[xfeat_metric_columns],
        on="pair_number",
        how="left",
        validate="one_to_one",
    )

    orb = pd.read_csv(orb_pairs_path)
    orb["stride"] = pd.to_numeric(orb["stride"], errors="raise").astype(int)
    orb["pair_number"] = pd.to_numeric(
        orb["pair_number"], errors="raise"
    ).astype(int)
    orb = orb[orb["stride"] == 1].copy()
    orb_subset = labels.merge(
        orb,
        on="pair_number",
        how="left",
        validate="one_to_one",
    )

    xfeat_scene = summarize_scene_method(
        "xfeat",
        xfeat_subset,
        match_column="matches",
        time_column="matching_ransac_time_ms",
        amortized_feature_ms_per_pair=(
            1000.0 * xfeat_feature_seconds / EXPECTED_PAIRS
        ),
    )
    orb_scene = summarize_scene_method(
        "orb",
        orb_subset,
        match_column="good_matches",
        time_column="elapsed_ms",
        amortized_feature_ms_per_pair=(
            1000.0 * orb_feature_seconds / EXPECTED_PAIRS
        ),
    )
    return pd.concat(
        [orb_scene, xfeat_scene],
        ignore_index=True,
    ).sort_values(["primary_scene", "method"])


def crossing_distance(
    crossings: list[dict[str, Any]],
    threshold: float,
) -> float | None:
    for row in crossings:
        if float(row["threshold_m"]) == float(threshold):
            value = row["distance_after_alignment_prefix_m"]
            return None if value is None else float(value)
    return None


def load_orb_crossings(
    aligned_path: Path,
) -> list[dict[str, Any]]:
    aligned = pd.read_csv(aligned_path)
    aligned = aligned[
        aligned["variant"].astype(str) == "se2_scale_normalized"
    ].sort_values("sequence_frame_id")
    errors = pd.to_numeric(
        aligned["prefix_locked_error_m"], errors="coerce"
    ).to_numpy(dtype=float)
    distance = pd.to_numeric(
        aligned["reference_cumulative_distance_m"], errors="coerce"
    ).to_numpy(dtype=float)
    return [
        first_sustained_crossing(
            errors,
            distance,
            threshold,
            start_index=ALIGNMENT_PREFIX_FRAMES - 1,
            sustain_frames=SUSTAIN_FRAMES,
        )
        for threshold in ERROR_THRESHOLDS_M
    ]


def decide_promotion(
    orb: dict[str, Any],
    xfeat: dict[str, Any],
    scene_summary: pd.DataFrame,
    orb_crossings: list[dict[str, Any]],
    xfeat_crossings: list[dict[str, Any]],
) -> dict[str, Any]:
    full_affine_guard = (
        xfeat["affine_success_rate"]
        >= orb["affine_success_rate"] - 0.005
    )
    complete_chain_guard = xfeat["affine_success_rate"] == 1.0

    scene_guards: list[dict[str, Any]] = []
    for scene in sorted(scene_summary["primary_scene"].unique()):
        group = scene_summary[
            scene_summary["primary_scene"] == scene
        ]
        if len(group) < 2:
            continue
        orb_row = group[group["method"] == "orb"]
        xfeat_row = group[group["method"] == "xfeat"]
        if orb_row.empty or xfeat_row.empty:
            continue
        pair_count = int(xfeat_row.iloc[0]["pairs"])
        if pair_count < 10:
            continue
        orb_rate = float(orb_row.iloc[0]["affine_success_rate"])
        xfeat_rate = float(xfeat_row.iloc[0]["affine_success_rate"])
        scene_guards.append(
            {
                "primary_scene": scene,
                "pairs": pair_count,
                "orb_affine_success_rate": orb_rate,
                "xfeat_affine_success_rate": xfeat_rate,
                "passed": bool(xfeat_rate >= orb_rate - 0.05),
            }
        )

    scene_guard_pass = all(row["passed"] for row in scene_guards)

    rmse_improvement_fraction = (
        (orb["prefix_locked_rmse_m"] - xfeat["prefix_locked_rmse_m"])
        / orb["prefix_locked_rmse_m"]
    )
    p95_ratio = xfeat["prefix_locked_p95_m"] / orb["prefix_locked_p95_m"]
    max_ratio = xfeat["prefix_locked_max_m"] / orb["prefix_locked_max_m"]

    runtime_improvement_fraction = (
        (orb["total_frontend_ms_per_pair"]
         - xfeat["total_frontend_ms_per_pair"])
        / orb["total_frontend_ms_per_pair"]
    )

    accuracy_benefit = bool(
        rmse_improvement_fraction >= 0.05
        and p95_ratio <= 1.02
        and max_ratio <= 1.02
    )
    runtime_benefit = bool(
        runtime_improvement_fraction >= 0.25
        and xfeat["prefix_locked_rmse_m"]
        <= orb["prefix_locked_rmse_m"] * 1.02
        and p95_ratio <= 1.02
        and max_ratio <= 1.02
    )

    failure_reduction_fraction = (
        (orb["failure_rate_gt40m"] - xfeat["failure_rate_gt40m"])
        / max(orb["failure_rate_gt40m"], 1e-12)
    )
    orb_40 = crossing_distance(orb_crossings, 40.0)
    xfeat_40 = crossing_distance(xfeat_crossings, 40.0)

    if orb_40 is None:
        safe_horizon_improvement_fraction = 0.0
    elif xfeat_40 is None:
        safe_horizon_improvement_fraction = float("inf")
    else:
        safe_horizon_improvement_fraction = (
            xfeat_40 - orb_40
        ) / max(orb_40, 1e-12)

    robustness_benefit = bool(
        (
            failure_reduction_fraction >= 0.20
            or safe_horizon_improvement_fraction >= 0.20
        )
        and xfeat["total_frontend_ms_per_pair"]
        <= orb["total_frontend_ms_per_pair"] * 2.0
        and p95_ratio <= 1.05
        and max_ratio <= 1.05
    )

    guards_pass = bool(
        full_affine_guard and complete_chain_guard and scene_guard_pass
    )
    material_benefit = bool(
        accuracy_benefit or runtime_benefit or robustness_benefit
    )
    promote = bool(guards_pass and material_benefit)

    return {
        "decision": "PROMOTE_XFEAT" if promote else "KEEP_ORB",
        "promote_xfeat": promote,
        "robustness_guards": {
            "full_affine_guard_pass": bool(full_affine_guard),
            "complete_chain_guard_pass": bool(complete_chain_guard),
            "scene_guard_pass": bool(scene_guard_pass),
            "scene_guards": scene_guards,
            "all_guards_pass": guards_pass,
        },
        "material_benefits": {
            "accuracy_benefit_pass": accuracy_benefit,
            "runtime_benefit_pass": runtime_benefit,
            "robustness_benefit_pass": robustness_benefit,
            "any_material_benefit": material_benefit,
        },
        "comparison_values": {
            "prefix_rmse_improvement_fraction": float(
                rmse_improvement_fraction
            ),
            "runtime_improvement_fraction": float(
                runtime_improvement_fraction
            ),
            "p95_ratio_xfeat_over_orb": float(p95_ratio),
            "max_ratio_xfeat_over_orb": float(max_ratio),
            "failure_rate_reduction_fraction": float(
                failure_reduction_fraction
            ),
            "safe_horizon_40m_improvement_fraction": (
                None
                if not np.isfinite(safe_horizon_improvement_fraction)
                else float(safe_horizon_improvement_fraction)
            ),
            "orb_40m_crossing_distance_m": orb_40,
            "xfeat_40m_crossing_distance_m": xfeat_40,
        },
        "closeout_rule": (
            "S7A closes after this single full comparison. No XFeat "
            "threshold or model sweep follows."
        ),
    }


def save_plots(
    xfeat_aligned: pd.DataFrame,
    xfeat_pairs: pd.DataFrame,
    orb_aligned_path: Path,
    orb_pairs_path: Path,
    figures_dir: Path,
) -> None:
    ensure_dir(figures_dir)

    orb_aligned = pd.read_csv(orb_aligned_path)
    orb_aligned = orb_aligned[
        orb_aligned["variant"].astype(str) == "se2_scale_normalized"
    ].sort_values("sequence_frame_id")

    plt.figure(figsize=(9, 8))
    plt.plot(
        xfeat_aligned["reference_x_m"],
        xfeat_aligned["reference_y_m"],
        label="Reference — evaluation only",
    )
    plt.plot(
        orb_aligned["prefix_aligned_x_m"],
        orb_aligned["prefix_aligned_y_m"],
        label="ORB prefix-locked",
    )
    plt.plot(
        xfeat_aligned["prefix_aligned_x_m"],
        xfeat_aligned["prefix_aligned_y_m"],
        label="XFeat prefix-locked",
    )
    plt.axis("equal")
    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")
    plt.title("S7A.2 ORB versus XFeat prefix-locked trajectory")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        figures_dir / "s7a2_orb_xfeat_prefix_locked_xy.png",
        dpi=180,
    )
    plt.close()

    plt.figure(figsize=(11, 6))
    plt.plot(
        orb_aligned["reference_cumulative_distance_m"],
        orb_aligned["prefix_locked_error_m"],
        label="ORB",
    )
    plt.plot(
        xfeat_aligned["reference_cumulative_distance_m"],
        xfeat_aligned["prefix_locked_error_m"],
        label="XFeat",
    )
    for threshold in ERROR_THRESHOLDS_M:
        plt.axhline(
            threshold,
            linestyle="--",
            linewidth=1.0,
            label=f"{threshold:g} m",
        )
    plt.xlabel("Reference cumulative distance [m] — evaluation only")
    plt.ylabel("Prefix-locked position error [m]")
    plt.title("S7A.2 relative-frontend error growth")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        figures_dir / "s7a2_orb_xfeat_error_vs_distance.png",
        dpi=180,
    )
    plt.close()

    orb_pairs = pd.read_csv(orb_pairs_path)
    orb_pairs["stride"] = pd.to_numeric(
        orb_pairs["stride"], errors="raise"
    ).astype(int)
    orb_stride1 = orb_pairs[
        orb_pairs["stride"] == 1
    ].sort_values("frame_index_a")

    plt.figure(figsize=(12, 6))
    plt.plot(
        orb_stride1["frame_index_a"],
        orb_stride1["inlier_ratio"],
        label="ORB",
    )
    plt.plot(
        xfeat_pairs["frame_index_a"],
        xfeat_pairs["inlier_ratio"],
        label="XFeat",
    )
    plt.xlabel("Stride-1 pair start frame")
    plt.ylabel("RANSAC inlier ratio")
    plt.title("S7A.2 ORB versus XFeat pairwise inlier ratio")
    plt.ylim(0.0, 1.02)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        figures_dir / "s7a2_orb_xfeat_inlier_ratio.png",
        dpi=180,
    )
    plt.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--xfeat-repo", type=Path, default=DEFAULT_XFEAT_REPO)
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )
    parser.add_argument("--resize-long", type=int, default=960)
    parser.add_argument("--top-k", type=int, default=1200)
    parser.add_argument("--detection-threshold", type=float, default=0.05)
    parser.add_argument("--min-cossim", type=float, default=0.82)
    parser.add_argument("--ransac-threshold", type=float, default=3.0)
    parser.add_argument(
        "--allow-unpinned-xfeat",
        action="store_true",
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    output_root = resolve_from_repo(repo_root, args.output_root)
    xfeat_repo = resolve_from_repo(repo_root, args.xfeat_repo)

    if args.resize_long != 960:
        raise RuntimeError("S7A.0 froze resize-long at 960.")
    if args.top_k != 1200:
        raise RuntimeError("S7A.0 froze top-k at 1200.")
    if abs(args.detection_threshold - 0.05) > 1e-12:
        raise RuntimeError("S7A.1 froze detection threshold at 0.05.")
    if abs(args.min_cossim - 0.82) > 1e-12:
        raise RuntimeError("S7A.1 froze minimum cosine at 0.82.")
    if abs(args.ransac_threshold - 3.0) > 1e-12:
        raise RuntimeError("S7A.0 froze RANSAC threshold at 3.0 px.")

    paths = {
        "sequence": resolve_from_repo(repo_root, DEFAULT_SEQUENCE_MANIFEST),
        "orb_pairs": resolve_from_repo(repo_root, DEFAULT_ORB_PAIRS),
        "orb_summary": resolve_from_repo(
            repo_root, DEFAULT_ORB_TRAJECTORY_SUMMARY
        ),
        "orb_aligned": resolve_from_repo(
            repo_root, DEFAULT_ORB_ALIGNED_TRAJECTORY
        ),
        "orb_runtime": resolve_from_repo(
            repo_root, DEFAULT_ORB_RUNTIME_REPORT
        ),
        "full_pairs": resolve_from_repo(
            repo_root, DEFAULT_FULL_PAIR_MANIFEST
        ),
        "protocol": resolve_from_repo(repo_root, DEFAULT_PROTOCOL),
        "smoke": resolve_from_repo(repo_root, DEFAULT_SMOKE_SUMMARY),
    }
    for label, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing {label}: {path}")

    protocol = load_json(paths["protocol"])
    smoke = load_json(paths["smoke"])
    if protocol.get("status") != "PASS_FROZEN":
        raise RuntimeError("S7A.0 protocol is not PASS_FROZEN.")
    if not bool(smoke.get("comparison", {}).get("smoke_gate_pass", False)):
        raise RuntimeError("S7A.1 smoke gate did not pass.")

    sequence = load_sequence_manifest(repo_root, paths["sequence"])
    full_pair_manifest = load_full_pair_manifest(paths["full_pairs"])

    torch, model, device, xfeat_commit = load_official_xfeat(
        xfeat_repo,
        args.device,
        args.top_k,
        args.detection_threshold,
    )
    if (
        xfeat_commit
        and xfeat_commit != PINNED_XFEAT_COMMIT
        and not args.allow_unpinned_xfeat
    ):
        raise RuntimeError(
            "XFeat checkout is not at the frozen commit.\n"
            f"Expected: {PINNED_XFEAT_COMMIT}\n"
            f"Found:    {xfeat_commit}"
        )

    metadata_dir = ensure_dir(
        output_root / "metadata" / "s7_relative_frontend"
    )
    reports_dir = ensure_dir(
        output_root / "reports" / "s7_relative_frontend"
    )
    figures_dir = ensure_dir(
        output_root / "figures" / "s7_relative_frontend" / "s7a2_full"
    )

    feature_df, pair_df, feature_seconds, wall_seconds = (
        run_streaming_frontend(
            torch,
            model,
            device,
            sequence,
            full_pair_manifest,
            args.resize_long,
            args.top_k,
            args.detection_threshold,
            args.min_cossim,
            args.ransac_threshold,
        )
    )
    peak_memory_mb = peak_process_memory_mb()

    feature_path = metadata_dir / "s7a_2_xfeat_frame_features.csv"
    pair_path = metadata_dir / "s7a_2_xfeat_full_pair_diagnostics.csv"
    feature_df.to_csv(feature_path, index=False)
    pair_df.to_csv(pair_path, index=False)

    affine_success_rate = float(pair_df["affine_ok"].mean())
    trajectory_available = bool(pair_df["affine_ok"].all())

    raw_trajectory_path = (
        metadata_dir / "s7a_2_xfeat_relative_trajectory_pixels.csv"
    )
    aligned_trajectory_path = (
        metadata_dir
        / "s7a_2_xfeat_relative_trajectory_aligned_eval_only.csv"
    )
    crossing_path = metadata_dir / "s7a_2_drift_threshold_crossings.csv"
    method_comparison_path = (
        metadata_dir / "s7a_2_full_method_comparison.csv"
    )
    scene_comparison_path = (
        metadata_dir / "s7a_2_scene_comparison.csv"
    )
    decision_path = metadata_dir / "s7a_2_promotion_decision.json"
    summary_json_path = (
        reports_dir / "s7a_2_xfeat_full_comparison.json"
    )
    report_path = (
        reports_dir / "s7a_2_xfeat_full_comparison_report.md"
    )

    orb_summary = load_orb_baseline(
        paths["orb_summary"],
        paths["orb_aligned"],
        paths["orb_runtime"],
        paths["orb_pairs"],
    )

    if not trajectory_available:
        decision = {
            "decision": "KEEP_ORB",
            "promote_xfeat": False,
            "reason": (
                "At least one XFeat pair failed affine estimation; the "
                "frozen complete-chain guard failed."
            ),
            "xfeat_affine_success_rate": affine_success_rate,
            "closeout_rule": (
                "S7A closes. No XFeat threshold sweep follows."
            ),
        }
        decision_path.write_text(
            json.dumps(decision, indent=2), encoding="utf-8"
        )
        payload = {
            "generated_utc": utc_now(),
            "stage": "S7A.2",
            "status": "COMPLETE_KEEP_ORB",
            "trajectory_available": False,
            "decision": decision,
            "configuration": vars(args),
            "xfeat_commit": xfeat_commit,
            "device": str(device),
            "feature_seconds": feature_seconds,
            "wall_seconds": wall_seconds,
            "peak_memory_mb": peak_memory_mb,
        }
        # Convert Paths in argparse payload.
        payload["configuration"] = {
            key: str(value) if isinstance(value, Path) else value
            for key, value in payload["configuration"].items()
        }
        summary_json_path.write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
        print("S7A.2 XFeat Full Trajectory Comparison")
        print("-------------------------------------")
        print("Status:                   COMPLETE_KEEP_ORB")
        print(f"Affine success rate:      {affine_success_rate:.4f}")
        print("Trajectory available:     False")
        print("Decision:                 KEEP_ORB")
        print(f"Pair diagnostics:         {pair_path.relative_to(repo_root)}")
        print(f"Decision JSON:            {decision_path.relative_to(repo_root)}")
        return 0

    if feature_df["width"].nunique() != 1 or feature_df["height"].nunique() != 1:
        raise RuntimeError(
            "Resized frame dimensions are not constant; cannot reuse "
            "the frozen S6A trajectory center convention."
        )

    trajectory = integrate_se2_scale_normalized(
        pair_df,
        width=int(feature_df.iloc[0]["width"]),
        height=int(feature_df.iloc[0]["height"]),
    )
    aligned, trajectory_summary, xfeat_crossings = evaluate_trajectory(
        trajectory, sequence
    )
    trajectory.to_csv(raw_trajectory_path, index=False)
    aligned.to_csv(aligned_trajectory_path, index=False)

    xfeat_summary = build_xfeat_method_summary(
        feature_df,
        pair_df,
        trajectory_summary,
        feature_seconds,
        wall_seconds,
        peak_memory_mb,
    )

    scene_summary = build_scene_comparison(
        full_pair_manifest,
        pair_df,
        paths["orb_pairs"],
        xfeat_feature_seconds=max(
            wall_seconds
            - float(pair_df["matching_ransac_time_ms"].sum() / 1000.0),
            0.0,
        ),
        orb_feature_seconds=orb_summary["feature_seconds"],
    )
    scene_summary.to_csv(scene_comparison_path, index=False)

    orb_crossings = load_orb_crossings(paths["orb_aligned"])
    crossing_rows = [
        {"method": "orb", **row} for row in orb_crossings
    ] + [
        {"method": "xfeat", **row} for row in xfeat_crossings
    ]
    pd.DataFrame(crossing_rows).to_csv(crossing_path, index=False)

    comparison_df = pd.DataFrame([orb_summary, xfeat_summary])
    comparison_df.to_csv(method_comparison_path, index=False)

    decision = decide_promotion(
        orb_summary,
        xfeat_summary,
        scene_summary,
        orb_crossings,
        xfeat_crossings,
    )
    decision_path.write_text(
        json.dumps(decision, indent=2), encoding="utf-8"
    )

    save_plots(
        aligned,
        pair_df,
        paths["orb_aligned"],
        paths["orb_pairs"],
        figures_dir,
    )

    status = (
        "COMPLETE_PROMOTE_XFEAT"
        if decision["promote_xfeat"]
        else "COMPLETE_KEEP_ORB"
    )
    payload = {
        "generated_utc": utc_now(),
        "stage": "S7A.2",
        "status": status,
        "scientific_scope": (
            "Single frozen full-sequence ORB-versus-XFeat relative "
            "frontend comparison."
        ),
        "official_xfeat": {
            "repository": OFFICIAL_XFEAT_REPOSITORY,
            "expected_commit": PINNED_XFEAT_COMMIT,
            "actual_commit": xfeat_commit,
            "checkout_path": str(xfeat_repo.relative_to(repo_root)),
        },
        "environment": {
            "python": sys.version,
            "torch": str(torch.__version__),
            "device": str(device),
            "platform": platform.platform(),
        },
        "configuration": {
            "resize_long": args.resize_long,
            "top_k": args.top_k,
            "detection_threshold": args.detection_threshold,
            "min_cossim": args.min_cossim,
            "ransac_threshold": args.ransac_threshold,
            "frames": EXPECTED_FRAMES,
            "pairs": EXPECTED_PAIRS,
            "alignment_prefix_frames": ALIGNMENT_PREFIX_FRAMES,
            "error_thresholds_m": ERROR_THRESHOLDS_M,
            "sustain_frames": SUSTAIN_FRAMES,
        },
        "orb_summary": orb_summary,
        "xfeat_summary": xfeat_summary,
        "orb_crossings": orb_crossings,
        "xfeat_crossings": xfeat_crossings,
        "decision": decision,
        "ground_truth_rule": (
            "Reference ENU was used only after the complete XFeat "
            "trajectory was constructed."
        ),
        "outputs": {
            "frame_features": str(feature_path.relative_to(repo_root)),
            "pair_diagnostics": str(pair_path.relative_to(repo_root)),
            "raw_trajectory": str(raw_trajectory_path.relative_to(repo_root)),
            "aligned_trajectory": str(
                aligned_trajectory_path.relative_to(repo_root)
            ),
            "method_comparison": str(
                method_comparison_path.relative_to(repo_root)
            ),
            "scene_comparison": str(
                scene_comparison_path.relative_to(repo_root)
            ),
            "threshold_crossings": str(crossing_path.relative_to(repo_root)),
            "promotion_decision": str(decision_path.relative_to(repo_root)),
            "figures": str(figures_dir.relative_to(repo_root)),
            "report": str(report_path.relative_to(repo_root)),
        },
    }
    summary_json_path.write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    report = f"""# S7A.2 — Full XFeat Relative-Frontend Comparison

Generated: `{payload["generated_utc"]}`

## Status

```text
{status}
```

## Full-sequence result

| Metric | ORB | XFeat |
|---|---:|---:|
| Affine success | {orb_summary["affine_success_rate"]:.4f} | {xfeat_summary["affine_success_rate"]:.4f} |
| Good-quality rate | {orb_summary["good_quality_rate"]:.4f} | {xfeat_summary["good_quality_rate"]:.4f} |
| Median inlier ratio | {orb_summary["inlier_ratio_median"]:.4f} | {xfeat_summary["inlier_ratio_median"]:.4f} |
| Inlier-ratio p05 | {orb_summary["inlier_ratio_p05"]:.4f} | {xfeat_summary["inlier_ratio_p05"]:.4f} |
| Prefix RMSE [m] | {orb_summary["prefix_locked_rmse_m"]:.3f} | {xfeat_summary["prefix_locked_rmse_m"]:.3f} |
| Prefix p95 [m] | {orb_summary["prefix_locked_p95_m"]:.3f} | {xfeat_summary["prefix_locked_p95_m"]:.3f} |
| Prefix max [m] | {orb_summary["prefix_locked_max_m"]:.3f} | {xfeat_summary["prefix_locked_max_m"]:.3f} |
| Final error [m] | {orb_summary["final_error_m"]:.3f} | {xfeat_summary["final_error_m"]:.3f} |
| Final drift [m/100m] | {orb_summary["final_drift_per_100m"]:.3f} | {xfeat_summary["final_drift_per_100m"]:.3f} |
| Failure rate >40 m | {orb_summary["failure_rate_gt40m"]:.4f} | {xfeat_summary["failure_rate_gt40m"]:.4f} |
| Frontend ms/pair | {orb_summary["total_frontend_ms_per_pair"]:.2f} | {xfeat_summary["total_frontend_ms_per_pair"]:.2f} |

## Decision

```text
{decision["decision"]}
```

S7A is closed after this run. No XFeat threshold, keypoint-budget, or model
sweep follows. The next stage proceeds with the selected relative frontend.
"""
    report_path.write_text(report, encoding="utf-8")

    print("S7A.2 XFeat Full Trajectory Comparison")
    print("-------------------------------------")
    print(f"Status:                    {status}")
    print(f"XFeat commit:              {xfeat_commit or 'unknown'}")
    print(f"Device:                    {device}")
    print(f"Frames:                    {len(feature_df)}")
    print(f"Pairs:                     {len(pair_df)}")
    print(
        "ORB affine success:        "
        f"{orb_summary['affine_success_rate']:.4f}"
    )
    print(
        "XFeat affine success:      "
        f"{xfeat_summary['affine_success_rate']:.4f}"
    )
    print(
        "ORB prefix RMSE m:         "
        f"{orb_summary['prefix_locked_rmse_m']:.3f}"
    )
    print(
        "XFeat prefix RMSE m:       "
        f"{xfeat_summary['prefix_locked_rmse_m']:.3f}"
    )
    print(
        "ORB prefix p95 m:          "
        f"{orb_summary['prefix_locked_p95_m']:.3f}"
    )
    print(
        "XFeat prefix p95 m:        "
        f"{xfeat_summary['prefix_locked_p95_m']:.3f}"
    )
    print(
        "ORB final drift m/100m:    "
        f"{orb_summary['final_drift_per_100m']:.3f}"
    )
    print(
        "XFeat final drift m/100m:  "
        f"{xfeat_summary['final_drift_per_100m']:.3f}"
    )
    print(
        "ORB frontend ms/pair:      "
        f"{orb_summary['total_frontend_ms_per_pair']:.2f}"
    )
    print(
        "XFeat frontend ms/pair:    "
        f"{xfeat_summary['total_frontend_ms_per_pair']:.2f}"
    )
    print(f"Peak process memory MB:     {peak_memory_mb:.1f}")
    print(f"Decision:                   {decision['decision']}")
    print(f"Method comparison:          {method_comparison_path.relative_to(repo_root)}")
    print(f"Scene comparison:           {scene_comparison_path.relative_to(repo_root)}")
    print(f"Decision JSON:              {decision_path.relative_to(repo_root)}")
    print(f"Figures:                    {figures_dir.relative_to(repo_root)}")
    print(f"JSON summary:               {summary_json_path.relative_to(repo_root)}")
    print(f"Report:                     {report_path.relative_to(repo_root)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print("S7A.2 XFeat Full Trajectory Comparison", file=sys.stderr)
        print("-------------------------------------", file=sys.stderr)
        print("Status: BLOCKED", file=sys.stderr)
        print(f"Reason: {exc}", file=sys.stderr)
        raise
