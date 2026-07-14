'''
Command Used:

export PYTHONPATH=$PWD/src

python scripts/satloc/s6a/s6a_4a_klt_vs_orb_relative_comparison_v2.py \
  --sequence traj01 \
  --resize-long 960 \
  --max-corners 1200 \
  --quality-level 0.01 \
  --min-distance 7 \
  --win-size 31 \
  --max-level 4 \
  --fb-threshold-px 1.5 \
  --ransac-threshold-px 3.0 \
  --alignment-prefix-frames 50 \
  --thresholds-m 10,20,40,80 \
  --sustain-frames 5

'''

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_MANIFEST = Path(
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
DEFAULT_OUTPUT_ROOT = Path("outputs/satloc")


def parse_float_list(value: str) -> list[float]:
    values: list[float] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        number = float(item)
        if number <= 0:
            raise argparse.ArgumentTypeError(
                "Thresholds must be positive."
            )
        values.append(number)

    if not values:
        raise argparse.ArgumentTypeError(
            "At least one threshold is required."
        )

    return sorted(set(values))


def resolve_image_path(value: Any) -> Path:
    path = Path(str(value)).expanduser()
    candidates = [path, Path.cwd() / path]

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    return path


def resize_long_side(
    image: np.ndarray,
    target_long_side: int,
) -> np.ndarray:
    height, width = image.shape[:2]
    long_side = max(height, width)

    if long_side == target_long_side:
        return image

    scale = target_long_side / float(long_side)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))

    interpolation = (
        cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    )
    return cv2.resize(
        image,
        (new_width, new_height),
        interpolation=interpolation,
    )


def read_gray(path: Path, resize_long: int) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(
            f"Could not read image: {path}"
        )
    return resize_long_side(image, resize_long)


def affine_metrics(
    matrix_2x3: np.ndarray,
    image_width: int,
    image_height: int,
) -> dict[str, float]:
    matrix = np.eye(3, dtype=float)
    matrix[:2, :] = matrix_2x3

    linear = matrix[:2, :2]
    determinant = float(np.linalg.det(linear))
    scale = math.sqrt(abs(determinant))
    rotation_deg = math.degrees(
        math.atan2(linear[1, 0], linear[0, 0])
    )

    center = np.array(
        [image_width / 2.0, image_height / 2.0, 1.0],
        dtype=float,
    )
    mapped_center = matrix @ center
    center_delta = mapped_center[:2] - center[:2]

    return {
        "affine_a00": float(matrix[0, 0]),
        "affine_a01": float(matrix[0, 1]),
        "affine_a10": float(matrix[1, 0]),
        "affine_a11": float(matrix[1, 1]),
        "affine_tx_px": float(matrix[0, 2]),
        "affine_ty_px": float(matrix[1, 2]),
        "affine_rotation_deg": float(rotation_deg),
        "affine_scale": float(scale),
        "center_content_dx_px": float(center_delta[0]),
        "center_content_dy_px": float(center_delta[1]),
        "center_content_motion_px": float(
            np.linalg.norm(center_delta)
        ),
    }


def estimate_klt_pair(
    image_a: np.ndarray,
    image_b: np.ndarray,
    max_corners: int,
    quality_level: float,
    min_distance: float,
    block_size: int,
    win_size: int,
    max_level: int,
    fb_threshold_px: float,
    ransac_threshold_px: float,
) -> dict[str, Any]:
    start_time = time.perf_counter()

    points_a = cv2.goodFeaturesToTrack(
        image_a,
        maxCorners=max_corners,
        qualityLevel=quality_level,
        minDistance=min_distance,
        blockSize=block_size,
        useHarrisDetector=False,
    )

    if points_a is None or len(points_a) < 3:
        return {
            "status": "insufficient_corners",
            "affine_ok": False,
            "detected_corners": 0 if points_a is None else len(points_a),
            "tracked_forward": 0,
            "fb_consistent_tracks": 0,
            "inliers": 0,
            "inlier_ratio": 0.0,
            "fb_error_median_px": float("nan"),
            "runtime_ms": 1000.0 * (
                time.perf_counter() - start_time
            ),
        }

    lk_criteria = (
        cv2.TERM_CRITERIA_EPS
        | cv2.TERM_CRITERIA_COUNT,
        30,
        0.01,
    )
    lk_params = {
        "winSize": (win_size, win_size),
        "maxLevel": max_level,
        "criteria": lk_criteria,
        "flags": 0,
        "minEigThreshold": 1e-4,
    }

    points_b, status_ab, _ = cv2.calcOpticalFlowPyrLK(
        image_a,
        image_b,
        points_a,
        None,
        **lk_params,
    )

    if points_b is None or status_ab is None:
        return {
            "status": "forward_flow_failed",
            "affine_ok": False,
            "detected_corners": int(len(points_a)),
            "tracked_forward": 0,
            "fb_consistent_tracks": 0,
            "inliers": 0,
            "inlier_ratio": 0.0,
            "fb_error_median_px": float("nan"),
            "runtime_ms": 1000.0 * (
                time.perf_counter() - start_time
            ),
        }

    points_a_back, status_ba, _ = (
        cv2.calcOpticalFlowPyrLK(
            image_b,
            image_a,
            points_b,
            None,
            **lk_params,
        )
    )

    if points_a_back is None or status_ba is None:
        return {
            "status": "backward_flow_failed",
            "affine_ok": False,
            "detected_corners": int(len(points_a)),
            "tracked_forward": int(status_ab.sum()),
            "fb_consistent_tracks": 0,
            "inliers": 0,
            "inlier_ratio": 0.0,
            "fb_error_median_px": float("nan"),
            "runtime_ms": 1000.0 * (
                time.perf_counter() - start_time
            ),
        }

    points_a_flat = points_a.reshape(-1, 2)
    points_b_flat = points_b.reshape(-1, 2)
    points_a_back_flat = points_a_back.reshape(-1, 2)

    status_ab_flat = status_ab.reshape(-1).astype(bool)
    status_ba_flat = status_ba.reshape(-1).astype(bool)

    finite = (
        np.isfinite(points_a_flat).all(axis=1)
        & np.isfinite(points_b_flat).all(axis=1)
        & np.isfinite(points_a_back_flat).all(axis=1)
    )
    valid_flow = status_ab_flat & status_ba_flat & finite

    fb_error = np.linalg.norm(
        points_a_back_flat - points_a_flat,
        axis=1,
    )
    consistent = valid_flow & (
        fb_error <= fb_threshold_px
    )

    src = points_a_flat[consistent]
    dst = points_b_flat[consistent]

    if len(src) < 3:
        return {
            "status": "insufficient_fb_tracks",
            "affine_ok": False,
            "detected_corners": int(len(points_a)),
            "tracked_forward": int(status_ab_flat.sum()),
            "fb_consistent_tracks": int(len(src)),
            "inliers": 0,
            "inlier_ratio": 0.0,
            "fb_error_median_px": (
                float(np.median(fb_error[consistent]))
                if np.any(consistent)
                else float("nan")
            ),
            "runtime_ms": 1000.0 * (
                time.perf_counter() - start_time
            ),
        }

    affine, inlier_mask = cv2.estimateAffinePartial2D(
        src.astype(np.float32),
        dst.astype(np.float32),
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_threshold_px,
        maxIters=3000,
        confidence=0.995,
        refineIters=10,
    )

    if affine is None or inlier_mask is None:
        return {
            "status": "affine_failed",
            "affine_ok": False,
            "detected_corners": int(len(points_a)),
            "tracked_forward": int(status_ab_flat.sum()),
            "fb_consistent_tracks": int(len(src)),
            "inliers": 0,
            "inlier_ratio": 0.0,
            "fb_error_median_px": float(
                np.median(fb_error[consistent])
            ),
            "runtime_ms": 1000.0 * (
                time.perf_counter() - start_time
            ),
        }

    inlier_mask_flat = inlier_mask.reshape(-1).astype(bool)
    inliers = int(inlier_mask_flat.sum())
    inlier_ratio = inliers / max(len(src), 1)

    metrics = affine_metrics(
        affine,
        image_width=image_a.shape[1],
        image_height=image_a.shape[0],
    )

    result = {
        "status": "ok",
        "affine_ok": True,
        "detected_corners": int(len(points_a)),
        "tracked_forward": int(status_ab_flat.sum()),
        "fb_consistent_tracks": int(len(src)),
        "inliers": inliers,
        "inlier_ratio": float(inlier_ratio),
        "fb_error_median_px": float(
            np.median(fb_error[consistent])
        ),
        "fb_error_p95_px": float(
            np.percentile(fb_error[consistent], 95)
        ),
        "runtime_ms": 1000.0 * (
            time.perf_counter() - start_time
        ),
        **metrics,
    }

    return result


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
    translation = (
        mapped_center[:2] - rotation @ center[:2]
    )

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

    step_image = (
        camera_b_center_in_a[:2] - center[:2]
    )

    return np.array(
        [step_image[0], -step_image[1]],
        dtype=float,
    )


def rotation_matrix(angle_rad: float) -> np.ndarray:
    cosine = math.cos(angle_rad)
    sine = math.sin(angle_rad)
    return np.array(
        [[cosine, -sine], [sine, cosine]],
        dtype=float,
    )


def integrate_pair_chain(
    pair_df: pd.DataFrame,
    width: int,
    height: int,
) -> pd.DataFrame:
    center = np.array(
        [width / 2.0, height / 2.0, 1.0],
        dtype=float,
    )

    position = np.zeros(2, dtype=float)
    yaw_rad = 0.0

    rows: list[dict[str, Any]] = [
        {
            "sequence_frame_id": 0,
            "visual_x_px": 0.0,
            "visual_y_px": 0.0,
            "visual_yaw_rad": 0.0,
            "visual_yaw_deg_unwrapped": 0.0,
            "step_motion_px": 0.0,
            "pair_safe_image_only": True,
        }
    ]

    for _, pair in pair_df.iterrows():
        matrix = affine_from_row(pair)
        normalized = normalize_affine_scale_about_center(
            matrix,
            center,
        )

        local_step = local_camera_step_from_scene_affine(
            normalized,
            center,
        )
        global_step = rotation_matrix(yaw_rad) @ local_step
        position = position + global_step

        yaw_rad += math.radians(
            float(pair["affine_rotation_deg"])
        )

        rows.append(
            {
                "sequence_frame_id": int(
                    pair["frame_index_b"]
                ),
                "visual_x_px": float(position[0]),
                "visual_y_px": float(position[1]),
                "visual_yaw_rad": float(yaw_rad),
                "visual_yaw_deg_unwrapped": float(
                    math.degrees(yaw_rad)
                ),
                "step_motion_px": float(
                    np.linalg.norm(global_step)
                ),
                "pair_safe_image_only": bool(
                    pair["pair_safe_image_only"]
                ),
            }
        )

    return pd.DataFrame(rows)


def fit_similarity(
    source: np.ndarray,
    target: np.ndarray,
) -> dict[str, Any]:
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)

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

    denominator = float(
        np.sum(source_centered * source_centered)
    )
    if denominator <= 1e-12:
        raise RuntimeError(
            "Visual trajectory has insufficient spread."
        )

    scale = float(
        np.sum(singular_values) / denominator
    )
    translation = (
        target_mean - scale * (rotation @ source_mean)
    )

    return {
        "scale": scale,
        "rotation": rotation,
        "translation": translation,
        "rotation_deg": float(
            math.degrees(
                math.atan2(
                    rotation[1, 0],
                    rotation[0, 0],
                )
            )
        ),
    }


def apply_similarity(
    points: np.ndarray,
    transform: dict[str, Any],
) -> np.ndarray:
    return (
        transform["scale"]
        * (
            np.asarray(points, dtype=float)
            @ transform["rotation"].T
        )
        + transform["translation"]
    )


def error_metrics(
    aligned: np.ndarray,
    reference: np.ndarray,
    cumulative_distance: np.ndarray,
    evaluation_start_index: int,
) -> dict[str, float]:
    errors = np.linalg.norm(
        aligned - reference,
        axis=1,
    )
    tail = errors[evaluation_start_index:]

    distance = float(
        cumulative_distance[-1]
        - cumulative_distance[evaluation_start_index]
    )
    final_error = float(errors[-1])

    return {
        "rmse_m": float(
            math.sqrt(np.mean(tail * tail))
        ),
        "mean_error_m": float(np.mean(tail)),
        "median_error_m": float(np.median(tail)),
        "p95_error_m": float(
            np.percentile(tail, 95)
        ),
        "max_error_m": float(np.max(tail)),
        "final_error_m": final_error,
        "evaluation_distance_m": distance,
        "final_drift_per_100m": (
            100.0 * final_error / distance
            if distance > 1e-9
            else float("nan")
        ),
    }


def first_sustained_crossing(
    errors: np.ndarray,
    cumulative_distance: np.ndarray,
    threshold_m: float,
    start_index: int,
    sustain_frames: int,
) -> dict[str, Any]:
    above = (
        np.isfinite(errors)
        & (errors >= threshold_m)
    )

    for index in range(
        start_index,
        len(errors) - sustain_frames + 1,
    ):
        if bool(
            np.all(
                above[index : index + sustain_frames]
            )
        ):
            return {
                "threshold_m": float(threshold_m),
                "crossed": True,
                "frame_index": int(index),
                "frames_after_prefix": int(
                    index - start_index
                ),
                "distance_after_prefix_m": float(
                    cumulative_distance[index]
                    - cumulative_distance[start_index]
                ),
                "error_at_crossing_m": float(
                    errors[index]
                ),
            }

    return {
        "threshold_m": float(threshold_m),
        "crossed": False,
        "frame_index": None,
        "frames_after_prefix": None,
        "distance_after_prefix_m": None,
        "error_at_crossing_m": None,
    }


def finite_percentile(
    values: pd.Series | np.ndarray,
    percentile: float,
) -> float:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return float("nan")
    return float(np.percentile(array, percentile))


def load_orb_comparison(
    orb_pairs_path: Path,
    orb_aligned_path: Path,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    orb_pairs = None
    orb_aligned = None

    if orb_pairs_path.exists():
        orb_pairs = pd.read_csv(orb_pairs_path)
        orb_pairs = orb_pairs[
            pd.to_numeric(
                orb_pairs["stride"],
                errors="coerce",
            )
            == 1
        ].copy()

    if orb_aligned_path.exists():
        orb_aligned = pd.read_csv(orb_aligned_path)
        if "variant" in orb_aligned.columns:
            orb_aligned = orb_aligned[
                orb_aligned["variant"].astype(str)
                == "se2_scale_normalized"
            ].copy()

    return orb_pairs, orb_aligned


def save_plots(
    pair_df: pd.DataFrame,
    aligned_df: pd.DataFrame,
    orb_pairs: pd.DataFrame | None,
    orb_aligned: pd.DataFrame | None,
    figures_dir: Path,
    thresholds_m: list[float],
) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 6))
    plt.plot(
        pair_df["frame_index_a"],
        pair_df["inlier_ratio"],
        label="KLT",
    )
    if orb_pairs is not None:
        plt.plot(
            orb_pairs["frame_index_a"],
            orb_pairs["inlier_ratio"],
            alpha=0.8,
            label="ORB",
        )
    plt.xlabel("Sequence frame index")
    plt.ylabel("RANSAC inlier ratio")
    plt.title("S6A.4A pairwise geometric consistency")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        figures_dir
        / "s6a4a_klt_vs_orb_inlier_ratio.png",
        dpi=180,
    )
    plt.close()

    plt.figure(figsize=(9, 8))
    plt.plot(
        aligned_df["reference_x_m"],
        aligned_df["reference_y_m"],
        label="Reference — evaluation only",
    )
    plt.plot(
        aligned_df["prefix_aligned_x_m"],
        aligned_df["prefix_aligned_y_m"],
        label="KLT prefix-locked",
    )
    if orb_aligned is not None:
        plt.plot(
            orb_aligned["prefix_aligned_x_m"],
            orb_aligned["prefix_aligned_y_m"],
            alpha=0.8,
            label="ORB prefix-locked",
        )
    plt.axis("equal")
    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")
    plt.title(
        "S6A.4A KLT versus ORB prefix-locked trajectory"
    )
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        figures_dir
        / "s6a4a_klt_vs_orb_prefix_locked_xy.png",
        dpi=180,
    )
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.plot(
        aligned_df[
            "reference_cumulative_distance_m"
        ],
        aligned_df["prefix_locked_error_m"],
        label="KLT",
    )
    if orb_aligned is not None:
        plt.plot(
            orb_aligned[
                "reference_cumulative_distance_m"
            ],
            orb_aligned["prefix_locked_error_m"],
            alpha=0.8,
            label="ORB",
        )
    for threshold in thresholds_m:
        plt.axhline(
            threshold,
            linestyle="--",
            linewidth=1.0,
            alpha=0.7,
        )
    plt.xlabel(
        "Reference cumulative distance [m] — evaluation only"
    )
    plt.ylabel("Position error [m]")
    plt.title(
        "S6A.4A KLT versus ORB error growth"
    )
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        figures_dir
        / "s6a4a_klt_vs_orb_error_vs_distance.png",
        dpi=180,
    )
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.plot(
        aligned_df["sequence_frame_id"],
        aligned_df["visual_yaw_deg_unwrapped"],
        label="KLT",
    )
    if orb_aligned is not None:
        plt.plot(
            orb_aligned["sequence_frame_id"],
            orb_aligned["visual_yaw_deg_unwrapped"],
            alpha=0.8,
            label="ORB",
        )
    plt.xlabel("Sequence frame index")
    plt.ylabel("Accumulated visual yaw [deg]")
    plt.title(
        "S6A.4A KLT versus ORB accumulated yaw"
    )
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        figures_dir
        / "s6a4a_klt_vs_orb_accumulated_yaw.png",
        dpi=180,
    )
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "S6A.4A: KLT optical-flow relative-motion baseline "
            "and direct comparison with the locked ORB baseline."
        )
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
    )
    parser.add_argument(
        "--orb-pairs",
        type=Path,
        default=DEFAULT_ORB_PAIRS,
    )
    parser.add_argument(
        "--orb-aligned",
        type=Path,
        default=DEFAULT_ORB_ALIGNED,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    parser.add_argument(
        "--sequence",
        default="traj01",
    )
    parser.add_argument(
        "--resize-long",
        type=int,
        default=960,
    )
    parser.add_argument(
        "--max-corners",
        type=int,
        default=1200,
    )
    parser.add_argument(
        "--quality-level",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--min-distance",
        type=float,
        default=7.0,
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=7,
    )
    parser.add_argument(
        "--win-size",
        type=int,
        default=31,
    )
    parser.add_argument(
        "--max-level",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--fb-threshold-px",
        type=float,
        default=1.5,
    )
    parser.add_argument(
        "--ransac-threshold-px",
        type=float,
        default=3.0,
    )
    parser.add_argument(
        "--min-fb-tracks",
        type=int,
        default=40,
    )
    parser.add_argument(
        "--min-inliers",
        type=int,
        default=30,
    )
    parser.add_argument(
        "--min-inlier-ratio",
        type=float,
        default=0.50,
    )
    parser.add_argument(
        "--alignment-prefix-frames",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--thresholds-m",
        type=parse_float_list,
        default=parse_float_list("10,20,40,80"),
    )
    parser.add_argument(
        "--sustain-frames",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
    )
    args = parser.parse_args()

    if not args.manifest.exists():
        raise FileNotFoundError(
            f"Missing manifest: {args.manifest}"
        )

    manifest = pd.read_csv(args.manifest)

    if "sequence" in manifest.columns:
        manifest = manifest[
            manifest["sequence"].astype(str)
            == args.sequence
        ].copy()

    required_manifest = {
        "sequence_frame_id",
        "token0_id",
        "image_path",
        "x_enu_m",
        "y_enu_m",
    }
    missing = sorted(
        required_manifest.difference(manifest.columns)
    )
    if missing:
        raise RuntimeError(
            f"Manifest missing columns: {missing}"
        )

    manifest["sequence_frame_id"] = pd.to_numeric(
        manifest["sequence_frame_id"],
        errors="coerce",
    )
    manifest = manifest.dropna(
        subset=["sequence_frame_id"]
    )
    manifest["sequence_frame_id"] = (
        manifest["sequence_frame_id"].astype(int)
    )
    manifest = manifest.sort_values(
        "sequence_frame_id",
        kind="mergesort",
    ).reset_index(drop=True)

    if args.max_frames is not None:
        manifest = manifest.head(
            args.max_frames
        ).copy()

    if len(manifest) < 3:
        raise RuntimeError(
            "At least three frames are required."
        )

    metadata_dir = (
        args.output_root
        / "metadata"
        / "s6a_relative_motion"
    )
    reports_dir = (
        args.output_root
        / "reports"
        / "s6a_relative_motion"
    )
    figures_dir = (
        args.output_root
        / "figures"
        / "s6a_relative_motion"
    )
    metadata_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    pair_path = (
        metadata_dir
        / "s6a4a_klt_pair_diagnostics.csv"
    )
    checkpoint_path = (
        metadata_dir
        / "s6a4a_klt_pair_diagnostics_checkpoint.csv"
    )
    failed_path = (
        metadata_dir
        / "s6a4a_klt_failed_affine_pairs.csv"
    )
    hybrid_pair_path = (
        metadata_dir
        / "s6a4a_klt_orb_hybrid_pair_diagnostics.csv"
    )
    failure_report_path = (
        reports_dir
        / "s6a4a_klt_failure_diagnostic.json"
    )

    pair_rows: list[dict[str, Any]] = []

    previous_path = resolve_image_path(
        manifest.iloc[0]["image_path"]
    )
    previous_image = read_gray(
        previous_path,
        args.resize_long,
    )

    image_height, image_width = previous_image.shape[:2]

    pair_start = time.perf_counter()

    for frame_index in range(len(manifest) - 1):
        current_row = manifest.iloc[frame_index]
        next_row = manifest.iloc[frame_index + 1]

        next_path = resolve_image_path(
            next_row["image_path"]
        )
        next_image = read_gray(
            next_path,
            args.resize_long,
        )

        if next_image.shape != previous_image.shape:
            raise RuntimeError(
                "Resized image dimensions changed inside sequence."
            )

        result = estimate_klt_pair(
            previous_image,
            next_image,
            max_corners=args.max_corners,
            quality_level=args.quality_level,
            min_distance=args.min_distance,
            block_size=args.block_size,
            win_size=args.win_size,
            max_level=args.max_level,
            fb_threshold_px=args.fb_threshold_px,
            ransac_threshold_px=(
                args.ransac_threshold_px
            ),
        )

        pair_safe = (
            bool(result.get("affine_ok", False))
            and int(
                result.get(
                    "fb_consistent_tracks",
                    0,
                )
            )
            >= args.min_fb_tracks
            and int(result.get("inliers", 0))
            >= args.min_inliers
            and float(
                result.get(
                    "inlier_ratio",
                    0.0,
                )
            )
            >= args.min_inlier_ratio
            and 0.70
            <= float(
                result.get(
                    "affine_scale",
                    float("nan"),
                )
            )
            <= 1.40
        )

        reference_dx = float(
            next_row["x_enu_m"]
            - current_row["x_enu_m"]
        )
        reference_dy = float(
            next_row["y_enu_m"]
            - current_row["y_enu_m"]
        )

        pair_rows.append(
            {
                "method": "klt",
                "stride": 1,
                "frame_index_a": int(
                    current_row["sequence_frame_id"]
                ),
                "frame_index_b": int(
                    next_row["sequence_frame_id"]
                ),
                "token0_a": int(
                    current_row["token0_id"]
                ),
                "token0_b": int(
                    next_row["token0_id"]
                ),
                "image_path_a": str(previous_path),
                "image_path_b": str(next_path),
                "pair_safe_image_only": bool(
                    pair_safe
                ),
                "reference_dx_m": reference_dx,
                "reference_dy_m": reference_dy,
                "reference_step_m": float(
                    math.hypot(
                        reference_dx,
                        reference_dy,
                    )
                ),
                **result,
            }
        )

        previous_image = next_image
        previous_path = next_path

        processed = frame_index + 1
        total = len(manifest) - 1
        if processed % 100 == 0 or processed == total:
            print(
                f"KLT pairs: {processed}/{total}"
            )
            pd.DataFrame(pair_rows).to_csv(
                checkpoint_path,
                index=False,
            )

    pair_elapsed = time.perf_counter() - pair_start
    pair_df = pd.DataFrame(pair_rows)
    pair_df.to_csv(pair_path, index=False)

    # Preserve the raw KLT result before applying any fallback.
    pair_df["motion_source"] = "klt"
    pair_df["fallback_reason"] = ""
    fallback_count = 0

    failed_affine = ~pair_df["affine_ok"].astype(bool)

    if bool(failed_affine.any()):
        if not args.orb_pairs.exists():
            raise FileNotFoundError(
                "KLT has failed affine pairs and the ORB pair CSV "
                f"is unavailable: {args.orb_pairs}"
            )

        orb_fallback_df = pd.read_csv(args.orb_pairs)

        if "stride" in orb_fallback_df.columns:
            orb_fallback_df = orb_fallback_df[
                pd.to_numeric(
                    orb_fallback_df["stride"],
                    errors="coerce",
                )
                == 1
            ].copy()

        for column in ["frame_index_a", "frame_index_b"]:
            orb_fallback_df[column] = pd.to_numeric(
                orb_fallback_df[column],
                errors="coerce",
            )

        required_orb_columns = [
            "affine_a00",
            "affine_a01",
            "affine_a10",
            "affine_a11",
            "affine_tx_px",
            "affine_ty_px",
            "affine_rotation_deg",
        ]

        unresolved_indices = []

        for pair_index in pair_df.index[failed_affine]:
            frame_a = int(
                pair_df.at[pair_index, "frame_index_a"]
            )
            frame_b = int(
                pair_df.at[pair_index, "frame_index_b"]
            )

            match = orb_fallback_df[
                (
                    orb_fallback_df["frame_index_a"]
                    == frame_a
                )
                & (
                    orb_fallback_df["frame_index_b"]
                    == frame_b
                )
            ]

            if len(match) != 1:
                unresolved_indices.append(pair_index)
                continue

            orb_row = match.iloc[0]

            missing_transform = [
                column
                for column in required_orb_columns
                if (
                    column not in orb_row.index
                    or pd.isna(orb_row[column])
                )
            ]

            if missing_transform:
                unresolved_indices.append(pair_index)
                continue

            orb_affine_ok = (
                str(orb_row.get("affine_ok", ""))
                .strip()
                .lower()
                in {"true", "1", "yes", "y", "t"}
            )

            orb_status = str(
                orb_row.get("status", "")
            ).strip().lower()

            if not orb_affine_ok:
                unresolved_indices.append(pair_index)
                continue

            # Preserve the original KLT diagnostics.
            pair_df.at[
                pair_index,
                "klt_status_original",
            ] = pair_df.at[pair_index, "status"]

            pair_df.at[
                pair_index,
                "klt_fb_consistent_tracks_original",
            ] = pair_df.at[
                pair_index,
                "fb_consistent_tracks",
            ]

            pair_df.at[
                pair_index,
                "klt_inliers_original",
            ] = pair_df.at[pair_index, "inliers"]

            pair_df.at[
                pair_index,
                "klt_inlier_ratio_original",
            ] = pair_df.at[
                pair_index,
                "inlier_ratio",
            ]

            # Copy only the exact ORB pair's geometric estimate.
            copy_columns = [
                "affine_a00",
                "affine_a01",
                "affine_a10",
                "affine_a11",
                "affine_tx_px",
                "affine_ty_px",
                "affine_rotation_deg",
                "affine_scale",
                "center_content_dx_px",
                "center_content_dy_px",
                "center_content_motion_px",
                "inliers",
                "inlier_ratio",
            ]

            for column in copy_columns:
                if (
                    column in orb_row.index
                    and pd.notna(orb_row[column])
                ):
                    pair_df.at[
                        pair_index,
                        column,
                    ] = orb_row[column]

            if "good_matches" in orb_row.index:
                pair_df.at[
                    pair_index,
                    "orb_fallback_good_matches",
                ] = orb_row["good_matches"]

            pair_df.at[
                pair_index,
                "affine_ok",
            ] = True

            pair_df.at[
                pair_index,
                "status",
            ] = "orb_fallback_good"

            pair_df.at[
                pair_index,
                "motion_source",
            ] = "orb_fallback"

            pair_df.at[
                pair_index,
                "fallback_reason",
            ] = pair_df.at[
                pair_index,
                "klt_status_original",
            ]

            # S6A.1 already validated the ORB stride-1 chain.
            # Prefer its explicit quality status when available.
            pair_df.at[
                pair_index,
                "pair_safe_image_only",
            ] = (
                orb_status == "good"
                or float(
                    orb_row.get("inlier_ratio", 0.0)
                )
                >= 0.35
            )

            fallback_count += 1

            print(
                "\nApplied ORB fallback:"
                f" frame {frame_a} -> {frame_b}"
            )
            print(
                "  KLT reason: "
                f"{pair_df.at[pair_index, 'fallback_reason']}"
            )
            print(
                "  ORB inliers: "
                f"{pair_df.at[pair_index, 'inliers']}"
            )
            print(
                "  ORB inlier ratio: "
                f"{pair_df.at[pair_index, 'inlier_ratio']:.3f}"
            )
            print(
                "  ORB rotation [deg]: "
                f"{pair_df.at[pair_index, 'affine_rotation_deg']:.3f}"
            )

        pair_df.to_csv(
            hybrid_pair_path,
            index=False,
        )

        unresolved_affine = (
            ~pair_df["affine_ok"].astype(bool)
        )

        if bool(unresolved_affine.any()):
            failed_df = pair_df.loc[
                unresolved_affine
            ].copy()

            failed_df.to_csv(
                failed_path,
                index=False,
            )

            status_counts = (
                failed_df["status"]
                .astype(str)
                .value_counts(dropna=False)
                .to_dict()
            )

            failure_report = {
                "stage": (
                    "S6A.4A_unresolved_after_orb_fallback"
                ),
                "sequence": args.sequence,
                "pairs_processed": int(len(pair_df)),
                "orb_fallback_pairs": int(
                    fallback_count
                ),
                "unresolved_pairs": int(
                    len(failed_df)
                ),
                "status_counts": {
                    str(key): int(value)
                    for key, value
                    in status_counts.items()
                },
                "failed_pairs": failed_df[
                    [
                        "frame_index_a",
                        "frame_index_b",
                        "token0_a",
                        "token0_b",
                        "status",
                        "detected_corners",
                        "tracked_forward",
                        "fb_consistent_tracks",
                        "inliers",
                        "inlier_ratio",
                        "reference_step_m",
                    ]
                ].to_dict(orient="records"),
            }

            with failure_report_path.open(
                "w",
                encoding="utf-8",
            ) as handle:
                json.dump(
                    failure_report,
                    handle,
                    indent=2,
                )

            print(
                "\nUnresolved KLT/ORB pairs remain:"
            )
            print(
                failed_df[
                    [
                        "frame_index_a",
                        "frame_index_b",
                        "status",
                    ]
                ].to_string(index=False)
            )
            print(failed_path)
            print(failure_report_path)
            return

    relative_method_name = (
        "klt_orb_fallback"
        if fallback_count > 0
        else "klt"
    )


    trajectory = integrate_pair_chain(
        pair_df,
        width=image_width,
        height=image_height,
    )

    merged = manifest[
        [
            "sequence_frame_id",
            "token0_id",
            "x_enu_m",
            "y_enu_m",
        ]
    ].merge(
        trajectory,
        on="sequence_frame_id",
        how="inner",
        validate="one_to_one",
    )

    reference = merged[
        ["x_enu_m", "y_enu_m"]
    ].to_numpy(dtype=float)
    reference = reference - reference[0]

    reference_steps = np.linalg.norm(
        np.diff(reference, axis=0),
        axis=1,
    )
    cumulative_distance = np.concatenate(
        [[0.0], np.cumsum(reference_steps)]
    )

    visual = merged[
        ["visual_x_px", "visual_y_px"]
    ].to_numpy(dtype=float)

    global_transform = fit_similarity(
        visual,
        reference,
    )
    global_aligned = apply_similarity(
        visual,
        global_transform,
    )

    prefix_count = min(
        max(args.alignment_prefix_frames, 3),
        len(merged),
    )
    prefix_transform = fit_similarity(
        visual[:prefix_count],
        reference[:prefix_count],
    )
    prefix_aligned = apply_similarity(
        visual,
        prefix_transform,
    )

    global_metrics = error_metrics(
        global_aligned,
        reference,
        cumulative_distance,
        evaluation_start_index=0,
    )
    prefix_metrics = error_metrics(
        prefix_aligned,
        reference,
        cumulative_distance,
        evaluation_start_index=prefix_count - 1,
    )

    global_errors = np.linalg.norm(
        global_aligned - reference,
        axis=1,
    )
    prefix_errors = np.linalg.norm(
        prefix_aligned - reference,
        axis=1,
    )

    crossings = [
        first_sustained_crossing(
            prefix_errors,
            cumulative_distance,
            threshold_m=threshold,
            start_index=prefix_count - 1,
            sustain_frames=args.sustain_frames,
        )
        for threshold in args.thresholds_m
    ]

    aligned_df = merged.copy()
    aligned_df["method"] = relative_method_name
    aligned_df["reference_x_m"] = reference[:, 0]
    aligned_df["reference_y_m"] = reference[:, 1]
    aligned_df[
        "reference_cumulative_distance_m"
    ] = cumulative_distance
    aligned_df["global_aligned_x_m"] = (
        global_aligned[:, 0]
    )
    aligned_df["global_aligned_y_m"] = (
        global_aligned[:, 1]
    )
    aligned_df["global_alignment_error_m"] = (
        global_errors
    )
    aligned_df["prefix_aligned_x_m"] = (
        prefix_aligned[:, 0]
    )
    aligned_df["prefix_aligned_y_m"] = (
        prefix_aligned[:, 1]
    )
    aligned_df["prefix_locked_error_m"] = (
        prefix_errors
    )

    orb_pairs, orb_aligned = load_orb_comparison(
        args.orb_pairs,
        args.orb_aligned,
    )

    # A smoke test must compare KLT and ORB on exactly the same
    # frame prefix. The original version accidentally compared a
    # 150-frame KLT run with the complete 1034-frame ORB trajectory.
    if orb_pairs is not None:
        orb_pairs = orb_pairs[
            pd.to_numeric(
                orb_pairs["frame_index_b"],
                errors="coerce",
            )
            < len(manifest)
        ].copy()

    if orb_aligned is not None:
        orb_aligned = orb_aligned[
            pd.to_numeric(
                orb_aligned["sequence_frame_id"],
                errors="coerce",
            )
            < len(manifest)
        ].copy()

    pair_summary = {
        "method": relative_method_name,
        "frames": int(len(manifest)),
        "pairs": int(len(pair_df)),
        "orb_fallback_pairs": int(fallback_count),
        "orb_fallback_rate": float(
            fallback_count / max(len(pair_df), 1)
        ),
        "affine_success_rate": float(
            pair_df["affine_ok"].mean()
        ),
        "image_only_good_rate": float(
            pair_df["pair_safe_image_only"].mean()
        ),
        "detected_corners_median": float(
            pair_df["detected_corners"].median()
        ),
        "fb_consistent_tracks_median": float(
            pair_df["fb_consistent_tracks"].median()
        ),
        "inliers_median": float(
            pair_df["inliers"].median()
        ),
        "inlier_ratio_median": float(
            pair_df["inlier_ratio"].median()
        ),
        "inlier_ratio_p05": finite_percentile(
            pair_df["inlier_ratio"],
            5.0,
        ),
        "fb_error_median_px": float(
            pair_df["fb_error_median_px"].median()
        ),
        "abs_rotation_p95_deg": finite_percentile(
            np.abs(
                pair_df["affine_rotation_deg"]
            ),
            95.0,
        ),
        "abs_scale_error_p95": finite_percentile(
            np.abs(
                pair_df["affine_scale"] - 1.0
            ),
            95.0,
        ),
        "pair_runtime_mean_ms": float(
            pair_df["runtime_ms"].mean()
        ),
        "full_pair_processing_s": float(
            pair_elapsed
        ),
    }

    trajectory_summary = {
        "global_alignment": {
            "scale_m_per_px": float(
                global_transform["scale"]
            ),
            "rotation_deg": float(
                global_transform["rotation_deg"]
            ),
            **global_metrics,
        },
        "prefix_locked_alignment": {
            "prefix_frames": int(prefix_count),
            "prefix_reference_distance_m": float(
                cumulative_distance[prefix_count - 1]
            ),
            "scale_m_per_px": float(
                prefix_transform["scale"]
            ),
            "rotation_deg": float(
                prefix_transform["rotation_deg"]
            ),
            **prefix_metrics,
        },
        "threshold_crossings": crossings,
    }

    metadata_dir = (
        args.output_root
        / "metadata"
        / "s6a_relative_motion"
    )
    reports_dir = (
        args.output_root
        / "reports"
        / "s6a_relative_motion"
    )
    figures_dir = (
        args.output_root
        / "figures"
        / "s6a_relative_motion"
    )
    metadata_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    pair_path = (
        metadata_dir
        / "s6a4a_klt_pair_diagnostics.csv"
    )
    trajectory_path = (
        metadata_dir
        / "s6a4a_klt_relative_trajectory_aligned.csv"
    )
    summary_path = (
        metadata_dir
        / "s6a4a_klt_vs_orb_summary.csv"
    )
    report_path = (
        reports_dir
        / "s6a4a_klt_vs_orb_comparison.json"
    )

    if fallback_count > 0:
        pair_df.to_csv(
            hybrid_pair_path,
            index=False,
        )
    else:
        pair_df.to_csv(
            pair_path,
            index=False,
        )

    aligned_df.to_csv(
        trajectory_path,
        index=False,
    )

    comparison_rows = [
        {
            "method": relative_method_name,
            "affine_success_rate": pair_summary[
                "affine_success_rate"
            ],
            "good_rate": pair_summary[
                "image_only_good_rate"
            ],
            "inlier_ratio_median": pair_summary[
                "inlier_ratio_median"
            ],
            "inlier_ratio_p05": pair_summary[
                "inlier_ratio_p05"
            ],
            "inliers_median": pair_summary[
                "inliers_median"
            ],
            "pair_runtime_mean_ms": pair_summary[
                "pair_runtime_mean_ms"
            ],
            "prefix_rmse_m": prefix_metrics[
                "rmse_m"
            ],
            "prefix_p95_m": prefix_metrics[
                "p95_error_m"
            ],
            "prefix_max_m": prefix_metrics[
                "max_error_m"
            ],
            "final_error_m": prefix_metrics[
                "final_error_m"
            ],
            "final_drift_per_100m": prefix_metrics[
                "final_drift_per_100m"
            ],
        }
    ]

    if orb_pairs is not None and orb_aligned is not None:
        orb_errors = orb_aligned[
            "prefix_locked_error_m"
        ].to_numpy(dtype=float)
        orb_prefix_start = prefix_count - 1
        orb_tail = orb_errors[orb_prefix_start:]
        orb_distance = float(
            orb_aligned[
                "reference_cumulative_distance_m"
            ].iloc[-1]
            - orb_aligned[
                "reference_cumulative_distance_m"
            ].iloc[orb_prefix_start]
        )

        comparison_rows.append(
            {
                "method": "orb",
                "affine_success_rate": float(
                    orb_pairs["affine_ok"].mean()
                ),
                "good_rate": float(
                    (
                        orb_pairs["status"].astype(str)
                        == "good"
                    ).mean()
                ),
                "inlier_ratio_median": float(
                    orb_pairs[
                        "inlier_ratio"
                    ].median()
                ),
                "inlier_ratio_p05": finite_percentile(
                    orb_pairs["inlier_ratio"],
                    5.0,
                ),
                "inliers_median": float(
                    orb_pairs["inliers"].median()
                ),
                "pair_runtime_mean_ms": float(
                    orb_pairs[
                        "runtime_ms"
                    ].mean()
                )
                if "runtime_ms" in orb_pairs.columns
                else float("nan"),
                "prefix_rmse_m": float(
                    math.sqrt(
                        np.mean(
                            orb_tail * orb_tail
                        )
                    )
                ),
                "prefix_p95_m": float(
                    np.percentile(
                        orb_tail,
                        95,
                    )
                ),
                "prefix_max_m": float(
                    np.max(orb_tail)
                ),
                "final_error_m": float(
                    orb_errors[-1]
                ),
                "final_drift_per_100m": (
                    100.0
                    * float(orb_errors[-1])
                    / orb_distance
                ),
            }
        )

    comparison_df = pd.DataFrame(
        comparison_rows
    )
    comparison_df.to_csv(
        summary_path,
        index=False,
    )

    save_plots(
        pair_df=pair_df,
        aligned_df=aligned_df,
        orb_pairs=orb_pairs,
        orb_aligned=orb_aligned,
        figures_dir=figures_dir,
        thresholds_m=args.thresholds_m,
    )

    report = {
        "stage": "S6A.4A",
        "sequence": args.sequence,
        "purpose": (
            "Compare KLT optical-flow relative motion with the "
            "locked ORB partial-affine baseline before freezing "
            "the relative estimator for S6B."
        ),
        "reference_rule": (
            "Reference ENU is used only after image-only KLT "
            "estimation for alignment, error metrics, and method "
            "comparison."
        ),
        "important_rotation_rule": (
            "Large rotations are not rejected by magnitude alone. "
            "KLT acceptance depends on forward/backward point "
            "consistency and affine RANSAC support."
        ),
        "configuration": vars(args),
        "klt_pair_summary": pair_summary,
        "klt_trajectory_summary": trajectory_summary,
        "method_comparison": comparison_df.to_dict(
            orient="records"
        ),
        "decision_rule": (
            "If KLT is clearly worse in weak-tail pair quality "
            "or prefix-locked drift, freeze ORB for S6B. If KLT "
            "is competitive or complementary, run KLT rolling-"
            "anchor horizons and test an ORB/KLT confidence-gated "
            "hybrid before freezing."
        ),
    }

    # Convert Path objects for JSON serialization.
    report["configuration"] = {
        key: str(value)
        if isinstance(value, Path)
        else value
        for key, value in report[
            "configuration"
        ].items()
    }

    with report_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            report,
            handle,
            indent=2,
        )

    print("\nS6A.4A KLT optical-flow comparison")
    print("-----------------------------------")
    print(f"Frames:                    {len(manifest)}")
    print(f"Pairs:                     {len(pair_df)}")
    print(
        "Affine success rate:       "
        f"{pair_summary['affine_success_rate']:.3f}"
    )
    print(
        "Image-only good rate:      "
        f"{pair_summary['image_only_good_rate']:.3f}"
    )
    print(
        "FB tracks median:          "
        f"{pair_summary['fb_consistent_tracks_median']:.1f}"
    )
    print(
        "RANSAC inliers median:     "
        f"{pair_summary['inliers_median']:.1f}"
    )
    print(
        "Inlier ratio median:       "
        f"{pair_summary['inlier_ratio_median']:.3f}"
    )
    print(
        "Inlier ratio p05:          "
        f"{pair_summary['inlier_ratio_p05']:.3f}"
    )
    print(
        "Point FB error median px:  "
        f"{pair_summary['fb_error_median_px']:.3f}"
    )
    print(
        "|rotation| p95 deg:        "
        f"{pair_summary['abs_rotation_p95_deg']:.3f}"
    )
    print(
        "|scale-1| p95:             "
        f"{pair_summary['abs_scale_error_p95']:.6f}"
    )
    print(
        "Pair runtime mean ms:      "
        f"{pair_summary['pair_runtime_mean_ms']:.2f}"
    )

    print("\nKLT prefix-locked trajectory")
    print("----------------------------")
    print(
        "Prefix RMSE [m]:           "
        f"{prefix_metrics['rmse_m']:.3f}"
    )
    print(
        "Prefix p95 [m]:            "
        f"{prefix_metrics['p95_error_m']:.3f}"
    )
    print(
        "Prefix max [m]:            "
        f"{prefix_metrics['max_error_m']:.3f}"
    )
    print(
        "Final error [m]:           "
        f"{prefix_metrics['final_error_m']:.3f}"
    )
    print(
        "Final drift [m/100m]:      "
        f"{prefix_metrics['final_drift_per_100m']:.3f}"
    )

    for crossing in crossings:
        if crossing["crossed"]:
            print(
                f"Sustained "
                f"{crossing['threshold_m']:g} m crossing: "
                f"{crossing['distance_after_prefix_m']:.1f} m, "
                f"{crossing['frames_after_prefix']} frames"
            )
        else:
            print(
                f"Sustained "
                f"{crossing['threshold_m']:g} m crossing: "
                "not reached"
            )

    print("\nMethod comparison")
    print("-----------------")
    print(
        comparison_df.to_string(
            index=False,
        )
    )

    print("\nSaved outputs")
    print("-------------")
    print(pair_path)
    print(trajectory_path)
    print(summary_path)
    print(report_path)
    print(figures_dir)


if __name__ == "__main__":
    main()
