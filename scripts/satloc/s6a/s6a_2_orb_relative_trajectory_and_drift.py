from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_OUTPUT_ROOT = Path("outputs/satloc")
DEFAULT_MANIFEST = Path(
    "outputs/satloc/metadata/s6a_relative_motion/s6a_sequence_manifest.csv"
)
DEFAULT_PAIR_CSV = Path(
    "outputs/satloc/metadata/s6a_relative_motion/"
    "s6a1_orb_affine_pair_diagnostics.csv"
)
DEFAULT_FEATURE_CSV = Path(
    "outputs/satloc/metadata/s6a_relative_motion/"
    "s6a1_orb_frame_features.csv"
)
DEFAULT_ENRICHED_CSV = Path(
    "outputs/satloc/metadata/s6a_relative_motion/"
    "s6a1b_pair_diagnostics_enriched.csv"
)


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


def bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    normalized = series.astype(str).str.strip().str.lower()
    return normalized.isin({"true", "1", "yes", "y", "t"})


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

    # Re-orthogonalize to the nearest proper 2-D rotation.
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
    try:
        inverse = np.linalg.inv(scene_affine)
    except np.linalg.LinAlgError as error:
        raise RuntimeError(
            "Encountered a singular affine transform."
        ) from error

    camera_b_center_in_a = inverse @ center
    camera_b_center_in_a /= camera_b_center_in_a[2]

    step_image = camera_b_center_in_a[:2] - center[:2]

    # Convert OpenCV image axes (x right, y down) to a Cartesian
    # visual frame (x right, y up).
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


def integrate_local_steps(
    pair_df: pd.DataFrame,
    center: np.ndarray,
    normalize_scale: bool,
) -> pd.DataFrame:
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
        matrix = affine_from_row(pair)
        working = (
            normalize_affine_scale_about_center(matrix, center)
            if normalize_scale
            else matrix
        )

        local_step = local_camera_step_from_scene_affine(
            working,
            center,
        )
        global_step = rotation_matrix(yaw_rad) @ local_step
        position = position + global_step

        # OpenCV's positive affine angle is clockwise on a y-down image.
        # After converting to y-up coordinates it equals the camera yaw
        # increment needed for the trajectory frame.
        yaw_increment_rad = math.radians(
            float(pair["affine_rotation_deg"])
        )
        yaw_rad += yaw_increment_rad

        pair_safe = bool(pair.get("pair_safe_image_only", True))

        rows.append(
            {
                "sequence_frame_id": int(pair["frame_index_b"]),
                "visual_x_px": float(position[0]),
                "visual_y_px": float(position[1]),
                "visual_yaw_rad": float(yaw_rad),
                "visual_yaw_deg_unwrapped": float(
                    math.degrees(yaw_rad)
                ),
                "step_x_local_px": float(local_step[0]),
                "step_y_local_px": float(local_step[1]),
                "step_x_global_px": float(global_step[0]),
                "step_y_global_px": float(global_step[1]),
                "step_motion_px": float(
                    np.linalg.norm(global_step)
                ),
                "pair_safe_image_only": pair_safe,
            }
        )

    return pd.DataFrame(rows)


def fit_similarity(
    source: np.ndarray,
    target: np.ndarray,
) -> dict[str, Any]:
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)

    valid = (
        np.isfinite(source).all(axis=1)
        & np.isfinite(target).all(axis=1)
    )
    source = source[valid]
    target = target[valid]

    if len(source) < 3:
        raise RuntimeError(
            "At least three finite points are required for alignment."
        )

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
            "Visual trajectory has insufficient spread for alignment."
        )

    scale = float(np.sum(singular_values) / denominator)
    translation = (
        target_mean - scale * (rotation @ source_mean)
    )

    return {
        "scale": scale,
        "rotation": rotation,
        "translation": translation,
        "rotation_deg": float(
            math.degrees(
                math.atan2(rotation[1, 0], rotation[0, 0])
            )
        ),
        "fit_points": int(len(source)),
    }


def apply_similarity(
    points: np.ndarray,
    transform: dict[str, Any],
) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    return (
        transform["scale"]
        * (points @ transform["rotation"].T)
        + transform["translation"]
    )


def trajectory_path_length(points: np.ndarray) -> float:
    differences = np.diff(np.asarray(points, dtype=float), axis=0)
    return float(
        np.sum(np.linalg.norm(differences, axis=1))
    )


def finite_stat(
    values: np.ndarray | pd.Series,
    function,
    default: float = float("nan"),
) -> float:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return default
    return float(function(array))


def error_summary(
    errors: np.ndarray,
    reference_distance: np.ndarray,
    evaluation_start_index: int,
) -> dict[str, float]:
    tail_errors = np.asarray(
        errors[evaluation_start_index:],
        dtype=float,
    )
    tail_distance = np.asarray(
        reference_distance[evaluation_start_index:],
        dtype=float,
    )

    valid = np.isfinite(tail_errors)
    tail_errors = tail_errors[valid]
    tail_distance = tail_distance[valid]

    if len(tail_errors) == 0:
        return {}

    distance_from_evaluation_start = (
        tail_distance - tail_distance[0]
    )
    final_distance = float(
        distance_from_evaluation_start[-1]
    )
    final_error = float(tail_errors[-1])

    return {
        "evaluation_points": int(len(tail_errors)),
        "rmse_m": float(
            math.sqrt(np.mean(tail_errors * tail_errors))
        ),
        "mean_error_m": float(np.mean(tail_errors)),
        "median_error_m": float(np.median(tail_errors)),
        "p95_error_m": float(
            np.percentile(tail_errors, 95)
        ),
        "max_error_m": float(np.max(tail_errors)),
        "final_error_m": final_error,
        "evaluation_distance_m": final_distance,
        "final_drift_per_100m": (
            100.0 * final_error / final_distance
            if final_distance > 1e-9
            else float("nan")
        ),
    }


def first_sustained_crossing(
    errors: np.ndarray,
    reference_distance: np.ndarray,
    threshold_m: float,
    start_index: int,
    sustain_frames: int,
) -> dict[str, Any]:
    errors = np.asarray(errors, dtype=float)
    reference_distance = np.asarray(
        reference_distance,
        dtype=float,
    )

    start_distance = float(reference_distance[start_index])
    above = (
        np.isfinite(errors)
        & (errors >= threshold_m)
    )

    for index in range(
        start_index,
        len(errors) - sustain_frames + 1,
    ):
        if bool(np.all(above[index : index + sustain_frames])):
            return {
                "threshold_m": float(threshold_m),
                "crossed": True,
                "frame_index": int(index),
                "frames_after_alignment_prefix": int(
                    index - start_index
                ),
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


def load_inputs(
    manifest_path: Path,
    pair_path: Path,
    feature_path: Path,
    enriched_path: Path,
    sequence: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    for path in (manifest_path, pair_path, feature_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing input: {path}")

    manifest = pd.read_csv(manifest_path)
    pairs = pd.read_csv(pair_path)
    features = pd.read_csv(feature_path)

    if "sequence" in manifest.columns:
        manifest = manifest[
            manifest["sequence"].astype(str) == sequence
        ].copy()

    required_manifest = {
        "sequence_frame_id",
        "token0_id",
        "x_enu_m",
        "y_enu_m",
    }
    missing_manifest = sorted(
        required_manifest.difference(manifest.columns)
    )
    if missing_manifest:
        raise RuntimeError(
            f"Manifest missing columns: {missing_manifest}"
        )

    required_pair = {
        "stride",
        "frame_index_a",
        "frame_index_b",
        "affine_ok",
        "status",
        "inliers",
        "inlier_ratio",
        "affine_a00",
        "affine_a01",
        "affine_a10",
        "affine_a11",
        "affine_tx_px",
        "affine_ty_px",
        "affine_rotation_deg",
    }
    missing_pair = sorted(
        required_pair.difference(pairs.columns)
    )
    if missing_pair:
        raise RuntimeError(
            f"Pair CSV missing columns: {missing_pair}"
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

    pairs["stride"] = pd.to_numeric(
        pairs["stride"],
        errors="coerce",
    )
    pairs = pairs[pairs["stride"] == 1].copy()
    pairs = pairs.sort_values(
        "frame_index_a",
        kind="mergesort",
    ).reset_index(drop=True)

    pairs["affine_ok"] = bool_series(pairs["affine_ok"])

    numeric_pair_columns = [
        "frame_index_a",
        "frame_index_b",
        "inliers",
        "inlier_ratio",
        "affine_a00",
        "affine_a01",
        "affine_a10",
        "affine_a11",
        "affine_tx_px",
        "affine_ty_px",
        "affine_rotation_deg",
    ]
    for column in numeric_pair_columns:
        pairs[column] = pd.to_numeric(
            pairs[column],
            errors="coerce",
        )

    expected_a = np.arange(len(manifest) - 1, dtype=int)
    expected_b = expected_a + 1
    actual_a = pairs["frame_index_a"].to_numpy(dtype=int)
    actual_b = pairs["frame_index_b"].to_numpy(dtype=int)

    if not (
        np.array_equal(actual_a, expected_a)
        and np.array_equal(actual_b, expected_b)
    ):
        raise RuntimeError(
            "Stride-1 pair chain is incomplete or incorrectly ordered."
        )

    if not bool(pairs["affine_ok"].all()):
        failed = pairs.loc[
            ~pairs["affine_ok"],
            ["frame_index_a", "frame_index_b"],
        ]
        raise RuntimeError(
            "Cannot accumulate a chain with failed affine pairs. "
            f"Failed rows: {failed.head().to_dict(orient='records')}"
        )

    pairs["pair_safe_image_only"] = (
        pairs["status"].astype(str).eq("good")
    )

    if enriched_path.exists():
        enriched = pd.read_csv(enriched_path)
        enriched["stride"] = pd.to_numeric(
            enriched["stride"],
            errors="coerce",
        )
        enriched = enriched[
            enriched["stride"] == 1
        ].copy()

        merge_columns = [
            "frame_index_a",
            "frame_index_b",
        ]
        optional_columns = [
            "fb_cycle_center_error_px",
            "flag_fb_inconsistent",
            "flag_current_weak",
        ]
        available = [
            column
            for column in optional_columns
            if column in enriched.columns
        ]

        pairs = pairs.merge(
            enriched[merge_columns + available],
            on=merge_columns,
            how="left",
            suffixes=("", "_s6a1b"),
        )

        if "flag_fb_inconsistent" in pairs.columns:
            pairs["flag_fb_inconsistent"] = bool_series(
                pairs["flag_fb_inconsistent"]
            )
            pairs["pair_safe_image_only"] &= (
                ~pairs["flag_fb_inconsistent"]
            )

        if "flag_current_weak" in pairs.columns:
            pairs["flag_current_weak"] = bool_series(
                pairs["flag_current_weak"]
            )
            pairs["pair_safe_image_only"] &= (
                ~pairs["flag_current_weak"]
            )

    features["sequence_frame_id"] = pd.to_numeric(
        features["sequence_frame_id"],
        errors="coerce",
    )
    features = features.dropna(
        subset=["sequence_frame_id"]
    )
    features["sequence_frame_id"] = (
        features["sequence_frame_id"].astype(int)
    )
    features = features.sort_values(
        "sequence_frame_id",
        kind="mergesort",
    ).reset_index(drop=True)

    if len(features) != len(manifest):
        raise RuntimeError(
            "Feature metadata and manifest row counts differ."
        )

    return manifest, pairs, features


def build_variant_output(
    variant_name: str,
    trajectory: pd.DataFrame,
    manifest: pd.DataFrame,
    prefix_frames: int,
    thresholds: list[float],
    sustain_frames: int,
) -> tuple[pd.DataFrame, dict[str, Any], list[dict[str, Any]]]:
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

    visual = merged[
        ["visual_x_px", "visual_y_px"]
    ].to_numpy(dtype=float)

    reference_steps = np.linalg.norm(
        np.diff(reference, axis=0),
        axis=1,
    )
    cumulative_reference_distance = np.concatenate(
        [[0.0], np.cumsum(reference_steps)]
    )

    global_transform = fit_similarity(
        visual,
        reference,
    )
    globally_aligned = apply_similarity(
        visual,
        global_transform,
    )
    global_error = np.linalg.norm(
        globally_aligned - reference,
        axis=1,
    )

    prefix_count = min(max(prefix_frames, 3), len(merged))
    prefix_transform = fit_similarity(
        visual[:prefix_count],
        reference[:prefix_count],
    )
    prefix_aligned = apply_similarity(
        visual,
        prefix_transform,
    )
    prefix_error = np.linalg.norm(
        prefix_aligned - reference,
        axis=1,
    )

    output = merged.copy()
    output["variant"] = variant_name
    output["reference_x_m"] = reference[:, 0]
    output["reference_y_m"] = reference[:, 1]
    output["reference_cumulative_distance_m"] = (
        cumulative_reference_distance
    )
    output["global_aligned_x_m"] = globally_aligned[:, 0]
    output["global_aligned_y_m"] = globally_aligned[:, 1]
    output["global_alignment_error_m"] = global_error
    output["prefix_aligned_x_m"] = prefix_aligned[:, 0]
    output["prefix_aligned_y_m"] = prefix_aligned[:, 1]
    output["prefix_locked_error_m"] = prefix_error

    distance_after_prefix = (
        cumulative_reference_distance
        - cumulative_reference_distance[prefix_count - 1]
    )
    output["distance_after_alignment_prefix_m"] = (
        distance_after_prefix
    )
    output["prefix_locked_drift_per_100m"] = np.where(
        distance_after_prefix > 1e-9,
        100.0 * prefix_error / distance_after_prefix,
        np.nan,
    )

    global_summary = error_summary(
        global_error,
        cumulative_reference_distance,
        evaluation_start_index=0,
    )
    prefix_summary = error_summary(
        prefix_error,
        cumulative_reference_distance,
        evaluation_start_index=prefix_count - 1,
    )

    crossings = [
        {
            "variant": variant_name,
            **first_sustained_crossing(
                prefix_error,
                cumulative_reference_distance,
                threshold,
                start_index=prefix_count - 1,
                sustain_frames=sustain_frames,
            ),
        }
        for threshold in thresholds
    ]

    summary = {
        "variant": variant_name,
        "frames": int(len(output)),
        "reference_path_m": trajectory_path_length(reference),
        "visual_path_px": trajectory_path_length(visual),
        "global_alignment": {
            "scale_m_per_px": float(
                global_transform["scale"]
            ),
            "rotation_deg": float(
                global_transform["rotation_deg"]
            ),
            **global_summary,
        },
        "prefix_locked_alignment": {
            "prefix_frames": int(prefix_count),
            "prefix_last_frame_index": int(prefix_count - 1),
            "prefix_reference_distance_m": float(
                cumulative_reference_distance[prefix_count - 1]
            ),
            "scale_m_per_px": float(
                prefix_transform["scale"]
            ),
            "rotation_deg": float(
                prefix_transform["rotation_deg"]
            ),
            **prefix_summary,
        },
    }

    return output, summary, crossings


def save_variant_plots(
    variant_name: str,
    output: pd.DataFrame,
    figures_dir: Path,
    prefix_frames: int,
    thresholds: list[float],
) -> None:
    safe_name = variant_name.replace(" ", "_")
    figures_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 8))
    plt.plot(
        output["reference_x_m"],
        output["reference_y_m"],
        label="Reference — evaluation only",
    )
    plt.plot(
        output["global_aligned_x_m"],
        output["global_aligned_y_m"],
        label="Visual — global shape alignment",
    )
    plt.axis("equal")
    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")
    plt.title(
        f"S6A.2 {variant_name}: globally aligned trajectory"
    )
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        figures_dir / f"s6a2_{safe_name}_global_xy.png",
        dpi=180,
    )
    plt.close()

    plt.figure(figsize=(9, 8))
    plt.plot(
        output["reference_x_m"],
        output["reference_y_m"],
        label="Reference — evaluation only",
    )
    plt.plot(
        output["prefix_aligned_x_m"],
        output["prefix_aligned_y_m"],
        label="Visual — prefix-locked alignment",
    )
    prefix_index = min(prefix_frames, len(output)) - 1
    plt.scatter(
        [output.loc[prefix_index, "reference_x_m"]],
        [output.loc[prefix_index, "reference_y_m"]],
        s=50,
        label="Alignment prefix ends",
    )
    plt.axis("equal")
    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")
    plt.title(
        f"S6A.2 {variant_name}: prefix-locked trajectory"
    )
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        figures_dir / f"s6a2_{safe_name}_prefix_locked_xy.png",
        dpi=180,
    )
    plt.close()

    plt.figure(figsize=(11, 6))
    plt.plot(
        output["reference_cumulative_distance_m"],
        output["prefix_locked_error_m"],
        label="Prefix-locked position error",
    )
    for threshold in thresholds:
        plt.axhline(
            threshold,
            linestyle="--",
            linewidth=1.0,
            label=f"{threshold:g} m threshold",
        )
    plt.xlabel("Reference cumulative distance [m] — evaluation only")
    plt.ylabel("Position error [m]")
    plt.title(
        f"S6A.2 {variant_name}: error growth with distance"
    )
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        figures_dir / f"s6a2_{safe_name}_error_vs_distance.png",
        dpi=180,
    )
    plt.close()

    plt.figure(figsize=(11, 6))
    plt.plot(
        output["reference_cumulative_distance_m"],
        output["prefix_locked_drift_per_100m"],
    )
    plt.xlabel("Reference cumulative distance [m] — evaluation only")
    plt.ylabel("Drift [m per 100 m]")
    plt.title(
        f"S6A.2 {variant_name}: prefix-locked drift rate"
    )
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        figures_dir / f"s6a2_{safe_name}_drift_per_100m.png",
        dpi=180,
    )
    plt.close()

    plt.figure(figsize=(11, 6))
    plt.plot(
        output["sequence_frame_id"],
        output["visual_yaw_deg_unwrapped"],
    )
    plt.xlabel("Sequence frame index")
    plt.ylabel("Accumulated visual yaw [deg]")
    plt.title(
        f"S6A.2 {variant_name}: accumulated visual yaw"
    )
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        figures_dir / f"s6a2_{safe_name}_yaw.png",
        dpi=180,
    )
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "S6A.2: accumulate the SatLoc traj01 stride-1 ORB "
            "partial-affine chain, align only for evaluation, and "
            "measure relative drift growth."
        )
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
    )
    parser.add_argument(
        "--pair-csv",
        type=Path,
        default=DEFAULT_PAIR_CSV,
    )
    parser.add_argument(
        "--feature-csv",
        type=Path,
        default=DEFAULT_FEATURE_CSV,
    )
    parser.add_argument(
        "--enriched-csv",
        type=Path,
        default=DEFAULT_ENRICHED_CSV,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    parser.add_argument("--sequence", default="traj01")
    parser.add_argument(
        "--alignment-prefix-frames",
        type=int,
        default=50,
        help=(
            "Number of initial frames used for the evaluation-only "
            "prefix similarity alignment."
        ),
    )
    parser.add_argument(
        "--error-thresholds-m",
        type=parse_float_list,
        default=parse_float_list("10,20,40,80"),
    )
    parser.add_argument(
        "--sustain-frames",
        type=int,
        default=5,
        help=(
            "A threshold crossing must persist for this many "
            "consecutive frames."
        ),
    )
    args = parser.parse_args()

    if args.alignment_prefix_frames < 3:
        raise ValueError(
            "--alignment-prefix-frames must be at least 3."
        )
    if args.sustain_frames < 1:
        raise ValueError("--sustain-frames must be positive.")

    metadata_dir = (
        args.output_root / "metadata" / "s6a_relative_motion"
    )
    reports_dir = (
        args.output_root / "reports" / "s6a_relative_motion"
    )
    figures_dir = (
        args.output_root / "figures" / "s6a_relative_motion"
    )
    metadata_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    manifest, pairs, features = load_inputs(
        args.manifest,
        args.pair_csv,
        args.feature_csv,
        args.enriched_csv,
        args.sequence,
    )

    width = int(features.iloc[0]["width"])
    height = int(features.iloc[0]["height"])
    center = np.array(
        [width / 2.0, height / 2.0, 1.0],
        dtype=float,
    )

    variants = {
        "se2_scale_normalized": integrate_local_steps(
            pairs,
            center,
            normalize_scale=True,
        ),
        "sim2_local_step": integrate_local_steps(
            pairs,
            center,
            normalize_scale=False,
        ),
    }

    raw_rows: list[pd.DataFrame] = []
    aligned_rows: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []
    crossing_rows: list[dict[str, Any]] = []

    for variant_name, trajectory in variants.items():
        raw_variant = trajectory.copy()
        raw_variant.insert(0, "variant", variant_name)
        raw_rows.append(raw_variant)

        output, summary, crossings = build_variant_output(
            variant_name,
            trajectory,
            manifest,
            prefix_frames=args.alignment_prefix_frames,
            thresholds=args.error_thresholds_m,
            sustain_frames=args.sustain_frames,
        )
        aligned_rows.append(output)
        summaries.append(summary)
        crossing_rows.extend(crossings)

        save_variant_plots(
            variant_name,
            output,
            figures_dir,
            prefix_frames=args.alignment_prefix_frames,
            thresholds=args.error_thresholds_m,
        )

    raw_df = pd.concat(raw_rows, ignore_index=True)
    aligned_df = pd.concat(aligned_rows, ignore_index=True)
    summary_df = pd.json_normalize(summaries, sep=".")
    crossing_df = pd.DataFrame(crossing_rows)

    unsafe_pairs = int(
        (~pairs["pair_safe_image_only"]).sum()
    )

    raw_path = (
        metadata_dir
        / "s6a2_orb_relative_trajectory_pixels.csv"
    )
    aligned_path = (
        metadata_dir
        / "s6a2_orb_relative_trajectory_aligned_eval_only.csv"
    )
    summary_path = (
        metadata_dir
        / "s6a2_orb_relative_trajectory_summary.csv"
    )
    crossing_path = (
        metadata_dir
        / "s6a2_drift_threshold_crossings.csv"
    )
    report_path = (
        reports_dir
        / "s6a2_orb_relative_trajectory_and_drift.json"
    )

    raw_df.to_csv(raw_path, index=False)
    aligned_df.to_csv(aligned_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    crossing_df.to_csv(crossing_path, index=False)

    report = {
        "stage": "S6A.2",
        "sequence": args.sequence,
        "estimator_rule": (
            "Only image-derived ORB partial-affine transforms are "
            "used to build the relative trajectory."
        ),
        "evaluation_rule": (
            "Reference ENU positions are used only after trajectory "
            "construction for similarity alignment and error metrics."
        ),
        "important_timing_note": (
            "SatLoc traj01 has no trusted timestamps in the current "
            "index. Safe horizons are therefore reported in frames "
            "and travelled metres, not seconds."
        ),
        "configuration": {
            "alignment_prefix_frames": (
                args.alignment_prefix_frames
            ),
            "error_thresholds_m": args.error_thresholds_m,
            "sustain_frames": args.sustain_frames,
            "image_width": width,
            "image_height": height,
        },
        "stride": 1,
        "pairs": int(len(pairs)),
        "image_only_unsafe_pairs_flagged": unsafe_pairs,
        "variants": summaries,
        "threshold_crossings": crossing_rows,
    }

    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print("\nS6A.2 ORB relative trajectory and drift")
    print("---------------------------------------")
    print(f"Sequence:                    {args.sequence}")
    print(f"Frames:                      {len(manifest)}")
    print(f"Stride-1 pairs:              {len(pairs)}")
    print(f"Image-only unsafe flags:     {unsafe_pairs}")
    print(
        "Alignment prefix:           "
        f"{args.alignment_prefix_frames} frames"
    )

    for summary in summaries:
        global_summary = summary["global_alignment"]
        prefix_summary = summary["prefix_locked_alignment"]

        print(f"\nVariant: {summary['variant']}")
        print(
            "  Global shape RMSE [m]:     "
            f"{global_summary['rmse_m']:.3f}"
        )
        print(
            "  Global shape p95 [m]:      "
            f"{global_summary['p95_error_m']:.3f}"
        )
        print(
            "  Global scale [m/px]:       "
            f"{global_summary['scale_m_per_px']:.6f}"
        )
        print(
            "  Prefix distance [m]:       "
            f"{prefix_summary['prefix_reference_distance_m']:.3f}"
        )
        print(
            "  Prefix-locked RMSE [m]:    "
            f"{prefix_summary['rmse_m']:.3f}"
        )
        print(
            "  Prefix-locked p95 [m]:     "
            f"{prefix_summary['p95_error_m']:.3f}"
        )
        print(
            "  Prefix-locked max [m]:     "
            f"{prefix_summary['max_error_m']:.3f}"
        )
        print(
            "  Final error [m]:           "
            f"{prefix_summary['final_error_m']:.3f}"
        )
        print(
            "  Final drift [m/100m]:      "
            f"{prefix_summary['final_drift_per_100m']:.3f}"
        )

        variant_crossings = [
            row
            for row in crossing_rows
            if row["variant"] == summary["variant"]
        ]
        for crossing in variant_crossings:
            if crossing["crossed"]:
                print(
                    f"  Sustained {crossing['threshold_m']:g} m "
                    "crossing: "
                    f"{crossing['distance_after_alignment_prefix_m']:.1f} m, "
                    f"{crossing['frames_after_alignment_prefix']} frames"
                )
            else:
                print(
                    f"  Sustained {crossing['threshold_m']:g} m "
                    "crossing: not reached"
                )

    print("\nSaved outputs")
    print("-------------")
    print(raw_path)
    print(aligned_path)
    print(summary_path)
    print(crossing_path)
    print(report_path)
    print(figures_dir)


if __name__ == "__main__":
    main()
