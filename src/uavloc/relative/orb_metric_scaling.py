from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from uavloc.relative.orb_relative_motion import (
    compute_path_length,
    fit_similarity_2d,
    resolve_project_paths,
)


ORB_INPUT_STAGE_NAME = "09b_orb_relative_motion_subset"
STAGE_NAME = "09c_orb_metric_scaling_subset"


def first_existing_column(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    lookup = {str(col).lower(): str(col) for col in df.columns}

    for name in candidates:
        if name.lower() in lookup:
            return lookup[name.lower()]

    return None


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def resolve_orb_input_trajectory(
    output_dir: Path,
    run_name: Optional[str],
    input_csv: Optional[str | Path] = None,
    input_stage_name: str = ORB_INPUT_STAGE_NAME,
) -> Path:
    if input_csv is not None:
        path = Path(input_csv)
        if not path.exists():
            raise FileNotFoundError(f"Explicit ORB trajectory CSV not found: {path}")
        return path

    if run_name is None or not str(run_name).strip():
        # Backward-compatible old sample/default path.
        path = output_dir / "trajectories" / "07_orb_relative_motion" / "orb_relative_trajectory.csv"
    else:
        path = (
            output_dir
            / "trajectories"
            / input_stage_name
            / str(run_name)
            / "orb_relative_trajectory.csv"
        )

    if not path.exists():
        raise FileNotFoundError(
            "ORB input trajectory not found.\n"
            f"Expected: {path}\n"
            "Run Block 09B first, or pass --input-csv explicitly."
        )

    return path


def load_stage07_trajectory(output_dir: Path) -> Path:
    path = output_dir / "trajectories" / ORB_INPUT_STAGE_NAME / "orb_relative_trajectory.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Missing Block 07 trajectory: {path}\n"
            "Run first:\n"
            "python scripts/run_orb_relative_motion.py --config configs/dataset_zurich.yaml --stride 1"
        )

    return path


def attach_synchronized_metadata(traj_df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    enriched_csv = output_dir / "metadata" / "synchronized_frames_enriched.csv"
    normal_csv = output_dir / "metadata" / "synchronized_frames.csv"

    sync_csv = enriched_csv if enriched_csv.exists() else normal_csv

    if not sync_csv.exists():
        return traj_df

    sync_df = pd.read_csv(sync_csv)
    sync_df.columns = [str(c).strip() for c in sync_df.columns]

    traj_df = traj_df.copy()
    traj_df.columns = [str(c).strip() for c in traj_df.columns]

    if "imgid" in traj_df.columns and "imgid" in sync_df.columns:
        left = traj_df.copy()
        right = sync_df.copy()

        left["_imgid_merge"] = pd.to_numeric(left["imgid"], errors="coerce")
        right["_imgid_merge"] = pd.to_numeric(right["imgid"], errors="coerce")

        right = right.drop_duplicates("_imgid_merge")

        merged = pd.merge(
            left,
            right,
            on="_imgid_merge",
            how="left",
            suffixes=("", "_sync"),
        )

        merged = merged.drop(columns=["_imgid_merge"])

        return merged

    if len(sync_df) == len(traj_df):
        merged = traj_df.copy()

        for col in sync_df.columns:
            if col not in merged.columns:
                merged[col] = sync_df[col].values
            else:
                merged[f"{col}_sync"] = sync_df[col].values

        return merged

    return traj_df


def find_calibration_file(raw_dir: Path) -> Optional[Path]:
    candidates = [
        raw_dir / "calibration_data.npz",
        raw_dir / "camera_calibration.npz",
        raw_dir / "calibration.npz",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    matches = sorted(raw_dir.glob("*.npz"))

    for match in matches:
        if "calib" in match.name.lower() or "camera" in match.name.lower():
            return match

    return matches[0] if matches else None


def maybe_camera_matrix(array: Any) -> Optional[np.ndarray]:
    arr = np.asarray(array, dtype=float)

    if arr.shape == (3, 3):
        return arr

    if arr.size == 9:
        return arr.reshape(3, 3)

    return None


def load_camera_intrinsics(raw_dir: Path, fx_arg: Optional[float], fy_arg: Optional[float]) -> Dict[str, Any]:
    if fx_arg is not None:
        fx = float(fx_arg)
        fy = float(fy_arg if fy_arg is not None else fx_arg)

        return {
            "available": True,
            "source": "command_line",
            "fx": fx,
            "fy": fy,
            "cx": None,
            "cy": None,
            "used_key": None,
            "npz_keys": [],
        }

    calib_file = find_calibration_file(raw_dir)

    if calib_file is None:
        raise FileNotFoundError(
            "No calibration .npz file found and --fx was not provided. "
            "For debug, pass --fx <focal_px> --fy <focal_px>."
        )

    data = np.load(calib_file, allow_pickle=True)
    keys = list(data.keys())

    preferred_keys = [
        "camera_matrix",
        "K",
        "mtx",
        "intrinsic_matrix",
        "intrinsics",
        "cameraMatrix",
        "cam_matrix",
    ]

    for key in preferred_keys:
        if key in data:
            K = maybe_camera_matrix(data[key])

            if K is not None:
                return {
                    "available": True,
                    "source": str(calib_file),
                    "fx": float(K[0, 0]),
                    "fy": float(K[1, 1]),
                    "cx": float(K[0, 2]),
                    "cy": float(K[1, 2]),
                    "used_key": key,
                    "npz_keys": keys,
                }

    for key in keys:
        K = maybe_camera_matrix(data[key])

        if K is None:
            continue

        if abs(K[2, 2]) > 1e-9 and K[0, 0] > 0 and K[1, 1] > 0:
            return {
                "available": True,
                "source": str(calib_file),
                "fx": float(K[0, 0]),
                "fy": float(K[1, 1]),
                "cx": float(K[0, 2]),
                "cy": float(K[1, 2]),
                "used_key": key,
                "npz_keys": keys,
            }

    raise ValueError(
        f"Calibration file found but no usable 3x3 camera matrix was detected: {calib_file}\n"
        f"Available keys: {keys}\n"
        "You can bypass this for debug using --fx <focal_px> --fy <focal_px>."
    )


def choose_height_column(
    df: pd.DataFrame,
    explicit_height_column: Optional[str],
    allow_absolute_altitude: bool,
) -> Tuple[Optional[str], str]:
    if explicit_height_column is not None:
        if explicit_height_column not in df.columns:
            raise ValueError(
                f"Requested height column '{explicit_height_column}' not found.\n"
                f"Available columns: {list(df.columns)}"
            )

        return explicit_height_column, "explicit"

    safe_height_candidates = [
        "height_agl_m",
        "agl_m",
        "altitude_agl_m",
        "groundtruth_agl_m",
        "z_agl_m",
        "height_m",
        "height",
        "range_m",
        "laser_altitude_m",
        "laser_height_m",
        "baro_height_m",
        "barometer_height_m",
    ]

    col = first_existing_column(df, safe_height_candidates)

    if col is not None:
        return col, "auto_safe_agl_or_height"

    if allow_absolute_altitude:
        risky_candidates = [
            "altitude_m",
            "altitude",
            "alt",
            "baro_altitude_m",
            "barometer_altitude",
        ]

        col = first_existing_column(df, risky_candidates)

        if col is not None:
            return col, "auto_absolute_altitude_warning"

    return None, "fixed_default"


def choose_yaw_column(df: pd.DataFrame, explicit_yaw_column: Optional[str]) -> Tuple[Optional[str], str]:
    if explicit_yaw_column is not None:
        if explicit_yaw_column not in df.columns:
            raise ValueError(
                f"Requested yaw column '{explicit_yaw_column}' not found.\n"
                f"Available columns: {list(df.columns)}"
            )

        return explicit_yaw_column, "explicit"

    yaw_candidates = [
        "yaw_deg",
        "heading_deg",
        "azimuth_deg",
        "azimuth",
        "omega_yaw_deg",
        "yaw",
        "yaw_rad",
        "heading_rad",
    ]

    col = first_existing_column(df, yaw_candidates)

    if col is not None:
        return col, "auto"

    return None, "fixed_heading"


def column_to_yaw_rad(series: pd.Series, col_name: str) -> np.ndarray:
    values = safe_numeric(series).to_numpy(dtype=float)

    lower = col_name.lower()

    if "rad" in lower:
        return values

    return np.deg2rad(values)


def build_height_m(
    df: pd.DataFrame,
    height_col: Optional[str],
    default_height_m: float,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    if height_col is None:
        values = np.full(len(df), float(default_height_m), dtype=float)

        return values, {
            "source": "fixed_default",
            "column": None,
            "default_height_m": float(default_height_m),
            "valid_count": int(len(values)),
            "median_height_m": float(np.nanmedian(values)),
        }

    raw = safe_numeric(df[height_col]).to_numpy(dtype=float)

    # Do not allow zero/negative height for scale.
    valid = np.isfinite(raw) & (raw > 0)

    values = raw.copy()
    values[~valid] = float(default_height_m)

    return values, {
        "source": "column_with_default_fill",
        "column": height_col,
        "default_height_m": float(default_height_m),
        "valid_count": int(valid.sum()),
        "invalid_filled_count": int((~valid).sum()),
        "median_height_m": float(np.nanmedian(values)),
        "min_height_m": float(np.nanmin(values)),
        "max_height_m": float(np.nanmax(values)),
    }


def build_yaw_rad(
    df: pd.DataFrame,
    yaw_col: Optional[str],
    fixed_heading_deg: float,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    if yaw_col is None:
        values = np.full(len(df), math.radians(float(fixed_heading_deg)), dtype=float)

        return values, {
            "source": "fixed_heading",
            "column": None,
            "fixed_heading_deg": float(fixed_heading_deg),
            "valid_count": int(len(values)),
        }

    raw = column_to_yaw_rad(df[yaw_col], yaw_col)
    valid = np.isfinite(raw)

    values = raw.copy()
    values[~valid] = math.radians(float(fixed_heading_deg))

    return values, {
        "source": "column_with_fixed_fill",
        "column": yaw_col,
        "fixed_heading_deg": float(fixed_heading_deg),
        "valid_count": int(valid.sum()),
        "invalid_filled_count": int((~valid).sum()),
        "median_yaw_deg": float(np.rad2deg(np.nanmedian(values))),
    }


def compute_confidence(df: pd.DataFrame) -> np.ndarray:
    status_ok = (df.get("tracking_status", "") == "ok").astype(float)

    inlier_ratio = safe_numeric(df.get("inlier_ratio", pd.Series(np.zeros(len(df))))).fillna(0.0)
    inliers = safe_numeric(df.get("ransac_inliers", pd.Series(np.zeros(len(df))))).fillna(0.0)

    inlier_score = np.clip(inlier_ratio.to_numpy(dtype=float), 0.0, 1.0)
    count_score = np.clip(inliers.to_numpy(dtype=float) / 300.0, 0.0, 1.0)

    confidence = status_ok.to_numpy(dtype=float) * inlier_score * count_score

    if len(confidence) > 0:
        confidence[0] = 1.0

    return confidence


def compute_reference_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    ref_x_col = first_existing_column(df, ["ref_x_enu_m", "x_enu_m", "x_enu_m_sync"])
    ref_y_col = first_existing_column(df, ["ref_y_enu_m", "y_enu_m", "y_enu_m_sync"])

    if ref_x_col is None or ref_y_col is None:
        return {"available": False, "reason": "reference x/y columns not found"}

    ref_x = safe_numeric(df[ref_x_col]).to_numpy(dtype=float)
    ref_y = safe_numeric(df[ref_y_col]).to_numpy(dtype=float)

    ref_xy = np.column_stack([ref_x, ref_y])
    ref_rel = ref_xy - ref_xy[0]

    est_xy = df[["estimated_x_enu_m", "estimated_y_enu_m"]].to_numpy(dtype=float)

    error = np.linalg.norm(est_xy - ref_rel, axis=1)

    df["ref_x_rel_m"] = ref_rel[:, 0]
    df["ref_y_rel_m"] = ref_rel[:, 1]
    df["position_error_m"] = error

    ref_path_length_m = compute_path_length(ref_rel)
    est_path_length_m = compute_path_length(est_xy)

    final_error_m = float(error[-1])

    metrics = {
        "available": True,
        "reference_x_column": ref_x_col,
        "reference_y_column": ref_y_col,
        "rmse_m": float(np.sqrt(np.nanmean(error**2))),
        "mean_error_m": float(np.nanmean(error)),
        "median_error_m": float(np.nanmedian(error)),
        "max_error_m": float(np.nanmax(error)),
        "final_error_m": final_error_m,
        "reference_path_length_m": float(ref_path_length_m),
        "estimated_path_length_m": float(est_path_length_m),
        "drift_per_100m": float((final_error_m / ref_path_length_m) * 100.0) if ref_path_length_m > 1e-9 else None,
    }

    try:
        fit = fit_similarity_2d(est_xy, ref_rel)
        aligned = fit["aligned"]
        aligned_error = np.linalg.norm(aligned - ref_rel, axis=1)

        df["aligned_x_m"] = aligned[:, 0]
        df["aligned_y_m"] = aligned[:, 1]
        df["aligned_error_m"] = aligned_error

        metrics["shape_alignment"] = {
            "available": True,
            "note": "Shape alignment uses reference only for evaluation, not for camera-only localization.",
            "scale": fit["scale"],
            "rotation_deg": float(math.degrees(fit["rotation_rad"])),
            "rmse_m": float(np.sqrt(np.nanmean(aligned_error**2))),
            "mean_error_m": float(np.nanmean(aligned_error)),
            "max_error_m": float(np.nanmax(aligned_error)),
        }

    except ValueError as exc:
        metrics["shape_alignment"] = {"available": False, "reason": str(exc)}

    return metrics


def plot_metric_xy(df: pd.DataFrame, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 7))

    ax.plot(df["estimated_x_enu_m"], df["estimated_y_enu_m"], linewidth=1.6, label="ORB metric estimate")
    ax.scatter(df["estimated_x_enu_m"].iloc[0], df["estimated_y_enu_m"].iloc[0], marker="o", label="start")
    ax.scatter(df["estimated_x_enu_m"].iloc[-1], df["estimated_y_enu_m"].iloc[-1], marker="x", label="end")

    ax.set_title("ORB Metric Trajectory — Approximate ENU")
    ax.set_xlabel("estimated x ENU [m]")
    ax.set_ylabel("estimated y ENU [m]")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_reference_comparison(df: pd.DataFrame, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    if "ref_x_rel_m" not in df.columns or "ref_y_rel_m" not in df.columns:
        return

    fig, ax = plt.subplots(figsize=(8, 7))

    ax.plot(df["ref_x_rel_m"], df["ref_y_rel_m"], linewidth=1.8, label="Reference GNSS/ENU")
    ax.plot(df["estimated_x_enu_m"], df["estimated_y_enu_m"], linewidth=1.4, label="ORB metric estimate")

    if "aligned_x_m" in df.columns and "aligned_y_m" in df.columns:
        ax.plot(df["aligned_x_m"], df["aligned_y_m"], linewidth=1.2, linestyle="--", label="ORB shape-aligned check")

    ax.scatter(df["ref_x_rel_m"].iloc[0], df["ref_y_rel_m"].iloc[0], marker="o", label="start")
    ax.scatter(df["ref_x_rel_m"].iloc[-1], df["ref_y_rel_m"].iloc[-1], marker="x", label="reference end")

    ax.set_title("ORB Metric Trajectory vs Reference")
    ax.set_xlabel("x ENU relative [m]")
    ax.set_ylabel("y ENU relative [m]")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_error_over_frame(df: pd.DataFrame, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    if "position_error_m" not in df.columns:
        return

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(df["frame_index"], df["position_error_m"], linewidth=1.5, label="metric position error")

    if "aligned_error_m" in df.columns:
        ax.plot(df["frame_index"], df["aligned_error_m"], linewidth=1.2, linestyle="--", label="shape-aligned error")

    ax.set_title("ORB Metric Trajectory Error over Frames")
    ax.set_xlabel("frame index")
    ax.set_ylabel("position error [m]")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def run_orb_metric_scaling(
    config_path: str | Path,
    run_name: Optional[str] = None,
    input_csv: Optional[str | Path] = None,
    input_stage_name: str = ORB_INPUT_STAGE_NAME,
    default_height_m: float = 50.0,
    fixed_heading_deg: float = 0.0,
    height_column: Optional[str] = None,
    yaw_column: Optional[str] = None,
    allow_absolute_altitude: bool = False,
    fx: Optional[float] = None,
    fy: Optional[float] = None,
    image_x_to_right_sign: float = 1.0,
    image_y_to_forward_sign: float = 1.0,
    scale_multiplier: float = 1.0,
    yaw_offset_deg: float = 0.0,
) -> Dict[str, Any]:
    _, _, raw_dir, output_dir, dataset_name = resolve_project_paths(config_path)

    orb_csv = resolve_orb_input_trajectory(
        output_dir=output_dir,
        run_name=run_name,
        input_csv=input_csv,
        input_stage_name=input_stage_name,
    )

    traj_df = pd.read_csv(orb_csv)
    traj_df.columns = [str(c).strip() for c in traj_df.columns]

    df = attach_synchronized_metadata(traj_df, output_dir)

    camera = load_camera_intrinsics(raw_dir, fx_arg=fx, fy_arg=fy)
    fx_px = float(camera["fx"])
    fy_px = float(camera["fy"])

    chosen_height_col, height_mode = choose_height_column(
        df,
        explicit_height_column=height_column,
        allow_absolute_altitude=allow_absolute_altitude,
    )
    height_m, height_summary = build_height_m(df, chosen_height_col, default_height_m)

    chosen_yaw_col, yaw_mode = choose_yaw_column(df, explicit_yaw_column=yaw_column)
    yaw_rad, yaw_summary = build_yaw_rad(df, chosen_yaw_col, fixed_heading_deg)

    if abs(float(yaw_offset_deg)) > 1e-12:
        yaw_rad = yaw_rad + math.radians(float(yaw_offset_deg))

    yaw_summary["yaw_offset_deg"] = float(yaw_offset_deg)

    dx_px = safe_numeric(df["delta_x_img_px"]).fillna(0.0).to_numpy(dtype=float)
    dy_px = safe_numeric(df["delta_y_img_px"]).fillna(0.0).to_numpy(dtype=float)

    meters_per_px_x = (height_m / fx_px) * float(scale_multiplier)
    meters_per_px_y = (height_m / fy_px) * float(scale_multiplier)

    camera_right_m = float(image_x_to_right_sign) * dx_px * meters_per_px_x
    camera_forward_m = float(image_y_to_forward_sign) * dy_px * meters_per_px_y

    # Heading convention:
    # yaw/heading = 0 deg points North, +90 deg points East.
    delta_east_m = camera_forward_m * np.sin(yaw_rad) + camera_right_m * np.cos(yaw_rad)
    delta_north_m = camera_forward_m * np.cos(yaw_rad) - camera_right_m * np.sin(yaw_rad)

    if len(delta_east_m) > 0:
        delta_east_m[0] = 0.0
        delta_north_m[0] = 0.0

    df["height_used_m"] = height_m
    df["yaw_used_rad"] = yaw_rad
    df["yaw_used_deg"] = np.rad2deg(yaw_rad)
    df["meters_per_px_x"] = meters_per_px_x
    df["meters_per_px_y"] = meters_per_px_y
    df["delta_east_m"] = delta_east_m
    df["delta_north_m"] = delta_north_m
    df["estimated_x_enu_m"] = np.cumsum(delta_east_m)
    df["estimated_y_enu_m"] = np.cumsum(delta_north_m)
    df["confidence"] = compute_confidence(df)
    df["method_name"] = "orb_metric_scaling_pinhole_altitude_heading"

    reference_metrics = compute_reference_metrics(df)

    if run_name is not None and str(run_name).strip():
        safe_run_name = str(run_name).strip()
        traj_dir = output_dir / "trajectories" / STAGE_NAME / safe_run_name
        report_dir = output_dir / "reports" / STAGE_NAME / safe_run_name
        figure_dir = output_dir / "figures" / STAGE_NAME / safe_run_name
    else:
        safe_run_name = "default"
        traj_dir = output_dir / "trajectories" / STAGE_NAME
        report_dir = output_dir / "reports" / STAGE_NAME
        figure_dir = output_dir / "figures" / STAGE_NAME

    for d in [traj_dir, report_dir, figure_dir]:
        d.mkdir(parents=True, exist_ok=True)

    trajectory_csv = traj_dir / "orb_metric_trajectory.csv"
    summary_json = report_dir / "orb_metric_scaling_summary.json"
    metric_plot = figure_dir / "orb_metric_xy.png"
    comparison_plot = figure_dir / "orb_metric_reference_comparison_xy.png"
    error_plot = figure_dir / "orb_metric_error_over_frame.png"

    df.to_csv(trajectory_csv, index=False)

    plot_metric_xy(df, metric_plot)
    plot_reference_comparison(df, comparison_plot)
    plot_error_over_frame(df, error_plot)

    estimated_xy = df[["estimated_x_enu_m", "estimated_y_enu_m"]].to_numpy(dtype=float)

    ref_eval = reference_metrics if reference_metrics.get("available", False) else {}

    summary = {
        "dataset_name": dataset_name,
        "stage": STAGE_NAME,
        "run_name": safe_run_name,
        "input_orb_trajectory_csv": str(orb_csv),
        "input_stage_name": input_stage_name,
        "output_stage_name": STAGE_NAME,
        "frames": int(len(df)),
        "camera_intrinsics": camera,
        "height_column": chosen_height_col,
        "yaw_column": chosen_yaw_col,
        "height": {
            "selection_mode": height_mode,
            **height_summary,
        },
        "yaw_heading": {
            "selection_mode": yaw_mode,
            **yaw_summary,
        },
        "axis_mapping": {
            "image_x_to_right_sign": float(image_x_to_right_sign),
            "image_y_to_forward_sign": float(image_y_to_forward_sign),
            "heading_convention": "0 deg = North, +90 deg = East",
        },
        "scale_multiplier": float(scale_multiplier),
        "yaw_offset_deg": float(yaw_offset_deg),
        "estimated_path_length_m": compute_path_length(estimated_xy),
        "estimated_x_range_m": [
            float(np.nanmin(df["estimated_x_enu_m"])),
            float(np.nanmax(df["estimated_x_enu_m"])),
        ],
        "estimated_y_range_m": [
            float(np.nanmin(df["estimated_y_enu_m"])),
            float(np.nanmax(df["estimated_y_enu_m"])),
        ],
        "confidence": {
            "mean": float(np.nanmean(df["confidence"])),
            "median": float(np.nanmedian(df["confidence"])),
            "min": float(np.nanmin(df["confidence"])),
        },
        "reference_evaluation": reference_metrics,

        # Convenience top-level fields for terminal printing.
        "reference_path_length_m": ref_eval.get("reference_path_length_m"),
        "rmse_m": ref_eval.get("rmse_m"),
        "mean_error_m": ref_eval.get("mean_error_m"),
        "median_error_m": ref_eval.get("median_error_m"),
        "max_error_m": ref_eval.get("max_error_m"),
        "final_error_m": ref_eval.get("final_error_m"),
        "drift_per_100m": ref_eval.get("drift_per_100m"),

        "outputs": {
            "trajectory_csv": str(trajectory_csv),
            "summary_json": str(summary_json),
            "metric_plot": str(metric_plot),
            "comparison_plot": str(comparison_plot),
            "error_plot": str(error_plot),
        },
        "important_warning": (
            "This is an approximate metric conversion. It assumes a mostly nadir camera, "
            "small frame-to-frame motion, planar ground, usable height above ground, and correct camera-to-body axis mapping. "
            "GNSS/reference is used only for evaluation."
        ),
    }

    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary