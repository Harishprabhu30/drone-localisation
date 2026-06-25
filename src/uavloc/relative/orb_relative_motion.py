from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml


STAGE_NAME = "07_orb_relative_motion"
IMAGE_EXTENSIONS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")


@dataclass
class OrbRelativeMotionConfig:
    stride: int = 1
    nfeatures: int = 2000
    ratio_test: float = 0.75
    max_hamming_distance: int = 96
    ransac_reproj_threshold: float = 3.0
    min_good_matches: int = 30
    min_ransac_inliers: int = 20
    max_reasonable_step_px: float = 250.0


def load_yaml_config(config_path: str | Path) -> Dict[str, Any]:
    config_path = Path(config_path)
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a YAML dictionary: {config_path}")

    return data


def _iter_nested_items(obj: Any) -> Iterable[Tuple[str, Any]]:
    if isinstance(obj, dict):
        for key, value in obj.items():
            yield str(key), value
            yield from _iter_nested_items(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from _iter_nested_items(value)


def _find_first_config_value(config: Dict[str, Any], names: Iterable[str]) -> Optional[Any]:
    wanted = {name.lower() for name in names}

    for key, value in _iter_nested_items(config):
        if key.lower() in wanted:
            return value

    return None


def _find_all_config_values(config: Dict[str, Any], names: Iterable[str]) -> List[Any]:
    wanted = {name.lower() for name in names}
    values: List[Any] = []

    for key, value in _iter_nested_items(config):
        if key.lower() in wanted:
            values.append(value)

    return values


def _repo_root_from_config(config_path: str | Path) -> Path:
    config_path = Path(config_path).resolve()

    if config_path.parent.name == "configs":
        return config_path.parent.parent

    return Path.cwd().resolve()


def _resolve_path(path_value: Any, repo_root: Path) -> Optional[Path]:
    if path_value is None:
        return None

    if isinstance(path_value, (list, tuple)):
        return None

    path_str = str(path_value).strip()
    if not path_str:
        return None

    p = Path(path_str)

    if p.is_absolute():
        return p

    return repo_root / p


def resolve_project_paths(config_path: str | Path) -> Tuple[Dict[str, Any], Path, Path, Path, str]:
    config = load_yaml_config(config_path)
    repo_root = _repo_root_from_config(config_path)

    dataset_name = _find_first_config_value(config, ["name", "dataset_name"])
    dataset_name = str(dataset_name) if dataset_name else "zurich_mav_sample"

    raw_value = _find_first_config_value(
        config,
        ["raw_data_dir", "raw_dir", "data_dir", "dataset_dir", "root_dir", "raw_root"],
    )
    output_value = _find_first_config_value(
        config,
        ["output_dir", "outputs_dir", "output_root", "results_dir"],
    )

    raw_dir = _resolve_path(raw_value, repo_root) if raw_value else repo_root / "data" / "raw" / dataset_name
    output_dir = _resolve_path(output_value, repo_root) if output_value else repo_root / "outputs" / dataset_name

    return config, repo_root, raw_dir, output_dir, dataset_name


def natural_sort_key(path: Path) -> List[Any]:
    parts = re.split(r"(\d+)", path.name)
    key: List[Any] = []

    for part in parts:
        if part.isdigit():
            key.append(int(part))
        else:
            key.append(part.lower())

    return key


def _extract_last_int(text: str) -> Optional[int]:
    matches = re.findall(r"\d+", text)

    if not matches:
        return None

    return int(matches[-1])


def find_image_directories(config: Dict[str, Any], raw_dir: Path, repo_root: Path) -> List[Path]:
    candidates: List[Path] = []

    image_values = _find_all_config_values(
        config,
        [
            "image_dir",
            "images_dir",
            "image_folder",
            "images_folder",
            "frames_dir",
            "frame_dir",
            "frames_folder",
            "mav_images_dir",
            "mav_image_dir",
            "mav_images",
            "mav_image_folder",
        ],
    )

    for value in image_values:
        p = _resolve_path(value, repo_root)

        if p is not None:
            candidates.append(p)

            if not p.is_absolute():
                candidates.append(raw_dir / str(value))

    candidates.extend(
        [
            raw_dir / "MAV Images",
            raw_dir / "MAV_Images",
            raw_dir / "MAVImages",
            raw_dir / "mav_images",
            raw_dir / "images",
            raw_dir / "frames",
        ]
    )

    unique: List[Path] = []
    seen = set()

    for p in candidates:
        p_key = p.resolve() if p.exists() else p

        if p_key in seen:
            continue

        seen.add(p_key)

        if p.exists() and p.is_dir():
            unique.append(p)

    return unique


def list_images(image_dirs: List[Path]) -> List[Path]:
    images: List[Path] = []

    for image_dir in image_dirs:
        for pattern in IMAGE_EXTENSIONS:
            images.extend(image_dir.glob(pattern))

    images = sorted(set(images), key=natural_sort_key)
    return images


def _first_existing_column(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    lookup = {col.lower(): col for col in df.columns}

    for name in candidates:
        if name.lower() in lookup:
            return lookup[name.lower()]

    return None


def resolve_image_paths(
    synced_df: pd.DataFrame,
    config: Dict[str, Any],
    raw_dir: Path,
    repo_root: Path,
) -> List[Path]:
    image_path_col = _first_existing_column(
        synced_df,
        [
            "image_path",
            "frame_path",
            "filepath",
            "file_path",
            "path",
            "image_file",
            "filename",
            "file_name",
            "image_name",
            "mav_image_path",
            "mav_image",
        ],
    )

    if image_path_col is not None:
        resolved: List[Path] = []

        for value in synced_df[image_path_col].tolist():
            if pd.isna(value):
                raise ValueError(f"Missing image path value in column '{image_path_col}'")

            value_str = str(value).strip()
            p = Path(value_str)

            candidates = [p] if p.is_absolute() else [repo_root / p, raw_dir / p, raw_dir / p.name]
            existing = next((candidate for candidate in candidates if candidate.exists()), None)

            if existing is None:
                raise FileNotFoundError(
                    f"Could not resolve image path '{value_str}'. Tried: "
                    + ", ".join(str(c) for c in candidates)
                )

            resolved.append(existing)

        return resolved

    image_dirs = find_image_directories(config, raw_dir, repo_root)
    images = list_images(image_dirs)

    if not images:
        raise FileNotFoundError(
            "Could not find MAV images. Checked common folders like 'MAV Images' and 'MAV_Images'. "
            "Add an image path column to synchronized_frames.csv or add image dir in dataset_zurich.yaml."
        )

    imgid_col = _first_existing_column(synced_df, ["imgid", "image_id", "frame_id", "id"])

    if imgid_col is not None:
        image_by_id: Dict[int, Path] = {}

        for image_path in images:
            image_id = _extract_last_int(image_path.stem)

            if image_id is not None and image_id not in image_by_id:
                image_by_id[image_id] = image_path

        resolved = []
        all_found = True

        for value in synced_df[imgid_col].tolist():
            try:
                image_id = int(float(value))
            except (TypeError, ValueError):
                all_found = False
                break

            p = image_by_id.get(image_id)

            if p is None:
                all_found = False
                break

            resolved.append(p)

        if all_found and len(resolved) == len(synced_df):
            return resolved

    if len(images) < len(synced_df):
        raise ValueError(
            f"Found {len(images)} images but synchronized CSV has {len(synced_df)} rows. "
            "Could not safely assign images by order."
        )

    return images[: len(synced_df)]


def find_reference_xy_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    candidate_pairs = [
        ("x_enu_m", "y_enu_m"),
        ("ref_x_enu_m", "ref_y_enu_m"),
        ("reference_x_enu_m", "reference_y_enu_m"),
        ("x_ref_enu_m", "y_ref_enu_m"),
        ("ref_x_m", "ref_y_m"),
        ("x_m", "y_m"),
        ("x", "y"),
    ]

    lookup = {col.lower(): col for col in df.columns}

    for x_name, y_name in candidate_pairs:
        if x_name.lower() in lookup and y_name.lower() in lookup:
            return lookup[x_name.lower()], lookup[y_name.lower()]

    return None, None


def find_timestamp_column(df: pd.DataFrame) -> Optional[str]:
    return _first_existing_column(
        df,
        [
            "timestamp",
            "frame_timestamp",
            "image_timestamp",
            "timestamp_s",
            "time",
            "t",
        ],
    )


def create_orb_detector(nfeatures: int):
    import cv2

    return cv2.ORB_create(
        nfeatures=nfeatures,
        scaleFactor=1.2,
        nlevels=8,
        edgeThreshold=31,
        firstLevel=0,
        WTA_K=2,
        scoreType=cv2.ORB_HARRIS_SCORE,
        patchSize=31,
        fastThreshold=20,
    )


def read_gray_image(path: Path) -> np.ndarray:
    import cv2

    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError(f"Failed to read image: {path}")

    return image


def match_orb_pair(
    image_prev: np.ndarray,
    image_curr: np.ndarray,
    orb,
    cfg: OrbRelativeMotionConfig,
) -> Dict[str, Any]:
    import cv2

    kp0, des0 = orb.detectAndCompute(image_prev, None)
    kp1, des1 = orb.detectAndCompute(image_curr, None)

    result: Dict[str, Any] = {
        "keypoints_prev": len(kp0) if kp0 is not None else 0,
        "keypoints_curr": len(kp1) if kp1 is not None else 0,
        "good_matches": 0,
        "ransac_inliers": 0,
        "inlier_ratio": 0.0,
        "homography_success": False,
        "feature_dx_px": 0.0,
        "feature_dy_px": 0.0,
        "delta_x_img_px": 0.0,
        "delta_y_img_px": 0.0,
        "status": "failed_no_descriptors",
    }

    if des0 is None or des1 is None or len(kp0) < 4 or len(kp1) < 4:
        return result

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn_matches = bf.knnMatch(des0, des1, k=2)

    good = []

    for pair in knn_matches:
        if len(pair) < 2:
            continue

        m, n = pair

        if m.distance < cfg.ratio_test * n.distance and m.distance <= cfg.max_hamming_distance:
            good.append(m)

    if len(good) < cfg.min_good_matches:
        result["good_matches"] = len(good)
        result["status"] = "failed_few_matches"
        return result

    pts0 = np.float32([kp0[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    pts1 = np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(pts0, pts1, cv2.RANSAC, cfg.ransac_reproj_threshold)

    if H is None or mask is None:
        result["good_matches"] = len(good)
        result["status"] = "failed_homography"
        return result

    mask_flat = mask.ravel().astype(bool)
    inlier_count = int(mask_flat.sum())
    inlier_ratio = float(inlier_count / max(len(good), 1))

    result["good_matches"] = len(good)
    result["ransac_inliers"] = inlier_count
    result["inlier_ratio"] = inlier_ratio
    result["homography_success"] = True

    if inlier_count < cfg.min_ransac_inliers:
        result["status"] = "failed_few_inliers"
        return result

    pts0_in = pts0.reshape(-1, 2)[mask_flat]
    pts1_in = pts1.reshape(-1, 2)[mask_flat]
    displacement = pts1_in - pts0_in

    feature_dx_px = float(np.median(displacement[:, 0]))
    feature_dy_px = float(np.median(displacement[:, 1]))

    # Feature motion is image-content motion.
    # For a downward-looking camera, camera/drone translation is approximately
    # the opposite direction in the image plane.
    delta_x_img_px = -feature_dx_px
    delta_y_img_px = -feature_dy_px

    step_norm = math.hypot(delta_x_img_px, delta_y_img_px)

    if step_norm > cfg.max_reasonable_step_px:
        result.update(
            {
                "feature_dx_px": feature_dx_px,
                "feature_dy_px": feature_dy_px,
                "delta_x_img_px": 0.0,
                "delta_y_img_px": 0.0,
                "status": "failed_unreasonable_step",
            }
        )
        return result

    result.update(
        {
            "feature_dx_px": feature_dx_px,
            "feature_dy_px": feature_dy_px,
            "delta_x_img_px": delta_x_img_px,
            "delta_y_img_px": delta_y_img_px,
            "status": "ok",
        }
    )

    return result


def fit_similarity_2d(source_xy: np.ndarray, target_xy: np.ndarray) -> Dict[str, Any]:
    source_xy = np.asarray(source_xy, dtype=float)
    target_xy = np.asarray(target_xy, dtype=float)

    valid = np.isfinite(source_xy).all(axis=1) & np.isfinite(target_xy).all(axis=1)
    src = source_xy[valid]
    dst = target_xy[valid]

    if len(src) < 3:
        raise ValueError("Need at least 3 valid points for trajectory alignment.")

    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)

    src_c = src - src_mean
    dst_c = dst - dst_mean

    src_energy = float(np.sum(src_c**2))

    if src_energy < 1e-9:
        raise ValueError("Estimated trajectory has almost zero movement; cannot align to reference.")

    covariance = src_c.T @ dst_c
    U, singular_values, Vt = np.linalg.svd(covariance)

    rotation = Vt.T @ U.T

    if np.linalg.det(rotation) < 0:
        Vt[-1, :] *= -1
        rotation = Vt.T @ U.T

    scale = float(np.sum(singular_values) / src_energy)
    translation = dst_mean - scale * (src_mean @ rotation.T)
    aligned = scale * (source_xy @ rotation.T) + translation

    angle_rad = float(math.atan2(rotation[1, 0], rotation[0, 0]))

    return {
        "aligned": aligned,
        "scale": scale,
        "rotation_rad": angle_rad,
        "translation_x": float(translation[0]),
        "translation_y": float(translation[1]),
        "valid_points": int(valid.sum()),
    }


def compute_path_length(xy: np.ndarray) -> float:
    xy = np.asarray(xy, dtype=float)

    if len(xy) < 2:
        return 0.0

    diffs = np.diff(xy, axis=0)
    return float(np.nansum(np.linalg.norm(diffs, axis=1)))


def plot_raw_trajectory(trajectory_df: pd.DataFrame, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 7))

    ax.plot(
        trajectory_df["raw_x_px"],
        trajectory_df["raw_y_px"],
        linewidth=1.5,
        label="ORB image-motion trajectory",
    )
    ax.scatter(
        trajectory_df["raw_x_px"].iloc[0],
        trajectory_df["raw_y_px"].iloc[0],
        marker="o",
        label="start",
    )
    ax.scatter(
        trajectory_df["raw_x_px"].iloc[-1],
        trajectory_df["raw_y_px"].iloc[-1],
        marker="x",
        label="end",
    )

    ax.set_title("ORB Relative Trajectory — Raw Image-Motion Units")
    ax.set_xlabel("Accumulated image motion x [px]")
    ax.set_ylabel("Accumulated image motion y [px]")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_reference_comparison(trajectory_df: pd.DataFrame, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 7))

    ax.plot(
        trajectory_df["ref_x_rel_m"],
        trajectory_df["ref_y_rel_m"],
        linewidth=1.8,
        label="Reference GNSS/ENU",
    )
    ax.plot(
        trajectory_df["aligned_x_m"],
        trajectory_df["aligned_y_m"],
        linewidth=1.5,
        label="ORB trajectory aligned for shape check",
    )

    ax.scatter(
        trajectory_df["ref_x_rel_m"].iloc[0],
        trajectory_df["ref_y_rel_m"].iloc[0],
        marker="o",
        label="start",
    )
    ax.scatter(
        trajectory_df["ref_x_rel_m"].iloc[-1],
        trajectory_df["ref_y_rel_m"].iloc[-1],
        marker="x",
        label="reference end",
    )

    ax.set_title("ORB Relative Trajectory vs Reference — Shape Comparison")
    ax.set_xlabel("x ENU relative [m]")
    ax.set_ylabel("y ENU relative [m]")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def run_orb_relative_motion(
    config_path: str | Path,
    stride: int = 1,
    max_frames: Optional[int] = None,
    orb_cfg: Optional[OrbRelativeMotionConfig] = None,
) -> Dict[str, Any]:
    start_time = time.perf_counter()

    if orb_cfg is None:
        orb_cfg = OrbRelativeMotionConfig(stride=stride)
    else:
        orb_cfg.stride = stride

    config, repo_root, raw_dir, output_dir, dataset_name = resolve_project_paths(config_path)

    synced_csv = output_dir / "metadata" / "synchronized_frames.csv"

    if not synced_csv.exists():
        raise FileNotFoundError(
            f"Missing synchronized frames CSV: {synced_csv}\n"
            "Run: python scripts/sync_zurich_frames.py --config configs/dataset_zurich.yaml"
        )

    synced_df = pd.read_csv(synced_csv)
    synced_df.columns = [str(col).strip() for col in synced_df.columns]

    if max_frames is not None:
        synced_df = synced_df.head(max_frames).copy()

    if len(synced_df) < 2:
        raise ValueError("Need at least 2 synchronized frames for relative motion estimation.")

    image_paths = resolve_image_paths(synced_df, config, raw_dir, repo_root)

    x_col, y_col = find_reference_xy_columns(synced_df)
    timestamp_col = find_timestamp_column(synced_df)
    imgid_col = _first_existing_column(synced_df, ["imgid", "image_id", "frame_id", "id"])

    traj_dir = output_dir / "trajectories" / STAGE_NAME
    report_dir = output_dir / "reports" / STAGE_NAME
    figure_dir = output_dir / "figures" / STAGE_NAME

    for d in [traj_dir, report_dir, figure_dir]:
        d.mkdir(parents=True, exist_ok=True)

    orb = create_orb_detector(orb_cfg.nfeatures)

    raw_x = 0.0
    raw_y = 0.0
    rows: List[Dict[str, Any]] = []

    def make_base_row(frame_index: int, image_path: Path) -> Dict[str, Any]:
        source_row = synced_df.iloc[frame_index]

        return {
            "frame_index": frame_index,
            "imgid": source_row[imgid_col] if imgid_col else frame_index,
            "timestamp": source_row[timestamp_col] if timestamp_col else np.nan,
            "image_path": str(image_path),
            "ref_x_enu_m": float(source_row[x_col]) if x_col else np.nan,
            "ref_y_enu_m": float(source_row[y_col]) if y_col else np.nan,
            "raw_x_px": np.nan,
            "raw_y_px": np.nan,
            "aligned_x_m": np.nan,
            "aligned_y_m": np.nan,
            "ref_x_rel_m": np.nan,
            "ref_y_rel_m": np.nan,
            "delta_x_img_px": 0.0,
            "delta_y_img_px": 0.0,
            "feature_dx_px": 0.0,
            "feature_dy_px": 0.0,
            "keypoints_prev": 0,
            "keypoints_curr": 0,
            "good_matches": 0,
            "ransac_inliers": 0,
            "inlier_ratio": 0.0,
            "homography_success": False,
            "tracking_status": "start",
            "method_name": f"orb_homography_image_motion_stride_{stride}",
        }

    first_row = make_base_row(0, image_paths[0])
    first_row["raw_x_px"] = raw_x
    first_row["raw_y_px"] = raw_y
    rows.append(first_row)

    previous_image = read_gray_image(image_paths[0])
    previous_index = 0

    for curr_index in range(stride, len(synced_df), stride):
        current_image = read_gray_image(image_paths[curr_index])
        pair_result = match_orb_pair(previous_image, current_image, orb, orb_cfg)

        if pair_result["status"] == "ok":
            raw_x += float(pair_result["delta_x_img_px"])
            raw_y += float(pair_result["delta_y_img_px"])

        row = make_base_row(curr_index, image_paths[curr_index])
        row.update(
            {
                "pair_from_frame_index": previous_index,
                "pair_to_frame_index": curr_index,
                "raw_x_px": raw_x,
                "raw_y_px": raw_y,
                "delta_x_img_px": float(pair_result["delta_x_img_px"]),
                "delta_y_img_px": float(pair_result["delta_y_img_px"]),
                "feature_dx_px": float(pair_result["feature_dx_px"]),
                "feature_dy_px": float(pair_result["feature_dy_px"]),
                "keypoints_prev": int(pair_result["keypoints_prev"]),
                "keypoints_curr": int(pair_result["keypoints_curr"]),
                "good_matches": int(pair_result["good_matches"]),
                "ransac_inliers": int(pair_result["ransac_inliers"]),
                "inlier_ratio": float(pair_result["inlier_ratio"]),
                "homography_success": bool(pair_result["homography_success"]),
                "tracking_status": str(pair_result["status"]),
                "method_name": f"orb_homography_image_motion_stride_{stride}",
            }
        )
        rows.append(row)

        previous_image = current_image
        previous_index = curr_index

    trajectory_df = pd.DataFrame(rows)

    alignment_summary: Dict[str, Any] = {"available": False}

    if x_col is not None and y_col is not None:
        ref_xy = trajectory_df[["ref_x_enu_m", "ref_y_enu_m"]].to_numpy(dtype=float)
        ref_rel = ref_xy - ref_xy[0]

        trajectory_df["ref_x_rel_m"] = ref_rel[:, 0]
        trajectory_df["ref_y_rel_m"] = ref_rel[:, 1]

        est_xy = trajectory_df[["raw_x_px", "raw_y_px"]].to_numpy(dtype=float)

        try:
            alignment = fit_similarity_2d(est_xy, ref_rel)
            aligned = alignment["aligned"]

            trajectory_df["aligned_x_m"] = aligned[:, 0]
            trajectory_df["aligned_y_m"] = aligned[:, 1]

            errors = np.linalg.norm(aligned - ref_rel, axis=1)

            alignment_summary = {
                "available": True,
                "note": (
                    "Similarity alignment is for shape comparison/evaluation only. "
                    "It is not used as GNSS input to the visual algorithm."
                ),
                "scale_m_per_px": alignment["scale"],
                "rotation_rad": alignment["rotation_rad"],
                "rotation_deg": float(math.degrees(alignment["rotation_rad"])),
                "translation_x_m": alignment["translation_x"],
                "translation_y_m": alignment["translation_y"],
                "valid_points": alignment["valid_points"],
                "aligned_rmse_m": float(np.sqrt(np.mean(errors**2))),
                "aligned_mean_error_m": float(np.mean(errors)),
                "aligned_max_error_m": float(np.max(errors)),
            }

        except ValueError as exc:
            alignment_summary = {"available": False, "reason": str(exc)}

    trajectory_csv = traj_dir / "orb_relative_trajectory.csv"
    summary_json = report_dir / "orb_relative_motion_summary.json"
    raw_plot = figure_dir / "orb_relative_xy.png"
    comparison_plot = figure_dir / "orb_reference_comparison_xy.png"

    trajectory_df.to_csv(trajectory_csv, index=False)
    plot_raw_trajectory(trajectory_df, raw_plot)

    if alignment_summary.get("available", False):
        plot_reference_comparison(trajectory_df, comparison_plot)

    statuses = trajectory_df["tracking_status"].value_counts().to_dict()
    ok_pairs = int((trajectory_df["tracking_status"] == "ok").sum())
    attempted_pairs = max(len(trajectory_df) - 1, 0)
    elapsed_s = time.perf_counter() - start_time

    raw_xy = trajectory_df[["raw_x_px", "raw_y_px"]].to_numpy(dtype=float)

    pair_rows = trajectory_df.loc[trajectory_df["tracking_status"] != "start"]

    summary = {
        "dataset_name": dataset_name,
        "stage": STAGE_NAME,
        "method": f"ORB + BFMatcher + ratio test + RANSAC homography, stride={stride}",
        "input_synchronized_csv": str(synced_csv),
        "frames_used": int(len(trajectory_df)),
        "attempted_pairs": attempted_pairs,
        "ok_pairs": ok_pairs,
        "failed_pairs": int(attempted_pairs - ok_pairs),
        "status_counts": {str(k): int(v) for k, v in statuses.items()},
        "median_good_matches": float(pair_rows["good_matches"].median()),
        "median_ransac_inliers": float(pair_rows["ransac_inliers"].median()),
        "median_inlier_ratio": float(pair_rows["inlier_ratio"].median()),
        "raw_path_length_px": compute_path_length(raw_xy),
        "raw_x_range_px": [float(np.nanmin(raw_xy[:, 0])), float(np.nanmax(raw_xy[:, 0]))],
        "raw_y_range_px": [float(np.nanmin(raw_xy[:, 1])), float(np.nanmax(raw_xy[:, 1]))],
        "alignment_to_reference": alignment_summary,
        "runtime_seconds": float(elapsed_s),
        "average_fps_processed": float(len(trajectory_df) / elapsed_s) if elapsed_s > 0 else None,
        "outputs": {
            "trajectory_csv": str(trajectory_csv),
            "summary_json": str(summary_json),
            "raw_plot": str(raw_plot),
            "comparison_plot": str(comparison_plot) if alignment_summary.get("available", False) else None,
        },
        "important_warning": (
            "This first version accumulates image-plane motion. Metric scale, yaw handling, "
            "altitude/camera calibration scaling, and IMU fusion are intentionally left for later blocks."
        ),
    }

    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary