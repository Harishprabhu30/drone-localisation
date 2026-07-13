from __future__ import annotations

import argparse
import json
import math
import re
import time
from pathlib import Path
from typing import Any

import cv2
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
DEFAULT_S6A1_JSON = Path(
    "outputs/satloc/reports/s6a_relative_motion/"
    "s6a1_orb_affine_stride_summary.json"
)


def parse_int_list(value: str) -> list[int]:
    values: list[int] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        number = int(item)
        if number <= 0:
            raise argparse.ArgumentTypeError("Values must be positive integers.")
        values.append(number)
    if not values:
        raise argparse.ArgumentTypeError("At least one integer is required.")
    return sorted(set(values))


def bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    normalized = series.astype(str).str.strip().str.lower()
    return normalized.isin({"true", "1", "yes", "y", "t"})


def resolve_path(value: Any) -> Path:
    path = Path(str(value)).expanduser()
    for candidate in (path, Path.cwd() / path):
        if candidate.exists():
            return candidate.resolve()
    return path


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


def finite_array(values: pd.Series | np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return array[np.isfinite(array)]


def finite_stat(values: pd.Series | np.ndarray, fn, default: float = float("nan")) -> float:
    array = finite_array(values)
    if array.size == 0:
        return default
    return float(fn(array))


def robust_location_scale(values: pd.Series | np.ndarray) -> tuple[float, float]:
    array = finite_array(values)
    if array.size == 0:
        return float("nan"), float("nan")
    median = float(np.median(array))
    mad = float(np.median(np.abs(array - median)))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale < 1e-12:
        std = float(np.std(array))
        scale = std if np.isfinite(std) and std >= 1e-12 else 1.0
    return median, scale


def robust_z(values: pd.Series | np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    median, scale = robust_location_scale(array)
    if not np.isfinite(median) or not np.isfinite(scale):
        return np.full_like(array, np.nan, dtype=float)
    return (array - median) / scale


def affine_to_homogeneous(affine: np.ndarray) -> np.ndarray:
    matrix = np.eye(3, dtype=float)
    matrix[:2, :] = np.asarray(affine, dtype=float)
    return matrix


def affine_metrics(affine: np.ndarray, width: int, height: int) -> dict[str, float]:
    a00, a01, tx = [float(value) for value in affine[0]]
    a10, a11, ty = [float(value) for value in affine[1]]
    scale_x = math.hypot(a00, a10)
    scale_y = math.hypot(a01, a11)
    scale = 0.5 * (scale_x + scale_y)
    rotation_deg = math.degrees(math.atan2(a10, a00))
    center = np.array([width / 2.0, height / 2.0, 1.0], dtype=float)
    transformed = affine @ center
    dx = float(transformed[0] - center[0])
    dy = float(transformed[1] - center[1])
    return {
        "scale": float(scale),
        "rotation_deg": float(rotation_deg),
        "center_dx_px": dx,
        "center_dy_px": dy,
        "center_motion_px": float(math.hypot(dx, dy)),
        "tx_px": tx,
        "ty_px": ty,
    }


def load_s6a1_configuration(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    configuration = payload.get("configuration", {})
    return configuration if isinstance(configuration, dict) else {}


def load_inputs(
    manifest_path: Path,
    pair_csv_path: Path,
    sequence: str,
    strides: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing S6A manifest: {manifest_path}")
    if not pair_csv_path.exists():
        raise FileNotFoundError(f"Missing S6A.1 pair CSV: {pair_csv_path}")

    manifest = pd.read_csv(manifest_path)
    pair_df = pd.read_csv(pair_csv_path)

    required_manifest = {
        "sequence_frame_id",
        "token0_id",
        "image_path_resolved",
    }
    missing_manifest = sorted(required_manifest.difference(manifest.columns))
    if missing_manifest:
        raise RuntimeError(f"Manifest is missing columns: {missing_manifest}")

    required_pair = {
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
        "center_content_motion_px",
        "reference_step_m_eval_only",
    }
    missing_pair = sorted(required_pair.difference(pair_df.columns))
    if missing_pair:
        raise RuntimeError(f"Pair CSV is missing columns: {missing_pair}")

    if "sequence" in manifest.columns:
        manifest = manifest[manifest["sequence"].astype(str) == sequence].copy()
    if manifest.empty:
        raise RuntimeError(f"No manifest rows found for sequence {sequence!r}.")

    manifest["sequence_frame_id"] = pd.to_numeric(
        manifest["sequence_frame_id"], errors="coerce"
    )
    manifest = manifest.dropna(subset=["sequence_frame_id"])
    manifest["sequence_frame_id"] = manifest["sequence_frame_id"].astype(int)
    manifest = manifest.sort_values("sequence_frame_id", kind="mergesort").reset_index(drop=True)

    expected = np.arange(len(manifest), dtype=int)
    actual = manifest["sequence_frame_id"].to_numpy(dtype=int)
    if not np.array_equal(expected, actual):
        raise RuntimeError("sequence_frame_id is not contiguous from zero.")

    pair_df["stride"] = pd.to_numeric(pair_df["stride"], errors="coerce")
    pair_df = pair_df.dropna(subset=["stride"])
    pair_df["stride"] = pair_df["stride"].astype(int)
    pair_df = pair_df[pair_df["stride"].isin(strides)].copy()
    if pair_df.empty:
        raise RuntimeError(f"No pair rows found for strides {strides}.")

    pair_df["affine_ok"] = bool_series(pair_df["affine_ok"])
    for column in (
        "pair_number",
        "frame_index_a",
        "frame_index_b",
        "token0_a",
        "token0_b",
        "good_matches",
        "inliers",
    ):
        pair_df[column] = pd.to_numeric(pair_df[column], errors="coerce")
    numeric_columns = (
        "inlier_ratio",
        "affine_a00",
        "affine_a01",
        "affine_a10",
        "affine_a11",
        "affine_tx_px",
        "affine_ty_px",
        "affine_scale",
        "affine_rotation_deg",
        "center_content_motion_px",
        "reference_step_m_eval_only",
    )
    for column in numeric_columns:
        pair_df[column] = pd.to_numeric(pair_df[column], errors="coerce")

    return manifest, pair_df.reset_index(drop=True)


def build_feature_cache(
    manifest: pd.DataFrame,
    resize_long: int,
    nfeatures: int,
) -> tuple[list[dict[str, Any]], float]:
    orb = cv2.ORB_create(
        nfeatures=nfeatures,
        scaleFactor=1.2,
        nlevels=8,
        edgeThreshold=31,
        patchSize=31,
        fastThreshold=20,
    )

    cache: list[dict[str, Any]] = []
    start = time.perf_counter()
    total = len(manifest)

    for index, row in manifest.iterrows():
        path = resolve_path(row["image_path_resolved"])
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise RuntimeError(f"OpenCV could not read image: {path}")
        gray = resize_long_side(gray, resize_long)
        keypoints, descriptors = orb.detectAndCompute(gray, None)
        points = (
            np.asarray([keypoint.pt for keypoint in keypoints], dtype=np.float32)
            if keypoints
            else np.empty((0, 2), dtype=np.float32)
        )
        cache.append(
            {
                "path": str(path),
                "gray": gray,
                "keypoints": keypoints,
                "points": points,
                "descriptors": descriptors,
                "width": int(gray.shape[1]),
                "height": int(gray.shape[0]),
            }
        )
        if (index + 1) % 100 == 0 or index + 1 == total:
            print(f"Feature cache: {index + 1}/{total}")

    return cache, time.perf_counter() - start


def match_and_affine(
    feature_a: dict[str, Any],
    feature_b: dict[str, Any],
    matcher: cv2.BFMatcher,
    ratio: float,
    ransac_thresh: float,
) -> dict[str, Any]:
    descriptors_a = feature_a["descriptors"]
    descriptors_b = feature_b["descriptors"]
    if descriptors_a is None or descriptors_b is None:
        return {
            "ok": False,
            "status": "no_descriptors",
            "good_matches": [],
            "inlier_mask": np.zeros(0, dtype=bool),
            "affine": None,
            "inliers": 0,
            "inlier_ratio": 0.0,
        }

    knn_matches = matcher.knnMatch(descriptors_a, descriptors_b, k=2)
    good: list[cv2.DMatch] = []
    for candidates in knn_matches:
        if len(candidates) != 2:
            continue
        best, second = candidates
        if best.distance < ratio * second.distance:
            good.append(best)

    if len(good) < 3:
        return {
            "ok": False,
            "status": "too_few_matches",
            "good_matches": good,
            "inlier_mask": np.zeros(len(good), dtype=bool),
            "affine": None,
            "inliers": 0,
            "inlier_ratio": 0.0,
        }

    points_a = np.asarray(
        [feature_a["points"][match.queryIdx] for match in good], dtype=np.float32
    )
    points_b = np.asarray(
        [feature_b["points"][match.trainIdx] for match in good], dtype=np.float32
    )

    affine, inlier_mask = cv2.estimateAffinePartial2D(
        points_a,
        points_b,
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_thresh,
        maxIters=3000,
        confidence=0.995,
        refineIters=10,
    )

    if affine is None or inlier_mask is None:
        return {
            "ok": False,
            "status": "affine_failed",
            "good_matches": good,
            "inlier_mask": np.zeros(len(good), dtype=bool),
            "affine": None,
            "inliers": 0,
            "inlier_ratio": 0.0,
        }

    mask = np.asarray(inlier_mask).ravel().astype(bool)
    inliers = int(mask.sum())
    metrics = affine_metrics(
        affine,
        width=feature_a["width"],
        height=feature_a["height"],
    )
    return {
        "ok": True,
        "status": "ok",
        "good_matches": good,
        "inlier_mask": mask,
        "affine": affine,
        "inliers": inliers,
        "inlier_ratio": float(inliers / max(len(good), 1)),
        **metrics,
    }


def stored_forward_affine(row: pd.Series) -> np.ndarray | None:
    values = np.array(
        [
            [row["affine_a00"], row["affine_a01"], row["affine_tx_px"]],
            [row["affine_a10"], row["affine_a11"], row["affine_ty_px"]],
        ],
        dtype=float,
    )
    if not np.isfinite(values).all():
        return None
    return values


def compute_forward_backward_consistency(
    pair_df: pd.DataFrame,
    feature_cache: list[dict[str, Any]],
    ratio: float,
    ransac_thresh: float,
) -> pd.DataFrame:
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    rows: list[dict[str, Any]] = []
    total = len(pair_df)

    for output_index, (_, row) in enumerate(pair_df.iterrows(), start=1):
        index_a = int(row["frame_index_a"])
        index_b = int(row["frame_index_b"])
        reverse = match_and_affine(
            feature_cache[index_b],
            feature_cache[index_a],
            matcher,
            ratio,
            ransac_thresh,
        )
        forward = stored_forward_affine(row) if bool(row["affine_ok"]) else None

        cycle_center_error = float("nan")
        cycle_translation = float("nan")
        cycle_rotation = float("nan")
        cycle_scale_error = float("nan")
        inverse_matrix_error = float("nan")

        if forward is not None and reverse.get("ok") and reverse.get("affine") is not None:
            forward_h = affine_to_homogeneous(forward)
            reverse_h = affine_to_homogeneous(reverse["affine"])
            cycle = reverse_h @ forward_h

            center = np.array(
                [
                    feature_cache[index_a]["width"] / 2.0,
                    feature_cache[index_a]["height"] / 2.0,
                    1.0,
                ],
                dtype=float,
            )
            center_cycle = cycle @ center
            cycle_center_error = float(np.linalg.norm(center_cycle[:2] - center[:2]))
            cycle_translation = float(np.linalg.norm(cycle[:2, 2]))
            cycle_rotation = float(math.degrees(math.atan2(cycle[1, 0], cycle[0, 0])))
            cycle_scale = float(math.sqrt(abs(np.linalg.det(cycle[:2, :2]))))
            cycle_scale_error = abs(cycle_scale - 1.0)

            try:
                true_inverse = np.linalg.inv(forward_h)
                inverse_matrix_error = float(
                    np.linalg.norm(reverse_h - true_inverse, ord="fro")
                )
            except np.linalg.LinAlgError:
                inverse_matrix_error = float("nan")

        rows.append(
            {
                "stride": int(row["stride"]),
                "pair_number": int(row["pair_number"]),
                "frame_index_a": index_a,
                "frame_index_b": index_b,
                "token0_a": int(row["token0_a"]),
                "token0_b": int(row["token0_b"]),
                "reverse_ok": bool(reverse.get("ok", False)),
                "reverse_good_matches": int(len(reverse.get("good_matches", []))),
                "reverse_inliers": int(reverse.get("inliers", 0)),
                "reverse_inlier_ratio": float(reverse.get("inlier_ratio", 0.0)),
                "reverse_rotation_deg": float(reverse.get("rotation_deg", float("nan"))),
                "reverse_scale": float(reverse.get("scale", float("nan"))),
                "fb_cycle_center_error_px": cycle_center_error,
                "fb_cycle_translation_px": cycle_translation,
                "fb_cycle_rotation_deg": cycle_rotation,
                "fb_cycle_scale_error": cycle_scale_error,
                "fb_inverse_matrix_error": inverse_matrix_error,
            }
        )

        if output_index % 100 == 0 or output_index == total:
            print(f"Forward/backward checks: {output_index}/{total}")

    return pd.DataFrame(rows)


def enrich_outlier_metrics(pair_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    enriched_groups: list[pd.DataFrame] = []
    threshold_rows: list[dict[str, Any]] = []

    for stride, group in pair_df.groupby("stride", sort=True):
        group = group.copy()
        group["abs_rotation_deg"] = np.abs(group["affine_rotation_deg"])
        group["abs_scale_error"] = np.abs(group["affine_scale"] - 1.0)

        group["inlier_ratio_low_z"] = -robust_z(group["inlier_ratio"])
        group["inliers_low_z"] = -robust_z(group["inliers"])
        group["rotation_high_z"] = robust_z(group["abs_rotation_deg"])
        group["scale_error_high_z"] = robust_z(group["abs_scale_error"])
        group["motion_high_z"] = np.abs(robust_z(group["center_content_motion_px"]))
        group["fb_cycle_high_z"] = robust_z(group["fb_cycle_center_error_px"])

        valid_reference = (
            np.isfinite(group["reference_step_m_eval_only"])
            & (group["reference_step_m_eval_only"] > 1e-6)
            & np.isfinite(group["center_content_motion_px"])
            & (group["center_content_motion_px"] > 0)
        )
        px_per_m = float("nan")
        if valid_reference.any():
            ratios = (
                group.loc[valid_reference, "center_content_motion_px"]
                / group.loc[valid_reference, "reference_step_m_eval_only"]
            )
            px_per_m = finite_stat(ratios, np.median)

        expected_motion = px_per_m * group["reference_step_m_eval_only"]
        residual_ratio = group["center_content_motion_px"] / expected_motion
        residual_ratio = residual_ratio.where(
            np.isfinite(residual_ratio) & (residual_ratio > 0)
        )
        group["eval_px_per_m_median"] = px_per_m
        group["eval_motion_residual_ratio"] = residual_ratio
        group["eval_abs_log_motion_residual"] = np.abs(np.log(residual_ratio))
        group["eval_motion_residual_high_z"] = robust_z(
            group["eval_abs_log_motion_residual"]
        )

        score_columns = [
            "inlier_ratio_low_z",
            "inliers_low_z",
            "rotation_high_z",
            "scale_error_high_z",
            "motion_high_z",
            "fb_cycle_high_z",
            "eval_motion_residual_high_z",
        ]
        score_matrix = np.column_stack(
            [np.maximum(pd.to_numeric(group[column], errors="coerce"), 0.0) for column in score_columns]
        )
        with np.errstate(all="ignore"):
            priority = np.nanmax(score_matrix, axis=1)
        priority[~np.isfinite(priority)] = 0.0
        group["inspection_priority_score"] = priority

        inlier_q05 = finite_stat(group["inlier_ratio"], lambda x: np.percentile(x, 5))
        inliers_q05 = finite_stat(group["inliers"], lambda x: np.percentile(x, 5))
        rotation_q99 = finite_stat(group["abs_rotation_deg"], lambda x: np.percentile(x, 99))
        scale_q99 = finite_stat(group["abs_scale_error"], lambda x: np.percentile(x, 99))
        fb_q99 = finite_stat(
            group["fb_cycle_center_error_px"], lambda x: np.percentile(x, 99)
        )
        eval_q99 = finite_stat(
            group["eval_abs_log_motion_residual"], lambda x: np.percentile(x, 99)
        )

        group["flag_current_weak"] = group["status"].astype(str) != "good"
        group["flag_low_inlier_ratio"] = group["inlier_ratio"] <= inlier_q05
        group["flag_low_inliers"] = group["inliers"] <= inliers_q05
        group["flag_large_rotation"] = group["abs_rotation_deg"] >= rotation_q99
        group["flag_scale_deviation"] = group["abs_scale_error"] >= scale_q99
        group["flag_motion_deviation"] = group["motion_high_z"] >= 4.0
        group["flag_fb_inconsistent"] = (
            group["fb_cycle_center_error_px"] >= max(3.0, fb_q99)
        )
        group["flag_eval_motion_residual"] = (
            group["eval_abs_log_motion_residual"] >= eval_q99
        )

        threshold_rows.append(
            {
                "stride": int(stride),
                "pairs": int(len(group)),
                "inlier_ratio_q05": inlier_q05,
                "inliers_q05": inliers_q05,
                "abs_rotation_deg_q99": rotation_q99,
                "abs_scale_error_q99": scale_q99,
                "fb_cycle_center_error_px_q99": fb_q99,
                "eval_abs_log_motion_residual_q99": eval_q99,
                "eval_px_per_m_median": px_per_m,
                "current_weak_count": int(group["flag_current_weak"].sum()),
                "fb_inconsistent_count": int(group["flag_fb_inconsistent"].sum()),
            }
        )
        enriched_groups.append(group)

    return pd.concat(enriched_groups, ignore_index=True), pd.DataFrame(threshold_rows)


def select_outliers(
    enriched: pd.DataFrame,
    top_per_reason: int,
    max_panels: int,
) -> pd.DataFrame:
    selected: dict[tuple[int, int, int], dict[str, Any]] = {}

    reason_specs = [
        ("current_weak", "flag_current_weak", "inspection_priority_score", False),
        ("lowest_inlier_ratio", None, "inlier_ratio", True),
        ("lowest_inliers", None, "inliers", True),
        ("largest_abs_rotation", None, "abs_rotation_deg", False),
        ("largest_scale_deviation", None, "abs_scale_error", False),
        ("largest_motion_deviation", None, "motion_high_z", False),
        ("largest_fb_cycle_error", None, "fb_cycle_center_error_px", False),
        (
            "largest_eval_motion_residual_eval_only",
            None,
            "eval_abs_log_motion_residual",
            False,
        ),
    ]

    for stride, group in enriched.groupby("stride", sort=True):
        for reason, flag_column, score_column, ascending in reason_specs:
            pool = group
            if flag_column is not None:
                pool = pool[pool[flag_column].astype(bool)]
            pool = pool[np.isfinite(pd.to_numeric(pool[score_column], errors="coerce"))]
            if pool.empty:
                continue
            chosen = pool.sort_values(score_column, ascending=ascending).head(top_per_reason)
            for _, row in chosen.iterrows():
                key = (
                    int(row["stride"]),
                    int(row["frame_index_a"]),
                    int(row["frame_index_b"]),
                )
                if key not in selected:
                    selected[key] = {
                        "row": row.to_dict(),
                        "reasons": [],
                    }
                selected[key]["reasons"].append(reason)

    rows: list[dict[str, Any]] = []
    for item in selected.values():
        row = dict(item["row"])
        row["inspection_reasons"] = ";".join(sorted(set(item["reasons"])))
        row["inspection_reason_count"] = len(set(item["reasons"]))
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    selected_df = pd.DataFrame(rows)
    selected_df = selected_df.sort_values(
        ["inspection_reason_count", "inspection_priority_score"],
        ascending=[False, False],
    ).reset_index(drop=True)
    if max_panels > 0:
        selected_df = selected_df.head(max_panels).copy()
    selected_df.insert(0, "panel_rank", np.arange(1, len(selected_df) + 1, dtype=int))
    return selected_df


def safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return value.strip("_") or "pair"


def draw_matches_image(
    feature_a: dict[str, Any],
    feature_b: dict[str, Any],
    result: dict[str, Any],
    inliers_only: bool,
    max_matches: int,
) -> np.ndarray:
    matches = list(result.get("good_matches", []))
    mask = np.asarray(result.get("inlier_mask", np.zeros(len(matches), dtype=bool))).astype(bool)
    if not matches:
        left = cv2.cvtColor(feature_a["gray"], cv2.COLOR_GRAY2BGR)
        right = cv2.cvtColor(feature_b["gray"], cv2.COLOR_GRAY2BGR)
        return np.hstack([left, right])

    indexed = list(enumerate(matches))
    if inliers_only:
        indexed = [(index, match) for index, match in indexed if index < len(mask) and mask[index]]
    indexed = sorted(indexed, key=lambda item: item[1].distance)[:max_matches]
    chosen_matches = [item[1] for item in indexed]

    return cv2.drawMatches(
        feature_a["gray"],
        feature_a["keypoints"],
        feature_b["gray"],
        feature_b["keypoints"],
        chosen_matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )


def save_pair_panel(
    selected_row: pd.Series,
    feature_cache: list[dict[str, Any]],
    ratio: float,
    ransac_thresh: float,
    max_draw_matches: int,
    panels_dir: Path,
) -> dict[str, Any]:
    index_a = int(selected_row["frame_index_a"])
    index_b = int(selected_row["frame_index_b"])
    feature_a = feature_cache[index_a]
    feature_b = feature_cache[index_b]
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    forward = match_and_affine(
        feature_a, feature_b, matcher, ratio, ransac_thresh
    )
    reverse = match_and_affine(
        feature_b, feature_a, matcher, ratio, ransac_thresh
    )

    all_matches_image = draw_matches_image(
        feature_a,
        feature_b,
        forward,
        inliers_only=False,
        max_matches=max_draw_matches,
    )
    inlier_matches_image = draw_matches_image(
        feature_a,
        feature_b,
        forward,
        inliers_only=True,
        max_matches=max_draw_matches,
    )
    reverse_inlier_image = draw_matches_image(
        feature_b,
        feature_a,
        reverse,
        inliers_only=True,
        max_matches=max_draw_matches,
    )

    reasons = str(selected_row["inspection_reasons"])
    title = (
        f"S6A.1B rank {int(selected_row['panel_rank'])} | stride {int(selected_row['stride'])} | "
        f"frame {index_a}→{index_b} | token {int(selected_row['token0_a'])}→"
        f"{int(selected_row['token0_b'])}\n"
        f"reasons: {reasons}\n"
        f"stored: good={int(selected_row['good_matches'])}, inliers={int(selected_row['inliers'])}, "
        f"IR={selected_row['inlier_ratio']:.3f}, rot={selected_row['affine_rotation_deg']:.3f}°, "
        f"scale={selected_row['affine_scale']:.6f}, motion={selected_row['center_content_motion_px']:.2f}px | "
        f"FB cycle={selected_row['fb_cycle_center_error_px']:.2f}px"
    )

    fig = plt.figure(figsize=(16, 12))
    grid = fig.add_gridspec(2, 2)

    axis_a = fig.add_subplot(grid[0, 0])
    axis_a.imshow(feature_a["gray"], cmap="gray")
    axis_a.set_title(f"Frame A: {index_a} / token {int(selected_row['token0_a'])}")
    axis_a.axis("off")

    axis_b = fig.add_subplot(grid[0, 1])
    axis_b.imshow(feature_b["gray"], cmap="gray")
    axis_b.set_title(f"Frame B: {index_b} / token {int(selected_row['token0_b'])}")
    axis_b.axis("off")

    axis_all = fig.add_subplot(grid[1, 0])
    axis_all.imshow(cv2.cvtColor(all_matches_image, cv2.COLOR_BGR2RGB))
    axis_all.set_title(
        f"Forward ratio-test matches (up to {max_draw_matches}); "
        f"recomputed IR={forward.get('inlier_ratio', 0.0):.3f}"
    )
    axis_all.axis("off")

    axis_inliers = fig.add_subplot(grid[1, 1])
    axis_inliers.imshow(cv2.cvtColor(inlier_matches_image, cv2.COLOR_BGR2RGB))
    axis_inliers.set_title(
        f"Forward RANSAC inliers (up to {max_draw_matches}); "
        f"reverse IR={reverse.get('inlier_ratio', 0.0):.3f}"
    )
    axis_inliers.axis("off")

    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92))

    filename = (
        f"s6a1b_rank{int(selected_row['panel_rank']):03d}_"
        f"s{int(selected_row['stride'])}_"
        f"f{index_a:04d}_to_f{index_b:04d}_"
        f"{safe_name(reasons)[:80]}.png"
    )
    output_path = panels_dir / filename
    fig.savefig(output_path, dpi=160)
    plt.close(fig)

    reverse_path = output_path.with_name(output_path.stem + "_reverse_inliers.png")
    cv2.imwrite(str(reverse_path), reverse_inlier_image)

    return {
        "panel_path": str(output_path),
        "reverse_inlier_image_path": str(reverse_path),
        "recomputed_forward_ok": bool(forward.get("ok", False)),
        "recomputed_forward_good_matches": int(len(forward.get("good_matches", []))),
        "recomputed_forward_inliers": int(forward.get("inliers", 0)),
        "recomputed_forward_inlier_ratio": float(forward.get("inlier_ratio", 0.0)),
        "recomputed_reverse_ok": bool(reverse.get("ok", False)),
        "recomputed_reverse_good_matches": int(len(reverse.get("good_matches", []))),
        "recomputed_reverse_inliers": int(reverse.get("inliers", 0)),
        "recomputed_reverse_inlier_ratio": float(reverse.get("inlier_ratio", 0.0)),
    }


def save_metric_plot(
    enriched: pd.DataFrame,
    selected: pd.DataFrame,
    value_column: str,
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    plt.figure(figsize=(12, 6))
    for stride, group in enriched.groupby("stride", sort=True):
        plt.plot(
            group["frame_index_a"],
            group[value_column],
            linewidth=1.0,
            label=f"stride {stride}",
        )
    if not selected.empty:
        plt.scatter(
            selected["frame_index_a"],
            selected[value_column],
            marker="x",
            s=40,
            label="selected inspection pairs",
        )
    plt.xlabel("Starting frame index")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def build_summary(enriched: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for stride, group in enriched.groupby("stride", sort=True):
        rows.append(
            {
                "stride": int(stride),
                "pairs": int(len(group)),
                "current_weak_pairs": int(group["flag_current_weak"].sum()),
                "lowest_inlier_ratio": finite_stat(group["inlier_ratio"], np.min),
                "lowest_inliers": finite_stat(group["inliers"], np.min),
                "largest_abs_rotation_deg": finite_stat(group["abs_rotation_deg"], np.max),
                "largest_abs_scale_error": finite_stat(group["abs_scale_error"], np.max),
                "largest_fb_cycle_center_error_px": finite_stat(
                    group["fb_cycle_center_error_px"], np.max
                ),
                "median_fb_cycle_center_error_px": finite_stat(
                    group["fb_cycle_center_error_px"], np.median
                ),
                "fb_inconsistent_pairs": int(group["flag_fb_inconsistent"].sum()),
                "largest_eval_abs_log_motion_residual": finite_stat(
                    group["eval_abs_log_motion_residual"], np.max
                ),
                "eval_px_per_m_median": finite_stat(
                    group["eval_px_per_m_median"], np.median
                ),
            }
        )
    return pd.DataFrame(rows)


def print_summary(
    summary_df: pd.DataFrame,
    selected_df: pd.DataFrame,
    feature_seconds: float,
    fb_seconds: float,
) -> None:
    print("\nS6A.1B affine outlier and failure inspection")
    print("---------------------------------------------")
    print(f"Feature-cache time:              {feature_seconds:.2f} s")
    print(f"Forward/backward check time:     {fb_seconds:.2f} s")
    print(f"Selected inspection panels:      {len(selected_df)}")

    for row in summary_df.to_dict(orient="records"):
        print(f"\nStride {row['stride']}")
        print(f"  Pairs:                         {row['pairs']}")
        print(f"  Current weak pairs:            {row['current_weak_pairs']}")
        print(f"  Lowest inlier ratio:           {row['lowest_inlier_ratio']:.3f}")
        print(f"  Lowest inlier count:           {row['lowest_inliers']:.0f}")
        print(f"  Largest |rotation| [deg]:      {row['largest_abs_rotation_deg']:.3f}")
        print(f"  Largest |scale-1|:             {row['largest_abs_scale_error']:.6f}")
        print(
            f"  Median FB cycle error [px]:    "
            f"{row['median_fb_cycle_center_error_px']:.3f}"
        )
        print(
            f"  Largest FB cycle error [px]:   "
            f"{row['largest_fb_cycle_center_error_px']:.3f}"
        )
        print(f"  FB-inconsistent pairs:         {row['fb_inconsistent_pairs']}")
        print(
            f"  Median px/m (evaluation only): "
            f"{row['eval_px_per_m_median']:.3f}"
        )

    print(
        "\nImportant: reference-derived motion residuals are diagnostic only. "
        "They are not used as an estimator confidence gate."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "SatLoc S6A.1B: inspect affine outliers, recompute reverse transforms, "
            "measure forward/backward cycle consistency, and generate match panels."
        )
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--pair-csv", type=Path, default=DEFAULT_PAIR_CSV)
    parser.add_argument("--s6a1-json", type=Path, default=DEFAULT_S6A1_JSON)
    parser.add_argument("--sequence", default="traj01")
    parser.add_argument("--strides", type=parse_int_list, default=parse_int_list("1,2,5"))
    parser.add_argument("--resize-long", type=int, default=None)
    parser.add_argument("--nfeatures", type=int, default=None)
    parser.add_argument("--ratio", type=float, default=None)
    parser.add_argument("--ransac-thresh", type=float, default=None)
    parser.add_argument("--top-per-reason", type=int, default=3)
    parser.add_argument("--max-panels", type=int, default=36)
    parser.add_argument("--max-draw-matches", type=int, default=80)
    args = parser.parse_args()

    configuration = load_s6a1_configuration(args.s6a1_json)
    resize_long = int(
        args.resize_long
        if args.resize_long is not None
        else configuration.get("resize_long", 960)
    )
    nfeatures = int(
        args.nfeatures
        if args.nfeatures is not None
        else configuration.get("nfeatures", 1200)
    )
    ratio = float(
        args.ratio if args.ratio is not None else configuration.get("ratio", 0.75)
    )
    ransac_thresh = float(
        args.ransac_thresh
        if args.ransac_thresh is not None
        else configuration.get("ransac_thresh", 3.0)
    )

    metadata_dir = args.output_root / "metadata" / "s6a_relative_motion"
    reports_dir = args.output_root / "reports" / "s6a_relative_motion"
    figures_dir = args.output_root / "figures" / "s6a_relative_motion"
    panels_dir = figures_dir / "s6a1b_affine_outlier_panels"
    for directory in (metadata_dir, reports_dir, figures_dir, panels_dir):
        directory.mkdir(parents=True, exist_ok=True)

    manifest, pair_df = load_inputs(
        args.manifest,
        args.pair_csv,
        args.sequence,
        args.strides,
    )

    feature_cache, feature_seconds = build_feature_cache(
        manifest,
        resize_long=resize_long,
        nfeatures=nfeatures,
    )

    fb_start = time.perf_counter()
    fb_df = compute_forward_backward_consistency(
        pair_df,
        feature_cache,
        ratio=ratio,
        ransac_thresh=ransac_thresh,
    )
    fb_seconds = time.perf_counter() - fb_start

    merge_keys = [
        "stride",
        "pair_number",
        "frame_index_a",
        "frame_index_b",
        "token0_a",
        "token0_b",
    ]
    enriched = pair_df.merge(fb_df, on=merge_keys, how="left", validate="one_to_one")
    enriched, thresholds_df = enrich_outlier_metrics(enriched)
    selected_df = select_outliers(
        enriched,
        top_per_reason=max(1, args.top_per_reason),
        max_panels=max(0, args.max_panels),
    )

    panel_records: list[dict[str, Any]] = []
    if not selected_df.empty:
        total_panels = len(selected_df)
        for index, (_, row) in enumerate(selected_df.iterrows(), start=1):
            panel_info = save_pair_panel(
                row,
                feature_cache,
                ratio=ratio,
                ransac_thresh=ransac_thresh,
                max_draw_matches=max(1, args.max_draw_matches),
                panels_dir=panels_dir,
            )
            record = row.to_dict()
            record.update(panel_info)
            panel_records.append(record)
            print(f"Panels: {index}/{total_panels}")
        selected_df = pd.DataFrame(panel_records)

    summary_df = build_summary(enriched)

    enriched_path = metadata_dir / "s6a1b_pair_diagnostics_enriched.csv"
    fb_path = metadata_dir / "s6a1b_forward_backward_consistency.csv"
    thresholds_path = metadata_dir / "s6a1b_stride_outlier_thresholds.csv"
    selected_path = metadata_dir / "s6a1b_selected_outlier_pairs.csv"
    summary_csv_path = metadata_dir / "s6a1b_outlier_failure_summary.csv"

    enriched.to_csv(enriched_path, index=False)
    fb_df.to_csv(fb_path, index=False)
    thresholds_df.to_csv(thresholds_path, index=False)
    selected_df.to_csv(selected_path, index=False)
    summary_df.to_csv(summary_csv_path, index=False)

    save_metric_plot(
        enriched,
        selected_df,
        "inlier_ratio",
        "RANSAC inlier ratio",
        "S6A.1B inlier ratio by sequence frame",
        figures_dir / "s6a1b_inlier_ratio_vs_frame.png",
    )
    save_metric_plot(
        enriched,
        selected_df,
        "abs_rotation_deg",
        "Absolute affine rotation [deg]",
        "S6A.1B affine rotation outliers",
        figures_dir / "s6a1b_abs_rotation_vs_frame.png",
    )
    save_metric_plot(
        enriched,
        selected_df,
        "abs_scale_error",
        "Absolute affine scale deviation |scale - 1|",
        "S6A.1B affine scale-deviation outliers",
        figures_dir / "s6a1b_scale_deviation_vs_frame.png",
    )
    save_metric_plot(
        enriched,
        selected_df,
        "fb_cycle_center_error_px",
        "Forward/backward cycle centre error [px]",
        "S6A.1B forward/backward transform consistency",
        figures_dir / "s6a1b_fb_cycle_error_vs_frame.png",
    )
    save_metric_plot(
        enriched,
        selected_df,
        "eval_abs_log_motion_residual",
        "Absolute log motion residual — evaluation only",
        "S6A.1B visual/reference motion residuals (evaluation only)",
        figures_dir / "s6a1b_eval_motion_residual_vs_frame.png",
    )

    report_payload = {
        "stage": "S6A.1B",
        "sequence": args.sequence,
        "purpose": (
            "Inspect affine outliers before trajectory accumulation; identify low-consensus, "
            "large-rotation, scale-change, motion-deviation, and forward/backward-inconsistent pairs."
        ),
        "configuration": {
            "manifest": str(args.manifest),
            "pair_csv": str(args.pair_csv),
            "s6a1_json": str(args.s6a1_json),
            "strides": args.strides,
            "resize_long": resize_long,
            "nfeatures": nfeatures,
            "ratio": ratio,
            "ransac_thresh": ransac_thresh,
            "top_per_reason": args.top_per_reason,
            "max_panels": args.max_panels,
            "max_draw_matches": args.max_draw_matches,
        },
        "locked_reference_rule": (
            "Reference displacement is used only for post-estimation motion-residual diagnostics. "
            "It is not used for feature matching, affine estimation, forward/backward checking, "
            "or estimator-side confidence selection."
        ),
        "feature_cache_seconds": float(feature_seconds),
        "forward_backward_seconds": float(fb_seconds),
        "selected_panel_count": int(len(selected_df)),
        "stride_summaries": summary_df.to_dict(orient="records"),
        "stride_thresholds": thresholds_df.to_dict(orient="records"),
        "outputs": {
            "enriched_pair_csv": str(enriched_path),
            "forward_backward_csv": str(fb_path),
            "thresholds_csv": str(thresholds_path),
            "selected_pairs_csv": str(selected_path),
            "summary_csv": str(summary_csv_path),
            "panels_dir": str(panels_dir),
        },
    }
    report_path = reports_dir / "s6a1b_affine_outlier_failure_summary.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report_payload, handle, indent=2)

    print_summary(summary_df, selected_df, feature_seconds, fb_seconds)
    print("\nSaved outputs")
    print("-------------")
    print(enriched_path)
    print(fb_path)
    print(thresholds_path)
    print(selected_path)
    print(summary_csv_path)
    print(report_path)
    print(figures_dir / "s6a1b_inlier_ratio_vs_frame.png")
    print(figures_dir / "s6a1b_abs_rotation_vs_frame.png")
    print(figures_dir / "s6a1b_scale_deviation_vs_frame.png")
    print(figures_dir / "s6a1b_fb_cycle_error_vs_frame.png")
    print(figures_dir / "s6a1b_eval_motion_residual_vs_frame.png")
    print(panels_dir)


if __name__ == "__main__":
    main()
