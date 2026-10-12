"""Frozen ORB homography + sub-tile projection contract used by VILLOC RG work.

This module packages the historically validated R4.11 / RG1 geometry frontend so
new executable scripts do not import one another. Do not change constants or
pixel-to-map conventions in-place; version the contract if an ablation is needed.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


PREPROCESS = "clahe_luma"
RESIZE_LONG = 1024
NFEATURES = 1800
FAST_THRESHOLD = 7
EDGE_THRESHOLD = 15
PATCH_SIZE = 31
LOWE_RATIO = 0.80
RANSAC_THRESHOLD_PX = 5.0
MIN_HOMOGRAPHY_INLIERS = 4


def contract() -> dict[str, Any]:
    return {
        "preprocess": PREPROCESS,
        "resize_long": RESIZE_LONG,
        "orb_nfeatures": NFEATURES,
        "fast_threshold": FAST_THRESHOLD,
        "edge_threshold": EDGE_THRESHOLD,
        "patch_size": PATCH_SIZE,
        "lowe_ratio": LOWE_RATIO,
        "ransac_threshold_px": RANSAC_THRESHOLD_PX,
        "min_homography_inliers": MIN_HOMOGRAPHY_INLIERS,
        "homography_direction": "query_pixel_to_satellite_tile_pixel",
        "query_point": "processed_image_center",
        "map_conversion": "raster_outer_edge_pixel_center_offset_0p5",
    }


def bbox_coverage(points: np.ndarray, image_hw: tuple[int, int]) -> float:
    h, w = image_hw
    if h <= 0 or w <= 0 or len(points) == 0:
        return 0.0
    xs = points[:, 0]
    ys = points[:, 1]
    area = max(0.0, float(xs.max() - xs.min())) * max(
        0.0, float(ys.max() - ys.min())
    )
    return float(np.clip(area / float(w * h), 0.0, 1.0))


@dataclass
class FeaturePack:
    ok: bool
    image_shape: tuple[int, int]
    keypoints: list
    descriptors: np.ndarray | None
    error: str | None


def read_image_for_verifier(path: Path) -> tuple[np.ndarray | None, str | None]:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return None, f"cv2.imread failed: {path}"

    h, w = img.shape[:2]
    scale = float(RESIZE_LONG) / float(max(h, w))
    # Historical verifier ONLY downsizes.
    if scale < 1.0:
        img = cv2.resize(
            img,
            (int(round(w * scale)), int(round(h * scale))),
            interpolation=cv2.INTER_AREA,
        )

    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y = ycrcb[:, :, 0]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(y), None


def create_detector():
    return cv2.ORB_create(
        nfeatures=NFEATURES,
        fastThreshold=FAST_THRESHOLD,
        edgeThreshold=EDGE_THRESHOLD,
        patchSize=PATCH_SIZE,
    )


def compute_features(path: Path, detector) -> FeaturePack:
    gray, error = read_image_for_verifier(path)
    if gray is None:
        return FeaturePack(False, (0, 0), [], None, error)

    keypoints, descriptors = detector.detectAndCompute(gray, None)
    if descriptors is None or len(keypoints) == 0:
        return FeaturePack(
            False,
            gray.shape[:2],
            keypoints or [],
            None,
            "no descriptors",
        )
    return FeaturePack(True, gray.shape[:2], keypoints, descriptors, None)


def verify_pair_with_geometry(q: FeaturePack, s: FeaturePack) -> dict[str, Any]:
    """Reproduce the frozen R4.11/RG1 query->tile ORB homography evidence."""
    result: dict[str, Any] = {
        "good_matches": 0,
        "inliers": 0,
        "inlier_ratio": 0.0,
        "query_inlier_coverage": 0.0,
        "sat_inlier_coverage": 0.0,
        "homography_ok": False,
        "verifier_score": 0.0,
        "H": None,
        "query_center_u_px": math.nan,
        "query_center_v_px": math.nan,
        "projected_tile_u_px": math.nan,
        "projected_tile_v_px": math.nan,
        "projective_denominator": math.nan,
        "inlier_reprojection_rmse_px": math.nan,
        "error": None,
    }
    if (
        not q.ok
        or not s.ok
        or q.descriptors is None
        or s.descriptors is None
    ):
        result["error"] = q.error or s.error or "features missing"
        return result

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    try:
        knn = matcher.knnMatch(q.descriptors, s.descriptors, k=2)
    except cv2.error as exc:
        result["error"] = f"knnMatch failed: {exc}"
        return result

    good = []
    for pair in knn:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < LOWE_RATIO * n.distance:
            good.append(m)

    result["good_matches"] = int(len(good))
    if len(good) < 4:
        result["verifier_score"] = float(0.03 * len(good))
        return result

    q_pts = np.float32([q.keypoints[m.queryIdx].pt for m in good])
    s_pts = np.float32([s.keypoints[m.trainIdx].pt for m in good])

    H, mask = cv2.findHomography(
        q_pts,
        s_pts,
        cv2.RANSAC,
        RANSAC_THRESHOLD_PX,
    )
    if mask is None:
        inlier_mask = np.zeros(len(good), dtype=bool)
    else:
        inlier_mask = mask.ravel().astype(bool)

    inliers = int(inlier_mask.sum())
    q_cov = bbox_coverage(q_pts[inlier_mask], q.image_shape)
    s_cov = bbox_coverage(s_pts[inlier_mask], s.image_shape)
    inlier_ratio = float(inliers / max(1, len(good)))
    homography_ok = bool(H is not None and inliers >= MIN_HOMOGRAPHY_INLIERS)

    verifier_score = (
        float(inliers)
        + 0.04 * float(len(good))
        + 8.0 * float(inlier_ratio)
        + 4.0 * float(math.sqrt(max(q_cov, 0.0)))
        + 2.0 * float(math.sqrt(max(s_cov, 0.0)))
    )
    result.update(
        {
            "inliers": inliers,
            "inlier_ratio": inlier_ratio,
            "query_inlier_coverage": q_cov,
            "sat_inlier_coverage": s_cov,
            "homography_ok": homography_ok,
            "verifier_score": verifier_score,
            "H": H,
        }
    )
    if not homography_ok:
        return result

    if inliers > 0:
        projected_inliers = cv2.perspectiveTransform(
            q_pts[inlier_mask].reshape(-1, 1, 2),
            H,
        ).reshape(-1, 2)
        residual = projected_inliers - s_pts[inlier_mask]
        result["inlier_reprojection_rmse_px"] = float(
            np.sqrt(np.mean(np.sum(residual ** 2, axis=1)))
        )

    qh, qw = q.image_shape
    center_u = float(qw) / 2.0
    center_v = float(qh) / 2.0
    result["query_center_u_px"] = center_u
    result["query_center_v_px"] = center_v

    denominator = float(
        H[2, 0] * center_u + H[2, 1] * center_v + H[2, 2]
    )
    result["projective_denominator"] = denominator

    point = np.float32([[[center_u, center_v]]])
    projected = cv2.perspectiveTransform(point, H).reshape(2)
    result["projected_tile_u_px"] = float(projected[0])
    result["projected_tile_v_px"] = float(projected[1])
    return result


def tile_pixel_to_map(
    u: float,
    v: float,
    tile_hw: tuple[int, int],
    left_easting: float,
    right_easting: float,
    bottom_northing: float,
    top_northing: float,
) -> tuple[float, float]:
    """Convert OpenCV pixel-centre coordinates to raster-outer-edge map bounds."""
    h, w = tile_hw
    easting = left_easting + ((u + 0.5) / float(w)) * (
        right_easting - left_easting
    )
    northing = top_northing - ((v + 0.5) / float(h)) * (
        top_northing - bottom_northing
    )
    return float(easting), float(northing)
