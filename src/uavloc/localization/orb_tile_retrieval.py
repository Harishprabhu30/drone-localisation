from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import cv2
import numpy as np


@dataclass
class ORBPairMatchResult:
    variant: str
    query_keypoints: int
    tile_keypoints: int
    raw_matches: int
    good_matches: int
    ransac_inliers: int
    inlier_ratio: float
    homography_success: bool
    score: float
    elapsed_ms: float
    homography: np.ndarray | None
    good_match_objects: list[Any]
    inlier_mask: np.ndarray | None
    query_kp: list[Any]
    tile_kp: list[Any]
    query_processed: np.ndarray
    tile_processed: np.ndarray


def preprocess_bgr(
    bgr: np.ndarray,
    variant: str,
    clahe_clip_limit: float = 2.0,
    clahe_tile_size: int = 8,
    alt_clahe_clip_limit: float = 1.0,
    alt_clahe_tile_size: int = 8,
    bilateral_d: int = 13,
    bilateral_sigma_color: float = 30,
    bilateral_sigma_space: float = 55,
) -> np.ndarray:
    """Return single-channel uint8 image for ORB."""

    if bgr is None:
        raise ValueError("Input image is None.")

    if len(bgr.shape) == 2:
        gray = bgr
    else:
        ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
        gray = ycrcb[:, :, 0]

    if variant in {"V0_gray", "gray"}:
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if len(bgr.shape) == 3 else bgr

    if variant in {"V1_luma", "luma"}:
        return gray

    if variant in {"V2_clahe_luma", "clahe_luma"}:
        clahe = cv2.createCLAHE(
            clipLimit=float(clahe_clip_limit),
            tileGridSize=(int(clahe_tile_size), int(clahe_tile_size)),
        )
        return clahe.apply(gray)

    if variant in {"V3_bilateral_clahe", "bilateral_clahe"}:
        clahe = cv2.createCLAHE(
            clipLimit=float(clahe_clip_limit),
            tileGridSize=(int(clahe_tile_size), int(clahe_tile_size)),
        )
        enhanced = clahe.apply(gray)
        return cv2.bilateralFilter(
            enhanced,
            int(bilateral_d),
            float(bilateral_sigma_color),
            float(bilateral_sigma_space),
        )

    if variant in {"V4_bilateral_alt_clahe", "bilateral_alt_clahe"}:
        clahe = cv2.createCLAHE(
            clipLimit=float(alt_clahe_clip_limit),
            tileGridSize=(int(alt_clahe_tile_size), int(alt_clahe_tile_size)),
        )
        enhanced = clahe.apply(gray)
        return cv2.bilateralFilter(
            enhanced,
            int(bilateral_d),
            float(bilateral_sigma_color),
            float(bilateral_sigma_space),
        )

    raise ValueError(f"Unknown preprocessing variant: {variant}")


def detect_orb(
    image_gray: np.ndarray,
    nfeatures: int = 1200,
) -> tuple[list[Any], np.ndarray | None]:
    orb = cv2.ORB_create(nfeatures=int(nfeatures))
    kp, des = orb.detectAndCompute(image_gray, None)
    return kp, des


def ratio_match_orb(
    query_des: np.ndarray | None,
    tile_des: np.ndarray | None,
    ratio: float = 0.75,
) -> list[Any]:
    if query_des is None or tile_des is None:
        return []

    if len(query_des) < 2 or len(tile_des) < 2:
        return []

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn_matches = matcher.knnMatch(query_des, tile_des, k=2)

    good = []
    for pair in knn_matches:
        if len(pair) != 2:
            continue
        m, n = pair
        if m.distance < float(ratio) * n.distance:
            good.append(m)

    good = sorted(good, key=lambda x: x.distance)
    return good


def estimate_homography_ransac(
    query_kp: list[Any],
    tile_kp: list[Any],
    good_matches: list[Any],
    ransac_thresh: float = 5.0,
) -> tuple[np.ndarray | None, np.ndarray | None, int, float, bool]:
    if len(good_matches) < 4:
        return None, None, 0, 0.0, False

    src_pts = np.float32([query_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([tile_kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, float(ransac_thresh))

    if mask is None:
        return H, None, 0, 0.0, False

    inliers = int(mask.ravel().sum())
    ratio = float(inliers / max(len(good_matches), 1))
    success = bool(H is not None and inliers >= 4)

    return H, mask.ravel().astype(bool), inliers, ratio, success


def match_orb_pair(
    query_bgr: np.ndarray,
    tile_bgr: np.ndarray,
    variant: str = "V2_clahe_luma",
    nfeatures: int = 1200,
    ratio: float = 0.75,
    ransac_thresh: float = 5.0,
    clahe_clip_limit: float = 2.0,
    clahe_tile_size: int = 8,
    alt_clahe_clip_limit: float = 1.0,
    alt_clahe_tile_size: int = 8,
    bilateral_d: int = 13,
    bilateral_sigma_color: float = 30,
    bilateral_sigma_space: float = 55,
) -> ORBPairMatchResult:
    start = perf_counter()

    query_processed = preprocess_bgr(
        query_bgr,
        variant=variant,
        clahe_clip_limit=clahe_clip_limit,
        clahe_tile_size=clahe_tile_size,
        alt_clahe_clip_limit=alt_clahe_clip_limit,
        alt_clahe_tile_size=alt_clahe_tile_size,
        bilateral_d=bilateral_d,
        bilateral_sigma_color=bilateral_sigma_color,
        bilateral_sigma_space=bilateral_sigma_space,
    )
    tile_processed = preprocess_bgr(
        tile_bgr,
        variant=variant,
        clahe_clip_limit=clahe_clip_limit,
        clahe_tile_size=clahe_tile_size,
        alt_clahe_clip_limit=alt_clahe_clip_limit,
        alt_clahe_tile_size=alt_clahe_tile_size,
        bilateral_d=bilateral_d,
        bilateral_sigma_color=bilateral_sigma_color,
        bilateral_sigma_space=bilateral_sigma_space,
    )

    query_kp, query_des = detect_orb(query_processed, nfeatures=nfeatures)
    tile_kp, tile_des = detect_orb(tile_processed, nfeatures=nfeatures)

    good = ratio_match_orb(query_des, tile_des, ratio=ratio)

    H, inlier_mask, inliers, inlier_ratio, homography_success = estimate_homography_ransac(
        query_kp=query_kp,
        tile_kp=tile_kp,
        good_matches=good,
        ransac_thresh=ransac_thresh,
    )

    score = float(inliers + 0.1 * len(good))
    elapsed_ms = float((perf_counter() - start) * 1000.0)

    raw_matches = 0
    if query_des is not None and tile_des is not None:
        raw_matches = int(len(query_des))

    return ORBPairMatchResult(
        variant=variant,
        query_keypoints=int(len(query_kp)),
        tile_keypoints=int(len(tile_kp)),
        raw_matches=raw_matches,
        good_matches=int(len(good)),
        ransac_inliers=int(inliers),
        inlier_ratio=float(inlier_ratio),
        homography_success=bool(homography_success),
        score=score,
        elapsed_ms=elapsed_ms,
        homography=H,
        good_match_objects=good,
        inlier_mask=inlier_mask,
        query_kp=query_kp,
        tile_kp=tile_kp,
        query_processed=query_processed,
        tile_processed=tile_processed,
    )