"""Richer blind geometric evidence for Top20 candidate discrimination.

The baseline ORB/RANSAC contract is intentionally identical to RG1/RG2.
This module adds diagnostics around the same correspondence set and homography;
it does not use reference coordinates or choose a candidate winner.
"""
from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np

from uavloc.geometry import orb_subtile


GRID_SIZE = 4


def _finite_summary(values: np.ndarray) -> dict[str, float]:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return {"median": math.nan, "p90": math.nan, "rmse": math.nan}
    return {
        "median": float(np.median(v)),
        "p90": float(np.quantile(v, 0.90)),
        "rmse": float(np.sqrt(np.mean(v * v))),
    }


def _convex_hull_coverage(points: np.ndarray, image_hw: tuple[int, int]) -> float:
    h, w = image_hw
    pts = np.asarray(points, dtype=np.float32)
    if h <= 0 or w <= 0 or len(pts) < 3:
        return 0.0
    hull = cv2.convexHull(pts.reshape(-1, 1, 2))
    area = float(cv2.contourArea(hull))
    return float(np.clip(area / float(h * w), 0.0, 1.0))


def _grid_support(points: np.ndarray, image_hw: tuple[int, int]) -> tuple[int, float, float]:
    h, w = image_hw
    pts = np.asarray(points, dtype=float)
    if h <= 0 or w <= 0 or len(pts) == 0:
        return 0, 0.0, 0.0
    gx = np.clip((pts[:, 0] / max(float(w), 1.0) * GRID_SIZE).astype(int), 0, GRID_SIZE - 1)
    gy = np.clip((pts[:, 1] / max(float(h), 1.0) * GRID_SIZE).astype(int), 0, GRID_SIZE - 1)
    ids = gy * GRID_SIZE + gx
    counts = np.bincount(ids, minlength=GRID_SIZE * GRID_SIZE).astype(float)
    occupied = int((counts > 0).sum())
    occupancy = float(occupied / float(GRID_SIZE * GRID_SIZE))
    probs = counts[counts > 0] / float(counts.sum())
    entropy = float(-(probs * np.log(probs)).sum() / math.log(GRID_SIZE * GRID_SIZE))
    return occupied, occupancy, entropy


def _pca_minor_major_ratio(points: np.ndarray) -> float:
    pts = np.asarray(points, dtype=float)
    if len(pts) < 3:
        return 0.0
    centered = pts - pts.mean(axis=0, keepdims=True)
    cov = centered.T @ centered / max(1, len(pts) - 1)
    vals = np.linalg.eigvalsh(cov)
    vals = np.sort(np.maximum(vals, 0.0))
    if vals[-1] <= 1e-12:
        return 0.0
    return float(vals[0] / vals[-1])


def _wrap_deg(value: np.ndarray | float) -> np.ndarray:
    v = np.asarray(value, dtype=float)
    return (v + 180.0) % 360.0 - 180.0


def _orientation_stats(q_angles: np.ndarray, t_angles: np.ndarray) -> dict[str, float]:
    if len(q_angles) == 0:
        return {
            "orientation_delta_mean_deg": math.nan,
            "orientation_resultant": math.nan,
            "orientation_mad_deg": math.nan,
        }
    delta = _wrap_deg(np.asarray(t_angles) - np.asarray(q_angles))
    radians = np.deg2rad(delta)
    mean_vec = np.mean(np.exp(1j * radians))
    resultant = float(abs(mean_vec))
    mean_deg = float(np.rad2deg(np.angle(mean_vec)))
    mad = float(np.median(np.abs(_wrap_deg(delta - mean_deg))))
    return {
        "orientation_delta_mean_deg": mean_deg,
        "orientation_resultant": resultant,
        "orientation_mad_deg": mad,
    }


def _scale_stats(q_sizes: np.ndarray, t_sizes: np.ndarray) -> dict[str, float]:
    q = np.asarray(q_sizes, dtype=float)
    t = np.asarray(t_sizes, dtype=float)
    mask = (q > 0) & (t > 0) & np.isfinite(q) & np.isfinite(t)
    if not mask.any():
        return {
            "log_scale_ratio_median": math.nan,
            "log_scale_ratio_mad": math.nan,
        }
    ratios = np.log(t[mask] / q[mask])
    med = float(np.median(ratios))
    return {
        "log_scale_ratio_median": med,
        "log_scale_ratio_mad": float(np.median(np.abs(ratios - med))),
    }


def _project_points(H: np.ndarray, points: np.ndarray) -> np.ndarray:
    return cv2.perspectiveTransform(
        np.asarray(points, dtype=np.float32).reshape(-1, 1, 2),
        np.asarray(H, dtype=float),
    ).reshape(-1, 2).astype(float)


def _jacobian_metrics(H: np.ndarray, center_xy: tuple[float, float]) -> dict[str, Any]:
    Hn = np.asarray(H, dtype=float).copy()
    if abs(Hn[2, 2]) > 1e-12:
        Hn /= Hn[2, 2]
    x, y = map(float, center_xy)
    den = float(Hn[2, 0] * x + Hn[2, 1] * y + Hn[2, 2])
    if not math.isfinite(den) or abs(den) <= 1e-12:
        return {
            "jacobian_det": math.nan,
            "jacobian_s1": math.nan,
            "jacobian_s2": math.nan,
            "jacobian_anisotropy": math.nan,
            "jacobian_rotation_deg": math.nan,
            "jacobian_reflection": False,
        }
    xp = float((Hn[0, 0] * x + Hn[0, 1] * y + Hn[0, 2]) / den)
    yp = float((Hn[1, 0] * x + Hn[1, 1] * y + Hn[1, 2]) / den)
    J = np.array([
        [(Hn[0, 0] - Hn[2, 0] * xp) / den,
         (Hn[0, 1] - Hn[2, 1] * xp) / den],
        [(Hn[1, 0] - Hn[2, 0] * yp) / den,
         (Hn[1, 1] - Hn[2, 1] * yp) / den],
    ], dtype=float)
    if not np.isfinite(J).all():
        return {
            "jacobian_det": math.nan,
            "jacobian_s1": math.nan,
            "jacobian_s2": math.nan,
            "jacobian_anisotropy": math.nan,
            "jacobian_rotation_deg": math.nan,
            "jacobian_reflection": False,
        }
    U, singular, Vt = np.linalg.svd(J)
    s1 = float(max(singular))
    s2 = float(min(singular))
    R = U @ Vt
    rotation = float(math.degrees(math.atan2(R[1, 0], R[0, 0])))
    det = float(np.linalg.det(J))
    return {
        "jacobian_det": det,
        "jacobian_s1": s1,
        "jacobian_s2": s2,
        "jacobian_anisotropy": float(s1 / s2) if s2 > 1e-12 else math.inf,
        "jacobian_rotation_deg": rotation,
        "jacobian_reflection": bool(det < 0),
    }


def _denominator_and_footprint(
    H: np.ndarray,
    query_hw: tuple[int, int],
    tile_hw: tuple[int, int],
) -> dict[str, Any]:
    qh, qw = query_hw
    th, tw = tile_hw
    Hn = np.asarray(H, dtype=float).copy()
    if abs(Hn[2, 2]) > 1e-12:
        Hn /= Hn[2, 2]

    sample = np.array([
        [0.0, 0.0],
        [float(qw - 1), 0.0],
        [float(qw - 1), float(qh - 1)],
        [0.0, float(qh - 1)],
        [float(qw) / 2.0, float(qh) / 2.0],
    ])
    den = Hn[2, 0] * sample[:, 0] + Hn[2, 1] * sample[:, 1] + Hn[2, 2]
    finite_den = den[np.isfinite(den)]
    if len(finite_den):
        absden = np.abs(finite_den)
        min_abs = float(absden.min())
        max_abs = float(absden.max())
        ratio = float(max_abs / min_abs) if min_abs > 1e-12 else math.inf
        signs = np.sign(finite_den[np.abs(finite_den) > 1e-12])
        sign_consistent = bool(len(signs) > 0 and np.all(signs == signs[0]))
    else:
        min_abs = max_abs = ratio = math.nan
        sign_consistent = False

    corners = sample[:4]
    try:
        projected = _project_points(Hn, corners)
    except cv2.error:
        projected = np.full((4, 2), np.nan)
    finite = bool(np.isfinite(projected).all())
    if not finite:
        return {
            "denominator_min_abs": min_abs,
            "denominator_max_abs": max_abs,
            "denominator_abs_ratio": ratio,
            "denominator_sign_consistent": sign_consistent,
            "footprint_finite": False,
            "footprint_convex": False,
            "footprint_area_fraction_tile": math.nan,
            "footprint_inside_corner_fraction": 0.0,
            "footprint_edge_ratio": math.nan,
        }

    contour = projected.astype(np.float32).reshape(-1, 1, 2)
    area = abs(float(cv2.contourArea(contour)))
    convex = bool(cv2.isContourConvex(contour))
    inside = (
        (projected[:, 0] >= 0)
        & (projected[:, 0] < float(tw))
        & (projected[:, 1] >= 0)
        & (projected[:, 1] < float(th))
    )
    edges = np.linalg.norm(np.roll(projected, -1, axis=0) - projected, axis=1)
    positive = edges[edges > 1e-12]
    edge_ratio = float(positive.max() / positive.min()) if len(positive) else math.nan
    return {
        "denominator_min_abs": min_abs,
        "denominator_max_abs": max_abs,
        "denominator_abs_ratio": ratio,
        "denominator_sign_consistent": sign_consistent,
        "footprint_finite": True,
        "footprint_convex": convex,
        "footprint_area_fraction_tile": float(area / float(max(1, th * tw))),
        "footprint_inside_corner_fraction": float(inside.mean()),
        "footprint_edge_ratio": edge_ratio,
    }


def analyze_pair(q: orb_subtile.FeaturePack, t: orb_subtile.FeaturePack) -> dict[str, Any]:
    """Compute baseline ORB homography and richer blind discrimination evidence."""
    result: dict[str, Any] = {
        "good_matches": 0,
        "inliers": 0,
        "inlier_ratio": 0.0,
        "homography_ok": False,
        "projected_tile_u_px": math.nan,
        "projected_tile_v_px": math.nan,
        "projective_denominator": math.nan,
        "query_bbox_coverage": 0.0,
        "tile_bbox_coverage": 0.0,
        "query_hull_coverage": 0.0,
        "tile_hull_coverage": 0.0,
        "query_grid_cells": 0,
        "query_grid_occupancy": 0.0,
        "query_grid_entropy": 0.0,
        "tile_grid_cells": 0,
        "tile_grid_occupancy": 0.0,
        "tile_grid_entropy": 0.0,
        "query_pca_minor_major_ratio": 0.0,
        "tile_pca_minor_major_ratio": 0.0,
        "good_match_distance_median": math.nan,
        "inlier_match_distance_median": math.nan,
        "lowe_ratio_actual_median": math.nan,
        "lowe_ratio_actual_p90": math.nan,
        "forward_error_median_px": math.nan,
        "forward_error_p90_px": math.nan,
        "forward_error_rmse_px": math.nan,
        "backward_error_median_px": math.nan,
        "backward_error_p90_px": math.nan,
        "backward_error_rmse_px": math.nan,
        "symmetric_error_median_px": math.nan,
        "symmetric_error_p90_px": math.nan,
        "symmetric_error_rmse_px": math.nan,
        "orientation_delta_mean_deg": math.nan,
        "orientation_resultant": math.nan,
        "orientation_mad_deg": math.nan,
        "log_scale_ratio_median": math.nan,
        "log_scale_ratio_mad": math.nan,
        "homography_condition_number": math.nan,
        "jacobian_det": math.nan,
        "jacobian_s1": math.nan,
        "jacobian_s2": math.nan,
        "jacobian_anisotropy": math.nan,
        "jacobian_rotation_deg": math.nan,
        "jacobian_reflection": False,
        "denominator_min_abs": math.nan,
        "denominator_max_abs": math.nan,
        "denominator_abs_ratio": math.nan,
        "denominator_sign_consistent": False,
        "footprint_finite": False,
        "footprint_convex": False,
        "footprint_area_fraction_tile": math.nan,
        "footprint_inside_corner_fraction": 0.0,
        "footprint_edge_ratio": math.nan,
        "error": None,
    }
    if not q.ok or not t.ok or q.descriptors is None or t.descriptors is None:
        result["error"] = q.error or t.error or "features missing"
        return result

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    try:
        knn = matcher.knnMatch(q.descriptors, t.descriptors, k=2)
    except cv2.error as exc:
        result["error"] = f"knnMatch failed: {exc}"
        return result

    accepted = []
    actual_ratios = []
    for pair in knn:
        if len(pair) < 2:
            continue
        m, n = pair
        ratio = float(m.distance / n.distance) if n.distance > 0 else 0.0
        if m.distance < orb_subtile.LOWE_RATIO * n.distance:
            accepted.append(m)
            actual_ratios.append(ratio)

    result["good_matches"] = int(len(accepted))
    if accepted:
        result["good_match_distance_median"] = float(
            np.median([m.distance for m in accepted])
        )
        result["lowe_ratio_actual_median"] = float(np.median(actual_ratios))
        result["lowe_ratio_actual_p90"] = float(np.quantile(actual_ratios, 0.90))
    if len(accepted) < 4:
        return result

    q_pts = np.float32([q.keypoints[m.queryIdx].pt for m in accepted])
    t_pts = np.float32([t.keypoints[m.trainIdx].pt for m in accepted])
    H, mask = cv2.findHomography(
        q_pts, t_pts, cv2.RANSAC, orb_subtile.RANSAC_THRESHOLD_PX
    )
    inlier_mask = (
        mask.ravel().astype(bool)
        if mask is not None else np.zeros(len(accepted), dtype=bool)
    )
    inliers = int(inlier_mask.sum())
    result["inliers"] = inliers
    result["inlier_ratio"] = float(inliers / max(1, len(accepted)))
    result["homography_ok"] = bool(
        H is not None and inliers >= orb_subtile.MIN_HOMOGRAPHY_INLIERS
    )
    if inliers:
        qi = q_pts[inlier_mask].astype(float)
        ti = t_pts[inlier_mask].astype(float)
        result["query_bbox_coverage"] = orb_subtile.bbox_coverage(qi, q.image_shape)
        result["tile_bbox_coverage"] = orb_subtile.bbox_coverage(ti, t.image_shape)
        result["query_hull_coverage"] = _convex_hull_coverage(qi, q.image_shape)
        result["tile_hull_coverage"] = _convex_hull_coverage(ti, t.image_shape)
        qcells, qocc, qent = _grid_support(qi, q.image_shape)
        tcells, tocc, tent = _grid_support(ti, t.image_shape)
        result.update({
            "query_grid_cells": qcells,
            "query_grid_occupancy": qocc,
            "query_grid_entropy": qent,
            "tile_grid_cells": tcells,
            "tile_grid_occupancy": tocc,
            "tile_grid_entropy": tent,
            "query_pca_minor_major_ratio": _pca_minor_major_ratio(qi),
            "tile_pca_minor_major_ratio": _pca_minor_major_ratio(ti),
            "inlier_match_distance_median": float(
                np.median([accepted[i].distance for i in np.flatnonzero(inlier_mask)])
            ),
        })
        qangles = np.array([q.keypoints[accepted[i].queryIdx].angle for i in np.flatnonzero(inlier_mask)])
        tangles = np.array([t.keypoints[accepted[i].trainIdx].angle for i in np.flatnonzero(inlier_mask)])
        result.update(_orientation_stats(qangles, tangles))
        qsizes = np.array([q.keypoints[accepted[i].queryIdx].size for i in np.flatnonzero(inlier_mask)])
        tsizes = np.array([t.keypoints[accepted[i].trainIdx].size for i in np.flatnonzero(inlier_mask)])
        result.update(_scale_stats(qsizes, tsizes))

    if not result["homography_ok"]:
        return result

    assert H is not None
    H = np.asarray(H, dtype=float)
    try:
        result["homography_condition_number"] = float(np.linalg.cond(H))
    except np.linalg.LinAlgError:
        result["homography_condition_number"] = math.inf

    qi = q_pts[inlier_mask].astype(float)
    ti = t_pts[inlier_mask].astype(float)
    forward = _project_points(H, qi)
    ferror = np.linalg.norm(forward - ti, axis=1)
    fstats = _finite_summary(ferror)
    result["forward_error_median_px"] = fstats["median"]
    result["forward_error_p90_px"] = fstats["p90"]
    result["forward_error_rmse_px"] = fstats["rmse"]

    try:
        Hinv = np.linalg.inv(H)
        backward = _project_points(Hinv, ti)
        berror = np.linalg.norm(backward - qi, axis=1)
    except (np.linalg.LinAlgError, cv2.error):
        berror = np.full(len(qi), np.nan)
    bstats = _finite_summary(berror)
    result["backward_error_median_px"] = bstats["median"]
    result["backward_error_p90_px"] = bstats["p90"]
    result["backward_error_rmse_px"] = bstats["rmse"]
    sym = np.sqrt(ferror * ferror + berror * berror)
    sstats = _finite_summary(sym)
    result["symmetric_error_median_px"] = sstats["median"]
    result["symmetric_error_p90_px"] = sstats["p90"]
    result["symmetric_error_rmse_px"] = sstats["rmse"]

    qh, qw = q.image_shape
    center = (float(qw) / 2.0, float(qh) / 2.0)
    point = np.float32([[[center[0], center[1]]]])
    projected = cv2.perspectiveTransform(point, H).reshape(2)
    result["projected_tile_u_px"] = float(projected[0])
    result["projected_tile_v_px"] = float(projected[1])
    result["projective_denominator"] = float(
        H[2, 0] * center[0] + H[2, 1] * center[1] + H[2, 2]
    )
    result.update(_jacobian_metrics(H, center))
    result.update(_denominator_and_footprint(H, q.image_shape, t.image_shape))
    return result
