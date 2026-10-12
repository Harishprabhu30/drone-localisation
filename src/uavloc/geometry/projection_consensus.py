"""Deterministic 2-D consensus primitives for frozen map projections.

No ground truth or map reference enters these functions. They operate only on
already-produced global-coordinate points and optional blind weights.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np


def _points(value) -> np.ndarray:
    pts = np.asarray(value, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) == 0:
        raise ValueError("Expected a non-empty Nx2 point array")
    if not np.isfinite(pts).all():
        raise ValueError("Consensus points must be finite")
    return pts


def _weights(value, n: int) -> np.ndarray:
    if value is None:
        return np.ones(n, dtype=float)
    w = np.asarray(value, dtype=float)
    if w.shape != (n,) or not np.isfinite(w).all() or (w < 0).any():
        raise ValueError("Weights must be finite non-negative length-N values")
    if float(w.sum()) <= 0:
        raise ValueError("At least one weight must be positive")
    return w


def coordinate_median(points) -> tuple[float, float]:
    pts = _points(points)
    out = np.median(pts, axis=0)
    return float(out[0]), float(out[1])


def geometric_median(
    points,
    *,
    weights=None,
    tolerance_m: float = 1e-7,
    max_iterations: int = 512,
) -> tuple[float, float]:
    """Weighted Weiszfeld geometric median with deterministic coincident handling."""
    pts = _points(points)
    w = _weights(weights, len(pts))
    if len(pts) == 1:
        return float(pts[0, 0]), float(pts[0, 1])

    x = np.average(pts, axis=0, weights=w)
    eps = max(float(tolerance_m), 1e-12)

    for _ in range(int(max_iterations)):
        d = np.linalg.norm(pts - x, axis=1)
        coincident = d <= eps
        if coincident.any():
            # A coincident input can be the optimum. Use the weighted subgradient
            # criterion; otherwise continue with a regularized Weiszfeld update.
            wc = float(w[coincident].sum())
            non = ~coincident
            if not non.any():
                return float(x[0]), float(x[1])
            unit_sum = np.sum(
                (w[non] / d[non])[:, None] * (pts[non] - x),
                axis=0,
            )
            if float(np.linalg.norm(unit_sum)) <= wc:
                return float(x[0]), float(x[1])
            safe_d = np.maximum(d, eps)
        else:
            safe_d = d

        effective = w / safe_d
        new_x = np.sum(effective[:, None] * pts, axis=0) / float(effective.sum())
        if float(np.linalg.norm(new_x - x)) <= eps:
            x = new_x
            break
        x = new_x

    return float(x[0]), float(x[1])


def medoid(points) -> tuple[int, float, float]:
    """Point minimizing total Euclidean distance; first index wins exact ties."""
    pts = _points(points)
    distance = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
    totals = distance.sum(axis=1)
    index = int(np.argmin(totals))
    return index, float(pts[index, 0]), float(pts[index, 1])


def inlier_huber_centroid(
    points,
    *,
    inlier_weights,
    huber_k: float = 1.345,
    tolerance_m: float = 1e-7,
    max_iterations: int = 256,
) -> tuple[float, float]:
    """Inlier-weighted robust centroid using deterministic Huber IRLS.

    Robust scale is frozen from the median radius around the coordinate median.
    This is a blind, data-derived scale; no reference position is involved.
    """
    pts = _points(points)
    base = _weights(inlier_weights, len(pts))
    if huber_k <= 0:
        raise ValueError("huber_k must be positive")
    if len(pts) == 1:
        return float(pts[0, 0]), float(pts[0, 1])

    origin = np.asarray(coordinate_median(pts), dtype=float)
    radius = np.linalg.norm(pts - origin, axis=1)
    positive = radius[radius > 0]
    if len(positive) == 0:
        return float(origin[0]), float(origin[1])

    scale = float(np.median(positive))
    delta = max(float(huber_k) * scale, 1e-9)
    x = np.average(pts, axis=0, weights=base)
    eps = max(float(tolerance_m), 1e-12)

    for _ in range(int(max_iterations)):
        d = np.linalg.norm(pts - x, axis=1)
        robust = np.ones(len(pts), dtype=float)
        mask = d > delta
        robust[mask] = delta / d[mask]
        effective = base * robust
        if float(effective.sum()) <= 0:
            raise RuntimeError("Huber effective weights collapsed to zero")
        new_x = np.sum(effective[:, None] * pts, axis=0) / float(effective.sum())
        if float(np.linalg.norm(new_x - x)) <= eps:
            x = new_x
            break
        x = new_x

    return float(x[0]), float(x[1])


def dominant_radius_cluster_medoid(
    points,
    *,
    radius_m: float,
) -> tuple[int, list[int], float, float]:
    """Largest fixed-radius support set, represented by its medoid.

    For every observed point, form the set of points within radius_m of that
    point. Select the support set with largest cardinality; break ties by lower
    total within-set distance and then lower center index. Return the medoid of
    the selected support. No reference information is involved.
    """
    pts = _points(points)
    radius = float(radius_m)
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError("radius_m must be finite and positive")
    distance = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)

    best_key = None
    best_support = None
    for i in range(len(pts)):
        support = np.flatnonzero(distance[i] <= radius + 1e-12)
        total = float(distance[np.ix_(support, support)].sum())
        key = (-len(support), total, i)
        if best_key is None or key < best_key:
            best_key = key
            best_support = support

    assert best_support is not None
    local = pts[best_support]
    local_distance = np.linalg.norm(
        local[:, None, :] - local[None, :, :], axis=2
    )
    medoid_local = int(np.argmin(local_distance.sum(axis=1)))
    medoid_global = int(best_support[medoid_local])
    return (
        medoid_global,
        [int(i) for i in best_support.tolist()],
        float(pts[medoid_global, 0]),
        float(pts[medoid_global, 1]),
    )


def dispersion(points) -> dict[str, Any]:
    pts = _points(points)
    gx, gy = geometric_median(pts)
    center = np.array([gx, gy], dtype=float)
    radial = np.linalg.norm(pts - center, axis=1)

    if len(pts) == 1:
        pairwise = np.array([0.0], dtype=float)
    else:
        matrix = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
        pairwise = matrix[np.triu_indices(len(pts), k=1)]

    span_x = float(pts[:, 0].max() - pts[:, 0].min())
    span_y = float(pts[:, 1].max() - pts[:, 1].min())
    return {
        "point_count": int(len(pts)),
        "radial_median_m": float(np.median(radial)),
        "radial_p90_m": float(np.quantile(radial, 0.90)),
        "radial_max_m": float(radial.max()),
        "pairwise_median_m": float(np.median(pairwise)),
        "pairwise_p90_m": float(np.quantile(pairwise, 0.90)),
        "pairwise_max_m": float(pairwise.max()),
        "bbox_span_easting_m": span_x,
        "bbox_span_northing_m": span_y,
        "bbox_diagonal_m": float(math.hypot(span_x, span_y)),
    }
