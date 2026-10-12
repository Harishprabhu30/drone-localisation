"""Blind locality-persistence primitives for RG2.4.

These utilities operate only on timestamped consensus coordinates. They do not
use reference positions, retrieval labels, or trajectory ground truth.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LocalityStep:
    index: int
    locality_index: int
    dt_s: float | None
    jump_m: float | None
    contiguous_time: bool
    linked_to_previous: bool


def euclidean(a, b) -> float:
    ax, ay = float(a[0]), float(a[1])
    bx, by = float(b[0]), float(b[1])
    return float(math.hypot(bx - ax, by - ay))


def max_pairwise_span(points) -> float:
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) == 0:
        raise ValueError("Expected a non-empty Nx2 point array")
    if not np.isfinite(pts).all():
        raise ValueError("Locality points must be finite")
    if len(pts) == 1:
        return 0.0
    diff = pts[:, None, :] - pts[None, :, :]
    dist = np.linalg.norm(diff, axis=2)
    return float(dist.max())


def segment_localities(
    timestamps_s,
    points,
    *,
    link_threshold_m: float,
    contiguous_gap_multiplier: float,
) -> tuple[list[LocalityStep], float, float]:
    """Segment a track using frozen spatial and temporal continuity rules.

    Returns steps, nominal_dt_s, and contiguous_max_dt_s.
    """
    t = np.asarray(timestamps_s, dtype=float)
    pts = np.asarray(points, dtype=float)
    if len(t) != len(pts) or len(t) == 0:
        raise ValueError("Timestamp/point length mismatch or empty track")
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("Expected Nx2 points")
    if not np.isfinite(t).all() or not np.isfinite(pts).all():
        raise ValueError("Timestamps and points must be finite")
    if len(t) > 1 and not np.all(np.diff(t) > 0):
        raise ValueError("Timestamps must be strictly increasing")
    threshold = float(link_threshold_m)
    if not math.isfinite(threshold) or threshold <= 0:
        raise ValueError("link_threshold_m must be finite and positive")
    multiplier = float(contiguous_gap_multiplier)
    if not math.isfinite(multiplier) or multiplier <= 0:
        raise ValueError("contiguous_gap_multiplier must be positive")

    if len(t) == 1:
        nominal_dt = 0.0
        max_dt = 0.0
        return [LocalityStep(0, 1, None, None, False, False)], nominal_dt, max_dt

    diffs = np.diff(t)
    nominal_dt = float(np.median(diffs))
    if nominal_dt <= 0:
        raise ValueError("Non-positive nominal sampling interval")
    max_dt = float(nominal_dt * multiplier)

    steps = [LocalityStep(0, 1, None, None, False, False)]
    locality = 1
    for i in range(1, len(t)):
        dt = float(t[i] - t[i - 1])
        jump = euclidean(pts[i - 1], pts[i])
        contiguous = bool(dt > 0 and dt <= max_dt)
        linked = bool(contiguous and jump <= threshold)
        if not linked:
            locality += 1
        steps.append(
            LocalityStep(
                index=i,
                locality_index=locality,
                dt_s=dt,
                jump_m=jump,
                contiguous_time=contiguous,
                linked_to_previous=linked,
            )
        )
    return steps, nominal_dt, max_dt


def path_length(points) -> float:
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) == 0:
        raise ValueError("Expected a non-empty Nx2 point array")
    if not np.isfinite(pts).all():
        raise ValueError("Points must be finite")
    if len(pts) == 1:
        return 0.0
    return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())


def support_redundancy_ratio(
    support_sets: list[set[str]],
) -> tuple[int, int, float]:
    """Return distinct windows, total observations, repeated-observation ratio.

    Ratio = 1 - distinct/observations. It is zero when every support-window
    observation is unique across the locality run and approaches one as the
    same windows are repeatedly observed.
    """
    if not support_sets:
        raise ValueError("support_sets must be non-empty")
    total = int(sum(len(s) for s in support_sets))
    if total <= 0:
        raise ValueError("Each locality requires supporting windows")
    distinct = int(len(set().union(*support_sets)))
    ratio = float(1.0 - (distinct / float(total)))
    return distinct, total, ratio
