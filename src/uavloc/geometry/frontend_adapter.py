"""Pluggable local-matcher contract for geometric-discrimination experiments.

The adapter boundary standardizes what a frontend must provide so ORB, XFeat,
LightGlue-based sparse matchers, and dense/semi-dense matchers can be compared
without changing downstream candidate-labeling or evaluation code.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class MatcherRuntime:
    feature_ms: float
    match_ms: float
    geometry_ms: float
    total_ms: float
    device: str
    backend: str


@dataclass
class CandidateGeometryMeasurement:
    frontend_name: str
    query_path: str
    tile_path: str
    good_matches: int
    inliers: int
    homography_ok: bool
    finite_projection: bool
    projected_tile_u_px: float
    projected_tile_v_px: float
    projected_easting: float
    projected_northing: float
    projected_inside_tile: bool
    evidence: dict[str, Any]
    runtime: MatcherRuntime


@runtime_checkable
class GeometricFrontend(Protocol):
    """Common frontend interface used by future GD comparative benchmarks."""

    name: str

    def describe(self) -> dict[str, Any]:
        """Return immutable algorithm/config metadata."""

    def measure_pair(
        self,
        query_path: Path,
        tile_path: Path,
        *,
        tile_bounds: tuple[float, float, float, float],
    ) -> CandidateGeometryMeasurement:
        """Measure one query↔tile pair without using reference/GT."""


def required_common_metrics() -> tuple[str, ...]:
    """Metrics every frontend benchmark must expose or mark unavailable."""
    return (
        "good_matches",
        "inliers",
        "homography_ok",
        "finite_projection",
        "projected_inside_tile",
        "projected_easting",
        "projected_northing",
        "feature_ms",
        "match_ms",
        "geometry_ms",
        "total_ms",
    )
