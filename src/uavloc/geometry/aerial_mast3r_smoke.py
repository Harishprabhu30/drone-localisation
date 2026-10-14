"""Utilities for the frozen CV3 aerial-MASt3R smoke protocol."""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Mast3RImageGeometry:
    original_width: int
    original_height: int
    resized_width: int
    resized_height: int
    crop_left: int
    crop_top: int
    processed_width: int
    processed_height: int

    @property
    def scale_x(self) -> float:
        return self.resized_width / float(self.original_width)

    @property
    def scale_y(self) -> float:
        return self.resized_height / float(self.original_height)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def mast3r_image_geometry(
    original_width: int,
    original_height: int,
    *,
    size: int = 512,
    square_ok: bool = False,
) -> Mast3RImageGeometry:
    """Mirror pinned DUSt3R load_images resize/crop geometry without image data."""
    if original_width <= 0 or original_height <= 0:
        raise ValueError("Image dimensions must be positive")
    long_edge = max(original_width, original_height)
    resized_width = int(round(original_width * size / float(long_edge)))
    resized_height = int(round(original_height * size / float(long_edge)))

    cx, cy = resized_width // 2, resized_height // 2
    halfw = ((2 * cx) // 16) * 8
    halfh = ((2 * cy) // 16) * 8
    if not square_ok and resized_width == resized_height:
        halfh = int(3 * halfw / 4)

    crop_left = int(cx - halfw)
    crop_top = int(cy - halfh)
    processed_width = int(2 * halfw)
    processed_height = int(2 * halfh)
    if processed_width <= 0 or processed_height <= 0:
        raise RuntimeError("Invalid MASt3R processed dimensions")

    return Mast3RImageGeometry(
        original_width=int(original_width),
        original_height=int(original_height),
        resized_width=int(resized_width),
        resized_height=int(resized_height),
        crop_left=crop_left,
        crop_top=crop_top,
        processed_width=processed_width,
        processed_height=processed_height,
    )


def processed_to_canonical_xy(
    xy: np.ndarray,
    geometry: Mast3RImageGeometry,
) -> np.ndarray:
    """Map processed MASt3R pixel coordinates back to original image coordinates.

    This follows the project's OpenCV/keypoint convention: coordinates scale by
    image-size ratios and the centered crop offset is restored before scaling.
    """
    pts = np.asarray(xy, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("Expected Nx2 coordinates")
    out = pts.copy()
    out[:, 0] = (out[:, 0] + geometry.crop_left) / geometry.scale_x
    out[:, 1] = (out[:, 1] + geometry.crop_top) / geometry.scale_y
    return out.astype(np.float32)


def tile_pixel_to_map(
    u: float,
    v: float,
    image_hw: tuple[int, int],
    bounds: tuple[float, float, float, float],
) -> tuple[float, float]:
    h, w = image_hw
    left, right, bottom, top = map(float, bounds)
    e = left + ((float(u) + 0.5) / float(w)) * (right - left)
    n = top - ((float(v) + 0.5) / float(h)) * (top - bottom)
    return float(e), float(n)


def finite_pair(e: float, n: float) -> bool:
    return bool(math.isfinite(e) and math.isfinite(n))


def json_safe(value):
    """Convert non-finite/numpy scalar values into strict JSON-compatible values."""
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value
