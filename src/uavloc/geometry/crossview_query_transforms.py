"""Reference-free query-image transforms for VILLOC CV1 diagnostics.

This module wraps the frozen ORB verifier preprocessing without modifying the
historical `orb_subtile` contract.  CV1 transforms are applied only after the
frozen verifier preprocessing step so the historical 1024-long-edge downsize
cannot cancel the intended scale perturbation.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from uavloc.geometry import orb_subtile


VALID_ROTATIONS = {0, 90, 180, 270}


def read_frozen_processed_query(path: Path) -> np.ndarray:
    """Return the exact grayscale image produced by the frozen ORB preprocessor."""
    gray, error = orb_subtile.read_image_for_verifier(path)
    if gray is None:
        raise RuntimeError(error or f"Failed to preprocess query image: {path}")
    return gray


def scale_processed_query(gray: np.ndarray, factor: float) -> np.ndarray:
    """Scale a processed query while preserving its full image content/FOV."""
    if factor <= 0:
        raise ValueError(f"Scale factor must be positive, got {factor}")
    if abs(float(factor) - 1.0) <= 1e-12:
        return gray.copy()

    h, w = gray.shape[:2]
    new_w = max(1, int(round(float(w) * float(factor))))
    new_h = max(1, int(round(float(h) * float(factor))))
    interpolation = cv2.INTER_AREA if factor < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(gray, (new_w, new_h), interpolation=interpolation)


def rotate_processed_query(gray: np.ndarray, degrees: int) -> np.ndarray:
    """Apply an exact quarter-turn with no interpolation."""
    degrees = int(degrees) % 360
    if degrees not in VALID_ROTATIONS:
        raise ValueError(
            f"CV1 rotation must be one of {sorted(VALID_ROTATIONS)}, got {degrees}"
        )
    if degrees == 0:
        return gray.copy()
    if degrees == 90:
        return cv2.rotate(gray, cv2.ROTATE_90_CLOCKWISE)
    if degrees == 180:
        return cv2.rotate(gray, cv2.ROTATE_180)
    return cv2.rotate(gray, cv2.ROTATE_90_COUNTERCLOCKWISE)


def feature_pack_from_gray(
    gray: np.ndarray,
    detector,
) -> orb_subtile.FeaturePack:
    """Build a frozen-contract FeaturePack from an already processed image."""
    keypoints, descriptors = detector.detectAndCompute(gray, None)
    if descriptors is None or len(keypoints) == 0:
        return orb_subtile.FeaturePack(
            False,
            gray.shape[:2],
            keypoints or [],
            None,
            "no descriptors",
        )
    return orb_subtile.FeaturePack(
        True,
        gray.shape[:2],
        keypoints,
        descriptors,
        None,
    )
