"""Pinned XFeat* semi-dense adapter for absolute UAV-to-map matching."""
from __future__ import annotations

import importlib
import math
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from uavloc.geometry.frontend_adapter import CandidateGeometryMeasurement, MatcherRuntime
from uavloc.geometry.xfeat_absolute_frontend import (
    MIN_HOMOGRAPHY_INLIERS,
    PINNED_COMMIT,
    RANSAC_THRESHOLD_PX,
    RESIZE_LONG,
    TOP_K,
    git_value,
    resize_long_side,
    sha256_file,
    tile_pixel_to_map,
)


FINE_CONFIDENCE = 0.25
MULTISCALE = True


class XFeatStarAbsoluteFrontend:
    """Semi-dense XFeat* frontend using detectAndComputeDense + refine_matches."""

    name = "xfeat_star_semidense_absolute_v1"

    def __init__(self, xfeat_repo: Path, *, device_name: str = "cpu") -> None:
        self.repo = Path(xfeat_repo).resolve()
        self.module_path = self.repo / "modules" / "xfeat.py"
        self.weights_path = self.repo / "weights" / "xfeat.pt"
        if not self.module_path.is_file():
            raise FileNotFoundError(self.module_path)
        if not self.weights_path.is_file():
            raise FileNotFoundError(self.weights_path)

        head = git_value(self.repo, "rev-parse", "HEAD")
        if head != PINNED_COMMIT:
            raise RuntimeError(
                f"XFeat checkout HEAD {head} != pinned {PINNED_COMMIT}"
            )
        dirty = git_value(
            self.repo, "status", "--porcelain", "--untracked-files=no"
        )
        if dirty:
            raise RuntimeError("Pinned XFeat tracked files are modified")

        if str(self.repo) not in sys.path:
            sys.path.insert(0, str(self.repo))
        for module_name in list(sys.modules):
            if module_name == "modules" or module_name.startswith("modules."):
                del sys.modules[module_name]

        self.torch = importlib.import_module("torch")
        XFeat = getattr(importlib.import_module("modules.xfeat"), "XFeat")

        if device_name == "auto":
            if self.torch.cuda.is_available():
                self.device = self.torch.device("cuda")
            elif (
                hasattr(self.torch.backends, "mps")
                and self.torch.backends.mps.is_available()
            ):
                self.device = self.torch.device("mps")
            else:
                self.device = self.torch.device("cpu")
        elif device_name == "cuda":
            if not self.torch.cuda.is_available():
                raise RuntimeError("CUDA requested but unavailable")
            self.device = self.torch.device("cuda")
        elif device_name == "mps":
            if not (
                hasattr(self.torch.backends, "mps")
                and self.torch.backends.mps.is_available()
            ):
                raise RuntimeError("MPS requested but unavailable")
            self.device = self.torch.device("mps")
        else:
            self.device = self.torch.device("cpu")

        self.model = XFeat(
            weights=str(self.weights_path),
            top_k=TOP_K,
            detection_threshold=0.05,
        )
        self.model.dev = self.device
        self.model.net = self.model.net.to(self.device).eval()
        self.model.eval()

        # Cache dense feature dictionaries on CPU. Pair-specific matching/refinement
        # still runs on the selected device.
        self._cache: dict[str, tuple[dict[str, Any], float]] = {}

    def _sync(self) -> None:
        if str(self.device).startswith("cuda") and self.torch.cuda.is_available():
            self.torch.cuda.synchronize(self.device)

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "pinned_commit": PINNED_COMMIT,
            "resolved_commit": git_value(self.repo, "rev-parse", "HEAD"),
            "weights_sha256": sha256_file(self.weights_path),
            "device": str(self.device),
            "resize_long": RESIZE_LONG,
            "top_k": TOP_K,
            "multiscale": MULTISCALE,
            "fine_confidence": FINE_CONFIDENCE,
            "ransac_threshold_px": RANSAC_THRESHOLD_PX,
            "min_homography_inliers": MIN_HOMOGRAPHY_INLIERS,
            "matching_api": "detectAndComputeDense+batch_match+refine_matches",
            "geometry_model": "query_pixel_to_satellite_tile_homography",
        }

    def _extract_dense(
        self,
        image_path: Path,
    ) -> tuple[dict[str, Any], float, bool]:
        key = str(Path(image_path).resolve())
        if key in self._cache:
            feature, _original_ms = self._cache[key]
            return feature, 0.0, True

        gray = cv2.imread(key, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise RuntimeError(f"OpenCV could not read {key}")
        gray = resize_long_side(gray)
        tensor = (
            self.torch.from_numpy(gray)
            .to(device=self.device, dtype=self.torch.float32)
            .unsqueeze(0)
            .unsqueeze(0)
            / 255.0
        )

        self._sync()
        start = time.perf_counter()
        dense = self.model.detectAndComputeDense(
            tensor,
            top_k=TOP_K,
            multiscale=MULTISCALE,
        )
        self._sync()
        feature_ms = (time.perf_counter() - start) * 1000.0

        feature = {
            "keypoints": dense["keypoints"].detach().cpu(),
            "descriptors": dense["descriptors"].detach().cpu(),
            "scales": dense["scales"].detach().cpu(),
            "height": int(gray.shape[0]),
            "width": int(gray.shape[1]),
        }
        self._cache[key] = (feature, float(feature_ms))
        return feature, float(feature_ms), False

    def _device_dense(self, feature: dict[str, Any]) -> dict[str, Any]:
        return {
            "keypoints": feature["keypoints"].to(self.device),
            "descriptors": feature["descriptors"].to(self.device),
            "scales": feature["scales"].to(self.device),
        }

    def measure_pair(
        self,
        query_path: Path,
        tile_path: Path,
        *,
        tile_bounds: tuple[float, float, float, float],
    ) -> CandidateGeometryMeasurement:
        total_start = time.perf_counter()
        q, q_ms, q_cache_hit = self._extract_dense(query_path)
        t, t_ms, t_cache_hit = self._extract_dense(tile_path)

        qd = self._device_dense(q)
        td = self._device_dense(t)

        self._sync()
        match_start = time.perf_counter()
        matches = self.model.batch_match(
            qd["descriptors"],
            td["descriptors"],
        )
        refined = self.model.refine_matches(
            qd,
            td,
            matches=matches,
            batch_idx=0,
            fine_conf=FINE_CONFIDENCE,
        )
        self._sync()
        match_ms = (time.perf_counter() - match_start) * 1000.0

        if len(refined):
            refined_np = refined.detach().cpu().numpy().astype(np.float32)
            qpts = refined_np[:, :2]
            tpts = refined_np[:, 2:]
        else:
            qpts = np.empty((0, 2), dtype=np.float32)
            tpts = np.empty((0, 2), dtype=np.float32)

        geometry_start = time.perf_counter()
        H = None
        mask = None
        if len(qpts) >= 4:
            H, mask = cv2.findHomography(
                qpts,
                tpts,
                cv2.RANSAC,
                RANSAC_THRESHOLD_PX,
            )
        geometry_ms = (time.perf_counter() - geometry_start) * 1000.0

        inlier_mask = (
            mask.ravel().astype(bool)
            if mask is not None
            else np.zeros(len(qpts), dtype=bool)
        )
        inliers = int(inlier_mask.sum())
        homography_ok = bool(
            H is not None and inliers >= MIN_HOMOGRAPHY_INLIERS
        )

        u = v = e = n = math.nan
        finite_projection = False
        inside = False
        if homography_ok:
            center = np.float32(
                [[[q["width"] / 2.0, q["height"] / 2.0]]]
            )
            projected = cv2.perspectiveTransform(
                center,
                np.asarray(H, dtype=float),
            ).reshape(2)
            u, v = float(projected[0]), float(projected[1])
            e, n = tile_pixel_to_map(
                u,
                v,
                (int(t["height"]), int(t["width"])),
                tile_bounds,
            )
            finite_projection = bool(math.isfinite(e) and math.isfinite(n))
            inside = bool(
                finite_projection
                and 0.0 <= u < float(t["width"])
                and 0.0 <= v < float(t["height"])
            )

        total_ms = (time.perf_counter() - total_start) * 1000.0
        runtime = MatcherRuntime(
            feature_ms=float(q_ms + t_ms),
            match_ms=float(match_ms),
            geometry_ms=float(geometry_ms),
            total_ms=float(total_ms),
            device=str(self.device),
            backend="official_verlab_xfeat_star",
        )
        evidence = {
            "query_feature_cache_hit": bool(q_cache_hit),
            "tile_feature_cache_hit": bool(t_cache_hit),
            "query_dense_features": int(q["keypoints"].shape[1]),
            "tile_dense_features": int(t["keypoints"].shape[1]),
            "matches": int(len(qpts)),
            "inliers": inliers,
            "inlier_ratio": float(inliers / max(1, len(qpts))),
            "projected_tile_u_px": float(u),
            "projected_tile_v_px": float(v),
        }

        return CandidateGeometryMeasurement(
            frontend_name=self.name,
            query_path=str(query_path),
            tile_path=str(tile_path),
            good_matches=int(len(qpts)),
            inliers=inliers,
            homography_ok=homography_ok,
            finite_projection=finite_projection,
            projected_tile_u_px=float(u),
            projected_tile_v_px=float(v),
            projected_easting=float(e),
            projected_northing=float(n),
            projected_inside_tile=inside,
            evidence=evidence,
            runtime=runtime,
        )
