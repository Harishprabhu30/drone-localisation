"""Official VERLab XFeat adapter for absolute UAV-to-map matching experiments."""
from __future__ import annotations

import hashlib
import importlib
import math
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from uavloc.geometry.frontend_adapter import CandidateGeometryMeasurement, MatcherRuntime


OFFICIAL_REPOSITORY = "https://github.com/verlab/accelerated_features.git"
PINNED_COMMIT = "e92685f57f8318b18725c5c8c0bd28c7fe188d9a"

RESIZE_LONG = 960
TOP_K = 1200
DETECTION_THRESHOLD = 0.05
MIN_COSSIM = 0.82
RANSAC_THRESHOLD_PX = 5.0
MIN_HOMOGRAPHY_INLIERS = 4


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def git_value(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def resize_long_side(image: np.ndarray, target: int = RESIZE_LONG) -> np.ndarray:
    h, w = image.shape[:2]
    long_side = max(h, w)
    if target <= 0 or long_side == target:
        return image
    scale = target / float(long_side)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(image, (nw, nh), interpolation=interp)


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


class XFeatAbsoluteFrontend:
    name = "xfeat_sparse_absolute_v1"

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
        dirty = git_value(self.repo, "status", "--porcelain", "--untracked-files=no")
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
            detection_threshold=DETECTION_THRESHOLD,
        )
        self.model.dev = self.device
        self.model.net = self.model.net.to(self.device).eval()
        self.model.eval()
        self._cache: dict[str, tuple[dict[str, Any], float]] = {}

    def _sync(self) -> None:
        if str(self.device).startswith("cuda") and self.torch.cuda.is_available():
            self.torch.cuda.synchronize(self.device)

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "official_repository": OFFICIAL_REPOSITORY,
            "pinned_commit": PINNED_COMMIT,
            "resolved_commit": git_value(self.repo, "rev-parse", "HEAD"),
            "weights_sha256": sha256_file(self.weights_path),
            "device": str(self.device),
            "resize_long": RESIZE_LONG,
            "top_k": TOP_K,
            "detection_threshold": DETECTION_THRESHOLD,
            "min_cossim": MIN_COSSIM,
            "ransac_threshold_px": RANSAC_THRESHOLD_PX,
            "min_homography_inliers": MIN_HOMOGRAPHY_INLIERS,
            "geometry_model": "query_pixel_to_satellite_tile_homography",
        }

    def _extract(self, image_path: Path) -> tuple[dict[str, Any], float]:
        key = str(Path(image_path).resolve())
        if key in self._cache:
            return self._cache[key]

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
        output = self.model.detectAndCompute(
            tensor,
            top_k=TOP_K,
            detection_threshold=DETECTION_THRESHOLD,
        )[0]
        self._sync()
        feature_ms = (time.perf_counter() - start) * 1000.0

        feature = {
            "keypoints": output["keypoints"].detach().cpu(),
            "descriptors": output["descriptors"].detach().cpu(),
            "scores": output["scores"].detach().cpu(),
            "height": int(gray.shape[0]),
            "width": int(gray.shape[1]),
        }
        self._cache[key] = (feature, float(feature_ms))
        return self._cache[key]

    def measure_pair(
        self,
        query_path: Path,
        tile_path: Path,
        *,
        tile_bounds: tuple[float, float, float, float],
    ) -> CandidateGeometryMeasurement:
        total_start = time.perf_counter()
        q, q_ms = self._extract(query_path)
        t, t_ms = self._extract(tile_path)

        qdesc = q["descriptors"].to(self.device)
        tdesc = t["descriptors"].to(self.device)

        self._sync()
        match_start = time.perf_counter()
        if len(qdesc) == 0 or len(tdesc) == 0:
            idx_q = self.torch.empty(0, dtype=self.torch.long)
            idx_t = self.torch.empty(0, dtype=self.torch.long)
        else:
            idx_q, idx_t = self.model.match(qdesc, tdesc, min_cossim=MIN_COSSIM)
        self._sync()
        match_ms = (time.perf_counter() - match_start) * 1000.0

        qpts = (
            q["keypoints"][idx_q.detach().cpu()].numpy().astype(np.float32)
            if len(idx_q) else np.empty((0, 2), dtype=np.float32)
        )
        tpts = (
            t["keypoints"][idx_t.detach().cpu()].numpy().astype(np.float32)
            if len(idx_t) else np.empty((0, 2), dtype=np.float32)
        )

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
            if mask is not None else np.zeros(len(qpts), dtype=bool)
        )
        inliers = int(inlier_mask.sum())
        homography_ok = bool(H is not None and inliers >= MIN_HOMOGRAPHY_INLIERS)

        u = v = e = n = math.nan
        finite_projection = False
        inside = False
        if homography_ok:
            center = np.float32([[[q["width"] / 2.0, q["height"] / 2.0]]])
            projected = cv2.perspectiveTransform(center, np.asarray(H, dtype=float)).reshape(2)
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

        inlier_ratio = float(inliers / max(1, len(qpts)))
        evidence = {
            "query_keypoints": int(len(q["keypoints"])),
            "tile_keypoints": int(len(t["keypoints"])),
            "matches": int(len(qpts)),
            "inliers": inliers,
            "inlier_ratio": inlier_ratio,
            "projected_tile_u_px": float(u),
            "projected_tile_v_px": float(v),
        }

        total_ms = (time.perf_counter() - total_start) * 1000.0
        runtime = MatcherRuntime(
            feature_ms=float(q_ms + t_ms),
            match_ms=float(match_ms),
            geometry_ms=float(geometry_ms),
            total_ms=float(total_ms),
            device=str(self.device),
            backend="official_verlab_xfeat",
        )
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
