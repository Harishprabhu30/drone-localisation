"""Pinned SIFT + LightGlue adapter for absolute UAV-to-map matching."""
from __future__ import annotations

import hashlib
import importlib
import importlib.util
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from uavloc.geometry.frontend_adapter import CandidateGeometryMeasurement, MatcherRuntime
from uavloc.geometry.lightglue_absolute_frontend import (
    OFFICIAL_REPOSITORY,
    PINNED_COMMIT,
    _tensor_tree_to_cpu,
    _tensor_tree_to_device,
    git_blob_sha1,
    git_value,
    sha256_file,
)
from uavloc.geometry.top20_geometric_evidence import (
    _denominator_and_footprint,
    _finite_summary,
    _grid_support,
    _jacobian_metrics,
    _pca_minor_major_ratio,
    _project_points,
)
from uavloc.geometry.xfeat_absolute_frontend import tile_pixel_to_map

FEATURE_TYPE = "sift"
SIFT_BACKEND = "opencv"
ROOTSIFT = True
MAX_KEYPOINTS = 4096
RESIZE_LONG = 1024
NMS_RADIUS = 0
DETECTION_THRESHOLD = 0.0066667
EDGE_THRESHOLD = 10
NUM_OCTAVES = 4
RANSAC_THRESHOLD_PX = 5.0
MIN_HOMOGRAPHY_INLIERS = 4
LIGHTGLUE_FILTER_THRESHOLD = 0.1
LIGHTGLUE_DEPTH_CONFIDENCE = 0.95
LIGHTGLUE_WIDTH_CONFIDENCE = 0.99
LIGHTGLUE_WEIGHT_FILE = "sift_lightglue_v0-1_arxiv.pth"

PINNED_SIFT_BLOB = "802fc1c2eb9ee852691e0e4dd67455f822f8405f"
PINNED_LIGHTGLUE_BLOB = "485833515eb0a80866b3d078d3e43d39ddcd84ae"


class SIFTLightGlueAbsoluteFrontend:
    name = "sift_lightglue_absolute_v1"

    def __init__(self, lightglue_repo: Path, *, device_name: str = "cpu") -> None:
        requested = Path(lightglue_repo).expanduser()
        self.repo: Path | None = None
        self.resolved_git_commit: str | None = None
        if requested.exists() and (requested / "lightglue").is_dir():
            self.repo = requested.resolve()
            head = git_value(self.repo, "rev-parse", "HEAD")
            if head != PINNED_COMMIT:
                raise RuntimeError(f"LightGlue checkout HEAD {head} != pinned {PINNED_COMMIT}")
            dirty = git_value(self.repo, "status", "--porcelain", "--untracked-files=no")
            if dirty:
                raise RuntimeError("Pinned LightGlue tracked files are modified")
            self.source_mode = "git_checkout"
            self.source_root = self.repo / "lightglue"
            self.resolved_git_commit = head
            if str(self.repo) not in sys.path:
                sys.path.insert(0, str(self.repo))
        else:
            spec = importlib.util.find_spec("lightglue")
            locations = list(spec.submodule_search_locations or []) if spec else []
            if not locations:
                raise RuntimeError("Pinned LightGlue checkout/package not available")
            self.source_mode = "installed_package"
            self.source_root = Path(locations[0]).resolve()

        observed_sift = git_blob_sha1(self.source_root / "sift.py")
        observed_lg = git_blob_sha1(self.source_root / "lightglue.py")
        if observed_sift != PINNED_SIFT_BLOB:
            raise RuntimeError("Pinned LightGlue sift.py blob mismatch")
        if observed_lg != PINNED_LIGHTGLUE_BLOB:
            raise RuntimeError("Pinned LightGlue lightglue.py blob mismatch")

        for module_name in list(sys.modules):
            if module_name == "lightglue" or module_name.startswith("lightglue."):
                del sys.modules[module_name]

        self.torch = importlib.import_module("torch")
        mod = importlib.import_module("lightglue")
        utils = importlib.import_module("lightglue.utils")
        SIFT = getattr(mod, "SIFT")
        LightGlue = getattr(mod, "LightGlue")
        self.load_image = getattr(utils, "load_image")

        if device_name == "cuda":
            if not self.torch.cuda.is_available():
                raise RuntimeError("CUDA requested but unavailable")
            self.device = self.torch.device("cuda")
        elif device_name == "mps":
            if not (hasattr(self.torch.backends, "mps") and self.torch.backends.mps.is_available()):
                raise RuntimeError("MPS requested but unavailable")
            self.device = self.torch.device("mps")
        elif device_name == "auto":
            self.device = self.torch.device("cuda" if self.torch.cuda.is_available() else "cpu")
        else:
            self.device = self.torch.device("cpu")

        self.extractor = SIFT(
            backend=SIFT_BACKEND,
            rootsift=ROOTSIFT,
            max_num_keypoints=MAX_KEYPOINTS,
            nms_radius=NMS_RADIUS,
            detection_threshold=DETECTION_THRESHOLD,
            edge_threshold=EDGE_THRESHOLD,
            num_octaves=NUM_OCTAVES,
        ).eval().to(self.device)
        self.matcher = LightGlue(
            features=FEATURE_TYPE,
            filter_threshold=LIGHTGLUE_FILTER_THRESHOLD,
            depth_confidence=LIGHTGLUE_DEPTH_CONFIDENCE,
            width_confidence=LIGHTGLUE_WIDTH_CONFIDENCE,
        ).eval().to(self.device)

        checkpoints = Path(self.torch.hub.get_dir()) / "checkpoints"
        self.lightglue_weights_path = checkpoints / LIGHTGLUE_WEIGHT_FILE
        if not self.lightglue_weights_path.is_file():
            raise FileNotFoundError(self.lightglue_weights_path)

        self._cache: dict[str, tuple[dict[str, Any], tuple[int, int], float]] = {}

    def _sync(self) -> None:
        if str(self.device).startswith("cuda") and self.torch.cuda.is_available():
            self.torch.cuda.synchronize(self.device)

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "official_repository": OFFICIAL_REPOSITORY,
            "pinned_commit": PINNED_COMMIT,
            "source_mode": self.source_mode,
            "source_root": str(self.source_root),
            "resolved_git_commit": self.resolved_git_commit,
            "device": str(self.device),
            "feature_type": FEATURE_TYPE,
            "sift_backend": SIFT_BACKEND,
            "rootsift": ROOTSIFT,
            "max_keypoints": MAX_KEYPOINTS,
            "resize_long": RESIZE_LONG,
            "nms_radius": NMS_RADIUS,
            "detection_threshold": DETECTION_THRESHOLD,
            "edge_threshold": EDGE_THRESHOLD,
            "num_octaves": NUM_OCTAVES,
            "lightglue_filter_threshold": LIGHTGLUE_FILTER_THRESHOLD,
            "lightglue_depth_confidence": LIGHTGLUE_DEPTH_CONFIDENCE,
            "lightglue_width_confidence": LIGHTGLUE_WIDTH_CONFIDENCE,
            "ransac_threshold_px": RANSAC_THRESHOLD_PX,
            "min_homography_inliers": MIN_HOMOGRAPHY_INLIERS,
            "sift_source_blob": git_blob_sha1(self.source_root / "sift.py"),
            "lightglue_source_blob": git_blob_sha1(self.source_root / "lightglue.py"),
            "lightglue_weights_sha256": sha256_file(self.lightglue_weights_path),
            "uses_scale_orientation_positional_encoding": True,
        }

    def _extract(self, image_path: Path) -> tuple[dict[str, Any], tuple[int, int], float, bool]:
        key = str(Path(image_path).resolve())
        if key in self._cache:
            feats, hw, _ = self._cache[key]
            return feats, hw, 0.0, True
        bgr = cv2.imread(key, cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"OpenCV could not read {key}")
        hw = (int(bgr.shape[0]), int(bgr.shape[1]))
        image = self.load_image(key).to(self.device)
        self._sync()
        start = time.perf_counter()
        feats = self.extractor.extract(image, resize=RESIZE_LONG)
        self._sync()
        ms = (time.perf_counter() - start) * 1000.0
        if "scales" not in feats or "oris" not in feats:
            raise RuntimeError("Pinned SIFT extractor did not expose scales/oris")
        cached = _tensor_tree_to_cpu(feats)
        self._cache[key] = (cached, hw, float(ms))
        return cached, hw, float(ms), False

    def measure_pair(
        self,
        query_path: Path,
        tile_path: Path,
        *,
        tile_bounds: tuple[float, float, float, float],
    ) -> CandidateGeometryMeasurement:
        total_start = time.perf_counter()
        q_cpu, q_hw, q_ms, q_hit = self._extract(query_path)
        t_cpu, t_hw, t_ms, t_hit = self._extract(tile_path)
        q = _tensor_tree_to_device(q_cpu, self.device)
        t = _tensor_tree_to_device(t_cpu, self.device)

        self._sync()
        match_start = time.perf_counter()
        output = self.matcher({"image0": q, "image1": t})
        self._sync()
        match_ms = (time.perf_counter() - match_start) * 1000.0

        matches = output["matches"][0].detach().cpu()
        scores = output["scores"][0].detach().cpu().numpy().astype(float)
        qk = q_cpu["keypoints"][0].numpy().astype(np.float32)
        tk = t_cpu["keypoints"][0].numpy().astype(np.float32)
        if len(matches):
            mi = matches.numpy().astype(int)
            qpts, tpts = qk[mi[:, 0]], tk[mi[:, 1]]
        else:
            qpts = np.empty((0, 2), dtype=np.float32)
            tpts = np.empty((0, 2), dtype=np.float32)

        geometry_start = time.perf_counter()
        H = mask = None
        if len(qpts) >= 4:
            H, mask = cv2.findHomography(qpts, tpts, cv2.RANSAC, RANSAC_THRESHOLD_PX)
        geometry_ms = (time.perf_counter() - geometry_start) * 1000.0
        inlier_mask = mask.ravel().astype(bool) if mask is not None else np.zeros(len(qpts), dtype=bool)
        inliers = int(inlier_mask.sum())
        homography_ok = bool(H is not None and inliers >= MIN_HOMOGRAPHY_INLIERS)

        u = v = e = n = math.nan
        finite_projection = inside = False
        if homography_ok:
            center = np.float32([[[q_hw[1] / 2.0, q_hw[0] / 2.0]]])
            projected = cv2.perspectiveTransform(center, np.asarray(H, dtype=float)).reshape(2)
            u, v = float(projected[0]), float(projected[1])
            e, n = tile_pixel_to_map(u, v, t_hw, tile_bounds)
            finite_projection = bool(math.isfinite(e) and math.isfinite(n))
            inside = bool(finite_projection and 0 <= u < t_hw[1] and 0 <= v < t_hw[0])

        evidence: dict[str, Any] = {
            "query_feature_cache_hit": bool(q_hit),
            "tile_feature_cache_hit": bool(t_hit),
            "query_keypoints": int(len(qk)),
            "tile_keypoints": int(len(tk)),
            "query_scales_present": "scales" in q_cpu,
            "query_orientations_present": "oris" in q_cpu,
            "tile_scales_present": "scales" in t_cpu,
            "tile_orientations_present": "oris" in t_cpu,
            "matches": int(len(qpts)),
            "match_score_sum": float(np.sum(scores)) if len(scores) else 0.0,
            "match_score_mean": float(np.mean(scores)) if len(scores) else 0.0,
            "match_score_median": float(np.median(scores)) if len(scores) else 0.0,
            "inliers": inliers,
            "inlier_ratio": float(inliers / max(1, len(qpts))),
            "projected_tile_u_px": float(u),
            "projected_tile_v_px": float(v),
            "lightglue_stop_layer": int(output["stop"]),
            "symmetric_error_median_px": math.nan,
            "jacobian_anisotropy": math.nan,
            "footprint_area_fraction_tile": math.nan,
        }
        if homography_ok:
            Hn = np.asarray(H, dtype=float)
            qi, ti = qpts[inlier_mask].astype(float), tpts[inlier_mask].astype(float)
            forward = _project_points(Hn, qi)
            ferr = np.linalg.norm(forward - ti, axis=1)
            try:
                backward = _project_points(np.linalg.inv(Hn), ti)
                berr = np.linalg.norm(backward - qi, axis=1)
            except (np.linalg.LinAlgError, cv2.error):
                berr = np.full(len(qi), np.nan)
            sym = np.sqrt(ferr * ferr + berr * berr)
            evidence["symmetric_error_median_px"] = _finite_summary(sym)["median"]
            center_xy = (float(q_hw[1]) / 2.0, float(q_hw[0]) / 2.0)
            evidence.update(_jacobian_metrics(Hn, center_xy))
            evidence.update(_denominator_and_footprint(Hn, q_hw, t_hw))

        total_ms = (time.perf_counter() - total_start) * 1000.0
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
            runtime=MatcherRuntime(
                feature_ms=float(q_ms + t_ms),
                match_ms=float(match_ms),
                geometry_ms=float(geometry_ms),
                total_ms=float(total_ms),
                device=str(self.device),
                backend="official_cvg_sift_lightglue",
            ),
        )
