"""Pinned SuperPoint + LightGlue adapter for absolute UAV-to-map matching."""
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
from uavloc.geometry.top20_geometric_evidence import (
    _convex_hull_coverage,
    _denominator_and_footprint,
    _finite_summary,
    _grid_support,
    _jacobian_metrics,
    _pca_minor_major_ratio,
    _project_points,
)
from uavloc.geometry.xfeat_absolute_frontend import tile_pixel_to_map


OFFICIAL_REPOSITORY = "https://github.com/cvg/LightGlue.git"
PINNED_COMMIT = "eb42fee2d71449efb0aa5c10549752b5d75384d8"

FEATURE_TYPE = "superpoint"
MAX_KEYPOINTS = 780
RESIZE_LONG = 1024
RANSAC_THRESHOLD_PX = 5.0
MIN_HOMOGRAPHY_INLIERS = 4

LIGHTGLUE_FILTER_THRESHOLD = 0.1
LIGHTGLUE_DEPTH_CONFIDENCE = 0.95
LIGHTGLUE_WIDTH_CONFIDENCE = 0.99

SUPERPOINT_WEIGHT_FILE = "superpoint_v1.pth"
LIGHTGLUE_WEIGHT_FILE = "superpoint_lightglue_v0-1_arxiv.pth"

PINNED_CORE_BLOBS = {
    "__init__.py": "b84d285cf2a29e3b17c8c2c052a45f856dcf89c0",
    "superpoint.py": "d6d380eb5737b92ce417e3ed0e5db1ee9d4a1d04",
    "lightglue.py": "485833515eb0a80866b3d078d3e43d39ddcd84ae",
    "utils.py": "d1c1ab2e94716b1c54191a6ed5d01023036836c1",
}


def git_blob_sha1(path: Path) -> str:
    data = path.read_bytes()
    header = f"blob {len(data)}\0".encode("utf-8")
    return hashlib.sha1(header + data).hexdigest()


def verify_core_source(package_dir: Path) -> dict[str, str]:
    observed: dict[str, str] = {}
    mismatches: list[str] = []
    for filename, expected in PINNED_CORE_BLOBS.items():
        path = package_dir / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = git_blob_sha1(path)
        observed[filename] = actual
        if actual != expected:
            mismatches.append(
                f"{filename}: expected {expected}, observed {actual}"
            )
    if mismatches:
        raise RuntimeError(
            "Installed LightGlue core source does not match pinned commit "
            f"{PINNED_COMMIT}: " + "; ".join(mismatches)
        )
    return observed


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


def _tensor_tree_to_device(value: Any, device: Any) -> Any:
    if hasattr(value, "to") and callable(value.to):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _tensor_tree_to_device(item, device) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        converted = [_tensor_tree_to_device(item, device) for item in value]
        return type(value)(converted)
    return value


def _tensor_tree_to_cpu(value: Any) -> Any:
    if hasattr(value, "detach") and callable(value.detach):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _tensor_tree_to_cpu(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        converted = [_tensor_tree_to_cpu(item) for item in value]
        return type(value)(converted)
    return value


class SuperPointLightGlueAbsoluteFrontend:
    """Official SuperPoint extractor + LightGlue matcher under a pinned checkout."""

    name = "superpoint_lightglue_absolute_v1"

    def __init__(
        self,
        lightglue_repo: Path,
        *,
        device_name: str = "cpu",
        max_keypoints: int | None = MAX_KEYPOINTS,
        resize_long: int = RESIZE_LONG,
        filter_threshold: float = LIGHTGLUE_FILTER_THRESHOLD,
        depth_confidence: float = LIGHTGLUE_DEPTH_CONFIDENCE,
        width_confidence: float = LIGHTGLUE_WIDTH_CONFIDENCE,
    ) -> None:
        requested_repo = Path(lightglue_repo).expanduser()
        self.repo: Path | None = None
        self.source_mode: str
        self.source_root: Path
        self.resolved_git_commit: str | None = None

        if requested_repo.exists() and (requested_repo / "lightglue").is_dir():
            self.repo = requested_repo.resolve()
            head = git_value(self.repo, "rev-parse", "HEAD")
            if head != PINNED_COMMIT:
                raise RuntimeError(
                    f"LightGlue checkout HEAD {head} != pinned {PINNED_COMMIT}"
                )
            dirty = git_value(
                self.repo,
                "status",
                "--porcelain",
                "--untracked-files=no",
            )
            if dirty:
                raise RuntimeError("Pinned LightGlue tracked files are modified")
            self.source_mode = "git_checkout"
            self.source_root = self.repo / "lightglue"
            self.resolved_git_commit = head
            if str(self.repo) not in sys.path:
                sys.path.insert(0, str(self.repo))
        else:
            spec = importlib.util.find_spec("lightglue")
            locations = (
                list(spec.submodule_search_locations or [])
                if spec is not None
                else []
            )
            if not locations:
                raise RuntimeError(
                    "LightGlue is neither vendored at "
                    f"{requested_repo} nor importable as a Python package. "
                    "Install the pinned official LightGlue package or provide "
                    "third_party/LightGlue."
                )
            self.source_mode = "installed_package"
            self.source_root = Path(locations[0]).resolve()

        self.verified_core_blobs = verify_core_source(self.source_root)

        for module_name in list(sys.modules):
            if module_name == "lightglue" or module_name.startswith("lightglue."):
                del sys.modules[module_name]

        self.torch = importlib.import_module("torch")
        lightglue_module = importlib.import_module("lightglue")
        utils_module = importlib.import_module("lightglue.utils")
        SuperPoint = getattr(lightglue_module, "SuperPoint")
        LightGlue = getattr(lightglue_module, "LightGlue")
        self.load_image = getattr(utils_module, "load_image")

        if device_name == "auto":
            if self.torch.cuda.is_available():
                self.device = self.torch.device("cuda")
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

        if max_keypoints is not None and int(max_keypoints) <= 0:
            raise ValueError("max_keypoints must be positive or None")
        if int(resize_long) <= 0:
            raise ValueError("resize_long must be positive")
        self.max_keypoints = (
            None if max_keypoints is None else int(max_keypoints)
        )
        self.resize_long = int(resize_long)
        self.filter_threshold = float(filter_threshold)
        self.depth_confidence = float(depth_confidence)
        self.width_confidence = float(width_confidence)

        self.extractor = (
            SuperPoint(max_num_keypoints=self.max_keypoints)
            .eval()
            .to(self.device)
        )
        self.matcher = (
            LightGlue(
                features=FEATURE_TYPE,
                filter_threshold=self.filter_threshold,
                depth_confidence=self.depth_confidence,
                width_confidence=self.width_confidence,
            )
            .eval()
            .to(self.device)
        )

        checkpoints = Path(self.torch.hub.get_dir()) / "checkpoints"
        self.superpoint_weights_path = checkpoints / SUPERPOINT_WEIGHT_FILE
        self.lightglue_weights_path = checkpoints / LIGHTGLUE_WEIGHT_FILE
        if not self.superpoint_weights_path.is_file():
            raise FileNotFoundError(self.superpoint_weights_path)
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
            "verified_core_blobs": dict(self.verified_core_blobs),
            "device": str(self.device),
            "feature_type": FEATURE_TYPE,
            "max_keypoints": self.max_keypoints,
            "resize_long": self.resize_long,
            "lightglue_filter_threshold": self.filter_threshold,
            "lightglue_depth_confidence": self.depth_confidence,
            "lightglue_width_confidence": self.width_confidence,
            "ransac_threshold_px": RANSAC_THRESHOLD_PX,
            "min_homography_inliers": MIN_HOMOGRAPHY_INLIERS,
            "superpoint_weights_sha256": sha256_file(self.superpoint_weights_path),
            "lightglue_weights_sha256": sha256_file(self.lightglue_weights_path),
            "geometry_model": "query_pixel_to_satellite_tile_homography",
        }

    def _extract(
        self,
        image_path: Path,
    ) -> tuple[dict[str, Any], tuple[int, int], float, bool]:
        key = str(Path(image_path).resolve())
        if key in self._cache:
            features, image_hw, _original_ms = self._cache[key]
            return features, image_hw, 0.0, True

        image_bgr = cv2.imread(key, cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise RuntimeError(f"OpenCV could not read {key}")
        image_hw = (int(image_bgr.shape[0]), int(image_bgr.shape[1]))

        image = self.load_image(key).to(self.device)
        self._sync()
        start = time.perf_counter()
        features = self.extractor.extract(image, resize=self.resize_long)
        self._sync()
        feature_ms = (time.perf_counter() - start) * 1000.0

        cached = _tensor_tree_to_cpu(features)
        self._cache[key] = (cached, image_hw, float(feature_ms))
        return cached, image_hw, float(feature_ms), False

    def measure_pair(
        self,
        query_path: Path,
        tile_path: Path,
        *,
        tile_bounds: tuple[float, float, float, float],
    ) -> CandidateGeometryMeasurement:
        total_start = time.perf_counter()
        q_cpu, q_hw, q_ms, q_cache_hit = self._extract(query_path)
        t_cpu, t_hw, t_ms, t_cache_hit = self._extract(tile_path)

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
            qpts = qk[mi[:, 0]]
            tpts = tk[mi[:, 1]]
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
                [[[q_hw[1] / 2.0, q_hw[0] / 2.0]]]
            )
            projected = cv2.perspectiveTransform(
                center,
                np.asarray(H, dtype=float),
            ).reshape(2)
            u, v = float(projected[0]), float(projected[1])
            e, n = tile_pixel_to_map(
                u,
                v,
                t_hw,
                tile_bounds,
            )
            finite_projection = bool(math.isfinite(e) and math.isfinite(n))
            inside = bool(
                finite_projection
                and 0.0 <= u < float(t_hw[1])
                and 0.0 <= v < float(t_hw[0])
            )

        evidence: dict[str, Any] = {
            "query_feature_cache_hit": bool(q_cache_hit),
            "tile_feature_cache_hit": bool(t_cache_hit),
            "query_keypoints": int(len(qk)),
            "tile_keypoints": int(len(tk)),
            "matches": int(len(qpts)),
            "match_score_sum": float(np.sum(scores)) if len(scores) else 0.0,
            "match_score_mean": float(np.mean(scores)) if len(scores) else 0.0,
            "match_score_median": float(np.median(scores)) if len(scores) else 0.0,
            "inliers": inliers,
            "inlier_ratio": float(inliers / max(1, len(qpts))),
            "projected_tile_u_px": float(u),
            "projected_tile_v_px": float(v),
            "query_hull_coverage": 0.0,
            "tile_hull_coverage": 0.0,
            "query_grid_occupancy": 0.0,
            "tile_grid_occupancy": 0.0,
            "query_grid_entropy": 0.0,
            "tile_grid_entropy": 0.0,
            "query_pca_minor_major_ratio": 0.0,
            "tile_pca_minor_major_ratio": 0.0,
            "forward_error_median_px": math.nan,
            "forward_error_p90_px": math.nan,
            "forward_error_rmse_px": math.nan,
            "backward_error_median_px": math.nan,
            "backward_error_p90_px": math.nan,
            "backward_error_rmse_px": math.nan,
            "symmetric_error_median_px": math.nan,
            "symmetric_error_p90_px": math.nan,
            "symmetric_error_rmse_px": math.nan,
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
            "lightglue_stop_layer": int(output["stop"]),
        }

        if inliers:
            qi = qpts[inlier_mask].astype(float)
            ti = tpts[inlier_mask].astype(float)
            evidence["query_hull_coverage"] = _convex_hull_coverage(qi, q_hw)
            evidence["tile_hull_coverage"] = _convex_hull_coverage(ti, t_hw)
            _, qocc, qent = _grid_support(qi, q_hw)
            _, tocc, tent = _grid_support(ti, t_hw)
            evidence["query_grid_occupancy"] = float(qocc)
            evidence["tile_grid_occupancy"] = float(tocc)
            evidence["query_grid_entropy"] = float(qent)
            evidence["tile_grid_entropy"] = float(tent)
            evidence["query_pca_minor_major_ratio"] = _pca_minor_major_ratio(qi)
            evidence["tile_pca_minor_major_ratio"] = _pca_minor_major_ratio(ti)

        if homography_ok:
            Hn = np.asarray(H, dtype=float)
            qi = qpts[inlier_mask].astype(float)
            ti = tpts[inlier_mask].astype(float)
            forward = _project_points(Hn, qi)
            ferr = np.linalg.norm(forward - ti, axis=1)
            fstats = _finite_summary(ferr)
            evidence["forward_error_median_px"] = fstats["median"]
            evidence["forward_error_p90_px"] = fstats["p90"]
            evidence["forward_error_rmse_px"] = fstats["rmse"]
            try:
                Hinv = np.linalg.inv(Hn)
                backward = _project_points(Hinv, ti)
                berr = np.linalg.norm(backward - qi, axis=1)
            except (np.linalg.LinAlgError, cv2.error):
                berr = np.full(len(qi), np.nan)
            bstats = _finite_summary(berr)
            evidence["backward_error_median_px"] = bstats["median"]
            evidence["backward_error_p90_px"] = bstats["p90"]
            evidence["backward_error_rmse_px"] = bstats["rmse"]
            sym = np.sqrt(ferr * ferr + berr * berr)
            sstats = _finite_summary(sym)
            evidence["symmetric_error_median_px"] = sstats["median"]
            evidence["symmetric_error_p90_px"] = sstats["p90"]
            evidence["symmetric_error_rmse_px"] = sstats["rmse"]
            try:
                evidence["homography_condition_number"] = float(np.linalg.cond(Hn))
            except np.linalg.LinAlgError:
                evidence["homography_condition_number"] = math.inf
            center_xy = (float(q_hw[1]) / 2.0, float(q_hw[0]) / 2.0)
            evidence.update(_jacobian_metrics(Hn, center_xy))
            evidence.update(_denominator_and_footprint(Hn, q_hw, t_hw))

        total_ms = (time.perf_counter() - total_start) * 1000.0
        runtime = MatcherRuntime(
            feature_ms=float(q_ms + t_ms),
            match_ms=float(match_ms),
            geometry_ms=float(geometry_ms),
            total_ms=float(total_ms),
            device=str(self.device),
            backend="official_cvg_superpoint_lightglue",
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
