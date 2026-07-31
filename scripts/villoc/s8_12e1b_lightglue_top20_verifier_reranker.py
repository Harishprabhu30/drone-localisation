#!/usr/bin/env python3
"""
S8.12E.1B — LightGlue Top-K verifier / reranker for Villoc DINO retrieval candidates.

Purpose
-------
DINOv2 retrieval already provides a strong Top-K candidate pool for Villoc 45° and 90°.
This script verifies/reranks the Top-K candidates using LightGlue/SuperPoint image evidence.
Reference coordinates are used only after reranking for evaluation metrics.

This is separate from S8.12E.1A ORB verifier because LightGlue has different runtime,
feature caching, model dependencies, and scoring behavior.

Typical use
-----------
1. Smoke test first:

python scripts/villoc/s8_12e1b_lightglue_top20_verifier_reranker.py \
  --config configs/dataset_villoc_45deg.yaml \
  --variant 1024_s512 \
  --tag dinov2_vits14_img224_center_square_avgpatch_cpu \
  --top-n 20 \
  --policy hybrid \
  --limit-queries 5 \
  --device auto \
  --max-keypoints 1024 \
  --resize-long 768 \
  2>&1 | tee outputs/villoc/45_deg/logs/s8_12e1_top20_verifier/s8_12e1b_45deg_lightglue_smoke5.log

2. Full 45° run

cd /Users/harishprabhu/Documents/drone-localisation
source .drone_venv/bin/activate
export PYTHONPATH=$PWD/src

mkdir -p outputs/villoc/45_deg/logs/s8_12e1_top20_verifier

python scripts/villoc/s8_12e1b_lightglue_top20_verifier_reranker.py \
  --config configs/dataset_villoc_45deg.yaml \
  --variant 1024_s512 \
  --tag dinov2_vits14_img224_center_square_avgpatch_cpu \
  --top-n 20 \
  --policy hybrid \
  --device auto \
  --max-keypoints 1024 \
  --resize-long 768 \
  --out-root outputs/villoc/45_deg/reports/s8_12e1b_lightglue_top20_verifier_reranker/1024_s512_hybrid_full \
  2>&1 | tee outputs/villoc/45_deg/logs/s8_12e1_top20_verifier/s8_12e1b_45deg_lightglue_hybrid_full.log

  3. Full 90° run

mkdir -p outputs/villoc/90_deg/logs/s8_12e1_top20_verifier

python scripts/villoc/s8_12e1b_lightglue_top20_verifier_reranker.py \
  --config configs/dataset_villoc_90deg.yaml \
  --variant 1024_s512 \
  --tag dinov2_vits14_img224_center_square_avgpatch_cpu \
  --top-n 20 \
  --policy hybrid \
  --device auto \
  --max-keypoints 1024 \
  --resize-long 768 \
  --out-root outputs/villoc/90_deg/reports/s8_12e1b_lightglue_top20_verifier_reranker/1024_s512_hybrid_full \
  2>&1 | tee outputs/villoc/90_deg/logs/s8_12e1_top20_verifier/s8_12e1b_90deg_lightglue_hybrid_full.log

Design rules
------------
- Config-driven path resolution; no hardcoded dataset angle is required.
- Query latitude/longitude and tile easting/northing are evaluation-only.
- Candidate-pool oracle uses tile bbox containment when available.
- Center-error thresholds are localization-accuracy diagnostics, not candidate availability.
- Reranking score uses image evidence and optional original DINO rank prior only.
- The script prints resolved files/columns and fails loudly if schemas differ.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Missing dependency: pyyaml. Install with: pip install pyyaml") from exc

try:
    import cv2
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Missing dependency: opencv-python. Install with: pip install opencv-python") from exc

try:
    from pyproj import Transformer
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Missing dependency: pyproj. Install with: pip install pyproj") from exc

try:
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Missing dependency: matplotlib. Install with: pip install matplotlib") from exc

try:
    import torch
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Missing dependency: torch. Install PyTorch first.") from exc

try:
    from lightglue import LightGlue, SuperPoint
    from lightglue.utils import load_image, rbd
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependency: lightglue. Install/activate the environment that was used for S8.12A/S5 LightGlue. "
        "Example if using the LightGlue repo package: pip install -e third_party/LightGlue"
    ) from exc


# ----------------------------- column aliases -----------------------------

QUERY_ID_ALIASES = [
    "query_id", "token0_id", "token_id", "sample_id", "frame_id", "uav_token_id",
]
TILE_ID_ALIASES = ["tile_id", "candidate_tile_id", "sat_tile_id", "map_tile_id"]
RANK_ALIASES = ["rank", "retrieval_rank", "dino_rank", "candidate_rank"]
SCORE_ALIASES = ["score", "retrieval_score", "dino_score", "similarity", "cosine"]

QUERY_IMAGE_ALIASES = [
    "image_path", "frame_path", "query_image_path", "uav_image_path", "image_path_relative",
]
TILE_PATH_ALIASES = ["tile_path", "image_path", "sat_image_path", "map_tile_path", "path"]

LAT_ALIASES = ["latitude", "lat", "body_lat", "query_lat", "uav_lat"]
LON_ALIASES = ["longitude", "lon", "body_lon", "query_lon", "uav_lon"]
QUERY_EASTING_ALIASES = [
    "s8_12e1_query_easting_3346_m", "s8_12e0_query_easting_3346_m",
    "x_lks94_m", "x_3346_m", "easting_m", "proj_x_m", "x_projected_m", "x_map_m",
]
QUERY_NORTHING_ALIASES = [
    "s8_12e1_query_northing_3346_m", "s8_12e0_query_northing_3346_m",
    "y_lks94_m", "y_3346_m", "northing_m", "proj_y_m", "y_projected_m", "y_map_m",
]

TILE_CENTER_X_ALIASES = [
    "center_easting", "tile_center_easting", "center_easting_m", "tile_center_easting_m",
    "center_x_enu_m", "tile_center_x_enu_m", "center_east_enu_m", "center_easting_enu_m",
    "center_x_m", "tile_center_x_m", "x_center_m", "easting_m",
    "center_x_lks94_m", "center_x_3346_m", "x_3346_m", "x_lks94_m", "proj_center_x_m",
    "center_x", "x_center",
]
TILE_CENTER_Y_ALIASES = [
    "center_northing", "tile_center_northing", "center_northing_m", "tile_center_northing_m",
    "center_y_enu_m", "tile_center_y_enu_m", "center_north_enu_m", "center_northing_enu_m",
    "center_y_m", "tile_center_y_m", "y_center_m", "northing_m",
    "center_y_lks94_m", "center_y_3346_m", "y_3346_m", "y_lks94_m", "proj_center_y_m",
    "center_y", "y_center",
]

TILE_BBOX_XMIN_ALIASES = [
    "left_easting", "left_easting_m",
    "xmin_m", "x_min_m", "min_x_m", "left_m", "bbox_xmin_m", "x0_m", "xmin",
]
TILE_BBOX_XMAX_ALIASES = [
    "right_easting", "right_easting_m",
    "xmax_m", "x_max_m", "max_x_m", "right_m", "bbox_xmax_m", "x1_m", "xmax",
]
TILE_BBOX_YMIN_ALIASES = [
    "bottom_northing", "bottom_northing_m",
    "ymin_m", "y_min_m", "min_y_m", "bottom_m", "bbox_ymin_m", "y0_m", "ymin",
]
TILE_BBOX_YMAX_ALIASES = [
    "top_northing", "top_northing_m",
    "ymax_m", "y_max_m", "max_y_m", "top_m", "bbox_ymax_m", "y1_m", "ymax",
]


# ----------------------------- small utilities -----------------------------

def info(msg: str) -> None:
    print(msg, flush=True)


def die(msg: str) -> None:
    raise SystemExit(f"ERROR: {msg}")


def read_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        die(f"Config did not parse as a dictionary: {path}")
    return data


def get_nested(d: Dict[str, Any], keys: Sequence[str]) -> Optional[Any]:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def maybe_abs(path: Optional[str | Path], repo_root: Path) -> Optional[Path]:
    if path is None:
        return None
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = repo_root / p
    return p.resolve()


def first_existing(candidates: Iterable[Path], label: str) -> Path:
    seen: List[Path] = []
    for p in candidates:
        p = p.resolve()
        seen.append(p)
        if p.exists() and p.is_file():
            return p
    msg = f"Could not find {label}. First candidates:\n" + "\n".join(f"  {p}" for p in seen[:12])
    die(msg)


def find_col(df: pd.DataFrame, aliases: Sequence[str], *, required: bool, label: str) -> Optional[str]:
    lower = {c.lower(): c for c in df.columns}
    for a in aliases:
        if a.lower() in lower:
            return lower[a.lower()]
    if required:
        cols = "\n".join(f"  {c}" for c in df.columns)
        die(f"Missing required column for {label}. Tried aliases: {aliases}\nAvailable columns:\n{cols}")
    return None


def normalize_id_series(s: pd.Series) -> pd.Series:
    out = s.astype(str).str.strip()
    out = out.str.replace(r"\.0$", "", regex=True)
    return out


def resolve_output_root(config: Dict[str, Any], config_path: Path, repo_root: Path) -> Path:
    candidates: List[Any] = [
        config.get("output_root"),
        get_nested(config, ["paths", "output_root"]),
        get_nested(config, ["roots", "output_root"]),
        get_nested(config, ["dataset", "output_root"]),
    ]
    for c in candidates:
        p = maybe_abs(c, repo_root) if c else None
        if p:
            return p

    stem = config_path.stem.lower()
    if "45" in stem:
        return (repo_root / "outputs/villoc/45_deg").resolve()
    if "90" in stem:
        return (repo_root / "outputs/villoc/90_deg").resolve()
    return (repo_root / "outputs/villoc").resolve()


def resolve_map_output_root(config: Dict[str, Any], repo_root: Path, output_root: Path) -> Path:
    candidates: List[Any] = [
        get_nested(config, ["map", "output_root"]),
        get_nested(config, ["maps", "output_root"]),
        get_nested(config, ["map", "tile_output_root"]),
        get_nested(config, ["paths", "map_output_root"]),
        get_nested(config, ["roots", "map_output_root"]),
    ]
    for c in candidates:
        p = maybe_abs(c, repo_root) if c else None
        if p and p.exists():
            return p

    if "45_deg" in str(output_root) or "45deg" in str(output_root):
        p = repo_root / "outputs/villoc/90_deg"
        if p.exists():
            return p.resolve()
    return output_root


def resolve_query_csv(args: argparse.Namespace, output_root: Path, repo_root: Path) -> Path:
    if args.query_csv:
        return maybe_abs(args.query_csv, repo_root)  # type: ignore[return-value]
    return first_existing([
        output_root / "metadata/s8_5_uav_frames_index_v_1fps.csv",
        output_root / "metadata/s8_10b_canonical_uav_query_manifest.csv",
    ], "query CSV")


def resolve_topk_csv(args: argparse.Namespace, output_root: Path, repo_root: Path) -> Path:
    if args.topk_csv:
        return maybe_abs(args.topk_csv, repo_root)  # type: ignore[return-value]
    return first_existing([
        output_root / f"retrieval/s8_11d/s8_11d_topk_{args.variant}_{args.tag}.csv",
        output_root / f"retrieval/s8_11d_topk_{args.variant}_{args.tag}.csv",
        output_root / f"metadata/s8_11d_topk_{args.variant}_{args.tag}.csv",
    ], "DINO Top-K retrieval CSV")


def resolve_tile_index_csv(args: argparse.Namespace, map_output_root: Path, output_root: Path, repo_root: Path) -> Path:
    if args.tile_index_csv:
        return maybe_abs(args.tile_index_csv, repo_root)  # type: ignore[return-value]
    candidates = [
        map_output_root / f"metadata/s8_9_satellite_tile_index_{args.variant}.csv",
        map_output_root / f"metadata/s8_9_tile_index_{args.variant}.csv",
        output_root / f"metadata/s8_9_satellite_tile_index_{args.variant}.csv",
        output_root / f"metadata/s8_9_tile_index_{args.variant}.csv",
        repo_root / f"outputs/villoc/90_deg/metadata/s8_9_satellite_tile_index_{args.variant}.csv",
    ]
    return first_existing(candidates, "tile index CSV")


def resolve_image_path(value: Any, repo_root: Path, extra_roots: Sequence[Path]) -> Optional[Path]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    raw = str(value).strip()
    if raw == "" or raw.lower() == "nan":
        return None
    p = Path(raw).expanduser()
    if p.is_absolute() and p.exists():
        return p.resolve()
    candidates = [repo_root / p]
    for root in extra_roots:
        candidates.append(root / p)
    for c in candidates:
        if c.exists():
            return c.resolve()
    return (repo_root / p).resolve()


def add_projected_query_xy_from_latlon(query_df: pd.DataFrame) -> pd.DataFrame:
    lat_col = find_col(query_df, LAT_ALIASES, required=False, label="query latitude")
    lon_col = find_col(query_df, LON_ALIASES, required=False, label="query longitude")
    if not lat_col or not lon_col:
        return query_df

    out = query_df.copy()
    lat = pd.to_numeric(out[lat_col], errors="coerce")
    lon = pd.to_numeric(out[lon_col], errors="coerce")
    ok = lat.notna() & lon.notna()
    if ok.sum() == 0:
        return out

    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3346", always_xy=True)
    east = np.full(len(out), np.nan, dtype=float)
    north = np.full(len(out), np.nan, dtype=float)
    east_ok, north_ok = transformer.transform(
        lon.loc[ok].to_numpy(dtype=float), lat.loc[ok].to_numpy(dtype=float)
    )
    mask = ok.to_numpy()
    east[mask] = east_ok
    north[mask] = north_ok
    out["s8_12e1_query_easting_3346_m"] = east
    out["s8_12e1_query_northing_3346_m"] = north
    return out


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        # MPS can be unstable/slow for some matching ops; keep CPU as safe default on macOS.
        return torch.device("cpu")
    return torch.device(requested)


def configure_threads(num_threads: int) -> None:
    if num_threads and num_threads > 0:
        torch.set_num_threads(num_threads)
        try:
            cv2.setNumThreads(num_threads)
        except Exception:
            pass


# ----------------------------- LightGlue engine -----------------------------

@dataclass
class LGResult:
    num_matches: int
    score_sum: float
    score_mean: float
    inliers: int
    inlier_ratio: float
    query_inlier_coverage: float
    sat_inlier_coverage: float
    homography_ok: bool
    verifier_score: float
    runtime_s: float
    error: Optional[str] = None


def bbox_coverage(points: np.ndarray, image_hw: Tuple[int, int]) -> float:
    if points.shape[0] < 4:
        return 0.0
    h, w = image_hw
    if h <= 0 or w <= 0:
        return 0.0
    xs = points[:, 0]
    ys = points[:, 1]
    area = max(0.0, float(xs.max() - xs.min())) * max(0.0, float(ys.max() - ys.min()))
    return float(np.clip(area / float(w * h), 0.0, 1.0))


def get_image_hw(path: Path, resize_long: int) -> Tuple[int, int]:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return (0, 0)
    h, w = img.shape[:2]
    if resize_long and resize_long > 0:
        scale = float(resize_long) / float(max(h, w))
        if scale < 1.0:
            h = int(round(h * scale))
            w = int(round(w * scale))
    return (h, w)


class LightGlueVerifier:
    def __init__(self, device: torch.device, max_keypoints: int, resize_long: int):
        self.device = device
        self.max_keypoints = int(max_keypoints)
        self.resize_long = int(resize_long)
        self.extractor = SuperPoint(max_num_keypoints=self.max_keypoints).eval().to(self.device)
        self.matcher = LightGlue(features="superpoint").eval().to(self.device)
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.shape_cache: Dict[str, Tuple[int, int]] = {}

    @torch.inference_mode()
    def extract(self, path: Path) -> Dict[str, Any]:
        key = str(path)
        if key in self.cache:
            return self.cache[key]
        if not path.exists():
            raise FileNotFoundError(str(path))
        image = load_image(str(path)).to(self.device)
        kwargs: Dict[str, Any] = {}
        if self.resize_long and self.resize_long > 0:
            # LightGlue's extractor supports a resize kwarg in current versions.
            kwargs["resize"] = self.resize_long
        try:
            feats = self.extractor.extract(image, **kwargs)
        except TypeError:
            # Fallback for versions where resize is not accepted.
            feats = self.extractor.extract(image)
        self.cache[key] = feats
        self.shape_cache[key] = get_image_hw(path, self.resize_long)
        return feats

    @torch.inference_mode()
    def match(self, q_path: Path, s_path: Path, ransac_thresh: float) -> LGResult:
        t0 = time.perf_counter()
        try:
            feats0 = self.extract(q_path)
            feats1 = self.extract(s_path)
            out = self.matcher({"image0": feats0, "image1": feats1})
            f0, f1, m01 = [rbd(x) for x in [feats0, feats1, out]]
        except Exception as exc:
            return LGResult(0, 0.0, 0.0, 0, 0.0, 0.0, 0.0, False, 0.0, time.perf_counter() - t0, str(exc))

        matches = m01.get("matches", None)
        if matches is None:
            return LGResult(0, 0.0, 0.0, 0, 0.0, 0.0, 0.0, False, 0.0, time.perf_counter() - t0, "missing matches key")
        if hasattr(matches, "detach"):
            matches_np = matches.detach().cpu().numpy()
        else:
            matches_np = np.asarray(matches)
        if matches_np.ndim != 2 or matches_np.shape[1] != 2 or matches_np.shape[0] == 0:
            return LGResult(0, 0.0, 0.0, 0, 0.0, 0.0, 0.0, False, 0.0, time.perf_counter() - t0, None)

        kpts0 = f0["keypoints"]
        kpts1 = f1["keypoints"]
        if hasattr(kpts0, "detach"):
            kpts0_np = kpts0.detach().cpu().numpy()
        else:
            kpts0_np = np.asarray(kpts0)
        if hasattr(kpts1, "detach"):
            kpts1_np = kpts1.detach().cpu().numpy()
        else:
            kpts1_np = np.asarray(kpts1)

        pts0 = kpts0_np[matches_np[:, 0]]
        pts1 = kpts1_np[matches_np[:, 1]]
        num_matches = int(len(matches_np))

        scores = m01.get("scores", None)
        if scores is not None:
            if hasattr(scores, "detach"):
                scores_np = scores.detach().cpu().numpy()
            else:
                scores_np = np.asarray(scores)
            score_sum = float(np.nansum(scores_np)) if scores_np.size else 0.0
            score_mean = float(np.nanmean(scores_np)) if scores_np.size else 0.0
        else:
            # Some versions do not return per-match scores in rbd output.
            score_sum = float(num_matches)
            score_mean = 1.0 if num_matches else 0.0

        if num_matches >= 4:
            H, mask = cv2.findHomography(pts0.astype(np.float32), pts1.astype(np.float32), cv2.RANSAC, float(ransac_thresh))
            if mask is not None:
                inlier_mask = mask.ravel().astype(bool)
                inliers = int(inlier_mask.sum())
            else:
                inlier_mask = np.zeros(num_matches, dtype=bool)
                inliers = 0
            homography_ok = bool(H is not None and inliers >= 4)
        else:
            inlier_mask = np.zeros(num_matches, dtype=bool)
            inliers = 0
            homography_ok = False

        inlier_ratio = float(inliers / max(1, num_matches))
        q_hw = self.shape_cache.get(str(q_path), (0, 0))
        s_hw = self.shape_cache.get(str(s_path), (0, 0))
        q_cov = bbox_coverage(pts0[inlier_mask], q_hw)
        s_cov = bbox_coverage(pts1[inlier_mask], s_hw)

        # Learned sparse verifier score. Keep it simple and auditable.
        verifier_score = (
            float(inliers)
            + 0.02 * float(num_matches)
            + 5.0 * float(inlier_ratio)
            + 2.0 * float(math.sqrt(max(q_cov, 0.0)))
            + 1.5 * float(math.sqrt(max(s_cov, 0.0)))
            + 0.10 * float(score_sum)
        )
        return LGResult(
            num_matches=num_matches,
            score_sum=score_sum,
            score_mean=score_mean,
            inliers=inliers,
            inlier_ratio=inlier_ratio,
            query_inlier_coverage=q_cov,
            sat_inlier_coverage=s_cov,
            homography_ok=homography_ok,
            verifier_score=verifier_score,
            runtime_s=time.perf_counter() - t0,
            error=None,
        )


# ----------------------------- input/eval logic -----------------------------

def prepare_inputs(query_df: pd.DataFrame, topk_df: pd.DataFrame, tile_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    topk_query_col = find_col(topk_df, QUERY_ID_ALIASES, required=True, label="topk query id")
    topk_tile_col = find_col(topk_df, TILE_ID_ALIASES, required=True, label="topk tile id")
    topk_rank_col = find_col(topk_df, RANK_ALIASES, required=True, label="topk rank")
    topk_score_col = find_col(topk_df, SCORE_ALIASES, required=False, label="topk score")

    query_id_col = find_col(query_df, QUERY_ID_ALIASES, required=True, label="query id")
    query_image_col = find_col(query_df, QUERY_IMAGE_ALIASES, required=True, label="query image path")

    tile_id_col = find_col(tile_df, TILE_ID_ALIASES, required=True, label="tile id")
    tile_path_col = find_col(tile_df, TILE_PATH_ALIASES, required=True, label="tile image path")
    tile_x_col = find_col(tile_df, TILE_CENTER_X_ALIASES, required=True, label="tile center X")
    tile_y_col = find_col(tile_df, TILE_CENTER_Y_ALIASES, required=True, label="tile center Y")
    tile_xmin_col = find_col(tile_df, TILE_BBOX_XMIN_ALIASES, required=False, label="tile bbox xmin")
    tile_xmax_col = find_col(tile_df, TILE_BBOX_XMAX_ALIASES, required=False, label="tile bbox xmax")
    tile_ymin_col = find_col(tile_df, TILE_BBOX_YMIN_ALIASES, required=False, label="tile bbox ymin")
    tile_ymax_col = find_col(tile_df, TILE_BBOX_YMAX_ALIASES, required=False, label="tile bbox ymax")

    query_df = add_projected_query_xy_from_latlon(query_df)
    query_x_col = find_col(query_df, QUERY_EASTING_ALIASES, required=False, label="query projected X")
    query_y_col = find_col(query_df, QUERY_NORTHING_ALIASES, required=False, label="query projected Y")

    t = topk_df.copy()
    q = query_df.copy()
    s = tile_df.copy()

    t["__query_id_norm"] = normalize_id_series(t[topk_query_col])
    t["__tile_id_norm"] = normalize_id_series(t[topk_tile_col])
    q["__query_id_norm"] = normalize_id_series(q[query_id_col])
    s["__tile_id_norm"] = normalize_id_series(s[tile_id_col])

    t[topk_rank_col] = pd.to_numeric(t[topk_rank_col], errors="coerce")
    if topk_score_col:
        t[topk_score_col] = pd.to_numeric(t[topk_score_col], errors="coerce")
    else:
        t["__topk_score_missing"] = np.nan
        topk_score_col = "__topk_score_missing"

    keep_q_cols = ["__query_id_norm", query_id_col, query_image_col]
    if query_x_col and query_y_col:
        keep_q_cols += [query_x_col, query_y_col]
    keep_q_cols = list(dict.fromkeys(keep_q_cols))

    keep_s_cols = ["__tile_id_norm", tile_id_col, tile_path_col, tile_x_col, tile_y_col]
    for c in [tile_xmin_col, tile_xmax_col, tile_ymin_col, tile_ymax_col]:
        if c:
            keep_s_cols.append(c)
    keep_s_cols = list(dict.fromkeys(keep_s_cols))

    merged = t.merge(q[keep_q_cols], on="__query_id_norm", how="left", suffixes=("", "_query"))
    merged = merged.merge(s[keep_s_cols], on="__tile_id_norm", how="left", suffixes=("", "_tile"))

    cols = {
        "topk_query_col": topk_query_col,
        "topk_tile_col": topk_tile_col,
        "topk_rank_col": topk_rank_col,
        "topk_score_col": topk_score_col,
        "query_id_col": query_id_col,
        "query_image_col": query_image_col,
        "tile_id_col": tile_id_col,
        "tile_path_col": tile_path_col,
        "tile_x_col": tile_x_col,
        "tile_y_col": tile_y_col,
    }
    if query_x_col and query_y_col:
        cols["query_x_col"] = query_x_col
        cols["query_y_col"] = query_y_col
    if tile_xmin_col and tile_xmax_col and tile_ymin_col and tile_ymax_col:
        cols["tile_xmin_col"] = tile_xmin_col
        cols["tile_xmax_col"] = tile_xmax_col
        cols["tile_ymin_col"] = tile_ymin_col
        cols["tile_ymax_col"] = tile_ymax_col
    return merged, cols


def evaluate_candidate_errors(df: pd.DataFrame, cols: Dict[str, str]) -> pd.DataFrame:
    """Evaluation-only geometry."""
    out = df.copy()
    qx = cols.get("query_x_col")
    qy = cols.get("query_y_col")
    tx = cols["tile_x_col"]
    ty = cols["tile_y_col"]

    if qx and qy and qx in out.columns and qy in out.columns:
        qx_s = pd.to_numeric(out[qx], errors="coerce")
        qy_s = pd.to_numeric(out[qy], errors="coerce")
        tx_s = pd.to_numeric(out[tx], errors="coerce")
        ty_s = pd.to_numeric(out[ty], errors="coerce")
        dx = tx_s - qx_s
        dy = ty_s - qy_s
        out["candidate_body_error_m"] = np.sqrt(dx * dx + dy * dy)
        out["candidate_hit_le_40m"] = out["candidate_body_error_m"] <= 40.0
        out["candidate_hit_le_80m"] = out["candidate_body_error_m"] <= 80.0
        out["candidate_hit_le_120m"] = out["candidate_body_error_m"] <= 120.0

        bbox_cols = [
            cols.get("tile_xmin_col"), cols.get("tile_xmax_col"),
            cols.get("tile_ymin_col"), cols.get("tile_ymax_col"),
        ]
        if all(bbox_cols):
            xmin = pd.to_numeric(out[cols["tile_xmin_col"]], errors="coerce")
            xmax = pd.to_numeric(out[cols["tile_xmax_col"]], errors="coerce")
            ymin = pd.to_numeric(out[cols["tile_ymin_col"]], errors="coerce")
            ymax = pd.to_numeric(out[cols["tile_ymax_col"]], errors="coerce")
            x_low = np.minimum(xmin, xmax)
            x_high = np.maximum(xmin, xmax)
            y_low = np.minimum(ymin, ymax)
            y_high = np.maximum(ymin, ymax)
            out["candidate_contains_body"] = (
                qx_s.between(x_low, x_high, inclusive="both")
                & qy_s.between(y_low, y_high, inclusive="both")
            )
        else:
            out["candidate_contains_body"] = False
    else:
        out["candidate_body_error_m"] = np.nan
        out["candidate_hit_le_40m"] = False
        out["candidate_hit_le_80m"] = False
        out["candidate_hit_le_120m"] = False
        out["candidate_contains_body"] = False
    return out


def best_error_within_rank(g: pd.DataFrame, rank_col: str, k: int) -> float:
    sub = g[pd.to_numeric(g[rank_col], errors="coerce") <= k]
    vals = pd.to_numeric(sub["candidate_body_error_m"], errors="coerce").dropna()
    return float(vals.min()) if len(vals) else np.nan


def any_contains_within_rank(g: pd.DataFrame, rank_col: str, k: int) -> bool:
    sub = g[pd.to_numeric(g[rank_col], errors="coerce") <= k]
    if len(sub) == 0 or "candidate_contains_body" not in sub.columns:
        return False
    return bool(sub["candidate_contains_body"].fillna(False).astype(bool).any())


def any_hit_within_rank(g: pd.DataFrame, rank_col: str, k: int, threshold_m: float) -> bool:
    sub = g[pd.to_numeric(g[rank_col], errors="coerce") <= k]
    vals = pd.to_numeric(sub["candidate_body_error_m"], errors="coerce")
    return bool((vals <= threshold_m).any())


def build_query_summary(scored: pd.DataFrame, cols: Dict[str, str], top_n: int, hit_threshold_m: float, policy: str) -> pd.DataFrame:
    qcol = "__query_id_norm"
    rank_col = cols["topk_rank_col"]
    rows: List[Dict[str, Any]] = []
    policy_score = "hybrid_score" if policy == "hybrid" else "verifier_score"

    for qid, g in scored.groupby(qcol, sort=False):
        g = g.copy().sort_values(rank_col, kind="mergesort")
        orig_top1 = g.iloc[0]
        reranked = g.sort_values([policy_score, "verifier_score", rank_col], ascending=[False, False, True], kind="mergesort")
        chosen = reranked.iloc[0]

        if "candidate_contains_body" in g.columns and g["candidate_contains_body"].fillna(False).astype(bool).any():
            oracle = g[g["candidate_contains_body"].fillna(False).astype(bool)]
            oracle_mode = "tile_bbox_contains_body"
        else:
            oracle = g[pd.to_numeric(g["candidate_body_error_m"], errors="coerce") <= hit_threshold_m]
            oracle_mode = f"center_error_le_{hit_threshold_m:g}m"

        if len(oracle):
            oracle_best = oracle.sort_values("candidate_body_error_m").iloc[0]
            oracle_first = oracle.sort_values(rank_col).iloc[0]
            oracle_available = True
            oracle_best_error = float(oracle_best["candidate_body_error_m"])
            oracle_first_rank = float(oracle_first[rank_col])
            oracle_tile_id = oracle_best.get(cols["topk_tile_col"])
        else:
            oracle_available = False
            oracle_best_error = np.nan
            oracle_first_rank = np.nan
            oracle_tile_id = None

        orig_err = float(orig_top1["candidate_body_error_m"]) if pd.notna(orig_top1["candidate_body_error_m"]) else np.nan
        chosen_err = float(chosen["candidate_body_error_m"]) if pd.notna(chosen["candidate_body_error_m"]) else np.nan

        rows.append({
            "query_id": qid,
            "original_top1_tile_id": orig_top1.get(cols["topk_tile_col"]),
            "original_top1_rank": orig_top1.get(rank_col),
            "original_top1_dino_score": orig_top1.get(cols["topk_score_col"]),
            "original_top1_error_m": orig_err,
            "original_top1_contains_body": bool(orig_top1.get("candidate_contains_body", False)),
            "original_top1_hit_le_40m": bool(pd.notna(orig_err) and orig_err <= 40.0),
            "original_top1_hit_le_80m": bool(pd.notna(orig_err) and orig_err <= 80.0),
            "original_top1_hit_le_120m": bool(pd.notna(orig_err) and orig_err <= 120.0),
            "original_top1_hit": bool(pd.notna(orig_err) and orig_err <= hit_threshold_m),
            "rerank_policy": policy,
            "reranked_top1_tile_id": chosen.get(cols["topk_tile_col"]),
            "reranked_top1_original_rank": chosen.get(rank_col),
            "reranked_top1_error_m": chosen_err,
            "reranked_top1_contains_body": bool(chosen.get("candidate_contains_body", False)),
            "reranked_top1_hit_le_40m": bool(pd.notna(chosen_err) and chosen_err <= 40.0),
            "reranked_top1_hit_le_80m": bool(pd.notna(chosen_err) and chosen_err <= 80.0),
            "reranked_top1_hit_le_120m": bool(pd.notna(chosen_err) and chosen_err <= 120.0),
            "reranked_top1_hit": bool(pd.notna(chosen_err) and chosen_err <= hit_threshold_m),
            "reranked_top1_verifier_score": chosen.get("verifier_score"),
            "reranked_top1_hybrid_score": chosen.get("hybrid_score"),
            "reranked_top1_num_matches": chosen.get("num_matches"),
            "reranked_top1_lg_score_sum": chosen.get("lg_score_sum"),
            "reranked_top1_lg_score_mean": chosen.get("lg_score_mean"),
            "reranked_top1_inliers": chosen.get("inliers"),
            "reranked_top1_inlier_ratio": chosen.get("inlier_ratio"),
            "reranked_top1_query_inlier_coverage": chosen.get("query_inlier_coverage"),
            "dino_contains_body_top5": any_contains_within_rank(g, rank_col, 5),
            "dino_contains_body_top10": any_contains_within_rank(g, rank_col, 10),
            "dino_contains_body_top20": any_contains_within_rank(g, rank_col, 20),
            "dino_hit_le_40m_top5": any_hit_within_rank(g, rank_col, 5, 40.0),
            "dino_hit_le_40m_top10": any_hit_within_rank(g, rank_col, 10, 40.0),
            "dino_hit_le_40m_top20": any_hit_within_rank(g, rank_col, 20, 40.0),
            "dino_best_error_top5_m": best_error_within_rank(g, rank_col, 5),
            "dino_best_error_top10_m": best_error_within_rank(g, rank_col, 10),
            "dino_best_error_top20_m": best_error_within_rank(g, rank_col, 20),
            "oracle_available_topn": oracle_available,
            "oracle_mode": oracle_mode,
            "oracle_best_error_m": oracle_best_error,
            "oracle_first_original_rank": oracle_first_rank,
            "oracle_best_tile_id": oracle_tile_id,
            "top_n": top_n,
            "hit_threshold_m": hit_threshold_m,
        })
    return pd.DataFrame(rows)


def summarize(query_summary: pd.DataFrame, args: argparse.Namespace, files: Dict[str, str], cols: Dict[str, str]) -> Dict[str, Any]:
    n = int(len(query_summary))

    def med(col: str) -> Optional[float]:
        if col not in query_summary.columns:
            return None
        v = pd.to_numeric(query_summary[col], errors="coerce").dropna()
        return float(v.median()) if len(v) else None

    def count_true(col: str) -> int:
        return int(query_summary[col].fillna(False).astype(bool).sum()) if col in query_summary.columns and n else 0

    improved = pd.to_numeric(query_summary["original_top1_error_m"], errors="coerce") - pd.to_numeric(query_summary["reranked_top1_error_m"], errors="coerce")
    return {
        "stage": "S8.12E.1B",
        "status": "PASS_LIGHTGLUE_TOPK_VERIFIER_RERANKER",
        "config": files["config"],
        "query_csv": files["query_csv"],
        "topk_csv": files["topk_csv"],
        "tile_index_csv": files["tile_index_csv"],
        "variant": args.variant,
        "tag": args.tag,
        "top_n": args.top_n,
        "verifier": "lightglue_superpoint",
        "device": args.device,
        "resolved_device": files.get("resolved_device"),
        "max_keypoints": args.max_keypoints,
        "resize_long": args.resize_long,
        "policy": args.policy,
        "hit_threshold_m": args.hit_threshold_m,
        "query_count": n,
        "pair_count": int(files.get("pair_count", 0)),
        "total_runtime_s": float(files.get("total_runtime_s", 0.0)),
        "mean_pair_runtime_s": float(files.get("mean_pair_runtime_s", 0.0)),
        "dino_contains_body_top5_hits": count_true("dino_contains_body_top5"),
        "dino_contains_body_top10_hits": count_true("dino_contains_body_top10"),
        "dino_contains_body_top20_hits": count_true("dino_contains_body_top20"),
        "dino_hit_le_40m_top20_hits": count_true("dino_hit_le_40m_top20"),
        "original_top1_contains_body_hits": count_true("original_top1_contains_body"),
        "reranked_top1_contains_body_hits": count_true("reranked_top1_contains_body"),
        "original_top1_hit_le_40m_hits": count_true("original_top1_hit_le_40m"),
        "original_top1_hit_le_80m_hits": count_true("original_top1_hit_le_80m"),
        "original_top1_hit_le_120m_hits": count_true("original_top1_hit_le_120m"),
        "reranked_top1_hit_le_40m_hits": count_true("reranked_top1_hit_le_40m"),
        "reranked_top1_hit_le_80m_hits": count_true("reranked_top1_hit_le_80m"),
        "reranked_top1_hit_le_120m_hits": count_true("reranked_top1_hit_le_120m"),
        "oracle_available_topn_hits": count_true("oracle_available_topn"),
        "original_top1_error_median_m": med("original_top1_error_m"),
        "reranked_top1_error_median_m": med("reranked_top1_error_m"),
        "oracle_best_error_median_m": med("oracle_best_error_m"),
        "dino_best_error_top5_median_m": med("dino_best_error_top5_m"),
        "dino_best_error_top10_median_m": med("dino_best_error_top10_m"),
        "dino_best_error_top20_median_m": med("dino_best_error_top20_m"),
        "rerank_improved_rows": int((improved > 0).sum()),
        "rerank_worsened_rows": int((improved < 0).sum()),
        "rerank_equal_rows": int((improved == 0).sum()),
        "resolved_columns": cols,
    }


def save_figures(query_summary: pd.DataFrame, out_dir: Path) -> None:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    if query_summary.empty:
        return

    vals = [
        pd.to_numeric(query_summary["original_top1_error_m"], errors="coerce").dropna(),
        pd.to_numeric(query_summary["reranked_top1_error_m"], errors="coerce").dropna(),
        pd.to_numeric(query_summary["dino_best_error_top20_m"], errors="coerce").dropna(),
    ]
    plt.figure(figsize=(7, 5))
    plt.boxplot(vals, labels=["DINO Top-1", "LG rerank", "Best Top-20"], showfliers=False)
    plt.ylabel("Center error [m]")
    plt.title("S8.12E.1B LightGlue reranking error")
    plt.tight_layout()
    plt.savefig(fig_dir / "s8_12e1b_lightglue_error_boxplot.png", dpi=180)
    plt.close()

    x = pd.to_numeric(query_summary["original_top1_error_m"], errors="coerce")
    y = pd.to_numeric(query_summary["reranked_top1_error_m"], errors="coerce")
    plt.figure(figsize=(6, 6))
    plt.scatter(x, y, s=18, alpha=0.75)
    lim = float(np.nanmax([x.max(), y.max(), 1.0]))
    plt.plot([0, lim], [0, lim], linestyle="--", linewidth=1)
    plt.xlabel("DINO Top-1 center error [m]")
    plt.ylabel("LightGlue reranked center error [m]")
    plt.title("Original vs LightGlue-reranked error")
    plt.tight_layout()
    plt.savefig(fig_dir / "s8_12e1b_original_vs_lightglue_error.png", dpi=180)
    plt.close()

    labels = ["DINO Top-1 contains", "LG Top-1 contains", "DINO Top-20 contains"]
    counts = [
        int(query_summary["original_top1_contains_body"].fillna(False).astype(bool).sum()),
        int(query_summary["reranked_top1_contains_body"].fillna(False).astype(bool).sum()),
        int(query_summary["dino_contains_body_top20"].fillna(False).astype(bool).sum()),
    ]
    plt.figure(figsize=(8, 4))
    plt.bar(labels, counts)
    plt.ylabel("Query count")
    plt.xticks(rotation=15, ha="right")
    plt.title("Containment hit counts")
    plt.tight_layout()
    plt.savefig(fig_dir / "s8_12e1b_containment_hit_counts.png", dpi=180)
    plt.close()


def write_checkpoint(rows: List[Dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    tmp = path.with_suffix(path.suffix + ".tmp")
    pd.DataFrame(rows).to_csv(tmp, index=False)
    tmp.replace(path)


def run_lightglue_scoring(
    merged: pd.DataFrame,
    cols: Dict[str, str],
    repo_root: Path,
    output_root: Path,
    map_output_root: Path,
    out_dir: Path,
    args: argparse.Namespace,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    rank_col = cols["topk_rank_col"]
    work = merged[pd.to_numeric(merged[rank_col], errors="coerce") <= args.top_n].copy()
    work = work.sort_values(["__query_id_norm", rank_col], kind="mergesort")

    unique_qids = list(work["__query_id_norm"].dropna().unique())
    if args.limit_queries and args.limit_queries > 0:
        unique_qids = unique_qids[: args.limit_queries]
        work = work[work["__query_id_norm"].isin(unique_qids)].copy()

    if args.max_pairs and args.max_pairs > 0:
        work = work.iloc[: args.max_pairs].copy()

    pair_count = int(len(work))
    info(f"LightGlue pairs to process: {pair_count}")

    device = choose_device(args.device)
    info(f"Resolved device: {device}")
    verifier = LightGlueVerifier(device=device, max_keypoints=args.max_keypoints, resize_long=args.resize_long)

    rows: List[Dict[str, Any]] = []
    t_start = time.perf_counter()
    checkpoint_path = out_dir / "s8_12e1b_candidate_lightglue_scores_checkpoint.csv"

    extra_roots = [output_root, map_output_root, repo_root]
    for idx, (_, row) in enumerate(work.iterrows(), start=1):
        q_path = resolve_image_path(row.get(cols["query_image_col"]), repo_root, extra_roots)
        s_path = resolve_image_path(row.get(cols["tile_path_col"]), repo_root, extra_roots)

        if q_path is None or s_path is None:
            res = LGResult(0, 0.0, 0.0, 0, 0.0, 0.0, 0.0, False, 0.0, 0.0, "missing path")
        elif not q_path.exists() or not s_path.exists():
            res = LGResult(0, 0.0, 0.0, 0, 0.0, 0.0, 0.0, False, 0.0, 0.0, f"image missing q={q_path.exists()} s={s_path.exists()}")
        else:
            res = verifier.match(q_path, s_path, ransac_thresh=args.ransac_thresh)

        d = row.to_dict()
        d.update({
            "query_image_abs": str(q_path) if q_path else None,
            "tile_image_abs": str(s_path) if s_path else None,
            "num_matches": res.num_matches,
            "lg_score_sum": res.score_sum,
            "lg_score_mean": res.score_mean,
            "inliers": res.inliers,
            "inlier_ratio": res.inlier_ratio,
            "query_inlier_coverage": res.query_inlier_coverage,
            "sat_inlier_coverage": res.sat_inlier_coverage,
            "homography_ok": res.homography_ok,
            "verifier_score": res.verifier_score,
            "pair_runtime_s": res.runtime_s,
            "verifier_error": res.error,
        })
        rows.append(d)

        if idx % args.progress_every == 0 or idx == pair_count:
            elapsed = time.perf_counter() - t_start
            rate = idx / max(elapsed, 1e-9)
            info(f"  processed {idx}/{pair_count} pairs | {rate:.2f} pairs/s | feature cache {len(verifier.cache)} images")
        if args.checkpoint_every > 0 and idx % args.checkpoint_every == 0:
            write_checkpoint(rows, checkpoint_path)

    total_runtime = time.perf_counter() - t_start
    scored = pd.DataFrame(rows)
    # Hybrid score: LightGlue evidence plus a small DINO rank prior.
    # The prior protects against high-score geometric false positives but still allows reranking.
    r = pd.to_numeric(scored[rank_col], errors="coerce")
    dino_prior = 1.0 / np.sqrt(np.maximum(r, 1.0))
    scored["dino_rank_prior"] = dino_prior
    scored["hybrid_score"] = scored["verifier_score"].astype(float) + args.dino_prior_weight * dino_prior
    scored["verifier_rank"] = scored.groupby("__query_id_norm")["verifier_score"].rank(method="first", ascending=False)
    scored["hybrid_rank"] = scored.groupby("__query_id_norm")["hybrid_score"].rank(method="first", ascending=False)

    metrics = {
        "pair_count": pair_count,
        "total_runtime_s": total_runtime,
        "mean_pair_runtime_s": float(scored["pair_runtime_s"].mean()) if len(scored) else 0.0,
        "resolved_device": str(device),
        "unique_feature_images": len(verifier.cache),
    }
    return scored, metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="S8.12E.1B LightGlue Top-K verifier/reranker")
    p.add_argument("--config", required=True, help="Dataset YAML config, e.g. configs/dataset_villoc_45deg.yaml")
    p.add_argument("--variant", default="1024_s512")
    p.add_argument("--tag", default="dinov2_vits14_img224_center_square_avgpatch_cpu")
    p.add_argument("--top-n", type=int, default=20)
    p.add_argument("--policy", choices=["hybrid", "verifier"], default="hybrid")
    p.add_argument("--hit-threshold-m", type=float, default=40.0)
    p.add_argument("--query-csv", default=None)
    p.add_argument("--topk-csv", default=None)
    p.add_argument("--tile-index-csv", default=None)
    p.add_argument("--out-root", default=None)
    p.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0. auto chooses cuda if available else cpu.")
    p.add_argument("--max-keypoints", type=int, default=1024)
    p.add_argument("--resize-long", type=int, default=768, help="Longest image side for LightGlue extractor. 0 disables explicit resize.")
    p.add_argument("--ransac-thresh", type=float, default=5.0)
    p.add_argument("--dino-prior-weight", type=float, default=8.0)
    p.add_argument("--limit-queries", type=int, default=0, help="Smoke/chunk mode: process only first N query IDs.")
    p.add_argument("--max-pairs", type=int, default=0, help="Smoke/chunk mode: process only first N pairs after filtering.")
    p.add_argument("--checkpoint-every", type=int, default=100)
    p.add_argument("--progress-every", type=int, default=50)
    p.add_argument("--num-threads", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    configure_threads(args.num_threads)

    config_path = Path(args.config).expanduser().resolve()
    if not config_path.exists():
        die(f"Config not found: {config_path}")
    repo_root = config_path.parent.parent.resolve() if config_path.parent.name == "configs" else Path.cwd().resolve()
    config = read_yaml(config_path)
    output_root = resolve_output_root(config, config_path, repo_root)
    map_output_root = resolve_map_output_root(config, repo_root, output_root)
    query_csv = resolve_query_csv(args, output_root, repo_root)
    topk_csv = resolve_topk_csv(args, output_root, repo_root)
    tile_index_csv = resolve_tile_index_csv(args, map_output_root, output_root, repo_root)

    if args.out_root:
        out_dir = maybe_abs(args.out_root, repo_root)  # type: ignore[assignment]
    else:
        out_dir = output_root / "reports" / "s8_12e1b_lightglue_top20_verifier_reranker" / f"{args.variant}_{args.policy}"
    assert out_dir is not None
    out_dir.mkdir(parents=True, exist_ok=True)

    info("S8.12E.1B LightGlue Top-K verifier/reranker")
    info("------------------------------------------------")
    info(f"config:     {config_path}")
    info(f"output root:{output_root}")
    info(f"map root:   {map_output_root}")
    info(f"query csv:  {query_csv}")
    info(f"topk csv:   {topk_csv}")
    info(f"tile index: {tile_index_csv}")
    info(f"out root:   {out_dir}")
    info("")

    query_df = pd.read_csv(query_csv)
    topk_df = pd.read_csv(topk_csv)
    tile_df = pd.read_csv(tile_index_csv)
    merged, cols = prepare_inputs(query_df, topk_df, tile_df)
    merged = evaluate_candidate_errors(merged, cols)

    info("Resolved columns")
    info("----------------")
    for k, v in cols.items():
        info(f"{k}: {v}")
    info("")

    scored, runtime_metrics = run_lightglue_scoring(
        merged, cols, repo_root, output_root, map_output_root, out_dir, args
    )

    query_summary = build_query_summary(scored, cols, args.top_n, args.hit_threshold_m, args.policy)
    files = {
        "config": str(config_path),
        "query_csv": str(query_csv),
        "topk_csv": str(topk_csv),
        "tile_index_csv": str(tile_index_csv),
        **{k: str(v) for k, v in runtime_metrics.items()},
    }
    summary = summarize(query_summary, args, files, cols)

    candidate_csv = out_dir / "s8_12e1b_all_candidate_lightglue_scores.csv"
    query_summary_csv = out_dir / "s8_12e1b_query_summary.csv"
    summary_json = out_dir / "s8_12e1b_summary.json"
    scored.to_csv(candidate_csv, index=False)
    query_summary.to_csv(query_summary_csv, index=False)
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    save_figures(query_summary, out_dir)

    mode = query_summary["oracle_mode"].mode().iloc[0] if "oracle_mode" in query_summary.columns and len(query_summary) else "unknown"
    info("Top-1 diagnostic")
    info("----------------")
    info(f"queries:                   {summary['query_count']}")
    info(f"pairs:                     {summary['pair_count']}")
    info(f"device:                    {summary['resolved_device']}")
    info(f"DINO Top-1 center<=40m:    {summary['original_top1_hit_le_40m_hits']} / {summary['query_count']}")
    info(f"DINO Top-1 contains body:  {summary['original_top1_contains_body_hits']} / {summary['query_count']}")
    info(f"LG Top-1 center<=40m:      {summary['reranked_top1_hit_le_40m_hits']} / {summary['query_count']}")
    info(f"LG Top-1 contains body:    {summary['reranked_top1_contains_body_hits']} / {summary['query_count']}")
    info(f"DINO contains body Top-5:  {summary['dino_contains_body_top5_hits']} / {summary['query_count']}")
    info(f"DINO contains body Top-10: {summary['dino_contains_body_top10_hits']} / {summary['query_count']}")
    info(f"DINO contains body Top-20: {summary['dino_contains_body_top20_hits']} / {summary['query_count']}")
    info(f"DINO center<=40m Top-20:   {summary['dino_hit_le_40m_top20_hits']} / {summary['query_count']}")
    info(f"DINO median center error:  {summary['original_top1_error_median_m']:.3f} m")
    info(f"LG median center error:    {summary['reranked_top1_error_median_m']:.3f} m")
    info(f"Best Top-20 median error:  {summary['dino_best_error_top20_median_m']:.3f} m")
    info(f"Oracle mode:              {mode}")
    info(f"rerank improved rows:      {summary['rerank_improved_rows']}")
    info(f"rerank worsened rows:      {summary['rerank_worsened_rows']}")
    info(f"total runtime:             {summary['total_runtime_s']:.1f} s")
    info(f"mean pair runtime:         {summary['mean_pair_runtime_s']:.3f} s")
    info("")
    info("Outputs")
    info("-------")
    info(f"candidate scores: {candidate_csv}")
    info(f"query summary:    {query_summary_csv}")
    info(f"summary:          {summary_json}")
    info(f"figures:          {out_dir / 'figures'}")
    info("")
    info("STATUS: PASS_LIGHTGLUE_TOPK_VERIFIER_RERANKER")


if __name__ == "__main__":
    main()
