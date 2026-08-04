#!/usr/bin/env python3
"""
S8.12E.1 — Top-K verifier / reranker for Villoc DINO retrieval candidates.

Purpose
-------
DINOv2 retrieval already provides a strong candidate pool for Villoc 45° and 90°.
This script verifies/reranks the Top-K candidates using only image evidence.
Reference coordinates are used only after reranking for evaluation metrics.

Using ORB as baseline verification to check if the script works and for comparision later.

Typical use
-----------
python scripts/villoc/s8_12e1_top20_verifier_reranker.py \
  --config configs/dataset_villoc_45deg.yaml \
  --variant 1024_s512 \
  --tag dinov2_vits14_img224_center_square_avgpatch_cpu \
  --top-n 20 \
  --verifier orb \
  --policy hybrid

2. 90deg dataset:

python scripts/villoc/s8_12e1_top20_verifier_reranker.py \
  --config configs/dataset_villoc_90deg.yaml \
  --variant 1024_s512 \
  --tag dinov2_vits14_img224_center_square_avgpatch_cpu \
  --top-n 20 \
  --verifier orb \
  --preprocess clahe_luma \
  --policy hybrid \
  2>&1 | tee outputs/villoc/90_deg/logs/s8_12e1_top20_verifier/s8_12e1_90deg_orb_hybrid_1024_s512.log

3. running traj01 villoc dataset:

cd /Users/harishprabhu/Documents/drone-localisation
source .drone_venv/bin/activate
export PYTHONPATH=$PWD/src

CFG=configs/dataset_villoc_traj01_90deg_stable120m.yaml
ROOT=outputs/villoc/traj01_90deg_stable120m
TAG=dinov2_vits14_img518_center_square_avgpatch_cpu
VARIANT=1024_s512

mkdir -p "$ROOT/logs/s8_12e1_top20_verifier_reranker"

python scripts/villoc/s8_12e1_top20_verifier_reranker.py \
  --config "$CFG" \
  --variant "$VARIANT" \
  --tag "$TAG" \
  --query-csv "$ROOT/metadata/s8_10b_canonical_uav_query_manifest.csv" \
  --topk-csv "$ROOT/retrieval/s8_11d/s8_11d_topk_${VARIANT}_${TAG}.csv" \
  --tile-index-csv "outputs/villoc/90_deg/metadata/s8_9_satellite_tile_index_${VARIANT}.csv" \
  --out-root "$ROOT/reports/s8_12e1_top20_verifier_reranker/${VARIANT}_orb_hybrid_top20_img518" \
  --top-n 20 \
  --hit-threshold-m 40 \
  --verifier orb \
  --preprocess clahe_luma \
  --nfeatures 1800 \
  --resize-long 1024 \
  --ratio 0.80 \
  --ransac-thresh 5.0 \
  --policy hybrid \
  --rank-prior-weight 2.0 \
  --progress-every 250 \
  2>&1 | tee "$ROOT/logs/s8_12e1_top20_verifier_reranker/s8_12e1_${VARIANT}_orb_hybrid_top20_img518.log"

-- Change Variant and run for 512_s256 after the above.

Design rules
------------
- Config-driven path resolution; no hardcoded dataset angle is required.
- Query latitude/longitude and tile easting/northing are used only for evaluation.
- Retrieval/reranking score uses image evidence and optional original DINO rank prior only.
- The script prints resolved files/columns and fails loudly if schemas differ.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
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
    "center_x_m", "tile_center_x_m", "x_center_m", "easting_m",
    "center_x_lks94_m", "center_x_3346_m", "x_3346_m", "x_lks94_m", "proj_center_x_m",
    "center_x", "x_center",
]
TILE_CENTER_Y_ALIASES = [
    "center_northing", "tile_center_northing", "center_northing_m", "tile_center_northing_m",
    "center_y_m", "tile_center_y_m", "y_center_m", "northing_m",
    "center_y_lks94_m", "center_y_3346_m", "y_3346_m", "y_lks94_m", "proj_center_y_m",
    "center_y", "y_center",
]

TILE_BBOX_XMIN_ALIASES = [
    "left_easting", "left_easting_m",
    "xmin_m", "x_min_m", "min_x_m", "bbox_xmin_m", "x0_m", "xmin",
]
TILE_BBOX_XMAX_ALIASES = [
    "right_easting", "right_easting_m",
    "xmax_m", "x_max_m", "max_x_m", "bbox_xmax_m", "x1_m", "xmax",
]
TILE_BBOX_YMIN_ALIASES = [
    "bottom_northing", "bottom_northing_m",
    "ymin_m", "y_min_m", "min_y_m", "bbox_ymin_m", "y0_m", "ymin",
]
TILE_BBOX_YMAX_ALIASES = [
    "top_northing", "top_northing_m",
    "ymax_m", "y_max_m", "max_y_m", "bbox_ymax_m", "y1_m", "ymax",
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

# def read_yaml(path:Path) -> Dict[str, Any]:


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


def coerce_numeric(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce")


def normalize_id_series(s: pd.Series) -> pd.Series:
    """Normalize IDs while preserving strings like sat_000001."""
    out = s.astype(str).str.strip()
    out = out.str.replace(r"\.0$", "", regex=True)
    return out


def normalize_tile_id_value(v: Any) -> str:
    s = str(v).strip()
    s = re.sub(r"\.0$", "", s)
    return s


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

    # Last-resort inference from config name.
    stem = config_path.stem.lower()
    if "45" in stem:
        return (repo_root / "outputs/villoc/45_deg").resolve()
    if "90" in stem:
        return (repo_root / "outputs/villoc/90_deg").resolve()
    return (repo_root / "outputs/villoc").resolve()


def resolve_map_output_root(config: Dict[str, Any], repo_root: Path, output_root: Path) -> Path:
    """Return output root where reusable map tile indexes likely live.

    45° Villoc reuses 90° orthophoto/tile database, so map/tile index files may be
    under outputs/villoc/90_deg even when query output root is 45_deg.
    """
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

    # Villoc 45° known reuse case.
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


# ----------------------------- verifier engine -----------------------------

@dataclass
class FeaturePack:
    ok: bool
    image_shape: Tuple[int, int]
    keypoints: Any
    descriptors: Optional[np.ndarray]
    error: Optional[str] = None


@dataclass
class MatchResult:
    good_matches: int
    inliers: int
    inlier_ratio: float
    query_inlier_coverage: float
    sat_inlier_coverage: float
    homography_ok: bool
    verifier_score: float
    error: Optional[str] = None


def read_image_for_verifier(path: Path, preprocess: str, resize_long: int) -> Tuple[Optional[np.ndarray], Optional[str]]:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return None, f"cv2.imread failed: {path}"

    h, w = img.shape[:2]
    if resize_long > 0:
        scale = float(resize_long) / float(max(h, w))
        if scale < 1.0:
            img = cv2.resize(img, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)

    if preprocess == "gray":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif preprocess == "luma":
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        gray = ycrcb[:, :, 0]
    elif preprocess == "clahe_luma":
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        y = ycrcb[:, :, 0]
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(y)
    else:
        return None, f"unknown preprocess: {preprocess}"
    return gray, None


def create_detector(verifier: str, nfeatures: int) -> Any:
    if verifier == "orb":
        return cv2.ORB_create(nfeatures=nfeatures, fastThreshold=7, edgeThreshold=15, patchSize=31)
    if verifier == "akaze":
        return cv2.AKAZE_create()
    die(f"Unknown verifier: {verifier}")


def compute_features(path: Path, detector: Any, preprocess: str, resize_long: int) -> FeaturePack:
    gray, err = read_image_for_verifier(path, preprocess, resize_long)
    if gray is None:
        return FeaturePack(False, (0, 0), [], None, err)
    kps, desc = detector.detectAndCompute(gray, None)
    if desc is None or len(kps) == 0:
        return FeaturePack(False, gray.shape[:2], kps or [], None, "no descriptors")
    return FeaturePack(True, gray.shape[:2], kps, desc, None)


def bbox_coverage(points: np.ndarray, shape_hw: Tuple[int, int]) -> float:
    if points.shape[0] < 4:
        return 0.0
    h, w = shape_hw
    if h <= 0 or w <= 0:
        return 0.0
    xs = points[:, 0]
    ys = points[:, 1]
    area = max(0.0, float(xs.max() - xs.min())) * max(0.0, float(ys.max() - ys.min()))
    return float(np.clip(area / float(w * h), 0.0, 1.0))


def verify_pair(q: FeaturePack, s: FeaturePack, ratio: float, ransac_thresh: float) -> MatchResult:
    if not q.ok or not s.ok or q.descriptors is None or s.descriptors is None:
        return MatchResult(0, 0, 0.0, 0.0, 0.0, False, 0.0, q.error or s.error or "features missing")

    norm = cv2.NORM_HAMMING
    matcher = cv2.BFMatcher(norm, crossCheck=False)
    try:
        knn = matcher.knnMatch(q.descriptors, s.descriptors, k=2)
    except cv2.error as exc:
        return MatchResult(0, 0, 0.0, 0.0, 0.0, False, 0.0, f"knnMatch failed: {exc}")

    good = []
    for pair in knn:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            good.append(m)

    if len(good) < 4:
        score = 0.03 * len(good)
        return MatchResult(len(good), 0, 0.0, 0.0, 0.0, False, score, None)

    q_pts = np.float32([q.keypoints[m.queryIdx].pt for m in good])
    s_pts = np.float32([s.keypoints[m.trainIdx].pt for m in good])

    H, mask = cv2.findHomography(q_pts, s_pts, cv2.RANSAC, ransac_thresh)
    if mask is None:
        inliers = 0
        q_cov = 0.0
        s_cov = 0.0
    else:
        inlier_mask = mask.ravel().astype(bool)
        inliers = int(inlier_mask.sum())
        q_cov = bbox_coverage(q_pts[inlier_mask], q.image_shape)
        s_cov = bbox_coverage(s_pts[inlier_mask], s.image_shape)

    inlier_ratio = float(inliers / max(1, len(good)))
    homography_ok = bool(H is not None and inliers >= 4)

    # Conservative sparse-verifier score. Coverage prevents tiny accidental clusters from winning.
    verifier_score = (
        float(inliers)
        + 0.04 * float(len(good))
        + 8.0 * float(inlier_ratio)
        + 4.0 * float(math.sqrt(max(q_cov, 0.0)))
        + 2.0 * float(math.sqrt(max(s_cov, 0.0)))
    )
    return MatchResult(len(good), inliers, inlier_ratio, q_cov, s_cov, homography_ok, verifier_score, None)


# ----------------------------- evaluation logic -----------------------------

def prepare_inputs(
    query_df: pd.DataFrame,
    topk_df: pd.DataFrame,
    tile_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, str]]:
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
    if tile_xmin_col and tile_xmax_col and tile_ymin_col and tile_ymax_col:
        cols["tile_xmin_col"] = tile_xmin_col
        cols["tile_xmax_col"] = tile_xmax_col
        cols["tile_ymin_col"] = tile_ymin_col
        cols["tile_ymax_col"] = tile_ymax_col
        
    if query_x_col and query_y_col:
        cols["query_x_col"] = query_x_col
        cols["query_y_col"] = query_y_col

    return merged, cols


def evaluate_candidate_errors(df: pd.DataFrame, cols: Dict[str, str]) -> pd.DataFrame:
    """
    Evaluation-only geometry.

    candidate_body_error_m:
      distance from query body/reference point to candidate tile center.

    candidate_contains_body:
      True when query body/reference point lies inside the candidate tile bbox.

    Important:
      For 1024_s512 at 0.2 m/px, tile width is about 204.8 m.
      A geometrically correct containing tile can still have center error > 40 m.
      Therefore, containment is the right candidate-pool oracle definition, while
      center-error thresholds are localization-accuracy diagnostics.
    """
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
            cols.get("tile_xmin_col"),
            cols.get("tile_xmax_col"),
            cols.get("tile_ymin_col"),
            cols.get("tile_ymax_col"),
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
    if len(sub) == 0:
        return np.nan
    vals = pd.to_numeric(sub["candidate_body_error_m"], errors="coerce").dropna()
    return float(vals.min()) if len(vals) else np.nan


def any_contains_within_rank(g: pd.DataFrame, rank_col: str, k: int) -> bool:
    sub = g[pd.to_numeric(g[rank_col], errors="coerce") <= k]
    if len(sub) == 0 or "candidate_contains_body" not in sub.columns:
        return False
    return bool(sub["candidate_contains_body"].fillna(False).astype(bool).any())


def any_hit_within_rank(g: pd.DataFrame, rank_col: str, k: int, threshold_m: float) -> bool:
    sub = g[pd.to_numeric(g[rank_col], errors="coerce") <= k]
    if len(sub) == 0:
        return False
    vals = pd.to_numeric(sub["candidate_body_error_m"], errors="coerce")
    return bool((vals <= threshold_m).any())


def build_query_summary(scored: pd.DataFrame, cols: Dict[str, str], top_n: int, hit_threshold_m: float, policy: str) -> pd.DataFrame:
    qcol = "__query_id_norm"
    rank_col = cols["topk_rank_col"]
    rows: List[Dict[str, Any]] = []

    policy_score = "hybrid_score" if policy == "hybrid" else "verifier_score"
    policy_rank = "hybrid_rank" if policy == "hybrid" else "verifier_rank"

    for qid, g in scored.groupby(qcol, sort=False):
        g = g.copy().sort_values(rank_col, kind="mergesort")
        orig_top1 = g.iloc[0]
        reranked = g.sort_values([policy_score, "verifier_score", rank_col], ascending=[False, False, True], kind="mergesort")
        chosen = reranked.iloc[0]
        # Candidate-pool oracle:
        # Prefer tile bbox containment when available. This matches S8.12D/S8.10-style
        # tile oracle logic better than center_error <= 40 m.
        if "candidate_contains_body" in g.columns and g["candidate_contains_body"].fillna(False).astype(bool).any():
            oracle = g[g["candidate_contains_body"].fillna(False).astype(bool)]
            oracle_mode = "tile_bbox_contains_body"
        else:
            # Fallback for datasets without tile bbox columns.
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
            "original_top1_hit": bool(pd.notna(orig_err) and orig_err <= hit_threshold_m),
            "rerank_policy": policy,
            "reranked_top1_tile_id": chosen.get(cols["topk_tile_col"]),
            "reranked_top1_original_rank": chosen.get(rank_col),
            "reranked_top1_error_m": chosen_err,
            "reranked_top1_hit": bool(pd.notna(chosen_err) and chosen_err <= hit_threshold_m),
            "reranked_top1_verifier_score": chosen.get("verifier_score"),
            "reranked_top1_hybrid_score": chosen.get("hybrid_score"),
            "reranked_top1_good_matches": chosen.get("good_matches"),
            "reranked_top1_inliers": chosen.get("inliers"),
            "reranked_top1_inlier_ratio": chosen.get("inlier_ratio"),
            "reranked_top1_query_inlier_coverage": chosen.get("query_inlier_coverage"),
            "original_top1_contains_body": bool(orig_top1.get("candidate_contains_body", False)),
            "reranked_top1_contains_body": bool(chosen.get("candidate_contains_body", False)),

            "original_top1_hit_le_40m": bool(pd.notna(orig_err) and orig_err <= 40.0),
            "original_top1_hit_le_80m": bool(pd.notna(orig_err) and orig_err <= 80.0),
            "original_top1_hit_le_120m": bool(pd.notna(orig_err) and orig_err <= 120.0),

            "reranked_top1_hit_le_40m": bool(pd.notna(chosen_err) and chosen_err <= 40.0),
            "reranked_top1_hit_le_80m": bool(pd.notna(chosen_err) and chosen_err <= 80.0),
            "reranked_top1_hit_le_120m": bool(pd.notna(chosen_err) and chosen_err <= 120.0),

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

    orig_hits = int(query_summary["original_top1_hit"].sum()) if n else 0
    rerank_hits = int(query_summary["reranked_top1_hit"].sum()) if n else 0
    oracle_hits = int(query_summary["oracle_available_topn"].sum()) if n else 0

    def count_true(col: str) -> int:
        return int(query_summary[col].fillna(False).astype(bool).sum()) if col in query_summary.columns and n else 0

    improved = pd.to_numeric(query_summary["original_top1_error_m"], errors="coerce") - pd.to_numeric(query_summary["reranked_top1_error_m"], errors="coerce")
    return {
        "stage": "S8.12E.1",
        "status": "PASS_TOPK_VERIFIER_RERANKER",
        "config": files["config"],
        "query_csv": files["query_csv"],
        "topk_csv": files["topk_csv"],
        "tile_index_csv": files["tile_index_csv"],
        "variant": args.variant,
        "tag": args.tag,
        "top_n": args.top_n,
        "verifier": args.verifier,
        "preprocess": args.preprocess,
        "resize_long": args.resize_long,
        "policy": args.policy,
        "hit_threshold_m": args.hit_threshold_m,
        "query_count": n,
        "original_top1_hits": orig_hits,
        "original_top1_hit_rate": float(orig_hits / n) if n else None,
        "reranked_top1_hits": rerank_hits,
        "reranked_top1_hit_rate": float(rerank_hits / n) if n else None,
        "oracle_available_topn_hits": oracle_hits,
        "oracle_available_topn_rate": float(oracle_hits / n) if n else None,

        "dino_contains_body_top5_hits": count_true("dino_contains_body_top5"),
        "dino_contains_body_top10_hits": count_true("dino_contains_body_top10"),
        "dino_contains_body_top20_hits": count_true("dino_contains_body_top20"),

        "dino_hit_le_40m_top5_hits": count_true("dino_hit_le_40m_top5"),
        "dino_hit_le_40m_top10_hits": count_true("dino_hit_le_40m_top10"),
        "dino_hit_le_40m_top20_hits": count_true("dino_hit_le_40m_top20"),

        "original_top1_contains_body_hits": count_true("original_top1_contains_body"),
        "reranked_top1_contains_body_hits": count_true("reranked_top1_contains_body"),

        "original_top1_hit_le_40m_hits": count_true("original_top1_hit_le_40m"),
        "original_top1_hit_le_80m_hits": count_true("original_top1_hit_le_80m"),
        "original_top1_hit_le_120m_hits": count_true("original_top1_hit_le_120m"),

        "reranked_top1_hit_le_40m_hits": count_true("reranked_top1_hit_le_40m"),
        "reranked_top1_hit_le_80m_hits": count_true("reranked_top1_hit_le_80m"),
        "reranked_top1_hit_le_120m_hits": count_true("reranked_top1_hit_le_120m"),

        "dino_best_error_top5_median_m": med("dino_best_error_top5_m"),
        "dino_best_error_top10_median_m": med("dino_best_error_top10_m"),
        "dino_best_error_top20_median_m": med("dino_best_error_top20_m"),

        "original_top1_error_median_m": med("original_top1_error_m"),
        "reranked_top1_error_median_m": med("reranked_top1_error_m"),
        "oracle_best_error_median_m": med("oracle_best_error_m"),
        "rerank_improved_rows": int((improved > 0).sum()),
        "rerank_worsened_rows": int((improved < 0).sum()),
        "rerank_equal_rows": int((improved == 0).sum()),
        "resolved_columns": cols,
    }


def save_figures(query_summary: pd.DataFrame, out_dir: Path) -> None:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    valid = query_summary.dropna(subset=["original_top1_error_m", "reranked_top1_error_m"])
    if len(valid):
        plt.figure(figsize=(7, 5))
        plt.boxplot([valid["original_top1_error_m"], valid["reranked_top1_error_m"]], labels=["DINO Top-1", "Reranked Top-1"], showfliers=False)
        plt.ylabel("Top-1 error [m]")
        plt.title("S8.12E.1 Top-1 error comparison")
        plt.tight_layout()
        plt.savefig(fig_dir / "s8_12e1_top1_error_boxplot.png", dpi=160)
        plt.close()

        plt.figure(figsize=(7, 5))
        x = valid["original_top1_error_m"].to_numpy(float)
        y = valid["reranked_top1_error_m"].to_numpy(float)
        plt.scatter(x, y, s=18, alpha=0.8)
        lim = max(float(np.nanmax(x)), float(np.nanmax(y)), 1.0)
        plt.plot([0, lim], [0, lim], linestyle="--", linewidth=1)
        plt.xlabel("DINO Top-1 error [m]")
        plt.ylabel("Reranked Top-1 error [m]")
        plt.title("Reranked improvement is below diagonal")
        plt.tight_layout()
        plt.savefig(fig_dir / "s8_12e1_original_vs_reranked_error.png", dpi=160)
        plt.close()

    plt.figure(figsize=(7, 5))
    labels = ["DINO Top-1", "Reranked Top-1", "Oracle in Top-N"]
    vals = [
        int(query_summary["original_top1_hit"].sum()),
        int(query_summary["reranked_top1_hit"].sum()),
        int(query_summary["oracle_available_topn"].sum()),
    ]
    plt.bar(labels, vals)
    plt.ylabel("Hit count")
    plt.title("S8.12E.1 hit counts")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(fig_dir / "s8_12e1_hit_counts.png", dpi=160)
    plt.close()


def run(args: argparse.Namespace) -> None:
    config_path = Path(args.config).expanduser().resolve()
    repo_root = Path(args.repo_root).expanduser().resolve() if args.repo_root else config_path.parent.parent.resolve()
    config = read_yaml(config_path)
    output_root = resolve_output_root(config, config_path, repo_root)
    map_output_root = resolve_map_output_root(config, repo_root, output_root)

    query_csv = resolve_query_csv(args, output_root, repo_root)
    topk_csv = resolve_topk_csv(args, output_root, repo_root)
    tile_index_csv = resolve_tile_index_csv(args, map_output_root, output_root, repo_root)

    out_root = maybe_abs(args.out_root, repo_root) if args.out_root else (
        output_root / "reports" / "s8_12e1_top20_verifier_reranker" / f"{args.variant}_{args.verifier}_{args.policy}"
    )
    assert out_root is not None
    out_root.mkdir(parents=True, exist_ok=True)

    info("S8.12E.1 Top-K verifier / reranker")
    info("-------------------------------------")
    info(f"config:       {config_path}")
    info(f"repo root:    {repo_root}")
    info(f"output root:  {output_root}")
    info(f"map root:     {map_output_root}")
    info(f"query csv:    {query_csv}")
    info(f"topk csv:     {topk_csv}")
    info(f"tile index:   {tile_index_csv}")
    info(f"out root:     {out_root}")

    query_df = pd.read_csv(query_csv)
    topk_df = pd.read_csv(topk_csv)
    tile_df = pd.read_csv(tile_index_csv)

    merged, cols = prepare_inputs(query_df, topk_df, tile_df)
    rank_col = cols["topk_rank_col"]
    merged = merged[pd.to_numeric(merged[rank_col], errors="coerce") <= args.top_n].copy()
    merged = merged.sort_values(["__query_id_norm", rank_col], kind="mergesort")
    merged = evaluate_candidate_errors(merged, cols)

    info("\nResolved columns")
    info("----------------")
    for k, v in cols.items():
        info(f"{k}: {v}")

    detector = create_detector(args.verifier, args.nfeatures)
    feature_cache: Dict[str, FeaturePack] = {}

    def get_features(path: Path) -> FeaturePack:
        key = str(path)
        if key not in feature_cache:
            feature_cache[key] = compute_features(path, detector, args.preprocess, args.resize_long)
        return feature_cache[key]

    rows: List[Dict[str, Any]] = []
    total_pairs = len(merged)
    for i, row in merged.reset_index(drop=True).iterrows():
        if (i + 1) % max(1, args.progress_every) == 0 or i == 0 or i + 1 == total_pairs:
            info(f"verifying pair {i + 1}/{total_pairs}")

        q_path = resolve_image_path(row.get(cols["query_image_col"]), repo_root, [output_root])
        s_path = resolve_image_path(row.get(cols["tile_path_col"]), repo_root, [map_output_root, output_root])

        if q_path is None or not q_path.exists():
            mr = MatchResult(0, 0, 0.0, 0.0, 0.0, False, 0.0, f"missing query image: {q_path}")
        elif s_path is None or not s_path.exists():
            mr = MatchResult(0, 0, 0.0, 0.0, 0.0, False, 0.0, f"missing tile image: {s_path}")
        else:
            qfeat = get_features(q_path)
            sfeat = get_features(s_path)
            mr = verify_pair(qfeat, sfeat, args.ratio, args.ransac_thresh)

        d = row.to_dict()
        d.update({
            "query_image_resolved": str(q_path) if q_path else None,
            "tile_image_resolved": str(s_path) if s_path else None,
            "good_matches": mr.good_matches,
            "inliers": mr.inliers,
            "inlier_ratio": mr.inlier_ratio,
            "query_inlier_coverage": mr.query_inlier_coverage,
            "sat_inlier_coverage": mr.sat_inlier_coverage,
            "homography_ok": mr.homography_ok,
            "verifier_score": mr.verifier_score,
            "verifier_error": mr.error,
        })
        rows.append(d)

    scored = pd.DataFrame(rows)
    # DINO prior from original rank only, avoiding unknown score scale.
    r = pd.to_numeric(scored[rank_col], errors="coerce")
    scored["rank_prior_score"] = (float(args.top_n) + 1.0 - r).clip(lower=0.0) / float(args.top_n)
    scored["hybrid_score"] = pd.to_numeric(scored["verifier_score"], errors="coerce").fillna(0.0) + args.rank_prior_weight * scored["rank_prior_score"].fillna(0.0)

    scored["verifier_rank"] = scored.groupby("__query_id_norm")["verifier_score"].rank(method="first", ascending=False)
    scored["hybrid_rank"] = scored.groupby("__query_id_norm")["hybrid_score"].rank(method="first", ascending=False)

    query_summary = build_query_summary(scored, cols, args.top_n, args.hit_threshold_m, args.policy)

    files = {
        "config": str(config_path),
        "query_csv": str(query_csv),
        "topk_csv": str(topk_csv),
        "tile_index_csv": str(tile_index_csv),
    }
    summary = summarize(query_summary, args, files, cols)

    scored_path = out_root / "s8_12e1_all_candidate_verifier_scores.csv"
    qsum_path = out_root / "s8_12e1_query_summary.csv"
    summary_path = out_root / "s8_12e1_summary.json"
    scored.to_csv(scored_path, index=False)
    query_summary.to_csv(qsum_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    save_figures(query_summary, out_root)

    info("\nTop-1 diagnostic")
    info("----------------")
    info(f"queries:                   {summary['query_count']}")
    info(f"DINO Top-1 center<=40m:    {summary['original_top1_hit_le_40m_hits']} / {summary['query_count']}")
    info(f"DINO Top-1 contains body:  {summary['original_top1_contains_body_hits']} / {summary['query_count']}")
    info(f"Rerank Top-1 center<=40m:  {summary['reranked_top1_hit_le_40m_hits']} / {summary['query_count']}")
    info(f"Rerank Top-1 contains body:{summary['reranked_top1_contains_body_hits']} / {summary['query_count']}")
    info(f"DINO contains body Top-5:  {summary['dino_contains_body_top5_hits']} / {summary['query_count']}")
    info(f"DINO contains body Top-10: {summary['dino_contains_body_top10_hits']} / {summary['query_count']}")
    info(f"DINO contains body Top-20: {summary['dino_contains_body_top20_hits']} / {summary['query_count']}")
    info(f"DINO center<=40m Top-20:   {summary['dino_hit_le_40m_top20_hits']} / {summary['query_count']}")
    info(f"DINO median center error:  {summary['original_top1_error_median_m']:.3f} m")
    info(f"Rerank median center err:  {summary['reranked_top1_error_median_m']:.3f} m")
    info(f"Best Top-20 median error:  {summary['dino_best_error_top20_median_m']:.3f} m")
    info(f"Oracle mode:              {query_summary['oracle_mode'].mode().iloc[0] if 'oracle_mode' in query_summary.columns and len(query_summary) else 'unknown'}")
    info(f"rerank improved rows:      {summary['rerank_improved_rows']}")
    info(f"rerank worsened rows:      {summary['rerank_worsened_rows']}")

    info("\nOutputs")
    info("-------")
    info(f"candidate scores: {scored_path}")
    info(f"query summary:    {qsum_path}")
    info(f"summary:          {summary_path}")
    info(f"figures:          {out_root / 'figures'}")
    info("\nSTATUS: PASS_TOPK_VERIFIER_RERANKER")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="S8.12E.1 Top-K sparse verifier / reranker for Villoc DINO candidates")
    p.add_argument("--config", required=True, help="Dataset YAML config, e.g. configs/dataset_villoc_45deg.yaml")
    p.add_argument("--variant", required=True, help="Tile variant, e.g. 1024_s512")
    p.add_argument("--tag", required=True, help="Retrieval descriptor tag used in S8.11D Top-K CSV")
    p.add_argument("--repo-root", default=None, help="Repository root override. Defaults to config parent/..")
    p.add_argument("--query-csv", default=None, help="Optional query manifest/index CSV override")
    p.add_argument("--topk-csv", default=None, help="Optional S8.11D Top-K CSV override")
    p.add_argument("--tile-index-csv", default=None, help="Optional S8.9 tile index CSV override")
    p.add_argument("--out-root", default=None, help="Optional output root override")

    p.add_argument("--top-n", type=int, default=20, help="Number of DINO candidates per query to verify/rerank")
    p.add_argument("--hit-threshold-m", type=float, default=40.0, help="Evaluation-only hit threshold in metres")
    p.add_argument("--oracle-k", type=int, default=None, help="Backward-compatible alias; ignored except for logging if supplied")

    p.add_argument("--verifier", choices=["orb", "akaze"], default="orb", help="Sparse verifier frontend")
    p.add_argument("--preprocess", choices=["gray", "luma", "clahe_luma"], default="clahe_luma")
    p.add_argument("--nfeatures", type=int, default=1800, help="ORB nfeatures. Ignored by AKAZE")
    p.add_argument("--resize-long", type=int, default=1024, help="Resize longer image side before feature extraction; 0 disables")
    p.add_argument("--ratio", type=float, default=0.80, help="Lowe ratio threshold")
    p.add_argument("--ransac-thresh", type=float, default=5.0, help="RANSAC reprojection threshold in resized pixels")

    p.add_argument("--policy", choices=["verifier", "hybrid"], default="hybrid", help="Top-1 selection policy")
    p.add_argument("--rank-prior-weight", type=float, default=2.0, help="Weight for original DINO rank prior in hybrid policy")
    p.add_argument("--progress-every", type=int, default=100, help="Progress print frequency in verified pairs")
    return p.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
