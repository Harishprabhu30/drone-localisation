#!/usr/bin/env python3
"""
S4C.0 — Macro-contour preflight for SatLoc.

Purpose:
  Build visual panels showing how UAV frames, GT/nearest satellite tiles,
  and optional S4B false positives look after macro-contour extraction.

This is diagnostic only:
  - UAV filename lon/lat is used only to find GT/nearest tile for visualization/evaluation.
  - No retrieval ranking is performed here.
  - No benchmark is performed here.

code best to run v1:
(.drone_venv) (base) harishprabhu@Harishs-Air drone-localisation % python scripts/satloc/s4c/s4c_0_macrocontour_preflight.py \
  --sequence traj01 \
  --tokens 1,100,166 \
  --preprocess luma \
  --resize-size 512 \
  --edge-method sobel \
  --blur-ksize 3 \
  --threshold-mode percentile \
  --threshold-percentile 65 \
  --close-ksize 3 \
  --open-ksize 1 \
  --min-component-area 25 \
  --include-s4b-false 
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path.cwd()

DEFAULT_UAV_INDEX = Path("outputs/satloc/metadata/uav_frames_index_enriched.csv")
DEFAULT_SAT_INDEX = Path("outputs/satloc/metadata/satellite_tiles_index_enriched.csv")

OUT_FIG_DIR = Path("outputs/satloc/figures/s4c_macrocontour_phog_chamfer/s4c0_preflight")
OUT_META_DIR = Path("outputs/satloc/metadata/s4c_macrocontour_phog_chamfer")
OUT_REPORT_DIR = Path("outputs/satloc/reports/s4c_macrocontour_phog_chamfer")


# -----------------------------
# Column / path helpers
# -----------------------------

def norm_col(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")


def find_col(df: pd.DataFrame, candidates: list[str], required: bool = False) -> Optional[str]:
    norm_map = {norm_col(c): c for c in df.columns}

    for cand in candidates:
        key = norm_col(cand)
        if key in norm_map:
            return norm_map[key]

    # soft contains match
    for cand in candidates:
        parts = [p for p in norm_col(cand).split("_") if p]
        for c in df.columns:
            nc = norm_col(c)
            if all(p in nc for p in parts):
                return c

    if required:
        raise KeyError(f"Could not find any of columns {candidates}. Available: {list(df.columns)}")
    return None


def parse_token_from_name(name: str) -> Optional[int]:
    base = Path(str(name)).name
    # SatLoc UAV example: 1@0@112.816130@28.297316.png
    m = re.match(r"^(\d+)@", base)
    if m:
        return int(m.group(1))

    # Fallback: first integer in name
    m = re.search(r"(\d+)", base)
    if m:
        return int(m.group(1))
    return None


def parse_lon_lat_from_name(name: str) -> tuple[Optional[float], Optional[float]]:
    base = Path(str(name)).name
    parts = base.split("@")
    if len(parts) >= 4:
        try:
            lon = float(parts[2])
            lat = float(Path(parts[3]).stem)
            return lon, lat
        except ValueError:
            return None, None
    return None, None


def safe_float(value: Any) -> Optional[float]:
    try:
        if pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def safe_int(value: Any) -> Optional[int]:
    try:
        if pd.isna(value):
            return None
        return int(float(value))
    except Exception:
        return None


def build_filename_index(search_dirs: list[Path]) -> dict[str, Path]:
    idx: dict[str, Path] = {}
    for d in search_dirs:
        if not d.exists():
            continue
        for p in d.rglob("*"):
            if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
                idx[p.name] = p
    return idx


def resolve_image_path(value: Any, filename_index: dict[str, Path], fallback_dirs: list[Path]) -> Optional[Path]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None

    s = str(value).strip()
    if not s:
        return None

    p = Path(s)

    candidates = [
        p,
        PROJECT_ROOT / p,
    ]

    for d in fallback_dirs:
        candidates.append(d / p.name)

    for c in candidates:
        if c.exists() and c.is_file():
            return c

    if p.name in filename_index:
        return filename_index[p.name]

    return None


def get_row_path(
    row: pd.Series,
    df: pd.DataFrame,
    filename_index: dict[str, Path],
    fallback_dirs: list[Path],
    kind: str,
) -> Optional[Path]:
    if kind == "uav":
        path_candidates = [
            "image_path", "uav_image_path", "frame_path", "path", "filepath", "file_path",
            "image_file", "filename", "file_name"
        ]
    else:
        path_candidates = [
            "tile_path", "satellite_tile_path", "image_path", "path", "filepath", "file_path",
            "tile_file", "filename", "file_name"
        ]

    path_col = find_col(df, path_candidates, required=False)
    if path_col is not None:
        return resolve_image_path(row[path_col], filename_index, fallback_dirs)

    return None


def get_tile_id(row: pd.Series, df: pd.DataFrame) -> Optional[int]:
    col = find_col(df, ["tile_id", "tile_index", "tile_number", "id", "index"], required=False)
    if col is not None:
        val = safe_int(row[col])
        if val is not None:
            return val

    file_col = find_col(df, ["filename", "file_name", "tile_file", "tile_path", "image_path", "path"], required=False)
    if file_col is not None:
        token = parse_token_from_name(str(row[file_col]))
        if token is not None:
            return token

        stem = Path(str(row[file_col])).stem
        m = re.search(r"(\d+)", stem)
        if m:
            return int(m.group(1))

    return None


def get_uav_token(row: pd.Series, df: pd.DataFrame) -> Optional[int]:
    col = find_col(df, ["token0_id", "token_id", "frame_id", "imgid", "id", "token"], required=False)
    if col is not None:
        val = safe_int(row[col])
        if val is not None:
            return val

    file_col = find_col(df, ["filename", "file_name", "image_path", "path"], required=False)
    if file_col is not None:
        return parse_token_from_name(str(row[file_col]))

    return None


def get_lon_lat(row: pd.Series, df: pd.DataFrame) -> tuple[Optional[float], Optional[float]]:
    lon_col = find_col(df, ["longitude", "lon", "uav_lon", "center_lon"], required=False)
    lat_col = find_col(df, ["latitude", "lat", "uav_lat", "center_lat"], required=False)

    lon = safe_float(row[lon_col]) if lon_col else None
    lat = safe_float(row[lat_col]) if lat_col else None

    if lon is not None and lat is not None:
        return lon, lat

    file_col = find_col(df, ["filename", "file_name", "image_path", "path"], required=False)
    if file_col is not None:
        return parse_lon_lat_from_name(str(row[file_col]))

    return None, None


# -----------------------------
# Geo helpers
# -----------------------------

def approx_lonlat_error_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    # Equirectangular approximation; enough for local ranking/diagnostic display.
    r = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    x = dlambda * math.cos((phi1 + phi2) / 2.0)
    y = dphi
    return r * math.sqrt(x * x + y * y)


def find_bbox_cols(df: pd.DataFrame) -> tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    lon_min = find_col(df, ["lon_min", "min_lon", "bbox_lon_min", "tile_lon_min", "west_lon"], required=False)
    lon_max = find_col(df, ["lon_max", "max_lon", "bbox_lon_max", "tile_lon_max", "east_lon"], required=False)
    lat_min = find_col(df, ["lat_min", "min_lat", "bbox_lat_min", "tile_lat_min", "south_lat"], required=False)
    lat_max = find_col(df, ["lat_max", "max_lat", "bbox_lat_max", "tile_lat_max", "north_lat"], required=False)
    return lon_min, lon_max, lat_min, lat_max


def find_center_cols(df: pd.DataFrame) -> tuple[Optional[str], Optional[str]]:
    center_lon = find_col(df, ["center_lon", "tile_center_lon", "lon_center", "longitude_center"], required=False)
    center_lat = find_col(df, ["center_lat", "tile_center_lat", "lat_center", "latitude_center"], required=False)

    if center_lon is None:
        center_lon = find_col(df, ["longitude", "lon"], required=False)
    if center_lat is None:
        center_lat = find_col(df, ["latitude", "lat"], required=False)

    return center_lon, center_lat


def select_gt_or_nearest_tile(sat_df: pd.DataFrame, uav_lon: float, uav_lat: float) -> tuple[pd.Series, str, float]:
    lon_min, lon_max, lat_min, lat_max = find_bbox_cols(sat_df)
    center_lon, center_lat = find_center_cols(sat_df)

    candidates = sat_df

    if all(c is not None for c in [lon_min, lon_max, lat_min, lat_max]):
        lon_lo = sat_df[lon_min].astype(float).where(sat_df[lon_min].astype(float) <= sat_df[lon_max].astype(float), sat_df[lon_max].astype(float))
        lon_hi = sat_df[lon_min].astype(float).where(sat_df[lon_min].astype(float) > sat_df[lon_max].astype(float), sat_df[lon_max].astype(float))
        lat_lo = sat_df[lat_min].astype(float).where(sat_df[lat_min].astype(float) <= sat_df[lat_max].astype(float), sat_df[lat_max].astype(float))
        lat_hi = sat_df[lat_min].astype(float).where(sat_df[lat_min].astype(float) > sat_df[lat_max].astype(float), sat_df[lat_max].astype(float))

        contains = sat_df[(uav_lon >= lon_lo) & (uav_lon <= lon_hi) & (uav_lat >= lat_lo) & (uav_lat <= lat_hi)]
        if len(contains) > 0:
            candidates = contains
            method = "bbox_contains_gt"
        else:
            method = "nearest_center_fallback"
    else:
        method = "nearest_center_no_bbox"

    if center_lon is not None and center_lat is not None:
        best_idx = None
        best_err = float("inf")
        for idx, row in candidates.iterrows():
            clon = safe_float(row[center_lon])
            clat = safe_float(row[center_lat])
            if clon is None or clat is None:
                continue
            err = approx_lonlat_error_m(uav_lon, uav_lat, clon, clat)
            if err < best_err:
                best_err = err
                best_idx = idx

        if best_idx is not None:
            return sat_df.loc[best_idx], method, best_err

    # If no center columns, return first candidate.
    return candidates.iloc[0], method + "_first_row", float("nan")


def find_tile_by_id(sat_df: pd.DataFrame, tile_id: int) -> Optional[pd.Series]:
    ids = []
    for _, row in sat_df.iterrows():
        tid = get_tile_id(row, sat_df)
        ids.append(tid)

    for idx, tid in zip(sat_df.index, ids):
        if tid == tile_id:
            return sat_df.loc[idx]

    return None


def tile_center_error(sat_row: pd.Series, sat_df: pd.DataFrame, uav_lon: float, uav_lat: float) -> float:
    center_lon, center_lat = find_center_cols(sat_df)
    if center_lon is None or center_lat is None:
        return float("nan")

    clon = safe_float(sat_row[center_lon])
    clat = safe_float(sat_row[center_lat])
    if clon is None or clat is None:
        return float("nan")

    return approx_lonlat_error_m(uav_lon, uav_lat, clon, clat)


# -----------------------------
# S4B false positive discovery
# -----------------------------

def find_optional_s4b_false_tile(token: int, gt_tile_id: Optional[int]) -> Optional[int]:
    """
    Tries to locate a previous S4B ranked CSV for this token and pick a high-ranked false tile.
    This is optional. If file formats differ, the script continues without it.
    """
    roots = [
        Path("outputs/satloc/metadata"),
        Path("outputs/satloc/reports"),
    ]

    token_patterns = [
        f"token{token:04d}",
        f"token{token:03d}",
        f"token{token}",
    ]

    csvs: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob("*.csv"):
            low = str(p).lower()
            if "s4b" not in low:
                continue
            if any(tp in low for tp in token_patterns):
                csvs.append(p)

    for csv_path in sorted(csvs):
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue

        tile_col = find_col(df, ["tile_id", "candidate_tile_id", "sat_tile_id", "tile", "id"], required=False)
        if tile_col is None:
            continue

        rank_col = find_col(df, ["rank", "original_rank", "local_rank"], required=False)
        score_col = find_col(df, ["score", "combined_score", "similarity", "new_score"], required=False)
        gt_col = find_col(df, ["contains_gt", "gt", "is_gt", "gt_match"], required=False)

        work = df.copy()

        if rank_col is not None:
            work = work.sort_values(rank_col, ascending=True)
        elif score_col is not None:
            work = work.sort_values(score_col, ascending=False)

        for _, row in work.iterrows():
            tid = safe_int(row[tile_col])
            if tid is None:
                continue
            if gt_tile_id is not None and tid == gt_tile_id:
                continue
            if gt_col is not None:
                val = str(row[gt_col]).strip().lower()
                if val in {"true", "1", "yes"}:
                    continue
            return tid

    return None


# -----------------------------
# Image + macro contour
# -----------------------------

@dataclass
class MacroResult:
    rgb: np.ndarray
    luma: np.ndarray
    raw_edge: np.ndarray
    blurred_edge: np.ndarray
    threshold_mask: np.ndarray
    cleaned_mask: np.ndarray
    contour_canvas: np.ndarray
    stats: dict[str, Any]


def center_crop_square(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    return img[y0:y0 + side, x0:x0 + side]


def ensure_odd(x: int) -> int:
    if x <= 1:
        return 1
    return x if x % 2 == 1 else x + 1


def preprocess_to_luma(rgb: np.ndarray, mode: str) -> np.ndarray:
    mode = mode.lower()

    if mode == "gray":
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    ycrcb = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
    y = ycrcb[:, :, 0]

    if mode == "luma":
        return y

    if mode == "clahe_luma":
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(y)

    raise ValueError(f"Unknown preprocess mode: {mode}")


def remove_small_components(binary: np.ndarray, min_area: int) -> tuple[np.ndarray, int, int]:
    binary = (binary > 0).astype(np.uint8) * 255
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    cleaned = np.zeros_like(binary)
    before = max(0, n_labels - 1)
    kept = 0

    for label in range(1, n_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= min_area:
            cleaned[labels == label] = 255
            kept += 1

    return cleaned, before, kept


def macro_contour_pipeline(
    image_path: Path,
    resize_size: int,
    preprocess: str,
    edge_method: str,
    blur_ksize: int,
    threshold_mode: str,
    threshold_percentile: float,
    close_ksize: int,
    open_ksize: int,
    min_component_area: int,
) -> MacroResult:
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Could not read image: {image_path}")

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = center_crop_square(rgb)
    rgb = cv2.resize(rgb, (resize_size, resize_size), interpolation=cv2.INTER_AREA)

    luma = preprocess_to_luma(rgb, preprocess)

    if edge_method == "sobel":
        gx = cv2.Sobel(luma, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(luma, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
        raw_edge = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    elif edge_method == "canny":
        raw_edge = cv2.Canny(luma, 60, 160)
    else:
        raise ValueError(f"Unsupported edge method: {edge_method}")

    blur_ksize = ensure_odd(blur_ksize)
    if blur_ksize > 1:
        blurred = cv2.GaussianBlur(raw_edge, (blur_ksize, blur_ksize), 0)
    else:
        blurred = raw_edge.copy()

    if threshold_mode == "otsu":
        threshold_value, thresh = cv2.threshold(
            blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
    elif threshold_mode == "percentile":
        threshold_value = float(np.percentile(blurred, threshold_percentile))
        _, thresh = cv2.threshold(blurred, threshold_value, 255, cv2.THRESH_BINARY)
    else:
        raise ValueError(f"Unsupported threshold mode: {threshold_mode}")

    close_ksize = ensure_odd(close_ksize)
    open_ksize = ensure_odd(open_ksize)

    closed = thresh.copy()
    if close_ksize > 1:
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))
        closed = cv2.morphologyEx(closed, cv2.MORPH_CLOSE, k_close)

    opened = closed.copy()
    if open_ksize > 1:
        k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_ksize, open_ksize))
        opened = cv2.morphologyEx(opened, cv2.MORPH_OPEN, k_open)

    cleaned, components_before, components_kept = remove_small_components(opened, min_component_area)

    # Final contour canvas: boundary of cleaned macro regions.
    k3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    contour = cv2.morphologyEx(cleaned, cv2.MORPH_GRADIENT, k3)

    stats = {
        "image_path": str(image_path),
        "resize_size": resize_size,
        "preprocess": preprocess,
        "edge_method": edge_method,
        "blur_ksize": blur_ksize,
        "threshold_mode": threshold_mode,
        "threshold_value": float(threshold_value),
        "threshold_percentile": threshold_percentile if threshold_mode == "percentile" else None,
        "close_ksize": close_ksize,
        "open_ksize": open_ksize,
        "min_component_area": min_component_area,
        "raw_edge_density": float((raw_edge > 0).mean()),
        "threshold_density": float((thresh > 0).mean()),
        "cleaned_density": float((cleaned > 0).mean()),
        "contour_density": float((contour > 0).mean()),
        "components_before": int(components_before),
        "components_kept": int(components_kept),
    }

    return MacroResult(
        rgb=rgb,
        luma=luma,
        raw_edge=raw_edge,
        blurred_edge=blurred,
        threshold_mask=thresh,
        cleaned_mask=cleaned,
        contour_canvas=contour,
        stats=stats,
    )


def render_panel(
    token: int,
    rows: list[dict[str, Any]],
    out_path: Path,
) -> None:
    cols = [
        ("RGB crop", "rgb", "rgb"),
        ("luma", "luma", "gray"),
        ("raw edge", "raw_edge", "gray"),
        ("blurred edge", "blurred_edge", "gray"),
        ("threshold mask", "threshold_mask", "gray"),
        ("cleaned macro mask", "cleaned_mask", "gray"),
        ("final contour canvas", "contour_canvas", "gray"),
    ]

    nrows = len(rows)
    ncols = len(cols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.6 * nrows))
    if nrows == 1:
        axes = np.expand_dims(axes, axis=0)

    for r, item in enumerate(rows):
        result: MacroResult = item["macro"]
        label = item["label"]

        for c, (title, attr, cmap) in enumerate(cols):
            ax = axes[r, c]
            img = getattr(result, attr)
            if cmap == "rgb":
                ax.imshow(img)
            else:
                ax.imshow(img, cmap="gray", vmin=0, vmax=255)

            ax.set_xticks([])
            ax.set_yticks([])

            if r == 0:
                ax.set_title(title, fontsize=10)

            if c == 0:
                ax.set_ylabel(label, fontsize=9)

        # Add compact stats on last column.
        st = result.stats
        axes[r, -1].set_xlabel(
            f"edge={st['raw_edge_density']:.3f} | "
            f"macro={st['cleaned_density']:.3f} | "
            f"cc {st['components_kept']}/{st['components_before']}",
            fontsize=8,
        )

    fig.suptitle(f"S4C.0 Macro-contour preflight — token {token}", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


# -----------------------------
# Main
# -----------------------------

def parse_tokens(s: str) -> list[int]:
    toks = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        toks.append(int(part))
    return toks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", default="traj01")
    parser.add_argument("--tokens", default="1,100,166")
    parser.add_argument("--uav-index", default=str(DEFAULT_UAV_INDEX))
    parser.add_argument("--sat-index", default=str(DEFAULT_SAT_INDEX))
    parser.add_argument("--preprocess", default="luma", choices=["gray", "luma", "clahe_luma"])
    parser.add_argument("--resize-size", type=int, default=512)
    parser.add_argument("--edge-method", default="sobel", choices=["sobel", "canny"])
    parser.add_argument("--blur-ksize", type=int, default=7)
    parser.add_argument("--threshold-mode", default="otsu", choices=["otsu", "percentile"])
    parser.add_argument("--threshold-percentile", type=float, default=75.0)
    parser.add_argument("--close-ksize", type=int, default=7)
    parser.add_argument("--open-ksize", type=int, default=3)
    parser.add_argument("--min-component-area", type=int, default=100)
    parser.add_argument("--include-s4b-false", action="store_true")
    args = parser.parse_args()

    tokens = parse_tokens(args.tokens)

    uav_index_path = Path(args.uav_index)
    sat_index_path = Path(args.sat_index)

    if not uav_index_path.exists():
        raise FileNotFoundError(f"Missing UAV index: {uav_index_path}")
    if not sat_index_path.exists():
        raise FileNotFoundError(f"Missing satellite index: {sat_index_path}")

    uav_df = pd.read_csv(uav_index_path)
    sat_df = pd.read_csv(sat_index_path)

    # Optional sequence filter.
    seq_col = find_col(uav_df, ["sequence", "seq", "trajectory", "traj"], required=False)
    if seq_col is not None:
        uav_df = uav_df[uav_df[seq_col].astype(str) == args.sequence].copy()

    # Build image filename index.
    fallback_uav_dirs = [
        Path("data/raw/satloc/part_1/UAV Data") / args.sequence,
        Path("data/raw/satloc/part_1/UAV Data"),
    ]
    fallback_sat_dirs = [
        Path("data/raw/satloc/part_1/Satellite Data/sat_image_ref"),
        Path("data/raw/satloc/part_1/Satellite Data"),
    ]
    filename_index = build_filename_index(fallback_uav_dirs + fallback_sat_dirs)

    OUT_FIG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_META_DIR.mkdir(parents=True, exist_ok=True)
    OUT_REPORT_DIR.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, Any]] = []
    panel_paths: list[str] = []

    # Precompute UAV tokens robustly.
    uav_tokens = []
    for _, row in uav_df.iterrows():
        uav_tokens.append(get_uav_token(row, uav_df))
    uav_df = uav_df.copy()
    uav_df["_s4c_token"] = uav_tokens

    for token in tokens:
        match = uav_df[uav_df["_s4c_token"] == token]
        if len(match) == 0:
            print(f"[WARN] Token {token}: UAV row not found. Skipping.")
            continue

        uav_row = match.iloc[0]
        uav_lon, uav_lat = get_lon_lat(uav_row, uav_df)
        if uav_lon is None or uav_lat is None:
            print(f"[WARN] Token {token}: could not read UAV lon/lat for GT diagnostic. Skipping.")
            continue

        uav_path = get_row_path(uav_row, uav_df, filename_index, fallback_uav_dirs, kind="uav")
        if uav_path is None:
            print(f"[WARN] Token {token}: UAV image path not found. Skipping.")
            continue

        gt_row, gt_method, gt_error_m = select_gt_or_nearest_tile(sat_df, uav_lon, uav_lat)
        gt_tile_id = get_tile_id(gt_row, sat_df)
        gt_path = get_row_path(gt_row, sat_df, filename_index, fallback_sat_dirs, kind="sat")

        if gt_path is None:
            print(f"[WARN] Token {token}: GT/nearest satellite tile path not found. Skipping.")
            continue

        panel_items: list[dict[str, Any]] = []

        sources = [
            {
                "role": "uav_query",
                "label": f"UAV token {token}",
                "path": uav_path,
                "tile_id": None,
                "center_error_m": 0.0,
                "selection_method": "query",
            },
            {
                "role": "gt_or_nearest_tile",
                "label": f"GT/nearest tile {gt_tile_id}\n{gt_method}, err={gt_error_m:.1f}m",
                "path": gt_path,
                "tile_id": gt_tile_id,
                "center_error_m": gt_error_m,
                "selection_method": gt_method,
            },
        ]

        if args.include_s4b_false:
            false_tile_id = find_optional_s4b_false_tile(token, gt_tile_id)
            if false_tile_id is not None:
                false_row = find_tile_by_id(sat_df, false_tile_id)
                if false_row is not None:
                    false_path = get_row_path(false_row, sat_df, filename_index, fallback_sat_dirs, kind="sat")
                    if false_path is not None:
                        false_err = tile_center_error(false_row, sat_df, uav_lon, uav_lat)
                        sources.append(
                            {
                                "role": "optional_s4b_false_positive",
                                "label": f"S4B false tile {false_tile_id}\nerr={false_err:.1f}m",
                                "path": false_path,
                                "tile_id": false_tile_id,
                                "center_error_m": false_err,
                                "selection_method": "s4b_ranked_csv_top_false",
                            }
                        )

        for source in sources:
            try:
                macro = macro_contour_pipeline(
                    image_path=source["path"],
                    resize_size=args.resize_size,
                    preprocess=args.preprocess,
                    edge_method=args.edge_method,
                    blur_ksize=args.blur_ksize,
                    threshold_mode=args.threshold_mode,
                    threshold_percentile=args.threshold_percentile,
                    close_ksize=args.close_ksize,
                    open_ksize=args.open_ksize,
                    min_component_area=args.min_component_area,
                )
            except Exception as e:
                print(f"[WARN] Token {token} {source['role']}: failed macro pipeline: {e}")
                continue

            panel_items.append(
                {
                    "label": source["label"],
                    "macro": macro,
                }
            )

            row_out = {
                "sequence": args.sequence,
                "token": token,
                "role": source["role"],
                "tile_id": source["tile_id"],
                "image_path": str(source["path"]),
                "uav_lon": uav_lon,
                "uav_lat": uav_lat,
                "center_error_m": source["center_error_m"],
                "selection_method": source["selection_method"],
            }
            row_out.update(macro.stats)
            manifest_rows.append(row_out)

        if len(panel_items) == 0:
            print(f"[WARN] Token {token}: no panel items generated.")
            continue

        fig_path = OUT_FIG_DIR / f"s4c0_token{token:04d}_{args.preprocess}_{args.edge_method}_r{args.resize_size}_macrocontour_preflight.png"
        render_panel(token, panel_items, fig_path)
        panel_paths.append(str(fig_path))
        print(f"[OK] Token {token}: saved {fig_path}")

    manifest_path = OUT_META_DIR / "s4c0_preflight_manifest.csv"
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)

    summary = {
        "stage": "S4C.0_macrocontour_preflight",
        "sequence": args.sequence,
        "tokens_requested": tokens,
        "tokens_processed": sorted(set(int(r["token"]) for r in manifest_rows)) if manifest_rows else [],
        "num_manifest_rows": len(manifest_rows),
        "panel_paths": panel_paths,
        "parameters": vars(args),
        "uav_index": str(uav_index_path),
        "sat_index": str(sat_index_path),
        "notes": [
            "Diagnostic only.",
            "UAV lon/lat is used only for GT/nearest-tile visualization and evaluation.",
            "No retrieval ranking or benchmark is performed in S4C.0.",
        ],
    }

    summary_path = OUT_REPORT_DIR / "s4c0_preflight_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("")
    print("S4C.0 macro-contour preflight complete")
    print("--------------------------------------")
    print(f"Manifest: {manifest_path}")
    print(f"Summary:  {summary_path}")
    print(f"Figures:  {OUT_FIG_DIR}")
    print(f"Panels:   {len(panel_paths)}")


if __name__ == "__main__":
    main()
