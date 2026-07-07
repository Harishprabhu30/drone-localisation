#!/usr/bin/env python3
"""
S4B.1b — GT 3x3 neighborhood + top-k structural failure diagnostics.

Reads an existing S4B.1 ranked result CSV, then visualizes:
1. UAV query + GT-centered 3x3 satellite neighborhood.
2. UAV query + top-k retrieved tiles.
3. Structural decomposition panel for:
   - query
   - GT 3x3 tiles
   - top-k retrieved tiles

This does not run retrieval again.
UAV lon/lat is used only for evaluation/debug labeling.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path.cwd()

DEFAULT_UAV_INDEX = Path("outputs/satloc/metadata/uav_frames_index_enriched.csv")
DEFAULT_SAT_INDEX = Path("outputs/satloc/metadata/satellite_tiles_index_enriched.csv")
DEFAULT_UAV_DIR = Path("data/raw/satloc/part_1/UAV Data/traj01")
DEFAULT_SAT_DIR = Path("data/raw/satloc/part_1/Satellite Data/sat_image_ref")

META_DIR = Path("outputs/satloc/metadata/s4b_structural_retrieval")
REPORT_DIR = Path("outputs/satloc/reports/s4b_structural_retrieval")
FIG_DIR = Path("outputs/satloc/figures/s4b_structural_retrieval/s4b1b_gt_neighborhood")


def norm_id(value) -> str:
    if pd.isna(value):
        return ""
    s = str(value).strip()
    try:
        f = float(s)
        if f.is_integer():
            return str(int(f))
    except Exception:
        pass
    return s


def find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    for cand in candidates:
        cand_l = cand.lower()
        for col in df.columns:
            if cand_l in col.lower():
                return col
    return None


def find_path_col(df: pd.DataFrame) -> Optional[str]:
    return find_col(df, [
        "image_path", "file_path", "filepath", "path",
        "uav_path", "sat_path", "tile_path",
        "image_file", "filename", "file", "name",
        "tile_filename", "sat_filename"
    ])


def parse_token_from_filename(path_like: str) -> str:
    name = Path(str(path_like)).name
    if "@" in name:
        return norm_id(name.split("@")[0])
    nums = re.findall(r"\d+", Path(name).stem)
    return norm_id(nums[0]) if nums else ""


def parse_numeric_id_from_filename(path_like: str) -> str:
    stem = Path(str(path_like)).stem
    if stem.isdigit():
        return norm_id(stem)
    nums = re.findall(r"\d+", stem)
    return norm_id(nums[-1]) if nums else ""


def resolve_path(value, base_dirs: List[Path]) -> Path:
    raw = str(value).strip()
    p = Path(raw)
    candidates = []

    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.append(ROOT / p)
        for base in base_dirs:
            candidates.append(base / p)
            candidates.append(base / p.name)

    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


def bbox_cols(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    return {
        "min_lon": find_col(df, ["min_lon", "lon_min", "bbox_min_lon", "west", "left_lon", "tile_min_lon"]),
        "max_lon": find_col(df, ["max_lon", "lon_max", "bbox_max_lon", "east", "right_lon", "tile_max_lon"]),
        "min_lat": find_col(df, ["min_lat", "lat_min", "bbox_min_lat", "south", "bottom_lat", "tile_min_lat"]),
        "max_lat": find_col(df, ["max_lat", "lat_max", "bbox_max_lat", "north", "top_lat", "tile_max_lat"]),
    }


def center_cols(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    return {
        "lon": find_col(df, ["center_lon", "lon_center", "tile_center_lon", "centroid_lon"]),
        "lat": find_col(df, ["center_lat", "lat_center", "tile_center_lat", "centroid_lat"]),
    }


def row_bbox(row: pd.Series, bc: Dict[str, Optional[str]]) -> Optional[Tuple[float, float, float, float]]:
    if not all(bc.values()):
        return None
    try:
        a = float(row[bc["min_lon"]])
        b = float(row[bc["max_lon"]])
        c = float(row[bc["min_lat"]])
        d = float(row[bc["max_lat"]])
    except Exception:
        return None

    west, east = sorted([a, b])
    south, north = sorted([c, d])
    return west, east, south, north


def row_center(row: pd.Series, bc: Dict[str, Optional[str]], cc: Dict[str, Optional[str]]) -> Optional[Tuple[float, float]]:
    if cc["lon"] and cc["lat"]:
        try:
            return float(row[cc["lon"]]), float(row[cc["lat"]])
        except Exception:
            pass

    bbox = row_bbox(row, bc)
    if bbox:
        west, east, south, north = bbox
        return (west + east) * 0.5, (south + north) * 0.5
    return None


def contains_lonlat(row: pd.Series, bc: Dict[str, Optional[str]], lon: float, lat: float) -> bool:
    bbox = row_bbox(row, bc)
    if bbox is None:
        return False
    west, east, south, north = bbox
    return west <= lon <= east and south <= lat <= north


def approx_error_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    lat0 = math.radians((lat1 + lat2) * 0.5)
    dx = (lon2 - lon1) * 111_320.0 * math.cos(lat0)
    dy = (lat2 - lat1) * 110_540.0
    return float(math.sqrt(dx * dx + dy * dy))


def load_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def preprocess_gray(rgb: np.ndarray, variant: str) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    if variant == "luma":
        return gray
    if variant == "clahe_luma":
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(gray)
    raise ValueError(f"Unknown preprocess: {variant}")


def fit_canvas_u8(gray: np.ndarray, size: int, mode: str) -> np.ndarray:
    if mode == "none":
        return gray

    h, w = gray.shape[:2]

    if mode == "stretch":
        return cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)

    if mode == "crop":
        side = min(h, w)
        y0 = (h - side) // 2
        x0 = (w - side) // 2
        crop = gray[y0:y0 + side, x0:x0 + side]
        return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)

    if mode == "pad":
        scale = min(size / w, size / h)
        nw = max(1, int(round(w * scale)))
        nh = max(1, int(round(h * scale)))
        resized = cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_AREA)
        pad_l = (size - nw) // 2
        pad_r = size - nw - pad_l
        pad_t = (size - nh) // 2
        pad_b = size - nh - pad_t
        return cv2.copyMakeBorder(resized, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_REPLICATE)

    raise ValueError(f"Unknown resize mode: {mode}")


def normalize_u8(arr: np.ndarray, p_low: float = 1.0, p_high: float = 99.0) -> np.ndarray:
    arr = arr.astype(np.float32)
    lo, hi = np.percentile(arr, [p_low, p_high])
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.uint8)
    out = np.clip((arr - lo) / (hi - lo), 0, 1)
    return (out * 255).astype(np.uint8)


def gradient_debug(rgb: np.ndarray, preprocess: str, resize_mode: str, resize_size: int, cells: int, bins: int):
    gray = preprocess_gray(rgb, preprocess)
    desc_gray = fit_canvas_u8(gray, resize_size, resize_mode)

    g = cv2.GaussianBlur(desc_gray.astype(np.float32) / 255.0, (3, 3), 0)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)

    mag = np.sqrt(gx * gx + gy * gy)
    ori = np.mod(np.arctan2(gy, gx), np.pi)

    sobel = normalize_u8(mag)

    hsv = np.zeros((mag.shape[0], mag.shape[1], 3), dtype=np.uint8)
    hsv[..., 0] = np.clip((ori / np.pi) * 179.0, 0, 179).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = sobel
    ori_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    h, w = mag.shape
    cell_h = h / cells
    cell_w = w / cells
    xs, ys, us, vs, strength = [], [], [], [], []

    for cy in range(cells):
        for cx in range(cells):
            y0 = int(round(cy * cell_h))
            y1 = int(round((cy + 1) * cell_h))
            x0 = int(round(cx * cell_w))
            x1 = int(round((cx + 1) * cell_w))

            co = ori[y0:y1, x0:x1].reshape(-1)
            cm = mag[y0:y1, x0:x1].reshape(-1)
            hist, edges = np.histogram(co, bins=bins, range=(0, np.pi), weights=cm)

            b = int(np.argmax(hist))
            theta = 0.5 * (edges[b] + edges[b + 1])

            xs.append((x0 + x1) * 0.5)
            ys.append((y0 + y1) * 0.5)
            us.append(math.cos(theta))
            vs.append(math.sin(theta))
            strength.append(float(hist[b]))

    strength = np.array(strength, dtype=np.float32)
    if strength.max() > 1e-8:
        strength = strength / strength.max()

    return {
        "gray": desc_gray,
        "sobel": sobel,
        "ori_rgb": ori_rgb,
        "xs": np.array(xs),
        "ys": np.array(ys),
        "us": np.array(us),
        "vs": np.array(vs),
        "strength": strength,
    }


def find_gt_tile(sat_df, bc, cc, lon, lat):
    candidates = []
    for _, row in sat_df.iterrows():
        if contains_lonlat(row, bc, lon, lat):
            cen = row_center(row, bc, cc)
            err = approx_error_m(lon, lat, cen[0], cen[1]) if cen else 1e9
            candidates.append((err, row))

    if candidates:
        candidates.sort(key=lambda x: x[0])
        return candidates[0][1], "bbox_contains"

    best = None
    best_err = 1e18
    for _, row in sat_df.iterrows():
        cen = row_center(row, bc, cc)
        if not cen:
            continue
        err = approx_error_m(lon, lat, cen[0], cen[1])
        if err < best_err:
            best_err = err
            best = row
    return best, "nearest_center"


def build_gt_3x3(sat_df, gt_row, bc, cc):
    gt_center = row_center(gt_row, bc, cc)
    gt_bbox = row_bbox(gt_row, bc)

    if gt_center is None:
        raise RuntimeError("GT tile center missing.")

    gt_lon, gt_lat = gt_center

    if gt_bbox:
        west, east, south, north = gt_bbox
        step_lon = abs(east - west)
        step_lat = abs(north - south)
    else:
        centers = []
        for _, row in sat_df.iterrows():
            c = row_center(row, bc, cc)
            if c:
                centers.append(c)
        lons = np.array(sorted(set(round(c[0], 9) for c in centers)))
        lats = np.array(sorted(set(round(c[1], 9) for c in centers)))
        step_lon = float(np.median(np.diff(lons))) if len(lons) > 1 else 0.0001
        step_lat = float(np.median(np.diff(lats))) if len(lats) > 1 else 0.0001

    selected = {}

    for oy in [-1, 0, 1]:
        for ox in [-1, 0, 1]:
            target_lon = gt_lon + ox * step_lon
            target_lat = gt_lat + oy * step_lat

            best_idx = None
            best_err = 1e18

            for idx, row in sat_df.iterrows():
                c = row_center(row, bc, cc)
                if not c:
                    continue
                err = approx_error_m(target_lon, target_lat, c[0], c[1])
                if err < best_err:
                    best_err = err
                    best_idx = idx

            selected[(ox, oy)] = sat_df.loc[best_idx]

    return selected


def annotate_for_tile(tile_id: str, ranked_lookup: Dict[str, dict], eval_lon: float, eval_lat: float, row, bc, cc):
    r = ranked_lookup.get(norm_id(tile_id), {})
    cen = row_center(row, bc, cc)
    err = approx_error_m(eval_lon, eval_lat, cen[0], cen[1]) if cen else np.nan

    rank = r.get("rank", None)
    dist = r.get("distance", None)
    sim = r.get("similarity", None)

    return {
        "rank": rank,
        "distance": dist,
        "similarity": sim,
        "center_error_m": err,
        "contains_gt": contains_lonlat(row, bc, eval_lon, eval_lat),
    }


def save_rgb_grid(query_path, gt_grid, ranked_lookup, sat_path_col, eval_lon, eval_lat, bc, cc, sat_dir, out_path):
    fig, axes = plt.subplots(3, 4, figsize=(16, 11))

    qrgb = load_rgb(query_path)
    axes[0, 0].imshow(qrgb)
    axes[0, 0].set_title("QUERY UAV")
    axes[0, 0].axis("off")

    axes[1, 0].axis("off")
    axes[2, 0].axis("off")

    # Display north/up row first: oy=1,0,-1
    rows = [1, 0, -1]
    cols = [-1, 0, 1]

    for r_i, oy in enumerate(rows):
        for c_i, ox in enumerate(cols):
            ax = axes[r_i, c_i + 1]
            row = gt_grid[(ox, oy)]
            tile_id = norm_id(row["_tile_id_norm"])
            path = resolve_path(row[sat_path_col], [sat_dir])
            rgb = load_rgb(path)

            ann = annotate_for_tile(tile_id, ranked_lookup, eval_lon, eval_lat, row, bc, cc)
            rank_txt = ann["rank"] if ann["rank"] is not None else "not ranked?"

            ax.imshow(rgb)
            ax.axis("off")
            ax.set_title(f"GT 3x3 ox={ox}, oy={oy}\ntile {tile_id}", fontsize=9)
            ax.text(
                0.02, 0.98,
                f"rank {rank_txt}\n"
                f"dist {ann['distance']:.4f}" if ann["distance"] is not None else f"rank {rank_txt}\ndist NA",
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=8,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.80),
            )
            ax.text(
                0.02, 0.10,
                f"err {ann['center_error_m']:.1f}m\nGT {ann['contains_gt']}",
                transform=ax.transAxes,
                va="bottom",
                ha="left",
                fontsize=8,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.80),
            )

    fig.suptitle("S4B.1b — UAV query + GT-centered 3x3 satellite neighborhood", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def save_topk_rgb_panel(query_path, topk_df, out_path):
    items = [("QUERY UAV", query_path, "")]
    for _, row in topk_df.iterrows():
        note = (
            f"rank {int(row['rank'])}\n"
            f"tile {row['tile_id']}\n"
            f"dist {float(row['distance']):.4f}\n"
            f"err {float(row['center_error_m']):.1f}m\n"
            f"GT {bool(row['contains_gt'])}"
        )
        items.append((f"top {int(row['rank'])}", Path(row["tile_path"]), note))

    cols = 4
    nrows = math.ceil(len(items) / cols)
    fig, axes = plt.subplots(nrows, cols, figsize=(16, 4 * nrows), squeeze=False)
    axes = axes.reshape(-1)

    for ax in axes:
        ax.axis("off")

    for ax, (title, path, note) in zip(axes, items):
        ax.imshow(load_rgb(path))
        ax.set_title(title, fontsize=9)
        ax.axis("off")
        if note:
            ax.text(
                0.02, 0.98, note,
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=8,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.80),
            )

    fig.suptitle("S4B.1b — top-k retrieved tiles", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def save_structure_panel(items, preprocess, resize_mode, resize_size, cells, bins, out_path):
    nrows = len(items)
    ncols = 5

    fig, axes = plt.subplots(nrows, ncols, figsize=(21, 3.4 * nrows), squeeze=False)

    for r, item in enumerate(items):
        label = item["label"]
        path = item["path"]
        note = item.get("note", "")

        rgb = load_rgb(path)
        dbg = gradient_debug(rgb, preprocess, resize_mode, resize_size, cells, bins)

        axes[r, 0].imshow(rgb)
        axes[r, 0].set_title(label, fontsize=8)
        if note:
            axes[r, 0].text(
                0.02, 0.98, note,
                transform=axes[r, 0].transAxes,
                va="top",
                ha="left",
                fontsize=7,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.80),
            )

        axes[r, 1].imshow(dbg["gray"], cmap="gray")
        axes[r, 1].set_title("descriptor luma", fontsize=8)

        axes[r, 2].imshow(dbg["sobel"], cmap="gray")
        axes[r, 2].set_title("Sobel magnitude", fontsize=8)

        axes[r, 3].imshow(dbg["ori_rgb"])
        axes[r, 3].set_title("gradient orientation", fontsize=8)

        axes[r, 4].imshow(dbg["sobel"], cmap="gray")
        scale = 0.45 * max(dbg["gray"].shape) / max(cells, 1)
        axes[r, 4].quiver(
            dbg["xs"],
            dbg["ys"],
            dbg["us"] * dbg["strength"] * scale,
            -dbg["vs"] * dbg["strength"] * scale,
            angles="xy",
            scale_units="xy",
            scale=1,
            width=0.004,
        )
        axes[r, 4].set_title("HOG dominant cell dirs", fontsize=8)

        for c in range(ncols):
            axes[r, c].axis("off")

    fig.suptitle("S4B.1b — structural failure/evidence panel", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=135)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", default="traj01")
    parser.add_argument("--token", required=True)
    parser.add_argument("--ranked-csv", default=None)
    parser.add_argument("--preprocess", choices=["luma", "clahe_luma"], default="luma")
    parser.add_argument("--descriptor-type", default="hog_edge")
    parser.add_argument("--resize-mode", choices=["stretch", "pad", "crop", "none"], default="crop")
    parser.add_argument("--resize-size", type=int, default=512)
    parser.add_argument("--cells", type=int, default=8)
    parser.add_argument("--bins", type=int, default=9)
    parser.add_argument("--edge-pool-size", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--structure-top-k", type=int, default=5)
    parser.add_argument("--uav-index", default=str(DEFAULT_UAV_INDEX))
    parser.add_argument("--sat-index", default=str(DEFAULT_SAT_INDEX))
    parser.add_argument("--uav-dir", default=str(DEFAULT_UAV_DIR))
    parser.add_argument("--sat-dir", default=str(DEFAULT_SAT_DIR))
    args = parser.parse_args()

    META_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    token = norm_id(args.token)
    token_int = int(token)

    if args.ranked_csv is None:
        ranked_csv = META_DIR / (
            f"s4b1_token{token_int:04d}_{args.preprocess}_{args.descriptor_type}_"
            f"mode{args.resize_mode}_r{args.resize_size}_c{args.cells}_b{args.bins}_e{args.edge_pool_size}_ranked_results.csv"
        )
    else:
        ranked_csv = Path(args.ranked_csv)

    if not ranked_csv.exists():
        raise FileNotFoundError(f"Missing ranked CSV: {ranked_csv}")

    uav_df = pd.read_csv(args.uav_index)
    sat_df = pd.read_csv(args.sat_index)
    ranked = pd.read_csv(ranked_csv)

    uav_path_col = find_path_col(uav_df)
    sat_path_col = find_path_col(sat_df)

    seq_col = find_col(uav_df, ["sequence", "seq", "traj", "trajectory"])
    if seq_col is not None:
        mask = uav_df[seq_col].astype(str).str.lower().eq(args.sequence.lower())
        if mask.any():
            uav_df = uav_df[mask].copy()

    token_col = find_col(uav_df, ["token0_id", "token", "frame_token", "frame_id", "id", "uav_id"])
    if token_col:
        uav_df["_token_norm"] = uav_df[token_col].map(norm_id)
    else:
        uav_df["_token_norm"] = uav_df[uav_path_col].map(parse_token_from_filename)

    lon_col = find_col(uav_df, ["longitude", "lon", "uav_lon", "gt_lon", "label_lon"])
    lat_col = find_col(uav_df, ["latitude", "lat", "uav_lat", "gt_lat", "label_lat"])

    tile_id_col = find_col(sat_df, ["tile_id", "sat_tile_id", "sat_id", "tile_index", "ref_id", "id"])
    if tile_id_col:
        sat_df["_tile_id_norm"] = sat_df[tile_id_col].map(norm_id)
    else:
        sat_df["_tile_id_norm"] = sat_df[sat_path_col].map(parse_numeric_id_from_filename)

    ranked["tile_id"] = ranked["tile_id"].map(norm_id)
    ranked_lookup = {
        norm_id(r["tile_id"]): r.to_dict()
        for _, r in ranked.iterrows()
    }

    qrows = uav_df[uav_df["_token_norm"].eq(token)]
    if qrows.empty:
        raise RuntimeError(f"Query token not found: {token}")

    qrow = qrows.iloc[0]
    qpath = resolve_path(qrow[uav_path_col], [Path(args.uav_dir)])
    qlon = float(qrow[lon_col])
    qlat = float(qrow[lat_col])

    bc = bbox_cols(sat_df)
    cc = center_cols(sat_df)

    gt_row, gt_method = find_gt_tile(sat_df, bc, cc, qlon, qlat)
    gt_tile_id = norm_id(gt_row["_tile_id_norm"])

    gt_grid = build_gt_3x3(sat_df, gt_row, bc, cc)

    run_name = (
        f"token{token_int:04d}_{args.preprocess}_{args.descriptor_type}_"
        f"mode{args.resize_mode}_r{args.resize_size}_c{args.cells}_b{args.bins}_e{args.edge_pool_size}"
    )

    gt_grid_png = FIG_DIR / f"s4b1b_{run_name}_gt3x3_rgb.png"
    topk_png = FIG_DIR / f"s4b1b_{run_name}_top{args.top_k}_rgb.png"
    structure_png = FIG_DIR / f"s4b1b_{run_name}_structure_gt3x3_top{args.structure_top_k}.png"
    neigh_csv = META_DIR / f"s4b1b_{run_name}_gt3x3_neighborhood.csv"
    summary_json = REPORT_DIR / f"s4b1b_{run_name}_summary.json"

    save_rgb_grid(qpath, gt_grid, ranked_lookup, sat_path_col, qlon, qlat, bc, cc, Path(args.sat_dir), gt_grid_png)

    topk_df = ranked.head(args.top_k).copy()
    save_topk_rgb_panel(qpath, topk_df, topk_png)

    neigh_rows = []
    structure_items = [{"label": f"QUERY token {token}", "path": qpath, "note": ""}]

    for oy in [1, 0, -1]:
        for ox in [-1, 0, 1]:
            row = gt_grid[(ox, oy)]
            tid = norm_id(row["_tile_id_norm"])
            path = resolve_path(row[sat_path_col], [Path(args.sat_dir)])
            ann = annotate_for_tile(tid, ranked_lookup, qlon, qlat, row, bc, cc)
            neigh_rows.append({
                "query_token": token,
                "offset_x": ox,
                "offset_y": oy,
                "tile_id": tid,
                "tile_path": str(path),
                **ann,
            })

            structure_items.append({
                "label": f"GT3x3 ox={ox}, oy={oy}, tile {tid}",
                "path": path,
                "note": (
                    f"rank {ann['rank']}\n"
                    f"dist {ann['distance']:.4f}" if ann["distance"] is not None else "dist NA"
                ) + f"\nerr {ann['center_error_m']:.1f}m\nGT {ann['contains_gt']}",
            })

    for _, r in ranked.head(args.structure_top_k).iterrows():
        structure_items.append({
            "label": f"TOP rank {int(r['rank'])}, tile {r['tile_id']}",
            "path": Path(r["tile_path"]),
            "note": (
                f"dist {float(r['distance']):.4f}\n"
                f"err {float(r['center_error_m']):.1f}m\n"
                f"GT {bool(r['contains_gt'])}"
            ),
        })

    save_structure_panel(
        structure_items,
        preprocess=args.preprocess,
        resize_mode=args.resize_mode,
        resize_size=args.resize_size,
        cells=args.cells,
        bins=args.bins,
        out_path=structure_png,
    )

    neigh_df = pd.DataFrame(neigh_rows)
    neigh_df.to_csv(neigh_csv, index=False)

    top_under_40 = ranked[ranked["center_error_m"] <= 40.0]
    top_under_60 = ranked[ranked["center_error_m"] <= 60.0]

    summary = {
        "stage": "S4B.1b",
        "query_token": token,
        "ranked_csv": str(ranked_csv),
        "gt_tile_id": gt_tile_id,
        "gt_method": gt_method,
        "gt3x3_csv": str(neigh_csv),
        "gt3x3_rgb_panel": str(gt_grid_png),
        "topk_rgb_panel": str(topk_png),
        "structure_panel": str(structure_png),
        "top1_tile": norm_id(ranked.iloc[0]["tile_id"]),
        "top1_error_m": float(ranked.iloc[0]["center_error_m"]),
        "first_rank_under_40m": int(top_under_40.iloc[0]["rank"]) if len(top_under_40) else None,
        "first_rank_under_60m": int(top_under_60.iloc[0]["rank"]) if len(top_under_60) else None,
        "best_top10_error_m": float(ranked.head(args.top_k)["center_error_m"].min()),
        "important_rule": "UAV lon/lat used only for evaluation/debug."
    }

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("S4B.1b GT-neighborhood failure diagnostics complete")
    print("---------------------------------------------------")
    print(f"Token:                  {token}")
    print(f"Ranked CSV:             {ranked_csv}")
    print(f"GT tile:                {gt_tile_id} ({gt_method})")
    print(f"Top-1 tile/error:       {summary['top1_tile']} / {summary['top1_error_m']:.2f} m")
    print(f"First rank under 40 m:  {summary['first_rank_under_40m']}")
    print(f"First rank under 60 m:  {summary['first_rank_under_60m']}")
    print(f"Best top-{args.top_k} error:    {summary['best_top10_error_m']:.2f} m")
    print(f"GT 3x3 RGB panel:       {gt_grid_png}")
    print(f"Top-k RGB panel:        {topk_png}")
    print(f"Structure panel:        {structure_png}")
    print(f"GT 3x3 CSV:             {neigh_csv}")
    print(f"Summary JSON:           {summary_json}")


if __name__ == "__main__":
    main()
