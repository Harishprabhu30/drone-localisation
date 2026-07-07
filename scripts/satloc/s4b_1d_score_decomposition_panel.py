#!/usr/bin/env python3
'''
export PYTHONPATH=$PWD/src

python scripts/satloc/s4b_1d_score_decomposition_panel.py \
  --sequence traj01 \
  --token 1 \
  --top-k 10 \
  --panel-top-k 5 \
  --preprocess luma \
  --descriptor-type hog_edge \
  --resize-mode crop \
  --resize-size 512 \
  --cells 8 \
  --bins 9 \
  --edge-pool-size 32
'''

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
FIG_DIR = Path("outputs/satloc/figures/s4b_structural_retrieval/s4b1d_score_decomposition")


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
    p = Path(str(value).strip())
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


def l2norm(v: np.ndarray) -> np.ndarray:
    v = v.astype(np.float32).reshape(-1)
    n = np.linalg.norm(v)
    if n <= 1e-8:
        return v * 0.0
    return v / n


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    aa = l2norm(a)
    bb = l2norm(b)
    return float(np.dot(aa, bb))


def extract_features(rgb: np.ndarray, preprocess: str, resize_mode: str, resize_size: int, cells: int, bins: int, edge_pool_size: int) -> Dict:
    gray0 = preprocess_gray(rgb, preprocess)
    gray = fit_canvas_u8(gray0, resize_size, resize_mode)

    g = cv2.GaussianBlur(gray.astype(np.float32) / 255.0, (3, 3), 0)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)

    mag = np.sqrt(gx * gx + gy * gy)
    ori = np.mod(np.arctan2(gy, gx), np.pi)

    H = np.zeros((cells, cells, bins), dtype=np.float32)
    h, w = mag.shape
    cell_h = h / cells
    cell_w = w / cells

    for cy in range(cells):
        for cx in range(cells):
            y0 = int(round(cy * cell_h))
            y1 = int(round((cy + 1) * cell_h))
            x0 = int(round(cx * cell_w))
            x1 = int(round((cx + 1) * cell_w))

            co = ori[y0:y1, x0:x1].reshape(-1)
            cm = mag[y0:y1, x0:x1].reshape(-1)
            hist, _ = np.histogram(co, bins=bins, range=(0.0, np.pi), weights=cm)
            H[cy, cx, :] = hist.astype(np.float32)

    Hn = l2norm(H)

    edge_small = cv2.resize(mag, (edge_pool_size, edge_pool_size), interpolation=cv2.INTER_AREA)
    En = l2norm(edge_small)

    Cn = l2norm(np.concatenate([Hn.reshape(-1), En.reshape(-1)]))

    sobel_u8 = normalize_u8(mag)

    return {
        "gray": gray,
        "mag": mag,
        "sobel_u8": sobel_u8,
        "hog": H,
        "hog_norm": Hn.reshape(cells, cells, bins),
        "edge_norm": En.reshape(edge_pool_size, edge_pool_size),
        "combined_norm": Cn,
    }


def per_cell_hog_similarity(qH: np.ndarray, cH: np.ndarray) -> np.ndarray:
    cells_y, cells_x, _ = qH.shape
    out = np.zeros((cells_y, cells_x), dtype=np.float32)

    for y in range(cells_y):
        for x in range(cells_x):
            out[y, x] = cosine(qH[y, x, :], cH[y, x, :])

    return out


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


def save_panel(query_path: Path, qfeat: Dict, candidate_rows: List[Dict], out_path: Path, title: str):
    nrows = 1 + len(candidate_rows)
    ncols = 5

    fig, axes = plt.subplots(nrows, ncols, figsize=(22, 3.5 * nrows), squeeze=False)

    qrgb = load_rgb(query_path)

    axes[0, 0].imshow(qrgb)
    axes[0, 0].set_title("QUERY RGB")
    axes[0, 1].imshow(qfeat["gray"], cmap="gray")
    axes[0, 1].set_title("query descriptor luma")
    axes[0, 2].imshow(qfeat["sobel_u8"], cmap="gray")
    axes[0, 2].set_title("query Sobel")
    axes[0, 3].imshow(np.ones((8, 8)), vmin=0, vmax=1)
    axes[0, 3].set_title("per-cell HOG sim\nquery vs query")
    axes[0, 4].axis("off")
    axes[0, 4].text(0, 0.9, "Reference query features", fontsize=11)

    for c in range(ncols):
        axes[0, c].axis("off")

    for r, item in enumerate(candidate_rows, start=1):
        rgb = load_rgb(Path(item["tile_path"]))
        cfeat = item["features"]

        absdiff = np.abs(
            qfeat["sobel_u8"].astype(np.float32) / 255.0
            - cfeat["sobel_u8"].astype(np.float32) / 255.0
        )

        cell_sim = per_cell_hog_similarity(qfeat["hog_norm"], cfeat["hog_norm"])

        axes[r, 0].imshow(rgb)
        axes[r, 0].set_title(item["label"], fontsize=8)

        axes[r, 1].imshow(cfeat["sobel_u8"], cmap="gray")
        axes[r, 1].set_title("candidate Sobel", fontsize=8)

        axes[r, 2].imshow(absdiff, cmap="gray", vmin=0, vmax=1)
        axes[r, 2].set_title("abs Sobel diff\nwhite = different", fontsize=8)

        im = axes[r, 3].imshow(cell_sim, vmin=0, vmax=1)
        axes[r, 3].set_title("per-cell HOG similarity", fontsize=8)
        fig.colorbar(im, ax=axes[r, 3], fraction=0.046, pad=0.04)

        axes[r, 4].axis("off")
        axes[r, 4].text(
            0,
            0.95,
            item["note"],
            va="top",
            fontsize=9,
            family="monospace",
        )

        for c in range(4):
            axes[r, c].axis("off")

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def save_score_plot(df: pd.DataFrame, out_path: Path):
    plot_df = df.sort_values("combined_similarity", ascending=False).head(16).copy()
    labels = plot_df["label_short"].tolist()
    x = np.arange(len(labels))
    width = 0.26

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - width, plot_df["hog_similarity"], width, label="HOG")
    ax.bar(x, plot_df["edge_similarity"], width, label="Edge")
    ax.bar(x + width, plot_df["combined_similarity"], width, label="Combined")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=70, ha="right")
    ax.set_ylabel("Cosine similarity")
    ax.set_title("S4B.1d score decomposition")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
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
    parser.add_argument("--panel-top-k", type=int, default=5)
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
    ranked_lookup = {norm_id(r["tile_id"]): r.to_dict() for _, r in ranked.iterrows()}

    qrow = uav_df[uav_df["_token_norm"].eq(token)].iloc[0]
    qpath = resolve_path(qrow[uav_path_col], [Path(args.uav_dir)])
    qlon = float(qrow[lon_col])
    qlat = float(qrow[lat_col])

    bc = bbox_cols(sat_df)
    cc = center_cols(sat_df)

    gt_row, gt_method = find_gt_tile(sat_df, bc, cc, qlon, qlat)
    gt_grid = build_gt_3x3(sat_df, gt_row, bc, cc)

    qfeat = extract_features(
        load_rgb(qpath),
        preprocess=args.preprocess,
        resize_mode=args.resize_mode,
        resize_size=args.resize_size,
        cells=args.cells,
        bins=args.bins,
        edge_pool_size=args.edge_pool_size,
    )

    candidates = []

    for oy in [1, 0, -1]:
        for ox in [-1, 0, 1]:
            row = gt_grid[(ox, oy)]
            tid = norm_id(row["_tile_id_norm"])
            path = resolve_path(row[sat_path_col], [Path(args.sat_dir)])
            r = ranked_lookup.get(tid, {})
            candidates.append({
                "candidate_type": "gt3x3",
                "offset_x": ox,
                "offset_y": oy,
                "tile_id": tid,
                "tile_path": str(path),
                "rank": r.get("rank"),
                "original_distance": r.get("distance"),
                "original_similarity": r.get("similarity"),
                "row_obj": row,
            })

    for _, r in ranked.head(args.top_k).iterrows():
        tid = norm_id(r["tile_id"])
        if any(c["tile_id"] == tid for c in candidates):
            continue

        srow = sat_df[sat_df["_tile_id_norm"].eq(tid)]
        if srow.empty:
            continue

        candidates.append({
            "candidate_type": "topk",
            "offset_x": None,
            "offset_y": None,
            "tile_id": tid,
            "tile_path": str(Path(r["tile_path"])),
            "rank": int(r["rank"]),
            "original_distance": float(r["distance"]),
            "original_similarity": float(r["similarity"]),
            "row_obj": srow.iloc[0],
        })

    output_rows = []
    panel_items = []

    for cand in candidates:
        rgb = load_rgb(Path(cand["tile_path"]))
        feat = extract_features(
            rgb,
            preprocess=args.preprocess,
            resize_mode=args.resize_mode,
            resize_size=args.resize_size,
            cells=args.cells,
            bins=args.bins,
            edge_pool_size=args.edge_pool_size,
        )

        hog_sim = cosine(qfeat["hog_norm"], feat["hog_norm"])
        edge_sim = cosine(qfeat["edge_norm"], feat["edge_norm"])
        combined_sim = cosine(qfeat["combined_norm"], feat["combined_norm"])
        sobel_diff_mean = float(np.mean(np.abs(qfeat["sobel_u8"].astype(np.float32) - feat["sobel_u8"].astype(np.float32)) / 255.0))

        cen = row_center(cand["row_obj"], bc, cc)
        err = approx_error_m(qlon, qlat, cen[0], cen[1]) if cen else np.nan
        contains = contains_lonlat(cand["row_obj"], bc, qlon, qlat)

        label_short = f"{cand['candidate_type']}_{cand['tile_id']}"
        if cand["candidate_type"] == "gt3x3":
            label = f"GT3x3 ox={cand['offset_x']} oy={cand['offset_y']} tile {cand['tile_id']}"
        else:
            label = f"TOP rank={cand['rank']} tile {cand['tile_id']}"

        note = (
            f"type: {cand['candidate_type']}\n"
            f"tile: {cand['tile_id']}\n"
            f"rank: {cand['rank']}\n"
            f"err_m: {err:.1f}\n"
            f"contains_gt: {contains}\n"
            f"HOG sim: {hog_sim:.4f}\n"
            f"edge sim: {edge_sim:.4f}\n"
            f"combined: {combined_sim:.4f}\n"
            f"Sobel diff mean: {sobel_diff_mean:.4f}"
        )

        row_out = {
            "query_token": token,
            "candidate_type": cand["candidate_type"],
            "offset_x": cand["offset_x"],
            "offset_y": cand["offset_y"],
            "tile_id": cand["tile_id"],
            "tile_path": cand["tile_path"],
            "rank": cand["rank"],
            "original_distance": cand["original_distance"],
            "original_similarity": cand["original_similarity"],
            "center_error_m": float(err),
            "contains_gt": bool(contains),
            "hog_similarity": hog_sim,
            "edge_similarity": edge_sim,
            "combined_similarity": combined_sim,
            "sobel_absdiff_mean": sobel_diff_mean,
            "label_short": label_short,
        }
        output_rows.append(row_out)

        panel_items.append({
            "label": label,
            "label_short": label_short,
            "tile_path": cand["tile_path"],
            "features": feat,
            "note": note,
            "combined_similarity": combined_sim,
            "center_error_m": float(err),
        })

    out_df = pd.DataFrame(output_rows)

    run_name = (
        f"token{token_int:04d}_{args.preprocess}_{args.descriptor_type}_"
        f"mode{args.resize_mode}_r{args.resize_size}_c{args.cells}_b{args.bins}_e{args.edge_pool_size}"
    )

    out_csv = META_DIR / f"s4b1d_{run_name}_score_decomposition.csv"
    panel_png = FIG_DIR / f"s4b1d_{run_name}_score_decomposition_panel.png"
    score_png = FIG_DIR / f"s4b1d_{run_name}_score_barplot.png"
    summary_json = REPORT_DIR / f"s4b1d_{run_name}_summary.json"

    out_df.to_csv(out_csv, index=False)

    ordered_panel = sorted(
        panel_items,
        key=lambda x: (x["center_error_m"] > 60.0, -x["combined_similarity"])
    )
    ordered_panel = ordered_panel[: max(10, args.panel_top_k + 5)]

    save_panel(
        query_path=qpath,
        qfeat=qfeat,
        candidate_rows=ordered_panel,
        out_path=panel_png,
        title=f"S4B.1d score decomposition — token {token}"
    )

    save_score_plot(out_df, score_png)

    best_combined = out_df.sort_values("combined_similarity", ascending=False).head(1).to_dict("records")[0]
    best_near = out_df.sort_values("center_error_m", ascending=True).head(1).to_dict("records")[0]
    top_ranked = ranked.head(1).to_dict("records")[0]

    summary = {
        "stage": "S4B.1d",
        "query_token": token,
        "gt_method": gt_method,
        "ranked_csv": str(ranked_csv),
        "output_csv": str(out_csv),
        "panel_png": str(panel_png),
        "score_barplot": str(score_png),
        "best_combined_candidate": best_combined,
        "best_near_candidate": best_near,
        "top_ranked_from_s4b1": top_ranked,
        "interpretation_hint": "Compare HOG sim vs edge sim. If false positives win mainly by edge/texture, forests are dominating Sobel/HOG."
    }

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("S4B.1d score decomposition complete")
    print("-----------------------------------")
    print(f"Token:                {token}")
    print(f"Ranked CSV:           {ranked_csv}")
    print(f"Candidates tested:    {len(out_df)}")
    print(f"Best combined tile:   {best_combined['tile_id']} sim {best_combined['combined_similarity']:.4f} err {best_combined['center_error_m']:.1f}m")
    print(f"Best near tile:       {best_near['tile_id']} sim {best_near['combined_similarity']:.4f} err {best_near['center_error_m']:.1f}m")
    print(f"CSV:                  {out_csv}")
    print(f"Panel:                {panel_png}")
    print(f"Score barplot:        {score_png}")
    print(f"Summary JSON:         {summary_json}")


if __name__ == "__main__":
    main()
