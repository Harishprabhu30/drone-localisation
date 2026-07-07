#!/usr/bin/env python3
"""
S4B.1c — Shift-tolerance diagnostic for HOG structural retrieval.

Purpose:
- Test whether GT-neighborhood rank variation is caused by spatial phase shift.
- Compare strict HOG dot product vs small sliding HOG cross-correlation.
- Runs only on GT 3x3 + top-k retrieved candidates, not full-map yet.

Output:
- CSV with strict score, best shifted score, best dx/dy shift.
- Heatmap figure showing shift-score surface per candidate.
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
FIG_DIR = Path("outputs/satloc/figures/s4b_structural_retrieval/s4b1c_shift_tolerance")


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


def make_hog_tensor(
    rgb: np.ndarray,
    preprocess: str,
    resize_mode: str,
    resize_size: int,
    cells: int,
    bins: int,
) -> np.ndarray:
    gray = preprocess_gray(rgb, preprocess)
    gray = fit_canvas_u8(gray, resize_size, resize_mode)

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

    n = np.linalg.norm(H)
    if n > 1e-8:
        H = H / n

    return H


def strict_similarity(A: np.ndarray, B: np.ndarray) -> float:
    a = A.reshape(-1)
    b = B.reshape(-1)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(np.dot(a / na, b / nb))


def shifted_similarity(
    A: np.ndarray,
    B: np.ndarray,
    max_shift: int,
) -> Tuple[float, int, int, np.ndarray, np.ndarray, np.ndarray]:
    """
    Slide B relative to A in cell units.
    dx > 0 means use B shifted right relative to A.
    Score uses normalized overlap dot product with overlap penalty.
    """
    cells_y, cells_x, _ = A.shape

    shifts = list(range(-max_shift, max_shift + 1))
    raw = np.zeros((len(shifts), len(shifts)), dtype=np.float32)
    penalized = np.zeros_like(raw)

    best_score = -1.0
    best_dx = 0
    best_dy = 0

    total_cells = cells_y * cells_x

    for iy, dy in enumerate(shifts):
        for ix, dx in enumerate(shifts):
            if dx >= 0:
                ax0, ax1 = 0, cells_x - dx
                bx0, bx1 = dx, cells_x
            else:
                ax0, ax1 = -dx, cells_x
                bx0, bx1 = 0, cells_x + dx

            if dy >= 0:
                ay0, ay1 = 0, cells_y - dy
                by0, by1 = dy, cells_y
            else:
                ay0, ay1 = -dy, cells_y
                by0, by1 = 0, cells_y + dy

            Apart = A[ay0:ay1, ax0:ax1, :]
            Bpart = B[by0:by1, bx0:bx1, :]

            overlap_cells = Apart.shape[0] * Apart.shape[1]
            if overlap_cells <= 0:
                score = 0.0
            else:
                score = strict_similarity(Apart, Bpart)

            penalty = math.sqrt(overlap_cells / total_cells)
            raw[iy, ix] = score
            penalized[iy, ix] = score * penalty

            if penalized[iy, ix] > best_score:
                best_score = float(penalized[iy, ix])
                best_dx = dx
                best_dy = dy

    return best_score, best_dx, best_dy, raw, penalized, np.array(shifts)


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


def save_shift_heatmap_panel(rows: List[dict], out_path: Path, max_shift: int):
    n = len(rows)
    fig, axes = plt.subplots(n, 2, figsize=(10, 3.0 * n), squeeze=False)

    for i, r in enumerate(rows):
        rgb = load_rgb(Path(r["path"]))
        axes[i, 0].imshow(rgb)
        axes[i, 0].axis("off")
        axes[i, 0].set_title(r["label"], fontsize=9)
        axes[i, 0].text(
            0.02, 0.98,
            f"rank {r.get('rank')}\n"
            f"err {r['center_error_m']:.1f}m\n"
            f"strict {r['strict_sim']:.4f}\n"
            f"shift {r['shift_sim']:.4f}\n"
            f"dx,dy {r['best_dx']},{r['best_dy']}",
            transform=axes[i, 0].transAxes,
            va="top",
            ha="left",
            fontsize=8,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.82),
        )

        im = axes[i, 1].imshow(
            r["shift_map"],
            origin="lower",
            extent=[-max_shift - 0.5, max_shift + 0.5, -max_shift - 0.5, max_shift + 0.5],
            aspect="equal",
        )
        axes[i, 1].set_title("Shift score heatmap", fontsize=9)
        axes[i, 1].set_xlabel("dx cells")
        axes[i, 1].set_ylabel("dy cells")
        axes[i, 1].scatter([r["best_dx"]], [r["best_dy"]], marker="x", s=80)
        fig.colorbar(im, ax=axes[i, 1], fraction=0.046, pad=0.04)

    fig.suptitle("S4B.1c — strict HOG vs shift-tolerant HOG diagnostic", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=145)
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
    parser.add_argument("--max-shift", type=int, default=2)
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

    qH = make_hog_tensor(
        load_rgb(qpath),
        preprocess=args.preprocess,
        resize_mode=args.resize_mode,
        resize_size=args.resize_size,
        cells=args.cells,
        bins=args.bins,
    )

    candidate_rows = []

    for oy in [1, 0, -1]:
        for ox in [-1, 0, 1]:
            row = gt_grid[(ox, oy)]
            tid = norm_id(row["_tile_id_norm"])
            path = resolve_path(row[sat_path_col], [Path(args.sat_dir)])
            r = ranked_lookup.get(tid, {})
            candidate_rows.append({
                "candidate_type": "gt3x3",
                "offset_x": ox,
                "offset_y": oy,
                "tile_id": tid,
                "path": str(path),
                "rank": r.get("rank"),
                "original_distance": r.get("distance"),
                "original_similarity": r.get("similarity"),
                "row_obj": row,
            })

    for _, r in ranked.head(args.top_k).iterrows():
        tid = norm_id(r["tile_id"])
        if any(x["tile_id"] == tid for x in candidate_rows):
            continue

        srow_match = sat_df[sat_df["_tile_id_norm"].eq(tid)]
        if srow_match.empty:
            continue

        candidate_rows.append({
            "candidate_type": "topk",
            "offset_x": None,
            "offset_y": None,
            "tile_id": tid,
            "path": str(Path(r["tile_path"])),
            "rank": int(r["rank"]),
            "original_distance": float(r["distance"]),
            "original_similarity": float(r["similarity"]),
            "row_obj": srow_match.iloc[0],
        })

    output_rows = []
    plot_rows = []

    for cand in candidate_rows:
        tid = cand["tile_id"]
        path = Path(cand["path"])
        srow = cand["row_obj"]

        sH = make_hog_tensor(
            load_rgb(path),
            preprocess=args.preprocess,
            resize_mode=args.resize_mode,
            resize_size=args.resize_size,
            cells=args.cells,
            bins=args.bins,
        )

        strict = strict_similarity(qH, sH)
        best, dx, dy, raw_map, penalized_map, shifts = shifted_similarity(qH, sH, args.max_shift)

        cen = row_center(srow, bc, cc)
        err = approx_error_m(qlon, qlat, cen[0], cen[1]) if cen else np.nan

        row_out = {
            "query_token": token,
            "candidate_type": cand["candidate_type"],
            "offset_x": cand["offset_x"],
            "offset_y": cand["offset_y"],
            "tile_id": tid,
            "tile_path": str(path),
            "rank": cand["rank"],
            "original_distance": cand["original_distance"],
            "original_similarity": cand["original_similarity"],
            "center_error_m": float(err),
            "contains_gt": contains_lonlat(srow, bc, qlon, qlat),
            "strict_hog_similarity": strict,
            "shift_hog_similarity": best,
            "shift_improvement": best - strict,
            "best_dx_cells": dx,
            "best_dy_cells": dy,
        }

        output_rows.append(row_out)

        label = f"{cand['candidate_type']} tile {tid}"
        if cand["candidate_type"] == "gt3x3":
            label += f" ox={cand['offset_x']} oy={cand['offset_y']}"
        else:
            label += f" rank={cand['rank']}"

        plot_rows.append({
            "label": label,
            "path": str(path),
            "rank": cand["rank"],
            "center_error_m": float(err),
            "strict_sim": strict,
            "shift_sim": best,
            "best_dx": dx,
            "best_dy": dy,
            "shift_map": penalized_map,
        })

    out_df = pd.DataFrame(output_rows)

    run_name = (
        f"token{token_int:04d}_{args.preprocess}_mode{args.resize_mode}_"
        f"r{args.resize_size}_c{args.cells}_b{args.bins}_shift{args.max_shift}"
    )

    out_csv = META_DIR / f"s4b1c_{run_name}_shift_diagnostic.csv"
    heatmap_png = FIG_DIR / f"s4b1c_{run_name}_shift_heatmaps.png"
    summary_json = REPORT_DIR / f"s4b1c_{run_name}_summary.json"

    out_df.to_csv(out_csv, index=False)
    save_shift_heatmap_panel(plot_rows, heatmap_png, args.max_shift)

    gt3 = out_df[out_df["candidate_type"] == "gt3x3"].copy()
    topk = out_df[out_df["candidate_type"] == "topk"].copy()

    summary = {
        "stage": "S4B.1c",
        "query_token": token,
        "gt_method": gt_method,
        "ranked_csv": str(ranked_csv),
        "max_shift_cells": args.max_shift,
        "best_gt3x3_by_strict": gt3.sort_values("strict_hog_similarity", ascending=False).head(1).to_dict("records"),
        "best_gt3x3_by_shift": gt3.sort_values("shift_hog_similarity", ascending=False).head(1).to_dict("records"),
        "best_any_by_shift": out_df.sort_values("shift_hog_similarity", ascending=False).head(1).to_dict("records"),
        "output_csv": str(out_csv),
        "heatmap_png": str(heatmap_png),
        "interpretation_hint": "If GT-neighborhood tiles gain large shift_improvement, spatial phase shift is confirmed."
    }

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("S4B.1c shift-tolerance diagnostic complete")
    print("------------------------------------------")
    print(f"Token:                {token}")
    print(f"Ranked CSV:           {ranked_csv}")
    print(f"Candidates tested:    {len(out_df)}")
    print(f"Max shift cells:      ±{args.max_shift}")
    print(f"Best GT3x3 strict:    tile {summary['best_gt3x3_by_strict'][0]['tile_id']} score {summary['best_gt3x3_by_strict'][0]['strict_hog_similarity']:.4f}")
    print(f"Best GT3x3 shifted:   tile {summary['best_gt3x3_by_shift'][0]['tile_id']} score {summary['best_gt3x3_by_shift'][0]['shift_hog_similarity']:.4f}")
    print(f"Best any shifted:     tile {summary['best_any_by_shift'][0]['tile_id']} score {summary['best_any_by_shift'][0]['shift_hog_similarity']:.4f}")
    print(f"CSV:                  {out_csv}")
    print(f"Heatmap panel:        {heatmap_png}")
    print(f"Summary JSON:         {summary_json}")


if __name__ == "__main__":
    main()
