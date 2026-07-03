#!/usr/bin/env python3
"""
S4B.1 — SatLoc simple global structural retrieval.

Method:
- Fixed resize
- Sobel/HOG-style global descriptor
- No rotation sweep
- No scale sweep
- Full satellite tile ranking using cosine similarity
- GT lon/lat used only after ranking for evaluation/debug
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
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

BASE_OUT = Path("outputs/satloc")
META_DIR = BASE_OUT / "metadata/s4b_structural_retrieval"
REPORT_DIR = BASE_OUT / "reports/s4b_structural_retrieval"
FIG_DIR = BASE_OUT / "figures/s4b_structural_retrieval/s4b1_global_retrieval"


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
    lower_to_real = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_to_real:
            return lower_to_real[cand.lower()]
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

    for cand in candidates:
        if cand.exists():
            return cand

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


def approx_lonlat_error_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    lat0 = math.radians((lat1 + lat2) * 0.5)
    dx = (lon2 - lon1) * 111_320.0 * math.cos(lat0)
    dy = (lat2 - lat1) * 110_540.0
    return float(math.sqrt(dx * dx + dy * dy))


def load_rgb(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def preprocess_gray(rgb: np.ndarray, variant: str) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    if variant == "luma":
        return gray

    if variant == "clahe_luma":
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(gray)

    raise ValueError(f"Unknown preprocess variant: {variant}")


def make_structural_descriptor(
    rgb: np.ndarray,
    resize_size: int,
    cells: int,
    bins: int,
    preprocess: str,
    descriptor_type: str,
) -> np.ndarray:
    gray = preprocess_gray(rgb, preprocess)
    small = cv2.resize(gray, (resize_size, resize_size), interpolation=cv2.INTER_AREA)

    g = small.astype(np.float32) / 255.0
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)

    mag = np.sqrt(gx * gx + gy * gy)
    ori = np.mod(np.arctan2(gy, gx), np.pi)

    parts = []

    if descriptor_type in ("hog", "hog_edge"):
        cell_h = resize_size // cells
        cell_w = resize_size // cells
        hist_parts = []

        for cy in range(cells):
            for cx in range(cells):
                y0, y1 = cy * cell_h, (cy + 1) * cell_h
                x0, x1 = cx * cell_w, (cx + 1) * cell_w

                cell_ori = ori[y0:y1, x0:x1].reshape(-1)
                cell_mag = mag[y0:y1, x0:x1].reshape(-1)

                hist, _ = np.histogram(
                    cell_ori,
                    bins=bins,
                    range=(0.0, np.pi),
                    weights=cell_mag,
                )
                hist_parts.append(hist.astype(np.float32))

        parts.append(np.concatenate(hist_parts))

    if descriptor_type in ("edge", "hog_edge"):
        edge_small = cv2.resize(mag, (32, 32), interpolation=cv2.INTER_AREA)
        edge_small = edge_small.reshape(-1).astype(np.float32)
        parts.append(edge_small)

    if not parts:
        raise ValueError(f"Unknown descriptor_type: {descriptor_type}")

    desc = np.concatenate(parts).astype(np.float32)
    norm = np.linalg.norm(desc)
    if norm > 1e-8:
        desc = desc / norm
    return desc


def build_or_load_sat_cache(
    sat_df: pd.DataFrame,
    sat_path_col: str,
    sat_dir: Path,
    cache_path: Path,
    resize_size: int,
    cells: int,
    bins: int,
    preprocess: str,
    descriptor_type: str,
    rebuild_cache: bool,
    max_tiles: Optional[int],
) -> Tuple[np.ndarray, List[str], List[str]]:
    if cache_path.exists() and not rebuild_cache:
        data = np.load(cache_path, allow_pickle=True)
        return data["descriptors"].astype(np.float32), list(data["tile_ids"]), list(data["tile_paths"])

    tile_ids = []
    tile_paths = []
    descriptors = []

    use_df = sat_df.copy()
    if max_tiles is not None:
        use_df = use_df.head(max_tiles)

    t0 = time.time()

    for i, row in use_df.iterrows():
        tile_id = row["_tile_id_norm"]
        path = resolve_path(row[sat_path_col], [sat_dir])

        try:
            rgb = load_rgb(path)
            desc = make_structural_descriptor(
                rgb=rgb,
                resize_size=resize_size,
                cells=cells,
                bins=bins,
                preprocess=preprocess,
                descriptor_type=descriptor_type,
            )
        except Exception as e:
            print(f"WARNING: skipping tile {tile_id}: {e}")
            continue

        tile_ids.append(tile_id)
        tile_paths.append(str(path))
        descriptors.append(desc)

        if len(descriptors) % 1000 == 0:
            print(f"Cached descriptors: {len(descriptors)} / {len(use_df)}")

    if not descriptors:
        raise RuntimeError("No satellite descriptors were built.")

    desc_arr = np.vstack(descriptors).astype(np.float32)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        descriptors=desc_arr,
        tile_ids=np.array(tile_ids, dtype=object),
        tile_paths=np.array(tile_paths, dtype=object),
        config=json.dumps({
            "resize_size": resize_size,
            "cells": cells,
            "bins": bins,
            "preprocess": preprocess,
            "descriptor_type": descriptor_type,
            "max_tiles": max_tiles,
        }),
    )

    print(f"Built satellite descriptor cache in {time.time() - t0:.2f}s")
    return desc_arr, tile_ids, tile_paths


def save_topk_panel(
    uav_path: Path,
    gt_path: Optional[Path],
    rows: pd.DataFrame,
    out_path: Path,
    title: str,
) -> None:
    panel_items = [("QUERY UAV", uav_path, "")]
    if gt_path is not None:
        panel_items.append(("GT/debug tile", gt_path, ""))

    for _, row in rows.iterrows():
        note = (
            f"rank {int(row['rank'])}\n"
            f"tile {row['tile_id']}\n"
            f"sim {row['similarity']:.4f}\n"
            f"dist {row['distance']:.4f}\n"
            f"GT {bool(row['contains_gt'])}\n"
            f"err {row['center_error_m']:.1f}m"
        )
        panel_items.append((f"top {int(row['rank'])}", Path(row["tile_path"]), note))

    n = len(panel_items)
    cols = 4
    rows_n = math.ceil(n / cols)

    fig, axes = plt.subplots(rows_n, cols, figsize=(4.2 * cols, 3.8 * rows_n), squeeze=False)
    axes_flat = axes.reshape(-1)

    for ax in axes_flat:
        ax.axis("off")

    for ax, (label, path, note) in zip(axes_flat, panel_items):
        rgb = load_rgb(path)
        ax.imshow(rgb)
        ax.set_title(label, fontsize=9)
        ax.axis("off")

        if note:
            ax.text(
                0.02, 0.98, note,
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=8,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.78),
            )

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def save_rank_plot(results: pd.DataFrame, out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(results["rank"], results["distance"])
    ax.set_xlabel("rank")
    ax.set_ylabel("cosine distance lower is better")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    correct = results[results["contains_gt"] == True]
    if len(correct):
        first = correct.iloc[0]
        ax.axvline(first["rank"], linestyle="--")
        ax.text(
            first["rank"],
            float(results["distance"].min()),
            f" first correct rank {int(first['rank'])}",
            rotation=90,
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--sequence", default="traj01")
    parser.add_argument("--token", required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--resize-size", type=int, default=256)
    parser.add_argument("--cells", type=int, default=8)
    parser.add_argument("--bins", type=int, default=9)
    parser.add_argument("--preprocess", choices=["luma", "clahe_luma"], default="luma")
    parser.add_argument("--descriptor-type", choices=["hog", "edge", "hog_edge"], default="hog_edge")
    parser.add_argument("--uav-index", default=str(DEFAULT_UAV_INDEX))
    parser.add_argument("--sat-index", default=str(DEFAULT_SAT_INDEX))
    parser.add_argument("--uav-dir", default=str(DEFAULT_UAV_DIR))
    parser.add_argument("--sat-dir", default=str(DEFAULT_SAT_DIR))
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--max-tiles", type=int, default=None)
    args = parser.parse_args()

    META_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    token = norm_id(args.token)

    uav_df = pd.read_csv(args.uav_index)
    sat_df = pd.read_csv(args.sat_index)

    uav_path_col = find_path_col(uav_df)
    sat_path_col = find_path_col(sat_df)
    if uav_path_col is None or sat_path_col is None:
        raise RuntimeError("Could not infer image path columns.")

    sequence_col = find_col(uav_df, ["sequence", "seq", "traj", "trajectory"])
    if sequence_col is not None:
        seq_mask = uav_df[sequence_col].astype(str).str.lower().eq(args.sequence.lower())
        if seq_mask.any():
            uav_df = uav_df[seq_mask].copy()

    token_col = find_col(uav_df, ["token0_id", "token", "frame_token", "frame_id", "id", "uav_id"])
    if token_col:
        uav_df["_token_norm"] = uav_df[token_col].map(norm_id)
    else:
        uav_df["_token_norm"] = uav_df[uav_path_col].map(parse_token_from_filename)

    lon_col = find_col(uav_df, ["longitude", "lon", "uav_lon", "gt_lon", "label_lon"])
    lat_col = find_col(uav_df, ["latitude", "lat", "uav_lat", "gt_lat", "label_lat"])
    if lon_col is None or lat_col is None:
        raise RuntimeError("Could not infer UAV eval lon/lat columns.")

    tile_id_col = find_col(sat_df, ["tile_id", "sat_tile_id", "sat_id", "tile_index", "ref_id", "id"])
    if tile_id_col:
        sat_df["_tile_id_norm"] = sat_df[tile_id_col].map(norm_id)
    else:
        sat_df["_tile_id_norm"] = sat_df[sat_path_col].map(parse_numeric_id_from_filename)

    bc = bbox_cols(sat_df)
    cc = center_cols(sat_df)

    query_rows = uav_df[uav_df["_token_norm"].eq(token)]
    if query_rows.empty:
        raise RuntimeError(f"Token {token} not found.")

    qrow = query_rows.iloc[0]
    uav_path = resolve_path(qrow[uav_path_col], [Path(args.uav_dir)])
    q_lon = float(qrow[lon_col])
    q_lat = float(qrow[lat_col])

    cache_name = (
        f"s4b1_sat_cache_{args.preprocess}_{args.descriptor_type}_"
        f"r{args.resize_size}_c{args.cells}_b{args.bins}"
    )
    if args.max_tiles:
        cache_name += f"_max{args.max_tiles}"
    cache_path = META_DIR / f"{cache_name}.npz"

    sat_desc, tile_ids, tile_paths = build_or_load_sat_cache(
        sat_df=sat_df,
        sat_path_col=sat_path_col,
        sat_dir=Path(args.sat_dir),
        cache_path=cache_path,
        resize_size=args.resize_size,
        cells=args.cells,
        bins=args.bins,
        preprocess=args.preprocess,
        descriptor_type=args.descriptor_type,
        rebuild_cache=args.rebuild_cache,
        max_tiles=args.max_tiles,
    )

    q_rgb = load_rgb(uav_path)
    q_desc = make_structural_descriptor(
        rgb=q_rgb,
        resize_size=args.resize_size,
        cells=args.cells,
        bins=args.bins,
        preprocess=args.preprocess,
        descriptor_type=args.descriptor_type,
    )

    t0 = time.time()
    similarity = sat_desc @ q_desc
    distance = 1.0 - similarity
    order = np.argsort(distance)
    elapsed = time.time() - t0

    eval_rows = []
    sat_lookup = sat_df.set_index("_tile_id_norm", drop=False)

    for rank_idx, desc_idx in enumerate(order, start=1):
        tile_id = norm_id(tile_ids[desc_idx])
        tile_path = tile_paths[desc_idx]

        contains = False
        center_error = np.nan

        if tile_id in sat_lookup.index:
            srow = sat_lookup.loc[tile_id]
            if isinstance(srow, pd.DataFrame):
                srow = srow.iloc[0]

            contains = contains_lonlat(srow, bc, q_lon, q_lat)
            center = row_center(srow, bc, cc)
            if center is not None:
                center_error = approx_lonlat_error_m(q_lon, q_lat, center[0], center[1])

        eval_rows.append({
            "rank": rank_idx,
            "tile_id": tile_id,
            "tile_path": tile_path,
            "similarity": float(similarity[desc_idx]),
            "distance": float(distance[desc_idx]),
            "contains_gt": bool(contains),
            "center_error_m": float(center_error) if np.isfinite(center_error) else np.nan,
        })

    results = pd.DataFrame(eval_rows)

    token_int = int(token)
    run_name = (
        f"token{token_int:04d}_{args.preprocess}_{args.descriptor_type}_"
        f"r{args.resize_size}_c{args.cells}_b{args.bins}"
    )

    results_csv = META_DIR / f"s4b1_{run_name}_ranked_results.csv"
    summary_json = REPORT_DIR / f"s4b1_{run_name}_summary.json"
    topk_panel = FIG_DIR / f"s4b1_{run_name}_top{args.top_k}_panel.png"
    rank_plot = FIG_DIR / f"s4b1_{run_name}_distance_by_rank.png"

    results.to_csv(results_csv, index=False)

    topk = results.head(args.top_k).copy()
    correct = results[results["contains_gt"] == True]

    first_correct_rank = int(correct.iloc[0]["rank"]) if len(correct) else None
    closest_correct_error = float(correct["center_error_m"].min()) if len(correct) else None

    gt_path = None
    if len(correct):
        gt_tile_id = norm_id(correct.iloc[0]["tile_id"])
        gt_path = Path(correct.iloc[0]["tile_path"])

    top1 = results.iloc[0]

    summary = {
        "stage": "S4B.1",
        "sequence": args.sequence,
        "query_token": token,
        "preprocess": args.preprocess,
        "descriptor_type": args.descriptor_type,
        "resize_size": args.resize_size,
        "cells": args.cells,
        "bins": args.bins,
        "tiles_ranked": int(len(results)),
        "elapsed_ranking_s": elapsed,
        "top1_tile_id": norm_id(top1["tile_id"]),
        "top1_distance": float(top1["distance"]),
        "top1_similarity": float(top1["similarity"]),
        "top1_center_error_m": float(top1["center_error_m"]) if np.isfinite(top1["center_error_m"]) else None,
        "recall_at_1": bool(results.head(1)["contains_gt"].any()),
        "recall_at_5": bool(results.head(5)["contains_gt"].any()),
        "recall_at_10": bool(results.head(10)["contains_gt"].any()),
        "first_correct_rank": first_correct_rank,
        "closest_correct_error_m": closest_correct_error,
        "cache_path": str(cache_path),
        "results_csv": str(results_csv),
        "topk_panel": str(topk_panel),
        "rank_plot": str(rank_plot),
        "important_rule": "UAV lon/lat is used only after ranking for evaluation/debug labeling.",
    }

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    save_topk_panel(
        uav_path=uav_path,
        gt_path=gt_path,
        rows=topk,
        out_path=topk_panel,
        title=f"S4B.1 structural global retrieval — token {token}",
    )

    save_rank_plot(
        results=results,
        out_path=rank_plot,
        title=f"S4B.1 distance by rank — token {token}",
    )

    print("S4B.1 structural global retrieval complete")
    print("------------------------------------------")
    print(f"Sequence: {args.sequence}")
    print(f"Query token: {token}")
    print(f"Preprocess: {args.preprocess}")
    print(f"Descriptor: {args.descriptor_type}")
    print(f"Tiles ranked: {len(results)}")
    print(f"Top-1 tile: {summary['top1_tile_id']}")
    print(f"Top-1 error [m]: {summary['top1_center_error_m']}")
    print(f"Recall@1/5/10: {summary['recall_at_1']} / {summary['recall_at_5']} / {summary['recall_at_10']}")
    print(f"First correct rank:{summary['first_correct_rank']}")
    print(f"Closest correct err m: {summary['closest_correct_error_m']}")
    print(f"Ranking time [s]: {elapsed:.4f}")
    print(f"Results CSV: {results_csv}")
    print(f"Summary JSON: {summary_json}")
    print(f"Top-k panel: {topk_panel}")
    print(f"Rank plot: {rank_plot}")


if __name__ == "__main__":
    main()
