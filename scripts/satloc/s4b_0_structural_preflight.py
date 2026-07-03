#!/usr/bin/env python3
"""
S4B.0 — SatLoc structural retrieval preflight / FOV-scale visual inspection.

Purpose:
- Compare UAV query frames with their GT-containing satellite tile.
- Optionally compare against known ORB full-map false-positive tiles.
- Visualize what a structural/global method may "see":
  RGB, grayscale/luma, Sobel magnitude, gradient orientation, fixed-resize normalized image.
- Save report CSV/JSON and visual panels.

Important:
- UAV filename lon/lat is used only for debug/evaluation tile lookup.
- No retrieval estimator is implemented here.
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

FIG_DIR = Path("outputs/satloc/figures/s4b_structural_retrieval/s4b0_preflight")
REPORT_DIR = Path("outputs/satloc/reports/s4b_structural_retrieval")

DEFAULT_ORB_FALSE_TILES = (
    "1:5471,"
    "115:5523,"
    "230:4722,"
    "345:4488,"
    "460:4791,"
    "574:3817,"
    "689:2887,"
    "804:3274,"
    "919:4220,"
    "1034:5177"
)


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


def parse_csv_list(text: str) -> List[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def parse_false_tile_map(text: str) -> Dict[str, str]:
    out = {}
    if not text:
        return out
    for item in text.split(","):
        item = item.strip()
        if not item or ":" not in item:
            continue
        token, tile = item.split(":", 1)
        out[norm_id(token)] = norm_id(tile)
    return out


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
    return find_col(
        df,
        [
            "image_path",
            "file_path",
            "filepath",
            "path",
            "uav_path",
            "sat_path",
            "tile_path",
            "image_file",
            "filename",
            "file",
            "name",
        ],
    )


def parse_token_from_filename(path_like: str) -> str:
    name = Path(str(path_like)).name
    # SatLoc UAV example: 1@0@112.816130@28.297316.png
    if "@" in name:
        return norm_id(name.split("@")[0])
    stem = Path(name).stem
    m = re.search(r"\d+", stem)
    return norm_id(m.group(0)) if m else ""


def parse_numeric_id_from_filename(path_like: str) -> str:
    stem = Path(str(path_like)).stem
    if stem.isdigit():
        return norm_id(stem)
    nums = re.findall(r"\d+", stem)
    if nums:
        return norm_id(nums[-1])
    return ""


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


def load_rgb(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def to_luma(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def normalize_u8(arr: np.ndarray, p_low: float = 1.0, p_high: float = 99.0) -> np.ndarray:
    arr = arr.astype(np.float32)
    lo, hi = np.percentile(arr, [p_low, p_high])
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.uint8)
    out = (arr - lo) / (hi - lo)
    out = np.clip(out, 0.0, 1.0)
    return (out * 255).astype(np.uint8)


def sobel_mag_ori(gray: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    g = blur.astype(np.float32) / 255.0

    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)

    mag = np.sqrt(gx * gx + gy * gy)
    ori = (np.arctan2(gy, gx) + 2.0 * np.pi) % (2.0 * np.pi)

    mag_u8 = normalize_u8(mag)
    strong_mask = mag > np.percentile(mag, 65.0)

    return mag_u8, ori, strong_mask


def resized_norm(gray: np.ndarray, size: int) -> np.ndarray:
    small = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    return cv2.normalize(small, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def approx_lonlat_error_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    lat0 = math.radians((lat1 + lat2) * 0.5)
    dx = (lon2 - lon1) * 111_320.0 * math.cos(lat0)
    dy = (lat2 - lat1) * 110_540.0
    return float(math.sqrt(dx * dx + dy * dy))


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


def bbox_size_m(bbox: Optional[Tuple[float, float, float, float]]) -> Tuple[Optional[float], Optional[float]]:
    if bbox is None:
        return None, None
    west, east, south, north = bbox
    lat0 = math.radians((south + north) * 0.5)
    width_m = abs(east - west) * 111_320.0 * math.cos(lat0)
    height_m = abs(north - south) * 110_540.0
    return float(width_m), float(height_m)


def choose_gt_tile(
    sat_df: pd.DataFrame,
    bc: Dict[str, Optional[str]],
    cc: Dict[str, Optional[str]],
    lon: float,
    lat: float,
) -> Tuple[Optional[pd.Series], str, int]:
    if all(bc.values()):
        mask = []
        for _, row in sat_df.iterrows():
            bbox = row_bbox(row, bc)
            if bbox is None:
                mask.append(False)
                continue
            west, east, south, north = bbox
            mask.append(west <= lon <= east and south <= lat <= north)

        candidates = sat_df[pd.Series(mask, index=sat_df.index)]
        if len(candidates) > 0:
            best_idx = None
            best_err = float("inf")
            for idx, row in candidates.iterrows():
                cen = row_center(row, bc, cc)
                if cen is None:
                    best_idx = idx
                    break
                err = approx_lonlat_error_m(lon, lat, cen[0], cen[1])
                if err < best_err:
                    best_err = err
                    best_idx = idx
            return sat_df.loc[best_idx], "bbox_contains_eval_point", int(len(candidates))

    if cc["lon"] and cc["lat"]:
        best_idx = None
        best_err = float("inf")
        for idx, row in sat_df.iterrows():
            cen = row_center(row, bc, cc)
            if cen is None:
                continue
            err = approx_lonlat_error_m(lon, lat, cen[0], cen[1])
            if err < best_err:
                best_err = err
                best_idx = idx
        if best_idx is not None:
            return sat_df.loc[best_idx], "nearest_center_fallback", 0

    return None, "not_found", 0


def build_decomp_item(label: str, path: Path, rgb: np.ndarray, resize_size: int) -> Dict:
    gray = to_luma(rgb)
    mag, ori, strong = sobel_mag_ori(gray)
    fixed = resized_norm(gray, resize_size)

    h, w = rgb.shape[:2]

    return {
        "label": label,
        "path": path,
        "rgb": rgb,
        "gray": gray,
        "sobel_mag": mag,
        "ori": ori,
        "strong": strong,
        "fixed": fixed,
        "h": h,
        "w": w,
        "aspect": float(w / h) if h else None,
    }


def save_deconstruction_panel(items: List[Dict], out_path: Path, title: str) -> None:
    cols = ["RGB original", "luma / grayscale", "Sobel magnitude", "gradient orientation", "fixed resize norm"]
    nrows = len(items)
    ncols = len(cols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.6 * nrows), squeeze=False)

    for r, item in enumerate(items):
        row_label = f"{item['label']}\n{item['w']}x{item['h']} px, aspect={item['aspect']:.3f}"

        axes[r, 0].imshow(item["rgb"])
        axes[r, 0].set_title(f"{row_label}\n{cols[0]}", fontsize=9)

        axes[r, 1].imshow(item["gray"], cmap="gray")
        axes[r, 1].set_title(cols[1], fontsize=9)

        axes[r, 2].imshow(item["sobel_mag"], cmap="gray")
        axes[r, 2].set_title(cols[2], fontsize=9)

        ori_masked = np.ma.masked_where(~item["strong"], item["ori"])
        axes[r, 3].imshow(ori_masked, cmap="hsv", vmin=0, vmax=2.0 * np.pi)
        axes[r, 3].set_facecolor("black")
        axes[r, 3].set_title(cols[3], fontsize=9)

        axes[r, 4].imshow(item["fixed"], cmap="gray")
        axes[r, 4].set_title(cols[4], fontsize=9)

        for c in range(ncols):
            axes[r, c].axis("off")

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def save_footprint_panel(items: List[Dict], notes: List[str], out_path: Path, title: str) -> None:
    n = len(items)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 5.0), squeeze=False)
    axes = axes[0]

    for ax, item, note in zip(axes, items, notes):
        ax.imshow(item["rgb"])
        ax.set_title(item["label"], fontsize=10)
        ax.text(
            0.02,
            0.98,
            note,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.78),
        )
        ax.axis("off")

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, help="Accepted for command consistency; not required by this script.")
    parser.add_argument("--sequence", default="traj01")
    parser.add_argument("--tokens", default="1,115,574")
    parser.add_argument("--uav-index", default=str(DEFAULT_UAV_INDEX))
    parser.add_argument("--sat-index", default=str(DEFAULT_SAT_INDEX))
    parser.add_argument("--uav-dir", default=str(DEFAULT_UAV_DIR))
    parser.add_argument("--sat-dir", default=str(DEFAULT_SAT_DIR))
    parser.add_argument("--resize-size", type=int, default=256)
    parser.add_argument("--orb-false-tiles", default=DEFAULT_ORB_FALSE_TILES)
    parser.add_argument("--no-false-panels", action="store_true")
    args = parser.parse_args()

    fig_dir = FIG_DIR
    report_dir = REPORT_DIR
    fig_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    uav_index_path = Path(args.uav_index)
    sat_index_path = Path(args.sat_index)
    uav_dir = Path(args.uav_dir)
    sat_dir = Path(args.sat_dir)

    if not uav_index_path.exists():
        raise FileNotFoundError(f"Missing UAV index: {uav_index_path}")
    if not sat_index_path.exists():
        raise FileNotFoundError(f"Missing satellite index: {sat_index_path}")

    uav_df = pd.read_csv(uav_index_path)
    sat_df = pd.read_csv(sat_index_path)

    uav_path_col = find_path_col(uav_df)
    sat_path_col = find_path_col(sat_df)

    if uav_path_col is None:
        raise RuntimeError(f"Could not infer UAV image path/filename column. Columns: {list(uav_df.columns)}")
    if sat_path_col is None:
        raise RuntimeError(f"Could not infer satellite tile path/filename column. Columns: {list(sat_df.columns)}")

    sequence_col = find_col(uav_df, ["sequence", "seq", "traj", "trajectory"])
    if sequence_col is not None:
        seq_mask = uav_df[sequence_col].astype(str).str.lower().eq(args.sequence.lower())
        if seq_mask.any():
            uav_df = uav_df[seq_mask].copy()

    token_col = find_col(uav_df, ["token0_id", "token", "frame_token", "frame_id", "id", "uav_id"])
    if token_col is not None:
        uav_df["_token_norm"] = uav_df[token_col].map(norm_id)
    else:
        uav_df["_token_norm"] = uav_df[uav_path_col].map(parse_token_from_filename)

    lon_col = find_col(uav_df, ["longitude", "lon", "uav_lon", "gt_lon", "label_lon"])
    lat_col = find_col(uav_df, ["latitude", "lat", "uav_lat", "gt_lat", "label_lat"])

    if lon_col is None or lat_col is None:
        raise RuntimeError(
            "Could not infer UAV eval lon/lat columns. "
            f"Columns: {list(uav_df.columns)}"
        )

    tile_id_col = find_col(sat_df, ["tile_id", "sat_tile_id", "sat_id", "tile_index", "ref_id", "id"])
    if tile_id_col is not None:
        sat_df["_tile_id_norm"] = sat_df[tile_id_col].map(norm_id)
    else:
        sat_df["_tile_id_norm"] = sat_df[sat_path_col].map(parse_numeric_id_from_filename)

    bc = bbox_cols(sat_df)
    cc = center_cols(sat_df)

    query_tokens = parse_csv_list(args.tokens)
    false_tile_map = parse_false_tile_map(args.orb_false_tiles)

    manifest_rows = []
    warnings = []

    for token_raw in query_tokens:
        token = norm_id(token_raw)
        matches = uav_df[uav_df["_token_norm"].eq(token)]

        if matches.empty:
            msg = f"Token {token} not found in UAV index."
            print(f"WARNING: {msg}")
            warnings.append(msg)
            continue

        uav_row = matches.iloc[0]
        uav_path = resolve_path(uav_row[uav_path_col], [uav_dir])
        uav_lon = float(uav_row[lon_col])
        uav_lat = float(uav_row[lat_col])

        gt_row, gt_method, gt_candidate_count = choose_gt_tile(sat_df, bc, cc, uav_lon, uav_lat)
        if gt_row is None:
            msg = f"Could not find GT/debug satellite tile for token {token}."
            print(f"WARNING: {msg}")
            warnings.append(msg)
            continue

        gt_tile_id = gt_row["_tile_id_norm"]
        gt_path = resolve_path(gt_row[sat_path_col], [sat_dir])

        sat_items = [("GT/debug tile", gt_tile_id, gt_row, gt_path)]

        false_tile_id = false_tile_map.get(token)
        false_row = None
        false_path = None

        if false_tile_id and not args.no_false_panels:
            false_matches = sat_df[sat_df["_tile_id_norm"].eq(norm_id(false_tile_id))]
            if not false_matches.empty:
                false_row = false_matches.iloc[0]
                false_path = resolve_path(false_row[sat_path_col], [sat_dir])
                sat_items.append(("ORB top1 false tile", false_tile_id, false_row, false_path))
            else:
                msg = f"False tile {false_tile_id} for token {token} not found in satellite index."
                print(f"WARNING: {msg}")
                warnings.append(msg)

        uav_rgb = load_rgb(uav_path)
        items = [build_decomp_item(f"UAV token {token}", uav_path, uav_rgb, args.resize_size)]

        for label, tid, row, path in sat_items:
            rgb = load_rgb(path)
            items.append(build_decomp_item(f"{label} {tid}", path, rgb, args.resize_size))

        decomp_path = fig_dir / f"token_{int(token):04d}_deconstruction.png"
        save_deconstruction_panel(
            items,
            decomp_path,
            title=f"S4B.0 structural preflight — token {token}: UAV vs satellite structure",
        )

        notes = []
        for item_idx, item in enumerate(items):
            if item_idx == 0:
                note = (
                    f"path: {item['path'].name}\n"
                    f"image px: {item['w']} x {item['h']}\n"
                    f"aspect: {item['aspect']:.3f}\n"
                    f"eval lon/lat only:\n{uav_lon:.6f}, {uav_lat:.6f}\n"
                    f"ground footprint: unknown"
                )
            else:
                sat_row = gt_row if item_idx == 1 else false_row
                bbox = row_bbox(sat_row, bc) if sat_row is not None else None
                bw, bh = bbox_size_m(bbox)
                cen = row_center(sat_row, bc, cc) if sat_row is not None else None
                center_err = None
                if cen is not None:
                    center_err = approx_lonlat_error_m(uav_lon, uav_lat, cen[0], cen[1])

                gsd_x = bw / item["w"] if bw is not None and item["w"] else None
                gsd_y = bh / item["h"] if bh is not None and item["h"] else None

                note = (
                    f"path: {item['path'].name}\n"
                    f"image px: {item['w']} x {item['h']}\n"
                    f"aspect: {item['aspect']:.3f}\n"
                    f"tile bbox: {'available' if bbox else 'missing'}\n"
                    f"bbox size m: "
                    f"{bw:.2f} x {bh:.2f}" if bw is not None and bh is not None else
                    f"path: {item['path'].name}\nimage px: {item['w']} x {item['h']}\naspect: {item['aspect']:.3f}\ntile bbox: missing"
                )
                if bw is not None and bh is not None:
                    note += f"\napprox sat GSD: {gsd_x:.3f}, {gsd_y:.3f} m/px"
                if center_err is not None:
                    note += f"\ncenter error to eval point: {center_err:.2f} m"

            notes.append(note)

        footprint_path = fig_dir / f"token_{int(token):04d}_footprint_size_notes.png"
        save_footprint_panel(
            items,
            notes,
            footprint_path,
            title=f"S4B.0 size / footprint notes — token {token}",
        )

        gt_bbox = row_bbox(gt_row, bc)
        gt_bw, gt_bh = bbox_size_m(gt_bbox)
        gt_center = row_center(gt_row, bc, cc)
        gt_center_error_m = (
            approx_lonlat_error_m(uav_lon, uav_lat, gt_center[0], gt_center[1])
            if gt_center is not None
            else None
        )

        false_center_error_m = None
        false_bbox_width_m = None
        false_bbox_height_m = None
        if false_row is not None:
            fb = row_bbox(false_row, bc)
            false_bbox_width_m, false_bbox_height_m = bbox_size_m(fb)
            fc = row_center(false_row, bc, cc)
            if fc is not None:
                false_center_error_m = approx_lonlat_error_m(uav_lon, uav_lat, fc[0], fc[1])

        uav_h, uav_w = uav_rgb.shape[:2]
        gt_rgb_tmp = load_rgb(gt_path)
        gt_h, gt_w = gt_rgb_tmp.shape[:2]

        manifest_rows.append(
            {
                "sequence": args.sequence,
                "query_token": token,
                "uav_path": str(uav_path),
                "uav_width_px": uav_w,
                "uav_height_px": uav_h,
                "uav_aspect": uav_w / uav_h,
                "uav_eval_lon": uav_lon,
                "uav_eval_lat": uav_lat,
                "gt_tile_id": gt_tile_id,
                "gt_tile_path": str(gt_path),
                "gt_selection_method": gt_method,
                "gt_candidate_count": gt_candidate_count,
                "gt_width_px": gt_w,
                "gt_height_px": gt_h,
                "gt_aspect": gt_w / gt_h,
                "gt_bbox_width_m": gt_bw,
                "gt_bbox_height_m": gt_bh,
                "gt_approx_gsd_x_m_per_px": gt_bw / gt_w if gt_bw is not None and gt_w else None,
                "gt_approx_gsd_y_m_per_px": gt_bh / gt_h if gt_bh is not None and gt_h else None,
                "gt_center_error_m": gt_center_error_m,
                "orb_false_tile_id": false_tile_id,
                "orb_false_center_error_m": false_center_error_m,
                "orb_false_bbox_width_m": false_bbox_width_m,
                "orb_false_bbox_height_m": false_bbox_height_m,
                "deconstruction_panel": str(decomp_path),
                "footprint_panel": str(footprint_path),
            }
        )

    manifest_df = pd.DataFrame(manifest_rows)

    manifest_path = report_dir / "s4b0_preflight_manifest.csv"
    summary_path = report_dir / "s4b0_preflight_summary.json"

    manifest_df.to_csv(manifest_path, index=False)

    summary = {
        "stage": "S4B.0",
        "sequence": args.sequence,
        "query_tokens_requested": query_tokens,
        "query_tokens_processed": manifest_df["query_token"].tolist() if not manifest_df.empty else [],
        "processed_count": int(len(manifest_df)),
        "uav_index": str(uav_index_path),
        "sat_index": str(sat_index_path),
        "figures_dir": str(fig_dir),
        "manifest_csv": str(manifest_path),
        "resize_size": args.resize_size,
        "uav_path_column": uav_path_col,
        "sat_path_column": sat_path_col,
        "uav_token_column": token_col,
        "uav_lon_column": lon_col,
        "uav_lat_column": lat_col,
        "sat_tile_id_column": tile_id_col,
        "sat_bbox_columns": bc,
        "sat_center_columns": cc,
        "warnings": warnings,
        "important_rule": (
            "UAV filename lon/lat is used only for debug/evaluation labeling in this script. "
            "It is not used for retrieval scoring."
        ),
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("S4B.0 structural preflight complete")
    print("-----------------------------------")
    print(f"Sequence:          {args.sequence}")
    print(f"Processed queries: {len(manifest_df)}")
    print(f"Figures dir:       {fig_dir}")
    print(f"Manifest CSV:      {manifest_path}")
    print(f"Summary JSON:      {summary_path}")

    if warnings:
        print("")
        print("Warnings:")
        for w in warnings:
            print(f"- {w}")


if __name__ == "__main__":
    main()
