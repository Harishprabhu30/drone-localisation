#!/usr/bin/env python3
"""
S5B.1B — Full-map domain-normalized PHOG candidate generation

Purpose
-------
S5B.1 showed all-zero recovery because it reranked only old S4C top-50/top-200
candidate lists. That cannot rescue candidate-pool failures if the correct tile
is absent from those lists.

S5B.1B fixes that by doing real full-map retrieval over the full satellite tile
index, expected around 8625 tiles.

Main question
-------------
For the 40 candidate-pool-failure tokens:
Can domain-normalized PHOG variants make the correct satellite tile enter top-50?

Locked rule
-----------
UAV lon/lat and satellite tile centers are used only after ranking for
evaluation. Retrieval/ranking uses only image-derived PHOG descriptors.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--token-lists",
        type=Path,
        default=Path("outputs/satloc/metadata/s5b_candidate_pool_improvement/s5b0_token_lists_top50_all73.json"),
    )
    p.add_argument(
        "--sat-index",
        type=Path,
        default=Path("outputs/satloc/metadata/satellite_tiles_index_enriched.csv"),
    )
    p.add_argument(
        "--uav-index",
        type=Path,
        default=Path("outputs/satloc/metadata/uav_frames_index_enriched.csv"),
    )
    p.add_argument("--out-base", type=Path, default=Path("outputs/satloc"))

    p.add_argument("--run-name", type=str, default="cpf_fullmap_smoke")
    p.add_argument("--split-key", type=str, default="candidate_pool_failure")
    p.add_argument("--tokens", type=str, default="")
    p.add_argument("--max-tokens", type=int, default=3, help="Use 0 for all selected tokens.")
    p.add_argument("--sequence", type=str, default="traj01")

    p.add_argument("--top-k-eval", type=int, default=50)
    p.add_argument("--save-top-n", type=int, default=200)
    p.add_argument("--threshold-m", type=float, default=40.0)

    p.add_argument("--resize-long", type=int, default=512)
    p.add_argument("--bins", type=int, default=9)
    p.add_argument("--levels", type=str, default="1,2,4")

    p.add_argument(
        "--variants",
        type=str,
        default="v1_lab_l_clahe,v2_sat_detail,v3_green_suppressed,v4_uav_blur_sat_sharpen,v5_edge_magnitude",
    )

    p.add_argument("--force-recompute-cache", action="store_true")
    p.add_argument("--save-panels", action="store_true")
    p.add_argument("--min-sat-tiles", type=int, default=1000)

    return p.parse_args()


def ensure_dirs(base: Path):
    d = {
        "metadata": base / "metadata" / "s5b_candidate_pool_improvement",
        "reports": base / "reports" / "s5b_candidate_pool_improvement",
        "figures": base / "figures" / "s5b_candidate_pool_improvement",
        "cache": base / "metadata" / "s5b_candidate_pool_improvement" / "descriptor_cache",
        "panels": base / "figures" / "s5b_candidate_pool_improvement" / "s5b1b_fullmap_panels",
    }
    for x in d.values():
        x.mkdir(parents=True, exist_ok=True)
    return d


def safe_str(x: Any) -> str:
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return str(x).strip()


def resolve_path(x: Any) -> Optional[Path]:
    s = safe_str(x)
    if not s:
        return None
    p = Path(s)
    if p.exists():
        return p
    p2 = Path.cwd() / p
    if p2.exists():
        return p2
    return None


def read_rgb(path: Optional[Path]) -> Optional[np.ndarray]:
    if path is None:
        return None
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def resize_longest(img: np.ndarray, longest: int) -> np.ndarray:
    if longest <= 0:
        return img
    h, w = img.shape[:2]
    m = max(h, w)
    if m <= longest:
        return img
    scale = longest / float(m)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)


def normalize_u8(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    lo, hi = np.percentile(x, [1, 99])
    if hi <= lo:
        return np.zeros_like(x, dtype=np.uint8)
    y = (x - lo) * 255.0 / (hi - lo)
    return np.clip(y, 0, 255).astype(np.uint8)


def clahe(gray: np.ndarray, clip: float) -> np.ndarray:
    return cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8)).apply(gray)


def lab_l(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)[:, :, 0]


def log_chroma_channels(rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Illumination-reduced log chromaticity channels.

    c_rg = log(R/G)
    c_bg = log(B/G)

    This reduces common brightness/illumination scale and keeps relative
    color/reflectance differences. It is not a final descriptor by itself;
    we convert it into PHOG-friendly structural maps below.
    """
    arr = rgb.astype(np.float32) + 1.0
    r = arr[:, :, 0]
    g = arr[:, :, 1]
    b = arr[:, :, 2]
    c_rg = np.log(r / g)
    c_bg = np.log(b / g)
    return c_rg, c_bg


def log_chroma_edge_magnitude(rgb: np.ndarray) -> np.ndarray:
    c_rg, c_bg = log_chroma_channels(rgb)

    gx1 = cv2.Sobel(c_rg, cv2.CV_32F, 1, 0, ksize=3)
    gy1 = cv2.Sobel(c_rg, cv2.CV_32F, 0, 1, ksize=3)
    gx2 = cv2.Sobel(c_bg, cv2.CV_32F, 1, 0, ksize=3)
    gy2 = cv2.Sobel(c_bg, cv2.CV_32F, 0, 1, ksize=3)

    mag = np.sqrt(gx1 * gx1 + gy1 * gy1 + gx2 * gx2 + gy2 * gy2)
    return normalize_u8(mag)


def suppress_green(rgb: np.ndarray, strength: float = 0.55) -> np.ndarray:
    arr = rgb.astype(np.float32).copy()
    r = arr[:, :, 0]
    g = arr[:, :, 1]
    b = arr[:, :, 2]
    exg = 2.0 * g - r - b
    mask = exg > np.percentile(exg, 60)
    arr[:, :, 1][mask] *= strength
    return np.clip(arr, 0, 255).astype(np.uint8)


def unsharp(gray: np.ndarray, amount: float = 1.2, sigma: float = 1.2) -> np.ndarray:
    blur = cv2.GaussianBlur(gray, (0, 0), sigma)
    sharp = cv2.addWeighted(gray, 1.0 + amount, blur, -amount, 0)
    return np.clip(sharp, 0, 255).astype(np.uint8)


def preprocess(rgb: np.ndarray, variant: str, domain: str, resize_long: int) -> np.ndarray:
    """
    domain: 'uav' or 'sat'
    returns single-channel uint8 common-space image.
    """
    rgb = resize_longest(rgb, resize_long)

    if variant == "v6_log_chroma_clahe":
        c_rg, c_bg = log_chroma_channels(rgb)
        c1 = normalize_u8(c_rg).astype(np.float32)
        c2 = normalize_u8(c_bg).astype(np.float32)
        chroma = normalize_u8(0.5 * c1 + 0.5 * c2)
        return clahe(chroma, 2.0 if domain == "uav" else 3.5)

    if variant == "v7_log_chroma_edges":
        edge = log_chroma_edge_magnitude(rgb)
        if domain == "sat":
            edge = cv2.dilate(edge, np.ones((3, 3), np.uint8), iterations=1)
        return clahe(edge, 2.0 if domain == "uav" else 3.0)

    if variant == "v8_lab_logchroma_fused":
        L = clahe(lab_l(rgb), 2.0 if domain == "uav" else 4.0)
        chroma_edge = log_chroma_edge_magnitude(rgb)
        if domain == "uav":
            fused = cv2.addWeighted(L, 0.75, chroma_edge, 0.25, 0)
        else:
            chroma_edge = cv2.dilate(chroma_edge, np.ones((3, 3), np.uint8), iterations=1)
            fused = cv2.addWeighted(L, 0.55, chroma_edge, 0.45, 0)
        return normalize_u8(fused)

    if variant == "v9_canny_structure":
        L = clahe(lab_l(rgb), 2.0 if domain == "uav" else 4.0)
        if domain == "sat":
            L = unsharp(L, amount=1.2, sigma=1.0)
            edges = cv2.Canny(L, 35, 110)
            edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
        else:
            L = cv2.GaussianBlur(L, (3, 3), 0)
            edges = cv2.Canny(L, 55, 150)
        edges = cv2.GaussianBlur(edges, (3, 3), 0)
        return edges

    if variant == "v1_lab_l_clahe":
        L = lab_l(rgb)
        return clahe(L, 2.0 if domain == "uav" else 4.0)

    if variant == "v2_sat_detail":
        L = clahe(lab_l(rgb), 2.0 if domain == "uav" else 4.0)
        if domain == "sat":
            smooth = cv2.bilateralFilter(L, 9, 50, 50)
            detail = cv2.subtract(L, smooth)
            out = cv2.addWeighted(L, 0.85, detail, 2.2, 0)
            return normalize_u8(out)
        mild = cv2.GaussianBlur(L, (3, 3), 0)
        return cv2.addWeighted(L, 0.75, mild, 0.25, 0)

    if variant == "v3_green_suppressed":
        rgbs = suppress_green(rgb, strength=0.50 if domain == "sat" else 0.70)
        L = lab_l(rgbs)
        return clahe(L, 2.0 if domain == "uav" else 3.5)

    if variant == "v4_uav_blur_sat_sharpen":
        L = clahe(lab_l(rgb), 2.0 if domain == "uav" else 3.5)
        if domain == "uav":
            return cv2.GaussianBlur(L, (5, 5), 0)
        return unsharp(L, amount=1.5, sigma=1.0)

    if variant == "v5_edge_magnitude":
        L = clahe(lab_l(rgb), 2.0 if domain == "uav" else 4.0)
        if domain == "uav":
            L = cv2.GaussianBlur(L, (3, 3), 0)
        else:
            L = unsharp(L, amount=1.0, sigma=1.0)
        gx = cv2.Scharr(L, cv2.CV_32F, 1, 0)
        gy = cv2.Scharr(L, cv2.CV_32F, 0, 1)
        mag = np.sqrt(gx * gx + gy * gy)
        return normalize_u8(mag)

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return clahe(gray, 2.0)


def phog_descriptor(gray: np.ndarray, bins: int, levels: List[int]) -> np.ndarray:
    gray = gray.astype(np.uint8)

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

    mag = np.sqrt(gx * gx + gy * gy)
    ang = (np.arctan2(gy, gx) + np.pi) % np.pi

    h, w = gray.shape[:2]
    feats = []

    for grid in levels:
        for yy in range(grid):
            y0 = int(round(yy * h / grid))
            y1 = int(round((yy + 1) * h / grid))
            for xx in range(grid):
                x0 = int(round(xx * w / grid))
                x1 = int(round((xx + 1) * w / grid))

                a = ang[y0:y1, x0:x1].ravel()
                m = mag[y0:y1, x0:x1].ravel()

                if len(a) == 0:
                    hist = np.zeros(bins, dtype=np.float32)
                else:
                    hist, _ = np.histogram(a, bins=bins, range=(0, np.pi), weights=m)
                    hist = hist.astype(np.float32)

                hist = hist / (np.linalg.norm(hist) + 1e-9)
                feats.append(hist)

    desc = np.concatenate(feats).astype(np.float32)
    desc = desc / (np.linalg.norm(desc) + 1e-9)
    return desc


def haversine_m(lon1, lat1, lon2, lat2) -> np.ndarray:
    """
    Vectorized approximate geodesic distance in meters.
    """
    lon1 = np.deg2rad(float(lon1))
    lat1 = np.deg2rad(float(lat1))
    lon2 = np.deg2rad(np.asarray(lon2, dtype=np.float64))
    lat2 = np.deg2rad(np.asarray(lat2, dtype=np.float64))

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(np.maximum(1.0 - a, 0.0)))
    return 6371000.0 * c


def choose_col(df: pd.DataFrame, candidates: List[str], required: bool = True, label: str = "") -> Optional[str]:
    lower_map = {c.lower(): c for c in df.columns}

    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]

    for c in df.columns:
        lc = c.lower()
        for want in candidates:
            if want.lower() in lc:
                return c

    if required:
        raise KeyError(f"Could not find column for {label}. Tried {candidates}. Available columns: {df.columns.tolist()}")
    return None


def choose_path_col(df: pd.DataFrame, label: str) -> str:
    candidates = [
        "image_path",
        "path",
        "file_path",
        "tile_path",
        "satellite_image_path",
        "sat_image_path",
        "candidate_image_path",
        "uav_image_path",
        "query_image_path",
    ]

    valid = []
    for c in df.columns:
        lc = c.lower()
        if any(k in lc for k in ["path", "image", "file"]):
            sample = df[c].dropna().astype(str).head(20).tolist()
            hits = 0
            for s in sample:
                if resolve_path(s) is not None:
                    hits += 1
            if hits > 0:
                valid.append((hits, c))

    if valid:
        valid = sorted(valid, reverse=True)
        return valid[0][1]

    return choose_col(df, candidates, True, label)


def choose_lon_lat_cols(df: pd.DataFrame, kind: str) -> Tuple[str, str]:
    """
    Avoid Loc1/Loc2 pixel-coordinate columns.
    Prefer enriched center_lon/center_lat or longitude/latitude.
    """
    cols = list(df.columns)
    lower = {c.lower(): c for c in cols}

    if kind == "sat":
        preferred_pairs = [
            ("center_lon", "center_lat"),
            ("tile_center_lon", "tile_center_lat"),
            ("center_longitude", "center_latitude"),
            ("lon_center", "lat_center"),
            ("longitude", "latitude"),
            ("lon", "lat"),
        ]
    else:
        preferred_pairs = [
            ("longitude", "latitude"),
            ("uav_lon", "uav_lat"),
            ("query_lon", "query_lat"),
            ("lon", "lat"),
            ("center_lon", "center_lat"),
        ]

    for lon, lat in preferred_pairs:
        if lon in lower and lat in lower:
            return lower[lon], lower[lat]

    # Bbox center fallback for satellite only.
    if kind == "sat":
        lon_min_opts = ["min_lon", "lon_min", "bbox_lon_min", "left_lon"]
        lon_max_opts = ["max_lon", "lon_max", "bbox_lon_max", "right_lon"]
        lat_min_opts = ["min_lat", "lat_min", "bbox_lat_min", "bottom_lat"]
        lat_max_opts = ["max_lat", "lat_max", "bbox_lat_max", "top_lat"]
        for a in lon_min_opts:
            for b in lon_max_opts:
                for c in lat_min_opts:
                    for d in lat_max_opts:
                        if a in lower and b in lower and c in lower and d in lower:
                            lon_col = "__computed_center_lon__"
                            lat_col = "__computed_center_lat__"
                            df[lon_col] = (pd.to_numeric(df[lower[a]], errors="coerce") + pd.to_numeric(df[lower[b]], errors="coerce")) / 2.0
                            df[lat_col] = (pd.to_numeric(df[lower[c]], errors="coerce") + pd.to_numeric(df[lower[d]], errors="coerce")) / 2.0
                            return lon_col, lat_col

    raise KeyError(
        f"Could not infer {kind} lon/lat columns. "
        f"Available columns: {df.columns.tolist()}"
    )


def choose_tile_id_col(df: pd.DataFrame) -> Optional[str]:
    for c in ["tile_id", "sat_tile_id", "ref_tile_id", "image_id", "id"]:
        for col in df.columns:
            if col.lower() == c:
                return col
    for col in df.columns:
        lc = col.lower()
        if "tile" in lc and "id" in lc:
            return col
    return None


def choose_token_col(df: pd.DataFrame) -> str:
    for c in ["token", "token0_id", "frame_token", "imgid", "frame_id"]:
        for col in df.columns:
            if col.lower() == c:
                return col
    for col in df.columns:
        lc = col.lower()
        if "token" in lc or "imgid" in lc or "frame" in lc:
            return col
    raise KeyError(f"Could not infer UAV token column. Available columns: {df.columns.tolist()}")


def load_token_list(path: Path, split_key: str, tokens_arg: str, max_tokens: int) -> List[str]:
    if tokens_arg.strip():
        tokens = [t.strip() for t in tokens_arg.split(",") if t.strip()]
    else:
        with open(path, "r") as f:
            lists = json.load(f)
        if split_key not in lists:
            raise KeyError(f"Split key {split_key} not found in {path}. Available: {list(lists.keys())}")
        tokens = [t.strip() for t in str(lists[split_key]).split(",") if t.strip()]

    if max_tokens > 0:
        tokens = tokens[:max_tokens]

    return tokens


def prepare_sat_index(path: Path, min_tiles: int) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing satellite index: {path}")

    df = pd.read_csv(path)
    path_col = choose_path_col(df, "satellite image path")
    lon_col, lat_col = choose_lon_lat_cols(df, "sat")
    tile_col = choose_tile_id_col(df)

    rows = []
    for i, r in df.iterrows():
        p = resolve_path(r.get(path_col))
        if p is None:
            continue

        lon = pd.to_numeric(pd.Series([r.get(lon_col)]), errors="coerce").iloc[0]
        lat = pd.to_numeric(pd.Series([r.get(lat_col)]), errors="coerce").iloc[0]
        if pd.isna(lon) or pd.isna(lat):
            continue

        tid = safe_str(r.get(tile_col)) if tile_col else p.stem
        if not tid:
            tid = p.stem

        rows.append(
            {
                "tile_id": tid,
                "sat_image_path": str(p),
                "sat_center_lon": float(lon),
                "sat_center_lat": float(lat),
            }
        )

    out = pd.DataFrame(rows).drop_duplicates("tile_id").reset_index(drop=True)

    if len(out) < min_tiles:
        raise RuntimeError(
            f"Satellite full-map index has only {len(out)} usable tiles. "
            f"Expected at least {min_tiles}. Check --sat-index/path/lon/lat columns."
        )

    return out


def prepare_uav_index(path: Path, sequence: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing UAV index: {path}")

    df = pd.read_csv(path)
    token_col = choose_token_col(df)
    path_col = choose_path_col(df, "UAV image path")
    lon_col, lat_col = choose_lon_lat_cols(df, "uav")

    if sequence:
        seq_cols = [c for c in df.columns if c.lower() in ["sequence", "traj", "trajectory", "seq"]]
        if seq_cols:
            sc = seq_cols[0]
            sub = df[df[sc].astype(str) == sequence].copy()
            if len(sub):
                df = sub

    rows = []
    for _, r in df.iterrows():
        p = resolve_path(r.get(path_col))
        if p is None:
            continue

        lon = pd.to_numeric(pd.Series([r.get(lon_col)]), errors="coerce").iloc[0]
        lat = pd.to_numeric(pd.Series([r.get(lat_col)]), errors="coerce").iloc[0]
        if pd.isna(lon) or pd.isna(lat):
            continue

        rows.append(
            {
                "token": safe_str(r.get(token_col)),
                "uav_image_path": str(p),
                "uav_lon": float(lon),
                "uav_lat": float(lat),
            }
        )

    out = pd.DataFrame(rows)
    if len(out) == 0:
        raise RuntimeError("No usable UAV rows found after path/lon/lat parsing.")

    out["token_int"] = pd.to_numeric(out["token"], errors="coerce")
    return out


def descriptor_cache_path(cache_dir: Path, variant: str, resize_long: int, bins: int, levels: List[int]) -> Path:
    level_tag = "-".join(str(x) for x in levels)
    return cache_dir / f"s5b1b_sat_desc_{variant}_r{resize_long}_b{bins}_l{level_tag}.npz"


def compute_satellite_cache(
    sat: pd.DataFrame,
    variant: str,
    resize_long: int,
    bins: int,
    levels: List[int],
    cache_path: Path,
    force: bool,
) -> Dict[str, Any]:
    if cache_path.exists() and not force:
        data = np.load(cache_path, allow_pickle=True)
        return {
            "tile_ids": data["tile_ids"].astype(str),
            "image_paths": data["image_paths"].astype(str),
            "lon": data["lon"].astype(np.float64),
            "lat": data["lat"].astype(np.float64),
            "desc": data["desc"].astype(np.float32),
            "loaded_from_cache": True,
        }

    descs = []
    tile_ids = []
    image_paths = []
    lons = []
    lats = []

    t0 = time.time()
    for i, r in sat.iterrows():
        if i % 500 == 0:
            print(f"    cache {variant}: {i}/{len(sat)} tiles")

        p = resolve_path(r["sat_image_path"])
        rgb = read_rgb(p)
        if rgb is None:
            continue

        gray = preprocess(rgb, variant, "sat", resize_long)
        desc = phog_descriptor(gray, bins=bins, levels=levels)

        descs.append(desc)
        tile_ids.append(str(r["tile_id"]))
        image_paths.append(str(p))
        lons.append(float(r["sat_center_lon"]))
        lats.append(float(r["sat_center_lat"]))

    if not descs:
        raise RuntimeError(f"No satellite descriptors computed for variant {variant}")

    arr = np.vstack(descs).astype(np.float32)
    arr = arr / (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-9)

    np.savez_compressed(
        cache_path,
        tile_ids=np.asarray(tile_ids, dtype=object),
        image_paths=np.asarray(image_paths, dtype=object),
        lon=np.asarray(lons, dtype=np.float64),
        lat=np.asarray(lats, dtype=np.float64),
        desc=arr,
    )

    print(f"    saved cache {variant}: {len(arr)} descriptors in {time.time() - t0:.1f}s -> {cache_path}")

    return {
        "tile_ids": np.asarray(tile_ids, dtype=str),
        "image_paths": np.asarray(image_paths, dtype=str),
        "lon": np.asarray(lons, dtype=np.float64),
        "lat": np.asarray(lats, dtype=np.float64),
        "desc": arr,
        "loaded_from_cache": False,
    }


def query_descriptor(uav_path: Path, variant: str, resize_long: int, bins: int, levels: List[int]) -> np.ndarray:
    rgb = read_rgb(uav_path)
    if rgb is None:
        raise RuntimeError(f"Could not read UAV image: {uav_path}")
    gray = preprocess(rgb, variant, "uav", resize_long)
    desc = phog_descriptor(gray, bins=bins, levels=levels)
    return desc.astype(np.float32)


def evaluate_ranked(
    ranked: pd.DataFrame,
    top_k: int,
    threshold_m: float,
) -> Dict[str, Any]:
    top1 = ranked.iloc[0]
    top1_error = float(top1["eval_error_m"])
    top1_hit = bool(top1_error <= threshold_m)

    topk = ranked.head(top_k)
    oracle_error = float(topk["eval_error_m"].min())
    oracle_hit = bool(oracle_error <= threshold_m)

    correct = ranked[ranked["eval_error_m"] <= threshold_m]
    first_correct_rank = None
    best_correct_error = None
    if len(correct):
        first_correct_rank = int(correct.iloc[0]["rank"])
        best_correct_error = float(correct["eval_error_m"].min())

    return {
        "top1_tile_id": str(top1["tile_id"]),
        "top1_error_m": top1_error,
        "top1_hit_le_threshold": top1_hit,
        "oracle_topk_error_m": oracle_error,
        "oracle_topk_hit_le_threshold": oracle_hit,
        "first_correct_rank": first_correct_rank,
        "best_correct_error_m": best_correct_error,
    }


def add_img(ax, img: Optional[np.ndarray], title: str, gray: bool = False):
    ax.axis("off")
    ax.set_title(title, fontsize=9)
    if img is None:
        ax.text(0.5, 0.5, "missing", ha="center", va="center")
    else:
        if gray:
            ax.imshow(img, cmap="gray")
        else:
            ax.imshow(img)


def save_token_panel(
    token: str,
    uav_path: Path,
    uav_lon: float,
    uav_lat: float,
    variant_top_rows: List[pd.Series],
    out_path: Path,
    resize_long: int,
):
    n = len(variant_top_rows)
    if n == 0:
        return

    uav_rgb = read_rgb(uav_path)
    fig, axes = plt.subplots(n, 4, figsize=(17, 3.8 * n))
    if n == 1:
        axes = np.asarray([axes])

    for i, r in enumerate(variant_top_rows):
        variant = str(r["variant"])
        sat_path = resolve_path(r["sat_image_path"])
        sat_rgb = read_rgb(sat_path)

        add_img(axes[i, 0], uav_rgb, f"UAV token {token}\nlon={uav_lon:.6f}, lat={uav_lat:.6f}")
        add_img(
            axes[i, 1],
            sat_rgb,
            f"{variant} top1\n"
            f"tile={r['tile_id']} rank={int(r['rank'])}\n"
            f"err={float(r['eval_error_m']):.1f}m dist={float(r['distance']):.3f}",
        )

        if uav_rgb is not None:
            u_proc = preprocess(uav_rgb, variant, "uav", resize_long)
        else:
            u_proc = None
        if sat_rgb is not None:
            s_proc = preprocess(sat_rgb, variant, "sat", resize_long)
        else:
            s_proc = None

        add_img(axes[i, 2], u_proc, "UAV processed", gray=True)
        add_img(axes[i, 3], s_proc, "SAT processed", gray=True)

    fig.suptitle("S5B.1B full-map domain-normalized PHOG top1 comparison", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_variant_summary(summary: pd.DataFrame, out_hit: Path, out_error: Path, top_k: int):
    if len(summary) == 0:
        return

    plt.figure(figsize=(12, 5.5))
    plt.bar(summary["variant"], summary["oracle_topk_hit_rate"])
    plt.xticks(rotation=35, ha="right")
    plt.ylim(0, 1.05)
    plt.ylabel(f"Oracle top-{top_k} hit rate")
    plt.title("S5B.1B full-map candidate-pool recovery by variant")
    plt.tight_layout()
    plt.savefig(out_hit, dpi=180)
    plt.close()

    plt.figure(figsize=(12, 5.5))
    plt.bar(summary["variant"], summary["median_top1_error_m"])
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("Median top-1 error [m]")
    plt.title("S5B.1B median top-1 error by variant")
    plt.tight_layout()
    plt.savefig(out_error, dpi=180)
    plt.close()


def main():
    args = parse_args()
    dirs = ensure_dirs(args.out_base)

    levels = [int(x.strip()) for x in args.levels.split(",") if x.strip()]
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    tokens = load_token_list(args.token_lists, args.split_key, args.tokens, args.max_tokens)

    started = time.time()

    print("S5B.1B full-map setup")
    print("---------------------")
    print(f"Tokens:        {len(tokens)} -> {tokens}")
    print(f"Variants:      {variants}")
    print(f"Resize-long:   {args.resize_long}")
    print(f"Top-K eval:    {args.top_k_eval}")
    print()

    sat = prepare_sat_index(args.sat_index, args.min_sat_tiles)
    uav = prepare_uav_index(args.uav_index, args.sequence)

    print(f"Satellite tiles loaded: {len(sat)}")
    print(f"UAV rows loaded:        {len(uav)}")
    print()

    all_ranked = []
    query_rows = []
    cache_reports = []

    # Precompute/load full satellite descriptor cache per variant.
    sat_caches = {}
    for variant in variants:
        print(f"[CACHE] variant={variant}")
        cache_path = descriptor_cache_path(dirs["cache"], variant, args.resize_long, args.bins, levels)
        cache = compute_satellite_cache(
            sat=sat,
            variant=variant,
            resize_long=args.resize_long,
            bins=args.bins,
            levels=levels,
            cache_path=cache_path,
            force=args.force_recompute_cache,
        )
        sat_caches[variant] = cache
        cache_reports.append(
            {
                "variant": variant,
                "cache_path": str(cache_path),
                "descriptor_count": int(cache["desc"].shape[0]),
                "descriptor_dim": int(cache["desc"].shape[1]),
                "loaded_from_cache": bool(cache["loaded_from_cache"]),
            }
        )
    print()

    for token in tokens:
        usub = uav[uav["token"].astype(str) == str(token)]
        if len(usub) == 0:
            print(f"[WARN] token={token}: UAV not found in index")
            for variant in variants:
                query_rows.append(
                    {
                        "token": token,
                        "variant": variant,
                        "status": "missing_uav_token",
                    }
                )
            continue

        urow = usub.iloc[0]
        uav_path = resolve_path(urow["uav_image_path"])
        uav_lon = float(urow["uav_lon"])
        uav_lat = float(urow["uav_lat"])

        print(f"[QUERY] token={token} uav={uav_path}")

        variant_top_rows = []

        for variant in variants:
            q_start = time.time()
            cache = sat_caches[variant]

            try:
                qdesc = query_descriptor(
                    uav_path=uav_path,
                    variant=variant,
                    resize_long=args.resize_long,
                    bins=args.bins,
                    levels=levels,
                )
            except Exception as exc:
                print(f"    [FAIL] {variant}: {exc}")
                query_rows.append(
                    {
                        "token": token,
                        "variant": variant,
                        "status": "query_descriptor_failed",
                        "error": repr(exc),
                    }
                )
                continue

            desc = cache["desc"]
            # Descriptors are normalized, so cosine distance = 1 - dot product.
            scores = desc @ qdesc
            distances = 1.0 - scores

            order = np.argsort(distances, kind="mergesort")

            eval_errors = haversine_m(
                uav_lon,
                uav_lat,
                cache["lon"],
                cache["lat"],
            )

            topN = min(args.save_top_n, len(order))
            rows = []
            for rank_idx, sat_idx in enumerate(order[:topN], start=1):
                rows.append(
                    {
                        "token": token,
                        "variant": variant,
                        "rank": rank_idx,
                        "tile_id": str(cache["tile_ids"][sat_idx]),
                        "sat_image_path": str(cache["image_paths"][sat_idx]),
                        "distance": float(distances[sat_idx]),
                        "score": float(scores[sat_idx]),
                        "eval_error_m": float(eval_errors[sat_idx]),
                        "sat_center_lon": float(cache["lon"][sat_idx]),
                        "sat_center_lat": float(cache["lat"][sat_idx]),
                        "uav_lon": uav_lon,
                        "uav_lat": uav_lat,
                        "uav_image_path": str(uav_path),
                    }
                )

            ranked_top = pd.DataFrame(rows)
            all_ranked.append(ranked_top)

            # For evaluation we need full ranked order, not just topN.
            full_eval_rows = pd.DataFrame(
                {
                    "rank": np.arange(1, len(order) + 1),
                    "eval_error_m": eval_errors[order],
                    "tile_id": cache["tile_ids"][order].astype(str),
                }
            )

            ev = evaluate_ranked(
                full_eval_rows,
                top_k=args.top_k_eval,
                threshold_m=args.threshold_m,
            )

            query_rows.append(
                {
                    "token": token,
                    "variant": variant,
                    "status": "ok",
                    "satellite_candidates": int(len(order)),
                    "runtime_s": float(time.time() - q_start),
                    "uav_image_path": str(uav_path),
                    "uav_lon": uav_lon,
                    "uav_lat": uav_lat,
                    **ev,
                }
            )

            if len(ranked_top):
                variant_top_rows.append(ranked_top.iloc[0])

            print(
                f"    {variant}: "
                f"top1={ev['top1_error_m']:.1f}m "
                f"oracle@{args.top_k_eval}={ev['oracle_topk_error_m']:.1f}m "
                f"hit={ev['oracle_topk_hit_le_threshold']} "
                f"first_correct={ev['first_correct_rank']} "
                f"time={time.time() - q_start:.1f}s"
            )

        if args.save_panels and variant_top_rows:
            save_token_panel(
                token=token,
                uav_path=uav_path,
                uav_lon=uav_lon,
                uav_lat=uav_lat,
                variant_top_rows=variant_top_rows,
                out_path=dirs["panels"] / args.run_name / f"s5b1b_token{int(float(token)):04d}_fullmap_variant_panel.png",
                resize_long=args.resize_long,
            )

    ranked_df = pd.concat(all_ranked, ignore_index=True) if all_ranked else pd.DataFrame()
    query_df = pd.DataFrame(query_rows)

    ok = query_df[query_df["status"] == "ok"].copy()

    summary_rows = []
    for variant, g in ok.groupby("variant"):
        top1_err = pd.to_numeric(g["top1_error_m"], errors="coerce")
        first_rank = pd.to_numeric(g["first_correct_rank"], errors="coerce")

        summary_rows.append(
            {
                "variant": variant,
                "tokens": int(len(g)),
                "top1_hits": int(g["top1_hit_le_threshold"].sum()),
                "top1_hit_rate": float(g["top1_hit_le_threshold"].mean()),
                "oracle_topk_hits": int(g["oracle_topk_hit_le_threshold"].sum()),
                "oracle_topk_hit_rate": float(g["oracle_topk_hit_le_threshold"].mean()),
                "median_top1_error_m": float(top1_err.median()) if top1_err.notna().any() else None,
                "median_oracle_topk_error_m": float(pd.to_numeric(g["oracle_topk_error_m"], errors="coerce").median()),
                "median_first_correct_rank": float(first_rank.median()) if first_rank.notna().any() else None,
                "mean_runtime_s": float(pd.to_numeric(g["runtime_s"], errors="coerce").mean()),
                "mean_satellite_candidates": float(pd.to_numeric(g["satellite_candidates"], errors="coerce").mean()),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    if len(summary_df):
        summary_df = summary_df.sort_values(
            ["oracle_topk_hit_rate", "top1_hit_rate", "median_oracle_topk_error_m"],
            ascending=[False, False, True],
            kind="mergesort",
        )

    suffix = f"_{args.run_name}" if args.run_name else ""

    ranked_out = dirs["metadata"] / f"s5b1b_fullmap_ranked_top{args.save_top_n}{suffix}.csv"
    query_out = dirs["metadata"] / f"s5b1b_fullmap_query_summary{suffix}.csv"
    variant_out = dirs["metadata"] / f"s5b1b_fullmap_variant_summary{suffix}.csv"
    cache_out = dirs["metadata"] / f"s5b1b_descriptor_cache_manifest{suffix}.csv"
    report_out = dirs["reports"] / f"s5b1b_fullmap_domain_normalized_phog_summary{suffix}.json"

    ranked_df.to_csv(ranked_out, index=False)
    query_df.to_csv(query_out, index=False)
    summary_df.to_csv(variant_out, index=False)
    pd.DataFrame(cache_reports).to_csv(cache_out, index=False)

    fig_hit = dirs["figures"] / f"s5b1b_fullmap_oracle_topk_hit_rate{suffix}.png"
    fig_err = dirs["figures"] / f"s5b1b_fullmap_median_error{suffix}.png"
    plot_variant_summary(summary_df, fig_hit, fig_err, args.top_k_eval)

    report = {
        "stage": "S5B.1B_fullmap_domain_normalized_phog",
        "run_name": args.run_name,
        "tokens": tokens,
        "num_tokens": int(len(tokens)),
        "variants": variants,
        "satellite_tiles": int(len(sat)),
        "top_k_eval": args.top_k_eval,
        "threshold_m": args.threshold_m,
        "resize_long": args.resize_long,
        "bins": args.bins,
        "levels": levels,
        "runtime_s": float(time.time() - started),
        "best_variant": summary_df.iloc[0].to_dict() if len(summary_df) else None,
        "outputs": {
            "ranked_topn_csv": str(ranked_out),
            "query_summary_csv": str(query_out),
            "variant_summary_csv": str(variant_out),
            "cache_manifest_csv": str(cache_out),
            "summary_json": str(report_out),
            "oracle_topk_hit_rate_figure": str(fig_hit),
            "median_error_figure": str(fig_err),
        },
        "locked_rule": "ranking used only image-derived PHOG descriptors; UAV lon/lat and satellite centers were used only after ranking for evaluation",
    }

    with open(report_out, "w") as f:
        json.dump(report, f, indent=2)

    print()
    print("S5B.1B full-map domain-normalized PHOG complete")
    print("-----------------------------------------------")
    print(f"Tokens processed:          {len(tokens)} -> {tokens}")
    print(f"Satellite tiles searched:  {len(sat)}")
    print(f"Variants tested:           {variants}")
    print(f"Resize-long:               {args.resize_long}")
    print(f"Top-K eval:                {args.top_k_eval}")
    print()
    print("Variant summary:")
    if len(summary_df):
        print(summary_df.to_string(index=False))
    else:
        print("No valid results.")
    print()
    print(f"Ranked top-N CSV:          {ranked_out}")
    print(f"Query summary CSV:         {query_out}")
    print(f"Variant summary CSV:       {variant_out}")
    print(f"Cache manifest CSV:        {cache_out}")
    print(f"Summary JSON:              {report_out}")
    print(f"Figures:                   {fig_hit}")
    print(f"                           {fig_err}")
    if args.save_panels:
        print(f"Panels dir:                {dirs['panels'] / args.run_name}")
    print()
    print("Locked rule: reference coordinates were used only after ranking for evaluation.")


if __name__ == "__main__":
    main()
