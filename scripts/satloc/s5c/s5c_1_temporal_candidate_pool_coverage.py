#!/usr/bin/env python3
"""
S5C.1 — Temporal Candidate-Pool Coverage Benchmark

Evaluates candidate-pool coverage for the S5C.0 temporal query manifest.

Important rules:
- No LightGlue is run here.
- Ranking uses only image descriptors.
- Reference coordinates are used only for oracle/evaluation columns.
- This stage decides whether S5C.2 should run LightGlue Top-20 or Top-50.

Outputs:
outputs/satloc/metadata/s5c_temporal/
  s5c1_candidate_scores_top50.csv
  s5c1_union_candidate_scores_top50.csv
  s5c1_query_summary.csv
  s5c1_variant_summary.csv
  s5c1_gap_to_next_oracle_success.csv

outputs/satloc/reports/s5c_temporal/
  s5c1_candidate_pool_coverage_summary.json

outputs/satloc/figures/s5c_temporal/
  s5c1_oracle_recall_by_k.png
  s5c1_best_top50_error_hist.png
  s5c1_success_timeline.png
  s5c1_query_gap_to_next_oracle_success.png

command Used:
1. For 10 frames Trial Run:

export PYTHONPATH=$PWD/src

python scripts/satloc/s5c/s5c_1_temporal_candidate_pool_coverage.py \
  --sequence traj01 \
  --max-queries 10 \
  --max-tiles 300 \
  --topk-save 50

2. For Full run:

export PYTHONPATH=$PWD/src

python scripts/satloc/s5c/s5c_1_temporal_candidate_pool_coverage.py \
  --sequence traj01 \
  --topk-save 50
  
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def now_s() -> float:
    return time.perf_counter()


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def print_header(title: str) -> None:
    print(title)
    print("-" * len(title))


def parse_csv_list(s: str, typ=str) -> List:
    out = []
    for x in str(s).split(","):
        x = x.strip()
        if x:
            out.append(typ(x))
    return out


def first_existing_col(df: pd.DataFrame, candidates: Sequence[str], required: bool = True, label: str = "") -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise RuntimeError(
            f"Could not find required column {label}. Tried {list(candidates)}. "
            f"Available columns: {list(df.columns)}"
        )
    return None


def resolve_path(path_like: str | Path, repo_root: Path) -> Path:
    p = Path(str(path_like))
    return p if p.is_absolute() else repo_root / p


def safe_float(x, default=np.nan) -> float:
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


@dataclass
class CoordSpec:
    mode: str
    x_col: str
    y_col: str


def find_xy_cols(df: pd.DataFrame, role: str) -> Optional[CoordSpec]:
    projected_pairs = [
        ("reference_x_m", "reference_y_m"),
        ("ref_x_m", "ref_y_m"),
        ("x_ref_m", "y_ref_m"),
        ("x_enu_m", "y_enu_m"),
        ("enu_x_m", "enu_y_m"),
        ("east_m", "north_m"),
        ("easting_m", "northing_m"),
        ("tile_x_m", "tile_y_m"),
        ("center_x_m", "center_y_m"),
        ("x_m", "y_m"),
        ("map_x_m", "map_y_m"),
    ]
    for x, y in projected_pairs:
        if x in df.columns and y in df.columns:
            return CoordSpec("xy_m", x, y)

    lonlat_pairs = [
        ("reference_lon", "reference_lat"),
        ("ref_lon", "ref_lat"),
        ("lon", "lat"),
        ("longitude", "latitude"),
        ("center_lon", "center_lat"),
        ("tile_lon", "tile_lat"),
    ]
    for lon, lat in lonlat_pairs:
        if lon in df.columns and lat in df.columns:
            return CoordSpec("lonlat", lon, lat)

    print(f"WARN: could not infer coordinate columns for {role}. Oracle errors will be NaN.")
    return None


def approx_lonlat_distance_m(lon1, lat1, lon2, lat2) -> np.ndarray:
    r = 6371008.8
    lon1 = np.asarray(lon1, dtype=np.float64)
    lat1 = np.asarray(lat1, dtype=np.float64)
    lon2 = np.asarray(lon2, dtype=np.float64)
    lat2 = np.asarray(lat2, dtype=np.float64)
    lon1r = np.deg2rad(lon1)
    lat1r = np.deg2rad(lat1)
    lon2r = np.deg2rad(lon2)
    lat2r = np.deg2rad(lat2)
    x = (lon2r - lon1r) * np.cos((lat1r + lat2r) / 2.0)
    y = lat2r - lat1r
    return r * np.sqrt(x * x + y * y)


def compute_eval_errors_for_query(q_row: pd.Series, tile_df: pd.DataFrame, q_coord: Optional[CoordSpec], t_coord: Optional[CoordSpec]) -> np.ndarray:
    if q_coord is None or t_coord is None:
        return np.full(len(tile_df), np.nan, dtype=np.float32)

    qx = safe_float(q_row[q_coord.x_col])
    qy = safe_float(q_row[q_coord.y_col])
    tx = tile_df[t_coord.x_col].astype(float).to_numpy()
    ty = tile_df[t_coord.y_col].astype(float).to_numpy()

    if np.isnan(qx) or np.isnan(qy):
        return np.full(len(tile_df), np.nan, dtype=np.float32)

    if q_coord.mode == "xy_m" and t_coord.mode == "xy_m":
        return np.sqrt((tx - qx) ** 2 + (ty - qy) ** 2).astype(np.float32)
    if q_coord.mode == "lonlat" and t_coord.mode == "lonlat":
        return approx_lonlat_distance_m(qx, qy, tx, ty).astype(np.float32)

    raise RuntimeError(f"Coordinate mode mismatch: query={q_coord.mode}, tile={t_coord.mode}")


def read_image(path: Path) -> Optional[np.ndarray]:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    return img


def center_square_resize(img_bgr: np.ndarray, size: int) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    side = min(h, w)
    y0 = max(0, (h - side) // 2)
    x0 = max(0, (w - side) // 2)
    crop = img_bgr[y0:y0 + side, x0:x0 + side]
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA) if crop.shape[:2] != (size, size) else crop


def normalize01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    mn = float(np.nanmin(x))
    mx = float(np.nanmax(x))
    if not np.isfinite(mn) or not np.isfinite(mx) or mx <= mn + 1e-6:
        return np.zeros_like(x, dtype=np.float32)
    return (x - mn) / (mx - mn)


def preprocess_variant(img_bgr: np.ndarray, variant: str, resize: int) -> np.ndarray:
    img = center_square_resize(img_bgr, resize)

    if variant == "original_luma_phog":
        return normalize01(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))

    if variant == "v3_green_suppressed":
        b, g, r = cv2.split(img.astype(np.float32) / 255.0)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        green_excess = np.maximum(0.0, g - 0.5 * (r + b))
        return normalize01(gray * (1.0 - 0.65 * normalize01(green_excess)))

    if variant == "v5_edge_magnitude":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        return normalize01(np.sqrt(gx * gx + gy * gy))

    if variant == "v8_lab_logchroma_fused":
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
        l = lab[:, :, 0] / 255.0
        a = (lab[:, :, 1] - 128.0) / 128.0
        b = (lab[:, :, 2] - 128.0) / 128.0
        chroma = np.log1p(np.sqrt(a * a + b * b))
        return normalize01(0.65 * normalize01(l) + 0.35 * normalize01(chroma))

    if variant == "v9_canny_structure":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(gray, 60, 160)
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
        return normalize01(edges)

    raise ValueError(f"Unknown variant: {variant}")


def phog_descriptor(image01: np.ndarray, levels: int = 3, bins: int = 9, eps: float = 1e-8) -> np.ndarray:
    img = image01.astype(np.float32)
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    ang = (np.rad2deg(np.arctan2(gy, gx)) + 180.0) % 180.0
    h, w = img.shape
    bin_edges = np.linspace(0.0, 180.0, bins + 1, dtype=np.float32)
    feats = []
    for lev in range(levels + 1):
        g = 2 ** lev
        for yy in range(g):
            y0 = int(round(yy * h / g)); y1 = int(round((yy + 1) * h / g))
            for xx in range(g):
                x0 = int(round(xx * w / g)); x1 = int(round((xx + 1) * w / g))
                a = ang[y0:y1, x0:x1].ravel()
                m = mag[y0:y1, x0:x1].ravel()
                hist, _ = np.histogram(a, bins=bin_edges, weights=m)
                hist = hist.astype(np.float32)
                n = np.linalg.norm(hist)
                if n > eps:
                    hist /= n
                feats.append(hist)
    desc = np.concatenate(feats).astype(np.float32)
    n = np.linalg.norm(desc)
    if n > eps:
        desc /= n
    return desc


def descriptor_for_path(path: Path, variant: str, resize: int, levels: int, bins: int) -> Optional[np.ndarray]:
    img = read_image(path)
    if img is None:
        return None
    return phog_descriptor(preprocess_variant(img, variant, resize), levels=levels, bins=bins)


def build_or_load_sat_descriptors(tile_df: pd.DataFrame, path_col: str, tile_id_col: str, variant: str, cache_dir: Path, repo_root: Path, resize: int, levels: int, bins: int, force_rebuild: bool) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, Dict]:
    ensure_dir(cache_dir)
    cache_path = cache_dir / f"s5c1_sat_desc_{variant}_r{resize}_l{levels}_b{bins}.npz"
    if cache_path.exists() and not force_rebuild:
        z = np.load(cache_path, allow_pickle=True)
        ok_mask = z["ok_mask"].astype(bool)

        if len(ok_mask) == len(tile_df):
            kept_df = tile_df.loc[ok_mask].reset_index(drop=True).copy()
            info = {
                "variant": variant,
                "cache_path": str(cache_path),
                "loaded_from_cache": True,
                "satellite_tiles_total": int(len(tile_df)),
                "satellite_tiles_with_descriptors": int(len(kept_df)),
                "failed_tiles": int((~ok_mask).sum()),
            }
            return z["tile_ids"], z["desc"].astype(np.float32), kept_df, info

        print(
            f"WARN: cache tile-count mismatch for {variant}: "
            f"cache has {len(ok_mask)}, current run has {len(tile_df)}. Rebuilding."
        )

    descs, tile_ids, ok_mask = [], [], []
    t0 = now_s()
    print(f"\nBuilding satellite descriptor cache for {variant}")
    print(f"Tiles: {len(tile_df)}")
    for i, row in tile_df.iterrows():
        if i % 500 == 0:
            print(f"  {i}/{len(tile_df)}")
        p = resolve_path(row[path_col], repo_root)
        d = descriptor_for_path(p, variant, resize, levels, bins)
        if d is None:
            ok_mask.append(False)
            continue
        descs.append(d)
        tile_ids.append(row[tile_id_col])
        ok_mask.append(True)
    ok_mask_np = np.asarray(ok_mask, dtype=bool)
    if not descs:
        raise RuntimeError(f"No satellite descriptors could be built for variant {variant}")
    desc = np.vstack(descs).astype(np.float32)
    tile_ids_np = np.asarray(tile_ids)
    np.savez_compressed(cache_path, tile_ids=tile_ids_np, desc=desc, ok_mask=ok_mask_np)
    kept_df = tile_df.loc[ok_mask_np].reset_index(drop=True).copy()
    info = {
        "variant": variant,
        "cache_path": str(cache_path),
        "loaded_from_cache": False,
        "satellite_tiles_total": int(len(tile_df)),
        "satellite_tiles_with_descriptors": int(len(kept_df)),
        "failed_tiles": int((~ok_mask_np).sum()),
        "build_time_s": float(now_s() - t0),
    }
    return tile_ids_np, desc, kept_df, info


def rank_candidates(q_desc: np.ndarray, sat_desc: np.ndarray, topk: int) -> Tuple[np.ndarray, np.ndarray]:
    sims = sat_desc @ q_desc.astype(np.float32)
    k = min(topk, len(sims))
    idx = np.argpartition(-sims, k - 1)[:k]
    idx = idx[np.argsort(-sims[idx])]
    return idx, sims[idx]


def summarize_topk_errors(errors: np.ndarray, ks: List[int], threshold_m: float) -> Dict:
    out = {}
    if len(errors) == 0 or not np.isfinite(errors).any():
        for k in ks:
            out[f"oracle_at_{k}"] = False
            out[f"best_top{k}_error_m"] = np.nan
        out["first_rank_under_threshold"] = np.nan
        return out
    for k in ks:
        k2 = min(k, len(errors))
        top = errors[:k2]
        out[f"oracle_at_{k}"] = bool(np.nanmin(top) <= threshold_m)
        out[f"best_top{k}_error_m"] = float(np.nanmin(top))
    first = np.where(errors <= threshold_m)[0]
    out["first_rank_under_threshold"] = float(first[0] + 1) if len(first) else np.nan
    return out


def compute_gap_to_next_success(df: pd.DataFrame, success_col: str, dist_col: Optional[str]) -> pd.Series:
    success_idx = df.index[df[success_col].fillna(False).astype(bool)].to_numpy()
    if len(success_idx) == 0:
        return pd.Series([np.nan] * len(df), index=df.index)
    vals = []
    if dist_col and dist_col in df.columns:
        d = df[dist_col].astype(float).to_numpy()
        for i in df.index:
            nxt = success_idx[success_idx >= i]
            vals.append(float(d[nxt[0]] - d[i]) if len(nxt) else np.nan)
    else:
        f = df["sequence_frame_id"].astype(float).to_numpy()
        for i in df.index:
            nxt = success_idx[success_idx >= i]
            vals.append(float(f[nxt[0]] - f[i]) if len(nxt) else np.nan)
    return pd.Series(vals, index=df.index)


def build_union_for_token(token_rows: pd.DataFrame, topk: int, ks: List[int], threshold_m: float) -> Tuple[pd.DataFrame, Dict]:
    if token_rows.empty:
        return token_rows.copy(), {}
    group = token_rows.groupby("tile_id").agg(
        token0_id=("token0_id", "first"),
        sequence_frame_id=("sequence_frame_id", "first"),
        best_source_rank=("candidate_rank", "min"),
        max_similarity=("similarity", "max"),
        support_count=("variant", "nunique"),
        variants=("variant", lambda x: ",".join(sorted(set(map(str, x))))),
        eval_error_m=("eval_error_m", "min"),
    ).reset_index()
    group = group.sort_values(["support_count", "max_similarity", "best_source_rank"], ascending=[False, False, True]).reset_index(drop=True)
    group["union_rank"] = np.arange(1, len(group) + 1)
    group = group.head(topk).copy()
    errors = group["eval_error_m"].astype(float).to_numpy()
    summ = summarize_topk_errors(errors, ks, threshold_m)
    summ.update({
        "mode": "union_domain_pool",
        "token0_id": int(group["token0_id"].iloc[0]) if len(group) else None,
        "sequence_frame_id": int(group["sequence_frame_id"].iloc[0]) if len(group) else None,
        "status": "ok",
        "top1_tile_id": int(group["tile_id"].iloc[0]) if len(group) else None,
        "top1_error_m": float(group["eval_error_m"].iloc[0]) if len(group) else np.nan,
        "top1_support_count": int(group["support_count"].iloc[0]) if len(group) else 0,
        "top1_similarity": float(group["max_similarity"].iloc[0]) if len(group) else np.nan,
        "candidates_saved": int(len(group)),
    })
    return group, summ


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequence", default="traj01")
    ap.add_argument("--manifest", type=Path, default=Path("outputs/satloc/metadata/s5c_temporal/s5c0_absolute_query_manifest.csv"))
    ap.add_argument("--satellite-index", type=Path, default=Path("outputs/satloc/metadata/satellite_tiles_index_enriched.csv"))
    ap.add_argument("--output-root", type=Path, default=Path("outputs/satloc"))
    ap.add_argument("--repo-root", type=Path, default=Path("."))
    ap.add_argument("--variants", default="original_luma_phog,v3_green_suppressed,v5_edge_magnitude,v8_lab_logchroma_fused,v9_canny_structure")
    ap.add_argument("--union-variants", default="v3_green_suppressed,v5_edge_magnitude,v8_lab_logchroma_fused,v9_canny_structure")
    ap.add_argument("--resize", type=int, default=512)
    ap.add_argument("--phog-levels", type=int, default=3)
    ap.add_argument("--phog-bins", type=int, default=9)
    ap.add_argument("--topk-save", type=int, default=50)
    ap.add_argument("--oracle-ks", default="10,20,50")
    ap.add_argument("--threshold-m", type=float, default=40.0)
    ap.add_argument("--max-queries", type=int, default=None)
    ap.add_argument("--max-tiles", type=int, default=None)
    ap.add_argument("--force-rebuild-cache", action="store_true")
    ap.add_argument("--skip-union", action="store_true")
    args = ap.parse_args()

    variants = parse_csv_list(args.variants, str)
    union_variants = set(parse_csv_list(args.union_variants, str))
    oracle_ks = sorted(set(parse_csv_list(args.oracle_ks, int)))

    meta = args.output_root / "metadata" / "s5c_temporal"
    reports = args.output_root / "reports" / "s5c_temporal"
    figs = args.output_root / "figures" / "s5c_temporal"
    cache = meta / "cache"
    for d in [meta, reports, figs, cache]:
        ensure_dir(d)

    t_all = now_s()
    print_header("S5C.1 Temporal Candidate-Pool Coverage Benchmark")

    manifest = pd.read_csv(args.manifest).sort_values("sequence_frame_id").reset_index(drop=True)
    if args.max_queries is not None:
        manifest = manifest.iloc[: args.max_queries].copy()
    tiles = pd.read_csv(args.satellite_index)
    if args.max_tiles is not None:
        tiles = tiles.iloc[: args.max_tiles].copy()

    token_col = first_existing_col(manifest, ["token0_id", "token"], label="manifest token")
    seq_frame_col = first_existing_col(manifest, ["sequence_frame_id", "frame_index_in_sequence", "global_frame_index"], label="manifest sequence frame")
    uav_path_col = first_existing_col(manifest, ["uav_image_path", "image_path", "frame_path", "path", "uav_path"], label="manifest UAV image path")
    tile_id_col = first_existing_col(tiles, ["tile_id", "tile_index", "sat_tile_id", "id"], label="tile id")
    tile_path_col = first_existing_col(tiles, ["sat_image_path", "satellite_image_path", "image_path", "tile_path", "path", "file_path"], label="satellite tile image path")
    q_coord = find_xy_cols(manifest, "query manifest")
    t_coord = find_xy_cols(tiles, "satellite index")
    dist_col = first_existing_col(manifest, ["reference_distance_m", "cumulative_reference_distance_m", "cum_distance_m", "reference_cumulative_distance_m"], required=False, label="reference distance")

    print(f"Sequence:                 {args.sequence}")
    print(f"Queries:                  {len(manifest)}")
    print(f"Satellite tiles:          {len(tiles)}")
    print(f"Variants:                 {', '.join(variants)}")
    print(f"Union variants:           {', '.join(sorted(union_variants))}")
    print(f"Top-K saved:              {args.topk_save}")
    print(f"Oracle Ks:                {oracle_ks}")
    print(f"Threshold [m]:            {args.threshold_m}")
    print(f"UAV path column:          {uav_path_col}")
    print(f"Tile path column:         {tile_path_col}")
    print(f"Query coordinate mode:    {q_coord.mode if q_coord else 'none'}")
    print(f"Tile coordinate mode:     {t_coord.mode if t_coord else 'none'}")
    if args.max_queries or args.max_tiles:
        print("WARN: running with smoke/debug limits.")

    all_candidate_rows = []
    all_query_summaries = []
    cache_infos = []

    for variant in variants:
        v_t0 = now_s()
        tile_ids, sat_desc, kept_tiles, cache_info = build_or_load_sat_descriptors(
            tiles, tile_path_col, tile_id_col, variant, cache, args.repo_root,
            args.resize, args.phog_levels, args.phog_bins, args.force_rebuild_cache
        )
        cache_infos.append(cache_info)
        print(f"\nRanking queries for variant: {variant}")
        print(f"Descriptor matrix: {sat_desc.shape}")

        for qi, q in manifest.iterrows():
            if qi % 25 == 0:
                print(f"  query {qi}/{len(manifest)}")
            q_path = resolve_path(q[uav_path_col], args.repo_root)
            q_desc = descriptor_for_path(q_path, variant, args.resize, args.phog_levels, args.phog_bins)
            token = int(q[token_col])
            seq_frame = int(q[seq_frame_col])
            if q_desc is None:
                row = {"mode": variant, "token0_id": token, "sequence_frame_id": seq_frame, "status": "query_image_read_failed", "candidates_saved": 0}
                for k in oracle_ks:
                    row[f"oracle_at_{k}"] = False
                    row[f"best_top{k}_error_m"] = np.nan
                row["first_rank_under_threshold"] = np.nan
                all_query_summaries.append(row)
                continue

            idx, sims = rank_candidates(q_desc, sat_desc, args.topk_save)
            top_tiles = kept_tiles.iloc[idx].copy().reset_index(drop=True)
            top_errors = compute_eval_errors_for_query(q, top_tiles, q_coord, t_coord)
            cand = pd.DataFrame({
                "mode": variant,
                "variant": variant,
                "token0_id": token,
                "sequence_frame_id": seq_frame,
                "candidate_rank": np.arange(1, len(idx) + 1),
                "tile_id": top_tiles[tile_id_col].to_numpy(),
                "similarity": sims.astype(float),
                "eval_error_m": top_errors.astype(float),
                "hit_le_threshold_eval_only": top_errors <= args.threshold_m,
            })
            all_candidate_rows.append(cand)
            summ = summarize_topk_errors(cand["eval_error_m"].astype(float).to_numpy(), oracle_ks, args.threshold_m)
            summ.update({
                "mode": variant,
                "token0_id": token,
                "sequence_frame_id": seq_frame,
                "status": "ok",
                "top1_tile_id": int(cand["tile_id"].iloc[0]) if len(cand) else None,
                "top1_error_m": float(cand["eval_error_m"].iloc[0]) if len(cand) else np.nan,
                "top1_similarity": float(cand["similarity"].iloc[0]) if len(cand) else np.nan,
                "candidates_saved": int(len(cand)),
            })
            all_query_summaries.append(summ)
        print(f"Finished {variant} in {(now_s() - v_t0)/60.0:.2f} min")

    candidate_scores = pd.concat(all_candidate_rows, ignore_index=True) if all_candidate_rows else pd.DataFrame()
    out_candidate = meta / f"s5c1_candidate_scores_top{args.topk_save}.csv"
    candidate_scores.to_csv(out_candidate, index=False)

    union_scores = pd.DataFrame()
    if not args.skip_union and not candidate_scores.empty:
        print("\nBuilding union candidate pools")
        use = candidate_scores[candidate_scores["variant"].isin(union_variants)].copy()
        union_rows = []
        union_summaries = []
        for token, g in use.groupby("token0_id", sort=True):
            pool, summ = build_union_for_token(g, args.topk_save, oracle_ks, args.threshold_m)
            if len(pool):
                pool["mode"] = "union_domain_pool"
                union_rows.append(pool)
            if summ:
                union_summaries.append(summ)
        if union_rows:
            union_scores = pd.concat(union_rows, ignore_index=True)
        all_query_summaries.extend(union_summaries)

    out_union = meta / f"s5c1_union_candidate_scores_top{args.topk_save}.csv"
    union_scores.to_csv(out_union, index=False)

    query_summary = pd.DataFrame(all_query_summaries).sort_values(["mode", "sequence_frame_id"]).reset_index(drop=True)
    source_cols = [c for c in ["token0_id", "query_source", "is_uniform", "is_existing73", "is_relative_risk", "risk_reason"] if c in manifest.columns]
    if "token0_id" in source_cols:
        query_summary = query_summary.merge(manifest[source_cols].drop_duplicates("token0_id"), on="token0_id", how="left")
    out_qsum = meta / "s5c1_query_summary.csv"
    query_summary.to_csv(out_qsum, index=False)

    variant_rows = []
    for mode, g in query_summary.groupby("mode"):
        row = {
            "mode": mode,
            "queries": int(len(g)),
            "status_ok": int((g["status"] == "ok").sum()) if "status" in g.columns else int(len(g)),
            "median_top1_error_m": float(g["top1_error_m"].median()) if "top1_error_m" in g.columns else np.nan,
            "median_first_rank_under_threshold": float(g["first_rank_under_threshold"].median()) if "first_rank_under_threshold" in g.columns else np.nan,
        }
        for k in oracle_ks:
            col = f"oracle_at_{k}"
            be = f"best_top{k}_error_m"
            row[f"oracle_at_{k}_hits"] = int(g[col].fillna(False).astype(bool).sum()) if col in g.columns else 0
            row[f"oracle_at_{k}_rate"] = float(g[col].fillna(False).astype(bool).mean()) if col in g.columns and len(g) else np.nan
            row[f"median_best_top{k}_error_m"] = float(g[be].median()) if be in g.columns else np.nan
        variant_rows.append(row)
    variant_summary = pd.DataFrame(variant_rows).sort_values("mode").reset_index(drop=True)
    out_vsum = meta / "s5c1_variant_summary.csv"
    variant_summary.to_csv(out_vsum, index=False)

    timeline_rows = []
    for mode, g in query_summary.groupby("mode"):
        g = g.sort_values("sequence_frame_id").reset_index(drop=True)
        for k in oracle_ks:
            col = f"oracle_at_{k}"
            if col not in g.columns:
                continue
            gap_col = f"gap_to_next_oracle_at_{k}"
            g2 = g.copy()
            g2[gap_col] = compute_gap_to_next_success(g2, col, dist_col)
            g2["oracle_k"] = k
            timeline_rows.append(g2[["mode", "token0_id", "sequence_frame_id", col, gap_col, "oracle_k"]])
    gap_df = pd.concat(timeline_rows, ignore_index=True) if timeline_rows else pd.DataFrame()
    out_gap = meta / "s5c1_gap_to_next_oracle_success.csv"
    gap_df.to_csv(out_gap, index=False)

    # Figures.
    try:
        plt.figure(figsize=(8, 4.5))
        modes = list(variant_summary["mode"])
        x = np.arange(len(modes))
        width = 0.8 / max(1, len(oracle_ks))
        for j, k in enumerate(oracle_ks):
            vals = variant_summary[f"oracle_at_{k}_rate"].to_numpy()
            plt.bar(x + (j - (len(oracle_ks) - 1) / 2) * width, vals, width=width, label=f"@{k}")
        plt.xticks(x, modes, rotation=30, ha="right")
        plt.ylabel(f"Oracle recall ≤ {args.threshold_m:.0f} m")
        plt.ylim(0, 1.0)
        plt.legend()
        plt.tight_layout()
        plt.savefig(figs / "s5c1_oracle_recall_by_k.png", dpi=180)
        plt.close()
    except Exception as e:
        print(f"WARN: failed oracle recall figure: {e}")

    try:
        main_mode = "union_domain_pool" if "union_domain_pool" in set(query_summary["mode"]) else variants[0]
        g = query_summary[query_summary["mode"] == main_mode]
        plt.figure(figsize=(7, 4))
        plt.hist(g[f"best_top{max(oracle_ks)}_error_m"].dropna(), bins=40)
        plt.axvline(args.threshold_m, linestyle="--")
        plt.xlabel(f"Best top-{max(oracle_ks)} error [m] eval only")
        plt.ylabel("Queries")
        plt.title(main_mode)
        plt.tight_layout()
        plt.savefig(figs / "s5c1_best_top50_error_hist.png", dpi=180)
        plt.close()
    except Exception as e:
        print(f"WARN: failed error histogram: {e}")

    try:
        main_mode = "union_domain_pool" if "union_domain_pool" in set(query_summary["mode"]) else variants[0]
        g = query_summary[query_summary["mode"] == main_mode].sort_values("sequence_frame_id")
        plt.figure(figsize=(9, 3))
        y = g[f"oracle_at_{max(oracle_ks)}"].fillna(False).astype(int)
        plt.scatter(g["sequence_frame_id"], y, s=10)
        plt.yticks([0, 1], ["miss", "oracle hit"])
        plt.xlabel("Sequence frame")
        plt.title(f"{main_mode} Oracle@{max(oracle_ks)} timeline")
        plt.tight_layout()
        plt.savefig(figs / "s5c1_success_timeline.png", dpi=180)
        plt.close()
    except Exception as e:
        print(f"WARN: failed timeline figure: {e}")

    try:
        if not gap_df.empty:
            main_mode = "union_domain_pool" if "union_domain_pool" in set(gap_df["mode"]) else variants[0]
            g = gap_df[(gap_df["mode"] == main_mode) & (gap_df["oracle_k"] == max(oracle_ks))]
            gap_col = f"gap_to_next_oracle_at_{max(oracle_ks)}"
            plt.figure(figsize=(7, 4))
            plt.hist(g[gap_col].dropna(), bins=30)
            plt.xlabel("Gap to next oracle success [m or frames]")
            plt.ylabel("Queries")
            plt.title(f"{main_mode} gap to next Oracle@{max(oracle_ks)}")
            plt.tight_layout()
            plt.savefig(figs / "s5c1_query_gap_to_next_oracle_success.png", dpi=180)
            plt.close()
    except Exception as e:
        print(f"WARN: failed gap figure: {e}")

    report = {
        "stage": "S5C.1",
        "sequence": args.sequence,
        "manifest": str(args.manifest),
        "satellite_index": str(args.satellite_index),
        "queries": int(len(manifest)),
        "satellite_tiles": int(len(tiles)),
        "variants": variants,
        "union_variants": sorted(list(union_variants)),
        "topk_save": int(args.topk_save),
        "oracle_ks": oracle_ks,
        "threshold_m": float(args.threshold_m),
        "resize": int(args.resize),
        "phog_levels": int(args.phog_levels),
        "phog_bins": int(args.phog_bins),
        "coordinate_query_mode": q_coord.mode if q_coord else None,
        "coordinate_tile_mode": t_coord.mode if t_coord else None,
        "cache_info": cache_infos,
        "outputs": {
            "candidate_scores": str(out_candidate),
            "union_candidate_scores": str(out_union),
            "query_summary": str(out_qsum),
            "variant_summary": str(out_vsum),
            "gap_to_next_success": str(out_gap),
            "figures_dir": str(figs),
        },
        "variant_summary": variant_summary.to_dict(orient="records"),
        "runtime_s": float(now_s() - t_all),
    }
    out_report = reports / "s5c1_candidate_pool_coverage_summary.json"
    with open(out_report, "w") as f:
        json.dump(report, f, indent=2)

    print()
    print_header("S5C.1 Candidate-Pool Coverage Summary")
    print(f"Queries processed:         {len(manifest)}")
    print(f"Satellite tiles:           {len(tiles)}")
    print(f"Candidate rows saved:      {len(candidate_scores)}")
    print(f"Union rows saved:          {len(union_scores)}")
    print(f"Runtime:                   {(now_s() - t_all)/60.0:.2f} min")

    print("\nOracle recall by mode")
    print("---------------------")
    cols = ["mode", "queries"] + [f"oracle_at_{k}_rate" for k in oracle_ks] + [f"oracle_at_{k}_hits" for k in oracle_ks]
    print(variant_summary[cols].to_string(index=False))

    print("\nSaved outputs")
    print("-------------")
    print(out_candidate)
    print(out_union)
    print(out_qsum)
    print(out_vsum)
    print(out_gap)
    print(out_report)
    print(figs)

    print("\nDecision reminder")
    print("-----------------")
    print("Use S5C.1 only to decide candidate-pool coverage and LightGlue top-K.")
    print("Do not use oracle/error columns for online candidate ranking.")


if __name__ == "__main__":
    main()
