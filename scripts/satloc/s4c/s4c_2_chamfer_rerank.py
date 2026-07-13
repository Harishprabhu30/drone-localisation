#!/usr/bin/env python3
"""
S4C.2 — Chamfer reranking for SatLoc macro-contour PHOG retrieval.

Input:
  S4C.1 PHOG ranked CSVs.

Pipeline:
  PHOG top-N candidates
  -> macro-contour extraction
  -> distance-transform Chamfer distance
  -> optional small translation search
  -> rerank candidates
  -> before/after comparison

Important:
  UAV filename lon/lat is used only for evaluation/debug after ranking.
  It is not used for PHOG scoring or Chamfer scoring.

code to run:

export PYTHONPATH=$PWD/src

python scripts/satloc/s4c/s4c_2_chamfer_rerank.py \
  --sequence traj01 \
  --tokens 1,40,50,58,60,67,74,79,90,100,107,117,129,139,166,259,269,276,288,300,310,326,336,350,366,387,405,421,434,450,474,482,494,503,516,533,546,564,573,577,591,614,631,653,662,679,694,710,731,746,760,768,781,794,808,820,833,844,874,886,905,914,927,937,946,952,963,971,990,1003,1015,1024,1034 \
  --preprocess luma \
  --resize-size 512 \
  --edge-method sobel \
  --blur-ksize 3 \
  --threshold-mode percentile \
  --threshold-percentile 65 \
  --close-ksize 3 \
  --open-ksize 1 \
  --min-component-area 65 \
  --phog-top-n 50 \
  --display-top-k 10 \
  --max-shift-px 24 \
  --shift-step-px 8 \
  --symmetric-weight 0.5 \
  --max-points 8000
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Optional

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


THIS_DIR = Path(__file__).resolve().parent
S4C0_PATH = THIS_DIR / "s4c_0_macrocontour_preflight.py"

if not S4C0_PATH.exists():
    raise FileNotFoundError(f"Missing helper: {S4C0_PATH}")

spec = importlib.util.spec_from_file_location("s4c0_helpers", S4C0_PATH)
s4c0 = importlib.util.module_from_spec(spec)
sys.modules["s4c0_helpers"] = s4c0
assert spec.loader is not None
spec.loader.exec_module(s4c0)


OUT_ROOT = Path("outputs/satloc")
S4C1_META_DIR = OUT_ROOT / "metadata/s4c_macrocontour_phog_chamfer/s4c1_phog_retrieval"

OUT_FIG_DIR = OUT_ROOT / "figures/s4c_macrocontour_phog_chamfer/s4c2_chamfer_rerank"
OUT_META_DIR = OUT_ROOT / "metadata/s4c_macrocontour_phog_chamfer/s4c2_chamfer_rerank"
OUT_REPORT_DIR = OUT_ROOT / "reports/s4c_macrocontour_phog_chamfer/s4c2_chamfer_rerank"


# ------------------------------------------------------------
# Small helpers
# ------------------------------------------------------------

def parse_tokens(text: str) -> list[int]:
    out: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    return out


def settings_slug(args: argparse.Namespace) -> str:
    thresh = (
        "otsu"
        if args.threshold_mode == "otsu"
        else f"pct{str(args.threshold_percentile).replace('.', 'p')}"
    )
    shift = f"shift{args.max_shift_px}s{args.shift_step_px}"
    return (
        f"macro_{args.preprocess}_{args.edge_method}_{thresh}"
        f"_b{args.blur_ksize}"
        f"_c{args.close_ksize}"
        f"_o{args.open_ksize}"
        f"_area{args.min_component_area}"
        f"_r{args.resize_size}"
        f"_chamfer_top{args.phog_top_n}_{shift}"
        f"_sym{str(args.symmetric_weight).replace('.', 'p')}"
    )


def prepare_uav_df(uav_df: pd.DataFrame, sequence: str) -> pd.DataFrame:
    seq_col = s4c0.find_col(uav_df, ["sequence", "seq", "trajectory", "traj"], required=False)
    if seq_col is not None:
        uav_df = uav_df[uav_df[seq_col].astype(str) == sequence].copy()
    else:
        uav_df = uav_df.copy()

    tokens = []
    for _, row in uav_df.iterrows():
        tokens.append(s4c0.get_uav_token(row, uav_df))

    uav_df["_s4c_token"] = tokens
    return uav_df


def build_filename_index(sequence: str) -> tuple[dict[str, Path], list[Path], list[Path]]:
    fallback_uav_dirs = [
        Path("data/raw/satloc/part_1/UAV Data") / sequence,
        Path("data/raw/satloc/part_1/UAV Data"),
    ]

    fallback_sat_dirs = [
        Path("data/raw/satloc/part_1/Satellite Data/sat_image_ref"),
        Path("data/raw/satloc/part_1/Satellite Data"),
    ]

    filename_index = s4c0.build_filename_index(fallback_uav_dirs + fallback_sat_dirs)
    return filename_index, fallback_uav_dirs, fallback_sat_dirs


def compute_macro(image_path: Path, args: argparse.Namespace) -> Any:
    return s4c0.macro_contour_pipeline(
        image_path=image_path,
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


def find_latest_ranked_csv(token: int, ranked_dir: Path) -> Path:
    pattern = f"s4c1_token{token:04d}_*_ranked.csv"
    matches = sorted(ranked_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    if not matches:
        raise FileNotFoundError(
            f"Could not find S4C.1 ranked CSV for token {token} using pattern:\n"
            f"{ranked_dir / pattern}"
        )
    return matches[0]


def first_rank_under(df: pd.DataFrame, rank_col: str, err_col: str, threshold_m: float) -> Optional[int]:
    valid = df[np.isfinite(df[err_col]) & (df[err_col] <= threshold_m)]
    if len(valid) == 0:
        return None
    return int(valid[rank_col].min())


def best_topk_error(df: pd.DataFrame, err_col: str, k: int) -> float:
    if len(df) == 0:
        return float("nan")
    return float(df.head(k)[err_col].min())


# ------------------------------------------------------------
# Chamfer functions
# ------------------------------------------------------------

def contour_points(binary: np.ndarray, max_points: int) -> np.ndarray:
    ys, xs = np.where(binary > 0)
    if len(xs) == 0:
        return np.zeros((0, 2), dtype=np.int32)

    pts = np.column_stack([xs, ys]).astype(np.int32)

    if len(pts) > max_points:
        idx = np.linspace(0, len(pts) - 1, max_points).astype(np.int32)
        pts = pts[idx]

    return pts


def distance_transform_for_edges(binary_edges: np.ndarray) -> np.ndarray:
    edges = (binary_edges > 0).astype(np.uint8)
    # distanceTransform gives distance to nearest zero pixel.
    # Therefore edges must be zero and background 255.
    inv = (1 - edges) * 255
    return cv2.distanceTransform(inv.astype(np.uint8), cv2.DIST_L2, 3)


def sample_dt_at_shift(
    dt: np.ndarray,
    pts_xy: np.ndarray,
    dx: int,
    dy: int,
    out_of_bounds_penalty: float,
) -> float:
    if pts_xy.size == 0:
        return out_of_bounds_penalty

    h, w = dt.shape[:2]
    xs = pts_xy[:, 0].astype(np.int32) + int(dx)
    ys = pts_xy[:, 1].astype(np.int32) + int(dy)

    valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)

    if not np.any(valid):
        return out_of_bounds_penalty

    values = np.empty((len(xs),), dtype=np.float32)
    values[~valid] = out_of_bounds_penalty
    values[valid] = dt[ys[valid], xs[valid]]

    return float(values.mean())


def chamfer_score_with_shift_search(
    query_edges: np.ndarray,
    candidate_edges: np.ndarray,
    max_shift_px: int,
    shift_step_px: int,
    symmetric_weight: float,
    max_points: int,
) -> dict[str, Any]:
    """
    Shift convention:
      dx, dy means query contour is shifted by (dx, dy) into candidate coordinates.

    q_to_c:
      query points + shift sampled on candidate distance transform.

    c_to_q:
      candidate points - shift sampled on query distance transform.
    """
    q_pts = contour_points(query_edges, max_points=max_points)
    c_pts = contour_points(candidate_edges, max_points=max_points)

    diag = float(math.sqrt(query_edges.shape[0] ** 2 + query_edges.shape[1] ** 2))
    penalty = diag

    if len(q_pts) == 0 or len(c_pts) == 0:
        return {
            "chamfer_score_px": penalty,
            "q_to_c_px": penalty,
            "c_to_q_px": penalty,
            "best_dx": 0,
            "best_dy": 0,
            "query_points": int(len(q_pts)),
            "candidate_points": int(len(c_pts)),
        }

    q_dt = distance_transform_for_edges(query_edges)
    c_dt = distance_transform_for_edges(candidate_edges)

    shifts = list(range(-max_shift_px, max_shift_px + 1, shift_step_px))
    if 0 not in shifts:
        shifts.append(0)
        shifts = sorted(set(shifts))

    best = {
        "chamfer_score_px": float("inf"),
        "q_to_c_px": float("inf"),
        "c_to_q_px": float("inf"),
        "best_dx": 0,
        "best_dy": 0,
    }

    for dy in shifts:
        for dx in shifts:
            q_to_c = sample_dt_at_shift(
                dt=c_dt,
                pts_xy=q_pts,
                dx=dx,
                dy=dy,
                out_of_bounds_penalty=penalty,
            )

            c_to_q = sample_dt_at_shift(
                dt=q_dt,
                pts_xy=c_pts,
                dx=-dx,
                dy=-dy,
                out_of_bounds_penalty=penalty,
            )

            score = q_to_c + symmetric_weight * c_to_q

            if score < best["chamfer_score_px"]:
                best = {
                    "chamfer_score_px": float(score),
                    "q_to_c_px": float(q_to_c),
                    "c_to_q_px": float(c_to_q),
                    "best_dx": int(dx),
                    "best_dy": int(dy),
                }

    best["query_points"] = int(len(q_pts))
    best["candidate_points"] = int(len(c_pts))
    return best


# ------------------------------------------------------------
# Plotting
# ------------------------------------------------------------

def render_before_after_panel(
    token: int,
    q_macro: Any,
    gt_macro: Any,
    gt_label: str,
    phog_top: pd.DataFrame,
    chamfer_top: pd.DataFrame,
    sat_df: pd.DataFrame,
    macro_cache: dict[int, Any],
    out_path: Path,
) -> None:
    k = max(len(phog_top), len(chamfer_top))
    if k == 0:
        return

    fig, axes = plt.subplots(4, k, figsize=(3.0 * k, 11.5), squeeze=False)

    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis("off")

    axes[0, 0].imshow(q_macro.rgb)
    axes[0, 0].set_title("UAV query RGB", fontsize=9)

    if k > 1:
        axes[0, 1].imshow(q_macro.contour_canvas, cmap="gray", vmin=0, vmax=255)
        axes[0, 1].set_title("UAV contour", fontsize=9)

    if k > 2:
        axes[0, 2].imshow(gt_macro.rgb)
        axes[0, 2].set_title("GT/near RGB", fontsize=9)

    if k > 3:
        axes[0, 3].imshow(gt_macro.contour_canvas, cmap="gray", vmin=0, vmax=255)
        axes[0, 3].set_title(gt_label, fontsize=8)

    axes[1, 0].set_ylabel("PHOG top-k", fontsize=10)
    axes[2, 0].set_ylabel("Chamfer top-k", fontsize=10)
    axes[3, 0].set_ylabel("Chamfer contours", fontsize=10)

    for col, (_, row) in enumerate(phog_top.iterrows()):
        row_pos = int(row["row_pos"])
        macro = macro_cache.get(row_pos)
        if macro is None:
            continue

        err = float(row["center_error_m"])
        title = (
            f"P{int(row['rank'])} tile {int(row['tile_id'])}\n"
            f"err={err:.1f}m"
        )
        axes[1, col].imshow(macro.rgb)
        axes[1, col].set_title(title, fontsize=8)

    for col, (_, row) in enumerate(chamfer_top.iterrows()):
        row_pos = int(row["row_pos"])
        macro = macro_cache.get(row_pos)
        if macro is None:
            continue

        err = float(row["center_error_m"])
        title = (
            f"C{int(row['chamfer_rank'])} tile {int(row['tile_id'])}\n"
            f"err={err:.1f}m ch={float(row['chamfer_score_px']):.2f}"
        )

        axes[2, col].imshow(macro.rgb)
        axes[2, col].set_title(title, fontsize=8)

        axes[3, col].imshow(macro.contour_canvas, cmap="gray", vmin=0, vmax=255)
        axes[3, col].set_title(
            f"dx={int(row['best_dx'])}, dy={int(row['best_dy'])}",
            fontsize=8,
        )

    fig.suptitle(f"S4C.2 Chamfer reranking — token {token}", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def render_error_barplot(summary_df: pd.DataFrame, out_path: Path) -> None:
    if len(summary_df) == 0:
        return

    labels = summary_df["token"].astype(str).tolist()
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.bar(x - width / 2, summary_df["phog_top1_error_m"], width, label="PHOG top-1")
    ax.bar(x + width / 2, summary_df["chamfer_top1_error_m"], width, label="Chamfer top-1")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("top-1 center error [m]")
    ax.set_xlabel("query token")
    ax.set_title("S4C.2 top-1 error before/after Chamfer reranking")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


# ------------------------------------------------------------
# Per-query reranking
# ------------------------------------------------------------

def run_token(
    token: int,
    args: argparse.Namespace,
    uav_df: pd.DataFrame,
    sat_df: pd.DataFrame,
    filename_index: dict[str, Path],
    fallback_uav_dirs: list[Path],
    fallback_sat_dirs: list[Path],
) -> Optional[dict[str, Any]]:
    ranked_csv = find_latest_ranked_csv(token, Path(args.s4c1_ranked_dir))
    ranked_df = pd.read_csv(ranked_csv)

    if len(ranked_df) == 0:
        print(f"[WARN] token {token}: empty ranked CSV")
        return None

    # Ensure original PHOG rank exists.
    if "rank" not in ranked_df.columns:
        ranked_df["rank"] = np.arange(1, len(ranked_df) + 1)

    ranked_df = ranked_df.sort_values("rank").reset_index(drop=True)
    candidates = ranked_df.head(args.phog_top_n).copy()

    match = uav_df[uav_df["_s4c_token"] == token]
    if len(match) == 0:
        print(f"[WARN] token {token}: UAV row not found")
        return None

    uav_row = match.iloc[0]
    uav_lon, uav_lat = s4c0.get_lon_lat(uav_row, uav_df)

    uav_path = s4c0.get_row_path(
        uav_row,
        uav_df,
        filename_index,
        fallback_uav_dirs,
        kind="uav",
    )

    if uav_path is None:
        print(f"[WARN] token {token}: UAV image path not found")
        return None

    gt_row, gt_method, gt_error_m = s4c0.select_gt_or_nearest_tile(
        sat_df,
        float(uav_lon),
        float(uav_lat),
    )

    gt_tile_id = s4c0.get_tile_id(gt_row, sat_df)
    gt_path = s4c0.get_row_path(
        gt_row,
        sat_df,
        filename_index,
        fallback_sat_dirs,
        kind="sat",
    )

    if gt_path is None:
        print(f"[WARN] token {token}: GT/nearest tile image path not found")
        return None

    t0 = time.perf_counter()

    q_macro = compute_macro(uav_path, args)
    gt_macro = compute_macro(gt_path, args)

    rerank_rows: list[dict[str, Any]] = []
    macro_cache: dict[int, Any] = {}

    print(f"[token {token}] ranked_csv={ranked_csv.name}")
    print(f"[token {token}] reranking PHOG top {len(candidates)}")

    for i, (_, cand) in enumerate(candidates.iterrows(), start=1):
        row_pos = int(cand["row_pos"])
        sat_row = sat_df.iloc[row_pos]

        cand_path = s4c0.get_row_path(
            sat_row,
            sat_df,
            filename_index,
            fallback_sat_dirs,
            kind="sat",
        )

        if cand_path is None:
            continue

        try:
            cand_macro = compute_macro(cand_path, args)
        except Exception as exc:
            print(f"[WARN] token {token} row_pos={row_pos}: {exc}")
            continue

        macro_cache[row_pos] = cand_macro

        chamfer = chamfer_score_with_shift_search(
            query_edges=q_macro.contour_canvas,
            candidate_edges=cand_macro.contour_canvas,
            max_shift_px=args.max_shift_px,
            shift_step_px=args.shift_step_px,
            symmetric_weight=args.symmetric_weight,
            max_points=args.max_points,
        )

        # Optional combined score: lower is better.
        # By default phog_weight = 0, so this is pure Chamfer.
        phog_score = float(cand["score_cosine"]) if "score_cosine" in cand else 0.0
        combined_score = float(chamfer["chamfer_score_px"]) - args.phog_score_weight * phog_score * args.resize_size

        out = cand.to_dict()
        out.update(chamfer)
        out["combined_rerank_score"] = combined_score
        out["candidate_cleaned_density"] = float(cand_macro.stats["cleaned_density"])
        out["candidate_contour_density"] = float(cand_macro.stats["contour_density"])
        out["query_cleaned_density"] = float(q_macro.stats["cleaned_density"])
        out["query_contour_density"] = float(q_macro.stats["contour_density"])
        out["candidate_image_path"] = str(cand_path)
        out["ranked_csv_input"] = str(ranked_csv)
        rerank_rows.append(out)

        if i % 10 == 0:
            print(f"[token {token}] processed {i}/{len(candidates)} candidates")

    if len(rerank_rows) == 0:
        print(f"[WARN] token {token}: no rerank rows")
        return None

    rerank_df = pd.DataFrame(rerank_rows)

    rerank_df = rerank_df.sort_values(
        ["combined_rerank_score", "chamfer_score_px", "rank"],
        ascending=[True, True, True],
    ).reset_index(drop=True)

    rerank_df["chamfer_rank"] = np.arange(1, len(rerank_df) + 1)

    # Keep original PHOG rank explicit.
    rerank_df = rerank_df.rename(columns={"rank": "phog_rank"})

    # For convenience after rename.
    if "phog_rank" in rerank_df.columns:
        pass

    slug = settings_slug(args)

    OUT_META_DIR.mkdir(parents=True, exist_ok=True)
    OUT_FIG_DIR.mkdir(parents=True, exist_ok=True)

    rerank_csv = OUT_META_DIR / f"s4c2_token{token:04d}_{slug}_reranked_top{args.phog_top_n}.csv"
    rerank_df.to_csv(rerank_csv, index=False)

    phog_top = candidates.head(args.display_top_k).copy()
    # Ensure render expects column name "rank".
    phog_top["rank"] = phog_top["rank"].astype(int)

    chamfer_top = rerank_df.head(args.display_top_k).copy()

    panel_path = OUT_FIG_DIR / f"s4c2_token{token:04d}_{slug}_before_after_top{args.display_top_k}.png"

    gt_label = f"GT/near tile {gt_tile_id}\n{gt_method}, err={gt_error_m:.1f}m"

    render_before_after_panel(
        token=token,
        q_macro=q_macro,
        gt_macro=gt_macro,
        gt_label=gt_label,
        phog_top=phog_top,
        chamfer_top=chamfer_top,
        sat_df=sat_df,
        macro_cache=macro_cache,
        out_path=panel_path,
    )

    runtime_s = time.perf_counter() - t0

    # Before metrics from full S4C.1 ranked CSV.
    phog_first_under_20 = first_rank_under(ranked_df, "rank", "center_error_m", 20.0)
    phog_first_under_40 = first_rank_under(ranked_df, "rank", "center_error_m", 40.0)
    phog_first_under_60 = first_rank_under(ranked_df, "rank", "center_error_m", 60.0)

    # After metrics from top-N reranked candidates only.
    chamfer_first_under_20 = first_rank_under(rerank_df, "chamfer_rank", "center_error_m", 20.0)
    chamfer_first_under_40 = first_rank_under(rerank_df, "chamfer_rank", "center_error_m", 40.0)
    chamfer_first_under_60 = first_rank_under(rerank_df, "chamfer_rank", "center_error_m", 60.0)

    phog_top1 = ranked_df.iloc[0]
    chamfer_top1 = rerank_df.iloc[0]

    summary = {
        "sequence": args.sequence,
        "token": token,
        "s4c1_ranked_csv": str(ranked_csv),
        "s4c2_reranked_csv": str(rerank_csv),
        "before_after_panel": str(panel_path),
        "runtime_s": runtime_s,
        "phog_top_n_used": args.phog_top_n,
        "display_top_k": args.display_top_k,
        "gt_tile_id": gt_tile_id,
        "gt_selection_method": gt_method,
        "gt_center_error_m": float(gt_error_m),

        "phog_top1_tile_id": int(phog_top1["tile_id"]),
        "phog_top1_error_m": float(phog_top1["center_error_m"]),
        "phog_best_display_topk_error_m": best_topk_error(ranked_df, "center_error_m", args.display_top_k),
        "phog_first_rank_under_20m": phog_first_under_20,
        "phog_first_rank_under_40m": phog_first_under_40,
        "phog_first_rank_under_60m": phog_first_under_60,

        "chamfer_top1_tile_id": int(chamfer_top1["tile_id"]),
        "chamfer_top1_error_m": float(chamfer_top1["center_error_m"]),
        "chamfer_top1_score_px": float(chamfer_top1["chamfer_score_px"]),
        "chamfer_top1_dx": int(chamfer_top1["best_dx"]),
        "chamfer_top1_dy": int(chamfer_top1["best_dy"]),
        "chamfer_best_display_topk_error_m": best_topk_error(rerank_df, "center_error_m", args.display_top_k),
        "chamfer_first_rank_under_20m": chamfer_first_under_20,
        "chamfer_first_rank_under_40m": chamfer_first_under_40,
        "chamfer_first_rank_under_60m": chamfer_first_under_60,

        "improved_top1_error": bool(float(chamfer_top1["center_error_m"]) < float(phog_top1["center_error_m"])),
        "settings": {
            "max_shift_px": args.max_shift_px,
            "shift_step_px": args.shift_step_px,
            "symmetric_weight": args.symmetric_weight,
            "max_points": args.max_points,
            "phog_score_weight": args.phog_score_weight,
        },
    }

    print(
        f"[OK] token {token}: "
        f"PHOG top1 {summary['phog_top1_error_m']:.1f}m -> "
        f"Chamfer top1 {summary['chamfer_top1_error_m']:.1f}m | "
        f"Chamfer first<=40m={summary['chamfer_first_rank_under_40m']} | "
        f"time={runtime_s:.1f}s"
    )

    return summary


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--sequence", default="traj01")
    parser.add_argument("--tokens", default="1,40,60,90,100,129,166")
    parser.add_argument("--uav-index", default="outputs/satloc/metadata/uav_frames_index_enriched.csv")
    parser.add_argument("--sat-index", default="outputs/satloc/metadata/satellite_tiles_index_enriched.csv")
    parser.add_argument("--s4c1-ranked-dir", default=str(S4C1_META_DIR))

    # Must match S4C.1 macro setting.
    parser.add_argument("--preprocess", default="luma", choices=["gray", "luma", "clahe_luma"])
    parser.add_argument("--resize-size", type=int, default=512)
    parser.add_argument("--edge-method", default="sobel", choices=["sobel", "canny"])
    parser.add_argument("--blur-ksize", type=int, default=3)
    parser.add_argument("--threshold-mode", default="percentile", choices=["otsu", "percentile"])
    parser.add_argument("--threshold-percentile", type=float, default=65.0)
    parser.add_argument("--close-ksize", type=int, default=3)
    parser.add_argument("--open-ksize", type=int, default=1)
    parser.add_argument("--min-component-area", type=int, default=65)

    # Rerank settings.
    parser.add_argument("--phog-top-n", type=int, default=50)
    parser.add_argument("--display-top-k", type=int, default=10)
    parser.add_argument("--max-shift-px", type=int, default=24)
    parser.add_argument("--shift-step-px", type=int, default=8)
    parser.add_argument("--symmetric-weight", type=float, default=0.5)
    parser.add_argument("--max-points", type=int, default=8000)
    parser.add_argument("--phog-score-weight", type=float, default=0.0)

    args = parser.parse_args()

    OUT_FIG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_META_DIR.mkdir(parents=True, exist_ok=True)
    OUT_REPORT_DIR.mkdir(parents=True, exist_ok=True)

    tokens = parse_tokens(args.tokens)

    uav_index = Path(args.uav_index)
    sat_index = Path(args.sat_index)

    if not uav_index.exists():
        raise FileNotFoundError(f"Missing UAV index: {uav_index}")
    if not sat_index.exists():
        raise FileNotFoundError(f"Missing satellite index: {sat_index}")

    uav_df = pd.read_csv(uav_index)
    sat_df = pd.read_csv(sat_index)

    uav_df = prepare_uav_df(uav_df, args.sequence)
    filename_index, fallback_uav_dirs, fallback_sat_dirs = build_filename_index(args.sequence)

    print("S4C.2 Chamfer reranking")
    print("-----------------------")
    print(f"Sequence:      {args.sequence}")
    print(f"Tokens:        {tokens}")
    print(f"PHOG top-N:    {args.phog_top_n}")
    print(f"Display top-k: {args.display_top_k}")
    print(f"Shift search:  ±{args.max_shift_px}px step {args.shift_step_px}px")
    print(f"Sym weight:    {args.symmetric_weight}")
    print(f"Max points:    {args.max_points}")
    print("")

    all_summaries: list[dict[str, Any]] = []

    for token in tokens:
        result = run_token(
            token=token,
            args=args,
            uav_df=uav_df,
            sat_df=sat_df,
            filename_index=filename_index,
            fallback_uav_dirs=fallback_uav_dirs,
            fallback_sat_dirs=fallback_sat_dirs,
        )
        if result is not None:
            all_summaries.append(result)

    summary_df = pd.DataFrame(all_summaries)

    slug = settings_slug(args)
    summary_csv = OUT_META_DIR / f"s4c2_{args.sequence}_{slug}_summary.csv"
    summary_json = OUT_REPORT_DIR / f"s4c2_{args.sequence}_{slug}_summary.json"
    error_plot = OUT_FIG_DIR / f"s4c2_{args.sequence}_{slug}_top1_error_before_after.png"

    summary_df.to_csv(summary_csv, index=False)

    render_error_barplot(summary_df, error_plot)

    aggregate: dict[str, Any] = {
        "stage": "S4C.2_chamfer_rerank",
        "sequence": args.sequence,
        "tokens_requested": tokens,
        "tokens_processed": [int(x["token"]) for x in all_summaries],
        "num_queries": len(all_summaries),
        "settings_slug": slug,
        "summary_csv": str(summary_csv),
        "error_plot": str(error_plot),
        "query_summaries": all_summaries,
        "notes": [
            "Chamfer reranking over PHOG top-N candidates.",
            "UAV lon/lat used only after ranking for evaluation/debug.",
            "No full-map Chamfer search is performed.",
        ],
    }

    if len(summary_df) > 0:
        aggregate.update(
            {
                "phog_mean_top1_error_m": float(summary_df["phog_top1_error_m"].mean()),
                "chamfer_mean_top1_error_m": float(summary_df["chamfer_top1_error_m"].mean()),
                "phog_median_top1_error_m": float(summary_df["phog_top1_error_m"].median()),
                "chamfer_median_top1_error_m": float(summary_df["chamfer_top1_error_m"].median()),
                "phog_top1_under_40m_rate": float((summary_df["phog_top1_error_m"] <= 40.0).mean()),
                "chamfer_top1_under_40m_rate": float((summary_df["chamfer_top1_error_m"] <= 40.0).mean()),
                "phog_display_topk_under_40m_rate": float((summary_df["phog_best_display_topk_error_m"] <= 40.0).mean()),
                "chamfer_display_topk_under_40m_rate": float((summary_df["chamfer_best_display_topk_error_m"] <= 40.0).mean()),
                "top1_error_improved_rate": float(summary_df["improved_top1_error"].mean()),
            }
        )

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(aggregate, f, indent=2)

    print("")
    print("S4C.2 complete")
    print("--------------")
    print(f"Summary CSV:  {summary_csv}")
    print(f"Summary JSON: {summary_json}")
    print(f"Error plot:   {error_plot}")
    print(f"Figures:      {OUT_FIG_DIR}")

    if len(summary_df) > 0:
        print("")
        print("Compact result")
        print("--------------")
        cols = [
            "token",
            "phog_top1_tile_id",
            "phog_top1_error_m",
            "chamfer_top1_tile_id",
            "chamfer_top1_error_m",
            "chamfer_first_rank_under_20m",
            "chamfer_first_rank_under_40m",
            "chamfer_first_rank_under_60m",
            "runtime_s",
        ]
        print(summary_df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
