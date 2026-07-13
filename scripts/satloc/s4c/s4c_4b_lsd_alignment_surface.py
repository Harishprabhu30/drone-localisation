#!/usr/bin/env python3
"""
S4C.4B — LSD / skeleton distance-transform alignment surface.

Purpose:
  Test whether vector-line or skeleton canvases produce a more meaningful
  alignment basin than raw macro-contour Chamfer.

For each selected token:
  Query UAV canvas is compared against:
    - GT/nearest tile, eval-only diagnostic
    - PHOG top-1 tile
    - Oracle best PHOG top-N tile, eval-only diagnostic

For each pair:
  - build line/skeleton canvas
  - compute distance-transform alignment surface over dx,dy shifts
  - save heatmap panel
  - record min score, best shift, basin/sharpness metrics

Important:
  - No retrieval ranking is performed here.
  - GT/oracle rows use reference/evaluation information only for diagnosis.
  - This is not a final reranker yet.

code used:
selected_subset from traj01 = 1,40,50,58,60,67,74,79,90,100,107,117,129,139,166,259,269,276,288,300,310,326,336,350,366,387,405,421,434,450,474,482,494,503,516,533,546,564,573,577,591,614,631,653,662,679,694,710,731,746,760,768,781,794,808,820,833,844,874,886,905,914,927,937,946,952,963,971,990,1003,1015,1024,1034 
1. Using Luma LSD:
export PYTHONPATH=$PWD/src

python scripts/satloc/s4c/s4c_4b_lsd_alignment_surface.py \
  --sequence traj01 \
  --tokens 1,40,60,90,100,129,166,269,516,905 \
  --preprocess luma \
  --resize-size 512 \
  --edge-method sobel \
  --blur-ksize 3 \
  --threshold-mode percentile \
  --threshold-percentile 65 \
  --close-ksize 3 \
  --open-ksize 1 \
  --min-component-area 65 \
  --min-line-length 24 \
  --max-lines 120 \
  --orientation-bins 18 \
  --oracle-top-n 50 \
  --canvas-mode luma_lsd \
  --line-thickness 2 \
  --max-shift-px 48 \
  --shift-step-px 8 \
  --symmetric-weight 0.25 \
  --max-points 8000

2. using skeleton LSD:
python scripts/satloc/s4c/s4c_4b_lsd_alignment_surface.py \
  --sequence traj01 \
  --tokens 1,40,60,90,100,129,166,269,516,905 \
  --preprocess luma \
  --resize-size 512 \
  --edge-method sobel \
  --blur-ksize 3 \
  --threshold-mode percentile \
  --threshold-percentile 65 \
  --close-ksize 3 \
  --open-ksize 1 \
  --min-component-area 65 \
  --min-line-length 24 \
  --max-lines 120 \
  --orientation-bins 18 \
  --oracle-top-n 50 \
  --canvas-mode skeleton_lsd \
  --line-thickness 2 \
  --max-shift-px 48 \
  --shift-step-px 8 \
  --symmetric-weight 0.25 \
  --max-points 8000
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


THIS_DIR = Path(__file__).resolve().parent

S4C0_PATH = THIS_DIR / "s4c_0_macrocontour_preflight.py"
S4C4A_PATH = THIS_DIR / "s4c_4a_vector_skeleton_preflight.py"

if not S4C0_PATH.exists():
    raise FileNotFoundError(f"Missing helper: {S4C0_PATH}")
if not S4C4A_PATH.exists():
    raise FileNotFoundError(f"Missing helper: {S4C4A_PATH}")

spec0 = importlib.util.spec_from_file_location("s4c0_helpers", S4C0_PATH)
s4c0 = importlib.util.module_from_spec(spec0)
sys.modules["s4c0_helpers"] = s4c0
assert spec0.loader is not None
spec0.loader.exec_module(s4c0)

spec4a = importlib.util.spec_from_file_location("s4c4a_helpers", S4C4A_PATH)
s4c4a = importlib.util.module_from_spec(spec4a)
sys.modules["s4c4a_helpers"] = s4c4a
assert spec4a.loader is not None
spec4a.loader.exec_module(s4c4a)


OUT_ROOT = Path("outputs/satloc")
DEFAULT_UAV_INDEX = OUT_ROOT / "metadata/uav_frames_index_enriched.csv"
DEFAULT_SAT_INDEX = OUT_ROOT / "metadata/satellite_tiles_index_enriched.csv"
DEFAULT_S4C1_DIR = OUT_ROOT / "metadata/s4c_macrocontour_phog_chamfer/s4c1_phog_retrieval"

OUT_FIG_DIR = OUT_ROOT / "figures/s4c_macrocontour_phog_chamfer/s4c4_vector_skeleton/s4c4b_alignment_surface"
OUT_META_DIR = OUT_ROOT / "metadata/s4c_macrocontour_phog_chamfer/s4c4_vector_skeleton"
OUT_REPORT_DIR = OUT_ROOT / "reports/satloc/s4c_macrocontour_phog_chamfer/s4c4_vector_skeleton"


def parse_tokens(text: str) -> list[int]:
    out = []
    for p in text.split(","):
        p = p.strip()
        if p:
            out.append(int(p))
    return out


def safe_float(x: Any, default: float = np.nan) -> float:
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    return obj


# -----------------------------
# Canvas creation
# -----------------------------

def lines_to_canvas(lines: np.ndarray, shape: tuple[int, int], thickness: int) -> np.ndarray:
    h, w = shape
    canvas = np.zeros((h, w), dtype=np.uint8)

    if lines is None or len(lines) == 0:
        return canvas

    for x1, y1, x2, y2 in lines.astype(int):
        cv2.line(canvas, (x1, y1), (x2, y2), 255, thickness, cv2.LINE_AA)

    return canvas


def get_alignment_canvas(diag: dict[str, Any], mode: str, thickness: int) -> np.ndarray:
    mode = mode.lower()

    if mode == "macro_contour":
        return (diag["macro"].contour_canvas > 0).astype(np.uint8) * 255

    if mode == "skeleton":
        return (diag["skeleton"] > 0).astype(np.uint8) * 255

    if mode == "luma_lsd":
        return lines_to_canvas(
            diag["lines_luma"],
            diag["macro"].luma.shape,
            thickness=thickness,
        )

    if mode == "skeleton_lsd":
        return lines_to_canvas(
            diag["lines_skeleton"],
            diag["macro"].luma.shape,
            thickness=thickness,
        )

    raise ValueError(f"Unknown canvas mode: {mode}")


def contour_points(binary: np.ndarray, max_points: int) -> np.ndarray:
    ys, xs = np.where(binary > 0)
    if len(xs) == 0:
        return np.zeros((0, 2), dtype=np.int32)

    pts = np.column_stack([xs, ys]).astype(np.int32)

    if len(pts) > max_points:
        # deterministic uniform sub-sampling
        idx = np.linspace(0, len(pts) - 1, max_points).astype(np.int32)
        pts = pts[idx]

    return pts


def distance_transform_for_edges(binary_edges: np.ndarray) -> np.ndarray:
    edges = (binary_edges > 0).astype(np.uint8)
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


def alignment_surface(
    query_canvas: np.ndarray,
    candidate_canvas: np.ndarray,
    max_shift_px: int,
    shift_step_px: int,
    symmetric_weight: float,
    max_points: int,
) -> dict[str, Any]:
    q = (query_canvas > 0).astype(np.uint8) * 255
    c = (candidate_canvas > 0).astype(np.uint8) * 255

    q_pts = contour_points(q, max_points=max_points)
    c_pts = contour_points(c, max_points=max_points)

    h, w = q.shape[:2]
    diag = float(math.sqrt(h * h + w * w))
    penalty = diag

    shifts = list(range(-max_shift_px, max_shift_px + 1, shift_step_px))
    if 0 not in shifts:
        shifts.append(0)
    shifts = sorted(set(shifts))

    surface = np.zeros((len(shifts), len(shifts)), dtype=np.float32)
    q_to_c_surface = np.zeros_like(surface)
    c_to_q_surface = np.zeros_like(surface)

    if len(q_pts) == 0 or len(c_pts) == 0:
        surface[:, :] = penalty
        best_y, best_x = 0, 0
    else:
        q_dt = distance_transform_for_edges(q)
        c_dt = distance_transform_for_edges(c)

        for yi, dy in enumerate(shifts):
            for xi, dx in enumerate(shifts):
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

                surface[yi, xi] = score
                q_to_c_surface[yi, xi] = q_to_c
                c_to_q_surface[yi, xi] = c_to_q

        best_flat = int(np.argmin(surface))
        best_y, best_x = np.unravel_index(best_flat, surface.shape)

    best_score = float(surface[best_y, best_x])
    center_idx = shifts.index(0)
    center_score = float(surface[center_idx, center_idx])

    finite = surface[np.isfinite(surface)]
    surface_mean = float(finite.mean()) if finite.size else float("nan")
    surface_std = float(finite.std()) if finite.size else float("nan")

    # Basin area: how many shifts are within tolerance of the best score.
    basin_tol = max(2.0, 0.10 * max(best_score, 1e-6))
    basin_area = int((surface <= best_score + basin_tol).sum())

    # Sharpness/contrast: higher means best alignment is more distinctive.
    contrast = float((surface_mean - best_score) / max(surface_std, 1e-6)) if np.isfinite(surface_std) else float("nan")

    return {
        "surface": surface,
        "q_to_c_surface": q_to_c_surface,
        "c_to_q_surface": c_to_q_surface,
        "shifts": shifts,
        "best_dx": int(shifts[best_x]),
        "best_dy": int(shifts[best_y]),
        "best_score": best_score,
        "center_score": center_score,
        "center_minus_best": float(center_score - best_score),
        "surface_mean": surface_mean,
        "surface_std": surface_std,
        "surface_contrast": contrast,
        "basin_tol": float(basin_tol),
        "basin_area": basin_area,
        "query_point_count": int(len(q_pts)),
        "candidate_point_count": int(len(c_pts)),
    }


def shifted_overlay(query_canvas: np.ndarray, candidate_canvas: np.ndarray, dx: int, dy: int) -> np.ndarray:
    q = (query_canvas > 0).astype(np.uint8)
    c = (candidate_canvas > 0).astype(np.uint8)

    h, w = q.shape[:2]
    shifted_q = np.zeros_like(q)

    xs, ys = np.meshgrid(np.arange(w), np.arange(h))
    src_x = xs - int(dx)
    src_y = ys - int(dy)
    valid = (src_x >= 0) & (src_x < w) & (src_y >= 0) & (src_y < h)

    shifted_q[valid] = q[src_y[valid], src_x[valid]]

    overlay = np.zeros((h, w, 3), dtype=np.uint8)

    # candidate = green, shifted query = red, overlap = yellow
    overlay[..., 1] = c * 180
    overlay[..., 0] = shifted_q * 220
    overlap = (c > 0) & (shifted_q > 0)
    overlay[overlap, 0] = 255
    overlay[overlap, 1] = 255

    return overlay


# -----------------------------
# Rendering
# -----------------------------

def render_token_panel(
    token: int,
    query_diag: dict[str, Any],
    query_canvas: np.ndarray,
    candidate_results: list[dict[str, Any]],
    mode: str,
    out_path: Path,
) -> None:
    if not candidate_results:
        return

    nrows = len(candidate_results)
    ncols = 6

    fig, axes = plt.subplots(nrows, ncols, figsize=(3.25 * ncols, 3.4 * nrows), squeeze=False)

    col_titles = [
        "UAV RGB",
        "Candidate RGB",
        f"UAV {mode}",
        f"Candidate {mode}",
        "alignment surface",
        "best shifted overlay",
    ]

    for r, result in enumerate(candidate_results):
        cand_diag = result["candidate_diag"]
        surface = result["surface_result"]["surface"]
        shifts = result["surface_result"]["shifts"]
        best_dx = result["surface_result"]["best_dx"]
        best_dy = result["surface_result"]["best_dy"]

        overlay = shifted_overlay(
            query_canvas,
            result["candidate_canvas"],
            dx=best_dx,
            dy=best_dy,
        )

        imgs = [
            query_diag["macro"].rgb,
            cand_diag["macro"].rgb,
            query_canvas,
            result["candidate_canvas"],
            surface,
            overlay,
        ]

        for c, img in enumerate(imgs):
            ax = axes[r, c]
            ax.set_xticks([])
            ax.set_yticks([])

            if c in [0, 1, 5]:
                ax.imshow(img)
            elif c in [2, 3]:
                ax.imshow(img, cmap="gray", vmin=0, vmax=255)
            elif c == 4:
                im = ax.imshow(
                    img,
                    extent=[min(shifts), max(shifts), max(shifts), min(shifts)],
                    aspect="auto",
                )
                ax.scatter([best_dx], [best_dy], marker="x", s=60)
                ax.set_xlabel("dx")
                ax.set_ylabel("dy")
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            if r == 0:
                ax.set_title(col_titles[c], fontsize=10)

        label = (
            f"{result['role']}\n"
            f"tile={result.get('tile_id')} rank={result.get('rank')}\n"
            f"err={result.get('center_error_m'):.1f}m\n"
            f"best={result['surface_result']['best_score']:.2f}px "
            f"dx={best_dx},dy={best_dy}\n"
            f"contrast={result['surface_result']['surface_contrast']:.2f} "
            f"basin={result['surface_result']['basin_area']}"
        )
        axes[r, 0].set_ylabel(label, fontsize=8)

    fig.suptitle(f"S4C.4B LSD/skeleton alignment surface — token {token} — mode={mode}", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=165)
    plt.close(fig)


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--sequence", default="traj01")
    parser.add_argument("--tokens", default="1,40,60,90,100,129,166,269,516,905")
    parser.add_argument("--uav-index", default=str(DEFAULT_UAV_INDEX))
    parser.add_argument("--sat-index", default=str(DEFAULT_SAT_INDEX))
    parser.add_argument("--s4c1-ranked-dir", default=str(DEFAULT_S4C1_DIR))

    # Macro settings must match S4C.4A / S4C.1.
    parser.add_argument("--preprocess", default="luma", choices=["gray", "luma", "clahe_luma"])
    parser.add_argument("--resize-size", type=int, default=512)
    parser.add_argument("--edge-method", default="sobel", choices=["sobel", "canny"])
    parser.add_argument("--blur-ksize", type=int, default=3)
    parser.add_argument("--threshold-mode", default="percentile", choices=["otsu", "percentile"])
    parser.add_argument("--threshold-percentile", type=float, default=65.0)
    parser.add_argument("--close-ksize", type=int, default=3)
    parser.add_argument("--open-ksize", type=int, default=1)
    parser.add_argument("--min-component-area", type=int, default=65)

    # LSD/vector settings.
    parser.add_argument("--min-line-length", type=float, default=24.0)
    parser.add_argument("--max-lines", type=int, default=120)
    parser.add_argument("--orientation-bins", type=int, default=18)
    parser.add_argument("--oracle-top-n", type=int, default=50)

    # Alignment settings.
    parser.add_argument(
        "--canvas-mode",
        default="luma_lsd",
        choices=["macro_contour", "skeleton", "luma_lsd", "skeleton_lsd"],
    )
    parser.add_argument("--line-thickness", type=int, default=2)
    parser.add_argument("--max-shift-px", type=int, default=48)
    parser.add_argument("--shift-step-px", type=int, default=8)
    parser.add_argument("--symmetric-weight", type=float, default=0.25)
    parser.add_argument("--max-points", type=int, default=8000)

    args = parser.parse_args()

    tokens = parse_tokens(args.tokens)

    OUT_FIG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_META_DIR.mkdir(parents=True, exist_ok=True)
    OUT_REPORT_DIR.mkdir(parents=True, exist_ok=True)

    uav_index = Path(args.uav_index)
    sat_index = Path(args.sat_index)
    ranked_dir = Path(args.s4c1_ranked_dir)

    if not uav_index.exists():
        raise FileNotFoundError(f"Missing UAV index: {uav_index}")
    if not sat_index.exists():
        raise FileNotFoundError(f"Missing satellite index: {sat_index}")
    if not ranked_dir.exists():
        raise FileNotFoundError(f"Missing S4C.1 ranked dir: {ranked_dir}")

    uav_df = pd.read_csv(uav_index)
    sat_df = pd.read_csv(sat_index)

    uav_df = s4c4a.prepare_uav_df(uav_df, args.sequence)
    filename_index, fallback_uav_dirs, fallback_sat_dirs = s4c4a.build_filename_index(args.sequence)

    print("S4C.4B LSD / skeleton alignment surface")
    print("---------------------------------------")
    print(f"Sequence:       {args.sequence}")
    print(f"Tokens:         {tokens}")
    print(f"Canvas mode:    {args.canvas_mode}")
    print(f"Shift search:   ±{args.max_shift_px}px step {args.shift_step_px}px")
    print(f"Sym weight:     {args.symmetric_weight}")
    print("")

    manifest_rows: list[dict[str, Any]] = []
    panel_paths: list[str] = []

    for token in tokens:
        sources = s4c4a.build_sources_for_token(
            token=token,
            uav_df=uav_df,
            sat_df=sat_df,
            ranked_dir=ranked_dir,
            filename_index=filename_index,
            fallback_uav_dirs=fallback_uav_dirs,
            fallback_sat_dirs=fallback_sat_dirs,
            args=args,
        )

        if not sources:
            print(f"[WARN] token {token}: no sources")
            continue

        query_source = None
        candidate_sources = []

        for source in sources:
            if source["role"] == "uav_query":
                query_source = source
            else:
                candidate_sources.append(source)

        if query_source is None:
            print(f"[WARN] token {token}: no UAV query source")
            continue

        try:
            query_diag = s4c4a.compute_vector_diagnostic(Path(query_source["path"]), args)
            query_canvas = get_alignment_canvas(query_diag, args.canvas_mode, args.line_thickness)
        except Exception as exc:
            print(f"[WARN] token {token}: query diagnostic failed: {exc}")
            continue

        candidate_results: list[dict[str, Any]] = []

        for cand_source in candidate_sources:
            try:
                cand_diag = s4c4a.compute_vector_diagnostic(Path(cand_source["path"]), args)
                cand_canvas = get_alignment_canvas(cand_diag, args.canvas_mode, args.line_thickness)

                surf = alignment_surface(
                    query_canvas=query_canvas,
                    candidate_canvas=cand_canvas,
                    max_shift_px=args.max_shift_px,
                    shift_step_px=args.shift_step_px,
                    symmetric_weight=args.symmetric_weight,
                    max_points=args.max_points,
                )
            except Exception as exc:
                print(f"[WARN] token {token} role={cand_source['role']}: {exc}")
                continue

            result = {
                "token": token,
                "role": cand_source["role"],
                "selection": cand_source["selection"],
                "tile_id": cand_source["tile_id"],
                "rank": cand_source["rank"],
                "center_error_m": safe_float(cand_source["center_error_m"]),
                "score_cosine": safe_float(cand_source["score_cosine"]),
                "candidate_diag": cand_diag,
                "candidate_canvas": cand_canvas,
                "surface_result": surf,
                "candidate_image_path": str(cand_source["path"]),
            }

            candidate_results.append(result)

            row = {
                "sequence": args.sequence,
                "token": token,
                "canvas_mode": args.canvas_mode,
                "role": cand_source["role"],
                "selection": cand_source["selection"],
                "tile_id": cand_source["tile_id"],
                "rank": cand_source["rank"],
                "center_error_m": safe_float(cand_source["center_error_m"]),
                "score_cosine": safe_float(cand_source["score_cosine"]),
                "candidate_image_path": str(cand_source["path"]),
                "query_point_count": surf["query_point_count"],
                "candidate_point_count": surf["candidate_point_count"],
                "best_score": surf["best_score"],
                "center_score": surf["center_score"],
                "center_minus_best": surf["center_minus_best"],
                "best_dx": surf["best_dx"],
                "best_dy": surf["best_dy"],
                "surface_mean": surf["surface_mean"],
                "surface_std": surf["surface_std"],
                "surface_contrast": surf["surface_contrast"],
                "basin_tol": surf["basin_tol"],
                "basin_area": surf["basin_area"],
                "query_canvas_density": float((query_canvas > 0).mean()),
                "candidate_canvas_density": float((cand_canvas > 0).mean()),
            }

            manifest_rows.append(row)

        if candidate_results:
            fig_path = OUT_FIG_DIR / (
                f"s4c4b_token{token:04d}_{args.canvas_mode}"
                f"_shift{args.max_shift_px}s{args.shift_step_px}_alignment_surface.png"
            )
            render_token_panel(
                token=token,
                query_diag=query_diag,
                query_canvas=query_canvas,
                candidate_results=candidate_results,
                mode=args.canvas_mode,
                out_path=fig_path,
            )
            panel_paths.append(str(fig_path))
            print(f"[OK] token {token}: saved {fig_path}")

    manifest_df = pd.DataFrame(manifest_rows)

    manifest_csv = OUT_META_DIR / (
        f"s4c4b_{args.canvas_mode}_alignment_surface_manifest.csv"
    )
    manifest_df.to_csv(manifest_csv, index=False)

    summary = {
        "stage": "S4C.4B_LSD_skeleton_alignment_surface",
        "sequence": args.sequence,
        "tokens_requested": tokens,
        "canvas_mode": args.canvas_mode,
        "num_rows": len(manifest_rows),
        "manifest_csv": str(manifest_csv),
        "panel_paths": panel_paths,
        "settings": vars(args),
        "notes": [
            "Diagnostic only; no final reranking performed.",
            "GT/nearest and oracle-best rows use evaluation/reference information only for diagnosis.",
            "Lower best_score means better line/skeleton alignment under translation.",
            "surface_contrast and basin_area describe whether the alignment surface has a distinctive basin.",
        ],
    }

    if len(manifest_df) > 0:
        metric_cols = [
            "role",
            "best_score",
            "center_minus_best",
            "surface_contrast",
            "basin_area",
            "candidate_canvas_density",
            "candidate_point_count",
            "center_error_m",
        ]
        existing = [c for c in metric_cols if c in manifest_df.columns]
        role_summary = manifest_df[existing].groupby("role").mean(numeric_only=True).reset_index()
        summary["role_mean_metrics"] = role_summary.to_dict(orient="records")

    summary_json = OUT_REPORT_DIR / (
        f"s4c4b_{args.canvas_mode}_alignment_surface_summary.json"
    )

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(json_safe(summary), f, indent=2)

    print("")
    print("S4C.4B complete")
    print("----------------")
    print(f"Manifest CSV: {manifest_csv}")
    print(f"Summary JSON: {summary_json}")
    print(f"Figures:      {OUT_FIG_DIR}")

    if len(manifest_df) > 0:
        print("")
        print("Compact role averages")
        print("---------------------")
        cols = [
            "role",
            "best_score",
            "center_minus_best",
            "surface_contrast",
            "basin_area",
            "candidate_canvas_density",
            "candidate_point_count",
            "center_error_m",
        ]
        cols = [c for c in cols if c in manifest_df.columns]
        print(manifest_df[cols].groupby("role").mean(numeric_only=True).to_string())

        print("")
        print("Per-token best scores")
        print("---------------------")
        show_cols = [
            "token",
            "role",
            "center_error_m",
            "best_score",
            "surface_contrast",
            "basin_area",
            "best_dx",
            "best_dy",
        ]
        print(manifest_df[show_cols].to_string(index=False))


if __name__ == "__main__":
    main()
