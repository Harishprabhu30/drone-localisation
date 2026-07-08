#!/usr/bin/env python3
"""
S4C.4A — Vector skeleton / LSD preflight diagnostics.

Purpose:
  Inspect whether vectorized structural cues are cleaner than dense macro-contours.

For selected tokens, show:
  - RGB
  - macro-contour canvas from S4C.0/S4C.1 settings
  - morphological skeleton
  - LSD/Hough line overlay
  - line orientation histogram

Rows:
  - UAV query
  - GT/nearest tile, eval-only diagnostic
  - PHOG top-1 tile
  - oracle/best top-N tile, eval-only diagnostic

Important:
  - UAV lon/lat and center_error_m are used only for diagnostics/evaluation.
  - No retrieval ranking is performed here.
  - No final method claim is made here.

code Used:
selected_subset from traj01 = 1,40,50,58,60,67,74,79,90,100,107,117,129,139,166,259,269,276,288,300,310,326,336,350,366,387,405,421,434,450,474,482,494,503,516,533,546,564,573,577,591,614,631,653,662,679,694,710,731,746,760,768,781,794,808,820,833,844,874,886,905,914,927,937,946,952,963,971,990,1003,1015,1024,1034 
export PYTHONPATH=$PWD/src

python scripts/satloc/s4c/s4c_4a_vector_skeleton_preflight.py \
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
  --oracle-top-n 50
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
import sys
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
    raise FileNotFoundError(f"Missing helper script: {S4C0_PATH}")

spec = importlib.util.spec_from_file_location("s4c0_helpers", S4C0_PATH)
s4c0 = importlib.util.module_from_spec(spec)
sys.modules["s4c0_helpers"] = s4c0
assert spec.loader is not None
spec.loader.exec_module(s4c0)


OUT_ROOT = Path("outputs/satloc")
DEFAULT_UAV_INDEX = OUT_ROOT / "metadata/uav_frames_index_enriched.csv"
DEFAULT_SAT_INDEX = OUT_ROOT / "metadata/satellite_tiles_index_enriched.csv"
DEFAULT_S4C1_DIR = OUT_ROOT / "metadata/s4c_macrocontour_phog_chamfer/s4c1_phog_retrieval"

OUT_FIG_DIR = OUT_ROOT / "figures/s4c_macrocontour_phog_chamfer/s4c4_vector_skeleton/s4c4a_preflight"
OUT_META_DIR = OUT_ROOT / "metadata/s4c_macrocontour_phog_chamfer/s4c4_vector_skeleton"
OUT_REPORT_DIR = OUT_ROOT / "reports/s4c_macrocontour_phog_chamfer/s4c4_vector_skeleton"


# -----------------------------
# Basic helpers
# -----------------------------

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


def safe_int(x: Any, default: Optional[int] = None) -> Optional[int]:
    try:
        if pd.isna(x):
            return default
        return int(float(x))
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


def find_latest_ranked_csv(token: int, ranked_dir: Path) -> Optional[Path]:
    pattern = f"s4c1_token{token:04d}_*_ranked.csv"
    matches = sorted(ranked_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    if not matches:
        return None
    return matches[0]


def find_tile_by_id(sat_df: pd.DataFrame, tile_id: int) -> Optional[pd.Series]:
    for _, row in sat_df.iterrows():
        tid = s4c0.get_tile_id(row, sat_df)
        if tid == tile_id:
            return row
    return None


def get_uav_row_for_token(uav_df: pd.DataFrame, token: int) -> Optional[pd.Series]:
    m = uav_df[uav_df["_s4c_token"] == token]
    if len(m) == 0:
        return None
    return m.iloc[0]


# -----------------------------
# Skeleton + vector line extraction
# -----------------------------

def binary_skeletonize(binary: np.ndarray, max_iter: int = 500) -> np.ndarray:
    """
    Morphological skeletonization using OpenCV only.
    binary: uint8 0/255.
    """
    img = (binary > 0).astype(np.uint8) * 255
    skel = np.zeros_like(img)

    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    for _ in range(max_iter):
        opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        temp = cv2.subtract(img, opened)
        eroded = cv2.erode(img, kernel)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()

        if cv2.countNonZero(img) == 0:
            break

    return skel


def extract_line_segments(
    image_gray: np.ndarray,
    skeleton: np.ndarray,
    source: str,
    min_line_length: float,
    max_lines: int,
) -> np.ndarray:
    """
    Returns Nx4 line segments [x1, y1, x2, y2].

    source:
      - skeleton: detect from binary skeleton
      - luma: detect from luma/gray image
    """
    source = source.lower()

    if source == "skeleton":
        detect_img = skeleton.copy()
    elif source == "luma":
        detect_img = image_gray.copy()
    else:
        raise ValueError(f"Unknown LSD source: {source}")

    lines_out: list[list[float]] = []

    if hasattr(cv2, "createLineSegmentDetector"):
        try:
            lsd = cv2.createLineSegmentDetector(0)
            lines = lsd.detect(detect_img)[0]

            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0].astype(float).tolist()
                    length = math.hypot(x2 - x1, y2 - y1)
                    if length >= min_line_length:
                        lines_out.append([x1, y1, x2, y2])
        except Exception:
            lines_out = []

    # Fallback or supplement if LSD returns too few lines.
    if len(lines_out) == 0:
        edges = detect_img
        if source == "luma":
            edges = cv2.Canny(detect_img, 60, 160)

        hough = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180.0,
            threshold=30,
            minLineLength=int(min_line_length),
            maxLineGap=8,
        )

        if hough is not None:
            for line in hough[:, 0, :]:
                x1, y1, x2, y2 = line.astype(float).tolist()
                length = math.hypot(x2 - x1, y2 - y1)
                if length >= min_line_length:
                    lines_out.append([x1, y1, x2, y2])

    if len(lines_out) == 0:
        return np.zeros((0, 4), dtype=np.float32)

    arr = np.array(lines_out, dtype=np.float32)

    lengths = np.sqrt((arr[:, 2] - arr[:, 0]) ** 2 + (arr[:, 3] - arr[:, 1]) ** 2)
    order = np.argsort(-lengths)
    arr = arr[order]

    if len(arr) > max_lines:
        arr = arr[:max_lines]

    return arr


def line_stats(lines: np.ndarray, bins: int = 18) -> dict[str, Any]:
    if lines.size == 0:
        return {
            "line_count": 0,
            "long_line_count_40px": 0,
            "total_line_length_px": 0.0,
            "mean_line_length_px": 0.0,
            "median_line_length_px": 0.0,
            "max_line_length_px": 0.0,
            "orientation_entropy": 0.0,
            "dominant_orientation_deg": None,
            "orientation_hist": [0.0] * bins,
        }

    dx = lines[:, 2] - lines[:, 0]
    dy = lines[:, 3] - lines[:, 1]
    lengths = np.sqrt(dx * dx + dy * dy)

    angles = np.degrees(np.arctan2(dy, dx))
    angles = np.mod(angles, 180.0)

    hist, edges = np.histogram(
        angles,
        bins=bins,
        range=(0.0, 180.0),
        weights=lengths,
    )

    hist = hist.astype(np.float64)
    hist_sum = float(hist.sum())

    if hist_sum > 1e-9:
        prob = hist / hist_sum
        entropy = float(-(prob[prob > 0] * np.log(prob[prob > 0])).sum() / np.log(bins))
        dom_idx = int(np.argmax(hist))
        dominant_deg = float((edges[dom_idx] + edges[dom_idx + 1]) / 2.0)
    else:
        entropy = 0.0
        dominant_deg = None

    return {
        "line_count": int(len(lines)),
        "long_line_count_40px": int((lengths >= 40.0).sum()),
        "total_line_length_px": float(lengths.sum()),
        "mean_line_length_px": float(lengths.mean()),
        "median_line_length_px": float(np.median(lengths)),
        "max_line_length_px": float(lengths.max()),
        "orientation_entropy": entropy,
        "dominant_orientation_deg": dominant_deg,
        "orientation_hist": hist.tolist(),
    }


def draw_lines_overlay(rgb: np.ndarray, lines: np.ndarray) -> np.ndarray:
    overlay = rgb.copy()

    # No custom color request; red is used internally for diagnostic overlay.
    # OpenCV RGB color below.
    for x1, y1, x2, y2 in lines.astype(int):
        cv2.line(overlay, (x1, y1), (x2, y2), (255, 0, 0), 2, cv2.LINE_AA)

    return overlay


def line_orientation_hist_image(stats: dict[str, Any], size: int = 512) -> np.ndarray:
    hist = np.array(stats.get("orientation_hist", []), dtype=np.float64)
    if hist.size == 0:
        hist = np.zeros((18,), dtype=np.float64)

    fig, ax = plt.subplots(figsize=(3.0, 3.0))
    x = np.linspace(0, 180, len(hist), endpoint=False)
    width = 180.0 / max(1, len(hist))
    ax.bar(x, hist, width=width * 0.9, align="edge")
    ax.set_xlim(0, 180)
    ax.set_xlabel("orientation [deg]")
    ax.set_ylabel("length sum")
    ax.set_title("line orientation")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    img = cv2.cvtColor(buf, cv2.COLOR_RGBA2RGB)
    plt.close(fig)

    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    return img


def compute_vector_diagnostic(
    image_path: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    macro = s4c0.macro_contour_pipeline(
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

    skeleton = binary_skeletonize(macro.contour_canvas)

    # LSD from luma usually gives long structural edges; skeleton gives contour-derived lines.
    lines_luma = extract_line_segments(
        image_gray=macro.luma,
        skeleton=skeleton,
        source="luma",
        min_line_length=args.min_line_length,
        max_lines=args.max_lines,
    )

    lines_skel = extract_line_segments(
        image_gray=macro.luma,
        skeleton=skeleton,
        source="skeleton",
        min_line_length=args.min_line_length,
        max_lines=args.max_lines,
    )

    stats_luma = line_stats(lines_luma, bins=args.orientation_bins)
    stats_skel = line_stats(lines_skel, bins=args.orientation_bins)

    overlay_luma = draw_lines_overlay(macro.rgb, lines_luma)
    overlay_skel = draw_lines_overlay(macro.rgb, lines_skel)

    hist_luma = line_orientation_hist_image(stats_luma, size=args.resize_size)
    hist_skel = line_orientation_hist_image(stats_skel, size=args.resize_size)

    out = {
        "macro": macro,
        "skeleton": skeleton,
        "lines_luma": lines_luma,
        "lines_skeleton": lines_skel,
        "overlay_luma": overlay_luma,
        "overlay_skeleton": overlay_skel,
        "hist_luma_img": hist_luma,
        "hist_skeleton_img": hist_skel,
        "stats_luma": stats_luma,
        "stats_skeleton": stats_skel,
        "skeleton_density": float((skeleton > 0).mean()),
    }

    return out


# -----------------------------
# Candidate source construction
# -----------------------------

def build_sources_for_token(
    token: int,
    uav_df: pd.DataFrame,
    sat_df: pd.DataFrame,
    ranked_dir: Path,
    filename_index: dict[str, Path],
    fallback_uav_dirs: list[Path],
    fallback_sat_dirs: list[Path],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []

    uav_row = get_uav_row_for_token(uav_df, token)
    if uav_row is None:
        print(f"[WARN] token {token}: UAV row not found")
        return sources

    uav_lon, uav_lat = s4c0.get_lon_lat(uav_row, uav_df)
    uav_path = s4c0.get_row_path(
        uav_row,
        uav_df,
        filename_index,
        fallback_uav_dirs,
        kind="uav",
    )

    if uav_path is None:
        print(f"[WARN] token {token}: UAV image not found")
        return sources

    sources.append(
        {
            "role": "uav_query",
            "label": f"UAV token {token}",
            "path": uav_path,
            "tile_id": None,
            "rank": None,
            "center_error_m": 0.0,
            "score_cosine": None,
            "selection": "query",
        }
    )

    if uav_lon is not None and uav_lat is not None:
        gt_row, gt_method, gt_err = s4c0.select_gt_or_nearest_tile(
            sat_df,
            float(uav_lon),
            float(uav_lat),
        )
        gt_path = s4c0.get_row_path(
            gt_row,
            sat_df,
            filename_index,
            fallback_sat_dirs,
            kind="sat",
        )
        gt_tile_id = s4c0.get_tile_id(gt_row, sat_df)

        if gt_path is not None:
            sources.append(
                {
                    "role": "gt_or_nearest_eval_only",
                    "label": f"GT/near tile {gt_tile_id}\nerr={gt_err:.1f}m",
                    "path": gt_path,
                    "tile_id": gt_tile_id,
                    "rank": None,
                    "center_error_m": gt_err,
                    "score_cosine": None,
                    "selection": gt_method,
                }
            )

    ranked_csv = find_latest_ranked_csv(token, ranked_dir)

    if ranked_csv is not None:
        ranked = pd.read_csv(ranked_csv)
        if "rank" not in ranked.columns:
            ranked["rank"] = np.arange(1, len(ranked) + 1)

        ranked = ranked.sort_values("rank").reset_index(drop=True)

        if len(ranked) > 0:
            top1 = ranked.iloc[0]
            sat_row = sat_df.iloc[int(top1["row_pos"])]
            path = s4c0.get_row_path(sat_row, sat_df, filename_index, fallback_sat_dirs, kind="sat")

            if path is not None:
                sources.append(
                    {
                        "role": "phog_top1",
                        "label": f"PHOG top1 tile {int(top1['tile_id'])}\nerr={safe_float(top1.get('center_error_m')):.1f}m",
                        "path": path,
                        "tile_id": int(top1["tile_id"]),
                        "rank": int(top1["rank"]),
                        "center_error_m": safe_float(top1.get("center_error_m")),
                        "score_cosine": safe_float(top1.get("score_cosine")),
                        "selection": "phog_top1",
                    }
                )

        topn = ranked.head(args.oracle_top_n).copy()
        if len(topn) > 0 and "center_error_m" in topn.columns:
            oracle = topn.sort_values("center_error_m").iloc[0]
            sat_row = sat_df.iloc[int(oracle["row_pos"])]
            path = s4c0.get_row_path(sat_row, sat_df, filename_index, fallback_sat_dirs, kind="sat")

            if path is not None:
                sources.append(
                    {
                        "role": "oracle_best_topn_eval_only",
                        "label": f"Oracle top{args.oracle_top_n} tile {int(oracle['tile_id'])}\nR{int(oracle['rank'])}, err={safe_float(oracle.get('center_error_m')):.1f}m",
                        "path": path,
                        "tile_id": int(oracle["tile_id"]),
                        "rank": int(oracle["rank"]),
                        "center_error_m": safe_float(oracle.get("center_error_m")),
                        "score_cosine": safe_float(oracle.get("score_cosine")),
                        "selection": "eval_only_best_topn",
                    }
                )
    else:
        print(f"[WARN] token {token}: no S4C.1 ranked CSV found; only query/GT shown")

    return sources


# -----------------------------
# Rendering
# -----------------------------

def render_token_panel(token: int, rows: list[dict[str, Any]], out_path: Path) -> None:
    cols = [
        ("RGB", "rgb"),
        ("macro contour", "macro_contour"),
        ("skeleton", "skeleton"),
        ("LSD on luma", "overlay_luma"),
        ("LSD on skeleton", "overlay_skeleton"),
        ("luma line hist", "hist_luma_img"),
        ("skeleton line hist", "hist_skeleton_img"),
    ]

    nrows = len(rows)
    ncols = len(cols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 3.35 * nrows), squeeze=False)

    for r, item in enumerate(rows):
        diag = item["diag"]
        label = item["label"]

        for c, (title, key) in enumerate(cols):
            ax = axes[r, c]
            ax.set_xticks([])
            ax.set_yticks([])

            if key == "rgb":
                ax.imshow(diag["macro"].rgb)
            elif key == "macro_contour":
                ax.imshow(diag["macro"].contour_canvas, cmap="gray", vmin=0, vmax=255)
            elif key == "skeleton":
                ax.imshow(diag["skeleton"], cmap="gray", vmin=0, vmax=255)
            else:
                ax.imshow(diag[key])

            if r == 0:
                ax.set_title(title, fontsize=10)
            if c == 0:
                ax.set_ylabel(label, fontsize=8)

        st_luma = diag["stats_luma"]
        st_skel = diag["stats_skeleton"]

        axes[r, 4].set_xlabel(
            f"LSD luma n={st_luma['line_count']} len={st_luma['total_line_length_px']:.0f}px\n"
            f"skel n={st_skel['line_count']} len={st_skel['total_line_length_px']:.0f}px",
            fontsize=7,
        )

    fig.suptitle(f"S4C.4A vector skeleton / LSD preflight — token {token}", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170)
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

    # Same macro settings as selected S4C.1.
    parser.add_argument("--preprocess", default="luma", choices=["gray", "luma", "clahe_luma"])
    parser.add_argument("--resize-size", type=int, default=512)
    parser.add_argument("--edge-method", default="sobel", choices=["sobel", "canny"])
    parser.add_argument("--blur-ksize", type=int, default=3)
    parser.add_argument("--threshold-mode", default="percentile", choices=["otsu", "percentile"])
    parser.add_argument("--threshold-percentile", type=float, default=65.0)
    parser.add_argument("--close-ksize", type=int, default=3)
    parser.add_argument("--open-ksize", type=int, default=1)
    parser.add_argument("--min-component-area", type=int, default=65)

    # Vector/LSD settings.
    parser.add_argument("--min-line-length", type=float, default=24.0)
    parser.add_argument("--max-lines", type=int, default=120)
    parser.add_argument("--orientation-bins", type=int, default=18)
    parser.add_argument("--oracle-top-n", type=int, default=50)

    args = parser.parse_args()

    tokens = parse_tokens(args.tokens)

    uav_index = Path(args.uav_index)
    sat_index = Path(args.sat_index)
    ranked_dir = Path(args.s4c1_ranked_dir)

    if not uav_index.exists():
        raise FileNotFoundError(f"Missing UAV index: {uav_index}")
    if not sat_index.exists():
        raise FileNotFoundError(f"Missing satellite index: {sat_index}")
    if not ranked_dir.exists():
        raise FileNotFoundError(f"Missing S4C.1 ranked dir: {ranked_dir}")

    OUT_FIG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_META_DIR.mkdir(parents=True, exist_ok=True)
    OUT_REPORT_DIR.mkdir(parents=True, exist_ok=True)

    uav_df = pd.read_csv(uav_index)
    sat_df = pd.read_csv(sat_index)
    uav_df = prepare_uav_df(uav_df, args.sequence)

    filename_index, fallback_uav_dirs, fallback_sat_dirs = build_filename_index(args.sequence)

    print("S4C.4A Vector skeleton / LSD preflight")
    print("--------------------------------------")
    print(f"Sequence:        {args.sequence}")
    print(f"Tokens:          {tokens}")
    print(f"Resize:          {args.resize_size}")
    print(f"Macro threshold: {args.threshold_mode} {args.threshold_percentile}")
    print(f"Min line length: {args.min_line_length}")
    print("")

    manifest_rows: list[dict[str, Any]] = []
    panel_paths: list[str] = []

    for token in tokens:
        sources = build_sources_for_token(
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

        panel_rows: list[dict[str, Any]] = []

        for source in sources:
            try:
                diag = compute_vector_diagnostic(Path(source["path"]), args)
            except Exception as exc:
                print(f"[WARN] token {token} role={source['role']}: {exc}")
                continue

            panel_rows.append(
                {
                    "label": source["label"],
                    "diag": diag,
                }
            )

            row = {
                "sequence": args.sequence,
                "token": token,
                "role": source["role"],
                "selection": source["selection"],
                "image_path": str(source["path"]),
                "tile_id": source["tile_id"],
                "rank": source["rank"],
                "center_error_m": source["center_error_m"],
                "score_cosine": source["score_cosine"],
                "macro_cleaned_density": diag["macro"].stats["cleaned_density"],
                "macro_contour_density": diag["macro"].stats["contour_density"],
                "skeleton_density": diag["skeleton_density"],
            }

            for prefix, stats in [
                ("luma_lsd", diag["stats_luma"]),
                ("skeleton_lsd", diag["stats_skeleton"]),
            ]:
                for k, v in stats.items():
                    if k == "orientation_hist":
                        continue
                    row[f"{prefix}_{k}"] = v
                row[f"{prefix}_orientation_hist"] = json.dumps(stats["orientation_hist"])

            manifest_rows.append(row)

        if panel_rows:
            fig_path = OUT_FIG_DIR / f"s4c4a_token{token:04d}_vector_skeleton_lsd_preflight.png"
            render_token_panel(token, panel_rows, fig_path)
            panel_paths.append(str(fig_path))
            print(f"[OK] token {token}: saved {fig_path}")

    manifest_df = pd.DataFrame(manifest_rows)

    manifest_csv = OUT_META_DIR / "s4c4a_vector_skeleton_preflight_manifest.csv"
    manifest_df.to_csv(manifest_csv, index=False)

    summary = {
        "stage": "S4C.4A_vector_skeleton_LSD_preflight",
        "sequence": args.sequence,
        "tokens_requested": tokens,
        "num_manifest_rows": len(manifest_rows),
        "manifest_csv": str(manifest_csv),
        "panel_paths": panel_paths,
        "settings": vars(args),
        "notes": [
            "Diagnostic only; no retrieval/reranking performed.",
            "GT/nearest and oracle-best top-N rows use reference information only for post-hoc diagnosis.",
            "Purpose is to check whether LSD/vector skeletons suppress vegetation clutter and preserve structural lines.",
        ],
    }

    if len(manifest_df) > 0:
        group_cols = [
            "role",
            "luma_lsd_line_count",
            "luma_lsd_total_line_length_px",
            "skeleton_lsd_line_count",
            "skeleton_lsd_total_line_length_px",
            "skeleton_density",
        ]
        existing = [c for c in group_cols if c in manifest_df.columns]
        if existing:
            summary["role_mean_metrics"] = (
                manifest_df[existing]
                .groupby("role")
                .mean(numeric_only=True)
                .reset_index()
                .to_dict(orient="records")
            )

    summary_json = OUT_REPORT_DIR / "s4c4a_vector_skeleton_preflight_summary.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(json_safe(summary), f, indent=2)

    print("")
    print("S4C.4A complete")
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
            "skeleton_density",
            "luma_lsd_line_count",
            "luma_lsd_total_line_length_px",
            "skeleton_lsd_line_count",
            "skeleton_lsd_total_line_length_px",
        ]
        cols = [c for c in cols if c in manifest_df.columns]
        print(manifest_df[cols].groupby("role").mean(numeric_only=True).to_string())


if __name__ == "__main__":
    main()
