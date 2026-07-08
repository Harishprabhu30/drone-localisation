#!/usr/bin/env python3
"""
S4C.6B — Efficient rotation-aware luma-LSD alignment.

Purpose:
  Add rotation tolerance to S4C.4C luma-LSD reranking.

Pipeline:
  S4C.1 PHOG top-N candidates
  ↓
  extract luma-LSD canvas for UAV and candidate tiles
  ↓
  rotate only the UAV canvas by small candidate angles
  ↓
  run dx/dy distance-transform alignment
  ↓
  keep best angle + shift + score
  ↓
  rerank PHOG top-N candidates

Important:
  - No reference coordinate is used in scoring.
  - center_error_m is used only after ranking for evaluation.
  - Rotation is applied only inside PHOG top-N, not across the full map.

Code Used:
export PYTHONPATH=$PWD/src

python scripts/satloc/s4c/s4c_6b_rotation_aware_luma_lsd.py \
  --sequence traj01 \
  --tokens 1,288,300,326,336,350,366,387,421,564,573,577,591,614,631,662 \
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
  --phog-top-n 50 \
  --canvas-mode luma_lsd \
  --line-thickness 2 \
  --max-shift-px 48 \
  --shift-step-px 8 \
  --symmetric-weight 0.25 \
  --max-points 8000 \
  --rotation-mode fixed \
  --rotation-angles=-30,-20,-10,0,10,20,30 \
  --save-panels \
  --display-top-k 5

2. Lighter orientation prior mode:
python scripts/satloc/s4c/s4c_6b_rotation_aware_luma_lsd.py \
  --sequence traj01 \
  --tokens 1,288,300,326,336,350,366,387,421,564,573,577,591,614,631,662 \
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
  --phog-top-n 50 \
  --canvas-mode luma_lsd \
  --line-thickness 2 \
  --max-shift-px 48 \
  --shift-step-px 8 \
  --symmetric-weight 0.25 \
  --max-points 8000 \
  --rotation-mode orientation_prior \
  --orientation-prior-top-k 2 \
  --orientation-prior-offsets=-10,0,10 \
  --orientation-prior-max-abs-angle 35 \
  --save-panels \
  --display-top-k 5
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
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

S4C4C_PATH = THIS_DIR / "s4c_4c_luma_lsd_phog_top50_reranker.py"
S4C4B_PATH = THIS_DIR / "s4c_4b_lsd_alignment_surface.py"

for p in [S4C4C_PATH, S4C4B_PATH]:
    if not p.exists():
        raise FileNotFoundError(f"Missing helper script: {p}")

spec4c = importlib.util.spec_from_file_location("s4c4c_helpers", S4C4C_PATH)
s4c4c = importlib.util.module_from_spec(spec4c)
sys.modules["s4c4c_helpers"] = s4c4c
assert spec4c.loader is not None
spec4c.loader.exec_module(s4c4c)

spec4b = importlib.util.spec_from_file_location("s4c4b_helpers", S4C4B_PATH)
s4c4b = importlib.util.module_from_spec(spec4b)
sys.modules["s4c4b_helpers"] = s4c4b
assert spec4b.loader is not None
spec4b.loader.exec_module(s4c4b)


OUT_ROOT = Path("outputs/satloc")
DEFAULT_UAV_INDEX = OUT_ROOT / "metadata/uav_frames_index_enriched.csv"
DEFAULT_SAT_INDEX = OUT_ROOT / "metadata/satellite_tiles_index_enriched.csv"
DEFAULT_S4C1_DIR = OUT_ROOT / "metadata/s4c_macrocontour_phog_chamfer/s4c1_phog_retrieval"

OUT_META_DIR = OUT_ROOT / "metadata/s4c_macrocontour_phog_chamfer/s4c6_rotation_aware_lsd"
OUT_REPORT_DIR = OUT_ROOT / "reports/s4c_macrocontour_phog_chamfer/s4c6_rotation_aware_lsd"
OUT_FIG_DIR = OUT_ROOT / "figures/s4c_macrocontour_phog_chamfer/s4c6_rotation_aware_lsd"


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


def parse_tokens(text: str, ranked_dir: Path) -> list[int]:
    text = str(text).strip()
    if text.lower() == "all":
        tokens = set()
        for p in ranked_dir.glob("s4c1_token*_ranked.csv"):
            m = re.search(r"s4c1_token(\d+)_", p.name)
            if m:
                tokens.add(int(m.group(1)))
        if not tokens:
            raise FileNotFoundError(f"No S4C.1 ranked CSVs found in {ranked_dir}")
        return sorted(tokens)

    return sorted([int(x.strip()) for x in text.split(",") if x.strip()])


def parse_angles(text: str) -> list[float]:
    out = []
    for x in str(text).split(","):
        x = x.strip()
        if x:
            out.append(float(x))
    if 0.0 not in out:
        out.append(0.0)
    return sorted(set(out))


def first_rank_under(df: pd.DataFrame, rank_col: str, threshold_m: float) -> Optional[int]:
    if "center_error_m" not in df.columns:
        return None

    sub = df[np.isfinite(df["center_error_m"]) & (df["center_error_m"] <= threshold_m)]
    if len(sub) == 0:
        return None

    return int(sub[rank_col].min())


def rotate_binary_canvas(canvas: np.ndarray, angle_deg: float) -> np.ndarray:
    if abs(angle_deg) < 1e-9:
        return canvas.copy()

    h, w = canvas.shape[:2]
    center = (w / 2.0, h / 2.0)
    mat = cv2.getRotationMatrix2D(center, angle_deg, 1.0)

    rot = cv2.warpAffine(
        canvas,
        mat,
        (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    return (rot > 0).astype(np.uint8) * 255


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
    overlay[..., 1] = c * 180
    overlay[..., 0] = shifted_q * 220

    overlap = (c > 0) & (shifted_q > 0)
    overlay[overlap, 0] = 255
    overlay[overlap, 1] = 255

    return overlay


def best_alignment_score_fast(
    query_canvas: np.ndarray,
    candidate_canvas: np.ndarray,
    max_shift_px: int,
    shift_step_px: int,
    symmetric_weight: float,
    max_points: int,
) -> dict[str, Any]:
    q = (query_canvas > 0).astype(np.uint8) * 255
    c = (candidate_canvas > 0).astype(np.uint8) * 255

    q_pts = s4c4b.contour_points(q, max_points=max_points)
    c_pts = s4c4b.contour_points(c, max_points=max_points)

    h, w = q.shape[:2]
    diag = float(math.sqrt(h * h + w * w))
    penalty = diag

    shifts = list(range(-max_shift_px, max_shift_px + 1, shift_step_px))
    if 0 not in shifts:
        shifts.append(0)
    shifts = sorted(set(shifts))

    if len(q_pts) == 0 or len(c_pts) == 0:
        return {
            "best_score": penalty,
            "best_dx": 0,
            "best_dy": 0,
            "center_score": penalty,
            "center_minus_best": 0.0,
            "query_point_count": int(len(q_pts)),
            "candidate_point_count": int(len(c_pts)),
            "basin_area": len(shifts) * len(shifts),
        }

    q_dt = s4c4b.distance_transform_for_edges(q)
    c_dt = s4c4b.distance_transform_for_edges(c)

    best_score = float("inf")
    best_dx = 0
    best_dy = 0
    center_score = None
    all_scores = []

    for dy in shifts:
        for dx in shifts:
            q_to_c = s4c4b.sample_dt_at_shift(
                dt=c_dt,
                pts_xy=q_pts,
                dx=dx,
                dy=dy,
                out_of_bounds_penalty=penalty,
            )

            c_to_q = s4c4b.sample_dt_at_shift(
                dt=q_dt,
                pts_xy=c_pts,
                dx=-dx,
                dy=-dy,
                out_of_bounds_penalty=penalty,
            )

            score = float(q_to_c + symmetric_weight * c_to_q)
            all_scores.append(score)

            if dx == 0 and dy == 0:
                center_score = score

            if score < best_score:
                best_score = score
                best_dx = int(dx)
                best_dy = int(dy)

    scores = np.array(all_scores, dtype=np.float64)
    basin_tol = max(2.0, 0.10 * max(best_score, 1e-6))
    basin_area = int((scores <= best_score + basin_tol).sum())

    return {
        "best_score": float(best_score),
        "best_dx": int(best_dx),
        "best_dy": int(best_dy),
        "center_score": float(center_score if center_score is not None else best_score),
        "center_minus_best": float((center_score if center_score is not None else best_score) - best_score),
        "query_point_count": int(len(q_pts)),
        "candidate_point_count": int(len(c_pts)),
        "basin_area": basin_area,
    }


def orientation_hist_from_diag(diag: dict[str, Any], bins: int) -> np.ndarray:
    hist = diag["stats_luma"].get("orientation_hist", None)
    if hist is None:
        return np.zeros((bins,), dtype=np.float64)

    arr = np.array(hist, dtype=np.float64)

    if len(arr) != bins:
        x_old = np.linspace(0, 1, len(arr), endpoint=False)
        x_new = np.linspace(0, 1, bins, endpoint=False)
        arr = np.interp(x_new, x_old, arr)

    s = arr.sum()
    if s > 1e-9:
        arr = arr / s

    return arr


def orientation_prior_angles(
    query_diag: dict[str, Any],
    cand_diag: dict[str, Any],
    bins: int,
    top_k: int,
    local_offsets: list[float],
    max_abs_angle: float,
) -> list[float]:
    q = orientation_hist_from_diag(query_diag, bins)
    c = orientation_hist_from_diag(cand_diag, bins)

    if q.sum() <= 1e-9 or c.sum() <= 1e-9:
        return [0.0]

    scores = []

    for shift in range(bins):
        q_shifted = np.roll(q, shift)
        score = float(np.dot(q_shifted, c))
        angle = shift * (180.0 / bins)

        if angle > 90.0:
            angle -= 180.0

        scores.append((score, angle))

    scores = sorted(scores, key=lambda x: x[0], reverse=True)
    base_angles = [a for _, a in scores[:top_k]]

    out = set([0.0])

    for base in base_angles:
        for off in local_offsets:
            a = float(base + off)
            if abs(a) <= max_abs_angle:
                out.add(round(a, 6))

    return sorted(out)


def angles_for_candidate(
    mode: str,
    fixed_angles: list[float],
    query_diag: dict[str, Any],
    cand_diag: dict[str, Any],
    args: argparse.Namespace,
) -> list[float]:
    if mode == "fixed":
        return fixed_angles

    if mode == "orientation_prior":
        offsets = parse_angles(args.orientation_prior_offsets)
        return orientation_prior_angles(
            query_diag=query_diag,
            cand_diag=cand_diag,
            bins=args.orientation_bins,
            top_k=args.orientation_prior_top_k,
            local_offsets=offsets,
            max_abs_angle=args.orientation_prior_max_abs_angle,
        )

    raise ValueError(f"Unknown rotation mode: {mode}")


def compute_rotation_aware_alignment(
    query_diag: dict[str, Any],
    query_canvas: np.ndarray,
    cand_diag: dict[str, Any],
    cand_canvas: np.ndarray,
    fixed_angles: list[float],
    args: argparse.Namespace,
) -> dict[str, Any]:
    angles = angles_for_candidate(
        mode=args.rotation_mode,
        fixed_angles=fixed_angles,
        query_diag=query_diag,
        cand_diag=cand_diag,
        args=args,
    )

    best = None

    for angle in angles:
        q_rot = rotate_binary_canvas(query_canvas, angle)

        result = best_alignment_score_fast(
            query_canvas=q_rot,
            candidate_canvas=cand_canvas,
            max_shift_px=args.max_shift_px,
            shift_step_px=args.shift_step_px,
            symmetric_weight=args.symmetric_weight,
            max_points=args.max_points,
        )

        result["angle_deg"] = float(angle)

        if best is None or result["best_score"] < best["best_score"]:
            best = result
            best["rotated_query_canvas"] = q_rot

    assert best is not None
    best["angles_tested"] = ",".join(str(float(a)) for a in angles)
    best["num_angles_tested"] = int(len(angles))

    return best


def summarize_profile(token: int, profile: str, ranked: pd.DataFrame) -> dict[str, Any]:
    top1 = ranked.iloc[0]
    top10 = ranked.head(10)

    err = safe_float(top1.get("center_error_m"))

    return {
        "token": int(token),
        "profile": profile,

        "top1_tile_id": int(top1["tile_id"]),
        "top1_error_m": err,
        "top1_phog_rank": int(top1["phog_rank"]),
        "top1_rot_lsd_rank": int(top1["rot_lsd_rank"]),
        "top1_rot_lsd_score": safe_float(top1["rot_lsd_best_score"]),
        "top1_angle_deg": safe_float(top1["rot_best_angle_deg"]),
        "top1_dx": int(top1["rot_best_dx"]),
        "top1_dy": int(top1["rot_best_dy"]),
        "top1_shift_mag_px": safe_float(top1["rot_shift_mag_px"]),
        "top1_boundary": bool(top1["rot_at_shift_boundary"]),

        "best_top10_error_m": safe_float(top10["center_error_m"].min()),
        "first_rank_under_20m": first_rank_under(ranked, "rerank_rank", 20.0),
        "first_rank_under_40m": first_rank_under(ranked, "rerank_rank", 40.0),
        "first_rank_under_60m": first_rank_under(ranked, "rerank_rank", 60.0),

        "top1_under_20m": bool(np.isfinite(err) and err <= 20.0),
        "top1_under_40m": bool(np.isfinite(err) and err <= 40.0),
        "top1_under_60m": bool(np.isfinite(err) and err <= 60.0),
        "top1_under_100m": bool(np.isfinite(err) and err <= 100.0),

        "top10_under_20m": bool((top10["center_error_m"] <= 20.0).any()),
        "top10_under_40m": bool((top10["center_error_m"] <= 40.0).any()),
        "top10_under_60m": bool((top10["center_error_m"] <= 60.0).any()),
    }


def apply_profile(df: pd.DataFrame, profile: str, phog_top_n: int) -> pd.DataFrame:
    out = df.copy()

    if profile == "phog_only":
        out = out.sort_values("phog_rank").reset_index(drop=True)

    elif profile == "rot_lsd_only":
        out = out.sort_values(["rot_lsd_rank", "phog_rank"]).reset_index(drop=True)

    elif profile == "rot_lsd_strong":
        boundary_penalty = out["rot_at_shift_boundary"].astype(bool).astype(float) * float(phog_top_n)

        out["rerank_score"] = (
            0.50 * out["phog_rank"].astype(float)
            + 1.00 * out["rot_lsd_rank"].astype(float)
            + 0.15 * out["rot_shift_rank"].astype(float)
            + 0.10 * out["rot_basin_rank"].astype(float)
            + 0.75 * boundary_penalty
        )

        out = out.sort_values(["rerank_score", "phog_rank", "rot_lsd_rank"]).reset_index(drop=True)

    elif profile == "rot_phog_protected":
        # Protect very top PHOG when it is also not terrible under rotated LSD.
        phog_top = out.sort_values("phog_rank").iloc[0]
        if int(phog_top["rot_lsd_rank"]) <= 5:
            out = out.sort_values("phog_rank").reset_index(drop=True)
        else:
            out = out.sort_values(["rot_lsd_rank", "phog_rank"]).reset_index(drop=True)

    else:
        raise ValueError(f"Unknown profile: {profile}")

    out["profile"] = profile
    out["rerank_rank"] = np.arange(1, len(out) + 1)

    return out


def read_rgb(path: str, size: int = 256) -> Optional[np.ndarray]:
    p = Path(str(path))
    if not p.exists():
        return None

    bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if bgr is None:
        return None

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    side = min(h, w)

    y0 = (h - side) // 2
    x0 = (w - side) // 2

    rgb = rgb[y0:y0 + side, x0:x0 + side]
    rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
    return rgb


def render_token_panel(token: int, profile_rankings: dict[str, pd.DataFrame], out_path: Path, display_top_k: int) -> None:
    profiles = [p for p in ["phog_only", "rot_lsd_only", "rot_lsd_strong", "rot_phog_protected"] if p in profile_rankings]

    if not profiles:
        return

    fig, axes = plt.subplots(
        len(profiles),
        display_top_k,
        figsize=(3.1 * display_top_k, 3.45 * len(profiles)),
        squeeze=False,
    )

    for r, profile in enumerate(profiles):
        df = profile_rankings[profile].head(display_top_k)

        for c in range(display_top_k):
            ax = axes[r, c]
            ax.axis("off")

            if c >= len(df):
                continue

            row = df.iloc[c]
            img = read_rgb(str(row["candidate_image_path"]))

            if img is not None:
                ax.imshow(img)

            ax.set_title(
                f"{profile} R{int(row['rerank_rank'])}\n"
                f"tile {int(row['tile_id'])} err={safe_float(row.get('center_error_m')):.1f}m\n"
                f"P{int(row['phog_rank'])} L{int(row['rot_lsd_rank'])} "
                f"S={safe_float(row['rot_lsd_best_score']):.1f}\n"
                f"a={safe_float(row['rot_best_angle_deg']):.0f} "
                f"dx={int(row['rot_best_dx'])},dy={int(row['rot_best_dy'])}",
                fontsize=8,
            )

            if c == 0:
                ax.set_ylabel(profile, fontsize=10)

    fig.suptitle(f"S4C.6B rotation-aware luma-LSD — token {token}", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def render_aggregate_plot(agg: pd.DataFrame, out_path: Path) -> None:
    if len(agg) == 0:
        return

    labels = agg["profile"].tolist()
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    ax.bar(x - width / 2, agg["top1_under_40m_rate"], width, label="Top1 <=40m")
    ax.bar(x + width / 2, agg["top10_under_40m_rate"], width, label="Top10 <=40m")

    ax.set_ylim(0, 1.0)
    ax.set_ylabel("rate")
    ax.set_title("S4C.6B rotation-aware luma-LSD")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def run_token(
    token: int,
    uav_df: pd.DataFrame,
    sat_df: pd.DataFrame,
    ranked_dir: Path,
    filename_index: dict[str, Path],
    fallback_uav_dirs: list[Path],
    fallback_sat_dirs: list[Path],
    fixed_angles: list[float],
    args: argparse.Namespace,
    canvas_cache: dict[str, tuple[dict[str, Any], np.ndarray]],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    uav_path = s4c4c.find_uav_query_path(
        token=token,
        uav_df=uav_df,
        filename_index=filename_index,
        fallback_uav_dirs=fallback_uav_dirs,
    )

    if uav_path is None:
        raise FileNotFoundError(f"Could not find UAV image for token {token}")

    candidates = s4c4c.load_phog_candidates(
        token=token,
        ranked_dir=ranked_dir,
        sat_df=sat_df,
        filename_index=filename_index,
        fallback_sat_dirs=fallback_sat_dirs,
        top_n=args.phog_top_n,
    )

    if len(candidates) == 0:
        raise RuntimeError(f"No candidates for token {token}")

    query_diag, query_canvas = s4c4c.build_canvas_cached(Path(uav_path), args, canvas_cache)

    rows = []

    print(f"[token {token}] rotation-aware reranking PHOG top {len(candidates)}")

    for idx, row in candidates.iterrows():
        cand_path = Path(str(row["candidate_image_path"]))
        cand_diag, cand_canvas = s4c4c.build_canvas_cached(cand_path, args, canvas_cache)

        best = compute_rotation_aware_alignment(
            query_diag=query_diag,
            query_canvas=query_canvas,
            cand_diag=cand_diag,
            cand_canvas=cand_canvas,
            fixed_angles=fixed_angles,
            args=args,
        )

        out = row.to_dict()
        out.update({
            "rot_lsd_best_score": best["best_score"],
            "rot_best_angle_deg": best["angle_deg"],
            "rot_best_dx": best["best_dx"],
            "rot_best_dy": best["best_dy"],
            "rot_center_score": best["center_score"],
            "rot_center_minus_best": best["center_minus_best"],
            "rot_basin_area": best["basin_area"],
            "rot_query_point_count": best["query_point_count"],
            "rot_candidate_point_count": best["candidate_point_count"],
            "rot_angles_tested": best["angles_tested"],
            "rot_num_angles_tested": best["num_angles_tested"],
        })

        out["rot_shift_mag_px"] = float(math.hypot(out["rot_best_dx"], out["rot_best_dy"]))
        out["rot_at_shift_boundary"] = bool(
            abs(int(out["rot_best_dx"])) >= int(args.max_shift_px)
            or abs(int(out["rot_best_dy"])) >= int(args.max_shift_px)
        )

        rows.append(out)

        if (idx + 1) % 10 == 0:
            print(f"[token {token}] processed {idx + 1}/{len(candidates)} candidates")

    df = pd.DataFrame(rows)

    df["rot_lsd_rank"] = df["rot_lsd_best_score"].rank(method="min", ascending=True).astype(int)
    df["rot_shift_rank"] = df["rot_shift_mag_px"].rank(method="min", ascending=True).astype(int)
    df["rot_basin_rank"] = df["rot_basin_area"].rank(method="min", ascending=True).astype(int)

    raw_csv = OUT_META_DIR / f"s4c6b_token{token:04d}_{args.rotation_mode}_raw_scores_top{args.phog_top_n}.csv"
    df.to_csv(raw_csv, index=False)

    profiles = ["phog_only", "rot_lsd_only", "rot_lsd_strong", "rot_phog_protected"]

    summary_rows = []
    profile_rankings = {}

    for profile in profiles:
        ranked = apply_profile(df, profile, phog_top_n=args.phog_top_n)
        profile_rankings[profile] = ranked
        summary_rows.append(summarize_profile(token, profile, ranked))

    profile_df = pd.concat(profile_rankings.values(), ignore_index=True)
    profile_csv = OUT_META_DIR / f"s4c6b_token{token:04d}_{args.rotation_mode}_profile_reranks_top{args.phog_top_n}.csv"
    profile_df.to_csv(profile_csv, index=False)

    if args.save_panels:
        panel_path = OUT_FIG_DIR / f"s4c6b_token{token:04d}_{args.rotation_mode}_panel_top{args.display_top_k}.png"
        render_token_panel(token, profile_rankings, panel_path, display_top_k=args.display_top_k)

    for r in summary_rows:
        r["raw_csv"] = str(raw_csv)
        r["profile_csv"] = str(profile_csv)

    compact = " | ".join(
        f"{r['profile']}: {r['top1_error_m']:.1f}m,a={r['top1_angle_deg']:.0f}"
        for r in summary_rows
    )
    print(f"[OK] token {token}: {compact}")

    return df, summary_rows


def aggregate_summary(summary_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for profile, group in summary_df.groupby("profile"):
        rows.append({
            "profile": profile,
            "num_queries": int(len(group)),
            "top1_under_20m_rate": float(group["top1_under_20m"].mean()),
            "top1_under_40m_rate": float(group["top1_under_40m"].mean()),
            "top1_under_60m_rate": float(group["top1_under_60m"].mean()),
            "top1_under_100m_rate": float(group["top1_under_100m"].mean()),
            "top10_under_40m_rate": float(group["top10_under_40m"].mean()),
            "median_top1_error_m": float(group["top1_error_m"].median()),
            "mean_top1_error_m": float(group["top1_error_m"].mean()),
            "median_best_top10_error_m": float(group["best_top10_error_m"].median()),
            "median_top1_rot_score": float(group["top1_rot_lsd_score"].median()),
            "median_top1_angle_deg": float(group["top1_angle_deg"].median()),
            "mean_abs_top1_angle_deg": float(group["top1_angle_deg"].abs().mean()),
        })

    out = pd.DataFrame(rows)

    if len(out) > 0:
        out = out.sort_values(
            ["top1_under_40m_rate", "top10_under_40m_rate", "median_top1_error_m"],
            ascending=[False, False, True],
        )

    return out


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--sequence", default="traj01")
    parser.add_argument("--tokens", default="1,60,90,129,166,269")
    parser.add_argument("--uav-index", default=str(DEFAULT_UAV_INDEX))
    parser.add_argument("--sat-index", default=str(DEFAULT_SAT_INDEX))
    parser.add_argument("--s4c1-ranked-dir", default=str(DEFAULT_S4C1_DIR))

    # Current luma-LSD settings.
    parser.add_argument("--preprocess", default="luma", choices=["gray", "luma", "clahe_luma"])
    parser.add_argument("--resize-size", type=int, default=512)
    parser.add_argument("--edge-method", default="sobel", choices=["sobel", "canny"])
    parser.add_argument("--blur-ksize", type=int, default=3)
    parser.add_argument("--threshold-mode", default="percentile", choices=["otsu", "percentile"])
    parser.add_argument("--threshold-percentile", type=float, default=65.0)
    parser.add_argument("--close-ksize", type=int, default=3)
    parser.add_argument("--open-ksize", type=int, default=1)
    parser.add_argument("--min-component-area", type=int, default=65)
    parser.add_argument("--min-line-length", type=float, default=24.0)
    parser.add_argument("--max-lines", type=int, default=120)
    parser.add_argument("--orientation-bins", type=int, default=18)

    # Candidate/alignment settings.
    parser.add_argument("--phog-top-n", type=int, default=50)
    parser.add_argument("--canvas-mode", default="luma_lsd", choices=["luma_lsd"])
    parser.add_argument("--line-thickness", type=int, default=2)
    parser.add_argument("--max-shift-px", type=int, default=48)
    parser.add_argument("--shift-step-px", type=int, default=8)
    parser.add_argument("--symmetric-weight", type=float, default=0.25)
    parser.add_argument("--max-points", type=int, default=8000)

    # Rotation settings.
    parser.add_argument("--rotation-mode", default="fixed", choices=["fixed", "orientation_prior"])
    parser.add_argument("--rotation-angles", default="-30,-20,-10,0,10,20,30")
    parser.add_argument("--orientation-prior-top-k", type=int, default=2)
    parser.add_argument("--orientation-prior-offsets", default="-10,0,10")
    parser.add_argument("--orientation-prior-max-abs-angle", type=float, default=35.0)

    # Output.
    parser.add_argument("--save-panels", action="store_true")
    parser.add_argument("--display-top-k", type=int, default=5)

    args = parser.parse_args()

    ranked_dir = Path(args.s4c1_ranked_dir)
    tokens = parse_tokens(args.tokens, ranked_dir)
    fixed_angles = parse_angles(args.rotation_angles)

    OUT_META_DIR.mkdir(parents=True, exist_ok=True)
    OUT_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_FIG_DIR.mkdir(parents=True, exist_ok=True)

    uav_df = pd.read_csv(args.uav_index)
    sat_df = pd.read_csv(args.sat_index)

    uav_df = s4c4c.s4c4a.prepare_uav_df(uav_df, args.sequence)
    filename_index, fallback_uav_dirs, fallback_sat_dirs = s4c4c.s4c4a.build_filename_index(args.sequence)

    print("S4C.6B efficient rotation-aware luma-LSD alignment")
    print("--------------------------------------------------")
    print(f"Sequence:       {args.sequence}")
    print(f"Tokens:         {tokens}")
    print(f"PHOG top-N:     {args.phog_top_n}")
    print(f"Rotation mode:  {args.rotation_mode}")
    print(f"Fixed angles:   {fixed_angles}")
    print(f"Shift search:   ±{args.max_shift_px}px step {args.shift_step_px}px")
    print("")

    start = time.time()
    canvas_cache: dict[str, tuple[dict[str, Any], np.ndarray]] = {}
    all_summary_rows = []

    for token in tokens:
        try:
            _, rows = run_token(
                token=token,
                uav_df=uav_df,
                sat_df=sat_df,
                ranked_dir=ranked_dir,
                filename_index=filename_index,
                fallback_uav_dirs=fallback_uav_dirs,
                fallback_sat_dirs=fallback_sat_dirs,
                fixed_angles=fixed_angles,
                args=args,
                canvas_cache=canvas_cache,
            )
            all_summary_rows.extend(rows)
        except Exception as exc:
            print(f"[WARN] token {token}: {exc}")

    summary_df = pd.DataFrame(all_summary_rows)

    if len(summary_df) == 0:
        raise RuntimeError("No successful token summaries.")

    aggregate_df = aggregate_summary(summary_df)

    slug = f"{args.sequence}_{args.rotation_mode}_top{args.phog_top_n}"
    summary_csv = OUT_META_DIR / f"s4c6b_{slug}_summary_by_token_profile.csv"
    aggregate_csv = OUT_META_DIR / f"s4c6b_{slug}_aggregate_by_profile.csv"
    summary_json = OUT_REPORT_DIR / f"s4c6b_{slug}_summary.json"
    plot_path = OUT_FIG_DIR / f"s4c6b_{slug}_profile_success_rates.png"

    summary_df.to_csv(summary_csv, index=False)
    aggregate_df.to_csv(aggregate_csv, index=False)
    render_aggregate_plot(aggregate_df, plot_path)

    output = {
        "stage": "S4C.6B_rotation_aware_luma_LSD_alignment",
        "sequence": args.sequence,
        "tokens": tokens,
        "settings": vars(args),
        "fixed_angles": fixed_angles,
        "elapsed_s": time.time() - start,
        "summary_csv": str(summary_csv),
        "aggregate_csv": str(aggregate_csv),
        "plot_path": str(plot_path),
        "notes": [
            "Only UAV luma-LSD canvas is rotated.",
            "Rotation search is only applied inside PHOG top-N candidates.",
            "No reference coordinate is used in scoring.",
            "center_error_m is used only for post-ranking evaluation.",
        ],
        "aggregate_by_profile": aggregate_df.to_dict(orient="records"),
    }

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(json_safe(output), f, indent=2)

    print("")
    print("S4C.6B complete")
    print("----------------")
    print(f"Summary CSV:   {summary_csv}")
    print(f"Aggregate CSV: {aggregate_csv}")
    print(f"Summary JSON:  {summary_json}")
    print(f"Plot:          {plot_path}")
    print("")
    print("Aggregate profile comparison")
    print("----------------------------")
    cols = [
        "profile",
        "num_queries",
        "top1_under_20m_rate",
        "top1_under_40m_rate",
        "top10_under_40m_rate",
        "median_top1_error_m",
        "median_best_top10_error_m",
        "median_top1_rot_score",
        "mean_abs_top1_angle_deg",
    ]
    print(aggregate_df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
