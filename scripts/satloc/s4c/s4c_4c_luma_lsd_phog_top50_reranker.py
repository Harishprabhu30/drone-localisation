#!/usr/bin/env python3
"""
S4C.4C — luma-LSD PHOG top-50 reranker.

Purpose:
  Use S4C.1 PHOG as coarse retrieval, then rerank top-N candidates using
  luma-LSD distance-transform alignment.

Important:
  - UAV filename lon/lat and center_error_m are NOT used in scoring.
  - center_error_m is used only after ranking for evaluation.
  - This is the first real reranking test from S4C.4B's diagnostic signal.

Profiles:
  phog_only      : baseline
  lsd_only       : pure luma-LSD alignment
  lsd_weak       : PHOG protected + weak LSD
  lsd_balanced   : PHOG + stronger LSD
  lsd_strong     : LSD-dominant but still PHOG-aware
  lsd_guard      : stronger penalty for boundary-shift / broad-basin alignments
 
code Used:
selected_subset from traj01 = 1,40,50,58,60,67,74,79,90,100,107,117,129,139,166,259,269,276,288,300,310,326,336,350,366,387,405,421,434,450,474,482,494,503,516,533,546,564,573,577,591,614,631,653,662,679,694,710,731,746,760,768,781,794,808,820,833,844,874,886,905,914,927,937,946,952,963,971,990,1003,1015,1024,1034
export PYTHONPATH=$PWD/src

python scripts/satloc/s4c/s4c_4c_luma_lsd_phog_top50_reranker.py \
  --sequence traj01 \
  --tokens 1,40,60,90,100,129,166,269,516,905 \ # use ALL to run all traj from earlier csv selected subset from traj01
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
  --save-panels \
  --display-top-k 5

2. to run all:
python scripts/satloc/s4c/s4c_4c_luma_lsd_phog_top50_reranker.py \
  --sequence traj01 \
  --tokens all \
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
  --max-points 8000
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
import sys
from dataclasses import dataclass
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
S4C4A_PATH = THIS_DIR / "s4c_4a_vector_skeleton_preflight.py"
S4C4B_PATH = THIS_DIR / "s4c_4b_lsd_alignment_surface.py"

for p in [S4C0_PATH, S4C4A_PATH, S4C4B_PATH]:
    if not p.exists():
        raise FileNotFoundError(f"Missing helper: {p}")

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

spec4b = importlib.util.spec_from_file_location("s4c4b_helpers", S4C4B_PATH)
s4c4b = importlib.util.module_from_spec(spec4b)
sys.modules["s4c4b_helpers"] = s4c4b
assert spec4b.loader is not None
spec4b.loader.exec_module(s4c4b)


OUT_ROOT = Path("outputs/satloc")
DEFAULT_UAV_INDEX = OUT_ROOT / "metadata/uav_frames_index_enriched.csv"
DEFAULT_SAT_INDEX = OUT_ROOT / "metadata/satellite_tiles_index_enriched.csv"
DEFAULT_S4C1_DIR = OUT_ROOT / "metadata/s4c_macrocontour_phog_chamfer/s4c1_phog_retrieval"

OUT_FIG_DIR = OUT_ROOT / "figures/s4c_macrocontour_phog_chamfer/s4c4_vector_skeleton/s4c4c_luma_lsd_rerank"
OUT_META_DIR = OUT_ROOT / "metadata/s4c_macrocontour_phog_chamfer/s4c4_vector_skeleton/s4c4c_luma_lsd_rerank"
OUT_REPORT_DIR = OUT_ROOT / "reports/s4c_macrocontour_phog_chamfer/s4c4_vector_skeleton/s4c4c_luma_lsd_rerank"


@dataclass
class Profile:
    name: str
    w_phog: float
    w_lsd: float
    w_shift: float
    w_basin: float
    w_boundary: float


def default_profiles() -> list[Profile]:
    return [
        Profile("phog_only", 1.0, 0.0, 0.0, 0.0, 0.0),
        Profile("lsd_only", 0.0, 1.0, 0.0, 0.0, 0.0),
        Profile("lsd_weak", 1.0, 0.25, 0.05, 0.05, 0.25),
        Profile("lsd_balanced", 1.0, 0.75, 0.10, 0.10, 0.50),
        Profile("lsd_strong", 0.50, 1.00, 0.15, 0.10, 0.75),
        Profile("lsd_guard", 1.0, 1.0, 0.20, 0.20, 1.00),
    ]


def parse_tokens(text: str, ranked_dir: Path) -> list[int]:
    text = text.strip()

    if text.lower() == "all":
        tokens: set[int] = set()
        for p in ranked_dir.glob("s4c1_token*_ranked.csv"):
            m = re.search(r"s4c1_token(\d+)_", p.name)
            if m:
                tokens.add(int(m.group(1)))
        if not tokens:
            raise FileNotFoundError(f"No S4C.1 ranked CSVs found in {ranked_dir}")
        return sorted(tokens)

    out = []
    for p in text.split(","):
        p = p.strip()
        if p:
            out.append(int(p))
    return sorted(out)


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


def find_latest_ranked_csv(token: int, ranked_dir: Path) -> Path:
    pattern = f"s4c1_token{token:04d}_*_ranked.csv"
    matches = sorted(ranked_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    if not matches:
        raise FileNotFoundError(f"No S4C.1 ranked CSV for token {token}: {ranked_dir / pattern}")
    return matches[0]


def get_score_col(df: pd.DataFrame) -> Optional[str]:
    for c in ["score_cosine", "combined_score", "score", "similarity"]:
        if c in df.columns:
            return c
    return None


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


def first_rank_under(df: pd.DataFrame, rank_col: str, threshold_m: float) -> Optional[int]:
    if "center_error_m" not in df.columns:
        return None

    sub = df[np.isfinite(df["center_error_m"]) & (df["center_error_m"] <= threshold_m)]
    if len(sub) == 0:
        return None

    return int(sub[rank_col].min())


def find_uav_query_path(
    token: int,
    uav_df: pd.DataFrame,
    filename_index: dict[str, Path],
    fallback_uav_dirs: list[Path],
) -> Optional[Path]:
    row = s4c4a.get_uav_row_for_token(uav_df, token)
    if row is None:
        return None

    return s4c0.get_row_path(
        row,
        uav_df,
        filename_index,
        fallback_uav_dirs,
        kind="uav",
    )


def load_phog_candidates(
    token: int,
    ranked_dir: Path,
    sat_df: pd.DataFrame,
    filename_index: dict[str, Path],
    fallback_sat_dirs: list[Path],
    top_n: int,
) -> pd.DataFrame:
    ranked_csv = find_latest_ranked_csv(token, ranked_dir)
    df = pd.read_csv(ranked_csv)

    if len(df) == 0:
        raise RuntimeError(f"Empty ranked CSV: {ranked_csv}")

    if "rank" not in df.columns:
        df["rank"] = np.arange(1, len(df) + 1)

    score_col = get_score_col(df)

    df = df.sort_values("rank").head(top_n).copy().reset_index(drop=True)
    df["token"] = token
    df["source_ranked_csv"] = str(ranked_csv)
    df["phog_rank"] = df["rank"].astype(int)

    if score_col is not None:
        df["phog_score"] = df[score_col].astype(float)
    else:
        df["phog_score"] = np.nan

    image_paths = []

    for _, row in df.iterrows():
        if "row_pos" in row and np.isfinite(row["row_pos"]):
            sat_row = sat_df.iloc[int(row["row_pos"])]
        else:
            tile_id = int(row["tile_id"])
            sat_row = s4c4a.find_tile_by_id(sat_df, tile_id)
            if sat_row is None:
                image_paths.append(None)
                continue

        path = s4c0.get_row_path(
            sat_row,
            sat_df,
            filename_index,
            fallback_sat_dirs,
            kind="sat",
        )
        image_paths.append(str(path) if path is not None else None)

    df["candidate_image_path"] = image_paths
    df = df[df["candidate_image_path"].notna()].copy().reset_index(drop=True)

    for col in ["center_error_m", "uav_lon", "uav_lat"]:
        if col not in df.columns:
            df[col] = np.nan

    return df


def build_canvas_cached(
    path: Path,
    args: argparse.Namespace,
    cache: dict[str, tuple[dict[str, Any], np.ndarray]],
) -> tuple[dict[str, Any], np.ndarray]:
    key = f"{path.resolve()}|{args.canvas_mode}|{args.line_thickness}"

    if key in cache:
        return cache[key]

    diag = s4c4a.compute_vector_diagnostic(path, args)
    canvas = s4c4b.get_alignment_canvas(diag, args.canvas_mode, args.line_thickness)

    cache[key] = (diag, canvas)
    return diag, canvas


def compute_lsd_alignment_for_token(
    token: int,
    uav_path: Path,
    candidates: pd.DataFrame,
    args: argparse.Namespace,
    cache: dict[str, tuple[dict[str, Any], np.ndarray]],
) -> pd.DataFrame:
    _, q_canvas = build_canvas_cached(uav_path, args, cache)

    rows = []

    for idx, row in candidates.iterrows():
        cand_path = Path(str(row["candidate_image_path"]))

        try:
            _, c_canvas = build_canvas_cached(cand_path, args, cache)

            surf = s4c4b.alignment_surface(
                query_canvas=q_canvas,
                candidate_canvas=c_canvas,
                max_shift_px=args.max_shift_px,
                shift_step_px=args.shift_step_px,
                symmetric_weight=args.symmetric_weight,
                max_points=args.max_points,
            )
        except Exception as exc:
            print(f"[WARN] token {token} candidate idx={idx}: {exc}")
            continue

        out = row.to_dict()
        out.update(
            {
                "lsd_best_score": surf["best_score"],
                "lsd_center_score": surf["center_score"],
                "lsd_center_minus_best": surf["center_minus_best"],
                "lsd_best_dx": surf["best_dx"],
                "lsd_best_dy": surf["best_dy"],
                "lsd_surface_mean": surf["surface_mean"],
                "lsd_surface_std": surf["surface_std"],
                "lsd_surface_contrast": surf["surface_contrast"],
                "lsd_basin_area": surf["basin_area"],
                "lsd_query_point_count": surf["query_point_count"],
                "lsd_candidate_point_count": surf["candidate_point_count"],
                "lsd_query_canvas_density": float((q_canvas > 0).mean()),
                "lsd_candidate_canvas_density": float((c_canvas > 0).mean()),
            }
        )

        out["lsd_shift_mag_px"] = float(math.hypot(out["lsd_best_dx"], out["lsd_best_dy"]))
        out["lsd_at_shift_boundary"] = bool(
            abs(int(out["lsd_best_dx"])) >= int(args.max_shift_px)
            or abs(int(out["lsd_best_dy"])) >= int(args.max_shift_px)
        )

        rows.append(out)

        if (idx + 1) % 10 == 0:
            print(f"[token {token}] processed {idx + 1}/{len(candidates)} candidates")

    out_df = pd.DataFrame(rows)

    if len(out_df) == 0:
        return out_df

    out_df["lsd_rank"] = (
        out_df["lsd_best_score"].rank(method="min", ascending=True).astype(int)
    )
    out_df["lsd_shift_rank"] = (
        out_df["lsd_shift_mag_px"].rank(method="min", ascending=True).astype(int)
    )
    out_df["lsd_basin_rank"] = (
        out_df["lsd_basin_area"].rank(method="min", ascending=True).astype(int)
    )

    return out_df


def apply_profile(df: pd.DataFrame, profile: Profile, top_n: int) -> pd.DataFrame:
    out = df.copy()

    boundary_penalty = out["lsd_at_shift_boundary"].astype(bool).astype(float) * float(top_n)

    out["rerank_score"] = (
        profile.w_phog * out["phog_rank"].astype(float)
        + profile.w_lsd * out["lsd_rank"].astype(float)
        + profile.w_shift * out["lsd_shift_rank"].astype(float)
        + profile.w_basin * out["lsd_basin_rank"].astype(float)
        + profile.w_boundary * boundary_penalty
    )

    out["profile"] = profile.name

    out = out.sort_values(
        ["rerank_score", "phog_rank", "lsd_rank"],
        ascending=[True, True, True],
    ).reset_index(drop=True)

    out["rerank_rank"] = np.arange(1, len(out) + 1)

    return out


def summarize_profile(token: int, profile_df: pd.DataFrame, profile: Profile) -> dict[str, Any]:
    top1 = profile_df.iloc[0]
    top10 = profile_df.head(10)

    return {
        "token": int(token),
        "profile": profile.name,
        "w_phog": profile.w_phog,
        "w_lsd": profile.w_lsd,
        "w_shift": profile.w_shift,
        "w_basin": profile.w_basin,
        "w_boundary": profile.w_boundary,

        "top1_tile_id": int(top1["tile_id"]),
        "top1_error_m": safe_float(top1.get("center_error_m")),
        "top1_phog_rank": int(top1["phog_rank"]),
        "top1_lsd_rank": int(top1["lsd_rank"]),
        "top1_lsd_best_score": safe_float(top1["lsd_best_score"]),
        "top1_best_dx": int(top1["lsd_best_dx"]),
        "top1_best_dy": int(top1["lsd_best_dy"]),
        "top1_at_shift_boundary": bool(top1["lsd_at_shift_boundary"]),

        "best_top10_error_m": safe_float(top10["center_error_m"].min()),
        "first_rank_under_20m": first_rank_under(profile_df, "rerank_rank", 20.0),
        "first_rank_under_40m": first_rank_under(profile_df, "rerank_rank", 40.0),
        "first_rank_under_60m": first_rank_under(profile_df, "rerank_rank", 60.0),

        "top1_under_20m": bool(np.isfinite(safe_float(top1.get("center_error_m"))) and safe_float(top1.get("center_error_m")) <= 20.0),
        "top1_under_40m": bool(np.isfinite(safe_float(top1.get("center_error_m"))) and safe_float(top1.get("center_error_m")) <= 40.0),
        "top1_under_60m": bool(np.isfinite(safe_float(top1.get("center_error_m"))) and safe_float(top1.get("center_error_m")) <= 60.0),

        "top10_under_20m": bool((top10["center_error_m"] <= 20.0).any()),
        "top10_under_40m": bool((top10["center_error_m"] <= 40.0).any()),
        "top10_under_60m": bool((top10["center_error_m"] <= 60.0).any()),
    }


def render_token_panel(
    token: int,
    profile_rankings: dict[str, pd.DataFrame],
    profiles_to_show: list[str],
    display_top_k: int,
    out_path: Path,
) -> None:
    rows = [p for p in profiles_to_show if p in profile_rankings]
    if not rows:
        return

    fig, axes = plt.subplots(
        len(rows),
        display_top_k,
        figsize=(3.0 * display_top_k, 3.35 * len(rows)),
        squeeze=False,
    )

    for r, profile_name in enumerate(rows):
        df = profile_rankings[profile_name].head(display_top_k)

        for c in range(display_top_k):
            ax = axes[r, c]
            ax.axis("off")

            if c >= len(df):
                continue

            row = df.iloc[c]
            img = read_rgb(str(row["candidate_image_path"]))

            if img is not None:
                ax.imshow(img)

            err = safe_float(row.get("center_error_m"))
            ax.set_title(
                f"{profile_name} R{int(row['rerank_rank'])}\n"
                f"tile {int(row['tile_id'])}, err={err:.1f}m\n"
                f"P{int(row['phog_rank'])} L{int(row['lsd_rank'])} "
                f"S={safe_float(row['lsd_best_score']):.1f}\n"
                f"dx={int(row['lsd_best_dx'])},dy={int(row['lsd_best_dy'])}",
                fontsize=8,
            )

            if c == 0:
                ax.set_ylabel(profile_name, fontsize=10)

    fig.suptitle(f"S4C.4C luma-LSD PHOG top-N rerank — token {token}", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def render_aggregate_plot(aggregate_df: pd.DataFrame, out_path: Path) -> None:
    if len(aggregate_df) == 0:
        return

    plot_df = aggregate_df.copy()
    labels = plot_df["profile"].astype(str).tolist()
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(11.0, 5.2))

    ax.bar(x - width / 2, plot_df["top1_under_40m_rate"], width, label="Top-1 <= 40 m")
    ax.bar(x + width / 2, plot_df["top10_under_40m_rate"], width, label="Top-10 <= 40 m")

    ax.set_ylim(0, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("rate")
    ax.set_title("S4C.4C profile comparison — <=40 m success")
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
    profiles: list[Profile],
    args: argparse.Namespace,
    canvas_cache: dict[str, tuple[dict[str, Any], np.ndarray]],
) -> list[dict[str, Any]]:
    uav_path = find_uav_query_path(
        token,
        uav_df=uav_df,
        filename_index=filename_index,
        fallback_uav_dirs=fallback_uav_dirs,
    )

    if uav_path is None:
        raise FileNotFoundError(f"Could not find UAV image for token {token}")

    candidates = load_phog_candidates(
        token=token,
        ranked_dir=ranked_dir,
        sat_df=sat_df,
        filename_index=filename_index,
        fallback_sat_dirs=fallback_sat_dirs,
        top_n=args.phog_top_n,
    )

    if len(candidates) == 0:
        raise RuntimeError(f"No candidates for token {token}")

    print(f"[token {token}] reranking PHOG top {len(candidates)}")

    lsd_df = compute_lsd_alignment_for_token(
        token=token,
        uav_path=Path(uav_path),
        candidates=candidates,
        args=args,
        cache=canvas_cache,
    )

    if len(lsd_df) == 0:
        raise RuntimeError(f"No LSD rows computed for token {token}")

    token_raw_csv = OUT_META_DIR / f"s4c4c_token{token:04d}_{args.canvas_mode}_raw_lsd_scores_top{args.phog_top_n}.csv"
    lsd_df.to_csv(token_raw_csv, index=False)

    profile_rankings: dict[str, pd.DataFrame] = {}
    summary_rows: list[dict[str, Any]] = []

    token_profile_rows = []

    for profile in profiles:
        prof_df = apply_profile(lsd_df, profile, top_n=args.phog_top_n)
        profile_rankings[profile.name] = prof_df

        prof_df = prof_df.copy()
        prof_df["profile"] = profile.name
        token_profile_rows.append(prof_df)

        summary = summarize_profile(token, prof_df, profile)
        summary["raw_lsd_csv"] = str(token_raw_csv)
        summary_rows.append(summary)

    token_profiles_df = pd.concat(token_profile_rows, ignore_index=True)
    token_profiles_csv = OUT_META_DIR / f"s4c4c_token{token:04d}_{args.canvas_mode}_profile_reranks_top{args.phog_top_n}.csv"
    token_profiles_df.to_csv(token_profiles_csv, index=False)

    panel_path = None
    if args.save_panels:
        panel_profiles = [p.strip() for p in args.panel_profiles.split(",") if p.strip()]
        panel_path = OUT_FIG_DIR / f"s4c4c_token{token:04d}_{args.canvas_mode}_panel_top{args.display_top_k}.png"
        render_token_panel(
            token=token,
            profile_rankings=profile_rankings,
            profiles_to_show=panel_profiles,
            display_top_k=args.display_top_k,
            out_path=panel_path,
        )

    for row in summary_rows:
        row["token_profiles_csv"] = str(token_profiles_csv)
        row["panel_path"] = str(panel_path) if panel_path is not None else None

    return summary_rows


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--sequence", default="traj01")
    parser.add_argument("--tokens", default="1,40,60,90,100,129,166,269,516,905")
    parser.add_argument("--uav-index", default=str(DEFAULT_UAV_INDEX))
    parser.add_argument("--sat-index", default=str(DEFAULT_SAT_INDEX))
    parser.add_argument("--s4c1-ranked-dir", default=str(DEFAULT_S4C1_DIR))

    # Macro / LSD settings.
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

    # Reranking candidate and alignment settings.
    parser.add_argument("--phog-top-n", type=int, default=50)
    parser.add_argument("--canvas-mode", default="luma_lsd", choices=["luma_lsd"])
    parser.add_argument("--line-thickness", type=int, default=2)
    parser.add_argument("--max-shift-px", type=int, default=48)
    parser.add_argument("--shift-step-px", type=int, default=8)
    parser.add_argument("--symmetric-weight", type=float, default=0.25)
    parser.add_argument("--max-points", type=int, default=8000)

    # Panels.
    parser.add_argument("--save-panels", action="store_true")
    parser.add_argument("--display-top-k", type=int, default=5)
    parser.add_argument(
        "--panel-profiles",
        default="phog_only,lsd_only,lsd_weak,lsd_balanced,lsd_guard",
    )

    args = parser.parse_args()

    ranked_dir = Path(args.s4c1_ranked_dir)
    tokens = parse_tokens(args.tokens, ranked_dir)

    OUT_META_DIR.mkdir(parents=True, exist_ok=True)
    OUT_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_FIG_DIR.mkdir(parents=True, exist_ok=True)

    uav_index = Path(args.uav_index)
    sat_index = Path(args.sat_index)

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

    profiles = default_profiles()

    print("S4C.4C luma-LSD PHOG top-N reranker")
    print("-----------------------------------")
    print(f"Sequence:       {args.sequence}")
    print(f"Tokens:         {tokens}")
    print(f"PHOG top-N:     {args.phog_top_n}")
    print(f"Canvas mode:    {args.canvas_mode}")
    print(f"Shift search:   ±{args.max_shift_px}px step {args.shift_step_px}px")
    print(f"Sym weight:     {args.symmetric_weight}")
    print("")

    canvas_cache: dict[str, tuple[dict[str, Any], np.ndarray]] = {}
    all_summary_rows: list[dict[str, Any]] = []

    for token in tokens:
        try:
            rows = run_token(
                token=token,
                uav_df=uav_df,
                sat_df=sat_df,
                ranked_dir=ranked_dir,
                filename_index=filename_index,
                fallback_uav_dirs=fallback_uav_dirs,
                fallback_sat_dirs=fallback_sat_dirs,
                profiles=profiles,
                args=args,
                canvas_cache=canvas_cache,
            )
        except Exception as exc:
            print(f"[WARN] token {token}: {exc}")
            continue

        all_summary_rows.extend(rows)

        compact = pd.DataFrame(rows)
        msg = []
        for p in ["phog_only", "lsd_only", "lsd_weak", "lsd_balanced", "lsd_guard"]:
            sub = compact[compact["profile"] == p]
            if len(sub):
                r = sub.iloc[0]
                msg.append(f"{p}: {r['top1_error_m']:.1f}m")
        print(f"[OK] token {token}: " + " | ".join(msg))

    summary_df = pd.DataFrame(all_summary_rows)

    summary_csv = OUT_META_DIR / f"s4c4c_{args.sequence}_{args.canvas_mode}_summary.csv"
    aggregate_csv = OUT_META_DIR / f"s4c4c_{args.sequence}_{args.canvas_mode}_aggregate_by_profile.csv"
    summary_json = OUT_REPORT_DIR / f"s4c4c_{args.sequence}_{args.canvas_mode}_summary.json"
    aggregate_plot = OUT_FIG_DIR / f"s4c4c_{args.sequence}_{args.canvas_mode}_profile_success_rates.png"

    summary_df.to_csv(summary_csv, index=False)

    aggregate_rows: list[dict[str, Any]] = []

    if len(summary_df) > 0:
        for profile_name, group in summary_df.groupby("profile"):
            aggregate_rows.append(
                {
                    "profile": profile_name,
                    "num_queries": int(len(group)),
                    "mean_top1_error_m": float(group["top1_error_m"].mean()),
                    "median_top1_error_m": float(group["top1_error_m"].median()),
                    "mean_best_top10_error_m": float(group["best_top10_error_m"].mean()),
                    "median_best_top10_error_m": float(group["best_top10_error_m"].median()),
                    "top1_under_20m_rate": float(group["top1_under_20m"].mean()),
                    "top1_under_40m_rate": float(group["top1_under_40m"].mean()),
                    "top1_under_60m_rate": float(group["top1_under_60m"].mean()),
                    "top10_under_20m_rate": float(group["top10_under_20m"].mean()),
                    "top10_under_40m_rate": float(group["top10_under_40m"].mean()),
                    "top10_under_60m_rate": float(group["top10_under_60m"].mean()),
                    "mean_top1_lsd_score": float(group["top1_lsd_best_score"].mean()),
                    "median_top1_lsd_score": float(group["top1_lsd_best_score"].median()),
                }
            )

    aggregate_df = pd.DataFrame(aggregate_rows)

    if len(aggregate_df) > 0:
        aggregate_df = aggregate_df.sort_values(
            ["top1_under_40m_rate", "top10_under_40m_rate", "median_top1_error_m"],
            ascending=[False, False, True],
        )

    aggregate_df.to_csv(aggregate_csv, index=False)
    render_aggregate_plot(aggregate_df, aggregate_plot)

    output = {
        "stage": "S4C.4C_luma_LSD_PHOG_topN_reranker",
        "sequence": args.sequence,
        "tokens": tokens,
        "settings": vars(args),
        "summary_csv": str(summary_csv),
        "aggregate_csv": str(aggregate_csv),
        "summary_json": str(summary_json),
        "aggregate_plot": str(aggregate_plot),
        "notes": [
            "No reference coordinates are used in scoring.",
            "center_error_m is used only for evaluation after ranking.",
            "luma-LSD alignment is tested as a PHOG top-N verification signal.",
        ],
        "aggregate_by_profile": aggregate_rows,
    }

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(json_safe(output), f, indent=2)

    print("")
    print("S4C.4C complete")
    print("----------------")
    print(f"Summary CSV:   {summary_csv}")
    print(f"Aggregate CSV: {aggregate_csv}")
    print(f"Summary JSON:  {summary_json}")
    print(f"Plot:          {aggregate_plot}")
    print(f"Figures:       {OUT_FIG_DIR}")

    if len(aggregate_df) > 0:
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
            "median_top1_lsd_score",
        ]
        print(aggregate_df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
