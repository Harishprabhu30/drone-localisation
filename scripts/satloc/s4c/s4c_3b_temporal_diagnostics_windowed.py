#!/usr/bin/env python3
"""
S4C.3B — Temporal diagnostics + anchored/windowed smoothing.

Input:
  Existing S4C.1 PHOG ranked CSVs.

Why this exists:
  S4C.3 global Viterbi smoothing can lock onto one smooth but wrong map path.
  This script avoids that by:
    1. diagnosing sparse token gaps,
    2. splitting at large token gaps,
    3. using local windowed smoothing instead of one global path,
    4. estimating PHOG anchor confidence,
    5. reporting whether anchors are actually reliable after evaluation.

Important:
  - UAV lon/lat and center_error_m are NOT used in scoring.
  - They are used only after ranking/selection for evaluation.
  - `--tokens all` means all available S4C.1 ranked CSVs, not all traj01 frames.

code:
export PYTHONPATH=$PWD/src

python scripts/satloc/s4c/s4c_3b_temporal_diagnostics_windowed.py \
  --sequence traj01 \
  --tokens all \
  --candidate-top-n 50 \
  --window-radius 2 \
  --max-gap-tokens 40 \
  --rank-weight 1.0 \
  --score-weight 0.25 \
  --base-transition-m 60 \
  --step-m-per-token 10 \
  --transition-weight 1.0 \
  --transition-scale-m 180 \
  --max-transition-cost 10

use this if PHOG top-1 is damaged:
python scripts/satloc/s4c/s4c_3b_temporal_diagnostics_windowed.py \
  --sequence traj01 \
  --tokens all \
  --candidate-top-n 50 \
  --window-radius 1 \
  --max-gap-tokens 40 \
  --rank-weight 2.0 \
  --score-weight 0.5 \
  --base-transition-m 60 \
  --step-m-per-token 10 \
  --transition-weight 0.5 \
  --transition-scale-m 220 \
  --max-transition-cost 5
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
DEFAULT_S4C1_DIR = OUT_ROOT / "metadata/s4c_macrocontour_phog_chamfer/s4c1_phog_retrieval"
DEFAULT_SAT_INDEX = OUT_ROOT / "metadata/satellite_tiles_index_enriched.csv"

OUT_META_DIR = OUT_ROOT / "metadata/s4c_macrocontour_phog_chamfer/s4c3b_temporal_diagnostics_windowed"
OUT_REPORT_DIR = OUT_ROOT / "reports/s4c_macrocontour_phog_chamfer/s4c3b_temporal_diagnostics_windowed"
OUT_FIG_DIR = OUT_ROOT / "figures/s4c_macrocontour_phog_chamfer/s4c3b_temporal_diagnostics_windowed"


# -----------------------------
# Basic helpers
# -----------------------------

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
    for part in text.split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    return sorted(out)


def find_latest_ranked_csv(token: int, ranked_dir: Path) -> Path:
    pattern = f"s4c1_token{token:04d}_*_ranked.csv"
    matches = sorted(ranked_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    if not matches:
        raise FileNotFoundError(f"No S4C.1 ranked CSV for token {token}: {ranked_dir / pattern}")
    return matches[0]


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
        val = float(obj)
        if math.isnan(val) or math.isinf(val):
            return None
        return val
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    return obj


def lonlat_distance_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    return float(s4c0.approx_lonlat_error_m(lon1, lat1, lon2, lat2))


def first_rank_under(df: pd.DataFrame, threshold_m: float) -> Optional[int]:
    if "center_error_m" not in df.columns:
        return None
    valid = df[np.isfinite(df["center_error_m"]) & (df["center_error_m"] <= threshold_m)]
    if len(valid) == 0:
        return None
    return int(valid["rank"].min())


# -----------------------------
# Candidate center loading
# -----------------------------

def candidate_center_from_sat_row(
    row: pd.Series,
    sat_df: pd.DataFrame,
    center_lon_col: Optional[str],
    center_lat_col: Optional[str],
    bbox_cols: tuple[Optional[str], Optional[str], Optional[str], Optional[str]],
) -> tuple[float, float]:
    if center_lon_col is not None and center_lat_col is not None:
        lon = safe_float(row[center_lon_col])
        lat = safe_float(row[center_lat_col])
        if np.isfinite(lon) and np.isfinite(lat):
            return lon, lat

    lon_min, lon_max, lat_min, lat_max = bbox_cols
    if all(c is not None for c in [lon_min, lon_max, lat_min, lat_max]):
        a = safe_float(row[lon_min])
        b = safe_float(row[lon_max])
        c = safe_float(row[lat_min])
        d = safe_float(row[lat_max])
        if all(np.isfinite(v) for v in [a, b, c, d]):
            return (a + b) / 2.0, (c + d) / 2.0

    raise ValueError("Could not determine candidate center lon/lat.")


def load_candidates_for_token(
    token: int,
    ranked_dir: Path,
    sat_df: pd.DataFrame,
    top_n: int,
) -> pd.DataFrame:
    csv_path = find_latest_ranked_csv(token, ranked_dir)
    df = pd.read_csv(csv_path)

    if len(df) == 0:
        raise RuntimeError(f"Empty ranked CSV: {csv_path}")

    if "rank" not in df.columns:
        df["rank"] = np.arange(1, len(df) + 1)

    df = df.sort_values("rank").head(top_n).copy().reset_index(drop=True)

    center_lon_col, center_lat_col = s4c0.find_center_cols(sat_df)
    bbox_cols = s4c0.find_bbox_cols(sat_df)

    lons = []
    lats = []

    for _, r in df.iterrows():
        row_pos = int(r["row_pos"])
        sat_row = sat_df.iloc[row_pos]
        lon, lat = candidate_center_from_sat_row(
            sat_row,
            sat_df,
            center_lon_col,
            center_lat_col,
            bbox_cols,
        )
        lons.append(lon)
        lats.append(lat)

    df["token"] = token
    df["source_ranked_csv"] = str(csv_path)
    df["candidate_center_lon"] = lons
    df["candidate_center_lat"] = lats
    df["candidate_local_index"] = np.arange(len(df))

    for col in ["center_error_m", "uav_lon", "uav_lat"]:
        if col not in df.columns:
            df[col] = np.nan

    return df


# -----------------------------
# Diagnostics
# -----------------------------

def candidate_spread_m(df: pd.DataFrame, k: int = 10) -> float:
    sub = df.sort_values("rank").head(k)
    if len(sub) <= 1:
        return 0.0

    top1 = sub.iloc[0]
    dists = []
    for _, r in sub.iloc[1:].iterrows():
        dists.append(
            lonlat_distance_m(
                float(top1["candidate_center_lon"]),
                float(top1["candidate_center_lat"]),
                float(r["candidate_center_lon"]),
                float(r["candidate_center_lat"]),
            )
        )

    return float(np.median(dists)) if dists else 0.0


def frame_diagnostics(
    token: int,
    df: pd.DataFrame,
    prev_token: Optional[int],
) -> dict[str, Any]:
    ranked = df.sort_values("rank").reset_index(drop=True)
    top1 = ranked.iloc[0]
    oracle = ranked.sort_values("center_error_m").iloc[0]

    score1 = safe_float(top1.get("score_cosine", np.nan))
    score2 = safe_float(ranked.iloc[1].get("score_cosine", np.nan)) if len(ranked) > 1 else np.nan

    return {
        "token": token,
        "token_gap_prev": None if prev_token is None else int(token - prev_token),

        "phog_top1_tile_id": int(top1["tile_id"]),
        "phog_top1_error_m": safe_float(top1["center_error_m"]),
        "phog_top1_score": score1,
        "phog_score_margin_1_2": score1 - score2 if np.isfinite(score1) and np.isfinite(score2) else np.nan,

        "first_rank_under_20m": first_rank_under(ranked, 20.0),
        "first_rank_under_40m": first_rank_under(ranked, 40.0),
        "first_rank_under_60m": first_rank_under(ranked, 60.0),

        "oracle_topn_tile_id": int(oracle["tile_id"]),
        "oracle_topn_error_m": safe_float(oracle["center_error_m"]),
        "oracle_topn_rank": int(oracle["rank"]),

        "top10_candidate_spread_m": candidate_spread_m(ranked, k=min(10, len(ranked))),
        "top20_candidate_spread_m": candidate_spread_m(ranked, k=min(20, len(ranked))),
    }


def assign_anchor_flags(diag_df: pd.DataFrame, margin_quantile: float, max_spread_m: float) -> pd.DataFrame:
    out = diag_df.copy()

    margin = out["phog_score_margin_1_2"].astype(float)
    finite_margin = margin[np.isfinite(margin)]

    if len(finite_margin) == 0:
        margin_thresh = float("inf")
    else:
        margin_thresh = float(np.quantile(finite_margin, margin_quantile))

    out["anchor_margin_threshold"] = margin_thresh
    out["is_unsupervised_anchor"] = (
        (out["phog_score_margin_1_2"].astype(float) >= margin_thresh)
        & (out["top10_candidate_spread_m"].astype(float) <= max_spread_m)
    )

    # Evaluation-only anchor quality.
    out["anchor_is_under_40m_eval_only"] = (
        out["is_unsupervised_anchor"] & (out["phog_top1_error_m"].astype(float) <= 40.0)
    )

    return out


# -----------------------------
# Windowed Viterbi
# -----------------------------

def compute_unary_costs(frame: pd.DataFrame, top_n: int, rank_weight: float, score_weight: float) -> np.ndarray:
    ranks = frame["rank"].astype(float).to_numpy()
    rank_cost = np.log1p(ranks) / max(np.log1p(float(top_n)), 1e-6)

    if "score_cosine" in frame.columns:
        scores = frame["score_cosine"].astype(float).to_numpy()
        s_min = np.nanmin(scores)
        s_max = np.nanmax(scores)
        if np.isfinite(s_min) and np.isfinite(s_max) and (s_max - s_min) > 1e-8:
            score_norm = (scores - s_min) / (s_max - s_min)
            score_cost = 1.0 - score_norm
        else:
            score_cost = np.zeros_like(rank_cost)
    else:
        score_cost = np.zeros_like(rank_cost)

    return rank_weight * rank_cost + score_weight * score_cost


def transition_cost_matrix(
    prev_df: pd.DataFrame,
    curr_df: pd.DataFrame,
    delta_token: int,
    base_transition_m: float,
    step_m_per_token: float,
    transition_weight: float,
    transition_scale_m: float,
    max_transition_cost: float,
) -> np.ndarray:
    n_prev = len(prev_df)
    n_curr = len(curr_df)

    out = np.zeros((n_prev, n_curr), dtype=np.float64)

    allowed_m = base_transition_m + step_m_per_token * abs(int(delta_token))
    allowed_m = max(allowed_m, 1.0)

    for i in range(n_prev):
        p = prev_df.iloc[i]
        for j in range(n_curr):
            c = curr_df.iloc[j]
            d = lonlat_distance_m(
                float(p["candidate_center_lon"]),
                float(p["candidate_center_lat"]),
                float(c["candidate_center_lon"]),
                float(c["candidate_center_lat"]),
            )
            excess = max(0.0, d - allowed_m)
            cost = transition_weight * (excess / max(transition_scale_m, 1e-6)) ** 2
            out[i, j] = min(cost, max_transition_cost)

    return out


def run_viterbi_window(frames: list[pd.DataFrame], tokens: list[int], args: argparse.Namespace) -> list[int]:
    unary = [
        compute_unary_costs(
            f,
            top_n=args.candidate_top_n,
            rank_weight=args.rank_weight,
            score_weight=args.score_weight,
        )
        for f in frames
    ]

    dp: list[np.ndarray] = []
    backptr: list[np.ndarray] = []

    dp.append(unary[0].copy())
    backptr.append(np.full((len(frames[0]),), -1, dtype=np.int32))

    for t in range(1, len(frames)):
        delta = int(tokens[t] - tokens[t - 1])

        trans = transition_cost_matrix(
            frames[t - 1],
            frames[t],
            delta_token=delta,
            base_transition_m=args.base_transition_m,
            step_m_per_token=args.step_m_per_token,
            transition_weight=args.transition_weight,
            transition_scale_m=args.transition_scale_m,
            max_transition_cost=args.max_transition_cost,
        )

        prev_total = dp[t - 1][:, None] + trans
        best_prev = np.argmin(prev_total, axis=0)
        best_cost = prev_total[best_prev, np.arange(prev_total.shape[1])]

        dp.append(unary[t] + best_cost)
        backptr.append(best_prev.astype(np.int32))

    path = [0] * len(frames)
    path[-1] = int(np.argmin(dp[-1]))

    for t in range(len(frames) - 1, 0, -1):
        path[t - 1] = int(backptr[t][path[t]])

    return path


def split_segments(tokens: list[int], max_gap_tokens: int) -> list[tuple[int, int]]:
    """
    Returns list of inclusive index ranges [start, end].
    """
    if not tokens:
        return []

    segments = []
    start = 0

    for i in range(1, len(tokens)):
        if (tokens[i] - tokens[i - 1]) > max_gap_tokens:
            segments.append((start, i - 1))
            start = i

    segments.append((start, len(tokens) - 1))
    return segments


def windowed_selection(
    frames: list[pd.DataFrame],
    tokens: list[int],
    diag_df: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:
    selected_rows = []

    segments = split_segments(tokens, args.max_gap_tokens)

    anchor_by_token = dict(
        zip(diag_df["token"].astype(int), diag_df["is_unsupervised_anchor"].astype(bool))
    )

    for seg_id, (start, end) in enumerate(segments):
        for center_idx in range(start, end + 1):
            token = tokens[center_idx]

            # Optional anchor forcing. Default should normally be false until anchor quality is proven.
            if args.force_anchors and anchor_by_token.get(token, False):
                chosen = frames[center_idx].sort_values("rank").iloc[0].copy()
                chosen["selection_reason"] = "forced_anchor_top1"
                chosen["segment_id"] = seg_id
                chosen["window_start_token"] = tokens[center_idx]
                chosen["window_end_token"] = tokens[center_idx]
                chosen["window_size"] = 1
                selected_rows.append(chosen)
                continue

            w0 = max(start, center_idx - args.window_radius)
            w1 = min(end, center_idx + args.window_radius)

            win_frames = frames[w0:w1 + 1]
            win_tokens = tokens[w0:w1 + 1]

            path = run_viterbi_window(win_frames, win_tokens, args)

            local_center = center_idx - w0
            chosen_idx = path[local_center]

            chosen = frames[center_idx].iloc[chosen_idx].copy()
            chosen["selection_reason"] = "windowed_viterbi"
            chosen["segment_id"] = seg_id
            chosen["window_start_token"] = win_tokens[0]
            chosen["window_end_token"] = win_tokens[-1]
            chosen["window_size"] = len(win_tokens)

            selected_rows.append(chosen)

    selected = pd.DataFrame(selected_rows)
    selected = selected.sort_values("token").reset_index(drop=True)
    selected["windowed_rank_index"] = np.arange(1, len(selected) + 1)
    return selected


# -----------------------------
# Metrics and plots
# -----------------------------

def metric_summary(df: pd.DataFrame, error_col: str, prefix: str) -> dict[str, Any]:
    err = df[error_col].astype(float)
    return {
        f"{prefix}_mean_error_m": float(err.mean()),
        f"{prefix}_median_error_m": float(err.median()),
        f"{prefix}_max_error_m": float(err.max()),
        f"{prefix}_under_20m_rate": float((err <= 20.0).mean()),
        f"{prefix}_under_40m_rate": float((err <= 40.0).mean()),
        f"{prefix}_under_60m_rate": float((err <= 60.0).mean()),
        f"{prefix}_under_100m_rate": float((err <= 100.0).mean()),
    }


def render_error_plot(compare: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 5.4))

    x = np.arange(len(compare))
    labels = compare["token"].astype(str).tolist()

    ax.plot(x, compare["phog_top1_error_m"], marker="o", linewidth=1.1, label="PHOG top-1")
    ax.plot(x, compare["windowed_error_m"], marker="o", linewidth=1.1, label="Windowed selected")
    ax.plot(x, compare["oracle_topn_error_m"], marker=".", linewidth=1.0, label="Oracle best top-N")

    ax.axhline(20, linestyle="--", linewidth=1.0)
    ax.axhline(40, linestyle="--", linewidth=1.0)
    ax.axhline(60, linestyle="--", linewidth=1.0)

    ax.set_yscale("symlog", linthresh=100)
    step = max(1, len(x) // 20)
    ax.set_xticks(x[::step])
    ax.set_xticklabels(labels[::step], rotation=45, ha="right")
    ax.set_xlabel("query token")
    ax.set_ylabel("center error [m], symlog")
    ax.set_title("S4C.3B windowed temporal smoothing — error comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def render_rank_plot(compare: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 4.8))

    x = np.arange(len(compare))
    labels = compare["token"].astype(str).tolist()

    ax.plot(x, compare["windowed_selected_phog_rank"], marker="o", linewidth=1.1)
    step = max(1, len(x) // 20)
    ax.set_xticks(x[::step])
    ax.set_xticklabels(labels[::step], rotation=45, ha="right")
    ax.set_xlabel("query token")
    ax.set_ylabel("selected original PHOG rank")
    ax.set_title("S4C.3B selected candidate PHOG rank")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def render_oracle_rank_plot(diag: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 4.8))

    x = np.arange(len(diag))
    labels = diag["token"].astype(str).tolist()

    ax.plot(x, diag["oracle_topn_rank"], marker="o", linewidth=1.1, label="oracle rank in top-N")
    ax.axhline(10, linestyle="--", linewidth=1.0)
    ax.axhline(20, linestyle="--", linewidth=1.0)
    ax.axhline(50, linestyle="--", linewidth=1.0)

    step = max(1, len(x) // 20)
    ax.set_xticks(x[::step])
    ax.set_xticklabels(labels[::step], rotation=45, ha="right")
    ax.set_xlabel("query token")
    ax.set_ylabel("rank of best available candidate")
    ax.set_title("S4C.3B diagnostic — oracle candidate rank inside PHOG top-N")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def render_path_plot(compare: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 7.0))

    ax.plot(
        compare["phog_top1_center_lon"],
        compare["phog_top1_center_lat"],
        marker=".",
        linewidth=0.8,
        label="PHOG top-1 path",
    )

    ax.plot(
        compare["windowed_center_lon"],
        compare["windowed_center_lat"],
        marker="o",
        linewidth=1.1,
        label="Windowed selected path",
    )

    if compare["uav_lon"].notna().any() and compare["uav_lat"].notna().any():
        ax.plot(
            compare["uav_lon"],
            compare["uav_lat"],
            marker=".",
            linewidth=1.0,
            label="Reference/eval-only sparse points",
        )

    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_title("S4C.3B selected sparse map path")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def render_anchor_scatter(diag: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 5.2))

    ax.scatter(
        diag["phog_score_margin_1_2"],
        diag["phog_top1_error_m"],
        s=32,
        label="all frames",
    )

    anchors = diag[diag["is_unsupervised_anchor"].astype(bool)]
    if len(anchors) > 0:
        ax.scatter(
            anchors["phog_score_margin_1_2"],
            anchors["phog_top1_error_m"],
            s=52,
            marker="x",
            label="unsupervised anchors",
        )

    ax.axhline(40, linestyle="--", linewidth=1.0)
    ax.set_yscale("symlog", linthresh=100)
    ax.set_xlabel("PHOG score margin top1-top2")
    ax.set_ylabel("PHOG top-1 error [m], symlog")
    ax.set_title("S4C.3B anchor diagnostic")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--sequence", default="traj01")
    parser.add_argument("--tokens", default="all")
    parser.add_argument("--s4c1-ranked-dir", default=str(DEFAULT_S4C1_DIR))
    parser.add_argument("--sat-index", default=str(DEFAULT_SAT_INDEX))
    parser.add_argument("--candidate-top-n", type=int, default=50)

    # Windowing / segmentation.
    parser.add_argument("--window-radius", type=int, default=2)
    parser.add_argument("--max-gap-tokens", type=int, default=40)

    # Cost weights.
    parser.add_argument("--rank-weight", type=float, default=1.0)
    parser.add_argument("--score-weight", type=float, default=0.25)
    parser.add_argument("--base-transition-m", type=float, default=60.0)
    parser.add_argument("--step-m-per-token", type=float, default=10.0)
    parser.add_argument("--transition-weight", type=float, default=1.0)
    parser.add_argument("--transition-scale-m", type=float, default=180.0)
    parser.add_argument("--max-transition-cost", type=float, default=10.0)

    # Anchor diagnostics.
    parser.add_argument("--anchor-margin-quantile", type=float, default=0.75)
    parser.add_argument("--anchor-max-spread-m", type=float, default=800.0)
    parser.add_argument("--force-anchors", action="store_true")

    args = parser.parse_args()

    ranked_dir = Path(args.s4c1_ranked_dir)
    sat_index = Path(args.sat_index)

    if not ranked_dir.exists():
        raise FileNotFoundError(f"Missing S4C.1 ranked dir: {ranked_dir}")
    if not sat_index.exists():
        raise FileNotFoundError(f"Missing satellite index: {sat_index}")

    OUT_META_DIR.mkdir(parents=True, exist_ok=True)
    OUT_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_FIG_DIR.mkdir(parents=True, exist_ok=True)

    tokens = parse_tokens(args.tokens, ranked_dir)
    sat_df = pd.read_csv(sat_index)

    print("S4C.3B Temporal diagnostics + windowed smoothing")
    print("------------------------------------------------")
    print(f"Sequence:        {args.sequence}")
    print(f"Available tokens:{len(tokens)}")
    print(f"Candidate top-N: {args.candidate_top_n}")
    print(f"Window radius:   {args.window_radius}")
    print(f"Max gap tokens:  {args.max_gap_tokens}")
    print(f"Force anchors:   {args.force_anchors}")
    print("")
    print("Important: --tokens all means all available S4C.1 ranked CSVs, not all traj01 frames.")
    print("")

    frames: list[pd.DataFrame] = []
    loaded_tokens: list[int] = []

    for token in tokens:
        try:
            cand = load_candidates_for_token(
                token=token,
                ranked_dir=ranked_dir,
                sat_df=sat_df,
                top_n=args.candidate_top_n,
            )
        except Exception as exc:
            print(f"[WARN] token {token}: {exc}")
            continue

        frames.append(cand)
        loaded_tokens.append(token)

    if len(frames) < 2:
        raise RuntimeError("Need at least two loaded frames.")

    diag_rows = []
    prev = None
    for token, cand in zip(loaded_tokens, frames):
        diag_rows.append(frame_diagnostics(token, cand, prev))
        prev = token

    diag_df = pd.DataFrame(diag_rows)
    diag_df = assign_anchor_flags(
        diag_df,
        margin_quantile=args.anchor_margin_quantile,
        max_spread_m=args.anchor_max_spread_m,
    )

    segments = split_segments(loaded_tokens, args.max_gap_tokens)

    selected_df = windowed_selection(
        frames=frames,
        tokens=loaded_tokens,
        diag_df=diag_df,
        args=args,
    )

    compare_rows = []

    for token, cand in zip(loaded_tokens, frames):
        top1 = cand.sort_values("rank").iloc[0]
        oracle = cand.sort_values("center_error_m").iloc[0]
        chosen = selected_df[selected_df["token"].astype(int) == int(token)].iloc[0]

        compare_rows.append(
            {
                "token": token,

                "phog_top1_tile_id": int(top1["tile_id"]),
                "phog_top1_error_m": safe_float(top1["center_error_m"]),
                "phog_top1_score": safe_float(top1.get("score_cosine", np.nan)),
                "phog_top1_center_lon": safe_float(top1["candidate_center_lon"]),
                "phog_top1_center_lat": safe_float(top1["candidate_center_lat"]),

                "windowed_tile_id": int(chosen["tile_id"]),
                "windowed_error_m": safe_float(chosen["center_error_m"]),
                "windowed_selected_phog_rank": int(chosen["rank"]),
                "windowed_score": safe_float(chosen.get("score_cosine", np.nan)),
                "windowed_center_lon": safe_float(chosen["candidate_center_lon"]),
                "windowed_center_lat": safe_float(chosen["candidate_center_lat"]),
                "selection_reason": chosen.get("selection_reason", ""),
                "segment_id": int(chosen.get("segment_id", -1)),
                "window_start_token": int(chosen.get("window_start_token", token)),
                "window_end_token": int(chosen.get("window_end_token", token)),
                "window_size": int(chosen.get("window_size", 1)),

                "oracle_topn_tile_id": int(oracle["tile_id"]),
                "oracle_topn_error_m": safe_float(oracle["center_error_m"]),
                "oracle_topn_rank": int(oracle["rank"]),

                "uav_lon": safe_float(chosen.get("uav_lon", np.nan)),
                "uav_lat": safe_float(chosen.get("uav_lat", np.nan)),
            }
        )

    compare_df = pd.DataFrame(compare_rows)

    slug = (
        f"top{args.candidate_top_n}"
        f"_wr{args.window_radius}"
        f"_gap{args.max_gap_tokens}"
        f"_rw{str(args.rank_weight).replace('.', 'p')}"
        f"_sw{str(args.score_weight).replace('.', 'p')}"
        f"_bt{int(args.base_transition_m)}"
        f"_spt{str(args.step_m_per_token).replace('.', 'p')}"
        f"_tw{str(args.transition_weight).replace('.', 'p')}"
        f"_forceA{int(args.force_anchors)}"
    )

    diag_csv = OUT_META_DIR / f"s4c3b_{args.sequence}_{slug}_diagnostics.csv"
    selected_csv = OUT_META_DIR / f"s4c3b_{args.sequence}_{slug}_selected_windowed.csv"
    compare_csv = OUT_META_DIR / f"s4c3b_{args.sequence}_{slug}_comparison.csv"

    diag_df.to_csv(diag_csv, index=False)
    selected_df.to_csv(selected_csv, index=False)
    compare_df.to_csv(compare_csv, index=False)

    error_plot = OUT_FIG_DIR / f"s4c3b_{args.sequence}_{slug}_error_comparison.png"
    rank_plot = OUT_FIG_DIR / f"s4c3b_{args.sequence}_{slug}_selected_rank.png"
    oracle_rank_plot = OUT_FIG_DIR / f"s4c3b_{args.sequence}_{slug}_oracle_rank.png"
    path_plot = OUT_FIG_DIR / f"s4c3b_{args.sequence}_{slug}_path_lonlat.png"
    anchor_plot = OUT_FIG_DIR / f"s4c3b_{args.sequence}_{slug}_anchor_scatter.png"

    render_error_plot(compare_df, error_plot)
    render_rank_plot(compare_df, rank_plot)
    render_oracle_rank_plot(diag_df, oracle_rank_plot)
    render_path_plot(compare_df, path_plot)
    render_anchor_scatter(diag_df, anchor_plot)

    summary = {
        "stage": "S4C.3B_temporal_diagnostics_windowed",
        "sequence": args.sequence,
        "num_available_ranked_frames": len(loaded_tokens),
        "tokens": loaded_tokens,
        "settings": vars(args),
        "segments": [{"start_token": loaded_tokens[a], "end_token": loaded_tokens[b], "length": b - a + 1} for a, b in segments],
        "num_segments": len(segments),
        "diagnostics_csv": str(diag_csv),
        "selected_csv": str(selected_csv),
        "comparison_csv": str(compare_csv),
        "figures": {
            "error_plot": str(error_plot),
            "rank_plot": str(rank_plot),
            "oracle_rank_plot": str(oracle_rank_plot),
            "path_plot": str(path_plot),
            "anchor_plot": str(anchor_plot),
        },
        "notes": [
            "--tokens all means all available S4C.1 ranked CSVs, not all frames in traj01.",
            "Large token gaps are split into separate segments.",
            "Windowed Viterbi is local; it avoids one global smooth-path lock-in.",
            "UAV lon/lat and center_error_m are used only after selection for evaluation.",
            "Anchor flags are diagnostic unless --force-anchors is enabled.",
        ],
    }

    summary.update(metric_summary(compare_df, "phog_top1_error_m", "phog_top1"))
    summary.update(metric_summary(compare_df, "windowed_error_m", "windowed"))
    summary.update(metric_summary(compare_df, "oracle_topn_error_m", "oracle_topn"))

    anchor_count = int(diag_df["is_unsupervised_anchor"].sum())
    if anchor_count > 0:
        anchor_eval = diag_df[diag_df["is_unsupervised_anchor"].astype(bool)]
        anchor_under40 = float((anchor_eval["phog_top1_error_m"].astype(float) <= 40.0).mean())
    else:
        anchor_under40 = None

    token_gaps = diag_df["token_gap_prev"].dropna().astype(float)

    summary.update(
        {
            "mean_token_gap": float(token_gaps.mean()) if len(token_gaps) else None,
            "median_token_gap": float(token_gaps.median()) if len(token_gaps) else None,
            "max_token_gap": float(token_gaps.max()) if len(token_gaps) else None,
            "anchor_count": anchor_count,
            "anchor_under_40m_rate_eval_only": anchor_under40,
            "mean_windowed_selected_phog_rank": float(compare_df["windowed_selected_phog_rank"].mean()),
            "median_windowed_selected_phog_rank": float(compare_df["windowed_selected_phog_rank"].median()),
            "oracle_median_rank": float(diag_df["oracle_topn_rank"].median()),
            "oracle_under_rank10_rate": float((diag_df["oracle_topn_rank"].astype(float) <= 10).mean()),
            "oracle_under_rank20_rate": float((diag_df["oracle_topn_rank"].astype(float) <= 20).mean()),
            "oracle_under_rank50_rate": float((diag_df["oracle_topn_rank"].astype(float) <= 50).mean()),
        }
    )

    summary_json = OUT_REPORT_DIR / f"s4c3b_{args.sequence}_{slug}_summary.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(json_safe(summary), f, indent=2)

    compact = pd.DataFrame(
        [
            {
                "method": "PHOG top-1",
                "under_20m_rate": summary["phog_top1_under_20m_rate"],
                "under_40m_rate": summary["phog_top1_under_40m_rate"],
                "under_60m_rate": summary["phog_top1_under_60m_rate"],
                "median_error_m": summary["phog_top1_median_error_m"],
                "mean_error_m": summary["phog_top1_mean_error_m"],
            },
            {
                "method": "Windowed selected",
                "under_20m_rate": summary["windowed_under_20m_rate"],
                "under_40m_rate": summary["windowed_under_40m_rate"],
                "under_60m_rate": summary["windowed_under_60m_rate"],
                "median_error_m": summary["windowed_median_error_m"],
                "mean_error_m": summary["windowed_mean_error_m"],
            },
            {
                "method": "Oracle best top-N",
                "under_20m_rate": summary["oracle_topn_under_20m_rate"],
                "under_40m_rate": summary["oracle_topn_under_40m_rate"],
                "under_60m_rate": summary["oracle_topn_under_60m_rate"],
                "median_error_m": summary["oracle_topn_median_error_m"],
                "mean_error_m": summary["oracle_topn_mean_error_m"],
            },
        ]
    )

    print("")
    print("S4C.3B complete")
    print("----------------")
    print(f"Diagnostics CSV: {diag_csv}")
    print(f"Selected CSV:    {selected_csv}")
    print(f"Comparison CSV:  {compare_csv}")
    print(f"Summary JSON:    {summary_json}")
    print(f"Figures:         {OUT_FIG_DIR}")
    print("")
    print("Compact result")
    print("--------------")
    print(compact.to_string(index=False))
    print("")
    print("Temporal diagnostics")
    print("--------------------")
    print(f"Segments:                 {len(segments)}")
    print(f"Mean token gap:            {summary['mean_token_gap']}")
    print(f"Median token gap:          {summary['median_token_gap']}")
    print(f"Max token gap:             {summary['max_token_gap']}")
    print(f"Anchor count:              {anchor_count}")
    print(f"Anchor <=40m rate eval:    {anchor_under40}")
    print(f"Oracle median rank:        {summary['oracle_median_rank']:.2f}")
    print(f"Oracle rank<=10 rate:      {summary['oracle_under_rank10_rate']:.3f}")
    print(f"Oracle rank<=20 rate:      {summary['oracle_under_rank20_rate']:.3f}")
    print(f"Mean selected PHOG rank:   {summary['mean_windowed_selected_phog_rank']:.2f}")
    print(f"Median selected PHOG rank: {summary['median_windowed_selected_phog_rank']:.2f}")


if __name__ == "__main__":
    main()
