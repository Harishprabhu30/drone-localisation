#!/usr/bin/env python3
"""
S4C.3 — Temporal consistency / sequence-level candidate smoothing.

Input:
  Existing S4C.1 PHOG ranked CSVs.

Purpose:
  Single-frame PHOG often places near-correct candidates in top-k, but top-1
  is unstable. This script performs Viterbi-style sequence smoothing over
  PHOG top-N candidates.

Scoring:
  unary cost:
    based only on PHOG rank and PHOG score

  transition cost:
    based only on candidate map-center distance between consecutive frames
    and token spacing

Important:
  - UAV filename lon/lat is not used in scoring.
  - center_error_m is used only for evaluation after selecting the path.
  - This is still classical/explainable map retrieval.

code command:
export PYTHONPATH=$PWD/src

python scripts/satloc/s4c/s4c_3_temporal_consistency.py \
  --sequence traj01 \
  --tokens all \
  --candidate-top-n 50 \
  --rank-weight 1.0 \
  --score-weight 0.25 \
  --base-transition-m 80 \
  --step-m-per-token 12 \
  --transition-weight 2.0 \
  --transition-scale-m 150 \
  --max-transition-cost 20
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

OUT_META_DIR = OUT_ROOT / "metadata/s4c_macrocontour_phog_chamfer/s4c3_temporal_consistency"
OUT_REPORT_DIR = OUT_ROOT / "reports/s4c_macrocontour_phog_chamfer/s4c3_temporal_consistency"
OUT_FIG_DIR = OUT_ROOT / "figures/s4c_macrocontour_phog_chamfer/s4c3_temporal_consistency"


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

    raise ValueError("Could not determine candidate center lon/lat from satellite index row.")


def load_candidates_for_token(
    token: int,
    ranked_dir: Path,
    sat_df: pd.DataFrame,
    top_n: int,
) -> pd.DataFrame:
    csv_path = find_latest_ranked_csv(token, ranked_dir)
    df = pd.read_csv(csv_path)

    if len(df) == 0:
        raise RuntimeError(f"Empty S4C.1 ranked CSV: {csv_path}")

    if "rank" not in df.columns:
        df["rank"] = np.arange(1, len(df) + 1)

    df = df.sort_values("rank").head(top_n).copy().reset_index(drop=True)

    center_lon_col, center_lat_col = s4c0.find_center_cols(sat_df)
    bbox_cols = s4c0.find_bbox_cols(sat_df)

    center_lons = []
    center_lats = []

    for _, row in df.iterrows():
        row_pos = int(row["row_pos"])
        sat_row = sat_df.iloc[row_pos]
        lon, lat = candidate_center_from_sat_row(
            sat_row,
            sat_df,
            center_lon_col,
            center_lat_col,
            bbox_cols,
        )
        center_lons.append(lon)
        center_lats.append(lat)

    df["token"] = token
    df["source_ranked_csv"] = str(csv_path)
    df["candidate_center_lon"] = center_lons
    df["candidate_center_lat"] = center_lats
    df["candidate_local_index"] = np.arange(len(df))

    # Evaluation/reference fields may exist from S4C.1 ranked CSV.
    for col in ["center_error_m", "uav_lon", "uav_lat"]:
        if col not in df.columns:
            df[col] = np.nan

    return df


def compute_unary_costs(
    candidates: pd.DataFrame,
    top_n: int,
    rank_weight: float,
    score_weight: float,
) -> np.ndarray:
    ranks = candidates["rank"].astype(float).to_numpy()
    rank_cost = np.log1p(ranks) / max(np.log1p(float(top_n)), 1e-6)

    if "score_cosine" in candidates.columns:
        scores = candidates["score_cosine"].astype(float).to_numpy()
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


def transition_matrix(
    prev_df: pd.DataFrame,
    curr_df: pd.DataFrame,
    delta_token: int,
    base_transition_m: float,
    step_m_per_token: float,
    transition_weight: float,
    transition_scale_m: float,
    max_transition_cost: float,
) -> np.ndarray:
    prev_lon = prev_df["candidate_center_lon"].astype(float).to_numpy()
    prev_lat = prev_df["candidate_center_lat"].astype(float).to_numpy()
    curr_lon = curr_df["candidate_center_lon"].astype(float).to_numpy()
    curr_lat = curr_df["candidate_center_lat"].astype(float).to_numpy()

    n_prev = len(prev_df)
    n_curr = len(curr_df)

    mat = np.zeros((n_prev, n_curr), dtype=np.float64)

    allowed_m = base_transition_m + step_m_per_token * abs(int(delta_token))
    allowed_m = max(allowed_m, 1.0)

    for i in range(n_prev):
        for j in range(n_curr):
            d = lonlat_distance_m(prev_lon[i], prev_lat[i], curr_lon[j], curr_lat[j])
            excess = max(0.0, d - allowed_m)
            cost = transition_weight * (excess / max(transition_scale_m, 1e-6)) ** 2
            mat[i, j] = min(cost, max_transition_cost)

    return mat


def run_viterbi(
    frames: list[pd.DataFrame],
    tokens: list[int],
    args: argparse.Namespace,
) -> tuple[list[int], list[np.ndarray], list[np.ndarray]]:
    unary_costs = [
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

    dp.append(unary_costs[0].copy())
    backptr.append(np.full((len(frames[0]),), -1, dtype=np.int32))

    for t in range(1, len(frames)):
        delta_token = int(tokens[t] - tokens[t - 1])

        trans = transition_matrix(
            prev_df=frames[t - 1],
            curr_df=frames[t],
            delta_token=delta_token,
            base_transition_m=args.base_transition_m,
            step_m_per_token=args.step_m_per_token,
            transition_weight=args.transition_weight,
            transition_scale_m=args.transition_scale_m,
            max_transition_cost=args.max_transition_cost,
        )

        prev_cost = dp[t - 1][:, None] + trans
        best_prev = np.argmin(prev_cost, axis=0)
        best_cost = prev_cost[best_prev, np.arange(prev_cost.shape[1])]

        dp.append(unary_costs[t] + best_cost)
        backptr.append(best_prev.astype(np.int32))

    path = [0] * len(frames)
    path[-1] = int(np.argmin(dp[-1]))

    for t in range(len(frames) - 1, 0, -1):
        path[t - 1] = int(backptr[t][path[t]])

    return path, unary_costs, dp


def add_transition_diagnostics(
    selected_df: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:
    out = selected_df.copy()

    trans_dists = [np.nan]
    allowed_ms = [np.nan]
    excess_ms = [np.nan]

    for i in range(1, len(out)):
        prev = out.iloc[i - 1]
        curr = out.iloc[i]

        d = lonlat_distance_m(
            float(prev["candidate_center_lon"]),
            float(prev["candidate_center_lat"]),
            float(curr["candidate_center_lon"]),
            float(curr["candidate_center_lat"]),
        )

        delta_token = int(curr["token"] - prev["token"])
        allowed = args.base_transition_m + args.step_m_per_token * abs(delta_token)
        excess = max(0.0, d - allowed)

        trans_dists.append(d)
        allowed_ms.append(allowed)
        excess_ms.append(excess)

    out["selected_transition_distance_m"] = trans_dists
    out["allowed_transition_m"] = allowed_ms
    out["transition_excess_m"] = excess_ms

    return out


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


def render_error_plot(compare_df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 5.5))

    x = np.arange(len(compare_df))
    labels = compare_df["token"].astype(str).tolist()

    ax.plot(x, compare_df["phog_top1_error_m"], marker="o", linewidth=1.2, label="PHOG top-1")
    ax.plot(x, compare_df["temporal_error_m"], marker="o", linewidth=1.2, label="Temporal selected")
    ax.plot(x, compare_df["oracle_topn_error_m"], marker=".", linewidth=1.0, label="Oracle best in top-N")

    ax.axhline(20, linestyle="--", linewidth=1.0)
    ax.axhline(40, linestyle="--", linewidth=1.0)
    ax.axhline(60, linestyle="--", linewidth=1.0)

    ax.set_yscale("symlog", linthresh=100)
    ax.set_xticks(x[:: max(1, len(x) // 20)])
    ax.set_xticklabels(labels[:: max(1, len(x) // 20)], rotation=45, ha="right")
    ax.set_xlabel("query token")
    ax.set_ylabel("center error [m], symlog")
    ax.set_title("S4C.3 temporal consistency — error before/after")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def render_rank_plot(compare_df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 4.8))

    x = np.arange(len(compare_df))
    labels = compare_df["token"].astype(str).tolist()

    ax.plot(x, compare_df["temporal_selected_phog_rank"], marker="o", linewidth=1.2)
    ax.set_xticks(x[:: max(1, len(x) // 20)])
    ax.set_xticklabels(labels[:: max(1, len(x) // 20)], rotation=45, ha="right")
    ax.set_xlabel("query token")
    ax.set_ylabel("selected candidate original PHOG rank")
    ax.set_title("S4C.3 temporal selected candidate rank")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def render_path_plot(compare_df: pd.DataFrame, out_path: Path) -> None:
    has_ref = (
        "uav_lon" in compare_df.columns
        and "uav_lat" in compare_df.columns
        and compare_df["uav_lon"].notna().any()
        and compare_df["uav_lat"].notna().any()
    )

    fig, ax = plt.subplots(figsize=(7.5, 7.0))

    ax.plot(
        compare_df["temporal_center_lon"],
        compare_df["temporal_center_lat"],
        marker="o",
        linewidth=1.2,
        label="Temporal selected path",
    )

    ax.plot(
        compare_df["phog_top1_center_lon"],
        compare_df["phog_top1_center_lat"],
        marker=".",
        linewidth=0.8,
        label="PHOG top-1 path",
    )

    if has_ref:
        ax.plot(
            compare_df["uav_lon"],
            compare_df["uav_lat"],
            marker=".",
            linewidth=1.0,
            label="Reference path / eval only",
        )

    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_title("S4C.3 selected map path")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")

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

    # Unary score weights.
    parser.add_argument("--rank-weight", type=float, default=1.0)
    parser.add_argument("--score-weight", type=float, default=0.25)

    # Transition model.
    parser.add_argument("--base-transition-m", type=float, default=80.0)
    parser.add_argument("--step-m-per-token", type=float, default=12.0)
    parser.add_argument("--transition-weight", type=float, default=2.0)
    parser.add_argument("--transition-scale-m", type=float, default=150.0)
    parser.add_argument("--max-transition-cost", type=float, default=20.0)

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

    print("S4C.3 Temporal consistency")
    print("-------------------------")
    print(f"Sequence:          {args.sequence}")
    print(f"Tokens:            {len(tokens)}")
    print(f"Candidate top-N:   {args.candidate_top_n}")
    print(f"Rank weight:       {args.rank_weight}")
    print(f"Score weight:      {args.score_weight}")
    print(f"Base transition:   {args.base_transition_m} m")
    print(f"Step/token:        {args.step_m_per_token} m")
    print(f"Transition weight: {args.transition_weight}")
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

        if len(cand) == 0:
            print(f"[WARN] token {token}: no candidates")
            continue

        frames.append(cand)
        loaded_tokens.append(token)

    if len(frames) < 2:
        raise RuntimeError("Need at least two frames for temporal consistency.")

    path_indices, unary_costs, dp_costs = run_viterbi(frames, loaded_tokens, args)

    selected_rows = []
    baseline_rows = []
    all_candidate_rows = []

    for frame_idx, (token, cand_df, selected_idx, unary, dp) in enumerate(
        zip(loaded_tokens, frames, path_indices, unary_costs, dp_costs)
    ):
        cand_df = cand_df.copy()
        cand_df["unary_cost"] = unary
        cand_df["viterbi_dp_cost"] = dp
        cand_df["sequence_frame_index"] = frame_idx
        cand_df["is_temporal_selected"] = False
        cand_df.loc[selected_idx, "is_temporal_selected"] = True

        selected = cand_df.iloc[selected_idx].copy()
        top1 = cand_df.sort_values("rank").iloc[0].copy()
        oracle = cand_df.sort_values("center_error_m").iloc[0].copy()

        selected_rows.append(selected)
        baseline_rows.append(
            {
                "sequence_frame_index": frame_idx,
                "token": token,

                "phog_top1_tile_id": int(top1["tile_id"]),
                "phog_top1_error_m": safe_float(top1["center_error_m"]),
                "phog_top1_rank": int(top1["rank"]),
                "phog_top1_score_cosine": safe_float(top1.get("score_cosine", np.nan)),
                "phog_top1_center_lon": safe_float(top1["candidate_center_lon"]),
                "phog_top1_center_lat": safe_float(top1["candidate_center_lat"]),

                "temporal_tile_id": int(selected["tile_id"]),
                "temporal_error_m": safe_float(selected["center_error_m"]),
                "temporal_selected_phog_rank": int(selected["rank"]),
                "temporal_score_cosine": safe_float(selected.get("score_cosine", np.nan)),
                "temporal_center_lon": safe_float(selected["candidate_center_lon"]),
                "temporal_center_lat": safe_float(selected["candidate_center_lat"]),

                "oracle_topn_tile_id": int(oracle["tile_id"]),
                "oracle_topn_error_m": safe_float(oracle["center_error_m"]),
                "oracle_topn_rank": int(oracle["rank"]),

                "uav_lon": safe_float(selected.get("uav_lon", np.nan)),
                "uav_lat": safe_float(selected.get("uav_lat", np.nan)),
            }
        )

        all_candidate_rows.append(cand_df)

    selected_df = pd.DataFrame(selected_rows)
    selected_df = add_transition_diagnostics(selected_df, args)

    compare_df = pd.DataFrame(baseline_rows)
    selected_transition_cols = selected_df[
        [
            "token",
            "selected_transition_distance_m",
            "allowed_transition_m",
            "transition_excess_m",
        ]
    ].copy()

    compare_df = compare_df.merge(selected_transition_cols, on="token", how="left")

    all_candidates_df = pd.concat(all_candidate_rows, ignore_index=True)

    slug = (
        f"top{args.candidate_top_n}"
        f"_rw{str(args.rank_weight).replace('.', 'p')}"
        f"_sw{str(args.score_weight).replace('.', 'p')}"
        f"_bt{int(args.base_transition_m)}"
        f"_spt{str(args.step_m_per_token).replace('.', 'p')}"
        f"_tw{str(args.transition_weight).replace('.', 'p')}"
    )

    selected_csv = OUT_META_DIR / f"s4c3_{args.sequence}_{slug}_selected_path.csv"
    compare_csv = OUT_META_DIR / f"s4c3_{args.sequence}_{slug}_comparison.csv"
    all_candidates_csv = OUT_META_DIR / f"s4c3_{args.sequence}_{slug}_candidate_costs.csv"

    selected_df.to_csv(selected_csv, index=False)
    compare_df.to_csv(compare_csv, index=False)
    all_candidates_df.to_csv(all_candidates_csv, index=False)

    error_plot = OUT_FIG_DIR / f"s4c3_{args.sequence}_{slug}_error_before_after.png"
    rank_plot = OUT_FIG_DIR / f"s4c3_{args.sequence}_{slug}_selected_rank.png"
    path_plot = OUT_FIG_DIR / f"s4c3_{args.sequence}_{slug}_path_lonlat.png"

    render_error_plot(compare_df, error_plot)
    render_rank_plot(compare_df, rank_plot)
    render_path_plot(compare_df, path_plot)

    summary = {
        "stage": "S4C.3_temporal_consistency",
        "sequence": args.sequence,
        "num_frames": len(compare_df),
        "tokens": loaded_tokens,
        "settings": vars(args),
        "selected_path_csv": str(selected_csv),
        "comparison_csv": str(compare_csv),
        "candidate_costs_csv": str(all_candidates_csv),
        "error_plot": str(error_plot),
        "rank_plot": str(rank_plot),
        "path_plot": str(path_plot),
        "notes": [
            "Scoring uses PHOG rank/score and candidate map-center transition smoothness.",
            "Reference/uav lon-lat and center_error_m are used only after selection for evaluation.",
            "This is a sequence-level smoother over PHOG top-N candidates, not a full-map re-search.",
        ],
    }

    summary.update(metric_summary(compare_df, "phog_top1_error_m", "phog_top1"))
    summary.update(metric_summary(compare_df, "temporal_error_m", "temporal"))
    summary.update(metric_summary(compare_df, "oracle_topn_error_m", "oracle_topn"))

    trans = compare_df["selected_transition_distance_m"].astype(float)
    summary.update(
        {
            "selected_mean_transition_m": float(trans.dropna().mean()),
            "selected_median_transition_m": float(trans.dropna().median()),
            "selected_max_transition_m": float(trans.dropna().max()),
            "selected_mean_phog_rank": float(compare_df["temporal_selected_phog_rank"].mean()),
            "selected_median_phog_rank": float(compare_df["temporal_selected_phog_rank"].median()),
        }
    )

    summary_json = OUT_REPORT_DIR / f"s4c3_{args.sequence}_{slug}_summary.json"

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(json_safe(summary), f, indent=2)

    print("S4C.3 complete")
    print("--------------")
    print(f"Selected path CSV: {selected_csv}")
    print(f"Comparison CSV:    {compare_csv}")
    print(f"Candidate costs:   {all_candidates_csv}")
    print(f"Summary JSON:      {summary_json}")
    print(f"Figures:           {OUT_FIG_DIR}")
    print("")
    print("Compact result")
    print("--------------")
    compact = pd.DataFrame(
        [
            {
                "method": "PHOG top-1",
                "top1_under_20m_rate": summary["phog_top1_under_20m_rate"],
                "top1_under_40m_rate": summary["phog_top1_under_40m_rate"],
                "top1_under_60m_rate": summary["phog_top1_under_60m_rate"],
                "median_error_m": summary["phog_top1_median_error_m"],
                "mean_error_m": summary["phog_top1_mean_error_m"],
            },
            {
                "method": "Temporal selected",
                "top1_under_20m_rate": summary["temporal_under_20m_rate"],
                "top1_under_40m_rate": summary["temporal_under_40m_rate"],
                "top1_under_60m_rate": summary["temporal_under_60m_rate"],
                "median_error_m": summary["temporal_median_error_m"],
                "mean_error_m": summary["temporal_mean_error_m"],
            },
            {
                "method": "Oracle best top-N",
                "top1_under_20m_rate": summary["oracle_topn_under_20m_rate"],
                "top1_under_40m_rate": summary["oracle_topn_under_40m_rate"],
                "top1_under_60m_rate": summary["oracle_topn_under_60m_rate"],
                "median_error_m": summary["oracle_topn_median_error_m"],
                "mean_error_m": summary["oracle_topn_mean_error_m"],
            },
        ]
    )
    print(compact.to_string(index=False))
    print("")
    print("Path diagnostics")
    print("----------------")
    print(f"Mean selected PHOG rank:   {summary['selected_mean_phog_rank']:.2f}")
    print(f"Median selected PHOG rank: {summary['selected_median_phog_rank']:.2f}")
    print(f"Median transition:         {summary['selected_median_transition_m']:.1f} m")
    print(f"Max transition:            {summary['selected_max_transition_m']:.1f} m")


if __name__ == "__main__":
    main()
