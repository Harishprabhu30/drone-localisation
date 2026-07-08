#!/usr/bin/env python3
"""
S4C.4D — Confidence-gated reranker.

Purpose:
  S4C.4C showed luma-LSD improves PHOG top-50 reranking, but LSD can still
  destroy already-good PHOG cases. This script tests unsupervised confidence
  gates that choose between PHOG top-1 and LSD top-1.

Input:
  Existing S4C.4C raw LSD score CSVs:
    s4c4c_tokenXXXX_luma_lsd_raw_lsd_scores_top50.csv

Important:
  - center_error_m is used only after selection for evaluation.
  - No reference coordinate is used in scoring/gating.
  - Gates use only PHOG score margin, LSD score gap, shift-boundary flag,
    and rank/score diagnostics.

Code Used:
export PYTHONPATH=$PWD/src

python scripts/satloc/s4c/s4c_4d_confidence_gated_reranker.py \
  --sequence traj01 \
  --tokens all \
  --max-good-shift-px 56 \
  --phog-lsd-agree-rank 5
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUT_ROOT = Path("outputs/satloc")

DEFAULT_S4C4C_DIR = OUT_ROOT / (
    "metadata/s4c_macrocontour_phog_chamfer/"
    "s4c4_vector_skeleton/s4c4c_luma_lsd_rerank"
)

OUT_META_DIR = OUT_ROOT / (
    "metadata/s4c_macrocontour_phog_chamfer/"
    "s4c4_vector_skeleton/s4c4d_confidence_gated_rerank"
)

OUT_REPORT_DIR = OUT_ROOT / (
    "reports/s4c_macrocontour_phog_chamfer/"
    "s4c4_vector_skeleton/s4c4d_confidence_gated_rerank"
)

OUT_FIG_DIR = OUT_ROOT / (
    "figures/s4c_macrocontour_phog_chamfer/"
    "s4c4_vector_skeleton/s4c4d_confidence_gated_rerank"
)


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


def parse_tokens(text: str, raw_dir: Path) -> list[int]:
    text = text.strip()

    if text.lower() == "all":
        tokens: set[int] = set()
        for p in raw_dir.glob("s4c4c_token*_luma_lsd_raw_lsd_scores_top*.csv"):
            m = re.search(r"s4c4c_token(\d+)_", p.name)
            if m:
                tokens.add(int(m.group(1)))
        if not tokens:
            raise FileNotFoundError(f"No S4C.4C raw CSVs found in {raw_dir}")
        return sorted(tokens)

    return sorted([int(x.strip()) for x in text.split(",") if x.strip()])


def find_raw_csv(token: int, raw_dir: Path) -> Path:
    pattern = f"s4c4c_token{token:04d}_luma_lsd_raw_lsd_scores_top*.csv"
    matches = sorted(raw_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    if not matches:
        raise FileNotFoundError(f"No raw S4C.4C CSV for token {token}: {raw_dir / pattern}")
    return matches[0]


def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "phog_rank" not in out.columns:
        if "rank" in out.columns:
            out["phog_rank"] = out["rank"].astype(int)
        else:
            out["phog_rank"] = np.arange(1, len(out) + 1)

    if "lsd_rank" not in out.columns:
        out["lsd_rank"] = out["lsd_best_score"].rank(method="min", ascending=True).astype(int)

    if "lsd_at_shift_boundary" not in out.columns:
        out["lsd_at_shift_boundary"] = False

    if "phog_score" not in out.columns:
        out["phog_score"] = np.nan

    for c in ["center_error_m", "lsd_best_score", "lsd_shift_mag_px", "lsd_basin_area"]:
        if c not in out.columns:
            out[c] = np.nan

    return out


def frame_features(token: int, df: pd.DataFrame, raw_csv: Path) -> dict[str, Any]:
    phog = df.sort_values("phog_rank").reset_index(drop=True)
    lsd = df.sort_values(["lsd_rank", "phog_rank"]).reset_index(drop=True)

    p1 = phog.iloc[0]
    p2 = phog.iloc[1] if len(phog) > 1 else phog.iloc[0]

    l1 = lsd.iloc[0]
    l2 = lsd.iloc[1] if len(lsd) > 1 else lsd.iloc[0]

    phog_score1 = safe_float(p1.get("phog_score"))
    phog_score2 = safe_float(p2.get("phog_score"))
    phog_margin = (
        phog_score1 - phog_score2
        if np.isfinite(phog_score1) and np.isfinite(phog_score2)
        else np.nan
    )

    lsd_score1 = safe_float(l1.get("lsd_best_score"))
    lsd_score2 = safe_float(l2.get("lsd_best_score"))
    lsd_gap = (
        lsd_score2 - lsd_score1
        if np.isfinite(lsd_score1) and np.isfinite(lsd_score2)
        else np.nan
    )

    phog_top1_lsd_rank = int(p1["lsd_rank"])
    lsd_top1_phog_rank = int(l1["phog_rank"])

    return {
        "token": int(token),
        "raw_csv": str(raw_csv),

        "phog_margin": phog_margin,
        "phog_score1": phog_score1,
        "phog_score2": phog_score2,
        "phog_top1_lsd_rank": phog_top1_lsd_rank,

        "lsd_gap": lsd_gap,
        "lsd_score1": lsd_score1,
        "lsd_score2": lsd_score2,
        "lsd_top1_phog_rank": lsd_top1_phog_rank,
        "lsd_top1_boundary": bool(l1["lsd_at_shift_boundary"]),
        "lsd_top1_shift_mag_px": safe_float(l1.get("lsd_shift_mag_px")),
        "lsd_top1_basin_area": safe_float(l1.get("lsd_basin_area")),
    }


def select_row(
    profile: str,
    df: pd.DataFrame,
    feat: dict[str, Any],
    thresholds: dict[str, float],
    args: argparse.Namespace,
) -> tuple[pd.Series, str]:
    phog = df.sort_values("phog_rank").reset_index(drop=True)
    lsd = df.sort_values(["lsd_rank", "phog_rank"]).reset_index(drop=True)

    phog_top = phog.iloc[0]
    lsd_top = lsd.iloc[0]

    phog_margin = safe_float(feat["phog_margin"])
    lsd_gap = safe_float(feat["lsd_gap"])
    lsd_boundary = bool(feat["lsd_top1_boundary"])
    lsd_shift = safe_float(feat["lsd_top1_shift_mag_px"])
    lsd_score1 = safe_float(feat["lsd_score1"])
    phog_top1_lsd_rank = int(feat["phog_top1_lsd_rank"])

    phog_anchor = (
        np.isfinite(phog_margin)
        and phog_margin >= thresholds["phog_margin_q75"]
    )

    lsd_confident = (
        np.isfinite(lsd_gap)
        and lsd_gap >= thresholds["lsd_gap_q50"]
        and not lsd_boundary
        and (not np.isfinite(lsd_shift) or lsd_shift <= args.max_good_shift_px)
    )

    lsd_very_confident = (
        np.isfinite(lsd_gap)
        and lsd_gap >= thresholds["lsd_gap_q75"]
        and not lsd_boundary
        and np.isfinite(lsd_score1)
        and lsd_score1 <= thresholds["lsd_score_q50"]
        and (not np.isfinite(lsd_shift) or lsd_shift <= args.max_good_shift_px)
    )

    phog_lsd_agree = phog_top1_lsd_rank <= args.phog_lsd_agree_rank

    if profile == "phog_only":
        return phog_top, "phog_only"

    if profile == "lsd_only":
        return lsd_top, "lsd_only"

    if profile == "gate_phog_anchor_else_lsd":
        if phog_anchor:
            return phog_top, "phog_anchor"
        return lsd_top, "fallback_lsd"

    if profile == "gate_lsd_conf_else_phog":
        if lsd_confident:
            return lsd_top, "lsd_confident"
        return phog_top, "fallback_phog"

    if profile == "gate_lsd_very_conf_else_phog":
        if lsd_very_confident:
            return lsd_top, "lsd_very_confident"
        return phog_top, "fallback_phog"

    if profile == "gate_agreement_protect":
        # Keep PHOG when PHOG and LSD evidence agree enough.
        if phog_lsd_agree or phog_anchor:
            return phog_top, "phog_protected"
        if lsd_confident:
            return lsd_top, "lsd_confident"
        return phog_top, "fallback_phog"

    if profile == "gate_anchor_then_lsd_very":
        if phog_anchor:
            return phog_top, "phog_anchor"
        if lsd_very_confident:
            return lsd_top, "lsd_very_confident"
        return phog_top, "fallback_phog"

    raise ValueError(f"Unknown profile: {profile}")


def evaluate_selected(
    token: int,
    profile: str,
    row: pd.Series,
    reason: str,
    feat: dict[str, Any],
) -> dict[str, Any]:
    err = safe_float(row.get("center_error_m"))

    return {
        "token": int(token),
        "profile": profile,
        "selection_reason": reason,

        "tile_id": int(row["tile_id"]),
        "center_error_m": err,
        "phog_rank": int(row["phog_rank"]),
        "lsd_rank": int(row["lsd_rank"]),
        "phog_score": safe_float(row.get("phog_score")),
        "lsd_best_score": safe_float(row.get("lsd_best_score")),
        "lsd_gap": safe_float(feat.get("lsd_gap")),
        "phog_margin": safe_float(feat.get("phog_margin")),
        "lsd_shift_mag_px": safe_float(row.get("lsd_shift_mag_px")),
        "lsd_at_shift_boundary": bool(row.get("lsd_at_shift_boundary", False)),

        "under_20m": bool(np.isfinite(err) and err <= 20.0),
        "under_40m": bool(np.isfinite(err) and err <= 40.0),
        "under_60m": bool(np.isfinite(err) and err <= 60.0),
        "under_100m": bool(np.isfinite(err) and err <= 100.0),
    }


def summarize_aggregate(selected_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for profile, group in selected_df.groupby("profile"):
        rows.append(
            {
                "profile": profile,
                "num_queries": int(len(group)),
                "top1_under_20m_rate": float(group["under_20m"].mean()),
                "top1_under_40m_rate": float(group["under_40m"].mean()),
                "top1_under_60m_rate": float(group["under_60m"].mean()),
                "top1_under_100m_rate": float(group["under_100m"].mean()),
                "median_error_m": float(group["center_error_m"].median()),
                "mean_error_m": float(group["center_error_m"].mean()),
                "median_phog_rank": float(group["phog_rank"].median()),
                "median_lsd_rank": float(group["lsd_rank"].median()),
                "mean_lsd_score": float(group["lsd_best_score"].mean()),
            }
        )

    out = pd.DataFrame(rows)

    if len(out) > 0:
        out = out.sort_values(
            ["top1_under_40m_rate", "median_error_m"],
            ascending=[False, True],
        )

    return out


def render_aggregate_plot(agg: pd.DataFrame, out_path: Path) -> None:
    if len(agg) == 0:
        return

    labels = agg["profile"].astype(str).tolist()
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 5.2))
    ax.bar(x - width / 2, agg["top1_under_20m_rate"], width, label="<=20 m")
    ax.bar(x + width / 2, agg["top1_under_40m_rate"], width, label="<=40 m")

    ax.set_ylim(0, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("success rate")
    ax.set_title("S4C.4D confidence-gated reranker")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--sequence", default="traj01")
    parser.add_argument("--tokens", default="all")
    parser.add_argument("--s4c4c-dir", default=str(DEFAULT_S4C4C_DIR))

    parser.add_argument("--max-good-shift-px", type=float, default=56.0)
    parser.add_argument("--phog-lsd-agree-rank", type=int, default=5)

    args = parser.parse_args()

    raw_dir = Path(args.s4c4c_dir)
    if not raw_dir.exists():
        raise FileNotFoundError(f"Missing S4C.4C dir: {raw_dir}")

    OUT_META_DIR.mkdir(parents=True, exist_ok=True)
    OUT_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_FIG_DIR.mkdir(parents=True, exist_ok=True)

    tokens = parse_tokens(args.tokens, raw_dir)

    profiles = [
        "phog_only",
        "lsd_only",
        "gate_phog_anchor_else_lsd",
        "gate_lsd_conf_else_phog",
        "gate_lsd_very_conf_else_phog",
        "gate_agreement_protect",
        "gate_anchor_then_lsd_very",
    ]

    print("S4C.4D confidence-gated reranker")
    print("--------------------------------")
    print(f"Sequence: {args.sequence}")
    print(f"Tokens:   {len(tokens)}")
    print(f"Input:    {raw_dir}")
    print("")

    token_dfs: dict[int, pd.DataFrame] = {}
    feature_rows: list[dict[str, Any]] = []

    for token in tokens:
        try:
            raw_csv = find_raw_csv(token, raw_dir)
            df = normalize_df(pd.read_csv(raw_csv))
            token_dfs[token] = df
            feature_rows.append(frame_features(token, df, raw_csv))
        except Exception as exc:
            print(f"[WARN] token {token}: {exc}")

    feat_df = pd.DataFrame(feature_rows)

    if len(feat_df) == 0:
        raise RuntimeError("No feature rows loaded.")

    finite_margin = feat_df["phog_margin"].replace([np.inf, -np.inf], np.nan).dropna()
    finite_lsd_gap = feat_df["lsd_gap"].replace([np.inf, -np.inf], np.nan).dropna()
    finite_lsd_score = feat_df["lsd_score1"].replace([np.inf, -np.inf], np.nan).dropna()

    thresholds = {
        "phog_margin_q50": float(finite_margin.quantile(0.50)) if len(finite_margin) else float("inf"),
        "phog_margin_q75": float(finite_margin.quantile(0.75)) if len(finite_margin) else float("inf"),
        "lsd_gap_q50": float(finite_lsd_gap.quantile(0.50)) if len(finite_lsd_gap) else float("inf"),
        "lsd_gap_q75": float(finite_lsd_gap.quantile(0.75)) if len(finite_lsd_gap) else float("inf"),
        "lsd_score_q50": float(finite_lsd_score.quantile(0.50)) if len(finite_lsd_score) else float("-inf"),
    }

    selected_rows: list[dict[str, Any]] = []

    for feat in feature_rows:
        token = int(feat["token"])
        df = token_dfs[token]

        for profile in profiles:
            row, reason = select_row(profile, df, feat, thresholds, args)
            selected_rows.append(evaluate_selected(token, profile, row, reason, feat))

    selected_df = pd.DataFrame(selected_rows)
    agg_df = summarize_aggregate(selected_df)

    selected_csv = OUT_META_DIR / f"s4c4d_{args.sequence}_selected_by_profile.csv"
    features_csv = OUT_META_DIR / f"s4c4d_{args.sequence}_frame_confidence_features.csv"
    aggregate_csv = OUT_META_DIR / f"s4c4d_{args.sequence}_aggregate_by_profile.csv"
    summary_json = OUT_REPORT_DIR / f"s4c4d_{args.sequence}_summary.json"
    plot_path = OUT_FIG_DIR / f"s4c4d_{args.sequence}_profile_success_rates.png"

    selected_df.to_csv(selected_csv, index=False)
    feat_df.to_csv(features_csv, index=False)
    agg_df.to_csv(aggregate_csv, index=False)

    render_aggregate_plot(agg_df, plot_path)

    output = {
        "stage": "S4C.4D_confidence_gated_reranker",
        "sequence": args.sequence,
        "tokens": tokens,
        "settings": vars(args),
        "thresholds": thresholds,
        "selected_csv": str(selected_csv),
        "features_csv": str(features_csv),
        "aggregate_csv": str(aggregate_csv),
        "plot_path": str(plot_path),
        "notes": [
            "No reference coordinate is used in scoring/gating.",
            "center_error_m is used only for post-selection evaluation.",
            "PHOG margin and LSD gap thresholds are computed from unsupervised score distributions.",
        ],
        "aggregate_by_profile": agg_df.to_dict(orient="records"),
    }

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(json_safe(output), f, indent=2)

    print("S4C.4D complete")
    print("----------------")
    print(f"Selected CSV:  {selected_csv}")
    print(f"Features CSV:  {features_csv}")
    print(f"Aggregate CSV: {aggregate_csv}")
    print(f"Summary JSON:  {summary_json}")
    print(f"Plot:          {plot_path}")
    print("")
    print("Thresholds")
    print("----------")
    for k, v in thresholds.items():
        print(f"{k}: {v}")
    print("")
    print("Aggregate profile comparison")
    print("----------------------------")
    print(agg_df.to_string(index=False))


if __name__ == "__main__":
    main()
