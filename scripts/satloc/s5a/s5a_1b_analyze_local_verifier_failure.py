#!/usr/bin/env python3
"""
S5A.1B — Analyze local verifier failure inside PHOG top-K

Purpose
-------
S5A.1 proved the local-verifier interface works, but the AKAZE/OpenCV verifier
performed worse than PHOG. This script diagnoses why by comparing PHOG top-1,
local-verifier top-1, and oracle-best candidate inside the PHOG top-K pool.

Locked rule
-----------
All columns containing reference error are used only for post-ranking evaluation.
This script does not perform retrieval or ranking.

Code Used:
export PYTHONPATH=$PWD/src

python scripts/satloc/s5a/s5a_1b_analyze_local_verifier_failure.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="S5A.1B local verifier failure analysis")
    parser.add_argument(
        "--query-summary",
        type=Path,
        default=Path("outputs/satloc/metadata/s5a_learned_local_verifier/s5a1_local_verifier_query_summary.csv"),
        help="S5A.1 query summary CSV.",
    )
    parser.add_argument(
        "--candidate-scores",
        type=Path,
        default=Path("outputs/satloc/metadata/s5a_learned_local_verifier/s5a1_local_verifier_candidate_scores.csv"),
        help="S5A.1 candidate-level local verifier scores CSV.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/satloc"),
        help="Base output directory.",
    )
    parser.add_argument("--threshold-m", type=float, default=40.0)
    return parser.parse_args()


def ensure_dirs(out_dir: Path) -> Dict[str, Path]:
    paths = {
        "metadata": out_dir / "metadata" / "s5a_learned_local_verifier",
        "reports": out_dir / "reports" / "s5a_learned_local_verifier",
        "figures": out_dir / "figures" / "s5a_learned_local_verifier",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def read_csv_checked(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")
    return pd.read_csv(path)


def as_bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s.fillna(False)
    text = s.fillna(False).astype(str).str.lower().str.strip()
    return text.isin(["true", "1", "yes", "y", "t"])


def numeric(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series([np.nan] * len(df), index=df.index)
    return pd.to_numeric(df[col], errors="coerce")


def safe_float(value: Any) -> Optional[float]:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def median_or_none(s: pd.Series) -> Optional[float]:
    vals = pd.to_numeric(s, errors="coerce").dropna()
    if len(vals) == 0:
        return None
    return float(vals.median())


def classify_query(row: pd.Series) -> str:
    phog_hit = bool(row["phog_hit"])
    local_hit = bool(row["local_hit"])
    oracle_hit = bool(row["oracle_hit"])

    if phog_hit and local_hit:
        return "both_success"
    if (not phog_hit) and local_hit:
        return "local_rescue"
    if phog_hit and (not local_hit):
        return "local_destroyed_phog"
    if (not local_hit) and oracle_hit:
        return "correct_available_but_local_missed"
    if not oracle_hit:
        return "candidate_pool_not_good_enough"
    return "both_fail"


def build_query_diagnostics(query_df: pd.DataFrame, threshold_m: float) -> pd.DataFrame:
    q = query_df.copy()
    q["token_str"] = q["token"].astype(str)

    q["phog_error_m"] = numeric(q, "phog_top1_error_m")
    q["local_error_m"] = numeric(q, "local_top1_error_m")
    q["oracle_error_m"] = numeric(q, "oracle_best_topk_error_m")

    if "phog_top1_hit_le_threshold" in q.columns:
        q["phog_hit"] = as_bool_series(q["phog_top1_hit_le_threshold"])
    else:
        q["phog_hit"] = q["phog_error_m"] <= threshold_m

    if "local_top1_hit_le_threshold" in q.columns:
        q["local_hit"] = as_bool_series(q["local_top1_hit_le_threshold"])
    else:
        q["local_hit"] = q["local_error_m"] <= threshold_m

    if "oracle_hit_le_threshold" in q.columns:
        q["oracle_hit"] = as_bool_series(q["oracle_hit_le_threshold"])
    else:
        q["oracle_hit"] = q["oracle_error_m"] <= threshold_m

    q["local_minus_phog_error_m"] = q["local_error_m"] - q["phog_error_m"]
    q["local_minus_oracle_error_m"] = q["local_error_m"] - q["oracle_error_m"]
    q["phog_minus_oracle_error_m"] = q["phog_error_m"] - q["oracle_error_m"]

    q["local_better_than_phog"] = q["local_error_m"] < q["phog_error_m"]
    q["local_worse_than_phog"] = q["local_error_m"] > q["phog_error_m"]
    q["local_decision_class"] = q.apply(classify_query, axis=1)

    return q


def build_oracle_candidate_diagnostics(candidate_df: pd.DataFrame) -> pd.DataFrame:
    c = candidate_df.copy()
    c["token_str"] = c["token"].astype(str)

    c["eval_error_m_num"] = numeric(c, "eval_error_m")
    c["local_score_num"] = numeric(c, "local_score")
    c["local_verifier_rank_num"] = numeric(c, "local_verifier_rank")
    c["candidate_pool_rank_num"] = numeric(c, "candidate_pool_rank")
    c["good_matches_num"] = numeric(c, "good_matches")
    c["ransac_inliers_num"] = numeric(c, "ransac_inliers")
    c["inlier_ratio_num"] = numeric(c, "inlier_ratio")
    c["phog_score_num"] = numeric(c, "phog_score")

    rows: List[Dict[str, Any]] = []

    for (token, group), g in c.groupby(["token_str", "failure_group"], dropna=False):
        g_valid = g.dropna(subset=["eval_error_m_num"]).copy()
        if len(g_valid) == 0:
            continue

        oracle = g_valid.sort_values("eval_error_m_num", kind="mergesort").iloc[0]

        local_top = g.sort_values(
            ["local_score_num", "ransac_inliers_num", "good_matches_num"],
            ascending=[False, False, False],
            kind="mergesort",
        ).iloc[0]

        phog_top = g.sort_values("candidate_pool_rank_num", kind="mergesort").iloc[0]

        oracle_score = safe_float(oracle.get("local_score_num"))
        local_top_score = safe_float(local_top.get("local_score_num"))

        rows.append(
            {
                "token": token,
                "failure_group": group,

                "oracle_tile_id": oracle.get("tile_id", ""),
                "oracle_error_m": safe_float(oracle.get("eval_error_m_num")),
                "oracle_candidate_pool_rank": safe_float(oracle.get("candidate_pool_rank_num")),
                "oracle_local_verifier_rank": safe_float(oracle.get("local_verifier_rank_num")),
                "oracle_local_score": oracle_score,
                "oracle_good_matches": safe_float(oracle.get("good_matches_num")),
                "oracle_ransac_inliers": safe_float(oracle.get("ransac_inliers_num")),
                "oracle_inlier_ratio": safe_float(oracle.get("inlier_ratio_num")),
                "oracle_phog_score": safe_float(oracle.get("phog_score_num")),

                "local_top1_tile_id": local_top.get("tile_id", ""),
                "local_top1_error_m": safe_float(local_top.get("eval_error_m_num")),
                "local_top1_score": local_top_score,
                "local_top1_candidate_pool_rank": safe_float(local_top.get("candidate_pool_rank_num")),
                "local_top1_good_matches": safe_float(local_top.get("good_matches_num")),
                "local_top1_ransac_inliers": safe_float(local_top.get("ransac_inliers_num")),
                "local_top1_inlier_ratio": safe_float(local_top.get("inlier_ratio_num")),

                "phog_top1_tile_id": phog_top.get("tile_id", ""),
                "phog_top1_error_m": safe_float(phog_top.get("eval_error_m_num")),
                "phog_top1_local_score": safe_float(phog_top.get("local_score_num")),

                "score_gap_local_top_minus_oracle": None
                if oracle_score is None or local_top_score is None
                else float(local_top_score - oracle_score),
            }
        )

    return pd.DataFrame(rows)


def build_group_diagnostics(q: pd.DataFrame, oracle_diag: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for group, g in q.groupby("failure_group", dropna=False):
        row: Dict[str, Any] = {
            "failure_group": str(group),
            "count": int(len(g)),

            "phog_hit_rate": float(g["phog_hit"].mean()),
            "local_hit_rate": float(g["local_hit"].mean()),
            "oracle_topk_hit_rate": float(g["oracle_hit"].mean()),

            "phog_median_error_m": median_or_none(g["phog_error_m"]),
            "local_median_error_m": median_or_none(g["local_error_m"]),
            "oracle_median_error_m": median_or_none(g["oracle_error_m"]),

            "median_local_minus_phog_error_m": median_or_none(g["local_minus_phog_error_m"]),
            "median_local_minus_oracle_error_m": median_or_none(g["local_minus_oracle_error_m"]),

            "local_better_than_phog_rate": float(g["local_better_than_phog"].mean()),
            "local_worse_than_phog_rate": float(g["local_worse_than_phog"].mean()),

            "local_rescue_count": int((g["local_decision_class"] == "local_rescue").sum()),
            "local_destroyed_phog_count": int((g["local_decision_class"] == "local_destroyed_phog").sum()),
            "correct_available_but_local_missed_count": int(
                (g["local_decision_class"] == "correct_available_but_local_missed").sum()
            ),
        }

        og = oracle_diag[oracle_diag["failure_group"].astype(str) == str(group)]
        if len(og):
            row["oracle_local_rank_median"] = median_or_none(og["oracle_local_verifier_rank"])
            row["oracle_candidate_pool_rank_median"] = median_or_none(og["oracle_candidate_pool_rank"])
            row["score_gap_local_top_minus_oracle_median"] = median_or_none(
                og["score_gap_local_top_minus_oracle"]
            )
        else:
            row["oracle_local_rank_median"] = None
            row["oracle_candidate_pool_rank_median"] = None
            row["score_gap_local_top_minus_oracle_median"] = None

        rows.append(row)

    return pd.DataFrame(rows).sort_values("failure_group")


def plot_group_hit_rates(group_df: pd.DataFrame, out_path: Path) -> None:
    if len(group_df) == 0:
        return

    x = np.arange(len(group_df))
    width = 0.26

    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.bar(x - width, group_df["phog_hit_rate"], width, label="PHOG top1")
    ax.bar(x, group_df["local_hit_rate"], width, label="OpenCV local verifier")
    ax.bar(x + width, group_df["oracle_topk_hit_rate"], width, label="Oracle in PHOG topK")

    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Hit rate <= threshold")
    ax.set_title("S5A.1B hit-rate comparison by failure group")
    ax.set_xticks(x)
    ax.set_xticklabels(group_df["failure_group"].tolist(), rotation=35, ha="right")
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_error_delta(q: pd.DataFrame, out_path: Path) -> None:
    if len(q) == 0:
        return

    groups = sorted(q["failure_group"].dropna().astype(str).unique().tolist())
    data = [
        q[q["failure_group"].astype(str) == g]["local_minus_phog_error_m"].dropna().values
        for g in groups
    ]

    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.boxplot(data, labels=groups, showfliers=False)
    ax.axhline(0.0, linestyle="--", linewidth=1)
    ax.set_ylabel("Local top1 error - PHOG top1 error [m]")
    ax.set_title("S5A.1B error delta by failure group; above zero means local verifier is worse")
    ax.tick_params(axis="x", rotation=35)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_oracle_local_rank(oracle_diag: pd.DataFrame, out_path: Path) -> None:
    if len(oracle_diag) == 0:
        return

    groups = sorted(oracle_diag["failure_group"].dropna().astype(str).unique().tolist())
    data = [
        pd.to_numeric(
            oracle_diag[oracle_diag["failure_group"].astype(str) == g]["oracle_local_verifier_rank"],
            errors="coerce",
        ).dropna().values
        for g in groups
    ]

    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.boxplot(data, labels=groups, showfliers=False)
    ax.axhline(1.0, linestyle="--", linewidth=1)
    ax.set_ylabel("Local-verifier rank of oracle-best PHOG topK candidate")
    ax.set_title("S5A.1B: where does the correct/nearest candidate rank under local score?")
    ax.tick_params(axis="x", rotation=35)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_score_vs_error(candidate_df: pd.DataFrame, out_path: Path) -> None:
    if len(candidate_df) == 0:
        return

    c = candidate_df.copy()
    c["eval_error_m_num"] = numeric(c, "eval_error_m")
    c["local_score_num"] = numeric(c, "local_score")
    c = c.dropna(subset=["eval_error_m_num", "local_score_num"])

    if len(c) == 0:
        return

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.scatter(c["local_score_num"], c["eval_error_m_num"], s=8, alpha=0.35)
    ax.set_xlabel("Local verifier score")
    ax.set_ylabel("Evaluation error [m]")
    ax.set_title("S5A.1B local score vs post-ranking error")
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def build_summary(
    q: pd.DataFrame,
    group_df: pd.DataFrame,
    oracle_diag: pd.DataFrame,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    decision_counts = q["local_decision_class"].value_counts(dropna=False).to_dict()

    summary: Dict[str, Any] = {
        "stage": "S5A.1B_local_verifier_failure_analysis",
        "threshold_m": args.threshold_m,
        "locked_rule": "reference coordinates/errors are evaluation-only; this script only diagnoses completed rankings",

        "num_queries": int(len(q)),
        "num_groups": int(q["failure_group"].nunique(dropna=True)),

        "phog_hit_rate": float(q["phog_hit"].mean()) if len(q) else 0.0,
        "local_hit_rate": float(q["local_hit"].mean()) if len(q) else 0.0,
        "oracle_topk_hit_rate": float(q["oracle_hit"].mean()) if len(q) else 0.0,

        "phog_median_error_m": median_or_none(q["phog_error_m"]),
        "local_median_error_m": median_or_none(q["local_error_m"]),
        "oracle_median_error_m": median_or_none(q["oracle_error_m"]),

        "median_local_minus_phog_error_m": median_or_none(q["local_minus_phog_error_m"]),
        "local_better_than_phog_rate": float(q["local_better_than_phog"].mean()) if len(q) else 0.0,
        "local_worse_than_phog_rate": float(q["local_worse_than_phog"].mean()) if len(q) else 0.0,

        "decision_counts": decision_counts,
        "by_failure_group": group_df.to_dict(orient="records"),
    }

    if len(oracle_diag):
        summary["oracle_local_rank_median"] = median_or_none(oracle_diag["oracle_local_verifier_rank"])
        summary["oracle_score_gap_top_minus_oracle_median"] = median_or_none(
            oracle_diag["score_gap_local_top_minus_oracle"]
        )

    return summary


def main() -> None:
    args = parse_args()
    dirs = ensure_dirs(args.out_dir)

    query_df = read_csv_checked(args.query_summary, "S5A.1 query summary")
    candidate_df = read_csv_checked(args.candidate_scores, "S5A.1 candidate scores")

    q_diag = build_query_diagnostics(query_df, args.threshold_m)
    oracle_diag = build_oracle_candidate_diagnostics(candidate_df)
    group_diag = build_group_diagnostics(q_diag, oracle_diag)

    query_out = dirs["metadata"] / "s5a1b_query_diagnostics.csv"
    oracle_out = dirs["metadata"] / "s5a1b_oracle_candidate_diagnostics.csv"
    group_out = dirs["metadata"] / "s5a1b_group_diagnostics.csv"
    summary_out = dirs["reports"] / "s5a1b_local_verifier_failure_analysis_summary.json"

    fig_hit = dirs["figures"] / "s5a1b_group_hit_rate_comparison.png"
    fig_delta = dirs["figures"] / "s5a1b_error_delta_by_group.png"
    fig_rank = dirs["figures"] / "s5a1b_oracle_local_rank_by_group.png"
    fig_scatter = dirs["figures"] / "s5a1b_local_score_vs_error.png"

    q_diag.to_csv(query_out, index=False)
    oracle_diag.to_csv(oracle_out, index=False)
    group_diag.to_csv(group_out, index=False)

    plot_group_hit_rates(group_diag, fig_hit)
    plot_error_delta(q_diag, fig_delta)
    plot_oracle_local_rank(oracle_diag, fig_rank)
    plot_score_vs_error(candidate_df, fig_scatter)

    summary = build_summary(q_diag, group_diag, oracle_diag, args)
    summary.update(
        {
            "query_diagnostics_csv": str(query_out),
            "oracle_candidate_diagnostics_csv": str(oracle_out),
            "group_diagnostics_csv": str(group_out),
            "group_hit_rate_figure": str(fig_hit),
            "error_delta_figure": str(fig_delta),
            "oracle_local_rank_figure": str(fig_rank),
            "score_vs_error_figure": str(fig_scatter),
        }
    )

    with open(summary_out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("S5A.1B local verifier failure analysis complete")
    print("------------------------------------------------")
    print(f"Queries analyzed:       {len(q_diag)}")
    print(f"Candidate rows:         {len(candidate_df)}")
    print(f"PHOG hit <=thr:         {summary['phog_hit_rate']:.3f}")
    print(f"Local hit <=thr:        {summary['local_hit_rate']:.3f}")
    print(f"Oracle topK <=thr:      {summary['oracle_topk_hit_rate']:.3f}")
    print(f"PHOG median error:      {summary['phog_median_error_m']:.3f} m")
    print(f"Local median error:     {summary['local_median_error_m']:.3f} m")
    print(f"Oracle median error:    {summary['oracle_median_error_m']:.3f} m")
    print(f"Median local-PHOG:      {summary['median_local_minus_phog_error_m']:.3f} m")
    print(f"Local better rate:      {summary['local_better_than_phog_rate']:.3f}")
    print(f"Local worse rate:       {summary['local_worse_than_phog_rate']:.3f}")
    print(f"Decision counts:        {summary['decision_counts']}")
    print(f"Query diagnostics CSV:  {query_out}")
    print(f"Oracle diagnostics CSV: {oracle_out}")
    print(f"Group diagnostics CSV:  {group_out}")
    print(f"Summary JSON:           {summary_out}")
    print(f"Figures:                {fig_hit}")
    print(f"                        {fig_delta}")
    print(f"                        {fig_rank}")
    print(f"                        {fig_scatter}")
    print()
#    print("Locked rule: reference/error columns were used only for post-ranking diagnosis.")


if __name__ == "__main__":
    main()
