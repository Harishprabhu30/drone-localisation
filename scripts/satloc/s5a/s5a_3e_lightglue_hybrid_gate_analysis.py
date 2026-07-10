#!/usr/bin/env python3
"""
S5A.3E — LightGlue hybrid / PHOG-protection gate analysis

Purpose
-------
Use already-computed S5A.3 LightGlue candidate scores and test whether simple
unsupervised ranking policies can reduce:
1. LightGlue destroying PHOG successes
2. LightGlue missing oracle candidates that are rank-2 or rank-7

Important:
- This script does not recompute matching.
- Ranking policies use only non-reference columns:
  lightglue_score, candidate_pool_rank, LightGlue rank, inliers, coverage.
- eval_error_m is used only after ranking for evaluation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--candidate-scores",
        type=Path,
        default=Path("outputs/satloc/metadata/s5a_learned_local_verifier/s5a3_lightglue_candidate_scores_top50_all73.csv"),
    )
    p.add_argument(
        "--query-summary",
        type=Path,
        default=Path("outputs/satloc/metadata/s5a_learned_local_verifier/s5a3_lightglue_query_summary_top50_all73.csv"),
    )
    p.add_argument("--run-name", type=str, default="top50_all73")
    p.add_argument("--threshold-m", type=float, default=40.0)
    p.add_argument("--out-base", type=Path, default=Path("outputs/satloc"))
    return p.parse_args()


def ensure_dirs(base: Path):
    d = {
        "metadata": base / "metadata" / "s5a_learned_local_verifier",
        "reports": base / "reports" / "s5a_learned_local_verifier",
        "figures": base / "figures" / "s5a_learned_local_verifier",
    }
    for p in d.values():
        p.mkdir(parents=True, exist_ok=True)
    return d


def num(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series([np.nan] * len(df), index=df.index)
    return pd.to_numeric(df[col], errors="coerce")


def safe_float(x: Any):
    try:
        if x is None or pd.isna(x):
            return None
        return float(x)
    except Exception:
        return None


def as_bool(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s.fillna(False)
    return s.fillna(False).astype(str).str.lower().isin(["true", "1", "yes"])


def prep_candidates(c: pd.DataFrame) -> pd.DataFrame:
    c = c.copy()
    c["token_str"] = c["token"].astype(str)

    c["eval_error_num"] = num(c, "eval_error_m")
    c["rank_num"] = num(c, "candidate_pool_rank")
    c["lg_rank_num"] = num(c, "lightglue_rank")
    c["lg_score_num"] = num(c, "lightglue_score")
    c["lg_inliers_num"] = num(c, "lightglue_ransac_inliers")
    c["lg_matches_num"] = num(c, "lightglue_matches")
    c["lg_ratio_num"] = num(c, "lightglue_inlier_ratio")
    c["uav_cov_num"] = num(c, "lightglue_uav_coverage")
    c["sat_cov_num"] = num(c, "lightglue_sat_coverage")
    c["min_cov_num"] = np.minimum(c["uav_cov_num"].fillna(0), c["sat_cov_num"].fillna(0))

    return c


def choose_lg_only(g: pd.DataFrame) -> pd.Series:
    return g.sort_values(
        ["lg_score_num", "lg_inliers_num", "lg_matches_num", "min_cov_num", "rank_num"],
        ascending=[False, False, False, False, True],
        kind="mergesort",
    ).iloc[0]


def choose_phog_only(g: pd.DataFrame) -> pd.Series:
    return g.sort_values("rank_num", kind="mergesort").iloc[0]


def choose_rank_bonus(g: pd.DataFrame, bonus: float) -> pd.Series:
    gg = g.copy()
    gg["policy_score"] = gg["lg_score_num"].fillna(-9999) + bonus / gg["rank_num"].clip(lower=1)
    return gg.sort_values(
        ["policy_score", "lg_score_num", "lg_inliers_num", "rank_num"],
        ascending=[False, False, False, True],
        kind="mergesort",
    ).iloc[0]


def choose_log_rank_penalty(g: pd.DataFrame, penalty: float) -> pd.Series:
    gg = g.copy()
    gg["policy_score"] = gg["lg_score_num"].fillna(-9999) - penalty * np.log1p(gg["rank_num"].clip(lower=1) - 1)
    return gg.sort_values(
        ["policy_score", "lg_score_num", "lg_inliers_num", "rank_num"],
        ascending=[False, False, False, True],
        kind="mergesort",
    ).iloc[0]


def choose_phog_protect_margin(g: pd.DataFrame, margin: float, max_lg_rank_for_phog: int) -> pd.Series:
    """
    Choose LightGlue top1 normally.
    But if PHOG top1 is also highly ranked by LightGlue and the score gap is small,
    keep PHOG top1.

    This is unsupervised:
    uses only PHOG rank, LightGlue score, LightGlue rank.
    """
    lg_top = choose_lg_only(g)
    phog_top = choose_phog_only(g)

    lg_score = safe_float(lg_top.get("lg_score_num")) or -9999.0
    phog_lg_score = safe_float(phog_top.get("lg_score_num")) or -9999.0
    phog_lg_rank = safe_float(phog_top.get("lg_rank_num"))

    if phog_lg_rank is not None:
        if phog_lg_rank <= max_lg_rank_for_phog and (lg_score - phog_lg_score) <= margin:
            return phog_top

    return lg_top


def choose_coverage_gate(g: pd.DataFrame, min_cov: float, min_inliers: int) -> pd.Series:
    """
    If LightGlue top1 is low-confidence by coverage/inlier count, fall back to PHOG top1.
    """
    lg_top = choose_lg_only(g)
    phog_top = choose_phog_only(g)

    cov = safe_float(lg_top.get("min_cov_num")) or 0.0
    inliers = safe_float(lg_top.get("lg_inliers_num")) or 0.0

    if cov < min_cov or inliers < min_inliers:
        return phog_top

    return lg_top


def policies():
    out = []

    out.append(("phog_only", lambda g: choose_phog_only(g)))
    out.append(("lightglue_only_current", lambda g: choose_lg_only(g)))

    for b in [1, 2, 4, 6, 8, 10, 12, 16, 20]:
        out.append((f"lg_plus_phog_rank_bonus_{b}", lambda g, b=b: choose_rank_bonus(g, float(b))))

    for p in [0.5, 1, 2, 3, 4, 5, 6]:
        out.append((f"lg_minus_log_rank_penalty_{p}", lambda g, p=p: choose_log_rank_penalty(g, float(p))))

    for margin in [0.5, 1, 2, 3, 4, 5, 6, 8, 10]:
        for topn in [2, 3, 5]:
            out.append((
                f"phog_protect_margin_{margin}_if_phog_lg_rank_le_{topn}",
                lambda g, margin=margin, topn=topn: choose_phog_protect_margin(g, float(margin), int(topn)),
            ))

    for cov in [0.06, 0.10, 0.15, 0.20]:
        for inl in [4, 6, 8, 10, 12]:
            out.append((
                f"coverage_gate_cov_{cov}_inl_{inl}",
                lambda g, cov=cov, inl=inl: choose_coverage_gate(g, float(cov), int(inl)),
            ))

    return out


def evaluate_policy(c: pd.DataFrame, q: pd.DataFrame, name: str, chooser):
    rows = []

    for token, g in c.groupby("token_str", dropna=False):
        chosen = chooser(g)

        qrow = q[q["token"].astype(str) == token]
        if len(qrow):
            qrow = qrow.iloc[0]
            phog_hit = bool(qrow.get("phog_hit", False))
            oracle_hit = bool(qrow.get("oracle_hit", False))
        else:
            phog_hit = False
            oracle_hit = False

        err = safe_float(chosen.get("eval_error_num"))
        hit = bool(err is not None and err <= 40.0)

        rows.append({
            "policy": name,
            "token": token,
            "failure_group": chosen.get("failure_group", ""),
            "chosen_tile_id": chosen.get("tile_id", ""),
            "chosen_candidate_pool_rank": safe_float(chosen.get("rank_num")),
            "chosen_lightglue_rank": safe_float(chosen.get("lg_rank_num")),
            "chosen_error_m": err,
            "hit_le_threshold": hit,
            "phog_hit": phog_hit,
            "oracle_hit": oracle_hit,
            "lightglue_score": safe_float(chosen.get("lg_score_num")),
            "lightglue_inliers": safe_float(chosen.get("lg_inliers_num")),
            "lightglue_matches": safe_float(chosen.get("lg_matches_num")),
            "min_coverage": safe_float(chosen.get("min_cov_num")),
        })

    return pd.DataFrame(rows)


def main():
    args = parse_args()
    dirs = ensure_dirs(args.out_base)
    suffix = f"_{args.run_name}" if args.run_name else ""

    c = prep_candidates(pd.read_csv(args.candidate_scores))
    q = pd.read_csv(args.query_summary)

    q["phog_hit"] = as_bool(q["phog_hit_le_threshold"])
    q["akaze_hit"] = as_bool(q["akaze_hit_le_threshold"])
    q["lightglue_hit"] = as_bool(q["lightglue_hit_le_threshold"])
    q["oracle_hit"] = as_bool(q["oracle_topk_hit_le_threshold"])

    all_policy_rows = []
    summary_rows = []

    for name, chooser in policies():
        dec = evaluate_policy(c, q, name, chooser)
        all_policy_rows.append(dec)

        dec["chosen_error_m"] = pd.to_numeric(dec["chosen_error_m"], errors="coerce")
        hit_rate = dec["hit_le_threshold"].mean()
        median_error = dec["chosen_error_m"].median()
        oracle_hit = dec["oracle_hit"].mean()

        destroyed_phog = int(((dec["phog_hit"]) & (~dec["hit_le_threshold"])).sum())
        rescued_over_phog = int(((~dec["phog_hit"]) & (dec["hit_le_threshold"])).sum())
        recoverable_missed = int(((dec["oracle_hit"]) & (~dec["hit_le_threshold"])).sum())
        unrecoverable = int((~dec["oracle_hit"]).sum())

        summary_rows.append({
            "policy": name,
            "hit_rate": hit_rate,
            "hits": int(dec["hit_le_threshold"].sum()),
            "median_error_m": median_error,
            "mean_error_m": dec["chosen_error_m"].mean(),
            "oracle_hit_rate": oracle_hit,
            "recoverable_missed": recoverable_missed,
            "unrecoverable_candidate_pool": unrecoverable,
            "rescued_over_phog": rescued_over_phog,
            "destroyed_phog_successes": destroyed_phog,
            "median_chosen_pool_rank": dec["chosen_candidate_pool_rank"].median(),
            "median_chosen_lg_rank": dec["chosen_lightglue_rank"].median(),
        })

    decisions = pd.concat(all_policy_rows, ignore_index=True)
    summary = pd.DataFrame(summary_rows)

    summary = summary.sort_values(
        ["hit_rate", "median_error_m", "destroyed_phog_successes"],
        ascending=[False, True, True],
        kind="mergesort",
    )

    best_policy = summary.iloc[0]["policy"]

    decisions_out = dirs["metadata"] / f"s5a3e_policy_token_decisions{suffix}.csv"
    summary_out = dirs["metadata"] / f"s5a3e_policy_summary{suffix}.csv"
    best_out = dirs["metadata"] / f"s5a3e_best_policy_decisions{suffix}.csv"
    report_out = dirs["reports"] / f"s5a3e_lightglue_hybrid_gate_analysis_summary{suffix}.json"

    decisions.to_csv(decisions_out, index=False)
    summary.to_csv(summary_out, index=False)

    best_dec = decisions[decisions["policy"] == best_policy].copy()
    best_dec.to_csv(best_out, index=False)

    fig1 = dirs["figures"] / f"s5a3e_top_policy_hit_rates{suffix}.png"
    top = summary.head(15).copy()
    plt.figure(figsize=(12, 5.5))
    plt.bar(top["policy"], top["hit_rate"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Hit rate <=40 m")
    plt.title("S5A.3E top hybrid/PHOG-protection policies")
    plt.tight_layout()
    plt.savefig(fig1, dpi=180)
    plt.close()

    fig2 = dirs["figures"] / f"s5a3e_top_policy_median_errors{suffix}.png"
    plt.figure(figsize=(12, 5.5))
    plt.bar(top["policy"], top["median_error_m"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Median error [m]")
    plt.title("S5A.3E median error for top policies")
    plt.tight_layout()
    plt.savefig(fig2, dpi=180)
    plt.close()

    report = {
        "stage": "S5A.3E_lightglue_hybrid_gate_analysis",
        "run_name": args.run_name,
        "threshold_m": args.threshold_m,
        "num_policies": int(len(summary)),
        "best_policy": str(best_policy),
        "best_policy_row": summary.iloc[0].to_dict(),
        "baseline_lightglue": summary[summary["policy"] == "lightglue_only_current"].iloc[0].to_dict(),
        "baseline_phog": summary[summary["policy"] == "phog_only"].iloc[0].to_dict(),
        "outputs": {
            "policy_summary_csv": str(summary_out),
            "policy_token_decisions_csv": str(decisions_out),
            "best_policy_decisions_csv": str(best_out),
            "summary_json": str(report_out),
            "hit_rate_figure": str(fig1),
            "median_error_figure": str(fig2),
        },
        "locked_rule": "policy ranking uses only LightGlue score/inliers/coverage/rank and PHOG candidate rank; eval_error is used only after selection",
    }

    with open(report_out, "w") as f:
        json.dump(report, f, indent=2)

    print("S5A.3E LightGlue hybrid/gate analysis complete")
    print("----------------------------------------------")
    print(f"Policies tested:        {len(summary)}")
    print()
    print("Top 12 policies:")
    print(summary.head(12).to_string(index=False))
    print()
    print("Baselines:")
    print(summary[summary['policy'].isin(['phog_only', 'lightglue_only_current'])].to_string(index=False))
    print()
    print(f"Policy summary CSV:     {summary_out}")
    print(f"All decisions CSV:      {decisions_out}")
    print(f"Best decisions CSV:     {best_out}")
    print(f"Summary JSON:           {report_out}")
    print(f"Figures:                {fig1}")
    print(f"                        {fig2}")
    print()
    print("Locked rule: evaluation errors were used only after ranking.")


if __name__ == "__main__":
    main()
