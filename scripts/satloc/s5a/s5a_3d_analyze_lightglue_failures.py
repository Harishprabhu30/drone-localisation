#!/usr/bin/env python3
"""
S5A.3D — Analyze LightGlue top-50 verifier failures

This script separates:
1. LightGlue successes
2. LightGlue rescues over PHOG/AKAZE
3. LightGlue destroyed PHOG successes
4. Oracle-available but LightGlue missed
5. Candidate-pool failures where no verifier can recover

Code Used:
export PYTHONPATH=$PWD/src

python scripts/satloc/s5a/s5a_3d_analyze_lightglue_failures.py \
  --run-name top50_all73
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--query-summary",
        type=Path,
        default=Path("outputs/satloc/metadata/s5a_learned_local_verifier/s5a3_lightglue_query_summary_top50_all73.csv"),
    )
    p.add_argument(
        "--candidate-scores",
        type=Path,
        default=Path("outputs/satloc/metadata/s5a_learned_local_verifier/s5a3_lightglue_candidate_scores_top50_all73.csv"),
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
    for x in d.values():
        x.mkdir(parents=True, exist_ok=True)
    return d


def as_bool(s):
    if s.dtype == bool:
        return s.fillna(False)
    return s.fillna(False).astype(str).str.lower().isin(["true", "1", "yes"])


def classify(row):
    phog = bool(row["phog_hit"])
    akaze = bool(row["akaze_hit"])
    lg = bool(row["lightglue_hit"])
    oracle = bool(row["oracle_hit"])

    if lg and not phog and not akaze:
        return "lightglue_rescue"
    if lg and phog:
        return "lightglue_preserved_phog_success"
    if phog and not lg:
        return "lightglue_destroyed_phog_success"
    if oracle and not lg:
        return "oracle_available_but_lightglue_missed"
    if not oracle:
        return "candidate_pool_failure_unrecoverable"
    if lg:
        return "lightglue_success_other"
    return "both_fail_other"


def main():
    args = parse_args()
    dirs = ensure_dirs(args.out_base)

    q = pd.read_csv(args.query_summary)
    c = pd.read_csv(args.candidate_scores)

    q["phog_hit"] = as_bool(q["phog_hit_le_threshold"])
    q["akaze_hit"] = as_bool(q["akaze_hit_le_threshold"])
    q["lightglue_hit"] = as_bool(q["lightglue_hit_le_threshold"])
    q["oracle_hit"] = as_bool(q["oracle_topk_hit_le_threshold"])

    numeric_cols = [
        "phog_top1_error_m",
        "akaze_top1_error_m",
        "lightglue_top1_error_m",
        "oracle_topk_error_m",
        "oracle_lightglue_rank",
        "lightglue_top1_score",
        "lightglue_top1_matches",
        "lightglue_top1_inliers",
        "lightglue_top1_inlier_ratio",
        "lightglue_top1_uav_coverage",
        "lightglue_top1_sat_coverage",
    ]
    for col in numeric_cols:
        if col in q.columns:
            q[col] = pd.to_numeric(q[col], errors="coerce")

    q["s5a3d_failure_class"] = q.apply(classify, axis=1)
    q["lightglue_minus_oracle_error_m"] = q["lightglue_top1_error_m"] - q["oracle_topk_error_m"]
    q["lightglue_minus_phog_error_m"] = q["lightglue_top1_error_m"] - q["phog_top1_error_m"]
    q["lightglue_close_to_oracle"] = q["oracle_lightglue_rank"].fillna(9999) <= 3

    # Sort most important diagnostic tables.
    failed = q[~q["lightglue_hit"]].copy()
    recoverable_missed = q[(q["oracle_hit"]) & (~q["lightglue_hit"])].copy()
    unrecoverable = q[~q["oracle_hit"]].copy()
    rescues = q[(q["lightglue_hit"]) & (~q["phog_hit"])].copy()
    destroyed = q[(q["phog_hit"]) & (~q["lightglue_hit"])].copy()

    failed = failed.sort_values(
        ["oracle_hit", "oracle_lightglue_rank", "lightglue_top1_error_m"],
        ascending=[False, True, False],
    )
    recoverable_missed = recoverable_missed.sort_values(
        ["oracle_lightglue_rank", "lightglue_minus_oracle_error_m"],
        ascending=[True, False],
    )
    unrecoverable = unrecoverable.sort_values("oracle_topk_error_m", ascending=True)
    rescues = rescues.sort_values("lightglue_top1_error_m", ascending=True)
    destroyed = destroyed.sort_values("lightglue_top1_error_m", ascending=False)

    # Group summary.
    group_rows = []
    for group, g in q.groupby("failure_group"):
        group_rows.append(
            {
                "failure_group": group,
                "count": len(g),
                "phog_hit_rate": g["phog_hit"].mean(),
                "akaze_hit_rate": g["akaze_hit"].mean(),
                "lightglue_hit_rate": g["lightglue_hit"].mean(),
                "oracle_hit_rate": g["oracle_hit"].mean(),
                "phog_median_error_m": g["phog_top1_error_m"].median(),
                "akaze_median_error_m": g["akaze_top1_error_m"].median(),
                "lightglue_median_error_m": g["lightglue_top1_error_m"].median(),
                "oracle_median_error_m": g["oracle_topk_error_m"].median(),
                "recoverable_missed_count": int(((g["oracle_hit"]) & (~g["lightglue_hit"])).sum()),
                "unrecoverable_pool_failure_count": int((~g["oracle_hit"]).sum()),
                "lightglue_rescue_count": int(((g["lightglue_hit"]) & (~g["phog_hit"])).sum()),
                "lightglue_destroyed_phog_count": int(((g["phog_hit"]) & (~g["lightglue_hit"])).sum()),
            }
        )
    group_df = pd.DataFrame(group_rows).sort_values("failure_group")

    class_df = (
        q["s5a3d_failure_class"]
        .value_counts()
        .rename_axis("s5a3d_failure_class")
        .reset_index(name="count")
    )
    class_df["rate"] = class_df["count"] / len(q)

    suffix = f"_{args.run_name}" if args.run_name else ""

    q_out = dirs["metadata"] / f"s5a3d_query_failure_analysis{suffix}.csv"
    group_out = dirs["metadata"] / f"s5a3d_group_failure_summary{suffix}.csv"
    class_out = dirs["metadata"] / f"s5a3d_failure_class_counts{suffix}.csv"
    failed_out = dirs["metadata"] / f"s5a3d_failed_tokens{suffix}.csv"
    missed_out = dirs["metadata"] / f"s5a3d_oracle_available_but_lg_missed{suffix}.csv"
    unrecoverable_out = dirs["metadata"] / f"s5a3d_candidate_pool_unrecoverable{suffix}.csv"
    rescue_out = dirs["metadata"] / f"s5a3d_lightglue_rescues{suffix}.csv"
    destroyed_out = dirs["metadata"] / f"s5a3d_lightglue_destroyed_phog{suffix}.csv"
    summary_out = dirs["reports"] / f"s5a3d_lightglue_failure_analysis_summary{suffix}.json"

    q.to_csv(q_out, index=False)
    group_df.to_csv(group_out, index=False)
    class_df.to_csv(class_out, index=False)
    failed.to_csv(failed_out, index=False)
    recoverable_missed.to_csv(missed_out, index=False)
    unrecoverable.to_csv(unrecoverable_out, index=False)
    rescues.to_csv(rescue_out, index=False)
    destroyed.to_csv(destroyed_out, index=False)

    # Figure 1: failure class counts.
    fig1 = dirs["figures"] / f"s5a3d_failure_class_counts{suffix}.png"
    plt.figure(figsize=(11, 5))
    plt.bar(class_df["s5a3d_failure_class"], class_df["count"])
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("Token count")
    plt.title("S5A.3D LightGlue outcome classes")
    plt.tight_layout()
    plt.savefig(fig1, dpi=180)
    plt.close()

    # Figure 2: group hit rates.
    fig2 = dirs["figures"] / f"s5a3d_group_hit_rates{suffix}.png"
    x = np.arange(len(group_df))
    w = 0.22
    plt.figure(figsize=(13, 5.5))
    plt.bar(x - 1.5*w, group_df["phog_hit_rate"], w, label="PHOG")
    plt.bar(x - 0.5*w, group_df["akaze_hit_rate"], w, label="AKAZE")
    plt.bar(x + 0.5*w, group_df["lightglue_hit_rate"], w, label="LightGlue")
    plt.bar(x + 1.5*w, group_df["oracle_hit_rate"], w, label="Oracle top50")
    plt.xticks(x, group_df["failure_group"], rotation=35, ha="right")
    plt.ylim(0, 1.05)
    plt.ylabel("Hit rate <= 40 m")
    plt.title("S5A.3D hit rates by original S4C.6 failure group")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig2, dpi=180)
    plt.close()

    # Figure 3: oracle rank for missed recoverable frames.
    fig3 = dirs["figures"] / f"s5a3d_oracle_rank_for_lg_missed{suffix}.png"
    if len(recoverable_missed):
        plot_df = recoverable_missed.sort_values("oracle_lightglue_rank")
        plt.figure(figsize=(10, 4.8))
        plt.bar(plot_df["token"].astype(str), plot_df["oracle_lightglue_rank"])
        plt.axhline(1, linestyle="--", linewidth=1)
        plt.ylabel("Oracle candidate's LightGlue rank")
        plt.xlabel("Token")
        plt.title("Recoverable frames missed by LightGlue")
        plt.tight_layout()
        plt.savefig(fig3, dpi=180)
        plt.close()

    summary = {
        "stage": "S5A.3D_lightglue_failure_analysis",
        "run_name": args.run_name,
        "threshold_m": args.threshold_m,
        "num_queries": int(len(q)),
        "phog_hits": int(q["phog_hit"].sum()),
        "akaze_hits": int(q["akaze_hit"].sum()),
        "lightglue_hits": int(q["lightglue_hit"].sum()),
        "oracle_hits": int(q["oracle_hit"].sum()),
        "phog_hit_rate": float(q["phog_hit"].mean()),
        "akaze_hit_rate": float(q["akaze_hit"].mean()),
        "lightglue_hit_rate": float(q["lightglue_hit"].mean()),
        "oracle_hit_rate": float(q["oracle_hit"].mean()),
        "lightglue_median_error_m": float(q["lightglue_top1_error_m"].median()),
        "oracle_median_error_m": float(q["oracle_topk_error_m"].median()),
        "recoverable_missed_count": int(len(recoverable_missed)),
        "unrecoverable_candidate_pool_count": int(len(unrecoverable)),
        "lightglue_rescue_count": int(len(rescues)),
        "lightglue_destroyed_phog_count": int(len(destroyed)),
        "outputs": {
            "query_analysis_csv": str(q_out),
            "group_summary_csv": str(group_out),
            "failure_class_counts_csv": str(class_out),
            "failed_tokens_csv": str(failed_out),
            "oracle_available_but_lg_missed_csv": str(missed_out),
            "candidate_pool_unrecoverable_csv": str(unrecoverable_out),
            "lightglue_rescues_csv": str(rescue_out),
            "lightglue_destroyed_phog_csv": str(destroyed_out),
            "summary_json": str(summary_out),
            "failure_class_figure": str(fig1),
            "group_hit_rates_figure": str(fig2),
            "oracle_rank_missed_figure": str(fig3),
        },
        "locked_rule": "eval_error/oracle columns used only after LightGlue ranking for analysis",
    }

    with open(summary_out, "w") as f:
        json.dump(summary, f, indent=2)

    print("S5A.3D LightGlue failure analysis complete")
    print("------------------------------------------")
    print(f"Queries:                         {len(q)}")
    print(f"PHOG hits:                       {summary['phog_hits']} / {len(q)}")
    print(f"AKAZE hits:                      {summary['akaze_hits']} / {len(q)}")
    print(f"LightGlue hits:                  {summary['lightglue_hits']} / {len(q)}")
    print(f"Oracle hits:                     {summary['oracle_hits']} / {len(q)}")
    print(f"Recoverable missed by LG:        {len(recoverable_missed)}")
    print(f"Candidate-pool unrecoverable:    {len(unrecoverable)}")
    print(f"LightGlue rescues over PHOG:     {len(rescues)}")
    print(f"LightGlue destroyed PHOG:        {len(destroyed)}")
    print()
    print("Failure class counts:")
    print(class_df.to_string(index=False))
    print()
    print("Recoverable missed tokens:")
    if len(recoverable_missed):
        cols = [
            "token",
            "failure_group",
            "phog_top1_error_m",
            "lightglue_top1_error_m",
            "oracle_topk_error_m",
            "oracle_lightglue_rank",
        ]
        print(recoverable_missed[cols].to_string(index=False))
    else:
        print("None")
    print()
    print(f"Query analysis CSV:              {q_out}")
    print(f"Group summary CSV:               {group_out}")
    print(f"Summary JSON:                    {summary_out}")
    print(f"Figures:                         {fig1}")
    print(f"                                 {fig2}")
    print(f"                                 {fig3}")
    print()
    print("Locked rule: reference/error columns were used only after LightGlue ranking.")


if __name__ == "__main__":
    main()
