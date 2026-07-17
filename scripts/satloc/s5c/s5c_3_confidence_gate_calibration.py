#!/usr/bin/env python3
"""
S5C.3 — Confidence-gate calibration for temporal LightGlue absolute fixes.

This stage does NOT perform localization or fusion.

It answers:
  Which LightGlue absolute fixes are safe enough to accept online?

Online acceptance may use only image/retrieval evidence:
  - LightGlue score
  - inliers
  - matches
  - coverage
  - union rank
  - top1-vs-top2 LightGlue margin

Evaluation columns such as chosen_error_m and hit_le_threshold are used only
after acceptance to measure precision/false corrections.

Command Used:

python scripts/satloc/s5c/s5c_3_confidence_gate_calibration.py

"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--query-summary",
        type=Path,
        default=Path("outputs/satloc/metadata/s5c_temporal/s5c2_lightglue_union_query_summary_top50_full263.csv"),
    )
    p.add_argument(
        "--candidate-scores",
        type=Path,
        default=Path("outputs/satloc/metadata/s5c_temporal/s5c2_lightglue_union_candidate_scores_top50_full263.csv"),
    )
    p.add_argument("--policy", default="lightglue_only")
    p.add_argument("--threshold-m", type=float, default=40.0)
    p.add_argument("--output-root", type=Path, default=Path("outputs/satloc"))
    return p.parse_args()


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def to_bool(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s.fillna(False)
    return (
        s.astype(str)
        .str.lower()
        .isin(["true", "1", "yes", "y"])
    )


def numeric(df: pd.DataFrame, col: str, default=np.nan) -> pd.Series:
    if col not in df.columns:
        return pd.Series([default] * len(df), index=df.index)
    return pd.to_numeric(df[col], errors="coerce")


def build_token_features(q: pd.DataFrame, c: pd.DataFrame, policy: str) -> pd.DataFrame:
    q = q[q["policy"] == policy].copy()
    if q.empty:
        raise RuntimeError(f"No query rows for policy={policy}")

    q["token"] = q["token"].astype(int)

    # Evaluation-only labels.
    q["hit_eval_only"] = to_bool(q["hit_le_threshold"])
    q["chosen_error_m_eval_only"] = numeric(q, "chosen_error_m")
    q["dangerous_false_eval_only"] = (~q["hit_eval_only"]) & (q["chosen_error_m_eval_only"] > 100.0)

    # Online features already present at decision row.
    q["lg_score"] = numeric(q, "chosen_lg_score")
    q["lg_inliers"] = numeric(q, "chosen_inliers")
    q["lg_matches"] = numeric(q, "chosen_matches")
    q["min_coverage"] = numeric(q, "chosen_min_coverage")
    q["chosen_union_rank"] = numeric(q, "chosen_union_rank")
    q["chosen_lg_rank"] = numeric(q, "chosen_lightglue_rank")

    # Candidate-level top1/top2 LightGlue margin.
    c = c.copy()
    c["token"] = c["token"].astype(int)
    c["lg_rank_num"] = numeric(c, "lightglue_rank")
    c["lg_score_num"] = numeric(c, "lg_score_num")
    c["lg_inliers_num"] = numeric(c, "lg_inliers_num")
    c["lg_matches_num"] = numeric(c, "lg_matches_num")
    c["min_cov_num"] = numeric(c, "min_cov_num")
    c["union_rank_num"] = numeric(c, "union_rank_num")

    margin_rows = []
    for token, g in c.groupby("token"):
        g = g.sort_values("lg_rank_num")
        top1 = g.iloc[0] if len(g) >= 1 else None
        top2 = g.iloc[1] if len(g) >= 2 else None

        top1_score = float(top1["lg_score_num"]) if top1 is not None else np.nan
        top2_score = float(top2["lg_score_num"]) if top2 is not None else np.nan

        if np.isfinite(top1_score) and np.isfinite(top2_score):
            margin = top1_score - top2_score
            ratio = top1_score / max(top2_score, 1e-6)
        else:
            margin = np.nan
            ratio = np.nan

        margin_rows.append({
            "token": int(token),
            "top1_lg_score": top1_score,
            "top2_lg_score": top2_score,
            "lg_score_margin_top1_top2": margin,
            "lg_score_ratio_top1_top2": ratio,
        })

    margins = pd.DataFrame(margin_rows)
    out = q.merge(margins, on="token", how="left")

    # Use 0 margin when no top2 exists; this is conservative.
    out["lg_score_margin_top1_top2"] = out["lg_score_margin_top1_top2"].fillna(0.0)
    out["lg_score_ratio_top1_top2"] = out["lg_score_ratio_top1_top2"].fillna(1.0)

    return out.sort_values("token").reset_index(drop=True)


def eval_gate(df: pd.DataFrame, name: str, mask: pd.Series, params: dict) -> dict:
    accepted = df[mask].copy()
    total = len(df)
    total_hits = int(df["hit_eval_only"].sum())

    n_acc = len(accepted)
    true_hits = int(accepted["hit_eval_only"].sum()) if n_acc else 0
    false_acc = n_acc - true_hits

    precision = true_hits / n_acc if n_acc else 0.0
    accepted_rate = n_acc / total if total else 0.0
    absolute_hit_rate = true_hits / total if total else 0.0
    hit_retention = true_hits / total_hits if total_hits else 0.0

    dangerous_false = int(accepted["dangerous_false_eval_only"].sum()) if n_acc else 0

    return {
        "gate": name,
        "accepted": int(n_acc),
        "accepted_rate": float(accepted_rate),
        "true_hits": int(true_hits),
        "false_accepts": int(false_acc),
        "precision_eval_only": float(precision),
        "absolute_hit_rate_eval_only": float(absolute_hit_rate),
        "hit_retention_vs_lg_hits_eval_only": float(hit_retention),
        "dangerous_false_accepts_gt100m_eval_only": int(dangerous_false),
        "median_error_m_eval_only": float(accepted["chosen_error_m_eval_only"].median()) if n_acc else np.nan,
        "p95_error_m_eval_only": float(accepted["chosen_error_m_eval_only"].quantile(0.95)) if n_acc else np.nan,
        "max_error_m_eval_only": float(accepted["chosen_error_m_eval_only"].max()) if n_acc else np.nan,
        **params,
    }


def sweep_gates(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    # Baseline: accept every LightGlue-only output.
    rows.append(eval_gate(df, "accept_all_lightglue_only", pd.Series(True, index=df.index), {
        "score_min": 0,
        "inliers_min": 0,
        "matches_min": 0,
        "coverage_min": 0,
        "margin_min": 0,
        "union_rank_max": 999,
    }))

    score_grid = [0, 10, 20, 30, 40, 50, 60, 80, 100, 120]
    inlier_grid = [0, 8, 12, 16, 24, 32, 48, 64, 80]
    match_grid = [0, 40, 80, 120, 160, 220]
    coverage_grid = [0, 0.125, 0.1875, 0.25, 0.3125, 0.375, 0.5]
    margin_grid = [0, 2, 5, 10, 20, 40]
    union_rank_grid = [999, 50, 40, 30, 20, 10]

    for score_min in score_grid:
        for inliers_min in inlier_grid:
            for matches_min in match_grid:
                for coverage_min in coverage_grid:
                    for margin_min in margin_grid:
                        for union_rank_max in union_rank_grid:
                            mask = (
                                (df["lg_score"] >= score_min) &
                                (df["lg_inliers"] >= inliers_min) &
                                (df["lg_matches"] >= matches_min) &
                                (df["min_coverage"] >= coverage_min) &
                                (df["lg_score_margin_top1_top2"] >= margin_min) &
                                (df["chosen_union_rank"] <= union_rank_max)
                            )
                            if mask.sum() == 0:
                                continue

                            name = (
                                f"s{score_min}_i{inliers_min}_m{matches_min}_"
                                f"c{coverage_min}_gap{margin_min}_ur{union_rank_max}"
                            )
                            rows.append(eval_gate(df, name, mask, {
                                "score_min": score_min,
                                "inliers_min": inliers_min,
                                "matches_min": matches_min,
                                "coverage_min": coverage_min,
                                "margin_min": margin_min,
                                "union_rank_max": union_rank_max,
                            }))

    out = pd.DataFrame(rows)

    # Useful sorting: high precision first, then more true hits, fewer dangerous false accepts.
    out = out.sort_values(
        [
            "precision_eval_only",
            "true_hits",
            "dangerous_false_accepts_gt100m_eval_only",
            "accepted",
        ],
        ascending=[False, False, True, False],
    ).reset_index(drop=True)

    return out


def choose_recommended(sweep: pd.DataFrame) -> pd.DataFrame:
    recs = []

    targets = [
        ("strict_precision_0p85", 0.85),
        ("balanced_precision_0p75", 0.75),
        ("permissive_precision_0p65", 0.65),
        ("exploratory_precision_0p55", 0.55),
    ]

    for name, target in targets:
        g = sweep[
            (sweep["precision_eval_only"] >= target) &
            (sweep["accepted"] >= 5)
        ].copy()

        if g.empty:
            continue

        g = g.sort_values(
            [
                "true_hits",
                "dangerous_false_accepts_gt100m_eval_only",
                "accepted",
                "precision_eval_only",
            ],
            ascending=[False, True, False, False],
        )

        row = g.iloc[0].copy()
        row["recommended_profile"] = name
        recs.append(row)

    if not recs:
        # fallback: best F1-like balance
        s = sweep.copy()
        precision = s["precision_eval_only"].fillna(0)
        recall = s["hit_retention_vs_lg_hits_eval_only"].fillna(0)
        s["f1_vs_lg_hits"] = 2 * precision * recall / (precision + recall + 1e-9)
        row = s.sort_values("f1_vs_lg_hits", ascending=False).iloc[0].copy()
        row["recommended_profile"] = "fallback_best_f1"
        recs.append(row)

    return pd.DataFrame(recs)


def make_figures(features: pd.DataFrame, sweep: pd.DataFrame, out_dir: Path):
    ensure_dir(out_dir)

    # Score vs error
    plt.figure(figsize=(7, 5))
    colors = features["hit_eval_only"].map({True: 1, False: 0})
    plt.scatter(features["lg_score"], features["chosen_error_m_eval_only"], c=colors, s=16, alpha=0.8)
    plt.axhline(40, linestyle="--")
    plt.xlabel("LightGlue score")
    plt.ylabel("Chosen error [m] eval only")
    plt.title("S5C.3 LightGlue score vs selected error")
    plt.tight_layout()
    plt.savefig(out_dir / "s5c3_lg_score_vs_error.png", dpi=180)
    plt.close()

    # Inliers vs error
    plt.figure(figsize=(7, 5))
    plt.scatter(features["lg_inliers"], features["chosen_error_m_eval_only"], c=colors, s=16, alpha=0.8)
    plt.axhline(40, linestyle="--")
    plt.xlabel("LightGlue inliers")
    plt.ylabel("Chosen error [m] eval only")
    plt.title("S5C.3 inliers vs selected error")
    plt.tight_layout()
    plt.savefig(out_dir / "s5c3_inliers_vs_error.png", dpi=180)
    plt.close()

    # Precision vs accepted count
    plt.figure(figsize=(7, 5))
    plt.scatter(sweep["accepted"], sweep["precision_eval_only"], s=8, alpha=0.5)
    plt.xlabel("Accepted corrections")
    plt.ylabel("Precision eval only")
    plt.title("S5C.3 confidence-gate sweep")
    plt.tight_layout()
    plt.savefig(out_dir / "s5c3_gate_precision_vs_accepted.png", dpi=180)
    plt.close()


def main():
    args = parse_args()

    meta = args.output_root / "metadata" / "s5c_temporal"
    reports = args.output_root / "reports" / "s5c_temporal"
    figs = args.output_root / "figures" / "s5c_temporal"
    for d in [meta, reports, figs]:
        ensure_dir(d)

    q = pd.read_csv(args.query_summary)
    c = pd.read_csv(args.candidate_scores)

    features = build_token_features(q, c, args.policy)
    sweep = sweep_gates(features)
    recommended = choose_recommended(sweep)

    features_out = meta / "s5c3_lightglue_confidence_features_full263.csv"
    sweep_out = meta / "s5c3_confidence_gate_sweep_full263.csv"
    rec_out = meta / "s5c3_recommended_confidence_gates_full263.csv"
    report_out = reports / "s5c3_confidence_gate_calibration_summary.json"

    features.to_csv(features_out, index=False)
    sweep.to_csv(sweep_out, index=False)
    recommended.to_csv(rec_out, index=False)

    make_figures(features, sweep, figs)

    baseline = sweep[sweep["gate"] == "accept_all_lightglue_only"].iloc[0].to_dict()

    report = {
        "stage": "S5C.3",
        "policy_input": args.policy,
        "tokens": int(len(features)),
        "threshold_m": float(args.threshold_m),
        "baseline_accept_all_lightglue_only": baseline,
        "recommended": recommended.to_dict(orient="records"),
        "outputs": {
            "features": str(features_out),
            "gate_sweep": str(sweep_out),
            "recommended_gates": str(rec_out),
            "figures_dir": str(figs),
        },
        "locked_rule": (
            "Acceptance rules use only LightGlue/retrieval evidence. "
            "Error and hit labels are evaluation-only."
        ),
    }

    report_out.write_text(json.dumps(report, indent=2))

    print("S5C.3 Confidence-Gate Calibration")
    print("---------------------------------")
    print(f"Policy input:        {args.policy}")
    print(f"Tokens:              {len(features)}")
    print(f"Baseline hits:       {int(baseline['true_hits'])}/{len(features)}")
    print(f"Baseline precision:  {baseline['precision_eval_only']:.3f}")
    print(f"Baseline median err: {baseline['median_error_m_eval_only']:.1f} m")

    print("\nRecommended gates")
    print("-----------------")
    cols = [
        "recommended_profile",
        "accepted",
        "true_hits",
        "false_accepts",
        "precision_eval_only",
        "hit_retention_vs_lg_hits_eval_only",
        "dangerous_false_accepts_gt100m_eval_only",
        "median_error_m_eval_only",
        "score_min",
        "inliers_min",
        "matches_min",
        "coverage_min",
        "margin_min",
        "union_rank_max",
    ]
    print(recommended[cols].to_string(index=False))

    print("\nSaved outputs")
    print("-------------")
    print(features_out)
    print(sweep_out)
    print(rec_out)
    print(report_out)
    print(figs)

    print("\nLocked rule")
    print("-----------")
    print("Confidence gates are online-evidence only; error columns are evaluation-only.")


if __name__ == "__main__":
    main()
