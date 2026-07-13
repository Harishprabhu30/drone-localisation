#!/usr/bin/env python3
"""
S5B.1C — Build union candidate pools from S5B.1B full-map variants

Purpose
-------
Combine top-N candidates from multiple S5B.1B full-map preprocessing variants,
deduplicate satellite tiles, and evaluate whether the correct tile enters the
union top-K pool.

This prepares S5B.2 LightGlue reranking.

Locked rule
-----------
eval_error_m is used only after union construction for evaluation.
The union pool itself is built from image-derived candidate rankings only.

Code Used:
export PYTHONPATH=$PWD/src

python scripts/satloc/s5b/s5b_1c_build_variant_union_pool.py \
  --run-name cpf_union_v3_v5_v8_v9 \
  --per-variant-top-n 50 \
  --union-top-k-eval 200
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
        "--old-ranked",
        type=Path,
        default=Path("outputs/satloc/metadata/s5b_candidate_pool_improvement/s5b1b_fullmap_ranked_top200_cpf_fullmap_all40.csv"),
    )
    p.add_argument(
        "--new-ranked",
        type=Path,
        default=Path("outputs/satloc/metadata/s5b_candidate_pool_improvement/s5b1b_fullmap_ranked_top200_cpf_fullmap_logchroma_all40.csv"),
    )
    p.add_argument("--out-base", type=Path, default=Path("outputs/satloc"))
    p.add_argument("--run-name", type=str, default="cpf_union_v3_v5_v8_v9")
    p.add_argument(
        "--variants",
        type=str,
        default="v3_green_suppressed,v5_edge_magnitude,v8_lab_logchroma_fused,v9_canny_structure",
    )
    p.add_argument("--per-variant-top-n", type=int, default=50)
    p.add_argument("--union-top-k-eval", type=int, default=200)
    p.add_argument("--threshold-m", type=float, default=40.0)
    return p.parse_args()


def ensure_dirs(base: Path):
    d = {
        "metadata": base / "metadata" / "s5b_candidate_pool_improvement",
        "reports": base / "reports" / "s5b_candidate_pool_improvement",
        "figures": base / "figures" / "s5b_candidate_pool_improvement",
    }
    for p in d.values():
        p.mkdir(parents=True, exist_ok=True)
    return d


def safe_float(x: Any):
    try:
        if x is None or pd.isna(x):
            return None
        return float(x)
    except Exception:
        return None


def build_union_for_token(g: pd.DataFrame, variants: List[str], per_variant_top_n: int) -> pd.DataFrame:
    parts = []

    for variant in variants:
        sub = g[g["variant"].astype(str) == variant].copy()
        if len(sub) == 0:
            continue

        sub["rank_num"] = pd.to_numeric(sub["rank"], errors="coerce")
        sub = sub.sort_values("rank_num", kind="mergesort").head(per_variant_top_n).copy()
        sub["source_variant"] = variant
        sub["source_rank"] = sub["rank_num"]
        sub["source_score"] = pd.to_numeric(sub["score"], errors="coerce")
        sub["source_distance"] = pd.to_numeric(sub["distance"], errors="coerce")
        parts.append(sub)

    if not parts:
        return pd.DataFrame()

    pool = pd.concat(parts, ignore_index=True)

    # Deduplicate by tile_id. Keep strongest normalized source.
    # Smaller rank is better. If same tile appears multiple times, keep:
    # 1. best source rank
    # 2. best distance
    # 3. collect all variants where it appears.
    pool["tile_id_str"] = pool["tile_id"].astype(str)
    pool["source_rank"] = pd.to_numeric(pool["source_rank"], errors="coerce")
    pool["source_distance"] = pd.to_numeric(pool["source_distance"], errors="coerce")

    rows = []
    for tile_id, tg in pool.groupby("tile_id_str"):
        tg = tg.sort_values(["source_rank", "source_distance"], ascending=[True, True], kind="mergesort")
        first = tg.iloc[0].copy()

        variants_joined = ",".join(sorted(tg["source_variant"].astype(str).unique()))
        best_rank = float(tg["source_rank"].min())
        best_distance = float(tg["source_distance"].min())

        # Union priority: tile appearing in multiple variants is stronger.
        support_count = int(tg["source_variant"].nunique())
        first["union_support_count"] = support_count
        first["union_variants"] = variants_joined
        first["union_best_source_rank"] = best_rank
        first["union_best_distance"] = best_distance

        rows.append(first)

    out = pd.DataFrame(rows)

    # Rank union candidates.
    # Primary: support count desc
    # Secondary: best source rank asc
    # Tertiary: best distance asc
    out = out.sort_values(
        ["union_support_count", "union_best_source_rank", "union_best_distance"],
        ascending=[False, True, True],
        kind="mergesort",
    ).copy()

    out["union_rank"] = np.arange(1, len(out) + 1)

    return out


def evaluate_union(pool: pd.DataFrame, top_k: int, threshold_m: float) -> Dict[str, Any]:
    if len(pool) == 0:
        return {
            "candidate_count": 0,
            "top1_error_m": None,
            "top1_hit_le_threshold": False,
            "oracle_topk_error_m": None,
            "oracle_topk_hit_le_threshold": False,
            "first_correct_rank": None,
            "best_correct_error_m": None,
        }

    pool = pool.copy()
    pool["eval_error_num"] = pd.to_numeric(pool["eval_error_m"], errors="coerce")

    top1_error = safe_float(pool.iloc[0]["eval_error_num"])
    top1_hit = bool(top1_error is not None and top1_error <= threshold_m)

    topk = pool.head(top_k).dropna(subset=["eval_error_num"])
    oracle_error = safe_float(topk["eval_error_num"].min()) if len(topk) else None
    oracle_hit = bool(oracle_error is not None and oracle_error <= threshold_m)

    correct = pool[pool["eval_error_num"] <= threshold_m].copy()
    first_correct_rank = None
    best_correct_error = None

    if len(correct):
        correct = correct.sort_values("union_rank", kind="mergesort")
        first_correct_rank = int(correct.iloc[0]["union_rank"])
        best_correct_error = float(correct["eval_error_num"].min())

    return {
        "candidate_count": int(len(pool)),
        "top1_error_m": top1_error,
        "top1_hit_le_threshold": top1_hit,
        "oracle_topk_error_m": oracle_error,
        "oracle_topk_hit_le_threshold": oracle_hit,
        "first_correct_rank": first_correct_rank,
        "best_correct_error_m": best_correct_error,
    }


def main():
    args = parse_args()
    dirs = ensure_dirs(args.out_base)
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    suffix = f"_{args.run_name}" if args.run_name else ""

    if not args.old_ranked.exists():
        raise FileNotFoundError(args.old_ranked)
    if not args.new_ranked.exists():
        raise FileNotFoundError(args.new_ranked)

    old = pd.read_csv(args.old_ranked)
    new = pd.read_csv(args.new_ranked)
    df = pd.concat([old, new], ignore_index=True)

    df = df[df["variant"].astype(str).isin(variants)].copy()
    df["token_str"] = df["token"].astype(str)

    union_rows = []
    query_rows = []

    for token, g in df.groupby("token_str"):
        pool = build_union_for_token(g, variants, args.per_variant_top_n)

        if len(pool):
            pool["token"] = token
            union_rows.append(pool)

        ev = evaluate_union(pool, args.union_top_k_eval, args.threshold_m)
        query_rows.append(
            {
                "token": token,
                "variants": ",".join(variants),
                "per_variant_top_n": args.per_variant_top_n,
                "union_top_k_eval": args.union_top_k_eval,
                **ev,
            }
        )

    union_df = pd.concat(union_rows, ignore_index=True) if union_rows else pd.DataFrame()
    query_df = pd.DataFrame(query_rows)

    query_df["token_int"] = pd.to_numeric(query_df["token"], errors="coerce")
    query_df = query_df.sort_values("token_int", kind="mergesort")

    hit = query_df[query_df["oracle_topk_hit_le_threshold"] == True].copy()
    recovered_tokens = sorted(hit["token"].astype(str).tolist(), key=lambda x: int(float(x)))

    summary = {
        "stage": "S5B.1C_variant_union_pool",
        "run_name": args.run_name,
        "variants": variants,
        "per_variant_top_n": args.per_variant_top_n,
        "union_top_k_eval": args.union_top_k_eval,
        "threshold_m": args.threshold_m,
        "tokens": int(len(query_df)),
        "oracle_topk_hits": int(query_df["oracle_topk_hit_le_threshold"].sum()),
        "oracle_topk_hit_rate": float(query_df["oracle_topk_hit_le_threshold"].mean()) if len(query_df) else 0.0,
        "top1_hits": int(query_df["top1_hit_le_threshold"].sum()),
        "top1_hit_rate": float(query_df["top1_hit_le_threshold"].mean()) if len(query_df) else 0.0,
        "median_candidate_count": float(pd.to_numeric(query_df["candidate_count"], errors="coerce").median()),
        "median_first_correct_rank": safe_float(pd.to_numeric(query_df["first_correct_rank"], errors="coerce").median()),
        "recovered_tokens": recovered_tokens,
        "locked_rule": "eval_error_m used only after union ranking for evaluation",
    }

    union_out = dirs["metadata"] / f"s5b1c_union_candidate_pool{suffix}.csv"
    query_out = dirs["metadata"] / f"s5b1c_union_query_summary{suffix}.csv"
    report_out = dirs["reports"] / f"s5b1c_union_candidate_pool_summary{suffix}.json"
    fig_out = dirs["figures"] / f"s5b1c_union_first_correct_rank{suffix}.png"

    union_df.to_csv(union_out, index=False)
    query_df.to_csv(query_out, index=False)

    report = dict(summary)
    report["outputs"] = {
        "union_candidate_pool_csv": str(union_out),
        "query_summary_csv": str(query_out),
        "summary_json": str(report_out),
        "first_correct_rank_figure": str(fig_out),
    }

    with open(report_out, "w") as f:
        json.dump(report, f, indent=2)

    plot_df = query_df.dropna(subset=["first_correct_rank"]).copy()
    if len(plot_df):
        plt.figure(figsize=(11, 5))
        plt.bar(plot_df["token"].astype(str), pd.to_numeric(plot_df["first_correct_rank"], errors="coerce"))
        plt.axhline(args.union_top_k_eval, linestyle="--", linewidth=1)
        plt.ylabel("First correct union rank")
        plt.xlabel("Token")
        plt.title("S5B.1C first correct rank in union candidate pool")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(fig_out, dpi=180)
        plt.close()

    print("S5B.1C union candidate-pool construction complete")
    print("-------------------------------------------------")
    print(f"Variants:                 {variants}")
    print(f"Per-variant top-N:         {args.per_variant_top_n}")
    print(f"Union top-K eval:          {args.union_top_k_eval}")
    print(f"Tokens:                    {summary['tokens']}")
    print(f"Oracle hits:               {summary['oracle_topk_hits']} / {summary['tokens']}")
    print(f"Oracle hit rate:           {summary['oracle_topk_hit_rate']:.3f}")
    print(f"Top1 hits:                 {summary['top1_hits']} / {summary['tokens']}")
    print(f"Median union candidates:   {summary['median_candidate_count']}")
    print(f"Median first correct rank: {summary['median_first_correct_rank']}")
    print(f"Recovered tokens:          {','.join(recovered_tokens)}")
    print()
    print(f"Union pool CSV:            {union_out}")
    print(f"Query summary CSV:         {query_out}")
    print(f"Summary JSON:              {report_out}")
    print(f"Figure:                    {fig_out}")
    print()
    print("Locked rule: reference/error was used only after union construction.")


if __name__ == "__main__":
    main()
