#!/usr/bin/env python3
'''

code to run: python scripts/satloc/s4b_1e_plot_texture_entropy.py --token 1

similarity vs edge density:
  if false positives sit high-right, dense texture is being rewarded.

similarity vs sparsity:
  if GT tiles have higher sparsity but lower score, our descriptor undervalues clean structure.

sorted bar plot:
  check whether high-scoring wrong tiles have high edge density and low sparsity.
'''

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


REPORT_DIR = Path("outputs/satloc/reports/s4b_structural_retrieval")
FIG_DIR = Path("outputs/satloc/figures/s4b_structural_retrieval/s4b1e_texture_entropy")
META_DIR = Path("outputs/satloc/metadata/s4b_structural_retrieval")


def group_name(category: str) -> str:
    category = str(category)
    if category == "TRUE_GT_CENTER":
        return "GT center"
    if "SHIFTED_GT_NEIGHBOR" in category:
        return "GT neighbor"
    if category == "RANK_1_WINNER":
        return "Rank 1 false"
    if "FALSE_POSITIVE" in category:
        return "False positive"
    return "Other"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", type=int, required=True)
    args = parser.parse_args()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    META_DIR.mkdir(parents=True, exist_ok=True)

    token_str = f"{args.token:04d}"
    json_path = REPORT_DIR / f"s4b1e_token{token_str}_entropy_verification.json"

    if not json_path.exists():
        raise FileNotFoundError(f"Missing diagnostic JSON: {json_path}")

    df = pd.read_json(json_path)
    df["group"] = df["category"].map(group_name)
    df["label"] = df["tile_id"].astype(str) + " / " + df["category"].astype(str)

    summary_csv = META_DIR / f"s4b1e_token{token_str}_texture_group_summary.csv"
    scatter1_png = FIG_DIR / f"s4b1e_token{token_str}_similarity_vs_edge_density.png"
    scatter2_png = FIG_DIR / f"s4b1e_token{token_str}_similarity_vs_sparsity.png"
    bar_png = FIG_DIR / f"s4b1e_token{token_str}_sorted_similarity_texture_table.png"

    group_summary = (
        df.groupby("group")
        .agg(
            count=("tile_id", "count"),
            mean_sim=("combined_sim", "mean"),
            mean_error_m=("center_err_m", "mean"),
            mean_entropy=("hog_entropy", "mean"),
            mean_edge_density_pct=("edge_density_pct", "mean"),
            mean_sparsity_cv=("structural_sparsity_cv", "mean"),
        )
        .reset_index()
        .sort_values("mean_sim", ascending=False)
    )
    group_summary.to_csv(summary_csv, index=False)

    # Plot 1: similarity vs edge density
    fig, ax = plt.subplots(figsize=(10, 7))
    for group, gdf in df.groupby("group"):
        ax.scatter(gdf["edge_density_pct"], gdf["combined_sim"], label=group, s=70)
        for _, r in gdf.iterrows():
            ax.annotate(str(r["tile_id"]), (r["edge_density_pct"], r["combined_sim"]), fontsize=8)

    ax.set_title(f"S4B.1e token {token_str}: similarity vs edge density")
    ax.set_xlabel("Edge density (%)")
    ax.set_ylabel("Combined HOG+edge similarity")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(scatter1_png, dpi=160)
    plt.close(fig)

    # Plot 2: similarity vs sparsity
    fig, ax = plt.subplots(figsize=(10, 7))
    for group, gdf in df.groupby("group"):
        ax.scatter(gdf["structural_sparsity_cv"], gdf["combined_sim"], label=group, s=70)
        for _, r in gdf.iterrows():
            ax.annotate(str(r["tile_id"]), (r["structural_sparsity_cv"], r["combined_sim"]), fontsize=8)

    ax.set_title(f"S4B.1e token {token_str}: similarity vs structural sparsity")
    ax.set_xlabel("Structural sparsity CV")
    ax.set_ylabel("Combined HOG+edge similarity")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(scatter2_png, dpi=160)
    plt.close(fig)

    # Plot 3: sorted table-like bar view
    plot_df = df.sort_values("combined_sim", ascending=True).copy()
    labels = plot_df["tile_id"].astype(str) + " | " + plot_df["group"]

    fig, ax = plt.subplots(figsize=(12, max(7, 0.45 * len(plot_df))))
    ax.barh(labels, plot_df["combined_sim"])
    ax.set_title(f"S4B.1e token {token_str}: candidates sorted by combined similarity")
    ax.set_xlabel("Combined HOG+edge similarity")
    ax.grid(True, axis="x", alpha=0.3)

    for i, (_, r) in enumerate(plot_df.iterrows()):
        ax.text(
            r["combined_sim"] + 0.002,
            i,
            f"err={r['center_err_m']:.1f}m, edge={r['edge_density_pct']:.1f}%, sp={r['structural_sparsity_cv']:.2f}",
            va="center",
            fontsize=8,
        )

    fig.tight_layout()
    fig.savefig(bar_png, dpi=160)
    plt.close(fig)

    print("S4B.1e texture entropy plots complete")
    print("-------------------------------------")
    print(f"Token:               {token_str}")
    print(f"Input JSON:          {json_path}")
    print(f"Group summary CSV:   {summary_csv}")
    print(f"Similarity-edge:     {scatter1_png}")
    print(f"Similarity-sparsity: {scatter2_png}")
    print(f"Sorted bar plot:     {bar_png}")


if __name__ == "__main__":
    main()
