#!/usr/bin/env python3
'''
code to run:
python scripts/satloc/s4b_1f_texture_penalty_rerank.py \
  --token 1 \
  --edge-penalty 0.050 \
  --sparsity-reward 0.045 \
  --entropy-penalty 0.015

it is clear now, the penalty added was helpful to push Gt tiles to rank 3 but it also pushed a FALSE tile with vegetation noise upward to rank 1 and 2.
Therefore, it is clearly not enough to improve the edgeg identification. Now, lets try suppressing the vegetation edges and more like:

"This specific cell is forest/noisy, so downweight it.
 This specific cell has clean pond/road/building boundary, so keep it."
'''
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


REPORT_DIR = Path("outputs/satloc/reports/s4b_structural_retrieval")
META_DIR = Path("outputs/satloc/metadata/s4b_structural_retrieval")
FIG_DIR = Path("outputs/satloc/figures/s4b_structural_retrieval/s4b1f_texture_penalty")


def minmax(s: pd.Series) -> pd.Series:
    lo = s.min()
    hi = s.max()
    if hi <= lo:
        return s * 0.0
    return (s - lo) / (hi - lo)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", type=int, required=True)
    parser.add_argument("--edge-penalty", type=float, default=0.050)
    parser.add_argument("--sparsity-reward", type=float, default=0.045)
    parser.add_argument("--entropy-penalty", type=float, default=0.015)
    args = parser.parse_args()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    META_DIR.mkdir(parents=True, exist_ok=True)

    token_str = f"{args.token:04d}"
    json_path = REPORT_DIR / f"s4b1e_token{token_str}_entropy_verification.json"

    if not json_path.exists():
        raise FileNotFoundError(f"Missing S4B.1e JSON: {json_path}")

    df = pd.read_json(json_path)

    df["edge_density_norm"] = minmax(df["edge_density_pct"])
    df["sparsity_norm"] = minmax(df["structural_sparsity_cv"])
    df["entropy_norm"] = minmax(df["hog_entropy"])

    df["texture_penalty_score"] = (
        df["combined_sim"]
        - args.edge_penalty * df["edge_density_norm"]
        + args.sparsity_reward * df["sparsity_norm"]
        - args.entropy_penalty * df["entropy_norm"]
    )

    df["original_rank_local"] = df["combined_sim"].rank(ascending=False, method="min").astype(int)
    df["texture_rank_local"] = df["texture_penalty_score"].rank(ascending=False, method="min").astype(int)

    df = df.sort_values("texture_penalty_score", ascending=False).copy()

    out_csv = META_DIR / f"s4b1f_token{token_str}_texture_penalty_rerank.csv"
    out_png = FIG_DIR / f"s4b1f_token{token_str}_texture_penalty_rerank.png"

    df.to_csv(out_csv, index=False)

    labels = df["tile_id"].astype(str) + " | " + df["category"].astype(str)

    fig, ax = plt.subplots(figsize=(13, max(7, 0.48 * len(df))))
    y = range(len(df))

    ax.barh(y, df["combined_sim"], label="original combined")
    ax.barh(y, df["texture_penalty_score"], alpha=0.65, label="texture-penalty score")

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Score")
    ax.set_title(
        f"S4B.1f token {token_str}: original vs texture-penalty rerank\n"
        f"score = combined - {args.edge_penalty}*edge + {args.sparsity_reward}*sparsity - {args.entropy_penalty}*entropy"
    )
    ax.grid(True, axis="x", alpha=0.3)
    ax.legend()

    for i, (_, r) in enumerate(df.iterrows()):
        ax.text(
            max(r["combined_sim"], r["texture_penalty_score"]) + 0.002,
            i,
            f"origR={r['original_rank_local']}, newR={r['texture_rank_local']}, "
            f"err={r['center_err_m']:.1f}m, edge={r['edge_density_pct']:.1f}%, sp={r['structural_sparsity_cv']:.2f}",
            va="center",
            fontsize=8,
        )

    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)

    print("S4B.1f texture-penalty rerank complete")
    print("--------------------------------------")
    print(f"Token:              {token_str}")
    print(f"Input JSON:         {json_path}")
    print(f"Output CSV:         {out_csv}")
    print(f"Output plot:        {out_png}")
    print("")
    print("Top candidates after texture penalty:")
    print(df[[
        "tile_id",
        "category",
        "center_err_m",
        "combined_sim",
        "texture_penalty_score",
        "edge_density_pct",
        "hog_entropy",
        "structural_sparsity_cv",
        "original_rank_local",
        "texture_rank_local",
    ]].head(12).to_string(index=False))


if __name__ == "__main__":
    main()
