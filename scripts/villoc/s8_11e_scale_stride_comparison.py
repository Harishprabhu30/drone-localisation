"""
Command Executed:

python scripts/villoc/s8_11e_scale_stride_comparison.py \
  2>&1 | tee outputs/villoc/90_deg/logs/s8_11e_scale_stride_comparison.log

"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path.cwd().resolve()

TAG = "dinov2_vits14_img224_center_square_avgpatch_cpu"

RETRIEVAL_DIR = ROOT / "outputs/villoc/90_deg/retrieval/s8_11d"
REPORT_DIR = ROOT / "outputs/villoc/90_deg/reports/s8_11e"
FIGURE_DIR = ROOT / "outputs/villoc/90_deg/figures/s8_11e"

SUMMARY_INPUT = (
    ROOT
    / "outputs/villoc/90_deg/reports/s8_11d"
    / f"s8_11d_independent_retrieval_summary_{TAG}.csv"
)

VARIANTS = ["512_s256", "1024_s512", "1024_s256"]
RECALL_KS = [1, 5, 10, 20, 50, 100]

QUERY_EVAL_INPUTS = {
    variant: (
        RETRIEVAL_DIR
        / f"s8_11d_query_eval_{variant}_{TAG}.csv"
    )
    for variant in VARIANTS
}


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_inputs() -> None:
    required = [SUMMARY_INPUT, *QUERY_EVAL_INPUTS.values()]
    missing = [str(path) for path in required if not path.exists()]

    if missing:
        raise FileNotFoundError(
            "Missing S8.11D inputs:\n" + "\n".join(missing)
        )


def normalize_query_id(series: pd.Series) -> pd.Series:
    def normalize(value) -> str:
        text = str(value).strip()
        if text.endswith(".0"):
            try:
                return str(int(float(text)))
            except ValueError:
                return text
        return text

    return series.map(normalize)


def load_query_tables() -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}

    for variant, path in QUERY_EVAL_INPUTS.items():
        df = pd.read_csv(path)
        df["query_id"] = normalize_query_id(df["query_id"])

        if df["query_id"].duplicated().any():
            duplicates = df.loc[
                df["query_id"].duplicated(keep=False),
                "query_id",
            ].tolist()
            raise ValueError(
                f"{variant}: duplicate query IDs: {duplicates[:20]}"
            )

        tables[variant] = df.sort_values("query_id").reset_index(drop=True)

    reference_ids = set(tables[VARIANTS[0]]["query_id"])

    for variant in VARIANTS[1:]:
        current_ids = set(tables[variant]["query_id"])
        if current_ids != reference_ids:
            raise ValueError(
                f"Query-ID mismatch for {variant}: "
                f"missing={sorted(reference_ids-current_ids)[:10]}, "
                f"extra={sorted(current_ids-reference_ids)[:10]}"
            )

    return tables


def build_merged_query_table(
    tables: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    merged = pd.DataFrame(
        {"query_id": tables[VARIANTS[0]]["query_id"].copy()}
    )

    keep_columns = [
        "top1_tile_id",
        "top1_score",
        "top1_center_error_m",
        "top1_is_oracle",
        "oracle_tile_count",
        "first_oracle_rank",
        "has_oracle",
        "top1_error_le_20m",
        "top1_error_le_40m",
        "top1_error_le_80m",
        "top1_error_le_120m",
    ]

    for variant, df in tables.items():
        available = ["query_id"] + [
            col for col in keep_columns if col in df.columns
        ]

        renamed = df[available].copy().rename(
            columns={
                col: f"{variant}__{col}"
                for col in available
                if col != "query_id"
            }
        )

        merged = merged.merge(
            renamed,
            on="query_id",
            how="left",
            validate="one_to_one",
        )

    rank_cols = [
        f"{variant}__first_oracle_rank"
        for variant in VARIANTS
    ]
    error_cols = [
        f"{variant}__top1_center_error_m"
        for variant in VARIANTS
    ]

    merged["best_oracle_rank_any_variant"] = (
        merged[rank_cols].min(axis=1, skipna=True)
    )
    merged["best_top1_error_any_variant_m"] = (
        merged[error_cols].min(axis=1, skipna=True)
    )

    def best_rank_variant(row: pd.Series) -> str:
        values = {
            variant: row[f"{variant}__first_oracle_rank"]
            for variant in VARIANTS
        }
        valid = {
            key: float(value)
            for key, value in values.items()
            if pd.notna(value)
        }

        if not valid:
            return "none"

        best = min(valid.values())
        winners = [
            key for key, value in valid.items()
            if value == best
        ]
        return "|".join(winners)

    def best_error_variant(row: pd.Series) -> str:
        values = {
            variant: row[f"{variant}__top1_center_error_m"]
            for variant in VARIANTS
        }
        valid = {
            key: float(value)
            for key, value in values.items()
            if pd.notna(value)
        }

        if not valid:
            return "none"

        best = min(valid.values())
        winners = [
            key for key, value in valid.items()
            if math.isclose(value, best, rel_tol=0.0, abs_tol=1e-9)
        ]
        return "|".join(winners)

    merged["best_oracle_rank_variant"] = merged.apply(
        best_rank_variant,
        axis=1,
    )
    merged["best_top1_error_variant"] = merged.apply(
        best_error_variant,
        axis=1,
    )

    for k in [1, 5, 10, 20, 50]:
        hit_cols = []

        for variant in VARIANTS:
            col = f"{variant}__first_oracle_rank"
            hit_col = f"{variant}__oracle_in_top{k}"
            merged[hit_col] = (
                merged[col].notna()
                & (merged[col].astype(float) <= k)
            )
            hit_cols.append(hit_col)

        merged[f"any_variant_oracle_in_top{k}"] = (
            merged[hit_cols].any(axis=1)
        )

    merged["hard_all_variants_rank_gt20"] = ~merged[
        "any_variant_oracle_in_top20"
    ]
    merged["hard_all_variants_top1_error_gt120m"] = np.logical_and.reduce(
        [
            merged[f"{variant}__top1_center_error_m"].fillna(np.inf) > 120.0
            for variant in VARIANTS
        ]
    )

    return merged


def build_recall_table(
    summary: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for _, row in summary.iterrows():
        variant = str(row["variant"])

        for k in RECALL_KS:
            rows.append(
                {
                    "variant": variant,
                    "k": k,
                    "recall": float(row[f"recall_at_{k}"]),
                    "hits": int(round(
                        float(row[f"recall_at_{k}"])
                        * int(row["query_count"])
                    )),
                    "query_count": int(row["query_count"]),
                }
            )

    return pd.DataFrame(rows)


def build_pairwise_rescue_table(
    merged: pd.DataFrame,
    k: int,
) -> pd.DataFrame:
    rows = []

    for source in VARIANTS:
        source_hit = merged[f"{source}__oracle_in_top{k}"]

        for rescuer in VARIANTS:
            rescuer_hit = merged[f"{rescuer}__oracle_in_top{k}"]

            source_fail = ~source_hit
            rescued = source_fail & rescuer_hit

            rows.append(
                {
                    "k": k,
                    "source_variant": source,
                    "rescuer_variant": rescuer,
                    "source_failures": int(source_fail.sum()),
                    "rescued_queries": int(rescued.sum()),
                    "rescue_rate_among_source_failures": (
                        float(rescued.sum() / source_fail.sum())
                        if source_fail.sum() > 0
                        else 0.0
                    ),
                }
            )

    return pd.DataFrame(rows)


def plot_recall_at_k(recall_df: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(9, 5.5))

    for variant in VARIANTS:
        sub = recall_df[recall_df["variant"] == variant]
        ax.plot(
            sub["k"],
            sub["recall"],
            marker="o",
            linewidth=2,
            label=variant,
        )

    ax.set_title("S8.11E DINOv2 Recall@K by Tile Variant")
    ax.set_xlabel("Candidate pool depth K")
    ax.set_ylabel("Oracle tile recall")
    ax.set_xticks(RECALL_KS)
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    path = FIGURE_DIR / f"s8_11e_recall_at_k_{TAG}.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_oracle_rank_boxplot(
    tables: dict[str, pd.DataFrame],
) -> Path:
    data = []
    labels = []

    for variant in VARIANTS:
        ranks = (
            tables[variant]["first_oracle_rank"]
            .dropna()
            .astype(float)
            .to_numpy()
        )
        data.append(ranks)
        labels.append(variant)

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.boxplot(
        data,
        labels=labels,
        showfliers=True,
    )

    ax.set_title("S8.11E First Oracle Rank Distribution")
    ax.set_xlabel("Tile variant")
    ax.set_ylabel("First oracle rank")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()

    path = FIGURE_DIR / f"s8_11e_first_oracle_rank_boxplot_{TAG}.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_rank_by_query(
    merged: pd.DataFrame,
) -> Path:
    plot_df = merged.copy()

    try:
        plot_df["_query_order"] = (
            pd.to_numeric(plot_df["query_id"], errors="raise")
        )
    except Exception:
        plot_df["_query_order"] = np.arange(len(plot_df))

    plot_df = plot_df.sort_values("_query_order").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(12, 5.8))

    for variant in VARIANTS:
        rank = (
            plot_df[f"{variant}__first_oracle_rank"]
            .astype(float)
            .clip(upper=50)
        )

        ax.plot(
            np.arange(len(plot_df)),
            rank,
            linewidth=1.4,
            label=variant,
        )

    ax.axhline(5, linestyle="--", linewidth=1, label="Top-5")
    ax.axhline(10, linestyle="--", linewidth=1, label="Top-10")
    ax.axhline(20, linestyle="--", linewidth=1, label="Top-20")

    ax.set_title("S8.11E Oracle Rank Across Ordered UAV Queries")
    ax.set_xlabel("Ordered UAV query")
    ax.set_ylabel("First oracle rank, clipped at 50")
    ax.set_ylim(0, 52)
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=3)
    fig.tight_layout()

    path = FIGURE_DIR / f"s8_11e_oracle_rank_by_query_{TAG}.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_top1_error_ecdf(
    tables: dict[str, pd.DataFrame],
) -> Path:
    fig, ax = plt.subplots(figsize=(9, 5.5))

    for variant in VARIANTS:
        errors = (
            tables[variant]["top1_center_error_m"]
            .dropna()
            .astype(float)
            .sort_values()
            .to_numpy()
        )

        y = np.arange(1, len(errors) + 1) / len(errors)

        ax.plot(
            errors,
            y,
            linewidth=2,
            label=variant,
        )

    ax.axvline(40, linestyle="--", linewidth=1, label="40 m")
    ax.axvline(80, linestyle="--", linewidth=1, label="80 m")
    ax.axvline(120, linestyle="--", linewidth=1, label="120 m")

    ax.set_title("S8.11E Top-1 Center Error ECDF")
    ax.set_xlabel("Top-1 tile-center error (m)")
    ax.set_ylabel("Fraction of queries")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    path = FIGURE_DIR / f"s8_11e_top1_error_ecdf_{TAG}.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def choose_candidate_policy(
    summary: pd.DataFrame,
    merged: pd.DataFrame,
) -> dict:
    indexed = summary.set_index("variant")

    primary_variant = "1024_s512"
    primary_k = 20

    primary_recall = float(
        indexed.loc[primary_variant, f"recall_at_{primary_k}"]
    )
    primary_hits = int(round(primary_recall * len(merged)))

    alternate_specs = [
        ("1024_s256", 50),
        ("512_s256", 20),
        ("512_s256", 50),
    ]

    alternates = []

    primary_hit_mask = merged[
        f"{primary_variant}__oracle_in_top{primary_k}"
    ]

    for variant, k in alternate_specs:
        alt_mask = merged[f"{variant}__oracle_in_top{k}"]
        rescued = (~primary_hit_mask) & alt_mask

        alternates.append(
            {
                "variant": variant,
                "top_k": k,
                "recall": float(
                    indexed.loc[variant, f"recall_at_{k}"]
                ),
                "rescues_over_primary_count": int(rescued.sum()),
                "rescued_query_ids": (
                    merged.loc[rescued, "query_id"]
                    .astype(str)
                    .tolist()
                ),
            }
        )

    union_20 = merged["any_variant_oracle_in_top20"]
    union_50 = merged["any_variant_oracle_in_top50"]

    policy = {
        "stage": "S8.11E",
        "status": "PASS_SCALE_STRIDE_COMPARISON",
        "created_at_utc": now_utc(),
        "baseline_protocol": TAG,
        "primary_lightglue_pool": {
            "variant": primary_variant,
            "top_k": primary_k,
            "reason": [
                "highest Recall@5 among independent variants",
                "highest Recall@10 among independent variants",
                "Recall@20 reaches 98.26 percent",
                "median first oracle rank is 3",
                "only 108 map tiles",
                "lower verification cost than dense 1024_s256",
            ],
            "query_count": int(len(merged)),
            "oracle_hits_within_pool": primary_hits,
            "oracle_recall_within_pool": primary_recall,
        },
        "diagnostic_alternates": alternates,
        "union_diagnostics": {
            "any_variant_top20_hits": int(union_20.sum()),
            "any_variant_top20_recall": float(union_20.mean()),
            "any_variant_top50_hits": int(union_50.sum()),
            "any_variant_top50_recall": float(union_50.mean()),
        },
        "hard_queries": {
            "all_variants_oracle_rank_gt20_count": int(
                merged["hard_all_variants_rank_gt20"].sum()
            ),
            "all_variants_oracle_rank_gt20_query_ids": (
                merged.loc[
                    merged["hard_all_variants_rank_gt20"],
                    "query_id",
                ]
                .astype(str)
                .tolist()
            ),
            "all_variants_top1_error_gt120m_count": int(
                merged[
                    "hard_all_variants_top1_error_gt120m"
                ].sum()
            ),
        },
        "next_stage": {
            "stage": "S8.12",
            "first_run": "LightGlue verification on 1024_s512 Top-20",
            "do_not_fuse_before": (
                "Primary independent LightGlue verification is complete"
            ),
        },
    }

    return policy


def build_markdown_report(
    summary: pd.DataFrame,
    rescue_tables: dict[int, pd.DataFrame],
    policy: dict,
    figure_paths: list[Path],
) -> str:
    indexed = summary.set_index("variant")

    lines = [
        "# S8.11E — DINOv2 Scale/Stride Comparison",
        "",
        f"Generated: {now_utc()}",
        "",
        "## Frozen input",
        "",
        f"- Descriptor/retrieval protocol: `{TAG}`",
        "- Query count: 115",
        "- Descriptor dimension: 384",
        "- Retrieval: cosine similarity over L2-normalized descriptors",
        "- Coordinates and oracle IDs were used only after ranking for evaluation.",
        "",
        "## Independent retrieval comparison",
        "",
        "| Variant | Tiles | R@1 | R@5 | R@10 | R@20 | R@50 | Median oracle rank | Median Top-1 error |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for variant in VARIANTS:
        row = indexed.loc[variant]
        lines.append(
            f"| `{variant}` "
            f"| {int(row['tile_count'])} "
            f"| {float(row['recall_at_1']):.3f} "
            f"| {float(row['recall_at_5']):.3f} "
            f"| {float(row['recall_at_10']):.3f} "
            f"| {float(row['recall_at_20']):.3f} "
            f"| {float(row['recall_at_50']):.3f} "
            f"| {float(row['first_oracle_rank_median']):.1f} "
            f"| {float(row['top1_center_error_median_m']):.2f} m |"
        )

    primary = policy["primary_lightglue_pool"]

    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"Primary LightGlue candidate pool: "
            f"`{primary['variant']}@{primary['top_k']}`.",
            "",
            f"This pool contains an oracle tile for "
            f"{primary['oracle_hits_within_pool']}/"
            f"{primary['query_count']} queries "
            f"({primary['oracle_recall_within_pool']:.3f} recall).",
            "",
            "The `1024_s512` database is preferred because its larger footprint "
            "provides useful structural context without introducing as many "
            "overlapping near-duplicate tiles as `1024_s256`.",
            "",
            "## Important limitation",
            "",
            "The current result proves candidate-generation quality only. "
            "It does not prove geometric consistency or final localization. "
            "LightGlue verification is required next.",
            "",
            "## Pairwise rescue diagnostics",
            "",
        ]
    )

    for k, rescue_df in rescue_tables.items():
        lines.append(f"### Top-{k}")
        lines.append("")
        lines.append(
            "| Failed source | Rescuer | Source failures | Rescued | Rescue rate |"
        )
        lines.append("|---|---|---:|---:|---:|")

        for _, row in rescue_df.iterrows():
            if row["source_variant"] == row["rescuer_variant"]:
                continue

            lines.append(
                f"| `{row['source_variant']}` "
                f"| `{row['rescuer_variant']}` "
                f"| {int(row['source_failures'])} "
                f"| {int(row['rescued_queries'])} "
                f"| {float(row['rescue_rate_among_source_failures']):.3f} |"
            )

        lines.append("")

    lines.extend(
        [
            "## Figures",
            "",
        ]
    )

    for path in figure_paths:
        relative = path.relative_to(ROOT)
        lines.append(f"- `{relative}`")

    lines.extend(
        [
            "",
            "## Next stage",
            "",
            "Run S8.12 first on:",
            "",
            "```text",
            "variant = 1024_s512",
            "candidate depth = Top-20",
            "verifier = LightGlue",
            "```",
            "",
            "Fusion or fallback to other tile variants should be tested only "
            "after the independent primary verifier baseline is measured.",
        ]
    )

    return "\n".join(lines)


def main() -> None:
    print("S8.11E — Scale/Stride Comparison")
    print("--------------------------------")

    ensure_inputs()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    summary = pd.read_csv(SUMMARY_INPUT)
    summary["variant"] = summary["variant"].astype(str)

    missing_variants = set(VARIANTS) - set(summary["variant"])
    if missing_variants:
        raise ValueError(
            f"Missing variants in summary: {sorted(missing_variants)}"
        )

    summary = (
        summary.set_index("variant")
        .loc[VARIANTS]
        .reset_index()
    )

    tables = load_query_tables()
    merged = build_merged_query_table(tables)
    recall_df = build_recall_table(summary)

    merged_path = (
        REPORT_DIR
        / f"s8_11e_query_level_scale_stride_diagnostics_{TAG}.csv"
    )
    recall_path = (
        REPORT_DIR
        / f"s8_11e_recall_at_k_long_{TAG}.csv"
    )

    merged.to_csv(merged_path, index=False)
    recall_df.to_csv(recall_path, index=False)

    rescue_tables: dict[int, pd.DataFrame] = {}

    for k in [5, 10, 20]:
        rescue = build_pairwise_rescue_table(merged, k)
        rescue_tables[k] = rescue

        rescue.to_csv(
            REPORT_DIR
            / f"s8_11e_pairwise_rescue_top{k}_{TAG}.csv",
            index=False,
        )

    figures = [
        plot_recall_at_k(recall_df),
        plot_oracle_rank_boxplot(tables),
        plot_rank_by_query(merged),
        plot_top1_error_ecdf(tables),
    ]

    policy = choose_candidate_policy(summary, merged)

    policy_path = (
        REPORT_DIR
        / f"s8_11e_candidate_pool_policy_{TAG}.json"
    )
    policy_path.write_text(
        json.dumps(policy, indent=2),
        encoding="utf-8",
    )

    report_text = build_markdown_report(
        summary=summary,
        rescue_tables=rescue_tables,
        policy=policy,
        figure_paths=figures,
    )

    report_path = (
        REPORT_DIR
        / f"README_s8_11e_scale_stride_comparison_{TAG}.md"
    )
    report_path.write_text(report_text, encoding="utf-8")

    compact_columns = [
        "variant",
        "tile_count",
        "recall_at_1",
        "recall_at_5",
        "recall_at_10",
        "recall_at_20",
        "recall_at_50",
        "first_oracle_rank_median",
        "top1_center_error_median_m",
        "top1_center_error_rmse_m",
    ]

    print()
    print("Independent comparison:")
    print(summary[compact_columns].to_string(index=False))

    print()
    print("Union diagnostics:")
    print(
        "Any variant Top-20:",
        policy["union_diagnostics"]["any_variant_top20_hits"],
        "/",
        len(merged),
        f"({policy['union_diagnostics']['any_variant_top20_recall']:.3f})",
    )
    print(
        "Any variant Top-50:",
        policy["union_diagnostics"]["any_variant_top50_hits"],
        "/",
        len(merged),
        f"({policy['union_diagnostics']['any_variant_top50_recall']:.3f})",
    )

    print()
    print("Primary candidate policy:")
    print(
        policy["primary_lightglue_pool"]["variant"],
        "Top-",
        policy["primary_lightglue_pool"]["top_k"],
        sep="",
    )

    print()
    print("Hard queries:")
    print(
        "All variants oracle rank >20:",
        policy["hard_queries"][
            "all_variants_oracle_rank_gt20_count"
        ],
    )
    print(
        "All variants Top-1 error >120m:",
        policy["hard_queries"][
            "all_variants_top1_error_gt120m_count"
        ],
    )

    print()
    print("--------------------------------")
    print("S8.11E COMPLETE")
    print("STATUS: PASS_SCALE_STRIDE_COMPARISON")
    print("Merged diagnostics:", merged_path)
    print("Candidate policy:", policy_path)
    print("README:", report_path)

    print()
    print("Figures:")
    for figure in figures:
        print(figure)


if __name__ == "__main__":
    main()
