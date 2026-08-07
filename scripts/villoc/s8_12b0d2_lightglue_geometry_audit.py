'''
Command Executed:

python scripts/villoc/s8_12b0d2_lightglue_geometry_audit.py \
  2>&1 | tee \
  outputs/villoc/90_deg/logs/s8_12b0d2_lightglue_geometry_audit.log

'''

from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path.cwd().resolve()

RUN_NAME = "s8_12b0_villoc_1024_s512_top20_smoke3"

CANDIDATE_CSV = (
    ROOT
    / "outputs/villoc/90_deg/metadata/s7d_lightglue"
    / f"s7d1_lightglue_candidate_scores_{RUN_NAME}.csv"
)

RANKED_CSV = (
    ROOT
    / "outputs/villoc/90_deg/metadata/s7d_lightglue"
    / f"s7d1_lightglue_candidate_scores_ranked_{RUN_NAME}.csv"
)

ORACLE_CSV = (
    ROOT
    / "outputs/villoc/90_deg/metadata"
    / "s8_10b_uav_tile_oracle_1024_s512.csv"
)

SELECTION_JSON = (
    ROOT
    / "outputs/villoc/90_deg/reports/s8_12b"
    / "s8_12b0_smoke_query_selection.json"
)

OUT_DIR = (
    ROOT
    / "outputs/villoc/90_deg/reports/s8_12b"
)

FIG_DIR = (
    ROOT
    / "outputs/villoc/90_deg/figures/s8_12b"
)

PAIR_OUT = (
    OUT_DIR
    / "s8_12b0d2_lightglue_pair_geometry_audit.csv"
)

QUERY_OUT = (
    OUT_DIR
    / "s8_12b0d2_lightglue_query_geometry_summary.csv"
)

GROUP_OUT = (
    OUT_DIR
    / "s8_12b0d2_oracle_vs_nonoracle_geometry_summary.csv"
)

REPORT_OUT = (
    OUT_DIR
    / "s8_12b0d2_lightglue_geometry_audit.json"
)

FIG_SCORE = (
    FIG_DIR
    / "s8_12b0d2_score_vs_inliers.png"
)

FIG_COVERAGE = (
    FIG_DIR
    / "s8_12b0d2_inlier_ratio_vs_coverage.png"
)

FIG_QUERY = (
    FIG_DIR
    / "s8_12b0d2_query_candidate_geometry.png"
)


def normalize_id(value: Any) -> str:
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass

    text = str(value).strip()

    if re.fullmatch(r"-?\d+\.0", text):
        return str(int(float(text)))

    return text


def find_column(
    df: pd.DataFrame,
    candidates: list[str],
    label: str,
    required: bool = True,
) -> str | None:
    lower = {
        str(column).lower(): column
        for column in df.columns
    }

    for candidate in candidates:
        if candidate.lower() in lower:
            return lower[candidate.lower()]

    if required:
        raise KeyError(
            f"{label}: none of {candidates} found.\n"
            f"Available columns:\n{list(df.columns)}"
        )

    return None


def numeric(
    df: pd.DataFrame,
    column: str | None,
    default: float = np.nan,
) -> pd.Series:
    if column is None or column not in df.columns:
        return pd.Series(
            [default] * len(df),
            index=df.index,
            dtype=float,
        )

    return pd.to_numeric(
        df[column],
        errors="coerce",
    )


def truthy(value: Any) -> bool:
    if value is None:
        return False

    try:
        if pd.isna(value):
            return False
    except Exception:
        pass

    if isinstance(value, str):
        return value.strip().lower() in {
            "true",
            "1",
            "yes",
            "y",
        }

    return bool(value)


def parse_tile_set(value: Any) -> set[str]:
    if value is None:
        return set()

    try:
        if pd.isna(value):
            return set()
    except Exception:
        pass

    if isinstance(value, (list, tuple, set, np.ndarray)):
        return {
            normalize_id(item)
            for item in value
            if normalize_id(item)
        }

    text = str(value).strip()

    if not text or text.lower() in {
        "nan",
        "none",
        "null",
        "[]",
    }:
        return set()

    try:
        parsed = ast.literal_eval(text)

        if isinstance(parsed, (list, tuple, set)):
            return {
                normalize_id(item)
                for item in parsed
                if normalize_id(item)
            }
    except Exception:
        pass

    cleaned = re.sub(
        r"[\[\]\(\)\{\}\"']",
        "",
        text,
    )

    return {
        normalize_id(part)
        for part in re.split(r"[,;|\s]+", cleaned)
        if normalize_id(part)
    }


def load_roles() -> dict[str, str]:
    data = json.loads(
        SELECTION_JSON.read_text(encoding="utf-8")
    )

    return {
        normalize_id(row["query_id"]): str(row["role"])
        for row in data["queries"]
    }


def load_oracle_sets() -> dict[str, set[str]]:
    df = pd.read_csv(ORACLE_CSV)

    query_col = find_column(
        df,
        ["query_id", "token0_id", "sample_id"],
        "oracle query ID",
    )

    packed_col = find_column(
        df,
        ["oracle_tile_ids"],
        "packed oracle tile IDs",
        required=False,
    )

    single_col = find_column(
        df,
        [
            "oracle_tile_id",
            "tile_id",
            "sat_tile_id",
        ],
        "single oracle tile ID",
        required=False,
    )

    df["_query_id"] = df[query_col].map(normalize_id)

    result: dict[str, set[str]] = {}

    for _, row in df.iterrows():
        query_id = row["_query_id"]

        if not query_id:
            continue

        result.setdefault(query_id, set())

        if packed_col is not None:
            result[query_id].update(
                parse_tile_set(row[packed_col])
            )

        if single_col is not None:
            tile_id = normalize_id(row[single_col])

            if tile_id:
                result[query_id].add(tile_id)

    return result


def classify_pair(row: pd.Series) -> str:
    matches = float(row["matches"])
    inliers = float(row["inliers"])
    ratio = float(row["inlier_ratio"])
    min_cov = float(row["min_coverage"])
    h_ok = bool(row["homography_success"])

    # These are diagnostic labels, not final acceptance thresholds.
    if not h_ok or inliers < 4:
        return "degenerate"

    if (
        inliers >= 15
        and ratio >= 0.25
        and min_cov >= 0.20
    ):
        return "strong_distributed"

    if (
        inliers >= 10
        and ratio >= 0.15
        and min_cov >= 0.10
    ):
        return "moderate"

    if (
        inliers >= 6
        and ratio >= 0.08
        and min_cov >= 0.05
    ):
        return "weak_localized"

    if matches >= 10:
        return "many_matches_low_consistency"

    return "insufficient"


def median_or_none(series: pd.Series) -> float | None:
    values = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()

    if values.empty:
        return None

    return float(values.median())


def main() -> None:
    print("S8.12B.0D.2 — LightGlue Geometry Audit")
    print("--------------------------------------")

    for path in [
        CANDIDATE_CSV,
        RANKED_CSV,
        ORACLE_CSV,
        SELECTION_JSON,
    ]:
        if not path.exists():
            raise FileNotFoundError(path)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    candidates = pd.read_csv(CANDIDATE_CSV)
    ranked = pd.read_csv(RANKED_CSV)

    roles = load_roles()
    oracle_sets = load_oracle_sets()

    query_col = find_column(
        ranked,
        ["query_id", "token0_id"],
        "query ID",
    )

    tile_col = find_column(
        ranked,
        ["tile_id", "sat_tile_id"],
        "tile ID",
    )

    candidate_rank_col = find_column(
        ranked,
        [
            "candidate_rank",
            "retrieval_rank",
            "rank",
        ],
        "candidate rank",
    )

    lg_rank_col = find_column(
        ranked,
        [
            "lightglue_rank",
            "lg_rank",
        ],
        "LightGlue rank",
    )

    status_col = find_column(
        ranked,
        ["lightglue_status"],
        "LightGlue status",
    )

    matches_col = find_column(
        ranked,
        ["lightglue_matches"],
        "match count",
    )

    inliers_col = find_column(
        ranked,
        ["lightglue_ransac_inliers"],
        "RANSAC inlier count",
    )

    ratio_col = find_column(
        ranked,
        ["lightglue_inlier_ratio"],
        "inlier ratio",
    )

    h_col = find_column(
        ranked,
        ["lightglue_homography_success"],
        "homography success",
    )

    uav_cov_col = find_column(
        ranked,
        ["lightglue_uav_coverage"],
        "UAV coverage",
    )

    sat_cov_col = find_column(
        ranked,
        ["lightglue_sat_coverage"],
        "satellite coverage",
    )

    score_col = find_column(
        ranked,
        ["lightglue_score"],
        "LightGlue score",
    )

    eval_col = find_column(
        ranked,
        [
            "eval_error_m",
            "center_error_m",
            "tile_center_error_m",
        ],
        "evaluation error",
        required=False,
    )

    runtime_col = find_column(
        ranked,
        ["runtime_s"],
        "runtime",
        required=False,
    )

    df = ranked.copy()

    df["query_id"] = df[query_col].map(normalize_id)
    df["tile_id"] = df[tile_col].map(normalize_id)
    df["role"] = df["query_id"].map(roles).fillna("unknown")

    df["candidate_rank"] = numeric(df, candidate_rank_col)
    df["lightglue_rank"] = numeric(df, lg_rank_col)

    df["matches"] = numeric(df, matches_col, 0).fillna(0)
    df["inliers"] = numeric(df, inliers_col, 0).fillna(0)
    df["inlier_ratio"] = numeric(df, ratio_col, 0).fillna(0)

    df["homography_success"] = (
        df[h_col].map(truthy)
    )

    df["uav_coverage"] = numeric(
        df,
        uav_cov_col,
        0,
    ).fillna(0)

    df["sat_coverage"] = numeric(
        df,
        sat_cov_col,
        0,
    ).fillna(0)

    df["min_coverage"] = np.minimum(
        df["uav_coverage"],
        df["sat_coverage"],
    )

    df["lightglue_score"] = numeric(
        df,
        score_col,
        -1,
    ).fillna(-1)

    df["eval_error_m"] = numeric(
        df,
        eval_col,
    )

    df["runtime_s"] = numeric(
        df,
        runtime_col,
    )

    df["status_ok"] = (
        df[status_col].astype(str).str.lower() == "ok"
    )

    df["is_geometric_oracle"] = [
        tile_id in oracle_sets.get(query_id, set())
        for query_id, tile_id in zip(
            df["query_id"],
            df["tile_id"],
        )
    ]

    df["match_survival_ratio"] = np.where(
        df["matches"] > 0,
        df["inliers"] / df["matches"],
        0.0,
    )

    # Confirm stored ratio and recomputed survival ratio agree.
    df["ratio_difference"] = (
        df["inlier_ratio"]
        - df["match_survival_ratio"]
    ).abs()

    df["geometry_class"] = df.apply(
        classify_pair,
        axis=1,
    )

    # Diagnostic online-safe acceptance proxy.
    df["passes_geometry_gate"] = (
        df["homography_success"]
        & (df["inliers"] >= 10)
        & (df["inlier_ratio"] >= 0.15)
        & (df["min_coverage"] >= 0.10)
    )

    pair_columns = [
        "role",
        "query_id",
        "tile_id",
        "candidate_rank",
        "lightglue_rank",
        "is_geometric_oracle",
        "status_ok",
        "matches",
        "inliers",
        "inlier_ratio",
        "match_survival_ratio",
        "ratio_difference",
        "homography_success",
        "uav_coverage",
        "sat_coverage",
        "min_coverage",
        "lightglue_score",
        "eval_error_m",
        "runtime_s",
        "geometry_class",
        "passes_geometry_gate",
    ]

    df[pair_columns].sort_values(
        ["query_id", "lightglue_rank"],
    ).to_csv(
        PAIR_OUT,
        index=False,
    )

    group_rows = []

    for label, group in df.groupby(
        "is_geometric_oracle",
        dropna=False,
    ):
        group_rows.append(
            {
                "group": (
                    "oracle"
                    if bool(label)
                    else "non_oracle"
                ),
                "pairs": int(len(group)),
                "homography_success_rate": float(
                    group["homography_success"].mean()
                ),
                "median_matches": median_or_none(
                    group["matches"]
                ),
                "median_inliers": median_or_none(
                    group["inliers"]
                ),
                "median_inlier_ratio": median_or_none(
                    group["inlier_ratio"]
                ),
                "median_min_coverage": median_or_none(
                    group["min_coverage"]
                ),
                "median_lightglue_score": median_or_none(
                    group["lightglue_score"]
                ),
                "geometry_gate_rate": float(
                    group["passes_geometry_gate"].mean()
                ),
            }
        )

    group_summary = pd.DataFrame(group_rows)
    group_summary.to_csv(GROUP_OUT, index=False)

    query_rows = []

    for query_id, group in df.groupby(
        "query_id",
        sort=True,
    ):
        g = group.copy()

        lg_top = g.sort_values(
            "lightglue_rank",
            kind="mergesort",
        ).iloc[0]

        retrieval_top = g.sort_values(
            "candidate_rank",
            kind="mergesort",
        ).iloc[0]

        oracle = g[g["is_geometric_oracle"]].copy()

        best_oracle = (
            oracle.sort_values(
                "lightglue_rank",
                kind="mergesort",
            ).iloc[0]
            if not oracle.empty
            else None
        )

        nonoracle = g[
            ~g["is_geometric_oracle"]
        ].copy()

        best_nonoracle = (
            nonoracle.sort_values(
                "lightglue_rank",
                kind="mergesort",
            ).iloc[0]
            if not nonoracle.empty
            else None
        )

        oracle_best_score = (
            float(best_oracle["lightglue_score"])
            if best_oracle is not None
            else None
        )

        nonoracle_best_score = (
            float(best_nonoracle["lightglue_score"])
            if best_nonoracle is not None
            else None
        )

        score_margin = (
            oracle_best_score - nonoracle_best_score
            if (
                oracle_best_score is not None
                and nonoracle_best_score is not None
            )
            else None
        )

        query_rows.append(
            {
                "role": roles.get(query_id, "unknown"),
                "query_id": query_id,
                "processed_candidates": int(len(g)),
                "geometric_oracle_candidates": int(
                    g["is_geometric_oracle"].sum()
                ),
                "retrieval_top_tile_id": (
                    retrieval_top["tile_id"]
                ),
                "retrieval_top_is_oracle": bool(
                    retrieval_top["is_geometric_oracle"]
                ),
                "lightglue_top_tile_id": (
                    lg_top["tile_id"]
                ),
                "lightglue_top_is_oracle": bool(
                    lg_top["is_geometric_oracle"]
                ),
                "lightglue_top_candidate_rank": float(
                    lg_top["candidate_rank"]
                ),
                "lightglue_top_matches": float(
                    lg_top["matches"]
                ),
                "lightglue_top_inliers": float(
                    lg_top["inliers"]
                ),
                "lightglue_top_inlier_ratio": float(
                    lg_top["inlier_ratio"]
                ),
                "lightglue_top_min_coverage": float(
                    lg_top["min_coverage"]
                ),
                "lightglue_top_homography_success": bool(
                    lg_top["homography_success"]
                ),
                "lightglue_top_geometry_class": (
                    lg_top["geometry_class"]
                ),
                "lightglue_top_passes_geometry_gate": bool(
                    lg_top["passes_geometry_gate"]
                ),
                "best_oracle_lightglue_rank": (
                    float(best_oracle["lightglue_rank"])
                    if best_oracle is not None
                    else None
                ),
                "best_oracle_candidate_rank": (
                    float(best_oracle["candidate_rank"])
                    if best_oracle is not None
                    else None
                ),
                "best_oracle_matches": (
                    float(best_oracle["matches"])
                    if best_oracle is not None
                    else None
                ),
                "best_oracle_inliers": (
                    float(best_oracle["inliers"])
                    if best_oracle is not None
                    else None
                ),
                "best_oracle_inlier_ratio": (
                    float(best_oracle["inlier_ratio"])
                    if best_oracle is not None
                    else None
                ),
                "best_oracle_min_coverage": (
                    float(best_oracle["min_coverage"])
                    if best_oracle is not None
                    else None
                ),
                "best_oracle_geometry_class": (
                    str(best_oracle["geometry_class"])
                    if best_oracle is not None
                    else None
                ),
                "best_oracle_score": oracle_best_score,
                "best_nonoracle_score": nonoracle_best_score,
                "oracle_score_margin_over_best_nonoracle": (
                    score_margin
                ),
            }
        )

    query_summary = pd.DataFrame(query_rows)
    query_summary.to_csv(QUERY_OUT, index=False)

    # Figure 1: score and inliers.
    plt.figure(figsize=(9, 6))

    for is_oracle, group in df.groupby(
        "is_geometric_oracle"
    ):
        label = (
            "Geometric oracle"
            if is_oracle
            else "Non-oracle"
        )

        plt.scatter(
            group["inliers"],
            group["lightglue_score"],
            label=label,
            alpha=0.75,
        )

    plt.xlabel("RANSAC inliers")
    plt.ylabel("LightGlue verifier score")
    plt.title(
        "S8.12B.0D.2 — Score versus RANSAC support"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_SCORE, dpi=180)
    plt.close()

    # Figure 2: ratio and coverage.
    plt.figure(figsize=(9, 6))

    for is_oracle, group in df.groupby(
        "is_geometric_oracle"
    ):
        label = (
            "Geometric oracle"
            if is_oracle
            else "Non-oracle"
        )

        plt.scatter(
            group["inlier_ratio"],
            group["min_coverage"],
            label=label,
            alpha=0.75,
        )

    plt.axvline(0.15, linestyle="--")
    plt.axhline(0.10, linestyle="--")
    plt.xlabel("RANSAC inlier ratio")
    plt.ylabel("Minimum UAV/satellite coverage")
    plt.title(
        "S8.12B.0D.2 — Consistency versus spatial spread"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_COVERAGE, dpi=180)
    plt.close()

    # Figure 3: per-query candidate geometry.
    plot_df = df.sort_values(
        ["query_id", "candidate_rank"]
    ).copy()

    x = np.arange(len(plot_df))

    plt.figure(figsize=(14, 6))

    plt.scatter(
        x,
        plot_df["inliers"],
        s=45,
        alpha=0.8,
    )

    oracle_indices = np.flatnonzero(
        plot_df["is_geometric_oracle"].to_numpy()
    )

    plt.scatter(
        oracle_indices,
        plot_df.iloc[oracle_indices]["inliers"],
        s=110,
        facecolors="none",
        edgecolors="black",
        linewidths=1.5,
        label="Geometric oracle",
    )

    boundaries = (
        plot_df.groupby("query_id")
        .size()
        .cumsum()
        .to_numpy()[:-1]
    )

    for boundary in boundaries:
        plt.axvline(
            boundary - 0.5,
            linestyle="--",
            linewidth=1,
        )

    plt.xlabel(
        "Candidates ordered by query and DINOv2 rank"
    )
    plt.ylabel("RANSAC inliers")
    plt.title(
        "S8.12B.0D.2 — Inlier support across 60 smoke pairs"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_QUERY, dpi=180)
    plt.close()

    top_geometry_passes = int(
        query_summary[
            "lightglue_top_passes_geometry_gate"
        ].sum()
    )

    top_oracle_count = int(
        query_summary[
            "lightglue_top_is_oracle"
        ].sum()
    )

    all_ok = bool(df["status_ok"].all())

    ratio_consistent = bool(
        (df["ratio_difference"] <= 1e-5).all()
    )

    oracle_median_score = median_or_none(
        df.loc[
            df["is_geometric_oracle"],
            "lightglue_score",
        ]
    )

    nonoracle_median_score = median_or_none(
        df.loc[
            ~df["is_geometric_oracle"],
            "lightglue_score",
        ]
    )

    oracle_median_inliers = median_or_none(
        df.loc[
            df["is_geometric_oracle"],
            "inliers",
        ]
    )

    nonoracle_median_inliers = median_or_none(
        df.loc[
            ~df["is_geometric_oracle"],
            "inliers",
        ]
    )

    # Conservative decision:
    # do not authorize the full benchmark merely because the smoke executed.
    if not all_ok or not ratio_consistent:
        status = "NEEDS_LIGHTGLUE_OUTPUT_FIX"
        decision = (
            "The stored LightGlue metrics are incomplete or internally "
            "inconsistent."
        )
    elif top_geometry_passes == 3 and top_oracle_count >= 2:
        status = "PASS_GEOMETRY_AUDIT_PROCEED_TO_SCALE_SWEEP"
        decision = (
            "The smoke winners have measurable geometric support. "
            "Proceed to the fixed crop-scale sweep before the full run."
        )
    else:
        status = "PASS_AUDIT_GEOMETRY_WEAK_PROCEED_TO_SCALE_SWEEP"
        decision = (
            "The audit completed, but one or more query winners have weak "
            "or spatially localized geometry. Do not launch the full run; "
            "proceed to the fixed satellite crop-scale sweep."
        )

    report = {
        "stage": "S8.12B.0D.2",
        "run_name": RUN_NAME,
        "candidate_pairs": int(len(df)),
        "queries": int(df["query_id"].nunique()),
        "all_status_ok": all_ok,
        "stored_ratio_matches_recomputed_ratio": (
            ratio_consistent
        ),
        "lightglue_top_is_geometric_oracle": (
            top_oracle_count
        ),
        "lightglue_top_passes_diagnostic_geometry_gate": (
            top_geometry_passes
        ),
        "oracle_median_lightglue_score": (
            oracle_median_score
        ),
        "nonoracle_median_lightglue_score": (
            nonoracle_median_score
        ),
        "oracle_median_inliers": (
            oracle_median_inliers
        ),
        "nonoracle_median_inliers": (
            nonoracle_median_inliers
        ),
        "diagnostic_geometry_gate": {
            "homography_success": True,
            "minimum_inliers": 10,
            "minimum_inlier_ratio": 0.15,
            "minimum_minimum_coverage": 0.10,
            "note": (
                "Diagnostic only. These thresholds are not yet a "
                "frozen production confidence gate."
            ),
        },
        "status": status,
        "decision": decision,
        "outputs": {
            "pair_audit_csv": str(PAIR_OUT),
            "query_summary_csv": str(QUERY_OUT),
            "oracle_vs_nonoracle_csv": str(GROUP_OUT),
            "score_figure": str(FIG_SCORE),
            "coverage_figure": str(FIG_COVERAGE),
            "query_figure": str(FIG_QUERY),
        },
    }

    REPORT_OUT.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print()
    print("Oracle versus non-oracle geometry:")
    print(group_summary.to_string(index=False))

    print()
    print("Query-level geometry:")
    display_columns = [
        "role",
        "query_id",
        "retrieval_top_is_oracle",
        "lightglue_top_is_oracle",
        "lightglue_top_candidate_rank",
        "lightglue_top_matches",
        "lightglue_top_inliers",
        "lightglue_top_inlier_ratio",
        "lightglue_top_min_coverage",
        "lightglue_top_geometry_class",
        "best_oracle_lightglue_rank",
        "best_oracle_inliers",
        "best_oracle_inlier_ratio",
        "best_oracle_min_coverage",
        "oracle_score_margin_over_best_nonoracle",
    ]

    print(
        query_summary[
            display_columns
        ].to_string(index=False)
    )

    print()
    print("Audit summary")
    print("-------------")
    print("Pairs:", len(df))
    print("Queries:", df["query_id"].nunique())
    print("All statuses OK:", all_ok)
    print(
        "Stored/recomputed ratios consistent:",
        ratio_consistent,
    )
    print(
        "LG Top-1 geometric oracle:",
        f"{top_oracle_count}/3",
    )
    print(
        "LG Top-1 passes diagnostic geometry gate:",
        f"{top_geometry_passes}/3",
    )
    print(
        "Oracle median inliers:",
        oracle_median_inliers,
    )
    print(
        "Non-oracle median inliers:",
        nonoracle_median_inliers,
    )
    print(
        "Oracle median score:",
        oracle_median_score,
    )
    print(
        "Non-oracle median score:",
        nonoracle_median_score,
    )

    print()
    print("--------------------------------------")
    print("S8.12B.0D.2 COMPLETE")
    print("STATUS:", status)
    print("DECISION:", decision)
    print("Pair audit:", PAIR_OUT)
    print("Query summary:", QUERY_OUT)
    print("Group summary:", GROUP_OUT)
    print("Report:", REPORT_OUT)
    print("Figures:")
    print(FIG_SCORE)
    print(FIG_COVERAGE)
    print(FIG_QUERY)


if __name__ == "__main__":
    main()
