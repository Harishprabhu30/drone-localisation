#!/usr/bin/env python3
"""
S7C.1 — Candidate-generation error taxonomy for SatLoc traj01.

Purpose
-------
Read existing S5C/S6B/S7 metadata and classify absolute retrieval failures into
candidate-pool, selection/verifier, and scene-specific ambiguity categories.

This block is READ-ONLY with respect to retrieval:
- no new candidates are generated
- no LightGlue rerun is performed
- no fusion/correction replay is performed

Ground-truth/reference rule
---------------------------
Evaluation/error columns are used only after the frozen S5C/S6B outputs already
exist, to label and audit failures. They must not be used by later S7C.2 online
candidate generation or ranking.

Expected primary inputs
-----------------------
outputs/satloc/metadata/s7_retrieval_upgrade/s7_0_query_manifest.csv
outputs/satloc/metadata/s7_retrieval_upgrade/s7_0_scene_sampled_frames.csv
outputs/satloc/metadata/s5c_temporal/s5c2_lightglue_union_candidate_scores_top50_full263.csv
outputs/satloc/metadata/s5c_temporal/s5c2_lightglue_union_query_summary_top50_full263.csv
outputs/satloc/metadata/s6b_relative_absolute/s6b0_absolute_correction_manifest.csv
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_QUERY_MANIFEST = Path(
    "outputs/satloc/metadata/s7_retrieval_upgrade/s7_0_query_manifest.csv"
)
DEFAULT_SCENE_SAMPLES = Path(
    "outputs/satloc/metadata/s7_retrieval_upgrade/s7_0_scene_sampled_frames.csv"
)
DEFAULT_CANDIDATE_SCORES = Path(
    "outputs/satloc/metadata/s5c_temporal/"
    "s5c2_lightglue_union_candidate_scores_top50_full263.csv"
)
DEFAULT_QUERY_SUMMARY = Path(
    "outputs/satloc/metadata/s5c_temporal/"
    "s5c2_lightglue_union_query_summary_top50_full263.csv"
)
DEFAULT_S6B_MANIFEST = Path(
    "outputs/satloc/metadata/s6b_relative_absolute/"
    "s6b0_absolute_correction_manifest.csv"
)

DEFAULT_METADATA_OUT = Path("outputs/satloc/metadata/s7_retrieval_upgrade")
DEFAULT_REPORT_OUT = Path("outputs/satloc/reports/s7_retrieval_upgrade")
DEFAULT_FIGURE_OUT = Path("outputs/satloc/figures/s7_retrieval_upgrade")

SCENE_ALIASES = [
    "primary_scene",
    "scene_label",
    "scene_group",
    "scene_type",
    "final_scene_label",
    "taxonomy_label",
]

TOKEN_ALIASES = [
    "token",
    "token0_id",
    "uav_token",
    "frame_token",
    "query_token",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", type=Path, default=Path("."))
    p.add_argument("--query-manifest", type=Path, default=DEFAULT_QUERY_MANIFEST)
    p.add_argument("--scene-samples", type=Path, default=DEFAULT_SCENE_SAMPLES)
    p.add_argument("--candidate-scores", type=Path, default=DEFAULT_CANDIDATE_SCORES)
    p.add_argument("--query-summary", type=Path, default=DEFAULT_QUERY_SUMMARY)
    p.add_argument("--s6b-manifest", type=Path, default=DEFAULT_S6B_MANIFEST)
    p.add_argument("--metadata-out", type=Path, default=DEFAULT_METADATA_OUT)
    p.add_argument("--report-out", type=Path, default=DEFAULT_REPORT_OUT)
    p.add_argument("--figure-out", type=Path, default=DEFAULT_FIGURE_OUT)
    p.add_argument("--policy", type=str, default="lightglue_only")
    p.add_argument("--threshold-m", type=float, default=40.0)
    p.add_argument("--near-threshold-m", type=float, default=100.0)
    p.add_argument("--strict", action="store_true")
    return p.parse_args()


def resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def read_csv(path: Path, required: bool = True) -> pd.DataFrame:
    if not path.exists():
        if required:
            raise FileNotFoundError(path)
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def numeric(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series([np.nan] * len(df), index=df.index)
    return pd.to_numeric(df[col], errors="coerce")


def bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes", "y", "t"})
    )


def first_existing_col(df: pd.DataFrame, aliases: Iterable[str]) -> str | None:
    for col in aliases:
        if col in df.columns:
            return col
    return None


def add_token_column(df: pd.DataFrame, name: str) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    token_col = first_existing_col(df, TOKEN_ALIASES)
    if token_col is None:
        raise ValueError(f"{name} has no recognized token column. Columns={list(df.columns)}")

    out = df.copy()
    out["token"] = pd.to_numeric(out[token_col], errors="coerce").astype("Int64")
    if out["token"].isna().any():
        bad = out.loc[out["token"].isna()].head(5)
        raise ValueError(f"{name} has non-numeric tokens. Examples:\n{bad}")
    out["token"] = out["token"].astype(int)
    return out


def require_columns(df: pd.DataFrame, required: list[str], name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}\nAvailable: {list(df.columns)}")


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if pd.isna(value):
        return None
    return value


def choose_scene_column(scene_df: pd.DataFrame) -> str | None:
    return first_existing_col(scene_df, SCENE_ALIASES)


def load_official_queries(path: Path) -> pd.DataFrame:
    df = add_token_column(read_csv(path), "official query manifest")
    keep = ["token"]
    for c in [
        "sequence",
        "sequence_frame_id",
        "frame_index_in_sequence",
        "token1_order",
        "query_source",
        "source",
    ]:
        if c in df.columns and c not in keep:
            keep.append(c)

    out = df[keep].drop_duplicates("token").sort_values("token").reset_index(drop=True)
    if len(out) != 263:
        raise ValueError(f"Expected 263 official S7C/S5C query tokens, found {len(out)}")
    return out


def load_scene_labels(path: Path) -> tuple[pd.DataFrame, str | None]:
    df_raw = read_csv(path, required=False)
    if df_raw.empty:
        return pd.DataFrame(columns=["token", "primary_scene", "scene_is_s7_sample"]), None

    df = add_token_column(df_raw, "scene sampled frames")
    scene_col = choose_scene_column(df)
    if scene_col is None:
        return pd.DataFrame(columns=["token", "primary_scene", "scene_is_s7_sample"]), None

    out = df[["token", scene_col]].copy()
    out = out.rename(columns={scene_col: "primary_scene"})
    out["primary_scene"] = (
        out["primary_scene"]
        .fillna("unlabeled")
        .astype(str)
        .str.strip()
        .replace({"": "unlabeled"})
    )
    out["scene_is_s7_sample"] = True
    out = out.drop_duplicates("token", keep="first")
    return out, scene_col


def load_query_summary(path: Path, policy: str) -> pd.DataFrame:
    df = add_token_column(read_csv(path), "LightGlue query summary")
    require_columns(
        df,
        [
            "policy",
            "chosen_tile_id",
            "chosen_error_m",
            "hit_le_threshold",
            "union_top1_error_m",
            "oracle_processed_error_m",
            "oracle_processed_hit_le_threshold",
            "oracle_union_rank",
            "oracle_lightglue_rank",
        ],
        "LightGlue query summary",
    )

    q = df.loc[df["policy"].astype(str) == policy].copy()
    if len(q) != 263:
        raise ValueError(
            f"Expected 263 query-summary rows for policy={policy!r}, found {len(q)} "
            f"from raw {len(df)} rows."
        )
    if q["token"].duplicated().any():
        dupes = q.loc[q["token"].duplicated(keep=False), ["token", "policy"]].head(10)
        raise ValueError(f"Duplicate policy/token rows in query summary:\n{dupes}")

    q["chosen_error_m"] = numeric(q, "chosen_error_m")
    q["union_top1_error_m"] = numeric(q, "union_top1_error_m")
    q["oracle_processed_error_m"] = numeric(q, "oracle_processed_error_m")
    q["oracle_union_rank"] = numeric(q, "oracle_union_rank")
    q["oracle_lightglue_rank"] = numeric(q, "oracle_lightglue_rank")
    q["hit_le_threshold"] = bool_series(q["hit_le_threshold"])
    q["oracle_processed_hit_le_threshold"] = bool_series(
        q["oracle_processed_hit_le_threshold"]
    )

    return q


def summarize_candidate_scores(path: Path, threshold_m: float) -> pd.DataFrame:
    df = add_token_column(read_csv(path), "candidate scores")
    require_columns(
        df,
        [
            "tile_id",
            "eval_error_m",
            "union_rank",
            "lightglue_rank",
            "lightglue_score",
            "lightglue_ransac_inliers",
            "lightglue_matches",
        ],
        "candidate scores",
    )

    df["eval_error_num"] = numeric(df, "eval_error_m")
    df["union_rank_num"] = numeric(df, "union_rank")
    df["lightglue_rank_num"] = numeric(df, "lightglue_rank")
    df["lightglue_score_num"] = numeric(df, "lightglue_score")
    df["lightglue_inliers_num"] = numeric(df, "lightglue_ransac_inliers")
    df["lightglue_matches_num"] = numeric(df, "lightglue_matches")

    rows: list[dict[str, Any]] = []
    for token, g in df.groupby("token", dropna=False):
        valid = g.dropna(subset=["eval_error_num"])
        recoverable = valid.loc[valid["eval_error_num"] <= threshold_m]
        near100 = valid.loc[valid["eval_error_num"] <= 100.0]

        top_union = g.sort_values("union_rank_num", kind="mergesort").iloc[0]
        top_lg = g.sort_values("lightglue_rank_num", kind="mergesort").iloc[0]

        best = valid.sort_values("eval_error_num", kind="mergesort").iloc[0] if len(valid) else None
        best_recoverable = (
            recoverable.sort_values("union_rank_num", kind="mergesort").iloc[0]
            if len(recoverable)
            else None
        )

        rows.append(
            {
                "token": int(token),
                "candidate_rows": int(len(g)),
                "candidate_unique_tiles": int(g["tile_id"].astype(str).nunique()),
                "candidate_error_min_m_eval_only": (
                    safe_float(best.get("eval_error_m")) if best is not None else np.nan
                ),
                "candidate_error_median_m_eval_only": (
                    float(valid["eval_error_num"].median()) if len(valid) else np.nan
                ),
                "candidate_oracle_le_threshold_eval_only": bool(len(recoverable) > 0),
                "candidate_oracle_le_100m_eval_only": bool(len(near100) > 0),
                "candidate_first_correct_union_rank_eval_only": (
                    safe_float(best_recoverable.get("union_rank")) if best_recoverable is not None else np.nan
                ),
                "candidate_first_correct_lg_rank_eval_only": (
                    safe_float(best_recoverable.get("lightglue_rank")) if best_recoverable is not None else np.nan
                ),
                "candidate_oracle_tile_id_eval_only": (
                    str(best.get("tile_id")) if best is not None else ""
                ),
                "union_top1_tile_id": str(top_union.get("tile_id")),
                "union_top1_error_m_eval_only": safe_float(top_union.get("eval_error_m")),
                "lightglue_top1_tile_id": str(top_lg.get("tile_id")),
                "lightglue_top1_error_m_eval_only": safe_float(top_lg.get("eval_error_m")),
                "lightglue_top1_score": safe_float(top_lg.get("lightglue_score")),
                "lightglue_top1_inliers": safe_float(top_lg.get("lightglue_ransac_inliers")),
                "lightglue_top1_matches": safe_float(top_lg.get("lightglue_matches")),
                "lightglue_top1_min_coverage": safe_float(
                    top_lg.get("min_cov_num", top_lg.get("chosen_min_coverage", np.nan))
                ),
            }
        )

    return pd.DataFrame(rows)


def load_s6b_optional(path: Path) -> pd.DataFrame:
    df_raw = read_csv(path, required=False)
    if df_raw.empty:
        return pd.DataFrame(columns=["token"])

    df = add_token_column(df_raw, "S6B correction manifest")
    keep = ["token"]
    for c in [
        "balanced_accept_online",
        "permissive_accept_online",
        "hit_eval_only",
        "dangerous_false_eval_only",
        "chosen_error_m_eval_only",
        "chosen_lg_score",
        "chosen_inliers",
        "chosen_matches",
        "chosen_min_coverage",
        "lg_score_margin_top1_top2",
        "reference_cumulative_distance_m",
    ]:
        if c in df.columns:
            keep.append(c)

    out = df[keep].drop_duplicates("token", keep="first").copy()
    for c in [
        "balanced_accept_online",
        "permissive_accept_online",
        "hit_eval_only",
        "dangerous_false_eval_only",
    ]:
        if c in out.columns:
            out[c] = bool_series(out[c])
    return out


def classify_group(row: pd.Series, threshold_m: float, near_threshold_m: float) -> tuple[str, str]:
    oracle_err = safe_float(row.get("oracle_processed_error_m"))
    union_top1_err = safe_float(row.get("union_top1_error_m"))

    oracle_hit = bool(row.get("oracle_processed_hit_le_threshold", False))
    chosen_hit = bool(row.get("hit_le_threshold", False))

    if chosen_hit:
        if math.isfinite(union_top1_err) and union_top1_err <= threshold_m:
            return "retrieval_top1_success", "Candidate generation and final selection both succeeded."
        return "lightglue_selection_success", "Candidate was in pool and LightGlue selected a <=threshold tile."

    if oracle_hit:
        oracle_lg_rank = safe_float(row.get("oracle_lightglue_rank"))
        if math.isfinite(oracle_lg_rank) and oracle_lg_rank <= 5:
            return (
                "selection_failure_correct_in_lg_top5",
                "Correct/near candidate was available and ranked close by LightGlue, but final top1 was wrong.",
            )
        return (
            "selection_failure_correct_in_pool",
            "Correct/near candidate was available in Top-50 pool, but verifier/ranking selected wrong tile.",
        )

    if math.isfinite(oracle_err) and oracle_err <= near_threshold_m:
        return (
            "weak_pool_near_candidate",
            "No <=threshold candidate, but a near candidate exists within the relaxed threshold.",
        )

    if math.isfinite(oracle_err):
        return (
            "candidate_pool_failure",
            "No correct or near-correct candidate was available in the processed Top-50 pool.",
        )

    return (
        "unclassified_missing_eval",
        "Insufficient evaluation columns to classify.",
    )


def classify_diagnostic_tags(row: pd.Series) -> list[str]:
    tags: list[str] = []

    scene = str(row.get("primary_scene", "unlabeled")).lower()
    if any(k in scene for k in ["forest", "canopy", "vegetation", "tree", "grass"]):
        tags.append("vegetation_forest_ambiguity")
    if any(k in scene for k in ["field", "agric", "agriculture", "crop", "open"]):
        tags.append("agricultural_field_ambiguity")
    if any(k in scene for k in ["water", "wetland", "pond", "river", "lake"]):
        tags.append("water_wetland_ambiguity")
    if any(k in scene for k in ["urban", "road", "building", "corridor"]):
        tags.append("urban_road_corridor_ambiguity")

    oracle_rank = safe_float(row.get("oracle_union_rank"))
    oracle_lg_rank = safe_float(row.get("oracle_lightglue_rank"))
    chosen_err = safe_float(row.get("chosen_error_m"))
    oracle_err = safe_float(row.get("oracle_processed_error_m"))

    if math.isfinite(oracle_rank) and oracle_rank > 20:
        tags.append("late_correct_candidate_union_rank")
    if math.isfinite(oracle_lg_rank) and oracle_lg_rank > 20:
        tags.append("verifier_ranks_correct_candidate_low")
    if math.isfinite(chosen_err) and chosen_err > 100:
        tags.append("dangerous_or_large_wrong_selection_eval_only")
    if math.isfinite(oracle_err) and 40.0 < oracle_err <= 100.0:
        tags.append("scale_fov_tile_granularity_near_miss")
    if math.isfinite(oracle_err) and oracle_err > 100:
        tags.append("true_tile_missing_or_far_pool")

    if "rotation" in scene:
        tags.append("rotation_mismatch_suspect")
    elif (
        math.isfinite(oracle_rank)
        and oracle_rank <= 20
        and math.isfinite(oracle_lg_rank)
        and oracle_lg_rank > 20
    ):
        tags.append("rotation_or_viewpoint_verifier_suspect")

    if not tags:
        tags.append("generic_structural_ambiguity")

    return sorted(set(tags))


def plot_counts(summary: pd.DataFrame, label_col: str, count_col: str, title: str, out_path: Path) -> None:
    if summary.empty:
        return
    plot_df = summary.sort_values(count_col, ascending=True)
    fig, ax = plt.subplots(figsize=(10, max(4.5, 0.35 * len(plot_df))))
    ax.barh(plot_df[label_col].astype(str), plot_df[count_col])
    ax.set_xlabel("Count")
    ax.set_title(title)
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    root = args.repo_root.resolve()

    metadata_out = resolve(root, args.metadata_out)
    report_out = resolve(root, args.report_out)
    figure_out = resolve(root, args.figure_out)
    for d in [metadata_out, report_out, figure_out]:
        d.mkdir(parents=True, exist_ok=True)

    queries = load_official_queries(resolve(root, args.query_manifest))
    scenes, scene_col = load_scene_labels(resolve(root, args.scene_samples))
    query_summary = load_query_summary(resolve(root, args.query_summary), args.policy)
    candidate_stats = summarize_candidate_scores(resolve(root, args.candidate_scores), args.threshold_m)
    s6b = load_s6b_optional(resolve(root, args.s6b_manifest))

    merged = queries.merge(
        query_summary,
        on="token",
        how="left",
        validate="one_to_one",
        indicator="_query_summary_join",
    )
    if not (merged["_query_summary_join"] == "both").all():
        missing = merged.loc[merged["_query_summary_join"] != "both", ["token"]].head(20)
        raise ValueError(f"Official tokens missing from query summary:\n{missing}")
    merged = merged.drop(columns="_query_summary_join")

    merged = merged.merge(
        candidate_stats,
        on="token",
        how="left",
        validate="one_to_one",
        indicator="_candidate_scores_join",
    )
    if not (merged["_candidate_scores_join"] == "both").all():
        missing = merged.loc[merged["_candidate_scores_join"] != "both", ["token"]].head(20)
        raise ValueError(f"Official tokens missing from candidate scores:\n{missing}")
    merged = merged.drop(columns="_candidate_scores_join")

    if not scenes.empty:
        merged = merged.merge(scenes, on="token", how="left", validate="one_to_one")
    else:
        merged["primary_scene"] = "unlabeled"
        merged["scene_is_s7_sample"] = False

    merged["primary_scene"] = (
        merged["primary_scene"]
        .fillna("unreviewed_temporal")
        .astype(str)
        .str.strip()
        .replace({"": "unreviewed_temporal"})
    )
    merged["scene_is_s7_sample"] = merged.get(
        "scene_is_s7_sample",
        pd.Series(False, index=merged.index),
    ).fillna(False).astype(bool)

    if not s6b.empty and len(s6b.columns) > 1:
        rename = {c: f"s6b_{c}" for c in s6b.columns if c != "token"}
        merged = merged.merge(
            s6b.rename(columns=rename),
            on="token",
            how="left",
            validate="one_to_one",
        )

    groups = merged.apply(
        lambda r: classify_group(r, args.threshold_m, args.near_threshold_m),
        axis=1,
        result_type="expand",
    )
    merged["taxonomy_group"] = groups[0]
    merged["taxonomy_reason"] = groups[1]
    merged["diagnostic_tags"] = merged.apply(
        lambda r: ";".join(classify_diagnostic_tags(r)),
        axis=1,
    )

    merged["chosen_error_sort"] = numeric(merged, "chosen_error_m").fillna(1e12)
    merged["oracle_error_sort"] = numeric(merged, "oracle_processed_error_m").fillna(1e12)
    merged = merged.sort_values(
        ["taxonomy_group", "oracle_error_sort", "chosen_error_sort", "token"],
        kind="mergesort",
    ).reset_index(drop=True)

    group_summary = (
        merged.groupby("taxonomy_group", dropna=False)
        .agg(
            count=("token", "count"),
            median_chosen_error_m=("chosen_error_m", "median"),
            median_oracle_error_m=("oracle_processed_error_m", "median"),
            median_oracle_union_rank=("oracle_union_rank", "median"),
            median_oracle_lightglue_rank=("oracle_lightglue_rank", "median"),
            median_candidate_rows=("candidate_rows", "median"),
            scene_sampled_count=("scene_is_s7_sample", "sum"),
        )
        .reset_index()
        .sort_values(["count", "taxonomy_group"], ascending=[False, True])
    )
    group_summary["rate"] = group_summary["count"] / len(merged)

    scene_summary = (
        merged.groupby(["primary_scene", "taxonomy_group"], dropna=False)
        .agg(
            count=("token", "count"),
            median_chosen_error_m=("chosen_error_m", "median"),
            median_oracle_error_m=("oracle_processed_error_m", "median"),
            median_oracle_union_rank=("oracle_union_rank", "median"),
            median_oracle_lightglue_rank=("oracle_lightglue_rank", "median"),
        )
        .reset_index()
        .sort_values(["primary_scene", "count"], ascending=[True, False])
    )

    tag_rows: list[dict[str, Any]] = []
    for _, row in merged.iterrows():
        for tag in str(row["diagnostic_tags"]).split(";"):
            tag_rows.append(
                {
                    "tag": tag,
                    "token": int(row["token"]),
                    "taxonomy_group": row["taxonomy_group"],
                    "primary_scene": row["primary_scene"],
                    "chosen_error_m": safe_float(row.get("chosen_error_m")),
                    "oracle_processed_error_m": safe_float(row.get("oracle_processed_error_m")),
                }
            )
    tag_df = pd.DataFrame(tag_rows)
    tag_summary = (
        tag_df.groupby("tag", dropna=False)
        .agg(
            count=("token", "count"),
            unique_tokens=("token", "nunique"),
            median_chosen_error_m=("chosen_error_m", "median"),
            median_oracle_error_m=("oracle_processed_error_m", "median"),
        )
        .reset_index()
        .sort_values(["unique_tokens", "tag"], ascending=[False, True])
    )

    shortlist_groups = {
        "candidate_pool_failure",
        "weak_pool_near_candidate",
        "selection_failure_correct_in_pool",
        "selection_failure_correct_in_lg_top5",
    }
    shortlist = merged.loc[merged["taxonomy_group"].isin(shortlist_groups)].copy()
    shortlist["priority_score"] = 0.0
    shortlist.loc[shortlist["taxonomy_group"] == "selection_failure_correct_in_lg_top5", "priority_score"] += 4
    shortlist.loc[shortlist["taxonomy_group"] == "weak_pool_near_candidate", "priority_score"] += 3
    shortlist.loc[shortlist["taxonomy_group"] == "selection_failure_correct_in_pool", "priority_score"] += 2
    shortlist.loc[shortlist["scene_is_s7_sample"], "priority_score"] += 1
    shortlist["priority_score"] += 1.0 / np.maximum(1.0, numeric(shortlist, "oracle_union_rank").fillna(999.0))
    shortlist = shortlist.sort_values(
        ["priority_score", "oracle_error_sort", "token"],
        ascending=[False, True, True],
        kind="mergesort",
    )

    taxonomy_out = metadata_out / "s7c1_candidate_generation_taxonomy_by_query.csv"
    group_out = metadata_out / "s7c1_taxonomy_group_summary.csv"
    scene_out = metadata_out / "s7c1_scene_taxonomy_summary.csv"
    tag_out = metadata_out / "s7c1_diagnostic_tag_summary.csv"
    shortlist_out = metadata_out / "s7c1_failure_token_shortlist.csv"
    report_json = report_out / "s7c1_candidate_generation_taxonomy_summary.json"
    group_fig = figure_out / "s7c1_taxonomy_group_counts.png"
    scene_fig = figure_out / "s7c1_scene_group_counts.png"
    tag_fig = figure_out / "s7c1_diagnostic_tag_counts.png"

    merged.drop(columns=["chosen_error_sort", "oracle_error_sort"]).to_csv(taxonomy_out, index=False)
    group_summary.to_csv(group_out, index=False)
    scene_summary.to_csv(scene_out, index=False)
    tag_summary.to_csv(tag_out, index=False)
    shortlist.head(60).drop(columns=["chosen_error_sort", "oracle_error_sort"]).to_csv(
        shortlist_out,
        index=False,
    )

    plot_counts(
        group_summary,
        "taxonomy_group",
        "count",
        "S7C.1 candidate-generation taxonomy counts",
        group_fig,
    )

    scene_group_counts = (
        scene_summary.assign(scene_group_label=lambda d: d["primary_scene"] + " | " + d["taxonomy_group"])
        .sort_values("count", ascending=False)
        .head(30)
        .sort_values("count", ascending=True)
    )
    plot_counts(
        scene_group_counts,
        "scene_group_label",
        "count",
        "S7C.1 top scene x taxonomy groups",
        scene_fig,
    )

    plot_counts(
        tag_summary.head(30).sort_values("unique_tokens", ascending=True),
        "tag",
        "unique_tokens",
        "S7C.1 diagnostic tag counts",
        tag_fig,
    )

    counts = group_summary.set_index("taxonomy_group")["count"].to_dict()
    rates = group_summary.set_index("taxonomy_group")["rate"].to_dict()

    report = {
        "stage": "S7C.1_candidate_generation_error_taxonomy",
        "status": "COMPLETE",
        "repo_root": str(root),
        "policy": args.policy,
        "threshold_m": args.threshold_m,
        "near_threshold_m": args.near_threshold_m,
        "num_official_queries": int(len(queries)),
        "num_scene_sampled_queries": int(merged["scene_is_s7_sample"].sum()),
        "scene_column_used": scene_col,
        "taxonomy_counts": {str(k): int(v) for k, v in counts.items()},
        "taxonomy_rates": {str(k): float(v) for k, v in rates.items()},
        "headline_metrics": {
            "candidate_pool_oracle_hits_le_threshold": int(
                bool_series(merged["oracle_processed_hit_le_threshold"]).sum()
            ),
            "candidate_pool_oracle_rate_le_threshold": float(
                bool_series(merged["oracle_processed_hit_le_threshold"]).mean()
            ),
            "lightglue_selected_hits_le_threshold": int(
                bool_series(merged["hit_le_threshold"]).sum()
            ),
            "lightglue_selected_hit_rate_le_threshold": float(
                bool_series(merged["hit_le_threshold"]).mean()
            ),
            "median_chosen_error_m_eval_only": float(numeric(merged, "chosen_error_m").median()),
            "median_oracle_error_m_eval_only": float(numeric(merged, "oracle_processed_error_m").median()),
            "median_oracle_union_rank_eval_only": float(numeric(merged, "oracle_union_rank").median()),
            "median_oracle_lightglue_rank_eval_only": float(numeric(merged, "oracle_lightglue_rank").median()),
        },
        "outputs": {
            "taxonomy_by_query_csv": str(taxonomy_out),
            "taxonomy_group_summary_csv": str(group_out),
            "scene_taxonomy_summary_csv": str(scene_out),
            "diagnostic_tag_summary_csv": str(tag_out),
            "failure_token_shortlist_csv": str(shortlist_out),
            "summary_json": str(report_json),
            "taxonomy_group_counts_png": str(group_fig),
            "scene_group_counts_png": str(scene_fig),
            "diagnostic_tag_counts_png": str(tag_fig),
        },
        "locked_rule": (
            "Evaluation/error/reference columns are used only after frozen retrieval "
            "outputs exist, for taxonomy and reporting. They are not used for S7C.2 "
            "candidate generation, ranking, or online acceptance."
        ),
    }

    with open(report_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=json_safe)

    print()
    print("S7C.1 — Candidate-generation error taxonomy")
    print("-------------------------------------------")
    print("Status:                         COMPLETE")
    print(f"Repository root:                {root}")
    print(f"Policy audited:                 {args.policy}")
    print(f"Threshold m:                    {args.threshold_m:.1f}")
    print(f"Near threshold m:               {args.near_threshold_m:.1f}")
    print(f"Official queries:               {len(queries)}")
    print(f"Scene sampled queries matched:  {int(merged['scene_is_s7_sample'].sum())}")
    print(f"Scene column used:              {scene_col or 'NONE'}")
    print()
    print("Taxonomy group summary:")
    print(group_summary.to_string(index=False))
    print()
    print("Headline:")
    print(
        f"  Candidate-pool oracle hits <= {args.threshold_m:.0f}m: "
        f"{report['headline_metrics']['candidate_pool_oracle_hits_le_threshold']}/{len(merged)}"
    )
    print(
        f"  LightGlue selected hits <= {args.threshold_m:.0f}m: "
        f"{report['headline_metrics']['lightglue_selected_hits_le_threshold']}/{len(merged)}"
    )
    print()
    print(f"Taxonomy CSV:                   {taxonomy_out}")
    print(f"Group summary CSV:              {group_out}")
    print(f"Scene summary CSV:              {scene_out}")
    print(f"Diagnostic tag summary CSV:     {tag_out}")
    print(f"Failure shortlist CSV:          {shortlist_out}")
    print(f"Summary JSON:                   {report_json}")
    print(f"Figure:                         {group_fig}")
    print(f"Figure:                         {scene_fig}")
    print(f"Figure:                         {tag_fig}")
    print()
    print("Locked rule: GT/reference/error labels used only for offline taxonomy.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
