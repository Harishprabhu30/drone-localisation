#!/usr/bin/env python3
"""
S7D.0 — Budget-aware candidate merge for verifier/fusion input.

This script merges the frozen S7B.2 DINOv2-VLAD retrieval streams:

- center_square: primary/anchor stream
- resize_square: auxiliary rescue stream

The candidate ordering is strictly GT-free. Merge policies use only:
query ID, tile ID, stream rank, stream score, stream identity, and duplicate
presence. Evaluation errors and S7B.4 failure labels are attached only after
the merged ranking has been finalized.

Policies
--------
A) center_first
   Center candidates first, then resize-unique candidates.

B) balanced_source_budget
   Reserve a center quota, then add resize-unique candidates scanned from an
   auxiliary rank window, then fill remaining slots without GT.

C) rrf
   Reciprocal-rank fusion using stream ranks only.

D) rescue_append
   Keep a fixed center anchor depth (default 100), append resize-unique
   candidates, then use any remaining deeper candidates if available.

The script is schema-tolerant: common query/tile/rank/score/error column names
are detected automatically. Use explicit --*-column overrides when needed.

Command Used: 

python scripts/satloc/s7d/s7d_0_budget_aware_candidate_merge.py \
  --budgets 100,125,150,200 \
  --hit-threshold-m 40 \
  --anchor-depth 100 \
  --balanced-center-share 0.80 \
  --balanced-resize-scan-share 0.40 \
  --rrf-k 60 \
  --tag center_resize_k32_img224

"""

from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_CENTER_CSV = (
    "outputs/satloc/metadata/s7b_dinov2_vlad/"
    "s7b2_dinov2_vlad_candidate_scores_torch_hub_dinov2_vits14_vlad_"
    "k32_img224_top100_q263_sat8625_cb500_full263_k32_cb500_img224.csv"
)
DEFAULT_RESIZE_CSV = (
    "outputs/satloc/metadata/s7b_dinov2_vlad/"
    "s7b2_dinov2_vlad_candidate_scores_torch_hub_dinov2_vits14_vlad_"
    "k32_img224_top100_q263_sat8625_cb500_full263_k32_cb500_img224_resize_square.csv"
)
DEFAULT_FAILURE_CSV = (
    "outputs/satloc/metadata/s7b_failure_rescue/"
    "s7b4_failure_token_diagnostics_center_resize_k32_img224_top100.csv"
)
DEFAULT_SHORTLIST_CSV = (
    "outputs/satloc/metadata/s7b_failure_rescue/"
    "s7b4_anchor_miss_shortlist_center_resize_k32_img224_top100.csv"
)
DEFAULT_MANIFEST_CSV = (
    "outputs/satloc/metadata/s7_retrieval_upgrade/s7_0_query_manifest.csv"
)
DEFAULT_SCENE_CSV = (
    "outputs/satloc/metadata/s7_retrieval_upgrade/"
    "s7_scene_labels_canonical_traj01.csv"
)
DEFAULT_OUTPUT_ROOT = "outputs/satloc"

QUERY_COLUMN_CANDIDATES = (
    "token0_id",
    "query_token0_id",
    "query_token",
    "query_id",
    "uav_token0_id",
    "uav_frame_id",
    "frame_id",
    "token",
)
TILE_COLUMN_CANDIDATES = (
    "tile_id",
    "satellite_tile_id",
    "candidate_tile_id",
    "sat_tile_id",
    "tile_index",
    "satellite_index",
)
RANK_COLUMN_CANDIDATES = (
    "rank",
    "retrieval_rank",
    "candidate_rank",
    "stream_rank",
    "vlad_rank",
    "union_rank",
)
SCORE_COLUMN_CANDIDATES = (
    "score",
    "retrieval_score",
    "similarity",
    "similarity_score",
    "cosine_similarity",
    "vlad_score",
    "rrf_score",
)
ERROR_COLUMN_CANDIDATES = (
    "eval_error_m",
    "center_error_m",
    "tile_center_error_m",
    "error_m",
    "distance_error_m",
    "candidate_error_m",
)
SCENE_COLUMN_CANDIDATES = (
    "primary_scene",
    "scene_label",
    "canonical_scene",
    "scene",
    "scene_name",
    "secondary_scene",
)
FAILURE_GROUP_COLUMN_CANDIDATES = (
    "failure_group",
    "group",
    "failure_type",
    "diagnostic_group",
)

POLICIES = (
    "center_first",
    "balanced_source_budget",
    "rrf",
    "rescue_append",
)


@dataclass(frozen=True)
class ColumnMap:
    query: str
    tile: str
    rank: str | None
    score: str | None
    error: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="S7D.0 GT-free center/resize candidate merge and post-ranking evaluation."
    )
    parser.add_argument("--center-csv", type=Path, default=Path(DEFAULT_CENTER_CSV))
    parser.add_argument("--resize-csv", type=Path, default=Path(DEFAULT_RESIZE_CSV))
    parser.add_argument("--failure-csv", type=Path, default=Path(DEFAULT_FAILURE_CSV))
    parser.add_argument("--anchor-miss-shortlist-csv", type=Path, default=Path(DEFAULT_SHORTLIST_CSV))
    parser.add_argument("--manifest-csv", type=Path, default=Path(DEFAULT_MANIFEST_CSV))
    parser.add_argument("--scene-csv", type=Path, default=Path(DEFAULT_SCENE_CSV))

    parser.add_argument(
        "--budgets",
        default="100,125,150,200",
        help="Comma-separated output candidate budgets.",
    )
    parser.add_argument("--hit-threshold-m", type=float, default=40.0)
    parser.add_argument("--anchor-depth", type=int, default=100)
    parser.add_argument("--balanced-center-share", type=float, default=0.80)
    parser.add_argument("--balanced-resize-scan-share", type=float, default=0.40)
    parser.add_argument("--rrf-k", type=float, default=60.0)
    parser.add_argument(
        "--policies",
        default=",".join(POLICIES),
        help=f"Comma-separated subset of: {','.join(POLICIES)}",
    )

    parser.add_argument("--query-column", default=None)
    parser.add_argument("--tile-column", default=None)
    parser.add_argument("--rank-column", default=None)
    parser.add_argument("--score-column", default=None)
    parser.add_argument("--error-column", default=None)

    parser.add_argument("--tag", default="center_resize_k32_img224")
    parser.add_argument("--output-root", type=Path, default=Path(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument(
        "--strict-query-set",
        action="store_true",
        help="Fail if center and resize query sets differ.",
    )
    return parser.parse_args()


def parse_int_list(text: str, *, name: str) -> list[int]:
    values: list[int] = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            value = int(item)
        except ValueError as exc:
            raise ValueError(f"{name} contains a non-integer value: {item!r}") from exc
        if value <= 0:
            raise ValueError(f"{name} values must be positive: {value}")
        values.append(value)
    if not values:
        raise ValueError(f"{name} must contain at least one positive integer.")
    return sorted(set(values))


def parse_policy_list(text: str) -> list[str]:
    requested = [item.strip() for item in text.split(",") if item.strip()]
    unknown = sorted(set(requested) - set(POLICIES))
    if unknown:
        raise ValueError(f"Unknown policies: {unknown}. Valid policies: {list(POLICIES)}")
    if not requested:
        raise ValueError("At least one policy is required.")
    return list(dict.fromkeys(requested))


def normalize_id(value: Any) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and np.isfinite(value):
        if float(value).is_integer():
            return str(int(value))
    text = str(value).strip()
    try:
        numeric = float(text)
    except ValueError:
        return text
    if np.isfinite(numeric) and numeric.is_integer():
        return str(int(numeric))
    return text


def infer_column(
    frame: pd.DataFrame,
    candidates: Sequence[str],
    *,
    explicit: str | None,
    required: bool,
    role: str,
) -> str | None:
    if explicit is not None:
        if explicit not in frame.columns:
            raise KeyError(
                f"Explicit {role} column {explicit!r} not found. "
                f"Available columns: {list(frame.columns)}"
            )
        return explicit
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    if required:
        raise KeyError(
            f"Could not infer {role} column. Tried {list(candidates)}. "
            f"Available columns: {list(frame.columns)}"
        )
    return None


def load_stream(
    path: Path,
    stream_name: str,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, ColumnMap, dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"{stream_name} candidate CSV not found: {path}")

    raw = pd.read_csv(path)
    if raw.empty:
        raise ValueError(f"{stream_name} candidate CSV is empty: {path}")

    columns = ColumnMap(
        query=infer_column(
            raw,
            QUERY_COLUMN_CANDIDATES,
            explicit=args.query_column,
            required=True,
            role="query",
        ),
        tile=infer_column(
            raw,
            TILE_COLUMN_CANDIDATES,
            explicit=args.tile_column,
            required=True,
            role="tile",
        ),
        rank=infer_column(
            raw,
            RANK_COLUMN_CANDIDATES,
            explicit=args.rank_column,
            required=False,
            role="rank",
        ),
        score=infer_column(
            raw,
            SCORE_COLUMN_CANDIDATES,
            explicit=args.score_column,
            required=False,
            role="score",
        ),
        error=infer_column(
            raw,
            ERROR_COLUMN_CANDIDATES,
            explicit=args.error_column,
            required=False,
            role="evaluation error",
        ),
    )

    frame = pd.DataFrame()
    frame["query_id"] = raw[columns.query].map(normalize_id)
    frame["tile_id"] = raw[columns.tile].map(normalize_id)
    frame["source_row_index"] = np.arange(len(raw), dtype=np.int64)

    if columns.rank is None:
        frame["stream_rank"] = frame.groupby("query_id", sort=False).cumcount() + 1
    else:
        frame["stream_rank"] = pd.to_numeric(raw[columns.rank], errors="coerce")
        missing_rank = frame["stream_rank"].isna()
        if missing_rank.any():
            fallback = frame.groupby("query_id", sort=False).cumcount() + 1
            frame.loc[missing_rank, "stream_rank"] = fallback.loc[missing_rank]
        frame["stream_rank"] = frame["stream_rank"].astype(int)

    if columns.score is None:
        frame["stream_score"] = np.nan
    else:
        frame["stream_score"] = pd.to_numeric(raw[columns.score], errors="coerce")

    if columns.error is None:
        frame["eval_error_m"] = np.nan
    else:
        frame["eval_error_m"] = pd.to_numeric(raw[columns.error], errors="coerce")

    invalid_id = (frame["query_id"] == "") | (frame["tile_id"] == "")
    if invalid_id.any():
        warnings.warn(
            f"{stream_name}: dropping {int(invalid_id.sum())} rows with missing query/tile IDs."
        )
        frame = frame.loc[~invalid_id].copy()

    frame = frame.sort_values(
        ["query_id", "stream_rank", "source_row_index"],
        kind="mergesort",
    )
    duplicate_rows = frame.duplicated(["query_id", "tile_id"], keep="first")
    duplicate_count = int(duplicate_rows.sum())
    if duplicate_count:
        warnings.warn(
            f"{stream_name}: dropping {duplicate_count} duplicate query/tile rows; "
            "the best stream rank is retained."
        )
        frame = frame.loc[~duplicate_rows].copy()

    frame["stream"] = stream_name
    frame = frame.reset_index(drop=True)

    per_query_depth = frame.groupby("query_id")["stream_rank"].max()
    audit = {
        "path": str(path),
        "rows": int(len(frame)),
        "queries": int(frame["query_id"].nunique()),
        "tiles": int(frame["tile_id"].nunique()),
        "duplicate_query_tile_rows_removed": duplicate_count,
        "min_depth": int(per_query_depth.min()),
        "median_depth": float(per_query_depth.median()),
        "max_depth": int(per_query_depth.max()),
        "columns": asdict(columns),
    }
    return frame, columns, audit


def build_eval_lookup(center: pd.DataFrame, resize: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    eval_rows = pd.concat(
        [
            center[["query_id", "tile_id", "eval_error_m"]].assign(eval_source="center"),
            resize[["query_id", "tile_id", "eval_error_m"]].assign(eval_source="resize"),
        ],
        ignore_index=True,
    )
    valid = eval_rows.dropna(subset=["eval_error_m"]).copy()
    if valid.empty:
        warnings.warn(
            "No evaluation-error column was found in either candidate CSV. "
            "Merged candidates will be generated, but recall/error metrics will be unavailable."
        )
        return (
            eval_rows[["query_id", "tile_id"]].drop_duplicates().assign(eval_error_m=np.nan),
            {"available": False, "max_cross_stream_difference_m": None},
        )

    discrepancy = (
        valid.groupby(["query_id", "tile_id"])["eval_error_m"]
        .agg(["min", "max", "count"])
        .reset_index()
    )
    discrepancy["difference_m"] = discrepancy["max"] - discrepancy["min"]
    max_difference = float(discrepancy["difference_m"].max())
    if max_difference > 1e-6:
        warnings.warn(
            "Evaluation error differs across streams for the same query/tile. "
            f"Maximum discrepancy: {max_difference:.6f} m. The minimum is used."
        )

    lookup = (
        valid.groupby(["query_id", "tile_id"], as_index=False)["eval_error_m"]
        .min()
        .sort_values(["query_id", "tile_id"])
    )
    return lookup, {
        "available": True,
        "max_cross_stream_difference_m": max_difference,
        "rows": int(len(lookup)),
    }


def build_online_candidate_table(center: pd.DataFrame, resize: pd.DataFrame) -> pd.DataFrame:
    """Build the GT-free per-query/tile stream metadata table."""
    center_online = center[
        ["query_id", "tile_id", "stream_rank", "stream_score"]
    ].rename(
        columns={
            "stream_rank": "center_rank",
            "stream_score": "center_score",
        }
    )
    resize_online = resize[
        ["query_id", "tile_id", "stream_rank", "stream_score"]
    ].rename(
        columns={
            "stream_rank": "resize_rank",
            "stream_score": "resize_score",
        }
    )
    union = center_online.merge(
        resize_online,
        on=["query_id", "tile_id"],
        how="outer",
        validate="one_to_one",
    )
    union["is_center"] = union["center_rank"].notna()
    union["is_resize"] = union["resize_rank"].notna()
    union["source_count"] = union[["is_center", "is_resize"]].sum(axis=1).astype(int)
    union["sources"] = np.select(
        [
            union["is_center"] & union["is_resize"],
            union["is_center"],
            union["is_resize"],
        ],
        ["center|resize", "center", "resize"],
        default="",
    )
    union["resize_unique"] = union["is_resize"] & ~union["is_center"]
    return union


def sorted_records(group: pd.DataFrame, rank_column: str) -> list[dict[str, Any]]:
    subset = group.loc[group[rank_column].notna()].sort_values(
        [rank_column, "tile_id"], kind="mergesort"
    )
    return subset.to_dict("records")


def append_unique(
    output: list[dict[str, Any]],
    candidates: Iterable[dict[str, Any]],
    seen: set[str],
    *,
    limit: int | None = None,
) -> None:
    for record in candidates:
        tile_id = str(record["tile_id"])
        if tile_id in seen:
            continue
        output.append(record)
        seen.add(tile_id)
        if limit is not None and len(output) >= limit:
            return


def order_center_first(group: pd.DataFrame, budget: int) -> list[dict[str, Any]]:
    center = sorted_records(group, "center_rank")
    resize = sorted_records(group, "resize_rank")
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    append_unique(output, center, seen, limit=budget)
    if len(output) < budget:
        append_unique(output, resize, seen, limit=budget)
    return output[:budget]


def order_rescue_append(
    group: pd.DataFrame,
    budget: int,
    *,
    anchor_depth: int,
) -> list[dict[str, Any]]:
    center = sorted_records(group, "center_rank")
    resize = sorted_records(group, "resize_rank")
    center_anchor = [
        row for row in center if int(row["center_rank"]) <= anchor_depth
    ]
    center_deeper = [
        row for row in center if int(row["center_rank"]) > anchor_depth
    ]

    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    append_unique(output, center_anchor, seen, limit=budget)
    if len(output) < budget:
        append_unique(output, resize, seen, limit=budget)
    if len(output) < budget:
        append_unique(output, center_deeper, seen, limit=budget)
    return output[:budget]


def order_balanced(
    group: pd.DataFrame,
    budget: int,
    *,
    center_share: float,
    resize_scan_share: float,
) -> list[dict[str, Any]]:
    center = sorted_records(group, "center_rank")
    resize = sorted_records(group, "resize_rank")

    center_quota = min(len(center), max(1, int(math.ceil(center_share * budget))))
    resize_scan_depth = min(
        len(resize), max(1, int(math.ceil(resize_scan_share * budget)))
    )

    center_primary = center[:center_quota]
    center_remaining = center[center_quota:]
    resize_scanned = resize[:resize_scan_depth]
    resize_remaining = resize[resize_scan_depth:]

    output: list[dict[str, Any]] = []
    seen: set[str] = set()

    append_unique(output, center_primary, seen, limit=budget)
    if len(output) < budget:
        append_unique(output, resize_scanned, seen, limit=budget)
    if len(output) < budget:
        append_unique(output, center_remaining, seen, limit=budget)
    if len(output) < budget:
        append_unique(output, resize_remaining, seen, limit=budget)
    return output[:budget]


def order_rrf(group: pd.DataFrame, budget: int, *, rrf_k: float) -> list[dict[str, Any]]:
    ranked = group.copy()
    center_component = np.where(
        ranked["center_rank"].notna(),
        1.0 / (rrf_k + ranked["center_rank"].astype(float)),
        0.0,
    )
    resize_component = np.where(
        ranked["resize_rank"].notna(),
        1.0 / (rrf_k + ranked["resize_rank"].astype(float)),
        0.0,
    )
    ranked["merge_score"] = center_component + resize_component
    ranked["best_stream_rank"] = ranked[["center_rank", "resize_rank"]].min(axis=1)
    ranked["_center_sort"] = ranked["center_rank"].fillna(np.inf)
    ranked["_resize_sort"] = ranked["resize_rank"].fillna(np.inf)
    ranked = ranked.sort_values(
        [
            "merge_score",
            "source_count",
            "best_stream_rank",
            "_center_sort",
            "_resize_sort",
            "tile_id",
        ],
        ascending=[False, False, True, True, True, True],
        kind="mergesort",
    )
    return ranked.head(budget).drop(
        columns=["_center_sort", "_resize_sort"], errors="ignore"
    ).to_dict("records")


def merge_candidates(
    online_union: pd.DataFrame,
    policies: Sequence[str],
    budgets: Sequence[int],
    *,
    anchor_depth: int,
    balanced_center_share: float,
    balanced_resize_scan_share: float,
    rrf_k: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for query_id, group in online_union.groupby("query_id", sort=True):
        for policy in policies:
            for budget in budgets:
                if policy == "center_first":
                    ordered = order_center_first(group, budget)
                elif policy == "balanced_source_budget":
                    ordered = order_balanced(
                        group,
                        budget,
                        center_share=balanced_center_share,
                        resize_scan_share=balanced_resize_scan_share,
                    )
                elif policy == "rrf":
                    ordered = order_rrf(group, budget, rrf_k=rrf_k)
                elif policy == "rescue_append":
                    ordered = order_rescue_append(
                        group,
                        budget,
                        anchor_depth=anchor_depth,
                    )
                else:
                    raise AssertionError(f"Unhandled policy: {policy}")

                for candidate_rank, record in enumerate(ordered, start=1):
                    row = {
                        "query_id": query_id,
                        "policy": policy,
                        "budget": int(budget),
                        "candidate_rank": int(candidate_rank),
                        "tile_id": str(record["tile_id"]),
                        "center_rank": record.get("center_rank", np.nan),
                        "resize_rank": record.get("resize_rank", np.nan),
                        "center_score": record.get("center_score", np.nan),
                        "resize_score": record.get("resize_score", np.nan),
                        "source_count": int(record.get("source_count", 0)),
                        "sources": record.get("sources", ""),
                        "is_center": bool(record.get("is_center", False)),
                        "is_resize": bool(record.get("is_resize", False)),
                        "resize_unique": bool(record.get("resize_unique", False)),
                        "merge_score": record.get("merge_score", np.nan),
                    }
                    rows.append(row)

    if not rows:
        raise RuntimeError("No merged candidate rows were generated.")
    return pd.DataFrame(rows)


def load_optional_query_metadata(
    path: Path,
    *,
    value_candidates: Sequence[str],
    value_name: str,
) -> pd.DataFrame | None:
    if not path.exists():
        warnings.warn(f"Optional {value_name} CSV not found; skipping: {path}")
        return None
    raw = pd.read_csv(path)
    if raw.empty:
        warnings.warn(f"Optional {value_name} CSV is empty; skipping: {path}")
        return None
    query_col = infer_column(
        raw,
        QUERY_COLUMN_CANDIDATES,
        explicit=None,
        required=True,
        role=f"{value_name} query",
    )
    value_col = infer_column(
        raw,
        value_candidates,
        explicit=None,
        required=True,
        role=value_name,
    )
    result = pd.DataFrame(
        {
            "query_id": raw[query_col].map(normalize_id),
            value_name: raw[value_col].astype(str),
        }
    )
    return result.drop_duplicates("query_id", keep="first")


def load_manifest_metadata(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        warnings.warn(f"Optional query manifest not found; skipping: {path}")
        return None
    raw = pd.read_csv(path)
    if raw.empty:
        warnings.warn(f"Optional query manifest is empty; skipping: {path}")
        return None
    query_col = infer_column(
        raw,
        QUERY_COLUMN_CANDIDATES,
        explicit=None,
        required=True,
        role="manifest query",
    )
    result = raw.copy()
    result["query_id"] = result[query_col].map(normalize_id)
    keep_columns = ["query_id"]
    preferred = (
        "sequence",
        "stream",
        "query_order",
        "sequence_order",
        "token0_id",
        "token1_order",
        "uav_image_path",
        "image_path",
        "query_source",
    )
    keep_columns.extend([col for col in preferred if col in result.columns and col != query_col])
    keep_columns = list(dict.fromkeys(keep_columns))
    return result[keep_columns].drop_duplicates("query_id", keep="first")


def build_anchor_baselines(
    center: pd.DataFrame,
    resize: pd.DataFrame,
    *,
    hit_threshold_m: float,
    anchor_depth: int,
) -> pd.DataFrame:
    def summarize(stream: pd.DataFrame, prefix: str) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for query_id, group in stream.groupby("query_id", sort=True):
            ranked = group.sort_values("stream_rank")
            within = ranked.loc[ranked["stream_rank"] <= anchor_depth]
            valid = within.dropna(subset=["eval_error_m"])
            hit = bool((valid["eval_error_m"] <= hit_threshold_m).any()) if not valid.empty else False
            best_error = float(valid["eval_error_m"].min()) if not valid.empty else np.nan
            rows.append(
                {
                    "query_id": query_id,
                    f"{prefix}_hit_at_anchor": hit,
                    f"{prefix}_oracle_error_at_anchor_m": best_error,
                }
            )
        return pd.DataFrame(rows)

    center_summary = summarize(center, "center")
    resize_summary = summarize(resize, "resize")
    return center_summary.merge(resize_summary, on="query_id", how="outer")


def evaluate_queries(
    merged: pd.DataFrame,
    *,
    hit_threshold_m: float,
    recall_depths: Sequence[int],
    anchor_baselines: pd.DataFrame,
    scene_metadata: pd.DataFrame | None,
    failure_metadata: pd.DataFrame | None,
    manifest_metadata: pd.DataFrame | None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (query_id, policy, budget), group in merged.groupby(
        ["query_id", "policy", "budget"], sort=True
    ):
        ranked = group.sort_values("candidate_rank")
        valid = ranked.dropna(subset=["eval_error_m"])
        top1_error = (
            float(ranked.iloc[0]["eval_error_m"])
            if pd.notna(ranked.iloc[0]["eval_error_m"])
            else np.nan
        )
        if valid.empty:
            oracle_error = np.nan
            oracle_rank = np.nan
        else:
            best_index = valid["eval_error_m"].idxmin()
            oracle_error = float(valid.loc[best_index, "eval_error_m"])
            oracle_rank = int(valid.loc[best_index, "candidate_rank"])

        row: dict[str, Any] = {
            "query_id": query_id,
            "policy": policy,
            "budget": int(budget),
            "candidate_count": int(len(ranked)),
            "top1_error_m": top1_error,
            "oracle_error_m": oracle_error,
            "oracle_rank": oracle_rank,
            "hit_at_budget": bool(
                (valid["eval_error_m"] <= hit_threshold_m).any()
            )
            if not valid.empty
            else False,
            "resize_unique_candidates_used": int(ranked["resize_unique"].sum()),
            "duplicate_source_candidates_used": int((ranked["source_count"] == 2).sum()),
        }
        for depth in recall_depths:
            subset = valid.loc[valid["candidate_rank"] <= depth]
            row[f"hit_at_{depth}"] = (
                bool((subset["eval_error_m"] <= hit_threshold_m).any())
                if not subset.empty
                else False
            )
        rows.append(row)

    summary = pd.DataFrame(rows)
    summary = summary.merge(anchor_baselines, on="query_id", how="left")
    summary["non_anchor_rescue"] = (
        ~summary["center_hit_at_anchor"].fillna(False)
        & summary["hit_at_budget"]
    )
    summary["resize_only_baseline_rescue"] = (
        ~summary["center_hit_at_anchor"].fillna(False)
        & summary["resize_hit_at_anchor"].fillna(False)
        & summary["hit_at_budget"]
    )

    for metadata in (scene_metadata, failure_metadata, manifest_metadata):
        if metadata is not None:
            overlapping = [
                col for col in metadata.columns if col in summary.columns and col != "query_id"
            ]
            metadata_to_join = metadata.drop(columns=overlapping, errors="ignore")
            summary = summary.merge(metadata_to_join, on="query_id", how="left")

    return summary


def aggregate_policy_summary(
    query_summary: pd.DataFrame,
    recall_depths: Sequence[int],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (policy, budget), group in query_summary.groupby(["policy", "budget"], sort=True):
        row: dict[str, Any] = {
            "policy": policy,
            "budget": int(budget),
            "queries": int(group["query_id"].nunique()),
            "candidate_count_min": int(group["candidate_count"].min()),
            "candidate_count_median": float(group["candidate_count"].median()),
            "candidate_count_max": int(group["candidate_count"].max()),
            "recall_at_budget": float(group["hit_at_budget"].mean()),
            "hits_at_budget": int(group["hit_at_budget"].sum()),
            "median_top1_error_m": float(group["top1_error_m"].median()),
            "median_oracle_error_m": float(group["oracle_error_m"].median()),
            "median_oracle_rank": float(group["oracle_rank"].median()),
            "non_anchor_rescues": int(group["non_anchor_rescue"].sum()),
            "resize_only_baseline_rescues": int(
                group["resize_only_baseline_rescue"].sum()
            ),
            "resize_unique_candidates_used_mean": float(
                group["resize_unique_candidates_used"].mean()
            ),
            "resize_unique_candidates_used_total": int(
                group["resize_unique_candidates_used"].sum()
            ),
        }
        for depth in recall_depths:
            column = f"hit_at_{depth}"
            row[f"recall_at_{depth}"] = float(group[column].mean())
            row[f"hits_at_{depth}"] = int(group[column].sum())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["budget", "policy"]).reset_index(drop=True)


def aggregate_group_summary(
    query_summary: pd.DataFrame,
    group_column: str,
) -> pd.DataFrame:
    if group_column not in query_summary.columns:
        return pd.DataFrame()
    usable = query_summary.loc[query_summary[group_column].notna()].copy()
    if usable.empty:
        return pd.DataFrame()
    grouped = (
        usable.groupby(["policy", "budget", group_column], dropna=False)
        .agg(
            queries=("query_id", "nunique"),
            hits_at_budget=("hit_at_budget", "sum"),
            recall_at_budget=("hit_at_budget", "mean"),
            median_top1_error_m=("top1_error_m", "median"),
            median_oracle_error_m=("oracle_error_m", "median"),
            non_anchor_rescues=("non_anchor_rescue", "sum"),
        )
        .reset_index()
    )
    return grouped


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if np.isnan(value):
            return None
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if pd.isna(value):
        return None
    raise TypeError(f"Cannot JSON-serialize {type(value)!r}")


def records_for_json(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    clean = frame.replace({np.nan: None})
    return clean.to_dict("records")


def save_figures(
    policy_summary: pd.DataFrame,
    query_summary: pd.DataFrame,
    scene_summary: pd.DataFrame,
    failure_summary: pd.DataFrame,
    recall_depths: Sequence[int],
    figure_dir: Path,
    tag: str,
) -> list[str]:
    figure_dir.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []
    max_budget = int(policy_summary["budget"].max())

    # Recall by policy at the largest evaluated budget.
    subset = policy_summary.loc[policy_summary["budget"] == max_budget]
    plt.figure(figsize=(9, 5))
    for _, row in subset.iterrows():
        depths = [depth for depth in recall_depths if depth <= max_budget]
        values = [row[f"recall_at_{depth}"] for depth in depths]
        plt.plot(depths, values, marker="o", label=row["policy"])
    plt.xlabel("Candidate depth")
    plt.ylabel("Recall")
    plt.title(f"S7D.0 recall by policy (budget={max_budget})")
    plt.ylim(0.0, 1.0)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    path = figure_dir / f"s7d0_recall_by_policy_{tag}.png"
    plt.savefig(path, dpi=180)
    plt.close()
    saved.append(str(path))

    # Recall at each full budget.
    plt.figure(figsize=(9, 5))
    for policy, group in policy_summary.groupby("policy", sort=True):
        ordered = group.sort_values("budget")
        plt.plot(
            ordered["budget"],
            ordered["recall_at_budget"],
            marker="o",
            label=policy,
        )
    plt.xlabel("Candidate budget")
    plt.ylabel("Recall within budget")
    plt.title("S7D.0 recall by candidate budget")
    plt.ylim(0.0, 1.0)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    path = figure_dir / f"s7d0_recall_by_budget_{tag}.png"
    plt.savefig(path, dpi=180)
    plt.close()
    saved.append(str(path))

    # Scene recall at largest budget.
    if not scene_summary.empty:
        scene_max = scene_summary.loc[scene_summary["budget"] == max_budget]
        plt.figure(figsize=(11, 5))
        for policy, group in scene_max.groupby("policy", sort=True):
            ordered = group.sort_values("scene")
            plt.plot(
                ordered["scene"],
                ordered["recall_at_budget"],
                marker="o",
                label=policy,
            )
        plt.xlabel("Scene")
        plt.ylabel("Recall within budget")
        plt.title(f"S7D.0 scene recall (budget={max_budget})")
        plt.ylim(0.0, 1.0)
        plt.xticks(rotation=30, ha="right")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        path = figure_dir / f"s7d0_scene_recall_{tag}.png"
        plt.savefig(path, dpi=180)
        plt.close()
        saved.append(str(path))

    # Failure-group recall at largest budget.
    if not failure_summary.empty:
        failure_max = failure_summary.loc[failure_summary["budget"] == max_budget]
        plt.figure(figsize=(12, 5))
        for policy, group in failure_max.groupby("policy", sort=True):
            ordered = group.sort_values("failure_group")
            plt.plot(
                ordered["failure_group"],
                ordered["recall_at_budget"],
                marker="o",
                label=policy,
            )
        plt.xlabel("S7B.4 failure group (diagnostic only)")
        plt.ylabel("Recall within budget")
        plt.title(f"S7D.0 failure-group recall (budget={max_budget})")
        plt.ylim(0.0, 1.0)
        plt.xticks(rotation=30, ha="right")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        path = figure_dir / f"s7d0_failure_group_recall_{tag}.png"
        plt.savefig(path, dpi=180)
        plt.close()
        saved.append(str(path))

    # Candidate count distribution at largest budget.
    count_subset = query_summary.loc[query_summary["budget"] == max_budget]
    plt.figure(figsize=(9, 5))
    for policy, group in count_subset.groupby("policy", sort=True):
        plt.hist(
            group["candidate_count"],
            bins=min(20, max(5, group["candidate_count"].nunique())),
            histtype="step",
            linewidth=1.8,
            label=policy,
        )
    plt.xlabel("Actual deduplicated candidate count")
    plt.ylabel("Queries")
    plt.title(f"S7D.0 candidate-count distribution (budget={max_budget})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    path = figure_dir / f"s7d0_candidate_count_distribution_{tag}.png"
    plt.savefig(path, dpi=180)
    plt.close()
    saved.append(str(path))

    return saved


def print_summary(
    center_audit: Mapping[str, Any],
    resize_audit: Mapping[str, Any],
    policy_summary: pd.DataFrame,
    *,
    hit_threshold_m: float,
) -> None:
    print("\nS7D.0 Budget-aware Candidate Merge")
    print("----------------------------------")
    print(f"Center queries:       {center_audit['queries']}")
    print(f"Center max depth:     {center_audit['max_depth']}")
    print(f"Resize queries:       {resize_audit['queries']}")
    print(f"Resize max depth:     {resize_audit['max_depth']}")
    print(f"Evaluation threshold: {hit_threshold_m:.1f} m")
    print()
    columns = [
        "policy",
        "budget",
        "hits_at_budget",
        "queries",
        "recall_at_budget",
        "non_anchor_rescues",
        "resize_only_baseline_rescues",
        "candidate_count_median",
        "median_oracle_error_m",
        "median_oracle_rank",
    ]
    display = policy_summary[columns].copy()
    display["recall_at_budget"] = display["recall_at_budget"].map(
        lambda value: f"{value:.4f}"
    )
    for column in ("median_oracle_error_m", "median_oracle_rank"):
        display[column] = display[column].map(
            lambda value: "nan" if pd.isna(value) else f"{value:.3f}"
        )
    print(display.to_string(index=False))


def main() -> int:
    args = parse_args()
    budgets = parse_int_list(args.budgets, name="budgets")
    policies = parse_policy_list(args.policies)

    if not 0.0 < args.balanced_center_share <= 1.0:
        raise ValueError("--balanced-center-share must be in (0, 1].")
    if args.balanced_resize_scan_share <= 0.0:
        raise ValueError("--balanced-resize-scan-share must be positive.")
    if args.anchor_depth <= 0:
        raise ValueError("--anchor-depth must be positive.")
    if args.rrf_k <= 0:
        raise ValueError("--rrf-k must be positive.")

    metadata_dir = args.output_root / "metadata" / "s7d_candidate_merge"
    report_dir = args.output_root / "reports" / "s7d_candidate_merge"
    figure_dir = args.output_root / "figures" / "s7d_candidate_merge"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_figures:
        figure_dir.mkdir(parents=True, exist_ok=True)

    center, center_columns, center_audit = load_stream(
        args.center_csv, "center", args
    )
    resize, resize_columns, resize_audit = load_stream(
        args.resize_csv, "resize", args
    )

    center_queries = set(center["query_id"])
    resize_queries = set(resize["query_id"])
    if center_queries != resize_queries:
        message = (
            "Center/resize query sets differ: "
            f"center-only={len(center_queries - resize_queries)}, "
            f"resize-only={len(resize_queries - center_queries)}."
        )
        if args.strict_query_set:
            raise ValueError(message)
        warnings.warn(message)

    eval_lookup, eval_audit = build_eval_lookup(center, resize)

    # Critical separation: policy ordering receives no evaluation/error columns.
    online_union = build_online_candidate_table(center, resize)
    merged_online = merge_candidates(
        online_union,
        policies,
        budgets,
        anchor_depth=args.anchor_depth,
        balanced_center_share=args.balanced_center_share,
        balanced_resize_scan_share=args.balanced_resize_scan_share,
        rrf_k=args.rrf_k,
    )
    merged = merged_online.merge(
        eval_lookup,
        on=["query_id", "tile_id"],
        how="left",
        validate="many_to_one",
    )

    recall_depths = sorted(
        set([1, 5, 10, 20, 50, 100, 125, 150, 200] + budgets)
    )
    anchor_baselines = build_anchor_baselines(
        center,
        resize,
        hit_threshold_m=args.hit_threshold_m,
        anchor_depth=args.anchor_depth,
    )

    scene_metadata = load_optional_query_metadata(
        args.scene_csv,
        value_candidates=SCENE_COLUMN_CANDIDATES,
        value_name="scene",
    )
    failure_metadata = load_optional_query_metadata(
        args.failure_csv,
        value_candidates=FAILURE_GROUP_COLUMN_CANDIDATES,
        value_name="failure_group",
    )
    manifest_metadata = load_manifest_metadata(args.manifest_csv)

    query_summary = evaluate_queries(
        merged,
        hit_threshold_m=args.hit_threshold_m,
        recall_depths=recall_depths,
        anchor_baselines=anchor_baselines,
        scene_metadata=scene_metadata,
        failure_metadata=failure_metadata,
        manifest_metadata=manifest_metadata,
    )
    policy_summary = aggregate_policy_summary(query_summary, recall_depths)
    scene_summary = aggregate_group_summary(query_summary, "scene")
    failure_group_summary = aggregate_group_summary(query_summary, "failure_group")

    candidate_path = metadata_dir / f"s7d0_candidate_scores_{args.tag}.csv"
    query_path = metadata_dir / f"s7d0_query_summary_{args.tag}.csv"
    policy_path = metadata_dir / f"s7d0_policy_summary_{args.tag}.csv"
    scene_path = metadata_dir / f"s7d0_scene_summary_{args.tag}.csv"
    failure_path = metadata_dir / f"s7d0_failure_group_summary_{args.tag}.csv"

    merged.to_csv(candidate_path, index=False)
    query_summary.to_csv(query_path, index=False)
    policy_summary.to_csv(policy_path, index=False)
    scene_summary.to_csv(scene_path, index=False)
    failure_group_summary.to_csv(failure_path, index=False)

    figures: list[str] = []
    if not args.no_figures:
        figures = save_figures(
            policy_summary,
            query_summary,
            scene_summary,
            failure_group_summary,
            recall_depths,
            figure_dir,
            args.tag,
        )

    max_center_depth = int(center_audit["max_depth"])
    max_resize_depth = int(resize_audit["max_depth"])
    policy_equivalence_note = None
    if max_center_depth <= args.anchor_depth:
        policy_equivalence_note = (
            "center_first and rescue_append are expected to be equivalent with "
            f"the current center export depth ({max_center_depth}) because "
            f"anchor_depth={args.anchor_depth}. They diverge after a deeper center export."
        )

    report = {
        "stage": "S7D.0",
        "title": "Budget-aware candidate merge for verifier/fusion input",
        "locked_online_rule": (
            "Candidate ordering uses stream identity, rank, score metadata, tile_id, "
            "and duplicate presence only. eval_error_m, scene labels, and failure groups "
            "are attached after ranking for diagnostics."
        ),
        "configuration": {
            "budgets": budgets,
            "policies": policies,
            "hit_threshold_m": args.hit_threshold_m,
            "anchor_depth": args.anchor_depth,
            "balanced_center_share": args.balanced_center_share,
            "balanced_resize_scan_share": args.balanced_resize_scan_share,
            "rrf_k": args.rrf_k,
            "tag": args.tag,
        },
        "inputs": {
            "center": center_audit,
            "resize": resize_audit,
            "evaluation_lookup": eval_audit,
            "failure_csv": str(args.failure_csv),
            "anchor_miss_shortlist_csv": str(args.anchor_miss_shortlist_csv),
            "manifest_csv": str(args.manifest_csv),
            "scene_csv": str(args.scene_csv),
        },
        "column_maps": {
            "center": asdict(center_columns),
            "resize": asdict(resize_columns),
        },
        "query_set_audit": {
            "center_queries": len(center_queries),
            "resize_queries": len(resize_queries),
            "center_only_queries": sorted(center_queries - resize_queries),
            "resize_only_queries": sorted(resize_queries - center_queries),
        },
        "available_unique_candidates": {
            "min_per_query": int(online_union.groupby("query_id").size().min()),
            "median_per_query": float(online_union.groupby("query_id").size().median()),
            "max_per_query": int(online_union.groupby("query_id").size().max()),
        },
        "policy_equivalence_note": policy_equivalence_note,
        "outputs": {
            "candidate_scores_csv": str(candidate_path),
            "query_summary_csv": str(query_path),
            "policy_summary_csv": str(policy_path),
            "scene_summary_csv": str(scene_path),
            "failure_group_summary_csv": str(failure_path),
            "figures": figures,
        },
        "policy_summary": records_for_json(policy_summary),
        "scene_summary": records_for_json(scene_summary),
        "failure_group_summary": records_for_json(failure_group_summary),
    }

    report_path = report_dir / f"s7d0_summary_{args.tag}.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, default=json_safe)

    print_summary(
        center_audit,
        resize_audit,
        policy_summary,
        hit_threshold_m=args.hit_threshold_m,
    )
    print("\nSaved:")
    for path in (
        candidate_path,
        query_path,
        policy_path,
        scene_path,
        failure_path,
        report_path,
    ):
        print(f"  {path}")
    for path in figures:
        print(f"  {path}")

    if policy_equivalence_note:
        print(f"\nNote: {policy_equivalence_note}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, KeyError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
