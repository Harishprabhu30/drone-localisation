#!/usr/bin/env python3
"""
S7D.1 — LightGlue verification on S7D merged candidates.

Purpose
-------
Run SuperPoint + LightGlue verification on the S7D.0 budget-aware merged
candidate pools:

  - balanced_source_budget Top-100: practical verifier input
  - rrf Top-150: higher-recall ablation
  - center_first Top-200: diagnostic ceiling only

Locked rule
-----------
Reference/evaluation fields such as eval_error_m, hit flags, oracle labels,
scene labels, and failure groups are used only after LightGlue ranking for
evaluation summaries. They are never used in LightGlue scoring, candidate
ranking, acceptance, or the optional hybrid prior.

Online-safe scoring uses only:
  - image content
  - candidate rank / merge metadata
  - tile_id
  - source_count / source stream metadata
  - LightGlue matches, RANSAC inliers, inlier ratio, and image coverage

Time block and live status
--------------------------
Long LightGlue runs can be stopped cleanly with --time-block-minutes. The script
writes a live JSON status file and checkpoint candidate CSV during execution.
Use --resume to continue an interrupted run without reprocessing finished
(query_id, tile_id) pairs.

Example
-------
python scripts/satloc/s7d/s7d_1_lightglue_merged_candidate_verifier.py \
  --candidate-pool outputs/satloc/metadata/s7d_candidate_merge/verifier_inputs/s7d1_input_balanced_source_budget_top100.csv \
  --run-name s7d1_balanced_top100 \
  --max-candidates 100 \
  --device auto \
  --resize-long 1024 \
  --max-keypoints 2048 \
  --time-block-minutes 120 \
  --checkpoint-every-candidates 25 \
  --status-every-candidates 5 \
  --resume

1. Primary run on s7d1_balanced_top100:

mkdir -p outputs/satloc/reports/s7d_lightglue/logs

python scripts/satloc/s7d/s7d_1_lightglue_merged_candidate_verifier.py \
  --candidate-pool outputs/satloc/metadata/s7d_candidate_merge/verifier_inputs/s7d1_input_balanced_source_budget_top100.csv \
  --run-name s7d1_balanced_top100 \
  --max-candidates 100 \
  --threshold-m 40 \
  --device auto \
  --resize-long 1024 \ # reduced to 512 
  --max-keypoints 2048 \ # reduce to 1024 for faster on CPu otherwise it takes 2 days on CPU, but on GPU 4 hours. if changed it takes 12hrs cpu and 1 hour gpu
  --ransac-thresh 5.0 \
  --time-block-minutes 120 \
  --checkpoint-every-candidates 25 \
  --status-every-candidates 5 \
  --status-every-seconds 30 \
  --resume \
  2>&1 | tee outputs/satloc/reports/s7d_lightglue/logs/s7d1_balanced_top100.log

2. higher recall ablation run on s7d1_rrf_top150:

python scripts/satloc/s7d/s7d_1_lightglue_merged_candidate_verifier.py \
  --candidate-pool outputs/satloc/metadata/s7d_candidate_merge/verifier_inputs/s7d1_input_rrf_top150.csv \
  --run-name s7d1_rrf_top150 \
  --max-candidates 150 \
  --threshold-m 40 \
  --device auto \
  --resize-long 1024 \
  --max-keypoints 2048 \
  --ransac-thresh 5.0 \
  --time-block-minutes 120 \
  --checkpoint-every-candidates 25 \
  --status-every-candidates 5 \
  --status-every-seconds 30 \
  --resume \
  2>&1 | tee outputs/satloc/reports/s7d_lightglue/logs/s7d1_rrf_top150.log

"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


QUERY_ID_CANDIDATES = (
    "query_id",
    "token0_id",
    "query_token0_id",
    "token",
    "uav_token0_id",
    "frame_id",
)
TILE_ID_CANDIDATES = (
    "tile_id",
    "satellite_tile_id",
    "candidate_tile_id",
    "sat_tile_id",
    "tile_index",
)
CANDIDATE_RANK_CANDIDATES = (
    "candidate_rank",
    "rank",
    "union_rank",
    "merge_rank",
    "retrieval_rank",
)
UAV_PATH_CANDIDATES = (
    "uav_image_path",
    "query_image_path",
    "image_path_uav",
    "uav_path",
    "frame_path",
    "path",
)
SAT_PATH_CANDIDATES = (
    "sat_image_path",
    "satellite_image_path",
    "candidate_image_path",
    "tile_image_path",
    "image_path_sat",
    "tile_path",
    "sat_path",
    "image_path",
    "path",
)
EVAL_ERROR_CANDIDATES = (
    "eval_error_m",
    "center_error_m",
    "tile_center_error_m",
    "error_m",
    "distance_error_m",
    "candidate_error_m",
)
SCENE_CANDIDATES = (
    "scene",
    "primary_scene",
    "scene_label",
    "canonical_scene",
    "scene_name",
)
FAILURE_GROUP_CANDIDATES = (
    "failure_group",
    "group",
    "failure_type",
    "diagnostic_group",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="S7D.1 LightGlue verifier for S7D merged candidate pools."
    )
    parser.add_argument(
        "--candidate-pool",
        type=Path,
        default=Path(
            "outputs/satloc/metadata/s7d_candidate_merge/verifier_inputs/"
            "s7d1_input_balanced_source_budget_top100.csv"
        ),
    )
    parser.add_argument("--out-base", type=Path, default=Path("outputs/satloc"))
    parser.add_argument("--run-name", default="s7d1_balanced_top100")
    parser.add_argument("--threshold-m", type=float, default=40.0)

    parser.add_argument(
        "--uav-index",
        type=Path,
        default=Path("outputs/satloc/metadata/uav_frames_index_enriched.csv"),
        help="Used only when candidate CSV lacks UAV image paths.",
    )
    parser.add_argument(
        "--sat-index",
        type=Path,
        default=Path("outputs/satloc/metadata/satellite_tiles_index_enriched.csv"),
        help="Used only when candidate CSV lacks satellite image paths.",
    )

    parser.add_argument("--query-column", default=None)
    parser.add_argument("--tile-column", default=None)
    parser.add_argument("--candidate-rank-column", default=None)
    parser.add_argument("--uav-path-column", default=None)
    parser.add_argument("--sat-path-column", default=None)
    parser.add_argument("--eval-error-column", default=None)

    parser.add_argument(
        "--query-ids",
        default="",
        help="Optional comma-separated query_id/token list. Applied before start-index/num-tokens.",
    )
    parser.add_argument("--start-index", type=int, default=0, help="0-based index into sorted selected queries.")
    parser.add_argument("--num-tokens", type=int, default=0, help="0 means all remaining queries.")
    parser.add_argument("--max-tokens", type=int, default=0, help="Alias cap after start-index; 0 means no cap.")
    parser.add_argument("--max-candidates", type=int, default=0, help="0 means all rows available per query.")

    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--resize-long", type=int, default=1024)
    parser.add_argument("--max-keypoints", type=int, default=2048)
    parser.add_argument("--ransac-thresh", type=float, default=5.0)

    parser.add_argument(
        "--time-block-minutes",
        type=float,
        default=0.0,
        help="0 means no time limit. If positive, stop cleanly after this many minutes.",
    )
    parser.add_argument("--checkpoint-every-candidates", type=int, default=25)
    parser.add_argument("--checkpoint-every-tokens", type=int, default=1)
    parser.add_argument("--status-every-candidates", type=int, default=5)
    parser.add_argument("--status-every-seconds", type=float, default=30.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save-panels", action="store_true")
    parser.add_argument("--panel-query-ids", default="", help="Optional query ids for panels after ranking.")
    parser.add_argument("--max-draw-matches", type=int, default=120)

    return parser.parse_args()


def ensure_dirs(out_base: Path) -> Dict[str, Path]:
    dirs = {
        "metadata": out_base / "metadata" / "s7d_lightglue",
        "reports": out_base / "reports" / "s7d_lightglue",
        "figures": out_base / "figures" / "s7d_lightglue",
        "panels": out_base / "figures" / "s7d_lightglue" / "panels",
        "status": out_base / "reports" / "s7d_lightglue" / "status",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def load_s5a3_module():
    path = Path("scripts/satloc/s5a/s5a_3_lightglue_topk_verifier.py")
    if not path.exists():
        raise FileNotFoundError(
            "Missing LightGlue helper script: "
            "scripts/satloc/s5a/s5a_3_lightglue_topk_verifier.py"
        )
    spec = importlib.util.spec_from_file_location("s5a3_lightglue_helpers", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def safe_str(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    text = str(value).strip()
    # Normalize numeric-looking IDs such as 67.0 -> 67.
    try:
        numeric = float(text)
        if np.isfinite(numeric) and numeric.is_integer():
            return str(int(numeric))
    except Exception:
        pass
    return text


def safe_float(value: Any) -> Optional[float]:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def numeric(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series([np.nan] * len(df), index=df.index)
    return pd.to_numeric(df[col], errors="coerce")


def infer_column(
    df: pd.DataFrame,
    candidates: Sequence[str],
    *,
    explicit: str | None,
    required: bool,
    role: str,
) -> str | None:
    if explicit:
        if explicit not in df.columns:
            raise KeyError(
                f"Explicit {role} column {explicit!r} not found. "
                f"Available columns: {list(df.columns)}"
            )
        return explicit
    for col in candidates:
        if col in df.columns:
            return col
    if required:
        raise KeyError(
            f"Could not infer {role} column. Tried {list(candidates)}. "
            f"Available columns: {list(df.columns)}"
        )
    return None


def resolve_path(value: Any) -> Optional[Path]:
    text = str(value).strip() if value is not None else ""
    if not text or text.lower() == "nan":
        return None
    path = Path(text)
    if path.exists():
        return path
    path2 = Path.cwd() / path
    if path2.exists():
        return path2
    return None


def load_index_path_lookup(
    index_path: Path,
    *,
    id_candidates: Sequence[str],
    explicit_id_col: str | None = None,
    explicit_path_col: str | None = None,
    path_candidates: Sequence[str],
    label: str,
) -> tuple[dict[str, str], dict[str, Any]]:
    if not index_path.exists():
        return {}, {"available": False, "reason": f"{label} index missing: {index_path}"}

    df = pd.read_csv(index_path)
    if df.empty:
        return {}, {"available": False, "reason": f"{label} index empty: {index_path}"}

    id_col = infer_column(
        df,
        id_candidates,
        explicit=explicit_id_col,
        required=True,
        role=f"{label} index id",
    )
    path_col = infer_column(
        df,
        path_candidates,
        explicit=explicit_path_col,
        required=True,
        role=f"{label} index image path",
    )

    lookup: dict[str, str] = {}
    missing_paths = 0
    for _, row in df.iterrows():
        key = safe_str(row.get(id_col))
        path_value = safe_str(row.get(path_col))
        if not key:
            continue
        if not path_value:
            missing_paths += 1
            continue
        lookup[key] = path_value

    return lookup, {
        "available": True,
        "path": str(index_path),
        "rows": int(len(df)),
        "id_column": id_col,
        "path_column": path_col,
        "lookup_size": int(len(lookup)),
        "missing_paths": int(missing_paths),
    }


def prepare_candidate_pool(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not args.candidate_pool.exists():
        raise FileNotFoundError(f"Candidate pool not found: {args.candidate_pool}")

    raw = pd.read_csv(args.candidate_pool)
    if raw.empty:
        raise ValueError(f"Candidate pool is empty: {args.candidate_pool}")

    query_col = infer_column(
        raw,
        QUERY_ID_CANDIDATES,
        explicit=args.query_column,
        required=True,
        role="query",
    )
    tile_col = infer_column(
        raw,
        TILE_ID_CANDIDATES,
        explicit=args.tile_column,
        required=True,
        role="tile",
    )
    rank_col = infer_column(
        raw,
        CANDIDATE_RANK_CANDIDATES,
        explicit=args.candidate_rank_column,
        required=False,
        role="candidate rank",
    )
    uav_path_col = infer_column(
        raw,
        UAV_PATH_CANDIDATES,
        explicit=args.uav_path_column,
        required=False,
        role="UAV path",
    )
    sat_path_col = infer_column(
        raw,
        SAT_PATH_CANDIDATES,
        explicit=args.sat_path_column,
        required=False,
        role="satellite path",
    )
    eval_col = infer_column(
        raw,
        EVAL_ERROR_CANDIDATES,
        explicit=args.eval_error_column,
        required=False,
        role="evaluation error",
    )

    pool = raw.copy()
    pool["query_id"] = pool[query_col].map(safe_str)
    pool["tile_id"] = pool[tile_col].map(safe_str)
    if rank_col:
        pool["candidate_rank"] = pd.to_numeric(pool[rank_col], errors="coerce")
    else:
        pool["candidate_rank"] = pool.groupby("query_id", sort=False).cumcount() + 1
    rank_missing = pool["candidate_rank"].isna()
    if rank_missing.any():
        fallback = pool.groupby("query_id", sort=False).cumcount() + 1
        pool.loc[rank_missing, "candidate_rank"] = fallback.loc[rank_missing]
    pool["candidate_rank"] = pool["candidate_rank"].astype(int)

    if eval_col:
        pool["eval_error_m"] = pd.to_numeric(pool[eval_col], errors="coerce")
    elif "eval_error_m" not in pool.columns:
        pool["eval_error_m"] = np.nan

    # Reconstruct image paths if they are absent.
    uav_lookup, uav_audit = ({}, {"available": False, "reason": "candidate CSV has UAV paths"})
    sat_lookup, sat_audit = ({}, {"available": False, "reason": "candidate CSV has satellite paths"})

    if uav_path_col:
        pool["uav_image_path"] = pool[uav_path_col].map(safe_str)
    else:
        uav_lookup, uav_audit = load_index_path_lookup(
            args.uav_index,
            id_candidates=("token0_id", "query_id", "token", "uav_frame_id", "frame_id"),
            path_candidates=UAV_PATH_CANDIDATES + (
                "image_path",
                "image_path_relative",
                "file_path",
                "filepath",
                "full_path",
            ),
            label="UAV",
        )
        pool["uav_image_path"] = pool["query_id"].map(uav_lookup).fillna("")

    if sat_path_col:
        pool["sat_image_path"] = pool[sat_path_col].map(safe_str)
    else:
        sat_lookup, sat_audit = load_index_path_lookup(
            args.sat_index,
            id_candidates=TILE_ID_CANDIDATES,
            path_candidates=SAT_PATH_CANDIDATES + (
                "image_path",
                "image_path_relative",
                "file_path",
                "filepath",
                "full_path",
            ),
            label="satellite",
        )
        pool["sat_image_path"] = pool["tile_id"].map(sat_lookup).fillna("")

    invalid = (pool["query_id"] == "") | (pool["tile_id"] == "")
    if invalid.any():
        pool = pool.loc[~invalid].copy()

    # Keep stable deterministic order.
    pool = pool.sort_values(["query_id", "candidate_rank", "tile_id"], kind="mergesort")
    pool = pool.drop_duplicates(["query_id", "tile_id"], keep="first").reset_index(drop=True)

    # Enforce per-query max-candidates after sorting.
    if args.max_candidates and args.max_candidates > 0:
        pool = (
            pool.groupby("query_id", group_keys=False, sort=True)
            .head(args.max_candidates)
            .reset_index(drop=True)
        )

    audit = {
        "candidate_pool": str(args.candidate_pool),
        "raw_rows": int(len(raw)),
        "prepared_rows": int(len(pool)),
        "queries": int(pool["query_id"].nunique()),
        "rank_max": int(pool["candidate_rank"].max()) if len(pool) else 0,
        "columns": {
            "query": query_col,
            "tile": tile_col,
            "candidate_rank": rank_col,
            "uav_path": uav_path_col,
            "sat_path": sat_path_col,
            "eval_error": eval_col,
        },
        "uav_index": uav_audit,
        "satellite_index": sat_audit,
        "missing_uav_paths": int((pool["uav_image_path"].astype(str) == "").sum()),
        "missing_sat_paths": int((pool["sat_image_path"].astype(str) == "").sum()),
    }
    return pool, audit


def select_queries(pool: pd.DataFrame, args: argparse.Namespace) -> list[str]:
    if args.query_ids.strip():
        requested = [safe_str(x) for x in args.query_ids.split(",") if safe_str(x)]
        query_set = set(pool["query_id"].astype(str))
        queries = [q for q in requested if q in query_set]
    else:
        queries = sorted(pool["query_id"].astype(str).unique().tolist(), key=lambda x: int(float(x)) if str(x).replace(".", "", 1).isdigit() else str(x))

    start = max(0, int(args.start_index))
    queries = queries[start:]

    cap = 0
    if args.num_tokens and args.num_tokens > 0:
        cap = args.num_tokens
    if args.max_tokens and args.max_tokens > 0:
        cap = min(cap, args.max_tokens) if cap > 0 else args.max_tokens
    if cap > 0:
        queries = queries[:cap]
    return queries


def lg_score(matches: int, inliers: int, ratio: float, uav_cov: float, sat_cov: float, h_ok: bool) -> float:
    spread = min(float(uav_cov), float(sat_cov))
    score = (
        float(inliers)
        + 0.04 * float(matches)
        + 4.0 * float(ratio)
        + 6.0 * float(spread)
    )
    if h_ok:
        score += 2.0
    return float(score)


def process_candidate(mod, runner, row: pd.Series, args: argparse.Namespace) -> dict[str, Any]:
    query_id = safe_str(row.get("query_id"))
    tile_id = safe_str(row.get("tile_id"))
    uav_path = resolve_path(row.get("uav_image_path"))
    sat_path = resolve_path(row.get("sat_image_path"))

    source_count = safe_float(row.get("source_count"))
    if source_count is None:
        source_count = 1.0

    candidate_rank = safe_float(row.get("candidate_rank"))
    if candidate_rank is None:
        candidate_rank = 999999.0

    out: dict[str, Any] = {
        "query_id": query_id,
        "tile_id": tile_id,
        "candidate_rank": int(candidate_rank) if float(candidate_rank).is_integer() else candidate_rank,
        "uav_image_path": str(uav_path) if uav_path else safe_str(row.get("uav_image_path")),
        "sat_image_path": str(sat_path) if sat_path else safe_str(row.get("sat_image_path")),
        "eval_error_m": safe_float(row.get("eval_error_m")),
        "policy": safe_str(row.get("policy")),
        "budget": safe_float(row.get("budget")),
        "center_rank": safe_float(row.get("center_rank")),
        "resize_rank": safe_float(row.get("resize_rank")),
        "source_count": source_count,
        "sources": safe_str(row.get("sources")),
        "resize_unique": bool(row.get("resize_unique")) if "resize_unique" in row.index else False,
        "lightglue_status": "not_started",
        "lightglue_error": "",
        "lightglue_matches": 0,
        "lightglue_ransac_inliers": 0,
        "lightglue_inlier_ratio": 0.0,
        "lightglue_homography_success": False,
        "lightglue_uav_coverage": 0.0,
        "lightglue_sat_coverage": 0.0,
        "lightglue_score": -1.0,
        "hybrid_merge_score": -1.0,
        "runtime_s": 0.0,
    }

    if uav_path is None or sat_path is None:
        out["lightglue_status"] = "missing_path"
        return out

    try:
        result = runner.match(uav_path, sat_path)
        pts0 = result["pts0"]
        pts1 = result["pts1"]
        inliers, ratio, h_ok, mask = mod.homography_stats(pts0, pts1, args.ransac_thresh)

        if mask is not None and len(mask) == len(pts0):
            inlier_pts0 = pts0[mask]
            inlier_pts1 = pts1[mask]
        else:
            inlier_pts0 = pts0
            inlier_pts1 = pts1

        rgb0 = result["rgb0"]
        rgb1 = result["rgb1"]
        uav_cov = mod.grid_coverage(inlier_pts0, rgb0.shape[:2], grid=4)
        sat_cov = mod.grid_coverage(inlier_pts1, rgb1.shape[:2], grid=4)

        score = lg_score(
            matches=int(result["matches"]),
            inliers=int(inliers),
            ratio=float(ratio),
            uav_cov=float(uav_cov),
            sat_cov=float(sat_cov),
            h_ok=bool(h_ok),
        )

        # Diagnostic online-safe prior, not the main frozen policy.
        # Uses only merge rank and duplicate/source metadata.
        hybrid_score = score + 1.5 * float(source_count) + 6.0 / math.sqrt(max(1.0, float(candidate_rank)))

        out.update(
            {
                "lightglue_status": "ok",
                "lightglue_matches": int(result["matches"]),
                "lightglue_ransac_inliers": int(inliers),
                "lightglue_inlier_ratio": float(ratio),
                "lightglue_homography_success": bool(h_ok),
                "lightglue_uav_coverage": float(uav_cov),
                "lightglue_sat_coverage": float(sat_cov),
                "lightglue_score": float(score),
                "hybrid_merge_score": float(hybrid_score),
                "runtime_s": float(result["runtime_s"]),
            }
        )
        return out
    except Exception as exc:
        out["lightglue_status"] = "failed"
        out["lightglue_error"] = repr(exc)
        return out


def rank_candidates(candidate_df: pd.DataFrame) -> pd.DataFrame:
    out = candidate_df.copy()
    if out.empty:
        return out

    out["lg_score_num"] = numeric(out, "lightglue_score")
    out["hybrid_score_num"] = numeric(out, "hybrid_merge_score")
    out["lg_inliers_num"] = numeric(out, "lightglue_ransac_inliers")
    out["lg_matches_num"] = numeric(out, "lightglue_matches")
    out["min_cov_num"] = np.minimum(
        numeric(out, "lightglue_uav_coverage").fillna(0),
        numeric(out, "lightglue_sat_coverage").fillna(0),
    )
    out["candidate_rank_num"] = numeric(out, "candidate_rank")

    parts = []
    for query_id, group in out.groupby("query_id", dropna=False):
        g = group.copy()

        lg = g.sort_values(
            ["lg_score_num", "lg_inliers_num", "lg_matches_num", "min_cov_num", "candidate_rank_num"],
            ascending=[False, False, False, False, True],
            kind="mergesort",
        )
        g.loc[lg.index, "lightglue_rank"] = np.arange(1, len(lg) + 1)

        hy = g.sort_values(
            ["hybrid_score_num", "lg_score_num", "lg_inliers_num", "candidate_rank_num"],
            ascending=[False, False, False, True],
            kind="mergesort",
        )
        g.loc[hy.index, "hybrid_merge_rank"] = np.arange(1, len(hy) + 1)

        mr = g.sort_values("candidate_rank_num", kind="mergesort")
        g.loc[mr.index, "merge_rank_only_rank"] = np.arange(1, len(mr) + 1)

        parts.append(g)

    return pd.concat(parts, ignore_index=True)


def summarize_policy(
    ranked: pd.DataFrame,
    threshold_m: float,
    *,
    rank_col: str,
    policy_name: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if ranked.empty:
        return pd.DataFrame(rows)

    r = ranked.copy()
    r["eval_error_num"] = numeric(r, "eval_error_m")
    r["policy_rank_num"] = numeric(r, rank_col)
    r["candidate_rank_num"] = numeric(r, "candidate_rank")

    for query_id, group in r.groupby("query_id", dropna=False):
        g = group.copy()
        chosen = g.sort_values("policy_rank_num", kind="mergesort").iloc[0]
        merge_top = g.sort_values("candidate_rank_num", kind="mergesort").iloc[0]

        valid = g.dropna(subset=["eval_error_num"])
        oracle = valid.sort_values("eval_error_num", kind="mergesort").iloc[0] if len(valid) else None

        chosen_error = safe_float(chosen.get("eval_error_m"))
        merge_error = safe_float(merge_top.get("eval_error_m"))
        oracle_error = safe_float(oracle.get("eval_error_m")) if oracle is not None else None

        rows.append(
            {
                "policy": policy_name,
                "query_id": query_id,
                "processed_candidates": int(len(g)),
                "chosen_tile_id": safe_str(chosen.get("tile_id")),
                "chosen_error_m": chosen_error,
                "hit_le_threshold": bool(chosen_error is not None and chosen_error <= threshold_m),
                "chosen_candidate_rank": safe_float(chosen.get("candidate_rank")),
                "chosen_lightglue_rank": safe_float(chosen.get("lightglue_rank")),
                "chosen_hybrid_rank": safe_float(chosen.get("hybrid_merge_rank")),
                "chosen_lg_score": safe_float(chosen.get("lightglue_score")),
                "chosen_hybrid_score": safe_float(chosen.get("hybrid_merge_score")),
                "chosen_inliers": safe_float(chosen.get("lightglue_ransac_inliers")),
                "chosen_matches": safe_float(chosen.get("lightglue_matches")),
                "chosen_min_coverage": safe_float(chosen.get("min_cov_num")),
                "merge_top1_error_m": merge_error,
                "merge_top1_hit_le_threshold": bool(merge_error is not None and merge_error <= threshold_m),
                "oracle_processed_error_m": oracle_error,
                "oracle_processed_hit_le_threshold": bool(oracle_error is not None and oracle_error <= threshold_m),
                "oracle_lightglue_rank": safe_float(oracle.get("lightglue_rank")) if oracle is not None else None,
                "oracle_hybrid_rank": safe_float(oracle.get("hybrid_merge_rank")) if oracle is not None else None,
                "oracle_candidate_rank": safe_float(oracle.get("candidate_rank")) if oracle is not None else None,
                "oracle_tile_id": safe_str(oracle.get("tile_id")) if oracle is not None else "",
            }
        )

    return pd.DataFrame(rows)


def policy_summary(query_summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if query_summary.empty:
        return pd.DataFrame(rows)

    for policy, group in query_summary.groupby("policy", sort=True):
        err = pd.to_numeric(group["chosen_error_m"], errors="coerce")
        oracle_err = pd.to_numeric(group["oracle_processed_error_m"], errors="coerce")
        rows.append(
            {
                "policy": policy,
                "queries": int(len(group)),
                "hits": int(group["hit_le_threshold"].sum()),
                "hit_rate": float(group["hit_le_threshold"].mean()) if len(group) else 0.0,
                "median_error_m": float(err.median()) if err.notna().any() else None,
                "mean_error_m": float(err.mean()) if err.notna().any() else None,
                "oracle_processed_hits": int(group["oracle_processed_hit_le_threshold"].sum()),
                "oracle_processed_hit_rate": float(group["oracle_processed_hit_le_threshold"].mean()) if len(group) else 0.0,
                "median_oracle_error_m": float(oracle_err.median()) if oracle_err.notna().any() else None,
                "median_oracle_lg_rank": safe_float(pd.to_numeric(group["oracle_lightglue_rank"], errors="coerce").median()),
                "median_oracle_candidate_rank": safe_float(pd.to_numeric(group["oracle_candidate_rank"], errors="coerce").median()),
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["hit_rate", "median_error_m"],
        ascending=[False, True],
        kind="mergesort",
    )


def write_status(status_path: Path, status: Mapping[str, Any]) -> None:
    tmp = status_path.with_suffix(status_path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(status, handle, indent=2, default=lambda x: None if pd.isna(x) else x)
    tmp.replace(status_path)


def checkpoint_outputs(
    rows: list[dict[str, Any]],
    cand_out: Path,
    status_path: Path,
    status: dict[str, Any],
) -> None:
    if rows:
        pd.DataFrame(rows).to_csv(cand_out, index=False)
    write_status(status_path, status)


def load_resume_rows(cand_out: Path) -> tuple[list[dict[str, Any]], set[tuple[str, str]]]:
    if not cand_out.exists():
        return [], set()
    df = pd.read_csv(cand_out)
    processed = set(zip(df["query_id"].map(safe_str), df["tile_id"].map(safe_str)))
    return df.to_dict("records"), processed


def plot_policy_summary(ps: pd.DataFrame, out_path: Path) -> None:
    if ps.empty:
        return
    plt.figure(figsize=(8, 5))
    plt.bar(ps["policy"], ps["hit_rate"])
    plt.ylim(0, 1.05)
    plt.ylabel("Hit rate <= threshold")
    plt.title("S7D.1 LightGlue merged-candidate policies")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def add_img(ax, img, title: str) -> None:
    ax.axis("off")
    ax.set_title(title, fontsize=9)
    if img is None:
        ax.text(0.5, 0.5, "missing", ha="center", va="center")
    else:
        ax.imshow(img)


def save_query_panel(mod, runner, ranked_query: pd.DataFrame, out_path: Path, args: argparse.Namespace) -> None:
    if ranked_query.empty:
        return
    query_id = safe_str(ranked_query.iloc[0]["query_id"])
    g = ranked_query.copy()
    g["eval_error_num"] = numeric(g, "eval_error_m")
    g["merge_rank_num"] = numeric(g, "candidate_rank")
    g["lg_rank_num"] = numeric(g, "lightglue_rank")
    g["hy_rank_num"] = numeric(g, "hybrid_merge_rank")

    rows = [
        ("Merge top1", g.sort_values("merge_rank_num", kind="mergesort").iloc[0]),
        ("LightGlue top1", g.sort_values("lg_rank_num", kind="mergesort").iloc[0]),
        ("Hybrid top1", g.sort_values("hy_rank_num", kind="mergesort").iloc[0]),
    ]
    valid = g.dropna(subset=["eval_error_num"])
    if len(valid):
        rows.append(("Oracle best", valid.sort_values("eval_error_num", kind="mergesort").iloc[0]))

    fig, axes = plt.subplots(len(rows), 3, figsize=(17, 4.2 * len(rows)))
    if len(rows) == 1:
        axes = np.asarray([axes])

    for i, (role, row) in enumerate(rows):
        uav_path = resolve_path(row.get("uav_image_path"))
        sat_path = resolve_path(row.get("sat_image_path"))
        if uav_path is None or sat_path is None:
            continue

        result = runner.match(uav_path, sat_path)
        pts0 = result["pts0"]
        pts1 = result["pts1"]
        inliers, ratio, h_ok, mask = mod.homography_stats(pts0, pts1, args.ransac_thresh)
        match_img = mod.draw_matches_canvas(
            result["rgb0"],
            result["rgb1"],
            pts0,
            pts1,
            mask,
            args.max_draw_matches,
        )
        add_img(axes[i, 0], result["rgb0"], f"UAV query {query_id}")
        add_img(
            axes[i, 1],
            result["rgb1"],
            f"{role}\n"
            f"tile={safe_str(row.get('tile_id'))} err={safe_float(row.get('eval_error_m'))}m\n"
            f"cand-r={safe_float(row.get('candidate_rank'))} lg-r={safe_float(row.get('lightglue_rank'))}",
        )
        add_img(
            axes[i, 2],
            match_img,
            f"score={safe_float(row.get('lightglue_score')):.2f} "
            f"inliers={safe_float(row.get('lightglue_ransac_inliers'))} "
            f"matches={safe_float(row.get('lightglue_matches'))}",
        )

    fig.suptitle(
        "S7D.1 LightGlue verification on merged candidates\n"
        "Oracle row is diagnostic only after ranking.",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    dirs = ensure_dirs(args.out_base)

    suffix = f"_{args.run_name}" if args.run_name else ""
    cand_out = dirs["metadata"] / f"s7d1_lightglue_candidate_scores{suffix}.csv"
    ranked_out = dirs["metadata"] / f"s7d1_lightglue_candidate_scores_ranked{suffix}.csv"
    query_out = dirs["metadata"] / f"s7d1_lightglue_query_summary{suffix}.csv"
    policy_out = dirs["metadata"] / f"s7d1_lightglue_policy_summary{suffix}.csv"
    env_out = dirs["reports"] / f"s7d1_environment{suffix}.json"
    report_out = dirs["reports"] / f"s7d1_lightglue_summary{suffix}.json"
    fig_out = dirs["figures"] / f"s7d1_lightglue_policy_hit_rates{suffix}.png"
    status_path = dirs["status"] / f"s7d1_live_status{suffix}.json"

    pool, pool_audit = prepare_candidate_pool(args)
    selected_queries = select_queries(pool, args)
    selected = pool[pool["query_id"].astype(str).isin(selected_queries)].copy()
    selected = selected.sort_values(["query_id", "candidate_rank", "tile_id"], kind="mergesort")

    total_candidates = int(len(selected))
    if total_candidates == 0:
        raise RuntimeError("No candidates selected for processing.")

    mod = load_s5a3_module()
    env = mod.detect_device(args.device)
    with env_out.open("w", encoding="utf-8") as handle:
        json.dump(env, handle, indent=2)

    runner = mod.LightGlueRunner(
        device=safe_str(env.get("selected_device", "cpu")),
        max_keypoints=args.max_keypoints,
        resize_long=args.resize_long,
    )

    rows: list[dict[str, Any]]
    processed_pairs: set[tuple[str, str]]
    if args.resume:
        rows, processed_pairs = load_resume_rows(cand_out)
    else:
        rows, processed_pairs = [], set()

    started = time.time()
    last_status = 0.0
    last_checkpoint_count = len(rows)
    processed_this_run = 0
    stop_reason = ""
    time_limit_s = args.time_block_minutes * 60.0 if args.time_block_minutes > 0 else 0.0

    status = {
        "stage": "S7D.1",
        "run_name": args.run_name,
        "status": "running",
        "candidate_pool": str(args.candidate_pool),
        "selected_queries": len(selected_queries),
        "total_candidates_selected": total_candidates,
        "already_processed_on_resume": len(processed_pairs),
        "processed_total_rows_written": len(rows),
        "processed_this_run": 0,
        "ok": 0,
        "missing_path": 0,
        "failed": 0,
        "current_query_id": None,
        "current_tile_id": None,
        "elapsed_s": 0.0,
        "avg_s_per_candidate_this_run": None,
        "estimated_remaining_s_selected_set": None,
        "time_block_minutes": args.time_block_minutes,
        "stop_reason": "",
        "candidate_scores_csv": str(cand_out),
    }
    write_status(status_path, status)

    print("\nS7D.1 LightGlue verification")
    print("----------------------------")
    print(f"Run name:        {args.run_name}")
    print(f"Candidates:      {total_candidates}")
    print(f"Queries:         {len(selected_queries)}")
    print(f"Resume rows:     {len(rows)}")
    print(f"Device selected: {env.get('selected_device')}")
    print(f"Status file:     {status_path}")
    print(f"Candidate CSV:   {cand_out}\n")

    for _, row in selected.iterrows():
        query_id = safe_str(row.get("query_id"))
        tile_id = safe_str(row.get("tile_id"))
        pair_key = (query_id, tile_id)
        if pair_key in processed_pairs:
            continue

        elapsed = time.time() - started
        if time_limit_s > 0 and elapsed >= time_limit_s and processed_this_run > 0:
            stop_reason = "time_block_reached"
            break

        status["current_query_id"] = query_id
        status["current_tile_id"] = tile_id

        result = process_candidate(mod, runner, row, args)
        rows.append(result)
        processed_pairs.add(pair_key)
        processed_this_run += 1

        now = time.time()
        elapsed = now - started
        avg = elapsed / max(1, processed_this_run)
        remaining = max(0, total_candidates - len(processed_pairs))
        est_remaining = avg * remaining if processed_this_run > 0 else None

        if (
            args.status_every_candidates > 0
            and processed_this_run % args.status_every_candidates == 0
        ) or (
            args.status_every_seconds > 0
            and now - last_status >= args.status_every_seconds
        ):
            df_tmp = pd.DataFrame(rows)
            status_counts = df_tmp["lightglue_status"].value_counts(dropna=False).to_dict() if "lightglue_status" in df_tmp else {}
            status.update(
                {
                    "processed_total_rows_written": len(rows),
                    "processed_this_run": processed_this_run,
                    "ok": int(status_counts.get("ok", 0)),
                    "missing_path": int(status_counts.get("missing_path", 0)),
                    "failed": int(status_counts.get("failed", 0)),
                    "elapsed_s": float(elapsed),
                    "avg_s_per_candidate_this_run": float(avg),
                    "estimated_remaining_s_selected_set": float(est_remaining) if est_remaining is not None else None,
                }
            )
            write_status(status_path, status)
            print(
                f"[S7D.1] {len(processed_pairs)}/{total_candidates} "
                f"query={query_id} tile={tile_id} "
                f"status={result['lightglue_status']} "
                f"avg={avg:.2f}s/pair"
            )
            last_status = now

        need_checkpoint = False
        if args.checkpoint_every_candidates > 0 and len(rows) - last_checkpoint_count >= args.checkpoint_every_candidates:
            need_checkpoint = True

        if need_checkpoint:
            status["status"] = "running_checkpoint"
            checkpoint_outputs(rows, cand_out, status_path, status)
            last_checkpoint_count = len(rows)

    if not stop_reason:
        stop_reason = "completed_selected_candidates"

    status["status"] = "ranking" if stop_reason == "completed_selected_candidates" else "stopped_partial"
    status["stop_reason"] = stop_reason
    status["elapsed_s"] = float(time.time() - started)
    checkpoint_outputs(rows, cand_out, status_path, status)

    cand = pd.DataFrame(rows)
    ranked = rank_candidates(cand)

    q_merge = summarize_policy(
        ranked,
        args.threshold_m,
        rank_col="merge_rank_only_rank",
        policy_name="merge_rank_only",
    )
    q_lg = summarize_policy(
        ranked,
        args.threshold_m,
        rank_col="lightglue_rank",
        policy_name="lightglue_only",
    )
    q_hybrid = summarize_policy(
        ranked,
        args.threshold_m,
        rank_col="hybrid_merge_rank",
        policy_name="lightglue_plus_merge_prior",
    )
    query_summary = pd.concat([q_merge, q_lg, q_hybrid], ignore_index=True)
    ps = policy_summary(query_summary)

    ranked.to_csv(ranked_out, index=False)
    query_summary.to_csv(query_out, index=False)
    ps.to_csv(policy_out, index=False)
    plot_policy_summary(ps, fig_out)

    panel_paths: list[str] = []
    if args.save_panels:
        panel_ids = [safe_str(x) for x in args.panel_query_ids.split(",") if safe_str(x)]
        if not panel_ids:
            # Default: first 5 queries where LG and merge disagree, then first 5 processed queries.
            lg = query_summary[query_summary["policy"] == "lightglue_only"].copy()
            mr = query_summary[query_summary["policy"] == "merge_rank_only"].copy()
            cmp = lg[["query_id", "chosen_tile_id"]].merge(
                mr[["query_id", "chosen_tile_id"]],
                on="query_id",
                suffixes=("_lg", "_merge"),
            )
            panel_ids = cmp.loc[cmp["chosen_tile_id_lg"] != cmp["chosen_tile_id_merge"], "query_id"].head(5).tolist()
            if not panel_ids:
                panel_ids = selected_queries[:5]

        for query_id in panel_ids:
            rq = ranked[ranked["query_id"].astype(str) == str(query_id)]
            if rq.empty:
                continue
            out_path = dirs["panels"] / args.run_name / f"s7d1_query{int(float(query_id)):04d}_lightglue_panel.png"
            save_query_panel(mod, runner, rq, out_path, args)
            panel_paths.append(str(out_path))

    report = {
        "stage": "S7D.1_lightglue_merged_candidate_verifier",
        "run_name": args.run_name,
        "locked_rule": (
            "eval_error_m, hit flags, oracle labels, scene labels, and failure groups "
            "are used only after LightGlue ranking for evaluation. LightGlue and hybrid "
            "ranking use image matching evidence and online-safe merge metadata only."
        ),
        "candidate_pool_audit": pool_audit,
        "selected_queries": selected_queries,
        "num_selected_queries": len(selected_queries),
        "num_selected_candidates": total_candidates,
        "num_rows_written": int(len(cand)),
        "threshold_m": args.threshold_m,
        "resize_long": args.resize_long,
        "max_keypoints": args.max_keypoints,
        "ransac_thresh": args.ransac_thresh,
        "time_block_minutes": args.time_block_minutes,
        "stop_reason": stop_reason,
        "runtime_s": float(time.time() - started),
        "environment": env,
        "status_counts": cand["lightglue_status"].value_counts(dropna=False).to_dict() if not cand.empty else {},
        "policy_summary": ps.to_dict(orient="records"),
        "outputs": {
            "candidate_scores_csv": str(cand_out),
            "ranked_candidate_scores_csv": str(ranked_out),
            "query_summary_csv": str(query_out),
            "policy_summary_csv": str(policy_out),
            "environment_json": str(env_out),
            "summary_json": str(report_out),
            "live_status_json": str(status_path),
            "policy_figure": str(fig_out),
            "panels": panel_paths,
        },
    }

    with report_out.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    status["status"] = "complete" if stop_reason == "completed_selected_candidates" else "partial_complete"
    status["stop_reason"] = stop_reason
    status["outputs"] = report["outputs"]
    write_status(status_path, status)

    print("\nS7D.1 summary")
    print("-------------")
    print(ps.to_string(index=False) if not ps.empty else "No policy summary rows.")
    print("\nSaved:")
    for path in [cand_out, ranked_out, query_out, policy_out, env_out, report_out, status_path, fig_out]:
        print(f"  {path}")
    if panel_paths:
        for path in panel_paths:
            print(f"  {path}")

    if stop_reason != "completed_selected_candidates":
        print(f"\nStopped cleanly: {stop_reason}")
        print("Rerun with --resume to continue this run.")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        traceback.print_exc()
        raise SystemExit(2) from exc
