#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = (
    "variant",
    "query_id",
    "rank",
    "tile_id",
    "score",
)


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize(df: pd.DataFrame, label: str) -> pd.DataFrame:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise RuntimeError(f"{label}: missing required columns {missing}")

    out = df.copy()
    out["variant"] = out["variant"].astype(str)
    out["query_id"] = out["query_id"].astype(str)
    out["tile_id"] = out["tile_id"].astype(str)
    out["rank"] = pd.to_numeric(out["rank"], errors="raise").astype(int)
    out["score"] = pd.to_numeric(out["score"], errors="raise").astype(float)

    if not np.isfinite(out["score"].to_numpy()).all():
        raise RuntimeError(f"{label}: non-finite score values")

    key = ["variant", "query_id", "rank"]
    if out.duplicated(key).any():
        raise RuntimeError(f"{label}: duplicate rows by {key}")

    return out.sort_values(key).reset_index(drop=True)


def tie_bucket_ids(scores: np.ndarray, tie_atol: float) -> np.ndarray:
    """Group adjacent ranked candidates into reference-score tie buckets.

    A new bucket begins only when the adjacent score drop exceeds tie_atol.
    Therefore candidates may permute only within a numerically indistinguishable
    score group; crossing a meaningful score gap is not considered equivalent.
    """
    if len(scores) == 0:
        return np.asarray([], dtype=int)

    buckets = np.zeros(len(scores), dtype=int)
    bucket = 0

    for i in range(1, len(scores)):
        if abs(float(scores[i - 1]) - float(scores[i])) > tie_atol:
            bucket += 1
        buckets[i] = bucket

    return buckets


def compare_query(
    old_q: pd.DataFrame,
    new_q: pd.DataFrame,
    score_atol: float,
    tie_atol: float,
) -> dict:
    if len(old_q) != len(new_q):
        return {
            "equivalent": False,
            "reason": "candidate_count_mismatch",
            "old_count": int(len(old_q)),
            "new_count": int(len(new_q)),
        }

    old_tiles = old_q["tile_id"].tolist()
    new_tiles = new_q["tile_id"].tolist()

    membership_equal = set(old_tiles) == set(new_tiles)
    if not membership_equal:
        return {
            "equivalent": False,
            "reason": "topk_membership_mismatch",
            "missing_from_new": sorted(set(old_tiles) - set(new_tiles)),
            "added_in_new": sorted(set(new_tiles) - set(old_tiles)),
        }

    old_by_tile = old_q.set_index("tile_id")["score"]
    new_by_tile = new_q.set_index("tile_id")["score"]

    tile_score_deltas = np.asarray(
        [
            abs(float(old_by_tile[t]) - float(new_by_tile[t]))
            for t in old_tiles
        ],
        dtype=float,
    )

    max_score_delta_by_tile = (
        float(tile_score_deltas.max())
        if len(tile_score_deltas)
        else 0.0
    )

    score_equivalent = bool(
        np.all(tile_score_deltas <= score_atol)
    )
    if not score_equivalent:
        bad = [
            {
                "tile_id": t,
                "old_score": float(old_by_tile[t]),
                "new_score": float(new_by_tile[t]),
                "abs_delta": abs(
                    float(old_by_tile[t])
                    - float(new_by_tile[t])
                ),
            }
            for t in old_tiles
            if abs(
                float(old_by_tile[t])
                - float(new_by_tile[t])
            ) > score_atol
        ]
        return {
            "equivalent": False,
            "reason": "score_delta_exceeds_tolerance",
            "max_score_delta_by_tile": max_score_delta_by_tile,
            "examples": bad[:20],
        }

    exact_order = old_tiles == new_tiles

    old_scores_ranked = old_q["score"].to_numpy(dtype=float)
    old_bucket_ids = tie_bucket_ids(
        old_scores_ranked,
        tie_atol=tie_atol,
    )

    tile_to_bucket = {
        tile: int(bucket)
        for tile, bucket in zip(old_tiles, old_bucket_ids)
    }

    old_bucket_sequence = [
        tile_to_bucket[tile]
        for tile in old_tiles
    ]
    new_bucket_sequence = [
        tile_to_bucket[tile]
        for tile in new_tiles
    ]

    tie_order_equivalent = (
        old_bucket_sequence
        == new_bucket_sequence
    )

    moved = []
    old_rank = {
        tile: int(rank)
        for tile, rank in zip(
            old_q["tile_id"],
            old_q["rank"],
        )
    }
    new_rank = {
        tile: int(rank)
        for tile, rank in zip(
            new_q["tile_id"],
            new_q["rank"],
        )
    }

    for tile in old_tiles:
        if old_rank[tile] != new_rank[tile]:
            moved.append(
                {
                    "tile_id": tile,
                    "old_rank": old_rank[tile],
                    "new_rank": new_rank[tile],
                    "reference_tie_bucket": tile_to_bucket[tile],
                    "old_score": float(old_by_tile[tile]),
                    "new_score": float(new_by_tile[tile]),
                }
            )

    if not tie_order_equivalent:
        return {
            "equivalent": False,
            "reason": "reorder_crosses_non_tied_score_gap",
            "exact_order": exact_order,
            "max_score_delta_by_tile": max_score_delta_by_tile,
            "moved_candidates": moved[:30],
        }

    return {
        "equivalent": True,
        "reason": (
            "exact"
            if exact_order and max_score_delta_by_tile == 0.0
            else (
                "score_rounding_only"
                if exact_order
                else "tie_equivalent_reorder"
            )
        ),
        "exact_order": exact_order,
        "membership_equal": True,
        "score_equivalent": True,
        "max_score_delta_by_tile": max_score_delta_by_tile,
        "moved_candidates": moved,
    }


def compare_artifacts(
    old: pd.DataFrame,
    new: pd.DataFrame,
    score_atol: float,
    tie_atol: float,
) -> dict:
    schema_equal = list(old.columns) == list(new.columns)

    old = normalize(old, "old")
    new = normalize(new, "new")

    old_groups = {
        (variant, qid): group.reset_index(drop=True)
        for (variant, qid), group in old.groupby(
            ["variant", "query_id"],
            sort=False,
        )
    }
    new_groups = {
        (variant, qid): group.reset_index(drop=True)
        for (variant, qid), group in new.groupby(
            ["variant", "query_id"],
            sort=False,
        )
    }

    old_keys = set(old_groups)
    new_keys = set(new_groups)

    if old_keys != new_keys:
        return {
            "status": "FAIL_ARTIFACT_PARITY",
            "schema_equal": schema_equal,
            "reason": "query_set_mismatch",
            "missing_from_new": sorted(old_keys - new_keys),
            "added_in_new": sorted(new_keys - old_keys),
        }

    per_query = {}
    failed = []
    tie_equivalent = []
    score_rounding_only = []
    exact = []

    for key in sorted(old_keys):
        variant, qid = key
        result = compare_query(
            old_groups[key],
            new_groups[key],
            score_atol=score_atol,
            tie_atol=tie_atol,
        )
        label = f"{variant}:{qid}"
        per_query[label] = result

        if not result["equivalent"]:
            failed.append(label)
        elif result["reason"] == "tie_equivalent_reorder":
            tie_equivalent.append(label)
        elif result["reason"] == "score_rounding_only":
            score_rounding_only.append(label)
        else:
            exact.append(label)

    if not schema_equal:
        status = "FAIL_ARTIFACT_PARITY"
    elif failed:
        status = "FAIL_ARTIFACT_PARITY"
    elif tie_equivalent or score_rounding_only:
        status = "PASS_TIE_EQUIVALENT_ARTIFACT_PARITY"
    else:
        status = "PASS_EXACT_ARTIFACT_PARITY"

    return {
        "status": status,
        "schema_equal": schema_equal,
        "score_atol": float(score_atol),
        "tie_atol": float(tie_atol),
        "query_count": int(len(old_keys)),
        "exact_query_count": int(len(exact)),
        "score_rounding_only_query_count": int(
            len(score_rounding_only)
        ),
        "tie_equivalent_query_count": int(
            len(tie_equivalent)
        ),
        "failed_query_count": int(len(failed)),
        "tie_equivalent_queries": tie_equivalent,
        "score_rounding_only_queries": score_rounding_only,
        "failed_queries": failed,
        "per_query": per_query,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare historical and current Top-K retrieval artifacts. "
            "Exact ordering is preferred; a reorder is accepted only when "
            "Top-K membership is unchanged, tile-matched scores agree within "
            "the configured tolerance, and the moved candidates remain inside "
            "a contiguous reference-score tie bucket."
        )
    )
    parser.add_argument("--old", type=Path, required=True)
    parser.add_argument("--new", type=Path, required=True)
    parser.add_argument(
        "--score-atol",
        type=float,
        default=1e-6,
    )
    parser.add_argument(
        "--tie-atol",
        type=float,
        default=1e-6,
    )
    parser.add_argument("--report", type=Path)

    args = parser.parse_args()

    if args.score_atol < 0 or args.tie_atol < 0:
        raise ValueError("Tolerances must be non-negative.")

    old_path = args.old.expanduser().resolve()
    new_path = args.new.expanduser().resolve()

    old = pd.read_csv(old_path)
    new = pd.read_csv(new_path)

    result = compare_artifacts(
        old,
        new,
        score_atol=args.score_atol,
        tie_atol=args.tie_atol,
    )

    report = {
        "stage": "A1_RETRIEVAL_ARTIFACT_PARITY",
        "created_at_utc": now_utc(),
        "old_artifact": str(old_path),
        "new_artifact": str(new_path),
        **result,
    }

    print("=" * 88)
    print("A1 — RETRIEVAL ARTIFACT PARITY")
    print("=" * 88)
    print("status:", report["status"])
    print("schema equal:", report["schema_equal"])
    print("queries:", report.get("query_count"))
    print("exact queries:", report.get("exact_query_count"))
    print(
        "score-rounding-only queries:",
        report.get("score_rounding_only_query_count"),
    )
    print(
        "tie-equivalent queries:",
        report.get("tie_equivalent_query_count"),
    )
    print("failed queries:", report.get("failed_query_count"))
    print("score atol:", args.score_atol)
    print("tie atol:", args.tie_atol)

    for label in report.get("tie_equivalent_queries", []):
        item = report["per_query"][label]
        print()
        print("TIE-EQUIVALENT:", label)
        print(
            json.dumps(
                item,
                indent=2,
            )
        )

    for label in report.get("failed_queries", [])[:20]:
        print()
        print("FAILED:", label)
        print(
            json.dumps(
                report["per_query"][label],
                indent=2,
            )
        )

    if args.report:
        report_path = args.report.expanduser().resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, indent=2),
            encoding="utf-8",
        )
        print()
        print("report:", report_path)

    if report["status"] == "FAIL_ARTIFACT_PARITY":
        raise RuntimeError(
            "Historical/current retrieval artifacts are not equivalent."
        )


if __name__ == "__main__":
    main()
