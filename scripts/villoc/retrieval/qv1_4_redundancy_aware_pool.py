#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

try:
    from scripts.villoc.retrieval.qv1_3_fixed_budget_multiview_pool import (
        load_reference_xy,
        tile_centers,
        spatially_distinct,
        evaluate_pool,
    )
except ModuleNotFoundError:
    from qv1_3_fixed_budget_multiview_pool import (
        load_reference_xy,
        tile_centers,
        spatially_distinct,
        evaluate_pool,
    )


ROOT = Path.cwd().resolve()
DEFAULT_CONFIG = Path(
    "configs/research/query_view_candidate_generation_v1.yaml"
)
DEFAULT_QV11_ROOT = Path(
    "outputs/research_runs/query_view_candidate_generation_v1/qv1_1"
)
DEFAULT_OUTPUT_ROOT = Path(
    "outputs/research_runs/query_view_candidate_generation_v1/qv1_4"
)


def resolve(path_like: str | Path) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else ROOT / path


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def row_item(
    *,
    qid: int,
    tile_id: str,
    source_view: str,
    source_rank: int,
    reason: str,
) -> dict[str, Any]:
    return {
        "query_id": int(qid),
        "tile_id": str(tile_id),
        "source_view": str(source_view),
        "source_rank": int(source_rank),
        "selection_reason": str(reason),
    }


def compress_center(
    *,
    qid: int,
    center: pd.DataFrame,
    centers: dict[str, tuple[float, float]],
    source_depth: int,
    radius_m: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    retained: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []

    for row in center.head(source_depth).itertuples(index=False):
        item = row_item(
            qid=qid,
            tile_id=str(row.tile_id),
            source_view="center_square",
            source_rank=int(row.rank),
            reason="center_spatially_unique",
        )

        if not retained:
            retained.append(item)
            continue

        if spatially_distinct(
            str(row.tile_id),
            retained,
            centers,
            radius_m,
        ):
            retained.append(item)
        else:
            item["selection_reason"] = "center_spatially_redundant_dropped"
            dropped.append(item)

    return retained, dropped


def round_robin_alternates(
    *,
    qid: int,
    by_view: dict[str, pd.DataFrame],
    alt_views: list[str],
    selected: list[dict[str, Any]],
    seen: set[str],
    centers: dict[str, tuple[float, float]],
    source_depth: int,
    radius_m: float,
    budget: int,
) -> None:
    lists = {
        view: list(
            by_view[view].head(source_depth).itertuples(index=False)
        )
        for view in alt_views
    }
    max_depth = max(len(rows) for rows in lists.values())

    for idx in range(max_depth):
        for view in alt_views:
            if len(selected) >= budget:
                return
            rows = lists[view]
            if idx >= len(rows):
                continue

            row = rows[idx]
            tile_id = str(row.tile_id)

            if tile_id in seen:
                continue

            if not spatially_distinct(
                tile_id,
                selected,
                centers,
                radius_m,
            ):
                continue

            selected.append(
                row_item(
                    qid=qid,
                    tile_id=tile_id,
                    source_view=view,
                    source_rank=int(row.rank),
                    reason="alternate_fills_redundant_center_slot",
                )
            )
            seen.add(tile_id)


def deterministic_refill(
    *,
    qid: int,
    selected: list[dict[str, Any]],
    seen: set[str],
    dropped_center: list[dict[str, Any]],
    by_view: dict[str, pd.DataFrame],
    alt_views: list[str],
    source_depth: int,
    budget: int,
) -> None:
    # Prefer restoring historical center evidence if the diversity pass
    # cannot fill the fixed budget.
    for item in dropped_center:
        if len(selected) >= budget:
            return
        tile_id = str(item["tile_id"])
        if tile_id in seen:
            continue
        restored = dict(item)
        restored["selection_reason"] = "refill_dropped_center"
        selected.append(restored)
        seen.add(tile_id)

    for view in alt_views:
        for row in by_view[view].head(source_depth).itertuples(index=False):
            if len(selected) >= budget:
                return
            tile_id = str(row.tile_id)
            if tile_id in seen:
                continue
            selected.append(
                row_item(
                    qid=qid,
                    tile_id=tile_id,
                    source_view=view,
                    source_rank=int(row.rank),
                    reason="refill_alternate_redundant",
                )
            )
            seen.add(tile_id)


def build_pool(
    *,
    qid: int,
    rankings: dict[str, pd.DataFrame],
    centers: dict[str, tuple[float, float]],
    policy: str,
    budget: int,
    center_source_depth: int,
    alternate_source_depth: int,
    radius_m: float,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    by_view = {
        view: frame[frame["query_id"] == qid].sort_values("rank")
        for view, frame in rankings.items()
    }

    center = by_view["center_square"]

    if policy == "center_top20":
        selected = [
            row_item(
                qid=qid,
                tile_id=str(row.tile_id),
                source_view="center_square",
                source_rank=int(row.rank),
                reason="center_control",
            )
            for row in center.head(budget).itertuples(index=False)
        ]
        return selected, {
            "center_unique_before_fill": budget,
            "center_redundant_dropped": 0,
            "alternate_added": 0,
            "center_refilled": 0,
        }

    if policy == "center_unique_lcr_fill20":
        alt_views = ["left_square", "right_square"]
    elif policy == "center_unique_allview_fill20":
        alt_views = ["left_square", "right_square", "resize_square"]
    else:
        raise ValueError(f"Unknown policy: {policy}")

    retained, dropped = compress_center(
        qid=qid,
        center=center,
        centers=centers,
        source_depth=center_source_depth,
        radius_m=radius_m,
    )

    if len(retained) > budget:
        raise RuntimeError(
            f"q{qid}: center compression retained {len(retained)} > budget {budget}"
        )

    selected = list(retained)
    seen = {str(item["tile_id"]) for item in selected}
    retained_count = len(selected)

    round_robin_alternates(
        qid=qid,
        by_view=by_view,
        alt_views=alt_views,
        selected=selected,
        seen=seen,
        centers=centers,
        source_depth=alternate_source_depth,
        radius_m=radius_m,
        budget=budget,
    )

    after_alt = len(selected)

    deterministic_refill(
        qid=qid,
        selected=selected,
        seen=seen,
        dropped_center=dropped,
        by_view=by_view,
        alt_views=alt_views,
        source_depth=alternate_source_depth,
        budget=budget,
    )

    if len(selected) != budget:
        raise RuntimeError(
            f"q{qid} {policy}: expected {budget}, got {len(selected)}"
        )

    center_refilled = sum(
        item["selection_reason"] == "refill_dropped_center"
        for item in selected
    )

    return selected, {
        "center_unique_before_fill": int(retained_count),
        "center_redundant_dropped": int(len(dropped)),
        "alternate_added": int(after_alt - retained_count),
        "center_refilled": int(center_refilled),
    }


def rescue_loss_stats(
    candidate: pd.DataFrame,
    baseline: pd.DataFrame,
) -> dict[str, dict[str, int]]:
    merged = candidate.merge(
        baseline,
        on="query_id",
        suffixes=("_candidate", "_baseline"),
        validate="one_to_one",
    )

    out = {}

    for metric in ("contain_r20", "le40_r20", "le80_r20"):
        c = merged[f"{metric}_candidate"].astype(bool)
        b = merged[f"{metric}_baseline"].astype(bool)

        out[metric] = {
            "rescues": int((c & ~b).sum()),
            "losses": int((~c & b).sum()),
            "both_good": int((c & b).sum()),
            "both_bad": int((~c & ~b).sum()),
            "net": int(c.sum() - b.sum()),
        }

    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--qv1-1-root", type=Path, default=DEFAULT_QV11_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()

    cfg = yaml.safe_load(resolve(args.config).read_text())
    qv = cfg["qv1_4"]
    control = cfg["control"]

    qv11_root = resolve(args.qv1_1_root)
    output_root = resolve(args.output_root)
    ranking_dir = qv11_root / "blind_rankings"

    ranking_paths = {
        view: ranking_dir / f"qv1_1_{view}_ranking.csv"
        for view in (
            "center_square",
            "left_square",
            "right_square",
            "resize_square",
        )
    }

    for path in ranking_paths.values():
        if not path.exists():
            raise FileNotFoundError(path)

    map_index_path = resolve(control["map_index"])
    reference_path = resolve(control["reference_attachment"])

    rankings = {}
    for view, path in ranking_paths.items():
        frame = pd.read_csv(path).copy()
        frame["query_id"] = pd.to_numeric(
            frame["query_id"], errors="raise"
        ).astype(int)
        frame["rank"] = pd.to_numeric(
            frame["rank"], errors="raise"
        ).astype(int)
        frame["tile_id"] = frame["tile_id"].astype(str)
        rankings[view] = frame

    map_index = pd.read_csv(map_index_path).copy()
    map_index["tile_id"] = map_index["tile_id"].astype(str)
    centers = tile_centers(map_index)

    qids = sorted(
        rankings["center_square"]["query_id"]
        .drop_duplicates()
        .astype(int)
        .tolist()
    )

    budget = int(qv["final_budget"])
    center_depth = int(qv["center_source_depth"])
    alt_depth = int(qv["alternate_source_depth"])
    radius = float(qv["spatial_redundancy_radius_m"])

    output_root.mkdir(parents=True, exist_ok=True)
    blind_dir = output_root / "blind_pools"
    eval_dir = output_root / "postfreeze_evaluation"
    blind_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    pools = {}
    build_stats = {}
    blind_hashes = {}

    for policy in qv["policies"]:
        rows = []
        stats_rows = []

        for qid in qids:
            selected, stats = build_pool(
                qid=int(qid),
                rankings=rankings,
                centers=centers,
                policy=str(policy),
                budget=budget,
                center_source_depth=center_depth,
                alternate_source_depth=alt_depth,
                radius_m=radius,
            )

            stats_rows.append(
                {
                    "query_id": int(qid),
                    **stats,
                }
            )

            for pool_rank, item in enumerate(selected, start=1):
                rows.append(
                    {
                        **item,
                        "pool_rank": int(pool_rank),
                    }
                )

        pool = pd.DataFrame(rows)
        stats_df = pd.DataFrame(stats_rows)

        counts = pool.groupby("query_id").size()
        if not (counts == budget).all():
            raise RuntimeError(
                f"{policy}: non-{budget} pool detected"
            )

        # Top1 must remain historical center Top1.
        top1 = pool[pool["pool_rank"] == 1]
        if not (top1["source_view"] == "center_square").all():
            raise RuntimeError(
                f"{policy}: center Top1 was not preserved"
            )
        if not (top1["source_rank"] == 1).all():
            raise RuntimeError(
                f"{policy}: historical center rank1 was not preserved"
            )

        path = blind_dir / f"qv1_4_{policy}.csv"
        stats_path = blind_dir / f"qv1_4_{policy}_build_stats.csv"
        pool.to_csv(path, index=False)
        stats_df.to_csv(stats_path, index=False)

        pools[str(policy)] = pool
        build_stats[str(policy)] = stats_df
        blind_hashes[str(policy)] = {
            "pool": sha256_file(path),
            "stats": sha256_file(stats_path),
        }

    # Post-freeze reference attachment only after all pools are written.
    reference = load_reference_xy(reference_path)

    summaries = {}
    eval_frames = {}
    source_stats = {}
    named = {}

    for policy, pool in pools.items():
        query_eval, summary = evaluate_pool(
            pool,
            map_index,
            reference,
        )
        summaries[policy] = summary
        eval_frames[policy] = query_eval

        stats_df = build_stats[policy]

        source_stats[policy] = {
            "mean_center_unique_before_fill": float(
                stats_df["center_unique_before_fill"].mean()
            ),
            "median_center_unique_before_fill": float(
                stats_df["center_unique_before_fill"].median()
            ),
            "mean_center_redundant_dropped": float(
                stats_df["center_redundant_dropped"].mean()
            ),
            "mean_alternate_added": float(
                stats_df["alternate_added"].mean()
            ),
            "mean_center_refilled": float(
                stats_df["center_refilled"].mean()
            ),
            "mean_final_center_candidates": float(
                (pool["source_view"] == "center_square").sum() / len(qids)
            ),
            "mean_final_left_candidates": float(
                (pool["source_view"] == "left_square").sum() / len(qids)
            ),
            "mean_final_right_candidates": float(
                (pool["source_view"] == "right_square").sum() / len(qids)
            ),
            "mean_final_resize_candidates": float(
                (pool["source_view"] == "resize_square").sum() / len(qids)
            ),
        }

        eval_path = eval_dir / f"qv1_4_{policy}_query_eval.csv"
        query_eval.to_csv(eval_path, index=False)

        for qid in qv["named_diagnostics"]:
            qid = int(qid)
            key = f"q{qid}"
            named.setdefault(key, {})

            erow = query_eval[query_eval["query_id"] == qid]
            prow = pool[pool["query_id"] == qid].sort_values("pool_rank")
            srow = stats_df[stats_df["query_id"] == qid]

            named[key][policy] = {
                "evaluation": (
                    erow.iloc[0].to_dict()
                    if len(erow) == 1
                    else None
                ),
                "build_stats": (
                    srow.iloc[0].to_dict()
                    if len(srow) == 1
                    else None
                ),
                "pool": prow[
                    [
                        "pool_rank",
                        "tile_id",
                        "source_view",
                        "source_rank",
                        "selection_reason",
                    ]
                ].to_dict("records"),
            }

    for policy in qv["policies"]:
        pool_path = blind_dir / f"qv1_4_{policy}.csv"
        stats_path = blind_dir / f"qv1_4_{policy}_build_stats.csv"

        if sha256_file(pool_path) != blind_hashes[str(policy)]["pool"]:
            raise RuntimeError(
                f"{policy}: blind pool changed during evaluation"
            )
        if sha256_file(stats_path) != blind_hashes[str(policy)]["stats"]:
            raise RuntimeError(
                f"{policy}: blind build stats changed during evaluation"
            )

    baseline = eval_frames["center_top20"]

    rescue_loss = {
        policy: rescue_loss_stats(
            frame,
            baseline,
        )
        for policy, frame in eval_frames.items()
    }

    deltas = {
        policy: {
            "contain_r20_delta": int(
                summary["contain_r20_hits"]
                - summaries["center_top20"]["contain_r20_hits"]
            ),
            "le40_r20_delta": int(
                summary["le40_r20_hits"]
                - summaries["center_top20"]["le40_r20_hits"]
            ),
            "le80_r20_delta": int(
                summary["le80_r20_hits"]
                - summaries["center_top20"]["le80_r20_hits"]
            ),
            "top1_contains_delta": int(
                summary["top1_contains_hits"]
                - summaries["center_top20"]["top1_contains_hits"]
            ),
        }
        for policy, summary in summaries.items()
    }

    report = {
        "stage": "QV1.4",
        "status": "PASS_QV1_REDUNDANCY_AWARE_MULTIVIEW_POOL",
        "created_at_utc": now_utc(),
        "scientific_question": (
            "Can alternate views replace only spatially redundant center "
            "candidates and improve a fixed Top20 without arbitrary center "
            "rank truncation?"
        ),
        "contract": qv,
        "summary": summaries,
        "delta_vs_center_top20": deltas,
        "rescue_loss_vs_center_top20": rescue_loss,
        "source_composition": source_stats,
        "named_diagnostics": named,
        "blind_pool_freeze": {
            policy: {
                "pool_path": str(
                    blind_dir / f"qv1_4_{policy}.csv"
                ),
                "pool_sha256": blind_hashes[str(policy)]["pool"],
                "build_stats_path": str(
                    blind_dir / f"qv1_4_{policy}_build_stats.csv"
                ),
                "build_stats_sha256": blind_hashes[str(policy)]["stats"],
            }
            for policy in qv["policies"]
        },
        "scope_guarantees": {
            "new_descriptor_inference_run": False,
            "orb_run": False,
            "bootstrap_run": False,
            "state_run": False,
            "reference_used_before_pool_freeze": False,
            "final_candidate_budget": budget,
            "historical_center_top1_preserved": True,
            "arbitrary_center_rank_cutoff_used": False,
        },
    }

    report_path = (
        output_root
        / "qv1_4_redundancy_aware_multiview_pool_report.json"
    )
    report_path.write_text(
        json.dumps(
            report,
            indent=2,
            default=str,
        )
    )

    print("=" * 120)
    print(
        "QV1.4 — REDUNDANCY-AWARE FIXED-BUDGET MULTI-VIEW POOL"
    )
    print("=" * 120)

    for policy in qv["policies"]:
        s = summaries[str(policy)]
        d = deltas[str(policy)]
        c = source_stats[str(policy)]
        rl = rescue_loss[str(policy)]

        print(
            f"{policy:34s} "
            f"contain={s['contain_r20_hits']:3d}/403 "
            f"<=40={s['le40_r20_hits']:3d}/403 "
            f"<=80={s['le80_r20_hits']:3d}/403 "
            f"Top1contain={s['top1_contains_hits']:3d}/403 "
            f"dContain={d['contain_r20_delta']:+d} "
            f"d40={d['le40_r20_delta']:+d} "
            f"d80={d['le80_r20_delta']:+d} "
            f"meanCenter={c['mean_final_center_candidates']:.2f}"
        )

        if policy != "center_top20":
            print(
                "    rescue/loss: "
                f"contain +{rl['contain_r20']['rescues']}"
                f"/-{rl['contain_r20']['losses']} "
                f"<=40 +{rl['le40_r20']['rescues']}"
                f"/-{rl['le40_r20']['losses']} "
                f"<=80 +{rl['le80_r20']['rescues']}"
                f"/-{rl['le80_r20']['losses']} "
                f"centerUnique={c['mean_center_unique_before_fill']:.2f} "
                f"altAdded={c['mean_alternate_added']:.2f}"
            )

    for qid in qv["named_diagnostics"]:
        print()
        print(f"q{int(qid)}:")

        for policy in qv["policies"]:
            item = named[f"q{int(qid)}"][str(policy)]
            ev = item["evaluation"]
            bs = item["build_stats"]

            print(
                f"  {policy}: "
                f"contain={ev['contain_r20']} "
                f"<=40={ev['le40_r20']} "
                f"<=80={ev['le80_r20']} "
                f"firstContain={ev['first_contain_rank']} "
                f"centerUnique={bs['center_unique_before_fill']} "
                f"altAdded={bs['alternate_added']}"
            )

    print()
    print(
        "STATUS: PASS_QV1_REDUNDANCY_AWARE_MULTIVIEW_POOL"
    )
    print(
        "report:",
        report_path,
    )


if __name__ == "__main__":
    main()
