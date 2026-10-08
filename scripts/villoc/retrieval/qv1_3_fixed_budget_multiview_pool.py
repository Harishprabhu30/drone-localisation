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
from pyproj import Transformer


ROOT = Path.cwd().resolve()
DEFAULT_CONFIG = Path(
    "configs/research/query_view_candidate_generation_v1.yaml"
)
DEFAULT_QV11_ROOT = Path(
    "outputs/research_runs/query_view_candidate_generation_v1/qv1_1"
)
DEFAULT_OUTPUT_ROOT = Path(
    "outputs/research_runs/query_view_candidate_generation_v1/qv1_3"
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


def load_reference_xy(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path).copy()
    frame["query_id"] = pd.to_numeric(
        frame["query_id"], errors="raise"
    ).astype(int)

    transformer = Transformer.from_crs(
        "EPSG:4326",
        "EPSG:3346",
        always_xy=True,
    )
    x, y = transformer.transform(
        pd.to_numeric(frame["eval_ref_lon"], errors="raise").to_numpy(float),
        pd.to_numeric(frame["eval_ref_lat"], errors="raise").to_numpy(float),
    )
    frame["gt_x"] = x
    frame["gt_y"] = y
    return frame.set_index("query_id")


def tile_centers(map_index: pd.DataFrame) -> dict[str, tuple[float, float]]:
    out = {}
    for row in map_index.itertuples(index=False):
        out[str(row.tile_id)] = (
            float(row.center_easting),
            float(row.center_northing),
        )
    return out


def spatially_distinct(
    tile_id: str,
    selected: list[dict[str, Any]],
    centers: dict[str, tuple[float, float]],
    radius_m: float,
) -> bool:
    if tile_id not in centers:
        raise RuntimeError(f"Missing map center for {tile_id}")

    c = centers[tile_id]

    for item in selected:
        other = str(item["tile_id"])
        oc = centers[other]
        if math.hypot(c[0] - oc[0], c[1] - oc[1]) <= float(radius_m):
            return False

    return True


def build_pool(
    *,
    qid: int,
    rankings: dict[str, pd.DataFrame],
    centers: dict[str, tuple[float, float]],
    policy: str,
    budget: int,
    protected_center_depth: int,
    alternate_source_depth: int,
    radius_m: float,
) -> list[dict[str, Any]]:
    by_view = {
        view: frame[frame["query_id"] == qid].sort_values("rank")
        for view, frame in rankings.items()
    }

    center = by_view["center_square"]

    if policy == "center_top20":
        rows = []
        for row in center.head(budget).itertuples(index=False):
            rows.append(
                {
                    "query_id": qid,
                    "tile_id": str(row.tile_id),
                    "source_view": "center_square",
                    "source_rank": int(row.rank),
                    "selection_reason": "center_control",
                }
            )
        return rows

    if policy == "center10_lcr_diverse20":
        alt_views = ["left_square", "right_square"]
    elif policy == "center10_allview_diverse20":
        alt_views = ["left_square", "right_square", "resize_square"]
    else:
        raise ValueError(f"Unknown policy: {policy}")

    selected: list[dict[str, Any]] = []
    seen = set()

    for row in center.head(protected_center_depth).itertuples(index=False):
        tile_id = str(row.tile_id)
        if tile_id in seen:
            continue
        selected.append(
            {
                "query_id": qid,
                "tile_id": tile_id,
                "source_view": "center_square",
                "source_rank": int(row.rank),
                "selection_reason": "protected_center",
            }
        )
        seen.add(tile_id)

    alt_lists = {
        view: list(
            by_view[view].head(alternate_source_depth).itertuples(index=False)
        )
        for view in alt_views
    }

    depth = max(len(v) for v in alt_lists.values())

    for rank_idx in range(depth):
        for view in alt_views:
            if len(selected) >= budget:
                break
            rows = alt_lists[view]
            if rank_idx >= len(rows):
                continue
            row = rows[rank_idx]
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
                {
                    "query_id": qid,
                    "tile_id": tile_id,
                    "source_view": view,
                    "source_rank": int(row.rank),
                    "selection_reason": "alternate_spatial_diversity",
                }
            )
            seen.add(tile_id)
        if len(selected) >= budget:
            break

    refill_sources = [
        ("center_square", center.iloc[protected_center_depth:budget]),
        *[
            (view, by_view[view].head(alternate_source_depth))
            for view in alt_views
        ],
    ]

    for view, frame in refill_sources:
        for row in frame.itertuples(index=False):
            if len(selected) >= budget:
                break
            tile_id = str(row.tile_id)
            if tile_id in seen:
                continue
            selected.append(
                {
                    "query_id": qid,
                    "tile_id": tile_id,
                    "source_view": view,
                    "source_rank": int(row.rank),
                    "selection_reason": "deterministic_refill",
                }
            )
            seen.add(tile_id)
        if len(selected) >= budget:
            break

    if len(selected) != budget:
        raise RuntimeError(
            f"q{qid} {policy}: expected {budget} candidates, got {len(selected)}"
        )

    return selected


def evaluate_pool(
    pool: pd.DataFrame,
    map_index: pd.DataFrame,
    reference: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    lookup = map_index.assign(
        tile_id=map_index["tile_id"].astype(str)
    ).set_index("tile_id")

    rows = []

    for qid, group in pool.groupby("query_id", sort=True):
        ref = reference.loc[int(qid)]
        x = float(ref["gt_x"])
        y = float(ref["gt_y"])

        evaluated = []

        for row in group.sort_values("pool_rank").itertuples(index=False):
            tile = lookup.loc[str(row.tile_id)]
            err = math.hypot(
                float(tile["center_easting"]) - x,
                float(tile["center_northing"]) - y,
            )
            contains = (
                float(tile["left_easting"]) <= x <= float(tile["right_easting"])
                and float(tile["bottom_northing"]) <= y <= float(tile["top_northing"])
            )
            evaluated.append(
                {
                    "pool_rank": int(row.pool_rank),
                    "tile_id": str(row.tile_id),
                    "source_view": str(row.source_view),
                    "source_rank": int(row.source_rank),
                    "selection_reason": str(row.selection_reason),
                    "center_error_m": float(err),
                    "contains": bool(contains),
                    "le40": bool(err <= 40.0),
                    "le80": bool(err <= 80.0),
                }
            )

        contain = [r for r in evaluated if r["contains"]]
        le40 = [r for r in evaluated if r["le40"]]
        le80 = [r for r in evaluated if r["le80"]]

        rows.append(
            {
                "query_id": int(qid),
                "contain_r20": bool(contain),
                "le40_r20": bool(le40),
                "le80_r20": bool(le80),
                "first_contain_rank": contain[0]["pool_rank"] if contain else None,
                "first_le40_rank": le40[0]["pool_rank"] if le40 else None,
                "first_le80_rank": le80[0]["pool_rank"] if le80 else None,
                "top1_tile_id": evaluated[0]["tile_id"],
                "top1_source_view": evaluated[0]["source_view"],
                "top1_center_error_m": evaluated[0]["center_error_m"],
                "top1_contains": evaluated[0]["contains"],
            }
        )

    query_eval = pd.DataFrame(rows)

    summary = {
        "query_count": int(len(query_eval)),
        "contain_r20_hits": int(query_eval["contain_r20"].sum()),
        "le40_r20_hits": int(query_eval["le40_r20"].sum()),
        "le80_r20_hits": int(query_eval["le80_r20"].sum()),
        "top1_contains_hits": int(query_eval["top1_contains"].sum()),
    }

    return query_eval, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--qv1-1-root", type=Path, default=DEFAULT_QV11_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()

    cfg = yaml.safe_load(resolve(args.config).read_text())
    qv = cfg["qv1_3"]
    control = cfg["control"]

    qv11_root = resolve(args.qv1_1_root)
    output_root = resolve(args.output_root)

    ranking_dir = qv11_root / "blind_rankings"

    ranking_paths = {
        view: ranking_dir / f"qv1_1_{view}_ranking.csv"
        for view in ("center_square", "left_square", "right_square", "resize_square")
    }

    for path in ranking_paths.values():
        if not path.exists():
            raise FileNotFoundError(path)

    map_index_path = resolve(control["map_index"])
    reference_path = resolve(control["reference_attachment"])

    if not map_index_path.exists():
        raise FileNotFoundError(map_index_path)
    if not reference_path.exists():
        raise FileNotFoundError(reference_path)

    rankings = {}

    for view, path in ranking_paths.items():
        frame = pd.read_csv(path).copy()
        frame["query_id"] = pd.to_numeric(frame["query_id"], errors="raise").astype(int)
        frame["rank"] = pd.to_numeric(frame["rank"], errors="raise").astype(int)
        frame["tile_id"] = frame["tile_id"].astype(str)
        rankings[view] = frame

    map_index = pd.read_csv(map_index_path).copy()
    map_index["tile_id"] = map_index["tile_id"].astype(str)
    centers = tile_centers(map_index)

    qids = sorted(
        rankings["center_square"]["query_id"].drop_duplicates().astype(int).tolist()
    )

    budget = int(qv["final_budget"])
    protected = int(qv["protected_center_depth"])
    alt_depth = int(qv["alternate_source_depth"])
    radius = float(qv["spatial_exclusion_radius_m"])

    output_root.mkdir(parents=True, exist_ok=True)
    blind_dir = output_root / "blind_pools"
    eval_dir = output_root / "postfreeze_evaluation"
    blind_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    pools = {}
    blind_hashes = {}

    for policy in qv["policies"]:
        rows = []

        for qid in qids:
            selected = build_pool(
                qid=int(qid),
                rankings=rankings,
                centers=centers,
                policy=str(policy),
                budget=budget,
                protected_center_depth=protected,
                alternate_source_depth=alt_depth,
                radius_m=radius,
            )

            for pool_rank, item in enumerate(selected, start=1):
                rows.append(
                    {
                        **item,
                        "pool_rank": int(pool_rank),
                    }
                )

        pool = pd.DataFrame(rows)

        counts = pool.groupby("query_id").size()
        if not (counts == budget).all():
            raise RuntimeError(f"{policy}: non-{budget} query pool detected")

        path = blind_dir / f"qv1_3_{policy}.csv"
        pool.to_csv(path, index=False)
        pools[str(policy)] = pool
        blind_hashes[str(policy)] = sha256_file(path)

    # Reference is loaded only after every blind pool has been frozen.
    reference = load_reference_xy(reference_path)

    summaries = {}
    named = {}
    source_stats = {}

    for policy, pool in pools.items():
        query_eval, summary = evaluate_pool(
            pool,
            map_index,
            reference,
        )

        summaries[policy] = summary

        source_counts = (
            pool["source_view"]
            .value_counts()
            .to_dict()
        )

        source_stats[policy] = {
            "total_candidates": int(len(pool)),
            "mean_center_candidates_per_query": float(
                (pool["source_view"] == "center_square").sum() / len(qids)
            ),
            "mean_left_candidates_per_query": float(
                (pool["source_view"] == "left_square").sum() / len(qids)
            ),
            "mean_right_candidates_per_query": float(
                (pool["source_view"] == "right_square").sum() / len(qids)
            ),
            "mean_resize_candidates_per_query": float(
                (pool["source_view"] == "resize_square").sum() / len(qids)
            ),
            "source_counts": {str(k): int(v) for k, v in source_counts.items()},
        }

        eval_path = eval_dir / f"qv1_3_{policy}_query_eval.csv"
        query_eval.to_csv(eval_path, index=False)

        for qid in qv["named_diagnostics"]:
            qid = int(qid)
            key = f"q{qid}"
            named.setdefault(key, {})
            row = query_eval[query_eval["query_id"] == qid]
            pool_rows = pool[pool["query_id"] == qid].sort_values("pool_rank")

            named[key][policy] = {
                "evaluation": (
                    row.iloc[0].to_dict()
                    if len(row) == 1
                    else None
                ),
                "pool": pool_rows[
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
        path = blind_dir / f"qv1_3_{policy}.csv"
        if sha256_file(path) != blind_hashes[str(policy)]:
            raise RuntimeError(
                f"Blind pool changed during evaluation: {policy}"
            )

    baseline = summaries["center_top20"]

    deltas = {
        policy: {
            "contain_r20_delta": int(summary["contain_r20_hits"] - baseline["contain_r20_hits"]),
            "le40_r20_delta": int(summary["le40_r20_hits"] - baseline["le40_r20_hits"]),
            "le80_r20_delta": int(summary["le80_r20_hits"] - baseline["le80_r20_hits"]),
            "top1_contains_delta": int(summary["top1_contains_hits"] - baseline["top1_contains_hits"]),
        }
        for policy, summary in summaries.items()
    }

    report = {
        "stage": "QV1.3",
        "status": "PASS_QV1_FIXED_BUDGET_MULTIVIEW_POOL",
        "created_at_utc": now_utc(),
        "scientific_question": (
            "Can complementary query views improve a fixed Top20 candidate pool "
            "while preserving center-square Top1?"
        ),
        "contract": qv,
        "summary": summaries,
        "delta_vs_center_top20": deltas,
        "source_composition": source_stats,
        "named_diagnostics": named,
        "blind_pool_freeze": {
            policy: {
                "path": str(blind_dir / f"qv1_3_{policy}.csv"),
                "sha256": blind_hashes[str(policy)],
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
            "center_top1_preserved": True,
        },
    }

    report_path = output_root / "qv1_3_fixed_budget_multiview_pool_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str))

    print("=" * 112)
    print("QV1.3 — FIXED-BUDGET MULTI-VIEW CANDIDATE POOL")
    print("=" * 112)

    for policy in qv["policies"]:
        s = summaries[str(policy)]
        d = deltas[str(policy)]
        src = source_stats[str(policy)]
        print(
            f"{policy:32s} "
            f"contain={s['contain_r20_hits']:3d}/403 "
            f"<=40={s['le40_r20_hits']:3d}/403 "
            f"<=80={s['le80_r20_hits']:3d}/403 "
            f"Top1contain={s['top1_contains_hits']:3d}/403 "
            f"dContain={d['contain_r20_delta']:+d} "
            f"d40={d['le40_r20_delta']:+d} "
            f"d80={d['le80_r20_delta']:+d} "
            f"meanCenter={src['mean_center_candidates_per_query']:.2f}"
        )

    for qid in qv["named_diagnostics"]:
        print()
        print(f"q{int(qid)}:")
        for policy in qv["policies"]:
            item = named[f"q{int(qid)}"][str(policy)]
            ev = item["evaluation"]
            useful = [
                row
                for row in item["pool"]
                if row["source_view"] != "center_square"
            ]
            print(
                f"  {policy}: "
                f"contain={ev['contain_r20']} "
                f"<=40={ev['le40_r20']} "
                f"<=80={ev['le80_r20']} "
                f"firstContain={ev['first_contain_rank']} "
                f"altCandidates={len(useful)}"
            )

    print()
    print("STATUS: PASS_QV1_FIXED_BUDGET_MULTIVIEW_POOL")
    print("report:", report_path)


if __name__ == "__main__":
    main()
