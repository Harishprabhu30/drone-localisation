#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from uavloc.retrieval import (
    DinoV2CachedRetrievalBackend,
    fuse_physical_regions,
    reciprocal_rank,
)


ROOT = Path.cwd().resolve()

DEFAULT_CONFIG = Path(
    "configs/research/retrieval_candidate_pool_v2_r2.yaml"
)
DEFAULT_R20_REPORT = Path(
    "outputs/research_runs/retrieval_candidate_pool_v2/"
    "r2_cross_scale_fusion/reports/r2_0_fusion_contract_preflight.json"
)
DEFAULT_OUTPUT_ROOT = Path(
    "outputs/research_runs/retrieval_candidate_pool_v2/"
    "r2_cross_scale_fusion/r2_1"
)

RECALL_KS = (1, 5, 20)


def resolve(path_like: str | Path) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else ROOT / path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_query_manifest(query_rep) -> pd.DataFrame:
    source = query_rep.metadata.get("source_csv")
    if not source:
        raise RuntimeError(
            "Canonical query cache metadata does not contain source_csv."
        )

    path = resolve(source)
    if not path.exists():
        raise FileNotFoundError(path)

    frame = pd.read_csv(path)
    required = {"query_id", "easting", "northing"}
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(
            f"Query manifest missing evaluation columns {sorted(missing)}"
        )

    frame["query_id"] = frame["query_id"].astype(str)
    indexed = frame.set_index("query_id")

    wanted = query_rep.ids.astype(str)
    missing_ids = [
        qid
        for qid in wanted
        if qid not in indexed.index
    ]
    if missing_ids:
        raise RuntimeError(
            "Query manifest missing cache query IDs: "
            + ", ".join(missing_ids[:20])
        )

    return indexed.loc[wanted].reset_index()


def load_tile_index(path: Path, variant: str, map_ids: np.ndarray) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {
        "tile_id",
        "tile_path",
        "grid_row",
        "grid_col",
        "pixel_col_off",
        "pixel_row_off",
        "tile_width_px",
        "tile_height_px",
        "left_easting",
        "bottom_northing",
        "right_easting",
        "top_northing",
        "center_easting",
        "center_northing",
        "ground_width_m",
        "ground_height_m",
    }
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(
            f"{variant}: tile index missing {sorted(missing)}"
        )

    frame["tile_id"] = frame["tile_id"].astype(str)
    expected = map_ids.astype(str)
    actual = frame["tile_id"].to_numpy()

    if len(actual) != len(expected) or not np.array_equal(actual, expected):
        raise RuntimeError(
            f"{variant}: tile-index/cache ID order mismatch."
        )

    return frame.reset_index(drop=True)


def make_candidate(
    *,
    variant: str,
    rank: int,
    score: float,
    row: pd.Series,
    rrf_k: float,
) -> dict:
    return {
        "variant": variant,
        "tile_id": str(row["tile_id"]),
        "tile_path": str(row["tile_path"]),
        "per_scale_rank": int(rank),
        "dino_score": float(score),
        "rrf_contribution": float(
            reciprocal_rank(int(rank), float(rrf_k))
        ),
        "grid_row": int(row["grid_row"]),
        "grid_col": int(row["grid_col"]),
        "pixel_col_off": int(row["pixel_col_off"]),
        "pixel_row_off": int(row["pixel_row_off"]),
        "tile_width_px": int(row["tile_width_px"]),
        "tile_height_px": int(row["tile_height_px"]),
        "ground_width_m": float(row["ground_width_m"]),
        "ground_height_m": float(row["ground_height_m"]),
        "left_easting": float(row["left_easting"]),
        "bottom_northing": float(row["bottom_northing"]),
        "right_easting": float(row["right_easting"]),
        "top_northing": float(row["top_northing"]),
        "center_easting": float(row["center_easting"]),
        "center_northing": float(row["center_northing"]),
    }


def contains_point(candidate: dict, x: float, y: float) -> bool:
    return bool(
        float(candidate["left_easting"]) <= x
        <= float(candidate["right_easting"])
        and float(candidate["bottom_northing"]) <= y
        <= float(candidate["top_northing"])
    )


def center_error(candidate: dict, x: float, y: float) -> float:
    return float(
        math.hypot(
            float(candidate["center_easting"]) - x,
            float(candidate["center_northing"]) - y,
        )
    )


def evaluate_region(
    region: dict,
    x: float,
    y: float,
) -> dict:
    representative = region["representative"]
    members = region["members"]

    member_contains = [
        contains_point(member, x, y)
        for member in members
    ]
    member_errors = [
        center_error(member, x, y)
        for member in members
    ]

    rep_contains = contains_point(
        representative,
        x,
        y,
    )
    rep_error = center_error(
        representative,
        x,
        y,
    )

    best_index = int(np.argmin(member_errors))

    return {
        "region_contains_query": bool(any(member_contains)),
        "representative_contains_query": bool(rep_contains),
        "representative_center_error_m": float(rep_error),
        "best_member_center_error_m": float(min(member_errors)),
        "best_member_variant": str(members[best_index]["variant"]),
        "best_member_tile_id": str(members[best_index]["tile_id"]),
        "any_member_le40": bool(min(member_errors) <= 40.0),
        "any_member_le80": bool(min(member_errors) <= 80.0),
        "representative_le40": bool(rep_error <= 40.0),
        "representative_le80": bool(rep_error <= 80.0),
    }


def summarize_boolean(frame: pd.DataFrame, column: str) -> tuple[int, float]:
    hits = int(frame[column].sum())
    rate = float(frame[column].mean())
    return hits, rate


def first_true_rank(values: list[bool]) -> int | None:
    for index, value in enumerate(values, start=1):
        if value:
            return index
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "R2.1 blind cross-scale physical-region RRF fusion followed by "
            "post-fusion evaluation. No ORB/state replay."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
    )
    parser.add_argument(
        "--r2-0-report",
        type=Path,
        default=DEFAULT_R20_REPORT,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    args = parser.parse_args()

    config_path = resolve(args.config)
    r20_report_path = resolve(args.r2_0_report)
    output_root = resolve(args.output_root)

    if not config_path.exists():
        raise FileNotFoundError(config_path)
    if not r20_report_path.exists():
        raise FileNotFoundError(r20_report_path)

    cfg = yaml.safe_load(config_path.read_text())
    r20 = json.loads(r20_report_path.read_text())

    if r20.get("status") != "PASS_R2_FUSION_CONTRACT_PREFLIGHT":
        raise RuntimeError(
            "R2.0 fusion contract has not passed."
        )

    fusion_cfg = cfg["fusion"]
    region_cfg = fusion_cfg["physical_region"]

    backend = DinoV2CachedRetrievalBackend()
    query_cache_path = resolve(cfg["control"]["query_cache"])
    query_rep = backend.load_representation(query_cache_path)

    if len(query_rep.ids) != int(cfg["control"]["query_count_expected"]):
        raise RuntimeError("Canonical query count mismatch.")

    scale_order = list(fusion_cfg["scale_order"])
    per_scale_depth = int(fusion_cfg["per_scale_depth"])
    final_top_k = int(fusion_cfg["final_top_k"])
    rrf_k = float(fusion_cfg["rrf_k"])

    rankings = {}
    indices = {}
    map_reps = {}
    retrieval_runtime_s = {}

    print("=" * 96)
    print("R2.1 PHASE 1 — BLIND PER-SCALE RANKING")
    print("=" * 96)

    for variant in scale_order:
        spec = cfg["experts"][variant]
        map_cache_path = resolve(spec["map_cache"])
        index_path = resolve(spec["index_csv"])

        map_rep = backend.load_representation(map_cache_path)
        backend.validate_pair(query_rep, map_rep)

        tile_index = load_tile_index(
            index_path,
            variant,
            map_rep.ids,
        )

        ranking = backend.rank_batch(
            query_rep,
            map_rep,
            per_scale_depth,
        )

        rankings[variant] = ranking
        indices[variant] = tile_index
        map_reps[variant] = map_rep
        retrieval_runtime_s[variant] = float(
            ranking.retrieval_runtime_s
        )

        print(
            f"{variant:10s} "
            f"tiles={len(map_rep.ids):3d} "
            f"depth={per_scale_depth} "
            f"runtime_ms/query="
            f"{1000.0 * ranking.retrieval_runtime_s / len(query_rep.ids):.4f}"
        )

    # --------------------------------------------------------------
    # PHASE 2 — blind physical-region fusion.
    # No query coordinates/reference are loaded yet.
    # --------------------------------------------------------------
    print()
    print("=" * 96)
    print("R2.1 PHASE 2 — BLIND PHYSICAL-REGION RRF FUSION")
    print("=" * 96)

    fused_by_experiment: dict[str, dict[str, list[dict]]] = {}
    blind_candidate_rows = []
    fusion_runtime_s = {}

    for experiment_name, experiment in cfg["experiments"].items():
        experts = list(experiment["experts"])
        experiment_scale_order = [
            name
            for name in scale_order
            if name in experts
        ]

        started = time.perf_counter()
        query_regions = {}

        for query_index, query_id in enumerate(
            query_rep.ids.astype(str)
        ):
            pool = []

            for variant in experiment_scale_order:
                ranking = rankings[variant]
                tile_index = indices[variant]

                for local_rank, (
                    map_index,
                    score,
                ) in enumerate(
                    zip(
                        ranking.indices[query_index],
                        ranking.scores[query_index],
                    ),
                    start=1,
                ):
                    row = tile_index.iloc[int(map_index)]

                    pool.append(
                        make_candidate(
                            variant=variant,
                            rank=local_rank,
                            score=float(score),
                            row=row,
                            rrf_k=rrf_k,
                        )
                    )

            regions = fuse_physical_regions(
                pool,
                scale_order=experiment_scale_order,
                max_center_distance_m=float(
                    region_cfg["max_center_distance_m"]
                ),
                min_iou=float(
                    region_cfg["minimum_bbox_iou"]
                ),
                final_top_k=final_top_k,
            )
            query_regions[str(query_id)] = regions

            for region in regions:
                representative = region["representative"]

                blind_candidate_rows.append(
                    {
                        "experiment": experiment_name,
                        "query_id": str(query_id),
                        "fused_rank": int(region["fused_rank"]),
                        "fused_rrf_score": float(
                            region["fused_rrf_score"]
                        ),
                        "support_scale_count": int(
                            region["support_scale_count"]
                        ),
                        "support_scales": "|".join(
                            region["support_scales"]
                        ),
                        "representative_variant": str(
                            representative["variant"]
                        ),
                        "representative_tile_id": str(
                            representative["tile_id"]
                        ),
                        "representative_tile_path": str(
                            representative["tile_path"]
                        ),
                        "representative_per_scale_rank": int(
                            representative["per_scale_rank"]
                        ),
                        "representative_dino_score": float(
                            representative["dino_score"]
                        ),
                        "representative_grid_row": int(
                            representative["grid_row"]
                        ),
                        "representative_grid_col": int(
                            representative["grid_col"]
                        ),
                        "representative_pixel_col_off": int(
                            representative["pixel_col_off"]
                        ),
                        "representative_pixel_row_off": int(
                            representative["pixel_row_off"]
                        ),
                        "representative_tile_width_px": int(
                            representative["tile_width_px"]
                        ),
                        "representative_ground_width_m": float(
                            representative["ground_width_m"]
                        ),
                        "representative_left_easting": float(
                            representative["left_easting"]
                        ),
                        "representative_bottom_northing": float(
                            representative["bottom_northing"]
                        ),
                        "representative_right_easting": float(
                            representative["right_easting"]
                        ),
                        "representative_top_northing": float(
                            representative["top_northing"]
                        ),
                        "representative_center_easting": float(
                            representative["center_easting"]
                        ),
                        "representative_center_northing": float(
                            representative["center_northing"]
                        ),
                        "members_json": json.dumps(
                            [
                                {
                                    "variant": member["variant"],
                                    "tile_id": member["tile_id"],
                                    "per_scale_rank": member["per_scale_rank"],
                                    "dino_score": member["dino_score"],
                                    "rrf_contribution": member[
                                        "rrf_contribution"
                                    ],
                                    "grid_row": member["grid_row"],
                                    "grid_col": member["grid_col"],
                                    "left_easting": member["left_easting"],
                                    "bottom_northing": member[
                                        "bottom_northing"
                                    ],
                                    "right_easting": member["right_easting"],
                                    "top_northing": member["top_northing"],
                                    "center_easting": member[
                                        "center_easting"
                                    ],
                                    "center_northing": member[
                                        "center_northing"
                                    ],
                                    "tile_path": member["tile_path"],
                                }
                                for member in region["members"]
                            ],
                            sort_keys=True,
                        ),
                    }
                )

        fused_by_experiment[experiment_name] = query_regions
        fusion_runtime_s[experiment_name] = float(
            time.perf_counter() - started
        )

        print(
            f"{experiment_name:24s} "
            f"experts={'+'.join(experiment_scale_order)} "
            f"Top-{final_top_k} "
            f"fusion_ms/query="
            f"{1000.0 * fusion_runtime_s[experiment_name] / len(query_rep.ids):.4f}"
        )

    blind_candidates = pd.DataFrame(blind_candidate_rows)

    output_root.mkdir(parents=True, exist_ok=True)
    blind_candidates_path = (
        output_root
        / "r2_1_blind_fused_top20.csv"
    )
    blind_candidates.to_csv(
        blind_candidates_path,
        index=False,
    )
    blind_candidates_sha = sha256_file(blind_candidates_path)

    # --------------------------------------------------------------
    # PHASE 3 — evaluation attachment only after fused Top-20 freeze.
    # --------------------------------------------------------------
    print()
    print("=" * 96)
    print("R2.1 PHASE 3 — POST-FUSION EVALUATION ATTACHMENT")
    print("=" * 96)

    query_manifest = load_query_manifest(query_rep)

    query_eval_rows = []
    region_eval_rows = []
    experiment_summaries = {}

    for experiment_name, query_regions in fused_by_experiment.items():
        for query_index, query_id in enumerate(
            query_rep.ids.astype(str)
        ):
            qrow = query_manifest.iloc[query_index]
            x = float(qrow["easting"])
            y = float(qrow["northing"])

            regions = query_regions[query_id]
            region_eval = [
                evaluate_region(region, x, y)
                for region in regions
            ]

            for region, evaluation in zip(regions, region_eval):
                representative = region["representative"]
                region_eval_rows.append(
                    {
                        "experiment": experiment_name,
                        "query_id": query_id,
                        "fused_rank": int(region["fused_rank"]),
                        "representative_variant": str(
                            representative["variant"]
                        ),
                        "representative_tile_id": str(
                            representative["tile_id"]
                        ),
                        **evaluation,
                    }
                )

            row = {
                "experiment": experiment_name,
                "query_id": query_id,
                "top1_region_contains": bool(
                    region_eval[0]["region_contains_query"]
                ),
                "top1_representative_contains": bool(
                    region_eval[0][
                        "representative_contains_query"
                    ]
                ),
                "top1_representative_center_error_m": float(
                    region_eval[0][
                        "representative_center_error_m"
                    ]
                ),
                "top1_best_member_center_error_m": float(
                    region_eval[0][
                        "best_member_center_error_m"
                    ]
                ),
                "first_region_contains_rank": (
                    first_true_rank(
                        [
                            item["region_contains_query"]
                            for item in region_eval
                        ]
                    )
                    or np.nan
                ),
                "first_representative_contains_rank": (
                    first_true_rank(
                        [
                            item["representative_contains_query"]
                            for item in region_eval
                        ]
                    )
                    or np.nan
                ),
                "first_region_le40_rank": (
                    first_true_rank(
                        [
                            item["any_member_le40"]
                            for item in region_eval
                        ]
                    )
                    or np.nan
                ),
                "first_region_le80_rank": (
                    first_true_rank(
                        [
                            item["any_member_le80"]
                            for item in region_eval
                        ]
                    )
                    or np.nan
                ),
                "first_representative_le40_rank": (
                    first_true_rank(
                        [
                            item["representative_le40"]
                            for item in region_eval
                        ]
                    )
                    or np.nan
                ),
                "first_representative_le80_rank": (
                    first_true_rank(
                        [
                            item["representative_le80"]
                            for item in region_eval
                        ]
                    )
                    or np.nan
                ),
                "top20_region_contains": bool(
                    any(
                        item["region_contains_query"]
                        for item in region_eval
                    )
                ),
                "top20_representative_contains": bool(
                    any(
                        item["representative_contains_query"]
                        for item in region_eval
                    )
                ),
                "top20_region_le40": bool(
                    any(item["any_member_le40"] for item in region_eval)
                ),
                "top20_region_le80": bool(
                    any(item["any_member_le80"] for item in region_eval)
                ),
                "top20_representative_le40": bool(
                    any(
                        item["representative_le40"]
                        for item in region_eval
                    )
                ),
                "top20_representative_le80": bool(
                    any(
                        item["representative_le80"]
                        for item in region_eval
                    )
                ),
                "top20_multiscale_region_count": int(
                    sum(
                        int(region["support_scale_count"] > 1)
                        for region in regions
                    )
                ),
                "top20_mean_support_scale_count": float(
                    np.mean(
                        [
                            region["support_scale_count"]
                            for region in regions
                        ]
                    )
                ),
                "representative_512_count": int(
                    sum(
                        region["representative"]["variant"]
                        == "512_s256"
                        for region in regions
                    )
                ),
                "representative_768_count": int(
                    sum(
                        region["representative"]["variant"]
                        == "768_s256"
                        for region in regions
                    )
                ),
                "representative_1024_count": int(
                    sum(
                        region["representative"]["variant"]
                        == "1024_s256"
                        for region in regions
                    )
                ),
            }

            query_eval_rows.append(row)

    query_eval = pd.DataFrame(query_eval_rows)
    region_eval_df = pd.DataFrame(region_eval_rows)

    summary_rows = []

    for experiment_name, sub in query_eval.groupby(
        "experiment",
        sort=False,
    ):
        summary = {
            "experiment": experiment_name,
            "query_count": int(len(sub)),
        }

        for k in RECALL_KS:
            # R2.1 stores Top-20 regions only. @1/@5/@20 are derived here.
            region_sub = region_eval_df[
                region_eval_df["experiment"] == experiment_name
            ]

            grouped = region_sub[
                region_sub["fused_rank"] <= k
            ].groupby("query_id", sort=False)

            region_contains = grouped[
                "region_contains_query"
            ].max()
            rep_contains = grouped[
                "representative_contains_query"
            ].max()
            region_le40 = grouped[
                "any_member_le40"
            ].max()
            region_le80 = grouped[
                "any_member_le80"
            ].max()
            rep_le40 = grouped[
                "representative_le40"
            ].max()
            rep_le80 = grouped[
                "representative_le80"
            ].max()

            summary[f"region_containment_recall_at_{k}"] = float(
                region_contains.mean()
            )
            summary[
                f"representative_containment_recall_at_{k}"
            ] = float(rep_contains.mean())
            summary[f"region_le40_recall_at_{k}"] = float(
                region_le40.mean()
            )
            summary[f"region_le80_recall_at_{k}"] = float(
                region_le80.mean()
            )
            summary[f"representative_le40_recall_at_{k}"] = float(
                rep_le40.mean()
            )
            summary[f"representative_le80_recall_at_{k}"] = float(
                rep_le80.mean()
            )

        summary["multiscale_regions_top20_mean"] = float(
            sub["top20_multiscale_region_count"].mean()
        )
        summary["support_scale_count_top20_mean"] = float(
            sub["top20_mean_support_scale_count"].mean()
        )
        summary["representative_512_top20_mean"] = float(
            sub["representative_512_count"].mean()
        )
        summary["representative_768_top20_mean"] = float(
            sub["representative_768_count"].mean()
        )
        summary["representative_1024_top20_mean"] = float(
            sub["representative_1024_count"].mean()
        )
        summary["fusion_runtime_ms_per_query"] = float(
            1000.0
            * fusion_runtime_s[experiment_name]
            / len(query_rep.ids)
        )

        summary_rows.append(summary)
        experiment_summaries[experiment_name] = summary

    summary_df = pd.DataFrame(summary_rows)

    named = {}
    for qid in ("57", "228", "390"):
        named[f"q{qid}"] = {}

        for experiment_name in cfg["experiments"]:
            sub = region_eval_df[
                (region_eval_df["experiment"] == experiment_name)
                & (region_eval_df["query_id"] == qid)
            ].sort_values("fused_rank")

            blind_sub = blind_candidates[
                (blind_candidates["experiment"] == experiment_name)
                & (blind_candidates["query_id"] == qid)
            ].sort_values("fused_rank")

            if sub.empty:
                named[f"q{qid}"][experiment_name] = {
                    "present": False,
                }
                continue

            merged = blind_sub.merge(
                sub,
                on=[
                    "experiment",
                    "query_id",
                    "fused_rank",
                    "representative_variant",
                    "representative_tile_id",
                ],
                how="left",
                validate="one_to_one",
            )

            useful = merged[
                merged["region_contains_query"]
                | merged["any_member_le40"]
                | merged["any_member_le80"]
            ]

            named[f"q{qid}"][experiment_name] = {
                "present": True,
                "first_region_contains_rank": (
                    int(
                        merged.loc[
                            merged["region_contains_query"],
                            "fused_rank",
                        ].iloc[0]
                    )
                    if merged["region_contains_query"].any()
                    else None
                ),
                "first_representative_contains_rank": (
                    int(
                        merged.loc[
                            merged["representative_contains_query"],
                            "fused_rank",
                        ].iloc[0]
                    )
                    if merged["representative_contains_query"].any()
                    else None
                ),
                "top5": merged.head(5)[
                    [
                        "fused_rank",
                        "fused_rrf_score",
                        "support_scales",
                        "representative_variant",
                        "representative_tile_id",
                        "representative_per_scale_rank",
                        "region_contains_query",
                        "representative_contains_query",
                        "representative_center_error_m",
                        "best_member_center_error_m",
                        "best_member_variant",
                        "best_member_tile_id",
                    ]
                ].to_dict(orient="records"),
                "useful_regions_top20": useful[
                    [
                        "fused_rank",
                        "support_scales",
                        "representative_variant",
                        "representative_tile_id",
                        "region_contains_query",
                        "representative_contains_query",
                        "representative_center_error_m",
                        "best_member_center_error_m",
                        "best_member_variant",
                        "best_member_tile_id",
                        "members_json",
                    ]
                ].to_dict(orient="records"),
            }

    query_eval_path = (
        output_root
        / "r2_1_query_level_evaluation.csv"
    )
    region_eval_path = (
        output_root
        / "r2_1_region_level_evaluation.csv"
    )
    summary_path = (
        output_root
        / "r2_1_experiment_summary.csv"
    )
    named_path = (
        output_root
        / "r2_1_named_diagnostics.json"
    )
    report_path = (
        output_root
        / "r2_1_cross_scale_fusion_report.json"
    )

    query_eval.to_csv(query_eval_path, index=False)
    region_eval_df.to_csv(region_eval_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    named_path.write_text(
        json.dumps(named, indent=2),
        encoding="utf-8",
    )

    report = {
        "stage": "R2.1",
        "status": "PASS_R2_BLIND_CROSS_SCALE_FUSION",
        "created_at_utc": now_utc(),
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "r2_0_gate": {
            "report": str(r20_report_path),
            "report_sha256": sha256_file(r20_report_path),
            "status": r20["status"],
        },
        "query_cache": {
            "path": str(query_cache_path),
            "sha256": sha256_file(query_cache_path),
            "query_count": int(len(query_rep.ids)),
        },
        "per_scale_retrieval_runtime_s": retrieval_runtime_s,
        "fusion_runtime_s": fusion_runtime_s,
        "blind_fused_candidates": {
            "path": str(blind_candidates_path),
            "sha256": blind_candidates_sha,
            "coordinates_from_query_used": False,
            "oracle_used": False,
        },
        "summaries": experiment_summaries,
        "named_diagnostics": named,
        "outputs": {
            "query_evaluation_csv": str(query_eval_path),
            "region_evaluation_csv": str(region_eval_path),
            "summary_csv": str(summary_path),
            "named_diagnostics_json": str(named_path),
        },
        "interpretation_contract": {
            "region_metric": (
                "A fused physical region is useful if any member tile is useful."
            ),
            "representative_metric": (
                "Representative metrics evaluate the one concrete tile that "
                "would be forwarded to an unchanged ORB Top-20 interface."
            ),
            "winner_not_auto_selected": True,
        },
        "scope_guarantees": {
            "query_coordinates_used_before_fusion": False,
            "reference_used_before_fusion": False,
            "oracle_used_before_fusion": False,
            "orb_run": False,
            "bootstrap_run": False,
            "state_policy_changed": False,
            "learned_weights_used": False,
        },
        "next_stage": {
            "stage": "R2.2 decision",
            "action": (
                "Interpret pair vs triple fusion and representative-loss "
                "effects. Only promising fusion proceeds to unchanged ORB."
            ),
        },
    }

    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print()
    print("=" * 96)
    print("R2.1 — BLIND CROSS-SCALE FUSION SUMMARY")
    print("=" * 96)

    display = [
        "experiment",
        "region_containment_recall_at_1",
        "region_containment_recall_at_5",
        "region_containment_recall_at_20",
        "representative_containment_recall_at_20",
        "region_le40_recall_at_20",
        "representative_le40_recall_at_20",
        "region_le80_recall_at_20",
        "representative_le80_recall_at_20",
        "multiscale_regions_top20_mean",
        "representative_512_top20_mean",
        "representative_768_top20_mean",
        "representative_1024_top20_mean",
        "fusion_runtime_ms_per_query",
    ]

    print(summary_df[display].to_string(index=False))

    for qid in ("57", "228", "390"):
        print()
        print(f"q{qid}:")
        print(json.dumps(named[f"q{qid}"], indent=2))

    print()
    print("STATUS: PASS_R2_BLIND_CROSS_SCALE_FUSION")
    print("blind fused Top-20:", blind_candidates_path)
    print("report:", report_path)


if __name__ == "__main__":
    main()
