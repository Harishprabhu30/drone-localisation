#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd

from uavloc.retrieval import DinoV2CachedRetrievalBackend


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Blind-safe DINOv2 Top-K retrieval "
            "from frozen query/map descriptor caches."
        )
    )

    parser.add_argument(
        "--query-cache",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--map-cache",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--run-root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--variant",
        default="512_s256",
    )

    parser.add_argument(
        "--tag",
        required=True,
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
    )

    args = parser.parse_args()

    started = time.perf_counter()

    query_path = (
        args.query_cache
        .expanduser()
        .resolve()
    )

    map_path = (
        args.map_cache
        .expanduser()
        .resolve()
    )

    run_root = (
        args.run_root
        .expanduser()
        .resolve()
    )

    if args.top_k <= 0:
        raise ValueError(
            "--top-k must be positive."
        )

    backend = DinoV2CachedRetrievalBackend()

    query = backend.load_representation(
        query_path
    )

    map_cache = backend.load_representation(
        map_path
    )

    ranking = backend.rank_batch(
        query,
        map_cache,
        args.top_k,
    )

    q_desc = query.descriptors
    q_ids = query.ids
    tile_ids = map_cache.ids

    top_k = int(
        ranking.indices.shape[1]
    )

    # -------------------------------------------------
    # BLIND RETRIEVAL
    #
    # Ranking is delegated to the A1 retrieval backend:
    #
    #     L2-normalized descriptor dot product
    #         == cosine similarity
    #     full descending np.argsort
    #         -> Top-K
    #
    # No coordinates, oracle labels, SRT, GPS, or
    # evaluation data are loaded here.
    # -------------------------------------------------

    order = ranking.indices

    candidate_rows = []
    query_rows = []

    for i, query_id in enumerate(
        q_ids
    ):
        selected = order[i]

        scores = ranking.scores[
            i
        ]

        for rank, (
            map_index,
            score,
        ) in enumerate(
            zip(
                selected,
                scores,
            ),
            start=1,
        ):
            candidate_rows.append(
                {
                    "variant":
                        args.variant,
                    "query_id":
                        str(query_id),
                    "rank":
                        int(rank),
                    "tile_id":
                        str(
                            tile_ids[
                                map_index
                            ]
                        ),
                    "score":
                        float(score),
                }
            )

        top1 = float(
            scores[0]
        )

        top2 = (
            float(scores[1])
            if len(scores) > 1
            else float("nan")
        )

        query_rows.append(
            {
                "variant":
                    args.variant,
                "query_id":
                    str(query_id),
                "top1_tile_id":
                    str(
                        tile_ids[
                            selected[0]
                        ]
                    ),
                "top1_score":
                    top1,
                "top2_score":
                    top2,
                "top1_top2_margin":
                    (
                        top1 - top2
                        if np.isfinite(top2)
                        else float("nan")
                    ),
                "topk_score_span":
                    (
                        top1
                        - float(
                            scores[-1]
                        )
                    ),
            }
        )

    topk_df = pd.DataFrame(
        candidate_rows
    )

    query_df = pd.DataFrame(
        query_rows
    )

    # Strict blind schema guard.
    forbidden = {
        "latitude",
        "longitude",
        "lat",
        "lon",
        "easting",
        "northing",
        "x_enu_m",
        "y_enu_m",
        "oracle",
        "is_oracle",
        "center_error_m",
        "error_m",
        "hit_le_40m",
        "candidate_body_error_m",
        "candidate_contains_body",
    }

    leaked = sorted(
        forbidden
        & set(topk_df.columns)
    )

    if leaked:
        raise RuntimeError(
            "Reference/evaluation columns leaked "
            f"into blind Top-K output: {leaked}"
        )

    counts = (
        topk_df.groupby(
            "query_id"
        )
        .size()
    )

    if not (
        counts == top_k
    ).all():
        raise RuntimeError(
            "Every query must have exactly "
            f"{top_k} candidates."
        )

    retrieval_dir = (
        run_root
        / "retrieval"
        / "s8_11d"
    )

    report_dir = (
        run_root
        / "reports"
        / "s8_11d_blind"
    )

    retrieval_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    topk_path = (
        retrieval_dir
        / (
            "s8_11d_topk_"
            f"{args.variant}_"
            f"{args.tag}.csv"
        )
    )

    query_summary_path = (
        retrieval_dir
        / (
            "s8_11d_blind_query_summary_"
            f"{args.variant}_"
            f"{args.tag}.csv"
        )
    )

    report_path = (
        report_dir
        / (
            "s8_11d_blind_retrieval_"
            f"{args.variant}_"
            f"{args.tag}.json"
        )
    )

    topk_df.to_csv(
        topk_path,
        index=False,
    )

    query_df.to_csv(
        query_summary_path,
        index=False,
    )

    finished = time.perf_counter()

    report = {
        "stage":
            "BLIND_DINO_TOPK_RETRIEVAL",

        "status":
            "PASS_BLIND_DINO_TOPK_RETRIEVAL",

        "variant":
            args.variant,

        "descriptor_tag":
            args.tag,

        "query_count":
            int(len(q_ids)),

        "map_tile_count":
            int(len(tile_ids)),

        "descriptor_dim":
            int(q_desc.shape[1]),

        "top_k":
            int(top_k),

        "candidate_rows":
            int(len(topk_df)),

        "retrieval": {
            "similarity":
                (
                    "cosine_dot_product_on_"
                    "l2_normalized_descriptors"
                ),
            "coordinates_used":
                False,
            "oracle_used":
                False,
            "gps_used":
                False,
            "srt_used":
                False,
            "ground_truth_used":
                False,
        },

        "protocol":
            backend.protocol_signature(
                query.metadata
            ),

        "score_summary": {
            "top1_mean":
                float(
                    query_df[
                        "top1_score"
                    ].mean()
                ),
            "top1_median":
                float(
                    query_df[
                        "top1_score"
                    ].median()
                ),
            "top1_top2_margin_median":
                float(
                    query_df[
                        "top1_top2_margin"
                    ].median()
                ),
        },

        "runtime": {
            "matrix_retrieval_s":
                float(
                    ranking.retrieval_runtime_s
                ),
            "total_stage_wall_s":
                float(
                    finished
                    - started
                ),
        },

        "inputs": {
            "query_cache":
                str(query_path),
            "map_cache":
                str(map_path),
        },

        "outputs": {
            "topk_csv":
                str(topk_path),
            "query_summary_csv":
                str(query_summary_path),
            "report":
                str(report_path),
        },
    }

    report_path.write_text(
        json.dumps(
            report,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    print("=" * 80)
    print(
        "STAGE 6 — BLIND DINO TOP-K RETRIEVAL"
    )
    print("=" * 80)

    print(
        "status:",
        report["status"],
    )

    print()
    print("Inputs")
    print("-" * 80)
    print(
        "queries:",
        report["query_count"],
    )
    print(
        "map tiles:",
        report["map_tile_count"],
    )
    print(
        "descriptor dim:",
        report["descriptor_dim"],
    )

    print()
    print("Retrieval")
    print("-" * 80)
    print(
        "top-k:",
        report["top_k"],
    )
    print(
        "candidate rows:",
        report["candidate_rows"],
    )
    print(
        "coordinates used: false"
    )
    print(
        "oracle used:      false"
    )
    print(
        "GPS used:         false"
    )
    print(
        "SRT used:         false"
    )

    print()
    print("Score diagnostics")
    print("-" * 80)
    print(
        "top1 mean:",
        report[
            "score_summary"
        ]["top1_mean"],
    )
    print(
        "top1 median:",
        report[
            "score_summary"
        ]["top1_median"],
    )
    print(
        "top1-top2 margin median:",
        report[
            "score_summary"
        ][
            "top1_top2_margin_median"
        ],
    )

    print()
    print("Saved")
    print("-" * 80)
    print(topk_path)
    print(query_summary_path)
    print(report_path)


if __name__ == "__main__":
    main()
