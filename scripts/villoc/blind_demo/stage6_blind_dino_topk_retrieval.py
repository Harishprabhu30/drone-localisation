#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROTOCOL_KEYS = [
    "model_name",
    "image_size",
    "crop_mode",
    "pooling",
    "normalization",
    "l2_normalize",
    "descriptor_dtype",
]


def load_cache(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)

    data = np.load(
        path,
        allow_pickle=False,
    )

    required = {
        "descriptors",
        "ids",
        "meta_json",
    }

    missing = (
        required
        - set(data.files)
    )

    if missing:
        raise RuntimeError(
            f"{path}: missing cache keys "
            f"{sorted(missing)}"
        )

    meta_raw = data["meta_json"]

    if hasattr(
        meta_raw,
        "item",
    ):
        meta_raw = meta_raw.item()

    meta = json.loads(
        str(meta_raw)
    )

    return {
        "descriptors": (
            data["descriptors"]
            .astype(np.float32)
        ),
        "ids": (
            data["ids"]
            .astype(str)
        ),
        "meta": meta,
    }


def protocol_signature(
    meta: dict[str, Any],
) -> dict[str, Any]:

    protocol = meta.get(
        "protocol",
        {},
    )

    return {
        key: protocol.get(key)
        for key in PROTOCOL_KEYS
    }


def validate_cache_pair(
    query: dict[str, Any],
    map_cache: dict[str, Any],
) -> None:

    q_desc = query["descriptors"]
    m_desc = map_cache["descriptors"]

    q_ids = query["ids"]
    m_ids = map_cache["ids"]

    if q_desc.ndim != 2:
        raise RuntimeError(
            "Query descriptors must be 2-D."
        )

    if m_desc.ndim != 2:
        raise RuntimeError(
            "Map descriptors must be 2-D."
        )

    if q_desc.shape[0] != len(q_ids):
        raise RuntimeError(
            "Query descriptor/ID row mismatch."
        )

    if m_desc.shape[0] != len(m_ids):
        raise RuntimeError(
            "Map descriptor/ID row mismatch."
        )

    if (
        q_desc.shape[1]
        != m_desc.shape[1]
    ):
        raise RuntimeError(
            "Query/map descriptor dimensions "
            "do not match."
        )

    if len(set(q_ids.tolist())) != len(q_ids):
        raise RuntimeError(
            "Duplicate query IDs."
        )

    if len(set(m_ids.tolist())) != len(m_ids):
        raise RuntimeError(
            "Duplicate map tile IDs."
        )

    if not np.isfinite(q_desc).all():
        raise RuntimeError(
            "Non-finite query descriptors."
        )

    if not np.isfinite(m_desc).all():
        raise RuntimeError(
            "Non-finite map descriptors."
        )

    q_sig = protocol_signature(
        query["meta"]
    )

    m_sig = protocol_signature(
        map_cache["meta"]
    )

    if q_sig != m_sig:
        raise RuntimeError(
            "Query/map DINO protocol mismatch.\n"
            f"query={q_sig}\n"
            f"map={m_sig}"
        )

    if not bool(
        q_sig.get("l2_normalize")
    ):
        raise RuntimeError(
            "Query descriptors are not marked "
            "L2 normalized."
        )

    if not bool(
        m_sig.get("l2_normalize")
    ):
        raise RuntimeError(
            "Map descriptors are not marked "
            "L2 normalized."
        )

    q_norm = np.linalg.norm(
        q_desc,
        axis=1,
    )

    m_norm = np.linalg.norm(
        m_desc,
        axis=1,
    )

    if not np.allclose(
        q_norm,
        1.0,
        atol=1e-3,
    ):
        raise RuntimeError(
            "Query descriptor norms are not "
            "approximately 1."
        )

    if not np.allclose(
        m_norm,
        1.0,
        atol=1e-3,
    ):
        raise RuntimeError(
            "Map descriptor norms are not "
            "approximately 1."
        )


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

    query = load_cache(
        query_path
    )

    map_cache = load_cache(
        map_path
    )

    validate_cache_pair(
        query,
        map_cache,
    )

    q_desc = query["descriptors"]
    m_desc = map_cache["descriptors"]

    q_ids = query["ids"]
    tile_ids = map_cache["ids"]

    top_k = min(
        args.top_k,
        len(tile_ids),
    )

    # -------------------------------------------------
    # BLIND RETRIEVAL
    #
    # Both caches were L2-normalized during descriptor
    # construction. Therefore:
    #
    #     dot product == cosine similarity
    #
    # No coordinates, oracle labels, SRT, GPS, or
    # evaluation data are loaded here.
    # -------------------------------------------------

    retrieval_started = (
        time.perf_counter()
    )

    similarity = (
        q_desc
        @ m_desc.T
    )

    order = np.argsort(
        -similarity,
        axis=1,
    )[:, :top_k]

    retrieval_finished = (
        time.perf_counter()
    )

    candidate_rows = []
    query_rows = []

    for i, query_id in enumerate(
        q_ids
    ):
        selected = order[i]

        scores = similarity[
            i,
            selected,
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
            protocol_signature(
                query["meta"]
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
                    retrieval_finished
                    - retrieval_started
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
