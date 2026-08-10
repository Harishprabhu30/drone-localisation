#!/usr/bin/env python3
"""
R4.9 — sub-tile geometric localization feasibility audit.

Goal
----
Determine whether the existing ORB verification frontend can support
continuous within-tile map observations rather than using satellite
tile centers as exact absolute positions.

This is a READ-ONLY / BLIND-SAFE audit.

No GT/reference/SRT/GPS is read.
No localization algorithm is modified.
No ORB matching is rerun.

The audit checks:
  1. current all-candidate CSV contract;
  2. whether homography matrices / matched points are persisted;
  3. source locations where homography is computed;
  4. likely homography direction from source usage;
  5. availability of image paths and tile map bounds;
  6. homography support statistics in the initialization prefix.

Command:

SCRIPT=scripts/villoc/research/minimum_confident_bootstrap/diagnostics/r4_9_subtile_geometry_feasibility_audit.py
RUN=outputs/demo_runs/traj01_blind_regression_001
R3=outputs/research_runs/minimum_confident_bootstrap/traj01_blind_r3_001
python "$SCRIPT" \
  --repo-root "$PWD" \
  --run-root "$RUN" \
  --research-root "$R3" \
  --prefix-max-query 38 \
  2>&1 | tee \
  "$R3/postfreeze_eval/r4_9_subtile_geometry_feasibility_audit.log"

"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Helpers
# ============================================================

def resolve_existing(
    repo: Path,
    value,
) -> Path | None:

    if pd.isna(value):
        return None

    text = str(value).strip()

    if not text:
        return None

    p = Path(text)

    candidates = []

    if p.is_absolute():
        candidates.append(p)

    else:
        candidates.append(
            repo / p
        )

        candidates.append(
            Path.cwd() / p
        )

    for candidate in candidates:

        if candidate.exists():
            return candidate.resolve()

    return None


def source_hits(
    repo: Path,
):

    patterns = [
        re.compile(
            r"findHomography",
            re.I,
        ),

        re.compile(
            r"perspectiveTransform",
            re.I,
        ),

        re.compile(
            r"homography_ok",
            re.I,
        ),

        re.compile(
            r"\bH\b.*=",
        ),
    ]


    roots = [
        repo / "scripts" / "villoc",
        repo / "src",
    ]


    hits = []


    for root in roots:

        if not root.exists():
            continue


        for path in root.rglob("*.py"):

            # Do not audit this diagnostic itself.
            if (
                path.name
                == "r4_9_subtile_geometry_feasibility_audit.py"
            ):
                continue


            try:
                lines = (
                    path.read_text(
                        errors="replace"
                    )
                    .splitlines()
                )

            except Exception:
                continue


            matching_lines = []


            for line_number, line in enumerate(
                lines,
                1,
            ):

                if any(
                    pattern.search(line)
                    for pattern
                    in patterns
                ):

                    matching_lines.append(
                        (
                            line_number,
                            line.rstrip(),
                        )
                    )


            if not matching_lines:
                continue


            # Capture ±6 lines around each hit so direction
            # of src/dst points can often be inferred.
            contexts = []

            seen_ranges = set()


            for line_number, _ in matching_lines:

                start = max(
                    1,
                    line_number - 6,
                )

                end = min(
                    len(lines),
                    line_number + 6,
                )


                key = (
                    start,
                    end,
                )


                if key in seen_ranges:
                    continue


                seen_ranges.add(
                    key
                )


                contexts.append(
                    {
                        "start_line":
                            start,

                        "end_line":
                            end,

                        "lines":
                            [
                                (
                                    i,
                                    lines[
                                        i - 1
                                    ]
                                )
                                for i
                                in range(
                                    start,
                                    end + 1,
                                )
                            ],
                    }
                )


            hits.append(
                {
                    "path":
                        str(
                            path.relative_to(
                                repo
                            )
                        ),

                    "direct_hits":
                        [
                            {
                                "line":
                                    int(n),

                                "text":
                                    text,
                            }
                            for n, text
                            in matching_lines
                        ],

                    "contexts":
                        contexts,
                }
            )


    return hits


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()


    parser.add_argument(
        "--repo-root",
        type=Path,
        required=True,
    )


    parser.add_argument(
        "--run-root",
        type=Path,
        required=True,
    )


    parser.add_argument(
        "--research-root",
        type=Path,
        required=True,
    )


    parser.add_argument(
        "--prefix-max-query",
        type=int,
        default=38,
    )


    args = parser.parse_args()


    repo = (
        args.repo_root
        .resolve()
    )


    run = (
        args.run_root
        .resolve()
    )


    research = (
        args.research_root
        .resolve()
    )


    max_query = int(
        args.prefix_max_query
    )


    out_dir = (
        research
        / "postfreeze_eval"
    )


    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    # ========================================================
    # Existing ORB all-candidate artifact
    # ========================================================

    candidate_csv = (
        run
        / "reports/"
          "s8_12e1_top20_verifier_reranker/"
          "512_s256_orb_hybrid_top20_img518/"
          "s8_12e1_all_candidate_verifier_scores.csv"
    )


    if not candidate_csv.exists():

        raise RuntimeError(
            "Candidate CSV not found:\n"
            + str(
                candidate_csv
            )
        )


    df = pd.read_csv(
        candidate_csv
    )


    df[
        "query_id"
    ] = pd.to_numeric(
        df[
            "query_id"
        ],
        errors="raise",
    ).astype(int)


    prefix = df[
        df[
            "query_id"
        ]
        <= max_query
    ].copy()


    # ========================================================
    # Contract inspection
    # ========================================================

    columns = list(
        df.columns
    )


    homography_like_columns = [
        c
        for c in columns
        if any(
            token in c.lower()
            for token in [
                "homography",
                "matrix",
                "transform",
                "project",
                "src_pt",
                "dst_pt",
                "keypoint",
                "match_x",
                "match_y",
            ]
        )
    ]


    map_geometry_columns = [
        c
        for c in [
            "center_easting",
            "center_northing",
            "left_easting",
            "right_easting",
            "bottom_northing",
            "top_northing",
        ]
        if c in df.columns
    ]


    image_path_columns = [
        c
        for c in [
            "image_path",
            "query_image_resolved",
            "tile_path",
            "tile_image_resolved",
        ]
        if c in df.columns
    ]


    # ========================================================
    # Image resolvability audit
    # ========================================================

    image_resolution = {}


    for column in image_path_columns:

        sample = (
            prefix[
                column
            ]
            .dropna()
            .drop_duplicates()
            .head(200)
        )


        existing = 0


        for value in sample:

            if (
                resolve_existing(
                    repo,
                    value,
                )
                is not None
            ):
                existing += 1


        image_resolution[
            column
        ] = {
            "sampled_unique_paths":
                int(
                    len(
                        sample
                    )
                ),

            "existing_paths":
                int(
                    existing
                ),
        }


    # ========================================================
    # Homography evidence statistics
    # ========================================================

    if (
        "homography_ok"
        not in prefix.columns
    ):

        raise RuntimeError(
            "homography_ok absent from "
            "all-candidate artifact."
        )


    homography_ok = (
        prefix[
            "homography_ok"
        ]
        .astype(str)
        .str.lower()
        .isin(
            [
                "true",
                "1",
                "yes",
            ]
        )
    )


    prefix[
        "__homography_ok"
    ] = homography_ok


    query_stats = (
        prefix
        .groupby(
            "query_id"
        )
        .agg(
            candidates=(
                "tile_id",
                "size",
            ),

            homography_ok_count=(
                "__homography_ok",
                "sum",
            ),

            max_inliers=(
                "inliers",
                "max",
            ),

            max_inlier_ratio=(
                "inlier_ratio",
                "max",
            ),

            max_query_coverage=(
                "query_inlier_coverage",
                "max",
            ),
        )
        .reset_index()
    )


    query_stats[
        "has_any_homography"
    ] = (
        query_stats[
            "homography_ok_count"
        ]
        > 0
    )


    # --------------------------------------------------------
    # Top-M availability specifically relevant to our current
    # minimum-confident bootstrap.
    # --------------------------------------------------------

    top4 = prefix[
        pd.to_numeric(
            prefix[
                "hybrid_rank"
            ],
            errors="coerce",
        )
        <= 4
    ].copy()


    top5 = prefix[
        pd.to_numeric(
            prefix[
                "hybrid_rank"
            ],
            errors="coerce",
        )
        <= 5
    ].copy()


    def support_summary(data):

        grouped = (
            data
            .groupby(
                "query_id"
            )[
                "__homography_ok"
            ]
            .sum()
        )


        return {
            "queries":
                int(
                    grouped.index.nunique()
                ),

            "queries_with_any_homography":
                int(
                    (
                        grouped
                        > 0
                    ).sum()
                ),

            "median_homography_candidates":
                float(
                    grouped.median()
                ),

            "max_homography_candidates":
                int(
                    grouped.max()
                ),
        }


    # ========================================================
    # Source audit
    # ========================================================

    hits = source_hits(
        repo
    )


    source_paths = [
        hit[
            "path"
        ]
        for hit in hits
    ]


    # ========================================================
    # Determine feasibility facts without guessing direction
    # ========================================================

    matrix_persisted = any(
        any(
            token in c.lower()
            for token in [
                "matrix",
                "homography_matrix",
                "h_00",
                "h00",
            ]
        )
        for c in homography_like_columns
    )


    point_correspondences_persisted = any(
        any(
            token in c.lower()
            for token in [
                "src_pt",
                "dst_pt",
                "keypoint",
                "match_x",
                "match_y",
            ]
        )
        for c in homography_like_columns
    )


    enough_tile_geometry = (
        len(
            map_geometry_columns
        )
        == 6
    )


    top4_support = support_summary(
        top4
    )


    top5_support = support_summary(
        top5
    )


    all20_support = support_summary(
        prefix
    )


    # ========================================================
    # Save report
    # ========================================================

    report = {
        "stage":
            "R4.9_SUBTILE_GEOMETRY_FEASIBILITY_AUDIT",

        "contract": {
            "gt_used":
                False,

            "reference_used":
                False,

            "srt_used":
                False,

            "gps_used":
                False,

            "algorithm_modified":
                False,

            "orb_rerun":
                False,
        },

        "candidate_artifact": {
            "path":
                str(
                    candidate_csv
                ),

            "rows_full":
                int(
                    len(
                        df
                    )
                ),

            "rows_prefix":
                int(
                    len(
                        prefix
                    )
                ),

            "query_min":
                int(
                    prefix[
                        "query_id"
                    ].min()
                ),

            "query_max":
                int(
                    prefix[
                        "query_id"
                    ].max()
                ),

            "columns":
                columns,

            "homography_like_columns":
                homography_like_columns,

            "homography_matrix_persisted":
                bool(
                    matrix_persisted
                ),

            "point_correspondences_persisted":
                bool(
                    point_correspondences_persisted
                ),

            "map_geometry_columns":
                map_geometry_columns,

            "enough_tile_geometry_for_pixel_to_map_conversion":
                bool(
                    enough_tile_geometry
                ),

            "image_path_columns":
                image_path_columns,

            "image_path_resolution":
                image_resolution,
        },

        "homography_support": {
            "top4":
                top4_support,

            "top5":
                top5_support,

            "top20":
                all20_support,
        },

        "source_scan": {
            "files_with_homography_related_code":
                source_paths,

            "hits":
                hits,
        },

        "feasibility": {
            "existing_csv_direct_subtile_reconstruction_possible":
                bool(
                    matrix_persisted
                    or point_correspondences_persisted
                ),

            "recompute_from_existing_images_potentially_possible":
                bool(
                    enough_tile_geometry
                    and len(
                        source_paths
                    )
                    > 0
                ),
        },
    }


    report_path = (
        out_dir
        / "r4_9_subtile_geometry_feasibility_audit.json"
    )


    query_path = (
        out_dir
        / "r4_9_homography_support_by_query.csv"
    )


    report_path.write_text(
        json.dumps(
            report,
            indent=2,
        )
    )


    query_stats.to_csv(
        query_path,
        index=False,
    )


    # ========================================================
    # Print
    # ========================================================

    print()
    print("=" * 112)
    print(
        "R4.9 — SUB-TILE GEOMETRIC "
        "LOCALIZATION FEASIBILITY AUDIT"
    )
    print("=" * 112)

    print(
        "candidate rows full:",
        len(
            df
        ),
    )

    print(
        f"candidate rows q1..q{max_query}:",
        len(
            prefix
        ),
    )


    print()
    print("Persisted artifact contract")
    print("-" * 112)

    print(
        "homography-like columns:",
        homography_like_columns,
    )

    print(
        "homography matrix persisted:",
        matrix_persisted,
    )

    print(
        "matched/keypoint coordinates persisted:",
        point_correspondences_persisted,
    )

    print(
        "map geometry columns:",
        map_geometry_columns,
    )

    print(
        "enough map geometry:",
        enough_tile_geometry,
    )


    print()
    print("Image-path resolvability")
    print("-" * 112)

    for column, stats in (
        image_resolution.items()
    ):

        print(
            f"{column}: "
            f"{stats['existing_paths']}/"
            f"{stats['sampled_unique_paths']} "
            f"sample paths exist"
        )


    print()
    print("Homography-supported candidates")
    print("-" * 112)

    for name, stats in [
        (
            "Top-4 hybrid",
            top4_support,
        ),
        (
            "Top-5 hybrid",
            top5_support,
        ),
        (
            "Top-20",
            all20_support,
        ),
    ]:

        print(
            name
        )

        print(
            "  queries with any homography:",
            f"{stats['queries_with_any_homography']}/"
            f"{stats['queries']}",
        )

        print(
            "  median supported candidates/query:",
            f"{stats['median_homography_candidates']:.1f}",
        )

        print(
            "  max supported candidates/query:",
            stats[
                "max_homography_candidates"
            ],
        )


    print()
    print("Source files with homography-related code")
    print("-" * 112)

    for path in source_paths:
        print(
            " ",
            path,
        )


    print()
    print("=" * 112)
    print("SOURCE CONTEXT")
    print("=" * 112)

    for hit in hits:

        print()
        print(
            f"--- {hit['path']} ---"
        )

        for context in hit[
            "contexts"
        ]:

            print(
                f"[lines "
                f"{context['start_line']}"
                f"-"
                f"{context['end_line']}]"
            )

            for line_number, text in (
                context[
                    "lines"
                ]
            ):

                print(
                    f"{line_number:5d}: "
                    f"{text}"
                )


    print()
    print("=" * 112)
    print("R4.9 FEASIBILITY")
    print("=" * 112)

    print(
        "directly recover sub-tile geometry "
        "from existing CSV:",
        report[
            "feasibility"
        ][
            "existing_csv_direct_subtile_reconstruction_possible"
        ],
    )

    print(
        "recompute from existing images/code "
        "potentially possible:",
        report[
            "feasibility"
        ][
            "recompute_from_existing_images_potentially_possible"
        ],
    )

    print()

    print(
        "query audit:",
        query_path,
    )

    print(
        "report:",
        report_path,
    )

    print()

    print(
        "STATUS: "
        "PASS_R4_9_SUBTILE_GEOMETRY_FEASIBILITY_AUDIT"
    )


if __name__ == "__main__":
    main()
