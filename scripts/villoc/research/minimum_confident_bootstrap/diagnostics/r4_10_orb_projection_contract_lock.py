#!/usr/bin/env python3
"""
R4.10 — exact ORB projection-contract lock.

Purpose
-------
Before recomputing query→satellite homographies for sub-tile localization,
recover the exact preprocessing and verifier parameter contract used by the
existing S8.12E.1 ORB reranker.

READ-ONLY / BLIND-SAFE.

No GT/reference/SRT/GPS is read.
No ORB matching is rerun.
No production code is modified.

Outputs:
  * exact relevant candidate-table columns
  * source contexts for preprocessing and verifier parameters
  * available run metadata containing those parameters
  * raw query/tile image dimensions
  * coordinate-direction contract

Command:

SCRIPT=scripts/villoc/research/minimum_confident_bootstrap/diagnostics/r4_10_orb_projection_contract_lock.py
RUN=outputs/demo_runs/traj01_blind_regression_001
R3=outputs/research_runs/minimum_confident_bootstrap/traj01_blind_r3_001


python "$SCRIPT" \
  --repo-root "$PWD" \
  --run-root "$RUN" \
  --research-root "$R3" \
  2>&1 | tee \
  "$R3/postfreeze_eval/r4_10_orb_projection_contract_lock.log"

"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import cv2
import pandas as pd


# ============================================================
# Helpers
# ============================================================

KEYWORDS = [
    "preprocess",
    "resize_long",
    "resize-long",
    "nfeatures",
    "ratio",
    "ransac",
    "orb",
    "center_square",
    "img518",
]


def relevant_columns(columns):
    out = []

    for col in columns:
        low = col.lower()

        if any(
            key.replace("-", "_") in low
            for key in KEYWORDS
        ):
            out.append(col)

    return out


def source_contexts(
    path: Path,
):
    lines = path.read_text(
        errors="replace"
    ).splitlines()

    patterns = [
        re.compile(
            r"def read_image_for_verifier",
            re.I,
        ),
        re.compile(
            r"def .*verif",
            re.I,
        ),
        re.compile(
            r"--preprocess",
            re.I,
        ),
        re.compile(
            r"--resize[-_]long",
            re.I,
        ),
        re.compile(
            r"--nfeatures",
            re.I,
        ),
        re.compile(
            r"--ratio",
            re.I,
        ),
        re.compile(
            r"--ransac",
            re.I,
        ),
        re.compile(
            r"findHomography",
            re.I,
        ),
        re.compile(
            r"ORB_create",
            re.I,
        ),
    ]

    hits = []

    for i, line in enumerate(
        lines,
        1,
    ):
        if any(
            p.search(line)
            for p in patterns
        ):
            hits.append(i)

    ranges = []

    for hit in hits:
        start = max(
            1,
            hit - 8,
        )
        end = min(
            len(lines),
            hit + 12,
        )

        if (
            ranges
            and start
            <= ranges[-1][1] + 2
        ):
            ranges[-1] = (
                ranges[-1][0],
                max(
                    ranges[-1][1],
                    end,
                ),
            )
        else:
            ranges.append(
                (
                    start,
                    end,
                )
            )

    contexts = []

    for start, end in ranges:
        contexts.append(
            {
                "start":
                    start,

                "end":
                    end,

                "lines":
                    [
                        {
                            "line":
                                i,
                            "text":
                                lines[
                                    i - 1
                                ],
                        }
                        for i
                        in range(
                            start,
                            end + 1,
                        )
                    ],
            }
        )

    return contexts


def scan_metadata(
    root: Path,
):
    results = []

    extensions = {
        ".json",
        ".yaml",
        ".yml",
        ".txt",
        ".md",
        ".csv",
    }

    for path in root.rglob("*"):

        if (
            not path.is_file()
            or path.suffix.lower()
            not in extensions
        ):
            continue

        # Skip enormous candidate tables.
        try:
            if (
                path.stat().st_size
                > 5_000_000
            ):
                continue
        except OSError:
            continue

        try:
            text = path.read_text(
                errors="replace"
            )
        except Exception:
            continue

        low = text.lower()

        matched = [
            key
            for key in KEYWORDS
            if key.lower()
            in low
        ]

        if not matched:
            continue

        snippets = []

        lines = text.splitlines()

        for i, line in enumerate(
            lines,
            1,
        ):

            if any(
                key.lower()
                in line.lower()
                for key in matched
            ):
                snippets.append(
                    {
                        "line":
                            i,

                        "text":
                            line[:500],
                    }
                )

                if len(
                    snippets
                ) >= 30:
                    break

        results.append(
            {
                "path":
                    str(
                        path
                    ),

                "matched_keywords":
                    matched,

                "snippets":
                    snippets,
            }
        )

    return results


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

    out_dir = (
        research
        / "postfreeze_eval"
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    # ========================================================
    # Candidate artifact
    # ========================================================

    candidate_csv = (
        run
        / "reports/"
          "s8_12e1_top20_verifier_reranker/"
          "512_s256_orb_hybrid_top20_img518/"
          "s8_12e1_all_candidate_verifier_scores.csv"
    )

    source = (
        repo
        / "scripts/"
          "villoc/"
          "s8_12e1_top20_verifier_reranker.py"
    )


    if not candidate_csv.exists():
        raise RuntimeError(
            f"Missing candidate CSV: {candidate_csv}"
        )

    if not source.exists():
        raise RuntimeError(
            f"Missing verifier source: {source}"
        )


    df = pd.read_csv(
        candidate_csv
    )


    relevant = relevant_columns(
        df.columns
    )


    # ========================================================
    # Representative q1 images
    # ========================================================

    q1 = (
        df[
            pd.to_numeric(
                df[
                    "query_id"
                ],
                errors="coerce",
            )
            == 1
        ]
        .sort_values(
            "hybrid_rank"
        )
        .iloc[0]
    )


    image_columns = [
        "image_path",
        "query_image_resolved",
        "tile_path",
        "tile_image_resolved",
    ]


    image_info = {}


    for column in image_columns:

        if column not in q1.index:
            continue

        value = q1[
            column
        ]

        if pd.isna(value):
            continue

        path = Path(
            str(value)
        )

        if not path.is_absolute():
            path = (
                repo
                / path
            )

        exists = path.exists()

        shape = None

        if exists:
            img = cv2.imread(
                str(path),
                cv2.IMREAD_COLOR,
            )

            if img is not None:
                shape = [
                    int(x)
                    for x in img.shape
                ]

        image_info[
            column
        ] = {
            "path":
                str(path),

            "exists":
                bool(
                    exists
                ),

            "raw_shape_hwc":
                shape,
        }


    # ========================================================
    # Source + run metadata
    # ========================================================

    contexts = source_contexts(
        source
    )


    reranker_root = (
        candidate_csv.parent
    )


    metadata_hits = scan_metadata(
        reranker_root
    )


    # ========================================================
    # Coordinate contract confirmed from source
    # ========================================================

    direction_contract = {
        "source_points":
            "query-image keypoints",

        "destination_points":
            "satellite-tile keypoints",

        "opencv_call":
            "cv2.findHomography(q_pts, s_pts, ...)",

        "mapping":
            "query pixel -> satellite-tile pixel",

        "projected_query_center_interpretation":
            (
                "ground location corresponding to "
                "query image centre, subject to "
                "homography validity"
            ),

        "tile_pixel_to_epsg3346_candidate_formula":
            {
                "easting":
                    (
                        "left_easting + "
                        "u/(W-1) * "
                        "(right_easting-left_easting)"
                    ),

                "northing":
                    (
                        "top_northing - "
                        "v/(H-1) * "
                        "(top_northing-bottom_northing)"
                    ),
            },

        "formula_status":
            (
                "candidate formula only until preprocessing/"
                "crop contract is locked"
            ),
    }


    report = {
        "stage":
            "R4.10_ORB_PROJECTION_CONTRACT_LOCK",

        "contract": {
            "gt_used":
                False,

            "reference_used":
                False,

            "orb_rerun":
                False,

            "production_code_modified":
                False,
        },

        "candidate_csv":
            str(
                candidate_csv
            ),

        "candidate_relevant_columns":
            relevant,

        "representative_raw_images":
            image_info,

        "homography_direction":
            direction_contract,

        "source_context":
            contexts,

        "run_metadata_hits":
            metadata_hits,
    }


    report_path = (
        out_dir
        / "r4_10_orb_projection_contract_lock.json"
    )


    report_path.write_text(
        json.dumps(
            report,
            indent=2,
        )
    )


    # ========================================================
    # Print
    # ========================================================

    print()
    print("=" * 112)
    print(
        "R4.10 — EXACT ORB PROJECTION CONTRACT LOCK"
    )
    print("=" * 112)

    print()
    print(
        "Relevant candidate-table columns:"
    )

    for col in relevant:
        print(
            " ",
            col,
        )


    print()
    print(
        "Representative raw image shapes:"
    )

    for key, value in (
        image_info.items()
    ):
        print(
            f"  {key}:"
        )

        print(
            "    exists:",
            value[
                "exists"
            ],
        )

        print(
            "    shape:",
            value[
                "raw_shape_hwc"
            ],
        )

        print(
            "    path:",
            value[
                "path"
            ],
        )


    print()
    print("=" * 112)
    print("CONFIRMED HOMOGRAPHY DIRECTION")
    print("=" * 112)

    print(
        "query keypoints -> satellite-tile keypoints"
    )

    print(
        "H maps query pixel -> satellite-tile pixel"
    )


    print()
    print("=" * 112)
    print("VERIFIER SOURCE CONTEXT")
    print("=" * 112)

    for context in contexts:

        print()
        print(
            f"[lines "
            f"{context['start']}"
            f"-"
            f"{context['end']}]"
        )

        for item in context[
            "lines"
        ]:

            print(
                f"{item['line']:5d}: "
                f"{item['text']}"
            )


    print()
    print("=" * 112)
    print("RUN METADATA / PARAMETER HITS")
    print("=" * 112)

    if not metadata_hits:

        print(
            "No separate reranker metadata file "
            "containing parameter strings found."
        )

    else:

        for hit in metadata_hits:

            print()
            print(
                "---",
                hit[
                    "path"
                ],
                "---",
            )

            print(
                "keywords:",
                ", ".join(
                    hit[
                        "matched_keywords"
                    ]
                ),
            )

            for snippet in hit[
                "snippets"
            ]:

                print(
                    f"{snippet['line']:5d}: "
                    f"{snippet['text']}"
                )


    print()
    print("=" * 112)
    print("R4.10 OUTPUT")
    print("=" * 112)

    print(
        "report:",
        report_path,
    )

    print()

    print(
        "STATUS: "
        "PASS_R4_10_ORB_PROJECTION_CONTRACT_AUDIT"
    )


if __name__ == "__main__":
    main()
