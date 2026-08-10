#!/usr/bin/env python3
"""
R4.14 — Top-M sub-tile availability audit.

POST-FREEZE diagnostic.

R4.13 established that q37/q38 degradation comes primarily from
the newest sub-tile observation, while the correct geographic tile
is already present in Top-4.

Question
--------
Does a more accurate continuous ORB observation exist deeper in the
already-computed Top-20 candidate set?

For q36/q37/q38 and K in {4,5,10,20}:
  * choose the minimum-GT-error projected candidate independently
    for each of the four evidence queries;
  * fit the resulting similarity transform;
  * compare transform disagreement to q1..q38 GT;
  * report candidate retrieval ranks.

This is explicitly oracle/post-freeze diagnosis.
No online policy is created and no frozen result is modified.

Command:

SCRIPT=scripts/villoc/research/minimum_confident_bootstrap/diagnostics/r4_14_topm_subtile_availability_audit.py
RUN=outputs/demo_runs/traj01_blind_regression_001
R3=outputs/research_runs/minimum_confident_bootstrap/traj01_blind_r3_001

python "$SCRIPT" \
  --run-root "$RUN" \
  --research-root "$R3" \
  2>&1 | tee \
  "$R3/postfreeze_eval/r4_14_topm_subtile_availability_audit.log"

"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


TARGET_UPDATES = [36, 37, 38]
TOP_KS = [4, 5, 10, 20]


def parse_int_ids(value):

    if pd.isna(value):
        return []

    text = (
        str(value)
        .strip()
        .replace("[", "")
        .replace("]", "")
        .replace("(", "")
        .replace(")", "")
    )

    if not text:
        return []

    return [
        int(x.strip())
        for x in text.split(",")
        if x.strip()
    ]


def fit_similarity(
    visual_xy,
    map_xy,
):

    visual_xy = np.asarray(
        visual_xy,
        dtype=float,
    )

    map_xy = np.asarray(
        map_xy,
        dtype=float,
    )

    z = (
        visual_xy[:, 0]
        + 1j * visual_xy[:, 1]
    )

    w = (
        map_xy[:, 0]
        + 1j * map_xy[:, 1]
    )

    z0 = z - z.mean()
    w0 = w - w.mean()

    denom = float(
        np.sum(
            np.abs(z0) ** 2
        )
    )

    if denom <= 1e-12:
        raise RuntimeError(
            "Degenerate similarity fit."
        )

    a = (
        np.sum(
            w0 * np.conj(z0)
        )
        / denom
    )

    b = (
        w.mean()
        - a * z.mean()
    )

    return {
        "a_real":
            float(a.real),

        "a_imag":
            float(a.imag),

        "b_real":
            float(b.real),

        "b_imag":
            float(b.imag),

        "scale":
            float(abs(a)),

        "rotation_deg":
            float(
                np.degrees(
                    np.angle(a)
                )
            ),
    }


def apply_similarity(
    xy,
    model,
):

    xy = np.asarray(
        xy,
        dtype=float,
    )

    z = (
        xy[:, 0]
        + 1j * xy[:, 1]
    )

    a = complex(
        model["a_real"],
        model["a_imag"],
    )

    b = complex(
        model["b_real"],
        model["b_imag"],
    )

    w = (
        a * z
        + b
    )

    return np.column_stack(
        [
            w.real,
            w.imag,
        ]
    )


def transform_disagreement(
    model,
    gt_model,
    probes,
):

    p = apply_similarity(
        probes,
        model,
    )

    g = apply_similarity(
        probes,
        gt_model,
    )

    return float(
        np.linalg.norm(
            p - g,
            axis=1,
        ).max()
    )


def main():

    parser = argparse.ArgumentParser()

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

    run = args.run_root.resolve()
    research = args.research_root.resolve()

    out = (
        research
        / "postfreeze_eval"
    )


    # ========================================================
    # Inputs
    # ========================================================

    pair_path = (
        out
        / "r4_11_postfreeze_subtile_projection_eval.csv"
    )

    update_path = (
        out
        / "r4_12_blind_subtile_update_summary.csv"
    )

    relative_path = (
        run
        / "metadata/"
          "s8_xfeat_relative_frontend/"
          "s8r4_xfeat_relative_trajectory_blind_raw.csv"
    )


    pair = pd.read_csv(
        pair_path
    )

    updates = pd.read_csv(
        update_path
    )

    relative = pd.read_csv(
        relative_path
    )


    pair[
        "query_id"
    ] = pd.to_numeric(
        pair["query_id"],
        errors="raise",
    ).astype(int)


    updates[
        "update_query_id"
    ] = pd.to_numeric(
        updates["update_query_id"],
        errors="raise",
    ).astype(int)


    relative[
        "query_id"
    ] = pd.to_numeric(
        relative["token0_id"],
        errors="raise",
    ).astype(int)


    relative_by_q = (
        relative
        .set_index(
            "query_id"
        )
    )


    # ========================================================
    # GT q1..q38 transform
    # ========================================================

    gt = (
        pair[
            [
                "query_id",
                "gt_easting",
                "gt_northing",
            ]
        ]
        .drop_duplicates(
            "query_id"
        )
    )


    prefix = (
        relative[
            relative["query_id"]
            <= 38
        ]
        .merge(
            gt,
            on="query_id",
            validate="one_to_one",
        )
        .sort_values(
            "query_id"
        )
    )


    gt_model = fit_similarity(
        prefix[
            [
                "visual_x_px",
                "visual_y_px",
            ]
        ].to_numpy(float),

        prefix[
            [
                "gt_easting",
                "gt_northing",
            ]
        ].to_numpy(float),
    )


    common_probes = (
        prefix[
            [
                "visual_x_px",
                "visual_y_px",
            ]
        ]
        .iloc[
            [0, -1]
        ]
        .to_numpy(float)
    )


    result_rows = []
    candidate_rows = []


    # ========================================================
    # Oracle Top-K availability
    # ========================================================

    for update_q in TARGET_UPDATES:

        update_row = (
            updates[
                updates[
                    "update_query_id"
                ]
                == update_q
            ]
            .iloc[0]
        )


        evidence_q = parse_int_ids(
            update_row[
                "evidence_query_ids"
            ]
        )


        if len(evidence_q) != 4:

            raise RuntimeError(
                f"q{update_q}: "
                f"expected four evidence queries, "
                f"got {evidence_q}"
            )


        visual_xy = (
            relative_by_q
            .loc[
                evidence_q,
                [
                    "visual_x_px",
                    "visual_y_px",
                ],
            ]
            .to_numpy(float)
        )


        for top_k in TOP_KS:

            selected_xy = []
            selected_tiles = []
            selected_hybrid_ranks = []
            selected_dino_ranks = []
            selected_errors = []


            for evidence_index, q in enumerate(
                evidence_q
            ):

                q_candidates = (
                    pair[
                        (
                            pair[
                                "query_id"
                            ]
                            == q
                        )
                        &
                        (
                            pair[
                                "hybrid_rank"
                            ]
                            <= top_k
                        )
                        &
                        np.isfinite(
                            pair[
                                "projected_error_m"
                            ]
                        )
                    ]
                    .copy()
                    .sort_values(
                        "projected_error_m"
                    )
                )


                if len(q_candidates) == 0:

                    raise RuntimeError(
                        f"q{q}, Top-{top_k}: "
                        "no finite projected candidate."
                    )


                best = (
                    q_candidates.iloc[0]
                )


                selected_xy.append(
                    [
                        float(
                            best[
                                "projected_easting"
                            ]
                        ),
                        float(
                            best[
                                "projected_northing"
                            ]
                        ),
                    ]
                )


                selected_tiles.append(
                    str(
                        best[
                            "tile_id"
                        ]
                    )
                )


                selected_hybrid_ranks.append(
                    int(
                        best[
                            "hybrid_rank"
                        ]
                    )
                )


                selected_dino_ranks.append(
                    int(
                        best[
                            "dino_rank"
                        ]
                    )
                )


                selected_errors.append(
                    float(
                        best[
                            "projected_error_m"
                        ]
                    )
                )


                candidate_rows.append(
                    {
                        "update_query_id":
                            update_q,

                        "top_k":
                            top_k,

                        "evidence_index":
                            evidence_index,

                        "evidence_query_id":
                            q,

                        "selected_tile_id":
                            str(
                                best[
                                    "tile_id"
                                ]
                            ),

                        "selected_hybrid_rank":
                            int(
                                best[
                                    "hybrid_rank"
                                ]
                            ),

                        "selected_dino_rank":
                            int(
                                best[
                                    "dino_rank"
                                ]
                            ),

                        "selected_projected_error_m":
                            float(
                                best[
                                    "projected_error_m"
                                ]
                            ),

                        "selected_gt_inside_tile":
                            bool(
                                best[
                                    "gt_inside_tile"
                                ]
                            ),

                        "selected_projected_inside_tile":
                            bool(
                                best[
                                    "projected_inside_tile"
                                ]
                            ),

                        "selected_inliers":
                            int(
                                best[
                                    "recomputed_inliers"
                                ]
                            ),

                        "selected_inlier_ratio":
                            float(
                                best[
                                    "recomputed_inlier_ratio"
                                ]
                            ),

                        "selected_query_coverage":
                            float(
                                best[
                                    "recomputed_query_inlier_coverage"
                                ]
                            ),

                        "selected_reprojection_rmse_px":
                            float(
                                best[
                                    "inlier_reprojection_rmse_px"
                                ]
                            ),
                    }
                )


            oracle_model = fit_similarity(
                visual_xy,
                np.asarray(
                    selected_xy,
                    dtype=float,
                ),
            )


            disagreement = (
                transform_disagreement(
                    oracle_model,
                    gt_model,
                    common_probes,
                )
            )


            result_rows.append(
                {
                    "update_query_id":
                        update_q,

                    "evidence_query_ids":
                        ",".join(
                            map(
                                str,
                                evidence_q,
                            )
                        ),

                    "top_k":
                        top_k,

                    "oracle_tile_ids":
                        ",".join(
                            selected_tiles
                        ),

                    "oracle_hybrid_ranks":
                        ",".join(
                            map(
                                str,
                                selected_hybrid_ranks,
                            )
                        ),

                    "oracle_dino_ranks":
                        ",".join(
                            map(
                                str,
                                selected_dino_ranks,
                            )
                        ),

                    "oracle_projected_errors_m":
                        ",".join(
                            f"{x:.3f}"
                            for x
                            in selected_errors
                        ),

                    "max_single_measurement_error_m":
                        float(
                            max(
                                selected_errors
                            )
                        ),

                    "median_single_measurement_error_m":
                        float(
                            np.median(
                                selected_errors
                            )
                        ),

                    "oracle_scale":
                        float(
                            oracle_model[
                                "scale"
                            ]
                        ),

                    "oracle_rotation_deg":
                        float(
                            oracle_model[
                                "rotation_deg"
                            ]
                        ),

                    "oracle_transform_disagreement_m":
                        float(
                            disagreement
                        ),
                }
            )


    results = pd.DataFrame(
        result_rows
    )

    candidate_detail = pd.DataFrame(
        candidate_rows
    )


    # ========================================================
    # Current-frame Top-20 detail
    # ========================================================

    current_rows = []


    for update_q in TARGET_UPDATES:

        current = (
            pair[
                pair[
                    "query_id"
                ]
                == update_q
            ]
            .copy()
            .sort_values(
                "projected_error_m"
            )
            .reset_index(
                drop=True
            )
        )


        current[
            "postfreeze_projected_error_rank"
        ] = np.arange(
            1,
            len(current) + 1,
        )


        for _, row in (
            current.head(20)
            .iterrows()
        ):

            current_rows.append(
                {
                    "query_id":
                        update_q,

                    "postfreeze_projected_error_rank":
                        int(
                            row[
                                "postfreeze_projected_error_rank"
                            ]
                        ),

                    "tile_id":
                        str(
                            row[
                                "tile_id"
                            ]
                        ),

                    "hybrid_rank":
                        int(
                            row[
                                "hybrid_rank"
                            ]
                        ),

                    "dino_rank":
                        int(
                            row[
                                "dino_rank"
                            ]
                        ),

                    "projected_error_m":
                        float(
                            row[
                                "projected_error_m"
                            ]
                        ),

                    "tile_center_error_m":
                        float(
                            row[
                                "tile_center_error_m"
                            ]
                        ),

                    "gt_inside_tile":
                        bool(
                            row[
                                "gt_inside_tile"
                            ]
                        ),

                    "projected_inside_tile":
                        bool(
                            row[
                                "projected_inside_tile"
                            ]
                        ),

                    "inliers":
                        int(
                            row[
                                "recomputed_inliers"
                            ]
                        ),

                    "inlier_ratio":
                        float(
                            row[
                                "recomputed_inlier_ratio"
                            ]
                        ),

                    "query_coverage":
                        float(
                            row[
                                "recomputed_query_inlier_coverage"
                            ]
                        ),

                    "reprojection_rmse_px":
                        float(
                            row[
                                "inlier_reprojection_rmse_px"
                            ]
                        ),
                }
            )


    current_detail = pd.DataFrame(
        current_rows
    )


    # ========================================================
    # Save
    # ========================================================

    results_path = (
        out
        / "r4_14_topm_subtile_oracle_summary.csv"
    )

    candidate_path = (
        out
        / "r4_14_topm_selected_candidate_detail.csv"
    )

    current_path = (
        out
        / "r4_14_current_frame_top20_projection_detail.csv"
    )

    report_path = (
        out
        / "r4_14_topm_subtile_availability_audit.json"
    )


    results.to_csv(
        results_path,
        index=False,
    )

    candidate_detail.to_csv(
        candidate_path,
        index=False,
    )

    current_detail.to_csv(
        current_path,
        index=False,
    )


    report = {
        "stage":
            "R4.14_TOPM_SUBTILE_AVAILABILITY_AUDIT",

        "status":
            "PASS_R4_14_TOPM_SUBTILE_AVAILABILITY_AUDIT_EXECUTION",

        "target_updates":
            TARGET_UPDATES,

        "top_ks":
            TOP_KS,

        "gt_prefix_transform": {
            "scale":
                gt_model[
                    "scale"
                ],

            "rotation_deg":
                gt_model[
                    "rotation_deg"
                ],
        },

        "contract": {
            "postfreeze_diagnostic":
                True,

            "gt_used":
                True,

            "candidate_selection_is_oracle_eval_only":
                True,

            "r3_modified":
                False,

            "r4_11_modified":
                False,

            "r4_12_modified":
                False,
        },
    }


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
    print("=" * 118)
    print(
        "R4.14 — TOP-M SUB-TILE AVAILABILITY AUDIT"
    )
    print("=" * 118)

    print(
        "GT prefix scale:",
        f"{gt_model['scale']:.6f}",
    )

    print(
        "GT prefix rotation:",
        f"{gt_model['rotation_deg']:.3f} deg",
    )


    print()
    print("=" * 118)
    print("TOP-M ORACLE TRANSFORM SUMMARY")
    print("=" * 118)

    print(
        results.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.3f}",
        )
    )


    for query_id in TARGET_UPDATES:

        print()
        print("=" * 118)
        print(
            f"q{query_id} CURRENT-FRAME "
            "TOP-20 SORTED BY PROJECTED ERROR"
        )
        print("=" * 118)

        sub = (
            current_detail[
                current_detail[
                    "query_id"
                ]
                == query_id
            ]
            .sort_values(
                "postfreeze_projected_error_rank"
            )
        )

        print(
            sub.to_string(
                index=False,
                float_format=lambda x:
                    f"{x:.3f}",
            )
        )


    print()
    print("=" * 118)
    print("R4.14 OUTPUT")
    print("=" * 118)

    print(
        "Top-M summary:",
        results_path,
    )

    print(
        "selected-candidate detail:",
        candidate_path,
    )

    print(
        "current-frame Top20 detail:",
        current_path,
    )

    print(
        "report:",
        report_path,
    )

    print()

    print(
        "STATUS: "
        "PASS_R4_14_TOPM_SUBTILE_AVAILABILITY_AUDIT_EXECUTION"
    )


if __name__ == "__main__":
    main()
