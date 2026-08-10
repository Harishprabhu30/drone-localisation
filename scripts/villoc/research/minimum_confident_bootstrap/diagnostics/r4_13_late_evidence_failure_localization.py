#!/usr/bin/env python3
"""
R4.13 — late-evidence failure localization.

POST-FREEZE diagnostic.

R4.12 showed:
  q24..q36: sub-tile bootstrap transform is near GT
  q37:       begins degrading
  q38:       best admissible hypothesis is already ~23 m from GT

Question
--------
Which evidence query/candidate causes the near-GT family to disappear?

For control/update queries q36, q37, q38:
  * inspect every Top-4 candidate for every selected evidence frame;
  * compare tile-center and sub-tile projected GT errors;
  * identify the blind-leader candidate choices;
  * identify the GT-best admissible hypothesis choices;
  * independently select the lowest projected-error candidate per
    evidence query and fit its transform.

GT is explicitly allowed because this is a post-freeze failure diagnostic.
Nothing here is used online or fed back into frozen R4.12.

Command:

SCRIPT=scripts/villoc/research/minimum_confident_bootstrap/diagnostics/r4_13_late_evidence_failure_localization.py
RUN=outputs/demo_runs/traj01_blind_regression_001
R3=outputs/research_runs/minimum_confident_bootstrap/traj01_blind_r3_001

python "$SCRIPT" \
  --run-root "$RUN" \
  --research-root "$R3" \
  2>&1 | tee \
  "$R3/postfreeze_eval/r4_13_late_evidence_failure_localization.log"
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


TARGET_UPDATES = [
    36,
    37,
    38,
]


# ============================================================
# Helpers
# ============================================================

def parse_ids(value):
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
        x.strip()
        for x in text.split(",")
        if x.strip()
    ]


def parse_int_ids(value):
    return [
        int(x)
        for x in parse_ids(
            value
        )
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
        model[
            "a_real"
        ],
        model[
            "a_imag"
        ],
    )

    b = complex(
        model[
            "b_real"
        ],
        model[
            "b_imag"
        ],
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
    pred = apply_similarity(
        probes,
        model,
    )

    gt = apply_similarity(
        probes,
        gt_model,
    )

    return float(
        np.linalg.norm(
            pred - gt,
            axis=1,
        ).max()
    )


# ============================================================
# Main
# ============================================================

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

    run = (
        args.run_root
        .resolve()
    )

    research = (
        args.research_root
        .resolve()
    )

    out = (
        research
        / "postfreeze_eval"
    )


    # ========================================================
    # Inputs
    # ========================================================

    candidate_path = (
        research
        / "candidate_evidence.csv"
    )

    r411_eval_path = (
        out
        / "r4_11_postfreeze_subtile_projection_eval.csv"
    )

    r412_hyp_path = (
        out
        / "r4_12_gt_annotated_subtile_hypotheses.csv"
    )

    r412_updates_path = (
        out
        / "r4_12_blind_subtile_update_summary.csv"
    )

    r412_eval_path = (
        out
        / "r4_12_postfreeze_subtile_update_evaluation.csv"
    )

    relative_path = (
        run
        / "metadata/"
          "s8_xfeat_relative_frontend/"
          "s8r4_xfeat_relative_trajectory_blind_raw.csv"
    )


    candidates = pd.read_csv(
        candidate_path
    )

    pair_eval = pd.read_csv(
        r411_eval_path
    )

    hypotheses = pd.read_csv(
        r412_hyp_path
    )

    update_summary = pd.read_csv(
        r412_updates_path
    )

    update_eval = pd.read_csv(
        r412_eval_path
    )

    relative = pd.read_csv(
        relative_path
    )


    for frame in [
        candidates,
        pair_eval,
    ]:

        frame[
            "query_id"
        ] = pd.to_numeric(
            frame[
                "query_id"
            ],
            errors="raise",
        ).astype(int)


    hypotheses[
        "update_query_id"
    ] = pd.to_numeric(
        hypotheses[
            "update_query_id"
        ],
        errors="raise",
    ).astype(int)


    update_summary[
        "update_query_id"
    ] = pd.to_numeric(
        update_summary[
            "update_query_id"
        ],
        errors="raise",
    ).astype(int)


    update_eval[
        "update_query_id"
    ] = pd.to_numeric(
        update_eval[
            "update_query_id"
        ],
        errors="raise",
    ).astype(int)


    relative[
        "query_id"
    ] = pd.to_numeric(
        relative[
            "token0_id"
        ],
        errors="raise",
    ).astype(int)


    relative_by_q = (
        relative
        .set_index(
            "query_id"
        )
    )


    # ========================================================
    # Build GT transform from q1..q38 using GT already attached
    # in R4.11 post-freeze pair evaluation.
    # ========================================================

    gt_positions = (
        pair_eval[
            [
                "query_id",
                "gt_easting",
                "gt_northing",
            ]
        ]
        .drop_duplicates(
            "query_id"
        )
        .sort_values(
            "query_id"
        )
    )


    gt_prefix = (
        relative[
            relative[
                "query_id"
            ]
            <= 38
        ]
        .merge(
            gt_positions,
            on="query_id",
            validate="one_to_one",
        )
        .sort_values(
            "query_id"
        )
    )


    gt_model = fit_similarity(
        gt_prefix[
            [
                "visual_x_px",
                "visual_y_px",
            ]
        ].to_numpy(float),

        gt_prefix[
            [
                "gt_easting",
                "gt_northing",
            ]
        ].to_numpy(float),
    )


    common_probes = (
        gt_prefix[
            [
                "visual_x_px",
                "visual_y_px",
            ]
        ]
        .iloc[
            [
                0,
                -1,
            ]
        ]
        .to_numpy(float)
    )


    # ========================================================
    # Candidate table = R3 choice rank + R4.11 GT evaluation
    # ========================================================

    pair_keep = pair_eval[
        [
            "query_id",
            "tile_id",
            "dino_rank",
            "hybrid_rank",
            "recomputed_inliers",
            "recomputed_inlier_ratio",
            "recomputed_query_inlier_coverage",
            "inlier_reprojection_rmse_px",
            "projected_inside_tile",
            "projected_easting",
            "projected_northing",
            "tile_center_error_m",
            "projected_error_m",
            "gt_inside_tile",
        ]
    ].copy()


    audit = candidates.merge(
        pair_keep,
        on=[
            "query_id",
            "tile_id",
        ],
        how="left",
        validate="one_to_one",
        suffixes=(
            "",
            "_r411",
        ),
    )


    # ========================================================
    # Analyze q36/q37/q38
    # ========================================================

    candidate_rows = []

    update_rows = []


    for update_q in TARGET_UPDATES:

        summary_row = (
            update_summary[
                update_summary[
                    "update_query_id"
                ]
                == update_q
            ]
            .iloc[0]
        )


        evidence_q = parse_int_ids(
            summary_row[
                "evidence_query_ids"
            ]
        )


        if len(
            evidence_q
        ) != 4:

            raise RuntimeError(
                f"q{update_q}: expected four "
                f"evidence queries, got {evidence_q}"
            )


        # ----------------------------------------------------
        # Blind leader = exactly the hypothesis R4.12 selected
        # from blind Pareto objectives.
        # ----------------------------------------------------

        blind_leader_id = int(
            summary_row[
                "best_blind_pareto_hypothesis_id"
            ]
        )


        blind_leader = (
            hypotheses[
                hypotheses[
                    "hypothesis_id"
                ]
                == blind_leader_id
            ]
            .iloc[0]
        )


        blind_tiles = parse_ids(
            blind_leader[
                "tile_ids"
            ]
        )


        blind_choice_ranks = parse_int_ids(
            blind_leader[
                "candidate_choice_ranks"
            ]
        )


        # ----------------------------------------------------
        # GT-best admissible R4.12 hypothesis.
        # ----------------------------------------------------

        hgroup = hypotheses[
            (
                hypotheses[
                    "update_query_id"
                ]
                == update_q
            )
            &
            (
                hypotheses[
                    "geometric_pass"
                ]
                .astype(str)
                .str.lower()
                .isin(
                    [
                        "true",
                        "1",
                    ]
                )
            )
        ].copy()


        gt_best = (
            hgroup
            .sort_values(
                "postfreeze_gt_common_disagreement_m"
            )
            .iloc[0]
        )


        gt_best_tiles = parse_ids(
            gt_best[
                "tile_ids"
            ]
        )


        gt_best_choice_ranks = parse_int_ids(
            gt_best[
                "candidate_choice_ranks"
            ]
        )


        independent_oracle_xy = []

        independent_oracle_tiles = []

        independent_oracle_errors = []


        for evidence_index, q in enumerate(
            evidence_q
        ):

            qgroup = (
                audit[
                    audit[
                        "query_id"
                    ]
                    == q
                ]
                .copy()
                .sort_values(
                    "candidate_choice_rank"
                )
            )


            if len(
                qgroup
            ) != 4:

                raise RuntimeError(
                    f"q{q}: expected four R3 candidates, "
                    f"found {len(qgroup)}."
                )


            qgroup[
                "projected_error_rank_top4"
            ] = (
                qgroup[
                    "projected_error_m"
                ]
                .rank(
                    method="min",
                    ascending=True,
                )
                .astype(int)
            )


            best_row = (
                qgroup
                .sort_values(
                    "projected_error_m"
                )
                .iloc[0]
            )


            independent_oracle_xy.append(
                [
                    float(
                        best_row[
                            "projected_easting"
                        ]
                    ),
                    float(
                        best_row[
                            "projected_northing"
                        ]
                    ),
                ]
            )


            independent_oracle_tiles.append(
                str(
                    best_row[
                        "tile_id"
                    ]
                )
            )


            independent_oracle_errors.append(
                float(
                    best_row[
                        "projected_error_m"
                    ]
                )
            )


            blind_tile = (
                blind_tiles[
                    evidence_index
                ]
            )


            gt_best_tile = (
                gt_best_tiles[
                    evidence_index
                ]
            )


            for _, row in qgroup.iterrows():

                candidate_rows.append(
                    {
                        "update_query_id":
                            update_q,

                        "evidence_index":
                            evidence_index,

                        "evidence_query_id":
                            q,

                        "tile_id":
                            str(
                                row[
                                    "tile_id"
                                ]
                            ),

                        "candidate_choice_rank":
                            int(
                                row[
                                    "candidate_choice_rank"
                                ]
                            ),

                        "dino_rank":
                            int(
                                row[
                                    "dino_rank"
                                ]
                            ),

                        "hybrid_rank":
                            int(
                                row[
                                    "hybrid_rank"
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

                        "projected_inside_tile":
                            bool(
                                row[
                                    "projected_inside_tile"
                                ]
                            ),

                        "gt_inside_tile":
                            bool(
                                row[
                                    "gt_inside_tile"
                                ]
                            ),

                        "tile_center_error_m":
                            float(
                                row[
                                    "tile_center_error_m"
                                ]
                            ),

                        "projected_error_m":
                            float(
                                row[
                                    "projected_error_m"
                                ]
                            ),

                        "projected_error_rank_top4":
                            int(
                                row[
                                    "projected_error_rank_top4"
                                ]
                            ),

                        "blind_leader_choice":
                            bool(
                                str(
                                    row[
                                        "tile_id"
                                    ]
                                )
                                == blind_tile
                                and
                                int(
                                    row[
                                        "candidate_choice_rank"
                                    ]
                                )
                                ==
                                blind_choice_ranks[
                                    evidence_index
                                ]
                            ),

                        "gt_best_admissible_choice":
                            bool(
                                str(
                                    row[
                                        "tile_id"
                                    ]
                                )
                                == gt_best_tile
                                and
                                int(
                                    row[
                                        "candidate_choice_rank"
                                    ]
                                )
                                ==
                                gt_best_choice_ranks[
                                    evidence_index
                                ]
                            ),

                        "independent_oracle_choice":
                            bool(
                                str(
                                    row[
                                        "tile_id"
                                    ]
                                )
                                ==
                                str(
                                    best_row[
                                        "tile_id"
                                    ]
                                )
                            ),
                    }
                )


        # ----------------------------------------------------
        # Independent per-frame Top4 oracle transform.
        #
        # This answers:
        # "Does the Top4 pool itself still contain a good set of
        #  continuous observations?"
        # ----------------------------------------------------

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


        oracle_model = fit_similarity(
            visual_xy,
            np.asarray(
                independent_oracle_xy,
                dtype=float,
            ),
        )


        oracle_disagreement = (
            transform_disagreement(
                oracle_model,
                gt_model,
                common_probes,
            )
        )


        eval_row = (
            update_eval[
                update_eval[
                    "update_query_id"
                ]
                == update_q
            ]
            .iloc[0]
        )


        update_rows.append(
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

                "blind_leader_hypothesis_id":
                    blind_leader_id,

                "blind_leader_gt_disagreement_m":
                    float(
                        eval_row[
                            "blind_leader_gt_disagreement_m"
                        ]
                    ),

                "blind_leader_scale":
                    float(
                        blind_leader[
                            "scale_m_per_visual_px"
                        ]
                    ),

                "blind_leader_rotation_deg":
                    float(
                        blind_leader[
                            "rotation_deg"
                        ]
                    ),

                "gt_best_admissible_hypothesis_id":
                    int(
                        gt_best[
                            "hypothesis_id"
                        ]
                    ),

                "gt_best_admissible_disagreement_m":
                    float(
                        gt_best[
                            "postfreeze_gt_common_disagreement_m"
                        ]
                    ),

                "independent_oracle_tile_ids":
                    ",".join(
                        independent_oracle_tiles
                    ),

                "independent_oracle_projected_errors_m":
                    ",".join(
                        f"{x:.3f}"
                        for x in independent_oracle_errors
                    ),

                "independent_oracle_max_single_measurement_error_m":
                    float(
                        max(
                            independent_oracle_errors
                        )
                    ),

                "independent_oracle_scale":
                    float(
                        oracle_model[
                            "scale"
                        ]
                    ),

                "independent_oracle_rotation_deg":
                    float(
                        oracle_model[
                            "rotation_deg"
                        ]
                    ),

                "independent_oracle_transform_disagreement_m":
                    float(
                        oracle_disagreement
                    ),
            }
        )


    candidate_audit = pd.DataFrame(
        candidate_rows
    )


    update_audit = pd.DataFrame(
        update_rows
    )


    # ========================================================
    # Save
    # ========================================================

    candidate_out = (
        out
        / "r4_13_late_evidence_candidate_audit.csv"
    )

    update_out = (
        out
        / "r4_13_late_evidence_update_decomposition.csv"
    )

    report_out = (
        out
        / "r4_13_late_evidence_failure_localization.json"
    )


    candidate_audit.to_csv(
        candidate_out,
        index=False,
    )


    update_audit.to_csv(
        update_out,
        index=False,
    )


    report = {
        "stage":
            "R4.13_LATE_EVIDENCE_FAILURE_LOCALIZATION",

        "status":
            "PASS_R4_13_LATE_EVIDENCE_FAILURE_LOCALIZATION_EXECUTION",

        "target_updates":
            TARGET_UPDATES,

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

            "r3_modified":
                False,

            "r4_11_modified":
                False,

            "r4_12_modified":
                False,
        },

        "outputs": {
            "candidate_audit":
                str(
                    candidate_out
                ),

            "update_decomposition":
                str(
                    update_out
                ),
        },
    }


    report_out.write_text(
        json.dumps(
            report,
            indent=2,
        )
    )


    # ========================================================
    # Print
    # ========================================================

    print()
    print("=" * 116)
    print(
        "R4.13 — LATE-EVIDENCE FAILURE LOCALIZATION"
    )
    print("=" * 116)

    print(
        "GT prefix scale:",
        f"{gt_model['scale']:.6f}",
    )

    print(
        "GT prefix rotation:",
        f"{gt_model['rotation_deg']:.3f} deg",
    )


    print()
    print("=" * 116)
    print("UPDATE-LEVEL DECOMPOSITION")
    print("=" * 116)

    print(
        update_audit.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.3f}",
        )
    )


    for update_q in TARGET_UPDATES:

        print()
        print("=" * 116)
        print(
            f"q{update_q} EVIDENCE-CANDIDATE DETAILS"
        )
        print("=" * 116)


        sub = candidate_audit[
            candidate_audit[
                "update_query_id"
            ]
            == update_q
        ]


        show = [
            "evidence_query_id",
            "tile_id",
            "candidate_choice_rank",
            "dino_rank",
            "hybrid_rank",
            "projected_error_m",
            "tile_center_error_m",
            "projected_inside_tile",
            "gt_inside_tile",
            "inliers",
            "inlier_ratio",
            "query_coverage",
            "reprojection_rmse_px",
            "blind_leader_choice",
            "gt_best_admissible_choice",
            "independent_oracle_choice",
        ]


        print(
            sub[
                show
            ].to_string(
                index=False,
                float_format=lambda x:
                    f"{x:.3f}",
            )
        )


        print()
        print(
            "Best projected candidate per evidence query:"
        )


        best_per_query = (
            sub
            .sort_values(
                [
                    "evidence_query_id",
                    "projected_error_m",
                ]
            )
            .groupby(
                "evidence_query_id",
                as_index=False,
            )
            .first()
        )


        print(
            best_per_query[
                [
                    "evidence_query_id",
                    "tile_id",
                    "candidate_choice_rank",
                    "dino_rank",
                    "hybrid_rank",
                    "projected_error_m",
                    "gt_inside_tile",
                ]
            ].to_string(
                index=False,
                float_format=lambda x:
                    f"{x:.3f}",
            )
        )


    print()
    print("=" * 116)
    print("R4.13 OUTPUT")
    print("=" * 116)

    print(
        "candidate audit:",
        candidate_out,
    )

    print(
        "update decomposition:",
        update_out,
    )

    print(
        "report:",
        report_out,
    )

    print()

    print(
        "STATUS: "
        "PASS_R4_13_LATE_EVIDENCE_FAILURE_LOCALIZATION_EXECUTION"
    )


if __name__ == "__main__":
    main()
