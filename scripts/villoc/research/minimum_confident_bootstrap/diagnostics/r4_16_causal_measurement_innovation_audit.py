#!/usr/bin/env python3
"""
R4.16 — causal absolute-measurement innovation audit.

Motivation
----------
R4.12:
    continuous ORB sub-tile observations recover an excellent transform
    through much of the startup prefix.

R4.13/R4.14:
    q37/q38 degradation comes from the newest absolute observation,
    not from Top-M candidate availability.

R4.15:
    local homography Jacobian conditioning is a strong blind reliability
    signal, while reprojection RMSE and convex-hull containment are not
    sufficient.

Question
--------
Can an already-established BLIND transform predict the next map position
well enough to identify an inconsistent new absolute observation BEFORE
that new observation contaminates the transform?

PHASE A — BLIND
----------------
For each R4.12 update after the first:

    previous update's frozen blind leader transform
        +
    current XFeat relative visual position
        ->
    prior predicted EPSG:3346 position

For each current query's frozen Top-4 ORB projected candidate:

    innovation_m =
        distance(prior predicted map point,
                 candidate sub-tile projected map point)

Also retain:
    Jacobian condition
    local area scale
    inliers
    DINO rank
    hybrid rank

No threshold.
No weighted score.
No GT/reference.

Freeze + hash.

PHASE B — POST-FREEZE
------------------------
Attach GT only after Phase A is frozen.

Evaluate:
    * prior prediction geographic error
    * candidate projected geographic error
    * rank of GT-best candidate by blind innovation
    * minimum blind innovation per update
    * q36/q37/q38 decomposition
    * correlations of innovation / Jacobian condition with projection error

No online policy is created.
R3 is not modified.

Command:

SCRIPT=scripts/villoc/research/minimum_confident_bootstrap/diagnostics/r4_16_causal_measurement_innovation_audit.py
RUN=outputs/demo_runs/traj01_blind_regression_001
R3=outputs/research_runs/minimum_confident_bootstrap/traj01_blind_r3_001
python "$SCRIPT" \
  --run-root "$RUN" \
  --research-root "$R3" \
  2>&1 | tee \
  "$R3/postfreeze_eval/r4_16_causal_measurement_innovation_audit.log"

"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


MAP_SPACING_M = 51.2


# ============================================================
# Helpers
# ============================================================

def sha256(path: Path) -> str:

    h = hashlib.sha256()

    with path.open("rb") as f:

        for block in iter(
            lambda: f.read(
                1024 * 1024
            ),
            b"",
        ):

            h.update(block)

    return h.hexdigest()


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
        float(
            model[
                "a_real"
            ]
        ),
        float(
            model[
                "a_imag"
            ]
        ),
    )

    b = complex(
        float(
            model[
                "b_real"
            ]
        ),
        float(
            model[
                "b_imag"
            ]
        ),
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
    # Blind inputs
    # ========================================================

    updates_path = (
        out
        / "r4_12_blind_subtile_update_summary.csv"
    )


    hypotheses_path = (
        out
        / "r4_12_blind_subtile_hypotheses.csv"
    )


    projections_path = (
        out
        / "r4_11_blind_subtile_projection_pairs.csv"
    )


    support_path = (
        out
        / "r4_15_blind_homography_center_support.csv"
    )


    candidate_path = (
        research
        / "candidate_evidence.csv"
    )


    relative_path = (
        run
        / "metadata/"
          "s8_xfeat_relative_frontend/"
          "s8r4_xfeat_relative_trajectory_blind_raw.csv"
    )


    required_paths = [
        updates_path,
        hypotheses_path,
        projections_path,
        support_path,
        candidate_path,
        relative_path,
    ]


    missing = [
        str(p)
        for p in required_paths
        if not p.exists()
    ]


    if missing:

        raise RuntimeError(
            "Missing R4.16 input(s):\n"
            + "\n".join(
                missing
            )
        )


    updates = pd.read_csv(
        updates_path
    )


    hypotheses = pd.read_csv(
        hypotheses_path
    )


    projections = pd.read_csv(
        projections_path
    )


    support = pd.read_csv(
        support_path
    )


    candidate = pd.read_csv(
        candidate_path
    )


    relative = pd.read_csv(
        relative_path
    )


    updates[
        "update_query_id"
    ] = pd.to_numeric(
        updates[
            "update_query_id"
        ],
        errors="raise",
    ).astype(int)


    hypotheses[
        "hypothesis_id"
    ] = pd.to_numeric(
        hypotheses[
            "hypothesis_id"
        ],
        errors="raise",
    ).astype(int)


    projections[
        "query_id"
    ] = pd.to_numeric(
        projections[
            "query_id"
        ],
        errors="raise",
    ).astype(int)


    support[
        "query_id"
    ] = pd.to_numeric(
        support[
            "query_id"
        ],
        errors="raise",
    ).astype(int)


    candidate[
        "query_id"
    ] = pd.to_numeric(
        candidate[
            "query_id"
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


    updates = (
        updates
        .sort_values(
            "update_query_id"
        )
        .reset_index(
            drop=True
        )
    )


    relative_by_q = (
        relative
        .set_index(
            "query_id"
        )
    )


    # ========================================================
    # Current query Top-4 blind candidate table
    # ========================================================

    top4 = candidate[
        [
            "query_id",
            "tile_id",
            "candidate_choice_rank",
            "rank",
            "hybrid_rank",
        ]
    ].copy()


    top4 = top4.rename(
        columns={
            "rank":
                "dino_rank",
        }
    )


    projection_keep = projections[
        [
            "query_id",
            "tile_id",
            "projected_easting",
            "projected_northing",
            "projected_inside_tile",
        ]
    ].copy()


    support_keep = support[
        [
            "query_id",
            "tile_id",
            "inliers",
            "jacobian_condition",
            "jacobian_local_area_scale",
            "reprojection_rmse_px",
            "q_center_inside_inlier_hull",
            "s_projected_inside_inlier_hull",
        ]
    ].copy()


    current_candidates = (
        top4
        .merge(
            projection_keep,
            on=[
                "query_id",
                "tile_id",
            ],
            how="left",
            validate="one_to_one",
        )
        .merge(
            support_keep,
            on=[
                "query_id",
                "tile_id",
            ],
            how="left",
            validate="one_to_one",
        )
    )


    # ========================================================
    #
    # PHASE A — BLIND CAUSAL INNOVATION
    #
    # ========================================================

    rows = []

    skipped_no_prior = []


    # ========================================================
    # Causal prior selection
    #
    # Some early R4.12 updates legitimately have no blind
    # Pareto leader. Innovation cannot be computed until a
    # transform has actually been established.
    #
    # For each current update, use the MOST RECENT EARLIER
    # update that has a finite frozen blind leader.
    #
    # This remains strictly causal:
    #   no future update
    #   no GT
    #   no retrospective replacement
    # ========================================================

    leader_numeric = pd.to_numeric(
        updates[
            "best_blind_pareto_hypothesis_id"
        ],
        errors="coerce",
    )


    for update_index in range(
        len(
            updates
        )
    ):

        current_update = (
            updates.iloc[
                update_index
            ]
        )


        current_q = int(
            current_update[
                "update_query_id"
            ]
        )


        prior_mask = (
            leader_numeric.iloc[
                :update_index
            ]
            .notna()
        )


        if not bool(
            prior_mask.any()
        ):

            skipped_no_prior.append(
                current_q
            )

            continue


        prior_positions = np.flatnonzero(
            prior_mask.to_numpy()
        )


        previous_index = int(
            prior_positions[
                -1
            ]
        )


        previous_update = (
            updates.iloc[
                previous_index
            ]
        )


        previous_q = int(
            previous_update[
                "update_query_id"
            ]
        )


        previous_leader_id = int(
            float(
                previous_update[
                    "best_blind_pareto_hypothesis_id"
                ]
            )
        )


        leader_match = hypotheses[
            hypotheses[
                "hypothesis_id"
            ]
            == previous_leader_id
        ]


        if len(
            leader_match
        ) != 1:

            raise RuntimeError(
                "Could not uniquely recover "
                f"previous leader hypothesis "
                f"{previous_leader_id}."
            )


        previous_leader = (
            leader_match.iloc[
                0
            ]
        )


        current_visual = (
            relative_by_q
            .loc[
                [
                    current_q
                ],
                [
                    "visual_x_px",
                    "visual_y_px",
                ],
            ]
            .to_numpy(float)
        )


        prior_prediction = apply_similarity(
            current_visual,
            previous_leader,
        )[0]


        q_candidates = (
            current_candidates[
                current_candidates[
                    "query_id"
                ]
                == current_q
            ]
            .copy()
            .sort_values(
                "candidate_choice_rank"
            )
        )


        if len(
            q_candidates
        ) != 4:

            raise RuntimeError(
                f"q{current_q}: expected exactly "
                f"4 current candidates, found "
                f"{len(q_candidates)}."
            )


        for _, row in (
            q_candidates.iterrows()
        ):

            projected_xy = np.asarray(
                [
                    float(
                        row[
                            "projected_easting"
                        ]
                    ),
                    float(
                        row[
                            "projected_northing"
                        ]
                    ),
                ]
            )


            innovation = float(
                np.linalg.norm(
                    projected_xy
                    - prior_prediction
                )
            )


            rows.append(
                {
                    "previous_update_query_id":
                        previous_q,

                    "current_update_query_id":
                        current_q,

                    "previous_blind_leader_hypothesis_id":
                        previous_leader_id,

                    "previous_blind_leader_scale":
                        float(
                            previous_leader[
                                "scale_m_per_visual_px"
                            ]
                        ),

                    "previous_blind_leader_rotation_deg":
                        float(
                            previous_leader[
                                "rotation_deg"
                            ]
                        ),

                    "prior_predicted_easting":
                        float(
                            prior_prediction[
                                0
                            ]
                        ),

                    "prior_predicted_northing":
                        float(
                            prior_prediction[
                                1
                            ]
                        ),

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

                    "projected_easting":
                        float(
                            row[
                                "projected_easting"
                            ]
                        ),

                    "projected_northing":
                        float(
                            row[
                                "projected_northing"
                            ]
                        ),

                    "projected_inside_tile":
                        bool(
                            row[
                                "projected_inside_tile"
                            ]
                        ),

                    "innovation_m":
                        innovation,

                    "innovation_map_spacing_units":
                        float(
                            innovation
                            / MAP_SPACING_M
                        ),

                    "inliers":
                        int(
                            row[
                                "inliers"
                            ]
                        ),

                    "jacobian_condition":
                        float(
                            row[
                                "jacobian_condition"
                            ]
                        ),

                    "jacobian_local_area_scale":
                        float(
                            row[
                                "jacobian_local_area_scale"
                            ]
                        ),

                    "reprojection_rmse_px":
                        float(
                            row[
                                "reprojection_rmse_px"
                            ]
                        ),

                    "q_center_inside_inlier_hull":
                        bool(
                            row[
                                "q_center_inside_inlier_hull"
                            ]
                        ),

                    "s_projected_inside_inlier_hull":
                        bool(
                            row[
                                "s_projected_inside_inlier_hull"
                            ]
                        ),
                }
            )


    blind = pd.DataFrame(
        rows
    )


    # Blind innovation ranking within each current query.
    blind[
        "innovation_rank"
    ] = (
        blind
        .groupby(
            "current_update_query_id"
        )[
            "innovation_m"
        ]
        .rank(
            method="min",
            ascending=True,
        )
        .astype(int)
    )


    # No decision threshold; descriptive only.
    blind_summary = (
        blind
        .groupby(
            "current_update_query_id"
        )
        .agg(
            previous_update_query_id=(
                "previous_update_query_id",
                "first",
            ),

            minimum_innovation_m=(
                "innovation_m",
                "min",
            ),

            median_innovation_m=(
                "innovation_m",
                "median",
            ),

            maximum_innovation_m=(
                "innovation_m",
                "max",
            ),

            minimum_innovation_map_spacing_units=(
                "innovation_map_spacing_units",
                "min",
            ),
        )
        .reset_index()
    )


    blind_path = (
        out
        / "r4_16_blind_measurement_innovation_candidates.csv"
    )


    blind_summary_path = (
        out
        / "r4_16_blind_measurement_innovation_summary.csv"
    )


    freeze_path = (
        out
        / "r4_16_blind_measurement_innovation_freeze_manifest.json"
    )


    blind.to_csv(
        blind_path,
        index=False,
    )


    blind_summary.to_csv(
        blind_summary_path,
        index=False,
    )


    freeze = {
        "stage":
            "R4.16_BLIND_CAUSAL_MEASUREMENT_INNOVATION_FREEZE",

        "definition": {
            "prior_transform":
                (
                    "most recent earlier R4.12 update "
                    "with a finite frozen blind Pareto leader"
                ),

            "current_position":
                (
                    "current XFeat relative "
                    "visual coordinate"
                ),

            "innovation":
                (
                    "Euclidean distance between "
                    "prior predicted map position "
                    "and current frozen ORB "
                    "sub-tile projection"
                ),

            "map_spacing_m_for_reporting_only":
                MAP_SPACING_M,

            "threshold_used":
                False,

            "weighted_score_used":
                False,
        },

        "inputs": {
            "r4_12_updates_sha256":
                sha256(
                    updates_path
                ),

            "r4_12_hypotheses_sha256":
                sha256(
                    hypotheses_path
                ),

            "r4_11_projection_sha256":
                sha256(
                    projections_path
                ),

            "r4_15_support_sha256":
                sha256(
                    support_path
                ),

            "candidate_evidence_sha256":
                sha256(
                    candidate_path
                ),

            "relative_trajectory_sha256":
                sha256(
                    relative_path
                ),
        },

        "counts": {
            "evaluated_updates":
                int(
                    blind[
                        "current_update_query_id"
                    ].nunique()
                ),

            "candidate_rows":
                int(
                    len(
                        blind
                    )
                ),

            "startup_updates_without_prior":
                int(
                    len(
                        skipped_no_prior
                    )
                ),

            "startup_update_query_ids_without_prior":
                [
                    int(q)
                    for q in skipped_no_prior
                ],
        },

        "blind_contract": {
            "gt_used":
                False,

            "reference_used":
                False,

            "oracle_used":
                False,

            "accept_reject_threshold_created":
                False,

            "r3_modified":
                False,
        },

        "outputs": {
            "candidate_csv":
                str(
                    blind_path
                ),

            "candidate_sha256":
                sha256(
                    blind_path
                ),

            "summary_csv":
                str(
                    blind_summary_path
                ),

            "summary_sha256":
                sha256(
                    blind_summary_path
                ),
        },
    }


    freeze_path.write_text(
        json.dumps(
            freeze,
            indent=2,
        )
    )


    freeze_sha = sha256(
        freeze_path
    )


    print()
    print("=" * 118)
    print(
        "R4.16 PHASE A — "
        "BLIND CAUSAL MEASUREMENT INNOVATION FROZEN"
    )
    print("=" * 118)


    print(
        "evaluated updates:",
        blind[
            "current_update_query_id"
        ].nunique(),
    )


    print(
        "candidate rows:",
        len(
            blind
        ),
    )


    print(
        "startup updates without causal prior:",
        skipped_no_prior,
    )


    print(
        "blind freeze SHA256:",
        freeze_sha,
    )


    print()
    print(
        "Last 10 blind innovation summaries:"
    )


    print(
        blind_summary
        .tail(10)
        .to_string(
            index=False,
            float_format=lambda x:
                f"{x:.3f}",
        )
    )


    # ========================================================
    #
    # PHASE B — FIRST GT READ
    #
    # ========================================================

    evaluation_path = (
        out
        / "r4_11_postfreeze_subtile_projection_eval.csv"
    )


    evaluation = pd.read_csv(
        evaluation_path
    )


    evaluation[
        "query_id"
    ] = pd.to_numeric(
        evaluation[
            "query_id"
        ],
        errors="raise",
    ).astype(int)


    labels = evaluation[
        [
            "query_id",
            "tile_id",
            "projected_error_m",
            "gt_easting",
            "gt_northing",
            "gt_inside_tile",
        ]
    ].copy()


    annotated = blind.merge(
        labels,
        left_on=[
            "current_update_query_id",
            "tile_id",
        ],
        right_on=[
            "query_id",
            "tile_id",
        ],
        how="left",
        validate="one_to_one",
    )


    annotated[
        "prior_prediction_error_m"
    ] = np.hypot(
        annotated[
            "prior_predicted_easting"
        ]
        -
        annotated[
            "gt_easting"
        ],

        annotated[
            "prior_predicted_northing"
        ]
        -
        annotated[
            "gt_northing"
        ],
    )


    # ========================================================
    # Per-update post-freeze decomposition
    # ========================================================

    update_rows = []


    for current_q, group in (
        annotated.groupby(
            "current_update_query_id",
            sort=True,
        )
    ):

        group = (
            group
            .copy()
            .sort_values(
                "innovation_m"
            )
        )


        innovation_best = (
            group.iloc[
                0
            ]
        )


        gt_best = (
            group
            .sort_values(
                "projected_error_m"
            )
            .iloc[
                0
            ]
        )


        hybrid_best = (
            group
            .sort_values(
                "hybrid_rank"
            )
            .iloc[
                0
            ]
        )


        gt_best_innovation_rank = int(
            gt_best[
                "innovation_rank"
            ]
        )


        update_rows.append(
            {
                "current_update_query_id":
                    int(
                        current_q
                    ),

                "previous_update_query_id":
                    int(
                        group[
                            "previous_update_query_id"
                        ].iloc[
                            0
                        ]
                    ),

                "prior_prediction_error_m":
                    float(
                        group[
                            "prior_prediction_error_m"
                        ].iloc[
                            0
                        ]
                    ),

                "minimum_blind_innovation_m":
                    float(
                        innovation_best[
                            "innovation_m"
                        ]
                    ),

                "minimum_blind_innovation_spacing_units":
                    float(
                        innovation_best[
                            "innovation_map_spacing_units"
                        ]
                    ),

                "innovation_best_tile_id":
                    str(
                        innovation_best[
                            "tile_id"
                        ]
                    ),

                "innovation_best_projected_error_m":
                    float(
                        innovation_best[
                            "projected_error_m"
                        ]
                    ),

                "innovation_best_jacobian_condition":
                    float(
                        innovation_best[
                            "jacobian_condition"
                        ]
                    ),

                "gt_best_tile_id":
                    str(
                        gt_best[
                            "tile_id"
                        ]
                    ),

                "gt_best_projected_error_m":
                    float(
                        gt_best[
                            "projected_error_m"
                        ]
                    ),

                "gt_best_innovation_m":
                    float(
                        gt_best[
                            "innovation_m"
                        ]
                    ),

                "gt_best_innovation_rank":
                    gt_best_innovation_rank,

                "gt_best_jacobian_condition":
                    float(
                        gt_best[
                            "jacobian_condition"
                        ]
                    ),

                "hybrid_top1_tile_id":
                    str(
                        hybrid_best[
                            "tile_id"
                        ]
                    ),

                "hybrid_top1_projected_error_m":
                    float(
                        hybrid_best[
                            "projected_error_m"
                        ]
                    ),
            }
        )


    update_eval = pd.DataFrame(
        update_rows
    )


    # ========================================================
    # Correlations
    # ========================================================

    correct_region = annotated[
        annotated[
            "gt_inside_tile"
        ]
        .astype(bool)
    ].copy()


    correlation_features = [
        "innovation_m",
        "jacobian_condition",
        "inliers",
        "jacobian_local_area_scale",
        "reprojection_rmse_px",
    ]


    corr_rows = []


    for feature in correlation_features:

        subset = (
            correct_region[
                [
                    feature,
                    "projected_error_m",
                ]
            ]
            .replace(
                [
                    np.inf,
                    -np.inf,
                ],
                np.nan,
            )
            .dropna()
        )


        correlation = (
            float(
                subset[
                    feature
                ].corr(
                    subset[
                        "projected_error_m"
                    ],
                    method="spearman",
                )
            )
            if len(
                subset
            )
            >= 3
            else math.nan
        )


        corr_rows.append(
            {
                "feature":
                    feature,

                "pairs":
                    int(
                        len(
                            subset
                        )
                    ),

                "spearman_vs_projected_error":
                    correlation,

                "abs_spearman":
                    (
                        abs(
                            correlation
                        )
                        if np.isfinite(
                            correlation
                        )
                        else math.nan
                    ),
            }
        )


    correlations = (
        pd.DataFrame(
            corr_rows
        )
        .sort_values(
            "abs_spearman",
            ascending=False,
        )
    )


    # ========================================================
    # Outputs
    # ========================================================

    annotated_path = (
        out
        / "r4_16_gt_annotated_measurement_innovation.csv"
    )


    update_eval_path = (
        out
        / "r4_16_postfreeze_measurement_innovation_updates.csv"
    )


    correlation_path = (
        out
        / "r4_16_measurement_innovation_feature_audit.csv"
    )


    report_path = (
        out
        / "r4_16_causal_measurement_innovation_audit.json"
    )


    annotated.to_csv(
        annotated_path,
        index=False,
    )


    update_eval.to_csv(
        update_eval_path,
        index=False,
    )


    correlations.to_csv(
        correlation_path,
        index=False,
    )


    report = {
        "stage":
            "R4.16_CAUSAL_MEASUREMENT_INNOVATION_AUDIT",

        "status":
            "PASS_R4_16_CAUSAL_MEASUREMENT_INNOVATION_AUDIT_EXECUTION",

        "blind_freeze_manifest_sha256":
            freeze_sha,

        "counts": {
            "evaluated_updates":
                int(
                    len(
                        update_eval
                    )
                ),

            "correct_region_candidate_pairs":
                int(
                    len(
                        correct_region
                    )
                ),
        },

        "contract": {
            "phase_a_used_gt":
                False,

            "gt_loaded_after_blind_freeze":
                True,

            "threshold_selected":
                False,

            "weighted_score_created":
                False,

            "r3_modified":
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
        "R4.16 PHASE B — "
        "POST-FREEZE MEASUREMENT INNOVATION AUDIT"
    )
    print("=" * 118)


    print()
    print(
        "Blind feature correlation with "
        "projected geographic error:"
    )


    print(
        correlations.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.3f}",
        )
    )


    print()
    print("=" * 118)
    print(
        "LAST 15 UPDATE DECOMPOSITIONS"
    )
    print("=" * 118)


    print(
        update_eval
        .tail(15)
        .to_string(
            index=False,
            float_format=lambda x:
                f"{x:.3f}",
        )
    )


    print()
    print("=" * 118)
    print(
        "q36 / q37 / q38"
    )
    print("=" * 118)


    late = update_eval[
        update_eval[
            "current_update_query_id"
        ].isin(
            [
                36,
                37,
                38,
            ]
        )
    ]


    print(
        late.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.3f}",
        )
    )


    print()
    print(
        "Detailed q36/q37/q38 candidates:"
    )


    late_candidates = annotated[
        annotated[
            "current_update_query_id"
        ].isin(
            [
                36,
                37,
                38,
            ]
        )
    ].copy()


    show = [
        "current_update_query_id",
        "tile_id",
        "candidate_choice_rank",
        "dino_rank",
        "hybrid_rank",
        "innovation_m",
        "innovation_rank",
        "projected_error_m",
        "prior_prediction_error_m",
        "jacobian_condition",
        "inliers",
        "jacobian_local_area_scale",
        "reprojection_rmse_px",
        "gt_inside_tile",
    ]


    print(
        late_candidates[
            show
        ]
        .sort_values(
            [
                "current_update_query_id",
                "innovation_rank",
            ]
        )
        .to_string(
            index=False,
            float_format=lambda x:
                f"{x:.3f}",
        )
    )


    print()
    print("=" * 118)
    print("R4.16 OUTPUT")
    print("=" * 118)


    print(
        "blind candidates:",
        blind_path,
    )


    print(
        "blind summary:",
        blind_summary_path,
    )


    print(
        "blind freeze manifest:",
        freeze_path,
    )


    print(
        "GT annotation:",
        annotated_path,
    )


    print(
        "update evaluation:",
        update_eval_path,
    )


    print(
        "feature audit:",
        correlation_path,
    )


    print(
        "report:",
        report_path,
    )


    print()


    print(
        "STATUS: "
        "PASS_R4_16_CAUSAL_MEASUREMENT_INNOVATION_AUDIT_EXECUTION"
    )


if __name__ == "__main__":
    main()
