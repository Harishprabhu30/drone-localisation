#!/usr/bin/env python3
"""
R4.18 — acquisition-to-mature-tracking counterfactual.

R4.17 showed that a fixed innovation gate has two failure modes:

  * tight gate from first provisional state:
        cannot recover from a bad seed;

  * loose gate:
        recovers, but later accepts degrading absolute measurements.

R4.18 explicitly separates:

    ACQUISITION
        provisional state may move freely;

    TRACKING
        innovation gate can defer inconsistent updates.

Blind maturity rule
-------------------
A state becomes mature after SUPPORT_REQUIRED consecutive transitions whose
minimum Top-4 sub-tile innovation is no greater than an activation threshold.

Predeclared activation thresholds:
    quarter spacing = 12.8 m
    half spacing    = 25.6 m

Predeclared tracking thresholds:
    quarter spacing = 12.8 m
    half spacing    = 25.6 m

All four combinations are frozen.
No Phase-A policy selection.

Support count = 3.
No GT/reference/oracle in Phase A.

During ACQUISITION:
    current R4.12 blind leader replaces state whenever one exists.

During TRACKING:
    current leader replaces state only when minimum innovation
    is within the policy tracking threshold; otherwise HOLD.

PHASE B attaches GT after freeze.

Command:

SCRIPT=scripts/villoc/research/minimum_confident_bootstrap/diagnostics/r4_18_acquisition_to_tracking_counterfactual.py
RUN=outputs/demo_runs/traj01_blind_regression_001
R3=outputs/research_runs/minimum_confident_bootstrap/traj01_blind_r3_001

python "$SCRIPT" \
  --run-root "$RUN" \
  --research-root "$R3" \
  2>&1 | tee \
  "$R3/postfreeze_eval/r4_18_acquisition_to_tracking_counterfactual.log"
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
SUPPORT_REQUIRED = 3

ACTIVATION_THRESHOLDS = {
    "quarter":
        0.25 * MAP_SPACING_M,

    "half":
        0.50 * MAP_SPACING_M,
}

TRACKING_THRESHOLDS = {
    "quarter":
        0.25 * MAP_SPACING_M,

    "half":
        0.50 * MAP_SPACING_M,
}


def sha256(path: Path) -> str:

    h = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
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
        float(model["a_real"]),
        float(model["a_imag"]),
    )

    b = complex(
        float(model["b_real"]),
        float(model["b_imag"]),
    )

    w = a * z + b

    return np.column_stack(
        [
            w.real,
            w.imag,
        ]
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

    candidates_path = (
        research
        / "candidate_evidence.csv"
    )

    relative_path = (
        run
        / "metadata/"
          "s8_xfeat_relative_frontend/"
          "s8r4_xfeat_relative_trajectory_blind_raw.csv"
    )


    for path in [
        updates_path,
        hypotheses_path,
        projections_path,
        candidates_path,
        relative_path,
    ]:

        if not path.exists():
            raise RuntimeError(
                f"Missing input: {path}"
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

    candidates = pd.read_csv(
        candidates_path
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


    candidates[
        "query_id"
    ] = pd.to_numeric(
        candidates[
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


    hypothesis_by_id = (
        hypotheses
        .set_index(
            "hypothesis_id"
        )
    )


    relative_by_q = (
        relative
        .set_index(
            "query_id"
        )
    )


    # ========================================================
    # Current Top-4 blind sub-tile observations
    # ========================================================

    candidate_small = candidates[
        [
            "query_id",
            "tile_id",
            "candidate_choice_rank",
            "rank",
            "hybrid_rank",
        ]
    ].rename(
        columns={
            "rank":
                "dino_rank",
        }
    )


    projection_small = projections[
        [
            "query_id",
            "tile_id",
            "projected_easting",
            "projected_northing",
            "projected_inside_tile",
        ]
    ]


    current_candidates = (
        candidate_small
        .merge(
            projection_small,
            on=[
                "query_id",
                "tile_id",
            ],
            how="left",
            validate="one_to_one",
        )
    )


    # ========================================================
    # First available R4.12 blind leader
    # ========================================================

    leader_numeric = pd.to_numeric(
        updates[
            "best_blind_pareto_hypothesis_id"
        ],
        errors="coerce",
    )


    valid_positions = np.flatnonzero(
        leader_numeric.notna().to_numpy()
    )


    if len(
        valid_positions
    ) == 0:

        raise RuntimeError(
            "No finite R4.12 blind leader."
        )


    seed_position = int(
        valid_positions[0]
    )


    seed_update = updates.iloc[
        seed_position
    ]


    seed_q = int(
        seed_update[
            "update_query_id"
        ]
    )


    seed_id = int(
        float(
            seed_update[
                "best_blind_pareto_hypothesis_id"
            ]
        )
    )


    # ========================================================
    # Define all four policies before execution
    # ========================================================

    policies = {}


    for activation_name, activation_threshold in (
        ACTIVATION_THRESHOLDS.items()
    ):

        for tracking_name, tracking_threshold in (
            TRACKING_THRESHOLDS.items()
        ):

            name = (
                f"activate_{activation_name}"
                f"_track_{tracking_name}"
            )

            policies[
                name
            ] = {
                "activation_threshold_m":
                    float(
                        activation_threshold
                    ),

                "tracking_threshold_m":
                    float(
                        tracking_threshold
                    ),
            }


    # ========================================================
    #
    # PHASE A — BLIND
    #
    # ========================================================

    states = {}


    for name in policies:

        states[
            name
        ] = {
            "mode":
                "ACQUISITION",

            "active_hypothesis_id":
                seed_id,

            "active_source_update_q":
                seed_q,

            "consistency_streak":
                0,

            "matured_at_query_id":
                None,
        }


    rows = []


    # Common seed row.
    for name, policy in policies.items():

        rows.append(
            {
                "policy":
                    name,

                "activation_threshold_m":
                    policy[
                        "activation_threshold_m"
                    ],

                "tracking_threshold_m":
                    policy[
                        "tracking_threshold_m"
                    ],

                "support_required":
                    SUPPORT_REQUIRED,

                "update_query_id":
                    seed_q,

                "mode_before":
                    "ACQUISITION",

                "minimum_innovation_m":
                    math.nan,

                "consistency_streak_before":
                    0,

                "consistency_streak_after":
                    0,

                "action":
                    "SEED",

                "mode_after":
                    "ACQUISITION",

                "matured_now":
                    False,

                "matured_at_query_id":
                    math.nan,

                "active_hypothesis_id_before":
                    math.nan,

                "active_source_update_q_before":
                    math.nan,

                "candidate_current_leader_id":
                    seed_id,

                "innovation_best_tile_id":
                    "",

                "innovation_best_choice_rank":
                    math.nan,

                "active_hypothesis_id_after":
                    seed_id,

                "active_source_update_q_after":
                    seed_q,
            }
        )


    for position in range(
        seed_position + 1,
        len(
            updates
        ),
    ):

        current_update = (
            updates.iloc[
                position
            ]
        )


        current_q = int(
            current_update[
                "update_query_id"
            ]
        )


        leader_value = (
            current_update[
                "best_blind_pareto_hypothesis_id"
            ]
        )


        has_current_leader = bool(
            pd.notna(
                leader_value
            )
        )


        current_leader_id = (
            int(
                float(
                    leader_value
                )
            )
            if has_current_leader
            else None
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
                f"q{current_q}: expected 4 "
                f"Top-4 candidates; found "
                f"{len(q_candidates)}."
            )


        for name, policy in policies.items():

            state = states[
                name
            ]


            mode_before = str(
                state[
                    "mode"
                ]
            )


            active_id_before = int(
                state[
                    "active_hypothesis_id"
                ]
            )


            active_source_before = int(
                state[
                    "active_source_update_q"
                ]
            )


            streak_before = int(
                state[
                    "consistency_streak"
                ]
            )


            active_model = (
                hypothesis_by_id.loc[
                    active_id_before
                ]
            )


            prior_prediction = (
                apply_similarity(
                    current_visual,
                    active_model,
                )[0]
            )


            local = q_candidates.copy()


            local[
                "innovation_m"
            ] = np.hypot(
                local[
                    "projected_easting"
                ]
                - float(
                    prior_prediction[
                        0
                    ]
                ),

                local[
                    "projected_northing"
                ]
                - float(
                    prior_prediction[
                        1
                    ]
                ),
            )


            innovation_best = (
                local
                .sort_values(
                    [
                        "innovation_m",
                        "candidate_choice_rank",
                    ]
                )
                .iloc[0]
            )


            minimum_innovation = float(
                innovation_best[
                    "innovation_m"
                ]
            )


            matured_now = False


            # =================================================
            # ACQUISITION MODE
            # =================================================

            if (
                state[
                    "mode"
                ]
                == "ACQUISITION"
            ):

                if not has_current_leader:

                    action = (
                        "ACQUISITION_NO_LEADER"
                    )

                    state[
                        "consistency_streak"
                    ] = 0


                else:

                    # Provisional state is allowed to move.
                    state[
                        "active_hypothesis_id"
                    ] = int(
                        current_leader_id
                    )

                    state[
                        "active_source_update_q"
                    ] = current_q


                    if (
                        minimum_innovation
                        <= policy[
                            "activation_threshold_m"
                        ]
                    ):

                        state[
                            "consistency_streak"
                        ] += 1

                    else:

                        state[
                            "consistency_streak"
                        ] = 0


                    action = (
                        "ACQUISITION_ACCEPT"
                    )


                    if (
                        state[
                            "consistency_streak"
                        ]
                        >= SUPPORT_REQUIRED
                    ):

                        state[
                            "mode"
                        ] = "TRACKING"

                        state[
                            "matured_at_query_id"
                        ] = current_q

                        matured_now = True


            # =================================================
            # TRACKING MODE
            # =================================================

            else:

                if not has_current_leader:

                    action = (
                        "TRACKING_HOLD_NO_LEADER"
                    )


                elif (
                    minimum_innovation
                    <= policy[
                        "tracking_threshold_m"
                    ]
                ):

                    action = (
                        "TRACKING_ACCEPT"
                    )

                    state[
                        "active_hypothesis_id"
                    ] = int(
                        current_leader_id
                    )

                    state[
                        "active_source_update_q"
                    ] = current_q


                else:

                    action = (
                        "TRACKING_HOLD_INNOVATION"
                    )


            rows.append(
                {
                    "policy":
                        name,

                    "activation_threshold_m":
                        policy[
                            "activation_threshold_m"
                        ],

                    "tracking_threshold_m":
                        policy[
                            "tracking_threshold_m"
                        ],

                    "support_required":
                        SUPPORT_REQUIRED,

                    "update_query_id":
                        current_q,

                    "mode_before":
                        mode_before,

                    "minimum_innovation_m":
                        minimum_innovation,

                    "consistency_streak_before":
                        streak_before,

                    "consistency_streak_after":
                        int(
                            state[
                                "consistency_streak"
                            ]
                        ),

                    "action":
                        action,

                    "mode_after":
                        str(
                            state[
                                "mode"
                            ]
                        ),

                    "matured_now":
                        matured_now,

                    "matured_at_query_id":
                        (
                            int(
                                state[
                                    "matured_at_query_id"
                                ]
                            )
                            if state[
                                "matured_at_query_id"
                            ]
                            is not None
                            else math.nan
                        ),

                    "active_hypothesis_id_before":
                        active_id_before,

                    "active_source_update_q_before":
                        active_source_before,

                    "candidate_current_leader_id":
                        (
                            current_leader_id
                            if has_current_leader
                            else math.nan
                        ),

                    "innovation_best_tile_id":
                        str(
                            innovation_best[
                                "tile_id"
                            ]
                        ),

                    "innovation_best_choice_rank":
                        int(
                            innovation_best[
                                "candidate_choice_rank"
                            ]
                        ),

                    "active_hypothesis_id_after":
                        int(
                            state[
                                "active_hypothesis_id"
                            ]
                        ),

                    "active_source_update_q_after":
                        int(
                            state[
                                "active_source_update_q"
                            ]
                        ),
                }
            )


    blind = pd.DataFrame(
        rows
    )


    blind_summary_rows = []


    for name, group in (
        blind.groupby(
            "policy",
            sort=True,
        )
    ):

        matured_values = (
            group[
                "matured_at_query_id"
            ]
            .dropna()
        )


        blind_summary_rows.append(
            {
                "policy":
                    name,

                "activation_threshold_m":
                    float(
                        group[
                            "activation_threshold_m"
                        ].iloc[0]
                    ),

                "tracking_threshold_m":
                    float(
                        group[
                            "tracking_threshold_m"
                        ].iloc[0]
                    ),

                "support_required":
                    SUPPORT_REQUIRED,

                "matured":
                    bool(
                        len(
                            matured_values
                        )
                        > 0
                    ),

                "matured_at_query_id":
                    (
                        int(
                            matured_values.iloc[
                                0
                            ]
                        )
                        if len(
                            matured_values
                        )
                        else math.nan
                    ),

                "acquisition_accepts":
                    int(
                        (
                            group[
                                "action"
                            ]
                            == "ACQUISITION_ACCEPT"
                        ).sum()
                    ),

                "tracking_accepts":
                    int(
                        (
                            group[
                                "action"
                            ]
                            == "TRACKING_ACCEPT"
                        ).sum()
                    ),

                "tracking_holds_innovation":
                    int(
                        (
                            group[
                                "action"
                            ]
                            == "TRACKING_HOLD_INNOVATION"
                        ).sum()
                    ),

                "final_active_source_update_q":
                    int(
                        group[
                            "active_source_update_q_after"
                        ].iloc[
                            -1
                        ]
                    ),

                "final_active_hypothesis_id":
                    int(
                        group[
                            "active_hypothesis_id_after"
                        ].iloc[
                            -1
                        ]
                    ),
            }
        )


    blind_summary = pd.DataFrame(
        blind_summary_rows
    )


    # ========================================================
    # Freeze
    # ========================================================

    blind_path = (
        out
        / "r4_18_blind_acquisition_tracking_timeline.csv"
    )


    blind_summary_path = (
        out
        / "r4_18_blind_acquisition_tracking_summary.csv"
    )


    freeze_path = (
        out
        / "r4_18_blind_acquisition_tracking_freeze_manifest.json"
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
            "R4.18_BLIND_ACQUISITION_TO_TRACKING_FREEZE",

        "map_spacing_m":
            MAP_SPACING_M,

        "support_required":
            SUPPORT_REQUIRED,

        "activation_thresholds_m":
            ACTIVATION_THRESHOLDS,

        "tracking_thresholds_m":
            TRACKING_THRESHOLDS,

        "policies":
            policies,

        "policy_selected":
            False,

        "state_machine": {
            "acquisition":
                (
                    "accept each available current R4.12 "
                    "blind leader; accumulate consecutive "
                    "small-innovation support"
                ),

            "maturity":
                (
                    "enter tracking after three consecutive "
                    "innovations within activation threshold"
                ),

            "tracking":
                (
                    "accept current leader only when current "
                    "minimum Top4 sub-tile innovation is "
                    "within tracking threshold; otherwise hold"
                ),
        },

        "seed": {
            "update_query_id":
                seed_q,

            "hypothesis_id":
                seed_id,
        },

        "inputs": {
            "updates_sha256":
                sha256(
                    updates_path
                ),

            "hypotheses_sha256":
                sha256(
                    hypotheses_path
                ),

            "projections_sha256":
                sha256(
                    projections_path
                ),

            "candidate_evidence_sha256":
                sha256(
                    candidates_path
                ),

            "relative_trajectory_sha256":
                sha256(
                    relative_path
                ),
        },

        "blind_contract": {
            "gt_used":
                False,

            "reference_used":
                False,

            "oracle_used":
                False,

            "policy_selected":
                False,

            "r3_modified":
                False,
        },

        "outputs": {
            "timeline_csv":
                str(
                    blind_path
                ),

            "timeline_sha256":
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
    print("=" * 120)
    print(
        "R4.18 PHASE A — "
        "BLIND ACQUISITION/TRACKING POLICIES FROZEN"
    )
    print("=" * 120)


    print(
        "seed update:",
        seed_q,
    )


    print(
        "seed hypothesis:",
        seed_id,
    )


    print()


    print(
        blind_summary.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.3f}",
        )
    )


    print(
        "blind freeze SHA256:",
        freeze_sha,
    )


    print()
    print(
        "q30..q38 blind state timelines:"
    )


    late_blind = blind[
        blind[
            "update_query_id"
        ]
        >= 30
    ]


    show_blind = [
        "policy",
        "update_query_id",
        "mode_before",
        "minimum_innovation_m",
        "consistency_streak_after",
        "action",
        "mode_after",
        "active_source_update_q_after",
    ]


    print(
        late_blind[
            show_blind
        ]
        .sort_values(
            [
                "policy",
                "update_query_id",
            ]
        )
        .to_string(
            index=False,
            float_format=lambda x:
                f"{x:.3f}",
        )
    )


    # ========================================================
    #
    # PHASE B — GT FIRST READ
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


    gt = (
        evaluation[
            [
                "query_id",
                "gt_easting",
                "gt_northing",
            ]
        ]
        .drop_duplicates(
            "query_id"
        )
        .set_index(
            "query_id"
        )
    )


    evaluated_rows = []


    for _, row in (
        blind.iterrows()
    ):

        q = int(
            row[
                "update_query_id"
            ]
        )


        active_id = int(
            row[
                "active_hypothesis_id_after"
            ]
        )


        active_model = (
            hypothesis_by_id.loc[
                active_id
            ]
        )


        visual = (
            relative_by_q
            .loc[
                [
                    q
                ],
                [
                    "visual_x_px",
                    "visual_y_px",
                ],
            ]
            .to_numpy(float)
        )


        prediction = (
            apply_similarity(
                visual,
                active_model,
            )[0]
        )


        gt_row = gt.loc[
            q
        ]


        state_error = float(
            np.hypot(
                prediction[
                    0
                ]
                - float(
                    gt_row[
                        "gt_easting"
                    ]
                ),

                prediction[
                    1
                ]
                - float(
                    gt_row[
                        "gt_northing"
                    ]
                ),
            )
        )


        evaluated_rows.append(
            {
                **row.to_dict(),

                "active_state_prediction_error_m":
                    state_error,
            }
        )


    evaluated = pd.DataFrame(
        evaluated_rows
    )


    policy_eval_rows = []


    for name, group in (
        evaluated.groupby(
            "policy",
            sort=True,
        )
    ):

        group = group.sort_values(
            "update_query_id"
        )


        matured_at_values = (
            group[
                "matured_at_query_id"
            ]
            .dropna()
        )


        matured_at = (
            int(
                matured_at_values.iloc[
                    0
                ]
            )
            if len(
                matured_at_values
            )
            else None
        )


        if matured_at is not None:

            mature_group = group[
                group[
                    "update_query_id"
                ]
                >= matured_at
            ]

        else:

            mature_group = group.iloc[
                0:0
            ]


        policy_eval_rows.append(
            {
                "policy":
                    name,

                "activation_threshold_m":
                    float(
                        group[
                            "activation_threshold_m"
                        ].iloc[0]
                    ),

                "tracking_threshold_m":
                    float(
                        group[
                            "tracking_threshold_m"
                        ].iloc[0]
                    ),

                "matured_at_query_id":
                    (
                        matured_at
                        if matured_at is not None
                        else math.nan
                    ),

                "median_full_state_error_m":
                    float(
                        group[
                            "active_state_prediction_error_m"
                        ].median()
                    ),

                "p90_full_state_error_m":
                    float(
                        group[
                            "active_state_prediction_error_m"
                        ].quantile(
                            0.90
                        )
                    ),

                "median_post_maturity_error_m":
                    (
                        float(
                            mature_group[
                                "active_state_prediction_error_m"
                            ].median()
                        )
                        if len(
                            mature_group
                        )
                        else math.nan
                    ),

                "p90_post_maturity_error_m":
                    (
                        float(
                            mature_group[
                                "active_state_prediction_error_m"
                            ].quantile(
                                0.90
                            )
                        )
                        if len(
                            mature_group
                        )
                        else math.nan
                    ),

                "final_state_error_m":
                    float(
                        group[
                            "active_state_prediction_error_m"
                        ].iloc[
                            -1
                        ]
                    ),

                "tracking_accepts":
                    int(
                        (
                            group[
                                "action"
                            ]
                            == "TRACKING_ACCEPT"
                        ).sum()
                    ),

                "tracking_holds":
                    int(
                        (
                            group[
                                "action"
                            ]
                            == "TRACKING_HOLD_INNOVATION"
                        ).sum()
                    ),

                "final_active_source_update_q":
                    int(
                        group[
                            "active_source_update_q_after"
                        ].iloc[
                            -1
                        ]
                    ),
            }
        )


    policy_eval = pd.DataFrame(
        policy_eval_rows
    )


    evaluated_path = (
        out
        / "r4_18_postfreeze_acquisition_tracking_evaluation.csv"
    )


    policy_eval_path = (
        out
        / "r4_18_postfreeze_acquisition_tracking_policy_summary.csv"
    )


    report_path = (
        out
        / "r4_18_acquisition_to_tracking_counterfactual.json"
    )


    evaluated.to_csv(
        evaluated_path,
        index=False,
    )


    policy_eval.to_csv(
        policy_eval_path,
        index=False,
    )


    report = {
        "stage":
            "R4.18_ACQUISITION_TO_TRACKING_COUNTERFACTUAL",

        "status":
            "PASS_R4_18_ACQUISITION_TO_TRACKING_EXECUTION",

        "blind_freeze_manifest_sha256":
            freeze_sha,

        "contract": {
            "phase_a_used_gt":
                False,

            "gt_loaded_after_policy_freeze":
                True,

            "all_four_policies_reported":
                True,

            "production_policy_selected":
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


    print()
    print("=" * 120)
    print(
        "R4.18 PHASE B — "
        "POST-FREEZE ACQUISITION/TRACKING EVALUATION"
    )
    print("=" * 120)


    print(
        policy_eval.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.3f}",
        )
    )


    print()
    print("=" * 120)
    print(
        "q34..q38 POST-FREEZE STATE TIMELINES"
    )
    print("=" * 120)


    late = evaluated[
        evaluated[
            "update_query_id"
        ]
        >= 34
    ]


    show = [
        "policy",
        "update_query_id",
        "mode_before",
        "action",
        "minimum_innovation_m",
        "active_source_update_q_before",
        "active_source_update_q_after",
        "active_state_prediction_error_m",
    ]


    print(
        late[
            show
        ]
        .sort_values(
            [
                "policy",
                "update_query_id",
            ]
        )
        .to_string(
            index=False,
            float_format=lambda x:
                f"{x:.3f}",
        )
    )


    print()
    print("=" * 120)
    print("R4.18 OUTPUT")
    print("=" * 120)


    print(
        "blind timeline:",
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
        "postfreeze evaluation:",
        evaluated_path,
    )


    print(
        "policy summary:",
        policy_eval_path,
    )


    print(
        "report:",
        report_path,
    )


    print()


    print(
        "STATUS: "
        "PASS_R4_18_ACQUISITION_TO_TRACKING_EXECUTION"
    )


if __name__ == "__main__":
    main()
