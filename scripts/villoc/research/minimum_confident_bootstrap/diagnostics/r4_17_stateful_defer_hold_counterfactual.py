#!/usr/bin/env python3
"""
R4.17 — stateful defer/hold counterfactual.

Question
--------
Can a strictly causal innovation gate prevent a bad new absolute
measurement from corrupting an already coherent blind transform?

This diagnostic simulates three PREDECLARED geometry-relative gates:

    quarter grid spacing = 12.8 m
    half grid spacing    = 25.6 m
    full grid spacing    = 51.2 m

Grid spacing = 51.2 m is inherited from the frozen 512_s256 map.

These thresholds are not optimized or selected in Phase A.

State machine
-------------
1. Seed from first R4.12 update having a finite blind leader.
2. State = last ACCEPTED blind leader transform.
3. For each later update:
      predict current map position from state + XFeat
      compute innovation to each current Top-4 sub-tile observation
      choose minimum-innovation candidate
4. For each predeclared gate independently:
      if minimum innovation <= gate:
          ACCEPT current R4.12 blind leader as new state
      else:
          DEFER and HOLD existing state

PHASE A — BLIND
---------------
No GT/reference/oracle.
All three policy timelines are frozen and hashed.

PHASE B — POST-FREEZE
---------------------
Attach GT only afterward and evaluate:
    active-state prediction error
    accepted vs deferred updates
    q36/q37/q38 behavior
    final prefix state quality

This does not modify R3.

Command:

SCRIPT=scripts/villoc/research/minimum_confident_bootstrap/diagnostics/r4_17_stateful_defer_hold_counterfactual.py
RUN=outputs/demo_runs/traj01_blind_regression_001
R3=outputs/research_runs/minimum_confident_bootstrap/traj01_blind_r3_001

python "$SCRIPT" \
  --run-root "$RUN" \
  --research-root "$R3" \
  2>&1 | tee \
  "$R3/postfreeze_eval/r4_17_stateful_defer_hold_counterfactual.log"

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

POLICIES = {
    "quarter_spacing":
        0.25 * MAP_SPACING_M,

    "half_spacing":
        0.50 * MAP_SPACING_M,

    "full_spacing":
        1.00 * MAP_SPACING_M,
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


    relative_by_q = (
        relative
        .set_index(
            "query_id"
        )
    )


    hypothesis_by_id = (
        hypotheses
        .set_index(
            "hypothesis_id"
        )
    )


    # ========================================================
    # Blind current Top-4 projected candidates
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
            validate="one_to_one",
        )
    )


    # ========================================================
    # Find first available blind leader.
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
            "No R4.12 blind leader available."
        )


    seed_position = int(
        valid_positions[0]
    )


    seed_update = (
        updates.iloc[
            seed_position
        ]
    )


    seed_q = int(
        seed_update[
            "update_query_id"
        ]
    )


    seed_leader_id = int(
        float(
            seed_update[
                "best_blind_pareto_hypothesis_id"
            ]
        )
    )


    if (
        seed_leader_id
        not in hypothesis_by_id.index
    ):

        raise RuntimeError(
            "Seed blind leader missing "
            "from hypothesis table."
        )


    # ========================================================
    #
    # PHASE A — THREE BLIND STATE MACHINES
    #
    # ========================================================

    policy_states = {}


    for policy_name in POLICIES:

        policy_states[
            policy_name
        ] = {
            "active_hypothesis_id":
                seed_leader_id,

            "active_source_update_q":
                seed_q,
        }


    rows = []


    # Record the common seed.
    for policy_name, threshold in (
        POLICIES.items()
    ):

        rows.append(
            {
                "policy":
                    policy_name,

                "threshold_m":
                    float(
                        threshold
                    ),

                "update_query_id":
                    seed_q,

                "action":
                    "SEED",

                "minimum_innovation_m":
                    math.nan,

                "minimum_innovation_spacing_units":
                    math.nan,

                "innovation_best_tile_id":
                    "",

                "innovation_best_choice_rank":
                    math.nan,

                "active_hypothesis_id_before":
                    math.nan,

                "active_source_update_q_before":
                    math.nan,

                "candidate_current_leader_id":
                    seed_leader_id,

                "active_hypothesis_id_after":
                    seed_leader_id,

                "active_source_update_q_after":
                    seed_q,
            }
        )


    for update_position in range(
        seed_position + 1,
        len(
            updates
        ),
    ):

        current_update = (
            updates.iloc[
                update_position
            ]
        )


        current_q = int(
            current_update[
                "update_query_id"
            ]
        )


        current_leader_value = (
            current_update[
                "best_blind_pareto_hypothesis_id"
            ]
        )


        current_has_leader = bool(
            pd.notna(
                current_leader_value
            )
        )


        current_leader_id = (
            int(
                float(
                    current_leader_value
                )
            )
            if current_has_leader
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
                f"current candidates, got "
                f"{len(q_candidates)}."
            )


        # ----------------------------------------------------
        # Policies have independent active states.
        # ----------------------------------------------------

        for policy_name, threshold in (
            POLICIES.items()
        ):

            state = (
                policy_states[
                    policy_name
                ]
            )


            active_id_before = int(
                state[
                    "active_hypothesis_id"
                ]
            )


            active_source_q_before = int(
                state[
                    "active_source_update_q"
                ]
            )


            active_model = (
                hypothesis_by_id.loc[
                    active_id_before
                ]
            )


            prior_prediction = apply_similarity(
                current_visual,
                active_model,
            )[0]


            local = q_candidates.copy()


            local[
                "innovation_m"
            ] = np.hypot(
                local[
                    "projected_easting"
                ]
                - float(
                    prior_prediction[0]
                ),

                local[
                    "projected_northing"
                ]
                - float(
                    prior_prediction[1]
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


            if not current_has_leader:

                action = (
                    "DEFER_NO_CURRENT_LEADER"
                )


            elif (
                minimum_innovation
                <= threshold
            ):

                action = "ACCEPT"

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
                    "DEFER_INNOVATION"
                )


            rows.append(
                {
                    "policy":
                        policy_name,

                    "threshold_m":
                        float(
                            threshold
                        ),

                    "update_query_id":
                        current_q,

                    "action":
                        action,

                    "minimum_innovation_m":
                        minimum_innovation,

                    "minimum_innovation_spacing_units":
                        float(
                            minimum_innovation
                            / MAP_SPACING_M
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

                    "active_hypothesis_id_before":
                        active_id_before,

                    "active_source_update_q_before":
                        active_source_q_before,

                    "candidate_current_leader_id":
                        (
                            current_leader_id
                            if current_has_leader
                            else math.nan
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


    summary = (
        blind
        .groupby(
            "policy"
        )
        .agg(
            threshold_m=(
                "threshold_m",
                "first",
            ),

            updates=(
                "update_query_id",
                "size",
            ),

            accepts=(
                "action",
                lambda x:
                    int(
                        (
                            x
                            == "ACCEPT"
                        ).sum()
                    ),
            ),

            innovation_defers=(
                "action",
                lambda x:
                    int(
                        (
                            x
                            == "DEFER_INNOVATION"
                        ).sum()
                    ),
            ),

            no_leader_defers=(
                "action",
                lambda x:
                    int(
                        (
                            x
                            == "DEFER_NO_CURRENT_LEADER"
                        ).sum()
                    ),
            ),

            final_active_source_update_q=(
                "active_source_update_q_after",
                "last",
            ),

            final_active_hypothesis_id=(
                "active_hypothesis_id_after",
                "last",
            ),
        )
        .reset_index()
    )


    blind_path = (
        out
        / "r4_17_blind_stateful_defer_hold_timeline.csv"
    )


    summary_path = (
        out
        / "r4_17_blind_stateful_defer_hold_summary.csv"
    )


    freeze_path = (
        out
        / "r4_17_blind_stateful_defer_hold_freeze_manifest.json"
    )


    blind.to_csv(
        blind_path,
        index=False,
    )


    summary.to_csv(
        summary_path,
        index=False,
    )


    freeze = {
        "stage":
            "R4.17_BLIND_STATEFUL_DEFER_HOLD_FREEZE",

        "map_spacing_m":
            MAP_SPACING_M,

        "policies": {
            key:
                float(value)
            for key, value
            in POLICIES.items()
        },

        "policy_selection_in_phase_a":
            False,

        "seed": {
            "update_query_id":
                seed_q,

            "blind_leader_hypothesis_id":
                seed_leader_id,
        },

        "state_rule":
            (
                "accept current frozen R4.12 blind leader "
                "iff minimum current Top4 sub-tile innovation "
                "against last accepted transform is within "
                "the policy threshold; otherwise hold state"
            ),

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

            "thresholds_optimized":
                False,

            "one_policy_selected":
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
                    summary_path
                ),

            "summary_sha256":
                sha256(
                    summary_path
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
        "R4.17 PHASE A — "
        "BLIND STATEFUL DEFER/HOLD POLICIES FROZEN"
    )
    print("=" * 118)


    print(
        "seed update:",
        seed_q,
    )


    print(
        "seed hypothesis:",
        seed_leader_id,
    )


    print()


    print(
        summary.to_string(
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
        "Last 12 blind policy rows:"
    )


    print(
        blind
        .groupby(
            "policy",
            group_keys=False,
        )
        .tail(4)
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


    eval_rows = []


    for _, row in blind.iterrows():

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


        predicted = apply_similarity(
            visual,
            active_model,
        )[0]


        gt_row = gt.loc[
            q
        ]


        error = float(
            np.hypot(
                predicted[0]
                - float(
                    gt_row[
                        "gt_easting"
                    ]
                ),

                predicted[1]
                - float(
                    gt_row[
                        "gt_northing"
                    ]
                ),
            )
        )


        eval_rows.append(
            {
                **row.to_dict(),

                "active_state_prediction_error_m":
                    error,
            }
        )


    evaluated = pd.DataFrame(
        eval_rows
    )


    policy_eval = (
        evaluated
        .groupby(
            "policy"
        )
        .agg(
            threshold_m=(
                "threshold_m",
                "first",
            ),

            median_state_error_m=(
                "active_state_prediction_error_m",
                "median",
            ),

            p90_state_error_m=(
                "active_state_prediction_error_m",
                lambda x:
                    float(
                        x.quantile(
                            0.90
                        )
                    ),
            ),

            max_state_error_m=(
                "active_state_prediction_error_m",
                "max",
            ),

            final_state_error_m=(
                "active_state_prediction_error_m",
                "last",
            ),

            accepts=(
                "action",
                lambda x:
                    int(
                        (
                            x
                            == "ACCEPT"
                        ).sum()
                    ),
            ),

            innovation_defers=(
                "action",
                lambda x:
                    int(
                        (
                            x
                            == "DEFER_INNOVATION"
                        ).sum()
                    ),
            ),

            final_active_source_update_q=(
                "active_source_update_q_after",
                "last",
            ),
        )
        .reset_index()
    )


    evaluated_path = (
        out
        / "r4_17_postfreeze_stateful_defer_hold_evaluation.csv"
    )


    policy_eval_path = (
        out
        / "r4_17_postfreeze_policy_summary.csv"
    )


    report_path = (
        out
        / "r4_17_stateful_defer_hold_counterfactual.json"
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
            "R4.17_STATEFUL_DEFER_HOLD_COUNTERFACTUAL",

        "status":
            "PASS_R4_17_STATEFUL_DEFER_HOLD_EXECUTION",

        "blind_freeze_manifest_sha256":
            freeze_sha,

        "contract": {
            "phase_a_used_gt":
                False,

            "gt_loaded_after_policy_freeze":
                True,

            "all_predeclared_policies_reported":
                True,

            "production_threshold_selected":
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
    print("=" * 118)
    print(
        "R4.17 PHASE B — "
        "POST-FREEZE STATEFUL POLICY EVALUATION"
    )
    print("=" * 118)


    print(
        policy_eval.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.3f}",
        )
    )


    print()
    print("=" * 118)
    print("q34..q38 STATE TIMELINES")
    print("=" * 118)


    late = evaluated[
        evaluated[
            "update_query_id"
        ]
        >= 34
    ]


    show = [
        "policy",
        "update_query_id",
        "action",
        "minimum_innovation_m",
        "active_source_update_q_before",
        "candidate_current_leader_id",
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
    print("=" * 118)
    print("R4.17 OUTPUT")
    print("=" * 118)


    print(
        "blind timeline:",
        blind_path,
    )


    print(
        "blind summary:",
        summary_path,
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
        "PASS_R4_17_STATEFUL_DEFER_HOLD_EXECUTION"
    )


if __name__ == "__main__":
    main()
