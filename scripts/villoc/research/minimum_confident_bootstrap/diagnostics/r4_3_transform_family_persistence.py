#!/usr/bin/env python3
"""
R4.3 — post-freeze transform-family persistence diagnostic.

Purpose
-------
Determine whether a geographically plausible transform family was already
present repeatedly among the geometrically admissible Top-M hypotheses before
the frozen R3 q38 commitment.

This is deliberately POST-FREEZE / GT-ALLOWED diagnostic code.

It:
  * does NOT rerun retrieval;
  * does NOT rerun ORB;
  * does NOT rerun XFeat;
  * does NOT modify the R3 bootstrap;
  * does NOT change any R3 parameter;
  * does NOT modify frozen blind outputs.

The fixed q1..lock GT transform and the best q38 admissible hypothesis are used
only retrospectively to study historical hypothesis-family persistence.

command:

chmod +x \
  scripts/villoc/research/minimum_confident_bootstrap/diagnostics/r4_3_transform_family_persistence.py

python -m py_compile \
  scripts/villoc/research/minimum_confident_bootstrap/diagnostics/r4_3_transform_family_persistence.py

  
RUN=outputs/demo_runs/traj01_blind_regression_001

R3=outputs/research_runs/minimum_confident_bootstrap/traj01_blind_r3_001

python \
  scripts/villoc/research/minimum_confident_bootstrap/diagnostics/r4_3_transform_family_persistence.py \
  --run-root "$RUN" \
  --research-root "$R3" \
  2>&1 | tee \
  "$R3/postfreeze_eval/r4_3_transform_family_persistence.log"

"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer


# ============================================================
# Similarity helpers
# ============================================================

def fit_similarity(
    visual_xy: np.ndarray,
    map_xy: np.ndarray,
) -> dict:

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
            "Degenerate visual geometry."
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

        "scale_m_per_visual_px":
            float(abs(a)),

        "rotation_deg":
            float(
                np.degrees(
                    np.angle(a)
                )
            ),
    }


def apply_similarity(
    visual_xy: np.ndarray,
    model: dict,
) -> np.ndarray:

    visual_xy = np.asarray(
        visual_xy,
        dtype=float,
    )

    z = (
        visual_xy[:, 0]
        + 1j * visual_xy[:, 1]
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


def model_from_row(
    row: pd.Series,
) -> dict:

    return {
        "a_real":
            float(row["a_real"]),

        "a_imag":
            float(row["a_imag"]),

        "b_real":
            float(row["b_real"]),

        "b_imag":
            float(row["b_imag"]),

        "scale_m_per_visual_px":
            float(
                row[
                    "scale_m_per_visual_px"
                ]
            ),

        "rotation_deg":
            float(
                row[
                    "rotation_deg"
                ]
            ),
    }


def prediction_disagreement(
    model_a: dict,
    model_b: dict,
    sentinel_xy: np.ndarray,
) -> tuple[float, float, float]:

    pa = apply_similarity(
        sentinel_xy,
        model_a,
    )

    pb = apply_similarity(
        sentinel_xy,
        model_b,
    )

    d = np.linalg.norm(
        pa - pb,
        axis=1,
    )

    return (
        float(d[0]),
        float(d[1]),
        float(d.max()),
    )


def bool_series(
    s: pd.Series,
) -> pd.Series:

    return (
        s.astype(str)
        .str.strip()
        .str.lower()
        .isin(
            [
                "true",
                "1",
                "yes",
            ]
        )
    )


def longest_consecutive_run(
    query_ids: list[int],
) -> tuple[int, int | None, int | None]:

    if not query_ids:
        return (
            0,
            None,
            None,
        )

    q = sorted(
        set(
            int(x)
            for x in query_ids
        )
    )

    best_len = 1
    best_start = q[0]
    best_end = q[0]

    start = q[0]
    prev = q[0]

    for x in q[1:]:

        if x == prev + 1:
            prev = x

        else:
            length = (
                prev
                - start
                + 1
            )

            if length > best_len:
                best_len = length
                best_start = start
                best_end = prev

            start = x
            prev = x

    length = (
        prev
        - start
        + 1
    )

    if length > best_len:
        best_len = length
        best_start = start
        best_end = prev

    return (
        int(best_len),
        int(best_start),
        int(best_end),
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

    run_root = (
        args.run_root
        .resolve()
    )

    research_root = (
        args.research_root
        .resolve()
    )


    # --------------------------------------------------------
    # Existing frozen research artifacts
    # --------------------------------------------------------

    report_path = (
        research_root
        / "provisional_bootstrap_report.json"
    )

    hypothesis_path = (
        research_root
        / "hypothesis_updates.csv"
    )

    timeline_path = (
        research_root
        / "provisional_bootstrap_timeline.csv"
    )

    relative_path = (
        run_root
        / "metadata/"
          "s8_xfeat_relative_frontend/"
          "s8r4_xfeat_relative_trajectory_blind_raw.csv"
    )

    reference_path = (
        run_root
        / "evaluation/"
          "reference_attachment.csv"
    )

    out_dir = (
        research_root
        / "postfreeze_eval"
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    # --------------------------------------------------------
    # Frozen lock
    # --------------------------------------------------------

    report = json.loads(
        report_path.read_text()
    )

    if (
        report[
            "localization_state"
        ]
        != "PROVISIONAL_ABSOLUTE_LOCK"
    ):
        raise RuntimeError(
            "R4.3 expects a frozen "
            "PROVISIONAL_ABSOLUTE_LOCK."
        )

    lock = report[
        "map_lock"
    ]

    lock_q = int(
        lock[
            "lock_query_id"
        ]
    )

    tile_size_m = float(
        report[
            "configuration"
        ][
            "tile_size_m_derived"
        ]
    )

    quarter_tile_m = (
        0.25
        * tile_size_m
    )

    half_tile_m = (
        0.50
        * tile_size_m
    )


    # --------------------------------------------------------
    # Load hypotheses produced by the frozen R3 algorithm.
    # --------------------------------------------------------

    hyp = pd.read_csv(
        hypothesis_path
    )

    hyp[
        "update_query_id"
    ] = pd.to_numeric(
        hyp[
            "update_query_id"
        ],
        errors="raise",
    ).astype(int)

    hyp[
        "pareto"
    ] = bool_series(
        hyp[
            "pareto"
        ]
    )

    # Do not inspect anything after the actual lock.
    hyp = hyp[
        hyp[
            "update_query_id"
        ]
        <= lock_q
    ].copy()

    if len(hyp) == 0:
        raise RuntimeError(
            "No stored hypotheses "
            "before frozen lock."
        )


    # --------------------------------------------------------
    # Reference trajectory — POST-FREEZE ONLY.
    # --------------------------------------------------------

    reference = pd.read_csv(
        reference_path
    )

    required_reference = {
        "query_id",
        "eval_ref_lat",
        "eval_ref_lon",
    }

    missing = (
        required_reference
        - set(
            reference.columns
        )
    )

    if missing:
        raise RuntimeError(
            "Reference attachment "
            "missing columns: "
            + str(
                sorted(missing)
            )
        )

    transformer = (
        Transformer.from_crs(
            "EPSG:4326",
            "EPSG:3346",
            always_xy=True,
        )
    )

    gt_e, gt_n = transformer.transform(
        reference[
            "eval_ref_lon"
        ].to_numpy(float),

        reference[
            "eval_ref_lat"
        ].to_numpy(float),
    )

    reference[
        "gt_easting"
    ] = gt_e

    reference[
        "gt_northing"
    ] = gt_n

    reference[
        "query_id"
    ] = pd.to_numeric(
        reference[
            "query_id"
        ],
        errors="raise",
    ).astype(int)


    # --------------------------------------------------------
    # XFeat visual trajectory
    # --------------------------------------------------------

    relative = pd.read_csv(
        relative_path
    )

    relative[
        "query_id"
    ] = pd.to_numeric(
        relative[
            "token0_id"
        ],
        errors="raise",
    ).astype(int)

    merged = (
        relative[
            relative[
                "query_id"
            ]
            <= lock_q
        ]
        .merge(
            reference[
                [
                    "query_id",
                    "gt_easting",
                    "gt_northing",
                ]
            ],
            on="query_id",
            validate="one_to_one",
        )
        .sort_values(
            "query_id"
        )
        .reset_index(
            drop=True
        )
    )

    if len(merged) != lock_q:
        print(
            "NOTE: prefix rows:",
            len(merged),
            "lock query:",
            lock_q,
        )


    # --------------------------------------------------------
    # Fixed retrospective GT transform over q1..lock.
    #
    # Important:
    # This is NOT a candidate algorithm signal.
    #
    # We intentionally use one common transform so transforms
    # from q6, q20, q33, q38 can all be compared in the same
    # coordinate system during post-freeze diagnosis.
    # --------------------------------------------------------

    gt_model = fit_similarity(
        merged[
            [
                "visual_x_px",
                "visual_y_px",
            ]
        ].to_numpy(float),

        merged[
            [
                "gt_easting",
                "gt_northing",
            ]
        ].to_numpy(float),
    )


    # --------------------------------------------------------
    # Common transform probes:
    # q1 and frozen lock query.
    #
    # Earlier historical models are evaluated at q38 only
    # retrospectively so every transform is measured with the
    # same spatial baseline. This is diagnostic, not causal.
    # --------------------------------------------------------

    sentinel = (
        merged[
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


    # --------------------------------------------------------
    # Attach GT-transform disagreement to every stored
    # geometrically admissible hypothesis.
    # --------------------------------------------------------

    gt_start = []
    gt_lock = []
    gt_max = []

    for _, row in hyp.iterrows():

        d0, d1, dm = (
            prediction_disagreement(
                model_from_row(row),
                gt_model,
                sentinel,
            )
        )

        gt_start.append(d0)
        gt_lock.append(d1)
        gt_max.append(dm)

    hyp[
        "gt_disagreement_start_m"
    ] = gt_start

    hyp[
        "gt_disagreement_lock_m"
    ] = gt_lock

    hyp[
        "gt_disagreement_max_m"
    ] = gt_max


    # --------------------------------------------------------
    # Define the retrospective q38 "good family anchor":
    #
    # the geometrically admissible q38 transform with minimum
    # disagreement from the fixed GT transform.
    #
    # Again: post-freeze diagnosis only.
    # --------------------------------------------------------

    lock_hyp = hyp[
        hyp[
            "update_query_id"
        ]
        == lock_q
    ].copy()

    if len(lock_hyp) == 0:
        raise RuntimeError(
            "No hypotheses stored "
            "at frozen lock query."
        )

    anchor_row = (
        lock_hyp
        .sort_values(
            [
                "gt_disagreement_max_m",
                "median_center_residual_m",
                "sum_hybrid_rank",
            ]
        )
        .iloc[0]
    )

    anchor_model = (
        model_from_row(
            anchor_row
        )
    )


    # --------------------------------------------------------
    # How close was every earlier hypothesis to that q38
    # transform family?
    # --------------------------------------------------------

    family_start = []
    family_lock = []
    family_max = []

    for _, row in hyp.iterrows():

        d0, d1, dm = (
            prediction_disagreement(
                model_from_row(row),
                anchor_model,
                sentinel,
            )
        )

        family_start.append(d0)
        family_lock.append(d1)
        family_max.append(dm)

    hyp[
        "anchor_family_start_m"
    ] = family_start

    hyp[
        "anchor_family_lock_m"
    ] = family_lock

    hyp[
        "anchor_family_max_m"
    ] = family_max


    # --------------------------------------------------------
    # Per-update retrospective summary.
    # --------------------------------------------------------

    rows = []

    for qid, group in hyp.groupby(
        "update_query_id",
        sort=True,
    ):

        group = group.copy()

        best_gt = (
            group
            .sort_values(
                [
                    "gt_disagreement_max_m",
                    "median_center_residual_m",
                ]
            )
            .iloc[0]
        )

        pareto_group = group[
            group[
                "pareto"
            ]
        ].copy()

        if len(pareto_group):

            best_pareto_gt = (
                pareto_group
                .sort_values(
                    [
                        "gt_disagreement_max_m",
                        "median_center_residual_m",
                    ]
                )
                .iloc[0]
            )

            best_pareto_gt_m = float(
                best_pareto_gt[
                    "gt_disagreement_max_m"
                ]
            )

        else:

            best_pareto_gt_m = (
                float("nan")
            )

        best_family = (
            group
            .sort_values(
                [
                    "anchor_family_max_m",
                    "median_center_residual_m",
                ]
            )
            .iloc[0]
        )

        family_10 = (
            group[
                "anchor_family_max_m"
            ]
            <= 10.0
        )

        family_20 = (
            group[
                "anchor_family_max_m"
            ]
            <= 20.0
        )

        family_quarter = (
            group[
                "anchor_family_max_m"
            ]
            <= quarter_tile_m
        )

        family_half = (
            group[
                "anchor_family_max_m"
            ]
            <= half_tile_m
        )

        family_quarter_pareto = (
            family_quarter
            & group[
                "pareto"
            ]
        )

        rows.append({

            "update_query_id":
                int(qid),

            "admissible_hypotheses":
                int(
                    len(group)
                ),

            "pareto_hypotheses":
                int(
                    group[
                        "pareto"
                    ].sum()
                ),

            "best_gt_disagreement_m":
                float(
                    best_gt[
                        "gt_disagreement_max_m"
                    ]
                ),

            "best_gt_is_pareto":
                bool(
                    best_gt[
                        "pareto"
                    ]
                ),

            "best_gt_tile_ids":
                str(
                    best_gt[
                        "tile_ids"
                    ]
                ),

            "best_gt_scale":
                float(
                    best_gt[
                        "scale_m_per_visual_px"
                    ]
                ),

            "best_gt_rotation_deg":
                float(
                    best_gt[
                        "rotation_deg"
                    ]
                ),

            "best_pareto_gt_disagreement_m":
                best_pareto_gt_m,

            "best_anchor_family_disagreement_m":
                float(
                    best_family[
                        "anchor_family_max_m"
                    ]
                ),

            "best_anchor_family_is_pareto":
                bool(
                    best_family[
                        "pareto"
                    ]
                ),

            "best_anchor_family_tile_ids":
                str(
                    best_family[
                        "tile_ids"
                    ]
                ),

            "best_anchor_family_scale":
                float(
                    best_family[
                        "scale_m_per_visual_px"
                    ]
                ),

            "best_anchor_family_rotation_deg":
                float(
                    best_family[
                        "rotation_deg"
                    ]
                ),

            "family_within_10m_count":
                int(
                    family_10.sum()
                ),

            "family_within_20m_count":
                int(
                    family_20.sum()
                ),

            "family_within_quarter_tile_count":
                int(
                    family_quarter.sum()
                ),

            "family_within_half_tile_count":
                int(
                    family_half.sum()
                ),

            "family_within_quarter_tile_pareto_count":
                int(
                    family_quarter_pareto.sum()
                ),
        })


    persistence = (
        pd.DataFrame(rows)
        .sort_values(
            "update_query_id"
        )
        .reset_index(
            drop=True
        )
    )


    # --------------------------------------------------------
    # Join original causal action for interpretation.
    # --------------------------------------------------------

    timeline = pd.read_csv(
        timeline_path
    )

    timeline[
        "update_query_id"
    ] = pd.to_numeric(
        timeline[
            "update_query_id"
        ],
        errors="raise",
    ).astype(int)

    persistence = (
        persistence
        .merge(
            timeline[
                [
                    "update_query_id",
                    "evidence_query_ids",
                    "action",
                    "transform_clusters",
                ]
            ],
            on="update_query_id",
            how="left",
            validate="one_to_one",
        )
    )


    # --------------------------------------------------------
    # Persistence counts/streaks
    # --------------------------------------------------------

    quarter_present = (
        persistence[
            persistence[
                "family_within_quarter_tile_count"
            ]
            > 0
        ][
            "update_query_id"
        ]
        .astype(int)
        .tolist()
    )

    half_present = (
        persistence[
            persistence[
                "family_within_half_tile_count"
            ]
            > 0
        ][
            "update_query_id"
        ]
        .astype(int)
        .tolist()
    )

    quarter_pareto_present = (
        persistence[
            persistence[
                "family_within_quarter_tile_pareto_count"
            ]
            > 0
        ][
            "update_query_id"
        ]
        .astype(int)
        .tolist()
    )

    (
        quarter_streak,
        quarter_streak_start,
        quarter_streak_end,
    ) = longest_consecutive_run(
        quarter_present
    )

    (
        half_streak,
        half_streak_start,
        half_streak_end,
    ) = longest_consecutive_run(
        half_present
    )

    (
        pareto_streak,
        pareto_streak_start,
        pareto_streak_end,
    ) = longest_consecutive_run(
        quarter_pareto_present
    )


    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    hypotheses_out = (
        out_dir
        / "r4_3_hypothesis_family_truth_diagnostic.csv"
    )

    persistence_out = (
        out_dir
        / "r4_3_transform_family_persistence.csv"
    )

    report_out = (
        out_dir
        / "r4_3_transform_family_persistence.json"
    )

    hyp.to_csv(
        hypotheses_out,
        index=False,
    )

    persistence.to_csv(
        persistence_out,
        index=False,
    )

    result = {

        "stage":
            "R4.3_POSTFREEZE_TRANSFORM_FAMILY_PERSISTENCE",

        "lock_query_id":
            lock_q,

        "tile_size_m":
            tile_size_m,

        "diagnostic_thresholds_m": {
            "10m":
                10.0,

            "20m":
                20.0,

            "quarter_tile":
                quarter_tile_m,

            "half_tile":
                half_tile_m,
        },

        "fixed_gt_prefix_transform": {
            "scale_m_per_visual_px":
                gt_model[
                    "scale_m_per_visual_px"
                ],

            "rotation_deg":
                gt_model[
                    "rotation_deg"
                ],
        },

        "q38_retrospective_anchor_family": {
            "tile_ids":
                str(
                    anchor_row[
                        "tile_ids"
                    ]
                ),

            "candidate_choice_ranks":
                str(
                    anchor_row[
                        "candidate_choice_ranks"
                    ]
                ),

            "pareto":
                bool(
                    anchor_row[
                        "pareto"
                    ]
                ),

            "scale_m_per_visual_px":
                float(
                    anchor_row[
                        "scale_m_per_visual_px"
                    ]
                ),

            "rotation_deg":
                float(
                    anchor_row[
                        "rotation_deg"
                    ]
                ),

            "median_center_residual_m":
                float(
                    anchor_row[
                        "median_center_residual_m"
                    ]
                ),

            "gt_disagreement_max_m":
                float(
                    anchor_row[
                        "gt_disagreement_max_m"
                    ]
                ),
        },

        "persistence": {

            "evaluated_updates":
                int(
                    len(
                        persistence
                    )
                ),

            "quarter_tile_family_present_updates":
                int(
                    len(
                        quarter_present
                    )
                ),

            "quarter_tile_family_first_query":
                (
                    int(
                        quarter_present[0]
                    )
                    if quarter_present
                    else None
                ),

            "quarter_tile_family_longest_consecutive_streak":
                quarter_streak,

            "quarter_tile_family_streak_start":
                quarter_streak_start,

            "quarter_tile_family_streak_end":
                quarter_streak_end,

            "half_tile_family_present_updates":
                int(
                    len(
                        half_present
                    )
                ),

            "half_tile_family_first_query":
                (
                    int(
                        half_present[0]
                    )
                    if half_present
                    else None
                ),

            "half_tile_family_longest_consecutive_streak":
                half_streak,

            "half_tile_family_streak_start":
                half_streak_start,

            "half_tile_family_streak_end":
                half_streak_end,

            "quarter_tile_family_retained_by_pareto_updates":
                int(
                    len(
                        quarter_pareto_present
                    )
                ),

            "quarter_tile_family_pareto_longest_consecutive_streak":
                pareto_streak,

            "quarter_tile_family_pareto_streak_start":
                pareto_streak_start,

            "quarter_tile_family_pareto_streak_end":
                pareto_streak_end,
        },

        "evaluation_contract": {
            "reference_used":
                True,

            "reference_usage":
                (
                    "postfreeze diagnosis only; "
                    "not algorithm input"
                ),

            "algorithm_parameters_modified":
                False,

            "blind_outputs_modified":
                False,
        },

        "outputs": {
            "hypothesis_diagnostic_csv":
                str(
                    hypotheses_out
                ),

            "persistence_csv":
                str(
                    persistence_out
                ),
        },
    }

    report_out.write_text(
        json.dumps(
            result,
            indent=2,
        )
    )


    # --------------------------------------------------------
    # Print evidence
    # --------------------------------------------------------

    print()
    print("=" * 110)
    print("R4.3 — POST-FREEZE TRANSFORM-FAMILY PERSISTENCE")
    print("=" * 110)

    print(
        "frozen lock query:",
        lock_q,
    )

    print(
        "tile size:",
        f"{tile_size_m:.3f} m",
    )

    print(
        "quarter-tile diagnostic:",
        f"{quarter_tile_m:.3f} m",
    )

    print(
        "half-tile diagnostic:",
        f"{half_tile_m:.3f} m",
    )

    print()

    print(
        "GT prefix transform:"
    )

    print(
        "  scale:",
        f"{gt_model['scale_m_per_visual_px']:.6f}",
    )

    print(
        "  rotation:",
        f"{gt_model['rotation_deg']:.3f} deg",
    )

    print()

    print(
        "q38 retrospective anchor family:"
    )

    print(
        "  tiles:",
        anchor_row[
            "tile_ids"
        ],
    )

    print(
        "  candidate choices:",
        anchor_row[
            "candidate_choice_ranks"
        ],
    )

    print(
        "  pareto:",
        bool(
            anchor_row[
                "pareto"
            ]
        ),
    )

    print(
        "  scale:",
        f"{anchor_row['scale_m_per_visual_px']:.6f}",
    )

    print(
        "  rotation:",
        f"{anchor_row['rotation_deg']:.3f} deg",
    )

    print(
        "  center residual:",
        f"{anchor_row['median_center_residual_m']:.3f} m",
    )

    print(
        "  GT transform disagreement:",
        f"{anchor_row['gt_disagreement_max_m']:.3f} m",
    )


    print()
    print("=" * 110)
    print("PERSISTENCE SUMMARY")
    print("=" * 110)

    print(
        "evaluated updates:",
        len(
            persistence
        ),
    )

    print(
        "quarter-tile family present:",
        len(
            quarter_present
        ),
        "updates",
    )

    print(
        "quarter-tile first query:",
        (
            quarter_present[0]
            if quarter_present
            else None
        ),
    )

    print(
        "quarter-tile longest streak:",
        quarter_streak,
        (
            f"(q{quarter_streak_start}"
            f"..q{quarter_streak_end})"
            if quarter_streak_start
            is not None
            else ""
        ),
    )

    print()

    print(
        "half-tile family present:",
        len(
            half_present
        ),
        "updates",
    )

    print(
        "half-tile first query:",
        (
            half_present[0]
            if half_present
            else None
        ),
    )

    print(
        "half-tile longest streak:",
        half_streak,
        (
            f"(q{half_streak_start}"
            f"..q{half_streak_end})"
            if half_streak_start
            is not None
            else ""
        ),
    )

    print()

    print(
        "quarter-tile family retained by Pareto:",
        len(
            quarter_pareto_present
        ),
        "updates",
    )

    print(
        "Pareto longest streak:",
        pareto_streak,
        (
            f"(q{pareto_streak_start}"
            f"..q{pareto_streak_end})"
            if pareto_streak_start
            is not None
            else ""
        ),
    )


    print()
    print("=" * 110)
    print("PER-UPDATE FAMILY EVIDENCE")
    print("=" * 110)

    show_cols = [
        "update_query_id",
        "evidence_query_ids",
        "admissible_hypotheses",
        "pareto_hypotheses",
        "best_gt_disagreement_m",
        "best_gt_is_pareto",
        "best_gt_scale",
        "best_gt_rotation_deg",
        "best_anchor_family_disagreement_m",
        "best_anchor_family_is_pareto",
        "family_within_quarter_tile_count",
        "family_within_quarter_tile_pareto_count",
        "transform_clusters",
        "action",
    ]

    print(
        persistence[
            show_cols
        ].to_string(
            index=False,
            float_format=lambda x:
                f"{x:.3f}",
        )
    )


    print()
    print("=" * 110)
    print("R4.3 OUTPUT")
    print("=" * 110)

    print(
        "hypothesis diagnostic:",
        hypotheses_out,
    )

    print(
        "persistence:",
        persistence_out,
    )

    print(
        "report:",
        report_out,
    )

    print()

    print(
        "No algorithm inputs, parameters, "
        "or frozen blind outputs were modified."
    )

    print(
        "STATUS: "
        "PASS_R4_3_TRANSFORM_FAMILY_PERSISTENCE_EXECUTION"
    )


if __name__ == "__main__":
    main()
