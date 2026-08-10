#!/usr/bin/env python3
"""
R4.8 — causal DINO-guided transform-cluster audit.

Motivation
----------
R4.7 found median original-DINO-rank sum to be the strongest existing
blind discriminator among persistent transform tracks.

This diagnostic asks a narrower causal question:

    At each update, BEFORE temporal track construction,
    does DINO-rank evidence prioritize the geographically better
    transform cluster?

PHASE A — BLIND
----------------
Uses the already-frozen R4.4 transform clusters.

Clusters are ranked lexicographically by:
  1. median_sum_dino_rank          lower is better
  2. representative_sum_dino_rank lower is better
  3. median_sum_hybrid_rank        lower is better
  4. cluster_diameter_m            lower is better
  5. local_cluster_id              deterministic tie-break

There is NO weighted score.

The blind ranking and DINO-Top1 timeline are frozen before GT.

PHASE B — POST-FREEZE GT
------------------------
Only afterward:
  * attach GT transform disagreement;
  * measure DINO-Top1 accuracy;
  * measure where the GT-best cluster ranked by blind DINO evidence;
  * measure Top-1 / Top-3 / Top-5 containment;
  * inspect temporal coherence of the blind-selected transform.

No R3 parameters or outputs are modified.

Command:

SCRIPT=scripts/villoc/research/minimum_confident_bootstrap/diagnostics/r4_8_causal_dino_cluster_audit.py
RUN=outputs/demo_runs/traj01_blind_regression_001
R3=outputs/research_runs/minimum_confident_bootstrap/traj01_blind_r3_001

python "$SCRIPT" \
  --run-root "$RUN" \
  --research-root "$R3" \
  2>&1 | tee \
  "$R3/postfreeze_eval/r4_8_causal_dino_cluster_audit.log"

"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer


# ============================================================
# Helpers
# ============================================================

def sha256(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def model_from_row(row) -> dict:
    return {
        "a_real":
            float(row["a_real"]),

        "a_imag":
            float(row["a_imag"]),

        "b_real":
            float(row["b_real"]),

        "b_imag":
            float(row["b_imag"]),
    }


def apply_similarity(
    xy: np.ndarray,
    model: dict,
) -> np.ndarray:

    p = np.asarray(
        xy,
        dtype=float,
    )

    z = (
        p[:, 0]
        + 1j * p[:, 1]
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


def prediction_distance(
    model_a,
    model_b,
    probes,
) -> float:

    a = apply_similarity(
        probes,
        model_a,
    )

    b = apply_similarity(
        probes,
        model_b,
    )

    d = np.linalg.norm(
        a - b,
        axis=1,
    )

    return float(
        d.max()
    )


def longest_true_run(values):
    best = 0
    current = 0

    for value in values:

        if bool(value):
            current += 1
            best = max(
                best,
                current,
            )

        else:
            current = 0

    return int(best)


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

    out_dir = (
        research_root
        / "postfreeze_eval"
    )


    # ========================================================
    # Frozen R4.4 blind data
    # ========================================================

    clusters_path = (
        out_dir
        / "r4_4_blind_cluster_updates.csv"
    )

    r44_manifest_path = (
        out_dir
        / "r4_4_blind_family_freeze_manifest.json"
    )

    relative_path = (
        run_root
        / "metadata/"
          "s8_xfeat_relative_frontend/"
          "s8r4_xfeat_relative_trajectory_blind_raw.csv"
    )


    manifest = json.loads(
        r44_manifest_path.read_text()
    )


    expected_hash = (
        manifest[
            "blind_outputs"
        ][
            "cluster_updates_sha256"
        ]
    )


    actual_hash = sha256(
        clusters_path
    )


    if (
        expected_hash
        != actual_hash
    ):
        raise RuntimeError(
            "R4.4 blind-cluster hash mismatch."
        )


    clusters = pd.read_csv(
        clusters_path
    )


    clusters[
        "update_query_id"
    ] = pd.to_numeric(
        clusters[
            "update_query_id"
        ],
        errors="raise",
    ).astype(int)


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


    relative = (
        relative
        .sort_values(
            "query_id"
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


    start_q = int(
        relative[
            "query_id"
        ].min()
    )


    lock_q = int(
        clusters[
            "update_query_id"
        ].max()
    )


    family_threshold_m = float(
        manifest[
            "configuration"
        ][
            "family_threshold_m"
        ]
    )


    # ========================================================
    #
    # PHASE A — BLIND PER-UPDATE DINO ORDERING
    #
    # ========================================================

    ranked_groups = []

    selected_rows = []

    previous_selected = None


    for update_q, group in clusters.groupby(
        "update_query_id",
        sort=True,
    ):

        update_q = int(
            update_q
        )


        group = (
            group
            .copy()
            .sort_values(
                [
                    "median_sum_dino_rank",
                    "representative_sum_dino_rank",
                    "median_sum_hybrid_rank",
                    "cluster_diameter_m",
                    "local_cluster_id",
                ],
                ascending=True,
            )
            .reset_index(
                drop=True
            )
        )


        group[
            "blind_dino_cluster_rank"
        ] = np.arange(
            1,
            len(group) + 1,
        )


        ranked_groups.append(
            group
        )


        selected = (
            group.iloc[0]
        )


        if previous_selected is None:

            temporal_jump_m = (
                math.nan
            )

        else:

            probes = (
                relative_by_q
                .loc[
                    [
                        start_q,
                        update_q,
                    ],
                    [
                        "visual_x_px",
                        "visual_y_px",
                    ],
                ]
                .to_numpy(float)
            )


            temporal_jump_m = (
                prediction_distance(
                    model_from_row(
                        previous_selected
                    ),
                    model_from_row(
                        selected
                    ),
                    probes,
                )
            )


        selected_rows.append(
            {
                "update_query_id":
                    update_q,

                "local_cluster_id":
                    int(
                        selected[
                            "local_cluster_id"
                        ]
                    ),

                "track_id":
                    int(
                        selected[
                            "track_id"
                        ]
                    ),

                "member_count":
                    int(
                        selected[
                            "member_count"
                        ]
                    ),

                "median_sum_dino_rank":
                    float(
                        selected[
                            "median_sum_dino_rank"
                        ]
                    ),

                "representative_sum_dino_rank":
                    float(
                        selected[
                            "representative_sum_dino_rank"
                        ]
                    ),

                "median_sum_hybrid_rank":
                    float(
                        selected[
                            "median_sum_hybrid_rank"
                        ]
                    ),

                "median_center_residual_m":
                    float(
                        selected[
                            "median_center_residual_m"
                        ]
                    ),

                "cluster_diameter_m":
                    float(
                        selected[
                            "cluster_diameter_m"
                        ]
                    ),

                "scale_m_per_visual_px":
                    float(
                        selected[
                            "scale_m_per_visual_px"
                        ]
                    ),

                "rotation_deg":
                    float(
                        selected[
                            "rotation_deg"
                        ]
                    ),

                "a_real":
                    float(
                        selected[
                            "a_real"
                        ]
                    ),

                "a_imag":
                    float(
                        selected[
                            "a_imag"
                        ]
                    ),

                "b_real":
                    float(
                        selected[
                            "b_real"
                        ]
                    ),

                "b_imag":
                    float(
                        selected[
                            "b_imag"
                        ]
                    ),

                "representative_tile_ids":
                    str(
                        selected[
                            "representative_tile_ids"
                        ]
                    ),

                "temporal_jump_from_previous_selected_m":
                    temporal_jump_m,
            }
        )


        previous_selected = (
            selected
        )


    ranked = pd.concat(
        ranked_groups,
        ignore_index=True,
    )


    selected = pd.DataFrame(
        selected_rows
    )


    blind_ranked_path = (
        out_dir
        / "r4_8_blind_dino_ranked_clusters.csv"
    )


    blind_selected_path = (
        out_dir
        / "r4_8_blind_dino_top1_timeline.csv"
    )


    blind_manifest_path = (
        out_dir
        / "r4_8_blind_dino_cluster_freeze_manifest.json"
    )


    ranked.to_csv(
        blind_ranked_path,
        index=False,
    )


    selected.to_csv(
        blind_selected_path,
        index=False,
    )


    blind_manifest = {
        "stage":
            "R4.8_BLIND_DINO_CLUSTER_RANK_FREEZE",

        "r4_4_cluster_sha256":
            actual_hash,

        "configuration": {
            "cluster_ranking":
                [
                    "median_sum_dino_rank ASC",
                    "representative_sum_dino_rank ASC",
                    "median_sum_hybrid_rank ASC",
                    "cluster_diameter_m ASC",
                    "local_cluster_id ASC",
                ],

            "weighted_score":
                False,

            "family_threshold_m_inherited_from_r4_4":
                family_threshold_m,
        },

        "counts": {
            "updates":
                int(
                    selected[
                        "update_query_id"
                    ].nunique()
                ),

            "ranked_cluster_rows":
                int(
                    len(
                        ranked
                    )
                ),
        },

        "blind_contract": {
            "gt_used":
                False,

            "reference_used":
                False,

            "oracle_used":
                False,

            "r3_modified":
                False,
        },

        "outputs": {
            "ranked_clusters_csv":
                str(
                    blind_ranked_path
                ),

            "ranked_clusters_sha256":
                sha256(
                    blind_ranked_path
                ),

            "dino_top1_timeline_csv":
                str(
                    blind_selected_path
                ),

            "dino_top1_timeline_sha256":
                sha256(
                    blind_selected_path
                ),
        },
    }


    blind_manifest_path.write_text(
        json.dumps(
            blind_manifest,
            indent=2,
        )
    )


    blind_manifest_sha = sha256(
        blind_manifest_path
    )


    print()
    print("=" * 110)
    print(
        "R4.8 PHASE A — "
        "BLIND DINO CLUSTER RANKING FROZEN"
    )
    print("=" * 110)

    print(
        "updates:",
        len(
            selected
        ),
    )

    print(
        "ranked clusters:",
        len(
            ranked
        ),
    )

    print(
        "family threshold inherited:",
        f"{family_threshold_m:.3f} m",
    )


    finite_jumps = (
        selected[
            "temporal_jump_from_previous_selected_m"
        ]
        .dropna()
    )


    print(
        "DINO-Top1 median temporal jump:",
        (
            f"{finite_jumps.median():.3f} m"
            if len(
                finite_jumps
            )
            else "n/a"
        ),
    )


    print(
        "DINO-Top1 p90 temporal jump:",
        (
            f"{finite_jumps.quantile(0.90):.3f} m"
            if len(
                finite_jumps
            )
            else "n/a"
        ),
    )


    print(
        "blind freeze SHA256:",
        blind_manifest_sha,
    )


    # ========================================================
    #
    # PHASE B — GT ATTACHMENT AFTER FREEZE
    #
    # ========================================================

    reference_path = (
        run_root
        / "evaluation/"
          "reference_attachment.csv"
    )


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
            "Reference attachment missing: "
            + str(
                sorted(
                    missing
                )
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


    prefix = (
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


    # --------------------------------------------------------
    # GT disagreement for every blind-ranked cluster,
    # measured causally at q1/current query probes.
    # --------------------------------------------------------

    gt_disagreements = []


    for _, row in ranked.iterrows():

        update_q = int(
            row[
                "update_query_id"
            ]
        )


        probes = (
            relative_by_q
            .loc[
                [
                    start_q,
                    update_q,
                ],
                [
                    "visual_x_px",
                    "visual_y_px",
                ],
            ]
            .to_numpy(float)
        )


        disagreement = prediction_distance(
            model_from_row(
                row
            ),
            gt_model,
            probes,
        )


        gt_disagreements.append(
            disagreement
        )


    annotated = (
        ranked.copy()
    )


    annotated[
        "postfreeze_gt_disagreement_m"
    ] = gt_disagreements


    # ========================================================
    # Per-update comparison
    # ========================================================

    update_rows = []


    for update_q, group in annotated.groupby(
        "update_query_id",
        sort=True,
    ):

        group = (
            group
            .sort_values(
                "blind_dino_cluster_rank"
            )
            .copy()
        )


        dino_top1 = (
            group.iloc[0]
        )


        gt_best = (
            group
            .sort_values(
                "postfreeze_gt_disagreement_m"
            )
            .iloc[0]
        )


        gt_best_dino_rank = int(
            gt_best[
                "blind_dino_cluster_rank"
            ]
        )


        update_rows.append(
            {
                "update_query_id":
                    int(
                        update_q
                    ),

                "cluster_count":
                    int(
                        len(
                            group
                        )
                    ),

                "dino_top1_cluster_id":
                    int(
                        dino_top1[
                            "local_cluster_id"
                        ]
                    ),

                "dino_top1_track_id":
                    int(
                        dino_top1[
                            "track_id"
                        ]
                    ),

                "dino_top1_gt_disagreement_m":
                    float(
                        dino_top1[
                            "postfreeze_gt_disagreement_m"
                        ]
                    ),

                "dino_top1_median_dino_rank_sum":
                    float(
                        dino_top1[
                            "median_sum_dino_rank"
                        ]
                    ),

                "dino_top1_scale":
                    float(
                        dino_top1[
                            "scale_m_per_visual_px"
                        ]
                    ),

                "dino_top1_rotation_deg":
                    float(
                        dino_top1[
                            "rotation_deg"
                        ]
                    ),

                "gt_best_cluster_id":
                    int(
                        gt_best[
                            "local_cluster_id"
                        ]
                    ),

                "gt_best_disagreement_m":
                    float(
                        gt_best[
                            "postfreeze_gt_disagreement_m"
                        ]
                    ),

                "gt_best_dino_rank":
                    gt_best_dino_rank,

                "gt_best_in_dino_top1":
                    bool(
                        gt_best_dino_rank
                        <= 1
                    ),

                "gt_best_in_dino_top3":
                    bool(
                        gt_best_dino_rank
                        <= 3
                    ),

                "gt_best_in_dino_top5":
                    bool(
                        gt_best_dino_rank
                        <= 5
                    ),
            }
        )


    timeline = pd.DataFrame(
        update_rows
    )


    quarter_tile_m = (
        0.5
        * family_threshold_m
    )


    # family_threshold = 25.6 m
    # inherited spacing = 51.2 m
    map_spacing_m = (
        2.0
        * family_threshold_m
    )


    timeline[
        "dino_top1_within_family_threshold"
    ] = (
        timeline[
            "dino_top1_gt_disagreement_m"
        ]
        <= family_threshold_m
    )


    timeline[
        "dino_top1_within_map_spacing"
    ] = (
        timeline[
            "dino_top1_gt_disagreement_m"
        ]
        <= map_spacing_m
    )


    # ========================================================
    # Summary
    # ========================================================

    n_updates = int(
        len(
            timeline
        )
    )


    top1_containment = int(
        timeline[
            "gt_best_in_dino_top1"
        ].sum()
    )


    top3_containment = int(
        timeline[
            "gt_best_in_dino_top3"
        ].sum()
    )


    top5_containment = int(
        timeline[
            "gt_best_in_dino_top5"
        ].sum()
    )


    median_top1_gt = float(
        timeline[
            "dino_top1_gt_disagreement_m"
        ].median()
    )


    p90_top1_gt = float(
        timeline[
            "dino_top1_gt_disagreement_m"
        ].quantile(
            0.90
        )
    )


    within_family = int(
        timeline[
            "dino_top1_within_family_threshold"
        ].sum()
    )


    within_spacing = int(
        timeline[
            "dino_top1_within_map_spacing"
        ].sum()
    )


    streak_family = longest_true_run(
        timeline[
            "dino_top1_within_family_threshold"
        ].tolist()
    )


    streak_spacing = longest_true_run(
        timeline[
            "dino_top1_within_map_spacing"
        ].tolist()
    )


    annotated_path = (
        out_dir
        / "r4_8_gt_annotated_dino_ranked_clusters.csv"
    )


    timeline_path = (
        out_dir
        / "r4_8_postfreeze_dino_cluster_timeline.csv"
    )


    report_path = (
        out_dir
        / "r4_8_causal_dino_cluster_audit.json"
    )


    annotated.to_csv(
        annotated_path,
        index=False,
    )


    timeline.to_csv(
        timeline_path,
        index=False,
    )


    report = {
        "stage":
            "R4.8_CAUSAL_DINO_GUIDED_CLUSTER_AUDIT",

        "status":
            "PASS_R4_8_CAUSAL_DINO_CLUSTER_AUDIT_EXECUTION",

        "blind_freeze_manifest_sha256":
            blind_manifest_sha,

        "counts": {
            "updates":
                n_updates,

            "gt_best_in_dino_top1":
                top1_containment,

            "gt_best_in_dino_top3":
                top3_containment,

            "gt_best_in_dino_top5":
                top5_containment,

            "dino_top1_within_25_6m":
                within_family,

            "dino_top1_within_51_2m":
                within_spacing,
        },

        "metrics": {
            "dino_top1_gt_median_disagreement_m":
                median_top1_gt,

            "dino_top1_gt_p90_disagreement_m":
                p90_top1_gt,

            "longest_dino_top1_within_25_6m_streak":
                streak_family,

            "longest_dino_top1_within_51_2m_streak":
                streak_spacing,
        },

        "gt_prefix_transform_postfreeze_only": {
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
            "phase_a_used_gt":
                False,

            "gt_loaded_after_blind_ranking_freeze":
                True,

            "weighted_score_used":
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
    # Print evidence
    # ========================================================

    print()
    print("=" * 112)
    print(
        "R4.8 PHASE B — "
        "POST-FREEZE DINO CLUSTER AUDIT"
    )
    print("=" * 112)

    print(
        "GT prefix transform:"
    )

    print(
        "  scale:",
        f"{gt_model['scale']:.6f}",
    )

    print(
        "  rotation:",
        f"{gt_model['rotation_deg']:.3f} deg",
    )

    print()

    print(
        "updates:",
        n_updates,
    )

    print(
        "GT-best cluster ranked DINO Top-1:",
        f"{top1_containment}/{n_updates}",
    )

    print(
        "GT-best cluster ranked DINO Top-3:",
        f"{top3_containment}/{n_updates}",
    )

    print(
        "GT-best cluster ranked DINO Top-5:",
        f"{top5_containment}/{n_updates}",
    )

    print()

    print(
        "DINO-Top1 GT median disagreement:",
        f"{median_top1_gt:.3f} m",
    )

    print(
        "DINO-Top1 GT p90 disagreement:",
        f"{p90_top1_gt:.3f} m",
    )

    print(
        "DINO-Top1 <=25.6 m:",
        f"{within_family}/{n_updates}",
    )

    print(
        "DINO-Top1 <=51.2 m:",
        f"{within_spacing}/{n_updates}",
    )

    print(
        "longest <=25.6 m streak:",
        streak_family,
    )

    print(
        "longest <=51.2 m streak:",
        streak_spacing,
    )


    print()
    print("=" * 112)
    print("PER-UPDATE DINO SELECTION")
    print("=" * 112)

    print(
        timeline[
            [
                "update_query_id",
                "cluster_count",
                "dino_top1_track_id",
                "dino_top1_median_dino_rank_sum",
                "dino_top1_scale",
                "dino_top1_rotation_deg",
                "dino_top1_gt_disagreement_m",
                "gt_best_disagreement_m",
                "gt_best_dino_rank",
                "gt_best_in_dino_top3",
                "gt_best_in_dino_top5",
            ]
        ].to_string(
            index=False,
            float_format=lambda x:
                f"{x:.3f}",
        )
    )


    print()
    print("=" * 112)
    print("R4.8 OUTPUT")
    print("=" * 112)

    print(
        "blind ranked clusters:",
        blind_ranked_path,
    )

    print(
        "blind DINO Top1 timeline:",
        blind_selected_path,
    )

    print(
        "blind freeze manifest:",
        blind_manifest_path,
    )

    print(
        "postfreeze timeline:",
        timeline_path,
    )

    print(
        "report:",
        report_path,
    )

    print()

    print(
        "STATUS: "
        "PASS_R4_8_CAUSAL_DINO_CLUSTER_AUDIT_EXECUTION"
    )


if __name__ == "__main__":
    main()
