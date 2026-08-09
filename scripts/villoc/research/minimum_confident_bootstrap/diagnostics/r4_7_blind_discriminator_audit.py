#!/usr/bin/env python3
"""
R4.7 — blind-observable discriminator audit.

Question
--------
R4.4 showed:
  * correct transform families can persist;
  * wrong transform families can also persist.

R4.5/R4.6 showed:
  * connected-component basins are structurally unsuitable.

Before designing another tracker, determine whether existing BLIND
observables distinguish good persistent tracks from bad persistent tracks.

PHASE A — BLIND
----------------
Aggregate track features from the already-frozen R4.4 blind outputs.
Freeze and hash them before GT is read.

PHASE B — POST-FREEZE GT
------------------------
Attach the already-existing post-freeze R4.4 GT labels.

This is a diagnostic only:
  * no weighted confidence score;
  * no new lock policy;
  * no R3 modification;
  * no parameter tuning.

Command:


SCRIPT=scripts/villoc/research/minimum_confident_bootstrap/diagnostics/r4_7_blind_discriminator_audit.py 

R3=outputs/research_runs/minimum_confident_bootstrap/traj01_blind_r3_001  
python "$SCRIPT" \   
--research-root "$R3" \   
--min-updates 5 \   
2>&1 | tee \   "$R3/postfreeze_eval/r4_7_blind_discriminator_audit.log"

"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


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


def percentile_rank(
    values: pd.Series,
    value: float,
    higher_is_better: bool,
) -> float:

    x = pd.to_numeric(
        values,
        errors="coerce",
    ).dropna()

    if len(x) == 0:
        return math.nan

    if higher_is_better:

        return float(
            100.0
            * np.mean(
                x <= value
            )
        )

    return float(
        100.0
        * np.mean(
            x >= value
        )
    )


def desired_rank(
    values: pd.Series,
    higher_is_better: bool,
) -> pd.Series:

    return values.rank(
        method="min",
        ascending=not higher_is_better,
    )


def circular_unwrapped_deg(
    values,
):
    x = np.asarray(
        values,
        dtype=float,
    )

    if len(x) == 0:
        return x

    return np.degrees(
        np.unwrap(
            np.radians(
                x
            )
        )
    )


def pareto_front(
    df: pd.DataFrame,
    max_features: list[str],
    min_features: list[str],
) -> pd.Series:
    """
    True where a row is not dominated.

    A dominates B iff:
      * A is at least as good on every declared objective, and
      * strictly better on at least one.

    No weighted score is formed.
    """

    n = len(df)

    nondominated = np.ones(
        n,
        dtype=bool,
    )

    values = (
        df[
            max_features
            + min_features
        ]
        .to_numpy(float)
    )

    nmax = len(
        max_features
    )

    for i in range(n):

        if not nondominated[i]:
            continue

        for j in range(n):

            if i == j:
                continue

            # j dominates i?
            max_ok = np.all(
                values[
                    j,
                    :nmax,
                ]
                >= values[
                    i,
                    :nmax,
                ]
            )

            min_ok = np.all(
                values[
                    j,
                    nmax:,
                ]
                <= values[
                    i,
                    nmax:,
                ]
            )

            if not (
                max_ok
                and min_ok
            ):
                continue

            strict = (
                np.any(
                    values[
                        j,
                        :nmax,
                    ]
                    >
                    values[
                        i,
                        :nmax,
                    ]
                )
                or
                np.any(
                    values[
                        j,
                        nmax:,
                    ]
                    <
                    values[
                        i,
                        nmax:,
                    ]
                )
            )

            if strict:
                nondominated[i] = False
                break

    return pd.Series(
        nondominated,
        index=df.index,
    )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--research-root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--min-updates",
        type=int,
        default=5,
    )

    args = parser.parse_args()

    root = (
        args.research_root
        .resolve()
    )

    out_dir = (
        root
        / "postfreeze_eval"
    )

    min_updates = int(
        args.min_updates
    )


    # ========================================================
    # R4.4 frozen blind inputs
    # ========================================================

    cluster_path = (
        out_dir
        / "r4_4_blind_cluster_updates.csv"
    )

    track_path = (
        out_dir
        / "r4_4_blind_family_tracks.csv"
    )

    manifest_path = (
        out_dir
        / "r4_4_blind_family_freeze_manifest.json"
    )


    required = [
        cluster_path,
        track_path,
        manifest_path,
    ]


    missing = [
        str(x)
        for x in required
        if not x.exists()
    ]


    if missing:

        raise RuntimeError(
            "Missing R4.4 inputs:\n"
            + "\n".join(
                missing
            )
        )


    manifest = json.loads(
        manifest_path.read_text()
    )


    if (
        sha256(
            cluster_path
        )
        != manifest[
            "blind_outputs"
        ][
            "cluster_updates_sha256"
        ]
    ):
        raise RuntimeError(
            "R4.4 cluster hash mismatch."
        )


    if (
        sha256(
            track_path
        )
        != manifest[
            "blind_outputs"
        ][
            "family_tracks_sha256"
        ]
    ):
        raise RuntimeError(
            "R4.4 track hash mismatch."
        )


    clusters = pd.read_csv(
        cluster_path
    )


    tracks = pd.read_csv(
        track_path
    )


    clusters[
        "track_id"
    ] = pd.to_numeric(
        clusters[
            "track_id"
        ],
        errors="raise",
    ).astype(int)


    clusters[
        "update_query_id"
    ] = pd.to_numeric(
        clusters[
            "update_query_id"
        ],
        errors="raise",
    ).astype(int)


    tracks[
        "track_id"
    ] = pd.to_numeric(
        tracks[
            "track_id"
        ],
        errors="raise",
    ).astype(int)


    # ========================================================
    #
    # PHASE A — BLIND FEATURE AGGREGATION
    #
    # No GT-labelled file has been read.
    #
    # ========================================================

    rows = []


    for track_id, group in clusters.groupby(
        "track_id",
        sort=True,
    ):

        group = (
            group
            .sort_values(
                "update_query_id"
            )
            .reset_index(
                drop=True
            )
        )


        base = tracks[
            tracks[
                "track_id"
            ]
            == track_id
        ]


        if len(base) != 1:

            raise RuntimeError(
                f"Expected one blind summary "
                f"for track {track_id}."
            )


        base = base.iloc[0]


        assoc = pd.to_numeric(
            group[
                "association_distance_m"
            ],
            errors="coerce",
        ).dropna()


        scales = pd.to_numeric(
            group[
                "scale_m_per_visual_px"
            ],
            errors="coerce",
        ).to_numpy(float)


        rotations = circular_unwrapped_deg(
            pd.to_numeric(
                group[
                    "rotation_deg"
                ],
                errors="coerce",
            ).to_numpy(float)
        )


        member_counts = pd.to_numeric(
            group[
                "member_count"
            ],
            errors="coerce",
        )


        total_support = float(
            member_counts.sum()
        )


        update_count = int(
            base[
                "update_count"
            ]
        )


        rows.append(
            {
                "track_id":
                    int(
                        track_id
                    ),

                # --------------------------------------------
                # Persistence
                # --------------------------------------------

                "start_query_id":
                    int(
                        base[
                            "start_query_id"
                        ]
                    ),

                "end_query_id":
                    int(
                        base[
                            "end_query_id"
                        ]
                    ),

                "update_count":
                    update_count,

                "reached_lock":
                    bool(
                        base[
                            "reached_lock"
                        ]
                    ),

                # --------------------------------------------
                # Support
                # --------------------------------------------

                "total_hypothesis_support":
                    total_support,

                "support_per_update":
                    float(
                        total_support
                        / update_count
                    ),

                "median_member_count":
                    float(
                        member_counts.median()
                    ),

                "max_member_count":
                    float(
                        member_counts.max()
                    ),

                "distinct_representative_tile_sequences":
                    int(
                        group[
                            "representative_tile_ids"
                        ].nunique()
                    ),

                # --------------------------------------------
                # Temporal transform stability
                # --------------------------------------------

                "median_association_distance_m":
                    (
                        float(
                            assoc.median()
                        )
                        if len(
                            assoc
                        )
                        else math.nan
                    ),

                "p90_association_distance_m":
                    (
                        float(
                            np.percentile(
                                assoc,
                                90,
                            )
                        )
                        if len(
                            assoc
                        )
                        else math.nan
                    ),

                "median_cluster_diameter_m":
                    float(
                        pd.to_numeric(
                            group[
                                "cluster_diameter_m"
                            ],
                            errors="coerce",
                        ).median()
                    ),

                "p90_cluster_diameter_m":
                    float(
                        pd.to_numeric(
                            group[
                                "cluster_diameter_m"
                            ],
                            errors="coerce",
                        ).quantile(
                            0.90
                        )
                    ),

                "scale_median":
                    float(
                        np.median(
                            scales
                        )
                    ),

                "scale_iqr":
                    float(
                        np.percentile(
                            scales,
                            75,
                        )
                        -
                        np.percentile(
                            scales,
                            25,
                        )
                    ),

                "scale_range":
                    float(
                        np.max(
                            scales
                        )
                        -
                        np.min(
                            scales
                        )
                    ),

                "rotation_median_deg":
                    float(
                        np.median(
                            rotations
                        )
                    ),

                "rotation_iqr_deg":
                    float(
                        np.percentile(
                            rotations,
                            75,
                        )
                        -
                        np.percentile(
                            rotations,
                            25,
                        )
                    ),

                "rotation_range_deg":
                    float(
                        np.max(
                            rotations
                        )
                        -
                        np.min(
                            rotations
                        )
                    ),

                # --------------------------------------------
                # Candidate / geometric evidence
                # --------------------------------------------

                "median_center_residual_m":
                    float(
                        pd.to_numeric(
                            group[
                                "median_center_residual_m"
                            ],
                            errors="coerce",
                        ).median()
                    ),

                "median_hybrid_rank_sum":
                    float(
                        pd.to_numeric(
                            group[
                                "median_sum_hybrid_rank"
                            ],
                            errors="coerce",
                        ).median()
                    ),

                "median_dino_rank_sum":
                    float(
                        pd.to_numeric(
                            group[
                                "median_sum_dino_rank"
                            ],
                            errors="coerce",
                        ).median()
                    ),
            }
        )


    blind_features = pd.DataFrame(
        rows
    )


    # --------------------------------------------------------
    # Keep ALL tracks in frozen output.
    # min_updates is only the declared analysis subset.
    # --------------------------------------------------------

    blind_features[
        "persistent_subset"
    ] = (
        blind_features[
            "update_count"
        ]
        >= min_updates
    )


    blind_features_path = (
        out_dir
        / "r4_7_blind_track_features.csv"
    )


    blind_manifest_path = (
        out_dir
        / "r4_7_blind_feature_freeze_manifest.json"
    )


    blind_features.to_csv(
        blind_features_path,
        index=False,
    )


    blind_manifest = {
        "stage":
            "R4.7_BLIND_OBSERVABLE_FEATURE_FREEZE",

        "r4_4_manifest_sha256":
            sha256(
                manifest_path
            ),

        "analysis_subset": {
            "minimum_track_updates":
                min_updates,

            "purpose":
                (
                    "diagnostic comparison only; "
                    "not an online threshold"
                ),
        },

        "feature_groups": {
            "persistence_support": [
                "update_count",
                "total_hypothesis_support",
                "support_per_update",
                "median_member_count",
                "distinct_representative_tile_sequences",
            ],

            "temporal_stability": [
                "median_association_distance_m",
                "p90_association_distance_m",
                "median_cluster_diameter_m",
                "scale_iqr",
                "scale_range",
                "rotation_iqr_deg",
                "rotation_range_deg",
            ],

            "candidate_geometry": [
                "median_center_residual_m",
                "median_hybrid_rank_sum",
                "median_dino_rank_sum",
            ],
        },

        "blind_contract": {
            "gt_used":
                False,

            "reference_used":
                False,

            "oracle_used":
                False,

            "weighted_confidence_score_created":
                False,

            "r3_modified":
                False,
        },

        "output": {
            "blind_features_csv":
                str(
                    blind_features_path
                ),

            "blind_features_sha256":
                sha256(
                    blind_features_path
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
        "R4.7 PHASE A — "
        "BLIND TRACK FEATURES FROZEN"
    )
    print("=" * 110)

    print(
        "all blind tracks:",
        len(
            blind_features
        ),
    )

    print(
        f"tracks with >= {min_updates} updates:",
        int(
            blind_features[
                "persistent_subset"
            ].sum()
        ),
    )

    print(
        "blind feature freeze SHA256:",
        blind_manifest_sha,
    )


    # ========================================================
    #
    # PHASE B — POST-FREEZE GT LABEL ATTACHMENT
    #
    # FIRST GT-labelled input read occurs here.
    #
    # ========================================================

    annotated_track_path = (
        out_dir
        / "r4_4_gt_annotated_family_tracks.csv"
    )


    annotated = pd.read_csv(
        annotated_track_path
    )


    annotated[
        "track_id"
    ] = pd.to_numeric(
        annotated[
            "track_id"
        ],
        errors="raise",
    ).astype(int)


    gt_columns = [
        "track_id",
        "postfreeze_gt_best_disagreement_m",
        "postfreeze_gt_median_disagreement_m",
        "postfreeze_gt_final_disagreement_m",
        "postfreeze_gt_final_query_id",
    ]


    analysis = (
        blind_features
        .merge(
            annotated[
                gt_columns
            ],
            on="track_id",
            how="left",
            validate="one_to_one",
        )
    )


    persistent = (
        analysis[
            analysis[
                "persistent_subset"
            ]
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )


    if len(
        persistent
    ) == 0:

        raise RuntimeError(
            "No persistent tracks "
            "for R4.7 analysis."
        )


    gt_target = (
        "postfreeze_gt_median_disagreement_m"
    )


    # ========================================================
    # Predeclared blind features and desired direction
    # ========================================================

    feature_specs = [
        (
            "update_count",
            True,
            "persistence",
        ),

        (
            "total_hypothesis_support",
            True,
            "support",
        ),

        (
            "support_per_update",
            True,
            "support",
        ),

        (
            "median_member_count",
            True,
            "support",
        ),

        (
            "distinct_representative_tile_sequences",
            True,
            "support",
        ),

        (
            "median_association_distance_m",
            False,
            "stability",
        ),

        (
            "p90_association_distance_m",
            False,
            "stability",
        ),

        (
            "median_cluster_diameter_m",
            False,
            "stability",
        ),

        (
            "scale_iqr",
            False,
            "stability",
        ),

        (
            "scale_range",
            False,
            "stability",
        ),

        (
            "rotation_iqr_deg",
            False,
            "stability",
        ),

        (
            "rotation_range_deg",
            False,
            "stability",
        ),

        (
            "median_center_residual_m",
            False,
            "geometry",
        ),

        (
            "median_hybrid_rank_sum",
            False,
            "retrieval",
        ),

        (
            "median_dino_rank_sum",
            False,
            "retrieval",
        ),
    ]


    # ========================================================
    # GT-best persistent track
    # ========================================================

    gt_best = (
        persistent
        .sort_values(
            gt_target
        )
        .iloc[0]
    )


    gt_best_track_id = int(
        gt_best[
            "track_id"
        ]
    )


    # ========================================================
    # Per-feature diagnostics
    # ========================================================

    feature_rows = []


    for (
        feature,
        higher_is_better,
        group_name,
    ) in feature_specs:

        pair = (
            persistent[
                [
                    feature,
                    gt_target,
                ]
            ]
            .dropna()
        )


        if len(pair) >= 3:

            spearman = float(
                pair[
                    feature
                ].corr(
                    pair[
                        gt_target
                    ],
                    method="spearman",
                )
            )

        else:

            spearman = math.nan


        ranks = desired_rank(
            persistent[
                feature
            ],
            higher_is_better,
        )


        gt_best_index = (
            persistent.index[
                persistent[
                    "track_id"
                ]
                == gt_best_track_id
            ][0]
        )


        gt_best_feature_rank = float(
            ranks.loc[
                gt_best_index
            ]
        )


        best_blind_index = int(
            ranks.idxmin()
        )


        best_blind = persistent.loc[
            best_blind_index
        ]


        feature_rows.append(
            {
                "feature":
                    feature,

                "group":
                    group_name,

                "desired_direction":
                    (
                        "higher"
                        if higher_is_better
                        else "lower"
                    ),

                "spearman_vs_gt_median_error":
                    spearman,

                "gt_best_track_id":
                    gt_best_track_id,

                "gt_best_feature_value":
                    float(
                        gt_best[
                            feature
                        ]
                    ),

                "gt_best_blind_rank":
                    gt_best_feature_rank,

                "gt_best_blind_percentile":
                    percentile_rank(
                        persistent[
                            feature
                        ],
                        float(
                            gt_best[
                                feature
                            ]
                        ),
                        higher_is_better,
                    ),

                "blind_best_track_id":
                    int(
                        best_blind[
                            "track_id"
                        ]
                    ),

                "blind_best_feature_value":
                    float(
                        best_blind[
                            feature
                        ]
                    ),

                "blind_best_track_gt_median_error_m":
                    float(
                        best_blind[
                            gt_target
                        ]
                    ),
            }
        )


    feature_audit = pd.DataFrame(
        feature_rows
    )


    # ========================================================
    # Compact Pareto audit
    #
    # No score. Six independent objectives.
    # ========================================================

    pareto_max = [
        "update_count",
        "support_per_update",
    ]


    pareto_min = [
        "median_association_distance_m",
        "median_cluster_diameter_m",
        "median_center_residual_m",
        "median_hybrid_rank_sum",
    ]


    pareto_input = (
        persistent[
            [
                "track_id",
                *pareto_max,
                *pareto_min,
            ]
        ]
        .dropna()
        .copy()
    )


    pareto_input[
        "blind_pareto"
    ] = pareto_front(
        pareto_input,
        pareto_max,
        pareto_min,
    )


    pareto_ids = set(
        pareto_input[
            pareto_input[
                "blind_pareto"
            ]
        ][
            "track_id"
        ].astype(int)
    )


    persistent[
        "blind_pareto"
    ] = persistent[
        "track_id"
    ].astype(int).isin(
        pareto_ids
    )


    # How many Pareto tracks are actually good post-freeze?
    pareto_tracks = (
        persistent[
            persistent[
                "blind_pareto"
            ]
        ]
        .sort_values(
            gt_target
        )
    )


    # ========================================================
    # Save post-freeze diagnosis
    # ========================================================

    analysis_path = (
        out_dir
        / "r4_7_gt_annotated_track_features.csv"
    )


    feature_audit_path = (
        out_dir
        / "r4_7_feature_discriminator_audit.csv"
    )


    report_path = (
        out_dir
        / "r4_7_blind_discriminator_audit.json"
    )


    analysis.to_csv(
        analysis_path,
        index=False,
    )


    feature_audit.to_csv(
        feature_audit_path,
        index=False,
    )


    report = {
        "stage":
            "R4.7_BLIND_OBSERVABLE_DISCRIMINATOR_AUDIT",

        "status":
            "PASS_R4_7_BLIND_DISCRIMINATOR_AUDIT_EXECUTION",

        "blind_feature_freeze_manifest_sha256":
            blind_manifest_sha,

        "persistent_subset_min_updates":
            min_updates,

        "persistent_track_count":
            int(
                len(
                    persistent
                )
            ),

        "postfreeze_gt_best_persistent_track": {
            "track_id":
                gt_best_track_id,

            "gt_median_disagreement_m":
                float(
                    gt_best[
                        gt_target
                    ]
                ),

            "update_count":
                int(
                    gt_best[
                        "update_count"
                    ]
                ),

            "scale_median":
                float(
                    gt_best[
                        "scale_median"
                    ]
                ),

            "rotation_median_deg":
                float(
                    gt_best[
                        "rotation_median_deg"
                    ]
                ),
        },

        "blind_pareto": {
            "objectives_maximize":
                pareto_max,

            "objectives_minimize":
                pareto_min,

            "pareto_track_count":
                int(
                    len(
                        pareto_tracks
                    )
                ),

            "gt_best_track_on_pareto":
                bool(
                    gt_best_track_id
                    in pareto_ids
                ),
        },

        "contract": {
            "phase_a_used_gt":
                False,

            "gt_attached_after_blind_feature_freeze":
                True,

            "weighted_score_created":
                False,

            "this_is_online_policy_selection":
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
    print("=" * 112)
    print(
        "R4.7 PHASE B — "
        "POST-FREEZE BLIND-DISCRIMINATOR AUDIT"
    )
    print("=" * 112)

    print(
        "persistent tracks:",
        len(
            persistent
        ),
    )

    print()

    print(
        "GT-best persistent track:"
    )

    print(
        "  track:",
        gt_best_track_id,
    )

    print(
        "  GT median disagreement:",
        f"{gt_best[gt_target]:.3f} m",
    )

    print(
        "  updates:",
        int(
            gt_best[
                "update_count"
            ]
        ),
    )

    print(
        "  scale median:",
        f"{gt_best['scale_median']:.6f}",
    )

    print(
        "  rotation median:",
        f"{gt_best['rotation_median_deg']:.3f} deg",
    )


    print()
    print("=" * 112)
    print("INDIVIDUAL BLIND FEATURE AUDIT")
    print("=" * 112)

    print(
        feature_audit[
            [
                "feature",
                "group",
                "desired_direction",
                "spearman_vs_gt_median_error",
                "gt_best_feature_value",
                "gt_best_blind_rank",
                "gt_best_blind_percentile",
                "blind_best_track_id",
                "blind_best_track_gt_median_error_m",
            ]
        ].to_string(
            index=False,
            float_format=lambda x:
                f"{x:.3f}",
        )
    )


    print()
    print("=" * 112)
    print("BLIND PARETO FRONT")
    print("=" * 112)

    print(
        "objectives maximize:",
        pareto_max,
    )

    print(
        "objectives minimize:",
        pareto_min,
    )

    print(
        "Pareto tracks:",
        len(
            pareto_tracks
        ),
    )

    print(
        "GT-best track on Pareto:",
        (
            gt_best_track_id
            in pareto_ids
        ),
    )

    print()

    show_cols = [
        "track_id",
        "update_count",
        "support_per_update",
        "median_association_distance_m",
        "median_cluster_diameter_m",
        "median_center_residual_m",
        "median_hybrid_rank_sum",
        "scale_median",
        "rotation_median_deg",
        "postfreeze_gt_median_disagreement_m",
    ]


    print(
        pareto_tracks[
            show_cols
        ]
        .head(30)
        .to_string(
            index=False,
            float_format=lambda x:
                f"{x:.3f}",
        )
    )


    print()
    print("=" * 112)
    print(
        "GT-BEST 20 PERSISTENT TRACKS "
        "WITH BLIND FEATURES"
    )
    print("=" * 112)

    print(
        persistent
        .sort_values(
            gt_target
        )[
            show_cols
            + [
                "blind_pareto"
            ]
        ]
        .head(20)
        .to_string(
            index=False,
            float_format=lambda x:
                f"{x:.3f}",
        )
    )


    print()
    print("=" * 112)
    print("R4.7 OUTPUT")
    print("=" * 112)

    print(
        "blind features:",
        blind_features_path,
    )

    print(
        "blind freeze manifest:",
        blind_manifest_path,
    )

    print(
        "GT annotated features:",
        analysis_path,
    )

    print(
        "feature audit:",
        feature_audit_path,
    )

    print(
        "report:",
        report_path,
    )

    print()

    print(
        "STATUS: "
        "PASS_R4_7_BLIND_DISCRIMINATOR_AUDIT_EXECUTION"
    )


if __name__ == "__main__":
    main()
