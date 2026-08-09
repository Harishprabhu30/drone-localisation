#!/usr/bin/env python3
"""
R4.4 — causal all-family tracker.

POST-FREEZE diagnostic.

PHASE A — BLIND
  - reads frozen R3 geometrically-admissible hypotheses
  - clusters ALL admissible transforms
  - associates clusters through time
  - freezes/hashes blind transform-family tracks

PHASE B — POST-FREEZE GT
  - only after Phase A has been frozen, reads reference
  - retrospectively labels the already-frozen tracks

This script does NOT:
  - change R3
  - change any threshold in R3
  - rerun DINO / ORB / XFeat
  - use GT for clustering or tracking
  - modify frozen blind outputs

Command:

SCRIPT=scripts/villoc/research/minimum_confident_bootstrap/diagnostics/r4_4_causal_family_tracker.py

RUN=outputs/demo_runs/traj01_blind_regression_001

R3=outputs/research_runs/minimum_confident_bootstrap/traj01_blind_r3_001

TILES=outputs/villoc/90_deg/metadata/s8_9_satellite_tile_index_512_s256.csv

python "$SCRIPT" \
  --run-root "$RUN" \
  --research-root "$R3" \
  --tile-index-csv "$TILES" \
  2>&1 | tee \
  "$R3/postfreeze_eval/r4_4_causal_family_tracker.log"

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
# Utilities
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

    z0 = (
        z
        - z.mean()
    )

    w0 = (
        w
        - w.mean()
    )

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


def model_from_row(
    row,
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


def prediction_distance(
    model_a: dict,
    model_b: dict,
    probes: np.ndarray,
) -> float:

    pa = apply_similarity(
        probes,
        model_a,
    )

    pb = apply_similarity(
        probes,
        model_b,
    )

    d = np.linalg.norm(
        pa - pb,
        axis=1,
    )

    return float(
        d.max()
    )


# ============================================================
# Map-center spacing
# ============================================================

def nominal_grid_spacing(
    tile_df: pd.DataFrame,
) -> tuple[float, dict]:

    required = {
        "center_easting",
        "center_northing",
    }

    missing = (
        required
        - set(tile_df.columns)
    )

    if missing:
        raise RuntimeError(
            "Tile index missing: "
            + str(
                sorted(missing)
            )
        )

    x = np.sort(
        pd.to_numeric(
            tile_df[
                "center_easting"
            ],
            errors="coerce",
        )
        .dropna()
        .round(3)
        .unique()
    )

    y = np.sort(
        pd.to_numeric(
            tile_df[
                "center_northing"
            ],
            errors="coerce",
        )
        .dropna()
        .round(3)
        .unique()
    )

    dx = np.diff(x)
    dy = np.diff(y)

    values = np.r_[
        dx[dx > 1e-6],
        dy[dy > 1e-6],
    ]

    if len(values) == 0:
        raise RuntimeError(
            "Could not derive map-center spacing."
        )

    rounded = pd.Series(
        np.round(
            values,
            3,
        )
    )

    counts = (
        rounded
        .value_counts()
    )

    spacing = float(
        counts.index[0]
    )

    return (
        spacing,
        {
            "unique_x_centers":
                int(len(x)),

            "unique_y_centers":
                int(len(y)),

            "positive_adjacent_differences":
                int(len(values)),

            "mode_spacing_m":
                spacing,

            "mode_count":
                int(counts.iloc[0]),
        },
    )


# ============================================================
# Per-update transform clustering
# ============================================================

def cluster_update(
    group: pd.DataFrame,
    probes: np.ndarray,
    threshold_m: float,
) -> list[dict]:
    """
    Complete-link-style deterministic greedy clustering.

    A transform can join a family only if it is within the
    family threshold of EVERY existing member.

    This prevents a chain of intermediate transforms from
    joining two transform modes that are actually farther
    apart than the family tolerance.

    Ordering is based only on predicted map positions at the
    two probes, not on center residual or GT.
    """

    items = []

    for _, row in group.iterrows():

        model = model_from_row(
            row
        )

        pred = (
            apply_similarity(
                probes,
                model,
            )
            .reshape(-1)
        )

        items.append(
            {
                "row":
                    row,

                "model":
                    model,

                "prediction":
                    pred,
            }
        )

    items.sort(
        key=lambda x:
            (
                *tuple(
                    np.round(
                        x["prediction"],
                        6,
                    )
                ),
                str(
                    x[
                        "row"
                    ][
                        "tile_ids"
                    ]
                ),
            )
    )


    # --------------------------------------------------------
    # Complete-link family formation
    # --------------------------------------------------------

    clusters = []

    for item in items:

        best_index = None
        best_mean_distance = None

        for i, cluster in enumerate(
            clusters
        ):

            distances = [
                prediction_distance(
                    item["model"],
                    member["model"],
                    probes,
                )
                for member
                in cluster["members"]
            ]

            complete_link = max(
                distances
            )

            if (
                complete_link
                <= threshold_m
            ):

                mean_distance = float(
                    np.mean(
                        distances
                    )
                )

                if (
                    best_mean_distance
                    is None
                    or mean_distance
                    < best_mean_distance
                ):
                    best_index = i
                    best_mean_distance = (
                        mean_distance
                    )

        if best_index is None:

            clusters.append(
                {
                    "members":
                        [item]
                }
            )

        else:

            clusters[
                best_index
            ][
                "members"
            ].append(
                item
            )


    # --------------------------------------------------------
    # Transform-space medoid representative
    # --------------------------------------------------------

    output = []

    for cluster in clusters:

        members = (
            cluster[
                "members"
            ]
        )

        n = len(
            members
        )

        if n == 1:

            medoid_index = 0

            diameter = 0.0

            mean_pair_distance = 0.0

        else:

            distances = np.zeros(
                (
                    n,
                    n,
                ),
                dtype=float,
            )

            for i in range(n):

                for j in range(
                    i + 1,
                    n,
                ):

                    d = prediction_distance(
                        members[i]["model"],
                        members[j]["model"],
                        probes,
                    )

                    distances[
                        i,
                        j,
                    ] = d

                    distances[
                        j,
                        i,
                    ] = d

            mean_per_member = (
                distances.mean(
                    axis=1
                )
            )

            medoid_index = int(
                np.argmin(
                    mean_per_member
                )
            )

            diameter = float(
                distances.max()
            )

            upper = distances[
                np.triu_indices(
                    n,
                    1,
                )
            ]

            mean_pair_distance = float(
                upper.mean()
            )

        representative = members[
            medoid_index
        ]

        rows = [
            x["row"]
            for x in members
        ]

        output.append(
            {
                "representative":
                    representative,

                "members":
                    members,

                "member_count":
                    int(n),

                "diameter_m":
                    diameter,

                "mean_pair_distance_m":
                    mean_pair_distance,

                "median_center_residual_m":
                    float(
                        np.median(
                            [
                                float(
                                    row[
                                        "median_center_residual_m"
                                    ]
                                )
                                for row
                                in rows
                            ]
                        )
                    ),

                "median_sum_hybrid_rank":
                    float(
                        np.median(
                            [
                                float(
                                    row[
                                        "sum_hybrid_rank"
                                    ]
                                )
                                for row
                                in rows
                            ]
                        )
                    ),

                "median_sum_dino_rank":
                    float(
                        np.median(
                            [
                                float(
                                    row[
                                        "sum_dino_rank"
                                    ]
                                )
                                for row
                                in rows
                            ]
                        )
                    ),

                "unique_tile_sequences":
                    int(
                        len(
                            set(
                                str(
                                    row[
                                        "tile_ids"
                                    ]
                                )
                                for row
                                in rows
                            )
                        )
                    ),
            }
        )


    # Largest modes shown first.
    output.sort(
        key=lambda x:
            (
                -x[
                    "member_count"
                ],

                x[
                    "diameter_m"
                ],

                str(
                    x[
                        "representative"
                    ][
                        "row"
                    ][
                        "tile_ids"
                    ]
                ),
            )
    )

    return output


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

    parser.add_argument(
        "--tile-index-csv",
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

    tile_index_path = (
        args.tile_index_csv
        .resolve()
    )

    out_dir = (
        research_root
        / "postfreeze_eval"
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    # ========================================================
    # Inputs available before GT
    # ========================================================

    hypothesis_path = (
        research_root
        / "hypothesis_updates.csv"
    )

    report_path = (
        research_root
        / "provisional_bootstrap_report.json"
    )

    relative_path = (
        run_root
        / "metadata/"
          "s8_xfeat_relative_frontend/"
          "s8r4_xfeat_relative_trajectory_blind_raw.csv"
    )


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
            "R4.4 expects frozen "
            "PROVISIONAL_ABSOLUTE_LOCK."
        )

    lock_q = int(
        report[
            "map_lock"
        ][
            "lock_query_id"
        ]
    )


    hypotheses = pd.read_csv(
        hypothesis_path
    )

    hypotheses[
        "update_query_id"
    ] = pd.to_numeric(
        hypotheses[
            "update_query_id"
        ],
        errors="raise",
    ).astype(int)

    hypotheses = hypotheses[
        hypotheses[
            "update_query_id"
        ]
        <= lock_q
    ].copy()

    if len(hypotheses) == 0:

        raise RuntimeError(
            "No stored admissible hypotheses."
        )


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
        relative[
            relative[
                "query_id"
            ]
            <= lock_q
        ]
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


    tile_index = pd.read_csv(
        tile_index_path
    )

    (
        center_spacing_m,
        spacing_diagnostic,
    ) = nominal_grid_spacing(
        tile_index
    )


    # --------------------------------------------------------
    # Family threshold is derived from GRID SPACING,
    # not from the 102.4 m tile width.
    #
    # 512_s256:
    # center spacing ≈ 51.2 m
    # family threshold ≈ 25.6 m
    #
    # This is an R4.4 diagnostic definition only.
    # --------------------------------------------------------

    family_threshold_m = (
        0.5
        * center_spacing_m
    )


    # ========================================================
    #
    # PHASE A
    #
    # BLIND ALL-FAMILY TRACKING
    #
    # IMPORTANT:
    # reference_attachment.csv has NOT been read.
    #
    # ========================================================

    cluster_rows = []

    tracks = {}

    next_track_id = 1

    previous_active = {}

    update_ids = sorted(
        hypotheses[
            "update_query_id"
        ].unique()
    )


    for update_q in update_ids:

        update_q = int(
            update_q
        )

        if (
            start_q
            not in relative_by_q.index
            or update_q
            not in relative_by_q.index
        ):

            raise RuntimeError(
                "Missing XFeat point for "
                f"q{update_q}."
            )


        # ----------------------------------------------------
        # Transform comparison probes:
        #   q1 + current causal query.
        #
        # Longer visual baseline naturally increases
        # sensitivity to scale/rotation disagreement.
        # ----------------------------------------------------

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


        group = hypotheses[
            hypotheses[
                "update_query_id"
            ]
            == update_q
        ].copy()


        clusters = cluster_update(
            group,
            probes,
            family_threshold_m,
        )


        current = {}

        for local_cluster_id, cluster in enumerate(
            clusters,
            1,
        ):

            representative = (
                cluster[
                    "representative"
                ]
            )

            current[
                local_cluster_id
            ] = {
                "cluster":
                    cluster,

                "model":
                    representative[
                        "model"
                    ],

                "row":
                    representative[
                        "row"
                    ],
            }


        # ----------------------------------------------------
        # Associate ONLY with previous evaluated update.
        #
        # One-to-one greedy assignment from smallest transform
        # disagreement.
        # ----------------------------------------------------

        association_candidates = []

        for (
            track_id,
            previous,
        ) in previous_active.items():

            for (
                local_cluster_id,
                current_cluster,
            ) in current.items():

                distance = prediction_distance(
                    previous[
                        "model"
                    ],

                    current_cluster[
                        "model"
                    ],

                    probes,
                )

                if (
                    distance
                    <= family_threshold_m
                ):

                    association_candidates.append(
                        (
                            distance,
                            int(track_id),
                            int(
                                local_cluster_id
                            ),
                        )
                    )


        association_candidates.sort(
            key=lambda x:
                (
                    x[0],
                    x[1],
                    x[2],
                )
        )


        used_tracks = set()
        used_clusters = set()

        associations = {}


        for (
            distance,
            track_id,
            local_cluster_id,
        ) in association_candidates:

            if (
                track_id
                in used_tracks
                or local_cluster_id
                in used_clusters
            ):
                continue

            associations[
                local_cluster_id
            ] = (
                track_id,
                float(distance),
                "MATCH_PREVIOUS_UPDATE",
            )

            used_tracks.add(
                track_id
            )

            used_clusters.add(
                local_cluster_id
            )


        # ----------------------------------------------------
        # Unmatched cluster starts new transform-family track.
        # ----------------------------------------------------

        for local_cluster_id in current:

            if (
                local_cluster_id
                not in associations
            ):

                track_id = (
                    next_track_id
                )

                next_track_id += 1

                associations[
                    local_cluster_id
                ] = (
                    track_id,
                    math.nan,
                    "NEW_TRACK",
                )


        next_active = {}


        for (
            local_cluster_id,
            current_cluster,
        ) in current.items():

            (
                track_id,
                association_distance_m,
                association_mode,
            ) = associations[
                local_cluster_id
            ]


            cluster = (
                current_cluster[
                    "cluster"
                ]
            )

            representative_row = (
                current_cluster[
                    "row"
                ]
            )

            model = (
                current_cluster[
                    "model"
                ]
            )


            if track_id not in tracks:

                tracks[
                    track_id
                ] = {
                    "track_id":
                        int(track_id),

                    "start_query_id":
                        update_q,

                    "end_query_id":
                        update_q,

                    "update_count":
                        0,

                    "queries":
                        [],

                    "member_counts":
                        [],

                    "diameters_m":
                        [],

                    "association_distances_m":
                        [],

                    "scales":
                        [],

                    "rotations_deg":
                        [],

                    "representative_tile_sequences":
                        [],
                }


            track = tracks[
                track_id
            ]

            track[
                "end_query_id"
            ] = update_q

            track[
                "update_count"
            ] += 1

            track[
                "queries"
            ].append(
                update_q
            )

            track[
                "member_counts"
            ].append(
                int(
                    cluster[
                        "member_count"
                    ]
                )
            )

            track[
                "diameters_m"
            ].append(
                float(
                    cluster[
                        "diameter_m"
                    ]
                )
            )

            if np.isfinite(
                association_distance_m
            ):

                track[
                    "association_distances_m"
                ].append(
                    float(
                        association_distance_m
                    )
                )

            track[
                "scales"
            ].append(
                float(
                    model[
                        "scale_m_per_visual_px"
                    ]
                )
            )

            track[
                "rotations_deg"
            ].append(
                float(
                    model[
                        "rotation_deg"
                    ]
                )
            )

            track[
                "representative_tile_sequences"
            ].append(
                str(
                    representative_row[
                        "tile_ids"
                    ]
                )
            )


            next_active[
                track_id
            ] = {
                "model":
                    model,

                "query_id":
                    update_q,

                "local_cluster_id":
                    int(
                        local_cluster_id
                    ),
            }


            cluster_rows.append(
                {
                    "update_query_id":
                        update_q,

                    "local_cluster_id":
                        int(
                            local_cluster_id
                        ),

                    "track_id":
                        int(
                            track_id
                        ),

                    "association_mode":
                        association_mode,

                    "association_distance_m":
                        association_distance_m,

                    "member_count":
                        int(
                            cluster[
                                "member_count"
                            ]
                        ),

                    "cluster_diameter_m":
                        float(
                            cluster[
                                "diameter_m"
                            ]
                        ),

                    "mean_pair_distance_m":
                        float(
                            cluster[
                                "mean_pair_distance_m"
                            ]
                        ),

                    "unique_tile_sequences":
                        int(
                            cluster[
                                "unique_tile_sequences"
                            ]
                        ),

                    "median_center_residual_m":
                        float(
                            cluster[
                                "median_center_residual_m"
                            ]
                        ),

                    "median_sum_hybrid_rank":
                        float(
                            cluster[
                                "median_sum_hybrid_rank"
                            ]
                        ),

                    "median_sum_dino_rank":
                        float(
                            cluster[
                                "median_sum_dino_rank"
                            ]
                        ),

                    "representative_tile_ids":
                        str(
                            representative_row[
                                "tile_ids"
                            ]
                        ),

                    "representative_candidate_choice_ranks":
                        str(
                            representative_row[
                                "candidate_choice_ranks"
                            ]
                        ),

                    "representative_center_residual_m":
                        float(
                            representative_row[
                                "median_center_residual_m"
                            ]
                        ),

                    "representative_sum_hybrid_rank":
                        float(
                            representative_row[
                                "sum_hybrid_rank"
                            ]
                        ),

                    "representative_sum_dino_rank":
                        float(
                            representative_row[
                                "sum_dino_rank"
                            ]
                        ),

                    "a_real":
                        float(
                            model[
                                "a_real"
                            ]
                        ),

                    "a_imag":
                        float(
                            model[
                                "a_imag"
                            ]
                        ),

                    "b_real":
                        float(
                            model[
                                "b_real"
                            ]
                        ),

                    "b_imag":
                        float(
                            model[
                                "b_imag"
                            ]
                        ),

                    "scale_m_per_visual_px":
                        float(
                            model[
                                "scale_m_per_visual_px"
                            ]
                        ),

                    "rotation_deg":
                        float(
                            model[
                                "rotation_deg"
                            ]
                        ),
                }
            )


        previous_active = (
            next_active
        )


    # ========================================================
    # Blind track summaries
    # ========================================================

    cluster_df = pd.DataFrame(
        cluster_rows
    )


    track_rows = []

    for (
        track_id,
        track,
    ) in tracks.items():

        queries = (
            track[
                "queries"
            ]
        )

        consecutive = all(
            (
                b - a
            )
            == 1
            for (
                a,
                b,
            )
            in zip(
                queries[:-1],
                queries[1:],
            )
        )

        reached_lock = (
            int(
                track[
                    "end_query_id"
                ]
            )
            == lock_q
        )

        association_distances = (
            track[
                "association_distances_m"
            ]
        )


        track_rows.append(
            {
                "track_id":
                    int(
                        track_id
                    ),

                "start_query_id":
                    int(
                        track[
                            "start_query_id"
                        ]
                    ),

                "end_query_id":
                    int(
                        track[
                            "end_query_id"
                        ]
                    ),

                "update_count":
                    int(
                        track[
                            "update_count"
                        ]
                    ),

                "consecutive":
                    bool(
                        consecutive
                    ),

                "reached_lock":
                    bool(
                        reached_lock
                    ),

                "median_member_count":
                    float(
                        np.median(
                            track[
                                "member_counts"
                            ]
                        )
                    ),

                "max_member_count":
                    int(
                        np.max(
                            track[
                                "member_counts"
                            ]
                        )
                    ),

                "median_cluster_diameter_m":
                    float(
                        np.median(
                            track[
                                "diameters_m"
                            ]
                        )
                    ),

                "max_cluster_diameter_m":
                    float(
                        np.max(
                            track[
                                "diameters_m"
                            ]
                        )
                    ),

                "median_association_distance_m":
                    (
                        float(
                            np.median(
                                association_distances
                            )
                        )
                        if association_distances
                        else math.nan
                    ),

                "max_association_distance_m":
                    (
                        float(
                            np.max(
                                association_distances
                            )
                        )
                        if association_distances
                        else math.nan
                    ),

                "scale_median":
                    float(
                        np.median(
                            track[
                                "scales"
                            ]
                        )
                    ),

                "scale_range":
                    float(
                        np.max(
                            track[
                                "scales"
                            ]
                        )
                        - np.min(
                            track[
                                "scales"
                            ]
                        )
                    ),

                "rotation_median_deg":
                    float(
                        np.median(
                            track[
                                "rotations_deg"
                            ]
                        )
                    ),

                "rotation_range_deg":
                    float(
                        np.max(
                            track[
                                "rotations_deg"
                            ]
                        )
                        - np.min(
                            track[
                                "rotations_deg"
                            ]
                        )
                    ),

                "query_ids":
                    ",".join(
                        map(
                            str,
                            queries,
                        )
                    ),

                "representative_tile_sequences":
                    " | ".join(
                        track[
                            "representative_tile_sequences"
                        ]
                    ),
            }
        )


    tracks_df = pd.DataFrame(
        track_rows
    )


    # --------------------------------------------------------
    # Diagnostic ranking only.
    #
    # This is NOT a proposed lock rule.
    #
    # Priority:
    #   1. survives to lock update
    #   2. longer persistence
    #   3. smaller temporal association motion
    # --------------------------------------------------------

    tracks_df = (
        tracks_df
        .sort_values(
            [
                "reached_lock",
                "update_count",
                "median_association_distance_m",
                "track_id",
            ],
            ascending=[
                False,
                False,
                True,
                True,
            ],
            na_position="last",
        )
        .reset_index(
            drop=True
        )
    )


    tracks_df[
        "blind_persistence_rank"
    ] = np.arange(
        1,
        len(tracks_df) + 1,
    )


    # ========================================================
    # Freeze Phase-A outputs BEFORE GT
    # ========================================================

    blind_clusters_path = (
        out_dir
        / "r4_4_blind_cluster_updates.csv"
    )

    blind_tracks_path = (
        out_dir
        / "r4_4_blind_family_tracks.csv"
    )

    blind_manifest_path = (
        out_dir
        / "r4_4_blind_family_freeze_manifest.json"
    )


    cluster_df.to_csv(
        blind_clusters_path,
        index=False,
    )

    tracks_df.to_csv(
        blind_tracks_path,
        index=False,
    )


    blind_manifest = {
        "stage":
            "R4.4_BLIND_FAMILY_TRACK_FREEZE",

        "lock_query_id":
            lock_q,

        "blind_inputs": {
            "hypothesis_updates_csv":
                str(
                    hypothesis_path
                ),

            "hypothesis_updates_sha256":
                sha256(
                    hypothesis_path
                ),

            "relative_trajectory_csv":
                str(
                    relative_path
                ),

            "relative_trajectory_sha256":
                sha256(
                    relative_path
                ),

            "tile_index_csv":
                str(
                    tile_index_path
                ),

            "tile_index_sha256":
                sha256(
                    tile_index_path
                ),
        },

        "configuration": {
            "map_center_spacing_m_derived":
                center_spacing_m,

            "family_threshold_m":
                family_threshold_m,

            "family_threshold_definition":
                (
                    "0.5 * nominal map-center "
                    "spacing"
                ),

            "within_update_clustering":
                (
                    "complete-link-style transform "
                    "prediction clustering"
                ),

            "cluster_representative":
                (
                    "transform-space medoid"
                ),

            "temporal_association":
                (
                    "one-to-one association with "
                    "previous evaluated update only"
                ),

            "blind_track_ranking":
                (
                    "diagnostic only: reached_lock, "
                    "update_count, temporal association distance"
                ),
        },

        "map_spacing_diagnostic":
            spacing_diagnostic,

        "blind_contract": {
            "reference_used":
                False,

            "oracle_used":
                False,

            "evaluation_error_used":
                False,

            "gt_used_for_clustering":
                False,

            "gt_used_for_tracking":
                False,
        },
    }


    blind_manifest[
        "blind_outputs"
    ] = {
        "cluster_updates_csv":
            str(
                blind_clusters_path
            ),

        "cluster_updates_sha256":
            sha256(
                blind_clusters_path
            ),

        "family_tracks_csv":
            str(
                blind_tracks_path
            ),

        "family_tracks_sha256":
            sha256(
                blind_tracks_path
            ),
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
    print("=" * 108)
    print("R4.4 PHASE A — BLIND FAMILY TRACKS FROZEN")
    print("=" * 108)

    print(
        "map-center spacing:",
        f"{center_spacing_m:.3f} m",
    )

    print(
        "family threshold:",
        f"{family_threshold_m:.3f} m",
    )

    print(
        "evaluated updates:",
        len(
            update_ids
        ),
    )

    print(
        "blind clusters:",
        len(
            cluster_df
        ),
    )

    print(
        "blind family tracks:",
        len(
            tracks_df
        ),
    )

    print(
        "tracks reaching q38:",
        int(
            tracks_df[
                "reached_lock"
            ].sum()
        ),
    )

    print(
        "blind freeze manifest SHA256:",
        blind_manifest_sha,
    )


    # ========================================================
    #
    # PHASE B
    #
    # POST-FREEZE GT LABEL ATTACHMENT
    #
    # This is the FIRST place reference is read.
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


    merged = (
        relative
        .merge(
            reference[
                [
                    "query_id",
                    "gt_easting",
                    "gt_northing",
                ]
            ],
            on="query_id",
            how="inner",
            validate="one_to_one",
        )
        .sort_values(
            "query_id"
        )
        .reset_index(
            drop=True
        )
    )


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
    # Attach GT disagreement to already-frozen cluster rows.
    # --------------------------------------------------------

    gt_disagreements = []


    for _, row in cluster_df.iterrows():

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


    annotated_clusters = (
        cluster_df.copy()
    )


    annotated_clusters[
        "postfreeze_gt_disagreement_m"
    ] = gt_disagreements


    # ========================================================
    # GT labels per frozen blind track
    # ========================================================

    annotated_track_rows = []

    grouped = (
        annotated_clusters
        .groupby(
            "track_id",
            sort=True,
        )
    )


    for _, track_row in tracks_df.iterrows():

        track_id = int(
            track_row[
                "track_id"
            ]
        )

        group = (
            grouped
            .get_group(
                track_id
            )
            .sort_values(
                "update_query_id"
            )
        )

        values = (
            group[
                "postfreeze_gt_disagreement_m"
            ]
            .to_numpy(float)
        )


        annotated_track_rows.append(
            {
                **track_row.to_dict(),

                "postfreeze_gt_best_disagreement_m":
                    float(
                        np.min(
                            values
                        )
                    ),

                "postfreeze_gt_median_disagreement_m":
                    float(
                        np.median(
                            values
                        )
                    ),

                "postfreeze_gt_final_disagreement_m":
                    float(
                        values[-1]
                    ),

                "postfreeze_gt_final_query_id":
                    int(
                        group[
                            "update_query_id"
                        ].iloc[-1]
                    ),
            }
        )


    annotated_tracks = pd.DataFrame(
        annotated_track_rows
    )


    annotated_tracks[
        "postfreeze_gt_rank_by_median"
    ] = (
        annotated_tracks[
            "postfreeze_gt_median_disagreement_m"
        ]
        .rank(
            method="min",
            ascending=True,
        )
        .astype(int)
    )


    # ========================================================
    # Save GT-annotated copies
    # ========================================================

    annotated_clusters_path = (
        out_dir
        / "r4_4_gt_annotated_cluster_updates.csv"
    )

    annotated_tracks_path = (
        out_dir
        / "r4_4_gt_annotated_family_tracks.csv"
    )

    report_out = (
        out_dir
        / "r4_4_causal_family_tracker.json"
    )


    annotated_clusters.to_csv(
        annotated_clusters_path,
        index=False,
    )

    annotated_tracks.to_csv(
        annotated_tracks_path,
        index=False,
    )


    # ========================================================
    # Diagnostic views
    # ========================================================

    top_blind = (
        annotated_tracks
        .sort_values(
            "blind_persistence_rank"
        )
        .head(15)
    )


    persistent = annotated_tracks[
        annotated_tracks[
            "update_count"
        ]
        >= 3
    ].copy()


    best_gt_persistent = (
        persistent
        .sort_values(
            [
                "postfreeze_gt_median_disagreement_m",
                "postfreeze_gt_final_disagreement_m",
            ]
        )
        .head(15)
    )


    result = {
        "stage":
            "R4.4_CAUSAL_ALL_FAMILY_TRACKER",

        "status":
            "PASS_R4_4_CAUSAL_FAMILY_TRACKER_EXECUTION",

        "blind_family_freeze_manifest_sha256":
            blind_manifest_sha,

        "configuration":
            blind_manifest[
                "configuration"
            ],

        "counts": {
            "evaluated_updates":
                int(
                    len(
                        update_ids
                    )
                ),

            "blind_clusters":
                int(
                    len(
                        cluster_df
                    )
                ),

            "blind_family_tracks":
                int(
                    len(
                        tracks_df
                    )
                ),

            "tracks_reaching_lock":
                int(
                    tracks_df[
                        "reached_lock"
                    ].sum()
                ),

            "tracks_with_at_least_3_updates":
                int(
                    (
                        tracks_df[
                            "update_count"
                        ]
                        >= 3
                    ).sum()
                ),
        },

        "gt_prefix_transform_postfreeze_only": {
            "scale_m_per_visual_px":
                gt_model[
                    "scale_m_per_visual_px"
                ],

            "rotation_deg":
                gt_model[
                    "rotation_deg"
                ],
        },

        "blind_top_persistence_tracks_postfreeze_labels":
            top_blind.to_dict(
                "records"
            ),

        "gt_best_persistent_tracks":
            best_gt_persistent.to_dict(
                "records"
            ),

        "contract": {
            "tracking_used_gt":
                False,

            "gt_loaded_only_after_blind_tracks_frozen":
                True,

            "r3_algorithm_modified":
                False,

            "r3_parameters_modified":
                False,

            "r3_blind_outputs_modified":
                False,
        },

        "outputs": {
            "blind_cluster_updates":
                str(
                    blind_clusters_path
                ),

            "blind_family_tracks":
                str(
                    blind_tracks_path
                ),

            "blind_family_freeze_manifest":
                str(
                    blind_manifest_path
                ),

            "gt_annotated_clusters":
                str(
                    annotated_clusters_path
                ),

            "gt_annotated_tracks":
                str(
                    annotated_tracks_path
                ),
        },
    }


    report_out.write_text(
        json.dumps(
            result,
            indent=2,
        )
    )


    # ========================================================
    # Print evidence
    # ========================================================

    print()
    print("=" * 108)
    print("R4.4 PHASE B — POST-FREEZE GT LABELS")
    print("=" * 108)

    print(
        "GT prefix scale:",
        f"{gt_model['scale_m_per_visual_px']:.6f}",
    )

    print(
        "GT prefix rotation:",
        f"{gt_model['rotation_deg']:.3f} deg",
    )


    show_columns = [
        "blind_persistence_rank",
        "track_id",
        "start_query_id",
        "end_query_id",
        "update_count",
        "reached_lock",
        "median_member_count",
        "median_association_distance_m",
        "scale_median",
        "rotation_median_deg",
        "postfreeze_gt_best_disagreement_m",
        "postfreeze_gt_median_disagreement_m",
        "postfreeze_gt_final_disagreement_m",
        "postfreeze_gt_rank_by_median",
    ]


    print()
    print("=" * 108)
    print(
        "TOP BLIND-PERSISTENCE TRACKS — "
        "GT LABELS ATTACHED AFTER FREEZE"
    )
    print("=" * 108)

    print(
        top_blind[
            show_columns
        ].to_string(
            index=False,
            float_format=lambda x:
                f"{x:.3f}",
        )
    )


    print()
    print("=" * 108)
    print(
        "GT-BEST TRACKS AMONG TRACKS "
        "WITH >=3 UPDATES"
    )
    print("=" * 108)

    print(
        best_gt_persistent[
            show_columns
        ].to_string(
            index=False,
            float_format=lambda x:
                f"{x:.3f}",
        )
    )


    print()
    print("=" * 108)
    print("R4.4 OUTPUT")
    print("=" * 108)

    print(
        "blind clusters:",
        blind_clusters_path,
    )

    print(
        "blind tracks:",
        blind_tracks_path,
    )

    print(
        "blind freeze manifest:",
        blind_manifest_path,
    )

    print(
        "GT annotated tracks:",
        annotated_tracks_path,
    )

    print(
        "report:",
        report_out,
    )

    print()

    print(
        "STATUS: "
        "PASS_R4_4_CAUSAL_FAMILY_TRACKER_EXECUTION"
    )


if __name__ == "__main__":
    main()
