#!/usr/bin/env python3
"""
R4.12 — sub-tile bootstrap geometry counterfactual.

Question
--------
If we keep the R3 bootstrap hypothesis machinery but replace each candidate
tile centre with its blind ORB-homography projected sub-tile observation,
do the previously observed Pareto/clustering failures disappear?

PHASE A — BLIND
----------------
Uses:
  * frozen R3 Top-4 candidate_evidence.csv
  * frozen R3 evidence-frame schedule
  * frozen XFeat relative trajectory
  * frozen R4.11 projected map observations

Keeps R3-style:
  * 4 evidence queries
  * Top-M=4 candidates/query
  * explicit Cartesian enumeration
  * >=3 unique tile IDs
  * visual span >=100 px
  * map span >=50 m
  * positive finite similarity scale
  * median residual <=51.2 m
  * max residual <=102.4 m
  * Pareto: residual, hybrid-rank sum, DINO-rank sum
  * transform clustering threshold =51.2 m

Only changed measurement:
    tile centre -> ORB projected EPSG:3346 point

No GT/reference is read.

PHASE B — POST-FREEZE
------------------------
After Phase-A outputs are hashed:
  * attach q1..q38 GT transform
  * measure which admissible / Pareto transforms are actually correct
  * inspect q38 in detail

This is a diagnostic only. It does not modify R3.

Command:

SCRIPT=scripts/villoc/research/minimum_confident_bootstrap/diagnostics/r4_12_subtile_bootstrap_geometry_counterfactual.py
RUN=outputs/demo_runs/traj01_blind_regression_001
R3=outputs/research_runs/minimum_confident_bootstrap/traj01_blind_r3_001


python "$SCRIPT" \
  --run-root "$RUN" \
  --research-root "$R3" \
  2>&1 | tee \
  "$R3/postfreeze_eval/r4_12_subtile_bootstrap_geometry_counterfactual.log"

"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer


VISUAL_SPAN_MIN_PX = 100.0
MAP_SPAN_MIN_M = 50.0
MEDIAN_RESIDUAL_MAX_M = 51.2
MAX_RESIDUAL_MAX_M = 102.4
CLUSTER_THRESHOLD_M = 51.2


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


def pairwise_span(points: np.ndarray) -> float:
    points = np.asarray(
        points,
        dtype=float,
    )

    best = 0.0

    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            best = max(
                best,
                float(
                    np.linalg.norm(
                        points[i]
                        - points[j]
                    )
                ),
            )

    return best


def farthest_pair(points: np.ndarray):
    points = np.asarray(
        points,
        dtype=float,
    )

    best = -1.0
    pair = (
        0,
        len(points) - 1,
    )

    for i in range(len(points)):
        for j in range(i + 1, len(points)):

            distance = float(
                np.linalg.norm(
                    points[i]
                    - points[j]
                )
            )

            if distance > best:
                best = distance
                pair = (
                    i,
                    j,
                )

    return pair


def fit_similarity(
    visual_xy: np.ndarray,
    map_xy: np.ndarray,
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
        return None

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

    scale = float(
        abs(a)
    )

    rotation = float(
        np.degrees(
            np.angle(a)
        )
    )

    if (
        not np.isfinite(scale)
        or scale <= 0.0
    ):
        return None

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
            scale,

        "rotation_deg":
            rotation,
    }


def apply_similarity(
    visual_xy: np.ndarray,
    model,
):
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


def model_from_row(row):
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


def pareto_mask(df: pd.DataFrame):
    """
    Minimise:
      median_projected_residual_m
      sum_hybrid_rank
      sum_dino_rank
    """

    values = df[
        [
            "median_projected_residual_m",
            "sum_hybrid_rank",
            "sum_dino_rank",
        ]
    ].to_numpy(float)

    keep = np.ones(
        len(df),
        dtype=bool,
    )

    for i in range(len(df)):

        for j in range(len(df)):

            if i == j:
                continue

            at_least_as_good = np.all(
                values[j]
                <= values[i]
            )

            strictly_better = np.any(
                values[j]
                < values[i]
            )

            if (
                at_least_as_good
                and strictly_better
            ):
                keep[i] = False
                break

    return keep


def cluster_models(
    pareto: pd.DataFrame,
    sentinels: np.ndarray,
):
    """
    Preserve the R3-style representative clustering idea.

    A hypothesis joins an existing cluster when its predictions
    at BOTH farthest visual evidence sentinels are <=51.2 m from
    that cluster representative.

    This intentionally retains the old permissive threshold so
    R4.12 isolates the map-observation change.
    """

    ordered = (
        pareto
        .sort_values(
            [
                "median_projected_residual_m",
                "sum_hybrid_rank",
                "sum_dino_rank",
                "candidate_choice_ranks",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    clusters = []

    for _, row in ordered.iterrows():

        model = model_from_row(
            row
        )

        prediction = apply_similarity(
            sentinels,
            model,
        )

        assigned = False

        for cluster in clusters:

            rep_prediction = (
                cluster[
                    "representative_prediction"
                ]
            )

            distances = np.linalg.norm(
                prediction
                - rep_prediction,
                axis=1,
            )

            if bool(
                np.all(
                    distances
                    <= CLUSTER_THRESHOLD_M
                )
            ):

                cluster[
                    "member_indices"
                ].append(
                    int(
                        row[
                            "hypothesis_id"
                        ]
                    )
                )

                cluster[
                    "member_count"
                ] += 1

                cluster[
                    "max_rep_disagreement_m"
                ] = max(
                    cluster[
                        "max_rep_disagreement_m"
                    ],
                    float(
                        distances.max()
                    ),
                )

                assigned = True
                break

        if not assigned:

            clusters.append(
                {
                    "representative_hypothesis_id":
                        int(
                            row[
                                "hypothesis_id"
                            ]
                        ),

                    "representative_prediction":
                        prediction,

                    "member_indices":
                        [
                            int(
                                row[
                                    "hypothesis_id"
                                ]
                            )
                        ],

                    "member_count":
                        1,

                    "max_rep_disagreement_m":
                        0.0,
                }
            )

    return clusters


def parse_query_ids(value):
    return [
        int(x.strip())
        for x in str(value).split(",")
        if x.strip()
    ]


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

    out_dir = (
        research
        / "postfreeze_eval"
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    # ========================================================
    # Blind inputs
    # ========================================================

    candidates_path = (
        research
        / "candidate_evidence.csv"
    )

    timeline_path = (
        research
        / "provisional_bootstrap_timeline.csv"
    )

    projection_path = (
        out_dir
        / "r4_11_blind_subtile_projection_pairs.csv"
    )

    projection_manifest_path = (
        out_dir
        / "r4_11_blind_subtile_projection_freeze_manifest.json"
    )

    relative_path = (
        run
        / "metadata/"
          "s8_xfeat_relative_frontend/"
          "s8r4_xfeat_relative_trajectory_blind_raw.csv"
    )


    required = [
        candidates_path,
        timeline_path,
        projection_path,
        projection_manifest_path,
        relative_path,
    ]

    missing = [
        str(p)
        for p in required
        if not p.exists()
    ]

    if missing:

        raise RuntimeError(
            "Missing blind inputs:\n"
            + "\n".join(
                missing
            )
        )


    # Verify R4.11 blind projection freeze.
    projection_manifest = json.loads(
        projection_manifest_path.read_text()
    )

    expected_projection_hash = (
        projection_manifest[
            "blind_output"
        ][
            "sha256"
        ]
    )

    actual_projection_hash = sha256(
        projection_path
    )

    if (
        expected_projection_hash
        != actual_projection_hash
    ):

        raise RuntimeError(
            "R4.11 projected-pair hash mismatch."
        )


    if not bool(
        projection_manifest[
            "reproduction_gate"
        ][
            "pass"
        ]
    ):

        raise RuntimeError(
            "R4.11 ORB reproduction gate did not pass."
        )


    candidate = pd.read_csv(
        candidates_path
    )

    timeline = pd.read_csv(
        timeline_path
    )

    projected = pd.read_csv(
        projection_path
    )

    relative = pd.read_csv(
        relative_path
    )


    candidate[
        "query_id"
    ] = pd.to_numeric(
        candidate[
            "query_id"
        ],
        errors="raise",
    ).astype(int)

    projected[
        "query_id"
    ] = pd.to_numeric(
        projected[
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


    # --------------------------------------------------------
    # Join the exact R3 Top-4 candidate pool to frozen R4.11
    # continuous map observations.
    # --------------------------------------------------------

    projection_keep = projected[
        [
            "query_id",
            "tile_id",
            "projected_easting",
            "projected_northing",
            "projected_inside_tile",
            "inlier_reprojection_rmse_px",
            "projective_denominator",
            "recomputed_homography_ok",
        ]
    ].copy()


    joined = candidate.merge(
        projection_keep,
        on=[
            "query_id",
            "tile_id",
        ],
        how="left",
        validate="one_to_one",
    )


    if joined[
        "projected_easting"
    ].isna().any():

        bad = joined[
            joined[
                "projected_easting"
            ].isna()
        ][
            [
                "query_id",
                "tile_id",
            ]
        ]

        raise RuntimeError(
            "Missing R4.11 projected observations "
            "for R3 candidates:\n"
            + bad.to_string(
                index=False
            )
        )


    # No new online gate in R4.12.
    # Outside-tile projections are RETAINED deliberately.
    if not bool(
        joined[
            "recomputed_homography_ok"
        ].all()
    ):

        raise RuntimeError(
            "Expected every R3 candidate to have "
            "a valid recomputed homography."
        )


    candidates_by_q = {
        int(q):
            g.sort_values(
                "candidate_choice_rank"
            ).to_dict(
                "records"
            )
        for q, g
        in joined.groupby(
            "query_id",
            sort=True,
        )
    }


    visual_by_q = (
        relative
        .set_index(
            "query_id"
        )
    )


    # ========================================================
    #
    # PHASE A — BLIND SUB-TILE HYPOTHESES
    #
    # ========================================================

    hypothesis_rows = []

    update_rows = []

    cluster_rows = []

    hypothesis_id = 1


    for _, timeline_row in (
        timeline
        .sort_values(
            "update_query_id"
        )
        .iterrows()
    ):

        update_q = int(
            timeline_row[
                "update_query_id"
            ]
        )

        evidence_q = parse_query_ids(
            timeline_row[
                "evidence_query_ids"
            ]
        )

        if len(
            evidence_q
        ) != 4:

            raise RuntimeError(
                f"q{update_q}: expected exactly "
                f"4 evidence queries, got {evidence_q}"
            )


        visual_xy = (
            visual_by_q
            .loc[
                evidence_q,
                [
                    "visual_x_px",
                    "visual_y_px",
                ],
            ]
            .to_numpy(float)
        )


        visual_span = pairwise_span(
            visual_xy
        )


        sentinel_i, sentinel_j = (
            farthest_pair(
                visual_xy
            )
        )

        sentinels = visual_xy[
            [
                sentinel_i,
                sentinel_j,
            ]
        ]


        candidate_lists = []

        for q in evidence_q:

            if q not in candidates_by_q:

                raise RuntimeError(
                    f"No candidates for q{q}."
                )

            q_candidates = candidates_by_q[
                q
            ]

            if len(
                q_candidates
            ) != 4:

                raise RuntimeError(
                    f"q{q}: expected Top-4, "
                    f"found {len(q_candidates)}."
                )

            candidate_lists.append(
                q_candidates
            )


        local_rows = []


        for combo in itertools.product(
            *candidate_lists
        ):

            tile_ids = [
                str(
                    c[
                        "tile_id"
                    ]
                )
                for c in combo
            ]


            choice_ranks = [
                int(
                    c[
                        "candidate_choice_rank"
                    ]
                )
                for c in combo
            ]


            dino_ranks = [
                int(
                    c[
                        "rank"
                    ]
                )
                for c in combo
            ]


            hybrid_ranks = [
                int(
                    c[
                        "hybrid_rank"
                    ]
                )
                for c in combo
            ]


            projected_xy = np.asarray(
                [
                    [
                        float(
                            c[
                                "projected_easting"
                            ]
                        ),
                        float(
                            c[
                                "projected_northing"
                            ]
                        ),
                    ]
                    for c in combo
                ],
                dtype=float,
            )


            unique_tile_count = len(
                set(
                    tile_ids
                )
            )


            map_span = pairwise_span(
                projected_xy
            )


            model = fit_similarity(
                visual_xy,
                projected_xy,
            )


            geometric_pass = False
            median_residual = math.nan
            max_residual = math.nan


            if model is not None:

                predicted = apply_similarity(
                    visual_xy,
                    model,
                )

                residual = np.linalg.norm(
                    predicted
                    - projected_xy,
                    axis=1,
                )

                median_residual = float(
                    np.median(
                        residual
                    )
                )

                max_residual = float(
                    np.max(
                        residual
                    )
                )


                geometric_pass = bool(
                    unique_tile_count
                    >= 3
                    and
                    visual_span
                    >= VISUAL_SPAN_MIN_PX
                    and
                    map_span
                    >= MAP_SPAN_MIN_M
                    and
                    median_residual
                    <= MEDIAN_RESIDUAL_MAX_M
                    and
                    max_residual
                    <= MAX_RESIDUAL_MAX_M
                    and
                    np.isfinite(
                        model[
                            "scale_m_per_visual_px"
                        ]
                    )
                    and
                    model[
                        "scale_m_per_visual_px"
                    ]
                    > 0.0
                )


            row = {
                "hypothesis_id":
                    hypothesis_id,

                "update_query_id":
                    update_q,

                "evidence_query_ids":
                    ",".join(
                        map(
                            str,
                            evidence_q,
                        )
                    ),

                "tile_ids":
                    ",".join(
                        tile_ids
                    ),

                "candidate_choice_ranks":
                    ",".join(
                        map(
                            str,
                            choice_ranks,
                        )
                    ),

                "dino_ranks":
                    ",".join(
                        map(
                            str,
                            dino_ranks,
                        )
                    ),

                "hybrid_ranks":
                    ",".join(
                        map(
                            str,
                            hybrid_ranks,
                        )
                    ),

                "sum_dino_rank":
                    float(
                        sum(
                            dino_ranks
                        )
                    ),

                "sum_hybrid_rank":
                    float(
                        sum(
                            hybrid_ranks
                        )
                    ),

                "unique_tile_count":
                    int(
                        unique_tile_count
                    ),

                "visual_span_px":
                    float(
                        visual_span
                    ),

                "projected_map_span_m":
                    float(
                        map_span
                    ),

                "median_projected_residual_m":
                    median_residual,

                "max_projected_residual_m":
                    max_residual,

                "all_projected_inside_tile":
                    bool(
                        all(
                            bool(
                                c[
                                    "projected_inside_tile"
                                ]
                            )
                            for c in combo
                        )
                    ),

                "mean_inlier_reprojection_rmse_px":
                    float(
                        np.mean(
                            [
                                float(
                                    c[
                                        "inlier_reprojection_rmse_px"
                                    ]
                                )
                                for c in combo
                            ]
                        )
                    ),

                "geometric_pass":
                    geometric_pass,

                "pareto":
                    False,
            }


            if model is not None:
                row.update(
                    model
                )

            else:
                row.update(
                    {
                        "a_real":
                            math.nan,

                        "a_imag":
                            math.nan,

                        "b_real":
                            math.nan,

                        "b_imag":
                            math.nan,

                        "scale_m_per_visual_px":
                            math.nan,

                        "rotation_deg":
                            math.nan,
                    }
                )


            local_rows.append(
                row
            )

            hypothesis_rows.append(
                row
            )

            hypothesis_id += 1


        local = pd.DataFrame(
            local_rows
        )


        admissible = local[
            local[
                "geometric_pass"
            ]
        ].copy()


        if len(
            admissible
        ):

            mask = pareto_mask(
                admissible
            )

            pareto_ids = set(
                admissible.loc[
                    mask,
                    "hypothesis_id",
                ].astype(int)
            )

        else:

            pareto_ids = set()


        # Update global stored rows.
        for row in hypothesis_rows:

            if (
                int(
                    row[
                        "update_query_id"
                    ]
                )
                == update_q
                and
                int(
                    row[
                        "hypothesis_id"
                    ]
                )
                in pareto_ids
            ):

                row[
                    "pareto"
                ] = True


        pareto = local[
            local[
                "hypothesis_id"
            ].isin(
                pareto_ids
            )
        ].copy()


        clusters = (
            cluster_models(
                pareto,
                sentinels,
            )
            if len(
                pareto
            )
            else []
        )


        for local_cluster_id, cluster in enumerate(
            clusters,
            1,
        ):

            cluster_rows.append(
                {
                    "update_query_id":
                        update_q,

                    "local_cluster_id":
                        local_cluster_id,

                    "representative_hypothesis_id":
                        cluster[
                            "representative_hypothesis_id"
                        ],

                    "member_count":
                        cluster[
                            "member_count"
                        ],

                    "max_rep_disagreement_m":
                        cluster[
                            "max_rep_disagreement_m"
                        ],

                    "member_hypothesis_ids":
                        ",".join(
                            map(
                                str,
                                cluster[
                                    "member_indices"
                                ],
                            )
                        ),
                }
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

                "enumerated_hypotheses":
                    int(
                        len(
                            local
                        )
                    ),

                "geometric_pass_hypotheses":
                    int(
                        len(
                            admissible
                        )
                    ),

                "pareto_hypotheses":
                    int(
                        len(
                            pareto
                        )
                    ),

                "transform_clusters":
                    int(
                        len(
                            clusters
                        )
                    ),

                "all_inside_geometric_pass":
                    int(
                        admissible[
                            "all_projected_inside_tile"
                        ].sum()
                    )
                    if len(
                        admissible
                    )
                    else 0,

                "best_blind_pareto_hypothesis_id":
                    (
                        int(
                            pareto
                            .sort_values(
                                [
                                    "median_projected_residual_m",
                                    "sum_hybrid_rank",
                                    "sum_dino_rank",
                                ]
                            )
                            .iloc[0][
                                "hypothesis_id"
                            ]
                        )
                        if len(
                            pareto
                        )
                        else None
                    ),
            }
        )


    hypotheses = pd.DataFrame(
        hypothesis_rows
    )


    updates = pd.DataFrame(
        update_rows
    )


    clusters_df = pd.DataFrame(
        cluster_rows
    )


    # ========================================================
    # Blind freeze
    # ========================================================

    hypotheses_path = (
        out_dir
        / "r4_12_blind_subtile_hypotheses.csv"
    )


    updates_path = (
        out_dir
        / "r4_12_blind_subtile_update_summary.csv"
    )


    clusters_path = (
        out_dir
        / "r4_12_blind_subtile_transform_clusters.csv"
    )


    freeze_manifest_path = (
        out_dir
        / "r4_12_blind_subtile_geometry_freeze_manifest.json"
    )


    hypotheses.to_csv(
        hypotheses_path,
        index=False,
    )


    updates.to_csv(
        updates_path,
        index=False,
    )


    clusters_df.to_csv(
        clusters_path,
        index=False,
    )


    freeze_manifest = {
        "stage":
            "R4.12_BLIND_SUBTILE_BOOTSTRAP_GEOMETRY_FREEZE",

        "single_changed_variable":
            (
                "map observation changed from "
                "satellite tile centre to frozen "
                "ORB projected sub-tile point"
            ),

        "configuration": {
            "evidence_queries":
                4,

            "candidates_per_query":
                4,

            "visual_span_min_px":
                VISUAL_SPAN_MIN_PX,

            "map_span_min_m":
                MAP_SPAN_MIN_M,

            "median_residual_max_m":
                MEDIAN_RESIDUAL_MAX_M,

            "max_residual_max_m":
                MAX_RESIDUAL_MAX_M,

            "cluster_threshold_m":
                CLUSTER_THRESHOLD_M,

            "outside_tile_projection_rejected":
                False,

            "pareto_objectives":
                [
                    "median projected residual",
                    "sum hybrid rank",
                    "sum DINO rank",
                ],
        },

        "inputs": {
            "r3_candidate_evidence_sha256":
                sha256(
                    candidates_path
                ),

            "r3_timeline_sha256":
                sha256(
                    timeline_path
                ),

            "r4_11_projection_sha256":
                actual_projection_hash,

            "relative_trajectory_sha256":
                sha256(
                    relative_path
                ),
        },

        "counts": {
            "updates":
                int(
                    len(
                        updates
                    )
                ),

            "hypotheses":
                int(
                    len(
                        hypotheses
                    )
                ),

            "geometric_pass":
                int(
                    hypotheses[
                        "geometric_pass"
                    ].sum()
                ),

            "pareto":
                int(
                    hypotheses[
                        "pareto"
                    ].sum()
                ),

            "transform_clusters":
                int(
                    len(
                        clusters_df
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

            "r3_lock_policy_reimplemented":
                False,
        },

        "outputs": {
            "hypotheses_csv":
                str(
                    hypotheses_path
                ),

            "hypotheses_sha256":
                sha256(
                    hypotheses_path
                ),

            "updates_csv":
                str(
                    updates_path
                ),

            "updates_sha256":
                sha256(
                    updates_path
                ),

            "clusters_csv":
                str(
                    clusters_path
                ),

            "clusters_sha256":
                sha256(
                    clusters_path
                ),
        },
    }


    freeze_manifest_path.write_text(
        json.dumps(
            freeze_manifest,
            indent=2,
        )
    )


    freeze_sha = sha256(
        freeze_manifest_path
    )


    print()
    print("=" * 112)
    print(
        "R4.12 PHASE A — "
        "BLIND SUB-TILE BOOTSTRAP GEOMETRY FROZEN"
    )
    print("=" * 112)

    print(
        "updates:",
        len(
            updates
        ),
    )

    print(
        "hypotheses:",
        len(
            hypotheses
        ),
    )

    print(
        "geometric pass:",
        int(
            hypotheses[
                "geometric_pass"
            ].sum()
        ),
    )

    print(
        "Pareto hypotheses:",
        int(
            hypotheses[
                "pareto"
            ].sum()
        ),
    )

    print(
        "transform clusters:",
        len(
            clusters_df
        ),
    )

    print(
        "blind freeze SHA256:",
        freeze_sha,
    )


    print()
    print("Last 10 blind updates")
    print("-" * 112)

    print(
        updates
        .tail(10)
        .to_string(
            index=False
        )
    )


    # ========================================================
    #
    # PHASE B — GT ATTACHMENT
    #
    # FIRST GT READ OCCURS HERE
    #
    # ========================================================

    reference_path = (
        run
        / "evaluation/"
          "reference_attachment.csv"
    )


    reference = pd.read_csv(
        reference_path
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


    lock_q = int(
        updates[
            "update_query_id"
        ].max()
    )


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


    if gt_model is None:
        raise RuntimeError(
            "Could not fit GT prefix transform."
        )


    common_probes = (
        prefix[
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


    gt_prediction = apply_similarity(
        common_probes,
        gt_model,
    )


    gt_disagreement = []


    for _, row in hypotheses.iterrows():

        if not bool(
            row[
                "geometric_pass"
            ]
        ):

            gt_disagreement.append(
                math.nan
            )

            continue


        pred = apply_similarity(
            common_probes,
            model_from_row(
                row
            ),
        )


        d = np.linalg.norm(
            pred
            - gt_prediction,
            axis=1,
        )


        gt_disagreement.append(
            float(
                d.max()
            )
        )


    annotated = hypotheses.copy()


    annotated[
        "postfreeze_gt_common_disagreement_m"
    ] = gt_disagreement


    evaluation_rows = []


    for update_q, group in annotated.groupby(
        "update_query_id",
        sort=True,
    ):

        admissible = group[
            group[
                "geometric_pass"
            ]
        ].copy()


        pareto = admissible[
            admissible[
                "pareto"
            ]
        ].copy()


        if len(
            admissible
        ):

            best_all = (
                admissible
                .sort_values(
                    "postfreeze_gt_common_disagreement_m"
                )
                .iloc[0]
            )

        else:

            best_all = None


        if len(
            pareto
        ):

            best_pareto = (
                pareto
                .sort_values(
                    "postfreeze_gt_common_disagreement_m"
                )
                .iloc[0]
            )


            blind_leader = (
                pareto
                .sort_values(
                    [
                        "median_projected_residual_m",
                        "sum_hybrid_rank",
                        "sum_dino_rank",
                    ]
                )
                .iloc[0]
            )

        else:

            best_pareto = None
            blind_leader = None


        evaluation_rows.append(
            {
                "update_query_id":
                    int(
                        update_q
                    ),

                "geometric_pass":
                    int(
                        len(
                            admissible
                        )
                    ),

                "pareto_count":
                    int(
                        len(
                            pareto
                        )
                    ),

                "best_admissible_gt_disagreement_m":
                    (
                        float(
                            best_all[
                                "postfreeze_gt_common_disagreement_m"
                            ]
                        )
                        if best_all is not None
                        else math.nan
                    ),

                "best_admissible_is_pareto":
                    (
                        bool(
                            best_all[
                                "pareto"
                            ]
                        )
                        if best_all is not None
                        else False
                    ),

                "best_pareto_gt_disagreement_m":
                    (
                        float(
                            best_pareto[
                                "postfreeze_gt_common_disagreement_m"
                            ]
                        )
                        if best_pareto is not None
                        else math.nan
                    ),

                "blind_leader_gt_disagreement_m":
                    (
                        float(
                            blind_leader[
                                "postfreeze_gt_common_disagreement_m"
                            ]
                        )
                        if blind_leader is not None
                        else math.nan
                    ),

                "blind_leader_scale":
                    (
                        float(
                            blind_leader[
                                "scale_m_per_visual_px"
                            ]
                        )
                        if blind_leader is not None
                        else math.nan
                    ),

                "blind_leader_rotation_deg":
                    (
                        float(
                            blind_leader[
                                "rotation_deg"
                            ]
                        )
                        if blind_leader is not None
                        else math.nan
                    ),

                "blind_leader_tile_ids":
                    (
                        str(
                            blind_leader[
                                "tile_ids"
                            ]
                        )
                        if blind_leader is not None
                        else ""
                    ),
            }
        )


    evaluation = pd.DataFrame(
        evaluation_rows
    )


    # ========================================================
    # q38 detail
    # ========================================================

    q38 = annotated[
        annotated[
            "update_query_id"
        ]
        == lock_q
    ].copy()


    q38_admissible = q38[
        q38[
            "geometric_pass"
        ]
    ].copy()


    q38_pareto = q38_admissible[
        q38_admissible[
            "pareto"
        ]
    ].copy()


    q38_detail = (
        q38_pareto
        .sort_values(
            [
                "median_projected_residual_m",
                "sum_hybrid_rank",
                "sum_dino_rank",
            ]
        )
    )


    annotated_path = (
        out_dir
        / "r4_12_gt_annotated_subtile_hypotheses.csv"
    )


    evaluation_path = (
        out_dir
        / "r4_12_postfreeze_subtile_update_evaluation.csv"
    )


    report_path = (
        out_dir
        / "r4_12_subtile_bootstrap_geometry_counterfactual.json"
    )


    annotated.to_csv(
        annotated_path,
        index=False,
    )


    evaluation.to_csv(
        evaluation_path,
        index=False,
    )


    report = {
        "stage":
            "R4.12_SUBTILE_BOOTSTRAP_GEOMETRY_COUNTERFACTUAL",

        "status":
            "PASS_R4_12_SUBTILE_BOOTSTRAP_GEOMETRY_EXECUTION",

        "blind_freeze_manifest_sha256":
            freeze_sha,

        "gt_prefix_transform_postfreeze_only": {
            "scale_m_per_visual_px":
                float(
                    gt_model[
                        "scale_m_per_visual_px"
                    ]
                ),

            "rotation_deg":
                float(
                    gt_model[
                        "rotation_deg"
                    ]
                ),
        },

        "q38": {
            "admissible":
                int(
                    len(
                        q38_admissible
                    )
                ),

            "pareto":
                int(
                    len(
                        q38_pareto
                    )
                ),

            "clusters":
                int(
                    updates[
                        updates[
                            "update_query_id"
                        ]
                        == lock_q
                    ][
                        "transform_clusters"
                    ].iloc[0]
                ),

            "best_pareto_gt_disagreement_m":
                (
                    float(
                        q38_pareto[
                            "postfreeze_gt_common_disagreement_m"
                        ].min()
                    )
                    if len(
                        q38_pareto
                    )
                    else None
                ),
        },

        "contract": {
            "phase_a_used_gt":
                False,

            "gt_loaded_after_blind_geometry_freeze":
                True,

            "only_measurement_model_changed":
                True,

            "r3_lock_policy_reimplemented":
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
    print("=" * 112)
    print(
        "R4.12 PHASE B — "
        "POST-FREEZE SUB-TILE BOOTSTRAP GEOMETRY"
    )
    print("=" * 112)


    print(
        "GT prefix scale:",
        f"{gt_model['scale_m_per_visual_px']:.6f}",
    )


    print(
        "GT prefix rotation:",
        f"{gt_model['rotation_deg']:.3f} deg",
    )


    print()
    print("Last 15 update evaluations")
    print("-" * 112)


    print(
        evaluation
        .tail(15)
        .to_string(
            index=False,
            float_format=lambda x:
                f"{x:.3f}",
        )
    )


    print()
    print("=" * 112)
    print(
        f"q{lock_q} SUB-TILE PARETO HYPOTHESES"
    )
    print("=" * 112)


    show = [
        "hypothesis_id",
        "tile_ids",
        "candidate_choice_ranks",
        "median_projected_residual_m",
        "max_projected_residual_m",
        "sum_hybrid_rank",
        "sum_dino_rank",
        "all_projected_inside_tile",
        "scale_m_per_visual_px",
        "rotation_deg",
        "postfreeze_gt_common_disagreement_m",
    ]


    if len(
        q38_detail
    ):

        print(
            q38_detail[
                show
            ].to_string(
                index=False,
                float_format=lambda x:
                    f"{x:.3f}",
            )
        )

    else:

        print(
            "No q38 Pareto hypotheses."
        )


    print()
    print("=" * 112)
    print("R4.12 OUTPUT")
    print("=" * 112)


    print(
        "blind hypotheses:",
        hypotheses_path,
    )


    print(
        "blind updates:",
        updates_path,
    )


    print(
        "blind clusters:",
        clusters_path,
    )


    print(
        "blind freeze manifest:",
        freeze_manifest_path,
    )


    print(
        "GT annotated hypotheses:",
        annotated_path,
    )


    print(
        "evaluation:",
        evaluation_path,
    )


    print(
        "report:",
        report_path,
    )


    print()


    print(
        "STATUS: "
        "PASS_R4_12_SUBTILE_BOOTSTRAP_GEOMETRY_EXECUTION"
    )


if __name__ == "__main__":
    main()
