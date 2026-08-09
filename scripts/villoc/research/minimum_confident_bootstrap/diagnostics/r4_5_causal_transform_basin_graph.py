#!/usr/bin/env python3
"""
R4.5 — causal transform-basin graph diagnostic.

Purpose
-------
R4.4 showed that useful transform families can split across multiple
one-to-one track IDs. This diagnostic keeps the already-frozen blind
R4.4 clusters but changes their temporal representation:

    cluster nodes
        +
    many-to-many temporal compatibility edges
        ->
    connected transform basins

PHASE A — BLIND
----------------
Uses only:
  * R4.4 frozen blind cluster outputs
  * XFeat blind relative trajectory
  * R4.4 blind family threshold
  * causal query order

It does NOT read reference / GT.

The blind basin graph is written and hashed before Phase B.

PHASE B — POST-FREEZE GT
------------------------
Only after the basin graph is frozen:
  * load attached traj01 reference
  * label the already-frozen basin nodes/basins
  * compare duration, continuity, support, spread, and correctness

This script does NOT modify R3 or R4.4.

command:

SCRIPT=scripts/villoc/research/minimum_confident_bootstrap/diagnostics/r4_5_causal_transform_basin_graph.py 

RUN=outputs/demo_runs/traj01_blind_regression_001  
R3=outputs/research_runs/minimum_confident_bootstrap/traj01_blind_r3_001  
python "$SCRIPT" \   
--run-root "$RUN" \   
--research-root "$R3" \   
2>&1 | tee \   "$R3/postfreeze_eval/r4_5_causal_transform_basin_graph.log"

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
# Basic helpers
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


def longest_consecutive_run(
    query_ids,
):
    q = sorted(
        set(
            int(x)
            for x in query_ids
        )
    )

    if not q:
        return (
            0,
            None,
            None,
        )

    best_len = 1
    best_start = q[0]
    best_end = q[0]

    start = q[0]
    prev = q[0]

    for x in q[1:]:

        if x == prev + 1:
            prev = x
            continue

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
# Union-find for temporal graph connected components
# ============================================================

class UnionFind:

    def __init__(self):
        self.parent = {}
        self.rank = {}

    def add(self, x):
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0

    def find(self, x):

        parent = self.parent[x]

        if parent != x:
            self.parent[x] = self.find(
                parent
            )

        return self.parent[x]

    def union(self, a, b):

        ra = self.find(a)
        rb = self.find(b)

        if ra == rb:
            return

        if (
            self.rank[ra]
            < self.rank[rb]
        ):
            ra, rb = rb, ra

        self.parent[rb] = ra

        if (
            self.rank[ra]
            == self.rank[rb]
        ):
            self.rank[ra] += 1


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

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    # ========================================================
    # R4.4 frozen BLIND inputs
    # ========================================================

    r44_clusters_path = (
        out_dir
        / "r4_4_blind_cluster_updates.csv"
    )

    r44_tracks_path = (
        out_dir
        / "r4_4_blind_family_tracks.csv"
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


    required_files = [
        r44_clusters_path,
        r44_tracks_path,
        r44_manifest_path,
        relative_path,
    ]

    missing = [
        str(x)
        for x in required_files
        if not x.exists()
    ]

    if missing:
        raise RuntimeError(
            "Required R4.4 inputs missing:\n"
            + "\n".join(missing)
        )


    # --------------------------------------------------------
    # Verify R4.4 blind freeze
    # --------------------------------------------------------

    r44_manifest = json.loads(
        r44_manifest_path.read_text()
    )


    expected_cluster_hash = (
        r44_manifest[
            "blind_outputs"
        ][
            "cluster_updates_sha256"
        ]
    )

    actual_cluster_hash = sha256(
        r44_clusters_path
    )


    if (
        expected_cluster_hash
        != actual_cluster_hash
    ):
        raise RuntimeError(
            "R4.4 blind cluster hash mismatch."
        )


    expected_track_hash = (
        r44_manifest[
            "blind_outputs"
        ][
            "family_tracks_sha256"
        ]
    )

    actual_track_hash = sha256(
        r44_tracks_path
    )


    if (
        expected_track_hash
        != actual_track_hash
    ):
        raise RuntimeError(
            "R4.4 blind track hash mismatch."
        )


    family_threshold_m = float(
        r44_manifest[
            "configuration"
        ][
            "family_threshold_m"
        ]
    )


    # ========================================================
    # Load blind cluster nodes
    # ========================================================

    nodes = pd.read_csv(
        r44_clusters_path
    )


    nodes[
        "update_query_id"
    ] = pd.to_numeric(
        nodes[
            "update_query_id"
        ],
        errors="raise",
    ).astype(int)


    # Give every frozen cluster a stable node ID.
    nodes = (
        nodes
        .sort_values(
            [
                "update_query_id",
                "local_cluster_id",
            ]
        )
        .reset_index(
            drop=True
        )
    )


    nodes[
        "node_id"
    ] = [
        f"N{i:05d}"
        for i
        in range(
            1,
            len(nodes) + 1,
        )
    ]


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


    update_ids = sorted(
        nodes[
            "update_query_id"
        ].unique()
    )


    lock_q = int(
        max(
            update_ids
        )
    )


    # ========================================================
    #
    # PHASE A — BLIND TEMPORAL BASIN GRAPH
    #
    # Reference has NOT been loaded.
    #
    # ========================================================

    uf = UnionFind()

    for node_id in nodes[
        "node_id"
    ]:
        uf.add(
            node_id
        )


    edge_rows = []

    incoming = {
        node_id: 0
        for node_id
        in nodes[
            "node_id"
        ]
    }

    outgoing = {
        node_id: 0
        for node_id
        in nodes[
            "node_id"
        ]
    }


    nodes_by_update = {
        int(q):
            group.copy()
        for q, group
        in nodes.groupby(
            "update_query_id"
        )
    }


    # --------------------------------------------------------
    # Connect every compatible pair across adjacent evaluated
    # updates.
    #
    # No one-to-one restriction.
    #
    # A cluster can therefore:
    #   split 1 -> many
    #   merge many -> 1
    # --------------------------------------------------------

    for (
        previous_q,
        current_q,
    ) in zip(
        update_ids[:-1],
        update_ids[1:],
    ):

        previous_q = int(
            previous_q
        )

        current_q = int(
            current_q
        )


        if (
            start_q
            not in relative_by_q.index
            or current_q
            not in relative_by_q.index
        ):
            raise RuntimeError(
                "Missing XFeat probes for "
                f"q{current_q}."
            )


        probes = (
            relative_by_q
            .loc[
                [
                    start_q,
                    current_q,
                ],
                [
                    "visual_x_px",
                    "visual_y_px",
                ],
            ]
            .to_numpy(float)
        )


        previous_nodes = (
            nodes_by_update[
                previous_q
            ]
        )

        current_nodes = (
            nodes_by_update[
                current_q
            ]
        )


        for _, prev_row in (
            previous_nodes.iterrows()
        ):

            prev_model = model_from_row(
                prev_row
            )


            for _, curr_row in (
                current_nodes.iterrows()
            ):

                curr_model = model_from_row(
                    curr_row
                )


                distance = prediction_distance(
                    prev_model,
                    curr_model,
                    probes,
                )


                if (
                    distance
                    <= family_threshold_m
                ):

                    prev_id = str(
                        prev_row[
                            "node_id"
                        ]
                    )

                    curr_id = str(
                        curr_row[
                            "node_id"
                        ]
                    )


                    uf.union(
                        prev_id,
                        curr_id,
                    )


                    outgoing[
                        prev_id
                    ] += 1

                    incoming[
                        curr_id
                    ] += 1


                    edge_rows.append(
                        {
                            "from_node_id":
                                prev_id,

                            "to_node_id":
                                curr_id,

                            "from_query_id":
                                previous_q,

                            "to_query_id":
                                current_q,

                            "query_gap":
                                int(
                                    current_q
                                    - previous_q
                                ),

                            "prediction_distance_m":
                                float(
                                    distance
                                ),
                        }
                    )


    edges = pd.DataFrame(
        edge_rows
    )


    # ========================================================
    # Connected components -> blind transform basins
    # ========================================================

    root_to_nodes = {}

    for node_id in nodes[
        "node_id"
    ]:

        root = uf.find(
            node_id
        )

        root_to_nodes.setdefault(
            root,
            [],
        ).append(
            node_id
        )


    # Deterministic basin ordering:
    # earliest query, then largest node support.
    component_info = []


    node_indexed = (
        nodes
        .set_index(
            "node_id",
            drop=False,
        )
    )


    for root, node_ids in (
        root_to_nodes.items()
    ):

        component = (
            node_indexed
            .loc[
                node_ids
            ]
            .copy()
        )

        component_info.append(
            {
                "root":
                    root,

                "node_ids":
                    node_ids,

                "first_query":
                    int(
                        component[
                            "update_query_id"
                        ].min()
                    ),

                "node_count":
                    int(
                        len(component)
                    ),
            }
        )


    component_info.sort(
        key=lambda x:
            (
                x[
                    "first_query"
                ],

                -x[
                    "node_count"
                ],

                str(
                    x[
                        "root"
                    ]
                ),
            )
    )


    root_to_basin = {
        item[
            "root"
        ]:
            f"B{i:04d}"
        for i, item
        in enumerate(
            component_info,
            1,
        )
    }


    node_to_basin = {}

    for node_id in nodes[
        "node_id"
    ]:

        node_to_basin[
            node_id
        ] = root_to_basin[
            uf.find(
                node_id
            )
        ]


    nodes[
        "basin_id"
    ] = nodes[
        "node_id"
    ].map(
        node_to_basin
    )


    nodes[
        "incoming_edges"
    ] = nodes[
        "node_id"
    ].map(
        incoming
    )


    nodes[
        "outgoing_edges"
    ] = nodes[
        "node_id"
    ].map(
        outgoing
    )


    if len(edges):

        edges[
            "basin_id"
        ] = edges[
            "from_node_id"
        ].map(
            node_to_basin
        )


    # ========================================================
    # Common blind probe space q1 / q38
    #
    # Used only to DESCRIBE basin spread.
    # Still no GT.
    # ========================================================

    common_probes = (
        relative_by_q
        .loc[
            [
                start_q,
                lock_q,
            ],
            [
                "visual_x_px",
                "visual_y_px",
            ],
        ]
        .to_numpy(float)
    )


    prediction_vectors = []


    for _, row in nodes.iterrows():

        prediction = (
            apply_similarity(
                common_probes,
                model_from_row(row),
            )
            .reshape(-1)
        )

        prediction_vectors.append(
            prediction
        )


    prediction_vectors = np.asarray(
        prediction_vectors,
        dtype=float,
    )


    for i in range(
        prediction_vectors.shape[1]
    ):

        nodes[
            f"common_prediction_{i}"
        ] = prediction_vectors[
            :,
            i,
        ]


    # ========================================================
    # Blind basin summaries
    # ========================================================

    basin_rows = []


    for basin_id, group in (
        nodes.groupby(
            "basin_id",
            sort=True,
        )
    ):

        group = (
            group
            .sort_values(
                [
                    "update_query_id",
                    "local_cluster_id",
                ]
            )
            .copy()
        )


        query_ids = sorted(
            group[
                "update_query_id"
            ].unique()
        )


        first_q = int(
            min(
                query_ids
            )
        )

        last_q = int(
            max(
                query_ids
            )
        )


        (
            longest_run,
            longest_start,
            longest_end,
        ) = longest_consecutive_run(
            query_ids
        )


        span = (
            last_q
            - first_q
            + 1
        )


        occupancy_fraction = (
            len(
                query_ids
            )
            / span
        )


        nodes_per_update = (
            group
            .groupby(
                "update_query_id"
            )
            .size()
            .to_numpy()
        )


        incoming_values = (
            group[
                "incoming_edges"
            ]
            .to_numpy(int)
        )

        outgoing_values = (
            group[
                "outgoing_edges"
            ]
            .to_numpy(int)
        )


        split_nodes = int(
            np.sum(
                outgoing_values
                > 1
            )
        )

        merge_nodes = int(
            np.sum(
                incoming_values
                > 1
            )
        )


        if len(edges):

            basin_edges = edges[
                edges[
                    "basin_id"
                ]
                == basin_id
            ]

            edge_count = int(
                len(
                    basin_edges
                )
            )

            median_edge_distance = (
                float(
                    basin_edges[
                        "prediction_distance_m"
                    ].median()
                )
                if len(
                    basin_edges
                )
                else math.nan
            )

            max_edge_distance = (
                float(
                    basin_edges[
                        "prediction_distance_m"
                    ].max()
                )
                if len(
                    basin_edges
                )
                else math.nan
            )

        else:

            edge_count = 0
            median_edge_distance = (
                math.nan
            )
            max_edge_distance = (
                math.nan
            )


        prediction_cols = [
            f"common_prediction_{i}"
            for i in range(4)
        ]


        pred = group[
            prediction_cols
        ].to_numpy(float)


        # Robust center in common transform-prediction space.
        pred_center = np.median(
            pred,
            axis=0,
        )


        pred_distance = np.max(
            np.column_stack(
                [
                    np.linalg.norm(
                        pred[:, 0:2]
                        - pred_center[
                            None,
                            0:2,
                        ],
                        axis=1,
                    ),

                    np.linalg.norm(
                        pred[:, 2:4]
                        - pred_center[
                            None,
                            2:4,
                        ],
                        axis=1,
                    ),
                ]
            ),
            axis=1,
        )


        # Representative node = nearest transform to robust
        # common-prediction center.
        representative_index = int(
            np.argmin(
                pred_distance
            )
        )


        representative = (
            group.iloc[
                representative_index
            ]
        )


        basin_rows.append(
            {
                "basin_id":
                    basin_id,

                "first_query_id":
                    first_q,

                "last_query_id":
                    last_q,

                "update_count":
                    int(
                        len(
                            query_ids
                        )
                    ),

                "query_span":
                    int(
                        span
                    ),

                "occupancy_fraction":
                    float(
                        occupancy_fraction
                    ),

                "longest_consecutive_updates":
                    int(
                        longest_run
                    ),

                "longest_streak_start":
                    longest_start,

                "longest_streak_end":
                    longest_end,

                "reached_lock":
                    bool(
                        lock_q
                        in query_ids
                    ),

                "node_count":
                    int(
                        len(
                            group
                        )
                    ),

                "edge_count":
                    edge_count,

                "median_nodes_per_update":
                    float(
                        np.median(
                            nodes_per_update
                        )
                    ),

                "max_nodes_per_update":
                    int(
                        np.max(
                            nodes_per_update
                        )
                    ),

                "split_node_count":
                    split_nodes,

                "merge_node_count":
                    merge_nodes,

                "median_edge_distance_m":
                    median_edge_distance,

                "max_edge_distance_m":
                    max_edge_distance,

                "total_hypothesis_support":
                    int(
                        group[
                            "member_count"
                        ].sum()
                    ),

                "median_cluster_member_count":
                    float(
                        group[
                            "member_count"
                        ].median()
                    ),

                "distinct_representative_tile_sequences":
                    int(
                        group[
                            "representative_tile_ids"
                        ].nunique()
                    ),

                "median_center_residual_m":
                    float(
                        group[
                            "median_center_residual_m"
                        ].median()
                    ),

                "median_hybrid_rank_sum":
                    float(
                        group[
                            "median_sum_hybrid_rank"
                        ].median()
                    ),

                "median_dino_rank_sum":
                    float(
                        group[
                            "median_sum_dino_rank"
                        ].median()
                    ),

                "scale_median":
                    float(
                        group[
                            "scale_m_per_visual_px"
                        ].median()
                    ),

                "scale_min":
                    float(
                        group[
                            "scale_m_per_visual_px"
                        ].min()
                    ),

                "scale_max":
                    float(
                        group[
                            "scale_m_per_visual_px"
                        ].max()
                    ),

                "rotation_median_deg":
                    float(
                        group[
                            "rotation_deg"
                        ].median()
                    ),

                "rotation_min_deg":
                    float(
                        group[
                            "rotation_deg"
                        ].min()
                    ),

                "rotation_max_deg":
                    float(
                        group[
                            "rotation_deg"
                        ].max()
                    ),

                "common_prediction_spread_median_m":
                    float(
                        np.median(
                            pred_distance
                        )
                    ),

                "common_prediction_spread_max_m":
                    float(
                        np.max(
                            pred_distance
                        )
                    ),

                "representative_node_id":
                    str(
                        representative[
                            "node_id"
                        ]
                    ),

                "representative_query_id":
                    int(
                        representative[
                            "update_query_id"
                        ]
                    ),

                "representative_tile_ids":
                    str(
                        representative[
                            "representative_tile_ids"
                        ]
                    ),

                "representative_a_real":
                    float(
                        representative[
                            "a_real"
                        ]
                    ),

                "representative_a_imag":
                    float(
                        representative[
                            "a_imag"
                        ]
                    ),

                "representative_b_real":
                    float(
                        representative[
                            "b_real"
                        ]
                    ),

                "representative_b_imag":
                    float(
                        representative[
                            "b_imag"
                        ]
                    ),

                "representative_scale":
                    float(
                        representative[
                            "scale_m_per_visual_px"
                        ]
                    ),

                "representative_rotation_deg":
                    float(
                        representative[
                            "rotation_deg"
                        ]
                    ),

                "query_ids":
                    ",".join(
                        map(
                            str,
                            query_ids,
                        )
                    ),
            }
        )


    basins = pd.DataFrame(
        basin_rows
    )


    # --------------------------------------------------------
    # Neutral blind diagnostic ordering.
    #
    # This is NOT a lock policy.
    #
    # Unlike R4.4, reaching q38 does not dominate.
    # --------------------------------------------------------

    basins = (
        basins
        .sort_values(
            [
                "update_count",
                "longest_consecutive_updates",
                "occupancy_fraction",
                "total_hypothesis_support",
                "common_prediction_spread_median_m",
                "basin_id",
            ],
            ascending=[
                False,
                False,
                False,
                False,
                True,
                True,
            ],
        )
        .reset_index(
            drop=True
        )
    )


    basins[
        "blind_duration_rank"
    ] = np.arange(
        1,
        len(
            basins
        ) + 1,
    )


    # ========================================================
    # Freeze the basin graph BEFORE GT
    # ========================================================

    blind_nodes_path = (
        out_dir
        / "r4_5_blind_basin_nodes.csv"
    )

    blind_edges_path = (
        out_dir
        / "r4_5_blind_basin_edges.csv"
    )

    blind_basins_path = (
        out_dir
        / "r4_5_blind_basins.csv"
    )

    blind_manifest_path = (
        out_dir
        / "r4_5_blind_basin_freeze_manifest.json"
    )


    nodes.to_csv(
        blind_nodes_path,
        index=False,
    )

    edges.to_csv(
        blind_edges_path,
        index=False,
    )

    basins.to_csv(
        blind_basins_path,
        index=False,
    )


    blind_manifest = {
        "stage":
            "R4.5_BLIND_TRANSFORM_BASIN_GRAPH_FREEZE",

        "r4_4_blind_manifest_sha256":
            sha256(
                r44_manifest_path
            ),

        "configuration": {
            "family_threshold_m":
                family_threshold_m,

            "node_definition":
                (
                    "one frozen R4.4 blind transform "
                    "cluster at one update"
                ),

            "edge_definition":
                (
                    "all compatible node pairs across "
                    "adjacent evaluated updates"
                ),

            "edge_transform_distance":
                (
                    "maximum prediction disagreement "
                    "at q1 and current query"
                ),

            "edge_directionality":
                "temporal",

            "basin_definition":
                (
                    "connected component of temporal "
                    "compatibility graph"
                ),

            "one_to_one_assignment":
                False,

            "splits_allowed":
                True,

            "merges_allowed":
                True,

            "blind_duration_rank":
                (
                    "diagnostic only; update_count, "
                    "continuous occupancy, support, spread"
                ),
        },

        "counts": {
            "blind_cluster_nodes":
                int(
                    len(
                        nodes
                    )
                ),

            "blind_temporal_edges":
                int(
                    len(
                        edges
                    )
                ),

            "blind_basins":
                int(
                    len(
                        basins
                    )
                ),

            "basins_reaching_lock":
                int(
                    basins[
                        "reached_lock"
                    ].sum()
                ),
        },

        "blind_contract": {
            "reference_used":
                False,

            "gt_used":
                False,

            "oracle_used":
                False,

            "evaluation_error_used":
                False,

            "r3_modified":
                False,

            "r4_4_modified":
                False,
        },
    }


    blind_manifest[
        "outputs"
    ] = {
        "nodes_csv":
            str(
                blind_nodes_path
            ),

        "nodes_sha256":
            sha256(
                blind_nodes_path
            ),

        "edges_csv":
            str(
                blind_edges_path
            ),

        "edges_sha256":
            sha256(
                blind_edges_path
            ),

        "basins_csv":
            str(
                blind_basins_path
            ),

        "basins_sha256":
            sha256(
                blind_basins_path
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
    print("=" * 110)
    print("R4.5 PHASE A — BLIND TRANSFORM BASIN GRAPH FROZEN")
    print("=" * 110)

    print(
        "R4.4 cluster hash verified:",
        actual_cluster_hash,
    )

    print(
        "family threshold:",
        f"{family_threshold_m:.3f} m",
    )

    print(
        "blind nodes:",
        len(
            nodes
        ),
    )

    print(
        "blind temporal edges:",
        len(
            edges
        ),
    )

    print(
        "blind basins:",
        len(
            basins
        ),
    )

    print(
        "basins reaching q38:",
        int(
            basins[
                "reached_lock"
            ].sum()
        ),
    )

    print(
        "blind basin freeze SHA256:",
        blind_manifest_sha,
    )


    # ========================================================
    #
    # PHASE B — POST-FREEZE GT LABELS
    #
    # FIRST GT READ OCCURS HERE.
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
            "Reference missing columns: "
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
    # Common q1/q38 GT comparison for every already-frozen node
    # --------------------------------------------------------

    gt_common_prediction = (
        apply_similarity(
            common_probes,
            gt_model,
        )
    )


    node_gt_disagreement = []


    for _, row in nodes.iterrows():

        prediction = apply_similarity(
            common_probes,
            model_from_row(row),
        )


        disagreement = np.linalg.norm(
            prediction
            - gt_common_prediction,
            axis=1,
        )


        node_gt_disagreement.append(
            float(
                disagreement.max()
            )
        )


    annotated_nodes = (
        nodes.copy()
    )


    annotated_nodes[
        "postfreeze_gt_common_disagreement_m"
    ] = node_gt_disagreement


    # ========================================================
    # Post-freeze basin GT labels
    # ========================================================

    annotated_basin_rows = []


    for _, basin_row in (
        basins.iterrows()
    ):

        basin_id = str(
            basin_row[
                "basin_id"
            ]
        )


        group = (
            annotated_nodes[
                annotated_nodes[
                    "basin_id"
                ]
                == basin_id
            ]
            .sort_values(
                [
                    "update_query_id",
                    "local_cluster_id",
                ]
            )
        )


        values = (
            group[
                "postfreeze_gt_common_disagreement_m"
            ]
            .to_numpy(float)
        )


        final_q = int(
            group[
                "update_query_id"
            ].max()
        )


        final_group = group[
            group[
                "update_query_id"
            ]
            == final_q
        ]


        representative_node_id = str(
            basin_row[
                "representative_node_id"
            ]
        )


        representative_group = group[
            group[
                "node_id"
            ]
            == representative_node_id
        ]


        if len(
            representative_group
        ) != 1:
            raise RuntimeError(
                "Could not locate frozen "
                f"representative {representative_node_id}."
            )


        representative_gt = float(
            representative_group[
                "postfreeze_gt_common_disagreement_m"
            ].iloc[0]
        )


        annotated_basin_rows.append(
            {
                **basin_row.to_dict(),

                "postfreeze_gt_best_node_disagreement_m":
                    float(
                        np.min(
                            values
                        )
                    ),

                "postfreeze_gt_median_node_disagreement_m":
                    float(
                        np.median(
                            values
                        )
                    ),

                "postfreeze_gt_p95_node_disagreement_m":
                    float(
                        np.percentile(
                            values,
                            95,
                        )
                    ),

                "postfreeze_gt_representative_disagreement_m":
                    representative_gt,

                "postfreeze_gt_final_query_id":
                    final_q,

                "postfreeze_gt_final_best_disagreement_m":
                    float(
                        final_group[
                            "postfreeze_gt_common_disagreement_m"
                        ].min()
                    ),

                "postfreeze_gt_final_median_disagreement_m":
                    float(
                        final_group[
                            "postfreeze_gt_common_disagreement_m"
                        ].median()
                    ),
            }
        )


    annotated_basins = pd.DataFrame(
        annotated_basin_rows
    )


    annotated_basins[
        "postfreeze_gt_rank_by_median"
    ] = (
        annotated_basins[
            "postfreeze_gt_median_node_disagreement_m"
        ]
        .rank(
            method="min",
            ascending=True,
        )
        .astype(int)
    )


    annotated_nodes_path = (
        out_dir
        / "r4_5_gt_annotated_basin_nodes.csv"
    )

    annotated_basins_path = (
        out_dir
        / "r4_5_gt_annotated_basins.csv"
    )

    report_out = (
        out_dir
        / "r4_5_causal_transform_basin_graph.json"
    )


    annotated_nodes.to_csv(
        annotated_nodes_path,
        index=False,
    )

    annotated_basins.to_csv(
        annotated_basins_path,
        index=False,
    )


    # ========================================================
    # Diagnostic views
    # ========================================================

    top_blind = (
        annotated_basins
        .sort_values(
            "blind_duration_rank"
        )
        .head(20)
    )


    persistent = annotated_basins[
        annotated_basins[
            "update_count"
        ]
        >= 5
    ].copy()


    gt_best_persistent = (
        persistent
        .sort_values(
            [
                "postfreeze_gt_median_node_disagreement_m",
                "postfreeze_gt_representative_disagreement_m",
            ]
        )
        .head(20)
    )


    result = {
        "stage":
            "R4.5_CAUSAL_TRANSFORM_BASIN_GRAPH",

        "status":
            "PASS_R4_5_CAUSAL_TRANSFORM_BASIN_GRAPH_EXECUTION",

        "blind_basin_freeze_manifest_sha256":
            blind_manifest_sha,

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

        "counts":
            blind_manifest[
                "counts"
            ],

        "top_blind_duration_basins_postfreeze_labels":
            top_blind.to_dict(
                "records"
            ),

        "gt_best_persistent_basins":
            gt_best_persistent.to_dict(
                "records"
            ),

        "contract": {
            "basin_graph_used_gt":
                False,

            "gt_loaded_after_basin_freeze":
                True,

            "r3_modified":
                False,

            "r4_4_modified":
                False,
        },

        "outputs": {
            "blind_nodes":
                str(
                    blind_nodes_path
                ),

            "blind_edges":
                str(
                    blind_edges_path
                ),

            "blind_basins":
                str(
                    blind_basins_path
                ),

            "blind_freeze_manifest":
                str(
                    blind_manifest_path
                ),

            "gt_annotated_nodes":
                str(
                    annotated_nodes_path
                ),

            "gt_annotated_basins":
                str(
                    annotated_basins_path
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
    # Print
    # ========================================================

    show_columns = [
        "blind_duration_rank",
        "basin_id",
        "first_query_id",
        "last_query_id",
        "update_count",
        "longest_consecutive_updates",
        "occupancy_fraction",
        "reached_lock",
        "node_count",
        "total_hypothesis_support",
        "split_node_count",
        "merge_node_count",
        "median_edge_distance_m",
        "common_prediction_spread_median_m",
        "scale_median",
        "rotation_median_deg",
        "postfreeze_gt_best_node_disagreement_m",
        "postfreeze_gt_median_node_disagreement_m",
        "postfreeze_gt_representative_disagreement_m",
        "postfreeze_gt_final_best_disagreement_m",
        "postfreeze_gt_rank_by_median",
    ]


    print()
    print("=" * 110)
    print("R4.5 PHASE B — POST-FREEZE GT BASIN LABELS")
    print("=" * 110)

    print(
        "GT prefix scale:",
        f"{gt_model['scale_m_per_visual_px']:.6f}",
    )

    print(
        "GT prefix rotation:",
        f"{gt_model['rotation_deg']:.3f} deg",
    )


    print()
    print("=" * 110)
    print(
        "TOP BLIND-DURATION BASINS — "
        "GT LABELS ATTACHED AFTER FREEZE"
    )
    print("=" * 110)

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
    print("=" * 110)
    print(
        "GT-BEST BASINS AMONG BASINS "
        "WITH >=5 UPDATE OCCUPANCY"
    )
    print("=" * 110)

    print(
        gt_best_persistent[
            show_columns
        ].to_string(
            index=False,
            float_format=lambda x:
                f"{x:.3f}",
        )
    )


    print()
    print("=" * 110)
    print("R4.5 OUTPUT")
    print("=" * 110)

    print(
        "blind nodes:",
        blind_nodes_path,
    )

    print(
        "blind edges:",
        blind_edges_path,
    )

    print(
        "blind basins:",
        blind_basins_path,
    )

    print(
        "blind freeze manifest:",
        blind_manifest_path,
    )

    print(
        "GT annotated basins:",
        annotated_basins_path,
    )

    print(
        "report:",
        report_out,
    )

    print()

    print(
        "STATUS: "
        "PASS_R4_5_CAUSAL_TRANSFORM_BASIN_GRAPH_EXECUTION"
    )


if __name__ == "__main__":
    main()
