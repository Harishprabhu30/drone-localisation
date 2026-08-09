#!/usr/bin/env python3
"""
R4.6 — blind graph-percolation sensitivity diagnostic.

Question
--------
Did R4.5 collapse into one giant basin because 0.5 map-center spacing
was above a graph-percolation transition, or is temporal connected-
component chaining structurally unusable across reasonable map-derived
tolerances?

PHASE A — BLIND
----------------
For four predeclared fractions of nominal map-center spacing:

    1/8, 1/4, 3/8, 1/2

build the same temporal compatibility graph as R4.5 and summarize
connected components.

No GT/reference is read.

All Phase-A tables are written and hashed.

PHASE B — POST-FREEZE GT
------------------------
Only afterward attach GT disagreement to the already-frozen components.

This is a sensitivity diagnosis, NOT threshold selection.

Command:

SCRIPT=scripts/villoc/research/minimum_confident_bootstrap/diagnostics/r4_6_graph_percolation_sensitivity.py 

RUN=outputs/demo_runs/traj01_blind_regression_001  
R3=outputs/research_runs/minimum_confident_bootstrap/traj01_blind_r3_001   
python "$SCRIPT" \  
--run-root "$RUN" \   
--research-root "$R3" \   
2>&1 | tee \   "$R3/postfreeze_eval/r4_6_graph_percolation_sensitivity.log"

"""

from __future__ import annotations

import argparse
import hashlib
import json
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


def model_from_row(row):
    return {
        "a_real": float(row["a_real"]),
        "a_imag": float(row["a_imag"]),
        "b_real": float(row["b_real"]),
        "b_imag": float(row["b_imag"]),
    }


def apply_similarity(xy, model):
    xy = np.asarray(
        xy,
        dtype=float,
    )

    z = (
        xy[:, 0]
        + 1j * xy[:, 1]
    )

    a = complex(
        model["a_real"],
        model["a_imag"],
    )

    b = complex(
        model["b_real"],
        model["b_imag"],
    )

    w = a * z + b

    return np.column_stack(
        [
            w.real,
            w.imag,
        ]
    )


def prediction_distance(
    model_a,
    model_b,
    probes,
):
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


def fit_similarity(
    visual_xy,
    map_xy,
):
    visual_xy = np.asarray(
        visual_xy,
        float,
    )

    map_xy = np.asarray(
        map_xy,
        float,
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


def longest_consecutive_run(query_ids):
    q = sorted(
        set(
            int(x)
            for x in query_ids
        )
    )

    if not q:
        return 0

    best = 1
    current = 1

    for a, b in zip(
        q[:-1],
        q[1:],
    ):
        if b == a + 1:
            current += 1
            best = max(
                best,
                current,
            )
        else:
            current = 1

    return int(best)


class UnionFind:

    def __init__(self):
        self.parent = {}
        self.rank = {}

    def add(self, x):
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(
                self.parent[x]
            )
        return self.parent[x]

    def union(self, a, b):
        ra = self.find(a)
        rb = self.find(b)

        if ra == rb:
            return

        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra

        self.parent[rb] = ra

        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


# ============================================================
# Graph builder
# ============================================================

def build_components(
    nodes,
    relative_by_q,
    start_q,
    threshold_m,
):

    uf = UnionFind()

    for node_id in nodes["node_id"]:
        uf.add(
            node_id
        )

    updates = sorted(
        nodes[
            "update_query_id"
        ].unique()
    )

    groups = {
        int(q):
            g.copy()
        for q, g
        in nodes.groupby(
            "update_query_id"
        )
    }

    edge_count = 0

    for previous_q, current_q in zip(
        updates[:-1],
        updates[1:],
    ):

        previous_q = int(
            previous_q
        )

        current_q = int(
            current_q
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

        previous = groups[
            previous_q
        ]

        current = groups[
            current_q
        ]

        for _, a in previous.iterrows():

            model_a = model_from_row(
                a
            )

            for _, b in current.iterrows():

                d = prediction_distance(
                    model_a,
                    model_from_row(b),
                    probes,
                )

                if d <= threshold_m:

                    uf.union(
                        str(
                            a[
                                "node_id"
                            ]
                        ),
                        str(
                            b[
                                "node_id"
                            ]
                        ),
                    )

                    edge_count += 1

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

    return (
        root_to_nodes,
        int(edge_count),
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
        args.run_root.resolve()
    )

    research_root = (
        args.research_root.resolve()
    )

    out_dir = (
        research_root
        / "postfreeze_eval"
    )


    # --------------------------------------------------------
    # R4.5 frozen nodes and R4.4 spacing definition
    # --------------------------------------------------------

    nodes_path = (
        out_dir
        / "r4_5_blind_basin_nodes.csv"
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


    nodes = pd.read_csv(
        nodes_path
    )

    nodes[
        "update_query_id"
    ] = pd.to_numeric(
        nodes[
            "update_query_id"
        ]
    ).astype(int)


    r44_manifest = json.loads(
        r44_manifest_path.read_text()
    )


    center_spacing_m = float(
        r44_manifest[
            "configuration"
        ][
            "map_center_spacing_m_derived"
        ]
    )


    relative = pd.read_csv(
        relative_path
    )

    relative[
        "query_id"
    ] = pd.to_numeric(
        relative[
            "token0_id"
        ]
    ).astype(int)


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
        nodes[
            "update_query_id"
        ].max()
    )


    # --------------------------------------------------------
    # Predeclared systematic diagnostic sweep.
    # --------------------------------------------------------

    fractions = [
        0.125,
        0.250,
        0.375,
        0.500,
    ]


    phase_a_rows = []

    membership_rows = []


    # ========================================================
    # PHASE A — BLIND
    # ========================================================

    for fraction in fractions:

        threshold_m = (
            fraction
            * center_spacing_m
        )

        (
            components,
            edge_count,
        ) = build_components(
            nodes,
            relative_by_q,
            start_q,
            threshold_m,
        )


        component_records = []

        for index, (
            root,
            node_ids,
        ) in enumerate(
            components.items(),
            1,
        ):

            group = nodes[
                nodes[
                    "node_id"
                ].isin(
                    node_ids
                )
            ]


            query_ids = sorted(
                group[
                    "update_query_id"
                ].unique()
            )


            record = {
                "threshold_fraction":
                    fraction,

                "threshold_m":
                    threshold_m,

                "component_local_id":
                    int(index),

                "node_count":
                    int(
                        len(
                            group
                        )
                    ),

                "update_count":
                    int(
                        len(
                            query_ids
                        )
                    ),

                "first_query_id":
                    int(
                        min(
                            query_ids
                        )
                    ),

                "last_query_id":
                    int(
                        max(
                            query_ids
                        )
                    ),

                "longest_consecutive_updates":
                    longest_consecutive_run(
                        query_ids
                    ),

                "reached_lock":
                    bool(
                        lock_q
                        in query_ids
                    ),

                "total_hypothesis_support":
                    int(
                        group[
                            "member_count"
                        ].sum()
                    ),
            }


            component_records.append(
                record
            )


            for node_id in node_ids:

                membership_rows.append(
                    {
                        "threshold_fraction":
                            fraction,

                        "threshold_m":
                            threshold_m,

                        "component_local_id":
                            int(
                                index
                            ),

                        "node_id":
                            node_id,
                    }
                )


        comp = pd.DataFrame(
            component_records
        )


        largest = (
            comp.sort_values(
                "node_count",
                ascending=False,
            )
            .iloc[0]
        )


        multi_update = comp[
            comp[
                "update_count"
            ]
            >= 5
        ]


        phase_a_rows.append(
            {
                "threshold_fraction":
                    fraction,

                "threshold_m":
                    threshold_m,

                "edge_count":
                    edge_count,

                "component_count":
                    int(
                        len(
                            comp
                        )
                    ),

                "largest_component_nodes":
                    int(
                        largest[
                            "node_count"
                        ]
                    ),

                "largest_component_fraction":
                    float(
                        largest[
                            "node_count"
                        ]
                        / len(
                            nodes
                        )
                    ),

                "largest_component_updates":
                    int(
                        largest[
                            "update_count"
                        ]
                    ),

                "largest_component_longest_streak":
                    int(
                        largest[
                            "longest_consecutive_updates"
                        ]
                    ),

                "components_with_5plus_updates":
                    int(
                        len(
                            multi_update
                        )
                    ),

                "components_reaching_lock":
                    int(
                        comp[
                            "reached_lock"
                        ].sum()
                    ),

                "max_update_count_any_component":
                    int(
                        comp[
                            "update_count"
                        ].max()
                    ),

                "max_longest_streak_any_component":
                    int(
                        comp[
                            "longest_consecutive_updates"
                        ].max()
                    ),
            }
        )


    phase_a = pd.DataFrame(
        phase_a_rows
    )


    membership = pd.DataFrame(
        membership_rows
    )


    blind_summary_path = (
        out_dir
        / "r4_6_blind_percolation_summary.csv"
    )

    blind_membership_path = (
        out_dir
        / "r4_6_blind_component_membership.csv"
    )

    blind_manifest_path = (
        out_dir
        / "r4_6_blind_percolation_freeze_manifest.json"
    )


    phase_a.to_csv(
        blind_summary_path,
        index=False,
    )


    membership.to_csv(
        blind_membership_path,
        index=False,
    )


    blind_manifest = {
        "stage":
            "R4.6_BLIND_GRAPH_PERCOLATION_SWEEP",

        "map_center_spacing_m":
            center_spacing_m,

        "threshold_fractions":
            fractions,

        "thresholds_m":
            [
                f * center_spacing_m
                for f in fractions
            ],

        "input_nodes_sha256":
            sha256(
                nodes_path
            ),

        "blind_contract": {
            "gt_used":
                False,

            "reference_used":
                False,

            "thresholds_selected_using_gt":
                False,

            "r3_modified":
                False,

            "r4_5_modified":
                False,
        },

        "outputs": {
            "summary_csv":
                str(
                    blind_summary_path
                ),

            "summary_sha256":
                sha256(
                    blind_summary_path
                ),

            "membership_csv":
                str(
                    blind_membership_path
                ),

            "membership_sha256":
                sha256(
                    blind_membership_path
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
    print("=" * 108)
    print(
        "R4.6 PHASE A — "
        "BLIND GRAPH-PERCOLATION SWEEP FROZEN"
    )
    print("=" * 108)

    print(
        "map-center spacing:",
        f"{center_spacing_m:.3f} m",
    )

    print()

    print(
        phase_a.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.3f}",
        )
    )

    print()

    print(
        "blind freeze SHA256:",
        blind_manifest_sha,
    )


    # ========================================================
    # PHASE B — GT labels only after blind sweep freeze
    # ========================================================

    reference_path = (
        run_root
        / "evaluation/"
          "reference_attachment.csv"
    )


    reference = pd.read_csv(
        reference_path
    )


    transformer = Transformer.from_crs(
        "EPSG:4326",
        "EPSG:3346",
        always_xy=True,
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
        ]
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


    gt_prediction = apply_similarity(
        common_probes,
        gt_model,
    )


    node_gt = {}


    for _, row in nodes.iterrows():

        prediction = apply_similarity(
            common_probes,
            model_from_row(row),
        )


        d = np.linalg.norm(
            prediction
            - gt_prediction,
            axis=1,
        )


        node_gt[
            str(
                row[
                    "node_id"
                ]
            )
        ] = float(
            d.max()
        )


    gt_rows = []


    for (
        fraction,
        threshold_group,
    ) in membership.groupby(
        "threshold_fraction"
    ):

        component_stats = []


        for component_id, group in (
            threshold_group.groupby(
                "component_local_id"
            )
        ):

            values = np.asarray(
                [
                    node_gt[
                        str(
                            node_id
                        )
                    ]
                    for node_id
                    in group[
                        "node_id"
                    ]
                ],
                dtype=float,
            )


            component_stats.append(
                {
                    "component_local_id":
                        int(
                            component_id
                        ),

                    "node_count":
                        int(
                            len(
                                values
                            )
                        ),

                    "gt_best_node_m":
                        float(
                            values.min()
                        ),

                    "gt_median_node_m":
                        float(
                            np.median(
                                values
                            )
                        ),
                }
            )


        stats = pd.DataFrame(
            component_stats
        )


        best_component = (
            stats.sort_values(
                [
                    "gt_median_node_m",
                    "gt_best_node_m",
                ]
            )
            .iloc[0]
        )


        component_containing_best_node = (
            stats.sort_values(
                "gt_best_node_m"
            )
            .iloc[0]
        )


        gt_rows.append(
            {
                "threshold_fraction":
                    float(
                        fraction
                    ),

                "threshold_m":
                    float(
                        fraction
                        * center_spacing_m
                    ),

                "gt_best_component_by_median_id":
                    int(
                        best_component[
                            "component_local_id"
                        ]
                    ),

                "gt_best_component_median_m":
                    float(
                        best_component[
                            "gt_median_node_m"
                        ]
                    ),

                "gt_best_component_nodes":
                    int(
                        best_component[
                            "node_count"
                        ]
                    ),

                "component_containing_gt_best_node_id":
                    int(
                        component_containing_best_node[
                            "component_local_id"
                        ]
                    ),

                "component_containing_gt_best_node_best_m":
                    float(
                        component_containing_best_node[
                            "gt_best_node_m"
                        ]
                    ),

                "component_containing_gt_best_node_median_m":
                    float(
                        component_containing_best_node[
                            "gt_median_node_m"
                        ]
                    ),

                "component_containing_gt_best_node_nodes":
                    int(
                        component_containing_best_node[
                            "node_count"
                        ]
                    ),
            }
        )


    gt_summary = pd.DataFrame(
        gt_rows
    )


    combined = (
        phase_a
        .merge(
            gt_summary,
            on=[
                "threshold_fraction",
                "threshold_m",
            ],
            validate="one_to_one",
        )
    )


    combined_path = (
        out_dir
        / "r4_6_gt_annotated_percolation_summary.csv"
    )


    report_path = (
        out_dir
        / "r4_6_graph_percolation_sensitivity.json"
    )


    combined.to_csv(
        combined_path,
        index=False,
    )


    report = {
        "stage":
            "R4.6_GRAPH_PERCOLATION_SENSITIVITY",

        "status":
            "PASS_R4_6_GRAPH_PERCOLATION_SENSITIVITY_EXECUTION",

        "blind_freeze_manifest_sha256":
            blind_manifest_sha,

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

            "gt_loaded_after_phase_a_freeze":
                True,

            "this_is_threshold_selection":
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
    print("=" * 108)
    print(
        "R4.6 PHASE B — "
        "POST-FREEZE GT COMPONENT LABELS"
    )
    print("=" * 108)

    print(
        combined.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.3f}",
        )
    )

    print()
    print(
        "report:",
        report_path,
    )

    print()

    print(
        "STATUS: "
        "PASS_R4_6_GRAPH_PERCOLATION_SENSITIVITY_EXECUTION"
    )


if __name__ == "__main__":
    main()
