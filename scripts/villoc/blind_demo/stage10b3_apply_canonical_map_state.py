#!/usr/bin/env python3
"""
R5.4F — timeline-aware canonical map alignment.

Consumes:
    villoc.blind_map_state_timeline.v1

The consumer is completely bootstrap-backend agnostic.

Examples
--------
Demo:
    q30 -> transform A
    q30..end use A.

Stateful trajectory:
    q9  -> A
    q10 -> B
    q11 -> C
    ...
    each transform becomes effective causally from its event query.

No-lock:
    no events
    entire trajectory remains relative only.

Important
---------
A later transform is NEVER back-applied to earlier rows.

No GT / GPS / SRT / reference information is used.
"""

from __future__ import annotations

import argparse
import time
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


TIMELINE_SCHEMA = (
    "villoc.blind_map_state_timeline.v1"
)

COORDINATE_CONTRACT = (
    "xfeat_visual_xy_to_epsg3346_causal_similarity_timeline_v1"
)

TRANSFORM_FIELDS = (
    "a_real",
    "a_imag",
    "b_real",
    "b_imag",
    "scale_m_per_visual_px",
    "rotation_deg",
)


def sha256(path: Path) -> str:

    h = hashlib.sha256()

    with path.open("rb") as f:

        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def choose_column(
    df,
    candidates,
    label,
):

    for column in candidates:

        if column in df.columns:
            return column

    raise RuntimeError(
        f"Could not resolve {label}; "
        f"tried {candidates}"
    )


def normalize_transform(
    transform,
    label,
):

    if not isinstance(
        transform,
        dict,
    ):

        raise RuntimeError(
            f"{label}: transform is not a dictionary."
        )

    result = {}

    for key in TRANSFORM_FIELDS:

        if key not in transform:

            raise RuntimeError(
                f"{label}: missing {key}"
            )

        value = float(
            transform[key]
        )

        if not math.isfinite(
            value
        ):

            raise RuntimeError(
                f"{label}: non-finite {key}"
            )

        result[key] = value

    if (
        result[
            "scale_m_per_visual_px"
        ]
        <= 0.0
    ):

        raise RuntimeError(
            f"{label}: non-positive scale."
        )

    return result


def path_length(
    x,
    y,
) -> float:

    x = np.asarray(
        x,
        dtype=float,
    )

    y = np.asarray(
        y,
        dtype=float,
    )

    if len(x) < 2:
        return 0.0

    valid = (
        np.isfinite(x)
        & np.isfinite(y)
    )

    idx = np.flatnonzero(
        valid
    )

    if len(idx) < 2:
        return 0.0

    # Only sum adjacent valid rows.
    total = 0.0

    for a, b in zip(
        idx[:-1],
        idx[1:],
    ):

        if b != a + 1:
            continue

        total += math.hypot(
            x[b] - x[a],
            y[b] - y[a],
        )

    return float(total)


def main():

    stage_start = time.perf_counter()

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--map-state-timeline",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--relative-trajectory",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--max-query-id",
        type=int,
    )

    args = parser.parse_args()


    timeline_path = (
        args.map_state_timeline.resolve()
    )

    relative_path = (
        args.relative_trajectory.resolve()
    )

    manifest_path = (
        args.manifest.resolve()
    )

    output_root = (
        args.output_root.resolve()
    )


    for path in [
        timeline_path,
        relative_path,
        manifest_path,
    ]:

        if not path.exists():

            raise RuntimeError(
                f"Missing input: {path}"
            )


    # ========================================================
    # Canonical causal state timeline
    # ========================================================

    state = json.loads(
        timeline_path.read_text()
    )


    if (
        state.get(
            "schema"
        )
        != TIMELINE_SCHEMA
    ):

        raise RuntimeError(
            "Unexpected timeline schema: "
            f"{state.get('schema')}"
        )


    blind_contract = state.get(
        "blind_contract",
        {}
    )


    for key in [
        "gps_used",
        "srt_used",
        "reference_used",
        "oracle_used",
        "evaluation_error_used",
    ]:

        if bool(
            blind_contract.get(
                key,
                False,
            )
        ):

            raise RuntimeError(
                f"Blind-contract violation: {key}=True"
            )


    backend = str(
        state[
            "source_backend"
        ]
    )

    localization_state = str(
        state[
            "localization_state"
        ]
    )

    map_state_available = bool(
        state[
            "map_state_available"
        ]
    )

    map_state_trust = str(
        state[
            "map_state_trust"
        ]
    )

    map_crs = str(
        state[
            "map_crs"
        ]
    )


    raw_events = state.get(
        "events",
        []
    )


    if not isinstance(
        raw_events,
        list,
    ):

        raise RuntimeError(
            "events must be a list."
        )


    if (
        map_state_available
        and not raw_events
    ):

        raise RuntimeError(
            "map_state_available=True but timeline has no events."
        )


    if (
        not map_state_available
        and raw_events
    ):

        raise RuntimeError(
            "No-map-state timeline unexpectedly contains events."
        )


    events = []


    for i, event in enumerate(
        raw_events
    ):

        q = int(
            event[
                "effective_from_query_id"
            ]
        )

        transform = normalize_transform(
            event[
                "transform"
            ],
            f"event {i}",
        )

        events.append(
            {
                **event,
                "effective_from_query_id":
                    q,

                "transform":
                    transform,
            }
        )


    events = sorted(
        events,
        key=lambda e:
            e[
                "effective_from_query_id"
            ],
    )


    event_queries = [
        event[
            "effective_from_query_id"
        ]
        for event in events
    ]


    if len(
        event_queries
    ) != len(
        set(
            event_queries
        )
    ):

        raise RuntimeError(
            "Multiple causal map-state events "
            "share the same effective query."
        )


    if any(
        b <= a
        for a, b in zip(
            event_queries[:-1],
            event_queries[1:],
        )
    ):

        raise RuntimeError(
            "Map-state events are not strictly causal."
        )


    # ========================================================
    # Blind relative trajectory + manifest
    # ========================================================

    relative = pd.read_csv(
        relative_path
    )

    manifest = pd.read_csv(
        manifest_path
    )


    relative_query_col = choose_column(
        relative,
        [
            "query_id",
            "token0_id",
        ],
        "relative query ID",
    )

    manifest_query_col = choose_column(
        manifest,
        [
            "query_id",
            "token0_id",
        ],
        "manifest query ID",
    )

    timestamp_col = choose_column(
        manifest,
        [
            "timestamp_s",
            "video_time_s",
        ],
        "manifest timestamp",
    )

    image_col = choose_column(
        manifest,
        [
            "image_path",
            "image",
        ],
        "manifest image path",
    )

    visual_x_col = choose_column(
        relative,
        [
            "visual_x_px",
        ],
        "visual X",
    )

    visual_y_col = choose_column(
        relative,
        [
            "visual_y_px",
        ],
        "visual Y",
    )


    relative = relative.copy()

    manifest = manifest.copy()


    relative[
        "_query_id"
    ] = pd.to_numeric(
        relative[
            relative_query_col
        ],
        errors="raise",
    ).astype(int)


    manifest[
        "_query_id"
    ] = pd.to_numeric(
        manifest[
            manifest_query_col
        ],
        errors="raise",
    ).astype(int)


    if args.max_query_id is not None:

        relative = relative[
            relative[
                "_query_id"
            ]
            <= args.max_query_id
        ].copy()

        manifest = manifest[
            manifest[
                "_query_id"
            ]
            <= args.max_query_id
        ].copy()


    if (
        relative[
            "_query_id"
        ].duplicated().any()
    ):

        raise RuntimeError(
            "Relative trajectory contains duplicate query IDs."
        )


    if (
        manifest[
            "_query_id"
        ].duplicated().any()
    ):

        raise RuntimeError(
            "Manifest contains duplicate query IDs."
        )


    if set(
        relative[
            "_query_id"
        ]
    ) != set(
        manifest[
            "_query_id"
        ]
    ):

        raise RuntimeError(
            "Relative trajectory and manifest query sets differ."
        )


    manifest_small = manifest[
        [
            "_query_id",
            timestamp_col,
            image_col,
        ]
    ].rename(
        columns={
            timestamp_col:
                "timestamp_s",

            image_col:
                "image_path",
        }
    )


    data = relative.merge(
        manifest_small,
        on="_query_id",
        how="left",
        validate="one_to_one",
    )


    data = (
        data
        .sort_values(
            "_query_id"
        )
        .reset_index(
            drop=True
        )
    )


    q = data[
        "_query_id"
    ].to_numpy(int)


    visual_x = pd.to_numeric(
        data[
            visual_x_col
        ],
        errors="raise",
    ).to_numpy(float)


    visual_y = pd.to_numeric(
        data[
            visual_y_col
        ],
        errors="raise",
    ).to_numpy(float)


    # ========================================================
    # Allocate output
    # ========================================================

    n = len(data)

    estimated_e = np.full(
        n,
        np.nan,
    )

    estimated_n = np.full(
        n,
        np.nan,
    )

    relative_x_m = np.full(
        n,
        np.nan,
    )

    relative_y_m = np.full(
        n,
        np.nan,
    )

    map_aligned = np.zeros(
        n,
        dtype=int,
    )

    active_event_index = np.full(
        n,
        np.nan,
    )

    active_source_query = np.full(
        n,
        np.nan,
    )

    active_scale = np.full(
        n,
        np.nan,
    )

    active_rotation = np.full(
        n,
        np.nan,
    )

    active_action = np.full(
        n,
        "",
        dtype=object,
    )

    initialization_state = np.full(
        n,
        localization_state,
        dtype=object,
    )

    alignment_source = np.full(
        n,
        "relative_only_no_map_state",
        dtype=object,
    )


    # ========================================================
    # Causal piecewise application
    # ========================================================

    if events:

        first_q = events[
            0
        ][
            "effective_from_query_id"
        ]


        initialization_state[
            q < first_q
        ] = (
            "RELATIVE_ONLY_PRE_MAP_STATE"
        )

        alignment_source[
            q < first_q
        ] = (
            "relative_only_pre_map_state"
        )


        for index, event in enumerate(
            events
        ):

            start_q = int(
                event[
                    "effective_from_query_id"
                ]
            )


            if index + 1 < len(
                events
            ):

                end_q_exclusive = int(
                    events[
                        index + 1
                    ][
                        "effective_from_query_id"
                    ]
                )

                mask = (
                    (q >= start_q)
                    &
                    (q < end_q_exclusive)
                )

            else:

                mask = (
                    q >= start_q
                )


            if not mask.any():
                continue


            transform = event[
                "transform"
            ]


            ar = float(
                transform[
                    "a_real"
                ]
            )

            ai = float(
                transform[
                    "a_imag"
                ]
            )

            br = float(
                transform[
                    "b_real"
                ]
            )

            bi = float(
                transform[
                    "b_imag"
                ]
            )


            rx = (
                ar * visual_x[
                    mask
                ]
                -
                ai * visual_y[
                    mask
                ]
            )


            ry = (
                ai * visual_x[
                    mask
                ]
                +
                ar * visual_y[
                    mask
                ]
            )


            relative_x_m[
                mask
            ] = rx

            relative_y_m[
                mask
            ] = ry


            estimated_e[
                mask
            ] = (
                rx + br
            )

            estimated_n[
                mask
            ] = (
                ry + bi
            )


            map_aligned[
                mask
            ] = 1


            active_event_index[
                mask
            ] = int(
                event[
                    "event_index"
                ]
            )


            active_source_query[
                mask
            ] = int(
                event[
                    "source_update_query_id"
                ]
            )


            active_scale[
                mask
            ] = float(
                transform[
                    "scale_m_per_visual_px"
                ]
            )


            active_rotation[
                mask
            ] = float(
                transform[
                    "rotation_deg"
                ]
            )


            active_action[
                mask
            ] = str(
                event[
                    "action"
                ]
            )


            initialization_state[
                mask
            ] = (
                localization_state
            )


            alignment_source[
                mask
            ] = (
                f"{backend}:"
                f"{map_state_trust}:"
                f"event_{int(event['event_index'])}"
            )


    # ========================================================
    # Motion diagnostics
    # ========================================================

    step_motion_px = np.zeros(
        n,
        dtype=float,
    )


    if n > 1:

        step_motion_px[
            1:
        ] = np.hypot(
            np.diff(
                visual_x
            ),
            np.diff(
                visual_y
            ),
        )


    cumulative_map_distance = np.full(
        n,
        np.nan,
    )


    running = 0.0
    seen_map = False


    for i in range(n):

        if not map_aligned[i]:
            continue


        if not seen_map:

            running = 0.0
            cumulative_map_distance[i] = 0.0
            seen_map = True
            continue


        if (
            i > 0
            and map_aligned[
                i - 1
            ]
        ):

            running += math.hypot(
                estimated_e[i]
                - estimated_e[
                    i - 1
                ],

                estimated_n[i]
                - estimated_n[
                    i - 1
                ],
            )


        cumulative_map_distance[
            i
        ] = running


    # ========================================================
    # Output schema
    # ========================================================

    out = pd.DataFrame(
        {
            "frame_index":
                np.arange(
                    n,
                    dtype=int,
                ),

            "timestamp_s":
                pd.to_numeric(
                    data[
                        "timestamp_s"
                    ],
                    errors="raise",
                ),

            "image_path":
                data[
                    "image_path"
                ].astype(str),

            "sequence_frame_id":
                q,

            "query_id":
                q,

            "token0_id":
                q,

            "visual_x_px":
                visual_x,

            "visual_y_px":
                visual_y,

            "relative_x_m":
                relative_x_m,

            "relative_y_m":
                relative_y_m,

            "relative_cumulative_distance_m":
                cumulative_map_distance,

            "estimated_map_x":
                estimated_e,

            "estimated_map_y":
                estimated_n,

            "map_aligned_available":
                map_aligned,

            "initialization_state":
                initialization_state,

            "map_crs":
                map_crs,

            "map_alignment_source":
                alignment_source,

            "map_state_trust":
                map_state_trust,

            "bootstrap_backend":
                backend,

            "active_map_state_event_index":
                active_event_index,

            "active_map_state_source_query":
                active_source_query,

            "active_map_state_action":
                active_action,

            "bootstrap_transform_frozen":
                map_aligned,

            "bootstrap_lock_frame":
                active_source_query,

            "bootstrap_scale_m_per_visual_px":
                active_scale,

            "bootstrap_rotation_deg":
                active_rotation,

            "step_motion_px":
                step_motion_px,

            "pair_safe_image_only":
                True,

            "coordinate_contract":
                COORDINATE_CONTRACT,

            "reference_used":
                False,
        }
    )


    for column in [
        "visual_yaw_rad",
        "visual_yaw_deg_unwrapped",
    ]:

        if column in data.columns:

            out[column] = data[
                column
            ]


    # ========================================================
    # Contract assertions
    # ========================================================

    aligned_rows = int(
        out[
            "map_aligned_available"
        ].sum()
    )


    if not events:

        if aligned_rows != 0:

            raise RuntimeError(
                "No-event timeline produced aligned rows."
            )

    else:

        first_event_q = int(
            events[
                0
            ][
                "effective_from_query_id"
            ]
        )


        expected = int(
            (
                q
                >= first_event_q
            ).sum()
        )


        if aligned_rows != expected:

            raise RuntimeError(
                "Causal aligned-row count mismatch."
            )


        before = out[
            out[
                "query_id"
            ]
            < first_event_q
        ]


        if (
            before[
                "estimated_map_x"
            ].notna().any()
            or before[
                "estimated_map_y"
            ].notna().any()
        ):

            raise RuntimeError(
                "Pre-maturity rows received map coordinates."
            )


    applied_event_indices = sorted(
        int(v)
        for v in pd.Series(
            active_event_index
        )
        .dropna()
        .unique()
    )


    expected_event_indices = sorted(
        int(
            event[
                "event_index"
            ]
        )
        for event in events
        if int(
            event[
                "effective_from_query_id"
            ]
        )
        <= int(
            q.max()
        )
    )


    if (
        applied_event_indices
        != expected_event_indices
    ):

        raise RuntimeError(
            "Not all causal timeline events were applied.\n"
            f"expected: {expected_event_indices}\n"
            f"applied:  {applied_event_indices}"
        )


    # ========================================================
    # Save
    # ========================================================

    trajectory_dir = (
        output_root
        / "trajectories"
    )

    report_dir = (
        output_root
        / "reports/"
          "blind_map_alignment"
    )


    trajectory_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    trajectory_path = (
        trajectory_dir
        / "blind_map_aligned_relative_trajectory.csv"
    )


    report_path = (
        report_dir
        / "blind_map_alignment_report.json"
    )


    out.to_csv(
        trajectory_path,
        index=False,
    )


    aligned = out[
        out[
            "map_aligned_available"
        ]
        == 1
    ]


    stage_wall_s = float(
        time.perf_counter()
        - stage_start
    )

    report = {
        "runtime": {
            "total_stage_wall_s":
                stage_wall_s,
        },

        "stage":
            "R5.4F_CAUSAL_CANONICAL_MAP_ALIGNMENT",

        "status":
            (
                "PASS_CAUSAL_CANONICAL_MAP_ALIGNMENT"
                if events
                else
                "PASS_CAUSAL_CANONICAL_MAP_ALIGNMENT_NO_MAP_STATE"
            ),

        "blind_contract": {
            "gps_used":
                False,

            "srt_used":
                False,

            "reference_used":
                False,

            "oracle_used":
                False,

            "evaluation_error_used":
                False,
        },

        "timeline": {
            "schema":
                state[
                    "schema"
                ],

            "source_backend":
                backend,

            "localization_state":
                localization_state,

            "map_state_trust":
                map_state_trust,

            "declared_event_count":
                int(
                    state[
                        "event_count"
                    ]
                ),

            "applied_event_count":
                len(
                    applied_event_indices
                ),

            "applied_event_indices":
                applied_event_indices,

            "first_effective_query_id":
                state.get(
                    "first_effective_query_id"
                ),

            "final_source_query_id":
                state.get(
                    "final_source_query_id"
                ),
        },

        "rows": {
            "total":
                int(
                    len(out)
                ),

            "relative_only":
                int(
                    len(out)
                    - aligned_rows
                ),

            "map_aligned":
                aligned_rows,
        },

        "map_trajectory": {
            "first_query_id":
                (
                    int(
                        aligned[
                            "query_id"
                        ].iloc[0]
                    )
                    if len(
                        aligned
                    )
                    else None
                ),

            "last_query_id":
                (
                    int(
                        aligned[
                            "query_id"
                        ].iloc[-1]
                    )
                    if len(
                        aligned
                    )
                    else None
                ),

            "causal_path_length_m":
                path_length(
                    out[
                        "estimated_map_x"
                    ],
                    out[
                        "estimated_map_y"
                    ],
                ),

            "easting_min":
                (
                    float(
                        aligned[
                            "estimated_map_x"
                        ].min()
                    )
                    if len(
                        aligned
                    )
                    else None
                ),

            "easting_max":
                (
                    float(
                        aligned[
                            "estimated_map_x"
                        ].max()
                    )
                    if len(
                        aligned
                    )
                    else None
                ),

            "northing_min":
                (
                    float(
                        aligned[
                            "estimated_map_y"
                        ].min()
                    )
                    if len(
                        aligned
                    )
                    else None
                ),

            "northing_max":
                (
                    float(
                        aligned[
                            "estimated_map_y"
                        ].max()
                    )
                    if len(
                        aligned
                    )
                    else None
                ),
        },

        "coordinate_contract": {
            "name":
                COORDINATE_CONTRACT,

            "map_crs":
                map_crs,

            "causal_rule":
                (
                    "each transform event applies only from "
                    "its effective query until the next event"
                ),
        },

        "input_sha256": {
            "map_state_timeline":
                sha256(
                    timeline_path
                ),

            "relative_trajectory":
                sha256(
                    relative_path
                ),

            "manifest":
                sha256(
                    manifest_path
                ),
        },

        "outputs": {
            "trajectory":
                str(
                    trajectory_path
                ),

            "trajectory_sha256":
                sha256(
                    trajectory_path
                ),
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
    print("=" * 108)
    print(
        "R5.4F — CAUSAL CANONICAL MAP ALIGNMENT"
    )
    print("=" * 108)

    print(
        "backend:",
        backend,
    )

    print(
        "localization state:",
        localization_state,
    )

    print(
        "map-state trust:",
        map_state_trust,
    )

    print(
        "declared timeline events:",
        len(
            events
        ),
    )

    print(
        "applied timeline events:",
        len(
            applied_event_indices
        ),
    )

    print(
        "applied event indices:",
        applied_event_indices,
    )

    print(
        "total rows:",
        len(out),
    )

    print(
        "relative-only rows:",
        len(out)
        - aligned_rows,
    )

    print(
        "map-aligned rows:",
        aligned_rows,
    )


    if len(
        aligned
    ):

        print(
            "map query range:",
            int(
                aligned[
                    "query_id"
                ].iloc[0]
            ),
            "->",
            int(
                aligned[
                    "query_id"
                ].iloc[-1]
            ),
        )


    print(
        "causal map path length m:",
        report[
            "map_trajectory"
        ][
            "causal_path_length_m"
        ],
    )

    print(
        "trajectory:",
        trajectory_path,
    )

    print(
        "report:",
        report_path,
    )

    print()

    print(
        "STATUS:",
        report[
            "status"
        ],
    )


if __name__ == "__main__":
    main()
