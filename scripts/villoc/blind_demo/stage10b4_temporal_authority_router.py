#!/usr/bin/env python3
"""
R5.4G — temporal-authority compatibility router.

Purpose
-------
Produce the historical Stage-09 output contract without applying a
second localization/fusion policy when the bootstrap backend already
owns temporal absolute corrections.

For minimum_confident_v2:

    R3-v2 timeline
        -> causal accept/hold decisions
        -> Stage 08 causal map trajectory
        -> THIS ROUTER
        -> downstream-compatible fused trajectory

No additional correction is estimated here.

Historical legacy_strict remains owned by:
    stage10b4_blind_temporal_fusion.py

This script is therefore a temporal-ownership adapter, not a new
localization algorithm.

No GT / GPS / SRT / reference is used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd


TIMELINE_SCHEMA = (
    "villoc.blind_map_state_timeline.v1"
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


def require(
    condition,
    message,
):

    if not condition:
        raise RuntimeError(message)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--map-trajectory",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--map-state-timeline",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--run-root",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    t0 = time.perf_counter()

    map_trajectory_path = (
        args.map_trajectory.resolve()
    )

    timeline_path = (
        args.map_state_timeline.resolve()
    )

    run_root = (
        args.run_root.resolve()
    )


    for path in [
        map_trajectory_path,
        timeline_path,
    ]:

        require(
            path.exists(),
            f"Missing input: {path}",
        )


    # ========================================================
    # Read frozen canonical timeline
    # ========================================================

    timeline = json.loads(
        timeline_path.read_text()
    )


    require(
        timeline.get("schema")
        == TIMELINE_SCHEMA,
        (
            "Unexpected map-state timeline schema: "
            f"{timeline.get('schema')}"
        ),
    )


    blind_contract = timeline.get(
        "blind_contract",
        {},
    )


    for key in [
        "gps_used",
        "srt_used",
        "reference_used",
        "oracle_used",
        "evaluation_error_used",
    ]:

        require(
            not bool(
                blind_contract.get(
                    key,
                    False,
                )
            ),
            f"Blind-contract violation: {key}=True",
        )


    backend = str(
        timeline[
            "source_backend"
        ]
    )


    # --------------------------------------------------------
    # CRITICAL OWNERSHIP RULE
    #
    # This compatibility path is currently promoted only for
    # minimum_confident_v2.
    #
    # A positive legacy_strict lock must still pass through
    # the historical stage10b4 temporal fusion.
    # --------------------------------------------------------

    require(
        backend
        == "minimum_confident_v2",
        (
            "This router currently owns Stage 09 only for "
            "minimum_confident_v2. "
            "legacy_strict must use "
            "stage10b4_blind_temporal_fusion.py."
        ),
    )


    localization_state = str(
        timeline[
            "localization_state"
        ]
    )

    map_state_available = bool(
        timeline[
            "map_state_available"
        ]
    )

    map_state_trust = str(
        timeline[
            "map_state_trust"
        ]
    )

    events = timeline.get(
        "events",
        [],
    )


    require(
        isinstance(
            events,
            list,
        ),
        "Timeline events must be a list.",
    )


    require(
        int(
            timeline[
                "event_count"
            ]
        )
        == len(events),
        "Timeline event_count does not match events list.",
    )


    if map_state_available:

        require(
            localization_state
            == "PROVISIONAL_ABSOLUTE_LOCK",
            (
                "minimum_confident_v2 positive map state "
                "must remain PROVISIONAL_ABSOLUTE_LOCK."
            ),
        )

        require(
            len(events) >= 1,
            (
                "Positive map state has no causal "
                "timeline events."
            ),
        )

    else:

        require(
            localization_state
            == "NO_PROVISIONAL_LOCK",
            (
                "minimum_confident_v2 no-map state must "
                "remain NO_PROVISIONAL_LOCK."
            ),
        )

        require(
            len(events) == 0,
            (
                "No-map-state timeline unexpectedly "
                "contains events."
            ),
        )


    # ========================================================
    # Read Stage-08 causal map trajectory
    # ========================================================

    trajectory = pd.read_csv(
        map_trajectory_path
    )


    require(
        len(trajectory) > 0,
        "Map trajectory is empty.",
    )


    required_columns = {
        "frame_index",
        "timestamp_s",
        "image_path",
        "sequence_frame_id",
        "query_id",
        "token0_id",
        "visual_x_px",
        "visual_y_px",
        "relative_x_m",
        "relative_y_m",
        "relative_cumulative_distance_m",
        "estimated_map_x",
        "estimated_map_y",
        "map_aligned_available",
        "initialization_state",
        "map_crs",
        "map_alignment_source",
        "map_state_trust",
        "bootstrap_backend",
        "active_map_state_event_index",
        "active_map_state_source_query",
        "active_map_state_action",
        "bootstrap_scale_m_per_visual_px",
        "bootstrap_rotation_deg",
        "step_motion_px",
        "pair_safe_image_only",
        "coordinate_contract",
        "reference_used",
    }


    missing = sorted(
        required_columns
        - set(
            trajectory.columns
        )
    )


    require(
        not missing,
        (
            "Causal map trajectory missing columns: "
            f"{missing}"
        ),
    )


    require(
        not trajectory[
            "reference_used"
        ].astype(bool).any(),
        "Stage-08 trajectory reports reference_used=True.",
    )


    require(
        set(
            trajectory[
                "bootstrap_backend"
            ].astype(str).unique()
        )
        == {
            backend
        },
        (
            "Stage-08 backend provenance does not "
            "match timeline backend."
        ),
    )


    # ========================================================
    # Verify event application represented in Stage 08
    # ========================================================

    declared_event_ids = {
        int(
            event[
                "event_index"
            ]
        )
        for event in events
    }


    applied_event_ids = {
        int(v)
        for v in pd.to_numeric(
            trajectory[
                "active_map_state_event_index"
            ],
            errors="coerce",
        )
        .dropna()
        .unique()
    }


    require(
        declared_event_ids
        == applied_event_ids,
        (
            "Stage-08 trajectory does not represent "
            "the full causal timeline.\n"
            f"declared={sorted(declared_event_ids)}\n"
            f"applied ={sorted(applied_event_ids)}"
        ),
    )


    map_available_count = int(
        pd.to_numeric(
            trajectory[
                "map_aligned_available"
            ],
            errors="raise",
        ).sum()
    )


    if map_state_available:

        require(
            map_available_count > 0,
            (
                "Positive map-state timeline produced "
                "zero map-aligned rows."
            ),
        )

    else:

        require(
            map_available_count == 0,
            (
                "NO_PROVISIONAL_LOCK produced "
                "map-aligned rows."
            ),
        )


    # ========================================================
    # Temporal authority: pass through R3-v2 state history
    # ========================================================

    fused = trajectory.copy()


    # Historical temporal-fusion contract carried an explicit
    # baseline map estimate before secondary correction.
    #
    # For R3-v2 there is NO secondary correction. Therefore:
    #
    #     relative_map == estimated_map
    #
    # where available.
    fused[
        "relative_map_x"
    ] = fused[
        "estimated_map_x"
    ]


    fused[
        "relative_map_y"
    ] = fused[
        "estimated_map_y"
    ]


    # Final algorithm-level localization result.
    #
    # Row-level causal state remains available separately in:
    #     initialization_state
    fused[
        "localization_state"
    ] = localization_state


    fused[
        "temporal_authority"
    ] = (
        "minimum_confident_v2_state_timeline"
    )


    fused[
        "secondary_temporal_fusion_applied"
    ] = False


    # --------------------------------------------------------
    # Compatibility diagnostics.
    #
    # Do NOT pretend these are legacy strict-A corrections.
    # They describe R3-v2's already-applied map-state events.
    # --------------------------------------------------------

    event_mask = (
        pd.to_numeric(
            fused[
                "active_map_state_event_index"
            ],
            errors="coerce",
        )
        .notna()
        &
        (
            fused[
                "query_id"
            ].astype(int)
            ==
            pd.to_numeric(
                fused[
                    "active_map_state_source_query"
                ],
                errors="coerce",
            )
        )
    )


    fused[
        "strict_a_blind"
    ] = False

    fused[
        "strict_b_blind"
    ] = False


    fused[
        "correction_candidate"
    ] = event_mask


    fused[
        "correction_accepted"
    ] = event_mask


    fused[
        "correction_applied"
    ] = event_mask


    reasons = np.full(
        len(fused),
        "",
        dtype=object,
    )


    pre_map = (
        fused[
            "map_aligned_available"
        ].astype(int)
        == 0
    )


    reasons[
        pre_map.to_numpy()
    ] = (
        "RELATIVE_ONLY_PRE_PROVISIONAL_MAP_STATE"
    )


    propagated = (
        ~pre_map
        & ~event_mask
    )


    reasons[
        propagated.to_numpy()
    ] = (
        "R3V2_HELD_STATE_PROPAGATION"
    )


    reasons[
        event_mask.to_numpy()
    ] = (
        "R3V2_CAUSAL_MAP_STATE_EVENT"
    )


    fused[
        "correction_reason"
    ] = reasons


    # These legacy numeric diagnostics are intentionally N/A.
    # R3-v2 already made its innovation decision upstream.
    for column in [
        "temporal_residual_m",
        "temporal_threshold_m",
        "distance_since_anchor_m",
        "correction_delta_easting_m",
        "correction_delta_northing_m",
        "correction_magnitude_m",
        "fusion_alpha",
    ]:

        fused[
            column
        ] = np.nan


    fused[
        "anchor_token0_id_before"
    ] = pd.to_numeric(
        fused[
            "active_map_state_source_query"
        ],
        errors="coerce",
    )


    fused[
        "temporal_policy"
    ] = (
        "r3v2_causal_state_timeline_no_secondary_fusion"
    )


    fused[
        "map_estimate_source"
    ] = np.where(
        fused[
            "map_aligned_available"
        ].astype(int)
        == 1,
        "minimum_confident_v2_causal_map_state",
        "relative_visual_only",
    )


    # ========================================================
    # Correction/event manifest
    # ========================================================

    manifest = pd.DataFrame(
        {
            "frame_index":
                fused[
                    "frame_index"
                ],

            "timestamp_s":
                fused[
                    "timestamp_s"
                ],

            "sequence_frame_id":
                fused[
                    "sequence_frame_id"
                ],

            "query_id":
                fused[
                    "query_id"
                ],

            "token0_id":
                fused[
                    "token0_id"
                ],

            "localization_state":
                localization_state,

            "map_aligned_available":
                fused[
                    "map_aligned_available"
                ],

            "map_state_trust":
                map_state_trust,

            "active_map_state_event_index":
                fused[
                    "active_map_state_event_index"
                ],

            "active_map_state_source_query":
                fused[
                    "active_map_state_source_query"
                ],

            "active_map_state_action":
                fused[
                    "active_map_state_action"
                ],

            "correction_candidate":
                fused[
                    "correction_candidate"
                ],

            "correction_accepted":
                fused[
                    "correction_accepted"
                ],

            "correction_applied":
                fused[
                    "correction_applied"
                ],

            "correction_reason":
                fused[
                    "correction_reason"
                ],

            # Legacy Stage-11 compatibility only.
            #
            # These metrics belong to the historical
            # Stage10B4 temporal policy. R3-v2 does not
            # compute or use them, so they remain N/A.
            "distance_since_anchor_m":
                np.nan,

            "temporal_residual_m":
                np.nan,

            "temporal_threshold_m":
                np.nan,

            "temporal_policy":
                fused[
                    "temporal_policy"
                ],

            "secondary_temporal_fusion_applied":
                False,

            "reference_used":
                False,
        }
    )


    # ========================================================
    # Output
    # ========================================================

    metadata_dir = (
        run_root
        / "metadata/"
          "blind_temporal_fusion"
    )

    trajectory_dir = (
        run_root
        / "trajectories"
    )

    report_dir = (
        run_root
        / "reports/"
          "blind_temporal_fusion"
    )


    metadata_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    trajectory_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    manifest_path = (
        metadata_dir
        / "blind_temporal_correction_manifest.csv"
    )

    fused_path = (
        trajectory_dir
        / "blind_temporal_fused_trajectory.csv"
    )

    report_path = (
        report_dir
        / "blind_temporal_fusion_report.json"
    )


    manifest.to_csv(
        manifest_path,
        index=False,
    )

    fused.to_csv(
        fused_path,
        index=False,
    )


    event_count = int(
        event_mask.sum()
    )


    require(
        event_count
        == len(events),
        (
            "Compatibility event count differs "
            "from canonical timeline event count."
        ),
    )


    # Every Stage-08 coordinate must survive exactly.
    for column in [
        "estimated_map_x",
        "estimated_map_y",
    ]:

        before = pd.to_numeric(
            trajectory[
                column
            ],
            errors="coerce",
        ).to_numpy(float)

        after = pd.to_numeric(
            fused[
                column
            ],
            errors="coerce",
        ).to_numpy(float)

        require(
            np.array_equal(
                np.isnan(before),
                np.isnan(after),
            ),
            (
                f"{column}: NaN mask changed "
                "inside Stage-09 router."
            ),
        )

        finite = np.isfinite(
            before
        )

        require(
            np.allclose(
                before[
                    finite
                ],
                after[
                    finite
                ],
                atol=0.0,
                rtol=0.0,
            ),
            (
                f"{column}: map estimates were "
                "modified by Stage-09 router."
            ),
        )


    runtime_s = float(
        time.perf_counter()
        - t0
    )


    report = {
        "stage":
            "R5.4G_TEMPORAL_AUTHORITY_ROUTER",

        "status":
            (
                "PASS_R3V2_TEMPORAL_AUTHORITY_ROUTER"
                if map_state_available
                else
                "PASS_R3V2_TEMPORAL_AUTHORITY_ROUTER_NO_MAP_STATE"
            ),

        "localization_state":
            localization_state,

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

        "policy": {
            "temporal_authority":
                "minimum_confident_v2",

            "authority_source":
                "canonical causal map-state timeline",

            "secondary_temporal_fusion_applied":
                False,

            "legacy_stage10b4_applied":
                False,

            "legacy_temporal_diagnostics_available":
                False,

            "legacy_temporal_diagnostics_note":
                (
                    "distance_since_anchor_m, "
                    "temporal_residual_m, and "
                    "temporal_threshold_m are retained as "
                    "N/A compatibility columns only. "
                    "R3-v2 uses its own causal acquisition/"
                    "tracking innovation logic upstream."
                ),

            "reason":
                (
                    "R3-v2 already owns causal absolute "
                    "accept/hold decisions; applying the legacy "
                    "strict-A temporal corrector would create "
                    "a second competing map-state authority."
                ),
        },

        "bootstrap": {
            "source_backend":
                backend,

            "localization_state":
                localization_state,

            "map_state_available":
                map_state_available,

            "map_state_trust":
                map_state_trust,

            "first_effective_query_id":
                timeline.get(
                    "first_effective_query_id"
                ),

            "final_source_query_id":
                timeline.get(
                    "final_source_query_id"
                ),

            "causal_map_state_events":
                len(events),
        },

        "counts": {
            "trajectory_rows":
                int(
                    len(fused)
                ),

            "map_aligned_rows":
                int(
                    map_available_count
                ),

            "relative_only_rows":
                int(
                    len(fused)
                    - map_available_count
                ),

            "canonical_state_events":
                int(
                    len(events)
                ),

            "accepted_state_events":
                event_count,

            "secondary_corrections":
                0,
        },

        "accepted_summary": {
            "accepted_map_state_events":
                event_count,

            "first_event_query_id":
                (
                    int(
                        events[0][
                            "effective_from_query_id"
                        ]
                    )
                    if events
                    else None
                ),

            "last_event_query_id":
                (
                    int(
                        events[-1][
                            "effective_from_query_id"
                        ]
                    )
                    if events
                    else None
                ),

            "secondary_temporal_corrections":
                0,
        },

        "runtime": {
            "total_stage_wall_s":
                runtime_s,
        },

        "inputs_sha256": {
            "map_trajectory":
                sha256(
                    map_trajectory_path
                ),

            "map_state_timeline":
                sha256(
                    timeline_path
                ),
        },

        "outputs": {
            "correction_manifest":
                str(
                    manifest_path
                ),

            "correction_manifest_sha256":
                sha256(
                    manifest_path
                ),

            "fused_trajectory":
                str(
                    fused_path
                ),

            "fused_trajectory_sha256":
                sha256(
                    fused_path
                ),
        },

        "important_note":
            (
                "The fused trajectory is a compatibility "
                "surface. Its map coordinates are unchanged "
                "from Stage 08. R3-v2, not Stage 09, is the "
                "temporal absolute-correction authority."
            ),
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
        "R5.4G — TEMPORAL AUTHORITY ROUTER"
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
        "temporal authority:",
        "minimum_confident_v2",
    )

    print(
        "legacy Stage10B4 applied:",
        False,
    )

    print(
        "secondary fusion applied:",
        False,
    )

    print(
        "trajectory rows:",
        len(fused),
    )

    print(
        "map-aligned rows:",
        map_available_count,
    )

    print(
        "relative-only rows:",
        len(fused)
        - map_available_count,
    )

    print(
        "causal state events:",
        len(events),
    )

    print(
        "coordinates modified:",
        False,
    )

    print(
        "fused trajectory:",
        fused_path,
    )

    print(
        "correction manifest:",
        manifest_path,
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
