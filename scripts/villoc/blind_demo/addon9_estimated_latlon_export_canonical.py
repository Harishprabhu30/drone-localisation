#!/usr/bin/env python3
"""
R5.4H — canonical estimated lat/lon export.

Consumes
--------
1. Stage-09 downstream-compatible fused trajectory.
2. Stage-09 temporal-authority report.
3. Blind DINO/ORB query summary.
4. Optional map tile index for map-bound sanity.

Purpose
-------
Convert already-estimated map coordinates into WGS84 output coordinates.

This script does NOT:
    * acquire GPS,
    * read SRT,
    * use GT/reference,
    * choose an absolute candidate,
    * alter the map trajectory,
    * change R3-v2 state,
    * upgrade PROVISIONAL_ABSOLUTE_LOCK to ABSOLUTE_LOCKED.

For minimum_confident_v2:

    map state / tracking decisions
        are already frozen upstream.

This stage is output conversion + evidence packaging only.

Required wording
----------------
estimated_lat/lon are visual map-matching outputs, not GPS inputs.
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


REQUIRED_LABEL = (
    "estimated_lat/lon are visual map-matching outputs, "
    "not GPS inputs."
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


def bool_series(
    series: pd.Series,
) -> pd.Series:

    if pd.api.types.is_bool_dtype(
        series
    ):
        return series.fillna(False)

    if pd.api.types.is_numeric_dtype(
        series
    ):
        return (
            pd.to_numeric(
                series,
                errors="coerce",
            )
            .fillna(0)
            != 0
        )

    normalized = (
        series
        .astype(str)
        .str.strip()
        .str.lower()
    )

    return normalized.isin(
        {
            "true",
            "1",
            "1.0",
            "yes",
            "y",
        }
    )


def json_safe(
    value,
):

    if isinstance(
        value,
        np.integer,
    ):
        return int(value)

    if isinstance(
        value,
        np.floating,
    ):

        if not math.isfinite(
            float(value)
        ):
            return None

        return float(value)

    if isinstance(
        value,
        np.bool_,
    ):
        return bool(value)

    raise TypeError(
        f"Not JSON serializable: {type(value)}"
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--fused-trajectory",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--fusion-report",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--absolute-query-summary",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--run-root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--tile-index",
        type=Path,
    )

    parser.add_argument(
        "--source-crs",
        default="EPSG:3346",
    )

    parser.add_argument(
        "--target-crs",
        default="EPSG:4326",
    )

    args = parser.parse_args()

    stage_start = (
        time.perf_counter()
    )


    fused_path = (
        args.fused_trajectory.resolve()
    )

    fusion_report_path = (
        args.fusion_report.resolve()
    )

    qsum_path = (
        args.absolute_query_summary.resolve()
    )

    run_root = (
        args.run_root.resolve()
    )

    tile_index_path = (
        args.tile_index.resolve()
        if args.tile_index
        else None
    )


    required_paths = [
        fused_path,
        fusion_report_path,
        qsum_path,
    ]


    if tile_index_path is not None:

        required_paths.append(
            tile_index_path
        )


    for path in required_paths:

        require(
            path.exists(),
            f"Missing input: {path}",
        )


    # ========================================================
    # Freeze input hashes before any processing
    # ========================================================

    input_hashes_before = {
        "fused_trajectory":
            sha256(
                fused_path
            ),

        "fusion_report":
            sha256(
                fusion_report_path
            ),

        "absolute_query_summary":
            sha256(
                qsum_path
            ),
    }


    if tile_index_path is not None:

        input_hashes_before[
            "tile_index"
        ] = sha256(
            tile_index_path
        )


    # ========================================================
    # Stage-09 authority report
    # ========================================================

    fusion_report = json.loads(
        fusion_report_path.read_text()
    )


    allowed_status = {
        "PASS_R3V2_TEMPORAL_AUTHORITY_ROUTER",
        "PASS_R3V2_TEMPORAL_AUTHORITY_ROUTER_NO_MAP_STATE",
    }


    fusion_status = str(
        fusion_report.get(
            "status"
        )
    )


    require(
        fusion_status
        in allowed_status,
        (
            "Unexpected canonical Stage-09 status: "
            f"{fusion_status}"
        ),
    )


    localization_state = str(
        fusion_report.get(
            "localization_state"
        )
    )


    blind_contract = fusion_report.get(
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
            blind_contract.get(
                key
            )
            is False,
            (
                "Blind fusion contract violated: "
                f"{key}="
                f"{blind_contract.get(key)!r}"
            ),
        )


    policy = fusion_report.get(
        "policy",
        {},
    )


    require(
        policy.get(
            "temporal_authority"
        )
        == "minimum_confident_v2",
        (
            "Canonical exporter currently expects "
            "minimum_confident_v2 temporal authority."
        ),
    )


    require(
        policy.get(
            "secondary_temporal_fusion_applied"
        )
        is False,
        (
            "Unexpected secondary temporal fusion "
            "on R3-v2 trajectory."
        ),
    )


    bootstrap = fusion_report.get(
        "bootstrap",
        {},
    )


    map_state_available = bool(
        bootstrap.get(
            "map_state_available"
        )
    )


    map_state_trust = str(
        bootstrap.get(
            "map_state_trust"
        )
    )


    first_effective_query_id = (
        int(
            bootstrap[
                "first_effective_query_id"
            ]
        )
        if bootstrap.get(
            "first_effective_query_id"
        )
        is not None
        else None
    )


    # ========================================================
    # Fused trajectory
    # ========================================================

    fused = pd.read_csv(
        fused_path
    )


    required_fused = {
        "frame_index",
        "timestamp_s",
        "image_path",
        "sequence_frame_id",
        "query_id",
        "token0_id",
        "relative_x_m",
        "relative_y_m",
        "estimated_map_x",
        "estimated_map_y",
        "map_aligned_available",
        "map_crs",
        "map_state_trust",
        "bootstrap_backend",
        "correction_accepted",
        "active_map_state_event_index",
        "active_map_state_source_query",
        "active_map_state_action",
        "localization_state",
        "temporal_authority",
        "reference_used",
    }


    missing = sorted(
        required_fused
        - set(
            fused.columns
        )
    )


    require(
        not missing,
        (
            "Canonical fused trajectory missing "
            f"columns: {missing}"
        ),
    )


    require(
        len(fused) > 0,
        "Fused trajectory is empty.",
    )


    fused[
        "query_id"
    ] = pd.to_numeric(
        fused[
            "query_id"
        ],
        errors="raise",
    ).astype(int)


    require(
        not fused[
            "query_id"
        ].duplicated().any(),
        "Fused trajectory has duplicate query IDs.",
    )


    require(
        not bool_series(
            fused[
                "reference_used"
            ]
        ).any(),
        "Fused trajectory reports reference_used=True.",
    )


    require(
        set(
            fused[
                "bootstrap_backend"
            ].astype(str).unique()
        )
        == {
            "minimum_confident_v2"
        },
        (
            "Unexpected bootstrap backend in "
            "canonical fused trajectory."
        ),
    )


    require(
        set(
            fused[
                "temporal_authority"
            ].astype(str).unique()
        )
        == {
            "minimum_confident_v2_state_timeline"
        },
        (
            "Unexpected temporal authority in "
            "canonical fused trajectory."
        ),
    )


    available = bool_series(
        fused[
            "map_aligned_available"
        ]
    ).to_numpy(bool)


    map_x = pd.to_numeric(
        fused[
            "estimated_map_x"
        ],
        errors="coerce",
    ).to_numpy(float)


    map_y = pd.to_numeric(
        fused[
            "estimated_map_y"
        ],
        errors="coerce",
    ).to_numpy(float)


    # ========================================================
    # Positive / no-map contract
    # ========================================================

    if map_state_available:

        require(
            localization_state
            == "PROVISIONAL_ABSOLUTE_LOCK",
            (
                "Positive R3-v2 output must remain "
                "PROVISIONAL_ABSOLUTE_LOCK."
            ),
        )


        require(
            map_state_trust
            == "PROVISIONAL",
            (
                "Positive R3-v2 map state must retain "
                "PROVISIONAL trust."
            ),
        )


        require(
            first_effective_query_id
            is not None,
            (
                "Positive R3-v2 report has no "
                "first effective query."
            ),
        )


        require(
            available.any(),
            (
                "Positive map state has no "
                "map-aligned trajectory rows."
            ),
        )


        require(
            np.isfinite(
                map_x[
                    available
                ]
            ).all()
            and
            np.isfinite(
                map_y[
                    available
                ]
            ).all(),
            (
                "Available map coordinates contain "
                "non-finite values."
            ),
        )


        pre = (
            fused[
                "query_id"
            ].to_numpy(int)
            < first_effective_query_id
        )


        require(
            np.isnan(
                map_x[
                    pre
                ]
            ).all()
            and
            np.isnan(
                map_y[
                    pre
                ]
            ).all(),
            (
                "Pre-provisional-lock map positions "
                "are exposed."
            ),
        )


    else:

        require(
            localization_state
            == "NO_PROVISIONAL_LOCK",
            (
                "No-map R3-v2 state must remain "
                "NO_PROVISIONAL_LOCK."
            ),
        )


        require(
            not available.any(),
            (
                "NO_PROVISIONAL_LOCK trajectory "
                "contains map-aligned rows."
            ),
        )


        require(
            np.isnan(
                map_x
            ).all()
            and
            np.isnan(
                map_y
            ).all(),
            (
                "NO_PROVISIONAL_LOCK trajectory "
                "contains map coordinates."
            ),
        )


    # ========================================================
    # Blind DINO + ORB evidence
    # ========================================================

    qsum = pd.read_csv(
        qsum_path,
        usecols=[
            "query_id",
            "original_top1_tile_id",
            "original_top1_dino_score",
            "reranked_top1_tile_id",
            "reranked_top1_verifier_score",
            "reranked_top1_hybrid_score",
            "reranked_top1_inliers",
        ],
    )


    qsum[
        "query_id"
    ] = pd.to_numeric(
        qsum[
            "query_id"
        ],
        errors="raise",
    ).astype(int)


    require(
        not qsum[
            "query_id"
        ].duplicated().any(),
        "Absolute query summary has duplicate query IDs.",
    )


    out = fused.merge(
        qsum,
        on="query_id",
        how="left",
        validate="one_to_one",
    )


    require(
        len(out)
        == len(fused),
        (
            "Query-evidence join changed "
            "trajectory row count."
        ),
    )


    require(
        not out[
            "original_top1_tile_id"
        ].isna().any(),
        "Missing DINO Top-1 evidence.",
    )


    require(
        not out[
            "reranked_top1_tile_id"
        ].isna().any(),
        "Missing ORB selected evidence.",
    )


    # ========================================================
    # EPSG:3346 -> WGS84
    # ========================================================

    source_crs_values = (
        out[
            "map_crs"
        ]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )


    require(
        all(
            value
            == args.source_crs
            for value
            in source_crs_values
        ),
        (
            "Unexpected source CRS values: "
            f"{source_crs_values}"
        ),
    )


    estimated_lon = np.full(
        len(out),
        np.nan,
        dtype=float,
    )

    estimated_lat = np.full(
        len(out),
        np.nan,
        dtype=float,
    )


    transform_start = (
        time.perf_counter()
    )


    if available.any():

        try:
            from pyproj import Transformer
        except Exception as exc:

            raise RuntimeError(
                "pyproj is required for canonical "
                "estimated lat/lon export."
            ) from exc


        transformer = (
            Transformer.from_crs(
                args.source_crs,
                args.target_crs,
                always_xy=True,
            )
        )


        lon, lat = transformer.transform(
            map_x[
                available
            ],
            map_y[
                available
            ],
        )


        estimated_lon[
            available
        ] = lon

        estimated_lat[
            available
        ] = lat


    transform_s = (
        time.perf_counter()
        - transform_start
    )


    # ========================================================
    # Causal state-event semantics
    # ========================================================

    accepted_event = bool_series(
        out[
            "correction_accepted"
        ]
    ).to_numpy(bool)


    event_index = pd.to_numeric(
        out[
            "active_map_state_event_index"
        ],
        errors="coerce",
    ).to_numpy(float)


    # First accepted state event is the initial map-lock event.
    map_lock_event = np.zeros(
        len(out),
        dtype=bool,
    )


    accepted_indices = np.flatnonzero(
        accepted_event
    )


    if len(
        accepted_indices
    ):

        map_lock_event[
            accepted_indices[
                0
            ]
        ] = True


    correction_source = np.full(
        len(out),
        None,
        dtype=object,
    )


    for idx in accepted_indices:

        action = str(
            out.iloc[
                idx
            ][
                "active_map_state_action"
            ]
        )


        if map_lock_event[idx]:

            correction_source[idx] = (
                "minimum_confident_v2_acquisition"
            )

        elif action == "TRACKING_ACCEPT":

            correction_source[idx] = (
                "minimum_confident_v2_tracking"
            )

        else:

            correction_source[idx] = (
                "minimum_confident_v2_state_event"
            )


    # ========================================================
    # Required output fields
    # ========================================================

    confidence_score = pd.to_numeric(
        out[
            "reranked_top1_hybrid_score"
        ],
        errors="coerce",
    )


    # Runtime fields remain explicit null values here.
    #
    # The canonical exporter must not silently reuse timings
    # from a different bootstrap/fusion implementation.
    runtime_relative_ms = np.full(
        len(out),
        np.nan,
    )

    runtime_retrieval_ms = np.full(
        len(out),
        np.nan,
    )

    runtime_rerank_ms = np.full(
        len(out),
        np.nan,
    )


    submission = pd.DataFrame(
        {
            "frame_index":
                out[
                    "frame_index"
                ],

            "sequence_frame_id":
                out[
                    "sequence_frame_id"
                ],

            "query_id":
                out[
                    "query_id"
                ],

            "token0_id":
                out[
                    "token0_id"
                ],

            "timestamp_s":
                out[
                    "timestamp_s"
                ],

            "image_path":
                out[
                    "image_path"
                ],

            "relative_x_m":
                out[
                    "relative_x_m"
                ],

            "relative_y_m":
                out[
                    "relative_y_m"
                ],

            "estimated_map_x":
                out[
                    "estimated_map_x"
                ],

            "estimated_map_y":
                out[
                    "estimated_map_y"
                ],

            "estimated_lat":
                estimated_lat,

            "estimated_lon":
                estimated_lon,

            "map_aligned_available":
                available,

            "map_lock_event":
                map_lock_event,

            "confidence_score":
                confidence_score,

            "accepted_correction":
                accepted_event,

            "correction_source":
                correction_source,

            "dino_top1_tile_id":
                out[
                    "original_top1_tile_id"
                ],

            "dino_top1_score":
                pd.to_numeric(
                    out[
                        "original_top1_dino_score"
                    ],
                    errors="coerce",
                ),

            "orb_selected_tile_id":
                out[
                    "reranked_top1_tile_id"
                ],

            "orb_score":
                pd.to_numeric(
                    out[
                        "reranked_top1_verifier_score"
                    ],
                    errors="coerce",
                ),

            "orb_inliers":
                pd.to_numeric(
                    out[
                        "reranked_top1_inliers"
                    ],
                    errors="coerce",
                ),

            "runtime_relative_ms":
                runtime_relative_ms,

            "runtime_retrieval_ms":
                runtime_retrieval_ms,

            "runtime_rerank_ms":
                runtime_rerank_ms,

            # ------------------------------------------------
            # Canonical semantic extensions.
            # ------------------------------------------------
            "localization_state":
                localization_state,

            "map_state_trust":
                map_state_trust,

            "bootstrap_backend":
                "minimum_confident_v2",

            "temporal_authority":
                "minimum_confident_v2_state_timeline",

            "active_map_state_event_index":
                event_index,

            "active_map_state_source_query":
                pd.to_numeric(
                    out[
                        "active_map_state_source_query"
                    ],
                    errors="coerce",
                ),
        }
    )


    # ========================================================
    # Assertions on output
    # ========================================================

    require(
        int(
            submission[
                "map_aligned_available"
            ].sum()
        )
        == int(
            available.sum()
        ),
        "Map-aligned row count changed during export.",
    )


    require(
        submission[
            "estimated_lat"
        ].notna().to_numpy().tolist()
        ==
        available.tolist(),
        (
            "Estimated-lat availability differs "
            "from map availability."
        ),
    )


    require(
        submission[
            "estimated_lon"
        ].notna().to_numpy().tolist()
        ==
        available.tolist(),
        (
            "Estimated-lon availability differs "
            "from map availability."
        ),
    )


    require(
        localization_state
        != "ABSOLUTE_LOCKED",
        (
            "Canonical R3-v2 export illegally "
            "upgraded provisional state."
        ),
    )


    # ========================================================
    # Blind map-coverage sanity
    # ========================================================

    map_coverage = None


    if tile_index_path is not None:

        tile_index = pd.read_csv(
            tile_index_path
        )


        required_tile = {
            "left_easting",
            "right_easting",
            "bottom_northing",
            "top_northing",
        }


        missing_tile = sorted(
            required_tile
            - set(
                tile_index.columns
            )
        )


        require(
            not missing_tile,
            (
                "Tile index missing map-bound fields: "
                f"{missing_tile}"
            ),
        )


        xmin = float(
            pd.to_numeric(
                tile_index[
                    "left_easting"
                ],
                errors="raise",
            ).min()
        )

        xmax = float(
            pd.to_numeric(
                tile_index[
                    "right_easting"
                ],
                errors="raise",
            ).max()
        )

        ymin = float(
            pd.to_numeric(
                tile_index[
                    "bottom_northing"
                ],
                errors="raise",
            ).min()
        )

        ymax = float(
            pd.to_numeric(
                tile_index[
                    "top_northing"
                ],
                errors="raise",
            ).max()
        )


        inside = (
            available
            &
            (map_x >= xmin)
            &
            (map_x <= xmax)
            &
            (map_y >= ymin)
            &
            (map_y <= ymax)
        )


        available_count = int(
            available.sum()
        )


        inside_count = int(
            inside.sum()
        )


        map_coverage = {
            "map_bounds_epsg3346": {
                "xmin":
                    xmin,

                "xmax":
                    xmax,

                "ymin":
                    ymin,

                "ymax":
                    ymax,
            },

            "available_positions":
                available_count,

            "inside_map_bounds":
                inside_count,

            "outside_map_bounds":
                (
                    available_count
                    - inside_count
                ),

            "all_available_inside_map_bounds":
                bool(
                    available_count
                    == inside_count
                ),
        }


    # ========================================================
    # Verify source inputs unchanged
    # ========================================================

    input_hashes_after = {
        "fused_trajectory":
            sha256(
                fused_path
            ),

        "fusion_report":
            sha256(
                fusion_report_path
            ),

        "absolute_query_summary":
            sha256(
                qsum_path
            ),
    }


    if tile_index_path is not None:

        input_hashes_after[
            "tile_index"
        ] = sha256(
            tile_index_path
        )


    require(
        input_hashes_before
        == input_hashes_after,
        (
            "One or more frozen inputs changed "
            "during canonical Add-on 9."
        ),
    )


    # ========================================================
    # Save
    # ========================================================

    trajectory_dir = (
        run_root
        / "trajectories"
    )

    report_dir = (
        run_root
        / "reports/"
          "addon9_estimated_latlon"
    )


    trajectory_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    submission_path = (
        trajectory_dir
        / "submission_estimated_trajectory.csv"
    )

    report_path = (
        report_dir
        / "estimated_latlon_export_report.json"
    )


    write_start = (
        time.perf_counter()
    )


    submission.to_csv(
        submission_path,
        index=False,
    )


    write_s = (
        time.perf_counter()
        - write_start
    )


    post = submission.loc[
        submission[
            "map_aligned_available"
        ]
    ]


    stage_s = (
        time.perf_counter()
        - stage_start
    )


    geographic_range = {
        "estimated_lat_min":
            (
                float(
                    post[
                        "estimated_lat"
                    ].min()
                )
                if len(post)
                else None
            ),

        "estimated_lat_max":
            (
                float(
                    post[
                        "estimated_lat"
                    ].max()
                )
                if len(post)
                else None
            ),

        "estimated_lon_min":
            (
                float(
                    post[
                        "estimated_lon"
                    ].min()
                )
                if len(post)
                else None
            ),

        "estimated_lon_max":
            (
                float(
                    post[
                        "estimated_lon"
                    ].max()
                )
                if len(post)
                else None
            ),
    }


    report = {
        "stage":
            "ADDON9_ESTIMATED_LATLON_EXPORT",

        "status":
            (
                "PASS_ADDON9_ESTIMATED_LATLON_EXPORT"
                if map_state_available
                else
                "PASS_ADDON9_NO_ABSOLUTE_EXPORT_NO_MAP_STATE"
            ),

        "export_contract":
            "canonical_r3v2_visual_map_output_v1",

        "required_label":
            REQUIRED_LABEL,

        "localization_state":
            localization_state,

        "map_state_trust":
            map_state_trust,

        "bootstrap_backend":
            "minimum_confident_v2",

        "temporal_authority":
            "minimum_confident_v2_state_timeline",

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

            "ground_truth_used_for_decisions":
                False,

            "prelock_backfill_performed":
                False,

            "estimated_latlon_are_output_not_input":
                True,
        },

        "coordinate_conversion": {
            "source_crs":
                args.source_crs,

            "target_crs":
                args.target_crs,

            "always_xy":
                True,

            "library":
                "pyproj",

            "coordinates_modified_before_conversion":
                False,
        },

        "confidence_contract": {
            "confidence_score_source":
                "reranked_top1_hybrid_score",

            "confidence_score_is_probability":
                False,

            "confidence_role":
                "blind candidate evidence only",

            "note":
                (
                    "The ORB hybrid score is exported "
                    "as retrieval/verifier evidence. "
                    "R3-v2 map-state confidence is represented "
                    "separately by localization_state, "
                    "map_state_trust, and causal state events."
                ),
        },

        "temporal_contract": {
            "secondary_fusion_applied":
                False,

            "accepted_correction_means":
                (
                    "a causal minimum_confident_v2 "
                    "map-state event became effective"
                ),

            "map_lock_event_means":
                (
                    "first causal provisional map-state "
                    "event only"
                ),
        },

        "runtime_contract": {
            "runtime_columns_are":
                (
                    "explicitly unavailable in this "
                    "standalone integration replay"
                ),

            "runtime_relative_ms":
                None,

            "runtime_retrieval_ms":
                None,

            "runtime_rerank_ms":
                None,
        },

        "rows": {
            "total":
                int(
                    len(
                        submission
                    )
                ),

            "prelock_without_map_position":
                int(
                    (
                        ~submission[
                            "map_aligned_available"
                        ]
                    ).sum()
                ),

            "map_aligned_available":
                int(
                    submission[
                        "map_aligned_available"
                    ].sum()
                ),

            "estimated_latlon_available":
                int(
                    submission[
                        "estimated_lat"
                    ].notna().sum()
                ),

            "accepted_corrections":
                int(
                    submission[
                        "accepted_correction"
                    ].sum()
                ),

            "map_lock_events":
                int(
                    submission[
                        "map_lock_event"
                    ].sum()
                ),
        },

        "geographic_range":
            geographic_range,

        "map_coverage_sanity":
            map_coverage,

        "runtime": {
            "coordinate_transform_s":
                float(
                    transform_s
                ),

            "output_write_s":
                float(
                    write_s
                ),

            "total_stage_wall_s":
                float(
                    stage_s
                ),
        },

        "inputs_sha256":
            input_hashes_before,

        "outputs": {
            "submission_estimated_trajectory":
                str(
                    submission_path
                ),

            "submission_sha256":
                sha256(
                    submission_path
                ),

            "report":
                str(
                    report_path
                ),
        },

        "blind_mode_note":
            (
                "Reference unavailable: accuracy metrics "
                "not computed. Estimated lat/lon are "
                "visual map-matching outputs, not GPS inputs."
            ),

        "semantic_guarantees": {
            "provisional_state_preserved":
                True,

            "absolute_locked_emitted":
                False,

            "map_coordinates_modified":
                False,

            "gt_or_reference_consumed":
                False,
        },
    }


    report_path.write_text(
        json.dumps(
            report,
            indent=2,
            default=json_safe,
            allow_nan=False,
        )
    )


    # ========================================================
    # Print
    # ========================================================

    print()
    print("=" * 108)
    print(
        "R5.4H — CANONICAL ESTIMATED LAT/LON EXPORT"
    )
    print("=" * 108)

    print(
        "localization state:",
        localization_state,
    )

    print(
        "map-state trust:",
        map_state_trust,
    )

    print(
        "bootstrap backend:",
        "minimum_confident_v2",
    )

    print(
        "temporal authority:",
        "minimum_confident_v2_state_timeline",
    )

    print(
        "total rows:",
        len(
            submission
        ),
    )

    print(
        "map positions available:",
        int(
            submission[
                "map_aligned_available"
            ].sum()
        ),
    )

    print(
        "estimated lat/lon available:",
        int(
            submission[
                "estimated_lat"
            ].notna().sum()
        ),
    )

    print(
        "accepted causal state events:",
        int(
            submission[
                "accepted_correction"
            ].sum()
        ),
    )

    print(
        "initial provisional lock events:",
        int(
            submission[
                "map_lock_event"
            ].sum()
        ),
    )

    print(
        "pre-lock backfill:",
        False,
    )

    print(
        "GT/reference used:",
        False,
    )

    print(
        "ABSOLUTE_LOCKED emitted:",
        False,
    )


    if len(post):

        print(
            "estimated latitude range:",
            float(
                post[
                    "estimated_lat"
                ].min()
            ),
            "->",
            float(
                post[
                    "estimated_lat"
                ].max()
            ),
        )

        print(
            "estimated longitude range:",
            float(
                post[
                    "estimated_lon"
                ].min()
            ),
            "->",
            float(
                post[
                    "estimated_lon"
                ].max()
            ),
        )


    if map_coverage is not None:

        print(
            "map positions inside prepared map:",
            map_coverage[
                "inside_map_bounds"
            ],
            "/",
            map_coverage[
                "available_positions"
            ],
        )


    print()
    print(
        REQUIRED_LABEL
    )

    print()
    print(
        "submission:",
        submission_path,
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
