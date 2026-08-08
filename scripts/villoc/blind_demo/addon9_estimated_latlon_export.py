'''
ROOT=outputs/villoc/traj01_90deg_stable120m
MAPROOT=outputs/villoc/90_deg
RUN="$ROOT/blind_demo_addons/eval_traj01_known_gt"

QSUM="$ROOT/reports/s8_12e1_top20_verifier_reranker/512_s256_orb_hybrid_top20_img518/s8_12e1_query_summary.csv"

python scripts/villoc/blind_demo/addon9_estimated_latlon_export.py \
  --fused-trajectory \
  "$RUN/trajectories/blind_temporal_fused_trajectory.csv" \
  --fusion-report \
  "$RUN/reports/blind_temporal_fusion/blind_temporal_fusion_report.json" \
  --absolute-query-summary "$QSUM" \
  --timing-summary \
  "$RUN/metrics/timing_summary.csv" \
  --tile-index \
  "$MAPROOT/metadata/s8_9_satellite_tile_index_512_s256.csv" \
  --run-root "$RUN" \
  --source-crs EPSG:3346 \
  --target-crs EPSG:4326 \
  2>&1 | tee \
  "$RUN/logs/addon9_estimated_latlon_export.log"
'''

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REQUIRED_OUTPUT_COLUMNS = [
    "frame_index",
    "timestamp_s",
    "image_path",

    "relative_x_m",
    "relative_y_m",

    "estimated_map_x",
    "estimated_map_y",
    "estimated_lat",
    "estimated_lon",

    "map_aligned_available",
    "confidence_score",
    "accepted_correction",
    "correction_source",

    "dino_top1_tile_id",
    "dino_top1_score",
    "orb_selected_tile_id",
    "orb_score",
    "orb_inliers",

    "runtime_relative_ms",
    "runtime_retrieval_ms",
    "runtime_rerank_ms",
]


FORBIDDEN = {
    "x_enu_m",
    "y_enu_m",
    "reference_x_m",
    "reference_y_m",
    "reference_cumulative_distance_m",
    "prefix_aligned_x_m",
    "prefix_aligned_y_m",
    "global_aligned_x_m",
    "global_aligned_y_m",
    "prefix_locked_error_m",
    "global_alignment_error_m",
    "abs_error_m_eval_only",
    "abs_hit_le_40m_eval_only",
    "abs_contains_body_eval_only",
    "oracle_hit40_accept_eval_only",
    "fusion_error_m",
    "ground_truth_x",
    "ground_truth_y",
    "gps_lat",
    "gps_lon",
}


REQUIRED_LABEL = (
    "estimated_lat/lon are visual map-matching outputs, "
    "not GPS inputs."
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def bool_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False)

    return (
        s.astype(str)
        .str.strip()
        .str.lower()
        .isin({
            "true",
            "1",
            "yes",
            "y",
            "t",
        })
    )


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, float):
        if not math.isfinite(value):
            return None

    if pd.isna(value):
        return None

    return value


def timing_value(
    timing: pd.DataFrame | None,
    stage: str,
    *,
    prefer_secondary: bool = False,
) -> float | None:

    if timing is None:
        return None

    if "stage" not in timing.columns:
        return None

    row = timing[
        timing["stage"].astype(str)
        == stage
    ]

    if row.empty:
        return None

    row = row.iloc[0]

    if prefer_secondary:
        if "ms_per_secondary_item" in row.index:
            value = pd.to_numeric(
                pd.Series([
                    row["ms_per_secondary_item"]
                ]),
                errors="coerce",
            ).iloc[0]

            if pd.notna(value):
                return float(value)

    if "ms_per_work_item" in row.index:
        value = pd.to_numeric(
            pd.Series([
                row["ms_per_work_item"]
            ]),
            errors="coerce",
        ).iloc[0]

        if pd.notna(value):
            return float(value)

    if (
        prefer_secondary
        and "runtime_s" in row.index
        and "secondary_count" in row.index
    ):
        runtime_s = pd.to_numeric(
            pd.Series([
                row["runtime_s"]
            ]),
            errors="coerce",
        ).iloc[0]

        count = pd.to_numeric(
            pd.Series([
                row["secondary_count"]
            ]),
            errors="coerce",
        ).iloc[0]

        if (
            pd.notna(runtime_s)
            and pd.notna(count)
            and float(count) > 0
        ):
            return float(
                1000.0
                * float(runtime_s)
                / float(count)
            )

    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Export blind visually estimated map "
            "positions as latitude/longitude."
        )
    )

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
        "--timing-summary",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--tile-index",
        type=Path,
        default=None,
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

    stage_start = time.perf_counter()

    fused_path = (
        args.fused_trajectory
        .expanduser()
        .resolve()
    )

    fusion_report_path = (
        args.fusion_report
        .expanduser()
        .resolve()
    )

    qsum_path = (
        args.absolute_query_summary
        .expanduser()
        .resolve()
    )

    run_root = (
        args.run_root
        .expanduser()
        .resolve()
    )

    timing_path = (
        args.timing_summary
        .expanduser()
        .resolve()
        if args.timing_summary
        else None
    )

    tile_index_path = (
        args.tile_index
        .expanduser()
        .resolve()
        if args.tile_index
        else None
    )

    for path in [
        fused_path,
        fusion_report_path,
        qsum_path,
    ]:
        require(
            path.exists(),
            f"Missing input: {path}",
        )

    if timing_path is not None:
        require(
            timing_path.exists(),
            f"Missing timing summary: {timing_path}",
        )

    if tile_index_path is not None:
        require(
            tile_index_path.exists(),
            f"Missing tile index: {tile_index_path}",
        )

    input_hashes_before = {
        "fused_trajectory":
            sha256_file(fused_path),

        "fusion_report":
            sha256_file(fusion_report_path),

        "absolute_query_summary":
            sha256_file(qsum_path),
    }

    if timing_path is not None:
        input_hashes_before[
            "timing_summary"
        ] = sha256_file(
            timing_path
        )

    if tile_index_path is not None:
        input_hashes_before[
            "tile_index"
        ] = sha256_file(
            tile_index_path
        )

    # =====================================================
    # Frozen blind fused trajectory.
    # =====================================================

    fused = pd.read_csv(
        fused_path
    )

    require(
        len(fused) == 403,
        (
            "Expected 403 fused rows, got "
            f"{len(fused)}."
        ),
    )

    leaked = sorted(
        FORBIDDEN
        & set(fused.columns)
    )

    require(
        not leaked,
        (
            "Forbidden reference/evaluation columns "
            f"in fused input: {leaked}"
        ),
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
        "correction_applied",
        "correction_reason",
        "reranked_top1_tile_id",
        "reranked_top1_verifier_score",
        "reranked_top1_hybrid_score",
        "reranked_top1_inliers",
        "map_crs",
    }

    missing = sorted(
        required_fused
        - set(fused.columns)
    )

    require(
        not missing,
        (
            "Fused trajectory missing columns: "
            f"{missing}"
        ),
    )

    fused["query_id"] = pd.to_numeric(
        fused["query_id"],
        errors="raise",
    ).astype(int)

    fused["sequence_frame_id"] = (
        pd.to_numeric(
            fused["sequence_frame_id"],
            errors="raise",
        )
        .astype(int)
    )

    fused = (
        fused.sort_values(
            "sequence_frame_id",
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    # =====================================================
    # Verify blind fusion contract.
    # =====================================================

    fusion_report = json.loads(
        fusion_report_path.read_text(
            encoding="utf-8"
        )
    )

    require(
        fusion_report.get("status")
        == "PASS_BLIND_TEMPORAL_FUSION",
        (
            "Fusion report is not a passing "
            "blind temporal fusion report."
        ),
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
        "ground_truth_used_for_decisions",
    ]:
        require(
            blind_contract.get(key)
            is False,
            (
                "Blind fusion contract violated: "
                f"{key}="
                f"{blind_contract.get(key)!r}"
            ),
        )

    lock_frame = int(
        fusion_report[
            "bootstrap"
        ][
            "lock_frame"
        ]
    )

    # =====================================================
    # DINO evidence.
    #
    # Read ONLY blind-safe query-summary columns.
    # =====================================================

    qsum = pd.read_csv(
        qsum_path,
        usecols=[
            "query_id",
            "original_top1_tile_id",
            "original_top1_dino_score",
        ],
    )

    qsum["query_id"] = pd.to_numeric(
        qsum["query_id"],
        errors="raise",
    ).astype(int)

    require(
        not qsum["query_id"].duplicated().any(),
        "DINO query summary has duplicate query IDs.",
    )

    out = fused.merge(
        qsum,
        on="query_id",
        how="left",
        validate="one_to_one",
    )

    require(
        len(out) == 403,
        (
            "DINO evidence join changed "
            "trajectory row count."
        ),
    )

    require(
        not out[
            "original_top1_tile_id"
        ].isna().any(),
        (
            "Missing DINO Top-1 evidence "
            "for one or more queries."
        ),
    )

    # =====================================================
    # EPSG:3346 -> WGS84.
    # =====================================================

    try:
        from pyproj import Transformer
    except Exception as exc:
        raise RuntimeError(
            "pyproj is required for Add-on 9 "
            "estimated lat/lon export."
        ) from exc

    source_crs_values = (
        out["map_crs"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    require(
        all(
            value == args.source_crs
            for value in source_crs_values
        ),
        (
            "Unexpected map CRS in fused trajectory: "
            f"{source_crs_values}"
        ),
    )

    available = bool_series(
        out[
            "map_aligned_available"
        ]
    ).to_numpy(bool)

    map_x = pd.to_numeric(
        out["estimated_map_x"],
        errors="coerce",
    ).to_numpy(float)

    map_y = pd.to_numeric(
        out["estimated_map_y"],
        errors="coerce",
    ).to_numpy(float)

    require(
        np.isnan(
            map_x[:lock_frame]
        ).all()
        and np.isnan(
            map_y[:lock_frame]
        ).all(),
        "Pre-lock map coordinates are exposed.",
    )

    require(
        np.isfinite(
            map_x[available]
        ).all()
        and np.isfinite(
            map_y[available]
        ).all(),
        (
            "Available map positions contain "
            "non-finite values."
        ),
    )

    transformer = Transformer.from_crs(
        args.source_crs,
        args.target_crs,
        always_xy=True,
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

    lon, lat = transformer.transform(
        map_x[available],
        map_y[available],
    )

    transform_s = (
        time.perf_counter()
        - transform_start
    )

    estimated_lon[
        available
    ] = np.asarray(
        lon,
        dtype=float,
    )

    estimated_lat[
        available
    ] = np.asarray(
        lat,
        dtype=float,
    )

    require(
        np.isfinite(
            estimated_lon[
                available
            ]
        ).all(),
        "Non-finite estimated longitude.",
    )

    require(
        np.isfinite(
            estimated_lat[
                available
            ]
        ).all(),
        "Non-finite estimated latitude.",
    )

    require(
        (
            (
                estimated_lat[
                    available
                ]
                >= -90.0
            )
            & (
                estimated_lat[
                    available
                ]
                <= 90.0
            )
        ).all(),
        "Estimated latitude outside valid range.",
    )

    require(
        (
            (
                estimated_lon[
                    available
                ]
                >= -180.0
            )
            & (
                estimated_lon[
                    available
                ]
                <= 180.0
            )
        ).all(),
        "Estimated longitude outside valid range.",
    )

    out[
        "estimated_lat"
    ] = estimated_lat

    out[
        "estimated_lon"
    ] = estimated_lon

    # =====================================================
    # Required confidence/correction evidence.
    #
    # confidence_score is NOT a probability.
    # It is the existing ORB-hybrid ranking score.
    # =====================================================

    out[
        "confidence_score"
    ] = pd.to_numeric(
        out[
            "reranked_top1_hybrid_score"
        ],
        errors="coerce",
    )

    out[
        "accepted_correction"
    ] = bool_series(
        out[
            "correction_applied"
        ]
    )

    map_lock_event = np.zeros(
        len(out),
        dtype=bool,
    )

    map_lock_event[
        lock_frame
    ] = True

    out[
        "map_lock_event"
    ] = map_lock_event

    correction_source = np.empty(
        len(out),
        dtype=object,
    )

    correction_source[:] = None

    for i in range(
        len(out)
    ):
        if not available[i]:
            correction_source[i] = None

        elif i == lock_frame:
            correction_source[i] = (
                "blind_map_bootstrap"
            )

        elif bool(
            out.iloc[i][
                "accepted_correction"
            ]
        ):
            correction_source[i] = (
                "orb_temporal_soft_correction"
            )

        else:
            correction_source[i] = (
                "relative_propagation"
            )

    out[
        "correction_source"
    ] = correction_source

    out[
        "dino_top1_tile_id"
    ] = out[
        "original_top1_tile_id"
    ]

    out[
        "dino_top1_score"
    ] = pd.to_numeric(
        out[
            "original_top1_dino_score"
        ],
        errors="coerce",
    )

    out[
        "orb_selected_tile_id"
    ] = out[
        "reranked_top1_tile_id"
    ]

    out[
        "orb_score"
    ] = pd.to_numeric(
        out[
            "reranked_top1_verifier_score"
        ],
        errors="coerce",
    )

    out[
        "orb_inliers"
    ] = pd.to_numeric(
        out[
            "reranked_top1_inliers"
        ],
        errors="coerce",
    )

    # =====================================================
    # Runtime evidence.
    #
    # These are measured benchmark AVERAGES,
    # repeated for each row. They are not claimed
    # as individual per-frame stopwatch measurements.
    # =====================================================

    timing = None

    if timing_path is not None:
        timing = pd.read_csv(
            timing_path
        )

    relative_ms = timing_value(
        timing,
        "relative_odometry",
    )

    retrieval_ms = timing_value(
        timing,
        "dino_retrieval_against_map_cache",
    )

    rerank_ms = timing_value(
        timing,
        "orb_topk_reranking",
        prefer_secondary=True,
    )

    out[
        "runtime_relative_ms"
    ] = (
        relative_ms
        if relative_ms is not None
        else np.nan
    )

    out[
        "runtime_retrieval_ms"
    ] = (
        retrieval_ms
        if retrieval_ms is not None
        else np.nan
    )

    out[
        "runtime_rerank_ms"
    ] = (
        rerank_ms
        if rerank_ms is not None
        else np.nan
    )

    # =====================================================
    # Optional blind-safe map-coverage sanity.
    # =====================================================

    map_coverage = None

    if tile_index_path is not None:
        tiles = pd.read_csv(
            tile_index_path,
            usecols=[
                "left_easting",
                "right_easting",
                "bottom_northing",
                "top_northing",
            ],
        )

        xmin = float(
            tiles[
                "left_easting"
            ].min()
        )

        xmax = float(
            tiles[
                "right_easting"
            ].max()
        )

        ymin = float(
            tiles[
                "bottom_northing"
            ].min()
        )

        ymax = float(
            tiles[
                "top_northing"
            ].max()
        )

        inside = (
            available
            & (map_x >= xmin)
            & (map_x <= xmax)
            & (map_y >= ymin)
            & (map_y <= ymax)
        )

        map_coverage = {
            "tile_bbox_epsg3346": {
                "xmin": xmin,
                "xmax": xmax,
                "ymin": ymin,
                "ymax": ymax,
            },
            "available_positions": int(
                available.sum()
            ),
            "inside_tile_index_bbox": int(
                inside.sum()
            ),
            "outside_tile_index_bbox": int(
                available.sum()
                - inside.sum()
            ),
        }

    # =====================================================
    # Final submission schema.
    # =====================================================

    # Extra identity/provenance columns are deliberate
    # so optional evaluation can attach by query_id.
    submission_columns = [
        "frame_index",
        "sequence_frame_id",
        "query_id",
        "token0_id",
        "timestamp_s",
        "image_path",

        "relative_x_m",
        "relative_y_m",

        "estimated_map_x",
        "estimated_map_y",
        "estimated_lat",
        "estimated_lon",

        "map_aligned_available",
        "map_lock_event",

        "confidence_score",
        "accepted_correction",
        "correction_source",

        "dino_top1_tile_id",
        "dino_top1_score",
        "orb_selected_tile_id",
        "orb_score",
        "orb_inliers",

        "runtime_relative_ms",
        "runtime_retrieval_ms",
        "runtime_rerank_ms",
    ]

    submission = out[
        submission_columns
    ].copy()

    missing_required = sorted(
        set(
            REQUIRED_OUTPUT_COLUMNS
        )
        - set(
            submission.columns
        )
    )

    require(
        not missing_required,
        (
            "Final submission missing required "
            f"columns: {missing_required}"
        ),
    )

    leaked = sorted(
        FORBIDDEN
        & set(
            submission.columns
        )
    )

    require(
        not leaked,
        (
            "Final submission contains forbidden "
            f"columns: {leaked}"
        ),
    )

    # Causal contract:
    # no retrospective lat/lon.
    require(
        submission.loc[
            ~available,
            [
                "estimated_map_x",
                "estimated_map_y",
                "estimated_lat",
                "estimated_lon",
            ],
        ].isna().all().all(),
        (
            "Pre-lock geographic estimates "
            "were backfilled."
        ),
    )

    # =====================================================
    # Save.
    # =====================================================

    trajectory_dir = (
        run_root
        / "trajectories"
    )

    report_dir = (
        run_root
        / "reports/addon9_estimated_latlon"
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

    write_start = time.perf_counter()

    submission.to_csv(
        submission_path,
        index=False,
    )

    write_s = (
        time.perf_counter()
        - write_start
    )

    hashes_after = {
        "fused_trajectory":
            sha256_file(fused_path),

        "fusion_report":
            sha256_file(fusion_report_path),

        "absolute_query_summary":
            sha256_file(qsum_path),
    }

    if timing_path is not None:
        hashes_after[
            "timing_summary"
        ] = sha256_file(
            timing_path
        )

    if tile_index_path is not None:
        hashes_after[
            "tile_index"
        ] = sha256_file(
            tile_index_path
        )

    require(
        input_hashes_before
        == hashes_after,
        (
            "One or more frozen inputs changed "
            "during Add-on 9."
        ),
    )

    stage_s = (
        time.perf_counter()
        - stage_start
    )

    post = submission.loc[
        available
    ]

    accepted_count = int(
        submission[
            "accepted_correction"
        ].sum()
    )

    report = {
        "stage": (
            "ADDON9_ESTIMATED_LATLON_EXPORT"
        ),
        "status": (
            "PASS_ADDON9_ESTIMATED_LATLON_EXPORT"
        ),
        "required_label": REQUIRED_LABEL,
        "blind_contract": {
            "gps_used": False,
            "srt_used": False,
            "reference_used": False,
            "oracle_used": False,
            "evaluation_error_used": False,
            "prelock_backfill_performed": False,
            "estimated_latlon_are_output_not_input": True,
        },
        "coordinate_conversion": {
            "source_crs": args.source_crs,
            "target_crs": args.target_crs,
            "always_xy": True,
            "library": "pyproj",
        },
        "confidence_contract": {
            "confidence_score_source": (
                "reranked_top1_hybrid_score"
            ),
            "confidence_score_is_probability": False,
            "note": (
                "Existing ORB hybrid ranking score "
                "is exported as confidence evidence; "
                "it is not calibrated as probability."
            ),
        },
        "runtime_contract": {
            "runtime_columns_are": (
                "measured benchmark averages "
                "repeated per row"
            ),
            "runtime_relative_ms": relative_ms,
            "runtime_retrieval_ms": retrieval_ms,
            "runtime_rerank_ms": rerank_ms,
        },
        "rows": {
            "total": int(
                len(submission)
            ),
            "prelock_without_map_position": int(
                (~available).sum()
            ),
            "map_aligned_available": int(
                available.sum()
            ),
            "estimated_latlon_available": int(
                post[
                    "estimated_lat"
                ].notna().sum()
            ),
            "accepted_corrections": (
                accepted_count
            ),
            "map_lock_events": int(
                submission[
                    "map_lock_event"
                ].sum()
            ),
        },
        "geographic_range": {
            "estimated_lat_min": float(
                post[
                    "estimated_lat"
                ].min()
            ),
            "estimated_lat_max": float(
                post[
                    "estimated_lat"
                ].max()
            ),
            "estimated_lon_min": float(
                post[
                    "estimated_lon"
                ].min()
            ),
            "estimated_lon_max": float(
                post[
                    "estimated_lon"
                ].max()
            ),
        },
        "map_coverage_sanity": (
            map_coverage
        ),
        "runtime": {
            "coordinate_transform_s": float(
                transform_s
            ),
            "output_write_s": float(
                write_s
            ),
            "total_stage_wall_s": float(
                stage_s
            ),
        },
        "inputs_sha256": (
            input_hashes_before
        ),
        "outputs": {
            "submission_estimated_trajectory":
                str(
                    submission_path
                ),
            "report":
                str(
                    report_path
                ),
        },
        "blind_mode_note": (
            "Without GT/reference, estimated "
            "lat/lon are exported but accuracy "
            "is not computed by this stage."
        ),
    }

    report_path.write_text(
        json.dumps(
            report,
            indent=2,
            default=json_safe,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    print("=" * 80)
    print(
        "ADD-ON 9 — ESTIMATED LAT/LON EXPORT"
    )
    print("=" * 80)

    print()
    print(REQUIRED_LABEL)

    print()
    print("Blind contract")
    print("-" * 80)
    print(
        "GPS used                   : false"
    )
    print(
        "SRT used                   : false"
    )
    print(
        "reference used             : false"
    )
    print(
        "pre-lock backfill          : false"
    )

    print()
    print("Availability")
    print("-" * 80)
    print(
        "total rows                 :",
        len(submission),
    )
    print(
        "pre-lock unavailable       :",
        int((~available).sum()),
    )
    print(
        "map positions available    :",
        int(available.sum()),
    )
    print(
        "estimated lat/lon available:",
        int(
            post[
                "estimated_lat"
            ].notna().sum()
        ),
    )
    print(
        "accepted corrections       :",
        accepted_count,
    )

    print()
    print("Estimated geographic range")
    print("-" * 80)
    print(
        "latitude                   :",
        (
            f"{post['estimated_lat'].min():.8f}"
            " .. "
            f"{post['estimated_lat'].max():.8f}"
        ),
    )
    print(
        "longitude                  :",
        (
            f"{post['estimated_lon'].min():.8f}"
            " .. "
            f"{post['estimated_lon'].max():.8f}"
        ),
    )

    print()
    print("Runtime evidence")
    print("-" * 80)
    print(
        "relative average           :",
        relative_ms,
        "ms",
    )
    print(
        "retrieval average          :",
        retrieval_ms,
        "ms",
    )
    print(
        "ORB rerank average         :",
        rerank_ms,
        "ms/query",
    )
    print(
        "export stage               :",
        f"{stage_s:.6f} s",
    )

    if map_coverage is not None:
        print()
        print("Map coverage sanity")
        print("-" * 80)
        print(
            "inside tile-index bbox    :",
            map_coverage[
                "inside_tile_index_bbox"
            ],
        )
        print(
            "outside tile-index bbox   :",
            map_coverage[
                "outside_tile_index_bbox"
            ],
        )

    print()
    print("Saved")
    print("-" * 80)
    print(submission_path)
    print(report_path)

    print()
    print(
        "status: "
        "PASS_ADDON9_ESTIMATED_LATLON_EXPORT"
    )


if __name__ == "__main__":
    main()
