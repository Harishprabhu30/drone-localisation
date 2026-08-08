'''
ROOT=outputs/villoc/traj01_90deg_stable120m
RUN="$ROOT/blind_demo_addons/eval_traj01_known_gt"

python scripts/villoc/blind_demo/stage10b3_apply_blind_map_lock.py \
  --blind-manifest \
  "$RUN/metadata/blind_query_manifest.csv" \
  --raw-relative \
  "$RUN/metadata/s8_xfeat_relative_frontend/s8r4_xfeat_relative_trajectory_blind_raw.csv" \
  --bootstrap-report \
  "$RUN/reports/blind_map_bootstrap/blind_map_bootstrap_report.json" \
  --run-root "$RUN" \
  --map-crs EPSG:3346 \
  2>&1 | tee \
  "$RUN/logs/stage10b3_apply_blind_map_lock.log"
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


FORBIDDEN_OUTPUT_COLUMNS = {
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
    "lat",
    "lon",
    "latitude",
    "longitude",
    "gps_lat",
    "gps_lon",
    "ground_truth_x",
    "ground_truth_y",
    "fusion_error_m",
    "abs_error_m_eval_only",
}


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


def false_only(series: pd.Series) -> bool:
    if pd.api.types.is_bool_dtype(series):
        return bool((~series.fillna(True)).all())

    values = (
        series.astype(str)
        .str.strip()
        .str.lower()
    )

    return bool(
        values.isin(
            {
                "false",
                "0",
                "no",
            }
        ).all()
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


def trajectory_length(xy: np.ndarray) -> float:
    if len(xy) < 2:
        return 0.0

    step = np.linalg.norm(
        np.diff(
            xy,
            axis=0,
        ),
        axis=1,
    )

    return float(
        np.sum(step)
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Apply a frozen blind visual-to-map bootstrap "
            "transform to the raw XFeat trajectory."
        )
    )

    parser.add_argument(
        "--blind-manifest",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--raw-relative",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--bootstrap-report",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--run-root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--map-crs",
        default="EPSG:3346",
    )

    args = parser.parse_args()

    stage_start = time.perf_counter()

    blind_path = (
        args.blind_manifest
        .expanduser()
        .resolve()
    )

    relative_path = (
        args.raw_relative
        .expanduser()
        .resolve()
    )

    bootstrap_path = (
        args.bootstrap_report
        .expanduser()
        .resolve()
    )

    run_root = (
        args.run_root
        .expanduser()
        .resolve()
    )

    for path in [
        blind_path,
        relative_path,
        bootstrap_path,
    ]:
        require(
            path.exists(),
            f"Missing input: {path}",
        )

    input_hashes_before = {
        "blind_manifest": sha256_file(
            blind_path
        ),
        "raw_relative": sha256_file(
            relative_path
        ),
        "bootstrap_report": sha256_file(
            bootstrap_path
        ),
    }

    # =====================================================
    # Read ONLY blind-safe manifest columns.
    # =====================================================

    manifest = pd.read_csv(
        blind_path,
        usecols=[
            "frame_index",
            "timestamp_s",
            "image_path",
            "sequence_frame_id",
            "query_id",
            "token0_id",
            "reference_available",
        ],
    )

    require(
        len(manifest) > 0,
        "Blind manifest is empty.",
    )

    require(
        false_only(
            manifest[
                "reference_available"
            ]
        ),
        (
            "Blind manifest unexpectedly contains "
            "reference_available=true."
        ),
    )

    raw = pd.read_csv(
        relative_path,
        usecols=[
            "sequence_frame_id",
            "token0_id",
            "visual_x_px",
            "visual_y_px",
            "visual_yaw_rad",
            "visual_yaw_deg_unwrapped",
            "step_x_local_px",
            "step_y_local_px",
            "step_x_global_px",
            "step_y_global_px",
            "step_motion_px",
            "pair_safe_image_only",
            "coordinate_contract",
            "reference_used",
        ],
    )

    require(
        len(raw) > 0,
        "Raw relative trajectory is empty.",
    )

    require(
        false_only(
            raw["reference_used"]
        ),
        (
            "Raw relative trajectory is not "
            "reference-free."
        ),
    )

    require(
        (
            raw["coordinate_contract"]
            .astype(str)
            == "relative_visual_image_only"
        ).all(),
        (
            "Unexpected raw relative coordinate "
            "contract."
        ),
    )

    bootstrap = json.loads(
        bootstrap_path.read_text(
            encoding="utf-8"
        )
    )

    require(
        bootstrap.get("status")
        == "PASS_BLIND_MAP_BOOTSTRAP",
        (
            "Bootstrap report is not a passing "
            "blind map lock."
        ),
    )

    blind_contract = bootstrap.get(
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
            blind_contract.get(key)
            is False,
            (
                "Bootstrap blind contract violated: "
                f"{key}={blind_contract.get(key)!r}"
            ),
        )

    lock = bootstrap.get(
        "map_lock"
    )

    require(
        lock is not None,
        "Bootstrap report has no map_lock.",
    )

    required_transform = [
        "a_real",
        "a_imag",
        "b_real",
        "b_imag",
        "scale_m_per_visual_px",
        "rotation_deg",
        "lock_timestamp_s",
        "lock_sequence_frame_id",
    ]

    missing_transform = [
        key
        for key in required_transform
        if key not in lock
    ]

    require(
        not missing_transform,
        (
            "Bootstrap map_lock missing fields: "
            f"{missing_transform}"
        ),
    )

    # =====================================================
    # Identity join.
    # =====================================================

    manifest[
        "sequence_frame_id"
    ] = pd.to_numeric(
        manifest[
            "sequence_frame_id"
        ],
        errors="raise",
    ).astype(int)

    manifest[
        "token0_id"
    ] = pd.to_numeric(
        manifest[
            "token0_id"
        ],
        errors="raise",
    ).astype(int)

    raw[
        "sequence_frame_id"
    ] = pd.to_numeric(
        raw[
            "sequence_frame_id"
        ],
        errors="raise",
    ).astype(int)

    raw[
        "token0_id"
    ] = pd.to_numeric(
        raw[
            "token0_id"
        ],
        errors="raise",
    ).astype(int)

    out = manifest.merge(
        raw,
        on=[
            "sequence_frame_id",
            "token0_id",
        ],
        how="inner",
        validate="one_to_one",
    )

    out = (
        out.sort_values(
            "sequence_frame_id",
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    require(
        len(out)
        == len(manifest)
        == len(raw),
        (
            "Manifest/raw trajectory row-count "
            "mismatch after identity join."
        ),
    )

    expected_frames = np.arange(
        len(out),
        dtype=int,
    )

    require(
        np.array_equal(
            out[
                "sequence_frame_id"
            ].to_numpy(int),
            expected_frames,
        ),
        (
            "sequence_frame_id is not contiguous "
            "from zero."
        ),
    )

    # =====================================================
    # Frozen visual -> EPSG:3346 similarity.
    #
    # complex form:
    #   map = a * visual + b
    #
    # a = ar + i*ai
    #
    # E = ar*x - ai*y + br
    # N = ai*x + ar*y + bi
    # =====================================================

    ar = float(
        lock["a_real"]
    )

    ai = float(
        lock["a_imag"]
    )

    br = float(
        lock["b_real"]
    )

    bi = float(
        lock["b_imag"]
    )

    visual_x = pd.to_numeric(
        out["visual_x_px"],
        errors="raise",
    ).to_numpy(float)

    visual_y = pd.to_numeric(
        out["visual_y_px"],
        errors="raise",
    ).to_numpy(float)

    transformed_easting = (
        ar * visual_x
        - ai * visual_y
        + br
    )

    transformed_northing = (
        ai * visual_x
        + ar * visual_y
        + bi
    )

    lock_frame = int(
        lock[
            "lock_sequence_frame_id"
        ]
    )

    lock_time = float(
        lock[
            "lock_timestamp_s"
        ]
    )

    require(
        0 <= lock_frame < len(out),
        (
            "Bootstrap lock frame is outside "
            "trajectory bounds."
        ),
    )

    actual_lock_time = float(
        out.loc[
            out["sequence_frame_id"]
            == lock_frame,
            "timestamp_s",
        ].iloc[0]
    )

    require(
        abs(
            actual_lock_time
            - lock_time
        )
        <= 1e-6,
        (
            "Bootstrap lock timestamp does not "
            "match blind manifest timestamp."
        ),
    )

    available = (
        out[
            "sequence_frame_id"
        ].to_numpy(int)
        >= lock_frame
    )

    # Online contract:
    # do not expose retrospectively transformed
    # pre-lock coordinates.
    out[
        "map_aligned_available"
    ] = available

    out[
        "initialization_state"
    ] = np.where(
        available,
        "map_locked",
        "acquiring",
    )

    out[
        "estimated_map_x"
    ] = np.nan

    out[
        "estimated_map_y"
    ] = np.nan

    out.loc[
        available,
        "estimated_map_x",
    ] = transformed_easting[
        available
    ]

    out.loc[
        available,
        "estimated_map_y",
    ] = transformed_northing[
        available
    ]

    # =====================================================
    # Metric, map-oriented relative displacement.
    #
    # This intentionally starts at (0,0) at the causal
    # lock frame. It is NOT reference ENU.
    # =====================================================

    lock_easting = float(
        transformed_easting[
            lock_frame
        ]
    )

    lock_northing = float(
        transformed_northing[
            lock_frame
        ]
    )

    metric_rel_x = (
        transformed_easting
        - lock_easting
    )

    metric_rel_y = (
        transformed_northing
        - lock_northing
    )

    out["relative_x_m"] = np.nan
    out["relative_y_m"] = np.nan

    out.loc[
        available,
        "relative_x_m",
    ] = metric_rel_x[
        available
    ]

    out.loc[
        available,
        "relative_y_m",
    ] = metric_rel_y[
        available
    ]

    # Blind-safe cumulative relative distance
    # starting from map lock.
    post_xy = np.column_stack([
        metric_rel_x[
            available
        ],
        metric_rel_y[
            available
        ],
    ])

    if len(post_xy) > 0:
        post_step = np.linalg.norm(
            np.diff(
                post_xy,
                axis=0,
            ),
            axis=1,
        )

        post_cumulative = np.concatenate([
            [0.0],
            np.cumsum(
                post_step
            ),
        ])
    else:
        post_cumulative = np.asarray(
            [],
            dtype=float,
        )

    out[
        "relative_cumulative_distance_m"
    ] = np.nan

    out.loc[
        available,
        "relative_cumulative_distance_m",
    ] = post_cumulative

    # Provenance / contract columns.
    out["map_crs"] = args.map_crs

    out[
        "map_alignment_source"
    ] = np.where(
        available,
        "blind_visual_to_map_bootstrap",
        "unavailable_prelock",
    )

    out[
        "bootstrap_transform_frozen"
    ] = True

    out[
        "bootstrap_lock_frame"
    ] = lock_frame

    out[
        "bootstrap_lock_time_s"
    ] = lock_time

    out[
        "bootstrap_scale_m_per_visual_px"
    ] = float(
        lock[
            "scale_m_per_visual_px"
        ]
    )

    out[
        "bootstrap_rotation_deg"
    ] = float(
        lock[
            "rotation_deg"
        ]
    )

    # =====================================================
    # Leakage guard.
    # =====================================================

    leaked = sorted(
        FORBIDDEN_OUTPUT_COLUMNS
        & set(out.columns)
    )

    require(
        not leaked,
        (
            "Reference/evaluation fields leaked "
            "into blind map trajectory: "
            f"{leaked}"
        ),
    )

    prelock = (
        ~out[
            "map_aligned_available"
        ].astype(bool)
    )

    require(
        out.loc[
            prelock,
            [
                "estimated_map_x",
                "estimated_map_y",
                "relative_x_m",
                "relative_y_m",
            ],
        ].isna().all().all(),
        (
            "Pre-lock coordinates were "
            "retrospectively exposed."
        ),
    )

    postlock = (
        out[
            "map_aligned_available"
        ].astype(bool)
    )

    require(
        np.isfinite(
            out.loc[
                postlock,
                [
                    "estimated_map_x",
                    "estimated_map_y",
                    "relative_x_m",
                    "relative_y_m",
                ],
            ].to_numpy(float)
        ).all(),
        (
            "Post-lock map trajectory contains "
            "non-finite coordinates."
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
        / "reports/blind_map_alignment"
    )

    trajectory_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    out_path = (
        trajectory_dir
        / "blind_map_aligned_relative_trajectory.csv"
    )

    report_path = (
        report_dir
        / "blind_map_alignment_report.json"
    )

    # Keep the output intentionally compact and
    # blind-safe.
    front = [
        "frame_index",
        "timestamp_s",
        "image_path",
        "sequence_frame_id",
        "query_id",
        "token0_id",
        "visual_x_px",
        "visual_y_px",
        "visual_yaw_rad",
        "visual_yaw_deg_unwrapped",
        "relative_x_m",
        "relative_y_m",
        "relative_cumulative_distance_m",
        "estimated_map_x",
        "estimated_map_y",
        "map_aligned_available",
        "initialization_state",
        "map_crs",
        "map_alignment_source",
        "bootstrap_transform_frozen",
        "bootstrap_lock_frame",
        "bootstrap_lock_time_s",
        "bootstrap_scale_m_per_visual_px",
        "bootstrap_rotation_deg",
        "step_motion_px",
        "pair_safe_image_only",
        "coordinate_contract",
        "reference_used",
    ]

    front = [
        col
        for col in front
        if col in out.columns
    ]

    out = out[front].copy()

    write_start = time.perf_counter()

    out.to_csv(
        out_path,
        index=False,
    )

    write_seconds = (
        time.perf_counter()
        - write_start
    )

    input_hashes_after = {
        "blind_manifest": sha256_file(
            blind_path
        ),
        "raw_relative": sha256_file(
            relative_path
        ),
        "bootstrap_report": sha256_file(
            bootstrap_path
        ),
    }

    require(
        input_hashes_before
        == input_hashes_after,
        (
            "One or more frozen blind inputs "
            "changed during Stage 10B.3."
        ),
    )

    stage_seconds = (
        time.perf_counter()
        - stage_start
    )

    post = out[
        out[
            "map_aligned_available"
        ].astype(bool)
    ].copy()

    post_map_xy = post[
        [
            "estimated_map_x",
            "estimated_map_y",
        ]
    ].to_numpy(float)

    report = {
        "stage": (
            "STAGE_10B3_APPLY_BLIND_MAP_LOCK"
        ),
        "status": (
            "PASS_BLIND_MAP_ALIGNED_RELATIVE_TRAJECTORY"
        ),
        "purpose": (
            "Apply the frozen blind visual-to-map "
            "bootstrap transform to the reference-free "
            "XFeat trajectory."
        ),
        "blind_contract": {
            "gps_used": False,
            "srt_used": False,
            "reference_used": False,
            "oracle_used": False,
            "evaluation_error_used": False,
            "prelock_backfill_performed": False,
            "map_coordinates_from_visual_matching": True,
        },
        "coordinate_contract": {
            "raw_relative": (
                "relative_visual_image_only"
            ),
            "metric_relative": (
                "map_oriented_metric_displacement_"
                "from_online_lock"
            ),
            "map_coordinate_system": (
                args.map_crs
            ),
            "map_trajectory_type": (
                "bootstrap_aligned_relative_baseline"
            ),
            "postlock_corrections_applied": False,
        },
        "bootstrap": {
            "lock_frame": lock_frame,
            "lock_timestamp_s": lock_time,
            "scale_m_per_visual_px": float(
                lock[
                    "scale_m_per_visual_px"
                ]
            ),
            "rotation_deg": float(
                lock[
                    "rotation_deg"
                ]
            ),
            "lock_estimated_easting_m": (
                lock_easting
            ),
            "lock_estimated_northing_m": (
                lock_northing
            ),
            "unique_anchor_count": int(
                lock[
                    "unique_anchor_count"
                ]
            ),
            "support": int(
                lock.get(
                    "refined_support",
                    lock["support"],
                )
            ),
        },
        "rows": {
            "total": int(
                len(out)
            ),
            "prelock_unavailable": int(
                prelock.sum()
            ),
            "postlock_available": int(
                postlock.sum()
            ),
        },
        "postlock_summary": {
            "first_frame": int(
                post.iloc[0][
                    "sequence_frame_id"
                ]
            ),
            "last_frame": int(
                post.iloc[-1][
                    "sequence_frame_id"
                ]
            ),
            "first_timestamp_s": float(
                post.iloc[0][
                    "timestamp_s"
                ]
            ),
            "last_timestamp_s": float(
                post.iloc[-1][
                    "timestamp_s"
                ]
            ),
            "relative_path_length_m": float(
                trajectory_length(
                    post[
                        [
                            "relative_x_m",
                            "relative_y_m",
                        ]
                    ].to_numpy(float)
                )
            ),
            "estimated_map_x_min": float(
                post[
                    "estimated_map_x"
                ].min()
            ),
            "estimated_map_x_max": float(
                post[
                    "estimated_map_x"
                ].max()
            ),
            "estimated_map_y_min": float(
                post[
                    "estimated_map_y"
                ].min()
            ),
            "estimated_map_y_max": float(
                post[
                    "estimated_map_y"
                ].max()
            ),
        },
        "input_sha256": (
            input_hashes_before
        ),
        "runtime": {
            "trajectory_build_s": float(
                stage_seconds
                - write_seconds
            ),
            "output_write_s": float(
                write_seconds
            ),
            "total_stage_wall_s": float(
                stage_seconds
            ),
        },
        "outputs": {
            "trajectory": str(
                out_path
            ),
            "report": str(
                report_path
            ),
        },
        "important_note": (
            "This is a blind map-aligned relative "
            "baseline after online map lock. "
            "Post-lock sparse absolute corrections "
            "have not yet been replayed."
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
        "STAGE 10B.3 — APPLY BLIND MAP LOCK"
    )
    print("=" * 80)

    print()
    print("Blind contract")
    print("-" * 80)
    print("GPS used                : false")
    print("SRT used                : false")
    print("reference used          : false")
    print("oracle used             : false")
    print("pre-lock backfill       : false")

    print()
    print("Bootstrap")
    print("-" * 80)
    print(
        "lock frame              :",
        lock_frame,
    )
    print(
        "lock time               :",
        f"{lock_time:.3f} s",
    )
    print(
        "scale                   :",
        f"{float(lock['scale_m_per_visual_px']):.6f} m/px",
    )
    print(
        "rotation                :",
        f"{float(lock['rotation_deg']):.3f} deg",
    )
    print(
        "lock estimated EPSG3346 :",
        f"({lock_easting:.3f}, "
        f"{lock_northing:.3f})",
    )

    print()
    print("Availability")
    print("-" * 80)
    print(
        "total rows              :",
        len(out),
    )
    print(
        "pre-lock unavailable    :",
        int(prelock.sum()),
    )
    print(
        "post-lock available     :",
        int(postlock.sum()),
    )

    print()
    print("Post-lock trajectory")
    print("-" * 80)
    print(
        "frames                  :",
        f"{int(post.iloc[0]['sequence_frame_id'])}"
        " .. "
        f"{int(post.iloc[-1]['sequence_frame_id'])}",
    )
    print(
        "relative path length    :",
        f"{report['postlock_summary']['relative_path_length_m']:.3f} m",
    )
    print(
        "map X range             :",
        f"{report['postlock_summary']['estimated_map_x_min']:.3f}"
        " .. "
        f"{report['postlock_summary']['estimated_map_x_max']:.3f}",
    )
    print(
        "map Y range             :",
        f"{report['postlock_summary']['estimated_map_y_min']:.3f}"
        " .. "
        f"{report['postlock_summary']['estimated_map_y_max']:.3f}",
    )

    print()
    print("Runtime")
    print("-" * 80)
    print(
        "total stage             :",
        f"{stage_seconds:.6f} s",
    )

    print()
    print("Saved")
    print("-" * 80)
    print(out_path)
    print(report_path)

    print()
    print(
        "status: "
        "PASS_BLIND_MAP_ALIGNED_RELATIVE_TRAJECTORY"
    )


if __name__ == "__main__":
    main()
