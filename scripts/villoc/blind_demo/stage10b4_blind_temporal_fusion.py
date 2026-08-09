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


FORBIDDEN_BLIND_COLUMNS = {
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
    "lat",
    "lon",
    "latitude",
    "longitude",
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


def select_reranked_candidate(
    qsum: pd.DataFrame,
    cand: pd.DataFrame,
) -> pd.DataFrame:

    q = qsum.copy()
    c = cand.copy()

    q["query_id"] = pd.to_numeric(
        q["query_id"],
        errors="raise",
    ).astype(int)

    c["query_id"] = pd.to_numeric(
        c["query_id"],
        errors="raise",
    ).astype(int)

    selected = q.merge(
        c,
        left_on=[
            "query_id",
            "reranked_top1_tile_id",
        ],
        right_on=[
            "query_id",
            "tile_id",
        ],
        how="left",
        validate="one_to_one",
    )

    require(
        not selected["tile_id"].isna().any(),
        (
            "Could not recover every ORB-reranked "
            "Top-1 candidate."
        ),
    )

    return selected


def build_strict_gates(
    selected: pd.DataFrame,
) -> pd.DataFrame:

    x = selected.copy()

    homography = bool_series(
        x["homography_ok"]
    )

    inliers = pd.to_numeric(
        x["reranked_top1_inliers"],
        errors="coerce",
    )

    ratio = pd.to_numeric(
        x["reranked_top1_inlier_ratio"],
        errors="coerce",
    )

    coverage = pd.to_numeric(
        x[
            "reranked_top1_query_inlier_coverage"
        ],
        errors="coerce",
    )

    rank = pd.to_numeric(
        x[
            "reranked_top1_original_rank"
        ],
        errors="coerce",
    )

    x["strict_a_blind"] = (
        homography
        & (inliers >= 25)
        & (ratio >= 0.30)
        & (coverage >= 0.10)
        & (rank <= 15)
    )

    x["strict_b_blind"] = (
        homography
        & (inliers >= 40)
        & (ratio >= 0.35)
        & (coverage >= 0.12)
        & (rank <= 12)
    )

    return x


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Blind-safe temporal consistency gating "
            "and soft relative-absolute fusion."
        )
    )

    parser.add_argument(
        "--map-trajectory",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--bootstrap-report",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--absolute-query-summary",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--absolute-candidate-scores",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--run-root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--max-residual-m",
        type=float,
        default=50.0,
    )

    parser.add_argument(
        "--residual-ratio",
        type=float,
        default=0.30,
    )

    parser.add_argument(
        "--min-gap-m",
        type=float,
        default=30.0,
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=0.25,
    )

    args = parser.parse_args()

    require(
        0.0 < args.alpha <= 1.0,
        "--alpha must be in (0, 1].",
    )

    require(
        args.max_residual_m > 0,
        "--max-residual-m must be positive.",
    )

    require(
        args.residual_ratio >= 0,
        "--residual-ratio must be non-negative.",
    )

    require(
        args.min_gap_m >= 0,
        "--min-gap-m must be non-negative.",
    )

    stage_start = time.perf_counter()

    map_path = (
        args.map_trajectory
        .expanduser()
        .resolve()
    )

    bootstrap_path = (
        args.bootstrap_report
        .expanduser()
        .resolve()
    )

    qsum_path = (
        args.absolute_query_summary
        .expanduser()
        .resolve()
    )

    cand_path = (
        args.absolute_candidate_scores
        .expanduser()
        .resolve()
    )

    run_root = (
        args.run_root
        .expanduser()
        .resolve()
    )

    for path in [
        map_path,
        bootstrap_path,
        qsum_path,
        cand_path,
    ]:
        require(
            path.exists(),
            f"Missing input: {path}",
        )

    hashes_before = {
        "map_trajectory": sha256_file(
            map_path
        ),
        "bootstrap_report": sha256_file(
            bootstrap_path
        ),
        "absolute_query_summary": sha256_file(
            qsum_path
        ),
        "absolute_candidate_scores": sha256_file(
            cand_path
        ),
    }

    # =====================================================
    # Read Stage 10B.3 causal map trajectory.
    # =====================================================

    traj = pd.read_csv(map_path)

    required_traj = {
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
        "reference_used",
    }

    missing = sorted(
        required_traj
        - set(traj.columns)
    )

    require(
        not missing,
        (
            "Map trajectory missing required "
            f"columns: {missing}"
        ),
    )

    leaked = sorted(
        FORBIDDEN_BLIND_COLUMNS
        & set(traj.columns)
    )

    require(
        not leaked,
        (
            "Reference/evaluation leakage already "
            f"present in input map trajectory: {leaked}"
        ),
    )

    require(
        len(traj) > 0,
        "Input trajectory is empty.",
    )

    traj["sequence_frame_id"] = pd.to_numeric(
        traj["sequence_frame_id"],
        errors="raise",
    ).astype(int)

    traj["query_id"] = pd.to_numeric(
        traj["query_id"],
        errors="raise",
    ).astype(int)

    traj["token0_id"] = pd.to_numeric(
        traj["token0_id"],
        errors="raise",
    ).astype(int)

    traj = (
        traj.sort_values(
            "sequence_frame_id",
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    expected = np.arange(
        len(traj),
        dtype=int,
    )

    require(
        np.array_equal(
            traj[
                "sequence_frame_id"
            ].to_numpy(int),
            expected,
        ),
        (
            "sequence_frame_id must be "
            "contiguous from zero."
        ),
    )

    # =====================================================
    # Read frozen bootstrap.
    # =====================================================

    bootstrap = json.loads(
        bootstrap_path.read_text(
            encoding="utf-8"
        )
    )

    bootstrap_status = bootstrap.get(
        "status"
    )

    require(
        bootstrap_status
        in {
            "PASS_BLIND_MAP_BOOTSTRAP",
            "PASS_BLIND_MAP_BOOTSTRAP_NO_LOCK",
        },
        (
            "Bootstrap report is not a valid "
            "blind bootstrap outcome: "
            f"{bootstrap_status!r}"
        ),
    )

    contract = bootstrap.get(
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
            contract.get(key)
            is False,
            (
                "Bootstrap contract violation: "
                f"{key}={contract.get(key)!r}"
            ),
        )

    # =====================================================
    # No-map-lock continuation.
    #
    # Temporal fusion operates in metric map coordinates.
    # Without a trusted visual-to-map bootstrap there is
    # no valid metric residual, map displacement or map
    # correction to compute.
    #
    # Therefore:
    #   - preserve visual-relative motion,
    #   - apply zero absolute corrections,
    #   - expose zero map-aligned rows,
    #   - do not fabricate metric/map coordinates.
    # =====================================================

    if (
        bootstrap_status
        == "PASS_BLIND_MAP_BOOTSTRAP_NO_LOCK"
    ):
        require(
            bootstrap.get(
                "localization_state"
            )
            == "NO_TRUSTED_ABSOLUTE_LOCK",
            (
                "No-lock bootstrap has unexpected "
                "localization state."
            ),
        )

        require(
            bootstrap.get(
                "map_lock"
            )
            is None,
            (
                "No-lock bootstrap unexpectedly "
                "contains a map lock."
            ),
        )

        require(
            not bool_series(
                traj[
                    "map_aligned_available"
                ]
            ).any(),
            (
                "No-lock Stage10B3 trajectory "
                "contains map-aligned rows."
            ),
        )

        require(
            not bool_series(
                traj[
                    "reference_used"
                ]
            ).any(),
            (
                "No-lock trajectory unexpectedly "
                "uses reference information."
            ),
        )

        unavailable_cols = [
            "relative_x_m",
            "relative_y_m",
            "relative_cumulative_distance_m",
            "estimated_map_x",
            "estimated_map_y",
        ]

        for col in unavailable_cols:
            require(
                pd.to_numeric(
                    traj[col],
                    errors="coerce",
                )
                .isna()
                .all(),
                (
                    "No-lock trajectory contains "
                    f"unexpected metric/map values: {col}"
                ),
            )

        visual_x = pd.to_numeric(
            traj[
                "visual_x_px"
            ],
            errors="raise",
        ).to_numpy(float)

        visual_y = pd.to_numeric(
            traj[
                "visual_y_px"
            ],
            errors="raise",
        ).to_numpy(float)

        require(
            np.isfinite(
                visual_x
            ).all()
            and np.isfinite(
                visual_y
            ).all(),
            (
                "Visual-relative trajectory contains "
                "non-finite coordinates."
            ),
        )

        visual_xy = np.column_stack(
            [
                visual_x,
                visual_y,
            ]
        )

        if len(
            visual_xy
        ) >= 2:
            visual_steps = np.linalg.norm(
                np.diff(
                    visual_xy,
                    axis=0,
                ),
                axis=1,
            )

            visual_path_length_px = float(
                visual_steps.sum()
            )

        else:
            visual_path_length_px = 0.0

        df = traj.copy()

        # No absolute/map correction can exist.
        df[
            "strict_a_blind"
        ] = False

        df[
            "strict_b_blind"
        ] = False

        df[
            "correction_candidate"
        ] = False

        df[
            "correction_accepted"
        ] = False

        df[
            "correction_applied"
        ] = False

        df[
            "correction_reason"
        ] = (
            "skipped_no_trusted_map_lock"
        )

        for col in [
            "temporal_residual_m",
            "temporal_threshold_m",
            "distance_since_anchor_m",
            "anchor_token0_id_before",
            "correction_delta_easting_m",
            "correction_delta_northing_m",
            "correction_magnitude_m",
            "relative_map_x",
            "relative_map_y",
        ]:
            df[col] = np.nan

        # Fusion alpha is a configured policy value,
        # but no fusion was actually executed.
        df[
            "fusion_alpha"
        ] = np.nan

        df[
            "temporal_policy"
        ] = (
            "not_applicable_no_map_lock"
        )

        df[
            "map_estimate_source"
        ] = (
            "unavailable_no_trusted_absolute_lock"
        )

        # =================================================
        # Save stable Stage10B4 outputs.
        # =================================================

        metadata_dir = (
            run_root
            / "metadata/blind_temporal_fusion"
        )

        trajectory_dir = (
            run_root
            / "trajectories"
        )

        report_dir = (
            run_root
            / "reports/blind_temporal_fusion"
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

        trajectory_path = (
            trajectory_dir
            / "blind_temporal_fused_trajectory.csv"
        )

        report_path = (
            report_dir
            / "blind_temporal_fusion_report.json"
        )

        manifest_cols = [
            "sequence_frame_id",
            "query_id",
            "token0_id",
            "timestamp_s",
            "map_aligned_available",
            "strict_a_blind",
            "strict_b_blind",
            "correction_candidate",
            "correction_accepted",
            "correction_applied",
            "correction_reason",
        ]

        trajectory_cols = [
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
            "step_motion_px",
            "relative_x_m",
            "relative_y_m",
            "relative_cumulative_distance_m",
            "relative_map_x",
            "relative_map_y",
            "estimated_map_x",
            "estimated_map_y",
            "map_aligned_available",
            "initialization_state",
            "map_crs",
            "correction_candidate",
            "correction_accepted",
            "correction_applied",
            "correction_reason",
            "temporal_residual_m",
            "temporal_threshold_m",
            "distance_since_anchor_m",
            "anchor_token0_id_before",
            "correction_delta_easting_m",
            "correction_delta_northing_m",
            "correction_magnitude_m",
            "fusion_alpha",
            "temporal_policy",
            "map_estimate_source",
        ]

        manifest_cols = [
            col
            for col in manifest_cols
            if col in df.columns
        ]

        trajectory_cols = [
            col
            for col in trajectory_cols
            if col in df.columns
        ]

        manifest_out = df[
            manifest_cols
        ].copy()

        trajectory_out = df[
            trajectory_cols
        ].copy()

        for name, out_df in [
            (
                "correction manifest",
                manifest_out,
            ),
            (
                "relative-only trajectory",
                trajectory_out,
            ),
        ]:
            leaked = sorted(
                FORBIDDEN_BLIND_COLUMNS
                & set(
                    out_df.columns
                )
            )

            require(
                not leaked,
                (
                    f"{name} contains forbidden "
                    f"columns: {leaked}"
                ),
            )

        write_start = time.perf_counter()

        manifest_out.to_csv(
            manifest_path,
            index=False,
        )

        trajectory_out.to_csv(
            trajectory_path,
            index=False,
        )

        write_s = (
            time.perf_counter()
            - write_start
        )

        hashes_after = {
            "map_trajectory":
                sha256_file(
                    map_path
                ),
            "bootstrap_report":
                sha256_file(
                    bootstrap_path
                ),
            "absolute_query_summary":
                sha256_file(
                    qsum_path
                ),
            "absolute_candidate_scores":
                sha256_file(
                    cand_path
                ),
        }

        require(
            hashes_before
            == hashes_after,
            (
                "Frozen Stage10B4 inputs changed "
                "during no-lock continuation."
            ),
        )

        stage_s = (
            time.perf_counter()
            - stage_start
        )

        report = {
            "stage": (
                "STAGE_10B4_BLIND_TEMPORAL_FUSION"
            ),
            "status": (
                "PASS_BLIND_TEMPORAL_FUSION_"
                "SKIPPED_NO_MAP_LOCK"
            ),
            "localization_state": (
                "NO_TRUSTED_ABSOLUTE_LOCK"
            ),
            "trajectory_state": (
                "RELATIVE_VISUAL_ONLY"
            ),
            "fusion_state": (
                "NOT_APPLICABLE_NO_MAP_LOCK"
            ),
            "no_lock_reason": (
                bootstrap.get(
                    "no_lock_reason"
                )
            ),
            "blind_contract": {
                "gps_used": False,
                "srt_used": False,
                "reference_used": False,
                "oracle_used": False,
                "evaluation_error_used": False,
                "ground_truth_used_for_decisions": False,
                "map_coordinates_from_visual_matching": False,
                "prelock_backfill_performed": False,
            },
            "policy": {
                "configured_alpha": float(
                    args.alpha
                ),
                "temporal_fusion_executed": False,
                "reason": (
                    "Metric temporal fusion requires "
                    "a trusted causal map lock."
                ),
            },
            "availability": {
                "relative_visual_available": True,
                "metric_relative_available": False,
                "map_alignment_available": False,
                "absolute_coordinates_available": False,
                "temporal_fusion_available": False,
            },
            "counts": {
                "rows": int(
                    len(df)
                ),
                "map_aligned_rows": 0,
                "correction_candidates": 0,
                "accepted_corrections": 0,
                "applied_corrections": 0,
            },
            "visual_relative_summary": {
                "first_frame": int(
                    df.iloc[0][
                        "sequence_frame_id"
                    ]
                ),
                "last_frame": int(
                    df.iloc[-1][
                        "sequence_frame_id"
                    ]
                ),
                "first_timestamp_s": float(
                    df.iloc[0][
                        "timestamp_s"
                    ]
                ),
                "last_timestamp_s": float(
                    df.iloc[-1][
                        "timestamp_s"
                    ]
                ),
                "visual_path_length_px": (
                    visual_path_length_px
                ),
            },
            "input_sha256": hashes_before,
            "runtime": {
                "output_write_s": float(
                    write_s
                ),
                "total_stage_wall_s": float(
                    stage_s
                ),
            },
            "outputs": {
                "correction_manifest": str(
                    manifest_path
                ),
                "trajectory": str(
                    trajectory_path
                ),
                "report": str(
                    report_path
                ),
            },
            "important_note": (
                "Temporal fusion was correctly "
                "skipped because no trusted absolute "
                "map lock exists. The blind visual-"
                "relative trajectory remains available."
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
            "STAGE 10B.4 — TEMPORAL FUSION "
            "NO-LOCK CONTINUATION"
        )
        print("=" * 80)

        print()
        print("Localization state")
        print("-" * 80)

        print(
            "state                   : "
            "NO_TRUSTED_ABSOLUTE_LOCK"
        )

        print(
            "trajectory              : "
            "RELATIVE_VISUAL_ONLY"
        )

        print(
            "temporal fusion         : skipped"
        )

        print(
            "reason                  :",
            bootstrap.get(
                "no_lock_reason"
            ),
        )

        print()
        print("Availability")
        print("-" * 80)

        print(
            "relative visual         : available"
        )

        print(
            "metric relative         : unavailable"
        )

        print(
            "map alignment           : unavailable"
        )

        print(
            "absolute coordinates    : unavailable"
        )

        print(
            "accepted corrections    : 0"
        )

        print()
        print("Trajectory")
        print("-" * 80)

        print(
            "rows                    :",
            len(df),
        )

        print(
            "visual path length      :",
            f"{visual_path_length_px:.3f} px",
        )

        print()
        print("Saved")
        print("-" * 80)

        print(manifest_path)
        print(trajectory_path)
        print(report_path)

        print()
        print(
            "status: "
            "PASS_BLIND_TEMPORAL_FUSION_"
            "SKIPPED_NO_MAP_LOCK"
        )

        return

    lock = bootstrap.get(
        "map_lock"
    )

    require(
        lock is not None,
        "Bootstrap report contains no map lock.",
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
        0 <= lock_frame < len(traj),
        "Invalid bootstrap lock frame.",
    )

    # =====================================================
    # Recover ORB-selected map candidates.
    #
    # IMPORTANT:
    # No error, hit, oracle or reference columns are read.
    # =====================================================

    qsum = pd.read_csv(
        qsum_path,
        usecols=[
            "query_id",
            "reranked_top1_tile_id",
            "reranked_top1_original_rank",
            "reranked_top1_inliers",
            "reranked_top1_inlier_ratio",
            "reranked_top1_query_inlier_coverage",
            "reranked_top1_verifier_score",
            "reranked_top1_hybrid_score",
        ],
    )

    cand = pd.read_csv(
        cand_path,
        usecols=[
            "query_id",
            "tile_id",
            "center_easting",
            "center_northing",
            "left_easting",
            "right_easting",
            "bottom_northing",
            "top_northing",
            "homography_ok",
        ],
    )

    selected = select_reranked_candidate(
        qsum,
        cand,
    )

    selected = build_strict_gates(
        selected
    )

    correction_cols = [
        "query_id",
        "reranked_top1_tile_id",
        "center_easting",
        "center_northing",
        "left_easting",
        "right_easting",
        "bottom_northing",
        "top_northing",
        "reranked_top1_original_rank",
        "reranked_top1_inliers",
        "reranked_top1_inlier_ratio",
        "reranked_top1_query_inlier_coverage",
        "reranked_top1_verifier_score",
        "reranked_top1_hybrid_score",
        "strict_a_blind",
        "strict_b_blind",
    ]

    df = traj.merge(
        selected[
            correction_cols
        ],
        on="query_id",
        how="left",
        validate="one_to_one",
    )

    require(
        len(df) == len(traj),
        "Absolute evidence join changed row count.",
    )

    require(
        not df[
            "center_easting"
        ].isna().any(),
        (
            "Some trajectory rows have no selected "
            "absolute candidate."
        ),
    )

    # =====================================================
    # Establish causal availability.
    # =====================================================

    available = bool_series(
        df[
            "map_aligned_available"
        ]
    ).to_numpy(bool)

    require(
        available[
            lock_frame
        ],
        (
            "Map trajectory is unavailable at the "
            "declared bootstrap lock frame."
        ),
    )

    require(
        not available[
            :lock_frame
        ].any(),
        (
            "Pre-lock map positions were exposed "
            "causally."
        ),
    )

    require(
        available[
            lock_frame:
        ].all(),
        (
            "Map trajectory has unavailable rows "
            "after lock."
        ),
    )

    require(
        bool(
            df.iloc[
                lock_frame
            ][
                "strict_b_blind"
            ]
        ),
        (
            "The bootstrap lock frame is not a "
            "strict-B absolute observation."
        ),
    )

    base_xy = df[
        [
            "estimated_map_x",
            "estimated_map_y",
        ]
    ].to_numpy(float)

    relative_cumulative = pd.to_numeric(
        df[
            "relative_cumulative_distance_m"
        ],
        errors="coerce",
    ).to_numpy(float)

    abs_xy = df[
        [
            "center_easting",
            "center_northing",
        ]
    ].to_numpy(float)

    strict_a = bool_series(
        df["strict_a_blind"]
    ).to_numpy(bool)

    # =====================================================
    # Blind temporal gate.
    #
    # Same selected Villoc policy family:
    #
    #   candidate = strict A
    #   bootstrap = strict B
    #   residual floor = 50 m
    #   residual ratio = 0.30
    #   minimum gap = 30 m
    #
    # Difference from old F3:
    # rel_xy is now causal map-aligned relative
    # trajectory, NOT GT/prefix-aligned ENU.
    # =====================================================

    candidate = np.zeros(
        len(df),
        dtype=bool,
    )

    accepted = np.zeros(
        len(df),
        dtype=bool,
    )

    residual = np.full(
        len(df),
        np.nan,
        dtype=float,
    )

    threshold = np.full(
        len(df),
        np.nan,
        dtype=float,
    )

    distance_since_anchor = np.full(
        len(df),
        np.nan,
        dtype=float,
    )

    anchor_before = np.full(
        len(df),
        np.nan,
        dtype=float,
    )

    reason = np.array(
        [
            "prelock_unavailable"
            if i < lock_frame
            else "not_candidate"
            for i in range(
                len(df)
            )
        ],
        dtype=object,
    )

    # Map lock itself is the established bootstrap anchor.
    last_anchor = lock_frame

    reason[
        lock_frame
    ] = "bootstrap_map_lock_anchor"

    residual[
        lock_frame
    ] = 0.0

    threshold[
        lock_frame
    ] = np.inf

    distance_since_anchor[
        lock_frame
    ] = 0.0

    for i in range(
        lock_frame + 1,
        len(df),
    ):
        if not strict_a[i]:
            continue

        candidate[i] = True

        anchor_before[i] = float(
            df.iloc[
                last_anchor
            ][
                "token0_id"
            ]
        )

        distance_since = float(
            relative_cumulative[i]
            - relative_cumulative[
                last_anchor
            ]
        )

        distance_since_anchor[
            i
        ] = distance_since

        if (
            distance_since
            < args.min_gap_m
        ):
            reason[i] = (
                "gap_too_short"
            )
            continue

        relative_delta = (
            base_xy[i]
            - base_xy[
                last_anchor
            ]
        )

        absolute_delta = (
            abs_xy[i]
            - abs_xy[
                last_anchor
            ]
        )

        relative_delta_norm = float(
            np.linalg.norm(
                relative_delta
            )
        )

        r = float(
            np.linalg.norm(
                absolute_delta
                - relative_delta
            )
        )

        t = float(
            max(
                args.max_residual_m,
                args.residual_ratio
                * max(
                    relative_delta_norm,
                    1e-9,
                ),
            )
        )

        residual[i] = r
        threshold[i] = t

        if r <= t:
            accepted[i] = True
            reason[i] = (
                "temporal_agreement_accept"
            )

            last_anchor = i

        else:
            reason[i] = (
                "temporal_disagreement_reject"
            )

    # =====================================================
    # Blind fusion replay.
    #
    # Use the continuously propagated relative delta.
    # Accepted map anchors pull the current state by alpha.
    # =====================================================

    fused_xy = np.full(
        (
            len(df),
            2,
        ),
        np.nan,
        dtype=float,
    )

    correction_applied = np.zeros(
        len(df),
        dtype=bool,
    )

    correction_dx = np.full(
        len(df),
        np.nan,
        dtype=float,
    )

    correction_dy = np.full(
        len(df),
        np.nan,
        dtype=float,
    )

    correction_magnitude = np.full(
        len(df),
        np.nan,
        dtype=float,
    )

    fused_xy[
        lock_frame
    ] = base_xy[
        lock_frame
    ]

    correction_dx[
        lock_frame
    ] = 0.0

    correction_dy[
        lock_frame
    ] = 0.0

    correction_magnitude[
        lock_frame
    ] = 0.0

    for i in range(
        lock_frame + 1,
        len(df),
    ):
        relative_step = (
            base_xy[i]
            - base_xy[
                i - 1
            ]
        )

        predicted = (
            fused_xy[
                i - 1
            ]
            + relative_step
        )

        fused_xy[i] = predicted

        if accepted[i]:
            innovation = (
                abs_xy[i]
                - predicted
            )

            correction_dx[i] = float(
                args.alpha
                * innovation[0]
            )

            correction_dy[i] = float(
                args.alpha
                * innovation[1]
            )

            correction_magnitude[i] = float(
                args.alpha
                * np.linalg.norm(
                    innovation
                )
            )

            fused_xy[i] = (
                (
                    1.0
                    - args.alpha
                )
                * predicted
                + args.alpha
                * abs_xy[i]
            )

            correction_applied[
                i
            ] = True

    # =====================================================
    # Build blind outputs.
    # =====================================================

    df[
        "correction_candidate"
    ] = candidate

    df[
        "correction_accepted"
    ] = accepted

    df[
        "correction_applied"
    ] = correction_applied

    df[
        "temporal_residual_m"
    ] = residual

    df[
        "temporal_threshold_m"
    ] = threshold

    df[
        "distance_since_anchor_m"
    ] = distance_since_anchor

    df[
        "anchor_token0_id_before"
    ] = anchor_before

    df[
        "correction_reason"
    ] = reason

    df[
        "correction_delta_easting_m"
    ] = correction_dx

    df[
        "correction_delta_northing_m"
    ] = correction_dy

    df[
        "correction_magnitude_m"
    ] = correction_magnitude

    # Preserve Stage 10B.3 map estimate as the
    # uncorrected relative baseline.
    df[
        "relative_map_x"
    ] = df[
        "estimated_map_x"
    ]

    df[
        "relative_map_y"
    ] = df[
        "estimated_map_y"
    ]

    # Final Stage 10B.4 map estimate.
    df[
        "estimated_map_x"
    ] = fused_xy[:, 0]

    df[
        "estimated_map_y"
    ] = fused_xy[:, 1]

    df[
        "fusion_alpha"
    ] = float(
        args.alpha
    )

    df[
        "temporal_policy"
    ] = (
        "blind_temporal_a_bootb_"
        "res50_r030_gap30"
    )

    df[
        "map_estimate_source"
    ] = np.where(
        available,
        (
            "blind_relative_plus_"
            "temporal_absolute_fusion"
        ),
        "unavailable_prelock",
    )

    # Pre-lock must stay unavailable.
    require(
        df.loc[
            :lock_frame - 1,
            [
                "estimated_map_x",
                "estimated_map_y",
            ],
        ].isna().all().all(),
        (
            "Fusion exposed pre-lock map "
            "coordinates."
        ),
    )

    # Post-lock final map trajectory must be finite.
    require(
        np.isfinite(
            df.loc[
                lock_frame:,
                [
                    "estimated_map_x",
                    "estimated_map_y",
                ],
            ].to_numpy(float)
        ).all(),
        (
            "Post-lock fused trajectory "
            "contains non-finite positions."
        ),
    )

    # Nothing may be accepted/applied before lock.
    require(
        not accepted[
            :lock_frame + 1
        ].any(),
        (
            "Temporal correction accepted at or "
            "before bootstrap lock."
        ),
    )

    require(
        not correction_applied[
            :lock_frame + 1
        ].any(),
        (
            "Correction applied at or before "
            "bootstrap lock."
        ),
    )

    # Accepted means candidate + strong enough gap
    # + temporal agreement.
    accepted_indices = np.flatnonzero(
        accepted
    )

    for i in accepted_indices:
        require(
            candidate[i],
            (
                "Accepted correction is not a "
                "candidate."
            ),
        )

        require(
            distance_since_anchor[i]
            >= args.min_gap_m
            - 1e-9,
            (
                "Accepted correction violates "
                "minimum gap."
            ),
        )

        require(
            residual[i]
            <= threshold[i]
            + 1e-9,
            (
                "Accepted correction violates "
                "temporal threshold."
            ),
        )

    # =====================================================
    # Save compact correction manifest.
    # =====================================================

    metadata_dir = (
        run_root
        / "metadata/blind_temporal_fusion"
    )

    trajectory_dir = (
        run_root
        / "trajectories"
    )

    report_dir = (
        run_root
        / "reports/blind_temporal_fusion"
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

    trajectory_path = (
        trajectory_dir
        / "blind_temporal_fused_trajectory.csv"
    )

    report_path = (
        report_dir
        / "blind_temporal_fusion_report.json"
    )

    manifest_cols = [
        "sequence_frame_id",
        "query_id",
        "token0_id",
        "timestamp_s",
        "map_aligned_available",
        "strict_a_blind",
        "strict_b_blind",
        "reranked_top1_tile_id",
        "center_easting",
        "center_northing",
        "reranked_top1_original_rank",
        "reranked_top1_inliers",
        "reranked_top1_inlier_ratio",
        "reranked_top1_query_inlier_coverage",
        "reranked_top1_verifier_score",
        "reranked_top1_hybrid_score",
        "correction_candidate",
        "correction_accepted",
        "temporal_residual_m",
        "temporal_threshold_m",
        "distance_since_anchor_m",
        "anchor_token0_id_before",
        "correction_reason",
    ]

    trajectory_cols = [
        "frame_index",
        "timestamp_s",
        "image_path",
        "sequence_frame_id",
        "query_id",
        "token0_id",
        "relative_x_m",
        "relative_y_m",
        "relative_cumulative_distance_m",
        "relative_map_x",
        "relative_map_y",
        "estimated_map_x",
        "estimated_map_y",
        "map_aligned_available",
        "initialization_state",
        "map_crs",
        "strict_a_blind",
        "strict_b_blind",
        "reranked_top1_tile_id",
        "center_easting",
        "center_northing",
        "reranked_top1_original_rank",
        "reranked_top1_inliers",
        "reranked_top1_inlier_ratio",
        "reranked_top1_query_inlier_coverage",
        "reranked_top1_verifier_score",
        "reranked_top1_hybrid_score",
        "correction_candidate",
        "correction_accepted",
        "correction_applied",
        "correction_reason",
        "temporal_residual_m",
        "temporal_threshold_m",
        "distance_since_anchor_m",
        "anchor_token0_id_before",
        "correction_delta_easting_m",
        "correction_delta_northing_m",
        "correction_magnitude_m",
        "fusion_alpha",
        "temporal_policy",
        "map_estimate_source",
    ]

    manifest_out = df[
        manifest_cols
    ].copy()

    trajectory_out = df[
        trajectory_cols
    ].copy()

    for name, out_df in [
        (
            "correction manifest",
            manifest_out,
        ),
        (
            "fused trajectory",
            trajectory_out,
        ),
    ]:
        leaked = sorted(
            FORBIDDEN_BLIND_COLUMNS
            & set(
                out_df.columns
            )
        )

        require(
            not leaked,
            (
                f"{name} contains forbidden "
                f"columns: {leaked}"
            ),
        )

    write_start = time.perf_counter()

    manifest_out.to_csv(
        manifest_path,
        index=False,
    )

    trajectory_out.to_csv(
        trajectory_path,
        index=False,
    )

    write_s = (
        time.perf_counter()
        - write_start
    )

    hashes_after = {
        "map_trajectory": sha256_file(
            map_path
        ),
        "bootstrap_report": sha256_file(
            bootstrap_path
        ),
        "absolute_query_summary": sha256_file(
            qsum_path
        ),
        "absolute_candidate_scores": sha256_file(
            cand_path
        ),
    }

    require(
        hashes_before == hashes_after,
        (
            "Frozen Stage 10B.4 inputs changed "
            "during execution."
        ),
    )

    stage_s = (
        time.perf_counter()
        - stage_start
    )

    candidate_post = int(
        candidate[
            lock_frame + 1:
        ].sum()
    )

    accepted_post = int(
        accepted.sum()
    )

    rejected_temporal = int(
        (
            reason
            == "temporal_disagreement_reject"
        ).sum()
    )

    rejected_gap = int(
        (
            reason
            == "gap_too_short"
        ).sum()
    )

    accepted_rows = df.loc[
        accepted
    ].copy()

    finite_residuals = pd.to_numeric(
        accepted_rows[
            "temporal_residual_m"
        ],
        errors="coerce",
    ).dropna()

    finite_corrections = pd.to_numeric(
        df.loc[
            correction_applied,
            "correction_magnitude_m",
        ],
        errors="coerce",
    ).dropna()

    report = {
        "stage": (
            "STAGE_10B4_BLIND_TEMPORAL_FUSION"
        ),
        "status": (
            "PASS_BLIND_TEMPORAL_FUSION"
        ),
        "blind_contract": {
            "gps_used": False,
            "srt_used": False,
            "reference_used": False,
            "oracle_used": False,
            "evaluation_error_used": False,
            "ground_truth_used_for_decisions": False,
            "map_coordinates_from_visual_matching": True,
            "prelock_backfill_performed": False,
        },
        "policy": {
            "name": (
                "blind_temporal_a_bootb_"
                "res50_r030_gap30"
            ),
            "candidate_gate": (
                "strict_a_blind"
            ),
            "bootstrap_gate": (
                "strict_b_blind"
            ),
            "bootstrap_source": (
                "Stage 10B.2 causal map lock"
            ),
            "max_residual_m": float(
                args.max_residual_m
            ),
            "residual_ratio": float(
                args.residual_ratio
            ),
            "minimum_gap_m": float(
                args.min_gap_m
            ),
            "alpha": float(
                args.alpha
            ),
        },
        "bootstrap": {
            "lock_frame": lock_frame,
            "lock_timestamp_s": lock_time,
            "lock_query_id": int(
                df.iloc[
                    lock_frame
                ]["query_id"]
            ),
            "lock_tile_id": str(
                df.iloc[
                    lock_frame
                ][
                    "reranked_top1_tile_id"
                ]
            ),
        },
        "counts": {
            "rows": int(
                len(df)
            ),
            "strict_a_total": int(
                bool_series(
                    df[
                        "strict_a_blind"
                    ]
                ).sum()
            ),
            "strict_b_total": int(
                bool_series(
                    df[
                        "strict_b_blind"
                    ]
                ).sum()
            ),
            "postlock_temporal_candidates": (
                candidate_post
            ),
            "accepted_corrections": (
                accepted_post
            ),
            "gap_rejections": (
                rejected_gap
            ),
            "temporal_rejections": (
                rejected_temporal
            ),
        },
        "accepted_summary": {
            "first_accepted_frame": (
                int(
                    accepted_rows.iloc[
                        0
                    ][
                        "sequence_frame_id"
                    ]
                )
                if len(
                    accepted_rows
                )
                else None
            ),
            "first_accepted_time_s": (
                float(
                    accepted_rows.iloc[
                        0
                    ][
                        "timestamp_s"
                    ]
                )
                if len(
                    accepted_rows
                )
                else None
            ),
            "last_accepted_frame": (
                int(
                    accepted_rows.iloc[
                        -1
                    ][
                        "sequence_frame_id"
                    ]
                )
                if len(
                    accepted_rows
                )
                else None
            ),
            "last_accepted_time_s": (
                float(
                    accepted_rows.iloc[
                        -1
                    ][
                        "timestamp_s"
                    ]
                )
                if len(
                    accepted_rows
                )
                else None
            ),
            "median_temporal_residual_m": (
                float(
                    finite_residuals.median()
                )
                if len(
                    finite_residuals
                )
                else None
            ),
            "median_applied_correction_m": (
                float(
                    finite_corrections.median()
                )
                if len(
                    finite_corrections
                )
                else None
            ),
            "max_applied_correction_m": (
                float(
                    finite_corrections.max()
                )
                if len(
                    finite_corrections
                )
                else None
            ),
        },
        "runtime": {
            "temporal_and_fusion_compute_s": float(
                stage_s - write_s
            ),
            "output_write_s": float(
                write_s
            ),
            "total_stage_wall_s": float(
                stage_s
            ),
        },
        "inputs_sha256": hashes_before,
        "outputs": {
            "correction_manifest": str(
                manifest_path
            ),
            "fused_trajectory": str(
                trajectory_path
            ),
            "report": str(
                report_path
            ),
        },
        "important_note": (
            "Accepted correction count is not "
            "required to equal historical S8.F3 "
            "because the relative coordinate source "
            "is now causal and blind-safe."
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
        "STAGE 10B.4 — BLIND TEMPORAL FUSION"
    )
    print("=" * 80)

    print()
    print("Blind contract")
    print("-" * 80)
    print(
        "GPS used                 : false"
    )
    print(
        "SRT used                 : false"
    )
    print(
        "reference used           : false"
    )
    print(
        "oracle used              : false"
    )
    print(
        "evaluation error used    : false"
    )
    print(
        "pre-lock backfill        : false"
    )

    print()
    print("Frozen policy")
    print("-" * 80)
    print(
        "candidate gate           : strict A"
    )
    print(
        "bootstrap gate           : strict B"
    )
    print(
        "residual floor           :",
        f"{args.max_residual_m:.1f} m",
    )
    print(
        "residual ratio           :",
        f"{args.residual_ratio:.2f}",
    )
    print(
        "minimum gap              :",
        f"{args.min_gap_m:.1f} m",
    )
    print(
        "fusion alpha             :",
        f"{args.alpha:.2f}",
    )

    print()
    print("Bootstrap")
    print("-" * 80)
    print(
        "lock frame               :",
        lock_frame,
    )
    print(
        "lock time                :",
        f"{lock_time:.3f} s",
    )
    print(
        "lock tile                :",
        df.iloc[
            lock_frame
        ][
            "reranked_top1_tile_id"
        ],
    )

    print()
    print("Temporal gating")
    print("-" * 80)
    print(
        "post-lock candidates     :",
        candidate_post,
    )
    print(
        "accepted corrections     :",
        accepted_post,
    )
    print(
        "gap rejections           :",
        rejected_gap,
    )
    print(
        "temporal rejections      :",
        rejected_temporal,
    )

    if len(
        accepted_rows
    ):
        print(
            "first accepted          :",
            (
                f"frame "
                f"{int(accepted_rows.iloc[0]['sequence_frame_id'])}, "
                f"{float(accepted_rows.iloc[0]['timestamp_s']):.3f} s"
            ),
        )

        print(
            "last accepted           :",
            (
                f"frame "
                f"{int(accepted_rows.iloc[-1]['sequence_frame_id'])}, "
                f"{float(accepted_rows.iloc[-1]['timestamp_s']):.3f} s"
            ),
        )

    print()
    print("Fusion")
    print("-" * 80)

    print(
        "map poses available      :",
        int(
            available.sum()
        ),
    )

    print(
        "corrections applied      :",
        int(
            correction_applied.sum()
        ),
    )

    if len(
        finite_corrections
    ):
        print(
            "median correction       :",
            f"{float(finite_corrections.median()):.3f} m",
        )

        print(
            "max correction          :",
            f"{float(finite_corrections.max()):.3f} m",
        )

    print()
    print("Runtime")
    print("-" * 80)

    print(
        "total stage              :",
        f"{stage_s:.6f} s",
    )

    print()
    print("Saved")
    print("-" * 80)
    print(manifest_path)
    print(trajectory_path)
    print(report_path)

    print()
    print(
        "status: PASS_BLIND_TEMPORAL_FUSION"
    )


if __name__ == "__main__":
    main()
