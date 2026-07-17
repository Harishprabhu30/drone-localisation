#!/usr/bin/env python3
"""S6B.0 — Build the absolute correction manifest.

Joins:
    S5C.3 online confidence evidence
    S5C.2 selected-candidate evidence
    S6A sequence/frame mapping
    S6A official relative trajectory
    Satellite tile coordinates

Reference coordinates and error labels remain evaluation-only.
No correction replay is performed in this stage.

command Used:

export PYTHONPATH=$PWD/src

python scripts/satloc/s6b/s6b_0_absolute_correction_manifest.py \
  2>&1 | tee \
  outputs/satloc/reports/s6b_relative_absolute/s6b0_absolute_correction_manifest.log

"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ABSOLUTE_COORDINATE_ERROR_TOLERANCE_M = 10.0

DEFAULT_FEATURES = Path(
    "outputs/satloc/metadata/s5c_temporal/"
    "s5c3_lightglue_confidence_features_full263.csv"
)
DEFAULT_GATES = Path(
    "outputs/satloc/metadata/s5c_temporal/"
    "s5c3_recommended_confidence_gates_full263.csv"
)
DEFAULT_QUERY_SUMMARY = Path(
    "outputs/satloc/metadata/s5c_temporal/"
    "s5c2_lightglue_union_query_summary_top50_full263.csv"
)
DEFAULT_CANDIDATES = Path(
    "outputs/satloc/metadata/s5c_temporal/"
    "s5c2_lightglue_union_candidate_scores_top50_full263.csv"
)
DEFAULT_SEQUENCE = Path(
    "outputs/satloc/metadata/s6a_relative_motion/"
    "s6a_sequence_manifest.csv"
)
DEFAULT_TRAJECTORY = Path(
    "outputs/satloc/metadata/s6a_relative_motion/"
    "s6a2_orb_relative_trajectory_aligned_eval_only.csv"
)
DEFAULT_SATELLITE_INDEX = Path(
    "outputs/satloc/metadata/satellite_tiles_index_enriched.csv"
)

DEFAULT_OUT_DIR = Path(
    "outputs/satloc/metadata/s6b_relative_absolute"
)
DEFAULT_REPORT_DIR = Path(
    "outputs/satloc/reports/s6b_relative_absolute"
)
DEFAULT_FIGURE_DIR = Path(
    "outputs/satloc/figures/s6b_relative_absolute"
)

OFFICIAL_RELATIVE_VARIANT = "se2_scale_normalized"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--gates", type=Path, default=DEFAULT_GATES)
    parser.add_argument(
        "--query-summary",
        type=Path,
        default=DEFAULT_QUERY_SUMMARY,
    )
    parser.add_argument(
        "--candidate-scores",
        type=Path,
        default=DEFAULT_CANDIDATES,
    )
    parser.add_argument(
        "--sequence-manifest",
        type=Path,
        default=DEFAULT_SEQUENCE,
    )
    parser.add_argument(
        "--relative-trajectory",
        type=Path,
        default=DEFAULT_TRAJECTORY,
    )
    parser.add_argument(
        "--satellite-index",
        type=Path,
        default=DEFAULT_SATELLITE_INDEX,
    )
    parser.add_argument(
        "--relative-variant",
        default=OFFICIAL_RELATIVE_VARIANT,
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=DEFAULT_REPORT_DIR,
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=DEFAULT_FIGURE_DIR,
    )
    return parser.parse_args()


def require_columns(
    df: pd.DataFrame,
    required: list[str],
    name: str,
) -> None:
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")


def bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)

    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes", "y"})
    )


def json_safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if pd.isna(value):
        return None
    return value


def safe_name(value: Any) -> str:
    text = re.sub(r"[^a-zA-Z0-9]+", "_", str(value).strip().lower())
    return text.strip("_")


def assert_one_row_per_key(
    df: pd.DataFrame,
    keys: list[str],
    name: str,
) -> None:
    duplicate_mask = df.duplicated(keys, keep=False)
    if duplicate_mask.any():
        examples = df.loc[duplicate_mask, keys].head(10)
        raise ValueError(
            f"{name} contains duplicate join keys {keys}:\n{examples}"
        )


def allclose_or_raise(
    left: pd.Series,
    right: pd.Series,
    name: str,
    atol: float = 1e-8,
) -> float:
    left_values = pd.to_numeric(left, errors="coerce").to_numpy(float)
    right_values = pd.to_numeric(right, errors="coerce").to_numpy(float)

    if not np.allclose(
        left_values,
        right_values,
        atol=atol,
        rtol=0.0,
        equal_nan=True,
    ):
        delta = np.abs(left_values - right_values)
        max_delta = float(np.nanmax(delta))
        raise ValueError(
            f"Validation mismatch for {name}; max delta={max_delta}"
        )

    delta = np.abs(left_values - right_values)
    return float(np.nanmax(delta)) if len(delta) else 0.0


def gate_mask(
    df: pd.DataFrame,
    gate_row: pd.Series,
) -> pd.Series:
    mask = pd.Series(True, index=df.index)

    minimum_rules = [
        ("chosen_lg_score", "score_min"),
        ("chosen_inliers", "inliers_min"),
        ("chosen_matches", "matches_min"),
        ("chosen_min_coverage", "coverage_min"),
        ("lg_score_margin_top1_top2", "margin_min"),
    ]

    for feature_column, threshold_column in minimum_rules:
        threshold = float(gate_row[threshold_column])
        if threshold > 0:
            values = pd.to_numeric(df[feature_column], errors="coerce")
            mask &= values >= threshold

    union_rank_max = float(gate_row["union_rank_max"])
    if union_rank_max > 0:
        union_rank = pd.to_numeric(
            df["chosen_union_rank"],
            errors="coerce",
        )
        mask &= union_rank <= union_rank_max

    return mask.fillna(False)


def satellite_columns(
    satellite_index: pd.DataFrame,
    id_column: str,
    prefix: str,
) -> pd.DataFrame:
    rename = {
        "tile_index": id_column,
        "filename": f"{prefix}_filename",
        "tile_path": f"{prefix}_path",
        "tile_path_relative": f"{prefix}_path_relative",
        "lon_center": f"{prefix}_lon",
        "lat_center": f"{prefix}_lat",
        "utm_x_m": f"{prefix}_utm_x_m",
        "utm_y_m": f"{prefix}_utm_y_m",
        "x_enu_global_m": f"{prefix}_x_enu_global_m",
        "y_enu_global_m": f"{prefix}_y_enu_global_m",
        "x_enu_m": f"{prefix}_x_enu_m",
        "y_enu_m": f"{prefix}_y_enu_m",
        "lon_tl": f"{prefix}_lon_tl",
        "lat_tl": f"{prefix}_lat_tl",
        "lon_br": f"{prefix}_lon_br",
        "lat_br": f"{prefix}_lat_br",
    }

    output = satellite_index[list(rename)].rename(columns=rename)
    assert_one_row_per_key(output, [id_column], prefix)
    return output


def add_policy_gap_column(
    manifest: pd.DataFrame,
    accept_column: str,
) -> str:
    output_column = (
        accept_column.removesuffix("_accept_online")
        .removesuffix("_accept_eval_only")
        + "_distance_since_previous_accept_m"
    )

    manifest[output_column] = np.nan

    ordered = manifest.loc[
        bool_series(manifest[accept_column])
    ].sort_values("sequence_frame_id")

    gaps = ordered["reference_cumulative_distance_m"].diff()
    manifest.loc[ordered.index, output_column] = gaps.to_numpy()

    return output_column


def build_gap_rows(
    manifest: pd.DataFrame,
    accept_column: str,
    total_start_m: float,
    total_end_m: float,
) -> list[dict[str, Any]]:
    accepted = manifest.loc[
        bool_series(manifest[accept_column])
    ].sort_values("reference_cumulative_distance_m")

    if accepted.empty:
        return [{
            "policy": accept_column,
            "gap_type": "whole_trajectory_no_corrections",
            "from_token": None,
            "to_token": None,
            "start_distance_m": total_start_m,
            "end_distance_m": total_end_m,
            "gap_m": total_end_m - total_start_m,
        }]

    rows: list[dict[str, Any]] = []

    first = accepted.iloc[0]
    rows.append({
        "policy": accept_column,
        "gap_type": "start_boundary",
        "from_token": None,
        "to_token": int(first["token"]),
        "start_distance_m": total_start_m,
        "end_distance_m": float(
            first["reference_cumulative_distance_m"]
        ),
        "gap_m": float(
            first["reference_cumulative_distance_m"] - total_start_m
        ),
    })

    accepted_records = accepted.to_dict(orient="records")
    for previous, current in zip(
        accepted_records[:-1],
        accepted_records[1:],
    ):
        start_distance = float(
            previous["reference_cumulative_distance_m"]
        )
        end_distance = float(
            current["reference_cumulative_distance_m"]
        )

        rows.append({
            "policy": accept_column,
            "gap_type": "between_corrections",
            "from_token": int(previous["token"]),
            "to_token": int(current["token"]),
            "start_distance_m": start_distance,
            "end_distance_m": end_distance,
            "gap_m": end_distance - start_distance,
        })

    last = accepted.iloc[-1]
    rows.append({
        "policy": accept_column,
        "gap_type": "end_boundary",
        "from_token": int(last["token"]),
        "to_token": None,
        "start_distance_m": float(
            last["reference_cumulative_distance_m"]
        ),
        "end_distance_m": total_end_m,
        "gap_m": float(
            total_end_m - last["reference_cumulative_distance_m"]
        ),
    })

    return rows

def policy_summary(
    manifest: pd.DataFrame,
    accept_column: str,
    gap_df: pd.DataFrame,
) -> dict[str, Any]:
    accepted_mask = bool_series(manifest[accept_column])
    accepted = manifest.loc[accepted_mask]

    if accept_column == "oracle_accept_eval_only":
        # Oracle replay uses the best candidate available in the
        # processed Top-50 pool, not the LightGlue-selected candidate.
        policy_hit = bool_series(
            manifest["oracle_processed_hit_le_threshold"]
        )
        policy_error = pd.to_numeric(
            manifest["oracle_processed_error_m"],
            errors="coerce",
        )
        policy_dangerous = policy_error > 100.0
        error_source = "oracle_processed_error_m"
    else:
        # Realistic gates use the LightGlue-selected absolute position.
        policy_hit = bool_series(
            manifest["hit_eval_only"]
        )
        policy_error = pd.to_numeric(
            manifest["chosen_error_m_eval_only"],
            errors="coerce",
        )
        policy_dangerous = bool_series(
            manifest["dangerous_false_eval_only"]
        )
        error_source = "chosen_error_m_eval_only"

    true_count = int(
        (accepted_mask & policy_hit).sum()
    )
    false_count = int(accepted_mask.sum()) - true_count
    dangerous_count = int(
        (accepted_mask & policy_dangerous).sum()
    )

    accepted_errors = policy_error.loc[accepted_mask]

    # Correction-spacing statistics for this policy.
    policy_gaps = gap_df.loc[
        gap_df["policy"] == accept_column
    ]

    between = policy_gaps.loc[
        policy_gaps["gap_type"] == "between_corrections",
        "gap_m",
    ].dropna()

    all_coverage = policy_gaps["gap_m"].dropna()

    return {
        "accept_column": accept_column,
        "accepted": int(accepted_mask.sum()),
        "true_accepts_eval_only": true_count,
        "false_accepts_eval_only": false_count,
        "dangerous_false_accepts_gt100m_eval_only":
            dangerous_count,
        "precision_eval_only": (
            true_count / int(accepted_mask.sum())
            if int(accepted_mask.sum()) > 0
            else None
        ),
        "accepted_error_source_eval_only": error_source,
        "median_accepted_error_m_eval_only": (
            float(accepted_errors.median())
            if len(accepted_errors)
            else None
        ),
        "inter_correction_gap_median": (
            float(between.median())
            if len(between)
            else None
        ),
        "inter_correction_gap_p95": (
            float(between.quantile(0.95))
            if len(between)
            else None
        ),
        "inter_correction_gap_max": (
            float(between.max())
            if len(between)
            else None
        ),
        "coverage_gap_including_boundaries_max_m": (
            float(all_coverage.max())
            if len(all_coverage)
            else None
        ),
        "first_accepted_frame": (
            int(accepted["sequence_frame_id"].min())
            if len(accepted)
            else None
        ),
        "last_accepted_frame": (
            int(accepted["sequence_frame_id"].max())
            if len(accepted)
            else None
        ),
    }

def plot_correction_distribution(
    manifest: pd.DataFrame,
    policy_columns: list[str],
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(13, 4.8))

    for row_index, policy in enumerate(policy_columns):
        accepted = manifest.loc[
            bool_series(manifest[policy])
        ].sort_values("reference_cumulative_distance_m")

        ax.scatter(
            accepted["reference_cumulative_distance_m"],
            np.full(len(accepted), row_index),
            s=34,
            label=policy,
        )

    ax.set_yticks(range(len(policy_columns)))
    ax.set_yticklabels(policy_columns)
    ax.set_xlabel("Reference cumulative trajectory distance [m] — evaluation only")
    ax.set_ylabel("Correction policy")
    ax.set_title("S6B.0 accepted correction distribution along traj01")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_balanced_map(
    trajectory: pd.DataFrame,
    manifest: pd.DataFrame,
    output_path: Path,
) -> None:
    balanced = manifest.loc[
        bool_series(manifest["balanced_accept_online"])
    ]

    balanced_true = balanced.loc[
        bool_series(balanced["hit_eval_only"])
    ]
    balanced_false = balanced.loc[
        ~bool_series(balanced["hit_eval_only"])
    ]

    fig, ax = plt.subplots(figsize=(9, 8))

    ax.plot(
        trajectory["reference_x_m"],
        trajectory["reference_y_m"],
        linewidth=2.0,
        label="Reference trajectory — evaluation only",
    )
    ax.plot(
        trajectory["prefix_aligned_x_m"],
        trajectory["prefix_aligned_y_m"],
        linewidth=1.3,
        label="ORB prefix-aligned trajectory — evaluation replay only",
    )

    ax.scatter(
        balanced_true["chosen_abs_x_enu_m"],
        balanced_true["chosen_abs_y_enu_m"],
        marker="o",
        s=50,
        label="Balanced accepted: true <=40 m — evaluation label",
    )
    ax.scatter(
        balanced_false["chosen_abs_x_enu_m"],
        balanced_false["chosen_abs_y_enu_m"],
        marker="x",
        s=70,
        label="Balanced accepted: false >40 m — evaluation label",
    )

    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("Local ENU X [m]")
    ax.set_ylabel("Local ENU Y [m]")
    ax.set_title("S6B.0 balanced-gate absolute correction locations")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_balanced_gaps(
    gap_df: pd.DataFrame,
    output_path: Path,
) -> None:
    balanced_gaps = gap_df.loc[
        gap_df["policy"] == "balanced_accept_online",
        "gap_m",
    ].dropna()

    fig, ax = plt.subplots(figsize=(9, 5))

    bins = min(15, max(5, int(math.sqrt(max(len(balanced_gaps), 1)))))
    ax.hist(balanced_gaps, bins=bins)

    ax.set_xlabel("Distance gap [m]")
    ax.set_ylabel("Count")
    ax.set_title(
        "S6B.0 balanced-gate correction gaps\n"
        "including start and end trajectory boundaries"
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> int:
    args = parse_args()

    for directory in (
        args.out_dir,
        args.report_dir,
        args.figure_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    input_paths = {
        "features": args.features,
        "gates": args.gates,
        "query_summary": args.query_summary,
        "candidate_scores": args.candidate_scores,
        "sequence_manifest": args.sequence_manifest,
        "relative_trajectory": args.relative_trajectory,
        "satellite_index": args.satellite_index,
    }

    missing = [
        str(path)
        for path in input_paths.values()
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing required inputs:\n" + "\n".join(missing)
        )

    features = pd.read_csv(args.features, low_memory=False)
    gates = pd.read_csv(args.gates, low_memory=False)
    query_summary = pd.read_csv(
        args.query_summary,
        low_memory=False,
    )
    candidate_scores = pd.read_csv(
        args.candidate_scores,
        low_memory=False,
    )
    sequence = pd.read_csv(
        args.sequence_manifest,
        low_memory=False,
    )
    trajectory_all = pd.read_csv(
        args.relative_trajectory,
        low_memory=False,
    )
    satellite_index = pd.read_csv(
        args.satellite_index,
        low_memory=False,
    )

    require_columns(
        features,
        [
            "policy",
            "token",
            "chosen_tile_id",
            "oracle_tile_id",
            "chosen_lg_score",
            "chosen_inliers",
            "chosen_matches",
            "chosen_min_coverage",
            "chosen_union_rank",
            "lg_score_margin_top1_top2",
            "hit_eval_only",
            "chosen_error_m_eval_only",
            "dangerous_false_eval_only",
            "oracle_processed_hit_le_threshold",
        ],
        "S5C.3 features",
    )
    
    require_columns(
        gates,
        [
            "gate",
            "score_min",
            "inliers_min",
            "matches_min",
            "coverage_min",
            "margin_min",
            "union_rank_max",
            "recommended_profile",
        ],
        "S5C.3 gates",
    )

    require_columns(
        sequence,
        [
            "sequence_frame_id",
            "token0_id",
            "frame_index_in_sequence",
            "image_path",
            "filename",
            "lon",
            "lat",
            "x_enu_m",
            "y_enu_m",
        ],
        "S6A sequence manifest",
    )

    require_columns(
        trajectory_all,
        [
            "sequence_frame_id",
            "token0_id",
            "variant",
            "visual_x_px",
            "visual_y_px",
            "visual_yaw_rad",
            "reference_x_m",
            "reference_y_m",
            "reference_cumulative_distance_m",
            "prefix_aligned_x_m",
            "prefix_aligned_y_m",
            "prefix_locked_error_m",
        ],
        "S6A relative trajectory",
    )
    require_columns(
        satellite_index,
        [
            "tile_index",
            "filename",
            "lon_center",
            "lat_center",
            "x_enu_m",
            "y_enu_m",
        ],
        "satellite index",
    )

    if len(features) != 263:
        raise ValueError(
            f"Expected 263 S5C.3 feature rows, found {len(features)}"
        )
    assert_one_row_per_key(features, ["token"], "S5C.3 features")

    feature_policies = set(features["policy"].astype(str))
    if feature_policies != {"lightglue_only"}:
        raise ValueError(
            "S5C.3 features must contain only lightglue_only; "
            f"found {sorted(feature_policies)}"
        )

    # Validate the 789-row S5C.2 summary without directly joining
    # all three policies.
    lightglue_summary = query_summary.loc[
        query_summary["policy"].astype(str) == "lightglue_only"
    ].copy()

    if len(lightglue_summary) != 263:
        raise ValueError(
            "Expected 263 lightglue_only S5C.2 summary rows, "
            f"found {len(lightglue_summary)}"
        )
    assert_one_row_per_key(
        lightglue_summary,
        ["token"],
        "S5C.2 lightglue_only summary",
    )

    feature_check = features[
        ["token", "chosen_tile_id", "chosen_lg_score"]
    ].merge(
        lightglue_summary[
            ["token", "chosen_tile_id", "chosen_lg_score"]
        ],
        on="token",
        how="inner",
        validate="one_to_one",
        suffixes=("_s5c3", "_s5c2"),
    )

    if len(feature_check) != 263:
        raise ValueError("S5C.2/S5C.3 token validation join failed")

    if not (
        feature_check["chosen_tile_id_s5c3"]
        == feature_check["chosen_tile_id_s5c2"]
    ).all():
        raise ValueError(
            "S5C.2 and S5C.3 chosen_tile_id values disagree"
        )

    s5c2_s5c3_score_max_delta = allclose_or_raise(
        feature_check["chosen_lg_score_s5c3"],
        feature_check["chosen_lg_score_s5c2"],
        "S5C.2 vs S5C.3 chosen_lg_score",
    )

    # Sequence manifest fields are renamed so reference/evaluation
    # data cannot be mistaken for online correction inputs.
    sequence_selected = sequence[
        [
            "sequence_frame_id",
            "sequence",
            "image_path",
            "image_path_relative",
            "filename",
            "token0_id",
            "token1_order",
            "frame_index_in_sequence",
            "global_frame_index",
            "lon",
            "lat",
            "utm_x_m",
            "utm_y_m",
            "x_enu_global_m",
            "y_enu_global_m",
            "x_enu_m",
            "y_enu_m",
        ]
    ].rename(columns={
        "filename": "uav_filename",
        "lon": "reference_lon_eval_only",
        "lat": "reference_lat_eval_only",
        "utm_x_m": "reference_utm_x_m_eval_only",
        "utm_y_m": "reference_utm_y_m_eval_only",
        "x_enu_global_m": "reference_x_enu_global_m_eval_only",
        "y_enu_global_m": "reference_y_enu_global_m_eval_only",
        "x_enu_m": "reference_x_enu_m_eval_only",
        "y_enu_m": "reference_y_enu_m_eval_only",
    })

    assert_one_row_per_key(
        sequence_selected,
        ["token0_id"],
        "S6A sequence manifest token0_id",
    )
    assert_one_row_per_key(
        sequence_selected,
        ["sequence_frame_id"],
        "S6A sequence manifest sequence_frame_id",
    )

    manifest = features.merge(
        sequence_selected,
        left_on="token",
        right_on="token0_id",
        how="left",
        validate="one_to_one",
        indicator="_sequence_join",
    )

    if not (manifest["_sequence_join"] == "both").all():
        failed = manifest.loc[
            manifest["_sequence_join"] != "both",
            ["token", "_sequence_join"],
        ]
        raise ValueError(
            f"Unmatched S5C token to S6A token0_id:\n{failed}"
        )
    manifest = manifest.drop(columns="_sequence_join")

    available_variants = sorted(
        trajectory_all["variant"].astype(str).unique()
    )
    if args.relative_variant not in available_variants:
        raise ValueError(
            f"Relative variant {args.relative_variant!r} not found. "
            f"Available variants: {available_variants}"
        )

    trajectory = trajectory_all.loc[
        trajectory_all["variant"].astype(str)
        == args.relative_variant
    ].copy()

    if len(trajectory) != 1034:
        raise ValueError(
            f"Expected 1034 official trajectory rows, found {len(trajectory)}"
        )

    assert_one_row_per_key(
        trajectory,
        ["sequence_frame_id", "token0_id"],
        "filtered S6A trajectory",
    )

    trajectory_selected = trajectory[
        [
            "sequence_frame_id",
            "token0_id",
            "variant",
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
            "reference_x_m",
            "reference_y_m",
            "reference_cumulative_distance_m",
            "global_aligned_x_m",
            "global_aligned_y_m",
            "global_alignment_error_m",
            "prefix_aligned_x_m",
            "prefix_aligned_y_m",
            "prefix_locked_error_m",
            "distance_after_alignment_prefix_m",
            "prefix_locked_drift_per_100m",
        ]
    ].rename(columns={
        "variant": "relative_variant",
        "visual_x_px": "relative_visual_x_px",
        "visual_y_px": "relative_visual_y_px",
        "visual_yaw_rad": "relative_visual_yaw_rad",
        "visual_yaw_deg_unwrapped":
            "relative_visual_yaw_deg_unwrapped",
        "step_x_local_px": "relative_step_x_local_px",
        "step_y_local_px": "relative_step_y_local_px",
        "step_x_global_px": "relative_step_x_global_px",
        "step_y_global_px": "relative_step_y_global_px",
        "step_motion_px": "relative_step_motion_px",
        "pair_safe_image_only": "relative_pair_safe_image_only",
        "global_alignment_error_m":
            "global_alignment_error_m_eval_only",
        "prefix_locked_error_m":
            "prefix_locked_error_m_eval_only",
        "prefix_locked_drift_per_100m":
            "prefix_locked_drift_per_100m_eval_only",
    })

    manifest = manifest.merge(
        trajectory_selected,
        on=["sequence_frame_id", "token0_id"],
        how="left",
        validate="one_to_one",
        indicator="_trajectory_join",
    )

    if not (manifest["_trajectory_join"] == "both").all():
        failed = manifest.loc[
            manifest["_trajectory_join"] != "both",
            ["token", "sequence_frame_id", "_trajectory_join"],
        ]
        raise ValueError(
            f"Unmatched S6A trajectory rows:\n{failed}"
        )
    manifest = manifest.drop(columns="_trajectory_join")

    # Join chosen and oracle tile coordinates separately.
    chosen_tiles = satellite_columns(
        satellite_index,
        id_column="chosen_tile_id",
        prefix="chosen_abs",
    )
    oracle_tiles = satellite_columns(
        satellite_index,
        id_column="oracle_tile_id",
        prefix="oracle_abs_eval_only",
    )

    manifest = manifest.merge(
        chosen_tiles,
        on="chosen_tile_id",
        how="left",
        validate="many_to_one",
        indicator="_chosen_tile_join",
    )
    if not (manifest["_chosen_tile_join"] == "both").all():
        failed = manifest.loc[
            manifest["_chosen_tile_join"] != "both",
            ["token", "chosen_tile_id"],
        ]
        raise ValueError(f"Chosen tile join failed:\n{failed}")
    manifest = manifest.drop(columns="_chosen_tile_join")

    manifest = manifest.merge(
        oracle_tiles,
        on="oracle_tile_id",
        how="left",
        validate="many_to_one",
        indicator="_oracle_tile_join",
    )
    if not (manifest["_oracle_tile_join"] == "both").all():
        failed = manifest.loc[
            manifest["_oracle_tile_join"] != "both",
            ["token", "oracle_tile_id"],
        ]
        raise ValueError(f"Oracle tile join failed:\n{failed}")
    manifest = manifest.drop(columns="_oracle_tile_join")

    # Validate that each chosen LightGlue result corresponds to one
    # candidate-score row.

    # Convert satellite tile UTM coordinates into the same traj01
    # start-local frame used by the S6A reference and prefix-aligned
    # relative trajectory.
    #
    # Important:
    # UAV and satellite x_enu_global_m columns do NOT share an origin.
    # They were generated independently from the first row of each
    # dataframe. UTM coordinates are the common projected frame.

    trajectory_origin_utm_x = (
        pd.to_numeric(
            manifest["reference_utm_x_m_eval_only"],
            errors="raise",
        )
        - pd.to_numeric(
            manifest["reference_x_enu_m_eval_only"],
            errors="raise",
        )
    )

    trajectory_origin_utm_y = (
        pd.to_numeric(
            manifest["reference_utm_y_m_eval_only"],
            errors="raise",
        )
        - pd.to_numeric(
            manifest["reference_y_enu_m_eval_only"],
            errors="raise",
        )
    )

    origin_x_spread = float(
        trajectory_origin_utm_x.max()
        - trajectory_origin_utm_x.min()
    )
    origin_y_spread = float(
        trajectory_origin_utm_y.max()
        - trajectory_origin_utm_y.min()
    )

    if origin_x_spread > 1e-3 or origin_y_spread > 1e-3:
        raise ValueError(
            "The traj01 UTM-to-local origin is not constant: "
            f"x spread={origin_x_spread:.9f} m, "
            f"y spread={origin_y_spread:.9f} m"
        )

    traj01_origin_utm_x_m = float(
        trajectory_origin_utm_x.median()
    )
    traj01_origin_utm_y_m = float(
        trajectory_origin_utm_y.median()
    )

    manifest[
        "traj01_origin_utm_x_m_eval_only"
    ] = traj01_origin_utm_x_m

    manifest[
        "traj01_origin_utm_y_m_eval_only"
    ] = traj01_origin_utm_y_m

    # Selected LightGlue tile in traj01 start-local coordinates.
    manifest["chosen_abs_x_traj01_m"] = (
        pd.to_numeric(
            manifest["chosen_abs_utm_x_m"],
            errors="raise",
        )
        - traj01_origin_utm_x_m
    )

    manifest["chosen_abs_y_traj01_m"] = (
        pd.to_numeric(
            manifest["chosen_abs_utm_y_m"],
            errors="raise",
        )
        - traj01_origin_utm_y_m
    )

    # Evaluation-only oracle tile in traj01 start-local coordinates.
    manifest["oracle_abs_eval_only_x_traj01_m"] = (
        pd.to_numeric(
            manifest[
                "oracle_abs_eval_only_utm_x_m"
            ],
            errors="raise",
        )
        - traj01_origin_utm_x_m
    )

    manifest["oracle_abs_eval_only_y_traj01_m"] = (
        pd.to_numeric(
            manifest[
                "oracle_abs_eval_only_utm_y_m"
            ],
            errors="raise",
        )
        - traj01_origin_utm_y_m
    )

    # Verify that S6A reference coordinates and the sequence-manifest
    # local ENU coordinates represent the same traj01 frame.
    reference_frame_delta = np.hypot(
        pd.to_numeric(
            manifest["reference_x_m"],
            errors="raise",
        )
        - pd.to_numeric(
            manifest["reference_x_enu_m_eval_only"],
            errors="raise",
        ),
        pd.to_numeric(
            manifest["reference_y_m"],
            errors="raise",
        )
        - pd.to_numeric(
            manifest["reference_y_enu_m_eval_only"],
            errors="raise",
        ),
    )

    reference_frame_max_delta = float(
        reference_frame_delta.max()
    )

    if reference_frame_max_delta > 1e-3:
        raise ValueError(
            "S6A reference coordinates do not match the traj01 "
            "sequence-local ENU coordinates. "
            f"Maximum delta={reference_frame_max_delta:.6f} m"
        )

    # Evaluation-only consistency diagnostics.
    manifest[
        "chosen_abs_error_from_traj01_xy_m_eval_only"
    ] = np.hypot(
        manifest["chosen_abs_x_traj01_m"]
        - manifest["reference_x_m"],
        manifest["chosen_abs_y_traj01_m"]
        - manifest["reference_y_m"],
    )

    manifest[
        "oracle_abs_error_from_traj01_xy_m_eval_only"
    ] = np.hypot(
        manifest["oracle_abs_eval_only_x_traj01_m"]
        - manifest["reference_x_m"],
        manifest["oracle_abs_eval_only_y_traj01_m"]
        - manifest["reference_y_m"],
    )

    chosen_error_delta = np.abs(
        manifest[
            "chosen_abs_error_from_traj01_xy_m_eval_only"
        ]
        - pd.to_numeric(
            manifest["chosen_error_m_eval_only"],
            errors="raise",
        )
    )

    oracle_error_delta = np.abs(
        manifest[
            "oracle_abs_error_from_traj01_xy_m_eval_only"
        ]
        - pd.to_numeric(
            manifest["oracle_processed_error_m"],
            errors="raise",
        )
    )

    chosen_error_delta_max = float(
        chosen_error_delta.max()
    )
    oracle_error_delta_max = float(
        oracle_error_delta.max()
    )

    # Stored errors use geodesic lon/lat distance while the new
    # coordinates use projected UTM distance. Small differences are
    # expected, but kilometre-scale differences are not.
    
    if (
        chosen_error_delta_max
        > ABSOLUTE_COORDINATE_ERROR_TOLERANCE_M
        or oracle_error_delta_max
        > ABSOLUTE_COORDINATE_ERROR_TOLERANCE_M
    ):
        
        raise ValueError(
            "UTM-derived traj01 absolute coordinates do not agree "
            "with stored geodesic evaluation errors. "
            f"Tolerance="
            f"{ABSOLUTE_COORDINATE_ERROR_TOLERANCE_M:.3f} m; "
            f"chosen max delta={chosen_error_delta_max:.3f} m; "
            f"oracle max delta={oracle_error_delta_max:.3f} m"
        )    

    selected_candidates = candidate_scores[
        [
            "token",
            "tile_id",
            "uav_image_path",
            "sat_image_path",
            "lightglue_status",
            "lightglue_matches",
            "lightglue_ransac_inliers",
            "lightglue_inlier_ratio",
            "lightglue_uav_coverage",
            "lightglue_sat_coverage",
            "lightglue_score",
            "lg_score_num",
            "lg_inliers_num",
            "lg_matches_num",
            "min_cov_num",
            "union_rank_num",
            "lightglue_rank",
            "s5c2_chunk",
        ]
    ].rename(columns={
        "tile_id": "chosen_tile_id",
        "uav_image_path": "selected_uav_image_path",
        "sat_image_path": "selected_sat_image_path",
        "lightglue_status": "selected_lightglue_status",
        "lightglue_matches": "candidate_lightglue_matches",
        "lightglue_ransac_inliers":
            "candidate_lightglue_ransac_inliers",
        "lightglue_inlier_ratio":
            "candidate_lightglue_inlier_ratio",
        "lightglue_uav_coverage":
            "candidate_lightglue_uav_coverage",
        "lightglue_sat_coverage":
            "candidate_lightglue_sat_coverage",
        "lightglue_score": "candidate_lightglue_score",
        "lg_score_num": "candidate_lg_score_num",
        "lg_inliers_num": "candidate_lg_inliers_num",
        "lg_matches_num": "candidate_lg_matches_num",
        "min_cov_num": "candidate_min_cov_num",
        "union_rank_num": "candidate_union_rank_num",
        "lightglue_rank": "candidate_lightglue_rank",
        "s5c2_chunk": "selected_candidate_s5c2_chunk",
    })

    assert_one_row_per_key(
        selected_candidates,
        ["token", "chosen_tile_id"],
        "S5C.2 candidate score token/tile",
    )

    manifest = manifest.merge(
        selected_candidates,
        on=["token", "chosen_tile_id"],
        how="left",
        validate="one_to_one",
        indicator="_candidate_join",
    )

    if not (manifest["_candidate_join"] == "both").all():
        failed = manifest.loc[
            manifest["_candidate_join"] != "both",
            ["token", "chosen_tile_id"],
        ]
        raise ValueError(
            f"Selected candidate-score join failed:\n{failed}"
        )
    manifest = manifest.drop(columns="_candidate_join")

    candidate_validation = {
        "lg_score_max_delta": allclose_or_raise(
            manifest["chosen_lg_score"],
            manifest["candidate_lg_score_num"],
            "chosen LightGlue score",
        ),
        "inliers_max_delta": allclose_or_raise(
            manifest["chosen_inliers"],
            manifest["candidate_lg_inliers_num"],
            "chosen LightGlue inliers",
        ),
        "matches_max_delta": allclose_or_raise(
            manifest["chosen_matches"],
            manifest["candidate_lg_matches_num"],
            "chosen LightGlue matches",
        ),
        "coverage_max_delta": allclose_or_raise(
            manifest["chosen_min_coverage"],
            manifest["candidate_min_cov_num"],
            "chosen minimum coverage",
        ),
        "union_rank_max_delta": allclose_or_raise(
            manifest["chosen_union_rank"],
            manifest["candidate_union_rank_num"],
            "chosen union rank",
        ),
    }

    # Explicit evaluation-only aliases.
    manifest["hit_eval_only"] = bool_series(
        manifest["hit_eval_only"]
    )
    manifest["dangerous_false_eval_only"] = bool_series(
        manifest["dangerous_false_eval_only"]
    )
    manifest["oracle_accept_eval_only"] = bool_series(
        manifest["oracle_processed_hit_le_threshold"]
    )

    gate_columns: list[str] = []
    gate_definitions: dict[str, Any] = {}

    for _, gate_row in gates.iterrows():
        # `gate` contains an encoded threshold signature such as
        # s0_i48_m0_c0_5_gap0_ur999.
        # `recommended_profile` contains the semantic profile name.
        profile_value = gate_row.get("recommended_profile")

        if pd.isna(profile_value) or not str(profile_value).strip():
            profile_value = gate_row["gate"]

        gate_name = safe_name(profile_value)
        column = f"{gate_name}_accept_online"

        if column in gate_columns:
            raise ValueError(
                f"Duplicate confidence-gate profile column: {column}"
            )

        manifest[column] = gate_mask(manifest, gate_row)
        gate_columns.append(column)

        gate_definitions[gate_name] = {
            "source_gate": str(gate_row["gate"]),
            "recommended_profile": str(profile_value),
            **{
                key: json_safe(gate_row[key])
                for key in [
                    "score_min",
                    "inliers_min",
                    "matches_min",
                    "coverage_min",
                    "margin_min",
                    "union_rank_max",
                ]
            },
        }

    balanced_candidates = [
        column for column in gate_columns
        if "balanced" in column
    ]
    permissive_candidates = [
        column for column in gate_columns
        if "permissive" in column
    ]
    exploratory_candidates = [
        column for column in gate_columns
        if "exploratory" in column
    ]

    if len(balanced_candidates) != 1:
        raise ValueError(
            f"Could not uniquely resolve balanced gate: {gate_columns}"
        )
    if len(permissive_candidates) != 1:
        raise ValueError(
            f"Could not uniquely resolve permissive gate: {gate_columns}"
        )

    manifest["balanced_accept_online"] = bool_series(
        manifest[balanced_candidates[0]]
    )
    manifest["permissive_accept_online"] = bool_series(
        manifest[permissive_candidates[0]]
    )

    if len(exploratory_candidates) == 1:
        manifest["exploratory_accept_online"] = bool_series(
            manifest[exploratory_candidates[0]]
        )

    manifest["balanced_true_accept_eval_only"] = (
        manifest["balanced_accept_online"]
        & manifest["hit_eval_only"]
    )
    manifest["balanced_false_accept_eval_only"] = (
        manifest["balanced_accept_online"]
        & ~manifest["hit_eval_only"]
    )
    manifest["balanced_dangerous_false_eval_only"] = (
        manifest["balanced_accept_online"]
        & manifest["dangerous_false_eval_only"]
    )

    manifest["permissive_true_accept_eval_only"] = (
        manifest["permissive_accept_online"]
        & manifest["hit_eval_only"]
    )
    manifest["permissive_false_accept_eval_only"] = (
        manifest["permissive_accept_online"]
        & ~manifest["hit_eval_only"]
    )

    policy_columns = [
        "oracle_accept_eval_only",
        "balanced_accept_online",
        "permissive_accept_online",
    ]

    if "exploratory_accept_online" in manifest:
        policy_columns.append("exploratory_accept_online")

    for policy in policy_columns:
        add_policy_gap_column(manifest, policy)

    manifest = manifest.sort_values(
        ["sequence_frame_id", "token"]
    ).reset_index(drop=True)

    # Frozen-result assertions protect the maintained chain.
    baseline_hits = int(manifest["hit_eval_only"].sum())
    oracle_accepts = int(manifest["oracle_accept_eval_only"].sum())
    balanced_accepts = int(manifest["balanced_accept_online"].sum())
    balanced_true = int(
        manifest["balanced_true_accept_eval_only"].sum()
    )
    balanced_false = int(
        manifest["balanced_false_accept_eval_only"].sum()
    )
    balanced_dangerous = int(
        manifest["balanced_dangerous_false_eval_only"].sum()
    )
    permissive_accepts = int(
        manifest["permissive_accept_online"].sum()
    )

    expected = {
        "lightglue_hits": (baseline_hits, 68),
        "oracle_accepts": (oracle_accepts, 102),
        "balanced_accepts": (balanced_accepts, 33),
        "balanced_true": (balanced_true, 25),
        "balanced_false": (balanced_false, 8),
        "balanced_dangerous": (balanced_dangerous, 0),
        "permissive_accepts": (permissive_accepts, 80),
    }

    for name, (actual, expected_value) in expected.items():
        if actual != expected_value:
            raise ValueError(
                f"Frozen S5C check failed for {name}: "
                f"expected {expected_value}, found {actual}"
            )

    trajectory = trajectory.sort_values("sequence_frame_id")
    total_start_m = float(
        trajectory["reference_cumulative_distance_m"].iloc[0]
    )
    total_end_m = float(
        trajectory["reference_cumulative_distance_m"].iloc[-1]
    )

    all_gap_rows: list[dict[str, Any]] = []
    for policy in policy_columns:
        all_gap_rows.extend(
            build_gap_rows(
                manifest,
                policy,
                total_start_m,
                total_end_m,
            )
        )

    gap_df = pd.DataFrame(all_gap_rows)

    manifest_path = (
        args.out_dir / "s6b0_absolute_correction_manifest.csv"
    )
    balanced_path = (
        args.out_dir / "s6b0_balanced_accepted_corrections.csv"
    )
    permissive_path = (
        args.out_dir / "s6b0_permissive_accepted_corrections.csv"
    )
    gap_path = (
        args.out_dir / "s6b0_correction_spacing_by_policy.csv"
    )
    summary_path = (
        args.report_dir
        / "s6b0_absolute_correction_manifest_summary.json"
    )

    manifest.to_csv(manifest_path, index=False)
    manifest.loc[
        manifest["balanced_accept_online"]
    ].to_csv(balanced_path, index=False)
    manifest.loc[
        manifest["permissive_accept_online"]
    ].to_csv(permissive_path, index=False)
    gap_df.to_csv(gap_path, index=False)

    policy_summaries = {
        policy: policy_summary(manifest, policy, gap_df)
        for policy in policy_columns
    }

    summary = {
        "stage": "S6B.0",
        "relative_variant": args.relative_variant,
        "available_relative_variants": available_variants,
        "important_warning": (
            "prefix_aligned_x_m and prefix_aligned_y_m are "
            "evaluation-aligned trajectory fields. They may be used "
            "for controlled replay evaluation but are not raw online "
            "localization outputs."
        ),
        "inputs": {
            key: str(value)
            for key, value in input_paths.items()
        },
        "rows": {
            "correction_manifest": int(len(manifest)),
            "balanced_accepted": balanced_accepts,
            "permissive_accepted": permissive_accepts,
            "oracle_processed_available": oracle_accepts,
        },
        "frozen_s5c_checks": {
            name: {
                "actual": actual,
                "expected": expected_value,
                "passed": actual == expected_value,
            }
            for name, (actual, expected_value) in expected.items()
        },
        "join_rules": {
            "s5c_to_sequence": "token == token0_id",
            "sequence_to_trajectory": (
                "sequence_frame_id + token0_id after filtering "
                f"variant == {args.relative_variant}"
            ),
            "chosen_tile": "chosen_tile_id == tile_index",
            "oracle_tile_eval_only": "oracle_tile_id == tile_index",
            "selected_candidate": (
                "token + chosen_tile_id == token + tile_id"
            ),
        },
        "candidate_validation": candidate_validation,
        "s5c2_s5c3_score_max_delta":
            s5c2_s5c3_score_max_delta,
        "gate_definitions": gate_definitions,
        "policy_spacing_and_accuracy": policy_summaries,
        "trajectory": {
            "frames": int(len(trajectory)),
            "reference_total_distance_m_eval_only":
                total_end_m - total_start_m,
        },
        "outputs": {
            "manifest": str(manifest_path),
            "balanced_corrections": str(balanced_path),
            "permissive_corrections": str(permissive_path),
            "spacing": str(gap_path),
        },
    }

    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    plot_correction_distribution(
        manifest,
        policy_columns,
        args.figure_dir
        / "s6b0_correction_distribution_by_distance.png",
    )
    plot_balanced_map(
        trajectory,
        manifest,
        args.figure_dir
        / "s6b0_balanced_correction_map.png",
    )
    plot_balanced_gaps(
        gap_df,
        args.figure_dir
        / "s6b0_balanced_correction_gap_histogram.png",
    )

    print("S6B.0 Absolute Correction Manifest")
    print("---------------------------------")
    print(f"Relative variant:          {args.relative_variant}")
    print(f"Manifest rows:             {len(manifest)}")
    print(f"LightGlue hits eval-only:  {baseline_hits}/263")
    print(f"Oracle available:          {oracle_accepts}/263")
    print(
        f"Balanced accepted:         {balanced_accepts} "
        f"({balanced_true} true, {balanced_false} false)"
    )
    print(
        f"Balanced dangerous >100m: {balanced_dangerous}"
    )
    print(f"Permissive accepted:       {permissive_accepts}")
    print()
    print(f"Manifest: {manifest_path}")
    print(f"Summary:  {summary_path}")
    print(f"Spacing:  {gap_path}")
    print(f"Figures:  {args.figure_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

