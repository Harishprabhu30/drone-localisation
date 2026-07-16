#!/usr/bin/env python3
"""S6B.1A — Controlled position-only absolute-correction replay.

Policies:
    1. relative_only
    2. oracle_position_reset_eval_only
    3. balanced_position_reset_online
    4. permissive_position_reset_online

The replay preserves all S6A ORB relative increments and heading.
At an accepted absolute fix, only a persistent X/Y offset is replaced.

Important:
    prefix_aligned_x_m / prefix_aligned_y_m are evaluation-aligned.
    Therefore, this is a controlled evaluation replay, not yet a fully
    online/deployable estimator.

Command Used:

export PYTHONPATH=$PWD/src

python scripts/satloc/s6b/s6b_1a_position_only_correction_replay.py \
  2>&1 | tee \
  outputs/satloc/reports/s6b_relative_absolute/s6b1a_position_only_replay.log

"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FIX_ERROR_CONSISTENCY_TOLERANCE_M = 10.0

DEFAULT_CORRECTION_MANIFEST = Path(
    "outputs/satloc/metadata/s6b_relative_absolute/"
    "s6b0_absolute_correction_manifest.csv"
)

DEFAULT_RELATIVE_TRAJECTORY = Path(
    "outputs/satloc/metadata/s6a_relative_motion/"
    "s6a2_orb_relative_trajectory_aligned_eval_only.csv"
)

DEFAULT_METADATA_DIR = Path(
    "outputs/satloc/metadata/s6b_relative_absolute"
)

DEFAULT_REPORT_DIR = Path(
    "outputs/satloc/reports/s6b_relative_absolute"
)

DEFAULT_FIGURE_DIR = Path(
    "outputs/satloc/figures/s6b_relative_absolute"
)

DEFAULT_VARIANT = "se2_scale_normalized"
DEFAULT_BASELINE_X = "prefix_aligned_x_m"
DEFAULT_BASELINE_Y = "prefix_aligned_y_m"


POLICIES = [
    {
        "policy": "relative_only",
        "policy_type": "controlled_relative_baseline_eval_replay",
        "accept_column": None,
        "fix_x_column": None,
        "fix_y_column": None,
        "tile_column": None,
        "manifest_error_column": None,
        "hit_column": None,
        "dangerous_column": None,
    },
    {
        "policy": "oracle_position_reset_eval_only",
        "policy_type": "candidate_pool_oracle_eval_only",
        "accept_column": "oracle_accept_eval_only",
        "fix_x_column": "oracle_abs_eval_only_x_traj01_m",
        "fix_y_column": "oracle_abs_eval_only_y_traj01_m",
        "tile_column": "oracle_tile_id",
        "manifest_error_column": "oracle_processed_error_m",
        "hit_column": "oracle_processed_hit_le_threshold",
        "dangerous_column": None,
    },
    {
        "policy": "balanced_position_reset_online",
        "policy_type": "online_gate_controlled_eval_replay",
        "accept_column": "balanced_accept_online",
        "fix_x_column": "chosen_abs_x_traj01_m",
        "fix_y_column": "chosen_abs_y_traj01_m",
        "tile_column": "chosen_tile_id",
        "manifest_error_column": "chosen_error_m_eval_only",
        "hit_column": "hit_eval_only",
        "dangerous_column": "dangerous_false_eval_only",
    },
    {
        "policy": "permissive_position_reset_online",
        "policy_type": "online_gate_ablation_controlled_eval_replay",
        "accept_column": "permissive_accept_online",
        "fix_x_column": "chosen_abs_x_traj01_m",
        "fix_y_column": "chosen_abs_y_traj01_m",
        "tile_column": "chosen_tile_id",
        "manifest_error_column": "chosen_error_m_eval_only",
        "hit_column": "hit_eval_only",
        "dangerous_column": "dangerous_false_eval_only",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--correction-manifest",
        type=Path,
        default=DEFAULT_CORRECTION_MANIFEST,
    )
    parser.add_argument(
        "--relative-trajectory",
        type=Path,
        default=DEFAULT_RELATIVE_TRAJECTORY,
    )
    parser.add_argument(
        "--relative-variant",
        default=DEFAULT_VARIANT,
    )
    parser.add_argument(
        "--baseline-x-column",
        default=DEFAULT_BASELINE_X,
    )
    parser.add_argument(
        "--baseline-y-column",
        default=DEFAULT_BASELINE_Y,
    )
    parser.add_argument(
        "--metadata-dir",
        type=Path,
        default=DEFAULT_METADATA_DIR,
    )
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
        raise ValueError(
            f"{name} is missing required columns: {missing}"
        )


def bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)

    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes", "y"})
    )


def bool_value(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)

    return str(value).strip().lower() in {
        "true",
        "1",
        "yes",
        "y",
    }


def clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): clean_json(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [clean_json(item) for item in value]

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, np.generic):
        return clean_json(value.item())

    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return value

    if pd.isna(value):
        return None

    return value


def assert_unique(
    df: pd.DataFrame,
    columns: list[str],
    name: str,
) -> None:
    duplicate_mask = df.duplicated(columns, keep=False)

    if duplicate_mask.any():
        examples = df.loc[
            duplicate_mask,
            columns,
        ].head(10)

        raise ValueError(
            f"{name} contains duplicate keys {columns}:\n"
            f"{examples}"
        )


def prepare_trajectory(
    trajectory_all: pd.DataFrame,
    variant: str,
    baseline_x_column: str,
    baseline_y_column: str,
) -> pd.DataFrame:
    require_columns(
        trajectory_all,
        [
            "sequence_frame_id",
            "token0_id",
            "variant",
            "reference_x_m",
            "reference_y_m",
            "reference_cumulative_distance_m",
            baseline_x_column,
            baseline_y_column,
        ],
        "S6A relative trajectory",
    )

    available_variants = sorted(
        trajectory_all["variant"].astype(str).unique()
    )

    if variant not in available_variants:
        raise ValueError(
            f"Requested variant {variant!r} was not found. "
            f"Available variants: {available_variants}"
        )

    trajectory = trajectory_all.loc[
        trajectory_all["variant"].astype(str) == variant
    ].copy()

    trajectory = trajectory.sort_values(
        "sequence_frame_id"
    ).reset_index(drop=True)

    if len(trajectory) != 1034:
        raise ValueError(
            f"Expected 1034 rows after variant filtering, "
            f"found {len(trajectory)}"
        )

    assert_unique(
        trajectory,
        ["sequence_frame_id"],
        "Filtered S6A trajectory",
    )
    assert_unique(
        trajectory,
        ["token0_id"],
        "Filtered S6A trajectory",
    )

    trajectory["baseline_x_m"] = pd.to_numeric(
        trajectory[baseline_x_column],
        errors="raise",
    )
    trajectory["baseline_y_m"] = pd.to_numeric(
        trajectory[baseline_y_column],
        errors="raise",
    )

    trajectory["baseline_error_m"] = np.hypot(
        trajectory["baseline_x_m"]
        - trajectory["reference_x_m"],
        trajectory["baseline_y_m"]
        - trajectory["reference_y_m"],
    )

    return trajectory


def prepare_policy_corrections(
    manifest: pd.DataFrame,
    policy_spec: dict[str, Any],
) -> pd.DataFrame:
    accept_column = policy_spec["accept_column"]

    if accept_column is None:
        return manifest.iloc[0:0].copy()

    required = [
        "sequence_frame_id",
        "token",
        accept_column,
        policy_spec["fix_x_column"],
        policy_spec["fix_y_column"],
        policy_spec["tile_column"],
        policy_spec["manifest_error_column"],
        policy_spec["hit_column"],
    ]

    if policy_spec["dangerous_column"] is not None:
        required.append(policy_spec["dangerous_column"])

    require_columns(
        manifest,
        required,
        f"S6B.0 manifest for {policy_spec['policy']}",
    )

    accepted = manifest.loc[
        bool_series(manifest[accept_column])
    ].copy()

    accepted = accepted.sort_values(
        "sequence_frame_id"
    ).reset_index(drop=True)

    assert_unique(
        accepted,
        ["sequence_frame_id"],
        f"Accepted corrections for {policy_spec['policy']}",
    )

    return accepted


def replay_policy(
    trajectory: pd.DataFrame,
    corrections: pd.DataFrame,
    policy_spec: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    policy_name = policy_spec["policy"]
    policy_type = policy_spec["policy_type"]

    correction_lookup = {
        int(row["sequence_frame_id"]): row
        for _, row in corrections.iterrows()
    }

    first_sequence_frame = int(
        trajectory["sequence_frame_id"].iloc[0]
    )
    start_distance = float(
        trajectory["reference_cumulative_distance_m"].iloc[0]
    )

    current_offset_x = 0.0
    current_offset_y = 0.0

    previous_event_distance: float | None = None
    event_index = 0

    replay_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []

    for _, frame in trajectory.iterrows():
        sequence_frame_id = int(frame["sequence_frame_id"])
        token0_id = int(frame["token0_id"])

        baseline_x = float(frame["baseline_x_m"])
        baseline_y = float(frame["baseline_y_m"])

        reference_x = float(frame["reference_x_m"])
        reference_y = float(frame["reference_y_m"])

        cumulative_distance = float(
            frame["reference_cumulative_distance_m"]
        )

        pre_x = baseline_x + current_offset_x
        pre_y = baseline_y + current_offset_y

        pre_error = float(
            math.hypot(
                pre_x - reference_x,
                pre_y - reference_y,
            )
        )

        correction_applied = False
        correction_role = "none"
        correction_token: int | None = None
        correction_tile_id: int | None = None
        correction_fix_x: float | None = None
        correction_fix_y: float | None = None
        correction_manifest_error: float | None = None
        correction_fix_error: float | None = None
        correction_error_consistency_delta: float | None = None
        correction_true: bool | None = None
        correction_false: bool | None = None
        correction_dangerous: bool | None = None
        correction_shift: float | None = None
        correction_improvement: float | None = None
        correction_gap: float | None = None

        event = correction_lookup.get(sequence_frame_id)

        if event is not None:
            correction_applied = True
            correction_token = int(event["token"])
            correction_tile_id = int(
                event[policy_spec["tile_column"]]
            )

            correction_fix_x = float(
                event[policy_spec["fix_x_column"]]
            )
            correction_fix_y = float(
                event[policy_spec["fix_y_column"]]
            )

            correction_manifest_error = float(
                event[policy_spec["manifest_error_column"]]
            )

            correction_true = bool_value(
                event[policy_spec["hit_column"]]
            )
            correction_false = not correction_true

            if policy_spec["dangerous_column"] is None:
                correction_dangerous = (
                    correction_false
                    and correction_manifest_error > 100.0
                )
            else:
                correction_dangerous = bool_value(
                    event[policy_spec["dangerous_column"]]
                )

            correction_role = (
                "initialization"
                if sequence_frame_id == first_sequence_frame
                else "relocalization"
            )

            correction_shift = float(
                math.hypot(
                    correction_fix_x - pre_x,
                    correction_fix_y - pre_y,
                )
            )

            # Position-only reset:
            # preserve the ORB heading/increments and replace only
            # the persistent global X/Y offset.
            offset_before_x = current_offset_x
            offset_before_y = current_offset_y

            current_offset_x = correction_fix_x - baseline_x
            current_offset_y = correction_fix_y - baseline_y

            fused_x = baseline_x + current_offset_x
            fused_y = baseline_y + current_offset_y

            correction_fix_error = float(
                math.hypot(
                    fused_x - reference_x,
                    fused_y - reference_y,
                )
            )

            correction_error_consistency_delta = abs(
                correction_fix_error
                - correction_manifest_error
            )

            correction_improvement = (
                pre_error - correction_fix_error
            )

            if previous_event_distance is None:
                correction_gap = (
                    cumulative_distance - start_distance
                )
            else:
                correction_gap = (
                    cumulative_distance
                    - previous_event_distance
                )

            previous_event_distance = cumulative_distance

            event_rows.append({
                "policy": policy_name,
                "policy_type": policy_type,
                "event_index": event_index,
                "correction_role": correction_role,
                "sequence_frame_id": sequence_frame_id,
                "token": correction_token,
                "tile_id": correction_tile_id,
                "reference_cumulative_distance_m":
                    cumulative_distance,
                "gap_since_previous_event_m":
                    correction_gap,
                "baseline_x_m": baseline_x,
                "baseline_y_m": baseline_y,
                "pre_correction_x_m": pre_x,
                "pre_correction_y_m": pre_y,
                "pre_correction_error_m": pre_error,
                "absolute_fix_x_m": correction_fix_x,
                "absolute_fix_y_m": correction_fix_y,
                "post_correction_x_m": fused_x,
                "post_correction_y_m": fused_y,
                "post_correction_error_m":
                    correction_fix_error,
                "manifest_fix_error_m_eval_only":
                    correction_manifest_error,
                "fix_error_consistency_delta_m":
                    correction_error_consistency_delta,
                "position_shift_applied_m":
                    correction_shift,
                "error_improvement_m":
                    correction_improvement,
                "error_worsened": correction_improvement < 0.0,
                "correction_true_eval_only":
                    correction_true,
                "correction_false_eval_only":
                    correction_false,
                "correction_dangerous_false_gt100m_eval_only":
                    correction_dangerous,
                "offset_before_x_m": offset_before_x,
                "offset_before_y_m": offset_before_y,
                "offset_after_x_m": current_offset_x,
                "offset_after_y_m": current_offset_y,
            })

            event_index += 1

        else:
            fused_x = pre_x
            fused_y = pre_y

        fused_error = float(
            math.hypot(
                fused_x - reference_x,
                fused_y - reference_y,
            )
        )

        replay_rows.append({
            "policy": policy_name,
            "policy_type": policy_type,
            "sequence_frame_id": sequence_frame_id,
            "token0_id": token0_id,
            "reference_cumulative_distance_m":
                cumulative_distance,
            "reference_x_m": reference_x,
            "reference_y_m": reference_y,
            "baseline_x_m": baseline_x,
            "baseline_y_m": baseline_y,
            "baseline_error_m":
                float(frame["baseline_error_m"]),
            "fused_x_m": fused_x,
            "fused_y_m": fused_y,
            "fused_error_m": fused_error,
            "active_offset_x_m": current_offset_x,
            "active_offset_y_m": current_offset_y,
            "correction_applied": correction_applied,
            "correction_role": correction_role,
            "correction_token": correction_token,
            "correction_tile_id": correction_tile_id,
            "correction_fix_x_m": correction_fix_x,
            "correction_fix_y_m": correction_fix_y,
            "correction_manifest_error_m_eval_only":
                correction_manifest_error,
            "correction_fix_error_m_eval_only":
                correction_fix_error,
            "correction_error_consistency_delta_m":
                correction_error_consistency_delta,
            "correction_true_eval_only": correction_true,
            "correction_false_eval_only": correction_false,
            "correction_dangerous_false_gt100m_eval_only":
                correction_dangerous,
            "pre_correction_error_m":
                pre_error if correction_applied else np.nan,
            "correction_position_shift_m":
                correction_shift,
            "correction_error_improvement_m":
                correction_improvement,
            "gap_since_previous_correction_m":
                correction_gap,
        })

    return (
        pd.DataFrame(replay_rows),
        pd.DataFrame(event_rows),
    )


def first_crossing_distance(
    policy_trajectory: pd.DataFrame,
    threshold_m: float,
) -> float | None:
    crossed = policy_trajectory.loc[
        policy_trajectory["fused_error_m"] > threshold_m
    ]

    if crossed.empty:
        return None

    return float(
        crossed["reference_cumulative_distance_m"].iloc[0]
    )


def event_impact_summary(
    events: pd.DataFrame,
) -> dict[str, Any]:
    if events.empty:
        return {
            "events": 0,
            "initializations": 0,
            "relocalizations": 0,
            "true_events_eval_only": 0,
            "false_events_eval_only": 0,
            "dangerous_false_events_gt100m_eval_only": 0,
            "events_improving_error": 0,
            "events_worsening_error": 0,
            "median_error_improvement_m": None,
            "median_true_event_improvement_m_eval_only": None,
            "median_false_event_improvement_m_eval_only": None,
            "maximum_false_event_worsening_m_eval_only": None,
        }

    true_mask = bool_series(
        events["correction_true_eval_only"]
    )
    false_mask = bool_series(
        events["correction_false_eval_only"]
    )
    dangerous_mask = bool_series(
        events[
            "correction_dangerous_false_gt100m_eval_only"
        ]
    )

    improvement = pd.to_numeric(
        events["error_improvement_m"],
        errors="coerce",
    )

    false_improvement = improvement.loc[false_mask]
    false_worsening = false_improvement.loc[
        false_improvement < 0.0
    ]

    return {
        "events": int(len(events)),
        "initializations": int(
            (events["correction_role"] == "initialization").sum()
        ),
        "relocalizations": int(
            (events["correction_role"] == "relocalization").sum()
        ),
        "true_events_eval_only": int(true_mask.sum()),
        "false_events_eval_only": int(false_mask.sum()),
        "dangerous_false_events_gt100m_eval_only":
            int(dangerous_mask.sum()),
        "events_improving_error": int((improvement > 0.0).sum()),
        "events_worsening_error": int((improvement < 0.0).sum()),
        "median_error_improvement_m":
            float(improvement.median()),
        "median_true_event_improvement_m_eval_only": (
            float(improvement.loc[true_mask].median())
            if true_mask.any()
            else None
        ),
        "median_false_event_improvement_m_eval_only": (
            float(false_improvement.median())
            if false_mask.any()
            else None
        ),
        "maximum_false_event_worsening_m_eval_only": (
            float(-false_worsening.min())
            if len(false_worsening)
            else 0.0
        ),
        "maximum_fix_error_consistency_delta_m": float(
            pd.to_numeric(
                events["fix_error_consistency_delta_m"],
                errors="coerce",
            ).max()
        ),
    }


def calculate_metrics(
    policy_trajectory: pd.DataFrame,
    events: pd.DataFrame,
    total_distance_m: float,
) -> dict[str, Any]:
    errors = pd.to_numeric(
        policy_trajectory["fused_error_m"],
        errors="raise",
    )

    metrics: dict[str, Any] = {
        "policy": str(policy_trajectory["policy"].iloc[0]),
        "policy_type": str(
            policy_trajectory["policy_type"].iloc[0]
        ),
        "frames": int(len(policy_trajectory)),
        "rmse_m": float(
            math.sqrt(float(np.mean(np.square(errors))))
        ),
        "mean_error_m": float(errors.mean()),
        "median_error_m": float(errors.median()),
        "p95_error_m": float(errors.quantile(0.95)),
        "max_error_m": float(errors.max()),
        "final_error_m": float(errors.iloc[-1]),
        "final_drift_per_100m": (
            float(errors.iloc[-1] / total_distance_m * 100.0)
            if total_distance_m > 0.0
            else None
        ),
        "failure_rate_gt40m": float(
            (errors > 40.0).mean()
        ),
        "failure_rate_gt80m": float(
            (errors > 80.0).mean()
        ),
        "frames_within_20m_rate": float(
            (errors <= 20.0).mean()
        ),
        "frames_within_40m_rate": float(
            (errors <= 40.0).mean()
        ),
        "frames_within_80m_rate": float(
            (errors <= 80.0).mean()
        ),
    }

    for threshold in (10.0, 20.0, 40.0, 80.0):
        key = f"first_crossing_gt{int(threshold)}m_distance_m"
        metrics[key] = first_crossing_distance(
            policy_trajectory,
            threshold,
        )

    metrics.update(event_impact_summary(events))

    return metrics


def build_gap_segments(
    policy_trajectory: pd.DataFrame,
    events: pd.DataFrame,
) -> pd.DataFrame:
    policy_name = str(policy_trajectory["policy"].iloc[0])

    first_frame = int(
        policy_trajectory["sequence_frame_id"].iloc[0]
    )
    last_frame = int(
        policy_trajectory["sequence_frame_id"].iloc[-1]
    )

    first_distance = float(
        policy_trajectory[
            "reference_cumulative_distance_m"
        ].iloc[0]
    )
    last_distance = float(
        policy_trajectory[
            "reference_cumulative_distance_m"
        ].iloc[-1]
    )

    segment_rows: list[dict[str, Any]] = []

    def add_segment(
        segment_index: int,
        gap_type: str,
        start_frame: int,
        end_frame: int,
        from_token: int | None,
        to_token: int | None,
        start_distance: float,
        end_distance: float,
    ) -> None:
        segment = policy_trajectory.loc[
            (
                policy_trajectory["sequence_frame_id"]
                >= start_frame
            )
            & (
                policy_trajectory["sequence_frame_id"]
                <= end_frame
            )
        ]

        if segment.empty:
            return

        max_row = segment.loc[
            segment["fused_error_m"].idxmax()
        ]

        segment_rows.append({
            "policy": policy_name,
            "segment_index": segment_index,
            "gap_type": gap_type,
            "from_token": from_token,
            "to_token": to_token,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "start_distance_m": start_distance,
            "end_distance_m": end_distance,
            "gap_m": end_distance - start_distance,
            "frames_in_segment": int(len(segment)),
            "mean_error_m": float(
                segment["fused_error_m"].mean()
            ),
            "median_error_m": float(
                segment["fused_error_m"].median()
            ),
            "p95_error_m": float(
                segment["fused_error_m"].quantile(0.95)
            ),
            "max_error_m": float(
                segment["fused_error_m"].max()
            ),
            "end_error_m": float(
                segment["fused_error_m"].iloc[-1]
            ),
            "max_error_frame": int(
                max_row["sequence_frame_id"]
            ),
            "max_error_token": int(
                max_row["token0_id"]
            ),
            "max_error_distance_m": float(
                max_row["reference_cumulative_distance_m"]
            ),
        })

    if events.empty:
        add_segment(
            segment_index=0,
            gap_type="whole_trajectory_no_corrections",
            start_frame=first_frame,
            end_frame=last_frame,
            from_token=None,
            to_token=None,
            start_distance=first_distance,
            end_distance=last_distance,
        )

        return pd.DataFrame(segment_rows)

    events = events.sort_values(
        "sequence_frame_id"
    ).reset_index(drop=True)

    first_event = events.iloc[0]

    add_segment(
        segment_index=0,
        gap_type="start_boundary",
        start_frame=first_frame,
        end_frame=int(first_event["sequence_frame_id"]),
        from_token=None,
        to_token=int(first_event["token"]),
        start_distance=first_distance,
        end_distance=float(
            first_event["reference_cumulative_distance_m"]
        ),
    )

    segment_index = 1

    for event_position in range(len(events) - 1):
        current = events.iloc[event_position]
        following = events.iloc[event_position + 1]

        add_segment(
            segment_index=segment_index,
            gap_type="between_corrections",
            start_frame=int(current["sequence_frame_id"]),
            end_frame=int(following["sequence_frame_id"]),
            from_token=int(current["token"]),
            to_token=int(following["token"]),
            start_distance=float(
                current["reference_cumulative_distance_m"]
            ),
            end_distance=float(
                following["reference_cumulative_distance_m"]
            ),
        )

        segment_index += 1

    last_event = events.iloc[-1]

    add_segment(
        segment_index=segment_index,
        gap_type="end_boundary",
        start_frame=int(last_event["sequence_frame_id"]),
        end_frame=last_frame,
        from_token=int(last_event["token"]),
        to_token=None,
        start_distance=float(
            last_event["reference_cumulative_distance_m"]
        ),
        end_distance=last_distance,
    )

    return pd.DataFrame(segment_rows)


def add_baseline_deltas(
    metrics: pd.DataFrame,
) -> pd.DataFrame:
    baseline_rows = metrics.loc[
        metrics["policy"] == "relative_only"
    ]

    if len(baseline_rows) != 1:
        raise ValueError(
            "Expected exactly one relative_only metrics row"
        )

    baseline = baseline_rows.iloc[0]

    for metric in [
        "rmse_m",
        "mean_error_m",
        "median_error_m",
        "p95_error_m",
        "max_error_m",
        "final_error_m",
        "final_drift_per_100m",
        "failure_rate_gt40m",
    ]:
        metrics[f"{metric}_delta_vs_relative"] = (
            metrics[metric] - float(baseline[metric])
        )

        if float(baseline[metric]) != 0.0:
            metrics[
                f"{metric}_relative_change_fraction"
            ] = (
                metrics[metric] - float(baseline[metric])
            ) / float(baseline[metric])
        else:
            metrics[
                f"{metric}_relative_change_fraction"
            ] = np.nan

    return metrics


def plot_trajectories(
    replay: pd.DataFrame,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 9))

    reference = replay.loc[
        replay["policy"] == "relative_only"
    ]

    ax.plot(
        reference["reference_x_m"],
        reference["reference_y_m"],
        linewidth=2.3,
        label="Reference — evaluation only",
    )

    for policy_name in replay["policy"].drop_duplicates():
        policy = replay.loc[
            replay["policy"] == policy_name
        ]

        ax.plot(
            policy["fused_x_m"],
            policy["fused_y_m"],
            linewidth=1.4,
            label=policy_name,
        )

    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("Local ENU X [m]")
    ax.set_ylabel("Local ENU Y [m]")
    ax.set_title(
        "S6B.1A position-only correction replay trajectories"
    )
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_error_by_distance(
    replay: pd.DataFrame,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(13, 6))

    for policy_name in replay["policy"].drop_duplicates():
        policy = replay.loc[
            replay["policy"] == policy_name
        ]

        ax.plot(
            policy["reference_cumulative_distance_m"],
            policy["fused_error_m"],
            linewidth=1.3,
            label=policy_name,
        )

    ax.axhline(
        20.0,
        linestyle="--",
        linewidth=1.0,
        label="20 m budget",
    )
    ax.axhline(
        40.0,
        linestyle="--",
        linewidth=1.0,
        label="40 m budget",
    )
    ax.axhline(
        80.0,
        linestyle="--",
        linewidth=1.0,
        label="80 m budget",
    )

    ax.set_xlabel(
        "Reference cumulative trajectory distance [m] — evaluation only"
    )
    ax.set_ylabel("Position error [m]")
    ax.set_title(
        "S6B.1A replay error along traj01"
    )
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_balanced_correction_impact(
    events: pd.DataFrame,
    output_path: Path,
) -> None:
    balanced = events.loc[
        events["policy"]
        == "balanced_position_reset_online"
    ].sort_values("reference_cumulative_distance_m")

    if balanced.empty:
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.scatter(
        balanced["reference_cumulative_distance_m"],
        balanced["pre_correction_error_m"],
        s=45,
        label="Error immediately before correction",
    )
    ax.scatter(
        balanced["reference_cumulative_distance_m"],
        balanced["post_correction_error_m"],
        s=45,
        label="Error immediately after correction",
    )

    ax.set_xlabel(
        "Reference cumulative trajectory distance [m] — evaluation only"
    )
    ax.set_ylabel("Position error [m]")
    ax.set_title(
        "S6B.1A balanced-gate correction impact"
    )
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_balanced_worst_gap(
    replay: pd.DataFrame,
    gaps: pd.DataFrame,
    output_path: Path,
) -> None:
    balanced_gaps = gaps.loc[
        gaps["policy"]
        == "balanced_position_reset_online"
    ]

    if balanced_gaps.empty:
        return

    worst = balanced_gaps.loc[
        balanced_gaps["gap_m"].idxmax()
    ]

    start_distance = float(worst["start_distance_m"])
    end_distance = float(worst["end_distance_m"])

    selected_policies = [
        "relative_only",
        "oracle_position_reset_eval_only",
        "balanced_position_reset_online",
        "permissive_position_reset_online",
    ]

    fig, ax = plt.subplots(figsize=(13, 6))

    for policy_name in selected_policies:
        policy = replay.loc[
            (replay["policy"] == policy_name)
            & (
                replay["reference_cumulative_distance_m"]
                >= start_distance
            )
            & (
                replay["reference_cumulative_distance_m"]
                <= end_distance
            )
        ]

        ax.plot(
            policy["reference_cumulative_distance_m"],
            policy["fused_error_m"],
            linewidth=1.4,
            label=policy_name,
        )

    ax.axvline(
        start_distance,
        linestyle="--",
        linewidth=1.0,
    )
    ax.axvline(
        end_distance,
        linestyle="--",
        linewidth=1.0,
    )

    ax.set_xlabel(
        "Reference cumulative trajectory distance [m] — evaluation only"
    )
    ax.set_ylabel("Position error [m]")
    ax.set_title(
        "S6B.1A largest balanced correction blackout\n"
        f"{start_distance:.1f} m to {end_distance:.1f} m "
        f"(gap {float(worst['gap_m']):.1f} m)"
    )
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> int:
    args = parse_args()

    for directory in (
        args.metadata_dir,
        args.report_dir,
        args.figure_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    if not args.correction_manifest.exists():
        raise FileNotFoundError(
            args.correction_manifest
        )

    if not args.relative_trajectory.exists():
        raise FileNotFoundError(
            args.relative_trajectory
        )

    manifest = pd.read_csv(
        args.correction_manifest,
        low_memory=False,
    )

    trajectory_all = pd.read_csv(
        args.relative_trajectory,
        low_memory=False,
    )

    trajectory = prepare_trajectory(
        trajectory_all=trajectory_all,
        variant=args.relative_variant,
        baseline_x_column=args.baseline_x_column,
        baseline_y_column=args.baseline_y_column,
    )

    require_columns(
        manifest,
        [
            "sequence_frame_id",
            "token",
            "chosen_tile_id",
            "oracle_tile_id",
            "chosen_abs_x_traj01_m",
            "chosen_abs_y_traj01_m",
            "oracle_abs_eval_only_x_traj01_m",
            "oracle_abs_eval_only_y_traj01_m",
            "chosen_error_m_eval_only",
            "oracle_processed_error_m",
            "hit_eval_only",
            "oracle_processed_hit_le_threshold",
            "dangerous_false_eval_only",
            "balanced_accept_online",
            "permissive_accept_online",
            "oracle_accept_eval_only",
        ],
        "S6B.0 correction manifest",
    )

    assert_unique(
        manifest,
        ["sequence_frame_id"],
        "S6B.0 correction manifest",
    )

    valid_frames = set(
        trajectory["sequence_frame_id"].astype(int)
    )

    manifest_frames = set(
        manifest["sequence_frame_id"].astype(int)
    )

    missing_frames = sorted(
        manifest_frames - valid_frames
    )

    if missing_frames:
        raise ValueError(
            "Correction manifest contains frames not present "
            f"in the relative trajectory: {missing_frames[:20]}"
        )

    total_distance_m = float(
        trajectory[
            "reference_cumulative_distance_m"
        ].iloc[-1]
        - trajectory[
            "reference_cumulative_distance_m"
        ].iloc[0]
    )

    all_replay: list[pd.DataFrame] = []
    all_events: list[pd.DataFrame] = []
    all_gaps: list[pd.DataFrame] = []
    metrics_rows: list[dict[str, Any]] = []

    for policy_spec in POLICIES:
        corrections = prepare_policy_corrections(
            manifest,
            policy_spec,
        )

        policy_replay, policy_events = replay_policy(
            trajectory=trajectory,
            corrections=corrections,
            policy_spec=policy_spec,
        )

        policy_gaps = build_gap_segments(
            policy_trajectory=policy_replay,
            events=policy_events,
        )

        policy_metrics = calculate_metrics(
            policy_trajectory=policy_replay,
            events=policy_events,
            total_distance_m=total_distance_m,
        )

        all_replay.append(policy_replay)

        if not policy_events.empty:
            all_events.append(policy_events)

        all_gaps.append(policy_gaps)
        metrics_rows.append(policy_metrics)

    replay = pd.concat(
        all_replay,
        ignore_index=True,
    )

    if all_events:
        events = pd.concat(
            all_events,
            ignore_index=True,
        )
    else:
        events = pd.DataFrame()

    gaps = pd.concat(
        all_gaps,
        ignore_index=True,
    )

    if not events.empty:
        maximum_fix_error_delta = float(
            pd.to_numeric(
                events["fix_error_consistency_delta_m"],
                errors="coerce",
            ).max()
        )

        if (
            maximum_fix_error_delta
            > FIX_ERROR_CONSISTENCY_TOLERANCE_M
        ):

            raise ValueError(
                "Absolute-fix coordinates do not agree with "
                "their stored evaluation errors. This usually "
                "means that trajectory and map coordinates use "
                "different origins or coordinate frames. "
                f"Tolerance="
                f"{FIX_ERROR_CONSISTENCY_TOLERANCE_M:.3f} m; "
                f"maximum error delta="
                f"{maximum_fix_error_delta:.3f} m"
            )
        
    metrics = pd.DataFrame(metrics_rows)
    metrics = add_baseline_deltas(metrics)

    expected_event_counts = {
        "oracle_position_reset_eval_only": 102,
        "balanced_position_reset_online": 33,
        "permissive_position_reset_online": 80,
    }

    for policy_name, expected_count in expected_event_counts.items():
        actual_count = int(
            (
                events["policy"] == policy_name
            ).sum()
        )

        if actual_count != expected_count:
            raise ValueError(
                f"Frozen correction-count check failed for "
                f"{policy_name}: expected {expected_count}, "
                f"found {actual_count}"
            )

    replay_path = (
        args.metadata_dir
        / "s6b1a_position_only_replay_trajectories.csv"
    )
    events_path = (
        args.metadata_dir
        / "s6b1a_position_only_correction_events.csv"
    )
    metrics_path = (
        args.metadata_dir
        / "s6b1a_position_only_replay_metrics.csv"
    )
    gaps_path = (
        args.metadata_dir
        / "s6b1a_position_only_gap_segments.csv"
    )
    summary_path = (
        args.report_dir
        / "s6b1a_position_only_replay_summary.json"
    )

    replay.to_csv(replay_path, index=False)
    events.to_csv(events_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    gaps.to_csv(gaps_path, index=False)

    policy_summaries = {
        row["policy"]: {
            key: clean_json(value)
            for key, value in row.items()
        }
        for row in metrics.to_dict(orient="records")
    }

    worst_gaps: dict[str, Any] = {}

    for policy_name in gaps["policy"].drop_duplicates():
        policy_gaps = gaps.loc[
            gaps["policy"] == policy_name
        ]

        worst = policy_gaps.loc[
            policy_gaps["gap_m"].idxmax()
        ]

        worst_gaps[policy_name] = {
            key: clean_json(value)
            for key, value in worst.to_dict().items()
        }

    summary = {
        "stage": "S6B.1A",
        "title": "Controlled position-only correction replay",
        "relative_variant": args.relative_variant,
        "baseline_x_column": args.baseline_x_column,
        "baseline_y_column": args.baseline_y_column,
        "reference_total_distance_m_eval_only":
            total_distance_m,
        "important_warning": (
            "This replay uses prefix-aligned S6A coordinates. "
            "It is a controlled evaluation replay and must not "
            "be described as a fully online/deployable estimator."
        ),
        "correction_rule": (
            "At each accepted frame, set persistent offset to "
            "absolute_fix_xy minus baseline_xy. Preserve all "
            "relative increments and heading."
        ),
        "acceptance_rule": {
            "oracle": (
                "Evaluation-only oracle_processed_hit_le_threshold"
            ),
            "balanced": (
                "Previously frozen online-safe S5C.3 "
                "balanced_accept_online flag"
            ),
            "permissive": (
                "Previously frozen online-safe S5C.3 "
                "permissive_accept_online flag"
            ),
        },
        "policy_metrics": policy_summaries,
        "worst_gap_by_policy": worst_gaps,
        "outputs": {
            "replay_trajectories": replay_path,
            "correction_events": events_path,
            "metrics": metrics_path,
            "gap_segments": gaps_path,
        },
    }

    summary_path.write_text(
        json.dumps(
            clean_json(summary),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    plot_trajectories(
        replay,
        args.figure_dir
        / "s6b1a_position_only_trajectory_comparison.png",
    )

    plot_error_by_distance(
        replay,
        args.figure_dir
        / "s6b1a_position_only_error_by_distance.png",
    )

    plot_balanced_correction_impact(
        events,
        args.figure_dir
        / "s6b1a_balanced_correction_impact.png",
    )

    plot_balanced_worst_gap(
        replay,
        gaps,
        args.figure_dir
        / "s6b1a_balanced_worst_gap_error.png",
    )

    print("S6B.1A Controlled Position-Only Correction Replay")
    print("--------------------------------------------------")
    print(f"Relative variant:        {args.relative_variant}")
    print(
        f"Baseline columns:        "
        f"{args.baseline_x_column}, "
        f"{args.baseline_y_column}"
    )
    print(f"Frames per policy:       {len(trajectory)}")
    print(f"Reference distance:      {total_distance_m:.3f} m")
    print()

    print(
        metrics[
            [
                "policy",
                "events",
                "true_events_eval_only",
                "false_events_eval_only",
                "rmse_m",
                "p95_error_m",
                "max_error_m",
                "final_error_m",
                "final_drift_per_100m",
                "failure_rate_gt40m",
            ]
        ].to_string(index=False)
    )

    print()
    print(f"Replay trajectories: {replay_path}")
    print(f"Correction events:   {events_path}")
    print(f"Metrics:             {metrics_path}")
    print(f"Gap segments:        {gaps_path}")
    print(f"Summary:             {summary_path}")
    print(f"Figures:             {args.figure_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
