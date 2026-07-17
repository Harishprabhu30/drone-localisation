#!/usr/bin/env python3
"""S6B.1C.2 — Adaptive soft-correction replay sweep.

Tests hard and soft position corrections using:

    balanced confidence
    balanced OR temporal-r40
    balanced OR temporal-r60

For a soft correction:

    innovation = absolute_fix - current_fused_position
    offset_new = offset_old + alpha * innovation

Frame-zero initialization always uses alpha=1.0 so soft policies do
not receive an artificial advantage from the evaluation-aligned
relative starting position.

Ground-truth labels are used only for evaluation after policy masks
and correction strengths have been defined.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_DIAGNOSTICS = Path(
    "outputs/satloc/metadata/s6b_relative_absolute/"
    "s6b1c0_temporal_agreement_diagnostics.csv"
)

DEFAULT_TRAJECTORY = Path(
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

FIX_ERROR_CONSISTENCY_TOLERANCE_M = 10.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--diagnostics",
        type=Path,
        default=DEFAULT_DIAGNOSTICS,
    )
    parser.add_argument(
        "--trajectory",
        type=Path,
        default=DEFAULT_TRAJECTORY,
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


def load_s6b1a_module():
    path = Path(__file__).with_name(
        "s6b_1a_position_only_correction_replay.py"
    )

    if not path.exists():
        raise FileNotFoundError(path)

    spec = importlib.util.spec_from_file_location(
        "s6b1a_replay",
        path,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Could not import replay helpers from {path}"
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def require_columns(
    frame: pd.DataFrame,
    columns: list[str],
    name: str,
) -> None:
    missing = sorted(set(columns) - set(frame.columns))

    if missing:
        raise ValueError(
            f"{name} is missing required columns: {missing}"
        )


def bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)

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


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, np.generic):
        return json_safe(value.item())

    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return value

    if pd.isna(value):
        return None

    return value


def alpha_name(value: float) -> str:
    return f"{int(round(value * 100)):03d}"


def build_masks(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()

    balanced = bool_series(
        frame["balanced_accept_online"]
    )

    permissive = bool_series(
        frame["permissive_accept_online"]
    )

    initialization = (
        frame["sequence_frame_id"]
        == frame["sequence_frame_id"].min()
    )

    temporal_r40 = (
        permissive
        & (
            initialization
            | (
                pd.to_numeric(
                    frame[
                        "temporal_support_any_r40_count"
                    ],
                    errors="raise",
                )
                >= 1
            )
        )
    )

    temporal_r60 = (
        permissive
        & (
            initialization
            | (
                pd.to_numeric(
                    frame[
                        "temporal_support_any_r60_count"
                    ],
                    errors="raise",
                )
                >= 1
            )
        )
    )

    frame["balanced_mask_online"] = balanced
    frame["permissive_mask_online"] = permissive
    frame["temporal_r40_mask_online"] = temporal_r40
    frame["temporal_r60_mask_online"] = temporal_r60

    frame["hybrid_r40_mask_online"] = (
        balanced | temporal_r40
    )

    frame["hybrid_r60_mask_online"] = (
        balanced | temporal_r60
    )

    frame["hybrid_r40_temporal_only_online"] = (
        temporal_r40 & ~balanced
    )

    frame["hybrid_r60_temporal_only_online"] = (
        temporal_r60 & ~balanced
    )

    expected_counts = {
        "balanced_mask_online": 33,
        "permissive_mask_online": 80,
        "temporal_r40_mask_online": 46,
        "temporal_r60_mask_online": 62,
        "hybrid_r40_mask_online": 61,
        "hybrid_r60_mask_online": 72,
    }

    for column, expected in expected_counts.items():
        actual = int(bool_series(frame[column]).sum())

        if actual != expected:
            raise ValueError(
                f"Frozen mask-count check failed for {column}: "
                f"expected {expected}, found {actual}"
            )

    return frame


def policy_definitions() -> list[dict[str, Any]]:
    policies: list[dict[str, Any]] = [
        {
            "policy": "relative_only",
            "policy_type": "relative_baseline",
            "accept_column": None,
            "mode": "none",
            "uniform_alpha": None,
            "balanced_alpha": None,
            "temporal_alpha": None,
        },
        {
            "policy": "balanced_hard_reset_online",
            "policy_type": "balanced_hard_reset",
            "accept_column": "balanced_mask_online",
            "mode": "hard",
            "uniform_alpha": 1.0,
            "balanced_alpha": 1.0,
            "temporal_alpha": None,
        },
        {
            "policy": "permissive_hard_reset_online",
            "policy_type": "permissive_hard_reset",
            "accept_column": "permissive_mask_online",
            "mode": "hard",
            "uniform_alpha": 1.0,
            "balanced_alpha": None,
            "temporal_alpha": None,
        },
        {
            "policy": "hybrid_r40_hard_reset_online",
            "policy_type": "hybrid_r40_hard_reset",
            "accept_column": "hybrid_r40_mask_online",
            "mode": "hard",
            "uniform_alpha": 1.0,
            "balanced_alpha": 1.0,
            "temporal_alpha": 1.0,
        },
        {
            "policy": "hybrid_r60_hard_reset_online",
            "policy_type": "hybrid_r60_hard_reset",
            "accept_column": "hybrid_r60_mask_online",
            "mode": "hard",
            "uniform_alpha": 1.0,
            "balanced_alpha": 1.0,
            "temporal_alpha": 1.0,
        },
    ]

    for radius in [40, 60]:
        for alpha in [0.25, 0.50, 0.75]:
            policies.append(
                {
                    "policy": (
                        f"hybrid_r{radius}_uniform_"
                        f"a{alpha_name(alpha)}_online"
                    ),
                    "policy_type": (
                        f"hybrid_r{radius}_uniform_soft"
                    ),
                    "accept_column":
                        f"hybrid_r{radius}_mask_online",
                    "mode": "uniform",
                    "uniform_alpha": alpha,
                    "balanced_alpha": alpha,
                    "temporal_alpha": alpha,
                }
            )

    for radius in [40, 60]:
        for balanced_alpha in [0.50, 0.75, 1.00]:
            for temporal_alpha in [0.25, 0.50, 0.75]:
                policies.append(
                    {
                        "policy": (
                            f"hybrid_r{radius}_adaptive_"
                            f"b{alpha_name(balanced_alpha)}_"
                            f"t{alpha_name(temporal_alpha)}_online"
                        ),
                        "policy_type": (
                            f"hybrid_r{radius}_adaptive_soft"
                        ),
                        "accept_column":
                            f"hybrid_r{radius}_mask_online",
                        "mode": "adaptive",
                        "uniform_alpha": None,
                        "balanced_alpha":
                            balanced_alpha,
                        "temporal_alpha":
                            temporal_alpha,
                    }
                )

    return policies


def correction_alpha(
    event: pd.Series,
    policy: dict[str, Any],
    initialization: bool,
) -> tuple[float, str]:
    # Initialization remains a hard absolute placement for every
    # correction policy. Otherwise soft policies would benefit from
    # the evaluation-aligned relative starting point.
    if initialization:
        return 1.0, "initialization_hard"

    mode = policy["mode"]

    if mode == "hard":
        return 1.0, "hard_reset"

    if mode == "uniform":
        return (
            float(policy["uniform_alpha"]),
            "uniform_soft",
        )

    if mode == "adaptive":
        if bool_value(event["balanced_accept_online"]):
            return (
                float(policy["balanced_alpha"]),
                "balanced_confidence",
            )

        return (
            float(policy["temporal_alpha"]),
            "temporal_only_support",
        )

    raise ValueError(
        f"Unsupported correction mode: {mode}"
    )


def replay_policy(
    trajectory: pd.DataFrame,
    correction_frame: pd.DataFrame,
    policy: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    policy_name = policy["policy"]
    policy_type = policy["policy_type"]
    accept_column = policy["accept_column"]

    if accept_column is None:
        accepted = correction_frame.iloc[0:0].copy()
    else:
        accepted = correction_frame.loc[
            bool_series(correction_frame[accept_column])
        ].copy()

    if accepted.duplicated("sequence_frame_id").any():
        raise ValueError(
            f"Duplicate accepted frames for {policy_name}"
        )

    lookup = {
        int(row["sequence_frame_id"]): row
        for _, row in accepted.iterrows()
    }

    first_frame = int(
        trajectory["sequence_frame_id"].iloc[0]
    )

    first_distance = float(
        trajectory[
            "reference_cumulative_distance_m"
        ].iloc[0]
    )

    current_offset_x = 0.0
    current_offset_y = 0.0

    previous_event_distance: float | None = None
    event_index = 0

    trajectory_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []

    for _, row in trajectory.iterrows():
        frame_id = int(row["sequence_frame_id"])
        token0_id = int(row["token0_id"])

        baseline_x = float(row["baseline_x_m"])
        baseline_y = float(row["baseline_y_m"])

        reference_x = float(row["reference_x_m"])
        reference_y = float(row["reference_y_m"])

        distance_m = float(
            row["reference_cumulative_distance_m"]
        )

        pre_x = baseline_x + current_offset_x
        pre_y = baseline_y + current_offset_y

        pre_error = float(
            math.hypot(
                pre_x - reference_x,
                pre_y - reference_y,
            )
        )

        event = lookup.get(frame_id)

        correction_applied = event is not None
        correction_role = "none"
        correction_alpha_value: float | None = None
        correction_alpha_source: str | None = None
        correction_token: int | None = None
        correction_tile_id: int | None = None
        absolute_fix_x: float | None = None
        absolute_fix_y: float | None = None
        innovation_m: float | None = None
        position_shift_m: float | None = None
        post_error: float | None = None
        error_improvement: float | None = None
        correction_gap_m: float | None = None
        correction_true: bool | None = None
        correction_false: bool | None = None
        correction_dangerous: bool | None = None
        manifest_fix_error: float | None = None
        fix_coordinate_error: float | None = None
        fix_error_delta: float | None = None

        if event is not None:
            initialization = frame_id == first_frame

            correction_role = (
                "initialization"
                if initialization
                else "relocalization"
            )

            correction_alpha_value, correction_alpha_source = (
                correction_alpha(
                    event=event,
                    policy=policy,
                    initialization=initialization,
                )
            )

            correction_token = int(event["token"])
            correction_tile_id = int(
                event["chosen_tile_id"]
            )

            absolute_fix_x = float(
                event["chosen_abs_x_traj01_m"]
            )
            absolute_fix_y = float(
                event["chosen_abs_y_traj01_m"]
            )

            innovation_x = absolute_fix_x - pre_x
            innovation_y = absolute_fix_y - pre_y

            innovation_m = float(
                math.hypot(
                    innovation_x,
                    innovation_y,
                )
            )

            offset_before_x = current_offset_x
            offset_before_y = current_offset_y

            current_offset_x += (
                correction_alpha_value * innovation_x
            )
            current_offset_y += (
                correction_alpha_value * innovation_y
            )

            fused_x = baseline_x + current_offset_x
            fused_y = baseline_y + current_offset_y

            position_shift_m = float(
                correction_alpha_value * innovation_m
            )

            post_error = float(
                math.hypot(
                    fused_x - reference_x,
                    fused_y - reference_y,
                )
            )

            error_improvement = pre_error - post_error 

            correction_true = bool_value(
                event["hit_eval_only"]
            )
            correction_false = not correction_true
            correction_dangerous = bool_value(
                event["dangerous_false_eval_only"]
            )

            manifest_fix_error = float(
                event["chosen_error_m_eval_only"]
            )

            fix_coordinate_error = float(
                math.hypot(
                    absolute_fix_x - reference_x,
                    absolute_fix_y - reference_y,
                )
            )

            fix_error_delta = abs(
                fix_coordinate_error
                - manifest_fix_error
            )

            if fix_error_delta > (
                FIX_ERROR_CONSISTENCY_TOLERANCE_M
            ):
                raise ValueError(
                    "Absolute coordinate consistency failed at "
                    f"frame {frame_id}: "
                    f"delta={fix_error_delta:.3f} m"
                )

            if previous_event_distance is None:
                correction_gap_m = (
                    distance_m - first_distance
                )
            else:
                correction_gap_m = (
                    distance_m
                    - previous_event_distance
                )

            previous_event_distance = distance_m

            event_rows.append(
                {
                    "policy": policy_name,
                    "policy_type": policy_type,
                    "event_index": event_index,
                    "correction_role": correction_role,
                    "sequence_frame_id": frame_id,
                    "token": correction_token,
                    "tile_id": correction_tile_id,
                    "reference_cumulative_distance_m":
                        distance_m,
                    "gap_since_previous_event_m":
                        correction_gap_m,
                    "baseline_x_m": baseline_x,
                    "baseline_y_m": baseline_y,
                    "pre_correction_x_m": pre_x,
                    "pre_correction_y_m": pre_y,
                    "pre_correction_error_m": pre_error,
                    "absolute_fix_x_m": absolute_fix_x,
                    "absolute_fix_y_m": absolute_fix_y,
                    "absolute_fix_coordinate_error_m_eval_only":
                        fix_coordinate_error,
                    "manifest_fix_error_m_eval_only":
                        manifest_fix_error,
                    "fix_error_consistency_delta_m":
                        fix_error_delta,
                    "correction_alpha":
                        correction_alpha_value,
                    "correction_alpha_source":
                        correction_alpha_source,
                    "absolute_innovation_m":
                        innovation_m,
                    "position_shift_applied_m":
                        position_shift_m,
                    "post_correction_x_m": fused_x,
                    "post_correction_y_m": fused_y,
                    "post_correction_error_m": post_error,
                    "error_improvement_m":
                        error_improvement,
                    "error_worsened":
                        error_improvement < 0.0,
                    "correction_true_eval_only":
                        correction_true,
                    "correction_false_eval_only":
                        correction_false,
                    "correction_dangerous_false_gt100m_eval_only":
                        correction_dangerous,
                    "offset_before_x_m":
                        offset_before_x,
                    "offset_before_y_m":
                        offset_before_y,
                    "offset_after_x_m":
                        current_offset_x,
                    "offset_after_y_m":
                        current_offset_y,
                }
            )

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

        trajectory_rows.append(
            {
                "policy": policy_name,
                "policy_type": policy_type,
                "sequence_frame_id": frame_id,
                "token0_id": token0_id,
                "reference_cumulative_distance_m":
                    distance_m,
                "reference_x_m": reference_x,
                "reference_y_m": reference_y,
                "baseline_x_m": baseline_x,
                "baseline_y_m": baseline_y,
                "baseline_error_m":
                    float(row["baseline_error_m"]),
                "fused_x_m": fused_x,
                "fused_y_m": fused_y,
                "fused_error_m": fused_error,
                "active_offset_x_m":
                    current_offset_x,
                "active_offset_y_m":
                    current_offset_y,
                "correction_applied":
                    correction_applied,
                "correction_role":
                    correction_role,
                "correction_token":
                    correction_token,
                "correction_tile_id":
                    correction_tile_id,
                "correction_alpha":
                    correction_alpha_value,
                "correction_alpha_source":
                    correction_alpha_source,
                "correction_absolute_innovation_m":
                    innovation_m,
                "correction_position_shift_m":
                    position_shift_m,
                "pre_correction_error_m": (
                    pre_error
                    if correction_applied
                    else np.nan
                ),
                "correction_error_improvement_m":
                    error_improvement,
                "correction_true_eval_only":
                    correction_true,
                "correction_false_eval_only":
                    correction_false,
                "correction_dangerous_false_gt100m_eval_only":
                    correction_dangerous,
            }
        )

    return (
        pd.DataFrame(trajectory_rows),
        pd.DataFrame(event_rows),
    )


def add_policy_statistics(
    metrics: dict[str, Any],
    events: pd.DataFrame,
    gaps: pd.DataFrame,
) -> dict[str, Any]:
    result = dict(metrics)

    if events.empty:
        result.update(
            {
                "accepted": 0,
                "precision_eval_only": None,
                "mean_relocalization_alpha": None,
                "median_absolute_innovation_m": None,
                "median_position_shift_applied_m": None,
                "coverage_gap_including_boundaries_max_m":
                    None,
            }
        )

        return result

    true_mask = bool_series(
        events["correction_true_eval_only"]
    )

    relocalizations = events.loc[
        events["correction_role"] == "relocalization"
    ]

    result["accepted"] = int(len(events))
    result["precision_eval_only"] = float(
        true_mask.mean()
    )

    result["mean_relocalization_alpha"] = (
        float(
            relocalizations[
                "correction_alpha"
            ].mean()
        )
        if len(relocalizations)
        else None
    )

    result["median_absolute_innovation_m"] = float(
        events["absolute_innovation_m"].median()
    )

    result["median_position_shift_applied_m"] = float(
        events["position_shift_applied_m"].median()
    )

    result[
        "coverage_gap_including_boundaries_max_m"
    ] = (
        float(gaps["gap_m"].max())
        if len(gaps)
        else None
    )

    return result


def main() -> int:
    args = parse_args()
    replay_module = load_s6b1a_module()

    for directory in [
        args.metadata_dir,
        args.report_dir,
        args.figure_dir,
    ]:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    diagnostics = pd.read_csv(
        args.diagnostics,
        low_memory=False,
    )

    trajectory_all = pd.read_csv(
        args.trajectory,
        low_memory=False,
    )

    require_columns(
        diagnostics,
        [
            "sequence_frame_id",
            "token",
            "chosen_tile_id",
            "chosen_abs_x_traj01_m",
            "chosen_abs_y_traj01_m",
            "chosen_error_m_eval_only",
            "hit_eval_only",
            "dangerous_false_eval_only",
            "balanced_accept_online",
            "permissive_accept_online",
            "temporal_support_any_r40_count",
            "temporal_support_any_r60_count",
        ],
        "S6B.1C.0 diagnostics",
    )

    diagnostics = diagnostics.sort_values(
        "sequence_frame_id"
    ).reset_index(drop=True)

    frame = build_masks(diagnostics)

    trajectory = replay_module.prepare_trajectory(
        trajectory_all=trajectory_all,
        variant=args.relative_variant,
        baseline_x_column=args.baseline_x_column,
        baseline_y_column=args.baseline_y_column,
    )

    total_distance_m = float(
        trajectory[
            "reference_cumulative_distance_m"
        ].iloc[-1]
        - trajectory[
            "reference_cumulative_distance_m"
        ].iloc[0]
    )

    policy_list = policy_definitions()

    replay_frames: list[pd.DataFrame] = []
    event_frames: list[pd.DataFrame] = []
    gap_frames: list[pd.DataFrame] = []
    metrics_rows: list[dict[str, Any]] = []

    for policy in policy_list:
        policy_trajectory, events = replay_policy(
            trajectory=trajectory,
            correction_frame=frame,
            policy=policy,
        )

        gaps = replay_module.build_gap_segments(
            policy_trajectory=policy_trajectory,
            events=events,
        )

        basic_metrics = replay_module.calculate_metrics(
            policy_trajectory=policy_trajectory,
            events=events,
            total_distance_m=total_distance_m,
        )

        metrics = add_policy_statistics(
            metrics=basic_metrics,
            events=events,
            gaps=gaps,
        )

        metrics["mode"] = policy["mode"]
        metrics["uniform_alpha"] = (
            policy["uniform_alpha"]
        )
        metrics["balanced_alpha"] = (
            policy["balanced_alpha"]
        )
        metrics["temporal_alpha"] = (
            policy["temporal_alpha"]
        )

        replay_frames.append(policy_trajectory)
        gap_frames.append(gaps)
        metrics_rows.append(metrics)

        if not events.empty:
            event_frames.append(events)

    replay = pd.concat(
        replay_frames,
        ignore_index=True,
    )

    events = pd.concat(
        event_frames,
        ignore_index=True,
    )

    gaps = pd.concat(
        gap_frames,
        ignore_index=True,
    )

    metrics = pd.DataFrame(metrics_rows)
    metrics = replay_module.add_baseline_deltas(
        metrics
    )

    candidates = metrics.loc[
        (
            metrics["policy"] != "relative_only"
        )
        & (
            metrics[
                "dangerous_false_events_gt100m_eval_only"
            ]
            == 0
        )
    ].copy()

    candidates = candidates.sort_values(
        [
            "rmse_m",
            "p95_error_m",
            "failure_rate_gt40m",
        ]
    ).reset_index(drop=True)

    if candidates.empty:
        raise ValueError(
            "No zero-danger candidate policies were produced"
        )

    best_policy = str(
        candidates.iloc[0]["policy"]
    )

    masks_path = (
        args.metadata_dir
        / "s6b1c2_adaptive_soft_policy_masks.csv"
    )

    replay_path = (
        args.metadata_dir
        / "s6b1c2_adaptive_soft_trajectories.csv"
    )

    events_path = (
        args.metadata_dir
        / "s6b1c2_adaptive_soft_events.csv"
    )

    metrics_path = (
        args.metadata_dir
        / "s6b1c2_adaptive_soft_metrics.csv"
    )

    gaps_path = (
        args.metadata_dir
        / "s6b1c2_adaptive_soft_gap_segments.csv"
    )

    summary_path = (
        args.report_dir
        / "s6b1c2_adaptive_soft_summary.json"
    )

    frame[
        [
            "sequence_frame_id",
            "token",
            "balanced_mask_online",
            "permissive_mask_online",
            "temporal_r40_mask_online",
            "temporal_r60_mask_online",
            "hybrid_r40_mask_online",
            "hybrid_r60_mask_online",
            "hybrid_r40_temporal_only_online",
            "hybrid_r60_temporal_only_online",
            "hit_eval_only",
            "dangerous_false_eval_only",
        ]
    ].to_csv(
        masks_path,
        index=False,
    )

    replay.to_csv(replay_path, index=False)
    events.to_csv(events_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    gaps.to_csv(gaps_path, index=False)

    summary = {
        "stage": "S6B.1C.2",
        "title": (
            "Adaptive soft-correction controlled replay sweep"
        ),
        "relative_variant": args.relative_variant,
        "reference_total_distance_m_eval_only":
            total_distance_m,
        "initialization_rule": (
            "Frame-zero initialization always uses alpha=1.0. "
            "Soft alpha applies only to later relocalizations."
        ),
        "adaptive_rule": (
            "Balanced observations use balanced_alpha. "
            "Temporal-only observations use temporal_alpha."
        ),
        "selection_warning": (
            "The reported best policy is selected using evaluation "
            "metrics on traj01 and is exploratory, not independently "
            "validated."
        ),
        "best_zero_danger_policy_by_rmse_eval_only":
            best_policy,
        "best_zero_danger_metrics_eval_only": {
            key: json_safe(value)
            for key, value in candidates.iloc[0].to_dict().items()
        },
        "outputs": {
            "policy_masks": masks_path,
            "trajectories": replay_path,
            "events": events_path,
            "metrics": metrics_path,
            "gap_segments": gaps_path,
        },
    }

    summary_path.write_text(
        json.dumps(
            json_safe(summary),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    selected_policies = [
        "relative_only",
        "balanced_hard_reset_online",
        "permissive_hard_reset_online",
        "hybrid_r60_hard_reset_online",
        best_policy,
    ]

    selected_policies = list(
        dict.fromkeys(selected_policies)
    )

    selected_replay = replay.loc[
        replay["policy"].isin(selected_policies)
    ].copy()

    replay_module.plot_error_by_distance(
        selected_replay,
        args.figure_dir
        / "s6b1c2_best_soft_error_by_distance.png",
    )

    replay_module.plot_trajectories(
        selected_replay,
        args.figure_dir
        / "s6b1c2_best_soft_trajectory_comparison.png",
    )

    print("S6B.1C.2 Adaptive Soft-Correction Sweep")
    print("---------------------------------------")
    print(f"Policies tested:       {len(metrics)}")
    print(f"Frames per policy:     {len(trajectory)}")
    print(f"Reference distance:    {total_distance_m:.3f} m")
    print(f"Best zero-danger RMSE: {best_policy}")
    print()

    display_columns = [
        "policy",
        "mode",
        "accepted",
        "true_events_eval_only",
        "false_events_eval_only",
        "dangerous_false_events_gt100m_eval_only",
        "precision_eval_only",
        "mean_relocalization_alpha",
        "rmse_m",
        "p95_error_m",
        "max_error_m",
        "final_error_m",
        "failure_rate_gt40m",
    ]

    print("Top zero-danger policies")
    print("------------------------")

    print(
        candidates[
            display_columns
        ]
        .head(15)
        .to_string(index=False)
    )

    print()
    print(f"Policy masks: {masks_path}")
    print(f"Trajectories: {replay_path}")
    print(f"Events:       {events_path}")
    print(f"Metrics:      {metrics_path}")
    print(f"Gap segments: {gaps_path}")
    print(f"Summary:      {summary_path}")
    print(f"Figures:      {args.figure_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
