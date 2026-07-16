#!/usr/bin/env python3
"""S6B.1C.1 — Temporal-confirmation hard-reset replay.

Compares:
    - relative-only
    - balanced base
    - permissive base
    - temporally confirmed permissive, residual <= 40 m
    - temporally confirmed permissive, residual <= 60 m
    - balanced OR temporal-40
    - balanced OR temporal-60

All realistic masks use online-safe information only. Ground-truth
labels are used afterward for evaluation and reporting.

This remains a controlled evaluation replay because the S6A baseline
uses prefix-aligned coordinates.

Command Used:

export PYTHONPATH=$PWD/src

python scripts/satloc/s6b/s6b_1c1_temporal_hard_reset_replay.py \
  2>&1 | tee \
  outputs/satloc/reports/s6b_relative_absolute/s6b1c1_temporal_hard_reset_replay.log

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


def load_s6b1a_module():
    module_path = Path(__file__).with_name(
        "s6b_1a_position_only_correction_replay.py"
    )

    if not module_path.exists():
        raise FileNotFoundError(
            f"Required S6B.1A module not found: {module_path}"
        )

    spec = importlib.util.spec_from_file_location(
        "s6b1a_replay",
        module_path,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Could not load S6B.1A module: {module_path}"
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


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


def require_columns(
    frame: pd.DataFrame,
    columns: list[str],
    name: str,
) -> None:
    missing = sorted(set(columns) - set(frame.columns))

    if missing:
        raise ValueError(
            f"{name} is missing columns: {missing}"
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


def build_policy_masks(
    frame: pd.DataFrame,
) -> pd.DataFrame:
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

    frame["balanced_hard_accept_online"] = balanced
    frame["permissive_hard_accept_online"] = permissive

    frame[
        "temporal_r40_hard_accept_online"
    ] = temporal_r40

    frame[
        "temporal_r60_hard_accept_online"
    ] = temporal_r60

    frame[
        "hybrid_balanced_or_r40_accept_online"
    ] = balanced | temporal_r40

    frame[
        "hybrid_balanced_or_r60_accept_online"
    ] = balanced | temporal_r60

    return frame


def policy_definitions() -> list[dict[str, Any]]:
    common = {
        "fix_x_column": "chosen_abs_x_traj01_m",
        "fix_y_column": "chosen_abs_y_traj01_m",
        "tile_column": "chosen_tile_id",
        "manifest_error_column":
            "chosen_error_m_eval_only",
        "hit_column": "hit_eval_only",
        "dangerous_column":
            "dangerous_false_eval_only",
    }

    return [
        {
            "policy": "relative_only",
            "policy_type":
                "controlled_relative_baseline_eval_replay",
            "accept_column": None,
            "fix_x_column": None,
            "fix_y_column": None,
            "tile_column": None,
            "manifest_error_column": None,
            "hit_column": None,
            "dangerous_column": None,
        },
        {
            "policy": "balanced_hard_reset_online",
            "policy_type":
                "balanced_confidence_hard_reset",
            "accept_column":
                "balanced_hard_accept_online",
            **common,
        },
        {
            "policy": "permissive_hard_reset_online",
            "policy_type":
                "permissive_confidence_hard_reset",
            "accept_column":
                "permissive_hard_accept_online",
            **common,
        },
        {
            "policy":
                "temporal_r40_hard_reset_online",
            "policy_type":
                "permissive_plus_temporal_r40_hard_reset",
            "accept_column":
                "temporal_r40_hard_accept_online",
            **common,
        },
        {
            "policy":
                "temporal_r60_hard_reset_online",
            "policy_type":
                "permissive_plus_temporal_r60_hard_reset",
            "accept_column":
                "temporal_r60_hard_accept_online",
            **common,
        },
        {
            "policy":
                "hybrid_balanced_or_r40_hard_reset_online",
            "policy_type":
                "balanced_or_temporal_r40_hard_reset",
            "accept_column":
                "hybrid_balanced_or_r40_accept_online",
            **common,
        },
        {
            "policy":
                "hybrid_balanced_or_r60_hard_reset_online",
            "policy_type":
                "balanced_or_temporal_r60_hard_reset",
            "accept_column":
                "hybrid_balanced_or_r60_accept_online",
            **common,
        },
    ]


def acceptance_summary(
    frame: pd.DataFrame,
    accept_column: str,
    policy: str,
) -> dict[str, Any]:
    accepted_mask = bool_series(frame[accept_column])

    hit = bool_series(frame["hit_eval_only"])
    dangerous = bool_series(
        frame["dangerous_false_eval_only"]
    )

    accepted = frame.loc[
        accepted_mask
    ].sort_values("sequence_frame_id")

    accepted_count = int(len(accepted))
    true_count = int((accepted_mask & hit).sum())
    false_count = accepted_count - true_count

    distances = accepted[
        "reference_cumulative_distance_m"
    ].to_numpy(dtype=float)

    full_start = float(
        frame["reference_cumulative_distance_m"].iloc[0]
    )
    full_end = float(
        frame["reference_cumulative_distance_m"].iloc[-1]
    )

    if len(distances):
        between = np.diff(distances)

        coverage = np.concatenate(
            [
                np.asarray([distances[0] - full_start]),
                between,
                np.asarray([full_end - distances[-1]]),
            ]
        )

        gap_median = (
            float(np.median(between))
            if len(between)
            else None
        )

        gap_p95 = (
            float(np.quantile(between, 0.95))
            if len(between)
            else None
        )

        gap_max = float(np.max(coverage))
    else:
        gap_median = None
        gap_p95 = None
        gap_max = None

    return {
        "policy": policy,
        "accept_column": accept_column,
        "accepted": accepted_count,
        "true_accepts_eval_only": true_count,
        "false_accepts_eval_only": false_count,
        "dangerous_false_accepts_gt100m_eval_only":
            int((accepted_mask & dangerous).sum()),
        "precision_eval_only": (
            true_count / accepted_count
            if accepted_count
            else None
        ),
        "median_accepted_error_m_eval_only": (
            float(
                frame.loc[
                    accepted_mask,
                    "chosen_error_m_eval_only",
                ].median()
            )
            if accepted_count
            else None
        ),
        "inter_event_gap_median_m": gap_median,
        "inter_event_gap_p95_m": gap_p95,
        "coverage_gap_including_boundaries_max_m":
            gap_max,
    }


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
            "reference_cumulative_distance_m",
        ],
        "S6B.1C.0 temporal diagnostics",
    )

    diagnostics = diagnostics.sort_values(
        "sequence_frame_id"
    ).reset_index(drop=True)

    if diagnostics.duplicated(
        "sequence_frame_id"
    ).any():
        raise ValueError(
            "Temporal diagnostics contain duplicate "
            "sequence_frame_id values"
        )

    frame = build_policy_masks(diagnostics)

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

    policy_specs = policy_definitions()

    all_replay: list[pd.DataFrame] = []
    all_events: list[pd.DataFrame] = []
    all_gaps: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []
    acceptance_rows: list[dict[str, Any]] = []

    for spec in policy_specs:
        accept_column = spec["accept_column"]

        if accept_column is None:
            corrections = frame.iloc[0:0].copy()
        else:
            accepted_mask = bool_series(
                frame[accept_column]
            )

            corrections = frame.loc[
                accepted_mask
            ].copy()

            acceptance_rows.append(
                acceptance_summary(
                    frame=frame,
                    accept_column=accept_column,
                    policy=spec["policy"],
                )
            )

        policy_replay, policy_events = (
            replay_module.replay_policy(
                trajectory=trajectory,
                corrections=corrections,
                policy_spec=spec,
            )
        )

        policy_gaps = (
            replay_module.build_gap_segments(
                policy_trajectory=policy_replay,
                events=policy_events,
            )
        )

        metrics = replay_module.calculate_metrics(
            policy_trajectory=policy_replay,
            events=policy_events,
            total_distance_m=total_distance_m,
        )

        all_replay.append(policy_replay)
        all_gaps.append(policy_gaps)
        metric_rows.append(metrics)

        if not policy_events.empty:
            all_events.append(policy_events)

    replay = pd.concat(
        all_replay,
        ignore_index=True,
    )

    events = pd.concat(
        all_events,
        ignore_index=True,
    )

    gaps = pd.concat(
        all_gaps,
        ignore_index=True,
    )

    metrics = pd.DataFrame(metric_rows)
    metrics = replay_module.add_baseline_deltas(
        metrics
    )

    acceptance = pd.DataFrame(acceptance_rows)

    expected_counts = {
        "balanced_hard_reset_online": 33,
        "permissive_hard_reset_online": 80,
        "temporal_r40_hard_reset_online": 46,
        "temporal_r60_hard_reset_online": 62,
    }

    actual_counts = (
        events.groupby("policy")
        .size()
        .to_dict()
    )

    for policy, expected in expected_counts.items():
        actual = int(actual_counts.get(policy, 0))

        if actual != expected:
            raise ValueError(
                f"Frozen event-count check failed for "
                f"{policy}: expected {expected}, "
                f"found {actual}"
            )

    metrics = metrics.merge(
        acceptance,
        on="policy",
        how="left",
        validate="one_to_one",
        suffixes=("", "_acceptance"),
    )

    masks_path = (
        args.metadata_dir
        / "s6b1c1_temporal_policy_masks.csv"
    )

    replay_path = (
        args.metadata_dir
        / "s6b1c1_temporal_hard_reset_trajectories.csv"
    )

    events_path = (
        args.metadata_dir
        / "s6b1c1_temporal_hard_reset_events.csv"
    )

    metrics_path = (
        args.metadata_dir
        / "s6b1c1_temporal_hard_reset_metrics.csv"
    )

    gaps_path = (
        args.metadata_dir
        / "s6b1c1_temporal_hard_reset_gap_segments.csv"
    )

    summary_path = (
        args.report_dir
        / "s6b1c1_temporal_hard_reset_summary.json"
    )

    mask_columns = [
        "sequence_frame_id",
        "token",
        "balanced_hard_accept_online",
        "permissive_hard_accept_online",
        "temporal_r40_hard_accept_online",
        "temporal_r60_hard_accept_online",
        "hybrid_balanced_or_r40_accept_online",
        "hybrid_balanced_or_r60_accept_online",
        "temporal_support_any_r40_count",
        "temporal_support_any_r60_count",
        "hit_eval_only",
        "dangerous_false_eval_only",
        "chosen_error_m_eval_only",
    ]

    frame[mask_columns].to_csv(
        masks_path,
        index=False,
    )

    replay.to_csv(replay_path, index=False)
    events.to_csv(events_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    gaps.to_csv(gaps_path, index=False)

    summary = {
        "stage": "S6B.1C.1",
        "title": (
            "Temporal-confirmation hard-reset replay"
        ),
        "relative_variant": args.relative_variant,
        "reference_total_distance_m_eval_only":
            total_distance_m,
        "important_warning": (
            "This is a controlled evaluation replay using "
            "prefix-aligned relative coordinates."
        ),
        "online_acceptance_rules": {
            "temporal_r40": (
                "permissive confidence and at least one of "
                "the previous three query observations has "
                "displacement residual <=40 m; frame zero "
                "is retained as initialization"
            ),
            "temporal_r60": (
                "permissive confidence and at least one of "
                "the previous three query observations has "
                "displacement residual <=60 m; frame zero "
                "is retained as initialization"
            ),
            "hybrid_r40": (
                "balanced confidence OR temporal_r40"
            ),
            "hybrid_r60": (
                "balanced confidence OR temporal_r60"
            ),
        },
        "policy_metrics": {
            row["policy"]: {
                key: json_safe(value)
                for key, value in row.items()
            }
            for row in metrics.to_dict(
                orient="records"
            )
        },
        "outputs": {
            "policy_masks": masks_path,
            "replay_trajectories": replay_path,
            "correction_events": events_path,
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

    replay_module.plot_trajectories(
        replay,
        args.figure_dir
        / "s6b1c1_temporal_hard_reset_trajectories.png",
    )

    replay_module.plot_error_by_distance(
        replay,
        args.figure_dir
        / "s6b1c1_temporal_hard_reset_error_by_distance.png",
    )

    print(
        "S6B.1C.1 Temporal Hard-Reset Replay"
    )
    print(
        "-----------------------------------"
    )
    print(
        f"Relative variant: {args.relative_variant}"
    )
    print(
        f"Frames per policy: {len(trajectory)}"
    )
    print(
        f"Reference distance: {total_distance_m:.3f} m"
    )
    print()

    display_columns = [
        "policy",
        "accepted",
        "true_accepts_eval_only",
        "false_accepts_eval_only",
        "dangerous_false_accepts_gt100m_eval_only",
        "precision_eval_only",
        "rmse_m",
        "p95_error_m",
        "max_error_m",
        "final_error_m",
        "failure_rate_gt40m",
        "coverage_gap_including_boundaries_max_m",
    ]

    available_columns = [
        column
        for column in display_columns
        if column in metrics.columns
    ]

    print(
        metrics[available_columns]
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
