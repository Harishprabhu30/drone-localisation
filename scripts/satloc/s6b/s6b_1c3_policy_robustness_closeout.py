#!/usr/bin/env python3
"""S6B.1C.3 — Adaptive-policy robustness and segment closeout.

This block does not create a new online correction rule.

It evaluates the completed S6B.1C.2 policies across fixed-distance
trajectory windows and compares:

    global RMSE
    p95 and maximum error
    threshold failure rate
    worst-window RMSE
    number of windows improved over hybrid-r60 hard reset
    event effects by correction source

Reference/error quantities are evaluation-only.

Command Used:

export PYTHONPATH=$PWD/src

python scripts/satloc/s6b/s6b_1c3_policy_robustness_closeout.py \
  2>&1 | tee \
  outputs/satloc/reports/s6b_relative_absolute/s6b1c3_policy_robustness_closeout.log
  
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


DEFAULT_METRICS = Path(
    "outputs/satloc/metadata/s6b_relative_absolute/"
    "s6b1c2_adaptive_soft_metrics.csv"
)

DEFAULT_TRAJECTORIES = Path(
    "outputs/satloc/metadata/s6b_relative_absolute/"
    "s6b1c2_adaptive_soft_trajectories.csv"
)

DEFAULT_EVENTS = Path(
    "outputs/satloc/metadata/s6b_relative_absolute/"
    "s6b1c2_adaptive_soft_events.csv"
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

HARD_REFERENCE_POLICY = "hybrid_r60_hard_reset_online"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--metrics",
        type=Path,
        default=DEFAULT_METRICS,
    )
    parser.add_argument(
        "--trajectories",
        type=Path,
        default=DEFAULT_TRAJECTORIES,
    )
    parser.add_argument(
        "--events",
        type=Path,
        default=DEFAULT_EVENTS,
    )
    parser.add_argument(
        "--window-m",
        type=float,
        default=500.0,
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


def build_window_metrics(
    trajectories: pd.DataFrame,
    window_m: float,
) -> pd.DataFrame:
    frame = trajectories.copy()

    start_distance = float(
        frame["reference_cumulative_distance_m"].min()
    )

    end_distance = float(
        frame["reference_cumulative_distance_m"].max()
    )

    frame["window_id"] = np.floor(
        (
            frame["reference_cumulative_distance_m"]
            - start_distance
        )
        / window_m
    ).astype(int)

    rows: list[dict[str, Any]] = []

    for (policy, window_id), group in frame.groupby(
        ["policy", "window_id"],
        sort=True,
    ):
        errors = pd.to_numeric(
            group["fused_error_m"],
            errors="raise",
        )

        window_start = (
            start_distance + int(window_id) * window_m
        )

        window_end = min(
            window_start + window_m,
            end_distance,
        )

        rows.append(
            {
                "policy": str(policy),
                "window_id": int(window_id),
                "window_start_m": window_start,
                "window_end_m": window_end,
                "window_center_m":
                    0.5 * (window_start + window_end),
                "frames": int(len(group)),
                "rmse_m": float(
                    math.sqrt(
                        float(
                            np.mean(
                                np.square(errors)
                            )
                        )
                    )
                ),
                "mean_error_m": float(errors.mean()),
                "median_error_m":
                    float(errors.median()),
                "p95_error_m":
                    float(errors.quantile(0.95)),
                "max_error_m": float(errors.max()),
                "failure_rate_gt40m": float(
                    (errors > 40.0).mean()
                ),
            }
        )

    return pd.DataFrame(rows)


def summarize_windows(
    window_metrics: pd.DataFrame,
) -> pd.DataFrame:
    hard = window_metrics.loc[
        window_metrics["policy"]
        == HARD_REFERENCE_POLICY,
        [
            "window_id",
            "rmse_m",
        ],
    ].rename(
        columns={
            "rmse_m": "hard_reference_window_rmse_m"
        }
    )

    if hard.empty:
        raise ValueError(
            f"Missing hard-reference policy: "
            f"{HARD_REFERENCE_POLICY}"
        )

    frame = window_metrics.merge(
        hard,
        on="window_id",
        how="left",
        validate="many_to_one",
    )

    frame["window_rmse_delta_vs_hard_m"] = (
        frame["rmse_m"]
        - frame["hard_reference_window_rmse_m"]
    )

    rows: list[dict[str, Any]] = []

    for policy, group in frame.groupby(
        "policy",
        sort=False,
    ):
        delta = group[
            "window_rmse_delta_vs_hard_m"
        ]

        rows.append(
            {
                "policy": policy,
                "windows": int(len(group)),
                "median_window_rmse_m":
                    float(group["rmse_m"].median()),
                "p90_window_rmse_m":
                    float(group["rmse_m"].quantile(0.90)),
                "worst_window_rmse_m":
                    float(group["rmse_m"].max()),
                "median_window_failure_rate_gt40m":
                    float(
                        group[
                            "failure_rate_gt40m"
                        ].median()
                    ),
                "windows_better_than_hard":
                    int((delta < 0.0).sum()),
                "windows_equal_to_hard":
                    int(np.isclose(delta, 0.0).sum()),
                "windows_worse_than_hard":
                    int((delta > 0.0).sum()),
                "median_window_rmse_delta_vs_hard_m":
                    float(delta.median()),
                "maximum_window_improvement_vs_hard_m":
                    float(-delta.min()),
                "maximum_window_worsening_vs_hard_m":
                    float(delta.max()),
            }
        )

    return pd.DataFrame(rows)


def build_event_source_summary(
    events: pd.DataFrame,
    policies: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    selected = events.loc[
        events["policy"].isin(policies)
    ].copy()

    for (policy, source), group in selected.groupby(
        [
            "policy",
            "correction_alpha_source",
        ],
        dropna=False,
        sort=False,
    ):
        true_mask = bool_series(
            group["correction_true_eval_only"]
        )

        false_mask = bool_series(
            group["correction_false_eval_only"]
        )

        improvement = pd.to_numeric(
            group["error_improvement_m"],
            errors="coerce",
        )

        rows.append(
            {
                "policy": policy,
                "correction_alpha_source": source,
                "events": int(len(group)),
                "true_events_eval_only":
                    int(true_mask.sum()),
                "false_events_eval_only":
                    int(false_mask.sum()),
                "precision_eval_only":
                    float(true_mask.mean()),
                "mean_alpha": float(
                    group["correction_alpha"].mean()
                ),
                "median_absolute_innovation_m": float(
                    group[
                        "absolute_innovation_m"
                    ].median()
                ),
                "median_position_shift_applied_m": float(
                    group[
                        "position_shift_applied_m"
                    ].median()
                ),
                "events_improving_error":
                    int((improvement > 0.0).sum()),
                "events_worsening_error":
                    int((improvement < 0.0).sum()),
                "median_error_improvement_m":
                    float(improvement.median()),
                "median_true_improvement_m_eval_only": (
                    float(
                        improvement.loc[
                            true_mask
                        ].median()
                    )
                    if true_mask.any()
                    else None
                ),
                "median_false_improvement_m_eval_only": (
                    float(
                        improvement.loc[
                            false_mask
                        ].median()
                    )
                    if false_mask.any()
                    else None
                ),
            }
        )

    return pd.DataFrame(rows)


def build_policy_selection(
    metrics: pd.DataFrame,
    window_summary: pd.DataFrame,
) -> tuple[pd.DataFrame, str, str]:
    candidates = metrics.loc[
        metrics["policy"]
        .astype(str)
        .str.startswith("hybrid_r60_")
    ].copy()

    candidates = candidates.loc[
        pd.to_numeric(
            candidates[
                "dangerous_false_events_gt100m_eval_only"
            ],
            errors="coerce",
        )
        == 0
    ].copy()

    candidates = candidates.merge(
        window_summary,
        on="policy",
        how="left",
        validate="one_to_one",
    )

    if candidates.empty:
        raise ValueError(
            "No zero-danger hybrid-r60 policies found"
        )

    performance_policy = str(
        candidates.sort_values(
            ["rmse_m", "p95_error_m"]
        ).iloc[0]["policy"]
    )

    best_rmse = float(
        candidates["rmse_m"].min()
    )

    # Keep policies within 3% of the best global RMSE before
    # evaluating tail and segment robustness.
    near_best = candidates.loc[
        candidates["rmse_m"]
        <= best_rmse * 1.03
    ].copy()

    rank_columns = [
        "rmse_m",
        "p95_error_m",
        "max_error_m",
        "failure_rate_gt40m",
        "worst_window_rmse_m",
    ]

    for column in rank_columns:
        near_best[f"{column}_rank"] = (
            near_best[column]
            .rank(
                method="min",
                ascending=True,
            )
        )

    near_best["robust_rank_sum"] = sum(
        near_best[f"{column}_rank"]
        for column in rank_columns
    )

    near_best = near_best.sort_values(
        [
            "robust_rank_sum",
            "rmse_m",
            "worst_window_rmse_m",
        ]
    ).reset_index(drop=True)

    robust_policy = str(
        near_best.iloc[0]["policy"]
    )

    candidates = candidates.merge(
        near_best[
            [
                "policy",
                "robust_rank_sum",
            ]
        ],
        on="policy",
        how="left",
        validate="one_to_one",
    )

    candidates["is_performance_policy"] = (
        candidates["policy"] == performance_policy
    )

    candidates["is_robust_policy"] = (
        candidates["policy"] == robust_policy
    )

    candidates = candidates.sort_values(
        [
            "rmse_m",
            "p95_error_m",
        ]
    ).reset_index(drop=True)

    return (
        candidates,
        performance_policy,
        robust_policy,
    )


def plot_window_rmse(
    window_metrics: pd.DataFrame,
    policies: list[str],
    output_path: Path,
) -> None:
    selected = window_metrics.loc[
        window_metrics["policy"].isin(policies)
    ]

    fig, ax = plt.subplots(figsize=(13, 6))

    for policy, group in selected.groupby(
        "policy",
        sort=False,
    ):
        group = group.sort_values(
            "window_center_m"
        )

        ax.plot(
            group["window_center_m"],
            group["rmse_m"],
            marker="o",
            linewidth=1.4,
            label=policy,
        )

    ax.set_xlabel(
        "Trajectory distance-window centre [m] "
        "— evaluation only"
    )
    ax.set_ylabel("Window RMSE [m]")
    ax.set_title(
        "S6B.1C.3 policy stability across 500 m windows"
    )
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> int:
    args = parse_args()

    for directory in [
        args.metadata_dir,
        args.report_dir,
        args.figure_dir,
    ]:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    metrics = pd.read_csv(
        args.metrics,
        low_memory=False,
    )

    trajectories = pd.read_csv(
        args.trajectories,
        low_memory=False,
    )

    events = pd.read_csv(
        args.events,
        low_memory=False,
    )

    require_columns(
        metrics,
        [
            "policy",
            "rmse_m",
            "p95_error_m",
            "max_error_m",
            "final_error_m",
            "failure_rate_gt40m",
            "dangerous_false_events_gt100m_eval_only",
        ],
        "S6B.1C.2 metrics",
    )

    require_columns(
        trajectories,
        [
            "policy",
            "reference_cumulative_distance_m",
            "fused_error_m",
        ],
        "S6B.1C.2 trajectories",
    )

    require_columns(
        events,
        [
            "policy",
            "correction_alpha_source",
            "correction_alpha",
            "absolute_innovation_m",
            "position_shift_applied_m",
            "error_improvement_m",
            "correction_true_eval_only",
            "correction_false_eval_only",
        ],
        "S6B.1C.2 events",
    )

    window_metrics = build_window_metrics(
        trajectories=trajectories,
        window_m=args.window_m,
    )

    window_summary = summarize_windows(
        window_metrics
    )

    (
        policy_selection,
        performance_policy,
        robust_policy,
    ) = build_policy_selection(
        metrics=metrics,
        window_summary=window_summary,
    )

    event_source_summary = build_event_source_summary(
        events=events,
        policies=list(
            dict.fromkeys(
                [
                    performance_policy,
                    robust_policy,
                    HARD_REFERENCE_POLICY,
                ]
            )
        ),
    )

    window_metrics_path = (
        args.metadata_dir
        / "s6b1c3_policy_window_metrics.csv"
    )

    window_summary_path = (
        args.metadata_dir
        / "s6b1c3_policy_window_summary.csv"
    )

    policy_selection_path = (
        args.metadata_dir
        / "s6b1c3_policy_selection_summary.csv"
    )

    event_summary_path = (
        args.metadata_dir
        / "s6b1c3_event_source_summary.csv"
    )

    summary_path = (
        args.report_dir
        / "s6b1c3_policy_robustness_summary.json"
    )

    figure_path = (
        args.figure_dir
        / "s6b1c3_policy_window_rmse.png"
    )

    window_metrics.to_csv(
        window_metrics_path,
        index=False,
    )

    window_summary.to_csv(
        window_summary_path,
        index=False,
    )

    policy_selection.to_csv(
        policy_selection_path,
        index=False,
    )

    event_source_summary.to_csv(
        event_summary_path,
        index=False,
    )

    selected_plot_policies = list(
        dict.fromkeys(
            [
                "relative_only",
                "balanced_hard_reset_online",
                "permissive_hard_reset_online",
                HARD_REFERENCE_POLICY,
                performance_policy,
                robust_policy,
            ]
        )
    )

    plot_window_rmse(
        window_metrics=window_metrics,
        policies=selected_plot_policies,
        output_path=figure_path,
    )

    summary = {
        "stage": "S6B.1C.3",
        "title": (
            "Adaptive policy robustness and segment closeout"
        ),
        "window_size_m": args.window_m,
        "performance_policy_by_global_rmse_eval_only":
            performance_policy,
        "robust_policy_by_exploratory_rank_sum_eval_only":
            robust_policy,
        "selection_warning": (
            "Both selections use traj01 evaluation metrics. "
            "They are experimental policy choices and are not "
            "independent validation results."
        ),
        "robust_ranking_metrics": [
            "global RMSE",
            "global p95",
            "global maximum error",
            "failure rate above 40 m",
            "worst 500 m window RMSE",
        ],
        "outputs": {
            "window_metrics": window_metrics_path,
            "window_summary": window_summary_path,
            "policy_selection": policy_selection_path,
            "event_source_summary": event_summary_path,
            "figure": figure_path,
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

    print("S6B.1C.3 Policy Robustness Closeout")
    print("-----------------------------------")
    print(f"Window size:         {args.window_m:.1f} m")
    print(f"Performance policy:  {performance_policy}")
    print(f"Robust policy:       {robust_policy}")
    print()

    display_columns = [
        "policy",
        "rmse_m",
        "p95_error_m",
        "max_error_m",
        "failure_rate_gt40m",
        "median_window_rmse_m",
        "worst_window_rmse_m",
        "windows_better_than_hard",
        "windows_worse_than_hard",
        "robust_rank_sum",
        "is_performance_policy",
        "is_robust_policy",
    ]

    print("Hybrid-r60 policy comparison")
    print("----------------------------")

    print(
        policy_selection[
            [
                column
                for column in display_columns
                if column in policy_selection.columns
            ]
        ]
        .head(20)
        .to_string(index=False)
    )

    print()
    print("Correction-source impact")
    print("------------------------")

    print(
        event_source_summary.to_string(
            index=False
        )
    )

    print()
    print(f"Window metrics:  {window_metrics_path}")
    print(f"Window summary:  {window_summary_path}")
    print(f"Policy summary:  {policy_selection_path}")
    print(f"Event summary:   {event_summary_path}")
    print(f"Summary:         {summary_path}")
    print(f"Figure:          {figure_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
