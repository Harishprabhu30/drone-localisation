#!/usr/bin/env python3
"""S6B.2D — Raw causal-fusion closeout and controlled-vs-causal comparison.

This block creates no new online policy. It evaluates the completed S6B.2C
post-lock replay and records the main scientific result:

1. Causal raw fusion strongly improves over raw bootstrapped relative-only.
2. Fixed adaptive soft blending does not beat hybrid-r60 hard reset after the
   causal bootstrap, even though it helped in the controlled aligned replay.

Reference positions, errors, hit labels, and policy selection metrics are
evaluation-only.

Command Used:

export PYTHONPATH=$PWD/src

python scripts/satloc/s6b/s6b_2d_raw_causal_fusion_closeout.py \
  2>&1 | tee \
  outputs/satloc/reports/s6b_relative_absolute/s6b2d_raw_causal_fusion_closeout.log
  
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


DEFAULT_RAW_METRICS = Path(
    "outputs/satloc/metadata/s6b_relative_absolute/"
    "s6b2c_raw_causal_fusion_metrics.csv"
)

DEFAULT_RAW_TRAJECTORIES = Path(
    "outputs/satloc/metadata/s6b_relative_absolute/"
    "s6b2c_raw_causal_fusion_trajectories.csv"
)

DEFAULT_RAW_EVENTS = Path(
    "outputs/satloc/metadata/s6b_relative_absolute/"
    "s6b2c_raw_causal_fusion_events.csv"
)

DEFAULT_CONTROLLED_METRICS = Path(
    "outputs/satloc/metadata/s6b_relative_absolute/"
    "s6b1c2_adaptive_soft_metrics.csv"
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

RAW_RELATIVE_POLICY = "raw_bootstrap_relative_only"
RAW_BALANCED_POLICY = "raw_bootstrap_balanced_hard_online"
RAW_HARD_POLICY = "raw_bootstrap_hybrid_r60_hard_online"
RAW_T050_POLICY = "raw_bootstrap_hybrid_r60_b100_t050_online"
RAW_T075_POLICY = "raw_bootstrap_hybrid_r60_b100_t075_online"

CONTROLLED_POLICY_MAP = {
    RAW_HARD_POLICY: "hybrid_r60_hard_reset_online",
    RAW_T050_POLICY: "hybrid_r60_adaptive_b100_t050_online",
    RAW_T075_POLICY: "hybrid_r60_adaptive_b100_t075_online",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--raw-metrics",
        type=Path,
        default=DEFAULT_RAW_METRICS,
    )
    parser.add_argument(
        "--raw-trajectories",
        type=Path,
        default=DEFAULT_RAW_TRAJECTORIES,
    )
    parser.add_argument(
        "--raw-events",
        type=Path,
        default=DEFAULT_RAW_EVENTS,
    )
    parser.add_argument(
        "--controlled-metrics",
        type=Path,
        default=DEFAULT_CONTROLLED_METRICS,
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
                "median_error_m":
                    float(errors.median()),
                "p95_error_m":
                    float(errors.quantile(0.95)),
                "max_error_m":
                    float(errors.max()),
                "failure_rate_gt40m":
                    float((errors > 40.0).mean()),
            }
        )

    return pd.DataFrame(rows)


def summarize_windows(
    window_metrics: pd.DataFrame,
) -> pd.DataFrame:
    hard = window_metrics.loc[
        window_metrics["policy"] == RAW_HARD_POLICY,
        ["window_id", "rmse_m"],
    ].rename(
        columns={
            "rmse_m": "hard_window_rmse_m"
        }
    )

    if hard.empty:
        raise ValueError(
            f"Missing raw hard-reference policy: "
            f"{RAW_HARD_POLICY}"
        )

    frame = window_metrics.merge(
        hard,
        on="window_id",
        how="left",
        validate="many_to_one",
    )

    frame["window_rmse_delta_vs_raw_hard_m"] = (
        frame["rmse_m"]
        - frame["hard_window_rmse_m"]
    )

    rows: list[dict[str, Any]] = []

    for policy, group in frame.groupby(
        "policy",
        sort=False,
    ):
        delta = group[
            "window_rmse_delta_vs_raw_hard_m"
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
                "windows_better_than_raw_hard":
                    int((delta < 0.0).sum()),
                "windows_equal_to_raw_hard":
                    int(np.isclose(delta, 0.0).sum()),
                "windows_worse_than_raw_hard":
                    int((delta > 0.0).sum()),
                "median_window_rmse_delta_vs_raw_hard_m":
                    float(delta.median()),
                "maximum_window_improvement_vs_raw_hard_m":
                    float(-delta.min()),
                "maximum_window_worsening_vs_raw_hard_m":
                    float(delta.max()),
            }
        )

    return pd.DataFrame(rows)


def build_event_summary(
    events: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for (policy, source), group in events.groupby(
        ["policy", "correction_alpha_source"],
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
                "mean_alpha":
                    float(
                        group[
                            "correction_alpha"
                        ].mean()
                    ),
                "median_absolute_innovation_m":
                    float(
                        group[
                            "absolute_innovation_m"
                        ].median()
                    ),
                "median_position_shift_applied_m":
                    float(
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


def build_controlled_vs_raw(
    raw_metrics: pd.DataFrame,
    controlled_metrics: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    metric_columns = [
        "rmse_m",
        "p95_error_m",
        "max_error_m",
        "final_error_m",
        "failure_rate_gt40m",
    ]

    for raw_policy, controlled_policy in (
        CONTROLLED_POLICY_MAP.items()
    ):
        raw_rows = raw_metrics.loc[
            raw_metrics["policy"] == raw_policy
        ]

        controlled_rows = controlled_metrics.loc[
            controlled_metrics["policy"]
            == controlled_policy
        ]

        if len(raw_rows) != 1:
            raise ValueError(
                f"Expected one raw row for {raw_policy}, "
                f"found {len(raw_rows)}"
            )

        if len(controlled_rows) != 1:
            raise ValueError(
                f"Expected one controlled row for "
                f"{controlled_policy}, found "
                f"{len(controlled_rows)}"
            )

        raw_row = raw_rows.iloc[0]
        controlled_row = controlled_rows.iloc[0]

        result: dict[str, Any] = {
            "raw_policy": raw_policy,
            "controlled_policy": controlled_policy,
        }

        for metric in metric_columns:
            controlled_value = float(
                controlled_row[metric]
            )
            raw_value = float(raw_row[metric])

            result[
                f"controlled_{metric}"
            ] = controlled_value

            result[
                f"raw_post_lock_{metric}"
            ] = raw_value

            result[
                f"raw_minus_controlled_{metric}"
            ] = raw_value - controlled_value

        rows.append(result)

    return pd.DataFrame(rows)


def build_policy_selection(
    raw_metrics: pd.DataFrame,
    window_summary: pd.DataFrame,
) -> tuple[pd.DataFrame, str, str]:
    candidates = raw_metrics.loc[
        raw_metrics["policy"] != RAW_RELATIVE_POLICY
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
            "No zero-danger raw causal policies found"
        )

    performance_policy = str(
        candidates.sort_values(
            [
                "rmse_m",
                "p95_error_m",
                "failure_rate_gt40m",
            ]
        ).iloc[0]["policy"]
    )

    rank_columns = [
        "rmse_m",
        "p95_error_m",
        "max_error_m",
        "failure_rate_gt40m",
        "worst_window_rmse_m",
    ]

    for column in rank_columns:
        candidates[f"{column}_rank"] = (
            candidates[column]
            .rank(
                method="min",
                ascending=True,
            )
        )

    candidates["robust_rank_sum"] = sum(
        candidates[f"{column}_rank"]
        for column in rank_columns
    )

    robust_policy = str(
        candidates.sort_values(
            [
                "robust_rank_sum",
                "rmse_m",
                "worst_window_rmse_m",
            ]
        ).iloc[0]["policy"]
    )

    candidates["is_performance_policy"] = (
        candidates["policy"] == performance_policy
    )

    candidates["is_robust_policy"] = (
        candidates["policy"] == robust_policy
    )

    candidates = candidates.sort_values(
        ["rmse_m", "p95_error_m"]
    ).reset_index(drop=True)

    return (
        candidates,
        performance_policy,
        robust_policy,
    )


def plot_window_rmse(
    window_metrics: pd.DataFrame,
    output_path: Path,
) -> None:
    selected_policies = [
        RAW_RELATIVE_POLICY,
        RAW_BALANCED_POLICY,
        RAW_HARD_POLICY,
        RAW_T050_POLICY,
        RAW_T075_POLICY,
    ]

    selected = window_metrics.loc[
        window_metrics["policy"].isin(
            selected_policies
        )
    ]

    fig, ax = plt.subplots(figsize=(14, 7))

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
            linewidth=1.3,
            label=policy,
        )

    ax.set_xlabel(
        "Post-lock distance-window centre [m] "
        "— evaluation only"
    )
    ax.set_ylabel("Window RMSE [m]")
    ax.set_title(
        "S6B.2D raw causal-fusion stability across 500 m windows"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

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

    raw_metrics = pd.read_csv(
        args.raw_metrics,
        low_memory=False,
    )

    raw_trajectories = pd.read_csv(
        args.raw_trajectories,
        low_memory=False,
    )

    raw_events = pd.read_csv(
        args.raw_events,
        low_memory=False,
    )

    controlled_metrics = pd.read_csv(
        args.controlled_metrics,
        low_memory=False,
    )

    require_columns(
        raw_metrics,
        [
            "policy",
            "rmse_m",
            "p95_error_m",
            "max_error_m",
            "final_error_m",
            "failure_rate_gt40m",
            "dangerous_false_events_gt100m_eval_only",
        ],
        "S6B.2C raw metrics",
    )

    require_columns(
        raw_trajectories,
        [
            "policy",
            "reference_cumulative_distance_m",
            "fused_error_m",
        ],
        "S6B.2C raw trajectories",
    )

    require_columns(
        raw_events,
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
        "S6B.2C raw events",
    )

    require_columns(
        controlled_metrics,
        [
            "policy",
            "rmse_m",
            "p95_error_m",
            "max_error_m",
            "final_error_m",
            "failure_rate_gt40m",
        ],
        "S6B.1C.2 controlled metrics",
    )

    window_metrics = build_window_metrics(
        trajectories=raw_trajectories,
        window_m=args.window_m,
    )

    window_summary = summarize_windows(
        window_metrics
    )

    event_summary = build_event_summary(
        raw_events
    )

    controlled_vs_raw = build_controlled_vs_raw(
        raw_metrics=raw_metrics,
        controlled_metrics=controlled_metrics,
    )

    (
        policy_selection,
        performance_policy,
        robust_policy,
    ) = build_policy_selection(
        raw_metrics=raw_metrics,
        window_summary=window_summary,
    )

    raw_relative_rmse = float(
        raw_metrics.loc[
            raw_metrics["policy"]
            == RAW_RELATIVE_POLICY,
            "rmse_m",
        ].iloc[0]
    )

    raw_hard_rmse = float(
        raw_metrics.loc[
            raw_metrics["policy"]
            == RAW_HARD_POLICY,
            "rmse_m",
        ].iloc[0]
    )

    raw_t050_rmse = float(
        raw_metrics.loc[
            raw_metrics["policy"]
            == RAW_T050_POLICY,
            "rmse_m",
        ].iloc[0]
    )

    raw_t075_rmse = float(
        raw_metrics.loc[
            raw_metrics["policy"]
            == RAW_T075_POLICY,
            "rmse_m",
        ].iloc[0]
    )

    conclusions = {
        "fusion_survives_causal_bootstrap": (
            raw_hard_rmse < raw_relative_rmse
        ),
        "adaptive_t050_improves_over_raw_relative": (
            raw_t050_rmse < raw_relative_rmse
        ),
        "adaptive_t075_improves_over_raw_relative": (
            raw_t075_rmse < raw_relative_rmse
        ),
        "adaptive_t050_beats_raw_hard": (
            raw_t050_rmse < raw_hard_rmse
        ),
        "adaptive_t075_beats_raw_hard": (
            raw_t075_rmse < raw_hard_rmse
        ),
        "scientific_interpretation": (
            "Absolute-temporal fusion remains highly beneficial after causal "
            "raw bootstrap, but fixed source-based soft alpha does not beat "
            "full hybrid-r60 reset on post-lock RMSE. The raw mapped relative "
            "trajectory carries larger residual map-frame drift, so partial "
            "corrections can leave too much error unremoved."
        ),
    }

    window_metrics_path = (
        args.metadata_dir
        / "s6b2d_raw_policy_window_metrics.csv"
    )

    window_summary_path = (
        args.metadata_dir
        / "s6b2d_raw_policy_window_summary.csv"
    )

    event_summary_path = (
        args.metadata_dir
        / "s6b2d_raw_event_source_summary.csv"
    )

    comparison_path = (
        args.metadata_dir
        / "s6b2d_controlled_vs_raw_comparison.csv"
    )

    selection_path = (
        args.metadata_dir
        / "s6b2d_raw_policy_selection.csv"
    )

    summary_path = (
        args.report_dir
        / "s6b2d_raw_causal_fusion_closeout_summary.json"
    )

    figure_path = (
        args.figure_dir
        / "s6b2d_raw_policy_window_rmse.png"
    )

    window_metrics.to_csv(
        window_metrics_path,
        index=False,
    )

    window_summary.to_csv(
        window_summary_path,
        index=False,
    )

    event_summary.to_csv(
        event_summary_path,
        index=False,
    )

    controlled_vs_raw.to_csv(
        comparison_path,
        index=False,
    )

    policy_selection.to_csv(
        selection_path,
        index=False,
    )

    plot_window_rmse(
        window_metrics=window_metrics,
        output_path=figure_path,
    )

    summary = {
        "stage": "S6B.2D",
        "title": (
            "Raw causal-fusion closeout and controlled-vs-causal comparison"
        ),
        "window_size_m": args.window_m,
        "performance_policy_by_post_lock_rmse_eval_only":
            performance_policy,
        "robust_policy_by_exploratory_rank_sum_eval_only":
            robust_policy,
        "conclusions":
            conclusions,
        "important_scope": (
            "All raw metrics cover only frames at or after the causal bootstrap "
            "lock. Before lock, only local relative localization is available."
        ),
        "selection_warning": (
            "Policy ranking uses traj01 evaluation metrics and is exploratory, "
            "not independent validation."
        ),
        "outputs": {
            "window_metrics":
                window_metrics_path,
            "window_summary":
                window_summary_path,
            "event_source_summary":
                event_summary_path,
            "controlled_vs_raw_comparison":
                comparison_path,
            "policy_selection":
                selection_path,
            "figure":
                figure_path,
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

    print("S6B.2D Raw Causal-Fusion Closeout")
    print("---------------------------------")
    print(f"Window size:          {args.window_m:.1f} m")
    print(f"Performance policy:   {performance_policy}")
    print(f"Robust policy:        {robust_policy}")
    print()

    print("Scientific verdict")
    print("------------------")
    print(
        "Fusion survives causal bootstrap: "
        f"{conclusions['fusion_survives_causal_bootstrap']}"
    )
    print(
        "t050 beats raw relative-only:      "
        f"{conclusions['adaptive_t050_improves_over_raw_relative']}"
    )
    print(
        "t075 beats raw relative-only:      "
        f"{conclusions['adaptive_t075_improves_over_raw_relative']}"
    )
    print(
        "t050 beats raw hybrid hard:        "
        f"{conclusions['adaptive_t050_beats_raw_hard']}"
    )
    print(
        "t075 beats raw hybrid hard:        "
        f"{conclusions['adaptive_t075_beats_raw_hard']}"
    )

    print()
    print("Raw policy comparison")
    print("---------------------")

    display_columns = [
        "policy",
        "rmse_m",
        "p95_error_m",
        "max_error_m",
        "failure_rate_gt40m",
        "median_window_rmse_m",
        "worst_window_rmse_m",
        "windows_better_than_raw_hard",
        "windows_worse_than_raw_hard",
        "robust_rank_sum",
        "is_performance_policy",
        "is_robust_policy",
    ]

    print(
        policy_selection[
            [
                column
                for column in display_columns
                if column in policy_selection.columns
            ]
        ].to_string(index=False)
    )

    print()
    print("Correction-source impact")
    print("------------------------")

    print(
        event_summary.to_string(index=False)
    )

    print()
    print(f"Window metrics:       {window_metrics_path}")
    print(f"Window summary:       {window_summary_path}")
    print(f"Event summary:        {event_summary_path}")
    print(f"Controlled/raw:       {comparison_path}")
    print(f"Policy selection:     {selection_path}")
    print(f"Summary:              {summary_path}")
    print(f"Figure:               {figure_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
