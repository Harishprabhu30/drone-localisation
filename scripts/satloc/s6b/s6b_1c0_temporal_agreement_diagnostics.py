#!/usr/bin/env python3
"""S6B.1C.0 — Online-safe temporal agreement diagnostics.

Purpose
-------
Measure whether consecutive absolute observations agree with the
relative displacement measured by S6A.

For two correction opportunities i and j:

    relative_delta = relative_xy[j] - relative_xy[i]
    absolute_delta = absolute_xy[j] - absolute_xy[i]

    temporal_residual =
        norm(absolute_delta - relative_delta)

A small temporal residual means the absolute observations and the
relative estimator agree about UAV movement.

Locked rule
-----------
Ground-truth/reference fields are used only after online acceptance
masks have been created, for evaluation and reporting.

Online inputs:
    - chosen absolute tile-centre coordinates
    - balanced/permissive confidence flags
    - relative trajectory displacement
    - sequence ordering

Evaluation-only inputs:
    - hit_eval_only
    - chosen_error_m_eval_only
    - dangerous_false_eval_only

Command Used:

export PYTHONPATH=$PWD/src

python scripts/satloc/s6b/s6b_1c0_temporal_agreement_diagnostics.py \
  2>&1 | tee \
  outputs/satloc/reports/s6b_relative_absolute/s6b1c0_temporal_agreement.log

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


DEFAULT_MANIFEST = Path(
    "outputs/satloc/metadata/s6b_relative_absolute/"
    "s6b0_absolute_correction_manifest.csv"
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

TEMPORAL_THRESHOLDS_M = [20.0, 40.0, 60.0, 80.0, 100.0]
TEMPORAL_LAGS = [1, 2, 3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
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


def assert_unique(
    frame: pd.DataFrame,
    columns: list[str],
    name: str,
) -> None:
    duplicates = frame.duplicated(columns, keep=False)

    if duplicates.any():
        examples = frame.loc[
            duplicates,
            columns,
        ].head(10)

        raise ValueError(
            f"{name} has duplicate keys {columns}:\n"
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
            "reference_cumulative_distance_m",
            baseline_x_column,
            baseline_y_column,
        ],
        "S6A relative trajectory",
    )

    trajectory = trajectory_all.loc[
        trajectory_all["variant"].astype(str) == variant
    ].copy()

    trajectory = trajectory.sort_values(
        "sequence_frame_id"
    ).reset_index(drop=True)

    if len(trajectory) != 1034:
        raise ValueError(
            f"Expected 1034 rows for {variant}, "
            f"found {len(trajectory)}"
        )

    assert_unique(
        trajectory,
        ["sequence_frame_id"],
        "Filtered S6A trajectory",
    )

    trajectory["relative_x_m"] = pd.to_numeric(
        trajectory[baseline_x_column],
        errors="raise",
    )

    trajectory["relative_y_m"] = pd.to_numeric(
        trajectory[baseline_y_column],
        errors="raise",
    )

    return trajectory[
        [
            "sequence_frame_id",
            "token0_id",
            "reference_cumulative_distance_m",
            "relative_x_m",
            "relative_y_m",
        ]
    ].copy()


def add_temporal_features(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    frame = frame.sort_values(
        "sequence_frame_id"
    ).reset_index(drop=True)

    permissive = bool_series(
        frame["permissive_accept_online"]
    )

    for lag in TEMPORAL_LAGS:
        prefix = f"lag{lag}"

        frame[f"{prefix}_sequence_frame_id"] = (
            frame["sequence_frame_id"].shift(lag)
        )

        frame[f"{prefix}_token"] = (
            frame["token"].shift(lag)
        )

        frame[f"{prefix}_distance_gap_m"] = (
            frame["reference_cumulative_distance_m"]
            - frame[
                "reference_cumulative_distance_m"
            ].shift(lag)
        )

        relative_dx = (
            frame["relative_x_m"]
            - frame["relative_x_m"].shift(lag)
        )

        relative_dy = (
            frame["relative_y_m"]
            - frame["relative_y_m"].shift(lag)
        )

        absolute_dx = (
            frame["chosen_abs_x_traj01_m"]
            - frame[
                "chosen_abs_x_traj01_m"
            ].shift(lag)
        )

        absolute_dy = (
            frame["chosen_abs_y_traj01_m"]
            - frame[
                "chosen_abs_y_traj01_m"
            ].shift(lag)
        )

        frame[f"{prefix}_relative_dx_m"] = relative_dx
        frame[f"{prefix}_relative_dy_m"] = relative_dy

        frame[f"{prefix}_absolute_dx_m"] = absolute_dx
        frame[f"{prefix}_absolute_dy_m"] = absolute_dy

        frame[f"{prefix}_relative_step_m"] = np.hypot(
            relative_dx,
            relative_dy,
        )

        frame[f"{prefix}_absolute_step_m"] = np.hypot(
            absolute_dx,
            absolute_dy,
        )

        frame[f"{prefix}_temporal_residual_m"] = np.hypot(
            absolute_dx - relative_dx,
            absolute_dy - relative_dy,
        )

        frame[f"{prefix}_previous_permissive"] = (
            permissive.shift(lag).fillna(False)
        )

    for threshold in TEMPORAL_THRESHOLDS_M:
        threshold_name = int(threshold)

        any_support_columns: list[pd.Series] = []
        permissive_support_columns: list[pd.Series] = []

        for lag in TEMPORAL_LAGS:
            residual = frame[
                f"lag{lag}_temporal_residual_m"
            ]

            consistent = residual <= threshold

            any_support_columns.append(
                consistent.fillna(False)
            )

            permissive_support_columns.append(
                (
                    consistent
                    & bool_series(
                        frame[
                            f"lag{lag}_previous_permissive"
                        ]
                    )
                ).fillna(False)
            )

        frame[
            f"temporal_support_any_r{threshold_name}_count"
        ] = sum(
            item.astype(int)
            for item in any_support_columns
        )

        frame[
            f"temporal_support_permissive_r{threshold_name}_count"
        ] = sum(
            item.astype(int)
            for item in permissive_support_columns
        )

    return frame


def spacing_statistics(
    frame: pd.DataFrame,
    accepted_mask: pd.Series,
) -> dict[str, Any]:
    accepted = frame.loc[
        accepted_mask
    ].sort_values("sequence_frame_id")

    if accepted.empty:
        return {
            "first_accepted_frame": None,
            "last_accepted_frame": None,
            "inter_event_gap_median": None,
            "inter_event_gap_p95": None,
            "inter_event_gap_max": None,
            "coverage_gap_including_boundaries_max_m": None,
        }

    distances = accepted[
        "reference_cumulative_distance_m"
    ].to_numpy(dtype=float)

    full_start = float(
        frame["reference_cumulative_distance_m"].iloc[0]
    )
    full_end = float(
        frame["reference_cumulative_distance_m"].iloc[-1]
    )

    between = np.diff(distances)

    coverage_gaps = np.concatenate(
        [
            np.asarray([distances[0] - full_start]),
            between,
            np.asarray([full_end - distances[-1]]),
        ]
    )

    return {
        "first_accepted_frame": int(
            accepted["sequence_frame_id"].iloc[0]
        ),
        "last_accepted_frame": int(
            accepted["sequence_frame_id"].iloc[-1]
        ),
        "inter_event_gap_median": (
            float(np.median(between))
            if len(between)
            else None
        ),
        "inter_event_gap_p95": (
            float(np.quantile(between, 0.95))
            if len(between)
            else None
        ),
        "inter_event_gap_max": (
            float(np.max(between))
            if len(between)
            else None
        ),
        "coverage_gap_including_boundaries_max_m": (
            float(np.max(coverage_gaps))
            if len(coverage_gaps)
            else None
        ),
    }


def evaluate_policy(
    frame: pd.DataFrame,
    accepted_mask: pd.Series,
    policy_name: str,
    base_gate: str,
    support_source: str,
    residual_threshold_m: float | None,
    minimum_support: int,
) -> dict[str, Any]:
    accepted_mask = accepted_mask.fillna(False)

    hit = bool_series(frame["hit_eval_only"])
    dangerous = bool_series(
        frame["dangerous_false_eval_only"]
    )

    accepted_count = int(accepted_mask.sum())
    true_count = int((accepted_mask & hit).sum())
    false_count = accepted_count - true_count

    row: dict[str, Any] = {
        "policy": policy_name,
        "base_gate": base_gate,
        "support_source": support_source,
        "residual_threshold_m": residual_threshold_m,
        "minimum_support": minimum_support,
        "accepted": accepted_count,
        "true_accepts_eval_only": true_count,
        "false_accepts_eval_only": false_count,
        "dangerous_false_accepts_gt100m_eval_only": int(
            (accepted_mask & dangerous).sum()
        ),
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
    }

    row.update(
        spacing_statistics(
            frame,
            accepted_mask,
        )
    )

    return row


def build_policy_sweep(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    initialization = (
        frame["sequence_frame_id"]
        == frame["sequence_frame_id"].min()
    )

    for base_gate in [
        "balanced_accept_online",
        "permissive_accept_online",
    ]:
        base_mask = bool_series(frame[base_gate])

        rows.append(
            evaluate_policy(
                frame=frame,
                accepted_mask=base_mask,
                policy_name=f"{base_gate}_base",
                base_gate=base_gate,
                support_source="none",
                residual_threshold_m=None,
                minimum_support=0,
            )
        )

        for threshold in TEMPORAL_THRESHOLDS_M:
            threshold_name = int(threshold)

            for support_source in [
                "any_previous_query",
                "previous_permissive_query",
            ]:
                if support_source == "any_previous_query":
                    support_column = (
                        f"temporal_support_any_"
                        f"r{threshold_name}_count"
                    )
                else:
                    support_column = (
                        f"temporal_support_permissive_"
                        f"r{threshold_name}_count"
                    )

                for minimum_support in [1, 2]:
                    support_mask = (
                        frame[support_column]
                        >= minimum_support
                    )

                    accepted_mask = (
                        base_mask
                        & (
                            initialization
                            | support_mask
                        )
                    )

                    policy_name = (
                        f"{base_gate}_"
                        f"{support_source}_"
                        f"r{threshold_name}_"
                        f"support{minimum_support}"
                    )

                    rows.append(
                        evaluate_policy(
                            frame=frame,
                            accepted_mask=accepted_mask,
                            policy_name=policy_name,
                            base_gate=base_gate,
                            support_source=support_source,
                            residual_threshold_m=threshold,
                            minimum_support=minimum_support,
                        )
                    )

    return pd.DataFrame(rows)


def residual_summary(
    frame: pd.DataFrame,
    gate_column: str,
) -> dict[str, Any]:
    gate = bool_series(frame[gate_column])
    hit = bool_series(frame["hit_eval_only"])

    residual = pd.to_numeric(
        frame["lag1_temporal_residual_m"],
        errors="coerce",
    )

    result: dict[str, Any] = {}

    for label, mask in [
        ("all", gate),
        ("true_eval_only", gate & hit),
        ("false_eval_only", gate & ~hit),
    ]:
        values = residual.loc[mask].dropna()

        result[label] = {
            "rows": int(len(values)),
            "median_m": (
                float(values.median())
                if len(values)
                else None
            ),
            "p75_m": (
                float(values.quantile(0.75))
                if len(values)
                else None
            ),
            "p95_m": (
                float(values.quantile(0.95))
                if len(values)
                else None
            ),
            "max_m": (
                float(values.max())
                if len(values)
                else None
            ),
        }

    return result


def plot_residual_histogram(
    frame: pd.DataFrame,
    output_path: Path,
) -> None:
    gate = bool_series(
        frame["permissive_accept_online"]
    )
    hit = bool_series(frame["hit_eval_only"])

    true_values = frame.loc[
        gate & hit,
        "lag1_temporal_residual_m",
    ].dropna()

    false_values = frame.loc[
        gate & ~hit,
        "lag1_temporal_residual_m",
    ].dropna()

    upper = float(
        frame["lag1_temporal_residual_m"]
        .dropna()
        .quantile(0.98)
    )

    bins = np.linspace(
        0.0,
        max(upper, 1.0),
        30,
    )

    fig, ax = plt.subplots(figsize=(11, 6))

    ax.hist(
        true_values,
        bins=bins,
        alpha=0.6,
        label="Permissive true fixes — evaluation only",
    )

    ax.hist(
        false_values,
        bins=bins,
        alpha=0.6,
        label="Permissive false fixes — evaluation only",
    )

    ax.set_xlabel(
        "Lag-1 temporal displacement residual [m]"
    )
    ax.set_ylabel("Correction opportunities")
    ax.set_title(
        "S6B.1C.0 temporal agreement of absolute observations"
    )
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_residual_by_distance(
    frame: pd.DataFrame,
    output_path: Path,
) -> None:
    gate = bool_series(
        frame["permissive_accept_online"]
    )
    hit = bool_series(frame["hit_eval_only"])

    fig, ax = plt.subplots(figsize=(13, 6))

    true_rows = frame.loc[
        gate & hit
    ]

    false_rows = frame.loc[
        gate & ~hit
    ]

    ax.scatter(
        true_rows["reference_cumulative_distance_m"],
        true_rows["lag1_temporal_residual_m"],
        s=38,
        label="True fix — evaluation only",
    )

    ax.scatter(
        false_rows["reference_cumulative_distance_m"],
        false_rows["lag1_temporal_residual_m"],
        s=38,
        label="False fix — evaluation only",
    )

    for threshold in TEMPORAL_THRESHOLDS_M:
        ax.axhline(
            threshold,
            linestyle="--",
            linewidth=0.8,
            alpha=0.5,
        )

    ax.set_xlabel(
        "Reference cumulative distance [m] — evaluation only"
    )
    ax.set_ylabel(
        "Lag-1 temporal displacement residual [m]"
    )
    ax.set_title(
        "S6B.1C.0 temporal residual along traj01"
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

    manifest = pd.read_csv(
        args.manifest,
        low_memory=False,
    )

    trajectory_all = pd.read_csv(
        args.trajectory,
        low_memory=False,
    )

    require_columns(
        manifest,
        [
            "sequence_frame_id",
            "reference_cumulative_distance_m",
            "token",
            "chosen_abs_x_traj01_m",
            "chosen_abs_y_traj01_m",
            "chosen_error_m_eval_only",
            "hit_eval_only",
            "dangerous_false_eval_only",
            "balanced_accept_online",
            "permissive_accept_online",
        ],
        "S6B.0 correction manifest",
    )

    assert_unique(
        manifest,
        ["sequence_frame_id"],
        "S6B.0 correction manifest",
    )

    trajectory = prepare_trajectory(
        trajectory_all=trajectory_all,
        variant=args.relative_variant,
        baseline_x_column=args.baseline_x_column,
        baseline_y_column=args.baseline_y_column,
    )

    # The manifest already contains token0_id and reference distance.
    # Join only the relative-position columns to avoid pandas suffixes
    # such as reference_cumulative_distance_m_x / _y.
    trajectory_for_join = trajectory[
        [
            "sequence_frame_id",
            "relative_x_m",
            "relative_y_m",
        ]
    ].copy()

    frame = manifest.merge(
        trajectory_for_join,
        on="sequence_frame_id",
        how="left",
        validate="one_to_one",
    )

    if frame[
        [
            "relative_x_m",
            "relative_y_m",
            "reference_cumulative_distance_m",
        ]
    ].isna().any().any():
        raise ValueError(
            "Some correction opportunities did not join to "
            "the S6A relative trajectory"
        )

    frame = add_temporal_features(frame)

    sweep = build_policy_sweep(frame)

    diagnostics_path = (
        args.metadata_dir
        / "s6b1c0_temporal_agreement_diagnostics.csv"
    )

    sweep_path = (
        args.metadata_dir
        / "s6b1c0_temporal_confirmation_policy_sweep.csv"
    )

    summary_path = (
        args.report_dir
        / "s6b1c0_temporal_agreement_summary.json"
    )

    frame.to_csv(
        diagnostics_path,
        index=False,
    )

    sweep.to_csv(
        sweep_path,
        index=False,
    )

    summary = {
        "stage": "S6B.1C.0",
        "relative_variant": args.relative_variant,
        "definition": (
            "Temporal residual is the Euclidean difference "
            "between absolute displacement and relative "
            "displacement across correction opportunities."
        ),
        "online_only_policy_inputs": [
            "chosen_abs_x_traj01_m",
            "chosen_abs_y_traj01_m",
            "relative_x_m",
            "relative_y_m",
            "balanced_accept_online",
            "permissive_accept_online",
            "sequence ordering",
        ],
        "evaluation_only_outputs": [
            "hit_eval_only",
            "chosen_error_m_eval_only",
            "dangerous_false_eval_only",
            "precision statistics",
        ],
        "lag1_residual_summary": {
            "balanced": residual_summary(
                frame,
                "balanced_accept_online",
            ),
            "permissive": residual_summary(
                frame,
                "permissive_accept_online",
            ),
        },
        "outputs": {
            "diagnostics": diagnostics_path,
            "policy_sweep": sweep_path,
            "summary": summary_path,
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

    plot_residual_histogram(
        frame,
        args.figure_dir
        / "s6b1c0_temporal_residual_histogram.png",
    )

    plot_residual_by_distance(
        frame,
        args.figure_dir
        / "s6b1c0_temporal_residual_by_distance.png",
    )

    print("S6B.1C.0 Temporal Agreement Diagnostics")
    print("----------------------------------------")
    print(f"Correction opportunities: {len(frame)}")
    print(
        "Temporal lags:           "
        f"{TEMPORAL_LAGS}"
    )
    print(
        "Residual thresholds:     "
        f"{TEMPORAL_THRESHOLDS_M}"
    )

    for gate in [
        "balanced_accept_online",
        "permissive_accept_online",
    ]:
        summary_gate = residual_summary(
            frame,
            gate,
        )

        print()
        print(gate)
        print("-" * len(gate))

        for label, values in summary_gate.items():
            print(
                f"{label:16s} "
                f"rows={values['rows']:3d}  "
                f"median={values['median_m']}  "
                f"p95={values['p95_m']}"
            )

    candidates = sweep.loc[
        (
            sweep[
                "dangerous_false_accepts_gt100m_eval_only"
            ]
            == 0
        )
        & (
            sweep["accepted"] > 0
        )
    ].copy()

    candidates = candidates.sort_values(
        [
            "accepted",
            "precision_eval_only",
        ],
        ascending=[False, False],
    )

    print()
    print("Zero-danger temporal candidates")
    print("--------------------------------")

    print(
        candidates[
            [
                "policy",
                "accepted",
                "true_accepts_eval_only",
                "false_accepts_eval_only",
                "precision_eval_only",
                "inter_event_gap_p95",
                "coverage_gap_including_boundaries_max_m",
            ]
        ]
        .head(20)
        .to_string(index=False)
    )

    print()
    print(f"Diagnostics: {diagnostics_path}")
    print(f"Policy sweep: {sweep_path}")
    print(f"Summary:     {summary_path}")
    print(f"Figures:     {args.figure_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
