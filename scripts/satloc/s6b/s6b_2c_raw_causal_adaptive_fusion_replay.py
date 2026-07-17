#!/usr/bin/env python3
"""S6B.2C — Raw causal bootstrap + temporal adaptive fusion replay.

Purpose
-------
Evaluate whether the S6B.1C adaptive-fusion improvement remains after the
relative trajectory is mapped into the map frame using the causal bootstrap
transform selected in S6B.2B, rather than the evaluation-derived prefix
alignment used in S6B.1.

Online/causal-style inputs
--------------------------
- Raw ORB visual trajectory.
- S6B.2B selected similarity transform.
- Chosen absolute tile-centre coordinates.
- Balanced/permissive confidence flags.
- Raw-trajectory temporal displacement agreement.
- Sequence order.

Evaluation-only inputs
----------------------
- Reference trajectory coordinates and cumulative distance.
- hit_eval_only, chosen_error_m_eval_only, dangerous_false_eval_only.
- All reported error metrics.

Fairness rules
--------------
- Metrics start at the bootstrap lock frame.
- The lock-frame observation initializes the transform and is not applied again
  as a correction.
- S6B.1C temporal masks are not reused. Temporal support is recomputed from the
  raw bootstrapped trajectory.

Command Used:

export PYTHONPATH=$PWD/src

python scripts/satloc/s6b/s6b_2c_raw_causal_adaptive_fusion_replay.py \
  2>&1 | tee \
  outputs/satloc/reports/s6b_relative_absolute/s6b2c_raw_causal_adaptive_fusion_replay.log

"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_RAW_TRAJECTORY = Path(
    "outputs/satloc/metadata/s6a_relative_motion/"
    "s6a2_orb_relative_trajectory_pixels.csv"
)
DEFAULT_ALIGNED_TRAJECTORY = Path(
    "outputs/satloc/metadata/s6a_relative_motion/"
    "s6a2_orb_relative_trajectory_aligned_eval_only.csv"
)
DEFAULT_DIAGNOSTICS = Path(
    "outputs/satloc/metadata/s6b_relative_absolute/"
    "s6b1c0_temporal_agreement_diagnostics.csv"
)
DEFAULT_BOOTSTRAP_TRANSFORM = Path(
    "outputs/satloc/metadata/s6b_relative_absolute/"
    "s6b2b_selected_bootstrap_transform.json"
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
TEMPORAL_LAGS = [1, 2, 3]
TEMPORAL_RADIUS_M = 60.0
FIX_ERROR_CONSISTENCY_TOLERANCE_M = 10.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-trajectory", type=Path, default=DEFAULT_RAW_TRAJECTORY)
    parser.add_argument(
        "--aligned-trajectory",
        type=Path,
        default=DEFAULT_ALIGNED_TRAJECTORY,
    )
    parser.add_argument("--diagnostics", type=Path, default=DEFAULT_DIAGNOSTICS)
    parser.add_argument(
        "--bootstrap-transform",
        type=Path,
        default=DEFAULT_BOOTSTRAP_TRANSFORM,
    )
    parser.add_argument("--relative-variant", default=DEFAULT_VARIANT)
    parser.add_argument("--baseline-x-column", default=DEFAULT_BASELINE_X)
    parser.add_argument("--baseline-y-column", default=DEFAULT_BASELINE_Y)
    parser.add_argument("--metadata-dir", type=Path, default=DEFAULT_METADATA_DIR)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    return parser.parse_args()


def load_s6b1a_module():
    module_path = Path(__file__).with_name(
        "s6b_1a_position_only_correction_replay.py"
    )
    if not module_path.exists():
        raise FileNotFoundError(
            "Required helper script was not found beside S6B.2C: "
            f"{module_path}"
        )
    spec = importlib.util.spec_from_file_location("s6b1a_replay", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load S6B.1A helpers from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def require_columns(frame: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def assert_unique(frame: pd.DataFrame, columns: list[str], name: str) -> None:
    duplicates = frame.duplicated(columns, keep=False)
    if duplicates.any():
        example = frame.loc[duplicates, columns].head(10)
        raise ValueError(f"{name} has duplicate keys {columns}:\n{example}")


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
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
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


def rotation_matrix(angle_rad: float) -> np.ndarray:
    cosine = math.cos(angle_rad)
    sine = math.sin(angle_rad)
    return np.asarray(
        [[cosine, -sine], [sine, cosine]],
        dtype=float,
    )


def prepare_raw_map_trajectory(
    raw_all: pd.DataFrame,
    reference_trajectory: pd.DataFrame,
    transform: dict[str, Any],
    variant: str,
) -> pd.DataFrame:
    require_columns(
        raw_all,
        ["variant", "sequence_frame_id", "visual_x_px", "visual_y_px"],
        "Raw S6A trajectory",
    )
    raw = raw_all.loc[raw_all["variant"].astype(str) == variant].copy()
    raw = raw.sort_values("sequence_frame_id").reset_index(drop=True)

    if len(raw) != 1034:
        raise ValueError(f"Expected 1034 raw rows for {variant}, found {len(raw)}")
    assert_unique(raw, ["sequence_frame_id"], "Filtered raw trajectory")

    require_columns(
        reference_trajectory,
        [
            "sequence_frame_id",
            "token0_id",
            "reference_cumulative_distance_m",
            "reference_x_m",
            "reference_y_m",
        ],
        "Prepared reference trajectory",
    )

    scale = float(transform["scale_m_per_px"])
    rotation_rad = float(transform["rotation_rad"])
    translation = np.asarray(
        [
            float(transform["translation_x_m"]),
            float(transform["translation_y_m"]),
        ],
        dtype=float,
    )

    visual = raw[["visual_x_px", "visual_y_px"]].to_numpy(dtype=float)
    rotation = rotation_matrix(rotation_rad)
    mapped = scale * (visual @ rotation.T) + translation

    raw["raw_map_x_m"] = mapped[:, 0]
    raw["raw_map_y_m"] = mapped[:, 1]

    frame = raw[
        [
            "sequence_frame_id",
            "visual_x_px",
            "visual_y_px",
            "raw_map_x_m",
            "raw_map_y_m",
        ]
    ].merge(
        reference_trajectory[
            [
                "sequence_frame_id",
                "token0_id",
                "reference_cumulative_distance_m",
                "reference_x_m",
                "reference_y_m",
            ]
        ],
        on="sequence_frame_id",
        how="left",
        validate="one_to_one",
    )

    if frame[
        [
            "reference_cumulative_distance_m",
            "reference_x_m",
            "reference_y_m",
        ]
    ].isna().any().any():
        raise ValueError(
            "Raw trajectory did not join completely to evaluation reference"
        )

    frame["raw_map_error_m_eval_only"] = np.hypot(
        frame["raw_map_x_m"] - frame["reference_x_m"],
        frame["raw_map_y_m"] - frame["reference_y_m"],
    )
    return frame


def add_raw_temporal_features(
    diagnostics: pd.DataFrame,
    raw_map_trajectory: pd.DataFrame,
    lock_frame: int,
) -> pd.DataFrame:
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
            "reference_cumulative_distance_m",
        ],
        "S6B correction diagnostics",
    )
    assert_unique(
        diagnostics,
        ["sequence_frame_id"],
        "S6B correction diagnostics",
    )

    query_frame = diagnostics.merge(
        raw_map_trajectory[
            ["sequence_frame_id", "raw_map_x_m", "raw_map_y_m"]
        ],
        on="sequence_frame_id",
        how="left",
        validate="one_to_one",
    )

    if query_frame[["raw_map_x_m", "raw_map_y_m"]].isna().any().any():
        raise ValueError(
            "Some correction opportunities did not join to raw map positions"
        )

    # The lock observation may support later temporal checks. Corrections begin
    # strictly after the lock frame.
    query_frame = query_frame.loc[
        query_frame["sequence_frame_id"] >= lock_frame
    ].copy()
    query_frame = query_frame.sort_values(
        "sequence_frame_id"
    ).reset_index(drop=True)

    support_columns: list[pd.Series] = []

    for lag in TEMPORAL_LAGS:
        prefix = f"raw_lag{lag}"

        relative_dx = (
            query_frame["raw_map_x_m"]
            - query_frame["raw_map_x_m"].shift(lag)
        )
        relative_dy = (
            query_frame["raw_map_y_m"]
            - query_frame["raw_map_y_m"].shift(lag)
        )
        absolute_dx = (
            query_frame["chosen_abs_x_traj01_m"]
            - query_frame["chosen_abs_x_traj01_m"].shift(lag)
        )
        absolute_dy = (
            query_frame["chosen_abs_y_traj01_m"]
            - query_frame["chosen_abs_y_traj01_m"].shift(lag)
        )

        residual = np.hypot(
            absolute_dx - relative_dx,
            absolute_dy - relative_dy,
        )

        query_frame[f"{prefix}_relative_dx_m"] = relative_dx
        query_frame[f"{prefix}_relative_dy_m"] = relative_dy
        query_frame[f"{prefix}_absolute_dx_m"] = absolute_dx
        query_frame[f"{prefix}_absolute_dy_m"] = absolute_dy
        query_frame[f"{prefix}_temporal_residual_m"] = residual

        support_columns.append(
            (residual <= TEMPORAL_RADIUS_M).fillna(False)
        )

    query_frame["raw_temporal_support_r60_count"] = sum(
        item.astype(int) for item in support_columns
    )

    balanced = bool_series(query_frame["balanced_accept_online"])
    permissive = bool_series(query_frame["permissive_accept_online"])
    after_lock = query_frame["sequence_frame_id"] > lock_frame

    temporal_r60 = (
        permissive
        & (query_frame["raw_temporal_support_r60_count"] >= 1)
        & after_lock
    )
    balanced_after_lock = balanced & after_lock

    query_frame["raw_balanced_accept_online"] = balanced_after_lock
    query_frame["raw_temporal_r60_accept_online"] = temporal_r60
    query_frame["raw_hybrid_r60_accept_online"] = (
        balanced_after_lock | temporal_r60
    )
    query_frame["raw_hybrid_r60_temporal_only_online"] = (
        temporal_r60 & ~balanced_after_lock
    )
    return query_frame


def policy_definitions() -> list[dict[str, Any]]:
    return [
        {
            "policy": "raw_bootstrap_relative_only",
            "accept_column": None,
            "mode": "none",
            "balanced_alpha": None,
            "temporal_alpha": None,
        },
        {
            "policy": "raw_bootstrap_balanced_hard_online",
            "accept_column": "raw_balanced_accept_online",
            "mode": "balanced_hard",
            "balanced_alpha": 1.0,
            "temporal_alpha": None,
        },
        {
            "policy": "raw_bootstrap_hybrid_r60_hard_online",
            "accept_column": "raw_hybrid_r60_accept_online",
            "mode": "hybrid_hard",
            "balanced_alpha": 1.0,
            "temporal_alpha": 1.0,
        },
        {
            "policy": "raw_bootstrap_hybrid_r60_b100_t050_online",
            "accept_column": "raw_hybrid_r60_accept_online",
            "mode": "hybrid_adaptive",
            "balanced_alpha": 1.0,
            "temporal_alpha": 0.50,
        },
        {
            "policy": "raw_bootstrap_hybrid_r60_b100_t075_online",
            "accept_column": "raw_hybrid_r60_accept_online",
            "mode": "hybrid_adaptive",
            "balanced_alpha": 1.0,
            "temporal_alpha": 0.75,
        },
    ]


def replay_policy(
    post_lock_trajectory: pd.DataFrame,
    query_frame: pd.DataFrame,
    policy: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    accept_column = policy["accept_column"]

    if accept_column is None:
        accepted = query_frame.iloc[0:0].copy()
    else:
        accepted = query_frame.loc[
            bool_series(query_frame[accept_column])
        ].copy()

    assert_unique(
        accepted,
        ["sequence_frame_id"],
        f"Accepted corrections for {policy['policy']}",
    )

    correction_lookup = {
        int(row["sequence_frame_id"]): row
        for _, row in accepted.iterrows()
    }

    current_offset_x = 0.0
    current_offset_y = 0.0
    previous_event_distance: float | None = None
    event_index = 0

    trajectory_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []

    post_lock_start_distance = float(
        post_lock_trajectory[
            "reference_cumulative_distance_m"
        ].iloc[0]
    )

    for _, row in post_lock_trajectory.iterrows():
        frame_id = int(row["sequence_frame_id"])
        raw_x = float(row["raw_map_x_m"])
        raw_y = float(row["raw_map_y_m"])
        reference_x = float(row["reference_x_m"])
        reference_y = float(row["reference_y_m"])
        distance_m = float(row["reference_cumulative_distance_m"])

        pre_x = raw_x + current_offset_x
        pre_y = raw_y + current_offset_y
        pre_error = float(
            math.hypot(pre_x - reference_x, pre_y - reference_y)
        )

        event = correction_lookup.get(frame_id)
        correction_applied = event is not None
        alpha: float | None = None
        alpha_source: str | None = None
        improvement: float | None = None
        innovation_m: float | None = None
        position_shift_m: float | None = None
        correction_true: bool | None = None
        correction_false: bool | None = None
        correction_dangerous: bool | None = None

        if event is not None:
            balanced = bool_value(event["raw_balanced_accept_online"])

            if policy["mode"] in {"balanced_hard", "hybrid_hard"}:
                alpha = 1.0
                alpha_source = (
                    "balanced_confidence"
                    if balanced
                    else "temporal_only_support"
                )
            elif policy["mode"] == "hybrid_adaptive":
                if balanced:
                    alpha = float(policy["balanced_alpha"])
                    alpha_source = "balanced_confidence"
                else:
                    alpha = float(policy["temporal_alpha"])
                    alpha_source = "temporal_only_support"
            else:
                raise ValueError(
                    f"Unsupported replay mode: {policy['mode']}"
                )

            absolute_x = float(event["chosen_abs_x_traj01_m"])
            absolute_y = float(event["chosen_abs_y_traj01_m"])
            innovation_x = absolute_x - pre_x
            innovation_y = absolute_y - pre_y
            innovation_m = float(math.hypot(innovation_x, innovation_y))

            current_offset_x += alpha * innovation_x
            current_offset_y += alpha * innovation_y

            fused_x = raw_x + current_offset_x
            fused_y = raw_y + current_offset_y
            position_shift_m = alpha * innovation_m

            post_error = float(
                math.hypot(
                    fused_x - reference_x,
                    fused_y - reference_y,
                )
            )
            improvement = pre_error - post_error

            correction_true = bool_value(event["hit_eval_only"])
            correction_false = not correction_true
            correction_dangerous = bool_value(
                event["dangerous_false_eval_only"]
            )

            coordinate_fix_error = float(
                math.hypot(
                    absolute_x - reference_x,
                    absolute_y - reference_y,
                )
            )
            manifest_fix_error = float(
                event["chosen_error_m_eval_only"]
            )
            consistency_delta = abs(
                coordinate_fix_error - manifest_fix_error
            )

            if consistency_delta > FIX_ERROR_CONSISTENCY_TOLERANCE_M:
                raise ValueError(
                    "Absolute-coordinate consistency failed at "
                    f"frame {frame_id}: delta={consistency_delta:.3f} m"
                )

            if previous_event_distance is None:
                gap_since_previous = (
                    distance_m - post_lock_start_distance
                )
            else:
                gap_since_previous = (
                    distance_m - previous_event_distance
                )
            previous_event_distance = distance_m

            event_rows.append(
                {
                    "policy": policy["policy"],
                    "event_index": event_index,
                    "sequence_frame_id": frame_id,
                    "token": int(event["token"]),
                    "chosen_tile_id": int(event["chosen_tile_id"]),
                    "reference_cumulative_distance_m": distance_m,
                    "gap_since_previous_event_m": gap_since_previous,
                    "correction_alpha": alpha,
                    "correction_alpha_source": alpha_source,
                    "raw_temporal_support_r60_count": int(
                        event["raw_temporal_support_r60_count"]
                    ),
                    "raw_lag1_temporal_residual_m": event.get(
                        "raw_lag1_temporal_residual_m",
                        np.nan,
                    ),
                    "pre_correction_error_m": pre_error,
                    "absolute_innovation_m": innovation_m,
                    "position_shift_applied_m": position_shift_m,
                    "post_correction_error_m": post_error,
                    "error_improvement_m": improvement,
                    "error_worsened": improvement < 0.0,
                    "correction_true_eval_only": correction_true,
                    "correction_false_eval_only": correction_false,
                    "correction_dangerous_false_gt100m_eval_only":
                        correction_dangerous,
                    "chosen_error_m_eval_only": manifest_fix_error,
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
                "policy": policy["policy"],
                "sequence_frame_id": frame_id,
                "token0_id": int(row["token0_id"]),
                "reference_cumulative_distance_m": distance_m,
                "reference_x_m": reference_x,
                "reference_y_m": reference_y,
                "raw_map_x_m": raw_x,
                "raw_map_y_m": raw_y,
                "raw_map_error_m_eval_only": float(
                    row["raw_map_error_m_eval_only"]
                ),
                "fused_x_m": fused_x,
                "fused_y_m": fused_y,
                "fused_error_m": fused_error,
                "active_offset_x_m": current_offset_x,
                "active_offset_y_m": current_offset_y,
                "correction_applied": correction_applied,
                "correction_alpha": alpha,
                "correction_alpha_source": alpha_source,
                "correction_absolute_innovation_m": innovation_m,
                "correction_position_shift_m": position_shift_m,
                "correction_error_improvement_m": improvement,
                "correction_true_eval_only": correction_true,
                "correction_false_eval_only": correction_false,
                "correction_dangerous_false_gt100m_eval_only":
                    correction_dangerous,
            }
        )

    return pd.DataFrame(trajectory_rows), pd.DataFrame(event_rows)


def calculate_metrics(
    trajectory: pd.DataFrame,
    events: pd.DataFrame,
    lock_frame: int,
    lock_distance_m: float,
    full_end_distance_m: float,
) -> dict[str, Any]:
    errors = pd.to_numeric(
        trajectory["fused_error_m"],
        errors="raise",
    ).to_numpy(dtype=float)

    post_lock_distance_m = full_end_distance_m - lock_distance_m

    if events.empty:
        accepted = 0
        true_events = 0
        false_events = 0
        dangerous_events = 0
        precision = None
        mean_alpha = None
        median_improvement = None
    else:
        true_mask = bool_series(
            events["correction_true_eval_only"]
        )
        dangerous_mask = bool_series(
            events[
                "correction_dangerous_false_gt100m_eval_only"
            ]
        )

        accepted = int(len(events))
        true_events = int(true_mask.sum())
        false_events = accepted - true_events
        dangerous_events = int(dangerous_mask.sum())
        precision = true_events / accepted
        mean_alpha = float(events["correction_alpha"].mean())
        median_improvement = float(
            events["error_improvement_m"].median()
        )

    return {
        "policy": str(trajectory["policy"].iloc[0]),
        "lock_frame": lock_frame,
        "lock_reference_distance_m_eval_only": lock_distance_m,
        "post_lock_frames": int(len(trajectory)),
        "post_lock_distance_m_eval_only": post_lock_distance_m,
        "accepted": accepted,
        "true_events_eval_only": true_events,
        "false_events_eval_only": false_events,
        "dangerous_false_events_gt100m_eval_only": dangerous_events,
        "precision_eval_only": precision,
        "mean_correction_alpha": mean_alpha,
        "median_event_improvement_m_eval_only": median_improvement,
        "rmse_m": float(
            math.sqrt(float(np.mean(np.square(errors))))
        ),
        "mean_error_m": float(np.mean(errors)),
        "median_error_m": float(np.median(errors)),
        "p95_error_m": float(np.quantile(errors, 0.95)),
        "max_error_m": float(np.max(errors)),
        "final_error_m": float(errors[-1]),
        "failure_rate_gt40m": float(np.mean(errors > 40.0)),
        "final_error_per_100m_post_lock": (
            float(errors[-1]) / post_lock_distance_m * 100.0
            if post_lock_distance_m > 0.0
            else None
        ),
    }


def add_relative_deltas(metrics: pd.DataFrame) -> pd.DataFrame:
    baseline_rows = metrics.loc[
        metrics["policy"] == "raw_bootstrap_relative_only"
    ]
    if len(baseline_rows) != 1:
        raise ValueError(
            "Expected exactly one raw relative-only metric row"
        )

    baseline = baseline_rows.iloc[0]
    frame = metrics.copy()

    for column in [
        "rmse_m",
        "p95_error_m",
        "max_error_m",
        "final_error_m",
        "failure_rate_gt40m",
    ]:
        baseline_value = float(baseline[column])
        frame[f"{column}_reduction_vs_raw_relative_pct"] = (
            (baseline_value - frame[column])
            / baseline_value
            * 100.0
            if baseline_value != 0.0
            else np.nan
        )

    return frame


def residual_summary(
    query_frame: pd.DataFrame,
    lock_frame: int,
) -> dict[str, Any]:
    permissive = bool_series(
        query_frame["permissive_accept_online"]
    )
    after_lock = query_frame["sequence_frame_id"] > lock_frame
    hit = bool_series(query_frame["hit_eval_only"])
    residual = pd.to_numeric(
        query_frame["raw_lag1_temporal_residual_m"],
        errors="coerce",
    )

    result: dict[str, Any] = {}

    for label, mask in [
        (
            "permissive_true_eval_only",
            permissive & after_lock & hit,
        ),
        (
            "permissive_false_eval_only",
            permissive & after_lock & ~hit,
        ),
    ]:
        values = residual.loc[mask].dropna()
        result[label] = {
            "rows": int(len(values)),
            "median_m": (
                float(values.median()) if len(values) else None
            ),
            "p95_m": (
                float(values.quantile(0.95))
                if len(values)
                else None
            ),
        }

    return result


def plot_error_by_distance(
    replay: pd.DataFrame,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(14, 7))

    for policy, group in replay.groupby("policy", sort=False):
        group = group.sort_values(
            "reference_cumulative_distance_m"
        )
        ax.plot(
            group["reference_cumulative_distance_m"],
            group["fused_error_m"],
            linewidth=1.25,
            label=policy,
        )

    ax.axhline(
        40.0,
        linestyle="--",
        linewidth=1.0,
        label="40 m evaluation threshold",
    )
    ax.set_xlabel(
        "Reference cumulative distance [m] — evaluation only"
    )
    ax.set_ylabel("Position error [m] — evaluation only")
    ax.set_title(
        "S6B.2C raw causal-bootstrap post-lock error"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_trajectories(
    replay: pd.DataFrame,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 10))

    reference = replay.loc[
        replay["policy"] == "raw_bootstrap_relative_only"
    ].sort_values("sequence_frame_id")

    ax.plot(
        reference["reference_x_m"],
        reference["reference_y_m"],
        linewidth=2.0,
        label="Reference — evaluation only",
    )

    for policy, group in replay.groupby("policy", sort=False):
        group = group.sort_values("sequence_frame_id")
        ax.plot(
            group["fused_x_m"],
            group["fused_y_m"],
            linewidth=1.1,
            label=policy,
        )

    ax.set_xlabel("traj01 X [m]")
    ax.set_ylabel("traj01 Y [m]")
    ax.set_title(
        "S6B.2C raw bootstrap and adaptive-fusion trajectories"
    )
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    replay_helpers = load_s6b1a_module()

    for directory in [
        args.metadata_dir,
        args.report_dir,
        args.figure_dir,
    ]:
        directory.mkdir(parents=True, exist_ok=True)

    raw_all = pd.read_csv(
        args.raw_trajectory,
        low_memory=False,
    )
    aligned_all = pd.read_csv(
        args.aligned_trajectory,
        low_memory=False,
    )
    diagnostics = pd.read_csv(
        args.diagnostics,
        low_memory=False,
    )
    transform = json.loads(
        args.bootstrap_transform.read_text(
            encoding="utf-8"
        )
    )

    for key in [
        "lock_sequence_frame_id",
        "lock_reference_distance_m_eval_only",
        "scale_m_per_px",
        "rotation_rad",
        "translation_x_m",
        "translation_y_m",
    ]:
        if key not in transform:
            raise ValueError(
                f"Bootstrap transform is missing key: {key}"
            )

    lock_frame = int(transform["lock_sequence_frame_id"])
    lock_distance_m = float(
        transform["lock_reference_distance_m_eval_only"]
    )

    reference_trajectory = replay_helpers.prepare_trajectory(
        trajectory_all=aligned_all,
        variant=args.relative_variant,
        baseline_x_column=args.baseline_x_column,
        baseline_y_column=args.baseline_y_column,
    )

    raw_map_trajectory = prepare_raw_map_trajectory(
        raw_all=raw_all,
        reference_trajectory=reference_trajectory,
        transform=transform,
        variant=args.relative_variant,
    )

    lock_query = diagnostics.loc[
        diagnostics["sequence_frame_id"] == lock_frame
    ]
    if len(lock_query) != 1:
        raise ValueError(
            f"Expected one diagnostics row at lock frame "
            f"{lock_frame}, found {len(lock_query)}"
        )

    lock_raw = raw_map_trajectory.loc[
        raw_map_trajectory["sequence_frame_id"] == lock_frame
    ]
    if len(lock_raw) != 1:
        raise ValueError(
            f"Expected one raw-map row at lock frame "
            f"{lock_frame}, found {len(lock_raw)}"
        )

    lock_anchor_delta_m = float(
        math.hypot(
            float(lock_raw.iloc[0]["raw_map_x_m"])
            - float(
                lock_query.iloc[0]["chosen_abs_x_traj01_m"]
            ),
            float(lock_raw.iloc[0]["raw_map_y_m"])
            - float(
                lock_query.iloc[0]["chosen_abs_y_traj01_m"]
            ),
        )
    )

    if lock_anchor_delta_m > 1e-3:
        raise ValueError(
            "Selected bootstrap transform does not map the lock "
            f"anchor correctly: delta={lock_anchor_delta_m:.6f} m"
        )

    query_frame = add_raw_temporal_features(
        diagnostics=diagnostics,
        raw_map_trajectory=raw_map_trajectory,
        lock_frame=lock_frame,
    )

    post_lock_trajectory = raw_map_trajectory.loc[
        raw_map_trajectory["sequence_frame_id"] >= lock_frame
    ].copy()
    post_lock_trajectory = post_lock_trajectory.sort_values(
        "sequence_frame_id"
    ).reset_index(drop=True)

    if post_lock_trajectory.empty:
        raise ValueError(
            "No trajectory frames were available at or after lock"
        )

    full_end_distance_m = float(
        raw_map_trajectory[
            "reference_cumulative_distance_m"
        ].iloc[-1]
    )

    replay_frames: list[pd.DataFrame] = []
    event_frames: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []

    for policy in policy_definitions():
        policy_trajectory, policy_events = replay_policy(
            post_lock_trajectory=post_lock_trajectory,
            query_frame=query_frame,
            policy=policy,
        )

        metrics = calculate_metrics(
            trajectory=policy_trajectory,
            events=policy_events,
            lock_frame=lock_frame,
            lock_distance_m=lock_distance_m,
            full_end_distance_m=full_end_distance_m,
        )

        replay_frames.append(policy_trajectory)
        metric_rows.append(metrics)

        if not policy_events.empty:
            event_frames.append(policy_events)

    replay = pd.concat(
        replay_frames,
        ignore_index=True,
    )
    events = (
        pd.concat(
            event_frames,
            ignore_index=True,
        )
        if event_frames
        else pd.DataFrame()
    )
    metrics = add_relative_deltas(
        pd.DataFrame(metric_rows)
    )

    diagnostics_path = (
        args.metadata_dir
        / "s6b2c_raw_temporal_diagnostics.csv"
    )
    masks_path = (
        args.metadata_dir
        / "s6b2c_raw_policy_masks.csv"
    )
    mapped_path = (
        args.metadata_dir
        / "s6b2c_raw_bootstrapped_map_trajectory.csv"
    )
    replay_path = (
        args.metadata_dir
        / "s6b2c_raw_causal_fusion_trajectories.csv"
    )
    events_path = (
        args.metadata_dir
        / "s6b2c_raw_causal_fusion_events.csv"
    )
    metrics_path = (
        args.metadata_dir
        / "s6b2c_raw_causal_fusion_metrics.csv"
    )
    summary_path = (
        args.report_dir
        / "s6b2c_raw_causal_fusion_summary.json"
    )
    error_figure_path = (
        args.figure_dir
        / "s6b2c_raw_causal_error_by_distance.png"
    )
    trajectory_figure_path = (
        args.figure_dir
        / "s6b2c_raw_causal_trajectory_comparison.png"
    )

    query_frame.to_csv(
        diagnostics_path,
        index=False,
    )
    query_frame[
        [
            "sequence_frame_id",
            "token",
            "raw_balanced_accept_online",
            "raw_temporal_r60_accept_online",
            "raw_hybrid_r60_accept_online",
            "raw_hybrid_r60_temporal_only_online",
            "raw_temporal_support_r60_count",
            "hit_eval_only",
            "dangerous_false_eval_only",
        ]
    ].to_csv(
        masks_path,
        index=False,
    )
    raw_map_trajectory.to_csv(
        mapped_path,
        index=False,
    )
    replay.to_csv(
        replay_path,
        index=False,
    )
    events.to_csv(
        events_path,
        index=False,
    )
    metrics.to_csv(
        metrics_path,
        index=False,
    )

    plot_error_by_distance(
        replay=replay,
        output_path=error_figure_path,
    )
    plot_trajectories(
        replay=replay,
        output_path=trajectory_figure_path,
    )

    temporal_summary = residual_summary(
        query_frame,
        lock_frame,
    )

    summary = {
        "stage": "S6B.2C",
        "title": (
            "Raw causal-bootstrap temporal adaptive-fusion replay"
        ),
        "relative_variant": args.relative_variant,
        "bootstrap_lock": {
            "frame": lock_frame,
            "reference_distance_m_eval_only":
                lock_distance_m,
            "anchor_reconstruction_delta_m":
                lock_anchor_delta_m,
            "scale_m_per_px":
                float(transform["scale_m_per_px"]),
            "rotation_deg":
                float(transform["rotation_deg"]),
        },
        "fairness_rules": [
            "Metrics begin at the bootstrap lock frame.",
            "The lock-frame observation is not applied again as a correction.",
            "Temporal support is recomputed from the raw bootstrapped trajectory.",
            "Reference coordinates and correctness labels are evaluation-only.",
        ],
        "raw_temporal_lag1_summary_eval_only":
            temporal_summary,
        "policy_metrics": {
            row["policy"]: {
                key: json_safe(value)
                for key, value in row.items()
            }
            for row in metrics.to_dict(
                orient="records"
            )
        },
        "interpretation_rule": (
            "Adaptive fusion survives the causal mapping test if an "
            "adaptive raw-bootstrap policy improves post-lock trajectory "
            "metrics over raw_bootstrap_relative_only, and preferably over "
            "raw_bootstrap_balanced_hard_online."
        ),
        "selection_warning": (
            "All thresholds and policy choices remain exploratory on traj01 "
            "and require independent validation."
        ),
        "outputs": {
            "raw_temporal_diagnostics":
                diagnostics_path,
            "policy_masks":
                masks_path,
            "raw_bootstrapped_map_trajectory":
                mapped_path,
            "fusion_trajectories":
                replay_path,
            "correction_events":
                events_path,
            "metrics":
                metrics_path,
            "error_figure":
                error_figure_path,
            "trajectory_figure":
                trajectory_figure_path,
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

    print("S6B.2C Raw Causal Adaptive-Fusion Replay")
    print("----------------------------------------")
    print(f"Relative variant:       {args.relative_variant}")
    print(f"Bootstrap lock frame:   {lock_frame}")
    print(
        "Bootstrap distance:     "
        f"{lock_distance_m:.3f} m (evaluation only)"
    )
    print(
        "Post-lock distance:     "
        f"{full_end_distance_m - lock_distance_m:.3f} m"
    )
    print(
        "Lock anchor delta:      "
        f"{lock_anchor_delta_m:.9f} m"
    )

    print()
    print("Raw lag-1 temporal separation")
    print("-----------------------------")
    for label, values in temporal_summary.items():
        print(
            f"{label:30s} "
            f"rows={values['rows']:3d}  "
            f"median={values['median_m']}  "
            f"p95={values['p95_m']}"
        )

    print()
    print("Post-lock policy comparison")
    print("---------------------------")

    display_columns = [
        "policy",
        "accepted",
        "true_events_eval_only",
        "false_events_eval_only",
        "dangerous_false_events_gt100m_eval_only",
        "precision_eval_only",
        "mean_correction_alpha",
        "rmse_m",
        "p95_error_m",
        "max_error_m",
        "final_error_m",
        "failure_rate_gt40m",
        "rmse_m_reduction_vs_raw_relative_pct",
    ]

    print(
        metrics[display_columns]
        .to_string(index=False)
    )

    print()
    print(f"Raw temporal diagnostics: {diagnostics_path}")
    print(f"Policy masks:             {masks_path}")
    print(f"Mapped raw trajectory:    {mapped_path}")
    print(f"Fusion trajectories:      {replay_path}")
    print(f"Correction events:        {events_path}")
    print(f"Metrics:                  {metrics_path}")
    print(f"Summary:                  {summary_path}")
    print(f"Figures:                  {args.figure_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
