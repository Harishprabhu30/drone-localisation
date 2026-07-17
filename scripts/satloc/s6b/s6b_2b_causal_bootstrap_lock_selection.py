#!/usr/bin/env python3
"""S6B.2B — Causal raw-trajectory bootstrap lock selection.

Selects the earliest sufficiently stable raw-visual-to-map similarity
transformation using online-safe hypothesis-consensus quantities only.

Ground-truth labels are reported afterward and are never used in the
lock decision.

Command USed:

export PYTHONPATH=$PWD/src

python scripts/satloc/s6b/s6b_2b_causal_bootstrap_lock_selection.py \
  2>&1 | tee \
  outputs/satloc/reports/s6b_relative_absolute/s6b2b_causal_bootstrap_lock_selection.log

"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


WINNERS_PATH = Path(
    "outputs/satloc/metadata/s6b_relative_absolute/"
    "s6b2a_raw_bootstrap_causal_winners.csv"
)

EVENTS_PATH = Path(
    "outputs/satloc/metadata/s6b_relative_absolute/"
    "s6b2a_balanced_bootstrap_events.csv"
)

OUTPUT_DIR = Path(
    "outputs/satloc/metadata/s6b_relative_absolute"
)

REPORT_DIR = Path(
    "outputs/satloc/reports/s6b_relative_absolute"
)

STABILITY_WINDOW = 3

MIN_AVAILABLE_EVENTS = 6
MIN_R60_INLIERS = 6
MIN_R60_INLIER_RATE = 0.70
MAX_MEDIAN_RESIDUAL_M = 20.0
MAX_P95_RESIDUAL_M = 110.0
MIN_ANCHOR_MAP_SEPARATION_M = 300.0

MAX_SCALE_RELATIVE_SPAN = 0.15
MAX_ROTATION_SPREAD_DEG = 5.0


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


def wrap_degrees(angle_deg: float) -> float:
    return float(
        (angle_deg + 180.0) % 360.0 - 180.0
    )


def circular_mean_degrees(
    values_deg: np.ndarray,
) -> float:
    radians = np.deg2rad(values_deg)

    mean_angle = math.atan2(
        float(np.mean(np.sin(radians))),
        float(np.mean(np.cos(radians))),
    )

    return wrap_degrees(math.degrees(mean_angle))


def circular_spread_degrees(
    values_deg: np.ndarray,
) -> float:
    centre = circular_mean_degrees(values_deg)

    deviations = np.asarray(
        [
            abs(wrap_degrees(float(value) - centre))
            for value in values_deg
        ],
        dtype=float,
    )

    return float(deviations.max())


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    winners = pd.read_csv(
        WINNERS_PATH,
        low_memory=False,
    )

    events = pd.read_csv(
        EVENTS_PATH,
        low_memory=False,
    )

    require_columns(
        winners,
        [
            "available_event_count",
            "anchor_start_event_index",
            "anchor_end_event_index",
            "anchor_start_frame",
            "anchor_end_frame",
            "anchor_start_token",
            "anchor_end_token",
            "map_separation_m",
            "scale_m_per_px",
            "rotation_deg",
            "translation_x_m",
            "translation_y_m",
            "prefix_inliers_r60",
            "prefix_inlier_rate_r60",
            "median_prefix_residual_m",
            "p95_prefix_residual_m",
            "max_prefix_residual_m",
            "anchor_start_true_eval_only",
            "anchor_end_true_eval_only",
        ],
        "S6B.2A causal winners",
    )

    require_columns(
        events,
        [
            "bootstrap_event_index",
            "sequence_frame_id",
            "token",
            "reference_cumulative_distance_m",
            "hit_eval_only",
            "chosen_error_m_eval_only",
        ],
        "S6B.2A balanced events",
    )

    winners = winners.sort_values(
        "available_event_count"
    ).reset_index(drop=True)

    winners["rotation_wrapped_deg"] = (
        winners["rotation_deg"]
        .astype(float)
        .map(wrap_degrees)
    )

    winners["current_consensus_pass_online"] = (
        (
            winners["available_event_count"]
            >= MIN_AVAILABLE_EVENTS
        )
        & (
            winners["prefix_inliers_r60"]
            >= MIN_R60_INLIERS
        )
        & (
            winners["prefix_inlier_rate_r60"]
            >= MIN_R60_INLIER_RATE
        )
        & (
            winners["median_prefix_residual_m"]
            <= MAX_MEDIAN_RESIDUAL_M
        )
        & (
            winners["p95_prefix_residual_m"]
            <= MAX_P95_RESIDUAL_M
        )
        & (
            winners["map_separation_m"]
            >= MIN_ANCHOR_MAP_SEPARATION_M
        )
    )

    stability_rows: list[dict[str, Any]] = []

    for row_index, row in winners.iterrows():
        start_index = (
            row_index - STABILITY_WINDOW + 1
        )

        result: dict[str, Any] = {
            "available_event_count":
                int(row["available_event_count"]),
            "anchor_end_frame":
                int(row["anchor_end_frame"]),
            "anchor_end_token":
                int(row["anchor_end_token"]),
            "scale_m_per_px":
                float(row["scale_m_per_px"]),
            "rotation_wrapped_deg":
                float(row["rotation_wrapped_deg"]),
            "translation_x_m":
                float(row["translation_x_m"]),
            "translation_y_m":
                float(row["translation_y_m"]),
            "prefix_inliers_r60":
                int(row["prefix_inliers_r60"]),
            "prefix_inlier_rate_r60":
                float(row["prefix_inlier_rate_r60"]),
            "p95_prefix_residual_m":
                float(row["p95_prefix_residual_m"]),
            "current_consensus_pass_online":
                bool(row["current_consensus_pass_online"]),
            "stability_window_available":
                start_index >= 0,
            "all_window_states_pass_online": False,
            "scale_relative_span": None,
            "rotation_spread_deg": None,
            "scale_stability_pass_online": False,
            "rotation_stability_pass_online": False,
            "bootstrap_lock_pass_online": False,
        }

        if start_index >= 0:
            window = winners.iloc[
                start_index: row_index + 1
            ]

            scales = window[
                "scale_m_per_px"
            ].to_numpy(dtype=float)

            rotations = window[
                "rotation_wrapped_deg"
            ].to_numpy(dtype=float)

            scale_centre = float(
                np.median(scales)
            )

            scale_relative_span = float(
                (
                    np.max(scales)
                    - np.min(scales)
                )
                / max(abs(scale_centre), 1e-12)
            )

            rotation_spread = (
                circular_spread_degrees(rotations)
            )

            all_states_pass = bool(
                window[
                    "current_consensus_pass_online"
                ].all()
            )

            scale_pass = (
                scale_relative_span
                <= MAX_SCALE_RELATIVE_SPAN
            )

            rotation_pass = (
                rotation_spread
                <= MAX_ROTATION_SPREAD_DEG
            )

            lock_pass = (
                all_states_pass
                and scale_pass
                and rotation_pass
            )

            result.update(
                {
                    "all_window_states_pass_online":
                        all_states_pass,
                    "scale_relative_span":
                        scale_relative_span,
                    "rotation_spread_deg":
                        rotation_spread,
                    "scale_stability_pass_online":
                        scale_pass,
                    "rotation_stability_pass_online":
                        rotation_pass,
                    "bootstrap_lock_pass_online":
                        lock_pass,
                }
            )

        stability_rows.append(result)

    stability = pd.DataFrame(stability_rows)

    lock_candidates = stability.loc[
        stability["bootstrap_lock_pass_online"]
    ].copy()

    if lock_candidates.empty:
        raise ValueError(
            "No causal bootstrap lock satisfied the current "
            "online-only consensus and stability conditions."
        )

    selected_stability = lock_candidates.iloc[0]

    selected_event_count = int(
        selected_stability["available_event_count"]
    )

    selected_winner_rows = winners.loc[
        winners["available_event_count"]
        == selected_event_count
    ]

    if len(selected_winner_rows) != 1:
        raise ValueError(
            "Expected exactly one selected causal winner for "
            f"event count {selected_event_count}, found "
            f"{len(selected_winner_rows)}"
        )

    selected_winner = selected_winner_rows.iloc[0]

    lock_frame = int(
        selected_winner["anchor_end_frame"]
    )

    lock_event_rows = events.loc[
        events["sequence_frame_id"] == lock_frame
    ]

    if len(lock_event_rows) != 1:
        raise ValueError(
            f"Could not uniquely resolve lock event at frame "
            f"{lock_frame}"
        )

    lock_event = lock_event_rows.iloc[0]

    # Evaluation-only annotations attached after online lock selection.
    prefix_events = events.loc[
        events["bootstrap_event_index"]
        < selected_event_count
    ].copy()

    true_prefix_count = int(
        prefix_events["hit_eval_only"]
        .map(bool_value)
        .sum()
    )

    false_prefix_count = (
        int(len(prefix_events))
        - true_prefix_count
    )

    selected_transform = {
        "available_event_count":
            selected_event_count,
        "lock_sequence_frame_id":
            lock_frame,
        "lock_token":
            int(lock_event["token"]),
        "lock_reference_distance_m_eval_only":
            float(
                lock_event[
                    "reference_cumulative_distance_m"
                ]
            ),
        "scale_m_per_px":
            float(selected_winner["scale_m_per_px"]),
        "rotation_deg":
            float(
                selected_winner[
                    "rotation_wrapped_deg"
                ]
            ),
        "rotation_rad":
            math.radians(
                float(
                    selected_winner[
                        "rotation_wrapped_deg"
                    ]
                )
            ),
        "translation_x_m":
            float(selected_winner["translation_x_m"]),
        "translation_y_m":
            float(selected_winner["translation_y_m"]),
        "anchor_start_frame":
            int(selected_winner["anchor_start_frame"]),
        "anchor_end_frame":
            int(selected_winner["anchor_end_frame"]),
        "anchor_start_token":
            int(selected_winner["anchor_start_token"]),
        "anchor_end_token":
            int(selected_winner["anchor_end_token"]),
        "anchor_map_separation_m":
            float(selected_winner["map_separation_m"]),
        "prefix_inliers_r60":
            int(selected_winner["prefix_inliers_r60"]),
        "prefix_inlier_rate_r60":
            float(
                selected_winner[
                    "prefix_inlier_rate_r60"
                ]
            ),
        "median_prefix_residual_m":
            float(
                selected_winner[
                    "median_prefix_residual_m"
                ]
            ),
        "p95_prefix_residual_m":
            float(
                selected_winner[
                    "p95_prefix_residual_m"
                ]
            ),
        "scale_relative_span_over_lock_window":
            float(
                selected_stability[
                    "scale_relative_span"
                ]
            ),
        "rotation_spread_deg_over_lock_window":
            float(
                selected_stability[
                    "rotation_spread_deg"
                ]
            ),
        "prefix_true_events_eval_only":
            true_prefix_count,
        "prefix_false_events_eval_only":
            false_prefix_count,
    }

    stability_path = (
        OUTPUT_DIR
        / "s6b2b_bootstrap_lock_stability.csv"
    )

    lock_path = (
        OUTPUT_DIR
        / "s6b2b_selected_bootstrap_transform.json"
    )

    summary_path = (
        REPORT_DIR
        / "s6b2b_causal_bootstrap_lock_summary.json"
    )

    stability.to_csv(
        stability_path,
        index=False,
    )

    lock_path.write_text(
        json.dumps(
            json_safe(selected_transform),
            indent=2,
        ),
        encoding="utf-8",
    )

    summary = {
        "stage": "S6B.2B",
        "title": "Causal raw bootstrap lock selection",
        "selection_inputs_online_only": [
            "accepted balanced absolute coordinates",
            "raw visual coordinates",
            "r60 consensus inliers",
            "consensus residuals",
            "scale stability",
            "rotation stability",
            "sequence order",
        ],
        "evaluation_only_annotations": [
            "reference distance at lock",
            "true/false accepted-event counts",
        ],
        "conditions": {
            "stability_window":
                STABILITY_WINDOW,
            "minimum_available_events":
                MIN_AVAILABLE_EVENTS,
            "minimum_r60_inliers":
                MIN_R60_INLIERS,
            "minimum_r60_inlier_rate":
                MIN_R60_INLIER_RATE,
            "maximum_median_residual_m":
                MAX_MEDIAN_RESIDUAL_M,
            "maximum_p95_residual_m":
                MAX_P95_RESIDUAL_M,
            "minimum_anchor_map_separation_m":
                MIN_ANCHOR_MAP_SEPARATION_M,
            "maximum_scale_relative_span":
                MAX_SCALE_RELATIVE_SPAN,
            "maximum_rotation_spread_deg":
                MAX_ROTATION_SPREAD_DEG,
        },
        "selected_transform":
            selected_transform,
        "outputs": {
            "stability_diagnostics":
                stability_path,
            "selected_transform":
                lock_path,
        },
    }

    summary_path.write_text(
        json.dumps(
            json_safe(summary),
            indent=2,
        ),
        encoding="utf-8",
    )

    print("S6B.2B Causal Bootstrap Lock Selection")
    print("--------------------------------------")
    print(
        f"Selected after observations: "
        f"{selected_event_count}"
    )
    print(f"Lock frame:                 {lock_frame}")
    print(
        f"Lock token:                 "
        f"{int(lock_event['token'])}"
    )
    print(
        "Lock reference distance:    "
        f"{float(lock_event['reference_cumulative_distance_m']):.3f} m "
        "(evaluation only)"
    )
    print(
        f"Scale:                      "
        f"{selected_transform['scale_m_per_px']:.6f} m/px"
    )
    print(
        f"Rotation:                   "
        f"{selected_transform['rotation_deg']:.6f} deg"
    )
    print(
        f"Translation:                "
        f"({selected_transform['translation_x_m']:.3f}, "
        f"{selected_transform['translation_y_m']:.3f}) m"
    )
    print(
        f"r60 prefix support:         "
        f"{selected_transform['prefix_inliers_r60']}/"
        f"{selected_event_count}"
    )
    print(
        f"Median prefix residual:     "
        f"{selected_transform['median_prefix_residual_m']:.3f} m"
    )
    print(
        f"p95 prefix residual:        "
        f"{selected_transform['p95_prefix_residual_m']:.3f} m"
    )
    print(
        f"Scale span over lock window:"
        f" {selected_transform['scale_relative_span_over_lock_window']:.4f}"
    )
    print(
        f"Rotation spread:            "
        f"{selected_transform['rotation_spread_deg_over_lock_window']:.3f} deg"
    )
    print(
        f"Prefix true/false eval-only:"
        f" {true_prefix_count}/{false_prefix_count}"
    )

    print()
    print(f"Stability: {stability_path}")
    print(f"Transform: {lock_path}")
    print(f"Summary:   {summary_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
