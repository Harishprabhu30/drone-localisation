#!/usr/bin/env python3
"""S6B.2A — Raw ORB map-frame bootstrap preflight.

Tests whether confidence-gated absolute observations can provide
online scale, map rotation and translation for the raw ORB trajectory.

Hypothesis generation uses only:
    raw visual positions
    accepted absolute tile-centre positions
    sequence order

Evaluation labels are attached only after hypothesis construction.

Command Used:

export PYTHONPATH=$PWD/src

python scripts/satloc/s6b/s6b_2a_raw_bootstrap_preflight.py \
  2>&1 | tee \
  outputs/satloc/reports/s6b_relative_absolute/s6b2a_raw_bootstrap_preflight.log
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


RAW_PATH = Path(
    "outputs/satloc/metadata/s6a_relative_motion/"
    "s6a2_orb_relative_trajectory_pixels.csv"
)

ALIGNED_PATH = Path(
    "outputs/satloc/metadata/s6a_relative_motion/"
    "s6a2_orb_relative_trajectory_aligned_eval_only.csv"
)

DIAGNOSTICS_PATH = Path(
    "outputs/satloc/metadata/s6b_relative_absolute/"
    "s6b1c0_temporal_agreement_diagnostics.csv"
)

OUTPUT_DIR = Path(
    "outputs/satloc/metadata/s6b_relative_absolute"
)

REPORT_DIR = Path(
    "outputs/satloc/reports/s6b_relative_absolute"
)

VARIANT = "se2_scale_normalized"
RESIDUAL_THRESHOLDS_M = [40.0, 60.0, 80.0]


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
        return [
            json_safe(item)
            for item in value
        ]

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

    return np.array(
        [
            [cosine, -sine],
            [sine, cosine],
        ],
        dtype=float,
    )


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    raw_all = pd.read_csv(RAW_PATH)
    aligned_all = pd.read_csv(ALIGNED_PATH)
    diagnostics = pd.read_csv(DIAGNOSTICS_PATH)

    require_columns(
        raw_all,
        [
            "variant",
            "sequence_frame_id",
            "visual_x_px",
            "visual_y_px",
            "visual_yaw_rad",
        ],
        "Raw S6A trajectory",
    )

    require_columns(
        aligned_all,
        [
            "variant",
            "sequence_frame_id",
            "visual_x_px",
            "visual_y_px",
        ],
        "Aligned S6A trajectory",
    )

    require_columns(
        diagnostics,
        [
            "sequence_frame_id",
            "token",
            "reference_cumulative_distance_m",
            "chosen_abs_x_traj01_m",
            "chosen_abs_y_traj01_m",
            "balanced_accept_online",
            "hit_eval_only",
            "chosen_error_m_eval_only",
        ],
        "S6B temporal diagnostics",
    )

    raw = raw_all.loc[
        raw_all["variant"].astype(str) == VARIANT
    ].copy()

    aligned = aligned_all.loc[
        aligned_all["variant"].astype(str) == VARIANT
    ].copy()

    raw = raw.sort_values(
        "sequence_frame_id"
    ).reset_index(drop=True)

    aligned = aligned.sort_values(
        "sequence_frame_id"
    ).reset_index(drop=True)

    if len(raw) != 1034:
        raise ValueError(
            f"Expected 1034 raw rows, found {len(raw)}"
        )

    if raw["sequence_frame_id"].duplicated().any():
        raise ValueError(
            "Raw trajectory contains duplicate frame IDs"
        )

    visual_check = raw[
        [
            "sequence_frame_id",
            "visual_x_px",
            "visual_y_px",
        ]
    ].merge(
        aligned[
            [
                "sequence_frame_id",
                "visual_x_px",
                "visual_y_px",
            ]
        ],
        on="sequence_frame_id",
        how="inner",
        validate="one_to_one",
        suffixes=("_raw", "_aligned"),
    )

    visual_x_delta = np.abs(
        visual_check["visual_x_px_raw"]
        - visual_check["visual_x_px_aligned"]
    )

    visual_y_delta = np.abs(
        visual_check["visual_y_px_raw"]
        - visual_check["visual_y_px_aligned"]
    )

    maximum_visual_copy_delta = float(
        max(
            visual_x_delta.max(),
            visual_y_delta.max(),
        )
    )

    if maximum_visual_copy_delta > 1e-9:
        raise ValueError(
            "Raw and aligned files do not contain identical "
            f"visual trajectories: max delta="
            f"{maximum_visual_copy_delta}"
        )

    events = diagnostics.loc[
        bool_series(
            diagnostics["balanced_accept_online"]
        )
    ].copy()

    events = events.merge(
        raw[
            [
                "sequence_frame_id",
                "visual_x_px",
                "visual_y_px",
                "visual_yaw_rad",
            ]
        ],
        on="sequence_frame_id",
        how="left",
        validate="one_to_one",
    )

    events = events.sort_values(
        "sequence_frame_id"
    ).reset_index(drop=True)

    events["bootstrap_event_index"] = np.arange(
        len(events),
        dtype=int,
    )

    hypothesis_rows: list[dict[str, Any]] = []

    # At end event j, construct hypotheses only from observations
    # available up to j. This keeps the preflight causal.
    for end_index in range(1, len(events)):
        prefix = events.iloc[: end_index + 1].copy()

        source_points = prefix[
            ["visual_x_px", "visual_y_px"]
        ].to_numpy(dtype=float)

        target_points = prefix[
            [
                "chosen_abs_x_traj01_m",
                "chosen_abs_y_traj01_m",
            ]
        ].to_numpy(dtype=float)

        for start_index in range(end_index):
            source_delta = (
                source_points[end_index]
                - source_points[start_index]
            )

            target_delta = (
                target_points[end_index]
                - target_points[start_index]
            )

            visual_separation_px = float(
                np.linalg.norm(source_delta)
            )

            map_separation_m = float(
                np.linalg.norm(target_delta)
            )

            if (
                visual_separation_px <= 1e-6
                or map_separation_m <= 1e-6
            ):
                continue

            scale_m_per_px = (
                map_separation_m
                / visual_separation_px
            )

            rotation_rad = (
                math.atan2(
                    target_delta[1],
                    target_delta[0],
                )
                - math.atan2(
                    source_delta[1],
                    source_delta[0],
                )
            )

            rotation = rotation_matrix(rotation_rad)

            translation = (
                target_points[start_index]
                - scale_m_per_px
                * (
                    rotation
                    @ source_points[start_index]
                )
            )

            predicted = (
                scale_m_per_px
                * (source_points @ rotation.T)
                + translation
            )

            residuals = np.linalg.norm(
                predicted - target_points,
                axis=1,
            )

            row: dict[str, Any] = {
                "available_event_count":
                    int(len(prefix)),
                "anchor_start_event_index":
                    int(start_index),
                "anchor_end_event_index":
                    int(end_index),
                "anchor_start_frame":
                    int(
                        prefix.iloc[start_index][
                            "sequence_frame_id"
                        ]
                    ),
                "anchor_end_frame":
                    int(
                        prefix.iloc[end_index][
                            "sequence_frame_id"
                        ]
                    ),
                "anchor_start_token":
                    int(prefix.iloc[start_index]["token"]),
                "anchor_end_token":
                    int(prefix.iloc[end_index]["token"]),
                "visual_separation_px":
                    visual_separation_px,
                "map_separation_m":
                    map_separation_m,
                "scale_m_per_px":
                    scale_m_per_px,
                "rotation_deg":
                    math.degrees(rotation_rad),
                "translation_x_m":
                    float(translation[0]),
                "translation_y_m":
                    float(translation[1]),
                "median_prefix_residual_m":
                    float(np.median(residuals)),
                "p95_prefix_residual_m":
                    float(np.quantile(residuals, 0.95)),
                "max_prefix_residual_m":
                    float(np.max(residuals)),
                # Evaluation-only annotations:
                "anchor_start_true_eval_only":
                    bool(
                        prefix.iloc[start_index][
                            "hit_eval_only"
                        ]
                    ),
                "anchor_end_true_eval_only":
                    bool(
                        prefix.iloc[end_index][
                            "hit_eval_only"
                        ]
                    ),
            }

            for threshold in RESIDUAL_THRESHOLDS_M:
                threshold_name = int(threshold)

                inlier_mask = residuals <= threshold

                row[
                    f"prefix_inliers_r{threshold_name}"
                ] = int(inlier_mask.sum())

                row[
                    f"prefix_inlier_rate_r{threshold_name}"
                ] = float(inlier_mask.mean())

                # Evaluation only, never used for hypothesis ranking.
                row[
                    f"prefix_true_inliers_r{threshold_name}_eval_only"
                ] = int(
                    (
                        inlier_mask
                        & bool_series(
                            prefix["hit_eval_only"]
                        ).to_numpy()
                    ).sum()
                )

            hypothesis_rows.append(row)

    hypotheses = pd.DataFrame(hypothesis_rows)

    if hypotheses.empty:
        raise ValueError(
            "No valid two-anchor bootstrap hypotheses"
        )

    winner_rows: list[pd.Series] = []

    for event_count, group in hypotheses.groupby(
        "available_event_count",
        sort=True,
    ):
        # Online-only ranking: maximize agreement with accepted
        # absolute observations, then minimize residual.
        winner = group.sort_values(
            [
                "prefix_inliers_r60",
                "median_prefix_residual_m",
                "map_separation_m",
            ],
            ascending=[False, True, False],
        ).iloc[0]

        winner_rows.append(winner)

    winners = pd.DataFrame(winner_rows).reset_index(
        drop=True
    )

    events_path = (
        OUTPUT_DIR
        / "s6b2a_balanced_bootstrap_events.csv"
    )

    hypotheses_path = (
        OUTPUT_DIR
        / "s6b2a_raw_bootstrap_hypotheses.csv"
    )

    winners_path = (
        OUTPUT_DIR
        / "s6b2a_raw_bootstrap_causal_winners.csv"
    )

    summary_path = (
        REPORT_DIR
        / "s6b2a_raw_bootstrap_preflight_summary.json"
    )

    events.to_csv(events_path, index=False)
    hypotheses.to_csv(hypotheses_path, index=False)
    winners.to_csv(winners_path, index=False)

    summary = {
        "stage": "S6B.2A",
        "relative_variant": VARIANT,
        "raw_frames": int(len(raw)),
        "balanced_absolute_observations": int(len(events)),
        "maximum_raw_vs_aligned_visual_copy_delta":
            maximum_visual_copy_delta,
        "hypotheses": int(len(hypotheses)),
        "causal_winners": int(len(winners)),
        "locked_rule": (
            "Hypotheses and rankings use only raw visual "
            "coordinates and confidence-gated absolute "
            "coordinates. Ground-truth labels are attached "
            "only after ranking for evaluation."
        ),
        "outputs": {
            "events": events_path,
            "hypotheses": hypotheses_path,
            "causal_winners": winners_path,
        },
    }

    summary_path.write_text(
        json.dumps(
            json_safe(summary),
            indent=2,
        ),
        encoding="utf-8",
    )

    print("S6B.2A Raw Bootstrap Preflight")
    print("------------------------------")
    print(f"Raw frames:                    {len(raw)}")
    print(f"Balanced observations:         {len(events)}")
    print(
        "Raw/aligned visual max delta:  "
        f"{maximum_visual_copy_delta:.12f}"
    )
    print(f"Two-anchor hypotheses:         {len(hypotheses)}")
    print(f"Causal winner states:          {len(winners)}")

    print()
    print("First balanced observations")
    print("---------------------------")

    print(
        events[
            [
                "bootstrap_event_index",
                "sequence_frame_id",
                "token",
                "reference_cumulative_distance_m",
                "visual_x_px",
                "visual_y_px",
                "chosen_abs_x_traj01_m",
                "chosen_abs_y_traj01_m",
                "hit_eval_only",
                "chosen_error_m_eval_only",
            ]
        ]
        .head(12)
        .to_string(index=False)
    )

    print()
    print("Early causal bootstrap winners")
    print("------------------------------")

    print(
        winners.loc[
            winners["available_event_count"] >= 3,
            [
                "available_event_count",
                "anchor_start_token",
                "anchor_end_token",
                "scale_m_per_px",
                "rotation_deg",
                "prefix_inliers_r60",
                "prefix_inlier_rate_r60",
                "median_prefix_residual_m",
                "p95_prefix_residual_m",
                "anchor_start_true_eval_only",
                "anchor_end_true_eval_only",
            ],
        ]
        .head(15)
        .to_string(index=False)
    )

    print()
    print(f"Events:      {events_path}")
    print(f"Hypotheses:  {hypotheses_path}")
    print(f"Winners:     {winners_path}")
    print(f"Summary:     {summary_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
