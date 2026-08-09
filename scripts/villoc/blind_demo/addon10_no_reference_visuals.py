'''
ROOT=outputs/villoc/traj01_90deg_stable120m
RUN="$ROOT/blind_demo_addons/eval_traj01_known_gt"

python scripts/villoc/blind_demo/addon10_no_reference_visuals.py \
  --relative-trajectory \
  "$RUN/trajectories/blind_map_aligned_relative_trajectory.csv" \
  --submission \
  "$RUN/trajectories/submission_estimated_trajectory.csv" \
  --temporal-manifest \
  "$RUN/metadata/blind_temporal_fusion/blind_temporal_correction_manifest.csv" \
  --tile-index \
  "outputs/villoc/90_deg/metadata/s8_9_satellite_tile_index_512_s256.csv" \
  --run-root "$RUN" \
  2>&1 | tee \
  "$RUN/logs/stage11b_addon10_no_reference_visuals.log"
'''

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import folium
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REQUIRED_BLIND_NOTE = (
    "Reference unavailable: accuracy metrics not computed."
)

ESTIMATED_LATLON_NOTE = (
    "estimated_lat/lon are visual map-matching outputs, "
    "not GPS inputs."
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def bool_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False)

    return (
        s.astype(str)
        .str.strip()
        .str.lower()
        .isin({
            "true",
            "1",
            "yes",
            "y",
            "t",
        })
    )


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, float):
        if not math.isfinite(value):
            return None

    if pd.isna(value):
        return None

    return value


def require_columns(
    df: pd.DataFrame,
    columns: list[str],
    label: str,
) -> None:

    missing = sorted(
        set(columns)
        - set(df.columns)
    )

    require(
        not missing,
        (
            f"{label} missing required columns: "
            f"{missing}"
        ),
    )


def reject_reference_columns(
    df: pd.DataFrame,
    label: str,
) -> list[str]:

    forbidden_tokens = [
        "eval_ref",
        "ground_truth",
        "groundtruth",
        "oracle",
        "hit_le_",
        "error_m_eval",
        "reference_x",
        "reference_y",
        "reference_lat",
        "reference_lon",
        "gt_x",
        "gt_y",
    ]

    leaked = [
        c for c in df.columns
        if any(
            token in c.lower()
            for token in forbidden_tokens
        )
    ]

    require(
        not leaked,
        (
            f"{label} contains forbidden "
            f"reference/evaluation columns: {leaked}"
        ),
    )

    return leaked



def save_visual_relative_xy(
    relative: pd.DataFrame,
    output_path: Path,
) -> dict[str, Any]:

    require_columns(
        relative,
        [
            "timestamp_s",
            "visual_x_px",
            "visual_y_px",
        ],
        "visual relative trajectory",
    )

    x = pd.to_numeric(
        relative[
            "visual_x_px"
        ],
        errors="coerce",
    )

    y = pd.to_numeric(
        relative[
            "visual_y_px"
        ],
        errors="coerce",
    )

    valid = (
        x.notna()
        & y.notna()
    )

    x = x.loc[
        valid
    ].to_numpy(float)

    y = y.loc[
        valid
    ].to_numpy(float)

    require(
        len(x) >= 2,
        (
            "Not enough visual-relative "
            "poses to plot."
        ),
    )

    xy = np.column_stack(
        [
            x,
            y,
        ]
    )

    step = np.linalg.norm(
        np.diff(
            xy,
            axis=0,
        ),
        axis=1,
    )

    path_length_px = float(
        step.sum()
    )

    fig, ax = plt.subplots(
        figsize=(9, 8)
    )

    ax.plot(
        x,
        y,
        linewidth=1.6,
        label=(
            "XFeat integrated "
            "visual trajectory"
        ),
    )

    ax.scatter(
        [x[0]],
        [y[0]],
        marker="o",
        s=70,
        label="Visual origin",
        zorder=4,
    )

    ax.scatter(
        [x[-1]],
        [y[-1]],
        marker="X",
        s=80,
        label="Final visual pose",
        zorder=4,
    )

    ax.set_xlabel(
        "Integrated visual X [px-scale]"
    )

    ax.set_ylabel(
        "Integrated visual Y [px-scale]"
    )

    ax.set_title(
        "Blind relative trajectory — "
        "visual coordinate frame"
    )

    ax.axis("equal")

    ax.grid(
        True,
        alpha=0.3,
    )

    ax.legend()

    fig.text(
        0.5,
        0.01,
        (
            REQUIRED_BLIND_NOTE
            + " No trusted absolute map lock: "
            "coordinates are visual pixel-scale, "
            "not metres or geographic position."
        ),
        ha="center",
        fontsize=8,
    )

    fig.tight_layout(
        rect=[
            0,
            0.04,
            1,
            1,
        ]
    )

    fig.savefig(
        output_path,
        dpi=180,
    )

    plt.close(fig)

    return {
        "poses_plotted": int(
            len(x)
        ),
        "coordinate_unit": (
            "integrated_visual_pixel_scale"
        ),
        "start_x_px": float(
            x[0]
        ),
        "start_y_px": float(
            y[0]
        ),
        "final_x_px": float(
            x[-1]
        ),
        "final_y_px": float(
            y[-1]
        ),
        "visual_path_length_px": (
            path_length_px
        ),
    }


def save_visual_motion_vs_time(
    relative: pd.DataFrame,
    step_output_path: Path,
    cumulative_output_path: Path,
) -> dict[str, Any]:

    require_columns(
        relative,
        [
            "timestamp_s",
            "visual_x_px",
            "visual_y_px",
        ],
        "visual relative trajectory",
    )

    time_s = pd.to_numeric(
        relative[
            "timestamp_s"
        ],
        errors="raise",
    ).to_numpy(float)

    x = pd.to_numeric(
        relative[
            "visual_x_px"
        ],
        errors="raise",
    ).to_numpy(float)

    y = pd.to_numeric(
        relative[
            "visual_y_px"
        ],
        errors="raise",
    ).to_numpy(float)

    require(
        np.isfinite(
            time_s
        ).all()
        and np.isfinite(
            x
        ).all()
        and np.isfinite(
            y
        ).all(),
        (
            "Visual-relative trajectory "
            "contains non-finite values."
        ),
    )

    if (
        "step_motion_px"
        in relative.columns
    ):
        step = pd.to_numeric(
            relative[
                "step_motion_px"
            ],
            errors="coerce",
        ).to_numpy(float)

        # The first relative row may legitimately
        # have no previous frame.
        if len(step):
            step[
                ~np.isfinite(step)
            ] = 0.0

    else:
        xy = np.column_stack(
            [
                x,
                y,
            ]
        )

        step = np.concatenate(
            [
                [0.0],
                np.linalg.norm(
                    np.diff(
                        xy,
                        axis=0,
                    ),
                    axis=1,
                ),
            ]
        )

    cumulative = np.cumsum(
        step
    )

    # ------------------------------------------------
    # Per-frame visual motion.
    # ------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(12, 5.5)
    )

    ax.plot(
        time_s,
        step,
        linewidth=1.25,
    )

    ax.set_xlabel(
        "Video time [s]"
    )

    ax.set_ylabel(
        "Relative step motion [px-scale]"
    )

    ax.set_title(
        "Blind relative visual motion vs time"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    fig.text(
        0.5,
        0.01,
        (
            "Visual motion only; this is not "
            "metric drift error. "
            + REQUIRED_BLIND_NOTE
        ),
        ha="center",
        fontsize=8,
    )

    fig.tight_layout(
        rect=[
            0,
            0.04,
            1,
            1,
        ]
    )

    fig.savefig(
        step_output_path,
        dpi=180,
    )

    plt.close(fig)

    # ------------------------------------------------
    # Accumulated visual motion.
    # ------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(12, 5.5)
    )

    ax.plot(
        time_s,
        cumulative,
        linewidth=1.35,
    )

    ax.set_xlabel(
        "Video time [s]"
    )

    ax.set_ylabel(
        "Cumulative visual motion [px-scale]"
    )

    ax.set_title(
        "Accumulated blind relative visual motion"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    fig.text(
        0.5,
        0.01,
        (
            "Accumulated visual motion is not "
            "accumulated localization drift/error. "
            "Drift error requires post-freeze "
            "reference evaluation. "
            + REQUIRED_BLIND_NOTE
        ),
        ha="center",
        fontsize=8,
    )

    fig.tight_layout(
        rect=[
            0,
            0.04,
            1,
            1,
        ]
    )

    fig.savefig(
        cumulative_output_path,
        dpi=180,
    )

    plt.close(fig)

    return {
        "rows": int(
            len(relative)
        ),
        "step_motion_median_px": float(
            np.median(
                step
            )
        ),
        "step_motion_p95_px": float(
            np.percentile(
                step,
                95,
            )
        ),
        "step_motion_max_px": float(
            np.max(
                step
            )
        ),
        "cumulative_visual_motion_px": float(
            cumulative[-1]
        ),
        "drift_error_computed": False,
    }


def save_no_lock_correction_timeline(
    temporal: pd.DataFrame,
    output_path: Path,
) -> dict[str, Any]:

    require_columns(
        temporal,
        [
            "timestamp_s",
            "correction_candidate",
            "correction_accepted",
            "correction_reason",
        ],
        "temporal manifest",
    )

    time_s = pd.to_numeric(
        temporal[
            "timestamp_s"
        ],
        errors="raise",
    ).to_numpy(float)

    candidate = bool_series(
        temporal[
            "correction_candidate"
        ]
    ).to_numpy(bool)

    accepted = bool_series(
        temporal[
            "correction_accepted"
        ]
    ).to_numpy(bool)

    require(
        not (
            accepted
            & ~candidate
        ).any(),
        (
            "Accepted correction exists "
            "outside candidate mask."
        ),
    )

    fig, ax = plt.subplots(
        figsize=(12, 4.8)
    )

    ax.plot(
        time_s,
        np.zeros_like(
            time_s
        ),
        linewidth=1.0,
        alpha=0.45,
    )

    if candidate.any():
        ax.scatter(
            time_s[
                candidate
            ],
            np.ones(
                int(
                    candidate.sum()
                )
            ),
            marker="x",
            s=40,
            label=(
                "Correction candidate"
            ),
        )

    if accepted.any():
        ax.scatter(
            time_s[
                accepted
            ],
            np.full(
                int(
                    accepted.sum()
                ),
                2.0,
            ),
            marker="D",
            s=50,
            label=(
                "Accepted correction"
            ),
        )

    if not candidate.any():
        ax.text(
            0.5,
            0.55,
            (
                "No temporal correction "
                "candidates — no trusted "
                "absolute map lock"
            ),
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=11,
        )

    ax.set_xlabel(
        "Video time [s]"
    )

    ax.set_ylabel(
        "Correction state"
    )

    ax.set_yticks(
        [
            0,
            1,
            2,
        ],
        labels=[
            "relative-only",
            "candidate",
            "accepted",
        ],
    )

    ax.set_ylim(
        -0.5,
        2.5,
    )

    ax.set_title(
        "Blind absolute-correction timeline"
    )

    ax.grid(
        True,
        alpha=0.25,
    )

    if (
        candidate.any()
        or accepted.any()
    ):
        ax.legend()

    fig.text(
        0.5,
        0.01,
        REQUIRED_BLIND_NOTE,
        ha="center",
        fontsize=8,
    )

    fig.tight_layout(
        rect=[
            0,
            0.04,
            1,
            1,
        ]
    )

    fig.savefig(
        output_path,
        dpi=180,
    )

    plt.close(fig)

    reason_counts = {
        str(k): int(v)
        for k, v in (
            temporal[
                "correction_reason"
            ]
            .fillna("UNKNOWN")
            .astype(str)
            .value_counts()
            .to_dict()
            .items()
        )
    }

    return {
        "candidate_corrections": int(
            candidate.sum()
        ),
        "accepted_corrections": int(
            accepted.sum()
        ),
        "reason_counts": (
            reason_counts
        ),
    }


def save_relative_xy(
    relative: pd.DataFrame,
    output_path: Path,
) -> dict[str, Any]:

    available = bool_series(
        relative[
            "map_aligned_available"
        ]
    )

    plot_df = relative.loc[
        available
    ].copy()

    x = pd.to_numeric(
        plot_df[
            "relative_x_m"
        ],
        errors="coerce",
    )

    y = pd.to_numeric(
        plot_df[
            "relative_y_m"
        ],
        errors="coerce",
    )

    valid = (
        x.notna()
        & y.notna()
    )

    x = x.loc[
        valid
    ].to_numpy(float)

    y = y.loc[
        valid
    ].to_numpy(float)

    require(
        len(x) >= 2,
        "Not enough relative XY poses to plot.",
    )

    fig, ax = plt.subplots(
        figsize=(9, 8)
    )

    ax.plot(
        x,
        y,
        linewidth=1.6,
        label="Blind relative trajectory",
    )

    ax.scatter(
        [x[0]],
        [y[0]],
        marker="o",
        s=70,
        label="Map-lock origin",
        zorder=4,
    )

    ax.scatter(
        [x[-1]],
        [y[-1]],
        marker="X",
        s=80,
        label="Final estimate",
        zorder=4,
    )

    ax.set_xlabel(
        "Relative X [m]"
    )

    ax.set_ylabel(
        "Relative Y [m]"
    )

    ax.set_title(
        "Blind map-aligned relative trajectory"
    )

    ax.axis("equal")
    ax.grid(
        True,
        alpha=0.3,
    )
    ax.legend()

    fig.text(
        0.5,
        0.01,
        REQUIRED_BLIND_NOTE,
        ha="center",
        fontsize=8,
    )

    fig.tight_layout(
        rect=[
            0,
            0.03,
            1,
            1,
        ]
    )

    fig.savefig(
        output_path,
        dpi=180,
    )

    plt.close(fig)

    return {
        "poses_plotted": int(
            len(x)
        ),
        "start_x_m": float(
            x[0]
        ),
        "start_y_m": float(
            y[0]
        ),
        "final_x_m": float(
            x[-1]
        ),
        "final_y_m": float(
            y[-1]
        ),
    }


def save_fused_xy(
    submission: pd.DataFrame,
    output_path: Path,
) -> dict[str, Any]:

    available = bool_series(
        submission[
            "map_aligned_available"
        ]
    )

    plot_df = submission.loc[
        available
    ].copy()

    plot_df[
        "estimated_map_x"
    ] = pd.to_numeric(
        plot_df[
            "estimated_map_x"
        ],
        errors="coerce",
    )

    plot_df[
        "estimated_map_y"
    ] = pd.to_numeric(
        plot_df[
            "estimated_map_y"
        ],
        errors="coerce",
    )

    plot_df = plot_df.dropna(
        subset=[
            "estimated_map_x",
            "estimated_map_y",
        ]
    ).copy()

    require(
        len(plot_df) >= 2,
        "Not enough fused map poses to plot.",
    )

    accepted = bool_series(
        plot_df[
            "accepted_correction"
        ]
    )

    lock = bool_series(
        plot_df[
            "map_lock_event"
        ]
    )

    fig, ax = plt.subplots(
        figsize=(9, 8)
    )

    ax.plot(
        plot_df[
            "estimated_map_x"
        ],
        plot_df[
            "estimated_map_y"
        ],
        linewidth=1.7,
        label="Blind fused estimate",
    )

    if lock.any():
        lock_df = plot_df.loc[
            lock
        ]

        ax.scatter(
            lock_df[
                "estimated_map_x"
            ],
            lock_df[
                "estimated_map_y"
            ],
            marker="o",
            s=80,
            label="Blind map lock",
            zorder=5,
        )

    if accepted.any():
        accepted_df = plot_df.loc[
            accepted
        ]

        ax.scatter(
            accepted_df[
                "estimated_map_x"
            ],
            accepted_df[
                "estimated_map_y"
            ],
            marker="D",
            s=45,
            label="Accepted corrections",
            zorder=5,
        )

    final = plot_df.iloc[
        -1
    ]

    ax.scatter(
        [
            float(
                final[
                    "estimated_map_x"
                ]
            )
        ],
        [
            float(
                final[
                    "estimated_map_y"
                ]
            )
        ],
        marker="X",
        s=90,
        label="Final estimate",
        zorder=6,
    )

    ax.set_xlabel(
        "Easting [m] — EPSG:3346"
    )

    ax.set_ylabel(
        "Northing [m] — EPSG:3346"
    )

    ax.set_title(
        "Blind fused map trajectory"
    )

    ax.ticklabel_format(
        style="plain",
        useOffset=False,
    )

    ax.axis("equal")

    ax.grid(
        True,
        alpha=0.3,
    )

    ax.legend()

    fig.text(
        0.5,
        0.01,
        (
            REQUIRED_BLIND_NOTE
            + " "
            + ESTIMATED_LATLON_NOTE
        ),
        ha="center",
        fontsize=8,
    )

    fig.tight_layout(
        rect=[
            0,
            0.04,
            1,
            1,
        ]
    )

    fig.savefig(
        output_path,
        dpi=180,
    )

    plt.close(fig)

    return {
        "poses_plotted": int(
            len(plot_df)
        ),
        "accepted_corrections_plotted": int(
            accepted.sum()
        ),
        "map_lock_events_plotted": int(
            lock.sum()
        ),
    }


def save_confidence_vs_time(
    submission: pd.DataFrame,
    output_path: Path,
) -> dict[str, Any]:

    time_s = pd.to_numeric(
        submission[
            "timestamp_s"
        ],
        errors="raise",
    ).to_numpy(float)

    confidence = pd.to_numeric(
        submission[
            "confidence_score"
        ],
        errors="coerce",
    ).to_numpy(float)

    accepted = bool_series(
        submission[
            "accepted_correction"
        ]
    ).to_numpy(bool)

    lock = bool_series(
        submission[
            "map_lock_event"
        ]
    ).to_numpy(bool)

    finite = np.isfinite(
        confidence
    )

    require(
        bool(
            finite.any()
        ),
        "No finite confidence values available.",
    )

    fig, ax = plt.subplots(
        figsize=(12, 5.5)
    )

    ax.plot(
        time_s[
            finite
        ],
        confidence[
            finite
        ],
        linewidth=1.25,
        label=(
            "ORB/DINO hybrid confidence score"
        ),
    )

    if accepted.any():
        ax.scatter(
            time_s[
                accepted
            ],
            confidence[
                accepted
            ],
            marker="D",
            s=45,
            label="Accepted corrections",
            zorder=5,
        )

    if lock.any():
        ax.scatter(
            time_s[
                lock
            ],
            confidence[
                lock
            ],
            marker="o",
            s=75,
            label="Map lock",
            zorder=5,
        )

    ax.set_xlabel(
        "Video time [s]"
    )

    ax.set_ylabel(
        "Hybrid confidence score"
    )

    ax.set_title(
        "Blind absolute-localization confidence vs time"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    ax.legend()

    fig.text(
        0.5,
        0.01,
        (
            "Confidence score is a ranking/verifier "
            "score, not a calibrated probability. "
            + REQUIRED_BLIND_NOTE
        ),
        ha="center",
        fontsize=8,
    )

    fig.tight_layout(
        rect=[
            0,
            0.04,
            1,
            1,
        ]
    )

    fig.savefig(
        output_path,
        dpi=180,
    )

    plt.close(fig)

    finite_values = confidence[
        finite
    ]

    return {
        "finite_confidence_rows": int(
            finite.sum()
        ),
        "confidence_min": float(
            np.min(
                finite_values
            )
        ),
        "confidence_median": float(
            np.median(
                finite_values
            )
        ),
        "confidence_max": float(
            np.max(
                finite_values
            )
        ),
        "accepted_markers": int(
            accepted.sum()
        ),
    }


def save_correction_timeline(
    temporal: pd.DataFrame,
    output_path: Path,
) -> dict[str, Any]:

    time_s = pd.to_numeric(
        temporal[
            "timestamp_s"
        ],
        errors="raise",
    ).to_numpy(float)

    candidate = bool_series(
        temporal[
            "correction_candidate"
        ]
    ).to_numpy(bool)

    accepted = bool_series(
        temporal[
            "correction_accepted"
        ]
    ).to_numpy(bool)

    residual = pd.to_numeric(
        temporal[
            "temporal_residual_m"
        ],
        errors="coerce",
    ).to_numpy(float)

    threshold = pd.to_numeric(
        temporal[
            "temporal_threshold_m"
        ],
        errors="coerce",
    ).to_numpy(float)

    require(
        not bool(
            (
                accepted
                & ~candidate
            ).any()
        ),
        (
            "Accepted correction exists outside "
            "correction_candidate mask."
        ),
    )

    fig, ax = plt.subplots(
        figsize=(12, 6)
    )

    finite_residual = np.isfinite(
        residual
    )

    finite_threshold = np.isfinite(
        threshold
    )

    if finite_residual.any():
        ax.plot(
            time_s[
                finite_residual
            ],
            residual[
                finite_residual
            ],
            linewidth=1.15,
            label="Temporal residual",
        )

    if finite_threshold.any():
        ax.plot(
            time_s[
                finite_threshold
            ],
            threshold[
                finite_threshold
            ],
            linestyle="--",
            linewidth=1.15,
            label="Temporal acceptance threshold",
        )

    rejected_candidate = (
        candidate
        & ~accepted
    )

    if rejected_candidate.any():
        marker_y = residual[
            rejected_candidate
        ]

        marker_time = time_s[
            rejected_candidate
        ]

        finite_marker = np.isfinite(
            marker_y
        )

        ax.scatter(
            marker_time[
                finite_marker
            ],
            marker_y[
                finite_marker
            ],
            marker="x",
            s=40,
            label="Rejected candidate",
            zorder=5,
        )

    if accepted.any():
        accepted_y = residual[
            accepted
        ]

        accepted_time = time_s[
            accepted
        ]

        finite_acc = np.isfinite(
            accepted_y
        )

        ax.scatter(
            accepted_time[
                finite_acc
            ],
            accepted_y[
                finite_acc
            ],
            marker="D",
            s=50,
            label="Accepted correction",
            zorder=6,
        )

    ax.set_xlabel(
        "Video time [s]"
    )

    ax.set_ylabel(
        "Temporal residual / threshold [m]"
    )

    ax.set_title(
        "Blind correction decisions over time"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    ax.legend()

    fig.text(
        0.5,
        0.01,
        REQUIRED_BLIND_NOTE,
        ha="center",
        fontsize=8,
    )

    fig.tight_layout(
        rect=[
            0,
            0.04,
            1,
            1,
        ]
    )

    fig.savefig(
        output_path,
        dpi=180,
    )

    plt.close(fig)

    reason_counts = {
        str(k): int(v)
        for k, v in (
            temporal.loc[
                candidate,
                "correction_reason",
            ]
            .fillna(
                "UNKNOWN"
            )
            .astype(str)
            .value_counts()
            .to_dict()
            .items()
        )
    }

    return {
        "candidate_corrections": int(
            candidate.sum()
        ),
        "accepted_corrections": int(
            accepted.sum()
        ),
        "rejected_candidates": int(
            (
                candidate
                & ~accepted
            ).sum()
        ),
        "candidate_reason_counts":
            reason_counts,
    }


def map_bounds_from_tile_index(
    tile_index: pd.DataFrame | None,
) -> list[list[float]] | None:

    if tile_index is None:
        return None

    required = [
        "top_left_lat",
        "top_left_lon",
        "bottom_right_lat",
        "bottom_right_lon",
    ]

    if any(
        c not in tile_index.columns
        for c in required
    ):
        return None

    lat_values = pd.concat([
        pd.to_numeric(
            tile_index[
                "top_left_lat"
            ],
            errors="coerce",
        ),
        pd.to_numeric(
            tile_index[
                "bottom_right_lat"
            ],
            errors="coerce",
        ),
    ]).dropna()

    lon_values = pd.concat([
        pd.to_numeric(
            tile_index[
                "top_left_lon"
            ],
            errors="coerce",
        ),
        pd.to_numeric(
            tile_index[
                "bottom_right_lon"
            ],
            errors="coerce",
        ),
    ]).dropna()

    if (
        len(
            lat_values
        ) == 0
        or len(
            lon_values
        ) == 0
    ):
        return None

    return [
        [
            float(
                lat_values.min()
            ),
            float(
                lon_values.min()
            ),
        ],
        [
            float(
                lat_values.max()
            ),
            float(
                lon_values.max()
            ),
        ],
    ]


def save_folium_map(
    submission: pd.DataFrame,
    tile_index: pd.DataFrame | None,
    output_path: Path,
) -> dict[str, Any]:

    available = bool_series(
        submission[
            "map_aligned_available"
        ]
    )

    lat = pd.to_numeric(
        submission[
            "estimated_lat"
        ],
        errors="coerce",
    )

    lon = pd.to_numeric(
        submission[
            "estimated_lon"
        ],
        errors="coerce",
    )

    valid = (
        available
        & lat.notna()
        & lon.notna()
    )

    map_df = submission.loc[
        valid
    ].copy()

    require(
        len(map_df) >= 2,
        (
            "Need at least two estimated lat/lon "
            "poses for Folium map."
        ),
    )

    lat_values = pd.to_numeric(
        map_df[
            "estimated_lat"
        ],
        errors="raise",
    ).to_numpy(float)

    lon_values = pd.to_numeric(
        map_df[
            "estimated_lon"
        ],
        errors="raise",
    ).to_numpy(float)

    center = [
        float(
            np.median(
                lat_values
            )
        ),
        float(
            np.median(
                lon_values
            )
        ),
    ]

    fmap = folium.Map(
        location=center,
        zoom_start=16,
        control_scale=True,
        tiles="OpenStreetMap",
    )

    folium.PolyLine(
        list(
            zip(
                lat_values.tolist(),
                lon_values.tolist(),
            )
        ),
        weight=4,
        opacity=0.85,
        tooltip=(
            "Blind fused visual localization trajectory"
        ),
    ).add_to(
        fmap
    )

    lock = bool_series(
        map_df[
            "map_lock_event"
        ]
    )

    if lock.any():
        lock_row = map_df.loc[
            lock
        ].iloc[
            0
        ]

        folium.Marker(
            location=[
                float(
                    lock_row[
                        "estimated_lat"
                    ]
                ),
                float(
                    lock_row[
                        "estimated_lon"
                    ]
                ),
            ],
            tooltip="Blind map lock",
            popup=(
                "Blind map lock<br>"
                f"frame={int(lock_row['sequence_frame_id'])}<br>"
                f"time={float(lock_row['timestamp_s']):.3f}s"
            ),
        ).add_to(
            fmap
        )

    accepted = bool_series(
        map_df[
            "accepted_correction"
        ]
    )

    accepted_df = map_df.loc[
        accepted
    ].copy()

    for _, row in accepted_df.iterrows():

        folium.CircleMarker(
            location=[
                float(
                    row[
                        "estimated_lat"
                    ]
                ),
                float(
                    row[
                        "estimated_lon"
                    ]
                ),
            ],
            radius=4,
            fill=True,
            fill_opacity=0.85,
            tooltip=(
                "Accepted visual correction"
            ),
            popup=(
                f"frame={int(row['sequence_frame_id'])}<br>"
                f"time={float(row['timestamp_s']):.3f}s<br>"
                f"source={row['correction_source']}<br>"
                f"tile={row['orb_selected_tile_id']}<br>"
                f"score={float(row['orb_score']):.3f}<br>"
                f"inliers={int(row['orb_inliers'])}"
            ),
        ).add_to(
            fmap
        )

    final = map_df.iloc[
        -1
    ]

    folium.Marker(
        location=[
            float(
                final[
                    "estimated_lat"
                ]
            ),
            float(
                final[
                    "estimated_lon"
                ]
            ),
        ],
        tooltip="Final blind estimate",
        popup=(
            "Final blind estimate<br>"
            f"frame={int(final['sequence_frame_id'])}<br>"
            f"time={float(final['timestamp_s']):.3f}s"
        ),
    ).add_to(
        fmap
    )

    tile_bounds = map_bounds_from_tile_index(
        tile_index
    )

    if tile_bounds is not None:

        folium.Rectangle(
            bounds=tile_bounds,
            weight=2,
            fill=False,
            tooltip="Prepared visual map/tile coverage",
        ).add_to(
            fmap
        )

    trajectory_bounds = [
        [
            float(
                np.min(
                    lat_values
                )
            ),
            float(
                np.min(
                    lon_values
                )
            ),
        ],
        [
            float(
                np.max(
                    lat_values
                )
            ),
            float(
                np.max(
                    lon_values
                )
            ),
        ],
    ]

    fmap.fit_bounds(
        trajectory_bounds,
        padding=(
            20,
            20,
        ),
    )

    note_html = f"""
    <div style="
        position: fixed;
        bottom: 18px;
        left: 18px;
        z-index: 9999;
        background: white;
        padding: 10px 12px;
        border: 1px solid #777;
        border-radius: 5px;
        font-size: 12px;
        max-width: 430px;
    ">
      <b>Blind visual localization estimate</b><br>
      {REQUIRED_BLIND_NOTE}<br>
      {ESTIMATED_LATLON_NOTE}
    </div>
    """

    fmap.get_root().html.add_child(
        folium.Element(
            note_html
        )
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fmap.save(
        str(
            output_path
        )
    )

    return {
        "estimated_latlon_poses": int(
            len(map_df)
        ),
        "accepted_corrections_mapped": int(
            len(
                accepted_df
            )
        ),
        "map_lock_markers": int(
            lock.sum()
        ),
        "tile_coverage_rectangle":
            bool(
                tile_bounds
                is not None
            ),
        "trajectory_bounds": (
            trajectory_bounds
        ),
    }


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Add-on 10: generate no-reference-safe "
            "blind trajectory plots and Folium map."
        )
    )

    parser.add_argument(
        "--relative-trajectory",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--submission",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--temporal-manifest",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--run-root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--tile-index",
        type=Path,
        default=None,
        help=(
            "Optional blind-safe prepared map tile index. "
            "Used only for map coverage context."
        ),
    )

    args = parser.parse_args()

    started = time.perf_counter()

    relative_path = (
        args.relative_trajectory
        .expanduser()
        .resolve()
    )

    submission_path = (
        args.submission
        .expanduser()
        .resolve()
    )

    temporal_path = (
        args.temporal_manifest
        .expanduser()
        .resolve()
    )

    run_root = (
        args.run_root
        .expanduser()
        .resolve()
    )

    tile_index_path = (
        args.tile_index
        .expanduser()
        .resolve()
        if args.tile_index
        is not None
        else None
    )

    for path in [
        relative_path,
        submission_path,
        temporal_path,
    ]:
        require(
            path.exists(),
            f"Missing input: {path}",
        )

    if tile_index_path is not None:
        require(
            tile_index_path.exists(),
            (
                "Tile index does not exist: "
                f"{tile_index_path}"
            ),
        )

    relative_sha_before = sha256_file(
        relative_path
    )

    submission_sha_before = sha256_file(
        submission_path
    )

    temporal_sha_before = sha256_file(
        temporal_path
    )

    relative = pd.read_csv(
        relative_path
    )

    submission = pd.read_csv(
        submission_path
    )

    temporal = pd.read_csv(
        temporal_path
    )

    tile_index = (
        pd.read_csv(
            tile_index_path
        )
        if tile_index_path
        is not None
        else None
    )

    require(
        len(relative) > 0,
        "Relative trajectory is empty.",
    )

    require(
        len(submission) > 0,
        "Submission is empty.",
    )

    require(
        len(temporal) > 0,
        "Temporal manifest is empty.",
    )

    require(
        len(relative)
        == len(submission)
        == len(temporal),
        (
            "Blind visualization input "
            "row counts differ: "
            f"relative={len(relative)}, "
            f"submission={len(submission)}, "
            f"temporal={len(temporal)}."
        ),
    )

    # =====================================================
    # Relative-only visualization state.
    #
    # No trusted map lock means:
    # - visual-relative trajectory CAN be plotted;
    # - retrieval/verifier confidence CAN be plotted;
    # - correction availability CAN be shown;
    # - metric fused XY CANNOT be plotted;
    # - geographic Folium trajectory CANNOT be created.
    # =====================================================

    no_map_lock = bool(
        "localization_state"
        in submission.columns
        and submission[
            "localization_state"
        ]
        .astype(str)
        .eq(
            "NO_TRUSTED_ABSOLUTE_LOCK"
        )
        .all()
    )

    if no_map_lock:

        require_columns(
            relative,
            [
                "sequence_frame_id",
                "timestamp_s",
                "visual_x_px",
                "visual_y_px",
                "map_aligned_available",
                "reference_used",
            ],
            "relative trajectory",
        )

        require_columns(
            submission,
            [
                "frame_index",
                "sequence_frame_id",
                "query_id",
                "timestamp_s",
                "visual_x_px",
                "visual_y_px",
                "estimated_map_x",
                "estimated_map_y",
                "estimated_lat",
                "estimated_lon",
                "map_aligned_available",
                "map_lock_event",
                "localization_state",
                "confidence_score",
                "accepted_correction",
                "orb_selected_tile_id",
                "orb_score",
                "orb_inliers",
            ],
            "submission",
        )

        require_columns(
            temporal,
            [
                "sequence_frame_id",
                "query_id",
                "timestamp_s",
                "correction_candidate",
                "correction_accepted",
                "correction_reason",
            ],
            "temporal manifest",
        )

        reject_reference_columns(
            relative,
            "relative trajectory",
        )

        reject_reference_columns(
            submission,
            "submission",
        )

        reject_reference_columns(
            temporal,
            "temporal manifest",
        )

        require(
            not bool_series(
                relative[
                    "reference_used"
                ]
            ).any(),
            (
                "Relative trajectory reports "
                "reference_used=True."
            ),
        )

        require(
            not bool_series(
                submission[
                    "map_aligned_available"
                ]
            ).any(),
            (
                "No-lock submission contains "
                "map-aligned rows."
            ),
        )

        for col in [
            "estimated_map_x",
            "estimated_map_y",
            "estimated_lat",
            "estimated_lon",
        ]:
            require(
                pd.to_numeric(
                    submission[col],
                    errors="coerce",
                )
                .isna()
                .all(),
                (
                    "No-lock submission contains "
                    f"unexpected values in {col}."
                ),
            )

        if tile_index is not None:
            reject_reference_columns(
                tile_index,
                "tile index",
            )

        figures_dir = (
            run_root
            / "figures"
        )

        maps_dir = (
            run_root
            / "maps"
        )

        reports_dir = (
            run_root
            / "reports"
            / "addon10_no_reference_visuals"
        )

        figures_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        maps_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        reports_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        relative_xy_path = (
            figures_dir
            / "estimated_relative_xy.png"
        )

        step_motion_path = (
            figures_dir
            / "visual_step_motion_vs_time.png"
        )

        cumulative_motion_path = (
            figures_dir
            / "visual_cumulative_motion_vs_time.png"
        )

        confidence_path = (
            figures_dir
            / "confidence_vs_time.png"
        )

        timeline_path = (
            figures_dir
            / "accepted_corrections_timeline.png"
        )

        # These outputs are intentionally unavailable.
        fused_xy_path = (
            figures_dir
            / "estimated_fused_xy.png"
        )

        map_path = (
            maps_dir
            / "estimated_fused_map.html"
        )

        # Remove stale outputs from a previous execution
        # so a no-lock run can never accidentally expose
        # an old geographic/fused result.
        fused_xy_path.unlink(
            missing_ok=True
        )

        map_path.unlink(
            missing_ok=True
        )

        report_path = (
            reports_dir
            / "no_reference_visuals_report.json"
        )

        plot_started = (
            time.perf_counter()
        )

        relative_info = (
            save_visual_relative_xy(
                relative,
                relative_xy_path,
            )
        )

        motion_info = (
            save_visual_motion_vs_time(
                relative,
                step_motion_path,
                cumulative_motion_path,
            )
        )

        confidence_info = (
            save_confidence_vs_time(
                submission,
                confidence_path,
            )
        )

        timeline_info = (
            save_no_lock_correction_timeline(
                temporal,
                timeline_path,
            )
        )

        plot_generation_s = (
            time.perf_counter()
            - plot_started
        )

        relative_sha_after = sha256_file(
            relative_path
        )

        submission_sha_after = sha256_file(
            submission_path
        )

        temporal_sha_after = sha256_file(
            temporal_path
        )

        inputs_unchanged = bool(
            relative_sha_before
            == relative_sha_after
            and submission_sha_before
            == submission_sha_after
            and temporal_sha_before
            == temporal_sha_after
        )

        require(
            inputs_unchanged,
            (
                "Blind visualization inputs "
                "changed during generation."
            ),
        )

        generated_outputs = [
            relative_xy_path,
            step_motion_path,
            cumulative_motion_path,
            confidence_path,
            timeline_path,
        ]

        for output in generated_outputs:
            require(
                output.exists(),
                (
                    "Expected visual output "
                    f"missing: {output}"
                ),
            )

            require(
                output.stat().st_size > 0,
                (
                    "Visual output is empty: "
                    f"{output}"
                ),
            )

        require(
            not fused_xy_path.exists(),
            (
                "No-lock run unexpectedly "
                "generated fused XY."
            ),
        )

        require(
            not map_path.exists(),
            (
                "No-lock run unexpectedly "
                "generated a Folium map."
            ),
        )

        total_stage_wall_s = (
            time.perf_counter()
            - started
        )

        report = {
            "stage": (
                "STAGE_11_ADDON10"
            ),
            "addon": (
                "no_reference_safe_"
                "plots_and_maps"
            ),
            "status": (
                "PASS_ADDON10_RELATIVE_ONLY_"
                "VISUALS_NO_MAP_LOCK"
            ),
            "localization_state": (
                "NO_TRUSTED_ABSOLUTE_LOCK"
            ),
            "visualization_state": (
                "RELATIVE_VISUAL_ONLY"
            ),
            "blind_contract": {
                "reference_argument_supported":
                    False,
                "reference_used":
                    False,
                "srt_used":
                    False,
                "gps_used":
                    False,
                "ground_truth_used":
                    False,
                "evaluation_error_used":
                    False,
                "drift_error_computed":
                    False,
                "required_note":
                    REQUIRED_BLIND_NOTE,
            },
            "availability": {
                "relative_visual_xy":
                    True,
                "visual_motion_vs_time":
                    True,
                "absolute_confidence":
                    True,
                "metric_relative_xy":
                    False,
                "fused_map_xy":
                    False,
                "estimated_latlon":
                    False,
                "folium_geographic_map":
                    False,
                "drift_error":
                    False,
            },
            "relative_xy":
                relative_info,
            "visual_motion":
                motion_info,
            "confidence_vs_time":
                confidence_info,
            "correction_timeline":
                timeline_info,
            "folium_map": {
                "generated": False,
                "reason": (
                    "No trusted absolute map "
                    "lock / estimated lat-lon."
                ),
            },
            "fused_xy": {
                "generated": False,
                "reason": (
                    "No trusted map-aligned "
                    "trajectory."
                ),
            },
            "timing": {
                "plot_generation_s": float(
                    plot_generation_s
                ),
                "folium_html_map_generation_s":
                    0.0,
                "total_stage_wall_s": float(
                    total_stage_wall_s
                ),
                "plot_count": 5,
            },
            "outputs": {
                "estimated_relative_xy":
                    str(
                        relative_xy_path
                    ),
                "visual_step_motion_vs_time":
                    str(
                        step_motion_path
                    ),
                "visual_cumulative_motion_vs_time":
                    str(
                        cumulative_motion_path
                    ),
                "confidence_vs_time":
                    str(
                        confidence_path
                    ),
                "accepted_corrections_timeline":
                    str(
                        timeline_path
                    ),
                "estimated_fused_xy":
                    None,
                "estimated_fused_map":
                    None,
                "report":
                    str(
                        report_path
                    ),
            },
            "important_note": (
                "The displayed trajectory is "
                "reference-free visual-relative "
                "motion in integrated pixel-scale "
                "coordinates. It is not a metric "
                "or geographic estimate. Actual "
                "drift/error and geographic "
                "evaluation require the post-freeze "
                "reference attachment."
            ),
        }

        report_path.write_text(
            json.dumps(
                report,
                indent=2,
                default=json_safe,
                allow_nan=False,
            ),
            encoding="utf-8",
        )

        print("=" * 88)
        print(
            "STAGE 11B — ADD-ON 10 "
            "RELATIVE-ONLY VISUALS"
        )
        print("=" * 88)

        print()
        print("Localization")
        print("-" * 88)

        print(
            "state                 : "
            "NO_TRUSTED_ABSOLUTE_LOCK"
        )

        print(
            "relative visual XY    : available"
        )

        print(
            "metric/fused XY       : unavailable"
        )

        print(
            "estimated lat/lon     : unavailable"
        )

        print(
            "Folium geographic map : skipped"
        )

        print(
            "drift error           : not computed"
        )

        print()
        print("Relative trajectory")
        print("-" * 88)

        print(
            "poses                 :",
            relative_info[
                "poses_plotted"
            ],
        )

        print(
            "visual path length    :",
            (
                f"{relative_info['visual_path_length_px']:.3f} "
                "px-scale"
            ),
        )

        print(
            "cumulative motion     :",
            (
                f"{motion_info['cumulative_visual_motion_px']:.3f} "
                "px-scale"
            ),
        )

        print()
        print("Absolute evidence")
        print("-" * 88)

        print(
            "confidence rows       :",
            confidence_info[
                "finite_confidence_rows"
            ],
        )

        print(
            "correction candidates :",
            timeline_info[
                "candidate_corrections"
            ],
        )

        print(
            "accepted corrections  :",
            timeline_info[
                "accepted_corrections"
            ],
        )

        print()
        print("Saved outputs")
        print("-" * 88)

        for output in generated_outputs:
            print(output)

        print(report_path)

        print()
        print(REQUIRED_BLIND_NOTE)

        print(
            "No trusted map lock: "
            "Folium/geographic trajectory "
            "intentionally not generated."
        )

        print()
        print(
            "status: "
            "PASS_ADDON10_RELATIVE_ONLY_"
            "VISUALS_NO_MAP_LOCK"
        )

        return

    require_columns(
        relative,
        [
            "sequence_frame_id",
            "timestamp_s",
            "relative_x_m",
            "relative_y_m",
            "estimated_map_x",
            "estimated_map_y",
            "map_aligned_available",
            "reference_used",
        ],
        "relative trajectory",
    )

    require_columns(
        submission,
        [
            "frame_index",
            "sequence_frame_id",
            "query_id",
            "timestamp_s",
            "relative_x_m",
            "relative_y_m",
            "estimated_map_x",
            "estimated_map_y",
            "estimated_lat",
            "estimated_lon",
            "map_aligned_available",
            "map_lock_event",
            "confidence_score",
            "accepted_correction",
            "correction_source",
            "orb_selected_tile_id",
            "orb_score",
            "orb_inliers",
        ],
        "submission",
    )

    require_columns(
        temporal,
        [
            "sequence_frame_id",
            "query_id",
            "timestamp_s",
            "correction_candidate",
            "correction_accepted",
            "temporal_residual_m",
            "temporal_threshold_m",
            "distance_since_anchor_m",
            "correction_reason",
        ],
        "temporal manifest",
    )

    reject_reference_columns(
        relative,
        "relative trajectory",
    )

    reject_reference_columns(
        submission,
        "submission",
    )

    reject_reference_columns(
        temporal,
        "temporal manifest",
    )

    require(
        not bool_series(
            relative[
                "reference_used"
            ]
        ).any(),
        (
            "relative trajectory reports "
            "reference_used=True."
        ),
    )

    if tile_index is not None:
        reject_reference_columns(
            tile_index,
            "tile index",
        )

    figures_dir = (
        run_root
        / "figures"
    )

    maps_dir = (
        run_root
        / "maps"
    )

    reports_dir = (
        run_root
        / "reports"
        / "addon10_no_reference_visuals"
    )

    figures_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    maps_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    reports_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    relative_xy_path = (
        figures_dir
        / "estimated_relative_xy.png"
    )

    fused_xy_path = (
        figures_dir
        / "estimated_fused_xy.png"
    )

    confidence_path = (
        figures_dir
        / "confidence_vs_time.png"
    )

    timeline_path = (
        figures_dir
        / "accepted_corrections_timeline.png"
    )

    map_path = (
        maps_dir
        / "estimated_fused_map.html"
    )

    report_path = (
        reports_dir
        / "no_reference_visuals_report.json"
    )

    # =====================================================
    # Static plot generation timing.
    # =====================================================

    plot_started = time.perf_counter()

    relative_info = save_relative_xy(
        relative,
        relative_xy_path,
    )

    fused_info = save_fused_xy(
        submission,
        fused_xy_path,
    )

    confidence_info = save_confidence_vs_time(
        submission,
        confidence_path,
    )

    timeline_info = save_correction_timeline(
        temporal,
        timeline_path,
    )

    plot_generation_s = (
        time.perf_counter()
        - plot_started
    )

    # =====================================================
    # Folium timing kept separate for Add-on 4 registry.
    # =====================================================

    map_started = time.perf_counter()

    map_info = save_folium_map(
        submission,
        tile_index,
        map_path,
    )

    folium_map_generation_s = (
        time.perf_counter()
        - map_started
    )

    # =====================================================
    # Input immutability check.
    # =====================================================

    relative_sha_after = sha256_file(
        relative_path
    )

    submission_sha_after = sha256_file(
        submission_path
    )

    temporal_sha_after = sha256_file(
        temporal_path
    )

    inputs_unchanged = bool(
        relative_sha_before
        == relative_sha_after
        and submission_sha_before
        == submission_sha_after
        and temporal_sha_before
        == temporal_sha_after
    )

    require(
        inputs_unchanged,
        (
            "One or more blind input files changed "
            "during visualization generation."
        ),
    )

    total_stage_wall_s = (
        time.perf_counter()
        - started
    )

    outputs = [
        relative_xy_path,
        fused_xy_path,
        confidence_path,
        timeline_path,
        map_path,
    ]

    for path in outputs:
        require(
            path.exists(),
            (
                "Expected output missing: "
                f"{path}"
            ),
        )

        require(
            path.stat().st_size > 0,
            (
                "Output is empty: "
                f"{path}"
            ),
        )

    report = {
        "stage": "STAGE_11_ADDON10",
        "addon": (
            "no_reference_safe_plots_and_maps"
        ),
        "status": (
            "PASS_ADDON10_NO_REFERENCE_VISUALS"
        ),
        "blind_contract": {
            "reference_argument_supported":
                False,
            "reference_used":
                False,
            "srt_used":
                False,
            "gps_used":
                False,
            "ground_truth_used":
                False,
            "evaluation_error_used":
                False,
            "required_note":
                REQUIRED_BLIND_NOTE,
            "estimated_latlon_note":
                ESTIMATED_LATLON_NOTE,
        },
        "inputs": {
            "relative_trajectory":
                str(
                    relative_path
                ),
            "submission":
                str(
                    submission_path
                ),
            "temporal_manifest":
                str(
                    temporal_path
                ),
            "tile_index": (
                str(
                    tile_index_path
                )
                if tile_index_path
                is not None
                else None
            ),
        },
        "input_sha256": {
            "relative_trajectory":
                relative_sha_before,
            "submission":
                submission_sha_before,
            "temporal_manifest":
                temporal_sha_before,
        },
        "input_files_unchanged":
            inputs_unchanged,
        "relative_xy":
            relative_info,
        "fused_xy":
            fused_info,
        "confidence_vs_time":
            confidence_info,
        "correction_timeline":
            timeline_info,
        "folium_map":
            map_info,
        "timing": {
            "plot_generation_s":
                float(
                    plot_generation_s
                ),
            "folium_html_map_generation_s":
                float(
                    folium_map_generation_s
                ),
            "total_stage_wall_s":
                float(
                    total_stage_wall_s
                ),
            "plot_count": 4,
            "plot_generation_ms_per_plot":
                float(
                    1000.0
                    * plot_generation_s
                    / 4.0
                ),
        },
        "outputs": {
            "estimated_relative_xy":
                str(
                    relative_xy_path
                ),
            "estimated_fused_xy":
                str(
                    fused_xy_path
                ),
            "confidence_vs_time":
                str(
                    confidence_path
                ),
            "accepted_corrections_timeline":
                str(
                    timeline_path
                ),
            "estimated_fused_map":
                str(
                    map_path
                ),
            "report":
                str(
                    report_path
                ),
        },
    }

    report_path.write_text(
        json.dumps(
            report,
            indent=2,
            default=json_safe,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    print("=" * 88)
    print(
        "STAGE 11B — ADD-ON 10 "
        "NO-REFERENCE-SAFE VISUALS"
    )
    print("=" * 88)

    print()
    print("Blind contract")
    print("-" * 88)

    print(
        "reference used       : false"
    )

    print(
        "SRT/GPS used         : false"
    )

    print(
        "evaluation errors    : false"
    )

    print(
        "inputs unchanged     :",
        inputs_unchanged,
    )

    print()
    print("Blind trajectory")
    print("-" * 88)

    print(
        "relative poses       :",
        relative_info[
            "poses_plotted"
        ],
    )

    print(
        "fused map poses      :",
        fused_info[
            "poses_plotted"
        ],
    )

    print(
        "map lock events      :",
        fused_info[
            "map_lock_events_plotted"
        ],
    )

    print(
        "accepted corrections :",
        fused_info[
            "accepted_corrections_plotted"
        ],
    )

    print()
    print("Confidence")
    print("-" * 88)

    print(
        "finite rows          :",
        confidence_info[
            "finite_confidence_rows"
        ],
    )

    print(
        "min / median / max   :",
        (
            f"{confidence_info['confidence_min']:.3f} / "
            f"{confidence_info['confidence_median']:.3f} / "
            f"{confidence_info['confidence_max']:.3f}"
        ),
    )

    print()
    print("Correction timeline")
    print("-" * 88)

    print(
        "candidates           :",
        timeline_info[
            "candidate_corrections"
        ],
    )

    print(
        "accepted             :",
        timeline_info[
            "accepted_corrections"
        ],
    )

    print(
        "rejected             :",
        timeline_info[
            "rejected_candidates"
        ],
    )

    print(
        "reasons              :",
        timeline_info[
            "candidate_reason_counts"
        ],
    )

    print()
    print("Folium map")
    print("-" * 88)

    print(
        "lat/lon poses        :",
        map_info[
            "estimated_latlon_poses"
        ],
    )

    print(
        "accepted markers     :",
        map_info[
            "accepted_corrections_mapped"
        ],
    )

    print(
        "tile coverage shown  :",
        map_info[
            "tile_coverage_rectangle"
        ],
    )

    print()
    print("Runtime")
    print("-" * 88)

    print(
        "plot generation      :",
        f"{plot_generation_s:.6f} s",
    )

    print(
        "Folium HTML map      :",
        f"{folium_map_generation_s:.6f} s",
    )

    print(
        "total stage wall     :",
        f"{total_stage_wall_s:.6f} s",
    )

    print()
    print(REQUIRED_BLIND_NOTE)

    print(
        ESTIMATED_LATLON_NOTE
    )

    print()
    print("Saved outputs")
    print("-" * 88)

    for path in outputs:
        print(path)

    print(report_path)

    print()
    print(
        "status: "
        "PASS_ADDON10_NO_REFERENCE_VISUALS"
    )


if __name__ == "__main__":
    main()
