#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


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


def metrics(errors: np.ndarray) -> dict[str, float | None]:
    x = np.asarray(
        errors,
        dtype=float,
    )

    x = x[
        np.isfinite(x)
    ]

    if len(x) == 0:
        return {
            "count": 0,
            "rmse_m": None,
            "mean_m": None,
            "median_m": None,
            "p95_m": None,
            "max_m": None,
        }

    return {
        "count": int(len(x)),
        "rmse_m": float(
            math.sqrt(
                np.mean(
                    x * x
                )
            )
        ),
        "mean_m": float(
            np.mean(x)
        ),
        "median_m": float(
            np.median(x)
        ),
        "p95_m": float(
            np.percentile(
                x,
                95,
            )
        ),
        "max_m": float(
            np.max(x)
        ),
    }



# ADDON1_INTEGRATED_DRIFT_TIME


def cumulative_distance_xy(
    x: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    xy = np.column_stack([
        np.asarray(x, dtype=float),
        np.asarray(y, dtype=float),
    ])

    if len(xy) == 0:
        return np.asarray([], dtype=float)

    if len(xy) == 1:
        return np.asarray([0.0], dtype=float)

    step = np.linalg.norm(
        np.diff(
            xy,
            axis=0,
        ),
        axis=1,
    )

    return np.concatenate([
        [0.0],
        np.cumsum(step),
    ])


def first_sustained_crossing(
    values: np.ndarray,
    threshold_m: float,
    frame_ids: np.ndarray,
    timestamps_s: np.ndarray,
    elapsed_s: np.ndarray,
    distances_m: np.ndarray,
    sustain_frames: int,
) -> dict[str, Any]:

    values = np.asarray(
        values,
        dtype=float,
    )

    above = (
        np.isfinite(values)
        & (values >= threshold_m)
    )

    if sustain_frames <= 0:
        raise ValueError(
            "sustain_frames must be >= 1"
        )

    for i in range(
        0,
        len(values) - sustain_frames + 1,
    ):
        if bool(
            np.all(
                above[
                    i:i + sustain_frames
                ]
            )
        ):
            return {
                "threshold_m": float(
                    threshold_m
                ),
                "crossed": True,
                "sequence_frame_id": int(
                    frame_ids[i]
                ),
                "timestamp_s": float(
                    timestamps_s[i]
                ),
                "elapsed_since_map_lock_s": float(
                    elapsed_s[i]
                ),
                "distance_since_map_lock_m": float(
                    distances_m[i]
                ),
                "value_at_crossing_m": float(
                    values[i]
                ),
                "sustain_frames": int(
                    sustain_frames
                ),
            }

    return {
        "threshold_m": float(
            threshold_m
        ),
        "crossed": False,
        "sequence_frame_id": None,
        "timestamp_s": None,
        "elapsed_since_map_lock_s": None,
        "distance_since_map_lock_m": None,
        "value_at_crossing_m": None,
        "sustain_frames": int(
            sustain_frames
        ),
    }



# ADDON2_ADDON3_INTEGRATED_EVALUATION


def parse_thresholds_m(
    value: str,
) -> list[float]:

    out = []

    for item in value.split(","):
        item = item.strip()

        if not item:
            continue

        out.append(
            float(item)
        )

    if not out:
        raise ValueError(
            "At least one threshold is required."
        )

    return sorted(
        set(out)
    )


def finite_errors(
    values: pd.Series | np.ndarray,
) -> np.ndarray:

    x = pd.to_numeric(
        pd.Series(values),
        errors="coerce",
    ).to_numpy(float)

    return x[
        np.isfinite(x)
    ]


def threshold_method_summary(
    errors: np.ndarray,
    thresholds_m: list[float],
) -> dict[str, Any]:

    x = np.asarray(
        errors,
        dtype=float,
    )

    x = x[
        np.isfinite(x)
    ]

    population = int(
        len(x)
    )

    counts = {}

    fractions = {}

    for threshold in thresholds_m:
        tag = f"{threshold:g}"

        count = int(
            (
                x <= threshold
            ).sum()
        )

        counts[tag] = count

        fractions[tag] = (
            count / population
            if population
            else None
        )

    return {
        "population": population,
        "counts": counts,
        "fractions": fractions,
        "median_error_m": (
            float(
                np.median(x)
            )
            if population
            else None
        ),
        "p95_error_m": (
            float(
                np.percentile(
                    x,
                    95,
                )
            )
            if population
            else None
        ),
        "max_error_m": (
            float(
                np.max(x)
            )
            if population
            else None
        ),
    }


def dino_best_errors_for_topk(
    dino: pd.DataFrame,
    topk: int,
) -> np.ndarray:

    subset = dino.loc[
        pd.to_numeric(
            dino["rank"],
            errors="raise",
        )
        <= int(topk)
    ].copy()

    best = (
        subset.groupby(
            "query_id",
            sort=True,
        )[
            "center_error_m"
        ]
        .min()
    )

    require(
        len(best) == 403,
        (
            f"DINO Top-{topk}: expected "
            f"403 queries, got {len(best)}."
        ),
    )

    return pd.to_numeric(
        best,
        errors="raise",
    ).to_numpy(float)


def threshold_summary_rows(
    methods: dict[str, dict[str, Any]],
    thresholds_m: list[float],
) -> pd.DataFrame:

    rows = []

    for method_name, summary in methods.items():

        for threshold in thresholds_m:

            tag = f"{threshold:g}"

            rows.append({
                "method": method_name,
                "threshold_m": float(
                    threshold
                ),
                "count_within_threshold": int(
                    summary[
                        "counts"
                    ][
                        tag
                    ]
                ),
                "population": int(
                    summary[
                        "population"
                    ]
                ),
                "fraction_within_threshold": (
                    summary[
                        "fractions"
                    ][
                        tag
                    ]
                ),
            })

    return pd.DataFrame(
        rows
    )


def flatten_drift_summary(
    summary: dict[str, Any],
) -> dict[str, Any]:

    row = {
        "status": summary[
            "status"
        ],

        "evaluation_start_frame":
            summary[
                "evaluation_window"
            ][
                "start_frame"
            ],

        "evaluation_end_frame":
            summary[
                "evaluation_window"
            ][
                "end_frame"
            ],

        "evaluation_start_time_s":
            summary[
                "evaluation_window"
            ][
                "start_time_s"
            ],

        "evaluation_end_time_s":
            summary[
                "evaluation_window"
            ][
                "end_time_s"
            ],

        "evaluation_duration_s":
            summary[
                "evaluation_window"
            ][
                "duration_s"
            ],

        "evaluation_distance_m":
            summary[
                "evaluation_window"
            ][
                "distance_m"
            ],

        "absolute_rmse_m":
            summary[
                "absolute_position_error"
            ][
                "rmse_m"
            ],

        "absolute_mean_m":
            summary[
                "absolute_position_error"
            ][
                "mean_m"
            ],

        "absolute_median_m":
            summary[
                "absolute_position_error"
            ][
                "median_m"
            ],

        "absolute_p95_m":
            summary[
                "absolute_position_error"
            ][
                "p95_m"
            ],

        "absolute_max_m":
            summary[
                "absolute_position_error"
            ][
                "max_m"
            ],

        "absolute_first_m":
            summary[
                "absolute_position_error"
            ][
                "first_error_m"
            ],

        "absolute_final_m":
            summary[
                "absolute_position_error"
            ][
                "final_error_m"
            ],

        "final_drift_from_map_lock_m":
            summary[
                "drift_from_map_lock"
            ][
                "final_drift_m"
            ],

        "drift_m_per_100m":
            summary[
                "drift_from_map_lock"
            ][
                "drift_m_per_100m"
            ],

        "drift_m_per_min":
            summary[
                "drift_from_map_lock"
            ][
                "drift_m_per_min"
            ],

        "drift_m_per_s":
            summary[
                "drift_from_map_lock"
            ][
                "drift_m_per_s"
            ],

        "average_reference_speed_mps":
            summary[
                "motion_context_evaluation_only"
            ][
                "average_reference_speed_mps"
            ],

        "average_reference_speed_kmh":
            summary[
                "motion_context_evaluation_only"
            ][
                "average_reference_speed_kmh"
            ],

        "average_reference_speed_m_per_min":
            summary[
                "motion_context_evaluation_only"
            ][
                "average_reference_speed_m_per_min"
            ],

        "seconds_per_100m":
            summary[
                "motion_context_evaluation_only"
            ][
                "seconds_per_100m"
            ],

        "drift_rate_crosscheck_abs_diff":
            summary[
                "motion_context_evaluation_only"
            ][
                "drift_rate_crosscheck"
            ][
                "absolute_difference"
            ],
    }

    for prefix, key in [
        (
            "absolute_error",
            "absolute_position_error_crossings",
        ),
        (
            "drift_from_lock",
            "drift_from_map_lock_crossings",
        ),
    ]:
        for crossing in summary[key]:
            threshold = int(
                crossing[
                    "threshold_m"
                ]
            )

            base = (
                f"{prefix}_{threshold}m"
            )

            row[
                f"{base}_crossed"
            ] = crossing["crossed"]

            row[
                f"{base}_frame"
            ] = crossing[
                "sequence_frame_id"
            ]

            row[
                f"{base}_elapsed_s"
            ] = crossing[
                "elapsed_since_map_lock_s"
            ]

            row[
                f"{base}_distance_m"
            ] = crossing[
                "distance_since_map_lock_m"
            ]

    return row


def save_error_vs_time_plot(
    elapsed_s: np.ndarray,
    absolute_error_m: np.ndarray,
    drift_from_lock_m: np.ndarray,
    out_path: Path,
) -> None:

    out_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig, ax = plt.subplots(
        figsize=(12, 6)
    )

    ax.plot(
        elapsed_s,
        absolute_error_m,
        label="Absolute position error",
    )

    ax.plot(
        elapsed_s,
        drift_from_lock_m,
        label="Drift from map lock",
    )

    for threshold in [
        10.0,
        20.0,
        40.0,
    ]:
        ax.axhline(
            threshold,
            linestyle="--",
            linewidth=1,
            alpha=0.55,
        )

    ax.set_xlabel(
        "Time since blind map lock [s]"
    )

    ax.set_ylabel(
        "Error / drift [m]"
    )

    ax.set_title(
        "Blind fused trajectory error vs time"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        out_path,
        dpi=180,
    )

    plt.close(fig)


def save_error_vs_distance_plot(
    distance_m: np.ndarray,
    absolute_error_m: np.ndarray,
    drift_from_lock_m: np.ndarray,
    out_path: Path,
) -> None:

    out_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig, ax = plt.subplots(
        figsize=(12, 6)
    )

    ax.plot(
        distance_m,
        absolute_error_m,
        label="Absolute position error",
    )

    ax.plot(
        distance_m,
        drift_from_lock_m,
        label="Drift from map lock",
    )

    for threshold in [
        10.0,
        20.0,
        40.0,
    ]:
        ax.axhline(
            threshold,
            linestyle="--",
            linewidth=1,
            alpha=0.55,
        )

    ax.set_xlabel(
        "Reference distance since blind map lock [m]"
    )

    ax.set_ylabel(
        "Error / drift [m]"
    )

    ax.set_title(
        "Blind fused trajectory error vs distance"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        out_path,
        dpi=180,
    )

    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Attach optional GT to a frozen blind "
            "submission and compute evaluation metrics."
        )
    )

    p.add_argument(
        "--submission",
        type=Path,
        required=True,
    )

    p.add_argument(
        "--freeze-record",
        type=Path,
        required=True,
    )

    p.add_argument(
        "--reference-attachment",
        type=Path,
        required=True,
    )

    p.add_argument(
        "--correction-manifest",
        type=Path,
        required=True,
    )

    p.add_argument(
        "--run-root",
        type=Path,
        required=True,
    )

    p.add_argument(
        "--reference-crs",
        default="EPSG:4326",
    )

    p.add_argument(
        "--map-crs",
        default="EPSG:3346",
    )

    p.add_argument(
        "--sustain-frames",
        type=int,
        default=5,
        help=(
            "Consecutive frames required to declare "
            "a 10/20/40 m threshold crossing."
        ),
    )

    p.add_argument(
        "--dino-topk",
        type=Path,
        required=True,
        help=(
            "Saved DINO Top-100 retrieval CSV. "
            "Evaluation-only errors are consumed "
            "only after submission freeze."
        ),
    )

    p.add_argument(
        "--orb-query-summary",
        type=Path,
        required=True,
    )

    p.add_argument(
        "--lightglue-query-summary",
        type=Path,
        required=True,
    )

    p.add_argument(
        "--thresholds-m",
        default="10,20,40,80,100",
    )

    args = p.parse_args()

    require(
        args.sustain_frames >= 1,
        "--sustain-frames must be >= 1.",
    )

    started = time.perf_counter()

    submission_path = (
        args.submission
        .expanduser()
        .resolve()
    )

    freeze_path = (
        args.freeze_record
        .expanduser()
        .resolve()
    )

    reference_path = (
        args.reference_attachment
        .expanduser()
        .resolve()
    )

    correction_path = (
        args.correction_manifest
        .expanduser()
        .resolve()
    )

    run_root = (
        args.run_root
        .expanduser()
        .resolve()
    )

    dino_topk_path = (
        args.dino_topk
        .expanduser()
        .resolve()
    )

    orb_query_summary_path = (
        args.orb_query_summary
        .expanduser()
        .resolve()
    )

    lightglue_query_summary_path = (
        args.lightglue_query_summary
        .expanduser()
        .resolve()
    )

    thresholds_eval_m = parse_thresholds_m(
        args.thresholds_m
    )

    for path in [
        submission_path,
        freeze_path,
        reference_path,
        correction_path,
        dino_topk_path,
        orb_query_summary_path,
        lightglue_query_summary_path,
    ]:
        require(
            path.exists(),
            f"Missing input: {path}",
        )

    # =====================================================
    # Hard freeze verification BEFORE reading GT.
    # =====================================================

    freeze = json.loads(
        freeze_path.read_text(
            encoding="utf-8"
        )
    )

    require(
        freeze.get("status")
        == "PASS_BLIND_SUBMISSION_FROZEN",
        "Freeze record is not passing.",
    )

    frozen_sha = (
        freeze[
            "submission"
        ][
            "sha256"
        ]
    )

    sha_before = sha256_file(
        submission_path
    )

    require(
        sha_before == frozen_sha,
        (
            "Frozen submission SHA mismatch BEFORE "
            "evaluation. Localization output changed "
            "after freeze."
        ),
    )

    # =====================================================
    # Read frozen blind output.
    # =====================================================

    sub = pd.read_csv(
        submission_path
    )

    require(
        len(sub) == 403,
        (
            "Expected 403 submission rows, got "
            f"{len(sub)}."
        ),
    )

    required_sub = {
        "frame_index",
        "sequence_frame_id",
        "query_id",
        "token0_id",
        "timestamp_s",
        "estimated_map_x",
        "estimated_map_y",
        "estimated_lat",
        "estimated_lon",
        "map_aligned_available",
        "map_lock_event",
        "accepted_correction",
    }

    missing = sorted(
        required_sub
        - set(sub.columns)
    )

    require(
        not missing,
        (
            "Submission missing columns: "
            f"{missing}"
        ),
    )

    for col in [
        "frame_index",
        "sequence_frame_id",
        "query_id",
        "token0_id",
    ]:
        sub[col] = pd.to_numeric(
            sub[col],
            errors="raise",
        ).astype(int)

    # =====================================================
    # NOW reference becomes visible.
    # =====================================================

    ref = pd.read_csv(
        reference_path,
        usecols=[
            "frame_index",
            "sequence_frame_id",
            "query_id",
            "token0_id",
            "timestamp_s",
            "eval_ref_lat",
            "eval_ref_lon",
            "eval_reference_available",
            "eval_reference_time_delta_ms",
            "evaluation_only",
        ],
    )

    require(
        len(ref) == 403,
        (
            "Expected 403 reference rows, got "
            f"{len(ref)}."
        ),
    )

    for col in [
        "frame_index",
        "sequence_frame_id",
        "query_id",
        "token0_id",
    ]:
        ref[col] = pd.to_numeric(
            ref[col],
            errors="raise",
        ).astype(int)

    require(
        bool_series(
            ref[
                "evaluation_only"
            ]
        ).all(),
        (
            "Reference attachment contains rows "
            "not marked evaluation_only."
        ),
    )

    ref_available = bool_series(
        ref[
            "eval_reference_available"
        ]
    )

    require(
        ref_available.all(),
        (
            "One or more reference rows are "
            "unavailable."
        ),
    )

    # =====================================================
    # Identity join.
    # =====================================================

    joined = sub.merge(
        ref,
        on=[
            "frame_index",
            "sequence_frame_id",
            "query_id",
            "token0_id",
        ],
        how="inner",
        suffixes=(
            "_estimate",
            "_reference",
        ),
        validate="one_to_one",
    )

    require(
        len(joined) == 403,
        (
            "Estimate/reference identity join "
            "did not preserve 403 rows."
        ),
    )

    estimate_ts = pd.to_numeric(
        joined[
            "timestamp_s_estimate"
        ],
        errors="raise",
    ).to_numpy(float)

    reference_ts = pd.to_numeric(
        joined[
            "timestamp_s_reference"
        ],
        errors="raise",
    ).to_numpy(float)

    identity_time_delta_ms = (
        1000.0
        * (
            reference_ts
            - estimate_ts
        )
    )

    require(
        float(
            np.max(
                np.abs(
                    identity_time_delta_ms
                )
            )
        )
        < 1e-6,
        (
            "Submission/reference timestamp "
            "identity mismatch."
        ),
    )

    # =====================================================
    # Project reference lat/lon to SAME EPSG:3346
    # coordinate system used by blind estimates.
    # =====================================================

    try:
        from pyproj import Transformer
    except Exception as exc:
        raise RuntimeError(
            "pyproj is required."
        ) from exc

    transformer = Transformer.from_crs(
        args.reference_crs,
        args.map_crs,
        always_xy=True,
    )

    ref_lon = pd.to_numeric(
        joined[
            "eval_ref_lon"
        ],
        errors="raise",
    ).to_numpy(float)

    ref_lat = pd.to_numeric(
        joined[
            "eval_ref_lat"
        ],
        errors="raise",
    ).to_numpy(float)

    ref_x, ref_y = transformer.transform(
        ref_lon,
        ref_lat,
    )

    ref_x = np.asarray(
        ref_x,
        dtype=float,
    )

    ref_y = np.asarray(
        ref_y,
        dtype=float,
    )

    joined[
        "eval_ref_map_x"
    ] = ref_x

    joined[
        "eval_ref_map_y"
    ] = ref_y

    # =====================================================
    # Full continuous fused trajectory evaluation.
    # =====================================================

    available = bool_series(
        joined[
            "map_aligned_available"
        ]
    ).to_numpy(bool)

    est_x = pd.to_numeric(
        joined[
            "estimated_map_x"
        ],
        errors="coerce",
    ).to_numpy(float)

    est_y = pd.to_numeric(
        joined[
            "estimated_map_y"
        ],
        errors="coerce",
    ).to_numpy(float)

    error = np.full(
        len(joined),
        np.nan,
        dtype=float,
    )

    error[
        available
    ] = np.hypot(
        est_x[
            available
        ]
        - ref_x[
            available
        ],

        est_y[
            available
        ]
        - ref_y[
            available
        ],
    )

    joined[
        "evaluation_position_error_m"
    ] = error

    continuous_metrics = metrics(
        error[
            available
        ]
    )

    available_indices = np.flatnonzero(
        available
    )

    require(
        len(
            available_indices
        ) == 386,
        (
            "Expected 386 available estimates, got "
            f"{len(available_indices)}."
        ),
    )

    first_available_index = int(
        available_indices[0]
    )

    last_available_index = int(
        available_indices[-1]
    )

    continuous_metrics.update({
        "first_available_frame": int(
            joined.iloc[
                first_available_index
            ][
                "sequence_frame_id"
            ]
        ),
        "first_available_time_s": float(
            joined.iloc[
                first_available_index
            ][
                "timestamp_s_estimate"
            ]
        ),
        "first_available_error_m": float(
            error[
                first_available_index
            ]
        ),
        "final_frame": int(
            joined.iloc[
                last_available_index
            ][
                "sequence_frame_id"
            ]
        ),
        "final_time_s": float(
            joined.iloc[
                last_available_index
            ][
                "timestamp_s_estimate"
            ]
        ),
        "final_error_m": float(
            error[
                last_available_index
            ]
        ),
    })


    # =====================================================
    # Integrated Add-on 1:
    # drift / time / distance analysis.
    #
    # IMPORTANT:
    # Absolute error at map lock may be non-zero.
    # Therefore pure post-lock drift is measured as
    # CHANGE IN THE 2-D ESTIMATION ERROR VECTOR relative
    # to the first causal map pose.
    # =====================================================

    reference_cumulative_distance = (
        cumulative_distance_xy(
            ref_x,
            ref_y,
        )
    )

    joined[
        "eval_reference_cumulative_distance_m"
    ] = reference_cumulative_distance

    eval_frame_ids = (
        joined.loc[
            available,
            "sequence_frame_id",
        ]
        .to_numpy(int)
    )

    eval_timestamps = (
        estimate_ts[
            available
        ]
    )

    eval_elapsed_s = (
        eval_timestamps
        - eval_timestamps[0]
    )

    eval_reference_distance_m = (
        reference_cumulative_distance[
            available
        ]
        - reference_cumulative_distance[
            available
        ][0]
    )

    eval_absolute_error_m = (
        error[
            available
        ]
    )

    eval_error_vectors = np.column_stack([
        est_x[
            available
        ]
        - ref_x[
            available
        ],

        est_y[
            available
        ]
        - ref_y[
            available
        ],
    ])

    eval_drift_vectors = (
        eval_error_vectors
        - eval_error_vectors[0]
    )

    eval_drift_from_lock_m = (
        np.linalg.norm(
            eval_drift_vectors,
            axis=1,
        )
    )

    elapsed_full = np.full(
        len(joined),
        np.nan,
        dtype=float,
    )

    distance_full = np.full(
        len(joined),
        np.nan,
        dtype=float,
    )

    drift_full = np.full(
        len(joined),
        np.nan,
        dtype=float,
    )

    elapsed_full[
        available
    ] = eval_elapsed_s

    distance_full[
        available
    ] = eval_reference_distance_m

    drift_full[
        available
    ] = eval_drift_from_lock_m

    joined[
        "evaluation_elapsed_since_map_lock_s"
    ] = elapsed_full

    joined[
        "evaluation_distance_since_map_lock_m"
    ] = distance_full

    joined[
        "evaluation_drift_from_map_lock_m"
    ] = drift_full

    evaluation_duration_s = float(
        eval_elapsed_s[-1]
    )

    evaluation_distance_m = float(
        eval_reference_distance_m[-1]
    )

    final_drift_from_lock_m = float(
        eval_drift_from_lock_m[-1]
    )

    drift_m_per_100m = (
        100.0
        * final_drift_from_lock_m
        / evaluation_distance_m
        if evaluation_distance_m > 1e-9
        else float("nan")
    )

    drift_m_per_min = (
        60.0
        * final_drift_from_lock_m
        / evaluation_duration_s
        if evaluation_duration_s > 1e-9
        else float("nan")
    )

    drift_m_per_s = (
        final_drift_from_lock_m
        / evaluation_duration_s
        if evaluation_duration_s > 1e-9
        else float("nan")
    )

    # =====================================================
    # MOTION_CONTEXT_RATE_CROSSCHECK
    #
    # Reference ground-motion speed is evaluation-only.
    # It is never used by localization.
    # =====================================================

    average_reference_speed_mps = (
        evaluation_distance_m
        / evaluation_duration_s
        if evaluation_duration_s > 1e-9
        else float("nan")
    )

    average_reference_speed_kmh = (
        average_reference_speed_mps
        * 3.6
    )

    average_reference_speed_m_per_min = (
        average_reference_speed_mps
        * 60.0
    )

    seconds_per_100m = (
        100.0
        / average_reference_speed_mps
        if average_reference_speed_mps > 1e-9
        else float("nan")
    )

    reference_step_distance_m = np.diff(
        eval_reference_distance_m
    )

    reference_step_dt_s = np.diff(
        eval_timestamps
    )

    valid_speed_step = (
        np.isfinite(
            reference_step_distance_m
        )
        & np.isfinite(
            reference_step_dt_s
        )
        & (
            reference_step_dt_s
            > 1e-9
        )
    )

    reference_step_speed_mps = (
        reference_step_distance_m[
            valid_speed_step
        ]
        / reference_step_dt_s[
            valid_speed_step
        ]
    )

    median_reference_step_speed_mps = (
        float(
            np.median(
                reference_step_speed_mps
            )
        )
        if len(
            reference_step_speed_mps
        )
        else float("nan")
    )

    p95_reference_step_speed_mps = (
        float(
            np.percentile(
                reference_step_speed_mps,
                95,
            )
        )
        if len(
            reference_step_speed_mps
        )
        else float("nan")
    )

    max_reference_step_speed_mps = (
        float(
            np.max(
                reference_step_speed_mps
            )
        )
        if len(
            reference_step_speed_mps
        )
        else float("nan")
    )

    drift_m_per_min_from_distance_rate = (
        drift_m_per_100m
        * average_reference_speed_m_per_min
        / 100.0
    )

    drift_rate_crosscheck_abs_diff = abs(
        drift_m_per_min_from_distance_rate
        - drift_m_per_min
    )

    thresholds_m = [
        10.0,
        20.0,
        40.0,
    ]

    absolute_crossings = [
        first_sustained_crossing(
            values=eval_absolute_error_m,
            threshold_m=threshold,
            frame_ids=eval_frame_ids,
            timestamps_s=eval_timestamps,
            elapsed_s=eval_elapsed_s,
            distances_m=eval_reference_distance_m,
            sustain_frames=args.sustain_frames,
        )
        for threshold in thresholds_m
    ]

    drift_crossings = [
        first_sustained_crossing(
            values=eval_drift_from_lock_m,
            threshold_m=threshold,
            frame_ids=eval_frame_ids,
            timestamps_s=eval_timestamps,
            elapsed_s=eval_elapsed_s,
            distances_m=eval_reference_distance_m,
            sustain_frames=args.sustain_frames,
        )
        for threshold in thresholds_m
    ]

    drift_summary = {
        "stage": (
            "ADDON1_INTEGRATED_DRIFT_TIME"
        ),

        "status": (
            "PASS_ADDON1_INTEGRATED_DRIFT_TIME"
        ),

        "metric_contract": {
            "absolute_position_error": (
                "Euclidean distance between blind "
                "estimated EPSG:3346 position and "
                "withheld reference EPSG:3346 position."
            ),

            "drift_from_map_lock": (
                "Magnitude of the change in the 2-D "
                "estimation-error vector relative to "
                "the first causal map-aligned pose."
            ),

            "initial_absolute_offset_excluded_from_drift":
                True,

            "threshold_crossing_requires_consecutive_frames":
                int(
                    args.sustain_frames
                ),

            "reference_used_for_evaluation_only":
                True,
        },

        "evaluation_window": {
            "start_frame": int(
                eval_frame_ids[0]
            ),

            "end_frame": int(
                eval_frame_ids[-1]
            ),

            "start_time_s": float(
                eval_timestamps[0]
            ),

            "end_time_s": float(
                eval_timestamps[-1]
            ),

            "duration_s": (
                evaluation_duration_s
            ),

            "distance_m": (
                evaluation_distance_m
            ),

            "evaluated_poses": int(
                len(
                    eval_absolute_error_m
                )
            ),
        },

        "motion_context_evaluation_only": {
            "reference_distance_m":
                evaluation_distance_m,

            "reference_duration_s":
                evaluation_duration_s,

            "average_reference_speed_mps":
                average_reference_speed_mps,

            "average_reference_speed_kmh":
                average_reference_speed_kmh,

            "average_reference_speed_m_per_min":
                average_reference_speed_m_per_min,

            "seconds_per_100m":
                seconds_per_100m,

            "median_reference_step_speed_mps":
                median_reference_step_speed_mps,

            "p95_reference_step_speed_mps":
                p95_reference_step_speed_mps,

            "max_reference_step_speed_mps":
                max_reference_step_speed_mps,

            "drift_rate_crosscheck": {
                "drift_m_per_100m":
                    drift_m_per_100m,

                "reference_motion_m_per_min":
                    average_reference_speed_m_per_min,

                "derived_drift_m_per_min":
                    drift_m_per_min_from_distance_rate,

                "direct_drift_m_per_min":
                    drift_m_per_min,

                "absolute_difference":
                    drift_rate_crosscheck_abs_diff,
            },

            "note": (
                "Reference speed is calculated only "
                "after GT attachment for evaluation "
                "and is never used by localization."
            ),
        },

        "absolute_position_error": {
            **metrics(
                eval_absolute_error_m
            ),

            "first_error_m": float(
                eval_absolute_error_m[0]
            ),

            "final_error_m": float(
                eval_absolute_error_m[-1]
            ),
        },

        "drift_from_map_lock": {
            "initial_drift_m": float(
                eval_drift_from_lock_m[0]
            ),

            "final_drift_m": (
                final_drift_from_lock_m
            ),

            "max_drift_m": float(
                np.max(
                    eval_drift_from_lock_m
                )
            ),

            "median_drift_m": float(
                np.median(
                    eval_drift_from_lock_m
                )
            ),

            "p95_drift_m": float(
                np.percentile(
                    eval_drift_from_lock_m,
                    95,
                )
            ),

            "drift_m_per_100m": (
                drift_m_per_100m
            ),

            "drift_m_per_min": (
                drift_m_per_min
            ),

            "drift_m_per_s": (
                drift_m_per_s
            ),
        },

        "absolute_position_error_crossings":
            absolute_crossings,

        "drift_from_map_lock_crossings":
            drift_crossings,
    }

    # Also expose the evaluation-window fields through
    # the main continuous-trajectory summary.
    continuous_metrics.update({
        "evaluation_duration_s":
            evaluation_duration_s,

        "evaluation_distance_m":
            evaluation_distance_m,

        "final_drift_from_map_lock_m":
            final_drift_from_lock_m,

        "drift_from_map_lock_m_per_100m":
            drift_m_per_100m,

        "drift_from_map_lock_m_per_min":
            drift_m_per_min,

        "drift_from_map_lock_m_per_s":
            drift_m_per_s,

        "average_reference_speed_mps":
            average_reference_speed_mps,

        "average_reference_speed_kmh":
            average_reference_speed_kmh,

        "average_reference_speed_m_per_min":
            average_reference_speed_m_per_min,

        "seconds_per_100m":
            seconds_per_100m,
    })

    # =====================================================
    # Map-lock event evaluation.
    # =====================================================

    lock_mask = bool_series(
        joined[
            "map_lock_event"
        ]
    ).to_numpy(bool)

    require(
        int(
            lock_mask.sum()
        ) == 1,
        (
            "Expected exactly one map-lock event."
        ),
    )

    lock_index = int(
        np.flatnonzero(
            lock_mask
        )[0]
    )

    map_lock_eval = {
        "frame": int(
            joined.iloc[
                lock_index
            ][
                "sequence_frame_id"
            ]
        ),
        "time_s": float(
            joined.iloc[
                lock_index
            ][
                "timestamp_s_estimate"
            ]
        ),
        "estimated_map_x": float(
            est_x[
                lock_index
            ]
        ),
        "estimated_map_y": float(
            est_y[
                lock_index
            ]
        ),
        "reference_map_x": float(
            ref_x[
                lock_index
            ]
        ),
        "reference_map_y": float(
            ref_y[
                lock_index
            ]
        ),
        "position_error_m": float(
            error[
                lock_index
            ]
        ),
    }

    # =====================================================
    # Accepted-correction evaluation.
    #
    # Two different quantities:
    #
    # 1. absolute_candidate_error_m:
    #    how good was selected ORB map tile center?
    #
    # 2. fused_position_error_m:
    #    how good was final fused estimate after applying
    #    alpha=0.25 at that frame?
    # =====================================================

    corr = pd.read_csv(
        correction_path,
        usecols=[
            "sequence_frame_id",
            "query_id",
            "token0_id",
            "timestamp_s",
            "center_easting",
            "center_northing",
            "correction_candidate",
            "correction_accepted",
            "strict_a_blind",
            "strict_b_blind",
            "reranked_top1_tile_id",
            "reranked_top1_inliers",
            "reranked_top1_verifier_score",
            "reranked_top1_hybrid_score",
            "temporal_residual_m",
            "temporal_threshold_m",
            "distance_since_anchor_m",
            "correction_reason",
        ],
    )

    for col in [
        "sequence_frame_id",
        "query_id",
        "token0_id",
    ]:
        corr[col] = pd.to_numeric(
            corr[col],
            errors="raise",
        ).astype(int)

    corr = corr.merge(
        joined[
            [
                "sequence_frame_id",
                "query_id",
                "token0_id",
                "eval_ref_map_x",
                "eval_ref_map_y",
                "estimated_map_x",
                "estimated_map_y",
                "evaluation_position_error_m",
            ]
        ],
        on=[
            "sequence_frame_id",
            "query_id",
            "token0_id",
        ],
        how="left",
        validate="one_to_one",
    )

    corr[
        "absolute_candidate_error_m"
    ] = np.hypot(
        pd.to_numeric(
            corr[
                "center_easting"
            ],
            errors="coerce",
        )
        - pd.to_numeric(
            corr[
                "eval_ref_map_x"
            ],
            errors="coerce",
        ),

        pd.to_numeric(
            corr[
                "center_northing"
            ],
            errors="coerce",
        )
        - pd.to_numeric(
            corr[
                "eval_ref_map_y"
            ],
            errors="coerce",
        ),
    )

    accepted = bool_series(
        corr[
            "correction_accepted"
        ]
    )

    accepted_rows = corr.loc[
        accepted
    ].copy()

    require(
        len(
            accepted_rows
        ) == 14,
        (
            "Expected 14 accepted corrections, got "
            f"{len(accepted_rows)}."
        ),
    )

    accepted_rows[
        "absolute_candidate_error_m"
    ] = np.hypot(
        pd.to_numeric(
            accepted_rows[
                "center_easting"
            ],
            errors="raise",
        )
        - pd.to_numeric(
            accepted_rows[
                "eval_ref_map_x"
            ],
            errors="raise",
        ),

        pd.to_numeric(
            accepted_rows[
                "center_northing"
            ],
            errors="raise",
        )
        - pd.to_numeric(
            accepted_rows[
                "eval_ref_map_y"
            ],
            errors="raise",
        ),
    )

    accepted_rows[
        "fused_position_error_m"
    ] = pd.to_numeric(
        accepted_rows[
            "evaluation_position_error_m"
        ],
        errors="raise",
    )

    abs_candidate_errors = (
        accepted_rows[
            "absolute_candidate_error_m"
        ].to_numpy(float)
    )

    fused_at_correction_errors = (
        accepted_rows[
            "fused_position_error_m"
        ].to_numpy(float)
    )

    correction_eval = {
        "accepted_count": int(
            len(
                accepted_rows
            )
        ),

        "absolute_candidate_metrics":
            metrics(
                abs_candidate_errors
            ),

        "fused_position_at_correction_metrics":
            metrics(
                fused_at_correction_errors
            ),

        "absolute_candidate_le10m": int(
            (
                abs_candidate_errors
                <= 10.0
            ).sum()
        ),

        "absolute_candidate_le20m": int(
            (
                abs_candidate_errors
                <= 20.0
            ).sum()
        ),

        "absolute_candidate_le40m": int(
            (
                abs_candidate_errors
                <= 40.0
            ).sum()
        ),

        "absolute_candidate_gt40m": int(
            (
                abs_candidate_errors
                > 40.0
            ).sum()
        ),

        "absolute_candidate_gt100m": int(
            (
                abs_candidate_errors
                > 100.0
            ).sum()
        ),
    }


    # =====================================================
    # Integrated Add-on 2:
    # threshold sensitivity.
    #
    # All numeric localization-error fields below are
    # evaluation-only and become visible only after the
    # frozen submission SHA has been verified.
    # =====================================================

    dino = pd.read_csv(
        dino_topk_path,
        usecols=[
            "query_id",
            "rank",
            "center_error_m",
        ],
    )

    require(
        len(dino) == 40300,
        (
            "Expected 40300 DINO Top-100 rows, got "
            f"{len(dino)}."
        ),
    )

    dino[
        "query_id"
    ] = pd.to_numeric(
        dino[
            "query_id"
        ],
        errors="raise",
    ).astype(int)

    dino[
        "rank"
    ] = pd.to_numeric(
        dino[
            "rank"
        ],
        errors="raise",
    ).astype(int)

    dino[
        "center_error_m"
    ] = pd.to_numeric(
        dino[
            "center_error_m"
        ],
        errors="raise",
    )

    require(
        dino[
            "query_id"
        ].nunique()
        == 403,
        (
            "DINO Top-100 does not contain "
            "403 unique queries."
        ),
    )

    require(
        int(
            dino[
                "rank"
            ].min()
        ) == 1
        and int(
            dino[
                "rank"
            ].max()
        ) == 100,
        (
            "DINO rank contract is not 1..100."
        ),
    )

    orb_qsum = pd.read_csv(
        orb_query_summary_path,
        usecols=[
            "query_id",
            "reranked_top1_error_m",
        ],
    )

    require(
        len(
            orb_qsum
        ) == 403,
        (
            "Expected 403 ORB summary rows, got "
            f"{len(orb_qsum)}."
        ),
    )

    require(
        orb_qsum[
            "query_id"
        ].nunique()
        == 403,
        (
            "ORB query summary does not contain "
            "403 unique queries."
        ),
    )

    lg_qsum = pd.read_csv(
        lightglue_query_summary_path,
        usecols=[
            "query_id",
            "reranked_top1_error_m",
        ],
    )

    require(
        len(
            lg_qsum
        ) == 403,
        (
            "Expected 403 LightGlue summary rows, got "
            f"{len(lg_qsum)}."
        ),
    )

    require(
        lg_qsum[
            "query_id"
        ].nunique()
        == 403,
        (
            "LightGlue query summary does not contain "
            "403 unique queries."
        ),
    )

    threshold_methods = {
        "dino_top1":
            threshold_method_summary(
                dino_best_errors_for_topk(
                    dino,
                    1,
                ),
                thresholds_eval_m,
            ),

        "dino_top5":
            threshold_method_summary(
                dino_best_errors_for_topk(
                    dino,
                    5,
                ),
                thresholds_eval_m,
            ),

        "dino_top10":
            threshold_method_summary(
                dino_best_errors_for_topk(
                    dino,
                    10,
                ),
                thresholds_eval_m,
            ),

        "dino_top20":
            threshold_method_summary(
                dino_best_errors_for_topk(
                    dino,
                    20,
                ),
                thresholds_eval_m,
            ),

        "dino_top100":
            threshold_method_summary(
                dino_best_errors_for_topk(
                    dino,
                    100,
                ),
                thresholds_eval_m,
            ),

        "orb_selected":
            threshold_method_summary(
                finite_errors(
                    orb_qsum[
                        "reranked_top1_error_m"
                    ]
                ),
                thresholds_eval_m,
            ),

        "lightglue_selected":
            threshold_method_summary(
                finite_errors(
                    lg_qsum[
                        "reranked_top1_error_m"
                    ]
                ),
                thresholds_eval_m,
            ),

        "accepted_corrections":
            threshold_method_summary(
                finite_errors(
                    accepted_rows[
                        "absolute_candidate_error_m"
                    ]
                ),
                thresholds_eval_m,
            ),
    }

    require(
        threshold_methods[
            "accepted_corrections"
        ][
            "population"
        ] == 14,
        (
            "Expected threshold analysis over "
            "14 blind accepted corrections."
        ),
    )

    threshold_sensitivity = {
        "stage": (
            "ADDON2_INTEGRATED_THRESHOLD_SENSITIVITY"
        ),

        "status": (
            "PASS_ADDON2_INTEGRATED_THRESHOLD_SENSITIVITY"
        ),

        "evaluation_only": True,

        "important_rule": (
            "Thresholds are applied only after GT/reference "
            "attachment. They do not participate in blind "
            "retrieval, verification, correction acceptance, "
            "or fusion."
        ),

        "thresholds_m":
            thresholds_eval_m,

        "population_contract": {
            "retrieval_and_reranker_methods":
                "403 queries",

            "accepted_corrections":
                (
                    "Only the 14 corrections accepted "
                    "by the blind temporal gate."
                ),
        },

        "methods":
            threshold_methods,
    }

    threshold_table = (
        threshold_summary_rows(
            threshold_methods,
            thresholds_eval_m,
        )
    )

    # =====================================================
    # Integrated Add-on 3:
    # accepted-correction safety summary.
    # =====================================================

    candidate_mask = bool_series(
        corr[
            "correction_candidate"
        ]
    )

    accepted_mask_all = bool_series(
        corr[
            "correction_accepted"
        ]
    )

    require(
        not bool(
            (
                accepted_mask_all
                & ~candidate_mask
            ).any()
        ),
        (
            "Found accepted correction that was "
            "not marked correction_candidate."
        ),
    )

    candidate_rows = corr.loc[
        candidate_mask
    ].copy()

    accepted_safety_rows = corr.loc[
        candidate_mask
        & accepted_mask_all
    ].copy()

    rejected_safety_rows = corr.loc[
        candidate_mask
        & ~accepted_mask_all
    ].copy()

    accepted_safety_errors = finite_errors(
        accepted_safety_rows[
            "absolute_candidate_error_m"
        ]
    )

    candidate_total = int(
        len(
            candidate_rows
        )
    )

    accepted_total = int(
        len(
            accepted_safety_rows
        )
    )

    rejected_total = int(
        len(
            rejected_safety_rows
        )
    )

    require(
        accepted_total == 14,
        (
            "Expected 14 accepted blind corrections, got "
            f"{accepted_total}."
        ),
    )

    require(
        candidate_total
        == accepted_total
        + rejected_total,
        (
            "Candidate accounting mismatch."
        ),
    )

    accepted_le10 = int(
        (
            accepted_safety_errors
            <= 10.0
        ).sum()
    )

    accepted_le20 = int(
        (
            accepted_safety_errors
            <= 20.0
        ).sum()
    )

    accepted_le40 = int(
        (
            accepted_safety_errors
            <= 40.0
        ).sum()
    )

    accepted_gt40 = int(
        (
            accepted_safety_errors
            > 40.0
        ).sum()
    )

    accepted_le100 = int(
        (
            accepted_safety_errors
            <= 100.0
        ).sum()
    )

    accepted_gt100 = int(
        (
            accepted_safety_errors
            > 100.0
        ).sum()
    )

    rejection_reason_counts = {
        str(k): int(v)
        for k, v in (
            rejected_safety_rows[
                "correction_reason"
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

    accepted_error_metrics = metrics(
        accepted_safety_errors
    )

    correction_safety = {
        "stage": (
            "ADDON3_INTEGRATED_ACCEPTED_CORRECTION_SAFETY"
        ),

        "status": (
            "PASS_ADDON3_INTEGRATED_ACCEPTED_CORRECTION_SAFETY"
        ),

        "evaluation_only": True,

        "important_rule": (
            "Correction acceptance was made blindly. "
            "Reference error is attached only afterward "
            "to evaluate the safety of those decisions."
        ),

        "candidate_total":
            candidate_total,

        "accepted_total":
            accepted_total,

        "rejected_total":
            rejected_total,

        "accepted_le10m":
            accepted_le10,

        "accepted_le20m":
            accepted_le20,

        "accepted_le40m":
            accepted_le40,

        "accepted_gt40m_false":
            accepted_gt40,

        "accepted_le100m":
            accepted_le100,

        "accepted_gt100m_dangerous":
            accepted_gt100,

        "precision_le40":
            (
                accepted_le40
                / accepted_total
                if accepted_total
                else None
            ),

        "precision_le100":
            (
                accepted_le100
                / accepted_total
                if accepted_total
                else None
            ),

        "accepted_error_metrics":
            accepted_error_metrics,

        "rejection_reason_counts":
            rejection_reason_counts,
    }

    correction_safety_row = {
        "candidate_total":
            candidate_total,

        "accepted_total":
            accepted_total,

        "rejected_total":
            rejected_total,

        "accepted_le10m":
            accepted_le10,

        "accepted_le20m":
            accepted_le20,

        "accepted_le40m":
            accepted_le40,

        "accepted_gt40m_false":
            accepted_gt40,

        "accepted_le100m":
            accepted_le100,

        "accepted_gt100m_dangerous":
            accepted_gt100,

        "precision_le40":
            (
                accepted_le40
                / accepted_total
                if accepted_total
                else None
            ),

        "precision_le100":
            (
                accepted_le100
                / accepted_total
                if accepted_total
                else None
            ),

        "median_accepted_error_m":
            accepted_error_metrics[
                "median_m"
            ],

        "p95_accepted_error_m":
            accepted_error_metrics[
                "p95_m"
            ],

        "max_accepted_error_m":
            accepted_error_metrics[
                "max_m"
            ],
    }

    for reason, count in (
        rejection_reason_counts.items()
    ):
        correction_safety_row[
            f"rejected_reason_{reason}"
        ] = int(
            count
        )

    # =====================================================
    # Save evaluation-only artifacts.
    # =====================================================

    evaluation_dir = (
        run_root
        / "evaluation"
    )

    evaluation_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics_dir = (
        run_root
        / "metrics"
    )

    figures_dir = (
        run_root
        / "figures"
    )

    metrics_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    figures_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    drift_json_path = (
        metrics_dir
        / "drift_time_summary.json"
    )

    drift_csv_path = (
        metrics_dir
        / "drift_time_summary.csv"
    )

    threshold_json_path = (
        metrics_dir
        / "threshold_sensitivity.json"
    )

    threshold_csv_path = (
        metrics_dir
        / "threshold_sensitivity.csv"
    )

    safety_json_path = (
        metrics_dir
        / "accepted_correction_safety_summary.json"
    )

    safety_csv_path = (
        metrics_dir
        / "accepted_correction_safety_summary.csv"
    )

    error_vs_time_path = (
        figures_dir
        / "error_vs_time.png"
    )

    error_vs_distance_path = (
        figures_dir
        / "error_vs_distance.png"
    )

    trajectory_eval_path = (
        evaluation_dir
        / "blind_submission_trajectory_evaluation.csv"
    )

    correction_eval_path = (
        evaluation_dir
        / "blind_submission_accepted_correction_evaluation.csv"
    )

    summary_path = (
        evaluation_dir
        / "blind_submission_evaluation_summary.json"
    )

    eval_out = joined[
        [
            "frame_index",
            "sequence_frame_id",
            "query_id",
            "token0_id",
            "timestamp_s_estimate",
            "map_aligned_available",
            "estimated_map_x",
            "estimated_map_y",
            "estimated_lat",
            "estimated_lon",
            "eval_ref_lat",
            "eval_ref_lon",
            "eval_ref_map_x",
            "eval_ref_map_y",
            "eval_reference_time_delta_ms",
            "eval_reference_cumulative_distance_m",
            "evaluation_elapsed_since_map_lock_s",
            "evaluation_distance_since_map_lock_m",
            "evaluation_position_error_m",
            "evaluation_drift_from_map_lock_m",
            "accepted_correction",
            "map_lock_event",
        ]
    ].copy()

    eval_out[
        "evaluation_only"
    ] = True

    accepted_rows[
        "evaluation_only"
    ] = True

    write_start = time.perf_counter()

    eval_out.to_csv(
        trajectory_eval_path,
        index=False,
    )

    accepted_rows.to_csv(
        correction_eval_path,
        index=False,
    )

    drift_json_path.write_text(
        json.dumps(
            drift_summary,
            indent=2,
            default=json_safe,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    pd.DataFrame([
        flatten_drift_summary(
            drift_summary
        )
    ]).to_csv(
        drift_csv_path,
        index=False,
    )

    threshold_json_path.write_text(
        json.dumps(
            threshold_sensitivity,
            indent=2,
            default=json_safe,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    threshold_table.to_csv(
        threshold_csv_path,
        index=False,
    )

    safety_json_path.write_text(
        json.dumps(
            correction_safety,
            indent=2,
            default=json_safe,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    pd.DataFrame([
        correction_safety_row
    ]).to_csv(
        safety_csv_path,
        index=False,
    )

    save_error_vs_time_plot(
        elapsed_s=eval_elapsed_s,
        absolute_error_m=eval_absolute_error_m,
        drift_from_lock_m=eval_drift_from_lock_m,
        out_path=error_vs_time_path,
    )

    save_error_vs_distance_plot(
        distance_m=eval_reference_distance_m,
        absolute_error_m=eval_absolute_error_m,
        drift_from_lock_m=eval_drift_from_lock_m,
        out_path=error_vs_distance_path,
    )

    write_s = (
        time.perf_counter()
        - write_start
    )

    # =====================================================
    # SHA verification AFTER evaluation.
    # =====================================================

    sha_after = sha256_file(
        submission_path
    )

    require(
        sha_after == frozen_sha,
        (
            "Frozen submission changed during "
            "evaluation."
        ),
    )

    total_s = (
        time.perf_counter()
        - started
    )

    summary = {
        "stage": (
            "STAGE_10B6B_EVALUATE_"
            "FROZEN_BLIND_SUBMISSION"
        ),

        "status": (
            "PASS_FROZEN_BLIND_"
            "SUBMISSION_EVALUATION"
        ),

        "evaluation_contract": {
            "submission_frozen_before_gt": True,
            "submission_sha256": frozen_sha,
            "submission_sha_unchanged":
                bool(
                    sha_before
                    == sha_after
                    == frozen_sha
                ),
            "reference_used_for_localization":
                False,
            "reference_used_for_evaluation":
                True,
            "evaluation_only": True,
            "reference_coordinate_source":
                "eval_ref_lat/eval_ref_lon",
            "reference_projected_to":
                args.map_crs,
            "eval_ref_x_enu_used":
                False,
        },

        "continuous_fused_trajectory":
            continuous_metrics,

        "drift_time_metrics":
            drift_summary,

        "threshold_sensitivity":
            threshold_sensitivity,

        "accepted_correction_safety":
            correction_safety,

        "map_lock_evaluation":
            map_lock_eval,

        "accepted_correction_evaluation":
            correction_eval,

        "runtime": {
            "output_write_s": float(
                write_s
            ),
            "total_stage_wall_s": float(
                total_s
            ),
        },

        "outputs": {
            "trajectory_evaluation":
                str(
                    trajectory_eval_path
                ),
            "accepted_correction_evaluation":
                str(
                    correction_eval_path
                ),

            "drift_time_summary_json":
                str(
                    drift_json_path
                ),

            "drift_time_summary_csv":
                str(
                    drift_csv_path
                ),

            "error_vs_time_figure":
                str(
                    error_vs_time_path
                ),

            "error_vs_distance_figure":
                str(
                    error_vs_distance_path
                ),

            "threshold_sensitivity_json":
                str(
                    threshold_json_path
                ),

            "threshold_sensitivity_csv":
                str(
                    threshold_csv_path
                ),

            "accepted_correction_safety_json":
                str(
                    safety_json_path
                ),

            "accepted_correction_safety_csv":
                str(
                    safety_csv_path
                ),

            "summary":
                str(
                    summary_path
                ),
        },
    }

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            default=json_safe,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    print("=" * 88)
    print(
        "STAGE 10B.6B — FROZEN BLIND "
        "SUBMISSION EVALUATION"
    )
    print("=" * 88)

    print()
    print("Freeze integrity")
    print("-" * 88)

    print(
        "frozen SHA:",
        frozen_sha,
    )

    print(
        "SHA unchanged:",
        sha_before
        == sha_after
        == frozen_sha,
    )

    print()
    print("Continuous fused trajectory")
    print("-" * 88)

    print(
        "evaluated poses :",
        continuous_metrics[
            "count"
        ],
    )

    print(
        "RMSE            :",
        f"{continuous_metrics['rmse_m']:.3f} m",
    )

    print(
        "mean            :",
        f"{continuous_metrics['mean_m']:.3f} m",
    )

    print(
        "median          :",
        f"{continuous_metrics['median_m']:.3f} m",
    )

    print(
        "p95             :",
        f"{continuous_metrics['p95_m']:.3f} m",
    )

    print(
        "max             :",
        f"{continuous_metrics['max_m']:.3f} m",
    )

    print(
        "first error     :",
        f"{continuous_metrics['first_available_error_m']:.3f} m",
    )

    print(
        "final error     :",
        f"{continuous_metrics['final_error_m']:.3f} m",
    )


    print()
    print("Drift / time-distance")
    print("-" * 88)

    print(
        "evaluation duration :",
        f"{evaluation_duration_s:.3f} s",
    )

    print(
        "evaluation distance :",
        f"{evaluation_distance_m:.3f} m",
    )

    print(
        "final drift from lock:",
        f"{final_drift_from_lock_m:.3f} m",
    )

    print(
        "drift / 100 m       :",
        f"{drift_m_per_100m:.6f} m/100m",
    )

    print(
        "drift / minute      :",
        f"{drift_m_per_min:.6f} m/min",
    )

    print(
        "drift / second      :",
        f"{drift_m_per_s:.6f} m/s",
    )

    print()
    print("Reference motion / rate consistency")
    print("-" * 88)

    print(
        "average ground speed:",
        f"{average_reference_speed_mps:.6f} m/s",
    )

    print(
        "average ground speed:",
        f"{average_reference_speed_kmh:.3f} km/h",
    )

    print(
        "distance / minute   :",
        f"{average_reference_speed_m_per_min:.3f} m/min",
    )

    print(
        "time / 100 m        :",
        f"{seconds_per_100m:.3f} s/100m",
    )

    print(
        "median step speed   :",
        f"{median_reference_step_speed_mps:.6f} m/s",
    )

    print(
        "p95 step speed      :",
        f"{p95_reference_step_speed_mps:.6f} m/s",
    )

    print(
        "max step speed      :",
        f"{max_reference_step_speed_mps:.6f} m/s",
    )

    print()
    print(
        "cross-check         : "
        f"{drift_m_per_100m:.6f} m/100m"
        " × "
        f"{average_reference_speed_m_per_min:.3f} m/min"
        " / 100"
    )

    print(
        "derived drift/min   :",
        f"{drift_m_per_min_from_distance_rate:.6f} m/min",
    )

    print(
        "direct drift/min    :",
        f"{drift_m_per_min:.6f} m/min",
    )

    print(
        "rate difference     :",
        f"{drift_rate_crosscheck_abs_diff:.12f}",
    )

    print()
    print("Absolute-error threshold crossings")
    print("-" * 88)

    for crossing in absolute_crossings:
        threshold = int(
            crossing[
                "threshold_m"
            ]
        )

        if crossing[
            "crossed"
        ]:
            print(
                f"{threshold:2d} m : "
                f"frame "
                f"{crossing['sequence_frame_id']}, "
                f"+{crossing['elapsed_since_map_lock_s']:.3f} s, "
                f"+{crossing['distance_since_map_lock_m']:.3f} m, "
                f"error={crossing['value_at_crossing_m']:.3f} m"
            )
        else:
            print(
                f"{threshold:2d} m : not crossed"
            )

    print()
    print("Drift-from-lock threshold crossings")
    print("-" * 88)

    for crossing in drift_crossings:
        threshold = int(
            crossing[
                "threshold_m"
            ]
        )

        if crossing[
            "crossed"
        ]:
            print(
                f"{threshold:2d} m : "
                f"frame "
                f"{crossing['sequence_frame_id']}, "
                f"+{crossing['elapsed_since_map_lock_s']:.3f} s, "
                f"+{crossing['distance_since_map_lock_m']:.3f} m, "
                f"drift={crossing['value_at_crossing_m']:.3f} m"
            )
        else:
            print(
                f"{threshold:2d} m : not crossed"
            )

    print()
    print("Map lock")
    print("-" * 88)

    print(
        "frame           :",
        map_lock_eval[
            "frame"
        ],
    )

    print(
        "time            :",
        f"{map_lock_eval['time_s']:.3f} s",
    )

    print(
        "lock error      :",
        f"{map_lock_eval['position_error_m']:.3f} m",
    )


    print()
    print("Threshold sensitivity — evaluation only")
    print("-" * 88)

    print(
        "retrieval/reranker population: 403 queries"
    )

    print(
        "accepted-correction population:",
        threshold_methods[
            "accepted_corrections"
        ][
            "population"
        ],
    )

    method_order = [
        "dino_top1",
        "dino_top5",
        "dino_top10",
        "dino_top20",
        "dino_top100",
        "orb_selected",
        "lightglue_selected",
        "accepted_corrections",
    ]

    header = (
        f"{'threshold':>10}"
        + "".join(
            f"{name:>22}"
            for name in method_order
        )
    )

    print()
    print(header)

    for threshold in thresholds_eval_m:

        tag = f"{threshold:g}"

        row = (
            f"{threshold:>8.0f} m"
        )

        for method in method_order:

            count = (
                threshold_methods[
                    method
                ][
                    "counts"
                ][
                    tag
                ]
            )

            population = (
                threshold_methods[
                    method
                ][
                    "population"
                ]
            )

            row += (
                f"{count:>10}/{population:<11}"
            )

        print(row)

    print()
    print("Accepted absolute corrections")
    print("-" * 88)

    print(
        "accepted        :",
        correction_eval[
            "accepted_count"
        ],
    )

    print(
        "<= 10 m         :",
        correction_eval[
            "absolute_candidate_le10m"
        ],
    )

    print(
        "<= 20 m         :",
        correction_eval[
            "absolute_candidate_le20m"
        ],
    )

    print(
        "<= 40 m         :",
        correction_eval[
            "absolute_candidate_le40m"
        ],
    )

    print(
        "> 40 m          :",
        correction_eval[
            "absolute_candidate_gt40m"
        ],
    )

    print(
        "> 100 m         :",
        correction_eval[
            "absolute_candidate_gt100m"
        ],
    )

    acm = correction_eval[
        "absolute_candidate_metrics"
    ]

    print(
        "candidate median:",
        f"{acm['median_m']:.3f} m",
    )

    print(
        "candidate p95   :",
        f"{acm['p95_m']:.3f} m",
    )

    print(
        "candidate max   :",
        f"{acm['max_m']:.3f} m",
    )


    print()
    print("Blind correction safety — evaluation only")
    print("-" * 88)

    print(
        "candidate corrections :",
        candidate_total,
    )

    print(
        "accepted corrections  :",
        accepted_total,
    )

    print(
        "rejected corrections  :",
        rejected_total,
    )

    print(
        "accepted <= 10 m      :",
        accepted_le10,
    )

    print(
        "accepted <= 20 m      :",
        accepted_le20,
    )

    print(
        "accepted <= 40 m      :",
        accepted_le40,
    )

    print(
        "accepted > 40 m       :",
        accepted_gt40,
    )

    print(
        "accepted > 100 m      :",
        accepted_gt100,
    )

    print(
        "precision <=40 m      :",
        f"{accepted_le40 / accepted_total:.6f}",
    )

    print(
        "precision <=100 m     :",
        f"{accepted_le100 / accepted_total:.6f}",
    )

    print(
        "rejection reasons     :",
        rejection_reason_counts,
    )

    print()
    print("Evaluation contract")
    print("-" * 88)

    print(
        "GT used for localization : false"
    )

    print(
        "GT used for evaluation   : true"
    )

    print(
        "eval_ref_x/y ENU used    : false"
    )

    print()
    print(
        "status: "
        "PASS_FROZEN_BLIND_SUBMISSION_EVALUATION"
    )


if __name__ == "__main__":
    main()
