'''
ROOT=outputs/villoc/traj01_90deg_stable120m
RUN="$ROOT/blind_demo_addons/eval_traj01_known_gt"

python scripts/villoc/blind_demo/addon8_optional_evaluation_attachment.py \
  --blind-manifest \
  "$RUN/metadata/blind_query_manifest.csv" \
  --reference-csv \
  "$ROOT/trajectories/s8_3_reference_trajectory_V_1fps.csv" \
  --run-root "$RUN" \
  --max-time-delta-s 0.05 \
  2>&1 | tee \
  "$RUN/logs/addon8_optional_evaluation_attachment.log"
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FORBIDDEN_BLIND_COLUMNS = {
    "lat",
    "lon",
    "latitude",
    "longitude",
    "x_enu_m",
    "y_enu_m",
    "reference_x_m",
    "reference_y_m",
    "reference_cumulative_distance_m",
    "gps_lat",
    "gps_lon",
    "oracle",
    "error_m",
    "fusion_error_m",
    "ground_truth_x",
    "ground_truth_y",
}


FORBIDDEN_ESTIMATE_COLUMNS = {
    "reference_x_m",
    "reference_y_m",
    "reference_cumulative_distance_m",
    "fusion_error_m",
    "abs_error_m_eval_only",
    "abs_hit_le_40m_eval_only",
    "ground_truth_x",
    "ground_truth_y",
    "gt_x",
    "gt_y",
}


REFERENCE_EXPORT_COLUMNS = [
    "lat",
    "lon",
    "x_enu_m",
    "y_enu_m",
    "z_enu_m",
    "rel_alt_m",
    "abs_alt_m",
    "gb_yaw_deg",
    "gb_pitch_deg",
    "gb_roll_deg",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(
                1024 * 1024
            ),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def require(
    condition: bool,
    message: str,
) -> None:
    if not condition:
        raise RuntimeError(message)


def false_series(
    series: pd.Series,
) -> np.ndarray:
    out = []

    for value in series:
        if isinstance(
            value,
            (bool, np.bool_),
        ):
            out.append(
                bool(value)
            )
            continue

        text = str(
            value
        ).strip().lower()

        if text in {
            "false",
            "0",
            "no",
        }:
            out.append(False)

        elif text in {
            "true",
            "1",
            "yes",
        }:
            out.append(True)

        else:
            raise RuntimeError(
                "Unrecognized boolean value: "
                f"{value!r}"
            )

    return np.asarray(
        out,
        dtype=bool,
    )


def json_safe(
    value: Any,
) -> Any:
    if isinstance(
        value,
        Path,
    ):
        return str(value)

    if isinstance(
        value,
        np.generic,
    ):
        return value.item()

    if isinstance(
        value,
        float,
    ):
        if not math.isfinite(
            value
        ):
            return None

    if pd.isna(value):
        return None

    return value


def haversine_m(
    lat1: np.ndarray,
    lon1: np.ndarray,
    lat2: np.ndarray,
    lon2: np.ndarray,
) -> np.ndarray:
    radius = 6378137.0

    lat1_rad = np.radians(
        lat1
    )

    lon1_rad = np.radians(
        lon1
    )

    lat2_rad = np.radians(
        lat2
    )

    lon2_rad = np.radians(
        lon2
    )

    dlat = (
        lat2_rad
        - lat1_rad
    )

    dlon = (
        lon2_rad
        - lon1_rad
    )

    a = (
        np.sin(
            dlat / 2.0
        ) ** 2
        + np.cos(
            lat1_rad
        )
        * np.cos(
            lat2_rad
        )
        * np.sin(
            dlon / 2.0
        ) ** 2
    )

    return (
        2.0
        * radius
        * np.arctan2(
            np.sqrt(a),
            np.sqrt(
                1.0 - a
            ),
        )
    )


def error_metrics(
    errors: np.ndarray,
) -> dict[str, Any]:
    errors = errors[
        np.isfinite(errors)
    ]

    if len(errors) == 0:
        return {}

    return {
        "count": int(
            len(errors)
        ),
        "rmse_m": float(
            math.sqrt(
                np.mean(
                    errors * errors
                )
            )
        ),
        "mean_error_m": float(
            np.mean(errors)
        ),
        "median_error_m": float(
            np.median(errors)
        ),
        "p95_error_m": float(
            np.percentile(
                errors,
                95,
            )
        ),
        "max_error_m": float(
            np.max(errors)
        ),
        "final_error_m": float(
            errors[-1]
        ),
        "count_le_10m": int(
            (errors <= 10).sum()
        ),
        "count_le_20m": int(
            (errors <= 20).sum()
        ),
        "count_le_40m": int(
            (errors <= 40).sum()
        ),
        "count_gt_100m": int(
            (errors > 100).sum()
        ),
    }


def save_error_plot(
    table: pd.DataFrame,
    output_path: Path,
) -> None:
    valid = table[
        pd.to_numeric(
            table[
                "eval_position_error_m"
            ],
            errors="coerce",
        ).notna()
    ].copy()

    if valid.empty:
        return

    fig, ax = plt.subplots(
        figsize=(11, 5)
    )

    ax.plot(
        valid["timestamp_s"],
        valid[
            "eval_position_error_m"
        ],
    )

    for threshold in [
        10,
        20,
        40,
        100,
    ]:
        ax.axhline(
            threshold,
            linestyle="--",
            linewidth=1,
            alpha=0.45,
        )

    ax.set_xlabel(
        "Video time [s]"
    )

    ax.set_ylabel(
        "Position error [m]"
    )

    ax.set_title(
        "Optional evaluation attachment — "
        "position error"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    fig.tight_layout()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Add-on 8: attach optional reference "
            "data to already-frozen blind outputs."
        )
    )

    parser.add_argument(
        "--blind-manifest",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--reference-csv",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--run-root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--blind-time-col",
        default="timestamp_s",
    )

    parser.add_argument(
        "--reference-time-col",
        default="target_time_s",
    )

    parser.add_argument(
        "--max-time-delta-s",
        type=float,
        default=0.05,
    )

    # Optional genuinely blind-safe estimate.
    parser.add_argument(
        "--estimate-csv",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--estimate-id-col",
        default="query_id",
    )

    parser.add_argument(
        "--estimate-time-col",
        default="timestamp_s",
    )

    parser.add_argument(
        "--estimate-lat-col",
        default="estimated_lat",
    )

    parser.add_argument(
        "--estimate-lon-col",
        default="estimated_lon",
    )

    parser.add_argument(
        "--reference-lat-col",
        default="lat",
    )

    parser.add_argument(
        "--reference-lon-col",
        default="lon",
    )

    args = parser.parse_args()

    stage_started = (
        time.perf_counter()
    )

    blind_path = (
        args.blind_manifest.resolve()
    )

    reference_path = (
        args.reference_csv.resolve()
    )

    run_root = (
        args.run_root.resolve()
    )

    require(
        blind_path.exists(),
        f"Missing blind manifest: {blind_path}",
    )

    require(
        reference_path.exists(),
        f"Missing reference CSV: {reference_path}",
    )

    if (
        args.estimate_csv
        is not None
    ):
        estimate_path = (
            args.estimate_csv.resolve()
        )

        require(
            estimate_path.exists(),
            (
                "Missing estimate CSV: "
                f"{estimate_path}"
            ),
        )

    else:
        estimate_path = None

    evaluation_dir = (
        run_root
        / "evaluation"
    )

    figures_dir = (
        run_root
        / "figures"
        / "evaluation"
    )

    evaluation_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    figures_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    attachment_path = (
        evaluation_dir
        / "reference_attachment.csv"
    )

    summary_path = (
        evaluation_dir
        / "evaluation_summary.json"
    )

    error_plot_path = (
        figures_dir
        / "evaluation_error_vs_time.png"
    )

    # =====================================================
    # Freeze evidence BEFORE reading reference
    # =====================================================

    blind_sha_before = (
        sha256_file(
            blind_path
        )
    )

    estimate_sha_before = (
        sha256_file(
            estimate_path
        )
        if estimate_path
        is not None
        else None
    )

    # =====================================================
    # Blind input validation
    # =====================================================

    blind_read_started = (
        time.perf_counter()
    )

    blind = pd.read_csv(
        blind_path
    )

    blind_read_finished = (
        time.perf_counter()
    )

    require(
        len(blind) > 0,
        "Blind manifest is empty.",
    )

    require(
        args.blind_time_col
        in blind.columns,
        (
            "Blind manifest missing time column: "
            f"{args.blind_time_col}"
        ),
    )

    require(
        "reference_available"
        in blind.columns,
        (
            "Blind manifest missing "
            "reference_available."
        ),
    )

    require(
        not false_series(
            blind[
                "reference_available"
            ]
        ).any(),
        (
            "Blind manifest contains "
            "reference_available=true."
        ),
    )

    leaked_blind = sorted(
        FORBIDDEN_BLIND_COLUMNS
        & set(
            blind.columns
        )
    )

    require(
        not leaked_blind,
        (
            "Reference/evaluation columns leaked "
            "into blind input: "
            f"{leaked_blind}"
        ),
    )

    blind[
        "_blind_time_s"
    ] = pd.to_numeric(
        blind[
            args.blind_time_col
        ],
        errors="raise",
    )

    blind[
        "_blind_row_id"
    ] = np.arange(
        len(blind),
        dtype=int,
    )

    # =====================================================
    # Reference read — evaluation side starts HERE
    # =====================================================

    reference_read_started = (
        time.perf_counter()
    )

    reference = pd.read_csv(
        reference_path
    )

    reference_read_finished = (
        time.perf_counter()
    )

    require(
        args.reference_time_col
        in reference.columns,
        (
            "Reference CSV missing time column: "
            f"{args.reference_time_col}"
        ),
    )

    reference[
        "_eval_reference_time_s"
    ] = pd.to_numeric(
        reference[
            args.reference_time_col
        ],
        errors="raise",
    )

    # Only explicitly selected reference columns are
    # copied into the evaluation attachment.
    available_ref_cols = [
        col
        for col in (
            REFERENCE_EXPORT_COLUMNS
        )
        if col in reference.columns
    ]

    reference_subset = (
        reference[
            [
                "_eval_reference_time_s",
                *available_ref_cols,
            ]
        ]
        .copy()
    )

    reference_subset = (
        reference_subset.rename(
            columns={
                col: f"eval_ref_{col}"
                for col in available_ref_cols
            }
        )
    )

    # =====================================================
    # Timestamp alignment
    # =====================================================

    alignment_started = (
        time.perf_counter()
    )

    blind_sorted = (
        blind.sort_values(
            "_blind_time_s",
            kind="mergesort",
        )
    )

    reference_sorted = (
        reference_subset.sort_values(
            "_eval_reference_time_s",
            kind="mergesort",
        )
    )

    attached = pd.merge_asof(
        blind_sorted,
        reference_sorted,
        left_on="_blind_time_s",
        right_on=(
            "_eval_reference_time_s"
        ),
        direction="nearest",
        tolerance=float(
            args.max_time_delta_s
        ),
    )

    attached[
        "eval_reference_available"
    ] = attached[
        "_eval_reference_time_s"
    ].notna()

    attached[
        "eval_reference_time_delta_ms"
    ] = (
        (
            attached[
                "_eval_reference_time_s"
            ]
            - attached[
                "_blind_time_s"
            ]
        )
        .abs()
        * 1000.0
    )

    attached = (
        attached.sort_values(
            "_blind_row_id",
            kind="mergesort",
        )
        .reset_index(
            drop=True
        )
    )

    attached[
        "evaluation_only"
    ] = True

    matched_reference = int(
        attached[
            "eval_reference_available"
        ].sum()
    )

    alignment_finished = (
        time.perf_counter()
    )

    # =====================================================
    # Optional blind-safe estimated trajectory
    # =====================================================

    evaluation_metrics: dict[str, Any] | None = None

    estimate_mode = "none"

    error_started = (
        time.perf_counter()
    )

    if estimate_path is not None:
        estimate = pd.read_csv(
            estimate_path
        )

        leaked_estimate = sorted(
            FORBIDDEN_ESTIMATE_COLUMNS
            & set(
                estimate.columns
            )
        )

        require(
            not leaked_estimate,
            (
                "Estimate CSV is not blind-safe. "
                "Evaluation/reference columns present: "
                f"{leaked_estimate}"
            ),
        )

        estimate_columns = set(
            estimate.columns
        )

        require(
            args.estimate_lat_col
            in estimate_columns,
            (
                "Estimate CSV missing latitude: "
                f"{args.estimate_lat_col}"
            ),
        )

        require(
            args.estimate_lon_col
            in estimate_columns,
            (
                "Estimate CSV missing longitude: "
                f"{args.estimate_lon_col}"
            ),
        )

        if (
            args.estimate_id_col
            in estimate.columns
            and args.estimate_id_col
            in attached.columns
        ):
            estimate_mode = (
                "exact_query_id"
            )

            est_subset = estimate[
                [
                    args.estimate_id_col,
                    args.estimate_lat_col,
                    args.estimate_lon_col,
                ]
            ].copy()

            est_subset = (
                est_subset.rename(
                    columns={
                        args.estimate_lat_col:
                            "eval_estimated_lat",
                        args.estimate_lon_col:
                            "eval_estimated_lon",
                    }
                )
            )

            attached = attached.merge(
                est_subset,
                on=args.estimate_id_col,
                how="left",
                validate="one_to_one",
            )

        else:
            require(
                args.estimate_time_col
                in estimate.columns,
                (
                    "Estimate CSV has neither "
                    "compatible query ID nor "
                    f"{args.estimate_time_col}."
                ),
            )

            estimate_mode = (
                "nearest_timestamp"
            )

            estimate[
                "_estimate_time_s"
            ] = pd.to_numeric(
                estimate[
                    args.estimate_time_col
                ],
                errors="raise",
            )

            est_subset = estimate[
                [
                    "_estimate_time_s",
                    args.estimate_lat_col,
                    args.estimate_lon_col,
                ]
            ].copy()

            est_subset = (
                est_subset.rename(
                    columns={
                        args.estimate_lat_col:
                            "eval_estimated_lat",
                        args.estimate_lon_col:
                            "eval_estimated_lon",
                    }
                )
            )

            attached = pd.merge_asof(
                attached.sort_values(
                    "_blind_time_s"
                ),
                est_subset.sort_values(
                    "_estimate_time_s"
                ),
                left_on="_blind_time_s",
                right_on="_estimate_time_s",
                direction="nearest",
                tolerance=float(
                    args.max_time_delta_s
                ),
            )

            attached = (
                attached.sort_values(
                    "_blind_row_id"
                )
                .reset_index(
                    drop=True
                )
            )

        require(
            (
                f"eval_ref_{args.reference_lat_col}"
            )
            in attached.columns,
            (
                "Attached reference does not "
                "contain reference latitude."
            ),
        )

        require(
            (
                f"eval_ref_{args.reference_lon_col}"
            )
            in attached.columns,
            (
                "Attached reference does not "
                "contain reference longitude."
            ),
        )

        est_lat = pd.to_numeric(
            attached[
                "eval_estimated_lat"
            ],
            errors="coerce",
        ).to_numpy(
            dtype=float
        )

        est_lon = pd.to_numeric(
            attached[
                "eval_estimated_lon"
            ],
            errors="coerce",
        ).to_numpy(
            dtype=float
        )

        ref_lat = pd.to_numeric(
            attached[
                f"eval_ref_{args.reference_lat_col}"
            ],
            errors="coerce",
        ).to_numpy(
            dtype=float
        )

        ref_lon = pd.to_numeric(
            attached[
                f"eval_ref_{args.reference_lon_col}"
            ],
            errors="coerce",
        ).to_numpy(
            dtype=float
        )

        valid = (
            np.isfinite(
                est_lat
            )
            & np.isfinite(
                est_lon
            )
            & np.isfinite(
                ref_lat
            )
            & np.isfinite(
                ref_lon
            )
        )

        errors = np.full(
            len(attached),
            np.nan,
            dtype=float,
        )

        errors[
            valid
        ] = haversine_m(
            est_lat[valid],
            est_lon[valid],
            ref_lat[valid],
            ref_lon[valid],
        )

        attached[
            "eval_position_error_m"
        ] = errors

        evaluation_metrics = (
            error_metrics(
                errors
            )
        )

        save_error_plot(
            attached,
            error_plot_path,
        )

    error_finished = (
        time.perf_counter()
    )

    # Drop internal merge helpers.
    attached = attached.drop(
        columns=[
            col
            for col in [
                "_blind_time_s",
                "_blind_row_id",
                "_eval_reference_time_s",
                "_estimate_time_s",
            ]
            if col in attached.columns
        ]
    )

    attached.to_csv(
        attachment_path,
        index=False,
    )

    # =====================================================
    # Prove blind artifacts were unchanged
    # =====================================================

    blind_sha_after = (
        sha256_file(
            blind_path
        )
    )

    estimate_sha_after = (
        sha256_file(
            estimate_path
        )
        if estimate_path
        is not None
        else None
    )

    require(
        blind_sha_before
        == blind_sha_after,
        (
            "Blind manifest changed during "
            "evaluation attachment."
        ),
    )

    if estimate_path is not None:
        require(
            estimate_sha_before
            == estimate_sha_after,
            (
                "Blind estimate changed during "
                "evaluation attachment."
            ),
        )

    stage_finished = (
        time.perf_counter()
    )

    if estimate_path is None:
        status = (
            "PASS_REFERENCE_ATTACHMENT_ONLY"
        )

        error_reason = (
            "No genuinely blind-safe estimated "
            "trajectory was supplied. Reference "
            "attachment is validated now; GT error "
            "metrics will activate automatically "
            "when a frozen blind estimate is supplied."
        )

    else:
        status = (
            "PASS_OPTIONAL_EVALUATION_WITH_ERRORS"
        )

        error_reason = None

    summary = {
        "stage": (
            "ADDON8_OPTIONAL_EVALUATION_ATTACHMENT"
        ),
        "status": status,
        "evaluation_only": True,
        "one_way_boundary": {
            "reference_available_to_localization": False,
            "reference_read_only_after_blind_input_hash": True,
            "blind_manifest_modified": False,
            "blind_estimate_modified": False,
            "reference_never_written_to_blind_manifest": True,
        },
        "inputs": {
            "blind_manifest": str(
                blind_path
            ),
            "blind_manifest_sha256": (
                blind_sha_before
            ),
            "reference_csv": str(
                reference_path
            ),
            "reference_csv_sha256": (
                sha256_file(
                    reference_path
                )
            ),
            "estimate_csv": (
                str(
                    estimate_path
                )
                if estimate_path
                is not None
                else None
            ),
            "estimate_sha256": (
                estimate_sha_before
            ),
        },
        "alignment": {
            "method": (
                "nearest_timestamp"
            ),
            "max_time_delta_s": float(
                args.max_time_delta_s
            ),
            "blind_rows": int(
                len(blind)
            ),
            "reference_rows": int(
                len(reference)
            ),
            "matched_reference_rows": (
                matched_reference
            ),
            "unmatched_reference_rows": int(
                len(blind)
                - matched_reference
            ),
            "match_fraction": float(
                matched_reference
                / len(blind)
            ),
            "median_time_delta_ms": (
                float(
                    attached[
                        "eval_reference_time_delta_ms"
                    ].median()
                )
            ),
            "max_time_delta_ms": (
                float(
                    attached[
                        "eval_reference_time_delta_ms"
                    ].max()
                )
            ),
        },
        "estimate_evaluation": {
            "estimate_supplied": (
                estimate_path
                is not None
            ),
            "estimate_alignment_mode": (
                estimate_mode
            ),
            "metrics_available": (
                evaluation_metrics
                is not None
            ),
            "metrics": (
                evaluation_metrics
            ),
            "reason_if_unavailable": (
                error_reason
            ),
        },
        "runtime": {
            "blind_manifest_read_s": float(
                blind_read_finished
                - blind_read_started
            ),
            "reference_read_s": float(
                reference_read_finished
                - reference_read_started
            ),
            "reference_alignment_s": float(
                alignment_finished
                - alignment_started
            ),
            "error_metrics_s": float(
                error_finished
                - error_started
            ),
            "total_stage_wall_s": float(
                stage_finished
                - stage_started
            ),
        },
        "outputs": {
            "reference_attachment_csv": (
                str(
                    attachment_path
                )
            ),
            "evaluation_summary_json": (
                str(
                    summary_path
                )
            ),
            "error_plot": (
                str(
                    error_plot_path
                )
                if evaluation_metrics
                is not None
                else None
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

    print("=" * 80)
    print(
        "STAGE 9A — ADD-ON 8 "
        "OPTIONAL EVALUATION ATTACHMENT"
    )
    print("=" * 80)

    print(
        "status:",
        status,
    )

    print()
    print("One-way boundary")
    print("-" * 80)

    print(
        "reference available to localization : false"
    )

    print(
        "blind manifest modified             : false"
    )

    print(
        "blind estimate modified             : false"
    )

    print(
        "blind manifest SHA unchanged        :",
        blind_sha_before
        == blind_sha_after,
    )

    print()
    print("Reference alignment")
    print("-" * 80)

    print(
        f"blind rows               : "
        f"{len(blind)}"
    )

    print(
        f"reference rows           : "
        f"{len(reference)}"
    )

    print(
        f"matched rows             : "
        f"{matched_reference}"
    )

    print(
        f"match fraction           : "
        f"{matched_reference / len(blind):.6f}"
    )

    print(
        "median time delta        : "
        f"{attached['eval_reference_time_delta_ms'].median():.3f} ms"
    )

    print(
        "max time delta           : "
        f"{attached['eval_reference_time_delta_ms'].max():.3f} ms"
    )

    print()
    print("Estimate evaluation")
    print("-" * 80)

    print(
        "blind estimate supplied  :",
        estimate_path
        is not None,
    )

    print(
        "GT metrics available     :",
        evaluation_metrics
        is not None,
    )

    if evaluation_metrics:
        print(
            json.dumps(
                evaluation_metrics,
                indent=2,
            )
        )

    else:
        print(
            "reason:",
            error_reason,
        )

    print()
    print("Runtime")
    print("-" * 80)

    for key, value in (
        summary["runtime"].items()
    ):
        print(
            f"{key:28s}: "
            f"{value:.6f} s"
        )

    print()
    print("Saved")
    print("-" * 80)

    print(attachment_path)
    print(summary_path)

    if evaluation_metrics:
        print(error_plot_path)


if __name__ == "__main__":
    main()
