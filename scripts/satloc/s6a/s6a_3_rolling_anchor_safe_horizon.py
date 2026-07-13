'''
Command Used:

export PYTHONPATH=$PWD/src

python scripts/satloc/s6a/s6a_3_rolling_anchor_safe_horizon.py \
  --variant se2_scale_normalized \
  --thresholds-m 10,20,40,80 \
  --anchor-step-frames 25 \
  --min-remaining-distance-m 2000 \
  --sustain-frames 5 \
  --heading-lookback-frames 10 \
  --representative-threshold-m 20
  
'''

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_ALIGNED_CSV = Path(
    "outputs/satloc/metadata/s6a_relative_motion/"
    "s6a2_orb_relative_trajectory_aligned_eval_only.csv"
)
DEFAULT_S6A2_REPORT = Path(
    "outputs/satloc/reports/s6a_relative_motion/"
    "s6a2_orb_relative_trajectory_and_drift.json"
)
DEFAULT_OUTPUT_ROOT = Path("outputs/satloc")


def parse_float_list(value: str) -> list[float]:
    values: list[float] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        number = float(item)
        if number <= 0:
            raise argparse.ArgumentTypeError(
                "Error thresholds must be positive."
            )
        values.append(number)

    if not values:
        raise argparse.ArgumentTypeError(
            "At least one threshold is required."
        )

    return sorted(set(values))


def bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)

    normalized = series.astype(str).str.strip().str.lower()
    return normalized.isin({"true", "1", "yes", "y", "t"})


def rotation_matrix(angle_rad: float) -> np.ndarray:
    cosine = math.cos(angle_rad)
    sine = math.sin(angle_rad)
    return np.array(
        [[cosine, -sine], [sine, cosine]],
        dtype=float,
    )


def angle_of(vector: np.ndarray) -> float:
    return float(math.atan2(vector[1], vector[0]))


def wrap_angle_rad(angle_rad: float) -> float:
    return float(
        (angle_rad + math.pi) % (2.0 * math.pi) - math.pi
    )


def first_sustained_crossing(
    errors: np.ndarray,
    threshold_m: float,
    sustain_frames: int,
) -> int | None:
    above = np.isfinite(errors) & (errors >= threshold_m)

    if len(above) < sustain_frames:
        return None

    for index in range(
        1,
        len(above) - sustain_frames + 1,
    ):
        if bool(np.all(above[index : index + sustain_frames])):
            return int(index)

    return None


def percentile_or_nan(
    values: pd.Series | np.ndarray,
    percentile: float,
) -> float:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return float("nan")
    return float(np.percentile(array, percentile))


def median_or_nan(
    values: pd.Series | np.ndarray,
) -> float:
    return percentile_or_nan(values, 50.0)


def build_anchor_indices(
    frame_count: int,
    start_anchor: int,
    anchor_step: int,
    cumulative_distance: np.ndarray,
    min_remaining_distance_m: float,
) -> list[int]:
    anchors: list[int] = []

    for anchor in range(
        start_anchor,
        frame_count - 1,
        anchor_step,
    ):
        remaining_distance = float(
            cumulative_distance[-1]
            - cumulative_distance[anchor]
        )

        if remaining_distance < min_remaining_distance_m:
            continue

        anchors.append(anchor)

    return anchors


def position_only_corrected_future(
    visual: np.ndarray,
    reference: np.ndarray,
    anchor: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    offset = reference[anchor] - visual[anchor]
    corrected = visual[anchor:] + offset

    return corrected, {
        "heading_reset_applied": False,
        "heading_correction_deg": 0.0,
        "heading_window_start": None,
        "heading_window_frames_used": 0,
    }


def position_heading_corrected_future(
    visual: np.ndarray,
    reference: np.ndarray,
    anchor: int,
    heading_lookback_frames: int,
    min_heading_baseline_m: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    start = max(0, anchor - heading_lookback_frames)

    visual_vector = visual[anchor] - visual[start]
    reference_vector = reference[anchor] - reference[start]

    visual_norm = float(np.linalg.norm(visual_vector))
    reference_norm = float(np.linalg.norm(reference_vector))

    if (
        visual_norm < min_heading_baseline_m
        or reference_norm < min_heading_baseline_m
    ):
        corrected, metadata = position_only_corrected_future(
            visual,
            reference,
            anchor,
        )
        metadata.update(
            {
                "heading_reset_fallback_reason": (
                    "insufficient_heading_baseline"
                ),
                "heading_window_start": int(start),
                "heading_window_frames_used": int(anchor - start),
                "visual_heading_baseline_m": visual_norm,
                "reference_heading_baseline_m": reference_norm,
            }
        )
        return corrected, metadata

    correction_angle = wrap_angle_rad(
        angle_of(reference_vector)
        - angle_of(visual_vector)
    )
    rotation = rotation_matrix(correction_angle)

    future_relative = visual[anchor:] - visual[anchor]
    corrected = (
        future_relative @ rotation.T
        + reference[anchor]
    )

    return corrected, {
        "heading_reset_applied": True,
        "heading_correction_deg": float(
            math.degrees(correction_angle)
        ),
        "heading_window_start": int(start),
        "heading_window_frames_used": int(anchor - start),
        "visual_heading_baseline_m": visual_norm,
        "reference_heading_baseline_m": reference_norm,
        "heading_reset_fallback_reason": None,
    }


def evaluate_anchor_mode(
    mode: str,
    visual: np.ndarray,
    reference: np.ndarray,
    cumulative_distance: np.ndarray,
    safe_flags: np.ndarray,
    anchor: int,
    thresholds_m: list[float],
    sustain_frames: int,
    heading_lookback_frames: int,
    min_heading_baseline_m: float,
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    if mode == "position_only":
        corrected, correction_metadata = (
            position_only_corrected_future(
                visual,
                reference,
                anchor,
            )
        )
    elif mode == "position_heading_reset_eval_only":
        corrected, correction_metadata = (
            position_heading_corrected_future(
                visual,
                reference,
                anchor,
                heading_lookback_frames,
                min_heading_baseline_m,
            )
        )
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    reference_future = reference[anchor:]
    errors = np.linalg.norm(
        corrected - reference_future,
        axis=1,
    )

    distance_after_anchor = (
        cumulative_distance[anchor:]
        - cumulative_distance[anchor]
    )
    available_distance = float(distance_after_anchor[-1])
    available_frames = int(len(errors) - 1)

    curve = pd.DataFrame(
        {
            "mode": mode,
            "anchor_frame_index": int(anchor),
            "future_offset_frames": np.arange(
                len(errors),
                dtype=int,
            ),
            "absolute_frame_index": np.arange(
                anchor,
                anchor + len(errors),
                dtype=int,
            ),
            "distance_after_anchor_m": distance_after_anchor,
            "position_error_m": errors,
        }
    )

    results: list[dict[str, Any]] = []

    for threshold_m in thresholds_m:
        crossing_offset = first_sustained_crossing(
            errors,
            threshold_m,
            sustain_frames,
        )

        if crossing_offset is None:
            crossed = False
            horizon_frames = available_frames
            horizon_distance_m = available_distance
            crossing_frame_index = None
            error_at_crossing_m = None
            unsafe_before_horizon = int(
                np.sum(~safe_flags[anchor + 1 :])
            )
        else:
            crossed = True
            horizon_frames = int(crossing_offset)
            horizon_distance_m = float(
                distance_after_anchor[crossing_offset]
            )
            crossing_frame_index = int(
                anchor + crossing_offset
            )
            error_at_crossing_m = float(
                errors[crossing_offset]
            )
            unsafe_before_horizon = int(
                np.sum(
                    ~safe_flags[
                        anchor + 1 : crossing_frame_index + 1
                    ]
                )
            )

        results.append(
            {
                "mode": mode,
                "anchor_frame_index": int(anchor),
                "anchor_reference_x_m": float(
                    reference[anchor, 0]
                ),
                "anchor_reference_y_m": float(
                    reference[anchor, 1]
                ),
                "anchor_reference_distance_m": float(
                    cumulative_distance[anchor]
                ),
                "threshold_m": float(threshold_m),
                "sustain_frames": int(sustain_frames),
                "crossed": bool(crossed),
                "crossing_frame_index": crossing_frame_index,
                "safe_horizon_frames": int(horizon_frames),
                "safe_horizon_distance_m": float(
                    horizon_distance_m
                ),
                "available_frames": available_frames,
                "available_distance_m": available_distance,
                "error_at_crossing_m": error_at_crossing_m,
                "unsafe_pairs_before_horizon": (
                    unsafe_before_horizon
                ),
                **correction_metadata,
            }
        )

    return results, curve


def summarize_results(
    results_df: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for (mode, threshold), group in results_df.groupby(
        ["mode", "threshold_m"],
        sort=True,
    ):
        crossed = group[group["crossed"]].copy()
        censored = group[~group["crossed"]].copy()

        distance_values = crossed["safe_horizon_distance_m"]
        frame_values = crossed["safe_horizon_frames"]

        p10_distance = percentile_or_nan(
            distance_values,
            10.0,
        )
        median_distance = percentile_or_nan(
            distance_values,
            50.0,
        )
        p90_distance = percentile_or_nan(
            distance_values,
            90.0,
        )

        p10_frames = percentile_or_nan(
            frame_values,
            10.0,
        )
        median_frames = percentile_or_nan(
            frame_values,
            50.0,
        )
        p90_frames = percentile_or_nan(
            frame_values,
            90.0,
        )

        enough_crossings = len(crossed) >= 5
        conservative_distance = (
            p10_distance
            if enough_crossings
            else float("nan")
        )
        conservative_frames = (
            p10_frames
            if enough_crossings
            else float("nan")
        )

        rows.append(
            {
                "mode": mode,
                "threshold_m": float(threshold),
                "anchors": int(len(group)),
                "crossed_anchors": int(len(crossed)),
                "censored_anchors": int(len(censored)),
                "crossing_rate": float(
                    len(crossed) / max(len(group), 1)
                ),
                "safe_distance_min_m_crossed": (
                    float(distance_values.min())
                    if len(crossed)
                    else float("nan")
                ),
                "safe_distance_p10_m_crossed": p10_distance,
                "safe_distance_median_m_crossed": (
                    median_distance
                ),
                "safe_distance_p90_m_crossed": p90_distance,
                "safe_distance_max_m_crossed": (
                    float(distance_values.max())
                    if len(crossed)
                    else float("nan")
                ),
                "safe_frames_min_crossed": (
                    float(frame_values.min())
                    if len(crossed)
                    else float("nan")
                ),
                "safe_frames_p10_crossed": p10_frames,
                "safe_frames_median_crossed": median_frames,
                "safe_frames_p90_crossed": p90_frames,
                "safe_frames_max_crossed": (
                    float(frame_values.max())
                    if len(crossed)
                    else float("nan")
                ),
                "observed_or_censored_distance_p10_m": (
                    percentile_or_nan(
                        group["safe_horizon_distance_m"],
                        10.0,
                    )
                ),
                "observed_or_censored_distance_median_m": (
                    median_or_nan(
                        group["safe_horizon_distance_m"]
                    )
                ),
                "conservative_safe_horizon_m": (
                    conservative_distance
                ),
                "conservative_safe_horizon_frames": (
                    conservative_frames
                ),
                "suggested_absolute_search_start_m": (
                    0.60 * conservative_distance
                    if np.isfinite(conservative_distance)
                    else float("nan")
                ),
                "suggested_acceptance_deadline_m": (
                    0.85 * conservative_distance
                    if np.isfinite(conservative_distance)
                    else float("nan")
                ),
                "suggested_absolute_search_start_frames": (
                    0.60 * conservative_frames
                    if np.isfinite(conservative_frames)
                    else float("nan")
                ),
                "suggested_acceptance_deadline_frames": (
                    0.85 * conservative_frames
                    if np.isfinite(conservative_frames)
                    else float("nan")
                ),
                "heuristic_note": (
                    "Search start = 60% and desired acceptance "
                    "deadline = 85% of the crossed-anchor p10 "
                    "safe horizon. Treat as a planning heuristic, "
                    "not a locked operational threshold."
                ),
            }
        )

    return pd.DataFrame(rows)


def save_horizon_vs_anchor_plots(
    results_df: pd.DataFrame,
    figures_dir: Path,
) -> None:
    for mode, mode_df in results_df.groupby("mode"):
        plt.figure(figsize=(12, 7))

        for threshold, group in mode_df.groupby(
            "threshold_m",
            sort=True,
        ):
            crossed = group["crossed"].to_numpy(dtype=bool)
            x = group["anchor_reference_distance_m"]
            y = group["safe_horizon_distance_m"]

            line = plt.plot(
                x,
                y,
                marker="o",
                linewidth=1.2,
                label=f"{threshold:g} m error budget",
            )[0]

            censored_group = group[~crossed]
            if len(censored_group):
                plt.scatter(
                    censored_group[
                        "anchor_reference_distance_m"
                    ],
                    censored_group[
                        "safe_horizon_distance_m"
                    ],
                    marker="^",
                    facecolors="none",
                    edgecolors=line.get_color(),
                )

        plt.xlabel(
            "Anchor position along reference path [m] "
            "— evaluation only"
        )
        plt.ylabel("Safe horizon after correction anchor [m]")
        plt.title(
            f"S6A.3 rolling-anchor safe horizons: {mode}"
        )
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            figures_dir
            / f"s6a3_safe_horizon_vs_anchor_{mode}.png",
            dpi=180,
        )
        plt.close()


def save_summary_plot(
    summary_df: pd.DataFrame,
    figures_dir: Path,
) -> None:
    plt.figure(figsize=(10, 7))

    for mode, group in summary_df.groupby("mode"):
        group = group.sort_values("threshold_m")

        x = group["threshold_m"].to_numpy(dtype=float)
        median = group[
            "safe_distance_median_m_crossed"
        ].to_numpy(dtype=float)
        p10 = group[
            "safe_distance_p10_m_crossed"
        ].to_numpy(dtype=float)
        p90 = group[
            "safe_distance_p90_m_crossed"
        ].to_numpy(dtype=float)

        lower = median - p10
        upper = p90 - median

        plt.errorbar(
            x,
            median,
            yerr=np.vstack([lower, upper]),
            marker="o",
            capsize=4,
            label=mode,
        )

    plt.xlabel("Position-error budget [m]")
    plt.ylabel(
        "Safe horizon [m]\n"
        "median with p10–p90 crossed-anchor range"
    )
    plt.title("S6A.3 rolling-anchor safe-horizon summary")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        figures_dir / "s6a3_safe_horizon_summary.png",
        dpi=180,
    )
    plt.close()


def save_anchor_map(
    results_df: pd.DataFrame,
    base_df: pd.DataFrame,
    figures_dir: Path,
    threshold_m: float,
    mode: str,
) -> None:
    selection = results_df[
        (results_df["mode"] == mode)
        & np.isclose(
            results_df["threshold_m"],
            threshold_m,
        )
    ].copy()

    if selection.empty:
        return

    plt.figure(figsize=(9, 8))
    plt.plot(
        base_df["reference_x_m"],
        base_df["reference_y_m"],
        linewidth=1.0,
        label="Reference trajectory",
    )

    scatter = plt.scatter(
        selection["anchor_reference_x_m"],
        selection["anchor_reference_y_m"],
        c=selection["safe_horizon_distance_m"],
        s=45,
        label="Correction anchors",
    )

    plt.colorbar(
        scatter,
        label=f"Safe horizon to {threshold_m:g} m error [m]",
    )
    plt.axis("equal")
    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")
    plt.title(
        f"S6A.3 {mode}: spatial safe-horizon map"
    )
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        figures_dir
        / f"s6a3_anchor_map_{mode}_{threshold_m:g}m.png",
        dpi=180,
    )
    plt.close()


def save_representative_curves(
    results_df: pd.DataFrame,
    curves_df: pd.DataFrame,
    figures_dir: Path,
    threshold_m: float,
    mode: str,
) -> None:
    selection = results_df[
        (results_df["mode"] == mode)
        & np.isclose(
            results_df["threshold_m"],
            threshold_m,
        )
        & results_df["crossed"]
    ].copy()

    if len(selection) < 3:
        return

    selection = selection.sort_values(
        "safe_horizon_distance_m"
    )

    candidate_rows = [
        ("worst", selection.iloc[0]),
        (
            "median",
            selection.iloc[len(selection) // 2],
        ),
        ("best", selection.iloc[-1]),
    ]

    plt.figure(figsize=(11, 7))

    for label, row in candidate_rows:
        anchor = int(row["anchor_frame_index"])
        curve = curves_df[
            (curves_df["mode"] == mode)
            & (curves_df["anchor_frame_index"] == anchor)
        ].copy()

        plt.plot(
            curve["distance_after_anchor_m"],
            curve["position_error_m"],
            label=(
                f"{label}: anchor {anchor}, "
                f"horizon {row['safe_horizon_distance_m']:.0f} m"
            ),
        )

    plt.axhline(
        threshold_m,
        linestyle="--",
        linewidth=1.2,
        label=f"{threshold_m:g} m error budget",
    )
    plt.xlabel("Distance travelled after anchor [m]")
    plt.ylabel("Position error after simulated correction [m]")
    plt.title(
        f"S6A.3 representative rolling-anchor errors: {mode}"
    )
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        figures_dir
        / (
            f"s6a3_representative_error_curves_"
            f"{mode}_{threshold_m:g}m.png"
        ),
        dpi=180,
    )
    plt.close()


def load_configuration(
    report_path: Path,
) -> dict[str, Any]:
    if not report_path.exists():
        return {}

    with report_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "S6A.3: evaluate rolling-anchor safe horizons after "
            "simulated absolute drift corrections."
        )
    )
    parser.add_argument(
        "--aligned-csv",
        type=Path,
        default=DEFAULT_ALIGNED_CSV,
    )
    parser.add_argument(
        "--s6a2-report",
        type=Path,
        default=DEFAULT_S6A2_REPORT,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    parser.add_argument(
        "--variant",
        default="se2_scale_normalized",
    )
    parser.add_argument(
        "--thresholds-m",
        type=parse_float_list,
        default=parse_float_list("10,20,40,80"),
    )
    parser.add_argument(
        "--anchor-step-frames",
        type=int,
        default=25,
    )
    parser.add_argument(
        "--start-anchor-frame",
        type=int,
        default=None,
        help=(
            "Default: end of the S6A.2 alignment prefix."
        ),
    )
    parser.add_argument(
        "--min-remaining-distance-m",
        type=float,
        default=2000.0,
        help=(
            "Exclude anchors too close to sequence end, where "
            "censoring would dominate the safe-horizon estimate."
        ),
    )
    parser.add_argument(
        "--sustain-frames",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--heading-lookback-frames",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--min-heading-baseline-m",
        type=float,
        default=20.0,
    )
    parser.add_argument(
        "--representative-threshold-m",
        type=float,
        default=20.0,
    )
    args = parser.parse_args()

    if args.anchor_step_frames < 1:
        raise ValueError(
            "--anchor-step-frames must be positive."
        )
    if args.sustain_frames < 1:
        raise ValueError(
            "--sustain-frames must be positive."
        )
    if args.heading_lookback_frames < 1:
        raise ValueError(
            "--heading-lookback-frames must be positive."
        )
    if args.min_remaining_distance_m <= 0:
        raise ValueError(
            "--min-remaining-distance-m must be positive."
        )

    if not args.aligned_csv.exists():
        raise FileNotFoundError(
            f"Missing S6A.2 aligned trajectory: "
            f"{args.aligned_csv}"
        )

    full_df = pd.read_csv(args.aligned_csv)
    if "variant" not in full_df.columns:
        raise RuntimeError(
            "Aligned CSV does not contain a 'variant' column."
        )

    base_df = full_df[
        full_df["variant"].astype(str) == args.variant
    ].copy()

    if base_df.empty:
        raise RuntimeError(
            f"Variant {args.variant!r} not found."
        )

    base_df = base_df.sort_values(
        "sequence_frame_id",
        kind="mergesort",
    ).reset_index(drop=True)

    required_columns = {
        "sequence_frame_id",
        "reference_x_m",
        "reference_y_m",
        "reference_cumulative_distance_m",
        "prefix_aligned_x_m",
        "prefix_aligned_y_m",
    }
    missing = sorted(
        required_columns.difference(base_df.columns)
    )
    if missing:
        raise RuntimeError(
            f"Aligned CSV missing columns: {missing}"
        )

    visual = base_df[
        ["prefix_aligned_x_m", "prefix_aligned_y_m"]
    ].to_numpy(dtype=float)
    reference = base_df[
        ["reference_x_m", "reference_y_m"]
    ].to_numpy(dtype=float)
    cumulative_distance = base_df[
        "reference_cumulative_distance_m"
    ].to_numpy(dtype=float)

    if "pair_safe_image_only" in base_df.columns:
        safe_flags = bool_series(
            base_df["pair_safe_image_only"]
        ).to_numpy(dtype=bool)
    else:
        safe_flags = np.ones(len(base_df), dtype=bool)

    s6a2_report = load_configuration(
        args.s6a2_report
    )
    prefix_frames = int(
        s6a2_report.get("configuration", {}).get(
            "alignment_prefix_frames",
            50,
        )
    )

    start_anchor = (
        int(args.start_anchor_frame)
        if args.start_anchor_frame is not None
        else max(prefix_frames - 1, 0)
    )

    anchors = build_anchor_indices(
        frame_count=len(base_df),
        start_anchor=start_anchor,
        anchor_step=args.anchor_step_frames,
        cumulative_distance=cumulative_distance,
        min_remaining_distance_m=(
            args.min_remaining_distance_m
        ),
    )

    if not anchors:
        raise RuntimeError(
            "No eligible anchors. Reduce "
            "--min-remaining-distance-m or change the start/step."
        )

    modes = [
        "position_only",
        "position_heading_reset_eval_only",
    ]

    result_rows: list[dict[str, Any]] = []
    curve_frames: list[pd.DataFrame] = []

    for anchor_index, anchor in enumerate(anchors, start=1):
        if anchor_index % 10 == 0 or anchor_index == len(anchors):
            print(
                f"Anchors: {anchor_index}/{len(anchors)}"
            )

        for mode in modes:
            rows, curve = evaluate_anchor_mode(
                mode=mode,
                visual=visual,
                reference=reference,
                cumulative_distance=cumulative_distance,
                safe_flags=safe_flags,
                anchor=anchor,
                thresholds_m=args.thresholds_m,
                sustain_frames=args.sustain_frames,
                heading_lookback_frames=(
                    args.heading_lookback_frames
                ),
                min_heading_baseline_m=(
                    args.min_heading_baseline_m
                ),
            )
            result_rows.extend(rows)
            curve_frames.append(curve)

    results_df = pd.DataFrame(result_rows)
    curves_df = pd.concat(
        curve_frames,
        ignore_index=True,
    )
    summary_df = summarize_results(results_df)

    metadata_dir = (
        args.output_root / "metadata" / "s6a_relative_motion"
    )
    reports_dir = (
        args.output_root / "reports" / "s6a_relative_motion"
    )
    figures_dir = (
        args.output_root / "figures" / "s6a_relative_motion"
    )
    metadata_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    results_path = (
        metadata_dir
        / "s6a3_rolling_anchor_threshold_results.csv"
    )
    curves_path = (
        metadata_dir
        / "s6a3_rolling_anchor_error_curves.csv"
    )
    summary_path = (
        metadata_dir
        / "s6a3_safe_horizon_summary.csv"
    )
    recommendation_path = (
        metadata_dir
        / "s6a3_correction_trigger_recommendations.csv"
    )
    report_path = (
        reports_dir
        / "s6a3_rolling_anchor_safe_horizon.json"
    )

    results_df.to_csv(results_path, index=False)
    curves_df.to_csv(curves_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    recommendation_columns = [
        "mode",
        "threshold_m",
        "anchors",
        "crossed_anchors",
        "crossing_rate",
        "conservative_safe_horizon_m",
        "conservative_safe_horizon_frames",
        "suggested_absolute_search_start_m",
        "suggested_acceptance_deadline_m",
        "suggested_absolute_search_start_frames",
        "suggested_acceptance_deadline_frames",
        "heuristic_note",
    ]
    summary_df[recommendation_columns].to_csv(
        recommendation_path,
        index=False,
    )

    save_horizon_vs_anchor_plots(
        results_df,
        figures_dir,
    )
    save_summary_plot(
        summary_df,
        figures_dir,
    )
    save_anchor_map(
        results_df,
        base_df,
        figures_dir,
        threshold_m=args.representative_threshold_m,
        mode="position_only",
    )
    save_anchor_map(
        results_df,
        base_df,
        figures_dir,
        threshold_m=args.representative_threshold_m,
        mode="position_heading_reset_eval_only",
    )
    save_representative_curves(
        results_df,
        curves_df,
        figures_dir,
        threshold_m=args.representative_threshold_m,
        mode="position_only",
    )
    save_representative_curves(
        results_df,
        curves_df,
        figures_dir,
        threshold_m=args.representative_threshold_m,
        mode="position_heading_reset_eval_only",
    )

    report = {
        "stage": "S6A.3",
        "variant": args.variant,
        "input": str(args.aligned_csv),
        "reference_rule": (
            "Reference trajectory is used only for rolling-anchor "
            "evaluation and for the explicitly named evaluation-only "
            "heading-reset upper-bound simulation."
        ),
        "correction_modes": {
            "position_only": (
                "Translates the visual trajectory so the anchor "
                "position equals the reference position. Existing "
                "heading and metric scale are preserved. This is the "
                "closest simulation of the current S5 absolute output."
            ),
            "position_heading_reset_eval_only": (
                "Also rotates the future visual trajectory using a "
                "past-window reference heading. This is an evaluation-"
                "only upper bound for a future absolute correction "
                "that can reliably estimate orientation."
            ),
        },
        "configuration": {
            "thresholds_m": args.thresholds_m,
            "anchor_step_frames": args.anchor_step_frames,
            "start_anchor_frame": start_anchor,
            "eligible_anchors": len(anchors),
            "min_remaining_distance_m": (
                args.min_remaining_distance_m
            ),
            "sustain_frames": args.sustain_frames,
            "heading_lookback_frames": (
                args.heading_lookback_frames
            ),
            "min_heading_baseline_m": (
                args.min_heading_baseline_m
            ),
        },
        "summary": summary_df.to_dict(
            orient="records"
        ),
    }

    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print("\nS6A.3 rolling-anchor safe-horizon analysis")
    print("------------------------------------------")
    print(f"Variant:                  {args.variant}")
    print(f"Eligible anchors:         {len(anchors)}")
    print(
        "Anchor step:              "
        f"{args.anchor_step_frames} frames"
    )
    print(
        "Minimum remaining path:   "
        f"{args.min_remaining_distance_m:.1f} m"
    )
    print(
        "Sustained crossing:       "
        f"{args.sustain_frames} frames"
    )

    for _, row in summary_df.iterrows():
        print(
            f"\nMode: {row['mode']} | "
            f"error budget: {row['threshold_m']:g} m"
        )
        print(
            "  Crossed anchors:        "
            f"{int(row['crossed_anchors'])}/"
            f"{int(row['anchors'])}"
        )
        print(
            "  Safe distance min [m]:  "
            f"{row['safe_distance_min_m_crossed']:.1f}"
        )
        print(
            "  Safe distance p10 [m]:  "
            f"{row['safe_distance_p10_m_crossed']:.1f}"
        )
        print(
            "  Safe distance median:   "
            f"{row['safe_distance_median_m_crossed']:.1f} m"
        )
        print(
            "  Safe distance p90 [m]:  "
            f"{row['safe_distance_p90_m_crossed']:.1f}"
        )
        print(
            "  Conservative horizon:   "
            f"{row['conservative_safe_horizon_m']:.1f} m"
        )
        print(
            "  Search-start heuristic: "
            f"{row['suggested_absolute_search_start_m']:.1f} m"
        )
        print(
            "  Acceptance deadline:    "
            f"{row['suggested_acceptance_deadline_m']:.1f} m"
        )

    print("\nSaved outputs")
    print("-------------")
    print(results_path)
    print(curves_path)
    print(summary_path)
    print(recommendation_path)
    print(report_path)
    print(figures_dir)


if __name__ == "__main__":
    main()
