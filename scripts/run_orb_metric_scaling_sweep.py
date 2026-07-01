from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from uavloc.relative.orb_metric_scaling import run_orb_metric_scaling
from uavloc.relative.orb_relative_motion import resolve_project_paths


def parse_float_list(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep ORB metric scaling sign, yaw offset, and scale multiplier."
    )

    parser.add_argument("--config", required=True)
    parser.add_argument("--base-run-name", required=True)
    parser.add_argument("--input-csv", default=None)

    parser.add_argument("--height-column", default="height_agl_m")
    parser.add_argument("--yaw-column", default="yaw_deg")

    parser.add_argument(
        "--scale-multipliers",
        default="1.0,1.5,2.0,2.5,3.0,3.2,3.5",
        help="Comma-separated scale multipliers.",
    )
    parser.add_argument(
        "--yaw-offsets-deg",
        default="0,90,-90,180",
        help="Comma-separated yaw offsets in degrees.",
    )

    parser.add_argument("--input-stage-name", default="09b_orb_relative_motion_subset")
    parser.add_argument("--sweep-name", default=None)

    return parser.parse_args()


def get_ref_eval(summary: dict[str, Any]) -> dict[str, Any]:
    ref = summary.get("reference_evaluation", {})
    if not isinstance(ref, dict):
        return {}
    return ref


def get_shape_eval(summary: dict[str, Any]) -> dict[str, Any]:
    ref = get_ref_eval(summary)
    shape = ref.get("shape_alignment", {})
    if not isinstance(shape, dict):
        return {}
    return shape


def main() -> None:
    args = parse_args()

    _, _, _, output_dir, dataset_name = resolve_project_paths(args.config)

    scale_multipliers = parse_float_list(args.scale_multipliers)
    yaw_offsets_deg = parse_float_list(args.yaw_offsets_deg)

    sign_options = [
        (1.0, 1.0),
        (1.0, -1.0),
        (-1.0, 1.0),
        (-1.0, -1.0),
    ]

    sweep_name = args.sweep_name
    if sweep_name is None:
        sweep_name = f"{args.base_run_name}_axis_yaw_scale_sweep"

    sweep_report_dir = output_dir / "reports" / "09c_orb_metric_scaling_sweep" / sweep_name
    sweep_report_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []

    total = len(sign_options) * len(yaw_offsets_deg) * len(scale_multipliers)
    counter = 0

    for sx, sy in sign_options:
        for yaw_offset in yaw_offsets_deg:
            for scale in scale_multipliers:
                counter += 1

                run_name = (
                    f"sweep_{args.base_run_name}"
                    f"_sx{sx:+.0f}_sy{sy:+.0f}"
                    f"_yaw{yaw_offset:+.0f}"
                    f"_scale{str(scale).replace('.', 'p')}"
                )

                print(f"[{counter}/{total}] {run_name}")

                try:
                    summary = run_orb_metric_scaling(
                        config_path=Path(args.config),
                        run_name=run_name,
                        input_csv=args.input_csv,
                        input_stage_name=args.input_stage_name,
                        height_column=args.height_column,
                        yaw_column=args.yaw_column,
                        image_x_to_right_sign=sx,
                        image_y_to_forward_sign=sy,
                        scale_multiplier=scale,
                        yaw_offset_deg=yaw_offset,
                    )

                    ref = get_ref_eval(summary)
                    shape = get_shape_eval(summary)

                    row = {
                        "status": "ok",
                        "dataset_name": dataset_name,
                        "base_run_name": args.base_run_name,
                        "sweep_run_name": run_name,
                        "image_x_to_right_sign": sx,
                        "image_y_to_forward_sign": sy,
                        "yaw_offset_deg": yaw_offset,
                        "scale_multiplier": scale,
                        "frames": summary.get("frames"),
                        "estimated_path_length_m": summary.get("estimated_path_length_m"),
                        "reference_path_length_m": ref.get("reference_path_length_m"),
                        "rmse_m": ref.get("rmse_m"),
                        "mean_error_m": ref.get("mean_error_m"),
                        "median_error_m": ref.get("median_error_m"),
                        "max_error_m": ref.get("max_error_m"),
                        "final_error_m": ref.get("final_error_m"),
                        "drift_per_100m": ref.get("drift_per_100m"),
                        "shape_rmse_m": shape.get("rmse_m"),
                        "shape_rotation_deg": shape.get("rotation_deg"),
                        "shape_scale": shape.get("scale"),
                        "trajectory_csv": summary.get("outputs", {}).get("trajectory_csv"),
                        "comparison_plot": summary.get("outputs", {}).get("comparison_plot"),
                        "error_plot": summary.get("outputs", {}).get("error_plot"),
                    }

                except Exception as exc:
                    row = {
                        "status": "failed",
                        "dataset_name": dataset_name,
                        "base_run_name": args.base_run_name,
                        "sweep_run_name": run_name,
                        "image_x_to_right_sign": sx,
                        "image_y_to_forward_sign": sy,
                        "yaw_offset_deg": yaw_offset,
                        "scale_multiplier": scale,
                        "error": str(exc),
                    }

                rows.append(row)

    df = pd.DataFrame(rows)

    if "rmse_m" in df.columns:
        df_sorted = df.sort_values(
            by=["status", "rmse_m", "final_error_m"],
            ascending=[False, True, True],
            na_position="last",
        )
    else:
        df_sorted = df

    summary_csv = sweep_report_dir / "metric_scaling_sweep_summary.csv"
    summary_json = sweep_report_dir / "metric_scaling_sweep_summary.json"

    df_sorted.to_csv(summary_csv, index=False)

    payload = {
        "dataset_name": dataset_name,
        "base_run_name": args.base_run_name,
        "sweep_name": sweep_name,
        "total_runs": int(len(df_sorted)),
        "summary_csv": str(summary_csv),
        "top_10": df_sorted.head(10).to_dict(orient="records"),
    }

    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("\nSweep complete")
    print("--------------")
    print(f"Summary CSV:  {summary_csv}")
    print(f"Summary JSON: {summary_json}")

    print("\nTop 10 by RMSE")
    print(df_sorted.head(10).to_string(index=False))


if __name__ == "__main__":
    main()