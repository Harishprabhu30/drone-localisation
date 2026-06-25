from __future__ import annotations

import argparse
import json
from pathlib import Path

from uavloc.relative.orb_metric_scaling import run_orb_metric_scaling


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ORB metric scaling and reference evaluation.")
    parser.add_argument("--config", required=True, help="Path to dataset YAML config.")

    parser.add_argument(
        "--default-height-m",
        type=float,
        default=50.0,
        help="Fallback AGL height in meters when no safe height column is found.",
    )
    parser.add_argument(
        "--height-column",
        default=None,
        help="Explicit column to use as AGL height in meters, e.g. height or height_agl_m.",
    )
    parser.add_argument(
        "--allow-absolute-altitude",
        action="store_true",
        help="Allow altitude/alt columns as height source. Use carefully; GPS altitude is not AGL.",
    )

    parser.add_argument(
        "--fixed-heading-deg",
        type=float,
        default=0.0,
        help="Fallback heading/yaw in degrees. Convention: 0=N, 90=E.",
    )
    parser.add_argument(
        "--yaw-column",
        default=None,
        help="Explicit yaw/heading column, e.g. azimuth or yaw_deg.",
    )

    parser.add_argument("--fx", type=float, default=None, help="Optional focal length x in pixels.")
    parser.add_argument("--fy", type=float, default=None, help="Optional focal length y in pixels.")

    parser.add_argument(
        "--image-x-to-right-sign",
        type=float,
        default=1.0,
        choices=[-1.0, 1.0],
        help="Axis sign: +1 means image x motion maps to drone right, -1 flips it.",
    )
    parser.add_argument(
        "--image-y-to-forward-sign",
        type=float,
        default=1.0,
        choices=[-1.0, 1.0],
        help="Axis sign: +1 means image y motion maps to drone forward, -1 flips it.",
    )
    parser.add_argument(
        "--scale-multiplier",
        type=float,
        default=1.0,
        help="Debug scale multiplier. Keep 1.0 for normal runs.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    summary = run_orb_metric_scaling(
        config_path=Path(args.config),
        default_height_m=args.default_height_m,
        fixed_heading_deg=args.fixed_heading_deg,
        height_column=args.height_column,
        yaw_column=args.yaw_column,
        allow_absolute_altitude=args.allow_absolute_altitude,
        fx=args.fx,
        fy=args.fy,
        image_x_to_right_sign=args.image_x_to_right_sign,
        image_y_to_forward_sign=args.image_y_to_forward_sign,
        scale_multiplier=args.scale_multiplier,
    )

    print("ORB metric scaling generated")
    print("----------------------------")
    print(f"Frames:                 {summary['frames']}")
    print(f"Camera fx/fy [px]:      {summary['camera_intrinsics']['fx']:.3f} / {summary['camera_intrinsics']['fy']:.3f}")
    print(f"Calibration source:     {summary['camera_intrinsics']['source']}")
    print(f"Height source:          {summary['height']['source']}")
    print(f"Height column:          {summary['height'].get('column')}")
    print(f"Median height [m]:      {summary['height']['median_height_m']:.3f}")
    print(f"Yaw source:             {summary['yaw_heading']['source']}")
    print(f"Yaw column:             {summary['yaw_heading'].get('column')}")
    print(f"Estimated path [m]:     {summary['estimated_path_length_m']:.3f}")
    print(f"Mean confidence:        {summary['confidence']['mean']:.3f}")

    ref = summary.get("reference_evaluation", {})

    if ref.get("available", False):
        print("\nReference evaluation")
        print("--------------------")
        print(f"RMSE [m]:              {ref['rmse_m']:.3f}")
        print(f"Mean error [m]:        {ref['mean_error_m']:.3f}")
        print(f"Max error [m]:         {ref['max_error_m']:.3f}")
        print(f"Final error [m]:       {ref['final_error_m']:.3f}")
        print(f"Reference path [m]:    {ref['reference_path_length_m']:.3f}")
        print(f"Drift per 100 m:       {ref['drift_per_100m']}")

        shape = ref.get("shape_alignment", {})

        if shape.get("available", False):
            print("\nShape-aligned check")
            print("-------------------")
            print(f"Shape RMSE [m]:        {shape['rmse_m']:.3f}")
            print(f"Shape rotation [deg]:  {shape['rotation_deg']:.2f}")
            print(f"Shape scale:           {shape['scale']:.6f}")
    else:
        print(f"\nReference evaluation: not available ({ref.get('reason', 'unknown reason')})")

    print("\nOutputs")
    print("-------")

    for key, value in summary["outputs"].items():
        print(f"{key}: {value}")

    print("\nHeight/yaw/camera details")
    print(json.dumps(
        {
            "camera_intrinsics": summary["camera_intrinsics"],
            "height": summary["height"],
            "yaw_heading": summary["yaw_heading"],
            "axis_mapping": summary["axis_mapping"],
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()