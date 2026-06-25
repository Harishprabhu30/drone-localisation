from __future__ import annotations

import argparse
import json
from pathlib import Path

from uavloc.data.enrich_zurich_sync import run_enrich_zurich_sync


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Enrich Zurich synchronized frames with pose, height, yaw, and barometer data.")
    parser.add_argument("--config", required=True, help="Path to dataset YAML config.")
    parser.add_argument("--pose-tolerance", type=float, default=200000.0, help="Nearest timestamp tolerance for OnboardPose.")
    parser.add_argument("--barometer-tolerance", type=float, default=500000.0, help="Nearest timestamp tolerance for BarometricPressure.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    summary = run_enrich_zurich_sync(
        config_path=Path(args.config),
        pose_tolerance=args.pose_tolerance,
        barometer_tolerance=args.barometer_tolerance,
    )

    print("Zurich synchronized frames enriched")
    print("-----------------------------------")
    print(f"Frames:          {summary['frames']}")
    print(f"Output CSV:      {summary['output_enriched_csv']}")
    print(f"Input sync CSV:  {summary['input_sync_csv']}")

    selection = summary["selection_summary"]
    print(f"Height source:   {selection.get('selected_height_source')}")
    print(f"Yaw source:      {selection.get('selected_yaw_source')}")

    if "height_agl_m" in selection:
        h = selection["height_agl_m"]
        print(f"Height median:   {h['median']:.3f} m")
        print(f"Height range:    {h['min']:.3f} to {h['max']:.3f} m")
        print(f"Height valid:    {h['valid_count']}")

    if "yaw_deg" in selection:
        y = selection["yaw_deg"]
        print(f"Yaw median:      {y['median']:.3f} deg")
        print(f"Yaw range:       {y['min']:.3f} to {y['max']:.3f} deg")
        print(f"Yaw valid:       {y['valid_count']}")

    print("\nSource status")
    print(json.dumps(summary["source_status"], indent=2))


if __name__ == "__main__":
    main()
