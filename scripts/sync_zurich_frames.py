import argparse
from pathlib import Path

import pandas as pd

from uavloc.utils.config import load_config
from uavloc.data.frame_sync import (
    build_image_index,
    synchronize_images_with_reference,
    summarize_frame_sync,
    save_json,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to dataset config YAML")
    args = parser.parse_args()

    config = load_config(args.config)

    output_dir = Path(config["paths"]["output_dir"])
    image_dir = Path(config["paths"]["mav_images_dir"])

    reference_csv = output_dir / "trajectories" / "reference_trajectory.csv"

    if not reference_csv.exists():
        raise FileNotFoundError(
            f"Reference trajectory not found: {reference_csv}\n"
            "Run build_reference_trajectory.py first."
        )

    metadata_dir = output_dir / "metadata"
    report_dir = output_dir / "reports"

    metadata_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    image_df = build_image_index(image_dir)
    reference_df = pd.read_csv(reference_csv)

    synced = synchronize_images_with_reference(image_df, reference_df)

    output_csv = metadata_dir / "synchronized_frames.csv"
    synced.to_csv(output_csv, index=False)

    summary = summarize_frame_sync(synced, image_df, reference_df)

    summary_path = report_dir / "frame_sync_summary.json"
    save_json(summary, summary_path)

    print("Frame-to-telemetry synchronization generated")
    print("-------------------------------------------")
    print(f"Image directory:        {image_dir}")
    print(f"Reference CSV:          {reference_csv}")
    print(f"Synchronized CSV:       {output_csv}")
    print(f"Summary JSON:           {summary_path}")
    print(f"Images found:           {summary['image_count']}")
    print(f"Reference rows:         {summary['reference_rows']}")
    print(f"Synced rows:            {summary['synced_rows']}")
    print(f"Rows with reference:    {summary['rows_with_reference']}")
    print(f"Rows missing reference: {summary['rows_missing_reference']}")
    print(f"Image imgid range:      {summary['image_imgid_min']} to {summary['image_imgid_max']}")
    print(f"Reference imgid range:  {summary['reference_imgid_min']} to {summary['reference_imgid_max']}")


if __name__ == "__main__":
    main()