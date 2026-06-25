import argparse
from pathlib import Path

import pandas as pd

from uavloc.utils.config import load_config
from uavloc.inspection.image_quality import (
    build_frame_quality_table,
    create_quality_summary,
    save_json,
    plot_sample_frames,
    plot_quality_metrics,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to dataset config YAML")
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = Path(config["paths"]["output_dir"])

    synced_csv = output_dir / "metadata" / "synchronized_frames.csv"

    if not synced_csv.exists():
        raise FileNotFoundError(
            f"Synchronized frames file not found: {synced_csv}\n"
            "Run sync_zurich_frames.py first."
        )

    metadata_dir = output_dir / "metadata"
    report_dir = output_dir / "reports"
    figures_dir = output_dir / "figures"

    metadata_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    synced_df = pd.read_csv(synced_csv)

    quality_df = build_frame_quality_table(synced_df)
    summary = create_quality_summary(quality_df)

    quality_csv = metadata_dir / "frame_quality.csv"
    summary_json = report_dir / "image_quality_summary.json"
    sample_grid_png = figures_dir / "sample_frames_grid.png"
    quality_plots_png = figures_dir / "image_quality_plots.png"

    quality_df.to_csv(quality_csv, index=False)
    save_json(summary, summary_json)
    plot_sample_frames(quality_df, sample_grid_png)
    plot_quality_metrics(quality_df, quality_plots_png)

    print("Frame image quality inspection generated")
    print("----------------------------------------")
    print(f"Input synced CSV:     {synced_csv}")
    print(f"Frame quality CSV:    {quality_csv}")
    print(f"Summary JSON:         {summary_json}")
    print(f"Sample frame grid:    {sample_grid_png}")
    print(f"Quality plots:        {quality_plots_png}")
    print(f"Total frames:         {summary['total_frames']}")
    print(f"Valid images:         {summary['valid_images']}")
    print(f"Failed images:        {summary['failed_images']}")
    print(f"Weak candidates:      {summary.get('weak_frame_candidates', 0)}")
    print(f"Blur score median:    {summary['blur_score_laplacian_var']['median']:.2f}")
    print(f"ORB keypoints median: {summary['orb_keypoints']['median']:.2f}")


if __name__ == "__main__":
    main()