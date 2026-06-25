import argparse
from pathlib import Path

import pandas as pd

from uavloc.utils.config import load_config
from uavloc.relative.orb_pair_diagnostics import (
    run_orb_pair_diagnostics,
    summarize_orb_pairs,
    save_json,
    plot_pair_quality,
    draw_pair_matches,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to dataset config YAML")
    parser.add_argument("--max-pairs", type=int, default=None, help="Optional limit for quick testing")
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = Path(config["paths"]["output_dir"])

    synced_csv = output_dir / "metadata" / "synchronized_frames.csv"

    if not synced_csv.exists():
        raise FileNotFoundError(
            f"Synchronized frames file not found: {synced_csv}\n"
            "Run sync_zurich_frames.py first."
        )

    stage_name = "05_orb_pair_diagnostics"

    metadata_dir = output_dir / "metadata" / stage_name
    report_dir = output_dir / "reports" / stage_name
    figures_dir = output_dir / "figures" / stage_name

    metadata_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    synced_df = pd.read_csv(synced_csv)

    pair_df = run_orb_pair_diagnostics(
        synced_df=synced_df,
        max_pairs=args.max_pairs,
        nfeatures=2000,
        ratio_thresh=0.75,
        ransac_reproj_thresh=3.0,
    )

    summary = summarize_orb_pairs(pair_df)

    pair_csv = metadata_dir / "orb_pair_quality.csv"
    summary_json = report_dir / "orb_pair_summary.json"
    plot_png = figures_dir / "orb_pair_quality_plots.png"

    pair_df.to_csv(pair_csv, index=False)
    save_json(summary, summary_json)
    plot_pair_quality(pair_df, plot_png)

    valid_pairs = pair_df[
        (pair_df["homography_found"] == True)
        & pair_df["ransac_inliers"].notna()
        & pair_df["inlier_ratio"].notna()
    ].copy()

    if not valid_pairs.empty:
        good_pair = valid_pairs.sort_values(
            by=["pair_quality", "ransac_inliers", "inlier_ratio"],
            ascending=[True, False, False],
        ).iloc[0]

        weak_pair = valid_pairs.sort_values(
            by=["ransac_inliers", "inlier_ratio"],
            ascending=[True, True],
        ).iloc[0]

        draw_pair_matches(
            synced_df=synced_df,
            pair_row=good_pair,
            output_path=figures_dir / "orb_matches_good_pair.png",
        )

        draw_pair_matches(
            synced_df=synced_df,
            pair_row=weak_pair,
            output_path=figures_dir / "orb_matches_weak_pair.png",
        )

    print("ORB frame-pair diagnostics generated")
    print("------------------------------------")
    print(f"Input synced CSV:       {synced_csv}")
    print(f"Pair quality CSV:       {pair_csv}")
    print(f"Summary JSON:           {summary_json}")
    print(f"Quality plot:           {plot_png}")
    print(f"Total pairs:            {summary['total_pairs']}")
    print(f"Good pairs:             {summary['good_pairs']}")
    print(f"Medium pairs:           {summary['medium_pairs']}")
    print(f"Weak pairs:             {summary['weak_pairs']}")
    print(f"Homography successes:   {summary['homography_success_count']}")

    if "good_matches" in summary:
        print(f"Good matches median:    {summary['good_matches']['median']:.2f}")

    if "ransac_inliers" in summary:
        print(f"RANSAC inliers median:  {summary['ransac_inliers']['median']:.2f}")

    if "inlier_ratio" in summary:
        print(f"Inlier ratio median:    {summary['inlier_ratio']['median']:.3f}")


if __name__ == "__main__":
    main()