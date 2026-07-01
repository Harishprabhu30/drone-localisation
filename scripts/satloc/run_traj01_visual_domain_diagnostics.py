from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml

from uavloc.analysis.visual_domain import (
    add_relative_quality_flags,
    compute_image_stats,
    ensure_dir,
    save_json,
    summarize_stats,
)


def load_config(config_path: str | Path) -> dict:
    with Path(config_path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_traj01_reference(output_dir: Path) -> pd.DataFrame:
    ref_path = output_dir / "trajectories" / "uav_reference_trajectory.csv"
    if not ref_path.exists():
        raise FileNotFoundError(
            f"Missing reference trajectory: {ref_path}. "
            "Run scripts/satloc/build_satloc_coordinate_index.py first."
        )

    df = pd.read_csv(ref_path)
    df = df[df["sequence"] == "traj01"].copy()

    if df.empty:
        raise ValueError("No traj01 rows found in uav_reference_trajectory.csv")

    df = df.sort_values(["frame_index_in_sequence"]).reset_index(drop=True)
    return df


def plot_metric_over_frame(df: pd.DataFrame, y_col: str, output_path: Path, title: str) -> None:
    ensure_dir(output_path.parent)

    plt.figure(figsize=(10, 4))
    plt.plot(df["frame_index_in_sequence"], df[y_col], linewidth=1.2)
    plt.xlabel("traj01 frame index")
    plt.ylabel(y_col)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_histogram(df: pd.DataFrame, col: str, output_path: Path, title: str) -> None:
    ensure_dir(output_path.parent)

    s = pd.to_numeric(df[col], errors="coerce").dropna()

    plt.figure(figsize=(7, 4))
    plt.hist(s, bins=40)
    plt.xlabel(col)
    plt.ylabel("count")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_scatter(df: pd.DataFrame, x_col: str, y_col: str, output_path: Path, title: str) -> None:
    ensure_dir(output_path.parent)

    plt.figure(figsize=(6, 5))
    plt.scatter(df[x_col], df[y_col], s=8, alpha=0.6)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run visual-domain diagnostics for SatLoc traj01 UAV images."
    )
    parser.add_argument("--config", required=True, help="Path to configs/dataset_satloc.yaml")
    parser.add_argument(
        "--max-dim",
        type=int,
        default=900,
        help="Maximum image dimension used for diagnostic computation.",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    output_dir = Path(config["paths"]["output_dir"])

    stage_name = "s3_5_visual_domain_traj01"

    metadata_dir = ensure_dir(output_dir / "metadata" / stage_name)
    reports_dir = ensure_dir(output_dir / "reports" / stage_name)
    figures_dir = ensure_dir(output_dir / "figures" / stage_name)

    traj_df = load_traj01_reference(output_dir)

    records = []

    print("SatLoc traj01 visual-domain diagnostics")
    print("---------------------------------------")
    print(f"Frames to inspect: {len(traj_df)}")

    for i, row in traj_df.iterrows():
        image_path = Path(row["image_path"])

        stats = compute_image_stats(image_path, max_dim=args.max_dim)

        stats.update(
            {
                "sequence": row["sequence"],
                "frame_index_in_sequence": int(row["frame_index_in_sequence"]),
                "global_frame_index": int(row["global_frame_index"]),
                "token0_id": int(row["token0_id"]),
                "token1_order": int(row["token1_order"]),
                "lon": float(row["lon"]),
                "lat": float(row["lat"]),
                "x_enu_m": float(row["x_enu_m"]),
                "y_enu_m": float(row["y_enu_m"]),
            }
        )

        records.append(stats)

        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(traj_df)}")

    stats_df = pd.DataFrame(records)
    stats_df = add_relative_quality_flags(stats_df)

    stats_csv = metadata_dir / "traj01_image_stats.csv"
    stats_df.to_csv(stats_csv, index=False)

    summary = summarize_stats(stats_df)
    summary["stage"] = stage_name
    summary["config"] = str(args.config)
    summary["outputs"] = {
        "stats_csv": str(stats_csv),
    }

    summary_json = reports_dir / "traj01_visual_domain_summary.json"
    save_json(summary, summary_json)

    # Metric-over-frame plots
    plot_metric_over_frame(
        stats_df,
        "luma_mean",
        figures_dir / "traj01_luma_mean_over_frame.png",
        "traj01 luma mean over frame index",
    )
    plot_metric_over_frame(
        stats_df,
        "luma_std",
        figures_dir / "traj01_luma_contrast_over_frame.png",
        "traj01 luma std / contrast over frame index",
    )
    plot_metric_over_frame(
        stats_df,
        "laplacian_variance",
        figures_dir / "traj01_blur_sharpness_over_frame.png",
        "traj01 Laplacian variance sharpness over frame index",
    )
    plot_metric_over_frame(
        stats_df,
        "edge_density",
        figures_dir / "traj01_edge_density_over_frame.png",
        "traj01 edge density over frame index",
    )
    plot_metric_over_frame(
        stats_df,
        "orb_keypoint_count",
        figures_dir / "traj01_orb_keypoints_over_frame.png",
        "traj01 ORB keypoint count over frame index",
    )
    plot_metric_over_frame(
        stats_df,
        "akaze_keypoint_count",
        figures_dir / "traj01_akaze_keypoints_over_frame.png",
        "traj01 AKAZE keypoint count over frame index",
    )
    plot_metric_over_frame(
        stats_df,
        "green_ratio_mean",
        figures_dir / "traj01_green_ratio_over_frame.png",
        "traj01 green ratio over frame index",
    )

    # Histograms
    histogram_cols = [
        "luma_mean",
        "luma_std",
        "laplacian_variance",
        "edge_density",
        "entropy_gray",
        "orb_keypoint_count",
        "akaze_keypoint_count",
        "hough_line_count",
        "green_ratio_mean",
        "colorfulness",
    ]

    for col in histogram_cols:
        plot_histogram(
            stats_df,
            col,
            figures_dir / f"hist_{col}.png",
            f"traj01 distribution: {col}",
        )

    # Relationship plots
    plot_scatter(
        stats_df,
        "laplacian_variance",
        "orb_keypoint_count",
        figures_dir / "scatter_sharpness_vs_orb_keypoints.png",
        "Sharpness vs ORB keypoints",
    )
    plot_scatter(
        stats_df,
        "edge_density",
        "orb_keypoint_count",
        figures_dir / "scatter_edge_density_vs_orb_keypoints.png",
        "Edge density vs ORB keypoints",
    )
    plot_scatter(
        stats_df,
        "green_ratio_mean",
        "orb_keypoint_count",
        figures_dir / "scatter_green_ratio_vs_orb_keypoints.png",
        "Green ratio vs ORB keypoints",
    )

    print()
    print("S3.5A visual-domain diagnostics complete")
    print("----------------------------------------")
    print(f"Saved stats CSV: {stats_csv}")
    print(f"Saved summary JSON: {summary_json}")
    print(f"Saved figures dir: {figures_dir}")
    print()
    print("Key medians")
    print("-----------")
    for col in [
        "luma_mean",
        "luma_std",
        "laplacian_variance",
        "edge_density",
        "entropy_gray",
        "orb_keypoint_count",
        "akaze_keypoint_count",
        "green_ratio_mean",
    ]:
        if col in stats_df.columns:
            print(f"{col}: {stats_df[col].median():.4f}")


if __name__ == "__main__":
    main()