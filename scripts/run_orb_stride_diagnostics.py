import argparse
from pathlib import Path
import json
from typing import Dict, Any, List

import pandas as pd
import matplotlib.pyplot as plt

from uavloc.utils.config import load_config
from uavloc.relative.orb_pair_diagnostics import compute_orb_pair_match


def classify_pair(row: pd.Series) -> str:
    if row["status"] != "ok" or not row["homography_found"]:
        return "weak"

    if row["good_matches"] >= 80 and row["ransac_inliers"] >= 40 and row["inlier_ratio"] >= 0.30:
        return "good"

    if row["good_matches"] >= 40 and row["ransac_inliers"] >= 20 and row["inlier_ratio"] >= 0.20:
        return "medium"

    return "weak"


def run_stride_diagnostics(
    synced_df: pd.DataFrame,
    strides: List[int],
    max_starts: int | None = None,
    nfeatures: int = 2000,
    ratio_thresh: float = 0.75,
    ransac_reproj_thresh: float = 3.0,
) -> pd.DataFrame:
    rows = []

    for stride in strides:
        max_start_index = len(synced_df) - stride

        if max_starts is not None:
            max_start_index = min(max_start_index, max_starts)

        for i in range(max_start_index):
            row1 = synced_df.iloc[i]
            row2 = synced_df.iloc[i + stride]

            try:
                result, _, _, _, _, _, _ = compute_orb_pair_match(
                    image_path_1=Path(row1["image_path"]),
                    image_path_2=Path(row2["image_path"]),
                    nfeatures=nfeatures,
                    ratio_thresh=ratio_thresh,
                    ransac_reproj_thresh=ransac_reproj_thresh,
                )
            except Exception as exc:
                result = {
                    "kp1": None,
                    "kp2": None,
                    "raw_matches": None,
                    "good_matches": None,
                    "homography_found": False,
                    "ransac_inliers": None,
                    "inlier_ratio": None,
                    "mean_match_distance": None,
                    "median_match_distance": None,
                    "status": "error",
                    "error": str(exc),
                }

            dx_ref = float(row2["x_enu_m"] - row1["x_enu_m"])
            dy_ref = float(row2["y_enu_m"] - row1["y_enu_m"])
            dz_ref = float(row2["z_enu_m"] - row1["z_enu_m"])
            ref_step_distance_m = float((dx_ref**2 + dy_ref**2) ** 0.5)

            output_row = {
                "stride": stride,
                "start_index": i,
                "imgid_1": int(row1["imgid"]),
                "imgid_2": int(row2["imgid"]),
                "image_1": row1["image_path"],
                "image_2": row2["image_path"],
                "timestamp_1": row1["timestamp"],
                "timestamp_2": row2["timestamp"],
                "ref_dx_m": dx_ref,
                "ref_dy_m": dy_ref,
                "ref_dz_m": dz_ref,
                "ref_step_distance_m": ref_step_distance_m,
            }

            output_row.update(result)
            rows.append(output_row)

    pair_df = pd.DataFrame(rows)
    pair_df["pair_quality"] = pair_df.apply(classify_pair, axis=1)

    return pair_df


def summarize_by_stride(pair_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for stride, group in pair_df.groupby("stride"):
        rows.append(
            {
                "stride": int(stride),
                "total_pairs": int(len(group)),
                "good_pairs": int((group["pair_quality"] == "good").sum()),
                "medium_pairs": int((group["pair_quality"] == "medium").sum()),
                "weak_pairs": int((group["pair_quality"] == "weak").sum()),
                "homography_success_rate": float(group["homography_found"].mean()),
                "good_matches_median": float(group["good_matches"].median()),
                "ransac_inliers_median": float(group["ransac_inliers"].median()),
                "inlier_ratio_median": float(group["inlier_ratio"].median()),
                "ref_step_distance_m_median": float(group["ref_step_distance_m"].median()),
                "ref_step_distance_m_max": float(group["ref_step_distance_m"].max()),
            }
        )

    return pd.DataFrame(rows).sort_values("stride")


def save_json(data: Dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w") as f:
        json.dump(data, f, indent=2)


def plot_stride_summary(summary_df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)

    axes[0].plot(summary_df["stride"], summary_df["good_matches_median"], marker="o")
    axes[0].set_ylabel("Median good matches")
    axes[0].set_title("ORB Stride Diagnostics")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(summary_df["stride"], summary_df["ransac_inliers_median"], marker="o")
    axes[1].set_ylabel("Median RANSAC inliers")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(summary_df["stride"], summary_df["inlier_ratio_median"], marker="o")
    axes[2].set_xlabel("Frame stride")
    axes[2].set_ylabel("Median inlier ratio")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_stride_ref_distance(summary_df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(summary_df["stride"], summary_df["ref_step_distance_m_median"], marker="o", label="Median")
    plt.plot(summary_df["stride"], summary_df["ref_step_distance_m_max"], marker="x", label="Max")
    plt.xlabel("Frame stride")
    plt.ylabel("Reference displacement [m]")
    plt.title("Reference Motion Distance by Frame Stride")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to dataset config YAML")
    parser.add_argument(
        "--strides",
        nargs="+",
        type=int,
        default=[1, 2, 3, 5, 10],
        help="Frame strides to test",
    )
    parser.add_argument(
        "--max-starts",
        type=int,
        default=None,
        help="Optional maximum starting frames per stride for faster testing",
    )

    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = Path(config["paths"]["output_dir"])

    synced_csv = output_dir / "metadata" / "synchronized_frames.csv"

    if not synced_csv.exists():
        raise FileNotFoundError(
            f"Synchronized frames file not found: {synced_csv}\n"
            "Run sync_zurich_frames.py first."
        )

    stage_name = "06_orb_stride_diagnostics"

    metadata_dir = output_dir / "metadata" / stage_name
    report_dir = output_dir / "reports" / stage_name
    figures_dir = output_dir / "figures" / stage_name

    metadata_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    synced_df = pd.read_csv(synced_csv)

    pair_df = run_stride_diagnostics(
        synced_df=synced_df,
        strides=args.strides,
        max_starts=args.max_starts,
        nfeatures=2000,
        ratio_thresh=0.75,
        ransac_reproj_thresh=3.0,
    )

    summary_df = summarize_by_stride(pair_df)

    pair_csv = metadata_dir / "orb_stride_pair_quality.csv"
    summary_csv = metadata_dir / "orb_stride_summary.csv"
    summary_json = report_dir / "orb_stride_summary.json"
    quality_plot = figures_dir / "orb_stride_quality_summary.png"
    distance_plot = figures_dir / "orb_stride_reference_distance.png"

    pair_df.to_csv(pair_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)

    save_json(
        {
            "strides": args.strides,
            "max_starts": args.max_starts,
            "total_tested_pairs": int(len(pair_df)),
            "summary": summary_df.to_dict(orient="records"),
        },
        summary_json,
    )

    plot_stride_summary(summary_df, quality_plot)
    plot_stride_ref_distance(summary_df, distance_plot)

    print("ORB stride diagnostics generated")
    print("--------------------------------")
    print(f"Input synced CSV:     {synced_csv}")
    print(f"Pair CSV:             {pair_csv}")
    print(f"Summary CSV:          {summary_csv}")
    print(f"Summary JSON:         {summary_json}")
    print(f"Quality plot:         {quality_plot}")
    print(f"Distance plot:        {distance_plot}")
    print("")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()