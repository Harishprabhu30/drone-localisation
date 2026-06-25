from __future__ import annotations

import argparse
import json
from pathlib import Path

from uavloc.relative.orb_relative_motion import OrbRelativeMotionConfig, run_orb_relative_motion


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run first ORB relative motion baseline.")
    parser.add_argument("--config", required=True, help="Path to dataset YAML config.")
    parser.add_argument("--stride", type=int, default=1, help="Frame stride. Use 1 for first baseline.")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional debug limit, e.g. 50.")
    parser.add_argument("--nfeatures", type=int, default=2000, help="ORB feature count.")
    parser.add_argument("--ratio-test", type=float, default=0.75, help="Lowe ratio test threshold.")
    parser.add_argument("--max-hamming-distance", type=int, default=96, help="Maximum ORB descriptor distance kept.")
    parser.add_argument("--ransac-threshold", type=float, default=3.0, help="Homography RANSAC reprojection threshold in pixels.")
    parser.add_argument("--min-good-matches", type=int, default=30, help="Minimum good matches required.")
    parser.add_argument("--min-ransac-inliers", type=int, default=20, help="Minimum RANSAC inliers required.")
    parser.add_argument("--max-reasonable-step-px", type=float, default=250.0, help="Reject very large image-motion jumps.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.stride < 1:
        raise ValueError("--stride must be >= 1")

    orb_cfg = OrbRelativeMotionConfig(
        stride=args.stride,
        nfeatures=args.nfeatures,
        ratio_test=args.ratio_test,
        max_hamming_distance=args.max_hamming_distance,
        ransac_reproj_threshold=args.ransac_threshold,
        min_good_matches=args.min_good_matches,
        min_ransac_inliers=args.min_ransac_inliers,
        max_reasonable_step_px=args.max_reasonable_step_px,
    )

    summary = run_orb_relative_motion(
        config_path=Path(args.config),
        stride=args.stride,
        max_frames=args.max_frames,
        orb_cfg=orb_cfg,
    )

    print("ORB relative motion generated")
    print("-----------------------------")
    print(f"Frames used:         {summary['frames_used']}")
    print(f"Attempted pairs:     {summary['attempted_pairs']}")
    print(f"OK pairs:            {summary['ok_pairs']}")
    print(f"Failed pairs:        {summary['failed_pairs']}")
    print(f"Median matches:      {summary['median_good_matches']:.1f}")
    print(f"Median inliers:      {summary['median_ransac_inliers']:.1f}")
    print(f"Median inlier ratio: {summary['median_inlier_ratio']:.3f}")

    alignment = summary.get("alignment_to_reference", {})

    if alignment.get("available", False):
        print(f"Aligned RMSE [m]:    {alignment['aligned_rmse_m']:.3f}")
        print(f"Scale [m/px]:        {alignment['scale_m_per_px']:.6f}")
        print(f"Rotation [deg]:      {alignment['rotation_deg']:.2f}")
    else:
        print(f"Alignment:           not available ({alignment.get('reason', 'unknown reason')})")

    print("\nOutputs")
    print("-------")

    for key, value in summary["outputs"].items():
        if value is not None:
            print(f"{key}: {value}")

    print("\nStatus counts")
    print(json.dumps(summary["status_counts"], indent=2))


if __name__ == "__main__":
    main()