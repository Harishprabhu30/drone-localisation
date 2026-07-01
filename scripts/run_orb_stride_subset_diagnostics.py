from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from uavloc.relative.orb_relative_motion import resolve_project_paths


STAGE_NAME = "09a_orb_stride_subset_diagnostics"


def parse_strides(value: str) -> List[int]:
    strides = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        stride = int(item)
        if stride <= 0:
            raise ValueError("Stride must be positive.")
        strides.append(stride)

    if not strides:
        raise ValueError("At least one stride is required.")

    return sorted(set(strides))


def resolve_image_path(path_value: str, project_root: Path) -> Path:
    p = Path(str(path_value))

    if p.exists():
        return p

    candidate = project_root / p
    if candidate.exists():
        return candidate

    return p


def load_gray_image(path: Path, resize_width: Optional[int] = None) -> Optional[np.ndarray]:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    if img is None:
        return None

    if resize_width is not None and resize_width > 0 and img.shape[1] > resize_width:
        scale = resize_width / float(img.shape[1])
        new_h = max(1, int(round(img.shape[0] * scale)))
        img = cv2.resize(img, (resize_width, new_h), interpolation=cv2.INTER_AREA)

    return img


def classify_pair(
    good_matches: int,
    inliers: int,
    inlier_ratio: float,
    min_good_matches: int,
    min_inliers_good: int,
    min_inlier_ratio_good: float,
    min_inliers_medium: int,
    min_inlier_ratio_medium: float,
) -> str:
    if (
        good_matches >= min_good_matches
        and inliers >= min_inliers_good
        and inlier_ratio >= min_inlier_ratio_good
    ):
        return "good"

    if inliers >= min_inliers_medium and inlier_ratio >= min_inlier_ratio_medium:
        return "medium"

    return "weak"


def evaluate_orb_pair(
    img0_path: Path,
    img1_path: Path,
    orb: cv2.ORB,
    matcher: cv2.BFMatcher,
    resize_width: Optional[int],
    ratio_test: float,
    ransac_reproj_threshold_px: float,
    min_good_matches: int,
    min_inliers_good: int,
    min_inlier_ratio_good: float,
    min_inliers_medium: int,
    min_inlier_ratio_medium: float,
) -> Dict[str, Any]:
    t0 = time.perf_counter()

    img0 = load_gray_image(img0_path, resize_width=resize_width)
    img1 = load_gray_image(img1_path, resize_width=resize_width)

    if img0 is None or img1 is None:
        return {
            "status": "failed",
            "quality": "weak",
            "failure_reason": "image_load_failed",
            "runtime_s": time.perf_counter() - t0,
        }

    kp0, des0 = orb.detectAndCompute(img0, None)
    kp1, des1 = orb.detectAndCompute(img1, None)

    kp0_count = 0 if kp0 is None else len(kp0)
    kp1_count = 0 if kp1 is None else len(kp1)

    if des0 is None or des1 is None or kp0_count < 8 or kp1_count < 8:
        return {
            "status": "failed",
            "quality": "weak",
            "failure_reason": "not_enough_keypoints_or_descriptors",
            "kp0": kp0_count,
            "kp1": kp1_count,
            "runtime_s": time.perf_counter() - t0,
        }

    raw_knn = matcher.knnMatch(des0, des1, k=2)

    good = []
    for pair in raw_knn:
        if len(pair) < 2:
            continue

        m, n = pair
        if m.distance < ratio_test * n.distance:
            good.append(m)

    good_matches = len(good)

    if good_matches < 8:
        return {
            "status": "failed",
            "quality": "weak",
            "failure_reason": "not_enough_good_matches",
            "kp0": kp0_count,
            "kp1": kp1_count,
            "raw_matches": len(raw_knn),
            "good_matches": good_matches,
            "inliers": 0,
            "inlier_ratio": 0.0,
            "runtime_s": time.perf_counter() - t0,
        }

    pts0 = np.float32([kp0[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    pts1 = np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(
        pts0,
        pts1,
        cv2.RANSAC,
        ransacReprojThreshold=ransac_reproj_threshold_px,
    )

    if H is None or mask is None:
        return {
            "status": "failed",
            "quality": "weak",
            "failure_reason": "homography_failed",
            "kp0": kp0_count,
            "kp1": kp1_count,
            "raw_matches": len(raw_knn),
            "good_matches": good_matches,
            "inliers": 0,
            "inlier_ratio": 0.0,
            "runtime_s": time.perf_counter() - t0,
        }

    mask_flat = mask.ravel().astype(bool)
    inliers = int(mask_flat.sum())
    inlier_ratio = float(inliers / max(good_matches, 1))

    quality = classify_pair(
        good_matches=good_matches,
        inliers=inliers,
        inlier_ratio=inlier_ratio,
        min_good_matches=min_good_matches,
        min_inliers_good=min_inliers_good,
        min_inlier_ratio_good=min_inlier_ratio_good,
        min_inliers_medium=min_inliers_medium,
        min_inlier_ratio_medium=min_inlier_ratio_medium,
    )

    return {
        "status": "ok",
        "quality": quality,
        "failure_reason": "",
        "kp0": kp0_count,
        "kp1": kp1_count,
        "raw_matches": len(raw_knn),
        "good_matches": good_matches,
        "inliers": inliers,
        "inlier_ratio": inlier_ratio,
        "runtime_s": time.perf_counter() - t0,
    }


def reference_distance(row0: pd.Series, row1: pd.Series) -> float:
    if "x_enu_m" not in row0.index or "y_enu_m" not in row0.index:
        return float("nan")

    x0 = pd.to_numeric(row0.get("x_enu_m"), errors="coerce")
    y0 = pd.to_numeric(row0.get("y_enu_m"), errors="coerce")
    x1 = pd.to_numeric(row1.get("x_enu_m"), errors="coerce")
    y1 = pd.to_numeric(row1.get("y_enu_m"), errors="coerce")

    if not np.isfinite([x0, y0, x1, y1]).all():
        return float("nan")

    return float(np.hypot(x1 - x0, y1 - y0))


def filter_frames(
    df: pd.DataFrame,
    start_imgid: Optional[int],
    end_imgid: Optional[int],
    frame_step: int,
    max_frames: Optional[int],
) -> pd.DataFrame:
    out = df.copy()

    out["imgid"] = pd.to_numeric(out["imgid"], errors="coerce")
    out = out[out["imgid"].notna()].copy()
    out["imgid"] = out["imgid"].astype(int)

    out = out.sort_values("imgid").reset_index(drop=True)

    if start_imgid is not None:
        out = out[out["imgid"] >= start_imgid]

    if end_imgid is not None:
        out = out[out["imgid"] <= end_imgid]

    if frame_step > 1:
        out = out.iloc[::frame_step].copy()

    if max_frames is not None and max_frames > 0:
        out = out.head(max_frames).copy()

    return out.reset_index(drop=True)


def summarize_pairs(pair_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for stride, g in pair_df.groupby("stride"):
        total = len(g)
        ok = int((g["status"] == "ok").sum())
        failed = int((g["status"] == "failed").sum())
        good = int((g["quality"] == "good").sum())
        medium = int((g["quality"] == "medium").sum())
        weak = int((g["quality"] == "weak").sum())

        row = {
            "stride": int(stride),
            "total_pairs": int(total),
            "ok_pairs": ok,
            "failed_pairs": failed,
            "good_pairs": good,
            "medium_pairs": medium,
            "weak_pairs": weak,
            "ok_ratio": float(ok / total) if total else 0.0,
            "good_ratio": float(good / total) if total else 0.0,
            "median_good_matches": float(pd.to_numeric(g["good_matches"], errors="coerce").median()),
            "median_inliers": float(pd.to_numeric(g["inliers"], errors="coerce").median()),
            "median_inlier_ratio": float(pd.to_numeric(g["inlier_ratio"], errors="coerce").median()),
            "mean_runtime_s": float(pd.to_numeric(g["runtime_s"], errors="coerce").mean()),
            "total_runtime_s": float(pd.to_numeric(g["runtime_s"], errors="coerce").sum()),
            "median_reference_distance_m": float(pd.to_numeric(g["reference_distance_m"], errors="coerce").median()),
            "max_reference_distance_m": float(pd.to_numeric(g["reference_distance_m"], errors="coerce").max()),
        }

        rows.append(row)

    return pd.DataFrame(rows).sort_values("stride").reset_index(drop=True)


def plot_quality_summary(summary_df: pd.DataFrame, out_path: Path) -> None:
    fig, ax1 = plt.subplots(figsize=(10, 5))

    x = np.arange(len(summary_df))
    labels = [str(s) for s in summary_df["stride"]]

    ax1.bar(x, summary_df["good_ratio"])
    ax1.set_xlabel("Stride")
    ax1.set_ylabel("Good pair ratio")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylim(0, 1.05)

    ax2 = ax1.twinx()
    ax2.plot(x, summary_df["median_inlier_ratio"], marker="o")
    ax2.set_ylabel("Median inlier ratio")
    ax2.set_ylim(0, 1.05)

    ax1.set_title("ORB Stride Subset Quality Summary")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_reference_distance(summary_df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(summary_df["stride"], summary_df["median_reference_distance_m"], marker="o")
    ax.set_xlabel("Stride")
    ax.set_ylabel("Median reference distance per pair [m]")
    ax.set_title("Reference Displacement vs ORB Stride")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def run_stride_subset_diagnostics(
    config_path: str | Path,
    start_imgid: Optional[int],
    end_imgid: Optional[int],
    frame_step: int,
    strides: List[int],
    run_name: str,
    max_frames: Optional[int],
    resize_width: Optional[int],
    nfeatures: int,
    ratio_test: float,
    ransac_reproj_threshold_px: float,
    min_good_matches: int,
    min_inliers_good: int,
    min_inlier_ratio_good: float,
    min_inliers_medium: int,
    min_inlier_ratio_medium: float,
) -> Dict[str, Any]:
    project_root, _, _, output_dir, dataset_name = resolve_project_paths(config_path)

    sync_csv = output_dir / "metadata" / "synchronized_frames.csv"

    if not sync_csv.exists():
        raise FileNotFoundError(
            f"Missing synchronized frames CSV: {sync_csv}\n"
            "Run sync_zurich_frames.py first."
        )

    frames_df = pd.read_csv(sync_csv)

    if "image_path" not in frames_df.columns:
        raise ValueError("synchronized_frames.csv must contain image_path column.")

    selected = filter_frames(
        frames_df,
        start_imgid=start_imgid,
        end_imgid=end_imgid,
        frame_step=frame_step,
        max_frames=max_frames,
    )

    if len(selected) < 2:
        raise ValueError("Not enough frames selected for stride diagnostics.")

    metadata_dir = output_dir / "metadata" / STAGE_NAME / run_name
    report_dir = output_dir / "reports" / STAGE_NAME / run_name
    figure_dir = output_dir / "figures" / STAGE_NAME / run_name

    metadata_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    orb = cv2.ORB_create(nfeatures=nfeatures)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    pair_rows = []
    global_start = time.perf_counter()

    for stride in strides:
        if len(selected) <= stride:
            continue

        for i in range(0, len(selected) - stride):
            row0 = selected.iloc[i]
            row1 = selected.iloc[i + stride]

            img0_path = resolve_image_path(str(row0["image_path"]), project_root)
            img1_path = resolve_image_path(str(row1["image_path"]), project_root)

            result = evaluate_orb_pair(
                img0_path=img0_path,
                img1_path=img1_path,
                orb=orb,
                matcher=matcher,
                resize_width=resize_width,
                ratio_test=ratio_test,
                ransac_reproj_threshold_px=ransac_reproj_threshold_px,
                min_good_matches=min_good_matches,
                min_inliers_good=min_inliers_good,
                min_inlier_ratio_good=min_inlier_ratio_good,
                min_inliers_medium=min_inliers_medium,
                min_inlier_ratio_medium=min_inlier_ratio_medium,
            )

            pair_rows.append(
                {
                    "dataset_name": dataset_name,
                    "run_name": run_name,
                    "stride": int(stride),
                    "pair_index": int(i),
                    "imgid_0": int(row0["imgid"]),
                    "imgid_1": int(row1["imgid"]),
                    "image_0": str(img0_path),
                    "image_1": str(img1_path),
                    "reference_distance_m": reference_distance(row0, row1),
                    **result,
                }
            )

    pair_df = pd.DataFrame(pair_rows)

    if pair_df.empty:
        raise RuntimeError("No stride pairs were evaluated.")

    summary_df = summarize_pairs(pair_df)

    pair_csv = metadata_dir / "orb_stride_subset_pair_quality.csv"
    summary_csv = metadata_dir / "orb_stride_subset_summary.csv"
    summary_json = report_dir / "orb_stride_subset_summary.json"
    quality_plot = figure_dir / "orb_stride_subset_quality_summary.png"
    reference_plot = figure_dir / "orb_stride_subset_reference_distance.png"

    pair_df.to_csv(pair_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)

    plot_quality_summary(summary_df, quality_plot)
    plot_reference_distance(summary_df, reference_plot)

    total_runtime_s = time.perf_counter() - global_start

    summary = {
        "dataset_name": dataset_name,
        "stage": STAGE_NAME,
        "run_name": run_name,
        "config_path": str(config_path),
        "input_sync_csv": str(sync_csv),
        "selected_frames": int(len(selected)),
        "selected_imgid_min": int(selected["imgid"].min()),
        "selected_imgid_max": int(selected["imgid"].max()),
        "frame_step": int(frame_step),
        "strides": strides,
        "total_pairs_evaluated": int(len(pair_df)),
        "total_runtime_s": float(total_runtime_s),
        "parameters": {
            "resize_width": resize_width,
            "nfeatures": nfeatures,
            "ratio_test": ratio_test,
            "ransac_reproj_threshold_px": ransac_reproj_threshold_px,
            "min_good_matches": min_good_matches,
            "min_inliers_good": min_inliers_good,
            "min_inlier_ratio_good": min_inlier_ratio_good,
            "min_inliers_medium": min_inliers_medium,
            "min_inlier_ratio_medium": min_inlier_ratio_medium,
        },
        "stride_summary": summary_df.to_dict(orient="records"),
        "outputs": {
            "pair_csv": str(pair_csv),
            "summary_csv": str(summary_csv),
            "summary_json": str(summary_json),
            "quality_plot": str(quality_plot),
            "reference_plot": str(reference_plot),
        },
    }

    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ORB stride diagnostics on a selected full-dataset frame subset."
    )
    parser.add_argument("--config", required=True, help="Path to dataset YAML config.")
    parser.add_argument("--start-imgid", type=int, default=None, help="First imgid to include.")
    parser.add_argument("--end-imgid", type=int, default=None, help="Last imgid to include.")
    parser.add_argument("--frame-step", type=int, default=1, help="Keep every Nth selected frame.")
    parser.add_argument("--max-frames", type=int, default=None, help="Safety cap on selected frames.")
    parser.add_argument("--strides", default="1,2,3,5,10", help="Comma-separated stride list.")
    parser.add_argument("--run-name", default=None, help="Output run name.")

    parser.add_argument("--resize-width", type=int, default=960, help="Resize width for faster diagnostics. Use 0 for original size.")
    parser.add_argument("--nfeatures", type=int, default=2000)
    parser.add_argument("--ratio-test", type=float, default=0.75)
    parser.add_argument("--ransac-threshold", type=float, default=5.0)

    parser.add_argument("--min-good-matches", type=int, default=50)
    parser.add_argument("--min-inliers-good", type=int, default=50)
    parser.add_argument("--min-inlier-ratio-good", type=float, default=0.50)
    parser.add_argument("--min-inliers-medium", type=int, default=25)
    parser.add_argument("--min-inlier-ratio-medium", type=float, default=0.35)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.frame_step <= 0:
        raise ValueError("--frame-step must be positive.")

    strides = parse_strides(args.strides)

    resize_width = None if args.resize_width is None or args.resize_width <= 0 else args.resize_width

    run_name = args.run_name
    if run_name is None:
        start = "start" if args.start_imgid is None else f"{args.start_imgid:05d}"
        end = "end" if args.end_imgid is None else f"{args.end_imgid:05d}"
        run_name = f"imgid_{start}_{end}_step{args.frame_step}"

    summary = run_stride_subset_diagnostics(
        config_path=args.config,
        start_imgid=args.start_imgid,
        end_imgid=args.end_imgid,
        frame_step=args.frame_step,
        strides=strides,
        run_name=run_name,
        max_frames=args.max_frames,
        resize_width=resize_width,
        nfeatures=args.nfeatures,
        ratio_test=args.ratio_test,
        ransac_reproj_threshold_px=args.ransac_threshold,
        min_good_matches=args.min_good_matches,
        min_inliers_good=args.min_inliers_good,
        min_inlier_ratio_good=args.min_inlier_ratio_good,
        min_inliers_medium=args.min_inliers_medium,
        min_inlier_ratio_medium=args.min_inlier_ratio_medium,
    )

    print("ORB stride subset diagnostics generated")
    print("---------------------------------------")
    print(f"Dataset:             {summary['dataset_name']}")
    print(f"Run name:            {summary['run_name']}")
    print(f"Selected frames:     {summary['selected_frames']}")
    print(f"Imgid range:          {summary['selected_imgid_min']} to {summary['selected_imgid_max']}")
    print(f"Frame step:           {summary['frame_step']}")
    print(f"Strides:              {summary['strides']}")
    print(f"Pairs evaluated:      {summary['total_pairs_evaluated']}")
    print(f"Total runtime [s]:    {summary['total_runtime_s']:.2f}")

    print("\nStride summary")
    print("--------------")
    for row in summary["stride_summary"]:
        print(
            f"stride {row['stride']}: "
            f"good {row['good_pairs']}/{row['total_pairs']}, "
            f"ok_ratio {row['ok_ratio']:.3f}, "
            f"median inlier ratio {row['median_inlier_ratio']:.3f}, "
            f"median inliers {row['median_inliers']:.1f}, "
            f"median ref dist {row['median_reference_distance_m']:.3f} m"
        )

    print("\nOutputs")
    print("-------")
    for k, v in summary["outputs"].items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
