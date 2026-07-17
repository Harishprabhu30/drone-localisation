'''
Command Used:
1. Trial Run

export PYTHONPATH=$PWD/src

python scripts/satloc/s6a/s6a_1_orb_affine_stride_diagnostics.py \
  --sequence traj01 \
  --strides 1,2,5 \
  --resize-long 960 \
  --nfeatures 1200 \
  --ratio 0.75 \
  --ransac-thresh 3.0 \
  --max-frames 150

2. Full run of traj 01:
export PYTHONPATH=$PWD/src

python scripts/satloc/s6a/s6a_1_orb_affine_stride_diagnostics.py \
  --sequence traj01 \
  --strides 1,2,5 \
  --resize-long 960 \
  --nfeatures 1200 \
  --ratio 0.75 \
  --ransac-thresh 3.0
'''


from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_INDEX = Path("outputs/satloc/metadata/uav_frames_index_enriched.csv")
DEFAULT_OUTPUT_ROOT = Path("outputs/satloc")


def parse_strides(value: str) -> list[int]:
    strides: list[int] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        stride = int(item)
        if stride <= 0:
            raise argparse.ArgumentTypeError("Strides must be positive integers.")
        strides.append(stride)
    if not strides:
        raise argparse.ArgumentTypeError("At least one stride is required.")
    return sorted(set(strides))


def resolve_image_path(value: Any) -> Path:
    path = Path(str(value)).expanduser()
    candidates = [path, Path.cwd() / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return path


def resize_long_side(image: np.ndarray, target_long_side: int) -> np.ndarray:
    height, width = image.shape[:2]
    long_side = max(height, width)
    if target_long_side <= 0 or long_side == target_long_side:
        return image
    scale = target_long_side / float(long_side)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(image, (new_width, new_height), interpolation=interpolation)


def finite_stat(values: pd.Series | np.ndarray, fn, default: float = float("nan")) -> float:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return default
    return float(fn(array))


def prepare_sequence_manifest(
    index_path: Path,
    sequence: str,
    max_frames: int,
) -> pd.DataFrame:
    if not index_path.exists():
        raise FileNotFoundError(f"Missing SatLoc index: {index_path}")

    index_df = pd.read_csv(index_path)
    required = {
        "sequence",
        "image_path",
        "filename",
        "token0_id",
        "frame_index_in_sequence",
        "x_enu_m",
        "y_enu_m",
    }
    missing = sorted(required.difference(index_df.columns))
    if missing:
        raise RuntimeError(f"Missing required index columns: {missing}")

    traj = index_df[index_df["sequence"].astype(str) == sequence].copy()
    if traj.empty:
        raise RuntimeError(f"No rows found for sequence {sequence!r}.")

    traj["token0_id"] = pd.to_numeric(traj["token0_id"], errors="coerce")
    traj["frame_index_in_sequence"] = pd.to_numeric(
        traj["frame_index_in_sequence"], errors="coerce"
    )
    traj = traj.dropna(subset=["token0_id", "frame_index_in_sequence"])
    traj = traj.sort_values("frame_index_in_sequence", kind="mergesort").reset_index(drop=True)

    if max_frames > 0:
        traj = traj.iloc[:max_frames].copy().reset_index(drop=True)

    expected_frame_index = np.arange(len(traj), dtype=int)
    actual_frame_index = traj["frame_index_in_sequence"].to_numpy(dtype=int)
    if not np.array_equal(actual_frame_index, expected_frame_index):
        raise RuntimeError(
            "frame_index_in_sequence is not contiguous from zero after filtering. "
            "Re-run S6A.0B or inspect the index before continuing."
        )

    token_ids = traj["token0_id"].to_numpy(dtype=int)
    if len(token_ids) > 1 and not np.all(np.diff(token_ids) == 1):
        raise RuntimeError(
            "token0_id is not strictly consecutive in frame_index_in_sequence order."
        )

    traj.insert(0, "sequence_frame_id", expected_frame_index)
    traj["image_path_resolved"] = traj["image_path"].map(
        lambda value: str(resolve_image_path(value))
    )

    missing_paths = [
        path for path in traj["image_path_resolved"].map(Path) if not path.exists()
    ]
    if missing_paths:
        raise FileNotFoundError(
            f"{len(missing_paths)} image files are missing. First missing: {missing_paths[0]}"
        )

    return traj


def extract_feature_cache(
    manifest: pd.DataFrame,
    resize_long: int,
    nfeatures: int,
) -> tuple[list[dict[str, Any]], pd.DataFrame, float]:
    orb = cv2.ORB_create(
        nfeatures=nfeatures,
        scaleFactor=1.2,
        nlevels=8,
        edgeThreshold=31,
        patchSize=31,
        fastThreshold=20,
    )

    cache: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    start = time.perf_counter()

    total = len(manifest)
    for index, row in manifest.iterrows():
        path = Path(row["image_path_resolved"])
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise RuntimeError(f"OpenCV could not read: {path}")
        gray = resize_long_side(gray, resize_long)
        keypoints, descriptors = orb.detectAndCompute(gray, None)
        points = (
            np.asarray([keypoint.pt for keypoint in keypoints], dtype=np.float32)
            if keypoints
            else np.empty((0, 2), dtype=np.float32)
        )
        cache.append(
            {
                "points": points,
                "descriptors": descriptors,
                "height": int(gray.shape[0]),
                "width": int(gray.shape[1]),
            }
        )
        rows.append(
            {
                "sequence_frame_id": int(row["sequence_frame_id"]),
                "token0_id": int(row["token0_id"]),
                "keypoints": int(len(points)),
                "width": int(gray.shape[1]),
                "height": int(gray.shape[0]),
                "read_ok": True,
            }
        )
        if (index + 1) % 100 == 0 or index + 1 == total:
            print(f"Feature cache: {index + 1}/{total}")

    elapsed = time.perf_counter() - start
    return cache, pd.DataFrame(rows), elapsed


def match_pair(
    feature_a: dict[str, Any],
    feature_b: dict[str, Any],
    matcher: cv2.BFMatcher,
    ratio: float,
    ransac_thresh: float,
) -> dict[str, Any]:
    descriptors_a = feature_a["descriptors"]
    descriptors_b = feature_b["descriptors"]
    points_a_all = feature_a["points"]
    points_b_all = feature_b["points"]

    if descriptors_a is None or descriptors_b is None:
        return {
            "status": "no_descriptors",
            "good_matches": 0,
            "inliers": 0,
            "inlier_ratio": 0.0,
            "affine_ok": False,
        }

    pair_start = time.perf_counter()
    knn_matches = matcher.knnMatch(descriptors_a, descriptors_b, k=2)
    good = []
    for candidates in knn_matches:
        if len(candidates) != 2:
            continue
        best, second = candidates
        if best.distance < ratio * second.distance:
            good.append(best)

    if len(good) < 3:
        return {
            "status": "too_few_matches",
            "good_matches": int(len(good)),
            "inliers": 0,
            "inlier_ratio": 0.0,
            "affine_ok": False,
            "elapsed_ms": (time.perf_counter() - pair_start) * 1000.0,
        }

    points_a = np.asarray(
        [points_a_all[match.queryIdx] for match in good], dtype=np.float32
    )
    points_b = np.asarray(
        [points_b_all[match.trainIdx] for match in good], dtype=np.float32
    )

    affine, inlier_mask = cv2.estimateAffinePartial2D(
        points_a,
        points_b,
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_thresh,
        maxIters=3000,
        confidence=0.995,
        refineIters=10,
    )

    elapsed_ms = (time.perf_counter() - pair_start) * 1000.0
    if affine is None or inlier_mask is None:
        return {
            "status": "affine_failed",
            "good_matches": int(len(good)),
            "inliers": 0,
            "inlier_ratio": 0.0,
            "affine_ok": False,
            "elapsed_ms": elapsed_ms,
        }

    mask = np.asarray(inlier_mask).ravel().astype(bool)
    inliers = int(mask.sum())
    inlier_ratio = inliers / max(len(good), 1)

    a00, a01, tx = [float(value) for value in affine[0]]
    a10, a11, ty = [float(value) for value in affine[1]]
    scale_x = math.hypot(a00, a10)
    scale_y = math.hypot(a01, a11)
    scale = 0.5 * (scale_x + scale_y)
    rotation_deg = math.degrees(math.atan2(a10, a00))

    center = np.array(
        [feature_a["width"] / 2.0, feature_a["height"] / 2.0, 1.0],
        dtype=float,
    )
    center_transformed = affine @ center
    center_dx = float(center_transformed[0] - center[0])
    center_dy = float(center_transformed[1] - center[1])
    center_motion = math.hypot(center_dx, center_dy)

    if (
        len(good) >= 30
        and inliers >= 20
        and inlier_ratio >= 0.35
        and 0.70 <= scale <= 1.40
    ):
        quality = "good"
    else:
        quality = "weak"

    return {
        "status": quality,
        "good_matches": int(len(good)),
        "inliers": inliers,
        "inlier_ratio": float(inlier_ratio),
        "affine_ok": True,
        "affine_a00": a00,
        "affine_a01": a01,
        "affine_a10": a10,
        "affine_a11": a11,
        "affine_tx_px": tx,
        "affine_ty_px": ty,
        "affine_scale": float(scale),
        "affine_rotation_deg": float(rotation_deg),
        "center_content_dx_px": center_dx,
        "center_content_dy_px": center_dy,
        "center_content_motion_px": float(center_motion),
        "elapsed_ms": float(elapsed_ms),
    }


def run_stride_diagnostics(
    manifest: pd.DataFrame,
    feature_cache: list[dict[str, Any]],
    strides: list[int],
    ratio: float,
    ransac_thresh: float,
) -> pd.DataFrame:
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    rows: list[dict[str, Any]] = []

    for stride in strides:
        chain_indices = np.arange(0, len(manifest), stride, dtype=int)
        attempted = max(len(chain_indices) - 1, 0)
        print(f"\nStride {stride}: {attempted} chain pairs")

        for pair_number, (index_a, index_b) in enumerate(
            zip(chain_indices[:-1], chain_indices[1:])
        ):
            row_a = manifest.iloc[int(index_a)]
            row_b = manifest.iloc[int(index_b)]
            result = match_pair(
                feature_cache[int(index_a)],
                feature_cache[int(index_b)],
                matcher,
                ratio,
                ransac_thresh,
            )

            ref_dx = float(row_b["x_enu_m"] - row_a["x_enu_m"])
            ref_dy = float(row_b["y_enu_m"] - row_a["y_enu_m"])
            ref_step = math.hypot(ref_dx, ref_dy)

            rows.append(
                {
                    "stride": int(stride),
                    "pair_number": int(pair_number),
                    "frame_index_a": int(index_a),
                    "frame_index_b": int(index_b),
                    "token0_a": int(row_a["token0_id"]),
                    "token0_b": int(row_b["token0_id"]),
                    "reference_dx_m_eval_only": ref_dx,
                    "reference_dy_m_eval_only": ref_dy,
                    "reference_step_m_eval_only": ref_step,
                    **result,
                }
            )

            if (pair_number + 1) % 100 == 0 or pair_number + 1 == attempted:
                print(f"  pairs: {pair_number + 1}/{attempted}")

    return pd.DataFrame(rows)


def build_summary(pair_df: pd.DataFrame, feature_seconds: float) -> pd.DataFrame:
    summaries: list[dict[str, Any]] = []
    for stride, group in pair_df.groupby("stride", sort=True):
        affine_success = group["affine_ok"].astype(bool)
        successful = group[affine_success]
        summaries.append(
            {
                "stride": int(stride),
                "frames_in_chain": int(len(group) + 1),
                "attempted_pairs": int(len(group)),
                "affine_successes": int(affine_success.sum()),
                "affine_success_rate": float(affine_success.mean()),
                "good_quality_pairs": int((group["status"] == "good").sum()),
                "good_quality_rate": float((group["status"] == "good").mean()),
                "good_matches_median": finite_stat(successful["good_matches"], np.median),
                "inliers_median": finite_stat(successful["inliers"], np.median),
                "inlier_ratio_median": finite_stat(successful["inlier_ratio"], np.median),
                "inlier_ratio_p05": finite_stat(
                    successful["inlier_ratio"], lambda x: np.percentile(x, 5)
                ),
                "center_motion_px_median": finite_stat(
                    successful["center_content_motion_px"], np.median
                ),
                "abs_rotation_deg_median": finite_stat(
                    np.abs(successful["affine_rotation_deg"]), np.median
                ),
                "abs_rotation_deg_p95": finite_stat(
                    np.abs(successful["affine_rotation_deg"]),
                    lambda x: np.percentile(x, 95),
                ),
                "scale_median": finite_stat(successful["affine_scale"], np.median),
                "abs_scale_error_p95": finite_stat(
                    np.abs(successful["affine_scale"] - 1.0),
                    lambda x: np.percentile(x, 95),
                ),
                "reference_step_m_median_eval_only": finite_stat(
                    group["reference_step_m_eval_only"], np.median
                ),
                "pair_runtime_ms_mean": finite_stat(successful["elapsed_ms"], np.mean),
                "feature_cache_seconds_shared": float(feature_seconds),
            }
        )
    return pd.DataFrame(summaries)


def save_plots(pair_df: pd.DataFrame, summary_df: pd.DataFrame, figures_dir: Path) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 6))
    for stride, group in pair_df.groupby("stride", sort=True):
        plt.plot(
            group["pair_number"],
            group["inlier_ratio"],
            linewidth=1.0,
            label=f"stride {stride}",
        )
    plt.xlabel("Pair number in stride chain")
    plt.ylabel("RANSAC inlier ratio")
    plt.title("S6A.1 ORB affine inlier ratio")
    plt.ylim(0.0, 1.02)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "s6a1_inlier_ratio_by_stride.png", dpi=180)
    plt.close()

    plt.figure(figsize=(10, 6))
    x = np.arange(len(summary_df))
    width = 0.35
    plt.bar(x - width / 2, summary_df["good_matches_median"], width, label="Good matches")
    plt.bar(x + width / 2, summary_df["inliers_median"], width, label="RANSAC inliers")
    plt.xticks(x, [f"stride {value}" for value in summary_df["stride"]])
    plt.ylabel("Median count")
    plt.title("S6A.1 median ORB/RANSAC evidence")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "s6a1_match_evidence_summary.png", dpi=180)
    plt.close()

    plt.figure(figsize=(10, 6))
    for stride, group in pair_df[pair_df["affine_ok"]].groupby("stride", sort=True):
        plt.scatter(
            group["reference_step_m_eval_only"],
            group["center_content_motion_px"],
            s=12,
            alpha=0.5,
            label=f"stride {stride}",
        )
    plt.xlabel("Reference step [m] — evaluation only")
    plt.ylabel("Estimated image-center content motion [px]")
    plt.title("S6A.1 visual motion versus reference displacement")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "s6a1_visual_motion_vs_reference_step.png", dpi=180)
    plt.close()

    plt.figure(figsize=(12, 6))
    for stride, group in pair_df[pair_df["affine_ok"]].groupby("stride", sort=True):
        plt.scatter(
            group["affine_rotation_deg"],
            group["affine_scale"],
            s=12,
            alpha=0.5,
            label=f"stride {stride}",
        )
    plt.xlabel("Affine rotation [deg]")
    plt.ylabel("Affine scale")
    plt.title("S6A.1 affine rotation/scale stability")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "s6a1_rotation_scale_stability.png", dpi=180)
    plt.close()


def print_summary(summary_df: pd.DataFrame, feature_seconds: float) -> None:
    print("\nS6A.1 ORB affine stride diagnostics")
    print("-----------------------------------")
    print(f"Shared feature-cache time: {feature_seconds:.2f} s")
    for row in summary_df.to_dict(orient="records"):
        print(f"\nStride {row['stride']}")
        print(f"  Frames in chain:          {row['frames_in_chain']}")
        print(f"  Attempted pairs:          {row['attempted_pairs']}")
        print(f"  Affine success rate:      {row['affine_success_rate']:.3f}")
        print(f"  Good-quality rate:        {row['good_quality_rate']:.3f}")
        print(f"  Good matches median:      {row['good_matches_median']:.1f}")
        print(f"  RANSAC inliers median:    {row['inliers_median']:.1f}")
        print(f"  Inlier ratio median:      {row['inlier_ratio_median']:.3f}")
        print(f"  Inlier ratio p05:         {row['inlier_ratio_p05']:.3f}")
        print(f"  Center motion median px:  {row['center_motion_px_median']:.3f}")
        print(f"  |rotation| p95 deg:       {row['abs_rotation_deg_p95']:.3f}")
        print(f"  Scale median:             {row['scale_median']:.6f}")
        print(f"  |scale-1| p95:            {row['abs_scale_error_p95']:.6f}")
        print(
            "  Reference step median m: "
            f"{row['reference_step_m_median_eval_only']:.3f} (evaluation only)"
        )
        print(f"  Pair runtime mean ms:     {row['pair_runtime_ms_mean']:.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "SatLoc S6A.1 ORB + partial-affine RANSAC diagnostics on validated "
            "traj01 temporal order."
        )
    )
    parser.add_argument("--index-path", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--sequence", default="traj01")
    parser.add_argument("--strides", type=parse_strides, default=parse_strides("1,2,5"))
    parser.add_argument("--resize-long", type=int, default=960)
    parser.add_argument("--nfeatures", type=int, default=1200)
    parser.add_argument("--ratio", type=float, default=0.75)
    parser.add_argument("--ransac-thresh", type=float, default=3.0)
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Optional prefix length for a smoke test; 0 means the full sequence.",
    )
    args = parser.parse_args()

    metadata_dir = args.output_root / "metadata" / "s6a_relative_motion"
    reports_dir = args.output_root / "reports" / "s6a_relative_motion"
    figures_dir = args.output_root / "figures" / "s6a_relative_motion"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    manifest = prepare_sequence_manifest(args.index_path, args.sequence, args.max_frames)
    manifest_path = metadata_dir / "s6a_sequence_manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    feature_cache, feature_df, feature_seconds = extract_feature_cache(
        manifest,
        resize_long=args.resize_long,
        nfeatures=args.nfeatures,
    )
    feature_path = metadata_dir / "s6a1_orb_frame_features.csv"
    feature_df.to_csv(feature_path, index=False)

    pair_df = run_stride_diagnostics(
        manifest,
        feature_cache,
        strides=args.strides,
        ratio=args.ratio,
        ransac_thresh=args.ransac_thresh,
    )
    pair_path = metadata_dir / "s6a1_orb_affine_pair_diagnostics.csv"
    pair_df.to_csv(pair_path, index=False)

    summary_df = build_summary(pair_df, feature_seconds)
    summary_csv_path = metadata_dir / "s6a1_orb_affine_stride_summary.csv"
    summary_df.to_csv(summary_csv_path, index=False)

    payload = {
        "stage": "S6A.1",
        "sequence": args.sequence,
        "ordering_rule": (
            "Sort by frame_index_in_sequence and assert token0_id is strictly consecutive. "
            "token1_order is not used as temporal order."
        ),
        "reference_usage_rule": (
            "x_enu_m/y_enu_m are used only for post-estimation diagnostics, never for "
            "feature matching, affine estimation, or quality scoring."
        ),
        "configuration": {
            "index_path": str(args.index_path),
            "strides": args.strides,
            "resize_long": args.resize_long,
            "nfeatures": args.nfeatures,
            "ratio": args.ratio,
            "ransac_thresh": args.ransac_thresh,
            "max_frames": args.max_frames,
        },
        "frames": int(len(manifest)),
        "feature_cache_seconds": float(feature_seconds),
        "summaries": summary_df.to_dict(orient="records"),
    }
    summary_json_path = reports_dir / "s6a1_orb_affine_stride_summary.json"
    with summary_json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    save_plots(pair_df, summary_df, figures_dir)
    print_summary(summary_df, feature_seconds)

    print("\nSaved outputs")
    print("-------------")
    print(manifest_path)
    print(feature_path)
    print(pair_path)
    print(summary_csv_path)
    print(summary_json_path)
    print(figures_dir / "s6a1_inlier_ratio_by_stride.png")
    print(figures_dir / "s6a1_match_evidence_summary.png")
    print(figures_dir / "s6a1_visual_motion_vs_reference_step.png")
    print(figures_dir / "s6a1_rotation_scale_stability.png")


if __name__ == "__main__":
    main()
