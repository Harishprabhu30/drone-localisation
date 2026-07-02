from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


STAGE_NAME = "s3_5_visual_domain_traj01"
DEFAULT_UAV_TRAJ = "outputs/satloc/trajectories/uav_reference_trajectory.csv"
DEFAULT_TILE_INDEX = "outputs/satloc/metadata/satellite_tiles_index_enriched.csv"


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_config(config_path: str | Path) -> dict:
    with Path(config_path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_int_list(text: str | None) -> List[int]:
    if not text:
        return []
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_ranges(text: str | None) -> List[Tuple[int, int, int]]:
    """
    Example:
      1-100:10,250-350:8
    """
    if not text:
        return []

    ranges = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue

        range_part, count_part = part.split(":")
        start_text, end_text = range_part.split("-")
        ranges.append((int(start_text), int(end_text), int(count_part)))

    return ranges


def sample_evenly(start: int, end: int, count: int) -> List[int]:
    if count <= 0:
        return []

    values = np.linspace(start, end, count)
    return sorted(set(int(round(v)) for v in values))


def resize_to_max_dim(img: np.ndarray, max_dim: int = 900) -> np.ndarray:
    h, w = img.shape[:2]
    scale = max_dim / max(h, w)

    if scale >= 1.0:
        return img

    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def make_luma(rgb: np.ndarray) -> np.ndarray:
    r = rgb[:, :, 0].astype(np.float32)
    g = rgb[:, :, 1].astype(np.float32)
    b = rgb[:, :, 2].astype(np.float32)

    luma = 0.299 * r + 0.587 * g + 0.114 * b
    return np.clip(luma, 0, 255).astype(np.uint8)


def make_clahe(gray: np.ndarray, clip_limit: float, tile_size: int) -> np.ndarray:
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=(tile_size, tile_size),
    )
    return clahe.apply(gray)


def make_bilateral(
    gray: np.ndarray,
    d: int,
    sigma_color: float,
    sigma_space: float,
) -> np.ndarray:
    return cv2.bilateralFilter(
        gray,
        d=d,
        sigmaColor=sigma_color,
        sigmaSpace=sigma_space,
    )


def build_variants(
    rgb: np.ndarray,
    clahe_clip_limit: float,
    clahe_tile_size: int,
    alt_clahe_clip_limit: float,
    alt_clahe_tile_size: int,
    bilateral_d: int,
    bilateral_sigma_color: float,
    bilateral_sigma_space: float,
) -> Dict[str, np.ndarray]:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    luma = make_luma(rgb)

    clahe_luma = make_clahe(
        luma,
        clip_limit=clahe_clip_limit,
        tile_size=clahe_tile_size,
    )

    alt_clahe_luma = make_clahe(
        luma,
        clip_limit=alt_clahe_clip_limit,
        tile_size=alt_clahe_tile_size,
    )

    bilateral_clahe = make_bilateral(
        clahe_luma,
        d=bilateral_d,
        sigma_color=bilateral_sigma_color,
        sigma_space=bilateral_sigma_space,
    )

    bilateral_alt_clahe = make_bilateral(
        alt_clahe_luma,
        d=bilateral_d,
        sigma_color=bilateral_sigma_color,
        sigma_space=bilateral_sigma_space,
    )

    return {
        "V1_luma": luma,
        "V2_clahe_luma": clahe_luma,
        "V3_bilateral_clahe": bilateral_clahe,
        "V4_bilateral_alt_clahe": bilateral_alt_clahe,
    }


def select_uav_rows(uav_df: pd.DataFrame, frames: List[int], ranges: List[Tuple[int, int, int]]) -> pd.DataFrame:
    selected_frames = set(frames)

    for start, end, count in ranges:
        selected_frames.update(sample_evenly(start, end, count))

    if not selected_frames:
        raise ValueError("No frames selected. Use --frames or --ranges.")

    traj01 = uav_df[uav_df["sequence"] == "traj01"].copy()
    traj01["frame_index_in_sequence"] = traj01["frame_index_in_sequence"].astype(int)

    selected = (
        traj01[traj01["frame_index_in_sequence"].isin(sorted(selected_frames))]
        .copy()
        .sort_values("frame_index_in_sequence")
        .reset_index(drop=True)
    )

    if selected.empty:
        raise ValueError("No selected traj01 frames found in UAV trajectory CSV.")

    return selected


def tile_contains_lonlat(tile_row: pd.Series, lon: float, lat: float) -> bool:
    lon_min = min(float(tile_row["lon_tl"]), float(tile_row["lon_br"]))
    lon_max = max(float(tile_row["lon_tl"]), float(tile_row["lon_br"]))
    lat_min = min(float(tile_row["lat_tl"]), float(tile_row["lat_br"]))
    lat_max = max(float(tile_row["lat_tl"]), float(tile_row["lat_br"]))

    return (lon_min <= lon <= lon_max) and (lat_min <= lat <= lat_max)


def find_true_tile(uav_row: pd.Series, tile_df: pd.DataFrame) -> Tuple[pd.Series, int, bool, float]:
    """
    Finds satellite tiles whose bbox contains the UAV GT lon/lat.
    If multiple tiles contain the point, choose the one with closest tile center in UTM.
    If none contain the point, choose nearest center as fallback.
    """
    lon = float(uav_row["lon"])
    lat = float(uav_row["lat"])

    contains_mask = tile_df.apply(lambda r: tile_contains_lonlat(r, lon, lat), axis=1)
    candidate_df = tile_df[contains_mask].copy()

    found_containing = not candidate_df.empty

    if candidate_df.empty:
        candidate_df = tile_df.copy()

    uav_x = float(uav_row["utm_x_m"])
    uav_y = float(uav_row["utm_y_m"])

    dx = candidate_df["utm_x_m"].astype(float) - uav_x
    dy = candidate_df["utm_y_m"].astype(float) - uav_y
    distances = np.sqrt(dx * dx + dy * dy)

    best_idx = distances.idxmin()
    best_tile = candidate_df.loc[best_idx]
    center_error_m = float(distances.loc[best_idx])

    return best_tile, int(contains_mask.sum()), found_containing, center_error_m


def compute_orb(image_u8: np.ndarray, nfeatures: int) -> Tuple[List[cv2.KeyPoint], np.ndarray | None]:
    orb = cv2.ORB_create(nfeatures=nfeatures)
    kps, desc = orb.detectAndCompute(image_u8, None)
    return kps if kps is not None else [], desc


def match_orb_descriptors(
    desc_uav: np.ndarray | None,
    desc_tile: np.ndarray | None,
    ratio: float,
) -> Tuple[List[cv2.DMatch], int]:
    if desc_uav is None or desc_tile is None:
        return [], 0

    if len(desc_uav) < 2 or len(desc_tile) < 2:
        return [], 0

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    raw_knn = matcher.knnMatch(desc_uav, desc_tile, k=2)

    good = []
    valid_knn = 0

    for pair in raw_knn:
        if len(pair) < 2:
            continue

        m, n = pair
        valid_knn += 1

        if m.distance < ratio * n.distance:
            good.append(m)

    good = sorted(good, key=lambda m: m.distance)
    return good, valid_knn


def estimate_homography(
    kp_uav: List[cv2.KeyPoint],
    kp_tile: List[cv2.KeyPoint],
    matches: List[cv2.DMatch],
    ransac_thresh: float,
) -> Tuple[bool, int, float, np.ndarray | None]:
    if len(matches) < 8:
        return False, 0, 0.0, None

    src_pts = np.float32([kp_uav[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_tile[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_thresh)

    if H is None or mask is None:
        return False, 0, 0.0, None

    inliers = int(mask.ravel().sum())
    inlier_ratio = float(inliers / max(len(matches), 1))

    return True, inliers, inlier_ratio, mask.ravel().astype(int)


def draw_match_image(
    uav_rgb: np.ndarray,
    tile_rgb: np.ndarray,
    kp_uav: List[cv2.KeyPoint],
    kp_tile: List[cv2.KeyPoint],
    matches: List[cv2.DMatch],
    inlier_mask: np.ndarray | None,
    max_draw: int = 60,
) -> np.ndarray:
    if not matches:
        blank = np.zeros((300, 900, 3), dtype=np.uint8)
        blank[:, :] = [255, 255, 255]
        return blank

    ranked = sorted(list(enumerate(matches)), key=lambda x: x[1].distance)
    ranked = ranked[:max_draw]

    selected_indices = [idx for idx, _ in ranked]
    selected_matches = [m for _, m in ranked]

    draw_mask = None
    if inlier_mask is not None:
        draw_mask = [int(inlier_mask[idx]) for idx in selected_indices]

    uav_bgr = cv2.cvtColor(uav_rgb, cv2.COLOR_RGB2BGR)
    tile_bgr = cv2.cvtColor(tile_rgb, cv2.COLOR_RGB2BGR)

    out_bgr = cv2.drawMatches(
        uav_bgr,
        kp_uav,
        tile_bgr,
        kp_tile,
        selected_matches,
        None,
        matchColor=(0, 255, 0),
        singlePointColor=(180, 180, 180),
        matchesMask=draw_mask,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    out_rgb = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)
    return out_rgb


def run_variant_match(
    variant_name: str,
    uav_img_u8: np.ndarray,
    tile_img_u8: np.ndarray,
    uav_rgb: np.ndarray,
    tile_rgb: np.ndarray,
    nfeatures: int,
    ratio: float,
    ransac_thresh: float,
) -> Tuple[dict, np.ndarray]:
    t0 = time.perf_counter()

    kp_uav, desc_uav = compute_orb(uav_img_u8, nfeatures=nfeatures)
    kp_tile, desc_tile = compute_orb(tile_img_u8, nfeatures=nfeatures)

    good_matches, raw_knn_count = match_orb_descriptors(desc_uav, desc_tile, ratio=ratio)

    homography_success, inliers, inlier_ratio, inlier_mask = estimate_homography(
        kp_uav,
        kp_tile,
        good_matches,
        ransac_thresh=ransac_thresh,
    )

    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    if good_matches:
        distances = np.array([m.distance for m in good_matches], dtype=np.float32)
        mean_distance = float(distances.mean())
        median_distance = float(np.median(distances))
    else:
        mean_distance = None
        median_distance = None

    score = float(inliers + 0.1 * len(good_matches))

    match_img = draw_match_image(
        uav_rgb=uav_rgb,
        tile_rgb=tile_rgb,
        kp_uav=kp_uav,
        kp_tile=kp_tile,
        matches=good_matches,
        inlier_mask=inlier_mask,
        max_draw=60,
    )

    result = {
        "variant": variant_name,
        "uav_keypoints": int(len(kp_uav)),
        "tile_keypoints": int(len(kp_tile)),
        "raw_knn_matches": int(raw_knn_count),
        "good_matches": int(len(good_matches)),
        "homography_success": bool(homography_success),
        "ransac_inliers": int(inliers),
        "inlier_ratio": float(inlier_ratio),
        "mean_good_match_distance": mean_distance,
        "median_good_match_distance": median_distance,
        "score": score,
        "elapsed_ms": float(elapsed_ms),
    }

    return result, match_img


def save_frame_panel(
    frame_row: pd.Series,
    tile_row: pd.Series,
    match_images: Dict[str, np.ndarray],
    variant_results: List[dict],
    output_path: Path,
) -> None:
    ensure_dir(output_path.parent)

    variants = list(match_images.keys())

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    axes_flat = axes.flatten()

    for ax, variant in zip(axes_flat, variants):
        ax.imshow(match_images[variant])
        ax.axis("off")

        row = next(r for r in variant_results if r["variant"] == variant)
        ax.set_title(
            f"{variant}\n"
            f"good={row['good_matches']}, inliers={row['ransac_inliers']}, "
            f"inlier_ratio={row['inlier_ratio']:.3f}, score={row['score']:.1f}",
            fontsize=9,
        )

    for ax in axes_flat[len(variants):]:
        ax.axis("off")

    fig.suptitle(
        f"S3.5D.3 UAV ↔ true satellite tile ORB matching\n"
        f"frame={int(frame_row['frame_index_in_sequence'])}, UAV={frame_row['filename']}, "
        f"tile={tile_row['filename']}",
        fontsize=12,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare ORB matching between traj01 UAV frames and their true satellite tiles."
    )

    parser.add_argument("--config", required=True, help="Path to configs/dataset_satloc.yaml")
    parser.add_argument("--uav-traj-csv", default=DEFAULT_UAV_TRAJ)
    parser.add_argument("--tile-index-csv", default=DEFAULT_TILE_INDEX)

    parser.add_argument("--frames", default=None, help="Comma-separated traj01 frame indices.")
    parser.add_argument("--ranges", default=None, help="Example: 1-100:10")
    parser.add_argument("--max-frames", type=int, default=30)

    parser.add_argument("--max-dim", type=int, default=900)
    parser.add_argument("--nfeatures", type=int, default=1200)
    parser.add_argument("--ratio", type=float, default=0.75)
    parser.add_argument("--ransac-thresh", type=float, default=5.0)

    parser.add_argument("--clahe-clip-limit", type=float, default=2.0)
    parser.add_argument("--clahe-tile-size", type=int, default=8)

    parser.add_argument("--alt-clahe-clip-limit", type=float, default=1.0)
    parser.add_argument("--alt-clahe-tile-size", type=int, default=8)

    parser.add_argument("--bilateral-d", type=int, default=13)
    parser.add_argument("--bilateral-sigma-color", type=float, default=30.0)
    parser.add_argument("--bilateral-sigma-space", type=float, default=55.0)

    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = Path(config["paths"]["output_dir"])

    uav_df = pd.read_csv(args.uav_traj_csv)
    tile_df = pd.read_csv(args.tile_index_csv)

    selected_df = select_uav_rows(
        uav_df=uav_df,
        frames=parse_int_list(args.frames),
        ranges=parse_ranges(args.ranges),
    )

    if len(selected_df) > args.max_frames:
        print(f"Selected {len(selected_df)} frames, limiting to {args.max_frames}")
        selected_df = selected_df.head(args.max_frames).copy()

    metadata_dir = ensure_dir(output_dir / "metadata" / STAGE_NAME)
    reports_dir = ensure_dir(output_dir / "reports" / STAGE_NAME)
    figures_dir = ensure_dir(output_dir / "figures" / STAGE_NAME / "true_tile_orb_matching")

    results_path = metadata_dir / "true_tile_orb_matching_results.csv"
    summary_path = reports_dir / "true_tile_orb_matching_summary.csv"
    manifest_path = metadata_dir / "true_tile_orb_matching_manifest.csv"

    all_results = []
    manifest_rows = []

    print("S3.5D.3 true-tile ORB matching")
    print("--------------------------------")
    print(f"Selected frames: {len(selected_df)}")
    print(f"ORB nfeatures:   {args.nfeatures}")
    print(f"Ratio threshold: {args.ratio}")
    print(f"RANSAC thresh:   {args.ransac_thresh}")

    for _, frame_row in selected_df.iterrows():
        frame_idx = int(frame_row["frame_index_in_sequence"])

        tile_row, containing_count, found_containing, center_error_m = find_true_tile(frame_row, tile_df)

        uav_bgr = cv2.imread(str(frame_row["image_path"]), cv2.IMREAD_COLOR)
        tile_bgr = cv2.imread(str(tile_row["tile_path"]), cv2.IMREAD_COLOR)

        if uav_bgr is None:
            print(f"Skipping frame {frame_idx}: could not read UAV image")
            continue

        if tile_bgr is None:
            print(f"Skipping frame {frame_idx}: could not read tile image")
            continue

        uav_bgr = resize_to_max_dim(uav_bgr, max_dim=args.max_dim)
        tile_bgr = resize_to_max_dim(tile_bgr, max_dim=args.max_dim)

        uav_rgb = cv2.cvtColor(uav_bgr, cv2.COLOR_BGR2RGB)
        tile_rgb = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2RGB)

        uav_variants = build_variants(
            rgb=uav_rgb,
            clahe_clip_limit=args.clahe_clip_limit,
            clahe_tile_size=args.clahe_tile_size,
            alt_clahe_clip_limit=args.alt_clahe_clip_limit,
            alt_clahe_tile_size=args.alt_clahe_tile_size,
            bilateral_d=args.bilateral_d,
            bilateral_sigma_color=args.bilateral_sigma_color,
            bilateral_sigma_space=args.bilateral_sigma_space,
        )

        tile_variants = build_variants(
            rgb=tile_rgb,
            clahe_clip_limit=args.clahe_clip_limit,
            clahe_tile_size=args.clahe_tile_size,
            alt_clahe_clip_limit=args.alt_clahe_clip_limit,
            alt_clahe_tile_size=args.alt_clahe_tile_size,
            bilateral_d=args.bilateral_d,
            bilateral_sigma_color=args.bilateral_sigma_color,
            bilateral_sigma_space=args.bilateral_sigma_space,
        )

        frame_results = []
        match_images = {}

        for variant_name in ["V1_luma", "V2_clahe_luma", "V3_bilateral_clahe", "V4_bilateral_alt_clahe"]:
            result, match_img = run_variant_match(
                variant_name=variant_name,
                uav_img_u8=uav_variants[variant_name],
                tile_img_u8=tile_variants[variant_name],
                uav_rgb=uav_rgb,
                tile_rgb=tile_rgb,
                nfeatures=args.nfeatures,
                ratio=args.ratio,
                ransac_thresh=args.ransac_thresh,
            )

            result.update(
                {
                    "frame_index_in_sequence": frame_idx,
                    "uav_filename": frame_row["filename"],
                    "uav_image_path": frame_row["image_path"],
                    "gt_lon": float(frame_row["lon"]),
                    "gt_lat": float(frame_row["lat"]),
                    "true_tile_filename": tile_row["filename"],
                    "true_tile_path": tile_row["tile_path"],
                    "true_tile_contains_gt": bool(found_containing),
                    "num_tiles_containing_gt": int(containing_count),
                    "true_tile_center_error_m": float(center_error_m),
                    "tile_lon_center": float(tile_row["lon_center"]),
                    "tile_lat_center": float(tile_row["lat_center"]),
                }
            )

            frame_results.append(result)
            match_images[variant_name] = match_img
            all_results.append(result)

        panel_path = figures_dir / f"traj01_frame_{frame_idx:04d}_true_tile_orb_matches.png"

        save_frame_panel(
            frame_row=frame_row,
            tile_row=tile_row,
            match_images=match_images,
            variant_results=frame_results,
            output_path=panel_path,
        )

        manifest_rows.append(
            {
                "frame_index_in_sequence": frame_idx,
                "uav_filename": frame_row["filename"],
                "true_tile_filename": tile_row["filename"],
                "true_tile_contains_gt": bool(found_containing),
                "num_tiles_containing_gt": int(containing_count),
                "true_tile_center_error_m": float(center_error_m),
                "panel_path": str(panel_path),
            }
        )

        print(
            f"Frame {frame_idx}: tile={tile_row['filename']}, "
            f"contains={found_containing}, containing_tiles={containing_count}, "
            f"center_error={center_error_m:.2f} m, panel={panel_path}"
        )

    results_df = pd.DataFrame(all_results)
    manifest_df = pd.DataFrame(manifest_rows)

    results_df.to_csv(results_path, index=False)
    manifest_df.to_csv(manifest_path, index=False)

    summary_df = (
        results_df.groupby("variant")
        .agg(
            frames=("frame_index_in_sequence", "count"),
            good_matches_mean=("good_matches", "mean"),
            good_matches_median=("good_matches", "median"),
            ransac_inliers_mean=("ransac_inliers", "mean"),
            ransac_inliers_median=("ransac_inliers", "median"),
            inlier_ratio_mean=("inlier_ratio", "mean"),
            homography_success_rate=("homography_success", "mean"),
            score_mean=("score", "mean"),
            elapsed_ms_mean=("elapsed_ms", "mean"),
        )
        .reset_index()
    )

    summary_df.to_csv(summary_path, index=False)

    print()
    print("S3.5D.3 complete")
    print("----------------")
    print(f"Saved results:   {results_path}")
    print(f"Saved summary:   {summary_path}")
    print(f"Saved manifest:  {manifest_path}")
    print(f"Saved figures:   {figures_dir}")
    print()
    print("Summary")
    print("-------")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()