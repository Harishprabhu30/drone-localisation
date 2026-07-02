from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


STAGE_NAME = "s3_5_visual_domain_traj01"
DEFAULT_STATS_CSV = "outputs/satloc/metadata/s3_5_visual_domain_traj01/traj01_image_stats.csv"


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


def resize_to_max_dim(img: np.ndarray, max_dim: int = 1000) -> np.ndarray:
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


def sobel_mag(gray: np.ndarray) -> np.ndarray:
    gray_f = gray.astype(np.float32) / 255.0

    gx = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3)

    mag = np.sqrt(gx * gx + gy * gy)
    max_v = float(mag.max())
    if max_v <= 1e-9:
        return np.zeros_like(gray)

    return np.clip((mag / max_v) * 255, 0, 255).astype(np.uint8)


def make_green_texture_mask(rgb: np.ndarray, sobel_u8: np.ndarray) -> np.ndarray:
    """
    Diagnostic only.
    Finds green-ish + high-gradient texture regions.
    This helps estimate whether keypoints concentrate in repetitive vegetation.
    """
    r = rgb[:, :, 0].astype(np.float32)
    g = rgb[:, :, 1].astype(np.float32)
    b = rgb[:, :, 2].astype(np.float32)

    rgb_sum = r + g + b + 1e-6
    green_ratio = g / rgb_sum
    excess_green = 2.0 * g - r - b

    greenish = (green_ratio > np.percentile(green_ratio, 70)) & (
        excess_green > np.percentile(excess_green, 65)
    )
    high_texture = sobel_u8 > np.percentile(sobel_u8, 70)

    mask = (greenish & high_texture).astype(np.uint8) * 255

    kernel = np.ones((5, 5), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    return mask


def compute_orb(
    image_u8: np.ndarray,
    nfeatures: int,
) -> tuple[list[cv2.KeyPoint], np.ndarray | None]:
    orb = cv2.ORB_create(nfeatures=nfeatures)
    kps, desc = orb.detectAndCompute(image_u8, None)
    return kps if kps is not None else [], desc


def draw_keypoints(rgb: np.ndarray, keypoints: list[cv2.KeyPoint]) -> np.ndarray:
    out_bgr = cv2.drawKeypoints(
        cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
        keypoints,
        None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )
    return cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)


def keypoint_density_map(
    keypoints: list[cv2.KeyPoint],
    height: int,
    width: int,
    blur_ksize: int = 31,
) -> np.ndarray:
    density = np.zeros((height, width), dtype=np.float32)

    for kp in keypoints:
        x, y = kp.pt
        ix = min(width - 1, max(0, int(round(x))))
        iy = min(height - 1, max(0, int(round(y))))
        density[iy, ix] += 1.0

    if blur_ksize % 2 == 0:
        blur_ksize += 1

    density = cv2.GaussianBlur(density, (blur_ksize, blur_ksize), 0)

    max_v = float(density.max())
    if max_v > 1e-9:
        density = density / max_v

    return density


def signed_density_difference(
    base_kps: list[cv2.KeyPoint],
    candidate_kps: list[cv2.KeyPoint],
    height: int,
    width: int,
) -> np.ndarray:
    base_density = keypoint_density_map(base_kps, height, width)
    cand_density = keypoint_density_map(candidate_kps, height, width)

    diff = cand_density - base_density
    return diff


def plot_signed_diff(ax, diff: np.ndarray, title: str) -> None:
    vmax = max(abs(float(diff.min())), abs(float(diff.max())), 1e-6)
    ax.imshow(diff, cmap="coolwarm", vmin=-vmax, vmax=vmax)
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def grid_stats(
    keypoints: list[cv2.KeyPoint],
    width: int,
    height: int,
    rows: int = 6,
    cols: int = 8,
) -> tuple[float, float]:
    if not keypoints:
        return 0.0, 0.0

    grid = np.zeros((rows, cols), dtype=np.int32)

    for kp in keypoints:
        x, y = kp.pt
        c = min(cols - 1, max(0, int(x / max(width, 1) * cols)))
        r = min(rows - 1, max(0, int(y / max(height, 1) * rows)))
        grid[r, c] += 1

    occupancy = float((grid > 0).sum() / (rows * cols))
    concentration = float(grid.max() / max(len(keypoints), 1))

    return occupancy, concentration


def keypoint_mask_ratio(keypoints: list[cv2.KeyPoint], mask_u8: np.ndarray) -> float:
    if not keypoints:
        return 0.0

    h, w = mask_u8.shape[:2]
    inside = 0

    for kp in keypoints:
        x, y = kp.pt
        ix = min(w - 1, max(0, int(round(x))))
        iy = min(h - 1, max(0, int(round(y))))

        if mask_u8[iy, ix] > 0:
            inside += 1

    return float(inside / len(keypoints))


def keypoint_response_stats(keypoints: list[cv2.KeyPoint]) -> tuple[float, float]:
    if not keypoints:
        return 0.0, 0.0

    responses = np.array([kp.response for kp in keypoints], dtype=np.float32)
    return float(responses.mean()), float(np.median(responses))

# old
# def overlap_ratio(
#     base_kps: list[cv2.KeyPoint],
#     candidate_kps: list[cv2.KeyPoint],
#     radius_px: float = 5.0,
# ) -> float:
#     """
#     Fraction of candidate keypoints that are near at least one baseline keypoint.
#     Low value means the candidate preprocessing moves ORB to new locations.
#     """
#     if not base_kps or not candidate_kps:
#         return 0.0

#     base_pts = np.array([kp.pt for kp in base_kps], dtype=np.float32)
#     cand_pts = np.array([kp.pt for kp in candidate_kps], dtype=np.float32)

#     matched = 0

#     for pt in cand_pts:
#         d = np.sqrt(((base_pts - pt) ** 2).sum(axis=1))
#         if float(d.min()) <= radius_px:
#             matched += 1

#     return float(matched / len(candidate_kps))

def overlap_ratio(
    base_kps: list[cv2.KeyPoint],
    candidate_kps: list[cv2.KeyPoint],
    radius_px: float = 5.0,
) -> float:
    if not base_kps or not candidate_kps:
        return 0.0

    base_pts = np.array([kp.pt for kp in base_kps], dtype=np.float32)
    cand_pts = np.array([kp.pt for kp in candidate_kps], dtype=np.float32)

    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(base_pts)
        distances, _ = tree.query(cand_pts, k=1)
        return float((distances <= radius_px).mean())

    except Exception:
        matched = 0
        chunk_size = 512

        for start in range(0, len(cand_pts), chunk_size):
            chunk = cand_pts[start:start + chunk_size]
            diff = chunk[:, None, :] - base_pts[None, :, :]
            dist2 = np.sum(diff * diff, axis=2)
            min_dist = np.sqrt(np.min(dist2, axis=1))
            matched += int((min_dist <= radius_px).sum())

        return float(matched / len(candidate_kps))

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
        "V0_gray": gray,
        "V1_luma": luma,
        "V2_clahe_luma": clahe_luma,
        "V3_bilateral_clahe": bilateral_clahe,
        "V4_bilateral_alt_clahe": bilateral_alt_clahe,
    }


def build_panel_and_stats(
    row: pd.Series,
    output_path: Path,
    max_dim: int,
    nfeatures: int,
    clahe_clip_limit: float,
    clahe_tile_size: int,
    alt_clahe_clip_limit: float,
    alt_clahe_tile_size: int,
    bilateral_d: int,
    bilateral_sigma_color: float,
    bilateral_sigma_space: float,
) -> pd.DataFrame:
    image_path = Path(row["image_path"])

    img_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise RuntimeError(f"Could not read image: {image_path}")

    img_bgr = resize_to_max_dim(img_bgr, max_dim=max_dim)
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    h, w = rgb.shape[:2]

    variants = build_variants(
        rgb=rgb,
        clahe_clip_limit=clahe_clip_limit,
        clahe_tile_size=clahe_tile_size,
        alt_clahe_clip_limit=alt_clahe_clip_limit,
        alt_clahe_tile_size=alt_clahe_tile_size,
        bilateral_d=bilateral_d,
        bilateral_sigma_color=bilateral_sigma_color,
        bilateral_sigma_space=bilateral_sigma_space,
    )

    green_texture_mask = make_green_texture_mask(rgb, sobel_mag(variants["V1_luma"]))

    variant_kps = {}
    overlays = {}
    records = []

    base_gray_kps = None
    base_luma_kps = None

    for name, image_u8 in variants.items():
        kps, desc = compute_orb(image_u8, nfeatures=nfeatures)
        variant_kps[name] = kps
        overlays[name] = draw_keypoints(rgb, kps)

        if name == "V0_gray":
            base_gray_kps = kps
        if name == "V1_luma":
            base_luma_kps = kps

    assert base_gray_kps is not None
    assert base_luma_kps is not None

    for name, kps in variant_kps.items():
        grid_occ, concentration = grid_stats(kps, width=w, height=h)
        response_mean, response_median = keypoint_response_stats(kps)

        records.append(
            {
                "frame_index_in_sequence": int(row["frame_index_in_sequence"]),
                "token0_id": int(row["token0_id"]),
                "token1_order": int(row["token1_order"]),
                "filename": row["filename"],
                "variant": name,
                "nfeatures": int(nfeatures),
                "keypoint_count": int(len(kps)),
                "grid_occupancy_ratio": grid_occ,
                "concentration_score": concentration,
                "mean_response": response_mean,
                "median_response": response_median,
                "green_texture_keypoint_ratio": keypoint_mask_ratio(kps, green_texture_mask),
                "overlap_with_gray_ratio": overlap_ratio(base_gray_kps, kps),
                "overlap_with_luma_ratio": overlap_ratio(base_luma_kps, kps),
            }
        )

    stats_df = pd.DataFrame(records)

    diff_gray_vs_candidate = signed_density_difference(
        base_gray_kps,
        variant_kps["V4_bilateral_alt_clahe"],
        h,
        w,
    )

    diff_luma_vs_candidate = signed_density_difference(
        base_luma_kps,
        variant_kps["V4_bilateral_alt_clahe"],
        h,
        w,
    )

    diff_clahe_vs_candidate = signed_density_difference(
        variant_kps["V2_clahe_luma"],
        variant_kps["V4_bilateral_alt_clahe"],
        h,
        w,
    )

    panels = [
        ("RGB original", rgb, None, "image"),
        ("ORB on raw gray", overlays["V0_gray"], None, "image"),
        ("ORB on luma", overlays["V1_luma"], None, "image"),

        ("ORB on CLAHE-luma", overlays["V2_clahe_luma"], None, "image"),
        ("ORB on bilateral(CLAHE)", overlays["V3_bilateral_clahe"], None, "image"),
        ("ORB on bilateral(alt-CLAHE)", overlays["V4_bilateral_alt_clahe"], None, "image"),

        ("Density diff: candidate - gray", diff_gray_vs_candidate, None, "diff"),
        ("Density diff: candidate - luma", diff_luma_vs_candidate, None, "diff"),
        ("Density diff: candidate - CLAHE", diff_clahe_vs_candidate, None, "diff"),
    ]

    ensure_dir(output_path.parent)

    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    axes_flat = axes.flatten()

    for ax, (title, img, cmap, kind) in zip(axes_flat, panels):
        if kind == "diff":
            plot_signed_diff(ax, img, title)
        else:
            ax.imshow(img, cmap=cmap)
            ax.set_title(title, fontsize=9)
            ax.axis("off")

    fig.suptitle(
        f"S3.5D.2 ORB preprocessing comparison — traj01 frame {int(row['frame_index_in_sequence'])}\n"
        f"candidate = bilateral(alt-CLAHE), nfeatures={nfeatures}",
        fontsize=12,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=170)
    plt.close(fig)

    return stats_df


def build_selection(
    stats_df: pd.DataFrame,
    frames: List[int],
    ranges: List[Tuple[int, int, int]],
) -> pd.DataFrame:
    selected_frames = set(frames)

    for start, end, count in ranges:
        selected_frames.update(sample_evenly(start, end, count))

    if not selected_frames:
        raise ValueError("No frames selected. Use --frames or --ranges.")

    selected = (
        stats_df[stats_df["frame_index_in_sequence"].astype(int).isin(sorted(selected_frames))]
        .copy()
        .sort_values("frame_index_in_sequence")
        .reset_index(drop=True)
    )

    return selected


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare ORB keypoint behavior across preprocessing variants."
    )
    parser.add_argument("--config", required=True, help="Path to configs/dataset_satloc.yaml")
    parser.add_argument("--stats-csv", default=DEFAULT_STATS_CSV)

    parser.add_argument("--frames", default=None, help="Comma-separated frame indices.")
    parser.add_argument("--ranges", default=None, help="Example: 1-100:20")
    parser.add_argument("--max-dim", type=int, default=1000)
    parser.add_argument("--max-panels", type=int, default=80)

    parser.add_argument("--nfeatures", type=int, default=1200)

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

    stats_df = pd.read_csv(args.stats_csv)
    stats_df["frame_index_in_sequence"] = stats_df["frame_index_in_sequence"].astype(int)

    selected_df = build_selection(
        stats_df=stats_df,
        frames=parse_int_list(args.frames),
        ranges=parse_ranges(args.ranges),
    )

    if len(selected_df) > args.max_panels:
        print(
            f"Selected {len(selected_df)} frames but max-panels={args.max_panels}. "
            f"Keeping first {args.max_panels}."
        )
        selected_df = selected_df.head(args.max_panels).copy()

    panel_dir = ensure_dir(output_dir / "figures" / STAGE_NAME / "orb_preprocessing_comparison")
    metadata_dir = ensure_dir(output_dir / "metadata" / STAGE_NAME)

    manifest_path = metadata_dir / "orb_preprocessing_comparison_manifest.csv"
    stats_out_path = metadata_dir / "orb_preprocessing_comparison_stats.csv"
    summary_out_path = metadata_dir / "orb_preprocessing_comparison_summary.csv"

    all_stats = []
    panel_paths = []

    print("S3.5D.2 ORB preprocessing comparison")
    print("------------------------------------")
    print(f"Selected frames: {len(selected_df)}")
    print(f"ORB nfeatures:   {args.nfeatures}")

    for _, row in selected_df.iterrows():
        frame_idx = int(row["frame_index_in_sequence"])
        output_path = panel_dir / f"traj01_frame_{frame_idx:04d}_orb_preprocessing_comparison.png"

        frame_stats = build_panel_and_stats(
            row=row,
            output_path=output_path,
            max_dim=args.max_dim,
            nfeatures=args.nfeatures,
            clahe_clip_limit=args.clahe_clip_limit,
            clahe_tile_size=args.clahe_tile_size,
            alt_clahe_clip_limit=args.alt_clahe_clip_limit,
            alt_clahe_tile_size=args.alt_clahe_tile_size,
            bilateral_d=args.bilateral_d,
            bilateral_sigma_color=args.bilateral_sigma_color,
            bilateral_sigma_space=args.bilateral_sigma_space,
        )

        frame_stats["panel_path"] = str(output_path)
        all_stats.append(frame_stats)
        panel_paths.append(str(output_path))

        print(f"Saved frame {frame_idx}: {output_path}")

    selected_df = selected_df.copy()
    selected_df["panel_path"] = panel_paths
    selected_df.to_csv(manifest_path, index=False)

    stats_all_df = pd.concat(all_stats, ignore_index=True)
    stats_all_df.to_csv(stats_out_path, index=False)

    summary_df = (
        stats_all_df.groupby("variant")
        .agg(
            frames=("frame_index_in_sequence", "count"),
            keypoint_count_mean=("keypoint_count", "mean"),
            keypoint_count_median=("keypoint_count", "median"),
            grid_occupancy_mean=("grid_occupancy_ratio", "mean"),
            concentration_mean=("concentration_score", "mean"),
            green_texture_ratio_mean=("green_texture_keypoint_ratio", "mean"),
            mean_response_mean=("mean_response", "mean"),
            overlap_with_gray_mean=("overlap_with_gray_ratio", "mean"),
            overlap_with_luma_mean=("overlap_with_luma_ratio", "mean"),
        )
        .reset_index()
    )

    summary_df.to_csv(summary_out_path, index=False)

    print()
    print("S3.5D.2 complete")
    print("----------------")
    print(f"Saved panels: {panel_dir}")
    print(f"Saved manifest: {manifest_path}")
    print(f"Saved stats CSV: {stats_out_path}")
    print(f"Saved summary CSV: {summary_out_path}")
    print()
    print("Summary")
    print("-------")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()