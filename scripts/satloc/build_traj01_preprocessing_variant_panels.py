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
    """
    Format:
      1-150:8,250-350:8,400-500:8
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


def normalize_uint8(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    mn = float(np.nanmin(arr))
    mx = float(np.nanmax(arr))

    if mx - mn < 1e-9:
        return np.zeros(arr.shape, dtype=np.uint8)

    out = (arr - mn) / (mx - mn)
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)


def make_luma(rgb: np.ndarray) -> np.ndarray:
    r = rgb[:, :, 0].astype(np.float32)
    g = rgb[:, :, 1].astype(np.float32)
    b = rgb[:, :, 2].astype(np.float32)
    return np.clip(0.299 * r + 0.587 * g + 0.114 * b, 0, 255).astype(np.uint8)


def make_clahe(gray: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    return clahe.apply(gray)


def make_bilateral_then_clahe(luma: np.ndarray) -> np.ndarray:
    smoothed = cv2.bilateralFilter(
        luma,
        d=9,
        sigmaColor=55,
        sigmaSpace=55,
    )
    return make_clahe(smoothed)


def sobel_mag(gray: np.ndarray) -> np.ndarray:
    gray_f = gray.astype(np.float32) / 255.0
    gx = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    return normalize_uint8(mag)


def build_structure_mask_from_sobel(
    sobel_u8: np.ndarray,
    percentile: float = 82.0,
    dilate_iter: int = 2,
) -> np.ndarray:
    """
    Structure candidate mask.

    This is not a semantic road/building mask.
    It says: keep areas near strong structural gradients.
    """
    threshold = np.percentile(sobel_u8, percentile)
    binary = ((sobel_u8 >= threshold) * 255).astype(np.uint8)

    kernel = np.ones((5, 5), dtype=np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    if dilate_iter > 0:
        binary = cv2.dilate(binary, kernel, iterations=dilate_iter)

    return binary


def make_green_texture_mask(rgb: np.ndarray, sobel_u8: np.ndarray) -> np.ndarray:
    """
    Diagnostic only.

    Marks green-ish, high-frequency regions that may create many repetitive keypoints.
    We do not use this as the ORB mask yet.
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


def draw_keypoints(rgb: np.ndarray, keypoints: list[cv2.KeyPoint]) -> np.ndarray:
    out_bgr = cv2.drawKeypoints(
        cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
        keypoints,
        None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )
    return cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)


def compute_orb_keypoints(
    image_u8: np.ndarray,
    mask_u8: np.ndarray | None = None,
    nfeatures: int = 2500,
) -> tuple[list[cv2.KeyPoint], np.ndarray | None]:
    orb = cv2.ORB_create(nfeatures=nfeatures)
    keypoints, descriptors = orb.detectAndCompute(image_u8, mask_u8)
    return keypoints if keypoints is not None else [], descriptors


def keypoint_grid_coverage(
    keypoints: list[cv2.KeyPoint],
    width: int,
    height: int,
    grid_rows: int = 6,
    grid_cols: int = 8,
) -> tuple[float, float]:
    """
    Returns:
      grid_occupancy_ratio: occupied grid cells / total grid cells
      concentration_score: max cell count / total keypoints
    """
    if not keypoints:
        return 0.0, 0.0

    counts = np.zeros((grid_rows, grid_cols), dtype=np.int32)

    for kp in keypoints:
        x, y = kp.pt
        col = min(grid_cols - 1, max(0, int(x / max(width, 1) * grid_cols)))
        row = min(grid_rows - 1, max(0, int(y / max(height, 1) * grid_rows)))
        counts[row, col] += 1

    occupied = int((counts > 0).sum())
    total_cells = grid_rows * grid_cols
    total_kps = len(keypoints)

    occupancy = occupied / total_cells
    concentration = counts.max() / max(total_kps, 1)

    return float(occupancy), float(concentration)


def keypoint_mask_ratio(
    keypoints: list[cv2.KeyPoint],
    mask_u8: np.ndarray,
) -> float:
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


def build_preprocessed_variants(rgb: np.ndarray) -> Dict[str, Dict[str, np.ndarray | None]]:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    luma = make_luma(rgb)
    clahe_luma = make_clahe(luma)
    bilateral_clahe = make_bilateral_then_clahe(luma)

    sobel_luma = sobel_mag(luma)
    sobel_clahe = sobel_mag(clahe_luma)

    structure_mask = build_structure_mask_from_sobel(sobel_clahe)
    green_texture_mask = make_green_texture_mask(rgb, sobel_luma)

    return {
        "V0_raw_gray_orb": {
            "image": gray,
            "mask": None,
        },
        "V1_luma_orb": {
            "image": luma,
            "mask": None,
        },
        "V2_clahe_luma_orb": {
            "image": clahe_luma,
            "mask": None,
        },
        "V3_bilateral_clahe_orb": {
            "image": bilateral_clahe,
            "mask": None,
        },
        "V4_structure_masked_clahe_orb": {
            "image": clahe_luma,
            "mask": structure_mask,
        },
        "V5_structure_masked_bilateral_clahe_orb": {
            "image": bilateral_clahe,
            "mask": structure_mask,
        },
        "_diagnostic_gray": {
            "image": gray,
            "mask": None,
        },
        "_diagnostic_luma": {
            "image": luma,
            "mask": None,
        },
        "_diagnostic_clahe_luma": {
            "image": clahe_luma,
            "mask": None,
        },
        "_diagnostic_bilateral_clahe": {
            "image": bilateral_clahe,
            "mask": None,
        },
        "_diagnostic_sobel_luma": {
            "image": sobel_luma,
            "mask": None,
        },
        "_diagnostic_sobel_clahe": {
            "image": sobel_clahe,
            "mask": None,
        },
        "_diagnostic_structure_mask": {
            "image": structure_mask,
            "mask": None,
        },
        "_diagnostic_green_texture_mask": {
            "image": green_texture_mask,
            "mask": None,
        },
    }


def compute_variant_stats(
    frame_row: pd.Series,
    rgb: np.ndarray,
    variants: Dict[str, Dict[str, np.ndarray | None]],
) -> tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    h, w = rgb.shape[:2]

    structure_mask = variants["_diagnostic_structure_mask"]["image"]
    green_texture_mask = variants["_diagnostic_green_texture_mask"]["image"]

    overlay_images: Dict[str, np.ndarray] = {}
    records = []

    for variant_name, payload in variants.items():
        if variant_name.startswith("_diagnostic"):
            continue

        image_u8 = payload["image"]
        mask_u8 = payload["mask"]

        assert isinstance(image_u8, np.ndarray)

        keypoints, descriptors = compute_orb_keypoints(image_u8, mask_u8=mask_u8)
        overlay_images[variant_name] = draw_keypoints(rgb, keypoints)

        grid_occupancy, concentration = keypoint_grid_coverage(keypoints, w, h)
        mean_response, median_response = keypoint_response_stats(keypoints)

        assert isinstance(structure_mask, np.ndarray)
        assert isinstance(green_texture_mask, np.ndarray)

        records.append(
            {
                "frame_index_in_sequence": int(frame_row["frame_index_in_sequence"]),
                "token0_id": int(frame_row["token0_id"]),
                "token1_order": int(frame_row["token1_order"]),
                "filename": frame_row["filename"],
                "variant_name": variant_name,
                "keypoint_count": int(len(keypoints)),
                "descriptor_count": int(0 if descriptors is None else len(descriptors)),
                "grid_occupancy_ratio": grid_occupancy,
                "keypoint_concentration_score": concentration,
                "mean_response": mean_response,
                "median_response": median_response,
                "keypoints_on_structure_ratio": keypoint_mask_ratio(keypoints, structure_mask),
                "keypoints_on_green_texture_ratio": keypoint_mask_ratio(keypoints, green_texture_mask),
                "mask_used": mask_u8 is not None,
            }
        )

    return pd.DataFrame(records), overlay_images


def build_panel(
    row: pd.Series,
    output_path: Path,
    max_dim: int,
) -> pd.DataFrame:
    image_path = Path(row["image_path"])

    img_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise RuntimeError(f"Could not read image: {image_path}")

    img_bgr = resize_to_max_dim(img_bgr, max_dim=max_dim)
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    variants = build_preprocessed_variants(rgb)
    variant_stats_df, overlay_images = compute_variant_stats(row, rgb, variants)

    gray = variants["_diagnostic_gray"]["image"]
    luma = variants["_diagnostic_luma"]["image"]
    clahe_luma = variants["_diagnostic_clahe_luma"]["image"]
    bilateral_clahe = variants["_diagnostic_bilateral_clahe"]["image"]
    sobel_luma = variants["_diagnostic_sobel_luma"]["image"]
    sobel_clahe = variants["_diagnostic_sobel_clahe"]["image"]
    structure_mask = variants["_diagnostic_structure_mask"]["image"]

    panels = [
        ("RGB original", rgb, None),
        ("Luma", luma, "gray"),
        ("CLAHE luma", clahe_luma, "gray"),
        ("Bilateral + CLAHE", bilateral_clahe, "gray"),

        ("Sobel on luma", sobel_luma, "gray"),
        ("Sobel on CLAHE", sobel_clahe, "gray"),
        ("Structure mask from Sobel-CLAHE", structure_mask, "gray"),
        ("V0 ORB raw gray", overlay_images["V0_raw_gray_orb"], None),

        ("V1 ORB luma", overlay_images["V1_luma_orb"], None),
        ("V2 ORB CLAHE luma", overlay_images["V2_clahe_luma_orb"], None),
        ("V3 ORB bilateral+CLAHE", overlay_images["V3_bilateral_clahe_orb"], None),
        ("V4 ORB masked CLAHE", overlay_images["V4_structure_masked_clahe_orb"], None),

        ("V5 ORB masked bilateral+CLAHE", overlay_images["V5_structure_masked_bilateral_clahe_orb"], None),
    ]

    ensure_dir(output_path.parent)

    fig, axes = plt.subplots(4, 4, figsize=(18, 15))
    axes_flat = axes.flatten()

    for ax, (title, img, cmap) in zip(axes_flat[:len(panels)], panels):
        assert isinstance(img, np.ndarray)
        ax.imshow(img, cmap=cmap)
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    for ax in axes_flat[len(panels):]:
        ax.axis("off")

    stats_lines = [
        f"frame_index: {int(row['frame_index_in_sequence'])}",
        f"filename: {row['filename']}",
        "",
        "variant keypoints / coverage:",
    ]

    for _, srow in variant_stats_df.iterrows():
        stats_lines.append(
            f"{srow['variant_name']}: "
            f"kps={int(srow['keypoint_count'])}, "
            f"grid={srow['grid_occupancy_ratio']:.2f}, "
            f"green={srow['keypoints_on_green_texture_ratio']:.2f}"
        )

    axes_flat[15].text(
        0.0,
        1.0,
        "\n".join(stats_lines),
        va="top",
        ha="left",
        fontsize=8,
        family="monospace",
    )

    fig.suptitle(
        f"S3.5D preprocessing variants — traj01 frame {int(row['frame_index_in_sequence'])}",
        fontsize=13,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=170)
    plt.close(fig)

    return variant_stats_df


def build_selection(stats_df: pd.DataFrame, frames: List[int], ranges: List[Tuple[int, int, int]]) -> pd.DataFrame:
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
        description="Build S3.5D preprocessing variant panels and ORB keypoint distribution stats."
    )
    parser.add_argument("--config", required=True, help="Path to configs/dataset_satloc.yaml")
    parser.add_argument("--stats-csv", default=DEFAULT_STATS_CSV)
    parser.add_argument("--frames", default=None, help="Comma-separated frame indices.")
    parser.add_argument("--ranges", default=None, help="Example: 1-150:6,250-350:6,400-500:6")
    parser.add_argument("--max-dim", type=int, default=900)
    parser.add_argument("--max-panels", type=int, default=60)
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

    panel_dir = ensure_dir(output_dir / "figures" / STAGE_NAME / "preprocessing_variant_panels")
    metadata_dir = ensure_dir(output_dir / "metadata" / STAGE_NAME)

    manifest_path = metadata_dir / "preprocessing_variant_panel_manifest.csv"
    stats_out_path = metadata_dir / "preprocessing_variant_keypoint_stats.csv"

    all_variant_stats = []
    panel_paths = []

    print("S3.5D preprocessing variant panels")
    print("----------------------------------")
    print(f"Selected frames: {len(selected_df)}")

    for _, row in selected_df.iterrows():
        frame_idx = int(row["frame_index_in_sequence"])
        output_path = panel_dir / f"traj01_frame_{frame_idx:04d}_preprocessing_variants.png"

        variant_stats_df = build_panel(row, output_path, max_dim=args.max_dim)
        variant_stats_df["panel_path"] = str(output_path)

        all_variant_stats.append(variant_stats_df)
        panel_paths.append(str(output_path))

        print(f"Saved frame {frame_idx}: {output_path}")

    selected_df = selected_df.copy()
    selected_df["panel_path"] = panel_paths
    selected_df.to_csv(manifest_path, index=False)

    if all_variant_stats:
        pd.concat(all_variant_stats, ignore_index=True).to_csv(stats_out_path, index=False)

    print()
    print("S3.5D complete")
    print("--------------")
    print(f"Saved panels: {panel_dir}")
    print(f"Saved manifest: {manifest_path}")
    print(f"Saved variant stats: {stats_out_path}")


if __name__ == "__main__":
    main()