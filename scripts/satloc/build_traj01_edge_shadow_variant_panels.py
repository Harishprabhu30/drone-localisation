from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

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


def make_gray(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def make_clahe(gray: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def sobel_mag(gray: np.ndarray) -> np.ndarray:
    gray_f = gray.astype(np.float32) / 255.0
    gx = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    return normalize_uint8(mag)


def adaptive_sobel_binary(sobel_u8: np.ndarray, percentile: float = 85.0) -> np.ndarray:
    threshold = np.percentile(sobel_u8, percentile)
    return ((sobel_u8 >= threshold) * 255).astype(np.uint8)


def make_shadow_candidate_mask(rgb: np.ndarray) -> np.ndarray:
    """
    Simple diagnostic shadow heuristic.

    Shadows often have:
    - low value / brightness
    - reduced luma
    - not too much saturation
    This is not a final shadow detector.
    """
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    luma = make_luma(rgb)

    low_value = v < np.percentile(v, 38)
    low_luma = luma < np.percentile(luma, 40)
    not_high_saturation = s < np.percentile(s, 75)

    mask = low_value & low_luma & not_high_saturation

    mask_u8 = (mask.astype(np.uint8) * 255)

    kernel = np.ones((5, 5), dtype=np.uint8)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)

    return mask_u8


def overlay_mask(rgb: np.ndarray, mask_u8: np.ndarray) -> np.ndarray:
    out = rgb.copy()
    overlay = out.copy()

    # red overlay for shadow candidates
    overlay[mask_u8 > 0] = [255, 0, 0]

    blended = cv2.addWeighted(out, 0.7, overlay, 0.3, 0)
    return blended


def suppress_shadow_edges(edge_or_grad: np.ndarray, shadow_mask: np.ndarray) -> np.ndarray:
    out = edge_or_grad.copy()
    out[shadow_mask > 0] = 0
    return out


def draw_hough_lines(rgb: np.ndarray, binary_edges: np.ndarray) -> np.ndarray:
    out = rgb.copy()

    lines = cv2.HoughLinesP(
        binary_edges,
        rho=1,
        theta=np.pi / 180,
        threshold=60,
        minLineLength=40,
        maxLineGap=8,
    )

    if lines is None:
        return out

    for line in lines[:500]:
        x1, y1, x2, y2 = line[0]
        cv2.line(out, (x1, y1), (x2, y2), (255, 0, 0), 1)

    return out


def build_variant_panel(row: pd.Series, output_path: Path, max_dim: int = 900) -> None:
    image_path = Path(row["image_path"])

    img_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise RuntimeError(f"Could not read image: {image_path}")

    img_bgr = resize_to_max_dim(img_bgr, max_dim=max_dim)
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    gray = make_gray(rgb)    
    luma = make_luma(rgb)
    clahe = make_clahe(luma)

    sobel_gray = sobel_mag(gray)
    sobel_luma = sobel_mag(luma)
    sobel_clahe = sobel_mag(clahe)

    sobel_luma_bin = adaptive_sobel_binary(sobel_luma, percentile=85)
    sobel_clahe_bin = adaptive_sobel_binary(sobel_clahe, percentile=85)

    canny_luma = cv2.Canny(luma, 80, 160)
    canny_clahe = cv2.Canny(clahe, 80, 160)

    shadow_mask = make_shadow_candidate_mask(rgb)
    shadow_overlay = overlay_mask(rgb, shadow_mask)

    sobel_clahe_shadow_suppressed = suppress_shadow_edges(sobel_clahe, shadow_mask)
    sobel_clahe_bin_shadow_suppressed = suppress_shadow_edges(sobel_clahe_bin, shadow_mask)

#    hough_raw = draw_hough_lines(rgb, sobel_clahe_bin)
#    hough_shadow_suppressed = draw_hough_lines(rgb, sobel_clahe_bin_shadow_suppressed)

    panels = [
        ("RGB original", rgb, None),
        ("Luma", luma, "gray"),
        ("CLAHE luma", clahe, "gray"),
        ("Sobel on luma", sobel_luma, "gray"),
        ("Sobel on CLAHE", sobel_clahe, "gray"),
        ("Top 15% Sobel-CLAHE edges", sobel_clahe_bin, "gray"),
        ("Canny on luma", canny_luma, "gray"),
        ("Canny on CLAHE", canny_clahe, "gray"),
        ("Shadow candidate overlay", shadow_overlay, None),
        ("Sobel-CLAHE shadow suppressed", sobel_clahe_shadow_suppressed, "gray"),
        ("Binary edges shadow suppressed", sobel_clahe_bin_shadow_suppressed, "gray"),
#        ("Hough lines on Sobel-CLAHE", hough_raw, None),
#        ("Hough after shadow suppression", hough_shadow_suppressed, None),
    ]

    ensure_dir(output_path.parent)

    fig, axes = plt.subplots(4, 4, figsize=(18, 15))
    axes_flat = axes.flatten()

    for ax, (title, img, cmap) in zip(axes_flat[:len(panels)], panels):
        ax.imshow(img, cmap=cmap)
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    for ax in axes_flat[len(panels):]:
        ax.axis("off")

    metrics = [
        f"frame_index: {int(row['frame_index_in_sequence'])}",
        f"filename: {row['filename']}",
        f"luma_mean: {float(row['luma_mean']):.2f}",
        f"luma_std: {float(row['luma_std']):.2f}",
        f"laplacian_variance: {float(row['laplacian_variance']):.2f}",
        f"edge_density: {float(row['edge_density']):.4f}",
        f"akaze_keypoints: {int(row['akaze_keypoint_count'])}",
        f"hough_lines: {int(row['hough_line_count'])}",
        f"green_ratio: {float(row['green_ratio_mean']):.4f}",
    ]

    axes_flat[15].text(
        0.0,
        1.0,
        "\n".join(metrics),
        ha="left",
        va="top",
        fontsize=9,
        family="monospace",
    )

    fig.suptitle(
        f"S3.5C edge / shadow preprocessing variants — traj01 frame {int(row['frame_index_in_sequence'])}",
        fontsize=13,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=170)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build CLAHE/Sobel/shadow preprocessing variant panels for traj01."
    )
    parser.add_argument("--config", required=True, help="Path to configs/dataset_satloc.yaml")
    parser.add_argument("--stats-csv", default=DEFAULT_STATS_CSV)
    parser.add_argument("--frames", default=None, help="Comma-separated frame indices.")
    parser.add_argument("--ranges", default=None, help="Example: 1-150:8,250-350:8,400-500:8")
    parser.add_argument("--max-dim", type=int, default=900)
    parser.add_argument("--max-panels", type=int, default=60)
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = Path(config["paths"]["output_dir"])

    stats_df = pd.read_csv(args.stats_csv)
    stats_df["frame_index_in_sequence"] = stats_df["frame_index_in_sequence"].astype(int)

    selected_frames = set(parse_int_list(args.frames))

    for start, end, count in parse_ranges(args.ranges):
        selected_frames.update(sample_evenly(start, end, count))

    if not selected_frames:
        raise ValueError("No frames selected. Use --frames or --ranges.")

    selected_df = (
        stats_df[stats_df["frame_index_in_sequence"].isin(sorted(selected_frames))]
        .sort_values("frame_index_in_sequence")
        .reset_index(drop=True)
    )

    if len(selected_df) > args.max_panels:
        selected_df = selected_df.head(args.max_panels).copy()

    panel_dir = ensure_dir(output_dir / "figures" / STAGE_NAME / "edge_shadow_variant_panels")
    manifest_path = output_dir / "metadata" / STAGE_NAME / "edge_shadow_variant_manifest.csv"

    panel_paths = []

    print("S3.5C edge/shadow variant panels")
    print("--------------------------------")
    print(f"Selected frames: {len(selected_df)}")

    for _, row in selected_df.iterrows():
        frame_idx = int(row["frame_index_in_sequence"])
        output_path = panel_dir / f"traj01_frame_{frame_idx:04d}_edge_shadow_variants.png"

        build_variant_panel(row, output_path, max_dim=args.max_dim)

        panel_paths.append(str(output_path))
        print(f"Saved frame {frame_idx}: {output_path}")

    selected_df = selected_df.copy()
    selected_df["panel_path"] = panel_paths

    keep_cols = [
        "frame_index_in_sequence",
        "token0_id",
        "token1_order",
        "filename",
        "image_path",
        "luma_mean",
        "luma_std",
        "laplacian_variance",
        "edge_density",
        "entropy_gray",
        "orb_keypoint_count",
        "akaze_keypoint_count",
        "hough_line_count",
        "green_ratio_mean",
        "flag_blurry_relative",
        "flag_low_edge_relative",
        "flag_high_edge_relative",
        "flag_green_dominant_relative",
        "panel_path",
    ]

    existing_cols = [c for c in keep_cols if c in selected_df.columns]
    ensure_dir(manifest_path.parent)
    selected_df[existing_cols].to_csv(manifest_path, index=False)

    print()
    print("S3.5C complete")
    print("--------------")
    print(f"Saved panels:   {panel_dir}")
    print(f"Saved manifest: {manifest_path}")


if __name__ == "__main__":
    main()