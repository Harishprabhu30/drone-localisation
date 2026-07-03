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

def absdiff_uint8(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return cv2.absdiff(a, b)


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
      1-150:6,250-350:6,400-500:6
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


def resize_to_max_dim(img: np.ndarray, max_dim: int = 1000) -> np.ndarray:
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

    luma = 0.299 * r + 0.587 * g + 0.114 * b
    return np.clip(luma, 0, 255).astype(np.uint8)


# def make_clahe(gray: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
#     clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
#     return clahe.apply(gray)


# def make_bilateral(gray: np.ndarray) -> np.ndarray:
#     return cv2.bilateralFilter(
#         gray,
#         d=7,
#         sigmaColor=75,
#         sigmaSpace=75,
#     )

# adding adjustable make_clahe and make_bilateral
def make_clahe(gray: np.ndarray, clip_limit: float = 2.0, tile_size: int = 8) -> np.ndarray:
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=(tile_size, tile_size),
    )
    return clahe.apply(gray)


def make_bilateral(
    gray: np.ndarray,
    d: int = 9,
    sigma_color: float = 55,
    sigma_space: float = 55,
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
    return normalize_uint8(mag)


# def build_focused_panel(row: pd.Series, output_path: Path, max_dim: int = 1000) -> None: --> old
def build_focused_panel(
    row: pd.Series,
    output_path: Path,
    max_dim: int = 1000,
    clahe_clip_limit: float = 2.0,
    clahe_tile_size: int = 8,
    small_clahe_clip_limit: float = 1.0,
    small_clahe_tile_size: int = 8,
    bilateral_d: int = 9,
    bilateral_sigma_color: float = 55,
    bilateral_sigma_space: float = 55,
) -> None:
    image_path = Path(row["image_path"])

    img_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise RuntimeError(f"Could not read image: {image_path}")

    img_bgr = resize_to_max_dim(img_bgr, max_dim=max_dim)
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# old v1
    # luma = make_luma(rgb)
    # clahe_luma = make_clahe(luma)

    # bilateral_luma = make_bilateral(luma)
    # bilateral_on_clahe = make_bilateral(clahe_luma)

    # sobel_luma = sobel_mag(luma)
    # sobel_clahe_luma = sobel_mag(clahe_luma)
    # sobel_bilateral_luma = sobel_mag(bilateral_luma)
    # sobel_bilateral_on_clahe = sobel_mag(bilateral_on_clahe)

# v3 
    luma = make_luma(rgb)

    clahe_luma = make_clahe(
        luma,
        clip_limit=clahe_clip_limit,
        tile_size=clahe_tile_size,
    )

    alt_clahe_luma = make_clahe(
        luma,
        clip_limit=small_clahe_clip_limit,
        tile_size=small_clahe_tile_size,
    )

    bilateral_on_clahe = make_bilateral(
        clahe_luma,
        d=bilateral_d,
        sigma_color=bilateral_sigma_color,
        sigma_space=bilateral_sigma_space,
    )

    bilateral_on_alt_clahe = make_bilateral(
        alt_clahe_luma,
        d=bilateral_d,
        sigma_color=bilateral_sigma_color,
        sigma_space=bilateral_sigma_space,
    )

    sobel_luma = sobel_mag(luma)
    sobel_clahe_luma = sobel_mag(clahe_luma)
    sobel_bilateral_on_clahe = sobel_mag(bilateral_on_clahe)
    sobel_bilateral_on_alt_clahe = sobel_mag(bilateral_on_alt_clahe)

    diff_bilateral = absdiff_uint8(bilateral_on_clahe, bilateral_on_alt_clahe)
    diff_sobel = absdiff_uint8(sobel_bilateral_on_clahe, sobel_bilateral_on_alt_clahe)

    diff_bilateral_mean = float(diff_bilateral.mean())
    diff_bilateral_p95 = float(np.percentile(diff_bilateral, 95))

    diff_sobel_mean = float(diff_sobel.mean())
    diff_sobel_p95 = float(np.percentile(diff_sobel, 95))

    panels = [
        ("RGB original", rgb, None),
        ("Sobel on luma", sobel_luma, "gray"),
        ("Sobel on CLAHE-luma", sobel_clahe_luma, "gray"),

        ("Bilateral( CLAHE-luma )", bilateral_on_clahe, "gray"),
        ("Bilateral( alt-CLAHE-luma )", bilateral_on_alt_clahe, "gray"),
        ("Abs diff: bilateral variants", diff_bilateral, "gray"),

        ("Sobel on bilateral( CLAHE-luma )", sobel_bilateral_on_clahe, "gray"),
        ("Sobel on bilateral( alt-CLAHE-luma )", sobel_bilateral_on_alt_clahe, "gray"),
        ("Abs diff: Sobel bilateral variants", diff_sobel, "gray"),
    ]

    ensure_dir(output_path.parent)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes_flat = axes.flatten()

    for ax, (title, img, cmap) in zip(axes_flat, panels):
        ax.imshow(img, cmap=cmap)
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    fig.suptitle(
        f"Focused bilateral+CLAHE comparison — traj01 frame {int(row['frame_index_in_sequence'])} — {row['filename']}\n"
        f"diff_bilateral_mean={diff_bilateral_mean:.2f}, diff_bilateral_p95={diff_bilateral_p95:.2f}, "
        f"diff_sobel_mean={diff_sobel_mean:.2f}, diff_sobel_p95={diff_sobel_p95:.2f}",
        fontsize=12,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close(fig)


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
        description="Build focused luma/CLAHE/bilateral/Sobel panels for SatLoc traj01."
    )
    parser.add_argument("--config", required=True, help="Path to configs/dataset_satloc.yaml")
    parser.add_argument("--stats-csv", default=DEFAULT_STATS_CSV)
    parser.add_argument("--frames", default=None, help="Comma-separated frame indices.")
    parser.add_argument("--ranges", default=None, help="Example: 1-150:6,250-350:6,400-500:6")
    parser.add_argument("--max-dim", type=int, default=1000)
    parser.add_argument("--max-panels", type=int, default=80)

    parser.add_argument("--clahe-clip-limit", type=float, default=2.0)
    parser.add_argument("--clahe-tile-size", type=int, default=12)

    parser.add_argument("--small-clahe-clip-limit", type=float, default=2.0)
    parser.add_argument("--small-clahe-tile-size", type=int, default=4)

    parser.add_argument("--bilateral-d", type=int, default=9)
    parser.add_argument("--bilateral-sigma-color", type=float, default=55.0)
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

    panel_dir = ensure_dir(output_dir / "figures" / STAGE_NAME / "focused_sobel_panels")
    metadata_dir = ensure_dir(output_dir / "metadata" / STAGE_NAME)
    manifest_path = metadata_dir / "focused_sobel_panel_manifest.csv"

    panel_paths = []

    print("S3.5D.1 focused Sobel preprocessing panels")
    print("------------------------------------------")
    print(f"Selected frames: {len(selected_df)}")

    for _, row in selected_df.iterrows():
        frame_idx = int(row["frame_index_in_sequence"])
        output_path = panel_dir / f"traj01_frame_{frame_idx:04d}_focused_sobel.png"

        # build_focused_panel(row, output_path, max_dim=args.max_dim) --> old
        build_focused_panel(
            row,
            output_path,
            max_dim=args.max_dim,
            clahe_clip_limit=args.clahe_clip_limit,
            clahe_tile_size=args.clahe_tile_size,
            small_clahe_clip_limit=args.small_clahe_clip_limit,
            small_clahe_tile_size=args.small_clahe_tile_size,
            bilateral_d=args.bilateral_d,
            bilateral_sigma_color=args.bilateral_sigma_color,
            bilateral_sigma_space=args.bilateral_sigma_space,
        )    

        panel_paths.append(str(output_path))
        print(f"Saved frame {frame_idx}: {output_path}")

    selected_df = selected_df.copy()
    selected_df["panel_path"] = panel_paths
    selected_df.to_csv(manifest_path, index=False)

    print()
    print("S3.5D.1 complete")
    print("----------------")
    print(f"Saved panels:   {panel_dir}")
    print(f"Saved manifest: {manifest_path}")


if __name__ == "__main__":
    main()