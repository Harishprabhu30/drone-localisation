from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Set, Tuple

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

    values = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        values.append(int(part))

    return values


def parse_ranges(text: str | None) -> List[Tuple[int, int, int]]:
    """
    Format:
      "1-150:25,250-350:20,400-500:20"

    Meaning:
      from 1 to 150, sample 25 frames evenly
      from 250 to 350, sample 20 frames evenly
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

        start = int(start_text)
        end = int(end_text)
        count = int(count_part)

        ranges.append((start, end, count))

    return ranges


def sample_evenly_from_range(start: int, end: int, count: int) -> List[int]:
    if count <= 0:
        return []

    if start > end:
        start, end = end, start

    if count == 1:
        return [start]

    values = np.linspace(start, end, count)
    return sorted(set(int(round(v)) for v in values))


def select_metric_extremes(
    stats_df: pd.DataFrame,
    metric: str,
    n: int,
    mode: str,
) -> pd.DataFrame:
    if metric not in stats_df.columns:
        raise ValueError(f"Metric not found in stats CSV: {metric}")

    work = stats_df.copy()
    work[metric] = pd.to_numeric(work[metric], errors="coerce")
    work = work.dropna(subset=[metric])

    if mode == "low":
        selected = work.nsmallest(n, metric)
    elif mode == "high":
        selected = work.nlargest(n, metric)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    selected = selected.copy()
    selected["selection_reason"] = f"{mode}_{metric}"
    return selected


def parse_metric_selection(text: str | None) -> List[Tuple[str, int]]:
    """
    Format:
      "laplacian_variance:8,edge_density:8,akaze_keypoint_count:8"
    """
    if not text:
        return []

    parsed = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue

        metric, n_text = part.split(":")
        parsed.append((metric.strip(), int(n_text)))

    return parsed


def load_optional_labels(labels_csv: str | None) -> pd.DataFrame | None:
    if not labels_csv:
        return None

    path = Path(labels_csv)
    if not path.exists():
        raise FileNotFoundError(f"Labels CSV not found: {path}")

    labels_df = pd.read_csv(path)

    required = {"frame_index_in_sequence", "label"}
    missing = required - set(labels_df.columns)
    if missing:
        raise ValueError(f"Labels CSV missing columns: {missing}")

    return labels_df


def resize_to_max_dim(img: np.ndarray, max_dim: int = 900) -> np.ndarray:
    h, w = img.shape[:2]
    scale = max_dim / max(h, w)

    if scale >= 1.0:
        return img

    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    min_v = float(np.nanmin(arr))
    max_v = float(np.nanmax(arr))

    if max_v - min_v < 1e-9:
        return np.zeros(arr.shape, dtype=np.uint8)

    out = (arr - min_v) / (max_v - min_v)
    out = np.clip(out * 255.0, 0, 255).astype(np.uint8)
    return out


def make_luma(rgb: np.ndarray) -> np.ndarray:
    r = rgb[:, :, 0].astype(np.float32)
    g = rgb[:, :, 1].astype(np.float32)
    b = rgb[:, :, 2].astype(np.float32)

    luma = 0.299 * r + 0.587 * g + 0.114 * b
    return np.clip(luma, 0, 255).astype(np.uint8)


def make_clahe(gray: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def make_sobel_magnitude(gray: np.ndarray) -> np.ndarray:
    gray_f = gray.astype(np.float32) / 255.0
    gx = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    return normalize_to_uint8(mag)


def draw_hough_lines(rgb: np.ndarray, edges: np.ndarray) -> np.ndarray:
    out = rgb.copy()

    lines = cv2.HoughLinesP(
        edges,
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


def draw_orb_keypoints(rgb: np.ndarray, gray: np.ndarray) -> Tuple[np.ndarray, int]:
    orb = cv2.ORB_create(nfeatures=2500)
    kps = orb.detect(gray, None)

    out_bgr = cv2.drawKeypoints(
        cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
        kps,
        None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )

    out_rgb = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)
    return out_rgb, len(kps)


def draw_akaze_keypoints(rgb: np.ndarray, gray: np.ndarray) -> Tuple[np.ndarray, int]:
    akaze = cv2.AKAZE_create()
    kps = akaze.detect(gray, None)

    out_bgr = cv2.drawKeypoints(
        cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
        kps,
        None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )

    out_rgb = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)
    return out_rgb, len(kps)


def metric_text(row: pd.Series) -> str:
    keys = [
        "frame_index_in_sequence",
        "token0_id",
        "token1_order",
        "luma_mean",
        "luma_std",
        "laplacian_variance",
        "edge_density",
        "entropy_gray",
        "orb_keypoint_count",
        "akaze_keypoint_count",
        "hough_line_count",
        "green_ratio_mean",
    ]

    lines = []
    for key in keys:
        if key not in row:
            continue

        value = row[key]
        if isinstance(value, float):
            lines.append(f"{key}: {value:.4f}")
        else:
            lines.append(f"{key}: {value}")

    if "label" in row and pd.notna(row["label"]):
        lines.append(f"label: {row['label']}")

    if "notes" in row and pd.notna(row["notes"]):
        lines.append(f"notes: {row['notes']}")

    if "selection_reason" in row and pd.notna(row["selection_reason"]):
        lines.append(f"selected: {row['selection_reason']}")

    return "\n".join(lines)


def build_decomposition_panel(
    row: pd.Series,
    output_path: Path,
    max_dim: int = 900,
) -> None:
    image_path = Path(row["image_path"])

    img_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise RuntimeError(f"Could not read image: {image_path}")

    img_bgr = resize_to_max_dim(img_bgr, max_dim=max_dim)
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    luma = make_luma(rgb)
    clahe = make_clahe(luma)
    edges = cv2.Canny(luma, 80, 160)
    sobel = make_sobel_magnitude(luma)
    lines_rgb = draw_hough_lines(rgb, edges)
    orb_rgb, orb_count = draw_orb_keypoints(rgb, luma)
    akaze_rgb, akaze_count = draw_akaze_keypoints(rgb, luma)

    panels = [
        ("RGB original", rgb, None),
        ("Luma / perceived brightness", luma, "gray"),
        ("CLAHE luma / local contrast", clahe, "gray"),
        ("Canny edges", edges, "gray"),
        ("Sobel gradient magnitude", sobel, "gray"),
        ("Hough line segments", lines_rgb, None),
        (f"ORB keypoints ({orb_count})", orb_rgb, None),
        (f"AKAZE keypoints ({akaze_count})", akaze_rgb, None),
    ]

    ensure_dir(output_path.parent)

    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    axes_flat = axes.flatten()

    for ax, (title, img, cmap) in zip(axes_flat[:8], panels):
        ax.imshow(img, cmap=cmap)
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    axes_flat[8].axis("off")
    axes_flat[8].text(
        0.0,
        1.0,
        metric_text(row),
        va="top",
        ha="left",
        fontsize=9,
        family="monospace",
    )

    fig.suptitle(
        f"SatLoc traj01 decomposition — frame {int(row['frame_index_in_sequence'])} — {row['filename']}",
        fontsize=13,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close(fig)


def build_selection(
    stats_df: pd.DataFrame,
    frames: List[int],
    ranges: List[Tuple[int, int, int]],
    select_low: List[Tuple[str, int]],
    select_high: List[Tuple[str, int]],
    labels_df: pd.DataFrame | None,
) -> pd.DataFrame:
    selected_parts = []

    if frames:
        part = stats_df[stats_df["frame_index_in_sequence"].isin(frames)].copy()
        part["selection_reason"] = "manual_frame_arg"
        selected_parts.append(part)

    for start, end, count in ranges:
        sampled_frames = sample_evenly_from_range(start, end, count)
        part = stats_df[stats_df["frame_index_in_sequence"].isin(sampled_frames)].copy()
        part["selection_reason"] = f"range_{start}_{end}_n{count}"
        selected_parts.append(part)

    for metric, n in select_low:
        selected_parts.append(select_metric_extremes(stats_df, metric, n, mode="low"))

    for metric, n in select_high:
        selected_parts.append(select_metric_extremes(stats_df, metric, n, mode="high"))

    if labels_df is not None:
        labelled_frames = labels_df["frame_index_in_sequence"].astype(int).tolist()
        part = stats_df[stats_df["frame_index_in_sequence"].isin(labelled_frames)].copy()
        part["selection_reason"] = "labels_csv"
        selected_parts.append(part)

    if not selected_parts:
        raise ValueError(
            "No frames selected. Use --frames, --ranges, --select-low, --select-high, or --labels-csv."
        )

    selected = pd.concat(selected_parts, ignore_index=True)

    # Merge labels if provided.
    if labels_df is not None:
        labels_df = labels_df.copy()
        labels_df["frame_index_in_sequence"] = labels_df["frame_index_in_sequence"].astype(int)
        selected = selected.merge(
            labels_df,
            on="frame_index_in_sequence",
            how="left",
            suffixes=("", "_label"),
        )

    # Keep one row per frame but combine selection reasons.
    selected["frame_index_in_sequence"] = selected["frame_index_in_sequence"].astype(int)

    reason_df = (
        selected.groupby("frame_index_in_sequence")["selection_reason"]
        .apply(lambda x: ";".join(sorted(set(str(v) for v in x if pd.notna(v)))))
        .reset_index()
    )

    selected = (
        selected.sort_values(["frame_index_in_sequence"])
        .drop_duplicates(subset=["frame_index_in_sequence"], keep="first")
        .drop(columns=["selection_reason"])
        .merge(reason_df, on="frame_index_in_sequence", how="left")
        .sort_values(["frame_index_in_sequence"])
        .reset_index(drop=True)
    )

    return selected


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build visual decomposition panels for selected SatLoc traj01 frames."
    )
    parser.add_argument("--config", required=True, help="Path to configs/dataset_satloc.yaml")
    parser.add_argument(
        "--stats-csv",
        default=DEFAULT_STATS_CSV,
        help="Path to traj01 image stats CSV from S3.5A.",
    )
    parser.add_argument(
        "--frames",
        default=None,
        help="Comma-separated frame_index_in_sequence values, e.g. 1,50,120,260,420.",
    )
    parser.add_argument(
        "--ranges",
        default=None,
        help="Frame ranges to sample, e.g. 1-150:12,250-350:10,400-500:10.",
    )
    parser.add_argument(
        "--select-low",
        default=None,
        help="Select n lowest frames by metric, e.g. laplacian_variance:8,edge_density:8.",
    )
    parser.add_argument(
        "--select-high",
        default=None,
        help="Select n highest frames by metric, e.g. edge_density:8,akaze_keypoint_count:8.",
    )
    parser.add_argument(
        "--labels-csv",
        default=None,
        help="Optional CSV with frame_index_in_sequence,label,notes columns.",
    )
    parser.add_argument(
        "--max-dim",
        type=int,
        default=900,
        help="Maximum image dimension for panel visualization.",
    )
    parser.add_argument(
        "--max-panels",
        type=int,
        default=80,
        help="Safety limit for number of panels generated.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = Path(config["paths"]["output_dir"])

    stats_csv = Path(args.stats_csv)
    if not stats_csv.exists():
        raise FileNotFoundError(
            f"Missing stats CSV: {stats_csv}. "
            "Run scripts/satloc/run_traj01_visual_domain_diagnostics.py first."
        )

    stats_df = pd.read_csv(stats_csv)
    stats_df["frame_index_in_sequence"] = stats_df["frame_index_in_sequence"].astype(int)

    labels_df = load_optional_labels(args.labels_csv)

    frames = parse_int_list(args.frames)
    ranges = parse_ranges(args.ranges)
    select_low = parse_metric_selection(args.select_low)
    select_high = parse_metric_selection(args.select_high)

    selected_df = build_selection(
        stats_df=stats_df,
        frames=frames,
        ranges=ranges,
        select_low=select_low,
        select_high=select_high,
        labels_df=labels_df,
    )

    if len(selected_df) > args.max_panels:
        print(
            f"Selected {len(selected_df)} frames but max-panels={args.max_panels}. "
            f"Keeping first {args.max_panels} by frame index."
        )
        selected_df = selected_df.head(args.max_panels).copy()

    panel_dir = ensure_dir(output_dir / "figures" / STAGE_NAME / "decomposition_panels")
    manifest_path = output_dir / "metadata" / STAGE_NAME / "decomposition_panel_manifest.csv"

    panel_paths = []

    print("S3.5B traj01 decomposition panels")
    print("---------------------------------")
    print(f"Selected frames: {len(selected_df)}")

    for _, row in selected_df.iterrows():
        frame_idx = int(row["frame_index_in_sequence"])
        token0 = int(row["token0_id"]) if pd.notna(row["token0_id"]) else frame_idx

        safe_reason = str(row.get("selection_reason", "selected"))
        safe_reason = (
            safe_reason.replace(";", "_")
            .replace("/", "_")
            .replace(":", "_")
            .replace(" ", "_")
        )

        output_path = panel_dir / f"traj01_frame_{frame_idx:04d}_token_{token0:04d}_{safe_reason}.png"

        build_decomposition_panel(row, output_path, max_dim=args.max_dim)
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
        "lon",
        "lat",
        "x_enu_m",
        "y_enu_m",
        "selection_reason",
        "label",
        "notes",
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
        "flag_low_orb_relative",
        "flag_green_dominant_relative",
        "panel_path",
    ]

    existing_cols = [c for c in keep_cols if c in selected_df.columns]
    ensure_dir(manifest_path.parent)
    selected_df[existing_cols].to_csv(manifest_path, index=False)

    print()
    print("S3.5B complete")
    print("--------------")
    print(f"Saved panel directory: {panel_dir}")
    print(f"Saved manifest: {manifest_path}")


if __name__ == "__main__":
    main()