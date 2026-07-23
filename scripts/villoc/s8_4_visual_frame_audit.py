'''
Command Executed:

export PYTHONPATH=$PWD/src
python scripts/villoc/s8_4_visual_frame_audit.py \
  --config configs/dataset_villoc_90deg.yaml \
  --stream V \
  --sample-rate-fps 1 \
  --cols 6 \
  --thumb-width 360 \
  --batch-size 30

'''

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


def compute_image_metrics(image_bgr: np.ndarray) -> dict:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]

    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    mean_gray = float(gray.mean())
    std_gray = float(gray.std())

    # Simple edge density for audit only.
    edges = cv2.Canny(gray, 80, 160)
    edge_density = float((edges > 0).mean())

    return {
        "image_width": int(w),
        "image_height": int(h),
        "gray_mean": mean_gray,
        "gray_std": std_gray,
        "laplacian_var": lap_var,
        "edge_density": edge_density,
    }


def make_thumb_with_label(
    image_bgr: np.ndarray,
    label: str,
    thumb_width: int,
    label_height: int = 72,
) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    scale = thumb_width / float(w)
    thumb_h = max(1, int(round(h * scale)))

    thumb = cv2.resize(image_bgr, (thumb_width, thumb_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((thumb_h + label_height, thumb_width, 3), dtype=np.uint8)
    canvas[:thumb_h, :, :] = thumb

    # Label bar.
    y0 = thumb_h
    cv2.rectangle(canvas, (0, y0), (thumb_width, thumb_h + label_height), (0, 0, 0), -1)

    lines = label.split("\n")
    y = y0 + 20
    for line in lines[:3]:
        cv2.putText(
            canvas,
            line,
            (8, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        y += 20

    return canvas


def save_contact_sheet(
    df: pd.DataFrame,
    out_path: Path,
    *,
    title: str,
    cols: int = 6,
    thumb_width: int = 360,
) -> None:
    if df.empty:
        return

    thumbs = []

    for _, row in df.iterrows():
        img_path = Path(row["frame_path"])
        image = cv2.imread(str(img_path))
        if image is None:
            continue

        label = (
            f"id {int(row['sample_id']):03d} | t={float(row['target_time_s']):.1f}s\n"
            f"alt={float(row['rel_alt_m']):.1f}m | yaw={float(row['gb_yaw_deg']):.1f}deg\n"
            f"x={float(row['x_enu_m']):.0f} y={float(row['y_enu_m']):.0f}"
        )
        thumbs.append(make_thumb_with_label(image, label, thumb_width))

    if not thumbs:
        return

    tile_h, tile_w = thumbs[0].shape[:2]
    rows = int(math.ceil(len(thumbs) / cols))

    sheet = np.ones((rows * tile_h, cols * tile_w, 3), dtype=np.uint8) * 255

    for i, thumb in enumerate(thumbs):
        r = i // cols
        c = i % cols
        sheet[r * tile_h : (r + 1) * tile_h, c * tile_w : (c + 1) * tile_w] = thumb

    # Add top title strip.
    title_h = 48
    titled = np.ones((sheet.shape[0] + title_h, sheet.shape[1], 3), dtype=np.uint8) * 255
    titled[title_h:, :, :] = sheet
    cv2.putText(
        titled,
        title,
        (12, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), titled)


def save_metric_plot(df: pd.DataFrame, y_col: str, out_path: Path, title: str, ylabel: str) -> None:
    plt.figure(figsize=(10, 4))
    plt.plot(df["target_time_s"], df[y_col], marker="o", markersize=2, linewidth=1)
    plt.grid(True, alpha=0.3)
    plt.xlabel("Video time [s]")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--stream", default="V")
    parser.add_argument("--sample-rate-fps", type=float, default=1.0)
    parser.add_argument("--cols", type=int, default=6)
    parser.add_argument("--thumb-width", type=int, default=360)
    parser.add_argument("--batch-size", type=int, default=30)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    output_root = Path(cfg["dataset"]["output_root"])

    metadata_dir = output_root / "metadata"
    reports_dir = output_root / "reports"
    figures_dir = output_root / "figures" / "s8_4_visual_audit"
    figures_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    traj_csv = (
        output_root
        / "trajectories"
        / f"s8_3_reference_trajectory_{args.stream}_{args.sample_rate_fps:g}fps.csv"
    )

    if not traj_csv.exists():
        raise FileNotFoundError(f"Missing S8.3 trajectory CSV: {traj_csv}")

    df = pd.read_csv(traj_csv)

    metric_rows = []

    for _, row in df.iterrows():
        img_path = Path(row["frame_path"])
        image = cv2.imread(str(img_path))

        metric_row = row.to_dict()
        metric_row["image_read_ok"] = image is not None

        if image is not None:
            metric_row.update(compute_image_metrics(image))
        else:
            metric_row.update(
                {
                    "image_width": None,
                    "image_height": None,
                    "gray_mean": None,
                    "gray_std": None,
                    "laplacian_var": None,
                    "edge_density": None,
                }
            )

        metric_rows.append(metric_row)

    audit_df = pd.DataFrame(metric_rows)

    audit_csv = metadata_dir / f"s8_4_visual_frame_audit_{args.stream}_{args.sample_rate_fps:g}fps.csv"
    audit_df.to_csv(audit_csv, index=False)

    # Overview sheet: about 24 evenly spaced frames.
    overview_n = min(24, len(audit_df))
    overview_idx = np.linspace(0, len(audit_df) - 1, overview_n).round().astype(int)
    overview_df = audit_df.iloc[overview_idx].copy()

    overview_sheet = figures_dir / f"s8_4_contact_sheet_overview_{args.stream}_{args.sample_rate_fps:g}fps.png"
    save_contact_sheet(
        overview_df,
        overview_sheet,
        title=f"S8.4 Villoc {args.stream} {args.sample_rate_fps:g}FPS overview",
        cols=args.cols,
        thumb_width=args.thumb_width,
    )

    # Batch sheets: all sampled frames.
    batch_paths = []
    for start in range(0, len(audit_df), args.batch_size):
        end = min(start + args.batch_size, len(audit_df))
        batch_df = audit_df.iloc[start:end].copy()
        batch_path = (
            figures_dir
            / f"s8_4_contact_sheet_{args.stream}_{args.sample_rate_fps:g}fps_{start:03d}_{end-1:03d}.png"
        )
        save_contact_sheet(
            batch_df,
            batch_path,
            title=f"S8.4 Villoc {args.stream} {args.sample_rate_fps:g}FPS frames {start:03d}-{end-1:03d}",
            cols=args.cols,
            thumb_width=args.thumb_width,
        )
        batch_paths.append(str(batch_path))

    blur_fig = figures_dir / f"s8_4_laplacian_blur_{args.stream}_{args.sample_rate_fps:g}fps.png"
    brightness_fig = figures_dir / f"s8_4_brightness_{args.stream}_{args.sample_rate_fps:g}fps.png"
    contrast_fig = figures_dir / f"s8_4_contrast_{args.stream}_{args.sample_rate_fps:g}fps.png"
    edge_fig = figures_dir / f"s8_4_edge_density_{args.stream}_{args.sample_rate_fps:g}fps.png"
    altitude_fig = figures_dir / f"s8_4_rel_alt_with_samples_{args.stream}_{args.sample_rate_fps:g}fps.png"

    save_metric_plot(audit_df, "laplacian_var", blur_fig, "S8.4 Blur/Sharpness Audit", "Laplacian variance")
    save_metric_plot(audit_df, "gray_mean", brightness_fig, "S8.4 Brightness Audit", "Gray mean")
    save_metric_plot(audit_df, "gray_std", contrast_fig, "S8.4 Contrast Audit", "Gray std")
    save_metric_plot(audit_df, "edge_density", edge_fig, "S8.4 Edge Density Audit", "Edge density")

    plt.figure(figsize=(10, 4))
    plt.plot(audit_df["target_time_s"], audit_df["rel_alt_m"], marker="o", markersize=2, linewidth=1)
    plt.grid(True, alpha=0.3)
    plt.xlabel("Video time [s]")
    plt.ylabel("Relative altitude [m]")
    plt.title("S8.4 Relative Altitude at Extracted 1FPS Frames")
    plt.tight_layout()
    plt.savefig(altitude_fig, dpi=180)
    plt.close()

    summary = {
        "stream_id": args.stream,
        "sample_rate_fps": args.sample_rate_fps,
        "input_trajectory_csv": str(traj_csv),
        "output_audit_csv": str(audit_csv),
        "rows": int(len(audit_df)),
        "image_read_ok": int(audit_df["image_read_ok"].sum()),
        "image_read_failed": int((~audit_df["image_read_ok"]).sum()),
        "image_width_unique": sorted([int(x) for x in audit_df["image_width"].dropna().unique()]),
        "image_height_unique": sorted([int(x) for x in audit_df["image_height"].dropna().unique()]),
        "laplacian_var_min": float(audit_df["laplacian_var"].min()),
        "laplacian_var_median": float(audit_df["laplacian_var"].median()),
        "laplacian_var_max": float(audit_df["laplacian_var"].max()),
        "gray_mean_min": float(audit_df["gray_mean"].min()),
        "gray_mean_median": float(audit_df["gray_mean"].median()),
        "gray_mean_max": float(audit_df["gray_mean"].max()),
        "gray_std_min": float(audit_df["gray_std"].min()),
        "gray_std_median": float(audit_df["gray_std"].median()),
        "gray_std_max": float(audit_df["gray_std"].max()),
        "edge_density_min": float(audit_df["edge_density"].min()),
        "edge_density_median": float(audit_df["edge_density"].median()),
        "edge_density_max": float(audit_df["edge_density"].max()),
        "rel_alt_min_m": float(audit_df["rel_alt_m"].min()),
        "rel_alt_max_m": float(audit_df["rel_alt_m"].max()),
        "yaw_min_deg": float(audit_df["gb_yaw_deg"].min()),
        "yaw_max_deg": float(audit_df["gb_yaw_deg"].max()),
        "pitch_median_deg": float(audit_df["gb_pitch_deg"].median()),
        "figures": {
            "overview_contact_sheet": str(overview_sheet),
            "batch_contact_sheets": batch_paths,
            "blur": str(blur_fig),
            "brightness": str(brightness_fig),
            "contrast": str(contrast_fig),
            "edge_density": str(edge_fig),
            "altitude": str(altitude_fig),
        },
    }

    report_path = reports_dir / f"s8_4_visual_frame_audit_{args.stream}_{args.sample_rate_fps:g}fps_summary.json"
    report_path.write_text(json.dumps(summary, indent=2))

    print("S8.4 Villoc visual frame audit complete")
    print("--------------------------------------")
    print(f"Rows:               {len(audit_df)}")
    print(f"Images read OK:     {summary['image_read_ok']}")
    print(f"Images failed:      {summary['image_read_failed']}")
    print(f"Audit CSV:          {audit_csv}")
    print(f"Report:             {report_path}")
    print(f"Overview sheet:     {overview_sheet}")
    print(f"Batch sheets:       {len(batch_paths)}")
    print(f"Laplacian median:   {summary['laplacian_var_median']:.2f}")
    print(f"Brightness median:  {summary['gray_mean_median']:.2f}")
    print(f"Contrast median:    {summary['gray_std_median']:.2f}")
    print(f"Edge density med:   {summary['edge_density_median']:.4f}")
    print(f"Rel alt range:      {summary['rel_alt_min_m']:.2f} .. {summary['rel_alt_max_m']:.2f} m")
    print(f"Yaw range:          {summary['yaw_min_deg']:.2f} .. {summary['yaw_max_deg']:.2f} deg")
    print(f"Pitch median:       {summary['pitch_median_deg']:.2f} deg")


if __name__ == "__main__":
    main()
