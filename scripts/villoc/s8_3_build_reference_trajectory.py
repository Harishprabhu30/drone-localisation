'''
Command Executed:

export PYTHONPATH=$PWD/src
python scripts/villoc/s8_3_build_reference_trajectory.py \
  --config configs/dataset_villoc_90deg.yaml \
  --stream V \
  --sample-rate-fps 1

'''

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml


def latlon_to_local_enu_fallback(lat, lon, alt, lat0, lon0, alt0):
    """Small-area ENU approximation. Good enough for short local UAV trajectory audit."""
    r_earth = 6378137.0
    lat_rad = math.radians(lat)
    lat0_rad = math.radians(lat0)

    x_east = math.radians(lon - lon0) * r_earth * math.cos(lat0_rad)
    y_north = math.radians(lat - lat0) * r_earth
    z_up = alt - alt0

    return x_east, y_north, z_up


def add_local_enu(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = df.copy()

    valid = df[["lat", "lon", "abs_alt_m"]].dropna()
    if valid.empty:
        raise ValueError("No valid lat/lon/abs_alt_m rows found.")

    lat0 = float(valid.iloc[0]["lat"])
    lon0 = float(valid.iloc[0]["lon"])
    alt0 = float(valid.iloc[0]["abs_alt_m"])

    origin = {
        "lat0": lat0,
        "lon0": lon0,
        "alt0_m": alt0,
        "origin_policy": "first_valid_extracted_frame",
        "enu_method": "small_area_equirectangular_fallback",
    }

    xs, ys, zs = [], [], []
    for _, row in df.iterrows():
        x, y, z = latlon_to_local_enu_fallback(
            float(row["lat"]),
            float(row["lon"]),
            float(row["abs_alt_m"]),
            lat0,
            lon0,
            alt0,
        )
        xs.append(x)
        ys.append(y)
        zs.append(z)

    df["x_enu_m"] = xs
    df["y_enu_m"] = ys
    df["z_enu_m"] = zs

    return df, origin


def path_length_m(df: pd.DataFrame) -> float:
    if len(df) < 2:
        return 0.0

    dx = df["x_enu_m"].diff()
    dy = df["y_enu_m"].diff()
    return float((dx.pow(2) + dy.pow(2)).pow(0.5).sum())


def save_xy_plot(df: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(8, 8))
    plt.plot(df["x_enu_m"], df["y_enu_m"], marker="o", markersize=2, linewidth=1)
    plt.scatter(df["x_enu_m"].iloc[0], df["y_enu_m"].iloc[0], marker="o", s=80, label="start")
    plt.scatter(df["x_enu_m"].iloc[-1], df["y_enu_m"].iloc[-1], marker="x", s=80, label="end")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.xlabel("East X [m]")
    plt.ylabel("North Y [m]")
    plt.title("Villoc 90° V 1FPS Reference Trajectory [ENU]")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def save_altitude_plot(df: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(10, 4))
    plt.plot(df["target_time_s"], df["rel_alt_m"], label="relative altitude [m]")
    plt.plot(df["target_time_s"], df["abs_alt_m"], label="absolute altitude [m]")
    plt.grid(True, alpha=0.3)
    plt.xlabel("Video time [s]")
    plt.ylabel("Altitude [m]")
    plt.title("Villoc 90° V 1FPS Altitude Profile")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def save_yaw_plot(df: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(10, 4))
    plt.plot(df["target_time_s"], df["gb_yaw_deg"])
    plt.grid(True, alpha=0.3)
    plt.xlabel("Video time [s]")
    plt.ylabel("Gimbal/body yaw [deg]")
    plt.title("Villoc 90° V 1FPS Yaw Profile")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--stream", default="V")
    parser.add_argument("--sample-rate-fps", type=float, default=1.0)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())

    output_root = Path(cfg["dataset"]["output_root"])
    metadata_dir = output_root / "metadata"
    reports_dir = output_root / "reports"
    figures_dir = output_root / "figures"
    trajectories_dir = output_root / "trajectories"

    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    trajectories_dir.mkdir(parents=True, exist_ok=True)

    in_csv = metadata_dir / f"s8_2_extracted_frames_{args.stream}_{args.sample_rate_fps:g}fps.csv"
    if not in_csv.exists():
        raise FileNotFoundError(f"Missing S8.2 metadata: {in_csv}")

    df = pd.read_csv(in_csv)
    traj_df, origin = add_local_enu(df)

    # Standard trajectory columns first, diagnostics after.
    ordered_cols = [
        "sample_id",
        "stream_id",
        "target_time_s",
        "srt_video_time_s",
        "alignment_error_ms",
        "frame_cnt",
        "frame_path",
        "lat",
        "lon",
        "rel_alt_m",
        "abs_alt_m",
        "x_enu_m",
        "y_enu_m",
        "z_enu_m",
        "gb_yaw_deg",
        "gb_pitch_deg",
        "gb_roll_deg",
        "focal_len_mm",
        "dzoom_ratio",
        "modality",
        "role",
        "source_video",
        "extraction_status",
    ]
    ordered_cols = [c for c in ordered_cols if c in traj_df.columns]
    extra_cols = [c for c in traj_df.columns if c not in ordered_cols]
    traj_df = traj_df[ordered_cols + extra_cols]

    out_csv = trajectories_dir / f"s8_3_reference_trajectory_{args.stream}_{args.sample_rate_fps:g}fps.csv"
    traj_df.to_csv(out_csv, index=False)

    xy_fig = figures_dir / f"s8_3_reference_xy_{args.stream}_{args.sample_rate_fps:g}fps.png"
    alt_fig = figures_dir / f"s8_3_altitude_profile_{args.stream}_{args.sample_rate_fps:g}fps.png"
    yaw_fig = figures_dir / f"s8_3_yaw_profile_{args.stream}_{args.sample_rate_fps:g}fps.png"

    save_xy_plot(traj_df, xy_fig)
    save_altitude_plot(traj_df, alt_fig)
    save_yaw_plot(traj_df, yaw_fig)

    total_path_m = path_length_m(traj_df)
    displacement_m = math.hypot(
        float(traj_df["x_enu_m"].iloc[-1] - traj_df["x_enu_m"].iloc[0]),
        float(traj_df["y_enu_m"].iloc[-1] - traj_df["y_enu_m"].iloc[0]),
    )

    summary = {
        "stream_id": args.stream,
        "sample_rate_fps": args.sample_rate_fps,
        "input_csv": str(in_csv),
        "output_trajectory_csv": str(out_csv),
        "rows": int(len(traj_df)),
        "origin": origin,
        "x_range_m": [
            float(traj_df["x_enu_m"].min()),
            float(traj_df["x_enu_m"].max()),
        ],
        "y_range_m": [
            float(traj_df["y_enu_m"].min()),
            float(traj_df["y_enu_m"].max()),
        ],
        "z_range_m": [
            float(traj_df["z_enu_m"].min()),
            float(traj_df["z_enu_m"].max()),
        ],
        "lat_range": [
            float(traj_df["lat"].min()),
            float(traj_df["lat"].max()),
        ],
        "lon_range": [
            float(traj_df["lon"].min()),
            float(traj_df["lon"].max()),
        ],
        "rel_alt_range_m": [
            float(traj_df["rel_alt_m"].min()),
            float(traj_df["rel_alt_m"].max()),
        ],
        "abs_alt_range_m": [
            float(traj_df["abs_alt_m"].min()),
            float(traj_df["abs_alt_m"].max()),
        ],
        "yaw_range_deg": [
            float(traj_df["gb_yaw_deg"].min()),
            float(traj_df["gb_yaw_deg"].max()),
        ],
        "pitch_median_deg": float(traj_df["gb_pitch_deg"].median()),
        "total_2d_path_length_m": total_path_m,
        "start_to_end_2d_displacement_m": displacement_m,
        "alignment_error_ms_median": float(traj_df["alignment_error_ms"].median()),
        "alignment_error_ms_max": float(traj_df["alignment_error_ms"].max()),
        "figures": {
            "xy": str(xy_fig),
            "altitude": str(alt_fig),
            "yaw": str(yaw_fig),
        },
    }

    report_path = reports_dir / f"s8_3_reference_trajectory_{args.stream}_{args.sample_rate_fps:g}fps_summary.json"
    report_path.write_text(json.dumps(summary, indent=2))

    print("S8.3 Villoc reference trajectory complete")
    print("----------------------------------------")
    print(f"Input:             {in_csv}")
    print(f"Rows:              {len(traj_df)}")
    print(f"Trajectory CSV:    {out_csv}")
    print(f"Report:            {report_path}")
    print(f"XY figure:         {xy_fig}")
    print(f"Altitude figure:   {alt_fig}")
    print(f"Yaw figure:        {yaw_fig}")
    print(f"2D path length:    {total_path_m:.2f} m")
    print(f"Start-end disp:    {displacement_m:.2f} m")
    print(f"X range:           {summary['x_range_m'][0]:.2f} .. {summary['x_range_m'][1]:.2f} m")
    print(f"Y range:           {summary['y_range_m'][0]:.2f} .. {summary['y_range_m'][1]:.2f} m")
    print(f"Rel alt range:     {summary['rel_alt_range_m'][0]:.2f} .. {summary['rel_alt_range_m'][1]:.2f} m")
    print(f"Pitch median:      {summary['pitch_median_deg']:.2f} deg")


if __name__ == "__main__":
    main()
