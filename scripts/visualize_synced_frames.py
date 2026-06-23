import argparse
from pathlib import Path

import folium
import matplotlib.pyplot as plt
import pandas as pd

from uavloc.utils.config import load_config


def plot_synced_xy(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 6))
    plt.plot(df["x_enu_m"], df["y_enu_m"], linewidth=1.2, label="Synced frame path")
    plt.scatter(df["x_enu_m"], df["y_enu_m"], s=10, alpha=0.7, label="Frames")
    plt.scatter(df["x_enu_m"].iloc[0], df["y_enu_m"].iloc[0], s=80, marker="o", label="Start")
    plt.scatter(df["x_enu_m"].iloc[-1], df["y_enu_m"].iloc[-1], s=80, marker="x", label="End")

    plt.xlabel("X ENU [m]")
    plt.ylabel("Y ENU [m]")
    plt.title("Synchronized MAV Frame Locations")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def build_synced_map(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    center_lat = float(df["lat"].mean())
    center_lon = float(df["lon"].mean())

    m = folium.Map(location=[center_lat, center_lon], zoom_start=20, control_scale=True)

    coords = list(zip(df["lat"], df["lon"]))

    folium.PolyLine(
        coords,
        weight=4,
        opacity=0.9,
        tooltip="Synced frame trajectory",
    ).add_to(m)

    folium.Marker(
        [df["lat"].iloc[0], df["lon"].iloc[0]],
        popup=f"Start frame: {df['imgid'].iloc[0]}",
        tooltip="Start",
    ).add_to(m)

    folium.Marker(
        [df["lat"].iloc[-1], df["lon"].iloc[-1]],
        popup=f"End frame: {df['imgid'].iloc[-1]}",
        tooltip="End",
    ).add_to(m)

    # Add a few sampled frame markers, not all 350, to keep map light.
    sample_df = df.iloc[::50].copy()

    for _, row in sample_df.iterrows():
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=3,
            popup=f"imgid: {int(row['imgid'])}<br>{row['image_filename']}",
            fill=True,
        ).add_to(m)

    m.save(str(output_path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to dataset config YAML")
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = Path(config["paths"]["output_dir"])

    synced_csv = output_dir / "metadata" / "synchronized_frames.csv"

    if not synced_csv.exists():
        raise FileNotFoundError(
            f"Synchronized frames file not found: {synced_csv}\n"
            "Run sync_zurich_frames.py first."
        )

    df = pd.read_csv(synced_csv)

    if df.empty:
        raise ValueError("synchronized_frames.csv is empty.")

    figures_dir = output_dir / "figures"
    maps_dir = output_dir / "maps"

    xy_path = figures_dir / "synced_frame_locations_xy.png"
    map_path = maps_dir / "synced_frame_locations_map.html"

    plot_synced_xy(df, xy_path)
    build_synced_map(df, map_path)

    print("Synced frame visualization generated")
    print("------------------------------------")
    print(f"Synced CSV: {synced_csv}")
    print(f"XY plot:    {xy_path}")
    print(f"Map HTML:   {map_path}")
    print(f"Frames:     {len(df)}")
    print(f"Imgid range:{int(df['imgid'].min())} to {int(df['imgid'].max())}")
    print(f"X range [m]: {df['x_enu_m'].min():.3f} to {df['x_enu_m'].max():.3f}")
    print(f"Y range [m]: {df['y_enu_m'].min():.3f} to {df['y_enu_m'].max():.3f}")


if __name__ == "__main__":
    main()