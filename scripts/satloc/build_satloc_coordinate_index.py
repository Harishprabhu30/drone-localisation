from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml

from uavloc.data.satloc_coordinate_index import (
    build_satellite_tile_index,
    build_uav_reference_index,
    ensure_dir,
    save_json,
)


SEQUENCE_COLORS = {
    "traj01": "blue",
    "traj03": "orange",
    "traj04": "green",
}


def plot_single_sequence_lonlat(group: pd.DataFrame, output_path: Path) -> None:
    ensure_dir(output_path.parent)

    sequence = group["sequence"].iloc[0]
    color = SEQUENCE_COLORS.get(sequence, "blue")

    plt.figure(figsize=(8, 7))
    plt.plot(group["lon"], group["lat"], linewidth=1.8, color=color)
    plt.scatter(group["lon"].iloc[0], group["lat"].iloc[0], marker="o", s=60, color=color, label="start")
    plt.scatter(group["lon"].iloc[-1], group["lat"].iloc[-1], marker="x", s=80, color="red", label="end")

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title(f"{sequence} UAV Reference Trajectory in lon/lat")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_single_sequence_local_enu(group: pd.DataFrame, output_path: Path) -> None:
    ensure_dir(output_path.parent)

    sequence = group["sequence"].iloc[0]
    color = SEQUENCE_COLORS.get(sequence, "blue")

    plt.figure(figsize=(8, 7))
    plt.plot(group["x_enu_m"], group["y_enu_m"], linewidth=1.8, color=color)
    plt.scatter(group["x_enu_m"].iloc[0], group["y_enu_m"].iloc[0], marker="o", s=60, color=color, label="start")
    plt.scatter(group["x_enu_m"].iloc[-1], group["y_enu_m"].iloc[-1], marker="x", s=80, color="red", label="end")

    plt.xlabel("x ENU from sequence start [m]")
    plt.ylabel("y ENU from sequence start [m]")
    plt.title(f"{sequence} UAV Reference Trajectory in local ENU")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_all_sequences_lonlat(traj_df: pd.DataFrame, output_path: Path) -> None:
    ensure_dir(output_path.parent)

    plt.figure(figsize=(8, 7))
    for sequence, group in traj_df.groupby("sequence", sort=False):
        color = SEQUENCE_COLORS.get(sequence, None)
        plt.plot(group["lon"], group["lat"], linewidth=1.5, label=sequence, color=color)
        plt.scatter(group["lon"].iloc[0], group["lat"].iloc[0], marker="o", s=40, color=color)
        plt.scatter(group["lon"].iloc[-1], group["lat"].iloc[-1], marker="x", s=60, color=color)

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("SatLoc UAV Reference Trajectories in lon/lat")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_all_sequences_global_enu(traj_df: pd.DataFrame, output_path: Path) -> None:
    ensure_dir(output_path.parent)

    plt.figure(figsize=(8, 7))
    for sequence, group in traj_df.groupby("sequence", sort=False):
        color = SEQUENCE_COLORS.get(sequence, None)
        plt.plot(group["x_enu_global_m"], group["y_enu_global_m"], linewidth=1.5, label=sequence, color=color)
        plt.scatter(group["x_enu_global_m"].iloc[0], group["y_enu_global_m"].iloc[0], marker="o", s=40, color=color)
        plt.scatter(group["x_enu_global_m"].iloc[-1], group["y_enu_global_m"].iloc[-1], marker="x", s=60, color=color)

    plt.xlabel("x ENU global [m]")
    plt.ylabel("y ENU global [m]")
    plt.title("SatLoc UAV Reference Trajectories in global ENU")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

def load_config(config_path: str | Path) -> dict:
    with Path(config_path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def plot_reference_trajectory_xy(traj_df: pd.DataFrame, output_path: Path) -> None:
    ensure_dir(output_path.parent)

    plt.figure(figsize=(8, 7))

    for sequence, group in traj_df.groupby("sequence", sort=False):
        plt.plot(group["x_enu_m"], group["y_enu_m"], linewidth=1.5, label=sequence)
        plt.scatter(group["x_enu_m"].iloc[0], group["y_enu_m"].iloc[0], marker="o", s=40)
        plt.scatter(group["x_enu_m"].iloc[-1], group["y_enu_m"].iloc[-1], marker="x", s=50)

    plt.xlabel("x ENU from sequence start [m]")
    plt.ylabel("y ENU from sequence start [m]")
    plt.title("SatLoc UAV Reference Trajectories from Filename Coordinates")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_reference_trajectory_lonlat(traj_df: pd.DataFrame, output_path: Path) -> None:
    ensure_dir(output_path.parent)

    plt.figure(figsize=(8, 7))

    for sequence, group in traj_df.groupby("sequence", sort=False):
        plt.plot(group["lon"], group["lat"], linewidth=1.5, label=sequence)
        plt.scatter(group["lon"].iloc[0], group["lat"].iloc[0], marker="o", s=40)
        plt.scatter(group["lon"].iloc[-1], group["lat"].iloc[-1], marker="x", s=50)

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("SatLoc UAV Reference Trajectories in lon/lat")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def write_single_sequence_folium_map(
    group: pd.DataFrame,
    tile_df: pd.DataFrame,
    output_path: Path,
) -> bool:
    try:
        import folium
    except Exception:
        return False

    ensure_dir(output_path.parent)

    sequence = group["sequence"].iloc[0]
    color = SEQUENCE_COLORS.get(sequence, "blue")

    center_lat = float(group["lat"].median())
    center_lon = float(group["lon"].median())

    fmap = folium.Map(location=[center_lat, center_lon], zoom_start=15, tiles="OpenStreetMap")

    coords = list(zip(group["lat"].tolist(), group["lon"].tolist()))
    folium.PolyLine(coords, color=color, weight=3, tooltip=sequence).add_to(fmap)

    folium.Marker(
        location=[group["lat"].iloc[0], group["lon"].iloc[0]],
        popup=f"{sequence} start: {group['filename'].iloc[0]}",
        icon=folium.Icon(color="green"),
    ).add_to(fmap)

    folium.Marker(
        location=[group["lat"].iloc[-1], group["lon"].iloc[-1]],
        popup=f"{sequence} end: {group['filename'].iloc[-1]}",
        icon=folium.Icon(color="red"),
    ).add_to(fmap)

    # show only nearby tile centers for readability
    if not tile_df.empty:
        lat_min, lat_max = group["lat"].min(), group["lat"].max()
        lon_min, lon_max = group["lon"].min(), group["lon"].max()

        margin_lat = 0.003
        margin_lon = 0.003

        local_tiles = tile_df[
            (tile_df["lat_center"] >= lat_min - margin_lat) &
            (tile_df["lat_center"] <= lat_max + margin_lat) &
            (tile_df["lon_center"] >= lon_min - margin_lon) &
            (tile_df["lon_center"] <= lon_max + margin_lon)
        ]

        sample_tiles = local_tiles.iloc[::max(1, len(local_tiles) // 150)] if len(local_tiles) > 0 else local_tiles

        for _, row in sample_tiles.iterrows():
            folium.CircleMarker(
                location=[row["lat_center"], row["lon_center"]],
                radius=2,
                color="purple",
                fill=True,
                fill_opacity=0.6,
                popup=str(row["filename"]),
            ).add_to(fmap)

    fmap.save(str(output_path))
    return True


def write_all_sequences_folium_map(
    traj_df: pd.DataFrame,
    tile_df: pd.DataFrame,
    output_path: Path,
) -> bool:
    try:
        import folium
    except Exception:
        return False

    ensure_dir(output_path.parent)

    center_lat = float(traj_df["lat"].median())
    center_lon = float(traj_df["lon"].median())

    fmap = folium.Map(location=[center_lat, center_lon], zoom_start=15, tiles="OpenStreetMap")

    for sequence, group in traj_df.groupby("sequence", sort=False):
        color = SEQUENCE_COLORS.get(sequence, "blue")
        coords = list(zip(group["lat"].tolist(), group["lon"].tolist()))

        folium.PolyLine(coords, color=color, weight=3, tooltip=sequence).add_to(fmap)

        folium.Marker(
            location=[group["lat"].iloc[0], group["lon"].iloc[0]],
            popup=f"{sequence} start",
            icon=folium.Icon(color="green"),
        ).add_to(fmap)

        folium.Marker(
            location=[group["lat"].iloc[-1], group["lon"].iloc[-1]],
            popup=f"{sequence} end",
            icon=folium.Icon(color="red"),
        ).add_to(fmap)

    fmap.save(str(output_path))
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build SatLoc UAV reference trajectory and satellite tile coordinate index."
    )
    parser.add_argument("--config", required=True, help="Path to configs/dataset_satloc.yaml")
    args = parser.parse_args()

    config = load_config(args.config)

    raw_data_dir = Path(config["paths"]["raw_data_dir"])
    output_dir = Path(config["paths"]["output_dir"])

    metadata_dir = ensure_dir(output_dir / "metadata")
    trajectories_dir = ensure_dir(output_dir / "trajectories")
    reports_dir = ensure_dir(output_dir / "reports")
    figures_dir = ensure_dir(output_dir / "figures")
    maps_dir = ensure_dir(output_dir / "maps")
    ref_fig_dir = ensure_dir(figures_dir / "reference_trajectories")
    ref_map_dir = ensure_dir(maps_dir / "reference_trajectories")

    uav_df, uav_summary = build_uav_reference_index(raw_data_dir)
    tile_df, tile_summary = build_satellite_tile_index(raw_data_dir)

    uav_index_path = metadata_dir / "uav_frames_index_enriched.csv"
    tile_index_path = metadata_dir / "satellite_tiles_index_enriched.csv"
    ref_traj_path = trajectories_dir / "uav_reference_trajectory.csv"
    summary_path = reports_dir / "satloc_coordinate_summary.json"

    uav_df.to_csv(uav_index_path, index=False)
    uav_df.to_csv(ref_traj_path, index=False)
    tile_df.to_csv(tile_index_path, index=False)

#    xy_fig_path = figures_dir / "uav_reference_trajectory_xy.png"
#    lonlat_fig_path = figures_dir / "uav_reference_trajectory_lonlat.png"
#    map_path = maps_dir / "uav_reference_trajectory_map.html"

#    plot_reference_trajectory_xy(uav_df, xy_fig_path)
#    plot_reference_trajectory_lonlat(uav_df, lonlat_fig_path)
#    folium_written = write_folium_map(uav_df, tile_df, map_path)

    all_lonlat_fig = ref_fig_dir / "all_lonlat.png"
    all_global_enu_fig = ref_fig_dir / "all_global_enu.png"

    plot_all_sequences_lonlat(uav_df, all_lonlat_fig)
    plot_all_sequences_global_enu(uav_df, all_global_enu_fig)

    per_sequence_outputs = []

    for sequence, group in uav_df.groupby("sequence", sort=False):
        lonlat_fig = ref_fig_dir / f"{sequence}_lonlat.png"
        local_enu_fig = ref_fig_dir / f"{sequence}_local_enu.png"
        html_map = ref_map_dir / f"{sequence}_map.html"

        plot_single_sequence_lonlat(group, lonlat_fig)
        plot_single_sequence_local_enu(group, local_enu_fig)
        map_written = write_single_sequence_folium_map(group, tile_df, html_map)

        per_sequence_outputs.append(
            {
                "sequence": sequence,
                "lonlat_figure": str(lonlat_fig),
                "local_enu_figure": str(local_enu_fig),
                "folium_map": str(html_map) if map_written else None,
            }
        )

    all_map_path = ref_map_dir / "all_sequences_map.html"
    all_map_written = write_all_sequences_folium_map(uav_df, tile_df, all_map_path)

    summary = {
        "dataset_name": config.get("dataset", {}).get("name", "satloc"),
        "raw_data_dir": str(raw_data_dir),
        "output_dir": str(output_dir),
        "uav": uav_summary,
        "satellite_tiles": tile_summary,
        "outputs": {
            "uav_frames_index_enriched_csv": str(uav_index_path),
            "satellite_tiles_index_enriched_csv": str(tile_index_path),
            "uav_reference_trajectory_csv": str(ref_traj_path),
            "all_lonlat_png": str(all_lonlat_fig),
            "all_global_enu_png": str(all_global_enu_fig),
            "all_sequences_map_html": str(all_map_path) if all_map_written else None,
            "per_sequence_outputs": per_sequence_outputs,
        },
    }

    save_json(summary, summary_path)

    print("SatLoc coordinate index complete")
    print("--------------------------------")
    print(f"Raw data dir:                  {raw_data_dir}")
    print(f"Output dir:                    {output_dir}")
    print()
    print(f"UAV images:                    {len(uav_df)}")
    print(f"UAV parsed coords:             {(uav_df['parse_status'] == 'ok').sum()}")
    print(f"UAV sequences:                 {uav_df['sequence'].nunique()}")
    print()
    print(f"Satellite tile rows:           {len(tile_df)}")
    print(f"Satellite tile files matched:  {tile_df['tile_exists'].sum()}")
    print()
    print(f"Saved UAV index:               {uav_index_path}")
    print(f"Saved tile index:              {tile_index_path}")
    print(f"Saved UAV reference traj:      {ref_traj_path}")
    print(f"Saved summary:                 {summary_path}")
#    print(f"Saved XY figure:               {xy_fig_path}")
#    print(f"Saved lon/lat figure:          {lonlat_fig_path}")
    print(f"Saved combined lon/lat figure:  {all_lonlat_fig}")
    print(f"Saved combined global ENU fig:  {all_global_enu_fig}")
    if all_map_written:
        print(f"Saved combined map:            {all_map_path}")

#    if folium_written:
#        print(f"Saved map:                     {map_path}")
#    else:
#        print("Folium map skipped:             folium not installed")


if __name__ == "__main__":
    main()