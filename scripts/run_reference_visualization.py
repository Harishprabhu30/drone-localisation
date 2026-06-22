import argparse
import pandas as pd
from pathlib import Path

from uavloc.utils.config import load_config
from uavloc.visualization.trajectory_plot import plot_xy_trajectory, plot_altitude_profile, plot_speed_profile
from uavloc.visualization.folium_map import build_trajectory_map

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to dataset config yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = Path(config["paths"]["output_dir"])

    reference_csv = output_dir / "trajectories" / "reference_trajectory.csv"
    
    if not reference_csv.exists():
        raise FileNotFoundError(f"Reference trajectory not found: {reference_csv}" \
        f"Run build_reference_trajectory.py first.")
    
    df = pd.read_csv(reference_csv)

    figures_dir = output_dir / "figures"
    maps_dir = output_dir / "maps"

    xy_path = figures_dir / "trajectory_xy.png"
    alt_path = figures_dir / "altitude_profile.png"
    speed_path = figures_dir / "speed_profile.png"
    map_path = maps_dir / "trajectory_map.html"

    plot_xy_trajectory(df, xy_path)
    plot_altitude_profile(df, alt_path)
    plot_speed_profile(df, speed_path)
    build_trajectory_map(df, map_path)

    print("Reference trajectory visualization generated")
    print("------------------------------------------")
    print(f"XY plot: {xy_path}")
    print(f"Altitude plot: {alt_path}")
    print(f"Speed plot: {speed_path}")
    print(f"Map HTML: {map_path}")


if __name__ == "__main__":
    main()
