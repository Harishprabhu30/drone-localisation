from pathlib import Path
import folium
import pandas as pd

def build_trajectory_map(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not {"lat", "lon"}.issubset(df.columns):
        raise ValueError("Reference dataframe must contain 'lat' and 'lon' columns.")
    
    center_lat = float(df['lat'].iloc[0])
    center_lon = float(df['lon'].iloc[0])

    m = folium.Map(location=[center_lat, center_lon], zoom_start=18, control_scale=True)

    coords = list(zip(df["lat"], df["lon"]))
    folium.PolyLine(coords, weight=3, opacity=0.9).add_to(m)

    folium.Marker(
        [df["lat"].iloc[0], df["lon"].iloc[0]],
        popup="start",
        tooltip="Start",
    ).add_to(m)

    folium.Marker(
        [df["lat"].iloc[-1], df["lon"].iloc[-1]],
        popup="End",
        tooltip="End",
    ).add_to(m)

    m.save(str(output_path))