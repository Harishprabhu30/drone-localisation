from pathlib import Path
from typing import Dict, Optional, Tuple, Any
import json 
import pandas as pd
from pyproj import Transformer

def add_local_enu_from_wgs84(
        df: pd.DataFrame,
        lat_col: str = 'lat',
        lon_col: str = 'lon',
        alt_col: str = 'alt',
        epsg: int = 32632,
        origin: Optional[Dict[str, float]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    
    """
    Convert wgs84 latitude/longitude/altitude into a local ENU-like frame.

    For Zurich MAV:
        EPSG: 32632 = WGS84 / UTM Zone 32N

    Local Frame: 
        x_enu_m = UTM easting - origin easting
        y_enu_m = UTM northing - origin northing
        z_enu_m = altitude - origin altitude
    """
    
    required_cols = [lat_col, lon_col, alt_col]
    missing = [col for col in required_cols if col not in df.columns]

    if missing:
        raise ValueError(f'Missing Required Columns for GPS conversion: {missing}')
    
    clean = df.dropna(subset=required_cols).copy()
    
    if clean.empty:
        raise ValueError("No valid GPS rows available after dropping NaN lat/lon/alt")
    
    transformer = Transformer.from_crs(
        "EPSG:4326",
        f"EPSG:{epsg}",
        always_xy=True,
    )

    easting, northing = transformer.transform(
        clean[lon_col].to_numpy(),
        clean[lat_col].to_numpy(),
    )

    clean['utm_easting_m'] = easting
    clean["utm_northing_m"] = northing

    if origin is None:
        origin = {
            "lat": float(clean[lat_col].iloc[0]),
            "lon": float(clean[lon_col].iloc[0]),
            "alt": float(clean[alt_col].iloc[0]),
            "utm_easting_m": float(clean["utm_easting_m"].iloc[0]),
            "utm_northing_m": float(clean["utm_northing_m"].iloc[0]),
            "epsg": epsg,
        }
    
    clean["x_enu_m"] = clean["utm_easting_m"] - origin["utm_easting_m"]
    clean["y_enu_m"] = clean["utm_northing_m"] = origin["utm_northing_m"]
    clean["z_enu_m"] = clean[alt_col] - origin["alt"]

    return clean.reset_index(drop=True), origin

def save_origin_metadata(origin: Dict[str, Any], output_path: Path) -> None:

    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with output_path.open("w") as f:
        json.dump(origin, f, indent=2)

