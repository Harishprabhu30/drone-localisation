from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from pyproj import CRS, Transformer
import rasterio
from rasterio.transform import xy as rasterio_xy


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def parse_uav_filename(path: str | Path) -> Dict[str, Any]:
    """
    SatLoc UAV filename pattern appears like:
        token0@token1@longitude@latitude.png

    Example:
        1@0@112.816130@28.297316.png
    """
    path = Path(path)
    stem = path.stem
    parts = stem.split("@")

    result: Dict[str, Any] = {
        "token0": None,
        "token1": None,
        "lon": None,
        "lat": None,
        "parse_status": "failed",
    }

    if len(parts) < 4:
        result["parse_status"] = "not_enough_at_tokens"
        return result

    try:
        result["token0"] = int(parts[0])
        result["token1"] = int(parts[1])
        result["lon"] = float(parts[2])
        result["lat"] = float(parts[3])

        if not (-180.0 <= result["lon"] <= 180.0):
            result["parse_status"] = "invalid_lon"
            return result

        if not (-90.0 <= result["lat"] <= 90.0):
            result["parse_status"] = "invalid_lat"
            return result

        result["parse_status"] = "ok"
        return result

    except Exception as exc:
        result["parse_status"] = f"parse_error: {exc}"
        return result


def list_images(root: Path) -> List[Path]:
    if not root.exists():
        return []

    return sorted(
        [
            p for p in root.rglob("*")
            if p.is_file()
            and p.suffix.lower() in IMAGE_EXTENSIONS
            and p.name != ".DS_Store"
        ]
    )


def make_utm_crs_from_lonlat(lon: float, lat: float) -> CRS:
    zone = int((lon + 180.0) / 6.0) + 1
    zone = max(1, min(zone, 60))
    epsg = 32600 + zone if lat >= 0 else 32700 + zone
    return CRS.from_epsg(epsg)


def add_enu_columns(
    df: pd.DataFrame,
    lon_col: str = "lon",
    lat_col: str = "lat",
    group_col: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    valid = df[lon_col].notna() & df[lat_col].notna()
    if not valid.any():
        raise ValueError("No valid lon/lat rows found for ENU conversion.")

    lon0 = float(df.loc[valid, lon_col].iloc[0])
    lat0 = float(df.loc[valid, lat_col].iloc[0])

    crs_wgs84 = CRS.from_epsg(4326)
    crs_utm = make_utm_crs_from_lonlat(lon0, lat0)
    transformer = Transformer.from_crs(crs_wgs84, crs_utm, always_xy=True)

    x_all, y_all = transformer.transform(df[lon_col].astype(float).values, df[lat_col].astype(float).values)

    df = df.copy()
    df["utm_x_m"] = x_all
    df["utm_y_m"] = y_all

    origin_x = float(df.loc[valid, "utm_x_m"].iloc[0])
    origin_y = float(df.loc[valid, "utm_y_m"].iloc[0])

    df["x_enu_global_m"] = df["utm_x_m"] - origin_x
    df["y_enu_global_m"] = df["utm_y_m"] - origin_y

    if group_col is not None:
        df["x_enu_m"] = 0.0
        df["y_enu_m"] = 0.0

        for group_value, group_df in df.groupby(group_col, sort=False):
            group_valid = group_df[lon_col].notna() & group_df[lat_col].notna()
            if not group_valid.any():
                continue

            first_idx = group_df.loc[group_valid].index[0]
            gx0 = float(df.loc[first_idx, "utm_x_m"])
            gy0 = float(df.loc[first_idx, "utm_y_m"])

            idxs = group_df.index
            df.loc[idxs, "x_enu_m"] = df.loc[idxs, "utm_x_m"] - gx0
            df.loc[idxs, "y_enu_m"] = df.loc[idxs, "utm_y_m"] - gy0
    else:
        df["x_enu_m"] = df["x_enu_global_m"]
        df["y_enu_m"] = df["y_enu_global_m"]

    origin = {
        "origin_lon": lon0,
        "origin_lat": lat0,
        "origin_utm_x_m": origin_x,
        "origin_utm_y_m": origin_y,
        "utm_crs": crs_utm.to_string(),
    }

    return df, origin


def build_uav_reference_index(raw_data_dir: Path) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    uav_dir = raw_data_dir / "UAV Data"
    sequence_dirs = sorted([p for p in uav_dir.glob("traj*") if p.is_dir()])

    records: List[Dict[str, Any]] = []

    for sequence_dir in sequence_dirs:
        image_paths = list_images(sequence_dir)

        for image_path in image_paths:
            parsed = parse_uav_filename(image_path)

            records.append(
                {
                    "sequence": sequence_dir.name,
                    "image_path": str(image_path),
                    "image_path_relative": str(image_path.relative_to(raw_data_dir)),
                    "filename": image_path.name,
                    "token0_id": parsed["token0"],
                    "token1_order": parsed["token1"],
                    "lon": parsed["lon"],
                    "lat": parsed["lat"],
                    "parse_status": parsed["parse_status"],
                }
            )

    df = pd.DataFrame(records)

    if df.empty:
        summary = {
            "sequence_count": len(sequence_dirs),
            "total_uav_images": 0,
            "parsed_uav_images": 0,
            "warning": "No UAV images found.",
        }
        return df, summary

    # df = df.sort_values(["sequence", "token1_order", "token0_id"], na_position="last").reset_index(drop=True) - produces mesh/intermingling like traj path on plot and folium
    df = df.sort_values(["sequence", "token0_id", "token1_order"], na_position="last").reset_index(drop=True)

    df["frame_index_in_sequence"] = df.groupby("sequence").cumcount()
    df["global_frame_index"] = range(len(df))

    df, origin = add_enu_columns(df, lon_col="lon", lat_col="lat", group_col="sequence")

    summary = {
        "sequence_count": int(df["sequence"].nunique()),
        "total_uav_images": int(len(df)),
        "parsed_uav_images": int((df["parse_status"] == "ok").sum()),
        "origin": origin,
        "sequences": [],
    }

    for sequence, group in df.groupby("sequence", sort=False):
        dx = group["x_enu_m"].diff()
        dy = group["y_enu_m"].diff()
        step = (dx.pow(2) + dy.pow(2)).pow(0.5)
        path_length = float(step.fillna(0).sum())

        summary["sequences"].append(
            {
                "sequence": sequence,
                "frames": int(len(group)),
                "lon_min": float(group["lon"].min()),
                "lon_max": float(group["lon"].max()),
                "lat_min": float(group["lat"].min()),
                "lat_max": float(group["lat"].max()),
                "x_range_m": [
                    float(group["x_enu_m"].min()),
                    float(group["x_enu_m"].max()),
                ],
                "y_range_m": [
                    float(group["y_enu_m"].min()),
                    float(group["y_enu_m"].max()),
                ],
                "path_length_m": path_length,
                "first_filename": str(group["filename"].iloc[0]),
                "last_filename": str(group["filename"].iloc[-1]),
            }
        )

    return df, summary


def pixel_bbox_to_lonlat_bbox(
    dataset: rasterio.io.DatasetReader,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> Dict[str, float]:
    """
    ref_sample.csv columns are named Loc1 lon/lat and Loc2 lon/lat,
    but values are pixel coordinates in the large GeoTIFF/DOM.

    x = column pixel
    y = row pixel
    """
    transform = dataset.transform

    lon_tl, lat_tl = rasterio_xy(transform, y1, x1, offset="center")
    lon_br, lat_br = rasterio_xy(transform, y2, x2, offset="center")

    xc = (x1 + x2) / 2.0
    yc = (y1 + y2) / 2.0
    lon_c, lat_c = rasterio_xy(transform, yc, xc, offset="center")

    return {
        "tile_x1_px": float(x1),
        "tile_y1_px": float(y1),
        "tile_x2_px": float(x2),
        "tile_y2_px": float(y2),
        "lon_tl": float(lon_tl),
        "lat_tl": float(lat_tl),
        "lon_br": float(lon_br),
        "lat_br": float(lat_br),
        "lon_center": float(lon_c),
        "lat_center": float(lat_c),
    }


def build_satellite_tile_index(raw_data_dir: Path) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    sat_dir = raw_data_dir / "Satellite Data"
    ref_csv_path = sat_dir / "ref_sample.csv"
    geotiff_path = sat_dir / "CS_Area1.tif"
    tile_dir = sat_dir / "sat_image_ref"

    ref_df = pd.read_csv(ref_csv_path)

    tile_paths = list_images(tile_dir)
    tile_path_by_name = {p.name: p for p in tile_paths}

    records: List[Dict[str, Any]] = []

    with rasterio.open(geotiff_path) as dataset:
        for _, row in ref_df.iterrows():
            name = str(row["Name"])
            tile_path = tile_path_by_name.get(name)

            x1 = float(row["Loc1 lon"])
            y1 = float(row["Loc1 lat"])
            x2 = float(row["Loc2 lon"])
            y2 = float(row["Loc2 lat"])

            geo = pixel_bbox_to_lonlat_bbox(dataset, x1, y1, x2, y2)

            record = {
                "tile_index": int(row["Index"]),
                "filename": name,
                "tile_path": str(tile_path) if tile_path is not None else None,
                "tile_path_relative": str(tile_path.relative_to(raw_data_dir)) if tile_path is not None else None,
                "ref_rel_path": str(row["Rel path"]),
                "tile_exists": tile_path is not None,
                **geo,
            }
            records.append(record)

        geotiff_summary = {
            "path": str(geotiff_path),
            "width": int(dataset.width),
            "height": int(dataset.height),
            "crs": str(dataset.crs),
            "bounds": {
                "left": float(dataset.bounds.left),
                "bottom": float(dataset.bounds.bottom),
                "right": float(dataset.bounds.right),
                "top": float(dataset.bounds.top),
            },
            "transform": tuple(dataset.transform),
        }

    df = pd.DataFrame(records)

    if not df.empty:
        df, origin = add_enu_columns(df, lon_col="lon_center", lat_col="lat_center", group_col=None)
    else:
        origin = None

    summary = {
        "ref_csv": str(ref_csv_path),
        "geotiff": geotiff_summary,
        "tile_dir": str(tile_dir),
        "ref_rows": int(len(ref_df)),
        "tile_files_found": int(len(tile_paths)),
        "tile_index_rows": int(len(df)),
        "tile_files_matched": int(df["tile_exists"].sum()) if not df.empty else 0,
        "origin": origin,
    }

    return df, summary


def save_json(data: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)