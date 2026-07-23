'''
Command Executed:

export PYTHONPATH=$PWD/src

python scripts/villoc/s8_10b_uav_tile_oracle_audit.py \
  --config configs/dataset_villoc_90deg.yaml \
  --src-tif data/processed/villoc/90_deg/maps/ort10lt_2024_2026/ort10lt_2024_2026_aoi300m.tif \
  --uav-index-csv outputs/villoc/90_deg/metadata/s8_5_uav_frames_index_v_1fps.csv \
  --trajectory-csv outputs/villoc/90_deg/trajectories/s8_3_reference_trajectory_V_1fps.csv \
  --trajectory-crs EPSG:3346 \
  --variant "512_s256:outputs/villoc/90_deg/metadata/s8_9_satellite_tile_index_512_s256.csv" \
  --variant "1024_s512:outputs/villoc/90_deg/metadata/s8_9_satellite_tile_index_1024_s512.csv" \
  --variant "1024_s256:outputs/villoc/90_deg/metadata/s8_9_satellite_tile_index_1024_s256.csv"

'''

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import yaml
from pyproj import CRS, Transformer


COLUMN_ALIASES: dict[str, list[str]] = {
    "query_id": [
        "query_id",
        "token0_id",
        "token_id",
        "uav_id",
        "id",
    ],
    "token0_id": [
        "token0_id",
        "query_id",
        "token_id",
    ],
    "frame_index": [
        "zero_based_frame_index",
        "frame_index",
        "frame_idx",
        "frame_id",
        "frame_number",
        "frame_no",
        "source_frame_index",
        "video_frame_index",
        "frame_cnt",
    ],
    "timestamp_s": [
        "timestamp_s",
        "time_s",
        "video_time_s",
        "elapsed_s",
        "timestamp",
    ],
    "latitude": [
        "latitude",
        "lat",
        "gps_lat",
        "gps_latitude",
        "srt_latitude",
    ],
    "longitude": [
        "longitude",
        "lon",
        "lng",
        "gps_lon",
        "gps_longitude",
        "srt_longitude",
    ],
    "easting": [
        "easting",
        "x",
        "x_m",
        "easting_m",
        "epsg3346_x",
        "lks94_x",
        "ref_easting",
        "reference_easting",
    ],
    "northing": [
        "northing",
        "y",
        "y_m",
        "northing_m",
        "epsg3346_y",
        "lks94_y",
        "ref_northing",
        "reference_northing",
    ],
    "filename": [
        "filename",
        "image_filename",
        "frame_filename",
        "uav_filename",
        "file_name",
    ],
    "image_path": [
        "image_path",
        "frame_path",
        "uav_image_path",
        "path",
        "filepath",
        "file_path",
    ],
    "yaw_deg": [
        "yaw_deg",
        "yaw",
        "gimbal_yaw_deg",
        "aircraft_yaw_deg",
    ],
    "pitch_deg": [
        "pitch_deg",
        "pitch",
        "gimbal_pitch_deg",
    ],
    "relative_altitude_m": [
        "relative_altitude_m",
        "relative_alt_m",
        "rel_alt_m",
        "altitude_relative_m",
        "rel_alt",
    ],
}


VARIANT_REQUIRED_COLUMNS = {
    "tile_id",
    "tile_path",
    "grid_row",
    "grid_col",
    "left_easting",
    "bottom_northing",
    "right_easting",
    "top_northing",
    "center_easting",
    "center_northing",
    "is_right_edge_tile",
    "is_bottom_edge_tile",
}


def find_column(
    dataframe: pd.DataFrame,
    logical_name: str,
    required: bool = True,
) -> str | None:
    aliases = COLUMN_ALIASES[logical_name]

    exact_lookup = {
        str(column).strip().lower(): str(column)
        for column in dataframe.columns
    }

    for alias in aliases:
        if alias.lower() in exact_lookup:
            return exact_lookup[alias.lower()]

    if required:
        raise KeyError(
            f"Could not locate a column for '{logical_name}'. "
            f"Tried aliases: {aliases}. "
            f"Available columns: {list(dataframe.columns)}"
        )

    return None


def first_existing_column(
    dataframe: pd.DataFrame,
    names: Iterable[str],
) -> str | None:
    lower_lookup = {
        str(column).strip().lower(): str(column)
        for column in dataframe.columns
    }

    for name in names:
        if name.lower() in lower_lookup:
            return lower_lookup[name.lower()]

    return None


def normalize_boolean(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)

    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
        "yes": True,
        "no": False,
    }

    result = (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map(mapping)
    )

    return result.fillna(False).astype(bool)


def safe_int(value: Any, fallback: int) -> int:
    try:
        if pd.isna(value):
            return fallback

        return int(round(float(value)))
    except (TypeError, ValueError):
        return fallback


def canonical_query_filename(
    token0_id: int,
    frame_index: int,
    source_frame_cnt: int,
    latitude: float,
    longitude: float,
    extension: str,
) -> str:
    extension = extension.lower().lstrip(".")

    if not extension:
        extension = "jpg"

    return (
        f"token{token0_id:06d}_"
        f"frame{frame_index:06d}_"
        f"src{source_frame_cnt:06d}_"
        f"lat{latitude:.6f}_"
        f"lon{longitude:.6f}."
        f"{extension}"
    )


def infer_image_extension(
    filename: str | None,
    image_path: str | None,
) -> str:
    for candidate in [filename, image_path]:
        if candidate:
            suffix = Path(str(candidate)).suffix.lower().lstrip(".")

            if suffix in {"jpg", "jpeg", "png", "tif", "tiff", "webp"}:
                return suffix

    return "jpg"


def read_uav_records(
    uav_index_csv: Path,
    trajectory_csv: Path,
    trajectory_crs: str,
) -> pd.DataFrame:
    uav_index = pd.read_csv(uav_index_csv)
    trajectory = pd.read_csv(trajectory_csv)

    print("UAV index columns:")
    print(list(uav_index.columns))
    print()
    print("Trajectory columns:")
    print(list(trajectory.columns))
    print()

    trajectory_frame_col = find_column(
        trajectory,
        "frame_index",
        required=False,
    )

    index_frame_col = find_column(
        uav_index,
        "frame_index",
        required=False,
    )

    trajectory_token_col = find_column(
        trajectory,
        "token0_id",
        required=False,
    )

    index_token_col = find_column(
        uav_index,
        "token0_id",
        required=False,
    )

    merge_left: str | None = None
    merge_right: str | None = None
    merge_key_name: str | None = None

    index_sample_col = first_existing_column(
        uav_index,
        ["sample_id"],
    )

    trajectory_sample_col = first_existing_column(
        trajectory,
        ["sample_id"],
    )

    if (
        index_sample_col is not None
        and trajectory_sample_col is not None
    ):
        merge_left = index_sample_col
        merge_right = trajectory_sample_col
        merge_key_name = "sample_id"

    elif (
        index_token_col is not None
        and trajectory_token_col is not None
    ):
        merge_left = index_token_col
        merge_right = trajectory_token_col
        merge_key_name = "token0_id"

    elif (
        index_frame_col is not None
        and trajectory_frame_col is not None
    ):
        merge_left = index_frame_col
        merge_right = trajectory_frame_col
        merge_key_name = "frame_index"

    if merge_left is not None and merge_right is not None:
        merged = uav_index.merge(
            trajectory,
            left_on=merge_left,
            right_on=merge_right,
            how="inner",
            suffixes=("_index", "_trajectory"),
            validate="one_to_one",
        )

        print(
            f"Merged UAV index and trajectory using {merge_key_name}: "
            f"{merge_left} ↔ {merge_right}"
        )

        if len(merged) != len(uav_index):
            raise RuntimeError(
                f"UAV index has {len(uav_index)} rows but merge produced "
                f"{len(merged)} rows. Check the frame/token alignment."
            )
    else:
        if len(uav_index) != len(trajectory):
            raise RuntimeError(
                "Could not identify a common token/frame column and the "
                "two CSV files have different row counts."
            )

        merged = pd.concat(
            [
                uav_index.reset_index(drop=True).add_suffix("_index"),
                trajectory.reset_index(drop=True).add_suffix("_trajectory"),
            ],
            axis=1,
        )

        print(
            "No explicit common token/frame key found. "
            "Merged by row order after confirming equal row counts."
        )

    def locate_merged_column(
        logical_name: str,
        required: bool = True,
    ) -> str | None:
        aliases = COLUMN_ALIASES[logical_name]

        candidates: list[str] = []

        for alias in aliases:
            candidates.extend(
                [
                    alias,
                    f"{alias}_trajectory",
                    f"{alias}_index",
                ]
            )

        return first_existing_column(merged, candidates) or (
            find_column(merged, logical_name, required=required)
            if required
            else None
        )

    lat_col = locate_merged_column("latitude")
    lon_col = locate_merged_column("longitude")

    east_col = locate_merged_column("easting", required=False)
    north_col = locate_merged_column("northing", required=False)

    token_col = locate_merged_column("token0_id", required=False)
    query_col = locate_merged_column("query_id", required=False)
    frame_col = locate_merged_column(
        "frame_index",
        required=False,
    )

    source_frame_col = first_existing_column(
        merged,
        [
            "source_frame_cnt",
            "source_frame_cnt_index",
            "frame_cnt_trajectory",
            "frame_cnt",
        ],
    )

    filename_col = locate_merged_column("filename", required=False)
    path_col = locate_merged_column("image_path", required=False)
    timestamp_col = locate_merged_column("timestamp_s", required=False)
    yaw_col = locate_merged_column("yaw_deg", required=False)
    pitch_col = locate_merged_column("pitch_deg", required=False)
    rel_alt_col = locate_merged_column(
        "relative_altitude_m",
        required=False,
    )

    records = pd.DataFrame()

    records["latitude"] = pd.to_numeric(
        merged[lat_col],
        errors="raise",
    )

    records["longitude"] = pd.to_numeric(
        merged[lon_col],
        errors="raise",
    )

    if east_col is not None and north_col is not None:
        records["easting"] = pd.to_numeric(
            merged[east_col],
            errors="raise",
        )

        records["northing"] = pd.to_numeric(
            merged[north_col],
            errors="raise",
        )

        coordinate_source = "existing_projected_columns"
    else:
        transformer = Transformer.from_crs(
            CRS.from_epsg(4326),
            CRS.from_user_input(trajectory_crs),
            always_xy=True,
        )

        eastings, northings = transformer.transform(
            records["longitude"].to_numpy(),
            records["latitude"].to_numpy(),
        )

        records["easting"] = eastings
        records["northing"] = northings
        coordinate_source = "transformed_from_lat_lon"

    row_numbers = np.arange(1, len(records) + 1)

    if token_col is not None:
        records["token0_id"] = pd.to_numeric(
            merged[token_col],
            errors="coerce",
        ).fillna(pd.Series(row_numbers, index=merged.index)).astype(int)
    else:
        records["token0_id"] = row_numbers

    if query_col is not None:
        records["query_id"] = pd.to_numeric(
            merged[query_col],
            errors="coerce",
        ).fillna(records["token0_id"]).astype(int)
    else:
        records["query_id"] = records["token0_id"].astype(int)

    if frame_col is not None:
        records["frame_index"] = [
            safe_int(value, fallback)
            for value, fallback in zip(
                merged[frame_col],
                row_numbers,
            )
        ]
    else:
        records["frame_index"] = row_numbers

    if source_frame_col is not None:
        records["source_frame_cnt"] = [
            safe_int(value, fallback)
            for value, fallback in zip(
                merged[source_frame_col],
                records["frame_index"],
            )
        ]
    else:
        records["source_frame_cnt"] = (
            records["frame_index"]
        )

    if filename_col is not None:
        records["original_filename"] = (
            merged[filename_col]
            .fillna("")
            .astype(str)
        )
    else:
        records["original_filename"] = ""

    if path_col is not None:
        records["image_path"] = (
            merged[path_col]
            .fillna("")
            .astype(str)
        )
    else:
        records["image_path"] = ""

    missing_original_name = (
        records["original_filename"]
        .fillna("")
        .astype(str)
        .str.strip()
        .isin(["", "nan", "None"])
    )

    records.loc[
        missing_original_name,
        "original_filename",
    ] = records.loc[
        missing_original_name,
        "image_path",
    ].map(
        lambda value: (
            Path(str(value)).name
            if str(value).strip()
            else ""
        )
    )

    if timestamp_col is not None:
        records["timestamp_s"] = pd.to_numeric(
            merged[timestamp_col],
            errors="coerce",
        )
    else:
        records["timestamp_s"] = np.nan

    if yaw_col is not None:
        records["yaw_deg"] = pd.to_numeric(
            merged[yaw_col],
            errors="coerce",
        )
    else:
        records["yaw_deg"] = np.nan

    if pitch_col is not None:
        records["pitch_deg"] = pd.to_numeric(
            merged[pitch_col],
            errors="coerce",
        )
    else:
        records["pitch_deg"] = np.nan

    if rel_alt_col is not None:
        records["relative_altitude_m"] = pd.to_numeric(
            merged[rel_alt_col],
            errors="coerce",
        )
    else:
        records["relative_altitude_m"] = np.nan

    canonical_names: list[str] = []

    for row in records.itertuples(index=False):
        extension = infer_image_extension(
            filename=row.original_filename,
            image_path=row.image_path,
        )

        canonical_names.append(
            canonical_query_filename(
                token0_id=int(row.token0_id),
                frame_index=int(row.frame_index),
                source_frame_cnt=int(
                    row.source_frame_cnt
                ),
                latitude=float(row.latitude),
                longitude=float(row.longitude),
                extension=extension,
            )
        )

    records["canonical_query_filename"] = canonical_names
    records["coordinate_source"] = coordinate_source
    records["trajectory_crs"] = trajectory_crs

    if records["query_id"].duplicated().any():
        duplicates = records.loc[
            records["query_id"].duplicated(False),
            "query_id",
        ].tolist()

        raise RuntimeError(
            f"Duplicate query IDs found: {duplicates[:20]}"
        )

    if records["canonical_query_filename"].duplicated().any():
        duplicates = records.loc[
            records["canonical_query_filename"].duplicated(False),
            "canonical_query_filename",
        ].tolist()

        raise RuntimeError(
            "Canonical query filenames are not unique: "
            f"{duplicates[:20]}"
        )

    return records


def read_tile_index(path: Path) -> pd.DataFrame:
    tiles = pd.read_csv(path)

    missing = sorted(
        VARIANT_REQUIRED_COLUMNS - set(tiles.columns)
    )

    if missing:
        raise KeyError(
            f"Tile index {path} is missing columns: {missing}"
        )

    numeric_columns = [
        "grid_row",
        "grid_col",
        "left_easting",
        "bottom_northing",
        "right_easting",
        "top_northing",
        "center_easting",
        "center_northing",
    ]

    for column in numeric_columns:
        tiles[column] = pd.to_numeric(
            tiles[column],
            errors="raise",
        )

    tiles["is_right_edge_tile"] = normalize_boolean(
        tiles["is_right_edge_tile"]
    )

    tiles["is_bottom_edge_tile"] = normalize_boolean(
        tiles["is_bottom_edge_tile"]
    )

    tiles["is_anchored_edge_tile"] = (
        tiles["is_right_edge_tile"]
        | tiles["is_bottom_edge_tile"]
    )

    return tiles


def serialize_ids(values: list[str]) -> str:
    return "|".join(values)


def serialize_distances(values: list[float]) -> str:
    return "|".join(f"{value:.6f}" for value in values)


def point_inside_bounds(
    x: float,
    y: float,
    left: float,
    bottom: float,
    right: float,
    top: float,
    tolerance: float = 1e-8,
) -> bool:
    return (
        left - tolerance <= x <= right + tolerance
        and bottom - tolerance <= y <= top + tolerance
    )


def audit_variant(
    variant_name: str,
    tiles: pd.DataFrame,
    uav_records: pd.DataFrame,
    raster_bounds: rasterio.coords.BoundingBox,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    tile_left = tiles["left_easting"].to_numpy(dtype=float)
    tile_bottom = tiles["bottom_northing"].to_numpy(dtype=float)
    tile_right = tiles["right_easting"].to_numpy(dtype=float)
    tile_top = tiles["top_northing"].to_numpy(dtype=float)

    tile_center_x = tiles["center_easting"].to_numpy(dtype=float)
    tile_center_y = tiles["center_northing"].to_numpy(dtype=float)

    tile_ids = tiles["tile_id"].astype(str).to_numpy()
    tile_paths = tiles["tile_path"].astype(str).to_numpy()
    tile_rows = tiles["grid_row"].astype(int).to_numpy()
    tile_cols = tiles["grid_col"].astype(int).to_numpy()

    right_edge_flags = tiles[
        "is_right_edge_tile"
    ].to_numpy(dtype=bool)

    bottom_edge_flags = tiles[
        "is_bottom_edge_tile"
    ].to_numpy(dtype=bool)

    anchored_flags = tiles[
        "is_anchored_edge_tile"
    ].to_numpy(dtype=bool)

    result_rows: list[dict[str, Any]] = []

    for record in uav_records.itertuples(index=False):
        x = float(record.easting)
        y = float(record.northing)

        inside_crop = point_inside_bounds(
            x=x,
            y=y,
            left=float(raster_bounds.left),
            bottom=float(raster_bounds.bottom),
            right=float(raster_bounds.right),
            top=float(raster_bounds.top),
        )

        containing_mask = (
            (tile_left <= x)
            & (x <= tile_right)
            & (tile_bottom <= y)
            & (y <= tile_top)
        )

        containing_indices = np.flatnonzero(containing_mask)

        distances = np.hypot(
            tile_center_x - x,
            tile_center_y - y,
        )

        distance_order = np.argsort(distances)

        nearest_index = int(distance_order[0])

        containing_distance_order = sorted(
            containing_indices.tolist(),
            key=lambda index: (
                distances[index],
                tile_ids[index],
            ),
        )

        oracle_ids = [
            str(tile_ids[index])
            for index in containing_distance_order
        ]

        oracle_paths = [
            str(tile_paths[index])
            for index in containing_distance_order
        ]

        oracle_center_errors = [
            float(distances[index])
            for index in containing_distance_order
        ]

        oracle_grid_cells = [
            f"r{tile_rows[index]:02d}c{tile_cols[index]:02d}"
            for index in containing_distance_order
        ]

        oracle_contains_right_edge = any(
            bool(right_edge_flags[index])
            for index in containing_indices
        )

        oracle_contains_bottom_edge = any(
            bool(bottom_edge_flags[index])
            for index in containing_indices
        )

        oracle_contains_anchored_edge = any(
            bool(anchored_flags[index])
            for index in containing_indices
        )

        nearest_is_oracle = bool(
            containing_mask[nearest_index]
        )

        nearest_five = distance_order[:5]

        result_rows.append(
            {
                "variant": variant_name,
                "query_id": int(record.query_id),
                "token0_id": int(record.token0_id),
                "frame_index": int(record.frame_index),
                "canonical_query_filename": (
                    record.canonical_query_filename
                ),
                "original_filename": record.original_filename,
                "image_path": record.image_path,
                "timestamp_s": record.timestamp_s,
                "latitude": float(record.latitude),
                "longitude": float(record.longitude),
                "easting": x,
                "northing": y,
                "yaw_deg": record.yaw_deg,
                "pitch_deg": record.pitch_deg,
                "relative_altitude_m": (
                    record.relative_altitude_m
                ),
                "inside_map_crop": inside_crop,
                "oracle_tile_count": len(
                    containing_indices
                ),
                "has_oracle_tile": (
                    len(containing_indices) > 0
                ),
                "oracle_tile_ids": serialize_ids(
                    oracle_ids
                ),
                "oracle_tile_paths": serialize_ids(
                    oracle_paths
                ),
                "oracle_grid_cells": serialize_ids(
                    oracle_grid_cells
                ),
                "oracle_center_errors_m": (
                    serialize_distances(
                        oracle_center_errors
                    )
                ),
                "best_oracle_tile_id": (
                    oracle_ids[0]
                    if oracle_ids
                    else ""
                ),
                "best_oracle_center_error_m": (
                    oracle_center_errors[0]
                    if oracle_center_errors
                    else np.nan
                ),
                "nearest_tile_id": str(
                    tile_ids[nearest_index]
                ),
                "nearest_tile_path": str(
                    tile_paths[nearest_index]
                ),
                "nearest_grid_row": int(
                    tile_rows[nearest_index]
                ),
                "nearest_grid_col": int(
                    tile_cols[nearest_index]
                ),
                "nearest_center_easting": float(
                    tile_center_x[nearest_index]
                ),
                "nearest_center_northing": float(
                    tile_center_y[nearest_index]
                ),
                "nearest_center_error_m": float(
                    distances[nearest_index]
                ),
                "nearest_tile_is_oracle": (
                    nearest_is_oracle
                ),
                "nearest_tile_is_right_edge": bool(
                    right_edge_flags[nearest_index]
                ),
                "nearest_tile_is_bottom_edge": bool(
                    bottom_edge_flags[nearest_index]
                ),
                "nearest_tile_is_anchored_edge": bool(
                    anchored_flags[nearest_index]
                ),
                "oracle_contains_right_edge_tile": (
                    oracle_contains_right_edge
                ),
                "oracle_contains_bottom_edge_tile": (
                    oracle_contains_bottom_edge
                ),
                "oracle_contains_anchored_edge_tile": (
                    oracle_contains_anchored_edge
                ),
                "geometric_rank1_tile_id": str(
                    tile_ids[distance_order[0]]
                ),
                "geometric_rank2_tile_id": str(
                    tile_ids[distance_order[1]]
                ),
                "geometric_rank3_tile_id": str(
                    tile_ids[distance_order[2]]
                ),
                "geometric_rank4_tile_id": str(
                    tile_ids[distance_order[3]]
                ),
                "geometric_rank5_tile_id": str(
                    tile_ids[distance_order[4]]
                ),
                "geometric_rank1_error_m": float(
                    distances[distance_order[0]]
                ),
                "geometric_rank2_error_m": float(
                    distances[distance_order[1]]
                ),
                "geometric_rank3_error_m": float(
                    distances[distance_order[2]]
                ),
                "geometric_rank4_error_m": float(
                    distances[distance_order[3]]
                ),
                "geometric_rank5_error_m": float(
                    distances[distance_order[4]]
                ),
            }
        )

    result = pd.DataFrame(result_rows)

    nearest_errors = result[
        "nearest_center_error_m"
    ].to_numpy(dtype=float)

    oracle_counts = result[
        "oracle_tile_count"
    ].to_numpy(dtype=int)

    coverage_failures = int(
        (~result["inside_map_crop"]).sum()
    )

    oracle_failures = int(
        (~result["has_oracle_tile"]).sum()
    )

    nearest_non_oracle_count = int(
        (~result["nearest_tile_is_oracle"]).sum()
    )

    status = (
        "PASS_UAV_TILE_ORACLE_COVERAGE"
        if (
            coverage_failures == 0
            and oracle_failures == 0
            and nearest_non_oracle_count == 0
        )
        else "FAIL_UAV_TILE_ORACLE_COVERAGE"
    )

    summary = {
        "variant": variant_name,
        "status": status,
        "uav_query_count": int(len(result)),
        "tile_count": int(len(tiles)),
        "queries_inside_map_crop": int(
            result["inside_map_crop"].sum()
        ),
        "queries_outside_map_crop": coverage_failures,
        "queries_with_oracle_tile": int(
            result["has_oracle_tile"].sum()
        ),
        "queries_without_oracle_tile": oracle_failures,
        "nearest_tile_not_oracle_count": (
            nearest_non_oracle_count
        ),
        "oracle_tile_count": {
            "minimum": int(np.min(oracle_counts)),
            "mean": float(np.mean(oracle_counts)),
            "median": float(np.median(oracle_counts)),
            "maximum": int(np.max(oracle_counts)),
            "p05": float(np.percentile(oracle_counts, 5)),
            "p95": float(np.percentile(oracle_counts, 95)),
        },
        "nearest_center_error_m": {
            "minimum": float(np.min(nearest_errors)),
            "mean": float(np.mean(nearest_errors)),
            "median": float(np.median(nearest_errors)),
            "p95": float(np.percentile(nearest_errors, 95)),
            "maximum": float(np.max(nearest_errors)),
        },
        "queries_with_anchored_edge_oracle": int(
            result[
                "oracle_contains_anchored_edge_tile"
            ].sum()
        ),
        "queries_with_right_edge_oracle": int(
            result[
                "oracle_contains_right_edge_tile"
            ].sum()
        ),
        "queries_with_bottom_edge_oracle": int(
            result[
                "oracle_contains_bottom_edge_tile"
            ].sum()
        ),
        "nearest_tile_anchored_edge_count": int(
            result[
                "nearest_tile_is_anchored_edge"
            ].sum()
        ),
    }

    return result, summary


def plot_variant(
    variant_name: str,
    tiles: pd.DataFrame,
    result: pd.DataFrame,
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(11, 13))

    for tile in tiles.itertuples(index=False):
        width = float(
            tile.right_easting - tile.left_easting
        )

        height = float(
            tile.top_northing - tile.bottom_northing
        )

        rectangle = plt.Rectangle(
            (
                float(tile.left_easting),
                float(tile.bottom_northing),
            ),
            width,
            height,
            fill=False,
            linewidth=0.35,
            alpha=0.25,
        )

        axis.add_patch(rectangle)

    axis.plot(
        result["easting"],
        result["northing"],
        linewidth=1.5,
        marker="o",
        markersize=2.5,
        label="UAV reference trajectory",
    )

    axis.scatter(
        result["easting"].iloc[0],
        result["northing"].iloc[0],
        marker="s",
        s=60,
        label="Start",
    )

    axis.scatter(
        result["easting"].iloc[-1],
        result["northing"].iloc[-1],
        marker="X",
        s=70,
        label="End",
    )

    axis.set_title(
        f"S8.10B UAV trajectory and tile coverage — "
        f"{variant_name}"
    )

    axis.set_xlabel("Easting — EPSG:3346 (m)")
    axis.set_ylabel("Northing — EPSG:3346 (m)")
    axis.set_aspect("equal", adjustable="box")
    axis.grid(True, linewidth=0.4, alpha=0.35)
    axis.legend()
    figure.tight_layout()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        output_path,
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", required=True)
    parser.add_argument("--src-tif", required=True)
    parser.add_argument("--uav-index-csv", required=True)
    parser.add_argument("--trajectory-csv", required=True)

    parser.add_argument(
        "--trajectory-crs",
        default="EPSG:3346",
    )

    parser.add_argument(
        "--variant",
        action="append",
        required=True,
        help="NAME:TILE_INDEX_CSV",
    )

    args = parser.parse_args()

    config_path = Path(args.config)
    src_tif = Path(args.src_tif)
    uav_index_csv = Path(args.uav_index_csv)
    trajectory_csv = Path(args.trajectory_csv)

    for required_path in [
        config_path,
        src_tif,
        uav_index_csv,
        trajectory_csv,
    ]:
        if not required_path.exists():
            raise FileNotFoundError(required_path)

    config = yaml.safe_load(
        config_path.read_text()
    )

    output_root = Path(
        config["dataset"]["output_root"]
    )

    metadata_dir = output_root / "metadata"
    reports_dir = output_root / "reports"
    figures_dir = output_root / "figures"

    metadata_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    reports_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    figures_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    uav_records = read_uav_records(
        uav_index_csv=uav_index_csv,
        trajectory_csv=trajectory_csv,
        trajectory_crs=args.trajectory_crs,
    )

    canonical_manifest_path = (
        metadata_dir
        / "s8_10b_canonical_uav_query_manifest.csv"
    )

    uav_records.to_csv(
        canonical_manifest_path,
        index=False,
    )

    variant_summaries: list[dict[str, Any]] = []
    variant_outputs: dict[str, dict[str, str]] = {}

    with rasterio.open(src_tif) as raster:
        if raster.crs is None:
            raise RuntimeError(
                "Source raster does not contain a CRS."
            )

        raster_crs = CRS.from_user_input(raster.crs)
        requested_crs = CRS.from_user_input(
            args.trajectory_crs
        )

        raster_epsg = raster_crs.to_epsg()
        requested_epsg = requested_crs.to_epsg()

        if (
            raster_epsg is not None
            and requested_epsg is not None
            and raster_epsg != requested_epsg
        ):
            raise RuntimeError(
                "Trajectory CRS and raster CRS have different "
                "recognized EPSG codes.\n"
                f"Trajectory CRS: {requested_crs}\n"
                f"Trajectory EPSG: {requested_epsg}\n"
                f"Raster CRS: {raster_crs}\n"
                f"Raster EPSG: {raster_epsg}"
            )

        if raster_epsg is None:
            print(
                "Raster CRS has no recoverable EPSG authority. "
                "Proceeding with coordinate-domain validation "
                "because its Lithuania TM projection parameters "
                "match the accepted LKS-94 raster used in S8.7–S8.10A."
            )
        else:
            print(
                f"Raster CRS recognized as EPSG:{raster_epsg}."
            )

        for variant_spec in args.variant:
            parts = variant_spec.split(":", maxsplit=1)

            if len(parts) != 2:
                raise ValueError(
                    "--variant must use "
                    "NAME:TILE_INDEX_CSV"
                )

            variant_name = parts[0]
            tile_index_path = Path(parts[1])

            if not tile_index_path.exists():
                raise FileNotFoundError(
                    tile_index_path
                )

            tiles = read_tile_index(
                tile_index_path
            )

            result, summary = audit_variant(
                variant_name=variant_name,
                tiles=tiles,
                uav_records=uav_records,
                raster_bounds=raster.bounds,
            )

            result_csv = (
                metadata_dir
                / (
                    "s8_10b_uav_tile_oracle_"
                    f"{variant_name}.csv"
                )
            )

            variant_json = (
                reports_dir
                / (
                    "s8_10b_uav_tile_oracle_audit_"
                    f"{variant_name}.json"
                )
            )

            figure_path = (
                figures_dir
                / (
                    "s8_10b_trajectory_vs_tiles_"
                    f"{variant_name}.png"
                )
            )

            result.to_csv(
                result_csv,
                index=False,
            )

            plot_variant(
                variant_name=variant_name,
                tiles=tiles,
                result=result,
                output_path=figure_path,
            )

            variant_report = {
                "stage": "S8.10B",
                "variant": variant_name,
                "status": summary["status"],
                "source_raster": str(src_tif),
                "uav_index_csv": str(
                    uav_index_csv
                ),
                "trajectory_csv": str(
                    trajectory_csv
                ),
                "tile_index_csv": str(
                    tile_index_path
                ),
                "summary": summary,
                "outputs": {
                    "oracle_csv": str(
                        result_csv
                    ),
                    "figure": str(
                        figure_path
                    ),
                    "report_json": str(
                        variant_json
                    ),
                },
                "reference_rule": (
                    "SRT/GPS coordinates are used only "
                    "for query identity, geometric oracle "
                    "construction, visualization, and "
                    "evaluation. They must not be supplied "
                    "to image retrieval, candidate ranking, "
                    "or the geometric verifier."
                ),
            }

            variant_json.write_text(
                json.dumps(
                    variant_report,
                    indent=2,
                )
            )

            variant_summaries.append(
                summary
            )

            variant_outputs[variant_name] = {
                "oracle_csv": str(result_csv),
                "report_json": str(
                    variant_json
                ),
                "figure": str(figure_path),
            }

    summary_dataframe = pd.DataFrame(
        [
            {
                "variant": item["variant"],
                "status": item["status"],
                "uav_query_count": (
                    item["uav_query_count"]
                ),
                "tile_count": item["tile_count"],
                "queries_inside_map_crop": (
                    item[
                        "queries_inside_map_crop"
                    ]
                ),
                "queries_without_oracle_tile": (
                    item[
                        "queries_without_oracle_tile"
                    ]
                ),
                "oracle_tiles_min": (
                    item["oracle_tile_count"][
                        "minimum"
                    ]
                ),
                "oracle_tiles_mean": (
                    item["oracle_tile_count"][
                        "mean"
                    ]
                ),
                "oracle_tiles_median": (
                    item["oracle_tile_count"][
                        "median"
                    ]
                ),
                "oracle_tiles_max": (
                    item["oracle_tile_count"][
                        "maximum"
                    ]
                ),
                "nearest_error_mean_m": (
                    item[
                        "nearest_center_error_m"
                    ]["mean"]
                ),
                "nearest_error_median_m": (
                    item[
                        "nearest_center_error_m"
                    ]["median"]
                ),
                "nearest_error_p95_m": (
                    item[
                        "nearest_center_error_m"
                    ]["p95"]
                ),
                "nearest_error_max_m": (
                    item[
                        "nearest_center_error_m"
                    ]["maximum"]
                ),
                "anchored_edge_oracle_queries": (
                    item[
                        "queries_with_anchored_edge_oracle"
                    ]
                ),
            }
            for item in variant_summaries
        ]
    )

    summary_csv = (
        metadata_dir
        / "s8_10b_uav_tile_oracle_summary.csv"
    )

    summary_dataframe.to_csv(
        summary_csv,
        index=False,
    )

    all_passed = all(
        item["status"]
        == "PASS_UAV_TILE_ORACLE_COVERAGE"
        for item in variant_summaries
    )

    overall_status = (
        "PASS_UAV_TILE_ORACLE_AUDIT"
        if all_passed
        else "FAIL_UAV_TILE_ORACLE_AUDIT"
    )

    overall_report_path = (
        reports_dir
        / "s8_10b_uav_tile_oracle_audit.json"
    )

    overall_report = {
        "stage": "S8.10B",
        "status": overall_status,
        "uav_query_count": len(uav_records),
        "canonical_query_naming": {
            "pattern": (
                "token{token0_id:06d}_"
                "frame{frame_index:06d}_"
                "src{source_frame_cnt:06d}_"
                "lat{latitude:.6f}_"
                "lon{longitude:.6f}.{extension}"
            ),
            "example": (
                uav_records[
                    "canonical_query_filename"
                ].iloc[0]
            ),
            "physical_files_renamed": False,
            "reason": (
                "Canonical names are stored as metadata "
                "to preserve existing extracted-frame paths."
            ),
        },
        "canonical_uav_query_manifest": str(
            canonical_manifest_path
        ),
        "variant_count": len(
            variant_summaries
        ),
        "variants_passed": int(
            sum(
                item["status"]
                == "PASS_UAV_TILE_ORACLE_COVERAGE"
                for item in variant_summaries
            )
        ),
        "variants_failed": int(
            sum(
                item["status"]
                != "PASS_UAV_TILE_ORACLE_COVERAGE"
                for item in variant_summaries
            )
        ),
        "variants": variant_summaries,
        "outputs": {
            "summary_csv": str(
                summary_csv
            ),
            "overall_report_json": str(
                overall_report_path
            ),
            "variant_outputs": (
                variant_outputs
            ),
        },
        "reference_rule": (
            "SRT-derived latitude, longitude, "
            "and projected coordinates are used only "
            "for query identity, map coverage auditing, "
            "oracle construction, visualization, and "
            "evaluation. They must never enter retrieval "
            "or verifier ranking."
        ),
        "next_stage": (
            "S8.10C visual diagnostics and tile-scale "
            "comparison, followed by freezing the "
            "retrieval benchmark protocol."
        ),
    }

    overall_report_path.write_text(
        json.dumps(
            overall_report,
            indent=2,
        )
    )

    print()
    print(
        "S8.10B UAV trajectory coverage and "
        "oracle-tile audit complete"
    )
    print(
        "-------------------------------------"
        "------------------"
    )
    print(
        f"UAV queries:             "
        f"{len(uav_records)}"
    )
    print(
        f"Canonical example:       "
        f"{uav_records['canonical_query_filename'].iloc[0]}"
    )
    print(
        f"Physical images renamed: False"
    )
    print(
        f"Variants audited:        "
        f"{len(variant_summaries)}"
    )
    print(
        f"Overall status:          "
        f"{overall_status}"
    )
    print()

    print(
        summary_dataframe.to_string(
            index=False
        )
    )

    print()
    print(
        f"Saved query manifest:    "
        f"{canonical_manifest_path}"
    )
    print(
        f"Saved summary:           "
        f"{summary_csv}"
    )
    print(
        f"Saved overall report:    "
        f"{overall_report_path}"
    )


if __name__ == "__main__":
    main()
