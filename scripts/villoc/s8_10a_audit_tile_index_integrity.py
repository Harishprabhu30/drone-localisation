'''
Command Executed:

export PYTHONPATH=$PWD/src

python scripts/villoc/s8_10a_audit_tile_index_integrity.py \
  --config configs/dataset_villoc_90deg.yaml \
  --src-tif data/processed/villoc/90_deg/maps/ort10lt_2024_2026/ort10lt_2024_2026_aoi300m.tif \
  --variant "512_s256:512:256:data/processed/villoc/90_deg/maps/ort10lt_2024_2026/tiles_512_s256:outputs/villoc/90_deg/metadata/s8_9_satellite_tile_index_512_s256.csv" \
  --variant "1024_s512:1024:512:data/processed/villoc/90_deg/maps/ort10lt_2024_2026/tiles_1024_s512:outputs/villoc/90_deg/metadata/s8_9_satellite_tile_index_1024_s512.csv" \
  --variant "1024_s256:1024:256:data/processed/villoc/90_deg/maps/ort10lt_2024_2026/tiles_1024_s256:outputs/villoc/90_deg/metadata/s8_9_satellite_tile_index_1024_s256.csv"

'''

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio
import yaml
from PIL import Image
from pyproj import CRS, Transformer
from rasterio.windows import Window, bounds as window_bounds


REQUIRED_COLUMNS = {
    "tile_id",
    "tile_number",
    "tile_path",
    "filename",
    "grid_row",
    "grid_col",
    "pixel_col_off",
    "pixel_row_off",
    "tile_width_px",
    "tile_height_px",
    "left_easting",
    "bottom_northing",
    "right_easting",
    "top_northing",
    "center_easting",
    "center_northing",
    "center_lon",
    "center_lat",
    "pixel_size_x_m",
    "pixel_size_y_m",
    "ground_width_m",
    "ground_height_m",
    "is_right_edge_tile",
    "is_bottom_edge_tile",
}


@dataclass(frozen=True)
class Variant:
    name: str
    tile_size_px: int
    stride_px: int
    tiles_dir: Path
    index_csv: Path


def parse_variant(spec: str) -> Variant:
    """
    Parse:

    NAME:TILE_SIZE:STRIDE:TILES_DIR:INDEX_CSV
    """
    parts = spec.split(":", maxsplit=4)

    if len(parts) != 5:
        raise ValueError(
            "Invalid --variant specification.\n"
            "Expected:\n"
            "NAME:TILE_SIZE:STRIDE:TILES_DIR:INDEX_CSV"
        )

    name, tile_size, stride, tiles_dir, index_csv = parts

    return Variant(
        name=name,
        tile_size_px=int(tile_size),
        stride_px=int(stride),
        tiles_dir=Path(tiles_dir),
        index_csv=Path(index_csv),
    )


def axis_starts(
    length_px: int,
    tile_size_px: int,
    stride_px: int,
) -> list[int]:
    if length_px < tile_size_px:
        raise ValueError(
            f"Raster axis {length_px}px is smaller than "
            f"tile size {tile_size_px}px."
        )

    if stride_px <= 0:
        raise ValueError("Stride must be positive.")

    if stride_px > tile_size_px:
        raise ValueError(
            "Stride cannot exceed tile size because gaps would occur."
        )

    last_start = length_px - tile_size_px

    starts = list(
        range(
            0,
            last_start + 1,
            stride_px,
        )
    )

    if not starts:
        starts = [0]

    if starts[-1] != last_start:
        starts.append(last_start)

    return starts


def nearly_equal(
    actual: float,
    expected: float,
    tolerance: float,
) -> bool:
    return abs(float(actual) - float(expected)) <= tolerance


def normalize_bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)

    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
        "yes": True,
        "no": False,
    }

    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map(mapping)
    )


def interval_has_no_gaps(
    starts: list[int],
    tile_size_px: int,
    axis_length_px: int,
) -> tuple[bool, list[dict[str, int]]]:
    intervals = sorted(
        (start, start + tile_size_px)
        for start in starts
    )

    issues: list[dict[str, int]] = []

    if not intervals:
        return False, [
            {
                "previous_end": 0,
                "next_start": axis_length_px,
                "gap_px": axis_length_px,
            }
        ]

    if intervals[0][0] != 0:
        issues.append(
            {
                "previous_end": 0,
                "next_start": intervals[0][0],
                "gap_px": intervals[0][0],
            }
        )

    covered_end = intervals[0][1]

    for start, end in intervals[1:]:
        if start > covered_end:
            issues.append(
                {
                    "previous_end": covered_end,
                    "next_start": start,
                    "gap_px": start - covered_end,
                }
            )

        covered_end = max(covered_end, end)

    if covered_end < axis_length_px:
        issues.append(
            {
                "previous_end": covered_end,
                "next_start": axis_length_px,
                "gap_px": axis_length_px - covered_end,
            }
        )

    return len(issues) == 0, issues


def add_failure(
    failures: list[dict[str, Any]],
    variant_name: str,
    check_group: str,
    check_name: str,
    message: str,
    tile_id: str | None = None,
    row_index: int | None = None,
) -> None:
    failures.append(
        {
            "variant": variant_name,
            "check_group": check_group,
            "check_name": check_name,
            "tile_id": tile_id,
            "row_index": row_index,
            "message": message,
        }
    )


def audit_variant(
    variant: Variant,
    src: rasterio.io.DatasetReader,
    geometry_tolerance_m: float,
    latlon_tolerance_deg: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    failures: list[dict[str, Any]] = []

    if not variant.tiles_dir.exists():
        add_failure(
            failures,
            variant.name,
            "filesystem",
            "tiles_directory_exists",
            f"Tiles directory does not exist: {variant.tiles_dir}",
        )

    if not variant.index_csv.exists():
        add_failure(
            failures,
            variant.name,
            "index",
            "index_csv_exists",
            f"Index CSV does not exist: {variant.index_csv}",
        )

        return (
            {
                "variant": variant.name,
                "status": "FAIL",
                "failure_count": len(failures),
            },
            failures,
        )

    df = pd.read_csv(variant.index_csv)

    missing_columns = sorted(REQUIRED_COLUMNS - set(df.columns))

    if missing_columns:
        add_failure(
            failures,
            variant.name,
            "index",
            "required_columns",
            f"Missing columns: {missing_columns}",
        )

        return (
            {
                "variant": variant.name,
                "status": "FAIL",
                "index_rows": len(df),
                "failure_count": len(failures),
            },
            failures,
        )

    expected_x_starts = axis_starts(
        src.width,
        variant.tile_size_px,
        variant.stride_px,
    )

    expected_y_starts = axis_starts(
        src.height,
        variant.tile_size_px,
        variant.stride_px,
    )

    expected_count = (
        len(expected_x_starts)
        * len(expected_y_starts)
    )

    expected_grid_pairs = {
        (grid_row, grid_col)
        for grid_row in range(len(expected_y_starts))
        for grid_col in range(len(expected_x_starts))
    }

    actual_grid_pairs = set(
        zip(
            df["grid_row"].astype(int),
            df["grid_col"].astype(int),
        )
    )

    if len(df) != expected_count:
        add_failure(
            failures,
            variant.name,
            "index",
            "expected_row_count",
            f"Index has {len(df)} rows; expected {expected_count}.",
        )

    if df["tile_id"].duplicated().any():
        duplicated = (
            df.loc[df["tile_id"].duplicated(False), "tile_id"]
            .astype(str)
            .tolist()
        )

        add_failure(
            failures,
            variant.name,
            "index",
            "unique_tile_ids",
            f"Duplicate tile IDs: {duplicated[:20]}",
        )

    if df["tile_path"].duplicated().any():
        duplicated = (
            df.loc[df["tile_path"].duplicated(False), "tile_path"]
            .astype(str)
            .tolist()
        )

        add_failure(
            failures,
            variant.name,
            "index",
            "unique_tile_paths",
            f"Duplicate tile paths: {duplicated[:20]}",
        )

    if df["filename"].duplicated().any():
        duplicated = (
            df.loc[df["filename"].duplicated(False), "filename"]
            .astype(str)
            .tolist()
        )

        add_failure(
            failures,
            variant.name,
            "index",
            "unique_filenames",
            f"Duplicate filenames: {duplicated[:20]}",
        )

    missing_grid_pairs = sorted(
        expected_grid_pairs - actual_grid_pairs
    )

    unexpected_grid_pairs = sorted(
        actual_grid_pairs - expected_grid_pairs
    )

    if missing_grid_pairs:
        add_failure(
            failures,
            variant.name,
            "grid",
            "complete_grid",
            f"Missing grid cells: {missing_grid_pairs[:20]}",
        )

    if unexpected_grid_pairs:
        add_failure(
            failures,
            variant.name,
            "grid",
            "valid_grid_coordinates",
            f"Unexpected grid cells: {unexpected_grid_pairs[:20]}",
        )

    expected_ids = [
        f"sat_{number:06d}"
        for number in range(1, expected_count + 1)
    ]

    actual_ids = df.sort_values("tile_number")[
        "tile_id"
    ].astype(str).tolist()

    if actual_ids != expected_ids:
        add_failure(
            failures,
            variant.name,
            "index",
            "sequential_tile_ids",
            "Tile IDs are not a complete sequential sat_000001... sequence.",
        )

    disk_files = sorted(
        path
        for path in variant.tiles_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ) if variant.tiles_dir.exists() else []

    indexed_paths = {
        str(Path(path).resolve())
        for path in df["tile_path"].astype(str)
    }

    disk_paths = {
        str(path.resolve())
        for path in disk_files
    }

    missing_on_disk = sorted(indexed_paths - disk_paths)
    unindexed_on_disk = sorted(disk_paths - indexed_paths)

    if missing_on_disk:
        add_failure(
            failures,
            variant.name,
            "filesystem",
            "all_indexed_files_exist",
            f"{len(missing_on_disk)} indexed files are missing. "
            f"Examples: {missing_on_disk[:10]}",
        )

    if unindexed_on_disk:
        add_failure(
            failures,
            variant.name,
            "filesystem",
            "no_unindexed_images",
            f"{len(unindexed_on_disk)} image files are not indexed. "
            f"Examples: {unindexed_on_disk[:10]}",
        )

    if len(disk_files) != expected_count:
        add_failure(
            failures,
            variant.name,
            "filesystem",
            "expected_image_count",
            f"Found {len(disk_files)} images; expected {expected_count}.",
        )

    source_crs = CRS.from_user_input(src.crs)

    transformer_to_wgs84 = Transformer.from_crs(
        source_crs,
        CRS.from_epsg(4326),
        always_xy=True,
    )

    image_sizes: set[tuple[int, int]] = set()
    corrupt_file_count = 0

    expected_right_flags = 0
    expected_bottom_flags = 0

    normalized_right_flags = normalize_bool_series(
        df["is_right_edge_tile"]
    )

    normalized_bottom_flags = normalize_bool_series(
        df["is_bottom_edge_tile"]
    )

    if normalized_right_flags.isna().any():
        add_failure(
            failures,
            variant.name,
            "index",
            "right_edge_boolean_values",
            "Could not parse all is_right_edge_tile values as booleans.",
        )

    if normalized_bottom_flags.isna().any():
        add_failure(
            failures,
            variant.name,
            "index",
            "bottom_edge_boolean_values",
            "Could not parse all is_bottom_edge_tile values as booleans.",
        )

    for row_index, row in df.iterrows():
        tile_id = str(row["tile_id"])
        tile_path = Path(str(row["tile_path"]))

        grid_row = int(row["grid_row"])
        grid_col = int(row["grid_col"])

        if not (
            0 <= grid_row < len(expected_y_starts)
            and 0 <= grid_col < len(expected_x_starts)
        ):
            continue

        expected_col_off = expected_x_starts[grid_col]
        expected_row_off = expected_y_starts[grid_row]

        actual_col_off = int(row["pixel_col_off"])
        actual_row_off = int(row["pixel_row_off"])

        if actual_col_off != expected_col_off:
            add_failure(
                failures,
                variant.name,
                "grid",
                "pixel_col_offset",
                f"Expected col_off={expected_col_off}, "
                f"found {actual_col_off}.",
                tile_id,
                int(row_index),
            )

        if actual_row_off != expected_row_off:
            add_failure(
                failures,
                variant.name,
                "grid",
                "pixel_row_offset",
                f"Expected row_off={expected_row_off}, "
                f"found {actual_row_off}.",
                tile_id,
                int(row_index),
            )

        if int(row["tile_width_px"]) != variant.tile_size_px:
            add_failure(
                failures,
                variant.name,
                "index",
                "declared_tile_width",
                f"Expected {variant.tile_size_px}, "
                f"found {row['tile_width_px']}.",
                tile_id,
                int(row_index),
            )

        if int(row["tile_height_px"]) != variant.tile_size_px:
            add_failure(
                failures,
                variant.name,
                "index",
                "declared_tile_height",
                f"Expected {variant.tile_size_px}, "
                f"found {row['tile_height_px']}.",
                tile_id,
                int(row_index),
            )

        window = Window(
            col_off=expected_col_off,
            row_off=expected_row_off,
            width=variant.tile_size_px,
            height=variant.tile_size_px,
        )

        expected_left, expected_bottom, expected_right, expected_top = (
            window_bounds(
                window,
                src.transform,
            )
        )

        expected_center_easting = (
            expected_left + expected_right
        ) / 2.0

        expected_center_northing = (
            expected_bottom + expected_top
        ) / 2.0

        geometry_checks = {
            "left_easting": expected_left,
            "bottom_northing": expected_bottom,
            "right_easting": expected_right,
            "top_northing": expected_top,
            "center_easting": expected_center_easting,
            "center_northing": expected_center_northing,
            "ground_width_m": expected_right - expected_left,
            "ground_height_m": expected_top - expected_bottom,
            "pixel_size_x_m": abs(float(src.transform.a)),
            "pixel_size_y_m": abs(float(src.transform.e)),
        }

        for column, expected_value in geometry_checks.items():
            actual_value = float(row[column])

            if not nearly_equal(
                actual_value,
                expected_value,
                geometry_tolerance_m,
            ):
                add_failure(
                    failures,
                    variant.name,
                    "geometry",
                    column,
                    f"Expected {expected_value:.9f}, "
                    f"found {actual_value:.9f}.",
                    tile_id,
                    int(row_index),
                )

        expected_lon, expected_lat = transformer_to_wgs84.transform(
            expected_center_easting,
            expected_center_northing,
        )

        if not nearly_equal(
            float(row["center_lon"]),
            expected_lon,
            latlon_tolerance_deg,
        ):
            add_failure(
                failures,
                variant.name,
                "geometry",
                "center_lon",
                f"Expected {expected_lon:.12f}, "
                f"found {float(row['center_lon']):.12f}.",
                tile_id,
                int(row_index),
            )

        if not nearly_equal(
            float(row["center_lat"]),
            expected_lat,
            latlon_tolerance_deg,
        ):
            add_failure(
                failures,
                variant.name,
                "geometry",
                "center_lat",
                f"Expected {expected_lat:.12f}, "
                f"found {float(row['center_lat']):.12f}.",
                tile_id,
                int(row_index),
            )

        expected_right_flag = (
            expected_col_off == expected_x_starts[-1]
        )

        expected_bottom_flag = (
            expected_row_off == expected_y_starts[-1]
        )

        expected_right_flags += int(expected_right_flag)
        expected_bottom_flags += int(expected_bottom_flag)

        actual_right_flag = normalized_right_flags.iloc[row_index]
        actual_bottom_flag = normalized_bottom_flags.iloc[row_index]

        if (
            pd.notna(actual_right_flag)
            and bool(actual_right_flag) != expected_right_flag
        ):
            add_failure(
                failures,
                variant.name,
                "grid",
                "right_edge_flag",
                f"Expected {expected_right_flag}, "
                f"found {actual_right_flag}.",
                tile_id,
                int(row_index),
            )

        if (
            pd.notna(actual_bottom_flag)
            and bool(actual_bottom_flag) != expected_bottom_flag
        ):
            add_failure(
                failures,
                variant.name,
                "grid",
                "bottom_edge_flag",
                f"Expected {expected_bottom_flag}, "
                f"found {actual_bottom_flag}.",
                tile_id,
                int(row_index),
            )

        expected_filename = tile_path.name

        if str(row["filename"]) != expected_filename:
            add_failure(
                failures,
                variant.name,
                "filesystem",
                "filename_matches_path",
                f"filename column is {row['filename']}; "
                f"path basename is {expected_filename}.",
                tile_id,
                int(row_index),
            )

        if tile_path.exists():
            try:
                with Image.open(tile_path) as image:
                    image.verify()

                with Image.open(tile_path) as image:
                    image_sizes.add(image.size)

                    if image.size != (
                        variant.tile_size_px,
                        variant.tile_size_px,
                    ):
                        add_failure(
                            failures,
                            variant.name,
                            "image",
                            "image_dimensions",
                            f"Expected "
                            f"{variant.tile_size_px}x"
                            f"{variant.tile_size_px}, "
                            f"found {image.size}.",
                            tile_id,
                            int(row_index),
                        )

                    if image.mode not in {"RGB", "RGBA"}:
                        add_failure(
                            failures,
                            variant.name,
                            "image",
                            "image_mode",
                            f"Unexpected image mode: {image.mode}.",
                            tile_id,
                            int(row_index),
                        )

            except Exception as exc:
                corrupt_file_count += 1

                add_failure(
                    failures,
                    variant.name,
                    "image",
                    "image_readability",
                    f"Failed to read image: {exc}",
                    tile_id,
                    int(row_index),
                )

    x_no_gaps, x_gap_issues = interval_has_no_gaps(
        expected_x_starts,
        variant.tile_size_px,
        src.width,
    )

    y_no_gaps, y_gap_issues = interval_has_no_gaps(
        expected_y_starts,
        variant.tile_size_px,
        src.height,
    )

    if not x_no_gaps:
        add_failure(
            failures,
            variant.name,
            "coverage",
            "horizontal_no_gaps",
            f"Horizontal gap issues: {x_gap_issues}",
        )

    if not y_no_gaps:
        add_failure(
            failures,
            variant.name,
            "coverage",
            "vertical_no_gaps",
            f"Vertical gap issues: {y_gap_issues}",
        )

    actual_x_starts = sorted(
        df["pixel_col_off"].astype(int).unique().tolist()
    )

    actual_y_starts = sorted(
        df["pixel_row_off"].astype(int).unique().tolist()
    )

    if actual_x_starts != expected_x_starts:
        add_failure(
            failures,
            variant.name,
            "grid",
            "horizontal_start_sequence",
            f"Expected {expected_x_starts}; "
            f"found {actual_x_starts}.",
        )

    if actual_y_starts != expected_y_starts:
        add_failure(
            failures,
            variant.name,
            "grid",
            "vertical_start_sequence",
            f"Expected {expected_y_starts}; "
            f"found {actual_y_starts}.",
        )

    horizontal_steps = [
        right - left
        for left, right in zip(
            expected_x_starts[:-1],
            expected_x_starts[1:],
        )
    ]

    vertical_steps = [
        bottom - top
        for top, bottom in zip(
            expected_y_starts[:-1],
            expected_y_starts[1:],
        )
    ]

    horizontal_overlaps = [
        variant.tile_size_px - step
        for step in horizontal_steps
    ]

    vertical_overlaps = [
        variant.tile_size_px - step
        for step in vertical_steps
    ]

    status = "PASS" if not failures else "FAIL"

    summary = {
        "variant": variant.name,
        "status": status,
        "tile_size_px": variant.tile_size_px,
        "stride_px": variant.stride_px,
        "declared_overlap_px": (
            variant.tile_size_px - variant.stride_px
        ),
        "tiles_directory": str(variant.tiles_dir),
        "index_csv": str(variant.index_csv),
        "source_raster_width_px": src.width,
        "source_raster_height_px": src.height,
        "expected_grid_columns": len(expected_x_starts),
        "expected_grid_rows": len(expected_y_starts),
        "expected_tile_count": expected_count,
        "index_row_count": len(df),
        "disk_image_count": len(disk_files),
        "unique_tile_ids": int(df["tile_id"].nunique()),
        "unique_tile_paths": int(df["tile_path"].nunique()),
        "image_sizes_found": [
            list(size)
            for size in sorted(image_sizes)
        ],
        "corrupt_file_count": corrupt_file_count,
        "missing_indexed_file_count": len(missing_on_disk),
        "unindexed_disk_file_count": len(unindexed_on_disk),
        "expected_x_starts_px": expected_x_starts,
        "expected_y_starts_px": expected_y_starts,
        "horizontal_steps_px": horizontal_steps,
        "vertical_steps_px": vertical_steps,
        "horizontal_overlaps_px": horizontal_overlaps,
        "vertical_overlaps_px": vertical_overlaps,
        "right_edge_start_px": expected_x_starts[-1],
        "bottom_edge_start_px": expected_y_starts[-1],
        "right_edge_is_anchored": (
            expected_x_starts[-1]
            != (len(expected_x_starts) - 1)
            * variant.stride_px
        ),
        "bottom_edge_is_anchored": (
            expected_y_starts[-1]
            != (len(expected_y_starts) - 1)
            * variant.stride_px
        ),
        "horizontal_full_coverage": x_no_gaps,
        "vertical_full_coverage": y_no_gaps,
        "expected_right_edge_tile_count": expected_right_flags,
        "expected_bottom_edge_tile_count": expected_bottom_flags,
        "failure_count": len(failures),
    }

    return summary, failures


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", required=True)
    parser.add_argument("--src-tif", required=True)

    parser.add_argument(
        "--variant",
        action="append",
        required=True,
        help=(
            "Repeat for each variant:\n"
            "NAME:TILE_SIZE:STRIDE:TILES_DIR:INDEX_CSV"
        ),
    )

    parser.add_argument(
        "--geometry-tolerance-m",
        type=float,
        default=1e-5,
    )

    parser.add_argument(
        "--latlon-tolerance-deg",
        type=float,
        default=1e-9,
    )

    args = parser.parse_args()

    config_path = Path(args.config)
    src_tif = Path(args.src_tif)

    if not config_path.exists():
        raise FileNotFoundError(config_path)

    if not src_tif.exists():
        raise FileNotFoundError(src_tif)

    cfg = yaml.safe_load(config_path.read_text())
    output_root = Path(cfg["dataset"]["output_root"])

    reports_dir = output_root / "reports"
    metadata_dir = output_root / "metadata"

    reports_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    variants = [
        parse_variant(spec)
        for spec in args.variant
    ]

    all_summaries: list[dict[str, Any]] = []
    all_failures: list[dict[str, Any]] = []

    with rasterio.open(src_tif) as src:
        if src.crs is None:
            raise RuntimeError("Source raster has no CRS.")

        for variant in variants:
            summary, failures = audit_variant(
                variant=variant,
                src=src,
                geometry_tolerance_m=args.geometry_tolerance_m,
                latlon_tolerance_deg=args.latlon_tolerance_deg,
            )

            all_summaries.append(summary)
            all_failures.extend(failures)

    summary_df = pd.DataFrame(all_summaries)

    summary_csv = (
        metadata_dir
        / "s8_10a_tile_index_integrity_summary.csv"
    )

    failures_csv = (
        metadata_dir
        / "s8_10a_tile_index_integrity_failures.csv"
    )

    report_json = (
        reports_dir
        / "s8_10a_tile_index_integrity_audit.json"
    )

    summary_df.to_csv(summary_csv, index=False)

    failures_df = pd.DataFrame(
        all_failures,
        columns=[
            "variant",
            "check_group",
            "check_name",
            "tile_id",
            "row_index",
            "message",
        ],
    )

    failures_df.to_csv(failures_csv, index=False)

    overall_status = (
        "PASS_TILE_INDEX_INTEGRITY"
        if all(
            item["status"] == "PASS"
            for item in all_summaries
        )
        else "FAIL_TILE_INDEX_INTEGRITY"
    )

    report = {
        "stage": "S8.10A",
        "status": overall_status,
        "source_raster": str(src_tif),
        "geometry_tolerance_m": args.geometry_tolerance_m,
        "latlon_tolerance_deg": args.latlon_tolerance_deg,
        "variant_count": len(all_summaries),
        "variants_passed": sum(
            item["status"] == "PASS"
            for item in all_summaries
        ),
        "variants_failed": sum(
            item["status"] != "PASS"
            for item in all_summaries
        ),
        "total_failure_count": len(all_failures),
        "variants": all_summaries,
        "outputs": {
            "summary_csv": str(summary_csv),
            "failures_csv": str(failures_csv),
            "report_json": str(report_json),
        },
        "scope": (
            "Tile CSV integrity, filesystem consistency, image readability, "
            "pixel dimensions, raster-derived map geometry, edge anchoring, "
            "and full raster coverage. UAV reference positions are excluded "
            "from S8.10A and will be evaluated in S8.10B."
        ),
        "next_stage": (
            "S8.10B UAV trajectory coverage and oracle-tile audit."
        ),
    }

    report_json.write_text(json.dumps(report, indent=2))

    print("S8.10A tile/index integrity audit complete")
    print("-----------------------------------------")
    print(f"Source raster:       {src_tif}")
    print(f"Variants audited:    {len(all_summaries)}")
    print(f"Variants passed:     {report['variants_passed']}")
    print(f"Variants failed:     {report['variants_failed']}")
    print(f"Total failures:      {len(all_failures)}")
    print(f"Overall status:      {overall_status}")
    print()

    display_columns = [
        "variant",
        "status",
        "expected_tile_count",
        "index_row_count",
        "disk_image_count",
        "unique_tile_ids",
        "corrupt_file_count",
        "failure_count",
    ]

    print(
        summary_df[display_columns]
        .to_string(index=False)
    )

    print()
    print(f"Saved summary:       {summary_csv}")
    print(f"Saved failures:      {failures_csv}")
    print(f"Saved report:        {report_json}")


if __name__ == "__main__":
    main()
