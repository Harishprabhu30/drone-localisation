'''
Command Executed:

export PYTHONPATH=$PWD/src

python scripts/villoc/s8_9_generate_reference_tiles.py \
  --config configs/dataset_villoc_90deg.yaml \
  --src-tif data/processed/villoc/90_deg/maps/ort10lt_2024_2026/ort10lt_2024_2026_aoi300m.tif \
  --tiles-dir data/processed/villoc/90_deg/maps/ort10lt_2024_2026/tiles_512_s256 \
  --tile-size-px 512 \
  --stride-px 256 \
  --image-format jpg \
  --jpeg-quality 95

'''

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio
import yaml
from PIL import Image, ImageDraw
from pyproj import CRS, Transformer
from rasterio.windows import Window, bounds as window_bounds


def axis_starts(
    length_px: int,
    tile_size_px: int,
    stride_px: int,
) -> list[int]:
    """
    Return tile starts that cover the complete axis.

    The final start is explicitly anchored to length_px - tile_size_px
    whenever the regular stride grid does not land exactly on the edge.
    """
    if length_px <= 0:
        raise ValueError("Axis length must be positive.")

    if tile_size_px <= 0:
        raise ValueError("Tile size must be positive.")

    if stride_px <= 0:
        raise ValueError("Stride must be positive.")

    if stride_px > tile_size_px:
        raise ValueError(
            "Stride cannot exceed tile size because that would leave gaps."
        )

    if length_px < tile_size_px:
        raise ValueError(
            f"Raster axis ({length_px}px) is smaller than the requested "
            f"tile size ({tile_size_px}px)."
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def validate_rgb_array(array: np.ndarray) -> None:
    if array.ndim != 3:
        raise RuntimeError(
            f"Expected CHW raster array, received shape {array.shape}."
        )

    if array.shape[0] < 3:
        raise RuntimeError(
            f"Expected at least three raster bands, received {array.shape[0]}."
        )

    if array.dtype != np.uint8:
        raise RuntimeError(
            f"Expected uint8 raster data, received {array.dtype}."
        )


def create_contact_sheet(
    tile_rows: list[dict[str, Any]],
    output_path: Path,
    sample_n: int = 30,
    thumb_size: int = 180,
    columns: int = 5,
) -> None:
    if not tile_rows:
        return

    sample_n = min(sample_n, len(tile_rows))

    sample_indices = np.linspace(
        0,
        len(tile_rows) - 1,
        sample_n,
        dtype=int,
    )

    sampled = [tile_rows[int(index)] for index in sample_indices]

    rows = int(math.ceil(len(sampled) / columns))
    label_height = 34

    canvas = Image.new(
        "RGB",
        (
            columns * thumb_size,
            rows * (thumb_size + label_height),
        ),
        "white",
    )

    draw = ImageDraw.Draw(canvas)

    for sheet_index, tile_record in enumerate(sampled):
        row = sheet_index // columns
        col = sheet_index % columns

        x = col * thumb_size
        y = row * (thumb_size + label_height)

        tile_path = Path(tile_record["tile_path"])

        with Image.open(tile_path) as image:
            image = image.convert("RGB")
            image.thumbnail((thumb_size, thumb_size))

            paste_x = x + (thumb_size - image.width) // 2
            paste_y = y + (thumb_size - image.height) // 2
            canvas.paste(image, (paste_x, paste_y))

        label = (
            f"{tile_record['tile_id']} "
            f"r{tile_record['grid_row']:02d} "
            f"c{tile_record['grid_col']:02d}"
        )

        draw.text(
            (x + 5, y + thumb_size + 6),
            label,
            fill="black",
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=95)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", required=True)
    parser.add_argument("--src-tif", required=True)

    parser.add_argument(
        "--tiles-dir",
        default=(
            "data/processed/villoc/90_deg/maps/"
            "ort10lt_2024_2026/tiles_512_s256"
        ),
    )

    parser.add_argument("--tile-size-px", type=int, default=512)
    parser.add_argument("--stride-px", type=int, default=256)

    parser.add_argument(
        "--image-format",
        choices=["jpg", "png"],
        default="jpg",
    )

    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()

    if not 1 <= args.jpeg_quality <= 100:
        raise ValueError("--jpeg-quality must be between 1 and 100.")

    config_path = Path(args.config)
    src_tif = Path(args.src_tif)
    tiles_dir = Path(args.tiles_dir)

    if not config_path.exists():
        raise FileNotFoundError(config_path)

    if not src_tif.exists():
        raise FileNotFoundError(src_tif)

    cfg = yaml.safe_load(config_path.read_text())
    output_root = Path(cfg["dataset"]["output_root"])

    metadata_dir = output_root / "metadata"
    reports_dir = output_root / "reports"
    figures_dir = output_root / "figures"

    metadata_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    tiles_dir.mkdir(parents=True, exist_ok=True)

    index_csv = (
        metadata_dir
        / (
            f"s8_9_satellite_tile_index_"
            f"{args.tile_size_px}_s{args.stride_px}.csv"
        )
    )

    report_json = (
        reports_dir
        / (
            f"s8_9_reference_tile_generation_"
            f"{args.tile_size_px}_s{args.stride_px}.json"
        )
    )

    contact_sheet_path = (
        figures_dir
        / (
            f"s8_9_reference_tiles_contact_sheet_"
            f"{args.tile_size_px}_s{args.stride_px}.jpg"
        )
    )

    source_sha256 = sha256_file(src_tif)

    with rasterio.open(src_tif) as src:
        if src.crs is None:
            raise RuntimeError("Source raster has no CRS.")

        if src.count < 3:
            raise RuntimeError(
                f"Source raster must have at least 3 bands; found {src.count}."
            )

        x_starts = axis_starts(
            src.width,
            args.tile_size_px,
            args.stride_px,
        )

        y_starts = axis_starts(
            src.height,
            args.tile_size_px,
            args.stride_px,
        )

        expected_count = len(x_starts) * len(y_starts)

        source_crs = CRS.from_user_input(src.crs)
        transformer_to_wgs84 = Transformer.from_crs(
            source_crs,
            CRS.from_epsg(4326),
            always_xy=True,
        )

        tile_records: list[dict[str, Any]] = []

        tile_number = 0

        for grid_row, row_off in enumerate(y_starts):
            for grid_col, col_off in enumerate(x_starts):
                tile_number += 1

                tile_id = f"sat_{tile_number:06d}"

                window = Window(
                    col_off=col_off,
                    row_off=row_off,
                    width=args.tile_size_px,
                    height=args.tile_size_px,
                )

                data = src.read(
                    indexes=[1, 2, 3],
                    window=window,
                )

                validate_rgb_array(data)

                if data.shape != (
                    3,
                    args.tile_size_px,
                    args.tile_size_px,
                ):
                    raise RuntimeError(
                        f"Unexpected tile shape for {tile_id}: {data.shape}"
                    )

                rgb = np.moveaxis(data, 0, -1)
                image = Image.fromarray(rgb, mode="RGB")

                filename = f"{tile_id}.{args.image_format}"
                tile_path = tiles_dir / filename

                if tile_path.exists() and not args.overwrite:
                    raise FileExistsError(
                        f"Tile already exists: {tile_path}\n"
                        "Use --overwrite to regenerate the tile database."
                    )

                if args.image_format == "jpg":
                    image.save(
                        tile_path,
                        format="JPEG",
                        quality=args.jpeg_quality,
                        subsampling=0,
                        optimize=True,
                    )
                else:
                    image.save(
                        tile_path,
                        format="PNG",
                        optimize=True,
                    )

                left, bottom, right, top = window_bounds(
                    window,
                    src.transform,
                )

                center_easting = (left + right) / 2.0
                center_northing = (bottom + top) / 2.0

                center_lon, center_lat = transformer_to_wgs84.transform(
                    center_easting,
                    center_northing,
                )

                top_left_lon, top_left_lat = transformer_to_wgs84.transform(
                    left,
                    top,
                )

                bottom_right_lon, bottom_right_lat = (
                    transformer_to_wgs84.transform(
                        right,
                        bottom,
                    )
                )

                tile_records.append(
                    {
                        "tile_id": tile_id,
                        "tile_number": tile_number,
                        "tile_path": str(tile_path),
                        "filename": filename,
                        "map_source": "Geoportal ORT10LT 2024-2026",
                        "source_raster": str(src_tif),
                        "source_crs": src.crs.to_string(),
                        "grid_row": grid_row,
                        "grid_col": grid_col,
                        "pixel_col_off": int(col_off),
                        "pixel_row_off": int(row_off),
                        "tile_width_px": args.tile_size_px,
                        "tile_height_px": args.tile_size_px,
                        "left_easting": float(left),
                        "bottom_northing": float(bottom),
                        "right_easting": float(right),
                        "top_northing": float(top),
                        "center_easting": float(center_easting),
                        "center_northing": float(center_northing),
                        "center_lon": float(center_lon),
                        "center_lat": float(center_lat),
                        "top_left_lon": float(top_left_lon),
                        "top_left_lat": float(top_left_lat),
                        "bottom_right_lon": float(bottom_right_lon),
                        "bottom_right_lat": float(bottom_right_lat),
                        "pixel_size_x_m": abs(float(src.transform.a)),
                        "pixel_size_y_m": abs(float(src.transform.e)),
                        "ground_width_m": float(right - left),
                        "ground_height_m": float(top - bottom),
                        "is_right_edge_tile": col_off == x_starts[-1],
                        "is_bottom_edge_tile": row_off == y_starts[-1],
                    }
                )

    if len(tile_records) != expected_count:
        raise RuntimeError(
            f"Generated {len(tile_records)} tiles, "
            f"expected {expected_count}."
        )

    tile_index = pd.DataFrame(tile_records)

    if tile_index["tile_id"].duplicated().any():
        raise RuntimeError("Duplicate tile IDs were generated.")

    if tile_index["tile_path"].duplicated().any():
        raise RuntimeError("Duplicate tile paths were generated.")

    tile_index.to_csv(index_csv, index=False)

    create_contact_sheet(
        tile_rows=tile_records,
        output_path=contact_sheet_path,
        sample_n=30,
        thumb_size=180,
        columns=5,
    )

    ground_stride_x_m = (
        args.stride_px
        * float(tile_index["pixel_size_x_m"].iloc[0])
    )

    ground_stride_y_m = (
        args.stride_px
        * float(tile_index["pixel_size_y_m"].iloc[0])
    )

    report = {
        "stage": "S8.9",
        "status": "PASS_REFERENCE_TILE_DATABASE_GENERATED",
        "source_raster": str(src_tif),
        "source_raster_sha256": source_sha256,
        "tiles_directory": str(tiles_dir),
        "tile_index_csv": str(index_csv),
        "contact_sheet": str(contact_sheet_path),
        "image_format": args.image_format,
        "jpeg_quality": (
            args.jpeg_quality
            if args.image_format == "jpg"
            else None
        ),
        "source_width_px": int(tile_index["pixel_col_off"].max())
        + args.tile_size_px,
        "source_height_px": int(tile_index["pixel_row_off"].max())
        + args.tile_size_px,
        "tile_size_px": args.tile_size_px,
        "stride_px": args.stride_px,
        "overlap_px": args.tile_size_px - args.stride_px,
        "overlap_percent": (
            100.0
            * (args.tile_size_px - args.stride_px)
            / args.tile_size_px
        ),
        "grid_columns": len(x_starts),
        "grid_rows": len(y_starts),
        "expected_tile_count": expected_count,
        "generated_tile_count": len(tile_records),
        "ground_tile_width_m": float(
            tile_index["ground_width_m"].median()
        ),
        "ground_tile_height_m": float(
            tile_index["ground_height_m"].median()
        ),
        "ground_stride_x_m": ground_stride_x_m,
        "ground_stride_y_m": ground_stride_y_m,
        "right_edge_start_px": int(x_starts[-1]),
        "bottom_edge_start_px": int(y_starts[-1]),
        "right_edge_anchored": (
            x_starts[-1] !=
            (len(x_starts) - 1) * args.stride_px
        ),
        "bottom_edge_anchored": (
            y_starts[-1] !=
            (len(y_starts) - 1) * args.stride_px
        ),
        "reference_rule": (
            "Tile coordinates and geographic centers are map metadata. "
            "They may be used for indexing, visualization, and evaluation. "
            "UAV ground-truth coordinates must not be used to rank these tiles."
        ),
        "next_stage": (
            "S8.10 visual and geometric tile-index audit, followed by "
            "S8.11 DINOv2 global descriptor extraction and retrieval."
        ),
    }

    report_json.write_text(json.dumps(report, indent=2))

    print("S8.9 reference-map tile generation complete")
    print("-------------------------------------------")
    print(f"Source raster:       {src_tif}")
    print(f"Source SHA256:       {source_sha256}")
    print(f"Tiles directory:     {tiles_dir}")
    print(
        f"Grid:                "
        f"{len(x_starts)} cols x {len(y_starts)} rows"
    )
    print(f"Generated tiles:     {len(tile_records)}")
    print(
        f"Tile size:           "
        f"{args.tile_size_px} x {args.tile_size_px} px"
    )
    print(
        f"Ground footprint:    "
        f"{report['ground_tile_width_m']:.2f} x "
        f"{report['ground_tile_height_m']:.2f} m"
    )
    print(
        f"Ground stride:       "
        f"{ground_stride_x_m:.2f} x "
        f"{ground_stride_y_m:.2f} m"
    )
    print(
        f"Right edge anchored: {report['right_edge_anchored']}"
    )
    print(
        f"Bottom edge anchored:{report['bottom_edge_anchored']}"
    )
    print(f"Saved index:         {index_csv}")
    print(f"Saved report:        {report_json}")
    print(f"Saved contact sheet: {contact_sheet_path}")


if __name__ == "__main__":
    main()
