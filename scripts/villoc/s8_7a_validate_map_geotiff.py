'''
Commad Executed:

export PYTHONPATH=$PWD/src

python scripts/villoc/s8_7a_validate_map_geotiff.py \
  --config configs/dataset_villoc_90deg.yaml \
  --tif data/raw/villoc/90_deg/maps/ort10lt_2024_2026/master/ort10lt_2024_2026_master.tif \
  --name ort10lt_2024_2026_master \
  --tile-size-px 512 \
  --stride-px 256

'''

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import rasterio
import yaml
from rasterio.crs import CRS
from rasterio.warp import transform_bounds


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def normalized_crs_name(crs: CRS | None) -> str | None:
    if crs is None:
        return None

    epsg = crs.to_epsg()
    if epsg is not None:
        return f"EPSG:{epsg}"

    return crs.to_string()


def estimate_grid_count(
    width_px: int,
    height_px: int,
    tile_size_px: int,
    stride_px: int,
) -> dict[str, int]:
    def count_axis(length_px: int) -> int:
        if length_px <= tile_size_px:
            return 1

        return int(math.ceil((length_px - tile_size_px) / stride_px)) + 1

    cols = count_axis(width_px)
    rows = count_axis(height_px)

    return {
        "tile_columns": cols,
        "tile_rows": rows,
        "tile_count": cols * rows,
    }


def intersect_bounds(
    raster_bounds: dict[str, float],
    target_bounds: dict[str, float],
) -> dict[str, float | bool]:
    left = max(raster_bounds["left"], target_bounds["xmin"])
    right = min(raster_bounds["right"], target_bounds["xmax"])
    bottom = max(raster_bounds["bottom"], target_bounds["ymin"])
    top = min(raster_bounds["top"], target_bounds["ymax"])

    has_overlap = right > left and top > bottom

    if not has_overlap:
        return {
            "has_overlap": False,
            "left": left,
            "right": right,
            "bottom": bottom,
            "top": top,
            "width_m": 0.0,
            "height_m": 0.0,
            "area_m2": 0.0,
        }

    return {
        "has_overlap": True,
        "left": left,
        "right": right,
        "bottom": bottom,
        "top": top,
        "width_m": right - left,
        "height_m": top - bottom,
        "area_m2": (right - left) * (top - bottom),
    }


def transform_target_bbox(
    target_bbox_3346: dict[str, float],
    raster_crs: CRS,
) -> dict[str, float]:
    target_crs = CRS.from_epsg(3346)

    if raster_crs == target_crs:
        return {
            "xmin": float(target_bbox_3346["xmin"]),
            "ymin": float(target_bbox_3346["ymin"]),
            "xmax": float(target_bbox_3346["xmax"]),
            "ymax": float(target_bbox_3346["ymax"]),
        }

    left, bottom, right, top = transform_bounds(
        target_crs,
        raster_crs,
        target_bbox_3346["xmin"],
        target_bbox_3346["ymin"],
        target_bbox_3346["xmax"],
        target_bbox_3346["ymax"],
        densify_pts=21,
    )

    return {
        "xmin": float(left),
        "ymin": float(bottom),
        "xmax": float(right),
        "ymax": float(top),
    }


def suitability_from_resolution(max_pixel_size_m: float) -> str:
    if max_pixel_size_m <= 0.35:
        return "EXCELLENT_NATIVE_ORTHOPHOTO"
    if max_pixel_size_m <= 0.60:
        return "GOOD_FOR_TILE_MATCHING"
    if max_pixel_size_m <= 1.00:
        return "USABLE_FOR_FIRST_BASELINE"
    if max_pixel_size_m <= 2.00:
        return "WEAK_LOW_DETAIL"
    return "TOO_COARSE_REEXPORT_NEEDED"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--tif", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--tile-size-px", type=int, default=512)
    parser.add_argument("--stride-px", type=int, default=256)
    args = parser.parse_args()

    config_path = Path(args.config)
    tif_path = Path(args.tif)

    if not tif_path.exists():
        raise FileNotFoundError(tif_path)

    cfg = yaml.safe_load(config_path.read_text())
    output_root = Path(cfg["dataset"]["output_root"])

    reports_dir = output_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    s8_6_plan_path = reports_dir / "s8_6_map_bbox_plan.json"
    s8_6_plan = load_json(s8_6_plan_path)
    target_bbox_3346 = s8_6_plan["padded_bboxes_3346"]["300m"]

    tfw_candidates = [
        tif_path.with_suffix(".tfw"),
        tif_path.with_suffix(".TFW"),
        Path(str(tif_path) + "w"),
    ]
    detected_world_file = next((p for p in tfw_candidates if p.exists()), None)

    file_size_bytes = tif_path.stat().st_size

    with rasterio.open(tif_path) as src:
        if src.crs is None:
            raise RuntimeError(
                "Rasterio could not determine the raster CRS. "
                "A .tfw supplies transform coordinates but usually not the CRS. "
                "Confirm that the TIFF embeds EPSG:3346 or add the CRS using GDAL/QGIS."
            )

        bounds = src.bounds
        transform = src.transform
        raster_crs = src.crs

        raster_bounds = {
            "left": float(bounds.left),
            "bottom": float(bounds.bottom),
            "right": float(bounds.right),
            "top": float(bounds.top),
        }

        pixel_size_x = abs(float(transform.a))
        pixel_size_y = abs(float(transform.e))
        max_pixel_size = max(pixel_size_x, pixel_size_y)

        raster_width_units = float(bounds.right - bounds.left)
        raster_height_units = float(bounds.top - bounds.bottom)

        target_in_raster_crs = transform_target_bbox(
            target_bbox_3346,
            raster_crs,
        )

        intersection = intersect_bounds(
            raster_bounds,
            target_in_raster_crs,
        )

        target_area = (
            (target_in_raster_crs["xmax"] - target_in_raster_crs["xmin"])
            * (target_in_raster_crs["ymax"] - target_in_raster_crs["ymin"])
        )

        target_coverage_pct = (
            100.0 * float(intersection["area_m2"]) / target_area
            if target_area > 0
            else 0.0
        )

        covers_target = target_coverage_pct >= 99.999

        # Convert target AOI dimensions to approximate source pixels.
        target_width_px = max(
            1,
            int(
                math.ceil(
                    (target_in_raster_crs["xmax"] - target_in_raster_crs["xmin"])
                    / pixel_size_x
                )
            ),
        )
        target_height_px = max(
            1,
            int(
                math.ceil(
                    (target_in_raster_crs["ymax"] - target_in_raster_crs["ymin"])
                    / pixel_size_y
                )
            ),
        )

        full_grid = estimate_grid_count(
            src.width,
            src.height,
            args.tile_size_px,
            args.stride_px,
        )

        target_grid = estimate_grid_count(
            target_width_px,
            target_height_px,
            args.tile_size_px,
            args.stride_px,
        )

        uncompressed_bytes = (
            int(src.width)
            * int(src.height)
            * int(src.count)
            * (int(src.dtypes[0].replace("uint", "").replace("int", "")) // 8)
            if src.dtypes and (
                src.dtypes[0].startswith("uint")
                or src.dtypes[0].startswith("int")
            )
            else None
        )

        suitability = suitability_from_resolution(max_pixel_size)

        validation_pass = (
            covers_target
            and max_pixel_size <= 1.0
            and src.count >= 3
            and src.width > 0
            and src.height > 0
        )

        status = (
            "PASS_MASTER_ORTHOPHOTO_ACCEPTED"
            if validation_pass
            else "REVIEW_REQUIRED"
        )

        report = {
            "stage": "S8.7A",
            "map_name": args.name,
            "status": status,
            "source_files": {
                "tif": str(tif_path),
                "world_file": (
                    str(detected_world_file)
                    if detected_world_file is not None
                    else None
                ),
                "file_size_bytes": file_size_bytes,
                "file_size_mb": file_size_bytes / (1024 ** 2),
            },
            "geotiff_info": {
                "driver": src.driver,
                "width_px": int(src.width),
                "height_px": int(src.height),
                "count_bands": int(src.count),
                "dtypes": list(src.dtypes),
                "crs": raster_crs.to_wkt(),
                "crs_normalized": normalized_crs_name(raster_crs),
                "bounds": raster_bounds,
                "transform": [
                    float(transform.a),
                    float(transform.b),
                    float(transform.c),
                    float(transform.d),
                    float(transform.e),
                    float(transform.f),
                ],
                "pixel_size_x": pixel_size_x,
                "pixel_size_y": pixel_size_y,
                "max_pixel_size": max_pixel_size,
                "raster_width_units": raster_width_units,
                "raster_height_units": raster_height_units,
                "compression": src.compression.value if src.compression else None,
                "color_interpretation": [
                    item.name for item in src.colorinterp
                ],
                "nodata": src.nodata,
                "estimated_uncompressed_bytes": uncompressed_bytes,
            },
            "target_aoi": {
                "source_bbox_epsg3346": target_bbox_3346,
                "bbox_in_raster_crs": target_in_raster_crs,
                "intersection": intersection,
                "coverage_percent": target_coverage_pct,
                "fully_covers_target": covers_target,
                "estimated_crop_width_px": target_width_px,
                "estimated_crop_height_px": target_height_px,
            },
            "tiling_preflight": {
                "tile_size_px": args.tile_size_px,
                "stride_px": args.stride_px,
                "overlap_px": args.tile_size_px - args.stride_px,
                "overlap_percent": (
                    100.0
                    * (args.tile_size_px - args.stride_px)
                    / args.tile_size_px
                ),
                "full_raster_grid": full_grid,
                "target_aoi_grid": target_grid,
                "recommended_action": (
                    "Crop the master raster to the S8.6 300 m AOI first, "
                    "then generate tiles from that crop. Do not tile the entire "
                    "500 MB master unless a full-area retrieval experiment is required."
                ),
            },
            "assessment": {
                "resolution_suitability": suitability,
                "validation_pass": validation_pass,
                "pass_requirements": {
                    "fully_covers_target": covers_target,
                    "maximum_pixel_size_le_1m": max_pixel_size <= 1.0,
                    "minimum_three_bands": src.count >= 3,
                },
            },
            "reference_rule": (
                "The SRT-derived AOI is allowed for map acquisition, cropping, "
                "tile indexing, visualization, and evaluation. It must not be used "
                "inside retrieval or verifier ranking."
            ),
        }

    report_path = (
        reports_dir
        / f"s8_7a_map_geotiff_validation_{args.name}.json"
    )
    report_path.write_text(json.dumps(report, indent=2))

    print("S8.7A master map validation and tiling preflight complete")
    print("-------------------------------------------------------")
    print(f"Map:                    {tif_path}")
    print(f"World file:             {detected_world_file}")
    print(f"File size:              {file_size_bytes / (1024 ** 2):.2f} MB")
    print(f"CRS:                    {report['geotiff_info']['crs_normalized']}")
    print(
        f"Raster size:            "
        f"{report['geotiff_info']['width_px']} x "
        f"{report['geotiff_info']['height_px']} px"
    )
    print(
        f"Pixel size:             "
        f"{pixel_size_x:.4f} x {pixel_size_y:.4f}"
    )
    print(f"Target AOI coverage:    {target_coverage_pct:.4f}%")
    print(f"Fully covers target:    {covers_target}")
    print(
        f"Target crop estimate:   "
        f"{target_width_px} x {target_height_px} px"
    )
    print(
        f"Estimated AOI tiles:    "
        f"{target_grid['tile_count']} "
        f"({target_grid['tile_columns']} cols x "
        f"{target_grid['tile_rows']} rows)"
    )
    print(f"Suitability:            {suitability}")
    print(f"Status:                 {status}")
    print(f"Saved report:           {report_path}")


if __name__ == "__main__":
    main()
