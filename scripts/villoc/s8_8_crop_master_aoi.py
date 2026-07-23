'''
Command Executed:

export PYTHONPATH=$PWD/src

python scripts/villoc/s8_8_crop_master_aoi.py \
  --config configs/dataset_villoc_90deg.yaml \
  --src-tif data/raw/villoc/90_deg/maps/ort10lt_2024_2026/master/ort10lt_2024_2026_master.tif \
  --output-name ort10lt_2024_2026_aoi300m.tif

'''

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
import yaml
from rasterio.crs import CRS
from rasterio.windows import Window, from_bounds
from rasterio.warp import transform_bounds


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def transform_target_bbox(
    bbox_3346: dict,
    raster_crs: CRS,
) -> tuple[float, float, float, float]:
    source_crs = CRS.from_epsg(3346)

    if raster_crs == source_crs:
        return (
            float(bbox_3346["xmin"]),
            float(bbox_3346["ymin"]),
            float(bbox_3346["xmax"]),
            float(bbox_3346["ymax"]),
        )

    return transform_bounds(
        source_crs,
        raster_crs,
        bbox_3346["xmin"],
        bbox_3346["ymin"],
        bbox_3346["xmax"],
        bbox_3346["ymax"],
        densify_pts=21,
    )


def clamp_window(window: Window, width: int, height: int) -> Window:
    col_off = max(0, int(np.floor(window.col_off)))
    row_off = max(0, int(np.floor(window.row_off)))

    col_end = min(width, int(np.ceil(window.col_off + window.width)))
    row_end = min(height, int(np.ceil(window.row_off + window.height)))

    return Window(
        col_off=col_off,
        row_off=row_off,
        width=max(0, col_end - col_off),
        height=max(0, row_end - row_off),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--src-tif", required=True)
    parser.add_argument("--output-name", default="ort10lt_2024_2026_aoi300m.tif")
    parser.add_argument(
        "--output-dir",
        default="data/processed/villoc/90_deg/maps/ort10lt_2024_2026",
    )
    parser.add_argument("--jpeg-quality", type=int, default=90)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    output_root = Path(cfg["dataset"]["output_root"])

    src_tif = Path(args.src_tif)
    if not src_tif.exists():
        raise FileNotFoundError(src_tif)

    map_plan_path = output_root / "reports" / "s8_6_map_bbox_plan.json"
    plan = load_json(map_plan_path)
    bbox_3346 = plan["padded_bboxes_3346"]["300m"]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_tif = output_dir / args.output_name

    reports_dir = output_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    with rasterio.open(src_tif) as src:
        if src.crs is None:
            raise RuntimeError("Source raster has no CRS.")

        left, bottom, right, top = transform_target_bbox(
            bbox_3346,
            src.crs,
        )

        requested_window = from_bounds(
            left,
            bottom,
            right,
            top,
            transform=src.transform,
        )

        window = clamp_window(
            requested_window,
            src.width,
            src.height,
        )

        if window.width <= 0 or window.height <= 0:
            raise RuntimeError("Computed crop window is empty.")

        data = src.read(window=window)
        crop_transform = src.window_transform(window)

        profile = src.profile.copy()
        profile.update(
            {
                "driver": "GTiff",
                "width": int(window.width),
                "height": int(window.height),
                "transform": crop_transform,
                "crs": src.crs,
                "count": src.count,
                "dtype": src.dtypes[0],
                "tiled": True,
                "blockxsize": 512,
                "blockysize": 512,
                "compress": "JPEG",
                "jpeg_quality": args.jpeg_quality,
                "BIGTIFF": "IF_SAFER",
            }
        )

        # JPEG compression with 4-band RGBA TIFF is not always supported cleanly.
        # If the fourth band is alpha, write only RGB to keep the crop compact
        # and compatible with OpenCV / retrieval scripts.
        write_data = data
        output_bands = src.count
        dropped_alpha = False

        if src.count == 4:
            write_data = data[:3]
            output_bands = 3
            dropped_alpha = True
            profile["count"] = 3

        with rasterio.open(out_tif, "w", **profile) as dst:
            dst.write(write_data)

            dst.update_tags(
                source_stage="S8.8",
                source_raster=str(src_tif),
                source_map="ORT10LT 2024-2026",
                crop_policy="S8.6 300m padded AOI",
                reference_usage="AOI preparation and evaluation only",
            )

        crop_bounds = rasterio.windows.bounds(
            window,
            src.transform,
        )

        pixel_size_x = abs(float(crop_transform.a))
        pixel_size_y = abs(float(crop_transform.e))

        summary = {
            "stage": "S8.8",
            "status": "PASS_CANDIDATE",
            "source_raster": str(src_tif),
            "output_raster": str(out_tif),
            "source_crs": src.crs.to_string(),
            "requested_bbox_epsg3346": bbox_3346,
            "requested_bbox_in_source_crs": {
                "left": left,
                "bottom": bottom,
                "right": right,
                "top": top,
            },
            "crop_window": {
                "col_off": int(window.col_off),
                "row_off": int(window.row_off),
                "width": int(window.width),
                "height": int(window.height),
            },
            "crop_bounds": {
                "left": float(crop_bounds[0]),
                "bottom": float(crop_bounds[1]),
                "right": float(crop_bounds[2]),
                "top": float(crop_bounds[3]),
            },
            "crop_width_px": int(window.width),
            "crop_height_px": int(window.height),
            "output_bands": output_bands,
            "source_bands": src.count,
            "dropped_alpha_band": dropped_alpha,
            "dtype": src.dtypes[0],
            "pixel_size_x": pixel_size_x,
            "pixel_size_y": pixel_size_y,
            "compression": "JPEG",
            "jpeg_quality": args.jpeg_quality,
            "output_file_size_bytes": out_tif.stat().st_size,
            "output_file_size_mb": out_tif.stat().st_size / (1024 ** 2),
            "reference_rule": (
                "SRT-derived coordinates were used only to define the crop AOI. "
                "They must not be used for retrieval or verifier ranking."
            ),
        }

    report_path = (
        reports_dir
        / "s8_8_master_aoi_crop_summary.json"
    )
    report_path.write_text(json.dumps(summary, indent=2))

    print("S8.8 master orthophoto AOI crop complete")
    print("---------------------------------------")
    print(f"Source:             {src_tif}")
    print(f"Output:             {out_tif}")
    print(
        f"Crop size:          "
        f"{summary['crop_width_px']} x "
        f"{summary['crop_height_px']} px"
    )
    print(
        f"Pixel size:         "
        f"{pixel_size_x:.4f} x {pixel_size_y:.4f}"
    )
    print(f"Output bands:       {output_bands}")
    print(f"Dropped alpha:      {dropped_alpha}")
    print(
        f"Output file size:   "
        f"{summary['output_file_size_mb']:.2f} MB"
    )
    print(f"Saved report:       {report_path}")


if __name__ == "__main__":
    main()
