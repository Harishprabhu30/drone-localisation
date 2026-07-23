'''
Command Executed:

export PYTHONPATH=$PWD/src
python scripts/villoc/s8_7a_validate_map_geotiff.py \
  --config configs/dataset_villoc_90deg.yaml \
  --tif data/raw/villoc/90_deg/maps/ort10lt_2024_2026/export_001/output.tif \
  --name ort10lt_2024_2026_export_001

'''

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def read_with_rasterio(tif_path: Path) -> dict:
    import rasterio

    with rasterio.open(tif_path) as src:
        bounds = src.bounds
        transform = src.transform

        return {
            "read_method": "rasterio",
            "path": str(tif_path),
            "width_px": int(src.width),
            "height_px": int(src.height),
            "count_bands": int(src.count),
            "dtype": str(src.dtypes[0]) if src.dtypes else None,
            "crs": str(src.crs),
            "bounds": {
                "left": float(bounds.left),
                "bottom": float(bounds.bottom),
                "right": float(bounds.right),
                "top": float(bounds.top),
            },
            "transform": list(transform)[:6],
            "pixel_size_x_m": abs(float(transform.a)),
            "pixel_size_y_m": abs(float(transform.e)),
            "width_m": float(bounds.right - bounds.left),
            "height_m": float(bounds.top - bounds.bottom),
        }


def load_s8_6_plan(output_root: Path) -> dict:
    plan_path = output_root / "reports" / "s8_6_map_bbox_plan.json"
    if not plan_path.exists():
        raise FileNotFoundError(plan_path)
    return json.loads(plan_path.read_text())


def bbox_contains(container: dict, inner: dict) -> bool:
    return (
        container["left"] <= inner["xmin"]
        and container["right"] >= inner["xmax"]
        and container["bottom"] <= inner["ymin"]
        and container["top"] >= inner["ymax"]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--tif", required=True)
    parser.add_argument("--name", default="ort10lt_2024_2026_export_001")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    output_root = Path(cfg["dataset"]["output_root"])
    reports_dir = output_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    tif_path = Path(args.tif)
    if not tif_path.exists():
        raise FileNotFoundError(tif_path)

    plan = load_s8_6_plan(output_root)
    target_bbox_3346 = plan["padded_bboxes_3346"]["300m"]

    info = read_with_rasterio(tif_path)

    exported_bounds = info["bounds"]
    covers_target = bbox_contains(exported_bounds, target_bbox_3346)

    target_width_m = target_bbox_3346["xmax"] - target_bbox_3346["xmin"]
    target_height_m = target_bbox_3346["ymax"] - target_bbox_3346["ymin"]

    # Suitability based on approximate map pixel size.
    px = max(info["pixel_size_x_m"], info["pixel_size_y_m"])
    if px <= 0.5:
        suitability = "GOOD_FOR_FIRST_TILE_MATCHING"
    elif px <= 1.0:
        suitability = "USABLE_BUT_NOT_IDEAL"
    elif px <= 2.0:
        suitability = "WEAK_LOW_DETAIL"
    else:
        suitability = "TOO_COARSE_REEXPORT_NEEDED"

    validation = {
        "stage": "S8.7A",
        "map_name": args.name,
        "status": "VALIDATED_REEXPORT_RECOMMENDED" if px > 1.0 else "VALIDATED_USABLE_CANDIDATE",
        "geotiff_info": info,
        "target_s8_6_300m_bbox_3346": target_bbox_3346,
        "target_width_m": target_width_m,
        "target_height_m": target_height_m,
        "export_covers_target_bbox": covers_target,
        "max_pixel_size_m": px,
        "suitability": suitability,
        "interpretation": {
            "main": (
                "The GeoTIFF is georeferenced and covers the requested area, "
                "but pixel size controls whether it is useful for image matching."
            ),
            "current_warning": (
                "If pixel size is several metres per pixel, the export is too coarse "
                "for ORB/LightGlue/DINO tile matching even if it is georeferenced."
            ),
        },
        "recommended_reexport": {
            "bbox_epsg3346": target_bbox_3346,
            "preferred_output_size_px": [4096, 4096],
            "acceptable_output_size_px": [2048, 2048],
            "target_pixel_size_m_per_px": "0.25-0.6 m/px preferred for first tests",
            "source": "ORT10LT 2024-2026",
            "crs": "EPSG:3346 / LKS-94",
        },
    }

    out_report = reports_dir / f"s8_7a_map_geotiff_validation_{args.name}.json"
    out_report.write_text(json.dumps(validation, indent=2))

    print("S8.7A map GeoTIFF validation complete")
    print("------------------------------------")
    print(f"GeoTIFF:              {tif_path}")
    print(f"CRS:                  {info['crs']}")
    print(f"Size px:              {info['width_px']} x {info['height_px']}")
    print(f"Bounds:               {info['bounds']}")
    print(f"Width/height m:        {info['width_m']:.2f} x {info['height_m']:.2f}")
    print(f"Pixel size m/px:       {info['pixel_size_x_m']:.3f} x {info['pixel_size_y_m']:.3f}")
    print(f"Covers S8.6 300m AOI:  {covers_target}")
    print(f"Suitability:           {suitability}")
    print(f"Saved report:          {out_report}")


if __name__ == "__main__":
    main()
