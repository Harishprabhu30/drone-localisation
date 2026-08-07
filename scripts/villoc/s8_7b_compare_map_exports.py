'''
Command Executed:

export PYTHONPATH=$PWD/src

python scripts/villoc/s8_7b_compare_map_exports.py \
  --config configs/dataset_villoc_90deg.yaml \
  --map-root data/raw/villoc/90_deg/maps/ort10lt_2024_2026

'''

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import rasterio
import yaml


def overlap_area(a: dict, b: dict) -> float:
    left = max(a["left"], b["xmin"])
    right = min(a["right"], b["xmax"])
    bottom = max(a["bottom"], b["ymin"])
    top = min(a["top"], b["ymax"])

    if right <= left or top <= bottom:
        return 0.0
    return float((right - left) * (top - bottom))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--map-root",
        default="data/raw/villoc/90_deg/maps/ort10lt_2024_2026",
    )
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    output_root = Path(cfg["dataset"]["output_root"])

    plan_path = output_root / "reports" / "s8_6_map_bbox_plan.json"
    plan = json.loads(plan_path.read_text())

    target = plan["padded_bboxes_3346"]["300m"]
    target_area = float((target["xmax"] - target["xmin"]) * (target["ymax"] - target["ymin"]))

    rows = []
    map_root = Path(args.map_root)

    tif_paths = list(map_root.glob("export_*/output.tif"))
    tif_paths.extend(map_root.glob("master/*.tif"))
    tif_paths = sorted(set(tif_paths))

    for tif_path in tif_paths:
        with rasterio.open(tif_path) as src:
            b = src.bounds
            bounds = {
                "left": float(b.left),
                "right": float(b.right),
                "bottom": float(b.bottom),
                "top": float(b.top),
            }

            area = overlap_area(bounds, target)
            coverage_pct = 100.0 * area / target_area if target_area > 0 else 0.0

            px_x = abs(float(src.transform.a))
            px_y = abs(float(src.transform.e))
            max_px = max(px_x, px_y)

            if max_px <= 0.5:
                suitability = "GOOD"
            elif max_px <= 1.0:
                suitability = "USABLE"
            elif max_px <= 2.0:
                suitability = "WEAK"
            else:
                suitability = "TOO_COARSE"

            rows.append(
                {
                    "export_name": (
                        tif_path.stem
                        if tif_path.parent.name == "master"
                        else tif_path.parent.name
                    ),
                    "tif_path": str(tif_path),
                    "width_px": int(src.width),
                    "height_px": int(src.height),
                    "bands": int(src.count),
                    "crs": str(src.crs),
                    "left": bounds["left"],
                    "right": bounds["right"],
                    "bottom": bounds["bottom"],
                    "top": bounds["top"],
                    "pixel_size_x_m": px_x,
                    "pixel_size_y_m": px_y,
                    "max_pixel_size_m": max_px,
                    "overlap_area_m2": area,
                    "target_coverage_pct": coverage_pct,
                    "suitability": suitability,
                    "covers_any_target": area > 0,
                }
            )

    df = pd.DataFrame(rows)

    out_csv = output_root / "metadata" / "s8_7b_map_export_comparison.csv"
    out_json = output_root / "reports" / "s8_7b_map_export_comparison.json"

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(out_csv, index=False)

    summary = {
        "stage": "S8.7B",
        "map_root": str(map_root),
        "target_bbox_3346": target,
        "target_area_m2": target_area,
        "exports_found": int(len(df)),
        "exports_with_any_overlap": int(df["covers_any_target"].sum()) if len(df) else 0,
        "max_single_export_coverage_pct": float(df["target_coverage_pct"].max()) if len(df) else 0.0,
        "comparison_csv": str(out_csv),
        "rows": rows,
        "interpretation": {
            "export_001": "Coverage reference but too coarse if max_pixel_size is several metres.",
            "export_002": "Resolution candidate but not useful unless it overlaps the target AOI.",
            "next": "Collect enough high-resolution overlapping exports to cover the 300 m AOI, then mosaic or tile.",
        },
    }

    out_json.write_text(json.dumps(summary, indent=2))

    print("S8.7B map export comparison complete")
    print("-----------------------------------")
    print(f"Exports found:              {len(df)}")
    if len(df):
        print(f"Exports with target overlap: {int(df['covers_any_target'].sum())}")
        print(f"Max single coverage:         {df['target_coverage_pct'].max():.2f}%")
        print()
        print(df[[
            "export_name",
            "width_px",
            "height_px",
            "max_pixel_size_m",
            "target_coverage_pct",
            "suitability",
            "covers_any_target",
        ]].to_string(index=False))
    print(f"\nSaved CSV:  {out_csv}")
    print(f"Saved JSON: {out_json}")


if __name__ == "__main__":
    main()
