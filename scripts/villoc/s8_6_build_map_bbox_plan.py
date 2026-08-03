'''
Command Executed:

export PYTHONPATH=$PWD/src
python scripts/villoc/s8_6_build_map_bbox_plan.py \
  --config configs/dataset_villoc_90deg.yaml \
  --stream V \
  --sample-rate-fps 1

2. running traj01 villoc dataset:

mkdir -p outputs/villoc/traj01_90deg_stable120m/logs/s8_6_map_bbox

python scripts/villoc/s8_6_build_map_bbox_plan.py \
  --config configs/dataset_villoc_traj01_90deg_stable120m.yaml \
  --stream V \
  --sample-rate-fps 1 \
  2>&1 | tee \
  outputs/villoc/traj01_90deg_stable120m/logs/s8_6_map_bbox/s8_6_map_bbox_plan_V_1fps.log

'''

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd
import yaml


def meters_to_lat_deg(m: float) -> float:
    return m / 111_320.0


def meters_to_lon_deg(m: float, lat_deg: float) -> float:
    return m / (111_320.0 * math.cos(math.radians(lat_deg)))


def padded_bbox_from_latlon(df: pd.DataFrame, pad_m: float) -> dict:
    lat_min = float(df["lat"].min())
    lat_max = float(df["lat"].max())
    lon_min = float(df["lon"].min())
    lon_max = float(df["lon"].max())
    lat_mid = 0.5 * (lat_min + lat_max)

    dlat = meters_to_lat_deg(pad_m)
    dlon = meters_to_lon_deg(pad_m, lat_mid)

    return {
        "pad_m": pad_m,
        "south": lat_min - dlat,
        "north": lat_max + dlat,
        "west": lon_min - dlon,
        "east": lon_max + dlon,
        "crs": "EPSG:4326",
    }


def try_epsg3346_bbox(bbox_4326: dict) -> dict | None:
    try:
        from pyproj import Transformer
    except Exception:
        return None

    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3346", always_xy=True)

    corners = [
        (bbox_4326["west"], bbox_4326["south"]),
        (bbox_4326["west"], bbox_4326["north"]),
        (bbox_4326["east"], bbox_4326["south"]),
        (bbox_4326["east"], bbox_4326["north"]),
    ]

    xs, ys = [], []
    for lon, lat in corners:
        x, y = transformer.transform(lon, lat)
        xs.append(x)
        ys.append(y)

    return {
        "xmin": min(xs),
        "ymin": min(ys),
        "xmax": max(xs),
        "ymax": max(ys),
        "crs": "EPSG:3346",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--stream", default="V")
    parser.add_argument("--sample-rate-fps", type=float, default=1.0)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    dataset_cfg = cfg["dataset"]
    output_root = Path(dataset_cfg["output_root"])

    dataset_name = dataset_cfg.get("name") or output_root.name
    sequence_name = dataset_cfg.get("sequence_name") or dataset_cfg.get("folder_name") or output_root.name

    index_csv = (
        output_root
        / "metadata"
        / f"s8_5_uav_frames_index_{args.stream.lower()}_{args.sample_rate_fps:g}fps.csv"
    )

    if not index_csv.exists():
        raise FileNotFoundError(index_csv)

    df = pd.read_csv(index_csv)

    raw_bbox_4326 = {
        "south": float(df["lat"].min()),
        "north": float(df["lat"].max()),
        "west": float(df["lon"].min()),
        "east": float(df["lon"].max()),
        "crs": "EPSG:4326",
    }

    padded = {
        "100m": padded_bbox_from_latlon(df, 100.0),
        "200m": padded_bbox_from_latlon(df, 200.0),
        "300m": padded_bbox_from_latlon(df, 300.0),
    }

    padded_3346 = {}
    for key, bbox in padded.items():
        bbox3346 = try_epsg3346_bbox(bbox)
        if bbox3346 is not None:
            padded_3346[key] = bbox3346

    plan = {
        "stage": "S8.6",
        "status": "MAP_SOURCE_PLAN_DRAFT",
        "dataset_name": dataset_name,
        "sequence_name": sequence_name,
        "uav_index_csv": str(index_csv),
        "uav_rows": int(len(df)),
        "flight_bbox_4326": raw_bbox_4326,
        "padded_bboxes_4326": padded,
        "padded_bboxes_3346": padded_3346,
        "recommended_first_aoi": "300m",
        "recommended_first_map_source": {
            "name": "Geoportal ORT10LT 2024-2026",
            "type": "ArcGIS MapServer / WMS view service",
            "preferred_crs": "EPSG:3346",
            "status": "candidate_public_source",
            "notes": [
                "Use public orthophoto for first real-data map matching.",
                "Manual PNG exports are visual previews only unless exact bbox and CRS are saved.",
                "Prefer reproducible REST/QGIS export over screenshots.",
            ],
        },
        "manual_png_exports": {
            "status": "preview_only_until_georeferenced",
            "provided_scales": ["1:10000", "1:5000"],
            "required_before_algorithm_use": [
                "exact bbox",
                "CRS",
                "image width",
                "image height",
                "source layer/year",
                "export date",
            ],
        },
        "tile_plan": {
            "tile_size_px": 512,
            "stride_px": 256,
            "overlap_percent": 50,
            "first_uav_test_manifest": str(output_root / "metadata" / f"s8_5_golden20_manifest_{args.stream.lower()}_{args.sample_rate_fps:g}fps.csv"),
            "final_uav_index": str(index_csv),
        },
        "gt_leakage_rule": "Use SRT lat/lon/ENU only for AOI definition, tile metadata, visualization, and evaluation. Never use it for retrieval ranking, verifier ranking, correction acceptance, or threshold tuning.",
        "open_questions": [
            "Can Geoportal export exact bbox coordinates for the PNG?",
            "Can QGIS connect to the ORT10LT service and export a georeferenced raster?",
            "Can ArcGIS REST Export Map be used directly for the 300m AOI?",
            "Are there license/caching restrictions for local experimental tile generation?",
            "Will the team allow public map screenshots in the report, or should location be anonymized?",
        ],
    }

    out_report = output_root / "reports" / "s8_6_map_bbox_plan.json"
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_report.write_text(json.dumps(plan, indent=2))

    print("S8.6 map bbox/source plan complete")
    print("---------------------------------")
    print(f"UAV rows:       {len(df)}")
    print(f"Raw bbox:       {raw_bbox_4326}")
    print(f"300m bbox 4326: {padded['300m']}")
    if "300m" in padded_3346:
        print(f"300m bbox 3346: {padded_3346['300m']}")
    print(f"Saved:          {out_report}")


if __name__ == "__main__":
    main()
