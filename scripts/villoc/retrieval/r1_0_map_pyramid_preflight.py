#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import rasterio
import yaml


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def axis_starts(length_px: int, tile_size_px: int, stride_px: int) -> list[int]:
    if length_px <= 0:
        raise ValueError("Raster axis length must be positive.")
    if tile_size_px <= 0:
        raise ValueError("tile_size_px must be positive.")
    if stride_px <= 0:
        raise ValueError("stride_px must be positive.")
    if stride_px > tile_size_px:
        raise ValueError("stride_px cannot exceed tile_size_px.")
    if length_px < tile_size_px:
        raise ValueError(
            f"Raster axis {length_px}px is smaller than tile size {tile_size_px}px."
        )

    last = length_px - tile_size_px
    starts = list(range(0, last + 1, stride_px))
    if not starts:
        starts = [0]
    if starts[-1] != last:
        starts.append(last)
    return starts


def validate_existing_index(
    *,
    level: dict,
    source_gsd_x: float,
    source_gsd_y: float,
) -> dict:
    path = Path(level["index_csv"])
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    required = {
        "tile_id",
        "pixel_col_off",
        "pixel_row_off",
        "tile_width_px",
        "tile_height_px",
        "center_easting",
        "center_northing",
        "ground_width_m",
        "ground_height_m",
    }
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(
            f"{level['name']}: existing index missing columns {sorted(missing)}"
        )

    expected_tile = int(level["tile_size_px"])

    if not (df["tile_width_px"].astype(int) == expected_tile).all():
        raise RuntimeError(f"{level['name']}: tile width does not match protocol.")
    if not (df["tile_height_px"].astype(int) == expected_tile).all():
        raise RuntimeError(f"{level['name']}: tile height does not match protocol.")

    median_width = float(df["ground_width_m"].median())
    median_height = float(df["ground_height_m"].median())

    expected_width = expected_tile * source_gsd_x
    expected_height = expected_tile * source_gsd_y

    if abs(median_width - expected_width) > 1e-6:
        raise RuntimeError(
            f"{level['name']}: ground width mismatch "
            f"{median_width} vs {expected_width}"
        )
    if abs(median_height - expected_height) > 1e-6:
        raise RuntimeError(
            f"{level['name']}: ground height mismatch "
            f"{median_height} vs {expected_height}"
        )

    return {
        "index_csv": str(path),
        "index_sha256": sha256_file(path),
        "tile_count": int(len(df)),
        "ground_width_m_median": median_width,
        "ground_height_m_median": median_height,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "R1.0 preflight for the fixed-stride Villoc ground-footprint pyramid."
        )
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path(
            "configs/research/retrieval_candidate_pool_v2_r1.yaml"
        ),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path(
            "outputs/research_runs/retrieval_candidate_pool_v2/"
            "r1_map_pyramid/reports/r1_0_map_pyramid_preflight.json"
        ),
    )
    args = parser.parse_args()

    protocol_path = args.protocol.expanduser().resolve()
    cfg = yaml.safe_load(protocol_path.read_text())

    control = cfg["control"]
    experiment = cfg["experiment"]

    source_tif = Path(control["source_tif"]).expanduser().resolve()
    query_cache = Path(control["query_cache"]).expanduser().resolve()

    if not source_tif.exists():
        raise FileNotFoundError(source_tif)
    if not query_cache.exists():
        raise FileNotFoundError(query_cache)

    with rasterio.open(source_tif) as src:
        if src.crs is None:
            raise RuntimeError("Source raster has no CRS.")

        crs = src.crs.to_string()
        if crs != str(control["crs_expected"]):
            raise RuntimeError(
                f"CRS mismatch: actual={crs}, expected={control['crs_expected']}"
            )

        gsd_x = abs(float(src.transform.a))
        gsd_y = abs(float(src.transform.e))
        expected_gsd = float(control["source_gsd_m_per_px_expected"])

        if abs(gsd_x - expected_gsd) > 1e-6:
            raise RuntimeError(
                f"Unexpected source GSD x: {gsd_x} vs {expected_gsd}"
            )
        if abs(gsd_y - expected_gsd) > 1e-6:
            raise RuntimeError(
                f"Unexpected source GSD y: {gsd_y} vs {expected_gsd}"
            )

        raster_width = int(src.width)
        raster_height = int(src.height)

    levels = []
    fixed_stride = int(experiment["fixed_stride_px"])

    for level in experiment["levels"]:
        name = str(level["name"])
        tile = int(level["tile_size_px"])
        stride = int(level["stride_px"])

        if stride != fixed_stride:
            raise RuntimeError(
                f"{name}: stride {stride} violates fixed R1 stride {fixed_stride}"
            )

        expected_footprint = float(level["expected_ground_footprint_m"])
        actual_footprint_x = tile * gsd_x
        actual_footprint_y = tile * gsd_y

        if abs(actual_footprint_x - expected_footprint) > 1e-6:
            raise RuntimeError(
                f"{name}: footprint mismatch "
                f"{actual_footprint_x} vs {expected_footprint}"
            )
        if abs(actual_footprint_y - expected_footprint) > 1e-6:
            raise RuntimeError(
                f"{name}: footprint mismatch "
                f"{actual_footprint_y} vs {expected_footprint}"
            )

        xs = axis_starts(raster_width, tile, stride)
        ys = axis_starts(raster_height, tile, stride)

        item = {
            "name": name,
            "status": str(level["status"]),
            "tile_size_px": tile,
            "stride_px": stride,
            "ground_footprint_m": actual_footprint_x,
            "ground_stride_m": stride * gsd_x,
            "overlap_px": tile - stride,
            "overlap_fraction": float((tile - stride) / tile),
            "grid_columns_expected": len(xs),
            "grid_rows_expected": len(ys),
            "tile_count_expected": len(xs) * len(ys),
            "right_edge_anchored": xs[-1] != (len(xs) - 1) * stride,
            "bottom_edge_anchored": ys[-1] != (len(ys) - 1) * stride,
            "tiles_dir": str(level["tiles_dir"]),
            "index_csv": str(level["index_csv"]),
            "map_cache": str(level["map_cache"]),
        }

        if level["status"] in {
            "frozen_control_reuse",
            "existing_endpoint_reuse",
        }:
            existing = validate_existing_index(
                level=level,
                source_gsd_x=gsd_x,
                source_gsd_y=gsd_y,
            )
            item["existing_index"] = existing

            if existing["tile_count"] != item["tile_count_expected"]:
                raise RuntimeError(
                    f"{name}: existing tile count {existing['tile_count']} "
                    f"!= expected {item['tile_count_expected']}"
                )

            map_cache = Path(level["map_cache"])
            if not map_cache.exists():
                raise FileNotFoundError(map_cache)

            item["map_cache_sha256"] = sha256_file(map_cache)

        levels.append(item)

    report = {
        "stage": "R1.0",
        "status": "PASS_R1_MAP_PYRAMID_PREFLIGHT",
        "created_at_utc": now_utc(),
        "protocol": str(protocol_path),
        "protocol_sha256": sha256_file(protocol_path),
        "source_raster": {
            "path": str(source_tif),
            "sha256": sha256_file(source_tif),
            "crs": crs,
            "width_px": raster_width,
            "height_px": raster_height,
            "gsd_x_m_per_px": gsd_x,
            "gsd_y_m_per_px": gsd_y,
        },
        "query_control": {
            "cache": str(query_cache),
            "sha256": sha256_file(query_cache),
            "query_count_expected": int(control["query_count_expected"]),
        },
        "fixed_variables": {
            "query_representation": cfg["retrieval_protocol"],
            "map_source_gsd_m_per_px": expected_gsd,
            "stride_px": fixed_stride,
            "stride_m": fixed_stride * gsd_x,
        },
        "independent_variable": experiment["independent_variable"],
        "levels": levels,
        "scope_guarantees": {
            "query_cache_rebuilt": False,
            "query_preprocessing_changed": False,
            "map_source_changed": False,
            "map_gsd_changed": False,
            "retrieval_backend_changed": False,
            "orb_policy_changed": False,
            "bootstrap_changed": False,
            "temporal_authority_changed": False,
        },
        "next_stage": {
            "stage": "R1.1",
            "action": (
                "Generate only missing 384_s256 and 768_s256 tile databases "
                "with the proven S8.9 tiler, then audit their geometry before "
                "descriptor encoding."
            ),
        },
    }

    report_path = args.report.expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("=" * 88)
    print("R1.0 — GROUND-FOOTPRINT MAP PYRAMID PREFLIGHT")
    print("=" * 88)
    print("status:", report["status"])
    print("source raster:", source_tif)
    print("source CRS:", crs)
    print("source GSD:", gsd_x, "x", gsd_y, "m/px")
    print("fixed stride:", fixed_stride, "px =", fixed_stride * gsd_x, "m")
    print()

    for level in levels:
        print(
            f"{level['name']:10s} "
            f"footprint={level['ground_footprint_m']:.2f} m "
            f"stride={level['ground_stride_m']:.2f} m "
            f"tiles={level['tile_count_expected']} "
            f"status={level['status']}"
        )

    print()
    print("report:", report_path)


if __name__ == "__main__":
    main()
