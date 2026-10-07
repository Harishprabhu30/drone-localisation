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
from pyproj import CRS, Transformer


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _float_close(a: float, b: float, atol: float = 1e-10) -> bool:
    return abs(float(a) - float(b)) <= atol


def _projection_contract(crs: CRS) -> dict:
    """Extract the projected-coordinate semantics needed by the R1 map grid.

    Some historical GeoTIFFs retain correct projection/ellipsoid parameters but
    lose the datum authority/name. In that case pyproj correctly refuses formal
    CRS equality. R1 still needs to know whether the stored projected x/y values
    use the same metric Transverse-Mercator coordinate contract.
    """
    operation = crs.coordinate_operation

    if not crs.is_projected or operation is None:
        return {"is_projected": False}

    params = {
        param.name: {
            "value": float(param.value),
            "unit_conversion_factor": float(param.unit_conversion_factor),
        }
        for param in operation.params
    }

    axes = sorted(
        (
            str(axis.direction).lower(),
            float(axis.unit_conversion_factor),
        )
        for axis in crs.axis_info
    )

    return {
        "is_projected": True,
        "method_name": str(operation.method_name),
        "params": params,
        "ellipsoid_semi_major_m": float(crs.ellipsoid.semi_major_metre),
        "ellipsoid_inverse_flattening": float(
            crs.ellipsoid.inverse_flattening
        ),
        "prime_meridian_longitude_deg": float(
            crs.prime_meridian.longitude
        ),
        "axes": axes,
    }


def _projection_contract_equivalent(actual: CRS, expected: CRS) -> bool:
    a = _projection_contract(actual)
    e = _projection_contract(expected)

    if not a.get("is_projected") or not e.get("is_projected"):
        return False

    if a["method_name"] != e["method_name"]:
        return False

    if set(a["params"]) != set(e["params"]):
        return False

    for name in a["params"]:
        ap = a["params"][name]
        ep = e["params"][name]

        if not _float_close(ap["value"], ep["value"]):
            return False

        if not _float_close(
            ap["unit_conversion_factor"],
            ep["unit_conversion_factor"],
        ):
            return False

    if not _float_close(
        a["ellipsoid_semi_major_m"],
        e["ellipsoid_semi_major_m"],
        atol=1e-6,
    ):
        return False

    if not _float_close(
        a["ellipsoid_inverse_flattening"],
        e["ellipsoid_inverse_flattening"],
        atol=1e-9,
    ):
        return False

    if not _float_close(
        a["prime_meridian_longitude_deg"],
        e["prime_meridian_longitude_deg"],
        atol=1e-12,
    ):
        return False

    # Axis serialization order may differ between historical WKT and EPSG,
    # while both still describe one east/west metric axis and one north/south
    # metric axis. Compare direction/unit semantics independent of order.
    if a["axes"] != e["axes"]:
        return False

    return True


def crs_compatibility(actual, expected: str) -> dict:
    """Classify CRS compatibility without pretending missing metadata exists."""
    actual_crs = CRS.from_user_input(actual)
    expected_crs = CRS.from_user_input(expected)

    actual_epsg = actual_crs.to_epsg()
    expected_epsg = expected_crs.to_epsg()

    if actual_crs == expected_crs or actual_crs.equals(expected_crs):
        mode = "formal_crs_equivalence"
        compatible = True
    elif (
        actual_epsg is not None
        and expected_epsg is not None
        and actual_epsg == expected_epsg
    ):
        mode = "epsg_authority_equivalence"
        compatible = True
    elif _projection_contract_equivalent(actual_crs, expected_crs):
        mode = "projection_contract_equivalence_with_unresolved_datum_authority"
        compatible = True
    else:
        mode = "incompatible"
        compatible = False

    return {
        "compatible": compatible,
        "mode": mode,
        "actual_epsg": actual_epsg,
        "expected_epsg": expected_epsg,
        "formal_crs_equal": bool(actual_crs.equals(expected_crs)),
        "projection_contract_equal": bool(
            _projection_contract_equivalent(actual_crs, expected_crs)
        ),
        "actual_projection_contract": _projection_contract(actual_crs),
        "expected_projection_contract": _projection_contract(expected_crs),
    }


def crs_equivalent(actual, expected: str) -> bool:
    """Backward-compatible boolean wrapper used by tests/callers."""
    return bool(crs_compatibility(actual, expected)["compatible"])


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
        expected_crs = str(control["crs_expected"])

        crs_check = crs_compatibility(
            src.crs,
            expected_crs,
        )

        if not crs_check["compatible"]:
            raise RuntimeError(
                "CRS mismatch: projected-coordinate contract is not "
                "compatible with the R1 expected CRS. "
                f"actual={crs}, expected={expected_crs}"
            )

        # Verify that interpreting AOI projected coordinates through the
        # historical WKT versus EPSG:3346 is identity at representative points.
        actual_crs_obj = CRS.from_user_input(src.crs)
        expected_crs_obj = CRS.from_user_input(expected_crs)
        transformer = Transformer.from_crs(
            actual_crs_obj,
            expected_crs_obj,
            always_xy=True,
        )

        left, bottom, right, top = src.bounds
        crs_probe_points = [
            (float(left), float(bottom)),
            (float(left), float(top)),
            (float(right), float(bottom)),
            (float(right), float(top)),
            (
                float((left + right) / 2.0),
                float((bottom + top) / 2.0),
            ),
        ]

        transformed_points = [
            transformer.transform(x, y)
            for x, y in crs_probe_points
        ]

        crs_identity_max_delta_m = max(
            max(
                abs(float(tx) - float(x)),
                abs(float(ty) - float(y)),
            )
            for (x, y), (tx, ty)
            in zip(crs_probe_points, transformed_points)
        )

        if crs_identity_max_delta_m > 1e-6:
            raise RuntimeError(
                "CRS compatibility check passed structurally but projected "
                "AOI coordinates are not identity-compatible with EPSG:3346. "
                f"max_delta_m={crs_identity_max_delta_m}"
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
            "crs_raw": crs,
            "crs_expected": expected_crs,
            "crs_compatible_for_r1": True,
            "crs_validation_mode": crs_check["mode"],
            "crs_actual_epsg": crs_check["actual_epsg"],
            "crs_expected_epsg": crs_check["expected_epsg"],
            "formal_crs_equal": crs_check["formal_crs_equal"],
            "projection_contract_equal": crs_check[
                "projection_contract_equal"
            ],
            "identity_probe_max_delta_m": crs_identity_max_delta_m,
            "transformer_description": transformer.description,
            "transformer_accuracy": transformer.accuracy,
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
    print("source CRS (raw):", crs)
    print("source CRS expected:", expected_crs)
    print("CRS validation mode:", crs_check["mode"])
    print("source CRS authority EPSG:", crs_check["actual_epsg"])
    print("expected CRS EPSG:", crs_check["expected_epsg"])
    print("projection contract equal:", crs_check["projection_contract_equal"])
    print("AOI identity max delta:", crs_identity_max_delta_m, "m")
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
