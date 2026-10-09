#!/usr/bin/env python3
"""RG2.0: map-index-only deterministic topology preflight. Never opens reference."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import yaml

from rg2_0_topology import load_tiles, topology


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = ROOT / "configs/research/local_map_neighborhood_subtile_consensus_v1.yaml"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with path.open("w", encoding="utf-8", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: (f"{value:.6f}" if isinstance(value, float)
                      else str(value).lower() if isinstance(value, bool)
                      else value)
                for key, value in row.items()
            })


def main() -> None:
    parser = argparse.ArgumentParser(description="RG2.0 deterministic 768_s256 map topology")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()

    cfg_path = resolve(str(args.config))
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if cfg.get("stage") != "RG2.0":
        raise ValueError("Wrong stage configuration")
    if cfg.get("ground_truth_allowed") is not False:
        raise ValueError("RG2.0 must explicitly forbid ground truth")
    if cfg.get("candidate_generation_enabled") is not False:
        raise ValueError("RG2.0 must not run candidate generation")
    if cfg.get("bootstrap_enabled") is not False:
        raise ValueError("RG2.0 must not invoke bootstrap")

    spec_path = resolve(cfg["trajectory_spec"])
    # The adapter's frozen trajectory spec declares map metadata. Only 'map'
    # and 'trajectory.role' are examined; NO reference path is opened.
    spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    variant = cfg["map_variant"]
    if spec["map"]["crs"] != "EPSG:3346":
        raise ValueError("Expected EPSG:3346 map CRS")
    if variant != "768_s256" or spec["map"]["primary_variant"] != variant:
        raise ValueError("Frozen map variant differs from trajectory contract")
    if spec["trajectory"]["role"] != "development":
        raise ValueError("Frozen development trajectory spec role changed")
    tile_index = resolve(spec["map"]["variants"][variant]["tile_index"])
    if not tile_index.is_file():
        raise FileNotFoundError(f"Generated 768_s256 tile index missing: {tile_index}")

    g = cfg["geometry_contract"]
    if g["tile_size_px"] != 768 or g["stride_px"] != 256:
        raise ValueError("Frozen tile-pixel geometry changed")
    if abs(g["pixel_size_m"] - 0.2) > 1e-12:
        raise ValueError("Frozen map resolution changed")
    footprint = g["tile_size_px"] * g["pixel_size_m"]
    stride = g["stride_px"] * g["pixel_size_m"]
    tol = float(g["geometry_tolerance_m"])
    if not (0 < tol < 0.1):
        raise ValueError("Geometry tolerance outside frozen preflight range")

    tiles = load_tiles(tile_index, expected_tile_px=g["tile_size_px"],
                       expected_footprint_m=footprint, tolerance_m=tol)
    tile_rows, pair_rows, summary = topology(
        tiles, stride_m=stride, tolerance_m=tol
    )
    if not pair_rows:
        raise ValueError("No neighboring/overlapping map windows")

    out = resolve(cfg["output_root"])
    out.mkdir(parents=True, exist_ok=True)
    tile_csv = out / "rg2_0_tile_geometry.csv"
    edge_csv = out / "rg2_0_tile_relationships.csv"
    report_json = out / "rg2_0_topology_preflight_report.json"
    report_md = out / "rg2_0_topology_preflight_report.md"
    write_csv(tile_csv, tile_rows)
    write_csv(edge_csv, pair_rows)

    report = {
        "status": "PASS_RG2_0_MAP_TOPOLOGY_PREFLIGHT",
        "stage": "RG2.0",
        "parent_rg1_commit": cfg["parent_rg1_commit"],
        "map_variant": variant,
        "map_crs": "EPSG:3346",
        "footprint_m": footprint,
        "stride_m": stride,
        "geometry_tolerance_m": tol,
        "topology": summary,
        "provenance": {
            "config_sha256": sha256(cfg_path),
            "trajectory_spec_sha256": sha256(spec_path),
            "tile_index_sha256": sha256(tile_index),
            "tile_geometry_sha256": sha256(tile_csv),
            "relationships_sha256": sha256(edge_csv),
        },
        "research_boundary": {
            "query_data_loaded": False,
            "reference_loaded": False,
            "gps_srt_loaded": False,
            "orb_executed": False,
            "projection_executed": False,
            "candidate_generation_executed": False,
            "bootstrap_state_executed": False,
        },
    }
    report_json.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# RG2.0 deterministic topology preflight",
        "",
        f"Status: **{report['status']}**",
        f"Map: \`{variant}\`, CRS \`EPSG:3346\`",
        f"Tiles: **{summary['tile_count']}**",
        f"Overlap pairs (undirected): **{summary['undirected_overlap_pairs']}**",
        f"Near pairs (undirected): **{summary['undirected_near_pairs']}**",
        f"Isolated overlap tiles: **{len(summary['isolated_overlap_tiles'])}**",
        "",
        "## Degree histograms",
        "",
        "Overlap: \`" + json.dumps(summary["overlap_degree_histogram"], sort_keys=True) + "\`",
        "Near: \`" + json.dumps(summary["near_degree_histogram"], sort_keys=True) + "\`",
        "",
        "## AOI bounds / edge behavior",
        "",
        "\`" + json.dumps(summary["aoi_bounds_epsg3346"], sort_keys=True) + "\`",
        "\`" + json.dumps(summary["aoi_edge_tile_counts"], sort_keys=True) + "\`",
        "",
        "## Reproducibility",
        "",
        "Input/output SHA256s and blind-safety booleans are in the JSON report.",
        "Only a local generated map tile index was read. No query/reference/GT.",
        "This is topology only; no localization quality or accuracy conclusion.",
        "",
    ]
    report_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"Status: {report['status']}")
    print(f"Tiles: {summary['tile_count']}")
    print(f"Undirected overlap pairs: {summary['undirected_overlap_pairs']}")
    print(f"Undirected near pairs: {summary['undirected_near_pairs']}")
    print("Overlap-degree histogram:", summary["overlap_degree_histogram"])
    print("Near-degree histogram:", summary["near_degree_histogram"])
    print("AOI-edge tile counts:", summary["aoi_edge_tile_counts"])
    print(f"Reference loaded: False")
    print(f"Report: {report_json}")
    print(f"Readable report: {report_md}")
    print(f"Topology SHA256: {sha256(edge_csv)}")


if __name__ == "__main__":
    main()
