#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml


ROOT = Path.cwd().resolve()


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def run(cmd: list[str]) -> None:
    print()
    print("$", " ".join(cmd))
    subprocess.run(
        cmd,
        cwd=ROOT,
        check=True,
    )


def resolve(path_like: str | Path) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else ROOT / path


def expected_index_for(
    build_output_root: Path,
    tile_size_px: int,
    stride_px: int,
) -> Path:
    return (
        build_output_root
        / "metadata"
        / f"s8_9_satellite_tile_index_{tile_size_px}_s{stride_px}.csv"
    )


def inspect_existing_state(
    *,
    tiles_dir: Path,
    index_csv: Path,
) -> str:
    tiles_exist = tiles_dir.exists()
    index_exists = index_csv.exists()

    tile_files = []
    if tiles_exist:
        tile_files = [
            p
            for p in tiles_dir.iterdir()
            if p.is_file()
        ]

    if not tiles_exist and not index_exists:
        return "absent"

    if tiles_exist and tile_files and index_exists:
        return "complete_candidate"

    if tiles_exist and not tile_files and not index_exists:
        return "empty_dir_only"

    return "partial_or_inconsistent"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "R1.1 generate only missing fixed-stride footprint levels and "
            "run the proven S8.10A geometry audit before descriptor encoding."
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
        "--build-config",
        type=Path,
        default=Path(
            "configs/research/retrieval_candidate_pool_v2_r1_map_build.yaml"
        ),
    )
    parser.add_argument(
        "--image-format",
        choices=["jpg", "png"],
        default="jpg",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=95,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Regenerate research tile levels even when outputs already exist. "
            "Never affects frozen control/reuse levels."
        ),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path(
            "outputs/research_runs/retrieval_candidate_pool_v2/"
            "r1_map_pyramid/reports/r1_1_missing_levels_build_and_audit.json"
        ),
    )
    args = parser.parse_args()

    protocol_path = resolve(args.protocol)
    build_config_path = resolve(args.build_config)

    if not protocol_path.exists():
        raise FileNotFoundError(protocol_path)
    if not build_config_path.exists():
        raise FileNotFoundError(build_config_path)

    protocol = yaml.safe_load(protocol_path.read_text())
    build_cfg = yaml.safe_load(build_config_path.read_text())

    source_tif = resolve(
        protocol["control"]["source_tif"]
    )
    build_output_root = resolve(
        build_cfg["dataset"]["output_root"]
    )

    if not source_tif.exists():
        raise FileNotFoundError(source_tif)

    generate_levels = [
        level
        for level in protocol["experiment"]["levels"]
        if level["status"] == "generate"
    ]

    expected_names = {"384_s256", "768_s256"}
    actual_names = {str(level["name"]) for level in generate_levels}

    if actual_names != expected_names:
        raise RuntimeError(
            "R1.1 is intentionally scoped to the two missing primary levels. "
            f"Expected {sorted(expected_names)}, got {sorted(actual_names)}."
        )

    tiler = (
        ROOT
        / "scripts/villoc/s8_9_generate_reference_tiles.py"
    )
    auditor = (
        ROOT
        / "scripts/villoc/s8_10a_audit_tile_index_integrity.py"
    )

    for script in (tiler, auditor):
        if not script.exists():
            raise FileNotFoundError(script)

    level_records = []

    for level in generate_levels:
        name = str(level["name"])
        tile_size = int(level["tile_size_px"])
        stride = int(level["stride_px"])
        tiles_dir = resolve(level["tiles_dir"])

        index_csv = expected_index_for(
            build_output_root,
            tile_size,
            stride,
        )

        configured_index = resolve(level["index_csv"])

        if index_csv != configured_index:
            raise RuntimeError(
                f"{name}: protocol index path does not match S8.9 build output. "
                f"expected={index_csv}, configured={configured_index}"
            )

        state = inspect_existing_state(
            tiles_dir=tiles_dir,
            index_csv=index_csv,
        )

        if state == "partial_or_inconsistent" and not args.overwrite:
            raise RuntimeError(
                f"{name}: partial/inconsistent existing outputs detected. "
                "Inspect or rerun R1.1 with --overwrite after confirming the "
                "research-only paths are safe to replace."
            )

        if state == "complete_candidate" and not args.overwrite:
            print()
            print(
                f"[SKIP GENERATION] {name}: tiles + index already exist; "
                "geometry audit will decide whether they are reusable."
            )
        else:
            cmd = [
                sys.executable,
                str(tiler),
                "--config",
                str(build_config_path),
                "--src-tif",
                str(source_tif),
                "--tiles-dir",
                str(tiles_dir),
                "--tile-size-px",
                str(tile_size),
                "--stride-px",
                str(stride),
                "--image-format",
                args.image_format,
            ]

            if args.image_format == "jpg":
                cmd.extend(
                    [
                        "--jpeg-quality",
                        str(args.jpeg_quality),
                    ]
                )

            if args.overwrite:
                cmd.append("--overwrite")

            run(cmd)

        level_records.append(
            {
                "name": name,
                "tile_size_px": tile_size,
                "stride_px": stride,
                "tiles_dir": str(tiles_dir),
                "index_csv": str(index_csv),
            }
        )

    audit_cmd = [
        sys.executable,
        str(auditor),
        "--config",
        str(build_config_path),
        "--src-tif",
        str(source_tif),
    ]

    for item in level_records:
        audit_cmd.extend(
            [
                "--variant",
                (
                    f"{item['name']}:"
                    f"{item['tile_size_px']}:"
                    f"{item['stride_px']}:"
                    f"{item['tiles_dir']}:"
                    f"{item['index_csv']}"
                ),
            ]
        )

    run(audit_cmd)

    audit_report_path = (
        build_output_root
        / "reports"
        / "s8_10a_tile_index_integrity_audit.json"
    )

    if not audit_report_path.exists():
        raise RuntimeError(
            f"Expected audit report was not produced: {audit_report_path}"
        )

    audit = json.loads(
        audit_report_path.read_text()
    )

    if audit.get("status") != "PASS_TILE_INDEX_INTEGRITY":
        raise RuntimeError(
            "R1.1 geometry audit failed. Descriptor encoding is blocked. "
            f"See {audit_report_path}"
        )

    level_summaries = []

    for item in level_records:
        index_path = Path(item["index_csv"])
        df = pd.read_csv(index_path)

        image_paths = [
            resolve(p)
            for p in df["tile_path"].astype(str).tolist()
        ]

        missing_images = [
            str(path)
            for path in image_paths
            if not path.exists()
        ]

        if missing_images:
            raise RuntimeError(
                f"{item['name']}: index references missing tile files; "
                f"first={missing_images[0]}"
            )

        level_summaries.append(
            {
                **item,
                "index_sha256": sha256_file(index_path),
                "tile_count": int(len(df)),
                "ground_width_m_median": float(
                    df["ground_width_m"].median()
                ),
                "ground_height_m_median": float(
                    df["ground_height_m"].median()
                ),
                "center_spacing_protocol_m": float(
                    protocol["experiment"]["fixed_stride_expected_m"]
                ),
                "all_indexed_tile_files_present": True,
            }
        )

    report = {
        "stage": "R1.1",
        "status": "PASS_R1_MISSING_LEVELS_GENERATED_AND_AUDITED",
        "created_at_utc": now_utc(),
        "protocol": str(protocol_path),
        "protocol_sha256": sha256_file(protocol_path),
        "build_config": str(build_config_path),
        "build_config_sha256": sha256_file(build_config_path),
        "source_raster": str(source_tif),
        "source_raster_sha256": sha256_file(source_tif),
        "generated_or_reused_research_levels": level_summaries,
        "geometry_audit": {
            "report": str(audit_report_path),
            "report_sha256": sha256_file(audit_report_path),
            "status": audit["status"],
            "variants_passed": audit["variants_passed"],
            "variants_failed": audit["variants_failed"],
            "total_failure_count": audit["total_failure_count"],
        },
        "frozen_assets_modified": False,
        "scope_guarantees": {
            "only_missing_levels_targeted": True,
            "frozen_512_s256_regenerated": False,
            "existing_1024_s256_regenerated": False,
            "query_cache_touched": False,
            "descriptor_encoding_run": False,
            "retrieval_run": False,
            "oracle_used_for_generation": False,
        },
        "next_stage": {
            "stage": "R1.2",
            "action": (
                "Build DINOv2 map descriptor caches only for 384_s256 and "
                "768_s256 after this geometry gate passes."
            ),
        },
    }

    report_path = resolve(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print()
    print("=" * 88)
    print("R1.1 — MISSING MAP LEVELS BUILD + GEOMETRY AUDIT")
    print("=" * 88)
    print("status:", report["status"])

    for item in level_summaries:
        print(
            f"{item['name']:10s} "
            f"tiles={item['tile_count']} "
            f"footprint={item['ground_width_m_median']:.2f} m "
            f"index_sha256={item['index_sha256'][:12]}..."
        )

    print("audit:", audit_report_path)
    print("report:", report_path)


if __name__ == "__main__":
    main()
