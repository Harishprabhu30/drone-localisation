#!/usr/bin/env python3

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def run(command):
    print()
    print("$", " ".join(str(x) for x in command))
    subprocess.run(
        [str(x) for x in command],
        check=True,
    )


def load_backend(config_path: Path) -> str:
    cfg = yaml.safe_load(
        config_path.read_text()
    )

    return str(
        cfg.get(
            "bootstrap",
            {},
        ).get(
            "backend",
            "legacy_strict",
        )
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--repo-root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--config",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--fused-trajectory",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--fusion-report",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--absolute-query-summary",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--run-root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--tile-index",
        type=Path,
    )

    parser.add_argument(
        "--source-crs",
        default="EPSG:3346",
    )

    parser.add_argument(
        "--target-crs",
        default="EPSG:4326",
    )

    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    run_root = args.run_root.resolve()

    backend = load_backend(
        args.config.resolve()
    )

    require(
        backend in {
            "legacy_strict",
            "minimum_confident_v2",
        },
        f"Unsupported bootstrap backend: {backend}",
    )

    print()
    print("=" * 100)
    print("STAGE 10 — ESTIMATED OUTPUT ROUTER")
    print("=" * 100)
    print("backend:", backend)

    if backend == "minimum_confident_v2":

        script = (
            repo_root
            / "scripts/villoc/blind_demo/"
              "addon9_estimated_latlon_export_canonical.py"
        )

    else:

        script = (
            repo_root
            / "scripts/villoc/blind_demo/"
              "addon9_estimated_latlon_export.py"
        )

    command = [
        sys.executable,
        script,

        "--fused-trajectory",
        args.fused_trajectory.resolve(),

        "--fusion-report",
        args.fusion_report.resolve(),

        "--absolute-query-summary",
        args.absolute_query_summary.resolve(),

        "--run-root",
        run_root,

        "--source-crs",
        args.source_crs,

        "--target-crs",
        args.target_crs,
    ]

    if args.tile_index is not None:

        command.extend(
            [
                "--tile-index",
                args.tile_index.resolve(),
            ]
        )

    run(command)

    expected = [
        (
            run_root
            / "trajectories/"
              "submission_estimated_trajectory.csv"
        ),
        (
            run_root
            / "reports/addon9_estimated_latlon/"
              "estimated_latlon_export_report.json"
        ),
    ]

    for path in expected:
        require(
            path.exists(),
            f"Stage-10 output missing: {path}",
        )

    print()
    print(
        "STATUS: PASS_STAGE10_ESTIMATED_OUTPUT_ROUTER"
    )


if __name__ == "__main__":
    main()
