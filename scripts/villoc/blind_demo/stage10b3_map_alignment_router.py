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
        "--blind-manifest",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--raw-relative",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--bootstrap-report",
        type=Path,
    )

    parser.add_argument(
        "--run-root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--map-crs",
        default="EPSG:3346",
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
    print("STAGE 08 — MAP ALIGNMENT ROUTER")
    print("=" * 100)
    print("backend:", backend)

    if backend == "minimum_confident_v2":

        timeline = (
            run_root
            / "reports/blind_map_bootstrap/"
              "canonical_map_state_timeline.json"
        )

        require(
            timeline.exists(),
            f"Missing canonical timeline: {timeline}",
        )

        run(
            [
                sys.executable,
                repo_root
                / "scripts/villoc/blind_demo/"
                  "stage10b3_apply_canonical_map_state.py",

                "--map-state-timeline",
                timeline,

                "--relative-trajectory",
                args.raw_relative.resolve(),

                "--manifest",
                args.blind_manifest.resolve(),

                "--output-root",
                run_root,
            ]
        )

    else:

        require(
            args.bootstrap_report is not None,
            (
                "legacy_strict requires "
                "--bootstrap-report."
            ),
        )

        run(
            [
                sys.executable,
                repo_root
                / "scripts/villoc/blind_demo/"
                  "stage10b3_apply_blind_map_lock.py",

                "--blind-manifest",
                args.blind_manifest.resolve(),

                "--raw-relative",
                args.raw_relative.resolve(),

                "--bootstrap-report",
                args.bootstrap_report.resolve(),

                "--run-root",
                run_root,

                "--map-crs",
                args.map_crs,
            ]
        )

    output = (
        run_root
        / "trajectories/"
          "blind_map_aligned_relative_trajectory.csv"
    )

    require(
        output.exists(),
        f"Stage-08 output missing: {output}",
    )

    print()
    print("output:", output)
    print(
        "STATUS: PASS_STAGE08_MAP_ALIGNMENT_ROUTER"
    )


if __name__ == "__main__":
    main()
