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
        "--map-trajectory",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--bootstrap-report",
        type=Path,
    )

    parser.add_argument(
        "--absolute-query-summary",
        type=Path,
    )

    parser.add_argument(
        "--absolute-candidate-scores",
        type=Path,
    )

    parser.add_argument(
        "--run-root",
        type=Path,
        required=True,
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
    print("STAGE 09 — TEMPORAL AUTHORITY ROUTER")
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
                  "stage10b4_temporal_authority_router.py",

                "--map-trajectory",
                args.map_trajectory.resolve(),

                "--map-state-timeline",
                timeline,

                "--run-root",
                run_root,
            ]
        )

    else:

        for value, name in [
            (
                args.bootstrap_report,
                "--bootstrap-report",
            ),
            (
                args.absolute_query_summary,
                "--absolute-query-summary",
            ),
            (
                args.absolute_candidate_scores,
                "--absolute-candidate-scores",
            ),
        ]:
            require(
                value is not None,
                f"legacy_strict requires {name}.",
            )

        run(
            [
                sys.executable,

                repo_root
                / "scripts/villoc/blind_demo/"
                  "stage10b4_blind_temporal_fusion.py",

                "--map-trajectory",
                args.map_trajectory.resolve(),

                "--bootstrap-report",
                args.bootstrap_report.resolve(),

                "--absolute-query-summary",
                args.absolute_query_summary.resolve(),

                "--absolute-candidate-scores",
                args.absolute_candidate_scores.resolve(),

                "--run-root",
                run_root,
            ]
        )

    expected = [
        (
            run_root
            / "trajectories/"
              "blind_temporal_fused_trajectory.csv"
        ),
        (
            run_root
            / "metadata/blind_temporal_fusion/"
              "blind_temporal_correction_manifest.csv"
        ),
        (
            run_root
            / "reports/blind_temporal_fusion/"
              "blind_temporal_fusion_report.json"
        ),
    ]

    for path in expected:
        require(
            path.exists(),
            f"Stage-09 output missing: {path}",
        )

    print()
    print(
        "STATUS: PASS_STAGE09_TEMPORAL_ROUTER"
    )


if __name__ == "__main__":
    main()
