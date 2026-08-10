#!/usr/bin/env python3
"""
R5.4J — pluggable blind bootstrap backend.

The orchestrator calls ONE Stage-07 interface:

    stage10b2_bootstrap_backend.py

The wrapper selects the implementation declared in config.

Supported backends
------------------
minimum_confident_v2
    frozen R3-v2 implementation
    +
    canonical map-state adapter
    +
    canonical causal timeline adapter

legacy_strict
    historical stage10b2 implementation
    +
    canonical adapters

The orchestrator does not know algorithm internals.

No GT / GPS / SRT / reference may enter this stage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import yaml


def require(
    condition,
    message,
):

    if not condition:
        raise RuntimeError(message)


def sha256(
    path: Path,
) -> str:

    h = hashlib.sha256()

    with path.open("rb") as f:

        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def run(
    command,
):

    print()
    print(
        "$",
        " ".join(
            str(x)
            for x in command
        ),
    )

    subprocess.run(
        [
            str(x)
            for x in command
        ],
        check=True,
    )


def resolve_repo_path(
    repo_root: Path,
    value,
) -> Path:

    path = Path(
        str(value)
    )

    if not path.is_absolute():

        path = (
            repo_root
            / path
        )

    return path.resolve()


def load_config(
    path: Path,
):

    data = yaml.safe_load(
        path.read_text()
    )

    require(
        isinstance(
            data,
            dict,
        ),
        "Demo config must be a mapping.",
    )

    return data


def minimum_confident_v2(
    *,
    repo_root: Path,
    source_run_root: Path,
    run_root: Path,
    config,
):

    bootstrap_cfg = config.get(
        "bootstrap",
        {},
    )

    implementation = resolve_repo_path(
        repo_root,
        bootstrap_cfg[
            "implementation"
        ],
    )

    architecture = resolve_repo_path(
        repo_root,
        bootstrap_cfg[
            "architecture_contract"
        ],
    )

    final_policy = resolve_repo_path(
        repo_root,
        bootstrap_cfg[
            "final_policy_contract"
        ],
    )

    architecture_sha_expected = str(
        bootstrap_cfg[
            "architecture_contract_sha256"
        ]
    )

    final_policy_sha_expected = str(
        bootstrap_cfg[
            "final_policy_contract_sha256"
        ]
    )


    for path in [
        implementation,
        architecture,
        final_policy,
    ]:

        require(
            path.exists(),
            f"Missing promoted bootstrap asset: {path}",
        )


    require(
        sha256(
            architecture
        )
        == architecture_sha_expected,
        (
            "Promoted architecture-contract "
            "SHA mismatch."
        ),
    )


    require(
        sha256(
            final_policy
        )
        == final_policy_sha_expected,
        (
            "Promoted final-policy-contract "
            "SHA mismatch."
        ),
    )


    manifest = (
        source_run_root
        / "metadata/"
          "blind_query_manifest.csv"
    )


    require(
        manifest.exists(),
        f"Missing blind query manifest: {manifest}",
    )


    report_root = (
        run_root
        / "reports/"
          "blind_map_bootstrap"
    )

    backend_root = (
        report_root
        / "minimum_confident_v2"
    )


    backend_root.mkdir(
        parents=True,
        exist_ok=True,
    )


    # --------------------------------------------------------
    # Frozen v2 localization backend.
    # --------------------------------------------------------

    run(
        [
            sys.executable,

            implementation,

            "--repo-root",
            repo_root,

            "--run-root",
            source_run_root,

            "--architecture-contract",
            architecture,

            "--expected-contract-sha256",
            architecture_sha_expected,

            "--out-root",
            backend_root,
        ]
    )


    policy_results = (
        backend_root
        / "r5_1_blind_policy_results.json"
    )

    policy_timeline = (
        backend_root
        / "r5_1_blind_policy_timeline.csv"
    )

    hypotheses = (
        backend_root
        / "r5_1_blind_subtile_hypotheses.csv"
    )

    implementation_freeze = (
        backend_root
        / "r5_1_blind_implementation_freeze_manifest.json"
    )


    for path in [
        policy_results,
        policy_timeline,
        hypotheses,
        implementation_freeze,
    ]:

        require(
            path.exists(),
            f"Missing v2 backend output: {path}",
        )


    map_state_path = (
        report_root
        / "canonical_map_state.json"
    )

    timeline_path = (
        report_root
        / "canonical_map_state_timeline.json"
    )


    # --------------------------------------------------------
    # Canonical final-state snapshot.
    # --------------------------------------------------------

    run(
        [
            sys.executable,

            repo_root
            / "scripts/villoc/blind_demo/bootstrap/"
              "map_state_adapter.py",

            "--backend",
            "minimum_confident_v2",

            "--source-report",
            policy_results,

            "--blind-freeze-manifest",
            implementation_freeze,

            "--final-method-contract",
            final_policy,

            "--expected-final-contract-sha256",
            final_policy_sha_expected,

            "--manifest",
            manifest,

            "--output",
            map_state_path,
        ]
    )


    # --------------------------------------------------------
    # Canonical causal state history.
    # --------------------------------------------------------

    run(
        [
            sys.executable,

            repo_root
            / "scripts/villoc/blind_demo/bootstrap/"
              "map_state_timeline_adapter.py",

            "--backend",
            "minimum_confident_v2",

            "--policy-results",
            policy_results,

            "--policy-timeline",
            policy_timeline,

            "--hypotheses",
            hypotheses,

            "--blind-freeze-manifest",
            implementation_freeze,

            "--final-method-contract",
            final_policy,

            "--expected-final-contract-sha256",
            final_policy_sha_expected,

            "--manifest",
            manifest,

            "--output",
            timeline_path,
        ]
    )


    map_state = json.loads(
        map_state_path.read_text()
    )

    timeline = json.loads(
        timeline_path.read_text()
    )


    require(
        map_state[
            "source_backend"
        ]
        == "minimum_confident_v2",
        "Canonical state backend mismatch.",
    )


    require(
        timeline[
            "source_backend"
        ]
        == "minimum_confident_v2",
        "Canonical timeline backend mismatch.",
    )


    return {
        "source_backend":
            "minimum_confident_v2",

        "localization_state":
            map_state[
                "localization_state"
            ],

        "map_state_available":
            bool(
                map_state[
                    "map_state_available"
                ]
            ),

        "map_state_trust":
            map_state[
                "map_state_trust"
            ],

        "canonical_map_state":
            map_state_path,

        "canonical_map_state_timeline":
            timeline_path,

        "causal_event_count":
            int(
                timeline[
                    "event_count"
                ]
            ),

        "implementation_output_root":
            backend_root,
    }


def legacy_strict(
    *,
    repo_root: Path,
    source_run_root: Path,
    run_root: Path,
):

    # Legacy operation currently assumes source and destination
    # are the same orchestrated run.
    require(
        source_run_root.resolve()
        == run_root.resolve(),
        (
            "legacy_strict backend currently requires "
            "source_run_root == run_root."
        ),
    )


    legacy_script = (
        repo_root
        / "scripts/villoc/blind_demo/"
          "stage10b2_blind_map_bootstrap_audit.py"
    )


    run(
        [
            sys.executable,

            legacy_script,

            "--root",
            run_root,

            "--run-root",
            run_root,
        ]
    )


    report_root = (
        run_root
        / "reports/"
          "blind_map_bootstrap"
    )

    legacy_report = (
        report_root
        / "blind_map_bootstrap_report.json"
    )


    require(
        legacy_report.exists(),
        (
            "Legacy bootstrap did not produce "
            "blind_map_bootstrap_report.json."
        ),
    )


    map_state_path = (
        report_root
        / "canonical_map_state.json"
    )

    timeline_path = (
        report_root
        / "canonical_map_state_timeline.json"
    )


    run(
        [
            sys.executable,

            repo_root
            / "scripts/villoc/blind_demo/bootstrap/"
              "map_state_adapter.py",

            "--backend",
            "legacy_strict",

            "--source-report",
            legacy_report,

            "--output",
            map_state_path,
        ]
    )


    run(
        [
            sys.executable,

            repo_root
            / "scripts/villoc/blind_demo/bootstrap/"
              "map_state_timeline_adapter.py",

            "--backend",
            "legacy_strict",

            "--canonical-snapshot",
            map_state_path,

            "--output",
            timeline_path,
        ]
    )


    map_state = json.loads(
        map_state_path.read_text()
    )

    timeline = json.loads(
        timeline_path.read_text()
    )


    return {
        "source_backend":
            "legacy_strict",

        "localization_state":
            map_state[
                "localization_state"
            ],

        "map_state_available":
            bool(
                map_state[
                    "map_state_available"
                ]
            ),

        "map_state_trust":
            map_state[
                "map_state_trust"
            ],

        "canonical_map_state":
            map_state_path,

        "canonical_map_state_timeline":
            timeline_path,

        "causal_event_count":
            int(
                timeline[
                    "event_count"
                ]
            ),

        "implementation_output_root":
            report_root,
    }


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
        "--run-root",
        type=Path,
        required=True,
    )

    # Testing / migration aid.
    #
    # Production orchestration omits this argument, making
    # source_run_root == run_root.
    parser.add_argument(
        "--source-run-root",
        type=Path,
    )

    args = parser.parse_args()

    t0 = time.perf_counter()

    repo_root = (
        args.repo_root.resolve()
    )

    run_root = (
        args.run_root.resolve()
    )

    source_run_root = (
        args.source_run_root.resolve()
        if args.source_run_root
        else run_root
    )

    config_path = (
        args.config.resolve()
    )


    config = load_config(
        config_path
    )


    bootstrap_cfg = config.get(
        "bootstrap",
        {},
    )


    backend = str(
        bootstrap_cfg.get(
            "backend",
            "legacy_strict",
        )
    )


    allowed = {
        "minimum_confident_v2",
        "legacy_strict",
    }


    require(
        backend in allowed,
        (
            "Unsupported bootstrap backend: "
            f"{backend}"
        ),
    )


    run_root.mkdir(
        parents=True,
        exist_ok=True,
    )


    print()
    print("=" * 108)
    print(
        "STAGE 07 — PLUGGABLE BLIND MAP BOOTSTRAP"
    )
    print("=" * 108)

    print(
        "backend:",
        backend,
    )

    print(
        "source run root:",
        source_run_root,
    )

    print(
        "output run root:",
        run_root,
    )


    if backend == "minimum_confident_v2":

        result = minimum_confident_v2(
            repo_root=repo_root,
            source_run_root=source_run_root,
            run_root=run_root,
            config=config,
        )

    else:

        result = legacy_strict(
            repo_root=repo_root,
            source_run_root=source_run_root,
            run_root=run_root,
        )


    report_root = (
        run_root
        / "reports/"
          "blind_map_bootstrap"
    )

    report_root.mkdir(
        parents=True,
        exist_ok=True,
    )


    # For legacy_strict the historical report already owns
    # blind_map_bootstrap_report.json. Do not overwrite it.
    #
    # For v2 create the standard Stage-07 summary surface.
    if backend == "minimum_confident_v2":

        report_path = (
            report_root
            / "blind_map_bootstrap_report.json"
        )

        report = {
            "stage":
                "STAGE_07_PLUGGABLE_BLIND_MAP_BOOTSTRAP",

            "status":
                "PASS_BLIND_MAP_BOOTSTRAP_BACKEND",

            "source_backend":
                result[
                    "source_backend"
                ],

            "localization_state":
                result[
                    "localization_state"
                ],

            "map_state_available":
                result[
                    "map_state_available"
                ],

            "map_state_trust":
                result[
                    "map_state_trust"
                ],

            "causal_map_state_events":
                result[
                    "causal_event_count"
                ],

            "blind_contract": {
                "gps_used":
                    False,

                "srt_used":
                    False,

                "reference_used":
                    False,

                "oracle_used":
                    False,

                "evaluation_error_used":
                    False,
            },

            "canonical_outputs": {
                "map_state":
                    str(
                        result[
                            "canonical_map_state"
                        ]
                    ),

                "map_state_sha256":
                    sha256(
                        result[
                            "canonical_map_state"
                        ]
                    ),

                "map_state_timeline":
                    str(
                        result[
                            "canonical_map_state_timeline"
                        ]
                    ),

                "map_state_timeline_sha256":
                    sha256(
                        result[
                            "canonical_map_state_timeline"
                        ]
                    ),
            },

            "implementation_output_root":
                str(
                    result[
                        "implementation_output_root"
                    ]
                ),

            "runtime": {
                "total_stage_wall_s":
                    float(
                        time.perf_counter()
                        - t0
                    ),
            },
        }


        report_path.write_text(
            json.dumps(
                report,
                indent=2,
            )
        )

    else:

        report_path = (
            report_root
            / "blind_map_bootstrap_report.json"
        )


    print()
    print(
        "localization state:",
        result[
            "localization_state"
        ],
    )

    print(
        "map state available:",
        result[
            "map_state_available"
        ],
    )

    print(
        "map state trust:",
        result[
            "map_state_trust"
        ],
    )

    print(
        "causal map-state events:",
        result[
            "causal_event_count"
        ],
    )

    print(
        "canonical map state:",
        result[
            "canonical_map_state"
        ],
    )

    print(
        "canonical timeline:",
        result[
            "canonical_map_state_timeline"
        ],
    )

    print(
        "stage report:",
        report_path,
    )

    print()

    print(
        "STATUS: PASS_STAGE07_PLUGGABLE_BOOTSTRAP_BACKEND"
    )


if __name__ == "__main__":
    main()
