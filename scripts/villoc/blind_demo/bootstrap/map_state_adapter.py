#!/usr/bin/env python3
"""
R5.4C — canonical blind bootstrap map-state adapter.

Purpose
-------
Decouple:

    bootstrap algorithm
        from
    map-alignment / fusion / export
        from
    orchestration.

Supported backends
------------------
1. minimum_confident_v2
       PROVISIONAL_ABSOLUTE_LOCK
       NO_PROVISIONAL_LOCK

2. legacy_strict
       ABSOLUTE_LOCKED
       NO_TRUSTED_ABSOLUTE_LOCK

All backends are converted into the same canonical contract:

    villoc.blind_map_state.v1

Important semantic rule
-----------------------
The adapter NEVER upgrades trust.

In particular:

    PROVISIONAL_ABSOLUTE_LOCK
        MUST remain provisional.

It is forbidden to translate it into:

    ABSOLUTE_LOCKED

simply to satisfy historical downstream scripts.

The generic downstream consumer should primarily inspect:

    map_state_available

and then preserve:

    localization_state
    map_state_trust
    source_backend

as provenance/trust metadata.

No GT/reference/GPS/SRT is used.

Command:

1. Test backend 1 — frozen R3-v2 demo result

TESTROOT=outputs/research_runs/minimum_confident_bootstrap/r5_4c_adapter_tests

mkdir -p "$TESTROOT"

V2ROOT=outputs/research_runs/minimum_confident_bootstrap/demo_recorded_flight_blind_r5_4_001

FINAL=outputs/research_runs/minimum_confident_bootstrap/traj01_blind_r3_001/r5_3_final_minimum_confident_bootstrap_contract.json

DEMO=outputs/demo_runs/blind_recorded_flight_orchestrated_replay_002

python \
  scripts/villoc/blind_demo/bootstrap/map_state_adapter.py \
  --backend minimum_confident_v2 \
  --source-report \
    "$V2ROOT/r5_1_blind_policy_results.json" \
  --blind-freeze-manifest \
    "$V2ROOT/r5_1_blind_implementation_freeze_manifest.json" \
  --final-method-contract \
    "$FINAL" \
  --expected-final-contract-sha256 \
    ab074e4f63e126eefe34639beec4703edffb9fbef59f726f48d8c7a5759a4ab6 \
  --manifest \
    "$DEMO/metadata/blind_query_manifest.csv" \
  --output \
    "$TESTROOT/minimum_confident_v2_demo.json"

2. Test backend 2 — historical strict demo no-lock
python \
  scripts/villoc/blind_demo/bootstrap/map_state_adapter.py \
  --backend legacy_strict \
  --source-report \
    outputs/demo_runs/blind_recorded_flight_orchestrated_replay_002/reports/blind_map_bootstrap/blind_map_bootstrap_report.json \
  --output \
    "$TESTROOT/legacy_demo_no_lock.json"

3. Test backend 3 — historical strict traj01 lock
python \
  scripts/villoc/blind_demo/bootstrap/map_state_adapter.py \
  --backend legacy_strict \
  --source-report \
    outputs/demo_runs/traj01_blind_regression_001/reports/blind_map_bootstrap/blind_map_bootstrap_report.json \
  --output \
    "$TESTROOT/legacy_traj01_lock.json"
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


SCHEMA = "villoc.blind_map_state.v1"

TRANSFORM_FIELDS = (
    "a_real",
    "a_imag",
    "b_real",
    "b_imag",
    "scale_m_per_visual_px",
    "rotation_deg",
)

FORBIDDEN_TRUE_KEYS = (
    "gt_used",
    "gps_used",
    "srt_used",
    "reference_used",
    "oracle_used",
    "evaluation_error_used",
)


def sha256(path: Path) -> str:

    h = hashlib.sha256()

    with path.open("rb") as f:

        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def load_json(path: Path):

    if not path.exists():

        raise RuntimeError(
            f"Missing JSON input: {path}"
        )

    return json.loads(
        path.read_text()
    )


def validate_blind_contract(
    contract,
    label,
):

    contract = (
        contract
        if isinstance(
            contract,
            dict,
        )
        else {}
    )

    violations = []

    for key in FORBIDDEN_TRUE_KEYS:

        if bool(
            contract.get(
                key,
                False,
            )
        ):

            violations.append(
                f"{key}=True"
            )

    if violations:

        raise RuntimeError(
            f"{label}: blind-contract violation: "
            + ", ".join(
                violations
            )
        )


def normalize_transform(
    transform,
    label,
):

    if not isinstance(
        transform,
        dict,
    ):

        raise RuntimeError(
            f"{label}: transform is not a dictionary."
        )

    missing = [
        key
        for key in TRANSFORM_FIELDS
        if key not in transform
    ]

    if missing:

        raise RuntimeError(
            f"{label}: transform missing fields: "
            f"{missing}"
        )

    normalized = {}

    for key in TRANSFORM_FIELDS:

        value = float(
            transform[
                key
            ]
        )

        if not math.isfinite(
            value
        ):

            raise RuntimeError(
                f"{label}: non-finite transform field "
                f"{key}={value}"
            )

        normalized[
            key
        ] = value

    if (
        normalized[
            "scale_m_per_visual_px"
        ]
        <= 0.0
    ):

        raise RuntimeError(
            f"{label}: scale must be positive."
        )

    return normalized


def manifest_timestamp_for_query(
    manifest_path: Path | None,
    query_id,
):

    if (
        manifest_path is None
        or query_id is None
    ):

        return None

    import pandas as pd

    df = pd.read_csv(
        manifest_path
    )

    if "query_id" not in df.columns:

        return None

    rows = df[
        pd.to_numeric(
            df[
                "query_id"
            ],
            errors="coerce",
        )
        == int(
            query_id
        )
    ]

    if len(
        rows
    ) != 1:

        return None

    if "timestamp_s" not in rows.columns:

        return None

    value = float(
        rows.iloc[
            0
        ][
            "timestamp_s"
        ]
    )

    return (
        value
        if math.isfinite(
            value
        )
        else None
    )


def adapt_minimum_confident_v2(
    source_report: Path,
    final_contract_path: Path,
    blind_freeze_path: Path,
    manifest_path: Path | None,
    expected_contract_sha256: str | None,
):

    source = load_json(
        source_report
    )

    final_contract = load_json(
        final_contract_path
    )

    freeze = load_json(
        blind_freeze_path
    )


    # --------------------------------------------------------
    # Integrity
    # --------------------------------------------------------

    contract_sha = sha256(
        final_contract_path
    )

    if (
        expected_contract_sha256
        and contract_sha
        != expected_contract_sha256
    ):

        raise RuntimeError(
            "Final method contract SHA mismatch.\n"
            f"expected: {expected_contract_sha256}\n"
            f"actual:   {contract_sha}"
        )


    validate_blind_contract(
        freeze.get(
            "blind_contract",
            {}
        ),
        "minimum_confident_v2 freeze",
    )


    # --------------------------------------------------------
    # Frozen final policy selection
    # --------------------------------------------------------

    final_policy = (
        final_contract[
            "final_policy"
        ][
            "name"
        ]
    )

    policy_results = source.get(
        "policy_results"
    )

    if not isinstance(
        policy_results,
        dict,
    ):

        raise RuntimeError(
            "minimum_confident_v2 source report has no "
            "policy_results dictionary."
        )

    if final_policy not in policy_results:

        raise RuntimeError(
            "Frozen final policy missing from source results: "
            f"{final_policy}"
        )


    policy = policy_results[
        final_policy
    ]

    state = str(
        policy[
            "localization_state"
        ]
    )


    if bool(
        policy.get(
            "forbidden_state_emitted",
            False,
        )
    ):

        raise RuntimeError(
            "minimum_confident_v2 reported "
            "forbidden_state_emitted=True."
        )


    if state == "PROVISIONAL_ABSOLUTE_LOCK":

        transform = normalize_transform(
            policy[
                "final_active_map_state"
            ],
            "minimum_confident_v2",
        )

        source_query_id = int(
            policy[
                "final_active_source_update_q"
            ]
        )

        maturity_query_id = int(
            policy[
                "matured_at_query_id"
            ]
        )

        timestamp_s = manifest_timestamp_for_query(
            manifest_path,
            source_query_id,
        )

        canonical_state = (
            "MAP_STATE_AVAILABLE"
        )

        map_state_available = True

        trust = "PROVISIONAL"


    elif state == "NO_PROVISIONAL_LOCK":

        transform = None
        source_query_id = None
        maturity_query_id = None
        timestamp_s = None

        canonical_state = (
            "NO_MAP_STATE"
        )

        map_state_available = False

        trust = "NONE"


    else:

        raise RuntimeError(
            "Unexpected minimum_confident_v2 state: "
            f"{state}"
        )


    return {
        "schema":
            SCHEMA,

        "stage":
            "STAGE_10B2_BOOTSTRAP_BACKEND_ADAPTER",

        "status":
            "PASS_BOOTSTRAP_BACKEND_ADAPTER",

        "source_backend":
            "minimum_confident_v2",

        "canonical_state":
            canonical_state,

        "localization_state":
            state,

        "map_state_available":
            map_state_available,

        "map_state_trust":
            trust,

        "map_crs":
            "EPSG:3346",

        "source_query_id":
            source_query_id,

        "source_timestamp_s":
            timestamp_s,

        "maturity_query_id":
            maturity_query_id,

        "transform":
            transform,

        "policy": {
            "name":
                final_policy,

            "activation_threshold_m":
                float(
                    policy[
                        "activation_threshold_m"
                    ]
                ),

            "tracking_threshold_m":
                float(
                    policy[
                        "tracking_threshold_m"
                    ]
                ),

            "maturity_support_required":
                int(
                    policy[
                        "maturity_support_required"
                    ]
                ),

            "selection_basis":
                final_contract[
                    "final_policy"
                ][
                    "selection_basis"
                ],
        },

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

        "provenance": {
            "source_policy_report":
                str(
                    source_report.resolve()
                ),

            "source_policy_report_sha256":
                sha256(
                    source_report
                ),

            "blind_freeze_manifest":
                str(
                    blind_freeze_path.resolve()
                ),

            "blind_freeze_manifest_sha256":
                sha256(
                    blind_freeze_path
                ),

            "final_method_contract":
                str(
                    final_contract_path.resolve()
                ),

            "final_method_contract_sha256":
                contract_sha,

            "manifest":
                (
                    str(
                        manifest_path.resolve()
                    )
                    if manifest_path
                    else None
                ),
        },

        "semantic_guarantees": {
            "trust_upgraded":
                False,

            "provisional_state_preserved":
                True,

            "transform_modified":
                False,
        },
    }


def adapt_legacy_strict(
    source_report: Path,
):

    source = load_json(
        source_report
    )


    validate_blind_contract(
        source.get(
            "blind_contract",
            {}
        ),
        "legacy_strict",
    )


    state = str(
        source.get(
            "localization_state"
        )
    )


    if state == "ABSOLUTE_LOCKED":

        lock = source.get(
            "map_lock"
        )

        transform = normalize_transform(
            lock,
            "legacy_strict",
        )


        source_query_id = None

        for key in (
            "lock_query_id",
            "lock_sequence_frame_id",
            "seed_query_j",
        ):

            if (
                key in lock
                and lock[
                    key
                ]
                is not None
            ):

                source_query_id = int(
                    lock[
                        key
                    ]
                )

                break


        source_timestamp_s = (
            float(
                lock[
                    "lock_timestamp_s"
                ]
            )
            if lock.get(
                "lock_timestamp_s"
            )
            is not None
            else None
        )


        canonical_state = (
            "MAP_STATE_AVAILABLE"
        )

        available = True
        trust = "LEGACY_TRUSTED"


    elif state == "NO_TRUSTED_ABSOLUTE_LOCK":

        if source.get(
            "map_lock"
        ) is not None:

            raise RuntimeError(
                "Legacy no-lock report unexpectedly "
                "contains map_lock."
            )

        transform = None
        source_query_id = None
        source_timestamp_s = None

        canonical_state = (
            "NO_MAP_STATE"
        )

        available = False
        trust = "NONE"


    else:

        raise RuntimeError(
            "Unexpected legacy_strict state: "
            f"{state}"
        )


    return {
        "schema":
            SCHEMA,

        "stage":
            "STAGE_10B2_BOOTSTRAP_BACKEND_ADAPTER",

        "status":
            "PASS_BOOTSTRAP_BACKEND_ADAPTER",

        "source_backend":
            "legacy_strict",

        "canonical_state":
            canonical_state,

        "localization_state":
            state,

        "map_state_available":
            available,

        "map_state_trust":
            trust,

        "map_crs":
            "EPSG:3346",

        "source_query_id":
            source_query_id,

        "source_timestamp_s":
            source_timestamp_s,

        "maturity_query_id":
            source_query_id,

        "transform":
            transform,

        "policy": {
            "name":
                "legacy_strict_stage10b2",

            "selection_basis":
                "historical promoted strict-A/B bootstrap",
        },

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

        "provenance": {
            "source_bootstrap_report":
                str(
                    source_report.resolve()
                ),

            "source_bootstrap_report_sha256":
                sha256(
                    source_report
                ),
        },

        "semantic_guarantees": {
            "trust_upgraded":
                False,

            "transform_modified":
                False,
        },
    }


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--backend",
        required=True,
        choices=[
            "minimum_confident_v2",
            "legacy_strict",
        ],
    )

    parser.add_argument(
        "--source-report",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--final-method-contract",
        type=Path,
    )

    parser.add_argument(
        "--blind-freeze-manifest",
        type=Path,
    )

    parser.add_argument(
        "--manifest",
        type=Path,
    )

    parser.add_argument(
        "--expected-final-contract-sha256",
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    args = parser.parse_args()


    if (
        args.backend
        == "minimum_confident_v2"
    ):

        if (
            args.final_method_contract
            is None
            or args.blind_freeze_manifest
            is None
        ):

            raise RuntimeError(
                "minimum_confident_v2 requires "
                "--final-method-contract and "
                "--blind-freeze-manifest."
            )


        canonical = adapt_minimum_confident_v2(
            source_report=(
                args.source_report.resolve()
            ),

            final_contract_path=(
                args.final_method_contract.resolve()
            ),

            blind_freeze_path=(
                args.blind_freeze_manifest.resolve()
            ),

            manifest_path=(
                args.manifest.resolve()
                if args.manifest
                else None
            ),

            expected_contract_sha256=(
                args.expected_final_contract_sha256
            ),
        )


    else:

        canonical = adapt_legacy_strict(
            source_report=(
                args.source_report.resolve()
            )
        )


    output = args.output.resolve()

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.write_text(
        json.dumps(
            canonical,
            indent=2,
        )
    )


    print()
    print("=" * 96)
    print(
        "R5.4C — CANONICAL BOOTSTRAP MAP-STATE ADAPTER"
    )
    print("=" * 96)

    print(
        "backend:",
        canonical[
            "source_backend"
        ],
    )

    print(
        "canonical state:",
        canonical[
            "canonical_state"
        ],
    )

    print(
        "localization state:",
        canonical[
            "localization_state"
        ],
    )

    print(
        "map state available:",
        canonical[
            "map_state_available"
        ],
    )

    print(
        "map state trust:",
        canonical[
            "map_state_trust"
        ],
    )

    print(
        "source query:",
        canonical[
            "source_query_id"
        ],
    )

    print(
        "source timestamp:",
        canonical[
            "source_timestamp_s"
        ],
    )

    print(
        "transform:",
        canonical[
            "transform"
        ],
    )

    print(
        "output:",
        output,
    )

    print(
        "output SHA256:",
        sha256(
            output
        ),
    )

    print()

    print(
        "STATUS: "
        "PASS_R5_4C_CANONICAL_MAP_STATE_ADAPTER"
    )


if __name__ == "__main__":
    main()
