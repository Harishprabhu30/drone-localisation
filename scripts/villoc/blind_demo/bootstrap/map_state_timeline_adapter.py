#!/usr/bin/env python3
"""
R5.4E — canonical causal blind map-state timeline adapter.

Why this exists
---------------
A final map-state snapshot is insufficient for a stateful localization
backend.

Example:

    q30 -> transform A becomes mature
    q50 -> transform B accepted
    q80 -> transform C accepted

A consumer must use:

    q30..49 -> A
    q50..79 -> B
    q80..   -> C

not merely the final transform C.

This adapter therefore provides:

    villoc.blind_map_state_timeline.v1

Supported sources:
    minimum_confident_v2
    legacy_strict

No GT/reference/GPS/SRT/evaluation information is consumed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import pandas as pd


SCHEMA = (
    "villoc.blind_map_state_timeline.v1"
)

TRANSFORM_FIELDS = (
    "a_real",
    "a_imag",
    "b_real",
    "b_imag",
    "scale_m_per_visual_px",
    "rotation_deg",
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


def normalize_transform(
    row,
    label,
):

    result = {}

    for key in TRANSFORM_FIELDS:

        if key not in row:

            raise RuntimeError(
                f"{label}: missing transform field {key}"
            )

        value = float(
            row[key]
        )

        if not math.isfinite(value):

            raise RuntimeError(
                f"{label}: non-finite {key}={value}"
            )

        result[key] = value


    if (
        result[
            "scale_m_per_visual_px"
        ]
        <= 0.0
    ):

        raise RuntimeError(
            f"{label}: non-positive scale."
        )


    return result


def timestamp_lookup(
    manifest_path: Path | None,
):

    if manifest_path is None:

        return {}


    manifest = pd.read_csv(
        manifest_path
    )


    if (
        "query_id" not in manifest.columns
        or "timestamp_s" not in manifest.columns
    ):

        return {}


    result = {}


    for _, row in manifest.iterrows():

        q = int(
            row[
                "query_id"
            ]
        )

        t = float(
            row[
                "timestamp_s"
            ]
        )

        if math.isfinite(t):

            result[q] = t


    return result


def verify_frozen_output(
    freeze,
    key,
    path: Path,
):

    expected = (
        freeze
        .get(
            "outputs",
            {}
        )
        .get(
            key,
            {}
        )
        .get(
            "sha256"
        )
    )


    if expected is None:

        raise RuntimeError(
            "Blind freeze does not contain "
            f"hash for output key {key!r}"
        )


    actual = sha256(
        path
    )


    if actual != expected:

        raise RuntimeError(
            f"Frozen output hash mismatch: {key}\n"
            f"expected: {expected}\n"
            f"actual:   {actual}"
        )


def minimum_confident_v2(
    policy_results_path: Path,
    policy_timeline_path: Path,
    hypotheses_path: Path,
    blind_freeze_path: Path,
    final_contract_path: Path,
    manifest_path: Path | None,
    expected_final_contract_sha256: str | None,
):

    results = load_json(
        policy_results_path
    )

    freeze = load_json(
        blind_freeze_path
    )

    final_contract = load_json(
        final_contract_path
    )


    # --------------------------------------------------------
    # Integrity
    # --------------------------------------------------------

    final_contract_sha = sha256(
        final_contract_path
    )


    if (
        expected_final_contract_sha256
        and final_contract_sha
        != expected_final_contract_sha256
    ):

        raise RuntimeError(
            "Final method contract SHA mismatch.\n"
            f"expected: {expected_final_contract_sha256}\n"
            f"actual:   {final_contract_sha}"
        )


    blind_contract = freeze.get(
        "blind_contract",
        {}
    )


    for key in [
        "gps_used",
        "srt_used",
        "reference_used",
        "oracle_used",
        "evaluation_error_used",
    ]:

        if bool(
            blind_contract.get(
                key,
                False,
            )
        ):

            raise RuntimeError(
                f"Blind freeze violation: {key}=True"
            )


    verify_frozen_output(
        freeze,
        "policy_results",
        policy_results_path,
    )

    verify_frozen_output(
        freeze,
        "policy_timeline",
        policy_timeline_path,
    )

    verify_frozen_output(
        freeze,
        "hypotheses",
        hypotheses_path,
    )


    # --------------------------------------------------------
    # Frozen policy
    # --------------------------------------------------------

    policy_name = (
        final_contract[
            "final_policy"
        ][
            "name"
        ]
    )


    policy_results = results.get(
        "policy_results",
        {}
    )


    if policy_name not in policy_results:

        raise RuntimeError(
            "Frozen final policy is absent from "
            "blind policy results."
        )


    policy_result = (
        policy_results[
            policy_name
        ]
    )


    localization_state = str(
        policy_result[
            "localization_state"
        ]
    )


    if localization_state not in {
        "PROVISIONAL_ABSOLUTE_LOCK",
        "NO_PROVISIONAL_LOCK",
    }:

        raise RuntimeError(
            "Unexpected v2 localization state: "
            f"{localization_state}"
        )


    timeline = pd.read_csv(
        policy_timeline_path
    )


    hypotheses = pd.read_csv(
        hypotheses_path
    )


    timeline = timeline[
        timeline[
            "policy"
        ]
        == policy_name
    ].copy()


    if len(timeline) == 0:

        raise RuntimeError(
            "Frozen final policy has no timeline rows."
        )


    timeline[
        "update_query_id"
    ] = pd.to_numeric(
        timeline[
            "update_query_id"
        ],
        errors="raise",
    ).astype(int)


    hypotheses[
        "hypothesis_id"
    ] = pd.to_numeric(
        hypotheses[
            "hypothesis_id"
        ],
        errors="raise",
    ).astype(int)


    hyp_by_id = (
        hypotheses
        .set_index(
            "hypothesis_id"
        )
    )


    time_by_query = timestamp_lookup(
        manifest_path
    )


    events = []


    # --------------------------------------------------------
    # No-map-state case
    # --------------------------------------------------------

    if (
        localization_state
        == "NO_PROVISIONAL_LOCK"
    ):

        return {
            "schema":
                SCHEMA,

            "status":
                "PASS_CANONICAL_MAP_STATE_TIMELINE_NO_MAP_STATE",

            "source_backend":
                "minimum_confident_v2",

            "localization_state":
                localization_state,

            "map_state_available":
                False,

            "map_state_trust":
                "NONE",

            "map_crs":
                "EPSG:3346",

            "policy":
                policy_name,

            "events":
                [],

            "event_count":
                0,

            "first_effective_query_id":
                None,

            "final_source_query_id":
                None,

            "blind_contract": {
                "gps_used": False,
                "srt_used": False,
                "reference_used": False,
                "oracle_used": False,
                "evaluation_error_used": False,
            },

            "provenance": {
                "policy_results":
                    str(
                        policy_results_path.resolve()
                    ),

                "policy_timeline":
                    str(
                        policy_timeline_path.resolve()
                    ),

                "hypotheses":
                    str(
                        hypotheses_path.resolve()
                    ),

                "blind_freeze":
                    str(
                        blind_freeze_path.resolve()
                    ),

                "final_method_contract":
                    str(
                        final_contract_path.resolve()
                    ),

                "final_method_contract_sha256":
                    final_contract_sha,
            },
        }


    # --------------------------------------------------------
    # Positive map-state case.
    #
    # We ignore ACQUISITION states before maturity.
    #
    # The first event is the transform that becomes active
    # when mode_after first reaches TRACKING.
    #
    # Thereafter we record only genuinely changed active
    # hypothesis IDs.
    # --------------------------------------------------------

    matured_at = int(
        policy_result[
            "matured_at_query_id"
        ]
    )


    after_maturity = timeline[
        timeline[
            "update_query_id"
        ]
        >= matured_at
    ].sort_values(
        "update_query_id"
    )


    previous_hypothesis_id = None


    for _, row in after_maturity.iterrows():

        mode_after = str(
            row[
                "mode_after"
            ]
        )


        if mode_after != "TRACKING":

            continue


        value = row[
            "active_hypothesis_id_after"
        ]


        if pd.isna(value):

            raise RuntimeError(
                "Tracking row has no active hypothesis."
            )


        hypothesis_id = int(
            float(value)
        )


        # A HOLD preserves the same active hypothesis.
        # Do not emit duplicate events.
        if (
            previous_hypothesis_id
            == hypothesis_id
        ):

            continue


        if hypothesis_id not in hyp_by_id.index:

            raise RuntimeError(
                "Timeline references unknown hypothesis: "
                f"{hypothesis_id}"
            )


        hyp = hyp_by_id.loc[
            hypothesis_id
        ]


        q = int(
            row[
                "update_query_id"
            ]
        )


        source_value = row[
            "active_source_update_q_after"
        ]


        source_q = (
            int(
                float(
                    source_value
                )
            )
            if not pd.isna(
                source_value
            )
            else q
        )


        events.append(
            {
                "event_index":
                    len(events),

                "effective_from_query_id":
                    q,

                "effective_from_timestamp_s":
                    time_by_query.get(
                        q
                    ),

                "source_update_query_id":
                    source_q,

                "source_hypothesis_id":
                    hypothesis_id,

                "action":
                    str(
                        row[
                            "action"
                        ]
                    ),

                "mode_after":
                    mode_after,

                "map_state_trust":
                    "PROVISIONAL",

                "transform":
                    normalize_transform(
                        hyp,
                        (
                            "minimum_confident_v2 "
                            f"hypothesis {hypothesis_id}"
                        ),
                    ),
            }
        )


        previous_hypothesis_id = (
            hypothesis_id
        )


    if not events:

        raise RuntimeError(
            "PROVISIONAL_ABSOLUTE_LOCK produced "
            "no canonical map-state events."
        )


    # --------------------------------------------------------
    # Cross-check final active state.
    # --------------------------------------------------------

    final_state = (
        policy_result[
            "final_active_map_state"
        ]
    )


    final_event_transform = (
        events[-1][
            "transform"
        ]
    )


    for key in TRANSFORM_FIELDS:

        if abs(
            float(
                final_state[
                    key
                ]
            )
            -
            float(
                final_event_transform[
                    key
                ]
            )
        ) > 1e-9:

            raise RuntimeError(
                "Final canonical timeline transform "
                "does not equal frozen final active state "
                f"for field {key}."
            )


    final_source_q = int(
        policy_result[
            "final_active_source_update_q"
        ]
    )


    if (
        int(
            events[-1][
                "source_update_query_id"
            ]
        )
        != final_source_q
    ):

        raise RuntimeError(
            "Final timeline source query does not "
            "match frozen policy result."
        )


    return {
        "schema":
            SCHEMA,

        "status":
            "PASS_CANONICAL_MAP_STATE_TIMELINE",

        "source_backend":
            "minimum_confident_v2",

        "localization_state":
            localization_state,

        "map_state_available":
            True,

        "map_state_trust":
            "PROVISIONAL",

        "map_crs":
            "EPSG:3346",

        "policy":
            policy_name,

        "matured_at_query_id":
            matured_at,

        "events":
            events,

        "event_count":
            len(events),

        "first_effective_query_id":
            int(
                events[0][
                    "effective_from_query_id"
                ]
            ),

        "final_source_query_id":
            final_source_q,

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

        "semantic_guarantees": {
            "pre_maturity_map_state_emitted":
                False,

            "duplicate_hold_events_emitted":
                False,

            "final_transform_verified":
                True,

            "trust_upgraded":
                False,
        },

        "provenance": {
            "policy_results":
                str(
                    policy_results_path.resolve()
                ),

            "policy_results_sha256":
                sha256(
                    policy_results_path
                ),

            "policy_timeline":
                str(
                    policy_timeline_path.resolve()
                ),

            "policy_timeline_sha256":
                sha256(
                    policy_timeline_path
                ),

            "hypotheses":
                str(
                    hypotheses_path.resolve()
                ),

            "hypotheses_sha256":
                sha256(
                    hypotheses_path
                ),

            "blind_freeze":
                str(
                    blind_freeze_path.resolve()
                ),

            "blind_freeze_sha256":
                sha256(
                    blind_freeze_path
                ),

            "final_method_contract":
                str(
                    final_contract_path.resolve()
                ),

            "final_method_contract_sha256":
                final_contract_sha,
        },
    }


def legacy_strict(
    canonical_snapshot_path: Path,
):

    snapshot = load_json(
        canonical_snapshot_path
    )


    if (
        snapshot.get(
            "schema"
        )
        != "villoc.blind_map_state.v1"
    ):

        raise RuntimeError(
            "Legacy source is not canonical "
            "map-state v1."
        )


    if (
        snapshot[
            "source_backend"
        ]
        != "legacy_strict"
    ):

        raise RuntimeError(
            "Expected legacy_strict canonical snapshot."
        )


    if not bool(
        snapshot[
            "map_state_available"
        ]
    ):

        return {
            "schema":
                SCHEMA,

            "status":
                "PASS_CANONICAL_MAP_STATE_TIMELINE_NO_MAP_STATE",

            "source_backend":
                "legacy_strict",

            "localization_state":
                snapshot[
                    "localization_state"
                ],

            "map_state_available":
                False,

            "map_state_trust":
                "NONE",

            "map_crs":
                snapshot[
                    "map_crs"
                ],

            "events":
                [],

            "event_count":
                0,

            "first_effective_query_id":
                None,

            "final_source_query_id":
                None,

            "blind_contract":
                snapshot[
                    "blind_contract"
                ],
        }


    q = int(
        snapshot[
            "source_query_id"
        ]
    )


    event = {
        "event_index":
            0,

        "effective_from_query_id":
            q,

        "effective_from_timestamp_s":
            snapshot.get(
                "source_timestamp_s"
            ),

        "source_update_query_id":
            q,

        "source_hypothesis_id":
            None,

        "action":
            "LEGACY_SINGLE_LOCK",

        "mode_after":
            "TRACKING",

        "map_state_trust":
            snapshot[
                "map_state_trust"
            ],

        "transform":
            normalize_transform(
                snapshot[
                    "transform"
                ],
                "legacy strict snapshot",
            ),
    }


    return {
        "schema":
            SCHEMA,

        "status":
            "PASS_CANONICAL_MAP_STATE_TIMELINE",

        "source_backend":
            "legacy_strict",

        "localization_state":
            snapshot[
                "localization_state"
            ],

        "map_state_available":
            True,

        "map_state_trust":
            snapshot[
                "map_state_trust"
            ],

        "map_crs":
            snapshot[
                "map_crs"
            ],

        "events":
            [
                event
            ],

        "event_count":
            1,

        "first_effective_query_id":
            q,

        "final_source_query_id":
            q,

        "blind_contract":
            snapshot[
                "blind_contract"
            ],

        "semantic_guarantees": {
            "trust_upgraded":
                False,

            "legacy_single_lock_preserved":
                True,
        },

        "provenance": {
            "canonical_snapshot":
                str(
                    canonical_snapshot_path.resolve()
                ),

            "canonical_snapshot_sha256":
                sha256(
                    canonical_snapshot_path
                ),
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
        "--policy-results",
        type=Path,
    )

    parser.add_argument(
        "--policy-timeline",
        type=Path,
    )

    parser.add_argument(
        "--hypotheses",
        type=Path,
    )

    parser.add_argument(
        "--blind-freeze-manifest",
        type=Path,
    )

    parser.add_argument(
        "--final-method-contract",
        type=Path,
    )

    parser.add_argument(
        "--expected-final-contract-sha256",
    )

    parser.add_argument(
        "--manifest",
        type=Path,
    )

    parser.add_argument(
        "--canonical-snapshot",
        type=Path,
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

        required = {
            "--policy-results":
                args.policy_results,

            "--policy-timeline":
                args.policy_timeline,

            "--hypotheses":
                args.hypotheses,

            "--blind-freeze-manifest":
                args.blind_freeze_manifest,

            "--final-method-contract":
                args.final_method_contract,
        }


        missing = [
            name
            for name, value
            in required.items()
            if value is None
        ]


        if missing:

            raise RuntimeError(
                "minimum_confident_v2 missing arguments: "
                + ", ".join(
                    missing
                )
            )


        output = minimum_confident_v2(
            policy_results_path=(
                args.policy_results.resolve()
            ),

            policy_timeline_path=(
                args.policy_timeline.resolve()
            ),

            hypotheses_path=(
                args.hypotheses.resolve()
            ),

            blind_freeze_path=(
                args.blind_freeze_manifest.resolve()
            ),

            final_contract_path=(
                args.final_method_contract.resolve()
            ),

            manifest_path=(
                args.manifest.resolve()
                if args.manifest
                else None
            ),

            expected_final_contract_sha256=(
                args.expected_final_contract_sha256
            ),
        )


    else:

        if args.canonical_snapshot is None:

            raise RuntimeError(
                "legacy_strict requires "
                "--canonical-snapshot."
            )


        output = legacy_strict(
            args.canonical_snapshot.resolve()
        )


    out_path = args.output.resolve()

    out_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    out_path.write_text(
        json.dumps(
            output,
            indent=2,
        )
    )


    print()
    print("=" * 108)
    print(
        "R5.4E — CANONICAL CAUSAL MAP-STATE TIMELINE"
    )
    print("=" * 108)

    print(
        "backend:",
        output[
            "source_backend"
        ],
    )

    print(
        "localization state:",
        output[
            "localization_state"
        ],
    )

    print(
        "map state available:",
        output[
            "map_state_available"
        ],
    )

    print(
        "event count:",
        output[
            "event_count"
        ],
    )

    print(
        "first effective query:",
        output[
            "first_effective_query_id"
        ],
    )

    print(
        "final source query:",
        output[
            "final_source_query_id"
        ],
    )


    if output[
        "events"
    ]:

        print()
        print(
            "Causal transform events"
        )
        print("-" * 108)

        for event in output[
            "events"
        ]:

            t = event[
                "transform"
            ]

            print(
                f"event={event['event_index']:02d} "
                f"effective_q={event['effective_from_query_id']} "
                f"source_q={event['source_update_query_id']} "
                f"action={event['action']} "
                f"scale={t['scale_m_per_visual_px']:.9f} "
                f"rotation={t['rotation_deg']:.6f}"
            )


    print()
    print(
        "output:",
        out_path,
    )

    print(
        "output SHA256:",
        sha256(
            out_path
        ),
    )

    print()

    print(
        "STATUS:",
        output[
            "status"
        ],
    )


if __name__ == "__main__":
    main()
