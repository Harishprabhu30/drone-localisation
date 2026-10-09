#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from uavloc.data.trajectory_adapter import (
    CANONICAL_BLIND_COLUMNS,
    FORBIDDEN_BLIND_REFERENCE_COLUMNS,
    TrajectorySpec,
    TrajectorySpecError,
    load_trajectory_spec,
)


ROOT = Path.cwd().resolve()
DEFAULT_TRAJECTORY = Path(
    "configs/trajectories/villoc_traj01_90deg_stable120m_v1.yaml"
)
DEFAULT_OUTPUT_ROOT = Path(
    "outputs/research_runs/trajectory_adapter_v1/ta2_traj01"
)


def resolve(path_like: str | Path) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else ROOT / path


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def false_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return ~series.fillna(False)

    normalized = (
        series.astype(str)
        .str.strip()
        .str.lower()
    )
    return normalized.isin(
        {"false", "0", "no", "n", "f"}
    )


def legacy_required_columns() -> set[str]:
    return {
        "query_id",
        "frame_index",
        "timestamp_s",
        "image_path",
        "image_width",
        "image_height",
        "reference_available",
        "assumed_rel_alt_m",
        "assumed_gimbal_pitch_deg",
        "view_assumption",
    }


def assert_blind_safe_columns(columns: list[str]) -> None:
    lower = {
        str(column).strip().lower()
        for column in columns
    }
    leaked = sorted(
        lower
        & {
            value.lower()
            for value in FORBIDDEN_BLIND_REFERENCE_COLUMNS
        }
    )
    if leaked:
        raise RuntimeError(
            "Reference/evaluation columns leaked into source manifest: "
            f"{leaked}"
        )


def assumed_signal_value(
    spec: TrajectorySpec,
    name: str,
) -> float:
    signal = spec.signal(name)
    if signal.source != "assumed":
        raise TrajectorySpecError(
            f"TA2 parity expects {name} to use source='assumed', "
            f"got {signal.source!r}."
        )
    return float(signal.value)


def canonicalize_legacy_manifest(
    *,
    legacy: pd.DataFrame,
    spec: TrajectorySpec,
) -> pd.DataFrame:
    missing = sorted(
        legacy_required_columns()
        - set(legacy.columns)
    )
    if missing:
        raise RuntimeError(
            f"Legacy blind manifest missing columns: {missing}"
        )

    assert_blind_safe_columns(
        list(legacy.columns)
    )

    if not false_series(
        legacy["reference_available"]
    ).all():
        raise RuntimeError(
            "Legacy blind manifest must have reference_available=false for every row."
        )

    rel_alt = assumed_signal_value(
        spec,
        "relative_altitude_m",
    )
    pitch = assumed_signal_value(
        spec,
        "gimbal_pitch_deg",
    )

    measured_alt = pd.to_numeric(
        legacy["assumed_rel_alt_m"],
        errors="raise",
    )
    measured_pitch = pd.to_numeric(
        legacy["assumed_gimbal_pitch_deg"],
        errors="raise",
    )

    if not (
        (measured_alt - rel_alt).abs()
        <= 1e-9
    ).all():
        raise RuntimeError(
            "Legacy assumed relative altitude does not match trajectory spec."
        )

    if not (
        (measured_pitch - pitch).abs()
        <= 1e-9
    ).all():
        raise RuntimeError(
            "Legacy assumed gimbal pitch does not match trajectory spec."
        )

    out = pd.DataFrame(
        {
            "trajectory_id": spec.trajectory_id,
            "query_id": pd.to_numeric(
                legacy["query_id"],
                errors="raise",
            ).astype(int),
            "frame_index": pd.to_numeric(
                legacy["frame_index"],
                errors="raise",
            ).astype(int),
            "timestamp_s": pd.to_numeric(
                legacy["timestamp_s"],
                errors="raise",
            ).astype(float),
            "image_path": legacy[
                "image_path"
            ].astype(str),
            "image_width": pd.to_numeric(
                legacy["image_width"],
                errors="raise",
            ).astype(int),
            "image_height": pd.to_numeric(
                legacy["image_height"],
                errors="raise",
            ).astype(int),
            "reference_available": False,
            "relative_altitude_m": float(
                rel_alt
            ),
            "relative_altitude_m_source": (
                spec.signal(
                    "relative_altitude_m"
                ).source
            ),
            "gimbal_pitch_deg": float(
                pitch
            ),
            "gimbal_pitch_deg_source": (
                spec.signal(
                    "gimbal_pitch_deg"
                ).source
            ),
            "view_assumption": legacy[
                "view_assumption"
            ].astype(str),
        }
    )

    optional_passthrough = (
        "token0_id",
        "sequence_frame_id",
        "source_video_frame_index",
        "source_video_frame_time_s",
        "sampling_alignment_error_ms",
        "source_video",
        "extraction_status",
    )

    for column in optional_passthrough:
        if column in legacy.columns:
            out[column] = legacy[column]

    canonical_missing = sorted(
        set(CANONICAL_BLIND_COLUMNS)
        - set(out.columns)
    )
    if canonical_missing:
        raise RuntimeError(
            f"Canonical output missing columns: {canonical_missing}"
        )

    assert_blind_safe_columns(
        list(out.columns)
    )

    return out


def parity_report(
    *,
    legacy: pd.DataFrame,
    canonical: pd.DataFrame,
    spec: TrajectorySpec,
) -> dict[str, Any]:
    compat = dict(
        spec.raw.get(
            "compatibility",
            {}
        )
    )

    query_ids = pd.to_numeric(
        canonical["query_id"],
        errors="raise",
    ).astype(int)
    frame_indices = pd.to_numeric(
        canonical["frame_index"],
        errors="raise",
    ).astype(int)
    timestamps = pd.to_numeric(
        canonical["timestamp_s"],
        errors="raise",
    ).astype(float)

    checks = {
        "query_count": int(
            len(canonical)
        ) == int(
            compat["expected_query_count"]
        ),
        "query_id_start": int(
            query_ids.iloc[0]
        ) == int(
            compat["expected_query_id_start"]
        ),
        "query_id_end": int(
            query_ids.iloc[-1]
        ) == int(
            compat["expected_query_id_end"]
        ),
        "query_id_unique": bool(
            query_ids.is_unique
        ),
        "frame_index_start": int(
            frame_indices.iloc[0]
        ) == int(
            compat["expected_frame_index_start"]
        ),
        "frame_index_end": int(
            frame_indices.iloc[-1]
        ) == int(
            compat["expected_frame_index_end"]
        ),
        "timestamp_start": abs(
            float(timestamps.iloc[0])
            - float(
                compat[
                    "expected_timestamp_start_s"
                ]
            )
        ) <= 1e-9,
        "timestamp_end": abs(
            float(timestamps.iloc[-1])
            - float(
                compat[
                    "expected_timestamp_end_s"
                ]
            )
        ) <= 1e-9,
        "width": bool(
            (
                canonical["image_width"]
                == int(
                    compat[
                        "expected_width_px"
                    ]
                )
            ).all()
        ),
        "height": bool(
            (
                canonical["image_height"]
                == int(
                    compat[
                        "expected_height_px"
                    ]
                )
            ).all()
        ),
        "view_assumption": bool(
            (
                canonical[
                    "view_assumption"
                ].astype(str)
                == str(
                    compat[
                        "expected_view_assumption"
                    ]
                )
            ).all()
        ),
        "image_path_exact": bool(
            canonical["image_path"]
            .astype(str)
            .reset_index(drop=True)
            .equals(
                legacy["image_path"]
                .astype(str)
                .reset_index(drop=True)
            )
        ),
        "query_id_exact": bool(
            canonical["query_id"]
            .reset_index(drop=True)
            .equals(
                pd.to_numeric(
                    legacy["query_id"],
                    errors="raise",
                )
                .astype(int)
                .reset_index(drop=True)
            )
        ),
        "frame_index_exact": bool(
            canonical["frame_index"]
            .reset_index(drop=True)
            .equals(
                pd.to_numeric(
                    legacy["frame_index"],
                    errors="raise",
                )
                .astype(int)
                .reset_index(drop=True)
            )
        ),
        "timestamp_exact": bool(
            (
                canonical["timestamp_s"]
                .to_numpy(float)
                == pd.to_numeric(
                    legacy["timestamp_s"],
                    errors="raise",
                ).to_numpy(float)
            ).all()
        ),
        "reference_false": bool(
            (
                canonical[
                    "reference_available"
                ]
                == False
            ).all()
        ),
    }

    return {
        "checks": checks,
        "pass": bool(
            all(checks.values())
        ),
        "query_count": int(
            len(canonical)
        ),
        "query_id_range": [
            int(
                query_ids.min()
            ),
            int(
                query_ids.max()
            ),
        ],
        "frame_index_range": [
            int(
                frame_indices.min()
            ),
            int(
                frame_indices.max()
            ),
        ],
        "timestamp_range_s": [
            float(
                timestamps.min()
            ),
            float(
                timestamps.max()
            ),
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "TA2: adapt the existing 403-query traj01 blind manifest "
            "to uavloc.trajectory.v1 canonical package format."
        )
    )
    parser.add_argument(
        "--trajectory",
        type=Path,
        default=DEFAULT_TRAJECTORY,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    parser.add_argument(
        "--check-input-paths",
        action="store_true",
        help=(
            "Check declared video/map/reference paths too. "
            "Reference is never read by TA2."
        ),
    )
    args = parser.parse_args()

    spec_path = resolve(
        args.trajectory
    )
    spec = load_trajectory_spec(
        spec_path,
        check_paths=args.check_input_paths,
    )

    if spec.trajectory_id != (
        "villoc_traj01_90deg_stable120m"
    ):
        raise RuntimeError(
            "TA2 is intentionally scoped to "
            "villoc_traj01_90deg_stable120m."
        )

    if spec.role != "development":
        raise RuntimeError(
            "TA2 traj01 must use role='development'."
        )

    legacy_path = resolve(
        spec.raw[
            "compatibility"
        ][
            "legacy_blind_manifest"
        ]
    )

    if not legacy_path.exists():
        raise FileNotFoundError(
            legacy_path
        )

    # Header check is performed before loading rows.
    header = pd.read_csv(
        legacy_path,
        nrows=0,
    )
    assert_blind_safe_columns(
        list(header.columns)
    )

    # Only blind-safe source data is read here.
    legacy = pd.read_csv(
        legacy_path
    )

    canonical = canonicalize_legacy_manifest(
        legacy=legacy,
        spec=spec,
    )

    parity = parity_report(
        legacy=legacy,
        canonical=canonical,
        spec=spec,
    )

    if not parity[
        "pass"
    ]:
        raise RuntimeError(
            "TA2 parity failed:\n"
            + json.dumps(
                parity,
                indent=2,
            )
        )

    output_root = resolve(
        args.output_root
    )
    blind_root = (
        output_root
        / "blind_package"
    )
    reports_root = (
        output_root
        / "reports"
    )

    blind_root.mkdir(
        parents=True,
        exist_ok=True,
    )
    reports_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    canonical_path = (
        blind_root
        / "canonical_blind_manifest.csv"
    )
    runtime_path = (
        blind_root
        / "trajectory_runtime_contract.json"
    )
    capability_path = (
        blind_root
        / "capabilities.json"
    )
    provenance_path = (
        blind_root
        / "provenance.json"
    )
    report_path = (
        reports_root
        / "ta2_traj01_adapter_report.json"
    )

    canonical.to_csv(
        canonical_path,
        index=False,
    )

    runtime_contract = (
        spec.blind_runtime_contract()
    )
    runtime_path.write_text(
        json.dumps(
            runtime_contract,
            indent=2,
        ),
        encoding="utf-8",
    )

    capability_path.write_text(
        json.dumps(
            spec.capability_summary(),
            indent=2,
        ),
        encoding="utf-8",
    )

    provenance = {
        "schema": (
            "uavloc.trajectory.provenance.v1"
        ),
        "created_at_utc": now_utc(),
        "trajectory_id": (
            spec.trajectory_id
        ),
        "trajectory_spec_path": str(
            args.trajectory
        ),
        "trajectory_spec_sha256": (
            sha256_file(
                spec_path
            )
        ),
        "source_manifest_path": str(
            spec.raw[
                "compatibility"
            ][
                "legacy_blind_manifest"
            ]
        ),
        "source_manifest_sha256": (
            sha256_file(
                legacy_path
            )
        ),
        "reference_read": False,
        "reference_path_exposed_in_blind_package": False,
    }

    provenance_path.write_text(
        json.dumps(
            provenance,
            indent=2,
        ),
        encoding="utf-8",
    )

    blind_hashes = {
        "canonical_blind_manifest": (
            sha256_file(
                canonical_path
            )
        ),
        "trajectory_runtime_contract": (
            sha256_file(
                runtime_path
            )
        ),
        "capabilities": (
            sha256_file(
                capability_path
            )
        ),
        "provenance": (
            sha256_file(
                provenance_path
            )
        ),
    }

    # Explicitly prove the blind runtime contract contains
    # neither reference path nor provider payload.
    runtime_text = runtime_path.read_text(
        encoding="utf-8"
    )
    reference_path_string = (
        spec.reference.path
        or ""
    )
    if (
        reference_path_string
        and reference_path_string
        in runtime_text
    ):
        raise RuntimeError(
            "Reference path leaked into blind runtime contract."
        )

    report = {
        "stage": "TA2",
        "status": (
            "PASS_TA2_TRAJ01_CANONICAL_ADAPTER"
        ),
        "created_at_utc": now_utc(),
        "trajectory_id": (
            spec.trajectory_id
        ),
        "role": spec.role,
        "source": {
            "type": spec.source_type,
            "legacy_blind_manifest": str(
                spec.raw[
                    "compatibility"
                ][
                    "legacy_blind_manifest"
                ]
            ),
            "legacy_blind_manifest_sha256": (
                sha256_file(
                    legacy_path
                )
            ),
        },
        "parity": parity,
        "signals": {
            name: {
                "source": signal.source,
                "value": signal.value,
                "dynamic": (
                    signal.is_dynamic
                ),
            }
            for name, signal in (
                spec.signals.items()
            )
        },
        "reference_boundary": {
            "declared_available": (
                spec.reference.available
            ),
            "mode": (
                spec.reference.mode
            ),
            "read_during_ta2": False,
            "included_in_canonical_manifest": False,
            "path_exposed_in_blind_package": False,
        },
        "blind_package": {
            "root": str(
                blind_root
            ),
            "hashes": blind_hashes,
        },
        "next_stage": {
            "stage": "TA3",
            "action": (
                "Adapt the recorded blind demo with reference.mode=unavailable "
                "using the same canonical package contract."
            ),
        },
    }

    report_path.write_text(
        json.dumps(
            report,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    print("=" * 104)
    print(
        "TA2 — TRAJ01 CANONICAL TRAJECTORY ADAPTER"
    )
    print("=" * 104)
    print(
        "trajectory:",
        spec.trajectory_id,
    )
    print(
        "role:",
        spec.role,
    )
    print(
        "queries:",
        parity[
            "query_count"
        ],
    )
    print(
        "query IDs:",
        parity[
            "query_id_range"
        ],
    )
    print(
        "frame indices:",
        parity[
            "frame_index_range"
        ],
    )
    print(
        "timestamps:",
        parity[
            "timestamp_range_s"
        ],
    )
    print(
        "exact image-path parity:",
        parity[
            "checks"
        ][
            "image_path_exact"
        ],
    )
    print(
        "exact timestamp parity:",
        parity[
            "checks"
        ][
            "timestamp_exact"
        ],
    )
    print(
        "reference declared:",
        spec.reference.available,
    )
    print(
        "reference read:",
        False,
    )
    print(
        "blind package:",
        blind_root,
    )
    print()
    print(
        "STATUS: PASS_TA2_TRAJ01_CANONICAL_ADAPTER"
    )
    print(
        "report:",
        report_path,
    )


if __name__ == "__main__":
    main()
