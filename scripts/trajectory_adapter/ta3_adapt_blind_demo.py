#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from scripts.trajectory_adapter.ta2_adapt_traj01 import (
    assert_blind_safe_columns,
    canonicalize_legacy_manifest,
    parity_report,
)
from uavloc.data.trajectory_adapter import (
    load_trajectory_spec,
)


ROOT = Path.cwd().resolve()
DEFAULT_TRAJECTORY = Path(
    "configs/trajectories/villoc_blind_recorded_flight_final_001_v1.yaml"
)
DEFAULT_OUTPUT_ROOT = Path(
    "outputs/research_runs/trajectory_adapter_v1/ta3_blind_demo"
)


def resolve(path_like):
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "TA3: adapt the frozen recorded blind demo into the "
            "uavloc.trajectory.v1 canonical blind package."
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
    )
    args = parser.parse_args()

    spec_path = resolve(args.trajectory)
    spec = load_trajectory_spec(
        spec_path,
        check_paths=args.check_input_paths,
    )

    if spec.role != "blind_stress":
        raise RuntimeError(
            "TA3 recorded demo must use role='blind_stress'."
        )

    if spec.reference.available:
        raise RuntimeError(
            "TA3 requires reference.mode='unavailable'."
        )

    legacy_path = resolve(
        spec.raw["compatibility"]["legacy_blind_manifest"]
    )
    if not legacy_path.exists():
        raise FileNotFoundError(legacy_path)

    header = pd.read_csv(
        legacy_path,
        nrows=0,
    )
    assert_blind_safe_columns(
        list(header.columns)
    )

    legacy = pd.read_csv(legacy_path)

    canonical = canonicalize_legacy_manifest(
        legacy=legacy,
        spec=spec,
    )

    parity = parity_report(
        legacy=legacy,
        canonical=canonical,
        spec=spec,
    )

    if not parity["pass"]:
        raise RuntimeError(
            "TA3 blind-demo parity failed:\n"
            + json.dumps(
                parity,
                indent=2,
            )
        )

    output_root = resolve(args.output_root)
    blind_root = output_root / "blind_package"
    reports_root = output_root / "reports"

    blind_root.mkdir(
        parents=True,
        exist_ok=True,
    )
    reports_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    canonical_path = blind_root / "canonical_blind_manifest.csv"
    runtime_path = blind_root / "trajectory_runtime_contract.json"
    capability_path = blind_root / "capabilities.json"
    provenance_path = blind_root / "provenance.json"
    report_path = reports_root / "ta3_blind_demo_adapter_report.json"

    canonical.to_csv(
        canonical_path,
        index=False,
    )

    runtime_contract = spec.blind_runtime_contract()
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
        "schema": "uavloc.trajectory.provenance.v1",
        "created_at_utc": now_utc(),
        "trajectory_id": spec.trajectory_id,
        "role": spec.role,
        "trajectory_spec_path": str(args.trajectory),
        "trajectory_spec_sha256": sha256_file(spec_path),
        "source_manifest_path": str(
            spec.raw["compatibility"]["legacy_blind_manifest"]
        ),
        "source_manifest_sha256": sha256_file(legacy_path),
        "reference_declared_available": False,
        "reference_read": False,
        "reference_path_exposed_in_blind_package": False,
        "ground_truth_required_for_adapter": False,
    }

    provenance_path.write_text(
        json.dumps(
            provenance,
            indent=2,
        ),
        encoding="utf-8",
    )

    blind_hashes = {
        "canonical_blind_manifest": sha256_file(canonical_path),
        "trajectory_runtime_contract": sha256_file(runtime_path),
        "capabilities": sha256_file(capability_path),
        "provenance": sha256_file(provenance_path),
    }

    runtime_text = runtime_path.read_text(
        encoding="utf-8",
    ).lower()

    forbidden_runtime_terms = (
        "srt",
        "gps",
        "gnss",
        "reference.path",
        "eval_ref_",
    )

    present = [
        term
        for term in forbidden_runtime_terms
        if term in runtime_text
    ]

    if present:
        raise RuntimeError(
            "Reference-like terms leaked into TA3 blind runtime contract: "
            f"{present}"
        )

    report = {
        "stage": "TA3",
        "status": "PASS_TA3_BLIND_DEMO_CANONICAL_ADAPTER",
        "created_at_utc": now_utc(),
        "trajectory_id": spec.trajectory_id,
        "role": spec.role,
        "parity": parity,
        "signals": {
            name: {
                "source": signal.source,
                "value": signal.value,
                "dynamic": signal.is_dynamic,
            }
            for name, signal in spec.signals.items()
        },
        "reference_boundary": {
            "declared_available": False,
            "mode": spec.reference.mode,
            "reference_path_declared": False,
            "read_during_ta3": False,
            "ground_truth_required": False,
            "included_in_canonical_manifest": False,
        },
        "blind_package": {
            "root": str(blind_root),
            "hashes": blind_hashes,
        },
        "research_policy": spec.raw.get(
            "research_policy",
            {},
        ),
        "next_stage": {
            "stage": "TA4",
            "action": (
                "Run one trajectory-independent frozen candidate-generation "
                "research harness against the TA2 and TA3 canonical packages. "
                "TA2 permits post-freeze accuracy evaluation; TA3 permits "
                "blind behavioral/stress diagnostics only."
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
    print("TA3 — BLIND DEMO CANONICAL TRAJECTORY ADAPTER")
    print("=" * 104)
    print("trajectory:", spec.trajectory_id)
    print("role:", spec.role)
    print("queries:", parity["query_count"])
    print("query IDs:", parity["query_id_range"])
    print("frame indices:", parity["frame_index_range"])
    print("timestamps:", parity["timestamp_range_s"])
    print(
        "exact image-path parity:",
        parity["checks"]["image_path_exact"],
    )
    print(
        "exact timestamp parity:",
        parity["checks"]["timestamp_exact"],
    )
    print(
        "reference declared:",
        spec.reference.available,
    )
    print("reference read:", False)
    print("ground truth required:", False)
    print("blind package:", blind_root)
    print()
    print("STATUS: PASS_TA3_BLIND_DEMO_CANONICAL_ADAPTER")
    print("report:", report_path)


if __name__ == "__main__":
    main()
