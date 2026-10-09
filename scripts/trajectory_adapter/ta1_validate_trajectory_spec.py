#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from uavloc.data.trajectory_adapter import (
    CANONICAL_BLIND_COLUMNS,
    FORBIDDEN_BLIND_REFERENCE_COLUMNS,
    load_trajectory_spec,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TA1: validate one uavloc.trajectory.v1 trajectory spec."
    )
    parser.add_argument(
        "--trajectory",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--check-paths",
        action="store_true",
    )
    args = parser.parse_args()

    spec = load_trajectory_spec(
        args.trajectory,
        check_paths=args.check_paths,
    )

    print("=" * 96)
    print("TA1 — TRAJECTORY CONTRACT VALIDATION")
    print("=" * 96)
    print(json.dumps(spec.capability_summary(), indent=2))
    print()
    print("canonical blind columns:")
    print(json.dumps(list(CANONICAL_BLIND_COLUMNS), indent=2))
    print()
    print("forbidden blind reference columns:")
    print(json.dumps(sorted(FORBIDDEN_BLIND_REFERENCE_COLUMNS), indent=2))
    print()
    print("STATUS: PASS_TA1_TRAJECTORY_CONTRACT")


if __name__ == "__main__":
    main()
