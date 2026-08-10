#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def bool_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False)

    return (
        s.astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes", "y", "t"})
    )


def main() -> None:

    p = argparse.ArgumentParser()

    p.add_argument(
        "--submission",
        type=Path,
        required=True,
    )

    p.add_argument(
        "--addon9-report",
        type=Path,
        required=True,
    )

    p.add_argument(
        "--run-root",
        type=Path,
        required=True,
    )

    args = p.parse_args()

    submission_path = (
        args.submission.resolve()
    )

    report_path = (
        args.addon9_report.resolve()
    )

    run_root = (
        args.run_root.resolve()
    )

    if not submission_path.exists():
        raise FileNotFoundError(
            submission_path
        )

    if not report_path.exists():
        raise FileNotFoundError(
            report_path
        )

    df = pd.read_csv(
        submission_path
    )

    report = json.loads(
        report_path.read_text(
            encoding="utf-8"
        )
    )

    if len(df) <= 0:
        raise RuntimeError(
            "Submission trajectory is empty."
        )

    allowed_addon9_status = {
        "PASS_ADDON9_ESTIMATED_LATLON_EXPORT",
        (
            "PASS_ADDON9_NO_ABSOLUTE_"
            "EXPORT_NO_MAP_LOCK"
        ),
        (
            "PASS_ADDON9_NO_ABSOLUTE_"
            "EXPORT_NO_MAP_STATE"
        ),
    }

    if (
        report.get("status")
        not in allowed_addon9_status
    ):
        raise RuntimeError(
            "Unexpected Add-on 9 status: "
            f"{report.get('status')!r}"
        )

    required_columns = [
        "map_aligned_available",
        "accepted_correction",
        "estimated_map_x",
        "estimated_map_y",
        "estimated_lat",
        "estimated_lon",
    ]

    missing = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise RuntimeError(
            "Submission missing freeze fields: "
            f"{missing}"
        )

    available = bool_series(
        df[
            "map_aligned_available"
        ]
    )

    accepted = bool_series(
        df[
            "accepted_correction"
        ]
    )

    unavailable = ~available

    # Any pose without map alignment must not
    # contain fabricated map/geographic output.
    if not (
        df.loc[
            unavailable,
            [
                "estimated_map_x",
                "estimated_map_y",
                "estimated_lat",
                "estimated_lon",
            ],
        ]
        .isna()
        .all()
        .all()
    ):
        raise RuntimeError(
            "Unavailable map rows contain "
            "map/geographic coordinates."
        )

    if (
        "localization_state"
        in df.columns
    ):

        states = (
            df[
                "localization_state"
            ]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        if len(states) != 1:
            raise RuntimeError(
                "Submission contains mixed "
                f"localization states: {states}"
            )

        localization_state = (
            states[0]
        )

    else:

        localization_state = (
            "ABSOLUTE_LOCKED"
            if available.any()
            else "NO_TRUSTED_ABSOLUTE_LOCK"
        )

    no_lock = (
        localization_state in {
            "NO_TRUSTED_ABSOLUTE_LOCK",
            "NO_PROVISIONAL_LOCK",
        }
    )

    if no_lock:

        if available.any():
            raise RuntimeError(
                "NO_TRUSTED_ABSOLUTE_LOCK "
                "contains map-aligned poses."
            )

        if accepted.any():
            raise RuntimeError(
                "NO_TRUSTED_ABSOLUTE_LOCK "
                "contains accepted corrections."
            )

        for col in [
            "estimated_map_x",
            "estimated_map_y",
            "estimated_lat",
            "estimated_lon",
        ]:
            if not (
                pd.to_numeric(
                    df[col],
                    errors="coerce",
                )
                .isna()
                .all()
            ):
                raise RuntimeError(
                    "No-lock submission contains "
                    f"unexpected {col} values."
                )

        expected_status = (
            (
                "PASS_ADDON9_NO_ABSOLUTE_"
                "EXPORT_NO_MAP_STATE"
            )
            if localization_state
            == "NO_PROVISIONAL_LOCK"
            else (
                "PASS_ADDON9_NO_ABSOLUTE_"
                "EXPORT_NO_MAP_LOCK"
            )
        )

        if (
            report.get("status")
            != expected_status
        ):
            raise RuntimeError(
                "No-lock submission is inconsistent "
                "with Add-on 9 report status."
            )

    else:

        if not available.any():
            raise RuntimeError(
                "Absolute localization state has "
                "no map-aligned poses."
            )

    freeze_dir = (
        run_root
        / "evaluation"
    )

    freeze_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    freeze_path = (
        freeze_dir
        / "blind_submission_freeze.json"
    )

    submission_sha = sha256_file(
        submission_path
    )

    report_sha = sha256_file(
        report_path
    )

    # Idempotent safety:
    # once frozen, a changed submission may not
    # silently replace the existing freeze record.
    if freeze_path.exists():

        existing = json.loads(
            freeze_path.read_text(
                encoding="utf-8"
            )
        )

        existing_sha = (
            existing
            .get(
                "submission",
                {},
            )
            .get(
                "sha256"
            )
        )

        if (
            existing_sha
            and existing_sha
            != submission_sha
        ):
            raise RuntimeError(
                "A freeze record already exists "
                "for a different submission SHA256. "
                "Refusing to overwrite the evaluation "
                "boundary."
            )

    map_positions = int(
        available.sum()
    )

    unavailable_positions = int(
        unavailable.sum()
    )

    accepted_count = int(
        accepted.sum()
    )

    latlon_available = (
        pd.to_numeric(
            df["estimated_lat"],
            errors="coerce",
        ).notna()
        & pd.to_numeric(
            df["estimated_lon"],
            errors="coerce",
        ).notna()
    )

    freeze = {
        "stage": (
            "STAGE_10B5D_FREEZE_BLIND_SUBMISSION"
        ),

        "status": (
            "PASS_BLIND_SUBMISSION_FROZEN"
        ),

        "registry_mode": (
            "PASS_GENERIC_BLIND_SUBMISSION_FREEZE"
        ),

        "frozen_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),

        "blind_contract": {
            "gps_used": False,
            "srt_used": False,
            "reference_used": False,
            "evaluation_used_for_localization":
                False,
            "reference_allowed_after_freeze":
                True,
        },

        "localization_state":
            localization_state,

        "submission": {
            "path":
                str(
                    submission_path
                ),

            "sha256":
                submission_sha,

            "rows":
                int(
                    len(df)
                ),

            "map_positions_available":
                map_positions,

            "map_positions_unavailable":
                unavailable_positions,

            "estimated_latlon_available":
                int(
                    latlon_available.sum()
                ),

            "accepted_corrections":
                accepted_count,
        },

        "addon9_report": {
            "path":
                str(
                    report_path
                ),

            "sha256":
                report_sha,

            "status":
                report.get(
                    "status"
                ),
        },

        "coverage_note": (
            report.get(
                "map_coverage_sanity"
            )
            if not no_lock
            else {
                "status":
                    "NOT_APPLICABLE_NO_MAP_LOCK",

                "reason": (
                    "No map/geographic estimate "
                    "was produced."
                ),
            }
        ),

        "evaluation_boundary": (
            "Reference/GT may be attached only "
            "after this frozen submission. "
            "Evaluation must not modify or "
            "regenerate localization."
        ),
    }

    freeze_path.write_text(
        json.dumps(
            freeze,
            indent=2,
        ),
        encoding="utf-8",
    )

    # Freeze operation itself must not alter
    # either source artifact.
    if (
        sha256_file(
            submission_path
        )
        != submission_sha
    ):
        raise RuntimeError(
            "Submission changed while freezing."
        )

    if (
        sha256_file(
            report_path
        )
        != report_sha
    ):
        raise RuntimeError(
            "Add-on 9 report changed while freezing."
        )

    print("=" * 88)
    print(
        "STAGE 10B.5D — FREEZE BLIND SUBMISSION"
    )
    print("=" * 88)

    print()

    print(
        "localization state       :",
        localization_state,
    )

    print(
        "submission rows          :",
        len(df),
    )

    print(
        "map positions available  :",
        map_positions,
    )

    print(
        "map positions unavailable:",
        unavailable_positions,
    )

    print(
        "estimated lat/lon poses  :",
        int(
            latlon_available.sum()
        ),
    )

    print(
        "accepted corrections     :",
        accepted_count,
    )

    print()

    print(
        "submission SHA256:"
    )

    print(
        submission_sha
    )

    print()

    print(
        "freeze record:"
    )

    print(
        freeze_path
    )

    print()

    print(
        "registry mode: "
        "PASS_GENERIC_BLIND_SUBMISSION_FREEZE"
    )

    print(
        "status: "
        "PASS_BLIND_SUBMISSION_FROZEN"
    )


if __name__ == "__main__":
    main()
