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

    submission_path = args.submission.resolve()
    report_path = args.addon9_report.resolve()
    run_root = args.run_root.resolve()

    if not submission_path.exists():
        raise FileNotFoundError(submission_path)

    if not report_path.exists():
        raise FileNotFoundError(report_path)

    df = pd.read_csv(submission_path)

    report = json.loads(
        report_path.read_text(
            encoding="utf-8"
        )
    )

    assert len(df) == 403

    assert (
        report["status"]
        == "PASS_ADDON9_ESTIMATED_LATLON_EXPORT"
    )

    available = bool_series(
        df["map_aligned_available"]
    )

    accepted = bool_series(
        df["accepted_correction"]
    )

    assert int(available.sum()) == 386
    assert int((~available).sum()) == 17
    assert int(accepted.sum()) == 14

    assert (
        df.loc[
            ~available,
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

    freeze = {
        "stage": (
            "STAGE_10B5D_FREEZE_BLIND_SUBMISSION"
        ),
        "status": (
            "PASS_BLIND_SUBMISSION_FROZEN"
        ),
        "frozen_at_utc": (
            datetime.now(timezone.utc)
            .isoformat()
        ),
        "blind_contract": {
            "gps_used": False,
            "srt_used": False,
            "reference_used": False,
            "evaluation_used_for_localization": False,
        },
        "submission": {
            "path": str(
                submission_path
            ),
            "sha256": submission_sha,
            "rows": 403,
            "map_positions_available": 386,
            "prelock_unavailable": 17,
            "accepted_corrections": 14,
        },
        "addon9_report": {
            "path": str(
                report_path
            ),
            "sha256": report_sha,
        },
        "coverage_note": {
            "outside_tile_index_bbox_rows": 8,
            "maximum_outside_distance_m": 3.1780870107468218,
            "action": "preserved_unmodified",
        },
        "evaluation_boundary": (
            "Reference/GT may be attached only after "
            "this frozen submission. Evaluation must "
            "not modify or regenerate localization."
        ),
    }

    freeze_path.write_text(
        json.dumps(
            freeze,
            indent=2,
        ),
        encoding="utf-8",
    )

    # Ensure writing the freeze record did not
    # alter the submission itself.
    assert (
        sha256_file(submission_path)
        == submission_sha
    )

    print("=" * 80)
    print(
        "STAGE 10B.5D — FREEZE BLIND SUBMISSION"
    )
    print("=" * 80)

    print()
    print(
        "submission rows          :",
        len(df),
    )

    print(
        "map positions available  :",
        int(available.sum()),
    )

    print(
        "accepted corrections     :",
        int(accepted.sum()),
    )

    print()
    print(
        "submission SHA256:"
    )

    print(submission_sha)

    print()
    print(
        "freeze record:"
    )

    print(freeze_path)

    print()
    print(
        "status: PASS_BLIND_SUBMISSION_FROZEN"
    )


if __name__ == "__main__":
    main()
