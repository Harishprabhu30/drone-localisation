#!/usr/bin/env python3
"""S7.0C — clean stratified trajectory sampling manifest builder.

Accepted inputs:
- 13-column frozen range CSV;
- 8-column preferred range CSV;
- earlier 8-column manual segment CSV.

The script sorts ranges, recomputes IDs and lengths, validates coverage, inserts only
genuine missing gaps as unreviewed, samples each range using its stride, joins canonical
frame metadata, and marks overlap with the frozen 263-query benchmark.

It does not use reference coordinates or retrieval results for selection.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


EXPECTED_START = 1
EXPECTED_END = 1034

FROZEN_COLUMNS = [
    "range_id",
    "source_range_id",
    "start_token0_id",
    "end_token0_id",
    "primary_scene",
    "secondary_scene",
    "stride",
    "selection_role",
    "confidence",
    "boundary_uncertainty_frames",
    "notes",
    "auto_inserted",
    "range_length_frames",
]

PREFERRED_COLUMNS = [
    "range_id",
    "start_token0_id",
    "end_token0_id",
    "primary_scene",
    "secondary_scene",
    "stride",
    "selection_role",
    "notes",
]

LEGACY_COLUMNS = [
    "segment_id",
    "start_token0_id",
    "end_token0_id",
    "primary_scene",
    "secondary_scene",
    "confidence",
    "boundary_uncertainty_frames",
    "notes",
]

PRIMARY_SCENES = {
    "unreviewed",
    "urban",
    "mixed_urban_natural",
    "forest_canopy",
    "agricultural_open_field",
    "water_wetland",
    "low_structure",
    "other",
}

SECONDARY_SCENES = {
    "none",
    "urban",
    "vegetation",
    "forest_canopy",
    "agricultural_open_field",
    "water_wetland",
    "road_corridor",
    "construction_bare_ground",
    "other",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def clean(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def parse_bool(value: Any) -> bool:
    return clean(value).lower() in {"1", "true", "yes", "y", "t"}


def parse_int(value: Any, field: str, line_number: int) -> int:
    try:
        return int(float(clean(value)))
    except Exception as exc:
        raise RuntimeError(
            f"Line {line_number}: {field} must be an integer, got {value!r}"
        ) from exc


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def tolerant_read(path: Path) -> tuple[pd.DataFrame, str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))

    if not rows:
        raise RuntimeError(f"Input CSV is empty: {path}")

    header = [clean(value) for value in rows[0]]
    if header == FROZEN_COLUMNS:
        schema = "frozen"
        expected = FROZEN_COLUMNS
    elif header == PREFERRED_COLUMNS:
        schema = "preferred"
        expected = PREFERRED_COLUMNS
    elif header == LEGACY_COLUMNS:
        schema = "legacy"
        expected = LEGACY_COLUMNS
    else:
        raise RuntimeError(
            "Unsupported range CSV header.\n"
            f"Found: {header}\n"
            f"Accepted: {FROZEN_COLUMNS}, {PREFERRED_COLUMNS}, or {LEGACY_COLUMNS}"
        )

    normalized: list[list[str]] = []
    width = len(expected)
    notes_index = expected.index("notes")

    for line_number, row in enumerate(rows[1:], start=2):
        if not row or not any(clean(value) for value in row):
            continue

        if len(row) < width:
            row = row + [""] * (width - len(row))
        elif len(row) > width:
            # Merge extra comma-separated pieces back into notes and preserve trailing fields.
            extra_count = len(row) - width
            merged_notes = ", ".join(
                clean(value)
                for value in row[notes_index : notes_index + extra_count + 1]
            )
            row = (
                row[:notes_index]
                + [merged_notes]
                + row[notes_index + extra_count + 1 :]
            )

        if len(row) != width:
            raise RuntimeError(
                f"Line {line_number}: could not normalize {len(row)} fields to {width}."
            )
        normalized.append([clean(value) for value in row])

    return pd.DataFrame(normalized, columns=expected, dtype=str), schema


def find_input(repo_root: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        candidate = explicit if explicit.is_absolute() else repo_root / explicit
        if not candidate.exists():
            raise FileNotFoundError(f"Input range CSV not found: {candidate}")
        return candidate.resolve()

    candidates = [
        repo_root
        / "outputs/satloc/metadata/s7_retrieval_upgrade/"
        "s7_0_scene_sampling_ranges.csv",
        repo_root
        / "outputs/satloc/metadata/s7_retrieval_upgrade/"
        "s7_0_scene_sampling_ranges_frozen.csv",
        repo_root
        / "outputs/satloc/metadata/s7_retrieval_upgrade/"
        "s7_0_scene_segments_manual.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    raise FileNotFoundError(
        "No S7.0C range CSV found. Checked:\n"
        + "\n".join(str(candidate) for candidate in candidates)
    )


def default_policy(primary_scene: str, length: int) -> tuple[int, str]:
    if primary_scene == "unreviewed":
        return (1, "gap_coverage") if length <= 10 else (10, "trajectory_coverage")
    return 5, "main_scene_region"


def normalize_ranges(
    raw: pd.DataFrame,
    schema: str,
    expected_start: int,
    expected_end: int,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    parsed: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []

    for index, row in raw.iterrows():
        line_number = index + 2
        start = parse_int(row["start_token0_id"], "start_token0_id", line_number)
        end = parse_int(row["end_token0_id"], "end_token0_id", line_number)

        if start > end:
            raise RuntimeError(f"Line {line_number}: reversed range {start}-{end}.")
        if start < expected_start or end > expected_end:
            raise RuntimeError(
                f"Line {line_number}: range {start}-{end} is outside "
                f"{expected_start}-{expected_end}."
            )

        primary = clean(row["primary_scene"]) or "unreviewed"
        secondary = clean(row["secondary_scene"]) or "none"

        if primary not in PRIMARY_SCENES:
            raise RuntimeError(
                f"Line {line_number}: invalid primary_scene {primary!r}."
            )
        if secondary not in SECONDARY_SCENES:
            raise RuntimeError(
                f"Line {line_number}: invalid secondary_scene {secondary!r}."
            )

        length = end - start + 1

        if schema == "frozen":
            source_range_id = clean(row["source_range_id"]) or clean(row["range_id"])
            supplied_stride = clean(row["stride"])
            supplied_role = clean(row["selection_role"])
            confidence = clean(row["confidence"])
            uncertainty = clean(row["boundary_uncertainty_frames"])
            notes = clean(row["notes"])
            auto_inserted = parse_bool(row["auto_inserted"])
        elif schema == "preferred":
            source_range_id = clean(row["range_id"])
            supplied_stride = clean(row["stride"])
            supplied_role = clean(row["selection_role"])
            confidence = ""
            uncertainty = ""
            notes = clean(row["notes"])
            auto_inserted = False
        else:
            source_range_id = clean(row["segment_id"])
            supplied_stride = ""
            supplied_role = ""
            confidence = clean(row["confidence"])
            uncertainty = clean(row["boundary_uncertainty_frames"])
            notes = clean(row["notes"])
            auto_inserted = False

        default_stride, default_role = default_policy(primary, length)
        stride = parse_int(supplied_stride, "stride", line_number) if supplied_stride else default_stride
        role = supplied_role or default_role

        if stride <= 0:
            raise RuntimeError(f"Line {line_number}: stride must be positive.")

        parsed.append(
            {
                "source_range_id": source_range_id,
                "start_token0_id": start,
                "end_token0_id": end,
                "primary_scene": primary,
                "secondary_scene": secondary,
                "stride": stride,
                "selection_role": role,
                "confidence": confidence,
                "boundary_uncertainty_frames": uncertainty,
                "notes": notes,
                "auto_inserted": auto_inserted,
            }
        )

    if not parsed:
        raise RuntimeError("No ranges were found.")

    parsed_df = pd.DataFrame(parsed).sort_values(
        ["start_token0_id", "end_token0_id"], kind="mergesort"
    ).reset_index(drop=True)

    complete: list[dict[str, Any]] = []
    cursor = expected_start
    gap_number = 0

    for _, row in parsed_df.iterrows():
        start = int(row["start_token0_id"])
        end = int(row["end_token0_id"])

        if start < cursor:
            previous = complete[-1]
            raise RuntimeError(
                "Overlapping ranges: "
                f"{previous['start_token0_id']}-{previous['end_token0_id']} and "
                f"{start}-{end}."
            )

        if start > cursor:
            gap_start = cursor
            gap_end = start - 1
            gap_length = gap_end - gap_start + 1
            gap_stride, gap_role = default_policy("unreviewed", gap_length)
            gap_number += 1
            complete.append(
                {
                    "source_range_id": f"auto_gap_{gap_number:02d}",
                    "start_token0_id": gap_start,
                    "end_token0_id": gap_end,
                    "primary_scene": "unreviewed",
                    "secondary_scene": "none",
                    "stride": gap_stride,
                    "selection_role": gap_role,
                    "confidence": "",
                    "boundary_uncertainty_frames": "",
                    "notes": "Automatically inserted uncovered chronological range",
                    "auto_inserted": True,
                }
            )
            warnings.append(
                {
                    "warning_type": "auto_inserted_gap",
                    "start_token0_id": gap_start,
                    "end_token0_id": gap_end,
                    "details": f"Inserted with stride {gap_stride}.",
                }
            )

        complete.append(row.to_dict())
        cursor = end + 1

    if cursor <= expected_end:
        gap_start = cursor
        gap_end = expected_end
        gap_length = gap_end - gap_start + 1
        gap_stride, gap_role = default_policy("unreviewed", gap_length)
        gap_number += 1
        complete.append(
            {
                "source_range_id": f"auto_gap_{gap_number:02d}",
                "start_token0_id": gap_start,
                "end_token0_id": gap_end,
                "primary_scene": "unreviewed",
                "secondary_scene": "none",
                "stride": gap_stride,
                "selection_role": gap_role,
                "confidence": "",
                "boundary_uncertainty_frames": "",
                "notes": "Automatically inserted uncovered chronological range",
                "auto_inserted": True,
            }
        )
        warnings.append(
            {
                "warning_type": "auto_inserted_gap",
                "start_token0_id": gap_start,
                "end_token0_id": gap_end,
                "details": f"Inserted with stride {gap_stride}.",
            }
        )

    final = pd.DataFrame(complete).sort_values(
        ["start_token0_id", "end_token0_id"], kind="mergesort"
    ).reset_index(drop=True)
    final.insert(0, "range_id", [f"range_{index:03d}" for index in range(1, len(final) + 1)])
    final["range_length_frames"] = (
        final["end_token0_id"].astype(int)
        - final["start_token0_id"].astype(int)
        + 1
    )

    coverage: list[int] = []
    for _, row in final.iterrows():
        coverage.extend(
            range(int(row["start_token0_id"]), int(row["end_token0_id"]) + 1)
        )

    expected = list(range(expected_start, expected_end + 1))
    if coverage != expected:
        raise RuntimeError("Normalized ranges do not cover token0_id 1..1034 exactly once.")

    final = final[FROZEN_COLUMNS]
    return final, warnings


def sample_ranges(ranges: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for _, range_row in ranges.iterrows():
        start = int(range_row["start_token0_id"])
        end = int(range_row["end_token0_id"])
        stride = int(range_row["stride"])

        selected = list(range(start, end + 1, stride))
        if selected[-1] != end:
            selected.append(end)

        for token in selected:
            rows.append(
                {
                    "token0_id": token,
                    "range_id": range_row["range_id"],
                    "source_range_id": range_row["source_range_id"],
                    "primary_scene": range_row["primary_scene"],
                    "secondary_scene": range_row["secondary_scene"],
                    "stride": stride,
                    "selection_role": range_row["selection_role"],
                    "is_range_start": token == start,
                    "is_range_end": token == end,
                    "range_start_token0_id": start,
                    "range_end_token0_id": end,
                    "auto_inserted_range": bool(range_row["auto_inserted"]),
                    "notes": range_row["notes"],
                }
            )

    sampled = pd.DataFrame(rows).sort_values("token0_id").reset_index(drop=True)
    if sampled["token0_id"].duplicated().any():
        duplicates = sampled.loc[
            sampled["token0_id"].duplicated(keep=False), "token0_id"
        ].tolist()
        raise RuntimeError(f"Duplicate sampled tokens: {duplicates}")

    sampled.insert(0, "sample_index", range(1, len(sampled) + 1))
    return sampled


def detect_token_column(frame: pd.DataFrame) -> str | None:
    for candidate in [
        "token0_id",
        "query_token0_id",
        "token",
        "query_token",
        "frame_token0_id",
    ]:
        if candidate in frame.columns:
            return candidate
    return None


def join_frame_audit(sampled: pd.DataFrame, path: Path) -> pd.DataFrame:
    if not path.exists():
        sampled["sequence_frame_id"] = ""
        sampled["resolved_image_path"] = ""
        sampled["frame_audit_available"] = False
        return sampled

    audit = pd.read_csv(path)
    token_column = detect_token_column(audit)
    if token_column is None:
        raise RuntimeError(f"No token column found in frame audit: {path}")

    audit[token_column] = pd.to_numeric(audit[token_column], errors="raise").astype(int)
    keep = [token_column]
    for column in ["sequence_frame_id", "resolved_image_path", "image_path", "read_ok"]:
        if column in audit.columns:
            keep.append(column)

    joined = sampled.merge(
        audit[keep].drop_duplicates(token_column),
        left_on="token0_id",
        right_on=token_column,
        how="left",
        validate="one_to_one",
    )
    if token_column != "token0_id":
        joined = joined.drop(columns=[token_column])
    if "resolved_image_path" not in joined.columns and "image_path" in joined.columns:
        joined = joined.rename(columns={"image_path": "resolved_image_path"})
    if "sequence_frame_id" not in joined.columns:
        joined["sequence_frame_id"] = ""
    if "resolved_image_path" not in joined.columns:
        joined["resolved_image_path"] = ""
    joined["frame_audit_available"] = True
    return joined


def mark_official_queries(sampled: pd.DataFrame, path: Path) -> tuple[pd.DataFrame, int, int]:
    sampled = sampled.copy()
    sampled["in_official_263_query_manifest"] = False

    if not path.exists():
        return sampled, 0, 0

    official = pd.read_csv(path)
    token_column = detect_token_column(official)
    if token_column is None:
        raise RuntimeError(f"No token column found in official query manifest: {path}")

    official_tokens = set(
        pd.to_numeric(official[token_column], errors="raise").astype(int).tolist()
    )
    sampled["in_official_263_query_manifest"] = sampled["token0_id"].isin(official_tokens)
    return (
        sampled,
        int(sampled["in_official_263_query_manifest"].sum()),
        len(official_tokens),
    )


def to_int_dict(series: pd.Series) -> dict[str, int]:
    return {str(key): int(value) for key, value in series.to_dict().items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--input-ranges", type=Path, default=None)
    parser.add_argument("--expected-start", type=int, default=EXPECTED_START)
    parser.add_argument("--expected-end", type=int, default=EXPECTED_END)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    metadata_dir = ensure_dir(
        repo_root / "outputs/satloc/metadata/s7_retrieval_upgrade"
    )
    reports_dir = ensure_dir(
        repo_root / "outputs/satloc/reports/s7_retrieval_upgrade"
    )

    input_path = find_input(repo_root, args.input_ranges)
    raw, schema = tolerant_read(input_path)
    ranges, warnings = normalize_ranges(
        raw, schema, args.expected_start, args.expected_end
    )
    sampled = sample_ranges(ranges)

    sampled = join_frame_audit(
        sampled, metadata_dir / "s7_0_frame_path_audit.csv"
    )
    sampled, official_overlap, official_count = mark_official_queries(
        sampled, metadata_dir / "s7_0_query_manifest.csv"
    )

    ranges_path = metadata_dir / "s7_0_scene_sampling_ranges_frozen.csv"
    sampled_path = metadata_dir / "s7_0_scene_sampled_frames.csv"
    warnings_path = metadata_dir / "s7_0_scene_sampling_warnings.csv"
    summary_path = metadata_dir / "s7_0_scene_sampling_summary.json"
    report_path = reports_dir / "s7_0_scene_sampling_report.md"

    ranges.to_csv(ranges_path, index=False)
    sampled.to_csv(sampled_path, index=False)
    pd.DataFrame(
        warnings,
        columns=["warning_type", "start_token0_id", "end_token0_id", "details"],
    ).to_csv(warnings_path, index=False)

    summary = {
        "generated_utc": utc_now(),
        "status": "PASS",
        "stage": "S7.0C",
        "sequence": "traj01",
        "input_csv": str(input_path.relative_to(repo_root)),
        "input_schema": schema,
        "normalized_range_count": int(len(ranges)),
        "auto_inserted_range_count": int(ranges["auto_inserted"].astype(bool).sum()),
        "sampled_frame_count": int(len(sampled)),
        "sampled_unique_token_count": int(sampled["token0_id"].nunique()),
        "primary_scene_sample_counts": to_int_dict(
            sampled["primary_scene"].value_counts().sort_index()
        ),
        "selection_role_sample_counts": to_int_dict(
            sampled["selection_role"].value_counts().sort_index()
        ),
        "stride_sample_counts": to_int_dict(
            sampled["stride"].value_counts().sort_index()
        ),
        "official_query_manifest_token_count": int(official_count),
        "sampled_official_query_intersection": int(official_overlap),
        "scientific_role": (
            "Scene-focused diagnostic/development manifest. "
            "The frozen official 263-query benchmark remains unchanged."
        ),
        "ground_truth_isolation": (
            "Selection uses user-reviewed visual ranges, token0_id order, and image-path "
            "metadata only; no reference coordinates or retrieval results."
        ),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    table = ranges[
        [
            "range_id",
            "start_token0_id",
            "end_token0_id",
            "primary_scene",
            "secondary_scene",
            "stride",
            "selection_role",
            "range_length_frames",
        ]
    ].to_markdown(index=False)

    report = f"""# S7.0C — Stratified Trajectory Sampling Manifest

Generated: `{summary["generated_utc"]}`

## Status

```text
PASS
```

## Clean chronological ranges

{table}

## Result

```text
Input schema:             {schema}
Normalized ranges:        {len(ranges)}
Auto-inserted ranges:     {int(ranges["auto_inserted"].astype(bool).sum())}
Selected frames:          {len(sampled)}
Unique selected tokens:   {sampled["token0_id"].nunique()}
Official-query overlap:   {official_overlap} / {official_count if official_count else "not available"}
```

The sampled manifest supports scene-focused diagnostics. The original 263-query
official retrieval benchmark is unchanged.
"""
    report_path.write_text(report, encoding="utf-8")

    print("S7.0C Stratified Trajectory Sampling Manifest")
    print("---------------------------------------------")
    print("Status:                 PASS")
    print(f"Input schema:           {schema}")
    print(f"Normalized ranges:      {len(ranges)}")
    print(f"Auto-inserted ranges:   {int(ranges['auto_inserted'].astype(bool).sum())}")
    print(f"Selected frames:        {len(sampled)}")
    print(f"Unique selected tokens: {sampled['token0_id'].nunique()}")
    if official_count:
        print(f"Official-263 overlap:   {official_overlap} / {official_count}")
    else:
        print("Official-263 overlap:   manifest not found")
    print(f"Frozen ranges:          {ranges_path.relative_to(repo_root)}")
    print(f"Sampled frames:         {sampled_path.relative_to(repo_root)}")
    print(f"Summary:                {summary_path.relative_to(repo_root)}")
    print(f"Report:                 {report_path.relative_to(repo_root)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print("S7.0C Stratified Trajectory Sampling Manifest", file=sys.stderr)
        print("---------------------------------------------", file=sys.stderr)
        print("Status: BLOCKED", file=sys.stderr)
        print(f"Reason: {exc}", file=sys.stderr)
        raise
