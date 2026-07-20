#!/usr/bin/env python3
"""S7.0C — scene-annotation validation, targeted refinement, and final expansion.

This block:
1. validates the coarse manual scene-range CSV;
2. reports invalid taxonomy values, gaps, overlaps, and ordering problems;
3. generates dense contact sheets only around uncertain boundaries and uncovered ranges;
4. once the CSV is complete and valid, expands ranges into one row per traj01 frame;
5. writes a frozen annotation summary.

It uses UAV imagery and canonical token0_id ordering only. It does not parse reference
coordinates from filenames and does not read retrieval scores, oracle labels, or errors.

Typical review run:

    python scripts/satloc/s7/s7_0c_scene_annotation_refinement.py \
      2>&1 | tee outputs/satloc/reports/s7_retrieval_upgrade/s7_0c_scene_annotation_refinement.log

After correcting the manual CSV, run the same command again. When all ranges are valid
and cover token0_id 1..1034 exactly once, the script automatically writes the final
frame-level labels and returns PASS_FROZEN.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import pandas as pd


DEFAULT_OUTPUT_ROOT = Path("outputs/satloc")
DEFAULT_MANUAL_CSV = Path(
    "outputs/satloc/metadata/s7_retrieval_upgrade/s7_0_scene_segments_manual.csv"
)
DEFAULT_FRAME_AUDIT = Path(
    "outputs/satloc/metadata/s7_retrieval_upgrade/s7_0_frame_path_audit.csv"
)

MANUAL_COLUMNS = [
    "segment_id",
    "start_token0_id",
    "end_token0_id",
    "primary_scene",
    "secondary_scene",
    "confidence",
    "boundary_uncertainty_frames",
    "notes",
]

PRIMARY_CLASSES = {
    "urban",
    "mixed_urban_natural",
    "forest_canopy",
    "agricultural_open_field",
    "water_wetland",
    "low_structure",
    "other",
}

SECONDARY_CLASSES = {
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

CONFIDENCE_VALUES = {"high", "medium", "low"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def norm_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def to_int(value: object, field: str, row_number: int) -> int:
    try:
        return int(float(value))
    except Exception as exc:
        raise ValueError(
            f"Row {row_number}: {field} must be an integer, got {value!r}"
        ) from exc


def contiguous_ranges(values: Iterable[int]) -> list[tuple[int, int]]:
    ordered = sorted(set(int(v) for v in values))
    if not ordered:
        return []
    ranges: list[tuple[int, int]] = []
    start = prev = ordered[0]
    for value in ordered[1:]:
        if value == prev + 1:
            prev = value
            continue
        ranges.append((start, prev))
        start = prev = value
    ranges.append((start, prev))
    return ranges


def letterbox(image: np.ndarray | None, width: int, height: int) -> np.ndarray:
    canvas = np.full((height, width, 3), 235, dtype=np.uint8)
    if image is None or image.size == 0:
        cv2.putText(
            canvas,
            "IMAGE READ FAILED",
            (15, height // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
        return canvas

    h, w = image.shape[:2]
    scale = min(width / max(w, 1), height / max(h, 1))
    rw = max(1, int(round(w * scale)))
    rh = max(1, int(round(h * scale)))
    resized = cv2.resize(image, (rw, rh), interpolation=cv2.INTER_AREA)
    x0 = (width - rw) // 2
    y0 = (height - rh) // 2
    canvas[y0:y0 + rh, x0:x0 + rw] = resized
    return canvas


def draw_sheet(
    frames: pd.DataFrame,
    output_path: Path,
    title: str,
    columns: int = 5,
    cell_width: int = 300,
    image_height: int = 205,
    title_height: int = 55,
) -> list[dict[str, object]]:
    rows = max(1, math.ceil(len(frames) / columns))
    header_h = 48
    sheet = np.full(
        (header_h + rows * (image_height + title_height), columns * cell_width, 3),
        255,
        dtype=np.uint8,
    )
    cv2.putText(
        sheet,
        title,
        (12, 31),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )

    manifest: list[dict[str, object]] = []
    for i, (_, row) in enumerate(frames.iterrows()):
        rr = i // columns
        cc = i % columns
        x0 = cc * cell_width
        y0 = header_h + rr * (image_height + title_height)

        image_path = Path(str(row["resolved_image_path"]))
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR) if image_path.exists() else None
        read_ok = image is not None and image.size > 0
        sheet[y0:y0 + image_height, x0:x0 + cell_width] = letterbox(
            image, cell_width, image_height
        )

        token = int(row["token0_id"])
        seq_id = norm_text(row.get("sequence_frame_id", ""))
        cv2.putText(
            sheet,
            f"token0_id={token}",
            (x0 + 7, y0 + image_height + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            sheet,
            f"sequence_frame_id={seq_id}",
            (x0 + 7, y0 + image_height + 43),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
        cv2.rectangle(
            sheet,
            (x0, y0),
            (x0 + cell_width - 1, y0 + image_height + title_height - 1),
            (180, 180, 180),
            1,
        )
        manifest.append(
            {
                "sheet_name": output_path.name,
                "cell_index": i,
                "token0_id": token,
                "sequence_frame_id": seq_id,
                "resolved_image_path": str(image_path),
                "read_ok": bool(read_ok),
            }
        )

    ensure_dir(output_path.parent)
    if not cv2.imwrite(str(output_path), sheet):
        raise RuntimeError(f"Failed to write contact sheet: {output_path}")
    return manifest



def read_manual_csv_tolerant(path: Path) -> pd.DataFrame:
    """Read the manual CSV while allowing unquoted commas inside the final notes field.

    The first seven commas define the first seven columns. Any additional comma-separated
    pieces are merged back into the eighth `notes` column.
    """
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))

    if not rows:
        return pd.DataFrame(columns=MANUAL_COLUMNS)

    header = [str(value).strip() for value in rows[0]]
    if header[: len(MANUAL_COLUMNS)] != MANUAL_COLUMNS:
        raise RuntimeError(
            "Unexpected manual CSV header. Expected exactly: "
            + ",".join(MANUAL_COLUMNS)
        )

    normalized_rows: list[list[str]] = []
    for line_number, row in enumerate(rows[1:], start=2):
        if not row or not any(str(value).strip() for value in row):
            continue

        if len(row) < len(MANUAL_COLUMNS):
            row = row + [""] * (len(MANUAL_COLUMNS) - len(row))
        elif len(row) > len(MANUAL_COLUMNS):
            # Preserve the first seven fields and merge the remainder into notes.
            row = row[:7] + [", ".join(part.strip() for part in row[7:])]

        normalized_rows.append(row)

    return pd.DataFrame(normalized_rows, columns=MANUAL_COLUMNS, dtype=str)


def validate_manual(
    manual: pd.DataFrame,
    expected_start: int,
    expected_end: int,
) -> tuple[pd.DataFrame, list[dict[str, object]], list[tuple[int, int]], list[tuple[int, int]]]:
    issues: list[dict[str, object]] = []
    missing_cols = [c for c in MANUAL_COLUMNS if c not in manual.columns]
    if missing_cols:
        raise RuntimeError(f"Manual CSV is missing columns: {missing_cols}")

    work = manual[MANUAL_COLUMNS].copy()
    work = work[work.apply(lambda r: any(norm_text(v) for v in r), axis=1)].copy()
    parsed_rows: list[dict[str, object]] = []

    for idx, row in work.iterrows():
        row_number = int(idx) + 2
        record = {c: norm_text(row[c]) for c in MANUAL_COLUMNS}
        try:
            start = to_int(row["start_token0_id"], "start_token0_id", row_number)
            end = to_int(row["end_token0_id"], "end_token0_id", row_number)
        except ValueError as exc:
            issues.append(
                {
                    "issue_type": "invalid_integer",
                    "row_number": row_number,
                    "segment_id": record["segment_id"],
                    "details": str(exc),
                }
            )
            continue

        primary = record["primary_scene"]
        secondary = record["secondary_scene"] or "none"
        confidence = record["confidence"].lower()
        record["secondary_scene"] = secondary
        record["confidence"] = confidence

        if start > end:
            issues.append(
                {
                    "issue_type": "reversed_range",
                    "row_number": row_number,
                    "segment_id": record["segment_id"],
                    "details": f"start={start} is greater than end={end}",
                }
            )
        if start < expected_start or end > expected_end:
            issues.append(
                {
                    "issue_type": "out_of_bounds",
                    "row_number": row_number,
                    "segment_id": record["segment_id"],
                    "details": f"range {start}-{end} must stay within {expected_start}-{expected_end}",
                }
            )
        if primary not in PRIMARY_CLASSES:
            issues.append(
                {
                    "issue_type": "invalid_primary_scene",
                    "row_number": row_number,
                    "segment_id": record["segment_id"],
                    "details": (
                        f"{primary!r} is not allowed. Choose one of: "
                        + ", ".join(sorted(PRIMARY_CLASSES))
                    ),
                }
            )
        if secondary not in SECONDARY_CLASSES:
            issues.append(
                {
                    "issue_type": "invalid_secondary_scene",
                    "row_number": row_number,
                    "segment_id": record["segment_id"],
                    "details": (
                        f"{secondary!r} is not allowed. Choose one of: "
                        + ", ".join(sorted(SECONDARY_CLASSES))
                    ),
                }
            )
        if confidence not in CONFIDENCE_VALUES:
            issues.append(
                {
                    "issue_type": "invalid_confidence",
                    "row_number": row_number,
                    "segment_id": record["segment_id"],
                    "details": (
                        f"{confidence!r} is not allowed. Choose high, medium, or low."
                    ),
                }
            )

        try:
            uncertainty = int(float(record["boundary_uncertainty_frames"] or 0))
        except Exception:
            uncertainty = -1
            issues.append(
                {
                    "issue_type": "invalid_boundary_uncertainty",
                    "row_number": row_number,
                    "segment_id": record["segment_id"],
                    "details": (
                        f"{record['boundary_uncertainty_frames']!r} must be a non-negative integer"
                    ),
                }
            )
        if uncertainty < 0:
            issues.append(
                {
                    "issue_type": "invalid_boundary_uncertainty",
                    "row_number": row_number,
                    "segment_id": record["segment_id"],
                    "details": "boundary_uncertainty_frames cannot be negative",
                }
            )

        parsed_rows.append(
            {
                **record,
                "start_token0_id": start,
                "end_token0_id": end,
                "boundary_uncertainty_frames": max(0, uncertainty),
                "_source_row_number": row_number,
            }
        )

    parsed = pd.DataFrame(parsed_rows)
    if parsed.empty:
        return parsed, issues, [(expected_start, expected_end)], []

    chronological = parsed.sort_values(
        ["start_token0_id", "end_token0_id"], kind="mergesort"
    ).reset_index(drop=True)
    original_order = list(parsed["start_token0_id"])
    sorted_order = list(chronological["start_token0_id"])
    if original_order != sorted_order:
        issues.append(
            {
                "issue_type": "non_chronological_order",
                "row_number": "",
                "segment_id": "",
                "details": "Rows must be ordered by start_token0_id before freezing.",
            }
        )

    coverage_count = np.zeros(expected_end - expected_start + 1, dtype=np.int32)
    for _, row in chronological.iterrows():
        lo = max(expected_start, int(row["start_token0_id"]))
        hi = min(expected_end, int(row["end_token0_id"]))
        if lo <= hi:
            coverage_count[lo - expected_start:hi - expected_start + 1] += 1

    uncovered_values = np.where(coverage_count == 0)[0] + expected_start
    overlap_values = np.where(coverage_count > 1)[0] + expected_start
    uncovered = contiguous_ranges(uncovered_values.tolist())
    overlaps = contiguous_ranges(overlap_values.tolist())

    for start, end in uncovered:
        issues.append(
            {
                "issue_type": "coverage_gap",
                "row_number": "",
                "segment_id": "",
                "details": f"Uncovered token0_id range: {start}-{end}",
            }
        )
    for start, end in overlaps:
        issues.append(
            {
                "issue_type": "coverage_overlap",
                "row_number": "",
                "segment_id": "",
                "details": f"Overlapping token0_id range: {start}-{end}",
            }
        )

    duplicate_ids = chronological["segment_id"][
        chronological["segment_id"].duplicated(keep=False)
    ].tolist()
    if duplicate_ids:
        issues.append(
            {
                "issue_type": "duplicate_segment_id",
                "row_number": "",
                "segment_id": "",
                "details": f"Duplicate segment IDs: {sorted(set(duplicate_ids))}",
            }
        )

    return chronological, issues, uncovered, overlaps


def select_sampled_range(
    audit: pd.DataFrame,
    start: int,
    end: int,
    stride: int,
) -> pd.DataFrame:
    subset = audit[
        (audit["token0_id"] >= start) & (audit["token0_id"] <= end)
    ].sort_values("token0_id")
    if subset.empty:
        return subset
    sampled = subset.iloc[::max(1, stride)].copy()
    if int(sampled.iloc[-1]["token0_id"]) != int(subset.iloc[-1]["token0_id"]):
        sampled = pd.concat([sampled, subset.iloc[[-1]]], ignore_index=True)
    return sampled.drop_duplicates("token0_id").sort_values("token0_id")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--manual-csv", type=Path, default=DEFAULT_MANUAL_CSV)
    parser.add_argument("--frame-audit", type=Path, default=DEFAULT_FRAME_AUDIT)
    parser.add_argument("--expected-start", type=int, default=1)
    parser.add_argument("--expected-end", type=int, default=1034)
    parser.add_argument(
        "--gap-stride",
        type=int,
        default=5,
        help="Sampling stride inside uncovered-range review sheets.",
    )
    parser.add_argument(
        "--max-gap-window",
        type=int,
        default=100,
        help="Maximum token span per uncovered-range contact sheet.",
    )
    parser.add_argument(
        "--boundary-default-radius",
        type=int,
        default=10,
        help="Radius used around boundaries if the CSV gives zero uncertainty.",
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()

    def resolve(path: Path) -> Path:
        return path.resolve() if path.is_absolute() else (repo_root / path).resolve()

    output_root = resolve(args.output_root)
    manual_csv = resolve(args.manual_csv)
    frame_audit_path = resolve(args.frame_audit)

    metadata_dir = ensure_dir(output_root / "metadata" / "s7_retrieval_upgrade")
    reports_dir = ensure_dir(output_root / "reports" / "s7_retrieval_upgrade")
    figures_dir = ensure_dir(
        output_root
        / "figures"
        / "s7_retrieval_upgrade"
        / "scene_annotation_refinement"
    )

    if not manual_csv.exists():
        raise FileNotFoundError(f"Missing manual annotation CSV: {manual_csv}")
    if not frame_audit_path.exists():
        raise FileNotFoundError(f"Missing S7.0B frame-path audit: {frame_audit_path}")

    manual = read_manual_csv_tolerant(manual_csv)
    audit = pd.read_csv(frame_audit_path)
    required_audit = {"token0_id", "resolved_image_path"}
    missing_audit = required_audit - set(audit.columns)
    if missing_audit:
        raise RuntimeError(
            f"Frame-path audit missing required columns: {sorted(missing_audit)}"
        )

    audit["token0_id"] = pd.to_numeric(audit["token0_id"], errors="raise").astype(int)
    audit = audit.sort_values("token0_id").reset_index(drop=True)
    expected_tokens = set(range(args.expected_start, args.expected_end + 1))
    actual_tokens = set(audit["token0_id"].tolist())
    if actual_tokens != expected_tokens:
        missing = contiguous_ranges(sorted(expected_tokens - actual_tokens))
        extra = contiguous_ranges(sorted(actual_tokens - expected_tokens))
        raise RuntimeError(
            f"Frame audit token coverage mismatch. Missing={missing}, extra={extra}"
        )

    parsed, issues, uncovered, overlaps = validate_manual(
        manual=manual,
        expected_start=args.expected_start,
        expected_end=args.expected_end,
    )

    issues_path = metadata_dir / "s7_0c_scene_annotation_issues.csv"
    pd.DataFrame(
        issues,
        columns=["issue_type", "row_number", "segment_id", "details"],
    ).to_csv(issues_path, index=False)

    uncovered_path = metadata_dir / "s7_0c_uncovered_ranges.csv"
    pd.DataFrame(
        [{"start_token0_id": a, "end_token0_id": b} for a, b in uncovered]
    ).to_csv(uncovered_path, index=False)

    overlaps_path = metadata_dir / "s7_0c_overlap_ranges.csv"
    pd.DataFrame(
        [{"start_token0_id": a, "end_token0_id": b} for a, b in overlaps]
    ).to_csv(overlaps_path, index=False)

    # Remove only S7.0C-generated review images.
    for stale in figures_dir.glob("s7_0c_*.png"):
        stale.unlink()

    sheet_manifest: list[dict[str, object]] = []

    # Generate dense sheets around each boundary between chronological rows.
    if not parsed.empty:
        for i in range(len(parsed) - 1):
            left = parsed.iloc[i]
            right = parsed.iloc[i + 1]
            boundary_left = int(left["end_token0_id"])
            boundary_right = int(right["start_token0_id"])
            radius = max(
                int(left["boundary_uncertainty_frames"]),
                int(right["boundary_uncertainty_frames"]),
                args.boundary_default_radius,
            )
            start = max(args.expected_start, boundary_left - radius)
            end = min(args.expected_end, boundary_right + radius)
            selected = select_sampled_range(audit, start, end, stride=1)
            name = (
                f"s7_0c_boundary_{i+1:02d}_tokens_{start:04d}_{end:04d}.png"
            )
            sheet_manifest.extend(
                draw_sheet(
                    selected,
                    figures_dir / name,
                    title=(
                        f"Boundary review: {left['segment_id']} -> "
                        f"{right['segment_id']} | tokens {start}-{end}"
                    ),
                )
            )

    # Generate targeted sheets for uncovered spans. Large spans are chunked.
    gap_sheet_number = 0
    for gap_start, gap_end in uncovered:
        window_start = gap_start
        while window_start <= gap_end:
            window_end = min(gap_end, window_start + args.max_gap_window - 1)
            selected = select_sampled_range(
                audit, window_start, window_end, stride=args.gap_stride
            )
            gap_sheet_number += 1
            name = (
                f"s7_0c_gap_{gap_sheet_number:02d}_tokens_"
                f"{window_start:04d}_{window_end:04d}_stride{args.gap_stride}.png"
            )
            sheet_manifest.extend(
                draw_sheet(
                    selected,
                    figures_dir / name,
                    title=f"Uncovered range review: tokens {window_start}-{window_end}",
                )
            )
            window_start = window_end + 1

    sheet_manifest_path = metadata_dir / "s7_0c_review_sheet_manifest.csv"
    pd.DataFrame(sheet_manifest).to_csv(sheet_manifest_path, index=False)

    invalid_types = {
        "invalid_integer",
        "reversed_range",
        "out_of_bounds",
        "invalid_primary_scene",
        "invalid_secondary_scene",
        "invalid_confidence",
        "invalid_boundary_uncertainty",
        "coverage_gap",
        "coverage_overlap",
        "duplicate_segment_id",
        "non_chronological_order",
    }
    blocking = [issue for issue in issues if issue["issue_type"] in invalid_types]

    report_path = reports_dir / "s7_0c_scene_annotation_refinement_report.md"

    if blocking:
        issue_lines = "\n".join(
            f"- **{item['issue_type']}**: {item['details']}" for item in blocking
        )
        report = f"""# S7.0C Scene Annotation Refinement Report

Generated: `{utc_now()}`

## Status

```text
ACTION_REQUIRED
```

## Validation findings

{issue_lines}

## Generated review material

```text
Issues CSV:        {issues_path.relative_to(repo_root)}
Uncovered ranges:  {uncovered_path.relative_to(repo_root)}
Overlap ranges:    {overlaps_path.relative_to(repo_root)}
Review sheets:     {figures_dir.relative_to(repo_root)}
Sheet manifest:    {sheet_manifest_path.relative_to(repo_root)}
```

Correct the manual CSV, keep rows chronological, and rerun this same script.
The block freezes automatically only when tokens {args.expected_start}–{args.expected_end}
are covered exactly once and all taxonomy values are valid.
"""
        report_path.write_text(report, encoding="utf-8")

        print("S7.0C Scene Annotation Refinement")
        print("---------------------------------")
        print("Status:             ACTION_REQUIRED")
        print(f"Manual rows:        {len(parsed)}")
        print(f"Validation issues:  {len(blocking)}")
        print(f"Uncovered ranges:   {len(uncovered)}")
        print(f"Overlap ranges:     {len(overlaps)}")
        print(f"Review sheets:      {len(list(figures_dir.glob('s7_0c_*.png')))}")
        print(f"Issues CSV:         {issues_path.relative_to(repo_root)}")
        print(f"Review-sheet dir:   {figures_dir.relative_to(repo_root)}")
        print(f"Report:             {report_path.relative_to(repo_root)}")
        print()
        print("Correct the manual CSV using the generated targeted sheets, then rerun.")
        return 2

    # Fully valid: expand ranges to one row per frame.
    parsed = parsed.sort_values("start_token0_id").reset_index(drop=True)
    frame_rows: list[dict[str, object]] = []
    for _, seg in parsed.iterrows():
        for token in range(
            int(seg["start_token0_id"]), int(seg["end_token0_id"]) + 1
        ):
            frame_rows.append(
                {
                    "token0_id": token,
                    "segment_id": seg["segment_id"],
                    "primary_scene": seg["primary_scene"],
                    "secondary_scene": seg["secondary_scene"],
                    "confidence": seg["confidence"],
                    "boundary_uncertainty_frames": int(
                        seg["boundary_uncertainty_frames"]
                    ),
                    "label_source": "manual_visual_v1",
                    "taxonomy_version": "s7.0-v1",
                }
            )

    frame_labels = pd.DataFrame(frame_rows).sort_values("token0_id")
    frame_labels_path = metadata_dir / "s7_0_traj01_frame_scene_labels.csv"
    frame_labels.to_csv(frame_labels_path, index=False)

    frozen_segments_path = metadata_dir / "s7_0_traj01_scene_segments_frozen.csv"
    parsed[
        [c for c in MANUAL_COLUMNS if c in parsed.columns]
    ].to_csv(frozen_segments_path, index=False)

    primary_counts = (
        frame_labels["primary_scene"].value_counts().sort_index().to_dict()
    )
    summary = {
        "generated_utc": utc_now(),
        "status": "PASS_FROZEN",
        "sequence": "traj01",
        "ordering_key": "token0_id",
        "expected_token_range": [args.expected_start, args.expected_end],
        "frame_count": int(len(frame_labels)),
        "segment_count": int(len(parsed)),
        "taxonomy_version": "s7.0-v1",
        "label_source": "manual_visual_v1",
        "primary_scene_frame_counts": {
            str(k): int(v) for k, v in primary_counts.items()
        },
        "ground_truth_isolation": (
            "UAV imagery and canonical sequence order only; no filename coordinates, "
            "reference trajectory, retrieval scores, oracle labels, or errors used."
        ),
    }
    summary_path = metadata_dir / "s7_0_traj01_scene_label_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    report = f"""# S7.0C Scene Annotation Refinement Report

Generated: `{utc_now()}`

## Status

```text
PASS_FROZEN
```

The manual annotation covers token0_id {args.expected_start}–{args.expected_end}
exactly once, uses only frozen taxonomy values, and has been expanded to one row per frame.

## Frozen outputs

```text
Segments:      {frozen_segments_path.relative_to(repo_root)}
Frame labels:  {frame_labels_path.relative_to(repo_root)}
Summary:       {summary_path.relative_to(repo_root)}
```

These labels are diagnostic grouping metadata only. They must not be used as hidden
online information for candidate selection, ranking, verification, correction acceptance,
or policy tuning per query.
"""
    report_path.write_text(report, encoding="utf-8")

    print("S7.0C Scene Annotation Refinement")
    print("---------------------------------")
    print("Status:             PASS_FROZEN")
    print(f"Segments:           {len(parsed)}")
    print(f"Frame labels:       {len(frame_labels)}")
    print(f"Frozen segments:    {frozen_segments_path.relative_to(repo_root)}")
    print(f"Frame-level labels: {frame_labels_path.relative_to(repo_root)}")
    print(f"Summary:            {summary_path.relative_to(repo_root)}")
    print(f"Report:             {report_path.relative_to(repo_root)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print("S7.0C Scene Annotation Refinement", file=sys.stderr)
        print("---------------------------------", file=sys.stderr)
        print("Status: BLOCKED", file=sys.stderr)
        print(f"Reason: {exc}", file=sys.stderr)
        raise
