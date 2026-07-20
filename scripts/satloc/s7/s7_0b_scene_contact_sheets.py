#!/usr/bin/env python3
"""S7.0B — traj01 scene taxonomy and coarse contact-sheet generation.

This block prepares manual scene-range annotation without running retrieval methods.
It does NOT parse longitude/latitude from UAV filenames and does not use reference
coordinates, method scores, errors, oracle labels, or retrieval outcomes.

Run from repository root:

    source .drone_venv/bin/activate
    export PYTHONPATH=$PWD/src
    python scripts/satloc/s7/s7_0b_scene_contact_sheets.py \
      2>&1 | tee outputs/satloc/reports/s7_retrieval_upgrade/s7_0b_scene_contact_sheets.log

Outputs:
    configs/satloc/s7_scene_taxonomy.yaml
    outputs/satloc/metadata/s7_retrieval_upgrade/s7_0_scene_segments_manual.csv
    outputs/satloc/metadata/s7_retrieval_upgrade/s7_0_contact_sheet_manifest.csv
    outputs/satloc/metadata/s7_retrieval_upgrade/s7_0_frame_path_audit.csv
    outputs/satloc/figures/s7_retrieval_upgrade/scene_contact_sheets/*.png
    outputs/satloc/reports/s7_retrieval_upgrade/s7_0b_scene_annotation_guide.md
    outputs/satloc/reports/s7_retrieval_upgrade/s7_0b_scene_contact_sheet_report.md
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import pandas as pd


DEFAULT_SEQUENCE_MANIFEST = Path(
    "outputs/satloc/metadata/s6a_relative_motion/s6a_sequence_manifest.csv"
)
DEFAULT_UAV_INDEX = Path("outputs/satloc/metadata/uav_frames_index_enriched.csv")
DEFAULT_DATASET_ROOT = Path("data/raw/satloc/part_1")
DEFAULT_OUTPUT_ROOT = Path("outputs/satloc")

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
PATH_COLUMNS = [
    "image_path",
    "uav_image_path",
    "frame_path",
    "file_path",
    "filepath",
    "path",
    "image_file",
    "image_filename",
    "filename",
    "file_name",
]

PRIMARY_CLASSES = [
    "urban",
    "mixed_urban_natural",
    "forest_canopy",
    "agricultural_open_field",
    "water_wetland",
    "low_structure",
    "other",
]

SECONDARY_CLASSES = [
    "none",
    "urban",
    "vegetation",
    "forest_canopy",
    "agricultural_open_field",
    "water_wetland",
    "road_corridor",
    "construction_bare_ground",
    "other",
]

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


@dataclass
class SheetCell:
    sheet_name: str
    cell_index: int
    source_row_index: int
    sequence_frame_id: str
    token0_id: str
    resolved_image_path: str
    read_ok: bool


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def find_first_column(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    return None


def scalar_to_string(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value)


def candidate_paths(raw_value: str, repo_root: Path, dataset_root: Path, sequence: str) -> list[Path]:
    value = raw_value.strip()
    if not value:
        return []

    raw_path = Path(value).expanduser()
    basename = raw_path.name
    candidates: list[Path] = []

    if raw_path.is_absolute():
        candidates.append(raw_path)
    else:
        candidates.extend(
            [
                repo_root / raw_path,
                dataset_root / raw_path,
                dataset_root / "UAV Data" / sequence / raw_path,
                dataset_root / "UAV Data" / sequence / basename,
                dataset_root / sequence / basename,
            ]
        )

    # Preserve order while removing duplicates.
    unique: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        key = os.path.normcase(str(path))
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def build_basename_index(dataset_root: Path, sequence: str) -> dict[str, list[Path]]:
    roots = [dataset_root / "UAV Data" / sequence, dataset_root / sequence]
    existing_roots = [root for root in roots if root.exists()]
    if not existing_roots:
        existing_roots = [dataset_root]

    index: dict[str, list[Path]] = {}
    for root in existing_roots:
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                index.setdefault(path.name, []).append(path)
    return index


def resolve_image_paths(
    sequence_df: pd.DataFrame,
    uav_df: pd.DataFrame,
    repo_root: Path,
    dataset_root: Path,
    sequence: str,
) -> pd.DataFrame:
    seq = sequence_df.copy()
    uav = uav_df.copy()

    if "sequence" in seq.columns:
        seq = seq[seq["sequence"].astype(str) == sequence].copy()
    if "sequence" in uav.columns:
        uav = uav[uav["sequence"].astype(str) == sequence].copy()

    required = {"token0_id", "sequence_frame_id"}
    missing = required - set(seq.columns)
    if missing:
        raise RuntimeError(f"Sequence manifest missing required columns: {sorted(missing)}")

    seq = seq.sort_values("sequence_frame_id", kind="mergesort").reset_index(drop=True)
    if seq["token0_id"].duplicated().any():
        raise RuntimeError("Duplicate token0_id values in traj01 sequence manifest.")

    # Add path-like columns from UAV index without overwriting sequence columns.
    if "token0_id" in uav.columns:
        keep = ["token0_id"] + [c for c in PATH_COLUMNS if c in uav.columns]
        keep = list(dict.fromkeys(keep))
        if len(keep) > 1:
            uav_small = uav[keep].drop_duplicates(subset=["token0_id"], keep="first")
            seq = seq.merge(uav_small, on="token0_id", how="left", suffixes=("", "_uav"))

    available_path_columns = [c for c in seq.columns if c in PATH_COLUMNS or c.endswith("_uav")]
    basename_index: dict[str, list[Path]] | None = None

    resolved_paths: list[str] = []
    resolution_sources: list[str] = []
    resolution_statuses: list[str] = []

    for _, row in seq.iterrows():
        resolved: Path | None = None
        source = ""
        fallback_basenames: list[str] = []

        for column in available_path_columns:
            raw = scalar_to_string(row.get(column, ""))
            if not raw:
                continue
            fallback_basenames.append(Path(raw).name)
            for candidate in candidate_paths(raw, repo_root, dataset_root, sequence):
                if candidate.exists() and candidate.is_file():
                    resolved = candidate.resolve()
                    source = column
                    break
            if resolved is not None:
                break

        if resolved is None and fallback_basenames:
            if basename_index is None:
                basename_index = build_basename_index(dataset_root, sequence)
            for basename in fallback_basenames:
                matches = basename_index.get(basename, [])
                if len(matches) == 1:
                    resolved = matches[0].resolve()
                    source = "basename_index"
                    break
                if len(matches) > 1:
                    # Prefer a path explicitly containing the sequence name.
                    sequence_matches = [p for p in matches if sequence in p.parts]
                    if len(sequence_matches) == 1:
                        resolved = sequence_matches[0].resolve()
                        source = "basename_index_sequence"
                        break

        resolved_paths.append(str(resolved) if resolved else "")
        resolution_sources.append(source)
        resolution_statuses.append("resolved" if resolved else "unresolved")

    seq["resolved_image_path"] = resolved_paths
    seq["path_resolution_source"] = resolution_sources
    seq["path_resolution_status"] = resolution_statuses
    return seq


def letterbox(image: np.ndarray, width: int, height: int) -> np.ndarray:
    canvas = np.full((height, width, 3), 235, dtype=np.uint8)
    if image is None or image.size == 0:
        cv2.putText(
            canvas,
            "IMAGE READ FAILED",
            (20, height // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
        return canvas

    src_h, src_w = image.shape[:2]
    scale = min(width / src_w, height / src_h)
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    x0 = (width - new_w) // 2
    y0 = (height - new_h) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return canvas


def wrapped_text(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        tentative = word if not current else f"{current} {word}"
        if len(tentative) <= max_chars:
            current = tentative
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines[:2]


def draw_contact_sheet(
    selected: pd.DataFrame,
    output_path: Path,
    cell_width: int,
    image_height: int,
    title_height: int,
    columns: int,
) -> list[SheetCell]:
    rows = max(1, math.ceil(len(selected) / columns))
    sheet = np.full((rows * (image_height + title_height), columns * cell_width, 3), 255, dtype=np.uint8)
    cells: list[SheetCell] = []

    for cell_index, (source_row_index, row) in enumerate(selected.iterrows()):
        grid_row = cell_index // columns
        grid_col = cell_index % columns
        x0 = grid_col * cell_width
        y0 = grid_row * (image_height + title_height)

        path = Path(str(row["resolved_image_path"]))
        image = cv2.imread(str(path), cv2.IMREAD_COLOR) if path.exists() else None
        read_ok = image is not None and image.size > 0
        panel = letterbox(image, cell_width, image_height)
        sheet[y0 : y0 + image_height, x0 : x0 + cell_width] = panel

        token = scalar_to_string(row["token0_id"])
        seq_id = scalar_to_string(row["sequence_frame_id"])
        title = f"sequence_frame_id={seq_id}   token0_id={token}"
        for line_no, line in enumerate(wrapped_text(title, 46)):
            cv2.putText(
                sheet,
                line,
                (x0 + 7, y0 + image_height + 22 + 20 * line_no),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.46,
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

        cells.append(
            SheetCell(
                sheet_name=output_path.name,
                cell_index=cell_index,
                source_row_index=int(source_row_index),
                sequence_frame_id=seq_id,
                token0_id=token,
                resolved_image_path=str(path),
                read_ok=read_ok,
            )
        )

    ensure_dir(output_path.parent)
    if not cv2.imwrite(str(output_path), sheet):
        raise RuntimeError(f"Failed to write contact sheet: {output_path}")
    return cells


def write_taxonomy_yaml(path: Path) -> None:
    if path.exists():
        return
    content = """# S7 scene taxonomy — frozen before retrieval method results
version: s7.0-v1
sequence: traj01
ordering_key: token0_id

annotation_purpose: >-
  Stratified evaluation and failure interpretation only. These labels must not be
  used to select candidates, tune scores per query, accept corrections, or access
  reference/GNSS coordinates during realistic localization.

primary_scene_classes:
  urban: Buildings, dense roads, paved blocks, roofs, or constructed layout dominate.
  mixed_urban_natural: Neither built structure nor natural terrain clearly dominates.
  forest_canopy: Dense trees or canopy mass dominates most of the visible area.
  agricultural_open_field: Fields, crop parcels, open agricultural land, or field boundaries dominate.
  water_wetland: Pond, river, water body, marsh, or shoreline is a dominant stable element.
  low_structure: Large weakly textured or weakly bounded area with few stable anchors.
  other: Visually meaningful scene not covered by the frozen classes.

secondary_scene_classes:
  - none
  - urban
  - vegetation
  - forest_canopy
  - agricultural_open_field
  - water_wetland
  - road_corridor
  - construction_bare_ground
  - other

confidence_values:
  - high
  - medium
  - low

manual_range_columns:
  - segment_id
  - start_token0_id
  - end_token0_id
  - primary_scene
  - secondary_scene
  - confidence
  - boundary_uncertainty_frames
  - notes

reserved_challenge_flags_for_later_paired_or_numeric_diagnostics:
  - rotation_challenge
  - fov_scale_challenge
  - shadow_heavy
  - repetitive_structure
  - seasonal_or_map_age_mismatch
  - transition_zone

manual_annotation_rules:
  - Use continuous inclusive token0_id ranges.
  - Cover token0_id 1 through 1034 without overlap in the final frozen file.
  - Use primary_scene for the dominant visual domain.
  - Use secondary_scene only when it adds useful context; otherwise use none.
  - Put uncertain boundary width in boundary_uncertainty_frames, typically 5 or 10.
  - Do not inspect retrieval scores, errors, oracle ranks, or method outputs while labeling.
  - Do not infer coordinates from image filenames.
"""
    ensure_dir(path.parent)
    path.write_text(content, encoding="utf-8")


def write_manual_template(path: Path, force: bool) -> None:
    if path.exists() and not force:
        return
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANUAL_COLUMNS)
        writer.writeheader()


def write_example(path: Path) -> None:
    if path.exists():
        return
    rows = [
        {
            "segment_id": "seg_001",
            "start_token0_id": "1",
            "end_token0_id": "84",
            "primary_scene": "urban",
            "secondary_scene": "vegetation",
            "confidence": "high",
            "boundary_uncertainty_frames": "10",
            "notes": "EXAMPLE ONLY — replace using contact-sheet observations.",
        },
        {
            "segment_id": "seg_002",
            "start_token0_id": "85",
            "end_token0_id": "126",
            "primary_scene": "mixed_urban_natural",
            "secondary_scene": "urban",
            "confidence": "medium",
            "boundary_uncertainty_frames": "10",
            "notes": "EXAMPLE ONLY — transition range.",
        },
    ]
    ensure_dir(path.parent)
    pd.DataFrame(rows, columns=MANUAL_COLUMNS).to_csv(path, index=False)


def write_annotation_guide(path: Path, manual_csv: Path, contact_dir: Path) -> None:
    text = f"""# S7.0B Manual Scene-Range Annotation Guide

Generated: `{utc_now()}`

## What to open

Open the PNG files in:

```text
{contact_dir.as_posix()}/
```

They show approximately every tenth `traj01` frame. Titles show only the canonical
`sequence_frame_id` and `token0_id`; filename coordinates are deliberately not displayed.

## What to edit

Edit:

```text
{manual_csv.as_posix()}
```

Add one row for each continuous scene segment. Ranges are inclusive.

## Allowed primary labels

```text
{os.linesep.join(PRIMARY_CLASSES)}
```

## Manual task for this coarse pass

1. Review the contact sheets in numerical order.
2. Mark approximate continuous scene ranges covering token0_id 1–1034.
3. Use `boundary_uncertainty_frames=10` where the change point is unclear.
4. Use `secondary_scene=none` unless a second domain is visibly important.
5. Do not label rotation, FOV/scale mismatch, or seasonal mismatch yet. Those need
   paired UAV–satellite or numeric diagnostics and will be handled later.
6. Do not inspect DINO, PHLO, LightGlue, retrieval errors, or oracle results while labeling.

You do **not** need to inspect all 1,034 frames individually. The next block will generate
dense boundary sheets only around the uncertain transitions you record.
"""
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--sequence", default="traj01")
    parser.add_argument("--sequence-manifest", type=Path, default=DEFAULT_SEQUENCE_MANIFEST)
    parser.add_argument("--uav-index", type=Path, default=DEFAULT_UAV_INDEX)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--expected-frames", type=int, default=1034)
    parser.add_argument("--sample-stride", type=int, default=10)
    parser.add_argument("--frames-per-range", type=int, default=200)
    parser.add_argument("--columns", type=int, default=5)
    parser.add_argument("--cell-width", type=int, default=320)
    parser.add_argument("--image-height", type=int, default=220)
    parser.add_argument("--title-height", type=int, default=48)
    parser.add_argument(
        "--force-empty-manual-template",
        action="store_true",
        help="Overwrite the manual CSV with an empty template. Normally leave this off.",
    )
    args = parser.parse_args()

    if args.sample_stride < 1 or args.frames_per_range < 1 or args.columns < 1:
        raise ValueError("sample-stride, frames-per-range and columns must be positive.")

    repo_root = args.repo_root.resolve()

    def resolve(path: Path) -> Path:
        return path.resolve() if path.is_absolute() else (repo_root / path).resolve()

    sequence_manifest = resolve(args.sequence_manifest)
    uav_index = resolve(args.uav_index)
    dataset_root = resolve(args.dataset_root)
    output_root = resolve(args.output_root)

    metadata_dir = ensure_dir(output_root / "metadata" / "s7_retrieval_upgrade")
    reports_dir = ensure_dir(output_root / "reports" / "s7_retrieval_upgrade")
    contact_dir = ensure_dir(
        output_root / "figures" / "s7_retrieval_upgrade" / "scene_contact_sheets"
    )
    taxonomy_path = repo_root / "configs" / "satloc" / "s7_scene_taxonomy.yaml"

    for path, name in [(sequence_manifest, "sequence manifest"), (uav_index, "UAV index"), (dataset_root, "dataset root")]:
        if not path.exists():
            raise FileNotFoundError(f"Missing {name}: {path}")

    sequence_df = pd.read_csv(sequence_manifest)
    uav_df = pd.read_csv(uav_index)
    resolved_df = resolve_image_paths(
        sequence_df=sequence_df,
        uav_df=uav_df,
        repo_root=repo_root,
        dataset_root=dataset_root,
        sequence=args.sequence,
    )

    if len(resolved_df) != args.expected_frames:
        raise RuntimeError(
            f"Expected {args.expected_frames} {args.sequence} frames, found {len(resolved_df)}."
        )

    unresolved_count = int((resolved_df["path_resolution_status"] != "resolved").sum())
    path_audit = metadata_dir / "s7_0_frame_path_audit.csv"
    resolved_df.to_csv(path_audit, index=False)
    if unresolved_count:
        examples = resolved_df.loc[
            resolved_df["path_resolution_status"] != "resolved",
            ["sequence_frame_id", "token0_id"],
        ].head(10)
        raise RuntimeError(
            f"Unable to resolve {unresolved_count} image paths. See {path_audit}. "
            f"First unresolved rows: {examples.to_dict(orient='records')}"
        )

    # Remove stale generated sheets only; manual annotations are never touched here.
    for stale in contact_dir.glob("s7_0b_traj01_*.png"):
        stale.unlink()

    all_cells: list[SheetCell] = []
    total = len(resolved_df)
    for range_start in range(0, total, args.frames_per_range):
        range_end = min(total, range_start + args.frames_per_range)
        indices = list(range(range_start, range_end, args.sample_stride))
        if (range_end - 1) not in indices:
            indices.append(range_end - 1)
        indices = sorted(set(indices))
        selected = resolved_df.iloc[indices]

        start_token = scalar_to_string(resolved_df.iloc[range_start]["token0_id"])
        end_token = scalar_to_string(resolved_df.iloc[range_end - 1]["token0_id"])
        sheet_name = f"s7_0b_traj01_tokens_{int(float(start_token)):04d}_{int(float(end_token)):04d}_stride{args.sample_stride}.png"
        sheet_path = contact_dir / sheet_name
        all_cells.extend(
            draw_contact_sheet(
                selected=selected,
                output_path=sheet_path,
                cell_width=args.cell_width,
                image_height=args.image_height,
                title_height=args.title_height,
                columns=args.columns,
            )
        )

    contact_manifest = metadata_dir / "s7_0_contact_sheet_manifest.csv"
    pd.DataFrame([cell.__dict__ for cell in all_cells]).to_csv(contact_manifest, index=False)

    read_failures = sum(not cell.read_ok for cell in all_cells)
    if read_failures:
        raise RuntimeError(
            f"Contact sheets contain {read_failures} unreadable sampled images. "
            f"See {contact_manifest}."
        )

    write_taxonomy_yaml(taxonomy_path)
    manual_csv = metadata_dir / "s7_0_scene_segments_manual.csv"
    write_manual_template(manual_csv, force=args.force_empty_manual_template)
    example_csv = metadata_dir / "s7_0_scene_segments_example_DO_NOT_USE.csv"
    write_example(example_csv)

    guide_path = reports_dir / "s7_0b_scene_annotation_guide.md"
    write_annotation_guide(
        path=guide_path,
        manual_csv=manual_csv.relative_to(repo_root),
        contact_dir=contact_dir.relative_to(repo_root),
    )

    report_path = reports_dir / "s7_0b_scene_contact_sheet_report.md"
    report = f"""# S7.0B Scene Contact-Sheet Report

Generated: `{utc_now()}`

## Status

```text
PASS
```

## Frozen input checks

```text
Sequence:                    {args.sequence}
Frames:                      {len(resolved_df)}
Resolved image paths:        {len(resolved_df) - unresolved_count}
Unresolved image paths:      {unresolved_count}
Contact sheets:              {len(list(contact_dir.glob('s7_0b_traj01_*.png')))}
Sampled contact-sheet cells: {len(all_cells)}
Image read failures:         {read_failures}
Sampling stride:             {args.sample_stride}
Frames per coarse range:     {args.frames_per_range}
```

## Ground-truth isolation

This block used only canonical sequence ordering and UAV image bytes. It did not parse
longitude/latitude from filenames and did not access retrieval scores, error labels,
oracle ranks, LightGlue results, or reference coordinates.

## Next manual action

Review the coarse contact sheets and fill continuous approximate scene ranges in:

```text
{manual_csv.relative_to(repo_root).as_posix()}
```

Do not inspect all 1,034 frames. Mark uncertain transitions with
`boundary_uncertainty_frames=10`; S7.0C will generate dense review sheets around them.
"""
    report_path.write_text(report, encoding="utf-8")

    print("S7.0B Scene Taxonomy and Contact Sheets")
    print("----------------------------------------")
    print("Status:             PASS")
    print(f"Frames resolved:    {len(resolved_df)} / {args.expected_frames}")
    print(f"Contact sheets:     {len(list(contact_dir.glob('s7_0b_traj01_*.png')))}")
    print(f"Sampled frames:     {len(all_cells)}")
    print(f"Taxonomy:           {taxonomy_path.relative_to(repo_root)}")
    print(f"Manual range CSV:   {manual_csv.relative_to(repo_root)}")
    print(f"Contact-sheet dir:  {contact_dir.relative_to(repo_root)}")
    print(f"Guide:              {guide_path.relative_to(repo_root)}")
    print()
    print("Manual work begins now: review only the coarse contact sheets and")
    print("enter approximate continuous ranges. Do not inspect all 1034 frames.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print("S7.0B Scene Taxonomy and Contact Sheets", file=sys.stderr)
        print("----------------------------------------", file=sys.stderr)
        print("Status: BLOCKED", file=sys.stderr)
        print(f"Reason: {exc}", file=sys.stderr)
        raise
