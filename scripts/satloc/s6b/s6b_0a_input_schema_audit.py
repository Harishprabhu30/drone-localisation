#!/usr/bin/env python3
"""S6B.0A input-schema and join-key audit.

This script does not perform correction replay.
It inspects the frozen S5C/S6A inputs before S6B.0 manifest construction.

Command USed:

export PYTHONPATH=$PWD/src

python scripts/satloc/s6b/s6b_0a_input_schema_audit.py \
  | tee outputs/satloc/reports/s6b_relative_absolute/s6b0a_input_schema_audit.log

"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd


INPUTS = {
    "s5c3_features": Path(
        "outputs/satloc/metadata/s5c_temporal/"
        "s5c3_lightglue_confidence_features_full263.csv"
    ),
    "s5c3_gates": Path(
        "outputs/satloc/metadata/s5c_temporal/"
        "s5c3_recommended_confidence_gates_full263.csv"
    ),
    "s5c2_query_summary": Path(
        "outputs/satloc/metadata/s5c_temporal/"
        "s5c2_lightglue_union_query_summary_top50_full263.csv"
    ),
    "s5c2_candidate_scores": Path(
        "outputs/satloc/metadata/s5c_temporal/"
        "s5c2_lightglue_union_candidate_scores_top50_full263.csv"
    ),
    "s6a_sequence_manifest": Path(
        "outputs/satloc/metadata/s6a_relative_motion/"
        "s6a_sequence_manifest.csv"
    ),
    "s6a_relative_trajectory": Path(
        "outputs/satloc/metadata/s6a_relative_motion/"
        "s6a2_orb_relative_trajectory_aligned_eval_only.csv"
    ),
    "satellite_index": Path(
        "outputs/satloc/metadata/satellite_tiles_index_enriched.csv"
    ),
}

OUTPUT_JSON = Path(
    "outputs/satloc/reports/s6b_relative_absolute/"
    "s6b0a_input_schema_audit.json"
)

KEYWORDS = (
    "token",
    "frame",
    "image",
    "sequence",
    "order",
    "tile",
    "chosen",
    "rank",
    "score",
    "inlier",
    "match",
    "coverage",
    "margin",
    "longitude",
    "latitude",
    "lon",
    "lat",
    "utm",
    "enu",
    "reference",
    "error",
    "hit",
    "accept",
    "distance",
    "path",
)


def json_safe(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def interesting_columns(columns: list[str]) -> list[str]:
    selected: list[str] = []
    for column in columns:
        lower = column.lower()
        if lower in {"x", "y", "id"} or any(k in lower for k in KEYWORDS):
            selected.append(column)
    return selected


def inspect_dataframe(name: str, path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    df = pd.read_csv(path, low_memory=False)
    columns = list(df.columns)
    interesting = interesting_columns(columns)

    column_stats: dict[str, Any] = {}
    for column in interesting:
        series = df[column]
        column_stats[column] = {
            "dtype": str(series.dtype),
            "null_count": int(series.isna().sum()),
            "non_null_count": int(series.notna().sum()),
            "unique_count": int(series.nunique(dropna=True)),
            "duplicate_non_null_count": int(
                series.dropna().duplicated(keep=False).sum()
            ),
            "sample_values": [
                json_safe(v) for v in series.dropna().head(5).tolist()
            ],
        }

    sample_columns = interesting if interesting else columns[:20]
    sample_columns = sample_columns[:30]

    record = {
        "path": str(path),
        "rows": int(len(df)),
        "columns_count": int(len(columns)),
        "columns": columns,
        "interesting_columns": interesting,
        "column_stats": column_stats,
        "sample_rows": [
            {k: json_safe(v) for k, v in row.items()}
            for row in df[sample_columns].head(2).to_dict(orient="records")
        ],
    }

    print("\n" + "=" * 88)
    print(name)
    print(path)
    print(f"shape: {df.shape}")
    print("columns:")
    for idx, column in enumerate(columns):
        print(f"  [{idx:02d}] {column:<48} {df[column].dtype}")

    print("\nPotential S6B columns:")
    for column in interesting:
        stat = column_stats[column]
        print(
            f"  {column:<48} "
            f"unique={stat['unique_count']:<7} "
            f"null={stat['null_count']:<7} "
            f"duplicates={stat['duplicate_non_null_count']}"
        )

    return df, record


def shared_columns(
    left_name: str,
    right_name: str,
    frames: dict[str, pd.DataFrame],
) -> list[str]:
    left = set(frames[left_name].columns)
    right = set(frames[right_name].columns)
    return sorted(left & right)


def main() -> int:
    missing = [str(path) for path in INPUTS.values() if not path.exists()]
    if missing:
        print("Missing required inputs:", file=sys.stderr)
        for path in missing:
            print(f"  - {path}", file=sys.stderr)
        return 2

    frames: dict[str, pd.DataFrame] = {}
    report: dict[str, Any] = {
        "stage": "S6B.0A",
        "purpose": "Inspect frozen S5C/S6A schemas before manifest construction",
        "inputs": {},
        "shared_columns": {},
    }

    for name, path in INPUTS.items():
        df, record = inspect_dataframe(name, path)
        frames[name] = df
        report["inputs"][name] = record

    join_pairs = [
        ("s5c3_features", "s5c2_query_summary"),
        ("s5c3_features", "s6a_sequence_manifest"),
        ("s5c2_query_summary", "s6a_sequence_manifest"),
        ("s6a_sequence_manifest", "s6a_relative_trajectory"),
        ("s5c2_query_summary", "satellite_index"),
        ("s5c2_candidate_scores", "satellite_index"),
    ]

    print("\n" + "=" * 88)
    print("SHARED-COLUMN AUDIT")

    for left, right in join_pairs:
        shared = shared_columns(left, right, frames)
        key = f"{left}__{right}"
        report["shared_columns"][key] = shared

        print(f"\n{left}  <->  {right}")
        if shared:
            for column in shared:
                print(f"  - {column}")
        else:
            print("  No identically named columns.")

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n" + "=" * 88)
    print(f"Saved audit: {OUTPUT_JSON}")
    print("\nS6B.0 manifest construction has not yet been performed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
