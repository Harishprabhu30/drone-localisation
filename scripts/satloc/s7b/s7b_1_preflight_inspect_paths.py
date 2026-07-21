#!/usr/bin/env python3
"""
S7B.1-preflight — SatLoc path and join-key inspector.

Purpose
-------
Read-only inspection before DINOv2:
1. Inspect S7 query manifest columns.
2. Inspect global/traj01 UAV index columns.
3. Inspect satellite tile index path columns.
4. Inspect S5C candidate-score image path hints.
5. Compute join-overlap matrix between query manifest and UAV index.
6. Test which path columns actually resolve to existing image files.
7. Write a JSON report and small CSV overlap reports.

This script does not load DINOv2, does not extract descriptors, and does not
modify any retrieval data.

Run
---
cd /Users/harishprabhu/Documents/drone-localisation
source .drone_venv/bin/activate
export PYTHONPATH=$PWD/src

python scripts/satloc/s7b/s7b_1_preflight_inspect_paths.py \
  --repo-root "$PWD" \
  2>&1 | tee outputs/satloc/reports/s7b_dinov2_global/s7b1_preflight_inspect_paths.log
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_QUERY_MANIFEST = Path(
    "outputs/satloc/metadata/s7_retrieval_upgrade/s7_0_query_manifest.csv"
)
DEFAULT_UAV_INDEX = Path("outputs/satloc/metadata/uav_frames_index_enriched.csv")
DEFAULT_SAT_INDEX = Path("outputs/satloc/metadata/satellite_tiles_index_enriched.csv")
DEFAULT_S5C_CANDIDATES = Path(
    "outputs/satloc/metadata/s5c_temporal/"
    "s5c2_lightglue_union_candidate_scores_top50_full263.csv"
)
DEFAULT_S5C_QUERY_SUMMARY = Path(
    "outputs/satloc/metadata/s5c_temporal/"
    "s5c2_lightglue_union_query_summary_top50_full263.csv"
)
DEFAULT_SCENE_LABELS = Path(
    "outputs/satloc/metadata/s7_retrieval_upgrade/s7_scene_labels_canonical_traj01.csv"
)

DEFAULT_METADATA_OUT = Path("outputs/satloc/metadata/s7b_dinov2_global")
DEFAULT_REPORT_OUT = Path("outputs/satloc/reports/s7b_dinov2_global")

KEY_NAME_HINTS = [
    "token",
    "token0",
    "token1",
    "sequence_frame",
    "frame_index",
    "global_frame",
    "s7_query_index",
    "query_index",
    "order",
    "index",
]

PATH_NAME_HINTS = [
    "path",
    "file",
    "filename",
    "image",
    "jpg",
    "jpeg",
    "png",
    "tile",
]

COORD_NAME_HINTS = [
    "lon",
    "lat",
    "utm",
    "enu",
    "x_m",
    "y_m",
]

COMMON_BASES = [
    Path("."),
    Path("data"),
    Path("dataset"),
    Path("datasets"),
    Path("outputs"),
    Path("outputs/satloc"),
    Path("data/satloc"),
    Path("datasets/satloc"),
    Path("datasets/SatLoc"),
    Path("satloc"),
    Path("SatLoc"),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", type=Path, default=Path("."))
    p.add_argument("--query-manifest", type=Path, default=DEFAULT_QUERY_MANIFEST)
    p.add_argument("--uav-index", type=Path, default=DEFAULT_UAV_INDEX)
    p.add_argument("--satellite-index", type=Path, default=DEFAULT_SAT_INDEX)
    p.add_argument("--s5c-candidates", type=Path, default=DEFAULT_S5C_CANDIDATES)
    p.add_argument("--s5c-query-summary", type=Path, default=DEFAULT_S5C_QUERY_SUMMARY)
    p.add_argument("--scene-labels", type=Path, default=DEFAULT_SCENE_LABELS)
    p.add_argument("--metadata-out", type=Path, default=DEFAULT_METADATA_OUT)
    p.add_argument("--report-out", type=Path, default=DEFAULT_REPORT_OUT)
    p.add_argument("--sample-rows", type=int, default=10)
    p.add_argument("--max-values-check", type=int, default=300)
    return p.parse_args()


def resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def read_csv(path: Path, name: str, required: bool = True) -> pd.DataFrame:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Missing {name}: {path}")
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def json_safe(x: Any) -> Any:
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, float) and not np.isfinite(x):
        return None
    try:
        if pd.isna(x):
            return None
    except Exception:
        pass
    return x


def likely_columns(df: pd.DataFrame, hints: list[str]) -> list[str]:
    cols = []
    for col in df.columns:
        low = str(col).lower()
        if any(h in low for h in hints):
            cols.append(col)
    return cols


def numeric_key_columns(df: pd.DataFrame) -> list[str]:
    cols = []
    for col in df.columns:
        low = str(col).lower()
        if not any(h in low for h in KEY_NAME_HINTS):
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        valid = int(s.notna().sum())
        nunique = int(s.dropna().astype(int).nunique()) if valid else 0
        if valid > 0 and nunique > 0:
            cols.append(col)
    return cols


def numeric_set(df: pd.DataFrame, col: str) -> set[int]:
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    if s.empty:
        return set()
    return set(s.astype(int).tolist())


def overlap_matrix(left: pd.DataFrame, right: pd.DataFrame, left_name: str, right_name: str) -> pd.DataFrame:
    left_cols = numeric_key_columns(left)
    right_cols = numeric_key_columns(right)

    rows = []
    for lc in left_cols:
        ls = numeric_set(left, lc)
        for rc in right_cols:
            rs = numeric_set(right, rc)
            inter = ls.intersection(rs)
            rows.append({
                "left_table": left_name,
                "left_col": lc,
                "left_unique": len(ls),
                "right_table": right_name,
                "right_col": rc,
                "right_unique": len(rs),
                "overlap": len(inter),
                "left_coverage": len(inter) / len(ls) if ls else 0.0,
                "right_coverage": len(inter) / len(rs) if rs else 0.0,
                "example_overlap_values": ",".join(map(str, sorted(list(inter))[:12])),
            })
    return pd.DataFrame(rows).sort_values(
        ["overlap", "left_coverage", "right_coverage"],
        ascending=[False, False, False],
    )


def candidate_paths(root: Path, value: Any) -> list[Path]:
    if value is None:
        return []
    try:
        if pd.isna(value):
            return []
    except Exception:
        pass

    text = str(value).strip()
    if not text:
        return []

    p = Path(text)
    out = []
    if p.is_absolute():
        out.append(p)
    else:
        out.append(root / p)
        for base in COMMON_BASES:
            out.append(root / base / p)

    if len(p.parts) == 1:
        name = p.name
        for base in [
            Path("data/satloc/uav"),
            Path("data/satloc/satellite"),
            Path("datasets/satloc/uav"),
            Path("datasets/satloc/satellite"),
            Path("outputs/satloc/uav"),
            Path("outputs/satloc/satellite"),
            Path("data"),
            Path("datasets"),
        ]:
            out.append(root / base / name)

    seen = set()
    dedup = []
    for x in out:
        key = str(x)
        if key not in seen:
            seen.add(key)
            dedup.append(x)
    return dedup


def resolve_one(root: Path, value: Any) -> str:
    for p in candidate_paths(root, value):
        if p.exists():
            return str(p)
    return ""


def path_column_report(root: Path, df: pd.DataFrame, table_name: str, max_values: int) -> pd.DataFrame:
    cols = likely_columns(df, PATH_NAME_HINTS)
    rows = []

    for col in cols:
        values = df[col].dropna().astype(str)
        values = values[values.str.strip().str.len() > 0]
        if values.empty:
            continue

        sample = values.head(max_values)
        resolved = [resolve_one(root, v) for v in sample]
        existing = sum(1 for x in resolved if x)

        example_raw = ""
        example_resolved = ""
        for raw, res in zip(sample.tolist(), resolved):
            if raw:
                example_raw = raw
                example_resolved = res
                if res:
                    break

        ext_counts = (
            values.map(lambda x: Path(str(x)).suffix.lower())
            .value_counts()
            .head(8)
            .to_dict()
        )

        rows.append({
            "table": table_name,
            "column": col,
            "nonempty_values": int(len(values)),
            "checked_values": int(len(sample)),
            "existing_count": int(existing),
            "existing_rate_checked": float(existing / len(sample)) if len(sample) else 0.0,
            "example_raw": example_raw,
            "example_resolved": example_resolved,
            "extension_counts": json.dumps(ext_counts),
        })

    return pd.DataFrame(rows).sort_values(
        ["existing_count", "nonempty_values"],
        ascending=[False, False],
    ) if rows else pd.DataFrame(columns=[
        "table", "column", "nonempty_values", "checked_values", "existing_count",
        "existing_rate_checked", "example_raw", "example_resolved", "extension_counts"
    ])


def table_overview(df: pd.DataFrame, name: str, sample_rows: int) -> dict[str, Any]:
    overview = {
        "name": name,
        "rows": int(len(df)),
        "cols": int(len(df.columns)),
        "columns": list(map(str, df.columns)),
        "key_like_columns": numeric_key_columns(df),
        "path_like_columns": likely_columns(df, PATH_NAME_HINTS),
        "coord_like_columns": likely_columns(df, COORD_NAME_HINTS),
    }

    if "sequence" in df.columns:
        overview["sequence_counts"] = {
            str(k): int(v)
            for k, v in df["sequence"].astype(str).value_counts(dropna=False).head(20).items()
        }

    return overview


def print_columns_block(name: str, df: pd.DataFrame) -> None:
    print()
    print(f"{name}")
    print("-" * len(name))
    print(f"rows={len(df)} cols={len(df.columns)}")
    print("columns:")
    for i, col in enumerate(df.columns):
        print(f"  {i:02d}: {col}")

    key_cols = numeric_key_columns(df)
    path_cols = likely_columns(df, PATH_NAME_HINTS)
    coord_cols = likely_columns(df, COORD_NAME_HINTS)
    print(f"key-like columns:  {key_cols}")
    print(f"path-like columns: {path_cols}")
    print(f"coord-like columns:{coord_cols}")

    if "sequence" in df.columns:
        print("sequence counts:")
        print(df["sequence"].astype(str).value_counts(dropna=False).head(20).to_string())


def print_top_path_report(title: str, report: pd.DataFrame, n: int = 12) -> None:
    print()
    print(title)
    print("-" * len(title))
    if report.empty:
        print("No path-like columns found.")
        return

    cols = [
        "table",
        "column",
        "nonempty_values",
        "checked_values",
        "existing_count",
        "existing_rate_checked",
        "example_raw",
        "example_resolved",
    ]
    print(report[cols].head(n).to_string(index=False))


def main() -> int:
    args = parse_args()
    root = args.repo_root.resolve()

    metadata_out = resolve(root, args.metadata_out)
    report_out = resolve(root, args.report_out)
    metadata_out.mkdir(parents=True, exist_ok=True)
    report_out.mkdir(parents=True, exist_ok=True)

    q_path = resolve(root, args.query_manifest)
    uav_path = resolve(root, args.uav_index)
    sat_path = resolve(root, args.satellite_index)
    cand_path = resolve(root, args.s5c_candidates)
    qsum_path = resolve(root, args.s5c_query_summary)
    scene_path = resolve(root, args.scene_labels)

    q = read_csv(q_path, "S7 query manifest")
    uav_global = read_csv(uav_path, "UAV frame index")
    sat = read_csv(sat_path, "satellite index")
    cand = read_csv(cand_path, "S5C candidate scores", required=False)
    qsum = read_csv(qsum_path, "S5C query summary", required=False)
    scene = read_csv(scene_path, "canonical scene labels", required=False)

    if "sequence" in uav_global.columns:
        uav_traj01 = uav_global.loc[
            uav_global["sequence"].astype(str).str.lower().eq("traj01")
        ].copy()
        if uav_traj01.empty:
            uav_traj01 = uav_global.copy()
    else:
        uav_traj01 = uav_global.copy()

    print()
    print("S7B.1-preflight — Path and join-key inspector")
    print("---------------------------------------------")
    print(f"Repository root: {root}")
    print()

    print_columns_block("S7 query manifest", q)
    print_columns_block("UAV index global", uav_global)
    print_columns_block("UAV index traj01-filtered", uav_traj01)
    print_columns_block("Satellite tile index", sat)
    if not cand.empty:
        print_columns_block("S5C candidate scores", cand)
    if not qsum.empty:
        print_columns_block("S5C query summary", qsum)
    if not scene.empty:
        print_columns_block("Canonical scene labels", scene)

    q_uav_overlap = overlap_matrix(q, uav_traj01, "s7_query_manifest", "uav_index_traj01")
    q_cand_overlap = overlap_matrix(q, cand, "s7_query_manifest", "s5c_candidate_scores") if not cand.empty else pd.DataFrame()
    q_qsum_overlap = overlap_matrix(q, qsum, "s7_query_manifest", "s5c_query_summary") if not qsum.empty else pd.DataFrame()
    q_scene_overlap = overlap_matrix(q, scene, "s7_query_manifest", "scene_labels") if not scene.empty else pd.DataFrame()

    print()
    print("Best query -> UAV index join candidates")
    print("---------------------------------------")
    print(q_uav_overlap.head(20).to_string(index=False) if not q_uav_overlap.empty else "No numeric overlap.")

    if not q_cand_overlap.empty:
        print()
        print("Best query -> S5C candidate-score join candidates")
        print("-------------------------------------------------")
        print(q_cand_overlap.head(20).to_string(index=False))

    if not q_qsum_overlap.empty:
        print()
        print("Best query -> S5C query-summary join candidates")
        print("-----------------------------------------------")
        print(q_qsum_overlap.head(20).to_string(index=False))

    if not q_scene_overlap.empty:
        print()
        print("Best query -> canonical scene-label join candidates")
        print("--------------------------------------------------")
        print(q_scene_overlap.head(20).to_string(index=False))

    q_paths = path_column_report(root, q, "s7_query_manifest", args.max_values_check)
    uav_paths = path_column_report(root, uav_traj01, "uav_index_traj01", args.max_values_check)
    sat_paths = path_column_report(root, sat, "satellite_index", args.max_values_check)
    cand_paths = path_column_report(root, cand, "s5c_candidate_scores", args.max_values_check) if not cand.empty else pd.DataFrame()

    print_top_path_report("Path resolution: S7 query manifest", q_paths)
    print_top_path_report("Path resolution: UAV index traj01", uav_paths)
    print_top_path_report("Path resolution: satellite index", sat_paths)
    if not cand_paths.empty:
        print_top_path_report("Path resolution: S5C candidate scores", cand_paths)

    # Print recommended keys based purely on overlap/path-existence.
    recommendation: dict[str, Any] = {}

    if not q_uav_overlap.empty:
        best = q_uav_overlap.iloc[0]
        recommendation["query_to_uav_join"] = {
            "query_column": best["left_col"],
            "uav_column": best["right_col"],
            "overlap": int(best["overlap"]),
            "left_coverage": float(best["left_coverage"]),
            "right_coverage": float(best["right_coverage"]),
        }

    if not q_cand_overlap.empty:
        best = q_cand_overlap.iloc[0]
        recommendation["query_to_s5c_candidate_join"] = {
            "query_column": best["left_col"],
            "candidate_column": best["right_col"],
            "overlap": int(best["overlap"]),
            "left_coverage": float(best["left_coverage"]),
            "right_coverage": float(best["right_coverage"]),
        }

    if not uav_paths.empty:
        best = uav_paths.iloc[0]
        recommendation["uav_best_path_column"] = {
            "column": best["column"],
            "existing_count_checked": int(best["existing_count"]),
            "existing_rate_checked": float(best["existing_rate_checked"]),
            "example_raw": best["example_raw"],
            "example_resolved": best["example_resolved"],
        }

    if not cand_paths.empty:
        best = cand_paths.iloc[0]
        recommendation["s5c_best_path_hint_column"] = {
            "column": best["column"],
            "existing_count_checked": int(best["existing_count"]),
            "existing_rate_checked": float(best["existing_rate_checked"]),
            "example_raw": best["example_raw"],
            "example_resolved": best["example_resolved"],
        }

    if not sat_paths.empty:
        best = sat_paths.iloc[0]
        recommendation["satellite_best_path_column"] = {
            "column": best["column"],
            "existing_count_checked": int(best["existing_count"]),
            "existing_rate_checked": float(best["existing_rate_checked"]),
            "example_raw": best["example_raw"],
            "example_resolved": best["example_resolved"],
        }

    print()
    print("Recommendation")
    print("--------------")
    print(json.dumps(recommendation, indent=2, default=json_safe))

    overlap_out = metadata_out / "s7b1_preflight_query_uav_join_overlap.csv"
    q_uav_overlap.to_csv(overlap_out, index=False)

    if not q_cand_overlap.empty:
        q_cand_overlap.to_csv(metadata_out / "s7b1_preflight_query_s5c_candidate_join_overlap.csv", index=False)
    if not q_qsum_overlap.empty:
        q_qsum_overlap.to_csv(metadata_out / "s7b1_preflight_query_s5c_summary_join_overlap.csv", index=False)
    if not q_scene_overlap.empty:
        q_scene_overlap.to_csv(metadata_out / "s7b1_preflight_query_scene_join_overlap.csv", index=False)

    path_report = pd.concat(
        [q_paths, uav_paths, sat_paths, cand_paths],
        ignore_index=True,
    )
    path_report_out = metadata_out / "s7b1_preflight_path_resolution_report.csv"
    path_report.to_csv(path_report_out, index=False)

    report = {
        "stage": "S7B.1_preflight_path_and_join_inspector",
        "status": "COMPLETE",
        "repo_root": str(root),
        "inputs": {
            "query_manifest": str(q_path),
            "uav_index": str(uav_path),
            "satellite_index": str(sat_path),
            "s5c_candidates": str(cand_path),
            "s5c_query_summary": str(qsum_path),
            "scene_labels": str(scene_path),
        },
        "tables": {
            "s7_query_manifest": table_overview(q, "s7_query_manifest", args.sample_rows),
            "uav_index_global": table_overview(uav_global, "uav_index_global", args.sample_rows),
            "uav_index_traj01": table_overview(uav_traj01, "uav_index_traj01", args.sample_rows),
            "satellite_index": table_overview(sat, "satellite_index", args.sample_rows),
            "s5c_candidates": table_overview(cand, "s5c_candidates", args.sample_rows) if not cand.empty else None,
            "s5c_query_summary": table_overview(qsum, "s5c_query_summary", args.sample_rows) if not qsum.empty else None,
            "scene_labels": table_overview(scene, "scene_labels", args.sample_rows) if not scene.empty else None,
        },
        "recommendation": recommendation,
        "outputs": {
            "query_uav_overlap_csv": str(overlap_out),
            "path_resolution_report_csv": str(path_report_out),
        },
    }

    report_out_path = report_out / "s7b1_preflight_inspect_paths_summary.json"
    with open(report_out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=json_safe)

    print()
    print("Outputs")
    print("-------")
    print(f"Join overlap CSV:       {overlap_out}")
    print(f"Path report CSV:        {path_report_out}")
    print(f"Summary JSON:           {report_out_path}")
    print()
    print("Next step: paste the Recommendation block and the top path-resolution rows.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
