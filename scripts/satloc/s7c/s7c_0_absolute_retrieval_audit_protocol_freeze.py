#!/usr/bin/env python3
"""S7C.0 absolute-retrieval audit and protocol freeze.

This block does not run retrieval, LightGlue, fusion, or ground-truth-guided
selection. It inventories the repository state, verifies the frozen S7
closeout documentation, audits required S5/S6/S7 metadata, checks schemas for
reference/evaluation-only leakage, and writes a reproducible S7C protocol.

Ground truth/reference coordinates may appear in source files for evaluation,
but they must never be consumed by online candidate generation, ranking,
verification, confidence gating, or correction acceptance.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


STAGE_NAME = "S7C.0 — Absolute Retrieval Audit and Protocol Freeze"
PROTOCOL_VERSION = "s7c0_v1.0"

EXPECTED_S7_ASSETS = (
    "s7a2_orb_xfeat_prefix_locked_xy.png",
    "s7a2_orb_xfeat_error_vs_distance.png",
    "s7a2_orb_xfeat_inlier_ratio.png",
    "s7b0_error_vs_distance.png",
    "s7b0_prefix_locked_xy.png",
    "s7b0_orb_flow_disagreement.png",
    "s7b0_orb_flow_inlier_ratio.png",
    "s7b1_error_vs_distance.png",
    "s7b1_prefix_locked_xy.png",
    "s7b1_confidence_signal.png",
    "s7b1_smoothing_delta.png",
)

# Exact names and patterns are intentionally bounded. The script does not scan
# arbitrary outputs and silently choose the newest file.
ARTIFACT_SPECS: dict[str, dict[str, Any]] = {
    "s7_relative_readme": {
        "kind": "documentation",
        "required": False,
        "paths": [
            "docs/reports/README_s7_relative_frontend_closeout_s7c_transition.md",
            "docs/README_s7_relative_frontend_closeout_s7c_transition.md",
        ],
    },
    "uav_frame_index": {
        "kind": "core_input",
        "required": True,
        "paths": ["outputs/satloc/metadata/uav_frames_index_enriched.csv"],
    },
    "satellite_tile_index": {
        "kind": "core_input",
        "required": True,
        "paths": ["outputs/satloc/metadata/satellite_tiles_index_enriched.csv"],
    },
    "s7_query_manifest": {
        "kind": "benchmark_manifest",
        "required": False,
        "paths": [
            "outputs/satloc/metadata/s7_retrieval_upgrade/s7_0_query_manifest.csv",
            "outputs/satloc/metadata/s7_retrieval_upgrade/s7c0_query_manifest.csv",
        ],
    },
    "s7_scene_sampled_frames": {
        "kind": "scene_manifest",
        "required": True,
        "paths": [
            "outputs/satloc/metadata/s7_retrieval_upgrade/s7_0_scene_sampled_frames.csv",
            "outputs/satloc/metadata/s7_retrieval_upgrade/s7_0c_scene_sampled_frames.csv",
        ],
    },
    "s7_scene_sampling_summary": {
        "kind": "scene_summary",
        "required": True,
        "paths": [
            "outputs/satloc/metadata/s7_retrieval_upgrade/s7_0_scene_sampling_summary.json",
            "outputs/satloc/reports/s7_retrieval_upgrade/s7_0_scene_sampling_summary.json",
        ],
    },
    "s5c_official_query_manifest": {
        "kind": "benchmark_manifest",
        "required": True,
        "paths": [
            "outputs/satloc/metadata/s5c_temporal/s5c0_absolute_query_manifest.csv"
        ],
    },
    "s5c_union_candidate_scores_top50": {
        "kind": "candidate_scores",
        "required": True,
        "paths": [
            "outputs/satloc/metadata/s5c_temporal/"
            "s5c2_lightglue_union_candidate_scores_top50_full263.csv"
        ],
    },
    "s5c_lightglue_query_summary": {
        "kind": "verifier_summary",
        "required": True,
        "paths": [
            "outputs/satloc/metadata/s5c_temporal/"
            "s5c2_lightglue_union_query_summary_top50_full263.csv"
        ],
    },
    "s5c_lightglue_policy_summary": {
        "kind": "verifier_summary",
        "required": True,
        "paths": [
            "outputs/satloc/metadata/s5c_temporal/"
            "s5c2_lightglue_union_policy_summary_top50_full263.csv"
        ],
    },
    "s6b_absolute_correction_manifest": {
        "kind": "fusion_handoff",
        "required": True,
        "paths": [
            "outputs/satloc/metadata/s6b_relative_absolute/"
            "s6b0_absolute_correction_manifest.csv"
        ],
    },
    "s5b_union_pool": {
        "kind": "candidate_pool_history",
        "required": False,
        "glob_patterns": [
            "outputs/satloc/metadata/s5b_candidate_pool_improvement/"
            "s5b1c_union_candidate_pool*.csv",
            "outputs/satloc/metadata/s5b_candidate_pool_improvement/"
            "s5b1c_union_query_summary*.csv",
        ],
    },
    "s4c_failure_analysis": {
        "kind": "failure_taxonomy_history",
        "required": False,
        "glob_patterns": [
            "outputs/satloc/metadata/s4c6_failure_analysis/*failure*group*.csv",
            "outputs/satloc/metadata/s4c6_failure_analysis/*.csv",
        ],
    },
}

# These fields may be retained for post-ranking evaluation only. Presence is not
# itself an error. The audit makes the isolation boundary explicit.
EVALUATION_ONLY_EXACT = {
    "chosen_error_m",
    "eval_error_m",
    "error_m",
    "top1_error_m",
    "oracle_error_m",
    "oracle_best_error_m",
    "oracle_rank",
    "first_correct_rank",
    "hit_le_threshold",
    "hit_eval_only",
    "dangerous_false_eval_only",
    "reference_x_enu_m",
    "reference_y_enu_m",
    "reference_lat",
    "reference_lon",
    "uav_lon",
    "uav_lat",
    "gt_lon",
    "gt_lat",
    "gt_x_enu_m",
    "gt_y_enu_m",
}

EVALUATION_ONLY_PREFIXES = (
    "oracle_",
    "eval_",
    "reference_",
    "ground_truth_",
    "gt_",
)

ONLINE_SCORE_HINTS = (
    "phog",
    "hog",
    "lsd",
    "lightglue",
    "lg_",
    "score",
    "match",
    "inlier",
    "coverage",
    "margin",
    "candidate_rank",
    "union_rank",
    "tile_center",
)

QUERY_ID_ALIASES = (
    "token0_id",
    "token",
    "query_token",
    "query_id",
    "frame_id",
    "uav_token",
)

SCENE_LABEL_ALIASES = (
    "scene_label",
    "scene_group",
    "scene_type",
    "final_scene_label",
    "taxonomy_label",
)


@dataclass
class ArtifactAudit:
    name: str
    kind: str
    required: bool
    status: str
    resolved_path: str | None
    bytes: int | None
    row_count: int | None
    columns: list[str]
    evaluation_only_columns: list[str]
    online_score_columns: list[str]
    notes: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=STAGE_NAME)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="Repository root. Defaults to current working directory.",
    )
    parser.add_argument("--sequence", default="traj01")
    parser.add_argument("--expected-frames", type=int, default=1034)
    parser.add_argument("--expected-pairs", type=int, default=1033)
    parser.add_argument("--expected-tiles", type=int, default=8625)
    parser.add_argument("--expected-official-queries", type=int, default=263)
    parser.add_argument("--expected-scene-sampled-frames", type=int, default=169)
    parser.add_argument(
        "--metadata-out",
        type=Path,
        default=Path("outputs/satloc/metadata/s7_retrieval_upgrade"),
    )
    parser.add_argument(
        "--report-out",
        type=Path,
        default=Path("outputs/satloc/reports/s7_retrieval_upgrade"),
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return exit code 2 when a required input or invariant is missing.",
    )
    return parser.parse_args()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def resolve_artifact(repo_root: Path, spec: dict[str, Any]) -> tuple[Path | None, list[str]]:
    notes: list[str] = []
    for rel in spec.get("paths", []):
        path = repo_root / rel
        if path.is_file():
            return path, notes

    matches: list[Path] = []
    for pattern in spec.get("glob_patterns", []):
        matches.extend(path for path in repo_root.glob(pattern) if path.is_file())
    matches = sorted(set(matches))
    if matches:
        if len(matches) > 1:
            notes.append(
                "Multiple bounded matches found; selected lexicographically first: "
                + ", ".join(str(p.relative_to(repo_root)) for p in matches[:8])
            )
        return matches[0], notes
    return None, notes


def read_csv_header_and_rows(path: Path) -> tuple[list[str], int, list[str]]:
    notes: list[str] = []
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            sample = handle.read(65536)
            handle.seek(0)
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
            except csv.Error:
                dialect = csv.excel
                notes.append("CSV dialect sniff failed; used comma-separated default.")
            reader = csv.reader(handle, dialect)
            try:
                header = [cell.strip() for cell in next(reader)]
            except StopIteration:
                return [], 0, ["CSV is empty."]
            row_count = sum(1 for row in reader if any(cell.strip() for cell in row))
            return header, row_count, notes
    except (OSError, UnicodeError, csv.Error) as exc:
        return [], -1, [f"CSV read error: {exc}"]


def classify_columns(columns: Iterable[str]) -> tuple[list[str], list[str]]:
    evaluation_only: list[str] = []
    online_score: list[str] = []
    for original in columns:
        col = original.strip().lower()
        if col in EVALUATION_ONLY_EXACT or col.startswith(EVALUATION_ONLY_PREFIXES):
            evaluation_only.append(original)
        if any(hint in col for hint in ONLINE_SCORE_HINTS):
            online_score.append(original)
    return sorted(set(evaluation_only)), sorted(set(online_score))


def audit_artifact(repo_root: Path, name: str, spec: dict[str, Any]) -> ArtifactAudit:
    path, notes = resolve_artifact(repo_root, spec)
    if path is None:
        return ArtifactAudit(
            name=name,
            kind=spec["kind"],
            required=bool(spec.get("required", False)),
            status="MISSING",
            resolved_path=None,
            bytes=None,
            row_count=None,
            columns=[],
            evaluation_only_columns=[],
            online_score_columns=[],
            notes=notes,
        )

    columns: list[str] = []
    row_count: int | None = None
    evaluation_only_columns: list[str] = []
    online_score_columns: list[str] = []

    if path.suffix.lower() == ".csv":
        columns, row_count, csv_notes = read_csv_header_and_rows(path)
        notes.extend(csv_notes)
        evaluation_only_columns, online_score_columns = classify_columns(columns)
    elif path.suffix.lower() == ".json":
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if isinstance(payload, list):
                row_count = len(payload)
            elif isinstance(payload, dict):
                columns = sorted(str(k) for k in payload.keys())
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            notes.append(f"JSON read error: {exc}")

    return ArtifactAudit(
        name=name,
        kind=spec["kind"],
        required=bool(spec.get("required", False)),
        status="FOUND",
        resolved_path=str(path.relative_to(repo_root)),
        bytes=path.stat().st_size,
        row_count=row_count,
        columns=columns,
        evaluation_only_columns=evaluation_only_columns,
        online_score_columns=online_score_columns,
        notes=notes,
    )


def get_audit(audits: Sequence[ArtifactAudit], name: str) -> ArtifactAudit:
    return next(audit for audit in audits if audit.name == name)


def find_column(columns: Sequence[str], aliases: Sequence[str]) -> str | None:
    lookup = {column.lower(): column for column in columns}
    for alias in aliases:
        if alias.lower() in lookup:
            return lookup[alias.lower()]
    return None


def validate_invariants(
    audits: Sequence[ArtifactAudit],
    args: argparse.Namespace,
    repo_root: Path,
) -> tuple[list[str], list[str]]:
    blockers: list[str] = []
    warnings: list[str] = []

    for audit in audits:
        if audit.required and audit.status != "FOUND":
            blockers.append(f"Required artifact missing: {audit.name}")
        if audit.status == "FOUND" and audit.row_count == -1:
            blockers.append(f"Unreadable CSV: {audit.name}")

    row_expectations = {
        "uav_frame_index": args.expected_frames,
        "satellite_tile_index": args.expected_tiles,
        "s5c_official_query_manifest": args.expected_official_queries,
        "s5c_lightglue_query_summary": args.expected_official_queries,
        "s6b_absolute_correction_manifest": args.expected_official_queries,
        "s7_scene_sampled_frames": args.expected_scene_sampled_frames,
    }
    for name, expected in row_expectations.items():
        audit = get_audit(audits, name)
        if audit.status != "FOUND" or audit.row_count is None or audit.row_count < 0:
            continue
        if audit.row_count != expected:
            blockers.append(
                f"Row-count invariant failed for {name}: "
                f"expected {expected}, found {audit.row_count}."
            )

    s7_query = get_audit(audits, "s7_query_manifest")
    if s7_query.status == "FOUND" and s7_query.row_count != args.expected_official_queries:
        blockers.append(
            f"S7 query manifest exists but has {s7_query.row_count} rows; "
            f"expected {args.expected_official_queries}."
        )
    elif s7_query.status != "FOUND":
        warnings.append(
            "S7-specific query manifest is absent; S7C.0 will freeze the existing "
            "S5C 263-query manifest as the official benchmark source until a copied, "
            "content-identical S7 manifest is created."
        )

    scene_audit = get_audit(audits, "s7_scene_sampled_frames")
    if scene_audit.status == "FOUND":
        scene_col = find_column(scene_audit.columns, SCENE_LABEL_ALIASES)
        query_col = find_column(scene_audit.columns, QUERY_ID_ALIASES)
        if scene_col is None:
            blockers.append(
                "Scene-sampled manifest has no recognized scene-label column. "
                f"Tried aliases: {', '.join(SCENE_LABEL_ALIASES)}."
            )
        if query_col is None:
            blockers.append(
                "Scene-sampled manifest has no recognized query/frame identifier column. "
                f"Tried aliases: {', '.join(QUERY_ID_ALIASES)}."
            )

    # Documentation is important, but it does not block the metadata audit.
    readme = get_audit(audits, "s7_relative_readme")
    if readme.status != "FOUND":
        warnings.append(
            "S7 relative closeout README is not present at the frozen docs/reports path."
        )

    assets_root = repo_root / "docs/assets/s7_relative_frontend"
    missing_assets = [name for name in EXPECTED_S7_ASSETS if not (assets_root / name).is_file()]
    if missing_assets:
        warnings.append(
            f"S7 documentation assets missing {len(missing_assets)}/{len(EXPECTED_S7_ASSETS)}: "
            + ", ".join(missing_assets)
        )

    # The candidate score table is allowed to contain evaluation columns, but it
    # must also expose online image-derived evidence for replay/audit.
    candidate_scores = get_audit(audits, "s5c_union_candidate_scores_top50")
    if candidate_scores.status == "FOUND" and not candidate_scores.online_score_columns:
        blockers.append(
            "Candidate-score table has no recognized online image-derived score/evidence columns."
        )

    return blockers, warnings


def write_inventory_csv(path: Path, audits: Sequence[ArtifactAudit]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "name",
        "kind",
        "required",
        "status",
        "resolved_path",
        "bytes",
        "row_count",
        "column_count",
        "evaluation_only_columns",
        "online_score_columns",
        "notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for audit in audits:
            writer.writerow(
                {
                    "name": audit.name,
                    "kind": audit.kind,
                    "required": audit.required,
                    "status": audit.status,
                    "resolved_path": audit.resolved_path or "",
                    "bytes": audit.bytes if audit.bytes is not None else "",
                    "row_count": audit.row_count if audit.row_count is not None else "",
                    "column_count": len(audit.columns),
                    "evaluation_only_columns": "|".join(audit.evaluation_only_columns),
                    "online_score_columns": "|".join(audit.online_score_columns),
                    "notes": " | ".join(audit.notes),
                }
            )


def build_protocol(args: argparse.Namespace, benchmark_source: str) -> dict[str, Any]:
    return {
        "stage": STAGE_NAME,
        "protocol_version": PROTOCOL_VERSION,
        "created_utc": utc_now_iso(),
        "dataset": {
            "name": "SatLoc",
            "sequence": args.sequence,
            "expected_uav_frames": args.expected_frames,
            "expected_relative_pairs": args.expected_pairs,
            "expected_satellite_tiles": args.expected_tiles,
            "official_query_count": args.expected_official_queries,
            "scene_sampled_frame_count": args.expected_scene_sampled_frames,
            "official_benchmark_source": benchmark_source,
        },
        "locked_relative_context": {
            "frontend": "ORB affine stride-1",
            "backend": "SE(2) scale-normalized accumulation",
            "decision": "KEEP_ORB",
            "prefix_rmse_m": 230.188,
            "final_drift_m_per_100m": 2.461,
        },
        "frozen_absolute_baselines": {
            "s5c_union_oracle_at_50": {"hits": 102, "queries": 263, "rate": 0.3878},
            "s5c_lightglue_selected_hits": {"hits": 68, "queries": 263, "rate": 0.2586},
            "s5c_balanced_gate": {
                "accepted": 33,
                "true_hits_eval_only": 25,
                "false_accepts_eval_only": 8,
                "precision_eval_only": 0.7576,
                "dangerous_false_accepts_gt_100m_eval_only": 0,
            },
            "s6b_relative_only_rmse_m": 224.668,
            "s6b_balanced_hard_reset_rmse_m": 84.559,
            "s6b_best_controlled_soft_rmse_m": 70.688,
        },
        "ground_truth_isolation": {
            "rule": (
                "Reference/GNSS/ENU/UAV filename coordinates are evaluation-only and "
                "must not influence candidate generation, ranking, feature matching, "
                "threshold selection, confidence gating, or correction acceptance."
            ),
            "online_usable_families": [
                "UAV image content",
                "satellite image content and georeferenced tile metadata",
                "PHOG/HOG/LSD/domain-normalized descriptor scores",
                "candidate ranks and candidate-union provenance",
                "LightGlue matches, inliers, coverage, score and margin",
                "runtime and memory measurements",
                "scene label only for reporting or predeclared scene-policy ablation",
            ],
            "evaluation_only_families": [
                "UAV/reference longitude and latitude",
                "reference ENU coordinates",
                "candidate error in metres",
                "hit labels and failure-group labels derived from error",
                "oracle identity, oracle rank and oracle error",
                "dangerous-false labels",
            ],
        },
        "primary_objective": (
            "Increase the probability that a correct or near-correct satellite tile "
            "enters a bounded candidate pool before LightGlue verification."
        ),
        "primary_metric": {
            "name": "candidate_pool_recall_at_k",
            "success_threshold_m": 40.0,
            "k_values": [1, 5, 10, 20, 50, 100, 200],
            "primary_k": 50,
        },
        "secondary_metrics": [
            "top1_error_m",
            "best_top5_error_m",
            "best_top10_error_m",
            "median_top1_error_m",
            "oracle_in_pool_error_m",
            "first_correct_rank",
            "candidate_pool_size",
            "LightGlue accepted precision and recall",
            "dangerous false accepts above 100 m",
            "runtime per query and per candidate",
            "peak memory where available",
            "scene-group breakdown",
        ],
        "scene_groups": [
            "forest_canopy",
            "urban",
            "urban_rotation",
            "agricultural_open_field",
            "water_wetland",
            "unreviewed_or_other",A
        ],
        "s7c1_failure_taxonomy": [
            "true_tile_missing_from_candidate_pool",
            "near_tile_present_but_not_selected",
            "rotation_mismatch",
            "scale_or_fov_mismatch",
            "vegetation_or_field_ambiguity",
            "water_or_wetland_ambiguity",
            "urban_or_road_corridor_ambiguity",
            "satellite_texture_sparsity_or_map_age_mismatch",
        ],
        "bounded_experiment_rules": {
            "no_giant_sweep": True,
            "candidate_union_must_report_unique_pool_size": True,
            "all_new_methods_compare_on_same_263_queries": True,
            "scene_subsets_are_diagnostic_not_replacement_benchmarks": True,
            "LightGlue_replay_only_after_candidate_recall_is_measured": True,
            "new_thresholds_must_be_declared_before_evaluation": True,
        },
        "s7c2_initial_bounded_variants": [
            "baseline existing four-variant union",
            "multi-rotation structural descriptor with descriptor-space circular shifts",
            "bounded FOV-normalized UAV crop/scale variants",
            "controlled candidate union with fixed per-variant depth",
        ],
        "stop_conditions": {
            "advance_to_s7c3": (
                "Candidate-pool recall@50 improves on the official 263-query benchmark "
                "without unacceptable runtime/pool growth and with gains across more than "
                "one scene group."
            ),
            "close_without_gain": (
                "No meaningful candidate recall gain after the declared bounded variants, "
                "or gains come only from excessive candidate-pool expansion."
            ),
        },
    }


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.expanduser().resolve()
    metadata_out = (repo_root / args.metadata_out).resolve()
    report_out = (repo_root / args.report_out).resolve()
    metadata_out.mkdir(parents=True, exist_ok=True)
    report_out.mkdir(parents=True, exist_ok=True)

    audits = [
        audit_artifact(repo_root, name, spec)
        for name, spec in ARTIFACT_SPECS.items()
    ]
    blockers, warnings = validate_invariants(audits, args, repo_root)

    s7_query = get_audit(audits, "s7_query_manifest")
    if s7_query.status == "FOUND" and s7_query.resolved_path:
        benchmark_source = s7_query.resolved_path
    else:
        benchmark_source = (
            get_audit(audits, "s5c_official_query_manifest").resolved_path
            or "MISSING"
        )

    assets_root = repo_root / "docs/assets/s7_relative_frontend"
    asset_status = {
        asset: (assets_root / asset).is_file() for asset in EXPECTED_S7_ASSETS
    }

    status = "READY" if not blockers else "BLOCKED"
    if not blockers and warnings:
        status = "READY_WITH_WARNINGS"

    inventory_path = metadata_out / "s7c0_input_inventory.csv"
    protocol_path = metadata_out / "s7c0_protocol_freeze.json"
    summary_path = report_out / "s7c0_audit_summary.json"

    write_inventory_csv(inventory_path, audits)
    protocol = build_protocol(args, benchmark_source)
    with protocol_path.open("w", encoding="utf-8") as handle:
        json.dump(protocol, handle, indent=2, sort_keys=False)
        handle.write("\n")

    summary = {
        "stage": STAGE_NAME,
        "status": status,
        "created_utc": utc_now_iso(),
        "repo_root": str(repo_root),
        "sequence": args.sequence,
        "blockers": blockers,
        "warnings": warnings,
        "documentation": {
            "readme_found": get_audit(audits, "s7_relative_readme").status == "FOUND",
            "readme_path": get_audit(audits, "s7_relative_readme").resolved_path,
            "asset_root": str(assets_root.relative_to(repo_root)),
            "assets_found": sum(asset_status.values()),
            "assets_expected": len(asset_status),
            "asset_status": asset_status,
        },
        "benchmark_source": benchmark_source,
        "artifacts": [asdict(audit) for audit in audits],
        "outputs": {
            "inventory_csv": str(inventory_path.relative_to(repo_root)),
            "protocol_json": str(protocol_path.relative_to(repo_root)),
            "summary_json": str(summary_path.relative_to(repo_root)),
        },
    }
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=False)
        handle.write("\n")

    print(STAGE_NAME)
    print("-" * len(STAGE_NAME))
    print(f"Status:                         {status}")
    print(f"Repository root:                {repo_root}")
    print(f"Sequence:                       {args.sequence}")
    print(f"Official benchmark source:      {benchmark_source}")
    print(
        "S7 README:                     "
        + (get_audit(audits, "s7_relative_readme").resolved_path or "MISSING")
    )
    print(
        "S7 documentation assets:       "
        f"{sum(asset_status.values())}/{len(asset_status)}"
    )
    print(
        "Required artifacts found:      "
        f"{sum(a.required and a.status == 'FOUND' for a in audits)}/"
        f"{sum(a.required for a in audits)}"
    )
    for name in (
        "uav_frame_index",
        "satellite_tile_index",
        "s5c_official_query_manifest",
        "s7_scene_sampled_frames",
        "s5c_union_candidate_scores_top50",
        "s6b_absolute_correction_manifest",
    ):
        audit = get_audit(audits, name)
        rows = "n/a" if audit.row_count is None else str(audit.row_count)
        print(f"{name + ':':32s} {audit.status:7s} rows={rows}")

    print(f"Blockers:                       {len(blockers)}")
    for blocker in blockers:
        print(f"  [BLOCK] {blocker}")
    print(f"Warnings:                       {len(warnings)}")
    for warning in warnings:
        print(f"  [WARN]  {warning}")

    print(f"Inventory CSV:                  {inventory_path.relative_to(repo_root)}")
    print(f"Protocol freeze JSON:           {protocol_path.relative_to(repo_root)}")
    print(f"Audit summary JSON:             {summary_path.relative_to(repo_root)}")

    if args.strict and blockers:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
