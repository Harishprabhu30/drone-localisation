#!/usr/bin/env python3
"""S7.0 Stage Preflight and Frozen Benchmark Specification.

This script does not run XFeat, DINOv2, VLAD, PHLO, LightGlue, or fusion.
It freezes the S7 benchmark contract and validates the completed S5C/S6 handoff.

Run from the repository root:

    source .drone_venv/bin/activate
    export PYTHONPATH=$PWD/src
    python scripts/satloc/s7/s7_0_stage_preflight.py \
      2>&1 | tee outputs/satloc/reports/s7_retrieval_upgrade/s7_0_stage_preflight.log

A BLOCKED result is intentional when a frozen input or documentation file is absent.
Fix the reported item and rerun; do not bypass the check by changing expected counts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


DEFAULT_OUTPUT_ROOT = Path("outputs/satloc")
DEFAULT_S5C_QUERY_MANIFEST = Path(
    "outputs/satloc/metadata/s5c_temporal/s5c0_absolute_query_manifest.csv"
)
DEFAULT_SEQUENCE_MANIFEST = Path(
    "outputs/satloc/metadata/s6a_relative_motion/s6a_sequence_manifest.csv"
)
DEFAULT_UAV_INDEX = Path(
    "outputs/satloc/metadata/uav_frames_index_enriched.csv"
)
DEFAULT_SATELLITE_INDEX = Path(
    "outputs/satloc/metadata/satellite_tiles_index_enriched.csv"
)
DEFAULT_DATASET_ROOT = Path("data/raw/satloc/part_1")

S6_README_CANDIDATES = (
    Path("README_satloc_s6_relative_absolute_fusion.md"),
    Path("docs/README_satloc_s6_relative_absolute_fusion.md"),
)
S7_PLAN_CANDIDATES = (
    Path("README_satloc_s7_candidate_generation_and_frontend_upgrade_plan.md"),
    Path("docs/README_satloc_s7_candidate_generation_and_frontend_upgrade_plan.md"),
)

EXPECTED_ONLINE_FIELDS = [
    "image_content",
    "descriptor_similarity",
    "candidate_rank",
    "lightglue_score",
    "matches",
    "inliers",
    "inlier_ratio",
    "coverage",
    "score_margin",
    "selected_tile_map_coordinate",
    "relative_visual_displacement",
    "temporal_displacement_agreement",
    "sequence_order",
    "time_or_distance_since_accepted_correction",
]

EVALUATION_ONLY_FIELDS = [
    "uav_filename_reference_longitude",
    "uav_filename_reference_latitude",
    "reference_trajectory_x_y",
    "chosen_error_m",
    "hit_eval_only",
    "hit_le_threshold",
    "oracle_tile_identity",
    "oracle_rank",
    "oracle_error",
    "dangerous_false_eval_only",
    "post_correction_ground_truth_improvement",
]


@dataclass
class CheckResult:
    name: str
    status: str
    observed: str
    expected: str
    detail: str


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def relative_or_absolute(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def run_git(repo_root: Path, args: list[str]) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    value = completed.stdout.strip()
    return value or None


def find_first_existing(repo_root: Path, candidates: Iterable[Path]) -> Path | None:
    for candidate in candidates:
        resolved = repo_root / candidate
        if resolved.exists():
            return resolved
    return None


def read_csv(path: Path, name: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception as exc:  # pragma: no cover - diagnostic path
        raise RuntimeError(f"Unable to read {name}: {path}: {exc}") from exc


def package_version(distribution_name: str) -> str | None:
    try:
        return importlib.metadata.version(distribution_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def collect_torch_environment() -> dict[str, Any]:
    info: dict[str, Any] = {
        "installed": False,
        "version": package_version("torch"),
        "cuda_available": None,
        "cuda_device_count": None,
        "cuda_devices": [],
        "mps_available": None,
        "mps_built": None,
    }
    try:
        import torch  # type: ignore
    except Exception as exc:
        info["import_error"] = repr(exc)
        return info

    info["installed"] = True
    try:
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["cuda_device_count"] = int(torch.cuda.device_count())
        if torch.cuda.is_available():
            info["cuda_devices"] = [
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ]
    except Exception as exc:  # pragma: no cover - hardware specific
        info["cuda_error"] = repr(exc)

    try:
        info["mps_available"] = bool(torch.backends.mps.is_available())
        info["mps_built"] = bool(torch.backends.mps.is_built())
    except Exception as exc:  # pragma: no cover - hardware specific
        info["mps_error"] = repr(exc)

    return info


def detect_unique_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for column in candidates:
        if column in df.columns:
            return column
    return None


def add_check(
    checks: list[CheckResult],
    name: str,
    passed: bool,
    observed: Any,
    expected: Any,
    detail: str,
) -> None:
    checks.append(
        CheckResult(
            name=name,
            status="PASS" if passed else "FAIL",
            observed=str(observed),
            expected=str(expected),
            detail=detail,
        )
    )


def method_registry_rows() -> list[dict[str, str]]:
    return [
        {
            "stage": "frozen_baseline",
            "method_id": "orb_se2_scale_normalized",
            "role": "relative_frontend",
            "status": "frozen_s6_baseline",
            "input_domain": "consecutive_uav_frames",
            "descriptor_or_state_dimension": "runtime_from_existing_s6_outputs",
            "cache_dtype": "not_applicable",
            "rotation_handling": "partial_affine_ransac",
            "primary_promotion_metric": "trajectory_rmse_and_runtime",
            "notes": "Official relative baseline; reference remains evaluation-only.",
        },
        {
            "stage": "S7A",
            "method_id": "xfeat_relative",
            "role": "relative_frontend_comparison",
            "status": "planned_bounded_comparison",
            "input_domain": "consecutive_uav_frames",
            "descriptor_or_state_dimension": "record_from_runtime_model_configuration",
            "cache_dtype": "record_at_execution",
            "rotation_handling": "model_and_geometric_estimator",
            "primary_promotion_metric": "drift_rmse_runtime_vs_orb",
            "notes": "Do not replace ORB unless the full-trajectory comparison justifies it.",
        },
        {
            "stage": "frozen_baseline",
            "method_id": "s5c_union_phog_v3_v5_v8_v9",
            "role": "absolute_candidate_generator",
            "status": "frozen_s5c_baseline",
            "input_domain": "uav_to_satellite",
            "descriptor_or_state_dimension": "existing_cache_runtime_defined",
            "cache_dtype": "existing_cache",
            "rotation_handling": "none",
            "primary_promotion_metric": "recall_at_20_and_50",
            "notes": "Baseline Oracle@50 availability: 102/263, evaluation only.",
        },
        {
            "stage": "S7B",
            "method_id": "dinov2_global",
            "role": "learned_global_retrieval",
            "status": "planned",
            "input_domain": "uav_to_satellite",
            "descriptor_or_state_dimension": "detected_from_selected_backbone_at_runtime",
            "cache_dtype": "float32_then_float16_ablation",
            "rotation_handling": "none_initially",
            "primary_promotion_metric": "recall_at_20_and_50",
            "notes": "Backbone and layer are frozen in S7B.0, not guessed here.",
        },
        {
            "stage": "S7B",
            "method_id": "dino_vlad_anyloc_style",
            "role": "learned_local_aggregation_retrieval",
            "status": "planned",
            "input_domain": "uav_to_satellite_patch_features",
            "descriptor_or_state_dimension": "clusters_times_local_dim_before_optional_pca",
            "cache_dtype": "float32_then_float16_ablation",
            "rotation_handling": "none_initially",
            "primary_promotion_metric": "recall_at_20_and_50",
            "notes": "Cluster count and optional PCA dimension freeze in S7B.0.",
        },
        {
            "stage": "S7C",
            "method_id": "phlo_gaussian_lsd",
            "role": "rotation_aware_structural_retrieval",
            "status": "planned",
            "input_domain": "multi_scale_line_segments",
            "descriptor_or_state_dimension": "spatial_cells_times_orientation_bins_times_levels",
            "cache_dtype": "float32",
            "rotation_handling": "line_coordinates_plus_orientation_rotation",
            "primary_promotion_metric": "recall_at_20_and_50_by_scene_group",
            "notes": "Bin shift alone is not considered full spatial-pyramid rotation handling.",
        },
        {
            "stage": "S7C",
            "method_id": "phlo_rgf_lsd",
            "role": "structure_frontend_ablation",
            "status": "planned_after_gaussian_smoke",
            "input_domain": "rgf_or_edge_preserving_scale_space_lines",
            "descriptor_or_state_dimension": "same_contract_as_phlo_gaussian_lsd",
            "cache_dtype": "float32",
            "rotation_handling": "line_coordinates_plus_orientation_rotation",
            "primary_promotion_metric": "delta_recall_vs_gaussian_phlo",
            "notes": "Promote only with measured retrieval improvement.",
        },
        {
            "stage": "S7C",
            "method_id": "natural_macro_structure",
            "role": "natural_scene_companion_stream",
            "status": "planned",
            "input_domain": "coarse_texture_canopy_field_boundary_mass",
            "descriptor_or_state_dimension": "freeze_after_diagnostic_design",
            "cache_dtype": "float32",
            "rotation_handling": "method_specific",
            "primary_promotion_metric": "natural_scene_recall_without_urban_regression",
            "notes": "Suppress unstable fine vegetation texture, not all vegetation information.",
        },
        {
            "stage": "S7D",
            "method_id": "reciprocal_rank_fusion",
            "role": "candidate_fusion",
            "status": "planned_first_fusion",
            "input_domain": "ranked_candidate_lists",
            "descriptor_or_state_dimension": "not_applicable",
            "cache_dtype": "not_applicable",
            "rotation_handling": "inherits_streams",
            "primary_promotion_metric": "recall_at_20_and_50",
            "notes": "Begin with rank fusion before score fusion.",
        },
        {
            "stage": "S7D",
            "method_id": "geographic_diversification",
            "role": "candidate_pool_diversification",
            "status": "planned",
            "input_domain": "ranked_tiles_and_map_coordinates",
            "descriptor_or_state_dimension": "not_applicable",
            "cache_dtype": "not_applicable",
            "rotation_handling": "not_applicable",
            "primary_promotion_metric": "recall_at_20_with_redundancy_reduction",
            "notes": "Uses candidate map coordinates only; never query reference coordinates.",
        },
        {
            "stage": "S7E",
            "method_id": "lightglue_topk_verifier",
            "role": "local_geometric_verifier",
            "status": "frozen_top50_reference_then_top10_top20_comparison",
            "input_domain": "uav_candidate_tile_pairs",
            "descriptor_or_state_dimension": "not_applicable",
            "cache_dtype": "runtime",
            "rotation_handling": "targeted_ablation_only",
            "primary_promotion_metric": "selected_hits_precision_runtime",
            "notes": "Measure wall-clock time; do not infer efficiency only from smaller K.",
        },
    ]


def metric_specification() -> dict[str, Any]:
    return {
        "version": "s7.0-v1",
        "sequence": "traj01",
        "correct_region_threshold_m_eval_only": 40.0,
        "retrieval_k_values": [1, 5, 10, 20, 50],
        "candidate_retrieval": {
            "recall_at_k": (
                "Fraction of queries having at least one candidate with evaluation-only "
                "centre error <= 40 m in the first K ranks."
            ),
            "median_correct_region_rank": (
                "Median first rank satisfying the evaluation-only 40 m condition; "
                "report candidate-pool failures separately."
            ),
            "mean_reciprocal_rank": (
                "Mean reciprocal first-correct rank; candidate-pool failures contribute zero."
            ),
            "candidate_pool_failure_count": (
                "Queries with no <=40 m candidate within the evaluated K."
            ),
        },
        "efficiency": [
            "satellite_cache_build_time_s",
            "query_descriptor_time_ms",
            "retrieval_search_time_ms",
            "descriptor_dimension",
            "descriptor_bytes_per_item",
            "cache_memory_bytes",
            "lightglue_candidates_per_query",
            "lightglue_wall_clock_time_s",
        ],
        "group_diagnostics": {
            "primary_scene_groups": [
                "urban",
                "mixed_urban_natural",
                "forest_canopy",
                "agricultural_open_field",
                "water_wetland",
                "low_structure",
                "other",
            ],
            "challenge_flags": [
                "rotation_challenge",
                "fov_scale_challenge",
                "shadow_heavy",
                "repetitive_structure",
                "seasonal_or_map_age_mismatch",
                "transition_zone",
            ],
            "annotation_rule": (
                "Scene labels are frozen before method results and used only for "
                "stratified evaluation/failure analysis, not online candidate selection."
            ),
        },
        "end_to_end": [
            "selected_absolute_hits_eval_only",
            "confidence_gated_accept_count",
            "correction_blackout_distance_m_eval_only",
            "post_lock_causal_rmse_m_eval_only",
            "post_lock_causal_p95_m_eval_only",
            "post_lock_causal_maximum_error_m_eval_only",
            "post_lock_failure_rate_eval_only",
        ],
        "primary_s7_target": (
            "Raise correct-region Recall@20 and Recall@50; then determine whether "
            "LightGlue verification can be reduced from Top-50 toward Top-20."
        ),
        "promotion_guardrails": [
            "Do not use evaluation labels in descriptor scoring, ranking, fusion, or gating.",
            "Do not promote visually attractive preprocessing without Recall@K improvement.",
            "Do not rerun S6 fusion after every retrieval ablation.",
            "Do not claim universal thresholds from traj01.",
        ],
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=False)
        handle.write("\n")


def write_method_registry(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("Method registry cannot be empty.")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(checks: list[CheckResult]) -> str:
    lines = [
        "| Check | Status | Observed | Expected | Detail |",
        "|---|---:|---:|---:|---|",
    ]
    for check in checks:
        values = [
            check.name,
            check.status,
            check.observed,
            check.expected,
            check.detail,
        ]
        escaped = [str(value).replace("|", "\\|").replace("\n", " ") for value in values]
        lines.append("| " + " | ".join(escaped) + " |")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="S7.0: validate and freeze the SatLoc traj01 S7 benchmark contract."
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--sequence", default="traj01")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument(
        "--s5c-query-manifest", type=Path, default=DEFAULT_S5C_QUERY_MANIFEST
    )
    parser.add_argument(
        "--sequence-manifest", type=Path, default=DEFAULT_SEQUENCE_MANIFEST
    )
    parser.add_argument("--uav-index", type=Path, default=DEFAULT_UAV_INDEX)
    parser.add_argument(
        "--satellite-index", type=Path, default=DEFAULT_SATELLITE_INDEX
    )
    parser.add_argument("--expected-frames", type=int, default=1034)
    parser.add_argument("--expected-queries", type=int, default=263)
    parser.add_argument("--expected-tiles", type=int, default=8625)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    if not repo_root.exists():
        raise FileNotFoundError(f"Repository root does not exist: {repo_root}")

    def resolve_from_repo(path: Path) -> Path:
        return path if path.is_absolute() else repo_root / path

    output_root = resolve_from_repo(args.output_root)
    metadata_dir = output_root / "metadata" / "s7_retrieval_upgrade"
    reports_dir = output_root / "reports" / "s7_retrieval_upgrade"
    figures_dir = output_root / "figures" / "s7_retrieval_upgrade"
    cache_dir = output_root / "cache" / "s7_retrieval_upgrade"
    assets_dir = repo_root / "docs" / "assets" / "s7_retrieval_upgrade"
    script_dir = repo_root / "scripts" / "satloc" / "s7"

    for directory in (
        metadata_dir,
        reports_dir,
        figures_dir,
        cache_dir,
        assets_dir,
        script_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    query_path = resolve_from_repo(args.s5c_query_manifest)
    sequence_path = resolve_from_repo(args.sequence_manifest)
    uav_index_path = resolve_from_repo(args.uav_index)
    satellite_index_path = resolve_from_repo(args.satellite_index)
    dataset_root = resolve_from_repo(args.dataset_root)
    s6_readme = find_first_existing(repo_root, S6_README_CANDIDATES)
    s7_plan = find_first_existing(repo_root, S7_PLAN_CANDIDATES)

    checks: list[CheckResult] = []

    required_paths = {
        "dataset_root": dataset_root,
        "s5c_query_manifest": query_path,
        "s6a_sequence_manifest": sequence_path,
        "uav_frames_index": uav_index_path,
        "satellite_tiles_index": satellite_index_path,
    }
    for name, path in required_paths.items():
        add_check(
            checks,
            f"exists:{name}",
            path.exists(),
            relative_or_absolute(path, repo_root),
            "existing path",
            "Required frozen input or dataset path.",
        )

    add_check(
        checks,
        "exists:s6_frozen_readme",
        s6_readme is not None,
        relative_or_absolute(s6_readme, repo_root) if s6_readme else "missing",
        "README_satloc_s6_relative_absolute_fusion.md",
        "Required frozen S6 handoff documentation.",
    )
    add_check(
        checks,
        "exists:s7_plan_readme",
        s7_plan is not None,
        relative_or_absolute(s7_plan, repo_root) if s7_plan else "missing",
        "README_satloc_s7_candidate_generation_and_frontend_upgrade_plan.md",
        "Required frozen S7 plan documentation.",
    )

    query_df: pd.DataFrame | None = None
    sequence_df: pd.DataFrame | None = None
    satellite_df: pd.DataFrame | None = None
    uav_df: pd.DataFrame | None = None

    if query_path.exists():
        query_df = read_csv(query_path, "S5C query manifest")
        add_check(
            checks,
            "s5c_query_count",
            len(query_df) == args.expected_queries,
            len(query_df),
            args.expected_queries,
            "Frozen temporally structured S5C query set.",
        )
        token_col = detect_unique_column(query_df, ["token0_id", "token"])
        add_check(
            checks,
            "s5c_query_token_column",
            token_col is not None,
            token_col or "missing",
            "token0_id",
            "token0_id is the canonical traj01 sequence token.",
        )
        if token_col is not None:
            duplicate_count = int(query_df[token_col].duplicated().sum())
            add_check(
                checks,
                "s5c_query_unique_tokens",
                duplicate_count == 0,
                duplicate_count,
                0,
                "No duplicate query tokens are allowed in the frozen benchmark.",
            )

    if sequence_path.exists():
        sequence_df = read_csv(sequence_path, "S6A sequence manifest")
        if "sequence" in sequence_df.columns:
            sequence_df = sequence_df[
                sequence_df["sequence"].astype(str) == args.sequence
            ].copy()
        add_check(
            checks,
            "traj01_sequence_frame_count",
            len(sequence_df) == args.expected_frames,
            len(sequence_df),
            args.expected_frames,
            "Correct sequence ordering is token0_id / sequence_frame_id.",
        )
        for column in ("token0_id", "sequence_frame_id"):
            add_check(
                checks,
                f"sequence_column:{column}",
                column in sequence_df.columns,
                "present" if column in sequence_df.columns else "missing",
                "present",
                "Required for deterministic frame ordering and joins.",
            )
        if "token0_id" in sequence_df.columns:
            duplicates = int(sequence_df["token0_id"].duplicated().sum())
            add_check(
                checks,
                "traj01_unique_token0_id",
                duplicates == 0,
                duplicates,
                0,
                "The earlier duplicated token ordering must not reappear.",
            )

    if satellite_index_path.exists():
        satellite_df = read_csv(satellite_index_path, "satellite tile index")
        add_check(
            checks,
            "satellite_tile_count",
            len(satellite_df) == args.expected_tiles,
            len(satellite_df),
            args.expected_tiles,
            "Full-map retrieval must use the frozen 8,625-tile database.",
        )
        sat_id_col = detect_unique_column(
            satellite_df,
            [
                "tile_id",
                "satellite_tile_id",
                "tile_filename",
                "filename",
                "image_path",
                "path",
            ],
        )
        if sat_id_col is not None:
            duplicates = int(satellite_df[sat_id_col].astype(str).duplicated().sum())
            add_check(
                checks,
                "satellite_tile_identity_unique",
                duplicates == 0,
                duplicates,
                0,
                f"Checked identity column {sat_id_col!r}.",
            )
        else:
            add_check(
                checks,
                "satellite_tile_identity_column",
                False,
                "not detected",
                "one known tile identity column",
                "Add the actual identity column to the detector without changing data.",
            )

    if uav_index_path.exists():
        uav_df = read_csv(uav_index_path, "UAV frame index")
        if "sequence" in uav_df.columns:
            traj_uav_df = uav_df[uav_df["sequence"].astype(str) == args.sequence]
            add_check(
                checks,
                "uav_index_traj01_count",
                len(traj_uav_df) == args.expected_frames,
                len(traj_uav_df),
                args.expected_frames,
                "Cross-check against the S6A sequence manifest.",
            )
        else:
            add_check(
                checks,
                "uav_index_sequence_column",
                False,
                "missing",
                "sequence",
                "Needed to cross-check traj01 frame count.",
            )

    if query_df is not None and sequence_df is not None:
        q_token = detect_unique_column(query_df, ["token0_id", "token"])
        if q_token is not None and "token0_id" in sequence_df.columns:
            query_tokens = set(query_df[q_token].astype(str))
            sequence_tokens = set(sequence_df["token0_id"].astype(str))
            missing_tokens = sorted(query_tokens - sequence_tokens)
            add_check(
                checks,
                "all_queries_exist_in_traj01",
                not missing_tokens,
                len(missing_tokens),
                0,
                (
                    "All frozen S5C queries must map to traj01. "
                    f"First missing: {missing_tokens[:5]}"
                ),
            )

    # Freeze a deterministic S7 copy of the query manifest even when other checks fail.
    frozen_query_path = metadata_dir / "s7_0_query_manifest.csv"
    if query_df is not None:
        frozen_query_df = query_df.copy()
        if "sequence_frame_id" in frozen_query_df.columns:
            frozen_query_df = frozen_query_df.sort_values(
                "sequence_frame_id", kind="mergesort"
            ).reset_index(drop=True)
        else:
            frozen_query_df = frozen_query_df.reset_index(drop=True)
        if "s7_query_index" in frozen_query_df.columns:
            frozen_query_df = frozen_query_df.drop(columns=["s7_query_index"])
        frozen_query_df.insert(0, "s7_query_index", range(len(frozen_query_df)))
        frozen_query_df["s7_benchmark_frozen"] = True
        frozen_query_df.to_csv(frozen_query_path, index=False)

    method_registry_path = metadata_dir / "s7_0_method_registry.csv"
    write_method_registry(method_registry_path, method_registry_rows())

    metric_spec_path = metadata_dir / "s7_0_metric_specification.json"
    write_json(metric_spec_path, metric_specification())

    packages = {
        name: package_version(name)
        for name in [
            "numpy",
            "pandas",
            "opencv-python",
            "opencv-contrib-python",
            "scikit-learn",
            "scipy",
            "matplotlib",
            "torch",
            "torchvision",
            "kornia",
            "lightglue",
        ]
    }
    environment = {
        "generated_at_utc": now_utc_iso(),
        "repo_root": str(repo_root),
        "python": {
            "version": sys.version,
            "executable": sys.executable,
            "implementation": platform.python_implementation(),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
        },
        "torch": collect_torch_environment(),
        "packages": packages,
        "git": {
            "commit": run_git(repo_root, ["rev-parse", "HEAD"]),
            "branch": run_git(repo_root, ["branch", "--show-current"]),
            "is_dirty": bool(run_git(repo_root, ["status", "--porcelain"])),
            "remote_origin": run_git(repo_root, ["remote", "get-url", "origin"]),
        },
    }
    environment_path = metadata_dir / "s7_0_environment.json"
    write_json(environment_path, environment)

    failed_checks = [check for check in checks if check.status == "FAIL"]
    overall_status = "PASS" if not failed_checks else "BLOCKED"

    input_hashes: dict[str, dict[str, Any]] = {}
    for name, path in {
        **required_paths,
        "s6_frozen_readme": s6_readme,
        "s7_plan_readme": s7_plan,
    }.items():
        if path is not None and path.is_file():
            input_hashes[name] = {
                "path": relative_or_absolute(path, repo_root),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        elif path is not None:
            input_hashes[name] = {
                "path": relative_or_absolute(path, repo_root),
                "sha256": None,
                "bytes": None,
            }
        else:
            input_hashes[name] = {"path": None, "sha256": None, "bytes": None}

    stage_manifest = {
        "stage": "S7.0",
        "name": "Stage Preflight and Frozen Benchmark Specification",
        "version": "s7.0-v1",
        "generated_at_utc": now_utc_iso(),
        "status": overall_status,
        "sequence": args.sequence,
        "frozen_counts": {
            "traj01_frames": args.expected_frames,
            "temporal_queries": args.expected_queries,
            "satellite_tiles": args.expected_tiles,
            "correct_region_threshold_m_eval_only": 40.0,
        },
        "primary_target": (
            "Improve correct-region Recall@20 and Recall@50, then evaluate "
            "whether LightGlue can move from Top-50 toward Top-20."
        ),
        "online_usable_evidence": EXPECTED_ONLINE_FIELDS,
        "evaluation_only_fields": EVALUATION_ONLY_FIELDS,
        "directories": {
            "scripts": relative_or_absolute(script_dir, repo_root),
            "metadata": relative_or_absolute(metadata_dir, repo_root),
            "reports": relative_or_absolute(reports_dir, repo_root),
            "figures": relative_or_absolute(figures_dir, repo_root),
            "cache": relative_or_absolute(cache_dir, repo_root),
            "docs_assets": relative_or_absolute(assets_dir, repo_root),
        },
        "inputs": input_hashes,
        "outputs": {
            "stage_manifest": relative_or_absolute(
                metadata_dir / "s7_0_stage_manifest.json", repo_root
            ),
            "query_manifest": relative_or_absolute(frozen_query_path, repo_root),
            "method_registry": relative_or_absolute(method_registry_path, repo_root),
            "environment": relative_or_absolute(environment_path, repo_root),
            "metric_specification": relative_or_absolute(metric_spec_path, repo_root),
            "preflight_report": relative_or_absolute(
                reports_dir / "s7_0_preflight_report.md", repo_root
            ),
        },
        "checks": [asdict(check) for check in checks],
        "failed_check_count": len(failed_checks),
        "manual_annotation_status": "not_started_expected_after_s7_0a_pass",
    }
    stage_manifest_path = metadata_dir / "s7_0_stage_manifest.json"
    write_json(stage_manifest_path, stage_manifest)

    report_path = reports_dir / "s7_0_preflight_report.md"
    report_lines = [
        "# S7.0 Preflight Report",
        "",
        f"**Status:** `{overall_status}`  ",
        f"**Generated:** `{stage_manifest['generated_at_utc']}`  ",
        f"**Sequence:** `{args.sequence}`",
        "",
        "## Frozen benchmark",
        "",
        f"- Trajectory frames: **{args.expected_frames}**",
        f"- Temporal queries: **{args.expected_queries}**",
        f"- Satellite tiles: **{args.expected_tiles}**",
        "- Correct-region threshold: **40 m, evaluation only**",
        "- Primary target: improve Recall@20 and Recall@50 before reducing LightGlue Top-K.",
        "",
        "## Validation checks",
        "",
        markdown_table(checks),
        "",
        "## Ground-truth isolation",
        "",
        "Online retrieval, ranking, fusion, acceptance, and correction may use image/model/geometric evidence only.",
        "Reference coordinates, oracle identities/ranks/errors, hit labels, and trajectory errors remain evaluation-only.",
        "",
        "## Generated artifacts",
        "",
        f"- `{relative_or_absolute(stage_manifest_path, repo_root)}`",
        f"- `{relative_or_absolute(frozen_query_path, repo_root)}`",
        f"- `{relative_or_absolute(method_registry_path, repo_root)}`",
        f"- `{relative_or_absolute(environment_path, repo_root)}`",
        f"- `{relative_or_absolute(metric_spec_path, repo_root)}`",
        f"- `{relative_or_absolute(report_path, repo_root)}`",
        "",
        "## Manual work",
        "",
        "No manual frame labelling is required in S7.0A.",
        "After this preflight passes, S7.0B will generate coarse contact sheets; manual work will be limited to marking continuous scene ranges and transition boundaries.",
        "",
    ]
    if failed_checks:
        report_lines.extend(
            [
                "## Blocking items",
                "",
                *[
                    f"- **{check.name}:** observed `{check.observed}`, expected `{check.expected}` — {check.detail}"
                    for check in failed_checks
                ],
                "",
                "Do not begin XFeat, DINOv2, VLAD, PHLO, or scene annotation until these checks pass.",
                "",
            ]
        )
    else:
        report_lines.extend(
            [
                "## Decision",
                "",
                "S7.0A passed. Proceed to S7.0B scene-taxonomy freeze and contact-sheet generation.",
                "",
            ]
        )

    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    print("S7.0 Stage Preflight")
    print("----------------------")
    print(f"Status:            {overall_status}")
    print(f"Checks:            {len(checks)}")
    print(f"Failed checks:     {len(failed_checks)}")
    print(f"Expected frames:   {args.expected_frames}")
    print(f"Expected queries:  {args.expected_queries}")
    print(f"Expected tiles:    {args.expected_tiles}")
    print(f"Metadata:          {relative_or_absolute(metadata_dir, repo_root)}")
    print(f"Report:            {relative_or_absolute(report_path, repo_root)}")
    if failed_checks:
        print("\nBlocking checks:")
        for check in failed_checks:
            print(
                f"- {check.name}: observed={check.observed!r}, "
                f"expected={check.expected!r}"
            )
        return 2

    print("\nS7.0A passed. No manual image annotation is required yet.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
