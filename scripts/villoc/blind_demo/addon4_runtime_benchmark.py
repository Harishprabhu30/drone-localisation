'''
ROOT=outputs/villoc/traj01_90deg_stable120m
MAPROOT=outputs/villoc/90_deg
RUN="$ROOT/blind_demo_addons/eval_traj01_known_gt"
TAG=dinov2_vits14_img518_center_square_avgpatch_cpu

python scripts/villoc/blind_demo/addon4_runtime_benchmark.py \
  --root "$ROOT" \
  --map-root "$MAPROOT" \
  --run-root "$RUN" \
  --variant 512_s256 \
  --tag "$TAG" \
  2>&1 | tee \
  "$RUN/logs/addon4_runtime_benchmark.log"
'''

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def safe_float(value: Any) -> float | None:
    if value is None:
        return None

    value = float(value)

    if not math.isfinite(value):
        return None

    return value


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)

    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def load_npz_meta(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)

    data = np.load(
        path,
        allow_pickle=False,
    )

    return json.loads(
        str(data["meta_json"])
    )


def timing_row(
    *,
    stage: str,
    component: str,
    cost_scope: str,
    runtime_s: float | None,
    work_count: int | None,
    work_unit: str | None,
    source_path: Path | None,
    measurement_status: str = "MEASURED",
    secondary_count: int | None = None,
    secondary_unit: str | None = None,
    cache_state: str | None = None,
    notes: str = "",
) -> dict[str, Any]:
    runtime_s = safe_float(runtime_s)

    ms_per_work_item = None
    work_items_per_s = None

    if (
        runtime_s is not None
        and work_count is not None
        and work_count > 0
    ):
        ms_per_work_item = (
            1000.0
            * runtime_s
            / work_count
        )

        if runtime_s > 0:
            work_items_per_s = (
                work_count
                / runtime_s
            )

    ms_per_secondary_item = None

    if (
        runtime_s is not None
        and secondary_count is not None
        and secondary_count > 0
    ):
        ms_per_secondary_item = (
            1000.0
            * runtime_s
            / secondary_count
        )

    # Explicit normalized fields requested by the
    # blind-demo add-on contract.
    ms_per_frame = None
    ms_per_pair = None
    ms_per_query = None
    ms_per_query_candidate_pair = None

    if work_unit in {
        "frame",
        "query_image",
        "query_frame",
    }:
        ms_per_frame = ms_per_work_item

    if work_unit in {
        "query",
        "query_image",
        "query_frame",
    }:
        ms_per_query = ms_per_work_item

    if work_unit == "frame_pair":
        ms_per_pair = ms_per_work_item

    if work_unit == "query_candidate_pair":
        ms_per_query_candidate_pair = (
            ms_per_work_item
        )

    if secondary_unit == "query":
        ms_per_query = (
            ms_per_secondary_item
        )

    return {
        "stage": stage,
        "component": component,
        "cost_scope": cost_scope,
        "measurement_status": measurement_status,
        "runtime_s": runtime_s,
        "work_count": work_count,
        "work_unit": work_unit,
        "ms_per_work_item": (
            ms_per_work_item
        ),
        "work_items_per_s": (
            work_items_per_s
        ),
        "secondary_count": (
            secondary_count
        ),
        "secondary_unit": (
            secondary_unit
        ),
        "ms_per_secondary_item": (
            ms_per_secondary_item
        ),
        "ms_per_frame": ms_per_frame,
        "ms_per_pair": ms_per_pair,
        "ms_per_query": ms_per_query,
        "ms_per_query_candidate_pair": (
            ms_per_query_candidate_pair
        ),
        "estimated_hz_fps": (
            work_items_per_s
        ),
        "estimated_rate_unit": (
            f"{work_unit}_per_s"
            if work_unit is not None
            else None
        ),
        "cache_state": cache_state,
        "source_path": (
            str(source_path.resolve())
            if source_path is not None
            else None
        ),
        "notes": notes,
    }


def save_runtime_plot(
    table: pd.DataFrame,
    output_path: Path,
) -> None:
    measured = table.loc[
        (table["measurement_status"] == "MEASURED")
        & pd.to_numeric(
            table["runtime_s"],
            errors="coerce",
        ).notna()
    ].copy()

    # XFeat feature/matching rows are component
    # breakdowns of relative_odometry, so exclude
    # them from the main stage-level comparison to
    # avoid double-counting visually.
    measured = measured.loc[
        ~measured["stage"].isin(
            [
                "relative_odometry_features",
                "relative_odometry_matching",
            ]
        )
    ].copy()

    measured["runtime_s"] = pd.to_numeric(
        measured["runtime_s"],
        errors="raise",
    )

    measured = measured.sort_values(
        "runtime_s",
        ascending=True,
        kind="mergesort",
    )

    if measured.empty:
        raise RuntimeError(
            "No measured runtime rows available "
            "for runtime_by_stage.png."
        )

    labels = [
        f"{component}\n[{scope}]"
        for component, scope in zip(
            measured["component"],
            measured["cost_scope"],
        )
    ]

    fig_height = max(
        6.0,
        0.72 * len(measured),
    )

    fig, ax = plt.subplots(
        figsize=(12, fig_height)
    )

    ax.barh(
        labels,
        measured["runtime_s"],
    )

    # Runtime spans from milliseconds/sub-second
    # to several hours, therefore logarithmic scale
    # is substantially more readable.
    ax.set_xscale("log")

    ax.set_xlabel(
        "Measured runtime [s] — logarithmic scale"
    )

    ax.set_ylabel(
        "Pipeline stage"
    )

    ax.set_title(
        "Villoc traj01 runtime by stage"
    )

    ax.grid(
        True,
        axis="x",
        alpha=0.25,
    )

    for i, value in enumerate(
        measured["runtime_s"].to_numpy(
            dtype=float
        )
    ):
        ax.text(
            value,
            i,
            f"  {value:.3g} s",
            va="center",
            fontsize=8,
        )

    fig.text(
        0.01,
        0.01,
        (
            "Map descriptor creation is offline/reusable. "
            "LightGlue is diagnostic-only. "
            "DINO retrieval assumes descriptor caches already exist."
        ),
        fontsize=8,
    )

    fig.tight_layout(
        rect=(0, 0.04, 1, 1)
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)


def pending_row(
    stage: str,
    component: str,
    cost_scope: str,
    notes: str,
) -> dict[str, Any]:
    return timing_row(
        stage=stage,
        component=component,
        cost_scope=cost_scope,
        runtime_s=None,
        work_count=None,
        work_unit=None,
        source_path=None,
        measurement_status="PENDING",
        notes=notes,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Add-on 4 canonical runtime benchmark "
            "registry for Villoc traj01."
        )
    )

    parser.add_argument(
        "--root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--map-root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--run-root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--tag",
        default=(
            "dinov2_vits14_img518_"
            "center_square_avgpatch_cpu"
        ),
    )

    parser.add_argument(
        "--variant",
        default="512_s256",
    )

    args = parser.parse_args()

    root = args.root.resolve()
    map_root = args.map_root.resolve()
    run_root = args.run_root.resolve()

    metrics_dir = (
        run_root / "metrics"
    )

    figures_dir = (
        run_root / "figures"
    )

    metrics_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    figures_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    tag = args.tag
    variant = args.variant

    # ---------------------------------------------------------
    # Exact trusted source paths
    # ---------------------------------------------------------

    xfeat_report_path = (
        root
        / "reports/s8_xfeat_relative_frontend/"
        "s8r4_xfeat_relative_frontend_report.json"
    )

    query_cache_path = (
        root
        / "descriptors"
        / (
            "s8_11c_dinov2_queries_v_1fps_"
            f"{tag}.npz"
        )
    )

    map_cache_path = (
        map_root
        / "descriptors"
        / (
            f"s8_11b_dinov2_map_{variant}_"
            f"{tag}.npz"
        )
    )

    dino_retrieval_report_path = (
        root
        / "reports/s8_11d"
        / (
            "s8_11d_independent_retrieval_"
            f"summary_{tag}.json"
        )
    )

    orb_report_path = (
        run_root
        / "benchmarks/orb_fullrun"
        / "s8_12e1_summary.json"
    )

    lightglue_report_path = (
        root
        / "reports/s8_12e1b_lightglue_top20"
        / (
            f"{variant}_lg_hybrid_top20_"
            "img518_full403"
        )
        / "s8_12e1b_summary.json"
    )

    f1_report_path = (
        root
        / "reports/s8_relative_absolute_fusion/"
        "s8_f1_absolute_correction_manifest_report.json"
    )

    f3_report_path = (
        root
        / "reports/s8_relative_absolute_fusion/"
        "s8_f3_temporal_agreement_gating_report.json"
    )

    f3b_report_path = (
        root
        / "reports/s8_relative_absolute_fusion/"
        "s8_f3b_temporal_policy_fusion_replay_report.json"
    )

    # ---------------------------------------------------------
    # Load trusted measurements
    # ---------------------------------------------------------

    xfeat = load_json(
        xfeat_report_path
    )

    query_meta = load_npz_meta(
        query_cache_path
    )

    map_meta = load_npz_meta(
        map_cache_path
    )

    dino = load_json(
        dino_retrieval_report_path
    )

    orb = load_json(
        orb_report_path
    )

    lightglue = load_json(
        lightglue_report_path
    )

    f1 = load_json(
        f1_report_path
    )

    f3 = load_json(
        f3_report_path
    )

    f3b = load_json(
        f3b_report_path
    )

    rows: list[dict[str, Any]] = []

    # ---------------------------------------------------------
    # XFeat
    # ---------------------------------------------------------

    xfeat_pair = xfeat[
        "pair_summary"
    ]

    xfeat_pairs = int(
        xfeat.get(
            "pairs",
            xfeat_pair.get(
                "pair_count",
                402,
            ),
        )
    )

    rows.append(
        timing_row(
            stage="relative_odometry",
            component="XFeat relative frontend",
            cost_scope="per_new_flight",
            runtime_s=(
                xfeat_pair[
                    "wall_seconds"
                ]
            ),
            work_count=xfeat_pairs,
            work_unit="frame_pair",
            source_path=(
                xfeat_report_path
            ),
            cache_state=(
                "XFeat frame features generated "
                "during this measured run"
            ),
            notes=(
                "Full relative-frontend wall time. "
                "Includes feature extraction, matching/"
                "RANSAC and supporting stage overhead."
            ),
        )
    )

    # Optional XFeat substage details.
    rows.append(
        timing_row(
            stage="relative_odometry_features",
            component="XFeat feature extraction",
            cost_scope="per_new_flight",
            runtime_s=(
                xfeat_pair[
                    "feature_seconds"
                ]
            ),
            work_count=403,
            work_unit="frame",
            source_path=(
                xfeat_report_path
            ),
            notes=(
                "Recorded XFeat feature extraction "
                "substage."
            ),
        )
    )

    rows.append(
        timing_row(
            stage="relative_odometry_matching",
            component="XFeat matching + RANSAC",
            cost_scope="per_new_flight",
            runtime_s=(
                xfeat_pair[
                    "matching_ransac_seconds"
                ]
            ),
            work_count=xfeat_pairs,
            work_unit="frame_pair",
            source_path=(
                xfeat_report_path
            ),
            notes=(
                "Recorded XFeat matching/RANSAC "
                "substage."
            ),
        )
    )

    # ---------------------------------------------------------
    # DINO query descriptor extraction
    # ---------------------------------------------------------

    query_count = int(
        query_meta["row_count"]
    )

    rows.append(
        timing_row(
            stage=(
                "dino_query_descriptor_extraction"
            ),
            component=(
                "DINOv2 query descriptor encoding"
            ),
            cost_scope="per_new_flight",
            runtime_s=(
                query_meta["runtime_s"]
            ),
            work_count=query_count,
            work_unit="query_image",
            source_path=query_cache_path,
            cache_state=(
                "descriptor cache creation"
            ),
            notes=(
                "CPU DINOv2 ViT-S/14 img518 "
                "query-image encoding."
            ),
        )
    )

    # ---------------------------------------------------------
    # Offline DINO map cache
    # ---------------------------------------------------------

    map_tile_count = int(
        map_meta["row_count"]
    )

    rows.append(
        timing_row(
            stage=(
                "dino_map_descriptor_cache_creation"
            ),
            component=(
                "DINOv2 map descriptor encoding"
            ),
            cost_scope="offline_map_preparation",
            runtime_s=(
                map_meta["runtime_s"]
            ),
            work_count=map_tile_count,
            work_unit="map_tile",
            source_path=map_cache_path,
            cache_state=(
                "one-time map descriptor cache build"
            ),
            notes=(
                "Reusable across flights while the "
                "same AOI/map/tile configuration is used."
            ),
        )
    )

    # ---------------------------------------------------------
    # DINO retrieval from cached descriptors
    # ---------------------------------------------------------

    variant_report = (
        dino["variants"][variant]
    )

    dino_queries = int(
        variant_report["query_count"]
    )

    rows.append(
        timing_row(
            stage=(
                "dino_retrieval_against_map_cache"
            ),
            component=(
                "DINO cosine ranking"
            ),
            cost_scope="per_new_flight",
            runtime_s=(
                variant_report["runtime_s"]
            ),
            work_count=dino_queries,
            work_unit="query",
            source_path=(
                dino_retrieval_report_path
            ),
            cache_state=(
                "query and map descriptors already cached"
            ),
            notes=(
                "Descriptor ranking only. Must not "
                "be interpreted as image-encoding time."
            ),
        )
    )

    # ---------------------------------------------------------
    # ORB full 403 x Top-20 benchmark
    # ---------------------------------------------------------

    orb_runtime = orb[
        "runtime"
    ]

    orb_pairs = int(
        orb_runtime[
            "query_candidate_pairs"
        ]
    )

    orb_queries = int(
        orb_runtime["queries"]
    )

    rows.append(
        timing_row(
            stage="orb_topk_reranking",
            component="ORB Top-20 verifier/reranker",
            cost_scope="per_new_flight",
            runtime_s=(
                orb_runtime[
                    "verifier_rerank_core_s"
                ]
            ),
            work_count=orb_pairs,
            work_unit=(
                "query_candidate_pair"
            ),
            secondary_count=(
                orb_queries
            ),
            secondary_unit="query",
            source_path=orb_report_path,
            cache_state=(
                "feature cache internal to one ORB run"
            ),
            notes=(
                "Measured full 403-query run using "
                "the promoted 512_s256 Top-20 setup."
            ),
        )
    )

    # ---------------------------------------------------------
    # LightGlue diagnostic full403
    # ---------------------------------------------------------

    lg_pairs = int(
        lightglue["pair_count"]
    )

    lg_queries = int(
        lightglue["query_count"]
    )

    rows.append(
        timing_row(
            stage="lightglue_reranking",
            component=(
                "LightGlue Top-20 verifier/reranker"
            ),
            cost_scope="diagnostic_optional",
            runtime_s=(
                lightglue[
                    "total_runtime_s"
                ]
            ),
            work_count=lg_pairs,
            work_unit=(
                "query_candidate_pair"
            ),
            secondary_count=(
                lg_queries
            ),
            secondary_unit="query",
            source_path=(
                lightglue_report_path
            ),
            cache_state=(
                "feature cache internal to one "
                "LightGlue run"
            ),
            notes=(
                "Full403 diagnostic only; not "
                "promoted into final fusion."
            ),
        )
    )

    # ---------------------------------------------------------
    # F1 / F3 / F3B newly instrumented originals
    # ---------------------------------------------------------

    for (
        stage,
        component,
        report,
        path,
    ) in [
        (
            "correction_manifest_build",
            "F1 correction manifest",
            f1,
            f1_report_path,
        ),
        (
            "temporal_gating",
            "F3 temporal agreement gating",
            f3,
            f3_report_path,
        ),
        (
            "fusion_replay",
            "F3B temporal fusion replay",
            f3b,
            f3b_report_path,
        ),
    ]:
        runtime = report["runtime"]

        rows.append(
            timing_row(
                stage=stage,
                component=component,
                cost_scope="per_new_flight",
                runtime_s=(
                    runtime[
                        "core_compute_s"
                    ]
                ),
                work_count=int(
                    runtime[
                        "normalization_count"
                    ]
                ),
                work_unit=(
                    runtime[
                        "normalization_unit"
                    ]
                ),
                source_path=path,
                notes=(
                    "Core compute only. Output writing "
                    "and plotting are recorded separately "
                    "inside the stage report."
                ),
            )
        )

    # ---------------------------------------------------------
    # Remaining README-required stages.
    # They are intentionally not guessed.
    # ---------------------------------------------------------

    rows.extend(
        [
            pending_row(
                "environment_check",
                "Environment preflight",
                "per_new_flight",
                "Implemented later as Add-on 12.",
            ),
            pending_row(
                "video_metadata_read",
                "Video metadata read",
                "per_new_flight",
                (
                    "Stage 5D will resolve/instrument "
                    "the existing extraction stage."
                ),
            ),
            pending_row(
                "frame_extraction",
                "Video frame extraction",
                "per_new_flight",
                (
                    "Stage 5D will resolve/instrument "
                    "the existing extraction stage."
                ),
            ),
            pending_row(
                "query_manifest_build",
                "Query manifest construction",
                "per_new_flight",
                (
                    "Current manifest is reference-aware; "
                    "blind builder arrives in Add-on 7."
                ),
            ),
            pending_row(
                "srt_reference_parsing",
                "Optional SRT/reference parsing",
                "evaluation_optional",
                (
                    "Evaluation-only stage; timing still "
                    "to be instrumented."
                ),
            ),
            pending_row(
                "estimated_latlon_export",
                "Estimated lat/lon export",
                "per_new_flight",
                "Implemented later as Add-on 9.",
            ),
            pending_row(
                "plot_generation",
                "No-reference plot generation",
                "supporting_output",
                "Implemented later as Add-on 10.",
            ),
            pending_row(
                "folium_html_map_generation",
                "Folium HTML map generation",
                "supporting_output",
                "Implemented later as Add-on 10.",
            ),
            pending_row(
                "run_summary_generation",
                "Run summary generation",
                "supporting_output",
                "Implemented later as Add-on 11.",
            ),
        ]
    )

    # ADDON7_RUNTIME_INGEST_BEGIN
    #
    # Add-on 7 provides genuine reference-free timings
    # for video metadata inspection, frame extraction,
    # and blind query-manifest construction.
    #
    # These replace the original PENDING placeholders.
    addon7_report_path = (
        run_root
        / "reports"
        / "blind_query_manifest_report.json"
    )

    if addon7_report_path.exists():
        addon7_report = json.loads(
            addon7_report_path.read_text(
                encoding="utf-8"
            )
        )

        addon7_runtime = (
            addon7_report.get(
                "runtime",
                {},
            )
        )

        addon7_sampling = (
            addon7_report.get(
                "sampling",
                {},
            )
        )

        addon7_count = int(
            addon7_sampling.get(
                "sample_count",
                0,
            )
        )

        addon7_specs = {
            "video_metadata_read": {
                "runtime_s": addon7_runtime.get(
                    "video_metadata_read_s"
                ),
                "work_count": 1,
                "work_unit": "video",
                "notes": (
                    "Measured by Add-on 7 from the raw "
                    "video without SRT/GPS/reference."
                ),
            },
            "frame_extraction": {
                "runtime_s": addon7_runtime.get(
                    "frame_extraction_s"
                ),
                "work_count": addon7_count,
                "work_unit": "frame",
                "notes": (
                    "Reference-free 1-fps frame extraction "
                    "measured by Add-on 7."
                ),
            },
            "query_manifest_build": {
                "runtime_s": addon7_runtime.get(
                    "query_manifest_build_s"
                ),
                "work_count": addon7_count,
                "work_unit": "query_frame",
                "notes": (
                    "Reference-free blind query manifest "
                    "construction measured by Add-on 7."
                ),
            },
        }

        found_addon7_stages = set()

        for row in rows:
            stage = row.get(
                "stage"
            )

            if stage not in addon7_specs:
                continue

            spec = addon7_specs[
                stage
            ]

            runtime_s = spec[
                "runtime_s"
            ]

            work_count = spec[
                "work_count"
            ]

            work_unit = spec[
                "work_unit"
            ]

            if runtime_s is None:
                continue

            runtime_s = float(
                runtime_s
            )

            work_count = int(
                work_count
            )

            ms_per_item = None
            items_per_s = None

            if (
                work_count > 0
                and runtime_s >= 0
            ):
                ms_per_item = (
                    1000.0
                    * runtime_s
                    / work_count
                )

                if runtime_s > 0:
                    items_per_s = (
                        work_count
                        / runtime_s
                    )

            row.update(
                {
                    "cost_scope": (
                        "per_new_flight"
                    ),
                    "measurement_status": (
                        "MEASURED"
                    ),
                    "runtime_s": (
                        runtime_s
                    ),
                    "work_count": (
                        work_count
                    ),
                    "work_unit": (
                        work_unit
                    ),
                    "ms_per_work_item": (
                        ms_per_item
                    ),
                    "work_items_per_s": (
                        items_per_s
                    ),
                    "secondary_count": None,
                    "secondary_unit": None,
                    "ms_per_secondary_item": None,
                    "ms_per_frame": (
                        ms_per_item
                        if work_unit
                        in {
                            "frame",
                            "query_frame",
                        }
                        else None
                    ),
                    "ms_per_pair": None,
                    "ms_per_query": (
                        ms_per_item
                        if work_unit
                        == "query_frame"
                        else None
                    ),
                    "ms_per_query_candidate_pair": None,
                    "estimated_hz_fps": (
                        items_per_s
                    ),
                    "estimated_rate_unit": (
                        f"{work_unit}_per_s"
                    ),
                    "cache_state": (
                        "not_applicable"
                    ),
                    "source_path": str(
                        addon7_report_path
                    ),
                    "notes": spec[
                        "notes"
                    ],
                }
            )

            found_addon7_stages.add(
                stage
            )

        expected_addon7_stages = set(
            addon7_specs
        )

        missing_addon7_rows = sorted(
            expected_addon7_stages
            - found_addon7_stages
        )

        if missing_addon7_rows:
            raise RuntimeError(
                "Add-on 4 has no placeholder rows "
                "for Add-on 7 timings: "
                f"{missing_addon7_rows}"
            )

    # ADDON7_RUNTIME_INGEST_END

    # ADDON8_SRT_RUNTIME_INGEST_BEGIN
    #
    # Raw SRT parsing is optional and evaluation-only.
    # If the Stage 9C report exists, this row MUST be
    # promoted from PENDING to MEASURED.
    addon8_srt_report_path = (
        run_root
        / "evaluation"
        / "reference_srt_parse_report.json"
    )

    if addon8_srt_report_path.exists():
        addon8_srt_report = json.loads(
            addon8_srt_report_path.read_text(
                encoding="utf-8"
            )
        )

        if (
            addon8_srt_report.get("status")
            != "PASS_OPTIONAL_REFERENCE_SRT_PARSE"
        ):
            raise RuntimeError(
                "Unexpected Add-on 8 SRT report "
                "status: "
                f"{addon8_srt_report.get('status')}"
            )

        addon8_runtime = (
            addon8_srt_report.get(
                "runtime",
                {}
            )
        )

        if (
            "srt_reference_parsing_s"
            not in addon8_runtime
        ):
            raise RuntimeError(
                "Add-on 8 SRT report exists but "
                "has no srt_reference_parsing_s."
            )

        runtime_s = float(
            addon8_runtime[
                "srt_reference_parsing_s"
            ]
        )

        row_count = int(
            addon8_srt_report.get(
                "rows",
                0,
            )
        )

        if row_count <= 0:
            raise RuntimeError(
                "Add-on 8 SRT report has "
                "invalid row count."
            )

        found = False

        for row in rows:
            if (
                row.get("stage")
                != "srt_reference_parsing"
            ):
                continue

            found = True

            ms_per_row = (
                1000.0
                * runtime_s
                / row_count
            )

            rows_per_s = (
                row_count
                / runtime_s
                if runtime_s > 0
                else None
            )

            row.update(
                {
                    "cost_scope": (
                        "evaluation_only"
                    ),
                    "measurement_status": (
                        "MEASURED"
                    ),
                    "runtime_s": runtime_s,
                    "work_count": row_count,
                    "work_unit": "srt_row",
                    "ms_per_work_item": (
                        ms_per_row
                    ),
                    "work_items_per_s": (
                        rows_per_s
                    ),
                    "secondary_count": None,
                    "secondary_unit": None,
                    "ms_per_secondary_item": None,
                    "ms_per_frame": None,
                    "ms_per_pair": None,
                    "ms_per_query": None,
                    "ms_per_query_candidate_pair": None,
                    "estimated_hz_fps": (
                        rows_per_s
                    ),
                    "estimated_rate_unit": (
                        "srt_row_per_s"
                    ),
                    "cache_state": (
                        "not_applicable"
                    ),
                    "source_path": str(
                        addon8_srt_report_path
                    ),
                    "notes": (
                        "Optional post-run raw SRT "
                        "reference parsing. Evaluation "
                        "only; unavailable to localization."
                    ),
                }
            )

            break

        if not found:
            raise RuntimeError(
                "Timing registry has no "
                "srt_reference_parsing row."
            )

        print(
            "[ADDON8] SRT runtime ingested:",
            f"{runtime_s:.6f}s",
            f"rows={row_count}",
        )

    else:
        print(
            "[ADDON8] SRT report not found; "
            "srt_reference_parsing remains pending:",
            addon8_srt_report_path,
        )

    # ADDON8_SRT_RUNTIME_INGEST_END

    # =========================================================
    # ADDON9_ADDON10_RUNTIME_INGEST_BEGIN
    #
    # Add-on 9 and Add-on 10 now provide measured runtime
    # evidence for three previously-PENDING README stages:
    #
    #   estimated_latlon_export
    #   plot_generation
    #   folium_html_map_generation
    #
    # The reports are trusted only if their stage status is
    # PASS. No runtime is guessed when a report is absent.
    # =========================================================

    def replace_pending_runtime_row(
        *,
        stage: str,
        component: str,
        cost_scope: str,
        runtime_s: float,
        work_count: int,
        work_unit: str,
        source_path: Path,
        cache_state: str,
        notes: str,
    ) -> None:

        matches = [
            i
            for i, row in enumerate(rows)
            if row.get("stage") == stage
        ]

        if len(matches) != 1:
            raise RuntimeError(
                "Expected exactly one runtime registry "
                f"row for {stage}, found {len(matches)}."
            )

        index = matches[0]

        if runtime_s <= 0:
            raise RuntimeError(
                f"{stage} runtime must be positive, "
                f"got {runtime_s}."
            )

        if work_count <= 0:
            raise RuntimeError(
                f"{stage} work_count must be positive, "
                f"got {work_count}."
            )

        rows[index] = timing_row(
            stage=stage,
            component=component,
            cost_scope=cost_scope,
            runtime_s=runtime_s,
            work_count=work_count,
            work_unit=work_unit,
            source_path=source_path,
            measurement_status="MEASURED",
            cache_state=cache_state,
            notes=notes,
        )

    # ---------------------------------------------------------
    # Add-on 9 — estimated lat/lon export
    # ---------------------------------------------------------

    addon9_report_path = (
        run_root
        / "reports"
        / "addon9_estimated_latlon"
        / "estimated_latlon_export_report.json"
    )

    if addon9_report_path.exists():

        addon9_report = json.loads(
            addon9_report_path.read_text(
                encoding="utf-8"
            )
        )

        if (
            addon9_report.get("status")
            != "PASS_ADDON9_ESTIMATED_LATLON_EXPORT"
        ):
            raise RuntimeError(
                "Unexpected Add-on 9 report status: "
                f"{addon9_report.get('status')}"
            )

        addon9_timing = addon9_report.get(
            "timing",
            addon9_report.get(
                "runtime",
                {},
            ),
        )

        if (
            "total_stage_wall_s"
            not in addon9_timing
        ):
            raise RuntimeError(
                "Add-on 9 report exists but has no "
                "timing.total_stage_wall_s."
            )

        addon9_runtime_s = float(
            addon9_timing[
                "total_stage_wall_s"
            ]
        )

        submission_path = (
            run_root
            / "trajectories"
            / "submission_estimated_trajectory.csv"
        )

        if not submission_path.exists():
            raise RuntimeError(
                "Add-on 9 PASS report exists but "
                "submission trajectory is missing: "
                f"{submission_path}"
            )

        submission_rows = len(
            pd.read_csv(
                submission_path,
                usecols=[
                    "frame_index"
                ],
            )
        )

        replace_pending_runtime_row(
            stage="estimated_latlon_export",
            component="Estimated lat/lon export",
            cost_scope="per_new_flight",
            runtime_s=addon9_runtime_s,
            work_count=submission_rows,
            work_unit="trajectory_row",
            source_path=addon9_report_path,
            cache_state=(
                "Blind fused trajectory already available; "
                "EPSG:3346-to-WGS84 export only."
            ),
            notes=(
                "Measured Add-on 9 wall time for blind "
                "estimated map XY to estimated lat/lon "
                "export and submission CSV generation. "
                "Estimated lat/lon are visual map-matching "
                "outputs, not GPS inputs."
            ),
        )

        print(
            "[ADDON9] estimated lat/lon export "
            "runtime ingested:",
            f"{addon9_runtime_s:.6f}s",
            f"rows={submission_rows}",
        )

    else:

        print(
            "[ADDON9] report not found; "
            "estimated_latlon_export remains pending:",
            addon9_report_path,
        )

    # ---------------------------------------------------------
    # Add-on 10 — blind-safe static plots and Folium map
    # ---------------------------------------------------------

    addon10_report_path = (
        run_root
        / "reports"
        / "addon10_no_reference_visuals"
        / "no_reference_visuals_report.json"
    )

    if addon10_report_path.exists():

        addon10_report = json.loads(
            addon10_report_path.read_text(
                encoding="utf-8"
            )
        )

        if (
            addon10_report.get("status")
            != "PASS_ADDON10_NO_REFERENCE_VISUALS"
        ):
            raise RuntimeError(
                "Unexpected Add-on 10 report status: "
                f"{addon10_report.get('status')}"
            )

        addon10_timing = addon10_report.get(
            "timing",
            {}
        )

        required_timing = [
            "plot_generation_s",
            "folium_html_map_generation_s",
            "plot_count",
        ]

        missing_timing = [
            key
            for key in required_timing
            if key not in addon10_timing
        ]

        if missing_timing:
            raise RuntimeError(
                "Add-on 10 report missing timing "
                f"fields: {missing_timing}"
            )

        plot_runtime_s = float(
            addon10_timing[
                "plot_generation_s"
            ]
        )

        folium_runtime_s = float(
            addon10_timing[
                "folium_html_map_generation_s"
            ]
        )

        plot_count = int(
            addon10_timing[
                "plot_count"
            ]
        )

        replace_pending_runtime_row(
            stage="plot_generation",
            component="No-reference plot generation",
            cost_scope="supporting_output",
            runtime_s=plot_runtime_s,
            work_count=plot_count,
            work_unit="plot",
            source_path=addon10_report_path,
            cache_state=(
                "Blind estimated trajectories and "
                "correction manifest already available."
            ),
            notes=(
                "Measured Add-on 10 generation of the "
                "four required no-reference-safe static "
                "PNG figures. Reference/GT/evaluation "
                "errors are not consumed."
            ),
        )

        replace_pending_runtime_row(
            stage="folium_html_map_generation",
            component="Folium HTML map generation",
            cost_scope="supporting_output",
            runtime_s=folium_runtime_s,
            work_count=1,
            work_unit="html_map",
            source_path=addon10_report_path,
            cache_state=(
                "Estimated lat/lon trajectory already "
                "available from Add-on 9."
            ),
            notes=(
                "Measured Add-on 10 blind-safe Folium "
                "HTML map generation. Estimated lat/lon "
                "are visual map-matching outputs, not "
                "GPS inputs."
            ),
        )

        print(
            "[ADDON10] plot runtime ingested:",
            f"{plot_runtime_s:.6f}s",
            f"plots={plot_count}",
        )

        print(
            "[ADDON10] Folium runtime ingested:",
            f"{folium_runtime_s:.6f}s",
            "maps=1",
        )

    else:

        print(
            "[ADDON10] report not found; "
            "plot_generation and "
            "folium_html_map_generation remain pending:",
            addon10_report_path,
        )

    # ADDON9_ADDON10_RUNTIME_INGEST_END

    # =========================================================
    # ADDON11_ADDON12_RUNTIME_INGEST_BEGIN
    #
    # Final two locked runtime-registry placeholders:
    #
    #   run_summary_generation
    #   environment_check
    #
    # Both are supporting/preflight costs. They are measured
    # for completeness but are not localization bottlenecks.
    # =========================================================

    # ---------------------------------------------------------
    # Add-on 11 — Markdown run summary
    # ---------------------------------------------------------

    addon11_report_path = (
        run_root
        / "reports"
        / "addon11_run_summary"
        / "run_summary_report.json"
    )

    if addon11_report_path.exists():

        addon11_report = json.loads(
            addon11_report_path.read_text(
                encoding="utf-8"
            )
        )

        if (
            addon11_report.get("status")
            != "PASS_ADDON11_RUN_SUMMARY"
        ):
            raise RuntimeError(
                "Unexpected Add-on 11 report status: "
                f"{addon11_report.get('status')}"
            )

        addon11_runtime = (
            addon11_report.get(
                "runtime",
                {}
            )
        )

        if (
            "run_summary_generation_s"
            not in addon11_runtime
        ):
            raise RuntimeError(
                "Add-on 11 report exists but has no "
                "runtime.run_summary_generation_s."
            )

        runtime_s = float(
            addon11_runtime[
                "run_summary_generation_s"
            ]
        )

        found = False

        for row in rows:

            if (
                row.get("stage")
                != "run_summary_generation"
            ):
                continue

            if runtime_s <= 0:
                raise RuntimeError(
                    "Add-on 11 runtime must be positive."
                )

            row.update(
                timing_row(
                    stage="run_summary_generation",
                    component="Run summary generation",
                    cost_scope="supporting_output",
                    runtime_s=runtime_s,
                    work_count=1,
                    work_unit="markdown_summary",
                    source_path=addon11_report_path,
                    measurement_status="MEASURED",
                    cache_state=(
                        "All blind/evaluation artifacts "
                        "already available."
                    ),
                    notes=(
                        "Measured Add-on 11 Markdown run-summary "
                        "generation. Supporting reporting cost; "
                        "not a localization bottleneck."
                    ),
                )
            )

            found = True
            break

        if not found:
            raise RuntimeError(
                "Timing registry has no "
                "run_summary_generation row."
            )

        print(
            "[ADDON11] run summary runtime ingested:",
            f"{runtime_s:.6f}s",
        )

    else:

        print(
            "[ADDON11] report not found; "
            "run_summary_generation remains pending:",
            addon11_report_path,
        )

    # ---------------------------------------------------------
    # Add-on 12 — preflight environment checker
    # ---------------------------------------------------------

    addon12_report_path = (
        run_root
        / "metrics"
        / "env_check_report.json"
    )

    if addon12_report_path.exists():

        addon12_report = json.loads(
            addon12_report_path.read_text(
                encoding="utf-8"
            )
        )

        if (
            addon12_report.get("status")
            != "PASS_ADDON12_ENVIRONMENT_CHECK"
        ):
            raise RuntimeError(
                "Unexpected Add-on 12 report status: "
                f"{addon12_report.get('status')}"
            )

        if (
            addon12_report.get("blind_safe")
            is not True
        ):
            raise RuntimeError(
                "Add-on 12 report is not blind-safe."
            )

        if (
            addon12_report.get(
                "evaluation_or_reference_read"
            )
            is not False
        ):
            raise RuntimeError(
                "Add-on 12 unexpectedly read "
                "evaluation/reference data."
            )

        addon12_runtime = (
            addon12_report.get(
                "runtime",
                {}
            )
        )

        if (
            "environment_check_s"
            not in addon12_runtime
        ):
            raise RuntimeError(
                "Add-on 12 report exists but has no "
                "runtime.environment_check_s."
            )

        runtime_s = float(
            addon12_runtime[
                "environment_check_s"
            ]
        )

        found = False

        for row in rows:

            if (
                row.get("stage")
                != "environment_check"
            ):
                continue

            if runtime_s <= 0:
                raise RuntimeError(
                    "Add-on 12 runtime must be positive."
                )

            row.update(
                timing_row(
                    stage="environment_check",
                    component="Environment preflight",
                    cost_scope="per_new_flight",
                    runtime_s=runtime_s,
                    work_count=1,
                    work_unit="preflight_check",
                    source_path=addon12_report_path,
                    measurement_status="MEASURED",
                    cache_state=(
                        "Checks local environment and prepared "
                        "pipeline prerequisites."
                    ),
                    notes=(
                        "Measured blind-safe startup preflight. "
                        "Includes dependency imports, video/map "
                        "readability and cache checks. This is "
                        "startup overhead, not online localization "
                        "latency."
                    ),
                )
            )

            found = True
            break

        if not found:
            raise RuntimeError(
                "Timing registry has no "
                "environment_check row."
            )

        print(
            "[ADDON12] environment-check runtime ingested:",
            f"{runtime_s:.6f}s",
        )

    else:

        print(
            "[ADDON12] environment report not found; "
            "environment_check remains pending:",
            addon12_report_path,
        )

    # ADDON11_ADDON12_RUNTIME_INGEST_END

    table = pd.DataFrame(
        rows
    )

    measured = (
        table["measurement_status"]
        == "MEASURED"
    )

    measured_count = int(
        measured.sum()
    )

    pending_count = int(
        (~measured).sum()
    )

    # ---------------------------------------------------------
    # Outputs
    # ---------------------------------------------------------

    csv_path = (
        metrics_dir
        / "timing_summary.csv"
    )

    json_path = (
        metrics_dir
        / "timing_summary.json"
    )

    figure_path = (
        figures_dir
        / "runtime_by_stage.png"
    )

    table.to_csv(
        csv_path,
        index=False,
    )

    save_runtime_plot(
        table,
        figure_path,
    )

    report = {
        "addon": (
            "Add-on 4 — Runtime benchmark layer"
        ),
        "status": (
            "PASS_PARTIAL_RUNTIME_REGISTRY"
            if pending_count
            else "PASS_RUNTIME_BENCHMARK"
        ),
        "variant": variant,
        "descriptor_tag": tag,
        "measured_rows": measured_count,
        "pending_rows": pending_count,
        "important_rules": [
            (
                "Recorded runtime must come from an "
                "explicitly named trusted artifact."
            ),
            (
                "DINO descriptor encoding and DINO "
                "cached retrieval are separate costs."
            ),
            (
                "Offline map descriptor construction "
                "must not be counted as per-flight cost."
            ),
            (
                "ORB timing is measured over the full "
                "403 x Top-20 workload."
            ),
            (
                "Pending stages remain null rather than "
                "receiving estimated or invented runtime."
            ),
        ],
        "rows": (
            table.astype(object)
            .where(
                pd.notna(table),
                None,
            ).to_dict(
                orient="records"
            )
        ),
        "outputs": {
            "csv": str(csv_path),
            "json": str(json_path),
            "figure": str(figure_path),
        },
    }

    json_path.write_text(
        json.dumps(
            report,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    # ---------------------------------------------------------
    # Console summary
    # ---------------------------------------------------------

    print("=" * 80)
    print(
        "STAGE 5C — ADD-ON 4 "
        "RUNTIME BENCHMARK REGISTRY"
    )
    print("=" * 80)

    print(
        f"status: {report['status']}"
    )

    print(
        f"measured rows: {measured_count}"
    )

    print(
        f"pending rows: {pending_count}"
    )

    print()
    print("Measured timing")
    print("-" * 80)

    show = table.loc[
        measured,
        [
            "stage",
            "runtime_s",
            "work_count",
            "work_unit",
            "ms_per_work_item",
            "work_items_per_s",
            "secondary_count",
            "secondary_unit",
            "ms_per_secondary_item",
            "cost_scope",
        ],
    ]

    print(
        show.to_string(
            index=False
        )
    )

    print()
    print("Pending")
    print("-" * 80)

    for stage in table.loc[
        ~measured,
        "stage",
    ]:
        print(stage)

    print()
    print("Saved")
    print("-" * 80)

    print(csv_path)
    print(json_path)
    print(figure_path)


if __name__ == "__main__":
    main()
