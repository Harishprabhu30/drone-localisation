'''
ROOT=outputs/villoc/traj01_90deg_stable120m
RUN="$ROOT/blind_demo_addons/eval_traj01_known_gt"

python scripts/villoc/blind_demo/addon6_deployment_cost_breakdown.py \
  --run-root "$RUN" \
  --sampled-query-rate-hz 1.0 \
  2>&1 | tee \
  "$RUN/logs/addon6_deployment_cost_breakdown.log"
'''

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


INTERPRETATION_RULE = (
    "The full experimental pipeline includes offline preparation, "
    "per-flight processing, and online-like query cost. Embedded "
    "feasibility should be judged mainly from online-like "
    "per-frame/per-query cost after map cache reuse, not from "
    "one-time map cache construction."
)


def finite_float(value: Any) -> float | None:
    if value is None:
        return None

    try:
        out = float(value)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(out):
        return None

    return out


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)

    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def fmt_seconds(value: float | None) -> str:
    if value is None:
        return "not measured"

    if value < 1.0:
        return f"{value * 1000.0:.1f} ms"

    if value < 60.0:
        return f"{value:.2f} s"

    minutes = value / 60.0

    if minutes < 60.0:
        return f"{minutes:.2f} min"

    return f"{minutes / 60.0:.2f} h"


def fmt_ms(value: float | None) -> str:
    if value is None:
        return "not measured"

    return f"{value:.2f} ms"


def fmt_mib(value: float | None) -> str:
    if value is None:
        return "not measured"

    return f"{value:.1f} MiB"


def timing_lookup(
    table: pd.DataFrame,
    stage: str,
) -> dict[str, Any] | None:
    sub = table.loc[
        table["stage"].astype(str) == stage
    ]

    if sub.empty:
        return None

    row = sub.iloc[0]

    return {
        key: (
            None
            if pd.isna(value)
            else value
        )
        for key, value in row.to_dict().items()
    }


def stage_runtime(
    table: pd.DataFrame,
    stage: str,
) -> float | None:
    row = timing_lookup(
        table,
        stage,
    )

    if row is None:
        return None

    if (
        str(
            row.get(
                "measurement_status"
            )
        )
        != "MEASURED"
    ):
        return None

    return finite_float(
        row.get(
            "runtime_s"
        )
    )


def normalized_value(
    table: pd.DataFrame,
    stage: str,
    field: str,
) -> float | None:
    row = timing_lookup(
        table,
        stage,
    )

    if row is None:
        return None

    if (
        str(
            row.get(
                "measurement_status"
            )
        )
        != "MEASURED"
    ):
        return None

    return finite_float(
        row.get(field)
    )


def memory_peak(
    resource: dict[str, Any],
    stage: str,
) -> float | None:
    record = (
        resource
        .get(
            "stage_memory",
            {}
        )
        .get(
            stage,
            {}
        )
    )

    return finite_float(
        record.get(
            "peak_process_tree_rss_mib"
        )
    )


def classification_row(
    *,
    category: str,
    stage: str,
    label: str,
    status: str,
    runtime_s: float | None,
    reusable: bool,
    notes: str,
) -> dict[str, Any]:
    return {
        "category": category,
        "stage": stage,
        "label": label,
        "status": status,
        "runtime_s": runtime_s,
        "reusable": reusable,
        "notes": notes,
    }


def online_row(
    *,
    stage: str,
    label: str,
    normalized_field: str,
    normalized_unit: str,
    latency_ms: float | None,
    peak_rss_mib: float | None,
    notes: str,
) -> dict[str, Any]:
    rate_hz = None

    if (
        latency_ms is not None
        and latency_ms > 0
    ):
        rate_hz = (
            1000.0 / latency_ms
        )

    return {
        "stage": stage,
        "label": label,
        "normalized_field": (
            normalized_field
        ),
        "normalized_unit": (
            normalized_unit
        ),
        "latency_ms": latency_ms,
        "equivalent_rate_hz": (
            rate_hz
        ),
        "peak_rss_mib": (
            peak_rss_mib
        ),
        "notes": notes,
    }



def run_native_blind_deployment_breakdown(
    *,
    run_root: Path,
    metrics_dir: Path,
    timing: pd.DataFrame,
    timing_report: dict[str, Any],
    resource: dict[str, Any],
    cache: dict[str, Any],
    sampled_query_rate_hz: float,
) -> None:

    # =====================================================
    # Native recorded-flight deployment interpretation.
    #
    # Uses ONLY the promoted blind-demo stages.
    # Historical LightGlue, F1/F3/F3B, smoke tests and
    # map-variant diagnostics are not included.
    # =====================================================

    if (
        timing_report.get(
            "registry_mode"
        )
        != "PASS_NATIVE_BLIND_RUNTIME_REGISTRY"
    ):
        raise RuntimeError(
            "Timing registry is not native blind mode."
        )

    if (
        resource.get(
            "registry_mode"
        )
        != "PASS_NATIVE_BLIND_RESOURCE_REGISTRY"
    ):
        raise RuntimeError(
            "Resource registry is not native blind mode."
        )

    # -------------------------------------------------
    # Offline reusable preparation.
    # -------------------------------------------------

    map_cache_runtime = stage_runtime(
        timing,
        "dino_map_descriptor_cache_creation",
    )

    offline_rows = [
        classification_row(
            category="offline_one_time",
            stage="orthophoto_crop",
            label="Orthophoto AOI crop",
            status="NOT_TIMED",
            runtime_s=None,
            reusable=True,
            notes=(
                "Existing prepared AOI is reused. "
                "Historical crop runtime was not recorded."
            ),
        ),
        classification_row(
            category="offline_one_time",
            stage="tile_generation",
            label="Satellite tile generation",
            status="NOT_TIMED",
            runtime_s=None,
            reusable=True,
            notes=(
                "Prepared 512_s256 tile set is reused "
                "for flights inside the same AOI."
            ),
        ),
        classification_row(
            category="offline_one_time",
            stage="tile_index_creation",
            label="Tile index creation",
            status="NOT_TIMED",
            runtime_s=None,
            reusable=True,
            notes=(
                "Prepared georeferenced tile index "
                "is reused."
            ),
        ),
        classification_row(
            category="offline_one_time",
            stage="dino_map_descriptor_cache_creation",
            label="DINO map descriptor cache build",
            status=(
                "MEASURED"
                if map_cache_runtime is not None
                else "NOT_TIMED"
            ),
            runtime_s=map_cache_runtime,
            reusable=True,
            notes=(
                "Measured reusable neural-map "
                "descriptor preparation."
            ),
        ),
    ]

    measured_offline_total_s = sum(
        row["runtime_s"]
        for row in offline_rows
        if row["runtime_s"] is not None
    )

    offline_not_timed = [
        row["stage"]
        for row in offline_rows
        if row["runtime_s"] is None
    ]

    # -------------------------------------------------
    # Per-new-flight stages.
    # -------------------------------------------------

    per_flight_specs = [
        (
            "video_metadata_read",
            "Video metadata read",
            "Recorded-flight metadata inspection.",
        ),
        (
            "frame_extraction",
            "Blind frame extraction",
            "Reference-free 1-fps image extraction.",
        ),
        (
            "query_manifest_build",
            "Blind query manifest",
            "Reference-free frame/query manifest.",
        ),
        (
            "relative_odometry",
            "XFeat relative frontend",
            "Relative visual motion for this flight.",
        ),
        (
            "dino_query_descriptor_extraction",
            "DINO query encoding",
            "Query descriptors must be computed per flight.",
        ),
        (
            "dino_retrieval_against_map_cache",
            "DINO cached retrieval",
            "Ranks current query descriptors against reusable map descriptors.",
        ),
        (
            "orb_topk_reranking",
            "ORB Top-20 verifier/reranker",
            "Primary promoted absolute-verification stage.",
        ),
        (
            "blind_map_bootstrap",
            "Blind map bootstrap",
            "Attempts causal trusted map initialization.",
        ),
        (
            "blind_map_alignment",
            "Map-alignment continuation",
            "Map alignment when locked; relative-only continuation otherwise.",
        ),
        (
            "blind_temporal_fusion",
            "Temporal fusion/control",
            "Metric fusion when locked; safely skipped for no-lock.",
        ),
        (
            "estimated_latlon_export",
            "Estimated-output export",
            "Stable output export; no coordinate conversion in no-lock state.",
        ),
    ]

    per_flight_rows = []

    for stage, label, notes in per_flight_specs:

        runtime_s = stage_runtime(
            timing,
            stage,
        )

        per_flight_rows.append(
            classification_row(
                category="per_new_flight",
                stage=stage,
                label=label,
                status=(
                    "MEASURED"
                    if runtime_s is not None
                    else "PENDING"
                ),
                runtime_s=runtime_s,
                reusable=False,
                notes=notes,
            )
        )

    measured_per_flight_total_s = sum(
        row["runtime_s"]
        for row in per_flight_rows
        if row["runtime_s"] is not None
    )

    per_flight_pending = [
        row["stage"]
        for row in per_flight_rows
        if row["runtime_s"] is None
    ]

    # -------------------------------------------------
    # Supporting output stages.
    # -------------------------------------------------

    plot_runtime = stage_runtime(
        timing,
        "plot_generation",
    )

    folium_record = timing_lookup(
        timing,
        "folium_html_map_generation",
    )

    supporting_output = {
        "plot_generation": {
            "status": (
                "MEASURED"
                if plot_runtime is not None
                else "PENDING"
            ),
            "runtime_s": plot_runtime,
        },
        "folium_html_map_generation": {
            "status": (
                folium_record.get(
                    "measurement_status"
                )
                if folium_record is not None
                else "UNKNOWN"
            ),
            "runtime_s": (
                finite_float(
                    folium_record.get(
                        "runtime_s"
                    )
                )
                if folium_record is not None
                else None
            ),
            "reason": (
                "Skipped because no trusted "
                "absolute latitude/longitude existed."
            ),
        },
    }

    # -------------------------------------------------
    # Normalized online-like costs.
    # -------------------------------------------------

    relative_ms = normalized_value(
        timing,
        "relative_odometry",
        "ms_per_pair",
    )

    dino_encode_ms = normalized_value(
        timing,
        "dino_query_descriptor_extraction",
        "ms_per_frame",
    )

    dino_retrieval_ms = normalized_value(
        timing,
        "dino_retrieval_against_map_cache",
        "ms_per_query",
    )

    orb_ms = normalized_value(
        timing,
        "orb_topk_reranking",
        "ms_per_query",
    )

    bootstrap_ms = normalized_value(
        timing,
        "blind_map_bootstrap",
        "ms_per_query",
    )

    alignment_ms = normalized_value(
        timing,
        "blind_map_alignment",
        "ms_per_work_item",
    )

    fusion_ms = normalized_value(
        timing,
        "blind_temporal_fusion",
        "ms_per_work_item",
    )

    online_rows = [
        online_row(
            stage="relative_odometry",
            label="XFeat relative motion",
            normalized_field="ms_per_pair",
            normalized_unit="ms/pair",
            latency_ms=relative_ms,
            peak_rss_mib=memory_peak(
                resource,
                "relative_odometry",
            ),
            notes=(
                "Consecutive-frame relative motion."
            ),
        ),
        online_row(
            stage="dino_query_descriptor_extraction",
            label="DINO query encoding",
            normalized_field="ms_per_frame",
            normalized_unit="ms/frame",
            latency_ms=dino_encode_ms,
            peak_rss_mib=memory_peak(
                resource,
                "dino_query_descriptor_extraction",
            ),
            notes=(
                "CPU DINOv2 encoding of one query image."
            ),
        ),
        online_row(
            stage="dino_retrieval_against_map_cache",
            label="DINO cached retrieval",
            normalized_field="ms_per_query",
            normalized_unit="ms/query",
            latency_ms=dino_retrieval_ms,
            peak_rss_mib=memory_peak(
                resource,
                "dino_retrieval_against_map_cache",
            ),
            notes=(
                "Cosine ranking against cached map descriptors."
            ),
        ),
        online_row(
            stage="orb_topk_reranking",
            label="ORB Top-20 verification",
            normalized_field="ms_per_query",
            normalized_unit="ms/query",
            latency_ms=orb_ms,
            peak_rss_mib=memory_peak(
                resource,
                "orb_topk_reranking",
            ),
            notes=(
                "Includes twenty UAV-to-map candidate checks."
            ),
        ),
        online_row(
            stage="blind_map_bootstrap",
            label="Blind map bootstrap",
            normalized_field="ms_per_query",
            normalized_unit="ms/query",
            latency_ms=bootstrap_ms,
            peak_rss_mib=memory_peak(
                resource,
                "blind_map_bootstrap",
            ),
            notes=(
                "Causal trusted-lock decision stage."
            ),
        ),
        online_row(
            stage="blind_map_alignment",
            label="Map-alignment continuation",
            normalized_field="ms_per_work_item",
            normalized_unit="ms/trajectory-row",
            latency_ms=alignment_ms,
            peak_rss_mib=memory_peak(
                resource,
                "blind_map_alignment",
            ),
            notes=(
                "Relative-only continuation in this no-lock run."
            ),
        ),
        online_row(
            stage="blind_temporal_fusion",
            label="Temporal fusion/control",
            normalized_field="ms_per_work_item",
            normalized_unit="ms/trajectory-row",
            latency_ms=fusion_ms,
            peak_rss_mib=memory_peak(
                resource,
                "blind_temporal_fusion",
            ),
            notes=(
                "Fusion-control overhead; metric fusion was "
                "not applicable because no map lock existed."
            ),
        ),
    ]

    serial_components = [
        row
        for row in online_rows
        if row["latency_ms"] is not None
    ]

    serial_online_ms = sum(
        float(
            row["latency_ms"]
        )
        for row in serial_components
    )

    serial_online_hz = (
        1000.0 / serial_online_ms
        if serial_online_ms > 0
        else None
    )

    target_rate_hz = float(
        sampled_query_rate_hz
    )

    target_budget_ms = (
        1000.0 / target_rate_hz
        if target_rate_hz > 0
        else None
    )

    realtime_factor = (
        serial_online_ms
        / target_budget_ms
        if (
            target_budget_ms is not None
            and target_budget_ms > 0
        )
        else None
    )

    nominal_rate_met = (
        realtime_factor <= 1.0
        if realtime_factor is not None
        else None
    )

    # -------------------------------------------------
    # Optimization priorities.
    # -------------------------------------------------

    optimization_rows = []

    for row in serial_components:

        latency_ms = float(
            row["latency_ms"]
        )

        share = (
            latency_ms / serial_online_ms
            if serial_online_ms > 0
            else 0.0
        )

        if share >= 0.40:
            priority = "VERY_HIGH"
        elif share >= 0.20:
            priority = "HIGH"
        elif share >= 0.05:
            priority = "MEDIUM"
        else:
            priority = "LOW"

        optimization_rows.append(
            {
                "stage":
                    row["stage"],

                "label":
                    row["label"],

                "latency_ms":
                    latency_ms,

                "serial_latency_share":
                    share,

                "serial_latency_percent":
                    100.0 * share,

                "optimization_priority":
                    priority,
            }
        )

    optimization_rows.sort(
        key=lambda item:
            item["latency_ms"],
        reverse=True,
    )

    # -------------------------------------------------
    # Memory interpretation.
    # -------------------------------------------------

    measured_peaks = [
        {
            "stage":
                stage,

            "peak_rss_mib":
                finite_float(
                    record.get(
                        "peak_process_tree_rss_mib"
                    )
                ),
        }
        for stage, record
        in resource.get(
            "stage_memory",
            {},
        ).items()
        if finite_float(
            record.get(
                "peak_process_tree_rss_mib"
            )
        ) is not None
    ]

    if measured_peaks:

        max_memory_stage = max(
            measured_peaks,
            key=lambda item:
                item["peak_rss_mib"],
        )

        memory_assessment = (
            "Live peak-RAM measurements are available "
            "for at least one stage."
        )

    else:

        max_memory_stage = None

        memory_assessment = (
            "No live per-stage peak-RAM measurements "
            "were captured for this run. Static system "
            "RAM and storage are known, but stage peak "
            "RSS must remain unknown until measured live."
        )

    # -------------------------------------------------
    # Cache/storage context from native resource keys.
    # -------------------------------------------------

    cache_context = {
        "query_descriptor_cache_mib":
            finite_float(
                cache[
                    "query_descriptor_cache"
                ].get(
                    "size_mib"
                )
            ),

        "map_descriptor_cache_mib":
            finite_float(
                cache[
                    "map_descriptor_cache"
                ].get(
                    "size_mib"
                )
            ),

        "prepared_tile_folder_mib":
            finite_float(
                cache[
                    "prepared_tile_folder"
                ].get(
                    "size_mib"
                )
            ),

        "blind_frame_folder_mib":
            finite_float(
                cache[
                    "blind_frame_folder"
                ].get(
                    "size_mib"
                )
            ),

        "blind_run_output_folder_mib":
            finite_float(
                cache[
                    "blind_run_output_folder"
                ].get(
                    "size_mib"
                )
            ),
    }

    theoretical_compute = {
        "required_for_current_cpu_baseline":
            False,

        "status":
            "NOT_REQUIRED_FOR_CURRENT_DEPLOYMENT_BASELINE",

        "reason": (
            "Measured latency and workload directly "
            "characterize this CPU implementation. "
            "FLOPS/TOPS become useful when comparing "
            "future GPU/NPU deployment targets."
        ),
    }

    # -------------------------------------------------
    # Final report.
    # -------------------------------------------------

    status = (
        "PASS_DEPLOYMENT_COST_BREAKDOWN"
        if not per_flight_pending
        else "PASS_PARTIAL_DEPLOYMENT_COST_BREAKDOWN"
    )

    report = {
        "addon":
            "Add-on 6 — Offline vs online cost breakdown",

        "status":
            status,

        "registry_mode":
            "PASS_NATIVE_BLIND_DEPLOYMENT_BREAKDOWN",

        "localization_state":
            timing_report.get(
                "localization_state"
            ),

        "interpretation_rule":
            INTERPRETATION_RULE,

        "timing_registry_status":
            timing_report.get(
                "status"
            ),

        "offline_one_time": {
            "rows":
                offline_rows,

            "measured_total_s":
                measured_offline_total_s,

            "measured_total_human":
                fmt_seconds(
                    measured_offline_total_s
                ),

            "not_timed_stages":
                offline_not_timed,
        },

        "per_new_flight": {
            "rows":
                per_flight_rows,

            "measured_total_s":
                measured_per_flight_total_s,

            "measured_total_human":
                fmt_seconds(
                    measured_per_flight_total_s
                ),

            "pending_stages":
                per_flight_pending,
        },

        "supporting_output": (
            supporting_output
        ),

        "online_like": {
            "rows":
                online_rows,

            "serial_equivalent_latency_ms":
                serial_online_ms,

            "serial_equivalent_latency_s":
                serial_online_ms
                / 1000.0,

            "serial_equivalent_capacity_hz":
                serial_online_hz,

            "nominal_input_query_rate_hz":
                target_rate_hz,

            "nominal_query_budget_ms":
                target_budget_ms,

            "serial_realtime_factor":
                realtime_factor,

            "nominal_rate_met":
                nominal_rate_met,

            "important_note": (
                "This is a serial-equivalent deployment "
                "planning estimate obtained by summing "
                "independently measured normalized stage "
                "latencies. It is not a claim that the "
                "current scripts run as a streaming "
                "real-time pipeline."
            ),
        },

        "optimization_priority":
            optimization_rows,

        "memory_assessment": {
            "system_ram_gib":
                finite_float(
                    resource.get(
                        "system",
                        {},
                    ).get(
                        "system_ram_total_gib"
                    )
                ),

            "largest_measured_stage_peak":
                max_memory_stage,

            "assessment":
                memory_assessment,
        },

        "cache_context":
            cache_context,

        "excluded_from_promoted_demo": [
            "LightGlue diagnostics",
            "ORB smoke runs",
            "map-variant comparison runs",
            "historical F1/F3/F3B evaluation pipeline",
        ],

        "theoretical_compute_metrics":
            theoretical_compute,
    }

    json_path = (
        metrics_dir
        / "deployment_cost_breakdown.json"
    )

    md_path = (
        metrics_dir
        / "deployment_cost_breakdown.md"
    )

    json_path.write_text(
        json.dumps(
            report,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    # -------------------------------------------------
    # Human-readable Markdown.
    # -------------------------------------------------

    lines = [
        "# Add-on 6 — Deployment Cost Breakdown",
        "",
        f"- Status: `{status}`",
        (
            "- Registry mode: "
            "`PASS_NATIVE_BLIND_DEPLOYMENT_BREAKDOWN`"
        ),
        (
            "- Localization state: "
            f"`{report['localization_state']}`"
        ),
        (
            "- Nominal sampled query rate: "
            f"`{target_rate_hz:g} Hz`"
        ),
        "",
        "## Offline reusable preparation",
        "",
        (
            "- Measured reusable map-cache build: "
            f"`{fmt_seconds(measured_offline_total_s)}`"
        ),
        (
            "- Not timed: "
            + (
                ", ".join(
                    offline_not_timed
                )
                if offline_not_timed
                else "none"
            )
        ),
        "",
        "## Per new flight",
        "",
        (
            "- Measured processing total: "
            f"`{fmt_seconds(measured_per_flight_total_s)}`"
        ),
        "",
        "## Serial-equivalent online-like latency",
        "",
        (
            "- Total: "
            f"`{serial_online_ms:.2f} ms/query`"
        ),
        (
            "- Equivalent capacity: "
            f"`{serial_online_hz:.3f} Hz`"
        ),
        (
            "- 1 Hz realtime factor: "
            f"`{realtime_factor:.3f}x`"
        ),
        (
            "- Nominal 1 Hz target met: "
            f"`{nominal_rate_met}`"
        ),
        "",
        "### Stage contribution",
        "",
        "| Stage | Latency | Share | Priority |",
        "|---|---:|---:|---|",
    ]

    for row in optimization_rows:

        lines.append(
            "| "
            + str(
                row["stage"]
            )
            + " | "
            + f"{row['latency_ms']:.2f} ms"
            + " | "
            + f"{row['serial_latency_percent']:.1f}%"
            + " | "
            + str(
                row["optimization_priority"]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Memory",
            "",
            memory_assessment,
            "",
            "## Interpretation",
            "",
            INTERPRETATION_RULE,
            "",
            (
                "Folium was intentionally skipped "
                "because this run had no trusted "
                "absolute map lock."
            ),
            "",
        ]
    )

    md_path.write_text(
        "\n".join(
            lines
        ),
        encoding="utf-8",
    )

    print("=" * 88)
    print(
        "STAGE 8F.3 — NATIVE BLIND "
        "DEPLOYMENT COST BREAKDOWN"
    )
    print("=" * 88)

    print()
    print(
        "registry mode         : "
        "PASS_NATIVE_BLIND_DEPLOYMENT_BREAKDOWN"
    )

    print(
        "status                :",
        status,
    )

    print(
        "localization state    :",
        report[
            "localization_state"
        ],
    )

    print()
    print("Cost classes")
    print("-" * 88)

    print(
        "offline measured      :",
        f"{measured_offline_total_s:.3f} s",
    )

    print(
        "per-new-flight total  :",
        f"{measured_per_flight_total_s:.3f} s",
    )

    print()
    print("Online-like serial estimate")
    print("-" * 88)

    print(
        "latency/query         :",
        f"{serial_online_ms:.3f} ms",
    )

    print(
        "equivalent capacity   :",
        f"{serial_online_hz:.3f} Hz",
    )

    print(
        "1 Hz realtime factor  :",
        f"{realtime_factor:.3f}x",
    )

    print(
        "1 Hz target met       :",
        nominal_rate_met,
    )

    print()
    print("Optimization priority")
    print("-" * 88)

    for row in optimization_rows:

        print(
            f"{row['stage']:40s}"
            f"{row['latency_ms']:10.3f} ms  "
            f"{row['serial_latency_percent']:6.2f}%  "
            f"{row['optimization_priority']}"
        )

    print()
    print("Memory")
    print("-" * 88)

    print(memory_assessment)

    print()
    print("Saved")
    print("-" * 88)

    print(json_path)
    print(md_path)

    print()
    print(
        "PASS_DEMO_STAGE8F3_NATIVE_DEPLOYMENT_BREAKDOWN"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Add-on 6 deployment cost breakdown: "
            "offline, per-new-flight, and "
            "online-like costs."
        )
    )

    parser.add_argument(
        "--run-root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--sampled-query-rate-hz",
        type=float,
        default=1.0,
        help=(
            "Nominal incoming sampled-query rate "
            "used only for serial-equivalent "
            "realtime interpretation."
        ),
    )

    args = parser.parse_args()

    run_root = (
        args.run_root.resolve()
    )

    metrics_dir = (
        run_root / "metrics"
    )

    timing_csv = (
        metrics_dir
        / "timing_summary.csv"
    )

    timing_json = (
        metrics_dir
        / "timing_summary.json"
    )

    resource_json = (
        metrics_dir
        / "resource_summary.json"
    )

    cache_json = (
        metrics_dir
        / "cache_size_summary.json"
    )

    for path in [
        timing_csv,
        timing_json,
        resource_json,
        cache_json,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                path
            )

    timing = pd.read_csv(
        timing_csv
    )

    timing_report = load_json(
        timing_json
    )

    resource = load_json(
        resource_json
    )

    cache = load_json(
        cache_json
    )

    if (
        timing_report.get(
            "registry_mode"
        )
        == "PASS_NATIVE_BLIND_RUNTIME_REGISTRY"
        and resource.get(
            "registry_mode"
        )
        == "PASS_NATIVE_BLIND_RESOURCE_REGISTRY"
    ):
        run_native_blind_deployment_breakdown(
            run_root=run_root,
            metrics_dir=metrics_dir,
            timing=timing,
            timing_report=timing_report,
            resource=resource,
            cache=cache,
            sampled_query_rate_hz=(
                args.sampled_query_rate_hz
            ),
        )
        return

    # =====================================================
    # 1. Offline / one-time map preparation
    # =====================================================

    map_cache_runtime = stage_runtime(
        timing,
        "dino_map_descriptor_cache_creation",
    )

    offline_rows = [
        classification_row(
            category="offline_one_time",
            stage="orthophoto_crop",
            label="Orthophoto AOI crop",
            status="NOT_TIMED",
            runtime_s=None,
            reusable=True,
            notes=(
                "Existing AOI/map asset is reused. "
                "Historical crop runtime was not "
                "recorded."
            ),
        ),
        classification_row(
            category="offline_one_time",
            stage="tile_generation",
            label="Satellite tile generation",
            status="NOT_TIMED",
            runtime_s=None,
            reusable=True,
            notes=(
                "Generated tile set is reusable for "
                "the same AOI and tile specification."
            ),
        ),
        classification_row(
            category="offline_one_time",
            stage="tile_index_creation",
            label="Tile index creation",
            status="NOT_TIMED",
            runtime_s=None,
            reusable=True,
            notes=(
                "Tile index already exists and is "
                "reused by later flights."
            ),
        ),
        classification_row(
            category="offline_one_time",
            stage=(
                "dino_map_descriptor_cache_creation"
            ),
            label=(
                "DINO map descriptor cache build"
            ),
            status=(
                "MEASURED"
                if map_cache_runtime
                is not None
                else "NOT_TIMED"
            ),
            runtime_s=(
                map_cache_runtime
            ),
            reusable=True,
            notes=(
                "Main measured offline neural "
                "preprocessing cost. Reusable while "
                "AOI, tiles, model and descriptor "
                "configuration remain unchanged."
            ),
        ),
    ]

    measured_offline_total_s = sum(
        row["runtime_s"]
        for row in offline_rows
        if row["runtime_s"]
        is not None
    )

    offline_pending = [
        row["stage"]
        for row in offline_rows
        if row["runtime_s"]
        is None
    ]

    # =====================================================
    # 2. Per-new-flight processing
    # =====================================================

    per_flight_specs = [
        (
            "video_metadata_read",
            "Video metadata read",
            (
                "Reads FPS, frame count, resolution "
                "and duration from each new video."
            ),
        ),
        (
            "frame_extraction",
            "Video frame extraction",
            "Each new video must be sampled/extracted.",
        ),
        (
            "query_manifest_build",
            "Blind query manifest construction",
            (
                "Builds the reference-free per-flight "
                "query sequence from extracted frames."
            ),
        ),
        (
            "dino_query_descriptor_extraction",
            "DINO query descriptor extraction",
            (
                "Repeats for every new flight; "
                "map descriptors do not."
            ),
        ),
        (
            "relative_odometry",
            "XFeat relative odometry",
            (
                "Repeats over consecutive sampled "
                "flight frames."
            ),
        ),
        (
            "dino_retrieval_against_map_cache",
            "DINO retrieval from map cache",
            (
                "Fast ranking stage once both query "
                "and map descriptors exist."
            ),
        ),
        (
            "orb_topk_reranking",
            "ORB Top-20 reranking",
            (
                "Repeats for each query and each "
                "retrieved candidate."
            ),
        ),
        (
            "correction_manifest_build",
            "Absolute correction manifest",
            (
                "Builds correction evidence from "
                "absolute localization results."
            ),
        ),
        (
            "temporal_gating",
            "Temporal consistency gate",
            (
                "Applies the promoted online-safe "
                "correction policy."
            ),
        ),
        (
            "fusion_replay",
            "Relative/absolute fusion",
            (
                "Produces fused trajectory using "
                "accepted corrections."
            ),
        ),
        (
            "estimated_latlon_export",
            "Estimated trajectory/lat-lon export",
            (
                "Implemented by later Add-on 9; "
                "runtime currently pending."
            ),
        ),
    ]

    per_flight_rows = []

    for (
        stage,
        label,
        notes,
    ) in per_flight_specs:
        runtime_s = stage_runtime(
            timing,
            stage,
        )

        per_flight_rows.append(
            classification_row(
                category="per_new_flight",
                stage=stage,
                label=label,
                status=(
                    "MEASURED"
                    if runtime_s
                    is not None
                    else "PENDING"
                ),
                runtime_s=runtime_s,
                reusable=False,
                notes=notes,
            )
        )

    measured_per_flight_total_s = sum(
        row["runtime_s"]
        for row in per_flight_rows
        if row["runtime_s"]
        is not None
    )

    per_flight_pending = [
        row["stage"]
        for row in per_flight_rows
        if row["runtime_s"]
        is None
    ]

    # =====================================================
    # 3. Online-like normalized costs
    # =====================================================

    relative_ms = normalized_value(
        timing,
        "relative_odometry",
        "ms_per_pair",
    )

    dino_encode_ms = normalized_value(
        timing,
        "dino_query_descriptor_extraction",
        "ms_per_frame",
    )

    dino_retrieval_ms = normalized_value(
        timing,
        "dino_retrieval_against_map_cache",
        "ms_per_query",
    )

    orb_ms = normalized_value(
        timing,
        "orb_topk_reranking",
        "ms_per_query",
    )

    temporal_ms = normalized_value(
        timing,
        "temporal_gating",
        "ms_per_query",
    )

    fusion_ms = normalized_value(
        timing,
        "fusion_replay",
        "ms_per_query",
    )

    online_rows = [
        online_row(
            stage="relative_odometry",
            label="XFeat relative motion",
            normalized_field="ms_per_pair",
            normalized_unit="ms/pair",
            latency_ms=relative_ms,
            peak_rss_mib=memory_peak(
                resource,
                "relative_odometry",
            ),
            notes=(
                "One consecutive-frame relative "
                "motion estimate."
            ),
        ),
        online_row(
            stage=(
                "dino_query_descriptor_extraction"
            ),
            label="DINO query encoding",
            normalized_field="ms_per_frame",
            normalized_unit="ms/frame",
            latency_ms=dino_encode_ms,
            peak_rss_mib=memory_peak(
                resource,
                "dino_query_descriptor_extraction",
            ),
            notes=(
                "Neural query encoding on CPU."
            ),
        ),
        online_row(
            stage=(
                "dino_retrieval_against_map_cache"
            ),
            label="DINO cached retrieval",
            normalized_field="ms_per_query",
            normalized_unit="ms/query",
            latency_ms=dino_retrieval_ms,
            peak_rss_mib=memory_peak(
                resource,
                "dino_retrieval_against_map_cache",
            ),
            notes=(
                "Ranking against already-cached "
                "map descriptors."
            ),
        ),
        online_row(
            stage="orb_topk_reranking",
            label="ORB Top-20 reranking",
            normalized_field="ms_per_query",
            normalized_unit="ms/query",
            latency_ms=orb_ms,
            peak_rss_mib=memory_peak(
                resource,
                "orb_topk_reranking",
            ),
            notes=(
                "Per-query cost includes Top-20 "
                "query-tile verification workload."
            ),
        ),
        online_row(
            stage="temporal_gating",
            label="Temporal correction gate",
            normalized_field="ms_per_query",
            normalized_unit="ms/query",
            latency_ms=temporal_ms,
            peak_rss_mib=memory_peak(
                resource,
                "temporal_gating",
            ),
            notes=(
                "Online-like accepted-correction "
                "decision cost."
            ),
        ),
        online_row(
            stage="fusion_replay",
            label="Relative/absolute fusion",
            normalized_field="ms_per_query",
            normalized_unit="ms/query",
            latency_ms=fusion_ms,
            peak_rss_mib=memory_peak(
                resource,
                "fusion_replay",
            ),
            notes=(
                "Current measured fusion replay "
                "normalized per query frame."
            ),
        ),
    ]

    # =====================================================
    # Serial-equivalent online-like latency
    # =====================================================

    serial_components = [
        row
        for row in online_rows
        if row[
            "latency_ms"
        ] is not None
    ]

    serial_online_ms = sum(
        float(
            row[
                "latency_ms"
            ]
        )
        for row in serial_components
    )

    serial_online_hz = (
        1000.0
        / serial_online_ms
        if serial_online_ms > 0
        else None
    )

    target_rate_hz = float(
        args.sampled_query_rate_hz
    )

    target_budget_ms = (
        1000.0 / target_rate_hz
        if target_rate_hz > 0
        else None
    )

    realtime_factor = None

    if (
        target_budget_ms is not None
        and target_budget_ms > 0
    ):
        realtime_factor = (
            serial_online_ms
            / target_budget_ms
        )

    target_realtime_met = (
        realtime_factor <= 1.0
        if realtime_factor
        is not None
        else None
    )

    # =====================================================
    # Optimization contribution ranking
    # =====================================================

    optimization_rows = []

    for row in serial_components:
        latency_ms = float(
            row["latency_ms"]
        )

        share = (
            latency_ms
            / serial_online_ms
            if serial_online_ms > 0
            else 0.0
        )

        if share >= 0.40:
            priority = "VERY_HIGH"

        elif share >= 0.20:
            priority = "HIGH"

        elif share >= 0.05:
            priority = "MEDIUM"

        else:
            priority = "LOW"

        optimization_rows.append(
            {
                "stage": row[
                    "stage"
                ],
                "label": row[
                    "label"
                ],
                "latency_ms": (
                    latency_ms
                ),
                "serial_latency_share": (
                    share
                ),
                "serial_latency_percent": (
                    100.0 * share
                ),
                "optimization_priority": (
                    priority
                ),
            }
        )

    optimization_rows.sort(
        key=lambda x: x[
            "latency_ms"
        ],
        reverse=True,
    )

    # =====================================================
    # Memory interpretation
    # =====================================================

    measured_peaks = [
        {
            "stage": stage,
            "peak_rss_mib": (
                finite_float(
                    record.get(
                        "peak_process_tree_rss_mib"
                    )
                )
            ),
        }
        for stage, record
        in resource.get(
            "stage_memory",
            {}
        ).items()
        if finite_float(
            record.get(
                "peak_process_tree_rss_mib"
            )
        ) is not None
    ]

    max_memory_stage = None

    if measured_peaks:
        max_memory_stage = max(
            measured_peaks,
            key=lambda x: x[
                "peak_rss_mib"
            ],
        )

    system_ram_gib = finite_float(
        resource.get(
            "system",
            {}
        ).get(
            "system_ram_total_gib"
        )
    )

    max_peak_fraction_ram = None

    if (
        max_memory_stage
        and system_ram_gib
        and system_ram_gib > 0
    ):
        total_ram_mib = (
            system_ram_gib
            * 1024.0
        )

        max_peak_fraction_ram = (
            max_memory_stage[
                "peak_rss_mib"
            ]
            / total_ram_mib
        )

    if (
        max_peak_fraction_ram
        is not None
        and max_peak_fraction_ram
        < 0.25
    ):
        memory_assessment = (
            "Individual measured stage peaks are "
            "well below total system RAM. The current "
            "CPU baseline appears more compute/latency "
            "limited than RAM-capacity limited. This "
            "does not represent simultaneous concurrent "
            "peak memory."
        )
    else:
        memory_assessment = (
            "Memory should be considered together "
            "with latency when selecting deployment "
            "hardware."
        )

    # =====================================================
    # Diagnostic LightGlue
    # =====================================================

    lg_row = timing_lookup(
        timing,
        "lightglue_reranking",
    )

    lightglue_diagnostic = {
        "included_in_promoted_pipeline": False,
        "role": "diagnostic_optional",
        "runtime_s": (
            finite_float(
                lg_row.get(
                    "runtime_s"
                )
            )
            if lg_row
            else None
        ),
        "ms_per_query": (
            finite_float(
                lg_row.get(
                    "ms_per_query"
                )
            )
            if lg_row
            else None
        ),
        "peak_rss_mib": memory_peak(
            resource,
            "lightglue_reranking",
        ),
        "memory_status": (
            resource
            .get(
                "stage_memory",
                {}
            )
            .get(
                "lightglue_reranking",
                {}
            )
            .get(
                "status"
            )
        ),
        "interpretation": (
            "LightGlue is retained as a diagnostic "
            "comparison, not part of the promoted "
            "CPU deployment path."
        ),
    }

    # =====================================================
    # FLOPS / TOPS decision
    # =====================================================

    theoretical_compute = {
        "required_for_current_cpu_baseline": False,
        "status": (
            "NOT_REQUIRED_FOR_CURRENT_DEPLOYMENT_BASELINE"
        ),
        "reason": (
            "Measured wall-time, normalized latency, "
            "throughput, memory, workload and cache "
            "reuse directly characterize the current "
            "CPU implementation. Theoretical FLOPS/TOPS "
            "alone would not predict this mixed "
            "DINO/XFeat/ORB pipeline latency."
        ),
        "add_later_when": (
            "Comparing candidate embedded GPUs/NPUs "
            "or quantized model variants."
        ),
        "future_metrics": [
            "model MACs/FLOPs",
            "FP32/FP16/INT8 precision",
            "target accelerator TOPS",
            "memory bandwidth",
            "measured accelerator latency",
            "power consumption if deployment hardware is available",
        ],
    }

    # =====================================================
    # Cache/storage context
    # =====================================================

    cache_context = {
        "query_descriptor_cache_mib": (
            finite_float(
                cache[
                    "query_descriptor_cache"
                ].get(
                    "size_mib"
                )
            )
        ),
        "map_descriptor_cache_mib": (
            finite_float(
                cache[
                    "map_descriptor_cache"
                ].get(
                    "size_mib"
                )
            )
        ),
        "tile_folder_mib": (
            finite_float(
                cache[
                    "tile_folder"
                ].get(
                    "size_mib"
                )
            )
        ),
        "dataset_output_folder_mib": (
            finite_float(
                cache[
                    "dataset_output_folder"
                ].get(
                    "size_mib"
                )
            )
        ),
    }

    # =====================================================
    # Final report
    # =====================================================

    status = (
        "PASS_PARTIAL_DEPLOYMENT_COST_BREAKDOWN"
        if (
            offline_pending
            or per_flight_pending
        )
        else "PASS_DEPLOYMENT_COST_BREAKDOWN"
    )

    report = {
        "addon": (
            "Add-on 6 — Offline vs online "
            "cost breakdown"
        ),
        "status": status,
        "interpretation_rule": (
            INTERPRETATION_RULE
        ),
        "source_artifacts": {
            "timing_summary_csv": (
                str(timing_csv)
            ),
            "timing_summary_json": (
                str(timing_json)
            ),
            "resource_summary_json": (
                str(resource_json)
            ),
            "cache_size_summary_json": (
                str(cache_json)
            ),
        },
        "timing_registry_status": (
            timing_report.get(
                "status"
            )
        ),
        "offline_one_time": {
            "rows": offline_rows,
            "measured_total_s": (
                measured_offline_total_s
            ),
            "measured_total_human": (
                fmt_seconds(
                    measured_offline_total_s
                )
            ),
            "not_timed_stages": (
                offline_pending
            ),
        },
        "per_new_flight": {
            "rows": per_flight_rows,
            "measured_total_s": (
                measured_per_flight_total_s
            ),
            "measured_total_human": (
                fmt_seconds(
                    measured_per_flight_total_s
                )
            ),
            "pending_stages": (
                per_flight_pending
            ),
            "important_note": (
                "This total sums only measured parent "
                "pipeline stages and deliberately "
                "excludes XFeat substage rows to avoid "
                "double counting. Pending extraction/"
                "export stages are not guessed."
            ),
        },
        "online_like": {
            "rows": online_rows,
            "serial_equivalent_latency_ms": (
                serial_online_ms
            ),
            "serial_equivalent_latency_s": (
                serial_online_ms
                / 1000.0
            ),
            "serial_equivalent_capacity_hz": (
                serial_online_hz
            ),
            "nominal_input_query_rate_hz": (
                target_rate_hz
            ),
            "nominal_query_budget_ms": (
                target_budget_ms
            ),
            "serial_realtime_factor": (
                realtime_factor
            ),
            "nominal_rate_met": (
                target_realtime_met
            ),
            "important_note": (
                "The serial-equivalent number is the "
                "sum of independently measured "
                "normalized stage latencies. It is a "
                "deployment planning estimate, not a "
                "claim that the current batch scripts "
                "execute as a real-time pipelined "
                "system. Parallelism, batching and "
                "stage overlap are not assumed."
            ),
        },
        "optimization_priority": (
            optimization_rows
        ),
        "memory_assessment": {
            "system_ram_gib": (
                system_ram_gib
            ),
            "largest_measured_stage_peak": (
                max_memory_stage
            ),
            "largest_peak_fraction_of_system_ram": (
                max_peak_fraction_ram
            ),
            "assessment": (
                memory_assessment
            ),
        },
        "cache_context": (
            cache_context
        ),
        "lightglue_diagnostic": (
            lightglue_diagnostic
        ),
        "theoretical_compute_metrics": (
            theoretical_compute
        ),
    }

    json_path = (
        metrics_dir
        / "deployment_cost_breakdown.json"
    )

    md_path = (
        metrics_dir
        / "deployment_cost_breakdown.md"
    )

    json_path.write_text(
        json.dumps(
            report,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    # =====================================================
    # Markdown
    # =====================================================

    lines = [
        "# Add-on 6 — Deployment Cost Breakdown",
        "",
        f"- Status: `{status}`",
        (
            "- Timing registry: "
            f"`{timing_report.get('status')}`"
        ),
        (
            "- Nominal sampled-query rate used "
            f"for interpretation: `{target_rate_hz:g} Hz`"
        ),
        "",
        "## Interpretation rule",
        "",
        f"> {INTERPRETATION_RULE}",
        "",
        "## 1. Offline / one-time map preparation",
        "",
        "| Stage | Status | Measured runtime | Reusable? |",
        "|---|---|---:|---|",
    ]

    for row in offline_rows:
        lines.append(
            "| "
            f"{row['label']} | "
            f"{row['status']} | "
            f"{fmt_seconds(row['runtime_s'])} | "
            f"{'yes' if row['reusable'] else 'no'} |"
        )

    lines.extend(
        [
            "",
            (
                "**Measured offline total:** "
                f"{fmt_seconds(measured_offline_total_s)}."
            ),
            "",
            (
                "The total above is partial because "
                "orthophoto crop, tile generation and "
                "tile-index construction were already "
                "available and have no recorded timing."
            ),
            "",
            "## 2. Per-new-flight processing",
            "",
            "| Stage | Status | Measured runtime |",
            "|---|---|---:|",
        ]
    )

    for row in per_flight_rows:
        lines.append(
            "| "
            f"{row['label']} | "
            f"{row['status']} | "
            f"{fmt_seconds(row['runtime_s'])} |"
        )

    lines.extend(
        [
            "",
            (
                "**Measured per-flight total:** "
                f"{fmt_seconds(measured_per_flight_total_s)}."
            ),
            "",
            (
                "This deliberately excludes XFeat "
                "feature/matching subrows because they "
                "are already included in the full XFeat "
                "stage wall time."
            ),
            "",
            "## 3. Online-like normalized cost",
            "",
            (
                "| Stage | Normalization | Latency | "
                "Equivalent rate | Peak RSS |"
            ),
            "|---|---|---:|---:|---:|",
        ]
    )

    for row in online_rows:
        rate = row[
            "equivalent_rate_hz"
        ]

        rate_text = (
            f"{rate:.3f} Hz"
            if rate is not None
            else "not measured"
        )

        lines.append(
            "| "
            f"{row['label']} | "
            f"{row['normalized_unit']} | "
            f"{fmt_ms(row['latency_ms'])} | "
            f"{rate_text} | "
            f"{fmt_mib(row['peak_rss_mib'])} |"
        )

    lines.extend(
        [
            "",
            (
                "**Serial-equivalent online-like "
                f"latency:** `{serial_online_ms:.2f} ms` "
                f"(`{serial_online_ms / 1000.0:.3f} s`)."
            ),
            (
                "**Serial-equivalent capacity:** "
                f"`{serial_online_hz:.3f} Hz`."
                if serial_online_hz is not None
                else
                "**Serial-equivalent capacity:** "
                "not available."
            ),
            (
                "**Nominal 1-sampled-query budget:** "
                f"`{target_budget_ms:.2f} ms`."
                if target_budget_ms is not None
                else
                "**Nominal query budget:** unavailable."
            ),
            (
                "**Serial realtime factor:** "
                f"`{realtime_factor:.2f}×`."
                if realtime_factor is not None
                else
                "**Serial realtime factor:** unavailable."
            ),
            "",
            (
                "This serial-equivalent value is a "
                "planning estimate from independently "
                "measured stage latencies. It does not "
                "assume parallel execution, batching, "
                "or asynchronous pipelining."
            ),
            "",
            "## 4. Optimization priority",
            "",
            (
                "| Rank | Stage | Latency | Share of "
                "serial online-like cost | Priority |"
            ),
            "|---:|---|---:|---:|---|",
        ]
    )

    for rank, row in enumerate(
        optimization_rows,
        start=1,
    ):
        lines.append(
            "| "
            f"{rank} | "
            f"{row['label']} | "
            f"{row['latency_ms']:.2f} ms | "
            f"{row['serial_latency_percent']:.1f}% | "
            f"{row['optimization_priority']} |"
        )

    lines.extend(
        [
            "",
            "### Engineering interpretation",
            "",
            (
                "The first optimization effort should "
                "target the highest-latency stages above, "
                "rather than the already-fast temporal "
                "gate, fusion bookkeeping, or cached "
                "DINO ranking."
            ),
            "",
            (
                "Likely optimization directions include "
                "hardware acceleration for DINO query "
                "encoding, reducing image resolution or "
                "using a lighter/quantized descriptor "
                "model, reducing/adapting ORB Top-K, "
                "caching reusable image features, and "
                "pipelining independent work where the "
                "algorithm permits it. These are "
                "deployment directions, not changes to "
                "the frozen localization result."
            ),
            "",
            "## 5. Memory/resource interpretation",
            "",
            (
                f"- System RAM: `{system_ram_gib} GiB`."
            ),
        ]
    )

    if max_memory_stage:
        lines.append(
            "- Largest measured individual stage peak: "
            f"`{max_memory_stage['stage']}` at "
            f"`{max_memory_stage['peak_rss_mib']:.1f} MiB`."
        )

    lines.extend(
        [
            f"- {memory_assessment}",
            "",
            "Important: these are individual stage "
            "peak measurements, not a simultaneous "
            "multi-stage/concurrent pipeline peak.",
            "",
            "## 6. Cache reuse",
            "",
            (
                "- Query descriptor cache: "
                f"`{cache_context['query_descriptor_cache_mib']:.3f} MiB`."
            ),
            (
                "- Map descriptor cache: "
                f"`{cache_context['map_descriptor_cache_mib']:.3f} MiB`."
            ),
            (
                "- Tile folder: "
                f"`{cache_context['tile_folder_mib']:.1f} MiB`."
            ),
            "",
            (
                "The map descriptor cache and map tiles "
                "are reusable across flights when the AOI, "
                "tile variant and descriptor model remain "
                "unchanged. Query descriptors must be "
                "regenerated for a new flight."
            ),
            "",
            "## 7. LightGlue diagnostic",
            "",
            (
                f"- Full diagnostic runtime: "
                f"`{fmt_seconds(lightglue_diagnostic['runtime_s'])}`."
            ),
            (
                f"- Per-query diagnostic latency: "
                f"`{fmt_ms(lightglue_diagnostic['ms_per_query'])}`."
            ),
            (
                "- LightGlue is not part of the promoted "
                "CPU deployment path."
            ),
            "",
            "## 8. Do we need FLOPS/TOPS now?",
            "",
            "**No for the current CPU baseline.**",
            "",
            (
                "Measured latency, throughput, RAM, "
                "workload and cache reuse are the relevant "
                "numbers for the current laptop/demo."
            ),
            "",
            (
                "FLOPS/TOPS should be added when comparing "
                "specific embedded GPU/NPU targets. At "
                "that stage the comparison should include "
                "model MACs/FLOPs, precision "
                "(FP32/FP16/INT8), accelerator TOPS, "
                "memory bandwidth, measured target-device "
                "latency and ideally power."
            ),
            "",
            "## 9. Known incomplete timings",
            "",
            "Offline timings not recorded:",
        ]
    )

    for stage in offline_pending:
        lines.append(
            f"- `{stage}`"
        )

    lines.extend(
        [
            "",
            "Per-flight timings pending:",
        ]
    )

    for stage in per_flight_pending:
        lines.append(
            f"- `{stage}`"
        )

    lines.extend(
        [
            "",
            (
                "These remain explicit pending values "
                "and will be filled as the later blind "
                "manifest/export/environment add-ons are "
                "implemented."
            ),
            "",
        ]
    )

    md_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    # =====================================================
    # Console
    # =====================================================

    print("=" * 80)
    print(
        "STAGE 7 — ADD-ON 6 "
        "DEPLOYMENT COST BREAKDOWN"
    )
    print("=" * 80)

    print(
        f"status: {status}"
    )

    print()
    print("Offline / one-time")
    print("-" * 80)

    for row in offline_rows:
        print(
            f"{row['label']:38s} "
            f"{row['status']:12s} "
            f"{fmt_seconds(row['runtime_s'])}"
        )

    print()
    print(
        "Measured offline total:",
        fmt_seconds(
            measured_offline_total_s
        ),
    )

    print()
    print("Per-new-flight")
    print("-" * 80)

    for row in per_flight_rows:
        print(
            f"{row['label']:38s} "
            f"{row['status']:10s} "
            f"{fmt_seconds(row['runtime_s'])}"
        )

    print()
    print(
        "Measured per-flight total:",
        fmt_seconds(
            measured_per_flight_total_s
        ),
    )

    print()
    print("Online-like")
    print("-" * 80)

    for row in online_rows:
        print(
            f"{row['label']:32s} "
            f"{fmt_ms(row['latency_ms']):>14s} "
            f"{fmt_mib(row['peak_rss_mib']):>16s}"
        )

    print()
    print(
        "Serial-equivalent latency:",
        f"{serial_online_ms:.2f} ms",
    )

    if serial_online_hz is not None:
        print(
            "Serial-equivalent capacity:",
            f"{serial_online_hz:.3f} Hz",
        )

    if realtime_factor is not None:
        print(
            "Realtime factor at "
            f"{target_rate_hz:g} Hz input:",
            f"{realtime_factor:.2f}x",
        )

    print()
    print("Optimization priority")
    print("-" * 80)

    for rank, row in enumerate(
        optimization_rows,
        start=1,
    ):
        print(
            f"{rank}. "
            f"{row['label']:30s} "
            f"{row['latency_ms']:9.2f} ms "
            f"{row['serial_latency_percent']:6.1f}% "
            f"{row['optimization_priority']}"
        )

    print()
    print("Saved")
    print("-" * 80)
    print(json_path)
    print(md_path)


if __name__ == "__main__":
    main()
