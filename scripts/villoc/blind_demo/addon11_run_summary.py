'''
1. without eval toggle, only blind test summary:

ROOT=outputs/villoc/traj01_90deg_stable120m
MAP_ROOT=outputs/villoc/90_deg
RUN="$ROOT/blind_demo_addons/eval_traj01_known_gt"

python scripts/villoc/blind_demo/addon11_run_summary.py \
  --root "$ROOT" \
  --map-root "$MAP_ROOT" \
  --run-root "$RUN" \
  2>&1 | tee \
  "$RUN/logs/stage12b_addon11_blind_summary.log"

2. with evaluation toggle:

python scripts/villoc/blind_demo/addon11_run_summary.py \
  --root "$ROOT" \
  --map-root "$MAP_ROOT" \
  --run-root "$RUN" \
  --include-evaluation \
  2>&1 | tee \
  "$RUN/logs/stage12b_addon11_evaluated_summary.log"

'''

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


BLIND_NOTE = (
    "Reference unavailable: accuracy metrics not computed."
)

LATLON_NOTE = (
    "Estimated latitude/longitude are visual map-matching "
    "outputs, not GPS inputs."
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load_json(path: Path, required: bool = True) -> dict[str, Any] | None:
    if not path.exists():
        if required:
            raise FileNotFoundError(path)
        return None

    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


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
        .isin({
            "true",
            "1",
            "yes",
            "y",
            "t",
        })
    )


def first_value(
    mapping: dict[str, Any] | None,
    keys: list[str],
    default: Any = None,
) -> Any:

    if not isinstance(mapping, dict):
        return default

    for key in keys:
        if key in mapping:
            value = mapping[key]

            if value is not None:
                return value

    return default


def recursive_find(
    obj: Any,
    candidate_keys: set[str],
) -> Any:

    if isinstance(obj, dict):

        for key, value in obj.items():
            if key in candidate_keys and value is not None:
                return value

        for value in obj.values():
            found = recursive_find(
                value,
                candidate_keys,
            )

            if found is not None:
                return found

    elif isinstance(obj, list):

        for value in obj:
            found = recursive_find(
                value,
                candidate_keys,
            )

            if found is not None:
                return found

    return None


def finite_float(value: Any) -> float | None:
    try:
        x = float(value)
    except Exception:
        return None

    return x if math.isfinite(x) else None


def fmt_number(
    value: Any,
    decimals: int = 3,
    suffix: str = "",
) -> str:

    value = finite_float(
        value
    )

    if value is None:
        return "n/a"

    return f"{value:.{decimals}f}{suffix}"


def fmt_int(value: Any) -> str:
    try:
        return str(
            int(value)
        )
    except Exception:
        return "n/a"


def rel_link(
    summary_dir: Path,
    target: Path,
) -> str:

    return Path(
        os.path.relpath(
            target,
            start=summary_dir,
        )
    ).as_posix()


def md_image(
    summary_dir: Path,
    target: Path,
    alt: str,
) -> str | None:

    if not target.exists():
        return None

    return (
        f"![{alt}]"
        f"({rel_link(summary_dir, target)})"
    )


def md_file_link(
    summary_dir: Path,
    target: Path,
    label: str,
) -> str | None:

    if not target.exists():
        return None

    return (
        f"[{label}]"
        f"({rel_link(summary_dir, target)})"
    )


def runtime_matches(
    rows: list[dict[str, Any]],
    stage_terms: list[str],
    component_terms: list[str],
) -> dict[str, Any] | None:

    for row in rows:

        if (
            str(
                row.get(
                    "measurement_status",
                    ""
                )
            ).upper()
            != "MEASURED"
        ):
            continue

        stage = str(
            row.get(
                "stage",
                ""
            )
        ).lower()

        component = str(
            row.get(
                "component",
                ""
            )
        ).lower()

        if any(
            term.lower() == stage
            for term in stage_terms
        ):
            return row

        if all(
            term.lower() in component
            for term in component_terms
        ):
            return row

    return None


def runtime_table_rows(
    timing_summary: dict[str, Any],
) -> list[tuple[str, dict[str, Any]]]:

    rows = timing_summary.get(
        "rows",
        [],
    )

    specs = [
        (
            "Relative frontend — XFeat",
            [
                "relative_odometry",
                "relative_frontend",
            ],
            [
                "xfeat",
            ],
        ),
        (
            "DINO query encoding",
            [
                "dino_query_descriptor_encoding",
                "dino_query_encoding",
            ],
            [
                "dino",
                "query",
            ],
        ),
        (
            "DINO cached retrieval",
            [
                "dino_retrieval_against_map_cache",
                "dino_cached_retrieval",
                "dino_retrieval",
            ],
            [
                "dino",
                "retrieval",
            ],
        ),
        (
            "ORB Top-20 verification/reranking",
            [
                "orb_top20_reranking",
                "orb_reranking",
            ],
            [
                "orb",
            ],
        ),
        (
            "Temporal gating",
            [
                "temporal_gating",
            ],
            [
                "temporal",
            ],
        ),
        (
            "Fusion",
            [
                "fusion_replay",
                "fusion",
            ],
            [
                "fusion",
            ],
        ),
    ]

    selected = []

    used_ids = set()

    for label, stages, component_terms in specs:

        row = runtime_matches(
            rows,
            stages,
            component_terms,
        )

        if row is None:
            continue

        row_id = (
            row.get(
                "stage"
            ),
            row.get(
                "component"
            ),
        )

        if row_id in used_ids:
            continue

        used_ids.add(
            row_id
        )

        selected.append(
            (
                label,
                row,
            )
        )

    return selected


def memory_entries(
    resource: dict[str, Any],
) -> list[tuple[str, float]]:

    stage_memory = resource.get(
        "stage_memory",
        {}
    )

    entries: list[
        tuple[str, Any]
    ] = []

    if isinstance(
        stage_memory,
        dict,
    ):
        entries.extend(
            stage_memory.items()
        )

    elif isinstance(
        stage_memory,
        list,
    ):
        for i, item in enumerate(
            stage_memory
        ):
            if isinstance(
                item,
                dict,
            ):
                label = str(
                    first_value(
                        item,
                        [
                            "stage",
                            "component",
                            "name",
                            "label",
                        ],
                        f"stage_{i}",
                    )
                )

                entries.append(
                    (
                        label,
                        item,
                    )
                )

    results = []

    for outer_label, value in entries:

        label = str(
            outer_label
        )

        if isinstance(
            value,
            dict,
        ):
            label = str(
                first_value(
                    value,
                    [
                        "stage",
                        "component",
                        "name",
                        "label",
                    ],
                    label,
                )
            )

        peak = recursive_find(
            value,
            {
                "peak_process_tree_rss_mib",
                "peak_root_rss_mib",
                "peak_rss_mib",
                "peak_memory_mib",
                "peak_rss_mb",
                "peak_memory_mb",
            },
        )

        peak = finite_float(
            peak
        )

        if peak is None:
            continue

        low = label.lower()

        if any(
            token in low
            for token in [
                "xfeat",
                "relative",
                "dino",
                "retrieval",
                "orb",
                "temporal",
                "fusion",
            ]
        ):
            label_aliases = {
                "relative_odometry":
                    "Relative frontend — XFeat",

                "dino_query_descriptor_extraction":
                    "DINO query encoding",

                "dino_retrieval_against_map_cache":
                    "DINO cached retrieval",

                "orb_topk_reranking":
                    "ORB Top-20 verification/reranking",

                "temporal_gating":
                    "Temporal gating",

                "fusion_replay":
                    "Fusion",
            }

            results.append(
                (
                    label_aliases.get(
                        label,
                        label,
                    ),
                    peak,
                )
            )

    return results


def threshold_count(
    threshold_summary: dict[str, Any] | None,
    method: str,
    threshold: float,
) -> tuple[int, int] | None:

    if not threshold_summary:
        return None

    methods = threshold_summary.get(
        "methods",
        {},
    )

    data = methods.get(
        method
    )

    if not isinstance(
        data,
        dict,
    ):
        return None

    tag = f"{threshold:g}"

    counts = data.get(
        "counts",
        {},
    )

    population = data.get(
        "population"
    )

    if (
        tag not in counts
        or population is None
    ):
        return None

    return (
        int(
            counts[tag]
        ),
        int(
            population
        ),
    )


def write_section(
    lines: list[str],
    title: str,
) -> None:

    lines.extend([
        "",
        f"## {title}",
        "",
    ])


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Add-on 11: generate one concise Markdown "
            "summary for a blind Villoc localization run."
        )
    )

    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Dataset output root.",
    )

    parser.add_argument(
        "--run-root",
        type=Path,
        required=True,
        help="Blind add-on run root.",
    )

    parser.add_argument(
        "--map-root",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--include-evaluation",
        action="store_true",
        help=(
            "Include only already-frozen post-run "
            "evaluation results. Without this flag, "
            "no evaluation artifact is read."
        ),
    )

    args = parser.parse_args()

    started = time.perf_counter()

    root = args.root.expanduser().resolve()
    run_root = (
        args.run_root
        .expanduser()
        .resolve()
    )

    map_root = (
        args.map_root
        .expanduser()
        .resolve()
        if args.map_root
        else None
    )

    summary_dir = (
        run_root
        / "summary"
    )

    report_dir = (
        run_root
        / "reports"
        / "addon11_run_summary"
    )

    summary_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    md_path = (
        summary_dir
        / "demo_run_summary.md"
    )

    report_path = (
        report_dir
        / "run_summary_report.json"
    )

    # =====================================================
    # Blind-only inputs.
    # =====================================================

    manifest_report_path = (
        run_root
        / "reports"
        / "blind_query_manifest_report.json"
    )

    manifest_path = (
        run_root
        / "metadata"
        / "blind_query_manifest.csv"
    )

    bootstrap_report_path = (
        run_root
        / "reports"
        / "blind_map_bootstrap"
        / "blind_map_bootstrap_report.json"
    )

    map_alignment_report_path = (
        run_root
        / "reports"
        / "blind_map_alignment"
        / "blind_map_alignment_report.json"
    )

    temporal_report_path = (
        run_root
        / "reports"
        / "blind_temporal_fusion"
        / "blind_temporal_fusion_report.json"
    )

    temporal_manifest_path = (
        run_root
        / "metadata"
        / "blind_temporal_fusion"
        / "blind_temporal_correction_manifest.csv"
    )

    relative_path = (
        run_root
        / "trajectories"
        / "blind_map_aligned_relative_trajectory.csv"
    )

    submission_path = (
        run_root
        / "trajectories"
        / "submission_estimated_trajectory.csv"
    )

    addon9_path = (
        run_root
        / "reports"
        / "addon9_estimated_latlon"
        / "estimated_latlon_export_report.json"
    )

    addon10_path = (
        run_root
        / "reports"
        / "addon10_no_reference_visuals"
        / "no_reference_visuals_report.json"
    )

    timing_path = (
        run_root
        / "metrics"
        / "timing_summary.json"
    )

    resource_path = (
        run_root
        / "metrics"
        / "resource_summary.json"
    )

    cache_path = (
        run_root
        / "metrics"
        / "cache_size_summary.json"
    )

    deployment_path = (
        run_root
        / "metrics"
        / "deployment_cost_breakdown.json"
    )

    required_blind_paths = [
        manifest_report_path,
        manifest_path,
        bootstrap_report_path,
        map_alignment_report_path,
        temporal_report_path,
        temporal_manifest_path,
        relative_path,
        submission_path,
        addon9_path,
        addon10_path,
        timing_path,
        resource_path,
        deployment_path,
    ]

    for path in required_blind_paths:
        require(
            path.exists(),
            f"Missing blind summary input: {path}",
        )

    submission_sha_before = (
        sha256_file(
            submission_path
        )
    )

    manifest_report = load_json(
        manifest_report_path
    )

    bootstrap = load_json(
        bootstrap_report_path
    )

    map_alignment = load_json(
        map_alignment_report_path
    )

    temporal_report = load_json(
        temporal_report_path
    )

    addon9 = load_json(
        addon9_path
    )

    addon10 = load_json(
        addon10_path
    )

    timing = load_json(
        timing_path
    )

    resource = load_json(
        resource_path
    )

    cache = load_json(
        cache_path,
        required=False,
    )

    deployment = load_json(
        deployment_path
    )

    manifest = pd.read_csv(
        manifest_path
    )

    relative = pd.read_csv(
        relative_path
    )

    submission = pd.read_csv(
        submission_path
    )

    temporal_manifest = pd.read_csv(
        temporal_manifest_path
    )

    require(
        len(manifest) == 403,
        f"Expected 403 blind manifest rows, got {len(manifest)}.",
    )

    require(
        len(submission) == 403,
        f"Expected 403 submission rows, got {len(submission)}.",
    )

    reference_available = (
        bool_series(
            manifest[
                "reference_available"
            ]
        )
        if "reference_available"
        in manifest.columns
        else pd.Series(
            False,
            index=manifest.index,
        )
    )

    require(
        not bool(
            reference_available.any()
        ),
        (
            "Blind manifest unexpectedly reports "
            "reference_available=True."
        ),
    )

    # =====================================================
    # Optional evaluation — read ONLY if explicitly enabled.
    # =====================================================

    evaluation_included = False

    evaluation_summary = None
    drift_summary = None
    threshold_summary = None
    safety_summary = None

    freeze_path = (
        run_root
        / "evaluation"
        / "blind_submission_freeze.json"
    )

    evaluation_path = (
        run_root
        / "evaluation"
        / "blind_submission_evaluation_summary.json"
    )

    drift_path = (
        run_root
        / "metrics"
        / "drift_time_summary.json"
    )

    threshold_path = (
        run_root
        / "metrics"
        / "threshold_sensitivity.json"
    )

    safety_path = (
        run_root
        / "metrics"
        / "accepted_correction_safety_summary.json"
    )

    if args.include_evaluation:

        for path in [
            freeze_path,
            evaluation_path,
            drift_path,
            threshold_path,
            safety_path,
        ]:
            require(
                path.exists(),
                (
                    "Evaluation mode requested but "
                    f"artifact missing: {path}"
                ),
            )

        freeze = load_json(
            freeze_path
        )

        require(
            freeze.get(
                "status"
            )
            == "PASS_BLIND_SUBMISSION_FROZEN",
            (
                "Evaluation requested but freeze "
                "record is not PASS."
            ),
        )

        evaluation_summary = load_json(
            evaluation_path
        )

        require(
            evaluation_summary.get(
                "status"
            )
            == (
                "PASS_FROZEN_BLIND_SUBMISSION_EVALUATION"
            ),
            (
                "Evaluation summary is not from a "
                "passing frozen-blind evaluation."
            ),
        )

        drift_summary = load_json(
            drift_path
        )

        threshold_summary = load_json(
            threshold_path
        )

        safety_summary = load_json(
            safety_path
        )

        evaluation_included = True

    # =====================================================
    # Blind run facts.
    # =====================================================

    source_video = (
        str(
            manifest[
                "source_video"
            ].iloc[
                0
            ]
        )
        if "source_video"
        in manifest.columns
        else "n/a"
    )

    timestamp = pd.to_numeric(
        manifest[
            "timestamp_s"
        ],
        errors="coerce",
    )

    map_available = bool_series(
        submission[
            "map_aligned_available"
        ]
    )

    accepted = bool_series(
        submission[
            "accepted_correction"
        ]
    )

    map_lock = bool_series(
        submission[
            "map_lock_event"
        ]
    )

    correction_candidate = bool_series(
        temporal_manifest[
            "correction_candidate"
        ]
    )

    correction_accepted = bool_series(
        temporal_manifest[
            "correction_accepted"
        ]
    )

    estimated_lat = pd.to_numeric(
        submission[
            "estimated_lat"
        ],
        errors="coerce",
    )

    estimated_lon = pd.to_numeric(
        submission[
            "estimated_lon"
        ],
        errors="coerce",
    )

    latlon_available = (
        estimated_lat.notna()
        & estimated_lon.notna()
    )

    confidence = pd.to_numeric(
        submission[
            "confidence_score"
        ],
        errors="coerce",
    )

    lock_row = (
        submission.loc[
            map_lock
        ].iloc[
            0
        ]
        if map_lock.any()
        else None
    )

    # =====================================================
    # Markdown.
    # =====================================================

    lines: list[str] = []

    lines.extend([
        "# Blind Visual Localization Run Summary",
        "",
        f"- **Dataset:** `{root.name}`",
        f"- **Run ID:** `{run_root.name}`",
        "- **Primary mode:** blind / no-reference localization",
        (
            "- **Evaluation attachment:** "
            + (
                "included after frozen blind output"
                if evaluation_included
                else "not included"
            )
        ),
        f"- **Status:** PASS",
        "",
        (
            "> "
            + (
                "The localization result is produced without "
                "SRT/GPS/ground truth. "
                + (
                    "Evaluation metrics below were attached "
                    "only after the blind trajectory was frozen."
                    if evaluation_included
                    else BLIND_NOTE
                )
            )
        ),
    ])

    # 1
    write_section(
        lines,
        "1. Dataset summary",
    )

    lines.extend([
        f"- Blind query frames: **{len(manifest)}**",
        (
            "- Video-time range: "
            f"**{fmt_number(timestamp.min(), 3)} s** "
            "to "
            f"**{fmt_number(timestamp.max(), 3)} s**"
        ),
        f"- Source video: `{source_video}`",
        (
            "- Assumed relative altitude: "
            f"**{fmt_number(manifest['assumed_rel_alt_m'].median(), 2)} m**"
            if "assumed_rel_alt_m" in manifest.columns
            else "- Assumed relative altitude: n/a"
        ),
        (
            "- Assumed gimbal pitch: "
            f"**{fmt_number(manifest['assumed_gimbal_pitch_deg'].median(), 2)}°**"
            if "assumed_gimbal_pitch_deg" in manifest.columns
            else "- Assumed gimbal pitch: n/a"
        ),
    ])

    # 2
    write_section(
        lines,
        "2. Run mode",
    )

    lines.extend([
        "- Localization inputs: camera imagery + prepared georeferenced map.",
        "- SRT/GPS/reference used by localization: **No**.",
        (
            "- Post-run evaluation enabled: "
            f"**{'Yes' if evaluation_included else 'No'}**."
        ),
        (
            "- Blind map coordinates become available only "
            "after causal map initialization."
        ),
    ])

    # 3
    write_section(
        lines,
        "3. Input files",
    )

    for label, path in [
        (
            "Blind query manifest",
            manifest_path,
        ),
        (
            "Blind map-aligned relative trajectory",
            relative_path,
        ),
        (
            "Blind temporal correction manifest",
            temporal_manifest_path,
        ),
        (
            "Frozen estimated trajectory",
            submission_path,
        ),
    ]:

        link = md_file_link(
            summary_dir,
            path,
            label,
        )

        if link:
            lines.append(
                f"- {link}"
            )

    # 4
    write_section(
        lines,
        "4. Map/AOI/cache reuse status",
    )

    lines.extend([
        (
            "- Map alignment source: "
            f"`{first_value(map_alignment, ['important_note'], 'blind map bootstrap')}`"
        ),
        (
            "- Prepared map root: "
            f"`{map_root}`"
            if map_root
            else "- Prepared map root: configured externally"
        ),
        (
            "- Runtime registry: "
            f"**{timing.get('measured_rows', 'n/a')} measured**, "
            f"**{timing.get('pending_rows', 'n/a')} pending**."
        ),
        (
            "- Map/cache reuse is treated separately from "
            "per-flight localization computation."
        ),
    ])

    # 5
    write_section(
        lines,
        "5. Relative localization summary",
    )

    postlock_distance = (
        pd.to_numeric(
            relative.loc[
                bool_series(
                    relative[
                        "map_aligned_available"
                    ]
                ),
                "relative_cumulative_distance_m",
            ],
            errors="coerce",
        )
        if "relative_cumulative_distance_m"
        in relative.columns
        else pd.Series(
            dtype=float
        )
    )

    lines.extend([
        (
            "- Map-aligned relative poses: "
            f"**{int(map_available.sum())}/{len(submission)}**."
        ),
        (
            "- Pre-lock poses without geographic output: "
            f"**{int((~map_available).sum())}**."
        ),
        (
            "- Map lock: "
            + (
                f"frame **{int(lock_row['sequence_frame_id'])}**, "
                f"time **{float(lock_row['timestamp_s']):.3f} s**."
                if lock_row is not None
                else "**not acquired**."
            )
        ),
        (
            "- Post-lock relative path length: "
            + (
                f"**{float(postlock_distance.max()):.2f} m**."
                if len(postlock_distance.dropna())
                else "n/a."
            )
        ),
    ])

    # 6
    write_section(
        lines,
        "6. Absolute retrieval/reranking summary",
    )

    finite_conf = confidence.dropna()

    lines.extend([
        (
            "- Absolute retrieval evidence is produced by "
            "DINOv2 and geometrically checked/reranked by ORB."
        ),
        (
            "- Queries with DINO/ORB output: "
            f"**{int(submission['orb_selected_tile_id'].notna().sum())}/{len(submission)}**."
        ),
        (
            "- Hybrid score median: "
            f"**{fmt_number(finite_conf.median(), 3)}** "
            "(ranking/verifier score; not a calibrated probability)."
        ),
        (
            "- Top-K correctness thresholds are not used "
            "during blind localization; they are evaluation-only."
        ),
    ])

    # 7
    write_section(
        lines,
        "7. Fusion/correction summary",
    )

    reason_counts = (
        temporal_manifest.loc[
            correction_candidate,
            "correction_reason",
        ]
        .fillna(
            "UNKNOWN"
        )
        .astype(str)
        .value_counts()
        .to_dict()
    )

    lines.extend([
        (
            "- Blind correction candidates: "
            f"**{int(correction_candidate.sum())}**."
        ),
        (
            "- Accepted corrections: "
            f"**{int(correction_accepted.sum())}**."
        ),
        (
            "- Rejected candidates: "
            f"**{int((correction_candidate & ~correction_accepted).sum())}**."
        ),
        (
            "- Fusion policy: continuous relative propagation "
            "with sparse confidence/temporal-gated absolute corrections."
        ),
        (
            "- Decision reasons: "
            + ", ".join(
                f"`{k}`={v}"
                for k, v in reason_counts.items()
            )
            + "."
        ),
    ])

    # 8
    write_section(
        lines,
        "8. Estimated latitude/longitude export",
    )

    lines.extend([
        (
            "- Estimated geographic poses: "
            f"**{int(latlon_available.sum())}/{len(submission)}**."
        ),
        (
            "- Latitude range: "
            f"**{fmt_number(estimated_lat.min(), 8)}** "
            "to "
            f"**{fmt_number(estimated_lat.max(), 8)}**."
        ),
        (
            "- Longitude range: "
            f"**{fmt_number(estimated_lon.min(), 8)}** "
            "to "
            f"**{fmt_number(estimated_lon.max(), 8)}**."
        ),
        f"- **{LATLON_NOTE}**",
    ])

    # 9
    write_section(
        lines,
        "9. Core runtime and resource summary",
    )

    lines.extend([
        (
            "The table below intentionally contains only "
            "**localization-critical computation**. Plot generation, "
            "Folium rendering, CSV writing, and summary-generation "
            "overhead are excluded from this headline view."
        ),
        "",
        "| Core stage | Measured runtime | Normalized cost | Scope |",
        "|---|---:|---:|---|",
    ])

    core_runtime = runtime_table_rows(
        timing
    )

    for label, row in core_runtime:

        runtime_s = finite_float(
            row.get(
                "runtime_s"
            )
        )

        ms_item = finite_float(
            row.get(
                "ms_per_work_item"
            )
        )

        work_unit = str(
            row.get(
                "work_unit",
                "item",
            )
        )

        secondary_ms = finite_float(
            row.get(
                "ms_per_secondary_item"
            )
        )

        secondary_unit = str(
            row.get(
                "secondary_unit",
                ""
            )
        )

        if (
            secondary_ms is not None
            and secondary_unit.lower()
            == "query"
        ):
            ms_item = secondary_ms
            work_unit = secondary_unit

        lines.append(
            "| "
            + label
            + " | "
            + (
                f"{runtime_s:.3f} s"
                if runtime_s is not None
                else "n/a"
            )
            + " | "
            + (
                f"{ms_item:.3f} ms/{work_unit}"
                if ms_item is not None
                else "n/a"
            )
            + " | "
            + str(
                row.get(
                    "cost_scope",
                    "n/a",
                )
            )
            + " |"
        )

    online_like = deployment.get(
        "online_like",
        {}
    )

    serial_ms = recursive_find(
        online_like,
        {
            "serial_equivalent_ms",
            "serial_cycle_ms",
            "serial_equivalent_cycle_ms",
        },
    )

    capacity_hz = recursive_find(
        online_like,
        {
            "estimated_capacity_hz",
            "serial_capacity_hz",
            "capacity_hz",
        },
    )

    if serial_ms is not None:
        lines.append(
            ""
        )

        lines.append(
            "- Serial-equivalent localization cycle: "
            f"**{fmt_number(serial_ms, 2)} ms**."
        )

    if capacity_hz is not None:
        lines.append(
            "- Approximate serial capacity on tested CPU: "
            f"**{fmt_number(capacity_hz, 3)} Hz**."
        )

    system = resource.get(
        "system",
        {}
    )

    accelerator = resource.get(
        "accelerator",
        {}
    )

    devices_used = resource.get(
        "devices_used",
        []
    )

    lines.extend([
        "",
        "### Resource context",
        "",
        (
            "- Execution devices: "
            f"`{devices_used}`."
        ),
        (
            "- CUDA available: "
            f"**{recursive_find(accelerator, {'cuda_available', 'cuda'})}**."
        ),
        (
            "- Total RAM: "
            f"**{fmt_number(recursive_find(system, {'system_ram_total_gib', 'total_ram_gib', 'ram_gib', 'memory_gib'}), 2, ' GiB')}**."
        ),
    ])

    mem = memory_entries(
        resource
    )

    if mem:
        lines.extend([
            "",
            "| Core stage/resource measurement | Peak RSS |",
            "|---|---:|",
        ])

        for label, peak in mem:
            lines.append(
                f"| {label} | {peak:.1f} MiB |"
            )

    lines.extend([
        "",
        (
            "**Interpretation:** the runtime/resource section is "
            "for deployment feasibility. Supporting visualization "
            "and report-generation costs are measured in the registry "
            "for completeness but are not treated as localization "
            "bottlenecks."
        ),
    ])

    # Blind visuals
    write_section(
        lines,
        "10. Blind-run visual diagnostics",
    )

    blind_figures = [
        (
            run_root
            / "figures"
            / "estimated_relative_xy.png",
            "Blind map-aligned relative trajectory",
        ),
        (
            run_root
            / "figures"
            / "estimated_fused_xy.png",
            "Blind fused map trajectory",
        ),
        (
            run_root
            / "figures"
            / "confidence_vs_time.png",
            "Blind absolute-localization confidence",
        ),
        (
            run_root
            / "figures"
            / "accepted_corrections_timeline.png",
            "Blind correction decisions",
        ),
    ]

    for path, alt in blind_figures:

        image = md_image(
            summary_dir,
            path,
            alt,
        )

        if image:
            lines.extend([
                f"### {alt}",
                "",
                image,
                "",
            ])

    map_link = md_file_link(
        summary_dir,
        (
            run_root
            / "maps"
            / "estimated_fused_map.html"
        ),
        "Open interactive blind estimated trajectory map",
    )

    if map_link:
        lines.append(
            f"- {map_link}"
        )

    lines.append(
        ""
    )

    if evaluation_included:
        lines.append(
            "> These blind-run figures use only estimated "
            "localization outputs; no GT/reference overlay "
            "is used in this visual section."
        )
    else:
        lines.append(
            f"> {BLIND_NOTE}"
        )

    # Evaluation section
    write_section(
        lines,
        "11. Post-freeze evaluation",
    )

    if not evaluation_included:

        lines.extend([
            f"**{BLIND_NOTE}**",
            "",
            (
                "Run this summary generator with "
                "`--include-evaluation` only after a frozen blind "
                "submission has been evaluated."
            ),
        ])

    else:

        continuous = evaluation_summary.get(
            "continuous_fused_trajectory",
            {},
        )

        rmse = first_value(
            continuous,
            [
                "rmse_m",
                "rmse",
            ],
        )

        median = first_value(
            continuous,
            [
                "median_error_m",
                "median_m",
                "median",
            ],
        )

        p95 = first_value(
            continuous,
            [
                "p95_error_m",
                "p95_m",
                "p95",
            ],
        )

        final_error = first_value(
            continuous,
            [
                "final_error_m",
                "final_m",
            ],
        )

        evaluated_poses = first_value(
            continuous,
            [
                "evaluated_poses",
                "evaluated_pose_count",
                "pose_count",
                "count",
            ],
        )

        lines.extend([
            (
                "- Evaluated fused poses: "
                f"**{fmt_int(evaluated_poses)}**."
            ),
            f"- RMSE: **{fmt_number(rmse, 3, ' m')}**.",
            (
                "- Median error: "
                f"**{fmt_number(median, 3, ' m')}**."
            ),
            (
                "- p95 error: "
                f"**{fmt_number(p95, 3, ' m')}**."
            ),
            (
                "- Final error: "
                f"**{fmt_number(final_error, 3, ' m')}**."
            ),
        ])

        drift_per100 = recursive_find(
            drift_summary,
            {
                "drift_per_100m",
                "drift_per_100m_m",
                "final_drift_per_100m",
            },
        )

        drift_min = recursive_find(
            drift_summary,
            {
                "drift_per_minute",
                "drift_per_min",
                "drift_per_min_m",
            },
        )

        if drift_per100 is not None:
            lines.append(
                "- Drift from map lock: "
                f"**{fmt_number(drift_per100, 6)} m/100 m**."
            )

        if drift_min is not None:
            lines.append(
                "- Drift rate: "
                f"**{fmt_number(drift_min, 6)} m/min**."
            )

        if safety_summary:

            lines.extend([
                "",
                "### Accepted-correction safety",
                "",
                (
                    "- Blindly accepted corrections: "
                    f"**{safety_summary.get('accepted_total', 'n/a')}**."
                ),
                (
                    "- Accepted within 40 m after GT attachment: "
                    f"**{safety_summary.get('accepted_le40m', 'n/a')}**."
                ),
                (
                    "- False accepted corrections >40 m: "
                    f"**{safety_summary.get('accepted_gt40m_false', 'n/a')}**."
                ),
                (
                    "- Dangerous accepted corrections >100 m: "
                    f"**{safety_summary.get('accepted_gt100m_dangerous', 'n/a')}**."
                ),
            ])

        lines.extend([
            "",
            "### Retrieval-depth diagnostic at ≤40 m",
            "",
            "| Method | Within 40 m | Population |",
            "|---|---:|---:|",
        ])

        for method, label in [
            (
                "dino_top1",
                "DINO Top-1",
            ),
            (
                "dino_top20",
                "DINO best in Top-20",
            ),
            (
                "dino_top100",
                "DINO best in Top-100",
            ),
            (
                "orb_selected",
                "ORB-selected",
            ),
            (
                "lightglue_selected",
                "LightGlue-selected",
            ),
            (
                "accepted_corrections",
                "Blind accepted corrections",
            ),
        ]:

            result = threshold_count(
                threshold_summary,
                method,
                40.0,
            )

            if result is None:
                continue

            count, population = result

            lines.append(
                f"| {label} | {count} | {population} |"
            )

        lines.extend([
            "",
            (
                "> These thresholds are evaluation-only and were "
                "not available to the blind localization pipeline."
            ),
        ])

        evaluation_figures = [
            (
                run_root
                / "figures"
                / "error_vs_time.png",
                "Post-freeze position error vs time",
            ),
            (
                run_root
                / "figures"
                / "error_vs_distance.png",
                "Post-freeze position error vs travelled distance",
            ),
            (
                run_root
                / "figures"
                / "estimated_vs_reference_xy.png",
                "Estimated trajectory vs reference",
            ),
            (
                run_root
                / "figures"
                / "fused_vs_reference_xy.png",
                "Fused trajectory vs reference",
            ),
            (
                run_root
                / "figures"
                / "evaluation_trajectory_comparison.png",
                "Estimated trajectory vs reference",
            ),
        ]

        included_eval_images = 0

        for path, alt in evaluation_figures:

            image = md_image(
                summary_dir,
                path,
                alt,
            )

            if image:
                included_eval_images += 1

                lines.extend([
                    "",
                    f"### {alt}",
                    "",
                    image,
                ])

        if included_eval_images == 0:
            lines.extend([
                "",
                (
                    "_No evaluation comparison figure was found. "
                    "The summary supports such figures when they "
                    "are generated by the post-freeze evaluator._"
                ),
            ])

    # 12
    write_section(
        lines,
        "12. Generated files",
    )

    generated = [
        (
            submission_path,
            "Estimated trajectory CSV",
        ),
        (
            run_root
            / "figures"
            / "estimated_relative_xy.png",
            "Blind relative XY figure",
        ),
        (
            run_root
            / "figures"
            / "estimated_fused_xy.png",
            "Blind fused XY figure",
        ),
        (
            run_root
            / "figures"
            / "confidence_vs_time.png",
            "Confidence timeline",
        ),
        (
            run_root
            / "figures"
            / "accepted_corrections_timeline.png",
            "Correction-decision timeline",
        ),
        (
            run_root
            / "maps"
            / "estimated_fused_map.html",
            "Interactive estimated trajectory map",
        ),
    ]

    for path, label in generated:
        link = md_file_link(
            summary_dir,
            path,
            label,
        )

        if link:
            lines.append(
                f"- {link}"
            )

    # 13
    write_section(
        lines,
        "13. Known limitations and next improvements",
    )

    lines.extend([
        (
            "- Absolute positioning currently relies on coarse "
            "map-tile evidence; sub-tile camera localization is "
            "a major accuracy improvement opportunity."
        ),
        (
            "- Relative visual motion remains continuous but "
            "accumulates error between trustworthy absolute anchors."
        ),
        (
            "- Absolute corrections are sparse and repeated nearby "
            "views may not provide independent map evidence."
        ),
        (
            "- Fixed soft-fusion weighting is a baseline; future "
            "work should investigate confidence/uncertainty-aware "
            "filtering or graph-based fusion."
        ),
        (
            "- DINO query encoding and ORB verification dominate "
            "the tested CPU computation. Higher absolute-update "
            "rates require scheduling/optimization and/or faster "
            "compute."
        ),
        (
            "- Retrieval ambiguity remains challenging in repeated "
            "structures, roads, vegetation, and appearance changes."
        ),
        (
            "- Evaluation metrics must remain separated from blind "
            "localization decisions and attached only after output freeze."
        ),
    ])

    md_text = (
        "\n".join(
            lines
        ).rstrip()
        + "\n"
    )

    md_path.write_text(
        md_text,
        encoding="utf-8",
    )

    submission_sha_after = (
        sha256_file(
            submission_path
        )
    )

    require(
        submission_sha_before
        == submission_sha_after,
        (
            "Submission trajectory changed while "
            "generating run summary."
        ),
    )

    runtime_s = (
        time.perf_counter()
        - started
    )

    report = {
        "stage": (
            "ADDON11_RUN_SUMMARY"
        ),
        "status": (
            "PASS_ADDON11_RUN_SUMMARY"
        ),
        "human_facing_outputs": {
            "markdown": str(
                md_path
            ),
            "html": None,
        },
        "evaluation_requested":
            bool(
                args.include_evaluation
            ),
        "evaluation_included":
            evaluation_included,
        "blind_contract": {
            "summary_can_run_without_reference":
                True,
            "evaluation_read_only_when_explicitly_enabled":
                True,
            "submission_unchanged":
                True,
            "submission_sha256":
                submission_sha_before,
        },
        "runtime": {
            "run_summary_generation_s":
                float(
                    runtime_s
                ),
        },
        "headline_runtime_policy": (
            "Only localization-critical computation "
            "is shown prominently in the Markdown."
        ),
        "supporting_runtime_policy": (
            "Plot/Folium/CSV/report-generation costs "
            "remain in the canonical registry but are "
            "not headline deployment metrics."
        ),
    }

    report_path.write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("=" * 88)
    print(
        "STAGE 12B — ADD-ON 11 "
        "MARKDOWN RUN SUMMARY"
    )
    print("=" * 88)

    print()
    print(
        "dataset               :",
        root.name,
    )

    print(
        "run ID                :",
        run_root.name,
    )

    print(
        "blind frames          :",
        len(
            manifest
        ),
    )

    print(
        "map poses             :",
        int(
            map_available.sum()
        ),
    )

    print(
        "accepted corrections  :",
        int(
            accepted.sum()
        ),
    )

    print(
        "evaluation requested  :",
        bool(
            args.include_evaluation
        ),
    )

    print(
        "evaluation included   :",
        evaluation_included,
    )

    print(
        "submission unchanged  :",
        submission_sha_before
        == submission_sha_after,
    )

    print()
    print(
        "headline runtime rows :",
        len(
            core_runtime
        ),
    )

    print(
        "summary generation    :",
        f"{runtime_s:.6f} s",
    )

    print()
    print(
        "saved Markdown        :",
        md_path,
    )

    print(
        "internal report       :",
        report_path,
    )

    print()
    print(
        "status: "
        "PASS_ADDON11_RUN_SUMMARY"
    )


if __name__ == "__main__":
    main()
