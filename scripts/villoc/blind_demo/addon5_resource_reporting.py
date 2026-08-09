'''
ROOT=outputs/villoc/traj01_90deg_stable120m
MAPROOT=outputs/villoc/90_deg
RUN="$ROOT/blind_demo_addons/eval_traj01_known_gt"
TAG=dinov2_vits14_img518_center_square_avgpatch_cpu

python scripts/villoc/blind_demo/addon5_resource_reporting.py \
  --root "$ROOT" \
  --map-root "$MAPROOT" \
  --run-root "$RUN" \
  --variant 512_s256 \
  --tag "$TAG" \
  2>&1 | tee \
  "$RUN/logs/addon5_resource_reporting_static.log"
'''

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def bytes_to_mib(value: int | float | None) -> float | None:
    if value is None:
        return None
    return float(value) / (1024.0 ** 2)


def bytes_to_gib(value: int | float | None) -> float | None:
    if value is None:
        return None
    return float(value) / (1024.0 ** 3)


def file_size(path: Path) -> int | None:
    if not path.exists() or not path.is_file():
        return None
    return int(path.stat().st_size)


def folder_size(path: Path) -> tuple[int | None, int]:
    if not path.exists() or not path.is_dir():
        return None, 0

    total = 0
    count = 0

    for p in path.rglob("*"):
        try:
            if p.is_file() and not p.is_symlink():
                total += p.stat().st_size
                count += 1
        except OSError:
            continue

    return int(total), int(count)


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}

    try:
        return json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )
    except Exception:
        return {}


def load_npz_meta(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}

    try:
        data = np.load(
            path,
            allow_pickle=False,
        )

        return json.loads(
            str(data["meta_json"])
        )

    except Exception:
        return {}


def recursive_values_for_key(
    obj: Any,
    needle: str,
    prefix: str = "",
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    if isinstance(obj, dict):
        for key, value in obj.items():
            path = (
                f"{prefix}.{key}"
                if prefix
                else str(key)
            )

            if needle.lower() in str(key).lower():
                if not isinstance(
                    value,
                    (dict, list),
                ):
                    rows.append(
                        {
                            "key": path,
                            "value": value,
                        }
                    )

            rows.extend(
                recursive_values_for_key(
                    value,
                    needle,
                    path,
                )
            )

    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            rows.extend(
                recursive_values_for_key(
                    value,
                    needle,
                    f"{prefix}[{i}]",
                )
            )

    return rows


def ru_maxrss_bytes() -> int:
    value = resource.getrusage(
        resource.RUSAGE_SELF
    ).ru_maxrss

    # macOS reports bytes.
    # Linux reports KiB.
    if sys.platform == "darwin":
        return int(value)

    return int(value * 1024)


def process_snapshot() -> dict[str, Any]:
    result: dict[str, Any] = {
        "method": None,
        "rss_bytes": None,
        "rss_mib": None,
        "vms_bytes": None,
        "vms_mib": None,
        "ru_maxrss_bytes": (
            ru_maxrss_bytes()
        ),
        "ru_maxrss_mib": (
            bytes_to_mib(
                ru_maxrss_bytes()
            )
        ),
    }

    try:
        import psutil

        process = psutil.Process(
            os.getpid()
        )

        mem = process.memory_info()

        result.update(
            {
                "method": "psutil",
                "rss_bytes": int(
                    mem.rss
                ),
                "rss_mib": (
                    bytes_to_mib(
                        mem.rss
                    )
                ),
                "vms_bytes": int(
                    mem.vms
                ),
                "vms_mib": (
                    bytes_to_mib(
                        mem.vms
                    )
                ),
            }
        )

    except Exception:
        result["method"] = (
            "resource_module_best_effort"
        )

    return result


def system_snapshot() -> dict[str, Any]:
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "system": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_version": (
            platform.python_version()
        ),
        "cpu_physical_cores": None,
        "cpu_logical_cores": (
            os.cpu_count()
        ),
        "system_ram_total_bytes": None,
        "system_ram_total_gib": None,
        "system_ram_available_bytes": None,
        "system_ram_available_gib": None,
        "psutil_available": False,
    }

    try:
        import psutil

        info[
            "psutil_available"
        ] = True

        info[
            "cpu_physical_cores"
        ] = psutil.cpu_count(
            logical=False
        )

        info[
            "cpu_logical_cores"
        ] = psutil.cpu_count(
            logical=True
        )

        vm = psutil.virtual_memory()

        info[
            "system_ram_total_bytes"
        ] = int(vm.total)

        info[
            "system_ram_total_gib"
        ] = bytes_to_gib(
            vm.total
        )

        info[
            "system_ram_available_bytes"
        ] = int(vm.available)

        info[
            "system_ram_available_gib"
        ] = bytes_to_gib(
            vm.available
        )

    except Exception:
        pass

    return info


def torch_snapshot() -> dict[str, Any]:
    result: dict[str, Any] = {
        "torch_available": False,
        "torch_version": None,
        "cuda_available": False,
        "cuda_device_count": 0,
        "gpu_names": [],
        "gpu_memory": [],
        "mps_available": False,
    }

    try:
        import torch

        result[
            "torch_available"
        ] = True

        result[
            "torch_version"
        ] = str(
            torch.__version__
        )

        cuda_available = bool(
            torch.cuda.is_available()
        )

        result[
            "cuda_available"
        ] = cuda_available

        if cuda_available:
            count = int(
                torch.cuda.device_count()
            )

            result[
                "cuda_device_count"
            ] = count

            for i in range(count):
                props = (
                    torch.cuda.get_device_properties(
                        i
                    )
                )

                result[
                    "gpu_names"
                ].append(
                    str(props.name)
                )

                result[
                    "gpu_memory"
                ].append(
                    {
                        "device_index": i,
                        "total_bytes": int(
                            props.total_memory
                        ),
                        "total_gib": (
                            bytes_to_gib(
                                props.total_memory
                            )
                        ),
                    }
                )

        try:
            result[
                "mps_available"
            ] = bool(
                torch.backends.mps.is_available()
            )
        except Exception:
            pass

    except Exception:
        pass

    return result


def count_rows(path: Path) -> int | None:
    if not path.exists():
        return None

    try:
        return int(
            len(
                pd.read_csv(path)
            )
        )
    except Exception:
        return None


def common_tile_parent(
    tile_index: Path,
) -> Path | None:
    if not tile_index.exists():
        return None

    try:
        df = pd.read_csv(
            tile_index
        )

        if "tile_path" not in df.columns:
            return None

        paths = [
            Path(str(p)).expanduser()
            for p in df[
                "tile_path"
            ].dropna()
        ]

        resolved = []

        repo_root = Path.cwd().resolve()

        for p in paths:
            if not p.is_absolute():
                p = (
                    repo_root / p
                )

            if p.exists():
                resolved.append(
                    str(
                        p.resolve()
                    )
                )

        if not resolved:
            return None

        common = Path(
            os.path.commonpath(
                resolved
            )
        )

        if common.is_file():
            return common.parent

        return common

    except Exception:
        return None


def append_metric(
    rows: list[dict[str, Any]],
    *,
    category: str,
    metric: str,
    value: Any,
    unit: str | None = None,
    status: str = "MEASURED",
    source: str | None = None,
    notes: str = "",
) -> None:
    rows.append(
        {
            "category": category,
            "metric": metric,
            "value": value,
            "unit": unit,
            "status": status,
            "source": source,
            "notes": notes,
        }
    )



def run_native_blind_resource_registry(
    *,
    run_root: Path,
    map_root: Path,
    variant: str,
    tag: str,
    metrics_dir: Path,
) -> None:

    # =====================================================
    # Resource inventory based ONLY on the current blind
    # recorded-flight run plus the reusable prepared map.
    #
    # Historical traj01 / LightGlue / F1 / F3 / F3B
    # artifacts are deliberately excluded.
    # =====================================================

    before = process_snapshot()

    # -----------------------------------------------------
    # Canonical current-run artifacts.
    # -----------------------------------------------------

    manifest = (
        run_root
        / "metadata"
        / "blind_query_manifest.csv"
    )

    query_cache = (
        run_root
        / "descriptors"
        / (
            "s8_11c_dinov2_queries_v_1fps_"
            f"{tag}.npz"
        )
    )

    map_cache = (
        map_root
        / "descriptors"
        / (
            f"s8_11b_dinov2_map_{variant}_"
            f"{tag}.npz"
        )
    )

    tile_index = (
        map_root
        / "metadata"
        / (
            "s8_9_satellite_tile_index_"
            f"{variant}.csv"
        )
    )

    xfeat_report = (
        run_root
        / "reports"
        / "s8_xfeat_relative_frontend"
        / "s8r4_xfeat_relative_frontend_report.json"
    )

    orb_report = (
        run_root
        / "reports"
        / "s8_12e1_top20_verifier_reranker"
        / f"{variant}_orb_hybrid_top20_img518"
        / "s8_12e1_summary.json"
    )

    submission = (
        run_root
        / "trajectories"
        / "submission_estimated_trajectory.csv"
    )

    timing_summary = (
        run_root
        / "metrics"
        / "timing_summary.csv"
    )

    required = [
        manifest,
        query_cache,
        map_cache,
        tile_index,
        xfeat_report,
        orb_report,
        submission,
        timing_summary,
    ]

    missing = [
        p
        for p in required
        if not p.exists()
    ]

    if missing:
        raise RuntimeError(
            "Missing native blind resource inputs: "
            + ", ".join(
                str(p)
                for p in missing
            )
        )

    # -----------------------------------------------------
    # Metadata / workload.
    # -----------------------------------------------------

    query_meta = load_npz_meta(
        query_cache
    )

    map_meta = load_npz_meta(
        map_cache
    )

    xfeat_json = load_json(
        xfeat_report
    )

    orb_json = load_json(
        orb_report
    )

    frames = count_rows(
        manifest
    )

    map_tiles = count_rows(
        tile_index
    )

    if frames is None or frames <= 0:
        raise RuntimeError(
            "Invalid blind manifest frame count."
        )

    if map_tiles is None or map_tiles <= 0:
        raise RuntimeError(
            "Invalid map tile count."
        )

    orb_runtime = orb_json.get(
        "runtime",
        {},
    )

    orb_pairs = int(
        orb_runtime.get(
            "query_candidate_pairs",
            0,
        )
    )

    orb_queries = int(
        orb_runtime.get(
            "queries",
            0,
        )
    )

    if orb_queries != frames:
        raise RuntimeError(
            "ORB query workload mismatch: "
            f"{orb_queries} vs {frames}"
        )

    if orb_pairs <= 0:
        raise RuntimeError(
            "Invalid ORB pair workload."
        )

    # -----------------------------------------------------
    # System / accelerator.
    # -----------------------------------------------------

    system = system_snapshot()
    torch_info = torch_snapshot()

    devices: dict[str, Any] = {
        "dino": None,
        "xfeat": None,
        "orb": "cpu",
    }

    protocol = query_meta.get(
        "protocol",
        {},
    )

    if isinstance(
        protocol,
        dict,
    ):
        devices[
            "dino"
        ] = protocol.get(
            "device"
        )

    if devices["dino"] is None:
        device_candidates = (
            recursive_values_for_key(
                query_meta,
                "device",
            )
        )

        if device_candidates:
            devices[
                "dino"
            ] = device_candidates[
                0
            ][
                "value"
            ]

    xfeat_devices = (
        recursive_values_for_key(
            xfeat_json,
            "device",
        )
    )

    if xfeat_devices:
        devices[
            "xfeat"
        ] = xfeat_devices[
            0
        ][
            "value"
        ]

    # -----------------------------------------------------
    # Storage.
    # -----------------------------------------------------

    query_cache_bytes = file_size(
        query_cache
    )

    map_cache_bytes = file_size(
        map_cache
    )

    tile_folder = common_tile_parent(
        tile_index
    )

    if tile_folder is not None:
        (
            tile_folder_bytes,
            tile_file_count,
        ) = folder_size(
            tile_folder
        )
    else:
        tile_folder_bytes = None
        tile_file_count = 0

    (
        run_root_bytes,
        run_root_files,
    ) = folder_size(
        run_root
    )

    frame_folder = (
        run_root
        / "frames"
        / "blind_query_1fps"
    )

    (
        frame_folder_bytes,
        frame_folder_files,
    ) = folder_size(
        frame_folder
    )

    submission_bytes = file_size(
        submission
    )

    timing_bytes = file_size(
        timing_summary
    )

    rows: list[
        dict[str, Any]
    ] = []

    # -----------------------------------------------------
    # System rows.
    # -----------------------------------------------------

    for key, value in system.items():

        unit = None

        if key.endswith(
            "_bytes"
        ):
            unit = "bytes"

        elif key.endswith(
            "_gib"
        ):
            unit = "GiB"

        append_metric(
            rows,
            category="system",
            metric=key,
            value=value,
            unit=unit,
        )

    for key, value in torch_info.items():

        append_metric(
            rows,
            category="accelerator",
            metric=key,
            value=(
                json.dumps(
                    value
                )
                if isinstance(
                    value,
                    (
                        list,
                        dict,
                    ),
                )
                else value
            ),
        )

    for stage, device in devices.items():

        append_metric(
            rows,
            category="stage_device",
            metric=stage,
            value=device,
            status=(
                "MEASURED"
                if device is not None
                else "UNKNOWN"
            ),
        )

    # -----------------------------------------------------
    # Storage rows.
    # -----------------------------------------------------

    storage_items = [
        (
            "query_descriptor_cache",
            query_cache,
            query_cache_bytes,
            (
                1
                if query_cache.exists()
                else 0
            ),
        ),
        (
            "map_descriptor_cache",
            map_cache,
            map_cache_bytes,
            (
                1
                if map_cache.exists()
                else 0
            ),
        ),
        (
            "prepared_tile_folder",
            tile_folder,
            tile_folder_bytes,
            tile_file_count,
        ),
        (
            "blind_frame_folder",
            frame_folder,
            frame_folder_bytes,
            frame_folder_files,
        ),
        (
            "blind_run_output_folder",
            run_root,
            run_root_bytes,
            run_root_files,
        ),
        (
            "submission_csv",
            submission,
            submission_bytes,
            1,
        ),
        (
            "timing_summary_csv",
            timing_summary,
            timing_bytes,
            1,
        ),
    ]

    for (
        name,
        source_path,
        size_bytes,
        file_count,
    ) in storage_items:

        status = (
            "MEASURED"
            if size_bytes is not None
            else "MISSING"
        )

        append_metric(
            rows,
            category="storage",
            metric=(
                f"{name}_size_bytes"
            ),
            value=size_bytes,
            unit="bytes",
            status=status,
            source=(
                str(
                    source_path
                )
                if source_path is not None
                else None
            ),
        )

        append_metric(
            rows,
            category="storage",
            metric=(
                f"{name}_size_mib"
            ),
            value=bytes_to_mib(
                size_bytes
            ),
            unit="MiB",
            status=status,
            source=(
                str(
                    source_path
                )
                if source_path is not None
                else None
            ),
        )

        append_metric(
            rows,
            category="storage",
            metric=(
                f"{name}_file_count"
            ),
            value=file_count,
            unit="files",
            status=status,
            source=(
                str(
                    source_path
                )
                if source_path is not None
                else None
            ),
        )

    # -----------------------------------------------------
    # Workload rows.
    # -----------------------------------------------------

    workload_items = {
        "number_of_query_frames":
            frames,

        "number_of_frame_pairs":
            max(
                frames - 1,
                0,
            ),

        "number_of_map_tiles":
            map_tiles,

        "dino_query_descriptor_rows":
            query_meta.get(
                "row_count"
            ),

        "dino_map_descriptor_rows":
            map_meta.get(
                "row_count"
            ),

        "orb_queries":
            orb_queries,

        "orb_query_candidate_pairs":
            orb_pairs,

        "orb_topk_per_query":
            (
                orb_pairs
                // orb_queries
                if orb_queries > 0
                else None
            ),
    }

    for key, value in workload_items.items():

        append_metric(
            rows,
            category="workload",
            metric=key,
            value=value,
            unit="count",
            status=(
                "MEASURED"
                if value is not None
                else "MISSING"
            ),
        )

    # -----------------------------------------------------
    # Per-stage peak RSS.
    #
    # This can only be populated from actual live-memory
    # sampling JSONs. No retrospective estimate is made.
    # -----------------------------------------------------

    primary_stages = [
        "relative_odometry",
        "dino_query_descriptor_extraction",
        "dino_retrieval_against_map_cache",
        "orb_topk_reranking",
        "blind_map_bootstrap",
        "blind_map_alignment",
        "blind_temporal_fusion",
    ]

    stage_memory_dir = (
        run_root
        / "metrics"
        / "stage_memory"
    )

    stage_memory: dict[
        str,
        Any,
    ] = {}

    live_measured_stages = []
    unavailable_stages = []

    for stage in primary_stages:

        measurement_path = (
            stage_memory_dir
            / f"{stage}.json"
        )

        measurement = (
            load_json(
                measurement_path
            )
            if measurement_path.exists()
            else {}
        )

        if (
            measurement.get(
                "status"
            )
            == "PASS"
            and isinstance(
                measurement.get(
                    "process_memory"
                ),
                dict,
            )
        ):

            memory = measurement[
                "process_memory"
            ]

            first_rss = memory.get(
                "rss_before_work_mib"
            )

            last_rss = memory.get(
                "rss_after_work_last_live_mib"
            )

            peak_rss = memory.get(
                "peak_process_tree_rss_mib"
            )

            peak_root = memory.get(
                "peak_root_rss_mib"
            )

            peak_process_count = memory.get(
                "peak_live_process_count"
            )

            increment = None

            if (
                first_rss is not None
                and peak_rss is not None
            ):
                increment = (
                    float(
                        peak_rss
                    )
                    - float(
                        first_rss
                    )
                )

            stage_memory[
                stage
            ] = {
                "status":
                    "MEASURED_LIVE",

                "source":
                    str(
                        measurement_path
                    ),

                "rss_first_live_sample_mib":
                    first_rss,

                "rss_last_live_sample_mib":
                    last_rss,

                "peak_root_rss_mib":
                    peak_root,

                "peak_process_tree_rss_mib":
                    peak_rss,

                "peak_increment_from_first_sample_mib":
                    increment,

                "peak_live_process_count":
                    peak_process_count,

                "measurement_method":
                    measurement.get(
                        "measurement_method"
                    ),

                "sample_interval_s":
                    measurement.get(
                        "sample_interval_s"
                    ),

                "sample_count":
                    measurement.get(
                        "sample_count"
                    ),

                "wall_time_s":
                    measurement.get(
                        "wall_time_s"
                    ),
            }

            live_measured_stages.append(
                stage
            )

            for suffix, value, unit in [
                (
                    "rss_first_live_sample_mib",
                    first_rss,
                    "MiB",
                ),
                (
                    "rss_last_live_sample_mib",
                    last_rss,
                    "MiB",
                ),
                (
                    "peak_rss_mib",
                    peak_rss,
                    "MiB",
                ),
                (
                    "peak_increment_mib",
                    increment,
                    "MiB",
                ),
                (
                    "peak_process_count",
                    peak_process_count,
                    "processes",
                ),
            ]:
                append_metric(
                    rows,
                    category="stage_memory",
                    metric=(
                        f"{stage}.{suffix}"
                    ),
                    value=value,
                    unit=unit,
                    source=str(
                        measurement_path
                    ),
                )

        else:

            reason = (
                "No successful live process-memory "
                "measurement exists for this stage. "
                "Peak RAM is not inferred retrospectively."
            )

            stage_memory[
                stage
            ] = {
                "status":
                    "PENDING_LIVE_MEASUREMENT",

                "source":
                    (
                        str(
                            measurement_path
                        )
                        if measurement_path.exists()
                        else None
                    ),

                "peak_process_tree_rss_mib":
                    None,

                "reason":
                    reason,
            }

            unavailable_stages.append(
                stage
            )

            append_metric(
                rows,
                category="stage_memory",
                metric=(
                    f"{stage}.peak_rss_mib"
                ),
                value=None,
                unit="MiB",
                status=(
                    "PENDING_LIVE_MEASUREMENT"
                ),
                source=(
                    str(
                        measurement_path
                    )
                    if measurement_path.exists()
                    else None
                ),
                notes=reason,
            )

    after = process_snapshot()

    # -----------------------------------------------------
    # Stable cache summary.
    # -----------------------------------------------------

    cache_summary = {
        "status":
            "PASS_CACHE_SIZE_SUMMARY",

        "registry_mode":
            "PASS_NATIVE_BLIND_RESOURCE_REGISTRY",

        "variant":
            variant,

        "descriptor_tag":
            tag,

        "query_descriptor_cache": {
            "path":
                str(
                    query_cache
                ),

            "exists":
                query_cache.exists(),

            "size_bytes":
                query_cache_bytes,

            "size_mib":
                bytes_to_mib(
                    query_cache_bytes
                ),

            "row_count":
                query_meta.get(
                    "row_count"
                ),

            "descriptor_shape":
                query_meta.get(
                    "descriptor_shape"
                ),
        },

        "map_descriptor_cache": {
            "path":
                str(
                    map_cache
                ),

            "exists":
                map_cache.exists(),

            "size_bytes":
                map_cache_bytes,

            "size_mib":
                bytes_to_mib(
                    map_cache_bytes
                ),

            "row_count":
                map_meta.get(
                    "row_count"
                ),

            "descriptor_shape":
                map_meta.get(
                    "descriptor_shape"
                ),

            "cost_scope":
                "offline_reusable",
        },

        "prepared_tile_folder": {
            "path":
                (
                    str(
                        tile_folder
                    )
                    if tile_folder is not None
                    else None
                ),

            "size_bytes":
                tile_folder_bytes,

            "size_mib":
                bytes_to_mib(
                    tile_folder_bytes
                ),

            "file_count":
                tile_file_count,
        },

        "blind_frame_folder": {
            "path":
                str(
                    frame_folder
                ),

            "size_bytes":
                frame_folder_bytes,

            "size_mib":
                bytes_to_mib(
                    frame_folder_bytes
                ),

            "file_count":
                frame_folder_files,
        },

        "blind_run_output_folder": {
            "path":
                str(
                    run_root
                ),

            "size_bytes":
                run_root_bytes,

            "size_mib":
                bytes_to_mib(
                    run_root_bytes
                ),

            "file_count":
                run_root_files,
        },
    }

    # -----------------------------------------------------
    # Final report.
    # -----------------------------------------------------

    report = {
        "addon":
            "Add-on 5 — Memory/resource reporting",

        "status":
            "PASS_ADDON5_RESOURCE_REPORTING",

        "registry_mode":
            "PASS_NATIVE_BLIND_RESOURCE_REGISTRY",

        "system":
            system,

        "accelerator":
            torch_info,

        "devices_used":
            devices,

        "process_memory": {
            "reporter_before":
                before,

            "reporter_after":
                after,
        },

        "workload":
            workload_items,

        "storage":
            cache_summary,

        "stage_memory":
            stage_memory,

        "stage_memory_status": {
            "status":
                "PASS_BEST_EFFORT_STAGE_MEMORY",

            "live_measured_count":
                len(
                    live_measured_stages
                ),

            "live_measured_stages":
                live_measured_stages,

            "not_live_measured_count":
                len(
                    unavailable_stages
                ),

            "not_live_measured_stages":
                unavailable_stages,

            "important_note": (
                "Per-stage peak RAM is only "
                "reported when captured during "
                "live execution. Missing values "
                "remain pending and are never "
                "estimated from runtime or file size."
            ),
        },

        "excluded_from_demo_resource_registry": [
            "LightGlue diagnostic runs",
            "ORB smoke tests",
            "Stage-7 map-variant diagnostics",
            "historical F1/F3/F3B evaluation pipeline",
        ],

        "important_rule": (
            "Static cache/device/storage information "
            "is measured directly. Per-stage peak RAM "
            "must come from live process sampling."
        ),
    }

    resource_csv = (
        metrics_dir
        / "resource_summary.csv"
    )

    resource_json = (
        metrics_dir
        / "resource_summary.json"
    )

    cache_json = (
        metrics_dir
        / "cache_size_summary.json"
    )

    pd.DataFrame(
        rows
    ).to_csv(
        resource_csv,
        index=False,
    )

    resource_json.write_text(
        json.dumps(
            report,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    cache_json.write_text(
        json.dumps(
            cache_summary,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    print("=" * 88)
    print(
        "STAGE 8F.2 — NATIVE BLIND "
        "RESOURCE/CACHE INVENTORY"
    )
    print("=" * 88)

    print()
    print(
        "registry mode           : "
        "PASS_NATIVE_BLIND_RESOURCE_REGISTRY"
    )

    print(
        "status                  : "
        "PASS_ADDON5_RESOURCE_REPORTING"
    )

    print()
    print("System")
    print("-" * 88)

    print(
        "platform                :",
        system[
            "platform"
        ],
    )

    print(
        "physical CPU cores      :",
        system[
            "cpu_physical_cores"
        ],
    )

    print(
        "logical CPU cores       :",
        system[
            "cpu_logical_cores"
        ],
    )

    print(
        "system RAM [GiB]        :",
        system[
            "system_ram_total_gib"
        ],
    )

    print(
        "MPS available           :",
        torch_info[
            "mps_available"
        ],
    )

    print()
    print("Devices")
    print("-" * 88)

    for name, device in devices.items():
        print(
            f"{name:24s}:",
            device,
        )

    print()
    print("Workload")
    print("-" * 88)

    for key, value in workload_items.items():
        print(
            f"{key:32s}:",
            value,
        )

    print()
    print("Cache / storage")
    print("-" * 88)

    print(
        "query cache [MiB]       :",
        bytes_to_mib(
            query_cache_bytes
        ),
    )

    print(
        "map cache [MiB]         :",
        bytes_to_mib(
            map_cache_bytes
        ),
    )

    print(
        "prepared tiles [MiB]    :",
        bytes_to_mib(
            tile_folder_bytes
        ),
    )

    print(
        "blind frames [MiB]      :",
        bytes_to_mib(
            frame_folder_bytes
        ),
    )

    print(
        "blind run outputs [MiB] :",
        bytes_to_mib(
            run_root_bytes
        ),
    )

    print()
    print("Peak stage memory")
    print("-" * 88)

    print(
        "live measured stages    :",
        len(
            live_measured_stages
        ),
    )

    print(
        "pending live stages     :",
        len(
            unavailable_stages
        ),
    )

    for stage in unavailable_stages:
        print(
            "  PENDING:",
            stage,
        )

    print()
    print("Saved")
    print("-" * 88)

    print(resource_csv)
    print(resource_json)
    print(cache_json)

    print()
    print(
        "PASS_DEMO_STAGE8F2_NATIVE_RESOURCE_REGISTRY"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Add-on 5 resource/cache inventory "
            "for Villoc blind-demo pipeline."
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
        "--variant",
        default="512_s256",
    )

    parser.add_argument(
        "--tag",
        default=(
            "dinov2_vits14_img518_"
            "center_square_avgpatch_cpu"
        ),
    )

    args = parser.parse_args()

    root = args.root.resolve()
    map_root = args.map_root.resolve()
    run_root = args.run_root.resolve()

    if not root.exists():
        raise FileNotFoundError(root)

    if not map_root.exists():
        raise FileNotFoundError(
            map_root
        )

    metrics_dir = (
        run_root / "metrics"
    )

    metrics_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    before = process_snapshot()

    variant = args.variant
    tag = args.tag

    # -----------------------------------------------------
    # Native recorded-flight resource path.
    # -----------------------------------------------------

    native_blind_marker = (
        run_root
        / "reports"
        / "blind_query_manifest_report.json"
    )

    if native_blind_marker.exists():
        run_native_blind_resource_registry(
            run_root=run_root,
            map_root=map_root,
            variant=variant,
            tag=tag,
            metrics_dir=metrics_dir,
        )
        return

    query_cache = (
        root
        / "descriptors"
        / (
            "s8_11c_dinov2_queries_v_1fps_"
            f"{tag}.npz"
        )
    )

    map_cache = (
        map_root
        / "descriptors"
        / (
            f"s8_11b_dinov2_map_{variant}_"
            f"{tag}.npz"
        )
    )

    tile_index = (
        map_root
        / "metadata"
        / (
            "s8_9_satellite_tile_index_"
            f"{variant}.csv"
        )
    )

    xfeat_features = (
        root
        / "metadata/s8_xfeat_relative_frontend/"
        "s8r4_xfeat_frame_features.csv"
    )

    orb_candidates = (
        run_root
        / "benchmarks/orb_fullrun/"
        "s8_12e1_all_candidate_verifier_scores.csv"
    )

    lg_candidates = (
        root
        / "reports/s8_12e1b_lightglue_top20/"
        / (
            f"{variant}_lg_hybrid_top20_"
            "img518_full403"
        )
        / "s8_12e1b_all_candidate_lightglue_scores.csv"
    )

    xfeat_report = (
        root
        / "reports/s8_xfeat_relative_frontend/"
        "s8r4_xfeat_relative_frontend_report.json"
    )

    lg_report = (
        root
        / "reports/s8_12e1b_lightglue_top20/"
        / (
            f"{variant}_lg_hybrid_top20_"
            "img518_full403"
        )
        / "s8_12e1b_summary.json"
    )

    orb_report = (
        run_root
        / "benchmarks/orb_fullrun/"
        "s8_12e1_summary.json"
    )

    query_meta = load_npz_meta(
        query_cache
    )

    map_meta = load_npz_meta(
        map_cache
    )

    xfeat_json = load_json(
        xfeat_report
    )

    lg_json = load_json(
        lg_report
    )

    orb_json = load_json(
        orb_report
    )

    system = system_snapshot()
    torch_info = torch_snapshot()

    # -----------------------------------------------------
    # Device provenance
    # -----------------------------------------------------

    devices: dict[str, Any] = {
        "dino": None,
        "xfeat": None,
        "orb": "cpu",
        "lightglue": None,
    }

    protocol = query_meta.get(
        "protocol",
        {},
    )

    if isinstance(protocol, dict):
        devices["dino"] = (
            protocol.get(
                "device"
            )
        )

    xfeat_device_candidates = (
        recursive_values_for_key(
            xfeat_json,
            "device",
        )
    )

    if xfeat_device_candidates:
        devices["xfeat"] = (
            xfeat_device_candidates[0][
                "value"
            ]
        )

    devices["lightglue"] = (
        lg_json.get(
            "resolved_device"
        )
    )

    if devices["lightglue"] is None:
        for candidate in (
            recursive_values_for_key(
                lg_json,
                "device",
            )
        ):
            devices[
                "lightglue"
            ] = candidate[
                "value"
            ]
            break

    # -----------------------------------------------------
    # Storage
    # -----------------------------------------------------

    query_cache_bytes = file_size(
        query_cache
    )

    map_cache_bytes = file_size(
        map_cache
    )

    tile_folder = (
        common_tile_parent(
            tile_index
        )
    )

    tile_folder_bytes, tile_file_count = (
        folder_size(
            tile_folder
        )
        if tile_folder is not None
        else (None, 0)
    )

    run_root_bytes, run_root_files = (
        folder_size(
            run_root
        )
    )

    dataset_output_bytes, dataset_output_files = (
        folder_size(
            root
        )
    )

    # -----------------------------------------------------
    # Workloads
    # -----------------------------------------------------

    frames = count_rows(
        xfeat_features
    )

    map_tiles = count_rows(
        tile_index
    )

    orb_pairs = count_rows(
        orb_candidates
    )

    lightglue_pairs = count_rows(
        lg_candidates
    )

    # -----------------------------------------------------
    # Long-form CSV rows
    # -----------------------------------------------------

    rows: list[dict[str, Any]] = []

    for key, value in system.items():
        unit = None

        if key.endswith(
            "_bytes"
        ):
            unit = "bytes"

        elif key.endswith(
            "_gib"
        ):
            unit = "GiB"

        append_metric(
            rows,
            category="system",
            metric=key,
            value=value,
            unit=unit,
        )

    for key, value in torch_info.items():
        append_metric(
            rows,
            category="accelerator",
            metric=key,
            value=(
                json.dumps(value)
                if isinstance(
                    value,
                    (list, dict),
                )
                else value
            ),
        )

    for stage, device in (
        devices.items()
    ):
        append_metric(
            rows,
            category="stage_device",
            metric=stage,
            value=device,
            status=(
                "MEASURED"
                if device is not None
                else "UNKNOWN"
            ),
        )

    storage_items = [
        (
            "query_descriptor_cache",
            query_cache,
            query_cache_bytes,
            1 if query_cache.exists() else 0,
        ),
        (
            "map_descriptor_cache",
            map_cache,
            map_cache_bytes,
            1 if map_cache.exists() else 0,
        ),
        (
            "tile_folder",
            tile_folder,
            tile_folder_bytes,
            tile_file_count,
        ),
        (
            "addon_run_output_folder",
            run_root,
            run_root_bytes,
            run_root_files,
        ),
        (
            "dataset_output_folder",
            root,
            dataset_output_bytes,
            dataset_output_files,
        ),
    ]

    for (
        name,
        path,
        size_bytes,
        file_count,
    ) in storage_items:
        append_metric(
            rows,
            category="storage",
            metric=(
                f"{name}_size_bytes"
            ),
            value=size_bytes,
            unit="bytes",
            status=(
                "MEASURED"
                if size_bytes is not None
                else "MISSING"
            ),
            source=(
                str(path)
                if path is not None
                else None
            ),
        )

        append_metric(
            rows,
            category="storage",
            metric=(
                f"{name}_size_mib"
            ),
            value=(
                bytes_to_mib(
                    size_bytes
                )
            ),
            unit="MiB",
            status=(
                "MEASURED"
                if size_bytes is not None
                else "MISSING"
            ),
            source=(
                str(path)
                if path is not None
                else None
            ),
        )

        append_metric(
            rows,
            category="storage",
            metric=(
                f"{name}_file_count"
            ),
            value=file_count,
            unit="files",
            source=(
                str(path)
                if path is not None
                else None
            ),
        )

    workload_items = {
        "number_of_frames": frames,
        "number_of_map_tiles": map_tiles,
        "orb_query_candidate_pairs": orb_pairs,
        "lightglue_query_candidate_pairs": (
            lightglue_pairs
        ),
    }

    for key, value in (
        workload_items.items()
    ):
        append_metric(
            rows,
            category="workload",
            metric=key,
            value=value,
            unit="count",
            status=(
                "MEASURED"
                if value is not None
                else "MISSING"
            ),
        )

    # -----------------------------------------------------
    # Live per-stage memory measurements
    # -----------------------------------------------------

    stage_names = [
        "relative_odometry",
        "dino_query_descriptor_extraction",
        "dino_retrieval_against_map_cache",
        "orb_topk_reranking",
        "lightglue_reranking",
        "correction_manifest_build",
        "temporal_gating",
        "fusion_replay",
    ]

    stage_memory_dir = (
        run_root
        / "metrics"
        / "stage_memory"
    )

    stage_memory: dict[str, Any] = {}
    live_measured_stages: list[str] = []
    unavailable_stages: list[str] = []

    for stage in stage_names:
        measurement_path = (
            stage_memory_dir
            / f"{stage}.json"
        )

        measurement = (
            load_json(
                measurement_path
            )
            if measurement_path.exists()
            else {}
        )

        if (
            measurement.get("status") == "PASS"
            and isinstance(
                measurement.get(
                    "process_memory"
                ),
                dict,
            )
        ):
            memory = measurement[
                "process_memory"
            ]

            rss_first = memory.get(
                "rss_before_work_mib"
            )

            rss_last = memory.get(
                "rss_after_work_last_live_mib"
            )

            peak_rss = memory.get(
                "peak_process_tree_rss_mib"
            )

            peak_root = memory.get(
                "peak_root_rss_mib"
            )

            peak_process_count = (
                memory.get(
                    "peak_live_process_count"
                )
            )

            peak_increment = None

            if (
                rss_first is not None
                and peak_rss is not None
            ):
                peak_increment = (
                    float(peak_rss)
                    - float(rss_first)
                )

            stage_record = {
                "status": "MEASURED_LIVE",
                "source": str(
                    measurement_path
                ),
                "measurement_method": (
                    measurement.get(
                        "measurement_method"
                    )
                ),
                # This is the first sample after the
                # stage process starts. It is not a
                # pre-import Python baseline.
                "rss_first_live_sample_mib": (
                    rss_first
                ),
                "rss_last_live_sample_mib": (
                    rss_last
                ),
                "peak_root_rss_mib": (
                    peak_root
                ),
                "peak_process_tree_rss_mib": (
                    peak_rss
                ),
                "peak_increment_from_first_sample_mib": (
                    peak_increment
                ),
                "peak_live_process_count": (
                    peak_process_count
                ),
                "sample_interval_s": (
                    measurement.get(
                        "sample_interval_s"
                    )
                ),
                "sample_count": (
                    measurement.get(
                        "sample_count"
                    )
                ),
                "monitored_wall_time_s": (
                    measurement.get(
                        "wall_time_s"
                    )
                ),
                "exit_code": (
                    measurement.get(
                        "exit_code"
                    )
                ),
            }

            stage_memory[
                stage
            ] = stage_record

            live_measured_stages.append(
                stage
            )

            append_metric(
                rows,
                category="stage_memory",
                metric=(
                    f"{stage}.rss_first_live_sample_mib"
                ),
                value=rss_first,
                unit="MiB",
                source=str(
                    measurement_path
                ),
                notes=(
                    "First live RSS sample after process "
                    "launch; not a pre-import baseline."
                ),
            )

            append_metric(
                rows,
                category="stage_memory",
                metric=(
                    f"{stage}.rss_last_live_sample_mib"
                ),
                value=rss_last,
                unit="MiB",
                source=str(
                    measurement_path
                ),
            )

            append_metric(
                rows,
                category="stage_memory",
                metric=(
                    f"{stage}.peak_rss_mib"
                ),
                value=peak_rss,
                unit="MiB",
                source=str(
                    measurement_path
                ),
                notes=(
                    "Peak process-tree RSS sampled "
                    "during live execution."
                ),
            )

            append_metric(
                rows,
                category="stage_memory",
                metric=(
                    f"{stage}.peak_increment_mib"
                ),
                value=peak_increment,
                unit="MiB",
                source=str(
                    measurement_path
                ),
            )

            append_metric(
                rows,
                category="stage_memory",
                metric=(
                    f"{stage}.peak_process_count"
                ),
                value=peak_process_count,
                unit="processes",
                source=str(
                    measurement_path
                ),
            )

            append_metric(
                rows,
                category="stage_memory",
                metric=(
                    f"{stage}.memory_monitor_wall_time_s"
                ),
                value=measurement.get(
                    "wall_time_s"
                ),
                unit="s",
                source=str(
                    measurement_path
                ),
            )

        else:
            if stage == "lightglue_reranking":
                status = (
                    "NOT_MEASURED_DIAGNOSTIC_EXPENSIVE"
                )

                reason = (
                    "LightGlue is diagnostic-only and its "
                    "full403 historical run takes about "
                    "3.5 hours; it was not rerun solely "
                    "to obtain peak RSS."
                )

            else:
                status = (
                    "PENDING_LIVE_MEASUREMENT"
                )

                reason = (
                    "No successful live memory JSON "
                    "was found for this stage."
                )

            stage_memory[
                stage
            ] = {
                "status": status,
                "source": (
                    str(measurement_path)
                    if measurement_path.exists()
                    else None
                ),
                "rss_first_live_sample_mib": None,
                "rss_last_live_sample_mib": None,
                "peak_root_rss_mib": None,
                "peak_process_tree_rss_mib": None,
                "peak_increment_from_first_sample_mib": None,
                "reason": reason,
            }

            unavailable_stages.append(
                stage
            )

            for suffix in [
                "rss_first_live_sample_mib",
                "rss_last_live_sample_mib",
                "peak_rss_mib",
            ]:
                append_metric(
                    rows,
                    category="stage_memory",
                    metric=f"{stage}.{suffix}",
                    value=None,
                    unit="MiB",
                    status=status,
                    source=(
                        str(measurement_path)
                        if measurement_path.exists()
                        else None
                    ),
                    notes=reason,
                )

    after = process_snapshot()

    # -----------------------------------------------------
    # Cache summary
    # -----------------------------------------------------

    cache_summary = {
        "status": "PASS_CACHE_SIZE_SUMMARY",
        "variant": variant,
        "descriptor_tag": tag,
        "query_descriptor_cache": {
            "path": str(
                query_cache
            ),
            "exists": (
                query_cache.exists()
            ),
            "size_bytes": (
                query_cache_bytes
            ),
            "size_mib": (
                bytes_to_mib(
                    query_cache_bytes
                )
            ),
            "row_count": (
                query_meta.get(
                    "row_count"
                )
            ),
            "descriptor_shape": (
                query_meta.get(
                    "descriptor_shape"
                )
            ),
        },
        "map_descriptor_cache": {
            "path": str(
                map_cache
            ),
            "exists": (
                map_cache.exists()
            ),
            "size_bytes": (
                map_cache_bytes
            ),
            "size_mib": (
                bytes_to_mib(
                    map_cache_bytes
                )
            ),
            "row_count": (
                map_meta.get(
                    "row_count"
                )
            ),
            "descriptor_shape": (
                map_meta.get(
                    "descriptor_shape"
                )
            ),
        },
        "tile_folder": {
            "path": (
                str(tile_folder)
                if tile_folder
                else None
            ),
            "size_bytes": (
                tile_folder_bytes
            ),
            "size_mib": (
                bytes_to_mib(
                    tile_folder_bytes
                )
            ),
            "file_count": (
                tile_file_count
            ),
        },
        "addon_run_output_folder": {
            "path": str(
                run_root
            ),
            "size_bytes": (
                run_root_bytes
            ),
            "size_mib": (
                bytes_to_mib(
                    run_root_bytes
                )
            ),
            "file_count": (
                run_root_files
            ),
        },
        "dataset_output_folder": {
            "path": str(root),
            "size_bytes": (
                dataset_output_bytes
            ),
            "size_mib": (
                bytes_to_mib(
                    dataset_output_bytes
                )
            ),
            "file_count": (
                dataset_output_files
            ),
        },
    }

    # -----------------------------------------------------
    # Main resource summary
    # -----------------------------------------------------

    report = {
        "addon": (
            "Add-on 5 — Memory/resource reporting"
        ),
        "status": (
            "PASS_ADDON5_RESOURCE_REPORTING"
        ),
        "system": system,
        "accelerator": torch_info,
        "devices_used": devices,
        "process_memory": {
            "reporter_before": before,
            "reporter_after": after,
        },
        "workload": workload_items,
        "storage": cache_summary,
        "stage_memory": stage_memory,
        "stage_memory_status": {
            "status": (
                "PASS_BEST_EFFORT_STAGE_MEMORY"
            ),
            "live_measured_count": int(
                len(live_measured_stages)
            ),
            "live_measured_stages": (
                live_measured_stages
            ),
            "not_live_measured_count": int(
                len(unavailable_stages)
            ),
            "not_live_measured_stages": (
                unavailable_stages
            ),
            "lightglue_exception": (
                "LightGlue is diagnostic-only and was "
                "not rerun solely for RAM measurement "
                "because the full run takes about "
                "3.5 hours."
            ),
        },
        "important_rule": (
            "Static cache/device/resource information "
            "is measured now; per-stage peak RAM must "
            "come from live process sampling and is "
            "never inferred from runtime or file size."
        ),
    }

    resource_csv = (
        metrics_dir
        / "resource_summary.csv"
    )

    resource_json = (
        metrics_dir
        / "resource_summary.json"
    )

    cache_json = (
        metrics_dir
        / "cache_size_summary.json"
    )

    pd.DataFrame(
        rows
    ).to_csv(
        resource_csv,
        index=False,
    )

    resource_json.write_text(
        json.dumps(
            report,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    cache_json.write_text(
        json.dumps(
            cache_summary,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    print("=" * 80)
    print(
        "STAGE 6A — ADD-ON 5 "
        "STATIC RESOURCE INVENTORY"
    )
    print("=" * 80)

    print(
        "status: PASS_ADDON5_RESOURCE_REPORTING"
    )

    print()
    print("System")
    print("-" * 80)

    print(
        f"platform:              "
        f"{system['platform']}"
    )

    print(
        f"physical CPU cores:    "
        f"{system['cpu_physical_cores']}"
    )

    print(
        f"logical CPU cores:     "
        f"{system['cpu_logical_cores']}"
    )

    print(
        f"system RAM:            "
        f"{system['system_ram_total_gib']} GiB"
    )

    print(
        f"psutil available:      "
        f"{system['psutil_available']}"
    )

    print()
    print("Accelerator")
    print("-" * 80)

    print(
        f"torch available:       "
        f"{torch_info['torch_available']}"
    )

    print(
        f"CUDA available:        "
        f"{torch_info['cuda_available']}"
    )

    print(
        f"MPS available:         "
        f"{torch_info['mps_available']}"
    )

    print(
        f"GPU names:             "
        f"{torch_info['gpu_names']}"
    )

    print()
    print("Recorded stage devices")
    print("-" * 80)

    for stage, device in (
        devices.items()
    ):
        print(
            f"{stage:20s}: {device}"
        )

    print()
    print("Workload")
    print("-" * 80)

    for key, value in (
        workload_items.items()
    ):
        print(
            f"{key:36s}: {value}"
        )

    print()
    print("Storage")
    print("-" * 80)

    print(
        "query descriptor cache : "
        f"{bytes_to_mib(query_cache_bytes)} MiB"
    )

    print(
        "map descriptor cache   : "
        f"{bytes_to_mib(map_cache_bytes)} MiB"
    )

    print(
        "tile folder            : "
        f"{bytes_to_mib(tile_folder_bytes)} MiB"
    )

    print(
        "add-on run output      : "
        f"{bytes_to_mib(run_root_bytes)} MiB"
    )

    print(
        "dataset output root    : "
        f"{bytes_to_mib(dataset_output_bytes)} MiB"
    )

    print()
    print("Stage peak RAM")
    print("-" * 80)

    for stage in stage_names:
        record = stage_memory[
            stage
        ]

        peak = record.get(
            "peak_process_tree_rss_mib"
        )

        if peak is not None:
            print(
                f"{stage:40s}: "
                f"{float(peak):.3f} MiB"
            )

        else:
            print(
                f"{stage:40s}: "
                f"{record['status']}"
            )

    print()
    print(
        "live measured stages:",
        len(live_measured_stages),
    )

    print(
        "not live measured:",
        unavailable_stages,
    )

    print()
    print("Saved")
    print("-" * 80)

    print(resource_json)
    print(resource_csv)
    print(cache_json)


if __name__ == "__main__":
    main()
