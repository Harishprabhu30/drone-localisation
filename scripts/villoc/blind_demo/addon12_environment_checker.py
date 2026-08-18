#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def add_check(
    checks: list[dict[str, Any]],
    *,
    name: str,
    status: str,
    required: bool,
    detail: str,
    value: Any = None,
) -> None:

    status = status.upper()

    if status not in {
        "PASS",
        "WARN",
        "FAIL",
        "SKIP",
    }:
        raise ValueError(
            f"Invalid check status: {status}"
        )

    checks.append(
        {
            "name": name,
            "status": status,
            "required": bool(required),
            "detail": detail,
            "value": value,
        }
    )

    print(
        f"[{status:4s}] "
        f"{name}: {detail}"
    )


def import_check(
    checks: list[dict[str, Any]],
    *,
    module_name: str,
    label: str,
    required: bool = True,
) -> Any | None:

    try:
        module = importlib.import_module(
            module_name
        )

        version = getattr(
            module,
            "__version__",
            None,
        )

        add_check(
            checks,
            name=label,
            status="PASS",
            required=required,
            detail=(
                "import succeeded"
                + (
                    f" (version {version})"
                    if version
                    else ""
                )
            ),
            value=version,
        )

        return module

    except Exception as exc:

        add_check(
            checks,
            name=label,
            status=(
                "FAIL"
                if required
                else "WARN"
            ),
            required=required,
            detail=(
                f"import failed: "
                f"{type(exc).__name__}: {exc}"
            ),
        )

        return None


def discover_xfeat_path(
    repo_root: Path,
) -> Path | None:

    candidates = [
        repo_root / "third_party/XFeat",
        repo_root / "third_party/xfeat",
        repo_root / "third_party/accelerated_features",
    ]

    for path in candidates:
        if path.exists():
            return path

    third_party = (
        repo_root
        / "third_party"
    )

    if third_party.exists():

        for path in third_party.iterdir():

            if (
                path.is_dir()
                and "xfeat"
                in path.name.lower()
            ):
                return path

    return None


def discover_lightglue_path(
    repo_root: Path,
) -> Path | None:

    candidates = [
        repo_root / "third_party/LightGlue",
        repo_root / "third_party/lightglue",
    ]

    for path in candidates:
        if path.exists():
            return path

    third_party = (
        repo_root
        / "third_party"
    )

    if third_party.exists():

        for path in third_party.iterdir():

            if (
                path.is_dir()
                and "lightglue"
                in path.name.lower()
            ):
                return path

    return None


def discover_map_caches(
    map_root: Path,
    variant: str,
    descriptor_tag: str,
) -> list[Path]:

    if not map_root.exists():
        return []

    variant_low = (
        variant.lower()
    )

    tag_low = (
        descriptor_tag.lower()
    )

    candidates: list[Path] = []

    for path in map_root.rglob("*.npz"):

        low = (
            path.as_posix()
            .lower()
        )

        if (
            variant_low in low
            and tag_low in low
            and "dino" in low
            and "map" in low
        ):
            candidates.append(
                path
            )

    return sorted(
        candidates
    )


def path_in_pythonpath(
    expected: Path,
) -> bool:

    raw = os.environ.get(
        "PYTHONPATH",
        "",
    )

    if not raw:
        return False

    expected = (
        expected.resolve()
    )

    for item in raw.split(
        os.pathsep
    ):

        if not item:
            continue

        try:
            resolved = (
                Path(item)
                .expanduser()
                .resolve()
            )
        except Exception:
            continue

        if resolved == expected:
            return True

    return False


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Add-on 12: blind-demo environment "
            "preflight checker."
        )
    )

    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path("."),
    )

    parser.add_argument(
        "--run-root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--config",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--video",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--map-root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--variant",
        default="512_s256",
    )

    parser.add_argument(
        "--tag",
        required=True,
        help=(
            "Exact promoted DINO descriptor tag, e.g. "
            "dinov2_vits14_img518_center_square_avgpatch_cpu."
        ),
    )

    parser.add_argument(
        "--tile-index",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--map-cache",
        type=Path,
        default=None,
        help=(
            "Optional explicit DINO map descriptor "
            "cache. If omitted, discover under map-root."
        ),
    )

    parser.add_argument(
        "--xfeat-path",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--require-lightglue",
        action="store_true",
        help=(
            "Require LightGlue only for optional "
            "diagnostic runs."
        ),
    )

    args = parser.parse_args()

    started = time.perf_counter()

    repo_root = (
        args.repo_root
        .expanduser()
        .resolve()
    )

    run_root = (
        args.run_root
        .expanduser()
        .resolve()
    )

    config_path = (
        args.config
        .expanduser()
        .resolve()
    )

    video_path = (
        args.video
        .expanduser()
        .resolve()
    )

    map_root = (
        args.map_root
        .expanduser()
        .resolve()
    )

    src_path = (
        repo_root
        / "src"
    )

    tile_index = (
        args.tile_index
        .expanduser()
        .resolve()
        if args.tile_index
        else (
            map_root
            / "metadata"
            / (
                "s8_9_satellite_tile_index_"
                f"{args.variant}.csv"
            )
        )
    )

    metrics_dir = (
        run_root
        / "metrics"
    )

    metrics_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_path = (
        metrics_dir
        / "env_check_report.json"
    )

    checks: list[
        dict[str, Any]
    ] = []

    print("=" * 88)
    print(
        "FINAL ADD-ON 12 — "
        "BLIND DEMO ENVIRONMENT CHECK"
    )
    print("=" * 88)
    print()

    # =========================================================
    # Python / environment
    # =========================================================

    python_ok = (
        sys.version_info
        >= (
            3,
            10,
        )
    )

    add_check(
        checks,
        name="python_version",
        status=(
            "PASS"
            if python_ok
            else "FAIL"
        ),
        required=True,
        detail=(
            f"{platform.python_version()} "
            "(requires >=3.10)"
        ),
        value=(
            platform.python_version()
        ),
    )

    virtual_env = (
        os.environ.get(
            "VIRTUAL_ENV"
        )
    )

    container_env = (
        os.environ.get(
            "DRONELOC_CONTAINER",
            "",
        )
        == "1"
    )

    in_venv = bool(
        virtual_env
        or sys.prefix
        != getattr(
            sys,
            "base_prefix",
            sys.prefix,
        )
        or container_env
    )

    add_check(
        checks,
        name="virtual_environment",
        status=(
            "PASS"
            if in_venv
            else "FAIL"
        ),
        required=True,
        detail=(
            "Container-isloated python environment"
            if container_env
            else (
                virtual_env
                if virtual_env
                else (
                    f"sys.prefix={sys.prefix}"
                    if in_venv
                    else "no active virtual environment detected"
                )
            )
        ),
        value=virtual_env,
    )

    src_exists = (
        src_path.exists()
    )

    add_check(
        checks,
        name="src_directory",
        status=(
            "PASS"
            if src_exists
            else "FAIL"
        ),
        required=True,
        detail=str(
            src_path
        ),
        value=str(
            src_path
        ),
    )

    pythonpath_ok = (
        src_exists
        and path_in_pythonpath(
            src_path
        )
    )

    add_check(
        checks,
        name="pythonpath_src",
        status=(
            "PASS"
            if pythonpath_ok
            else "FAIL"
        ),
        required=True,
        detail=(
            "repository src is present in PYTHONPATH"
            if pythonpath_ok
            else (
                "repository src is not present in PYTHONPATH; "
                "run: export PYTHONPATH=$PWD/src"
            )
        ),
        value=os.environ.get(
            "PYTHONPATH"
        ),
    )

    # =========================================================
    # Core imports
    # =========================================================

    cv2 = import_check(
        checks,
        module_name="cv2",
        label="opencv_import",
    )

    import_check(
        checks,
        module_name="numpy",
        label="numpy_import",
    )

    import_check(
        checks,
        module_name="pandas",
        label="pandas_import",
    )

    import_check(
        checks,
        module_name="matplotlib",
        label="matplotlib_import",
    )

    torch = import_check(
        checks,
        module_name="torch",
        label="torch_import",
    )

    yaml_module = import_check(
        checks,
        module_name="yaml",
        label="pyyaml_import",
    )

    import_check(
        checks,
        module_name="psutil",
        label="psutil_import",
    )

    import_check(
        checks,
        module_name="pyproj",
        label="pyproj_import",
    )

    import_check(
        checks,
        module_name="folium",
        label="folium_import",
    )

    # =========================================================
    # Accelerator / torch environment.
    # CUDA is information, not a requirement:
    # this demonstrated pipeline supports CPU-only execution.
    # =========================================================

    if torch is not None:

        cuda_available = bool(
            torch.cuda.is_available()
        )

        mps_available = bool(
            getattr(
                getattr(
                    torch,
                    "backends",
                    None,
                ),
                "mps",
                None,
            )
            and torch.backends.mps.is_available()
        )

        gpu_names = []

        if cuda_available:

            for i in range(
                torch.cuda.device_count()
            ):
                gpu_names.append(
                    torch.cuda.get_device_name(
                        i
                    )
                )

        add_check(
            checks,
            name="accelerator_state",
            status="PASS",
            required=True,
            detail=(
                f"CUDA={cuda_available}, "
                f"MPS={mps_available}; "
                "CPU-only execution is supported"
            ),
            value={
                "cuda_available":
                    cuda_available,
                "mps_available":
                    mps_available,
                "gpu_names":
                    gpu_names,
            },
        )

        # -----------------------------------------------------
        # DINO/PyTorch Hub readiness.
        # No network request is made here.
        # -----------------------------------------------------

        try:
            hub_dir = Path(
                torch.hub.get_dir()
            ).expanduser().resolve()

            cached_repos = (
                sorted(
                    p.name
                    for p in hub_dir.iterdir()
                    if (
                        p.is_dir()
                        and "dinov2"
                        in p.name.lower()
                    )
                )
                if hub_dir.exists()
                else []
            )

            checkpoint_dir = (
                hub_dir
                / "checkpoints"
            )

            cached_weights = (
                sorted(
                    p.name
                    for p
                    in checkpoint_dir.iterdir()
                    if (
                        p.is_file()
                        and "dino"
                        in p.name.lower()
                    )
                )
                if checkpoint_dir.exists()
                else []
            )

            dino_cached = bool(
                cached_repos
                or cached_weights
            )

            add_check(
                checks,
                name="dinov2_hub_readiness",
                status=(
                    "PASS"
                    if dino_cached
                    else "WARN"
                ),
                required=False,
                detail=(
                    (
                        "local DINO/PyTorch Hub cache detected"
                    )
                    if dino_cached
                    else (
                        "no obvious local DINO hub/cache entry "
                        "detected; map-cache check below is "
                        "still authoritative for prepared map data"
                    )
                ),
                value={
                    "torch_hub_dir":
                        str(
                            hub_dir
                        ),
                    "cached_repositories":
                        cached_repos,
                    "cached_weights":
                        cached_weights,
                },
            )

        except Exception as exc:

            add_check(
                checks,
                name="dinov2_hub_readiness",
                status="WARN",
                required=False,
                detail=(
                    "could not inspect torch hub cache: "
                    f"{type(exc).__name__}: {exc}"
                ),
            )

    # =========================================================
    # XFeat
    # =========================================================

    xfeat_path = (
        args.xfeat_path
        .expanduser()
        .resolve()
        if args.xfeat_path
        else discover_xfeat_path(
            repo_root
        )
    )

    xfeat_ok = bool(
        xfeat_path
        and xfeat_path.exists()
    )

    add_check(
        checks,
        name="xfeat_third_party",
        status=(
            "PASS"
            if xfeat_ok
            else "FAIL"
        ),
        required=True,
        detail=(
            str(
                xfeat_path
            )
            if xfeat_path
            else "XFeat third_party directory not found"
        ),
        value=(
            str(
                xfeat_path
            )
            if xfeat_path
            else None
        ),
    )

    # =========================================================
    # LightGlue — optional unless specifically requested.
    # =========================================================

    lightglue_path = (
        discover_lightglue_path(
            repo_root
        )
    )

    lightglue_importable = False

    try:
        importlib.import_module(
            "lightglue"
        )

        lightglue_importable = True

    except Exception:
        pass

    lightglue_available = bool(
        lightglue_importable
        or (
            lightglue_path
            and lightglue_path.exists()
        )
    )

    if args.require_lightglue:

        add_check(
            checks,
            name="lightglue_optional",
            status=(
                "PASS"
                if lightglue_available
                else "FAIL"
            ),
            required=True,
            detail=(
                "LightGlue diagnostic dependency available"
                if lightglue_available
                else (
                    "LightGlue requested but neither import "
                    "nor third_party path was found"
                )
            ),
            value=(
                str(
                    lightglue_path
                )
                if lightglue_path
                else (
                    "importable"
                    if lightglue_importable
                    else None
                )
            ),
        )

    else:

        add_check(
            checks,
            name="lightglue_optional",
            status=(
                "PASS"
                if lightglue_available
                else "SKIP"
            ),
            required=False,
            detail=(
                "available but not required by promoted ORB demo"
                if lightglue_available
                else (
                    "not requested; promoted blind demo uses ORB "
                    "rather than LightGlue"
                )
            ),
            value=(
                str(
                    lightglue_path
                )
                if lightglue_path
                else None
            ),
        )

    # =========================================================
    # Config
    # =========================================================

    config_exists = (
        config_path.exists()
        and config_path.is_file()
    )

    add_check(
        checks,
        name="config_file",
        status=(
            "PASS"
            if config_exists
            else "FAIL"
        ),
        required=True,
        detail=str(
            config_path
        ),
        value=str(
            config_path
        ),
    )

    if (
        config_exists
        and yaml_module is not None
    ):

        try:
            parsed_config = (
                yaml_module.safe_load(
                    config_path.read_text(
                        encoding="utf-8"
                    )
                )
            )

            config_valid = isinstance(
                parsed_config,
                dict,
            )

            add_check(
                checks,
                name="config_parse",
                status=(
                    "PASS"
                    if config_valid
                    else "FAIL"
                ),
                required=True,
                detail=(
                    "YAML parsed as mapping"
                    if config_valid
                    else (
                        "YAML parsed but top level "
                        "is not a mapping"
                    )
                ),
                value=(
                    sorted(
                        parsed_config.keys()
                    )
                    if config_valid
                    else None
                ),
            )

        except Exception as exc:

            add_check(
                checks,
                name="config_parse",
                status="FAIL",
                required=True,
                detail=(
                    f"{type(exc).__name__}: {exc}"
                ),
            )

    # =========================================================
    # Video
    # =========================================================

    video_exists = (
        video_path.exists()
        and video_path.is_file()
    )

    add_check(
        checks,
        name="video_file",
        status=(
            "PASS"
            if video_exists
            else "FAIL"
        ),
        required=True,
        detail=str(
            video_path
        ),
        value=(
            video_path.stat().st_size
            if video_exists
            else None
        ),
    )

    if (
        video_exists
        and cv2 is not None
    ):

        cap = cv2.VideoCapture(
            str(
                video_path
            )
        )

        opened = bool(
            cap.isOpened()
        )

        if opened:

            fps = float(
                cap.get(
                    cv2.CAP_PROP_FPS
                )
            )

            frame_count = int(
                cap.get(
                    cv2.CAP_PROP_FRAME_COUNT
                )
            )

            width = int(
                cap.get(
                    cv2.CAP_PROP_FRAME_WIDTH
                )
            )

            height = int(
                cap.get(
                    cv2.CAP_PROP_FRAME_HEIGHT
                )
            )

            cap.release()

            metadata_ok = (
                fps > 0
                and frame_count > 0
                and width > 0
                and height > 0
            )

            add_check(
                checks,
                name="video_readability",
                status=(
                    "PASS"
                    if metadata_ok
                    else "FAIL"
                ),
                required=True,
                detail=(
                    f"{frame_count} frames, "
                    f"{fps:.6f} fps, "
                    f"{width}x{height}"
                ),
                value={
                    "fps":
                        fps,
                    "frame_count":
                        frame_count,
                    "width":
                        width,
                    "height":
                        height,
                },
            )

        else:

            cap.release()

            add_check(
                checks,
                name="video_readability",
                status="FAIL",
                required=True,
                detail=(
                    "OpenCV could not open video"
                ),
            )

    # =========================================================
    # Map root / tile index
    # =========================================================

    map_root_ok = (
        map_root.exists()
        and map_root.is_dir()
    )

    add_check(
        checks,
        name="map_root",
        status=(
            "PASS"
            if map_root_ok
            else "FAIL"
        ),
        required=True,
        detail=str(
            map_root
        ),
        value=str(
            map_root
        ),
    )

    tile_index_ok = (
        tile_index.exists()
        and tile_index.is_file()
    )

    add_check(
        checks,
        name="map_tile_index",
        status=(
            "PASS"
            if tile_index_ok
            else "FAIL"
        ),
        required=True,
        detail=str(
            tile_index
        ),
        value=str(
            tile_index
        ),
    )

    if tile_index_ok:

        try:
            tile_df = pd.read_csv(
                tile_index
            )

            tile_valid = (
                len(
                    tile_df
                )
                > 0
            )

            add_check(
                checks,
                name="map_tile_index_readability",
                status=(
                    "PASS"
                    if tile_valid
                    else "FAIL"
                ),
                required=True,
                detail=(
                    f"{len(tile_df)} tile rows, "
                    f"{len(tile_df.columns)} columns"
                ),
                value={
                    "rows":
                        len(
                            tile_df
                        ),
                    "columns":
                        list(
                            tile_df.columns
                        ),
                },
            )

        except Exception as exc:

            add_check(
                checks,
                name="map_tile_index_readability",
                status="FAIL",
                required=True,
                detail=(
                    f"{type(exc).__name__}: {exc}"
                ),
            )

    # =========================================================
    # Prepared DINO map descriptor cache.
    # =========================================================

    if args.map_cache:

        explicit_cache = (
            args.map_cache
            .expanduser()
            .resolve()
        )

        map_caches = (
            [
                explicit_cache
            ]
            if explicit_cache.exists()
            else []
        )

    else:

        map_caches = (
            discover_map_caches(
                map_root,
                args.variant,
                args.tag,
            )
        )

    cache_ok = (
        len(
            map_caches
        )
        > 0
    )

    add_check(
        checks,
        name="dino_map_cache",
        status=(
            "PASS"
            if cache_ok
            else "FAIL"
        ),
        required=True,
        detail=(
            (
                f"{len(map_caches)} candidate cache(s); "
                f"using {map_caches[0]}"
            )
            if cache_ok
            else (
                "no DINO map .npz cache found for "
                f"variant {args.variant} and "
                f"descriptor tag {args.tag}"
            )
        ),
        value=[
            str(
                p
            )
            for p in map_caches
        ],
    )

    if cache_ok:

        selected_cache = (
            map_caches[
                0
            ]
        )

        try:
            with np.load(
                selected_cache,
                allow_pickle=False,
            ) as npz:

                keys = list(
                    npz.files
                )

            add_check(
                checks,
                name="dino_map_cache_readability",
                status=(
                    "PASS"
                    if keys
                    else "FAIL"
                ),
                required=True,
                detail=(
                    f"NPZ readable; keys={keys}"
                ),
                value={
                    "path":
                        str(
                            selected_cache
                        ),
                    "size_bytes":
                        selected_cache.stat().st_size,
                    "keys":
                        keys,
                },
            )

        except Exception as exc:

            add_check(
                checks,
                name="dino_map_cache_readability",
                status="FAIL",
                required=True,
                detail=(
                    f"{type(exc).__name__}: {exc}"
                ),
            )

    # =========================================================
    # Output writability.
    # =========================================================

    try:
        run_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        metrics_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        with tempfile.NamedTemporaryFile(
            mode="w",
            dir=metrics_dir,
            prefix=".env_check_write_test_",
            delete=False,
            encoding="utf-8",
        ) as f:

            f.write(
                "environment check write test\n"
            )

            temp_path = Path(
                f.name
            )

        temp_path.unlink()

        add_check(
            checks,
            name="output_writable",
            status="PASS",
            required=True,
            detail=(
                f"write/delete test passed in {metrics_dir}"
            ),
            value=str(
                metrics_dir
            ),
        )

    except Exception as exc:

        add_check(
            checks,
            name="output_writable",
            status="FAIL",
            required=True,
            detail=(
                f"{type(exc).__name__}: {exc}"
            ),
            value=str(
                metrics_dir
            ),
        )

    # =========================================================
    # Final result.
    # =========================================================

    required_failures = [
        row
        for row in checks
        if (
            row[
                "required"
            ]
            and row[
                "status"
            ]
            == "FAIL"
        )
    ]

    warnings = [
        row
        for row in checks
        if row[
            "status"
        ]
        == "WARN"
    ]

    skipped = [
        row
        for row in checks
        if row[
            "status"
        ]
        == "SKIP"
    ]

    passed = [
        row
        for row in checks
        if row[
            "status"
        ]
        == "PASS"
    ]

    overall_status = (
        "PASS_ADDON12_ENVIRONMENT_CHECK"
        if not required_failures
        else "FAIL_ADDON12_ENVIRONMENT_CHECK"
    )

    runtime_s = (
        time.perf_counter()
        - started
    )

    report = {
        "stage": (
            "ADDON12_ENVIRONMENT_CHECK"
        ),
        "status":
            overall_status,
        "blind_safe":
            True,
        "evaluation_or_reference_read":
            False,
        "repo_root":
            str(
                repo_root
            ),
        "run_root":
            str(
                run_root
            ),
        "variant":
            args.variant,
        "descriptor_tag":
            args.tag,
        "summary": {
            "pass_count":
                len(
                    passed
                ),
            "warn_count":
                len(
                    warnings
                ),
            "skip_count":
                len(
                    skipped
                ),
            "required_fail_count":
                len(
                    required_failures
                ),
        },
        "checks":
            checks,
        "runtime": {
            "environment_check_s":
                float(
                    runtime_s
                ),
        },
        "contract": {
            "cuda_required":
                False,
            "lightglue_required":
                bool(
                    args.require_lightglue
                ),
            "ground_truth_required":
                False,
            "srt_required":
                False,
            "prepared_map_cache_required":
                True,
            "map_tile_index_required":
                True,
        },
    }

    report_path.write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 88)
    print("ENVIRONMENT CHECK SUMMARY")
    print("=" * 88)

    print(
        "PASS          :",
        len(
            passed
        ),
    )

    print(
        "WARN          :",
        len(
            warnings
        ),
    )

    print(
        "SKIP          :",
        len(
            skipped
        ),
    )

    print(
        "REQUIRED FAIL :",
        len(
            required_failures
        ),
    )

    print(
        "runtime       :",
        f"{runtime_s:.6f} s",
    )

    print(
        "report        :",
        report_path,
    )

    print()
    print(
        "status:",
        overall_status,
    )

    if required_failures:
        raise SystemExit(
            2
        )


if __name__ == "__main__":
    main()
