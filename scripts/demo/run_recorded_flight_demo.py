'''
1. dry run:

source .drone_venv/bin/activate
export PYTHONPATH=$PWD/src

python \
  scripts/demo/run_recorded_flight_demo.py \
  --config configs/demo_villoc_traj01_blind_regression.yaml \
  --dry-run

2. Full run:

source .drone_venv/bin/activate
export PYTHONPATH=$PWD/src

python \
  scripts/demo/run_recorded_flight_demo.py \
  --config configs/demo_villoc_traj01_blind_regression.yaml

'''

#!/usr/bin/env python3

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml


# ============================================================
# Frozen blind-demo stage order
# ============================================================

STAGES = [
    (
        "01_environment_preflight",
        "scripts/villoc/blind_demo/"
        "addon12_environment_checker.py",
    ),
    (
        "02_blind_query_manifest",
        "scripts/villoc/blind_demo/"
        "addon7_blind_query_manifest.py",
    ),
    (
        "03_xfeat_relative_frontend",
        "scripts/villoc/"
        "s8_r4_xfeat_relative_frontend.py",
    ),
    (
        "04_dino_query_cache",
        "scripts/villoc/"
        "s8_11bc_build_dinov2_caches.py",
    ),
    (
        "05_dino_topk_retrieval",
        "scripts/villoc/blind_demo/"
        "stage6_blind_dino_topk_retrieval.py",
    ),
    (
        "06_orb_topk_verification",
        "scripts/villoc/"
        "s8_12e1_top20_verifier_reranker.py",
    ),
    (
        "07_blind_map_bootstrap",
        "scripts/villoc/blind_demo/"
        "stage10b2_bootstrap_backend.py",
    ),
    (
        "08_map_alignment",
        "scripts/villoc/blind_demo/"
        "stage10b3_map_alignment_router.py",
    ),
    (
        "09_temporal_fusion",
        "scripts/villoc/blind_demo/"
        "stage10b4_temporal_router.py",
    ),
    (
        "10_estimated_output_export",
        "scripts/villoc/blind_demo/"
        "addon9_estimated_output_router.py",
    ),
    (
        "11_blind_safe_visuals",
        "scripts/villoc/blind_demo/"
        "addon10_no_reference_visuals.py",
    ),
    (
        "12_runtime_registry_initial",
        "scripts/villoc/blind_demo/"
        "addon4_runtime_benchmark.py",
    ),
    (
        "13_resource_registry",
        "scripts/villoc/blind_demo/"
        "addon5_resource_reporting.py",
    ),
    (
        "14_deployment_breakdown",
        "scripts/villoc/blind_demo/"
        "addon6_deployment_cost_breakdown.py",
    ),
    (
        "15_markdown_run_summary",
        "scripts/villoc/blind_demo/"
        "addon11_run_summary.py",
    ),
    (
        "16_runtime_registry_refresh",
        "scripts/villoc/blind_demo/"
        "addon4_runtime_benchmark.py",
    ),
    (
        "17_freeze_blind_output",
        "scripts/villoc/blind_demo/"
        "stage10b5d_freeze_blind_submission.py",
    ),
]


def require(
    condition: bool,
    message: str,
) -> None:

    if not condition:
        raise RuntimeError(
            message
        )


def resolve_repo_path(
    repo_root: Path,
    value: str | Path,
) -> Path:

    path = Path(
        value
    ).expanduser()

    if not path.is_absolute():
        path = (
            repo_root
            / path
        )

    return path.resolve()


def load_config(
    path: Path,
) -> dict[str, Any]:

    data = yaml.safe_load(
        path.read_text(
            encoding="utf-8"
        )
    )

    if not isinstance(
        data,
        dict,
    ):
        raise RuntimeError(
            "Demo YAML must contain "
            "a top-level mapping."
        )

    return data


def descriptor_protocol(
    tag: str,
) -> dict[str, Any]:

    # Current frozen tag:
    #
    # dinov2_vits14_img518_
    # center_square_avgpatch_cpu
    #
    # The tag is already the canonical descriptor
    # protocol identifier, so do not duplicate these
    # low-level values in the operational YAML.

    size_match = re.search(
        r"_img(\d+)_",
        tag,
    )

    require(
        size_match is not None,
        (
            "Descriptor tag does not encode "
            f"image size: {tag}"
        ),
    )

    image_size = int(
        size_match.group(1)
    )

    if "center_square" in tag:
        crop_mode = (
            "center_square"
        )
    else:
        raise RuntimeError(
            "Unsupported descriptor crop "
            f"protocol in tag: {tag}"
        )

    if "avgpatch" in tag:
        pooling = "avgpatch"
    else:
        raise RuntimeError(
            "Unsupported descriptor pooling "
            f"protocol in tag: {tag}"
        )

    if tag.endswith(
        "_cpu"
    ):
        device = "cpu"
    else:
        raise RuntimeError(
            "Current promoted demo expects "
            "the frozen CPU descriptor tag."
        )

    return {
        "image_size":
            image_size,

        "crop_mode":
            crop_mode,

        "pooling":
            pooling,

        "device":
            device,
    }




class TeeStream:
    """
    Duplicate orchestrator stdout/stderr to:
      1. the real terminal
      2. one whole-run transcript file

    Per-stage logs remain separate and unchanged.
    """

    def __init__(
        self,
        *streams,
    ):
        self.streams = streams

    def write(
        self,
        data: str,
    ) -> int:

        for stream in self.streams:
            stream.write(
                data
            )
            stream.flush()

        return len(
            data
        )

    def flush(
        self,
    ) -> None:

        for stream in self.streams:
            stream.flush()

    def isatty(
        self,
    ) -> bool:

        return any(
            bool(
                getattr(
                    stream,
                    "isatty",
                    lambda: False,
                )()
            )
            for stream in self.streams
        )

    @property
    def encoding(
        self,
    ) -> str:

        return str(
            getattr(
                self.streams[0],
                "encoding",
                "utf-8",
            )
        )


def run_stage(
    *,
    index: int,
    stage_name: str,
    command: list[str],
    repo_root: Path,
    run_root: Path,
    env: dict[str, str],
) -> None:
    """
    Execute one promoted demo stage.

    Contract:
      - same Python interpreter as orchestrator
      - stdout/stderr streamed to terminal
      - same output simultaneously written to stage log
      - non-zero return code stops orchestration immediately
    """

    log_dir = (
        run_root
        / "logs"
    )

    log_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    log_path = (
        log_dir
        / (
            f"orchestrator_{index:02d}_"
            f"{stage_name}.log"
        )
    )

    command_text = (
        shlex.join(
            command
        )
    )

    print()
    print("=" * 100)
    print(
        f"ORCHESTRATOR STAGE {index:02d} — "
        f"{stage_name}"
    )
    print("=" * 100)

    print()
    print(
        "command:"
    )
    print(
        command_text
    )

    print()
    print(
        "log:"
    )
    print(
        log_path
    )

    print()

    with log_path.open(
        "w",
        encoding="utf-8",
    ) as log:

        log.write(
            f"stage={stage_name}\n"
        )

        log.write(
            f"index={index}\n"
        )

        log.write(
            f"python={sys.executable}\n"
        )

        log.write(
            f"command={command_text}\n"
        )

        log.write(
            "=" * 100
            + "\n"
        )

        log.flush()

        process = subprocess.Popen(
            command,
            cwd=repo_root,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        if process.stdout is None:
            raise RuntimeError(
                "Could not capture subprocess output."
            )

        for line in process.stdout:

            print(
                line,
                end="",
                flush=True,
            )

            log.write(
                line
            )

            log.flush()

        return_code = (
            process.wait()
        )

        log.write(
            "\n"
            + "=" * 100
            + "\n"
        )

        log.write(
            f"return_code={return_code}\n"
        )

    if return_code != 0:

        print()
        print("!" * 100)
        print(
            "FAIL-FAST ORCHESTRATION STOP"
        )
        print("!" * 100)

        print(
            "failed stage :",
            stage_name,
        )

        print(
            "stage index  :",
            index,
        )

        print(
            "return code  :",
            return_code,
        )

        print(
            "stage log    :",
            log_path,
        )

        raise RuntimeError(
            "Blind-demo orchestration stopped "
            f"at stage {index:02d} "
            f"({stage_name})."
        )

    print()
    print(
        f"PASS ORCHESTRATOR STAGE "
        f"{index:02d}: {stage_name}"
    )


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "One-command blind recorded-flight "
            "Villoc localization orchestrator."
        )
    )

    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help=(
            "Base blind-demo YAML."
        ),
    )

    parser.add_argument(
        "--video",
        type=Path,
        default=None,
        help=(
            "Optional video override. "
            "Otherwise YAML dataset.raw_root + "
            "streams.V.video is used."
        ),
    )

    parser.add_argument(
        "--run-id",
        default=None,
        help=(
            "Optional fresh run ID under the "
            "configured demo-run parent directory."
        ),
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Resolve and validate the orchestration "
            "contract without executing any stage."
        ),
    )

    return parser.parse_args()


def main() -> None:

    args = parse_args()

    repo_root = (
        Path.cwd()
        .resolve()
    )

    config_path = (
        args.config
        .expanduser()
        .resolve()
    )

    require(
        config_path.exists(),
        f"Config not found: {config_path}",
    )

    cfg = load_config(
        config_path
    )

    # ========================================================
    # Blind-contract validation
    # ========================================================

    blind = cfg.get(
        "blind",
        {},
    )

    streams = cfg.get(
        "streams",
        {},
    )

    stream_v = streams.get(
        "V",
        {},
    )

    require(
        blind.get(
            "enabled"
        )
        is True,
        (
            "Orchestrator currently supports "
            "blind mode only."
        ),
    )

    require(
        stream_v.get(
            "reference_available"
        )
        is False,
        (
            "Blind demo config must declare "
            "reference_available=false."
        ),
    )

    require(
        blind.get(
            "freeze_before_evaluation"
        )
        is True,
        (
            "Blind demo must freeze output "
            "before any later evaluation."
        ),
    )

    # ========================================================
    # Dataset / run paths
    # ========================================================

    dataset = cfg.get(
        "dataset",
        {},
    )

    raw_root = resolve_repo_path(
        repo_root,
        dataset[
            "raw_root"
        ],
    )

    configured_output_root = (
        resolve_repo_path(
            repo_root,
            dataset[
                "output_root"
            ],
        )
    )

    if args.run_id:

        run_root = (
            configured_output_root
            .parent
            / args.run_id
        ).resolve()

    else:

        run_root = (
            configured_output_root
        )

    if args.video is not None:

        video = (
            args.video
            .expanduser()
        )

        if not video.is_absolute():
            video = (
                repo_root
                / video
            )

        video = (
            video.resolve()
        )

    else:

        video = (
            raw_root
            / str(
                stream_v[
                    "video"
                ]
            )
        ).resolve()

    require(
        video.exists(),
        f"Video not found: {video}",
    )

    # ========================================================
    # Map / descriptor contract
    # ========================================================

    map_cfg = cfg.get(
        "map",
        {},
    )

    retrieval = cfg.get(
        "retrieval",
        {},
    )

    export_cfg = cfg.get(
        "export",
        {},
    )

    variant = str(
        map_cfg[
            "primary_variant"
        ]
    )

    top_k = int(
        retrieval[
            "top_k"
        ]
    )

    tag = str(
        retrieval[
            "descriptor_tag"
        ]
    )

    protocol = descriptor_protocol(
        tag
    )

    tile_cfg = (
        map_cfg[
            "tile_variants"
        ][
            variant
        ]
    )

    tile_index = (
        resolve_repo_path(
            repo_root,
            tile_cfg[
                "index_csv"
            ],
        )
    )

    tile_dir = (
        resolve_repo_path(
            repo_root,
            tile_cfg[
                "tiles_dir"
            ],
        )
    )

    map_cache = (
        resolve_repo_path(
            repo_root,
            retrieval[
                "map_descriptor_cache"
            ],
        )
    )

    map_aoi = (
        resolve_repo_path(
            repo_root,
            map_cfg[
                "canonical_aoi_tif"
            ],
        )
    )

    for (
        label,
        path,
    ) in [
        (
            "tile index",
            tile_index,
        ),
        (
            "tile directory",
            tile_dir,
        ),
        (
            "map cache",
            map_cache,
        ),
        (
            "map AOI",
            map_aoi,
        ),
    ]:

        require(
            path.exists(),
            f"{label} not found: {path}",
        )

    # outputs/villoc/90_deg/
    #
    # map cache:
    # outputs/villoc/90_deg/descriptors/file.npz

    map_root = (
        map_cache
        .parent
        .parent
    )

    map_crs = str(
        map_cfg[
            "crs"
        ]
    )

    target_crs = str(
        export_cfg[
            "target_crs"
        ]
    )

    require(
        top_k > 0,
        "retrieval.top_k must be > 0",
    )

    require(
        retrieval.get(
            "method"
        )
        == "dinov2",
        (
            "Current orchestrator contract "
            "expects retrieval.method=dinov2."
        ),
    )

    require(
        retrieval.get(
            "verifier"
        )
        == "orb_top20",
        (
            "Current orchestrator contract "
            "expects verifier=orb_top20."
        ),
    )

    require(
        cfg.get(
            "relative_localization",
            {},
        ).get(
            "frontend"
        )
        == "xfeat",
        (
            "Current orchestrator contract "
            "expects the promoted XFeat frontend."
        ),
    )

    require(
        cfg.get(
            "fusion",
            {},
        ).get(
            "method"
        )
        == "causal_temporal_soft_correction",
        (
            "Unexpected fusion method."
        ),
    )

    # ========================================================
    # Runtime-config snapshot contract
    # ========================================================

    runtime_cfg = copy.deepcopy(
        cfg
    )

    runtime_cfg[
        "dataset"
    ][
        "output_root"
    ] = str(
        run_root
    )

    # If --video overrides the base YAML, record that
    # resolved source in the per-run configuration snapshot.
    # The source YAML itself is never modified.
    if args.video is not None:

        runtime_cfg[
            "dataset"
        ][
            "raw_root"
        ] = str(
            video.parent
        )

        runtime_cfg[
            "streams"
        ][
            "V"
        ][
            "video"
        ] = (
            video.name
        )

    runtime_cfg_path = (
        run_root
        / "config_resolved.yaml"
    )

    # Stage 9A is dry-run only.
    # The actual file will be written by Stage 9B
    # immediately before execution of a fresh run.

    # ========================================================
    # Canonical run-relative artifacts
    # ========================================================

    artifacts = {
        "blind_manifest": (
            run_root
            / "metadata"
            / "blind_query_manifest.csv"
        ),

        "dino_query_cache": (
            run_root
            / "descriptors"
            / (
                "s8_11c_dinov2_queries_v_1fps_"
                f"{tag}.npz"
            )
        ),

        "orb_output_root": (
            run_root
            / "reports"
            / "s8_12e1_top20_verifier_reranker"
            / (
                f"{variant}_"
                "orb_hybrid_top20_img518"
            )
        ),

        "bootstrap_report": (
            run_root
            / "reports"
            / "blind_map_bootstrap"
            / "blind_map_bootstrap_report.json"
        ),

        "map_alignment_report": (
            run_root
            / "reports"
            / "blind_map_alignment"
            / "blind_map_alignment_report.json"
        ),

        "temporal_report": (
            run_root
            / "reports"
            / "blind_temporal_fusion"
            / "blind_temporal_fusion_report.json"
        ),

        "submission": (
            run_root
            / "trajectories"
            / "submission_estimated_trajectory.csv"
        ),

        "timing_summary": (
            run_root
            / "metrics"
            / "timing_summary.json"
        ),

        "resource_summary": (
            run_root
            / "metrics"
            / "resource_summary.json"
        ),

        "deployment_summary": (
            run_root
            / "metrics"
            / "deployment_cost_breakdown.json"
        ),

        "markdown_summary": (
            run_root
            / "summary"
            / "demo_run_summary.md"
        ),

        "freeze_record": (
            run_root
            / "evaluation"
            / "blind_submission_freeze.json"
        ),
    }

    # ========================================================
    # Source-script contract
    # ========================================================

    missing_scripts = []

    resolved_stages = []

    for (
        stage_name,
        script_rel,
    ) in STAGES:

        script_path = (
            repo_root
            / script_rel
        ).resolve()

        resolved_stages.append(
            (
                stage_name,
                script_path,
            )
        )

        if not script_path.exists():
            missing_scripts.append(
                script_path
            )

    require(
        not missing_scripts,
        (
            "Orchestrator stage scripts missing:\n"
            + "\n".join(
                str(p)
                for p in missing_scripts
            )
        ),
    )

    # ========================================================
    # Console contract
    # ========================================================

    print("=" * 100)
    print(
        "STAGE 9A — BLIND RECORDED-FLIGHT "
        "ORCHESTRATOR DRY-RUN"
    )
    print("=" * 100)

    print()
    print("Mode")
    print("-" * 100)

    print(
        "blind mode                  :",
        True,
    )

    print(
        "reference available         :",
        False,
    )

    print(
        "freeze before evaluation    :",
        True,
    )

    print()
    print("Resolved dataset")
    print("-" * 100)

    print(
        "base config                 :",
        config_path,
    )

    print(
        "runtime config snapshot     :",
        runtime_cfg_path,
    )

    print(
        "source video                :",
        video,
    )

    print(
        "run root                    :",
        run_root,
    )

    print(
        "run root currently exists   :",
        run_root.exists(),
    )

    print()
    print("Map / retrieval")
    print("-" * 100)

    print(
        "map root                    :",
        map_root,
    )

    print(
        "map AOI                     :",
        map_aoi,
    )

    print(
        "variant                     :",
        variant,
    )

    print(
        "Top-K                       :",
        top_k,
    )

    print(
        "tile index                  :",
        tile_index,
    )

    print(
        "map descriptor cache        :",
        map_cache,
    )

    print(
        "map CRS                     :",
        map_crs,
    )

    print(
        "target CRS                  :",
        target_crs,
    )

    print()
    print("DINO descriptor protocol")
    print("-" * 100)

    for key, value in (
        protocol.items()
    ):
        print(
            f"{key:28s}:",
            value,
        )

    print()
    print("Frozen execution order")
    print("-" * 100)

    for (
        index,
        (
            stage_name,
            script_path,
        ),
    ) in enumerate(
        resolved_stages,
        start=1,
    ):

        print(
            f"{index:02d}. "
            f"{stage_name:34s} "
            f"{script_path.relative_to(repo_root)}"
        )

    print()
    print("Canonical outputs")
    print("-" * 100)

    for (
        name,
        path,
    ) in artifacts.items():

        print(
            f"{name:28s}:",
            path,
        )

    print()
    print("Execution safety")
    print("-" * 100)

    print(
        "existing run overwrite      : REFUSED"
    )

    print(
        "reference/SRT input         : NOT SUPPORTED "
        "IN CURRENT BLIND ORCHESTRATOR"
    )

    print(
        "LightGlue                   : NOT PART OF "
        "PROMOTED DEMO"
    )

    print(
        "HTML summary                : DISABLED"
    )

    print(
        "map HTML                    : GENERATED ONLY "
        "WHEN LAT/LON EXISTS"
    )

    print()
    print("Stage 9A repository writes")
    print("-" * 100)

    print(
        "run directory created       : False"
    )

    print(
        "runtime YAML written        : False"
    )

    print(
        "localization executed       : False"
    )

    if args.dry_run:

        print()
        print(
            "STATUS: "
            "PASS_DEMO_STAGE9A_ORCHESTRATOR_DRY_RUN"
        )

        return

    # ========================================================
    # Stage 9B — guarded execution
    # ========================================================

    # A run root is immutable from the orchestrator's point
    # of view. Existing runs must never be silently reused or
    # overwritten.
    require(
        not run_root.exists(),
        (
            "Refusing to execute into an existing "
            f"run root: {run_root}\n"
            "Choose a fresh --run-id."
        ),
    )

    run_root.mkdir(
        parents=True,
        exist_ok=False,
    )

    logs_dir = (
        run_root
        / "logs"
    )

    logs_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Freeze the resolved operational config for this run.
    # --------------------------------------------------------

    runtime_cfg_path.write_text(
        yaml.safe_dump(
            runtime_cfg,
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Use the same active Python environment for every stage.
    # --------------------------------------------------------

    child_env = (
        os.environ.copy()
    )

    src_path = (
        repo_root
        / "src"
    )

    old_pythonpath = (
        child_env.get(
            "PYTHONPATH",
            "",
        )
    )

    child_env[
        "PYTHONPATH"
    ] = (
        str(
            src_path
        )
        + (
            os.pathsep
            + old_pythonpath
            if old_pythonpath
            else ""
        )
    )

    child_env[
        "PYTHONUNBUFFERED"
    ] = "1"

    py = (
        sys.executable
    )

    # ========================================================
    # Canonical internal artifact paths
    # ========================================================

    xfeat_repo = (
        repo_root
        / "third_party"
        / "accelerated_features"
    ).resolve()

    require(
        xfeat_repo.exists(),
        (
            "XFeat repository not found: "
            f"{xfeat_repo}"
        ),
    )

    sequence_name = str(
        dataset.get(
            "sequence_name",
            "blind_recorded_flight",
        )
    )

    sample_rate_fps = float(
        blind[
            "sample_rate_fps"
        ]
    )

    assumed_rel_alt_m = float(
        blind[
            "assumed_rel_alt_m"
        ]
    )

    assumed_pitch_deg = float(
        blind[
            "assumed_gimbal_pitch_deg"
        ]
    )

    view_assumption = str(
        blind.get(
            "view_assumption",
            "near_nadir",
        )
    )

    blind_manifest = (
        artifacts[
            "blind_manifest"
        ]
    )

    raw_relative = (
        run_root
        / "metadata"
        / "s8_xfeat_relative_frontend"
        / "s8r4_xfeat_relative_trajectory_blind_raw.csv"
    )

    topk_csv = (
        run_root
        / "retrieval"
        / "s8_11d"
        / (
            "s8_11d_topk_"
            f"{variant}_"
            f"{tag}.csv"
        )
    )

    orb_output_root = (
        artifacts[
            "orb_output_root"
        ]
    )

    orb_query_summary = (
        orb_output_root
        / "s8_12e1_query_summary.csv"
    )

    orb_candidate_scores = (
        orb_output_root
        / "s8_12e1_all_candidate_verifier_scores.csv"
    )

    bootstrap_report = (
        artifacts[
            "bootstrap_report"
        ]
    )

    map_trajectory = (
        run_root
        / "trajectories"
        / "blind_map_aligned_relative_trajectory.csv"
    )

    temporal_manifest = (
        run_root
        / "metadata"
        / "blind_temporal_fusion"
        / "blind_temporal_correction_manifest.csv"
    )

    temporal_fused = (
        run_root
        / "trajectories"
        / "blind_temporal_fused_trajectory.csv"
    )

    temporal_report = (
        artifacts[
            "temporal_report"
        ]
    )

    submission = (
        artifacts[
            "submission"
        ]
    )

    addon9_report = (
        run_root
        / "reports"
        / "addon9_estimated_latlon"
        / "estimated_latlon_export_report.json"
    )

    map_cache_root = (
        map_cache.parent
    )

    # ========================================================
    # Exact promoted 17-stage command chain
    # ========================================================

    commands: list[
        tuple[
            str,
            list[str],
        ]
    ] = [

        # ----------------------------------------------------
        # 01 — environment preflight
        # ----------------------------------------------------
        (
            "01_environment_preflight",
            [
                py,
                str(
                    repo_root
                    / STAGES[0][1]
                ),
                "--repo-root",
                str(
                    repo_root
                ),
                "--run-root",
                str(
                    run_root
                ),
                "--config",
                str(
                    runtime_cfg_path
                ),
                "--video",
                str(
                    video
                ),
                "--map-root",
                str(
                    map_root
                ),
                "--variant",
                variant,
                "--tag",
                tag,
                "--tile-index",
                str(
                    tile_index
                ),
                "--map-cache",
                str(
                    map_cache
                ),
                "--xfeat-path",
                str(
                    xfeat_repo
                ),
            ],
        ),

        # ----------------------------------------------------
        # 02 — blind query preparation
        # ----------------------------------------------------
        (
            "02_blind_query_manifest",
            [
                py,
                str(
                    repo_root
                    / STAGES[1][1]
                ),
                "--video",
                str(
                    video
                ),
                "--run-root",
                str(
                    run_root
                ),
                "--sample-rate-fps",
                str(
                    sample_rate_fps
                ),
                "--assumed-rel-alt-m",
                str(
                    assumed_rel_alt_m
                ),
                "--assumed-gimbal-pitch-deg",
                str(
                    assumed_pitch_deg
                ),
                "--view-assumption",
                view_assumption,
                "--map-aoi",
                str(
                    map_aoi
                ),
                "--map-tile-index",
                str(
                    tile_index
                ),
            ],
        ),

        # ----------------------------------------------------
        # 03 — XFeat relative frontend
        #
        # Low-level XFeat parameters deliberately remain
        # the validated script defaults.
        # ----------------------------------------------------
        (
            "03_xfeat_relative_frontend",
            [
                py,
                str(
                    repo_root
                    / STAGES[2][1]
                ),
                "--repo-root",
                str(
                    repo_root
                ),
                "--manifest",
                str(
                    blind_manifest
                ),
                "--output-root",
                str(
                    run_root
                ),
                "--xfeat-repo",
                str(
                    xfeat_repo
                ),
                "--sequence",
                sequence_name,
                "--device",
                str(
                    protocol[
                        "device"
                    ]
                ),
                "--blind-only",
            ],
        ),

        # ----------------------------------------------------
        # 04 — DINO query cache
        #
        # Query cache is per flight.
        # Map cache is reused.
        # ----------------------------------------------------
        (
            "04_dino_query_cache",
            [
                py,
                str(
                    repo_root
                    / STAGES[3][1]
                ),
                "--config",
                str(
                    runtime_cfg_path
                ),
                "--query-csv",
                str(
                    blind_manifest
                ),
                "--reuse-map-caches",
                "--map-cache-root",
                str(
                    map_cache_root
                ),
                "--batch-size",
                "1",
                "--image-size",
                str(
                    protocol[
                        "image_size"
                    ]
                ),
                "--crop-mode",
                str(
                    protocol[
                        "crop_mode"
                    ]
                ),
                "--pooling",
                str(
                    protocol[
                        "pooling"
                    ]
                ),
            ],
        ),

        # ----------------------------------------------------
        # 05 — DINO Top-K retrieval
        # ----------------------------------------------------
        (
            "05_dino_topk_retrieval",
            [
                py,
                str(
                    repo_root
                    / STAGES[4][1]
                ),
                "--query-cache",
                str(
                    artifacts[
                        "dino_query_cache"
                    ]
                ),
                "--map-cache",
                str(
                    map_cache
                ),
                "--run-root",
                str(
                    run_root
                ),
                "--variant",
                variant,
                "--tag",
                tag,
                "--top-k",
                str(
                    top_k
                ),
            ],
        ),

        # ----------------------------------------------------
        # 06 — ORB Top-K geometric verification
        # ----------------------------------------------------
        (
            "06_orb_topk_verification",
            [
                py,
                str(
                    repo_root
                    / STAGES[5][1]
                ),
                "--config",
                str(
                    runtime_cfg_path
                ),
                "--repo-root",
                str(
                    repo_root
                ),
                "--variant",
                variant,
                "--tag",
                tag,
                "--query-csv",
                str(
                    blind_manifest
                ),
                "--topk-csv",
                str(
                    topk_csv
                ),
                "--tile-index-csv",
                str(
                    tile_index
                ),
                "--out-root",
                str(
                    orb_output_root
                ),
                "--top-n",
                str(
                    top_k
                ),
                "--policy",
                "hybrid",
                "--blind-only",
            ],
        ),

        # ----------------------------------------------------
        # 07 — trusted map bootstrap
        # ----------------------------------------------------
        (
            "07_blind_map_bootstrap",
            [
                py,
                str(
                    repo_root
                    / STAGES[6][1]
                ),
                "--repo-root",
                str(
                    repo_root
                ),
                "--config",
                str(
                    runtime_cfg_path
                ),
                "--run-root",
                str(
                    run_root
                ),
            ],
        ),

        # ----------------------------------------------------
        # 08 — map-aligned / relative-only continuation
        # ----------------------------------------------------
        (
            "08_map_alignment",
            [
                py,
                str(
                    repo_root
                    / STAGES[7][1]
                ),
                "--repo-root",
                str(
                    repo_root
                ),
                "--config",
                str(
                    runtime_cfg_path
                ),
                "--blind-manifest",
                str(
                    blind_manifest
                ),
                "--raw-relative",
                str(
                    raw_relative
                ),
                "--bootstrap-report",
                str(
                    bootstrap_report
                ),
                "--run-root",
                str(
                    run_root
                ),
                "--map-crs",
                map_crs,
            ],
        ),

        # ----------------------------------------------------
        # 09 — causal temporal fusion/control
        # ----------------------------------------------------
        (
            "09_temporal_fusion",
            [
                py,
                str(
                    repo_root
                    / STAGES[8][1]
                ),
                "--repo-root",
                str(
                    repo_root
                ),
                "--config",
                str(
                    runtime_cfg_path
                ),
                "--map-trajectory",
                str(
                    map_trajectory
                ),
                "--bootstrap-report",
                str(
                    bootstrap_report
                ),
                "--absolute-query-summary",
                str(
                    orb_query_summary
                ),
                "--absolute-candidate-scores",
                str(
                    orb_candidate_scores
                ),
                "--run-root",
                str(
                    run_root
                ),
            ],
        ),

        # ----------------------------------------------------
        # 10 — stable output / lat-lon export when available
        # ----------------------------------------------------
        (
            "10_estimated_output_export",
            [
                py,
                str(
                    repo_root
                    / STAGES[9][1]
                ),
                "--repo-root",
                str(
                    repo_root
                ),
                "--config",
                str(
                    runtime_cfg_path
                ),
                "--fused-trajectory",
                str(
                    temporal_fused
                ),
                "--fusion-report",
                str(
                    temporal_report
                ),
                "--absolute-query-summary",
                str(
                    orb_query_summary
                ),
                "--run-root",
                str(
                    run_root
                ),
                "--tile-index",
                str(
                    tile_index
                ),
                "--source-crs",
                map_crs,
                "--target-crs",
                target_crs,
            ],
        ),

        # ----------------------------------------------------
        # 11 — reference-free visual diagnostics
        # ----------------------------------------------------
        (
            "11_blind_safe_visuals",
            [
                py,
                str(
                    repo_root
                    / STAGES[10][1]
                ),
                "--relative-trajectory",
                str(
                    map_trajectory
                ),
                "--submission",
                str(
                    submission
                ),
                "--temporal-manifest",
                str(
                    temporal_manifest
                ),
                "--run-root",
                str(
                    run_root
                ),
                "--tile-index",
                str(
                    tile_index
                ),
                "--ortho-tif",
                str(
                    (
                        repo_root
                        / map_cfg[
                            "canonical_aoi_tif"
                        ]
                    ).resolve()
                ),
            ],
        ),

        # ----------------------------------------------------
        # 12 — initial runtime registry
        # ----------------------------------------------------
        (
            "12_runtime_registry_initial",
            [
                py,
                str(
                    repo_root
                    / STAGES[11][1]
                ),
                "--root",
                str(
                    run_root
                ),
                "--map-root",
                str(
                    map_root
                ),
                "--run-root",
                str(
                    run_root
                ),
                "--tag",
                tag,
                "--variant",
                variant,
            ],
        ),

        # ----------------------------------------------------
        # 13 — resource/cache registry
        # ----------------------------------------------------
        (
            "13_resource_registry",
            [
                py,
                str(
                    repo_root
                    / STAGES[12][1]
                ),
                "--root",
                str(
                    run_root
                ),
                "--map-root",
                str(
                    map_root
                ),
                "--run-root",
                str(
                    run_root
                ),
                "--variant",
                variant,
                "--tag",
                tag,
            ],
        ),

        # ----------------------------------------------------
        # 14 — deployment cost breakdown
        # ----------------------------------------------------
        (
            "14_deployment_breakdown",
            [
                py,
                str(
                    repo_root
                    / STAGES[13][1]
                ),
                "--run-root",
                str(
                    run_root
                ),
                "--sampled-query-rate-hz",
                str(
                    sample_rate_fps
                ),
            ],
        ),

        # ----------------------------------------------------
        # 15 — Markdown-only summary
        # ----------------------------------------------------
        (
            "15_markdown_run_summary",
            [
                py,
                str(
                    repo_root
                    / STAGES[14][1]
                ),
                "--root",
                str(
                    run_root
                ),
                "--map-root",
                str(
                    map_root
                ),
                "--run-root",
                str(
                    run_root
                ),
            ],
        ),

        # ----------------------------------------------------
        # 16 — refresh runtime registry after summary exists
        # ----------------------------------------------------
        (
            "16_runtime_registry_refresh",
            [
                py,
                str(
                    repo_root
                    / STAGES[15][1]
                ),
                "--root",
                str(
                    run_root
                ),
                "--map-root",
                str(
                    map_root
                ),
                "--run-root",
                str(
                    run_root
                ),
                "--tag",
                tag,
                "--variant",
                variant,
            ],
        ),

        # ----------------------------------------------------
        # 17 — freeze/hash blind output
        # ----------------------------------------------------
        (
            "17_freeze_blind_output",
            [
                py,
                str(
                    repo_root
                    / STAGES[16][1]
                ),
                "--submission",
                str(
                    submission
                ),
                "--addon9-report",
                str(
                    addon9_report
                ),
                "--run-root",
                str(
                    run_root
                ),
            ],
        ),
    ]

    require(
        len(commands)
        == len(STAGES),
        (
            "Internal orchestrator stage-count "
            "mismatch."
        ),
    )

    for index, (
        expected,
        _,
    ) in enumerate(
        STAGES,
        start=1,
    ):

        actual = (
            commands[
                index - 1
            ][0]
        )

        require(
            actual
            == expected,
            (
                "Internal stage-order mismatch "
                f"at {index}: "
                f"{actual!r} != {expected!r}"
            ),
        )

    # ========================================================
    # Execute fail-fast
    #
    # The entire execution section is duplicated into one
    # orchestrator-level transcript in addition to the
    # individual per-stage logs.
    # ========================================================

    metrics_dir = (
        run_root
        / "metrics"
    )

    metrics_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    whole_log_path = (
        logs_dir
        / "orchestrator_full_run.log"
    )

    execution_report_path = (
        metrics_dir
        / "orchestrator_execution_summary.json"
    )

    original_stdout = (
        sys.stdout
    )

    original_stderr = (
        sys.stderr
    )

    execution_status = (
        "RUNNING"
    )

    completed_stages: list[str] = []

    current_stage: str | None = None
    current_stage_index: int | None = None

    error_type: str | None = None
    error_message: str | None = None

    orchestrator_started = (
        time.perf_counter()
    )

    with whole_log_path.open(
        "w",
        encoding="utf-8",
        buffering=1,
    ) as whole_log:

        sys.stdout = TeeStream(
            original_stdout,
            whole_log,
        )

        sys.stderr = TeeStream(
            original_stderr,
            whole_log,
        )

        try:

            print()
            print("=" * 100)
            print(
                "STAGE 9B — EXECUTING PROMOTED "
                "BLIND RECORDED-FLIGHT PIPELINE"
            )
            print("=" * 100)

            print()
            print(
                "run root        :",
                run_root,
            )

            print(
                "resolved config :",
                runtime_cfg_path,
            )

            print(
                "python          :",
                py,
            )

            print(
                "stage count     :",
                len(
                    commands
                ),
            )

            print(
                "whole-run log   :",
                whole_log_path,
            )

            for index, (
                stage_name,
                command,
            ) in enumerate(
                commands,
                start=1,
            ):

                current_stage = (
                    stage_name
                )

                current_stage_index = (
                    index
                )

                run_stage(
                    index=index,
                    stage_name=stage_name,
                    command=command,
                    repo_root=repo_root,
                    run_root=run_root,
                    env=child_env,
                )

                completed_stages.append(
                    stage_name
                )

            execution_status = (
                "PASS"
            )

            current_stage = None
            current_stage_index = None

            print()
            print("=" * 100)
            print(
                "BLIND RECORDED-FLIGHT "
                "ORCHESTRATION COMPLETE"
            )
            print("=" * 100)

            print()
            print(
                "run root         :",
                run_root,
            )

            print(
                "blind trajectory :",
                submission,
            )

            print(
                "Markdown summary :",
                artifacts[
                    "markdown_summary"
                ],
            )

            print(
                "freeze record    :",
                artifacts[
                    "freeze_record"
                ],
            )

        except BaseException as exc:

            execution_status = (
                "FAIL"
            )

            error_type = (
                type(
                    exc
                ).__name__
            )

            error_message = (
                str(
                    exc
                )
            )

            print()
            print("!" * 100)
            print(
                "ORCHESTRATOR EXECUTION FAILED"
            )
            print("!" * 100)

            print(
                "failed stage index :",
                current_stage_index,
            )

            print(
                "failed stage       :",
                current_stage,
            )

            print(
                "error type         :",
                error_type,
            )

            print(
                "error              :",
                error_message,
            )

            raise

        finally:

            orchestrator_total_wall_s = (
                time.perf_counter()
                - orchestrator_started
            )

            execution_report = {
                "stage":
                    "BLIND_RECORDED_FLIGHT_ORCHESTRATOR",

                "status":
                    (
                        "PASS_ORCHESTRATOR_EXECUTION"
                        if execution_status == "PASS"
                        else "FAIL_ORCHESTRATOR_EXECUTION"
                    ),

                "execution_status":
                    execution_status,

                "run_root":
                    str(
                        run_root
                    ),

                "resolved_config":
                    str(
                        runtime_cfg_path
                    ),

                "python":
                    str(
                        py
                    ),

                "planned_stage_count":
                    int(
                        len(
                            commands
                        )
                    ),

                "completed_stage_count":
                    int(
                        len(
                            completed_stages
                        )
                    ),

                "completed_stages":
                    completed_stages,

                "failed_stage_index":
                    current_stage_index,

                "failed_stage":
                    current_stage,

                "error_type":
                    error_type,

                "error_message":
                    error_message,

                "orchestrator_total_wall_s":
                    float(
                        orchestrator_total_wall_s
                    ),

                "orchestrator_total_wall_min":
                    float(
                        orchestrator_total_wall_s
                        / 60.0
                    ),

                "whole_terminal_log":
                    str(
                        whole_log_path
                    ),

                "per_stage_log_directory":
                    str(
                        logs_dir
                    ),
            }

            execution_report_path.write_text(
                json.dumps(
                    execution_report,
                    indent=2,
                    allow_nan=False,
                ),
                encoding="utf-8",
            )

            print()
            print("=" * 100)
            print(
                "ORCHESTRATOR EXECUTION SUMMARY"
            )
            print("=" * 100)

            print()
            print(
                "execution status  :",
                execution_status,
            )

            print(
                "stages completed  :",
                (
                    f"{len(completed_stages)}"
                    f"/{len(commands)}"
                ),
            )

            print(
                "total wall time   :",
                (
                    f"{orchestrator_total_wall_s:.3f} s "
                    f"({orchestrator_total_wall_s / 60.0:.3f} min)"
                ),
            )

            print(
                "whole-run log     :",
                whole_log_path,
            )

            print(
                "execution report  :",
                execution_report_path,
            )

            if (
                execution_status
                == "PASS"
            ):

                print()
                print(
                    "STATUS: "
                    "PASS_DEMO_STAGE9B_ORCHESTRATED_BLIND_RUN"
                )

            sys.stdout.flush()
            sys.stderr.flush()

            sys.stdout = (
                original_stdout
            )

            sys.stderr = (
                original_stderr
            )


if __name__ == "__main__":
    main()
