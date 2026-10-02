#!/usr/bin/env python3

from __future__ import annotations

import shutil
import time
import argparse
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path("/workspace")
LOCAL_VIDEO_STAGE_ROOT = Path("/tmp/drone_localisation")


CONFIG = (
    ROOT
    / "configs"
    / "demo_villoc_blind_recorded.yaml"
)

VIDEO = (
    ROOT
    / "inputs"
    / "flight.mp4"
)

OUTPUT_PARENT = (
    ROOT
    / "outputs"
    / "demo_runs"
)

AOI = (
    ROOT
    / "data"
    / "processed"
    / "villoc"
    / "90_deg"
    / "maps"
    / "ort10lt_2024_2026"
    / "ort10lt_2024_2026_aoi300m.tif"
)

TILES = (
    ROOT
    / "data"
    / "processed"
    / "villoc"
    / "90_deg"
    / "maps"
    / "ort10lt_2024_2026"
    / "tiles_512_s256"
)

TILE_INDEX = (
    ROOT
    / "outputs"
    / "villoc"
    / "90_deg"
    / "metadata"
    / "s8_9_satellite_tile_index_512_s256.csv"
)

MAP_DESC = (
    ROOT
    / "outputs"
    / "villoc"
    / "90_deg"
    / "descriptors"
    / (
        "s8_11b_dinov2_map_512_s256_"
        "dinov2_vits14_img518_"
        "center_square_avgpatch_cpu.npz"
    )
)

XFEAT = (
    ROOT
    / "third_party"
    / "accelerated_features"
)

DINO_REPO = (
    Path("/root/.cache/torch/hub")
    / "facebookresearch_dinov2_main"
)

DINO_WEIGHT = (
    Path("/root/.cache/torch/hub/checkpoints")
    / "dinov2_vits14_pretrain.pth"
)

ORCHESTRATOR = (
    ROOT
    / "scripts"
    / "demo"
    / "run_recorded_flight_demo.py"
)


def require(
    condition: bool,
    message: str,
) -> None:

    if not condition:
        raise RuntimeError(message)


def run_command(
    command: list[str],
) -> None:

    print()
    print("$", " ".join(command))
    print()

    subprocess.run(
        command,
        cwd=ROOT,
        check=True,
    )


def validate_run_id(
    run_id: str,
) -> None:

    require(
        bool(
            re.fullmatch(
                r"[A-Za-z0-9][A-Za-z0-9._-]*",
                run_id,
            )
        ),
        (
            "Invalid run ID. Use letters, numbers, "
            "dot, underscore or hyphen only."
        ),
    )

def stage_video_locally(
    run_id: str,
) -> tuple[Path, Path]:

    require(
        VIDEO.exists(),
        f"Mounted blind video missing: {VIDEO}",
    )

    stage_dir = (
        LOCAL_VIDEO_STAGE_ROOT
        / run_id
    )

    if stage_dir.exists():
        shutil.rmtree(
            stage_dir
        )

    stage_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    staged_video = (
        stage_dir
        / "flight.mp4"
    )

    source_size = (
        VIDEO.stat().st_size
    )

    print()
    print(
        "Staging blind video into "
        "container-local storage..."
    )
    print(
        f"source : {VIDEO}"
    )
    print(
        f"target : {staged_video}"
    )

    started = (
        time.perf_counter()
    )

    shutil.copyfile(
        VIDEO,
        staged_video,
    )

    elapsed = (
        time.perf_counter()
        - started
    )

    staged_size = (
        staged_video.stat().st_size
    )

    require(
        staged_size == source_size,
        (
            "Local video staging size mismatch: "
            f"source={source_size}, "
            f"staged={staged_size}"
        ),
    )

    print(
        f"bytes  : {staged_size}"
    )
    print(
        f"time_s : {elapsed:.3f}"
    )
    print(
        "LOCAL_VIDEO_STAGING: PASS"
    )

    return (
        staged_video,
        stage_dir,
    )


def check() -> None:

    print("=" * 78)
    print("DRONE LOCALISATION — DOCKER PREFLIGHT")
    print("=" * 78)

    required_paths = [
        ("RGB video", VIDEO),
        ("blind config", CONFIG),
        ("AOI orthophoto", AOI),
        ("tile directory", TILES),
        ("tile index", TILE_INDEX),
        ("map descriptors", MAP_DESC),
        ("XFeat repository", XFEAT),
        ("DINO repository", DINO_REPO),
        ("DINO checkpoint", DINO_WEIGHT),
        ("orchestrator", ORCHESTRATOR),
    ]

    for label, path in required_paths:

        require(
            path.exists(),
            f"{label} missing: {path}",
        )

        print(
            f"[PASS] {label}: {path}"
        )

    # --------------------------------------------------------
    # Fixed promoted map-grid contract
    # --------------------------------------------------------

    tile_count = sum(
        1
        for p in TILES.iterdir()
        if p.is_file()
    )

    require(
        tile_count == 475,
        (
            "Unexpected promoted map tile count: "
            f"{tile_count}; expected 475"
        ),
    )

    print(
        f"[PASS] promoted map tiles: {tile_count}"
    )

    # --------------------------------------------------------
    # Blind-input isolation
    # --------------------------------------------------------

    input_root = VIDEO.parent

    forbidden_suffixes = {
        ".srt",
        ".csv",
        ".json",
        ".gpx",
        ".kml",
    }

    forbidden = [
        p
        for p in input_root.iterdir()
        if (
            p.is_file()
            and p.suffix.lower()
            in forbidden_suffixes
        )
    ]

    require(
        not forbidden,
        (
            "Reference-like files visible in blind "
            f"input directory: {forbidden}"
        ),
    )

    print(
        "[PASS] blind input contains no "
        "SRT/GPS/reference-like sidecar"
    )

    # --------------------------------------------------------
    # Runtime imports
    # --------------------------------------------------------

    import cv2
    import numpy
    import pandas
    import psutil
    import pyproj
    import rasterio
    import torch

    print(
        f"[PASS] Python: {sys.version.split()[0]}"
    )
    print(
        f"[PASS] PyTorch: {torch.__version__}"
    )
    print(
        f"[PASS] OpenCV: {cv2.__version__}"
    )
    print(
        f"[PASS] Rasterio: {rasterio.__version__}"
    )
    print(
        f"[PASS] PyProj: {pyproj.__version__}"
    )
    print(
        f"[PASS] psutil: {psutil.__version__}"
    )

    require(
        not torch.cuda.is_available(),
        (
            "Current promoted container is expected "
            "to use the frozen CPU execution path."
        ),
    )

    print(
        "[PASS] CPU execution contract"
    )

    # --------------------------------------------------------
    # XFeat construction
    # --------------------------------------------------------

    from modules.xfeat import XFeat

    _ = XFeat()

    print(
        "[PASS] XFeat model construction"
    )

    # --------------------------------------------------------
    # DINO offline model construction
    # --------------------------------------------------------

    model = torch.hub.load(
        str(DINO_REPO),
        "dinov2_vits14",
        source="local",
        pretrained=True,
    )

    model.eval().to("cpu")

    print(
        "[PASS] DINOv2 local/offline model load"
    )

    # --------------------------------------------------------
    # Existing orchestrator contract
    # --------------------------------------------------------

    run_command(
        [
            sys.executable,
            str(ORCHESTRATOR),
            "--config",
            str(CONFIG),
            "--video",
            str(VIDEO),
            "--run-id",
            "__docker_preflight__",
            "--dry-run",
        ]
    )

    print()
    print("=" * 78)
    print("DOCKER_PREFLIGHT: PASS")
    print("=" * 78)


def blind(
    run_id: str,
) -> None:

    validate_run_id(
        run_id
    )

    run_root = (
        OUTPUT_PARENT
        / run_id
    )

    require(
        not run_root.exists(),
        (
            "Run already exists and will not be "
            f"overwritten: {run_root}"
        ),
    )

    staged_video = None
    stage_dir = None

    try:

        (
            staged_video,
            stage_dir,
        ) = stage_video_locally(
            run_id
        )

        print()
        print("=" * 78)
        print("BLIND LOCALIZATION")
        print("=" * 78)
        print()

        print(
            f"run ID          : {run_id}"
        )

        print(
            f"mounted input   : {VIDEO}"
        )

        print(
            f"staged video    : {staged_video}"
        )

        print(
            "reference input : NOT AVAILABLE"
        )

        print(
            "network         : DISABLED BY CONTAINER"
        )

        print(
            "evaluation      : NOT RUN"
        )

        print()

        run_command(
            [
                sys.executable,
                str(
                    ORCHESTRATOR
                ),
                "--config",
                str(
                    CONFIG
                ),
                "--video",
                str(
                    staged_video
                ),
                "--run-id",
                run_id,
            ]
        )

        submission = (
            run_root
            / "trajectories"
            / "submission_estimated_trajectory.csv"
        )

        freeze_record = (
            run_root
            / "evaluation"
            / "blind_submission_freeze.json"
        )

        require(
            submission.exists(),
            (
                "Blind run finished without expected "
                f"submission: {submission}"
            ),
        )

        require(
            freeze_record.exists(),
            (
                "Blind run finished without expected "
                f"freeze record: {freeze_record}"
            ),
        )

        print()
        print("=" * 78)
        print(
            "BLIND LOCALIZATION COMPLETE"
        )
        print("=" * 78)
        print()

        print(
            "Reference information used : NO"
        )

        print(
            "Evaluation executed         : NO"
        )

        print(
            f"Frozen trajectory          : {submission}"
        )

        print(
            f"Freeze record              : {freeze_record}"
        )

        print()
        print(
            "The blind result is now frozen. "
            "Any later reference-based evaluation "
            "must operate on this frozen result."
        )

        print()
        print(
            "DOCKER_BLIND_RUN: PASS"
        )

    finally:

        if (
            stage_dir is not None
            and stage_dir.exists()
        ):

            print()
            print(
                "Removing container-local "
                "video staging..."
            )

            shutil.rmtree(
                stage_dir,
                ignore_errors=True,
            )

            print(
                "LOCAL_VIDEO_CLEANUP: PASS"
            )


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Container command interface for the "
            "GNSS-denied UAV localisation demo."
        )
    )

    sub = parser.add_subparsers(
        dest="command",
        required=True,
    )

    sub.add_parser(
        "check",
        help=(
            "Verify the offline blind runtime "
            "environment without localising."
        ),
    )

    blind_parser = sub.add_parser(
        "blind",
        help=(
            "Run and freeze blind localisation."
        ),
    )

    blind_parser.add_argument(
        "--run-id",
        required=True,
    )

    return parser.parse_args()


def main() -> None:

    args = parse_args()

    if args.command == "check":
        check()

    elif args.command == "blind":
        blind(
            args.run_id
        )

    else:
        raise RuntimeError(
            f"Unknown command: {args.command}"
        )


if __name__ == "__main__":
    main()
