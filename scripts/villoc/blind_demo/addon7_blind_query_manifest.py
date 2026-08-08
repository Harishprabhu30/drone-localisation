'''
ROOT=outputs/villoc/traj01_90deg_stable120m
RUN="$ROOT/blind_demo_addons/eval_traj01_known_gt"

VIDEO=data/raw/villoc/traj01_90deg_stable120m/villoc_traj01_90deg_stable120m_V_merged.MP4

python scripts/villoc/blind_demo/addon7_blind_query_manifest.py \
  --video "$VIDEO" \
  --run-root "$RUN" \
  --sample-rate-fps 1 \
  --assumed-rel-alt-m 120 \
  --assumed-gimbal-pitch-deg -90 \
  --view-assumption near_nadir \
  --map-aoi \
  data/processed/villoc/90_deg/maps/ort10lt_2024_2026/ort10lt_2024_2026_aoi300m.tif \
  --map-tile-index \
  outputs/villoc/90_deg/metadata/s8_9_satellite_tile_index_512_s256.csv \
  --overwrite \
  2>&1 | tee \
  "$RUN/logs/addon7_blind_query_manifest.log"
'''

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import cv2
import pandas as pd


REQUIRED_COLUMNS = [
    "frame_index",
    "timestamp_s",
    "image_path",
    "assumed_rel_alt_m",
    "assumed_gimbal_pitch_deg",
    "reference_available",
]


FORBIDDEN_REFERENCE_COLUMNS = {
    "latitude",
    "longitude",
    "lat",
    "lon",
    "easting",
    "northing",
    "x_enu_m",
    "y_enu_m",
    "reference_x_m",
    "reference_y_m",
    "reference_cumulative_distance_m",
    "gps_lat",
    "gps_lon",
    "srt_lat",
    "srt_lon",
    "oracle",
    "hit_le_40m",
    "error_m",
}


def portable_path(path: Path) -> str:
    path = path.resolve()
    repo = Path.cwd().resolve()

    try:
        return str(path.relative_to(repo))
    except ValueError:
        return str(path)


def require_positive(
    value: float,
    name: str,
) -> None:
    if (
        not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(
            f"{name} must be positive."
        )


def image_sanity(
    frame_paths: list[Path],
) -> list[dict[str, Any]]:
    if not frame_paths:
        return []

    indices = sorted(
        {
            0,
            len(frame_paths) // 2,
            len(frame_paths) - 1,
        }
    )

    rows = []

    for i in indices:
        path = frame_paths[i]

        image = cv2.imread(
            str(path),
            cv2.IMREAD_COLOR,
        )

        if image is None:
            rows.append(
                {
                    "sample_index": i,
                    "image_path": (
                        portable_path(path)
                    ),
                    "decode_ok": False,
                }
            )
            continue

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY,
        )

        lap = cv2.Laplacian(
            gray,
            cv2.CV_64F,
        )

        rows.append(
            {
                "sample_index": i,
                "image_path": (
                    portable_path(path)
                ),
                "decode_ok": True,
                "width": int(
                    image.shape[1]
                ),
                "height": int(
                    image.shape[0]
                ),
                "mean_brightness": float(
                    gray.mean()
                ),
                "laplacian_variance": float(
                    lap.var()
                ),
            }
        )

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Add-on 7: Build a reference-free "
            "Villoc query manifest directly from video."
        )
    )

    parser.add_argument(
        "--video",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--run-root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--sample-rate-fps",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--assumed-rel-alt-m",
        type=float,
        required=True,
    )

    parser.add_argument(
        "--assumed-gimbal-pitch-deg",
        type=float,
        required=True,
    )

    parser.add_argument(
        "--view-assumption",
        default="near_nadir",
    )

    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=95,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    parser.add_argument(
        "--map-aoi",
        type=Path,
        default=None,
        help=(
            "Optional existing georeferenced AOI "
            "asset. Used only for path sanity."
        ),
    )

    parser.add_argument(
        "--map-tile-index",
        type=Path,
        default=None,
        help=(
            "Optional existing map tile index. "
            "Used only for path/row-count sanity."
        ),
    )

    args = parser.parse_args()

    stage_started = time.perf_counter()

    require_positive(
        args.sample_rate_fps,
        "sample-rate-fps",
    )

    require_positive(
        args.assumed_rel_alt_m,
        "assumed-rel-alt-m",
    )

    if not (
        1 <= args.jpeg_quality <= 100
    ):
        raise ValueError(
            "jpeg-quality must be 1..100."
        )

    video = args.video.resolve()
    run_root = args.run_root.resolve()

    if not video.exists():
        raise FileNotFoundError(
            video
        )

    frame_dir = (
        run_root
        / "frames"
        / (
            "blind_query_"
            f"{args.sample_rate_fps:g}fps"
        )
    )

    metadata_dir = (
        run_root / "metadata"
    )

    reports_dir = (
        run_root / "reports"
    )

    logs_dir = (
        run_root / "logs"
    )

    for directory in [
        frame_dir,
        metadata_dir,
        reports_dir,
        logs_dir,
    ]:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    manifest_path = (
        metadata_dir
        / "blind_query_manifest.csv"
    )

    report_path = (
        reports_dir
        / "blind_query_manifest_report.json"
    )

    # -----------------------------------------------------
    # Video metadata — NO SRT/GPS/reference involved.
    # -----------------------------------------------------

    metadata_started = time.perf_counter()

    cap = cv2.VideoCapture(
        str(video)
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {video}"
        )

    video_fps = float(
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

    require_positive(
        video_fps,
        "reported video FPS",
    )

    if frame_count <= 0:
        raise RuntimeError(
            "Video reports zero frames."
        )

    duration_s = (
        frame_count / video_fps
    )

    last_frame_time_s = (
        (frame_count - 1)
        / video_fps
    )

    metadata_finished = (
        time.perf_counter()
    )

    # -----------------------------------------------------
    # Blind sampling schedule
    #
    # Use only complete sample intervals. For a ~403.4 s
    # video at 1 fps this yields t=0..402 rather than
    # creating a final sample from an incomplete tail.
    # -----------------------------------------------------

    step_s = (
        1.0
        / args.sample_rate_fps
    )

    sample_times: list[float] = []

    sample_id = 0

    while True:
        target = (
            sample_id
            * step_s
        )

        if (
            target + step_s
            > duration_s + 1e-9
        ):
            break

        sample_times.append(
            float(target)
        )

        sample_id += 1

    if not sample_times:
        sample_times = [0.0]

    # -----------------------------------------------------
    # Frame extraction
    # -----------------------------------------------------

    extraction_started = (
        time.perf_counter()
    )

    rows: list[dict[str, Any]] = []
    extracted_paths: list[Path] = []

    extracted_ok = 0
    skipped_existing = 0
    failed_reads = 0

    for sequence_index, target_time_s in enumerate(
        sample_times
    ):
        source_frame_index = int(
            round(
                target_time_s
                * video_fps
            )
        )

        source_frame_index = min(
            source_frame_index,
            frame_count - 1,
        )

        source_frame_time_s = (
            source_frame_index
            / video_fps
        )

        alignment_error_ms = abs(
            source_frame_time_s
            - target_time_s
        ) * 1000.0

        out_name = (
            "blind_frame_"
            f"{sequence_index:05d}"
            "_srcframe_"
            f"{source_frame_index:06d}.jpg"
        )

        out_path = (
            frame_dir
            / out_name
        )

        extraction_status = (
            "skipped_exists"
        )

        if (
            args.overwrite
            or not out_path.exists()
        ):
            cap.set(
                cv2.CAP_PROP_POS_FRAMES,
                source_frame_index,
            )

            ok, frame = cap.read()

            if (
                not ok
                or frame is None
            ):
                extraction_status = (
                    "failed_read"
                )

                failed_reads += 1

            else:
                write_ok = cv2.imwrite(
                    str(out_path),
                    frame,
                    [
                        int(
                            cv2.IMWRITE_JPEG_QUALITY
                        ),
                        int(
                            args.jpeg_quality
                        ),
                    ],
                )

                if not write_ok:
                    extraction_status = (
                        "failed_write"
                    )

                    failed_reads += 1

                else:
                    extraction_status = "ok"
                    extracted_ok += 1

        else:
            skipped_existing += 1

        if extraction_status in {
            "ok",
            "skipped_exists",
        }:
            extracted_paths.append(
                out_path
            )

        # query_id/token0_id intentionally use a
        # stable 1-based identifier compatible with
        # the current Villoc query-cache convention.
        query_id = (
            sequence_index + 1
        )

        rows.append(
            {
                # README-required blind contract
                "frame_index": int(
                    sequence_index
                ),
                "timestamp_s": float(
                    target_time_s
                ),
                "image_path": (
                    portable_path(
                        out_path
                    )
                ),
                "assumed_rel_alt_m": float(
                    args.assumed_rel_alt_m
                ),
                "assumed_gimbal_pitch_deg": float(
                    args.assumed_gimbal_pitch_deg
                ),
                "reference_available": False,

                # Blind-safe downstream compatibility
                "sequence_frame_id": int(
                    sequence_index
                ),
                "query_id": int(
                    query_id
                ),
                "token0_id": int(
                    query_id
                ),

                # Video-only provenance
                "source_video_frame_index": int(
                    source_frame_index
                ),
                "source_video_frame_time_s": float(
                    source_frame_time_s
                ),
                "sampling_alignment_error_ms": float(
                    alignment_error_ms
                ),
                "source_video": (
                    portable_path(video)
                ),
                "view_assumption": (
                    args.view_assumption
                ),
                "extraction_status": (
                    extraction_status
                ),
                "image_width": int(
                    width
                ),
                "image_height": int(
                    height
                ),
            }
        )

    cap.release()

    extraction_finished = (
        time.perf_counter()
    )

    # -----------------------------------------------------
    # Manifest construction / validation
    # -----------------------------------------------------

    manifest_started = (
        time.perf_counter()
    )

    manifest = pd.DataFrame(
        rows
    )

    missing_required = sorted(
        set(REQUIRED_COLUMNS)
        - set(manifest.columns)
    )

    if missing_required:
        raise RuntimeError(
            "Missing required blind columns: "
            f"{missing_required}"
        )

    if manifest.empty:
        raise RuntimeError(
            "Blind manifest is empty."
        )

    if manifest[
        "reference_available"
    ].astype(bool).any():
        raise RuntimeError(
            "reference_available must be false "
            "for every blind manifest row."
        )

    present_forbidden = sorted(
        FORBIDDEN_REFERENCE_COLUMNS
        & set(
            manifest.columns
        )
    )

    if present_forbidden:
        raise RuntimeError(
            "Reference/evaluation columns leaked "
            "into blind manifest: "
            f"{present_forbidden}"
        )

    if manifest[
        "query_id"
    ].duplicated().any():
        raise RuntimeError(
            "query_id must be unique."
        )

    if manifest[
        "token0_id"
    ].duplicated().any():
        raise RuntimeError(
            "token0_id must be unique."
        )

    usable = manifest[
        "extraction_status"
    ].isin(
        {
            "ok",
            "skipped_exists",
        }
    )

    
    if not usable.all():
        failed_indices = manifest.index[
            ~usable
        ].tolist()

        failed_count = len(
            failed_indices
        )

        terminal_index = (
            len(manifest) - 1
        )

        terminal_only_failure = (
            failed_count == 1
            and failed_indices[0]
            == terminal_index
        )

        if terminal_only_failure:
            failed_row = manifest.loc[
                failed_indices[0]
            ]

            print()
            print(
                "[WARN] Final blind query could not "
                "be decoded."
            )
            print(
                "       Dropping terminal sample only:"
            )
            print(
                f"       timestamp_s="
                f"{failed_row['timestamp_s']:.3f}"
            )
            print(
                f"       source_frame="
                f"{failed_row['source_video_frame_index']}"
            )

            manifest = (
                manifest.loc[
                    usable
                ]
                .reset_index(
                    drop=True
                )
            )

        else:
            raise RuntimeError(
                f"{failed_count} frame extractions "
                "failed, including a non-terminal "
                "sample. Refusing to continue."
            )

    manifest.to_csv(
        manifest_path,
        index=False,
    )

    manifest_finished = (
        time.perf_counter()
    )

    # -----------------------------------------------------
    # Lightweight image/map sanity
    # -----------------------------------------------------

    sanity_started = (
        time.perf_counter()
    )

    sanity_images = image_sanity(
        extracted_paths
    )

    sanity_decode_pass = all(
        row.get(
            "decode_ok",
            False,
        )
        for row in sanity_images
    )

    map_aoi_status = None

    if args.map_aoi is not None:
        map_aoi_status = {
            "path": portable_path(
                args.map_aoi
            ),
            "exists": (
                args.map_aoi.exists()
            ),
        }

    map_tile_status = None

    if (
        args.map_tile_index
        is not None
    ):
        tile_count = None

        if args.map_tile_index.exists():
            try:
                tile_count = int(
                    len(
                        pd.read_csv(
                            args.map_tile_index
                        )
                    )
                )
            except Exception:
                tile_count = None

        map_tile_status = {
            "path": portable_path(
                args.map_tile_index
            ),
            "exists": (
                args.map_tile_index.exists()
            ),
            "rows": tile_count,
        }

    sanity_finished = (
        time.perf_counter()
    )

    stage_finished = (
        time.perf_counter()
    )

    # -----------------------------------------------------
    # Report
    # -----------------------------------------------------

    report = {
        "stage": (
            "ADDON7_BLIND_QUERY_MANIFEST"
        ),
        "status": (
            "PASS_BLIND_QUERY_MANIFEST"
            if sanity_decode_pass
            else "PASS_WITH_IMAGE_SANITY_WARNING"
        ),
        "blind_contract": {
            "reference_used": False,
            "srt_used": False,
            "gps_used": False,
            "ground_truth_used": False,
            "reference_available": False,
            "required_columns": (
                REQUIRED_COLUMNS
            ),
            "forbidden_reference_columns_present": (
                present_forbidden
            ),
        },
        "input": {
            "video": portable_path(
                video
            ),
            "sample_rate_fps": float(
                args.sample_rate_fps
            ),
            "assumed_rel_alt_m": float(
                args.assumed_rel_alt_m
            ),
            "assumed_gimbal_pitch_deg": float(
                args.assumed_gimbal_pitch_deg
            ),
            "view_assumption": (
                args.view_assumption
            ),
        },
        "video_metadata": {
            "fps": video_fps,
            "frame_count": frame_count,
            "width": width,
            "height": height,
            "duration_s": duration_s,
            "last_frame_time_s": (
                last_frame_time_s
            ),
        },
        "sampling": {
            "policy": (
                "uniform_video_time_complete_intervals"
            ),
            "requested_sample_rate_fps": float(
                args.sample_rate_fps
            ),
            "step_s": step_s,
            "sample_count": int(
                len(manifest)
            ),
            "first_timestamp_s": float(
                manifest[
                    "timestamp_s"
                ].iloc[0]
            ),
            "last_timestamp_s": float(
                manifest[
                    "timestamp_s"
                ].iloc[-1]
            ),
            "max_alignment_error_ms": float(
                manifest[
                    "sampling_alignment_error_ms"
                ].max()
            ),
        },
        "extraction": {
            "extracted_ok": int(
                extracted_ok
            ),
            "skipped_existing": int(
                skipped_existing
            ),
            "failed": int(
                failed_reads
            ),
            "frame_directory": (
                portable_path(
                    frame_dir
                )
            ),
        },
        "sanity": {
            "sampled_images": (
                sanity_images
            ),
            "decode_pass": bool(
                sanity_decode_pass
            ),
            "map_aoi": (
                map_aoi_status
            ),
            "map_tile_index": (
                map_tile_status
            ),
        },
        "runtime": {
            "video_metadata_read_s": float(
                metadata_finished
                - metadata_started
            ),
            "frame_extraction_s": float(
                extraction_finished
                - extraction_started
            ),
            "query_manifest_build_s": float(
                manifest_finished
                - manifest_started
            ),
            "sanity_check_s": float(
                sanity_finished
                - sanity_started
            ),
            "total_stage_wall_s": float(
                stage_finished
                - stage_started
            ),
        },
        "outputs": {
            "manifest": portable_path(
                manifest_path
            ),
            "report": portable_path(
                report_path
            ),
        },
    }

    report_path.write_text(
        json.dumps(
            report,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    print("=" * 80)
    print(
        "STAGE 8A — ADD-ON 7 "
        "BLIND QUERY MANIFEST"
    )
    print("=" * 80)

    print(
        "status:",
        report["status"],
    )

    print()
    print("Blind contract")
    print("-" * 80)

    print(
        "SRT used                 : false"
    )
    print(
        "GPS used                 : false"
    )
    print(
        "ground truth used        : false"
    )
    print(
        "reference_available      : false"
    )
    print(
        "forbidden columns present:",
        present_forbidden,
    )

    print()
    print("Video")
    print("-" * 80)

    print(
        f"fps:                     "
        f"{video_fps:.6f}"
    )
    print(
        f"frames:                  "
        f"{frame_count}"
    )
    print(
        f"resolution:              "
        f"{width} x {height}"
    )
    print(
        f"duration:                "
        f"{duration_s:.3f} s"
    )

    print()
    print("Sampling")
    print("-" * 80)

    print(
        f"requested rate:          "
        f"{args.sample_rate_fps:g} fps"
    )
    print(
        f"manifest rows:           "
        f"{len(manifest)}"
    )
    print(
        f"first timestamp:         "
        f"{manifest['timestamp_s'].iloc[0]:.3f} s"
    )
    print(
        f"last timestamp:          "
        f"{manifest['timestamp_s'].iloc[-1]:.3f} s"
    )
    print(
        f"alignment max:           "
        f"{manifest['sampling_alignment_error_ms'].max():.3f} ms"
    )

    print()
    print("Extraction")
    print("-" * 80)

    print(
        f"extracted ok:            "
        f"{extracted_ok}"
    )
    print(
        f"skipped existing:        "
        f"{skipped_existing}"
    )
    print(
        f"failed:                  "
        f"{failed_reads}"
    )

    print()
    print("Runtime")
    print("-" * 80)

    for key, value in (
        report["runtime"].items()
    ):
        print(
            f"{key:28s}: "
            f"{value:.6f} s"
        )

    print()
    print("Saved")
    print("-" * 80)

    print(manifest_path)
    print(report_path)


if __name__ == "__main__":
    main()
