'''
python scripts/villoc/blind_demo/validate_addon8_optional_evaluation_attachment.py \
  --blind-manifest \
  "$RUN/metadata/blind_query_manifest.csv" \
  --attachment \
  "$RUN/evaluation/reference_attachment.csv" \
  --summary \
  "$RUN/evaluation/evaluation_summary.json" \
  --expected-rows 403 \
  --max-time-delta-s 0.05 \
  2>&1 | tee \
  "$RUN/logs/validate_addon8_attachment_from_raw_srt.log"
'''

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from uavloc.data.villoc_srt import parse_villoc_srt


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Parse an optional Villoc SRT only on the "
            "post-run evaluation side."
        )
    )

    parser.add_argument(
        "--reference-srt",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--run-root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--video-id",
        default="V",
    )

    parser.add_argument(
        "--modality",
        default="rgb",
    )

    parser.add_argument(
        "--role",
        default="evaluation_reference",
    )

    args = parser.parse_args()

    stage_started = time.perf_counter()

    srt_path = (
        args.reference_srt
        .expanduser()
        .resolve()
    )

    run_root = (
        args.run_root
        .expanduser()
        .resolve()
    )

    if not srt_path.exists():
        raise FileNotFoundError(
            srt_path
        )

    evaluation_dir = (
        run_root / "evaluation"
    )

    evaluation_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_csv = (
        evaluation_dir
        / "reference_srt_parsed.csv"
    )

    report_path = (
        evaluation_dir
        / "reference_srt_parse_report.json"
    )

    input_sha_before = sha256_file(
        srt_path
    )

    # --------------------------------------------------
    # Reuse the exact existing Villoc parser.
    # --------------------------------------------------

    parse_started = (
        time.perf_counter()
    )

    df = parse_villoc_srt(
        srt_path,
        video_id=args.video_id,
        modality=args.modality,
        role=args.role,
    )

    parse_finished = (
        time.perf_counter()
    )

    if df.empty:
        raise RuntimeError(
            "Reference SRT parser returned zero rows."
        )

    required_columns = {
        "srt_index",
        "frame_cnt",
        "video_time_s",
        "lat",
        "lon",
        "rel_alt_m",
        "abs_alt_m",
        "gb_yaw_deg",
        "gb_pitch_deg",
        "source_srt",
    }

    missing = sorted(
        required_columns
        - set(df.columns)
    )

    if missing:
        raise RuntimeError(
            "Parsed SRT missing expected columns: "
            f"{missing}"
        )

    write_started = (
        time.perf_counter()
    )

    df.to_csv(
        output_csv,
        index=False,
    )

    write_finished = (
        time.perf_counter()
    )

    input_sha_after = sha256_file(
        srt_path
    )

    if (
        input_sha_before
        != input_sha_after
    ):
        raise RuntimeError(
            "Raw SRT changed during parsing."
        )

    stage_finished = (
        time.perf_counter()
    )

    valid_latlon = (
        df[["lat", "lon"]]
        .notna()
        .all(axis=1)
    )

    report = {
        "stage": (
            "ADDON8_OPTIONAL_REFERENCE_SRT_PARSE"
        ),
        "status": (
            "PASS_OPTIONAL_REFERENCE_SRT_PARSE"
        ),
        "evaluation_only": True,
        "reference_available_to_localization": False,
        "parser_reused": (
            "uavloc.data.villoc_srt.parse_villoc_srt"
        ),
        "input": {
            "reference_srt": str(
                srt_path
            ),
            "sha256": (
                input_sha_before
            ),
        },
        "output": {
            "parsed_reference_csv": str(
                output_csv
            ),
            "report": str(
                report_path
            ),
        },
        "rows": int(
            len(df)
        ),
        "valid_latlon_rows": int(
            valid_latlon.sum()
        ),
        "video_time_min_s": float(
            df["video_time_s"].min()
        ),
        "video_time_max_s": float(
            df["video_time_s"].max()
        ),
        "frame_cnt_min": int(
            df["frame_cnt"].min()
        ),
        "frame_cnt_max": int(
            df["frame_cnt"].max()
        ),
        "runtime": {
            "srt_reference_parsing_s": float(
                parse_finished
                - parse_started
            ),
            "parsed_csv_write_s": float(
                write_finished
                - write_started
            ),
            "total_stage_wall_s": float(
                stage_finished
                - stage_started
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
        "STAGE 9C.1 — OPTIONAL "
        "REFERENCE SRT PARSE"
    )

    print("=" * 80)

    print(
        "status: "
        "PASS_OPTIONAL_REFERENCE_SRT_PARSE"
    )

    print()
    print("Boundary")
    print("-" * 80)

    print(
        "evaluation_only                 : true"
    )

    print(
        "reference available localization: false"
    )

    print(
        "raw SRT modified                : false"
    )

    print(
        "parser                          : "
        "uavloc.data.villoc_srt.parse_villoc_srt"
    )

    print()
    print("Parsed reference")
    print("-" * 80)

    print(
        f"rows                            : "
        f"{len(df)}"
    )

    print(
        f"valid lat/lon                   : "
        f"{int(valid_latlon.sum())}"
    )

    print(
        "video time                      : "
        f"{df['video_time_s'].min():.3f} .. "
        f"{df['video_time_s'].max():.3f} s"
    )

    print(
        "frame count                     : "
        f"{int(df['frame_cnt'].min())} .. "
        f"{int(df['frame_cnt'].max())}"
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

    print(output_csv)
    print(report_path)


if __name__ == "__main__":
    main()
