'''
Command Executed:

export PYTHONPATH=$PWD/src
python scripts/villoc/s8_2_extract_sampled_frames.py \
  --config configs/dataset_villoc_90deg.yaml \
  --stream V \
  --sample-rate-fps 1 \
  --overwrite

'''

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import pandas as pd
import yaml


def nearest_row(df: pd.DataFrame, target_time_s: float) -> pd.Series:
    idx = (df["video_time_s"] - target_time_s).abs().idxmin()
    return df.loc[idx]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--stream", default="V")
    parser.add_argument("--sample-rate-fps", type=float, default=1.0)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())

    raw_root = Path(cfg["dataset"]["raw_root"])
    processed_root = Path(cfg["dataset"]["processed_root"])
    output_root = Path(cfg["dataset"]["output_root"])

    stream_id = args.stream
    stream_cfg = cfg["streams"][stream_id]

    video_path = raw_root / stream_cfg["video"]
    parsed_srt_path = output_root / "metadata" / f"s8_1_srt_parsed_{stream_id}.csv"

    if not video_path.exists():
        raise FileNotFoundError(video_path)
    if not parsed_srt_path.exists():
        raise FileNotFoundError(
            f"Parsed SRT not found: {parsed_srt_path}. Run S8.1 first."
        )

    srt_df = pd.read_csv(parsed_srt_path)

    frame_dir = processed_root / f"frames_{stream_id.lower()}_{args.sample_rate_fps:g}fps"
    metadata_dir = output_root / "metadata"
    reports_dir = output_root / "reports"

    frame_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    video_fps = float(cap.get(cv2.CAP_PROP_FPS))
    video_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_duration_s = video_frame_count / video_fps if video_fps > 0 else None

    max_srt_time = float(srt_df["video_time_s"].max())
    step_s = 1.0 / args.sample_rate_fps

    # Include t=0 and each whole sample step until the SRT ends.
    sample_times = []
    t = 0.0
    while t <= max_srt_time + 1e-9:
        sample_times.append(round(t, 6))
        t += step_s

    rows = []

    for sample_id, target_time_s in enumerate(sample_times):
        srt_row = nearest_row(srt_df, target_time_s)

        frame_cnt = int(srt_row["frame_cnt"])
        zero_based_frame_index = frame_cnt - 1

        out_name = f"{stream_id.lower()}_frame_{sample_id:05d}_srcframe_{frame_cnt:06d}.jpg"
        out_path = frame_dir / out_name

        extraction_status = "skipped_exists"

        if args.overwrite or not out_path.exists():
            cap.set(cv2.CAP_PROP_POS_FRAMES, zero_based_frame_index)
            ok, frame = cap.read()

            if not ok or frame is None:
                extraction_status = "failed_read"
            else:
                cv2.imwrite(
                    str(out_path),
                    frame,
                    [int(cv2.IMWRITE_JPEG_QUALITY), int(args.jpeg_quality)],
                )
                extraction_status = "ok"

        alignment_error_ms = abs(float(srt_row["video_time_s"]) - target_time_s) * 1000.0

        row = {
            "sample_id": sample_id,
            "stream_id": stream_id,
            "modality": stream_cfg["modality"],
            "role": stream_cfg["role"],
            "target_time_s": target_time_s,
            "srt_video_time_s": float(srt_row["video_time_s"]),
            "alignment_error_ms": alignment_error_ms,
            "srt_index": int(srt_row["srt_index"]),
            "frame_cnt": frame_cnt,
            "zero_based_frame_index": zero_based_frame_index,
            "frame_path": str(out_path),
            "source_video": str(video_path),
            "lat": float(srt_row["lat"]),
            "lon": float(srt_row["lon"]),
            "rel_alt_m": float(srt_row["rel_alt_m"]),
            "abs_alt_m": float(srt_row["abs_alt_m"]),
            "gb_yaw_deg": float(srt_row["gb_yaw_deg"]),
            "gb_pitch_deg": float(srt_row["gb_pitch_deg"]),
            "gb_roll_deg": float(srt_row["gb_roll_deg"]),
            "focal_len_mm": float(srt_row["focal_len_mm"]),
            "dzoom_ratio": float(srt_row["dzoom_ratio"]),
            "extraction_status": extraction_status,
        }

        # Preserve visual-camera-only metadata when present.
        for optional_col in ["iso", "shutter", "fnum", "ev", "color_md", "ae_meter_md"]:
            if optional_col in srt_df.columns:
                row[optional_col] = srt_row.get(optional_col)

        rows.append(row)

    cap.release()

    out_df = pd.DataFrame(rows)

    out_csv = metadata_dir / f"s8_2_extracted_frames_{stream_id}_{args.sample_rate_fps:g}fps.csv"
    out_df.to_csv(out_csv, index=False)

    ok_count = int((out_df["extraction_status"] == "ok").sum())
    skipped_count = int((out_df["extraction_status"] == "skipped_exists").sum())
    failed_count = int((out_df["extraction_status"] == "failed_read").sum())

    summary = {
        "stream_id": stream_id,
        "video_path": str(video_path),
        "parsed_srt_path": str(parsed_srt_path),
        "frame_dir": str(frame_dir),
        "output_csv": str(out_csv),
        "sample_rate_fps": args.sample_rate_fps,
        "samples_requested": int(len(out_df)),
        "extracted_ok": ok_count,
        "skipped_exists": skipped_count,
        "failed_read": failed_count,
        "video_fps_reported": video_fps,
        "video_frame_count_reported": video_frame_count,
        "video_width": video_width,
        "video_height": video_height,
        "video_duration_s_reported": video_duration_s,
        "srt_time_max_s": max_srt_time,
        "alignment_error_ms_max": float(out_df["alignment_error_ms"].max()),
        "alignment_error_ms_median": float(out_df["alignment_error_ms"].median()),
        "lat_min": float(out_df["lat"].min()),
        "lat_max": float(out_df["lat"].max()),
        "lon_min": float(out_df["lon"].min()),
        "lon_max": float(out_df["lon"].max()),
        "rel_alt_min_m": float(out_df["rel_alt_m"].min()),
        "rel_alt_max_m": float(out_df["rel_alt_m"].max()),
        "yaw_min_deg": float(out_df["gb_yaw_deg"].min()),
        "yaw_max_deg": float(out_df["gb_yaw_deg"].max()),
        "pitch_median_deg": float(out_df["gb_pitch_deg"].median()),
    }

    report_path = reports_dir / f"s8_2_extract_frames_{stream_id}_{args.sample_rate_fps:g}fps_summary.json"
    report_path.write_text(json.dumps(summary, indent=2))

    print("S8.2 Villoc sampled frame extraction complete")
    print("--------------------------------------------")
    print(f"Stream: {stream_id}")
    print(f"Video: {video_path}")
    print(f"Frame dir: {frame_dir}")
    print(f"Samples requested: {len(out_df)}")
    print(f"Extracted ok: {ok_count}")
    print(f"Skipped existing: {skipped_count}")
    print(f"Failed reads: {failed_count}")
    print(f"Alignment median ms: {summary['alignment_error_ms_median']:.3f}")
    print(f"Alignment max ms: {summary['alignment_error_ms_max']:.3f}")
    print(f"Saved metadata: {out_csv}")
    print(f"Saved report: {report_path}")


if __name__ == "__main__":
    main()
