from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from uavloc.data.villoc_srt import parse_villoc_srt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config_path = Path(args.config)
    cfg = yaml.safe_load(config_path.read_text())

    raw_root = Path(cfg["dataset"]["raw_root"])
    output_root = Path(cfg["dataset"]["output_root"])

    metadata_dir = output_root / "metadata"
    reports_dir = output_root / "reports"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    summaries = {}

    for stream_id, stream_cfg in cfg["streams"].items():
        srt_path = raw_root / stream_cfg["srt"]

        df = parse_villoc_srt(
            srt_path,
            video_id=stream_id,
            modality=stream_cfg["modality"],
            role=stream_cfg["role"],
        )

        out_csv = metadata_dir / f"s8_1_srt_parsed_{stream_id}.csv"
        df.to_csv(out_csv, index=False)

        summary = {
            "stream_id": stream_id,
            "srt_path": str(srt_path),
            "rows": int(len(df)),
            "frame_cnt_min": int(df["frame_cnt"].min()) if len(df) else None,
            "frame_cnt_max": int(df["frame_cnt"].max()) if len(df) else None,
            "video_time_min_s": float(df["video_time_s"].min()) if len(df) else None,
            "video_time_max_s": float(df["video_time_s"].max()) if len(df) else None,
            "lat_min": float(df["lat"].min()) if len(df) else None,
            "lat_max": float(df["lat"].max()) if len(df) else None,
            "lon_min": float(df["lon"].min()) if len(df) else None,
            "lon_max": float(df["lon"].max()) if len(df) else None,
            "rel_alt_min_m": float(df["rel_alt_m"].min()) if len(df) else None,
            "rel_alt_max_m": float(df["rel_alt_m"].max()) if len(df) else None,
            "yaw_min_deg": float(df["gb_yaw_deg"].min()) if len(df) else None,
            "yaw_max_deg": float(df["gb_yaw_deg"].max()) if len(df) else None,
            "pitch_median_deg": float(df["gb_pitch_deg"].median()) if len(df) else None,
            "output_csv": str(out_csv),
        }

        summaries[stream_id] = summary

    report_path = reports_dir / "s8_1_srt_parse_summary.json"
    report_path.write_text(json.dumps(summaries, indent=2))

    print("S8.1 Villoc SRT parse complete")
    print("------------------------------")
    for stream_id, summary in summaries.items():
        print(
            f"{stream_id}: rows={summary['rows']} "
            f"frames={summary['frame_cnt_min']}..{summary['frame_cnt_max']} "
            f"time={summary['video_time_min_s']:.3f}..{summary['video_time_max_s']:.3f}s "
            f"pitch_median={summary['pitch_median_deg']:.2f}"
        )
    print(f"\nSaved report: {report_path}")


if __name__ == "__main__":
    main()
