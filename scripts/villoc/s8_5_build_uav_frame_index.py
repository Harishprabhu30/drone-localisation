'''
Command Executed:

export PYTHONPATH=$PWD/src
python scripts/villoc/s8_5_build_uav_frame_index.py \
  --config configs/dataset_villoc_90deg.yaml \
  --stream V \
  --sample-rate-fps 1 \
  --golden-n 20

2. running traj01 villoc dataset:

mkdir -p outputs/villoc/traj01_90deg_stable120m/logs/s8_5_uav_index

python scripts/villoc/s8_5_build_uav_frame_index.py \
  --config configs/dataset_villoc_traj01_90deg_stable120m.yaml \
  --stream V \
  --sample-rate-fps 1 \
  --golden-n 20 \
  2>&1 | tee \
  outputs/villoc/traj01_90deg_stable120m/logs/s8_5_uav_index/s8_5_uav_index_V_1fps.log

'''

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def relpath_posix(path_value: str | Path, repo_root: Path) -> str:
    path = Path(path_value)
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except Exception:
        return path.as_posix()


def build_uav_index(
    audit_df: pd.DataFrame,
    *,
    repo_root: Path,
    dataset_name: str,
    sequence_name: str,
    view_angle_group: str,
) -> pd.DataFrame:
    df = audit_df.copy()

    # token0_id is the canonical sequential identity for this extracted dataset.
    # source_frame_cnt keeps the original video/SRT frame number.
    out = pd.DataFrame()

    out["dataset_name"] = [dataset_name] * len(df)
    out["sequence_name"] = [sequence_name] * len(df)
    out["stream_id"] = df["stream_id"]
    out["modality"] = df["modality"]
    out["role"] = df["role"]
    out["view_angle_group"] = [view_angle_group] * len(df)

    out["token0_id"] = df["sample_id"].astype(int) + 1
    out["sample_id"] = df["sample_id"].astype(int)
    out["token1_order"] = df["sample_id"].astype(int)
    out["source_frame_cnt"] = df["frame_cnt"].astype(int)
    out["zero_based_frame_index"] = df["zero_based_frame_index"].astype(int)

    out["timestamp_s"] = df["target_time_s"].astype(float)
    out["video_time_s"] = df["srt_video_time_s"].astype(float)
    out["alignment_error_ms"] = df["alignment_error_ms"].astype(float)

    out["image_path"] = df["frame_path"].astype(str)
    out["image_path_relative"] = [
        relpath_posix(p, repo_root) for p in df["frame_path"].astype(str)
    ]
    out["source_video"] = df["source_video"].astype(str)

    # Reference coordinates. These are allowed for indexing/evaluation/plots,
    # but must not be used later for retrieval/verifier ranking.
    out["lat"] = df["lat"].astype(float)
    out["lon"] = df["lon"].astype(float)
    out["latitude"] = df["lat"].astype(float)
    out["longitude"] = df["lon"].astype(float)
    out["x_enu_m"] = df["x_enu_m"].astype(float)
    out["y_enu_m"] = df["y_enu_m"].astype(float)
    out["z_enu_m"] = df["z_enu_m"].astype(float)

    out["rel_alt_m"] = df["rel_alt_m"].astype(float)
    out["abs_alt_m"] = df["abs_alt_m"].astype(float)

    out["gb_yaw_deg"] = df["gb_yaw_deg"].astype(float)
    out["gb_pitch_deg"] = df["gb_pitch_deg"].astype(float)
    out["gb_roll_deg"] = df["gb_roll_deg"].astype(float)

    out["focal_len_mm"] = df["focal_len_mm"].astype(float)
    out["dzoom_ratio"] = df["dzoom_ratio"].astype(float)

    # Visual audit metrics.
    out["image_width"] = df["image_width"].astype(int)
    out["image_height"] = df["image_height"].astype(int)
    out["laplacian_var"] = df["laplacian_var"].astype(float)
    out["gray_mean"] = df["gray_mean"].astype(float)
    out["gray_std"] = df["gray_std"].astype(float)
    out["edge_density"] = df["edge_density"].astype(float)

    # Optional visual metadata.
    for col in ["iso", "shutter", "fnum", "ev", "color_md", "ae_meter_md"]:
        if col in df.columns:
            out[col] = df[col]

    out["image_read_ok"] = df["image_read_ok"].astype(bool)
    out["extraction_status"] = df["extraction_status"]
    out["reference_usage"] = "evaluation_only"
    out["gt_leakage_rule"] = "do_not_use_lat_lon_or_enu_for_retrieval_ranking"

    return out


def build_golden_manifest(index_df: pd.DataFrame, n: int) -> pd.DataFrame:
    if len(index_df) <= n:
        golden = index_df.copy()
    else:
        idx = np.linspace(0, len(index_df) - 1, n).round().astype(int)
        idx = sorted(set(int(i) for i in idx))

        # If rounding produced fewer than n, fill with earliest missing indices.
        if len(idx) < n:
            for i in range(len(index_df)):
                if i not in idx:
                    idx.append(i)
                if len(idx) == n:
                    break
            idx = sorted(idx)

        golden = index_df.iloc[idx].copy()

    golden = golden.reset_index(drop=True)
    golden["golden_rank"] = np.arange(1, len(golden) + 1)
    golden["golden_reason"] = "uniform_time_coverage_for_first_smoke_tests"

    return golden


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--stream", default="V")
    parser.add_argument("--sample-rate-fps", type=float, default=1.0)
    parser.add_argument("--golden-n", type=int, default=20)
    args = parser.parse_args()

    repo_root = Path.cwd()
    cfg = yaml.safe_load(Path(args.config).read_text())

    dataset_cfg = cfg["dataset"]
    output_root = Path(dataset_cfg["output_root"])
    metadata_dir = output_root / "metadata"
    reports_dir = output_root / "reports"

    dataset_name = dataset_cfg.get("name") or output_root.name
    sequence_name = dataset_cfg.get("sequence_name") or dataset_cfg.get("folder_name") or output_root.name
    view_angle_group = dataset_cfg.get("view_angle_group") or dataset_cfg.get("view") or "unknown"

    audit_csv = metadata_dir / f"s8_4_visual_frame_audit_{args.stream}_{args.sample_rate_fps:g}fps.csv"
    if not audit_csv.exists():
        raise FileNotFoundError(f"Missing S8.4 audit CSV: {audit_csv}")

    audit_df = pd.read_csv(audit_csv)

    index_df = build_uav_index(
        audit_df,
        repo_root=repo_root,
        dataset_name=dataset_name,
        sequence_name=sequence_name,
        view_angle_group=view_angle_group,
    )

    golden_df = build_golden_manifest(index_df, args.golden_n)

    out_index_csv = metadata_dir / f"s8_5_uav_frames_index_{args.stream.lower()}_{args.sample_rate_fps:g}fps.csv"
    out_golden_csv = metadata_dir / f"s8_5_golden{args.golden_n}_manifest_{args.stream.lower()}_{args.sample_rate_fps:g}fps.csv"

    index_df.to_csv(out_index_csv, index=False)
    golden_df.to_csv(out_golden_csv, index=False)

    path_length = float(
        np.sqrt(
            index_df["x_enu_m"].diff().fillna(0.0).pow(2)
            + index_df["y_enu_m"].diff().fillna(0.0).pow(2)
        ).sum()
    )
    displacement = float(
        np.hypot(
            index_df["x_enu_m"].iloc[-1] - index_df["x_enu_m"].iloc[0],
            index_df["y_enu_m"].iloc[-1] - index_df["y_enu_m"].iloc[0],
        )
    )

    summary = {
        "stage": "S8.5",
        "status": "PASS_CANDIDATE",
        "dataset_name": dataset_name,
        "sequence_name": sequence_name,
        "view_angle_group": view_angle_group,
        "stream_id": args.stream,
        "sample_rate_fps": args.sample_rate_fps,
        "input_audit_csv": str(audit_csv),
        "output_uav_index_csv": str(out_index_csv),
        "output_golden_manifest_csv": str(out_golden_csv),
        "rows": int(len(index_df)),
        "golden_rows": int(len(golden_df)),
        "token0_id_min": int(index_df["token0_id"].min()),
        "token0_id_max": int(index_df["token0_id"].max()),
        "source_frame_cnt_min": int(index_df["source_frame_cnt"].min()),
        "source_frame_cnt_max": int(index_df["source_frame_cnt"].max()),
        "time_min_s": float(index_df["timestamp_s"].min()),
        "time_max_s": float(index_df["timestamp_s"].max()),
        "duration_s": float(index_df["timestamp_s"].max() - index_df["timestamp_s"].min()),
        "lat_range": [
            float(index_df["lat"].min()),
            float(index_df["lat"].max()),
        ],
        "lon_range": [
            float(index_df["lon"].min()),
            float(index_df["lon"].max()),
        ],
        "x_range_m": [
            float(index_df["x_enu_m"].min()),
            float(index_df["x_enu_m"].max()),
        ],
        "y_range_m": [
            float(index_df["y_enu_m"].min()),
            float(index_df["y_enu_m"].max()),
        ],
        "z_range_m": [
            float(index_df["z_enu_m"].min()),
            float(index_df["z_enu_m"].max()),
        ],
        "total_2d_path_length_m": path_length,
        "start_to_end_2d_displacement_m": displacement,
        "rel_alt_range_m": [
            float(index_df["rel_alt_m"].min()),
            float(index_df["rel_alt_m"].max()),
        ],
        "yaw_range_deg": [
            float(index_df["gb_yaw_deg"].min()),
            float(index_df["gb_yaw_deg"].max()),
        ],
        "pitch_median_deg": float(index_df["gb_pitch_deg"].median()),
        "image_width_unique": sorted([int(x) for x in index_df["image_width"].unique()]),
        "image_height_unique": sorted([int(x) for x in index_df["image_height"].unique()]),
        "laplacian_var_median": float(index_df["laplacian_var"].median()),
        "edge_density_median": float(index_df["edge_density"].median()),
        "alignment_error_ms_median": float(index_df["alignment_error_ms"].median()),
        "alignment_error_ms_max": float(index_df["alignment_error_ms"].max()),
        "reference_usage": "SRT lat/lon/ENU are for dataset indexing, map bbox, visualization, and evaluation only.",
        "gt_leakage_rule": "Do not use lat/lon/x_enu/y_enu/eval distance for retrieval, verifier ranking, correction acceptance, or threshold tuning.",
    }

    report_path = reports_dir / f"s8_5_uav_index_summary_{args.stream}_{args.sample_rate_fps:g}fps.json"
    report_path.write_text(json.dumps(summary, indent=2))

    print("S8.5 Villoc SatLoc-style UAV frame index complete")
    print("------------------------------------------------")
    print(f"Input audit CSV:        {audit_csv}")
    print(f"UAV index CSV:          {out_index_csv}")
    print(f"Golden manifest CSV:    {out_golden_csv}")
    print(f"Report:                 {report_path}")
    print(f"Rows:                   {len(index_df)}")
    print(f"Golden rows:            {len(golden_df)}")
    print(f"token0_id range:         {summary['token0_id_min']}..{summary['token0_id_max']}")
    print(f"source frame range:      {summary['source_frame_cnt_min']}..{summary['source_frame_cnt_max']}")
    print(f"Path length:            {path_length:.2f} m")
    print(f"Start-end displacement: {displacement:.2f} m")
    print(f"Rel alt range:          {summary['rel_alt_range_m'][0]:.2f}..{summary['rel_alt_range_m'][1]:.2f} m")
    print(f"Pitch median:           {summary['pitch_median_deg']:.2f} deg")
    print()
    print("Golden token0_ids:")
    print(",".join(str(int(x)) for x in golden_df["token0_id"].tolist()))


if __name__ == "__main__":
    main()
