from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from uavloc.relative.orb_relative_motion import resolve_project_paths


STAGE_NAME = "09c3_geometry_telemetry_diagnostics"


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def first_existing_column(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    lookup = {str(c).strip().lower(): str(c).strip() for c in df.columns}
    for name in candidates:
        if name.lower() in lookup:
            return lookup[name.lower()]
    return None


def angle_wrap_deg(angle: np.ndarray) -> np.ndarray:
    return (angle + 180.0) % 360.0 - 180.0


def compute_course_from_reference(df: pd.DataFrame) -> pd.Series:
    x_col = first_existing_column(df, ["x_enu_m", "ref_x_enu_m", "x_enu_m_sync"])
    y_col = first_existing_column(df, ["y_enu_m", "ref_y_enu_m", "y_enu_m_sync"])

    if x_col is None or y_col is None:
        return pd.Series(np.nan, index=df.index)

    x = safe_numeric(df[x_col]).to_numpy(dtype=float)
    y = safe_numeric(df[y_col]).to_numpy(dtype=float)

    dx = np.gradient(x)
    dy = np.gradient(y)

    # ENU course convention: 0 deg = North, +90 deg = East.
    course_deg = np.rad2deg(np.arctan2(dx, dy))
    return pd.Series(course_deg, index=df.index)


def summarize_numeric(series: pd.Series) -> dict[str, Any]:
    values = safe_numeric(series).to_numpy(dtype=float)
    finite = values[np.isfinite(values)]

    if finite.size == 0:
        return {
            "valid_count": 0,
            "median": None,
            "mean": None,
            "min": None,
            "max": None,
            "std": None,
        }

    return {
        "valid_count": int(finite.size),
        "median": float(np.nanmedian(finite)),
        "mean": float(np.nanmean(finite)),
        "min": float(np.nanmin(finite)),
        "max": float(np.nanmax(finite)),
        "std": float(np.nanstd(finite)),
    }


def load_sync(output_dir: Path) -> tuple[pd.DataFrame, Path]:
    enriched_csv = output_dir / "metadata" / "synchronized_frames_enriched.csv"
    normal_csv = output_dir / "metadata" / "synchronized_frames.csv"

    sync_csv = enriched_csv if enriched_csv.exists() else normal_csv
    if not sync_csv.exists():
        raise FileNotFoundError(
            f"Missing synchronized metadata. Expected one of:\n{enriched_csv}\n{normal_csv}"
        )

    df = pd.read_csv(sync_csv)
    df.columns = [str(c).strip() for c in df.columns]
    return df, sync_csv


def inspect_calibration(raw_dir: Path) -> dict[str, Any]:
    calib_file = raw_dir / "calibration_data.npz"
    if not calib_file.exists():
        matches = sorted(raw_dir.glob("*.npz"))
        calib_file = matches[0] if matches else None

    if calib_file is None or not calib_file.exists():
        return {"available": False, "reason": "no npz calibration file found"}

    data = np.load(calib_file, allow_pickle=True)
    keys = list(data.keys())

    result: dict[str, Any] = {
        "available": True,
        "path": str(calib_file),
        "keys": keys,
    }

    for key in ["intrinsic_matrix", "camera_matrix", "K", "mtx"]:
        if key in data:
            arr = np.asarray(data[key], dtype=float)
            if arr.shape == (3, 3):
                result.update(
                    {
                        "intrinsic_key": key,
                        "fx": float(arr[0, 0]),
                        "fy": float(arr[1, 1]),
                        "cx": float(arr[0, 2]),
                        "cy": float(arr[1, 2]),
                    }
                )
                break

    if "distCoeff" in data:
        result["distortion_key"] = "distCoeff"
        result["distortion_shape"] = list(np.asarray(data["distCoeff"]).shape)

    return result


def add_barometer_candidates(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    baro_col = first_existing_column(
        out,
        [
            "barometer_altitude",
            "baro_altitude_m",
            "barometer_altitude_m",
            "altitude_barometer",
            "altitude_sync",
            "altitude",
        ],
    )

    out["barometer_altitude_source_column"] = baro_col if baro_col is not None else ""

    if baro_col is None:
        out["baro_altitude_m"] = np.nan
        out["baro_relative_m"] = np.nan
        return out

    baro_alt = safe_numeric(out[baro_col]).to_numpy(dtype=float)
    out["baro_altitude_m"] = baro_alt

    finite = np.isfinite(baro_alt)
    if finite.any():
        first = baro_alt[np.where(finite)[0][0]]
        out["baro_relative_m"] = baro_alt - first
    else:
        out["baro_relative_m"] = np.nan

    for initial_height in [5.0, 10.0, 15.0, 20.0, 30.0, 50.0]:
        out[f"baro_height_candidate_init_{int(initial_height)}m"] = (
            initial_height + out["baro_relative_m"]
        )

    return out


def build_segment_summary(df: pd.DataFrame, name: str, start_imgid: int, end_imgid: int) -> dict[str, Any]:
    if "imgid" not in df.columns:
        seg = df.copy()
    else:
        imgid = safe_numeric(df["imgid"])
        seg = df[(imgid >= start_imgid) & (imgid <= end_imgid)].copy()

    summary: dict[str, Any] = {
        "segment_name": name,
        "start_imgid": int(start_imgid),
        "end_imgid": int(end_imgid),
        "rows": int(len(seg)),
    }

    cols_to_summarize = [
        "height_agl_m",
        "baro_altitude_m",
        "baro_relative_m",
        "baro_height_candidate_init_5m",
        "baro_height_candidate_init_10m",
        "baro_height_candidate_init_15m",
        "baro_height_candidate_init_20m",
        "yaw_deg",
        "reference_course_deg",
        "yaw_minus_course_deg",
        "vel_n_m_s",
        "vel_e_m_s",
        "vel_d_m_s",
    ]

    for col in cols_to_summarize:
        if col in seg.columns:
            summary[col] = summarize_numeric(seg[col])

    return summary


def plot_height_candidates(df: pd.DataFrame, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    x = safe_numeric(df["imgid"]) if "imgid" in df.columns else pd.Series(np.arange(len(df)))

    fig, ax = plt.subplots(figsize=(12, 5))

    for col in [
        "height_agl_m",
        "baro_relative_m",
        "baro_height_candidate_init_5m",
        "baro_height_candidate_init_10m",
        "baro_height_candidate_init_15m",
        "baro_height_candidate_init_20m",
    ]:
        if col in df.columns:
            ax.plot(x, safe_numeric(df[col]), linewidth=1.0, label=col)

    ax.set_title("Height / Barometer Candidate Comparison")
    ax.set_xlabel("imgid")
    ax.set_ylabel("height or relative altitude [m]")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_yaw_course(df: pd.DataFrame, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    x = safe_numeric(df["imgid"]) if "imgid" in df.columns else pd.Series(np.arange(len(df)))

    fig, ax = plt.subplots(figsize=(12, 5))

    if "yaw_deg" in df.columns:
        ax.plot(x, safe_numeric(df["yaw_deg"]), linewidth=1.0, label="yaw_deg")

    if "reference_course_deg" in df.columns:
        ax.plot(x, safe_numeric(df["reference_course_deg"]), linewidth=1.0, label="GNSS/reference course deg")

    if "yaw_minus_course_deg" in df.columns:
        ax.plot(x, safe_numeric(df["yaw_minus_course_deg"]), linewidth=1.0, label="yaw - course deg")

    ax.set_title("Yaw vs Reference Course")
    ax.set_xlabel("imgid")
    ax.set_ylabel("angle [deg]")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect metric geometry inputs: height, barometer, yaw, course, camera calibration."
    )
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    _, _, raw_dir, output_dir, dataset_name = resolve_project_paths(args.config)

    metadata_dir = output_dir / "metadata" / STAGE_NAME
    report_dir = output_dir / "reports" / STAGE_NAME
    figure_dir = output_dir / "figures" / STAGE_NAME

    for d in [metadata_dir, report_dir, figure_dir]:
        d.mkdir(parents=True, exist_ok=True)

    df, sync_csv = load_sync(output_dir)
    df = add_barometer_candidates(df)

    df["reference_course_deg"] = compute_course_from_reference(df)

    if "yaw_deg" in df.columns:
        yaw = safe_numeric(df["yaw_deg"]).to_numpy(dtype=float)
        course = safe_numeric(df["reference_course_deg"]).to_numpy(dtype=float)
        df["yaw_minus_course_deg"] = angle_wrap_deg(yaw - course)
    else:
        df["yaw_minus_course_deg"] = np.nan

    diagnostic_csv = metadata_dir / "metric_geometry_inputs_by_frame.csv"
    summary_json = report_dir / "metric_geometry_summary.json"
    height_plot = figure_dir / "height_candidates.png"
    yaw_plot = figure_dir / "yaw_vs_reference_course.png"

    df.to_csv(diagnostic_csv, index=False)

    segment_summaries = [
        build_segment_summary(df, "full_00001_01000", 1, 1000),
        build_segment_summary(df, "full_40000_41000", 40000, 41000),
    ]

    summary = {
        "dataset_name": dataset_name,
        "sync_csv": str(sync_csv),
        "rows": int(len(df)),
        "documentation_notes": {
            "zurich_mav_doc_altitude_agl": "Dataset documentation describes low-altitude flight around 5-15 m above ground.",
            "zurich_mav_doc_camera": "GoPro Hero 4 rolling-shutter camera, 1920x1080 images, approx. 30 ms readout.",
            "diagnostic_reason": (
                "Metric scaling needs true AGL and correct camera/body/yaw convention. "
                "Barometer altitude is inspected here as a candidate, not assumed to be true AGL."
            ),
        },
        "camera_calibration": inspect_calibration(raw_dir),
        "columns_present": list(df.columns),
        "global_summaries": {},
        "segments": segment_summaries,
        "outputs": {
            "diagnostic_csv": str(diagnostic_csv),
            "summary_json": str(summary_json),
            "height_plot": str(height_plot),
            "yaw_plot": str(yaw_plot),
        },
    }

    for col in [
        "height_agl_m",
        "baro_altitude_m",
        "baro_relative_m",
        "baro_height_candidate_init_5m",
        "baro_height_candidate_init_10m",
        "baro_height_candidate_init_15m",
        "baro_height_candidate_init_20m",
        "yaw_deg",
        "reference_course_deg",
        "yaw_minus_course_deg",
    ]:
        if col in df.columns:
            summary["global_summaries"][col] = summarize_numeric(df[col])

    plot_height_candidates(df, height_plot)
    plot_yaw_course(df, yaw_plot)

    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Metric geometry diagnostics generated")
    print("-------------------------------------")
    print(f"Dataset:        {dataset_name}")
    print(f"Input sync CSV: {sync_csv}")
    print(f"Rows:           {len(df)}")
    print(f"Diagnostic CSV: {diagnostic_csv}")
    print(f"Summary JSON:   {summary_json}")
    print(f"Height plot:    {height_plot}")
    print(f"Yaw plot:       {yaw_plot}")

    print("\nSegment summaries")
    print("-----------------")
    for seg in segment_summaries:
        print(f"\n{seg['segment_name']} rows={seg['rows']}")
        for col in ["height_agl_m", "baro_altitude_m", "baro_relative_m", "yaw_deg", "reference_course_deg", "yaw_minus_course_deg"]:
            if col in seg:
                print(f"{col}: {seg[col]}")


if __name__ == "__main__":
    main()