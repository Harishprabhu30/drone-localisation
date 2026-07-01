from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from uavloc.relative.orb_relative_motion import resolve_project_paths


STAGE_NAME = "09d_relative_evaluation_summary"


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def safe_get(data: dict[str, Any], path: list[str], default=None):
    cur: Any = data
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur

def classify_run(run_name: str) -> str:
    name = str(run_name)

    if name.startswith("full_"):
        return "official"
    if name.startswith("tuned_"):
        return "tuned_diagnostic"
    if name.startswith("debug_"):
        return "debug"
    if name.startswith("sweep_"):
        return "sweep"

    return "other"

def collect_orb_relative_rows(output_dir: Path) -> list[dict[str, Any]]:
    root = output_dir / "reports" / "09b_orb_relative_motion_subset"
    rows: list[dict[str, Any]] = []

    if not root.exists():
        return rows

    for summary_path in sorted(root.glob("*/orb_relative_motion_summary.json")):
        data = load_json(summary_path)
        run_name = data.get("run_name", summary_path.parent.name)
        subset = data.get("subset", {})

        rows.append(
            {
                "stage": "09b_orb_relative_motion",
                "run_name": run_name,
                "run_type": classify_run(run_name),
                "dataset_name": data.get("dataset_name"),
                "frames": data.get("frames_used"),
                "selected_rows_before_stride": data.get("selected_rows_before_stride"),
                "frames_used_after_stride": data.get("frames_used_after_stride"),
                "first_used_imgid": data.get("first_used_imgid"),
                "last_used_imgid": data.get("last_used_imgid"),
                "stride": data.get("stride"),
                "attempted_pairs": data.get("attempted_pairs"),
                "ok_pairs": data.get("ok_pairs"),
                "failed_pairs": data.get("failed_pairs"),
                "median_good_matches": data.get("median_good_matches"),
                "median_ransac_inliers": data.get("median_ransac_inliers"),
                "median_inlier_ratio": data.get("median_inlier_ratio"),
                "orb_aligned_rmse_m": safe_get(data, ["alignment_to_reference", "aligned_rmse_m"]),
                "orb_shape_scale_m_per_px": safe_get(data, ["alignment_to_reference", "scale_m_per_px"]),
                "orb_shape_rotation_deg": safe_get(data, ["alignment_to_reference", "rotation_deg"]),
                "height_column": None,
                "yaw_column": None,
                "estimated_path_length_m": None,
                "reference_path_length_m": None,
                "rmse_m": None,
                "mean_error_m": None,
                "max_error_m": None,
                "final_error_m": None,
                "drift_per_100m": None,
                "mean_confidence": None,
                "summary_json": str(summary_path),
                "notes": "ORB image-motion only. Shape alignment is diagnostic evaluation, not localization.",
            }
        )

    return rows


def collect_metric_rows(output_dir: Path) -> list[dict[str, Any]]:
    root = output_dir / "reports" / "09c_orb_metric_scaling_subset"
    rows: list[dict[str, Any]] = []

    if not root.exists():
        return rows

    for summary_path in sorted(root.glob("*/orb_metric_scaling_summary.json")):
        data = load_json(summary_path)
        run_name = data.get("run_name", summary_path.parent.name)
        ref = data.get("reference_evaluation", {})

        rows.append(
            {
                "stage": "09c_orb_metric_scaling",
                "run_name": run_name,
                "run_type": classify_run(run_name),
                "dataset_name": data.get("dataset_name"),
                "frames": data.get("frames"),
                "selected_rows_before_stride": None,
                "frames_used_after_stride": data.get("frames"),
                "first_used_imgid": None,
                "last_used_imgid": None,
                "stride": None,
                "attempted_pairs": None,
                "ok_pairs": None,
                "failed_pairs": None,
                "median_good_matches": None,
                "median_ransac_inliers": None,
                "median_inlier_ratio": None,
                "orb_aligned_rmse_m": None,
                "orb_shape_scale_m_per_px": None,
                "orb_shape_rotation_deg": None,
                "height_column": data.get("height_column") or safe_get(data, ["height", "column"]),
                "yaw_column": data.get("yaw_column") or safe_get(data, ["yaw_heading", "column"]),
                "estimated_path_length_m": data.get("estimated_path_length_m"),
                "reference_path_length_m": ref.get("reference_path_length_m"),
                "rmse_m": ref.get("rmse_m"),
                "mean_error_m": ref.get("mean_error_m"),
                "max_error_m": ref.get("max_error_m"),
                "final_error_m": ref.get("final_error_m"),
                "drift_per_100m": ref.get("drift_per_100m"),
                "shape_rmse_m": safe_get(ref, ["shape_alignment", "rmse_m"]),
                "shape_rotation_deg": safe_get(ref, ["shape_alignment", "rotation_deg"]),
                "shape_scale": safe_get(ref, ["shape_alignment", "scale"]),
                "mean_confidence": safe_get(data, ["confidence", "mean"]),
                "scale_multiplier": data.get("scale_multiplier"),
                "yaw_offset_deg": data.get("yaw_offset_deg"),
                "image_x_to_right_sign": safe_get(data, ["axis_mapping", "image_x_to_right_sign"]),
                "image_y_to_forward_sign": safe_get(data, ["axis_mapping", "image_y_to_forward_sign"]),
                "summary_json": str(summary_path),
                "notes": data.get("important_warning"),
            }
        )

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build stable relative localization evaluation summary table."
    )
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    _, _, _, output_dir, dataset_name = resolve_project_paths(args.config)

    report_dir = output_dir / "reports" / STAGE_NAME
    report_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    rows.extend(collect_orb_relative_rows(output_dir))
    rows.extend(collect_metric_rows(output_dir))

    df = pd.DataFrame(rows)

    if not df.empty:
        sort_cols = [c for c in ["stage", "run_name"] if c in df.columns]
        df = df.sort_values(sort_cols)

    csv_path = report_dir / "evaluation_summary_all_runs.csv"
    official_csv_path = report_dir / "evaluation_summary_official_runs.csv"
    diagnostic_csv_path = report_dir / "evaluation_summary_diagnostic_runs.csv"
    json_path = report_dir / "evaluation_summary.json"

    df.to_csv(csv_path, index=False)

    if "run_type" in df.columns:
        official_df = df[df["run_type"] == "official"].copy()
        diagnostic_df = df[df["run_type"] != "official"].copy()
    else:
        official_df = df.copy()
        diagnostic_df = pd.DataFrame()

    official_df.to_csv(official_csv_path, index=False)
    diagnostic_df.to_csv(diagnostic_csv_path, index=False)

    payload = {
        "dataset_name": dataset_name,
        "rows_all": int(len(df)),
        "rows_official": int(len(official_df)),
        "rows_diagnostic": int(len(diagnostic_df)),
        "csv_all_runs": str(csv_path),
        "csv_official_runs": str(official_csv_path),
        "csv_diagnostic_runs": str(diagnostic_csv_path),
        "columns": list(df.columns),
        "interpretation_notes": [
            "09B rows evaluate ORB image-motion tracking quality.",
            "09C rows evaluate metric ENU conversion using height/yaw/camera assumptions.",
            "NaN values are expected where a metric does not apply to that stage.",
            "GNSS/reference is used only for evaluation, not localization.",
            "Zurich MAV is useful for ORB tracking and failure analysis, but simple nadir altitude scaling is not reliable across all windows.",
        ],
    }

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("Relative evaluation summary generated")
    print("------------------------------------")
    print(f"Dataset:          {dataset_name}")
    print(f"Rows all:         {len(df)}")
    print(f"Rows official:    {len(official_df)}")
    print(f"Rows diagnostic:  {len(diagnostic_df)}")
    print(f"CSV all runs:     {csv_path}")
    print(f"CSV official:     {official_csv_path}")
    print(f"CSV diagnostic:   {diagnostic_csv_path}")
    print(f"JSON:             {json_path}")

    if not official_df.empty:
        display_cols = [
            "stage",
            "run_type",
            "run_name",
            "frames",
            "median_inlier_ratio",
            "estimated_path_length_m",
            "reference_path_length_m",
            "rmse_m",
            "final_error_m",
            "drift_per_100m",
            "mean_confidence",
        ]
        display_cols = [c for c in display_cols if c in official_df.columns]

        print("\nOfficial runs preview")
        print("---------------------")
        print(official_df[display_cols].to_string(index=False))
    else:
        print("\nOfficial runs preview")
        print("---------------------")
        print("No official runs found.")


if __name__ == "__main__":
    main()