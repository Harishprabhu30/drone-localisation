from pathlib import Path
from typing import Dict, List, Optional, Any
import json

import numpy as np
import pandas as pd


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


ZURICH_COLUMNS = {
    "onboard_gps_file": [
        "timestamp",
        "imgid",
        "lat_raw",
        "lon_raw",
        "alt_raw",
        "s_variance_m_s",
        "c_variance_rad",
        "fix_type",
        "eph_m",
        "epv_m",
        "vel_n_m_s",
        "vel_e_m_s",
        "vel_d_m_s",
        "num_sat",
    ],
    "groundtruth_agl_file": [
        "imgid",
        "x_gt",
        "y_gt",
        "z_gt",
        "omega_yaw_deg",
        "phi_pitch_deg",
        "kappa_roll_deg",
        "x_gps",
        "y_gps",
        "z_gps",
    ],
    "onboard_pose_file": [
        "timestamp",
        "omega_x",
        "omega_y",
        "omega_z",
        "accel_x",
        "accel_y",
        "accel_z",
        "vel_x",
        "vel_y",
        "vel_z",
        "acc_bias_x",
        "acc_bias_y",
        "acc_bias_z",
        "azimuth",
        "attitude_w",
        "attitude_x",
        "attitude_y",
        "attitude_z",
        "height",
        "altitude",
        "veh_pitch",
        "tether_angle",
        "tether_angle_dot",
        "tether_force",
        "gps_on",
    ],
    "barometer_file": [
        "timestamp",
        "pressure",
        "altitude",
        "temperature",
    ],
    "raw_accel_file": [
        "timestamp",
        "error_count",
        "x",
        "y",
        "z",
        "temperature",
        "range_rad_s",
        "scaling",
        "x_raw",
        "y_raw",
        "z_raw",
        "temperature_raw",
    ],
    "raw_gyro_file": [
        "timestamp",
        "error_count",
        "x",
        "y",
        "z",
        "temperature",
        "range_rad_s",
        "scaling",
        "x_raw",
        "y_raw",
        "z_raw",
        "temperature_raw",
    ],
    "streetview_gps_file": [
        "latitude",
        "longitude",
        "yaw_degree",
        "tilt_yaw_degree",
        "tilt_pitch_degree",
        "auxiliary",
    ],
}


def resolve_path(path_value: Optional[str]) -> Optional[Path]:
    if path_value is None:
        return None
    return Path(path_value)

def read_zurich_csv(path: Path, columns: List[str]) -> pd.DataFrame:
    """
    Robust CSV reader for Zurich MAV logs.

    Handles:
    - CSV files with header rows
    - CSV files without header rows
    - typo headers such as Timestemp / Timpstemp
    - extra trailing columns in some rows
    - blank rows
    """

    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    try:
        df = pd.read_csv(
            path,
            header=None,
            usecols=list(range(len(columns))),
            engine="python",
            skip_blank_lines=True,
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to read CSV file: {path}\nOriginal error: {exc}")

    df.columns = columns

    # Drop header-like rows that leaked into the data.
    # Example: Timestamp,imgid,... or imgid,x_gt,...
    expected_tokens = {c.strip().lower() for c in columns}
    extra_header_tokens = {
        "timestamp",
        "timestemp",
        "timpstemp",
        "imgid",
        "latitude",
        "longitude",
        "lat",
        "lon",
        "alt",
    }

    def is_header_like(row) -> bool:
        values = [str(v).strip().lower() for v in row.values]
        matches = sum(
            (v in expected_tokens) or (v in extra_header_tokens)
            for v in values
        )
        return matches >= 2

    header_like_mask = df.apply(is_header_like, axis=1)
    df = df.loc[~header_like_mask].copy()

    # Convert all fields to numeric.
    # Invalid text values become NaN instead of crashing the whole loader.
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Remove completely empty rows.
    df = df.dropna(how="all").reset_index(drop=True)

    return df

def summarize_dataframe(df: pd.DataFrame, timestamp_col: Optional[str] = "timestamp") -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "rows": int(len(df)),
        "columns": list(df.columns),
        "nan_counts": {col: int(df[col].isna().sum()) for col in df.columns},
    }

    if timestamp_col and timestamp_col in df.columns and len(df) > 0:
        valid_timestamps = pd.to_numeric(df[timestamp_col], errors="coerce").dropna()

        if len(valid_timestamps) > 0:
            t_min = float(valid_timestamps.min())
            t_max = float(valid_timestamps.max())
            summary["timestamp_min"] = t_min
            summary["timestamp_max"] = t_max
            summary["duration_raw_units"] = t_max - t_min
        else:
            summary["timestamp_warning"] = "No valid numeric timestamps found."

    return summary


def find_image_files(image_dir: Path) -> List[Path]:
    if not image_dir.exists():
        return []

    return sorted(
        [
            p for p in image_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ]
    )


def summarize_image_folder(image_dir: Path) -> Dict[str, Any]:
    image_files = find_image_files(image_dir)

    summary: Dict[str, Any] = {
        "path": str(image_dir),
        "exists": image_dir.exists(),
        "image_count": len(image_files),
        "extensions": sorted(list({p.suffix.lower() for p in image_files})),
    }

    if image_files:
        summary["first_image"] = str(image_files[0])
        summary["last_image"] = str(image_files[-1])

    return summary


def summarize_calibration(calibration_file: Optional[Path]) -> Dict[str, Any]:
    if calibration_file is None:
        return {"available": False, "reason": "No calibration file path provided"}

    if not calibration_file.exists():
        return {
            "available": False,
            "path": str(calibration_file),
            "reason": "Calibration file does not exist",
        }

    try:
        data = np.load(calibration_file)
        return {
            "available": True,
            "path": str(calibration_file),
            "keys": list(data.keys()),
        }
    except Exception as exc:
        return {
            "available": False,
            "path": str(calibration_file),
            "reason": f"Could not read calibration npz: {exc}",
        }


class ZurichMAVDataset:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.paths = config.get("paths", {})

    def path(self, key: str) -> Optional[Path]:
        return resolve_path(self.paths.get(key))

    def load_log(self, key: str) -> pd.DataFrame:
        if key not in ZURICH_COLUMNS:
            raise KeyError(f"Unknown Zurich log key: {key}")

        file_path = self.path(key)

        if file_path is None:
            raise ValueError(f"Path not provided in config for key: {key}")
        
        return read_zurich_csv(file_path, ZURICH_COLUMNS[key])    

    def load_all_available_logs(self) -> Dict[str, pd.DataFrame]:
        logs: Dict[str, pd.DataFrame] = {}

        for key in ZURICH_COLUMNS:
            file_path = self.path(key)

            if file_path is None:
                continue

            if file_path.exists():
                logs[key] = self.load_log(key)

        return logs

    def create_report(self) -> Dict[str, Any]:
        raw_data_dir = self.path("raw_data_dir")
        output_dir = self.path("output_dir")
        mav_images_dir = self.path("mav_images_dir")
        calib_images_dir = self.path("calib_images_dir")
        streetview_images_dir = self.path("streetview_images_dir")
        calibration_file = self.path("calibration_file")

        report: Dict[str, Any] = {
            "dataset_name": self.config.get("dataset_name"),
            "dataset_type": self.config.get("dataset_type"),
            "paths": {
                "raw_data_dir": str(raw_data_dir) if raw_data_dir else None,
                "output_dir": str(output_dir) if output_dir else None,
            },
            "image_folders": {},
            "log_files": {},
            "calibration": {},
            "notes": [],
        }

        if raw_data_dir:
            report["paths"]["raw_data_dir_exists"] = raw_data_dir.exists()

        if output_dir:
            report["paths"]["output_dir_exists"] = output_dir.exists()

        if mav_images_dir:
            report["image_folders"]["mav_images"] = summarize_image_folder(mav_images_dir)

        if calib_images_dir:
            report["image_folders"]["calib_images"] = summarize_image_folder(calib_images_dir)

        if streetview_images_dir:
            report["image_folders"]["streetview_images"] = summarize_image_folder(streetview_images_dir)

        logs = self.load_all_available_logs()

        for key, df in logs.items():
            timestamp_col = "timestamp" if "timestamp" in df.columns else None
            report["log_files"][key] = summarize_dataframe(df, timestamp_col=timestamp_col)

        if "onboard_gps_file" in logs:
            gps = logs["onboard_gps_file"].copy()

            gps_latlon_scale = float(self.config.get("telemetry", {}).get("gps_latlon_scale", 1e-7))
            gps_alt_scale = float(self.config.get("telemetry", {}).get("gps_alt_scale", 1e-3))

            gps["lat_deg"] = gps["lat_raw"] * gps_latlon_scale
            gps["lon_deg"] = gps["lon_raw"] * gps_latlon_scale
            gps["alt_m"] = gps["alt_raw"] * gps_alt_scale

            report["log_files"]["onboard_gps_file"]["lat_deg_min"] = float(gps["lat_deg"].min())
            report["log_files"]["onboard_gps_file"]["lat_deg_max"] = float(gps["lat_deg"].max())
            report["log_files"]["onboard_gps_file"]["lon_deg_min"] = float(gps["lon_deg"].min())
            report["log_files"]["onboard_gps_file"]["lon_deg_max"] = float(gps["lon_deg"].max())
            report["log_files"]["onboard_gps_file"]["alt_m_min"] = float(gps["alt_m"].min())
            report["log_files"]["onboard_gps_file"]["alt_m_max"] = float(gps["alt_m"].max())

        report["calibration"] = summarize_calibration(calibration_file)

        expected_files = [
            "onboard_gps_file",
            "groundtruth_agl_file",
            "onboard_pose_file",
            "barometer_file",
            "raw_accel_file",
            "raw_gyro_file",
        ]

        missing = []
        for key in expected_files:
            p = self.path(key)
            if p is None or not p.exists():
                missing.append(key)

        if missing:
            report["notes"].append(f"Missing expected files: {missing}")
        else:
            report["notes"].append("All main Zurich MAV log files found.")

        return report

    def save_report(self, report: Dict[str, Any]) -> Path:
        output_dir = self.path("output_dir")

        if output_dir is None:
            raise ValueError("output_dir is not defined in config.")

        report_dir = output_dir / "reports"
        report_dir.mkdir(parents=True, exist_ok=True)

        report_path = report_dir / "dataset_report.json"

        with report_path.open("w") as f:
            json.dump(report, f, indent=2)

        return report_path