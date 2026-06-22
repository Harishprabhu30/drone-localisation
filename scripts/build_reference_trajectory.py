import argparse
import json
from pathlib import Path

from uavloc.utils.config import load_config
from uavloc.data.zurich_loader import ZurichMAVDataset
from uavloc.geometry.gps_enu import add_local_enu_from_wgs84, save_origin_metadata

def scale_gps_columns(gps, config):
    telemetry_cfg = config.get("telemetry", {})

    latlon_scale = float(telemetry_cfg.get("gps_latlon_scale", 1e-7))
    alt_scale = float(telemetry_cfg.get("gps_alt_scale", 1e-3))

    gps = gps.copy()

    gps['lat'] = gps["lat_raw"] * latlon_scale
    gps['lon'] = gps['lon_raw'] * latlon_scale
    gps['alt'] = gps['alt_raw'] * alt_scale

    return gps

def build_reference_trajectory(config):
    dataset = ZurichMAVDataset(config)

    output_dir = Path(config["paths"]["output_dir"])
    trajectory_dir = output_dir / "trajectories"
    report_dir = output_dir / "reports"

    trajectory_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    gps = dataset.load_log("onboard_gps_file")
    gps = scale_gps_columns(gps, config)

    rows_before_filter = len(gps)

    gps = gps.dropna(subset=["lat", "lon", "alt"]).copy()

    used_fix_filter = False
    
    if "fix_type" in gps.columns:
        gps_good = gps[gps["fix_type"] >= 2].copy()

        if  len(gps_good > 10):
            gps = gps_good
            used_fix_filter = True
        
    epsg = int(config.get("coordinate", {}).get("utm_epsg", 32632))

    ref, origin = add_local_enu_from_wgs84(
        gps,
        lat_col="lat",
        lon_col="lon",
        alt_col="alt",
        epsg=epsg
    )

    ref["reference_type"] = "onboard_gps"

    output_columns = [
        "timestamp",
        "imgid",
        "lat",
        "lon",
        "alt",
        "x_enu_m",
        "y_enu_m",
        "z_enu_m",
        "reference_type",
        "fix_type", 
        "num_sat",
        "eph_m",
        "epv_m",
        "vel_n_m_s",
        "vel_e_m_s",
        "vel_d_m_s",
    ]

    output_columns = [col for col in output_columns if col in ref.columns]

    reference_path = trajectory_dir / "reference_trajectory.csv"
    ref[output_columns].to_csv(reference_path, index=False)

    origin_path = report_dir / "reference_origin.json"
    save_origin_metadata(origin, origin_path)

    summary = {
        "source": "onboard_gps_file",
        "rows_before_filter": int(rows_before_filter),
        "rows_after_filter": int(len(ref)),
        "used_fix_type_filter": used_fix_filter,
        "utm_epsg": epsg,
        "output_reference_trajectory": str(reference_path),
        "output_origin_metadata": str(origin_path),
        "x_range_m": [
            float(ref["x_enu_m"].min()),
            float(ref["x_enu_m"].max()),
        ],
        "y_range_m": [
            float(ref["y_enu_m"].min()),
            float(ref["y_enu_m"].max()),
        ],
        "z_range_m": [
            float(ref["z_enu_m"].min()),
            float(ref["z_enu_m"].max()),
        ],
    }

    summary_path = report_dir / "reference_trajectory_summary.json"

    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    return reference_path, origin_path, summary_path, summary

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to dataset config yaml")
    args = parser.parse_args()

    config = load_config(args.config)

    dataset_type = config.get("dataset_type")

    if dataset_type != "zurich_mav":
        raise ValueError(f"The script currently supports only Zurich MAV Dataset (zurich_mav) only. Got {dataset_type}")
    
    reference_path, origin_path, summary_path, summary = build_reference_trajectory(config)

    print("Reference Trajectory Generated")
    print("------------------------------")
    print(f"Reference CSV: {reference_path}")
    print(f"origin JSON: {origin_path}")
    print(f"Summary JSON: {summary_path}")
    # print(f"Rows: {summary["rows_after_filter"]}")
    print(f"Rows: {summary['rows_after_filter']}")

    print(f"X range [m]: {summary['x_range_m']}")
    print(f"Y range [m]: {summary['y_range_m']}")
    print(f"Z range [m]: {summary['z_range_m']}")

if __name__ == "__main__":
    main()
