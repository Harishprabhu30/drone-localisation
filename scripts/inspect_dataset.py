import argparse
from pathlib import Path

from uavloc.utils.config import load_config
from uavloc.data.zurich_loader import ZurichMAVDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to dataset config YAML")
    args = parser.parse_args()

    config = load_config(args.config)

    dataset_name = config["dataset_name"]
    dataset_type = config.get("dataset_type")
    raw_data_dir = Path(config["paths"]["raw_data_dir"])
    output_dir = Path(config["paths"]["output_dir"])

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Dataset name: {dataset_name}")
    print(f"Dataset type: {dataset_type}")
    print(f"Raw data dir: {raw_data_dir}")
    print(f"Output dir: {output_dir}")

    if not raw_data_dir.exists():
        print(f"WARNING: Raw data directory does not exist yet: {raw_data_dir}")
        return

    files = list(raw_data_dir.rglob("*"))
    print(f"Found {len(files)} files.")

    if dataset_type == "zurich_mav":
        dataset = ZurichMAVDataset(config)
        report = dataset.create_report()
        report_path = dataset.save_report(report)

        print("\nZurich MAV dataset summary")
        print("--------------------------")

        image_folders = report.get("image_folders", {})
        for name, summary in image_folders.items():
            print(f"{name}: {summary.get('image_count', 0)} images")

        log_files = report.get("log_files", {})
        for name, summary in log_files.items():
            print(f"{name}: {summary.get('rows', 0)} rows")

        calibration = report.get("calibration", {})
        print(f"Calibration available: {calibration.get('available')}")

        print(f"\nSaved dataset report to: {report_path}")

    else:
        print(f"No dataset-specific loader implemented for dataset_type={dataset_type}")


if __name__ == "__main__":
    main()