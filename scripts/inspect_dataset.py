import argparse
from pathlib import Path

from uavloc.utils.config import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to dataset config YAML")
    args = parser.parse_args()

    config = load_config(args.config)

    dataset_name = config["dataset_name"]
    raw_data_dir = Path(config["paths"]["raw_data_dir"])
    output_dir = Path(config["paths"]["output_dir"])

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Dataset name: {dataset_name}")
    print(f"Raw data dir: {raw_data_dir}")
    print(f"Output dir: {output_dir}")

    if not raw_data_dir.exists():
        print(f"WARNING: Raw data directory does not exist yet: {raw_data_dir}")
    else:
        files = list(raw_data_dir.rglob("*"))
        print(f"Found {len(files)} files.")


if __name__ == "__main__":
    main()