from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dataset_satloc.yaml")
    args = parser.parse_args()

    config_path = Path(args.config)
    cfg = load_yaml(config_path)

    print("\nS4A.0 SatLoc retrieval pre-flight")
    print("--------------------------------")

    print(f"Config: {config_path}")
    print("\nTop-level config keys:")
    for k in cfg.keys():
        print(f"  - {k}")

    uav_index = Path("outputs/satloc/metadata/uav_frames_index_enriched.csv")
    sat_index = Path("outputs/satloc/metadata/satellite_tiles_index_enriched.csv")

    print("\nExpected index files:")
    print(f"  UAV index: {uav_index} | exists={uav_index.exists()}")
    print(f"  SAT index: {sat_index} | exists={sat_index.exists()}")

    if not uav_index.exists() or not sat_index.exists():
        raise FileNotFoundError(
            "Missing index file. Run build_satloc_coordinate_index.py first."
        )

    uav_df = pd.read_csv(uav_index)
    sat_df = pd.read_csv(sat_index)

    print("\nUAV index:")
    print(f"  rows: {len(uav_df)}")
    print("  columns:")
    for c in uav_df.columns:
        print(f"    - {c}")

    print("\nSatellite index:")
    print(f"  rows: {len(sat_df)}")
    print("  columns:")
    for c in sat_df.columns:
        print(f"    - {c}")

    print("\nUAV sample rows:")
    print(uav_df.head(3).to_string(index=False))

    print("\nSatellite sample rows:")
    print(sat_df.head(3).to_string(index=False))

    print("\nPre-flight complete.")


if __name__ == "__main__":
    main()