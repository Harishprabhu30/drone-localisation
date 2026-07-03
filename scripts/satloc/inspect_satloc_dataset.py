from __future__ import annotations

import argparse
from pathlib import Path

from uavloc.data.satloc_loader import SatLocDataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect SatLoc dataset structure.")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to SatLoc dataset YAML config.",
    )
    args = parser.parse_args()

    dataset = SatLocDataset.from_config_path(args.config)
    report = dataset.inspect()

    print("SatLoc dataset summary")
    print("----------------------")
    print(f"Dataset name:        {report['dataset_name']}")
    print(f"Raw data dir:        {report['raw_data_dir']}")
    print(f"Output dir:          {report['output_dir']}")
    print()
    print(f"UAV sequences:       {report['uav']['sequence_count']}")
    print(f"Total UAV images:    {report['uav']['total_uav_images']}")
    print(f"Parsed UAV coords:   {report['uav']['parsed_coordinate_count']}")
    print()
    print(f"Satellite tiles:     {report['satellite']['tile_count']}")
    print(f"Parsed tile coords:  {report['satellite']['parsed_tile_coordinate_count']}")
    print()
    print(f"GeoTIFF exists:      {report['satellite']['geotiff']['exists']}")
    print(f"GeoTIFF width:       {report['satellite']['geotiff'].get('width')}")
    print(f"GeoTIFF height:      {report['satellite']['geotiff'].get('height')}")
    print(f"GeoTIFF CRS:         {report['satellite']['geotiff'].get('crs')}")
    print()
    print(f"Reference CSV rows:  {report['satellite']['ref_csv']['rows']}")
    print(f"Reference columns:   {report['satellite']['ref_csv']['columns']}")
    print()

    if report["warnings"]:
        print("Warnings")
        print("--------")
        for warning in report["warnings"]:
            print(f"- {warning}")
        print()

    report_path = Path(report["output_dir"]) / "reports" / "dataset_report.json"
    print(f"Saved report:        {report_path}")

    uav_index = report["uav"].get("uav_index_csv")
    if uav_index:
        print(f"Saved UAV index:     {uav_index}")

    tile_index = report["satellite"].get("satellite_tiles_index_csv")
    if tile_index:
        print(f"Saved tile index:    {tile_index}")


if __name__ == "__main__":
    main()