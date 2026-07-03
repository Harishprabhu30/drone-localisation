from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import yaml


IMAGE_EXTENSIONS_DEFAULT = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}


def load_yaml_config(config_path: str | Path) -> Dict[str, Any]:
    config_path = Path(config_path)
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_relative_path(path: Path, base: Path) -> str:
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def safe_read_csv(path: Path) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    if not path.exists():
        return None, "file_missing"

    try:
        return pd.read_csv(path), None
    except Exception as first_error:
        try:
            return pd.read_csv(path, engine="python", on_bad_lines="skip"), None
        except Exception as second_error:
            return None, f"csv_read_failed: {first_error}; {second_error}"


def inspect_image_header(path: Path) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "exists": path.exists(),
        "path": str(path),
        "suffix": path.suffix.lower(),
        "size_bytes": path.stat().st_size if path.exists() else None,
        "width": None,
        "height": None,
        "mode": None,
        "driver": None,
        "crs": None,
        "bounds": None,
        "transform": None,
        "rasterio_available": False,
        "pil_available": False,
        "error": None,
    }

    if not path.exists():
        info["error"] = "file_missing"
        return info

    try:
        from PIL import Image

        info["pil_available"] = True
        with Image.open(path) as img:
            info["width"] = int(img.width)
            info["height"] = int(img.height)
            info["mode"] = img.mode
    except Exception as e:
        info["error"] = f"pil_header_read_failed: {e}"

    if path.suffix.lower() in {".tif", ".tiff"}:
        try:
            import rasterio

            info["rasterio_available"] = True
            with rasterio.open(path) as ds:
                info["driver"] = ds.driver
                info["crs"] = str(ds.crs) if ds.crs else None
                info["bounds"] = {
                    "left": ds.bounds.left,
                    "bottom": ds.bounds.bottom,
                    "right": ds.bounds.right,
                    "top": ds.bounds.top,
                }
                info["transform"] = tuple(ds.transform)
                info["count"] = ds.count
                info["dtypes"] = list(ds.dtypes)
        except Exception as e:
            info["rasterio_error"] = str(e)

    return info


def parse_coordinates_from_filename(filename: str) -> Dict[str, Any]:
    stem = Path(filename).stem

    float_tokens = re.findall(r"[-+]?\d+\.\d+", stem)
    values = [float(v) for v in float_tokens]

    result: Dict[str, Any] = {
        "float_values": values,
        "parsed_lon": None,
        "parsed_lat": None,
        "parse_status": "no_decimal_tokens",
        "candidate_pairs": [],
    }

    if len(values) < 2:
        return result

    candidate_pairs = []
    for i in range(len(values) - 1):
        a = values[i]
        b = values[i + 1]

        # Candidate 1: lon, lat
        if -180.0 <= a <= 180.0 and -90.0 <= b <= 90.0:
            candidate_pairs.append(
                {
                    "order": "lon_lat",
                    "lon": a,
                    "lat": b,
                    "token_index": i,
                }
            )

        # Candidate 2: lat, lon
        if -90.0 <= a <= 90.0 and -180.0 <= b <= 180.0:
            candidate_pairs.append(
                {
                    "order": "lat_lon",
                    "lon": b,
                    "lat": a,
                    "token_index": i,
                }
            )

    result["candidate_pairs"] = candidate_pairs

    if not candidate_pairs:
        result["parse_status"] = "decimal_tokens_found_but_no_valid_latlon_pair"
        return result

    # Prefer lon/lat when longitude is outside latitude range, e.g. China longitude > 90.
    strong_lon_lat = [
        c for c in candidate_pairs
        if c["order"] == "lon_lat" and abs(c["lon"]) > 90.0
    ]

    selected = strong_lon_lat[0] if strong_lon_lat else candidate_pairs[0]

    result["parsed_lon"] = selected["lon"]
    result["parsed_lat"] = selected["lat"]
    result["parse_status"] = (
        "parsed_unambiguous" if len(candidate_pairs) == 1 else "parsed_with_candidates"
    )
    return result


@dataclass
class SatLocDataset:
    config: Dict[str, Any]

    @classmethod
    def from_config_path(cls, config_path: str | Path) -> "SatLocDataset":
        return cls(load_yaml_config(config_path))

    @property
    def dataset_name(self) -> str:
        return self.config.get("dataset", {}).get("name", "satloc")

    @property
    def raw_data_dir(self) -> Path:
        return Path(self.config["paths"]["raw_data_dir"])

    @property
    def output_dir(self) -> Path:
        return Path(self.config["paths"].get("output_dir", f"outputs/{self.dataset_name}"))

    @property
    def metadata_dir(self) -> Path:
        return self.output_dir / "metadata"

    @property
    def reports_dir(self) -> Path:
        return self.output_dir / "reports"

    def resolve_under_raw(self, relative_path: str) -> Path:
        return self.raw_data_dir / relative_path

    def image_extensions(self) -> set[str]:
        exts = self.config.get("satloc", {}).get("image_extensions")
        if not exts:
            return IMAGE_EXTENSIONS_DEFAULT
        return {str(e).lower() for e in exts}

    def list_images(self, root: Path) -> List[Path]:
        if not root.exists():
            return []

        exts = self.image_extensions()
        files = [
            p for p in root.rglob("*")
            if p.is_file() and p.suffix.lower() in exts and p.name != ".DS_Store"
        ]
        return sorted(files)

    def inspect(self) -> Dict[str, Any]:
        ensure_dir(self.metadata_dir)
        ensure_dir(self.reports_dir)

        paths_cfg = self.config["paths"]

        uav_dir = self.resolve_under_raw(paths_cfg["uav_data_dir"])
        sat_dir = self.resolve_under_raw(paths_cfg["satellite_data_dir"])
        geotiff_path = self.resolve_under_raw(paths_cfg["satellite_geotiff"])
        ref_csv_path = self.resolve_under_raw(paths_cfg["satellite_ref_csv"])
        tile_dir = self.resolve_under_raw(paths_cfg["satellite_tile_dir"])

        sequence_glob = self.config.get("satloc", {}).get("uav_sequence_glob", "traj*")
        sequence_dirs = sorted([p for p in uav_dir.glob(sequence_glob) if p.is_dir()])

        uav_records: List[Dict[str, Any]] = []
        sequence_summaries: List[Dict[str, Any]] = []

        for sequence_dir in sequence_dirs:
            image_paths = self.list_images(sequence_dir)

            sequence_summary = {
                "sequence": sequence_dir.name,
                "sequence_dir": str(sequence_dir),
                "image_count": len(image_paths),
                "first_image": str(image_paths[0]) if image_paths else None,
                "last_image": str(image_paths[-1]) if image_paths else None,
            }
            sequence_summaries.append(sequence_summary)

            for idx, image_path in enumerate(image_paths):
                coord = parse_coordinates_from_filename(image_path.name)
                uav_records.append(
                    {
                        "dataset": self.dataset_name,
                        "sequence": sequence_dir.name,
                        "frame_index_in_sequence": idx,
                        "image_path": str(image_path),
                        "image_path_relative": safe_relative_path(image_path, self.raw_data_dir),
                        "filename": image_path.name,
                        "suffix": image_path.suffix.lower(),
                        "parsed_lon": coord["parsed_lon"],
                        "parsed_lat": coord["parsed_lat"],
                        "parse_status": coord["parse_status"],
                        "float_values_in_filename": json.dumps(coord["float_values"]),
                        "candidate_pairs": json.dumps(coord["candidate_pairs"]),
                    }
                )

        uav_index_df = pd.DataFrame(uav_records)
        uav_index_path = self.metadata_dir / "uav_frames_index.csv"
        if not uav_index_df.empty:
            uav_index_df.to_csv(uav_index_path, index=False)

        satellite_tile_paths = self.list_images(tile_dir)
        sat_tile_records = []
        for idx, tile_path in enumerate(satellite_tile_paths):
            coord = parse_coordinates_from_filename(tile_path.name)
            sat_tile_records.append(
                {
                    "tile_index": idx,
                    "tile_path": str(tile_path),
                    "tile_path_relative": safe_relative_path(tile_path, self.raw_data_dir),
                    "filename": tile_path.name,
                    "suffix": tile_path.suffix.lower(),
                    "parsed_lon": coord["parsed_lon"],
                    "parsed_lat": coord["parsed_lat"],
                    "parse_status": coord["parse_status"],
                    "float_values_in_filename": json.dumps(coord["float_values"]),
                    "candidate_pairs": json.dumps(coord["candidate_pairs"]),
                }
            )

        sat_tiles_df = pd.DataFrame(sat_tile_records)
        sat_tiles_index_path = self.metadata_dir / "satellite_tiles_index.csv"
        if not sat_tiles_df.empty:
            sat_tiles_df.to_csv(sat_tiles_index_path, index=False)

        ref_df, ref_error = safe_read_csv(ref_csv_path)

        ref_csv_summary: Dict[str, Any] = {
            "path": str(ref_csv_path),
            "exists": ref_csv_path.exists(),
            "read_error": ref_error,
            "rows": int(len(ref_df)) if ref_df is not None else 0,
            "columns": list(ref_df.columns) if ref_df is not None else [],
            "head": ref_df.head(5).to_dict(orient="records") if ref_df is not None else [],
        }

        warnings: List[str] = []

        if not self.raw_data_dir.exists():
            warnings.append(f"Raw data dir missing: {self.raw_data_dir}")

        if not sequence_dirs:
            warnings.append(f"No UAV sequence dirs found under: {uav_dir}")

        if uav_index_df.empty:
            warnings.append("No UAV images found. Increase find depth or check image extensions.")

        if not geotiff_path.exists():
            warnings.append(f"GeoTIFF missing: {geotiff_path}")

        if not ref_csv_path.exists():
            warnings.append(f"Reference CSV missing: {ref_csv_path}")

        if not tile_dir.exists():
            warnings.append(f"Satellite tile directory missing: {tile_dir}")

        parsed_uav = 0
        if not uav_index_df.empty and "parsed_lon" in uav_index_df.columns:
            parsed_uav = int(uav_index_df["parsed_lon"].notna().sum())

        parsed_tiles = 0
        if not sat_tiles_df.empty and "parsed_lon" in sat_tiles_df.columns:
            parsed_tiles = int(sat_tiles_df["parsed_lon"].notna().sum())

        report: Dict[str, Any] = {
            "dataset_name": self.dataset_name,
            "dataset_type": self.config.get("dataset", {}).get("type", "satloc"),
            "raw_data_dir": str(self.raw_data_dir),
            "output_dir": str(self.output_dir),
            "paths": {
                "uav_data_dir": str(uav_dir),
                "satellite_data_dir": str(sat_dir),
                "satellite_geotiff": str(geotiff_path),
                "satellite_ref_csv": str(ref_csv_path),
                "satellite_tile_dir": str(tile_dir),
            },
            "uav": {
                "sequence_count": len(sequence_dirs),
                "total_uav_images": len(uav_records),
                "parsed_coordinate_count": parsed_uav,
                "sequences": sequence_summaries,
                "uav_index_csv": str(uav_index_path) if not uav_index_df.empty else None,
                "sample_records": uav_index_df.head(10).to_dict(orient="records")
                if not uav_index_df.empty
                else [],
            },
            "satellite": {
                "geotiff": inspect_image_header(geotiff_path),
                "ref_csv": ref_csv_summary,
                "tile_count": len(satellite_tile_paths),
                "parsed_tile_coordinate_count": parsed_tiles,
                "satellite_tiles_index_csv": str(sat_tiles_index_path)
                if not sat_tiles_df.empty
                else None,
                "sample_tiles": sat_tiles_df.head(10).to_dict(orient="records")
                if not sat_tiles_df.empty
                else [],
            },
            "warnings": warnings,
        }

        report_path = self.reports_dir / "dataset_report.json"
        with report_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        return report