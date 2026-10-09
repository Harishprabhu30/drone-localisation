"""RG2.0 deterministic map-window topology. Map metadata only: no query, GT, ORB or state."""
from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

REQUIRED = {
    "tile_id", "grid_row", "grid_col", "tile_width_px", "tile_height_px",
    "left_easting", "bottom_northing", "right_easting", "top_northing",
    "center_easting", "center_northing",
}


@dataclass(frozen=True)
class Tile:
    tile_id: str
    grid_row: int
    grid_col: int
    left: float
    bottom: float
    right: float
    top: float
    east: float
    north: float

    @property
    def area(self) -> float:
        return (self.right - self.left) * (self.top - self.bottom)


def load_tiles(
    path: Path,
    *,
    expected_tile_px: int,
    expected_footprint_m: float,
    tolerance_m: float,
) -> list[Tile]:
    """Fail closed on corrupt/ambiguous tile geometry; use no filename ordering."""
    tiles: list[Tile] = []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = sorted(REQUIRED - set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"Tile index missing required columns: {missing}")
        for line_no, row in enumerate(reader, start=2):
            try:
                tile_id = row["tile_id"].strip()
                grid_row = int(row["grid_row"])
                grid_col = int(row["grid_col"])
                width_px = int(row["tile_width_px"])
                height_px = int(row["tile_height_px"])
                left = float(row["left_easting"])
                right = float(row["right_easting"])
                bottom = float(row["bottom_northing"])
                top = float(row["top_northing"])
                east = float(row["center_easting"])
                north = float(row["center_northing"])
            except (TypeError, ValueError, KeyError) as exc:
                raise ValueError(f"Invalid tile-index row {line_no}: {exc}") from exc

            nums = (left, right, bottom, top, east, north)
            if not tile_id or not all(math.isfinite(v) for v in nums):
                raise ValueError(f"Empty ID or non-finite geometry at row {line_no}")
            if width_px != expected_tile_px or height_px != expected_tile_px:
                raise ValueError(f"Wrong tile dimensions at {tile_id}")
            if right <= left or top <= bottom:
                raise ValueError(f"Non-positive tile bounds at {tile_id}")
            if (abs((right - left) - expected_footprint_m) > tolerance_m or
                    abs((top - bottom) - expected_footprint_m) > tolerance_m):
                raise ValueError(f"Tile footprint differs from frozen map: {tile_id}")
            if (abs(east - (left + right) / 2) > tolerance_m or
                    abs(north - (bottom + top) / 2) > tolerance_m):
                raise ValueError(f"Tile center disagrees with bounds: {tile_id}")
            tiles.append(Tile(tile_id, grid_row, grid_col,
                              left, bottom, right, top, east, north))
    if not tiles:
        raise ValueError("Empty tile index")
    tiles.sort(key=lambda tile: tile.tile_id)
    ids = [tile.tile_id for tile in tiles]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate tile IDs")
    # Duplicate windows would create spurious support in later consensus.
    windows = [(round(t.left, 6), round(t.bottom, 6),
                round(t.right, 6), round(t.top, 6)) for t in tiles]
    if len(windows) != len(set(windows)):
        raise ValueError("Duplicate physical tile windows")
    return tiles


def overlap(a: Tile, b: Tile) -> tuple[float, float, float]:
    width = max(0.0, min(a.right, b.right) - max(a.left, b.left))
    height = max(0.0, min(a.top, b.top) - max(a.bottom, b.bottom))
    return width, height, width * height


def topology(
    tiles: list[Tile], *, stride_m: float, tolerance_m: float
) -> tuple[list[dict], list[dict], dict]:
    """Directed relationships; overlap and proximity are separate predicates."""
    xmin = min(t.left for t in tiles)
    xmax = max(t.right for t in tiles)
    ymin = min(t.bottom for t in tiles)
    ymax = max(t.top for t in tiles)
    near_radius = math.sqrt(2.0) * stride_m
    pair_rows: list[dict] = []
    overlap_degree = {t.tile_id: 0 for t in tiles}
    near_degree = {t.tile_id: 0 for t in tiles}
    for a, b in combinations(tiles, 2):
        dx = b.east - a.east
        dy = b.north - a.north
        distance = math.hypot(dx, dy)
        shared_w, shared_h, shared_area = overlap(a, b)
        overlaps = shared_w > tolerance_m and shared_h > tolerance_m
        nearby = distance <= near_radius + tolerance_m
        if not (overlaps or nearby):
            continue
        for src, dst, sx, sy in ((a, b, dx, dy), (b, a, -dx, -dy)):
            pair_rows.append({
                "tile_id": src.tile_id,
                "neighbor_tile_id": dst.tile_id,
                "delta_easting_m": sx,
                "delta_northing_m": sy,
                "center_distance_m": distance,
                "overlaps": overlaps,
                "nearby": nearby,
                "overlap_width_m": shared_w if overlaps else 0.0,
                "overlap_height_m": shared_h if overlaps else 0.0,
                "shared_area_m2": shared_area if overlaps else 0.0,
                "overlap_fraction_source": shared_area / src.area if overlaps else 0.0,
                "overlap_fraction_neighbor": shared_area / dst.area if overlaps else 0.0,
                "overlap_iou": shared_area / (src.area + dst.area - shared_area)
                                if overlaps else 0.0,
            })
            if overlaps:
                overlap_degree[src.tile_id] += 1
            if nearby:
                near_degree[src.tile_id] += 1
    pair_rows.sort(key=lambda r: (r["tile_id"], r["neighbor_tile_id"]))

    tile_rows = []
    for t in tiles:
        edge_w = abs(t.left - xmin) <= tolerance_m
        edge_e = abs(t.right - xmax) <= tolerance_m
        edge_s = abs(t.bottom - ymin) <= tolerance_m
        edge_n = abs(t.top - ymax) <= tolerance_m
        tile_rows.append({
            "tile_id": t.tile_id, "grid_row_metadata": t.grid_row,
            "grid_col_metadata": t.grid_col,
            "left_easting": t.left, "bottom_northing": t.bottom,
            "right_easting": t.right, "top_northing": t.top,
            "center_easting": t.east, "center_northing": t.north,
            "footprint_area_m2": t.area,
            "aoi_west_edge": edge_w, "aoi_east_edge": edge_e,
            "aoi_south_edge": edge_s, "aoi_north_edge": edge_n,
            "overlap_degree": overlap_degree[t.tile_id],
            "near_degree": near_degree[t.tile_id],
        })
    summary = {
        "tile_count": len(tiles),
        "directed_relationships": len(pair_rows),
        "undirected_overlap_pairs": sum(r["overlaps"] for r in pair_rows) // 2,
        "undirected_near_pairs": sum(r["nearby"] for r in pair_rows) // 2,
        "aoi_bounds_epsg3346": {"west": xmin, "east": xmax,
                                "south": ymin, "north": ymax},
        "aoi_edge_tile_counts": {
            key: sum(bool(r[key]) for r in tile_rows)
            for key in ("aoi_west_edge", "aoi_east_edge",
                        "aoi_south_edge", "aoi_north_edge")
        },
        "overlap_degree_histogram": {
            str(i): sum(d == i for d in overlap_degree.values())
            for i in sorted(set(overlap_degree.values()))
        },
        "near_degree_histogram": {
            str(i): sum(d == i for d in near_degree.values())
            for i in sorted(set(near_degree.values()))
        },
        "isolated_overlap_tiles": sorted(
            tile_id for tile_id, d in overlap_degree.items() if d == 0
        ),
        "near_radius_m": near_radius,
    }
    validate_topology(pair_rows, tile_rows)
    return tile_rows, pair_rows, summary


def validate_topology(pairs: list[dict], tiles: list[dict]) -> None:
    """Directed symmetry and degree accounting, no data-dependent tuning."""
    by_pair = {(r["tile_id"], r["neighbor_tile_id"]): r for r in pairs}
    if len(by_pair) != len(pairs):
        raise ValueError("Duplicate directed relationship")
    for (src, dst), row in by_pair.items():
        rev = by_pair.get((dst, src))
        if rev is None:
            raise ValueError(f"Asymmetric neighbor edge {src}->{dst}")
        for key in ("overlaps", "nearby"):
            if row[key] != rev[key]:
                raise ValueError(f"Asymmetric {key}: {src}, {dst}")
        for key in ("shared_area_m2", "overlap_iou", "center_distance_m"):
            if not math.isclose(row[key], rev[key], abs_tol=1e-7):
                raise ValueError(f"Asymmetric {key}: {src}, {dst}")
        for key in ("delta_easting_m", "delta_northing_m"):
            if not math.isclose(row[key], -rev[key], abs_tol=1e-7):
                raise ValueError(f"Asymmetric offset: {src}, {dst}")
        if not math.isclose(row["overlap_fraction_source"],
                            rev["overlap_fraction_neighbor"], abs_tol=1e-7):
            raise ValueError(f"Asymmetric overlap fraction: {src}, {dst}")
    for tile in tiles:
        outgoing = [r for r in pairs if r["tile_id"] == tile["tile_id"]]
        if sum(r["overlaps"] for r in outgoing) != tile["overlap_degree"]:
            raise ValueError("Overlap degree mismatch")
        if sum(r["nearby"] for r in outgoing) != tile["near_degree"]:
            raise ValueError("Near degree mismatch")
