"""RG2.0 map-only unit tests. Fixtures are synthetic, never reference-backed."""
from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from scripts.villoc.geometry.rg2_0_topology import (
    Tile, load_tiles, overlap, topology, validate_topology
)


def tile(name: str, x: float, y: float, row: int, col: int) -> Tile:
    return Tile(name, row, col, x, y, x + 153.6, y + 153.6,
                x + 76.8, y + 76.8)


def index(path: Path, rows: list[Tile]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "tile_id", "grid_row", "grid_col",
            "tile_width_px", "tile_height_px",
            "left_easting", "bottom_northing",
            "right_easting", "top_northing",
            "center_easting", "center_northing",
        ])
        writer.writeheader()
        for t in rows:
            writer.writerow({
                "tile_id": t.tile_id,
                "grid_row": t.grid_row, "grid_col": t.grid_col,
                "tile_width_px": 768, "tile_height_px": 768,
                "left_easting": t.left, "bottom_northing": t.bottom,
                "right_easting": t.right, "top_northing": t.top,
                "center_easting": t.east, "center_northing": t.north,
            })


class RG20TopologyTests(unittest.TestCase):
    def test_overlap_and_shifted_centers(self):
        a = tile("A", 0, 0, 0, 0)
        b = tile("B", 51.2, 0, 0, 1)
        width, height, shared = overlap(a, b)
        self.assertAlmostEqual(width, 102.4)
        self.assertAlmostEqual(height, 153.6)
        self.assertAlmostEqual(shared, 102.4 * 153.6)
        trows, rows, stats = topology([a, b], stride_m=51.2, tolerance_m=0.01)
        self.assertEqual(stats["undirected_overlap_pairs"], 1)
        self.assertEqual(stats["undirected_near_pairs"], 1)
        ab = next(r for r in rows if r["tile_id"] == "A")
        ba = next(r for r in rows if r["tile_id"] == "B")
        self.assertAlmostEqual(ab["delta_easting_m"], 51.2)
        self.assertAlmostEqual(ba["delta_easting_m"], -51.2)
        self.assertAlmostEqual(ab["overlap_fraction_source"], 2 / 3)
        self.assertEqual([r["overlap_degree"] for r in trows], [1, 1])

    def test_distinguish_overlap_from_near(self):
        a = tile("A", 0, 0, 0, 0)
        # Two stride offsets: still overlapping, but not in near-center radius.
        b = tile("B", 102.4, 0, 0, 2)
        _, edges, summary = topology([a, b], stride_m=51.2, tolerance_m=0.01)
        self.assertEqual(summary["undirected_overlap_pairs"], 1)
        self.assertEqual(summary["undirected_near_pairs"], 0)
        self.assertTrue(all(r["overlaps"] and not r["nearby"] for r in edges))

    def test_no_positive_area_at_touching_boundary(self):
        a = tile("A", 0, 0, 0, 0)
        b = tile("B", 153.6, 0, 0, 3)
        self.assertAlmostEqual(overlap(a, b)[2], 0)
        _, edges, summary = topology([a, b], stride_m=51.2, tolerance_m=0.01)
        self.assertEqual(edges, [])
        self.assertEqual(summary["undirected_overlap_pairs"], 0)

    def test_irregular_aoi_edge_and_determinism(self):
        tiles = [
            tile("C", 60.0, 51.2, 1, 1),
            tile("A", 0, 0, 0, 0),
            tile("B", 51.2, 0, 0, 1),
        ]
        first = topology(sorted(tiles, key=lambda t: t.tile_id),
                         stride_m=51.2, tolerance_m=0.01)
        second = topology(sorted(reversed(tiles), key=lambda t: t.tile_id),
                          stride_m=51.2, tolerance_m=0.01)
        self.assertEqual(first, second)
        by_id = {r["tile_id"]: r for r in first[0]}
        self.assertTrue(by_id["A"]["aoi_west_edge"])
        self.assertTrue(by_id["A"]["aoi_south_edge"])
        self.assertTrue(by_id["C"]["aoi_east_edge"])
        self.assertTrue(by_id["C"]["aoi_north_edge"])
        validate_topology(first[1], first[0])

    def test_load_reads_bounds_not_filename_or_csv_row_order(self):
        rows = [tile("z-last", 51.2, 0, 0, 1),
                tile("a-first", 0, 0, 0, 0)]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tiles.csv"
            index(path, rows)
            loaded = load_tiles(path, expected_tile_px=768,
                                expected_footprint_m=153.6, tolerance_m=0.01)
            self.assertEqual([t.tile_id for t in loaded], ["a-first", "z-last"])
            self.assertEqual(loaded[0].left, 0)

    def test_corrupt_center_rejected(self):
        rows = [tile("A", 0, 0, 0, 0)]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tiles.csv"
            index(path, rows)
            content = path.read_text()
            path.write_text(content.replace("76.8,76.8", "100.0,76.8"))
            with self.assertRaises(ValueError):
                load_tiles(path, expected_tile_px=768,
                           expected_footprint_m=153.6, tolerance_m=0.01)

    def test_duplicate_physical_windows_rejected(self):
        rows = [tile("A", 0, 0, 0, 0),
                tile("B", 0, 0, 0, 1)]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tiles.csv"
            index(path, rows)
            with self.assertRaisesRegex(ValueError, "Duplicate physical"):
                load_tiles(path, expected_tile_px=768,
                           expected_footprint_m=153.6, tolerance_m=0.01)


if __name__ == "__main__":
    unittest.main()
