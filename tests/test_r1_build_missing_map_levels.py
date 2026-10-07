from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.villoc.retrieval.r1_1_build_missing_map_levels import (
    inspect_existing_state,
)


class R1BuildMissingMapLevelsTests(unittest.TestCase):
    def test_absent_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertEqual(
                inspect_existing_state(
                    tiles_dir=root / "tiles",
                    index_csv=root / "index.csv",
                ),
                "absent",
            )

    def test_empty_directory_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tiles = root / "tiles"
            tiles.mkdir()

            self.assertEqual(
                inspect_existing_state(
                    tiles_dir=tiles,
                    index_csv=root / "index.csv",
                ),
                "empty_dir_only",
            )

    def test_complete_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tiles = root / "tiles"
            tiles.mkdir()
            (tiles / "sat_000001.jpg").write_bytes(b"x")
            index = root / "index.csv"
            index.write_text("tile_id\nsat_000001\n")

            self.assertEqual(
                inspect_existing_state(
                    tiles_dir=tiles,
                    index_csv=index,
                ),
                "complete_candidate",
            )

    def test_partial_state_is_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tiles = root / "tiles"
            tiles.mkdir()
            (tiles / "sat_000001.jpg").write_bytes(b"x")

            self.assertEqual(
                inspect_existing_state(
                    tiles_dir=tiles,
                    index_csv=root / "index.csv",
                ),
                "partial_or_inconsistent",
            )


if __name__ == "__main__":
    unittest.main()
