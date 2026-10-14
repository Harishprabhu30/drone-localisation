"""CV3.2 compact screen contract and selector tests."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.villoc.geometry.cv3_2_aerial_mast3r_screen import (
    freeze_reference_free_selection,
    verify_config,
)
from scripts.villoc.geometry.cv3_2_prepare_compact_bundle import (
    verify_config as verify_bundle_config,
)


class CV32Tests(unittest.TestCase):
    def setUp(self):
        self.cfg = yaml.safe_load(
            (ROOT / "configs/research/crossview_scale_rotation_domain_cv3_2.yaml").read_text()
        )

    def test_contract(self):
        verify_config(self.cfg)
        verify_bundle_config(self.cfg)
        self.assertEqual(self.cfg["uniform40"]["expected_queries"], 40)
        self.assertEqual(self.cfg["uniform40"]["expected_pairs"], 240)
        self.assertEqual(self.cfg["uniform40"]["pool_ranks"], [1,2,3,5,10,20])
        self.assertFalse(self.cfg["scientific_boundary"]["reference_allowed"])

    def test_selector_inliers_then_rank_then_tile(self):
        frame = pd.DataFrame([
            {"query_id": 1, "pool_rank": 5, "tile_id": "b", "inliers": 10,
             "homography_ok": True, "finite_projection": True, "projected_inside_tile": True,
             "projected_easting": 1.0, "projected_northing": 2.0},
            {"query_id": 1, "pool_rank": 2, "tile_id": "z", "inliers": 10,
             "homography_ok": True, "finite_projection": True, "projected_inside_tile": True,
             "projected_easting": 3.0, "projected_northing": 4.0},
            {"query_id": 1, "pool_rank": 2, "tile_id": "a", "inliers": 10,
             "homography_ok": True, "finite_projection": True, "projected_inside_tile": True,
             "projected_easting": 5.0, "projected_northing": 6.0},
            {"query_id": 1, "pool_rank": 1, "tile_id": "q", "inliers": 9,
             "homography_ok": True, "finite_projection": True, "projected_inside_tile": True,
             "projected_easting": 7.0, "projected_northing": 8.0},
        ])
        selected = freeze_reference_free_selection(frame)
        self.assertEqual(selected.loc[0, "selected_pool_rank"], 2)
        self.assertEqual(selected.loc[0, "selected_tile_id"], "a")
        self.assertEqual(selected.loc[0, "selected_inliers"], 10)
        self.assertEqual(selected.loc[0, "runner_up_inliers"], 10)
        self.assertEqual(selected.loc[0, "inlier_margin"], 0)


if __name__ == "__main__":
    unittest.main()
