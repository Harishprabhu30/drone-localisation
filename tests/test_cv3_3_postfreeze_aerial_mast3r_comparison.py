"""CV3.3 post-freeze compact truth-evaluation contract tests."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.villoc.geometry.cv3_3_postfreeze_aerial_mast3r_comparison import (
    evaluate_gate,
    recompute_selection,
    verify_config,
)


class CV33Tests(unittest.TestCase):
    def setUp(self):
        self.cfg = yaml.safe_load(
            (ROOT / "configs/research/crossview_scale_rotation_domain_cv3_3.yaml").read_text()
        )

    def test_contract(self):
        verify_config(self.cfg)
        self.assertEqual(self.cfg["candidate_slice"]["pool_ranks"], [1,2,3,5,10,20])
        self.assertFalse(self.cfg["scientific_boundary"]["threshold_tuning_allowed"])
        self.assertFalse(self.cfg["scientific_boundary"]["full_top20_allowed"])

    def test_recompute_selection_contract(self):
        frame = pd.DataFrame([
            {"query_id":1,"pool_rank":1,"tile_id":"z","inliers":5,
             "homography_ok":True,"finite_projection":True,"projected_inside_tile":True,
             "projected_easting":1.0,"projected_northing":2.0},
            {"query_id":1,"pool_rank":2,"tile_id":"b","inliers":8,
             "homography_ok":True,"finite_projection":True,"projected_inside_tile":True,
             "projected_easting":3.0,"projected_northing":4.0},
            {"query_id":1,"pool_rank":2,"tile_id":"a","inliers":8,
             "homography_ok":True,"finite_projection":True,"projected_inside_tile":True,
             "projected_easting":5.0,"projected_northing":6.0},
        ])
        selected = recompute_selection(frame)
        self.assertEqual(selected.loc[0,"selected_pool_rank"],2)
        self.assertEqual(selected.loc[0,"selected_tile_id"],"a")
        self.assertEqual(selected.loc[0,"selected_inliers"],8)

    def test_gate_or_logic(self):
        sift = {
            "selected_top1":{"contains_query":20,"center_le40":20,"center_le80":25},
            "selected_projection":{"le40":30,"unsafe_tail":8},
        }
        mast_containment = {
            "selected_top1":{"contains_query":23,"center_le40":18,"center_le80":24},
            "selected_projection":{"le40":20,"unsafe_tail":9},
        }
        result = evaluate_gate(
            sift, mast_containment,
            self.cfg["continuation_gate_vs_sift_six_candidate"],
        )
        self.assertTrue(result["containment_gate_pass"])
        self.assertTrue(result["pass"])

        mast_unsafe = {
            "selected_top1":{"contains_query":20,"center_le40":20,"center_le80":25},
            "selected_projection":{"le40":28,"unsafe_tail":5},
        }
        result = evaluate_gate(
            sift, mast_unsafe,
            self.cfg["continuation_gate_vs_sift_six_candidate"],
        )
        self.assertTrue(result["unsafe_tail_gate_pass"])
        self.assertTrue(result["pass"])

        mast_unsafe_too_much_le40_loss = {
            "selected_top1":{"contains_query":20,"center_le40":20,"center_le80":25},
            "selected_projection":{"le40":27,"unsafe_tail":5},
        }
        result = evaluate_gate(
            sift, mast_unsafe_too_much_le40_loss,
            self.cfg["continuation_gate_vs_sift_six_candidate"],
        )
        self.assertFalse(result["unsafe_tail_gate_pass"])
        self.assertFalse(result["pass"])


if __name__ == "__main__":
    unittest.main()
