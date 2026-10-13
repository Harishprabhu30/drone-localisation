"""CV1.2 post-freeze evaluator unit tests."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.villoc.geometry.cv1_2_postfreeze_scale_rotation_evaluation import (
    evaluate_gate,
    rank_candidates,
)


class CV12EvaluatorTests(unittest.TestCase):
    def test_rank_candidates_uses_inliers_then_pool_rank_then_tile(self):
        frame = pd.DataFrame([
            {"inliers": 10, "pool_rank": 3, "tile_id": "b"},
            {"inliers": 12, "pool_rank": 4, "tile_id": "c"},
            {"inliers": 12, "pool_rank": 2, "tile_id": "z"},
            {"inliers": 12, "pool_rank": 2, "tile_id": "a"},
        ])
        ordered = rank_candidates(frame)
        self.assertEqual(ordered.iloc[0]["tile_id"], "a")
        self.assertEqual(ordered.iloc[1]["tile_id"], "z")
        self.assertEqual(ordered.iloc[2]["tile_id"], "c")

    def test_gate_accepts_containment_gain(self):
        baseline = {
            "selected_top1": {"contains_query": 20, "center_le80": 20},
            "selected_projection": {"gt200": 10, "le40": 20},
        }
        condition = {
            "selected_top1": {"contains_query": 24, "center_le80": 20},
            "selected_projection": {"gt200": 10, "le40": 20},
        }
        gate = {
            "containment_gain_min": 4,
            "le80_gain_min": 4,
            "catastrophic_gt200_reduction_min": 3,
            "max_le40_projection_loss_if_using_catastrophic_gate": 2,
        }
        result = evaluate_gate(baseline, condition, gate)
        self.assertTrue(result["pass"])
        self.assertTrue(result["containment_gate_pass"])

    def test_catastrophic_gate_limits_le40_damage(self):
        baseline = {
            "selected_top1": {"contains_query": 20, "center_le80": 20},
            "selected_projection": {"gt200": 10, "le40": 20},
        }
        condition = {
            "selected_top1": {"contains_query": 20, "center_le80": 20},
            "selected_projection": {"gt200": 7, "le40": 17},
        }
        gate = {
            "containment_gain_min": 4,
            "le80_gain_min": 4,
            "catastrophic_gt200_reduction_min": 3,
            "max_le40_projection_loss_if_using_catastrophic_gate": 2,
        }
        result = evaluate_gate(baseline, condition, gate)
        self.assertFalse(result["catastrophic_tail_gate_pass"])
        self.assertFalse(result["pass"])


if __name__ == "__main__":
    unittest.main()
