"""CV1.4 full403 post-freeze validation tests."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.villoc.geometry.cv1_4_postfreeze_full403_scale_validation import (
    evaluate_replication_gate,
)


class CV14ValidationTests(unittest.TestCase):
    def test_replication_gate_passes_required_tail_reduction(self):
        baseline = {
            "selected_projection": {"gt200": 88, "le40": 293},
            "selected_top1": {"contains_query": 292, "center_le80": 287},
        }
        scale = {
            "selected_projection": {"gt200": 70, "le40": 290},
            "selected_top1": {"contains_query": 290, "center_le80": 284},
        }
        cfg = {
            "replication_gate": {
                "min_selected_projection_gt200_reduction": 10,
                "max_selected_projection_le40_loss": 5,
                "max_top1_containment_loss": 5,
                "max_top1_center_le80_loss": 5,
            }
        }
        self.assertTrue(evaluate_replication_gate(baseline, scale, cfg)["pass"])

    def test_replication_gate_fails_tail_requirement(self):
        baseline = {
            "selected_projection": {"gt200": 88, "le40": 293},
            "selected_top1": {"contains_query": 292, "center_le80": 287},
        }
        scale = {
            "selected_projection": {"gt200": 80, "le40": 295},
            "selected_top1": {"contains_query": 295, "center_le80": 290},
        }
        cfg = {
            "replication_gate": {
                "min_selected_projection_gt200_reduction": 10,
                "max_selected_projection_le40_loss": 5,
                "max_top1_containment_loss": 5,
                "max_top1_center_le80_loss": 5,
            }
        }
        self.assertFalse(evaluate_replication_gate(baseline, scale, cfg)["pass"])


if __name__ == "__main__":
    unittest.main()
