"""CV2.3 post-freeze comparison tests."""
from __future__ import annotations
import sys, unittest
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.villoc.geometry.cv2_3_postfreeze_sift_lightglue_comparison import gate, verify_config

class CV23Tests(unittest.TestCase):
    def setUp(self):
        self.cfg = yaml.safe_load((ROOT/"configs/research/crossview_scale_rotation_domain_cv2_3.yaml").read_text())

    def test_contract(self):
        verify_config(self.cfg)
        self.assertFalse(self.cfg["scientific_boundary"]["full403_allowed"])

    def test_gate_requires_gain_and_small_regressions(self):
        scale = {
            "selected_top1": {"contains_query": 32, "center_le40": 15, "center_le80": 31},
            "selected_projection": {"le40": 32, "gt200": 6},
        }
        sift = {
            "selected_top1": {"contains_query": 34, "center_le40": 14, "center_le80": 31},
            "selected_projection": {"le40": 31, "gt200": 7},
        }
        self.assertTrue(gate(scale, sift, self.cfg["continuation_gate_vs_scale_m2"])["pass"])

    def test_gate_fails_large_top1_regression(self):
        scale = {
            "selected_top1": {"contains_query": 32, "center_le40": 15, "center_le80": 31},
            "selected_projection": {"le40": 32, "gt200": 6},
        }
        sift = {
            "selected_top1": {"contains_query": 29, "center_le40": 17, "center_le80": 31},
            "selected_projection": {"le40": 32, "gt200": 6},
        }
        self.assertFalse(gate(scale, sift, self.cfg["continuation_gate_vs_scale_m2"])["pass"])

if __name__ == "__main__":
    unittest.main()
