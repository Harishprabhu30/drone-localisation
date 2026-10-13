"""CV2.2 SIFT+LightGlue uniform40 freeze tests."""
from __future__ import annotations
import sys, unittest
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.villoc.geometry.cv2_2_freeze_sift_lightglue_uniform40 import verify_config

class CV22Tests(unittest.TestCase):
    def setUp(self):
        self.cfg = yaml.safe_load((ROOT/"configs/research/crossview_scale_rotation_domain_cv2_2.yaml").read_text())

    def test_contract(self):
        verify_config(self.cfg)
        self.assertEqual(self.cfg["generation"]["expected_pairs"], 800)
        self.assertEqual(self.cfg["generation"]["expected_queries"], 40)

    def test_no_reference_or_scale_confounded_sift(self):
        self.assertFalse(self.cfg["scientific_boundary"]["reference_allowed"])
        self.assertFalse(self.cfg["scientific_boundary"]["scale_m2_applied_to_sift"])
        self.assertFalse(self.cfg["scientific_boundary"]["full403_allowed"])

if __name__ == "__main__":
    unittest.main()
