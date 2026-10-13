"""CV2 SIFT+LightGlue protocol/smoke contract tests."""
from __future__ import annotations
import sys, unittest
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.villoc.geometry.cv2_1_sift_lightglue_smoke import verify_config

class CV21Tests(unittest.TestCase):
    def setUp(self):
        self.cfg = yaml.safe_load((ROOT/"configs/research/crossview_scale_rotation_domain_cv2.yaml").read_text())

    def test_contract(self):
        verify_config(self.cfg)
        self.assertEqual(self.cfg["lightglue"]["feature_type"], "sift")
        self.assertEqual(self.cfg["uniform40"]["expected_pairs"], 800)

    def test_no_scale_confounded_sift(self):
        self.assertFalse(self.cfg["scientific_boundary"]["scale_m2_applied_to_sift"])
        self.assertFalse(self.cfg["scientific_boundary"]["threshold_tuning_allowed"])

if __name__ == "__main__":
    unittest.main()
