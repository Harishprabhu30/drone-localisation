"""CV2.4 SIFT+LightGlue full403 freeze contract tests."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.villoc.geometry.cv2_4_freeze_sift_lightglue_full403 import verify_config


class CV24Tests(unittest.TestCase):
    def setUp(self):
        self.cfg = yaml.safe_load(
            (ROOT / "configs/research/crossview_scale_rotation_domain_cv2_4.yaml").read_text()
        )

    def test_contract(self):
        verify_config(self.cfg)
        self.assertEqual(self.cfg["development"]["expected_queries"], 403)
        self.assertEqual(self.cfg["development"]["expected_pairs"], 8060)
        self.assertTrue(self.cfg["promotion"]["no_retuning_after_uniform40"])

    def test_reference_and_parameter_changes_forbidden(self):
        b = self.cfg["scientific_boundary"]
        self.assertFalse(b["reference_allowed"])
        self.assertFalse(b["scale_m2_applied_to_sift"])
        self.assertFalse(b["frontend_parameters_mutable"])
        self.assertFalse(b["threshold_tuning_allowed"])


if __name__ == "__main__":
    unittest.main()
