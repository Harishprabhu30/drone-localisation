"""CV3.1A environment/source preflight contract tests."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.villoc.geometry.cv3_1a_aerial_mast3r_environment_preflight import (
    verify_config,
)


class CV31APreflightTests(unittest.TestCase):
    def setUp(self):
        self.cfg = yaml.safe_load(
            (ROOT / "configs/research/crossview_scale_rotation_domain_cv3_1a.yaml").read_text()
        )

    def test_frozen_environment_contract(self):
        verify_config(self.cfg)
        self.assertEqual(self.cfg["protocol"]["image_load_size"], 512)
        self.assertEqual(self.cfg["protocol"]["dtype"], "float32")
        self.assertTrue(self.cfg["protocol"]["require_cuda"])

    def test_no_download_or_inference_in_preflight(self):
        self.assertFalse(self.cfg["protocol"]["allow_model_download"])
        self.assertFalse(self.cfg["protocol"]["allow_dataset_download"])
        self.assertFalse(self.cfg["scientific_boundary"]["checkpoint_download_allowed"])
        self.assertFalse(self.cfg["scientific_boundary"]["model_inference_allowed"])


if __name__ == "__main__":
    unittest.main()
