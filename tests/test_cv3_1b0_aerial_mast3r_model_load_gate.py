"""CV3.1B0 checkpoint/model-load gate contract tests."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.villoc.geometry.cv3_1b0_aerial_mast3r_model_load_gate import (
    verify_config,
)


class CV31B0Tests(unittest.TestCase):
    def setUp(self):
        self.cfg = yaml.safe_load(
            (ROOT / "configs/research/crossview_scale_rotation_domain_cv3_1b0.yaml").read_text()
        )

    def test_checkpoint_and_fp32_contract(self):
        verify_config(self.cfg)
        self.assertEqual(
            self.cfg["checkpoint"]["expected_model_sha256"],
            "093c3b4b3ad8294673774aae106309d5f6a1a975976bae574075fb7c5f94a67d",
        )
        self.assertEqual(self.cfg["model"]["dtype"], "float32")
        self.assertEqual(self.cfg["model"]["device"], "cuda")

    def test_no_inference_or_dataset(self):
        self.assertFalse(self.cfg["model"]["inference_allowed"])
        self.assertFalse(self.cfg["scientific_boundary"]["image_loading_allowed"])
        self.assertFalse(self.cfg["scientific_boundary"]["pair_inference_allowed"])
        self.assertFalse(self.cfg["scientific_boundary"]["aerialmegadepth_dataset_allowed"])


if __name__ == "__main__":
    unittest.main()
