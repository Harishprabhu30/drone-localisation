"""CV1.3 full403 scale freeze contract tests."""
from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.villoc.geometry.cv1_3_freeze_full403_scale_m2_evidence import (
    verify_config,
)


class CV13FreezeTests(unittest.TestCase):
    def setUp(self):
        path = REPO_ROOT / "configs/research/crossview_scale_rotation_domain_cv1_3.yaml"
        self.cfg = yaml.safe_load(path.read_text(encoding="utf-8"))

    def test_promoted_scale_is_frozen(self):
        self.assertEqual(self.cfg["promotion"]["promoted_condition"], "scale_m2")
        self.assertTrue(
            math.isclose(
                float(self.cfg["promotion"]["scale_factor"]),
                0.6944444444444444,
                rel_tol=0.0,
                abs_tol=1e-15,
            )
        )
        self.assertTrue(self.cfg["promotion"]["no_retuning_after_uniform40"])

    def test_config_contract_passes(self):
        verify_config(self.cfg)

    def test_reference_and_srt_are_forbidden(self):
        boundary = self.cfg["scientific_boundary"]
        self.assertFalse(boundary["reference_allowed"])
        self.assertFalse(boundary["srt_allowed"])
        self.assertFalse(boundary["transform_retuning_allowed"])


if __name__ == "__main__":
    unittest.main()
