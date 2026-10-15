from __future__ import annotations
import unittest
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]

class FA13ContractTests(unittest.TestCase):
    def test_contract(self):
        cfg = yaml.safe_load(
            (ROOT / "configs/research/crossview_scale_rotation_domain_fa1_3.yaml").read_text()
        )
        self.assertEqual(cfg["stage"], "FA1.3")
        self.assertEqual(cfg["context"]["temporal_radius_queries"], 2)
        self.assertEqual(cfg["context"]["map_spacing_m"], 51.2)
        self.assertEqual(cfg["context"]["neighborhood_radii_m"], [51.2, 102.4])
        self.assertTrue(cfg["scientific_boundary"]["context_features_reference_free"])
        self.assertFalse(cfg["scientific_boundary"]["threshold_tuning_allowed"])
        self.assertFalse(cfg["scientific_boundary"]["confidence_rule_training_allowed"])
        self.assertFalse(cfg["scientific_boundary"]["matcher_rerun_allowed"])

if __name__ == "__main__":
    unittest.main()
