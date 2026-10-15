from __future__ import annotations
import sys, unittest
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.villoc.geometry.fa1_1_sift_targeted_pair_diagnostics import verify_config

class FA11ContractTests(unittest.TestCase):
    def test_contract(self):
        cfg = yaml.safe_load(
            (ROOT / "configs/research/crossview_scale_rotation_domain_fa1_1.yaml").read_text()
        )
        verify_config(cfg)
        self.assertEqual(cfg["fa1_0"]["expected_union_queries"], 20)
        self.assertEqual(cfg["fa1_0"]["expected_oracle_rescuable"], 19)
        self.assertEqual(cfg["fa1_0"]["expected_severe_regression"], 7)
        self.assertTrue(cfg["scientific_boundary"]["targeted_matcher_rerun_allowed"])
        self.assertFalse(cfg["scientific_boundary"]["threshold_tuning_allowed"])
        self.assertFalse(cfg["scientific_boundary"]["selector_change_allowed"])

if __name__ == "__main__":
    unittest.main()
