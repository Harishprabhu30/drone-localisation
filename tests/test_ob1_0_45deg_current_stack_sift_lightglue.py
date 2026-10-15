from __future__ import annotations
import sys, unittest
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.villoc.geometry.ob1_0_45deg_current_stack_sift_lightglue import verify_config

class OB10ContractTests(unittest.TestCase):
    def test_contract(self):
        cfg = yaml.safe_load(
            (ROOT / "configs/research/villoc_45deg_sift_lightglue_ob1.yaml").read_text()
        )
        verify_config(cfg)
        self.assertEqual(cfg["source"]["expected_queries"], 112)
        self.assertEqual(cfg["map"]["variant"], "768_s256")
        self.assertEqual(cfg["retrieval"]["image_size"], 518)
        self.assertEqual(cfg["retrieval"]["top_k"], 20)
        self.assertEqual(cfg["retrieval"]["expected_pairs"], 2240)
        self.assertFalse(cfg["scientific_boundary"]["threshold_tuning_allowed"])
        self.assertTrue(cfg["scientific_boundary"]["truth_attached_only_after_freeze"])
        self.assertTrue(cfg["scientific_boundary"]["projected_query_center_vs_body_is_stress_metric"])

if __name__ == "__main__":
    unittest.main()
