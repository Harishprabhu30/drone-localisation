from __future__ import annotations
import sys, unittest
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.villoc.geometry.bv1_2_blind_sift_lightglue_minimum_confident import verify_config

class BV12ContractTests(unittest.TestCase):
    def test_contract(self):
        cfg = yaml.safe_load(
            (ROOT / "configs/research/blind_demo_sift_lightglue_transfer_bv1_2.yaml").read_text()
        )
        verify_config(cfg)
        self.assertEqual(cfg["policy"]["candidate_count"], 4)
        self.assertEqual(cfg["policy"]["evidence_queries"], 4)
        self.assertEqual(cfg["policy"]["activation_threshold_m"], 12.8)
        self.assertEqual(cfg["policy"]["tracking_threshold_m"], 12.8)
        self.assertEqual(cfg["policy"]["ranking"]["primary"], "inliers_desc")
        self.assertFalse(cfg["scientific_boundary"]["orb_recompute_allowed"])
        self.assertFalse(cfg["scientific_boundary"]["reference_allowed"])

if __name__ == "__main__":
    unittest.main()
