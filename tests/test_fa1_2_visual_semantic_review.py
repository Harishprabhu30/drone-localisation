from __future__ import annotations
import sys, unittest
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.villoc.geometry.fa1_2_visual_semantic_review import verify_config

class FA12ContractTests(unittest.TestCase):
    def test_contract(self):
        cfg = yaml.safe_load(
            (ROOT / "configs/research/crossview_scale_rotation_domain_fa1_2.yaml").read_text()
        )
        verify_config(cfg)
        self.assertEqual(cfg["fa1_0"]["expected_union_queries"], 20)
        self.assertEqual(cfg["fa1_1"]["expected_sift_pairs"], 40)
        self.assertFalse(cfg["scientific_boundary"]["matcher_rerun_allowed"])
        self.assertFalse(cfg["scientific_boundary"]["automated_semantic_labeling_allowed"])

if __name__ == "__main__":
    unittest.main()
