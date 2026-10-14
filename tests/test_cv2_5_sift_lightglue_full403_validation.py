"""CV2.5 full403 validation tests."""
from __future__ import annotations
import sys, unittest
from pathlib import Path
import yaml

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from scripts.villoc.geometry.cv2_5_postfreeze_sift_lightglue_full403_validation import evaluate_gate, verify_config

class CV25Tests(unittest.TestCase):
    def setUp(self):
        self.cfg=yaml.safe_load((ROOT/"configs/research/crossview_scale_rotation_domain_cv2_5.yaml").read_text())

    def test_contract(self):
        verify_config(self.cfg)
        self.assertFalse(self.cfg["scientific_boundary"]["jetson_cuda_evidence_used"])

    def test_gate_promotes_meaningful_tail_gain(self):
        scale={"selected_top1":{"contains_query":305,"center_le40":163,"center_le80":293},"selected_projection":{"le40":319,"unsafe_tail":63}}
        sift={"selected_top1":{"contains_query":307,"center_le40":165,"center_le80":294},"selected_projection":{"le40":321,"unsafe_tail":50}}
        r=evaluate_gate(scale,sift,self.cfg["validation_gate_vs_scale_m2"])
        self.assertTrue(r["pass"])

    def test_gate_does_not_promote_trivial_equal_result(self):
        scale={"selected_top1":{"contains_query":305,"center_le40":163,"center_le80":293},"selected_projection":{"le40":319,"unsafe_tail":63}}
        sift={"selected_top1":{"contains_query":306,"center_le40":164,"center_le80":294},"selected_projection":{"le40":320,"unsafe_tail":62}}
        r=evaluate_gate(scale,sift,self.cfg["validation_gate_vs_scale_m2"])
        self.assertFalse(r["pass"])
        self.assertTrue(r["preservation_pass"])

if __name__=="__main__":
    unittest.main()
