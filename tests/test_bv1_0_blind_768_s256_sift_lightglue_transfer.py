from __future__ import annotations
import sys, unittest
from pathlib import Path
import numpy as np, pandas as pd, yaml
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from scripts.villoc.geometry.bv1_0_blind_768_s256_sift_lightglue_transfer import verify_config, verify_cache_pair, select_winners

class BV10Tests(unittest.TestCase):
    def setUp(self):
        self.cfg=yaml.safe_load((ROOT/"configs/research/blind_demo_sift_lightglue_transfer_bv1.yaml").read_text())
    def test_contract(self):
        verify_config(self.cfg)
        self.assertEqual(self.cfg["map"]["variant"],"768_s256")
        self.assertTrue(self.cfg["retrieval"]["rerun_against_768_map_cache"])
        self.assertFalse(self.cfg["scientific_boundary"]["reference_allowed"])
        self.assertFalse(self.cfg["scientific_boundary"]["bootstrap_executed"])
    def test_selector(self):
        f=pd.DataFrame([
          {"query_id":"1","rank":3,"tile_id":"b","score":.7,"matches":10,"inliers":8,"homography_ok":True,"finite_projection":True,"projected_inside_tile":True,"projected_easting":1.,"projected_northing":2.},
          {"query_id":"1","rank":2,"tile_id":"z","score":.8,"matches":10,"inliers":8,"homography_ok":True,"finite_projection":True,"projected_inside_tile":True,"projected_easting":3.,"projected_northing":4.},
          {"query_id":"1","rank":2,"tile_id":"a","score":.8,"matches":10,"inliers":8,"homography_ok":True,"finite_projection":True,"projected_inside_tile":True,"projected_easting":5.,"projected_northing":6.},
          {"query_id":"1","rank":1,"tile_id":"q","score":.9,"matches":10,"inliers":7,"homography_ok":True,"finite_projection":True,"projected_inside_tile":True,"projected_easting":7.,"projected_northing":8.},
        ])
        s=select_winners(f)
        self.assertEqual(s.loc[0,"selected_dino_rank"],2); self.assertEqual(s.loc[0,"selected_tile_id"],"a")
    def test_cache_protocol_mismatch(self):
        p={"model_name":"dinov2_vits14","image_size":518,"crop_mode":"center_square","pooling":"avgpatch","normalization":"imagenet","l2_normalize":True,"descriptor_dtype":"float32"}
        a={"descriptors":np.array([[1.,0.]],np.float32),"ids":np.array(["1"]),"meta":{"protocol":p}}
        b={"descriptors":np.array([[1.,0.]],np.float32),"ids":np.array(["x"]),"meta":{"protocol":dict(p,image_size=224)}}
        with self.assertRaises(RuntimeError): verify_cache_pair(a,b)
if __name__=="__main__": unittest.main()
