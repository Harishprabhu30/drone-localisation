"""LF1.0 XFeat absolute frontend preflight tests."""
from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT=Path(__file__).resolve().parents[1]
SRC_ROOT=REPO_ROOT/"src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0,str(SRC_ROOT))

from uavloc.geometry.xfeat_absolute_frontend import (
    PINNED_COMMIT,
    resize_long_side,
    tile_pixel_to_map,
)
from scripts.villoc.geometry.lf1_0_xfeat_absolute_preflight_smoke import (
    verify_config,
)


class LF10Tests(unittest.TestCase):
    def test_resize_long_side(self):
        image=np.zeros((400,800),dtype=np.uint8)
        out=resize_long_side(image,960)
        self.assertEqual(out.shape,(480,960))

    def test_tile_pixel_to_map_center(self):
        e,n=tile_pixel_to_map(
            49.5,49.5,(100,100),(0.0,100.0,0.0,100.0)
        )
        self.assertAlmostEqual(e,50.0)
        self.assertAlmostEqual(n,50.0)

    def test_config_contract(self):
        cfg={
            "stage":"LF1.0",
            "xfeat":{
                "pinned_commit":PINNED_COMMIT,
                "resize_long":960,
                "top_k":1200,
                "detection_threshold":0.05,
                "min_cossim":0.82,
                "ransac_threshold_px":5.0,
                "min_homography_inliers":4,
            },
            "scientific_boundary":{
                "reference_allowed":False,
                "reference_attachment_enabled":False,
                "oracle_labels_allowed":False,
                "full_top20_benchmark_enabled":False,
                "candidate_reranking_enabled":False,
                "selector_enabled":False,
                "threshold_tuning_enabled":False,
                "learned_model_training_enabled":False,
                "bootstrap_enabled":False,
                "state_enabled":False,
            },
        }
        verify_config(cfg)
        cfg["scientific_boundary"]["reference_allowed"]=True
        with self.assertRaisesRegex(RuntimeError,"reference_allowed"):
            verify_config(cfg)

    def test_script_help_without_xfeat_import(self):
        script=REPO_ROOT/"scripts/villoc/geometry/lf1_0_xfeat_absolute_preflight_smoke.py"
        env=os.environ.copy(); env.pop("PYTHONPATH",None)
        p=subprocess.run(
            [sys.executable,str(script),"--help"],
            cwd=REPO_ROOT,env=env,capture_output=True,text=True,timeout=30
        )
        self.assertEqual(p.returncode,0,msg=p.stderr)
        self.assertIn("XFeat absolute frontend preflight",p.stdout)


if __name__=="__main__":
    unittest.main()
