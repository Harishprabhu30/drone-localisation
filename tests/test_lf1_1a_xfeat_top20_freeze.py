"""LF1.1-A full XFeat Top20 freeze tests."""
from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT=Path(__file__).resolve().parents[1]
SRC_ROOT=REPO_ROOT/"src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0,str(SRC_ROOT))

from scripts.villoc.geometry.lf1_1a_freeze_xfeat_top20 import (
    bool_value,
    finite_median,
    verify_config,
)
from uavloc.geometry.xfeat_absolute_frontend import PINNED_COMMIT


class LF11ATests(unittest.TestCase):
    def test_bool_value(self):
        self.assertTrue(bool_value("true"))
        self.assertTrue(bool_value(1))
        self.assertFalse(bool_value("false"))

    def test_finite_median(self):
        s=pd.Series([1.0,2.0,np.nan,np.inf,3.0])
        self.assertAlmostEqual(finite_median(s),2.0)

    def test_config_contract(self):
        cfg={
            "stage":"LF1.1-A",
            "xfeat":{
                "pinned_commit":PINNED_COMMIT,
                "resize_long":960,
                "top_k":1200,
                "detection_threshold":0.05,
                "min_cossim":0.82,
                "ransac_threshold_px":5.0,
                "min_homography_inliers":4,
            },
            "generation":{
                "pair_order":"query_id_then_pool_rank",
                "runtime_fields_frozen":False,
                "retain_failed_pairs":True,
            },
            "scientific_boundary":{
                "reference_allowed":False,
                "reference_attachment_enabled":False,
                "oracle_labels_allowed":False,
                "candidate_reranking_enabled":False,
                "selector_enabled":False,
                "threshold_tuning_enabled":False,
                "model_training_enabled":False,
                "temporal_fusion_enabled":False,
                "bootstrap_enabled":False,
                "state_enabled":False,
            },
        }
        verify_config(cfg)
        cfg["generation"]["runtime_fields_frozen"]=True
        with self.assertRaisesRegex(RuntimeError,"Runtime fields"):
            verify_config(cfg)

    def test_script_help_without_loading_xfeat_weights(self):
        script=REPO_ROOT/"scripts/villoc/geometry/lf1_1a_freeze_xfeat_top20.py"
        env=os.environ.copy(); env.pop("PYTHONPATH",None)
        p=subprocess.run(
            [sys.executable,str(script),"--help"],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(p.returncode,0,msg=p.stderr)
        self.assertIn("full reference-free XFeat Top20",p.stdout)


if __name__=="__main__":
    unittest.main()
