"""LF1.2-A XFeat* semi-dense smoke tests."""
from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path

REPO_ROOT=Path(__file__).resolve().parents[1]
SRC_ROOT=REPO_ROOT/"src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0,str(SRC_ROOT))

from scripts.villoc.geometry.lf1_2a_xfeat_star_preflight_smoke import verify_config
from uavloc.geometry.xfeat_absolute_frontend import PINNED_COMMIT
from uavloc.geometry.xfeat_star_absolute_frontend import (
    FINE_CONFIDENCE,
    MULTISCALE,
)


class LF12ATests(unittest.TestCase):
    def test_config_contract(self):
        cfg={
            "stage":"LF1.2-A",
            "xfeat_star":{
                "pinned_commit":PINNED_COMMIT,
                "resize_long":960,
                "top_k":1200,
                "multiscale":MULTISCALE,
                "fine_confidence":FINE_CONFIDENCE,
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

    def test_script_help_without_weights(self):
        script=REPO_ROOT/"scripts/villoc/geometry/lf1_2a_xfeat_star_preflight_smoke.py"
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
        self.assertIn("XFeat* semi-dense",p.stdout)


if __name__=="__main__":
    unittest.main()
