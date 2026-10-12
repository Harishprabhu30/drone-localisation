"""LF2.0-AH official maximum-accuracy LightGlue smoke tests."""
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

from scripts.villoc.geometry.lf2_0ah_lightglue_max_accuracy_smoke import verify_config
from uavloc.geometry.lightglue_absolute_frontend import (
    FEATURE_TYPE,
    LIGHTGLUE_FILTER_THRESHOLD,
    MIN_HOMOGRAPHY_INLIERS,
    PINNED_COMMIT,
    RANSAC_THRESHOLD_PX,
    RESIZE_LONG,
)


class LF20AHTests(unittest.TestCase):
    def test_official_max_accuracy_contract(self):
        cfg={
            "stage":"LF2.0-AH",
            "lightglue":{
                "pinned_commit":PINNED_COMMIT,
                "feature_type":FEATURE_TYPE,
                "max_keypoints":None,
                "resize_long":RESIZE_LONG,
                "filter_threshold":LIGHTGLUE_FILTER_THRESHOLD,
                "depth_confidence":-1.0,
                "width_confidence":-1.0,
                "ransac_threshold_px":RANSAC_THRESHOLD_PX,
                "min_homography_inliers":MIN_HOMOGRAPHY_INLIERS,
                "profile":"official_max_accuracy",
            },
            "scientific_boundary":{
                "reference_allowed":False,
                "reference_attachment_enabled":False,
                "oracle_labels_allowed":False,
                "full_top20_benchmark_enabled":False,
                "candidate_reranking_enabled":False,
                "selector_enabled":False,
                "filter_threshold_tuning_enabled":False,
                "learned_model_training_enabled":False,
                "bootstrap_enabled":False,
                "state_enabled":False,
            },
        }
        verify_config(cfg)
        cfg["lightglue"]["depth_confidence"]=0.95
        with self.assertRaisesRegex(RuntimeError,"depth_confidence"):
            verify_config(cfg)

    def test_script_help_direct_execution(self):
        script=(
            REPO_ROOT
            / "scripts/villoc/geometry/lf2_0ah_lightglue_max_accuracy_smoke.py"
        )
        env=os.environ.copy()
        env.pop("PYTHONPATH",None)
        process=subprocess.run(
            [sys.executable,str(script),"--help"],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(process.returncode,0,msg=process.stderr)
        self.assertIn(
            "official maximum-accuracy hard-pair smoke",
            process.stdout,
        )


if __name__=="__main__":
    unittest.main()
