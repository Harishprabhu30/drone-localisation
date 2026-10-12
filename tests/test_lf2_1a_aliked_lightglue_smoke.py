"""LF2.1-A ALIKED+LightGlue extractor-swap smoke tests."""
from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path

REPO_ROOT=Path(__file__).resolve().parents[1]
SRC_ROOT=REPO_ROOT/"src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0,str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0,str(SRC_ROOT))

from scripts.villoc.geometry.lf2_1a_aliked_lightglue_smoke import verify_config
from uavloc.geometry.aliked_lightglue_absolute_frontend import (
    ALIKED_MODEL_NAME,
    DETECTION_THRESHOLD,
    FEATURE_TYPE,
    MAX_KEYPOINTS,
    NMS_RADIUS,
)
from uavloc.geometry.lightglue_absolute_frontend import (
    LIGHTGLUE_DEPTH_CONFIDENCE,
    LIGHTGLUE_FILTER_THRESHOLD,
    LIGHTGLUE_WIDTH_CONFIDENCE,
    MIN_HOMOGRAPHY_INLIERS,
    PINNED_COMMIT,
    RANSAC_THRESHOLD_PX,
    RESIZE_LONG,
)


class LF21ATests(unittest.TestCase):
    def test_aliked_extractor_swap_contract(self):
        cfg={
            "stage":"LF2.1-A",
            "aliked_lightglue":{
                "pinned_commit":PINNED_COMMIT,
                "feature_type":FEATURE_TYPE,
                "aliked_model_name":ALIKED_MODEL_NAME,
                "max_keypoints":MAX_KEYPOINTS,
                "resize_long":RESIZE_LONG,
                "detection_threshold":DETECTION_THRESHOLD,
                "nms_radius":NMS_RADIUS,
                "filter_threshold":LIGHTGLUE_FILTER_THRESHOLD,
                "depth_confidence":LIGHTGLUE_DEPTH_CONFIDENCE,
                "width_confidence":LIGHTGLUE_WIDTH_CONFIDENCE,
                "ransac_threshold_px":RANSAC_THRESHOLD_PX,
                "min_homography_inliers":MIN_HOMOGRAPHY_INLIERS,
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
        cfg["aliked_lightglue"]["detection_threshold"]=0.1
        with self.assertRaisesRegex(RuntimeError,"detection_threshold"):
            verify_config(cfg)

    def test_script_help_direct_execution(self):
        script=REPO_ROOT/"scripts/villoc/geometry/lf2_1a_aliked_lightglue_smoke.py"
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
        self.assertIn("ALIKED+LightGlue extractor-swap preflight smoke",process.stdout)


if __name__=="__main__":
    unittest.main()
