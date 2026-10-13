"""LF2.1-B ALIKED+LightGlue uniform40 screening tests."""
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

from scripts.villoc.geometry.lf2_1b_freeze_aliked_lightglue_uniform40 import (
    uniform_query_selection,
    verify_config,
)
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

EXPECTED_QUERY_IDS=[
    1,11,22,32,42,53,63,73,83,94,
    104,114,125,135,145,156,166,176,187,197,
    207,217,228,238,248,259,269,279,290,300,
    310,321,331,341,351,362,372,382,393,403,
]


class LF21BTests(unittest.TestCase):
    def test_uniform40_matches_frozen_lf20b_subset(self):
        _, selected=uniform_query_selection(list(range(1,404)),40)
        self.assertEqual(selected,EXPECTED_QUERY_IDS)

    def test_aliked_screening_contract(self):
        cfg={
            "stage":"LF2.1-B",
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
            "screening":{
                "selection_method":"reuse_lf2_0b_uniform40_v1",
                "query_count":40,
                "candidates_per_query":20,
                "expected_pairs":800,
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
                "full_403_benchmark_enabled":False,
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
        cfg["aliked_lightglue"]["max_keypoints"]=1024
        with self.assertRaisesRegex(RuntimeError,"max_keypoints"):
            verify_config(cfg)

    def test_script_help_direct_execution(self):
        script=REPO_ROOT/"scripts/villoc/geometry/lf2_1b_freeze_aliked_lightglue_uniform40.py"
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
        self.assertIn("uniform40 Top20 screening freeze",process.stdout)


if __name__=="__main__":
    unittest.main()
