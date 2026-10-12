"""LF2.0-A SuperPoint+LightGlue absolute smoke tests."""
from __future__ import annotations

import os
import hashlib
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT=Path(__file__).resolve().parents[1]
SRC_ROOT=REPO_ROOT/"src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0,str(SRC_ROOT))

from scripts.villoc.geometry.lf2_0a_lightglue_absolute_preflight_smoke import verify_config
from uavloc.geometry.lightglue_absolute_frontend import (
    git_blob_sha1,
    FEATURE_TYPE,
    LIGHTGLUE_DEPTH_CONFIDENCE,
    LIGHTGLUE_FILTER_THRESHOLD,
    LIGHTGLUE_WIDTH_CONFIDENCE,
    MAX_KEYPOINTS,
    MIN_HOMOGRAPHY_INLIERS,
    PINNED_COMMIT,
    RANSAC_THRESHOLD_PX,
    RESIZE_LONG,
)


class LF20ATests(unittest.TestCase):
    def test_config_contract(self):
        cfg={
            "stage":"LF2.0-A",
            "lightglue":{
                "pinned_commit":PINNED_COMMIT,
                "feature_type":FEATURE_TYPE,
                "max_keypoints":MAX_KEYPOINTS,
                "resize_long":RESIZE_LONG,
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
        cfg["scientific_boundary"]["reference_allowed"]=True
        with self.assertRaisesRegex(RuntimeError,"reference_allowed"):
            verify_config(cfg)

    def test_git_blob_sha1(self):
        payload = b"lightglue-source-check\n"
        expected = hashlib.sha1(
            f"blob {len(payload)}\0".encode("utf-8") + payload
        ).hexdigest()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "source.py"
            path.write_bytes(payload)
            self.assertEqual(git_blob_sha1(path), expected)

    def test_script_help_direct_execution(self):
        script=REPO_ROOT/"scripts/villoc/geometry/lf2_0a_lightglue_absolute_preflight_smoke.py"
        env=os.environ.copy()
        env.pop("PYTHONPATH",None)
        p=subprocess.run(
            [sys.executable,str(script),"--help"],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(p.returncode,0,msg=p.stderr)
        self.assertIn("SuperPoint+LightGlue absolute frontend preflight smoke",p.stdout)


if __name__=="__main__":
    unittest.main()
