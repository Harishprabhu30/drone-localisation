"""LF2.0-B LightGlue uniform40 screening tests."""
from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from scripts.villoc.geometry.lf2_0b_freeze_lightglue_uniform40 import (
    uniform_query_selection,
    verify_config,
)
from uavloc.geometry.lightglue_absolute_frontend import (
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


class LF20BTests(unittest.TestCase):
    def test_uniform_query_selection_is_deterministic_and_includes_endpoints(self):
        query_ids = list(range(1, 404))
        indices, selected = uniform_query_selection(query_ids, 40)
        self.assertEqual(len(indices), 40)
        self.assertEqual(len(selected), 40)
        self.assertEqual(len(set(indices)), 40)
        self.assertEqual(indices[0], 0)
        self.assertEqual(indices[-1], 402)
        self.assertEqual(selected[0], 1)
        self.assertEqual(selected[-1], 403)
        indices2, selected2 = uniform_query_selection(query_ids, 40)
        self.assertEqual(indices, indices2)
        self.assertEqual(selected, selected2)

    def test_default_adaptive_profile_contract(self):
        cfg = {
            "stage": "LF2.0-B",
            "lightglue": {
                "pinned_commit": PINNED_COMMIT,
                "feature_type": FEATURE_TYPE,
                "max_keypoints": MAX_KEYPOINTS,
                "resize_long": RESIZE_LONG,
                "filter_threshold": LIGHTGLUE_FILTER_THRESHOLD,
                "depth_confidence": LIGHTGLUE_DEPTH_CONFIDENCE,
                "width_confidence": LIGHTGLUE_WIDTH_CONFIDENCE,
                "ransac_threshold_px": RANSAC_THRESHOLD_PX,
                "min_homography_inliers": MIN_HOMOGRAPHY_INLIERS,
                "profile": "default_adaptive",
            },
            "screening": {
                "selection_method": "uniform_index_over_sorted_query_ids_v1",
                "query_count": 40,
                "candidates_per_query": 20,
                "expected_pairs": 800,
                "include_endpoints": True,
            },
            "generation": {
                "pair_order": "query_id_then_pool_rank",
                "runtime_fields_frozen": False,
                "retain_failed_pairs": True,
            },
            "scientific_boundary": {
                "reference_allowed": False,
                "reference_attachment_enabled": False,
                "oracle_labels_allowed": False,
                "full_403_benchmark_enabled": False,
                "candidate_reranking_enabled": False,
                "selector_enabled": False,
                "threshold_tuning_enabled": False,
                "model_training_enabled": False,
                "temporal_fusion_enabled": False,
                "bootstrap_enabled": False,
                "state_enabled": False,
            },
        }
        verify_config(cfg)
        cfg["lightglue"]["depth_confidence"] = -1.0
        with self.assertRaisesRegex(RuntimeError, "depth_confidence"):
            verify_config(cfg)

    def test_script_help_direct_execution(self):
        script = REPO_ROOT / "scripts/villoc/geometry/lf2_0b_freeze_lightglue_uniform40.py"
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        process = subprocess.run(
            [sys.executable, str(script), "--help"],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(process.returncode, 0, msg=process.stderr)
        self.assertIn("uniform40 Top20 screening freeze", process.stdout)


if __name__ == "__main__":
    unittest.main()
