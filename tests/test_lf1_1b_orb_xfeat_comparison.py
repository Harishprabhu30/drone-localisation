"""LF1.1-B ORB vs XFeat post-freeze comparison tests."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from scripts.villoc.geometry.lf1_1b_postfreeze_orb_xfeat_comparison import (
    bool_series,
    homography_retention,
    projection_summary,
    ranking_summary,
    sort_candidates,
    verify_config,
)


class LF11BTests(unittest.TestCase):
    def test_bool_series(self):
        s = pd.Series(["true", "false", "1", "0", "yes", "no"])
        self.assertEqual(
            bool_series(s).tolist(),
            [True, False, True, False, True, False],
        )

    def test_valid_then_inliers_prefers_valid(self):
        f = pd.DataFrame({
            "query_id": [1, 1, 1],
            "pool_rank": [1, 2, 3],
            "tile_id": ["a", "b", "c"],
            "homography_ok": [False, True, True],
            "inliers": [100, 5, 7],
        })
        ordered = sort_candidates(f, "valid_then_inliers")
        self.assertEqual(ordered.iloc[0]["tile_id"], "c")
        self.assertEqual(ordered.iloc[1]["tile_id"], "b")
        self.assertEqual(ordered.iloc[2]["tile_id"], "a")

    def test_ranking_summary(self):
        f = pd.DataFrame({
            "query_id": [1, 1, 2, 2],
            "pool_rank": [1, 2, 1, 2],
            "tile_id": ["a", "b", "c", "d"],
            "homography_ok": [True, True, True, True],
            "inliers": [1, 5, 9, 2],
            "hit": [False, True, True, False],
        })
        r = ranking_summary(f, "hit", "inliers", [1, 2])
        self.assertEqual(r["hits"], {"top1": 2, "top2": 2})

    def test_homography_retention(self):
        f = pd.DataFrame({
            "query_id": [1, 1, 2, 2],
            "homography_ok": [True, False, True, False],
            "hit": [True, False, False, True],
        })
        r = homography_retention(f, "hit")
        self.assertEqual(r["positive_pairs"], 2)
        self.assertEqual(r["positive_valid_pairs"], 1)
        self.assertAlmostEqual(r["positive_pair_retention"], 0.5)
        self.assertEqual(r["negative_pairs"], 2)
        self.assertEqual(r["negative_rejected_pairs"], 1)
        self.assertAlmostEqual(r["negative_pair_rejection"], 0.5)
        self.assertEqual(r["queries_with_positive_original"], 2)
        self.assertEqual(r["queries_with_positive_after_homography"], 1)

    def test_projection_summary(self):
        r = projection_summary(pd.Series([10.0, 20.0, 50.0, 250.0, np.nan]))
        self.assertEqual(r["count"], 4)
        self.assertAlmostEqual(r["median_m"], 35.0)
        self.assertEqual(r["le40"], 2)
        self.assertEqual(r["le80"], 3)
        self.assertEqual(r["gt200"], 1)

    def test_config_boundary(self):
        cfg = {
            "stage": "LF1.1-B",
            "labels": ["contains_query", "center_le40", "center_le80"],
            "ranking": {
                "selectors": [
                    "retrieval_rank",
                    "inliers",
                    "valid_then_inliers",
                ]
            },
            "scientific_boundary": {
                "development_reference_allowed": True,
                "reference_after_both_freezes_only": True,
                "blind_stress_loaded": False,
                "blind_stress_reference_allowed": False,
                "evidence_mutable": False,
                "threshold_tuning_enabled": False,
                "selector_training_enabled": False,
                "frontend_parameters_mutable": False,
                "bootstrap_enabled": False,
                "state_enabled": False,
            },
        }
        verify_config(cfg)
        cfg["scientific_boundary"]["frontend_parameters_mutable"] = True
        with self.assertRaisesRegex(RuntimeError, "frontend_parameters_mutable"):
            verify_config(cfg)


if __name__ == "__main__":
    unittest.main()
