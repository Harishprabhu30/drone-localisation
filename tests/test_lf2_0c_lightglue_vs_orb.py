"""LF2.0-C LightGlue-vs-ORB screening audit tests."""
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

from scripts.villoc.geometry.lf2_0c_postfreeze_lightglue_uniform40_vs_orb import (
    evaluate_primary_gate,
    verify_config,
)


class LF20CTests(unittest.TestCase):
    def test_primary_gate_pass(self):
        cfg = {
            "primary_full_run_gate": {
                "max_top1_regression_queries_each_target": 2,
                "min_top1_improvement_queries_any_target": 2,
                "max_selected_projection_le40_regression_queries": 2,
                "max_selected_projection_gt200_increase_queries": 2,
            }
        }
        orb = {
            "top1": {"contains_query": 25, "center_le40": 14, "center_le80": 24},
            "projection": {"le40": 28, "gt200": 5},
        }
        lg = {
            "top1": {"contains_query": 27, "center_le40": 14, "center_le80": 23},
            "projection": {"le40": 27, "gt200": 6},
        }
        result = evaluate_primary_gate(cfg, orb, lg)
        self.assertTrue(result["pass"])
        self.assertEqual(result["decision"], "PROCEED_FULL_LIGHTGLUE")

    def test_primary_gate_fails_large_regression(self):
        cfg = {
            "primary_full_run_gate": {
                "max_top1_regression_queries_each_target": 2,
                "min_top1_improvement_queries_any_target": 2,
                "max_selected_projection_le40_regression_queries": 2,
                "max_selected_projection_gt200_increase_queries": 2,
            }
        }
        orb = {
            "top1": {"contains_query": 25, "center_le40": 14, "center_le80": 24},
            "projection": {"le40": 28, "gt200": 5},
        }
        lg = {
            "top1": {"contains_query": 28, "center_le40": 10, "center_le80": 25},
            "projection": {"le40": 27, "gt200": 6},
        }
        result = evaluate_primary_gate(cfg, orb, lg)
        self.assertFalse(result["pass"])

    def test_config_boundary(self):
        cfg = {
            "stage": "LF2.0-C",
            "labels": ["contains_query", "center_le40", "center_le80"],
            "ranking": {
                "orb_primary_selector": "inliers",
                "lightglue_primary_selector": "inliers",
                "lightglue_secondary_scores": [
                    "evidence_match_score_sum",
                    "evidence_match_score_mean",
                    "evidence_match_score_median",
                    "evidence_inlier_ratio",
                ],
            },
            "scientific_boundary": {
                "development_reference_allowed": True,
                "reference_after_screening_freeze_only": True,
                "blind_stress_loaded": False,
                "blind_stress_reference_allowed": False,
                "full_403_lightglue_loaded": False,
                "threshold_tuning_enabled": False,
                "selector_training_enabled": False,
                "frontend_parameters_mutable": False,
                "bootstrap_enabled": False,
                "state_enabled": False,
            },
        }
        verify_config(cfg)
        cfg["scientific_boundary"]["threshold_tuning_enabled"] = True
        with self.assertRaisesRegex(RuntimeError, "threshold_tuning_enabled"):
            verify_config(cfg)

    def test_script_help_direct_execution(self):
        script = REPO_ROOT / "scripts/villoc/geometry/lf2_0c_postfreeze_lightglue_uniform40_vs_orb.py"
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
        self.assertIn("LightGlue uniform40 vs ORB", process.stdout)


if __name__ == "__main__":
    unittest.main()
