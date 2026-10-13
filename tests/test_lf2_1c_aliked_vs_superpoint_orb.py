"""LF2.1-C ALIKED+LightGlue vs SuperPoint+LightGlue vs ORB audit tests."""
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

from scripts.villoc.geometry.lf2_1c_postfreeze_aliked_vs_superpoint_orb import (
    evaluate_primary_gate,
    verify_config,
)


class LF21CTests(unittest.TestCase):
    def test_primary_gate_pass(self):
        gate={
            "max_top1_regression_queries_each_target":2,
            "min_top1_improvement_queries_any_target":2,
            "max_selected_projection_le40_regression_queries":2,
            "max_selected_projection_gt200_increase_queries":2,
        }
        orb={"contains_query":25,"center_le40":14,"center_le80":24}
        candidate={"contains_query":27,"center_le40":14,"center_le80":23}
        orb_proj={"le40":28,"gt200":5}
        cand_proj={"le40":27,"gt200":6}
        result=evaluate_primary_gate(
            orb,candidate,orb_proj,cand_proj,gate
        )
        self.assertTrue(result["pass"])

    def test_primary_gate_fail(self):
        gate={
            "max_top1_regression_queries_each_target":2,
            "min_top1_improvement_queries_any_target":2,
            "max_selected_projection_le40_regression_queries":2,
            "max_selected_projection_gt200_increase_queries":2,
        }
        orb={"contains_query":25,"center_le40":14,"center_le80":24}
        candidate={"contains_query":20,"center_le40":16,"center_le80":20}
        orb_proj={"le40":28,"gt200":5}
        cand_proj={"le40":25,"gt200":8}
        result=evaluate_primary_gate(
            orb,candidate,orb_proj,cand_proj,gate
        )
        self.assertFalse(result["pass"])

    def test_config_boundary(self):
        cfg={
            "stage":"LF2.1-C",
            "labels":["contains_query","center_le40","center_le80"],
            "ranking":{
                "primary_selector":"inliers",
                "aliked_secondary_scores":[
                    "evidence_match_score_sum",
                    "evidence_match_score_mean",
                    "evidence_match_score_median",
                    "evidence_inlier_ratio",
                ],
            },
            "primary_full_run_gate":{
                "max_top1_regression_queries_each_target":2,
                "min_top1_improvement_queries_any_target":2,
                "max_selected_projection_le40_regression_queries":2,
                "max_selected_projection_gt200_increase_queries":2,
            },
            "secondary_formulation_gate":{
                "min_top1_gain_over_aliked_inliers_on_two_targets":4,
                "max_top1_gap_to_orb_each_target":2,
            },
            "scientific_boundary":{
                "development_reference_allowed":True,
                "reference_after_all_screening_freezes_only":True,
                "blind_stress_loaded":False,
                "blind_stress_reference_allowed":False,
                "full_403_aliked_lightglue_loaded":False,
                "threshold_tuning_enabled":False,
                "selector_training_enabled":False,
                "frontend_parameters_mutable":False,
                "bootstrap_enabled":False,
                "state_enabled":False,
            },
        }
        verify_config(cfg)
        cfg["scientific_boundary"]["frontend_parameters_mutable"]=True
        with self.assertRaisesRegex(RuntimeError,"frontend_parameters_mutable"):
            verify_config(cfg)

    def test_script_help_direct_execution(self):
        script=REPO_ROOT/"scripts/villoc/geometry/lf2_1c_postfreeze_aliked_vs_superpoint_orb.py"
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
            "ALIKED+LightGlue vs SuperPoint+LightGlue vs ORB uniform40 audit",
            process.stdout,
        )


if __name__=="__main__":
    unittest.main()
