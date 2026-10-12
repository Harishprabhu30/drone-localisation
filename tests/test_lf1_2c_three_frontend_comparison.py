"""LF1.2-C three-frontend comparison tests."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT=Path(__file__).resolve().parents[1]
SRC_ROOT=REPO_ROOT/"src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0,str(SRC_ROOT))

from scripts.villoc.geometry.lf1_2c_postfreeze_three_frontend_comparison import (
    verify_config,
)


class LF12CTests(unittest.TestCase):
    def test_config_boundary(self):
        cfg={
            "stage":"LF1.2-C",
            "labels":["contains_query","center_le40","center_le80"],
            "ranking":{
                "selectors":[
                    "retrieval_rank",
                    "inliers",
                    "valid_then_inliers",
                ]
            },
            "scientific_boundary":{
                "development_reference_allowed":True,
                "reference_after_all_freezes_only":True,
                "blind_stress_loaded":False,
                "blind_stress_reference_allowed":False,
                "evidence_mutable":False,
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


if __name__=="__main__":
    unittest.main()
