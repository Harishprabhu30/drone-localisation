"""RG2.4-B post-freeze persistence correctness audit tests."""
from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT=Path(__file__).resolve().parents[1]
SRC_ROOT=REPO_ROOT/"src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0,str(SRC_ROOT))

from scripts.villoc.geometry.rg2_4b_postfreeze_persistence_correctness import (
    corr,
    error_summary,
    verify_config,
)


class RG24BTests(unittest.TestCase):
    def test_error_summary(self):
        s=error_summary(pd.Series([10.,20.,50.,100.]))
        self.assertEqual(s["count"],4)
        self.assertEqual(s["le40"],2)
        self.assertEqual(s["le80"],3)
        self.assertAlmostEqual(s["median_m"],35.0)

    def test_corr_monotonic(self):
        r=corr(pd.Series([1.,2.,3.,4.]),pd.Series([10.,20.,30.,40.]))
        self.assertAlmostEqual(r["pearson"],1.0)
        self.assertAlmostEqual(r["spearman"],1.0)

    def test_corr_constant_returns_none(self):
        r=corr(pd.Series([1.,1.,1.]),pd.Series([1.,2.,3.]))
        self.assertIsNone(r["pearson"])
        self.assertIsNone(r["spearman"])

    def test_config_forbids_blind_and_tuning(self):
        cfg={
            "stage":"RG2.4-B","map_variant":"768_s256","map_crs":"EPSG:3346",
            "audit_contract":{
                "deployment_rule_selection_allowed":False,
                "threshold_tuning_allowed":False,
            },
            "scientific_boundary":{
                "development_reference_allowed":True,
                "reference_after_rg2_4a_freeze_only":True,
                "blind_stress_loaded":False,
                "blind_stress_reference_allowed":False,
                "locality_ids_mutable":False,
                "threshold_tuning_enabled":False,
                "method_selection_enabled":False,
                "bootstrap_enabled":False,
                "state_enabled":False,
                "relative_frontend_fusion_enabled":False,
            },
        }
        verify_config(cfg)
        cfg["scientific_boundary"]["blind_stress_loaded"]=True
        with self.assertRaisesRegex(RuntimeError,"blind_stress_loaded"):
            verify_config(cfg)

    def test_nonfinite_error_summary_rejected(self):
        with self.assertRaisesRegex(RuntimeError,"Finite"):
            error_summary(pd.Series([1.,np.nan]))

    def test_script_help_without_pythonpath(self):
        script=REPO_ROOT/"scripts/villoc/geometry/rg2_4b_postfreeze_persistence_correctness.py"
        env=os.environ.copy(); env.pop("PYTHONPATH",None)
        p=subprocess.run(
            [sys.executable,str(script),"--help"],
            cwd=REPO_ROOT,env=env,capture_output=True,text=True,timeout=30
        )
        self.assertEqual(p.returncode,0,msg=p.stderr)
        self.assertIn("persistence correctness audit",p.stdout)


if __name__=="__main__":
    unittest.main()
