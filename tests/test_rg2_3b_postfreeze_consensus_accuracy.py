"""RG2.3-B post-freeze consensus accuracy audit tests."""
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

from scripts.villoc.geometry.rg2_3b_postfreeze_consensus_accuracy import (
    paired,
    rank_correlation,
    summary,
    verify_config,
)


class RG23BTests(unittest.TestCase):
    def test_summary_metrics(self):
        s=summary(pd.Series([1.,2.,3.,4.]),[2.,4.])
        self.assertEqual(s["count"],4)
        self.assertAlmostEqual(s["median_m"],2.5)
        self.assertEqual(s["threshold_hits"]["le2m"],2)
        self.assertEqual(s["threshold_hits"]["le4m"],4)

    def test_paired_direction(self):
        p=paired(pd.Series([1.,5.,3.]),pd.Series([2.,4.,3.]))
        self.assertEqual(p["improved_count"],1)
        self.assertEqual(p["worsened_count"],1)
        self.assertEqual(p["tie_count"],1)

    def test_rank_correlation(self):
        r=rank_correlation(pd.Series([1.,2.,3.]),pd.Series([2.,4.,6.]))
        self.assertAlmostEqual(r["pearson"],1.0)
        self.assertAlmostEqual(r["spearman"],1.0)

    def test_config_forbids_method_selection_and_blind_reference(self):
        cfg={
            "stage":"RG2.3-B",
            "map_variant":"768_s256",
            "map_crs":"EPSG:3346",
            "audit_contract":{
                "deployment_method_selection_allowed":False,
                "reference_tuning_allowed":False,
                "primary_projection_view":"all_finite",
                "methods":[
                    "coordinate_median","geometric_median","medoid",
                    "radius_cluster_medoid","inlier_huber_centroid",
                ],
            },
            "scientific_boundary":{
                "development_reference_allowed":True,
                "reference_after_consensus_freeze_only":True,
                "blind_stress_loaded":False,
                "blind_stress_reference_allowed":False,
                "consensus_coordinates_mutable":False,
                "orb_rerun_enabled":False,
                "projection_rerun_enabled":False,
                "consensus_rerun_enabled":False,
                "method_tuning_enabled":False,
                "bootstrap_enabled":False,
                "state_enabled":False,
            },
        }
        verify_config(cfg)
        cfg["scientific_boundary"]["blind_stress_loaded"]=True
        with self.assertRaisesRegex(RuntimeError,"blind_stress_loaded"):
            verify_config(cfg)

    def test_summary_rejects_nonfinite(self):
        with self.assertRaisesRegex(RuntimeError,"finite"):
            summary(pd.Series([1.,np.nan]),[40.])

    def test_script_help_without_pythonpath(self):
        script=REPO_ROOT/"scripts/villoc/geometry/rg2_3b_postfreeze_consensus_accuracy.py"
        env=os.environ.copy(); env.pop("PYTHONPATH",None)
        p=subprocess.run(
            [sys.executable,str(script),"--help"],
            cwd=REPO_ROOT,env=env,capture_output=True,text=True,timeout=30
        )
        self.assertEqual(p.returncode,0,msg=p.stderr)
        self.assertIn("post-freeze projection consensus accuracy",p.stdout)


if __name__=="__main__":
    unittest.main()
