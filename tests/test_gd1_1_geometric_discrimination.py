"""GD1.1 post-freeze geometric-discrimination tests."""
from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path

import pandas as pd

REPO_ROOT=Path(__file__).resolve().parents[1]
SRC_ROOT=REPO_ROOT/"src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0,str(SRC_ROOT))

from scripts.villoc.geometry.gd1_1_postfreeze_geometric_discrimination import (
    binary_auc,
    correlation,
    retrieval_baseline,
    verify_config,
)


class GD11Tests(unittest.TestCase):
    def test_auc_direction(self):
        labels=pd.Series([False,False,True,True])
        scores=pd.Series([1.,2.,3.,4.])
        r=binary_auc(labels,scores)
        self.assertAlmostEqual(r["auc"],1.0)
        self.assertEqual(r["direction"],"higher")
        self.assertAlmostEqual(r["auc_star"],1.0)

    def test_auc_lower_direction(self):
        labels=pd.Series([False,False,True,True])
        scores=pd.Series([4.,3.,2.,1.])
        r=binary_auc(labels,scores)
        self.assertEqual(r["direction"],"lower")
        self.assertAlmostEqual(r["auc_star"],1.0)

    def test_retrieval_baseline(self):
        f=pd.DataFrame({
            "query_id":[1,1,1,2,2,2],
            "pool_rank":[1,2,3,1,2,3],
            "hit":[False,True,False,False,False,True],
        })
        r=retrieval_baseline(f,"hit",[1,2,3])
        self.assertEqual(r,{"top1":0,"top2":1,"top3":2})

    def test_correlation(self):
        r=correlation(pd.Series([1.,2.,3.]),pd.Series([10.,20.,30.]))
        self.assertAlmostEqual(r["pearson"],1.0)
        self.assertAlmostEqual(r["spearman"],1.0)

    def test_config_forbids_promotion(self):
        cfg={
            "stage":"GD1.1","map_variant":"768_s256","map_crs":"EPSG:3346",
            "labels":["contains_query","center_le40","center_le80"],
            "scientific_boundary":{
                "development_reference_allowed":True,
                "reference_after_gd1_0_freeze_only":True,
                "blind_stress_loaded":False,
                "blind_stress_reference_allowed":False,
                "evidence_mutable":False,
                "reranking_promoted":False,
                "selector_promoted":False,
                "threshold_tuning_enabled":False,
                "learned_model_enabled":False,
                "learned_frontend_enabled":False,
                "bootstrap_enabled":False,
                "state_enabled":False,
            },
        }
        verify_config(cfg)
        cfg["scientific_boundary"]["selector_promoted"]=True
        with self.assertRaisesRegex(RuntimeError,"selector_promoted"):
            verify_config(cfg)

    def test_script_help_without_pythonpath(self):
        script=REPO_ROOT/"scripts/villoc/geometry/gd1_1_postfreeze_geometric_discrimination.py"
        env=os.environ.copy(); env.pop("PYTHONPATH",None)
        p=subprocess.run([sys.executable,str(script),"--help"],cwd=REPO_ROOT,
                         env=env,capture_output=True,text=True,timeout=30)
        self.assertEqual(p.returncode,0,msg=p.stderr)
        self.assertIn("geometric discrimination audit",p.stdout)


if __name__=="__main__":
    unittest.main()
