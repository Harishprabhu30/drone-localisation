"""GD1.3-B OOF query-relative/locality audit tests."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT=Path(__file__).resolve().parents[1]
SRC_ROOT=REPO_ROOT/"src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0,str(SRC_ROOT))

from uavloc.evaluation.oof_ranking import (
    binary_auc,
    contiguous_folds,
    fit_logistic,
    predict_logistic,
    query_rank_hits,
)
from scripts.villoc.geometry.gd1_3b_oof_query_relative_locality_audit import (
    feature_families,
    load_json_compat,
    verify_config,
)


class GD13BTests(unittest.TestCase):
    def test_contiguous_folds_cover_queries(self):
        folds=contiguous_folds(list(range(1,11)),5)
        self.assertEqual(set().union(*folds),set(range(1,11)))

    def test_logistic_simple(self):
        X=np.array([[-2.],[-1.],[1.],[2.]])
        y=np.array([0.,0.,1.,1.])
        beta=fit_logistic(X,y,l2=1.0,max_iterations=100,tolerance=1e-8,balanced=True)
        p=predict_logistic(X,beta)
        self.assertLess(p[0],0.5)
        self.assertGreater(p[-1],0.5)

    def test_auc(self):
        auc=binary_auc(pd.Series([False,False,True,True]),pd.Series([1.,2.,3.,4.]))
        self.assertAlmostEqual(auc,1.0)

    def test_query_rank_hits(self):
        f=pd.DataFrame({
            "query_id":[1,1,2,2],
            "pool_rank":[1,2,1,2],
            "tile_id":["a","b","c","d"],
            "hit":[False,True,True,False],
            "score":[0.1,0.9,0.8,0.2],
        })
        self.assertEqual(
            query_rank_hits(f,score_col="score",label_col="hit",topks=[1,2]),
            {"top1":2,"top2":2},
        )

    def test_feature_families(self):
        f=pd.DataFrame({
            "inliers":[1],
            "inliers__q_z":[0.1],
            "inliers__q_pct":[1.0],
            "inliers__missing":[0],
            "foo__q_z":[0.0],
            "foo__q_pct":[0.5],
            "foo__missing":[0],
            "nearby_component_size":[2],
            "nearby_component_min_pool_rank":[1],
            "overlaps_component_size":[3],
            "overlaps_component_min_pool_rank":[1],
            "pool_rank":[1],
        })
        fam=feature_families(f)
        self.assertIn("inliers__q_pct",fam["inliers_query_relative"])
        self.assertIn("nearby_component_size",fam["nearby_locality_only"])
        self.assertIn("overlaps_component_size",fam["overlap_locality_only"])

    def test_load_json_compat_accepts_historical_literal_newline(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"report.json"
            path.write_text('{"status":"ok"}\\n',encoding="utf-8")
            self.assertEqual(load_json_compat(path),{"status":"ok"})

    def test_load_json_compat_rejects_other_extra_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"report.json"
            path.write_text('{"status":"ok"} garbage',encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError,"Invalid JSON"):
                load_json_compat(path)

    def test_config_boundary(self):
        cfg={
            "stage":"GD1.3-B",
            "evaluation":{
                "fold_scheme":"contiguous_query_blocks",
                "no_hyperparameter_tuning":True,
                "logistic_l2":1.0,
            },
            "scientific_boundary":{
                "development_only":True,
                "blind_stress_loaded":False,
                "blind_stress_reference_used":False,
                "hyperparameter_tuning_enabled":False,
                "frontend_changed":False,
                "candidate_pool_changed":False,
                "deployable_model_exported":False,
                "bootstrap_enabled":False,
                "state_enabled":False,
            },
        }
        verify_config(cfg)
        cfg["scientific_boundary"]["frontend_changed"]=True
        with self.assertRaisesRegex(RuntimeError,"frontend_changed"):
            verify_config(cfg)


if __name__=="__main__":
    unittest.main()
