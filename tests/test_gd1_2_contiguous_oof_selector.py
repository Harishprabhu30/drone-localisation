"""GD1.2 contiguous out-of-fold selector diagnostics tests."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT=Path(__file__).resolve().parents[1]
SRC_ROOT=REPO_ROOT/"src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0,str(SRC_ROOT))

from scripts.villoc.geometry.gd1_2_contiguous_oof_selector_diagnostic import (
    contiguous_folds,
    fit_logistic,
    predict_logistic,
    query_rank_hits,
    sigmoid,
    verify_config,
)


class GD12Tests(unittest.TestCase):
    def test_sigmoid_bounds(self):
        out=sigmoid(np.array([-1000.,0.,1000.]))
        self.assertGreaterEqual(out[0],0.0)
        self.assertAlmostEqual(out[1],0.5)
        self.assertLessEqual(out[2],1.0)
        self.assertLess(out[0],out[1])
        self.assertLess(out[1],out[2])

    def test_contiguous_folds_cover_queries(self):
        folds=contiguous_folds(list(range(1,11)),5)
        merged=set().union(*folds)
        self.assertEqual(merged,set(range(1,11)))
        self.assertEqual(sum(len(f) for f in folds),10)

    def test_logistic_separates_simple_data(self):
        X=np.array([[-2.],[-1.],[1.],[2.]])
        y=np.array([0.,0.,1.,1.])
        beta=fit_logistic(
            X,y,l2=1.0,max_iterations=100,tolerance=1e-8,balanced=True
        )
        pred=predict_logistic(X,beta)
        self.assertLess(pred[0],0.5)
        self.assertGreater(pred[-1],0.5)

    def test_query_rank_hits(self):
        f=pd.DataFrame({
            "query_id":[1,1,1,2,2,2],
            "pool_rank":[1,2,3,1,2,3],
            "tile_id":["a","b","c","d","e","f"],
            "hit":[False,True,False,False,False,True],
            "score":[0.1,0.9,0.2,0.4,0.2,0.8],
        })
        r=query_rank_hits(f,"score","hit",[1,2,3])
        self.assertEqual(r,{"top1":2,"top2":2,"top3":2})

    def test_config_rejects_hyperparameter_tuning(self):
        cfg={
            "stage":"GD1.2",
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
        cfg["scientific_boundary"]["deployable_model_exported"]=True
        with self.assertRaisesRegex(RuntimeError,"deployable_model_exported"):
            verify_config(cfg)


if __name__=="__main__":
    unittest.main()
