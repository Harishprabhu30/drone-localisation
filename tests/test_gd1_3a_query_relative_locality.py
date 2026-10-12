"""GD1.3-A query-relative/locality freeze tests."""
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

from scripts.villoc.geometry.gd1_3a_freeze_query_relative_locality import (
    add_query_relative,
    connected_components,
    verify_config,
)


class GD13ATests(unittest.TestCase):
    def test_connected_components_nearby(self):
        rel=pd.DataFrame({
            "tile_id":["a","b","c","d"],
            "neighbor_tile_id":["b","a","d","c"],
            "nearby":[True,True,True,True],
            "overlaps":[False,False,False,False],
        })
        comp=connected_components(["a","b","c","d"],rel,"nearby")
        self.assertEqual(comp["a"],comp["b"])
        self.assertEqual(comp["c"],comp["d"])
        self.assertNotEqual(comp["a"],comp["c"])

    def test_query_relative_percentiles_and_z(self):
        f=pd.DataFrame({
            "query_id":[1,1,1,2,2,2],
            "x":[1.,2.,3.,10.,20.,30.],
        })
        out=add_query_relative(f,["x"])
        q1=out[out["query_id"]==1].sort_values("x")
        self.assertAlmostEqual(q1.iloc[0]["x__q_pct"],0.0)
        self.assertAlmostEqual(q1.iloc[1]["x__q_pct"],0.5)
        self.assertAlmostEqual(q1.iloc[2]["x__q_pct"],1.0)
        self.assertAlmostEqual(q1.iloc[1]["x__q_z"],0.0)

    def test_query_relative_missing_flag(self):
        f=pd.DataFrame({"query_id":[1,1,1],"x":[1.0,np.nan,3.0]})
        out=add_query_relative(f,["x"])
        self.assertEqual(out["x__missing"].tolist(),[0,1,0])
        self.assertTrue(np.isnan(out.loc[1,"x__q_z"]))

    def test_config_rejects_reference(self):
        cfg={
            "stage":"GD1.3-A",
            "query_relative":{
                "transforms":["robust_z","percentile","missing"],
            },
            "locality":{"graphs":["nearby","overlaps"]},
            "scientific_boundary":{
                "reference_allowed":False,
                "reference_attachment_enabled":False,
                "oracle_labels_allowed":False,
                "reranking_enabled":False,
                "selector_enabled":False,
                "threshold_tuning_enabled":False,
                "orb_rerun_enabled":False,
                "learned_frontend_enabled":False,
                "temporal_fusion_enabled":False,
                "bootstrap_enabled":False,
                "state_enabled":False,
            },
        }
        verify_config(cfg)
        cfg["scientific_boundary"]["reference_allowed"]=True
        with self.assertRaisesRegex(RuntimeError,"reference_allowed"):
            verify_config(cfg)


if __name__=="__main__":
    unittest.main()
