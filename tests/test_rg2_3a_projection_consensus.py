"""RG2.3-A deterministic consensus tests."""
from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT=Path(__file__).resolve().parents[1]
SRC_ROOT=REPO_ROOT/"src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0,str(SRC_ROOT))

from uavloc.geometry.projection_consensus import (
    coordinate_median,
    geometric_median,
    medoid,
    dominant_radius_cluster_medoid,
    inlier_huber_centroid,
    dispersion,
)


class RG23ATests(unittest.TestCase):
    def test_coordinate_and_geometric_median(self):
        pts=np.array([[0.,0.],[2.,0.],[1.,10.]])
        self.assertEqual(coordinate_median(pts),(1.0,0.0))
        gx,gy=geometric_median(pts)
        self.assertTrue(np.isfinite([gx,gy]).all())

    def test_medoid_deterministic(self):
        idx,x,y=medoid(np.array([[0.,0.],[1.,0.],[10.,0.]]))
        self.assertEqual(idx,1)
        self.assertEqual((x,y),(1.0,0.0))

    def test_radius_cluster_prefers_largest_support(self):
        pts=np.array([[0.,0.],[1.,0.],[2.,0.],[100.,0.]])
        idx,support,x,y=dominant_radius_cluster_medoid(pts,radius_m=2.1)
        self.assertEqual(support,[0,1,2])
        self.assertEqual(idx,1)
        self.assertEqual((x,y),(1.0,0.0))

    def test_huber_centroid_resists_outlier(self):
        pts=np.array([[0.,0.],[1.,0.],[2.,0.],[100.,0.]])
        x,y=inlier_huber_centroid(pts,inlier_weights=np.ones(4),huber_k=1.345)
        self.assertLess(x,20.0)
        self.assertAlmostEqual(y,0.0)

    def test_dispersion(self):
        d=dispersion(np.array([[0.,0.],[3.,4.]]))
        self.assertEqual(d["point_count"],2)
        self.assertAlmostEqual(d["pairwise_max_m"],5.0)
        self.assertAlmostEqual(d["bbox_diagonal_m"],5.0)

    def test_nonfinite_rejected(self):
        with self.assertRaises(ValueError):
            geometric_median(np.array([[0.,0.],[np.nan,1.]]))

    def test_script_help_without_pythonpath(self):
        script=REPO_ROOT/"scripts/villoc/geometry/rg2_3a_freeze_projection_consensus.py"
        env=os.environ.copy(); env.pop("PYTHONPATH",None)
        p=subprocess.run([sys.executable,str(script),"--help"],cwd=REPO_ROOT,
                         env=env,capture_output=True,text=True,timeout=30)
        self.assertEqual(p.returncode,0,msg=p.stderr)
        self.assertIn("reference-free projection consensus",p.stdout)


if __name__=="__main__":
    unittest.main()
