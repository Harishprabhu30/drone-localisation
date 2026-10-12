"""RG2.4-A locality persistence / evidence-independence tests."""
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

from uavloc.geometry.locality_persistence import (
    max_pairwise_span,
    path_length,
    segment_localities,
    support_redundancy_ratio,
)


class RG24ATests(unittest.TestCase):
    def test_segment_localities_spatial_threshold(self):
        t=[0.,1.,2.,3.]
        pts=np.array([[0.,0.],[5.,0.],[40.,0.],[42.,0.]])
        steps,nominal,max_dt=segment_localities(
            t,pts,link_threshold_m=12.8,contiguous_gap_multiplier=1.5
        )
        self.assertAlmostEqual(nominal,1.0)
        self.assertAlmostEqual(max_dt,1.5)
        self.assertEqual([s.locality_index for s in steps],[1,1,2,2])
        self.assertEqual([s.linked_to_previous for s in steps],[False,True,False,True])

    def test_segment_localities_temporal_gap_breaks_run(self):
        t=[0.,1.,4.]
        pts=np.array([[0.,0.],[1.,0.],[2.,0.]])
        steps,_,max_dt=segment_localities(
            t,pts,link_threshold_m=25.6,contiguous_gap_multiplier=1.5
        )
        self.assertAlmostEqual(max_dt,3.0)
        # median dt=(1+3)/2=2 => 3 s is still contiguous.
        self.assertEqual([s.locality_index for s in steps],[1,1,1])

        t=[0.,1.,10.,11.]
        pts=np.array([[0.,0.],[1.,0.],[2.,0.],[3.,0.]])
        steps,nominal,max_dt=segment_localities(
            t,pts,link_threshold_m=25.6,contiguous_gap_multiplier=1.5
        )
        self.assertAlmostEqual(nominal,1.0)
        self.assertAlmostEqual(max_dt,1.5)
        self.assertEqual([s.locality_index for s in steps],[1,1,2,2])

    def test_support_redundancy_ratio(self):
        distinct,total,ratio=support_redundancy_ratio(
            [{"A","B"},{"A","B"},{"B","C"}]
        )
        self.assertEqual(distinct,3)
        self.assertEqual(total,6)
        self.assertAlmostEqual(ratio,0.5)

    def test_path_and_span(self):
        pts=np.array([[0.,0.],[3.,4.],[6.,8.]])
        self.assertAlmostEqual(path_length(pts),10.0)
        self.assertAlmostEqual(max_pairwise_span(pts),10.0)

    def test_single_point(self):
        steps,nominal,max_dt=segment_localities(
            [5.],np.array([[1.,2.]]),
            link_threshold_m=12.8,contiguous_gap_multiplier=1.5
        )
        self.assertEqual(len(steps),1)
        self.assertEqual(steps[0].locality_index,1)
        self.assertEqual(nominal,0.0)
        self.assertEqual(max_dt,0.0)

    def test_nonmonotonic_timestamps_rejected(self):
        with self.assertRaisesRegex(ValueError,"strictly increasing"):
            segment_localities(
                [0.,1.,1.],np.array([[0.,0.],[1.,0.],[2.,0.]]),
                link_threshold_m=12.8,contiguous_gap_multiplier=1.5
            )

    def test_script_help_without_pythonpath(self):
        script=REPO_ROOT/"scripts/villoc/geometry/rg2_4a_freeze_locality_persistence.py"
        env=os.environ.copy(); env.pop("PYTHONPATH",None)
        p=subprocess.run(
            [sys.executable,str(script),"--help"],
            cwd=REPO_ROOT,env=env,capture_output=True,text=True,timeout=30
        )
        self.assertEqual(p.returncode,0,msg=p.stderr)
        self.assertIn("locality persistence",p.stdout)


if __name__=="__main__":
    unittest.main()
