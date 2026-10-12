"""GD1.0 richer Top20 geometric-evidence tests."""
from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from uavloc.geometry.top20_geometric_evidence import (
    GRID_SIZE,
    _convex_hull_coverage,
    _grid_support,
    _orientation_stats,
    _pca_minor_major_ratio,
    _scale_stats,
)


class GD10EvidenceTests(unittest.TestCase):
    def test_grid_support_spreads_across_cells(self):
        pts = np.array([[10., 10.], [90., 10.], [10., 90.], [90., 90.]])
        cells, occ, ent = _grid_support(pts, (100, 100))
        self.assertEqual(cells, 4)
        self.assertAlmostEqual(occ, 4 / (GRID_SIZE * GRID_SIZE))
        self.assertGreater(ent, 0.0)

    def test_convex_hull_coverage(self):
        pts = np.array([[0., 0.], [99., 0.], [99., 99.], [0., 99.]])
        cov = _convex_hull_coverage(pts, (100, 100))
        self.assertGreater(cov, 0.95)
        self.assertLessEqual(cov, 1.0)

    def test_pca_ratio_detects_collinearity(self):
        line = np.array([[0., 0.], [1., 0.], [2., 0.], [3., 0.]])
        square = np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.]])
        self.assertLess(_pca_minor_major_ratio(line), 1e-6)
        self.assertGreater(_pca_minor_major_ratio(square), 0.9)

    def test_orientation_consistency(self):
        q = np.array([10., 20., 30.])
        t = np.array([40., 50., 60.])
        s = _orientation_stats(q, t)
        self.assertAlmostEqual(s["orientation_delta_mean_deg"], 30.0, places=6)
        self.assertAlmostEqual(s["orientation_resultant"], 1.0, places=6)
        self.assertAlmostEqual(s["orientation_mad_deg"], 0.0, places=6)

    def test_scale_consistency(self):
        q = np.array([10., 20., 30.])
        t = np.array([20., 40., 60.])
        s = _scale_stats(q, t)
        self.assertAlmostEqual(s["log_scale_ratio_mad"], 0.0, places=6)

    def test_script_help_without_pythonpath(self):
        script = REPO_ROOT / "scripts/villoc/geometry/gd1_0_freeze_top20_geometric_evidence.py"
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        p = subprocess.run(
            [sys.executable, str(script), "--help"],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(p.returncode, 0, msg=p.stderr)
        self.assertIn("richer geometric evidence", p.stdout)


if __name__ == "__main__":
    unittest.main()
