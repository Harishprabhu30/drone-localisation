"""Frontend-adapter contract tests."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT=Path(__file__).resolve().parents[1]
SRC_ROOT=REPO_ROOT/"src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0,str(SRC_ROOT))

from uavloc.geometry.frontend_adapter import required_common_metrics


class FrontendAdapterTests(unittest.TestCase):
    def test_common_metrics_contract(self):
        metrics=required_common_metrics()
        self.assertIn("finite_projection",metrics)
        self.assertIn("projected_easting",metrics)
        self.assertIn("total_ms",metrics)
        self.assertEqual(len(metrics),len(set(metrics)))


if __name__=="__main__":
    unittest.main()
