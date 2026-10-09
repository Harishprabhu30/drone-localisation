from __future__ import annotations

import unittest

from scripts.trajectory_adapter.ta4_compare_trajectory_behavior import (
    numeric_delta,
)


class TA4CrossTrajectoryComparisonTests(unittest.TestCase):
    def test_numeric_delta(self):
        dev = {
            "a": {
                "b": 2.0,
            }
        }
        blind = {
            "a": {
                "b": 3.5,
            }
        }

        out = numeric_delta(
            dev,
            blind,
            ["a", "b"],
        )

        self.assertEqual(
            out["development"],
            2.0,
        )
        self.assertEqual(
            out["blind_stress"],
            3.5,
        )
        self.assertEqual(
            out["delta_blind_minus_development"],
            1.5,
        )


if __name__ == "__main__":
    unittest.main()
