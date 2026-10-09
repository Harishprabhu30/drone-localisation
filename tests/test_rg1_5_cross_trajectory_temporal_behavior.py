from __future__ import annotations

import unittest

import pandas as pd

from scripts.villoc.geometry.rg1_5_cross_trajectory_temporal_behavior import (
    contiguous_run_lengths,
)


class RG15CrossTrajectoryTemporalBehaviorTests(unittest.TestCase):
    def test_contiguous_run_lengths(self):
        runs = contiguous_run_lengths(
            pd.Series(
                [
                    False,
                    True,
                    True,
                    False,
                    True,
                    True,
                    True,
                    False,
                ]
            )
        )

        self.assertEqual(
            runs,
            [
                2,
                3,
            ],
        )


if __name__ == "__main__":
    unittest.main()
