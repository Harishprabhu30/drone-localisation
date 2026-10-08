from __future__ import annotations

import unittest

import numpy as np

from scripts.villoc.retrieval.r2_3_fused_orb_state_replay import (
    error_metrics,
    first_threshold_crossing,
)


class R23StateReplayTests(unittest.TestCase):
    def test_error_metrics(self):
        result = error_metrics(
            np.asarray([3.0, 4.0], dtype=float)
        )

        self.assertEqual(result["count"], 2)
        self.assertAlmostEqual(
            result["rmse_m"],
            (12.5) ** 0.5,
        )
        self.assertEqual(result["final_m"], 4.0)

    def test_first_threshold_crossing(self):
        result = first_threshold_crossing(
            np.asarray([1, 2, 3, 4]),
            np.asarray([10.0, 99.0, 101.0, 150.0]),
            100.0,
        )

        self.assertEqual(
            result,
            {
                "query_id": 3,
                "error_m": 101.0,
            },
        )

    def test_no_threshold_crossing(self):
        self.assertIsNone(
            first_threshold_crossing(
                np.asarray([1, 2, 3]),
                np.asarray([10.0, 20.0, 30.0]),
                100.0,
            )
        )


if __name__ == "__main__":
    unittest.main()
