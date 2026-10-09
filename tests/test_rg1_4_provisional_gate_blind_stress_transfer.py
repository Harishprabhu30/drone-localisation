from __future__ import annotations

import unittest

import pandas as pd

from scripts.villoc.geometry.rg1_4_provisional_gate_blind_stress_transfer import (
    choose_threshold,
)


class RG14ProvisionalGateBlindStressTransferTests(unittest.TestCase):
    def test_choose_highest_threshold_meeting_all_recalls(self):
        table = pd.DataFrame(
            {
                "family": ["inliers_only"] * 3,
                "inlier_threshold": [6, 7, 8],
                "accepted_query_fraction": [0.8, 0.7, 0.6],
                "contains_query_precision": [0.5, 0.6, 0.7],
                "contains_query_recall": [0.96, 0.92, 0.85],
                "le80_precision": [0.5, 0.6, 0.7],
                "le80_recall": [0.95, 0.91, 0.84],
                "le40_precision": [0.4, 0.5, 0.6],
                "le40_recall": [0.97, 0.93, 0.86],
                "accepted_projected_error_median": [5.0, 4.5, 4.0],
                "projection_improves_rate": [0.7, 0.8, 0.9],
            }
        )

        selected = choose_threshold(
            table,
            family="inliers_only",
            minimum_recall={
                "contains_query": 0.90,
                "le80": 0.90,
                "le40": 0.90,
            },
        )

        self.assertEqual(selected["inlier_threshold"], 7)


if __name__ == "__main__":
    unittest.main()
