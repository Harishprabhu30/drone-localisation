from __future__ import annotations

import unittest

import pandas as pd

from scripts.villoc.geometry.rg1_2_local_geometry_gate_calibration import (
    gate_mask,
    label_metrics,
)


class RG12LocalGeometryGateCalibrationTests(unittest.TestCase):
    def test_inlier_gate(self):
        frame = pd.DataFrame(
            {
                "inliers": [4, 8, 12],
                "inlier_ratio": [0.2, 0.4, 0.6],
                "projected_inside_tile": [True, True, False],
            }
        )

        mask = gate_mask(
            frame,
            family="inliers_only",
            inlier_threshold=8,
            ratio_threshold=None,
        )

        self.assertEqual(
            mask.tolist(),
            [
                False,
                True,
                True,
            ],
        )

    def test_label_metrics_query_retention(self):
        frame = pd.DataFrame(
            {
                "query_id": [1, 1, 2, 2],
                "contains_query": [
                    True,
                    False,
                    True,
                    False,
                ],
            }
        )

        accepted = pd.Series(
            [
                True,
                False,
                False,
                True,
            ]
        )

        out = label_metrics(
            frame,
            accepted,
            "contains_query",
        )

        self.assertAlmostEqual(
            out[
                "candidate_precision"
            ],
            0.5,
        )
        self.assertAlmostEqual(
            out[
                "positive_query_retention"
            ],
            0.5,
        )
        self.assertEqual(
            out[
                "wrong_only_accepted_query_count"
            ],
            1,
        )


if __name__ == "__main__":
    unittest.main()
