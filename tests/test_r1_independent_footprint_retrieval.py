from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from scripts.villoc.retrieval.r1_3_independent_footprint_retrieval import (
    first_true_rank,
    pairwise_spread,
    point_containment_mask,
    spatial_nms_count,
)


class R1IndependentFootprintRetrievalTests(unittest.TestCase):
    def test_first_true_rank_is_one_based(self):
        self.assertEqual(
            first_true_rank(
                np.asarray([False, False, True, False])
            ),
            3,
        )
        self.assertIsNone(
            first_true_rank(np.asarray([False, False]))
        )

    def test_point_containment_matches_shared_boundary(self):
        tiles = pd.DataFrame(
            {
                "left_easting": [0.0, 10.0],
                "bottom_northing": [0.0, 0.0],
                "right_easting": [10.0, 20.0],
                "top_northing": [10.0, 10.0],
            }
        )

        mask = point_containment_mask(
            10.0,
            5.0,
            tiles,
        )

        self.assertEqual(mask.tolist(), [True, True])

    def test_spatial_nms_count_respects_rank_order(self):
        centers = np.asarray(
            [
                [0.0, 0.0],
                [10.0, 0.0],
                [60.0, 0.0],
                [120.0, 0.0],
            ]
        )

        self.assertEqual(
            spatial_nms_count(
                centers,
                radius_m=51.2,
            ),
            3,
        )

    def test_pairwise_spread_two_points(self):
        centers = np.asarray(
            [
                [0.0, 0.0],
                [3.0, 4.0],
            ]
        )

        spread = pairwise_spread(centers)

        self.assertAlmostEqual(spread["mean_m"], 5.0)
        self.assertAlmostEqual(spread["median_m"], 5.0)


if __name__ == "__main__":
    unittest.main()
