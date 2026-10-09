from __future__ import annotations

import unittest

import pandas as pd

from scripts.villoc.retrieval.qv1_4_redundancy_aware_pool import (
    compress_center,
    rescue_loss_stats,
)


class QV14RedundancyAwarePoolTests(unittest.TestCase):
    def test_compress_center_keeps_unique_and_drops_nearby(self):
        center = pd.DataFrame(
            [
                {"tile_id": "a", "rank": 1},
                {"tile_id": "b", "rank": 2},
                {"tile_id": "c", "rank": 3},
            ]
        )
        centers = {
            "a": (0.0, 0.0),
            "b": (10.0, 0.0),
            "c": (100.0, 0.0),
        }

        kept, dropped = compress_center(
            qid=1,
            center=center,
            centers=centers,
            source_depth=3,
            radius_m=51.2,
        )

        self.assertEqual(
            [row["tile_id"] for row in kept],
            ["a", "c"],
        )
        self.assertEqual(
            [row["tile_id"] for row in dropped],
            ["b"],
        )

    def test_rescue_loss_stats(self):
        baseline = pd.DataFrame(
            {
                "query_id": [1, 2, 3, 4],
                "contain_r20": [True, True, False, False],
                "le40_r20": [True, False, False, True],
                "le80_r20": [True, True, False, False],
            }
        )
        candidate = pd.DataFrame(
            {
                "query_id": [1, 2, 3, 4],
                "contain_r20": [True, False, True, False],
                "le40_r20": [True, True, False, False],
                "le80_r20": [True, False, True, False],
            }
        )

        stats = rescue_loss_stats(
            candidate,
            baseline,
        )

        self.assertEqual(
            stats["contain_r20"]["rescues"],
            1,
        )
        self.assertEqual(
            stats["contain_r20"]["losses"],
            1,
        )
        self.assertEqual(
            stats["le40_r20"]["rescues"],
            1,
        )
        self.assertEqual(
            stats["le40_r20"]["losses"],
            1,
        )


if __name__ == "__main__":
    unittest.main()
