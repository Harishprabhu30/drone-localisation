from __future__ import annotations

import unittest

import pandas as pd

from scripts.villoc.retrieval.qv1_1_query_view_retrieval import (
    l2_normalize,
    rrf_fuse,
)


class QV11QueryViewRetrievalTests(unittest.TestCase):
    def test_l2_normalize(self):
        import numpy as np

        out = l2_normalize(
            np.asarray(
                [[3.0, 4.0]],
                dtype=np.float32,
            )
        )
        self.assertAlmostEqual(
            float(
                np.linalg.norm(
                    out[0]
                )
            ),
            1.0,
        )

    def test_rrf_fuse_prefers_multi_view_support(self):
        left = pd.DataFrame(
            [
                {
                    "query_id": 1,
                    "rank": 1,
                    "tile_id": "a",
                },
                {
                    "query_id": 1,
                    "rank": 2,
                    "tile_id": "b",
                },
            ]
        )
        center = pd.DataFrame(
            [
                {
                    "query_id": 1,
                    "rank": 1,
                    "tile_id": "b",
                },
                {
                    "query_id": 1,
                    "rank": 2,
                    "tile_id": "a",
                },
            ]
        )
        right = pd.DataFrame(
            [
                {
                    "query_id": 1,
                    "rank": 1,
                    "tile_id": "b",
                },
                {
                    "query_id": 1,
                    "rank": 2,
                    "tile_id": "a",
                },
            ]
        )

        fused = rrf_fuse(
            {
                "left_square": left,
                "center_square": center,
                "right_square": right,
            },
            [1],
            rrf_k=60.0,
            per_view_depth=2,
            final_top_k=2,
        )

        self.assertEqual(
            str(
                fused.iloc[
                    0
                ][
                    "tile_id"
                ]
            ),
            "b",
        )


if __name__ == "__main__":
    unittest.main()
