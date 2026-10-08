from __future__ import annotations

import unittest

import pandas as pd

from scripts.villoc.retrieval.r3_1_retrieval_only_baseline import (
    summarize,
)


class R31RetrievalOnlyBaselineTests(unittest.TestCase):
    def test_summary(self):
        frame = pd.DataFrame(
            {
                "center_error_m": [10.0, 50.0, 100.0],
                "contains_query": [True, False, True],
                "le40": [True, False, False],
                "le80": [True, True, False],
            }
        )

        result = summarize(frame)

        self.assertEqual(result["query_count"], 3)
        self.assertEqual(result["contains_hits"], 2)
        self.assertEqual(result["le40_hits"], 1)
        self.assertEqual(result["le80_hits"], 2)


if __name__ == "__main__":
    unittest.main()
