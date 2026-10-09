from __future__ import annotations

import unittest

import pandas as pd

from scripts.trajectory_adapter.ta4_run_frozen_qv import (
    jaccard,
    summarize,
)


class TA4FrozenQVRunnerTests(unittest.TestCase):
    def test_jaccard(self):
        self.assertAlmostEqual(
            jaccard({"a", "b"}, {"b", "c"}),
            1.0 / 3.0,
        )

    def test_jaccard_empty(self):
        self.assertEqual(
            jaccard(set(), set()),
            1.0,
        )

    def test_summarize(self):
        out = summarize([1.0, 2.0, 3.0])
        self.assertEqual(out["count"], 3)
        self.assertEqual(out["median"], 2.0)


if __name__ == "__main__":
    unittest.main()
