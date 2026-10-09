from __future__ import annotations

import unittest

import pandas as pd

from scripts.villoc.retrieval.qv1_5_precision_region_attribution import (
    classify_loss,
)


class QV15PrecisionRegionAttributionTests(unittest.TestCase):
    def test_region_safe(self):
        row = pd.Series({"contain_r20": True, "le80_r20": True})
        self.assertEqual(
            classify_loss(row),
            "precision_only_region_safe",
        )

    def test_le80_safe(self):
        row = pd.Series({"contain_r20": False, "le80_r20": True})
        self.assertEqual(
            classify_loss(row),
            "precision_only_le80_safe",
        )

    def test_true_regression(self):
        row = pd.Series({"contain_r20": False, "le80_r20": False})
        self.assertEqual(
            classify_loss(row),
            "true_region_regression",
        )


if __name__ == "__main__":
    unittest.main()
