from __future__ import annotations

import unittest

import pandas as pd

from scripts.villoc.retrieval.r3_3_confidence_feature_audit import (
    binary_auc,
    category,
    member_rank,
    rank_stats,
)


class R33ConfidenceFeatureAuditTests(unittest.TestCase):
    def test_category(self):
        self.assertEqual(category(True, False), "anchor_only_good")
        self.assertEqual(category(False, True), "fused_only_good")
        self.assertEqual(category(True, True), "both_good")
        self.assertEqual(category(False, False), "both_bad")

    def test_member_rank(self):
        members = [
            {"variant": "512_s256", "per_scale_rank": 2},
            {"variant": "768_s256", "per_scale_rank": 7},
        ]
        self.assertEqual(
            member_rank(members, "768_s256"),
            7.0,
        )

    def test_rank_stats(self):
        result = rank_stats(
            [
                {"per_scale_rank": 1},
                {"per_scale_rank": 2},
                {"per_scale_rank": 3},
            ]
        )
        self.assertAlmostEqual(result["mean"], 2.0)
        self.assertAlmostEqual(result["max"], 3.0)

    def test_auc_perfect_higher(self):
        result = binary_auc(
            pd.Series([False, False, True, True]),
            pd.Series([0.0, 1.0, 2.0, 3.0]),
        )
        self.assertIsNotNone(result)
        self.assertAlmostEqual(
            result["diagnostic_auc_best_direction"],
            1.0,
        )
        self.assertEqual(
            result["direction"],
            "higher_favors_fused",
        )

    def test_auc_perfect_lower(self):
        result = binary_auc(
            pd.Series([False, False, True, True]),
            pd.Series([3.0, 2.0, 1.0, 0.0]),
        )
        self.assertIsNotNone(result)
        self.assertAlmostEqual(
            result["diagnostic_auc_best_direction"],
            1.0,
        )
        self.assertEqual(
            result["direction"],
            "lower_favors_fused",
        )


if __name__ == "__main__":
    unittest.main()
