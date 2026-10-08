from __future__ import annotations

import unittest

from scripts.villoc.retrieval.r2_1_cross_scale_fusion import (
    center_error,
    contains_point,
    evaluate_region,
)


class R21EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.good = {
            "variant": "1024_s256",
            "tile_id": "good",
            "left_easting": 0.0,
            "bottom_northing": 0.0,
            "right_easting": 100.0,
            "top_northing": 100.0,
            "center_easting": 50.0,
            "center_northing": 50.0,
        }
        self.bad = {
            "variant": "512_s256",
            "tile_id": "bad",
            "left_easting": 200.0,
            "bottom_northing": 200.0,
            "right_easting": 300.0,
            "top_northing": 300.0,
            "center_easting": 250.0,
            "center_northing": 250.0,
        }

    def test_contains_point(self):
        self.assertTrue(
            contains_point(self.good, 25.0, 25.0)
        )
        self.assertFalse(
            contains_point(self.bad, 25.0, 25.0)
        )

    def test_center_error(self):
        self.assertAlmostEqual(
            center_error(self.good, 50.0, 50.0),
            0.0,
        )

    def test_region_usefulness_can_exceed_representative(self):
        region = {
            "representative": self.bad,
            "members": [self.bad, self.good],
        }

        result = evaluate_region(
            region,
            25.0,
            25.0,
        )

        self.assertTrue(
            result["region_contains_query"]
        )
        self.assertFalse(
            result["representative_contains_query"]
        )
        self.assertEqual(
            result["best_member_variant"],
            "1024_s256",
        )


if __name__ == "__main__":
    unittest.main()
