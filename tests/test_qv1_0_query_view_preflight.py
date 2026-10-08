from __future__ import annotations

import unittest

from scripts.villoc.retrieval.qv1_0_query_view_preflight import (
    crop_boxes,
    horizontal_union_fraction,
    overlap_px,
)


class QV10QueryViewPreflightTests(unittest.TestCase):
    def test_expected_3840x2160_boxes(self):
        boxes = crop_boxes(
            3840,
            2160,
        )
        self.assertEqual(
            boxes["left_square"],
            (0, 0, 2160, 2160),
        )
        self.assertEqual(
            boxes["center_square"],
            (840, 0, 3000, 2160),
        )
        self.assertEqual(
            boxes["right_square"],
            (1680, 0, 3840, 2160),
        )

    def test_full_horizontal_coverage(self):
        boxes = crop_boxes(
            3840,
            2160,
        )
        self.assertAlmostEqual(
            horizontal_union_fraction(
                boxes,
                3840,
            ),
            1.0,
        )

    def test_adjacent_crop_overlap(self):
        boxes = crop_boxes(
            3840,
            2160,
        )
        self.assertEqual(
            overlap_px(
                boxes["left_square"],
                boxes["center_square"],
            ),
            1320,
        )
        self.assertEqual(
            overlap_px(
                boxes["center_square"],
                boxes["right_square"],
            ),
            1320,
        )


if __name__ == "__main__":
    unittest.main()
