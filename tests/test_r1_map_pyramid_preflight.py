from __future__ import annotations

import unittest

from scripts.villoc.retrieval.r1_0_map_pyramid_preflight import axis_starts


class R1MapPyramidPreflightTests(unittest.TestCase):
    def test_axis_starts_anchors_last_window(self):
        self.assertEqual(
            axis_starts(1000, 384, 256),
            [0, 256, 512, 616],
        )

    def test_axis_starts_exact_grid(self):
        self.assertEqual(
            axis_starts(1024, 512, 256),
            [0, 256, 512],
        )

    def test_stride_larger_than_tile_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            axis_starts(1024, 256, 512)


if __name__ == "__main__":
    unittest.main()
