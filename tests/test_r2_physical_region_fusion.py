from __future__ import annotations

import unittest

from uavloc.retrieval import (
    bbox_iou,
    cross_scale_duplicate,
    fuse_physical_regions,
    reciprocal_rank,
)


def candidate(
    variant: str,
    tile_id: str,
    rank: int,
    left: float,
    bottom: float,
    right: float,
    top: float,
):
    return {
        "variant": variant,
        "tile_id": tile_id,
        "per_scale_rank": rank,
        "dino_score": 1.0 - 0.01 * rank,
        "rrf_contribution": reciprocal_rank(rank, 60.0),
        "left_easting": left,
        "bottom_northing": bottom,
        "right_easting": right,
        "top_northing": top,
        "center_easting": (left + right) / 2.0,
        "center_northing": (bottom + top) / 2.0,
    }


class R2PhysicalRegionFusionTests(unittest.TestCase):
    def test_bbox_iou_nested_boxes(self):
        small = candidate("512_s256", "a", 1, 0, 0, 100, 100)
        large = candidate("1024_s256", "b", 1, -50, -50, 150, 150)
        self.assertAlmostEqual(bbox_iou(small, large), 0.25)

    def test_duplicate_requires_cross_scale(self):
        a = candidate("512_s256", "a", 1, 0, 0, 100, 100)
        b = candidate("512_s256", "b", 2, 10, 0, 110, 100)
        self.assertFalse(
            cross_scale_duplicate(
                a,
                b,
                max_center_distance_m=51.2,
                min_iou=0.20,
            )
        )

    def test_duplicate_requires_center_and_overlap(self):
        a = candidate("512_s256", "a", 1, 0, 0, 100, 100)
        b = candidate("768_s256", "b", 1, -25, -25, 125, 125)
        self.assertTrue(
            cross_scale_duplicate(
                a,
                b,
                max_center_distance_m=51.2,
                min_iou=0.20,
            )
        )

        far = candidate("768_s256", "far", 1, 500, 500, 650, 650)
        self.assertFalse(
            cross_scale_duplicate(
                a,
                far,
                max_center_distance_m=51.2,
                min_iou=0.20,
            )
        )

    def test_multiscale_support_beats_single_candidate(self):
        candidates = [
            candidate("512_s256", "a", 1, 0, 0, 100, 100),
            candidate("768_s256", "b", 2, -25, -25, 125, 125),
            candidate("512_s256", "c", 2, 400, 400, 500, 500),
        ]

        regions = fuse_physical_regions(
            candidates,
            scale_order=["512_s256", "768_s256"],
            max_center_distance_m=51.2,
            min_iou=0.20,
            final_top_k=20,
        )

        self.assertEqual(regions[0]["support_scale_count"], 2)
        self.assertEqual(
            set(regions[0]["support_scales"]),
            {"512_s256", "768_s256"},
        )
        self.assertEqual(regions[0]["representative"]["tile_id"], "a")

    def test_fixed_seed_prevents_transitive_bridge(self):
        seed = candidate("512_s256", "seed", 1, 0, 0, 100, 100)
        mid = candidate("768_s256", "mid", 2, 40, 0, 190, 150)
        far = candidate("1024_s256", "far", 3, 90, -25, 290, 175)

        regions = fuse_physical_regions(
            [seed, mid, far],
            scale_order=["512_s256", "768_s256", "1024_s256"],
            max_center_distance_m=80.0,
            min_iou=0.10,
            final_top_k=20,
        )

        # mid can support seed; far is close/overlapping with mid but is not
        # sufficiently duplicate of the fixed seed, so it must not bridge in.
        self.assertEqual(len(regions), 2)


if __name__ == "__main__":
    unittest.main()
