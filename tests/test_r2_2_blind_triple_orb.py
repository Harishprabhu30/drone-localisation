from __future__ import annotations

import unittest

from scripts.villoc.retrieval.r2_2_blind_triple_orb import (
    composite_tile_id,
    split_composite_tile_id,
)


class R22MixedScaleOrbAdapterTests(unittest.TestCase):
    def test_composite_tile_id_round_trip(self):
        value = composite_tile_id(
            "1024_s256",
            "sat_000241",
        )
        self.assertEqual(
            value,
            "1024_s256::sat_000241",
        )
        self.assertEqual(
            split_composite_tile_id(value),
            ("1024_s256", "sat_000241"),
        )

    def test_invalid_composite_id_rejected(self):
        with self.assertRaises(ValueError):
            split_composite_tile_id(
                "sat_000241"
            )


if __name__ == "__main__":
    unittest.main()
