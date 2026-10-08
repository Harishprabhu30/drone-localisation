from __future__ import annotations

import unittest

from scripts.villoc.retrieval.r2_5_gate_leader_attribution import (
    parse_int_list,
    parse_str_list,
    variant_from_tile_id,
)


class R25GateLeaderAttributionTests(unittest.TestCase):
    def test_parse_int_list(self):
        self.assertEqual(
            parse_int_list("1,2,4,7"),
            [1, 2, 4, 7],
        )

    def test_parse_str_list(self):
        self.assertEqual(
            parse_str_list(
                "512_s256::sat_1,768_s256::sat_2"
            ),
            [
                "512_s256::sat_1",
                "768_s256::sat_2",
            ],
        )

    def test_variant_from_composite_id(self):
        self.assertEqual(
            variant_from_tile_id(
                "1024_s256::sat_000241"
            ),
            "1024_s256",
        )

    def test_variant_from_single_scale_id(self):
        self.assertEqual(
            variant_from_tile_id(
                "sat_000241"
            ),
            "single_scale",
        )


if __name__ == "__main__":
    unittest.main()
