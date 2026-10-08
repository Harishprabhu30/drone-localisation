from __future__ import annotations

import unittest

from scripts.villoc.retrieval.r3_2_consensus_selector import (
    best_support_region,
    find_768_anchor_region,
    select_for_query,
    select_region_member,
)


def region(
    rank,
    support,
    score,
    rep,
    members,
    support_scales="",
):
    return {
        "query_id": 1,
        "fused_rank": rank,
        "representative_composite_tile_id": rep,
        "representative_variant": rep.split("::", 1)[0],
        "representative_original_tile_id": rep.split("::", 1)[1],
        "fused_rrf_score": score,
        "support_scale_count": support,
        "support_scales": support_scales,
        "members": members,
    }


class R32ConsensusSelectorTests(unittest.TestCase):
    def setUp(self):
        self.anchor = region(
            2,
            1,
            0.016,
            "768_s256::sat_a",
            [
                {
                    "variant": "768_s256",
                    "tile_id": "sat_a",
                    "per_scale_rank": 1,
                }
            ],
            "768_s256",
        )
        self.consensus = region(
            1,
            2,
            0.030,
            "512_s256::sat_b",
            [
                {
                    "variant": "512_s256",
                    "tile_id": "sat_b",
                    "per_scale_rank": 1,
                },
                {
                    "variant": "768_s256",
                    "tile_id": "sat_c",
                    "per_scale_rank": 4,
                },
            ],
            "512_s256|768_s256",
        )
        self.three = region(
            3,
            3,
            0.040,
            "1024_s256::sat_d",
            [
                {
                    "variant": "512_s256",
                    "tile_id": "sat_e",
                    "per_scale_rank": 3,
                },
                {
                    "variant": "768_s256",
                    "tile_id": "sat_f",
                    "per_scale_rank": 5,
                },
                {
                    "variant": "1024_s256",
                    "tile_id": "sat_d",
                    "per_scale_rank": 1,
                },
            ],
            "512_s256|768_s256|1024_s256",
        )

    def test_best_support_region_prioritizes_support(self):
        got = best_support_region(
            [self.anchor, self.consensus, self.three]
        )
        self.assertEqual(
            got["support_scale_count"],
            3,
        )

    def test_find_anchor(self):
        got = find_768_anchor_region(
            [self.anchor, self.consensus],
            "sat_a",
        )
        self.assertEqual(
            got["fused_rank"],
            2,
        )

    def test_prefer_768_member(self):
        tile_id, reason = select_region_member(
            self.consensus,
            "prefer_768",
        )
        self.assertEqual(
            tile_id,
            "768_s256::sat_c",
        )
        self.assertEqual(
            reason,
            "preferred_768_member",
        )

    def test_strict_support_rescue_switches(self):
        got = select_for_query(
            qid=1,
            tile_id_768="sat_a",
            regions=[self.anchor, self.consensus],
            policy_name="768_strict_support_rescue",
            missing_anchor_support_count=1,
        )
        self.assertTrue(
            got["switched_from_768_top1"]
        )
        self.assertEqual(
            got["tile_id"],
            "768_s256::sat_c",
        )

    def test_strict_support_rescue_holds_on_equal_support(self):
        equal = dict(self.consensus)
        equal["support_scale_count"] = 1
        got = select_for_query(
            qid=1,
            tile_id_768="sat_a",
            regions=[self.anchor, equal],
            policy_name="768_strict_support_rescue",
            missing_anchor_support_count=1,
        )
        self.assertFalse(
            got["switched_from_768_top1"]
        )
        self.assertEqual(
            got["tile_id"],
            "768_s256::sat_a",
        )

    def test_three_scale_rescue_requires_three_scale_advantage(self):
        got = select_for_query(
            qid=1,
            tile_id_768="sat_a",
            regions=[self.anchor, self.three],
            policy_name="768_three_scale_rescue",
            missing_anchor_support_count=1,
        )
        self.assertTrue(
            got["switched_from_768_top1"]
        )
        self.assertEqual(
            got["tile_id"],
            "768_s256::sat_f",
        )


if __name__ == "__main__":
    unittest.main()
