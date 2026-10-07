from __future__ import annotations

import unittest

import pandas as pd

from scripts.villoc.retrieval.a1_compare_candidate_artifacts import (
    compare_artifacts,
)


def artifact(rows):
    return pd.DataFrame(
        rows,
        columns=[
            "variant",
            "query_id",
            "rank",
            "tile_id",
            "score",
        ],
    )


class ArtifactParityTests(unittest.TestCase):
    def test_exact_artifact_parity(self):
        old = artifact(
            [
                ["v", "1", 1, "a", 0.9],
                ["v", "1", 2, "b", 0.8],
            ]
        )
        new = old.copy()

        result = compare_artifacts(
            old,
            new,
            score_atol=1e-6,
            tie_atol=1e-6,
        )

        self.assertEqual(
            result["status"],
            "PASS_EXACT_ARTIFACT_PARITY",
        )

    def test_near_tie_swap_is_equivalent(self):
        old = artifact(
            [
                ["v", "188", 1, "a", 0.95],
                ["v", "188", 2, "b", 0.93],
                ["v", "188", 3, "c", 0.9143411],
                ["v", "188", 4, "d", 0.9143408],
            ]
        )
        new = artifact(
            [
                ["v", "188", 1, "a", 0.9500001],
                ["v", "188", 2, "b", 0.9299999],
                ["v", "188", 3, "d", 0.9143410],
                ["v", "188", 4, "c", 0.9143409],
            ]
        )

        result = compare_artifacts(
            old,
            new,
            score_atol=1e-6,
            tie_atol=1e-6,
        )

        self.assertEqual(
            result["status"],
            "PASS_TIE_EQUIVALENT_ARTIFACT_PARITY",
        )
        self.assertEqual(
            result["tie_equivalent_queries"],
            ["v:188"],
        )

    def test_meaningful_rank_swap_fails(self):
        old = artifact(
            [
                ["v", "1", 1, "a", 0.95],
                ["v", "1", 2, "b", 0.90],
            ]
        )
        new = artifact(
            [
                ["v", "1", 1, "b", 0.9000001],
                ["v", "1", 2, "a", 0.9499999],
            ]
        )

        result = compare_artifacts(
            old,
            new,
            score_atol=1e-6,
            tie_atol=1e-6,
        )

        self.assertEqual(
            result["status"],
            "FAIL_ARTIFACT_PARITY",
        )

    def test_membership_change_fails(self):
        old = artifact(
            [
                ["v", "1", 1, "a", 0.95],
                ["v", "1", 2, "b", 0.90],
            ]
        )
        new = artifact(
            [
                ["v", "1", 1, "a", 0.95],
                ["v", "1", 2, "c", 0.90],
            ]
        )

        result = compare_artifacts(
            old,
            new,
            score_atol=1e-6,
            tie_atol=1e-6,
        )

        self.assertEqual(
            result["status"],
            "FAIL_ARTIFACT_PARITY",
        )


if __name__ == "__main__":
    unittest.main()
