from __future__ import annotations

import unittest

import pandas as pd

from scripts.villoc.geometry.rg1_3_retrieval_anchor_trust_refinement import (
    evaluate_gate,
    gate_mask,
)


class RG13RetrievalAnchorTrustRefinementTests(unittest.TestCase):
    def test_gate_mask_ratio_inside(self):
        frame = pd.DataFrame(
            {
                "inliers": [5, 10, 15],
                "inlier_ratio": [0.1, 0.4, 0.6],
                "projected_inside_tile": [True, False, True],
            }
        )

        mask = gate_mask(
            frame,
            family="inliers_plus_ratio_inside",
            inlier_threshold=8,
            ratio_threshold=0.3,
        )

        self.assertEqual(
            mask.tolist(),
            [False, False, True],
        )

    def test_evaluate_gate_precision(self):
        frame = pd.DataFrame(
            {
                "contains_query": [True, False, True],
                "le80": [True, False, True],
                "le40": [True, False, False],
                "valid_projection": [True, True, True],
                "center_error_m": [10.0, 200.0, 60.0],
                "projected_error_m": [5.0, 220.0, 8.0],
            }
        )

        accepted = pd.Series([True, True, False])

        out = evaluate_gate(
            frame=frame,
            accepted=accepted,
            labels=["contains_query", "le80", "le40"],
        )

        self.assertAlmostEqual(
            out["contains_query"]["precision"],
            0.5,
        )
        self.assertEqual(
            out["accepted_query_count"],
            2,
        )


if __name__ == "__main__":
    unittest.main()
