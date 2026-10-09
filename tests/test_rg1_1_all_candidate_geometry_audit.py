from __future__ import annotations

import unittest

import pandas as pd

from scripts.villoc.geometry.rg1_1_all_candidate_geometry_audit import (
    binary_auc_star,
    rate_block,
)


class RG11AllCandidateGeometryAuditTests(unittest.TestCase):
    def test_auc_star_perfect_higher(self):
        out = binary_auc_star(
            pd.Series(
                [
                    False,
                    False,
                    True,
                    True,
                ]
            ),
            pd.Series(
                [
                    0.0,
                    1.0,
                    2.0,
                    3.0,
                ]
            ),
        )

        self.assertAlmostEqual(
            out[
                "auc_star"
            ],
            1.0,
        )
        self.assertEqual(
            out[
                "direction"
            ],
            "higher_favors_positive",
        )

    def test_auc_star_perfect_lower(self):
        out = binary_auc_star(
            pd.Series(
                [
                    False,
                    False,
                    True,
                    True,
                ]
            ),
            pd.Series(
                [
                    3.0,
                    2.0,
                    1.0,
                    0.0,
                ]
            ),
        )

        self.assertAlmostEqual(
            out[
                "auc_star"
            ],
            1.0,
        )
        self.assertEqual(
            out[
                "direction"
            ],
            "lower_favors_positive",
        )

    def test_rate_block(self):
        out = rate_block(
            pd.Series(
                [
                    True,
                    True,
                    False,
                ]
            ),
            pd.Series(
                [
                    True,
                    False,
                    True,
                ]
            ),
        )

        self.assertEqual(
            out[
                "count"
            ],
            2,
        )
        self.assertEqual(
            out[
                "valid_projection_count"
            ],
            1,
        )
        self.assertAlmostEqual(
            out[
                "valid_projection_rate"
            ],
            0.5,
        )


if __name__ == "__main__":
    unittest.main()
