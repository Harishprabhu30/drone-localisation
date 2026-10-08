from __future__ import annotations

import unittest

from scripts.villoc.retrieval.qv1_3_fixed_budget_multiview_pool import (
    spatially_distinct,
)


class QV13FixedBudgetPoolTests(unittest.TestCase):
    def test_spatially_distinct(self):
        centers = {
            "a": (0.0, 0.0),
            "b": (10.0, 0.0),
            "c": (100.0, 0.0),
        }
        selected = [
            {"tile_id": "a"}
        ]

        self.assertFalse(
            spatially_distinct(
                "b",
                selected,
                centers,
                51.2,
            )
        )
        self.assertTrue(
            spatially_distinct(
                "c",
                selected,
                centers,
                51.2,
            )
        )


if __name__ == "__main__":
    unittest.main()
