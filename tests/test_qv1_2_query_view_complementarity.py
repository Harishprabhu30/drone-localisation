from __future__ import annotations

import unittest

from scripts.villoc.retrieval.qv1_2_query_view_complementarity import (
    category,
)


class QV12QueryViewComplementarityTests(unittest.TestCase):
    def test_categories(self):
        self.assertEqual(
            category(True, True),
            "both",
        )
        self.assertEqual(
            category(True, False),
            "center_only",
        )
        self.assertEqual(
            category(False, True),
            "alt_only_rescue",
        )
        self.assertEqual(
            category(False, False),
            "neither",
        )


if __name__ == "__main__":
    unittest.main()
