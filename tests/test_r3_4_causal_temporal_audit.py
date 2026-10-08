from __future__ import annotations

import unittest

from scripts.villoc.retrieval.r3_4_causal_temporal_audit import dist


class R34CausalTemporalAuditTests(unittest.TestCase):
    def test_dist(self):
        self.assertAlmostEqual(
            dist((0.0, 0.0), (3.0, 4.0)),
            5.0,
        )


if __name__ == "__main__":
    unittest.main()
