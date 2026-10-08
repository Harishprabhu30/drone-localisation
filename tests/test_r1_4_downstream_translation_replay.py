from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.villoc.retrieval.r1_4_downstream_translation_replay import (
    metrics,
    safe_remove_research_run,
    variant_specs,
)


class R14DownstreamReplayTests(unittest.TestCase):
    def test_metrics(self):
        result = metrics(
            np.asarray([3.0, 4.0], dtype=float)
        )
        self.assertEqual(result["count"], 2)
        self.assertAlmostEqual(
            result["rmse_m"],
            (12.5) ** 0.5,
        )
        self.assertEqual(result["final_m"], 4.0)

    def test_variant_specs_requires_three_replay_levels(self):
        cfg = {
            "experiment": {
                "levels": [
                    {"name": "512_s256"},
                    {"name": "768_s256"},
                    {"name": "1024_s256"},
                ]
            }
        }

        result = variant_specs(cfg)
        self.assertEqual(
            list(result),
            [
                "512_s256",
                "768_s256",
                "1024_s256",
            ],
        )

    def test_safe_remove_rejects_outside_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "research"
            outside = root / "other"
            outside.mkdir()

            with self.assertRaisesRegex(
                RuntimeError,
                "outside R1.4",
            ):
                safe_remove_research_run(
                    outside,
                    output,
                )


if __name__ == "__main__":
    unittest.main()
