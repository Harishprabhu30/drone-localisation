from __future__ import annotations

import unittest

import pandas as pd

from scripts.villoc.retrieval.r2_4_state_failure_attribution import (
    last_event_at_or_before,
    next_event_after,
    transform_delta,
    wrap_degrees,
)


class R24FailureAttributionTests(unittest.TestCase):
    def test_wrap_degrees(self):
        self.assertAlmostEqual(
            wrap_degrees(174.0),
            174.0,
        )
        self.assertAlmostEqual(
            wrap_degrees(181.0),
            -179.0,
        )

    def test_transform_delta(self):
        previous = {
            "scale_m_per_visual_px": 0.2,
            "rotation_deg": -30.0,
            "b_real": 10.0,
            "b_imag": 20.0,
        }
        current = {
            "scale_m_per_visual_px": 0.1,
            "rotation_deg": 150.0,
            "b_real": 13.0,
            "b_imag": 24.0,
        }

        result = transform_delta(
            previous,
            current,
        )

        self.assertAlmostEqual(
            result["scale_ratio"],
            0.5,
        )
        self.assertAlmostEqual(
            result["scale_change_pct"],
            -50.0,
        )
        self.assertAlmostEqual(
            abs(result["rotation_delta_deg"]),
            180.0,
        )
        self.assertAlmostEqual(
            result["translation_delta_m"],
            5.0,
        )

    def test_event_neighborhood(self):
        events = pd.DataFrame(
            {
                "effective_query_id": [10, 20, 40],
                "source_query_id": [10, 20, 40],
            }
        )

        before = last_event_at_or_before(
            events,
            25,
        )
        after = next_event_after(
            events,
            25,
        )

        self.assertEqual(
            before["effective_query_id"],
            20,
        )
        self.assertEqual(
            after["effective_query_id"],
            40,
        )


if __name__ == "__main__":
    unittest.main()
