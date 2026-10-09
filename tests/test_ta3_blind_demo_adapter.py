from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd
import yaml

from scripts.trajectory_adapter.ta2_adapt_traj01 import (
    canonicalize_legacy_manifest,
    parity_report,
)
from uavloc.data.trajectory_adapter import (
    load_trajectory_spec,
)


def spec_payload() -> dict:
    return {
        "schema": "uavloc.trajectory.v1",
        "trajectory": {
            "id": "villoc_blind_recorded_flight_final_001",
            "role": "blind_stress",
        },
        "source": {
            "type": "video",
            "video": {
                "path": "demo.mp4",
            },
        },
        "sampling": {
            "mode": "uniform_time",
            "rate_hz": 1.0,
        },
        "signals": {
            "relative_altitude_m": {
                "source": "assumed",
                "value": 122.0,
            },
            "gimbal_pitch_deg": {
                "source": "assumed",
                "value": -90.0,
            },
        },
        "map": {
            "id": "map",
            "crs": "EPSG:3346",
            "variants": {
                "512_s256": {
                    "tile_index": "map.csv",
                },
            },
        },
        "reference": {
            "mode": "unavailable",
        },
        "compatibility": {
            "legacy_blind_manifest": "legacy.csv",
            "expected_query_count": 3,
            "expected_query_id_start": 1,
            "expected_query_id_end": 3,
            "expected_frame_index_start": 0,
            "expected_frame_index_end": 2,
            "expected_timestamp_start_s": 0.0,
            "expected_timestamp_end_s": 2.0,
            "expected_width_px": 3840,
            "expected_height_px": 2160,
            "expected_view_assumption": "near_nadir",
        },
        "research_policy": {
            "inspect_individual_failures": True,
            "allow_method_changes_after_run": False,
        },
    }


def legacy_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "query_id": [1, 2, 3],
            "frame_index": [0, 1, 2],
            "timestamp_s": [0.0, 1.0, 2.0],
            "image_path": ["a.jpg", "b.jpg", "c.jpg"],
            "image_width": [3840, 3840, 3840],
            "image_height": [2160, 2160, 2160],
            "reference_available": [False, False, False],
            "assumed_rel_alt_m": [122.0, 122.0, 122.0],
            "assumed_gimbal_pitch_deg": [-90.0, -90.0, -90.0],
            "view_assumption": ["near_nadir", "near_nadir", "near_nadir"],
        }
    )


class TA3BlindDemoAdapterTests(unittest.TestCase):
    def make_spec(self):
        tmp = tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".yaml",
            delete=False,
        )
        yaml.safe_dump(
            spec_payload(),
            tmp,
        )
        tmp.close()
        return load_trajectory_spec(
            Path(tmp.name)
        )

    def test_blind_demo_requires_no_reference(self):
        spec = self.make_spec()
        self.assertFalse(
            spec.reference.available
        )
        self.assertEqual(
            spec.role,
            "blind_stress",
        )

    def test_blind_demo_canonical_parity(self):
        spec = self.make_spec()
        legacy = legacy_frame()

        canonical = canonicalize_legacy_manifest(
            legacy=legacy,
            spec=spec,
        )

        report = parity_report(
            legacy=legacy,
            canonical=canonical,
            spec=spec,
        )

        self.assertTrue(
            report["pass"]
        )
        self.assertFalse(
            canonical[
                "reference_available"
            ].any()
        )

    def test_blind_runtime_contract_has_no_reference_payload(self):
        spec = self.make_spec()
        contract = spec.blind_runtime_contract()

        self.assertFalse(
            contract[
                "reference"
            ][
                "available_to_localization"
            ]
        )
        self.assertEqual(
            contract[
                "reference"
            ][
                "mode"
            ],
            "unavailable",
        )
        self.assertNotIn(
            "path",
            contract[
                "reference"
            ],
        )


if __name__ == "__main__":
    unittest.main()
