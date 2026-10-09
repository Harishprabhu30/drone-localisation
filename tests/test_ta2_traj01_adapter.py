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
            "id": "villoc_traj01_90deg_stable120m",
            "role": "development",
        },
        "source": {
            "type": "video",
            "video": {
                "path": "flight.mp4",
            },
        },
        "sampling": {
            "mode": "uniform_time",
            "rate_hz": 1.0,
        },
        "signals": {
            "relative_altitude_m": {
                "source": "assumed",
                "value": 120.0,
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
                "768_s256": {
                    "tile_index": "map.csv",
                },
            },
        },
        "reference": {
            "mode": "postfreeze_optional",
            "provider": "villoc_srt",
            "path": "flight.srt",
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
            "assumed_rel_alt_m": [120.0, 120.0, 120.0],
            "assumed_gimbal_pitch_deg": [-90.0, -90.0, -90.0],
            "view_assumption": ["near_nadir", "near_nadir", "near_nadir"],
        }
    )


class TA2Traj01AdapterTests(unittest.TestCase):
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

    def test_canonicalize_preserves_blind_parity(self):
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
        self.assertEqual(
            canonical[
                "trajectory_id"
            ].unique().tolist(),
            [
                "villoc_traj01_90deg_stable120m"
            ],
        )
        self.assertTrue(
            (
                canonical[
                    "relative_altitude_m_source"
                ]
                == "assumed"
            ).all()
        )

    def test_reference_remains_false_in_canonical_manifest(self):
        spec = self.make_spec()
        canonical = canonicalize_legacy_manifest(
            legacy=legacy_frame(),
            spec=spec,
        )
        self.assertFalse(
            canonical[
                "reference_available"
            ].any()
        )


if __name__ == "__main__":
    unittest.main()
