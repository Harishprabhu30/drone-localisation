from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

from uavloc.data.trajectory_adapter import (
    SCHEMA_ID,
    TrajectorySpecError,
    load_trajectory_spec,
)


def base_spec() -> dict:
    return {
        "schema": SCHEMA_ID,
        "trajectory": {
            "id": "unit_test_flight",
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
            "id": "map_a",
            "crs": "EPSG:3346",
            "variants": {
                "768_s256": {
                    "tile_index": "tiles.csv",
                },
            },
        },
        "reference": {
            "mode": "unavailable",
        },
        "research_policy": {
            "inspect_individual_failures": True,
            "allow_method_changes_after_run": True,
        },
    }


class TrajectoryAdapterV1Tests(unittest.TestCase):
    def write_spec(self, payload: dict) -> Path:
        tmp = tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".yaml",
            delete=False,
        )
        yaml.safe_dump(payload, tmp)
        tmp.close()
        return Path(tmp.name)

    def test_assumed_signals_are_supported(self):
        path = self.write_spec(base_spec())
        spec = load_trajectory_spec(path)

        self.assertTrue(
            spec.signal("relative_altitude_m").is_assumed
        )
        self.assertEqual(
            spec.signal("gimbal_pitch_deg").value,
            -90.0,
        )
        self.assertFalse(
            spec.reference.available
        )

    def test_provider_signal_supports_future_vio_or_imu_adapter(self):
        payload = base_spec()
        payload["signals"]["gimbal_pitch_deg"] = {
            "source": "provider",
            "provider": "vio_state",
            "field": "camera_pitch_deg",
        }

        path = self.write_spec(payload)
        spec = load_trajectory_spec(path)

        signal = spec.signal("gimbal_pitch_deg")
        self.assertTrue(signal.is_dynamic)
        self.assertEqual(
            signal.config["provider"],
            "vio_state",
        )

    def test_postfreeze_reference_is_separate_from_blind_contract(self):
        payload = base_spec()
        payload["reference"] = {
            "mode": "postfreeze_optional",
            "provider": "villoc_srt",
            "path": "flight.srt",
        }

        path = self.write_spec(payload)
        spec = load_trajectory_spec(path)
        blind = spec.blind_runtime_contract()

        self.assertTrue(spec.reference.available)
        self.assertFalse(
            blind["reference"]["available_to_localization"]
        )
        self.assertNotIn(
            "path",
            blind["reference"],
        )

    def test_assumed_requires_value(self):
        payload = base_spec()
        payload["signals"]["relative_altitude_m"] = {
            "source": "assumed",
        }

        path = self.write_spec(payload)

        with self.assertRaises(TrajectorySpecError):
            load_trajectory_spec(path)

    def test_held_out_cannot_allow_method_changes(self):
        payload = base_spec()
        payload["trajectory"]["role"] = "held_out"

        path = self.write_spec(payload)

        with self.assertRaises(TrajectorySpecError):
            load_trajectory_spec(path)


if __name__ == "__main__":
    unittest.main()
