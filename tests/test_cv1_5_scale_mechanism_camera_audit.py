"""CV1.5 scale mechanism/camera audit tests."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.villoc.geometry.cv1_5_scale_mechanism_camera_gsd_audit import (
    camera_report,
    sanitize_camera_metadata,
    verify_config,
)


class CV15AuditTests(unittest.TestCase):
    def setUp(self):
        path = REPO_ROOT / "configs/research/crossview_scale_rotation_domain_cv1_5.yaml"
        self.cfg = yaml.safe_load(path.read_text(encoding="utf-8"))

    def test_config_keeps_physical_gsd_claim_blocked(self):
        verify_config(self.cfg)
        self.assertFalse(self.cfg["scientific_boundary"]["physical_gsd_claim_allowed"])
        self.assertFalse(self.cfg["scientific_boundary"]["new_transform_conditions_allowed"])

    def test_camera_report_marks_missing_calibration_blockers(self):
        frame = pd.DataFrame({
            "focal_len_mm": [24.0, 24.0],
            "dzoom_ratio": [1.0, 1.0],
            "iso": [100, 100],
            "fnum": [2.8, 2.8],
            "ev": [0.0, 0.0],
            "rel_alt_m": [120.0, 120.0],
            "abs_alt_m": [300.0, 300.0],
            "gb_yaw_deg": [0.0, 1.0],
            "gb_pitch_deg": [-90.0, -90.0],
            "gb_roll_deg": [0.0, 0.0],
            "alignment_error_ms": [0.0, 0.0],
        })
        report = camera_report(frame, self.cfg)
        self.assertTrue(report["physical_gsd_claim_blocked"])
        self.assertIn("calibrated_intrinsics_unavailable", report["blockers"])
        self.assertIn("srt_focal_length_semantics_unverified", report["blockers"])

    def test_sanitized_camera_metadata_contains_no_gps(self):
        manifest = pd.DataFrame({
            "query_id": [1, 2],
            "timestamp_s": [0.0, 1.0],
        })
        srt = pd.DataFrame({
            "video_time_s": [0.0, 1.0],
            "focal_len_mm": [24.0, 24.0],
            "dzoom_ratio": [1.0, 1.0],
            "iso": [100, 100],
            "shutter": ["1/1000", "1/1000"],
            "fnum": [2.8, 2.8],
            "ev": [0.0, 0.0],
            "rel_alt_m": [120.0, 120.0],
            "abs_alt_m": [300.0, 300.0],
            "gb_yaw_deg": [0.0, 0.0],
            "gb_pitch_deg": [-90.0, -90.0],
            "gb_roll_deg": [0.0, 0.0],
            "lat": [54.7, 54.7],
            "lon": [25.2, 25.2],
        })
        out = sanitize_camera_metadata(manifest, srt, 50.0)
        self.assertNotIn("lat", out.columns)
        self.assertNotIn("lon", out.columns)
        self.assertEqual(len(out), 2)


if __name__ == "__main__":
    unittest.main()
