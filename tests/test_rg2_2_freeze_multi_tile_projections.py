"""RG2.2 reference-free multi-tile projection generation tests."""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from uavloc.geometry import orb_subtile
from scripts.villoc.geometry.rg2_2_freeze_multi_tile_projections import (
    assert_blind_safe_columns,
    bool_series,
    canonical_csv,
    freeze_files,
    verify_config,
)


class RG22Tests(unittest.TestCase):
    def test_frozen_geometry_contract(self):
        self.assertEqual(
            orb_subtile.contract(),
            {
                "preprocess": "clahe_luma",
                "resize_long": 1024,
                "orb_nfeatures": 1800,
                "fast_threshold": 7,
                "edge_threshold": 15,
                "patch_size": 31,
                "lowe_ratio": 0.8,
                "ransac_threshold_px": 5.0,
                "min_homography_inliers": 4,
                "homography_direction": "query_pixel_to_satellite_tile_pixel",
                "query_point": "processed_image_center",
                "map_conversion": "raster_outer_edge_pixel_center_offset_0p5",
            },
        )

    def test_pixel_to_map_uses_pixel_center_offset(self):
        e, n = orb_subtile.tile_pixel_to_map(
            0.0, 0.0, (768, 768),
            100.0, 253.6, 200.0, 353.6,
        )
        self.assertAlmostEqual(e, 100.1, places=6)
        self.assertAlmostEqual(n, 353.5, places=6)

        e2, n2 = orb_subtile.tile_pixel_to_map(
            767.0, 767.0, (768, 768),
            100.0, 253.6, 200.0, 353.6,
        )
        self.assertAlmostEqual(e2, 253.5, places=6)
        self.assertAlmostEqual(n2, 200.1, places=6)

    def test_failed_features_are_retained_as_failed_geometry(self):
        q = orb_subtile.FeaturePack(False, (576, 1024), [], None, "no descriptors")
        t = orb_subtile.FeaturePack(False, (768, 768), [], None, "no descriptors")
        result = orb_subtile.verify_pair_with_geometry(q, t)
        self.assertFalse(result["homography_ok"])
        self.assertEqual(result["inliers"], 0)
        self.assertEqual(result["error"], "no descriptors")

    def test_blind_guard_allows_sampling_alignment_error_ms(self):
        frame = pd.DataFrame({
            "query_id": [1],
            "reference_available": [False],
            "sampling_alignment_error_ms": [8.0],
        })
        assert_blind_safe_columns(frame, "synthetic")

    def test_blind_guard_rejects_reference_fields(self):
        for field in (
            "eval_ref_lat", "eval_ref_lon", "error_m",
            "ground_truth_error", "oracle_tile_identity",
            "reference_x_m", "projected_error_m",
        ):
            with self.subTest(field=field):
                frame = pd.DataFrame({
                    "query_id": [1],
                    "reference_available": [False],
                    field: [0.0],
                })
                with self.assertRaisesRegex(RuntimeError, "forbidden"):
                    assert_blind_safe_columns(frame, "synthetic")

    def test_bool_series_strict(self):
        out = bool_series(pd.Series(["true", "0", "yes", "n"]), "flag")
        self.assertEqual(out.tolist(), [True, False, True, False])
        with self.assertRaisesRegex(RuntimeError, "non-boolean"):
            bool_series(pd.Series(["maybe"]), "flag")

    def test_canonical_csv_is_deterministic_and_blanks_nonfinite(self):
        rows = [{
            "query_id": 1,
            "ok": True,
            "finite": 1.2345678912,
            "nan": float("nan"),
        }]
        a = canonical_csv(rows)
        b = canonical_csv(rows)
        self.assertEqual(a, b)
        text = a.decode("utf-8")
        self.assertIn("1.234567891", text)
        self.assertTrue(text.endswith("1,true,1.234567891,\n"))

    def test_freeze_refuses_changed_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "freeze.csv"
            freeze_files({path: b"same\n"})
            freeze_files({path: b"same\n"})
            with self.assertRaisesRegex(RuntimeError, "Refusing to overwrite"):
                freeze_files({path: b"different\n"})
            self.assertEqual(path.read_bytes(), b"same\n")

    def test_config_rejects_selection_or_consensus(self):
        cfg = {
            "stage": "RG2.2",
            "map_variant": "768_s256",
            "map_crs": "EPSG:3346",
            "geometry_contract": orb_subtile.contract(),
            "generation": {
                "candidate_set": "full_overlap",
                "selection_threshold": "none",
                "consensus_enabled": False,
                "freeze_failed_pairs": True,
            },
            "scientific_boundary": {
                "reference_allowed": False,
                "reference_attachment_enabled": False,
                "srt_gps_allowed": False,
                "oracle_labels_allowed": False,
                "retrieval_reranking_enabled": False,
                "candidate_selection_enabled": False,
                "inlier_gate_enabled": False,
                "consensus_enabled": False,
                "bootstrap_enabled": False,
                "state_enabled": False,
                "learned_model_enabled": False,
            },
        }
        verify_config(cfg)
        cfg["generation"]["consensus_enabled"] = True
        with self.assertRaisesRegex(RuntimeError, "selection/consensus"):
            verify_config(cfg)

    def test_script_help_runs_without_pythonpath(self):
        script = REPO_ROOT / "scripts/villoc/geometry/rg2_2_freeze_multi_tile_projections.py"
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        result = subprocess.run(
            [sys.executable, str(script), "--help"],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        self.assertEqual(
            result.returncode, 0,
            msg=f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}",
        )
        self.assertIn("reference-free multi-tile sub-tile projection", result.stdout)


if __name__ == "__main__":
    unittest.main()
