"""CV3.1B1 bundle/coordinate/scientific-boundary tests."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.villoc.geometry.cv3_1b1_aerial_mast3r_smoke import verify_config
from scripts.villoc.geometry.cv3_1b1_prepare_smoke_bundle import verify_config as verify_bundle_config
from uavloc.geometry.aerial_mast3r_smoke import (
    json_safe,
    mast3r_image_geometry,
    processed_to_canonical_xy,
)


class CV31B1Tests(unittest.TestCase):
    def setUp(self):
        self.cfg = yaml.safe_load(
            (ROOT / "configs/research/crossview_scale_rotation_domain_cv3_1b1.yaml").read_text()
        )

    def test_contract(self):
        verify_config(self.cfg)
        verify_bundle_config(self.cfg)
        self.assertEqual(self.cfg["sample"]["query_ids"], [1,207,403])
        self.assertEqual(self.cfg["sample"]["pool_ranks"], [1,10,20])
        self.assertFalse(self.cfg["scientific_boundary"]["reference_allowed"])


    def test_json_safe_converts_nonfinite_geometry_to_null(self):
        payload = {
            "ok": 1.25,
            "nan": float("nan"),
            "posinf": float("inf"),
            "neginf": float("-inf"),
            "nested": [np.float32("nan"), np.int64(7), np.bool_(True)],
        }
        safe = json_safe(payload)
        self.assertEqual(safe["ok"], 1.25)
        self.assertIsNone(safe["nan"])
        self.assertIsNone(safe["posinf"])
        self.assertIsNone(safe["neginf"])
        self.assertEqual(safe["nested"], [None, 7, True])

    def test_query_resize_geometry(self):
        g = mast3r_image_geometry(3840,2160,size=512,square_ok=False)
        self.assertEqual((g.processed_width,g.processed_height),(512,288))
        p = processed_to_canonical_xy(np.array([[256.0,144.0]],dtype=np.float32),g)
        np.testing.assert_allclose(p[0],[1920.0,1080.0],rtol=0,atol=1e-5)

    def test_square_tile_crop_geometry(self):
        g = mast3r_image_geometry(768,768,size=512,square_ok=False)
        self.assertEqual((g.processed_width,g.processed_height),(512,384))
        self.assertEqual((g.crop_left,g.crop_top),(0,64))
        p = processed_to_canonical_xy(np.array([[256.0,192.0]],dtype=np.float32),g)
        np.testing.assert_allclose(p[0],[384.0,384.0],rtol=0,atol=1e-5)


if __name__ == "__main__":
    unittest.main()
