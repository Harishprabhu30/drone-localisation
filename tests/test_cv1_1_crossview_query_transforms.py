"""CV1.1 fixed query-transform tests."""
from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from uavloc.geometry.crossview_query_transforms import (
    rotate_processed_query,
    scale_processed_query,
)


class CV11TransformTests(unittest.TestCase):
    def test_scale_preserves_full_array_content_shape_relation(self):
        img = np.arange(20, dtype=np.uint8).reshape(4, 5)
        smaller = scale_processed_query(img, 0.5)
        larger = scale_processed_query(img, 2.0)
        self.assertEqual(smaller.shape, (2, 2))
        self.assertEqual(larger.shape, (8, 10))

    def test_identity_scale_returns_equal_copy(self):
        img = np.arange(12, dtype=np.uint8).reshape(3, 4)
        out = scale_processed_query(img, 1.0)
        self.assertTrue(np.array_equal(out, img))
        self.assertIsNot(out, img)

    def test_quarter_turns_are_exact(self):
        img = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8)
        r90 = rotate_processed_query(img, 90)
        r180 = rotate_processed_query(img, 180)
        r270 = rotate_processed_query(img, 270)
        self.assertTrue(np.array_equal(r90, np.array([[4, 1], [5, 2], [6, 3]], dtype=np.uint8)))
        self.assertTrue(np.array_equal(r180, np.array([[6, 5, 4], [3, 2, 1]], dtype=np.uint8)))
        self.assertTrue(np.array_equal(r270, np.array([[3, 6], [2, 5], [1, 4]], dtype=np.uint8)))

    def test_invalid_rotation_rejected(self):
        with self.assertRaises(ValueError):
            rotate_processed_query(np.zeros((3, 3), dtype=np.uint8), 45)

    def test_script_help_without_pythonpath(self):
        script = REPO_ROOT / "scripts/villoc/geometry/cv1_1_freeze_scale_rotation_evidence.py"
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        p = subprocess.run(
            [sys.executable, str(script), "--help"],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(p.returncode, 0, msg=p.stderr)
        self.assertIn("CV1.1 reference-free", p.stdout)


if __name__ == "__main__":
    unittest.main()
