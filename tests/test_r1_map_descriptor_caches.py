from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.villoc.retrieval.r1_2_build_map_descriptor_caches import (
    cache_index_path,
    inspect_cache_state,
    select_r12_levels,
    validate_cache_against_index,
)
from uavloc.retrieval import RetrievalRepresentation


class R1MapDescriptorCacheTests(unittest.TestCase):
    def test_selects_only_primary_missing_levels_in_order(self):
        cfg = {
            "experiment": {
                "levels": [
                    {"name": "1024_s256", "status": "existing_endpoint_reuse"},
                    {"name": "768_s256", "status": "generate"},
                    {"name": "512_s256", "status": "frozen_control_reuse"},
                    {"name": "384_s256", "status": "generate"},
                ]
            }
        }

        selected = select_r12_levels(cfg)

        self.assertEqual(
            [item["name"] for item in selected],
            ["384_s256", "768_s256"],
        )

    def test_cache_index_path_is_conventional(self):
        cache = Path(
            "s8_11b_dinov2_map_384_s256_tag.npz"
        )
        self.assertEqual(
            cache_index_path(cache),
            Path(
                "s8_11b_dinov2_map_384_s256_tag_index.csv"
            ),
        )

    def test_partial_cache_state_is_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache = root / "cache.npz"
            index = root / "cache_index.csv"
            cache.write_bytes(b"x")

            self.assertEqual(
                inspect_cache_state(
                    cache,
                    index,
                ),
                "partial_or_inconsistent",
            )

    def test_cache_validation_requires_exact_index_order(self):
        rep = RetrievalRepresentation(
            descriptors=np.asarray(
                [
                    [1.0] + [0.0] * 383,
                    [0.0, 1.0] + [0.0] * 382,
                ],
                dtype=np.float32,
            ),
            ids=np.asarray(
                ["sat_000001", "sat_000002"],
                dtype=str,
            ),
            paths=None,
            metadata={
                "cache_kind": "map_cache",
                "variant": "384_s256",
                "row_count": 2,
                "descriptor_shape": [2, 384],
                "protocol_hash": "p",
                "checkpoint_sha256": "c",
                "source_csv_sha256": "s",
                "ids_hash": "i",
                "paths_hash": "h",
                "runtime_s": 1.0,
            },
            source_path=Path("synthetic.npz"),
        )

        wrong_order = pd.DataFrame(
            {
                "tile_id": [
                    "sat_000002",
                    "sat_000001",
                ],
                "tile_path": [
                    "b.jpg",
                    "a.jpg",
                ],
            }
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "row order",
        ):
            validate_cache_against_index(
                representation=rep,
                index_df=wrong_order,
                expected_variant="384_s256",
            )


if __name__ == "__main__":
    unittest.main()
