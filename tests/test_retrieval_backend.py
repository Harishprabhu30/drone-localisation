from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from uavloc.retrieval import DinoV2CachedRetrievalBackend, RetrievalRepresentation


def rep(descriptors, ids, *, pooling="avgpatch"):
    descriptors = np.asarray(descriptors, dtype=np.float32)
    descriptors = descriptors / np.linalg.norm(descriptors, axis=1, keepdims=True)
    return RetrievalRepresentation(
        descriptors=descriptors,
        ids=np.asarray(ids, dtype=str),
        paths=None,
        metadata={
            "protocol": {
                "model_name": "dinov2_vits14",
                "image_size": 518,
                "crop_mode": "center_square",
                "pooling": pooling,
                "normalization": "imagenet",
                "l2_normalize": True,
                "descriptor_dtype": "float32",
            }
        },
        source_path=Path("synthetic.npz"),
    )


class DinoV2BackendTests(unittest.TestCase):
    def test_rank_matches_legacy_numpy_control(self):
        query = rep([[1.0, 0.0], [0.2, 1.0]], ["57", "228"])
        map_rep = rep(
            [[1.0, 0.0], [0.8, 0.2], [0.0, 1.0], [-1.0, 0.0]],
            ["a", "b", "c", "d"],
        )

        similarity = query.descriptors @ map_rep.descriptors.T
        legacy_order = np.argsort(-similarity, axis=1)[:, :3]
        legacy_scores = np.take_along_axis(similarity, legacy_order, axis=1)

        backend = DinoV2CachedRetrievalBackend()
        result = backend.rank_batch(query, map_rep, top_k=3)

        self.assertTrue(np.array_equal(legacy_order, result.indices))
        self.assertTrue(np.array_equal(legacy_scores, result.scores))

    def test_protocol_mismatch_is_rejected(self):
        query = rep([[1.0, 0.0]], ["1"], pooling="avgpatch")
        map_rep = rep([[1.0, 0.0]], ["a"], pooling="cls")

        backend = DinoV2CachedRetrievalBackend()
        with self.assertRaisesRegex(RuntimeError, "protocol mismatch"):
            backend.validate_pair(query, map_rep)

    def test_non_positive_top_k_is_rejected(self):
        query = rep([[1.0, 0.0]], ["1"])
        map_rep = rep([[1.0, 0.0]], ["a"])

        backend = DinoV2CachedRetrievalBackend()
        with self.assertRaisesRegex(ValueError, "top_k must be positive"):
            backend.rank_batch(query, map_rep, top_k=0)


if __name__ == "__main__":
    unittest.main()
