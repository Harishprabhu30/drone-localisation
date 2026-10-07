from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .contracts import BatchRanking, RetrievalRepresentation


PROTOCOL_KEYS = (
    "model_name",
    "image_size",
    "crop_mode",
    "pooling",
    "normalization",
    "l2_normalize",
    "descriptor_dtype",
)


class DinoV2CachedRetrievalBackend:
    """Frozen DINOv2 cache + cosine-ranking backend.

    This backend preserves the existing Villoc control behavior:

        float32 L2-normalized descriptors
        -> matrix dot product (cosine similarity)
        -> np.argsort(-similarity)
        -> Top-K

    It intentionally does not load coordinates, oracles, GPS, SRT, or reference
    trajectories. Descriptor construction remains owned by the existing S8.11BC
    cache builder during A1.
    """

    @property
    def name(self) -> str:
        return "dinov2_cached_cosine_v1"

    @staticmethod
    def _protocol_signature(metadata: dict[str, Any]) -> dict[str, Any]:
        protocol = metadata.get("protocol", {})
        return {key: protocol.get(key) for key in PROTOCOL_KEYS}

    @staticmethod
    def _decode_meta(raw: np.ndarray | Any) -> dict[str, Any]:
        if hasattr(raw, "item"):
            raw = raw.item()
        return json.loads(str(raw))

    def load_representation(self, path: Path) -> RetrievalRepresentation:
        path = Path(path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(path)

        with np.load(path, allow_pickle=False) as data:
            required = {"descriptors", "ids", "meta_json"}
            missing = required - set(data.files)
            if missing:
                raise RuntimeError(
                    f"{path}: missing cache keys {sorted(missing)}"
                )

            descriptors = data["descriptors"].astype(np.float32)
            ids = data["ids"].astype(str)
            paths = data["paths"].astype(str) if "paths" in data.files else None
            metadata = self._decode_meta(data["meta_json"])

        representation = RetrievalRepresentation(
            descriptors=descriptors,
            ids=ids,
            paths=paths,
            metadata=metadata,
            source_path=path,
        )
        self._validate_representation(representation)
        return representation

    @staticmethod
    def _validate_representation(rep: RetrievalRepresentation) -> None:
        descriptors = rep.descriptors
        ids = rep.ids

        if descriptors.ndim != 2:
            raise RuntimeError(
                f"{rep.source_path}: descriptors must be 2-D; "
                f"got shape={descriptors.shape}"
            )
        if descriptors.shape[0] != len(ids):
            raise RuntimeError(
                f"{rep.source_path}: descriptor/ID row mismatch "
                f"({descriptors.shape[0]} != {len(ids)})"
            )
        if len(set(ids.tolist())) != len(ids):
            raise RuntimeError(f"{rep.source_path}: duplicate IDs")
        if not np.isfinite(descriptors).all():
            raise RuntimeError(f"{rep.source_path}: non-finite descriptors")
        if rep.paths is not None and len(rep.paths) != len(ids):
            raise RuntimeError(
                f"{rep.source_path}: path/ID row mismatch "
                f"({len(rep.paths)} != {len(ids)})"
            )

    def validate_pair(
        self,
        query: RetrievalRepresentation,
        map_representation: RetrievalRepresentation,
    ) -> None:
        self._validate_representation(query)
        self._validate_representation(map_representation)

        q_desc = query.descriptors
        m_desc = map_representation.descriptors

        if q_desc.shape[1] != m_desc.shape[1]:
            raise RuntimeError(
                "Query/map descriptor dimensions do not match: "
                f"{q_desc.shape[1]} != {m_desc.shape[1]}"
            )

        q_sig = self._protocol_signature(query.metadata)
        m_sig = self._protocol_signature(map_representation.metadata)
        if q_sig != m_sig:
            raise RuntimeError(
                "Query/map DINO protocol mismatch.\n"
                f"query={q_sig}\nmap={m_sig}"
            )

        if not bool(q_sig.get("l2_normalize")):
            raise RuntimeError("Query descriptors are not marked L2 normalized.")
        if not bool(m_sig.get("l2_normalize")):
            raise RuntimeError("Map descriptors are not marked L2 normalized.")

        q_norm = np.linalg.norm(q_desc, axis=1)
        m_norm = np.linalg.norm(m_desc, axis=1)
        if not np.allclose(q_norm, 1.0, atol=1e-3):
            raise RuntimeError("Query descriptor norms are not approximately 1.")
        if not np.allclose(m_norm, 1.0, atol=1e-3):
            raise RuntimeError("Map descriptor norms are not approximately 1.")

    def rank_batch(
        self,
        query: RetrievalRepresentation,
        map_representation: RetrievalRepresentation,
        top_k: int,
    ) -> BatchRanking:
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        self.validate_pair(query, map_representation)

        k = min(int(top_k), len(map_representation.ids))

        # Deliberately identical to the frozen control implementation.
        similarity = query.descriptors @ map_representation.descriptors.T
        order = np.argsort(-similarity, axis=1)[:, :k]
        scores = np.take_along_axis(similarity, order, axis=1)

        return BatchRanking(
            query_ids=query.ids.copy(),
            map_ids=map_representation.ids.copy(),
            indices=order,
            scores=scores,
        )

    def backend_metadata(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "representation": "precomputed_dinov2_float32_l2_descriptors",
            "similarity": "cosine_dot_product_on_l2_normalized_descriptors",
            "ordering": "numpy_argsort_descending",
            "coordinates_used": False,
            "oracle_used": False,
            "gps_used": False,
            "srt_used": False,
            "reference_used": False,
        }
