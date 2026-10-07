from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np


@dataclass(frozen=True)
class RetrievalRepresentation:
    """Descriptor representation loaded by a retrieval backend.

    The representation deliberately contains no geographic/oracle fields. Those
    belong to evaluation layers, not blind retrieval.
    """

    descriptors: np.ndarray
    ids: np.ndarray
    paths: np.ndarray | None
    metadata: dict[str, Any]
    source_path: Path


@dataclass(frozen=True)
class BatchRanking:
    """Top-K ranking for a batch of query descriptors."""

    query_ids: np.ndarray
    map_ids: np.ndarray
    indices: np.ndarray
    scores: np.ndarray
    retrieval_runtime_s: float


class RetrievalBackend(Protocol):
    """Narrow contract for candidate-generation backends.

    A1 intentionally keeps the contract at the representation/ranking boundary.
    Existing S8.11BC remains the frozen DINOv2 descriptor producer until a later
    gated stage deliberately abstracts image encoding.
    """

    @property
    def name(self) -> str:
        ...

    def load_representation(self, path: Path) -> RetrievalRepresentation:
        ...

    def validate_pair(
        self,
        query: RetrievalRepresentation,
        map_representation: RetrievalRepresentation,
    ) -> None:
        ...

    def rank_batch(
        self,
        query: RetrievalRepresentation,
        map_representation: RetrievalRepresentation,
        top_k: int,
    ) -> BatchRanking:
        ...

    def backend_metadata(self) -> dict[str, Any]:
        ...
