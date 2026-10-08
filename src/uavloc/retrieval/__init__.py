from .contracts import BatchRanking, RetrievalBackend, RetrievalRepresentation
from .dinov2_backend import DinoV2CachedRetrievalBackend
from .fusion import (
    bbox_iou,
    center_distance_m,
    cross_scale_duplicate,
    fuse_physical_regions,
    reciprocal_rank,
)

__all__ = [
    "BatchRanking",
    "DinoV2CachedRetrievalBackend",
    "RetrievalBackend",
    "RetrievalRepresentation",
    "bbox_iou",
    "center_distance_m",
    "cross_scale_duplicate",
    "fuse_physical_regions",
    "reciprocal_rank",
]
