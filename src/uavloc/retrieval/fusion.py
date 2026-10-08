from __future__ import annotations

import math
from typing import Any, Iterable


def reciprocal_rank(rank: int, k: float = 60.0) -> float:
    """Standard neutral reciprocal-rank-fusion contribution."""
    if int(rank) <= 0:
        raise ValueError("rank must be positive")
    if float(k) < 0.0:
        raise ValueError("RRF k must be non-negative")
    return 1.0 / (float(k) + float(rank))


def bbox_iou(a: dict[str, Any], b: dict[str, Any]) -> float:
    """Axis-aligned IoU in the map metric CRS."""
    left = max(float(a["left_easting"]), float(b["left_easting"]))
    right = min(float(a["right_easting"]), float(b["right_easting"]))
    bottom = max(
        float(a["bottom_northing"]),
        float(b["bottom_northing"]),
    )
    top = min(float(a["top_northing"]), float(b["top_northing"]))

    width = max(0.0, right - left)
    height = max(0.0, top - bottom)
    intersection = width * height

    area_a = max(
        0.0,
        float(a["right_easting"]) - float(a["left_easting"]),
    ) * max(
        0.0,
        float(a["top_northing"]) - float(a["bottom_northing"]),
    )
    area_b = max(
        0.0,
        float(b["right_easting"]) - float(b["left_easting"]),
    ) * max(
        0.0,
        float(b["top_northing"]) - float(b["bottom_northing"]),
    )

    union = area_a + area_b - intersection
    if union <= 0.0:
        return 0.0

    return float(intersection / union)


def center_distance_m(a: dict[str, Any], b: dict[str, Any]) -> float:
    return float(
        math.hypot(
            float(a["center_easting"]) - float(b["center_easting"]),
            float(a["center_northing"]) - float(b["center_northing"]),
        )
    )


def cross_scale_duplicate(
    a: dict[str, Any],
    b: dict[str, Any],
    *,
    max_center_distance_m: float,
    min_iou: float,
) -> bool:
    """Conservative R2 physical-region duplicate rule.

    Same-scale candidates are deliberately not collapsed in R2.0/R2.1.
    Cross-scale candidates must satisfy both a center-distance gate and an
    overlap gate. No UAV/query coordinates participate.
    """
    if str(a["variant"]) == str(b["variant"]):
        return False

    if center_distance_m(a, b) > float(max_center_distance_m) + 1e-9:
        return False

    return bbox_iou(a, b) + 1e-12 >= float(min_iou)


def _candidate_sort_key(
    candidate: dict[str, Any],
    scale_order: dict[str, int],
) -> tuple:
    return (
        int(candidate["per_scale_rank"]),
        int(scale_order[str(candidate["variant"])]),
        str(candidate["tile_id"]),
    )


def _representative_key(
    candidate: dict[str, Any],
    scale_order: dict[str, int],
) -> tuple:
    return (
        -float(candidate["rrf_contribution"]),
        int(candidate["per_scale_rank"]),
        int(scale_order[str(candidate["variant"])]),
        str(candidate["tile_id"]),
    )


def fuse_physical_regions(
    candidates: Iterable[dict[str, Any]],
    *,
    scale_order: list[str],
    max_center_distance_m: float,
    min_iou: float,
    final_top_k: int,
) -> list[dict[str, Any]]:
    """Fuse per-scale ranked candidates into inspectable physical regions.

    Algorithm v1
    ------------
    1. Visit candidates by per-scale rank, then fixed scale order.
    2. A candidate may join a region only if:
       - that region does not already contain the same scale; and
       - it is a cross-scale duplicate of the region's fixed seed candidate.
    3. If several regions qualify, choose highest IoU, then nearest center.
    4. Region RRF score is the sum of member reciprocal-rank contributions.
    5. Region representative is the member with the strongest individual RRF
       contribution. Exact ties use the fixed scale order only for determinism.
    6. Sort regions by fused RRF, support-scale count, representative quality.

    The fixed seed prevents one large tile from transitively bridging several
    distinct fine-scale regions.
    """
    if int(final_top_k) <= 0:
        raise ValueError("final_top_k must be positive")

    order = {
        str(name): index
        for index, name in enumerate(scale_order)
    }

    materialized = [dict(item) for item in candidates]
    unknown = sorted(
        {
            str(item["variant"])
            for item in materialized
            if str(item["variant"]) not in order
        }
    )
    if unknown:
        raise ValueError(f"Unknown scales in candidates: {unknown}")

    materialized.sort(
        key=lambda item: _candidate_sort_key(item, order)
    )

    regions: list[dict[str, Any]] = []

    for candidate in materialized:
        candidate_variant = str(candidate["variant"])
        matches: list[tuple[float, float, int]] = []

        for region_index, region in enumerate(regions):
            member_scales = {
                str(member["variant"])
                for member in region["members"]
            }
            if candidate_variant in member_scales:
                continue

            seed = region["seed"]
            if not cross_scale_duplicate(
                candidate,
                seed,
                max_center_distance_m=max_center_distance_m,
                min_iou=min_iou,
            ):
                continue

            matches.append(
                (
                    bbox_iou(candidate, seed),
                    center_distance_m(candidate, seed),
                    region_index,
                )
            )

        if matches:
            matches.sort(
                key=lambda item: (
                    -item[0],
                    item[1],
                    item[2],
                )
            )
            chosen_index = matches[0][2]
            regions[chosen_index]["members"].append(candidate)
        else:
            regions.append(
                {
                    "seed": candidate,
                    "members": [candidate],
                }
            )

    finalized: list[dict[str, Any]] = []

    for region in regions:
        members = list(region["members"])
        representative = min(
            members,
            key=lambda item: _representative_key(item, order),
        )
        fused_score = float(
            sum(float(item["rrf_contribution"]) for item in members)
        )

        finalized.append(
            {
                "seed": region["seed"],
                "representative": representative,
                "members": members,
                "fused_rrf_score": fused_score,
                "support_scale_count": int(len(members)),
                "support_scales": [
                    str(item["variant"])
                    for item in sorted(
                        members,
                        key=lambda item: order[str(item["variant"])],
                    )
                ],
                "best_per_scale_rank": int(
                    min(int(item["per_scale_rank"]) for item in members)
                ),
            }
        )

    finalized.sort(
        key=lambda region: (
            -float(region["fused_rrf_score"]),
            -int(region["support_scale_count"]),
            int(region["representative"]["per_scale_rank"]),
            int(order[str(region["representative"]["variant"])]),
            str(region["representative"]["tile_id"]),
        )
    )

    output = finalized[: int(final_top_k)]
    for index, region in enumerate(output, start=1):
        region["fused_rank"] = index

    return output
