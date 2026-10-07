#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from uavloc.retrieval import DinoV2CachedRetrievalBackend


BASELINE_COMMIT = "256595bed684c3bb6fbfccf7e0faaaeaec3d02f5"
DEFAULT_REPORT = Path(
    "outputs/research_runs/retrieval_candidate_pool_v2/"
    "a1_parity/a1_dinov2_retrieval_parity.json"
)
NAMED_DIAGNOSTICS = ("57", "228")


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def legacy_rank(query_desc: np.ndarray, map_desc: np.ndarray, top_k: int):
    """Frozen control ranking copied verbatim in mathematical behavior."""
    k = min(int(top_k), map_desc.shape[0])
    similarity = query_desc @ map_desc.T
    order = np.argsort(-similarity, axis=1)[:, :k]
    scores = np.take_along_axis(similarity, order, axis=1)
    return order, scores


def protocol_summary(metadata: dict) -> dict:
    protocol = metadata.get("protocol", {})
    keep = (
        "model_name",
        "device",
        "image_size",
        "crop_mode",
        "pooling",
        "normalization",
        "l2_normalize",
        "descriptor_dtype",
        "seed",
    )
    return {key: protocol.get(key) for key in keep}


def diagnostic_rows(
    query_ids: np.ndarray,
    tile_ids: np.ndarray,
    legacy_order: np.ndarray,
    backend_order: np.ndarray,
    requested: list[str],
) -> dict:
    index = {str(qid): i for i, qid in enumerate(query_ids)}
    out = {}
    for raw in requested:
        qid = str(raw).removeprefix("q")
        if qid not in index:
            out[f"q{qid}"] = {"present": False}
            continue

        i = index[qid]
        legacy_tiles = [str(tile_ids[j]) for j in legacy_order[i]]
        backend_tiles = [str(tile_ids[j]) for j in backend_order[i]]
        out[f"q{qid}"] = {
            "present": True,
            "ordering_equal": legacy_tiles == backend_tiles,
            "legacy_topk_tile_ids": legacy_tiles,
            "backend_topk_tile_ids": backend_tiles,
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "A1 parity gate: prove the new cached DINOv2 retrieval backend "
            "preserves frozen candidate ordering."
        )
    )
    parser.add_argument("--query-cache", type=Path, required=True)
    parser.add_argument("--map-cache", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--diagnostic-query-id",
        action="append",
        default=[],
        help="Additional query ID to record in the parity report.",
    )
    parser.add_argument(
        "--baseline-commit",
        default=BASELINE_COMMIT,
        help="Git commit used as the A1 starting baseline.",
    )
    args = parser.parse_args()

    if args.top_k <= 0:
        raise ValueError("--top-k must be positive")

    backend = DinoV2CachedRetrievalBackend()
    query = backend.load_representation(args.query_cache)
    map_representation = backend.load_representation(args.map_cache)
    backend.validate_pair(query, map_representation)

    legacy_order, legacy_scores = legacy_rank(
        query.descriptors,
        map_representation.descriptors,
        args.top_k,
    )
    backend_result = backend.rank_batch(
        query,
        map_representation,
        args.top_k,
    )

    ordering_equal = np.array_equal(legacy_order, backend_result.indices)
    score_equal = np.array_equal(legacy_scores, backend_result.scores)
    max_abs_score_delta = float(
        np.max(np.abs(legacy_scores - backend_result.scores))
    ) if legacy_scores.size else 0.0

    per_query_equal = np.all(
        legacy_order == backend_result.indices,
        axis=1,
    )
    mismatch_indices = np.flatnonzero(~per_query_equal)
    mismatch_query_ids = [str(query.ids[i]) for i in mismatch_indices[:50]]

    diagnostics = list(NAMED_DIAGNOSTICS)
    diagnostics.extend(args.diagnostic_query_id)

    report = {
        "stage": "A1_RETRIEVAL_ABSTRACTION_PARITY",
        "status": (
            "PASS_A1_RETRIEVAL_PARITY"
            if ordering_equal and score_equal
            else "FAIL_A1_RETRIEVAL_PARITY"
        ),
        "created_at_utc": now_utc(),
        "baseline_commit": args.baseline_commit,
        "backend": backend.backend_metadata(),
        "inputs": {
            "query_cache": str(args.query_cache.expanduser().resolve()),
            "map_cache": str(args.map_cache.expanduser().resolve()),
            "query_count": int(len(query.ids)),
            "map_tile_count": int(len(map_representation.ids)),
            "descriptor_dim": int(query.descriptors.shape[1]),
            "top_k": int(min(args.top_k, len(map_representation.ids))),
            "query_protocol": protocol_summary(query.metadata),
            "map_protocol": protocol_summary(map_representation.metadata),
            "query_protocol_hash": query.metadata.get("protocol_hash"),
            "map_protocol_hash": map_representation.metadata.get("protocol_hash"),
            "query_checkpoint_sha256": query.metadata.get("checkpoint_sha256"),
            "map_checkpoint_sha256": map_representation.metadata.get("checkpoint_sha256"),
        },
        "parity": {
            "candidate_ordering_exact": bool(ordering_equal),
            "scores_exact": bool(score_equal),
            "max_abs_score_delta": max_abs_score_delta,
            "mismatched_query_count": int(len(mismatch_indices)),
            "mismatched_query_ids_first50": mismatch_query_ids,
        },
        "named_diagnostics": diagnostic_rows(
            query.ids,
            map_representation.ids,
            legacy_order,
            backend_result.indices,
            diagnostics,
        ),
        "scope_guarantees": {
            "descriptor_generation_changed": False,
            "query_preprocessing_changed": False,
            "map_geometry_changed": False,
            "orb_policy_changed": False,
            "minimum_confident_v2_changed": False,
            "temporal_authority_changed": False,
            "coordinates_or_oracles_used_for_ranking": False,
        },
    }

    report_path = args.report.expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("=" * 88)
    print("A1 — DINOv2 RETRIEVAL ABSTRACTION PARITY")
    print("=" * 88)
    print("status:", report["status"])
    print("queries:", report["inputs"]["query_count"])
    print("map tiles:", report["inputs"]["map_tile_count"])
    print("top-k:", report["inputs"]["top_k"])
    print("candidate ordering exact:", ordering_equal)
    print("scores exact:", score_equal)
    print("max abs score delta:", max_abs_score_delta)
    print("mismatched queries:", len(mismatch_indices))
    print("report:", report_path)

    for name, item in report["named_diagnostics"].items():
        print(name + ":", item.get("ordering_equal", "not present"))

    if report["status"] != "PASS_A1_RETRIEVAL_PARITY":
        raise RuntimeError(
            "A1 parity gate failed. Do not begin retrieval research until "
            f"candidate ordering is reproduced. See {report_path}"
        )


if __name__ == "__main__":
    main()
