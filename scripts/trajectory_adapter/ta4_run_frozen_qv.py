#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from uavloc.data.trajectory_adapter import (  # noqa: E402
    FORBIDDEN_BLIND_REFERENCE_COLUMNS,
    load_trajectory_spec,
)
from scripts.villoc.retrieval.qv1_1_query_view_retrieval import (  # noqa: E402
    encode_queries,
    l2_normalize,
    load_model,
    load_npz_representation,
    rank_all,
)
from scripts.villoc.retrieval.qv1_3_fixed_budget_multiview_pool import (  # noqa: E402
    evaluate_pool,
    load_reference_xy,
    tile_centers,
)
from scripts.villoc.retrieval.qv1_4_redundancy_aware_pool import (  # noqa: E402
    build_pool,
)


ROOT = REPO_ROOT
DEFAULT_CONFIG = Path(
    "configs/research/trajectory_adapter_ta4_frozen_qv.yaml"
)


def resolve(path_like: str | Path) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else ROOT / path


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def assert_blind_manifest_safe(frame: pd.DataFrame) -> None:
    lower = {str(column).strip().lower() for column in frame.columns}
    forbidden = {str(column).lower() for column in FORBIDDEN_BLIND_REFERENCE_COLUMNS}
    leaked = sorted(lower & forbidden)
    if leaked:
        raise RuntimeError(
            f"Canonical blind manifest contains forbidden reference columns: {leaked}"
        )
    if "reference_available" not in frame.columns:
        raise RuntimeError("Canonical blind manifest missing reference_available.")
    values = (
        frame["reference_available"]
        .astype(str)
        .str.strip()
        .str.lower()
    )
    if not values.isin({"false", "0", "no", "n", "f"}).all():
        raise RuntimeError(
            "Canonical blind manifest must keep reference_available=false."
        )


def load_canonical_manifest(spec) -> tuple[pd.DataFrame, Path]:
    adapter = spec.raw.get("adapter", {})
    package_value = adapter.get("canonical_package")
    if not package_value:
        raise RuntimeError(
            f"{spec.trajectory_id}: adapter.canonical_package is required for TA4."
        )

    package_root = resolve(package_value)
    manifest_path = package_root / "canonical_blind_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)

    frame = pd.read_csv(manifest_path).copy()
    assert_blind_manifest_safe(frame)

    frame["query_id"] = pd.to_numeric(
        frame["query_id"], errors="raise"
    ).astype(int)

    if not (
        frame["trajectory_id"].astype(str)
        == spec.trajectory_id
    ).all():
        raise RuntimeError(
            "Canonical manifest trajectory_id does not match trajectory spec."
        )

    return frame, manifest_path


def save_descriptor_cache(
    *,
    path: Path,
    descriptors: np.ndarray,
    query_ids: list[str],
    image_paths: list[Path],
    meta: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        descriptors=descriptors,
        ids=np.asarray(query_ids, dtype=str),
        paths=np.asarray([str(path) for path in image_paths], dtype=str),
        meta_json=np.asarray(json.dumps(meta, indent=2)),
    )


def descriptor_sets_for_trajectory(
    *,
    spec,
    manifest: pd.DataFrame,
    output_root: Path,
    method: dict[str, Any],
    device: str,
    batch_size: int,
    force: bool,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
    query_ids = manifest["query_id"].astype(int).tolist()
    query_id_strings = [str(qid) for qid in query_ids]
    image_paths = [resolve(value) for value in manifest["image_path"].astype(str)]

    for path in image_paths:
        if not path.exists():
            raise FileNotFoundError(path)

    declared = (
        spec.raw
        .get("research_assets", {})
        .get("query_descriptor_caches", {})
    )

    desc_root = output_root / "descriptors"
    desc_root.mkdir(parents=True, exist_ok=True)

    descriptors: dict[str, np.ndarray] = {}
    provenance: dict[str, dict[str, Any]] = {}

    model = None
    torch = None

    for view in method["query_views"]:
        declared_path = declared.get(view)
        generated_path = (
            desc_root
            / (
                f"ta4_{view}_dinov2_vits14_"
                f"img{int(method['image_size'])}_avgpatch_{device}.npz"
            )
        )

        cache_path = None
        cache_origin = None

        if declared_path:
            candidate = resolve(declared_path)
            if candidate.exists() and not force:
                cache_path = candidate
                cache_origin = "declared_frozen_cache"

        if cache_path is None and generated_path.exists() and not force:
            cache_path = generated_path
            cache_origin = "ta4_reuse"

        if cache_path is not None:
            desc, ids, meta = load_npz_representation(cache_path)
            if ids.tolist() != query_id_strings:
                raise RuntimeError(
                    f"{view}: descriptor cache query IDs do not match canonical manifest."
                )
            descriptors[view] = l2_normalize(desc).astype(np.float32)
            provenance[view] = {
                "mode": cache_origin,
                "path": str(cache_path),
                "sha256": sha256_file(cache_path),
                "meta": meta,
            }
            print(f"[REUSE] {view}: {cache_path}")
            continue

        if model is None:
            model, torch = load_model(device)

        started = time.time()
        desc = encode_queries(
            paths=image_paths,
            mode=view,
            model=model,
            torch=torch,
            device=device,
            batch_size=batch_size,
            image_size=int(method["image_size"]),
        )
        runtime_s = float(time.time() - started)

        meta = {
            "stage": "TA4",
            "created_at_utc": now_utc(),
            "trajectory_id": spec.trajectory_id,
            "query_view": view,
            "model_name": method["model_name"],
            "image_size": int(method["image_size"]),
            "pooling": method["pooling"],
            "device": device,
            "query_count": len(query_ids),
            "reference_used": False,
            "runtime_s": runtime_s,
        }

        save_descriptor_cache(
            path=generated_path,
            descriptors=desc,
            query_ids=query_id_strings,
            image_paths=image_paths,
            meta=meta,
        )

        descriptors[view] = desc
        provenance[view] = {
            "mode": "ta4_generated",
            "path": str(generated_path),
            "sha256": sha256_file(generated_path),
            "meta": meta,
        }
        print(f"[WROTE] {view}: {generated_path}")

    return descriptors, provenance


def build_rankings(
    *,
    descriptor_sets: dict[str, np.ndarray],
    query_ids: list[int],
    map_desc: np.ndarray,
    map_ids: np.ndarray,
    depth: int,
) -> dict[str, pd.DataFrame]:
    rankings = {}

    for view, desc in descriptor_sets.items():
        ranking = rank_all(
            desc,
            map_desc,
            map_ids,
            depth=depth,
        )
        ranking["query_id"] = [
            int(query_ids[int(row)])
            for row in ranking["query_row"].tolist()
        ]
        ranking = ranking.drop(columns=["query_row"])
        rankings[view] = ranking

    return rankings


def freeze_rankings(
    *,
    rankings: dict[str, pd.DataFrame],
    output_root: Path,
) -> dict[str, dict[str, str]]:
    root = output_root / "blind_rankings"
    root.mkdir(parents=True, exist_ok=True)

    hashes = {}
    for view, frame in rankings.items():
        path = root / f"ta4_{view}_top20.csv"
        frame.to_csv(path, index=False)
        hashes[view] = {
            "path": str(path),
            "sha256": sha256_file(path),
        }

    return hashes


def freeze_pool(
    *,
    rankings: dict[str, pd.DataFrame],
    query_ids: list[int],
    centers: dict[str, tuple[float, float]],
    method: dict[str, Any],
    output_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    rows = []
    stats_rows = []

    for qid in query_ids:
        selected, stats = build_pool(
            qid=int(qid),
            rankings=rankings,
            centers=centers,
            policy=str(method["policy"]),
            budget=int(method["final_budget"]),
            center_source_depth=int(method["center_source_depth"]),
            alternate_source_depth=int(method["alternate_source_depth"]),
            radius_m=float(method["spatial_redundancy_radius_m"]),
        )

        stats_rows.append({"query_id": int(qid), **stats})

        for pool_rank, item in enumerate(selected, start=1):
            rows.append({**item, "pool_rank": int(pool_rank)})

    pool = pd.DataFrame(rows)
    stats = pd.DataFrame(stats_rows)

    counts = pool.groupby("query_id").size()
    if not (counts == int(method["final_budget"])).all():
        raise RuntimeError("TA4 produced a non-fixed candidate budget.")

    top1 = pool[pool["pool_rank"] == 1]
    if not (
        (top1["source_view"] == "center_square")
        & (top1["source_rank"] == 1)
    ).all():
        raise RuntimeError("TA4 failed to preserve center-square Top1.")

    root = output_root / "blind_pool"
    root.mkdir(parents=True, exist_ok=True)
    pool_path = root / "ta4_frozen_qv_pool.csv"
    stats_path = root / "ta4_frozen_qv_build_stats.csv"

    pool.to_csv(pool_path, index=False)
    stats.to_csv(stats_path, index=False)

    return pool, stats, {
        "pool_path": str(pool_path),
        "pool_sha256": sha256_file(pool_path),
        "build_stats_path": str(stats_path),
        "build_stats_sha256": sha256_file(stats_path),
    }


def jaccard(a: set[str], b: set[str]) -> float:
    union = a | b
    if not union:
        return 1.0
    return float(len(a & b) / len(union))


def summarize(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p95": None,
            "max": None,
        }
    arr = np.asarray(values, dtype=float)
    return {
        "count": int(len(arr)),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p95": float(np.quantile(arr, 0.95)),
        "max": float(arr.max()),
    }


def blind_behavior_report(
    *,
    rankings: dict[str, pd.DataFrame],
    pool: pd.DataFrame,
    build_stats: pd.DataFrame,
    centers: dict[str, tuple[float, float]],
    query_ids: list[int],
) -> dict[str, Any]:
    center_sets = {
        int(qid): set(group["tile_id"].astype(str))
        for qid, group in rankings["center_square"].groupby("query_id")
    }

    view_overlap = {}
    for view in ("left_square", "right_square", "resize_square"):
        by_q = {
            int(qid): set(group["tile_id"].astype(str))
            for qid, group in rankings[view].groupby("query_id")
        }
        vals = [
            jaccard(center_sets[qid], by_q[qid])
            for qid in query_ids
        ]
        view_overlap[view] = summarize(vals)

    pool_sets = {
        int(qid): set(group["tile_id"].astype(str))
        for qid, group in pool.groupby("query_id")
    }

    consecutive_jaccard = []
    top1_jump = []

    top1_lookup = (
        pool[pool["pool_rank"] == 1]
        .set_index("query_id")["tile_id"]
        .astype(str)
        .to_dict()
    )

    for previous, current in zip(query_ids[:-1], query_ids[1:]):
        consecutive_jaccard.append(
            jaccard(pool_sets[int(previous)], pool_sets[int(current)])
        )

        a = centers[top1_lookup[int(previous)]]
        b = centers[top1_lookup[int(current)]]
        top1_jump.append(
            float(np.hypot(a[0] - b[0], a[1] - b[1]))
        )

    source_counts = (
        pool["source_view"]
        .value_counts()
        .to_dict()
    )
    n_queries = len(query_ids)

    source_mean = {
        str(source): float(count / n_queries)
        for source, count in source_counts.items()
    }

    return {
        "query_count": int(n_queries),
        "fixed_candidate_budget": int(
            pool.groupby("query_id").size().iloc[0]
        ),
        "source_candidate_counts": {
            str(k): int(v)
            for k, v in source_counts.items()
        },
        "mean_candidates_per_query_by_source": source_mean,
        "construction": {
            "mean_center_unique_before_fill": float(
                build_stats["center_unique_before_fill"].mean()
            ),
            "median_center_unique_before_fill": float(
                build_stats["center_unique_before_fill"].median()
            ),
            "mean_center_redundant_dropped": float(
                build_stats["center_redundant_dropped"].mean()
            ),
            "mean_alternate_added": float(
                build_stats["alternate_added"].mean()
            ),
            "mean_center_refilled": float(
                build_stats["center_refilled"].mean()
            ),
        },
        "center_vs_view_top20_jaccard": view_overlap,
        "consecutive_final_pool_jaccard": summarize(consecutive_jaccard),
        "center_top1_map_jump_m": summarize(top1_jump),
    }


def compare_pool_parity(
    *,
    generated: pd.DataFrame,
    expected_path: Path,
) -> dict[str, Any]:
    expected = pd.read_csv(expected_path).copy()

    columns = [
        "query_id",
        "pool_rank",
        "tile_id",
        "source_view",
        "source_rank",
        "selection_reason",
    ]

    for frame in (generated, expected):
        missing = set(columns) - set(frame.columns)
        if missing:
            raise RuntimeError(
                f"Pool parity file missing columns: {sorted(missing)}"
            )

    lhs = (
        generated[columns]
        .sort_values(["query_id", "pool_rank"])
        .reset_index(drop=True)
        .astype(
            {
                "query_id": int,
                "pool_rank": int,
                "tile_id": str,
                "source_view": str,
                "source_rank": int,
                "selection_reason": str,
            }
        )
    )
    rhs = (
        expected[columns]
        .sort_values(["query_id", "pool_rank"])
        .reset_index(drop=True)
        .astype(
            {
                "query_id": int,
                "pool_rank": int,
                "tile_id": str,
                "source_view": str,
                "source_rank": int,
                "selection_reason": str,
            }
        )
    )

    exact = lhs.equals(rhs)

    mismatch_count = None
    if len(lhs) == len(rhs):
        mismatch_count = int((lhs != rhs).any(axis=1).sum())

    return {
        "expected_path": str(expected_path),
        "expected_sha256": sha256_file(expected_path),
        "row_count_generated": int(len(lhs)),
        "row_count_expected": int(len(rhs)),
        "exact": bool(exact),
        "mismatch_rows": mismatch_count,
    }


def optional_postfreeze_evaluation(
    *,
    spec,
    pool: pd.DataFrame,
    map_index: pd.DataFrame,
    blind_hash_before: str,
    pool_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    if not spec.reference.available:
        return {
            "performed": False,
            "reason": "reference_unavailable",
        }

    prepared = spec.reference.config.get("prepared_attachment")
    if not prepared:
        return {
            "performed": False,
            "reason": "no_prepared_reference_attachment_declared",
        }

    reference_path = resolve(prepared)
    if not reference_path.exists():
        raise FileNotFoundError(reference_path)

    if sha256_file(pool_path) != blind_hash_before:
        raise RuntimeError(
            "Blind pool changed before post-freeze reference evaluation."
        )

    reference = load_reference_xy(reference_path)
    query_eval, summary = evaluate_pool(
        pool,
        map_index,
        reference,
    )

    eval_root = output_root / "postfreeze_evaluation"
    eval_root.mkdir(parents=True, exist_ok=True)
    query_eval_path = eval_root / "ta4_query_eval.csv"
    query_eval.to_csv(query_eval_path, index=False)

    if sha256_file(pool_path) != blind_hash_before:
        raise RuntimeError(
            "Blind pool changed during post-freeze reference evaluation."
        )

    return {
        "performed": True,
        "reference_path": str(reference_path),
        "reference_sha256": sha256_file(reference_path),
        "summary": summary,
        "query_eval_path": str(query_eval_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "TA4: trajectory-independent frozen QV1.4 candidate-generation runner."
        )
    )
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--force-descriptors", action="store_true")
    args = parser.parse_args()

    spec = load_trajectory_spec(resolve(args.trajectory))
    cfg = yaml.safe_load(resolve(args.config).read_text())
    method = cfg["frozen_method"]

    if args.output_root is None:
        output_root = resolve(
            Path("outputs/research_runs/trajectory_adapter_v1/ta4")
            / spec.trajectory_id
        )
    else:
        output_root = resolve(args.output_root)

    output_root.mkdir(parents=True, exist_ok=True)

    manifest, manifest_path = load_canonical_manifest(spec)
    query_ids = manifest["query_id"].astype(int).tolist()

    variant_name = str(method["map_variant"])
    variants = spec.raw["map"]["variants"]
    if variant_name not in variants:
        raise RuntimeError(
            f"{spec.trajectory_id}: required map variant {variant_name!r} "
            "is not declared by the trajectory spec."
        )

    variant = variants[variant_name]
    map_index_path = resolve(variant["tile_index"])
    map_cache_path = resolve(variant["descriptor_cache"])

    if not map_index_path.exists():
        raise FileNotFoundError(map_index_path)
    if not map_cache_path.exists():
        raise FileNotFoundError(map_cache_path)

    map_index = pd.read_csv(map_index_path).copy()
    map_index["tile_id"] = map_index["tile_id"].astype(str)
    centers = tile_centers(map_index)

    map_desc, map_ids, map_meta = load_npz_representation(map_cache_path)
    map_desc = l2_normalize(map_desc).astype(np.float32)

    descriptor_sets, descriptor_provenance = descriptor_sets_for_trajectory(
        spec=spec,
        manifest=manifest,
        output_root=output_root,
        method=method,
        device=args.device,
        batch_size=int(args.batch_size),
        force=bool(args.force_descriptors),
    )

    rankings = build_rankings(
        descriptor_sets=descriptor_sets,
        query_ids=query_ids,
        map_desc=map_desc,
        map_ids=map_ids,
        depth=int(method["ranking_depth"]),
    )

    ranking_freeze = freeze_rankings(
        rankings=rankings,
        output_root=output_root,
    )

    pool, build_stats, pool_freeze = freeze_pool(
        rankings=rankings,
        query_ids=query_ids,
        centers=centers,
        method=method,
        output_root=output_root,
    )

    pool_path = Path(pool_freeze["pool_path"])
    blind_pool_hash = pool_freeze["pool_sha256"]

    blind_behavior = blind_behavior_report(
        rankings=rankings,
        pool=pool,
        build_stats=build_stats,
        centers=centers,
        query_ids=query_ids,
    )

    parity = None
    expected_pool = (
        spec.raw
        .get("research_assets", {})
        .get("parity_qv1_4_pool")
    )

    if expected_pool:
        parity = compare_pool_parity(
            generated=pool,
            expected_path=resolve(expected_pool),
        )
        if (
            cfg["development_parity"][
                "require_exact_pool_parity_when_declared"
            ]
            and not parity["exact"]
        ):
            raise RuntimeError(
                "TA4 failed exact frozen QV1.4 pool parity:\n"
                + json.dumps(parity, indent=2)
            )

    evaluation = optional_postfreeze_evaluation(
        spec=spec,
        pool=pool,
        map_index=map_index,
        blind_hash_before=blind_pool_hash,
        pool_path=pool_path,
        output_root=output_root,
    )

    expected_summary = (
        spec.raw
        .get("research_assets", {})
        .get("expected_qv1_4_summary")
    )
    summary_parity = None

    if expected_summary:
        if not evaluation.get("performed"):
            raise RuntimeError(
                "Expected QV1.4 accuracy summary declared, but post-freeze evaluation "
                "was not performed."
            )
        measured = evaluation["summary"]
        checks = {
            key: int(measured[key]) == int(value)
            for key, value in expected_summary.items()
        }
        summary_parity = {
            "expected": expected_summary,
            "measured": {
                key: measured[key]
                for key in expected_summary
            },
            "checks": checks,
            "pass": bool(all(checks.values())),
        }
        if (
            cfg["development_parity"][
                "require_expected_summary_when_declared"
            ]
            and not summary_parity["pass"]
        ):
            raise RuntimeError(
                "TA4 failed frozen QV1.4 accuracy-summary parity:\n"
                + json.dumps(summary_parity, indent=2)
            )

    if sha256_file(pool_path) != blind_pool_hash:
        raise RuntimeError(
            "TA4 blind pool changed after downstream diagnostics/evaluation."
        )

    status = (
        "PASS_TA4_FROZEN_QV_WITH_REFERENCE"
        if evaluation.get("performed")
        else "PASS_TA4_FROZEN_QV_BLIND_STRESS"
    )

    report = {
        "stage": "TA4",
        "status": status,
        "created_at_utc": now_utc(),
        "trajectory": {
            "id": spec.trajectory_id,
            "role": spec.role,
            "spec_path": str(args.trajectory),
            "canonical_manifest_path": str(manifest_path),
            "query_count": int(len(query_ids)),
        },
        "frozen_method": method,
        "map": {
            "variant": variant_name,
            "tile_index": str(map_index_path),
            "descriptor_cache": str(map_cache_path),
            "descriptor_cache_sha256": sha256_file(map_cache_path),
            "descriptor_meta": map_meta,
        },
        "query_descriptors": descriptor_provenance,
        "blind_ranking_freeze": ranking_freeze,
        "blind_pool_freeze": pool_freeze,
        "blind_behavior": blind_behavior,
        "development_pool_parity": parity,
        "postfreeze_evaluation": evaluation,
        "development_summary_parity": summary_parity,
        "scope_guarantees": {
            "same_frozen_candidate_method_across_trajectories": True,
            "map_variant": variant_name,
            "fixed_candidate_budget": int(method["final_budget"]),
            "orb_run": False,
            "bootstrap_run": False,
            "state_run": False,
            "reference_used_before_blind_pool_freeze": False,
            "accuracy_claim_allowed": bool(evaluation.get("performed")),
        },
    }

    report_path = output_root / "ta4_frozen_qv_trajectory_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, default=str),
        encoding="utf-8",
    )

    print("=" * 116)
    print("TA4 — TRAJECTORY-INDEPENDENT FROZEN QV CANDIDATE GENERATOR")
    print("=" * 116)
    print("trajectory:", spec.trajectory_id)
    print("role:", spec.role)
    print("queries:", len(query_ids))
    print("map variant:", variant_name)
    print("candidate budget:", int(method["final_budget"]))
    print(
        "mean center unique:",
        f"{blind_behavior['construction']['mean_center_unique_before_fill']:.2f}",
    )
    print(
        "mean alternates added:",
        f"{blind_behavior['construction']['mean_alternate_added']:.2f}",
    )
    print(
        "consecutive pool Jaccard median:",
        blind_behavior["consecutive_final_pool_jaccard"]["median"],
    )
    print(
        "center Top1 jump median m:",
        blind_behavior["center_top1_map_jump_m"]["median"],
    )
    print("reference evaluation:", bool(evaluation.get("performed")))

    if parity is not None:
        print("exact frozen QV1.4 pool parity:", parity["exact"])

    if evaluation.get("performed"):
        s = evaluation["summary"]
        print(
            "accuracy:",
            {
                "contain_r20": s["contain_r20_hits"],
                "le40_r20": s["le40_r20_hits"],
                "le80_r20": s["le80_r20_hits"],
                "top1_contains": s["top1_contains_hits"],
            },
        )

    print()
    print("STATUS:", status)
    print("report:", report_path)


if __name__ == "__main__":
    main()
