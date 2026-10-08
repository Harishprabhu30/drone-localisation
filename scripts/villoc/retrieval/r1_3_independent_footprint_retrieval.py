#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from uavloc.retrieval import DinoV2CachedRetrievalBackend


ROOT = Path.cwd().resolve()

DEFAULT_PROTOCOL = Path(
    "configs/research/retrieval_candidate_pool_v2_r1.yaml"
)
DEFAULT_R12_REPORT = Path(
    "outputs/research_runs/retrieval_candidate_pool_v2/"
    "r1_map_pyramid/reports/"
    "r1_2_map_descriptor_cache_build_and_validation.json"
)
DEFAULT_OUTPUT_ROOT = Path(
    "outputs/research_runs/retrieval_candidate_pool_v2/"
    "r1_map_pyramid/retrieval/r1_3"
)

VARIANT_ORDER = (
    "384_s256",
    "512_s256",
    "768_s256",
    "1024_s256",
)
RECALL_KS = (1, 5, 20, 50, 100)
PRIMARY_TOP_K = 20
SPATIAL_NMS_RADII_M = (51.2, 102.4)


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve(path_like: str | Path) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else ROOT / path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def first_true_rank(mask: np.ndarray) -> int | None:
    hits = np.flatnonzero(mask)
    return None if len(hits) == 0 else int(hits[0] + 1)


def spatial_nms_count(
    centers_xy: np.ndarray,
    radius_m: float,
) -> int:
    """Rank-ordered greedy center NMS used only after retrieval for evaluation."""
    selected: list[np.ndarray] = []

    for center in centers_xy:
        if not selected:
            selected.append(center)
            continue

        distances = [
            float(np.linalg.norm(center - other))
            for other in selected
        ]

        if all(distance > radius_m + 1e-9 for distance in distances):
            selected.append(center)

    return len(selected)


def pairwise_spread(centers_xy: np.ndarray) -> dict[str, float | None]:
    if len(centers_xy) == 0:
        return {
            "mean_m": None,
            "median_m": None,
            "p95_m": None,
            "centroid_rms_radius_m": None,
        }

    if len(centers_xy) == 1:
        return {
            "mean_m": 0.0,
            "median_m": 0.0,
            "p95_m": 0.0,
            "centroid_rms_radius_m": 0.0,
        }

    deltas = centers_xy[:, None, :] - centers_xy[None, :, :]
    distances = np.sqrt(np.sum(deltas * deltas, axis=2))
    upper = distances[np.triu_indices(len(centers_xy), k=1)]

    centroid = centers_xy.mean(axis=0)
    radii = np.sqrt(
        np.sum((centers_xy - centroid) ** 2, axis=1)
    )

    return {
        "mean_m": float(np.mean(upper)),
        "median_m": float(np.median(upper)),
        "p95_m": float(np.percentile(upper, 95)),
        "centroid_rms_radius_m": float(
            np.sqrt(np.mean(radii ** 2))
        ),
    }


def point_containment_mask(
    x: float,
    y: float,
    tile_index: pd.DataFrame,
) -> np.ndarray:
    """Match the historical S8.10B point-in-tile containment convention."""
    return (
        (tile_index["left_easting"].to_numpy(dtype=float) <= x)
        & (x <= tile_index["right_easting"].to_numpy(dtype=float))
        & (tile_index["bottom_northing"].to_numpy(dtype=float) <= y)
        & (y <= tile_index["top_northing"].to_numpy(dtype=float))
    )


def summarize_numeric(series: pd.Series) -> dict[str, float | None]:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return {
            "mean": None,
            "median": None,
            "p95": None,
        }

    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
    }


def add_flat_stats(
    target: dict,
    prefix: str,
    series: pd.Series,
) -> None:
    stats = summarize_numeric(series)
    for name, value in stats.items():
        target[f"{prefix}_{name}"] = value


def validate_query_manifest(
    query_manifest: pd.DataFrame,
    query_ids: np.ndarray,
) -> pd.DataFrame:
    required = {"query_id", "easting", "northing"}
    missing = required - set(query_manifest.columns)

    if missing:
        raise RuntimeError(
            f"Query manifest missing evaluation columns: {sorted(missing)}"
        )

    out = query_manifest.copy()
    out["query_id"] = out["query_id"].astype(str)

    if out["query_id"].duplicated().any():
        raise RuntimeError("Query manifest contains duplicate query IDs.")

    indexed = out.set_index("query_id")
    wanted = query_ids.astype(str)

    missing_ids = [
        qid
        for qid in wanted
        if qid not in indexed.index
    ]
    if missing_ids:
        raise RuntimeError(
            "Query manifest missing cache IDs: "
            + ", ".join(missing_ids[:20])
        )

    return indexed.loc[wanted].reset_index()


def validate_tile_index(
    tile_index: pd.DataFrame,
    map_ids: np.ndarray,
    variant: str,
) -> pd.DataFrame:
    required = {
        "tile_id",
        "left_easting",
        "bottom_northing",
        "right_easting",
        "top_northing",
        "center_easting",
        "center_northing",
    }
    missing = required - set(tile_index.columns)

    if missing:
        raise RuntimeError(
            f"{variant}: tile index missing columns {sorted(missing)}"
        )

    out = tile_index.copy()
    out["tile_id"] = out["tile_id"].astype(str)

    expected = map_ids.astype(str)
    actual = out["tile_id"].to_numpy()

    if len(actual) != len(expected):
        raise RuntimeError(
            f"{variant}: tile-index/cache row mismatch "
            f"({len(actual)} != {len(expected)})"
        )

    if not np.array_equal(actual, expected):
        mismatch = np.flatnonzero(actual != expected)
        raise RuntimeError(
            f"{variant}: tile-index/cache ID order mismatch; "
            f"first rows={mismatch[:10].tolist()}"
        )

    return out.reset_index(drop=True)


def resolve_query_manifest_path(
    query_metadata: dict,
    explicit: Path | None,
) -> Path:
    if explicit is not None:
        path = resolve(explicit)
        if not path.exists():
            raise FileNotFoundError(path)
        return path

    source = query_metadata.get("source_csv")
    if not source:
        raise RuntimeError(
            "Query-cache metadata has no source_csv. "
            "Pass --query-manifest explicitly."
        )

    path = Path(str(source))
    if not path.is_absolute():
        path = ROOT / path

    if not path.exists():
        raise FileNotFoundError(
            f"Query-cache source manifest missing: {path}. "
            "Pass --query-manifest explicitly."
        )

    return path


def build_variant_specs(cfg: dict) -> dict[str, dict]:
    by_name = {
        str(level["name"]): level
        for level in cfg["experiment"]["levels"]
    }

    missing = [
        name
        for name in VARIANT_ORDER
        if name not in by_name
    ]
    if missing:
        raise RuntimeError(
            f"R1.3 protocol missing variants: {missing}"
        )

    return {
        name: by_name[name]
        for name in VARIANT_ORDER
    }


def evaluate_variant(
    *,
    variant: str,
    ranking,
    map_representation,
    query_eval_manifest: pd.DataFrame,
    tile_index: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    query_ids = ranking.query_ids.astype(str)
    map_ids = ranking.map_ids.astype(str)

    tiles = validate_tile_index(
        tile_index,
        map_ids,
        variant,
    )

    tile_centers = tiles[
        ["center_easting", "center_northing"]
    ].to_numpy(dtype=float)

    candidate_rows: list[dict] = []
    query_rows: list[dict] = []

    for i, qid in enumerate(query_ids):
        qrow = query_eval_manifest.iloc[i]
        qx = float(qrow["easting"])
        qy = float(qrow["northing"])

        containing_full = point_containment_mask(
            qx,
            qy,
            tiles,
        )

        ranked_indices = ranking.indices[i]
        ranked_scores = ranking.scores[i]
        ranked_centers = tile_centers[ranked_indices]
        ranked_containing = containing_full[ranked_indices]

        ranked_errors = np.sqrt(
            np.sum(
                (
                    ranked_centers
                    - np.asarray([qx, qy], dtype=float)
                ) ** 2,
                axis=1,
            )
        )

        first_containing = first_true_rank(
            ranked_containing
        )
        first_le40 = first_true_rank(
            ranked_errors <= 40.0
        )
        first_le80 = first_true_rank(
            ranked_errors <= 80.0
        )

        top20_n = min(PRIMARY_TOP_K, len(ranked_indices))
        top20_centers = ranked_centers[:top20_n]
        spread = pairwise_spread(top20_centers)

        row = {
            "variant": variant,
            "query_id": qid,
            "top1_tile_id": str(
                map_ids[ranked_indices[0]]
            ),
            "top1_score": float(ranked_scores[0]),
            "top1_center_error_m": float(ranked_errors[0]),
            "top1_contains_query": bool(ranked_containing[0]),
            "containing_tile_count_full_map": int(
                containing_full.sum()
            ),
            "first_containing_rank": (
                float(first_containing)
                if first_containing is not None
                else np.nan
            ),
            "first_center_le40_rank": (
                float(first_le40)
                if first_le40 is not None
                else np.nan
            ),
            "first_center_le80_rank": (
                float(first_le80)
                if first_le80 is not None
                else np.nan
            ),
            "top20_contains_query": bool(
                np.any(ranked_containing[:top20_n])
            ),
            "top20_has_center_le40": bool(
                np.any(ranked_errors[:top20_n] <= 40.0)
            ),
            "top20_has_center_le80": bool(
                np.any(ranked_errors[:top20_n] <= 80.0)
            ),
            "top20_containing_candidate_count": int(
                ranked_containing[:top20_n].sum()
            ),
            "top20_center_error_min_m": float(
                np.min(ranked_errors[:top20_n])
            ),
            "top20_center_error_median_m": float(
                np.median(ranked_errors[:top20_n])
            ),
            "top20_pairwise_center_mean_m": spread["mean_m"],
            "top20_pairwise_center_median_m": spread["median_m"],
            "top20_pairwise_center_p95_m": spread["p95_m"],
            "top20_centroid_rms_radius_m": spread[
                "centroid_rms_radius_m"
            ],
        }

        for radius in SPATIAL_NMS_RADII_M:
            key = str(radius).replace(".", "p")
            row[
                f"top20_spatial_nms_count_r{key}m"
            ] = spatial_nms_count(
                top20_centers,
                radius,
            )

        for k in RECALL_KS:
            kk = min(k, len(ranked_indices))

            row[
                f"containment_recall_at_{k}"
            ] = bool(
                np.any(ranked_containing[:kk])
            )
            row[
                f"center_le40_recall_at_{k}"
            ] = bool(
                np.any(ranked_errors[:kk] <= 40.0)
            )
            row[
                f"center_le80_recall_at_{k}"
            ] = bool(
                np.any(ranked_errors[:kk] <= 80.0)
            )

        for col in (
            "scene",
            "scene_class",
            "scene_label",
            "s7_scene",
        ):
            if col in query_eval_manifest.columns:
                row[col] = qrow[col]

        query_rows.append(row)

        for rank_pos, (
            map_index,
            score,
            error_m,
            contains,
        ) in enumerate(
            zip(
                ranked_indices,
                ranked_scores,
                ranked_errors,
                ranked_containing,
            ),
            start=1,
        ):
            candidate_rows.append(
                {
                    "variant": variant,
                    "query_id": qid,
                    "rank": rank_pos,
                    "tile_id": str(map_ids[map_index]),
                    "score": float(score),
                    "contains_query_eval_only": bool(contains),
                    "center_error_m_eval_only": float(error_m),
                    "center_easting_eval_only": float(
                        tile_centers[map_index, 0]
                    ),
                    "center_northing_eval_only": float(
                        tile_centers[map_index, 1]
                    ),
                }
            )

    query_df = pd.DataFrame(query_rows)
    candidate_df = pd.DataFrame(candidate_rows)

    summary = {
        "variant": variant,
        "query_count": int(len(query_df)),
        "map_tile_count": int(len(map_ids)),
        "descriptor_dim": int(
            map_representation.descriptors.shape[1]
        ),
        "retrieval_runtime_s": float(
            ranking.retrieval_runtime_s
        ),
        "retrieval_runtime_ms_per_query": float(
            1000.0
            * ranking.retrieval_runtime_s
            / len(query_df)
        ),
        "map_cache_size_bytes": int(
            map_representation.source_path.stat().st_size
        ),
        "map_cache_size_mib": float(
            map_representation.source_path.stat().st_size
            / (1024.0 * 1024.0)
        ),
        "top1_contains_query_rate": float(
            query_df["top1_contains_query"].mean()
        ),
        "top1_center_le40_rate": float(
            (query_df["top1_center_error_m"] <= 40.0).mean()
        ),
        "top1_center_le80_rate": float(
            (query_df["top1_center_error_m"] <= 80.0).mean()
        ),
        "top20_contains_query_rate": float(
            query_df["top20_contains_query"].mean()
        ),
        "top20_center_le40_rate": float(
            query_df["top20_has_center_le40"].mean()
        ),
        "top20_center_le80_rate": float(
            query_df["top20_has_center_le80"].mean()
        ),
    }

    for col in (
        "first_containing_rank",
        "first_center_le40_rank",
        "first_center_le80_rank",
        "top1_center_error_m",
        "top20_center_error_min_m",
        "top20_pairwise_center_median_m",
        "top20_centroid_rms_radius_m",
        "top20_spatial_nms_count_r51p2m",
        "top20_spatial_nms_count_r102p4m",
    ):
        add_flat_stats(
            summary,
            col,
            query_df[col],
        )

    for k in RECALL_KS:
        for prefix in (
            "containment",
            "center_le40",
            "center_le80",
        ):
            col = f"{prefix}_recall_at_{k}"
            summary[col] = float(query_df[col].mean())
            summary[f"{prefix}_hits_at_{k}"] = int(
                query_df[col].sum()
            )

    return candidate_df, query_df, summary


def build_paired_comparison(
    query_eval: pd.DataFrame,
    control_variant: str = "512_s256",
) -> pd.DataFrame:
    metrics = [
        "first_containing_rank",
        "first_center_le40_rank",
        "first_center_le80_rank",
        "top1_center_error_m",
        "top20_center_error_min_m",
        "top20_pairwise_center_median_m",
        "top20_spatial_nms_count_r51p2m",
        "top20_spatial_nms_count_r102p4m",
    ]

    control = query_eval[
        query_eval["variant"] == control_variant
    ][["query_id"] + metrics].copy()

    control = control.rename(
        columns={
            metric: f"control__{metric}"
            for metric in metrics
        }
    )

    rows = []

    for variant, sub in query_eval.groupby(
        "variant",
        sort=False,
    ):
        merged = sub.merge(
            control,
            on="query_id",
            how="left",
            validate="one_to_one",
        )

        for metric in metrics:
            merged[
                f"delta_vs_control__{metric}"
            ] = (
                pd.to_numeric(merged[metric], errors="coerce")
                - pd.to_numeric(
                    merged[f"control__{metric}"],
                    errors="coerce",
                )
            )

        rows.append(merged)

    return pd.concat(rows, ignore_index=True)


def build_named_diagnostics(
    query_eval: pd.DataFrame,
    candidates: pd.DataFrame,
    query_ids: tuple[str, ...] = ("57", "228"),
) -> dict:
    out: dict[str, dict] = {}

    for qid in query_ids:
        variants: dict[str, dict] = {}

        for variant in VARIANT_ORDER:
            match = query_eval[
                (query_eval["variant"] == variant)
                & (query_eval["query_id"] == qid)
            ]

            if match.empty:
                variants[variant] = {"present": False}
                continue

            row = match.iloc[0]

            top = candidates[
                (candidates["variant"] == variant)
                & (candidates["query_id"] == qid)
                & (candidates["rank"] <= PRIMARY_TOP_K)
            ]

            def maybe_rank(column: str) -> int | None:
                value = row[column]
                return None if pd.isna(value) else int(value)

            variants[variant] = {
                "present": True,
                "top1_tile_id": str(row["top1_tile_id"]),
                "top1_center_error_m": float(
                    row["top1_center_error_m"]
                ),
                "top1_contains_query": bool(
                    row["top1_contains_query"]
                ),
                "first_containing_rank": maybe_rank(
                    "first_containing_rank"
                ),
                "first_center_le40_rank": maybe_rank(
                    "first_center_le40_rank"
                ),
                "first_center_le80_rank": maybe_rank(
                    "first_center_le80_rank"
                ),
                "top20_spatial_nms_count_r51p2m": int(
                    row["top20_spatial_nms_count_r51p2m"]
                ),
                "top20_spatial_nms_count_r102p4m": int(
                    row["top20_spatial_nms_count_r102p4m"]
                ),
                "top5": top.head(5)[
                    [
                        "rank",
                        "tile_id",
                        "score",
                        "contains_query_eval_only",
                        "center_error_m_eval_only",
                    ]
                ].to_dict(orient="records"),
            }

        out[f"q{qid}"] = variants

    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "R1.3 fixed-stride footprint comparison. All four descriptor "
            "rankings are computed before any coordinate/containment data are "
            "loaded for evaluation."
        )
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=DEFAULT_PROTOCOL,
    )
    parser.add_argument(
        "--r1-2-report",
        type=Path,
        default=DEFAULT_R12_REPORT,
    )
    parser.add_argument(
        "--query-manifest",
        type=Path,
        default=None,
        help=(
            "Optional explicit canonical query manifest. Otherwise use the "
            "source_csv stored in the canonical query-cache metadata."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    args = parser.parse_args()

    protocol_path = resolve(args.protocol)
    r12_report_path = resolve(args.r1_2_report)
    output_root = resolve(args.output_root)

    if not protocol_path.exists():
        raise FileNotFoundError(protocol_path)
    if not r12_report_path.exists():
        raise FileNotFoundError(r12_report_path)

    cfg = yaml.safe_load(protocol_path.read_text())
    r12 = json.loads(r12_report_path.read_text())

    if (
        r12.get("status")
        != "PASS_R1_MAP_DESCRIPTOR_CACHES_BUILT_AND_VALIDATED"
    ):
        raise RuntimeError(
            "R1.2 has not passed; R1.3 is blocked."
        )

    backend = DinoV2CachedRetrievalBackend()

    query_cache_path = resolve(
        cfg["control"]["query_cache"]
    )
    query_rep = backend.load_representation(
        query_cache_path
    )

    expected_query_count = int(
        cfg["control"]["query_count_expected"]
    )
    if len(query_rep.ids) != expected_query_count:
        raise RuntimeError(
            f"Canonical query count mismatch: "
            f"{len(query_rep.ids)} != {expected_query_count}"
        )

    variant_specs = build_variant_specs(cfg)

    map_reps = {}
    for variant in VARIANT_ORDER:
        path = resolve(
            variant_specs[variant]["map_cache"]
        )
        if not path.exists():
            raise FileNotFoundError(path)

        rep = backend.load_representation(path)
        backend.validate_pair(query_rep, rep)
        map_reps[variant] = rep

    max_rank = min(
        max(RECALL_KS),
        min(len(rep.ids) for rep in map_reps.values()),
    )

    print("=" * 88)
    print("R1.3 PHASE 1 — BLIND INDEPENDENT RETRIEVAL")
    print("=" * 88)

    rankings = {}
    cache_records = {}

    for variant in VARIANT_ORDER:
        rep = map_reps[variant]
        ranking = backend.rank_batch(
            query_rep,
            rep,
            max_rank,
        )
        rankings[variant] = ranking

        cache_records[variant] = {
            "path": str(rep.source_path),
            "sha256": sha256_file(rep.source_path),
            "tile_count": int(len(rep.ids)),
            "descriptor_shape": list(rep.descriptors.shape),
            "retrieval_runtime_s": float(
                ranking.retrieval_runtime_s
            ),
        }

        print(
            f"{variant:10s} "
            f"tiles={len(rep.ids):3d} "
            f"Top-{max_rank} "
            f"runtime_ms/query="
            f"{1000.0 * ranking.retrieval_runtime_s / len(query_rep.ids):.4f}"
        )

    print()
    print("=" * 88)
    print("R1.3 PHASE 2 — POST-RANKING EVALUATION ATTACHMENT")
    print("=" * 88)

    query_manifest_path = resolve_query_manifest_path(
        query_rep.metadata,
        args.query_manifest,
    )
    query_manifest = pd.read_csv(query_manifest_path)
    query_eval_manifest = validate_query_manifest(
        query_manifest,
        query_rep.ids,
    )

    output_root.mkdir(parents=True, exist_ok=True)

    all_candidates = []
    all_query_eval = []
    summaries = []

    for variant in VARIANT_ORDER:
        spec = variant_specs[variant]
        tile_index_path = resolve(spec["index_csv"])

        if not tile_index_path.exists():
            raise FileNotFoundError(tile_index_path)

        tile_index = pd.read_csv(tile_index_path)

        candidates_df, query_df, summary = evaluate_variant(
            variant=variant,
            ranking=rankings[variant],
            map_representation=map_reps[variant],
            query_eval_manifest=query_eval_manifest,
            tile_index=tile_index,
        )

        summary["ground_footprint_m_expected"] = float(
            spec["expected_ground_footprint_m"]
        )
        summary["stride_px"] = int(spec["stride_px"])
        summary["index_csv"] = str(tile_index_path)
        summary["index_sha256"] = sha256_file(tile_index_path)

        all_candidates.append(candidates_df)
        all_query_eval.append(query_df)
        summaries.append(summary)

    candidates = pd.concat(all_candidates, ignore_index=True)
    query_eval = pd.concat(all_query_eval, ignore_index=True)
    summary_df = pd.DataFrame(summaries)

    paired = build_paired_comparison(query_eval)
    diagnostics = build_named_diagnostics(
        query_eval,
        candidates,
    )

    candidate_path = output_root / "r1_3_ranked_candidates_top100.csv"
    query_eval_path = output_root / "r1_3_query_level_evaluation.csv"
    summary_path = output_root / "r1_3_variant_summary.csv"
    paired_path = output_root / "r1_3_paired_vs_512_s256.csv"
    diagnostics_path = output_root / "r1_3_q57_q228_diagnostics.json"
    report_path = (
        output_root
        / "r1_3_independent_footprint_retrieval_report.json"
    )

    candidates.to_csv(candidate_path, index=False)
    query_eval.to_csv(query_eval_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    paired.to_csv(paired_path, index=False)
    diagnostics_path.write_text(
        json.dumps(diagnostics, indent=2),
        encoding="utf-8",
    )

    query_runtime = query_rep.metadata.get("runtime_s")

    report = {
        "stage": "R1.3",
        "status": "PASS_R1_INDEPENDENT_FOOTPRINT_RETRIEVAL",
        "created_at_utc": now_utc(),
        "research_question": (
            "With map GSD, query representation, DINO protocol, retrieval "
            "backend and approximately 51.2 m map-center spacing fixed, how "
            "does represented map ground footprint affect candidate quality?"
        ),
        "protocol": str(protocol_path),
        "protocol_sha256": sha256_file(protocol_path),
        "r1_2_gate": {
            "report": str(r12_report_path),
            "report_sha256": sha256_file(r12_report_path),
            "status": r12["status"],
        },
        "blind_phase": {
            "query_cache": str(query_cache_path),
            "query_cache_sha256": sha256_file(query_cache_path),
            "query_count": int(len(query_rep.ids)),
            "query_descriptor_shape": list(
                query_rep.descriptors.shape
            ),
            "query_encoding_not_rerun": True,
            "query_cache_build_runtime_s_from_metadata": query_runtime,
            "backend": backend.backend_metadata(),
            "max_rank": int(max_rank),
            "map_caches": cache_records,
            "coordinates_loaded_before_ranking": False,
            "oracle_loaded_before_ranking": False,
        },
        "evaluation_phase": {
            "query_manifest": str(query_manifest_path),
            "query_manifest_sha256": sha256_file(query_manifest_path),
            "containment_rule": (
                "query EPSG:3346 point lies inside tile metric bounds"
            ),
            "center_thresholds_m": [40.0, 80.0],
            "spatial_nms_radii_m": list(SPATIAL_NMS_RADII_M),
            "spatial_nms_definition": (
                "rank-ordered greedy center suppression; evaluation only"
            ),
        },
        "variants": {
            item["variant"]: item
            for item in summaries
        },
        "named_diagnostics": diagnostics,
        "outputs": {
            "ranked_candidates_csv": str(candidate_path),
            "query_level_evaluation_csv": str(query_eval_path),
            "variant_summary_csv": str(summary_path),
            "paired_vs_control_csv": str(paired_path),
            "q57_q228_json": str(diagnostics_path),
        },
        "scope_guarantees": {
            "map_footprint_is_primary_changed_variable": True,
            "fixed_stride_px": int(
                cfg["experiment"]["fixed_stride_px"]
            ),
            "query_cache_rebuilt": False,
            "query_preprocessing_changed": False,
            "retrieval_backend_changed": False,
            "coordinates_used_for_ranking": False,
            "oracle_used_for_ranking": False,
            "orb_run": False,
            "bootstrap_run": False,
            "state_or_temporal_authority_run": False,
        },
        "decision_policy": (
            "Do not auto-select a winner. Interpret containment, <=40/80 m "
            "candidate availability, first useful rank, spatial diversity, "
            "q57/q228 behavior, cache size and retrieval cost together."
        ),
        "next_stage": {
            "stage": "R1.4",
            "action": (
                "Interpret and freeze the footprint ablation before deciding "
                "whether R2 cross-scale fusion is justified."
            ),
        },
    }

    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print()
    print("=" * 88)
    print("R1.3 — INDEPENDENT FOOTPRINT RETRIEVAL SUMMARY")
    print("=" * 88)

    display_cols = [
        "variant",
        "ground_footprint_m_expected",
        "map_tile_count",
        "containment_recall_at_1",
        "containment_recall_at_5",
        "containment_recall_at_20",
        "center_le40_recall_at_20",
        "center_le80_recall_at_20",
        "top20_spatial_nms_count_r51p2m_median",
        "top20_spatial_nms_count_r102p4m_median",
        "retrieval_runtime_ms_per_query",
    ]

    print(
        summary_df[display_cols].to_string(index=False)
    )

    print()
    print("q57:")
    print(json.dumps(diagnostics["q57"], indent=2))

    print()
    print("q228:")
    print(json.dumps(diagnostics["q228"], indent=2))

    print()
    print("STATUS: PASS_R1_INDEPENDENT_FOOTPRINT_RETRIEVAL")
    print("report:", report_path)


if __name__ == "__main__":
    main()
