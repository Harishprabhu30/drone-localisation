#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from pyproj import Transformer

from uavloc.retrieval import (
    DinoV2CachedRetrievalBackend,
    fuse_physical_regions,
    reciprocal_rank,
)


ROOT = Path.cwd().resolve()

DEFAULT_CONFIG = Path(
    "configs/research/retrieval_candidate_pool_v2_r2.yaml"
)
DEFAULT_SOURCE_RUN = Path(
    "outputs/demo_runs/traj01_blind_regression_001"
)
DEFAULT_REFERENCE_ATTACHMENT = Path(
    "outputs/research_runs/retrieval_candidate_pool_v2/"
    "r1_4_downstream_replay/runs/768_s256/"
    "evaluation/reference_attachment.csv"
)
DEFAULT_R14_REPORT = Path(
    "outputs/research_runs/retrieval_candidate_pool_v2/"
    "r1_4_downstream_replay/r1_4_downstream_replay_report.json"
)
DEFAULT_OUTPUT_ROOT = Path(
    "outputs/research_runs/retrieval_candidate_pool_v2/"
    "r2_cross_scale_fusion/r2_2_blind_orb"
)

QUERY_CACHE_NAME = (
    "s8_11c_dinov2_queries_v_1fps_"
    "dinov2_vits14_img518_center_square_avgpatch_cpu.npz"
)
BLIND_MANIFEST_REL = Path("metadata/blind_query_manifest.csv")


def resolve(path_like: str | Path) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else ROOT / path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def composite_tile_id(variant: str, tile_id: str) -> str:
    return f"{variant}::{tile_id}"


def split_composite_tile_id(value: str) -> tuple[str, str]:
    parts = str(value).split("::", 1)
    if len(parts) != 2 or not all(parts):
        raise ValueError(f"Invalid composite tile ID: {value}")
    return parts[0], parts[1]


def load_tile_index(path: Path, variant: str, map_ids: np.ndarray) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {
        "tile_id",
        "tile_path",
        "left_easting",
        "bottom_northing",
        "right_easting",
        "top_northing",
        "center_easting",
        "center_northing",
        "grid_row",
        "grid_col",
        "pixel_col_off",
        "pixel_row_off",
        "tile_width_px",
        "tile_height_px",
        "ground_width_m",
        "ground_height_m",
    }
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(
            f"{variant}: tile index missing columns {sorted(missing)}"
        )

    frame = frame.copy()
    frame["tile_id"] = frame["tile_id"].astype(str)
    expected = map_ids.astype(str)

    if len(frame) != len(expected):
        raise RuntimeError(f"{variant}: tile-index/cache row mismatch.")
    if not np.array_equal(frame["tile_id"].to_numpy(), expected):
        raise RuntimeError(f"{variant}: tile-index/cache ID order mismatch.")

    frame["source_variant"] = variant
    frame["source_tile_id"] = frame["tile_id"]
    frame["tile_id"] = [
        composite_tile_id(variant, tile_id)
        for tile_id in frame["source_tile_id"].astype(str)
    ]

    return frame.reset_index(drop=True)


def make_candidate(
    *,
    variant: str,
    rank: int,
    score: float,
    row: pd.Series,
    rrf_k: float,
) -> dict:
    return {
        "variant": variant,
        "tile_id": str(row["source_tile_id"]),
        "composite_tile_id": str(row["tile_id"]),
        "tile_path": str(row["tile_path"]),
        "per_scale_rank": int(rank),
        "dino_score": float(score),
        "rrf_contribution": float(
            reciprocal_rank(int(rank), float(rrf_k))
        ),
        "grid_row": int(row["grid_row"]),
        "grid_col": int(row["grid_col"]),
        "pixel_col_off": int(row["pixel_col_off"]),
        "pixel_row_off": int(row["pixel_row_off"]),
        "tile_width_px": int(row["tile_width_px"]),
        "tile_height_px": int(row["tile_height_px"]),
        "ground_width_m": float(row["ground_width_m"]),
        "ground_height_m": float(row["ground_height_m"]),
        "left_easting": float(row["left_easting"]),
        "bottom_northing": float(row["bottom_northing"]),
        "right_easting": float(row["right_easting"]),
        "top_northing": float(row["top_northing"]),
        "center_easting": float(row["center_easting"]),
        "center_northing": float(row["center_northing"]),
    }


def run(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print()
    print("$", " ".join(command))

    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log.write(line)
        code = process.wait()

    if code != 0:
        raise RuntimeError(
            f"Command failed with exit code {code}: "
            + " ".join(command)
        )


def load_reference_attachment(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"query_id", "eval_ref_lon", "eval_ref_lat"}
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(
            f"Reference attachment missing {sorted(missing)}"
        )

    frame = frame.copy()
    frame["query_id"] = pd.to_numeric(
        frame["query_id"],
        errors="raise",
    ).astype(int)

    transformer = Transformer.from_crs(
        "EPSG:4326",
        "EPSG:3346",
        always_xy=True,
    )
    x, y = transformer.transform(
        pd.to_numeric(
            frame["eval_ref_lon"],
            errors="raise",
        ).to_numpy(float),
        pd.to_numeric(
            frame["eval_ref_lat"],
            errors="raise",
        ).to_numpy(float),
    )
    frame["gt_x"] = x
    frame["gt_y"] = y
    return frame


def contains(row: pd.Series, x: float, y: float) -> bool:
    return bool(
        float(row["left_easting"]) <= x
        <= float(row["right_easting"])
        and float(row["bottom_northing"]) <= y
        <= float(row["top_northing"])
    )


def center_error(row: pd.Series, x: float, y: float) -> float:
    return float(
        math.hypot(
            float(row["center_easting"]) - x,
            float(row["center_northing"]) - y,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "R2.2 recompute promoted triple fusion on the strict blind query "
            "cache, adapt mixed-scale representatives into the unchanged ORB "
            "Top-20 verifier, then attach reference only for ORB evaluation."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--source-run-root",
        type=Path,
        default=DEFAULT_SOURCE_RUN,
    )
    parser.add_argument(
        "--reference-attachment",
        type=Path,
        default=DEFAULT_REFERENCE_ATTACHMENT,
    )
    parser.add_argument(
        "--r1-4-report",
        type=Path,
        default=DEFAULT_R14_REPORT,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    args = parser.parse_args()

    config_path = resolve(args.config)
    source_root = resolve(args.source_run_root)
    reference_path = resolve(args.reference_attachment)
    r14_report_path = resolve(args.r1_4_report)
    output_root = resolve(args.output_root)

    for path in (
        config_path,
        reference_path,
        r14_report_path,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    cfg = yaml.safe_load(config_path.read_text())
    fusion_cfg = cfg["fusion"]
    region_cfg = fusion_cfg["physical_region"]

    experiment_name = "triple_512_768_1024"
    experts = list(
        cfg["experiments"][experiment_name]["experts"]
    )

    if experts != ["512_s256", "768_s256", "1024_s256"]:
        raise RuntimeError(
            "R2.2 expects promoted triple expert order 512/768/1024."
        )

    blind_manifest = source_root / BLIND_MANIFEST_REL
    blind_query_cache = (
        source_root
        / "descriptors"
        / QUERY_CACHE_NAME
    )

    for path in (blind_manifest, blind_query_cache):
        if not path.exists():
            raise FileNotFoundError(path)

    manifest = pd.read_csv(blind_manifest)
    if len(manifest) != 403:
        raise RuntimeError(
            f"Expected 403 blind queries, got {len(manifest)}."
        )

    backend = DinoV2CachedRetrievalBackend()
    query_rep = backend.load_representation(blind_query_cache)

    if len(query_rep.ids) != 403:
        raise RuntimeError("Blind query cache must contain 403 queries.")

    per_scale_depth = int(fusion_cfg["per_scale_depth"])
    final_top_k = int(fusion_cfg["final_top_k"])
    rrf_k = float(fusion_cfg["rrf_k"])

    rankings = {}
    tile_indices = {}

    print("=" * 96)
    print("R2.2 PHASE 1 — STRICT-BLIND PER-SCALE RANKING")
    print("=" * 96)

    for variant in experts:
        spec = cfg["experts"][variant]
        map_rep = backend.load_representation(
            resolve(spec["map_cache"])
        )
        backend.validate_pair(query_rep, map_rep)

        index = load_tile_index(
            resolve(spec["index_csv"]),
            variant,
            map_rep.ids,
        )

        ranking = backend.rank_batch(
            query_rep,
            map_rep,
            per_scale_depth,
        )

        rankings[variant] = ranking
        tile_indices[variant] = index

        print(
            f"{variant:10s} "
            f"depth={per_scale_depth} "
            f"runtime_ms/query="
            f"{1000.0 * ranking.retrieval_runtime_s / len(query_rep.ids):.4f}"
        )

    print()
    print("=" * 96)
    print("R2.2 PHASE 2 — STRICT-BLIND TRIPLE REGION FUSION")
    print("=" * 96)

    topk_rows = []
    representative_keys = set()
    provenance_rows = []

    for query_index, query_id in enumerate(
        query_rep.ids.astype(str)
    ):
        pool = []

        for variant in experts:
            ranking = rankings[variant]
            index = tile_indices[variant]

            for local_rank, (map_index, score) in enumerate(
                zip(
                    ranking.indices[query_index],
                    ranking.scores[query_index],
                ),
                start=1,
            ):
                pool.append(
                    make_candidate(
                        variant=variant,
                        rank=local_rank,
                        score=float(score),
                        row=index.iloc[int(map_index)],
                        rrf_k=rrf_k,
                    )
                )

        regions = fuse_physical_regions(
            pool,
            scale_order=experts,
            max_center_distance_m=float(
                region_cfg["max_center_distance_m"]
            ),
            min_iou=float(
                region_cfg["minimum_bbox_iou"]
            ),
            final_top_k=final_top_k,
        )

        if len(regions) != final_top_k:
            raise RuntimeError(
                f"q{query_id}: fusion produced {len(regions)} regions, "
                f"expected {final_top_k}."
            )

        for region in regions:
            rep = region["representative"]
            composite = composite_tile_id(
                str(rep["variant"]),
                str(rep["tile_id"]),
            )
            representative_keys.add(composite)

            topk_rows.append(
                {
                    "variant": "r2_triple_512_768_1024",
                    "query_id": str(query_id),
                    "rank": int(region["fused_rank"]),
                    "tile_id": composite,
                    "score": float(region["fused_rrf_score"]),
                    "fused_rrf_score": float(
                        region["fused_rrf_score"]
                    ),
                    "support_scale_count": int(
                        region["support_scale_count"]
                    ),
                    "support_scales": "|".join(
                        region["support_scales"]
                    ),
                    "representative_variant": str(
                        rep["variant"]
                    ),
                    "representative_original_tile_id": str(
                        rep["tile_id"]
                    ),
                    "representative_per_scale_rank": int(
                        rep["per_scale_rank"]
                    ),
                    "representative_dino_score": float(
                        rep["dino_score"]
                    ),
                }
            )

            provenance_rows.append(
                {
                    "query_id": str(query_id),
                    "fused_rank": int(region["fused_rank"]),
                    "composite_tile_id": composite,
                    "members_json": json.dumps(
                        [
                            {
                                "variant": member["variant"],
                                "tile_id": member["tile_id"],
                                "per_scale_rank": member[
                                    "per_scale_rank"
                                ],
                                "dino_score": member["dino_score"],
                                "rrf_contribution": member[
                                    "rrf_contribution"
                                ],
                                "tile_path": member["tile_path"],
                            }
                            for member in region["members"]
                        ],
                        sort_keys=True,
                    ),
                }
            )

    topk = pd.DataFrame(topk_rows)
    provenance = pd.DataFrame(provenance_rows)

    mixed_parts = []
    for variant in experts:
        mixed_parts.append(tile_indices[variant])
    mixed = pd.concat(mixed_parts, ignore_index=True)

    if mixed["tile_id"].duplicated().any():
        raise RuntimeError(
            "Composite mixed-scale tile IDs are not unique."
        )

    mixed_representatives = mixed[
        mixed["tile_id"].isin(representative_keys)
    ].copy()

    if set(mixed_representatives["tile_id"]) != representative_keys:
        missing = sorted(
            representative_keys
            - set(mixed_representatives["tile_id"])
        )
        raise RuntimeError(
            "Mixed tile index missing fused representatives: "
            + ", ".join(missing[:20])
        )

    retrieval_dir = output_root / "retrieval"
    metadata_dir = output_root / "metadata"
    reports_dir = output_root / "reports"
    logs_dir = output_root / "logs"

    for path in (
        retrieval_dir,
        metadata_dir,
        reports_dir,
        logs_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)

    topk_path = (
        retrieval_dir
        / "r2_2_blind_triple_fused_top20.csv"
    )
    mixed_index_path = (
        metadata_dir
        / "r2_2_mixed_representative_tile_index.csv"
    )
    provenance_path = (
        metadata_dir
        / "r2_2_region_member_provenance.csv"
    )

    topk.to_csv(topk_path, index=False)
    mixed_representatives.to_csv(
        mixed_index_path,
        index=False,
    )
    provenance.to_csv(provenance_path, index=False)

    print("blind fused Top-20:", topk_path)
    print("mixed representative index:", mixed_index_path)

    # --------------------------------------------------------------
    # PHASE 3 — unchanged ORB/hybrid verifier over the fused Top-20.
    # --------------------------------------------------------------
    print()
    print("=" * 96)
    print("R2.2 PHASE 3 — UNCHANGED ORB/HYBRID VERIFICATION")
    print("=" * 96)

    orb_script = (
        ROOT
        / "scripts/villoc/s8_12e1_top20_verifier_reranker.py"
    )
    demo_config = (
        ROOT
        / "configs/demo_villoc_traj01_90deg_stable120m_blind_v2.yaml"
    )
    orb_root = (
        reports_dir
        / "r2_2_triple_orb_hybrid_top20"
    )

    run(
        [
            sys.executable,
            str(orb_script),
            "--config",
            str(demo_config),
            "--repo-root",
            str(ROOT),
            "--variant",
            "r2_triple_512_768_1024",
            "--tag",
            "dinov2_vits14_img518_center_square_avgpatch_cpu",
            "--query-csv",
            str(blind_manifest),
            "--topk-csv",
            str(topk_path),
            "--tile-index-csv",
            str(mixed_index_path),
            "--out-root",
            str(orb_root),
            "--top-n",
            "20",
            "--policy",
            "hybrid",
            "--blind-only",
        ],
        logs_dir / "r2_2_orb_hybrid.log",
    )

    scored_path = (
        orb_root
        / "s8_12e1_all_candidate_verifier_scores.csv"
    )
    summary_path = (
        orb_root
        / "s8_12e1_query_summary.csv"
    )

    for path in (scored_path, summary_path):
        if not path.exists():
            raise RuntimeError(
                f"ORB output missing: {path}"
            )

    # --------------------------------------------------------------
    # PHASE 4 — post-ORB evaluation attachment.
    # --------------------------------------------------------------
    print()
    print("=" * 96)
    print("R2.2 PHASE 4 — POST-ORB EVALUATION")
    print("=" * 96)

    reference = load_reference_attachment(reference_path)
    ref_by_q = reference.set_index("query_id")

    mixed_lookup = (
        mixed_representatives
        .set_index("tile_id")
    )

    scored = pd.read_csv(scored_path)
    scored["query_id"] = pd.to_numeric(
        scored["query_id"],
        errors="raise",
    ).astype(int)
    scored["hybrid_rank"] = pd.to_numeric(
        scored["hybrid_rank"],
        errors="coerce",
    )

    selected = scored[
        scored["hybrid_rank"] == 1
    ].copy()

    if len(selected) != 403:
        raise RuntimeError(
            f"Expected 403 ORB-selected rows, got {len(selected)}."
        )

    selected_eval_rows = []

    for _, row in selected.iterrows():
        qid = int(row["query_id"])
        composite = str(row["tile_id"])

        if composite not in mixed_lookup.index:
            raise RuntimeError(
                f"Selected composite tile missing: {composite}"
            )
        if qid not in ref_by_q.index:
            raise RuntimeError(
                f"Reference missing q{qid}"
            )

        tile = mixed_lookup.loc[composite]
        ref = ref_by_q.loc[qid]

        error = center_error(
            tile,
            float(ref["gt_x"]),
            float(ref["gt_y"]),
        )
        is_containing = contains(
            tile,
            float(ref["gt_x"]),
            float(ref["gt_y"]),
        )

        variant, original_tile_id = split_composite_tile_id(
            composite
        )

        selected_eval_rows.append(
            {
                "query_id": qid,
                "selected_composite_tile_id": composite,
                "selected_variant": variant,
                "selected_original_tile_id": original_tile_id,
                "selected_fused_rank": int(float(row["rank"])),
                "selected_hybrid_rank": int(float(row["hybrid_rank"])),
                "selected_verifier_rank": int(float(row["verifier_rank"])),
                "selected_inliers": int(float(row["inliers"])),
                "selected_homography_ok": bool(row["homography_ok"]),
                "selected_center_error_m": error,
                "selected_contains_query": is_containing,
                "selected_le40": bool(error <= 40.0),
                "selected_le80": bool(error <= 80.0),
            }
        )

    selected_eval = pd.DataFrame(selected_eval_rows)

    selected_eval_path = (
        reports_dir
        / "r2_2_orb_selected_postfreeze_evaluation.csv"
    )
    selected_eval.to_csv(
        selected_eval_path,
        index=False,
    )

    r14 = json.loads(
        r14_report_path.read_text()
    )

    summary = {
        "selected_contains_hits": int(
            selected_eval["selected_contains_query"].sum()
        ),
        "selected_contains_rate": float(
            selected_eval["selected_contains_query"].mean()
        ),
        "selected_le40_hits": int(
            selected_eval["selected_le40"].sum()
        ),
        "selected_le80_hits": int(
            selected_eval["selected_le80"].sum()
        ),
        "selected_center_error_mean_m": float(
            selected_eval["selected_center_error_m"].mean()
        ),
        "selected_center_error_median_m": float(
            selected_eval["selected_center_error_m"].median()
        ),
        "selected_center_error_p95_m": float(
            selected_eval["selected_center_error_m"].quantile(0.95)
        ),
        "selected_by_scale": {
            variant: int(
                (
                    selected_eval["selected_variant"] == variant
                ).sum()
            )
            for variant in experts
        },
    }

    controls = {}
    for variant in ("512_s256", "768_s256", "1024_s256"):
        r1v = r14["variants"][variant]
        controls[variant] = {
            "selected_contains_hits": int(
                r1v["orb_selected"]["selected_top1_contains_hits"]
            ),
            "selected_le40_hits": int(
                r1v["orb_selected"]["selected_top1_le40_hits"]
            ),
            "selected_le80_hits": int(
                r1v["orb_selected"]["selected_top1_le80_hits"]
            ),
        }

    named = {}
    for qid in (57, 228, 390):
        row = selected_eval[
            selected_eval["query_id"] == qid
        ]
        if len(row) != 1:
            named[f"q{qid}"] = {"present": False}
            continue

        item = row.iloc[0].to_dict()

        q_scored = scored[
            scored["query_id"] == qid
        ].sort_values("hybrid_rank")

        named[f"q{qid}"] = {
            "present": True,
            **item,
            "orb_top5": q_scored.head(5)[
                [
                    "tile_id",
                    "rank",
                    "representative_variant",
                    "representative_original_tile_id",
                    "support_scales",
                    "inliers",
                    "inlier_ratio",
                    "homography_ok",
                    "verifier_rank",
                    "hybrid_rank",
                    "verifier_score",
                    "hybrid_score",
                ]
            ].to_dict(orient="records"),
        }

    report = {
        "stage": "R2.2",
        "status": "PASS_R2_TRIPLE_FUSION_ORB_TRANSLATION",
        "created_at_utc": now_utc(),
        "config": str(config_path),
        "common_blind_checkpoint": {
            "source_run_root": str(source_root),
            "manifest": str(blind_manifest),
            "manifest_sha256": sha256_file(blind_manifest),
            "query_cache": str(blind_query_cache),
            "query_cache_sha256": sha256_file(blind_query_cache),
        },
        "fusion": {
            "experiment": experiment_name,
            "per_scale_depth": per_scale_depth,
            "final_top_k": final_top_k,
            "rrf_k": rrf_k,
            "physical_region": region_cfg,
            "composite_tile_ids": True,
        },
        "blind_outputs": {
            "top20": str(topk_path),
            "top20_sha256": sha256_file(topk_path),
            "mixed_tile_index": str(mixed_index_path),
            "provenance": str(provenance_path),
        },
        "orb": {
            "script": str(orb_script),
            "policy": "hybrid",
            "top_n": 20,
            "algorithm_changed": False,
            "scored_candidates": str(scored_path),
            "query_summary": str(summary_path),
        },
        "post_orb_evaluation": {
            "reference_attachment": str(reference_path),
            "reference_attachment_sha256": sha256_file(
                reference_path
            ),
            "summary": summary,
            "single_scale_controls_from_r1_4": controls,
            "selected_evaluation_csv": str(selected_eval_path),
        },
        "named_diagnostics": named,
        "scope_guarantees": {
            "reference_used_before_fusion": False,
            "reference_used_before_orb": False,
            "orb_policy_changed": False,
            "bootstrap_run": False,
            "state_policy_changed": False,
        },
        "next_stage": {
            "stage": "R2.3 decision",
            "action": (
                "Promote to unchanged state replay only if blind fused ORB "
                "selection is materially competitive with the 768 reference "
                "and q57 rescue behavior is understood."
            ),
        },
    }

    report_path = (
        reports_dir
        / "r2_2_triple_fusion_orb_report.json"
    )
    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print()
    print("=" * 96)
    print("R2.2 — TRIPLE FUSION -> ORB SUMMARY")
    print("=" * 96)
    print("selected contains:", summary["selected_contains_hits"], "/ 403")
    print("selected <=40 m:", summary["selected_le40_hits"], "/ 403")
    print("selected <=80 m:", summary["selected_le80_hits"], "/ 403")
    print("selected by scale:", summary["selected_by_scale"])
    print()
    print("R1.4 controls:")
    for variant, item in controls.items():
        print(variant, item)
    for qid in (57, 228, 390):
        print()
        print(f"q{qid}:")
        print(json.dumps(named[f"q{qid}"], indent=2))
    print()
    print("STATUS: PASS_R2_TRIPLE_FUSION_ORB_TRANSLATION")
    print("report:", report_path)


if __name__ == "__main__":
    main()
