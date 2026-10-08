#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from pyproj import Transformer


ROOT = Path.cwd().resolve()

DEFAULT_PROTOCOL = Path(
    "configs/research/retrieval_candidate_pool_v2_r1.yaml"
)
DEFAULT_DEMO_CONFIG = Path(
    "configs/demo_villoc_traj01_90deg_stable120m_blind_v2.yaml"
)
DEFAULT_SOURCE_RUN = Path(
    "outputs/demo_runs/traj01_blind_regression_001"
)
DEFAULT_REFERENCE = Path(
    "outputs/villoc/traj01_90deg_stable120m/trajectories/"
    "s8_3_reference_trajectory_V_1fps.csv"
)
DEFAULT_OUTPUT_ROOT = Path(
    "outputs/research_runs/retrieval_candidate_pool_v2/"
    "r1_4_downstream_replay"
)

VARIANT_ORDER = (
    "512_s256",
    "768_s256",
    "1024_s256",
)

QUERY_CACHE_NAME = (
    "s8_11c_dinov2_queries_v_1fps_"
    "dinov2_vits14_img518_center_square_avgpatch_cpu.npz"
)

RAW_RELATIVE_REL = Path(
    "metadata/s8_xfeat_relative_frontend/"
    "s8r4_xfeat_relative_trajectory_blind_raw.csv"
)
BLIND_MANIFEST_REL = Path(
    "metadata/blind_query_manifest.csv"
)


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


def bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)

    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes", "y", "t"})
    )


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def require_source_checkpoint(
    source_root: Path,
    query_cache: Path | None,
) -> dict[str, Path]:
    manifest = source_root / BLIND_MANIFEST_REL
    relative = source_root / RAW_RELATIVE_REL

    if query_cache is None:
        query_cache = (
            source_root
            / "descriptors"
            / QUERY_CACHE_NAME
        )

    required = {
        "manifest": manifest,
        "relative": relative,
        "query_cache": query_cache,
    }

    missing = [
        f"{name}: {path}"
        for name, path in required.items()
        if not path.exists()
    ]

    if missing:
        raise RuntimeError(
            "R1.4 common blind pre-retrieval checkpoint is incomplete.\n"
            + "\n".join(missing)
            + "\nPass --source-run-root / --query-cache explicitly if the "
              "checkpoint lives elsewhere."
        )

    manifest_df = pd.read_csv(manifest)
    relative_df = pd.read_csv(relative)

    if len(manifest_df) != 403:
        raise RuntimeError(
            f"Expected 403 blind manifest rows, got {len(manifest_df)}."
        )

    if len(relative_df) != 403:
        raise RuntimeError(
            f"Expected 403 relative rows, got {len(relative_df)}."
        )

    if "query_id" not in manifest_df.columns:
        raise RuntimeError("Blind manifest has no query_id.")

    return required


def variant_specs(protocol: dict) -> dict[str, dict]:
    by_name = {
        str(level["name"]): level
        for level in protocol["experiment"]["levels"]
    }

    missing = [
        name
        for name in VARIANT_ORDER
        if name not in by_name
    ]

    if missing:
        raise RuntimeError(
            f"R1.4 protocol missing variants: {missing}"
        )

    return {
        name: by_name[name]
        for name in VARIANT_ORDER
    }


def safe_remove_research_run(path: Path, output_root: Path) -> None:
    resolved = path.resolve()
    root = output_root.resolve()

    if root not in resolved.parents:
        raise RuntimeError(
            f"Refusing to remove path outside R1.4 output root: {resolved}"
        )

    if path.exists():
        shutil.rmtree(path)


def metrics(errors: np.ndarray) -> dict:
    values = np.asarray(errors, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return {
            "count": 0,
            "rmse_m": None,
            "mean_m": None,
            "median_m": None,
            "p95_m": None,
            "max_m": None,
            "final_m": None,
        }

    return {
        "count": int(len(values)),
        "rmse_m": float(math.sqrt(np.mean(values * values))),
        "mean_m": float(np.mean(values)),
        "median_m": float(np.median(values)),
        "p95_m": float(np.percentile(values, 95)),
        "max_m": float(np.max(values)),
        "final_m": float(values[-1]),
    }


def postfreeze_variant_diagnostics(
    *,
    variant: str,
    run_root: Path,
    tile_index_path: Path,
) -> dict:
    freeze_path = (
        run_root
        / "evaluation/blind_submission_freeze.json"
    )
    submission_path = (
        run_root
        / "trajectories/submission_estimated_trajectory.csv"
    )
    attachment_path = (
        run_root
        / "evaluation/reference_attachment.csv"
    )
    orb_root = (
        run_root
        / "reports/s8_12e1_top20_verifier_reranker"
        / f"{variant}_orb_hybrid_top20_img518"
    )
    orb_scores_path = (
        orb_root
        / "s8_12e1_all_candidate_verifier_scores.csv"
    )
    orb_summary_path = (
        orb_root
        / "s8_12e1_query_summary.csv"
    )
    bootstrap_path = (
        run_root
        / "reports/blind_map_bootstrap/"
          "blind_map_bootstrap_report.json"
    )
    timeline_path = (
        run_root
        / "reports/blind_map_bootstrap/"
          "canonical_map_state_timeline.json"
    )

    for path in (
        freeze_path,
        submission_path,
        attachment_path,
        orb_scores_path,
        orb_summary_path,
        bootstrap_path,
        timeline_path,
        tile_index_path,
    ):
        if not path.exists():
            raise RuntimeError(
                f"{variant}: missing post-freeze diagnostic input {path}"
            )

    freeze = load_json(freeze_path)
    expected_sha = freeze["submission"]["sha256"]
    actual_sha = sha256_file(submission_path)

    if actual_sha != expected_sha:
        raise RuntimeError(
            f"{variant}: frozen submission SHA mismatch."
        )

    attachment = pd.read_csv(attachment_path)
    submission = pd.read_csv(submission_path)

    if len(attachment) != 403 or len(submission) != 403:
        raise RuntimeError(
            f"{variant}: expected 403-row attachment/submission."
        )

    error_col = "eval_position_error_m"
    if error_col not in attachment.columns:
        raise RuntimeError(
            f"{variant}: optional evaluation did not produce {error_col}."
        )

    trajectory_metrics = metrics(
        pd.to_numeric(
            attachment[error_col],
            errors="coerce",
        ).to_numpy(dtype=float)
    )

    available = bool_series(
        submission["map_aligned_available"]
    )
    accepted = bool_series(
        submission["accepted_correction"]
    )

    timeline = load_json(timeline_path)
    events = timeline.get("events", [])
    source_update_queries = [
        int(event["source_update_query_id"])
        for event in events
    ]

    orb_scores = pd.read_csv(orb_scores_path)
    orb_summary = pd.read_csv(orb_summary_path)
    tiles = pd.read_csv(tile_index_path)

    for df, name in (
        (orb_scores, "ORB scores"),
        (orb_summary, "ORB summary"),
        (tiles, "tile index"),
    ):
        if len(df) == 0:
            raise RuntimeError(f"{variant}: empty {name}.")

    tile_lookup = (
        tiles.assign(
            tile_id=tiles["tile_id"].astype(str)
        )
        .set_index("tile_id")
    )

    reference = attachment[
        [
            "query_id",
            "eval_ref_lon",
            "eval_ref_lat",
        ]
    ].copy()
    reference["query_id"] = (
        pd.to_numeric(
            reference["query_id"],
            errors="raise",
        ).astype(int)
    )

    transformer = Transformer.from_crs(
        "EPSG:4326",
        "EPSG:3346",
        always_xy=True,
    )
    ref_x, ref_y = transformer.transform(
        pd.to_numeric(
            reference["eval_ref_lon"],
            errors="raise",
        ).to_numpy(float),
        pd.to_numeric(
            reference["eval_ref_lat"],
            errors="raise",
        ).to_numpy(float),
    )
    reference["gt_x"] = ref_x
    reference["gt_y"] = ref_y
    ref_by_q = reference.set_index("query_id")

    scored = orb_scores.copy()
    scored["query_id"] = (
        pd.to_numeric(
            scored["query_id"],
            errors="raise",
        ).astype(int)
    )
    scored["tile_id"] = scored["tile_id"].astype(str)
    scored["hybrid_rank"] = pd.to_numeric(
        scored["hybrid_rank"],
        errors="coerce",
    )

    selected = scored[
        scored["hybrid_rank"] == 1
    ].copy()

    if len(selected) != 403:
        raise RuntimeError(
            f"{variant}: expected 403 ORB hybrid Top-1 rows, got {len(selected)}."
        )

    selected_errors = []
    selected_contains = []

    for row in selected.itertuples(index=False):
        qid = int(row.query_id)
        tid = str(row.tile_id)

        if tid not in tile_lookup.index or qid not in ref_by_q.index:
            selected_errors.append(np.nan)
            selected_contains.append(False)
            continue

        tile = tile_lookup.loc[tid]
        ref = ref_by_q.loc[qid]

        error = math.hypot(
            float(tile["center_easting"]) - float(ref["gt_x"]),
            float(tile["center_northing"]) - float(ref["gt_y"]),
        )
        contains = (
            float(tile["left_easting"]) <= float(ref["gt_x"])
            <= float(tile["right_easting"])
            and float(tile["bottom_northing"]) <= float(ref["gt_y"])
            <= float(tile["top_northing"])
        )

        selected_errors.append(error)
        selected_contains.append(contains)

    selected["postfreeze_center_error_m"] = selected_errors
    selected["postfreeze_contains_query"] = selected_contains

    orb_metrics = {
        "selected_top1_contains_hits": int(
            selected["postfreeze_contains_query"].sum()
        ),
        "selected_top1_contains_rate": float(
            selected["postfreeze_contains_query"].mean()
        ),
        "selected_top1_le40_hits": int(
            (
                selected["postfreeze_center_error_m"]
                <= 40.0
            ).sum()
        ),
        "selected_top1_le80_hits": int(
            (
                selected["postfreeze_center_error_m"]
                <= 80.0
            ).sum()
        ),
        "selected_center_error": metrics(
            selected["postfreeze_center_error_m"].to_numpy(float)
        ),
    }

    qdiag = {}
    attachment_q = attachment.copy()
    attachment_q["query_id"] = pd.to_numeric(
        attachment_q["query_id"],
        errors="raise",
    ).astype(int)

    for qid in (57, 228):
        orb_row = selected[selected["query_id"] == qid]
        traj_row = attachment_q[
            attachment_q["query_id"] == qid
        ]

        qdiag[f"q{qid}"] = {
            "orb_present": bool(len(orb_row) == 1),
            "orb_tile_id": (
                str(orb_row.iloc[0]["tile_id"])
                if len(orb_row) == 1
                else None
            ),
            "orb_original_dino_rank": (
                int(float(orb_row.iloc[0]["rank"]))
                if len(orb_row) == 1
                else None
            ),
            "orb_inliers": (
                int(float(orb_row.iloc[0]["inliers"]))
                if len(orb_row) == 1
                and pd.notna(orb_row.iloc[0]["inliers"])
                else None
            ),
            "orb_center_error_m_postfreeze": (
                float(
                    orb_row.iloc[0][
                        "postfreeze_center_error_m"
                    ]
                )
                if len(orb_row) == 1
                else None
            ),
            "orb_contains_query_postfreeze": (
                bool(
                    orb_row.iloc[0][
                        "postfreeze_contains_query"
                    ]
                )
                if len(orb_row) == 1
                else None
            ),
            "is_map_state_source_update": (
                qid in source_update_queries
            ),
            "trajectory_error_m": (
                float(
                    traj_row.iloc[0][
                        "eval_position_error_m"
                    ]
                )
                if len(traj_row) == 1
                and pd.notna(
                    traj_row.iloc[0][
                        "eval_position_error_m"
                    ]
                )
                else None
            ),
        }

    bootstrap = load_json(bootstrap_path)

    return {
        "variant": variant,
        "freeze_submission_sha256": actual_sha,
        "localization_state": bootstrap.get(
            "localization_state"
        ),
        "map_state_available": bootstrap.get(
            "map_state_available"
        ),
        "map_state_trust": bootstrap.get(
            "map_state_trust"
        ),
        "matured_at_query_id": timeline.get(
            "matured_at_query_id"
        ),
        "map_state_event_count": int(
            timeline.get("event_count", 0)
        ),
        "map_state_source_update_queries": source_update_queries,
        "map_aligned_rows": int(available.sum()),
        "accepted_correction_rows": int(accepted.sum()),
        "trajectory_error": trajectory_metrics,
        "orb_selected": orb_metrics,
        "named_diagnostics": qdiag,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "R1.4 replay the unchanged blind absolute-localization/state chain "
            "from one common pre-retrieval checkpoint for 512_s256, "
            "768_s256 and 1024_s256 independently."
        )
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=DEFAULT_PROTOCOL,
    )
    parser.add_argument(
        "--demo-config",
        type=Path,
        default=DEFAULT_DEMO_CONFIG,
    )
    parser.add_argument(
        "--source-run-root",
        type=Path,
        default=DEFAULT_SOURCE_RUN,
    )
    parser.add_argument(
        "--query-cache",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--reference-csv",
        type=Path,
        default=DEFAULT_REFERENCE,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    parser.add_argument(
        "--variants",
        default=",".join(VARIANT_ORDER),
        help="Comma-separated subset of 512_s256,768_s256,1024_s256.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete existing selected R1.4 research run roots before replay.",
    )
    args = parser.parse_args()

    protocol_path = resolve(args.protocol)
    demo_config_path = resolve(args.demo_config)
    source_root = resolve(args.source_run_root)
    reference_path = resolve(args.reference_csv)
    output_root = resolve(args.output_root)
    explicit_query_cache = (
        resolve(args.query_cache)
        if args.query_cache is not None
        else None
    )

    for path in (
        protocol_path,
        demo_config_path,
        reference_path,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    protocol = yaml.safe_load(protocol_path.read_text())
    demo_cfg = yaml.safe_load(demo_config_path.read_text())

    if (
        demo_cfg.get("bootstrap", {}).get("backend")
        != "minimum_confident_v2"
    ):
        raise RuntimeError(
            "R1.4 requires the frozen minimum_confident_v2 backend."
        )

    source = require_source_checkpoint(
        source_root,
        explicit_query_cache,
    )

    variants_requested = tuple(
        value.strip()
        for value in str(args.variants).split(",")
        if value.strip()
    )

    invalid = [
        value
        for value in variants_requested
        if value not in VARIANT_ORDER
    ]
    if invalid:
        raise RuntimeError(
            f"Unsupported R1.4 variants: {invalid}"
        )

    specs = variant_specs(protocol)

    tag = str(
        demo_cfg["retrieval"]["descriptor_tag"]
    )
    top_k = int(
        demo_cfg["retrieval"]["top_k"]
    )
    map_crs = str(
        demo_cfg["map"]["crs"]
    )
    target_crs = str(
        demo_cfg["export"]["target_crs"]
    )

    scripts = {
        "retrieval": ROOT / "scripts/villoc/blind_demo/stage6_blind_dino_topk_retrieval.py",
        "orb": ROOT / "scripts/villoc/s8_12e1_top20_verifier_reranker.py",
        "bootstrap": ROOT / "scripts/villoc/blind_demo/stage10b2_bootstrap_backend.py",
        "alignment": ROOT / "scripts/villoc/blind_demo/stage10b3_map_alignment_router.py",
        "temporal": ROOT / "scripts/villoc/blind_demo/stage10b4_temporal_router.py",
        "export": ROOT / "scripts/villoc/blind_demo/addon9_estimated_output_router.py",
        "freeze": ROOT / "scripts/villoc/blind_demo/stage10b5d_freeze_blind_submission.py",
        "evaluate": ROOT / "scripts/villoc/blind_demo/addon8_optional_evaluation_attachment.py",
    }

    for path in scripts.values():
        if not path.exists():
            raise FileNotFoundError(path)

    output_root.mkdir(parents=True, exist_ok=True)

    common_hashes_before = {
        name: sha256_file(path)
        for name, path in source.items()
    }

    results = []

    for variant in variants_requested:
        spec = specs[variant]
        run_root = output_root / "runs" / variant

        if run_root.exists():
            if args.overwrite:
                safe_remove_research_run(
                    run_root,
                    output_root,
                )
            else:
                raise RuntimeError(
                    f"R1.4 run root already exists: {run_root}. "
                    "Use --overwrite only after preserving any evidence."
                )

        logs = run_root / "logs/r1_4"
        logs.mkdir(parents=True, exist_ok=True)

        map_cache = resolve(spec["map_cache"])
        tile_index = resolve(spec["index_csv"])

        for path in (map_cache, tile_index):
            if not path.exists():
                raise FileNotFoundError(path)

        topk_csv = (
            run_root
            / "retrieval/s8_11d"
            / f"s8_11d_topk_{variant}_{tag}.csv"
        )
        orb_root = (
            run_root
            / "reports/s8_12e1_top20_verifier_reranker"
            / f"{variant}_orb_hybrid_top20_img518"
        )
        orb_summary = (
            orb_root
            / "s8_12e1_query_summary.csv"
        )
        orb_scores = (
            orb_root
            / "s8_12e1_all_candidate_verifier_scores.csv"
        )
        bootstrap_report = (
            run_root
            / "reports/blind_map_bootstrap/"
              "blind_map_bootstrap_report.json"
        )
        map_trajectory = (
            run_root
            / "trajectories/"
              "blind_map_aligned_relative_trajectory.csv"
        )
        temporal_fused = (
            run_root
            / "trajectories/"
              "blind_temporal_fused_trajectory.csv"
        )
        temporal_report = (
            run_root
            / "reports/blind_temporal_fusion/"
              "blind_temporal_fusion_report.json"
        )
        submission = (
            run_root
            / "trajectories/"
              "submission_estimated_trajectory.csv"
        )
        addon9_report = (
            run_root
            / "reports/addon9_estimated_latlon/"
              "estimated_latlon_export_report.json"
        )

        print()
        print("=" * 100)
        print(f"R1.4 — DOWNSTREAM REPLAY {variant}")
        print("=" * 100)

        stage_started = time.perf_counter()

        run([
            sys.executable,
            str(scripts["retrieval"]),
            "--query-cache", str(source["query_cache"]),
            "--map-cache", str(map_cache),
            "--run-root", str(run_root),
            "--variant", variant,
            "--tag", tag,
            "--top-k", str(top_k),
        ], logs / "05_dino_topk_retrieval.log")

        run([
            sys.executable,
            str(scripts["orb"]),
            "--config", str(demo_config_path),
            "--repo-root", str(ROOT),
            "--variant", variant,
            "--tag", tag,
            "--query-csv", str(source["manifest"]),
            "--topk-csv", str(topk_csv),
            "--tile-index-csv", str(tile_index),
            "--out-root", str(orb_root),
            "--top-n", str(top_k),
            "--policy", "hybrid",
            "--blind-only",
        ], logs / "06_orb_topk_verification.log")

        run([
            sys.executable,
            str(scripts["bootstrap"]),
            "--repo-root", str(ROOT),
            "--config", str(demo_config_path),
            "--run-root", str(run_root),
            "--source-run-root", str(source_root),
            "--candidate-csv", str(orb_scores),
            "--relative-csv", str(source["relative"]),
            "--manifest-csv", str(source["manifest"]),
        ], logs / "07_blind_map_bootstrap.log")

        run([
            sys.executable,
            str(scripts["alignment"]),
            "--repo-root", str(ROOT),
            "--config", str(demo_config_path),
            "--blind-manifest", str(source["manifest"]),
            "--raw-relative", str(source["relative"]),
            "--bootstrap-report", str(bootstrap_report),
            "--run-root", str(run_root),
            "--map-crs", map_crs,
        ], logs / "08_map_alignment.log")

        run([
            sys.executable,
            str(scripts["temporal"]),
            "--repo-root", str(ROOT),
            "--config", str(demo_config_path),
            "--map-trajectory", str(map_trajectory),
            "--bootstrap-report", str(bootstrap_report),
            "--absolute-query-summary", str(orb_summary),
            "--absolute-candidate-scores", str(orb_scores),
            "--run-root", str(run_root),
        ], logs / "09_temporal_authority.log")

        run([
            sys.executable,
            str(scripts["export"]),
            "--repo-root", str(ROOT),
            "--config", str(demo_config_path),
            "--fused-trajectory", str(temporal_fused),
            "--fusion-report", str(temporal_report),
            "--absolute-query-summary", str(orb_summary),
            "--run-root", str(run_root),
            "--tile-index", str(tile_index),
            "--source-crs", map_crs,
            "--target-crs", target_crs,
        ], logs / "10_estimated_output_export.log")

        run([
            sys.executable,
            str(scripts["freeze"]),
            "--submission", str(submission),
            "--addon9-report", str(addon9_report),
            "--run-root", str(run_root),
        ], logs / "17_freeze_blind_output.log")

        freeze_path = (
            run_root
            / "evaluation/blind_submission_freeze.json"
        )
        if not freeze_path.exists():
            raise RuntimeError(
                f"{variant}: freeze record missing."
            )

        # --------------------------------------------------------------
        # POST-FREEZE ONLY: attach reference and evaluate.
        # --------------------------------------------------------------
        run([
            sys.executable,
            str(scripts["evaluate"]),
            "--blind-manifest", str(source["manifest"]),
            "--reference-csv", str(reference_path),
            "--run-root", str(run_root),
            "--estimate-csv", str(submission),
            "--max-time-delta-s", "0.05",
        ], logs / "postfreeze_optional_evaluation.log")

        diagnostic = postfreeze_variant_diagnostics(
            variant=variant,
            run_root=run_root,
            tile_index_path=tile_index,
        )
        diagnostic["run_wall_s"] = float(
            time.perf_counter() - stage_started
        )
        diagnostic["map_cache"] = str(map_cache)
        diagnostic["map_cache_sha256"] = sha256_file(map_cache)
        diagnostic["tile_index"] = str(tile_index)
        diagnostic["tile_index_sha256"] = sha256_file(tile_index)
        diagnostic["run_root"] = str(run_root)

        results.append(diagnostic)

        print()
        print(f"{variant} downstream result")
        print("-" * 100)
        print("state:", diagnostic["localization_state"])
        print("maturity q:", diagnostic["matured_at_query_id"])
        print("map-state events:", diagnostic["map_state_event_count"])
        print("map-aligned rows:", diagnostic["map_aligned_rows"])
        print("accepted corrections:", diagnostic["accepted_correction_rows"])
        print(
            "trajectory RMSE:",
            diagnostic["trajectory_error"]["rmse_m"],
        )
        print(
            "trajectory p95:",
            diagnostic["trajectory_error"]["p95_m"],
        )
        print(
            "ORB selected contains:",
            diagnostic["orb_selected"]["selected_top1_contains_hits"],
            "/ 403",
        )
        print(
            "q57:",
            json.dumps(
                diagnostic["named_diagnostics"]["q57"]
            ),
        )
        print(
            "q228:",
            json.dumps(
                diagnostic["named_diagnostics"]["q228"]
            ),
        )

    common_hashes_after = {
        name: sha256_file(path)
        for name, path in source.items()
    }

    if common_hashes_before != common_hashes_after:
        raise RuntimeError(
            "Common blind pre-retrieval checkpoint changed during R1.4."
        )

    summary_df = pd.DataFrame([
        {
            "variant": item["variant"],
            "localization_state": item["localization_state"],
            "matured_at_query_id": item["matured_at_query_id"],
            "map_state_event_count": item["map_state_event_count"],
            "map_aligned_rows": item["map_aligned_rows"],
            "accepted_correction_rows": item["accepted_correction_rows"],
            "trajectory_rmse_m": item["trajectory_error"]["rmse_m"],
            "trajectory_mean_m": item["trajectory_error"]["mean_m"],
            "trajectory_median_m": item["trajectory_error"]["median_m"],
            "trajectory_p95_m": item["trajectory_error"]["p95_m"],
            "trajectory_max_m": item["trajectory_error"]["max_m"],
            "trajectory_final_m": item["trajectory_error"]["final_m"],
            "orb_selected_contains_hits": item["orb_selected"]["selected_top1_contains_hits"],
            "orb_selected_le40_hits": item["orb_selected"]["selected_top1_le40_hits"],
            "orb_selected_le80_hits": item["orb_selected"]["selected_top1_le80_hits"],
            "q57_orb_error_m": item["named_diagnostics"]["q57"]["orb_center_error_m_postfreeze"],
            "q57_is_state_update": item["named_diagnostics"]["q57"]["is_map_state_source_update"],
            "q57_trajectory_error_m": item["named_diagnostics"]["q57"]["trajectory_error_m"],
            "q228_orb_error_m": item["named_diagnostics"]["q228"]["orb_center_error_m_postfreeze"],
            "q228_is_state_update": item["named_diagnostics"]["q228"]["is_map_state_source_update"],
            "q228_trajectory_error_m": item["named_diagnostics"]["q228"]["trajectory_error_m"],
            "run_wall_s": item["run_wall_s"],
        }
        for item in results
    ])

    summary_csv = output_root / "r1_4_downstream_variant_summary.csv"
    summary_json = output_root / "r1_4_downstream_replay_report.json"

    summary_df.to_csv(summary_csv, index=False)

    report = {
        "stage": "R1.4",
        "status": "PASS_R1_DOWNSTREAM_TRANSLATION_REPLAY",
        "created_at_utc": now_utc(),
        "scientific_question": (
            "When only the map ground-footprint representation changes, "
            "which retrieval gains survive unchanged ORB verification and "
            "the frozen minimum_confident_v2 state pipeline?"
        ),
        "common_pre_retrieval_checkpoint": {
            "source_run_root": str(source_root),
            "inputs": {
                name: {
                    "path": str(path),
                    "sha256_before": common_hashes_before[name],
                    "sha256_after": common_hashes_after[name],
                    "unchanged": True,
                }
                for name, path in source.items()
            },
        },
        "frozen_chain": {
            "retrieval_top_k": top_k,
            "orb_policy": "hybrid",
            "bootstrap_backend": "minimum_confident_v2",
            "map_alignment_router": "canonical_map_state_timeline",
            "temporal_authority": "minimum_confident_v2",
            "reference_attached_only_after_submission_freeze": True,
        },
        "variants": {
            item["variant"]: item
            for item in results
        },
        "outputs": {
            "summary_csv": str(summary_csv),
            "report_json": str(summary_json),
        },
        "non_goals": [
            "no ORB tuning",
            "no bootstrap threshold tuning",
            "no temporal policy tuning",
            "no cross-scale fusion",
            "no query representation change",
            "no relative frontend rerun",
        ],
        "next_stage": {
            "stage": "R1 closeout / R2 decision",
            "action": (
                "Interpret which scale improvements are geometrically and "
                "statefully actionable before implementing cross-scale fusion."
            ),
        },
    }

    summary_json.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print()
    print("=" * 100)
    print("R1.4 — DOWNSTREAM TRANSLATION SUMMARY")
    print("=" * 100)
    print(summary_df.to_string(index=False))
    print()
    print("STATUS: PASS_R1_DOWNSTREAM_TRANSLATION_REPLAY")
    print("summary:", summary_csv)
    print("report:", summary_json)


if __name__ == "__main__":
    main()
