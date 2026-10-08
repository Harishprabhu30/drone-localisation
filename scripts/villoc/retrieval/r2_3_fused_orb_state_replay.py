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


ROOT = Path.cwd().resolve()

DEFAULT_DEMO_CONFIG = Path(
    "configs/demo_villoc_traj01_90deg_stable120m_blind_v2.yaml"
)
DEFAULT_SOURCE_RUN = Path(
    "outputs/demo_runs/traj01_blind_regression_001"
)
DEFAULT_R22_ROOT = Path(
    "outputs/research_runs/retrieval_candidate_pool_v2/"
    "r2_cross_scale_fusion/r2_2_blind_orb"
)
DEFAULT_R14_REPORT = Path(
    "outputs/research_runs/retrieval_candidate_pool_v2/"
    "r1_4_downstream_replay/r1_4_downstream_replay_report.json"
)
DEFAULT_REFERENCE = Path(
    "outputs/villoc/traj01_90deg_stable120m/trajectories/"
    "s8_3_reference_trajectory_V_1fps.csv"
)
DEFAULT_OUTPUT_ROOT = Path(
    "outputs/research_runs/retrieval_candidate_pool_v2/"
    "r2_cross_scale_fusion/r2_3_state_replay"
)

BLIND_MANIFEST_REL = Path(
    "metadata/blind_query_manifest.csv"
)
RAW_RELATIVE_REL = Path(
    "metadata/s8_xfeat_relative_frontend/"
    "s8r4_xfeat_relative_trajectory_blind_raw.csv"
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


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)

    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes", "y", "t"})
    )


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


def safe_remove_run(path: Path, output_root: Path) -> None:
    resolved = path.resolve()
    root = output_root.resolve()

    if root not in resolved.parents:
        raise RuntimeError(
            f"Refusing to remove path outside R2.3 output root: {resolved}"
        )

    if path.exists():
        shutil.rmtree(path)


def error_metrics(errors: np.ndarray) -> dict:
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
        "rmse_m": float(
            math.sqrt(np.mean(values * values))
        ),
        "mean_m": float(np.mean(values)),
        "median_m": float(np.median(values)),
        "p95_m": float(np.percentile(values, 95)),
        "max_m": float(np.max(values)),
        "final_m": float(values[-1]),
    }


def first_threshold_crossing(
    query_ids: np.ndarray,
    errors: np.ndarray,
    threshold_m: float,
) -> dict | None:
    valid = np.isfinite(errors)
    hit = np.flatnonzero(
        valid & (errors > float(threshold_m))
    )

    if len(hit) == 0:
        return None

    index = int(hit[0])
    return {
        "query_id": int(query_ids[index]),
        "error_m": float(errors[index]),
    }


def event_for_query(events: list[dict], query_id: int) -> dict | None:
    for event in events:
        source_q = event.get("source_update_query_id")
        effective_q = event.get("effective_query_id")

        if (
            source_q is not None
            and int(source_q) == int(query_id)
        ) or (
            effective_q is not None
            and int(effective_q) == int(query_id)
        ):
            return event

    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "R2.3 replay only the frozen downstream state chain using the "
            "exact R2.2 fused+ORB blind candidate stream. Retrieval and ORB "
            "are not recomputed."
        )
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
        "--r2-2-root",
        type=Path,
        default=DEFAULT_R22_ROOT,
    )
    parser.add_argument(
        "--r1-4-report",
        type=Path,
        default=DEFAULT_R14_REPORT,
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
        "--overwrite",
        action="store_true",
        help="Replace an existing R2.3 research run only.",
    )
    args = parser.parse_args()

    demo_config_path = resolve(args.demo_config)
    source_root = resolve(args.source_run_root)
    r22_root = resolve(args.r2_2_root)
    r14_report_path = resolve(args.r1_4_report)
    reference_path = resolve(args.reference_csv)
    output_root = resolve(args.output_root)
    run_root = output_root / "run"

    for path in (
        demo_config_path,
        r14_report_path,
        reference_path,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    if run_root.exists():
        if args.overwrite:
            safe_remove_run(run_root, output_root)
        else:
            raise RuntimeError(
                f"R2.3 run root already exists: {run_root}. "
                "Use --overwrite only after preserving any evidence."
            )

    demo_cfg = yaml.safe_load(
        demo_config_path.read_text()
    )

    if (
        demo_cfg.get("bootstrap", {}).get("backend")
        != "minimum_confident_v2"
    ):
        raise RuntimeError(
            "R2.3 requires the frozen minimum_confident_v2 backend."
        )

    manifest = source_root / BLIND_MANIFEST_REL
    relative = source_root / RAW_RELATIVE_REL

    r22_report = (
        r22_root
        / "reports/r2_2_triple_fusion_orb_report.json"
    )
    mixed_index = (
        r22_root
        / "metadata/r2_2_mixed_representative_tile_index.csv"
    )
    fused_top20 = (
        r22_root
        / "retrieval/r2_2_blind_triple_fused_top20.csv"
    )
    orb_root = (
        r22_root
        / "reports/r2_2_triple_orb_hybrid_top20"
    )
    orb_scores = (
        orb_root
        / "s8_12e1_all_candidate_verifier_scores.csv"
    )
    orb_summary = (
        orb_root
        / "s8_12e1_query_summary.csv"
    )

    required_inputs = {
        "manifest": manifest,
        "relative": relative,
        "r2_2_report": r22_report,
        "mixed_tile_index": mixed_index,
        "fused_top20": fused_top20,
        "orb_scores": orb_scores,
        "orb_summary": orb_summary,
    }

    missing = [
        f"{name}: {path}"
        for name, path in required_inputs.items()
        if not path.exists()
    ]
    if missing:
        raise RuntimeError(
            "R2.3 required blind input is missing:\n"
            + "\n".join(missing)
        )

    r22 = load_json(r22_report)

    if (
        r22.get("status")
        != "PASS_R2_TRIPLE_FUSION_ORB_TRANSLATION"
    ):
        raise RuntimeError(
            "R2.2 has not passed; R2.3 state replay is blocked."
        )

    manifest_df = pd.read_csv(manifest)
    relative_df = pd.read_csv(relative)
    orb_scores_df = pd.read_csv(orb_scores)
    orb_summary_df = pd.read_csv(orb_summary)

    for name, frame in (
        ("manifest", manifest_df),
        ("relative", relative_df),
        ("ORB summary", orb_summary_df),
    ):
        if len(frame) != 403:
            raise RuntimeError(
                f"Expected 403 rows in {name}, got {len(frame)}."
            )

    if len(orb_scores_df) != 8060:
        raise RuntimeError(
            "Expected exactly 403 x 20 = 8060 R2.2 ORB candidate rows, "
            f"got {len(orb_scores_df)}."
        )

    if "reference_available" in manifest_df.columns:
        if bool_series(
            manifest_df["reference_available"]
        ).any():
            raise RuntimeError(
                "Blind manifest contains reference_available=true."
            )

    forbidden_candidate_columns = {
        "reference_x_m",
        "reference_y_m",
        "latitude",
        "longitude",
        "ground_truth_x_m",
        "ground_truth_y_m",
        "oracle_tile_id",
        "oracle_best_tile_id",
        "chosen_error_m",
        "candidate_body_error_m",
        "reranked_top1_error_m",
        "gt_easting",
        "gt_northing",
        "projected_error_m",
        "tile_center_error_m",
        "selected_center_error_m",
        "selected_contains_query",
    }
    leaked = sorted(
        forbidden_candidate_columns
        & set(orb_scores_df.columns)
    )
    if leaked:
        raise RuntimeError(
            "Reference/evaluation columns leaked into R2.2 ORB scores: "
            f"{leaked}"
        )

    input_hashes_before = {
        name: sha256_file(path)
        for name, path in required_inputs.items()
    }

    map_crs = str(demo_cfg["map"]["crs"])
    target_crs = str(demo_cfg["export"]["target_crs"])

    scripts = {
        "bootstrap": (
            ROOT
            / "scripts/villoc/blind_demo/stage10b2_bootstrap_backend.py"
        ),
        "alignment": (
            ROOT
            / "scripts/villoc/blind_demo/stage10b3_map_alignment_router.py"
        ),
        "temporal": (
            ROOT
            / "scripts/villoc/blind_demo/stage10b4_temporal_router.py"
        ),
        "export": (
            ROOT
            / "scripts/villoc/blind_demo/addon9_estimated_output_router.py"
        ),
        "freeze": (
            ROOT
            / "scripts/villoc/blind_demo/stage10b5d_freeze_blind_submission.py"
        ),
        "evaluate": (
            ROOT
            / "scripts/villoc/blind_demo/addon8_optional_evaluation_attachment.py"
        ),
    }

    for path in scripts.values():
        if not path.exists():
            raise FileNotFoundError(path)

    logs = run_root / "logs/r2_3"
    logs.mkdir(parents=True, exist_ok=True)

    bootstrap_report = (
        run_root
        / "reports/blind_map_bootstrap/"
          "blind_map_bootstrap_report.json"
    )
    timeline_path = (
        run_root
        / "reports/blind_map_bootstrap/"
          "canonical_map_state_timeline.json"
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
        / "trajectories/submission_estimated_trajectory.csv"
    )
    addon9_report = (
        run_root
        / "reports/addon9_estimated_latlon/"
          "estimated_latlon_export_report.json"
    )
    freeze_path = (
        run_root
        / "evaluation/blind_submission_freeze.json"
    )
    attachment_path = (
        run_root
        / "evaluation/reference_attachment.csv"
    )

    print("=" * 100)
    print("R2.3 — FROZEN FUSED+ORB STATE REPLAY")
    print("=" * 100)
    print("retrieval recomputed : false")
    print("ORB recomputed       : false")
    print("state policy changed : false")
    print("R2.2 ORB scores SHA  :", input_hashes_before["orb_scores"])
    print("R2.2 fused Top20 SHA :", input_hashes_before["fused_top20"])

    started = time.perf_counter()

    # ------------------------------------------------------------------
    # BLIND DOWNSTREAM CHAIN ONLY.
    # ------------------------------------------------------------------
    run([
        sys.executable,
        str(scripts["bootstrap"]),
        "--repo-root", str(ROOT),
        "--config", str(demo_config_path),
        "--run-root", str(run_root),
        "--source-run-root", str(source_root),
        "--candidate-csv", str(orb_scores),
        "--relative-csv", str(relative),
        "--manifest-csv", str(manifest),
    ], logs / "07_blind_map_bootstrap.log")

    run([
        sys.executable,
        str(scripts["alignment"]),
        "--repo-root", str(ROOT),
        "--config", str(demo_config_path),
        "--blind-manifest", str(manifest),
        "--raw-relative", str(relative),
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
        "--tile-index", str(mixed_index),
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

    if not freeze_path.exists():
        raise RuntimeError(
            "R2.3 blind submission did not freeze successfully."
        )

    submission_sha_at_freeze = sha256_file(submission)

    # ------------------------------------------------------------------
    # POST-FREEZE ONLY: attach reference and compute trajectory metrics.
    # ------------------------------------------------------------------
    run([
        sys.executable,
        str(scripts["evaluate"]),
        "--blind-manifest", str(manifest),
        "--reference-csv", str(reference_path),
        "--run-root", str(run_root),
        "--estimate-csv", str(submission),
        "--max-time-delta-s", "0.05",
    ], logs / "postfreeze_optional_evaluation.log")

    if sha256_file(submission) != submission_sha_at_freeze:
        raise RuntimeError(
            "Frozen R2.3 submission changed during post-freeze evaluation."
        )

    input_hashes_after = {
        name: sha256_file(path)
        for name, path in required_inputs.items()
    }

    if input_hashes_before != input_hashes_after:
        raise RuntimeError(
            "R2.2/source blind inputs changed during R2.3."
        )

    for path in (
        bootstrap_report,
        timeline_path,
        submission,
        attachment_path,
    ):
        if not path.exists():
            raise RuntimeError(
                f"R2.3 expected output missing: {path}"
            )

    bootstrap = load_json(bootstrap_report)
    timeline = load_json(timeline_path)
    freeze = load_json(freeze_path)

    expected_submission_sha = freeze["submission"]["sha256"]
    actual_submission_sha = sha256_file(submission)

    if actual_submission_sha != expected_submission_sha:
        raise RuntimeError(
            "R2.3 frozen submission SHA mismatch."
        )

    submission_df = pd.read_csv(submission)
    attachment = pd.read_csv(attachment_path)

    if len(submission_df) != 403 or len(attachment) != 403:
        raise RuntimeError(
            "R2.3 expected 403-row submission and evaluation attachment."
        )

    if "eval_position_error_m" not in attachment.columns:
        raise RuntimeError(
            "Post-freeze evaluation did not produce eval_position_error_m."
        )

    attachment["query_id"] = pd.to_numeric(
        attachment["query_id"],
        errors="raise",
    ).astype(int)

    errors = pd.to_numeric(
        attachment["eval_position_error_m"],
        errors="coerce",
    ).to_numpy(dtype=float)
    query_ids = attachment["query_id"].to_numpy(dtype=int)

    trajectory = error_metrics(errors)

    max_index = int(np.nanargmax(errors))
    max_error = {
        "query_id": int(query_ids[max_index]),
        "error_m": float(errors[max_index]),
    }

    threshold_crossings = {
        str(int(threshold)): first_threshold_crossing(
            query_ids,
            errors,
            threshold,
        )
        for threshold in (100.0, 200.0, 300.0)
    }

    available = bool_series(
        submission_df["map_aligned_available"]
    )
    accepted = bool_series(
        submission_df["accepted_correction"]
    )

    events = list(timeline.get("events", []))
    source_update_queries = [
        int(event["source_update_query_id"])
        for event in events
        if event.get("source_update_query_id") is not None
    ]

    named = {}

    r22_named = r22.get("named_diagnostics", {})

    for qid in (57, 228, 390):
        match = attachment[
            attachment["query_id"] == qid
        ]

        named[f"q{qid}"] = {
            "trajectory_error_m": (
                float(
                    match.iloc[0]["eval_position_error_m"]
                )
                if len(match) == 1
                and pd.notna(
                    match.iloc[0]["eval_position_error_m"]
                )
                else None
            ),
            "is_map_state_source_update": (
                qid in source_update_queries
            ),
            "map_state_event": event_for_query(
                events,
                qid,
            ),
            "r2_2_orb_selection": r22_named.get(
                f"q{qid}"
            ),
        }

    r14 = load_json(r14_report_path)

    controls = {
        variant: {
            "trajectory_error": (
                r14["variants"][variant]["trajectory_error"]
            ),
            "matured_at_query_id": (
                r14["variants"][variant]["matured_at_query_id"]
            ),
            "map_state_event_count": (
                r14["variants"][variant]["map_state_event_count"]
            ),
            "accepted_correction_rows": (
                r14["variants"][variant][
                    "accepted_correction_rows"
                ]
            ),
        }
        for variant in (
            "512_s256",
            "768_s256",
            "1024_s256",
        )
    }

    result = {
        "stage": "R2.3",
        "status": "PASS_R2_FUSED_ORB_STATE_REPLAY",
        "created_at_utc": now_utc(),
        "scientific_question": (
            "Does the exact R2.2 fused+ORB blind candidate stream, with "
            "minimum_confident_v2 and all downstream policies unchanged, "
            "avoid the catastrophic q390-class state failure and improve "
            "end-to-end trajectory behavior?"
        ),
        "reuse_contract": {
            "retrieval_recomputed": False,
            "orb_recomputed": False,
            "r2_2_artifacts_reused_by_sha256": {
                name: {
                    "path": str(required_inputs[name]),
                    "sha256_before": input_hashes_before[name],
                    "sha256_after": input_hashes_after[name],
                    "unchanged": True,
                }
                for name in (
                    "fused_top20",
                    "mixed_tile_index",
                    "orb_scores",
                    "orb_summary",
                    "r2_2_report",
                )
            },
        },
        "blind_chain": {
            "bootstrap_backend": "minimum_confident_v2",
            "state_policy_changed": False,
            "map_alignment_changed": False,
            "temporal_authority_changed": False,
            "reference_available_before_freeze": False,
            "submission_sha256": actual_submission_sha,
        },
        "result": {
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
            "trajectory_error": trajectory,
            "max_error": max_error,
            "first_error_threshold_crossings": threshold_crossings,
            "named_diagnostics": named,
        },
        "single_scale_controls_from_r1_4": controls,
        "runtime": {
            "downstream_and_evaluation_wall_s": float(
                time.perf_counter() - started
            ),
        },
        "outputs": {
            "run_root": str(run_root),
            "bootstrap_report": str(bootstrap_report),
            "canonical_map_state_timeline": str(timeline_path),
            "submission": str(submission),
            "reference_attachment": str(attachment_path),
        },
        "scope_guarantees": {
            "fusion_changed_in_r2_3": False,
            "orb_changed_in_r2_3": False,
            "bootstrap_changed_in_r2_3": False,
            "state_policy_changed_in_r2_3": False,
            "temporal_policy_changed_in_r2_3": False,
            "reference_used_before_freeze": False,
        },
        "next_stage": {
            "action": (
                "Interpret trajectory and named failure cases. If q57 remains "
                "a candidate-selection failure while q390 is state-safe, open "
                "a separate verifier/reranker research stage rather than "
                "changing R2 state policy."
            ),
        },
    }

    report_path = (
        output_root
        / "r2_3_fused_orb_state_replay_report.json"
    )
    report_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    report_path.write_text(
        json.dumps(
            result,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    summary_path = (
        output_root
        / "r2_3_state_replay_summary.csv"
    )
    summary_df = pd.DataFrame(
        [
            {
                "stream": "r2_triple_fused_orb",
                "localization_state": result["result"][
                    "localization_state"
                ],
                "matured_at_query_id": result["result"][
                    "matured_at_query_id"
                ],
                "map_state_event_count": result["result"][
                    "map_state_event_count"
                ],
                "accepted_correction_rows": result["result"][
                    "accepted_correction_rows"
                ],
                "map_aligned_rows": result["result"][
                    "map_aligned_rows"
                ],
                "trajectory_rmse_m": trajectory["rmse_m"],
                "trajectory_mean_m": trajectory["mean_m"],
                "trajectory_median_m": trajectory["median_m"],
                "trajectory_p95_m": trajectory["p95_m"],
                "trajectory_max_m": trajectory["max_m"],
                "trajectory_final_m": trajectory["final_m"],
                "max_error_query_id": max_error["query_id"],
                "first_gt100_query_id": (
                    threshold_crossings["100"]["query_id"]
                    if threshold_crossings["100"]
                    else None
                ),
                "q57_state_update": named["q57"][
                    "is_map_state_source_update"
                ],
                "q57_trajectory_error_m": named["q57"][
                    "trajectory_error_m"
                ],
                "q228_state_update": named["q228"][
                    "is_map_state_source_update"
                ],
                "q228_trajectory_error_m": named["q228"][
                    "trajectory_error_m"
                ],
                "q390_state_update": named["q390"][
                    "is_map_state_source_update"
                ],
                "q390_trajectory_error_m": named["q390"][
                    "trajectory_error_m"
                ],
            }
        ]
    )
    summary_df.to_csv(summary_path, index=False)

    print()
    print("=" * 100)
    print("R2.3 — FUSED+ORB STATE REPLAY SUMMARY")
    print("=" * 100)
    print(summary_df.to_string(index=False))

    print()
    print("Single-scale R1.4 controls")
    print("-" * 100)
    for variant, control in controls.items():
        error = control["trajectory_error"]
        print(
            f"{variant:10s} "
            f"RMSE={error['rmse_m']:.3f} "
            f"p95={error['p95_m']:.3f} "
            f"max={error['max_m']:.3f} "
            f"final={error['final_m']:.3f} "
            f"events={control['map_state_event_count']}"
        )

    for qid in (57, 228, 390):
        print()
        print(f"q{qid}:")
        print(
            json.dumps(
                named[f"q{qid}"],
                indent=2,
                default=str,
            )
        )

    print()
    print("STATUS: PASS_R2_FUSED_ORB_STATE_REPLAY")
    print("summary:", summary_path)
    print("report:", report_path)


if __name__ == "__main__":
    main()
