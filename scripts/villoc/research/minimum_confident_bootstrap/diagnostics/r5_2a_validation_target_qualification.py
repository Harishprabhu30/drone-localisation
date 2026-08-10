#!/usr/bin/env python3
"""
R5.2A — validation-target qualification.

Purpose
-------
Identify candidate trajectories/run roots for R5.2 without consuming
reference/GT values.

This stage:
    * scans local Villoc raw datasets;
    * scans prepared demo/research run roots;
    * checks whether the exact blind inputs required by
      minimum_confident_bootstrap_v2.py already exist;
    * reports provenance/usage restrictions;
    * does NOT read SRT contents;
    * does NOT evaluate accuracy;
    * does NOT select a policy.

Important roles
---------------
traj01
    development dataset -> must not be independent validation.

original Villoc 90_deg
    prior benchmark -> useful cross-sequence stress test,
    but not a pristine policy-selection holdout.

recorded-flight demonstration
    intended final blind holdout -> preserve unless we consciously
    decide to convert it into a validation trajectory.

unknown new Villoc folders/runs
    potential independent validation candidates -> inspect next.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def count_files(root: Path, suffix: str) -> int:
    if not root.exists():
        return 0
    return sum(
        1
        for p in root.rglob("*")
        if p.is_file()
        and p.suffix.lower() == suffix.lower()
    )


def find_some(root: Path, patterns, limit=6):
    if not root.exists():
        return []

    found = []

    for pattern in patterns:
        for path in root.rglob(pattern):
            if path.is_file():
                resolved = str(path.resolve())

                if resolved not in found:
                    found.append(resolved)

                if len(found) >= limit:
                    return found

    return found


def classify_name(name: str):

    lower = name.lower()

    if "traj01" in lower:
        return (
            "DEVELOPMENT_EXCLUDE",
            "Used to develop R3/R4/R5 architecture; not independent."
        )

    if (
        lower == "90_deg"
        or "90deg" in lower
        or "90_deg" in lower
    ):
        return (
            "PRIOR_BENCHMARK_STRESS_ONLY",
            "Compatible nadir benchmark but previously studied; not pristine."
        )

    if (
        "45deg" in lower
        or "45_deg" in lower
    ):
        return (
            "VIEW_GEOMETRY_MISMATCH",
            "Oblique-view dataset; frozen R3-v2 contract is near-nadir."
        )

    if (
        "demonstration" in lower
        or "recorded" in lower
        or "blind_villoc" in lower
        or "blind-villoc" in lower
    ):
        return (
            "FINAL_HOLDOUT_CANDIDATE",
            "Compatible blind recorded-flight target; preserve for final demo if possible."
        )

    return (
        "UNCLASSIFIED_CANDIDATE",
        "Potential new validation target; provenance/view geometry must be reviewed."
    )


def inspect_run_root(run_root: Path):

    candidate = (
        run_root
        / "reports/"
          "s8_12e1_top20_verifier_reranker/"
          "512_s256_orb_hybrid_top20_img518/"
          "s8_12e1_all_candidate_verifier_scores.csv"
    )

    relative = (
        run_root
        / "metadata/"
          "s8_xfeat_relative_frontend/"
          "s8r4_xfeat_relative_trajectory_blind_raw.csv"
    )

    manifest = (
        run_root
        / "metadata/"
          "blind_query_manifest.csv"
    )

    role, note = classify_name(
        run_root.name
    )

    candidate_rows = None
    candidate_queries = None
    manifest_rows = None
    relative_rows = None

    suspicious_columns = []

    for label, path in [
        ("candidate", candidate),
        ("relative", relative),
        ("manifest", manifest),
    ]:

        if not path.exists():
            continue

        try:
            frame = pd.read_csv(
                path,
                nrows=0,
            )

            for column in frame.columns:

                c = str(column).lower()

                if any(
                    token in c
                    for token in [
                        "ground_truth",
                        "gt_",
                        "reference_lat",
                        "reference_lon",
                        "error_m",
                        "oracle",
                    ]
                ):

                    suspicious_columns.append(
                        f"{label}:{column}"
                    )

        except Exception as exc:

            suspicious_columns.append(
                f"{label}:HEADER_READ_ERROR:{exc}"
            )

    if candidate.exists():

        try:
            c = pd.read_csv(
                candidate,
                usecols=[
                    "query_id",
                ],
            )

            candidate_rows = int(
                len(c)
            )

            candidate_queries = int(
                c["query_id"].nunique()
            )

        except Exception:
            pass

    if relative.exists():

        try:
            relative_rows = sum(
                1
                for _ in relative.open(
                    "r",
                    encoding="utf-8",
                    errors="replace",
                )
            ) - 1

        except Exception:
            pass

    if manifest.exists():

        try:
            manifest_rows = sum(
                1
                for _ in manifest.open(
                    "r",
                    encoding="utf-8",
                    errors="replace",
                )
            ) - 1

        except Exception:
            pass

    ready = bool(
        candidate.exists()
        and relative.exists()
        and manifest.exists()
        and not suspicious_columns
    )

    return {
        "run_root":
            str(
                run_root.resolve()
            ),

        "run_name":
            run_root.name,

        "role":
            role,

        "role_note":
            note,

        "candidate_csv":
            str(candidate),

        "candidate_exists":
            candidate.exists(),

        "relative_csv":
            str(relative),

        "relative_exists":
            relative.exists(),

        "manifest_csv":
            str(manifest),

        "manifest_exists":
            manifest.exists(),

        "candidate_rows":
            candidate_rows,

        "candidate_queries":
            candidate_queries,

        "relative_rows":
            relative_rows,

        "manifest_rows":
            manifest_rows,

        "suspicious_header_columns":
            ";".join(
                suspicious_columns
            ),

        "r5_v2_input_ready":
            ready,
    }


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--repo-root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--research-root",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    repo = args.repo_root.resolve()
    research = args.research_root.resolve()

    raw_root = (
        repo
        / "data/raw/villoc"
    )

    demo_runs = (
        repo
        / "outputs/demo_runs"
    )

    villoc_outputs = (
        repo
        / "outputs/villoc"
    )


    # ========================================================
    # A. Raw Villoc dataset inventory
    # ========================================================

    raw_rows = []

    if raw_root.exists():

        for child in sorted(
            p
            for p in raw_root.iterdir()
            if p.is_dir()
        ):

            role, note = classify_name(
                child.name
            )

            raw_rows.append(
                {
                    "dataset":
                        child.name,

                    "path":
                        str(
                            child.resolve()
                        ),

                    "mp4_count":
                        count_files(
                            child,
                            ".mp4",
                        ),

                    "srt_file_count_presence_only":
                        count_files(
                            child,
                            ".srt",
                        ),

                    "role":
                        role,

                    "role_note":
                        note,
                }
            )


    raw_df = pd.DataFrame(
        raw_rows
    )


    # ========================================================
    # B. Prepared run-root inventory
    # ========================================================

    candidate_roots = []

    for parent in [
        demo_runs,
        villoc_outputs,
    ]:

        if not parent.exists():
            continue

        for child in parent.iterdir():

            if child.is_dir():

                candidate_roots.append(
                    child
                )


    run_rows = [
        inspect_run_root(
            root
        )
        for root in sorted(
            candidate_roots,
            key=lambda p:
                str(p)
        )
    ]


    run_df = pd.DataFrame(
        run_rows
    )


    # ========================================================
    # C. Common frozen R5 prerequisites
    # ========================================================

    common = {
        "v2_script":
            (
                repo
                / "scripts/villoc/research/"
                  "minimum_confident_bootstrap/"
                  "minimum_confident_bootstrap_v2.py"
            ),

        "r5_0_contract":
            (
                research
                / "r5_0_r3v2_architecture_contract.json"
            ),

        "map_tile_index":
            (
                repo
                / "outputs/villoc/90_deg/metadata/"
                  "s8_9_satellite_tile_index_512_s256.csv"
            ),
    }


    map_cache_matches = find_some(
        repo
        / "outputs/villoc/90_deg/descriptors",
        [
            "*512_s256*img518*.npz",
            "*512_s256*.npz",
        ],
        limit=10,
    )


    # ========================================================
    # D. Recommendation logic
    # ========================================================

    ready_unknown = []

    ready_prior = []

    ready_holdout = []


    if len(run_df):

        ready = run_df[
            run_df[
                "r5_v2_input_ready"
            ]
            == True
        ]


        ready_unknown = (
            ready[
                ready[
                    "role"
                ]
                == "UNCLASSIFIED_CANDIDATE"
            ][
                "run_root"
            ]
            .tolist()
        )


        ready_prior = (
            ready[
                ready[
                    "role"
                ]
                == "PRIOR_BENCHMARK_STRESS_ONLY"
            ][
                "run_root"
            ]
            .tolist()
        )


        ready_holdout = (
            ready[
                ready[
                    "role"
                ]
                == "FINAL_HOLDOUT_CANDIDATE"
            ][
                "run_root"
            ]
            .tolist()
        )


    if ready_unknown:

        decision = (
            "USE_NEW_UNCLASSIFIED_CANDIDATE_AFTER_PROVENANCE_CHECK"
        )

        explanation = (
            "At least one prepared run appears independent. "
            "Review its provenance/view geometry before R5.2B."
        )

    elif ready_prior:

        decision = (
            "USE_PRIOR_BENCHMARK_AS_STRESS_TEST_ONLY"
        )

        explanation = (
            "No clean prepared candidate found, but a prior benchmark "
            "is v2-ready. It may test portability, not pristine policy selection."
        )

    elif ready_holdout:

        decision = (
            "PRESERVE_DEMO_HOLDOUT_OR_CONSCIOUSLY_CONVERT_IT"
        )

        explanation = (
            "The only directly v2-ready compatible target appears to be "
            "the final recorded-flight holdout. Do not consume it for policy "
            "selection unless we intentionally reserve another final demo."
        )

    else:

        decision = (
            "BUILD_BLIND_SAFE_VALIDATION_REPLAY"
        )

        explanation = (
            "No non-development run root currently contains the complete "
            "R5-v2 blind input trio. Build a validation replay before running R5.2."
        )


    # ========================================================
    # Save audit
    # ========================================================

    out_dir = (
        research
        / "r5_2_validation"
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    raw_out = (
        out_dir
        / "r5_2a_raw_dataset_inventory.csv"
    )

    run_out = (
        out_dir
        / "r5_2a_prepared_run_inventory.csv"
    )

    report_out = (
        out_dir
        / "r5_2a_validation_target_qualification.json"
    )


    raw_df.to_csv(
        raw_out,
        index=False,
    )


    run_df.to_csv(
        run_out,
        index=False,
    )


    report = {
        "stage":
            "R5.2A_VALIDATION_TARGET_QUALIFICATION",

        "gt_values_read":
            False,

        "srt_contents_read":
            False,

        "policy_selected":
            False,

        "common_prerequisites": {
            key: {
                "path":
                    str(path),

                "exists":
                    path.exists(),
            }
            for key, path
            in common.items()
        },

        "map_descriptor_cache_matches":
            map_cache_matches,

        "ready_unknown_candidates":
            ready_unknown,

        "ready_prior_benchmarks":
            ready_prior,

        "ready_final_holdouts":
            ready_holdout,

        "decision":
            decision,

        "explanation":
            explanation,
    }


    report_out.write_text(
        json.dumps(
            report,
            indent=2,
        )
    )


    # ========================================================
    # Print
    # ========================================================

    print()
    print("=" * 118)
    print(
        "R5.2A — VALIDATION TARGET QUALIFICATION"
    )
    print("=" * 118)


    print()
    print(
        "RAW VILLOC DATASETS "
        "(SRT presence only; contents NOT read)"
    )
    print("-" * 118)


    if len(raw_df):

        print(
            raw_df.to_string(
                index=False,
            )
        )

    else:

        print(
            "No data/raw/villoc subdirectories found."
        )


    print()
    print(
        "PREPARED RUN ROOTS"
    )
    print("-" * 118)


    if len(run_df):

        show = [
            "run_name",
            "role",
            "candidate_exists",
            "relative_exists",
            "manifest_exists",
            "candidate_queries",
            "r5_v2_input_ready",
            "suspicious_header_columns",
        ]

        print(
            run_df[
                show
            ].to_string(
                index=False,
            )
        )

    else:

        print(
            "No prepared output roots found."
        )


    print()
    print(
        "COMMON R5-v2 PREREQUISITES"
    )
    print("-" * 118)


    for name, path in (
        common.items()
    ):

        print(
            f"{name:24s}: "
            f"{'PASS' if path.exists() else 'MISSING'}"
        )

        print(
            f"  {path}"
        )


    print(
        "map descriptor caches:",
        len(
            map_cache_matches
        ),
    )


    print()
    print("=" * 118)
    print("QUALIFICATION DECISION")
    print("=" * 118)

    print(
        "decision:",
        decision,
    )

    print(
        "reason:",
        explanation,
    )

    print(
        "GT values read:",
        False,
    )

    print(
        "SRT contents read:",
        False,
    )

    print(
        "policy selected:",
        False,
    )

    print()
    print(
        "raw inventory:",
        raw_out,
    )

    print(
        "run inventory:",
        run_out,
    )

    print(
        "report:",
        report_out,
    )

    print()

    print(
        "STATUS: "
        "PASS_R5_2A_VALIDATION_TARGET_QUALIFICATION"
    )


if __name__ == "__main__":
    main()
