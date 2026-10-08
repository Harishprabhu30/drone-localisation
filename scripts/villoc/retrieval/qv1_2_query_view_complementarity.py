#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


ROOT = Path.cwd().resolve()
DEFAULT_CONFIG = Path(
    "configs/research/query_view_candidate_generation_v1.yaml"
)
DEFAULT_QV11_ROOT = Path(
    "outputs/research_runs/query_view_candidate_generation_v1/qv1_1"
)
DEFAULT_OUTPUT_ROOT = Path(
    "outputs/research_runs/query_view_candidate_generation_v1/qv1_2"
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


def bool_col(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes", "y", "t"})
    )


def category(center_good: bool, alt_good: bool) -> str:
    if center_good and alt_good:
        return "both"
    if center_good and not alt_good:
        return "center_only"
    if not center_good and alt_good:
        return "alt_only_rescue"
    return "neither"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
    )
    parser.add_argument(
        "--qv1-1-root",
        type=Path,
        default=DEFAULT_QV11_ROOT,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    args = parser.parse_args()

    cfg = yaml.safe_load(
        resolve(args.config).read_text()
    )
    qv = cfg["qv1_2"]
    qv11_root = resolve(args.qv1_1_root)
    output_root = resolve(args.output_root)

    ranking_dir = qv11_root / "blind_rankings"
    eval_dir = qv11_root / "postfreeze_evaluation"
    qv11_report_path = (
        qv11_root
        / "qv1_1_query_view_retrieval_report.json"
    )

    views = list(
        qv["independent_views"]
    )

    ranking_paths = {
        view: ranking_dir
        / f"qv1_1_{view}_ranking.csv"
        for view in views
    }
    eval_paths = {
        view: eval_dir
        / f"qv1_1_{view}_query_eval.csv"
        for view in views
    }

    for path in [
        qv11_report_path,
        *ranking_paths.values(),
        *eval_paths.values(),
    ]:
        if not path.exists():
            raise FileNotFoundError(path)

    qv11 = json.loads(
        qv11_report_path.read_text()
    )
    if (
        qv11.get("status")
        != "PASS_QV1_QUERY_VIEW_RETRIEVAL_COMPARISON"
    ):
        raise RuntimeError(
            "QV1.1 gate has not passed"
        )

    depth = int(
        qv["ranking_depth"]
    )

    rankings = {}
    evaluations = {}

    for view in views:
        ranking = pd.read_csv(
            ranking_paths[view]
        ).copy()
        ranking["query_id"] = pd.to_numeric(
            ranking["query_id"],
            errors="raise",
        ).astype(int)
        ranking["rank"] = pd.to_numeric(
            ranking["rank"],
            errors="raise",
        ).astype(int)
        ranking["tile_id"] = ranking[
            "tile_id"
        ].astype(str)
        rankings[view] = ranking[
            ranking["rank"] <= depth
        ].copy()

        ev = pd.read_csv(
            eval_paths[view]
        ).copy()
        ev["query_id"] = pd.to_numeric(
            ev["query_id"],
            errors="raise",
        ).astype(int)
        for col in (
            "top1_contains",
            "top1_le40",
            "top1_le80",
            "contain_r20",
            "le40_r20",
            "le80_r20",
        ):
            ev[col] = bool_col(
                ev[col]
            )
        evaluations[view] = ev.set_index(
            "query_id"
        )

    qids = sorted(
        evaluations[
            "center_square"
        ].index.astype(int).tolist()
    )

    if len(qids) != 403:
        raise RuntimeError(
            f"Expected 403 queries, got {len(qids)}"
        )

    set_lookup = {
        view: {
            int(qid): set(
                group[
                    "tile_id"
                ].astype(str).tolist()
            )
            for qid, group in frame.groupby(
                "query_id"
            )
        }
        for view, frame in rankings.items()
    }

    rows = []

    metrics = (
        "top1_contains",
        "top1_le40",
        "top1_le80",
        "contain_r20",
        "le40_r20",
        "le80_r20",
    )

    for qid in qids:
        row = {
            "query_id": int(qid),
        }

        for view in views:
            ev = evaluations[
                view
            ].loc[qid]
            for metric in metrics:
                row[
                    f"{view}_{metric}"
                ] = bool(
                    ev[metric]
                )

        center_set = set_lookup[
            "center_square"
        ][qid]

        for view in (
            "left_square",
            "right_square",
            "resize_square",
        ):
            other = set_lookup[
                view
            ][qid]
            inter = len(
                center_set
                & other
            )
            union = len(
                center_set
                | other
            )
            row[
                f"center_{view}_top20_intersection"
            ] = int(inter)
            row[
                f"center_{view}_top20_jaccard"
            ] = (
                float(inter / union)
                if union
                else 1.0
            )

        lcr_sets = [
            set_lookup[
                view
            ][qid]
            for view in (
                "left_square",
                "center_square",
                "right_square",
            )
        ]
        all_sets = [
            set_lookup[
                view
            ][qid]
            for view in views
        ]

        row[
            "lcr_top20_union_size"
        ] = int(
            len(
                set().union(
                    *lcr_sets
                )
            )
        )
        row[
            "all_views_top20_union_size"
        ] = int(
            len(
                set().union(
                    *all_sets
                )
            )
        )

        for metric in (
            "contain_r20",
            "le40_r20",
            "le80_r20",
        ):
            row[
                f"lcr_any_{metric}"
            ] = bool(
                any(
                    row[
                        f"{view}_{metric}"
                    ]
                    for view in (
                        "left_square",
                        "center_square",
                        "right_square",
                    )
                )
            )
            row[
                f"all_views_any_{metric}"
            ] = bool(
                any(
                    row[
                        f"{view}_{metric}"
                    ]
                    for view in views
                )
            )

        for metric in (
            "top1_contains",
            "top1_le40",
            "top1_le80",
        ):
            row[
                f"any_view_{metric}"
            ] = bool(
                any(
                    row[
                        f"{view}_{metric}"
                    ]
                    for view in views
                )
            )

        rows.append(row)

    per_query = pd.DataFrame(
        rows
    )

    rescue_summary = {}

    for view in (
        "left_square",
        "right_square",
        "resize_square",
    ):
        rescue_summary[view] = {}

        for metric in metrics:
            center = per_query[
                f"center_square_{metric}"
            ]
            alt = per_query[
                f"{view}_{metric}"
            ]

            categories = [
                category(
                    bool(c),
                    bool(a),
                )
                for c, a in zip(
                    center,
                    alt,
                )
            ]

            counts = pd.Series(
                categories
            ).value_counts()

            rescue_summary[
                view
            ][metric] = {
                "center_hits": int(
                    center.sum()
                ),
                "alt_hits": int(
                    alt.sum()
                ),
                "alt_only_rescues": int(
                    counts.get(
                        "alt_only_rescue",
                        0,
                    )
                ),
                "center_only_losses_if_replaced": int(
                    counts.get(
                        "center_only",
                        0,
                    )
                ),
                "both": int(
                    counts.get(
                        "both",
                        0,
                    )
                ),
                "neither": int(
                    counts.get(
                        "neither",
                        0,
                    )
                ),
            }

    ceiling = {
        "center_r20": {
            "contain": int(
                per_query[
                    "center_square_contain_r20"
                ].sum()
            ),
            "le40": int(
                per_query[
                    "center_square_le40_r20"
                ].sum()
            ),
            "le80": int(
                per_query[
                    "center_square_le80_r20"
                ].sum()
            ),
        },
        "lcr_any_r20": {
            "contain": int(
                per_query[
                    "lcr_any_contain_r20"
                ].sum()
            ),
            "le40": int(
                per_query[
                    "lcr_any_le40_r20"
                ].sum()
            ),
            "le80": int(
                per_query[
                    "lcr_any_le80_r20"
                ].sum()
            ),
        },
        "all_independent_views_any_r20": {
            "contain": int(
                per_query[
                    "all_views_any_contain_r20"
                ].sum()
            ),
            "le40": int(
                per_query[
                    "all_views_any_le40_r20"
                ].sum()
            ),
            "le80": int(
                per_query[
                    "all_views_any_le80_r20"
                ].sum()
            ),
        },
        "any_view_top1": {
            "contain": int(
                per_query[
                    "any_view_top1_contains"
                ].sum()
            ),
            "le40": int(
                per_query[
                    "any_view_top1_le40"
                ].sum()
            ),
            "le80": int(
                per_query[
                    "any_view_top1_le80"
                ].sum()
            ),
        },
        "candidate_union_size": {
            "lcr_mean": float(
                per_query[
                    "lcr_top20_union_size"
                ].mean()
            ),
            "lcr_median": float(
                per_query[
                    "lcr_top20_union_size"
                ].median()
            ),
            "all_views_mean": float(
                per_query[
                    "all_views_top20_union_size"
                ].mean()
            ),
            "all_views_median": float(
                per_query[
                    "all_views_top20_union_size"
                ].median()
            ),
        },
    }

    named = {}

    for qid in qv[
        "named_diagnostics"
    ]:
        qid = int(qid)
        row = per_query[
            per_query[
                "query_id"
            ]
            == qid
        ]
        named[
            f"q{qid}"
        ] = (
            row.iloc[
                0
            ].to_dict()
            if len(row) == 1
            else None
        )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    per_query_path = (
        output_root
        / "qv1_2_query_view_complementarity.csv"
    )
    per_query.to_csv(
        per_query_path,
        index=False,
    )

    report = {
        "stage": "QV1.2",
        "status": "PASS_QV1_QUERY_VIEW_COMPLEMENTARITY_AUDIT",
        "created_at_utc": now_utc(),
        "scientific_question": (
            "How much useful candidate availability do alternate query views "
            "add beyond the center-square 768 control?"
        ),
        "ranking_depth": depth,
        "rescue_summary": rescue_summary,
        "availability_ceiling": ceiling,
        "named_diagnostics": named,
        "interpretation_contract": {
            "new_selector_run": False,
            "new_descriptor_inference_run": False,
            "expanded_union_is_fixed_budget_result": False,
            "warning": (
                "Any-view and union metrics use a larger effective candidate "
                "budget. They measure candidate-generation complementarity, "
                "not deployable fixed-budget selection quality."
            ),
        },
        "input_hashes": {
            "qv1_1_report": {
                "path": str(
                    qv11_report_path
                ),
                "sha256": sha256_file(
                    qv11_report_path
                ),
            },
            "blind_rankings": {
                view: {
                    "path": str(
                        path
                    ),
                    "sha256": sha256_file(
                        path
                    ),
                }
                for view, path in ranking_paths.items()
            },
        },
        "outputs": {
            "per_query_csv": str(
                per_query_path
            ),
        },
        "next_decision": (
            "Retain alternate query views as candidate generators only if "
            "they provide meaningful unique rescues beyond center. Do not "
            "promote naive LCR RRF."
        ),
    }

    report_path = (
        output_root
        / "qv1_2_query_view_complementarity_report.json"
    )
    report_path.write_text(
        json.dumps(
            report,
            indent=2,
            default=str,
        )
    )

    print("=" * 112)
    print(
        "QV1.2 — QUERY-VIEW CANDIDATE COMPLEMENTARITY AUDIT"
    )
    print("=" * 112)

    print(
        "center R20:",
        ceiling[
            "center_r20"
        ],
    )
    print(
        "L/C/R any-view R20 ceiling:",
        ceiling[
            "lcr_any_r20"
        ],
    )
    print(
        "all independent views R20 ceiling:",
        ceiling[
            "all_independent_views_any_r20"
        ],
    )
    print(
        "any-view Top1 ceiling:",
        ceiling[
            "any_view_top1"
        ],
    )
    print(
        "candidate union size:",
        ceiling[
            "candidate_union_size"
        ],
    )

    print()
    print(
        "Unique rescues beyond center at Top20"
    )

    for view in (
        "left_square",
        "right_square",
        "resize_square",
    ):
        print()
        print(view)
        for metric in (
            "contain_r20",
            "le40_r20",
            "le80_r20",
        ):
            block = rescue_summary[
                view
            ][metric]
            print(
                f"  {metric:12s} "
                f"rescues={block['alt_only_rescues']:3d} "
                f"center-only={block['center_only_losses_if_replaced']:3d} "
                f"both={block['both']:3d} "
                f"neither={block['neither']:3d}"
            )

    for qid in qv[
        "named_diagnostics"
    ]:
        print()
        print(
            f"q{int(qid)}:"
        )
        print(
            json.dumps(
                named[
                    f"q{int(qid)}"
                ],
                indent=2,
                default=str,
            )
        )

    print()
    print(
        "STATUS: PASS_QV1_QUERY_VIEW_COMPLEMENTARITY_AUDIT"
    )
    print(
        "report:",
        report_path,
    )


if __name__ == "__main__":
    main()
