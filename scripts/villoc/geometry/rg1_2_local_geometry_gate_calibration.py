#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


ROOT = Path.cwd().resolve()
DEFAULT_CONFIG = Path(
    "configs/research/region_candidate_geometry_v1.yaml"
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


def numeric_summary(series: pd.Series) -> dict[str, Any]:
    values = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()

    if not len(values):
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p95": None,
            "max": None,
        }

    return {
        "count": int(len(values)),
        "mean": float(values.mean()),
        "median": float(values.median()),
        "p95": float(values.quantile(0.95)),
        "max": float(values.max()),
    }


def verify_rg1_1_artifacts(
    *,
    report_path: Path,
    blind_path: Path,
    evaluation_path: Path,
) -> dict[str, Any]:
    for path in (
        report_path,
        blind_path,
        evaluation_path,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    report = json.loads(
        report_path.read_text(
            encoding="utf-8"
        )
    )

    if report.get("status") != "PASS_RG1_1_ALL_CANDIDATE_GEOMETRY_AUDIT":
        raise RuntimeError(
            "RG1.1 has not passed."
        )

    expected_blind_sha = (
        report[
            "blind_freeze"
        ][
            "evidence_sha256"
        ]
    )

    measured_blind_sha = sha256_file(
        blind_path
    )

    if measured_blind_sha != expected_blind_sha:
        raise RuntimeError(
            "RG1.1 blind evidence hash mismatch."
        )

    if report[
        "scope_guarantees"
    ].get(
        "candidate_selection_enabled"
    ):
        raise RuntimeError(
            "RG1.1 report unexpectedly indicates candidate selection."
        )

    if report[
        "scope_guarantees"
    ].get(
        "reference_used_before_blind_freeze"
    ):
        raise RuntimeError(
            "RG1.1 report violates blind freeze boundary."
        )

    return report


def make_gate_id(
    *,
    family: str,
    inliers: int,
    ratio: float | None,
    inside: bool,
) -> str:
    parts = [
        family,
        f"i{int(inliers)}",
    ]

    if ratio is not None:
        parts.append(
            f"r{ratio:.2f}"
        )

    if inside:
        parts.append("inside")

    return "_".join(parts)


def gate_mask(
    frame: pd.DataFrame,
    *,
    family: str,
    inlier_threshold: int,
    ratio_threshold: float | None,
) -> pd.Series:
    accepted = (
        pd.to_numeric(
            frame[
                "inliers"
            ],
            errors="raise",
        )
        >= int(
            inlier_threshold
        )
    )

    if family in {
        "inliers_plus_ratio",
        "inliers_plus_ratio_inside",
    }:
        if ratio_threshold is None:
            raise RuntimeError(
                f"{family} requires ratio threshold."
            )

        accepted &= (
            pd.to_numeric(
                frame[
                    "inlier_ratio"
                ],
                errors="raise",
            )
            >= float(
                ratio_threshold
            )
        )

    if family in {
        "inliers_plus_inside",
        "inliers_plus_ratio_inside",
    }:
        accepted &= frame[
            "projected_inside_tile"
        ].astype(bool)

    return accepted.astype(bool)


def label_metrics(
    frame: pd.DataFrame,
    accepted: pd.Series,
    label: str,
) -> dict[str, Any]:
    truth = frame[
        label
    ].astype(bool)

    tp = int(
        (
            accepted
            & truth
        ).sum()
    )
    fp = int(
        (
            accepted
            & ~truth
        ).sum()
    )
    fn = int(
        (
            ~accepted
            & truth
        ).sum()
    )
    tn = int(
        (
            ~accepted
            & ~truth
        ).sum()
    )

    accepted_count = int(
        accepted.sum()
    )
    positive_count = int(
        truth.sum()
    )

    candidate_precision = (
        float(
            tp
            / accepted_count
        )
        if accepted_count
        else None
    )

    candidate_recall = (
        float(
            tp
            / positive_count
        )
        if positive_count
        else None
    )

    per_query = (
        pd.DataFrame(
            {
                "query_id": frame[
                    "query_id"
                ].astype(int),
                "accepted": accepted.astype(bool),
                "truth": truth,
            }
        )
        .groupby(
            "query_id",
            sort=True,
        )
        .apply(
            lambda group: pd.Series(
                {
                    "positive_available": bool(
                        group[
                            "truth"
                        ].any()
                    ),
                    "any_accepted": bool(
                        group[
                            "accepted"
                        ].any()
                    ),
                    "accepted_positive": bool(
                        (
                            group[
                                "accepted"
                            ]
                            & group[
                                "truth"
                            ]
                        ).any()
                    ),
                    "accepted_false": bool(
                        (
                            group[
                                "accepted"
                            ]
                            & ~group[
                                "truth"
                            ]
                        ).any()
                    ),
                    "accepted_count": int(
                        group[
                            "accepted"
                        ].sum()
                    ),
                }
            )
        )
        .reset_index()
    )

    positive_queries = int(
        per_query[
            "positive_available"
        ].sum()
    )
    retained_positive_queries = int(
        per_query[
            "accepted_positive"
        ].sum()
    )

    wrong_only = (
        per_query[
            "any_accepted"
        ]
        & ~per_query[
            "accepted_positive"
        ]
    )

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "candidate_precision": (
            candidate_precision
        ),
        "candidate_recall": (
            candidate_recall
        ),
        "positive_candidate_count": (
            positive_count
        ),
        "positive_query_count": (
            positive_queries
        ),
        "retained_positive_query_count": (
            retained_positive_queries
        ),
        "positive_query_retention": (
            float(
                retained_positive_queries
                / positive_queries
            )
            if positive_queries
            else None
        ),
        "queries_with_any_accepted": int(
            per_query[
                "any_accepted"
            ].sum()
        ),
        "wrong_only_accepted_query_count": int(
            wrong_only.sum()
        ),
        "accepted_candidates_per_query": (
            numeric_summary(
                per_query[
                    "accepted_count"
                ]
            )
        ),
    }


def projection_metrics(
    frame: pd.DataFrame,
    accepted: pd.Series,
    label: str,
) -> dict[str, Any]:
    truth = frame[
        label
    ].astype(bool)

    valid = (
        accepted
        & truth
        & frame[
            "valid_projection"
        ].astype(bool)
        & np.isfinite(
            pd.to_numeric(
                frame[
                    "projected_error_m"
                ],
                errors="coerce",
            )
        )
    )

    subset = frame[
        valid
    ].copy()

    if not len(
        subset
    ):
        return {
            "count": 0,
            "center_error_m": numeric_summary(
                pd.Series(
                    dtype=float
                )
            ),
            "projected_error_m": numeric_summary(
                pd.Series(
                    dtype=float
                )
            ),
            "projection_improves_count": 0,
            "projection_improves_rate": None,
        }

    improves = (
        pd.to_numeric(
            subset[
                "projected_error_m"
            ],
            errors="raise",
        )
        <
        pd.to_numeric(
            subset[
                "center_error_m"
            ],
            errors="raise",
        )
    )

    return {
        "count": int(
            len(
                subset
            )
        ),
        "center_error_m": (
            numeric_summary(
                subset[
                    "center_error_m"
                ]
            )
        ),
        "projected_error_m": (
            numeric_summary(
                subset[
                    "projected_error_m"
                ]
            )
        ),
        "projection_improves_count": int(
            improves.sum()
        ),
        "projection_improves_rate": float(
            improves.mean()
        ),
    }


def gate_row(
    *,
    frame: pd.DataFrame,
    family: str,
    inlier_threshold: int,
    ratio_threshold: float | None,
    labels: list[str],
) -> dict[str, Any]:
    accepted = gate_mask(
        frame,
        family=family,
        inlier_threshold=inlier_threshold,
        ratio_threshold=ratio_threshold,
    )

    accepted_count = int(
        accepted.sum()
    )
    total = int(
        len(
            frame
        )
    )

    row: dict[str, Any] = {
        "gate_id": make_gate_id(
            family=family,
            inliers=inlier_threshold,
            ratio=ratio_threshold,
            inside=(
                family
                in {
                    "inliers_plus_inside",
                    "inliers_plus_ratio_inside",
                }
            ),
        ),
        "family": family,
        "inlier_threshold": int(
            inlier_threshold
        ),
        "inlier_ratio_threshold": (
            float(
                ratio_threshold
            )
            if ratio_threshold
            is not None
            else None
        ),
        "require_projected_inside_tile": bool(
            family
            in {
                "inliers_plus_inside",
                "inliers_plus_ratio_inside",
            }
        ),
        "candidate_rows_total": total,
        "accepted_candidate_count": accepted_count,
        "accepted_candidate_fraction": (
            float(
                accepted_count
                / total
            )
            if total
            else None
        ),
    }

    per_query_counts = (
        pd.DataFrame(
            {
                "query_id": frame[
                    "query_id"
                ].astype(int),
                "accepted": accepted.astype(bool),
            }
        )
        .groupby(
            "query_id",
            sort=True,
        )[
            "accepted"
        ]
        .sum()
    )

    row[
        "accepted_candidates_per_query"
    ] = numeric_summary(
        per_query_counts
    )

    for label in labels:
        row[
            label
        ] = label_metrics(
            frame,
            accepted,
            label,
        )
        row[
            f"{label}_projection"
        ] = projection_metrics(
            frame,
            accepted,
            label,
        )

    return row


def pareto_frontier(
    rows: list[dict[str, Any]],
    *,
    label: str,
) -> list[dict[str, Any]]:
    usable = []

    for row in rows:
        block = row[
            label
        ]
        precision = block[
            "candidate_precision"
        ]
        retention = block[
            "positive_query_retention"
        ]

        if (
            precision is None
            or retention is None
        ):
            continue

        usable.append(
            (
                row,
                float(
                    precision
                ),
                float(
                    retention
                ),
                float(
                    row[
                        "accepted_candidate_fraction"
                    ]
                ),
            )
        )

    frontier = []

    for i, (
        row,
        precision,
        retention,
        fraction,
    ) in enumerate(
        usable
    ):
        dominated = False

        for j, (
            _,
            p2,
            r2,
            f2,
        ) in enumerate(
            usable
        ):
            if i == j:
                continue

            no_worse = (
                p2 >= precision
                and r2 >= retention
                and f2 <= fraction
            )

            strictly_better = (
                p2 > precision
                or r2 > retention
                or f2 < fraction
            )

            if (
                no_worse
                and strictly_better
            ):
                dominated = True
                break

        if not dominated:
            frontier.append(
                {
                    "gate_id": row[
                        "gate_id"
                    ],
                    "family": row[
                        "family"
                    ],
                    "inlier_threshold": row[
                        "inlier_threshold"
                    ],
                    "inlier_ratio_threshold": row[
                        "inlier_ratio_threshold"
                    ],
                    "require_projected_inside_tile": row[
                        "require_projected_inside_tile"
                    ],
                    "candidate_precision": precision,
                    "positive_query_retention": retention,
                    "accepted_candidate_fraction": fraction,
                    "accepted_candidates_per_query_median": row[
                        "accepted_candidates_per_query"
                    ][
                        "median"
                    ],
                    "wrong_only_accepted_query_count": row[
                        label
                    ][
                        "wrong_only_accepted_query_count"
                    ],
                    "projected_error_median": row[
                        f"{label}_projection"
                    ][
                        "projected_error_m"
                    ][
                        "median"
                    ],
                }
            )

    frontier.sort(
        key=lambda item: (
            -item[
                "positive_query_retention"
            ],
            -item[
                "candidate_precision"
            ],
            item[
                "accepted_candidate_fraction"
            ],
            item[
                "gate_id"
            ],
        )
    )

    return frontier


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "RG1.2: development-only calibration audit for local geometry "
            "evidence gates. No global candidate selection is promoted."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
    )
    args = parser.parse_args()

    config_path = resolve(
        args.config
    )
    cfg = yaml.safe_load(
        config_path.read_text(
            encoding="utf-8"
        )
    )
    rg = cfg[
        "rg1_2"
    ]

    report_path = resolve(
        rg[
            "rg1_1_report"
        ]
    )
    blind_path = resolve(
        rg[
            "rg1_1_blind_evidence"
        ]
    )
    evaluation_path = resolve(
        rg[
            "rg1_1_postfreeze_eval"
        ]
    )
    output_root = resolve(
        rg[
            "output_root"
        ]
    )

    rg1_report = verify_rg1_1_artifacts(
        report_path=report_path,
        blind_path=blind_path,
        evaluation_path=evaluation_path,
    )

    frame = pd.read_csv(
        evaluation_path
    ).copy()

    required = {
        "query_id",
        "pool_rank",
        "tile_id",
        "inliers",
        "inlier_ratio",
        "projected_inside_tile",
        "valid_projection",
        "projected_error_m",
        "center_error_m",
        "contains_query",
        "le80",
        "le40",
    }

    missing = sorted(
        required
        - set(
            frame.columns
        )
    )

    if missing:
        raise RuntimeError(
            f"RG1.1 evaluation is missing columns: {missing}"
        )

    labels = [
        str(
            value
        )
        for value in rg[
            "labels"
        ]
    ]
    families = [
        str(
            value
        )
        for value in rg[
            "families"
        ]
    ]
    inlier_thresholds = [
        int(
            value
        )
        for value in rg[
            "inlier_thresholds"
        ]
    ]
    ratio_thresholds = [
        float(
            value
        )
        for value in rg[
            "inlier_ratio_thresholds"
        ]
    ]

    rows: list[
        dict[
            str,
            Any,
        ]
    ] = []

    for family in families:
        for inlier_threshold in inlier_thresholds:
            if family in {
                "inliers_only",
                "inliers_plus_inside",
            }:
                rows.append(
                    gate_row(
                        frame=frame,
                        family=family,
                        inlier_threshold=inlier_threshold,
                        ratio_threshold=None,
                        labels=labels,
                    )
                )
            elif family in {
                "inliers_plus_ratio",
                "inliers_plus_ratio_inside",
            }:
                for ratio_threshold in ratio_thresholds:
                    rows.append(
                        gate_row(
                            frame=frame,
                            family=family,
                            inlier_threshold=inlier_threshold,
                            ratio_threshold=ratio_threshold,
                            labels=labels,
                        )
                    )
            else:
                raise RuntimeError(
                    f"Unsupported RG1.2 family: {family}"
                )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    flat_rows = []

    for row in rows:
        flat = {
            "gate_id": row[
                "gate_id"
            ],
            "family": row[
                "family"
            ],
            "inlier_threshold": row[
                "inlier_threshold"
            ],
            "inlier_ratio_threshold": row[
                "inlier_ratio_threshold"
            ],
            "require_projected_inside_tile": row[
                "require_projected_inside_tile"
            ],
            "accepted_candidate_count": row[
                "accepted_candidate_count"
            ],
            "accepted_candidate_fraction": row[
                "accepted_candidate_fraction"
            ],
            "accepted_candidates_per_query_mean": row[
                "accepted_candidates_per_query"
            ][
                "mean"
            ],
            "accepted_candidates_per_query_median": row[
                "accepted_candidates_per_query"
            ][
                "median"
            ],
            "accepted_candidates_per_query_p95": row[
                "accepted_candidates_per_query"
            ][
                "p95"
            ],
        }

        for label in labels:
            block = row[
                label
            ]
            projection = row[
                f"{label}_projection"
            ]

            flat.update(
                {
                    f"{label}_candidate_precision": block[
                        "candidate_precision"
                    ],
                    f"{label}_candidate_recall": block[
                        "candidate_recall"
                    ],
                    f"{label}_positive_query_retention": block[
                        "positive_query_retention"
                    ],
                    f"{label}_retained_positive_query_count": block[
                        "retained_positive_query_count"
                    ],
                    f"{label}_positive_query_count": block[
                        "positive_query_count"
                    ],
                    f"{label}_wrong_only_accepted_query_count": block[
                        "wrong_only_accepted_query_count"
                    ],
                    f"{label}_projection_count": projection[
                        "count"
                    ],
                    f"{label}_center_error_median": projection[
                        "center_error_m"
                    ][
                        "median"
                    ],
                    f"{label}_projected_error_median": projection[
                        "projected_error_m"
                    ][
                        "median"
                    ],
                    f"{label}_projection_improves_rate": projection[
                        "projection_improves_rate"
                    ],
                }
            )

        flat_rows.append(
            flat
        )

    table = pd.DataFrame(
        flat_rows
    )

    table_path = (
        output_root
        / "rg1_2_geometry_gate_calibration_table.csv"
    )
    table.to_csv(
        table_path,
        index=False,
    )

    frontiers = {
        label: pareto_frontier(
            rows,
            label=label,
        )
        for label in labels
    }

    report = {
        "stage": "RG1.2",
        "status": "PASS_RG1_2_LOCAL_GEOMETRY_GATE_CALIBRATION_AUDIT",
        "created_at_utc": now_utc(),
        "scientific_question": (
            "Can simple blind-safe geometry evidence gates reduce the local "
            "candidate set while retaining useful region candidates and "
            "preserving sub-tile projection quality?"
        ),
        "source": {
            "rg1_1_report": str(
                report_path
            ),
            "rg1_1_blind_evidence": str(
                blind_path
            ),
            "rg1_1_blind_evidence_sha256": sha256_file(
                blind_path
            ),
            "rg1_1_postfreeze_eval": str(
                evaluation_path
            ),
            "rg1_1_status": rg1_report[
                "status"
            ],
        },
        "calibration_contract": {
            "families": families,
            "inlier_thresholds": (
                inlier_thresholds
            ),
            "inlier_ratio_thresholds": (
                ratio_thresholds
            ),
            "labels": labels,
            "gate_features_allowed": rg[
                "gate_features_allowed"
            ],
            "gate_count": int(
                len(
                    rows
                )
            ),
        },
        "pareto_frontiers": frontiers,
        "promotion": rg[
            "promotion"
        ],
        "scope_guarantees": {
            "gate_uses_reference_features": False,
            "global_candidate_winner_selected": False,
            "top20_reranked": False,
            "learned_classifier_fitted": False,
            "blind_stress_used_for_tuning": False,
            "qv1_candidate_pool_modified": False,
            "bootstrap_enabled": False,
            "state_enabled": False,
        },
        "outputs": {
            "calibration_table": str(
                table_path
            ),
        },
        "next_decision": (
            "Inspect aggregate Pareto behavior. If a simple gate family shows "
            "a stable precision/positive-query-retention tradeoff and preserves "
            "projection quality, freeze one provisional development-derived "
            "gate for behavioral stress testing. Otherwise stop ORB gate design."
        ),
    }

    report_path_out = (
        output_root
        / "rg1_2_local_geometry_gate_calibration_report.json"
    )
    report_path_out.write_text(
        json.dumps(
            report,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    print("=" * 116)
    print(
        "RG1.2 — LOCAL GEOMETRY GATE CALIBRATION AUDIT"
    )
    print("=" * 116)
    print(
        "candidate pairs:",
        len(
            frame
        ),
    )
    print(
        "gate variants:",
        len(
            rows
        ),
    )
    print(
        "gate features:",
        rg[
            "gate_features_allowed"
        ],
    )
    print(
        "global selection:",
        False,
    )
    print(
        "promotion:",
        False,
    )

    for label in labels:
        frontier = frontiers[
            label
        ]
        print()
        print(
            f"{label} Pareto frontier "
            f"({len(frontier)} gates):"
        )

        for item in frontier[
            :12
        ]:
            print(
                "  "
                f"{item['gate_id']:40s} "
                f"precision={item['candidate_precision']:.4f} "
                f"queryRet={item['positive_query_retention']:.4f} "
                f"acceptFrac={item['accepted_candidate_fraction']:.4f} "
                f"medianCand/Q={item['accepted_candidates_per_query_median']} "
                f"wrongOnlyQ={item['wrong_only_accepted_query_count']} "
                f"projMed={item['projected_error_median']}"
            )

    print()
    print(
        "STATUS: PASS_RG1_2_LOCAL_GEOMETRY_GATE_CALIBRATION_AUDIT"
    )
    print(
        "table:",
        table_path,
    )
    print(
        "report:",
        report_path_out,
    )


if __name__ == "__main__":
    main()
