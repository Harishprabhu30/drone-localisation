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
    values = pd.to_numeric(series, errors="coerce").dropna()
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


def load_and_verify_rg1(
    *,
    rg1_1_report: Path,
    evaluation_path: Path,
    rg1_2_report: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    for path in (rg1_1_report, evaluation_path, rg1_2_report):
        if not path.exists():
            raise FileNotFoundError(path)

    r11 = json.loads(rg1_1_report.read_text(encoding="utf-8"))
    r12 = json.loads(rg1_2_report.read_text(encoding="utf-8"))

    if r11.get("status") != "PASS_RG1_1_ALL_CANDIDATE_GEOMETRY_AUDIT":
        raise RuntimeError("RG1.1 has not passed.")

    if r12.get("status") != "PASS_RG1_2_LOCAL_GEOMETRY_GATE_CALIBRATION_AUDIT":
        raise RuntimeError("RG1.2 has not passed.")

    if r12.get("promotion", {}).get("enabled"):
        raise RuntimeError(
            "RG1.2 unexpectedly promoted a geometry gate."
        )

    return r11, r12


def gate_mask(
    frame: pd.DataFrame,
    *,
    family: str,
    inlier_threshold: int,
    ratio_threshold: float | None,
) -> pd.Series:
    accepted = (
        pd.to_numeric(
            frame["inliers"],
            errors="raise",
        )
        >= int(inlier_threshold)
    )

    if family in {
        "inliers_plus_ratio",
        "inliers_plus_ratio_inside",
    }:
        if ratio_threshold is None:
            raise RuntimeError(f"{family} requires ratio threshold.")

        accepted &= (
            pd.to_numeric(
                frame["inlier_ratio"],
                errors="raise",
            )
            >= float(ratio_threshold)
        )

    if family in {
        "inliers_plus_inside",
        "inliers_plus_ratio_inside",
    }:
        accepted &= frame["projected_inside_tile"].astype(bool)

    return accepted.astype(bool)


def gate_id(
    family: str,
    inlier_threshold: int,
    ratio_threshold: float | None,
) -> str:
    parts = [family, f"i{int(inlier_threshold)}"]
    if ratio_threshold is not None:
        parts.append(f"r{float(ratio_threshold):.2f}")
    if "inside" in family:
        parts.append("inside")
    return "_".join(parts)


def evaluate_gate(
    *,
    frame: pd.DataFrame,
    accepted: pd.Series,
    labels: list[str],
) -> dict[str, Any]:
    accepted_count = int(accepted.sum())
    n = int(len(frame))

    out: dict[str, Any] = {
        "accepted_query_count": accepted_count,
        "accepted_query_fraction": (
            float(accepted_count / n)
            if n
            else None
        ),
    }

    accepted_frame = frame[accepted].copy()

    out["accepted_center_error_m"] = numeric_summary(
        accepted_frame["center_error_m"]
    )
    out["accepted_projected_error_m"] = numeric_summary(
        accepted_frame["projected_error_m"]
    )

    valid_projection = (
        accepted
        & frame["valid_projection"].astype(bool)
        & np.isfinite(
            pd.to_numeric(
                frame["projected_error_m"],
                errors="coerce",
            )
        )
    )
    valid_frame = frame[valid_projection].copy()

    out["valid_projection_count"] = int(valid_projection.sum())
    out["valid_projection_fraction_of_accepted"] = (
        float(valid_projection.sum() / accepted_count)
        if accepted_count
        else None
    )

    if len(valid_frame):
        improves = (
            pd.to_numeric(
                valid_frame["projected_error_m"],
                errors="raise",
            )
            <
            pd.to_numeric(
                valid_frame["center_error_m"],
                errors="raise",
            )
        )
        out["projection_improves_count"] = int(improves.sum())
        out["projection_improves_rate"] = float(improves.mean())
    else:
        out["projection_improves_count"] = 0
        out["projection_improves_rate"] = None

    for label in labels:
        truth = frame[label].astype(bool)

        tp = int((accepted & truth).sum())
        fp = int((accepted & ~truth).sum())
        positive_count = int(truth.sum())

        precision = (
            float(tp / accepted_count)
            if accepted_count
            else None
        )
        recall = (
            float(tp / positive_count)
            if positive_count
            else None
        )

        accepted_true = frame[
            accepted & truth
        ].copy()

        out[label] = {
            "accepted_true_count": tp,
            "accepted_false_count": fp,
            "precision": precision,
            "recall": recall,
            "positive_count_total": positive_count,
            "accepted_true_center_error_m": numeric_summary(
                accepted_true["center_error_m"]
            ),
            "accepted_true_projected_error_m": numeric_summary(
                accepted_true["projected_error_m"]
            ),
        }

    return out


def pareto_frontier(
    rows: list[dict[str, Any]],
    *,
    label: str,
) -> list[dict[str, Any]]:
    usable = []

    for row in rows:
        block = row["metrics"][label]
        precision = block["precision"]
        recall = block["recall"]
        coverage = row["metrics"]["accepted_query_fraction"]

        if (
            precision is None
            or recall is None
            or coverage is None
        ):
            continue

        usable.append(
            (
                row,
                float(precision),
                float(recall),
                float(coverage),
            )
        )

    frontier = []

    for i, (
        row,
        precision,
        recall,
        coverage,
    ) in enumerate(usable):
        dominated = False

        for j, (
            _,
            p2,
            r2,
            c2,
        ) in enumerate(usable):
            if i == j:
                continue

            no_worse = (
                p2 >= precision
                and r2 >= recall
                and c2 <= coverage
            )
            strictly_better = (
                p2 > precision
                or r2 > recall
                or c2 < coverage
            )

            if no_worse and strictly_better:
                dominated = True
                break

        if not dominated:
            metrics = row["metrics"]
            frontier.append(
                {
                    "gate_id": row["gate_id"],
                    "family": row["family"],
                    "inlier_threshold": row["inlier_threshold"],
                    "inlier_ratio_threshold": row["inlier_ratio_threshold"],
                    "accepted_query_fraction": coverage,
                    "precision": precision,
                    "recall": recall,
                    "accepted_true_count": metrics[label]["accepted_true_count"],
                    "accepted_false_count": metrics[label]["accepted_false_count"],
                    "projected_error_median": (
                        metrics[label][
                            "accepted_true_projected_error_m"
                        ]["median"]
                    ),
                    "center_error_median": (
                        metrics[label][
                            "accepted_true_center_error_m"
                        ]["median"]
                    ),
                    "projection_improves_rate": metrics[
                        "projection_improves_rate"
                    ],
                }
            )

    frontier.sort(
        key=lambda item: (
            -item["recall"],
            -item["precision"],
            item["accepted_query_fraction"],
            item["gate_id"],
        )
    )

    return frontier


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "RG1.3: test whether geometry can selectively trust/refine the "
            "frozen retrieval anchor without switching candidates."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
    )
    args = parser.parse_args()

    config_path = resolve(args.config)
    cfg = yaml.safe_load(
        config_path.read_text(encoding="utf-8")
    )
    rg = cfg["rg1_3"]

    rg1_1_report = resolve(rg["rg1_1_report"])
    evaluation_path = resolve(rg["rg1_1_postfreeze_eval"])
    rg1_2_report = resolve(rg["rg1_2_report"])
    output_root = resolve(rg["output_root"])

    r11, r12 = load_and_verify_rg1(
        rg1_1_report=rg1_1_report,
        evaluation_path=evaluation_path,
        rg1_2_report=rg1_2_report,
    )

    frame = pd.read_csv(evaluation_path).copy()

    anchor_cfg = rg["anchor"]
    anchor_rank = int(anchor_cfg["pool_rank"])
    expected_source = str(anchor_cfg["expected_source_view"])
    expected_source_rank = int(anchor_cfg["expected_source_rank"])

    anchor = frame[
        pd.to_numeric(
            frame["pool_rank"],
            errors="raise",
        ).astype(int)
        == anchor_rank
    ].copy()

    if anchor["query_id"].nunique() != 403 or len(anchor) != 403:
        raise RuntimeError(
            f"Expected exactly 403 retrieval-anchor rows, got {len(anchor)}"
        )

    if not (
        (
            anchor["source_view"].astype(str)
            == expected_source
        )
        &
        (
            pd.to_numeric(
                anchor["source_rank"],
                errors="raise",
            ).astype(int)
            == expected_source_rank
        )
    ).all():
        raise RuntimeError(
            "RG1.3 anchor is not the frozen center-square retrieval Top1."
        )

    labels = [str(x) for x in rg["labels"]]
    families = [str(x) for x in rg["families"]]
    inlier_thresholds = [
        int(x)
        for x in rg["inlier_thresholds"]
    ]
    ratio_thresholds = [
        float(x)
        for x in rg["inlier_ratio_thresholds"]
    ]

    rows: list[dict[str, Any]] = []

    for family in families:
        for inlier_threshold in inlier_thresholds:
            if family in {
                "inliers_only",
                "inliers_plus_inside",
            }:
                thresholds = [None]
            else:
                thresholds = ratio_thresholds

            for ratio_threshold in thresholds:
                accepted = gate_mask(
                    anchor,
                    family=family,
                    inlier_threshold=inlier_threshold,
                    ratio_threshold=ratio_threshold,
                )

                rows.append(
                    {
                        "gate_id": gate_id(
                            family,
                            inlier_threshold,
                            ratio_threshold,
                        ),
                        "family": family,
                        "inlier_threshold": inlier_threshold,
                        "inlier_ratio_threshold": ratio_threshold,
                        "require_projected_inside_tile": (
                            family in {
                                "inliers_plus_inside",
                                "inliers_plus_ratio_inside",
                            }
                        ),
                        "metrics": evaluate_gate(
                            frame=anchor,
                            accepted=accepted,
                            labels=labels,
                        ),
                    }
                )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    flat_rows = []
    for row in rows:
        metrics = row["metrics"]
        flat = {
            "gate_id": row["gate_id"],
            "family": row["family"],
            "inlier_threshold": row["inlier_threshold"],
            "inlier_ratio_threshold": row["inlier_ratio_threshold"],
            "require_projected_inside_tile": row[
                "require_projected_inside_tile"
            ],
            "accepted_query_count": metrics["accepted_query_count"],
            "accepted_query_fraction": metrics["accepted_query_fraction"],
            "valid_projection_count": metrics["valid_projection_count"],
            "valid_projection_fraction_of_accepted": metrics[
                "valid_projection_fraction_of_accepted"
            ],
            "projection_improves_rate": metrics[
                "projection_improves_rate"
            ],
            "accepted_center_error_median": metrics[
                "accepted_center_error_m"
            ]["median"],
            "accepted_projected_error_median": metrics[
                "accepted_projected_error_m"
            ]["median"],
        }

        for label in labels:
            block = metrics[label]
            flat.update(
                {
                    f"{label}_precision": block["precision"],
                    f"{label}_recall": block["recall"],
                    f"{label}_accepted_true_count": block[
                        "accepted_true_count"
                    ],
                    f"{label}_accepted_false_count": block[
                        "accepted_false_count"
                    ],
                    f"{label}_true_center_error_median": block[
                        "accepted_true_center_error_m"
                    ]["median"],
                    f"{label}_true_projected_error_median": block[
                        "accepted_true_projected_error_m"
                    ]["median"],
                }
            )

        flat_rows.append(flat)

    table = pd.DataFrame(flat_rows)
    table_path = (
        output_root
        / "rg1_3_anchor_trust_refinement_table.csv"
    )
    table.to_csv(table_path, index=False)

    frontiers = {
        label: pareto_frontier(
            rows,
            label=label,
        )
        for label in labels
    }

    baseline = {
        label: {
            "positive_count": int(
                anchor[label].astype(bool).sum()
            ),
            "precision_if_always_trusted": float(
                anchor[label].astype(bool).mean()
            ),
        }
        for label in labels
    }

    report = {
        "stage": "RG1.3",
        "status": "PASS_RG1_3_RETRIEVAL_ANCHOR_TRUST_REFINEMENT_AUDIT",
        "created_at_utc": now_utc(),
        "scientific_question": (
            "Can blind-safe ORB geometry selectively decide when the frozen "
            "retrieval Top1 anchor is trustworthy for continuous sub-tile "
            "projection, without switching to another candidate?"
        ),
        "source": {
            "rg1_1_report": str(rg1_1_report),
            "rg1_1_status": r11["status"],
            "rg1_2_report": str(rg1_2_report),
            "rg1_2_status": r12["status"],
            "evaluation_path": str(evaluation_path),
            "evaluation_sha256": sha256_file(evaluation_path),
        },
        "anchor_contract": {
            "pool_rank": anchor_rank,
            "source_view": expected_source,
            "source_rank": expected_source_rank,
            "query_count": int(len(anchor)),
            "switching_allowed": False,
        },
        "baseline_anchor_accuracy": baseline,
        "gate_count": int(len(rows)),
        "pareto_frontiers": frontiers,
        "promotion": rg["promotion"],
        "scope_guarantees": {
            "alternate_candidate_considered": False,
            "candidate_switching_enabled": False,
            "top20_reranked": False,
            "learned_model_fitted": False,
            "blind_stress_used_for_tuning": False,
            "bootstrap_enabled": False,
            "state_enabled": False,
            "temporal_fusion_enabled": False,
        },
        "outputs": {
            "table": str(table_path),
        },
        "next_decision": (
            "If geometry can reach materially higher anchor precision at useful "
            "coverage while preserving low projected error, freeze one "
            "provisional trust/refinement gate for reference-free behavioral "
            "stress testing. Otherwise keep geometry as post-acceptance "
            "refinement only and do not use it for confidence."
        ),
    }

    report_path = (
        output_root
        / "rg1_3_retrieval_anchor_trust_refinement_report.json"
    )
    report_path.write_text(
        json.dumps(
            report,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    print("=" * 116)
    print(
        "RG1.3 — RETRIEVAL-ANCHOR GEOMETRY TRUST / REFINEMENT AUDIT"
    )
    print("=" * 116)
    print("anchor queries:", len(anchor))
    print("candidate switching:", False)
    print("promotion:", False)
    print()

    for label in labels:
        b = baseline[label]
        print(
            f"{label} baseline always-trust precision="
            f"{b['precision_if_always_trusted']:.4f} "
            f"({b['positive_count']}/403)"
        )

        frontier = frontiers[label]
        print(
            f"{label} selective-trust Pareto frontier "
            f"({len(frontier)} gates):"
        )

        for item in frontier[:15]:
            print(
                "  "
                f"{item['gate_id']:40s} "
                f"precision={item['precision']:.4f} "
                f"recall={item['recall']:.4f} "
                f"coverage={item['accepted_query_fraction']:.4f} "
                f"TP={item['accepted_true_count']} "
                f"FP={item['accepted_false_count']} "
                f"centerMed={item['center_error_median']} "
                f"projMed={item['projected_error_median']} "
                f"projImprove={item['projection_improves_rate']}"
            )
        print()

    print(
        "STATUS: PASS_RG1_3_RETRIEVAL_ANCHOR_TRUST_REFINEMENT_AUDIT"
    )
    print("table:", table_path)
    print("report:", report_path)


if __name__ == "__main__":
    main()
