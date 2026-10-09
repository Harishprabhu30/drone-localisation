#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path.cwd().resolve()
DEFAULT_DEVELOPMENT_REPORT = Path(
    "outputs/research_runs/trajectory_adapter_v1/ta4/"
    "villoc_traj01_90deg_stable120m/ta4_frozen_qv_trajectory_report.json"
)
DEFAULT_BLIND_REPORT = Path(
    "outputs/research_runs/trajectory_adapter_v1/ta4/"
    "villoc_blind_recorded_flight_final_001/ta4_frozen_qv_trajectory_report.json"
)
DEFAULT_OUTPUT = Path(
    "outputs/research_runs/trajectory_adapter_v1/ta4/"
    "ta4_cross_trajectory_behavior_comparison.json"
)


def resolve(path_like: str | Path) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else ROOT / path


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_report(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def required_nested(report: dict[str, Any], keys: list[str]) -> Any:
    value: Any = report
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            raise RuntimeError(
                "Missing report key: " + ".".join(keys)
            )
        value = value[key]
    return value


def numeric_delta(
    development: dict[str, Any],
    blind: dict[str, Any],
    keys: list[str],
) -> dict[str, float]:
    dev = float(required_nested(development, keys))
    test = float(required_nested(blind, keys))
    return {
        "development": dev,
        "blind_stress": test,
        "delta_blind_minus_development": test - dev,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--development-report",
        type=Path,
        default=DEFAULT_DEVELOPMENT_REPORT,
    )
    parser.add_argument(
        "--blind-report",
        type=Path,
        default=DEFAULT_BLIND_REPORT,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    args = parser.parse_args()

    dev_path = resolve(args.development_report)
    blind_path = resolve(args.blind_report)
    output_path = resolve(args.output)

    development = load_report(dev_path)
    blind = load_report(blind_path)

    if development.get("status") != "PASS_TA4_FROZEN_QV_WITH_REFERENCE":
        raise RuntimeError(
            "Development TA4 report must pass with post-freeze reference."
        )

    if blind.get("status") != "PASS_TA4_FROZEN_QV_BLIND_STRESS":
        raise RuntimeError(
            "Blind TA4 report must pass as blind stress."
        )

    if development["trajectory"]["role"] != "development":
        raise RuntimeError(
            "Development report role mismatch."
        )

    if blind["trajectory"]["role"] != "blind_stress":
        raise RuntimeError(
            "Blind report role mismatch."
        )

    parity = development.get(
        "development_pool_parity"
    )
    if not parity or not parity.get("exact"):
        raise RuntimeError(
            "Development report does not have exact frozen QV1.4 pool parity."
        )

    summary_parity = development.get(
        "development_summary_parity"
    )
    if not summary_parity or not summary_parity.get("pass"):
        raise RuntimeError(
            "Development report does not have frozen QV1.4 summary parity."
        )

    if not development["postfreeze_evaluation"].get("performed"):
        raise RuntimeError(
            "Development report must include post-freeze accuracy evaluation."
        )

    if blind["postfreeze_evaluation"].get("performed"):
        raise RuntimeError(
            "Blind stress report must not include absolute reference evaluation."
        )

    if blind["scope_guarantees"].get("accuracy_claim_allowed"):
        raise RuntimeError(
            "Blind stress report incorrectly allows accuracy claims."
        )

    metric_paths = {
        "mean_center_unique_before_fill": [
            "blind_behavior",
            "construction",
            "mean_center_unique_before_fill",
        ],
        "mean_alternate_added": [
            "blind_behavior",
            "construction",
            "mean_alternate_added",
        ],
        "consecutive_pool_jaccard_median": [
            "blind_behavior",
            "consecutive_final_pool_jaccard",
            "median",
        ],
        "center_top1_map_jump_median_m": [
            "blind_behavior",
            "center_top1_map_jump_m",
            "median",
        ],
    }

    comparison = {
        name: numeric_delta(
            development,
            blind,
            path,
        )
        for name, path in metric_paths.items()
    }

    view_overlap = {}
    for view in (
        "left_square",
        "right_square",
        "resize_square",
    ):
        view_overlap[view] = numeric_delta(
            development,
            blind,
            [
                "blind_behavior",
                "center_vs_view_top20_jaccard",
                view,
                "median",
            ],
        )

    source_views = sorted(
        set(
            development[
                "blind_behavior"
            ][
                "mean_candidates_per_query_by_source"
            ].keys()
        )
        | set(
            blind[
                "blind_behavior"
            ][
                "mean_candidates_per_query_by_source"
            ].keys()
        )
    )

    source_comparison = {}
    for source in source_views:
        dev = float(
            development[
                "blind_behavior"
            ][
                "mean_candidates_per_query_by_source"
            ].get(
                source,
                0.0,
            )
        )
        test = float(
            blind[
                "blind_behavior"
            ][
                "mean_candidates_per_query_by_source"
            ].get(
                source,
                0.0,
            )
        )
        source_comparison[source] = {
            "development": dev,
            "blind_stress": test,
            "delta_blind_minus_development": test - dev,
        }

    report = {
        "stage": "TA4_COMPARE",
        "status": "PASS_TA4_CROSS_TRAJECTORY_BEHAVIOR_COMPARISON",
        "created_at_utc": now_utc(),
        "development": {
            "trajectory_id": development["trajectory"]["id"],
            "query_count": development["trajectory"]["query_count"],
            "exact_pool_parity": True,
            "summary_parity": True,
            "accuracy": development["postfreeze_evaluation"]["summary"],
        },
        "blind_stress": {
            "trajectory_id": blind["trajectory"]["id"],
            "query_count": blind["trajectory"]["query_count"],
            "reference_evaluation_performed": False,
            "accuracy_claim_allowed": False,
        },
        "blind_behavior_comparison": comparison,
        "center_vs_view_top20_jaccard_median": view_overlap,
        "mean_candidates_per_query_by_source": source_comparison,
        "interpretation_contract": {
            "absolute_accuracy_comparison_available": False,
            "blind_stress_accuracy_unknown": True,
            "allowed_claim": (
                "Compare retrieval/candidate-pool behavior across trajectories; "
                "do not infer geographic correctness for the blind flight."
            ),
        },
    }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    output_path.write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("=" * 112)
    print(
        "TA4 — CROSS-TRAJECTORY BLIND-BEHAVIOR COMPARISON"
    )
    print("=" * 112)
    print(
        "development:",
        report["development"]["trajectory_id"],
        f"({report['development']['query_count']} queries)",
    )
    print(
        "blind stress:",
        report["blind_stress"]["trajectory_id"],
        f"({report['blind_stress']['query_count']} queries)",
    )
    print()

    for name, block in comparison.items():
        print(
            f"{name:38s} "
            f"dev={block['development']:.4f} "
            f"blind={block['blind_stress']:.4f} "
            f"delta={block['delta_blind_minus_development']:+.4f}"
        )

    print()
    print("center-vs-view Top20 Jaccard medians:")
    for view, block in view_overlap.items():
        print(
            f"  {view:14s} "
            f"dev={block['development']:.4f} "
            f"blind={block['blind_stress']:.4f} "
            f"delta={block['delta_blind_minus_development']:+.4f}"
        )

    print()
    print(
        "STATUS: PASS_TA4_CROSS_TRAJECTORY_BEHAVIOR_COMPARISON"
    )
    print(
        "report:",
        output_path,
    )


if __name__ == "__main__":
    main()
