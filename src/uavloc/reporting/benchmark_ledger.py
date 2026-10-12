"""Canonical method benchmark ledger utilities.

The ledger is reporting-only. It reads completed result artifacts and normalizes
their metrics; it must never feed training, thresholds, ranking, or selection.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class MetricRecord:
    method_id: str
    display_name: str
    family: str
    frontend: str
    retrieval: str
    selector: str
    dataset_id: str
    split: str
    candidate_pool: str
    map_variant: str
    source_id: str
    source_stage: str
    source_status: str
    source_report: str
    source_sha256: str
    target: str
    metric: str
    value: float | None
    unit: str
    numerator: float | None
    denominator: float | None
    higher_is_better: bool | None
    source_json_path: str
    notes: str


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    """Strict JSON reader with compatibility for the historical GD1.3-A literal-newline defect."""
    raw = path.read_text(encoding="utf-8")
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        stripped = raw.rstrip()
        if stripped.endswith("\\n"):
            candidate = stripped[:-2]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass
        raise RuntimeError(f"Invalid JSON report: {path}") from exc


def resolve_json_path(root: Any, path: str) -> Any:
    """Resolve a simple dotted JSON path, with integer list indices supported."""
    if not path:
        return root
    current = root
    for token in path.split("."):
        if isinstance(current, dict):
            if token not in current:
                raise KeyError(f"Missing key {token!r} while resolving {path!r}")
            current = current[token]
        elif isinstance(current, list):
            try:
                index = int(token)
            except ValueError as exc:
                raise KeyError(
                    f"List path token {token!r} is not an integer in {path!r}"
                ) from exc
            current = current[index]
        else:
            raise KeyError(
                f"Cannot descend through {type(current).__name__} at {token!r} "
                f"while resolving {path!r}"
            )
    return current


def render_path(template: str, *, method: str, target: str) -> str:
    return template.format(method=method, target=target)


def to_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def infer_unit(metric: str) -> str:
    if metric.endswith("_rate") or metric.endswith("_auc") or metric == "pair_auc_oof":
        return "ratio"
    if metric.endswith("_ms") or "_ms_" in metric:
        return "ms"
    if metric.endswith("_s") or "_s_" in metric:
        return "s"
    if metric.endswith("_m") or "_m_" in metric:
        return "m"
    if metric.endswith("_px") or "_px_" in metric:
        return "px"
    if metric.endswith("_bytes"):
        return "bytes"
    if metric.endswith("_count") or metric.endswith("_hits") or metric in {
        "feature_count",
        "queries",
        "pairs",
        "unique_tiles",
        "homography_ok",
        "finite_projection",
        "inside_projection",
        "queries_any_homography",
        "queries_all20_homographies",
        "queries_no_homography",
        "median_matches",
        "median_inliers",
    }:
        return "count"
    return "scalar"


def metric_direction(metric: str) -> bool | None:
    lower_tokens = (
        "error",
        "rmse",
        "latency",
        "runtime",
        "time",
        "condition_number",
        "anisotropy",
    )
    higher_tokens = (
        "top1",
        "top3",
        "top5",
        "top10",
        "top20",
        "auc",
        "recall",
        "precision",
        "homography_ok",
        "finite_projection",
        "inside_projection",
        "retention",
    )
    name = metric.lower()
    if any(token in name for token in lower_tokens):
        return False
    if any(token in name for token in higher_tokens):
        return True
    return None


def normalize_hit_metric(
    metric: str,
    value: float | None,
    query_count: int | None,
) -> list[tuple[str, float | None, str, float | None, float | None]]:
    rows = [(metric, value, infer_unit(metric), None, None)]
    if (
        value is not None
        and query_count
        and metric.startswith("top")
        and metric.endswith("_hits")
    ):
        rate_name = metric[:-5] + "_rate"
        rows.append(
            (
                rate_name,
                float(value) / float(query_count),
                "ratio",
                float(value),
                float(query_count),
            )
        )
    return rows


def normalize_coverage_metric(
    metric: str,
    value: float | None,
    denominator: int | None,
) -> list[tuple[str, float | None, str, float | None, float | None]]:
    rows = [(metric, value, infer_unit(metric), None, None)]
    if (
        value is not None
        and denominator
        and metric in {"homography_ok", "finite_projection", "inside_projection"}
    ):
        rows.append(
            (
                metric + "_rate",
                float(value) / float(denominator),
                "ratio",
                float(value),
                float(denominator),
            )
        )
    return rows


def records_to_frame(records: list[MetricRecord]) -> pd.DataFrame:
    columns = list(MetricRecord.__dataclass_fields__)
    if not records:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame([asdict(record) for record in records], columns=columns)


def format_ratio(value: float | None) -> str:
    if value is None or not math.isfinite(float(value)):
        return "—"
    return f"{100.0 * float(value):.1f}%"


def format_number(value: float | None, unit: str) -> str:
    if value is None or not math.isfinite(float(value)):
        return "—"
    value = float(value)
    if unit == "ratio":
        return format_ratio(value)
    if unit == "count":
        return f"{int(round(value))}"
    if unit in {"m", "px", "ms", "s"}:
        return f"{value:.3f}"
    return f"{value:.4f}"


def report_markdown(
    long_frame: pd.DataFrame,
    methods_frame: pd.DataFrame,
    *,
    title: str,
) -> str:
    lines = [
        f"# {title}",
        "",
        "Generated from completed benchmark reports. Reporting-only; do not use "
        "this generated file as a training or threshold-selection input.",
        "",
    ]

    targets = [
        target
        for target in ("contains_query", "center_le40", "center_le80")
        if target in set(long_frame["target"].astype(str))
    ]
    for target in targets:
        lines.extend([f"## {target}", ""])
        subset = long_frame[
            (long_frame["target"] == target)
            & long_frame["metric"].isin(
                [
                    "pair_auc_oof",
                    "top1_rate",
                    "top3_rate",
                    "top5_rate",
                    "top10_rate",
                    "top20_rate",
                ]
            )
        ].copy()
        if subset.empty:
            lines.extend(["No metrics available.", ""])
            continue
        pivot = subset.pivot_table(
            index=["method_id", "display_name"],
            columns="metric",
            values="value",
            aggfunc="first",
        ).reset_index()
        order = [
            "method_id",
            "display_name",
            "pair_auc_oof",
            "top1_rate",
            "top3_rate",
            "top5_rate",
            "top10_rate",
            "top20_rate",
        ]
        for column in order:
            if column not in pivot.columns:
                pivot[column] = None
        pivot = pivot[order]

        lines.append("| Method | AUC | Top1 | Top3 | Top5 | Top10 | Top20 |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for row in pivot.itertuples(index=False):
            lines.append(
                "| "
                + f"{row.display_name} | "
                + f"{format_ratio(row.pair_auc_oof)} | "
                + f"{format_ratio(row.top1_rate)} | "
                + f"{format_ratio(row.top3_rate)} | "
                + f"{format_ratio(row.top5_rate)} | "
                + f"{format_ratio(row.top10_rate)} | "
                + f"{format_ratio(row.top20_rate)} |"
            )
        lines.append("")

    overall = long_frame[long_frame["target"] == "__overall__"].copy()
    if not overall.empty:
        metrics = [
            "homography_ok_rate",
            "finite_projection_rate",
            "inside_projection_rate",
            "median_matches",
            "median_inliers",
        ]
        available = [metric for metric in metrics if metric in set(overall["metric"])]
        if available:
            lines.extend(["## Frontend coverage diagnostics", ""])
            lines.append(
                "| Method | Homography | Finite projection | Inside projection | "
                "Median matches | Median inliers |"
            )
            lines.append("|---|---:|---:|---:|---:|---:|")
            method_lookup = (
                methods_frame.set_index("method_id")["display_name"].to_dict()
                if not methods_frame.empty else {}
            )
            for method_id in sorted(overall["method_id"].unique()):
                block = overall[overall["method_id"] == method_id]
                values = {
                    row.metric: (row.value, row.unit)
                    for row in block.itertuples(index=False)
                }
                def cell(name: str) -> str:
                    value, unit = values.get(name, (None, "scalar"))
                    return format_number(value, unit)
                lines.append(
                    f"| {method_lookup.get(method_id, method_id)} | "
                    f"{cell('homography_ok_rate')} | "
                    f"{cell('finite_projection_rate')} | "
                    f"{cell('inside_projection_rate')} | "
                    f"{cell('median_matches')} | "
                    f"{cell('median_inliers')} |"
                )
            lines.append("")

    lines.extend(
        [
            "## Provenance",
            "",
            "| Method | Frontend | Selector/formulation | Source stage | Source SHA256 |",
            "|---|---|---|---|---|",
        ]
    )
    for row in methods_frame.sort_values("method_id").itertuples(index=False):
        sha = str(row.source_sha256)
        short_sha = sha[:12] if sha else "—"
        lines.append(
            f"| {row.display_name} | {row.frontend} | {row.selector} | "
            f"{row.source_stage} | {short_sha} |"
        )
    lines.append("")
    return "\n".join(lines)
