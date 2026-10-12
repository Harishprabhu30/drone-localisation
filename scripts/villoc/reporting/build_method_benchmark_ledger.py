#!/usr/bin/env python3
"""Build the canonical VILLOC method/metric benchmark ledger.

Reporting-only utility:
- reads completed/frozen result reports;
- verifies stage status and pinned artifact hashes when configured;
- normalizes heterogeneous metrics into one long schema;
- writes a wide comparison CSV and report-ready Markdown tables.

It must never feed ranking, tuning, model training, or deployment decisions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import pandas as pd
import yaml

from uavloc.reporting.benchmark_ledger import (
    MetricRecord,
    infer_unit,
    load_json,
    metric_direction,
    normalize_coverage_metric,
    normalize_hit_metric,
    records_to_frame,
    render_path,
    report_markdown,
    resolve_json_path,
    sha256_file,
    to_float,
)

ROOT = REPO_ROOT
DEFAULT_CONFIG = ROOT / "configs/reporting/villoc_method_benchmark_registry.yaml"


def resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def canonical_json(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def canonical_csv(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(
        index=False,
        lineterminator="\n",
        float_format="%.9f",
    ).encode("utf-8")


def write_outputs(files: dict[Path, bytes]) -> None:
    for path, data in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)


def validate_config(cfg: dict[str, Any]) -> None:
    if int(cfg.get("schema_version", 0)) != 1:
        raise RuntimeError("Expected benchmark registry schema_version: 1")

    dataset = cfg.get("dataset", {})
    for key in ("id", "split", "query_count", "candidate_pool", "map_variant"):
        if key not in dataset:
            raise RuntimeError(f"dataset.{key} is required")

    methods = cfg.get("methods", [])
    method_ids = [str(method["method_id"]) for method in methods]
    if len(method_ids) != len(set(method_ids)):
        raise RuntimeError("method_id values must be unique")

    sources = cfg.get("sources", {})
    for method in methods:
        source_id = str(method["source"])
        if source_id not in sources:
            raise RuntimeError(
                f"Method {method['method_id']} refers to unknown source {source_id}"
            )


def load_sources(
    cfg: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    loaded: dict[str, dict[str, Any]] = {}
    source_rows: list[dict[str, Any]] = []

    for source_id, spec in cfg["sources"].items():
        report_path = resolve(spec["report"])
        required = bool(spec.get("required", True))
        if not report_path.is_file():
            if required:
                raise FileNotFoundError(
                    f"Required benchmark source missing: {report_path}"
                )
            source_rows.append(
                {
                    "source_id": source_id,
                    "loaded": False,
                    "reason": "missing_optional_source",
                    "report": str(report_path.relative_to(ROOT)),
                }
            )
            continue

        report = load_json(report_path)
        actual_status = str(report.get("status", ""))
        expected_status = str(spec["expected_status"])
        if actual_status != expected_status:
            raise RuntimeError(
                f"{source_id} status {actual_status!r} != {expected_status!r}"
            )

        expected_artifact = spec.get("expected_artifact")
        if expected_artifact:
            json_path = str(expected_artifact["json_path"])
            actual_hash = str(resolve_json_path(report, json_path))
            expected_hash = str(expected_artifact["sha256"])
            if actual_hash != expected_hash:
                raise RuntimeError(
                    f"{source_id} pinned artifact hash mismatch: "
                    f"{actual_hash} != {expected_hash}"
                )

        report_sha = sha256_file(report_path)
        loaded[source_id] = {
            "spec": spec,
            "report": report,
            "path": report_path,
            "sha256": report_sha,
            "status": actual_status,
        }
        source_rows.append(
            {
                "source_id": source_id,
                "loaded": True,
                "reason": "",
                "source_stage": str(spec["source_stage"]),
                "source_status": actual_status,
                "report": str(report_path.relative_to(ROOT)),
                "report_sha256": report_sha,
            }
        )
    return loaded, source_rows


def record(
    *,
    method: dict[str, Any],
    dataset: dict[str, Any],
    source_id: str,
    source: dict[str, Any],
    target: str,
    metric: str,
    value: float | None,
    unit: str,
    numerator: float | None,
    denominator: float | None,
    source_json_path: str,
) -> MetricRecord:
    return MetricRecord(
        method_id=str(method["method_id"]),
        display_name=str(method["display_name"]),
        family=str(method.get("family", "")),
        frontend=str(method.get("frontend", "")),
        retrieval=str(method.get("retrieval", "")),
        selector=str(method.get("selector", "")),
        dataset_id=str(dataset["id"]),
        split=str(dataset["split"]),
        candidate_pool=str(dataset["candidate_pool"]),
        map_variant=str(dataset["map_variant"]),
        source_id=source_id,
        source_stage=str(source["spec"]["source_stage"]),
        source_status=str(source["status"]),
        source_report=str(source["path"].relative_to(ROOT)),
        source_sha256=str(source["sha256"]),
        target=target,
        metric=metric,
        value=value,
        unit=unit,
        numerator=numerator,
        denominator=denominator,
        higher_is_better=metric_direction(metric),
        source_json_path=source_json_path,
        notes=str(method.get("notes", "")),
    )


def collect_method_records(
    cfg: dict[str, Any],
    loaded_sources: dict[str, dict[str, Any]],
) -> tuple[list[MetricRecord], list[dict[str, Any]]]:
    dataset = cfg["dataset"]
    query_count = int(dataset["query_count"])
    default_targets = [str(value) for value in cfg.get("targets", [])]

    records: list[MetricRecord] = []
    methods_meta: list[dict[str, Any]] = []

    for method in cfg["methods"]:
        source_id = str(method["source"])
        if source_id not in loaded_sources:
            continue
        source = loaded_sources[source_id]
        spec = source["spec"]
        report = source["report"]
        method_key = str(method.get("method_key", ""))

        methods_meta.append(
            {
                "method_id": str(method["method_id"]),
                "display_name": str(method["display_name"]),
                "family": str(method.get("family", "")),
                "frontend": str(method.get("frontend", "")),
                "retrieval": str(method.get("retrieval", "")),
                "selector": str(method.get("selector", "")),
                "dataset_id": str(dataset["id"]),
                "split": str(dataset["split"]),
                "candidate_pool": str(dataset["candidate_pool"]),
                "map_variant": str(dataset["map_variant"]),
                "source_id": source_id,
                "source_stage": str(spec["source_stage"]),
                "source_status": str(source["status"]),
                "source_report": str(source["path"].relative_to(ROOT)),
                "source_sha256": str(source["sha256"]),
                "notes": str(method.get("notes", "")),
            }
        )

        targets = [
            str(value)
            for value in method.get("targets", default_targets)
        ]
        metric_paths = spec.get("method_metric_paths", {})
        for target in targets:
            for metric, template in metric_paths.items():
                json_path = render_path(
                    str(template),
                    method=method_key,
                    target=target,
                )
                raw_value = resolve_json_path(report, json_path)
                value = to_float(raw_value)

                for (
                    normalized_metric,
                    normalized_value,
                    unit,
                    numerator,
                    denominator,
                ) in normalize_hit_metric(metric, value, query_count):
                    records.append(
                        record(
                            method=method,
                            dataset=dataset,
                            source_id=source_id,
                            source=source,
                            target=target,
                            metric=normalized_metric,
                            value=normalized_value,
                            unit=unit,
                            numerator=numerator,
                            denominator=denominator,
                            source_json_path=json_path,
                        )
                    )

            optional_metric_paths = spec.get("optional_method_metric_paths", {})
            for metric, template in optional_metric_paths.items():
                json_path = render_path(
                    str(template),
                    method=method_key,
                    target=target,
                )
                try:
                    raw_value = resolve_json_path(report, json_path)
                except KeyError:
                    continue
                value = to_float(raw_value)
                for (
                    normalized_metric,
                    normalized_value,
                    unit,
                    numerator,
                    denominator,
                ) in normalize_hit_metric(metric, value, query_count):
                    records.append(
                        record(
                            method=method,
                            dataset=dataset,
                            source_id=source_id,
                            source=source,
                            target=target,
                            metric=normalized_metric,
                            value=normalized_value,
                            unit=unit,
                            numerator=numerator,
                            denominator=denominator,
                            source_json_path=json_path,
                        )
                    )

        global_paths = spec.get("global_metric_paths", {})
        pair_count = (
            int(spec["pair_count"])
            if spec.get("pair_count") is not None
            else None
        )
        for metric, json_path in global_paths.items():
            value = to_float(resolve_json_path(report, str(json_path)))
            for (
                normalized_metric,
                normalized_value,
                unit,
                numerator,
                denominator,
            ) in normalize_coverage_metric(metric, value, pair_count):
                records.append(
                    record(
                        method=method,
                        dataset=dataset,
                        source_id=source_id,
                        source=source,
                        target="__overall__",
                        metric=normalized_metric,
                        value=normalized_value,
                        unit=unit,
                        numerator=numerator,
                        denominator=denominator,
                        source_json_path=str(json_path),
                    )
                )

    return records, methods_meta


def wide_table(
    long_frame: pd.DataFrame,
    methods_frame: pd.DataFrame,
) -> pd.DataFrame:
    if long_frame.empty:
        return methods_frame.copy()

    values = long_frame.copy()
    values["column"] = values["target"].astype(str) + "." + values["metric"].astype(str)
    pivot = values.pivot_table(
        index="method_id",
        columns="column",
        values="value",
        aggfunc="first",
    ).reset_index()
    pivot.columns.name = None

    metadata_columns = [
        "method_id",
        "display_name",
        "family",
        "frontend",
        "retrieval",
        "selector",
        "dataset_id",
        "split",
        "candidate_pool",
        "map_variant",
        "source_stage",
        "source_status",
        "source_report",
        "source_sha256",
        "notes",
    ]
    metadata = methods_frame[metadata_columns].drop_duplicates("method_id")
    return metadata.merge(
        pivot,
        on="method_id",
        how="left",
        validate="one_to_one",
    ).sort_values("method_id", kind="mergesort").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build canonical VILLOC method benchmark ledger and report tables"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()

    config_path = resolve(args.config)
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    validate_config(cfg)

    loaded_sources, source_rows = load_sources(cfg)
    records, method_rows = collect_method_records(cfg, loaded_sources)

    long_frame = records_to_frame(records).sort_values(
        ["method_id", "target", "metric"],
        kind="mergesort",
    ).reset_index(drop=True)
    methods_frame = pd.DataFrame(method_rows).sort_values(
        "method_id",
        kind="mergesort",
    ).reset_index(drop=True)
    sources_frame = pd.DataFrame(source_rows).sort_values(
        "source_id",
        kind="mergesort",
    ).reset_index(drop=True)
    wide_frame = wide_table(long_frame, methods_frame)

    out_root = resolve(cfg["output_root"])
    long_path = out_root / "method_metrics_long.csv"
    wide_path = out_root / "method_metrics_wide.csv"
    methods_path = out_root / "method_registry.csv"
    sources_path = out_root / "source_provenance.csv"
    markdown_path = out_root / "method_benchmark_tables.md"
    ledger_path = out_root / "method_benchmark_ledger.json"

    long_bytes = canonical_csv(long_frame)
    wide_bytes = canonical_csv(wide_frame)
    methods_bytes = canonical_csv(methods_frame)
    sources_bytes = canonical_csv(sources_frame)
    markdown = report_markdown(
        long_frame,
        methods_frame,
        title=str(cfg["title"]),
    )
    markdown_bytes = (markdown + "\n").encode("utf-8")

    ledger = {
        "schema_version": int(cfg["schema_version"]),
        "title": str(cfg["title"]),
        "status": "PASS_METHOD_BENCHMARK_LEDGER_BUILT",
        "reporting_only": True,
        "dataset": cfg["dataset"],
        "targets": cfg.get("targets", []),
        "method_count": int(len(methods_frame)),
        "metric_record_count": int(len(long_frame)),
        "source_count": int(len(loaded_sources)),
        "provenance": {
            "config_path": str(config_path.relative_to(ROOT)),
            "config_sha256": sha256_file(config_path),
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "sources": {
                row["source_id"]: {
                    "report": row.get("report"),
                    "report_sha256": row.get("report_sha256"),
                    "source_stage": row.get("source_stage"),
                    "source_status": row.get("source_status"),
                    "loaded": row.get("loaded"),
                }
                for row in source_rows
            },
        },
        "outputs": {
            "method_metrics_long": {
                "path": str(long_path.relative_to(ROOT)),
                "sha256": hashlib.sha256(long_bytes).hexdigest(),
                "rows": int(len(long_frame)),
            },
            "method_metrics_wide": {
                "path": str(wide_path.relative_to(ROOT)),
                "sha256": hashlib.sha256(wide_bytes).hexdigest(),
                "rows": int(len(wide_frame)),
            },
            "method_registry": {
                "path": str(methods_path.relative_to(ROOT)),
                "sha256": hashlib.sha256(methods_bytes).hexdigest(),
                "rows": int(len(methods_frame)),
            },
            "source_provenance": {
                "path": str(sources_path.relative_to(ROOT)),
                "sha256": hashlib.sha256(sources_bytes).hexdigest(),
                "rows": int(len(sources_frame)),
            },
            "report_markdown": {
                "path": str(markdown_path.relative_to(ROOT)),
                "sha256": hashlib.sha256(markdown_bytes).hexdigest(),
            },
        },
    }
    ledger_bytes = canonical_json(ledger)

    write_outputs(
        {
            long_path: long_bytes,
            wide_path: wide_bytes,
            methods_path: methods_bytes,
            sources_path: sources_bytes,
            markdown_path: markdown_bytes,
            ledger_path: ledger_bytes,
        }
    )

    print("Status:", ledger["status"])
    print("Methods:", ledger["method_count"])
    print("Metric records:", ledger["metric_record_count"])
    print("Sources:", ledger["source_count"])
    print("Long ledger:", long_path)
    print("Wide report table:", wide_path)
    print("Markdown report tables:", markdown_path)
    print("Ledger JSON:", ledger_path)
    print(
        "Long ledger SHA256:",
        ledger["outputs"]["method_metrics_long"]["sha256"],
    )
    print(
        "Report Markdown SHA256:",
        ledger["outputs"]["report_markdown"]["sha256"],
    )


if __name__ == "__main__":
    main()
