"""Method benchmark ledger tests."""
from __future__ import annotations

import json
import sys
import yaml
import tempfile
import unittest
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from uavloc.reporting.benchmark_ledger import (
    MetricRecord,
    load_json,
    normalize_coverage_metric,
    normalize_hit_metric,
    records_to_frame,
    report_markdown,
    resolve_json_path,
)


class MethodBenchmarkLedgerTests(unittest.TestCase):
    def test_resolve_json_path(self):
        payload = {"results": {"x": [{"value": 3}, {"value": 7}]}}
        self.assertEqual(
            resolve_json_path(payload, "results.x.1.value"),
            7,
        )

    def test_hit_metric_derives_rate(self):
        rows = normalize_hit_metric("top1_hits", 296.0, 403)
        self.assertEqual(rows[0][0], "top1_hits")
        self.assertEqual(rows[1][0], "top1_rate")
        self.assertAlmostEqual(rows[1][1], 296.0 / 403.0)
        self.assertEqual(rows[1][3], 296.0)
        self.assertEqual(rows[1][4], 403.0)

    def test_coverage_metric_derives_rate(self):
        rows = normalize_coverage_metric("homography_ok", 7995.0, 8060)
        self.assertEqual(rows[1][0], "homography_ok_rate")
        self.assertAlmostEqual(rows[1][1], 7995.0 / 8060.0)

    def test_load_json_accepts_historical_literal_newline(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "report.json"
            path.write_text('{"status":"PASS"}\\n', encoding="utf-8")
            self.assertEqual(load_json(path)["status"], "PASS")

    def test_load_json_rejects_other_garbage(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "report.json"
            path.write_text('{"status":"PASS"} garbage', encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "Invalid JSON"):
                load_json(path)

    def test_registry_does_not_require_feature_count(self):
        config_path = REPO_ROOT / "configs/reporting/villoc_method_benchmark_registry.yaml"
        cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        for source_id, source in cfg["sources"].items():
            required = source.get("method_metric_paths", {})
            self.assertNotIn(
                "feature_count",
                required,
                msg=f"{source_id} must not require feature_count",
            )

    def test_markdown_contains_report_table(self):
        record = MetricRecord(
            method_id="orb",
            display_name="ORB inliers",
            family="geometry",
            frontend="ORB",
            retrieval="QV1.4",
            selector="inliers",
            dataset_id="test",
            split="development",
            candidate_pool="top20",
            map_variant="768_s256",
            source_id="src",
            source_stage="GD",
            source_status="PASS",
            source_report="report.json",
            source_sha256="a" * 64,
            target="contains_query",
            metric="top1_rate",
            value=0.75,
            unit="ratio",
            numerator=3.0,
            denominator=4.0,
            higher_is_better=True,
            source_json_path="results",
            notes="",
        )
        long_frame = records_to_frame([record])
        methods = pd.DataFrame(
            [
                {
                    "method_id": "orb",
                    "display_name": "ORB inliers",
                    "frontend": "ORB",
                    "selector": "inliers",
                    "source_stage": "GD",
                    "source_sha256": "a" * 64,
                }
            ]
        )
        text = report_markdown(long_frame, methods, title="Benchmark")
        self.assertIn("# Benchmark", text)
        self.assertIn("ORB inliers", text)
        self.assertIn("75.0%", text)


if __name__ == "__main__":
    unittest.main()
