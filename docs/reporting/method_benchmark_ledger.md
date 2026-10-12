# Method benchmark ledger

Purpose: maintain one canonical, extensible record of localization methods and evaluation metrics for analysis, thesis/report tables, and future frontend comparisons.

## Design

The reporting layer reads completed result JSON files. It never feeds ranking, threshold tuning, model training, or deployment decisions.

Canonical outputs:

- `method_metrics_long.csv` — one row per method/target/metric with provenance;
- `method_metrics_wide.csv` — report-friendly one-row-per-method comparison table;
- `method_registry.csv` — method metadata;
- `source_provenance.csv` — source stage/status/hash table;
- `method_benchmark_ledger.json` — machine-readable summary/provenance;
- `method_benchmark_tables.md` — publication/report-ready Markdown tables.

Every normalized metric row records:

- method ID and display name;
- frontend;
- retrieval pool;
- selector/formulation;
- dataset and split;
- target;
- metric/value/unit;
- hit numerator/denominator when applicable;
- source stage/status/report;
- source report SHA256;
- original JSON path.

## Current registry

The initial registry includes:

- QV1.4 retrieval rank baseline;
- ORB inlier ranking;
- ORB GD1.2 evidence-family models;
- ORB GD1.3-B query-relative/locality models;
- raw ORB frontend coverage from GD1.0;
- raw XFeat frontend coverage from LF1.1-A.

The registry is configured in:

`configs/reporting/villoc_method_benchmark_registry.yaml`

## Run

```bash
source .drone_venv/bin/activate

python -m unittest tests.test_method_benchmark_ledger -v

python scripts/villoc/reporting/build_method_benchmark_ledger.py \
  --config configs/reporting/villoc_method_benchmark_registry.yaml
```

Expected status:

`PASS_METHOD_BENCHMARK_LEDGER_BUILT`

## Adding a future method

Normally no collector-code change is required.

1. Add the completed result report under `sources:`.
2. Pin its expected PASS status.
3. Pin a frozen output/evidence hash when one is available.
4. Provide JSON path templates for standardized metrics.
5. Add one or more method entries referencing that source.

Example selector-style source:

```yaml
sources:
  future_frontend_eval:
    report: outputs/.../report.json
    expected_status: PASS_...
    source_stage: LF2.1
    required: true
    expected_artifact:
      json_path: outputs.scores.sha256
      sha256: <frozen hash>
    method_metric_paths:
      pair_auc_oof: results.{target}.{method}.pair_auc_oof
      top1_hits: results.{target}.{method}.query_hits.top1
      top3_hits: results.{target}.{method}.query_hits.top3
      top5_hits: results.{target}.{method}.query_hits.top5
      top10_hits: results.{target}.{method}.query_hits.top10
```

Then:

```yaml
methods:
  - method_id: xfeat_inliers
    display_name: XFeat inliers
    family: geometry
    frontend: XFeat
    retrieval: QV1.4 Top20
    selector: inlier count
    source: future_frontend_eval
    method_key: xfeat_inliers
```

Top-k hit counts are automatically converted to rates using the locked dataset query count.

## Reporting rule

The generated Markdown/wide CSV are suitable as inputs when drafting figures/tables, but claims should still cite the underlying experiment stage and preserve development/blind/test distinctions.

Runtime/resource metrics can be added as additional source paths once they are measured under a controlled benchmark protocol. Do not mix machine-dependent smoke timings with accuracy metrics without recording device/runtime provenance.
