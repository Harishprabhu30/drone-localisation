#!/usr/bin/env bash
set -euo pipefail

# Run from repo root:
#   bash docs/copy_s5c_assets_to_docs.sh
#
# This copies compact figures/tables needed by the S5C README into docs/assets.
# It intentionally avoids heavy descriptor caches.

ASSET_DIR="docs/assets/s5c_temporal"
TABLE_DIR="$ASSET_DIR/tables"
REPORT_DIR="$ASSET_DIR/reports"

mkdir -p "$ASSET_DIR" "$TABLE_DIR" "$REPORT_DIR"

copy_if_exists () {
  src="$1"
  dst="$2"
  if [ -e "$src" ]; then
    cp -v "$src" "$dst"
  else
    echo "WARN missing: $src"
  fi
}

# S5C.0 figures
copy_if_exists outputs/satloc/figures/s5c_temporal/s5c0_query_spacing_histogram.png "$ASSET_DIR/s5c0_query_spacing_histogram.png"
copy_if_exists outputs/satloc/figures/s5c_temporal/s5c0_query_source_breakdown.png "$ASSET_DIR/s5c0_query_source_breakdown.png"
copy_if_exists outputs/satloc/figures/s5c_temporal/s5c0_query_locations.png "$ASSET_DIR/s5c0_query_locations.png"

# S5C.1 figures
copy_if_exists outputs/satloc/figures/s5c_temporal/s5b1b_fullmap_oracle_topk_hit_rate_s5c1_temporal_full263.png "$ASSET_DIR/s5b1b_fullmap_oracle_topk_hit_rate_s5c1_temporal_full263.png"
copy_if_exists outputs/satloc/figures/s5c_temporal/s5b1b_fullmap_median_error_s5c1_temporal_full263.png "$ASSET_DIR/s5b1b_fullmap_median_error_s5c1_temporal_full263.png"
copy_if_exists outputs/satloc/figures/s5c_temporal/s5b1c_union_first_correct_rank_s5c1_temporal_full263.png "$ASSET_DIR/s5b1c_union_first_correct_rank_s5c1_temporal_full263.png"

# S5C.2 chunk policy figures
for f in outputs/satloc/figures/s5c_temporal/s5b2_lightglue_union_policy_hit_rates_s5c2_temporal_union_top50_chunk*.png; do
  [ -e "$f" ] || continue
  cp -v "$f" "$ASSET_DIR/$(basename "$f")"
done

# S5C.3 figures
copy_if_exists outputs/satloc/figures/s5c_temporal/s5c3_lg_score_vs_error.png "$ASSET_DIR/s5c3_lg_score_vs_error.png"
copy_if_exists outputs/satloc/figures/s5c_temporal/s5c3_inliers_vs_error.png "$ASSET_DIR/s5c3_inliers_vs_error.png"
copy_if_exists outputs/satloc/figures/s5c_temporal/s5c3_gate_precision_vs_accepted.png "$ASSET_DIR/s5c3_gate_precision_vs_accepted.png"

# S5C tables / compact CSVs
copy_if_exists outputs/satloc/metadata/s5c_temporal/s5c0_absolute_query_manifest.csv "$TABLE_DIR/s5c0_absolute_query_manifest.csv"
copy_if_exists outputs/satloc/metadata/s5c_temporal/s5b1b_fullmap_variant_summary_s5c1_temporal_full263.csv "$TABLE_DIR/s5b1b_fullmap_variant_summary_s5c1_temporal_full263.csv"
copy_if_exists outputs/satloc/metadata/s5c_temporal/s5b1b_fullmap_query_summary_s5c1_temporal_full263.csv "$TABLE_DIR/s5b1b_fullmap_query_summary_s5c1_temporal_full263.csv"
copy_if_exists outputs/satloc/metadata/s5c_temporal/s5b1c_union_query_summary_s5c1_temporal_full263.csv "$TABLE_DIR/s5b1c_union_query_summary_s5c1_temporal_full263.csv"
copy_if_exists outputs/satloc/metadata/s5c_temporal/s5c2_lightglue_union_policy_summary_top50_full263.csv "$TABLE_DIR/s5c2_lightglue_union_policy_summary_top50_full263.csv"
copy_if_exists outputs/satloc/metadata/s5c_temporal/s5c2_lightglue_union_policy_summary_chunks_top50_full263.csv "$TABLE_DIR/s5c2_lightglue_union_policy_summary_chunks_top50_full263.csv"
copy_if_exists outputs/satloc/metadata/s5c_temporal/s5c2_lightglue_union_query_summary_top50_full263.csv "$TABLE_DIR/s5c2_lightglue_union_query_summary_top50_full263.csv"
copy_if_exists outputs/satloc/metadata/s5c_temporal/s5c3_lightglue_confidence_features_full263.csv "$TABLE_DIR/s5c3_lightglue_confidence_features_full263.csv"
copy_if_exists outputs/satloc/metadata/s5c_temporal/s5c3_recommended_confidence_gates_full263.csv "$TABLE_DIR/s5c3_recommended_confidence_gates_full263.csv"
copy_if_exists outputs/satloc/metadata/s5c_temporal/s5c3_confidence_gate_sweep_full263.csv "$TABLE_DIR/s5c3_confidence_gate_sweep_full263.csv"

# Reports
copy_if_exists outputs/satloc/reports/s5c_temporal/s5c2_lightglue_union_top50_full263_summary.json "$REPORT_DIR/s5c2_lightglue_union_top50_full263_summary.json"
copy_if_exists outputs/satloc/reports/s5c_temporal/s5c3_confidence_gate_calibration_summary.json "$REPORT_DIR/s5c3_confidence_gate_calibration_summary.json"
copy_if_exists outputs/satloc/reports/s5c_temporal/s5b1b_fullmap_domain_normalized_phog_summary_s5c1_temporal_full263.json "$REPORT_DIR/s5b1b_fullmap_domain_normalized_phog_summary_s5c1_temporal_full263.json"
copy_if_exists outputs/satloc/reports/s5c_temporal/s5b1c_union_candidate_pool_summary_s5c1_temporal_full263.json "$REPORT_DIR/s5b1c_union_candidate_pool_summary_s5c1_temporal_full263.json"

echo
echo "S5C assets copied into: $ASSET_DIR"
echo "Now copy README into docs:"
echo "  cp /path/to/README_s5c_temporal_absolute_benchmark.md docs/README_s5c_temporal_absolute_benchmark.md"
