#!/usr/bin/env bash
set -euo pipefail

# Copy compact S6A/S6B documentation assets from outputs/ into docs/assets/.
# Run from the repository root:
#
#   bash docs/copy_s6_relative_absolute_assets.sh

ROOT="$(pwd)"

S6A_FIG_SRC="$ROOT/outputs/satloc/figures/s6a_relative_motion"
S6A_META_SRC="$ROOT/outputs/satloc/metadata/s6a_relative_motion"
S6A_REPORT_SRC="$ROOT/outputs/satloc/reports/s6a_relative_motion"

S6B_FIG_SRC="$ROOT/outputs/satloc/figures/s6b_relative_absolute"
S6B_META_SRC="$ROOT/outputs/satloc/metadata/s6b_relative_absolute"
S6B_REPORT_SRC="$ROOT/outputs/satloc/reports/s6b_relative_absolute"

S6A_DST="$ROOT/docs/assets/s6a_relative_motion"
S6B_DST="$ROOT/docs/assets/s6b_relative_absolute"

mkdir -p \
  "$S6A_DST/figures" \
  "$S6A_DST/tables" \
  "$S6A_DST/reports" \
  "$S6B_DST/figures" \
  "$S6B_DST/tables" \
  "$S6B_DST/reports"

copy_if_exists() {
  local source="$1"
  local destination="$2"

  if [[ -f "$source" ]]; then
    cp -f "$source" "$destination"
    printf 'copied: %s\n' "$source"
  else
    printf 'missing, skipped: %s\n' "$source" >&2
  fi
}

copy_all_png() {
  local source_dir="$1"
  local destination_dir="$2"

  if [[ -d "$source_dir" ]]; then
    find "$source_dir" -maxdepth 1 -type f -name '*.png' -print0 \
      | while IFS= read -r -d '' file; do
          cp -f "$file" "$destination_dir/"
          printf 'copied: %s\n' "$file"
        done
  else
    printf 'missing figure directory, skipped: %s\n' "$source_dir" >&2
  fi
}

copy_all_json() {
  local source_dir="$1"
  local destination_dir="$2"

  if [[ -d "$source_dir" ]]; then
    find "$source_dir" -maxdepth 1 -type f -name '*.json' -print0 \
      | while IFS= read -r -d '' file; do
          cp -f "$file" "$destination_dir/"
          printf 'copied: %s\n' "$file"
        done
  else
    printf 'missing report directory, skipped: %s\n' "$source_dir" >&2
  fi
}

echo "Copying S6A figures..."
copy_all_png "$S6A_FIG_SRC" "$S6A_DST/figures"

echo "Copying S6A JSON reports..."
copy_all_json "$S6A_REPORT_SRC" "$S6A_DST/reports"

# Compact S6A summary tables. Missing names are harmless because historical
# script versions may use slightly different filenames.
for name in \
  s6a1_orb_stride_summary.csv \
  s6a2_orb_relative_trajectory_metrics.csv \
  s6a3_rolling_anchor_safe_horizon_summary.csv \
  s6a4a_klt_relative_motion_summary.csv
do
  copy_if_exists "$S6A_META_SRC/$name" "$S6A_DST/tables/$name"
done

echo "Copying S6B figures..."
copy_all_png "$S6B_FIG_SRC" "$S6B_DST/figures"

echo "Copying S6B JSON reports..."
copy_all_json "$S6B_REPORT_SRC" "$S6B_DST/reports"

# Copy only compact, report-ready S6B tables. Large trajectory and candidate
# score CSVs remain in outputs/ and are intentionally not copied.
for name in \
  s6b0_absolute_correction_spacing.csv \
  s6b1a_position_only_correction_metrics.csv \
  s6b1a_position_only_correction_events.csv \
  s6b1c0_temporal_confirmation_policy_sweep.csv \
  s6b1c1_temporal_hard_reset_metrics.csv \
  s6b1c2_adaptive_soft_metrics.csv \
  s6b1c3_policy_window_summary.csv \
  s6b1c3_policy_selection_summary.csv \
  s6b1c3_event_source_summary.csv \
  s6b2a_raw_bootstrap_causal_winners.csv \
  s6b2b_bootstrap_lock_stability.csv \
  s6b2c_raw_causal_fusion_metrics.csv \
  s6b2d_raw_policy_window_summary.csv \
  s6b2d_raw_event_source_summary.csv \
  s6b2d_controlled_vs_raw_comparison.csv \
  s6b2d_raw_policy_selection.csv
do
  copy_if_exists "$S6B_META_SRC/$name" "$S6B_DST/tables/$name"
done

copy_if_exists \
  "$S6B_META_SRC/s6b2b_selected_bootstrap_transform.json" \
  "$S6B_DST/reports/s6b2b_selected_bootstrap_transform.json"

echo
echo "Asset copy complete."
echo "S6A assets: $S6A_DST"
echo "S6B assets: $S6B_DST"
echo
echo "Inspect copied files with:"
echo "  find docs/assets/s6a_relative_motion docs/assets/s6b_relative_absolute -type f | sort"
