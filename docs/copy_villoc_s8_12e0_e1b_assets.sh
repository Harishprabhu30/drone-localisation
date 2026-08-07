#!/usr/bin/env bash
set -euo pipefail

# Minimal report asset copier for:
#   README_villoc_s8_12e0_e1b_lightglue_traj01_transition_minimal_assets.md
#
# Run from repository root:
#   cd /Users/harishprabhu/Documents/drone-localisation
#   bash copy_villoc_s8_12e0_e1b_report_assets_minimal.sh
#
# This intentionally copies ONLY report-linked figures and small summary JSON files.
# It does NOT copy full result folders, per-query CSVs, candidate-score CSVs, caches, or smoke/curated runs.

ASSET_ROOT="docs/assets/villoc_s8_12e0_e1b_lightglue_traj01_transition"
mkdir -p "$ASSET_ROOT/figures" "$ASSET_ROOT/tables"

copied=0
missing=0

copy_one() {
  local src="$1"
  local dst="$2"
  if [[ -f "$src" ]]; then
    mkdir -p "$(dirname "$dst")"
    cp "$src" "$dst"
    echo "copied: $src -> $dst"
    copied=$((copied + 1))
  else
    echo "missing: $src"
    missing=$((missing + 1))
  fi
}

# ----------------------------------------------------------------------------
# S8.12E.0 footprint-center-ray diagnostic: only the main report figures.
# ----------------------------------------------------------------------------
E0="outputs/villoc/45_deg/reports/s8_12e0_footprint_center_ray/1024_s512"
E0_DST_F="$ASSET_ROOT/figures/s8_12e0_1024_s512"
E0_DST_T="$ASSET_ROOT/tables/s8_12e0_1024_s512"

copy_one "$E0/figures/s8_12e0_top1_error_boxplot.png" \
         "$E0_DST_F/s8_12e0_top1_error_boxplot.png"
copy_one "$E0/figures/s8_12e0_offset_vs_footprint_improvement.png" \
         "$E0_DST_F/s8_12e0_offset_vs_footprint_improvement.png"
copy_one "$E0/figures/s8_12e0_corrected_body_scatter.png" \
         "$E0_DST_F/s8_12e0_corrected_body_scatter.png"
copy_one "$E0/s8_12e0_summary.json" \
         "$E0_DST_T/s8_12e0_summary.json"

# ----------------------------------------------------------------------------
# S8.12E.1A ORB verifier/reranker: final 45° and 90° report figures only.
# ----------------------------------------------------------------------------
ORB45="outputs/villoc/45_deg/reports/s8_12e1_top20_verifier_reranker/1024_s512_orb_hybrid"
ORB45_DST_F="$ASSET_ROOT/figures/s8_12e1a_orb_45deg_1024_s512_orb_hybrid"
ORB45_DST_T="$ASSET_ROOT/tables/s8_12e1a_orb_45deg_1024_s512_orb_hybrid"

copy_one "$ORB45/figures/s8_12e1_top1_error_boxplot.png" \
         "$ORB45_DST_F/s8_12e1_top1_error_boxplot.png"
copy_one "$ORB45/figures/s8_12e1_original_vs_reranked_error.png" \
         "$ORB45_DST_F/s8_12e1_original_vs_reranked_error.png"
copy_one "$ORB45/figures/s8_12e1_hit_counts.png" \
         "$ORB45_DST_F/s8_12e1_hit_counts.png"
copy_one "$ORB45/s8_12e1_summary.json" \
         "$ORB45_DST_T/s8_12e1_summary.json"

ORB90="outputs/villoc/90_deg/reports/s8_12e1_top20_verifier_reranker/1024_s512_orb_hybrid"
ORB90_DST_F="$ASSET_ROOT/figures/s8_12e1a_orb_90deg_1024_s512_orb_hybrid"
ORB90_DST_T="$ASSET_ROOT/tables/s8_12e1a_orb_90deg_1024_s512_orb_hybrid"

copy_one "$ORB90/figures/s8_12e1_top1_error_boxplot.png" \
         "$ORB90_DST_F/s8_12e1_top1_error_boxplot.png"
copy_one "$ORB90/figures/s8_12e1_original_vs_reranked_error.png" \
         "$ORB90_DST_F/s8_12e1_original_vs_reranked_error.png"
copy_one "$ORB90/figures/s8_12e1_hit_counts.png" \
         "$ORB90_DST_F/s8_12e1_hit_counts.png"
copy_one "$ORB90/s8_12e1_summary.json" \
         "$ORB90_DST_T/s8_12e1_summary.json"

# ----------------------------------------------------------------------------
# S8.12E.1B LightGlue verifier/reranker: final full-run figures only.
# Do not copy smoke/curated runs or all-candidate CSVs.
# ----------------------------------------------------------------------------
LG45="outputs/villoc/45_deg/reports/s8_12e1b_lightglue_top20_verifier_reranker/1024_s512_hybrid_full"
LG45_DST_F="$ASSET_ROOT/figures/s8_12e1b_lg_45deg_1024_s512_hybrid_full"
LG45_DST_T="$ASSET_ROOT/tables/s8_12e1b_lg_45deg_1024_s512_hybrid_full"

copy_one "$LG45/figures/s8_12e1b_lightglue_error_boxplot.png" \
         "$LG45_DST_F/s8_12e1b_lightglue_error_boxplot.png"
copy_one "$LG45/figures/s8_12e1b_original_vs_lightglue_error.png" \
         "$LG45_DST_F/s8_12e1b_original_vs_lightglue_error.png"
copy_one "$LG45/figures/s8_12e1b_containment_hit_counts.png" \
         "$LG45_DST_F/s8_12e1b_containment_hit_counts.png"
copy_one "$LG45/s8_12e1b_summary.json" \
         "$LG45_DST_T/s8_12e1b_summary.json"

LG90="outputs/villoc/90_deg/reports/s8_12e1b_lightglue_top20_verifier_reranker/1024_s512_hybrid_full"
LG90_DST_F="$ASSET_ROOT/figures/s8_12e1b_lg_90deg_1024_s512_hybrid_full"
LG90_DST_T="$ASSET_ROOT/tables/s8_12e1b_lg_90deg_1024_s512_hybrid_full"

copy_one "$LG90/figures/s8_12e1b_lightglue_error_boxplot.png" \
         "$LG90_DST_F/s8_12e1b_lightglue_error_boxplot.png"
copy_one "$LG90/figures/s8_12e1b_original_vs_lightglue_error.png" \
         "$LG90_DST_F/s8_12e1b_original_vs_lightglue_error.png"
copy_one "$LG90/figures/s8_12e1b_containment_hit_counts.png" \
         "$LG90_DST_F/s8_12e1b_containment_hit_counts.png"
copy_one "$LG90/s8_12e1b_summary.json" \
         "$LG90_DST_T/s8_12e1b_summary.json"

cat <<EOF

DONE.
Copied files:  $copied
Missing files: $missing
Asset root:    $ASSET_ROOT

This minimal script intentionally leaves these local only:
  - all-candidate CSVs
  - per-query CSVs
  - smoke/curated LightGlue folders
  - complete figure directories
  - descriptor caches
  - full output folders
EOF
