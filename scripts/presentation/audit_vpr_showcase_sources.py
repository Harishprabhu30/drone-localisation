#!/usr/bin/env python3
"""Audit VPR showcase inputs and build a reusable hardest-frame shortlist."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd

SOURCES = {
    "uav_index": ("outputs/satloc/metadata/uav_frames_index_enriched.csv", ["token0_id", "image_path"]),
    "satellite_index": ("outputs/satloc/metadata/satellite_tiles_index_enriched.csv", ["tile_index", "tile_path", "lon_center", "lat_center"]),
    "query_manifest": ("outputs/satloc/metadata/s7_retrieval_upgrade/s7_0_query_manifest.csv", ["token0_id", "image_path"]),
    "scene_labels": ("outputs/satloc/metadata/s7_retrieval_upgrade/s7_scene_labels_canonical_traj01.csv", ["token", "primary_scene"]),
    "s4c6_by_token": ("outputs/satloc/metadata/s4c_macrocontour_phog_chamfer/s4c6_failure_analysis/s4c6c_failure_groups_by_token.csv", ["token", "failure_group"]),
    "s4c6_summary": ("outputs/satloc/metadata/s4c_macrocontour_phog_chamfer/s4c6_failure_analysis/s4c6c_failure_group_summary.csv", ["failure_group", "count", "rate"]),
    "s7b4_failures": ("outputs/satloc/metadata/s7b_failure_rescue/s7b4_failure_token_diagnostics_center_resize_k32_img224_top100.csv", ["token", "primary_scene", "failure_group"]),
    "s7b4_anchor_misses": ("outputs/satloc/metadata/s7b_failure_rescue/s7b4_anchor_miss_shortlist_center_resize_k32_img224_top100.csv", ["token", "primary_scene", "failure_group"]),
    "s7b4_budget_recall": ("outputs/satloc/metadata/s7b_failure_rescue/s7b4_budget_recall_curves_center_resize_k32_img224_top100.csv", ["stream", "budget_k", "hits", "queries", "recall"]),
    "s7b4_scene_recall": ("outputs/satloc/metadata/s7b_failure_rescue/s7b4_scene_budget_recall_center_resize_k32_img224_top100.csv", ["stream", "primary_scene", "queries"]),
    "dinovlad_center": ("outputs/satloc/metadata/s7b_dinov2_vlad/s7b2_dinov2_vlad_candidate_scores_torch_hub_dinov2_vits14_vlad_k32_img224_top100_q263_sat8625_cb500_full263_k32_cb500_img224.csv", ["token", "rank", "tile_id", "eval_error_m", "query_image_path", "sat_image_path"]),
    "dinovlad_resize": ("outputs/satloc/metadata/s7b_dinov2_vlad/s7b2_dinov2_vlad_candidate_scores_torch_hub_dinov2_vits14_vlad_k32_img224_top100_q263_sat8625_cb500_full263_k32_cb500_img224_resize_square.csv", ["token", "rank", "tile_id", "eval_error_m", "query_image_path", "sat_image_path"]),
}

GLOBS = {
    "s4c4c_raw_token_csvs": "outputs/satloc/metadata/s4c_macrocontour_phog_chamfer/s4c4_vector_skeleton/s4c4c_luma_lsd_rerank/s4c4c_token*_luma_lsd_raw_lsd_scores_top*.csv",
    "s4c4c_panels": "outputs/satloc/figures/s4c_macrocontour_phog_chamfer/s4c4_vector_skeleton/s4c4c_luma_lsd_rerank/s4c4c_token*_luma_lsd_panel_top5.png",
    "s7b4_visual_panels": "outputs/satloc/figures/s7b_failure_rescue/visual_diagnostics/center_resize_k32_img224_top100/s7b4_visual_token_*.png",
}

def csv_audit(path: Path, required: list[str]) -> dict:
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception as exc:
        return {"read_ok": False, "error": str(exc)}
    missing = [c for c in required if c not in df.columns]
    return {"read_ok": True, "rows": len(df), "columns": list(df.columns), "missing_columns": missing, "schema_ok": not missing}

def build_shortlist(root: Path) -> pd.DataFrame:
    rows = []
    s4 = root / SOURCES["s4c6_by_token"][0]
    if s4.exists():
        df = pd.read_csv(s4, low_memory=False)
        for _, r in df.iterrows():
            rows.append({"token": int(r.token), "source": "s4c6", "scene": str(r.get("primary_scene", "")), "failure_group": str(r.failure_group), "phog_error_m": r.get("phog_top1_error_m"), "lsd_error_m": r.get("lsd_top1_error_m"), "oracle_top50_error_m": r.get("oracle_best_top50_error_m")})
    s7 = root / SOURCES["s7b4_failures"][0]
    if s7.exists():
        df = pd.read_csv(s7, low_memory=False)
        for _, r in df.iterrows():
            rows.append({"token": int(r.token), "source": "s7b4", "scene": str(r.get("primary_scene", "")), "failure_group": str(r.failure_group), "anchor_top1_error_m": r.get("anchor_top1_error_m"), "best_any_stream_error_m": r.get("best_error_any_stream_available_m"), "best_stream": r.get("best_stream_available")})
    if not rows:
        return pd.DataFrame(columns=["token", "sources", "scenes", "failure_groups", "selection_priority_score", "recommended_role"])
    long = pd.DataFrame(rows)
    out = []
    for token, g in long.groupby("token"):
        groups = sorted(set(g.failure_group.dropna().astype(str)))
        scenes = sorted(set(x for x in g.scene.dropna().astype(str) if x))
        sources = sorted(set(g.source.astype(str)))
        text = ",".join(groups)
        scene_text = ",".join(scenes)
        score = 0
        if "pool_failure_or_far_ambiguity" in text: score += 100
        if "candidate_pool_failure" in text: score += 80
        if "agricultural_open_field" in scene_text: score += 70
        if "non_anchor_rescue" in text: score += 55
        if "near_pool_miss_40_100m" in text: score += 45
        if "lsd_rescue" in text: score += 30
        role = "hero_candidate" if "non_anchor_rescue" in text else ("hardest_failure" if score >= 70 else "supporting_failure_case")
        row = {"token": int(token), "sources": ",".join(sources), "scenes": scene_text, "failure_groups": text, "selection_priority_score": score, "recommended_role": role}
        for col in ["phog_error_m", "lsd_error_m", "oracle_top50_error_m", "anchor_top1_error_m", "best_any_stream_error_m", "best_stream"]:
            if col in g.columns:
                vals = g[col].dropna()
                row[col] = vals.iloc[0] if len(vals) else None
        out.append(row)
    return pd.DataFrame(out).sort_values(["selection_priority_score", "token"], ascending=[False, True])

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=Path, default=Path("."))
    ap.add_argument("--output-dir", type=Path, default=Path("outputs/presentation/source_audit"))
    args = ap.parse_args()
    root = args.repo_root.resolve()
    out = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    report = {"repo_root": str(root), "sources": {}, "globs": {}}
    flat = []
    for name, (rel, required) in SOURCES.items():
        path = root / rel
        item = {"path": rel, "exists": path.exists()}
        if path.exists(): item["csv"] = csv_audit(path, required)
        report["sources"][name] = item
        c = item.get("csv", {})
        flat.append({"source_name": name, "path": rel, "exists": item["exists"], "rows": c.get("rows"), "schema_ok": c.get("schema_ok"), "missing_columns": ",".join(c.get("missing_columns", []))})
    for name, pattern in GLOBS.items():
        matches = sorted(root.glob(pattern))
        report["globs"][name] = {"pattern": pattern, "count": len(matches), "examples": [str(p.relative_to(root)) for p in matches[:20]]}
    shortlist = build_shortlist(root)
    shortlist.to_csv(out / "vpr_hardest_frame_shortlist.csv", index=False)
    pd.DataFrame(flat).to_csv(out / "vpr_showcase_source_audit.csv", index=False)
    (out / "vpr_showcase_source_audit.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"Audit: {out / 'vpr_showcase_source_audit.json'}")
    print(f"Shortlist: {out / 'vpr_hardest_frame_shortlist.csv'}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
