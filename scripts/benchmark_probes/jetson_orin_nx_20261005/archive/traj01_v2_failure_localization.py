#!/usr/bin/env python3
"""Read-only post-freeze diagnostic for traj01 R5.1 v2 tracking.

Uses the saved blind timeline and observations. Reference is opened only after
checking the R5.1 freeze's input/output hashes. GT candidate distances below
are evaluation-only and must never become online selection inputs.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer


REPO = Path.cwd().resolve()
RUN = REPO / "outputs/demo_runs/traj01_blind_regression_001"
RESEARCH = REPO / ("outputs/research_runs/minimum_confident_bootstrap/"
                   "traj01_blind_r5_1_v2_full403_20261005_001")
POLICY = "activate_quarter_track_quarter"


def load_probe():
    source = REPO / "outputs/benchmark_probes/traj01_r5_1_postfreeze_probe_full403.py"
    spec = importlib.util.spec_from_file_location("frozen_quality_probe", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def first_sustained(rows, column, threshold, duration=5):
    values = rows[column].to_numpy(float)
    for i in range(len(values) - duration + 1):
        if np.all(values[i:i + duration] > threshold):
            return int(rows.iloc[i].query_id)
    return None


def main():
    probe = load_probe()
    freeze_path = RESEARCH / "r5_1_blind_implementation_freeze_manifest.json"
    freeze = json.loads(freeze_path.read_text())
    print("blind_freeze_sha256", probe.sha256(freeze_path))
    probe.verify(REPO, freeze)
    assert int(freeze["counts"]["queries"]) == 403
    result = json.loads((RESEARCH / "r5_1_blind_policy_results.json").read_text())
    assert result["blind_contract"]["reference_used"] is False
    policy = result["policy_results"][POLICY]
    lock_q = int(policy["matured_at_query_id"])

    # The first GT/reference read occurs after verification of the blind freeze.
    ref = pd.read_csv(RUN / "evaluation/reference_attachment.csv")
    rel = pd.read_csv(probe.recorded_path(REPO, freeze["inputs"]["relative_csv"]))
    ref.query_id = pd.to_numeric(ref.query_id).astype(int)
    rel["query_id"] = pd.to_numeric(rel.token0_id).astype(int)
    assert ref.query_id.is_unique and rel.query_id.is_unique
    e, n = Transformer.from_crs("EPSG:4326", "EPSG:3346", always_xy=True).transform(
        ref.eval_ref_lon.to_numpy(float), ref.eval_ref_lat.to_numpy(float))
    truth = ref[["query_id"]].assign(gt_easting=e, gt_northing=n)
    visual = rel[["query_id", "visual_x_px", "visual_y_px"]].merge(
        truth, on="query_id", validate="one_to_one").set_index("query_id")

    timeline = pd.read_csv(RESEARCH / "r5_1_blind_policy_timeline.csv")
    timeline = timeline[timeline.policy == POLICY].set_index("update_query_id")
    hypotheses = pd.read_csv(RESEARCH / "r5_1_blind_subtile_hypotheses.csv")
    assert hypotheses.hypothesis_id.is_unique
    hypotheses = hypotheses.set_index("hypothesis_id")
    observations = pd.read_csv(RESEARCH / "r5_1_blind_top4_subtile_observations.csv")
    valid = observations.recomputed_homography_ok.astype(str).str.lower().eq("true")
    inside = observations.projected_inside_tile.astype(str).str.lower().eq("true")
    observations = observations[valid & inside].copy()
    observations = observations.groupby("query_id")

    assert all(q in visual.index and q in timeline.index for q in range(lock_q, 404))
    rows = []
    for q in range(lock_q, 404):
        v = visual.loc[q]
        t = timeline.loc[q]
        current = hypotheses.loc[int(t.active_hypothesis_id_after)]
        truth_xy = v[["gt_easting", "gt_northing"]].to_numpy(float)
        current_xy = probe.estimated_xy(v.visual_x_px, v.visual_y_px, current)
        fixed_xy = probe.estimated_xy(v.visual_x_px, v.visual_y_px, policy["maturity_map_state"])
        candidate_errors = []
        if q in observations.groups:
            group = observations.get_group(q)
            candidate_errors = np.hypot(
                group.projected_easting.to_numpy(float) - truth_xy[0],
                group.projected_northing.to_numpy(float) - truth_xy[1])
        previous_model_jump = np.nan
        if q - 1 in timeline.index:
            previous = hypotheses.loc[int(timeline.loc[q - 1].active_hypothesis_id_after)]
            previous_xy = probe.estimated_xy(v.visual_x_px, v.visual_y_px, previous)
            previous_model_jump = float(np.linalg.norm(current_xy - previous_xy))
        rows.append({
            "query_id": q,
            "error_m": float(np.linalg.norm(current_xy - truth_xy)),
            "fixed_q9_error_m": float(np.linalg.norm(fixed_xy - truth_xy)),
            "best_inside_top4_error_m": float(np.min(candidate_errors)) if len(candidate_errors) else np.nan,
            "action": str(t.action),
            "source_q": int(t.active_source_update_q_after),
            "hypothesis_id": int(t.active_hypothesis_id_after),
            "model_jump_at_q_m": previous_model_jump,
            "scale_m_per_visual_px": float(current.scale_m_per_visual_px),
            "rotation_deg": float(current.rotation_deg),
        })
    table = pd.DataFrame(rows)
    assert abs(table.iloc[0].error_m - 11.609) < 0.02, "q9 result changed; inspect before interpreting"
    print("policy", POLICY, "lock_q", lock_q, "evaluated_queries", len(table))
    print("SEGMENTS: RMSE is the saved active-state replay; fixed q9 is a counterfactual diagnostic")
    for low, high in ((9, 38), (39, 60), (61, 90), (91, 128),
                      (129, 200), (201, 300), (301, 390), (391, 403)):
        sub = table[table.query_id.between(low, high)]
        if sub.empty:
            continue
        print(json.dumps({
            "q": f"{low}-{high}", "n": len(sub),
            "active_rmse_m": round(float(np.sqrt(np.mean(sub.error_m ** 2))), 2),
            "fixed_q9_rmse_m": round(float(np.sqrt(np.mean(sub.fixed_q9_error_m ** 2))), 2),
            "active_last_error_m": round(float(sub.iloc[-1].error_m), 2),
            "top4_inside_within_40m": int((sub.best_inside_top4_error_m <= 40).sum()),
            "top4_inside_available": int(sub.best_inside_top4_error_m.notna().sum()),
            "tracking_accepts": int((sub.action == "TRACKING_ACCEPT").sum()),
        }))

    print("FIRST SUSTAINED 5-QUERY ERRORS (>40m, >100m):",
          first_sustained(table, "error_m", 40), first_sustained(table, "error_m", 100))
    print("FIRST SUSTAINED FIXED-q9 ERRORS (>40m, >100m):",
          first_sustained(table, "fixed_q9_error_m", 40),
          first_sustained(table, "fixed_q9_error_m", 100))
    accepted = table[table.action == "TRACKING_ACCEPT"].copy()
    print("LARGEST TRACKING MODEL JUMPS")
    for row in accepted.nlargest(12, "model_jump_at_q_m").itertuples():
        print(json.dumps({
            "q": row.query_id, "source_q": row.source_q,
            "model_jump_m": round(row.model_jump_at_q_m, 2),
            "active_error_m": round(row.error_m, 2),
            "fixed_q9_error_m": round(row.fixed_q9_error_m, 2),
            "best_inside_top4_error_m": round(row.best_inside_top4_error_m, 2),
            "scale": round(row.scale_m_per_visual_px, 4),
            "rotation_deg": round(row.rotation_deg, 2),
        }))
    print("FINAL QUERIES")
    for row in table.tail(15).itertuples():
        print(json.dumps({
            "q": row.query_id, "action": row.action, "source_q": row.source_q,
            "active_error_m": round(row.error_m, 2),
            "fixed_q9_error_m": round(row.fixed_q9_error_m, 2),
            "best_inside_top4_error_m": round(row.best_inside_top4_error_m, 2),
        }))


if __name__ == "__main__":
    main()
