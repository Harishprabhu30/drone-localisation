#!/usr/bin/env python3
"""Blind counterfactual replay of transform-continuity guards on traj01 R5.1.

Read-only: verifies the original freeze, reproduces the recorded quarter/quarter
policy exactly, then replays with geometry-derived 51.2 m map-state jump and
exploratory 90-degree rotation discontinuity guards. Ground truth is opened
only after all blind replays, solely to evaluate their saved paths. These are
experimental diagnostics, not validated production policies.
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
    path = REPO / "outputs/benchmark_probes/traj01_r5_1_postfreeze_probe_full403.py"
    spec = importlib.util.spec_from_file_location("frozen_quality_probe", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    probe = load_probe()
    freeze_path = RESEARCH / "r5_1_blind_implementation_freeze_manifest.json"
    freeze = json.loads(freeze_path.read_text())
    print("Blind freeze SHA256:", probe.sha256(freeze_path))
    probe.verify(REPO, freeze)
    assert int(freeze["counts"]["queries"]) == 403
    results = json.loads((RESEARCH / "r5_1_blind_policy_results.json").read_text())
    assert results["blind_contract"]["reference_used"] is False

    relative = pd.read_csv(probe.recorded_path(REPO, freeze["inputs"]["relative_csv"]))
    relative["query_id"] = pd.to_numeric(relative.token0_id).astype(int)
    visual = relative.set_index("query_id")
    assert visual.index.is_unique and all(q in visual.index for q in range(1, 404))
    observations = pd.read_csv(RESEARCH / "r5_1_blind_top4_subtile_observations.csv")
    observations = observations[
        observations.recomputed_homography_ok.astype(str).str.lower().eq("true")
    ].copy()
    groups = {int(q): group for q, group in observations.groupby("query_id")}
    hypotheses = pd.read_csv(RESEARCH / "r5_1_blind_subtile_hypotheses.csv")
    assert hypotheses.hypothesis_id.is_unique
    models = hypotheses.set_index("hypothesis_id")
    leaders = pd.read_csv(RESEARCH / "r5_1_blind_leader_updates.csv")
    leaders = {
        int(row.update_query_id): int(row.blind_leader_hypothesis_id)
        for row in leaders.itertuples()
        if pd.notna(row.blind_leader_hypothesis_id)
    }
    frozen = pd.read_csv(RESEARCH / "r5_1_blind_policy_timeline.csv")
    frozen = frozen[frozen.policy == POLICY].set_index("update_query_id")
    policy = results["policy_results"][POLICY]
    threshold = float(policy["tracking_threshold_m"])
    activation = float(policy["activation_threshold_m"])
    support = int(policy["maturity_support_required"])
    contract = json.loads((REPO / "configs/bootstrap/minimum_confident_v2/architecture_contract.json").read_text())
    max_jump_m = float(contract["candidate_policy_family"]["map_spacing_m"])
    max_rotation_deg = 90.0  # Exploratory half-turn exclusion, not frozen contract.
    seed = min(leaders)
    assert seed in frozen.index

    def replay(guard_mode):
        active = leaders[seed]
        source_q = seed
        mode = "ACQUISITION"
        streak = 0
        maturity_q = None
        rows = [{"q": seed, "action": "SEED", "id": active, "source_q": seed,
                 "leader_endpoint_innovation_m": np.nan,
                 "minimum_innovation_m": np.nan, "state_jump_m": np.nan,
                 "rotation_change_deg": np.nan}]
        for q in range(seed + 1, 404):
            v = visual.loc[q]
            prior = probe.estimated_xy(v.visual_x_px, v.visual_y_px, models.loc[active])
            candidates = groups.get(q)
            if candidates is None or candidates.empty:
                minimum = np.inf
            else:
                points = candidates[["projected_easting", "projected_northing"]].to_numpy(float)
                minimum = float(np.linalg.norm(points - prior[None, :], axis=1).min())
            leader_id = leaders.get(q)
            endpoint_innovation = np.inf
            state_jump_m = np.nan
            rotation_change_deg = np.nan
            if leader_id is not None:
                leader = models.loc[leader_id]
                proposed = probe.estimated_xy(v.visual_x_px, v.visual_y_px, leader)
                state_jump_m = float(np.linalg.norm(proposed - prior))
                old_deg = float(models.loc[active, "rotation_deg"])
                new_deg = float(leader.rotation_deg)
                rotation_change_deg = abs((new_deg - old_deg + 180.0) % 360.0 - 180.0)
                evidence = [int(value) for value in str(leader.evidence_query_ids).split(",")]
                tiles = str(leader.tile_ids).split(",")
                if q in evidence and candidates is not None:
                    selected_tile = tiles[evidence.index(q)]
                    selected = candidates[candidates.tile_id == selected_tile]
                    if len(selected) == 1:
                        point = selected.iloc[0][["projected_easting", "projected_northing"]].to_numpy(float)
                        endpoint_innovation = float(np.linalg.norm(point - prior))
            if mode == "ACQUISITION":
                if leader_id is None:
                    action = "ACQUISITION_NO_LEADER"
                    streak = 0
                else:
                    action = "ACQUISITION_ACCEPT"
                    active = leader_id
                    source_q = q
                    streak = streak + 1 if minimum <= activation else 0
                    if streak >= support:
                        mode = "TRACKING"
                        maturity_q = q
            elif leader_id is None:
                action = "TRACKING_HOLD_NO_LEADER"
            elif minimum <= threshold:
                fail_jump = guard_mode in ("map_spacing_jump", "both") and state_jump_m > max_jump_m
                fail_rotation = guard_mode in ("rotation_flip", "both") and rotation_change_deg > max_rotation_deg
                if fail_jump or fail_rotation:
                    action = "TRACKING_HOLD_TRANSFORM_CONTINUITY"
                else:
                    action = "TRACKING_ACCEPT"
                    active = leader_id
                    source_q = q
            else:
                action = "TRACKING_HOLD_INNOVATION"
            rows.append({"q": q, "action": action, "id": active,
                         "source_q": source_q,
                         "leader_endpoint_innovation_m": endpoint_innovation,
                         "minimum_innovation_m": minimum,
                         "state_jump_m": state_jump_m,
                         "rotation_change_deg": rotation_change_deg})
        return pd.DataFrame(rows).set_index("q"), maturity_q

    original, maturity = replay("original")
    assert maturity == int(policy["matured_at_query_id"])
    assert original.index.equals(frozen.index)
    mismatched = original[(original.action.to_numpy() != frozen.action.to_numpy()) |
                          (original.id.to_numpy() != frozen.active_hypothesis_id_after.to_numpy(int)) |
                          (original.source_q.to_numpy() != frozen.active_source_update_q_after.to_numpy(int))]
    assert mismatched.empty, f"Original replay differs at queries: {mismatched.index.tolist()[:20]}"
    print("Original policy replay: EXACT action, active hypothesis, and source match")
    trajectories = {"original": original}
    print("BLIND GUARD LIMITS", json.dumps({"map_spacing_jump_m": max_jump_m,
                                            "rotation_flip_deg": max_rotation_deg}))
    for name in ("map_spacing_jump", "rotation_flip", "both"):
        trajectory, guarded_maturity = replay(name)
        assert guarded_maturity == maturity
        trajectories[name] = trajectory
        blocked = trajectory[trajectory.action == "TRACKING_HOLD_TRANSFORM_CONTINUITY"]
        print("BLIND REPLAY", name, json.dumps({
            "maturity_q": maturity,
            "tracking_accepts": int((trajectory.action == "TRACKING_ACCEPT").sum()),
            "guarded_holds": blocked.index.tolist(),
            "final_source_q": int(trajectory.iloc[-1].source_q),
        }))
    for q in (134, 390):
        leader = models.loc[leaders[q]]
        before = models.loc[int(original.loc[q - 1, "id"])]
        v = visual.loc[q]
        prior_xy = probe.estimated_xy(v.visual_x_px, v.visual_y_px, before)
        leader_xy = probe.estimated_xy(v.visual_x_px, v.visual_y_px, leader)
        evidence = [int(value) for value in str(leader.evidence_query_ids).split(",")]
        selected_tile = str(leader.tile_ids).split(",")[evidence.index(q)]
        print(f"q{q} blind comparison:", json.dumps({
            "leader_q_tile": selected_tile,
            "old_rotation_deg": float(before.rotation_deg),
            "leader_rotation_deg": float(leader.rotation_deg),
            "model_jump_m": float(np.linalg.norm(leader_xy - prior_xy)),
            "original": {"action": str(original.loc[q, "action"]),
                         "active_source_q": int(original.loc[q, "source_q"]),
                         "minimum_innovation_m": float(original.loc[q, "minimum_innovation_m"]),
                         "leader_endpoint_innovation_m": float(original.loc[q, "leader_endpoint_innovation_m"])},
            "variants": {
                name: {"action": str(table.loc[q, "action"]),
                       "active_source_q": int(table.loc[q, "source_q"]),
                       "minimum_innovation_m": float(table.loc[q, "minimum_innovation_m"]),
                       "state_jump_m": float(table.loc[q, "state_jump_m"]),
                       "rotation_change_deg": float(table.loc[q, "rotation_change_deg"])}
                for name, table in trajectories.items()
            },
        }))

    # The first GT/reference read happens after every blind replay.
    reference = pd.read_csv(RUN / "evaluation/reference_attachment.csv")
    reference["query_id"] = pd.to_numeric(reference.query_id).astype(int)
    assert reference.query_id.is_unique
    east, north = Transformer.from_crs("EPSG:4326", "EPSG:3346", always_xy=True).transform(
        reference.eval_ref_lon.to_numpy(float), reference.eval_ref_lat.to_numpy(float))
    truth = reference[["query_id"]].assign(e=east, n=north).set_index("query_id")

    for name, table in trajectories.items():
        errors = {}
        for q in range(maturity, 404):
            v = visual.loc[q]
            estimate = probe.estimated_xy(v.visual_x_px, v.visual_y_px,
                                          models.loc[int(table.loc[q, "id"])])
            gt = truth.loc[q, ["e", "n"]].to_numpy(float)
            errors[q] = float(np.linalg.norm(estimate - gt))
        print("POSTFREEZE", name, json.dumps({
            "q9_38_rmse_m": round(float(np.sqrt(np.mean([errors[q] ** 2 for q in range(9, 39)]))), 3),
            "q9_403_rmse_m": round(float(np.sqrt(np.mean([e ** 2 for e in errors.values()]))), 3),
            "q390_403_rmse_m": round(float(np.sqrt(np.mean([errors[q] ** 2 for q in range(390, 404)]))), 3),
            "q390_error_m": round(errors[390], 3),
            "q403_error_m": round(errors[403], 3),
            "q133_error_m": round(errors[133], 3),
            "q134_error_m": round(errors[134], 3),
            "q135_error_m": round(errors[135], 3),
        }))


if __name__ == "__main__":
    main()
