#!/usr/bin/env python3
"""Read-only audit of the frozen traj01 R5.1 q390 map-state transition.

Blind evidence is printed first. Reference distances are computed only after
the input/output/architecture hashes of the frozen run have been verified.
No outputs or model parameters are changed.
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


def row_dict(row):
    return {key: (None if pd.isna(value) else value)
            for key, value in row.to_dict().items()}


def emit(name, value):
    print(name, json.dumps(value, default=str, allow_nan=False))


OBSERVATION_FIELDS = (
    "query_id", "tile_id", "candidate_choice_rank", "rank", "hybrid_rank",
    "recomputed_good_matches", "recomputed_inliers", "recomputed_homography_ok",
    "projected_easting", "projected_northing", "projected_inside_tile",
    "jacobian_condition", "jacobian_local_area_scale",
)


def observation_dict(row):
    return {name: row_dict(row)[name] for name in OBSERVATION_FIELDS if name in row.index}


def main():
    probe = load_probe()
    freeze_path = RESEARCH / "r5_1_blind_implementation_freeze_manifest.json"
    freeze = json.loads(freeze_path.read_text())
    emit("FREEZE_SHA256", probe.sha256(freeze_path))
    probe.verify(REPO, freeze)
    assert freeze["counts"]["queries"] == 403

    result = json.loads((RESEARCH / "r5_1_blind_policy_results.json").read_text())
    assert result["blind_contract"]["reference_used"] is False
    timeline = pd.read_csv(RESEARCH / "r5_1_blind_policy_timeline.csv")
    hypotheses = pd.read_csv(RESEARCH / "r5_1_blind_subtile_hypotheses.csv")
    leaders = pd.read_csv(RESEARCH / "r5_1_blind_leader_updates.csv")
    observations = pd.read_csv(RESEARCH / "r5_1_blind_top4_subtile_observations.csv")
    assert hypotheses.hypothesis_id.is_unique
    hyp = hypotheses.set_index("hypothesis_id")
    selected = timeline[timeline.policy == POLICY].set_index("update_query_id")
    assert int(selected.loc[390, "active_source_update_q_after"]) == 390
    assert str(selected.loc[390, "action"]) == "TRACKING_ACCEPT"
    old_id = int(selected.loc[389, "active_hypothesis_id_after"])
    new_id = int(selected.loc[390, "active_hypothesis_id_after"])

    print("BLIND AUDIT: q390 acceptance and selected evidence")
    for q in (187, 388, 389, 390, 391, 392, 403):
        emit(f"POLICY_q{q}", row_dict(selected.loc[q]))
    emit("LEADER_q390", row_dict(leaders[leaders.update_query_id == 390].iloc[0]))
    emit("OLD_ACTIVE_HYPOTHESIS", row_dict(hyp.loc[old_id]))
    emit("NEW_ACTIVE_HYPOTHESIS", row_dict(hyp.loc[new_id]))
    new = hyp.loc[new_id]
    evidence = [int(x) for x in str(new.evidence_query_ids).split(",")]
    tile_ids = str(new.tile_ids).split(",")
    emit("NEW_HYPOTHESIS_EVIDENCE", {"query_ids": evidence, "tile_ids": tile_ids})
    assert len(evidence) == len(tile_ids)
    for q, tile_id in zip(evidence, tile_ids):
        rows = observations[(observations.query_id == q) & (observations.tile_id == tile_id)]
        if len(rows) == 1:
            emit(f"SELECTED_OBSERVATION_q{q}", observation_dict(rows.iloc[0]))
        else:
            emit("MISSING_OR_AMBIGUOUS_OBSERVATION", {"q": q, "tile_id": tile_id, "count": len(rows)})
    for q in (389, 390, 391, 392, 393, 400, 403):
        candidates = observations[observations.query_id == q]
        emit(f"TOP4_q{q}", [observation_dict(row) for _, row in candidates.iterrows()])

    # The first reference read is below the completed blind freeze verification.
    reference = pd.read_csv(RUN / "evaluation/reference_attachment.csv")
    relative = pd.read_csv(probe.recorded_path(REPO, freeze["inputs"]["relative_csv"]))
    reference["query_id"] = pd.to_numeric(reference.query_id).astype(int)
    relative["query_id"] = pd.to_numeric(relative.token0_id).astype(int)
    assert reference.query_id.is_unique and relative.query_id.is_unique
    east, north = Transformer.from_crs(
        "EPSG:4326", "EPSG:3346", always_xy=True
    ).transform(reference.eval_ref_lon.to_numpy(float), reference.eval_ref_lat.to_numpy(float))
    joined = relative[["query_id", "visual_x_px", "visual_y_px"]].merge(
        reference[["query_id"]].assign(e=east, n=north), on="query_id", validate="one_to_one"
    ).set_index("query_id")
    maturity = result["policy_results"][POLICY]["maturity_map_state"]
    print("POST-FREEZE EVALUATION (diagnostic only)")
    for q in (389, 390, 391, 392, 393, 400, 403):
        visual = joined.loc[q]
        truth = visual[["e", "n"]].to_numpy(float)
        positions = {}
        for name, model in (("previous_q389", hyp.loc[old_id]),
                            ("accepted_q390", hyp.loc[new_id]),
                            ("maturity_q9", maturity)):
            xy = probe.estimated_xy(visual.visual_x_px, visual.visual_y_px, model)
            positions[name] = {"xy": xy.tolist(), "error_m": round(float(np.linalg.norm(xy - truth)), 2)}
        candidates = []
        for _, row in observations[observations.query_id == q].iterrows():
            xy = np.array([row.projected_easting, row.projected_northing], dtype=float)
            candidates.append({"tile_id": row.tile_id,
                               "choice_rank": int(row.candidate_choice_rank),
                               "error_m": round(float(np.linalg.norm(xy - truth)), 2)})
        emit(f"EVAL_q{q}", {"gt_xy": truth.tolist(), "models": positions,
                            "top4_candidate_errors": candidates})


if __name__ == "__main__":
    main()
