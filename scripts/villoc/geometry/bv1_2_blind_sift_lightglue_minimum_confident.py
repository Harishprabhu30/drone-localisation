#!/usr/bin/env python3
"""BV1.2 blind transfer of minimum-confident-v2 to frozen SIFT+LightGlue observations."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import numpy as np
import pandas as pd
import yaml

ROOT = REPO_ROOT
DEFAULT_CONFIG = ROOT / "configs/research/blind_demo_sift_lightglue_transfer_bv1_2.yaml"
HELPER_PATH = ROOT / "scripts/villoc/blind_demo/bootstrap/minimum_confident_v2_backend.py"


def resolve(value):
    p = Path(value).expanduser()
    return p if p.is_absolute() else ROOT / p


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_helpers():
    spec = importlib.util.spec_from_file_location("bv12_minconf_helpers", HELPER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not import frozen minimum-confident helper module")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def verify_config(cfg):
    if cfg.get("stage") != "BV1.2":
        raise RuntimeError("Expected BV1.2 config")
    p = cfg["policy"]
    if int(p["candidate_count"]) != 4 or int(p["evidence_queries"]) != 4:
        raise RuntimeError("Top-4 / four-evidence contract changed")
    if float(p["activation_threshold_m"]) != 12.8 or float(p["tracking_threshold_m"]) != 12.8:
        raise RuntimeError("State-machine thresholds changed")
    if p["ranking"]["primary"] != "inliers_desc":
        raise RuntimeError("SIFT ranking contract changed")
    b = cfg["scientific_boundary"]
    for key in ("reference_allowed", "srt_allowed", "gps_allowed", "oracle_allowed",
                "evaluation_metrics_allowed", "thresholds_tuned_after_blind_result",
                "candidate_count_changed", "evidence_frame_count_changed",
                "state_machine_thresholds_changed", "orb_recompute_allowed",
                "absolute_locked_allowed", "fusion_executed"):
        if b[key] is not False:
            raise RuntimeError(f"{key} must remain false")
    if b["bv1_0_evidence_immutable"] is not True:
        raise RuntimeError("BV1.0 evidence must remain immutable")


def make_hypothesis(helper, update_qid, evidence_qids, visual_xy, combo, policy, hid):
    map_xy = np.asarray(
        [[float(c["projected_easting"]), float(c["projected_northing"])] for c in combo],
        dtype=float,
    )
    if not np.all(np.isfinite(map_xy)):
        return None

    tile_ids = [str(c["tile_id"]) for c in combo]
    if len(set(tile_ids)) < int(policy["minimum_unique_tile_ids"]):
        return None

    map_span = helper.span(map_xy)
    if map_span < float(policy["minimum_map_span_m"]):
        return None

    try:
        model = helper.fit_similarity(visual_xy, map_xy)
    except ValueError:
        return None

    residual = np.linalg.norm(helper.apply_similarity(visual_xy, model) - map_xy, axis=1)
    med = float(np.median(residual))
    mx = float(np.max(residual))
    if med > float(policy["median_measurement_residual_limit_m"]):
        return None
    if mx > float(policy["maximum_measurement_residual_limit_m"]):
        return None

    return {
        "hypothesis_id": int(hid),
        "update_query_id": int(update_qid),
        "evidence_query_ids": [int(x) for x in evidence_qids],
        "tile_ids": tile_ids,
        "candidate_choice_ranks": [int(c["candidate_choice_rank"]) for c in combo],
        "map_observations": [(float(x), float(y)) for x, y in map_xy],
        "map_unique_tile_ids": int(len(set(tile_ids))),
        "map_span_m": float(map_span),
        "median_projected_residual_m": med,
        "max_projected_residual_m": mx,
        "sum_dino_rank": float(sum(float(c["rank"]) for c in combo)),
        "sum_sift_rank": float(sum(float(c["sift_rank"]) for c in combo)),
        "all_projected_inside_tile": bool(
            all(bool(c["projected_inside_tile"]) for c in combo)
        ),
        **model,
    }


def pareto(rows):
    keys = ("median_projected_residual_m", "sum_sift_rank", "sum_dino_rank")
    out = []
    for i, a in enumerate(rows):
        dominated = False
        for j, b in enumerate(rows):
            if i == j:
                continue
            if (
                all(float(b[k]) <= float(a[k]) for k in keys)
                and any(float(b[k]) < float(a[k]) for k in keys)
            ):
                dominated = True
                break
        if not dominated:
            out.append(a)
    return out


def model_payload(h):
    return {
        k: float(h[k])
        for k in (
            "a_real", "a_imag", "b_real", "b_imag",
            "scale_m_per_visual_px", "rotation_deg",
        )
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()

    cfg_path = resolve(args.config)
    cfg = yaml.safe_load(cfg_path.read_text())
    verify_config(cfg)
    helper = load_helpers()

    bv = cfg["bv1_0"]
    bv_report_path = resolve(bv["report"])
    pair_path = resolve(bv["pair_evidence"])
    bv_report = json.loads(bv_report_path.read_text())

    if bv_report.get("status") != bv["expected_status"]:
        raise RuntimeError("BV1.0 is not the frozen PASS")
    if bv_report["provenance"]["candidate_pool_sha256"] != bv["expected_candidate_pool_sha256"]:
        raise RuntimeError("BV1.0 candidate-pool hash mismatch")
    if bv_report["provenance"]["pair_evidence_sha256"] != bv["expected_pair_sha256"]:
        raise RuntimeError("BV1.0 pair hash mismatch")
    if bv_report["provenance"]["selection_evidence_sha256"] != bv["expected_selection_sha256"]:
        raise RuntimeError("BV1.0 selection hash mismatch")
    if sha256(pair_path) != bv["expected_pair_sha256"]:
        raise RuntimeError("BV1.0 pair evidence bytes changed")
    if bv_report["scientific_boundary"]["reference_loaded"] is not False:
        raise RuntimeError("BV1.0 reference boundary violated")

    source = resolve(cfg["source_demo"]["run_root"])
    relative_path = source / cfg["source_demo"]["relative_csv"]
    manifest_path = source / cfg["source_demo"]["manifest_csv"]

    cand = pd.read_csv(pair_path)
    rel = pd.read_csv(relative_path)
    manifest = pd.read_csv(manifest_path)
    helper.assert_blind_safe(cand, "BV1.0 SIFT pair evidence")
    helper.assert_blind_safe(rel, "blind relative trajectory")
    helper.assert_blind_safe(manifest, "blind query manifest")

    required_c = {
        "query_id", "rank", "tile_id", "matches", "inliers",
        "homography_ok", "finite_projection", "projected_inside_tile",
        "projected_easting", "projected_northing",
    }
    if required_c - set(cand.columns):
        raise RuntimeError(f"BV1.0 pair evidence missing {sorted(required_c - set(cand.columns))}")
    if {"token0_id", "visual_x_px", "visual_y_px"} - set(rel.columns):
        raise RuntimeError("Relative trajectory schema changed")
    if {"query_id", "timestamp_s"} - set(manifest.columns):
        raise RuntimeError("Manifest schema changed")

    cand["query_id"] = pd.to_numeric(cand["query_id"], errors="raise").astype(int)
    cand["rank"] = pd.to_numeric(cand["rank"], errors="raise").astype(int)
    cand["inliers"] = pd.to_numeric(cand["inliers"], errors="raise").astype(int)
    cand["tile_id"] = cand["tile_id"].astype(str)
    for col in ("homography_ok", "finite_projection", "projected_inside_tile"):
        cand[col] = cand[col].astype(str).str.lower().isin({"true", "1", "yes"})

    cand = cand.sort_values(
        ["query_id", "inliers", "rank", "tile_id"],
        ascending=[True, False, True, True],
        kind="mergesort",
    )
    cand["sift_rank"] = cand.groupby("query_id").cumcount() + 1
    cand["candidate_choice_rank"] = cand["sift_rank"]
    cand = cand[cand["sift_rank"] <= int(cfg["policy"]["candidate_count"])].copy()

    rel["query_id"] = pd.to_numeric(rel["token0_id"], errors="raise").astype(int)
    rel["visual_x_px"] = pd.to_numeric(rel["visual_x_px"], errors="raise")
    rel["visual_y_px"] = pd.to_numeric(rel["visual_y_px"], errors="raise")
    manifest["query_id"] = pd.to_numeric(manifest["query_id"], errors="raise").astype(int)
    manifest["timestamp_s"] = pd.to_numeric(manifest["timestamp_s"], errors="coerce")
    rel = rel.merge(
        manifest[["query_id", "timestamp_s"]].drop_duplicates("query_id"),
        on="query_id", how="left", validate="one_to_one",
    )

    common = sorted(set(cand["query_id"]) & set(rel["query_id"]))
    if len(common) != int(cfg["source_demo"]["expected_queries"]):
        raise RuntimeError("BV1.2 expected 123 blind queries")

    short = cand.groupby("query_id").size()
    short = short[short < int(cfg["policy"]["candidate_count"])]
    if len(short):
        raise RuntimeError(f"Queries with fewer than Top-4: {short.head().to_dict()}")

    observations = cand[
        [
            "query_id", "rank", "tile_id", "sift_rank", "candidate_choice_rank",
            "matches", "inliers", "homography_ok", "finite_projection",
            "projected_inside_tile", "projected_easting", "projected_northing",
        ]
    ].copy()
    ts = manifest.set_index("query_id")["timestamp_s"]
    observations.insert(1, "timestamp_s", observations["query_id"].map(ts))
    observations = observations.rename(
        columns={"matches": "stored_matches", "inliers": "stored_inliers"}
    )
    observations["valid_projected_observation"] = (
        observations["homography_ok"] & observations["finite_projection"]
    )

    by_q = {
        int(q): g[g["valid_projected_observation"]].to_dict("records")
        for q, g in observations.groupby("query_id", sort=True)
    }
    rel = rel[rel["query_id"].isin(common)].sort_values("query_id").reset_index(drop=True)
    rel_by_q = rel.set_index("query_id")

    policy = cfg["policy"]
    evidence_frames = int(policy["evidence_queries"])
    updates = []
    hypotheses = []
    leaders = {}
    hid = 1

    for q in common:
        prefix = rel[rel["query_id"] <= q]
        row = {
            "update_query_id": int(q),
            "timestamp_s": float(ts.get(q, np.nan)),
            "evidence_query_ids": "",
            "visual_span_px": math.nan,
            "enumerated_hypotheses": 0,
            "geometric_pass_hypotheses": 0,
            "pareto_hypotheses": 0,
            "blind_leader_hypothesis_id": math.nan,
            "blind_leader_scale": math.nan,
            "blind_leader_rotation_deg": math.nan,
            "action": "",
        }

        evidence = helper.select_evidence(prefix, evidence_frames)
        if len(evidence) != evidence_frames:
            row["action"] = "WAIT_INSUFFICIENT_EVIDENCE_FRAMES"
            updates.append(row)
            continue

        visual_xy = rel_by_q.loc[evidence, ["visual_x_px", "visual_y_px"]].to_numpy(float)
        visual_span = helper.span(visual_xy)
        row["evidence_query_ids"] = ",".join(map(str, evidence))
        row["visual_span_px"] = visual_span
        if visual_span < float(policy["minimum_visual_span_px"]):
            row["action"] = "WAIT_VISUAL_SPAN"
            updates.append(row)
            continue

        lists = [by_q.get(int(eq), []) for eq in evidence]
        if any(len(x) == 0 for x in lists):
            row["action"] = "NO_VALID_SUBTILE_CANDIDATE"
            updates.append(row)
            continue

        row["enumerated_hypotheses"] = int(np.prod([len(x) for x in lists]))
        admissible = []
        for combo in itertools.product(*lists):
            h = make_hypothesis(helper, q, evidence, visual_xy, combo, policy, hid)
            hid += 1
            if h is not None:
                admissible.append(h)

        row["geometric_pass_hypotheses"] = len(admissible)
        if not admissible:
            row["action"] = "NO_GEOMETRICALLY_ADMISSIBLE_HYPOTHESIS"
            updates.append(row)
            continue

        front = pareto(admissible)
        row["pareto_hypotheses"] = len(front)
        leader = min(
            front,
            key=lambda h: (
                h["median_projected_residual_m"],
                h["sum_sift_rank"],
                h["sum_dino_rank"],
                tuple(h["tile_ids"]),
            ),
        )
        leaders[int(q)] = leader
        row["blind_leader_hypothesis_id"] = int(leader["hypothesis_id"])
        row["blind_leader_scale"] = float(leader["scale_m_per_visual_px"])
        row["blind_leader_rotation_deg"] = float(leader["rotation_deg"])
        row["action"] = "BLIND_LEADER_AVAILABLE"

        front_ids = {int(h["hypothesis_id"]) for h in front}
        for h in admissible:
            serial = dict(h)
            serial["pareto"] = int(h["hypothesis_id"]) in front_ids
            serial["blind_leader"] = int(h["hypothesis_id"]) == int(leader["hypothesis_id"])
            for key in ("evidence_query_ids", "tile_ids", "candidate_choice_ranks"):
                serial[key] = ",".join(map(str, serial[key]))
            serial["map_observations"] = ";".join(
                f"{x:.6f}:{y:.6f}" for x, y in h["map_observations"]
            )
            hypotheses.append(serial)
        updates.append(row)

    updates_df = pd.DataFrame(updates)
    seeds = [q for q in common if q in leaders]
    if not seeds:
        raise RuntimeError("No blind SIFT+LightGlue sub-tile leader produced")

    seed_q = int(seeds[0])
    active = leaders[seed_q]
    active_source_q = seed_q
    mode = "ACQUISITION"
    streak = 0
    matured_at = None
    maturity_model = None
    activation = float(policy["activation_threshold_m"])
    tracking = float(policy["tracking_threshold_m"])
    support_required = int(policy["maturity_support_required"])

    timeline = [
        {
            "update_query_id": seed_q,
            "mode_before": "ACQUISITION",
            "minimum_innovation_m": math.nan,
            "consistency_streak_before": 0,
            "consistency_streak_after": 0,
            "action": "SEED",
            "mode_after": "ACQUISITION",
            "matured_now": False,
            "matured_at_query_id": math.nan,
            "innovation_best_tile_id": "",
            "innovation_best_choice_rank": math.nan,
            "active_source_update_q_before": math.nan,
            "active_source_update_q_after": seed_q,
            "active_hypothesis_id_after": int(active["hypothesis_id"]),
        }
    ]

    for q in [x for x in common if x > seed_q]:
        mode_before = mode
        source_before = active_source_q
        streak_before = streak
        current_leader = leaders.get(int(q))
        current_visual = rel_by_q.loc[[q], ["visual_x_px", "visual_y_px"]].to_numpy(float)
        prior_prediction = helper.apply_similarity(current_visual, active)[0]

        current = observations[
            (observations["query_id"] == q)
            & observations["valid_projected_observation"]
        ].copy()
        if len(current):
            current["innovation_m"] = np.hypot(
                current["projected_easting"] - float(prior_prediction[0]),
                current["projected_northing"] - float(prior_prediction[1]),
            )
            best = current.sort_values(
                ["innovation_m", "candidate_choice_rank"], kind="mergesort"
            ).iloc[0]
            min_innovation = float(best["innovation_m"])
            best_tile = str(best["tile_id"])
            best_choice = int(best["candidate_choice_rank"])
        else:
            min_innovation = math.inf
            best_tile = ""
            best_choice = math.nan

        matured_now = False
        if mode == "ACQUISITION":
            if current_leader is None:
                action = "ACQUISITION_NO_LEADER"
                streak = 0
            else:
                active = current_leader
                active_source_q = int(q)
                streak = streak + 1 if min_innovation <= activation else 0
                action = "ACQUISITION_ACCEPT"
                if streak >= support_required:
                    mode = "TRACKING"
                    matured_at = int(q)
                    maturity_model = dict(active)
                    matured_now = True
        else:
            if current_leader is None:
                action = "TRACKING_HOLD_NO_LEADER"
            elif min_innovation <= tracking:
                action = "TRACKING_ACCEPT"
                active = current_leader
                active_source_q = int(q)
            else:
                action = "TRACKING_HOLD_INNOVATION"

        timeline.append(
            {
                "update_query_id": int(q),
                "mode_before": mode_before,
                "minimum_innovation_m": min_innovation,
                "consistency_streak_before": streak_before,
                "consistency_streak_after": streak,
                "action": action,
                "mode_after": mode,
                "matured_now": matured_now,
                "matured_at_query_id": matured_at if matured_at is not None else math.nan,
                "innovation_best_tile_id": best_tile,
                "innovation_best_choice_rank": best_choice,
                "active_source_update_q_before": source_before,
                "active_source_update_q_after": active_source_q,
                "active_hypothesis_id_after": int(active["hypothesis_id"]),
            }
        )

    localization_state = (
        "PROVISIONAL_ABSOLUTE_LOCK" if matured_at is not None else "NO_PROVISIONAL_LOCK"
    )
    result = {
        "localization_state": localization_state,
        "activation_threshold_m": activation,
        "tracking_threshold_m": tracking,
        "maturity_support_required": support_required,
        "matured_at_query_id": matured_at,
        "maturity_map_state": model_payload(maturity_model) if maturity_model else None,
        "final_active_source_update_q": int(active_source_q),
        "final_active_map_state": model_payload(active),
        "forbidden_state_emitted": False,
    }

    out = resolve(cfg["output_root"])
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "observations": out / "bv1_2_blind_top4_sift_observations.csv",
        "hypotheses": out / "bv1_2_blind_subtile_hypotheses.csv",
        "leader_updates": out / "bv1_2_blind_leader_updates.csv",
        "policy_timeline": out / "bv1_2_blind_policy_timeline.csv",
        "policy_result": out / "bv1_2_blind_policy_result.json",
    }

    observations.to_csv(paths["observations"], index=False)
    pd.DataFrame(hypotheses).to_csv(paths["hypotheses"], index=False)
    updates_df.to_csv(paths["leader_updates"], index=False)
    pd.DataFrame(timeline).to_csv(paths["policy_timeline"], index=False)
    paths["policy_result"].write_text(
        json.dumps(
            {
                "stage": "BV1.2",
                "policy": "activate_quarter_track_quarter",
                "result": result,
                "blind_contract": {
                    "gps_used": False,
                    "srt_used": False,
                    "reference_used": False,
                    "oracle_used": False,
                    "absolute_locked_emitted": False,
                },
            },
            indent=2,
        )
    )

    output_provenance = {
        name: {"path": str(path), "sha256": sha256(path)}
        for name, path in paths.items()
    }
    report_out = out / "bv1_2_report.json"
    report_out.write_text(
        json.dumps(
            {
                "stage": "BV1.2",
                "status": "PASS_BV1_2_BLIND_SIFT_LIGHTGLUE_MINIMUM_CONFIDENT_TRANSFER",
                "localization_state": localization_state,
                "matured_at_query_id": matured_at,
                "first_leader_query": seed_q,
                "counts": {
                    "queries": len(common),
                    "top4_candidate_rows": len(observations),
                    "valid_projected_observations": int(
                        observations["valid_projected_observation"].sum()
                    ),
                    "leader_updates": int(
                        updates_df["blind_leader_hypothesis_id"].notna().sum()
                    ),
                    "geometric_pass_hypotheses": len(hypotheses),
                },
                "policy": {
                    "candidate_count": int(policy["candidate_count"]),
                    "evidence_queries": evidence_frames,
                    "activation_threshold_m": activation,
                    "tracking_threshold_m": tracking,
                    "maturity_support_required": support_required,
                    "ranking": "SIFT inliers desc -> DINO rank asc -> tile_id asc",
                },
                "provenance": {
                    "config_sha256": sha256(cfg_path),
                    "bv1_0_report_sha256": sha256(bv_report_path),
                    "bv1_0_pair_evidence_sha256": sha256(pair_path),
                    "relative_csv_sha256": sha256(relative_path),
                    "manifest_csv_sha256": sha256(manifest_path),
                },
                "blind_contract": {
                    "gt_used": False,
                    "reference_used": False,
                    "oracle_used": False,
                    "orb_recomputed": False,
                    "fusion_executed": False,
                    "absolute_locked_emitted": False,
                },
                "outputs": output_provenance,
            },
            indent=2,
        )
    )

    print("Status: PASS_BV1_2_BLIND_SIFT_LIGHTGLUE_MINIMUM_CONFIDENT_TRANSFER")
    print("Localization state:", localization_state)
    print("Queries:", len(common))
    print("Top4 observation rows:", len(observations))
    print("Valid projected observations:", int(observations["valid_projected_observation"].sum()))
    print("Leader updates:", int(updates_df["blind_leader_hypothesis_id"].notna().sum()))
    print("First leader query:", seed_q)
    print("Matured at query:", matured_at)
    print("Final active source query:", active_source_q)
    print("GT/reference/SRT/GPS used: False")
    print("ORB recomputed: False")
    print("Fusion executed: False")
    print("Report:", report_out)


if __name__ == "__main__":
    main()
