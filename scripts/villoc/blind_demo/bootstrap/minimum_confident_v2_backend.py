#!/usr/bin/env python3
"""
R5.1 — clean blind R3-v2 implementation candidate.

Implements the frozen R5.0 architecture without GT/reference/oracle data:
  * DINO/ORB Top-4 candidates
  * exact R4.11 ORB query->tile homography contract
  * continuous sub-tile EPSG:3346 observations
  * 4-frame XFeat-relative similarity hypotheses
  * blind Pareto leader per causal update
  * ACQUISITION -> TRACKING state machine
  * all four still-unselected geometry-relative policy candidates

This script deliberately does NOT select a production policy.
Each policy may emit only PROVISIONAL_ABSOLUTE_LOCK or NO_PROVISIONAL_LOCK.
It never emits ABSOLUTE_LOCKED.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

FORBIDDEN_EXACT = {
    "reference_x_m", "reference_y_m", "latitude", "longitude", "lat", "lon",
    "ground_truth_x_m", "ground_truth_y_m", "oracle_tile_id",
    "oracle_best_tile_id", "chosen_error_m", "candidate_body_error_m",
    "reranked_top1_error_m", "gt_easting", "gt_northing", "projected_error_m",
    "tile_center_error_m",
}
FORBIDDEN_PART = (
    "hit_le_", "ground_truth", "oracle_", "gt_error", "reference_enu",
    "postfreeze", "eval_error",
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def assert_blind_safe(df: pd.DataFrame, label: str) -> None:
    bad = []
    for c in df.columns:
        low = str(c).lower()
        if low in FORBIDDEN_EXACT or any(x in low for x in FORBIDDEN_PART):
            bad.append(c)
    bad = [c for c in bad if c != "reference_used"]
    if bad:
        raise RuntimeError(f"{label} contains forbidden evaluation columns: {bad}")
    if "reference_used" in df:
        used = df["reference_used"].astype(str).str.lower().isin({"true", "1", "yes"})
        if used.any():
            raise RuntimeError(f"{label}: reference_used contains True rows.")


def dynamic_import(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def fit_similarity(visual_xy, map_xy):
    v = np.asarray(visual_xy, float)
    m = np.asarray(map_xy, float)
    z = v[:, 0] + 1j * v[:, 1]
    w = m[:, 0] + 1j * m[:, 1]
    z0 = z - z.mean()
    w0 = w - w.mean()
    denom = float(np.sum(np.abs(z0) ** 2))
    if denom <= 1e-12 or not np.isfinite(denom):
        raise ValueError("degenerate visual geometry")
    a = np.sum(w0 * np.conj(z0)) / denom
    b = w.mean() - a * z.mean()
    model = {
        "a_real": float(a.real),
        "a_imag": float(a.imag),
        "b_real": float(b.real),
        "b_imag": float(b.imag),
        "scale_m_per_visual_px": float(abs(a)),
        "rotation_deg": float(np.degrees(np.angle(a))),
    }
    if not all(np.isfinite(list(model.values()))) or model["scale_m_per_visual_px"] <= 0:
        raise ValueError("invalid similarity")
    return model


def apply_similarity(xy, model):
    p = np.asarray(xy, float)
    z = p[:, 0] + 1j * p[:, 1]
    a = complex(float(model["a_real"]), float(model["a_imag"]))
    b = complex(float(model["b_real"]), float(model["b_imag"]))
    w = a * z + b
    return np.column_stack([w.real, w.imag])


def span(xy):
    p = np.asarray(xy, float)
    if len(p) < 2:
        return 0.0
    d = p[:, None, :] - p[None, :, :]
    return float(np.sqrt(np.sum(d * d, axis=2)).max())


def select_evidence(prefix: pd.DataFrame, n: int):
    if len(prefix) < n:
        return []
    pts = prefix[["visual_x_px", "visual_y_px"]].to_numpy(float)
    qids = prefix["query_id"].astype(int).to_numpy()
    chosen = [len(prefix) - 1]
    left = set(range(len(prefix) - 1))
    while len(chosen) < n and left:
        idx = max(
            left,
            key=lambda i: (
                min(float(np.linalg.norm(pts[i] - pts[j])) for j in chosen),
                int(qids[i]),
            ),
        )
        chosen.append(idx)
        left.remove(idx)
    return sorted(int(qids[i]) for i in chosen) if len(chosen) == n else []


def pareto(rows):
    keys = ("median_projected_residual_m", "sum_hybrid_rank", "sum_dino_rank")
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


def resolve_path(value, repo: Path) -> Path:
    p = Path(str(value))
    if not p.is_absolute():
        p = repo / p
    return p.resolve()


def bool_value(value) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def jacobian_condition(H: np.ndarray, u: float, v: float):
    a, b, c = np.asarray(H, float)[0]
    d, e, f = np.asarray(H, float)[1]
    g, h, i = np.asarray(H, float)[2]
    den = g * u + h * v + i
    if not np.isfinite(den) or abs(den) <= 1e-12:
        return math.inf, math.nan
    nu = a * u + b * v + c
    nv = d * u + e * v + f
    den2 = den ** 2
    J = np.asarray([
        [(a * den - nu * g) / den2, (b * den - nu * h) / den2],
        [(d * den - nv * g) / den2, (e * den - nv * h) / den2],
    ], float)
    s = np.linalg.svd(J, compute_uv=False)
    smax, smin = float(s.max()), float(s.min())
    condition = float(smax / smin) if smin > 1e-12 else math.inf
    det = float(abs(np.linalg.det(J)))
    local_scale = float(math.sqrt(det)) if np.isfinite(det) else math.nan
    return condition, local_scale


def recompute_projection(q, s, row, r411):
    result = {
        "recomputed_good_matches": 0,
        "recomputed_inliers": 0,
        "recomputed_homography_ok": False,
        "projected_tile_u_px": math.nan,
        "projected_tile_v_px": math.nan,
        "projected_easting": math.nan,
        "projected_northing": math.nan,
        "projected_inside_tile": False,
        "jacobian_condition": math.nan,
        "jacobian_local_area_scale": math.nan,
    }
    if not q.ok or not s.ok or q.descriptors is None or s.descriptors is None:
        return result

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn = matcher.knnMatch(q.descriptors, s.descriptors, k=2)
    good = []
    for pair in knn:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < float(r411.LOWE_RATIO) * n.distance:
            good.append(m)
    result["recomputed_good_matches"] = int(len(good))
    if len(good) < 4:
        return result

    q_pts = np.float32([q.keypoints[m.queryIdx].pt for m in good])
    s_pts = np.float32([s.keypoints[m.trainIdx].pt for m in good])
    H, mask = cv2.findHomography(q_pts, s_pts, cv2.RANSAC, float(r411.RANSAC_THRESHOLD))
    if H is None or mask is None:
        return result
    inliers = int(mask.ravel().astype(bool).sum())
    result["recomputed_inliers"] = inliers
    if inliers < 4:
        return result

    qh, qw = q.image_shape
    sh, sw = s.image_shape
    center = np.float32([[[float(qw) / 2.0, float(qh) / 2.0]]])
    projected = cv2.perspectiveTransform(center, H).reshape(2).astype(float)
    u, v = float(projected[0]), float(projected[1])

    left = float(row["left_easting"])
    right = float(row["right_easting"])
    bottom = float(row["bottom_northing"])
    top = float(row["top_northing"])
    easting = left + ((u + 0.5) / float(sw)) * (right - left)
    northing = top - ((v + 0.5) / float(sh)) * (top - bottom)
    cond, local_scale = jacobian_condition(H, float(qw) / 2.0, float(qh) / 2.0)

    result.update({
        "recomputed_homography_ok": True,
        "projected_tile_u_px": u,
        "projected_tile_v_px": v,
        "projected_easting": float(easting),
        "projected_northing": float(northing),
        "projected_inside_tile": bool(0.0 <= u < float(sw) and 0.0 <= v < float(sh)),
        "jacobian_condition": cond,
        "jacobian_local_area_scale": local_scale,
    })
    return result


def make_hypothesis(update_qid, evidence_qids, visual_xy, combo, cfg, hypothesis_id):
    map_xy = np.asarray(
        [[float(c["projected_easting"]), float(c["projected_northing"])] for c in combo],
        float,
    )
    if not np.all(np.isfinite(map_xy)):
        return None
    tile_ids = [str(c["tile_id"]) for c in combo]
    if len(set(tile_ids)) < int(cfg["min_unique_tile_ids"]):
        return None
    map_span = span(map_xy)
    if map_span < float(cfg["min_map_span_m"]):
        return None
    try:
        model = fit_similarity(visual_xy, map_xy)
    except ValueError:
        return None
    residual = np.linalg.norm(apply_similarity(visual_xy, model) - map_xy, axis=1)
    med = float(np.median(residual))
    mx = float(np.max(residual))
    if med > float(cfg["median_limit_m"]) or mx > float(cfg["max_limit_m"]):
        return None
    return {
        "hypothesis_id": int(hypothesis_id),
        "update_query_id": int(update_qid),
        "evidence_query_ids": [int(x) for x in evidence_qids],
        "tile_ids": tile_ids,
        "candidate_choice_ranks": [int(c["candidate_choice_rank"]) for c in combo],
        "map_observations": [(float(x), float(y)) for x, y in map_xy],
        "map_unique_tile_ids": int(len(set(tile_ids))),
        "map_span_m": map_span,
        "median_projected_residual_m": med,
        "max_projected_residual_m": mx,
        "sum_dino_rank": float(sum(float(c["rank"]) for c in combo)),
        "sum_hybrid_rank": float(sum(float(c["hybrid_rank"]) for c in combo)),
        "all_projected_inside_tile": bool(all(bool(c["projected_inside_tile"]) for c in combo)),
        "mean_jacobian_condition": float(np.nanmean([float(c["jacobian_condition"]) for c in combo])),
        **model,
    }


def serial_hypothesis(h, is_pareto, is_leader):
    r = dict(h)
    r["pareto"] = bool(is_pareto)
    r["blind_leader"] = bool(is_leader)
    for k in ("evidence_query_ids", "tile_ids", "candidate_choice_ranks"):
        r[k] = ",".join(map(str, r[k]))
    r["map_observations"] = ";".join(f"{x:.6f}:{y:.6f}" for x, y in h["map_observations"])
    return r


def model_payload(h):
    return {
        k: float(h[k])
        for k in (
            "a_real", "a_imag", "b_real", "b_imag",
            "scale_m_per_visual_px", "rotation_deg",
        )
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", type=Path, required=True)
    p.add_argument("--run-root", type=Path, required=True)
    p.add_argument("--architecture-contract", type=Path, required=True)
    p.add_argument("--expected-contract-sha256", required=True)
    p.add_argument("--out-root", type=Path, required=True)
    p.add_argument("--candidate-csv", type=Path)
    p.add_argument("--relative-csv", type=Path)
    p.add_argument("--manifest-csv", type=Path)
    p.add_argument("--max-query-id", type=int, default=0)
    p.add_argument("--r4-blind-equivalence-csv", type=Path)
    args = p.parse_args()

    repo = args.repo_root.resolve()
    run = args.run_root.resolve()
    contract_path = args.architecture_contract.resolve()
    out = args.out_root.resolve()
    out.mkdir(parents=True, exist_ok=True)

    actual_contract_sha = sha256(contract_path)
    if actual_contract_sha != str(args.expected_contract_sha256):
        raise RuntimeError(
            "R5.0 architecture contract hash mismatch:\n"
            f"expected: {args.expected_contract_sha256}\nactual:   {actual_contract_sha}"
        )
    contract = json.loads(contract_path.read_text())
    if contract.get("status") != "PASS_R5_0_R3V2_ARCHITECTURE_CONTRACT_FREEZE":
        raise RuntimeError("Architecture contract status is not PASS.")
    if bool(contract["candidate_policy_family"].get("final_policy_selected", True)):
        raise RuntimeError("R5.0 contract unexpectedly selects a final policy.")

    top_m = int(contract["absolute_observation"]["candidate_count"])
    evidence_frames = int(contract["bootstrap_hypothesis"]["evidence_queries"])
    if top_m != 4 or evidence_frames != 4:
        raise RuntimeError("R5.1 expects the frozen Top-4 / 4-evidence contract.")
    cfg = {
        "min_unique_tile_ids": int(contract["bootstrap_hypothesis"]["minimum_unique_tile_ids"]),
        "min_visual_span_px": float(contract["bootstrap_hypothesis"]["minimum_visual_span_px"]),
        "min_map_span_m": float(contract["bootstrap_hypothesis"]["minimum_map_span_m"]),
        "median_limit_m": float(contract["bootstrap_hypothesis"]["median_measurement_residual_limit_m"]),
        "max_limit_m": float(contract["bootstrap_hypothesis"]["maximum_measurement_residual_limit_m"]),
    }
    map_spacing = float(contract["candidate_policy_family"]["map_spacing_m"])
    support_required = int(contract["candidate_policy_family"]["maturity_support_required"])
    activation_thresholds = {
        str(k): float(v) for k, v in contract["candidate_policy_family"]["activation_thresholds"].items()
    }
    tracking_thresholds = {
        str(k): float(v) for k, v in contract["candidate_policy_family"]["tracking_thresholds"].items()
    }
    policies = {
        f"activate_{a.replace('_spacing','')}_track_{t.replace('_spacing','')}": {
            "activation_threshold_m": av,
            "tracking_threshold_m": tv,
        }
        for a, av in activation_thresholds.items()
        for t, tv in tracking_thresholds.items()
    }

    candidate_csv = (args.candidate_csv or (
        run / "reports/s8_12e1_top20_verifier_reranker/512_s256_orb_hybrid_top20_img518/"
              "s8_12e1_all_candidate_verifier_scores.csv"
    )).resolve()
    relative_csv = (args.relative_csv or (
        run / "metadata/s8_xfeat_relative_frontend/s8r4_xfeat_relative_trajectory_blind_raw.csv"
    )).resolve()
    manifest_csv = (args.manifest_csv or (run / "metadata/blind_query_manifest.csv")).resolve()
    r411_source = (
        repo / "scripts/villoc/research/minimum_confident_bootstrap/diagnostics/"
               "r4_11_blind_subtile_projection_recompute.py"
    ).resolve()
    for path in (candidate_csv, relative_csv, manifest_csv, r411_source):
        if not path.exists():
            raise RuntimeError(f"Missing R5.1 input: {path}")

    cand = pd.read_csv(candidate_csv)
    rel = pd.read_csv(relative_csv)
    manifest = pd.read_csv(manifest_csv)
    assert_blind_safe(cand, "candidate table")
    assert_blind_safe(rel, "relative trajectory")
    assert_blind_safe(manifest, "blind manifest")

    required_c = {
        "query_id", "rank", "tile_id", "hybrid_rank", "good_matches", "inliers", "homography_ok",
        "left_easting", "right_easting", "bottom_northing", "top_northing",
        "query_image_resolved", "tile_image_resolved",
    }
    required_r = {"token0_id", "visual_x_px", "visual_y_px"}
    required_m = {"query_id", "timestamp_s"}
    if required_c - set(cand):
        raise RuntimeError(f"candidate table missing: {sorted(required_c - set(cand))}")
    if required_r - set(rel):
        raise RuntimeError(f"relative trajectory missing: {sorted(required_r - set(rel))}")
    if required_m - set(manifest):
        raise RuntimeError(f"manifest missing: {sorted(required_m - set(manifest))}")

    cand["query_id"] = pd.to_numeric(cand["query_id"], errors="raise").astype(int)
    for c in ("rank", "hybrid_rank", "good_matches", "inliers", "left_easting", "right_easting", "bottom_northing", "top_northing"):
        cand[c] = pd.to_numeric(cand[c], errors="coerce")
    if "verifier_rank" not in cand:
        cand["verifier_rank"] = np.nan
    cand["verifier_rank"] = pd.to_numeric(cand["verifier_rank"], errors="coerce")
    cand = cand.sort_values(
        ["query_id", "hybrid_rank", "verifier_rank", "rank", "tile_id"], kind="mergesort"
    )
    cand["candidate_choice_rank"] = cand.groupby("query_id").cumcount() + 1
    cand = cand[cand["candidate_choice_rank"] <= top_m].copy()

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
    if args.max_query_id:
        common = [q for q in common if q <= int(args.max_query_id)]
    if not common:
        raise RuntimeError("No common blind query IDs.")
    cand = cand[cand["query_id"].isin(common)].copy()
    rel = rel[rel["query_id"].isin(common)].sort_values("query_id").reset_index(drop=True)

    short = cand.groupby("query_id").size()
    short = short[short < top_m]
    if len(short):
        raise RuntimeError(f"queries with fewer than Top-{top_m}: {short.head().to_dict()}")

    r411 = dynamic_import("r411_frozen_orb_contract", r411_source)
    detector = r411.create_detector()
    feature_cache = {}

    def features(path_value):
        pth = resolve_path(path_value, repo)
        key = str(pth)
        if key not in feature_cache:
            feature_cache[key] = r411.compute_features(pth, detector)
        return feature_cache[key]

    observation_rows = []
    mismatch_good = mismatch_inliers = mismatch_h = 0
    for idx, (_, row) in enumerate(cand.iterrows(), 1):
        qf = features(row["query_image_resolved"])
        sf = features(row["tile_image_resolved"])
        proj = recompute_projection(qf, sf, row, r411)
        mismatch_good += int(int(row["good_matches"]) != int(proj["recomputed_good_matches"]))
        mismatch_inliers += int(int(row["inliers"]) != int(proj["recomputed_inliers"]))
        mismatch_h += int(bool_value(row["homography_ok"]) != bool(proj["recomputed_homography_ok"]))
        observation_rows.append({
            "query_id": int(row["query_id"]),
            "timestamp_s": float(rel.loc[rel["query_id"] == int(row["query_id"]), "timestamp_s"].iloc[0])
                if pd.notna(rel.loc[rel["query_id"] == int(row["query_id"]), "timestamp_s"].iloc[0]) else math.nan,
            "tile_id": str(row["tile_id"]),
            "candidate_choice_rank": int(row["candidate_choice_rank"]),
            "rank": int(row["rank"]),
            "hybrid_rank": int(row["hybrid_rank"]),
            "stored_good_matches": int(row["good_matches"]),
            "stored_inliers": int(row["inliers"]),
            "stored_homography_ok": bool_value(row["homography_ok"]),
            **proj,
        })
        if idx % 50 == 0 or idx == len(cand):
            print("ORB projection pairs:", idx, "/", len(cand))

    observations = pd.DataFrame(observation_rows)
    reproduction_pass = bool(mismatch_good == 0 and mismatch_inliers == 0 and mismatch_h == 0)
    if not reproduction_pass:
        raise RuntimeError(
            "R5.1 ORB reproduction gate failed: "
            f"good={mismatch_good}, inliers={mismatch_inliers}, homography={mismatch_h}"
        )

    by_q = {
        int(q): g[g["recomputed_homography_ok"].astype(bool)].to_dict("records")
        for q, g in observations.groupby("query_id", sort=True)
    }
    rel_by_q = rel.set_index("query_id")

    update_rows = []
    hypothesis_rows = []
    leaders = {}
    next_hypothesis_id = 1

    for q in common:
        prefix = rel[rel["query_id"] <= q]
        ts_v = prefix.loc[prefix["query_id"] == q, "timestamp_s"]
        ts = float(ts_v.iloc[0]) if len(ts_v) and pd.notna(ts_v.iloc[0]) else math.nan
        row = {
            "update_query_id": int(q),
            "timestamp_s": ts,
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
        evidence = select_evidence(prefix, evidence_frames)
        if len(evidence) != evidence_frames:
            row["action"] = "WAIT_INSUFFICIENT_EVIDENCE_FRAMES"
            update_rows.append(row)
            continue
        visual_xy = rel_by_q.loc[evidence, ["visual_x_px", "visual_y_px"]].to_numpy(float)
        visual_span = span(visual_xy)
        row["evidence_query_ids"] = ",".join(map(str, evidence))
        row["visual_span_px"] = visual_span
        if visual_span < cfg["min_visual_span_px"]:
            row["action"] = "WAIT_VISUAL_SPAN"
            update_rows.append(row)
            continue
        lists = [by_q.get(int(eq), []) for eq in evidence]
        if any(len(x) == 0 for x in lists):
            row["action"] = "NO_VALID_SUBTILE_CANDIDATE"
            update_rows.append(row)
            continue
        row["enumerated_hypotheses"] = int(np.prod([len(x) for x in lists]))
        admissible = []
        for combo in itertools.product(*lists):
            hid = next_hypothesis_id
            next_hypothesis_id += 1
            h = make_hypothesis(q, evidence, visual_xy, combo, cfg, hid)
            if h is not None:
                admissible.append(h)
        row["geometric_pass_hypotheses"] = len(admissible)
        if not admissible:
            row["action"] = "NO_GEOMETRICALLY_ADMISSIBLE_HYPOTHESIS"
            update_rows.append(row)
            continue
        front = pareto(admissible)
        row["pareto_hypotheses"] = len(front)
        leader = min(
            front,
            key=lambda h: (
                h["median_projected_residual_m"],
                h["sum_hybrid_rank"],
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
            hypothesis_rows.append(
                serial_hypothesis(
                    h,
                    int(h["hypothesis_id"]) in front_ids,
                    int(h["hypothesis_id"]) == int(leader["hypothesis_id"]),
                )
            )
        update_rows.append(row)

    updates = pd.DataFrame(update_rows)
    valid_seed_q = [q for q in common if q in leaders]
    if not valid_seed_q:
        raise RuntimeError("No blind sub-tile leader was produced.")
    seed_q = int(valid_seed_q[0])
    seed_leader = leaders[seed_q]

    policy_rows = []
    policy_results = {}
    for policy_name, policy in sorted(policies.items()):
        mode = "ACQUISITION"
        active = seed_leader
        active_source_q = seed_q
        streak = 0
        matured_at = None
        maturity_model = None

        policy_rows.append({
            "policy": policy_name,
            "activation_threshold_m": float(policy["activation_threshold_m"]),
            "tracking_threshold_m": float(policy["tracking_threshold_m"]),
            "support_required": support_required,
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
        })

        for q in [x for x in common if x > seed_q]:
            mode_before = mode
            source_before = active_source_q
            streak_before = streak
            current_leader = leaders.get(int(q))
            current_visual = rel_by_q.loc[[q], ["visual_x_px", "visual_y_px"]].to_numpy(float)
            prior_prediction = apply_similarity(current_visual, active)[0]
            current_candidates = observations[
                (observations["query_id"] == q) & observations["recomputed_homography_ok"].astype(bool)
            ].copy()
            if len(current_candidates):
                current_candidates["innovation_m"] = np.hypot(
                    current_candidates["projected_easting"] - float(prior_prediction[0]),
                    current_candidates["projected_northing"] - float(prior_prediction[1]),
                )
                innovation_best = current_candidates.sort_values(
                    ["innovation_m", "candidate_choice_rank"], kind="mergesort"
                ).iloc[0]
                min_innovation = float(innovation_best["innovation_m"])
                best_tile = str(innovation_best["tile_id"])
                best_choice = int(innovation_best["candidate_choice_rank"])
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
                    if min_innovation <= float(policy["activation_threshold_m"]):
                        streak += 1
                    else:
                        streak = 0
                    action = "ACQUISITION_ACCEPT"
                    if streak >= support_required:
                        mode = "TRACKING"
                        matured_at = int(q)
                        maturity_model = dict(active)
                        matured_now = True
            else:
                if current_leader is None:
                    action = "TRACKING_HOLD_NO_LEADER"
                elif min_innovation <= float(policy["tracking_threshold_m"]):
                    action = "TRACKING_ACCEPT"
                    active = current_leader
                    active_source_q = int(q)
                else:
                    action = "TRACKING_HOLD_INNOVATION"

            policy_rows.append({
                "policy": policy_name,
                "activation_threshold_m": float(policy["activation_threshold_m"]),
                "tracking_threshold_m": float(policy["tracking_threshold_m"]),
                "support_required": support_required,
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
            })

        localization_state = "PROVISIONAL_ABSOLUTE_LOCK" if matured_at is not None else "NO_PROVISIONAL_LOCK"
        policy_results[policy_name] = {
            "localization_state": localization_state,
            "activation_threshold_m": float(policy["activation_threshold_m"]),
            "tracking_threshold_m": float(policy["tracking_threshold_m"]),
            "maturity_support_required": support_required,
            "matured_at_query_id": matured_at,
            "maturity_map_state": model_payload(maturity_model) if maturity_model is not None else None,
            "final_active_source_update_q": int(active_source_q),
            "final_active_map_state": model_payload(active),
            "forbidden_state_emitted": False,
        }

    policy_timeline = pd.DataFrame(policy_rows)

    observation_path = out / "r5_1_blind_top4_subtile_observations.csv"
    hypothesis_path = out / "r5_1_blind_subtile_hypotheses.csv"
    update_path = out / "r5_1_blind_leader_updates.csv"
    policy_timeline_path = out / "r5_1_blind_policy_timeline.csv"
    policy_results_path = out / "r5_1_blind_policy_results.json"
    freeze_path = out / "r5_1_blind_implementation_freeze_manifest.json"

    observations.to_csv(observation_path, index=False)
    pd.DataFrame(hypothesis_rows).to_csv(hypothesis_path, index=False)
    updates.to_csv(update_path, index=False)
    policy_timeline.to_csv(policy_timeline_path, index=False)
    policy_results_path.write_text(json.dumps({
        "stage": "R5.1_minimum_confident_bootstrap_v2",
        "policy_family_selected": False,
        "policy_results": policy_results,
        "blind_contract": {
            "gps_used": False,
            "srt_used": False,
            "reference_used": False,
            "oracle_used": False,
            "evaluation_error_used": False,
            "absolute_locked_emitted": False,
        },
    }, indent=2))

    freeze = {
        "stage": "R5.1_BLIND_IMPLEMENTATION_FREEZE",
        "architecture_contract_sha256": actual_contract_sha,
        "policy_family_selected": False,
        "configuration": {
            "top_m": top_m,
            "evidence_frames": evidence_frames,
            "map_spacing_m": map_spacing,
            "maturity_support_required": support_required,
            "policies": policies,
            **cfg,
            "max_query_id": int(args.max_query_id),
        },
        "orb_reproduction_gate": {
            "pass": reproduction_pass,
            "good_match_mismatches": mismatch_good,
            "inlier_mismatches": mismatch_inliers,
            "homography_ok_mismatches": mismatch_h,
        },
        "inputs": {
            "candidate_csv": str(candidate_csv),
            "candidate_sha256": sha256(candidate_csv),
            "relative_csv": str(relative_csv),
            "relative_sha256": sha256(relative_csv),
            "manifest_csv": str(manifest_csv),
            "manifest_sha256": sha256(manifest_csv),
            "r4_11_source": str(r411_source),
            "r4_11_source_sha256": sha256(r411_source),
        },
        "counts": {
            "queries": len(common),
            "top4_candidate_rows": len(cand),
            "valid_projected_observations": int(observations["recomputed_homography_ok"].astype(bool).sum()),
            "leader_updates": int(updates["blind_leader_hypothesis_id"].notna().sum()),
            "geometric_pass_hypotheses": len(hypothesis_rows),
        },
        "blind_contract": {
            "gt_used": False,
            "reference_used": False,
            "oracle_used": False,
            "policy_selected": False,
            "absolute_locked_emitted": False,
        },
        "outputs": {},
    }
    for name, path in {
        "observations": observation_path,
        "hypotheses": hypothesis_path,
        "leader_updates": update_path,
        "policy_timeline": policy_timeline_path,
        "policy_results": policy_results_path,
    }.items():
        freeze["outputs"][name] = {"path": str(path), "sha256": sha256(path)}
    freeze_path.write_text(json.dumps(freeze, indent=2))
    freeze_sha = sha256(freeze_path)

    print()
    print("=" * 118)
    print("R5.1 PHASE A — CLEAN BLIND R3-v2 IMPLEMENTATION FROZEN")
    print("=" * 118)
    print("queries:", len(common))
    print("Top-4 observation rows:", len(observations))
    print("valid projected observations:", int(observations["recomputed_homography_ok"].astype(bool).sum()))
    print("ORB reproduction: PASS")
    print("leader updates:", int(updates["blind_leader_hypothesis_id"].notna().sum()))
    print("first leader query:", seed_q)
    print("final policy selected:", False)
    print("GT/reference used:", False)
    print("blind freeze SHA256:", freeze_sha)
    print()
    for name in sorted(policy_results):
        r = policy_results[name]
        print(
            name,
            "state=", r["localization_state"],
            "matured_at=", r["matured_at_query_id"],
            "final_source_q=", r["final_active_source_update_q"],
        )

    equivalence_report = None
    overall_status = "PASS_R5_1_BLIND_IMPLEMENTATION_FREEZE"
    if args.r4_blind_equivalence_csv is not None:
        eq_path = args.r4_blind_equivalence_csv.resolve()
        expected = pd.read_csv(eq_path)
        assert_blind_safe(expected, "R4.18 blind equivalence timeline")
        expected["update_query_id"] = pd.to_numeric(expected["update_query_id"], errors="raise").astype(int)
        equivalence_rows = []
        all_pass = True
        for policy_name in sorted(policies):
            exp = expected[expected["policy"] == policy_name].copy()
            got = policy_timeline[policy_timeline["policy"] == policy_name].copy()
            merged = exp.merge(
                got,
                on="update_query_id",
                suffixes=("_expected", "_got"),
                how="inner",
                validate="one_to_one",
            )
            if len(merged) != len(exp) or len(merged) != len(got):
                policy_pass = False
                action_mismatch = mode_mismatch = source_mismatch = -1
                max_innovation_diff = math.inf
            else:
                action_mismatch = int((merged["action_expected"] != merged["action_got"]).sum())
                mode_mismatch = int((merged["mode_after_expected"] != merged["mode_after_got"]).sum())
                source_mismatch = int(
                    (
                        pd.to_numeric(merged["active_source_update_q_after_expected"], errors="coerce")
                        != pd.to_numeric(merged["active_source_update_q_after_got"], errors="coerce")
                    ).sum()
                )
                e = pd.to_numeric(merged["minimum_innovation_m_expected"], errors="coerce")
                g = pd.to_numeric(merged["minimum_innovation_m_got"], errors="coerce")
                finite = e.notna() & g.notna()
                max_innovation_diff = float(np.max(np.abs(e[finite] - g[finite]))) if finite.any() else 0.0
                policy_pass = bool(
                    action_mismatch == 0
                    and mode_mismatch == 0
                    and source_mismatch == 0
                    and max_innovation_diff <= 1e-6
                )
            all_pass = all_pass and policy_pass
            equivalence_rows.append({
                "policy": policy_name,
                "pass": policy_pass,
                "expected_rows": len(exp),
                "got_rows": len(got),
                "action_mismatches": action_mismatch,
                "mode_mismatches": mode_mismatch,
                "active_source_mismatches": source_mismatch,
                "max_innovation_abs_diff_m": max_innovation_diff,
            })
        equivalence_report = {
            "stage": "R5.1_BLIND_EQUIVALENCE_AUDIT",
            "reference_is_blind_frozen_r4_18": True,
            "gt_used": False,
            "reference_used": False,
            "all_policies_pass": all_pass,
            "rows": equivalence_rows,
        }
        eq_out = out / "r5_1_blind_equivalence_report.json"
        eq_out.write_text(json.dumps(equivalence_report, indent=2))
        print()
        print("=" * 118)
        print("R5.1 PHASE B — BLIND IMPLEMENTATION EQUIVALENCE AGAINST FROZEN R4.18")
        print("=" * 118)
        print(pd.DataFrame(equivalence_rows).to_string(index=False))
        print("all policies equivalent:", all_pass)
        if all_pass:
            overall_status = "PASS_R5_1_BLIND_IMPLEMENTATION_EQUIVALENCE"
        else:
            overall_status = "PARTIAL_R5_1_BLIND_IMPLEMENTATION_EQUIVALENCE_MISMATCH"

    report_path = out / "r5_1_minimum_confident_bootstrap_v2_report.json"
    report_path.write_text(json.dumps({
        "stage": "R5.1_MINIMUM_CONFIDENT_BOOTSTRAP_V2",
        "status": overall_status,
        "architecture_contract_sha256": actual_contract_sha,
        "blind_freeze_manifest_sha256": freeze_sha,
        "policy_family_selected": False,
        "policy_results": policy_results,
        "equivalence": equivalence_report,
        "blind_contract": {
            "gt_used": False,
            "reference_used": False,
            "oracle_used": False,
            "absolute_locked_emitted": False,
        },
    }, indent=2))

    print()
    print("STATUS:", overall_status)
    print("report:", report_path)


if __name__ == "__main__":
    main()
