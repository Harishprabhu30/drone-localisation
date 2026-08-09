#!/usr/bin/env python3
"""
R3 — minimum-confident multi-candidate provisional bootstrap.

Research-only, blind-safe bootstrap. It reads:
  * ORB/DINO all-candidate verifier scores
  * XFeat blind raw relative trajectory
  * blind query manifest timestamps

It does NOT read reference/GNSS/SRT/oracle/evaluation data and never emits
ABSOLUTE_LOCKED. Output state is either:
  PROVISIONAL_ABSOLUTE_LOCK
  NO_PROVISIONAL_LOCK
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


FORBIDDEN_EXACT = {
    "reference_x_m", "reference_y_m", "latitude", "longitude", "lat", "lon",
    "ground_truth_x_m", "ground_truth_y_m", "oracle_tile_id",
    "oracle_best_tile_id", "chosen_error_m", "candidate_body_error_m",
    "reranked_top1_error_m",
}
FORBIDDEN_PART = ("hit_le_", "ground_truth", "oracle_", "gt_error", "reference_enu")


def assert_blind_safe(df, label):
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


def fit_similarity(visual_xy, map_xy):
    """Least-squares 2D similarity: w = a*z + b."""
    v = np.asarray(visual_xy, float)
    m = np.asarray(map_xy, float)
    z = v[:, 0] + 1j * v[:, 1]
    w = m[:, 0] + 1j * m[:, 1]
    z0, w0 = z - z.mean(), w - w.mean()
    denom = float(np.sum(np.abs(z0) ** 2))
    if denom <= 1e-12 or not np.isfinite(denom):
        raise ValueError("degenerate visual geometry")
    a = np.sum(w0 * np.conj(z0)) / denom
    b = w.mean() - a * z.mean()
    model = {
        "a_real": float(a.real), "a_imag": float(a.imag),
        "b_real": float(b.real), "b_imag": float(b.imag),
        "scale_m_per_visual_px": float(abs(a)),
        "rotation_deg": float(np.degrees(np.angle(a))),
    }
    if not all(np.isfinite(list(model.values()))) or model["scale_m_per_visual_px"] <= 0:
        raise ValueError("invalid similarity")
    return model


def apply_similarity(xy, model):
    p = np.asarray(xy, float)
    z = p[:, 0] + 1j * p[:, 1]
    a = complex(model["a_real"], model["a_imag"])
    b = complex(model["b_real"], model["b_imag"])
    w = a * z + b
    return np.column_stack([w.real, w.imag])


def span(xy):
    p = np.asarray(xy, float)
    if len(p) < 2:
        return 0.0
    d = p[:, None, :] - p[None, :, :]
    return float(np.sqrt(np.sum(d * d, axis=2)).max())


def farthest_pair(xy):
    p = np.asarray(xy, float)
    d = p[:, None, :] - p[None, :, :]
    d2 = np.sum(d * d, axis=2)
    return tuple(map(int, np.unravel_index(np.argmax(d2), d2.shape)))


def select_evidence(prefix, n):
    """
    Causal farthest-point sampling.
    Current query is always included; earlier queries are added by maximin distance.
    """
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
    keys = ("median_center_residual_m", "sum_hybrid_rank", "sum_dino_rank")
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


def disagreement(model_a, model_b, sentinel_xy):
    a = apply_similarity(sentinel_xy, model_a)
    b = apply_similarity(sentinel_xy, model_b)
    d = np.linalg.norm(a - b, axis=1)
    return float(d[0]), float(d[1])


def cluster_models(rows, sentinel_xy, threshold_m):
    """Representative-based transform clustering in map-prediction space."""
    rows = sorted(
        rows,
        key=lambda r: (
            r["median_center_residual_m"],
            r["sum_hybrid_rank"],
            r["sum_dino_rank"],
            tuple(r["tile_ids"]),
        ),
    )
    clusters = []
    for h in rows:
        placed = False
        for c in clusters:
            d0, d1 = disagreement(h, c["rep"], sentinel_xy)
            if d0 <= threshold_m and d1 <= threshold_m:
                c["members"].append(h)
                c["max_rep_disagreement_m"] = max(c["max_rep_disagreement_m"], d0, d1)
                placed = True
                break
        if not placed:
            clusters.append({"rep": h, "members": [h], "max_rep_disagreement_m": 0.0})
    return clusters


def load_inputs(candidate_csv, relative_csv, manifest_csv, top_m):
    cand = pd.read_csv(candidate_csv)
    rel = pd.read_csv(relative_csv)
    manifest = pd.read_csv(manifest_csv)

    assert_blind_safe(cand, "candidate table")
    assert_blind_safe(rel, "relative trajectory")

    need_c = {
        "query_id", "rank", "tile_id", "score", "center_easting", "center_northing",
        "left_easting", "right_easting", "bottom_northing", "top_northing",
        "good_matches", "inliers", "inlier_ratio", "query_inlier_coverage",
        "sat_inlier_coverage", "homography_ok", "verifier_score", "hybrid_score",
        "hybrid_rank",
    }
    need_r = {"token0_id", "visual_x_px", "visual_y_px"}
    need_m = {"query_id", "timestamp_s"}
    if need_c - set(cand):
        raise RuntimeError(f"candidate table missing: {sorted(need_c - set(cand))}")
    if need_r - set(rel):
        raise RuntimeError(f"relative trajectory missing: {sorted(need_r - set(rel))}")
    if need_m - set(manifest):
        raise RuntimeError(f"manifest missing: {sorted(need_m - set(manifest))}")

    cand["query_id"] = pd.to_numeric(cand["query_id"], errors="raise").astype(int)
    for c in [
        "rank", "score", "center_easting", "center_northing", "left_easting",
        "right_easting", "bottom_northing", "top_northing", "good_matches",
        "inliers", "inlier_ratio", "query_inlier_coverage", "sat_inlier_coverage",
        "verifier_score", "hybrid_score", "hybrid_rank",
    ]:
        cand[c] = pd.to_numeric(cand[c], errors="coerce")
    if "verifier_rank" not in cand:
        cand["verifier_rank"] = np.nan
    cand["verifier_rank"] = pd.to_numeric(cand["verifier_rank"], errors="coerce")

    cand = cand.sort_values(
        ["query_id", "hybrid_rank", "verifier_rank", "rank", "tile_id"],
        kind="mergesort",
    )
    cand["candidate_choice_rank"] = cand.groupby("query_id").cumcount() + 1
    cand = cand[cand["candidate_choice_rank"] <= top_m].copy()
    short = cand.groupby("query_id").size()
    short = short[short < top_m]
    if len(short):
        raise RuntimeError(f"queries with fewer than Top-{top_m}: {short.head().to_dict()}")

    rel["query_id"] = pd.to_numeric(rel["token0_id"], errors="raise").astype(int)
    rel["visual_x_px"] = pd.to_numeric(rel["visual_x_px"], errors="raise")
    rel["visual_y_px"] = pd.to_numeric(rel["visual_y_px"], errors="raise")
    manifest["query_id"] = pd.to_numeric(manifest["query_id"], errors="raise").astype(int)
    manifest["timestamp_s"] = pd.to_numeric(manifest["timestamp_s"], errors="coerce")
    rel = rel.merge(
        manifest[["query_id", "timestamp_s"]].drop_duplicates("query_id"),
        on="query_id", how="left", validate="one_to_one",
    )
    return cand, rel


def make_hypothesis(update_qid, evidence_qids, visual_xy, combo, cfg):
    map_xy = np.array(
        [[float(c["center_easting"]), float(c["center_northing"])] for c in combo]
    )
    centers = [(round(float(x), 3), round(float(y), 3)) for x, y in map_xy]
    if len(set(centers)) < cfg["min_unique_centers"]:
        return None
    map_span = span(map_xy)
    if map_span < cfg["min_map_span_m"]:
        return None
    try:
        model = fit_similarity(visual_xy, map_xy)
    except ValueError:
        return None

    residual = np.linalg.norm(apply_similarity(visual_xy, model) - map_xy, axis=1)
    med, mx = float(np.median(residual)), float(np.max(residual))
    if med > cfg["median_limit_m"] or mx > cfg["max_limit_m"]:
        return None

    dino_ranks = [float(c["rank"]) for c in combo]
    hybrid_ranks = [float(c["hybrid_rank"]) for c in combo]
    return {
        "update_query_id": int(update_qid),
        "evidence_query_ids": [int(x) for x in evidence_qids],
        "tile_ids": [str(c["tile_id"]) for c in combo],
        "candidate_choice_ranks": [int(c["candidate_choice_rank"]) for c in combo],
        "map_centers": centers,
        "map_unique_centers": len(set(centers)),
        "map_span_m": map_span,
        "median_center_residual_m": med,
        "max_center_residual_m": mx,
        "sum_dino_rank": float(sum(dino_ranks)),
        "sum_hybrid_rank": float(sum(hybrid_ranks)),
        "dino_ranks": dino_ranks,
        "hybrid_ranks": hybrid_ranks,
        "homography_ok_count": int(sum(str(c["homography_ok"]).lower() in {"true", "1"} for c in combo)),
        "sum_inliers": float(np.nansum([c["inliers"] for c in combo])),
        "mean_inlier_ratio": float(np.nanmean([c["inlier_ratio"] for c in combo])),
        "mean_query_coverage": float(np.nanmean([c["query_inlier_coverage"] for c in combo])),
        "mean_dino_score": float(np.nanmean([c["score"] for c in combo])),
        "mean_hybrid_score": float(np.nanmean([c["hybrid_score"] for c in combo])),
        **model,
    }


def serial_hypothesis(h, is_pareto):
    r = dict(h)
    r["pareto"] = bool(is_pareto)
    for k in ("evidence_query_ids", "tile_ids", "candidate_choice_ranks", "dino_ranks", "hybrid_ranks"):
        r[k] = ",".join(map(str, r[k]))
    r["map_centers"] = ";".join(f"{x:.3f}:{y:.3f}" for x, y in h["map_centers"])
    return r


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-root", type=Path, required=True)
    p.add_argument("--candidate-csv", type=Path)
    p.add_argument("--relative-csv", type=Path)
    p.add_argument("--manifest-csv", type=Path)
    p.add_argument("--out-root", type=Path, required=True)

    p.add_argument("--top-m", type=int, default=4)
    p.add_argument("--evidence-frames", type=int, default=4)
    p.add_argument("--min-unique-centers", type=int, default=3)
    p.add_argument("--min-visual-span-px", type=float, default=100.0)
    p.add_argument("--min-map-span-m", type=float, default=50.0)
    p.add_argument("--median-residual-tile-fraction", type=float, default=0.5)
    p.add_argument("--max-residual-tile-fraction", type=float, default=1.0)
    p.add_argument("--cluster-tile-fraction", type=float, default=0.5)
    p.add_argument("--max-query-id", type=int, default=0)
    args = p.parse_args()

    run = args.run_root.resolve()
    candidate_csv = (args.candidate_csv or (
        run / "reports/s8_12e1_top20_verifier_reranker/512_s256_orb_hybrid_top20_img518/"
              "s8_12e1_all_candidate_verifier_scores.csv"
    )).resolve()
    relative_csv = (args.relative_csv or (
        run / "metadata/s8_xfeat_relative_frontend/s8r4_xfeat_relative_trajectory_blind_raw.csv"
    )).resolve()
    manifest_csv = (args.manifest_csv or (run / "metadata/blind_query_manifest.csv")).resolve()
    out = args.out_root.resolve()
    out.mkdir(parents=True, exist_ok=True)

    if args.top_m < 2 or args.evidence_frames < 3:
        raise RuntimeError("Need Top-M >=2 and evidence-frames >=3.")
    if args.top_m ** args.evidence_frames > 10000:
        raise RuntimeError("Hypothesis grid unexpectedly large.")

    cand, rel = load_inputs(candidate_csv, relative_csv, manifest_csv, args.top_m)
    common = sorted(set(cand["query_id"]) & set(rel["query_id"]))
    if args.max_query_id:
        common = [q for q in common if q <= args.max_query_id]
    if not common:
        raise RuntimeError("No common blind query IDs.")

    cand = cand[cand["query_id"].isin(common)].copy()
    rel = rel[rel["query_id"].isin(common)].sort_values("query_id").reset_index(drop=True)

    widths = (cand["right_easting"] - cand["left_easting"]).abs().to_numpy()
    heights = (cand["top_northing"] - cand["bottom_northing"]).abs().to_numpy()
    tile_size = float(np.nanmedian(np.r_[widths, heights]))
    cfg = {
        "min_unique_centers": args.min_unique_centers,
        "min_map_span_m": args.min_map_span_m,
        "median_limit_m": args.median_residual_tile_fraction * tile_size,
        "max_limit_m": args.max_residual_tile_fraction * tile_size,
        "cluster_threshold_m": args.cluster_tile_fraction * tile_size,
    }

    keep = [
        "query_id", "candidate_choice_rank", "tile_id", "rank", "score",
        "hybrid_rank", "hybrid_score", "verifier_rank", "verifier_score",
        "good_matches", "inliers", "inlier_ratio", "query_inlier_coverage",
        "sat_inlier_coverage", "homography_ok", "center_easting", "center_northing",
        "left_easting", "right_easting", "bottom_northing", "top_northing",
    ]
    candidate_out = out / "candidate_evidence.csv"
    cand[keep].to_csv(candidate_out, index=False)
    by_q = {int(q): g.to_dict("records") for q, g in cand.groupby("query_id", sort=True)}

    timeline, hyp_out, cluster_out = [], [], []
    provisional = None
    committed = None
    first_provisional = None

    for q in common:
        prefix = rel[rel["query_id"] <= q]
        ts = prefix.loc[prefix["query_id"] == q, "timestamp_s"]
        ts = float(ts.iloc[0]) if len(ts) and pd.notna(ts.iloc[0]) else None
        row = {
            "update_query_id": q, "timestamp_s": ts, "evidence_query_ids": "",
            "visual_span_px": np.nan, "hypotheses_raw": 0,
            "hypotheses_geometric_pass": 0, "pareto_hypotheses": 0,
            "transform_clusters": 0, "action": "",
            "state_after_update": "PROVISIONAL_CANDIDATE" if provisional else "RELATIVE_ONLY",
        }

        evidence_qids = select_evidence(prefix, args.evidence_frames)
        if len(evidence_qids) != args.evidence_frames:
            row["action"] = "WAIT_INSUFFICIENT_EVIDENCE_FRAMES"
            timeline.append(row)
            continue

        ev = rel.set_index("query_id").loc[evidence_qids]
        visual_xy = ev[["visual_x_px", "visual_y_px"]].to_numpy(float)
        visual_span = span(visual_xy)
        row["evidence_query_ids"] = ",".join(map(str, evidence_qids))
        row["visual_span_px"] = visual_span
        if visual_span < args.min_visual_span_px:
            row["action"] = "WAIT_VISUAL_SPAN"
            timeline.append(row)
            continue

        lists = [by_q[x] for x in evidence_qids]
        row["hypotheses_raw"] = int(np.prod([len(x) for x in lists]))
        admissible = []
        for combo in itertools.product(*lists):
            h = make_hypothesis(q, evidence_qids, visual_xy, combo, cfg)
            if h is not None:
                admissible.append(h)
        row["hypotheses_geometric_pass"] = len(admissible)
        if not admissible:
            row["action"] = "NO_GEOMETRICALLY_ADMISSIBLE_HYPOTHESIS"
            timeline.append(row)
            continue

        front = pareto(admissible)
        row["pareto_hypotheses"] = len(front)
        i, j = farthest_pair(visual_xy)
        sentinel_xy = visual_xy[[i, j]]
        clusters = cluster_models(front, sentinel_xy, cfg["cluster_threshold_m"])
        row["transform_clusters"] = len(clusters)

        front_ids = {id(h) for h in front}
        hyp_out.extend(serial_hypothesis(h, id(h) in front_ids) for h in admissible)
        for cid, c in enumerate(clusters, 1):
            r = c["rep"]
            cluster_out.append({
                "update_query_id": q, "timestamp_s": ts, "cluster_id": cid,
                "member_count": len(c["members"]),
                "max_rep_disagreement_m": c["max_rep_disagreement_m"],
                "representative_evidence_query_ids": ",".join(map(str, r["evidence_query_ids"])),
                "representative_tile_ids": ",".join(r["tile_ids"]),
                "representative_median_center_residual_m": r["median_center_residual_m"],
                "representative_sum_hybrid_rank": r["sum_hybrid_rank"],
                "representative_sum_dino_rank": r["sum_dino_rank"],
                "scale_m_per_visual_px": r["scale_m_per_visual_px"],
                "rotation_deg": r["rotation_deg"],
                "a_real": r["a_real"], "a_imag": r["a_imag"],
                "b_real": r["b_real"], "b_imag": r["b_imag"],
            })

        if len(clusters) != 1:
            row["action"] = "AMBIGUOUS_MULTIPLE_TRANSFORM_CLUSTERS"
            timeline.append(row)
            continue

        rep = clusters[0]["rep"]
        current = {
            "query_id": q, "timestamp_s": ts, "model": rep,
            "center_set": set(rep["map_centers"]), "rep": rep,
        }

        if provisional is None:
            provisional = current
            first_provisional = q
            row["action"] = "PROVISIONAL_CANDIDATE_CREATED"
            row["state_after_update"] = "PROVISIONAL_CANDIDATE"
            timeline.append(row)
            continue

        d0, d1 = disagreement(provisional["model"], current["model"], sentinel_xy)
        same = d0 <= cfg["cluster_threshold_m"] and d1 <= cfg["cluster_threshold_m"]
        new_centers = current["center_set"] - provisional["center_set"]
        row["model_disagreement_sentinel0_m"] = d0
        row["model_disagreement_sentinel1_m"] = d1
        row["new_independent_centers"] = len(new_centers)

        if same and new_centers:
            committed = current
            row["action"] = "PROVISIONAL_ABSOLUTE_LOCK_COMMITTED"
            row["state_after_update"] = "PROVISIONAL_ABSOLUTE_LOCK"
            timeline.append(row)
            break
        if same:
            row["action"] = "PROVISIONAL_PERSISTS_NO_NEW_CENTER"
            row["state_after_update"] = "PROVISIONAL_CANDIDATE"
            timeline.append(row)
            continue

        provisional = current
        row["action"] = "PROVISIONAL_REPLACED_INCOMPATIBLE_TRANSFORM"
        row["state_after_update"] = "PROVISIONAL_CANDIDATE"
        timeline.append(row)

    timeline_df = pd.DataFrame(timeline)
    pd.DataFrame(hyp_out).to_csv(out / "hypothesis_updates.csv", index=False)
    pd.DataFrame(cluster_out).to_csv(out / "transform_clusters.csv", index=False)
    timeline_df.to_csv(out / "provisional_bootstrap_timeline.csv", index=False)

    state = "PROVISIONAL_ABSOLUTE_LOCK" if committed else "NO_PROVISIONAL_LOCK"
    lock = None
    if committed:
        r = committed["rep"]
        lock = {
            **{k: r[k] for k in (
                "a_real", "a_imag", "b_real", "b_imag",
                "scale_m_per_visual_px", "rotation_deg",
            )},
            "lock_query_id": committed["query_id"],
            "lock_timestamp_s": committed["timestamp_s"],
            "evidence_query_ids": r["evidence_query_ids"],
            "tile_ids": r["tile_ids"],
            "candidate_choice_ranks": r["candidate_choice_ranks"],
            "median_center_residual_m": r["median_center_residual_m"],
            "max_center_residual_m": r["max_center_residual_m"],
            "map_span_m": r["map_span_m"],
            "map_unique_centers": r["map_unique_centers"],
            "sum_hybrid_rank": r["sum_hybrid_rank"],
            "sum_dino_rank": r["sum_dino_rank"],
        }

    blind_contract = {
        "gps_used": False, "srt_used": False, "reference_used": False,
        "oracle_used": False, "evaluation_error_used": False,
        "map_coordinates_allowed": True,
    }
    (out / "provisional_map_lock.json").write_text(
        json.dumps({
            "stage": "R3_minimum_confident_bootstrap",
            "localization_state": state,
            "map_lock": lock,
            "blind_contract": blind_contract,
        }, indent=2)
    )
    report = {
        "stage": "R3_minimum_confident_bootstrap",
        "status": "PASS_R3_PROVISIONAL_LOCK" if committed else "PASS_R3_NO_PROVISIONAL_LOCK",
        "localization_state": state,
        "run_root": str(run),
        "inputs": {
            "candidate_csv": str(candidate_csv),
            "relative_csv": str(relative_csv),
            "manifest_csv": str(manifest_csv),
        },
        "configuration": {
            "top_m": args.top_m, "evidence_frames": args.evidence_frames,
            "hypotheses_per_full_update": args.top_m ** args.evidence_frames,
            "min_unique_centers": args.min_unique_centers,
            "min_visual_span_px": args.min_visual_span_px,
            "min_map_span_m": args.min_map_span_m,
            "tile_size_m_derived": tile_size,
            "median_center_residual_limit_m": cfg["median_limit_m"],
            "max_center_residual_limit_m": cfg["max_limit_m"],
            "transform_cluster_threshold_m": cfg["cluster_threshold_m"],
            "max_query_id": args.max_query_id,
        },
        "counts": {
            "common_queries": len(common),
            "candidate_rows_topm": len(cand),
            "timeline_updates": len(timeline),
            "evaluated_updates": int(timeline_df["hypotheses_raw"].fillna(0).gt(0).sum()),
            "hypotheses_geometric_pass_total": len(hyp_out),
            "cluster_rows_total": len(cluster_out),
        },
        "acquisition": {
            "first_provisional_query_id": first_provisional,
            "lock_query_id": committed["query_id"] if committed else None,
            "lock_timestamp_s": committed["timestamp_s"] if committed else None,
        },
        "map_lock": lock,
        "blind_contract": blind_contract,
        "outputs": {
            "candidate_evidence_csv": str(candidate_out),
            "hypothesis_updates_csv": str(out / "hypothesis_updates.csv"),
            "transform_clusters_csv": str(out / "transform_clusters.csv"),
            "timeline_csv": str(out / "provisional_bootstrap_timeline.csv"),
            "provisional_map_lock_json": str(out / "provisional_map_lock.json"),
        },
    }
    report_path = out / "provisional_bootstrap_report.json"
    report_path.write_text(json.dumps(report, indent=2))

    print("R3 MINIMUM-CONFIDENT BOOTSTRAP")
    print("-" * 80)
    print("run root:", run)
    print("queries:", len(common))
    print("Top-M / evidence frames:", args.top_m, "/", args.evidence_frames)
    print("tile size derived [m]:", f"{tile_size:.3f}")
    print("median residual limit [m]:", f"{cfg['median_limit_m']:.3f}")
    print("cluster threshold [m]:", f"{cfg['cluster_threshold_m']:.3f}")
    print("timeline updates:", len(timeline))
    print("first provisional query:", first_provisional)
    print("localization state:", state)
    if committed:
        print("lock query:", committed["query_id"])
        print("lock timestamp [s]:", committed["timestamp_s"])
        print("scale [m/visual px]:", f"{committed['model']['scale_m_per_visual_px']:.6f}")
        print("rotation [deg]:", f"{committed['model']['rotation_deg']:.3f}")
        print("evidence queries:", committed["rep"]["evidence_query_ids"])
        print("tiles:", committed["rep"]["tile_ids"])
    else:
        print("\nlast actions:")
        print(timeline_df[
            ["update_query_id", "evidence_query_ids", "visual_span_px",
             "hypotheses_geometric_pass", "pareto_hypotheses",
             "transform_clusters", "action"]
        ].tail(12).to_string(index=False))
    print("\nSTATUS:", report["status"])
    print("report:", report_path)


if __name__ == "__main__":
    main()
