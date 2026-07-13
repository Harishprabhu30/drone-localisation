#!/usr/bin/env python3
"""
S5B.2 — LightGlue reranking inside S5B.1C union candidate pools

Purpose
-------
S5B.1C created multi-variant full-map union candidate pools for the 40 former
candidate-pool-failure tokens.

This script runs LightGlue/SuperPoint inside those union pools.

Locked rule
-----------
eval_error_m is used only after ranking for evaluation and oracle diagnostics.
LightGlue ranking uses only image matches, RANSAC inliers, inlier ratio,
coverage, and optional union-prior metadata.

Command Usde:
python scripts/satloc/s5b/s5b_2_lightglue_union_pool_verifier.py \
  --run-name cpf_union_recovered12 \
  --tokens 50,58,387,503,564,662,679,768,820,844,937,1034 \
  --max-tokens 0 \
  --max-candidates 200 \
  --resize-long 512 \
  --max-keypoints 1024 \
  --device cpu
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--union-pool",
        type=Path,
        default=Path("outputs/satloc/metadata/s5b_candidate_pool_improvement/s5b1c_union_candidate_pool_cpf_union_v3_v5_v8_v9.csv"),
    )
    p.add_argument("--out-base", type=Path, default=Path("outputs/satloc"))
    p.add_argument("--run-name", type=str, default="cpf_union_v3_v5_v8_v9_smoke")
    p.add_argument("--tokens", type=str, default="")
    p.add_argument("--max-tokens", type=int, default=3, help="Use 0 for all selected tokens.")
    p.add_argument("--max-candidates", type=int, default=80)
    p.add_argument("--threshold-m", type=float, default=40.0)

    p.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="cpu")
    p.add_argument("--resize-long", type=int, default=512)
    p.add_argument("--max-keypoints", type=int, default=1024)
    p.add_argument("--ransac-thresh", type=float, default=5.0)
    p.add_argument("--max-draw-matches", type=int, default=100)

    p.add_argument("--save-panels", action="store_true")
    return p.parse_args()


def ensure_dirs(base: Path):
    d = {
        "metadata": base / "metadata" / "s5b_candidate_pool_improvement",
        "reports": base / "reports" / "s5b_candidate_pool_improvement",
        "figures": base / "figures" / "s5b_candidate_pool_improvement",
        "panels": base / "figures" / "s5b_candidate_pool_improvement" / "s5b2_lightglue_union_panels",
    }
    for p in d.values():
        p.mkdir(parents=True, exist_ok=True)
    return d


def load_s5a3_module():
    path = Path("scripts/satloc/s5a/s5a_3_lightglue_topk_verifier.py")
    if not path.exists():
        raise FileNotFoundError(
            "Missing S5A.3 LightGlue helper script: "
            "scripts/satloc/s5a/s5a_3_lightglue_topk_verifier.py"
        )

    spec = importlib.util.spec_from_file_location("s5a3_lightglue_helpers", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def safe_str(x: Any) -> str:
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return str(x).strip()


def safe_float(x: Any):
    try:
        if x is None or pd.isna(x):
            return None
        return float(x)
    except Exception:
        return None


def resolve_path(x: Any) -> Optional[Path]:
    s = safe_str(x)
    if not s:
        return None
    p = Path(s)
    if p.exists():
        return p
    p2 = Path.cwd() / p
    if p2.exists():
        return p2
    return None


def select_tokens(pool: pd.DataFrame, tokens_arg: str, max_tokens: int) -> List[str]:
    if tokens_arg.strip():
        toks = [t.strip() for t in tokens_arg.split(",") if t.strip()]
    else:
        toks = sorted(pool["token"].astype(str).unique().tolist(), key=lambda x: int(float(x)))

    if max_tokens > 0:
        toks = toks[:max_tokens]
    return toks


def numeric(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series([np.nan] * len(df), index=df.index)
    return pd.to_numeric(df[col], errors="coerce")


def lg_score(matches: int, inliers: int, ratio: float, uav_cov: float, sat_cov: float, h_ok: bool) -> float:
    spread = min(float(uav_cov), float(sat_cov))
    score = (
        float(inliers)
        + 0.04 * float(matches)
        + 4.0 * float(ratio)
        + 6.0 * float(spread)
    )
    if h_ok:
        score += 2.0
    return float(score)


def process_candidate(mod, runner, token: str, row: pd.Series, args) -> Dict[str, Any]:
    uav_path = resolve_path(row.get("uav_image_path"))
    sat_path = resolve_path(row.get("sat_image_path"))

    out = {
        "token": token,
        "tile_id": safe_str(row.get("tile_id")),
        "uav_image_path": str(uav_path) if uav_path else "",
        "sat_image_path": str(sat_path) if sat_path else "",
        "eval_error_m": safe_float(row.get("eval_error_m")),
        "union_rank": safe_float(row.get("union_rank")),
        "union_support_count": safe_float(row.get("union_support_count")),
        "union_variants": safe_str(row.get("union_variants")),
        "union_best_source_rank": safe_float(row.get("union_best_source_rank")),
        "union_best_distance": safe_float(row.get("union_best_distance")),
        "lightglue_status": "not_started",
        "lightglue_error": "",
        "lightglue_matches": 0,
        "lightglue_ransac_inliers": 0,
        "lightglue_inlier_ratio": 0.0,
        "lightglue_homography_success": False,
        "lightglue_uav_coverage": 0.0,
        "lightglue_sat_coverage": 0.0,
        "lightglue_score": -1.0,
        "hybrid_union_score": -1.0,
        "runtime_s": 0.0,
    }

    if uav_path is None or sat_path is None:
        out["lightglue_status"] = "missing_path"
        return out

    try:
        result = runner.match(uav_path, sat_path)
        pts0 = result["pts0"]
        pts1 = result["pts1"]

        inliers, ratio, h_ok, mask = mod.homography_stats(pts0, pts1, args.ransac_thresh)

        if mask is not None and len(mask) == len(pts0):
            inlier_pts0 = pts0[mask]
            inlier_pts1 = pts1[mask]
        else:
            inlier_pts0 = pts0
            inlier_pts1 = pts1

        rgb0 = result["rgb0"]
        rgb1 = result["rgb1"]

        uav_cov = mod.grid_coverage(inlier_pts0, rgb0.shape[:2], grid=4)
        sat_cov = mod.grid_coverage(inlier_pts1, rgb1.shape[:2], grid=4)

        score = lg_score(
            matches=int(result["matches"]),
            inliers=int(inliers),
            ratio=float(ratio),
            uav_cov=float(uav_cov),
            sat_cov=float(sat_cov),
            h_ok=bool(h_ok),
        )

        union_rank = safe_float(row.get("union_rank")) or 999.0
        support = safe_float(row.get("union_support_count")) or 1.0

        # Experimental unsupervised prior:
        # LightGlue remains dominant, but we slightly reward tiles supported by
        # multiple global variants and not extremely low in union order.
        hybrid_score = score + 1.5 * support + 6.0 / math.sqrt(max(1.0, union_rank))

        out.update(
            {
                "lightglue_status": "ok",
                "lightglue_matches": int(result["matches"]),
                "lightglue_ransac_inliers": int(inliers),
                "lightglue_inlier_ratio": float(ratio),
                "lightglue_homography_success": bool(h_ok),
                "lightglue_uav_coverage": float(uav_cov),
                "lightglue_sat_coverage": float(sat_cov),
                "lightglue_score": float(score),
                "hybrid_union_score": float(hybrid_score),
                "runtime_s": float(result["runtime_s"]),
            }
        )
        return out

    except Exception as exc:
        out["lightglue_status"] = "failed"
        out["lightglue_error"] = repr(exc)
        return out


def rank_candidates(c: pd.DataFrame) -> pd.DataFrame:
    out = c.copy()
    out["lg_score_num"] = numeric(out, "lightglue_score")
    out["hybrid_score_num"] = numeric(out, "hybrid_union_score")
    out["lg_inliers_num"] = numeric(out, "lightglue_ransac_inliers")
    out["lg_matches_num"] = numeric(out, "lightglue_matches")
    out["min_cov_num"] = np.minimum(
        numeric(out, "lightglue_uav_coverage").fillna(0),
        numeric(out, "lightglue_sat_coverage").fillna(0),
    )
    out["union_rank_num"] = numeric(out, "union_rank")

    ranked_parts = []

    for token, g in out.groupby("token", dropna=False):
        gg = g.copy()

        lg = gg.sort_values(
            ["lg_score_num", "lg_inliers_num", "lg_matches_num", "min_cov_num", "union_rank_num"],
            ascending=[False, False, False, False, True],
            kind="mergesort",
        )
        gg.loc[lg.index, "lightglue_rank"] = np.arange(1, len(lg) + 1)

        hy = gg.sort_values(
            ["hybrid_score_num", "lg_score_num", "lg_inliers_num", "union_rank_num"],
            ascending=[False, False, False, True],
            kind="mergesort",
        )
        gg.loc[hy.index, "hybrid_union_rank"] = np.arange(1, len(hy) + 1)

        ranked_parts.append(gg)

    return pd.concat(ranked_parts, ignore_index=True) if ranked_parts else out


def summarize_policy(ranked: pd.DataFrame, threshold_m: float, rank_col: str, policy_name: str) -> pd.DataFrame:
    rows = []

    r = ranked.copy()
    r["eval_error_num"] = numeric(r, "eval_error_m")
    r["policy_rank_num"] = numeric(r, rank_col)
    r["union_rank_num"] = numeric(r, "union_rank")

    for token, g in r.groupby("token", dropna=False):
        g = g.copy()

        chosen = g.sort_values("policy_rank_num", kind="mergesort").iloc[0]
        union_top = g.sort_values("union_rank_num", kind="mergesort").iloc[0]

        valid = g.dropna(subset=["eval_error_num"])
        oracle = valid.sort_values("eval_error_num", kind="mergesort").iloc[0] if len(valid) else None

        chosen_err = safe_float(chosen.get("eval_error_m"))
        union_err = safe_float(union_top.get("eval_error_m"))
        oracle_err = safe_float(oracle.get("eval_error_m")) if oracle is not None else None

        rows.append(
            {
                "policy": policy_name,
                "token": token,
                "processed_candidates": int(len(g)),
                "chosen_tile_id": safe_str(chosen.get("tile_id")),
                "chosen_error_m": chosen_err,
                "hit_le_threshold": bool(chosen_err is not None and chosen_err <= threshold_m),
                "chosen_union_rank": safe_float(chosen.get("union_rank")),
                "chosen_lightglue_rank": safe_float(chosen.get("lightglue_rank")),
                "chosen_hybrid_rank": safe_float(chosen.get("hybrid_union_rank")),
                "chosen_lg_score": safe_float(chosen.get("lightglue_score")),
                "chosen_hybrid_score": safe_float(chosen.get("hybrid_union_score")),
                "chosen_inliers": safe_float(chosen.get("lightglue_ransac_inliers")),
                "chosen_matches": safe_float(chosen.get("lightglue_matches")),
                "chosen_min_coverage": safe_float(chosen.get("min_cov_num")),
                "union_top1_error_m": union_err,
                "union_top1_hit_le_threshold": bool(union_err is not None and union_err <= threshold_m),
                "oracle_processed_error_m": oracle_err,
                "oracle_processed_hit_le_threshold": bool(oracle_err is not None and oracle_err <= threshold_m),
                "oracle_lightglue_rank": safe_float(oracle.get("lightglue_rank")) if oracle is not None else None,
                "oracle_hybrid_rank": safe_float(oracle.get("hybrid_union_rank")) if oracle is not None else None,
                "oracle_union_rank": safe_float(oracle.get("union_rank")) if oracle is not None else None,
                "oracle_tile_id": safe_str(oracle.get("tile_id")) if oracle is not None else "",
            }
        )

    return pd.DataFrame(rows)


def policy_summary(q: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for policy, g in q.groupby("policy"):
        err = pd.to_numeric(g["chosen_error_m"], errors="coerce")
        rows.append(
            {
                "policy": policy,
                "tokens": int(len(g)),
                "hits": int(g["hit_le_threshold"].sum()),
                "hit_rate": float(g["hit_le_threshold"].mean()) if len(g) else 0.0,
                "median_error_m": float(err.median()) if err.notna().any() else None,
                "oracle_processed_hits": int(g["oracle_processed_hit_le_threshold"].sum()),
                "oracle_processed_hit_rate": float(g["oracle_processed_hit_le_threshold"].mean()) if len(g) else 0.0,
                "median_oracle_error_m": float(pd.to_numeric(g["oracle_processed_error_m"], errors="coerce").median()),
                "median_oracle_lg_rank": safe_float(pd.to_numeric(g["oracle_lightglue_rank"], errors="coerce").median()),
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["hit_rate", "median_error_m"],
        ascending=[False, True],
        kind="mergesort",
    )


def add_img(ax, img, title):
    ax.axis("off")
    ax.set_title(title, fontsize=9)
    if img is None:
        ax.text(0.5, 0.5, "missing", ha="center", va="center")
    else:
        ax.imshow(img)


def save_token_panel(mod, runner, ranked_token: pd.DataFrame, out_path: Path, args):
    token = safe_str(ranked_token.iloc[0]["token"])

    g = ranked_token.copy()
    g["eval_error_num"] = numeric(g, "eval_error_m")
    g["union_rank_num"] = numeric(g, "union_rank")
    g["lg_rank_num"] = numeric(g, "lightglue_rank")
    g["hy_rank_num"] = numeric(g, "hybrid_union_rank")

    rows = [
        ("Union top1", g.sort_values("union_rank_num", kind="mergesort").iloc[0]),
        ("LightGlue top1", g.sort_values("lg_rank_num", kind="mergesort").iloc[0]),
        ("Hybrid top1", g.sort_values("hy_rank_num", kind="mergesort").iloc[0]),
    ]

    valid = g.dropna(subset=["eval_error_num"])
    if len(valid):
        rows.append(("Oracle best", valid.sort_values("eval_error_num", kind="mergesort").iloc[0]))

    fig, axes = plt.subplots(len(rows), 3, figsize=(17, 4.2 * len(rows)))
    if len(rows) == 1:
        axes = np.asarray([axes])

    for i, (role, row) in enumerate(rows):
        uav_path = resolve_path(row.get("uav_image_path"))
        sat_path = resolve_path(row.get("sat_image_path"))

        if uav_path is None or sat_path is None:
            continue

        result = runner.match(uav_path, sat_path)
        pts0 = result["pts0"]
        pts1 = result["pts1"]
        inliers, ratio, h_ok, mask = mod.homography_stats(pts0, pts1, args.ransac_thresh)
        match_img = mod.draw_matches_canvas(
            result["rgb0"],
            result["rgb1"],
            pts0,
            pts1,
            mask,
            args.max_draw_matches,
        )

        add_img(axes[i, 0], result["rgb0"], f"UAV token {token}")
        add_img(
            axes[i, 1],
            result["rgb1"],
            f"{role}\n"
            f"tile={safe_str(row.get('tile_id'))} err={safe_float(row.get('eval_error_m')):.1f}m\n"
            f"union-r={safe_float(row.get('union_rank'))} lg-r={safe_float(row.get('lightglue_rank'))} hy-r={safe_float(row.get('hybrid_union_rank'))}",
        )
        add_img(
            axes[i, 2],
            match_img,
            f"LG score={safe_float(row.get('lightglue_score')):.2f} "
            f"inliers={safe_float(row.get('lightglue_ransac_inliers'))} "
            f"matches={safe_float(row.get('lightglue_matches'))}",
        )

    fig.suptitle(
        "S5B.2 LightGlue reranking inside S5B union pool\n"
        "Green = RANSAC inlier, red = outlier. Oracle shown only after ranking.",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_policy_summary(ps: pd.DataFrame, out_path: Path):
    if len(ps) == 0:
        return
    plt.figure(figsize=(8, 5))
    plt.bar(ps["policy"], ps["hit_rate"])
    plt.ylim(0, 1.05)
    plt.ylabel("Hit rate <= threshold")
    plt.title("S5B.2 LightGlue union-pool policies")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def main():
    args = parse_args()
    dirs = ensure_dirs(args.out_base)
    mod = load_s5a3_module()

    if not args.union_pool.exists():
        raise FileNotFoundError(args.union_pool)

    pool = pd.read_csv(args.union_pool)
    pool["token_str"] = pool["token"].astype(str)
    pool["union_rank_num"] = numeric(pool, "union_rank")

    tokens = select_tokens(pool, args.tokens, args.max_tokens)

    env = mod.detect_device(args.device)
    runner = mod.LightGlueRunner(
        device=safe_str(env.get("selected_device", "cpu")),
        max_keypoints=args.max_keypoints,
        resize_long=args.resize_long,
    )

    candidate_rows = []
    started = time.time()

    for token in tokens:
        sub = pool[pool["token_str"] == str(token)].copy()
        sub = sub.sort_values("union_rank_num", kind="mergesort").head(args.max_candidates)

        print(f"[S5B.2] token={token} union_candidates={len(sub)}")

        for _, row in sub.iterrows():
            candidate_rows.append(process_candidate(mod, runner, str(token), row, args))

    cand = pd.DataFrame(candidate_rows)
    ranked = rank_candidates(cand)

    q_lg = summarize_policy(ranked, args.threshold_m, "lightglue_rank", "lightglue_only")
    q_hy = summarize_policy(ranked, args.threshold_m, "hybrid_union_rank", "lightglue_plus_union_prior")
    q_union = summarize_policy(ranked, args.threshold_m, "union_rank", "union_rank_only")
    q = pd.concat([q_union, q_lg, q_hy], ignore_index=True)

    ps = policy_summary(q)

    suffix = f"_{args.run_name}" if args.run_name else ""

    cand_out = dirs["metadata"] / f"s5b2_lightglue_union_candidate_scores{suffix}.csv"
    query_out = dirs["metadata"] / f"s5b2_lightglue_union_query_summary{suffix}.csv"
    policy_out = dirs["metadata"] / f"s5b2_lightglue_union_policy_summary{suffix}.csv"
    env_out = dirs["reports"] / f"s5b2_environment{suffix}.json"
    report_out = dirs["reports"] / f"s5b2_lightglue_union_summary{suffix}.json"
    fig_out = dirs["figures"] / f"s5b2_lightglue_union_policy_hit_rates{suffix}.png"

    ranked.to_csv(cand_out, index=False)
    q.to_csv(query_out, index=False)
    ps.to_csv(policy_out, index=False)

    with open(env_out, "w") as f:
        json.dump(env, f, indent=2)

    plot_policy_summary(ps, fig_out)

    panel_paths = []
    if args.save_panels:
        for token in tokens:
            rt = ranked[ranked["token"].astype(str) == str(token)]
            if len(rt) == 0:
                continue
            out_path = dirs["panels"] / args.run_name / f"s5b2_token{int(float(token)):04d}_union_lightglue_panel.png"
            save_token_panel(mod, runner, rt, out_path, args)
            panel_paths.append(str(out_path))

    report = {
        "stage": "S5B.2_lightglue_union_pool_verifier",
        "run_name": args.run_name,
        "tokens": tokens,
        "num_tokens": int(len(tokens)),
        "max_candidates": args.max_candidates,
        "threshold_m": args.threshold_m,
        "resize_long": args.resize_long,
        "max_keypoints": args.max_keypoints,
        "runtime_s": float(time.time() - started),
        "status_counts": ranked["lightglue_status"].value_counts(dropna=False).to_dict() if len(ranked) else {},
        "policy_summary": ps.to_dict(orient="records"),
        "outputs": {
            "candidate_scores_csv": str(cand_out),
            "query_summary_csv": str(query_out),
            "policy_summary_csv": str(policy_out),
            "environment_json": str(env_out),
            "summary_json": str(report_out),
            "policy_hit_rate_figure": str(fig_out),
            "panel_paths": panel_paths,
        },
        "locked_rule": "eval_error_m was used only after ranking for evaluation and oracle diagnostics",
    }

    with open(report_out, "w") as f:
        json.dump(report, f, indent=2)

    print()
    print("S5B.2 LightGlue union-pool verifier complete")
    print("---------------------------------------------")
    print(f"Tokens processed:          {len(tokens)} -> {tokens}")
    print(f"Max candidates/token:      {args.max_candidates}")
    print(f"Candidate rows:            {len(ranked)}")
    print(f"Status counts:             {report['status_counts']}")
    print()
    print("Policy summary:")
    print(ps.to_string(index=False))
    print()
    print(f"Candidate scores CSV:      {cand_out}")
    print(f"Query summary CSV:         {query_out}")
    print(f"Policy summary CSV:        {policy_out}")
    print(f"Summary JSON:              {report_out}")
    print(f"Figure:                    {fig_out}")
    if args.save_panels:
        print(f"Panels dir:                {dirs['panels'] / args.run_name}")
    print()
    print("Locked rule: reference/error columns were used only after ranking.")


if __name__ == "__main__":
    main()
