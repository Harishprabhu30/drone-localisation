#!/usr/bin/env python3
"""
S7B.3 — Multi-scale / FOV Retrieval Union Diagnostic

Combines multiple learned-retrieval candidate CSVs, usually:
  center_square DINOv2-VLAD + resize_square DINOv2-VLAD

Ranking uses Reciprocal Rank Fusion (RRF) from retrieval ranks only.
eval_error_m is used only after ranking for offline Recall@K evaluation.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", type=Path, default=Path("."))
    p.add_argument("--stream", action="append", required=True,
                   help="name=path to candidate CSV. Repeat this flag.")
    p.add_argument("--top-k", type=int, default=100)
    p.add_argument("--threshold-m", type=float, default=40.0)
    p.add_argument("--recall-ks", default="1,5,10,20,50,100")
    p.add_argument("--rrf-k", type=float, default=60.0)
    p.add_argument("--tag", default="")
    p.add_argument("--metadata-out", type=Path, default=Path("outputs/satloc/metadata/s7b_multiscale_fov"))
    p.add_argument("--report-out", type=Path, default=Path("outputs/satloc/reports/s7b_multiscale_fov"))
    p.add_argument("--figure-out", type=Path, default=Path("outputs/satloc/figures/s7b_multiscale_fov"))
    return p.parse_args()


def safe_name(x: Any) -> str:
    s = str(x).strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_") or "run"


def resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def json_safe(x: Any) -> Any:
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, float) and not math.isfinite(x):
        return None
    try:
        if pd.isna(x):
            return None
    except Exception:
        pass
    return x


def parse_ks(text: str, top_k: int) -> list[int]:
    ks = sorted(set(int(x.strip()) for x in text.split(",") if x.strip()))
    ks = [k for k in ks if 0 < k <= top_k]
    if top_k not in ks:
        ks.append(top_k)
    return ks


def parse_streams(items: list[str], root: Path) -> list[tuple[str, Path]]:
    streams = []
    names = set()
    for item in items:
        if "=" not in item:
            raise ValueError(f"--stream must be name=path, got: {item}")
        name, p = item.split("=", 1)
        name = safe_name(name)
        if name in names:
            raise ValueError(f"Duplicate stream name: {name}")
        names.add(name)
        path = resolve(root, Path(p))
        if not path.exists():
            raise FileNotFoundError(f"Missing stream CSV for {name}: {path}")
        streams.append((name, path))
    if len(streams) < 2:
        raise ValueError("Need at least two streams for S7B.3 union.")
    return streams


def find_score_col(df: pd.DataFrame) -> str | None:
    preferred = ["dinov2_vlad_similarity", "dinov2_similarity", "similarity", "score"]
    for c in preferred:
        if c in df.columns:
            return c
    for c in df.columns:
        low = c.lower()
        if "similarity" in low or low.endswith("_score"):
            return c
    return None


def load_stream(name: str, path: Path, threshold_m: float) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    required = ["token", "rank", "tile_id", "eval_error_m"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing columns {missing}. Available={list(df.columns)}")

    score_col = find_score_col(df)
    out = pd.DataFrame({
        "stream": name,
        "token": pd.to_numeric(df["token"], errors="raise").astype(int),
        "tile_id": df["tile_id"].astype(str),
        "stream_rank": pd.to_numeric(df["rank"], errors="raise").astype(int),
        "stream_score": pd.to_numeric(df[score_col], errors="coerce") if score_col else np.nan,
        "eval_error_m": pd.to_numeric(df["eval_error_m"], errors="coerce"),
        "primary_scene": df["primary_scene"].fillna("unlabeled").astype(str) if "primary_scene" in df.columns else "unlabeled",
    })
    for c in ["query_image_path", "sat_image_path"]:
        if c in df.columns:
            out[c] = df[c]
    out["stream_hit_eval_only"] = out["eval_error_m"].le(threshold_m)
    return out.drop_duplicates(["stream", "token", "tile_id"], keep="first")


def stream_summary(df: pd.DataFrame, name: str, ks: list[int], threshold_m: float) -> dict[str, Any]:
    rows = []
    for token, g in df.groupby("token"):
        g = g.sort_values("stream_rank")
        err = g["eval_error_m"].to_numpy(float)
        row = {"token": int(token)}
        row["top1_error_m"] = float(err[0]) if len(err) and np.isfinite(err[0]) else np.nan
        row["top1_hit"] = bool(len(err) and np.isfinite(err[0]) and err[0] <= threshold_m)
        if np.isfinite(err).any():
            p = int(np.nanargmin(err))
            row["oracle_error_m"] = float(err[p])
            row["oracle_rank"] = p + 1
        else:
            row["oracle_error_m"] = np.nan
            row["oracle_rank"] = np.nan
        for k in ks:
            kk = min(k, len(err))
            row[f"hit_at_{k}"] = bool(kk and np.isfinite(err[:kk]).any() and np.nanmin(err[:kk]) <= threshold_m)
        rows.append(row)
    q = pd.DataFrame(rows)
    d = {
        "stream": name,
        "queries": int(len(q)),
        "top1_hits": int(q["top1_hit"].sum()) if len(q) else 0,
        "top1_recall": float(q["top1_hit"].mean()) if len(q) else 0.0,
        "median_top1_error_m": float(pd.to_numeric(q["top1_error_m"], errors="coerce").median()),
        "median_oracle_error_m": float(pd.to_numeric(q["oracle_error_m"], errors="coerce").median()),
        "median_oracle_rank": float(pd.to_numeric(q["oracle_rank"], errors="coerce").median()),
    }
    for k in ks:
        d[f"hits_at_{k}"] = int(q[f"hit_at_{k}"].sum()) if len(q) else 0
        d[f"recall_at_{k}"] = float(q[f"hit_at_{k}"].mean()) if len(q) else 0.0
    return d


def build_union(all_df: pd.DataFrame, ks: list[int], threshold_m: float, top_k: int, rrf_k: float):
    stream_names = sorted(all_df["stream"].unique())
    rows = []
    for (token, tile_id), g in all_df.groupby(["token", "tile_id"], sort=False):
        ranks = pd.to_numeric(g["stream_rank"], errors="coerce")
        row = {
            "token": int(token),
            "tile_id": str(tile_id),
            "rrf_score": float((1.0 / (rrf_k + ranks)).sum()),
            "stream_count": int(g["stream"].nunique()),
            "best_stream_rank": int(ranks.min()),
            "present_streams": ",".join(sorted(g["stream"].unique())),
            "eval_error_m": float(pd.to_numeric(g["eval_error_m"], errors="coerce").min()),
            "primary_scene": g["primary_scene"].dropna().astype(str).iloc[0],
        }
        for s in stream_names:
            sg = g[g["stream"] == s]
            row[f"{s}_present"] = bool(len(sg))
            row[f"{s}_rank"] = int(sg["stream_rank"].iloc[0]) if len(sg) else np.nan
            row[f"{s}_score"] = float(sg["stream_score"].iloc[0]) if len(sg) and pd.notna(sg["stream_score"].iloc[0]) else np.nan
        for c in ["query_image_path", "sat_image_path"]:
            if c in g.columns and g[c].notna().any():
                row[c] = g[c].dropna().astype(str).iloc[0]
        rows.append(row)

    merged = pd.DataFrame(rows)
    cand_parts = []
    for token, g in merged.groupby("token", sort=True):
        g = g.sort_values(["rrf_score", "stream_count", "best_stream_rank"], ascending=[False, False, True]).copy()
        g["union_rank"] = np.arange(1, len(g) + 1)
        cand_parts.append(g.head(top_k))
    cand = pd.concat(cand_parts, ignore_index=True)

    qrows = []
    for token, g in cand.groupby("token", sort=True):
        g = g.sort_values("union_rank")
        err = g["eval_error_m"].to_numpy(float)
        row = {
            "token": int(token),
            "primary_scene": g["primary_scene"].iloc[0],
            "top1_tile_id": str(g.iloc[0]["tile_id"]),
            "top1_error_m_eval_only": float(err[0]) if len(err) and np.isfinite(err[0]) else np.nan,
            "top1_hit_le_threshold_eval_only": bool(len(err) and np.isfinite(err[0]) and err[0] <= threshold_m),
            "union_candidate_count": int(len(g)),
        }
        if np.isfinite(err).any():
            p = int(np.nanargmin(err))
            row["oracle_tile_id_eval_only"] = str(g.iloc[p]["tile_id"])
            row["oracle_error_m_eval_only"] = float(err[p])
            row["oracle_rank_eval_only"] = p + 1
        else:
            row["oracle_tile_id_eval_only"] = ""
            row["oracle_error_m_eval_only"] = np.nan
            row["oracle_rank_eval_only"] = np.nan
        for k in ks:
            kk = min(k, len(err))
            row[f"hit_at_{k}_eval_only"] = bool(kk and np.isfinite(err[:kk]).any() and np.nanmin(err[:kk]) <= threshold_m)
        qrows.append(row)
    q = pd.DataFrame(qrows)

    recall = pd.DataFrame([
        {
            "k": k,
            "hits": int(q[f"hit_at_{k}_eval_only"].sum()),
            "queries": int(len(q)),
            "recall": float(q[f"hit_at_{k}_eval_only"].mean()) if len(q) else 0.0,
        }
        for k in ks
    ])

    scene_rows = []
    for scene, g in q.groupby("primary_scene", dropna=False):
        row = {
            "primary_scene": scene,
            "queries": int(len(g)),
            "top1_hits": int(g["top1_hit_le_threshold_eval_only"].sum()),
            "top1_hit_rate": float(g["top1_hit_le_threshold_eval_only"].mean()),
            "median_top1_error_m": float(pd.to_numeric(g["top1_error_m_eval_only"], errors="coerce").median()),
            "median_oracle_error_m": float(pd.to_numeric(g["oracle_error_m_eval_only"], errors="coerce").median()),
            "median_oracle_rank": float(pd.to_numeric(g["oracle_rank_eval_only"], errors="coerce").median()),
        }
        for k in ks:
            row[f"recall_at_{k}"] = float(g[f"hit_at_{k}_eval_only"].mean())
        scene_rows.append(row)
    scene = pd.DataFrame(scene_rows).sort_values(["queries", "primary_scene"], ascending=[False, True])

    return cand, q, recall, scene


def rescue_breakdown(all_df: pd.DataFrame, query: pd.DataFrame, ks: list[int], threshold_m: float) -> pd.DataFrame:
    target_k = max(ks)
    streams = sorted(all_df["stream"].unique())
    hit_sets = {}
    for s in streams:
        sg = all_df[all_df["stream"] == s]
        hits = set()
        for token, g in sg.groupby("token"):
            g = g.sort_values("stream_rank").head(target_k)
            e = pd.to_numeric(g["eval_error_m"], errors="coerce").to_numpy(float)
            if len(e) and np.isfinite(e).any() and np.nanmin(e) <= threshold_m:
                hits.add(int(token))
        hit_sets[s] = hits
    union_hits = set(query.loc[query[f"hit_at_{target_k}_eval_only"], "token"].astype(int))
    any_single = set().union(*hit_sets.values()) if hit_sets else set()
    rows = [{"category": f"union_hits_at_{target_k}", "count": len(union_hits)}]
    for s, h in hit_sets.items():
        rows.append({"category": f"{s}_hits_at_{target_k}", "count": len(h)})
    rows.append({"category": f"union_new_vs_any_single_at_{target_k}", "count": len(union_hits - any_single)})
    for s, h in hit_sets.items():
        others = set().union(*[v for k, v in hit_sets.items() if k != s]) if len(hit_sets) > 1 else set()
        rows.append({"category": f"{s}_unique_hits_at_{target_k}", "count": len(h - others)})
    return pd.DataFrame(rows)


def plot_bar(df: pd.DataFrame, x: str, y: str, out: Path, title: str, ylim01=False, rotate=0):
    if df.empty or x not in df.columns or y not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(df[x].astype(str), df[y])
    if ylim01:
        ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    ax.tick_params(axis="x", rotation=rotate)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    root = args.repo_root.resolve()
    streams = parse_streams(args.stream, root)
    ks = parse_ks(args.recall_ks, args.top_k)
    tag = safe_name(args.tag or "_".join(n for n, _ in streams))

    metadata_out = resolve(root, args.metadata_out)
    report_out = resolve(root, args.report_out)
    figure_out = resolve(root, args.figure_out)
    for d in [metadata_out, report_out, figure_out]:
        d.mkdir(parents=True, exist_ok=True)

    print()
    print("S7B.3 — Multi-scale / FOV Retrieval Union Diagnostic")
    print("----------------------------------------------------")
    print(f"Repository root:        {root}")
    print(f"Tag:                    {tag}")
    print(f"Top-K:                  {args.top_k}")
    print(f"Threshold m:            {args.threshold_m}")
    print(f"RRF k:                  {args.rrf_k}")
    print("Streams:")
    for n, p in streams:
        print(f"  {n}: {p}")

    dfs = []
    summaries = []
    for n, p in streams:
        df = load_stream(n, p, args.threshold_m)
        dfs.append(df)
        summaries.append(stream_summary(df, n, ks, args.threshold_m))
        print(f"[S7B.3] Loaded {n}: rows={len(df)} tokens={df['token'].nunique()}")
    all_df = pd.concat(dfs, ignore_index=True)
    stream_comp = pd.DataFrame(summaries)

    cand, query, recall, scene = build_union(all_df, ks, args.threshold_m, args.top_k, args.rrf_k)
    rescue = rescue_breakdown(all_df, query, ks, args.threshold_m)

    cand_csv = metadata_out / f"s7b3_union_candidate_scores_{tag}.csv"
    query_csv = metadata_out / f"s7b3_union_query_summary_{tag}.csv"
    recall_csv = metadata_out / f"s7b3_union_recall_summary_{tag}.csv"
    scene_csv = metadata_out / f"s7b3_union_scene_summary_{tag}.csv"
    comp_csv = metadata_out / f"s7b3_union_stream_comparison_{tag}.csv"
    rescue_csv = metadata_out / f"s7b3_union_rescue_breakdown_{tag}.csv"
    summary_json = report_out / f"s7b3_union_summary_{tag}.json"

    cand.to_csv(cand_csv, index=False)
    query.to_csv(query_csv, index=False)
    recall.to_csv(recall_csv, index=False)
    scene.to_csv(scene_csv, index=False)
    stream_comp.to_csv(comp_csv, index=False)
    rescue.to_csv(rescue_csv, index=False)

    recall_fig = figure_out / f"s7b3_union_recall_at_k_{tag}.png"
    scene_fig = figure_out / f"s7b3_union_scene_recall_at_{max(ks)}_{tag}.png"
    rescue_fig = figure_out / f"s7b3_union_rescue_breakdown_{tag}.png"
    plot_bar(recall, "k", "recall", recall_fig, "S7B.3 RRF union Recall@K", ylim01=True)
    plot_bar(scene, "primary_scene", f"recall_at_{max(ks)}", scene_fig, f"S7B.3 scene Recall@{max(ks)}", ylim01=True, rotate=35)
    plot_bar(rescue, "category", "count", rescue_fig, "S7B.3 rescue breakdown", rotate=45)

    top1_hits = int(query["top1_hit_le_threshold_eval_only"].sum())
    topk_hits = int(query[f"hit_at_{max(ks)}_eval_only"].sum())
    summary = {
        "stage": "S7B.3_multiscale_fov_union_diagnostic",
        "status": "COMPLETE",
        "tag": tag,
        "streams": [{"name": n, "path": str(p)} for n, p in streams],
        "queries": int(len(query)),
        "top_k": int(args.top_k),
        "threshold_m": float(args.threshold_m),
        "rrf_k": float(args.rrf_k),
        "top1_hits_le_threshold": top1_hits,
        "top1_recall": float(top1_hits / len(query)) if len(query) else 0.0,
        f"top{max(ks)}_hits_le_threshold": topk_hits,
        f"top{max(ks)}_recall": float(topk_hits / len(query)) if len(query) else 0.0,
        "median_top1_error_m_eval_only": float(pd.to_numeric(query["top1_error_m_eval_only"], errors="coerce").median()),
        "median_oracle_error_m_eval_only": float(pd.to_numeric(query["oracle_error_m_eval_only"], errors="coerce").median()),
        "median_oracle_rank_eval_only": float(pd.to_numeric(query["oracle_rank_eval_only"], errors="coerce").median()),
        "outputs": {
            "candidate_scores_csv": str(cand_csv),
            "query_summary_csv": str(query_csv),
            "recall_summary_csv": str(recall_csv),
            "scene_summary_csv": str(scene_csv),
            "stream_comparison_csv": str(comp_csv),
            "rescue_breakdown_csv": str(rescue_csv),
            "summary_json": str(summary_json),
            "recall_figure": str(recall_fig),
            "scene_figure": str(scene_fig),
            "rescue_figure": str(rescue_fig),
        },
        "locked_rule": "RRF union uses retrieval ranks only; eval_error_m is used only after ranking for offline evaluation.",
    }
    summary_json.write_text(json.dumps(summary, indent=2, default=json_safe), encoding="utf-8")

    print()
    print("S7B.3 — Multi-scale / FOV Retrieval Union Diagnostic")
    print("----------------------------------------------------")
    print("Status:                         COMPLETE")
    print(f"Tag:                            {tag}")
    print(f"Queries:                        {len(query)}")
    print(f"Union candidate rows:           {len(cand)}")
    print(f"Top-K evaluated:                {args.top_k}")
    print(f"Threshold m:                    {args.threshold_m:.1f}")
    print(f"Top-1 hits:                     {top1_hits}/{len(query)}")
    print(f"Top-{max(ks)} hits:                  {topk_hits}/{len(query)}")
    print(f"Median top-1 error m:           {summary['median_top1_error_m_eval_only']:.3f}")
    print(f"Median oracle error m:          {summary['median_oracle_error_m_eval_only']:.3f}")
    print(f"Median oracle rank:             {summary['median_oracle_rank_eval_only']:.3f}")
    print()
    print("Stream comparison:")
    print(stream_comp.to_string(index=False))
    print()
    print("Union recall summary:")
    print(recall.to_string(index=False))
    print()
    print("Union scene summary:")
    print(scene.to_string(index=False))
    print()
    print("Rescue breakdown:")
    print(rescue.to_string(index=False))
    print()
    print(f"Candidate scores CSV:           {cand_csv}")
    print(f"Query summary CSV:              {query_csv}")
    print(f"Recall summary CSV:             {recall_csv}")
    print(f"Scene summary CSV:              {scene_csv}")
    print(f"Stream comparison CSV:          {comp_csv}")
    print(f"Rescue breakdown CSV:           {rescue_csv}")
    print(f"Summary JSON:                   {summary_json}")
    print(f"Recall figure:                  {recall_fig}")
    print(f"Scene figure:                   {scene_fig}")
    print(f"Rescue figure:                  {rescue_fig}")
    print()
    print("Locked rule: RRF union uses retrieval ranks only; GT/error used only after ranking.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
