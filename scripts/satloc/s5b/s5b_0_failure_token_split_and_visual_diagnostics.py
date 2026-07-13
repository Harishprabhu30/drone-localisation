#!/usr/bin/env python3
"""
S5B.0 — Token split and visual-domain diagnostics

Purpose
-------
Split the 73-frame S5A benchmark into:
1. Hybrid/LightGlue solved frames
2. Recoverable missed frames
3. Candidate-pool failure frames

Then compute visual diagnostics for UAV and PHOG-top1 satellite candidates:
- green dominance
- warmth / tint
- edge density
- dominant orientation concentration
- LAB color distance between UAV and satellite top1
- edge-density ratio

This prepares the next S5B branch:
candidate generation improvement for hard candidate-pool failures.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--query-analysis",
        type=Path,
        default=Path("outputs/satloc/metadata/s5a_learned_local_verifier/s5a3d_query_failure_analysis_top50_all73.csv"),
    )
    p.add_argument(
        "--best-policy",
        type=Path,
        default=Path("outputs/satloc/metadata/s5a_learned_local_verifier/s5a3e_best_policy_decisions_top50_all73.csv"),
    )
    p.add_argument(
        "--candidate-scores",
        type=Path,
        default=Path("outputs/satloc/metadata/s5a_learned_local_verifier/s5a3_lightglue_candidate_scores_top50_all73.csv"),
    )
    p.add_argument("--out-base", type=Path, default=Path("outputs/satloc"))
    p.add_argument("--run-name", type=str, default="top50_all73")
    p.add_argument("--resize-long", type=int, default=512)
    return p.parse_args()


def ensure_dirs(base: Path):
    d = {
        "metadata": base / "metadata" / "s5b_candidate_pool_improvement",
        "reports": base / "reports" / "s5b_candidate_pool_improvement",
        "figures": base / "figures" / "s5b_candidate_pool_improvement",
    }
    for p in d.values():
        p.mkdir(parents=True, exist_ok=True)
    return d


def resolve_path(value: Any) -> Optional[Path]:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    p = Path(text)
    if p.exists():
        return p
    p2 = Path.cwd() / p
    if p2.exists():
        return p2
    return None


def read_rgb(path: Optional[Path]) -> Optional[np.ndarray]:
    if path is None:
        return None
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def resize_longest(img: np.ndarray, longest: int) -> np.ndarray:
    h, w = img.shape[:2]
    m = max(h, w)
    if m <= longest:
        return img
    s = longest / float(m)
    return cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)


def image_metrics(rgb: Optional[np.ndarray], resize_long: int) -> Dict[str, float]:
    if rgb is None:
        return {
            "mean_l": np.nan,
            "mean_a": np.nan,
            "mean_b": np.nan,
            "warmth_r_minus_b": np.nan,
            "green_ratio": np.nan,
            "excess_green_mean": np.nan,
            "edge_density": np.nan,
            "orientation_peak_ratio": np.nan,
            "orientation_entropy": np.nan,
        }

    img = resize_longest(rgb, resize_long)
    rgb_f = img.astype(np.float32)

    r = rgb_f[:, :, 0]
    g = rgb_f[:, :, 1]
    b = rgb_f[:, :, 2]

    # Color / vegetation proxies.
    warmth = float(np.mean(r - b))
    exg = 2.0 * g - r - b
    exg_norm = exg / 255.0
    excess_green_mean = float(np.mean(exg_norm))

    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    green_mask = (h >= 35) & (h <= 90) & (s > 35) & (v > 35)
    green_ratio = float(green_mask.mean())

    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    mean_l = float(np.mean(lab[:, :, 0]))
    mean_a = float(np.mean(lab[:, :, 1]))
    mean_b = float(np.mean(lab[:, :, 2]))

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray_eq = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)

    edges = cv2.Canny(gray_eq, 60, 140)
    edge_density = float((edges > 0).mean())

    gx = cv2.Sobel(gray_eq, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_eq, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    ang = (np.arctan2(gy, gx) + np.pi) % np.pi

    valid = mag > np.percentile(mag, 75)
    if valid.sum() < 10:
        orientation_peak_ratio = 0.0
        orientation_entropy = 0.0
    else:
        hist, _ = np.histogram(ang[valid], bins=18, range=(0, np.pi), weights=mag[valid])
        total = hist.sum() + 1e-9
        prob = hist / total
        orientation_peak_ratio = float(hist.max() / total)
        orientation_entropy = float(-(prob * np.log(prob + 1e-9)).sum() / np.log(len(hist)))

    return {
        "mean_l": mean_l,
        "mean_a": mean_a,
        "mean_b": mean_b,
        "warmth_r_minus_b": warmth,
        "green_ratio": green_ratio,
        "excess_green_mean": excess_green_mean,
        "edge_density": edge_density,
        "orientation_peak_ratio": orientation_peak_ratio,
        "orientation_entropy": orientation_entropy,
    }


def as_bool(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s.fillna(False)
    return s.fillna(False).astype(str).str.lower().isin(["true", "1", "yes"])


def token_list_str(values):
    return ",".join([str(int(float(v))) for v in values])


def plot_box(df: pd.DataFrame, col: str, out: Path, title: str):
    groups = []
    labels = []
    for label, sub in df.groupby("s5b_split"):
        vals = pd.to_numeric(sub[col], errors="coerce").dropna().values
        if len(vals):
            groups.append(vals)
            labels.append(label)

    if not groups:
        return

    plt.figure(figsize=(10, 5.5))
    plt.boxplot(groups, labels=labels, showfliers=True)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel(col)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out, dpi=180)
    plt.close()


def plot_scatter(df: pd.DataFrame, xcol: str, ycol: str, out: Path, title: str):
    plt.figure(figsize=(8, 5.5))
    for label, sub in df.groupby("s5b_split"):
        x = pd.to_numeric(sub[xcol], errors="coerce")
        y = pd.to_numeric(sub[ycol], errors="coerce")
        plt.scatter(x, y, s=30, alpha=0.65, label=label)
    plt.xlabel(xcol)
    plt.ylabel(ycol)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=180)
    plt.close()


def main():
    args = parse_args()
    dirs = ensure_dirs(args.out_base)
    suffix = f"_{args.run_name}" if args.run_name else ""

    q = pd.read_csv(args.query_analysis)
    best = pd.read_csv(args.best_policy)
    cand = pd.read_csv(args.candidate_scores)

    q["token_str"] = q["token"].astype(str)
    best["token_str"] = best["token"].astype(str)
    cand["token_str"] = cand["token"].astype(str)

    q["oracle_hit"] = as_bool(q["oracle_topk_hit_le_threshold"])
    q["lightglue_hit"] = as_bool(q["lightglue_hit_le_threshold"])

    best["hybrid_hit"] = as_bool(best["hit_le_threshold"])

    merged = q.merge(
        best[["token_str", "chosen_error_m", "hit_le_threshold", "chosen_tile_id"]],
        on="token_str",
        how="left",
        suffixes=("", "_best_gate"),
    )

    merged["hybrid_hit"] = as_bool(merged["hit_le_threshold"])

    def split(row):
        if bool(row["hybrid_hit"]):
            return "hybrid_success"
        if bool(row["oracle_hit"]) and not bool(row["hybrid_hit"]):
            return "recoverable_missed"
        if not bool(row["oracle_hit"]):
            return "candidate_pool_failure"
        return "other"

    merged["s5b_split"] = merged.apply(split, axis=1)

    # Get UAV path and PHOG top1 satellite candidate path from candidate score table.
    cand["rank_num"] = pd.to_numeric(cand["candidate_pool_rank"], errors="coerce")
    phog_top = cand.sort_values(["token_str", "rank_num"], kind="mergesort").groupby("token_str").head(1)

    path_cols = ["uav_image_path", "candidate_image_path", "tile_id", "eval_error_m"]
    available_cols = ["token_str"] + [c for c in path_cols if c in phog_top.columns]
    merged = merged.merge(
        phog_top[available_cols],
        on="token_str",
        how="left",
        suffixes=("", "_phog_top1"),
    )

    rows = []
    for _, row in merged.iterrows():
        uav_path = resolve_path(row.get("uav_image_path"))
        sat_path = resolve_path(row.get("candidate_image_path"))

        uav_rgb = read_rgb(uav_path)
        sat_rgb = read_rgb(sat_path)

        u = image_metrics(uav_rgb, args.resize_long)
        s = image_metrics(sat_rgb, args.resize_long)

        out = row.to_dict()
        for k, v in u.items():
            out[f"uav_{k}"] = v
        for k, v in s.items():
            out[f"sat_top1_{k}"] = v

        out["uav_sat_lab_l_diff"] = abs(u["mean_l"] - s["mean_l"]) if np.isfinite(u["mean_l"]) and np.isfinite(s["mean_l"]) else np.nan
        out["uav_sat_lab_a_diff"] = abs(u["mean_a"] - s["mean_a"]) if np.isfinite(u["mean_a"]) and np.isfinite(s["mean_a"]) else np.nan
        out["uav_sat_lab_b_diff"] = abs(u["mean_b"] - s["mean_b"]) if np.isfinite(u["mean_b"]) and np.isfinite(s["mean_b"]) else np.nan
        out["uav_sat_edge_density_ratio"] = (
            u["edge_density"] / (s["edge_density"] + 1e-9)
            if np.isfinite(u["edge_density"]) and np.isfinite(s["edge_density"])
            else np.nan
        )
        out["uav_sat_green_ratio_diff"] = (
            u["green_ratio"] - s["green_ratio"]
            if np.isfinite(u["green_ratio"]) and np.isfinite(s["green_ratio"])
            else np.nan
        )
        out["uav_sat_warmth_diff"] = (
            u["warmth_r_minus_b"] - s["warmth_r_minus_b"]
            if np.isfinite(u["warmth_r_minus_b"]) and np.isfinite(s["warmth_r_minus_b"])
            else np.nan
        )

        rows.append(out)

    diag = pd.DataFrame(rows)
    diag["token_int"] = diag["token"].astype(int)
    diag = diag.sort_values("token_int")

    split_counts = (
        diag["s5b_split"]
        .value_counts()
        .rename_axis("s5b_split")
        .reset_index(name="count")
    )
    split_counts["rate"] = split_counts["count"] / len(diag)

    group_counts = (
        diag.groupby(["s5b_split", "failure_group"])
        .size()
        .reset_index(name="count")
        .sort_values(["s5b_split", "count"], ascending=[True, False])
    )

    token_lists = {
        "all_73": token_list_str(diag["token"]),
        "hybrid_success": token_list_str(diag[diag["s5b_split"] == "hybrid_success"]["token"]),
        "recoverable_missed": token_list_str(diag[diag["s5b_split"] == "recoverable_missed"]["token"]),
        "candidate_pool_failure": token_list_str(diag[diag["s5b_split"] == "candidate_pool_failure"]["token"]),
    }

    diag_out = dirs["metadata"] / f"s5b0_visual_domain_diagnostics{suffix}.csv"
    split_out = dirs["metadata"] / f"s5b0_token_split_counts{suffix}.csv"
    group_out = dirs["metadata"] / f"s5b0_split_by_failure_group{suffix}.csv"
    token_json = dirs["metadata"] / f"s5b0_token_lists{suffix}.json"
    report_out = dirs["reports"] / f"s5b0_failure_token_split_and_visual_diagnostics_summary{suffix}.json"

    diag.to_csv(diag_out, index=False)
    split_counts.to_csv(split_out, index=False)
    group_counts.to_csv(group_out, index=False)

    with open(token_json, "w") as f:
        json.dump(token_lists, f, indent=2)

    fig1 = dirs["figures"] / f"s5b0_split_counts{suffix}.png"
    plt.figure(figsize=(8, 5))
    plt.bar(split_counts["s5b_split"], split_counts["count"])
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("Token count")
    plt.title("S5B.0 token split after hybrid LightGlue")
    plt.tight_layout()
    plt.savefig(fig1, dpi=180)
    plt.close()

    plot_box(
        diag,
        "uav_green_ratio",
        dirs["figures"] / f"s5b0_uav_green_ratio_by_split{suffix}.png",
        "UAV green ratio by S5B split",
    )
    plot_box(
        diag,
        "sat_top1_green_ratio",
        dirs["figures"] / f"s5b0_sat_top1_green_ratio_by_split{suffix}.png",
        "Satellite PHOG-top1 green ratio by S5B split",
    )
    plot_box(
        diag,
        "uav_edge_density",
        dirs["figures"] / f"s5b0_uav_edge_density_by_split{suffix}.png",
        "UAV edge density by S5B split",
    )
    plot_box(
        diag,
        "sat_top1_edge_density",
        dirs["figures"] / f"s5b0_sat_top1_edge_density_by_split{suffix}.png",
        "Satellite PHOG-top1 edge density by S5B split",
    )
    plot_box(
        diag,
        "uav_orientation_peak_ratio",
        dirs["figures"] / f"s5b0_uav_orientation_peak_ratio_by_split{suffix}.png",
        "UAV orientation dominance by S5B split",
    )
    plot_box(
        diag,
        "uav_sat_lab_l_diff",
        dirs["figures"] / f"s5b0_uav_sat_lab_l_diff_by_split{suffix}.png",
        "UAV-vs-satellite L-channel difference by S5B split",
    )
    plot_scatter(
        diag,
        "uav_green_ratio",
        "uav_edge_density",
        dirs["figures"] / f"s5b0_uav_green_vs_edge_density{suffix}.png",
        "UAV green ratio vs edge density",
    )
    plot_scatter(
        diag,
        "uav_orientation_peak_ratio",
        "uav_edge_density",
        dirs["figures"] / f"s5b0_orientation_vs_edge_density{suffix}.png",
        "Orientation dominance vs edge density",
    )

    report = {
        "stage": "S5B.0_failure_token_split_and_visual_diagnostics",
        "run_name": args.run_name,
        "num_tokens": int(len(diag)),
        "split_counts": split_counts.to_dict(orient="records"),
        "token_lists": token_lists,
        "outputs": {
            "diagnostics_csv": str(diag_out),
            "split_counts_csv": str(split_out),
            "split_by_failure_group_csv": str(group_out),
            "token_lists_json": str(token_json),
            "summary_json": str(report_out),
            "split_counts_figure": str(fig1),
        },
        "locked_rule": "reference/error columns are used only to define post-ranking outcome groups; visual metrics are computed from images",
    }

    with open(report_out, "w") as f:
        json.dump(report, f, indent=2)

    print("S5B.0 token split and visual-domain diagnostics complete")
    print("-------------------------------------------------------")
    print(f"Tokens analyzed:        {len(diag)}")
    print()
    print("Split counts:")
    print(split_counts.to_string(index=False))
    print()
    print("Failure-group composition:")
    print(group_counts.to_string(index=False))
    print()
    print("Token lists:")
    for k, v in token_lists.items():
        print(f"{k}: {v}")
    print()
    print(f"Diagnostics CSV:        {diag_out}")
    print(f"Token lists JSON:       {token_json}")
    print(f"Summary JSON:           {report_out}")
    print(f"Figures dir:            {dirs['figures']}")
    print()
    print("Next: inspect whether candidate_pool_failure has higher vegetation, lower satellite edge density, or stronger orientation dominance.")


if __name__ == "__main__":
    main()
