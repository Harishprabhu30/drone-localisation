#!/usr/bin/env python3
"""
S4C.6C — Failure-group analysis.

Purpose:
  Analyze why S4C map-matching still fails after PHOG + luma-LSD + gating + rotation tests.

Groups:
  - stable_success
  - lsd_rescue
  - lsd_destroyed_phog_success
  - selection_failure_correct_in_pool
  - weak_pool_near_candidate
  - candidate_pool_failure

Uses:
  - S4C.4C raw luma-LSD top50 scores
  - S4C.4C summary by profile
  - S4C.4D gated result, optional
  - S4C.5A junction frontend manifest, optional
  - S4C.6B rotation result, optional

Important:
  - This is analysis only.
  - Reference error is used only after ranking for diagnosis.

Command USed:
export PYTHONPATH=$PWD/src

python scripts/satloc/s4c/s4c_6c_failure_group_analysis.py \
  --threshold-m 40 \
  --representatives-per-group 4
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
from pathlib import Path
from typing import Any, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUT_ROOT = Path("outputs/satloc")

DEFAULT_S4C4C_DIR = OUT_ROOT / (
    "metadata/s4c_macrocontour_phog_chamfer/"
    "s4c4_vector_skeleton/s4c4c_luma_lsd_rerank"
)

DEFAULT_S4C4D_SELECTED = OUT_ROOT / (
    "metadata/s4c_macrocontour_phog_chamfer/"
    "s4c4_vector_skeleton/s4c4d_confidence_gated_rerank/"
    "s4c4d_traj01_selected_by_profile.csv"
)

DEFAULT_S4C5A_MANIFEST = OUT_ROOT / (
    "metadata/s4c_macrocontour_phog_chamfer/"
    "s4c5_junction_lsd/s4c5a_junction_lsd_frontend_manifest.csv"
)

DEFAULT_S4C6B_DIR = OUT_ROOT / (
    "metadata/s4c_macrocontour_phog_chamfer/"
    "s4c6_rotation_aware_lsd"
)

OUT_META_DIR = OUT_ROOT / "metadata/s4c_macrocontour_phog_chamfer/s4c6_failure_analysis"
OUT_REPORT_DIR = OUT_ROOT / "reports/s4c_macrocontour_phog_chamfer/s4c6_failure_analysis"
OUT_FIG_DIR = OUT_ROOT / "figures/s4c_macrocontour_phog_chamfer/s4c6_failure_analysis"
OUT_ASSET_DIR = OUT_ROOT / "assets/s4c6_failure_analysis"


def safe_float(x: Any, default: float = np.nan) -> float:
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    return obj


def extract_token_from_path(path: Path) -> Optional[int]:
    m = re.search(r"token(\d+)_", path.name)
    if not m:
        return None
    return int(m.group(1))


def collect_raw_csvs(raw_dir: Path) -> list[Path]:
    files = sorted(raw_dir.glob("s4c4c_token*_luma_lsd_raw_lsd_scores_top*.csv"))
    if not files:
        raise FileNotFoundError(f"No S4C.4C raw files found in {raw_dir}")
    return files


def first_rank_under(df: pd.DataFrame, rank_col: str, threshold_m: float) -> Optional[int]:
    if "center_error_m" not in df.columns:
        return None
    sub = df[np.isfinite(df["center_error_m"]) & (df["center_error_m"] <= threshold_m)]
    if len(sub) == 0:
        return None
    return int(sub[rank_col].min())


def load_token_raw_features(raw_csv: Path) -> dict[str, Any]:
    token = extract_token_from_path(raw_csv)
    if token is None:
        raise ValueError(f"Cannot parse token from {raw_csv}")

    df = pd.read_csv(raw_csv)

    if "phog_rank" not in df.columns:
        if "rank" in df.columns:
            df["phog_rank"] = df["rank"].astype(int)
        else:
            df["phog_rank"] = np.arange(1, len(df) + 1)

    if "lsd_rank" not in df.columns:
        df["lsd_rank"] = df["lsd_best_score"].rank(method="min", ascending=True).astype(int)

    phog = df.sort_values("phog_rank").iloc[0]
    lsd = df.sort_values(["lsd_rank", "phog_rank"]).iloc[0]

    best_top50_error = safe_float(df["center_error_m"].min())
    best_top50_row = df.sort_values("center_error_m").iloc[0]

    out = {
        "token": int(token),
        "raw_csv": str(raw_csv),
        "candidate_count": int(len(df)),

        "phog_top1_tile_id": int(phog["tile_id"]),
        "phog_top1_error_m": safe_float(phog.get("center_error_m")),
        "phog_top1_lsd_rank": int(phog["lsd_rank"]),
        "phog_top1_lsd_score": safe_float(phog.get("lsd_best_score")),
        "phog_top1_lsd_shift_mag_px": safe_float(phog.get("lsd_shift_mag_px")),
        "phog_top1_lsd_boundary": bool(phog.get("lsd_at_shift_boundary", False)),

        "lsd_top1_tile_id": int(lsd["tile_id"]),
        "lsd_top1_error_m": safe_float(lsd.get("center_error_m")),
        "lsd_top1_phog_rank": int(lsd["phog_rank"]),
        "lsd_top1_lsd_rank": int(lsd["lsd_rank"]),
        "lsd_top1_lsd_score": safe_float(lsd.get("lsd_best_score")),
        "lsd_top1_shift_mag_px": safe_float(lsd.get("lsd_shift_mag_px")),
        "lsd_top1_boundary": bool(lsd.get("lsd_at_shift_boundary", False)),

        "oracle_best_top50_error_m": best_top50_error,
        "oracle_best_top50_tile_id": int(best_top50_row["tile_id"]),
        "oracle_best_top50_phog_rank": int(best_top50_row["phog_rank"]),
        "oracle_best_top50_lsd_rank": int(best_top50_row["lsd_rank"]),
        "oracle_best_top50_lsd_score": safe_float(best_top50_row.get("lsd_best_score")),

        "first_phog_rank_under_20m": first_rank_under(df, "phog_rank", 20.0),
        "first_phog_rank_under_40m": first_rank_under(df, "phog_rank", 40.0),
        "first_phog_rank_under_60m": first_rank_under(df, "phog_rank", 60.0),

        "first_lsd_rank_under_20m": first_rank_under(df, "lsd_rank", 20.0),
        "first_lsd_rank_under_40m": first_rank_under(df, "lsd_rank", 40.0),
        "first_lsd_rank_under_60m": first_rank_under(df, "lsd_rank", 60.0),
    }

    if "phog_score" in df.columns:
        p = df.sort_values("phog_rank").reset_index(drop=True)
        if len(p) >= 2:
            out["phog_margin_top1_top2"] = safe_float(p.iloc[0]["phog_score"]) - safe_float(p.iloc[1]["phog_score"])
        else:
            out["phog_margin_top1_top2"] = np.nan

    return out


def classify_failure(row: pd.Series, threshold_m: float = 40.0) -> str:
    phog = safe_float(row["phog_top1_error_m"])
    lsd = safe_float(row["lsd_top1_error_m"])
    oracle = safe_float(row["oracle_best_top50_error_m"])

    phog_ok = np.isfinite(phog) and phog <= threshold_m
    lsd_ok = np.isfinite(lsd) and lsd <= threshold_m
    oracle_ok = np.isfinite(oracle) and oracle <= threshold_m
    oracle_near = np.isfinite(oracle) and oracle <= 100.0

    if phog_ok and lsd_ok:
        return "stable_success"

    if (not phog_ok) and lsd_ok:
        return "lsd_rescue"

    if phog_ok and (not lsd_ok):
        return "lsd_destroyed_phog_success"

    if (not phog_ok) and (not lsd_ok) and oracle_ok:
        return "selection_failure_correct_in_pool"

    if (not phog_ok) and (not lsd_ok) and oracle_near:
        return "weak_pool_near_candidate"

    return "candidate_pool_failure"


def load_gate_result(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path)
    if "profile" not in df.columns:
        return pd.DataFrame()

    pref = df[df["profile"] == "gate_phog_anchor_else_lsd"].copy()
    if len(pref) == 0:
        return pd.DataFrame()

    keep = pref[["token", "center_error_m", "selection_reason", "phog_rank", "lsd_rank", "lsd_best_score"]].copy()
    keep = keep.rename(columns={
        "center_error_m": "gate_error_m",
        "selection_reason": "gate_selection_reason",
        "phog_rank": "gate_phog_rank",
        "lsd_rank": "gate_lsd_rank",
        "lsd_best_score": "gate_lsd_score",
    })
    return keep


def load_rotation_result(rot_dir: Path) -> pd.DataFrame:
    if not rot_dir.exists():
        return pd.DataFrame()

    files = sorted(rot_dir.glob("s4c6b_*_summary_by_token_profile.csv"))
    if not files:
        return pd.DataFrame()

    rows = []

    for p in files:
        try:
            df = pd.read_csv(p)
        except Exception:
            continue

        for token, g in df.groupby("token"):
            best = g.sort_values("top1_error_m").iloc[0]
            rows.append({
                "token": int(token),
                "rot_best_profile": best.get("profile"),
                "rot_best_error_m": safe_float(best.get("top1_error_m")),
                "rot_best_angle_deg": safe_float(best.get("top1_angle_deg")),
                "rot_summary_csv": str(p),
            })

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    out = out.sort_values("rot_best_error_m").groupby("token").head(1).reset_index(drop=True)
    return out


def load_junction_manifest(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path)
    if len(df) == 0:
        return pd.DataFrame()

    metrics = [
        "junction_count",
        "L_count",
        "T_count",
        "X_count",
        "junction_salience_sum",
        "junction_type_entropy",
        "luma_lsd_line_count",
        "luma_lsd_total_line_length_px",
    ]

    keep_roles = {
        "uav_query": "uav",
        "gt_or_nearest_eval_only": "gt",
        "phog_top1": "phog",
        "lsd_top1_from_s4c4c": "lsd",
        "oracle_best_topn_eval_only": "oracle",
    }

    rows = []

    for token, g in df.groupby("token"):
        row = {"token": int(token)}

        for role, prefix in keep_roles.items():
            sub = g[g["role"] == role]
            if len(sub) == 0:
                continue

            s = sub.iloc[0]
            for m in metrics:
                if m in s:
                    row[f"{prefix}_{m}"] = safe_float(s[m])

        rows.append(row)

    return pd.DataFrame(rows)


def add_heuristic_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for col in ["uav_junction_count", "uav_luma_lsd_line_count", "uav_luma_lsd_total_line_length_px"]:
        if col not in out.columns:
            out[col] = np.nan

    q25_junction = out["uav_junction_count"].quantile(0.25)
    q25_lines = out["uav_luma_lsd_line_count"].quantile(0.25)

    out["low_uav_junction_structure"] = out["uav_junction_count"] <= q25_junction
    out["low_uav_line_structure"] = out["uav_luma_lsd_line_count"] <= q25_lines

    out["correct_candidate_in_pool"] = out["oracle_best_top50_error_m"] <= 40.0
    out["near_candidate_in_pool"] = out["oracle_best_top50_error_m"] <= 100.0
    out["lsd_large_shift"] = out["lsd_top1_shift_mag_px"] >= 40.0
    out["lsd_boundary_shift"] = out["lsd_top1_boundary"].astype(bool)

    out["lsd_damage_m"] = out["lsd_top1_error_m"] - out["phog_top1_error_m"]
    out["lsd_improvement_m"] = out["phog_top1_error_m"] - out["lsd_top1_error_m"]
    out["selection_gap_to_oracle_m"] = out["lsd_top1_error_m"] - out["oracle_best_top50_error_m"]

    return out


def select_representatives(df: pd.DataFrame, per_group: int = 4) -> pd.DataFrame:
    reps = []

    for group, g in df.groupby("failure_group"):
        if group == "lsd_rescue":
            chosen = g.sort_values("lsd_improvement_m", ascending=False).head(per_group)
        elif group == "lsd_destroyed_phog_success":
            chosen = g.sort_values("lsd_damage_m", ascending=False).head(per_group)
        elif group == "selection_failure_correct_in_pool":
            chosen = g.sort_values("selection_gap_to_oracle_m", ascending=False).head(per_group)
        elif group == "candidate_pool_failure":
            chosen = g.sort_values("oracle_best_top50_error_m", ascending=False).head(per_group)
        elif group == "weak_pool_near_candidate":
            chosen = g.sort_values("oracle_best_top50_error_m").head(per_group)
        else:
            chosen = g.sort_values("lsd_top1_error_m").head(per_group)

        reps.append(chosen)

    if not reps:
        return pd.DataFrame()

    return pd.concat(reps, ignore_index=True)


def copy_if_exists(src: Path, dst_dir: Path, prefix: str) -> Optional[str]:
    if not src.exists():
        return None

    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / f"{prefix}_{src.name}"
    shutil.copy2(src, dst)
    return str(dst)


def copy_representative_assets(rep_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for _, row in rep_df.iterrows():
        token = int(row["token"])
        group = str(row["failure_group"])
        prefix = f"{group}_token{token:04d}"

        copied = []

        candidates = [
            OUT_ROOT / f"figures/s4c_macrocontour_phog_chamfer/s4c4_vector_skeleton/s4c4c_luma_lsd_rerank/s4c4c_token{token:04d}_luma_lsd_panel_top5.png",
            OUT_ROOT / f"figures/s4c_macrocontour_phog_chamfer/s4c5_junction_lsd/s4c5a_frontend/s4c5a_token{token:04d}_junction_lsd_frontend.png",
            OUT_ROOT / f"figures/s4c_macrocontour_phog_chamfer/s4c6_rotation_aware_lsd/s4c6b_token{token:04d}_fixed_panel_top5.png",
            OUT_ROOT / f"figures/s4c_macrocontour_phog_chamfer/s4c6_rotation_aware_lsd/s4c6b_token{token:04d}_orientation_prior_panel_top5.png",
        ]

        for src in candidates:
            out = copy_if_exists(src, OUT_ASSET_DIR / group, prefix)
            if out is not None:
                copied.append(out)

        rr = row.to_dict()
        rr["copied_assets"] = json.dumps(copied)
        rows.append(rr)

    return pd.DataFrame(rows)


def plot_group_counts(df: pd.DataFrame, out_path: Path) -> None:
    counts = df["failure_group"].value_counts().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(counts.index.astype(str), counts.values)
    ax.set_title("S4C.6C failure group counts")
    ax.set_ylabel("frame count")
    ax.set_xticklabels(counts.index.astype(str), rotation=35, ha="right")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def plot_phog_vs_lsd(df: pd.DataFrame, out_path: Path, threshold: float) -> None:
    fig, ax = plt.subplots(figsize=(7, 7))

    groups = sorted(df["failure_group"].unique())
    for group in groups:
        sub = df[df["failure_group"] == group]
        ax.scatter(sub["phog_top1_error_m"], sub["lsd_top1_error_m"], label=group, alpha=0.8)

    ax.axhline(threshold, linestyle="--")
    ax.axvline(threshold, linestyle="--")
    ax.set_xscale("symlog", linthresh=threshold)
    ax.set_yscale("symlog", linthresh=threshold)
    ax.set_xlabel("PHOG top1 error [m]")
    ax.set_ylabel("luma-LSD top1 error [m]")
    ax.set_title("PHOG vs luma-LSD top1 error")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def plot_oracle_gap(df: pd.DataFrame, out_path: Path) -> None:
    d = df.copy()
    d = d.sort_values("selection_gap_to_oracle_m", ascending=False).head(40)

    fig, ax = plt.subplots(figsize=(13, 5))
    x = np.arange(len(d))
    ax.bar(x, d["selection_gap_to_oracle_m"])
    ax.set_xticks(x)
    ax.set_xticklabels(d["token"].astype(str), rotation=90)
    ax.set_ylabel("LSD top1 error - oracle best top50 error [m]")
    ax.set_title("Selection gap to oracle candidate")
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def plot_structure_vs_error(df: pd.DataFrame, out_path: Path) -> None:
    if "uav_junction_count" not in df.columns:
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    groups = sorted(df["failure_group"].unique())
    for group in groups:
        sub = df[df["failure_group"] == group]
        ax.scatter(sub["uav_junction_count"], sub["lsd_top1_error_m"], label=group, alpha=0.8)

    ax.set_yscale("symlog", linthresh=40)
    ax.set_xlabel("UAV junction count")
    ax.set_ylabel("luma-LSD top1 error [m]")
    ax.set_title("Structure availability vs localization error")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def summarize_groups(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for group, g in df.groupby("failure_group"):
        rows.append({
            "failure_group": group,
            "count": int(len(g)),
            "rate": float(len(g) / max(1, len(df))),
            "median_phog_error_m": float(g["phog_top1_error_m"].median()),
            "median_lsd_error_m": float(g["lsd_top1_error_m"].median()),
            "median_oracle_top50_error_m": float(g["oracle_best_top50_error_m"].median()),
            "median_selection_gap_to_oracle_m": float(g["selection_gap_to_oracle_m"].median()),
            "low_uav_line_structure_rate": float(g["low_uav_line_structure"].mean()),
            "low_uav_junction_structure_rate": float(g["low_uav_junction_structure"].mean()),
            "lsd_large_shift_rate": float(g["lsd_large_shift"].mean()),
            "lsd_boundary_shift_rate": float(g["lsd_boundary_shift"].mean()),
        })

    out = pd.DataFrame(rows)
    return out.sort_values("count", ascending=False)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--s4c4c-dir", default=str(DEFAULT_S4C4C_DIR))
    parser.add_argument("--s4c4d-selected", default=str(DEFAULT_S4C4D_SELECTED))
    parser.add_argument("--s4c5a-manifest", default=str(DEFAULT_S4C5A_MANIFEST))
    parser.add_argument("--s4c6b-dir", default=str(DEFAULT_S4C6B_DIR))
    parser.add_argument("--threshold-m", type=float, default=40.0)
    parser.add_argument("--representatives-per-group", type=int, default=4)

    args = parser.parse_args()

    OUT_META_DIR.mkdir(parents=True, exist_ok=True)
    OUT_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_FIG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_ASSET_DIR.mkdir(parents=True, exist_ok=True)

    raw_dir = Path(args.s4c4c_dir)
    raw_csvs = collect_raw_csvs(raw_dir)

    print("S4C.6C failure-group analysis")
    print("-----------------------------")
    print(f"S4C.4C raw dir: {raw_dir}")
    print(f"Raw token files: {len(raw_csvs)}")
    print(f"Threshold: {args.threshold_m} m")
    print("")

    base_rows = []
    for raw_csv in raw_csvs:
        try:
            base_rows.append(load_token_raw_features(raw_csv))
        except Exception as exc:
            print(f"[WARN] {raw_csv.name}: {exc}")

    df = pd.DataFrame(base_rows)

    if len(df) == 0:
        raise RuntimeError("No token features loaded.")

    df["failure_group"] = df.apply(lambda r: classify_failure(r, threshold_m=args.threshold_m), axis=1)

    gate_df = load_gate_result(Path(args.s4c4d_selected))
    if len(gate_df) > 0:
        df = df.merge(gate_df, on="token", how="left")

    junction_df = load_junction_manifest(Path(args.s4c5a_manifest))
    if len(junction_df) > 0:
        df = df.merge(junction_df, on="token", how="left")

    rot_df = load_rotation_result(Path(args.s4c6b_dir))
    if len(rot_df) > 0:
        df = df.merge(rot_df, on="token", how="left")

    df = add_heuristic_flags(df)

    group_summary = summarize_groups(df)
    representatives = select_representatives(df, per_group=args.representatives_per_group)
    representatives_assets = copy_representative_assets(representatives)

    token_csv = OUT_META_DIR / "s4c6c_failure_groups_by_token.csv"
    group_csv = OUT_META_DIR / "s4c6c_failure_group_summary.csv"
    rep_csv = OUT_META_DIR / "s4c6c_representative_tokens.csv"
    rep_assets_csv = OUT_META_DIR / "s4c6c_representative_assets.csv"

    df.to_csv(token_csv, index=False)
    group_summary.to_csv(group_csv, index=False)
    representatives.to_csv(rep_csv, index=False)
    representatives_assets.to_csv(rep_assets_csv, index=False)

    group_plot = OUT_FIG_DIR / "s4c6c_failure_group_counts.png"
    scatter_plot = OUT_FIG_DIR / "s4c6c_phog_vs_lsd_error.png"
    oracle_gap_plot = OUT_FIG_DIR / "s4c6c_selection_gap_to_oracle.png"
    structure_plot = OUT_FIG_DIR / "s4c6c_structure_vs_error.png"

    plot_group_counts(df, group_plot)
    plot_phog_vs_lsd(df, scatter_plot, threshold=args.threshold_m)
    plot_oracle_gap(df, oracle_gap_plot)
    plot_structure_vs_error(df, structure_plot)

    summary_json = OUT_REPORT_DIR / "s4c6c_failure_group_analysis_summary.json"

    summary = {
        "stage": "S4C.6C_failure_group_analysis",
        "num_tokens": int(len(df)),
        "threshold_m": args.threshold_m,
        "token_csv": str(token_csv),
        "group_csv": str(group_csv),
        "representative_tokens_csv": str(rep_csv),
        "representative_assets_csv": str(rep_assets_csv),
        "asset_dir": str(OUT_ASSET_DIR),
        "figures": {
            "group_counts": str(group_plot),
            "phog_vs_lsd": str(scatter_plot),
            "selection_gap_to_oracle": str(oracle_gap_plot),
            "structure_vs_error": str(structure_plot),
        },
        "group_summary": group_summary.to_dict(orient="records"),
        "notes": [
            "Failure groups are based on PHOG top1, luma-LSD top1, and oracle best PHOG top50 error.",
            "Oracle best top50 is used only for analysis, not scoring.",
            "Representative existing panels are copied into the asset folder for README/report creation.",
        ],
    }

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(json_safe(summary), f, indent=2)

    print("S4C.6C complete")
    print("----------------")
    print(f"Token CSV:          {token_csv}")
    print(f"Group CSV:          {group_csv}")
    print(f"Representatives:    {rep_csv}")
    print(f"Representative img: {rep_assets_csv}")
    print(f"Assets:             {OUT_ASSET_DIR}")
    print(f"Summary JSON:       {summary_json}")
    print(f"Figures:            {OUT_FIG_DIR}")
    print("")
    print("Failure group summary")
    print("---------------------")
    print(group_summary.to_string(index=False))
    print("")
    print("Representative tokens")
    print("---------------------")
    cols = [
        "failure_group",
        "token",
        "phog_top1_error_m",
        "lsd_top1_error_m",
        "oracle_best_top50_error_m",
        "selection_gap_to_oracle_m",
        "low_uav_line_structure",
        "low_uav_junction_structure",
        "lsd_large_shift",
        "lsd_boundary_shift",
    ]
    cols = [c for c in cols if c in representatives.columns]
    print(representatives[cols].to_string(index=False))


if __name__ == "__main__":
    main()
