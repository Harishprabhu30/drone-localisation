#!/usr/bin/env python3
"""
S4C.6A — parameter sensitivity sweep for PHOG + luma-LSD reranker.

Purpose:
  Run a controlled YAML-defined sweep over current S4C.4C luma-LSD reranking
  parameters.

Important:
  - PHOG top-N candidate pool comes from existing S4C.1 ranked CSVs.
  - No reference coordinate is used in scoring.
  - center_error_m is used only after ranking for evaluation.
  - This is diagnostic/tuning on selected subset, not final held-out result.

Code Used:
export PYTHONPATH=$PWD/src

python scripts/satloc/s4c/s4c_6a_parameter_sensitivity_sweep.py \
  --config configs/satloc/s4c6a_parameter_sweep.yaml \
  --tokens 1,40,60,90,100,129,166,269,516,905 \
  --experiments all

- Test on all 73 frames selected from traj01 as subset:
export PYTHONPATH=$PWD/src

python scripts/satloc/s4c/s4c_6a_parameter_sensitivity_sweep.py \
  --config configs/satloc/s4c6a_parameter_sweep.yaml \
  --tokens all \
  --experiments baseline_current,dense_low_threshold_short_lines_shift64,thick_line_canvas,blur5_thick_symmetric,lsd_short_dense_lines,larger_shift_tolerance,stronger_symmetric_match
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
import sys
import time
from argparse import Namespace
from pathlib import Path
from typing import Any, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import yaml
except Exception as exc:
    raise RuntimeError("PyYAML is required. Install with: pip install pyyaml") from exc


THIS_DIR = Path(__file__).resolve().parent
S4C4C_PATH = THIS_DIR / "s4c_4c_luma_lsd_phog_top50_reranker.py"

if not S4C4C_PATH.exists():
    raise FileNotFoundError(f"Missing helper: {S4C4C_PATH}")

spec = importlib.util.spec_from_file_location("s4c4c_helpers", S4C4C_PATH)
s4c4c = importlib.util.module_from_spec(spec)
sys.modules["s4c4c_helpers"] = s4c4c
assert spec.loader is not None
spec.loader.exec_module(s4c4c)


OUT_ROOT = Path("outputs/satloc")
OUT_META_DIR = OUT_ROOT / "metadata/s4c_macrocontour_phog_chamfer/s4c6_parameter_sweep"
OUT_REPORT_DIR = OUT_ROOT / "reports/s4c_macrocontour_phog_chamfer/s4c6_parameter_sweep"
OUT_FIG_DIR = OUT_ROOT / "figures/s4c_macrocontour_phog_chamfer/s4c6_parameter_sweep"


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


def load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_tokens(text: str, ranked_dir: Path) -> list[int]:
    text = str(text).strip()

    if text.lower() == "all":
        tokens = set()
        for p in ranked_dir.glob("s4c1_token*_ranked.csv"):
            m = re.search(r"s4c1_token(\d+)_", p.name)
            if m:
                tokens.add(int(m.group(1)))
        if not tokens:
            raise FileNotFoundError(f"No S4C.1 ranked CSVs found in {ranked_dir}")
        return sorted(tokens)

    return sorted([int(x.strip()) for x in text.split(",") if x.strip()])


def filter_experiments(experiments: list[dict[str, Any]], names_text: str) -> list[dict[str, Any]]:
    names_text = str(names_text).strip()
    if names_text.lower() in ["all", "*", ""]:
        return experiments

    keep = {x.strip() for x in names_text.split(",") if x.strip()}
    out = [e for e in experiments if e["name"] in keep]

    missing = keep - {e["name"] for e in out}
    if missing:
        raise ValueError(f"Experiment names not found in YAML: {sorted(missing)}")

    return out


def merge_settings(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    out.update(overrides or {})
    return out


def experiment_args(settings: dict[str, Any]) -> Namespace:
    return Namespace(**settings)


def first_rank_under(df: pd.DataFrame, rank_col: str, threshold_m: float) -> Optional[int]:
    if "center_error_m" not in df.columns:
        return None

    sub = df[np.isfinite(df["center_error_m"]) & (df["center_error_m"] <= threshold_m)]
    if len(sub) == 0:
        return None

    return int(sub[rank_col].min())


def phog_margin_for_token(df: pd.DataFrame) -> float:
    phog = df.sort_values("phog_rank").reset_index(drop=True)

    if len(phog) < 2:
        return np.nan

    s1 = safe_float(phog.iloc[0].get("phog_score"))
    s2 = safe_float(phog.iloc[1].get("phog_score"))

    if not np.isfinite(s1) or not np.isfinite(s2):
        return np.nan

    return float(s1 - s2)


def rank_for_profile(
    df: pd.DataFrame,
    profile: str,
    phog_margin_q75: float,
    phog_top_n: int,
) -> tuple[pd.DataFrame, str]:
    out = df.copy()

    if profile == "phog_only":
        out = out.sort_values("phog_rank").reset_index(drop=True)
        out["profile_rank"] = np.arange(1, len(out) + 1)
        return out, "phog_rank"

    if profile == "lsd_only":
        out = out.sort_values(["lsd_rank", "phog_rank"]).reset_index(drop=True)
        out["profile_rank"] = np.arange(1, len(out) + 1)
        return out, "lsd_rank"

    if profile == "lsd_strong":
        boundary_penalty = out["lsd_at_shift_boundary"].astype(bool).astype(float) * float(phog_top_n)

        out["profile_score"] = (
            0.50 * out["phog_rank"].astype(float)
            + 1.00 * out["lsd_rank"].astype(float)
            + 0.15 * out["lsd_shift_rank"].astype(float)
            + 0.10 * out["lsd_basin_rank"].astype(float)
            + 0.75 * boundary_penalty
        )

        out = out.sort_values(["profile_score", "phog_rank", "lsd_rank"]).reset_index(drop=True)
        out["profile_rank"] = np.arange(1, len(out) + 1)
        return out, "lsd_strong_score"

    if profile == "gate_phog_anchor_else_lsd":
        margin = phog_margin_for_token(out)
        use_phog = np.isfinite(margin) and margin >= phog_margin_q75

        if use_phog:
            out = out.sort_values("phog_rank").reset_index(drop=True)
            reason = "phog_anchor"
        else:
            out = out.sort_values(["lsd_rank", "phog_rank"]).reset_index(drop=True)
            reason = "fallback_lsd"

        out["profile_rank"] = np.arange(1, len(out) + 1)
        out["gate_reason"] = reason
        return out, reason

    raise ValueError(f"Unknown profile: {profile}")


def summarize_token_profile(
    experiment_name: str,
    token: int,
    profile: str,
    ranked: pd.DataFrame,
    ranking_reason: str,
) -> dict[str, Any]:
    top1 = ranked.iloc[0]
    top10 = ranked.head(10)

    err = safe_float(top1.get("center_error_m"))

    return {
        "experiment": experiment_name,
        "token": int(token),
        "profile": profile,
        "ranking_reason": ranking_reason,

        "top1_tile_id": int(top1["tile_id"]),
        "top1_error_m": err,
        "top1_phog_rank": int(top1["phog_rank"]),
        "top1_lsd_rank": int(top1["lsd_rank"]),
        "top1_lsd_best_score": safe_float(top1.get("lsd_best_score")),
        "top1_lsd_shift_mag_px": safe_float(top1.get("lsd_shift_mag_px")),
        "top1_lsd_boundary": bool(top1.get("lsd_at_shift_boundary", False)),

        "best_top10_error_m": safe_float(top10["center_error_m"].min()),
        "first_rank_under_20m": first_rank_under(ranked, "profile_rank", 20.0),
        "first_rank_under_40m": first_rank_under(ranked, "profile_rank", 40.0),
        "first_rank_under_60m": first_rank_under(ranked, "profile_rank", 60.0),

        "top1_under_20m": bool(np.isfinite(err) and err <= 20.0),
        "top1_under_40m": bool(np.isfinite(err) and err <= 40.0),
        "top1_under_60m": bool(np.isfinite(err) and err <= 60.0),
        "top1_under_100m": bool(np.isfinite(err) and err <= 100.0),

        "top10_under_20m": bool((top10["center_error_m"] <= 20.0).any()),
        "top10_under_40m": bool((top10["center_error_m"] <= 40.0).any()),
        "top10_under_60m": bool((top10["center_error_m"] <= 60.0).any()),
    }


def run_experiment(
    exp: dict[str, Any],
    base_settings: dict[str, Any],
    tokens: list[int],
    phog_top_n: int,
    uav_df: pd.DataFrame,
    sat_df: pd.DataFrame,
    ranked_dir: Path,
    filename_index: dict[str, Path],
    fallback_uav_dirs: list[Path],
    fallback_sat_dirs: list[Path],
    save_candidate_scores: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    name = exp["name"]
    overrides = exp.get("overrides", {})
    settings = merge_settings(base_settings, overrides)

    args = experiment_args(settings)

    print("")
    print(f"=== Experiment: {name} ===")
    print(f"Rationale: {exp.get('rationale', '')}")
    print(f"Overrides: {overrides}")

    start = time.time()
    canvas_cache: dict[str, tuple[dict[str, Any], np.ndarray]] = {}

    candidate_rows = []

    for token in tokens:
        try:
            uav_path = s4c4c.find_uav_query_path(
                token=token,
                uav_df=uav_df,
                filename_index=filename_index,
                fallback_uav_dirs=fallback_uav_dirs,
            )

            if uav_path is None:
                print(f"[WARN] token {token}: missing UAV path")
                continue

            candidates = s4c4c.load_phog_candidates(
                token=token,
                ranked_dir=ranked_dir,
                sat_df=sat_df,
                filename_index=filename_index,
                fallback_sat_dirs=fallback_sat_dirs,
                top_n=phog_top_n,
            )

            lsd_df = s4c4c.compute_lsd_alignment_for_token(
                token=token,
                uav_path=Path(uav_path),
                candidates=candidates,
                args=args,
                cache=canvas_cache,
            )

            if len(lsd_df) == 0:
                print(f"[WARN] token {token}: no lsd rows")
                continue

            lsd_df["experiment"] = name
            candidate_rows.append(lsd_df)

            top_phog = lsd_df.sort_values("phog_rank").iloc[0]
            top_lsd = lsd_df.sort_values(["lsd_rank", "phog_rank"]).iloc[0]
            print(
                f"[OK] token {token}: "
                f"PHOG {safe_float(top_phog.get('center_error_m')):.1f}m | "
                f"LSD {safe_float(top_lsd.get('center_error_m')):.1f}m"
            )

        except Exception as exc:
            print(f"[WARN] token {token}: {exc}")

    if not candidate_rows:
        raise RuntimeError(f"No candidate rows produced for experiment {name}")

    candidate_df = pd.concat(candidate_rows, ignore_index=True)

    margins = []
    for token, group in candidate_df.groupby("token"):
        margins.append(phog_margin_for_token(group))

    finite_margins = pd.Series(margins).replace([np.inf, -np.inf], np.nan).dropna()
    phog_margin_q75 = float(finite_margins.quantile(0.75)) if len(finite_margins) else float("inf")

    profiles = [
        "phog_only",
        "lsd_only",
        "lsd_strong",
        "gate_phog_anchor_else_lsd",
    ]

    summary_rows = []
    full_profile_rows = []

    for token, group in candidate_df.groupby("token"):
        for profile in profiles:
            ranked, reason = rank_for_profile(
                group,
                profile=profile,
                phog_margin_q75=phog_margin_q75,
                phog_top_n=phog_top_n,
            )

            summary_rows.append(
                summarize_token_profile(
                    experiment_name=name,
                    token=int(token),
                    profile=profile,
                    ranked=ranked,
                    ranking_reason=reason,
                )
            )

            keep_cols = [
                "experiment",
                "token",
                "tile_id",
                "phog_rank",
                "lsd_rank",
                "center_error_m",
                "lsd_best_score",
                "lsd_shift_mag_px",
                "lsd_at_shift_boundary",
                "profile_rank",
            ]
            tmp = ranked[[c for c in keep_cols if c in ranked.columns]].copy()
            tmp["profile"] = profile
            tmp["ranking_reason"] = reason
            full_profile_rows.append(tmp)

    summary_df = pd.DataFrame(summary_rows)
    profile_df = pd.concat(full_profile_rows, ignore_index=True)

    elapsed_s = time.time() - start

    exp_info = {
        "experiment": name,
        "rationale": exp.get("rationale", ""),
        "overrides": overrides,
        "settings": settings,
        "tokens_completed": int(candidate_df["token"].nunique()),
        "candidate_rows": int(len(candidate_df)),
        "phog_margin_q75": phog_margin_q75,
        "elapsed_s": elapsed_s,
    }

    if save_candidate_scores:
        candidate_path = OUT_META_DIR / "candidate_scores" / f"s4c6a_{name}_candidate_scores.csv"
        candidate_path.parent.mkdir(parents=True, exist_ok=True)
        candidate_df.to_csv(candidate_path, index=False)
        exp_info["candidate_scores_csv"] = str(candidate_path)

    profile_path = OUT_META_DIR / "profile_rankings" / f"s4c6a_{name}_profile_rankings.csv"
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    profile_df.to_csv(profile_path, index=False)
    exp_info["profile_rankings_csv"] = str(profile_path)

    return summary_df, profile_df, exp_info


def aggregate_results(summary_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for (experiment, profile), group in summary_df.groupby(["experiment", "profile"]):
        rows.append(
            {
                "experiment": experiment,
                "profile": profile,
                "num_queries": int(len(group)),
                "top1_under_20m_rate": float(group["top1_under_20m"].mean()),
                "top1_under_40m_rate": float(group["top1_under_40m"].mean()),
                "top1_under_60m_rate": float(group["top1_under_60m"].mean()),
                "top1_under_100m_rate": float(group["top1_under_100m"].mean()),
                "top10_under_40m_rate": float(group["top10_under_40m"].mean()),
                "median_top1_error_m": float(group["top1_error_m"].median()),
                "mean_top1_error_m": float(group["top1_error_m"].mean()),
                "median_best_top10_error_m": float(group["best_top10_error_m"].median()),
                "mean_best_top10_error_m": float(group["best_top10_error_m"].mean()),
                "median_top1_lsd_score": float(group["top1_lsd_best_score"].median()),
                "mean_top1_lsd_score": float(group["top1_lsd_best_score"].mean()),
            }
        )

    out = pd.DataFrame(rows)

    if len(out) > 0:
        out = out.sort_values(
            ["top1_under_40m_rate", "top10_under_40m_rate", "median_top1_error_m"],
            ascending=[False, False, True],
        )

    return out


def render_top_plot(agg: pd.DataFrame, out_path: Path, top_n: int = 20) -> None:
    if len(agg) == 0:
        return

    plot_df = agg.head(top_n).copy()
    labels = (plot_df["experiment"] + " / " + plot_df["profile"]).tolist()
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 6.2))
    ax.bar(x - width / 2, plot_df["top1_under_40m_rate"], width, label="Top1 <=40m")
    ax.bar(x + width / 2, plot_df["top10_under_40m_rate"], width, label="Top10 <=40m")

    ax.set_ylim(0, 1.0)
    ax.set_ylabel("rate")
    ax.set_title("S4C.6A parameter sensitivity sweep")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", default="configs/satloc/s4c6a_parameter_sweep.yaml")
    parser.add_argument("--tokens", default=None, help="Override YAML tokens, e.g. all or 1,40,60")
    parser.add_argument("--experiments", default="all", help="Comma list or all")
    parser.add_argument("--save-candidate-scores", action="store_true")

    args = parser.parse_args()

    cfg_path = Path(args.config)
    cfg = load_yaml(cfg_path)

    sequence = cfg.get("sequence", "traj01")
    inputs = cfg["inputs"]
    sweep = cfg["sweep"]

    ranked_dir = Path(inputs["s4c1_ranked_dir"])
    token_text = args.tokens if args.tokens is not None else sweep.get("tokens", "all")
    tokens = parse_tokens(token_text, ranked_dir)

    phog_top_n = int(sweep.get("phog_top_n", 50))
    base_settings = sweep["base_settings"]
    experiments = filter_experiments(sweep["experiments"], args.experiments)

    OUT_META_DIR.mkdir(parents=True, exist_ok=True)
    OUT_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_FIG_DIR.mkdir(parents=True, exist_ok=True)

    uav_df = pd.read_csv(inputs["uav_index"])
    sat_df = pd.read_csv(inputs["sat_index"])

    uav_df = s4c4c.s4c4a.prepare_uav_df(uav_df, sequence)
    filename_index, fallback_uav_dirs, fallback_sat_dirs = s4c4c.s4c4a.build_filename_index(sequence)

    print("S4C.6A parameter sensitivity sweep")
    print("----------------------------------")
    print(f"Config:       {cfg_path}")
    print(f"Sequence:     {sequence}")
    print(f"Tokens:       {len(tokens)}")
    print(f"PHOG top-N:   {phog_top_n}")
    print(f"Experiments:  {[e['name'] for e in experiments]}")
    print("")

    all_summary = []
    exp_infos = []

    for exp in experiments:
        summary_df, _, exp_info = run_experiment(
            exp=exp,
            base_settings=base_settings,
            tokens=tokens,
            phog_top_n=phog_top_n,
            uav_df=uav_df,
            sat_df=sat_df,
            ranked_dir=ranked_dir,
            filename_index=filename_index,
            fallback_uav_dirs=fallback_uav_dirs,
            fallback_sat_dirs=fallback_sat_dirs,
            save_candidate_scores=args.save_candidate_scores,
        )

        all_summary.append(summary_df)
        exp_infos.append(exp_info)

    summary_all = pd.concat(all_summary, ignore_index=True)
    aggregate = aggregate_results(summary_all)

    summary_csv = OUT_META_DIR / f"s4c6a_{sequence}_summary_by_token_profile.csv"
    aggregate_csv = OUT_META_DIR / f"s4c6a_{sequence}_aggregate_by_experiment_profile.csv"
    exp_info_json = OUT_REPORT_DIR / f"s4c6a_{sequence}_experiment_info.json"
    summary_json = OUT_REPORT_DIR / f"s4c6a_{sequence}_summary.json"
    plot_path = OUT_FIG_DIR / f"s4c6a_{sequence}_top_profile_comparison.png"

    summary_all.to_csv(summary_csv, index=False)
    aggregate.to_csv(aggregate_csv, index=False)
    render_top_plot(aggregate, plot_path, top_n=20)

    out = {
        "stage": "S4C.6A_parameter_sensitivity_sweep",
        "config": str(cfg_path),
        "sequence": sequence,
        "tokens": tokens,
        "phog_top_n": phog_top_n,
        "experiments": exp_infos,
        "summary_csv": str(summary_csv),
        "aggregate_csv": str(aggregate_csv),
        "plot_path": str(plot_path),
        "notes": [
            "No reference coordinates are used during scoring/ranking.",
            "center_error_m is used only after ranking for evaluation.",
            "This sweep is diagnostic/tuning on the selected subset, not a final held-out result.",
            "Rotation is intentionally not included here; rotation-aware alignment should be S4C.6B.",
        ],
        "top_rows": aggregate.head(20).to_dict(orient="records"),
    }

    with open(exp_info_json, "w", encoding="utf-8") as f:
        json.dump(json_safe(exp_infos), f, indent=2)

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(json_safe(out), f, indent=2)

    print("")
    print("S4C.6A complete")
    print("----------------")
    print(f"Summary CSV:   {summary_csv}")
    print(f"Aggregate CSV: {aggregate_csv}")
    print(f"Summary JSON:  {summary_json}")
    print(f"Plot:          {plot_path}")
    print("")
    print("Top aggregate rows")
    print("------------------")
    cols = [
        "experiment",
        "profile",
        "num_queries",
        "top1_under_20m_rate",
        "top1_under_40m_rate",
        "top10_under_40m_rate",
        "median_top1_error_m",
        "median_best_top10_error_m",
        "median_top1_lsd_score",
    ]
    print(aggregate[cols].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
