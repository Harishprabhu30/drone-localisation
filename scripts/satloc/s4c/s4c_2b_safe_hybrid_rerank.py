#!/usr/bin/env python3
"""
S4C.2B — Safe hybrid reranking for SatLoc macro-contour retrieval.

Input:
  Existing S4C.2 reranked CSVs.

Purpose:
  Pure Chamfer damaged many good PHOG results. This script tests safer
  rank-fusion profiles:

      hybrid_score =
          w_phog * phog_rank
        + w_chamfer * chamfer_rank
        + w_density_mismatch * density_mismatch_rank
        + w_density_excess * density_excess_rank

Important:
  - No reference coordinate is used in scoring.
  - center_error_m is used only after ranking for evaluation.
  - This is a diagnostic rank-fusion block before any larger benchmark.

code to execute:
1. without panels and is fast:
export PYTHONPATH=$PWD/src

python scripts/satloc/s4c/s4c_2b_safe_hybrid_rerank.py \
  --sequence traj01 \
  --tokens all

2. with panels:
python scripts/satloc/s4c/s4c_2b_safe_hybrid_rerank.py \
  --sequence traj01 \
  --tokens 1,40,60,90,100,129,166,269,516,905 \
  --save-panels \
  --display-top-k 5 \
  --panel-profiles phog_only,chamfer_only,safe_weak,safe_balanced,rescue
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math


DEFAULT_S4C2_DIR = Path(
    "outputs/satloc/metadata/s4c_macrocontour_phog_chamfer/s4c2_chamfer_rerank"
)

OUT_ROOT = Path("outputs/satloc")
OUT_META_DIR = OUT_ROOT / "metadata/s4c_macrocontour_phog_chamfer/s4c2b_safe_hybrid_rerank"
OUT_REPORT_DIR = OUT_ROOT / "reports/s4c_macrocontour_phog_chamfer/s4c2b_safe_hybrid_rerank"
OUT_FIG_DIR = OUT_ROOT / "figures/s4c_macrocontour_phog_chamfer/s4c2b_safe_hybrid_rerank"


@dataclass
class Profile:
    name: str
    w_phog: float
    w_chamfer: float
    w_density_mismatch: float
    w_density_excess: float


def default_profiles() -> list[Profile]:
    return [
        Profile("phog_only", 1.00, 0.00, 0.00, 0.00),
        Profile("chamfer_only", 0.00, 1.00, 0.00, 0.00),

        # Safe profiles: PHOG remains the dominant prior.
        Profile("safe_weak", 1.00, 0.10, 0.10, 0.05),
        Profile("safe_balanced", 1.00, 0.25, 0.15, 0.10),

        # More aggressive rescue profile for cases where PHOG top-1 is bad.
        Profile("rescue", 1.00, 0.50, 0.20, 0.15),

        # Penalizes dense/mismatched contour fields more strongly.
        Profile("density_guard", 1.00, 0.20, 0.30, 0.30),
    ]


def parse_profiles(text: str) -> list[Profile]:
    if text.strip().lower() == "default":
        return default_profiles()

    profiles: list[Profile] = []

    # Format:
    # name:w_phog,w_chamfer,w_density_mismatch,w_density_excess;name2:...
    for chunk in text.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue

        if ":" not in chunk:
            raise ValueError(
                "Profile format must be name:w_phog,w_chamfer,w_density_mismatch,w_density_excess"
            )

        name, values = chunk.split(":", 1)
        nums = [float(x.strip()) for x in values.split(",")]
        if len(nums) != 4:
            raise ValueError(f"Profile {name} must have exactly 4 numeric weights.")

        profiles.append(Profile(name.strip(), nums[0], nums[1], nums[2], nums[3]))

    if not profiles:
        raise ValueError("No profiles parsed.")

    return profiles


def safe_float(x: Any, default: float = np.nan) -> float:
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def parse_token_list(text: str, s4c2_dir: Path) -> list[int]:
    text = text.strip()

    if text.lower() == "all":
        tokens: set[int] = set()
        for p in s4c2_dir.glob("s4c2_token*_reranked_top*.csv"):
            m = re.search(r"s4c2_token(\d+)_", p.name)
            if m:
                tokens.add(int(m.group(1)))
        if not tokens:
            raise FileNotFoundError(f"No S4C.2 token CSVs found in {s4c2_dir}")
        return sorted(tokens)

    tokens = []
    for part in text.split(","):
        part = part.strip()
        if part:
            tokens.append(int(part))

    return tokens


def find_latest_s4c2_csv(token: int, s4c2_dir: Path) -> Path:
    pattern = f"s4c2_token{token:04d}_*_reranked_top*.csv"
    matches = sorted(s4c2_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)

    if not matches:
        raise FileNotFoundError(f"No S4C.2 CSV found for token {token}: {s4c2_dir / pattern}")

    return matches[0]


def ensure_rank_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "phog_rank" not in df.columns:
        if "rank" in df.columns:
            df["phog_rank"] = df["rank"]
        else:
            df["phog_rank"] = np.arange(1, len(df) + 1)

    if "chamfer_rank" not in df.columns:
        if "chamfer_score_px" in df.columns:
            df = df.sort_values(["chamfer_score_px", "phog_rank"], ascending=[True, True]).copy()
            df["chamfer_rank"] = np.arange(1, len(df) + 1)
        else:
            df["chamfer_rank"] = np.arange(1, len(df) + 1)

    df["phog_rank"] = df["phog_rank"].astype(int)
    df["chamfer_rank"] = df["chamfer_rank"].astype(int)

    return df


def add_density_ranks(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "candidate_contour_density" in df.columns and "query_contour_density" in df.columns:
        cand = df["candidate_contour_density"].astype(float)
        query = df["query_contour_density"].astype(float)
        df["density_mismatch"] = (cand - query).abs()
        df["density_excess"] = np.maximum(0.0, cand - query)
    elif "candidate_cleaned_density" in df.columns and "query_cleaned_density" in df.columns:
        cand = df["candidate_cleaned_density"].astype(float)
        query = df["query_cleaned_density"].astype(float)
        df["density_mismatch"] = (cand - query).abs()
        df["density_excess"] = np.maximum(0.0, cand - query)
    else:
        df["density_mismatch"] = 0.0
        df["density_excess"] = 0.0

    df["density_mismatch_rank"] = (
        df["density_mismatch"].rank(method="min", ascending=True).fillna(len(df)).astype(int)
    )
    df["density_excess_rank"] = (
        df["density_excess"].rank(method="min", ascending=True).fillna(len(df)).astype(int)
    )

    return df


def first_rank_under(df: pd.DataFrame, rank_col: str, threshold_m: float) -> Optional[int]:
    if "center_error_m" not in df.columns:
        return None
    valid = df[np.isfinite(df["center_error_m"]) & (df["center_error_m"] <= threshold_m)]
    if len(valid) == 0:
        return None
    return int(valid[rank_col].min())


def summarize_profile(token: int, profile_df: pd.DataFrame, profile: Profile) -> dict[str, Any]:
    top1 = profile_df.iloc[0]

    first_20 = first_rank_under(profile_df, "hybrid_rank", 20.0)
    first_40 = first_rank_under(profile_df, "hybrid_rank", 40.0)
    first_60 = first_rank_under(profile_df, "hybrid_rank", 60.0)

    top10 = profile_df.head(10)

    out = {
        "token": token,
        "profile": profile.name,
        "w_phog": profile.w_phog,
        "w_chamfer": profile.w_chamfer,
        "w_density_mismatch": profile.w_density_mismatch,
        "w_density_excess": profile.w_density_excess,

        "top1_tile_id": int(top1["tile_id"]),
        "top1_error_m": safe_float(top1["center_error_m"]),
        "top1_phog_rank": int(top1["phog_rank"]),
        "top1_chamfer_rank": int(top1["chamfer_rank"]),
        "top1_hybrid_score": safe_float(top1["hybrid_score"]),

        "best_top10_error_m": safe_float(top10["center_error_m"].min()),
        "first_rank_under_20m": first_20,
        "first_rank_under_40m": first_40,
        "first_rank_under_60m": first_60,

        "top1_under_20m": bool(np.isfinite(safe_float(top1["center_error_m"])) and top1["center_error_m"] <= 20.0),
        "top1_under_40m": bool(np.isfinite(safe_float(top1["center_error_m"])) and top1["center_error_m"] <= 40.0),
        "top1_under_60m": bool(np.isfinite(safe_float(top1["center_error_m"])) and top1["center_error_m"] <= 60.0),

        "top10_under_20m": bool(np.isfinite(top10["center_error_m"]).any() and (top10["center_error_m"] <= 20.0).any()),
        "top10_under_40m": bool(np.isfinite(top10["center_error_m"]).any() and (top10["center_error_m"] <= 40.0).any()),
        "top10_under_60m": bool(np.isfinite(top10["center_error_m"]).any() and (top10["center_error_m"] <= 60.0).any()),
    }

    return out


def apply_profile(df: pd.DataFrame, profile: Profile) -> pd.DataFrame:
    out = df.copy()

    out["hybrid_score"] = (
        profile.w_phog * out["phog_rank"].astype(float)
        + profile.w_chamfer * out["chamfer_rank"].astype(float)
        + profile.w_density_mismatch * out["density_mismatch_rank"].astype(float)
        + profile.w_density_excess * out["density_excess_rank"].astype(float)
    )

    out["profile"] = profile.name

    out = out.sort_values(
        ["hybrid_score", "phog_rank", "chamfer_rank"],
        ascending=[True, True, True],
    ).reset_index(drop=True)

    out["hybrid_rank"] = np.arange(1, len(out) + 1)

    return out


def read_rgb(path: str, size: int = 256) -> Optional[np.ndarray]:
    p = Path(str(path))
    if not p.exists():
        return None

    bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if bgr is None:
        return None

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    h, w = rgb.shape[:2]
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    rgb = rgb[y0:y0 + side, x0:x0 + side]
    rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)

    return rgb


def render_profile_panel(
    token: int,
    profile_rankings: dict[str, pd.DataFrame],
    panel_profiles: list[str],
    display_top_k: int,
    out_path: Path,
) -> None:
    rows = [p for p in panel_profiles if p in profile_rankings]
    if not rows:
        return

    fig, axes = plt.subplots(
        len(rows),
        display_top_k,
        figsize=(3.0 * display_top_k, 3.2 * len(rows)),
        squeeze=False,
    )

    for r, profile_name in enumerate(rows):
        df = profile_rankings[profile_name].head(display_top_k)

        for c in range(display_top_k):
            ax = axes[r, c]
            ax.axis("off")
            ax.set_xticks([])
            ax.set_yticks([])

            if c >= len(df):
                continue

            row = df.iloc[c]
            img_path = row.get("candidate_image_path", None)
            img = read_rgb(str(img_path)) if img_path is not None else None

            if img is not None:
                ax.imshow(img)

            err = safe_float(row.get("center_error_m", np.nan))
            tile_id = int(row.get("tile_id", -1))

            ax.set_title(
                f"{profile_name} R{int(row['hybrid_rank'])}\n"
                f"tile {tile_id}, err={err:.1f}m\n"
                f"P{int(row['phog_rank'])} C{int(row['chamfer_rank'])}",
                fontsize=8,
            )

            if c == 0:
                ax.set_ylabel(profile_name, fontsize=10)

    fig.suptitle(f"S4C.2B safe hybrid rerank — token {token}", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def render_profile_aggregate_plot(aggregate_df: pd.DataFrame, out_path: Path) -> None:
    if len(aggregate_df) == 0:
        return

    plot_df = aggregate_df.copy()
    labels = plot_df["profile"].astype(str).tolist()
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(11, 5.0))

    ax.bar(x - width / 2, plot_df["top1_under_40m_rate"], width, label="Top-1 <= 40 m")
    ax.bar(x + width / 2, plot_df["top10_under_40m_rate"], width, label="Top-10 <= 40 m")

    ax.set_ylim(0, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("rate")
    ax.set_title("S4C.2B profile comparison — <=40 m success rate")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        val = float(obj)
        if math.isnan(val) or math.isinf(val):
            return None
        return val
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    return obj


def run_token(
    token: int,
    s4c2_dir: Path,
    profiles: list[Profile],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], Optional[Path]]:
    s4c2_csv = find_latest_s4c2_csv(token, s4c2_dir)
    df = pd.read_csv(s4c2_csv)

    if len(df) == 0:
        return [], None

    df = ensure_rank_columns(df)
    df = add_density_ranks(df)

    profile_rankings: dict[str, pd.DataFrame] = {}
    summary_rows: list[dict[str, Any]] = []

    for profile in profiles:
        prof_df = apply_profile(df, profile)
        profile_rankings[profile.name] = prof_df

        summary = summarize_profile(token, prof_df, profile)
        summary["s4c2_input_csv"] = str(s4c2_csv)
        summary_rows.append(summary)

    token_long_rows = []
    for profile_name, prof_df in profile_rankings.items():
        keep = prof_df.copy()
        keep["token"] = token
        token_long_rows.append(keep)

    token_long_df = pd.concat(token_long_rows, ignore_index=True)

    OUT_META_DIR.mkdir(parents=True, exist_ok=True)
    token_csv = OUT_META_DIR / f"s4c2b_token{token:04d}_safe_hybrid_profiles.csv"
    token_long_df.to_csv(token_csv, index=False)

    panel_path = None
    if args.save_panels:
        panel_profiles = [x.strip() for x in args.panel_profiles.split(",") if x.strip()]
        panel_path = OUT_FIG_DIR / f"s4c2b_token{token:04d}_safe_hybrid_panel_top{args.display_top_k}.png"
        render_profile_panel(
            token=token,
            profile_rankings=profile_rankings,
            panel_profiles=panel_profiles,
            display_top_k=args.display_top_k,
            out_path=panel_path,
        )

    for row in summary_rows:
        row["s4c2b_token_csv"] = str(token_csv)
        row["panel_path"] = str(panel_path) if panel_path is not None else None

    return summary_rows, panel_path


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--sequence", default="traj01")
    parser.add_argument("--tokens", default="all")
    parser.add_argument("--s4c2-dir", default=str(DEFAULT_S4C2_DIR))
    parser.add_argument("--profiles", default="default")
    parser.add_argument("--display-top-k", type=int, default=10)
    parser.add_argument("--save-panels", action="store_true")
    parser.add_argument(
        "--panel-profiles",
        default="phog_only,chamfer_only,safe_weak,safe_balanced,rescue",
    )

    args = parser.parse_args()

    s4c2_dir = Path(args.s4c2_dir)
    if not s4c2_dir.exists():
        raise FileNotFoundError(f"Missing S4C.2 dir: {s4c2_dir}")

    OUT_META_DIR.mkdir(parents=True, exist_ok=True)
    OUT_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_FIG_DIR.mkdir(parents=True, exist_ok=True)

    profiles = parse_profiles(args.profiles)
    tokens = parse_token_list(args.tokens, s4c2_dir)

    print("S4C.2B Safe hybrid reranking")
    print("----------------------------")
    print(f"Sequence: {args.sequence}")
    print(f"Tokens:   {len(tokens)}")
    print(f"S4C2 dir: {s4c2_dir}")
    print("")
    print("Profiles")
    print("--------")
    for p in profiles:
        print(
            f"{p.name}: "
            f"phog={p.w_phog}, chamfer={p.w_chamfer}, "
            f"density_mismatch={p.w_density_mismatch}, density_excess={p.w_density_excess}"
        )
    print("")

    all_summary_rows: list[dict[str, Any]] = []
    panel_paths: list[str] = []

    for token in tokens:
        try:
            rows, panel = run_token(token, s4c2_dir, profiles, args)
        except Exception as exc:
            print(f"[WARN] token {token}: {exc}")
            continue

        if panel is not None:
            panel_paths.append(str(panel))

        all_summary_rows.extend(rows)

        if rows:
            # Print compact per-token best lines for main profiles.
            compact = pd.DataFrame(rows)
            msg_parts = []
            for name in ["phog_only", "safe_weak", "safe_balanced", "rescue"]:
                sub = compact[compact["profile"] == name]
                if len(sub):
                    r = sub.iloc[0]
                    msg_parts.append(f"{name}: {r['top1_error_m']:.1f}m")
            print(f"[OK] token {token}: " + " | ".join(msg_parts))

    summary_df = pd.DataFrame(all_summary_rows)

    summary_csv = OUT_META_DIR / f"s4c2b_{args.sequence}_safe_hybrid_summary.csv"
    aggregate_csv = OUT_META_DIR / f"s4c2b_{args.sequence}_safe_hybrid_aggregate_by_profile.csv"
    summary_json = OUT_REPORT_DIR / f"s4c2b_{args.sequence}_safe_hybrid_summary.json"
    aggregate_plot = OUT_FIG_DIR / f"s4c2b_{args.sequence}_profile_success_rates.png"

    summary_df.to_csv(summary_csv, index=False)

    aggregate_rows: list[dict[str, Any]] = []

    if len(summary_df) > 0:
        for profile_name, group in summary_df.groupby("profile"):
            aggregate_rows.append(
                {
                    "profile": profile_name,
                    "num_queries": int(len(group)),
                    "mean_top1_error_m": float(group["top1_error_m"].mean()),
                    "median_top1_error_m": float(group["top1_error_m"].median()),
                    "mean_best_top10_error_m": float(group["best_top10_error_m"].mean()),
                    "median_best_top10_error_m": float(group["best_top10_error_m"].median()),
                    "top1_under_20m_rate": float(group["top1_under_20m"].mean()),
                    "top1_under_40m_rate": float(group["top1_under_40m"].mean()),
                    "top1_under_60m_rate": float(group["top1_under_60m"].mean()),
                    "top10_under_20m_rate": float(group["top10_under_20m"].mean()),
                    "top10_under_40m_rate": float(group["top10_under_40m"].mean()),
                    "top10_under_60m_rate": float(group["top10_under_60m"].mean()),
                }
            )

    aggregate_df = pd.DataFrame(aggregate_rows)
    if len(aggregate_df) > 0:
        aggregate_df = aggregate_df.sort_values(
            ["top1_under_40m_rate", "top10_under_40m_rate", "median_top1_error_m"],
            ascending=[False, False, True],
        )

    aggregate_df.to_csv(aggregate_csv, index=False)
    render_profile_aggregate_plot(aggregate_df, aggregate_plot)

    output = {
        "stage": "S4C.2B_safe_hybrid_rerank",
        "sequence": args.sequence,
        "tokens": tokens,
        "profiles": [p.__dict__ for p in profiles],
        "summary_csv": str(summary_csv),
        "aggregate_csv": str(aggregate_csv),
        "aggregate_plot": str(aggregate_plot),
        "panel_paths": panel_paths,
        "notes": [
            "No reference coordinates used in scoring.",
            "center_error_m used only for evaluation after ranking.",
            "Safe profiles keep PHOG as dominant prior and use Chamfer as weak evidence.",
        ],
        "aggregate_by_profile": aggregate_rows,
    }

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(json_safe(output), f, indent=2)

    print("")
    print("S4C.2B complete")
    print("----------------")
    print(f"Summary CSV:   {summary_csv}")
    print(f"Aggregate CSV: {aggregate_csv}")
    print(f"Summary JSON:  {summary_json}")
    print(f"Plot:          {aggregate_plot}")
    print(f"Figures:       {OUT_FIG_DIR}")

    if len(aggregate_df) > 0:
        print("")
        print("Aggregate profile comparison")
        print("----------------------------")
        cols = [
            "profile",
            "num_queries",
            "top1_under_20m_rate",
            "top1_under_40m_rate",
            "top10_under_40m_rate",
            "median_top1_error_m",
            "median_best_top10_error_m",
        ]
        print(aggregate_df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
