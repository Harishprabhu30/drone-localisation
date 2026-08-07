'''
Run command:

ROOT=outputs/villoc/traj01_90deg_stable120m

mkdir -p "$ROOT/logs/s8_relative_absolute_fusion"

python scripts/villoc/s8_fusion/s8_f3_temporal_agreement_gating.py \
  --root "$ROOT" \
  2>&1 | tee "$ROOT/logs/s8_relative_absolute_fusion/s8_f3_temporal_agreement_gating.log"

'''

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def bool_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False)
    return (
        s.astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes", "y", "t"})
    )


def require_columns(df: pd.DataFrame, cols: list[str], name: str) -> None:
    missing = sorted(set(cols) - set(df.columns))
    if missing:
        raise RuntimeError(f"{name} missing columns: {missing}")


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if pd.isna(value):
        return None
    return value


def compute_cumulative_distance(xy: np.ndarray) -> np.ndarray:
    steps = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(steps)])


def policy_gap_rows(df: pd.DataFrame, policy: str) -> list[dict[str, Any]]:
    accepted = df.loc[bool_series(df[policy])].sort_values("reference_cumulative_distance_m")
    start_m = float(df["reference_cumulative_distance_m"].min())
    end_m = float(df["reference_cumulative_distance_m"].max())

    if accepted.empty:
        return [{
            "policy": policy,
            "gap_type": "whole_trajectory_no_corrections",
            "from_token0_id": None,
            "to_token0_id": None,
            "start_distance_m": start_m,
            "end_distance_m": end_m,
            "gap_m": end_m - start_m,
        }]

    rows = []

    first = accepted.iloc[0]
    rows.append({
        "policy": policy,
        "gap_type": "start_boundary",
        "from_token0_id": None,
        "to_token0_id": int(first["token0_id"]),
        "start_distance_m": start_m,
        "end_distance_m": float(first["reference_cumulative_distance_m"]),
        "gap_m": float(first["reference_cumulative_distance_m"] - start_m),
    })

    rec = accepted.to_dict(orient="records")
    for prev, cur in zip(rec[:-1], rec[1:]):
        rows.append({
            "policy": policy,
            "gap_type": "between_corrections",
            "from_token0_id": int(prev["token0_id"]),
            "to_token0_id": int(cur["token0_id"]),
            "start_distance_m": float(prev["reference_cumulative_distance_m"]),
            "end_distance_m": float(cur["reference_cumulative_distance_m"]),
            "gap_m": float(cur["reference_cumulative_distance_m"] - prev["reference_cumulative_distance_m"]),
        })

    last = accepted.iloc[-1]
    rows.append({
        "policy": policy,
        "gap_type": "end_boundary",
        "from_token0_id": int(last["token0_id"]),
        "to_token0_id": None,
        "start_distance_m": float(last["reference_cumulative_distance_m"]),
        "end_distance_m": end_m,
        "gap_m": float(end_m - last["reference_cumulative_distance_m"]),
    })

    return rows


def build_temporal_policy(
    df: pd.DataFrame,
    *,
    tag: str,
    candidate_col: str,
    bootstrap_col: str,
    max_residual_m: float,
    residual_ratio: float,
    min_gap_m: float,
) -> None:
    candidate = bool_series(df[candidate_col]).to_numpy(bool)
    bootstrap = bool_series(df[bootstrap_col]).to_numpy(bool)

    rel_xy = df[["prefix_aligned_x_m", "prefix_aligned_y_m"]].to_numpy(dtype=float)
    abs_xy = df[["abs_pred_x_m", "abs_pred_y_m"]].to_numpy(dtype=float)
    rel_cum = df["relative_cumulative_distance_online_m"].to_numpy(dtype=float)

    accept = np.zeros(len(df), dtype=bool)
    residuals = np.full(len(df), np.nan, dtype=float)
    thresholds = np.full(len(df), np.nan, dtype=float)
    since_anchor = np.full(len(df), np.nan, dtype=float)
    anchor_before = np.full(len(df), np.nan, dtype=float)
    reason = np.array(["not_candidate"] * len(df), dtype=object)

    last_anchor: int | None = None

    for i in range(len(df)):
        if not candidate[i]:
            continue

        reason[i] = "candidate_not_evaluated"

        if last_anchor is None:
            if bootstrap[i]:
                accept[i] = True
                residuals[i] = 0.0
                thresholds[i] = np.inf
                since_anchor[i] = 0.0
                reason[i] = "bootstrap_accept"
                last_anchor = i
            else:
                reason[i] = "waiting_for_bootstrap"
            continue

        anchor_before[i] = float(df.iloc[last_anchor]["token0_id"])

        distance_since = float(rel_cum[i] - rel_cum[last_anchor])
        since_anchor[i] = distance_since

        if distance_since < min_gap_m:
            reason[i] = "gap_too_short"
            continue

        rel_delta = rel_xy[i] - rel_xy[last_anchor]
        abs_delta = abs_xy[i] - abs_xy[last_anchor]

        rel_delta_norm = float(np.linalg.norm(rel_delta))
        residual = float(np.linalg.norm(abs_delta - rel_delta))
        threshold = float(max(max_residual_m, residual_ratio * max(rel_delta_norm, 1e-9)))

        residuals[i] = residual
        thresholds[i] = threshold

        if residual <= threshold:
            accept[i] = True
            reason[i] = "temporal_agreement_accept"
            last_anchor = i
        else:
            reason[i] = "temporal_disagreement_reject"

    df[f"{tag}_candidate_online"] = candidate
    df[f"{tag}_accept_online"] = accept
    df[f"{tag}_temporal_residual_m"] = residuals
    df[f"{tag}_temporal_threshold_m"] = thresholds
    df[f"{tag}_distance_since_anchor_m"] = since_anchor
    df[f"{tag}_anchor_token0_id_before"] = anchor_before
    df[f"{tag}_reason"] = reason


def summarize_policy(df: pd.DataFrame, gap_df: pd.DataFrame, tag: str) -> dict[str, Any]:
    policy = f"{tag}_accept_online"
    candidate_col = f"{tag}_candidate_online"
    reason_col = f"{tag}_reason"

    accepted_mask = bool_series(df[policy])
    candidate_mask = bool_series(df[candidate_col])
    accepted = df.loc[accepted_mask].copy()

    true_mask = bool_series(df["abs_hit_le_40m_eval_only"])
    contains_mask = bool_series(df["abs_contains_body_eval_only"])
    dangerous_mask = pd.to_numeric(df["abs_error_m_eval_only"], errors="coerce") > 100.0

    accepted_count = int(accepted_mask.sum())
    true_count = int((accepted_mask & true_mask).sum())
    contains_count = int((accepted_mask & contains_mask).sum())
    dangerous_count = int((accepted_mask & dangerous_mask).sum())
    false_count = accepted_count - true_count

    accepted_errors = pd.to_numeric(
        df.loc[accepted_mask, "abs_error_m_eval_only"],
        errors="coerce",
    ).dropna()

    residuals = pd.to_numeric(
        df.loc[accepted_mask, f"{tag}_temporal_residual_m"],
        errors="coerce",
    ).replace([np.inf, -np.inf], np.nan).dropna()

    rejected_temporal = int((df[reason_col].astype(str) == "temporal_disagreement_reject").sum())
    waiting_bootstrap = int((df[reason_col].astype(str) == "waiting_for_bootstrap").sum())
    short_gap = int((df[reason_col].astype(str) == "gap_too_short").sum())

    policy_gaps = gap_df[gap_df["policy"] == policy].copy()
    between = policy_gaps.loc[policy_gaps["gap_type"] == "between_corrections", "gap_m"].dropna()
    coverage = policy_gaps["gap_m"].dropna()

    return {
        "policy": policy,
        "candidate_events_online": int(candidate_mask.sum()),
        "accepted": accepted_count,
        "temporal_rejected_events_online": rejected_temporal,
        "waiting_for_bootstrap_events_online": waiting_bootstrap,
        "gap_too_short_events_online": short_gap,
        "true_accepts_le40_eval_only": true_count,
        "contains_body_accepts_eval_only": contains_count,
        "false_accepts_eval_only": false_count,
        "dangerous_false_accepts_gt100m_eval_only": dangerous_count,
        "precision_le40_eval_only": true_count / accepted_count if accepted_count else None,
        "contains_precision_eval_only": contains_count / accepted_count if accepted_count else None,
        "median_accepted_error_m_eval_only": float(accepted_errors.median()) if len(accepted_errors) else None,
        "p95_accepted_error_m_eval_only": float(accepted_errors.quantile(0.95)) if len(accepted_errors) else None,
        "median_temporal_residual_m_online": float(residuals.median()) if len(residuals) else None,
        "p95_temporal_residual_m_online": float(residuals.quantile(0.95)) if len(residuals) else None,
        "inter_correction_gap_median": float(between.median()) if len(between) else None,
        "inter_correction_gap_p95": float(between.quantile(0.95)) if len(between) else None,
        "inter_correction_gap_max": float(between.max()) if len(between) else None,
        "coverage_gap_including_boundaries_max_m": float(coverage.max()) if len(coverage) else None,
        "first_accepted_token0_id": int(accepted["token0_id"].min()) if len(accepted) else None,
        "last_accepted_token0_id": int(accepted["token0_id"].max()) if len(accepted) else None,
    }


def save_policy_distribution_plot(df: pd.DataFrame, policies: list[str], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(13, 5.5))

    for row, policy in enumerate(policies):
        accepted = df.loc[bool_series(df[policy])].sort_values("reference_cumulative_distance_m")
        if accepted.empty:
            continue

        true = accepted.loc[bool_series(accepted["abs_hit_le_40m_eval_only"])]
        false = accepted.loc[~bool_series(accepted["abs_hit_le_40m_eval_only"])]

        ax.scatter(
            true["reference_cumulative_distance_m"],
            np.full(len(true), row),
            s=28,
            marker="o",
            label="true <=40m" if row == 0 else None,
        )
        ax.scatter(
            false["reference_cumulative_distance_m"],
            np.full(len(false), row),
            s=28,
            marker="x",
            label="false >40m" if row == 0 else None,
        )

    ax.set_yticks(range(len(policies)))
    ax.set_yticklabels(policies, fontsize=8)
    ax.set_xlabel("Reference cumulative distance [m] — evaluation only")
    ax.set_ylabel("Temporal policy")
    ax.set_title("S8.F3 temporal-agreement accepted corrections")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_residual_plot(df: pd.DataFrame, tags: list[str], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 6))

    for tag in tags:
        rcol = f"{tag}_temporal_residual_m"
        tcol = f"{tag}_temporal_threshold_m"
        acol = f"{tag}_accept_online"

        sub = df.loc[pd.to_numeric(df[rcol], errors="coerce").notna()].copy()
        sub = sub.replace([np.inf, -np.inf], np.nan).dropna(subset=[rcol, tcol])

        if sub.empty:
            continue

        ax.scatter(
            sub["relative_cumulative_distance_online_m"],
            sub[rcol],
            s=12,
            alpha=0.65,
            label=f"{tag} residual",
        )
        ax.plot(
            sub["relative_cumulative_distance_online_m"],
            sub[tcol],
            linewidth=1.2,
            alpha=0.75,
            label=f"{tag} threshold",
        )

    ax.set_xlabel("Relative cumulative distance [m] — online")
    ax.set_ylabel("Temporal residual / threshold [m]")
    ax.set_title("S8.F3 temporal agreement residuals")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="S8.F3 temporal-agreement gating for Villoc relative+absolute fusion."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("outputs/villoc/traj01_90deg_stable120m"),
    )
    parser.add_argument("--manifest", type=Path, default=None)
    args = parser.parse_args()

    root = args.root
    manifest_path = args.manifest or (
        root / "metadata/s8_relative_absolute_fusion/s8_f1_absolute_correction_manifest.csv"
    )

    metadata_dir = root / "metadata/s8_relative_absolute_fusion"
    reports_dir = root / "reports/s8_relative_absolute_fusion"
    figures_dir = root / "figures/s8_relative_absolute_fusion"
    logs_dir = root / "logs/s8_relative_absolute_fusion"

    for d in [metadata_dir, reports_dir, figures_dir, logs_dir]:
        d.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(manifest_path)
    df = df.sort_values("sequence_frame_id", kind="mergesort").reset_index(drop=True)

    require_columns(
        df,
        [
            "sequence_frame_id", "token0_id",
            "prefix_aligned_x_m", "prefix_aligned_y_m",
            "abs_pred_x_m", "abs_pred_y_m",
            "reference_cumulative_distance_m",
            "strict_a_accept_online", "strict_b_accept_online",
            "abs_hit_le_40m_eval_only",
            "abs_contains_body_eval_only",
            "abs_error_m_eval_only",
        ],
        "S8.F1 manifest",
    )

    if len(df) != 403:
        raise RuntimeError(f"Expected 403 rows, got {len(df)}")

    rel_xy = df[["prefix_aligned_x_m", "prefix_aligned_y_m"]].to_numpy(dtype=float)
    df["relative_cumulative_distance_online_m"] = compute_cumulative_distance(rel_xy)

    policy_specs = [
        {
            "tag": "f3_temporal_a_bootb_res30_r025_gap30",
            "candidate_col": "strict_a_accept_online",
            "bootstrap_col": "strict_b_accept_online",
            "max_residual_m": 30.0,
            "residual_ratio": 0.25,
            "min_gap_m": 30.0,
        },
        {
            "tag": "f3_temporal_a_bootb_res50_r030_gap30",
            "candidate_col": "strict_a_accept_online",
            "bootstrap_col": "strict_b_accept_online",
            "max_residual_m": 50.0,
            "residual_ratio": 0.30,
            "min_gap_m": 30.0,
        },
        {
            "tag": "f3_temporal_a_bootb_res30_r025_gap50",
            "candidate_col": "strict_a_accept_online",
            "bootstrap_col": "strict_b_accept_online",
            "max_residual_m": 30.0,
            "residual_ratio": 0.25,
            "min_gap_m": 50.0,
        },
        {
            "tag": "f3_temporal_a_bootb_res50_r030_gap50",
            "candidate_col": "strict_a_accept_online",
            "bootstrap_col": "strict_b_accept_online",
            "max_residual_m": 50.0,
            "residual_ratio": 0.30,
            "min_gap_m": 50.0,
        },
        {
            "tag": "f3_temporal_a_bootb_res50_r030_gap100",
            "candidate_col": "strict_a_accept_online",
            "bootstrap_col": "strict_b_accept_online",
            "max_residual_m": 50.0,
            "residual_ratio": 0.30,
            "min_gap_m": 100.0,
        },
        {
            "tag": "f3_temporal_b_res30_r025_gap30",
            "candidate_col": "strict_b_accept_online",
            "bootstrap_col": "strict_b_accept_online",
            "max_residual_m": 30.0,
            "residual_ratio": 0.25,
            "min_gap_m": 30.0,
        },
    ]

    tags = []
    policies = []

    for spec in policy_specs:
        for col in [spec["candidate_col"], spec["bootstrap_col"]]:
            if col not in df.columns:
                raise RuntimeError(f"Missing policy source column: {col}")

        build_temporal_policy(df, **spec)
        tags.append(spec["tag"])
        policies.append(f"{spec['tag']}_accept_online")

    gap_rows = []
    for policy in policies:
        gap_rows.extend(policy_gap_rows(df, policy))
    gap_df = pd.DataFrame(gap_rows)

    summary_df = pd.DataFrame([
        summarize_policy(df, gap_df, tag)
        for tag in tags
    ])

    summary_df = summary_df.sort_values(
        [
            "dangerous_false_accepts_gt100m_eval_only",
            "false_accepts_eval_only",
            "coverage_gap_including_boundaries_max_m",
            "accepted",
        ],
        ascending=[True, True, True, False],
        kind="mergesort",
    ).reset_index(drop=True)

    manifest_out = metadata_dir / "s8_f3_temporal_agreement_manifest.csv"
    summary_out = metadata_dir / "s8_f3_temporal_policy_summary.csv"
    gaps_out = metadata_dir / "s8_f3_temporal_policy_gaps.csv"
    report_out = reports_dir / "s8_f3_temporal_agreement_gating_report.json"
    dist_plot = figures_dir / "s8_f3_temporal_policy_correction_distribution.png"
    residual_plot = figures_dir / "s8_f3_temporal_residuals.png"

    df.to_csv(manifest_out, index=False)
    summary_df.to_csv(summary_out, index=False)
    gap_df.to_csv(gaps_out, index=False)

    save_policy_distribution_plot(df, policies, dist_plot)
    save_residual_plot(df, tags[:3], residual_plot)

    report = {
        "stage": "S8.F3",
        "status": "PASS_S8_F3_TEMPORAL_AGREEMENT_GATING",
        "purpose": (
            "Build temporal-agreement correction gates using XFeat relative motion "
            "and ORB-reranked absolute corrections."
        ),
        "important_rule": (
            "Temporal policy decisions use only relative displacement, absolute predicted "
            "tile-center displacement, and online ORB confidence gates. Reference errors, "
            "hit labels, body containment, and oracle information are evaluation-only."
        ),
        "inputs": {
            "s8_f1_manifest": str(manifest_path),
        },
        "outputs": {
            "manifest": str(manifest_out),
            "summary": str(summary_out),
            "gaps": str(gaps_out),
            "report": str(report_out),
            "distribution_plot": str(dist_plot),
            "residual_plot": str(residual_plot),
        },
        "rows": {
            "manifest": int(len(df)),
            "summary": int(len(summary_df)),
            "gaps": int(len(gap_df)),
        },
        "policy_specs": policy_specs,
        "policy_columns": policies,
        "summary": summary_df.where(pd.notna(summary_df), None).to_dict(orient="records"),
    }

    report_out.write_text(
        json.dumps(report, indent=2, default=json_safe),
        encoding="utf-8",
    )

    print("S8.F3 Temporal-agreement gating")
    print("-" * 72)
    print("status:", report["status"])
    print("rows:", len(df))
    print()
    print("Temporal policy summary")
    print("-" * 72)
    print(summary_df.to_string(index=False))
    print()
    print("Saved outputs")
    print("-" * 72)
    print(manifest_out)
    print(summary_out)
    print(gaps_out)
    print(report_out)
    print(dist_plot)
    print(residual_plot)


if __name__ == "__main__":
    main()
