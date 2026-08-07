'''
Run command:

ROOT=outputs/villoc/traj01_90deg_stable120m

python scripts/villoc/s8_fusion/s8_f2_controlled_fusion_replay.py \
  --root "$ROOT" \
  --alphas 1.0,0.75,0.5,0.25 \
  --eval-start-index 49 \
  --thresholds-m 10,20,40,80,120 \
  --sustain-frames 5 \
  2>&1 | tee "$ROOT/logs/s8_relative_absolute_fusion/s8_f2_controlled_fusion_replay.log"

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


def parse_csv_list(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def parse_float_list(value: str) -> list[float]:
    out = []
    for x in value.split(","):
        x = x.strip()
        if x:
            out.append(float(x))
    if not out:
        raise ValueError("At least one alpha is required.")
    return out


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


def error_metrics(
    xy: np.ndarray,
    ref: np.ndarray,
    cumulative_distance: np.ndarray,
    start_index: int,
) -> dict[str, float]:
    errors = np.linalg.norm(xy - ref, axis=1)
    tail = errors[start_index:]
    distance = float(cumulative_distance[-1] - cumulative_distance[start_index])
    final_error = float(errors[-1])

    return {
        "rmse_m": float(math.sqrt(np.mean(tail * tail))),
        "mean_error_m": float(np.mean(tail)),
        "median_error_m": float(np.median(tail)),
        "p95_error_m": float(np.percentile(tail, 95)),
        "max_error_m": float(np.max(tail)),
        "final_error_m": final_error,
        "evaluation_distance_m": distance,
        "final_drift_per_100m": float(100.0 * final_error / distance) if distance > 1e-9 else float("nan"),
    }


def first_sustained_crossing(
    errors: np.ndarray,
    cumulative_distance: np.ndarray,
    threshold_m: float,
    start_index: int,
    sustain_frames: int,
) -> dict[str, Any]:
    above = np.isfinite(errors) & (errors >= threshold_m)

    for index in range(start_index, len(errors) - sustain_frames + 1):
        if bool(np.all(above[index:index + sustain_frames])):
            return {
                "threshold_m": float(threshold_m),
                "crossed": True,
                "frame_index": int(index),
                "token0_id": int(index + 1),
                "frames_after_eval_start": int(index - start_index),
                "distance_after_eval_start_m": float(
                    cumulative_distance[index] - cumulative_distance[start_index]
                ),
                "error_at_crossing_m": float(errors[index]),
            }

    return {
        "threshold_m": float(threshold_m),
        "crossed": False,
        "frame_index": None,
        "token0_id": None,
        "frames_after_eval_start": None,
        "distance_after_eval_start_m": None,
        "error_at_crossing_m": None,
    }


def replay_policy(
    manifest: pd.DataFrame,
    policy: str,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    rel = manifest[["prefix_aligned_x_m", "prefix_aligned_y_m"]].to_numpy(dtype=float)
    abs_xy = manifest[["abs_pred_x_m", "abs_pred_y_m"]].to_numpy(dtype=float)
    accepted = bool_series(manifest[policy]).to_numpy(bool)

    fused = np.zeros_like(rel)
    correction_applied = np.zeros(len(rel), dtype=bool)

    fused[0] = rel[0]
    if accepted[0]:
        fused[0] = (1.0 - alpha) * fused[0] + alpha * abs_xy[0]
        correction_applied[0] = True

    for i in range(1, len(rel)):
        delta = rel[i] - rel[i - 1]
        fused[i] = fused[i - 1] + delta

        if accepted[i]:
            fused[i] = (1.0 - alpha) * fused[i] + alpha * abs_xy[i]
            correction_applied[i] = True

    return fused, correction_applied


def save_error_plot(
    trajectory_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plot_rows = []

    # Always include baseline.
    plot_rows.append(("no_correction_baseline", "0.0"))

    # Include representative hard-reset policies.
    preferred = [
        "strict_a_accept_online",
        "strict_b_accept_online",
        "oracle_hit40_accept_eval_only",
        "all_reranked_accept_ablation",
    ]

    for p in preferred:
        if ((summary_df["policy"] == p) & (summary_df["alpha"] == 1.0)).any():
            plot_rows.append((p, "1.0"))

    fig, ax = plt.subplots(figsize=(12, 6))

    for policy, alpha_str in plot_rows:
        alpha = float(alpha_str)
        sub = trajectory_df[
            (trajectory_df["policy"] == policy)
            & (trajectory_df["alpha"] == alpha)
        ].sort_values("sequence_frame_id")

        if sub.empty:
            continue

        ax.plot(
            sub["reference_cumulative_distance_m"],
            sub["fusion_error_m"],
            label=f"{policy}, alpha={alpha:g}",
        )

    for threshold in [10, 20, 40, 80, 120]:
        ax.axhline(threshold, linestyle="--", linewidth=1, alpha=0.55)

    ax.set_xlabel("Reference cumulative distance [m] — evaluation only")
    ax.set_ylabel("Position error [m]")
    ax.set_title("S8.F2 relative+absolute fusion replay error")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_xy_plot(
    trajectory_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    choices = [
        ("no_correction_baseline", 0.0),
        ("strict_a_accept_online", 1.0),
        ("strict_b_accept_online", 1.0),
        ("oracle_hit40_accept_eval_only", 1.0),
    ]

    fig, ax = plt.subplots(figsize=(9, 8))

    base = trajectory_df[
        (trajectory_df["policy"] == "no_correction_baseline")
        & (trajectory_df["alpha"] == 0.0)
    ].sort_values("sequence_frame_id")

    ax.plot(
        base["reference_x_m"],
        base["reference_y_m"],
        label="Reference — evaluation only",
        linewidth=2,
    )

    for policy, alpha in choices:
        sub = trajectory_df[
            (trajectory_df["policy"] == policy)
            & (trajectory_df["alpha"] == alpha)
        ].sort_values("sequence_frame_id")

        if sub.empty:
            continue

        ax.plot(
            sub["fused_x_m"],
            sub["fused_y_m"],
            label=f"{policy}, alpha={alpha:g}",
        )

    ax.axis("equal")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_title("S8.F2 relative+absolute fused trajectories")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_summary_bar(summary_df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    preferred = summary_df[
        (
            ((summary_df["policy"] == "no_correction_baseline") & (summary_df["alpha"] == 0.0))
            | (summary_df["alpha"] == 1.0)
        )
    ].copy()

    preferred = preferred.sort_values("rmse_m", kind="mergesort").head(12)

    labels = preferred["policy"] + "\nα=" + preferred["alpha"].map(lambda x: f"{x:g}")

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(range(len(preferred)), preferred["rmse_m"])
    ax.set_xticks(range(len(preferred)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("RMSE [m]")
    ax.set_title("S8.F2 fusion replay RMSE comparison")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="S8.F2 controlled Villoc relative+absolute fusion replay."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("outputs/villoc/traj01_90deg_stable120m"),
    )
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument(
        "--policies",
        default=(
            "all_reranked_accept_ablation,"
            "strict_a_accept_online,"
            "strict_b_accept_online,"
            "interval_50m_strict_a_accept_online,"
            "interval_100m_strict_a_accept_online,"
            "interval_200m_strict_a_accept_online,"
            "interval_400m_strict_a_accept_online,"
            "oracle_hit40_accept_eval_only"
        ),
    )
    parser.add_argument("--alphas", default="1.0,0.75,0.5,0.25")
    parser.add_argument("--eval-start-index", type=int, default=49)
    parser.add_argument("--thresholds-m", default="10,20,40,80,120")
    parser.add_argument("--sustain-frames", type=int, default=5)
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

    manifest = pd.read_csv(manifest_path)

    require_columns(
        manifest,
        [
            "sequence_frame_id", "token0_id",
            "reference_x_m", "reference_y_m",
            "reference_cumulative_distance_m",
            "prefix_aligned_x_m", "prefix_aligned_y_m",
            "abs_pred_x_m", "abs_pred_y_m",
        ],
        "S8.F1 correction manifest",
    )

    manifest = manifest.sort_values("sequence_frame_id", kind="mergesort").reset_index(drop=True)

    if len(manifest) != 403:
        raise RuntimeError(f"Expected 403 rows, got {len(manifest)}.")

    policies = parse_csv_list(args.policies)
    alphas = parse_float_list(args.alphas)
    thresholds = [float(x) for x in parse_csv_list(args.thresholds_m)]

    for p in policies:
        if p not in manifest.columns:
            raise RuntimeError(f"Policy column missing from manifest: {p}")

    ref = manifest[["reference_x_m", "reference_y_m"]].to_numpy(dtype=float)
    rel = manifest[["prefix_aligned_x_m", "prefix_aligned_y_m"]].to_numpy(dtype=float)
    cumulative = pd.to_numeric(
        manifest["reference_cumulative_distance_m"],
        errors="raise",
    ).to_numpy(float)

    eval_start = min(max(args.eval_start_index, 0), len(manifest) - 1)

    trajectory_rows = []
    summary_rows = []
    crossing_rows = []

    # Baseline without any correction.
    baseline_errors = np.linalg.norm(rel - ref, axis=1)
    baseline_metrics = error_metrics(rel, ref, cumulative, eval_start)

    summary_rows.append({
        "policy": "no_correction_baseline",
        "alpha": 0.0,
        "accepted_corrections": 0,
        "true_accepted_le40_eval_only": 0,
        "false_accepted_eval_only": 0,
        "dangerous_accepted_gt100m_eval_only": 0,
        **baseline_metrics,
    })

    for i, row in manifest.iterrows():
        trajectory_rows.append({
            "policy": "no_correction_baseline",
            "alpha": 0.0,
            "sequence_frame_id": int(row["sequence_frame_id"]),
            "token0_id": int(row["token0_id"]),
            "reference_cumulative_distance_m": float(row["reference_cumulative_distance_m"]),
            "reference_x_m": float(row["reference_x_m"]),
            "reference_y_m": float(row["reference_y_m"]),
            "fused_x_m": float(row["prefix_aligned_x_m"]),
            "fused_y_m": float(row["prefix_aligned_y_m"]),
            "fusion_error_m": float(baseline_errors[i]),
            "correction_applied": False,
            "correction_true_le40_eval_only": False,
            "correction_error_m_eval_only": np.nan,
        })

    for threshold in thresholds:
        crossing_rows.append({
            "policy": "no_correction_baseline",
            "alpha": 0.0,
            **first_sustained_crossing(
                baseline_errors,
                cumulative,
                threshold,
                eval_start,
                args.sustain_frames,
            ),
        })

    for policy in policies:
        accepted = bool_series(manifest[policy]).to_numpy(bool)
        accepted_count = int(accepted.sum())
        hit = bool_series(manifest["abs_hit_le_40m_eval_only"]).to_numpy(bool)
        dangerous = (
            pd.to_numeric(manifest["abs_error_m_eval_only"], errors="coerce").to_numpy(float)
            > 100.0
        )

        for alpha in alphas:
            fused, applied = replay_policy(manifest, policy, alpha)
            errors = np.linalg.norm(fused - ref, axis=1)
            metrics = error_metrics(fused, ref, cumulative, eval_start)

            summary_rows.append({
                "policy": policy,
                "alpha": float(alpha),
                "accepted_corrections": accepted_count,
                "true_accepted_le40_eval_only": int((accepted & hit).sum()),
                "false_accepted_eval_only": int(accepted_count - (accepted & hit).sum()),
                "dangerous_accepted_gt100m_eval_only": int((accepted & dangerous).sum()),
                **metrics,
            })

            for i, row in manifest.iterrows():
                trajectory_rows.append({
                    "policy": policy,
                    "alpha": float(alpha),
                    "sequence_frame_id": int(row["sequence_frame_id"]),
                    "token0_id": int(row["token0_id"]),
                    "reference_cumulative_distance_m": float(row["reference_cumulative_distance_m"]),
                    "reference_x_m": float(row["reference_x_m"]),
                    "reference_y_m": float(row["reference_y_m"]),
                    "fused_x_m": float(fused[i, 0]),
                    "fused_y_m": float(fused[i, 1]),
                    "fusion_error_m": float(errors[i]),
                    "correction_applied": bool(applied[i]),
                    "correction_true_le40_eval_only": bool(hit[i]) if applied[i] else False,
                    "correction_error_m_eval_only": float(row["abs_error_m_eval_only"]) if applied[i] else np.nan,
                })

            for threshold in thresholds:
                crossing_rows.append({
                    "policy": policy,
                    "alpha": float(alpha),
                    **first_sustained_crossing(
                        errors,
                        cumulative,
                        threshold,
                        eval_start,
                        args.sustain_frames,
                    ),
                })

    trajectory_df = pd.DataFrame(trajectory_rows)
    summary_df = pd.DataFrame(summary_rows)
    crossing_df = pd.DataFrame(crossing_rows)

    summary_df = summary_df.sort_values(
        ["rmse_m", "p95_error_m", "final_error_m"],
        kind="mergesort",
    ).reset_index(drop=True)

    trajectory_path = metadata_dir / "s8_f2_fusion_replay_trajectory.csv"
    summary_path = metadata_dir / "s8_f2_fusion_replay_summary.csv"
    crossing_path = metadata_dir / "s8_f2_fusion_threshold_crossings.csv"
    report_path = reports_dir / "s8_f2_controlled_fusion_replay_report.json"

    trajectory_df.to_csv(trajectory_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    crossing_df.to_csv(crossing_path, index=False)

    error_plot = figures_dir / "s8_f2_fusion_error_vs_distance.png"
    xy_plot = figures_dir / "s8_f2_fusion_xy.png"
    bar_plot = figures_dir / "s8_f2_fusion_rmse_summary.png"

    save_error_plot(trajectory_df, summary_df, error_plot)
    save_xy_plot(trajectory_df, summary_df, xy_plot)
    save_summary_bar(summary_df, bar_plot)

    best = summary_df.iloc[0].to_dict()
    baseline = summary_df[
        summary_df["policy"] == "no_correction_baseline"
    ].iloc[0].to_dict()

    report = {
        "stage": "S8.F2",
        "status": "PASS_S8_F2_CONTROLLED_FUSION_REPLAY",
        "purpose": "Controlled position-only replay of XFeat relative trajectory corrected by S8.F1 absolute events.",
        "important_rule": (
            "Correction policy decisions use only S8.F1 policy columns. "
            "Reference/error/hit labels are used only for post-replay evaluation."
        ),
        "inputs": {
            "manifest": str(manifest_path),
        },
        "outputs": {
            "trajectory": str(trajectory_path),
            "summary": str(summary_path),
            "threshold_crossings": str(crossing_path),
            "report": str(report_path),
            "error_plot": str(error_plot),
            "xy_plot": str(xy_plot),
            "bar_plot": str(bar_plot),
        },
        "rows": {
            "manifest": int(len(manifest)),
            "trajectory": int(len(trajectory_df)),
            "summary": int(len(summary_df)),
            "crossings": int(len(crossing_df)),
        },
        "eval_start_index": int(eval_start),
        "eval_start_token0_id": int(manifest.iloc[eval_start]["token0_id"]),
        "policies": policies,
        "alphas": alphas,
        "baseline": {k: json_safe(v) for k, v in baseline.items()},
        "best": {k: json_safe(v) for k, v in best.items()},
    }

    report_path.write_text(
        json.dumps(report, indent=2, default=json_safe),
        encoding="utf-8",
    )

    print("S8.F2 Controlled fusion replay")
    print("-" * 72)
    print("status:", report["status"])
    print("eval start index:", eval_start)
    print()
    print("Top summary rows")
    print("-" * 72)
    cols = [
        "policy", "alpha", "accepted_corrections",
        "true_accepted_le40_eval_only",
        "false_accepted_eval_only",
        "dangerous_accepted_gt100m_eval_only",
        "rmse_m", "mean_error_m", "median_error_m",
        "p95_error_m", "max_error_m",
        "final_error_m", "final_drift_per_100m",
    ]
    print(summary_df[cols].head(20).to_string(index=False))
    print()
    print("Baseline")
    print("-" * 72)
    print(pd.DataFrame([baseline])[cols].to_string(index=False))
    print()
    print("Saved outputs")
    print("-" * 72)
    print(trajectory_path)
    print(summary_path)
    print(crossing_path)
    print(report_path)
    print(error_plot)
    print(xy_plot)
    print(bar_plot)


if __name__ == "__main__":
    main()
