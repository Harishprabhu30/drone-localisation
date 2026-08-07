'''
Run command :

ROOT=outputs/villoc/traj01_90deg_stable120m

python scripts/villoc/s8_fusion/s8_f1_build_absolute_correction_manifest.py \
  --root "$ROOT" \
  --intervals-m 50,100,200,400 \
  2>&1 | tee "$ROOT/logs/s8_relative_absolute_fusion/s8_f1_absolute_correction_manifest.log"

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


def parse_intervals(value: str) -> list[float]:
    out = []
    for x in value.split(","):
        x = x.strip()
        if x:
            out.append(float(x))
    if not out:
        raise ValueError("At least one interval is required.")
    return sorted(set(out))


def require_columns(df: pd.DataFrame, cols: list[str], name: str) -> None:
    missing = sorted(set(cols) - set(df.columns))
    if missing:
        raise RuntimeError(f"{name} missing columns: {missing}")


def bool_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False)
    return (
        s.astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes", "y", "t"})
    )


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


def safe_float_series(df: pd.DataFrame, col: str, default: float = np.nan) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def select_reranked_top1(qsum: pd.DataFrame, cand: pd.DataFrame) -> pd.DataFrame:
    q = qsum.copy()
    c = cand.copy()

    q["query_id"] = pd.to_numeric(q["query_id"], errors="raise").astype(int)
    c["query_id"] = pd.to_numeric(c["query_id"], errors="raise").astype(int)

    # First preference: exact reranked tile ID from query summary.
    selected = q.merge(
        c,
        left_on=["query_id", "reranked_top1_tile_id"],
        right_on=["query_id", "tile_id"],
        how="left",
        suffixes=("_qsum", "_cand"),
        validate="one_to_one",
    )

    missing = selected["tile_id"].isna()
    if missing.any():
        # Fallback: candidate with hybrid_rank == 1.
        if "hybrid_rank" not in c.columns:
            raise RuntimeError(
                "Could not match reranked_top1_tile_id and candidate scores have no hybrid_rank."
            )

        h = c.copy()
        h["hybrid_rank"] = pd.to_numeric(h["hybrid_rank"], errors="coerce")
        h = h[h["hybrid_rank"] == 1].copy()
        if h["query_id"].duplicated().any():
            raise RuntimeError("hybrid_rank==1 has duplicate query IDs.")

        fallback = q.loc[missing, ["query_id"]].merge(
            h,
            on="query_id",
            how="left",
            validate="one_to_one",
        )

        if fallback["tile_id"].isna().any():
            bad = fallback.loc[fallback["tile_id"].isna(), "query_id"].head(10).tolist()
            raise RuntimeError(f"Could not select reranked candidate for query IDs: {bad}")

        # Update candidate-side columns for missing rows.
        selected = selected.set_index("query_id")
        fallback = fallback.set_index("query_id")
        for col in fallback.columns:
            if col not in selected.columns:
                selected[col] = np.nan
            selected.loc[fallback.index, col] = fallback[col]
        selected = selected.reset_index()

    if len(selected) != len(q):
        raise RuntimeError(f"Selected rows mismatch: {len(selected)} vs {len(q)}")

    if selected["query_id"].duplicated().any():
        raise RuntimeError("Selected reranked candidates contain duplicate query IDs.")

    return selected


def mark_interval_attempts(dist: pd.Series, interval_m: float) -> pd.Series:
    d = pd.to_numeric(dist, errors="coerce").to_numpy(float)
    out = np.zeros(len(d), dtype=bool)
    if len(d) == 0:
        return pd.Series(out)

    last_attempt = d[0]
    out[0] = True

    for i in range(1, len(d)):
        if not np.isfinite(d[i]):
            continue
        if d[i] - last_attempt >= interval_m:
            out[i] = True
            last_attempt = d[i]

    return pd.Series(out)


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

    records = accepted.to_dict(orient="records")
    for prev, cur in zip(records[:-1], records[1:]):
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


def summarize_policy(df: pd.DataFrame, gap_df: pd.DataFrame, policy: str) -> dict[str, Any]:
    accepted_mask = bool_series(df[policy])
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

    policy_gaps = gap_df[gap_df["policy"] == policy].copy()
    between = policy_gaps.loc[policy_gaps["gap_type"] == "between_corrections", "gap_m"].dropna()
    coverage = policy_gaps["gap_m"].dropna()

    attempt_col = policy.replace("_accept_online", "_attempt_online")
    attempts = None
    if attempt_col in df.columns:
        attempts = int(bool_series(df[attempt_col]).sum())

    return {
        "policy": policy,
        "attempts_online": attempts,
        "accepted": accepted_count,
        "true_accepts_le40_eval_only": true_count,
        "contains_body_accepts_eval_only": contains_count,
        "false_accepts_eval_only": false_count,
        "dangerous_false_accepts_gt100m_eval_only": dangerous_count,
        "precision_le40_eval_only": true_count / accepted_count if accepted_count else None,
        "contains_precision_eval_only": contains_count / accepted_count if accepted_count else None,
        "median_accepted_error_m_eval_only": float(accepted_errors.median()) if len(accepted_errors) else None,
        "p95_accepted_error_m_eval_only": float(accepted_errors.quantile(0.95)) if len(accepted_errors) else None,
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
    ax.set_yticklabels(policies)
    ax.set_xlabel("Reference cumulative distance [m] — evaluation only")
    ax.set_ylabel("Correction policy")
    ax.set_title("S8.F1 accepted absolute correction events along Villoc traj01")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="S8.F1: Build Villoc relative+absolute correction manifest."
    )
    parser.add_argument("--root", type=Path, default=Path("outputs/villoc/traj01_90deg_stable120m"))
    parser.add_argument("--intervals-m", default="50,100,200,400")
    parser.add_argument("--hit-threshold-m", type=float, default=40.0)
    parser.add_argument("--relative-csv", type=Path, default=None)
    parser.add_argument("--query-manifest", type=Path, default=None)
    parser.add_argument("--absolute-query-summary", type=Path, default=None)
    parser.add_argument("--absolute-candidate-scores", type=Path, default=None)
    args = parser.parse_args()

    root = args.root
    intervals = parse_intervals(args.intervals_m)

    relative_csv = args.relative_csv or (
        root / "metadata/s8_xfeat_relative_frontend/s8r4_xfeat_relative_trajectory_aligned_eval_only.csv"
    )
    query_manifest_path = args.query_manifest or (
        root / "metadata/s8_10b_canonical_uav_query_manifest.csv"
    )
    abs_qsum_path = args.absolute_query_summary or (
        root / "reports/s8_12e1_top20_verifier_reranker/512_s256_orb_hybrid_top20_img518/s8_12e1_query_summary.csv"
    )
    abs_cand_path = args.absolute_candidate_scores or (
        root / "reports/s8_12e1_top20_verifier_reranker/512_s256_orb_hybrid_top20_img518/s8_12e1_all_candidate_verifier_scores.csv"
    )

    metadata_dir = root / "metadata/s8_relative_absolute_fusion"
    reports_dir = root / "reports/s8_relative_absolute_fusion"
    figures_dir = root / "figures/s8_relative_absolute_fusion"
    logs_dir = root / "logs/s8_relative_absolute_fusion"

    for d in [metadata_dir, reports_dir, figures_dir, logs_dir]:
        d.mkdir(parents=True, exist_ok=True)

    rel = pd.read_csv(relative_csv)
    qman = pd.read_csv(query_manifest_path)
    qsum = pd.read_csv(abs_qsum_path)
    cand = pd.read_csv(abs_cand_path)

    require_columns(
        rel,
        [
            "sequence_frame_id", "token0_id",
            "reference_x_m", "reference_y_m",
            "reference_cumulative_distance_m",
            "prefix_aligned_x_m", "prefix_aligned_y_m", "prefix_locked_error_m",
        ],
        "relative trajectory",
    )
    require_columns(
        qman,
        ["query_id", "token0_id", "frame_index", "easting", "northing"],
        "query manifest",
    )
    require_columns(
        qsum,
        [
            "query_id",
            "reranked_top1_tile_id",
            "reranked_top1_original_rank",
            "reranked_top1_error_m",
            "reranked_top1_hit_le_40m",
            "reranked_top1_contains_body",
            "reranked_top1_inliers",
            "reranked_top1_inlier_ratio",
            "reranked_top1_query_inlier_coverage",
            "reranked_top1_verifier_score",
            "reranked_top1_hybrid_score",
            "dino_contains_body_top20",
            "dino_hit_le_40m_top20",
        ],
        "absolute query summary",
    )
    require_columns(
        cand,
        [
            "query_id", "tile_id",
            "center_easting", "center_northing",
            "left_easting", "right_easting", "bottom_northing", "top_northing",
            "candidate_hit_le_40m", "candidate_contains_body",
            "hybrid_rank",
        ],
        "absolute candidate scores",
    )

    for df, name in [(rel, "relative"), (qman, "query_manifest"), (qsum, "query_summary")]:
        if "token0_id" in df.columns:
            df["token0_id"] = pd.to_numeric(df["token0_id"], errors="raise").astype(int)
        if "query_id" in df.columns:
            df["query_id"] = pd.to_numeric(df["query_id"], errors="raise").astype(int)

    selected = select_reranked_top1(qsum, cand)

    # Query-origin converts EPSG:3346 map coordinates into the same local frame as reference_x/reference_y.
    qman = qman.sort_values("query_id", kind="mergesort").reset_index(drop=True)
    origin_easting = float(qman.iloc[0]["easting"])
    origin_northing = float(qman.iloc[0]["northing"])

    base = rel.merge(
        qman[
            [
                "query_id", "token0_id", "frame_index",
                "easting", "northing",
                "latitude", "longitude",
                "source_frame_cnt", "canonical_query_filename",
            ]
        ],
        on="token0_id",
        how="inner",
        validate="one_to_one",
    )

    manifest = base.merge(
        selected,
        on="query_id",
        how="inner",
        validate="one_to_one",
    )

    # Defensive cleanup if any upstream table accidentally carried token0_id.
    if "token0_id" not in manifest.columns:
        if "token0_id_x" in manifest.columns:
            manifest["token0_id"] = manifest["token0_id_x"]
        elif "token0_id_y" in manifest.columns:
            manifest["token0_id"] = manifest["token0_id_y"]

    if len(manifest) != 403:
        raise RuntimeError(f"Expected 403 manifest rows, got {len(manifest)}.")

    manifest = manifest.sort_values("sequence_frame_id", kind="mergesort").reset_index(drop=True)

    # Absolute prediction in both EPSG:3346 and local trajectory coordinates.
    manifest["abs_pred_easting_m"] = pd.to_numeric(manifest["center_easting"], errors="coerce")
    manifest["abs_pred_northing_m"] = pd.to_numeric(manifest["center_northing"], errors="coerce")
    manifest["abs_pred_x_m"] = manifest["abs_pred_easting_m"] - origin_easting
    manifest["abs_pred_y_m"] = manifest["abs_pred_northing_m"] - origin_northing

    manifest["query_easting_m"] = pd.to_numeric(manifest["easting"], errors="coerce")
    manifest["query_northing_m"] = pd.to_numeric(manifest["northing"], errors="coerce")

    manifest["abs_error_m_eval_only"] = pd.to_numeric(
        manifest["reranked_top1_error_m"],
        errors="coerce",
    )
    manifest["abs_hit_le_40m_eval_only"] = bool_series(manifest["reranked_top1_hit_le_40m"])
    manifest["abs_contains_body_eval_only"] = bool_series(manifest["reranked_top1_contains_body"])
    manifest["abs_dangerous_gt100m_eval_only"] = manifest["abs_error_m_eval_only"] > 100.0

    # Online-available evidence only. No error/oracle/reference labels are used here.
    manifest["abs_original_rank_online"] = pd.to_numeric(
        manifest["reranked_top1_original_rank"],
        errors="coerce",
    )
    manifest["abs_inliers_online"] = pd.to_numeric(
        manifest["reranked_top1_inliers"],
        errors="coerce",
    )
    manifest["abs_inlier_ratio_online"] = pd.to_numeric(
        manifest["reranked_top1_inlier_ratio"],
        errors="coerce",
    )
    manifest["abs_query_coverage_online"] = pd.to_numeric(
        manifest["reranked_top1_query_inlier_coverage"],
        errors="coerce",
    )
    manifest["abs_verifier_score_online"] = pd.to_numeric(
        manifest["reranked_top1_verifier_score"],
        errors="coerce",
    )
    manifest["abs_hybrid_score_online"] = pd.to_numeric(
        manifest["reranked_top1_hybrid_score"],
        errors="coerce",
    )

    if "homography_ok" in manifest.columns:
        manifest["abs_homography_ok_online"] = bool_series(manifest["homography_ok"])
    else:
        manifest["abs_homography_ok_online"] = True

    manifest["all_reranked_accept_ablation"] = True

    manifest["strict_a_accept_online"] = (
        manifest["abs_homography_ok_online"]
        & (manifest["abs_inliers_online"] >= 25)
        & (manifest["abs_inlier_ratio_online"] >= 0.30)
        & (manifest["abs_query_coverage_online"] >= 0.10)
        & (manifest["abs_original_rank_online"] <= 15)
    )

    manifest["strict_b_accept_online"] = (
        manifest["abs_homography_ok_online"]
        & (manifest["abs_inliers_online"] >= 40)
        & (manifest["abs_inlier_ratio_online"] >= 0.35)
        & (manifest["abs_query_coverage_online"] >= 0.12)
        & (manifest["abs_original_rank_online"] <= 12)
    )

    manifest["oracle_hit40_accept_eval_only"] = manifest["abs_hit_le_40m_eval_only"]

    for interval in intervals:
        interval_tag = f"{int(interval)}m" if float(interval).is_integer() else f"{interval:g}m"
        attempt_col = f"interval_{interval_tag}_attempt_online"
        accept_col = f"interval_{interval_tag}_strict_a_accept_online"

        manifest[attempt_col] = mark_interval_attempts(
            manifest["reference_cumulative_distance_m"],
            interval,
        ).to_numpy(bool)

        manifest[accept_col] = manifest[attempt_col] & bool_series(manifest["strict_a_accept_online"])

    # Reorder useful columns first.
    front = [
        "sequence_frame_id", "token0_id", "query_id", "frame_index",
        "reference_cumulative_distance_m",
        "reference_x_m", "reference_y_m",
        "prefix_aligned_x_m", "prefix_aligned_y_m", "prefix_locked_error_m",
        "query_easting_m", "query_northing_m",
        "abs_pred_easting_m", "abs_pred_northing_m", "abs_pred_x_m", "abs_pred_y_m",
        "reranked_top1_tile_id",
        "abs_original_rank_online",
        "abs_inliers_online", "abs_inlier_ratio_online", "abs_query_coverage_online",
        "abs_verifier_score_online", "abs_hybrid_score_online",
        "abs_homography_ok_online",
        "abs_error_m_eval_only", "abs_hit_le_40m_eval_only",
        "abs_contains_body_eval_only", "abs_dangerous_gt100m_eval_only",
        "all_reranked_accept_ablation",
        "strict_a_accept_online", "strict_b_accept_online",
        "oracle_hit40_accept_eval_only",
    ]
    front = [c for c in front if c in manifest.columns]
    manifest = manifest[front + [c for c in manifest.columns if c not in front]]

    policy_columns = [
        "all_reranked_accept_ablation",
        "strict_a_accept_online",
        "strict_b_accept_online",
    ]
    for interval in intervals:
        interval_tag = f"{int(interval)}m" if float(interval).is_integer() else f"{interval:g}m"
        policy_columns.append(f"interval_{interval_tag}_strict_a_accept_online")
    policy_columns.append("oracle_hit40_accept_eval_only")

    gap_rows = []
    for policy in policy_columns:
        gap_rows.extend(policy_gap_rows(manifest, policy))
    gap_df = pd.DataFrame(gap_rows)

    policy_summary = pd.DataFrame([
        summarize_policy(manifest, gap_df, policy)
        for policy in policy_columns
    ])

    manifest_path = metadata_dir / "s8_f1_absolute_correction_manifest.csv"
    summary_path = metadata_dir / "s8_f1_policy_summary.csv"
    gaps_path = metadata_dir / "s8_f1_policy_gaps.csv"
    report_path = reports_dir / "s8_f1_absolute_correction_manifest_report.json"
    figure_path = figures_dir / "s8_f1_policy_correction_distribution.png"

    manifest.to_csv(manifest_path, index=False)
    policy_summary.to_csv(summary_path, index=False)
    gap_df.to_csv(gaps_path, index=False)
    save_policy_distribution_plot(manifest, policy_columns, figure_path)

    report = {
        "stage": "S8.F1",
        "status": "PASS_S8_F1_ABSOLUTE_CORRECTION_MANIFEST",
        "purpose": "Build Villoc absolute correction manifest for later relative+absolute replay.",
        "important_rule": (
            "Online policies use only reranker evidence. Reference coordinates, errors, "
            "hit labels, and oracle/body containment are evaluation-only."
        ),
        "inputs": {
            "relative_csv": str(relative_csv),
            "query_manifest": str(query_manifest_path),
            "absolute_query_summary": str(abs_qsum_path),
            "absolute_candidate_scores": str(abs_cand_path),
        },
        "outputs": {
            "manifest": str(manifest_path),
            "policy_summary": str(summary_path),
            "policy_gaps": str(gaps_path),
            "report": str(report_path),
            "figure": str(figure_path),
        },
        "rows": {
            "relative": int(len(rel)),
            "query_manifest": int(len(qman)),
            "absolute_query_summary": int(len(qsum)),
            "absolute_candidate_scores": int(len(cand)),
            "correction_manifest": int(len(manifest)),
        },
        "origin_epsg3346": {
            "easting_m": origin_easting,
            "northing_m": origin_northing,
        },
        "policy_columns": policy_columns,
        "policy_summary": policy_summary.where(pd.notna(policy_summary), None).to_dict(orient="records"),
    }

    report_path.write_text(
        json.dumps(report, indent=2, default=json_safe),
        encoding="utf-8",
    )

    print("S8.F1 Absolute correction manifest")
    print("-" * 72)
    print("status:", report["status"])
    print("rows:", len(manifest))
    print()
    print("Policy summary")
    print("-" * 72)
    print(policy_summary.to_string(index=False))
    print()
    print("Saved outputs")
    print("-" * 72)
    print(manifest_path)
    print(summary_path)
    print(gaps_path)
    print(report_path)
    print(figure_path)


if __name__ == "__main__":
    main()
