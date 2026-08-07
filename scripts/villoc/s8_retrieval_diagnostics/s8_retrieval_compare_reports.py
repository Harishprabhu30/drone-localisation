#!/usr/bin/env python3
"""
General retrieval-diagnostics report comparison.

This script compares any number of retrieval diagnostic runs produced by:

    scripts/villoc/s8_retrieval_diagnostics/s8_retrieval_diagnostics.py

It is intentionally dataset-general:
- 45° vs 90°
- new angle vs old angle
- new AOI vs old AOI
- same dataset with different variants
- same dataset with different descriptor tags

Inputs are report.json files from diagnostic output folders.

Example:
python scripts/villoc/s8_retrieval_diagnostics/s8_retrieval_compare_reports.py \
  --run 45deg:outputs/villoc/45_deg/diagnostics/s8_retrieval_diagnostics/1024_s512/report.json \
  --run 90deg:outputs/villoc/90_deg/diagnostics/s8_retrieval_diagnostics/1024_s512/report.json \
  --oracle-k 20 \
  --out-root outputs/villoc/reports/s8_retrieval_comparisons/villoc_45_vs_90_1024_s512

2. running traj01 villoc dataset:

export PYTHONPATH=$PWD/src

CFG=configs/dataset_villoc_traj01_90deg_stable120m.yaml
ROOT=outputs/villoc/traj01_90deg_stable120m
TAG=dinov2_vits14_img518_center_square_avgpatch_cpu

OUT_ROOT=$ROOT/retrieval/s8_12d_retrieval_diagnostics

mkdir -p "$OUT_ROOT"
mkdir -p "$ROOT/logs/s8_12d_retrieval_diagnostics"

for VARIANT in 512_s256 1024_s512 1024_s256; do
  echo
  echo "============================================================"
  echo "S8.12D diagnostics: $VARIANT"
  echo "============================================================"

  python scripts/villoc/s8_retrieval_diagnostics/s8_retrieval_diagnostics.py \
    --config "$CFG" \
    --variant "$VARIANT" \
    --tag "$TAG" \
    --query-eval-csv "$ROOT/retrieval/s8_11d/s8_11d_query_eval_${VARIANT}_${TAG}.csv" \
    --topk-csv "$ROOT/retrieval/s8_11d/s8_11d_topk_${VARIANT}_${TAG}.csv" \
    --query-manifest-csv "$ROOT/metadata/s8_10b_canonical_uav_query_manifest.csv" \
    --tile-index-csv "outputs/villoc/90_deg/metadata/s8_9_satellite_tile_index_${VARIANT}.csv" \
    --out-root "$OUT_ROOT" \
    --oracle-k 20 \
    --max-panels 16 \
    --top-n-tiles 5 \
    --high-conf-quantile 0.90 \
    2>&1 | tee "$ROOT/logs/s8_12d_retrieval_diagnostics/s8_12d_${VARIANT}_${TAG}.log"
done

"""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import re
from typing import Any

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path.cwd()


def root_join(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def safe_label(label: str) -> str:
    label = label.strip()
    label = re.sub(r"[^A-Za-z0-9_.-]+", "_", label)
    return label.strip("_") or "run"


def parse_run_arg(value: str) -> tuple[str, Path]:
    if ":" not in value:
        raise ValueError(
            f"Invalid --run value: {value!r}. Expected LABEL:PATH_TO_REPORT_JSON"
        )
    label, path = value.split(":", 1)
    return safe_label(label), root_join(path)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_or_empty(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def normalize_id(x: Any) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    try:
        f = float(s)
        if f.is_integer():
            return str(int(f))
    except Exception:
        pass
    return s


def as_bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s.fillna(False)
    return s.astype(str).str.lower().isin(["1", "true", "yes", "y"])


def maybe_float(x: Any) -> float | None:
    try:
        if pd.isna(x):
            return None
        return float(x)
    except Exception:
        return None


def infer_diag_root(report_path: Path) -> Path:
    # report.json sits directly inside the diagnostics variant folder.
    return report_path.parent


def count_rows(path: Path) -> int:
    df = read_csv_or_empty(path)
    return int(len(df))


def load_bucket_counts(diag_root: Path) -> dict[str, int]:
    csv_dir = diag_root / "csv"
    counts: dict[str, int] = {}

    if not csv_dir.exists():
        return counts

    for path in sorted(csv_dir.glob("*.csv")):
        name = path.stem
        counts[name] = count_rows(path)

    return counts


def load_all_queries(diag_root: Path) -> pd.DataFrame:
    path = diag_root / "csv" / "all_queries_with_diagnostic_flags.csv"
    df = read_csv_or_empty(path)

    if df.empty:
        return df

    if "query_id" not in df.columns:
        for c in ["token0_id", "uav_query_id", "sample_id"]:
            if c in df.columns:
                df["query_id"] = df[c]
                break

    if "query_id" in df.columns:
        df["query_id"] = df["query_id"].map(normalize_id)

    return df


def classify_primary_bucket(row: pd.Series, oracle_k: int) -> str:
    top1 = False
    if "top1_is_oracle" in row.index:
        top1 = str(row["top1_is_oracle"]).lower() in ["1", "true", "yes", "y"]

    if top1:
        return "good_top1"

    k5_col = "recall_at_5"
    k_col = f"recall_at_{oracle_k}"

    if k5_col in row.index and str(row[k5_col]).lower() in ["1", "true", "yes", "y"]:
        return "oracle_in_top5_but_not_top1"

    if k_col in row.index and str(row[k_col]).lower() in ["1", "true", "yes", "y"]:
        return f"oracle_in_top{oracle_k}_but_not_top1"

    return f"oracle_missing_top{oracle_k}"


def summarize_run(label: str, report_path: Path, oracle_k: int) -> dict[str, Any]:
    report = read_json(report_path)
    diag_root = infer_diag_root(report_path)
    bucket_counts = load_bucket_counts(diag_root)
    allq = load_all_queries(diag_root)

    row: dict[str, Any] = {
        "label": label,
        "report_json": str(report_path),
        "diag_root": str(diag_root),
        "variant": report.get("variant"),
        "tag": report.get("tag"),
        "view_label": report.get("view_label"),
        "config": report.get("config"),
        "query_count": int(bucket_counts.get("all_queries_with_diagnostic_flags", len(allq))),
    }

    for bucket, count in bucket_counts.items():
        row[f"bucket_{bucket}"] = int(count)

    if not allq.empty:
        if "top1_is_oracle" in allq.columns:
            top1 = as_bool_series(allq["top1_is_oracle"])
            row["top1_hits"] = int(top1.sum())
            row["top1_hit_rate"] = float(top1.mean())

        recall_col = f"recall_at_{oracle_k}"
        if recall_col in allq.columns:
            rk = as_bool_series(allq[recall_col])
            row[f"recall_at_{oracle_k}_hits"] = int(rk.sum())
            row[f"recall_at_{oracle_k}"] = float(rk.mean())

        for k in [1, 5, 10, 20, 50, 100]:
            c = f"recall_at_{k}"
            if c in allq.columns:
                s = as_bool_series(allq[c])
                row[f"recall_at_{k}_hits"] = int(s.sum())
                row[f"recall_at_{k}"] = float(s.mean())

        if "first_oracle_rank" in allq.columns:
            vals = pd.to_numeric(allq["first_oracle_rank"], errors="coerce").dropna()
            if len(vals):
                row["first_oracle_rank_median"] = float(vals.median())
                row["first_oracle_rank_p95"] = float(vals.quantile(0.95))

        if "top1_center_error_m" in allq.columns:
            vals = pd.to_numeric(allq["top1_center_error_m"], errors="coerce").dropna()
            if len(vals):
                row["top1_center_error_median_m"] = float(vals.median())
                row["top1_center_error_rmse_m"] = float(np.sqrt(np.mean(np.square(vals))))
                row["top1_center_error_p95_m"] = float(vals.quantile(0.95))

    # Fallback from bucket counts if all_queries lacks top1 rate.
    if "top1_hits" not in row and "good_top1" in bucket_counts:
        qn = max(1, int(row["query_count"]))
        row["top1_hits"] = int(bucket_counts["good_top1"])
        row["top1_hit_rate"] = float(bucket_counts["good_top1"] / qn)

    if f"recall_at_{oracle_k}_hits" not in row:
        good = bucket_counts.get("good_top1", 0)
        in_k = bucket_counts.get(f"oracle_in_top{oracle_k}_but_not_top1", 0)
        qn = max(1, int(row["query_count"]))
        row[f"recall_at_{oracle_k}_hits"] = int(good + in_k)
        row[f"recall_at_{oracle_k}"] = float((good + in_k) / qn)

    return row


def prepare_query_table(label: str, report_path: Path, oracle_k: int) -> pd.DataFrame:
    diag_root = infer_diag_root(report_path)
    df = load_all_queries(diag_root)

    if df.empty:
        return df

    df = df.copy()
    df["run_label"] = label

    if "query_id" not in df.columns:
        return df

    df["query_id"] = df["query_id"].map(normalize_id)

    if "top1_is_oracle" in df.columns:
        df["top1_good"] = as_bool_series(df["top1_is_oracle"])
    else:
        good_path = diag_root / "csv" / "good_top1.csv"
        good = read_csv_or_empty(good_path)
        if "query_id" in good.columns:
            good_ids = set(good["query_id"].map(normalize_id))
            df["top1_good"] = df["query_id"].isin(good_ids)
        else:
            df["top1_good"] = False

    recall_col = f"recall_at_{oracle_k}"
    if recall_col in df.columns:
        df[f"oracle_in_top{oracle_k}"] = as_bool_series(df[recall_col])
    else:
        df[f"oracle_in_top{oracle_k}"] = False

    df["primary_bucket"] = df.apply(lambda r: classify_primary_bucket(r, oracle_k), axis=1)

    keep = [
        "query_id",
        "run_label",
        "primary_bucket",
        "top1_good",
        f"oracle_in_top{oracle_k}",
        "top1_tile_id",
        "first_oracle_rank",
        "top1_center_error_m",
        "top1_score",
        "top1_similarity",
        "rel_alt_m",
        "gb_yaw_deg",
        "gb_pitch_deg",
        "source_frame_cnt",
        "timestamp_s",
    ]
    keep = [c for c in keep if c in df.columns]
    return df[keep].copy()


def write_markdown_summary(summary_df: pd.DataFrame, out_path: Path, oracle_k: int) -> None:
    preferred_cols = [
        "label",
        "variant",
        "query_count",
        "top1_hits",
        "top1_hit_rate",
        f"recall_at_{oracle_k}_hits",
        f"recall_at_{oracle_k}",
        "bucket_good_top1",
        "bucket_bad_top1",
        "bucket_oracle_in_top5_but_not_top1",
        f"bucket_oracle_in_top{oracle_k}_but_not_top1",
        f"bucket_oracle_missing_top{oracle_k}",
        "bucket_high_conf_wrong_top1",
        "bucket_easy_correct",
        "bucket_hard_correct",
        "first_oracle_rank_median",
        "top1_center_error_median_m",
    ]
    cols = [c for c in preferred_cols if c in summary_df.columns]
    table = summary_df[cols].copy()

    for c in table.columns:
        if table[c].dtype.kind in "fc":
            table[c] = table[c].map(lambda x: "" if pd.isna(x) else round(float(x), 4))

    md = []
    md.append("# Retrieval diagnostics comparison")
    md.append("")
    md.append(f"Oracle pool threshold: Top-{oracle_k}")
    md.append("")
    md.append(table.to_markdown(index=False))
    md.append("")
    md.append("Notes:")
    md.append("")
    md.append("- `top1_hit_rate` means DINO Top-1 is an oracle tile.")
    md.append(f"- `recall_at_{oracle_k}` means an oracle tile exists somewhere inside Top-{oracle_k}.")
    md.append("- Coordinates/oracles are diagnostic/evaluation labels only, not retrieval inputs.")
    out_path.write_text("\n".join(md), encoding="utf-8")


def plot_bucket_counts(summary_df: pd.DataFrame, out_path: Path, oracle_k: int) -> None:
    buckets = [
        "bucket_good_top1",
        "bucket_bad_top1",
        "bucket_oracle_in_top5_but_not_top1",
        f"bucket_oracle_in_top{oracle_k}_but_not_top1",
        f"bucket_oracle_missing_top{oracle_k}",
        "bucket_high_conf_wrong_top1",
    ]
    buckets = [b for b in buckets if b in summary_df.columns]

    if not buckets:
        return

    plot_df = summary_df[["label", *buckets]].copy()
    x = np.arange(len(plot_df["label"]))
    width = 0.8 / max(1, len(buckets))

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, b in enumerate(buckets):
        vals = pd.to_numeric(plot_df[b], errors="coerce").fillna(0).to_numpy()
        ax.bar(x + i * width, vals, width, label=b.replace("bucket_", ""))

    ax.set_xticks(x + width * (len(buckets) - 1) / 2)
    ax.set_xticklabels(plot_df["label"].tolist(), rotation=0)
    ax.set_ylabel("Query count")
    ax.set_title("Retrieval diagnostic bucket counts")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_rates(summary_df: pd.DataFrame, out_path: Path, oracle_k: int) -> None:
    rate_cols = [
        "top1_hit_rate",
        f"recall_at_{oracle_k}",
    ]
    rate_cols = [c for c in rate_cols if c in summary_df.columns]
    if not rate_cols:
        return

    x = np.arange(len(summary_df["label"]))
    width = 0.8 / len(rate_cols)

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, c in enumerate(rate_cols):
        vals = pd.to_numeric(summary_df[c], errors="coerce").fillna(0).to_numpy() * 100
        ax.bar(x + i * width, vals, width, label=c)

    ax.set_xticks(x + width * (len(rate_cols) - 1) / 2)
    ax.set_xticklabels(summary_df["label"].tolist())
    ax.set_ylabel("Rate (%)")
    ax.set_title(f"Top-1 hit rate and Recall@{oracle_k}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def build_pairwise_outputs(
    *,
    query_tables: dict[str, pd.DataFrame],
    out_dir: Path,
    oracle_k: int,
    join_key: str,
) -> list[dict[str, Any]]:
    labels = list(query_tables.keys())
    pair_reports: list[dict[str, Any]] = []

    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            left_label = labels[i]
            right_label = labels[j]
            left = query_tables[left_label]
            right = query_tables[right_label]

            if left.empty or right.empty:
                continue
            if join_key not in left.columns or join_key not in right.columns:
                continue

            left2 = left.add_prefix(f"{left_label}__").rename(
                columns={f"{left_label}__{join_key}": join_key}
            )
            right2 = right.add_prefix(f"{right_label}__").rename(
                columns={f"{right_label}__{join_key}": join_key}
            )

            merged = left2.merge(right2, on=join_key, how="inner")
            if merged.empty:
                continue

            lg = f"{left_label}__top1_good"
            rg = f"{right_label}__top1_good"
            lk = f"{left_label}__oracle_in_top{oracle_k}"
            rk = f"{right_label}__oracle_in_top{oracle_k}"

            if lg in merged.columns and rg in merged.columns:
                merged["both_top1_good"] = merged[lg].astype(bool) & merged[rg].astype(bool)
                merged[f"{left_label}_only_top1_good"] = merged[lg].astype(bool) & ~merged[rg].astype(bool)
                merged[f"{right_label}_only_top1_good"] = ~merged[lg].astype(bool) & merged[rg].astype(bool)
                merged["neither_top1_good"] = ~merged[lg].astype(bool) & ~merged[rg].astype(bool)

            if lk in merged.columns and rk in merged.columns:
                merged[f"both_oracle_in_top{oracle_k}"] = merged[lk].astype(bool) & merged[rk].astype(bool)
                merged[f"{left_label}_only_oracle_in_top{oracle_k}"] = merged[lk].astype(bool) & ~merged[rk].astype(bool)
                merged[f"{right_label}_only_oracle_in_top{oracle_k}"] = ~merged[lk].astype(bool) & merged[rk].astype(bool)
                merged[f"neither_oracle_in_top{oracle_k}"] = ~merged[lk].astype(bool) & ~merged[rk].astype(bool)

            pair_name = f"{left_label}_vs_{right_label}"
            pair_csv = out_dir / f"pairwise_{pair_name}_by_{join_key}.csv"
            merged.to_csv(pair_csv, index=False)

            report = {
                "pair": pair_name,
                "left": left_label,
                "right": right_label,
                "join_key": join_key,
                "common_queries": int(len(merged)),
                "csv": str(pair_csv),
            }

            for c in [
                "both_top1_good",
                f"{left_label}_only_top1_good",
                f"{right_label}_only_top1_good",
                "neither_top1_good",
                f"both_oracle_in_top{oracle_k}",
                f"{left_label}_only_oracle_in_top{oracle_k}",
                f"{right_label}_only_oracle_in_top{oracle_k}",
                f"neither_oracle_in_top{oracle_k}",
            ]:
                if c in merged.columns:
                    report[c] = int(merged[c].sum())

            pair_reports.append(report)

            # Bucket transition matrix.
            lb = f"{left_label}__primary_bucket"
            rb = f"{right_label}__primary_bucket"
            if lb in merged.columns and rb in merged.columns:
                matrix = pd.crosstab(merged[lb], merged[rb])
                matrix_csv = out_dir / f"bucket_transition_{pair_name}_by_{join_key}.csv"
                matrix.to_csv(matrix_csv)
                report["bucket_transition_csv"] = str(matrix_csv)

    return pair_reports


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="Run definition as LABEL:PATH_TO_REPORT_JSON. Can be repeated.",
    )
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--oracle-k", type=int, default=20)
    parser.add_argument(
        "--join-key",
        default="query_id",
        help="Key for optional pairwise overlap. Only valid if shared meaningfully across runs.",
    )
    parser.add_argument(
        "--no-pairwise",
        action="store_true",
        help="Only produce aggregate comparison, no query-level pairwise comparison.",
    )
    args = parser.parse_args()

    out_dir = root_join(args.out_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    runs = [parse_run_arg(v) for v in args.run]
    if len(runs) < 2:
        raise ValueError("Provide at least two --run entries.")

    print("General retrieval diagnostics report comparison")
    print("-" * 80)
    print("out_root:", out_dir)
    print("oracle_k:", args.oracle_k)
    print("runs:")
    for label, path in runs:
        print(f"  {label}: {path}")

    summary_rows = []
    query_tables: dict[str, pd.DataFrame] = {}

    for label, report_path in runs:
        summary_rows.append(summarize_run(label, report_path, args.oracle_k))
        query_tables[label] = prepare_query_table(label, report_path, args.oracle_k)

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = out_dir / "run_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    markdown_path = out_dir / "run_summary.md"
    write_markdown_summary(summary_df, markdown_path, args.oracle_k)

    plot_bucket_counts(summary_df, fig_dir / "bucket_counts.png", args.oracle_k)
    plot_rates(summary_df, fig_dir / f"top1_vs_recall_at_{args.oracle_k}.png", args.oracle_k)

    pair_reports = []
    if not args.no_pairwise:
        pair_reports = build_pairwise_outputs(
            query_tables=query_tables,
            out_dir=out_dir,
            oracle_k=args.oracle_k,
            join_key=args.join_key,
        )

    report = {
        "status": "PASS_RETRIEVAL_DIAGNOSTICS_COMPARISON",
        "oracle_k": args.oracle_k,
        "out_root": str(out_dir),
        "runs": [{"label": label, "report_json": str(path)} for label, path in runs],
        "outputs": {
            "run_summary_csv": str(summary_csv),
            "run_summary_md": str(markdown_path),
            "figures_dir": str(fig_dir),
        },
        "pairwise_reports": pair_reports,
        "notes": {
            "pairwise_warning": (
                "Pairwise query_id comparison is meaningful only if the compared runs share "
                "the same query identity/sampling semantics. For different flights or angle "
                "recordings, aggregate comparison is safer unless query IDs have been aligned."
            ),
            "gt_rule": "Oracle/reference labels are used only after retrieval ranking for diagnostics.",
        },
    }

    report_path = out_dir / "comparison_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("-" * 80)
    print("STATUS: PASS_RETRIEVAL_DIAGNOSTICS_COMPARISON")
    print("Summary CSV:", summary_csv)
    print("Summary MD:", markdown_path)
    print("Report:", report_path)
    print("Figures:", fig_dir)


if __name__ == "__main__":
    main()
