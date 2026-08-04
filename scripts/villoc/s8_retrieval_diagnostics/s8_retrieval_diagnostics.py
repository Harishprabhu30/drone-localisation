#!/usr/bin/env python3
"""
General retrieval diagnostics for Villoc/SatLoc-style image-to-map retrieval.

Given one retrieval run, this script:
  1. Splits queries into reusable failure-analysis buckets.
  2. Writes bucket CSVs.
  3. Creates diagnostic panels:
       UAV query | Top-1 tile | Top-2 tile | Top-3 tile | First oracle tile in Top-K
  4. Writes a summary JSON.

The script is dataset-general:
  - It reads dataset paths from --config.
  - It can diagnose 90°, 45°, or future Villoc datasets.
  - It can reuse map tile indexes from another dataset if the YAML points there.
  - It does not rank using coordinates/oracles; those are used only after ranking
    for diagnostics/evaluation.

Expected retrieval outputs from s8_11d_independent_dinov2_retrieval.py:
  outputs/<dataset>/retrieval/s8_11d/s8_11d_topk_<variant>_<tag>.csv
  outputs/<dataset>/retrieval/s8_11d/s8_11d_query_eval_<variant>_<tag>.csv

source .drone_venv/bin/activate
export PYTHONPATH=$PWD/src

Hqo to run: 
1. 45deg dataset:

mkdir -p outputs/villoc/45_deg/logs/s8_12d_45deg_diagnostics

python scripts/villoc/s8_retrieval_diagnostics.py \
  --config configs/dataset_villoc_45deg.yaml \
  --variant 1024_s512 \
  --tag dinov2_vits14_img224_center_square_avgpatch_cpu \
  --oracle-k 20 \
  --max-panels 40 \
  --top-n-tiles 3 \
  2>&1 | tee outputs/villoc/45_deg/logs/s8_12d_45deg_diagnostics/s8_retrieval_diagnostics_1024_s512.log

2. 90deg dataset:

mkdir -p outputs/villoc/90_deg/logs/s8_retrieval_diagnostics

python scripts/villoc/s8_retrieval_diagnostics.py \
  --config configs/dataset_villoc_90deg.yaml \
  --variant 1024_s512 \
  --tag dinov2_vits14_img224_center_square_avgpatch_cpu \
  --oracle-k 20 \
  --max-panels 40 \
  --top-n-tiles 3 \
  2>&1 | tee outputs/villoc/90_deg/logs/s8_retrieval_diagnostics/s8_retrieval_diagnostics_1024_s512.log

3. running traj01 villoc dataset:

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
import textwrap
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import yaml
except ImportError as exc:
    raise SystemExit("Missing dependency: PyYAML. Install with: pip install pyyaml") from exc


DEFAULT_TAG = "dinov2_vits14_img224_center_square_avgpatch_cpu"


def repo_root() -> Path:
    return Path.cwd()


def root_join(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise TypeError(f"Config is not a YAML mapping: {path}")
    return data


def read_csv_required(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")
    return pd.read_csv(path)


def normalize_id(value: Any) -> str:
    if pd.isna(value):
        return ""
    s = str(value).strip()
    try:
        f = float(s)
        if f.is_integer():
            return str(int(f))
    except Exception:
        pass
    return s


def parse_bool_value(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "oracle", "inside"}


def bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.astype(bool)
    return series.map(parse_bool_value).astype(bool)


def safe_col(df: pd.DataFrame, candidates: list[str], label: str) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    raise KeyError(
        f"{label}: none of columns found: {candidates}. Available columns: {list(df.columns)}"
    )


def optional_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def metric_float(value: Any) -> float | None:
    try:
        if pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def infer_output_root(root: Path, cfg: dict[str, Any]) -> Path:
    try:
        return root_join(root, cfg["dataset"]["output_root"])
    except KeyError as exc:
        raise KeyError("Config must contain dataset.output_root") from exc


def infer_view_label(cfg: dict[str, Any]) -> str:
    dataset = cfg.get("dataset", {})
    return str(dataset.get("view_angle_group") or dataset.get("name") or "dataset")


def infer_query_manifest(root: Path, cfg: dict[str, Any], explicit: str | None) -> Path:
    if explicit:
        return root_join(root, explicit)

    out_root = infer_output_root(root, cfg)
    candidates = [
        out_root / "metadata/s8_10b_canonical_uav_query_manifest.csv",
        out_root / "metadata/s8_5_uav_frames_index_v_1fps.csv",
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        "Could not infer query manifest. Tried:\n"
        + "\n".join(str(p) for p in candidates)
        + "\nPass --query-manifest-csv explicitly."
    )


def infer_topk_csv(root: Path, cfg: dict[str, Any], variant: str, tag: str, explicit: str | None) -> Path:
    if explicit:
        return root_join(root, explicit)
    out_root = infer_output_root(root, cfg)
    return out_root / "retrieval/s8_11d" / f"s8_11d_topk_{variant}_{tag}.csv"


def infer_query_eval_csv(root: Path, cfg: dict[str, Any], variant: str, tag: str, explicit: str | None) -> Path:
    if explicit:
        return root_join(root, explicit)
    out_root = infer_output_root(root, cfg)
    return out_root / "retrieval/s8_11d" / f"s8_11d_query_eval_{variant}_{tag}.csv"


def infer_tile_index_csv(root: Path, cfg: dict[str, Any], variant: str, explicit: str | None) -> Path:
    if explicit:
        return root_join(root, explicit)

    variants = cfg.get("map", {}).get("tile_variants", None)
    if variants is None:
        raise KeyError("Config missing map.tile_variants. Pass --tile-index-csv explicitly.")

    if isinstance(variants, dict):
        if variant not in variants:
            raise KeyError(f"Variant {variant!r} not found in map.tile_variants")
        return root_join(root, variants[variant]["index_csv"])

    if isinstance(variants, list):
        for item in variants:
            if item.get("name") == variant:
                return root_join(root, item["index_csv"])

    raise KeyError(f"Could not infer tile index for variant={variant!r}. Pass --tile-index-csv.")


def infer_out_root(root: Path, cfg: dict[str, Any], variant: str, explicit: str | None) -> Path:
    if explicit:
        base = root_join(root, explicit)
    else:
        base = infer_output_root(root, cfg) / "diagnostics/s8_retrieval_diagnostics"
    return base / variant


def resolve_existing_path(root: Path, value: Any) -> Path | None:
    if pd.isna(value):
        return None
    s = str(value).strip()
    if not s:
        return None

    path = Path(s)
    if path.exists():
        return path

    candidate = root / path
    if candidate.exists():
        return candidate

    return None


def blank_image(label: str, size: tuple[int, int] = (512, 512)) -> Image.Image:
    img = Image.new("RGB", size, color=(35, 35, 35))
    draw = ImageDraw.Draw(img)
    msg = f"Missing image\n{label}"
    draw.multiline_text((20, 20), msg, fill=(230, 230, 230), spacing=6)
    return img


def open_rgb(path: Path | None, label: str) -> Image.Image:
    if path is None or not path.exists():
        return blank_image(label)
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return blank_image(f"Unreadable\n{path}")


def prepare_manifest(root: Path, manifest_csv: Path) -> pd.DataFrame:
    df = read_csv_required(manifest_csv, "query manifest")
    if "query_id" not in df.columns:
        if "token0_id" in df.columns:
            df["query_id"] = df["token0_id"]
        else:
            raise KeyError(f"{manifest_csv}: missing query_id and token0_id")

    df["query_id"] = df["query_id"].map(normalize_id)

    path_col = safe_col(df, ["image_path", "image_path_relative", "frame_path"], "query manifest")
    df["query_image_resolved"] = df[path_col].map(lambda v: str(resolve_existing_path(root, v) or ""))

    keep = [
        "query_id",
        path_col,
        "query_image_resolved",
        "token0_id",
        "sample_id",
        "source_frame_cnt",
        "zero_based_frame_index",
        "timestamp_s",
        "video_time_s",
        "alignment_error_ms",
        "rel_alt_m",
        "abs_alt_m",
        "gb_yaw_deg",
        "gb_pitch_deg",
        "gb_roll_deg",
        "lat",
        "lon",
        "latitude",
        "longitude",
    ]
    keep = [c for c in keep if c in df.columns]
    out = df[keep].copy()
    if path_col != "query_image_path":
        out = out.rename(columns={path_col: "query_image_path"})
    return out


def prepare_tile_index(root: Path, tile_index_csv: Path) -> pd.DataFrame:
    df = read_csv_required(tile_index_csv, "tile index")
    if "tile_id" not in df.columns:
        raise KeyError(f"{tile_index_csv}: missing tile_id")
    df["tile_id"] = df["tile_id"].map(normalize_id)

    path_col = safe_col(df, ["tile_path", "image_path", "path"], "tile index")
    df["tile_image_resolved"] = df[path_col].map(lambda v: str(resolve_existing_path(root, v) or ""))

    keep = [
        "tile_id",
        path_col,
        "tile_image_resolved",
        "tile_number",
        "filename",
        "grid_row",
        "grid_col",
        "center_easting",
        "center_northing",
        "xmin",
        "ymin",
        "xmax",
        "ymax",
    ]
    keep = [c for c in keep if c in df.columns]
    out = df[keep].copy()
    if path_col != "tile_path":
        out = out.rename(columns={path_col: "tile_path"})
    return out


def prepare_query_eval(query_eval_csv: Path) -> pd.DataFrame:
    df = read_csv_required(query_eval_csv, "query eval")
    if "query_id" not in df.columns:
        raise KeyError(f"{query_eval_csv}: missing query_id")

    df["query_id"] = df["query_id"].map(normalize_id)

    if "top1_tile_id" in df.columns:
        df["top1_tile_id"] = df["top1_tile_id"].map(normalize_id)

    if "top1_is_oracle" in df.columns:
        df["top1_is_oracle"] = bool_series(df["top1_is_oracle"])
    else:
        raise KeyError(f"{query_eval_csv}: missing top1_is_oracle")

    for col in list(df.columns):
        if col.startswith("recall_at_"):
            df[col] = bool_series(df[col])

    if "first_oracle_rank" in df.columns:
        df["first_oracle_rank"] = pd.to_numeric(df["first_oracle_rank"], errors="coerce")

    if "top1_center_error_m" in df.columns:
        df["top1_center_error_m"] = pd.to_numeric(df["top1_center_error_m"], errors="coerce")

    return df


def prepare_topk(topk_csv: Path) -> pd.DataFrame:
    df = read_csv_required(topk_csv, "top-k candidates")
    for required in ["query_id", "tile_id"]:
        if required not in df.columns:
            raise KeyError(f"{topk_csv}: missing {required}")

    df["query_id"] = df["query_id"].map(normalize_id)
    df["tile_id"] = df["tile_id"].map(normalize_id)

    if "is_oracle" in df.columns:
        df["is_oracle"] = bool_series(df["is_oracle"])
    else:
        df["is_oracle"] = False

    rank_col = safe_col(df, ["rank", "retrieval_rank"], "top-k candidates")
    df[rank_col] = pd.to_numeric(df[rank_col], errors="coerce")
    df = df.sort_values(["query_id", rank_col]).copy()

    for col in ["score", "similarity", "cosine_score", "center_error_m"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def first_oracle_candidate(topk_for_query: pd.DataFrame) -> pd.Series | None:
    if topk_for_query.empty or "is_oracle" not in topk_for_query.columns:
        return None
    sub = topk_for_query[topk_for_query["is_oracle"]].copy()
    if sub.empty:
        return None
    rank_col = safe_col(sub, ["rank", "retrieval_rank"], "top-k query subset")
    return sub.sort_values(rank_col).iloc[0]


def add_bucket_flags(df: pd.DataFrame, oracle_k: int, high_conf_quantile: float) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = df.copy()

    out["good_top1"] = out["top1_is_oracle"]
    out["bad_top1"] = ~out["top1_is_oracle"]

    if "first_oracle_rank" in out.columns:
        rank = out["first_oracle_rank"]
        out["oracle_in_top5_but_not_top1"] = (~out["top1_is_oracle"]) & rank.notna() & (rank <= 5)
        out[f"oracle_in_top{oracle_k}_but_not_top1"] = (~out["top1_is_oracle"]) & rank.notna() & (rank <= oracle_k)
        out[f"oracle_missing_top{oracle_k}"] = rank.isna() | (rank > oracle_k)
    else:
        out["oracle_in_top5_but_not_top1"] = False
        out[f"oracle_in_top{oracle_k}_but_not_top1"] = False
        out[f"oracle_missing_top{oracle_k}"] = False

    score_col = optional_col(out, ["top1_score", "score", "similarity", "cosine_score"])
    high_conf_threshold = None
    if score_col is not None:
        scores = pd.to_numeric(out[score_col], errors="coerce")
        valid = scores.dropna()
        if len(valid):
            high_conf_threshold = float(valid.quantile(high_conf_quantile))
            out["high_conf_wrong_top1"] = (~out["top1_is_oracle"]) & (scores >= high_conf_threshold)
        else:
            out["high_conf_wrong_top1"] = False
    else:
        out["high_conf_wrong_top1"] = False

    if "top1_center_error_m" in out.columns:
        err = out["top1_center_error_m"]
        out["easy_correct"] = out["top1_is_oracle"] & err.notna() & (err <= 40)
        out["hard_correct"] = out["top1_is_oracle"] & (~out["easy_correct"])
    else:
        out["easy_correct"] = False
        out["hard_correct"] = out["top1_is_oracle"]

    meta = {
        "oracle_k": oracle_k,
        "high_conf_quantile": high_conf_quantile,
        "score_column_for_high_conf": score_col,
        "high_conf_threshold": high_conf_threshold,
    }
    return out, meta


def write_bucket_csvs(df: pd.DataFrame, csv_dir: Path, oracle_k: int) -> dict[str, dict[str, Any]]:
    csv_dir.mkdir(parents=True, exist_ok=True)

    buckets = [
        "good_top1",
        "bad_top1",
        "oracle_in_top5_but_not_top1",
        f"oracle_in_top{oracle_k}_but_not_top1",
        f"oracle_missing_top{oracle_k}",
        "high_conf_wrong_top1",
        "easy_correct",
        "hard_correct",
    ]

    outputs: dict[str, dict[str, Any]] = {}
    for bucket in buckets:
        if bucket not in df.columns:
            continue
        sub = df[df[bucket]].copy()
        path = csv_dir / f"{bucket}.csv"
        sub.to_csv(path, index=False)
        outputs[bucket] = {"path": str(path), "rows": int(len(sub))}
        print(f"[CSV] {bucket:<35} rows={len(sub):>4} path={path}")

    all_path = csv_dir / "all_queries_with_diagnostic_flags.csv"
    df.to_csv(all_path, index=False)
    outputs["all_queries_with_diagnostic_flags"] = {"path": str(all_path), "rows": int(len(df))}
    print(f"[CSV] {'all_queries_with_diagnostic_flags':<35} rows={len(df):>4} path={all_path}")
    return outputs


def choose_examples(df: pd.DataFrame, bucket: str, max_panels: int) -> pd.DataFrame:
    sub = df[df[bucket]].copy()
    if sub.empty:
        return sub

    if bucket in {"good_top1", "easy_correct"} and "top1_center_error_m" in sub.columns:
        sub = sub.sort_values("top1_center_error_m", ascending=True)
    elif bucket in {"bad_top1", "high_conf_wrong_top1"}:
        sort_cols = [c for c in ["first_oracle_rank", "top1_center_error_m"] if c in sub.columns]
        if sort_cols:
            sub = sub.sort_values(sort_cols, ascending=[True] * len(sort_cols))
    elif bucket.startswith("oracle_missing") and "top1_center_error_m" in sub.columns:
        sub = sub.sort_values("top1_center_error_m", ascending=False)
    elif "first_oracle_rank" in sub.columns:
        sub = sub.sort_values("first_oracle_rank", ascending=True)

    return sub.head(max_panels)


def panel_label_for_candidate(row: pd.Series | None, rank_name: str, score_col: str | None) -> tuple[str, Path | None]:
    if row is None:
        return f"{rank_name}\nmissing", None

    tile_id = normalize_id(row.get("tile_id", ""))
    bits = [rank_name, f"tile={tile_id}"]

    if score_col and score_col in row.index:
        value = metric_float(row.get(score_col))
        if value is not None:
            bits.append(f"score={value:.4f}")

    if "center_error_m" in row.index:
        err = metric_float(row.get("center_error_m"))
        if err is not None:
            bits.append(f"err={err:.1f} m")

    if "is_oracle" in row.index:
        bits.append(f"oracle={parse_bool_value(row.get('is_oracle'))}")

    tile_path = None
    if "tile_image_resolved" in row.index:
        raw = str(row.get("tile_image_resolved") or "")
        tile_path = Path(raw) if raw else None

    return "\n".join(bits), tile_path


def make_panel(
    *,
    out_path: Path,
    query_path: Path | None,
    top_rows: list[pd.Series | None],
    oracle_row: pd.Series | None,
    title: str,
    subtitle: str,
    score_col: str | None,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    q_img = open_rgb(query_path, "query")
    images: list[tuple[str, Image.Image]] = [("UAV query", q_img)]

    for idx, row in enumerate(top_rows, start=1):
        label, tile_path = panel_label_for_candidate(row, f"Top-{idx}", score_col)
        images.append((label, open_rgb(tile_path, label)))

    oracle_label, oracle_path = panel_label_for_candidate(oracle_row, "First oracle", score_col)
    images.append((oracle_label, open_rgb(oracle_path, oracle_label)))

    n = len(images)
    fig_w = max(14, 3.2 * n)
    fig, axes = plt.subplots(1, n, figsize=(fig_w, 4.8))
    if n == 1:
        axes = [axes]

    for ax, (label, img) in zip(axes, images):
        ax.imshow(img)
        ax.set_title(label, fontsize=9)
        ax.axis("off")

    fig.suptitle(
        title + "\n" + "\n".join(textwrap.wrap(subtitle, width=135)),
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.86])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def make_diagnostic_panels(
    *,
    df: pd.DataFrame,
    topk: pd.DataFrame,
    figures_dir: Path,
    buckets: list[str],
    max_panels: int,
    top_n_tiles: int,
) -> dict[str, int]:
    figures_dir.mkdir(parents=True, exist_ok=True)

    rank_col = safe_col(topk, ["rank", "retrieval_rank"], "top-k candidates")
    score_col = optional_col(topk, ["score", "similarity", "cosine_score"])

    counts: dict[str, int] = {}

    for bucket in buckets:
        if bucket not in df.columns:
            continue

        examples = choose_examples(df, bucket, max_panels)
        counts[bucket] = int(len(examples))
        bucket_dir = figures_dir / bucket
        bucket_dir.mkdir(parents=True, exist_ok=True)

        for _, row in examples.iterrows():
            qid = normalize_id(row["query_id"])
            q_topk = topk[topk["query_id"] == qid].sort_values(rank_col).copy()

            top_rows: list[pd.Series | None] = []
            for _, cand in q_topk.head(top_n_tiles).iterrows():
                top_rows.append(cand)
            while len(top_rows) < top_n_tiles:
                top_rows.append(None)

            oracle_row = first_oracle_candidate(q_topk)

            query_path_raw = str(row.get("query_image_resolved", "") or "")
            query_path = Path(query_path_raw) if query_path_raw else None

            rel_alt = metric_float(row.get("rel_alt_m"))
            yaw = metric_float(row.get("gb_yaw_deg"))
            pitch = metric_float(row.get("gb_pitch_deg"))
            top1_err = metric_float(row.get("top1_center_error_m"))
            first_oracle_rank = metric_float(row.get("first_oracle_rank"))

            subtitle_parts = [
                f"query_id={qid}",
                f"bucket={bucket}",
                f"top1_is_oracle={bool(row.get('top1_is_oracle'))}",
            ]
            if top1_err is not None:
                subtitle_parts.append(f"top1_error={top1_err:.2f} m")
            if first_oracle_rank is not None:
                subtitle_parts.append(f"first_oracle_rank={first_oracle_rank:.0f}")
            if rel_alt is not None:
                subtitle_parts.append(f"rel_alt={rel_alt:.2f} m")
            if yaw is not None:
                subtitle_parts.append(f"yaw={yaw:.2f}°")
            if pitch is not None:
                subtitle_parts.append(f"pitch={pitch:.2f}°")

            out_path = bucket_dir / f"{bucket}_query_{qid}.png"
            make_panel(
                out_path=out_path,
                query_path=query_path,
                top_rows=top_rows,
                oracle_row=oracle_row,
                title="General retrieval diagnostic panel",
                subtitle=" | ".join(subtitle_parts),
                score_col=score_col,
            )

        print(f"[FIG] {bucket:<35} panels={len(examples):>4} dir={bucket_dir}")

    return counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="General retrieval diagnostics: split good/bad tokens and create visual panels."
    )
    parser.add_argument("--config", required=True, help="Dataset YAML, e.g. configs/dataset_villoc_45deg.yaml")
    parser.add_argument("--variant", required=True, help="Tile variant, e.g. 1024_s512")
    parser.add_argument("--tag", default=DEFAULT_TAG, help="Descriptor/retrieval tag")
    parser.add_argument("--query-eval-csv", default=None, help="Optional explicit query-eval CSV")
    parser.add_argument("--topk-csv", default=None, help="Optional explicit top-k CSV")
    parser.add_argument("--query-manifest-csv", default=None, help="Optional explicit query manifest CSV")
    parser.add_argument("--tile-index-csv", default=None, help="Optional explicit tile index CSV")
    parser.add_argument("--out-root", default=None, help="Optional output root. Variant subfolder is appended.")
    parser.add_argument("--oracle-k", type=int, default=20, help="K used for oracle-in-topK and oracle-missing-topK buckets")
    parser.add_argument("--max-panels", type=int, default=40, help="Max panels per bucket")
    parser.add_argument("--top-n-tiles", type=int, default=3, help="How many ranked candidate tiles to show beside the query")
    parser.add_argument("--high-conf-quantile", type=float, default=0.90, help="Quantile threshold for high_conf_wrong_top1")
    args = parser.parse_args()

    root = repo_root()
    cfg_path = root_join(root, args.config)
    cfg = load_yaml(cfg_path)

    view_label = infer_view_label(cfg)
    query_eval_csv = infer_query_eval_csv(root, cfg, args.variant, args.tag, args.query_eval_csv)
    topk_csv = infer_topk_csv(root, cfg, args.variant, args.tag, args.topk_csv)
    query_manifest_csv = infer_query_manifest(root, cfg, args.query_manifest_csv)
    tile_index_csv = infer_tile_index_csv(root, cfg, args.variant, args.tile_index_csv)
    out_root = infer_out_root(root, cfg, args.variant, args.out_root)

    csv_dir = out_root / "csv"
    figures_dir = out_root / "figures"
    out_root.mkdir(parents=True, exist_ok=True)

    print("General retrieval diagnostics")
    print("-" * 80)
    print("config:", cfg_path)
    print("view_label:", view_label)
    print("variant:", args.variant)
    print("tag:", args.tag)
    print("query_eval_csv:", query_eval_csv)
    print("topk_csv:", topk_csv)
    print("query_manifest_csv:", query_manifest_csv)
    print("tile_index_csv:", tile_index_csv)
    print("out_root:", out_root)

    query_eval = prepare_query_eval(query_eval_csv)
    topk = prepare_topk(topk_csv)
    manifest = prepare_manifest(root, query_manifest_csv)
    tile_index = prepare_tile_index(root, tile_index_csv)

    enriched = query_eval.merge(manifest, on="query_id", how="left", suffixes=("", "_manifest"))
    topk_enriched = topk.merge(tile_index, on="tile_id", how="left", suffixes=("", "_tile"))

    rank_col = safe_col(topk_enriched, ["rank", "retrieval_rank"], "top-k candidates")
    score_col = optional_col(topk_enriched, ["score", "similarity", "cosine_score"])
    if score_col and "top1_score" not in enriched.columns:
        top1_scores = (
            topk_enriched[topk_enriched[rank_col] == 1]
            [["query_id", score_col]]
            .rename(columns={score_col: "top1_score"})
        )
        enriched = enriched.merge(top1_scores, on="query_id", how="left")

    enriched, bucket_meta = add_bucket_flags(
        enriched,
        oracle_k=args.oracle_k,
        high_conf_quantile=args.high_conf_quantile,
    )

    bucket_outputs = write_bucket_csvs(enriched, csv_dir, args.oracle_k)

    panel_buckets = [
        "good_top1",
        "bad_top1",
        "oracle_in_top5_but_not_top1",
        f"oracle_in_top{args.oracle_k}_but_not_top1",
        f"oracle_missing_top{args.oracle_k}",
        "high_conf_wrong_top1",
        "easy_correct",
        "hard_correct",
    ]

    panel_counts = make_diagnostic_panels(
        df=enriched,
        topk=topk_enriched,
        figures_dir=figures_dir,
        buckets=panel_buckets,
        max_panels=args.max_panels,
        top_n_tiles=args.top_n_tiles,
    )

    summary = {
        "status": "PASS_RETRIEVAL_DIAGNOSTICS",
        "script": "scripts/villoc/s8_retrieval_diagnostics.py",
        "config": str(cfg_path),
        "view_label": view_label,
        "variant": args.variant,
        "tag": args.tag,
        "inputs": {
            "query_eval_csv": str(query_eval_csv),
            "topk_csv": str(topk_csv),
            "query_manifest_csv": str(query_manifest_csv),
            "tile_index_csv": str(tile_index_csv),
        },
        "outputs": {
            "out_root": str(out_root),
            "csv_dir": str(csv_dir),
            "figures_dir": str(figures_dir),
            "bucket_csvs": bucket_outputs,
            "panel_counts": panel_counts,
        },
        "counts": {
            "query_count": int(len(enriched)),
            "topk_rows": int(len(topk_enriched)),
            "top1_correct": int(enriched["good_top1"].sum()),
            "top1_wrong": int(enriched["bad_top1"].sum()),
            f"oracle_in_top{args.oracle_k}_but_not_top1": int(enriched[f"oracle_in_top{args.oracle_k}_but_not_top1"].sum()),
            f"oracle_missing_top{args.oracle_k}": int(enriched[f"oracle_missing_top{args.oracle_k}"].sum()),
            "high_conf_wrong_top1": int(enriched["high_conf_wrong_top1"].sum()),
        },
        "bucket_meta": bucket_meta,
        "rules": {
            "descriptor_ranking_used_coordinates": False,
            "descriptor_ranking_used_oracle": False,
            "coordinates_and_oracles_used_only_after_ranking_for_diagnostics": True,
        },
    }

    report_path = out_root / "report.json"
    report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("-" * 80)
    print("STATUS: PASS_RETRIEVAL_DIAGNOSTICS")
    print("Report:", report_path)


if __name__ == "__main__":
    main()
