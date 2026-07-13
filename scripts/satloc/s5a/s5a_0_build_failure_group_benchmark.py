#!/usr/bin/env python3
"""
S5A.0 — Build SatLoc failure-group benchmark manifest.

Purpose
-------
Convert the S4C.6C failure-group outputs into a clean S5A benchmark manifest.
This block does not run any new matcher. It only prepares the controlled split
that will be used by S5A local/learned verifiers inside PHOG top-K candidates.

Locked evaluation rule
----------------------
UAV filename lon/lat and reference coordinates are evaluation-only. This script
may read previous evaluation outputs, but it does not create any retrieval score
or use reference coordinates for ranking.

Code Used:
-------
export PYTHONPATH=$PWD/src
python scripts/satloc/s5a/s5a_0_build_failure_group_benchmark.py \
  --sequence traj01 \
  --s4c6-dir outputs/satloc/metadata/s4c_macrocontour_phog_chamfer/s4c6_failure_analysis \
  --out-dir outputs/satloc
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import pandas as pd


GROUP_RECOMMENDED_USE = {
    "selection_failure_correct_in_pool": "primary_verifier_target",
    "lsd_destroyed_phog_success": "phog_protection_test",
    "lsd_rescue": "preserve_lsd_rescue",
    "stable_success": "sanity_preserve_success",
    "candidate_pool_failure": "candidate_generation_limit",
    "weak_pool_near_candidate": "candidate_generation_boundary",
}

GROUP_PRIORITY = {
    "selection_failure_correct_in_pool": 1,
    "lsd_destroyed_phog_success": 2,
    "lsd_rescue": 3,
    "stable_success": 4,
    "weak_pool_near_candidate": 5,
    "candidate_pool_failure": 6,
}

TOKEN_COLUMN_CANDIDATES = [
    "token",
    "query_token",
    "uav_token",
    "frame_token",
    "token0_id",
    "uav_token0_id",
    "frame_id",
    "imgid",
]

GROUP_COLUMN_CANDIDATES = [
    "failure_group",
    "group",
    "failure_type",
    "class",
    "category",
]

PHOG_ERROR_CANDIDATES = [
    "phog_top1_error_m",
    "phog_error_m",
    "phog_top1_error",
    "phog_top1_center_error_m",
    "top1_phog_error_m",
    "phog_err_m",
]

LSD_ERROR_CANDIDATES = [
    "lsd_top1_error_m",
    "lsd_error_m",
    "lsd_top1_error",
    "lsd_top1_center_error_m",
    "top1_lsd_error_m",
    "lsd_err_m",
]

ORACLE_ERROR_CANDIDATES = [
    "oracle_top50_error_m",
    "oracle_best_top50_error_m",
    "best_top50_error_m",
    "phog_oracle_top50_error_m",
    "oracle_error_m",
    "best_candidate_error_m",
]

PATH_COLUMN_CANDIDATES = [
    "uav_image_path",
    "image_path",
    "frame_path",
    "path",
    "file_path",
    "filepath",
    "filename",
    "file_name",
    "uav_path",
]


# ----------------------------- utilities -----------------------------


def normalize_column_name(name: Any) -> str:
    text = str(name).strip().lower()
    text = text.replace("<=", "le")
    text = text.replace("@", "_")
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text


def read_csv_safe(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        # Some diagnostic CSVs can contain odd quoting or comments. Python engine is slower
        # but more forgiving, and this script is run only once per stage.
        return pd.read_csv(path, engine="python")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [normalize_column_name(c) for c in out.columns]
    return out


def find_column(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    cols = list(df.columns)
    for cand in candidates:
        norm = normalize_column_name(cand)
        if norm in cols:
            return norm

    # Flexible fallback: prefer exact token substrings but avoid longitude/latitude.
    candidate_norms = [normalize_column_name(c) for c in candidates]
    for col in cols:
        if any(cand in col for cand in candidate_norms):
            if "lat" in col or "lon" in col:
                continue
            return col
    return None


def to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    # Extract first float-like value if text contains units.
    match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def to_int_token(value: Any) -> Optional[int]:
    number = to_float(value)
    if number is None:
        return None
    return int(round(number))


def extract_tokens_from_text(value: Any) -> List[int]:
    if value is None:
        return []
    text = str(value)
    tokens = []
    for match in re.findall(r"\d+", text):
        try:
            val = int(match)
        except ValueError:
            continue
        # SatLoc traj01 tokens are positive frame ids. Avoid collecting very large
        # numbers from timestamps/paths if this fallback is used.
        if 1 <= val <= 100000:
            tokens.append(val)
    return tokens


def canonical_group_name(value: Any) -> str:
    text = str(value).strip()
    text = normalize_column_name(text)
    # Common variations.
    aliases = {
        "candidate_pool": "candidate_pool_failure",
        "pool_failure": "candidate_pool_failure",
        "weak_pool": "weak_pool_near_candidate",
        "selection_failure": "selection_failure_correct_in_pool",
        "correct_in_pool": "selection_failure_correct_in_pool",
        "lsd_destroyed": "lsd_destroyed_phog_success",
        "phog_destroyed": "lsd_destroyed_phog_success",
        "stable": "stable_success",
        "success": "stable_success",
        "rescue": "lsd_rescue",
    }
    return aliases.get(text, text)


def group_priority(group: str) -> int:
    return GROUP_PRIORITY.get(group, 99)


# ----------------------------- input discovery -----------------------------


def explode_table_to_token_rows(path: Path, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Return a normalized token-level table if possible.

    The S4C.6C exact per-token output filename may change. This function accepts
    either an explicit token column or a representative_tokens-style column that
    stores multiple tokens in one row.
    """
    df = normalize_columns(df)
    token_col = find_column(df, TOKEN_COLUMN_CANDIDATES)
    group_col = find_column(df, GROUP_COLUMN_CANDIDATES)

    phog_col = find_column(df, PHOG_ERROR_CANDIDATES)
    lsd_col = find_column(df, LSD_ERROR_CANDIDATES)
    oracle_col = find_column(df, ORACLE_ERROR_CANDIDATES)

    rows: List[Dict[str, Any]] = []
    mode = "none"

    if group_col is None:
        return pd.DataFrame(), {
            "source": str(path),
            "usable": False,
            "reason": "no failure-group column found",
        }

    if token_col is not None:
        mode = "explicit_token_column"
        for _, row in df.iterrows():
            token = to_int_token(row.get(token_col))
            if token is None:
                continue
            rows.append(
                {
                    "token": token,
                    "failure_group": canonical_group_name(row.get(group_col)),
                    "phog_top1_error_m": to_float(row.get(phog_col)) if phog_col else None,
                    "lsd_top1_error_m": to_float(row.get(lsd_col)) if lsd_col else None,
                    "oracle_top50_error_m": to_float(row.get(oracle_col)) if oracle_col else None,
                    "source_table": str(path),
                    "source_mode": mode,
                }
            )
    else:
        # Representative tables often have a column like "tokens" or one text
        # field containing "387, 573, 577, 366". Only use likely token columns.
        tokenish_cols = [c for c in df.columns if "token" in c]
        if not tokenish_cols:
            return pd.DataFrame(), {
                "source": str(path),
                "usable": False,
                "reason": "no token or tokens column found",
            }
        mode = "expanded_token_list"
        for _, row in df.iterrows():
            group = canonical_group_name(row.get(group_col))
            found_tokens: List[int] = []
            for c in tokenish_cols:
                found_tokens.extend(extract_tokens_from_text(row.get(c)))
            for token in sorted(set(found_tokens)):
                rows.append(
                    {
                        "token": token,
                        "failure_group": group,
                        "phog_top1_error_m": to_float(row.get(phog_col)) if phog_col else None,
                        "lsd_top1_error_m": to_float(row.get(lsd_col)) if lsd_col else None,
                        "oracle_top50_error_m": to_float(row.get(oracle_col)) if oracle_col else None,
                        "source_table": str(path),
                        "source_mode": mode,
                    }
                )

    out = pd.DataFrame(rows)
    if out.empty:
        return out, {
            "source": str(path),
            "usable": False,
            "reason": "no valid token rows extracted",
        }

    out = out.drop_duplicates(subset=["token", "failure_group"]).reset_index(drop=True)
    info = {
        "source": str(path),
        "usable": True,
        "rows_extracted": int(len(out)),
        "unique_tokens": int(out["token"].nunique()),
        "mode": mode,
        "token_column": token_col,
        "group_column": group_col,
        "phog_error_column": phog_col,
        "lsd_error_column": lsd_col,
        "oracle_error_column": oracle_col,
    }
    return out, info


def discover_failure_token_table(s4c6_dir: Path) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    csv_paths = sorted(s4c6_dir.glob("*.csv"))
    infos: List[Dict[str, Any]] = []
    candidates: List[pd.DataFrame] = []

    for path in csv_paths:
        try:
            df = read_csv_safe(path)
        except Exception as exc:
            infos.append({"source": str(path), "usable": False, "reason": f"read failed: {exc}"})
            continue
        extracted, info = explode_table_to_token_rows(path, df)
        infos.append(info)
        if not extracted.empty:
            candidates.append(extracted)

    if not candidates:
        return pd.DataFrame(columns=["token", "failure_group"]), infos

    # Prefer the table with most unique tokens. Representative-token CSVs are a valid
    # fallback, but if a 73-row assignment table exists, it will win automatically.
    candidates = sorted(candidates, key=lambda x: (x["token"].nunique(), len(x)), reverse=True)
    chosen = candidates[0].copy()

    # If multiple usable tables include missing columns, merge extra error columns from them.
    for extra in candidates[1:]:
        if extra["token"].nunique() < chosen["token"].nunique():
            continue
        chosen = chosen.merge(
            extra.drop(columns=["failure_group"], errors="ignore"),
            on="token",
            how="left",
            suffixes=("", "_extra"),
        )
        for base in ["phog_top1_error_m", "lsd_top1_error_m", "oracle_top50_error_m"]:
            extra_col = f"{base}_extra"
            if extra_col in chosen.columns:
                chosen[base] = chosen[base].combine_first(chosen[extra_col])
                chosen = chosen.drop(columns=[extra_col])

    # One token should belong to one failure group. If duplicated, keep the row with
    # the most complete error information.
    def completeness(row: pd.Series) -> int:
        return sum(pd.notna(row.get(c)) for c in ["phog_top1_error_m", "lsd_top1_error_m", "oracle_top50_error_m"])

    chosen["_complete"] = chosen.apply(completeness, axis=1)
    chosen = chosen.sort_values(["token", "_complete"], ascending=[True, False])
    chosen = chosen.drop_duplicates(subset=["token"], keep="first").drop(columns=["_complete"])
    chosen = chosen.sort_values(["failure_group", "token"], key=lambda s: s.map(group_priority) if s.name == "failure_group" else s)
    return chosen.reset_index(drop=True), infos


def read_summary_json(s4c6_dir: Path) -> Tuple[Optional[Dict[str, Any]], Optional[Path]]:
    json_paths = sorted(s4c6_dir.glob("*.json"))
    # The prompt places JSON under reports/, not metadata/. Try sibling reports path.
    possible = list(json_paths)
    s = str(s4c6_dir)
    if "/metadata/" in s:
        reports_dir = Path(s.replace("/metadata/", "/reports/"))
        possible.extend(sorted(reports_dir.glob("*.json")))
    for path in possible:
        try:
            with path.open("r", encoding="utf-8") as f:
                return json.load(f), path
        except Exception:
            continue
    return None, None


def read_group_summary(s4c6_dir: Path) -> Tuple[Optional[pd.DataFrame], Optional[Path]]:
    possible_names = [
        "s4c6c_failure_group_summary.csv",
        "failure_group_summary.csv",
    ]
    for name in possible_names:
        path = s4c6_dir / name
        if path.exists():
            return normalize_columns(read_csv_safe(path)), path
    for path in sorted(s4c6_dir.glob("*summary*.csv")):
        try:
            df = normalize_columns(read_csv_safe(path))
            if find_column(df, GROUP_COLUMN_CANDIDATES) is not None:
                return df, path
        except Exception:
            continue
    return None, None


# ----------------------------- path enrichment -----------------------------


def discover_uav_index(out_dir: Path, sequence: str, explicit: Optional[Path]) -> Optional[Path]:
    if explicit and explicit.exists():
        return explicit
    candidates = [
        out_dir / "metadata" / "uav_frames_index_enriched.csv",
        out_dir / "metadata" / "uav_frames_index.csv",
        out_dir / "metadata" / f"{sequence}_uav_frames_index_enriched.csv",
        out_dir / "metadata" / f"{sequence}_uav_frames_index.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def build_uav_path_map(index_path: Optional[Path], sequence: str) -> Tuple[Dict[int, str], Dict[str, Any]]:
    if index_path is None:
        return {}, {"index_path": None, "status": "not_found"}
    try:
        df = normalize_columns(read_csv_safe(index_path))
    except Exception as exc:
        return {}, {"index_path": str(index_path), "status": "read_failed", "error": str(exc)}

    # Restrict sequence if a sequence column exists.
    seq_col = find_column(df, ["sequence", "traj", "trajectory"])
    if seq_col is not None:
        df = df[df[seq_col].astype(str).str.lower() == sequence.lower()].copy()

    token_col = find_column(df, TOKEN_COLUMN_CANDIDATES)
    path_col = find_column(df, PATH_COLUMN_CANDIDATES)
    if token_col is None or path_col is None:
        return {}, {
            "index_path": str(index_path),
            "status": "missing_columns",
            "token_column": token_col,
            "path_column": path_col,
            "columns": list(df.columns),
        }

    mapping: Dict[int, str] = {}
    for _, row in df.iterrows():
        token = to_int_token(row.get(token_col))
        path = row.get(path_col)
        if token is None or pd.isna(path):
            continue
        mapping[token] = str(path)

    return mapping, {
        "index_path": str(index_path),
        "status": "ok",
        "rows": int(len(df)),
        "mapped_tokens": int(len(mapping)),
        "token_column": token_col,
        "path_column": path_col,
    }


def index_ranked_csvs(ranked_root: Path) -> List[Path]:
    if not ranked_root.exists():
        return []
    files = []
    for p in ranked_root.rglob("*.csv"):
        name = p.name.lower()
        if any(key in name for key in ["rank", "retrieval", "candidate", "top", "phog", "rerank"]):
            files.append(p)
    return sorted(files)


def token_regex(token: int) -> re.Pattern[str]:
    # Match token1, token0001, token_0001, token-0001 without matching token10.
    return re.compile(rf"token[_-]?0*{token}(?!\d)", re.IGNORECASE)


def find_ranked_csv_for_token(token: int, ranked_csvs: Sequence[Path]) -> str:
    pattern = token_regex(token)
    matches = [p for p in ranked_csvs if pattern.search(p.name) or pattern.search(str(p.parent.name))]
    if not matches:
        return ""
    # Prefer PHOG ranked candidate outputs over analysis summaries.
    def score_path(path: Path) -> Tuple[int, int]:
        text = str(path).lower()
        score = 0
        if "phog" in text:
            score += 5
        if "top50" in text or "top_50" in text:
            score += 4
        if "rank" in text:
            score += 3
        if "candidate" in text:
            score += 2
        if "summary" in text:
            score -= 3
        return (score, -len(str(path)))

    matches = sorted(matches, key=score_path, reverse=True)
    return str(matches[0])


# ----------------------------- output generation -----------------------------


def build_manifest(
    token_table: pd.DataFrame,
    threshold_m: float,
    uav_path_map: Dict[int, str],
    ranked_csvs: Sequence[Path],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for _, row in token_table.iterrows():
        token = int(row["token"])
        group = canonical_group_name(row["failure_group"])
        phog_err = to_float(row.get("phog_top1_error_m"))
        lsd_err = to_float(row.get("lsd_top1_error_m"))
        oracle_err = to_float(row.get("oracle_top50_error_m"))

        candidate_pool_has_40m = bool(oracle_err is not None and oracle_err <= threshold_m)
        selection_failure_flag = bool(
            group == "selection_failure_correct_in_pool"
            or (candidate_pool_has_40m and (lsd_err is not None and lsd_err > threshold_m))
        )

        rows.append(
            {
                "token": token,
                "failure_group": group,
                "uav_image_path": uav_path_map.get(token, ""),
                "phog_ranked_csv_path": find_ranked_csv_for_token(token, ranked_csvs),
                "phog_top1_error_m": phog_err,
                "lsd_top1_error_m": lsd_err,
                "oracle_top50_error_m": oracle_err,
                "candidate_pool_has_40m": candidate_pool_has_40m,
                "selection_failure_flag": selection_failure_flag,
                "recommended_use": GROUP_RECOMMENDED_USE.get(group, "manual_review"),
                "source_table": row.get("source_table", ""),
                "source_mode": row.get("source_mode", ""),
            }
        )

    manifest = pd.DataFrame(rows)
    if manifest.empty:
        return manifest
    manifest["group_priority"] = manifest["failure_group"].map(lambda g: group_priority(str(g)))
    manifest = manifest.sort_values(["group_priority", "token"]).drop(columns=["group_priority"])
    return manifest.reset_index(drop=True)


def plot_group_counts(manifest: pd.DataFrame, fig_path: Path) -> None:
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    counts = manifest["failure_group"].value_counts().sort_index()
    if counts.empty:
        return

    fig, ax = plt.subplots(figsize=(11, 5.5))
    positions = range(len(counts))
    bars = ax.bar(positions, counts.values)
    ax.set_title("S5A.0 benchmark tokens by S4C.6C failure group")
    ax.set_ylabel("Token count")
    ax.set_xlabel("Failure group")
    ax.set_xticks(list(positions))
    ax.set_xticklabels(counts.index.astype(str), rotation=35, ha="right")
    for bar, value in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(int(value)), ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=180)
    plt.close(fig)


def make_report(
    manifest: pd.DataFrame,
    args: argparse.Namespace,
    discovery_infos: List[Dict[str, Any]],
    group_summary_path: Optional[Path],
    summary_json_path: Optional[Path],
    uav_index_info: Dict[str, Any],
    ranked_csv_count: int,
    manifest_path: Path,
    figure_path: Path,
) -> Dict[str, Any]:
    counts = manifest["failure_group"].value_counts().to_dict() if not manifest.empty else {}
    missing_uav_paths = int((manifest.get("uav_image_path", pd.Series(dtype=str)).astype(str) == "").sum()) if not manifest.empty else 0
    missing_ranked_paths = int((manifest.get("phog_ranked_csv_path", pd.Series(dtype=str)).astype(str) == "").sum()) if not manifest.empty else 0

    report = {
        "stage": "S5A.0_failure_group_benchmark_manifest",
        "sequence": args.sequence,
        "threshold_m": args.threshold_m,
        "input_s4c6_dir": str(args.s4c6_dir),
        "group_summary_path": str(group_summary_path) if group_summary_path else None,
        "summary_json_path": str(summary_json_path) if summary_json_path else None,
        "outputs": {
            "manifest_csv": str(manifest_path),
            "group_count_figure": str(figure_path),
        },
        "manifest": {
            "rows": int(len(manifest)),
            "unique_tokens": int(manifest["token"].nunique()) if not manifest.empty else 0,
            "failure_group_counts": {str(k): int(v) for k, v in counts.items()},
            "candidate_pool_has_40m_count": int(manifest["candidate_pool_has_40m"].sum()) if not manifest.empty else 0,
            "selection_failure_flag_count": int(manifest["selection_failure_flag"].sum()) if not manifest.empty else 0,
            "missing_uav_image_paths": missing_uav_paths,
            "missing_phog_ranked_csv_paths": missing_ranked_paths,
        },
        "path_enrichment": {
            "uav_index": uav_index_info,
            "ranked_csvs_indexed": ranked_csv_count,
        },
        "input_discovery": discovery_infos,
        "locked_rule": "Reference coordinates are evaluation-only and must not be used inside retrieval/ranking/scoring.",
        "next_step": "S5A.1 local verifier interface inside PHOG top-K using this manifest.",
    }
    return report


# ----------------------------- main -----------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build S5A failure-group benchmark manifest from S4C.6C outputs.")
    parser.add_argument("--sequence", default="traj01", help="SatLoc UAV sequence name, e.g. traj01.")
    parser.add_argument(
        "--s4c6-dir",
        type=Path,
        default=Path("outputs/satloc/metadata/s4c_macrocontour_phog_chamfer/s4c6_failure_analysis"),
        help="Directory containing S4C.6C failure-analysis CSV outputs.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/satloc"),
        help="Base SatLoc output directory.",
    )
    parser.add_argument(
        "--ranked-root",
        type=Path,
        default=Path("outputs/satloc/metadata/s4c_macrocontour_phog_chamfer"),
        help="Root to search for per-token PHOG ranked candidate CSVs.",
    )
    parser.add_argument(
        "--uav-index",
        type=Path,
        default=None,
        help="Optional path to UAV frame index CSV. If omitted, common SatLoc output paths are searched.",
    )
    parser.add_argument("--threshold-m", type=float, default=40.0, help="Success/error threshold in meters.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    s4c6_dir = args.s4c6_dir
    if not s4c6_dir.exists():
        raise FileNotFoundError(f"S4C.6C directory not found: {s4c6_dir}")

    stage_rel = Path("s5a_learned_local_verifier")
    metadata_dir = args.out_dir / "metadata" / stage_rel
    reports_dir = args.out_dir / "reports" / stage_rel
    figures_dir = args.out_dir / "figures" / stage_rel
    for d in [metadata_dir, reports_dir, figures_dir]:
        d.mkdir(parents=True, exist_ok=True)

    token_table, discovery_infos = discover_failure_token_table(s4c6_dir)
    if token_table.empty:
        raise RuntimeError(
            "Could not build token-level failure table. Check that S4C.6C CSVs contain "
            "a token column/list and a failure_group column."
        )

    group_summary, group_summary_path = read_group_summary(s4c6_dir)
    summary_json, summary_json_path = read_summary_json(s4c6_dir)

    uav_index_path = discover_uav_index(args.out_dir, args.sequence, args.uav_index)
    uav_path_map, uav_index_info = build_uav_path_map(uav_index_path, args.sequence)

    ranked_csvs = index_ranked_csvs(args.ranked_root)
    manifest = build_manifest(token_table, args.threshold_m, uav_path_map, ranked_csvs)

    manifest_path = metadata_dir / "s5a0_failure_group_benchmark_manifest.csv"
    report_path = reports_dir / "s5a0_failure_group_benchmark_summary.json"
    figure_path = figures_dir / "s5a0_failure_group_counts.png"

    manifest.to_csv(manifest_path, index=False)
    plot_group_counts(manifest, figure_path)

    report = make_report(
        manifest=manifest,
        args=args,
        discovery_infos=discovery_infos,
        group_summary_path=group_summary_path,
        summary_json_path=summary_json_path,
        uav_index_info=uav_index_info,
        ranked_csv_count=len(ranked_csvs),
        manifest_path=manifest_path,
        figure_path=figure_path,
    )

    # Add compact expected counts if the group summary table was available.
    if group_summary is not None:
        group_col = find_column(group_summary, GROUP_COLUMN_CANDIDATES)
        count_col = find_column(group_summary, ["count", "n", "num", "tokens", "frames"])
        if group_col and count_col:
            expected = {}
            for _, row in group_summary.iterrows():
                group = canonical_group_name(row.get(group_col))
                count = to_int_token(row.get(count_col))
                if count is not None:
                    expected[group] = count
            report["s4c6_group_summary_counts"] = expected

    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("S5A.0 failure-group benchmark manifest complete")
    print("-------------------------------------------------")
    print(f"Sequence:             {args.sequence}")
    print(f"Tokens in manifest:   {len(manifest)}")
    print(f"Failure groups:       {dict(Counter(manifest['failure_group']))}")
    print(f"Manifest CSV:         {manifest_path}")
    print(f"Summary JSON:         {report_path}")
    print(f"Group count figure:   {figure_path}")
    print(f"Missing UAV paths:    {report['manifest']['missing_uav_image_paths']}")
    print(f"Missing ranked CSVs:  {report['manifest']['missing_phog_ranked_csv_paths']}")
    print()
    print("Locked rule: reference coordinates are evaluation-only; do not use them in retrieval/ranking/scoring.")


if __name__ == "__main__":
    main()
