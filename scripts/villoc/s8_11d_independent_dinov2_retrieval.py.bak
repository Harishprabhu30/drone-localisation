'''
Command Executed:

python scripts/villoc/s8_11d_independent_dinov2_retrieval.py \
  2>&1 | tee outputs/villoc/90_deg/logs/s8_11d_independent_dinov2_retrieval.log

'''

from __future__ import annotations

import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path.cwd().resolve()

TAG = "dinov2_vits14_img224_center_square_avgpatch_cpu"

QUERY_CSV = ROOT / "outputs/villoc/90_deg/metadata/s8_10b_canonical_uav_query_manifest.csv"

QUERY_CACHE = ROOT / f"outputs/villoc/90_deg/descriptors/s8_11c_dinov2_queries_v_1fps_{TAG}.npz"

VARIANTS = {
    "512_s256": {
        "tile_index": ROOT / "outputs/villoc/90_deg/metadata/s8_9_satellite_tile_index_512_s256.csv",
        "oracle": ROOT / "outputs/villoc/90_deg/metadata/s8_10b_uav_tile_oracle_512_s256.csv",
        "map_cache": ROOT / f"outputs/villoc/90_deg/descriptors/s8_11b_dinov2_map_512_s256_{TAG}.npz",
    },
    "1024_s512": {
        "tile_index": ROOT / "outputs/villoc/90_deg/metadata/s8_9_satellite_tile_index_1024_s512.csv",
        "oracle": ROOT / "outputs/villoc/90_deg/metadata/s8_10b_uav_tile_oracle_1024_s512.csv",
        "map_cache": ROOT / f"outputs/villoc/90_deg/descriptors/s8_11b_dinov2_map_1024_s512_{TAG}.npz",
    },
    "1024_s256": {
        "tile_index": ROOT / "outputs/villoc/90_deg/metadata/s8_9_satellite_tile_index_1024_s256.csv",
        "oracle": ROOT / "outputs/villoc/90_deg/metadata/s8_10b_uav_tile_oracle_1024_s256.csv",
        "map_cache": ROOT / f"outputs/villoc/90_deg/descriptors/s8_11b_dinov2_map_1024_s256_{TAG}.npz",
    },
}

OUT_DIR = ROOT / "outputs/villoc/90_deg/retrieval/s8_11d"
REPORT_DIR = ROOT / "outputs/villoc/90_deg/reports/s8_11d"

RECALL_KS = [1, 5, 10, 20, 50, 100]


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_npz_cache(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)

    data = np.load(path, allow_pickle=False)

    meta_raw = data["meta_json"]
    if hasattr(meta_raw, "item"):
        meta_raw = meta_raw.item()

    return {
        "descriptors": data["descriptors"].astype(np.float32),
        "ids": data["ids"].astype(str),
        "paths": data["paths"].astype(str),
        "meta": json.loads(str(meta_raw)),
    }


def safe_col(df: pd.DataFrame, candidates: list[str], table_name: str) -> str:
    lower = {c.lower(): c for c in df.columns}

    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]

    raise KeyError(
        f"{table_name}: none of these columns found: {candidates}. "
        f"Available columns: {list(df.columns)}"
    )


def truthy_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s

    lowered = s.astype(str).str.strip().str.lower()
    return lowered.isin(["1", "true", "yes", "y", "oracle", "inside", "contained"])


def load_oracle_sets(path: Path) -> dict[str, set[str]]:
    """
    Load S8.10B oracle audit CSV.

    S8.10B oracle files are wide per-query audit tables, not long relation
    tables. The oracle tile IDs are stored in a packed column:

        oracle_tile_ids

    This function converts that wide table into:

        {query_id: {tile_id_1, tile_id_2, ...}}

    Coordinates/oracles are still evaluation-only. They are not used for
    descriptor extraction or ranking.
    """
    import ast
    import re

    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    q_col = safe_col(
        df,
        ["query_id", "token0_id", "uav_query_id", "uav_id"],
        f"oracle:{path.name}",
    )

    def normalize_id(value) -> str | None:
        if value is None:
            return None

        try:
            if pd.isna(value):
                return None
        except Exception:
            pass

        s = str(value).strip()
        if not s or s.lower() in {"nan", "none", "null"}:
            return None

        s = s.strip("'\" ")

        # Convert CSV float-looking integer IDs such as "12.0" to "12".
        if re.fullmatch(r"\d+\.0", s):
            s = str(int(float(s)))

        return s

    def parse_packed_ids(value) -> list[str]:
        if value is None:
            return []

        try:
            if pd.isna(value):
                return []
        except Exception:
            pass

        s = str(value).strip()

        if not s or s.lower() in {"nan", "none", "null", "[]", "{}"}:
            return []

        # Preferred path: Python/JSON-like list string, e.g. "[1, 2, 3]"
        if s.startswith("[") or s.startswith("(") or s.startswith("{"):
            try:
                parsed = ast.literal_eval(s)
                if isinstance(parsed, (list, tuple, set)):
                    out = []
                    for item in parsed:
                        nid = normalize_id(item)
                        if nid is not None:
                            out.append(nid)
                    return out

                nid = normalize_id(parsed)
                return [nid] if nid is not None else []
            except Exception:
                pass

        # Fallback path: delimiter-separated values.
        # Supports "1,2,3", "1|2|3", "1;2;3".
        cleaned = (
            s.replace("[", "")
             .replace("]", "")
             .replace("(", "")
             .replace(")", "")
             .replace("{", "")
             .replace("}", "")
        )

        if "|" in cleaned:
            parts = cleaned.split("|")
        elif ";" in cleaned:
            parts = cleaned.split(";")
        elif "," in cleaned:
            parts = cleaned.split(",")
        else:
            parts = cleaned.split()

        out = []
        for part in parts:
            nid = normalize_id(part)
            if nid is not None:
                out.append(nid)

        return out

    groups: dict[str, set[str]] = {}

    if "oracle_tile_ids" in df.columns:
        for _, row in df.iterrows():
            qid = normalize_id(row[q_col])
            if qid is None:
                continue

            # Respect explicit no-oracle flag if present.
            if "has_oracle_tile" in df.columns:
                flag = str(row.get("has_oracle_tile", "")).strip().lower()
                if flag in {"false", "0", "no", "n"}:
                    groups[qid] = set()
                    continue

            groups[qid] = set(parse_packed_ids(row["oracle_tile_ids"]))

        return groups

    # Backward-compatible fallback for long oracle tables.
    tile_col = safe_col(
        df,
        ["tile_id", "oracle_tile_id", "sat_tile_id", "map_tile_id"],
        f"oracle:{path.name}",
    )

    possible_flag_cols = [
        c for c in df.columns
        if c.lower() in [
            "is_oracle",
            "is_oracle_tile",
            "oracle",
            "contains_query",
            "query_inside_tile",
            "inside_tile",
        ]
    ]

    if possible_flag_cols:
        flag_col = possible_flag_cols[0]
        df = df[truthy_series(df[flag_col])].copy()

    for qid, sub in df.groupby(q_col):
        nqid = normalize_id(qid)
        if nqid is None:
            continue

        groups[nqid] = {
            tid
            for tid in (normalize_id(v) for v in sub[tile_col].tolist())
            if tid is not None
        }

    return groups


def center_error_m(
    query_id: str,
    tile_id: str,
    query_xy: dict[str, tuple[float, float]],
    tile_xy: dict[str, tuple[float, float]],
) -> float:
    if query_id not in query_xy:
        return float("nan")
    if tile_id not in tile_xy:
        return float("nan")

    qx, qy = query_xy[query_id]
    tx, ty = tile_xy[tile_id]

    return float(math.hypot(qx - tx, qy - ty))


def first_oracle_rank(ranked_tile_ids: list[str], oracle_tiles: set[str]) -> int | None:
    if not oracle_tiles:
        return None

    for idx, tid in enumerate(ranked_tile_ids, start=1):
        if tid in oracle_tiles:
            return idx

    return None


def metric_float(x: float) -> float | None:
    if x is None:
        return None
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None
    return float(x)


def evaluate_variant(
    variant: str,
    q_cache: dict,
    m_cache: dict,
    query_full: pd.DataFrame,
    tile_index: pd.DataFrame,
    oracle_sets: dict[str, set[str]],
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    started = time.time()

    q_desc = q_cache["descriptors"]
    m_desc = m_cache["descriptors"]

    q_ids = q_cache["ids"].astype(str)
    tile_ids = m_cache["ids"].astype(str)

    if q_desc.shape[0] != len(q_ids):
        raise ValueError(f"{variant}: query descriptor rows do not match query IDs")
    if m_desc.shape[0] != len(tile_ids):
        raise ValueError(f"{variant}: map descriptor rows do not match tile IDs")
    if q_desc.shape[1] != m_desc.shape[1]:
        raise ValueError(f"{variant}: descriptor dimensions do not match")

    q_col = safe_col(query_full, ["query_id"], "query_manifest")
    qe_col = safe_col(query_full, ["easting"], "query_manifest")
    qn_col = safe_col(query_full, ["northing"], "query_manifest")

    tile_col = safe_col(tile_index, ["tile_id"], f"tile_index:{variant}")
    te_col = safe_col(tile_index, ["center_easting"], f"tile_index:{variant}")
    tn_col = safe_col(tile_index, ["center_northing"], f"tile_index:{variant}")

    query_xy = {
        str(row[q_col]): (float(row[qe_col]), float(row[qn_col]))
        for _, row in query_full.iterrows()
    }
    tile_xy = {
        str(row[tile_col]): (float(row[te_col]), float(row[tn_col]))
        for _, row in tile_index.iterrows()
    }

    max_k = min(max(RECALL_KS), len(tile_ids))

    # Descriptors were L2-normalized during cache creation, so dot product = cosine similarity.
    sim = q_desc @ m_desc.T

    candidate_rows = []
    query_rows = []

    for i, qid in enumerate(q_ids):
        scores = sim[i]

        # Full sort is fine here because Villoc has only 108-475 tiles per variant.
        order = np.argsort(-scores)[:max_k]

        ranked_tile_ids = [str(tile_ids[j]) for j in order]
        ranked_scores = [float(scores[j]) for j in order]

        oracle_tiles = oracle_sets.get(str(qid), set())
        oracle_rank = first_oracle_rank(ranked_tile_ids, oracle_tiles)

        top1_tile = ranked_tile_ids[0]
        top1_score = ranked_scores[0]
        top1_error = center_error_m(str(qid), top1_tile, query_xy, tile_xy)
        top1_is_oracle = top1_tile in oracle_tiles

        q_eval = {
            "variant": variant,
            "query_id": str(qid),
            "top1_tile_id": top1_tile,
            "top1_score": top1_score,
            "top1_center_error_m": top1_error,
            "top1_is_oracle": bool(top1_is_oracle),
            "oracle_tile_count": int(len(oracle_tiles)),
            "first_oracle_rank": oracle_rank if oracle_rank is not None else np.nan,
            "has_oracle": bool(len(oracle_tiles) > 0),
        }

        for k in RECALL_KS:
            kk = min(k, len(tile_ids))
            q_eval[f"recall_at_{k}"] = bool(
                oracle_rank is not None and oracle_rank <= kk
            )

        q_eval["top1_error_le_20m"] = bool(top1_error <= 20.0) if not math.isnan(top1_error) else False
        q_eval["top1_error_le_40m"] = bool(top1_error <= 40.0) if not math.isnan(top1_error) else False
        q_eval["top1_error_le_80m"] = bool(top1_error <= 80.0) if not math.isnan(top1_error) else False
        q_eval["top1_error_le_120m"] = bool(top1_error <= 120.0) if not math.isnan(top1_error) else False

        query_rows.append(q_eval)

        for rank, (tid, score) in enumerate(zip(ranked_tile_ids, ranked_scores), start=1):
            err = center_error_m(str(qid), tid, query_xy, tile_xy)

            candidate_rows.append(
                {
                    "variant": variant,
                    "query_id": str(qid),
                    "rank": rank,
                    "tile_id": tid,
                    "score": score,
                    "is_oracle": bool(tid in oracle_tiles),
                    "center_error_m": err,
                }
            )

    candidates_df = pd.DataFrame(candidate_rows)
    query_eval_df = pd.DataFrame(query_rows)

    errors = query_eval_df["top1_center_error_m"].astype(float).to_numpy()
    valid_errors = errors[np.isfinite(errors)]

    summary = {
        "variant": variant,
        "status": "PASS",
        "query_count": int(len(q_ids)),
        "tile_count": int(len(tile_ids)),
        "descriptor_dim": int(q_desc.shape[1]),
        "max_rank_evaluated": int(max_k),
        "queries_with_oracle": int(query_eval_df["has_oracle"].sum()),
        "queries_without_oracle": int((~query_eval_df["has_oracle"]).sum()),
        "top1_oracle_hits": int(query_eval_df["top1_is_oracle"].sum()),
        "top1_oracle_hit_rate": float(query_eval_df["top1_is_oracle"].mean()),
        "top1_error_le_20m_rate": float(query_eval_df["top1_error_le_20m"].mean()),
        "top1_error_le_40m_rate": float(query_eval_df["top1_error_le_40m"].mean()),
        "top1_error_le_80m_rate": float(query_eval_df["top1_error_le_80m"].mean()),
        "top1_error_le_120m_rate": float(query_eval_df["top1_error_le_120m"].mean()),
        "top1_center_error_mean_m": metric_float(float(np.mean(valid_errors))) if len(valid_errors) else None,
        "top1_center_error_median_m": metric_float(float(np.median(valid_errors))) if len(valid_errors) else None,
        "top1_center_error_rmse_m": metric_float(float(np.sqrt(np.mean(valid_errors ** 2)))) if len(valid_errors) else None,
        "top1_center_error_p95_m": metric_float(float(np.percentile(valid_errors, 95))) if len(valid_errors) else None,
        "runtime_s": float(time.time() - started),
        "retrieval_input": {
            "query_cache": "descriptors only",
            "map_cache": "descriptors only",
            "similarity": "cosine_dot_product_on_l2_normalized_descriptors",
        },
        "evaluation_only_inputs": {
            "query_coordinates": ["easting", "northing"],
            "tile_coordinates": ["center_easting", "center_northing"],
            "oracle_tile_sets": True,
        },
    }

    for k in RECALL_KS:
        summary[f"recall_at_{k}"] = float(query_eval_df[f"recall_at_{k}"].mean())
        summary[f"hits_at_{k}"] = int(query_eval_df[f"recall_at_{k}"].sum())

    ranks = query_eval_df["first_oracle_rank"].dropna().astype(float).to_numpy()
    if len(ranks):
        summary["first_oracle_rank_median"] = float(np.median(ranks))
        summary["first_oracle_rank_p95"] = float(np.percentile(ranks, 95))
    else:
        summary["first_oracle_rank_median"] = None
        summary["first_oracle_rank_p95"] = None

    return candidates_df, query_eval_df, summary


def main() -> None:
    print("S8.11D — Independent DINOv2 retrieval evaluation")
    print("------------------------------------------------")
    print("ROOT:", ROOT)
    print("TAG:", TAG)
    print("QUERY_CACHE:", QUERY_CACHE)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    if not QUERY_CSV.exists():
        raise FileNotFoundError(QUERY_CSV)
    if not QUERY_CACHE.exists():
        raise FileNotFoundError(QUERY_CACHE)

    query_full = pd.read_csv(QUERY_CSV)
    q_cache = load_npz_cache(QUERY_CACHE)

    all_summaries = []
    report = {
        "stage": "S8.11D",
        "status": "PASS",
        "created_at_utc": now_utc(),
        "tag": TAG,
        "query_cache": str(QUERY_CACHE),
        "variants": {},
        "leakage_rule": {
            "descriptor_retrieval_used_coordinates": False,
            "descriptor_retrieval_used_oracle": False,
            "coordinates_and_oracles_used_only_after_ranking_for_evaluation": True,
        },
    }

    for variant, paths in VARIANTS.items():
        print()
        print(f"[VARIANT] {variant}")

        for label, path in paths.items():
            print(f"  {label}: {path}")
            if not path.exists():
                raise FileNotFoundError(path)

        tile_index = pd.read_csv(paths["tile_index"])
        oracle_sets = load_oracle_sets(paths["oracle"])
        m_cache = load_npz_cache(paths["map_cache"])

        candidates_df, query_eval_df, summary = evaluate_variant(
            variant=variant,
            q_cache=q_cache,
            m_cache=m_cache,
            query_full=query_full,
            tile_index=tile_index,
            oracle_sets=oracle_sets,
        )

        candidates_path = OUT_DIR / f"s8_11d_topk_{variant}_{TAG}.csv"
        query_eval_path = OUT_DIR / f"s8_11d_query_eval_{variant}_{TAG}.csv"

        candidates_df.to_csv(candidates_path, index=False)
        query_eval_df.to_csv(query_eval_path, index=False)

        summary["outputs"] = {
            "topk_candidates_csv": str(candidates_path),
            "query_eval_csv": str(query_eval_path),
        }

        all_summaries.append(summary)
        report["variants"][variant] = summary

        print(
            f"  Recall@1={summary['recall_at_1']:.3f} "
            f"Recall@5={summary['recall_at_5']:.3f} "
            f"Recall@10={summary['recall_at_10']:.3f} "
            f"Recall@20={summary['recall_at_20']:.3f} "
            f"Recall@50={summary['recall_at_50']:.3f} "
            f"Recall@100={summary['recall_at_100']:.3f}"
        )
        print(
            f"  Top1 oracle hit rate={summary['top1_oracle_hit_rate']:.3f} "
            f"Top1 median err={summary['top1_center_error_median_m']:.2f} m "
            f"RMSE={summary['top1_center_error_rmse_m']:.2f} m"
        )
        print(f"  Wrote: {candidates_path}")
        print(f"  Wrote: {query_eval_path}")

    summary_df = pd.DataFrame(all_summaries)

    preferred_cols = [
        "variant",
        "query_count",
        "tile_count",
        "descriptor_dim",
        "recall_at_1",
        "recall_at_5",
        "recall_at_10",
        "recall_at_20",
        "recall_at_50",
        "recall_at_100",
        "top1_oracle_hit_rate",
        "top1_center_error_median_m",
        "top1_center_error_rmse_m",
        "top1_center_error_p95_m",
        "top1_error_le_40m_rate",
        "top1_error_le_80m_rate",
        "first_oracle_rank_median",
        "runtime_s",
    ]

    summary_csv = REPORT_DIR / f"s8_11d_independent_retrieval_summary_{TAG}.csv"
    summary_json = REPORT_DIR / f"s8_11d_independent_retrieval_summary_{TAG}.json"

    summary_df[preferred_cols].to_csv(summary_csv, index=False)
    summary_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print()
    print("------------------------------------------------")
    print("S8.11D COMPLETE")
    print("STATUS: PASS_INDEPENDENT_RETRIEVAL_EVAL")
    print("Summary CSV:", summary_csv)
    print("Summary JSON:", summary_json)

    print()
    print("Compact summary:")
    print(summary_df[preferred_cols].to_string(index=False))


if __name__ == "__main__":
    main()
