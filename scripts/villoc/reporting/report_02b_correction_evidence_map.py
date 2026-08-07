'''
Run Command Used:

python scripts/villoc/reporting/report_02b_correction_evidence_map.py \
  --config configs/reporting/report_villoc_traj01_s8_figures.yaml \
  --min-gap-label-m 40 \
  --max-gap-labels 12 \
  2>&1 | tee outputs/villoc/traj01_90deg_stable120m/reporting/logs/report_02b_correction_evidence_map.log

'''

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import yaml
except ImportError as exc:
    raise SystemExit("Install PyYAML with: pip install pyyaml") from exc


# ---------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------

def norm_col(c: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(c).lower()).strip("_")


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_csv(path: Path, label: str, required: bool = True) -> pd.DataFrame:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"{label}: missing file: {path}")
        return pd.DataFrame()
    df = pd.read_csv(path)
    if len(df) == 0 and required:
        raise ValueError(f"{label}: empty CSV: {path}")
    return df


def find_first_existing_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    if df is None or len(df.columns) == 0:
        return None
    normalized = {norm_col(c): c for c in df.columns}
    for cand in candidates:
        key = norm_col(cand)
        if key in normalized:
            return normalized[key]
    return None


def find_col_contains(
    df: pd.DataFrame,
    include_any: List[str],
    exclude_any: Optional[List[str]] = None,
    numeric_only: bool = False,
) -> Optional[str]:
    exclude_any = exclude_any or []
    for c in df.columns:
        nc = norm_col(c)
        if numeric_only and not pd.api.types.is_numeric_dtype(df[c]):
            continue
        if any(k in nc for k in include_any) and not any(k in nc for k in exclude_any):
            return c
    return None


def value_from_row(row: pd.Series, candidates: List[str], default: Any = "") -> Any:
    norm_to_col = {norm_col(c): c for c in row.index}
    for cand in candidates:
        key = norm_col(cand)
        if key in norm_to_col:
            val = row[norm_to_col[key]]
            if pd.notna(val):
                return val
    return default


def choose_identity_col(df: pd.DataFrame) -> Optional[str]:
    return find_first_existing_col(
        df,
        [
            "token0_id",
            "query_id",
            "sample_id",
            "frame_id",
            "frame_idx",
            "zero_based_frame_index",
            "index",
        ],
    )


def choose_common_identity_col(left: pd.DataFrame, right: pd.DataFrame) -> Optional[str]:
    candidates = [
        "token0_id",
        "query_id",
        "sample_id",
        "frame_id",
        "frame_idx",
        "zero_based_frame_index",
        "index",
    ]
    for c in candidates:
        lc = find_first_existing_col(left, [c])
        rc = find_first_existing_col(right, [c])
        if lc and rc:
            return lc
    return None


def choose_xy_cols(df: pd.DataFrame, role: str) -> Tuple[str, str, str]:
    if role == "reference":
        pairs = [
            ("x_enu_m", "y_enu_m"),
            ("reference_x_m", "reference_y_m"),
            ("ref_x_m", "ref_y_m"),
            ("gt_x_m", "gt_y_m"),
            ("x_ref_m", "y_ref_m"),
            ("x_m", "y_m"),
        ]
    else:
        pairs = [
            ("fused_x_m", "fused_y_m"),
            ("x_fused_m", "y_fused_m"),
            ("estimated_x_m", "estimated_y_m"),
            ("est_x_m", "est_y_m"),
            ("x_est_m", "y_est_m"),
            ("trajectory_x_m", "trajectory_y_m"),
            ("x_m", "y_m"),
        ]

    for x, y in pairs:
        xc = find_first_existing_col(df, [x])
        yc = find_first_existing_col(df, [y])
        if xc and yc:
            return xc, yc, f"matched explicit pair {x}/{y}"

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    x_candidates, y_candidates = [], []

    for c in numeric_cols:
        nc = norm_col(c)
        if role != "reference" and any(bad in nc for bad in ["ref", "reference", "gt", "truth"]):
            continue
        if nc in ["x", "x_m"] or nc.endswith("_x_m") or nc.endswith("_x") or "east" in nc:
            x_candidates.append(c)
        if nc in ["y", "y_m"] or nc.endswith("_y_m") or nc.endswith("_y") or "north" in nc:
            y_candidates.append(c)

    if x_candidates and y_candidates:
        return x_candidates[0], y_candidates[0], "fallback numeric x/y pattern"

    raise ValueError(f"Could not infer XY columns for role={role}. Columns: {list(df.columns)}")


def choose_distance_col(df: pd.DataFrame) -> Optional[str]:
    return find_first_existing_col(
        df,
        [
            "traveled_distance_m",
            "travelled_distance_m",
            "distance_m",
            "path_distance_m",
            "cum_distance_m",
            "cumulative_distance_m",
            "distance_along_path_m",
        ],
    )


def choose_error_col(df: pd.DataFrame) -> Optional[str]:
    return find_first_existing_col(
        df,
        [
            "error_m",
            "position_error_m",
            "pos_error_m",
            "eval_error_m",
            "fused_error_m",
            "trajectory_error_m",
            "xy_error_m",
        ],
    )


def compute_cumulative_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    dx = np.diff(x, prepend=x[0])
    dy = np.diff(y, prepend=y[0])
    step = np.sqrt(dx * dx + dy * dy)
    step[0] = 0.0
    return np.cumsum(step)


def enu_to_latlon_approx(
    x: np.ndarray,
    y: np.ndarray,
    x0: float,
    y0: float,
    lat0: float,
    lon0: float,
) -> Tuple[np.ndarray, np.ndarray]:
    r = 6378137.0
    lat0_rad = math.radians(lat0)
    d_north = y - y0
    d_east = x - x0
    lat = lat0 + (d_north / r) * (180.0 / math.pi)
    lon = lon0 + (d_east / (r * math.cos(lat0_rad))) * (180.0 / math.pi)
    return lat, lon


# ---------------------------------------------------------------------
# Fusion filtering and event extraction
# ---------------------------------------------------------------------

def filter_by_method_and_alpha(
    df: pd.DataFrame,
    method_name: str,
    alpha: Optional[float],
    label: str,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    info: Dict[str, Any] = {
        "label": label,
        "input_rows": int(len(df)),
        "method_name": method_name,
        "alpha": alpha,
        "method_filter_column": None,
        "alpha_filter_column": None,
        "output_rows": None,
        "fallback": None,
    }

    out = df.copy()

    string_cols = [
        c for c in out.columns
        if out[c].dtype == object or "string" in str(out[c].dtype).lower()
    ]

    method_l = method_name.lower()
    chosen_col = None
    chosen_mask = None

    for c in string_cols:
        s = out[c].astype(str).str.lower()
        mask = s.str.contains(re.escape(method_l), na=False)
        if mask.any():
            chosen_col = c
            chosen_mask = mask
            break

    if chosen_col is not None and chosen_mask is not None:
        out = out.loc[chosen_mask].copy()
        info["method_filter_column"] = chosen_col
    else:
        info["fallback"] = "no exact method filter matched; using full table"

    if alpha is not None and len(out) > 0:
        alpha_cols = [
            c for c in out.columns
            if "alpha" in norm_col(c) and pd.api.types.is_numeric_dtype(out[c])
        ]
        if alpha_cols:
            ac = alpha_cols[0]
            mask = np.isclose(out[ac].astype(float), float(alpha), atol=1e-9)
            if mask.any():
                out = out.loc[mask].copy()
                info["alpha_filter_column"] = ac

    info["output_rows"] = int(len(out))
    return out, info


def collapse_to_one_row_per_query(df: pd.DataFrame, label: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    info: Dict[str, Any] = {"label": label, "input_rows": int(len(df)), "identity_col": None}

    ident = choose_identity_col(df)
    if ident is None:
        out = df.reset_index(drop=True)
        info["output_rows"] = int(len(out))
        info["note"] = "no identity column found"
        return out, info

    info["identity_col"] = ident

    if df[ident].duplicated().any():
        out = df.sort_values(by=ident).drop_duplicates(subset=[ident], keep="first").copy()
        info["note"] = "collapsed duplicate identity rows"
    else:
        out = df.sort_values(by=ident).copy()
        info["note"] = "already unique"

    info["output_rows"] = int(len(out))
    return out.reset_index(drop=True), info


def add_report_distance_error(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    ref_xy: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> pd.DataFrame:
    out = df.copy()
    x = out[x_col].astype(float).to_numpy()
    y = out[y_col].astype(float).to_numpy()

    dist_col = choose_distance_col(out)
    err_col = choose_error_col(out)

    if dist_col:
        out["_report_distance_m"] = out[dist_col].astype(float)
    else:
        out["_report_distance_m"] = compute_cumulative_distance(x, y)

    if err_col:
        out["_report_error_m"] = out[err_col].astype(float)
    elif ref_xy is not None and len(ref_xy[0]) == len(out):
        rx, ry = ref_xy
        out["_report_error_m"] = np.sqrt((x - rx) ** 2 + (y - ry) ** 2)
    else:
        out["_report_error_m"] = np.nan

    return out


def find_event_rows(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    info = {
        "event_columns_used": [],
        "input_rows": int(len(df)),
        "event_rows": 0,
    }

    if len(df) == 0:
        return df.copy(), info

    event_mask = pd.Series(False, index=df.index)

    for c in df.columns:
        nc = norm_col(c)
        if any(k in nc for k in ["accepted", "correction_applied", "is_correction", "event"]):
            s = df[c]
            before = int(event_mask.sum())

            if pd.api.types.is_bool_dtype(s):
                event_mask = event_mask | s.fillna(False)
            elif pd.api.types.is_numeric_dtype(s):
                event_mask = event_mask | (s.fillna(0).astype(float) > 0)
            else:
                sl = s.astype(str).str.lower()
                event_mask = event_mask | sl.isin(["true", "1", "yes", "accepted", "accept", "applied"])

            after = int(event_mask.sum())
            if after > before:
                info["event_columns_used"].append(c)

    events = df.loc[event_mask].copy()
    info["event_rows"] = int(len(events))
    return events, info


# ---------------------------------------------------------------------
# Absolute selection metadata
# ---------------------------------------------------------------------

def best_candidate_by_query(candidate_scores: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Build one selected-candidate row per query from candidate score table.
    It tries rank columns first, then score columns.
    """
    info: Dict[str, Any] = {
        "input_rows": int(len(candidate_scores)),
        "identity_col": None,
        "rank_col": None,
        "score_col": None,
        "selection_rule": None,
        "output_rows": 0,
    }

    if candidate_scores.empty:
        return pd.DataFrame(), info

    ident = choose_identity_col(candidate_scores)
    if ident is None:
        info["selection_rule"] = "failed_no_identity_col"
        return pd.DataFrame(), info

    info["identity_col"] = ident

    rank_candidates = [
        "rerank_rank",
        "orb_rank",
        "verifier_rank",
        "final_rank",
        "selected_rank",
        "rank_after_rerank",
        "rank",
        "candidate_rank",
    ]
    rank_col = find_first_existing_col(candidate_scores, rank_candidates)

    if rank_col and pd.api.types.is_numeric_dtype(candidate_scores[rank_col]):
        out = (
            candidate_scores
            .sort_values([ident, rank_col])
            .drop_duplicates(subset=[ident], keep="first")
            .copy()
        )
        info["rank_col"] = rank_col
        info["selection_rule"] = f"min {rank_col}"
        info["output_rows"] = int(len(out))
        return out, info

    score_candidates = [
        "orb_score",
        "verifier_score",
        "hybrid_score",
        "score",
        "rerank_score",
        "final_score",
        "inliers",
        "num_inliers",
    ]
    score_col = find_first_existing_col(candidate_scores, score_candidates)

    if score_col and pd.api.types.is_numeric_dtype(candidate_scores[score_col]):
        out = (
            candidate_scores
            .sort_values([ident, score_col], ascending=[True, False])
            .drop_duplicates(subset=[ident], keep="first")
            .copy()
        )
        info["score_col"] = score_col
        info["selection_rule"] = f"max {score_col}"
        info["output_rows"] = int(len(out))
        return out, info

    info["selection_rule"] = "failed_no_rank_or_score_col"
    return pd.DataFrame(), info


def row_by_identity(df: pd.DataFrame, ident_col: str, ident_value: Any) -> Optional[pd.Series]:
    if df.empty or ident_col not in df.columns:
        return None
    m = df[ident_col].astype(str) == str(ident_value)
    if m.any():
        return df.loc[m].iloc[0]
    return None


def extract_selected_metadata(row: Optional[pd.Series]) -> Dict[str, Any]:
    if row is None:
        return {
            "selected_tile_id": "",
            "selected_tile_variant": "",
            "selected_tile_center_x_m": "",
            "selected_tile_center_y_m": "",
            "selected_tile_latitude": "",
            "selected_tile_longitude": "",
            "dino_rank": "",
            "rerank_rank": "",
            "orb_score": "",
            "orb_good_matches": "",
            "orb_inliers": "",
            "orb_inlier_ratio": "",
            "query_coverage": "",
        }

    return {
        "selected_tile_id": value_from_row(
            row,
            [
                "selected_tile_id",
                "chosen_tile_id",
                "top1_tile_id",
                "tile_id",
                "candidate_tile_id",
                "sat_tile_id",
                "satellite_tile_id",
                "tile_index",
                "map_tile_id",
            ],
        ),
        "selected_tile_variant": value_from_row(
            row,
            [
                "variant",
                "tile_variant",
                "selected_tile_variant",
                "map_variant",
            ],
        ),
        "selected_tile_center_x_m": value_from_row(
            row,
            [
                "selected_tile_center_x_m",
                "tile_center_x_m",
                "candidate_center_x_m",
                "center_x_m",
                "x_center_m",
            ],
        ),
        "selected_tile_center_y_m": value_from_row(
            row,
            [
                "selected_tile_center_y_m",
                "tile_center_y_m",
                "candidate_center_y_m",
                "center_y_m",
                "y_center_m",
            ],
        ),
        "selected_tile_latitude": value_from_row(
            row,
            [
                "selected_tile_latitude",
                "tile_center_lat",
                "tile_center_latitude",
                "candidate_latitude",
                "center_lat",
                "lat",
                "latitude",
            ],
        ),
        "selected_tile_longitude": value_from_row(
            row,
            [
                "selected_tile_longitude",
                "tile_center_lon",
                "tile_center_longitude",
                "candidate_longitude",
                "center_lon",
                "lon",
                "longitude",
            ],
        ),
        "dino_rank": value_from_row(
            row,
            [
                "dino_rank",
                "original_rank",
                "retrieval_rank",
                "candidate_rank",
                "rank",
            ],
        ),
        "rerank_rank": value_from_row(
            row,
            [
                "rerank_rank",
                "orb_rank",
                "verifier_rank",
                "final_rank",
                "rank_after_rerank",
            ],
        ),
        "orb_score": value_from_row(
            row,
            [
                "orb_score",
                "verifier_score",
                "hybrid_score",
                "score",
                "rerank_score",
            ],
        ),
        "orb_good_matches": value_from_row(
            row,
            [
                "orb_good_matches",
                "good_matches",
                "num_good_matches",
                "matches",
                "num_matches",
            ],
        ),
        "orb_inliers": value_from_row(
            row,
            [
                "orb_inliers",
                "inliers",
                "ransac_inliers",
                "num_inliers",
            ],
        ),
        "orb_inlier_ratio": value_from_row(
            row,
            [
                "orb_inlier_ratio",
                "inlier_ratio",
                "ransac_inlier_ratio",
            ],
        ),
        "query_coverage": value_from_row(
            row,
            [
                "query_coverage",
                "coverage",
                "uav_coverage",
            ],
        ),
    }


# ---------------------------------------------------------------------
# Event table
# ---------------------------------------------------------------------

def build_enriched_event_table(
    ref: pd.DataFrame,
    ref_xy: Tuple[str, str],
    query_manifest: pd.DataFrame,
    orb_query_summary: pd.DataFrame,
    best_candidate: pd.DataFrame,
    periodic_events: pd.DataFrame,
    periodic_xy: Tuple[str, str],
    temporal_events: pd.DataFrame,
    temporal_xy: Tuple[str, str],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    info: Dict[str, Any] = {
        "query_manifest_identity": None,
        "orb_query_summary_identity": None,
        "best_candidate_identity": None,
        "events_built": 0,
    }

    lat_col = find_first_existing_col(ref, ["lat", "latitude", "reference_latitude"])
    lon_col = find_first_existing_col(ref, ["lon", "longitude", "reference_longitude"])

    if not lat_col or not lon_col:
        raise ValueError("Reference trajectory must contain lat/lon.")

    ref_x, ref_y = ref_xy
    x0 = float(ref[ref_x].iloc[0])
    y0 = float(ref[ref_y].iloc[0])
    lat0 = float(ref[lat_col].iloc[0])
    lon0 = float(ref[lon_col].iloc[0])

    qm_ident = choose_identity_col(query_manifest)
    oq_ident = choose_identity_col(orb_query_summary)
    bc_ident = choose_identity_col(best_candidate)

    info["query_manifest_identity"] = qm_ident
    info["orb_query_summary_identity"] = oq_ident
    info["best_candidate_identity"] = bc_ident

    rows: List[Dict[str, Any]] = []

    def add_events(source: str, events: pd.DataFrame, xy: Tuple[str, str]) -> None:
        event_ident = choose_identity_col(events)

        sorted_events = events.sort_values("_report_distance_m").copy()

        for i, (_, row) in enumerate(sorted_events.iterrows(), start=1):
            query_identity = row[event_ident] if event_ident and event_ident in row.index else ""

            qm_row = row_by_identity(query_manifest, qm_ident, query_identity) if qm_ident else None
            oq_row = row_by_identity(orb_query_summary, oq_ident, query_identity) if oq_ident else None
            bc_row = row_by_identity(best_candidate, bc_ident, query_identity) if bc_ident else None

            selected_meta = extract_selected_metadata(oq_row)
            selected_from = "orb_query_summary"

            # Fill missing query-summary fields from candidate score table if possible.
            bc_meta = extract_selected_metadata(bc_row)
            for k, v in selected_meta.items():
                if v == "" or pd.isna(v):
                    if bc_meta.get(k, "") != "":
                        selected_meta[k] = bc_meta[k]
                        selected_from = "candidate_scores_best_row"

            x = float(row[xy[0]])
            y = float(row[xy[1]])
            lat, lon = enu_to_latlon_approx(
                np.array([x]),
                np.array([y]),
                x0=x0,
                y0=y0,
                lat0=lat0,
                lon0=lon0,
            )

            token0_id = value_from_row(row, ["token0_id"], "")
            if token0_id == "" and qm_row is not None:
                token0_id = value_from_row(qm_row, ["token0_id"], query_identity)

            source_frame = ""
            if qm_row is not None:
                source_frame = value_from_row(
                    qm_row,
                    [
                        "source_frame_cnt",
                        "source_frame",
                        "src_frame",
                        "srcframe",
                        "frame_cnt",
                        "zero_based_frame_index",
                    ],
                )

            rows.append({
                "event_id": f"{source[0].upper()}{i:02d}",
                "correction_source": source,
                "query_identity": query_identity,
                "token0_id": token0_id,
                "source_frame_cnt": source_frame,
                "x_map_m": x,
                "y_map_m": y,
                "latitude_approx": float(lat[0]),
                "longitude_approx": float(lon[0]),
                "traveled_distance_m": float(row.get("_report_distance_m", np.nan)),
                "error_m_eval_only": float(row.get("_report_error_m", np.nan)),
                "selected_metadata_source": selected_from,
                **selected_meta,
                "accepted": True,
                "context_category": "uncertain",
                "context_notes": "",
                "annotation_confidence": "",
            })

    add_events("periodic", periodic_events, periodic_xy)
    add_events("temporal", temporal_events, temporal_xy)

    out = pd.DataFrame(rows)
    if len(out) == 0:
        info["events_built"] = 0
        return out, info

    out = out.sort_values(["traveled_distance_m", "correction_source", "event_id"]).reset_index(drop=True)

    # Distance from previous correction of same source.
    out["distance_from_previous_same_source_m"] = np.nan
    for source, group in out.groupby("correction_source"):
        idx = group.sort_values("traveled_distance_m").index
        vals = out.loc[idx, "traveled_distance_m"].astype(float).to_numpy()
        gaps = np.diff(vals, prepend=np.nan)
        out.loc[idx, "distance_from_previous_same_source_m"] = gaps

    # Distance from previous accepted correction of any source.
    vals = out["traveled_distance_m"].astype(float).to_numpy()
    out["distance_from_previous_any_accepted_m"] = np.diff(vals, prepend=np.nan)

    info["events_built"] = int(len(out))
    return out, info


# ---------------------------------------------------------------------
# Plots and Folium
# ---------------------------------------------------------------------

def plot_evidence_xy(
    ref: pd.DataFrame,
    ref_xy: Tuple[str, str],
    events: pd.DataFrame,
    out_png: Path,
    min_gap_label_m: float,
    max_gap_labels: int,
) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11.0, 9.0))

    ax.plot(
        ref[ref_xy[0]], ref[ref_xy[1]],
        color="black",
        linewidth=1.5,
        label="Reference trajectory",
    )

    style = {
        "periodic": {"color": "royalblue", "marker": "o"},
        "temporal": {"color": "seagreen", "marker": "o"},
    }

    for source, group in events.groupby("correction_source"):
        st = style.get(source, {"color": "gray", "marker": "o"})
        ax.scatter(
            group["x_map_m"],
            group["y_map_m"],
            s=62,
            marker=st["marker"],
            facecolors="none",
            edgecolors=st["color"],
            linewidths=1.8,
            label=f"{source} accepted corrections",
        )

        for _, r in group.iterrows():
            ax.text(
                float(r["x_map_m"]),
                float(r["y_map_m"]),
                str(r["event_id"]),
                fontsize=8,
                color=st["color"],
                ha="left",
                va="bottom",
            )

    # Draw and label important same-source gaps.
    gap_candidates = []
    for source, group in events.groupby("correction_source"):
        st = style.get(source, {"color": "gray", "marker": "o"})
        g = group.sort_values("traveled_distance_m").reset_index(drop=True)
        for i in range(1, len(g)):
            r0 = g.iloc[i - 1]
            r1 = g.iloc[i]
            gap = float(r1["distance_from_previous_same_source_m"])
            if np.isfinite(gap) and gap >= min_gap_label_m:
                gap_candidates.append((gap, source, r0, r1, st["color"]))

    # Prefer largest gaps if too many labels.
    gap_candidates = sorted(gap_candidates, key=lambda x: x[0], reverse=True)[:max_gap_labels]

    for gap, source, r0, r1, color in gap_candidates:
        x0, y0 = float(r0["x_map_m"]), float(r0["y_map_m"])
        x1, y1 = float(r1["x_map_m"]), float(r1["y_map_m"])
        mx, my = (x0 + x1) / 2.0, (y0 + y1) / 2.0

        ax.plot(
            [x0, x1],
            [y0, y1],
            color=color,
            linewidth=0.8,
            linestyle="--",
            alpha=0.45,
        )
        ax.text(
            mx,
            my,
            f"Δ{gap:.0f} m",
            fontsize=8,
            color=color,
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec=color, alpha=0.75),
        )

    ax.scatter(
        ref[ref_xy[0]].iloc[0],
        ref[ref_xy[1]].iloc[0],
        s=90,
        marker="*",
        color="black",
        label="Start",
    )
    ax.scatter(
        ref[ref_xy[0]].iloc[-1],
        ref[ref_xy[1]].iloc[-1],
        s=70,
        marker="s",
        color="black",
        label="End",
    )

    ax.set_title("Accepted correction evidence and inter-correction gaps", fontsize=15, fontweight="bold")
    ax.set_xlabel("Local X / East [m]")
    ax.set_ylabel("Local Y / North [m]")
    ax.axis("equal")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def make_interactive_map(
    ref: pd.DataFrame,
    events: pd.DataFrame,
    out_html: Path,
) -> Dict[str, Any]:
    try:
        import folium
    except Exception as exc:
        return {"status": "skipped", "reason": f"folium import failed: {exc}"}

    lat_col = find_first_existing_col(ref, ["lat", "latitude", "reference_latitude"])
    lon_col = find_first_existing_col(ref, ["lon", "longitude", "reference_longitude"])

    if not lat_col or not lon_col:
        return {"status": "skipped", "reason": "reference lat/lon columns not found"}

    center = [float(ref[lat_col].median()), float(ref[lon_col].median())]
    fmap = folium.Map(location=center, zoom_start=17, tiles="OpenStreetMap", control_scale=True)

    folium.PolyLine(
        locations=ref[[lat_col, lon_col]].dropna().values.tolist(),
        color="black",
        weight=2,
        opacity=0.85,
        tooltip="Reference trajectory",
    ).add_to(fmap)

    color_by_source = {"periodic": "blue", "temporal": "green"}

    for _, row in events.iterrows():
        source = str(row["correction_source"])
        color = color_by_source.get(source, "gray")

        popup_items = [
            ("Event", row.get("event_id", "")),
            ("Source", source),
            ("Token", row.get("token0_id", "")),
            ("Query identity", row.get("query_identity", "")),
            ("Source frame", row.get("source_frame_cnt", "")),
            ("Selected tile", row.get("selected_tile_id", "")),
            ("Tile variant", row.get("selected_tile_variant", "")),
            ("DINO/original rank", row.get("dino_rank", "")),
            ("Rerank rank", row.get("rerank_rank", "")),
            ("ORB score", row.get("orb_score", "")),
            ("ORB matches", row.get("orb_good_matches", "")),
            ("ORB inliers", row.get("orb_inliers", "")),
            ("ORB inlier ratio", row.get("orb_inlier_ratio", "")),
            ("Query coverage", row.get("query_coverage", "")),
            ("Distance along path", f"{row.get('traveled_distance_m', np.nan):.1f} m"),
            ("Gap from previous same source", f"{row.get('distance_from_previous_same_source_m', np.nan):.1f} m"),
            ("Gap from previous accepted", f"{row.get('distance_from_previous_any_accepted_m', np.nan):.1f} m"),
            ("Error eval-only", f"{row.get('error_m_eval_only', np.nan):.2f} m"),
            ("Context", row.get("context_category", "")),
            ("Notes", row.get("context_notes", "")),
        ]

        html = "<br>".join([f"<b>{k}:</b> {v}" for k, v in popup_items])

        folium.CircleMarker(
            location=[float(row["latitude_approx"]), float(row["longitude_approx"])],
            radius=6,
            color=color,
            fill=False,
            tooltip=f"{row.get('event_id', '')} | {source} | token {row.get('token0_id', '')}",
            popup=folium.Popup(html, max_width=460),
        ).add_to(fmap)

    # Same-source gap lines.
    for source, group in events.groupby("correction_source"):
        color = color_by_source.get(source, "gray")
        g = group.sort_values("traveled_distance_m").reset_index(drop=True)
        for i in range(1, len(g)):
            r0 = g.iloc[i - 1]
            r1 = g.iloc[i]
            gap = float(r1.get("distance_from_previous_same_source_m", np.nan))
            if not np.isfinite(gap):
                continue

            folium.PolyLine(
                locations=[
                    [float(r0["latitude_approx"]), float(r0["longitude_approx"])],
                    [float(r1["latitude_approx"]), float(r1["longitude_approx"])],
                ],
                color=color,
                weight=1.5,
                opacity=0.45,
                dash_array="5,5",
                tooltip=f"{source} correction gap: {gap:.1f} m",
            ).add_to(fmap)

    folium.Marker(
        location=[float(ref[lat_col].iloc[0]), float(ref[lon_col].iloc[0])],
        tooltip="Start",
        icon=folium.Icon(color="black", icon="play"),
    ).add_to(fmap)

    folium.Marker(
        location=[float(ref[lat_col].iloc[-1]), float(ref[lon_col].iloc[-1])],
        tooltip="End",
        icon=folium.Icon(color="gray", icon="stop"),
    ).add_to(fmap)

    out_html.parent.mkdir(parents=True, exist_ok=True)
    fmap.save(str(out_html))

    return {"status": "written", "path": str(out_html)}


def write_summary_md(events: pd.DataFrame, out_md: Path) -> None:
    out_md.parent.mkdir(parents=True, exist_ok=True)

    lines: List[str] = []
    lines.append("# Correction evidence summary\n")
    lines.append(f"- Total accepted correction events: {len(events)}")
    lines.append("")

    if len(events) == 0:
        lines.append("No accepted correction events detected.")
        out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    lines.append("## Event counts\n")
    counts = events["correction_source"].value_counts()
    for source, count in counts.items():
        lines.append(f"- {source}: {count}")
    lines.append("")

    lines.append("## Gap statistics by correction source\n")
    lines.append("| Source | Events | Median same-source gap [m] | Max same-source gap [m] |")
    lines.append("|---|---:|---:|---:|")
    for source, group in events.groupby("correction_source"):
        gaps = group["distance_from_previous_same_source_m"].dropna().astype(float)
        med = gaps.median() if len(gaps) else np.nan
        mx = gaps.max() if len(gaps) else np.nan
        lines.append(f"| {source} | {len(group)} | {med:.1f} | {mx:.1f} |")

    lines.append("")
    lines.append("## Report interpretation draft\n")
    lines.append(
        "Accepted correction events are not uniformly spaced. Dense clusters of corrections "
        "can indicate areas where the map evidence is visually distinctive, while long gaps "
        "can indicate weaker or less trustworthy map-matching evidence. The static PNG should "
        "show only event labels and selected distance gaps, while the Folium map stores the "
        "full token, source-frame, selected-tile, rank, and verifier evidence in popups."
    )

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/reporting/report_villoc_traj01_s8_figures.yaml",
    )
    parser.add_argument("--repo-root", default=".")
    parser.add_argument(
        "--out-prefix",
        default="outputs/villoc/traj01_90deg_stable120m/reporting",
    )
    parser.add_argument("--min-gap-label-m", type=float, default=40.0)
    parser.add_argument("--max-gap-labels", type=int, default=12)
    args = parser.parse_args()

    repo = Path(args.repo_root).resolve()
    cfg = load_yaml(repo / args.config)

    inputs = cfg["inputs"]
    story = cfg["story_policy"]

    ref = read_csv(repo / inputs["reference_trajectory"]["path"], "reference")
    query_manifest = read_csv(repo / inputs["query_manifest"]["path"], "query_manifest")
    orb_query_summary = read_csv(repo / inputs["orb_absolute_query_summary"]["path"], "orb_query_summary")
    candidate_scores = read_csv(repo / inputs["orb_absolute_candidate_scores"]["path"], "candidate_scores")

    f2 = read_csv(repo / inputs["fusion_f2_trajectory"]["path"], "fusion_f2")
    f3b = read_csv(repo / inputs["temporal_fusion_trajectory"]["path"], "temporal_fusion")

    best_candidate, best_candidate_info = best_candidate_by_query(candidate_scores)

    periodic_method = story["best_metric_result"]["method_name"]
    periodic_alpha = story["best_metric_result"].get("alpha", None)

    temporal_method = story["primary_result"]["method_name"]
    temporal_alpha = story["primary_result"].get("alpha", None)

    periodic, periodic_filter_info = filter_by_method_and_alpha(f2, periodic_method, periodic_alpha, "periodic")
    temporal, temporal_filter_info = filter_by_method_and_alpha(f3b, temporal_method, temporal_alpha, "temporal")

    periodic, periodic_collapse_info = collapse_to_one_row_per_query(periodic, "periodic")
    temporal, temporal_collapse_info = collapse_to_one_row_per_query(temporal, "temporal")
    ref, ref_collapse_info = collapse_to_one_row_per_query(ref, "reference")

    min_len = min(len(ref), len(periodic), len(temporal))
    if min_len < 10:
        raise ValueError(f"Too few rows after filtering/collapse: {min_len}")

    ref = ref.iloc[:min_len].reset_index(drop=True)
    periodic = periodic.iloc[:min_len].reset_index(drop=True)
    temporal = temporal.iloc[:min_len].reset_index(drop=True)

    ref_x, ref_y, ref_reason = choose_xy_cols(ref, "reference")
    periodic_x, periodic_y, periodic_reason = choose_xy_cols(periodic, "fusion")
    temporal_x, temporal_y, temporal_reason = choose_xy_cols(temporal, "fusion")

    ref_xy_arr = (
        ref[ref_x].astype(float).to_numpy(),
        ref[ref_y].astype(float).to_numpy(),
    )

    periodic = add_report_distance_error(periodic, periodic_x, periodic_y, ref_xy_arr)
    temporal = add_report_distance_error(temporal, temporal_x, temporal_y, ref_xy_arr)

    periodic_events, periodic_event_info = find_event_rows(periodic)
    temporal_events, temporal_event_info = find_event_rows(temporal)

    events, event_table_info = build_enriched_event_table(
        ref=ref,
        ref_xy=(ref_x, ref_y),
        query_manifest=query_manifest,
        orb_query_summary=orb_query_summary,
        best_candidate=best_candidate,
        periodic_events=periodic_events,
        periodic_xy=(periodic_x, periodic_y),
        temporal_events=temporal_events,
        temporal_xy=(temporal_x, temporal_y),
    )

    out_root = repo / args.out_prefix
    figs = out_root / "figures"
    maps = out_root / "maps"
    manifests = out_root / "manifests"
    tables = out_root / "tables"

    for d in [figs, maps, manifests, tables]:
        d.mkdir(parents=True, exist_ok=True)

    events_csv = manifests / "report_02b_correction_evidence_events.csv"
    xy_png = figs / "fig_correction_evidence_xy_with_gap_labels.png"
    html_map = maps / "fig_correction_evidence_interactive_map.html"
    summary_md = manifests / "report_02b_correction_evidence_summary.md"
    manifest_json = manifests / "report_02b_correction_evidence_manifest.json"

    events.to_csv(events_csv, index=False)

    plot_evidence_xy(
        ref=ref,
        ref_xy=(ref_x, ref_y),
        events=events,
        out_png=xy_png,
        min_gap_label_m=args.min_gap_label_m,
        max_gap_labels=args.max_gap_labels,
    )

    folium_info = make_interactive_map(
        ref=ref,
        events=events,
        out_html=html_map,
    )

    write_summary_md(events, summary_md)

    manifest = {
        "status": "PASS_REPORT_02B_CORRECTION_EVIDENCE_MAP",
        "outputs": {
            "events_csv": str(events_csv),
            "xy_png": str(xy_png),
            "folium_html": str(html_map),
            "summary_md": str(summary_md),
            "manifest_json": str(manifest_json),
        },
        "row_counts": {
            "reference": len(ref),
            "periodic_rows": len(periodic),
            "temporal_rows": len(temporal),
            "periodic_events": len(periodic_events),
            "temporal_events": len(temporal_events),
            "enriched_events": len(events),
            "candidate_score_rows": len(candidate_scores),
            "best_candidate_rows": len(best_candidate),
        },
        "columns_used": {
            "reference": {"x": ref_x, "y": ref_y, "reason": ref_reason},
            "periodic": {"x": periodic_x, "y": periodic_y, "reason": periodic_reason},
            "temporal": {"x": temporal_x, "y": temporal_y, "reason": temporal_reason},
        },
        "filtering": {
            "periodic": periodic_filter_info,
            "periodic_collapse": periodic_collapse_info,
            "temporal": temporal_filter_info,
            "temporal_collapse": temporal_collapse_info,
            "reference_collapse": ref_collapse_info,
        },
        "event_detection": {
            "periodic": periodic_event_info,
            "temporal": temporal_event_info,
        },
        "best_candidate_selection": best_candidate_info,
        "event_table_info": event_table_info,
        "folium": folium_info,
        "notes": [
            "Reference and error fields remain evaluation-only.",
            "Token/source-frame/tile/rank/verifier metadata is used for report audit and visual explanation.",
            "Static PNG intentionally shows compact event labels and selected gap distances.",
            "Folium map contains the detailed token/source-frame/tile/rank/verifier evidence in popups.",
        ],
    }

    with manifest_json.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("STATUS: PASS_REPORT_02B_CORRECTION_EVIDENCE_MAP")
    print(f"Events CSV:      {events_csv}")
    print(f"XY gap PNG:      {xy_png}")
    print(f"Folium map:      {html_map} [{folium_info.get('status')}]")
    print(f"Summary MD:      {summary_md}")
    print(f"Manifest JSON:   {manifest_json}")
    print("")
    print("Rows:")
    print(json.dumps(manifest["row_counts"], indent=2))
    print("")
    print("Best candidate selection:")
    print(json.dumps(best_candidate_info, indent=2))
    print("")
    print("Event table identity info:")
    print(json.dumps(event_table_info, indent=2))


if __name__ == "__main__":
    main()
