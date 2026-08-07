#!/usr/bin/env python3
"""
S8.12E.0 Footprint-center-ray diagnostic for Villoc 45-degree retrieval.

Purpose
-------
Evaluate whether 45-degree oblique DINO retrieval is closer to the visible
forward ground footprint than to the UAV body ground position.

This script is evaluation/diagnostic only:
- It never uses GNSS/reference/tile geometry to rank candidates.
- It reads an already-created Top-K retrieval CSV.
- It uses body coordinates, attitude, altitude, and tile coordinates only after
  ranking to compute diagnostic errors.

Main outputs
------------
<out-root>/s8_12e0_all_ranked_body_vs_footprint.csv
<out-root>/s8_12e0_top1_body_vs_footprint.csv
<out-root>/s8_12e0_footprint_center_ray_summary.json
<out-root>/figures/s8_12e0_top1_error_boxplot.png
<out-root>/figures/s8_12e0_offset_vs_footprint_improvement.png
<out-root>/figures/s8_12e0_corrected_body_scatter.png

Expected run
------------
python scripts/villoc/s8_12e0_footprint_center_ray_diagnostic.py \
  --config configs/dataset_villoc_45deg.yaml \
  --variant 1024_s512 \
  --tag dinov2_vits14_img224_center_square_avgpatch_cpu \
  --oracle-k 20
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependency: PyYAML. Install with: pip install pyyaml"
    ) from exc

try:
    from pyproj import Transformer
except ImportError as exc: 
    raise SystemExit(
        "Missing dependency: pyproj. Install with: pip install pyproj"
        ) from exc

try:
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependency: matplotlib. Install with: pip install matplotlib"
    ) from exc


# -----------------------------------------------------------------------------
# Small utility helpers
# -----------------------------------------------------------------------------


def die(message: str, exit_code: int = 2) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(exit_code)


def info(message: str) -> None:
    print(message, flush=True)


def load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        die(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        die(f"Config did not parse to a mapping: {path}")
    return data


def normalize_path(path_like: str | Path, base: Path) -> Path:
    p = Path(path_like).expanduser()
    if not p.is_absolute():
        p = base / p
    return p.resolve()


def flatten_config_values(obj: Any, prefix: str = "") -> List[Tuple[str, Any]]:
    out: List[Tuple[str, Any]] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            out.extend(flatten_config_values(v, key))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            key = f"{prefix}[{i}]"
            out.extend(flatten_config_values(v, key))
    else:
        out.append((prefix, obj))
    return out


def find_config_value(config: Dict[str, Any], target_keys: Sequence[str]) -> Optional[Any]:
    wanted = {k.lower() for k in target_keys}
    for key, value in flatten_config_values(config):
        tail = key.split(".")[-1].split("[")[0].lower()
        if tail in wanted and value not in (None, ""):
            return value
    return None


def all_existing_config_paths(config: Dict[str, Any], repo_root: Path) -> List[Path]:
    paths: List[Path] = []
    for _, value in flatten_config_values(config):
        if not isinstance(value, str):
            continue
        if not any(sep in value for sep in ("/", "\\")) and not value.endswith(('.csv', '.json', '.tif', '.npz')):
            continue
        p = normalize_path(value, repo_root)
        if p.exists():
            paths.append(p)
    return sorted(set(paths))


def read_csv_checked(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        die(f"{label} not found: {path}")
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        die(f"Could not read {label}: {path}\n{exc}")
    if df.empty:
        die(f"{label} is empty: {path}")
    df.columns = [str(c).strip() for c in df.columns]
    return df


def clean_col_name(col: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(col).strip().lower()).strip("_")


def column_lookup(df: pd.DataFrame) -> Dict[str, str]:
    return {clean_col_name(c): c for c in df.columns}


def find_col(
    df: pd.DataFrame,
    aliases: Sequence[str],
    required: bool = True,
    label: str = "dataframe",
) -> Optional[str]:
    lookup = column_lookup(df)
    for alias in aliases:
        key = clean_col_name(alias)
        if key in lookup:
            return lookup[key]
    if required:
        die(
            f"Missing required column in {label}.\n"
            f"Accepted aliases: {list(aliases)}\n"
            f"Available columns: {list(df.columns)}"
        )
    return None


def numeric_series(df: pd.DataFrame, col: str, label: str) -> pd.Series:
    s = pd.to_numeric(df[col], errors="coerce")
    if s.notna().sum() == 0:
        die(f"Column {col!r} in {label} is not numeric or is all NaN.")
    return s


def coerce_numeric_columns(df: pd.DataFrame, cols: Sequence[str], label: str) -> None:
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    missing = df[list(cols)].isna().any(axis=1).sum()
    if missing:
        info(f"WARNING: {label}: {missing} rows have NaN in required numeric columns and may be dropped.")


# -----------------------------------------------------------------------------
# Path resolution
# -----------------------------------------------------------------------------


def resolve_output_root(config: Dict[str, Any], repo_root: Path, cli_output_root: Optional[str]) -> Path:
    if cli_output_root:
        return normalize_path(cli_output_root, repo_root)
    value = find_config_value(config, ["output_root", "outputs_root", "output_dir", "out_root"])
    if value:
        return normalize_path(str(value), repo_root)
    die("Could not resolve output_root from config. Pass --output-root explicitly.")


def resolve_optional_path(path_str: Optional[str], repo_root: Path) -> Optional[Path]:
    if not path_str:
        return None
    p = normalize_path(path_str, repo_root)
    if not p.exists():
        die(f"Provided path does not exist: {p}")
    return p


def score_path_candidate(path: Path, must_terms: Sequence[str], prefer_terms: Sequence[str], avoid_terms: Sequence[str]) -> int:
    text = str(path).lower()
    score = 0
    for term in must_terms:
        if term and term.lower() not in text:
            return -10_000
        score += 20
    for term in prefer_terms:
        if term.lower() in text:
            score += 5
    for term in avoid_terms:
        if term.lower() in text:
            score -= 8
    # Prefer shorter/more canonical paths when tied.
    score -= min(len(path.parts), 20)
    return score


def choose_csv_by_schema(
    candidates: Sequence[Path],
    label: str,
    schema_checker,
) -> Path:
    checked: List[Tuple[int, Path, str]] = []
    for p in candidates:
        try:
            head = pd.read_csv(p, nrows=5)
            ok, reason, bonus = schema_checker(head)
            if ok:
                checked.append((bonus, p, reason))
        except Exception:
            continue
    if not checked:
        shown = "\n".join(str(p) for p in candidates[:20])
        die(f"Could not find a valid {label} CSV among candidates. First candidates:\n{shown}")
    checked.sort(key=lambda x: (x[0], -len(str(x[1]))), reverse=True)
    return checked[0][1]


def has_any(df: pd.DataFrame, aliases: Sequence[str]) -> bool:
    lookup = column_lookup(df)
    return any(clean_col_name(a) in lookup for a in aliases)


QUERY_ID_ALIASES = ["query_id", "token0_id", "sample_id", "frame_index", "zero_based_frame_index"]
TOPK_QUERY_ID_ALIASES = ["query_id", "token0_id", "sample_id", "uav_query_id"]
TILE_ID_ALIASES = ["tile_id", "sat_tile_id", "satellite_tile_id", "candidate_tile_id", "map_tile_id"]
RANK_ALIASES = ["rank", "retrieval_rank", "candidate_rank", "topk_rank", "top_k_rank"]
SCORE_ALIASES = ["score", "similarity", "cosine", "cosine_similarity", "retrieval_score", "sim"]

BODY_X_ALIASES = [
    "x_enu_m", "body_x_enu_m", "body_east_enu_m", "east_enu_m", "east_m",
    "easting_enu_m", "x_m", "body_x_m",
]
BODY_Y_ALIASES = [
    "y_enu_m", "body_y_enu_m", "body_north_enu_m", "north_enu_m", "north_m",
    "northing_enu_m", "y_m", "body_y_m",
]

PROJ_X_ALIASES = [
    "s8_12e0_query_easting_3346_m",
    "x_lks94_m", "y_lks_94_m_DO_NOT_USE",  # harmless typo guard, ignored by paired picker
    "x_3346_m", "easting_m", "center_easting_m", "lks94_x_m", "proj_x_m",
    "x_epsg3346_m", "x_projected_m", "x_map_m",
]

PROJ_Y_ALIASES = [
    "s8_12e0_query_northing_3346_m",
    "y_lks94_m", "y_3346_m", "northing_m", "center_northing_m", "lks94_y_m",
    "proj_y_m", "y_epsg3346_m", "y_projected_m", "y_map_m",
]

ALT_ALIASES = ["rel_alt_m", "relative_altitude_m", "height_agl_m", "altitude_agl_m", "agl_m"]
YAW_ALIASES = ["gb_yaw_deg", "yaw_deg", "gimbal_yaw_deg", "heading_deg", "camera_yaw_deg"]
PITCH_ALIASES = ["gb_pitch_deg", "pitch_deg", "gimbal_pitch_deg", "camera_pitch_deg"]

LAT_ALIASES = ["latitude", "lat", "body_lat", "query_lat", "uav_lat"]
LON_ALIASES = ["longitude", "lon", "body_lon", "query_lon", "uav_lon"]

TILE_CENTER_X_ALIASES = [
    "center_easting", "tile_center_easting", "center_easting_m", "tile_center_easting_m",
    "center_x_enu_m", "tile_center_x_enu_m", "center_east_enu_m", "center_easting_enu_m",
    "center_x_m", "tile_center_x_m", "x_center_m", "easting_m",
    "center_x_lks94_m", "center_x_3346_m", "x_3346_m", "x_lks94_m", "proj_center_x_m",
    "center_x", "x_center",
]
TILE_CENTER_Y_ALIASES = [
    "center_northing", "tile_center_northing", "center_northing_m", "tile_center_northing_m",
    "center_y_enu_m", "tile_center_y_enu_m", "center_north_enu_m", "center_northing_enu_m",
    "center_y_m", "tile_center_y_m", "y_center_m", "northing_m",
    "center_y_lks94_m", "center_y_3346_m", "y_3346_m", "y_lks94_m", "proj_center_y_m",
    "center_y", "y_center",
]

TILE_BBOX_XMIN_ALIASES = [
    "left_easting", "left_easting_m",
    "xmin_m", "x_min_m", "min_x_m", "left_m", "bbox_xmin_m", "x0_m", "xmin",
]
TILE_BBOX_XMAX_ALIASES = [
    "right_easting", "right_easting_m",
    "xmax_m", "x_max_m", "max_x_m", "right_m", "bbox_xmax_m", "x1_m", "xmax",
]
TILE_BBOX_YMIN_ALIASES = [
    "bottom_northing", "bottom_northing_m",
    "ymin_m", "y_min_m", "min_y_m", "bottom_m", "bbox_ymin_m", "y0_m", "ymin",
]
TILE_BBOX_YMAX_ALIASES = [
    "top_northing", "top_northing_m",
    "ymax_m", "y_max_m", "max_y_m", "top_m", "bbox_ymax_m", "y1_m", "ymax",
]

@dataclass
class ResolvedPaths:
    output_root: Path
    query_csv: Path
    topk_csv: Path
    tile_index_csv: Path
    out_root: Path


def resolve_query_csv(config: Dict[str, Any], repo_root: Path, output_root: Path, cli_path: Optional[str]) -> Path:
    provided = resolve_optional_path(cli_path, repo_root)
    if provided:
        return provided

    candidates: List[Path] = []
    # Config paths first.
    for p in all_existing_config_paths(config, repo_root):
        name = p.name.lower()
        if p.suffix.lower() == ".csv" and ("uav" in name or "query" in name or "trajectory" in name):
            candidates.append(p)

    patterns = [
        "metadata/s8_5_uav_frames_index*.csv",
        "metadata/*uav_frames_index*.csv",
        "metadata/s8_10b_canonical_uav_query_manifest*.csv",
        "metadata/*canonical*uav*query*.csv",
        "trajectories/s8_3_reference_trajectory*.csv",
    ]
    for pat in patterns:
        candidates.extend(output_root.glob(pat))
    candidates = sorted(set(candidates))

    def checker(head: pd.DataFrame) -> Tuple[bool, str, int]:
        has_id = has_any(head, QUERY_ID_ALIASES)
        has_alt = has_any(head, ALT_ALIASES)
        has_yaw = has_any(head, YAW_ALIASES)
        has_pitch = has_any(head, PITCH_ALIASES)
        has_body = has_any(head, BODY_X_ALIASES + PROJ_X_ALIASES) and has_any(head, BODY_Y_ALIASES + PROJ_Y_ALIASES)
        ok = has_id and has_alt and has_yaw and has_pitch and has_body
        bonus = 0
        n = " ".join(map(str.lower, map(str, head.columns)))
        if "x_enu" in n and "y_enu" in n:
            bonus += 10
        if "s8_5" in str(head).lower():
            bonus += 2
        return ok, "query schema", bonus

    if candidates:
        try:
            return choose_csv_by_schema(candidates, "query/body telemetry", checker)
        except SystemExit:
            pass

    die(
        "Could not auto-resolve query/body telemetry CSV. Pass --query-manifest-csv.\n"
        "Expected something like outputs/villoc/45_deg/metadata/s8_5_uav_frames_index_v_1fps.csv"
    )


def resolve_tile_index_csv(config: Dict[str, Any], repo_root: Path, output_root: Path, variant: str, cli_path: Optional[str]) -> Path:
    provided = resolve_optional_path(cli_path, repo_root)
    if provided:
        return provided

    candidates: List[Path] = []
    for p in all_existing_config_paths(config, repo_root):
        name = p.name.lower()
        if p.suffix.lower() == ".csv" and "tile" in name and variant.lower() in name:
            candidates.append(p)

    # Current output root, then same map reused from 90_deg if this is 45_deg.
    roots = [output_root]
    text = str(output_root)
    for src, dst in [("45_deg", "90_deg"), ("45deg", "90deg")]:
        if src in text:
            roots.append(Path(text.replace(src, dst)))

    for root in roots:
        candidates.extend(root.glob(f"metadata/s8_9_satellite_tile_index_{variant}.csv"))
        candidates.extend(root.glob(f"metadata/*tile*index*{variant}*.csv"))
        candidates.extend(root.glob(f"**/*tile*index*{variant}*.csv"))

    candidates = sorted(set(p.resolve() for p in candidates if p.exists()))

    def checker(head: pd.DataFrame) -> Tuple[bool, str, int]:
        has_tile = has_any(head, TILE_ID_ALIASES)
        has_center = has_any(head, TILE_CENTER_X_ALIASES) and has_any(head, TILE_CENTER_Y_ALIASES)
        has_bbox = (
            has_any(head, TILE_BBOX_XMIN_ALIASES) and has_any(head, TILE_BBOX_XMAX_ALIASES)
            and has_any(head, TILE_BBOX_YMIN_ALIASES) and has_any(head, TILE_BBOX_YMAX_ALIASES)
        )
        ok = has_tile and (has_center or has_bbox)
        bonus = 0
        joined = " ".join(map(str.lower, map(str, head.columns)))
        if "center" in joined:
            bonus += 8
        return ok, "tile index schema", bonus

    if candidates:
        return choose_csv_by_schema(candidates, "tile index", checker)

    die(
        "Could not auto-resolve tile index CSV. Pass --tile-index-csv.\n"
        f"Expected something like outputs/villoc/90_deg/metadata/s8_9_satellite_tile_index_{variant}.csv"
    )


def resolve_topk_csv(config: Dict[str, Any], repo_root: Path, output_root: Path, variant: str, tag: str, cli_path: Optional[str]) -> Path:
    provided = resolve_optional_path(cli_path, repo_root)
    if provided:
        return provided

    candidates: List[Path] = []
    for p in all_existing_config_paths(config, repo_root):
        name = p.name.lower()
        if p.suffix.lower() == ".csv" and variant.lower() in name and ("retrieval" in name or "top" in name):
            candidates.append(p)

    search_roots = [output_root]
    for root in search_roots:
        candidates.extend(root.glob(f"**/*{variant}*.csv"))

    candidates = sorted(set(p.resolve() for p in candidates if p.exists()))

    # Coarse path-name filtering/scoring before schema read.
    must_terms = [variant]
    prefer_terms = ["topk", "top_k", "retrieval", "ranked", "s8_11d", "metadata", tag]
    avoid_terms = ["summary", "report", "diagnostics", "good_top1", "bad_top1", "all_queries_with", "comparison"]
    scored = []
    for p in candidates:
        s = score_path_candidate(p, must_terms, prefer_terms, avoid_terms)
        if s > -1000:
            scored.append((s, p))
    scored.sort(key=lambda x: x[0], reverse=True)
    candidates = [p for _, p in scored]

    def checker(head: pd.DataFrame) -> Tuple[bool, str, int]:
        has_query = has_any(head, TOPK_QUERY_ID_ALIASES)
        has_tile = has_any(head, TILE_ID_ALIASES)
        has_rank_or_multi = has_any(head, RANK_ALIASES) or len(head) > 1
        ok = has_query and has_tile and has_rank_or_multi
        bonus = 0
        cols = " ".join(map(str.lower, map(str, head.columns)))
        if "rank" in cols:
            bonus += 8
        if has_any(head, SCORE_ALIASES):
            bonus += 3
        return ok, "top-k retrieval schema", bonus

    if candidates:
        return choose_csv_by_schema(candidates, "Top-K retrieval", checker)

    die(
        "Could not auto-resolve Top-K retrieval CSV. Pass --topk-csv.\n"
        "Tip: use the raw S8.11D ranked/top-k CSV, not the S8.12D report summary CSV."
    )


def resolve_out_root(output_root: Path, variant: str, cli_out_root: Optional[str], repo_root: Path) -> Path:
    if cli_out_root:
        return normalize_path(cli_out_root, repo_root)
    return (output_root / "reports" / "s8_12e0_footprint_center_ray" / variant).resolve()


# -----------------------------------------------------------------------------
# Coordinate column selection
# -----------------------------------------------------------------------------


@dataclass
class XYPair:
    x_col: str
    y_col: str
    label: str
    median_abs: float


def make_xy_pair(df: pd.DataFrame, x_aliases: Sequence[str], y_aliases: Sequence[str], label: str) -> Optional[XYPair]:
    x = find_col(df, x_aliases, required=False)
    y = find_col(df, y_aliases, required=False)
    if not x or not y or x == y:
        return None
    xs = pd.to_numeric(df[x], errors="coerce")
    ys = pd.to_numeric(df[y], errors="coerce")
    if xs.notna().sum() == 0 or ys.notna().sum() == 0:
        return None
    med = float(np.nanmedian(np.abs(np.r_[xs.to_numpy(dtype=float), ys.to_numpy(dtype=float)])))
    return XYPair(x, y, label, med)


def collect_query_xy_pairs(df: pd.DataFrame) -> List[XYPair]:
    pairs: List[XYPair] = []
    # ENU aliases first.
    p = make_xy_pair(df, BODY_X_ALIASES, BODY_Y_ALIASES, "enu_or_body_m")
    if p:
        pairs.append(p)
    p = make_xy_pair(df, PROJ_X_ALIASES, PROJ_Y_ALIASES, "projected_or_lks94_m")
    if p:
        pairs.append(p)
    # Catch common projected columns not caught above.
    p = make_xy_pair(
        df,
        ["easting", "x_projected", "x_lks", "x_3346"],
        ["northing", "y_projected", "y_lks", "y_3346"],
        "projected_generic",
    )
    if p and all((p.x_col, p.y_col) != (q.x_col, q.y_col) for q in pairs):
        pairs.append(p)
    return pairs


def add_tile_center_if_needed(tile_df: pd.DataFrame) -> Tuple[pd.DataFrame, str, str, str]:
    cx = find_col(tile_df, TILE_CENTER_X_ALIASES, required=False)
    cy = find_col(tile_df, TILE_CENTER_Y_ALIASES, required=False)
    if cx and cy and cx != cy:
        return tile_df, cx, cy, "existing_center_columns"

    xmin = find_col(tile_df, TILE_BBOX_XMIN_ALIASES, required=False)
    xmax = find_col(tile_df, TILE_BBOX_XMAX_ALIASES, required=False)
    ymin = find_col(tile_df, TILE_BBOX_YMIN_ALIASES, required=False)
    ymax = find_col(tile_df, TILE_BBOX_YMAX_ALIASES, required=False)
    if xmin and xmax and ymin and ymax:
        out = tile_df.copy()
        for col in [xmin, xmax, ymin, ymax]:
            out[col] = pd.to_numeric(out[col], errors="coerce")
        out["__tile_center_x_m"] = (out[xmin] + out[xmax]) / 2.0
        out["__tile_center_y_m"] = (out[ymin] + out[ymax]) / 2.0
        return out, "__tile_center_x_m", "__tile_center_y_m", "computed_from_bbox"

    die(
        "Tile index lacks usable center or bbox columns.\n"
        f"Available columns: {list(tile_df.columns)}\n"
        f"Accepted center-x aliases: {TILE_CENTER_X_ALIASES}\n"
        f"Accepted center-y aliases: {TILE_CENTER_Y_ALIASES}"
    )

def add_projected_query_xy_from_latlon(query_df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[str], Optional[str]]:
    """
    Add EPSG:3346 / LKS-94 query coordinates from latitude/longitude.

    Villoc S8.5 query CSV commonly stores:
      - x_enu_m / y_enu_m in local ENU
      - latitude / longitude in EPSG:4326

    The S8.9 tile index stores:
      - center_easting / center_northing in EPSG:3346

    Therefore, for tile-center distance diagnostics, query lat/lon must be projected
    to EPSG:3346 before comparing against tile easting/northing.
    """
    lat_col = find_col(query_df, LAT_ALIASES, required=False, label="query/body telemetry")
    lon_col = find_col(query_df, LON_ALIASES, required=False, label="query/body telemetry")

    if not lat_col or not lon_col:
        return query_df, None, None

    out = query_df.copy()
    lat = pd.to_numeric(out[lat_col], errors="coerce")
    lon = pd.to_numeric(out[lon_col], errors="coerce")

    ok = lat.notna() & lon.notna()
    if ok.sum() == 0:
        return out, None, None

    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3346", always_xy=True)
    east = np.full(len(out), np.nan, dtype=float)
    north = np.full(len(out), np.nan, dtype=float)

    east_ok, north_ok = transformer.transform(
        lon.loc[ok].to_numpy(dtype=float),
        lat.loc[ok].to_numpy(dtype=float),
    )

    east[ok.to_numpy()] = east_ok
    north[ok.to_numpy()] = north_ok

    out["s8_12e0_query_easting_3346_m"] = east
    out["s8_12e0_query_northing_3346_m"] = north

    return out, "s8_12e0_query_easting_3346_m", "s8_12e0_query_northing_3346_m"


def choose_compatible_query_xy(query_df: pd.DataFrame, tile_df: pd.DataFrame, tile_x: str, tile_y: str) -> XYPair:
    pairs = collect_query_xy_pairs(query_df)
    if not pairs:
        die(
            "Could not find body/query coordinate columns.\n"
            f"Available query columns: {list(query_df.columns)}\n"
            f"Accepted x aliases include: {BODY_X_ALIASES + PROJ_X_ALIASES}\n"
            f"Accepted y aliases include: {BODY_Y_ALIASES + PROJ_Y_ALIASES}"
        )

    tile_vals = np.r_[
        pd.to_numeric(tile_df[tile_x], errors="coerce").to_numpy(dtype=float),
        pd.to_numeric(tile_df[tile_y], errors="coerce").to_numpy(dtype=float),
    ]
    tile_med_abs = float(np.nanmedian(np.abs(tile_vals)))

    def frame_penalty(q: XYPair) -> float:
        # Same order of magnitude is important. ENU is usually < 10^4, LKS94 northing ~ 10^6.
        qv = max(q.median_abs, 1.0)
        tv = max(tile_med_abs, 1.0)
        p = abs(math.log10(tv / qv))
        qname = f"{q.x_col} {q.y_col}".lower()
        tname = f"{tile_x} {tile_y}".lower()
        if "enu" in qname and "enu" in tname:
            p -= 1.0
        if any(k in qname for k in ["3346", "lks", "project", "easting", "northing"]):
            if any(k in tname for k in ["3346", "lks", "project", "easting", "northing"]):
                p -= 1.0
        return p

    ranked = sorted(((frame_penalty(p), p) for p in pairs), key=lambda x: x[0])
    chosen = ranked[0][1]

    if ranked[0][0] > 3.0:
        details = "\n".join(
            f"  query {p.label}: {p.x_col}, {p.y_col}, median_abs={p.median_abs:.3f}, penalty={pen:.3f}"
            for pen, p in ranked
        )
        die(
            "Query coordinates and tile coordinates appear to be in incompatible frames.\n"
            f"Tile columns: {tile_x}, {tile_y}, median_abs={tile_med_abs:.3f}\n"
            f"Query candidates:\n{details}\n"
            "Fix: pass a query/trajectory CSV that contains coordinates in the same frame as the tile index."
        )
    return chosen


# -----------------------------------------------------------------------------
# Core diagnostic logic
# -----------------------------------------------------------------------------


def normalize_query_ids(df: pd.DataFrame, aliases: Sequence[str], label: str, out_col: str = "__query_id") -> Tuple[pd.DataFrame, str]:
    col = find_col(df, aliases, required=True, label=label)
    out = df.copy()
    out[out_col] = pd.to_numeric(out[col], errors="coerce")
    if out[out_col].isna().any():
        # Fall back to string IDs if numeric conversion fails.
        out[out_col] = out[col].astype(str)
    else:
        out[out_col] = out[out_col].astype("Int64")
    return out, col


def normalize_tile_ids(df: pd.DataFrame, label: str, out_col: str = "__tile_id") -> Tuple[pd.DataFrame, str]:
    col = find_col(df, TILE_ID_ALIASES, required=True, label=label)
    out = df.copy()
    nums = pd.to_numeric(out[col], errors="coerce")
    if nums.isna().any():
        out[out_col] = out[col].astype(str)
    else:
        out[out_col] = nums.astype("Int64")
    return out, col


def prepare_topk(topk_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Optional[str]]]:
    df, q_col = normalize_query_ids(topk_df, TOPK_QUERY_ID_ALIASES, "Top-K retrieval", "__query_id")
    df, t_col = normalize_tile_ids(df, "Top-K retrieval", "__tile_id")

    rank_col = find_col(df, RANK_ALIASES, required=False, label="Top-K retrieval")
    if rank_col:
        df["__rank"] = pd.to_numeric(df[rank_col], errors="coerce")
    else:
        info("WARNING: Top-K CSV has no explicit rank column. Using row order within each query as rank.")
        df["__rank"] = df.groupby("__query_id").cumcount() + 1
        rank_col = "__rank"

    score_col = find_col(df, SCORE_ALIASES, required=False, label="Top-K retrieval")
    if score_col:
        df["__score"] = pd.to_numeric(df[score_col], errors="coerce")
    else:
        df["__score"] = np.nan

    df = df.dropna(subset=["__rank"]).copy()
    df["__rank"] = df["__rank"].astype(int)
    df = df.sort_values(["__query_id", "__rank"]).reset_index(drop=True)
    return df, {"query_id_col": q_col, "tile_id_col": t_col, "rank_col": rank_col, "score_col": score_col}


def compute_footprint(
    merged: pd.DataFrame,
    alt_col: str,
    yaw_col: str,
    pitch_col: str,
    body_x_col: str,
    body_y_col: str,
    tile_x_col: str,
    tile_y_col: str,
    yaw_offset_deg: float,
    yaw_sign: float,
    forward_scale: float,
    min_abs_pitch_deg: float,
    max_abs_pitch_deg: float,
) -> pd.DataFrame:
    out = merged.copy()

    for col in [alt_col, yaw_col, pitch_col, body_x_col, body_y_col, tile_x_col, tile_y_col]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    before = len(out)
    out = out.dropna(subset=[alt_col, yaw_col, pitch_col, body_x_col, body_y_col, tile_x_col, tile_y_col]).copy()
    dropped = before - len(out)
    if dropped:
        info(f"WARNING: dropped {dropped} merged rows with missing numeric geometry.")

    pitch_abs = np.abs(out[pitch_col].astype(float).to_numpy())
    pitch_abs = np.clip(pitch_abs, min_abs_pitch_deg, max_abs_pitch_deg)
    pitch_rad = np.deg2rad(pitch_abs)

    rel_alt = out[alt_col].astype(float).to_numpy()
    forward_offset = forward_scale * rel_alt / np.tan(pitch_rad)

    yaw_rad = np.deg2rad(yaw_sign * out[yaw_col].astype(float).to_numpy() + yaw_offset_deg)
    offset_x = forward_offset * np.sin(yaw_rad)  # ENU/LKS x=east
    offset_y = forward_offset * np.cos(yaw_rad)  # ENU/LKS y=north

    body_x = out[body_x_col].astype(float).to_numpy()
    body_y = out[body_y_col].astype(float).to_numpy()
    tile_x = out[tile_x_col].astype(float).to_numpy()
    tile_y = out[tile_y_col].astype(float).to_numpy()

    footprint_x = body_x + offset_x
    footprint_y = body_y + offset_y
    corrected_body_x = tile_x - offset_x
    corrected_body_y = tile_y - offset_y

    body_error = np.hypot(tile_x - body_x, tile_y - body_y)
    footprint_error = np.hypot(tile_x - footprint_x, tile_y - footprint_y)
    corrected_body_error = np.hypot(corrected_body_x - body_x, corrected_body_y - body_y)

    out["s8_12e0_pitch_abs_deg_used"] = pitch_abs
    out["s8_12e0_forward_offset_m"] = forward_offset
    out["s8_12e0_offset_x_east_m"] = offset_x
    out["s8_12e0_offset_y_north_m"] = offset_y
    out["s8_12e0_footprint_x_m"] = footprint_x
    out["s8_12e0_footprint_y_m"] = footprint_y
    out["s8_12e0_corrected_body_x_m"] = corrected_body_x
    out["s8_12e0_corrected_body_y_m"] = corrected_body_y
    out["s8_12e0_body_topk_error_m"] = body_error
    out["s8_12e0_footprint_topk_error_m"] = footprint_error
    out["s8_12e0_corrected_body_error_m"] = corrected_body_error
    out["s8_12e0_footprint_improvement_m"] = body_error - footprint_error
    out["s8_12e0_corrected_body_improvement_m"] = body_error - corrected_body_error

    return out


def summarize_errors(top1: pd.DataFrame, all_ranked: pd.DataFrame, args: argparse.Namespace, paths: ResolvedPaths, columns: Dict[str, Any]) -> Dict[str, Any]:
    def stats(series: pd.Series) -> Dict[str, Any]:
        s = pd.to_numeric(series, errors="coerce").dropna()
        if len(s) == 0:
            return {"count": 0}
        return {
            "count": int(len(s)),
            "mean_m": float(s.mean()),
            "median_m": float(s.median()),
            "p75_m": float(s.quantile(0.75)),
            "p95_m": float(s.quantile(0.95)),
            "min_m": float(s.min()),
            "max_m": float(s.max()),
        }

    threshold = float(args.hit_threshold_m)
    summary = {
        "stage": "S8.12E.0_footprint_center_ray_diagnostic",
        "status": "PASS_FOOTPRINT_CENTER_RAY_DIAGNOSTIC",
        "leakage_rule": "reference/body/footprint/tile geometry used only after image-only retrieval ranking",
        "config": str(args.config),
        "variant": args.variant,
        "tag": args.tag,
        "oracle_k_requested": int(args.oracle_k),
        "hit_threshold_m": threshold,
        "yaw_offset_deg": float(args.yaw_offset_deg),
        "yaw_sign": float(args.yaw_sign),
        "forward_scale": float(args.forward_scale),
        "paths": {
            "output_root": str(paths.output_root),
            "query_csv": str(paths.query_csv),
            "topk_csv": str(paths.topk_csv),
            "tile_index_csv": str(paths.tile_index_csv),
            "out_root": str(paths.out_root),
        },
        "columns": columns,
        "counts": {
            "ranked_rows": int(len(all_ranked)),
            "top1_rows": int(len(top1)),
            "unique_queries_in_ranked": int(all_ranked["__query_id"].nunique()),
            "unique_queries_top1": int(top1["__query_id"].nunique()),
        },
        "top1_error_stats": {
            "body_error": stats(top1["s8_12e0_body_topk_error_m"]),
            "footprint_error": stats(top1["s8_12e0_footprint_topk_error_m"]),
            "corrected_body_error": stats(top1["s8_12e0_corrected_body_error_m"]),
            "body_minus_footprint_improvement": stats(top1["s8_12e0_footprint_improvement_m"]),
        },
        "top1_hit_counts": {
            "body_le_threshold": int((top1["s8_12e0_body_topk_error_m"] <= threshold).sum()),
            "footprint_le_threshold": int((top1["s8_12e0_footprint_topk_error_m"] <= threshold).sum()),
            "corrected_body_le_threshold": int((top1["s8_12e0_corrected_body_error_m"] <= threshold).sum()),
            "footprint_improved_vs_body": int((top1["s8_12e0_footprint_improvement_m"] > 0).sum()),
            "footprint_worse_vs_body": int((top1["s8_12e0_footprint_improvement_m"] < 0).sum()),
        },
        "note": (
            "footprint_error and corrected_body_error are mathematically equivalent here: "
            "distance(tile_center, body+offset) == distance(tile_center-offset, body). "
            "Both are saved because they answer two different interpretations."
        ),
    }
    return summary


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------


def save_error_boxplot(top1: pd.DataFrame, fig_dir: Path) -> None:
    data = [
        top1["s8_12e0_body_topk_error_m"].dropna().to_numpy(),
        top1["s8_12e0_footprint_topk_error_m"].dropna().to_numpy(),
        top1["s8_12e0_corrected_body_error_m"].dropna().to_numpy(),
    ]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(data, labels=["body", "footprint", "corrected body"], showfliers=False)
    ax.set_ylabel("Top-1 error [m]")
    ax.set_title("S8.12E.0 Top-1 body vs footprint error")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "s8_12e0_top1_error_boxplot.png", dpi=180)
    plt.close(fig)


def save_offset_improvement_scatter(top1: pd.DataFrame, fig_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(
        top1["s8_12e0_forward_offset_m"],
        top1["s8_12e0_footprint_improvement_m"],
        s=24,
        alpha=0.75,
    )
    ax.axhline(0.0, linestyle="--", linewidth=1)
    ax.set_xlabel("Estimated forward footprint offset [m]")
    ax.set_ylabel("Body error - footprint error [m]")
    ax.set_title("Positive values mean footprint evaluation improved")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "s8_12e0_offset_vs_footprint_improvement.png", dpi=180)
    plt.close(fig)


def save_corrected_body_scatter(top1: pd.DataFrame, body_x: str, body_y: str, tile_x: str, tile_y: str, fig_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(top1[body_x], top1[body_y], s=18, alpha=0.8, label="body/reference eval")
    ax.scatter(top1[tile_x], top1[tile_y], s=18, alpha=0.55, label="retrieved tile center")
    ax.scatter(
        top1["s8_12e0_corrected_body_x_m"],
        top1["s8_12e0_corrected_body_y_m"],
        s=18,
        alpha=0.55,
        label="tile center back-shifted",
    )
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X / Easting [m]")
    ax.set_ylabel("Y / Northing [m]")
    ax.set_title("S8.12E.0 corrected body diagnostic")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "s8_12e0_corrected_body_scatter.png", dpi=180)
    plt.close(fig)


# -----------------------------------------------------------------------------
# CLI / main
# -----------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="S8.12E.0 footprint-center-ray diagnostic for Villoc oblique retrieval.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", required=True, type=Path, help="Dataset YAML config, e.g. configs/dataset_villoc_45deg.yaml")
    parser.add_argument("--variant", required=True, help="Tile/database variant, e.g. 1024_s512")
    parser.add_argument("--tag", required=True, help="Descriptor/retrieval tag used in S8.11D outputs")
    parser.add_argument("--oracle-k", type=int, default=20, help="Maximum rank depth to keep for diagnostic outputs")

    parser.add_argument("--output-root", default=None, help="Override config output_root")
    parser.add_argument("--query-manifest-csv", default=None, help="Override query/body telemetry CSV")
    parser.add_argument("--topk-csv", default=None, help="Override S8.11D ranked Top-K retrieval CSV")
    parser.add_argument("--tile-index-csv", default=None, help="Override satellite tile index CSV")
    parser.add_argument("--out-root", default=None, help="Override output folder")

    parser.add_argument("--yaw-offset-deg", type=float, default=0.0, help="Diagnostic yaw offset added before projecting the forward footprint")
    parser.add_argument("--yaw-sign", type=float, choices=[-1.0, 1.0], default=1.0, help="Use +1 if yaw is clockwise from north; -1 for the opposite convention")
    parser.add_argument("--forward-scale", type=float, default=1.0, help="Optional diagnostic multiplier on the estimated forward offset")
    parser.add_argument("--min-abs-pitch-deg", type=float, default=1.0, help="Lower clamp for abs(pitch) to avoid tan(0)")
    parser.add_argument("--max-abs-pitch-deg", type=float, default=89.9, help="Upper clamp for abs(pitch) to avoid tan(90) singularity")
    parser.add_argument("--hit-threshold-m", type=float, default=40.0, help="Evaluation-only hit threshold")
    parser.add_argument("--no-figures", action="store_true", help="Skip PNG plot generation")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    repo_root = Path.cwd().resolve()
    config_path = normalize_path(args.config, repo_root)
    config = load_yaml(config_path)

    output_root = resolve_output_root(config, repo_root, args.output_root)
    query_csv = resolve_query_csv(config, repo_root, output_root, args.query_manifest_csv)
    topk_csv = resolve_topk_csv(config, repo_root, output_root, args.variant, args.tag, args.topk_csv)
    tile_index_csv = resolve_tile_index_csv(config, repo_root, output_root, args.variant, args.tile_index_csv)
    out_root = resolve_out_root(output_root, args.variant, args.out_root, repo_root)
    fig_dir = out_root / "figures"
    out_root.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    paths = ResolvedPaths(
        output_root=output_root,
        query_csv=query_csv,
        topk_csv=topk_csv,
        tile_index_csv=tile_index_csv,
        out_root=out_root,
    )

    info("S8.12E.0 Footprint-center-ray diagnostic")
    info("------------------------------------------------")
    info(f"config:     {config_path}")
    info(f"output root:{output_root}")
    info(f"query csv:  {query_csv}")
    info(f"topk csv:   {topk_csv}")
    info(f"tile index: {tile_index_csv}")
    info(f"out root:   {out_root}")

    query_df = read_csv_checked(query_csv, "query/body telemetry CSV")
    topk_df_raw = read_csv_checked(topk_csv, "Top-K retrieval CSV")
    tile_df_raw = read_csv_checked(tile_index_csv, "tile index CSV")

    topk_df, topk_cols = prepare_topk(topk_df_raw)
    query_df, query_id_col = normalize_query_ids(query_df, QUERY_ID_ALIASES, "query/body telemetry", "__query_id")
    tile_df, tile_id_col = normalize_tile_ids(tile_df_raw, "tile index", "__tile_id")
    tile_df, tile_x_col, tile_y_col, tile_center_source = add_tile_center_if_needed(tile_df)

    query_df, projected_x_added, projected_y_added = add_projected_query_xy_from_latlon(query_df)
    if projected_x_added and projected_y_added:
        info(
            "Added projected query coordinates from lat/lon: "
            f"{projected_x_added}, {projected_y_added} [EPSG:3346]"
        )

    xy_pair = choose_compatible_query_xy(query_df, tile_df, tile_x_col, tile_y_col)
    body_x_col, body_y_col = xy_pair.x_col, xy_pair.y_col

    alt_col = find_col(query_df, ALT_ALIASES, required=True, label="query/body telemetry")
    yaw_col = find_col(query_df, YAW_ALIASES, required=True, label="query/body telemetry")
    pitch_col = find_col(query_df, PITCH_ALIASES, required=True, label="query/body telemetry")

    # Keep only the requested depth, but do not change candidate ordering.
    topk_df = topk_df[topk_df["__rank"] <= int(args.oracle_k)].copy()
    if topk_df.empty:
        die(f"No Top-K rows remain after applying --oracle-k {args.oracle_k}")

    # Avoid accidental duplicate query metadata rows. Keep first after sorting by query ID.
    query_cols_needed = ["__query_id", query_id_col, body_x_col, body_y_col, alt_col, yaw_col, pitch_col]
    query_extra_cols = [c for c in ["image_path", "canonical_query_filename", "timestamp_s", "sample_id", "token0_id"] if c in query_df.columns]
    query_keep = list(dict.fromkeys(query_cols_needed + query_extra_cols))
    query_small = query_df[query_keep].drop_duplicates("__query_id", keep="first").copy()
    query_small["__query_join_ok"] = 1

    tile_keep = ["__tile_id", tile_id_col, tile_x_col, tile_y_col]
    tile_extra_cols = [c for c in ["image_path", "tile_path", "tile_image_path", "filename", "tile_filename"] if c in tile_df.columns]
    tile_keep = list(dict.fromkeys(tile_keep + tile_extra_cols))
    tile_small = tile_df[tile_keep].drop_duplicates("__tile_id", keep="first").copy()
    tile_small["__tile_join_ok"] = 1

    merged = topk_df.merge(query_small, on="__query_id", how="left", suffixes=("", "_query"))
    missing_q = int(merged["__query_join_ok"].isna().sum())
    if missing_q:
        missing_ids = merged.loc[merged["__query_join_ok"].isna(), "__query_id"].drop_duplicates().head(20).tolist()
        die(
            f"Top-K rows could not be joined to query telemetry for {missing_q} rows. "
            f"Example missing query IDs: {missing_ids}\n"
            f"Top-K query column: {topk_cols['query_id_col']} ; Query CSV ID column: {query_id_col}"
        )

    merged = merged.merge(tile_small, on="__tile_id", how="left", suffixes=("", "_tile"))
    missing_t = int(merged["__tile_join_ok"].isna().sum())
    if missing_t:
        missing_ids = merged.loc[merged["__tile_join_ok"].isna(), "__tile_id"].drop_duplicates().head(20).tolist()
        die(
            f"Top-K rows could not be joined to tile index for {missing_t} rows. "
            f"Example missing tile IDs: {missing_ids}\n"
            f"Top-K tile column: {topk_cols['tile_id_col']} ; Tile index ID column: {tile_id_col}"
        )

    all_ranked = compute_footprint(
        merged=merged,
        alt_col=alt_col,
        yaw_col=yaw_col,
        pitch_col=pitch_col,
        body_x_col=body_x_col,
        body_y_col=body_y_col,
        tile_x_col=tile_x_col,
        tile_y_col=tile_y_col,
        yaw_offset_deg=float(args.yaw_offset_deg),
        yaw_sign=float(args.yaw_sign),
        forward_scale=float(args.forward_scale),
        min_abs_pitch_deg=float(args.min_abs_pitch_deg),
        max_abs_pitch_deg=float(args.max_abs_pitch_deg),
    )

    all_ranked = all_ranked.sort_values(["__query_id", "__rank"]).reset_index(drop=True)
    top1 = all_ranked.sort_values(["__query_id", "__rank"]).groupby("__query_id", as_index=False).head(1).copy()

    columns = {
        "topk": topk_cols,
        "query_id_col": query_id_col,
        "tile_id_col": tile_id_col,
        "body_x_col": body_x_col,
        "body_y_col": body_y_col,
        "altitude_col": alt_col,
        "yaw_col": yaw_col,
        "pitch_col": pitch_col,
        "tile_center_x_col": tile_x_col,
        "tile_center_y_col": tile_y_col,
        "tile_center_source": tile_center_source,
        "coordinate_pair_label": xy_pair.label,
    }

    summary = summarize_errors(top1, all_ranked, args, paths, columns)

    ranked_out = out_root / "s8_12e0_all_ranked_body_vs_footprint.csv"
    top1_out = out_root / "s8_12e0_top1_body_vs_footprint.csv"
    summary_out = out_root / "s8_12e0_footprint_center_ray_summary.json"

    all_ranked.to_csv(ranked_out, index=False)
    top1.to_csv(top1_out, index=False)
    with summary_out.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if not args.no_figures:
        save_error_boxplot(top1, fig_dir)
        save_offset_improvement_scatter(top1, fig_dir)
        save_corrected_body_scatter(top1, body_x_col, body_y_col, tile_x_col, tile_y_col, fig_dir)

    body_med = summary["top1_error_stats"]["body_error"].get("median_m")
    fp_med = summary["top1_error_stats"]["footprint_error"].get("median_m")
    corr_med = summary["top1_error_stats"]["corrected_body_error"].get("median_m")
    wins = summary["top1_hit_counts"]["footprint_improved_vs_body"

    info("\nResolved columns")
    info("----------------")
    for k, v in columns.items():
        info(f"{k}: {v}")

    info("\nTop-1 diagnostic")
    info("----------------")
    info(f"queries:                 {len(top1)}")
    info(f"body median error:       {body_med:.3f} m" if body_med is not None else "body median error:       n/a")
    info(f"footprint median error:  {fp_med:.3f} m" if fp_med is not None else "footprint median error:  n/a")
    info(f"corrected-body median:   {corr_med:.3f} m" if corr_med is not None else "corrected-body median:   n/a")
    info(f"footprint improved rows: {wins}/{len(top1)}")

    info("\nOutputs")
    info("-------")
    info(f"ranked csv: {ranked_out}")
    info(f"top1 csv:   {top1_out}")
    info(f"summary:    {summary_out}")
    if not args.no_figures:
        info(f"figures:    {fig_dir}")
    info("\nSTATUS: PASS_FOOTPRINT_CENTER_RAY_DIAGNOSTIC")


if __name__ == "__main__":
    main()
