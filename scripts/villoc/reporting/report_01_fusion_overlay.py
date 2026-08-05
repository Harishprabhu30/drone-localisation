'''
Run Command:

python scripts/villoc/reporting/report_01_fusion_overlay.py \
  --config configs/reporting/report_villoc_traj01_s8_figures.yaml \
  2>&1 | tee outputs/villoc/traj01_90deg_stable120m/reporting/logs/report_01_fusion_overlay.log

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
# Helpers
# ---------------------------------------------------------------------

def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_csv(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{label}: missing file: {path}")
    df = pd.read_csv(path)
    if len(df) == 0:
        raise ValueError(f"{label}: empty CSV: {path}")
    return df


def norm_col(c: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(c).lower()).strip("_")


def find_first_existing_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    normalized = {norm_col(c): c for c in df.columns}
    for cand in candidates:
        key = norm_col(cand)
        if key in normalized:
            return normalized[key]
    return None


def find_col_by_patterns(
    df: pd.DataFrame,
    include_any: List[str],
    exclude_any: Optional[List[str]] = None,
) -> Optional[str]:
    exclude_any = exclude_any or []
    for c in df.columns:
        nc = norm_col(c)
        if any(p in nc for p in include_any) and not any(e in nc for e in exclude_any):
            return c
    return None


def choose_xy_cols(df: pd.DataFrame, role: str) -> Tuple[str, str, str]:
    """
    Return x_col, y_col, reason.

    role:
      reference
      estimated
      fusion
    """
    if role == "reference":
        pairs = [
            ("x_enu_m", "y_enu_m"),
            ("reference_x_m", "reference_y_m"),
            ("ref_x_m", "ref_y_m"),
            ("gt_x_m", "gt_y_m"),
            ("x_ref_m", "y_ref_m"),
            ("x_m", "y_m"),
        ]
    elif role == "fusion":
        pairs = [
            ("fused_x_m", "fused_y_m"),
            ("x_fused_m", "y_fused_m"),
            ("estimated_x_m", "estimated_y_m"),
            ("est_x_m", "est_y_m"),
            ("x_est_m", "y_est_m"),
            ("trajectory_x_m", "trajectory_y_m"),
            ("x_m", "y_m"),
        ]
    else:
        pairs = [
            ("estimated_x_m", "estimated_y_m"),
            ("est_x_m", "est_y_m"),
            ("x_est_m", "y_est_m"),
            ("aligned_x_m", "aligned_y_m"),
            ("x_aligned_m", "y_aligned_m"),
            ("trajectory_x_m", "trajectory_y_m"),
            ("x_m", "y_m"),
        ]

    for x, y in pairs:
        xc = find_first_existing_col(df, [x])
        yc = find_first_existing_col(df, [y])
        if xc and yc:
            return xc, yc, f"matched explicit pair {x}/{y}"

    # Fallback: choose numeric x/y-like columns, avoiding obvious reference columns for estimated/fusion.
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    x_candidates = []
    y_candidates = []

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

    raise ValueError(
        f"Could not infer XY columns for role={role}. Columns are:\n{list(df.columns)}"
    )


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


def compute_cumulative_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    dx = np.diff(x, prepend=x[0])
    dy = np.diff(y, prepend=y[0])
    step = np.sqrt(dx * dx + dy * dy)
    step[0] = 0.0
    return np.cumsum(step)


def filter_by_method_and_alpha(
    df: pd.DataFrame,
    method_name: str,
    alpha: Optional[float],
    label: str,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Fusion tables can contain many policies stacked together.
    This function searches all string/object columns for the method name,
    then optionally filters an alpha column.
    """
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
        # Try softer matching by tokens.
        tokens = [t for t in re.split(r"[^a-zA-Z0-9]+", method_l) if len(t) >= 3]
        best_col = None
        best_mask = None
        best_count = 0

        for c in string_cols:
            s = out[c].astype(str).str.lower()
            mask = pd.Series(True, index=out.index)
            for t in tokens[:5]:
                mask = mask & s.str.contains(re.escape(t), na=False)
            count = int(mask.sum())
            if count > best_count:
                best_col, best_mask, best_count = c, mask, count

        if best_col is not None and best_mask is not None and best_count > 0:
            out = out.loc[best_mask].copy()
            info["method_filter_column"] = best_col
            info["fallback"] = "soft token method match"
        else:
            info["fallback"] = "no method filter matched; using full table"

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
    """
    If trajectory CSV has repeated rows due to multiple policies/events,
    collapse to one row per query using identity column if possible.
    """
    info: Dict[str, Any] = {"label": label, "input_rows": int(len(df)), "identity_col": None}

    ident = choose_identity_col(df)
    if ident is None:
        out = df.reset_index(drop=True)
        info["output_rows"] = int(len(out))
        info["note"] = "no identity column found; no collapse"
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


def extract_lat_lon_reference(df_ref: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    lat = find_first_existing_col(df_ref, ["lat", "latitude", "reference_latitude"])
    lon = find_first_existing_col(df_ref, ["lon", "longitude", "reference_longitude"])
    return lat, lon


def enu_to_latlon_approx(
    x: np.ndarray,
    y: np.ndarray,
    x0: float,
    y0: float,
    lat0: float,
    lon0: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Small-area approximate ENU -> WGS84 conversion.
    Good enough for Folium visualization over a short trajectory.
    """
    r = 6378137.0
    lat0_rad = math.radians(lat0)
    d_north = y - y0
    d_east = x - x0

    lat = lat0 + (d_north / r) * (180.0 / math.pi)
    lon = lon0 + (d_east / (r * math.cos(lat0_rad))) * (180.0 / math.pi)
    return lat, lon


def add_distance_and_error(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    ref_xy: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> pd.DataFrame:
    out = df.copy()
    x = out[x_col].astype(float).to_numpy()
    y = out[y_col].astype(float).to_numpy()

    dist_col = choose_distance_col(out)
    if dist_col is None:
        out["_report_distance_m"] = compute_cumulative_distance(x, y)
    else:
        out["_report_distance_m"] = out[dist_col].astype(float)

    err_col = choose_error_col(out)
    if err_col is not None:
        out["_report_error_m"] = out[err_col].astype(float)
    elif ref_xy is not None and len(ref_xy[0]) == len(out):
        rx, ry = ref_xy
        out["_report_error_m"] = np.sqrt((x - rx) ** 2 + (y - ry) ** 2)
    else:
        out["_report_error_m"] = np.nan

    return out


def find_event_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Best-effort correction event extraction.
    Returns rows where an accepted/correction/event-like column appears true.
    """
    if len(df) == 0:
        return df.copy()

    cols = list(df.columns)
    event_mask = pd.Series(False, index=df.index)

    for c in cols:
        nc = norm_col(c)
        if any(k in nc for k in ["accepted", "correction_applied", "is_correction", "event"]):
            s = df[c]
            if pd.api.types.is_bool_dtype(s):
                event_mask = event_mask | s.fillna(False)
            elif pd.api.types.is_numeric_dtype(s):
                event_mask = event_mask | (s.fillna(0).astype(float) > 0)
            else:
                sl = s.astype(str).str.lower()
                event_mask = event_mask | sl.isin(["true", "1", "yes", "accepted", "accept", "applied"])

    return df.loc[event_mask].copy()


def write_schema_manifest(path: Path, manifest: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------

def plot_xy_overlay(
    ref: pd.DataFrame,
    ref_xy: Tuple[str, str],
    xfeat: pd.DataFrame,
    xfeat_xy: Tuple[str, str],
    periodic: pd.DataFrame,
    periodic_xy: Tuple[str, str],
    temporal: pd.DataFrame,
    temporal_xy: Tuple[str, str],
    periodic_events: pd.DataFrame,
    temporal_events: pd.DataFrame,
    out_png: Path,
) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10.5, 9.0))

    ax.plot(
        ref[ref_xy[0]], ref[ref_xy[1]],
        color="black", linewidth=1.5, label="Reference trajectory"
    )
    ax.plot(
        xfeat[xfeat_xy[0]], xfeat[xfeat_xy[1]],
        color="orangered", linewidth=2.0, label="XFeat relative-only"
    )
    ax.plot(
        periodic[periodic_xy[0]], periodic[periodic_xy[1]],
        color="royalblue", linewidth=2.2, label="Periodic fusion"
    )
    ax.plot(
        temporal[temporal_xy[0]], temporal[temporal_xy[1]],
        color="seagreen", linewidth=2.2, label="Temporal-consistency fusion"
    )

    # Event markers
    if len(periodic_events) > 0:
        ax.scatter(
            periodic_events[periodic_xy[0]], periodic_events[periodic_xy[1]],
            s=34, marker="o", facecolors="none", edgecolors="royalblue",
            linewidths=1.2, label="Periodic accepted corrections"
        )

    if len(temporal_events) > 0:
        ax.scatter(
            temporal_events[temporal_xy[0]], temporal_events[temporal_xy[1]],
            s=42, marker="o", facecolors="none", edgecolors="seagreen",
            linewidths=1.4, label="Temporal accepted corrections"
        )

    # Start/end markers from reference
    ax.scatter(ref[ref_xy[0]].iloc[0], ref[ref_xy[1]].iloc[0], s=80, marker="*", color="black", label="Start")
    ax.scatter(ref[ref_xy[0]].iloc[-1], ref[ref_xy[1]].iloc[-1], s=70, marker="s", color="black", label="End")

    ax.set_title("Final trajectory comparison on Villoc traj01", fontsize=15, fontweight="bold")
    ax.set_xlabel("Local X / East [m]")
    ax.set_ylabel("Local Y / North [m]")
    ax.axis("equal")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def plot_error_vs_distance(
    xfeat: pd.DataFrame,
    periodic: pd.DataFrame,
    temporal: pd.DataFrame,
    periodic_events: pd.DataFrame,
    temporal_events: pd.DataFrame,
    out_png: Path,
) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11.5, 6.5))

    ax.plot(
        xfeat["_report_distance_m"], xfeat["_report_error_m"],
        color="orangered", linewidth=2.0, label="XFeat relative-only"
    )
    ax.plot(
        periodic["_report_distance_m"], periodic["_report_error_m"],
        color="royalblue", linewidth=2.0, label="Periodic fusion"
    )
    ax.plot(
        temporal["_report_distance_m"], temporal["_report_error_m"],
        color="seagreen", linewidth=2.0, label="Temporal-consistency fusion"
    )

    for thr in [10, 20, 40, 80]:
        ax.axhline(thr, color="gray", linewidth=0.7, linestyle="--", alpha=0.45)
        ax.text(
            xfeat["_report_distance_m"].max() * 1.002,
            thr,
            f"{thr} m",
            va="center",
            fontsize=8,
            color="gray",
        )

    # Vertical event markers
    if len(periodic_events) > 0 and "_report_distance_m" in periodic_events.columns:
        for d in periodic_events["_report_distance_m"].dropna().to_numpy():
            ax.axvline(d, color="royalblue", linewidth=0.55, alpha=0.18)

    if len(temporal_events) > 0 and "_report_distance_m" in temporal_events.columns:
        for d in temporal_events["_report_distance_m"].dropna().to_numpy():
            ax.axvline(d, color="seagreen", linewidth=0.75, alpha=0.25)

    ax.set_title("Error growth and correction events", fontsize=15, fontweight="bold")
    ax.set_xlabel("Traveled distance [m]")
    ax.set_ylabel("Position error [m]")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def make_folium_map(
    ref: pd.DataFrame,
    ref_xy: Tuple[str, str],
    xfeat: pd.DataFrame,
    xfeat_xy: Tuple[str, str],
    periodic: pd.DataFrame,
    periodic_xy: Tuple[str, str],
    temporal: pd.DataFrame,
    temporal_xy: Tuple[str, str],
    periodic_events: pd.DataFrame,
    temporal_events: pd.DataFrame,
    out_html: Path,
) -> Dict[str, Any]:
    try:
        import folium
    except Exception as exc:
        return {"status": "skipped", "reason": f"folium import failed: {exc}"}

    lat_col, lon_col = extract_lat_lon_reference(ref)
    if not lat_col or not lon_col:
        return {"status": "skipped", "reason": "reference lat/lon columns not found"}

    x0 = float(ref[ref_xy[0]].iloc[0])
    y0 = float(ref[ref_xy[1]].iloc[0])
    lat0 = float(ref[lat_col].iloc[0])
    lon0 = float(ref[lon_col].iloc[0])

    def to_latlon(df: pd.DataFrame, xy: Tuple[str, str]) -> List[Tuple[float, float]]:
        lat, lon = enu_to_latlon_approx(
            df[xy[0]].astype(float).to_numpy(),
            df[xy[1]].astype(float).to_numpy(),
            x0=x0,
            y0=y0,
            lat0=lat0,
            lon0=lon0,
        )
        return list(zip(lat.tolist(), lon.tolist()))

    center = [float(ref[lat_col].median()), float(ref[lon_col].median())]
    fmap = folium.Map(location=center, zoom_start=17, tiles="OpenStreetMap", control_scale=True)

    folium.PolyLine(
        locations=ref[[lat_col, lon_col]].dropna().values.tolist(),
        color="black",
        weight=2,
        opacity=0.85,
        tooltip="Reference trajectory",
    ).add_to(fmap)

    folium.PolyLine(
        locations=to_latlon(xfeat, xfeat_xy),
        color="orange",
        weight=3,
        opacity=0.85,
        tooltip="XFeat relative-only",
    ).add_to(fmap)

    folium.PolyLine(
        locations=to_latlon(periodic, periodic_xy),
        color="blue",
        weight=3,
        opacity=0.85,
        tooltip="Periodic fusion",
    ).add_to(fmap)

    folium.PolyLine(
        locations=to_latlon(temporal, temporal_xy),
        color="green",
        weight=3,
        opacity=0.85,
        tooltip="Temporal-consistency fusion",
    ).add_to(fmap)

    for _, row in periodic_events.iterrows():
        lat, lon = enu_to_latlon_approx(
            np.array([float(row[periodic_xy[0]])]),
            np.array([float(row[periodic_xy[1]])]),
            x0=x0,
            y0=y0,
            lat0=lat0,
            lon0=lon0,
        )
        folium.CircleMarker(
            location=[float(lat[0]), float(lon[0])],
            radius=4,
            color="blue",
            fill=False,
            tooltip="Periodic accepted correction",
        ).add_to(fmap)

    for _, row in temporal_events.iterrows():
        lat, lon = enu_to_latlon_approx(
            np.array([float(row[temporal_xy[0]])]),
            np.array([float(row[temporal_xy[1]])]),
            x0=x0,
            y0=y0,
            lat0=lat0,
            lon0=lon0,
        )
        folium.CircleMarker(
            location=[float(lat[0]), float(lon[0])],
            radius=5,
            color="green",
            fill=False,
            tooltip="Temporal accepted correction",
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

    return {"status": "written", "path": str(out_html), "lat_col": lat_col, "lon_col": lon_col}


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
    args = parser.parse_args()

    repo = Path(args.repo_root).resolve()
    cfg = load_yaml(repo / args.config)

    inputs = cfg["inputs"]
    story = cfg["story_policy"]

    # Load core tables
    ref = read_csv(repo / inputs["reference_trajectory"]["path"], "reference")
    xfeat = read_csv(repo / inputs["xfeat_relative_trajectory"]["path"], "xfeat_relative")
    f2 = read_csv(repo / inputs["fusion_f2_trajectory"]["path"], "fusion_f2")
    f3b = read_csv(repo / inputs["temporal_fusion_trajectory"]["path"], "temporal_fusion")

    periodic_method = story["best_metric_result"]["method_name"]
    periodic_alpha = story["best_metric_result"].get("alpha", None)

    temporal_method = story["primary_result"]["method_name"]
    temporal_alpha = story["primary_result"].get("alpha", None)

    periodic, periodic_filter_info = filter_by_method_and_alpha(
        f2, periodic_method, periodic_alpha, "periodic"
    )
    temporal, temporal_filter_info = filter_by_method_and_alpha(
        f3b, temporal_method, temporal_alpha, "temporal"
    )

    periodic, periodic_collapse_info = collapse_to_one_row_per_query(periodic, "periodic")
    temporal, temporal_collapse_info = collapse_to_one_row_per_query(temporal, "temporal")
    xfeat, xfeat_collapse_info = collapse_to_one_row_per_query(xfeat, "xfeat")
    ref, ref_collapse_info = collapse_to_one_row_per_query(ref, "reference")

    # Align lengths conservatively for error computation.
    min_len = min(len(ref), len(xfeat), len(periodic), len(temporal))
    if min_len < 10:
        raise ValueError(f"After filtering, too few rows remain: {min_len}")

    ref = ref.iloc[:min_len].reset_index(drop=True)
    xfeat = xfeat.iloc[:min_len].reset_index(drop=True)
    periodic = periodic.iloc[:min_len].reset_index(drop=True)
    temporal = temporal.iloc[:min_len].reset_index(drop=True)

    # Infer coordinates.
    ref_x, ref_y, ref_xy_reason = choose_xy_cols(ref, "reference")
    xfeat_x, xfeat_y, xfeat_xy_reason = choose_xy_cols(xfeat, "estimated")
    periodic_x, periodic_y, periodic_xy_reason = choose_xy_cols(periodic, "fusion")
    temporal_x, temporal_y, temporal_xy_reason = choose_xy_cols(temporal, "fusion")

    ref_xy_arr = (
        ref[ref_x].astype(float).to_numpy(),
        ref[ref_y].astype(float).to_numpy(),
    )

    xfeat = add_distance_and_error(xfeat, xfeat_x, xfeat_y, ref_xy_arr)
    periodic = add_distance_and_error(periodic, periodic_x, periodic_y, ref_xy_arr)
    temporal = add_distance_and_error(temporal, temporal_x, temporal_y, ref_xy_arr)

    # Try events from original filtered/collapsed tables. Add error/distance too.
    periodic_events = find_event_rows(periodic)
    temporal_events = find_event_rows(temporal)

    out_root = repo / args.out_prefix
    figs = out_root / "figures"
    maps = out_root / "maps"
    manifests = out_root / "manifests"
    tables = out_root / "tables"

    figs.mkdir(parents=True, exist_ok=True)
    maps.mkdir(parents=True, exist_ok=True)
    manifests.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)

    xy_png = figs / "fig_final_trajectory_map_overlay_xy.png"
    err_png = figs / "fig_fusion_error_vs_distance_with_events.png"
    folium_html = maps / "fig_final_trajectory_map_overlay.html"
    manifest_json = manifests / "report_01_fusion_overlay_schema_manifest.json"

    plot_xy_overlay(
        ref=ref,
        ref_xy=(ref_x, ref_y),
        xfeat=xfeat,
        xfeat_xy=(xfeat_x, xfeat_y),
        periodic=periodic,
        periodic_xy=(periodic_x, periodic_y),
        temporal=temporal,
        temporal_xy=(temporal_x, temporal_y),
        periodic_events=periodic_events,
        temporal_events=temporal_events,
        out_png=xy_png,
    )

    plot_error_vs_distance(
        xfeat=xfeat,
        periodic=periodic,
        temporal=temporal,
        periodic_events=periodic_events,
        temporal_events=temporal_events,
        out_png=err_png,
    )

    folium_info = make_folium_map(
        ref=ref,
        ref_xy=(ref_x, ref_y),
        xfeat=xfeat,
        xfeat_xy=(xfeat_x, xfeat_y),
        periodic=periodic,
        periodic_xy=(periodic_x, periodic_y),
        temporal=temporal,
        temporal_xy=(temporal_x, temporal_y),
        periodic_events=periodic_events,
        temporal_events=temporal_events,
        out_html=folium_html,
    )

    # Summary table from plotted curves.
    summary_rows = []
    for label, df in [
        ("XFeat relative-only", xfeat),
        ("Periodic fusion", periodic),
        ("Temporal-consistency fusion", temporal),
    ]:
        err = df["_report_error_m"].astype(float).to_numpy()
        err = err[np.isfinite(err)]
        if len(err) == 0:
            continue
        summary_rows.append({
            "method": label,
            "rows_plotted": len(df),
            "rmse_m": float(np.sqrt(np.mean(err ** 2))),
            "mae_m": float(np.mean(np.abs(err))),
            "median_error_m": float(np.median(err)),
            "p95_error_m": float(np.percentile(err, 95)),
            "max_error_m": float(np.max(err)),
            "final_error_m": float(err[-1]),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = tables / "table_final_system_summary_recomputed_from_plots.csv"
    summary_df.to_csv(summary_csv, index=False)

    manifest = {
        "status": "PASS_REPORT_01_FUSION_OVERLAY",
        "outputs": {
            "xy_overlay_png": str(xy_png),
            "error_vs_distance_png": str(err_png),
            "folium_html": str(folium_html),
            "summary_csv": str(summary_csv),
            "schema_manifest_json": str(manifest_json),
        },
        "row_counts": {
            "reference": len(ref),
            "xfeat": len(xfeat),
            "periodic": len(periodic),
            "temporal": len(temporal),
            "periodic_events_detected": len(periodic_events),
            "temporal_events_detected": len(temporal_events),
        },
        "filtering": {
            "periodic": periodic_filter_info,
            "periodic_collapse": periodic_collapse_info,
            "temporal": temporal_filter_info,
            "temporal_collapse": temporal_collapse_info,
            "xfeat_collapse": xfeat_collapse_info,
            "reference_collapse": ref_collapse_info,
        },
        "columns_used": {
            "reference": {"x": ref_x, "y": ref_y, "reason": ref_xy_reason},
            "xfeat": {"x": xfeat_x, "y": xfeat_y, "reason": xfeat_xy_reason},
            "periodic": {"x": periodic_x, "y": periodic_y, "reason": periodic_xy_reason},
            "temporal": {"x": temporal_x, "y": temporal_y, "reason": temporal_xy_reason},
        },
        "folium": folium_info,
        "summary_metrics_recomputed_from_plotted_rows": summary_rows,
    }

    write_schema_manifest(manifest_json, manifest)

    print("STATUS: PASS_REPORT_01_FUSION_OVERLAY")
    print(f"XY overlay:            {xy_png}")
    print(f"Error vs distance:     {err_png}")
    print(f"Folium map:            {folium_html} [{folium_info.get('status')}]")
    print(f"Summary CSV:           {summary_csv}")
    print(f"Schema manifest:       {manifest_json}")
    print("")
    print("Columns used:")
    print(json.dumps(manifest["columns_used"], indent=2))
    print("")
    print("Rows:")
    print(json.dumps(manifest["row_counts"], indent=2))


if __name__ == "__main__":
    main()
