'''
Run Command:

python scripts/villoc/reporting/report_02_correction_context_template.py \
  --config configs/reporting/report_villoc_traj01_s8_figures.yaml \
  2>&1 | tee outputs/villoc/traj01_90deg_stable120m/reporting/logs/report_02_correction_context_template.log

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


ALLOWED_CONTEXTS = [
    "urban_buildings",
    "road_parking_hard_surface",
    "vegetation_trees_grass",
    "mixed_edge_zone",
    "open_repetitive_area",
    "uncertain",
]


def norm_col(c: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(c).lower()).strip("_")


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


def find_first_existing_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    normalized = {norm_col(c): c for c in df.columns}
    for cand in candidates:
        key = norm_col(cand)
        if key in normalized:
            return normalized[key]
    return None


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


def make_annotation_template(
    ref: pd.DataFrame,
    ref_xy: Tuple[str, str],
    periodic: pd.DataFrame,
    periodic_xy: Tuple[str, str],
    temporal: pd.DataFrame,
    temporal_xy: Tuple[str, str],
    periodic_events: pd.DataFrame,
    temporal_events: pd.DataFrame,
) -> pd.DataFrame:
    lat_col = find_first_existing_col(ref, ["lat", "latitude", "reference_latitude"])
    lon_col = find_first_existing_col(ref, ["lon", "longitude", "reference_longitude"])

    if not lat_col or not lon_col:
        raise ValueError("Reference trajectory must contain lat/lon for annotation map.")

    x0 = float(ref[ref_xy[0]].iloc[0])
    y0 = float(ref[ref_xy[1]].iloc[0])
    lat0 = float(ref[lat_col].iloc[0])
    lon0 = float(ref[lon_col].iloc[0])

    rows = []

    def add_events(source: str, events: pd.DataFrame, xy: Tuple[str, str]) -> None:
        ident = choose_identity_col(events)
        for i, (_, row) in enumerate(events.iterrows(), start=1):
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

            rows.append({
                "event_id": f"{source}_{i:03d}",
                "correction_source": source,
                "query_identity": row[ident] if ident and ident in row.index else "",
                "sequence_order": i,
                "x_map_m": x,
                "y_map_m": y,
                "latitude_approx": float(lat[0]),
                "longitude_approx": float(lon[0]),
                "traveled_distance_m": float(row.get("_report_distance_m", np.nan)),
                "error_m_eval_only": float(row.get("_report_error_m", np.nan)),
                "accepted": True,
                "context_category": "uncertain",
                "context_notes": "",
                "annotation_confidence": "",
                "allowed_context_categories": "|".join(ALLOWED_CONTEXTS),
            })

    add_events("periodic", periodic_events, periodic_xy)
    add_events("temporal", temporal_events, temporal_xy)

    out = pd.DataFrame(rows)

    if len(out) > 0:
        out = out.sort_values(["traveled_distance_m", "correction_source", "event_id"]).reset_index(drop=True)

    return out


def plot_context_preview(
    ref: pd.DataFrame,
    ref_xy: Tuple[str, str],
    periodic_events: pd.DataFrame,
    periodic_xy: Tuple[str, str],
    temporal_events: pd.DataFrame,
    temporal_xy: Tuple[str, str],
    out_png: Path,
) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10.5, 9.0))

    ax.plot(
        ref[ref_xy[0]], ref[ref_xy[1]],
        color="black", linewidth=1.5, label="Reference trajectory"
    )

    if len(periodic_events) > 0:
        ax.scatter(
            periodic_events[periodic_xy[0]], periodic_events[periodic_xy[1]],
            s=52, marker="o", facecolors="none", edgecolors="royalblue",
            linewidths=1.5, label="Periodic accepted correction"
        )

    if len(temporal_events) > 0:
        ax.scatter(
            temporal_events[temporal_xy[0]], temporal_events[temporal_xy[1]],
            s=62, marker="o", facecolors="none", edgecolors="seagreen",
            linewidths=1.8, label="Temporal accepted correction"
        )

    ax.scatter(ref[ref_xy[0]].iloc[0], ref[ref_xy[1]].iloc[0], s=90, marker="*", color="black", label="Start")
    ax.scatter(ref[ref_xy[0]].iloc[-1], ref[ref_xy[1]].iloc[-1], s=70, marker="s", color="black", label="End")

    ax.set_title("Correction-event locations for context annotation", fontsize=15, fontweight="bold")
    ax.set_xlabel("Local X / East [m]")
    ax.set_ylabel("Local Y / North [m]")
    ax.axis("equal")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def make_folium_annotation_map(
    ref: pd.DataFrame,
    ref_xy: Tuple[str, str],
    annotation_df: pd.DataFrame,
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

    for _, row in annotation_df.iterrows():
        source = str(row["correction_source"])
        color = "blue" if source == "periodic" else "green"
        popup = (
            f"<b>{row['event_id']}</b><br>"
            f"source: {source}<br>"
            f"distance: {row['traveled_distance_m']:.1f} m<br>"
            f"error eval-only: {row['error_m_eval_only']:.2f} m<br>"
            f"context: {row['context_category']}<br>"
            f"notes: {row['context_notes']}"
        )

        folium.CircleMarker(
            location=[float(row["latitude_approx"]), float(row["longitude_approx"])],
            radius=5,
            color=color,
            fill=False,
            tooltip=f"{row['event_id']} | {source}",
            popup=folium.Popup(popup, max_width=360),
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

    ref = read_csv(repo / inputs["reference_trajectory"]["path"], "reference")
    f2 = read_csv(repo / inputs["fusion_f2_trajectory"]["path"], "fusion_f2")
    f3b = read_csv(repo / inputs["temporal_fusion_trajectory"]["path"], "temporal_fusion")

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

    annotation_df = make_annotation_template(
        ref=ref,
        ref_xy=(ref_x, ref_y),
        periodic=periodic,
        periodic_xy=(periodic_x, periodic_y),
        temporal=temporal,
        temporal_xy=(temporal_x, temporal_y),
        periodic_events=periodic_events,
        temporal_events=temporal_events,
    )

    out_root = repo / args.out_prefix
    manifests = out_root / "manifests"
    metadata = out_root / "manifests"
    figures = out_root / "figures"
    maps = out_root / "maps"

    manifests.mkdir(parents=True, exist_ok=True)
    metadata.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    maps.mkdir(parents=True, exist_ok=True)

    annotation_csv = manifests / "report_02_correction_context_annotation_template.csv"
    preview_png = figures / "fig_correction_context_event_preview_xy.png"
    folium_html = maps / "fig_correction_context_annotation_map.html"
    manifest_json = manifests / "report_02_correction_context_manifest.json"

    annotation_df.to_csv(annotation_csv, index=False)

    plot_context_preview(
        ref=ref,
        ref_xy=(ref_x, ref_y),
        periodic_events=periodic_events,
        periodic_xy=(periodic_x, periodic_y),
        temporal_events=temporal_events,
        temporal_xy=(temporal_x, temporal_y),
        out_png=preview_png,
    )

    folium_info = make_folium_annotation_map(
        ref=ref,
        ref_xy=(ref_x, ref_y),
        annotation_df=annotation_df,
        out_html=folium_html,
    )

    manifest = {
        "status": "PASS_REPORT_02_CORRECTION_CONTEXT_TEMPLATE",
        "outputs": {
            "annotation_template_csv": str(annotation_csv),
            "preview_png": str(preview_png),
            "folium_annotation_map": str(folium_html),
            "manifest_json": str(manifest_json),
        },
        "allowed_context_categories": ALLOWED_CONTEXTS,
        "row_counts": {
            "reference": len(ref),
            "periodic_rows": len(periodic),
            "temporal_rows": len(temporal),
            "periodic_events": len(periodic_events),
            "temporal_events": len(temporal_events),
            "annotation_rows": len(annotation_df),
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
        "columns_used": {
            "reference": {"x": ref_x, "y": ref_y, "reason": ref_reason},
            "periodic": {"x": periodic_x, "y": periodic_y, "reason": periodic_reason},
            "temporal": {"x": temporal_x, "y": temporal_y, "reason": temporal_reason},
        },
        "folium": folium_info,
    }

    with manifest_json.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("STATUS: PASS_REPORT_02_CORRECTION_CONTEXT_TEMPLATE")
    print(f"Annotation template CSV: {annotation_csv}")
    print(f"Preview XY figure:        {preview_png}")
    print(f"Folium annotation map:    {folium_html} [{folium_info.get('status')}]")
    print(f"Manifest JSON:            {manifest_json}")
    print("")
    print("Rows:")
    print(json.dumps(manifest["row_counts"], indent=2))
    print("")
    print("Event detection:")
    print(json.dumps(manifest["event_detection"], indent=2))


if __name__ == "__main__":
    main()
