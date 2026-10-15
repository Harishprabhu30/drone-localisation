#!/usr/bin/env python3
"""Visual comparison of frozen ORB controls and promoted SIFT+LightGlue on traj01.

This is a post-freeze development visualization. It uses the same reference
attachment already authorized and consumed by CV2.5.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from uavloc.evaluation.reference import load_reference_xy

ROOT = REPO_ROOT
CV25_CONFIG = ROOT / "configs/research/crossview_scale_rotation_domain_cv2_5.yaml"
CV25_SCRIPT = ROOT / "scripts/villoc/geometry/cv2_5_postfreeze_sift_lightglue_full403_validation.py"
BLIND_VIZ_SCRIPT = ROOT / "scripts/villoc/geometry/bv1_2_visualize_blind_comparison.py"

DEFAULT_MASTER_ORTHO = ROOT / (
    "data/raw/villoc/90_deg/maps/ort10lt_2024_2026/master/"
    "ort10lt_2024_2026_master.tif"
)
DEFAULT_AOI_ORTHO = ROOT / (
    "data/processed/villoc/90_deg/maps/ort10lt_2024_2026/"
    "ort10lt_2024_2026_aoi300m.tif"
)
DEFAULT_OUT = ROOT / (
    "outputs/research_runs/crossview_scale_rotation_domain_v1/cv2/cv2_5/"
    "visual_diagnostics"
)


def dynamic_import(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def resolve(value: Path | None, default: Path) -> Path:
    p = default if value is None else value.expanduser()
    if not p.is_absolute():
        p = ROOT / p
    return p.resolve()


def choose_ortho(user_path: Path | None) -> Path:
    if user_path is not None:
        return resolve(user_path, DEFAULT_AOI_ORTHO)
    if DEFAULT_MASTER_ORTHO.is_file():
        return DEFAULT_MASTER_ORTHO.resolve()
    return DEFAULT_AOI_ORTHO.resolve()


def selected_points(cv25, frame: pd.DataFrame, frontend: str) -> pd.DataFrame:
    rows = []
    for qid, group in frame.groupby("query_id", sort=True):
        ordered = cv25.rank(group).reset_index(drop=True)
        s = ordered.iloc[0]
        rows.append(
            {
                "frontend": frontend,
                "query_id": int(qid),
                "tile_id": str(s["tile_id"]),
                "pool_rank": int(s["pool_rank"]),
                "inliers": float(s["inliers"]),
                "projected_easting": float(s["projected_easting"])
                if pd.notna(s["projected_easting"]) else np.nan,
                "projected_northing": float(s["projected_northing"])
                if pd.notna(s["projected_northing"]) else np.nan,
                "projection_error_m": float(s["projection_error_m"])
                if pd.notna(s["projection_error_m"]) else np.nan,
                "gt_x": float(s["gt_x"]),
                "gt_y": float(s["gt_y"]),
                "contains_query": bool(s["contains_query"]),
                "center_le40": bool(s["center_le40"]),
                "center_le80": bool(s["center_le80"]),
            }
        )
    return pd.DataFrame(rows)


def finite_xy(frame: pd.DataFrame, x_col: str, y_col: str) -> pd.DataFrame:
    out = frame.copy()
    x = pd.to_numeric(out[x_col], errors="coerce")
    y = pd.to_numeric(out[y_col], errors="coerce")
    return out[np.isfinite(x) & np.isfinite(y)].copy()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize traj01 full403 ORB vs SIFT+LightGlue post-freeze results."
    )
    parser.add_argument("--config", type=Path, default=CV25_CONFIG)
    parser.add_argument("--orthophoto", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--map-margin-m", type=float, default=150.0)
    args = parser.parse_args()

    cv25 = dynamic_import("traj01_cv25", CV25_SCRIPT)
    viz = dynamic_import("traj01_viz_helpers", BLIND_VIZ_SCRIPT)

    cfg_path = resolve(args.config, CV25_CONFIG)
    cfg = yaml.safe_load(cfg_path.read_text())
    cv25.verify_config(cfg)

    orb_raw, scale_raw, sift_raw = cv25.verify_inputs(cfg)
    tile_path = cv25.verify_tiles(cfg)
    tiles = pd.read_csv(tile_path)

    ref_path = cv25.resolve(cfg["development"]["reference_attachment"])
    reference = load_reference_xy(ref_path)

    attached = {
        "original_orb": cv25.attach(orb_raw, tiles, reference, "original_orb"),
        "scale_m2_orb": cv25.attach(scale_raw, tiles, reference, "scale_m2_orb"),
        "sift_lightglue": cv25.attach(sift_raw, tiles, reference, "sift_lightglue"),
    }

    selected = {
        name: selected_points(cv25, frame, name)
        for name, frame in attached.items()
    }

    original = selected["original_orb"]
    scale = selected["scale_m2_orb"]
    sift = selected["sift_lightglue"]

    audit = (
        scale.add_prefix("scale_")
        .rename(columns={"scale_query_id": "query_id"})
        .merge(
            sift.add_prefix("sift_").rename(columns={"sift_query_id": "query_id"}),
            on="query_id",
            how="inner",
            validate="one_to_one",
        )
    )

    scale_err = pd.to_numeric(audit["scale_projection_error_m"], errors="coerce")
    sift_err = pd.to_numeric(audit["sift_projection_error_m"], errors="coerce")

    audit["sift_rescue_to_le40"] = (
        ((~np.isfinite(scale_err)) | (scale_err > 40.0))
        & np.isfinite(sift_err)
        & (sift_err <= 40.0)
    )
    audit["sift_catastrophic_rescue"] = (
        ((~np.isfinite(scale_err)) | (scale_err > 200.0))
        & np.isfinite(sift_err)
        & (sift_err <= 200.0)
    )
    audit["sift_damage_from_le40"] = (
        np.isfinite(scale_err)
        & (scale_err <= 40.0)
        & ((~np.isfinite(sift_err)) | (sift_err > 40.0))
    )
    audit["sift_catastrophic_damage"] = (
        np.isfinite(scale_err)
        & (scale_err <= 200.0)
        & ((~np.isfinite(sift_err)) | (sift_err > 200.0))
    )

    ref_frame = reference[["gt_x", "gt_y"]].reset_index()
    ref_frame["query_id"] = pd.to_numeric(ref_frame["query_id"], errors="raise").astype(int)

    point_sets = [
        (ref_frame["gt_x"], ref_frame["gt_y"]),
        (original["projected_easting"], original["projected_northing"]),
        (scale["projected_easting"], scale["projected_northing"]),
        (sift["projected_easting"], sift["projected_northing"]),
    ]
    view_bounds = viz.finite_plot_bounds(
        point_sets,
        margin_m=float(args.map_margin_m),
    )

    ortho_path = choose_ortho(args.orthophoto)
    image, extent, crs = viz.load_ortho_preview(
        ortho_path,
        requested_bounds=view_bounds,
    )
    if not viz.is_lks94_lithuania_tm(crs):
        raise RuntimeError(
            "Expected LKS94 / Lithuania TM (EPSG:3346-equivalent), "
            f"observed {crs}"
        )

    out_dir = (
        DEFAULT_OUT if args.out_dir is None
        else resolve(args.out_dir, DEFAULT_OUT)
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # Figure 1 — reference + selected absolute trajectories.
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(13, 11))
    viz.plot_background(ax, image, extent, view_bounds=view_bounds)

    ax.plot(
        ref_frame["gt_x"], ref_frame["gt_y"],
        linewidth=2.4, label="Reference trajectory",
    )

    original_f = finite_xy(original, "projected_easting", "projected_northing")
    scale_f = finite_xy(scale, "projected_easting", "projected_northing")
    sift_f = finite_xy(sift, "projected_easting", "projected_northing")

    ax.plot(
        original_f["projected_easting"], original_f["projected_northing"],
        linewidth=1.0, alpha=0.55, label="Original ORB selected positions",
    )
    ax.plot(
        scale_f["projected_easting"], scale_f["projected_northing"],
        linewidth=1.4, alpha=0.75, label="scale_m2 ORB selected positions",
    )
    ax.plot(
        sift_f["projected_easting"], sift_f["projected_northing"],
        linewidth=1.8, alpha=0.9, label="SIFT+LightGlue selected positions",
    )

    ax.set_title(
        "traj01_90deg_stable120m — full403 absolute frontend comparison\n"
        "Reference vs original ORB vs scale_m2 ORB vs SIFT+LightGlue"
    )
    ax.legend(loc="best")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    trajectory_path = (
        out_dir / "traj01_90deg_stable120m_frontend_trajectory_comparison_ortho.png"
    )
    fig.savefig(trajectory_path, dpi=220)
    plt.close(fig)

    # ---------------------------------------------------------
    # Figure 2 — SIFT vs promoted ORB selected points/rescues.
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(13, 11))
    viz.plot_background(ax, image, extent, view_bounds=view_bounds)

    ax.plot(
        ref_frame["gt_x"], ref_frame["gt_y"],
        linewidth=2.0, alpha=0.85, label="Reference trajectory",
    )

    ax.scatter(
        scale_f["projected_easting"], scale_f["projected_northing"],
        s=16, alpha=0.40, marker="o", label="scale_m2 ORB selected",
    )
    ax.scatter(
        sift_f["projected_easting"], sift_f["projected_northing"],
        s=20, alpha=0.55, marker=".", label="SIFT+LightGlue selected",
    )

    rescued = audit[audit["sift_rescue_to_le40"]].copy()
    catastrophic_rescued = audit[audit["sift_catastrophic_rescue"]].copy()
    damaged = audit[audit["sift_damage_from_le40"]].copy()
    catastrophic_damaged = audit[audit["sift_catastrophic_damage"]].copy()

    if len(rescued):
        ax.scatter(
            rescued["sift_projected_easting"],
            rescued["sift_projected_northing"],
            s=75, marker="*", label=f"SIFT rescue to <=40 m ({len(rescued)})",
        )
    if len(catastrophic_rescued):
        ax.scatter(
            catastrophic_rescued["sift_projected_easting"],
            catastrophic_rescued["sift_projected_northing"],
            s=70, marker="s",
            label=f"SIFT rescue from ORB >200/nonfinite ({len(catastrophic_rescued)})",
        )
    if len(damaged):
        finite_damage = finite_xy(
            damaged,
            "sift_projected_easting",
            "sift_projected_northing",
        )
        if len(finite_damage):
            ax.scatter(
                finite_damage["sift_projected_easting"],
                finite_damage["sift_projected_northing"],
                s=65, marker="X",
                label=f"SIFT loss from ORB <=40 m ({len(damaged)})",
            )
    if len(catastrophic_damaged):
        finite_damage = finite_xy(
            catastrophic_damaged,
            "sift_projected_easting",
            "sift_projected_northing",
        )
        if len(finite_damage):
            ax.scatter(
                finite_damage["sift_projected_easting"],
                finite_damage["sift_projected_northing"],
                s=80, marker="^",
                label=f"SIFT catastrophic damage ({len(catastrophic_damaged)})",
            )

    ax.set_title(
        "traj01_90deg_stable120m — selected-point comparison\n"
        "Promoted scale_m2 ORB vs SIFT+LightGlue with rescue/damage highlights"
    )
    ax.legend(loc="best")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    points_path = (
        out_dir / "traj01_90deg_stable120m_sift_vs_scale_m2_selected_points_ortho.png"
    )
    fig.savefig(points_path, dpi=220)
    plt.close(fig)

    original.to_csv(out_dir / "traj01_original_orb_selected_points.csv", index=False)
    scale.to_csv(out_dir / "traj01_scale_m2_orb_selected_points.csv", index=False)
    sift.to_csv(out_dir / "traj01_sift_lightglue_selected_points.csv", index=False)
    audit.to_csv(out_dir / "traj01_sift_vs_scale_m2_visual_audit.csv", index=False)

    print("PASS_TRAJ01_90DEG_STABLE120M_FRONTEND_VISUAL_DIAGNOSTICS")
    print("Orthophoto:", ortho_path)
    print("Plot bounds:", view_bounds)
    print("Reference queries:", len(ref_frame))
    print("Original ORB finite selected:", len(original_f))
    print("scale_m2 ORB finite selected:", len(scale_f))
    print("SIFT+LightGlue finite selected:", len(sift_f))
    print("SIFT rescues to <=40 m:", int(audit["sift_rescue_to_le40"].sum()))
    print("SIFT catastrophic rescues:", int(audit["sift_catastrophic_rescue"].sum()))
    print("SIFT losses from ORB <=40 m:", int(audit["sift_damage_from_le40"].sum()))
    print("SIFT catastrophic damages:", int(audit["sift_catastrophic_damage"].sum()))
    print("Trajectory figure:", trajectory_path)
    print("Selected-points figure:", points_path)


if __name__ == "__main__":
    main()
