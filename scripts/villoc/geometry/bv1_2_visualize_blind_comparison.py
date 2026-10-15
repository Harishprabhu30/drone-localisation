#!/usr/bin/env python3
"""Visual diagnostics for BV1.2 blind SIFT+LightGlue transfer.

Creates:
1) causal BV1.2 trajectory vs historical ORB blind-demo trajectory on the orthophoto;
2) gate-supported / held absolute observation points on the orthophoto.

No SRT, GT, reference trajectory, oracle labels, or error metrics are loaded.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import rasterio
    from rasterio.enums import Resampling
except ImportError as exc:
    raise RuntimeError(
        "rasterio is required for orthophoto visualization. "
        "Install it in the active project environment."
    ) from exc

ROOT = Path(__file__).resolve().parents[3]

DEFAULT_ORTHO = ROOT / (
    "data/processed/villoc/90_deg/maps/ort10lt_2024_2026/"
    "ort10lt_2024_2026_aoi300m.tif"
)
DEFAULT_BV12_ROOT = ROOT / (
    "outputs/research_runs/blind_demo_sift_lightglue_transfer_v1/bv1_2"
)
DEFAULT_RELATIVE = ROOT / (
    "outputs/demo_runs/blind_recorded_flight_final_001/metadata/"
    "s8_xfeat_relative_frontend/s8r4_xfeat_relative_trajectory_blind_raw.csv"
)
ORB_SUBMISSION_CANDIDATES = [
    ROOT / (
        "outputs/demo_runs/blind_recorded_flight_final_001/"
        "submission_estimated_trajectory.csv"
    ),
    ROOT / (
        "docs/demo/blind_recorded_flight_final_001/"
        "submission_estimated_trajectory.csv"
    ),
]


def resolve_optional(value: Path | None, default: Path) -> Path:
    path = default if value is None else value.expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes", "y", "t"})
    )


def find_orb_submission(user_path: Path | None) -> Path:
    if user_path is not None:
        path = user_path.expanduser()
        if not path.is_absolute():
            path = ROOT / path
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    for candidate in ORB_SUBMISSION_CANDIDATES:
        if candidate.is_file():
            return candidate.resolve()

    raise FileNotFoundError(
        "Could not find historical ORB blind submission. Tried:\n"
        + "\n".join(str(p) for p in ORB_SUBMISSION_CANDIDATES)
    )


def load_ortho_preview(path: Path, max_dim: int = 2200):
    if not path.is_file():
        raise FileNotFoundError(path)

    with rasterio.open(path) as ds:
        if ds.count >= 3:
            indexes = [1, 2, 3]
        else:
            indexes = [1]

        scale = min(1.0, float(max_dim) / max(ds.width, ds.height))
        out_w = max(1, int(round(ds.width * scale)))
        out_h = max(1, int(round(ds.height * scale)))

        data = ds.read(
            indexes,
            out_shape=(len(indexes), out_h, out_w),
            resampling=Resampling.bilinear,
        ).astype(np.float32)

        bounds = ds.bounds
        crs = ds.crs

    if data.shape[0] == 1:
        data = np.repeat(data, 3, axis=0)

    image = np.moveaxis(data[:3], 0, -1)

    # Contrast stretch for display only.
    valid = np.isfinite(image)
    if valid.any():
        for c in range(image.shape[2]):
            channel = image[..., c]
            finite = channel[np.isfinite(channel)]
            if len(finite):
                lo, hi = np.percentile(finite, [2, 98])
                if hi > lo:
                    image[..., c] = np.clip((channel - lo) / (hi - lo), 0, 1)
                else:
                    image[..., c] = 0
    image = np.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0)

    extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]
    return image, extent, crs


def is_lks94_lithuania_tm(crs) -> bool:
    """Accept EPSG:3346 or an equivalent LKS94 / Lithuania TM definition."""
    if crs is None:
        return False

    try:
        if crs.to_epsg() == 3346:
            return True
    except Exception:
        pass

    try:
        expected = rasterio.crs.CRS.from_epsg(3346)
        if crs == expected:
            return True
    except Exception:
        pass

    # Some GeoTIFFs carry the correct projection parameters but omit the
    # projected-CRS EPSG authority, so to_epsg() can return None.
    try:
        params = crs.to_dict()
        proj = str(params.get("proj", "")).lower()
        ellps = str(params.get("ellps", "")).upper()
        units = str(params.get("units", "")).lower()
        lon0 = float(params.get("lon_0"))
        lat0 = float(params.get("lat_0", 0.0))
        k = float(params.get("k", params.get("k_0")))
        x0 = float(params.get("x_0"))
        y0 = float(params.get("y_0", 0.0))
        return bool(
            proj == "tmerc"
            and ellps in {"GRS80", "GRS 1980"}
            and units in {"m", "metre", "meter"}
            and abs(lon0 - 24.0) <= 1e-9
            and abs(lat0 - 0.0) <= 1e-9
            and abs(k - 0.9998) <= 1e-12
            and abs(x0 - 500000.0) <= 1e-6
            and abs(y0 - 0.0) <= 1e-6
        )
    except Exception:
        return False


def load_bv12(root: Path):
    report_path = root / "bv1_2_report.json"
    timeline_path = root / "bv1_2_blind_policy_timeline.csv"
    hypotheses_path = root / "bv1_2_blind_subtile_hypotheses.csv"
    observations_path = root / "bv1_2_blind_top4_sift_observations.csv"

    for path in (report_path, timeline_path, hypotheses_path, observations_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    report = json.loads(report_path.read_text())
    timeline = pd.read_csv(timeline_path)
    hypotheses = pd.read_csv(hypotheses_path)
    observations = pd.read_csv(observations_path)

    if report.get("status") != "PASS_BV1_2_BLIND_SIFT_LIGHTGLUE_MINIMUM_CONFIDENT_TRANSFER":
        raise RuntimeError("BV1.2 report is not the frozen PASS")

    if report["blind_contract"]["reference_used"] is not False:
        raise RuntimeError("BV1.2 report unexpectedly indicates reference use")
    if report["blind_contract"]["orb_recomputed"] is not False:
        raise RuntimeError("BV1.2 report unexpectedly indicates ORB recomputation")

    return report, timeline, hypotheses, observations


def build_causal_bv12_trajectory(
    report: dict,
    timeline: pd.DataFrame,
    hypotheses: pd.DataFrame,
    relative: pd.DataFrame,
) -> pd.DataFrame:
    required_t = {
        "update_query_id",
        "active_hypothesis_id_after",
        "active_source_update_q_after",
        "mode_after",
        "action",
    }
    required_h = {
        "hypothesis_id", "a_real", "a_imag", "b_real", "b_imag",
        "scale_m_per_visual_px", "rotation_deg",
    }
    required_r = {"token0_id", "visual_x_px", "visual_y_px"}

    if required_t - set(timeline.columns):
        raise RuntimeError(f"Timeline missing: {sorted(required_t - set(timeline.columns))}")
    if required_h - set(hypotheses.columns):
        raise RuntimeError(f"Hypotheses missing: {sorted(required_h - set(hypotheses.columns))}")
    if required_r - set(relative.columns):
        raise RuntimeError(f"Relative trajectory missing: {sorted(required_r - set(relative.columns))}")

    tl = timeline.copy()
    tl["query_id"] = pd.to_numeric(tl["update_query_id"], errors="raise").astype(int)
    tl["active_hypothesis_id_after"] = pd.to_numeric(
        tl["active_hypothesis_id_after"], errors="coerce"
    )

    hyp = hypotheses.copy()
    hyp["hypothesis_id"] = pd.to_numeric(hyp["hypothesis_id"], errors="raise").astype(int)
    hyp = hyp.drop_duplicates("hypothesis_id").set_index("hypothesis_id")

    rel = relative.copy()
    rel["query_id"] = pd.to_numeric(rel["token0_id"], errors="raise").astype(int)
    rel["visual_x_px"] = pd.to_numeric(rel["visual_x_px"], errors="raise")
    rel["visual_y_px"] = pd.to_numeric(rel["visual_y_px"], errors="raise")
    rel = rel.set_index("query_id")

    matured_at = report.get("matured_at_query_id")
    if matured_at is None:
        return pd.DataFrame(
            columns=[
                "query_id", "map_x", "map_y", "active_hypothesis_id",
                "active_source_query", "mode_after", "action",
            ]
        )
    matured_at = int(matured_at)

    rows = []
    for row in tl.itertuples(index=False):
        qid = int(row.query_id)
        if qid < matured_at:
            continue

        hid_v = row.active_hypothesis_id_after
        if pd.isna(hid_v):
            continue
        hid = int(hid_v)

        if hid not in hyp.index or qid not in rel.index:
            continue

        h = hyp.loc[hid]
        r = rel.loc[qid]
        z = complex(float(r["visual_x_px"]), float(r["visual_y_px"]))
        a = complex(float(h["a_real"]), float(h["a_imag"]))
        b = complex(float(h["b_real"]), float(h["b_imag"]))
        w = a * z + b

        rows.append(
            {
                "query_id": qid,
                "map_x": float(w.real),
                "map_y": float(w.imag),
                "active_hypothesis_id": hid,
                "active_source_query": int(row.active_source_update_q_after),
                "mode_after": str(row.mode_after),
                "action": str(row.action),
                "scale_m_per_visual_px": float(h["scale_m_per_visual_px"]),
                "rotation_deg": float(h["rotation_deg"]),
            }
        )

    return pd.DataFrame(rows)


def build_bv12_gate_points(
    report: dict,
    timeline: pd.DataFrame,
    observations: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    tl = timeline.copy()
    tl["query_id"] = pd.to_numeric(tl["update_query_id"], errors="raise").astype(int)
    tl["minimum_innovation_m"] = pd.to_numeric(
        tl["minimum_innovation_m"], errors="coerce"
    )
    tl["innovation_best_choice_rank"] = pd.to_numeric(
        tl["innovation_best_choice_rank"], errors="coerce"
    )

    obs = observations.copy()
    obs["query_id"] = pd.to_numeric(obs["query_id"], errors="raise").astype(int)
    obs["candidate_choice_rank"] = pd.to_numeric(
        obs["candidate_choice_rank"], errors="raise"
    ).astype(int)
    obs["projected_easting"] = pd.to_numeric(obs["projected_easting"], errors="coerce")
    obs["projected_northing"] = pd.to_numeric(obs["projected_northing"], errors="coerce")

    activation = float(report["policy"]["activation_threshold_m"])
    tracking = float(report["policy"]["tracking_threshold_m"])

    joined = tl.merge(
        obs[
            [
                "query_id", "candidate_choice_rank", "tile_id",
                "projected_easting", "projected_northing",
            ]
        ],
        left_on=["query_id", "innovation_best_choice_rank"],
        right_on=["query_id", "candidate_choice_rank"],
        how="left",
        validate="many_to_one",
    )

    threshold = np.where(
        joined["mode_before"].astype(str).eq("ACQUISITION"),
        activation,
        tracking,
    )
    supported = (
        joined["minimum_innovation_m"].notna()
        & np.isfinite(joined["minimum_innovation_m"])
        & (joined["minimum_innovation_m"].to_numpy(float) <= threshold)
        & joined["action"].astype(str).isin({"ACQUISITION_ACCEPT", "TRACKING_ACCEPT"})
        & joined["projected_easting"].notna()
        & joined["projected_northing"].notna()
    )
    held = (
        joined["action"].astype(str).eq("TRACKING_HOLD_INNOVATION")
        & joined["projected_easting"].notna()
        & joined["projected_northing"].notna()
    )

    return joined.loc[supported].copy(), joined.loc[held].copy()


def load_orb_submission(path: Path) -> tuple[pd.DataFrame, int | None, pd.DataFrame]:
    df = pd.read_csv(path)
    required = {
        "query_id", "estimated_map_x", "estimated_map_y",
        "map_aligned_available", "accepted_correction",
    }
    if required - set(df.columns):
        raise RuntimeError(f"ORB submission missing {sorted(required - set(df.columns))}")

    df["query_id"] = pd.to_numeric(df["query_id"], errors="raise").astype(int)
    df["estimated_map_x"] = pd.to_numeric(df["estimated_map_x"], errors="coerce")
    df["estimated_map_y"] = pd.to_numeric(df["estimated_map_y"], errors="coerce")
    df["map_aligned_available"] = bool_series(df["map_aligned_available"])
    df["accepted_correction"] = bool_series(df["accepted_correction"])

    mapped = df[
        df["map_aligned_available"]
        & df["estimated_map_x"].notna()
        & df["estimated_map_y"].notna()
    ].copy()

    lock_query = int(mapped["query_id"].min()) if len(mapped) else None

    accepted = df[
        df["accepted_correction"]
        & df["estimated_map_x"].notna()
        & df["estimated_map_y"].notna()
    ].copy()

    return mapped, lock_query, accepted


def plot_background(ax, image, extent):
    ax.imshow(image, extent=extent, origin="upper")
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Easting [m] — EPSG:3346")
    ax.set_ylabel("Northing [m] — EPSG:3346")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize BV1.2 SIFT blind trajectory vs historical ORB blind demo."
    )
    parser.add_argument("--bv12-root", type=Path, default=None)
    parser.add_argument("--relative-csv", type=Path, default=None)
    parser.add_argument("--orthophoto", type=Path, default=None)
    parser.add_argument("--orb-submission", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    bv12_root = resolve_optional(args.bv12_root, DEFAULT_BV12_ROOT)
    relative_path = resolve_optional(args.relative_csv, DEFAULT_RELATIVE)
    ortho_path = resolve_optional(args.orthophoto, DEFAULT_ORTHO)
    orb_path = find_orb_submission(args.orb_submission)

    out_dir = (
        (bv12_root / "visual_diagnostics")
        if args.out_dir is None
        else resolve_optional(args.out_dir, bv12_root / "visual_diagnostics")
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    report, timeline, hypotheses, observations = load_bv12(bv12_root)
    relative = pd.read_csv(relative_path)

    sift_traj = build_causal_bv12_trajectory(
        report, timeline, hypotheses, relative
    )
    sift_supported, sift_held = build_bv12_gate_points(
        report, timeline, observations
    )
    orb_traj, orb_lock_query, orb_accepted = load_orb_submission(orb_path)

    image, extent, crs = load_ortho_preview(ortho_path)
    if not is_lks94_lithuania_tm(crs):
        raise RuntimeError(
            "Expected LKS94 / Lithuania TM (EPSG:3346-equivalent) orthophoto, "
            f"observed {crs}"
        )

    sift_lock_query = report.get("matured_at_query_id")
    sift_lock_query = int(sift_lock_query) if sift_lock_query is not None else None

    # ---------------------------------------------------------
    # Figure 1 — causal trajectory comparison.
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 10))
    plot_background(ax, image, extent)

    if len(orb_traj):
        ax.plot(
            orb_traj["estimated_map_x"],
            orb_traj["estimated_map_y"],
            linewidth=2.0,
            label=f"Historical ORB blind trajectory (lock q{orb_lock_query})",
        )
        ax.scatter(
            orb_traj["estimated_map_x"].iloc[0],
            orb_traj["estimated_map_y"].iloc[0],
            marker="s",
            s=70,
            label="ORB first map-aligned pose",
        )

    if len(sift_traj):
        ax.plot(
            sift_traj["map_x"],
            sift_traj["map_y"],
            linewidth=2.0,
            label=f"BV1.2 SIFT+LightGlue causal trajectory (lock q{sift_lock_query})",
        )
        ax.scatter(
            sift_traj["map_x"].iloc[0],
            sift_traj["map_y"].iloc[0],
            marker="*",
            s=140,
            label="SIFT first provisional-lock pose",
        )

    ax.set_title(
        "Blind map-aligned trajectory comparison\n"
        "Historical ORB demo vs BV1.2 SIFT+LightGlue — no SRT/GT/reference"
    )
    ax.legend(loc="best")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    trajectory_path = out_dir / "bv1_2_trajectory_vs_historical_orb_ortho.png"
    fig.savefig(trajectory_path, dpi=220)
    plt.close(fig)

    # ---------------------------------------------------------
    # Figure 2 — accepted / held evidence.
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 10))
    plot_background(ax, image, extent)

    if len(orb_traj):
        ax.plot(
            orb_traj["estimated_map_x"],
            orb_traj["estimated_map_y"],
            linewidth=1.0,
            alpha=0.55,
            label="Historical ORB map-aligned path",
        )

    if len(sift_traj):
        ax.plot(
            sift_traj["map_x"],
            sift_traj["map_y"],
            linewidth=1.0,
            alpha=0.55,
            label="BV1.2 causal SIFT path",
        )

    if len(orb_accepted):
        ax.scatter(
            orb_accepted["estimated_map_x"],
            orb_accepted["estimated_map_y"],
            marker="X",
            s=95,
            label=f"Historical ORB accepted corrections ({len(orb_accepted)})",
        )

    if len(sift_supported):
        ax.scatter(
            sift_supported["projected_easting"],
            sift_supported["projected_northing"],
            marker="o",
            s=55,
            label=f"BV1.2 gate-supported SIFT observations ({len(sift_supported)})",
        )

    if len(sift_held):
        ax.scatter(
            sift_held["projected_easting"],
            sift_held["projected_northing"],
            marker="^",
            s=45,
            alpha=0.65,
            label=f"BV1.2 tracking holds: innovation >12.8 m ({len(sift_held)})",
        )

    if len(sift_supported) and sift_lock_query is not None:
        lock_support = sift_supported[sift_supported["query_id"] == sift_lock_query]
        if len(lock_support):
            ax.scatter(
                lock_support["projected_easting"],
                lock_support["projected_northing"],
                marker="*",
                s=180,
                label=f"BV1.2 maturity support at q{sift_lock_query}",
            )

    ax.set_title(
        "Blind absolute-evidence decisions on orthophoto\n"
        "BV1.2 SIFT+LightGlue gate support / holds vs historical ORB corrections"
    )
    ax.legend(loc="best")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    accepted_path = out_dir / "bv1_2_accepted_points_vs_historical_orb_ortho.png"
    fig.savefig(accepted_path, dpi=220)
    plt.close(fig)

    # Reproducibility tables for what was actually drawn.
    sift_traj.to_csv(
        out_dir / "bv1_2_causal_trajectory_visualization.csv",
        index=False,
    )
    sift_supported.to_csv(
        out_dir / "bv1_2_gate_supported_observations_visualization.csv",
        index=False,
    )
    sift_held.to_csv(
        out_dir / "bv1_2_tracking_holds_visualization.csv",
        index=False,
    )

    print("PASS_BV1_2_BLIND_VISUAL_DIAGNOSTICS")
    print("Orthophoto:", ortho_path)
    print("Historical ORB submission:", orb_path)
    print("ORB map-aligned rows:", len(orb_traj))
    print("ORB lock query:", orb_lock_query)
    print("ORB accepted corrections:", len(orb_accepted))
    print("BV1.2 causal map-aligned rows:", len(sift_traj))
    print("BV1.2 lock query:", sift_lock_query)
    print("BV1.2 gate-supported observations:", len(sift_supported))
    print("BV1.2 tracking holds:", len(sift_held))
    print("Trajectory figure:", trajectory_path)
    print("Accepted-points figure:", accepted_path)
    print("Reference/SRT/GT loaded: False")


if __name__ == "__main__":
    main()
