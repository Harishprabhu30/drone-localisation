#!/usr/bin/env python3
"""
S4C.5A — Junction-aware luma-LSD frontend preflight.

Purpose:
  Detect L / T / X junctions from luma-LSD line segments and inspect whether
  junctions behave as stronger structural landmarks than isolated lines.

Rows per token:
  - UAV query
  - GT/nearest eval-only tile
  - PHOG top-1 tile
  - luma-LSD top-1 tile from S4C.4C, if available
  - oracle best top-N eval-only tile

Important:
  - This is diagnostic only.
  - No final reranking is performed.
  - GT/oracle/error fields are used only for post-hoc evaluation/debug.

Code Used:
export PYTHONPATH=$PWD/src

python scripts/satloc/s4c/s4c_5a_junction_lsd_frontend.py \
  --sequence traj01 \
  --tokens 1,40,60,90,100,129,166,269,516,905 \
  --preprocess luma \
  --resize-size 512 \
  --edge-method sobel \
  --blur-ksize 3 \
  --threshold-mode percentile \
  --threshold-percentile 65 \
  --close-ksize 3 \
  --open-ksize 1 \
  --min-component-area 65 \
  --min-line-length 24 \
  --max-lines 120 \
  --orientation-bins 18 \
  --endpoint-tol-px 12 \
  --cluster-radius-px 10 \
  --min-angle-deg 18 \
  --max-junctions 160 \
  --oracle-top-n 50
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Optional

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


THIS_DIR = Path(__file__).resolve().parent

S4C0_PATH = THIS_DIR / "s4c_0_macrocontour_preflight.py"
S4C4A_PATH = THIS_DIR / "s4c_4a_vector_skeleton_preflight.py"

for p in [S4C0_PATH, S4C4A_PATH]:
    if not p.exists():
        raise FileNotFoundError(f"Missing helper: {p}")

spec0 = importlib.util.spec_from_file_location("s4c0_helpers", S4C0_PATH)
s4c0 = importlib.util.module_from_spec(spec0)
sys.modules["s4c0_helpers"] = s4c0
assert spec0.loader is not None
spec0.loader.exec_module(s4c0)

spec4a = importlib.util.spec_from_file_location("s4c4a_helpers", S4C4A_PATH)
s4c4a = importlib.util.module_from_spec(spec4a)
sys.modules["s4c4a_helpers"] = s4c4a
assert spec4a.loader is not None
spec4a.loader.exec_module(s4c4a)


OUT_ROOT = Path("outputs/satloc")
DEFAULT_UAV_INDEX = OUT_ROOT / "metadata/uav_frames_index_enriched.csv"
DEFAULT_SAT_INDEX = OUT_ROOT / "metadata/satellite_tiles_index_enriched.csv"
DEFAULT_S4C1_DIR = OUT_ROOT / "metadata/s4c_macrocontour_phog_chamfer/s4c1_phog_retrieval"
DEFAULT_S4C4C_DIR = OUT_ROOT / "metadata/s4c_macrocontour_phog_chamfer/s4c4_vector_skeleton/s4c4c_luma_lsd_rerank"

OUT_FIG_DIR = OUT_ROOT / "figures/s4c_macrocontour_phog_chamfer/s4c5_junction_lsd/s4c5a_frontend"
OUT_META_DIR = OUT_ROOT / "metadata/s4c_macrocontour_phog_chamfer/s4c5_junction_lsd"
OUT_REPORT_DIR = OUT_ROOT / "reports/s4c_macrocontour_phog_chamfer/s4c5_junction_lsd"


def parse_tokens(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def safe_float(x: Any, default: float = np.nan) -> float:
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    return obj


def cross2(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def line_angle_deg(line: np.ndarray) -> float:
    x1, y1, x2, y2 = line.astype(float)
    ang = math.degrees(math.atan2(y2 - y1, x2 - x1))
    return float(ang % 180.0)


def undirected_angle_between(l1: np.ndarray, l2: np.ndarray) -> float:
    a = np.array([l1[2] - l1[0], l1[3] - l1[1]], dtype=np.float64)
    b = np.array([l2[2] - l2[0], l2[3] - l2[1]], dtype=np.float64)

    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)

    if na < 1e-6 or nb < 1e-6:
        return 0.0

    cosv = abs(float(np.dot(a, b) / (na * nb)))
    cosv = max(-1.0, min(1.0, cosv))
    return float(math.degrees(math.acos(cosv)))


def segment_intersection_candidate(
    l1: np.ndarray,
    l2: np.ndarray,
    endpoint_tol_px: float,
    min_angle_deg: float,
) -> Optional[dict[str, Any]]:
    p = np.array([l1[0], l1[1]], dtype=np.float64)
    r = np.array([l1[2] - l1[0], l1[3] - l1[1]], dtype=np.float64)

    q = np.array([l2[0], l2[1]], dtype=np.float64)
    s = np.array([l2[2] - l2[0], l2[3] - l2[1]], dtype=np.float64)

    len1 = float(np.linalg.norm(r))
    len2 = float(np.linalg.norm(s))

    if len1 < 1e-6 or len2 < 1e-6:
        return None

    angle = undirected_angle_between(l1, l2)
    if angle < min_angle_deg:
        return None

    denom = cross2(r, s)
    if abs(denom) < 1e-6:
        return None

    qp = q - p
    t = cross2(qp, s) / denom
    u = cross2(qp, r) / denom

    tol_t = endpoint_tol_px / max(len1, 1.0)
    tol_u = endpoint_tol_px / max(len2, 1.0)

    if not (-tol_t <= t <= 1.0 + tol_t and -tol_u <= u <= 1.0 + tol_u):
        return None

    pt = p + t * r

    e1 = t <= tol_t or t >= 1.0 - tol_t
    e2 = u <= tol_u or u >= 1.0 - tol_u

    if (not e1) and (not e2):
        jtype = "X"
    elif e1 != e2:
        jtype = "T"
    else:
        jtype = "L"

    salience = float((len1 + len2) * math.sin(math.radians(angle)))

    return {
        "x": float(pt[0]),
        "y": float(pt[1]),
        "type": jtype,
        "angle_deg": angle,
        "salience": salience,
        "line1_len": len1,
        "line2_len": len2,
    }


def cluster_junctions(cands: list[dict[str, Any]], radius_px: float) -> list[dict[str, Any]]:
    if not cands:
        return []

    priority = {"X": 3, "T": 2, "L": 1}
    cands = sorted(cands, key=lambda d: d["salience"], reverse=True)

    clusters: list[dict[str, Any]] = []

    for c in cands:
        p = np.array([c["x"], c["y"]], dtype=np.float64)
        assigned = False

        for cl in clusters:
            q = np.array([cl["x"], cl["y"]], dtype=np.float64)
            if np.linalg.norm(p - q) <= radius_px:
                w_old = cl["weight"]
                w_new = max(c["salience"], 1e-6)
                cl["x"] = float((cl["x"] * w_old + c["x"] * w_new) / (w_old + w_new))
                cl["y"] = float((cl["y"] * w_old + c["y"] * w_new) / (w_old + w_new))
                cl["weight"] = float(w_old + w_new)
                cl["salience"] = max(float(cl["salience"]), float(c["salience"]))
                cl["angle_deg"] = max(float(cl["angle_deg"]), float(c["angle_deg"]))
                cl["members"] += 1

                if priority[c["type"]] > priority[cl["type"]]:
                    cl["type"] = c["type"]

                assigned = True
                break

        if not assigned:
            clusters.append({
                "x": float(c["x"]),
                "y": float(c["y"]),
                "type": c["type"],
                "angle_deg": float(c["angle_deg"]),
                "salience": float(c["salience"]),
                "weight": float(max(c["salience"], 1e-6)),
                "members": 1,
            })

    return clusters


def detect_junctions(
    lines: np.ndarray,
    endpoint_tol_px: float,
    cluster_radius_px: float,
    min_angle_deg: float,
    max_junctions: int,
) -> list[dict[str, Any]]:
    cands: list[dict[str, Any]] = []

    if lines is None or len(lines) < 2:
        return []

    n = len(lines)

    for i in range(n):
        for j in range(i + 1, n):
            c = segment_intersection_candidate(
                lines[i],
                lines[j],
                endpoint_tol_px=endpoint_tol_px,
                min_angle_deg=min_angle_deg,
            )
            if c is not None:
                cands.append(c)

    clusters = cluster_junctions(cands, radius_px=cluster_radius_px)
    clusters = sorted(clusters, key=lambda d: d["salience"], reverse=True)

    if len(clusters) > max_junctions:
        clusters = clusters[:max_junctions]

    return clusters


def junction_stats(junctions: list[dict[str, Any]], image_shape: tuple[int, int]) -> dict[str, Any]:
    h, w = image_shape
    counts = {"L": 0, "T": 0, "X": 0}

    for j in junctions:
        if j["type"] in counts:
            counts[j["type"]] += 1

    total = len(junctions)
    salience_sum = float(sum(j["salience"] for j in junctions))
    density = float(total / max(1, h * w))

    vals = np.array([counts["L"], counts["T"], counts["X"]], dtype=np.float64)
    if vals.sum() > 0:
        p = vals / vals.sum()
        entropy = float(-(p[p > 0] * np.log(p[p > 0])).sum() / np.log(3))
    else:
        entropy = 0.0

    return {
        "junction_count": int(total),
        "L_count": int(counts["L"]),
        "T_count": int(counts["T"]),
        "X_count": int(counts["X"]),
        "junction_salience_sum": salience_sum,
        "junction_density": density,
        "junction_type_entropy": entropy,
    }


def draw_junction_overlay(rgb: np.ndarray, lines: np.ndarray, junctions: list[dict[str, Any]]) -> np.ndarray:
    out = rgb.copy()

    for x1, y1, x2, y2 in lines.astype(int):
        cv2.line(out, (x1, y1), (x2, y2), (255, 0, 0), 1, cv2.LINE_AA)

    colors = {
        "L": (255, 255, 0),
        "T": (0, 255, 255),
        "X": (255, 0, 255),
    }

    for j in junctions:
        x = int(round(j["x"]))
        y = int(round(j["y"]))
        col = colors.get(j["type"], (255, 255, 255))
        cv2.circle(out, (x, y), 6, col, 2, cv2.LINE_AA)
        cv2.putText(out, j["type"], (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)

    return out


def junction_canvas(shape: tuple[int, int], junctions: list[dict[str, Any]]) -> np.ndarray:
    h, w = shape
    can = np.zeros((h, w), dtype=np.uint8)
    for j in junctions:
        x = int(round(j["x"]))
        y = int(round(j["y"]))
        cv2.circle(can, (x, y), 4, 255, -1, cv2.LINE_AA)
    return can


def type_bar_image(stats: dict[str, Any], size: int = 512) -> np.ndarray:
    fig, ax = plt.subplots(figsize=(3, 3))
    labels = ["L", "T", "X"]
    vals = [stats["L_count"], stats["T_count"], stats["X_count"]]
    ax.bar(labels, vals)
    ax.set_title("junction types")
    ax.set_ylabel("count")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()

    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    img = cv2.cvtColor(buf, cv2.COLOR_RGBA2RGB)
    plt.close(fig)

    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def compute_junction_diag(image_path: Path, args: argparse.Namespace) -> dict[str, Any]:
    diag = s4c4a.compute_vector_diagnostic(image_path, args)
    lines = diag["lines_luma"]

    junctions = detect_junctions(
        lines=lines,
        endpoint_tol_px=args.endpoint_tol_px,
        cluster_radius_px=args.cluster_radius_px,
        min_angle_deg=args.min_angle_deg,
        max_junctions=args.max_junctions,
    )

    stats = junction_stats(junctions, diag["macro"].luma.shape)

    return {
        "vector_diag": diag,
        "lines": lines,
        "junctions": junctions,
        "junction_stats": stats,
        "junction_overlay": draw_junction_overlay(diag["macro"].rgb, lines, junctions),
        "junction_canvas": junction_canvas(diag["macro"].luma.shape, junctions),
        "type_bar": type_bar_image(stats, size=args.resize_size),
    }


def find_s4c4c_lsd_top1_source(
    token: int,
    raw_dir: Path,
    sat_df: pd.DataFrame,
    filename_index: dict[str, Path],
    fallback_sat_dirs: list[Path],
) -> Optional[dict[str, Any]]:
    pattern = f"s4c4c_token{token:04d}_luma_lsd_raw_lsd_scores_top*.csv"
    matches = sorted(raw_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)

    if not matches:
        return None

    df = pd.read_csv(matches[0])

    if len(df) == 0 or "lsd_rank" not in df.columns:
        return None

    row = df.sort_values(["lsd_rank", "phog_rank"]).iloc[0]

    if "candidate_image_path" in row and isinstance(row["candidate_image_path"], str):
        path = Path(row["candidate_image_path"])
    elif "row_pos" in row:
        sat_row = sat_df.iloc[int(row["row_pos"])]
        path = s4c0.get_row_path(sat_row, sat_df, filename_index, fallback_sat_dirs, kind="sat")
    else:
        return None

    if path is None or not Path(path).exists():
        return None

    return {
        "role": "lsd_top1_from_s4c4c",
        "label": f"LSD top1 tile {int(row['tile_id'])}\nerr={safe_float(row.get('center_error_m')):.1f}m",
        "path": Path(path),
        "tile_id": int(row["tile_id"]),
        "rank": int(row.get("phog_rank", -1)),
        "center_error_m": safe_float(row.get("center_error_m")),
        "score": safe_float(row.get("lsd_best_score")),
        "selection": "s4c4c_lsd_top1",
    }


def render_token_panel(token: int, rows: list[dict[str, Any]], out_path: Path) -> None:
    cols = [
        ("RGB", "rgb"),
        ("LSD lines + junctions", "junction_overlay"),
        ("macro contour", "macro"),
        ("junction canvas", "junction_canvas"),
        ("type counts", "type_bar"),
    ]

    fig, axes = plt.subplots(len(rows), len(cols), figsize=(3.25 * len(cols), 3.35 * len(rows)), squeeze=False)

    for r, item in enumerate(rows):
        jd = item["junction_diag"]
        vd = jd["vector_diag"]
        stats = jd["junction_stats"]

        imgs = {
            "rgb": vd["macro"].rgb,
            "junction_overlay": jd["junction_overlay"],
            "macro": vd["macro"].contour_canvas,
            "junction_canvas": jd["junction_canvas"],
            "type_bar": jd["type_bar"],
        }

        for c, (title, key) in enumerate(cols):
            ax = axes[r, c]
            ax.set_xticks([])
            ax.set_yticks([])

            img = imgs[key]
            if key in ["macro", "junction_canvas"]:
                ax.imshow(img, cmap="gray", vmin=0, vmax=255)
            else:
                ax.imshow(img)

            if r == 0:
                ax.set_title(title, fontsize=10)

            if c == 0:
                ax.set_ylabel(
                    f"{item['label']}\n"
                    f"J={stats['junction_count']} "
                    f"L/T/X={stats['L_count']}/{stats['T_count']}/{stats['X_count']}\n"
                    f"sal={stats['junction_salience_sum']:.0f}",
                    fontsize=8,
                )

    fig.suptitle(f"S4C.5A junction-aware luma-LSD frontend — token {token}", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=165)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--sequence", default="traj01")
    parser.add_argument("--tokens", default="1,40,60,90,100,129,166,269,516,905")
    parser.add_argument("--uav-index", default=str(DEFAULT_UAV_INDEX))
    parser.add_argument("--sat-index", default=str(DEFAULT_SAT_INDEX))
    parser.add_argument("--s4c1-ranked-dir", default=str(DEFAULT_S4C1_DIR))
    parser.add_argument("--s4c4c-dir", default=str(DEFAULT_S4C4C_DIR))

    # Match current S4C luma-LSD frontend settings.
    parser.add_argument("--preprocess", default="luma", choices=["gray", "luma", "clahe_luma"])
    parser.add_argument("--resize-size", type=int, default=512)
    parser.add_argument("--edge-method", default="sobel", choices=["sobel", "canny"])
    parser.add_argument("--blur-ksize", type=int, default=3)
    parser.add_argument("--threshold-mode", default="percentile", choices=["otsu", "percentile"])
    parser.add_argument("--threshold-percentile", type=float, default=65.0)
    parser.add_argument("--close-ksize", type=int, default=3)
    parser.add_argument("--open-ksize", type=int, default=1)
    parser.add_argument("--min-component-area", type=int, default=65)

    parser.add_argument("--min-line-length", type=float, default=24.0)
    parser.add_argument("--max-lines", type=int, default=120)
    parser.add_argument("--orientation-bins", type=int, default=18)

    # Junction settings.
    parser.add_argument("--endpoint-tol-px", type=float, default=12.0)
    parser.add_argument("--cluster-radius-px", type=float, default=10.0)
    parser.add_argument("--min-angle-deg", type=float, default=18.0)
    parser.add_argument("--max-junctions", type=int, default=160)
    parser.add_argument("--oracle-top-n", type=int, default=50)

    args = parser.parse_args()

    tokens = parse_tokens(args.tokens)

    OUT_FIG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_META_DIR.mkdir(parents=True, exist_ok=True)
    OUT_REPORT_DIR.mkdir(parents=True, exist_ok=True)

    uav_df = pd.read_csv(args.uav_index)
    sat_df = pd.read_csv(args.sat_index)

    uav_df = s4c4a.prepare_uav_df(uav_df, args.sequence)
    filename_index, fallback_uav_dirs, fallback_sat_dirs = s4c4a.build_filename_index(args.sequence)

    s4c1_dir = Path(args.s4c1_ranked_dir)
    s4c4c_dir = Path(args.s4c4c_dir)

    print("S4C.5A junction-aware luma-LSD frontend")
    print("---------------------------------------")
    print(f"Sequence: {args.sequence}")
    print(f"Tokens:   {tokens}")
    print(f"Junction endpoint tol: {args.endpoint_tol_px}px")
    print(f"Cluster radius:        {args.cluster_radius_px}px")
    print(f"Min angle:             {args.min_angle_deg} deg")
    print("")

    manifest_rows = []
    panel_paths = []

    for token in tokens:
        sources = s4c4a.build_sources_for_token(
            token=token,
            uav_df=uav_df,
            sat_df=sat_df,
            ranked_dir=s4c1_dir,
            filename_index=filename_index,
            fallback_uav_dirs=fallback_uav_dirs,
            fallback_sat_dirs=fallback_sat_dirs,
            args=args,
        )

        lsd_src = find_s4c4c_lsd_top1_source(
            token=token,
            raw_dir=s4c4c_dir,
            sat_df=sat_df,
            filename_index=filename_index,
            fallback_sat_dirs=fallback_sat_dirs,
        )

        if lsd_src is not None:
            # Put LSD top1 after PHOG top1.
            insert_idx = min(3, len(sources))
            sources = sources[:insert_idx] + [lsd_src] + sources[insert_idx:]

        rows_for_panel = []

        for src in sources:
            try:
                jd = compute_junction_diag(Path(src["path"]), args)
            except Exception as exc:
                print(f"[WARN] token {token} role={src['role']}: {exc}")
                continue

            stats = jd["junction_stats"]
            vd = jd["vector_diag"]

            rows_for_panel.append({
                "label": src["label"],
                "junction_diag": jd,
            })

            row = {
                "sequence": args.sequence,
                "token": token,
                "role": src["role"],
                "selection": src.get("selection"),
                "tile_id": src.get("tile_id"),
                "rank": src.get("rank"),
                "center_error_m": src.get("center_error_m"),
                "score": src.get("score", src.get("score_cosine")),
                "image_path": str(src["path"]),
                "luma_lsd_line_count": vd["stats_luma"]["line_count"],
                "luma_lsd_total_line_length_px": vd["stats_luma"]["total_line_length_px"],
                **stats,
            }

            manifest_rows.append(row)

        if rows_for_panel:
            fig_path = OUT_FIG_DIR / f"s4c5a_token{token:04d}_junction_lsd_frontend.png"
            render_token_panel(token, rows_for_panel, fig_path)
            panel_paths.append(str(fig_path))
            print(f"[OK] token {token}: saved {fig_path}")

    manifest_df = pd.DataFrame(manifest_rows)

    manifest_csv = OUT_META_DIR / "s4c5a_junction_lsd_frontend_manifest.csv"
    summary_json = OUT_REPORT_DIR / "s4c5a_junction_lsd_frontend_summary.json"

    manifest_df.to_csv(manifest_csv, index=False)

    summary = {
        "stage": "S4C.5A_junction_aware_luma_LSD_frontend",
        "sequence": args.sequence,
        "tokens": tokens,
        "manifest_csv": str(manifest_csv),
        "summary_json": str(summary_json),
        "figures": panel_paths,
        "settings": vars(args),
        "notes": [
            "Diagnostic only; no reranking performed.",
            "Junctions are classified from luma-LSD line intersections as L/T/X.",
            "GT/oracle/error fields are used only for post-hoc diagnosis.",
        ],
    }

    if len(manifest_df) > 0:
        role_cols = [
            "role",
            "junction_count",
            "L_count",
            "T_count",
            "X_count",
            "junction_salience_sum",
            "junction_type_entropy",
            "luma_lsd_line_count",
            "luma_lsd_total_line_length_px",
            "center_error_m",
        ]
        existing = [c for c in role_cols if c in manifest_df.columns]
        summary["role_mean_metrics"] = (
            manifest_df[existing]
            .groupby("role")
            .mean(numeric_only=True)
            .reset_index()
            .to_dict(orient="records")
        )

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(json_safe(summary), f, indent=2)

    print("")
    print("S4C.5A complete")
    print("----------------")
    print(f"Manifest CSV: {manifest_csv}")
    print(f"Summary JSON: {summary_json}")
    print(f"Figures:      {OUT_FIG_DIR}")

    if len(manifest_df) > 0:
        print("")
        print("Compact role averages")
        print("---------------------")
        cols = [
            "role",
            "junction_count",
            "L_count",
            "T_count",
            "X_count",
            "junction_salience_sum",
            "junction_type_entropy",
            "luma_lsd_line_count",
            "luma_lsd_total_line_length_px",
            "center_error_m",
        ]
        cols = [c for c in cols if c in manifest_df.columns]
        print(manifest_df[cols].groupby("role").mean(numeric_only=True).to_string())


if __name__ == "__main__":
    main()
