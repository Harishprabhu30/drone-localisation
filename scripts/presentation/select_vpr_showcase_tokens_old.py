#!/usr/bin/env python3
"""
Select presentation-ready VPR showcase tokens by joining classical failure analysis
with DINOv2-VLAD rescue / failure diagnostics.

Purpose
-------
This script answers questions such as:

1. Which tokens failed in the classical PHOG/LSD pipeline but were recovered by
   DINOv2-VLAD?
2. Which tokens best illustrate ORB/HOG/PHOG/LSD being fooled?
3. Which agricultural / open-field tokens remain hardest even for DINOv2-VLAD?
4. Which orientation-range tokens, for example 599-605, are supported by actual
   DINO and classical metrics?
5. Which near-miss tokens motivate multi-tile / 3x3-region map reasoning?

Ground-truth rule
-----------------
Evaluation columns such as eval_error_m, center_error_m and oracle labels are
used only after retrieval for diagnosis and presentation selection. They are not
used to rank inside the original estimators.

Run from repository root:

    export PYTHONPATH=$PWD/src
    python scripts/presentation/select_vpr_showcase_tokens.py \
      --repo-root "$PWD" \
      --out-dir outputs/presentation/token_selection \
      --make-panels \
      --top-n-panels 24

Outputs
-------
outputs/presentation/token_selection/
├── vpr_showcase_token_master.csv
├── hero_candidates_classical_fail_dino_success.csv
├── classical_failure_tokens.csv
├── agricultural_hard_tokens.csv
├── orientation_range_tokens.csv
├── near_miss_tokens.csv
├── vpr_showcase_token_selection_summary.json
├── figures/
│   ├── hero_classical_vs_dino_scatter.svg/.png
│   ├── recommended_role_counts.svg/.png
│   ├── dino_failure_scene_breakdown.svg/.png
│   └── top_hero_candidate_scores.svg/.png
└── panels/
    ├── hero_candidates/
    ├── agricultural_hard/
    ├── orientation_range/
    └── near_miss/
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont


PALETTE = {
    "bg": "#0B0C0E",
    "panel": "#15171A",
    "raised": "#1C1F23",
    "text": "#F3EFE6",
    "muted": "#A9A7A1",
    "grid": "#34363A",
    "mustard": "#D4A72C",
    "mustard_bright": "#F0C75E",
    "coral": "#D8685A",
    "teal": "#67A9A3",
    "neutral": "#8A8F98",
    "ivory": "#E8E2D6",
}

DEFAULTS = {
    "s4c6_by_token": "outputs/satloc/metadata/s4c_macrocontour_phog_chamfer/s4c6_failure_analysis/s4c6c_failure_groups_by_token.csv",
    "s7b4_failures": "outputs/satloc/metadata/s7b_failure_rescue/s7b4_failure_token_diagnostics_center_resize_k32_img224_top100.csv",
    "center_candidates": "outputs/satloc/metadata/s7b_dinov2_vlad/s7b2_dinov2_vlad_candidate_scores_torch_hub_dinov2_vits14_vlad_k32_img224_top100_q263_sat8625_cb500_full263_k32_cb500_img224.csv",
    "resize_candidates": "outputs/satloc/metadata/s7b_dinov2_vlad/s7b2_dinov2_vlad_candidate_scores_torch_hub_dinov2_vits14_vlad_k32_img224_top100_q263_sat8625_cb500_full263_k32_cb500_img224_resize_square.csv",
    "satellite_index": "outputs/satloc/metadata/satellite_tiles_index_enriched.csv",
    "query_manifest": "outputs/satloc/metadata/s7_retrieval_upgrade/s7_0_query_manifest.csv",
    "scene_labels": "outputs/satloc/metadata/s7_retrieval_upgrade/s7_scene_labels_canonical_traj01.csv",
    "s7b4_visual_panels_glob": "outputs/satloc/figures/s7b_failure_rescue/visual_diagnostics/center_resize_k32_img224_top100/s7b4_visual_token_*.png",
    "s4c4c_panels_glob": "outputs/satloc/figures/s4c_macrocontour_phog_chamfer/s4c4_vector_skeleton/s4c4c_luma_lsd_rerank/s4c4c_token*_luma_lsd_panel_top5.png",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Select VPR showcase hero/failure tokens.")
    p.add_argument("--repo-root", type=Path, default=Path("."))
    p.add_argument("--out-dir", type=Path, default=Path("outputs/presentation/token_selection"))
    p.add_argument("--threshold-m", type=float, default=40.0)
    p.add_argument("--near-threshold-m", type=float, default=100.0)
    p.add_argument("--orientation-range", default="599-605", help="Token range such as 599-605, or comma list.")
    p.add_argument("--top-n-panels", type=int, default=18)
    p.add_argument("--make-panels", action="store_true")
    p.add_argument("--panel-width", type=int, default=420)
    p.add_argument("--panel-height", type=int, default=320)

    for key, value in DEFAULTS.items():
        p.add_argument(f"--{key.replace('_', '-')}", type=Path, default=Path(value))
    return p.parse_args()


def resolve(root: Path, p: Path | str) -> Path:
    p = Path(str(p))
    return p if p.is_absolute() else root / p


def read_csv(path: Path, name: str, required: bool = True) -> pd.DataFrame:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Missing {name}: {path}")
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def parse_tokens(spec: str) -> set[int]:
    out: set[int] = set()
    for part in [p.strip() for p in spec.split(",") if p.strip()]:
        if "-" in part:
            a, b = part.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return out


def safe_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if pd.isna(v):
        return False
    return str(v).strip().lower() in {"true", "1", "yes", "y"}


def safe_float(v: Any, default: float = np.nan) -> float:
    try:
        if pd.isna(v):
            return default
        return float(v)
    except Exception:
        return default


def token_from_panel_name(path: Path) -> int | None:
    m = re.search(r"token[_-]?(\d+)", path.name)
    if not m:
        return None
    return int(m.group(1))


def map_existing_panels(root: Path, pattern: str) -> dict[int, str]:
    out: dict[int, str] = {}
    for p in sorted(root.glob(pattern)):
        t = token_from_panel_name(p)
        if t is not None and t not in out:
            out[t] = str(p)
    return out


def first_existing_image(root: Path, values: Iterable[Any]) -> str:
    for v in values:
        if v is None:
            continue
        try:
            if pd.isna(v):
                continue
        except Exception:
            pass
        s = str(v).strip()
        if not s:
            continue
        p = Path(s)
        candidates = [p] if p.is_absolute() else [root / p, p]
        for c in candidates:
            if c.exists():
                return str(c)
    return ""


def prepare_classical(df: pd.DataFrame) -> pd.DataFrame:
    keep = [
        "token", "failure_group", "phog_top1_error_m", "lsd_top1_error_m",
        "oracle_best_top50_error_m", "phog_top1_tile_id", "lsd_top1_tile_id",
        "oracle_best_top50_tile_id", "rot_best_error_m", "rot_best_angle_deg",
        "uav_junction_count", "uav_luma_lsd_line_count", "selection_gap_to_oracle_m",
    ]
    have = [c for c in keep if c in df.columns]
    out = df[have].copy()
    out = out.rename(columns={"failure_group": "classical_failure_group"})
    out["token"] = pd.to_numeric(out["token"], errors="raise").astype(int)
    return out


def prepare_dino(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["token"] = pd.to_numeric(out["token"], errors="raise").astype(int)
    out = out.rename(columns={"failure_group": "dino_failure_group"})
    for c in [
        "anchor_hit_at_100", "center_hit_at_100", "resize_hit_at_100",
        "any_stream_hit_at_100", "rescued_by_non_anchor_at_100",
    ]:
        if c in out.columns:
            out[c] = out[c].map(safe_bool)
    return out


def add_query_paths(master: pd.DataFrame, root: Path, query_manifest: pd.DataFrame, candidates: list[pd.DataFrame]) -> pd.DataFrame:
    qcols = [c for c in ["token0_id", "token", "image_path", "image_path_resolved", "image_path_relative"] if c in query_manifest.columns]
    if qcols:
        q = query_manifest[qcols].copy()
        token_col = "token0_id" if "token0_id" in q.columns else "token"
        q = q.rename(columns={token_col: "token"})
        q["token"] = pd.to_numeric(q["token"], errors="coerce").astype("Int64")
        q = q.dropna(subset=["token"]).drop_duplicates("token")
        q["query_image_path_from_manifest"] = q.apply(
            lambda r: first_existing_image(root, [r.get("image_path_resolved"), r.get("image_path"), r.get("image_path_relative")]), axis=1
        )
        master = master.merge(q[["token", "query_image_path_from_manifest"]], on="token", how="left")
    else:
        master["query_image_path_from_manifest"] = ""

    query_from_candidates: dict[int, str] = {}
    for df in candidates:
        if "query_image_path" not in df.columns or "token" not in df.columns:
            continue
        for token, g in df.groupby("token"):
            p = first_existing_image(root, g.sort_values("rank")["query_image_path"].tolist())
            if p and int(token) not in query_from_candidates:
                query_from_candidates[int(token)] = p
    master["query_image_path"] = master["token"].map(query_from_candidates).fillna(master["query_image_path_from_manifest"].fillna(""))
    return master


def candidate_top_and_best(root: Path, df: pd.DataFrame, stream_name: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["token"])
    d = df.copy()
    d["token"] = pd.to_numeric(d["token"], errors="raise").astype(int)
    d["rank"] = pd.to_numeric(d["rank"], errors="coerce")
    d["eval_error_m"] = pd.to_numeric(d["eval_error_m"], errors="coerce")
    rows = []
    for token, g in d.groupby("token"):
        g = g.sort_values("rank")
        top = g.iloc[0]
        best = g.sort_values(["eval_error_m", "rank"]).iloc[0]
        rows.append({
            "token": int(token),
            f"{stream_name}_top1_tile_id": top.get("tile_id", ""),
            f"{stream_name}_top1_error_m": safe_float(top.get("eval_error_m")),
            f"{stream_name}_top1_rank": int(top.get("rank")) if pd.notna(top.get("rank")) else np.nan,
            f"{stream_name}_top1_sat_path": first_existing_image(root, [top.get("sat_image_path")]),
            f"{stream_name}_best_tile_id": best.get("tile_id", ""),
            f"{stream_name}_best_error_m": safe_float(best.get("eval_error_m")),
            f"{stream_name}_best_rank": int(best.get("rank")) if pd.notna(best.get("rank")) else np.nan,
            f"{stream_name}_best_sat_path": first_existing_image(root, [best.get("sat_image_path")]),
        })
    return pd.DataFrame(rows)


def classify_and_score(master: pd.DataFrame, threshold: float, near_threshold: float, orientation_tokens: set[int]) -> pd.DataFrame:
    df = master.copy()
    df["has_classical_record"] = df["classical_failure_group"].notna()
    df["classical_failed"] = df["classical_failure_group"].astype(str).isin([
        "candidate_pool_failure",
        "selection_failure_correct_in_pool",
        "weak_pool_near_candidate",
        "lsd_destroyed_phog_success",
    ])
    if "lsd_top1_error_m" in df.columns:
        df["classical_failed"] = df["classical_failed"] | pd.to_numeric(df["lsd_top1_error_m"], errors="coerce").gt(threshold)

    df["dino_success_any"] = df.get("any_stream_hit_at_100", False).map(safe_bool) if "any_stream_hit_at_100" in df.columns else False
    df["dino_center_success"] = df.get("center_hit_at_100", False).map(safe_bool) if "center_hit_at_100" in df.columns else False
    df["dino_resize_success"] = df.get("resize_hit_at_100", False).map(safe_bool) if "resize_hit_at_100" in df.columns else False
    df["dino_resize_rescue"] = df.get("rescued_by_non_anchor_at_100", False).map(safe_bool) if "rescued_by_non_anchor_at_100" in df.columns else False
    df["dino_unresolved"] = ~df["dino_success_any"]

    dino_best = pd.to_numeric(df.get("best_error_any_stream_available_m", np.nan), errors="coerce")
    df["near_pool_miss"] = df["dino_failure_group"].astype(str).eq("near_pool_miss_40_100m") | ((dino_best > threshold) & (dino_best <= near_threshold))
    df["pool_failure_or_far_ambiguity"] = df["dino_failure_group"].astype(str).eq("pool_failure_or_far_ambiguity")
    df["agricultural_open_field"] = df["primary_scene"].astype(str).eq("agricultural_open_field")
    df["orientation_range"] = df["token"].isin(orientation_tokens)
    df["classical_fail_dino_success"] = df["classical_failed"] & df["dino_success_any"]

    def role(r: pd.Series) -> str:
        if bool(r["classical_fail_dino_success"]):
            return "hero_candidate"
        if bool(r["dino_resize_rescue"]):
            return "fov_resize_rescue"
        if bool(r["agricultural_open_field"]) and (bool(r["dino_unresolved"]) or bool(r["pool_failure_or_far_ambiguity"])):
            return "hardest_failure"
        if bool(r["pool_failure_or_far_ambiguity"]):
            return "hardest_failure"
        if bool(r["near_pool_miss"]):
            return "near_miss_multitile"
        if bool(r["orientation_range"]):
            return "orientation_candidate"
        if bool(r["classical_failed"]):
            return "classical_failure"
        return "supporting_case"

    def score(r: pd.Series) -> float:
        s = 0.0
        if bool(r["classical_fail_dino_success"]): s += 120
        if bool(r["dino_resize_rescue"]): s += 80
        if bool(r["classical_failed"]): s += 55
        if bool(r["dino_success_any"]): s += 50
        if bool(r["near_pool_miss"]): s += 35
        if bool(r["orientation_range"]): s += 35
        if bool(r["agricultural_open_field"]): s += 30
        if bool(r["pool_failure_or_far_ambiguity"]): s += 45
        if bool(r.get("has_s7b4_visual_panel", False)): s += 18
        if bool(r.get("has_s4c4c_panel", False)): s += 12
        be = safe_float(r.get("best_error_any_stream_available_m"))
        if np.isfinite(be):
            s += max(0.0, 40.0 - min(be, 40.0)) / 4.0
            if be > near_threshold: s -= 20
        if str(r.get("primary_scene", "")) == "agricultural_open_field" and bool(r["classical_fail_dino_success"]):
            # Rare but maybe visually repetitive; keep it but not as the only hero.
            s -= 10
        return s

    df["recommended_role"] = df.apply(role, axis=1)
    df["selection_priority_score"] = df.apply(score, axis=1)
    return df.sort_values(["selection_priority_score", "token"], ascending=[False, True])


def style_axes(ax):
    ax.set_facecolor("none")
    ax.tick_params(colors=PALETTE["muted"])
    ax.xaxis.label.set_color(PALETTE["text"])
    ax.yaxis.label.set_color(PALETTE["text"])
    ax.title.set_color(PALETTE["text"])
    for spine in ax.spines.values():
        spine.set_color(PALETTE["grid"])
    ax.grid(color=PALETTE["grid"], alpha=0.35)


def save_fig(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.patch.set_alpha(0)
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight", transparent=True, pad_inches=0.05)
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight", transparent=True, pad_inches=0.05)
    plt.close(fig)


def plot_hero_scatter(df: pd.DataFrame, out: Path):
    d = df[df["has_classical_record"]].copy()
    if d.empty:
        return
    d["classical_error_m"] = pd.to_numeric(d.get("lsd_top1_error_m"), errors="coerce")
    d["dino_best_error_m"] = pd.to_numeric(d.get("best_error_any_stream_available_m"), errors="coerce")
    d = d[np.isfinite(d["classical_error_m"]) & np.isfinite(d["dino_best_error_m"])]
    if d.empty:
        return

    fig, ax = plt.subplots(figsize=(12.8, 7.2))
    colors = np.where(d["classical_fail_dino_success"], PALETTE["mustard"],
             np.where(d["dino_success_any"], PALETTE["teal"], PALETTE["neutral"]))
    ax.scatter(d["classical_error_m"], d["dino_best_error_m"], s=70, c=colors, edgecolors=PALETTE["ivory"], linewidths=0.4, alpha=0.9)
    ax.axhline(40, color=PALETTE["mustard_bright"], linestyle="--", linewidth=1.5, alpha=0.8)
    ax.axvline(40, color=PALETTE["mustard_bright"], linestyle="--", linewidth=1.5, alpha=0.8)
    ax.set_xscale("symlog", linthresh=40)
    ax.set_yscale("symlog", linthresh=40)
    ax.set_xlabel("Classical luma-LSD top-1 error [m]")
    ax.set_ylabel("Best DINOv2-VLAD candidate error [m]")
    ax.set_title("Hero-token search: classical failure vs learned retrieval rescue")
    style_axes(ax)

    top = d[d["classical_fail_dino_success"]].sort_values("selection_priority_score", ascending=False).head(12)
    for _, r in top.iterrows():
        ax.annotate(str(int(r["token"])), (r["classical_error_m"], r["dino_best_error_m"]),
                    textcoords="offset points", xytext=(6, 5), fontsize=9, color=PALETTE["text"])
    save_fig(fig, out)


def plot_counts(df: pd.DataFrame, out: Path):
    c = df["recommended_role"].value_counts().sort_values()
    fig, ax = plt.subplots(figsize=(12.8, 7.2))
    ax.barh(c.index, c.values, color=PALETTE["mustard"])
    ax.set_xlabel("Token count")
    ax.set_title("Presentation token roles")
    for i, v in enumerate(c.values):
        ax.text(v + 0.5, i, str(int(v)), va="center", color=PALETTE["text"], fontsize=10)
    style_axes(ax)
    save_fig(fig, out)


def plot_scene_breakdown(df: pd.DataFrame, out: Path):
    cols = ["primary_scene", "dino_failure_group", "token"]
    if not set(cols).issubset(df.columns):
        return
    d = df.groupby(["primary_scene", "dino_failure_group"]).size().reset_index(name="count")
    if d.empty:
        return
    pivot = d.pivot(index="primary_scene", columns="dino_failure_group", values="count").fillna(0)
    order = pivot.sum(axis=1).sort_values().index
    pivot = pivot.loc[order]
    fig, ax = plt.subplots(figsize=(12.8, 7.2))
    left = np.zeros(len(pivot))
    colors = [PALETTE["neutral"], PALETTE["mustard"], PALETTE["teal"], PALETTE["coral"], PALETTE["ivory"]]
    for i, col in enumerate(pivot.columns):
        ax.barh(pivot.index, pivot[col].values, left=left, label=col, color=colors[i % len(colors)], alpha=0.9)
        left += pivot[col].values
    ax.set_xlabel("Token count")
    ax.set_title("DINOv2-VLAD outcome by scene")
    ax.legend(fontsize=8, frameon=False, labelcolor=PALETTE["text"], loc="lower right")
    style_axes(ax)
    save_fig(fig, out)


def plot_top_scores(df: pd.DataFrame, out: Path, n: int = 20):
    d = df[df["recommended_role"].isin(["hero_candidate", "fov_resize_rescue"])].head(n).copy()
    if d.empty:
        return
    d["label"] = d.apply(lambda r: f"{int(r['token'])} · {str(r.get('primary_scene',''))[:18]}", axis=1)
    d = d.sort_values("selection_priority_score")
    fig, ax = plt.subplots(figsize=(12.8, 7.2))
    ax.barh(d["label"], d["selection_priority_score"], color=PALETTE["mustard"])
    ax.set_xlabel("Selection priority score")
    ax.set_title("Top candidate tokens for the main story slide")
    for i, (_, r) in enumerate(d.iterrows()):
        note = "DINO rescue" if r["recommended_role"] == "hero_candidate" else "resize rescue"
        ax.text(r["selection_priority_score"] + 1, i, note, va="center", color=PALETTE["text"], fontsize=9)
    style_axes(ax)
    save_fig(fig, out)


def load_img(path: str, size: tuple[int, int]) -> Image.Image:
    w, h = size
    if not path or not Path(path).exists():
        img = Image.new("RGB", size, PALETTE["raised"])
        d = ImageDraw.Draw(img)
        d.text((16, 16), "missing", fill=PALETTE["muted"])
        return img
    img = Image.open(path).convert("RGB")
    img.thumbnail(size)
    can = Image.new("RGB", size, PALETTE["bg"])
    can.paste(img, ((w - img.width)//2, (h - img.height)//2))
    return can


def draw_title(img: Image.Image, title: str, subtitle: str = "", border: str | None = None) -> Image.Image:
    title_h = 66
    out = Image.new("RGB", (img.width, img.height + title_h), PALETTE["panel"])
    out.paste(img, (0, title_h))
    d = ImageDraw.Draw(out)
    d.text((12, 10), title[:54], fill=PALETTE["text"])
    if subtitle:
        d.text((12, 34), subtitle[:70], fill=PALETTE["muted"])
    if border:
        for k in range(4):
            d.rectangle([k, k, out.width - 1 - k, out.height - 1 - k], outline=border)
    return out


def feature_views(query_path: str, size: tuple[int, int]) -> list[Image.Image]:
    panels: list[Image.Image] = []
    if not query_path or not Path(query_path).exists():
        return panels
    try:
        import cv2
        arr = np.asarray(Image.open(query_path).convert("RGB"))
        small = np.asarray(load_img(query_path, size))
        gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
        orb = cv2.ORB_create(nfeatures=600)
        kps = orb.detect(gray, None)
        orb_img = cv2.drawKeypoints(small, kps, None, flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)
        panels.append(draw_title(Image.fromarray(orb_img), "ORB keypoints", "classical local cues"))
        sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(sobelx**2 + sobely**2)
        mag = (255 * (mag / max(1e-6, mag.max()))).astype(np.uint8)
        panels.append(draw_title(Image.fromarray(np.dstack([mag]*3)), "Sobel structure", "edges and boundaries"))
        try:
            lsd = cv2.createLineSegmentDetector(0)
            lines = lsd.detect(gray)[0]
            line_img = small.copy()
            if lines is not None:
                for line in lines[:180]:
                    x1, y1, x2, y2 = map(int, line[0])
                    cv2.line(line_img, (x1, y1), (x2, y2), (212, 167, 44), 1)
            panels.append(draw_title(Image.fromarray(line_img), "LSD line structure", "longer geometry cues"))
        except Exception:
            pass
    except Exception:
        return panels
    return panels


def tile_path_from_index(sat_index: pd.DataFrame, root: Path, tile_id: Any) -> str:
    if tile_id is None or pd.isna(tile_id):
        return ""
    sid = str(int(float(tile_id))) if str(tile_id).replace(".", "", 1).isdigit() else str(tile_id)
    if "tile_index" not in sat_index.columns:
        return ""
    key_series = sat_index["tile_index"].astype(str)
    matches = sat_index[key_series == sid]
    if matches.empty:
        # Some CSVs may store zero-padded/string tile ids differently.
        matches = sat_index[pd.to_numeric(sat_index["tile_index"], errors="coerce") == pd.to_numeric(pd.Series([tile_id]), errors="coerce").iloc[0]]
    if matches.empty:
        return ""
    row = matches.iloc[0]
    return first_existing_image(root, [row.get("tile_path"), row.get("tile_path_relative"), row.get("ref_rel_path")])


def make_token_panel(row: pd.Series, sat_index: pd.DataFrame, root: Path, out_path: Path, panel_size=(420, 320)) -> None:
    token = int(row["token"])
    panels: list[Image.Image] = []
    qpath = str(row.get("query_image_path", ""))
    panels.append(draw_title(load_img(qpath, panel_size), f"Token {token}: UAV query", str(row.get("primary_scene", "")), PALETTE["ivory"]))

    phog_path = tile_path_from_index(sat_index, root, row.get("phog_top1_tile_id"))
    lsd_path = tile_path_from_index(sat_index, root, row.get("lsd_top1_tile_id"))
    oracle_path = tile_path_from_index(sat_index, root, row.get("oracle_best_top50_tile_id"))

    if phog_path:
        panels.append(draw_title(load_img(phog_path, panel_size), "PHOG top-1", f"err={safe_float(row.get('phog_top1_error_m')):.1f} m", PALETTE["coral"]))
    if lsd_path:
        panels.append(draw_title(load_img(lsd_path, panel_size), "luma-LSD top-1", f"err={safe_float(row.get('lsd_top1_error_m')):.1f} m", PALETTE["coral"] if safe_float(row.get('lsd_top1_error_m')) > 40 else PALETTE["mustard"]))
    if oracle_path:
        panels.append(draw_title(load_img(oracle_path, panel_size), "Best classical pool candidate", f"err={safe_float(row.get('oracle_best_top50_error_m')):.1f} m", PALETTE["mustard"]))

    center_top = str(row.get("center_top1_sat_path", ""))
    center_best = str(row.get("center_best_sat_path", ""))
    resize_best = str(row.get("resize_best_sat_path", ""))
    if center_top:
        panels.append(draw_title(load_img(center_top, panel_size), "DINO center top-1", f"err={safe_float(row.get('center_top1_error_m')):.1f} m", PALETTE["teal"] if safe_float(row.get('center_top1_error_m')) <= 40 else PALETTE["neutral"]))
    if center_best:
        panels.append(draw_title(load_img(center_best, panel_size), "DINO center best@100", f"rank={safe_float(row.get('center_best_rank')):.0f}, err={safe_float(row.get('center_best_error_m')):.1f} m", PALETTE["mustard"] if safe_float(row.get('center_best_error_m')) <= 40 else PALETTE["neutral"]))
    if resize_best and resize_best != center_best:
        panels.append(draw_title(load_img(resize_best, panel_size), "DINO resize best@100", f"rank={safe_float(row.get('resize_best_rank')):.0f}, err={safe_float(row.get('resize_best_error_m')):.1f} m", PALETTE["mustard"] if safe_float(row.get('resize_best_error_m')) <= 40 else PALETTE["neutral"]))

    panels.extend(feature_views(qpath, panel_size))

    cols = 3
    cw = max(p.width for p in panels)
    ch = max(p.height for p in panels)
    rows = int(math.ceil(len(panels) / cols))
    gap = 18
    title_h = 70
    W = cols * cw + (cols - 1) * gap + 40
    H = title_h + rows * ch + (rows - 1) * gap + 40
    canvas = Image.new("RGB", (W, H), PALETTE["bg"])
    d = ImageDraw.Draw(canvas)
    d.text((24, 22), f"VPR showcase token {token} · {row.get('recommended_role','')}", fill=PALETTE["text"])
    d.text((24, 46), f"classical={row.get('classical_failure_group','')} · dino={row.get('dino_failure_group','')}", fill=PALETTE["muted"])
    for i, p in enumerate(panels):
        r, c = divmod(i, cols)
        x = 20 + c * (cw + gap)
        y = title_h + r * (ch + gap)
        canvas.paste(p, (x, y))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def write_group(df: pd.DataFrame, out: Path, filename: str, mask: pd.Series) -> pd.DataFrame:
    sub = df[mask].copy().sort_values(["selection_priority_score", "token"], ascending=[False, True])
    sub.to_csv(out / filename, index=False)
    return sub


def main() -> int:
    args = parse_args()
    root = args.repo_root.resolve()
    out = resolve(root, args.out_dir)
    fig_dir = out / "figures"
    panel_dir = out / "panels"
    out.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    s4 = prepare_classical(read_csv(resolve(root, args.s4c6_by_token), "s4c6_by_token", required=False))
    dino = prepare_dino(read_csv(resolve(root, args.s7b4_failures), "s7b4_failures"))
    center = read_csv(resolve(root, args.center_candidates), "center_candidates")
    resize = read_csv(resolve(root, args.resize_candidates), "resize_candidates")
    sat_index = read_csv(resolve(root, args.satellite_index), "satellite_index")
    query_manifest = read_csv(resolve(root, args.query_manifest), "query_manifest", required=False)

    master = dino.merge(s4, on="token", how="left")
    center_summary = candidate_top_and_best(root, center, "center")
    resize_summary = candidate_top_and_best(root, resize, "resize")
    master = master.merge(center_summary, on="token", how="left").merge(resize_summary, on="token", how="left")
    master = add_query_paths(master, root, query_manifest, [center, resize])

    s7_panels = map_existing_panels(root, str(args.s7b4_visual_panels_glob))
    s4_panels = map_existing_panels(root, str(args.s4c4c_panels_glob))
    master["s7b4_visual_panel_path"] = master["token"].map(s7_panels).fillna("")
    master["s4c4c_panel_path"] = master["token"].map(s4_panels).fillna("")
    master["has_s7b4_visual_panel"] = master["s7b4_visual_panel_path"].astype(bool)
    master["has_s4c4c_panel"] = master["s4c4c_panel_path"].astype(bool)

    orientation_tokens = parse_tokens(args.orientation_range)
    master = classify_and_score(master, args.threshold_m, args.near_threshold_m, orientation_tokens)

    master_csv = out / "vpr_showcase_token_master.csv"
    master.to_csv(master_csv, index=False)

    hero = write_group(master, out, "hero_candidates_classical_fail_dino_success.csv", master["recommended_role"].eq("hero_candidate") | master["dino_resize_rescue"])
    classical_fail = write_group(master, out, "classical_failure_tokens.csv", master["classical_failed"])
    agricultural = write_group(master, out, "agricultural_hard_tokens.csv", master["agricultural_open_field"])
    orientation = write_group(master, out, "orientation_range_tokens.csv", master["orientation_range"])
    near_miss = write_group(master, out, "near_miss_tokens.csv", master["near_pool_miss"])

    plot_hero_scatter(master, fig_dir / "hero_classical_vs_dino_scatter")
    plot_counts(master, fig_dir / "recommended_role_counts")
    plot_scene_breakdown(master, fig_dir / "dino_failure_scene_breakdown")
    plot_top_scores(master, fig_dir / "top_hero_candidate_scores", n=20)

    panel_outputs: dict[str, list[str]] = {}
    if args.make_panels:
        groups = {
            "hero_candidates": hero,
            "agricultural_hard": agricultural,
            "orientation_range": orientation,
            "near_miss": near_miss,
        }
        for name, g in groups.items():
            paths = []
            for _, row in g.head(args.top_n_panels).iterrows():
                path = panel_dir / name / f"token_{int(row['token']):04d}_{row['recommended_role']}.png"
                make_token_panel(row, sat_index, root, path, (args.panel_width, args.panel_height))
                paths.append(str(path.relative_to(root)))
            panel_outputs[name] = paths

    summary = {
        "status": "COMPLETE",
        "repo_root": str(root),
        "threshold_m": args.threshold_m,
        "near_threshold_m": args.near_threshold_m,
        "orientation_range": args.orientation_range,
        "tokens_total": int(master["token"].nunique()),
        "has_classical_record": int(master["has_classical_record"].sum()),
        "hero_candidates": int(len(hero)),
        "classical_failures": int(len(classical_fail)),
        "agricultural_cases": int(len(agricultural)),
        "orientation_range_cases": int(len(orientation)),
        "near_miss_cases": int(len(near_miss)),
        "role_counts": master["recommended_role"].value_counts().to_dict(),
        "outputs": {
            "master_csv": str(master_csv.relative_to(root)),
            "hero_candidates_csv": str((out / "hero_candidates_classical_fail_dino_success.csv").relative_to(root)),
            "classical_failure_csv": str((out / "classical_failure_tokens.csv").relative_to(root)),
            "agricultural_hard_csv": str((out / "agricultural_hard_tokens.csv").relative_to(root)),
            "orientation_range_csv": str((out / "orientation_range_tokens.csv").relative_to(root)),
            "near_miss_csv": str((out / "near_miss_tokens.csv").relative_to(root)),
            "figures_dir": str(fig_dir.relative_to(root)),
            "panels": panel_outputs,
        },
        "note": "Use CSVs for objective selection, then visually inspect panels before freezing final slide tokens.",
    }
    summary_path = out / "vpr_showcase_token_selection_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    print("VPR showcase token selection")
    print("----------------------------")
    print(f"Master CSV:             {master_csv}")
    print(f"Hero candidates:        {len(hero)}")
    print(f"Classical failures:     {len(classical_fail)}")
    print(f"Agricultural cases:     {len(agricultural)}")
    print(f"Orientation cases:      {len(orientation)}")
    print(f"Near-miss cases:        {len(near_miss)}")
    print(f"Summary JSON:           {summary_path}")
    if args.make_panels:
        print(f"Panels dir:             {panel_dir}")
    print("\nTop hero candidates:")
    cols = [
        "token", "selection_priority_score", "primary_scene", "recommended_role",
        "classical_failure_group", "dino_failure_group", "best_error_any_stream_available_m",
        "best_rank_any_stream_available", "best_stream_available",
        "has_s7b4_visual_panel", "has_s4c4c_panel",
    ]
    cols = [c for c in cols if c in hero.columns]
    if len(hero):
        print(hero[cols].head(20).to_string(index=False))
    else:
        print("No direct classical-failure + DINO-success hero candidates found. Review fov_resize_rescue and near_miss outputs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
