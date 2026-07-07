#!/usr/bin/env python3
'''
code to run:
export PYTHONPATH=$PWD/src

python scripts/satloc/s4b_1g_cell_texture_suppressed_rerank.py \
  --sequence traj01 \
  --token 1 \
  --preprocess luma \
  --resize-size 512 \
  --cells 8 \
  --bins 9 \
  --edge-threshold 0.05 \
  --blend-original 0.30

'''
from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path.cwd()

DEFAULT_UAV_INDEX = Path("outputs/satloc/metadata/uav_frames_index_enriched.csv")
DEFAULT_UAV_DIR = Path("data/raw/satloc/part_1/UAV Data/traj01")

META_DIR = Path("outputs/satloc/metadata/s4b_structural_retrieval")
FIG_DIR = Path("outputs/satloc/figures/s4b_structural_retrieval/s4b1g_cell_texture_suppressed")
REPORT_DIR = Path("outputs/satloc/reports/s4b_structural_retrieval")


def norm_id(value) -> str:
    if pd.isna(value):
        return ""
    s = str(value).strip()
    try:
        f = float(s)
        if f.is_integer():
            return str(int(f))
    except Exception:
        pass
    return s


def find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    for cand in candidates:
        cand_l = cand.lower()
        for col in df.columns:
            if cand_l in col.lower():
                return col
    return None


def find_path_col(df: pd.DataFrame) -> Optional[str]:
    return find_col(df, [
        "image_path", "file_path", "filepath", "path",
        "uav_path", "image_file", "filename", "file", "name"
    ])


def parse_token_from_filename(path_like: str) -> str:
    name = Path(str(path_like)).name
    if "@" in name:
        return norm_id(name.split("@")[0])
    nums = re.findall(r"\d+", Path(name).stem)
    return norm_id(nums[0]) if nums else ""


def resolve_path(value, base_dirs: List[Path]) -> Path:
    p = Path(str(value).strip())
    candidates = []

    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.append(ROOT / p)
        for base in base_dirs:
            candidates.append(base / p)
            candidates.append(base / p.name)

    for c in candidates:
        if c.exists():
            return c

    return candidates[0]


def load_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def preprocess_gray(rgb: np.ndarray, preprocess: str) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    if preprocess == "luma":
        return gray

    if preprocess == "clahe_luma":
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(gray)

    raise ValueError(f"Unknown preprocess: {preprocess}")


def crop_resize(gray: np.ndarray, resize_size: int) -> np.ndarray:
    h, w = gray.shape[:2]
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    crop = gray[y0:y0 + side, x0:x0 + side]
    return cv2.resize(crop, (resize_size, resize_size), interpolation=cv2.INTER_AREA)


def normalize01(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - lo) / (hi - lo)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32).reshape(-1)
    b = b.astype(np.float32).reshape(-1)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(np.dot(a / na, b / nb))


def extract_cell_features(
    rgb: np.ndarray,
    preprocess: str,
    resize_size: int,
    cells: int,
    bins: int,
    edge_threshold: float,
) -> Dict:
    gray = preprocess_gray(rgb, preprocess)
    gray = crop_resize(gray, resize_size)

    g = cv2.GaussianBlur(gray.astype(np.float32) / 255.0, (3, 3), 0)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)

    mag = np.sqrt(gx * gx + gy * gy)
    ori = np.mod(np.arctan2(gy, gx), np.pi)

    H = np.zeros((cells, cells, bins), dtype=np.float32)
    entropy = np.zeros((cells, cells), dtype=np.float32)
    edge_density = np.zeros((cells, cells), dtype=np.float32)
    energy = np.zeros((cells, cells), dtype=np.float32)

    h, w = mag.shape
    cell_h = h / cells
    cell_w = w / cells

    for cy in range(cells):
        for cx in range(cells):
            y0 = int(round(cy * cell_h))
            y1 = int(round((cy + 1) * cell_h))
            x0 = int(round(cx * cell_w))
            x1 = int(round((cx + 1) * cell_w))

            cmag = mag[y0:y1, x0:x1]
            cori = ori[y0:y1, x0:x1]

            hist, _ = np.histogram(
                cori.reshape(-1),
                bins=bins,
                range=(0.0, np.pi),
                weights=cmag.reshape(-1),
            )
            hist = hist.astype(np.float32)
            H[cy, cx, :] = hist

            s = float(hist.sum())
            if s > 1e-8:
                p = hist / s
                p = p[p > 0]
                entropy[cy, cx] = float(-np.sum(p * np.log2(p)) / math.log2(bins))
            else:
                entropy[cy, cx] = 0.0

            edge_density[cy, cx] = float(np.mean(cmag > edge_threshold))
            energy[cy, cx] = float(np.sum(cmag))

    energy_norm = normalize01(energy)

    # Core idea:
    # noisy vegetation cells usually have high entropy + high edge density.
    texture_badness = np.clip(entropy * edge_density, 0.0, 1.0)

    # Keep cells that have enough signal, but penalize dense chaotic texture.
    signal = np.sqrt(np.clip(energy_norm, 0.0, 1.0))
    cell_weight = signal * np.power(1.0 - texture_badness, 1.5)

    # Avoid fully killing all cells.
    cell_weight = 0.05 + 0.95 * normalize01(cell_weight)

    sobel_vis = normalize01(mag)

    return {
        "gray": gray,
        "mag": mag,
        "sobel_vis": sobel_vis,
        "hog": H,
        "entropy": entropy,
        "edge_density": edge_density,
        "energy_norm": energy_norm,
        "texture_badness": texture_badness,
        "cell_weight": cell_weight,
    }


def weighted_cell_hog_score(qfeat: Dict, cfeat: Dict) -> Tuple[float, np.ndarray, np.ndarray]:
    qH = qfeat["hog"]
    cH = cfeat["hog"]

    cells_y, cells_x, _ = qH.shape
    cell_sim = np.zeros((cells_y, cells_x), dtype=np.float32)

    for y in range(cells_y):
        for x in range(cells_x):
            cell_sim[y, x] = cosine(qH[y, x, :], cH[y, x, :])

    pair_weight = np.sqrt(qfeat["cell_weight"] * cfeat["cell_weight"])
    denom = float(pair_weight.sum())

    if denom < 1e-8:
        score = 0.0
    else:
        score = float(np.sum(cell_sim * pair_weight) / denom)

    contribution = cell_sim * pair_weight

    return score, cell_sim, contribution


def save_panel(query_rgb: np.ndarray, qfeat: Dict, rows: List[Dict], out_png: Path, title: str):
    nrows = 1 + len(rows)
    ncols = 6

    fig, axes = plt.subplots(nrows, ncols, figsize=(24, max(5, 3.4 * nrows)), squeeze=False)

    axes[0, 0].imshow(query_rgb)
    axes[0, 0].set_title("QUERY RGB")

    axes[0, 1].imshow(qfeat["sobel_vis"], cmap="gray")
    axes[0, 1].set_title("query Sobel")

    im = axes[0, 2].imshow(qfeat["entropy"], vmin=0, vmax=1)
    axes[0, 2].set_title("query entropy")
    fig.colorbar(im, ax=axes[0, 2], fraction=0.046, pad=0.04)

    im = axes[0, 3].imshow(qfeat["edge_density"], vmin=0, vmax=1)
    axes[0, 3].set_title("query edge density")
    fig.colorbar(im, ax=axes[0, 3], fraction=0.046, pad=0.04)

    im = axes[0, 4].imshow(qfeat["cell_weight"], vmin=0, vmax=1)
    axes[0, 4].set_title("query cell weight")
    fig.colorbar(im, ax=axes[0, 4], fraction=0.046, pad=0.04)

    axes[0, 5].axis("off")
    axes[0, 5].text(
        0.0,
        0.95,
        "Query maps\nYellow/high = used more\nDark/low = suppressed",
        va="top",
        fontsize=10,
        family="monospace",
    )

    for c in range(5):
        axes[0, c].axis("off")

    for r, item in enumerate(rows, start=1):
        axes[r, 0].imshow(item["rgb"])
        axes[r, 0].set_title(item["label"], fontsize=8)

        axes[r, 1].imshow(item["feat"]["sobel_vis"], cmap="gray")
        axes[r, 1].set_title("candidate Sobel", fontsize=8)

        im = axes[r, 2].imshow(item["feat"]["texture_badness"], vmin=0, vmax=1)
        axes[r, 2].set_title("texture badness\nentropy × density", fontsize=8)
        fig.colorbar(im, ax=axes[r, 2], fraction=0.046, pad=0.04)

        im = axes[r, 3].imshow(item["feat"]["cell_weight"], vmin=0, vmax=1)
        axes[r, 3].set_title("candidate cell weight", fontsize=8)
        fig.colorbar(im, ax=axes[r, 3], fraction=0.046, pad=0.04)

        im = axes[r, 4].imshow(item["cell_sim"], vmin=0, vmax=1)
        axes[r, 4].set_title("raw per-cell HOG sim", fontsize=8)
        fig.colorbar(im, ax=axes[r, 4], fraction=0.046, pad=0.04)

        im = axes[r, 5].imshow(item["contribution"], vmin=0, vmax=np.max(item["contribution"]) + 1e-6)
        axes[r, 5].set_title("weighted contribution", fontsize=8)
        fig.colorbar(im, ax=axes[r, 5], fraction=0.046, pad=0.04)

        for c in range(6):
            axes[r, c].axis("off")

        axes[r, 0].text(
            0.02,
            0.98,
            item["note"],
            transform=axes[r, 0].transAxes,
            va="top",
            fontsize=7.5,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.82),
            family="monospace",
        )

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(out_png, dpi=145)
    plt.close(fig)


def save_barplot(df: pd.DataFrame, out_png: Path, title: str):
    plot_df = df.sort_values("texture_suppressed_score", ascending=True).copy()
    labels = plot_df["tile_id"].astype(str) + " | " + plot_df["category"].astype(str)

    fig, ax = plt.subplots(figsize=(13, max(7, 0.48 * len(plot_df))))

    y = np.arange(len(plot_df))
    ax.barh(y, plot_df["combined_sim"], label="original combined")
    ax.barh(y, plot_df["texture_suppressed_score"], alpha=0.65, label="cell texture-suppressed")

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Score")
    ax.set_title(title)
    ax.grid(True, axis="x", alpha=0.3)
    ax.legend()

    for i, (_, row) in enumerate(plot_df.iterrows()):
        ax.text(
            max(row["combined_sim"], row["texture_suppressed_score"]) + 0.002,
            i,
            f"origR={int(row['original_local_rank'])}, newR={int(row['new_local_rank'])}, err={row['center_err_m']:.1f}m",
            va="center",
            fontsize=8,
        )

    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", default="traj01")
    parser.add_argument("--token", type=int, required=True)
    parser.add_argument("--preprocess", choices=["luma", "clahe_luma"], default="luma")
    parser.add_argument("--resize-size", type=int, default=512)
    parser.add_argument("--cells", type=int, default=8)
    parser.add_argument("--bins", type=int, default=9)
    parser.add_argument("--edge-threshold", type=float, default=0.05)
    parser.add_argument("--blend-original", type=float, default=0.30)
    parser.add_argument("--uav-index", default=str(DEFAULT_UAV_INDEX))
    parser.add_argument("--uav-dir", default=str(DEFAULT_UAV_DIR))
    args = parser.parse_args()

    META_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    token_str = f"{args.token:04d}"

    decomp_files = sorted(META_DIR.glob(f"s4b1d_token{token_str}_*.csv"))
    if not decomp_files:
        raise FileNotFoundError(
            f"Missing S4B.1d CSV for token {token_str}. Run S4B.1d first."
        )

    decomp_csv = decomp_files[0]
    df = pd.read_csv(decomp_csv)

    # Normalize S4B.1d column names to S4B.1g expected names.
    if "combined_sim" not in df.columns and "combined_similarity" in df.columns:
        df["combined_sim"] = df["combined_similarity"]

    if "center_err_m" not in df.columns and "center_error_m" in df.columns:
        df["center_err_m"] = df["center_error_m"]

    if "rank_assigned" not in df.columns:
        if "rank" in df.columns:
            df["rank_assigned"] = df["rank"]
        else:
            df["rank_assigned"] = np.nan

    if "category" not in df.columns:
        categories = []
        for _, r in df.iterrows():
            ctype = str(r.get("candidate_type", ""))

            if ctype == "gt3x3":
                ox = r.get("offset_x", np.nan)
                oy = r.get("offset_y", np.nan)

                if pd.notna(ox) and pd.notna(oy) and int(float(ox)) == 0 and int(float(oy)) == 0:
                    categories.append("TRUE_GT_CENTER")
                else:
                    categories.append("SHIFTED_GT_NEIGHBOR")
            else:
                rank_val = r.get("rank", np.nan)
                if pd.notna(rank_val) and int(float(rank_val)) == 1:
                    categories.append("RANK_1_WINNER")
                elif pd.notna(rank_val):
                    categories.append(f"FALSE_POSITIVE_RANK_{int(float(rank_val))}")
                else:
                    categories.append("FALSE_POSITIVE_RANK_NA")

        df["category"] = categories

    uav_df = pd.read_csv(args.uav_index)
    uav_path_col = find_path_col(uav_df)

    seq_col = find_col(uav_df, ["sequence", "seq", "traj", "trajectory"])
    if seq_col is not None:
        mask = uav_df[seq_col].astype(str).str.lower().eq(args.sequence.lower())
        if mask.any():
            uav_df = uav_df[mask].copy()

    token_col = find_col(uav_df, ["token0_id", "token", "frame_token", "frame_id", "id", "uav_id"])
    if token_col:
        uav_df["_token_norm"] = uav_df[token_col].map(norm_id)
    else:
        uav_df["_token_norm"] = uav_df[uav_path_col].map(parse_token_from_filename)

    qrow = uav_df[uav_df["_token_norm"].eq(str(args.token))].iloc[0]
    qpath = resolve_path(qrow[uav_path_col], [Path(args.uav_dir)])

    qrgb = load_rgb(qpath)
    qfeat = extract_cell_features(
        qrgb,
        preprocess=args.preprocess,
        resize_size=args.resize_size,
        cells=args.cells,
        bins=args.bins,
        edge_threshold=args.edge_threshold,
    )

    output_rows = []
    panel_rows = []

    for _, row in df.iterrows():
        tile_path = resolve_path(row["tile_path"], [])
        rgb = load_rgb(tile_path)

        feat = extract_cell_features(
            rgb,
            preprocess=args.preprocess,
            resize_size=args.resize_size,
            cells=args.cells,
            bins=args.bins,
            edge_threshold=args.edge_threshold,
        )

        weighted_score, cell_sim, contribution = weighted_cell_hog_score(qfeat, feat)

        combined_sim = float(row["combined_sim"])
        final_score = (args.blend_original * combined_sim) + ((1.0 - args.blend_original) * weighted_score)

        rec = row.to_dict()
        rec["weighted_cell_hog_score"] = weighted_score
        rec["texture_suppressed_score"] = final_score
        rec["mean_texture_badness"] = float(np.mean(feat["texture_badness"]))
        rec["mean_cell_weight"] = float(np.mean(feat["cell_weight"]))
        output_rows.append(rec)

        category = str(row["category"])
        rank = row.get("rank_assigned", row.get("rank", "NA"))

        note = (
            f"tile {row['tile_id']}\n"
            f"cat: {category}\n"
            f"orig rank: {rank}\n"
            f"err: {float(row['center_err_m']):.1f}m\n"
            f"orig sim: {combined_sim:.4f}\n"
            f"weighted HOG: {weighted_score:.4f}\n"
            f"final: {final_score:.4f}\n"
            f"badness mean: {np.mean(feat['texture_badness']):.3f}\n"
            f"weight mean: {np.mean(feat['cell_weight']):.3f}"
        )

        panel_rows.append({
            "rgb": rgb,
            "feat": feat,
            "cell_sim": cell_sim,
            "contribution": contribution,
            "label": f"{category} | tile {row['tile_id']}",
            "note": note,
            "score": final_score,
        })

    out_df = pd.DataFrame(output_rows)
    out_df["original_local_rank"] = out_df["combined_sim"].rank(ascending=False, method="min").astype(int)
    out_df["new_local_rank"] = out_df["texture_suppressed_score"].rank(ascending=False, method="min").astype(int)
    out_df = out_df.sort_values("texture_suppressed_score", ascending=False).copy()

    run_name = (
        f"token{token_str}_{args.preprocess}_r{args.resize_size}_"
        f"c{args.cells}_b{args.bins}_eth{args.edge_threshold}_blend{args.blend_original}"
    )

    out_csv = META_DIR / f"s4b1g_{run_name}_cell_texture_suppressed_rerank.csv"
    bar_png = FIG_DIR / f"s4b1g_{run_name}_barplot.png"
    panel_png = FIG_DIR / f"s4b1g_{run_name}_inspection_panel.png"

    out_df.to_csv(out_csv, index=False)

    # Show top candidates plus best near candidates in panel.
    top_panel = []
    used_tiles = set()

    for _, r in out_df.head(8).iterrows():
        tid = str(r["tile_id"])
        used_tiles.add(tid)

    near_df = out_df.sort_values("center_err_m", ascending=True).head(6)
    keep_ids = set(out_df.head(8)["tile_id"].astype(str)) | set(near_df["tile_id"].astype(str))

    for item, rec in zip(panel_rows, output_rows):
        if str(rec["tile_id"]) in keep_ids:
            top_panel.append(item)

    top_panel = sorted(top_panel, key=lambda x: x["score"], reverse=True)

    save_barplot(
        out_df,
        bar_png,
        title=f"S4B.1g token {token_str}: original vs cell texture-suppressed rerank"
    )

    save_panel(
        qrgb,
        qfeat,
        top_panel,
        panel_png,
        title=f"S4B.1g cell texture-suppressed inspection — token {token_str}"
    )

    print("S4B.1g cell texture-suppressed rerank complete")
    print("------------------------------------------------")
    print(f"Token:              {token_str}")
    print(f"Input S4B.1d CSV:   {decomp_csv}")
    print(f"Output CSV:         {out_csv}")
    print(f"Bar plot:           {bar_png}")
    print(f"Inspection panel:   {panel_png}")
    print("")
    print("Top candidates after S4B.1g:")
    cols = [
        "tile_id",
        "category",
        "center_err_m",
        "combined_sim",
        "weighted_cell_hog_score",
        "texture_suppressed_score",
        "mean_texture_badness",
        "mean_cell_weight",
        "original_local_rank",
        "new_local_rank",
    ]
    print(out_df[cols].head(15).to_string(index=False))


if __name__ == "__main__":
    main()
