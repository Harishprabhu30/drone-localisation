#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple
import cv2
import numpy as np
import pandas as pd

ROOT = Path.cwd()
DECOMP_DIR = Path("outputs/satloc/metadata/s4b_structural_retrieval")
REPORT_DIR = Path("outputs/satloc/reports/s4b_structural_retrieval")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

def load_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

def compute_gradients(rgb: np.ndarray, resize_size: int, preprocess: str) -> Tuple[np.ndarray, np.ndarray]:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    if preprocess == "clahe_luma":
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
    elif preprocess != "luma":
        raise ValueError(f"Unknown preprocess: {preprocess}")

    h, w = gray.shape[:2]
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    crop = gray[y0:y0 + side, x0:x0 + side]

    resized = cv2.resize(crop, (resize_size, resize_size), interpolation=cv2.INTER_AREA)

    g = cv2.GaussianBlur(resized.astype(np.float32) / 255.0, (3, 3), 0)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)

    mag = np.sqrt(gx * gx + gy * gy)
    ori = np.mod(np.arctan2(gy, gx), np.pi)

    return mag, ori

def compute_hog_grid(mag: np.ndarray, ori: np.ndarray, cells: int, bins: int) -> np.ndarray:
    H = np.zeros((cells, cells, bins), dtype=np.float32)
    h, w = mag.shape
    cell_h = h / cells
    cell_w = w / cells
    for cy in range(cells):
        for cx in range(cells):
            y0 = int(round(cy * cell_h))
            y1 = int(round((cy + 1) * cell_h))
            x0 = int(round(cx * cell_w))
            x1 = int(round((cx + 1) * cell_w))
            co = ori[y0:y1, x0:x1].reshape(-1)
            cm = mag[y0:y1, x0:x1].reshape(-1)
            hist, _ = np.histogram(co, bins=bins, range=(0.0, np.pi), weights=cm)
            H[cy, cx, :] = hist
    return H

def analyze_metrics(mag: np.ndarray, H: np.ndarray) -> Dict:
    # 1. HOG Cell Entropy
    cells_y, cells_x, bins = H.shape
    entropies = []
    for y in range(cells_y):
        for x in range(cells_x):
            hist = H[y, x, :].copy()
            s = hist.sum()
            if s > 1e-6:
                p = hist / s
                p = p[p > 0]
                entropy = -np.sum(p * np.log2(p))
                entropies.append(entropy)
    mean_entropy = float(np.mean(entropies)) if entropies else 0.0

    # 2. Edge Density (Fill Rate above a noise floor)
    # Using 10% of global max or fixed threshold
    noise_floor = 0.05
    edge_density = float(np.sum(mag > noise_floor) / mag.size)

    # 3. Structural Sparsity (Coefficient of Variation across cells)
    cell_energies = np.sum(H, axis=2)
    mean_energy = np.mean(cell_energies)
    if mean_energy > 1e-6:
        coef_of_variation = float(np.std(cell_energies) / mean_energy)
    else:
        coef_of_variation = 0.0

    return {
        "mean_hog_entropy": mean_entropy,
        "edge_density": edge_density,
        "structural_sparsity_cv": coef_of_variation
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", type=int, required=True)
    parser.add_argument("--preprocess", default="luma")
    parser.add_argument("--resize-size", type=int, default=512)
    parser.add_argument("--cells", type=int, default=8)
    parser.add_argument("--bins", type=int, default=9)
    args = parser.parse_args()

    # Find matching score decomposition file from step 1d
    token_str = f"{args.token:04d}"
    match_files = list(DECOMP_DIR.glob(f"s4b1d_token{token_str}_*.csv"))
    if not match_files:
        raise FileNotFoundError(f"Could not find score decomposition CSV for token {args.token} in {DECOMP_DIR}")
    
    decomp_csv = match_files[0]
    print(f"Loading previous score breakdown from: {decomp_csv.name}")
    df = pd.read_csv(decomp_csv)

    # Re-extract query metrics
    # Get any valid row to find query metadata or parse from path
    sample_row = df.iloc[0]
    
    # We will compute metrics for each row entry
    results = []
    for idx, row in df.iterrows():
        tpath = Path(row["tile_path"])
        if not tpath.exists():
            # Try absolute or relative fix
            if (ROOT / tpath).exists():
                tpath = ROOT / tpath
            else:
                continue
        
        rgb = load_rgb(tpath)
        mag, ori = compute_gradients(rgb, args.resize_size, args.preprocess)
        H = compute_hog_grid(mag, ori, args.cells, args.bins)
        metrics = analyze_metrics(mag, H)
        
        # Determine specific test category label
        cat = "UNKNOWN"
        if row["candidate_type"] == "gt3x3":
            if row["offset_x"] == 0 and row["offset_y"] == 0:
                cat = "TRUE_GT_CENTER"
            else:
                cat = "SHIFTED_GT_NEIGHBOR"
        else:
            if row["rank"] == 1:
                cat = "RANK_1_WINNER"
            else:
                rank_val = row["rank"]
                rank_txt = "NA" if pd.isna(rank_val) else str(int(rank_val))
                cat = f"FALSE_POSITIVE_RANK_{rank_txt}"

        rec = {
            "tile_id": row["tile_id"],
            "category": cat,
            "rank_assigned": row["rank"],
            "combined_sim": row["combined_similarity"],
            "hog_sim": row["hog_similarity"],
            "edge_sim": row["edge_similarity"],
            "center_err_m": row["center_error_m"],
            "hog_entropy": metrics["mean_hog_entropy"],
            "edge_density_pct": metrics["edge_density"] * 100.0,
            "structural_sparsity_cv": metrics["structural_sparsity_cv"]
        }
        results.append(rec)

    res_df = pd.DataFrame(results)
    
    # Generate report summary markdown table
    print("\n=========================================================================")
    print(f"    ROOT MATHEMATICAL DIAGNOSTIC REPORT FOR TOKEN: {token_str}")
    print("=========================================================================")
    print(f"{'Category':<24} | {'Rank':<4} | {'Sim':<6} | {'Err(m)':<6} | {'HOG Entropy':<11} | {'Edge Dens%':<10} | {'Sparsity(CV)':<12}")
    print("-" * 85)
    
    for _, r in res_df.sort_values(by=["combined_sim"], ascending=False).iterrows():
        print(f"{r['category']:<24} | {str(r['rank_assigned']):<4} | {r['combined_sim']:.4f} | {r['center_err_m']:<6.1f} | {r['hog_entropy']:<11.3f} | {r['edge_density_pct']:<10.1f} | {r['structural_sparsity_cv']:.3f}")
    print("=========================================================================\n")
    
    # Save validation metrics to a clean diagnostic JSON
    out_json = REPORT_DIR / f"s4b1e_token{token_str}_entropy_verification.json"
    res_df.to_json(out_json, orient="records", indent=2)
    print(f"Diagnostic validation saved to: {out_json}")

if __name__ == "__main__":
    main()
