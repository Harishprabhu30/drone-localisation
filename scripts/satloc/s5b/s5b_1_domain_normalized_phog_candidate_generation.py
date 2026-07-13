#!/usr/bin/env python3
"""
S5B.1 — Domain-normalized PHOG candidate generation on candidate-pool failures

Purpose
-------
The S5A LightGlue verifier is close to the PHOG top-50 oracle.
The remaining bottleneck is candidate-pool failure:
the correct tile is not inside PHOG top-50 for 40/73 tokens.

This script tests whether domain-normalized structural preprocessing can improve
candidate generation for those hard tokens.

Variants
--------
existing_phog_order:
    Uses the existing PHOG ranked CSV order as baseline.

v1_lab_l_clahe:
    Common LAB L-channel with mild UAV CLAHE and stronger satellite CLAHE.

v2_sat_detail:
    UAV mild normalized L; satellite detail-enhanced L using bilateral residual.

v3_green_suppressed:
    Suppress excessive green dominance before extracting structure.

v4_uav_blur_sat_sharpen:
    Reduce UAV over-sharpness and sharpen satellite structure.

v5_edge_magnitude:
    PHOG over normalized edge/gradient magnitude image.

Locked rule
-----------
Reference/eval_error_m is used only after ranking for evaluation.
All candidate ranking uses image-derived descriptors only.

Command USed:
export PYTHONPATH=$PWD/src

python scripts/satloc/s5b/s5b_1_domain_normalized_phog_candidate_generation.py \
  --run-name cpf_all40 \
  --split-key candidate_pool_failure \
  --max-tokens 0 \
  --max-candidates 0 \
  --save-top-n 200 \
  --top-k-eval 50 \
  --resize-long 256

* log chromaticity test:
export PYTHONPATH=$PWD/src

python scripts/satloc/s5b/s5b_1b_fullmap_domain_normalized_phog.py \
  --run-name cpf_fullmap_logchroma_all40 \
  --split-key candidate_pool_failure \
  --max-tokens 0 \
  --resize-long 512 \
  --top-k-eval 50 \
  --save-top-n 200 \
  --variants v6_log_chroma_clahe,v7_log_chroma_edges,v8_lab_logchroma_fused,v9_canny_structure
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


IMAGE_PATH_COLS = [
    "candidate_image_path",
    "sat_image_path",
    "satellite_image_path",
    "tile_image_path",
    "image_path",
    "tile_path",
    "ref_image_path",
]

UAV_PATH_COLS = [
    "uav_image_path",
    "query_image_path",
    "query_path",
    "image_query_path",
]

RANKED_CSV_COLS = [
    "ranked_csv_path",
    "ranked_csv",
    "phog_ranked_csv",
    "ranked_candidates_csv",
    "candidate_csv",
    "s4c_ranked_csv_path",
]

TILE_ID_COLS = [
    "tile_id",
    "candidate_tile_id",
    "ref_tile_id",
    "sat_tile_id",
    "image_id",
]

EVAL_ERROR_COLS = [
    "eval_error_m",
    "error_m",
    "distance_m",
    "center_error_m",
    "candidate_error_m",
]

BASELINE_RANK_COLS = [
    "candidate_pool_rank",
    "rank",
    "phog_rank",
    "retrieval_rank",
    "candidate_rank",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--token-lists",
        type=Path,
        default=Path("outputs/satloc/metadata/s5b_candidate_pool_improvement/s5b0_token_lists_top50_all73.json"),
    )
    p.add_argument(
        "--manifest",
        type=Path,
        default=Path("outputs/satloc/metadata/s5a_learned_local_verifier/s5a0_failure_group_benchmark_manifest.csv"),
    )
    p.add_argument(
        "--fallback-candidates",
        type=Path,
        default=Path("outputs/satloc/metadata/s5a_learned_local_verifier/s5a3_lightglue_candidate_scores_top50_all73.csv"),
        help="Fallback only if ranked full CSV path is missing. Usually only top-50.",
    )
    p.add_argument("--out-base", type=Path, default=Path("outputs/satloc"))
    p.add_argument("--run-name", type=str, default="candidate_pool_failure")
    p.add_argument("--split-key", type=str, default="candidate_pool_failure")
    p.add_argument("--tokens", type=str, default="")
    p.add_argument("--max-tokens", type=int, default=8, help="Use 0 for all selected tokens.")
    p.add_argument("--max-candidates", type=int, default=0, help="0 means use all rows in ranked CSV.")
    p.add_argument("--save-top-n", type=int, default=200)
    p.add_argument("--top-k-eval", type=int, default=50)
    p.add_argument("--threshold-m", type=float, default=40.0)
    p.add_argument("--resize-long", type=int, default=256)
    p.add_argument("--bins", type=int, default=9)
    p.add_argument("--levels", type=str, default="1,2,4")
    p.add_argument(
        "--variants",
        type=str,
        default="existing_phog_order,v1_lab_l_clahe,v2_sat_detail,v3_green_suppressed,v4_uav_blur_sat_sharpen,v5_edge_magnitude",
    )
    p.add_argument("--save-panels", action="store_true")
    return p.parse_args()


def ensure_dirs(base: Path):
    d = {
        "metadata": base / "metadata" / "s5b_candidate_pool_improvement",
        "reports": base / "reports" / "s5b_candidate_pool_improvement",
        "figures": base / "figures" / "s5b_candidate_pool_improvement",
        "panels": base / "figures" / "s5b_candidate_pool_improvement" / "s5b1_variant_panels",
    }
    for x in d.values():
        x.mkdir(parents=True, exist_ok=True)
    return d


def safe_str(x: Any) -> str:
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return str(x).strip()


def safe_float(x: Any) -> Optional[float]:
    try:
        if x is None or pd.isna(x):
            return None
        return float(x)
    except Exception:
        return None


def resolve_path(x: Any) -> Optional[Path]:
    s = safe_str(x)
    if not s:
        return None
    p = Path(s)
    if p.exists():
        return p
    p2 = Path.cwd() / p
    if p2.exists():
        return p2
    return None


def first_existing_path(row: pd.Series, cols: List[str]) -> Optional[Path]:
    for c in cols:
        if c in row.index:
            p = resolve_path(row.get(c))
            if p is not None:
                return p
    return None


def first_existing_value(row: pd.Series, cols: List[str], default: Any = "") -> Any:
    for c in cols:
        if c in row.index:
            v = row.get(c)
            if safe_str(v):
                return v
    return default


def read_rgb(path: Optional[Path]) -> Optional[np.ndarray]:
    if path is None:
        return None
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def resize_longest(img: np.ndarray, longest: int) -> np.ndarray:
    if longest <= 0:
        return img
    h, w = img.shape[:2]
    m = max(h, w)
    if m <= longest:
        return img
    scale = longest / float(m)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)


def normalize_u8(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    lo, hi = np.percentile(x, [1, 99])
    if hi <= lo:
        return np.zeros_like(x, dtype=np.uint8)
    y = (x - lo) * 255.0 / (hi - lo)
    return np.clip(y, 0, 255).astype(np.uint8)


def clahe(gray: np.ndarray, clip: float) -> np.ndarray:
    return cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8)).apply(gray)


def lab_l(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)[:, :, 0]


def suppress_green(rgb: np.ndarray, strength: float = 0.55) -> np.ndarray:
    arr = rgb.astype(np.float32).copy()
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    exg = 2.0 * g - r - b
    mask = exg > np.percentile(exg, 60)
    arr[:, :, 1][mask] *= strength
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def unsharp(gray: np.ndarray, amount: float = 1.2, sigma: float = 1.2) -> np.ndarray:
    blur = cv2.GaussianBlur(gray, (0, 0), sigma)
    sharp = cv2.addWeighted(gray, 1.0 + amount, blur, -amount, 0)
    return np.clip(sharp, 0, 255).astype(np.uint8)


def preprocess(rgb: np.ndarray, variant: str, domain: str) -> np.ndarray:
    """
    domain is 'uav' or 'sat'.
    Output is a single-channel uint8 structural image.
    """
    rgb = resize_longest(rgb, 256)

    if variant == "v1_lab_l_clahe":
        L = lab_l(rgb)
        return clahe(L, 2.0 if domain == "uav" else 4.0)

    if variant == "v2_sat_detail":
        L = clahe(lab_l(rgb), 2.0 if domain == "uav" else 4.0)
        if domain == "sat":
            smooth = cv2.bilateralFilter(L, 9, 50, 50)
            detail = cv2.subtract(L, smooth)
            out = cv2.addWeighted(L, 0.85, detail, 2.2, 0)
            return normalize_u8(out)
        else:
            mild = cv2.GaussianBlur(L, (3, 3), 0)
            return cv2.addWeighted(L, 0.75, mild, 0.25, 0)

    if variant == "v3_green_suppressed":
        rgbs = suppress_green(rgb, strength=0.50 if domain == "sat" else 0.70)
        L = lab_l(rgbs)
        return clahe(L, 2.0 if domain == "uav" else 3.5)

    if variant == "v4_uav_blur_sat_sharpen":
        L = clahe(lab_l(rgb), 2.0 if domain == "uav" else 3.5)
        if domain == "uav":
            return cv2.GaussianBlur(L, (5, 5), 0)
        return unsharp(L, amount=1.5, sigma=1.0)

    if variant == "v5_edge_magnitude":
        L = clahe(lab_l(rgb), 2.0 if domain == "uav" else 4.0)
        if domain == "uav":
            L = cv2.GaussianBlur(L, (3, 3), 0)
        else:
            L = unsharp(L, amount=1.0, sigma=1.0)

        gx = cv2.Scharr(L, cv2.CV_32F, 1, 0)
        gy = cv2.Scharr(L, cv2.CV_32F, 0, 1)
        mag = np.sqrt(gx * gx + gy * gy)
        return normalize_u8(mag)

    # fallback/current-ish luma
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return clahe(gray, 2.0)


def phog_descriptor(gray: np.ndarray, bins: int, levels: List[int]) -> np.ndarray:
    gray = gray.astype(np.uint8)

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

    mag = np.sqrt(gx * gx + gy * gy)
    ang = (np.arctan2(gy, gx) + np.pi) % np.pi

    h, w = gray.shape[:2]
    feats = []

    for grid in levels:
        for yy in range(grid):
            y0 = int(round(yy * h / grid))
            y1 = int(round((yy + 1) * h / grid))
            for xx in range(grid):
                x0 = int(round(xx * w / grid))
                x1 = int(round((xx + 1) * w / grid))

                a = ang[y0:y1, x0:x1].ravel()
                m = mag[y0:y1, x0:x1].ravel()

                if len(a) == 0:
                    hist = np.zeros(bins, dtype=np.float32)
                else:
                    hist, _ = np.histogram(a, bins=bins, range=(0, np.pi), weights=m)
                    hist = hist.astype(np.float32)

                s = np.linalg.norm(hist) + 1e-9
                feats.append(hist / s)

    desc = np.concatenate(feats).astype(np.float32)
    desc /= np.linalg.norm(desc) + 1e-9
    return desc


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(1.0 - np.dot(a, b) / ((np.linalg.norm(a) + 1e-9) * (np.linalg.norm(b) + 1e-9)))


def load_token_list(path: Path, split_key: str, tokens_arg: str, max_tokens: int) -> List[str]:
    if tokens_arg.strip():
        toks = [t.strip() for t in tokens_arg.split(",") if t.strip()]
    else:
        with open(path, "r") as f:
            lists = json.load(f)
        if split_key not in lists:
            raise KeyError(f"Split key '{split_key}' not found in {path}. Available: {list(lists.keys())}")
        toks = [t.strip() for t in lists[split_key].split(",") if t.strip()]

    if max_tokens > 0:
        toks = toks[:max_tokens]
    return toks


def find_manifest_row(manifest: pd.DataFrame, token: str) -> Optional[pd.Series]:
    if "token" not in manifest.columns:
        return None
    sub = manifest[manifest["token"].astype(str) == str(token)]
    if len(sub) == 0:
        return None
    return sub.iloc[0]


def get_ranked_csv_path(row: Optional[pd.Series]) -> Optional[Path]:
    if row is None:
        return None

    p = first_existing_path(row, RANKED_CSV_COLS)
    if p is not None:
        return p

    # Flexible fallback: any column containing both ranked and csv.
    for c in row.index:
        lc = c.lower()
        if "rank" in lc and "csv" in lc:
            p = resolve_path(row.get(c))
            if p is not None:
                return p
    return None


def get_uav_path(row: Optional[pd.Series], fallback_rows: pd.DataFrame, token: str) -> Optional[Path]:
    if row is not None:
        p = first_existing_path(row, UAV_PATH_COLS)
        if p is not None:
            return p

    if len(fallback_rows):
        sub = fallback_rows[fallback_rows["token"].astype(str) == str(token)]
        if len(sub):
            return first_existing_path(sub.iloc[0], UAV_PATH_COLS)
    return None


def candidate_path(row: pd.Series) -> Optional[Path]:
    return first_existing_path(row, IMAGE_PATH_COLS)


def tile_id(row: pd.Series) -> str:
    return safe_str(first_existing_value(row, TILE_ID_COLS, ""))


def eval_error(row: pd.Series) -> Optional[float]:
    for c in EVAL_ERROR_COLS:
        if c in row.index:
            v = safe_float(row.get(c))
            if v is not None:
                return v
    return None


def baseline_rank(row: pd.Series, fallback_idx: int) -> int:
    for c in BASELINE_RANK_COLS:
        if c in row.index:
            v = safe_float(row.get(c))
            if v is not None and math.isfinite(v):
                return int(v)
    return fallback_idx + 1


def load_candidates_for_token(
    token: str,
    manifest: pd.DataFrame,
    fallback: pd.DataFrame,
    max_candidates: int,
) -> Tuple[pd.DataFrame, str, Optional[Path], Optional[Path]]:
    mrow = find_manifest_row(manifest, token)
    ranked_csv = get_ranked_csv_path(mrow)
    uav_path = get_uav_path(mrow, fallback, token)

    source = ""
    if ranked_csv is not None and ranked_csv.exists():
        c = pd.read_csv(ranked_csv)
        source = str(ranked_csv)
    else:
        c = fallback[fallback["token"].astype(str) == str(token)].copy()
        source = "fallback_top50_candidate_scores"

    if len(c) == 0:
        return c, source, uav_path, ranked_csv

    rows = []
    for i, r in c.iterrows():
        p = candidate_path(r)
        if p is None:
            continue
        rows.append(
            {
                "token": token,
                "tile_id": tile_id(r),
                "candidate_image_path": str(p),
                "eval_error_m": eval_error(r),
                "existing_rank": baseline_rank(r, len(rows)),
            }
        )

    out = pd.DataFrame(rows)
    out = out.sort_values("existing_rank", kind="mergesort")

    if max_candidates > 0:
        out = out.head(max_candidates).copy()

    return out.reset_index(drop=True), source, uav_path, ranked_csv


def evaluate_order(df: pd.DataFrame, top_k: int, threshold_m: float) -> Dict[str, Any]:
    d = df.copy()
    d["eval_error_num"] = pd.to_numeric(d["eval_error_m"], errors="coerce")

    top1_err = None
    if len(d) and pd.notna(d.iloc[0]["eval_error_num"]):
        top1_err = float(d.iloc[0]["eval_error_num"])

    topk = d.head(top_k)
    oracle_topk_err = None
    oracle_topk_hit = False

    valid_topk = topk.dropna(subset=["eval_error_num"])
    if len(valid_topk):
        oracle_topk_err = float(valid_topk["eval_error_num"].min())
        oracle_topk_hit = bool(oracle_topk_err <= threshold_m)

    correct = d[d["eval_error_num"] <= threshold_m]
    first_correct_rank = None
    best_correct_error = None
    if len(correct):
        first_correct_rank = int(correct.iloc[0]["variant_rank"])
        best_correct_error = float(correct["eval_error_num"].min())

    return {
        "top1_error_m": top1_err,
        "top1_hit_le_threshold": bool(top1_err is not None and top1_err <= threshold_m),
        "oracle_topk_error_m": oracle_topk_err,
        "oracle_topk_hit_le_threshold": oracle_topk_hit,
        "first_correct_rank": first_correct_rank,
        "best_correct_error_m": best_correct_error,
    }


def add_image(ax, img: Optional[np.ndarray], title: str):
    ax.axis("off")
    ax.set_title(title, fontsize=9)
    if img is None:
        ax.text(0.5, 0.5, "missing", ha="center", va="center")
    else:
        ax.imshow(img)


def save_panel(token: str, uav_path: Path, best_rows: pd.DataFrame, out_path: Path):
    uav = read_rgb(uav_path)
    n = min(5, len(best_rows))
    fig, axes = plt.subplots(n, 3, figsize=(13, 3.8 * n))
    if n == 1:
        axes = np.asarray([axes])

    for i, (_, r) in enumerate(best_rows.head(n).iterrows()):
        sat = read_rgb(resolve_path(r["candidate_image_path"]))
        add_image(axes[i, 0], uav, f"UAV token {token}")
        add_image(
            axes[i, 1],
            sat,
            f"{r['variant']} top1\n"
            f"tile={r['tile_id']} rank={r['variant_rank']} err={r['eval_error_m']}",
        )

        proc_u = preprocess(resize_longest(uav, 256), r["variant"], "uav") if uav is not None and r["variant"] != "existing_phog_order" else None
        proc_s = preprocess(resize_longest(sat, 256), r["variant"], "sat") if sat is not None and r["variant"] != "existing_phog_order" else None

        if proc_u is not None and proc_s is not None:
            h = max(proc_u.shape[0], proc_s.shape[0])
            w = proc_u.shape[1] + proc_s.shape[1]
            canvas = np.zeros((h, w), dtype=np.uint8)
            canvas[:proc_u.shape[0], :proc_u.shape[1]] = proc_u
            canvas[:proc_s.shape[0], proc_u.shape[1]:] = proc_s
            axes[i, 2].imshow(canvas, cmap="gray")
        else:
            axes[i, 2].text(0.5, 0.5, "existing order", ha="center", va="center")
        axes[i, 2].axis("off")
        axes[i, 2].set_title("processed common-space view", fontsize=9)

    fig.suptitle(f"S5B.1 variant top1 comparison | token {token}", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main():
    args = parse_args()
    dirs = ensure_dirs(args.out_base)

    tokens = load_token_list(args.token_lists, args.split_key, args.tokens, args.max_tokens)
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    levels = [int(x.strip()) for x in args.levels.split(",") if x.strip()]

    if not args.manifest.exists():
        raise FileNotFoundError(f"Missing manifest: {args.manifest}")
    if not args.fallback_candidates.exists():
        raise FileNotFoundError(f"Missing fallback candidate scores: {args.fallback_candidates}")

    manifest = pd.read_csv(args.manifest)
    fallback = pd.read_csv(args.fallback_candidates)

    feature_cache: Dict[Tuple[str, str], np.ndarray] = {}
    image_cache: Dict[str, Optional[np.ndarray]] = {}

    def get_rgb_cached(path: Path) -> Optional[np.ndarray]:
        key = str(path)
        if key not in image_cache:
            image_cache[key] = read_rgb(path)
        return image_cache[key]

    def get_feature(path: Path, variant: str, domain: str) -> Optional[np.ndarray]:
        key = (variant + ":" + domain, str(path))
        if key in feature_cache:
            return feature_cache[key]

        rgb = get_rgb_cached(path)
        if rgb is None:
            return None

        gray = preprocess(rgb, variant, domain)
        desc = phog_descriptor(gray, bins=args.bins, levels=levels)
        feature_cache[key] = desc
        return desc

    all_ranked_rows: List[pd.DataFrame] = []
    query_rows: List[Dict[str, Any]] = []

    started = time.time()

    for token in tokens:
        candidates, source, uav_path, ranked_csv = load_candidates_for_token(
            token,
            manifest,
            fallback,
            args.max_candidates,
        )

        print(f"[S5B.1] token={token} candidates={len(candidates)} source={source}")

        if len(candidates) == 0 or uav_path is None:
            for variant in variants:
                query_rows.append(
                    {
                        "token": token,
                        "variant": variant,
                        "status": "missing_candidates_or_uav",
                        "candidate_count": int(len(candidates)),
                        "source": source,
                    }
                )
            continue

        # Existing PHOG ranked order baseline.
        if "existing_phog_order" in variants:
            base = candidates.copy()
            base["variant"] = "existing_phog_order"
            base["variant_distance"] = np.nan
            base["variant_score"] = np.nan
            base = base.sort_values("existing_rank", kind="mergesort").copy()
            base["variant_rank"] = np.arange(1, len(base) + 1)

            top_save = base.head(args.save_top_n).copy()
            all_ranked_rows.append(top_save)

            ev = evaluate_order(base, args.top_k_eval, args.threshold_m)
            query_rows.append(
                {
                    "token": token,
                    "variant": "existing_phog_order",
                    "status": "ok",
                    "candidate_count": int(len(base)),
                    "source": source,
                    "uav_image_path": str(uav_path),
                    "ranked_csv_path": str(ranked_csv) if ranked_csv else "",
                    **ev,
                }
            )

        for variant in variants:
            if variant == "existing_phog_order":
                continue

            ufeat = get_feature(uav_path, variant, "uav")
            if ufeat is None:
                query_rows.append(
                    {
                        "token": token,
                        "variant": variant,
                        "status": "missing_uav_image",
                        "candidate_count": int(len(candidates)),
                        "source": source,
                    }
                )
                continue

            rows = []
            missing_sat = 0

            for _, r in candidates.iterrows():
                sp = resolve_path(r["candidate_image_path"])
                if sp is None:
                    missing_sat += 1
                    continue

                sfeat = get_feature(sp, variant, "sat")
                if sfeat is None:
                    missing_sat += 1
                    continue

                dist = cosine_distance(ufeat, sfeat)
                rows.append(
                    {
                        "token": token,
                        "variant": variant,
                        "tile_id": r["tile_id"],
                        "candidate_image_path": r["candidate_image_path"],
                        "eval_error_m": r["eval_error_m"],
                        "existing_rank": r["existing_rank"],
                        "variant_distance": dist,
                        "variant_score": -dist,
                    }
                )

            if not rows:
                query_rows.append(
                    {
                        "token": token,
                        "variant": variant,
                        "status": "no_valid_satellite_features",
                        "candidate_count": int(len(candidates)),
                        "missing_satellite_images": int(missing_sat),
                        "source": source,
                    }
                )
                continue

            ranked = pd.DataFrame(rows).sort_values(
                ["variant_distance", "existing_rank"],
                ascending=[True, True],
                kind="mergesort",
            ).copy()
            ranked["variant_rank"] = np.arange(1, len(ranked) + 1)

            all_ranked_rows.append(ranked.head(args.save_top_n).copy())

            ev = evaluate_order(ranked, args.top_k_eval, args.threshold_m)
            query_rows.append(
                {
                    "token": token,
                    "variant": variant,
                    "status": "ok",
                    "candidate_count": int(len(ranked)),
                    "missing_satellite_images": int(missing_sat),
                    "source": source,
                    "uav_image_path": str(uav_path),
                    "ranked_csv_path": str(ranked_csv) if ranked_csv else "",
                    **ev,
                }
            )

        if args.save_panels:
            qtmp = pd.DataFrame([r for r in query_rows if str(r.get("token")) == str(token)])
            if len(qtmp):
                best_rows = []
                for v in variants:
                    if v == "existing_phog_order":
                        sub = candidates.copy()
                        sub["variant"] = v
                        sub["variant_rank"] = sub["existing_rank"]
                        sub = sub.sort_values("existing_rank").head(1)
                    else:
                        sub_all = [x for x in all_ranked_rows if len(x) and str(x.iloc[0]["token"]) == str(token) and str(x.iloc[0]["variant"]) == v]
                        sub = sub_all[-1].head(1) if sub_all else pd.DataFrame()
                    if len(sub):
                        best_rows.append(sub.iloc[0].to_dict())
                if best_rows:
                    panel_df = pd.DataFrame(best_rows)
                    save_panel(
                        token,
                        uav_path,
                        panel_df,
                        dirs["panels"] / args.run_name / f"s5b1_token{int(float(token)):04d}_variant_panel.png",
                    )

    ranked_out_df = pd.concat(all_ranked_rows, ignore_index=True) if all_ranked_rows else pd.DataFrame()
    query_df = pd.DataFrame(query_rows)

    variant_rows = []
    ok = query_df[query_df["status"] == "ok"].copy()

    for variant, g in ok.groupby("variant", dropna=False):
        g["top1_error_num"] = pd.to_numeric(g["top1_error_m"], errors="coerce")
        g["first_correct_rank_num"] = pd.to_numeric(g["first_correct_rank"], errors="coerce")

        variant_rows.append(
            {
                "variant": variant,
                "tokens": int(len(g)),
                "top1_hits": int(g["top1_hit_le_threshold"].sum()),
                "top1_hit_rate": float(g["top1_hit_le_threshold"].mean()),
                "oracle_topk_hits": int(g["oracle_topk_hit_le_threshold"].sum()),
                "oracle_topk_hit_rate": float(g["oracle_topk_hit_le_threshold"].mean()),
                "median_top1_error_m": float(g["top1_error_num"].median()) if g["top1_error_num"].notna().any() else None,
                "median_first_correct_rank": float(g["first_correct_rank_num"].median()) if g["first_correct_rank_num"].notna().any() else None,
                "mean_candidate_count": float(pd.to_numeric(g["candidate_count"], errors="coerce").mean()),
            }
        )

    variant_df = pd.DataFrame(variant_rows).sort_values(
        ["oracle_topk_hit_rate", "top1_hit_rate", "median_top1_error_m"],
        ascending=[False, False, True],
        kind="mergesort",
    )

    suffix = f"_{args.run_name}" if args.run_name else ""

    ranked_out = dirs["metadata"] / f"s5b1_variant_ranked_top{args.save_top_n}{suffix}.csv"
    query_out = dirs["metadata"] / f"s5b1_query_summary{suffix}.csv"
    variant_out = dirs["metadata"] / f"s5b1_variant_summary{suffix}.csv"
    report_out = dirs["reports"] / f"s5b1_domain_normalized_phog_summary{suffix}.json"

    ranked_out_df.to_csv(ranked_out, index=False)
    query_df.to_csv(query_out, index=False)
    variant_df.to_csv(variant_out, index=False)

    fig1 = dirs["figures"] / f"s5b1_oracle_topk_hit_rate_by_variant{suffix}.png"
    if len(variant_df):
        plt.figure(figsize=(12, 5.5))
        plt.bar(variant_df["variant"], variant_df["oracle_topk_hit_rate"])
        plt.xticks(rotation=35, ha="right")
        plt.ylim(0, 1.05)
        plt.ylabel(f"Oracle top-{args.top_k_eval} hit rate")
        plt.title("S5B.1 candidate-pool recovery by preprocessing variant")
        plt.tight_layout()
        plt.savefig(fig1, dpi=180)
        plt.close()

    fig2 = dirs["figures"] / f"s5b1_top1_hit_rate_by_variant{suffix}.png"
    if len(variant_df):
        plt.figure(figsize=(12, 5.5))
        plt.bar(variant_df["variant"], variant_df["top1_hit_rate"])
        plt.xticks(rotation=35, ha="right")
        plt.ylim(0, 1.05)
        plt.ylabel("Top-1 hit rate")
        plt.title("S5B.1 top-1 candidate generation by preprocessing variant")
        plt.tight_layout()
        plt.savefig(fig2, dpi=180)
        plt.close()

    report = {
        "stage": "S5B.1_domain_normalized_phog_candidate_generation",
        "run_name": args.run_name,
        "tokens": tokens,
        "num_tokens_requested": len(tokens),
        "variants": variants,
        "top_k_eval": args.top_k_eval,
        "threshold_m": args.threshold_m,
        "resize_long": args.resize_long,
        "bins": args.bins,
        "levels": levels,
        "runtime_s": time.time() - started,
        "best_variant_by_oracle_topk": variant_df.iloc[0].to_dict() if len(variant_df) else None,
        "outputs": {
            "ranked_topn_csv": str(ranked_out),
            "query_summary_csv": str(query_out),
            "variant_summary_csv": str(variant_out),
            "summary_json": str(report_out),
            "oracle_topk_figure": str(fig1),
            "top1_figure": str(fig2),
        },
        "locked_rule": "eval_error_m used only after ranking; ranking uses image-derived PHOG descriptors only",
    }

    with open(report_out, "w") as f:
        json.dump(report, f, indent=2)

    print("S5B.1 domain-normalized PHOG candidate generation complete")
    print("---------------------------------------------------------")
    print(f"Tokens processed:          {len(tokens)} -> {tokens}")
    print(f"Variants tested:           {variants}")
    print(f"Top-K eval:                {args.top_k_eval}")
    print(f"Ranked rows saved:         {len(ranked_out_df)}")
    print()
    print("Variant summary:")
    if len(variant_df):
        print(variant_df.to_string(index=False))
    else:
        print("No valid variant rows.")
    print()
    print(f"Ranked top-N CSV:          {ranked_out}")
    print(f"Query summary CSV:         {query_out}")
    print(f"Variant summary CSV:       {variant_out}")
    print(f"Summary JSON:              {report_out}")
    print(f"Figures:                   {fig1}")
    print(f"                           {fig2}")
    if args.save_panels:
        print(f"Panels dir:                {dirs['panels'] / args.run_name}")
    print()
    print("Locked rule: reference/error columns were used only after candidate ranking.")


if __name__ == "__main__":
    main()
