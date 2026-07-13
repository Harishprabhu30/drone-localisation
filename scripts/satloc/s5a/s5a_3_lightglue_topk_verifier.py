#!/usr/bin/env python3
"""
S5A.3 — LightGlue verifier ranking inside PHOG top-K

Purpose
-------
Use PHOG top-K as the candidate pool, then run LightGlue/SuperPoint only inside
that pool to rerank candidates.

This is the first S5A block where LightGlue actually selects/ranks a candidate.

Pipeline:
  UAV query
  + PHOG top-K satellite candidates
  -> SuperPoint features
  -> LightGlue matching
  -> RANSAC / spatial coverage scoring
  -> LightGlue top-1 candidate
  -> evaluate after ranking

Locked rule
-----------
Reference coordinates / eval_error_m / oracle columns are used only after ranking
for evaluation, summary, and diagnostic panels. They are not used in the
LightGlue score.

Command used:
export PYTHONPATH=$PWD/src

python scripts/satloc/s5a/s5a_3_lightglue_topk_verifier.py \
  --run-name top50_same4 \
  --top-k 50 \
  --tokens 129,100,760,833 \
  --resize-long 512 \
  --max-keypoints 1024 \
  --device cpu \
  --save-panels

2. For selection_failure_correct_in_pool frames:
export PYTHONPATH=$PWD/src

python scripts/satloc/s5a/s5a_3_lightglue_topk_verifier.py \
  --run-name top50_selection_destroyed_all11 \
  --top-k 50 \
  --tokens 79,107,129,276,546,694,710,914,1015,269,494 \
  --max-tokens 0 \
  --resize-long 512 \
  --max-keypoints 1024 \
  --device cpu \
  --save-panels

3. re-rreunning LG failed frames:
export PYTHONPATH=$PWD/src

python scripts/satloc/s5a/s5a_3_lightglue_topk_verifier.py \
  --run-name top50_recoverable_missed_panels \
  --top-k 50 \
  --tokens 914,40,946,79 \
  --max-tokens 0 \
  --resize-long 512 \
  --max-keypoints 1024 \
  --device cpu \
  --save-panels
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="S5A.3 LightGlue top-K verifier")

    parser.add_argument(
        "--query-diagnostics",
        type=Path,
        default=Path("outputs/satloc/metadata/s5a_learned_local_verifier/s5a1b_query_diagnostics.csv"),
    )
    parser.add_argument(
        "--candidate-scores",
        type=Path,
        default=Path("outputs/satloc/metadata/s5a_learned_local_verifier/s5a1_local_verifier_candidate_scores.csv"),
    )
    parser.add_argument(
        "--query-summary",
        type=Path,
        default=Path("outputs/satloc/metadata/s5a_learned_local_verifier/s5a1_local_verifier_query_summary.csv"),
    )
    parser.add_argument("--out-base", type=Path, default=Path("outputs/satloc"))

    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--run-name", type=str, default="", help="Optional suffix to avoid overwriting S5A.3 outputs.")
    parser.add_argument("--threshold-m", type=float, default=40.0)

    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--resize-long", type=int, default=512)
    parser.add_argument("--max-keypoints", type=int, default=1024)
    parser.add_argument("--ransac-thresh", type=float, default=5.0)
    parser.add_argument("--max-draw-matches", type=int, default=120)

    parser.add_argument(
        "--classes",
        type=str,
        default="correct_available_but_local_missed,local_destroyed_phog,local_rescue,candidate_pool_not_good_enough",
    )
    parser.add_argument("--tokens-per-class", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=4)
    parser.add_argument(
        "--tokens",
        type=str,
        default="",
        help="Optional comma-separated explicit token list. Overrides class sampling.",
    )

    parser.add_argument("--save-panels", action="store_true")
    parser.add_argument(
        "--panel-top-n",
        type=int,
        default=4,
        help="For each token, show PHOG top1, AKAZE top1, LightGlue top1, and oracle best.",
    )

    return parser.parse_args()


def ensure_dirs(out_base: Path) -> Dict[str, Path]:
    paths = {
        "metadata": out_base / "metadata" / "s5a_learned_local_verifier",
        "reports": out_base / "reports" / "s5a_learned_local_verifier",
        "figures": out_base / "figures" / "s5a_learned_local_verifier",
        "panels": out_base / "figures" / "s5a_learned_local_verifier" / "s5a3_lightglue_topk_panels",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def read_csv(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")
    return pd.read_csv(path)


def safe_str(value: Any) -> str:
    if value is None:
        return ""
    try:
        if isinstance(value, float) and math.isnan(value):
            return ""
    except Exception:
        pass
    return str(value).strip()


def safe_float(value: Any) -> Optional[float]:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def fmt(value: Any, nd: int = 2) -> str:
    value = safe_float(value)
    if value is None:
        return "NA"
    return f"{value:.{nd}f}"


def numeric_col(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series([np.nan] * len(df), index=df.index)
    return pd.to_numeric(df[col], errors="coerce")


def resolve_path(value: Any) -> Optional[Path]:
    text = safe_str(value)
    if not text:
        return None

    p = Path(text)
    if p.exists():
        return p

    p2 = Path.cwd() / p
    if p2.exists():
        return p2

    return None


def read_rgb(path: Path) -> Optional[np.ndarray]:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def resize_longest(img: np.ndarray, longest: int) -> np.ndarray:
    if longest <= 0:
        return img

    h, w = img.shape[:2]
    m = max(h, w)
    if m <= longest:
        return img

    scale = longest / float(m)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def package_exists(name: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(name) is not None


def detect_device(requested: str) -> Dict[str, Any]:
    env = {
        "python": sys.version,
        "platform": platform.platform(),
        "opencv_version": cv2.__version__,
        "torch_available": package_exists("torch"),
        "lightglue_available": package_exists("lightglue"),
        "selected_device": "cpu",
        "cuda_available": False,
        "mps_available": False,
    }

    if not env["torch_available"]:
        return env

    import torch

    env["torch_version"] = torch.__version__
    env["cuda_available"] = bool(torch.cuda.is_available())
    env["mps_available"] = bool(
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    )

    if requested == "cpu":
        env["selected_device"] = "cpu"
    elif requested == "cuda":
        env["selected_device"] = "cuda" if env["cuda_available"] else "cpu"
    elif requested == "mps":
        env["selected_device"] = "mps" if env["mps_available"] else "cpu"
    else:
        if env["cuda_available"]:
            env["selected_device"] = "cuda"
        elif env["mps_available"]:
            env["selected_device"] = "mps"
        else:
            env["selected_device"] = "cpu"

    return env


def torch_tensor_from_rgb(rgb: np.ndarray, device: str):
    import torch

    arr = rgb.astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(device)


def to_numpy_no_batch(x: Any) -> np.ndarray:
    try:
        import torch

        if torch.is_tensor(x):
            y = x.detach().cpu()
            while y.ndim >= 3 and y.shape[0] == 1:
                y = y[0]
            return y.numpy()
    except Exception:
        pass

    if isinstance(x, (list, tuple)):
        if len(x) == 0:
            return np.zeros((0, 2), dtype=np.float32)
        if len(x) == 1:
            return to_numpy_no_batch(x[0])
        return np.asarray(x)

    return np.asarray(x)


def extract_lightglue_matches(matches01: Dict[str, Any], feats0: Dict[str, Any], feats1: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    kpts0 = to_numpy_no_batch(feats0["keypoints"]).astype(np.float32)
    kpts1 = to_numpy_no_batch(feats1["keypoints"]).astype(np.float32)

    if "matches" in matches01:
        matches = to_numpy_no_batch(matches01["matches"])
        matches = np.asarray(matches)

        while matches.ndim >= 3 and matches.shape[0] == 1:
            matches = matches[0]

        if matches.size == 0 or matches.ndim != 2 or matches.shape[1] < 2:
            return np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32)

        matches = matches.astype(np.int64)
        valid = (
            (matches[:, 0] >= 0)
            & (matches[:, 1] >= 0)
            & (matches[:, 0] < len(kpts0))
            & (matches[:, 1] < len(kpts1))
        )
        matches = matches[valid]

        if len(matches) == 0:
            return np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32)

        return kpts0[matches[:, 0]].astype(np.float32), kpts1[matches[:, 1]].astype(np.float32)

    if "matches0" in matches01:
        matches0 = to_numpy_no_batch(matches01["matches0"])
        matches0 = np.asarray(matches0)

        while matches0.ndim >= 2 and matches0.shape[0] == 1:
            matches0 = matches0[0]

        matches0 = matches0.astype(np.int64)
        idx0 = np.where(matches0 > -1)[0]
        idx1 = matches0[idx0]

        valid = (
            (idx0 >= 0)
            & (idx1 >= 0)
            & (idx0 < len(kpts0))
            & (idx1 < len(kpts1))
        )
        idx0 = idx0[valid]
        idx1 = idx1[valid]

        if len(idx0) == 0:
            return np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32)

        return kpts0[idx0].astype(np.float32), kpts1[idx1].astype(np.float32)

    return np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32)


def homography_stats(pts0: np.ndarray, pts1: np.ndarray, thresh: float) -> Tuple[int, float, bool, Optional[np.ndarray]]:
    if pts0 is None or pts1 is None or len(pts0) < 4 or len(pts1) < 4:
        return 0, 0.0, False, None

    src = np.asarray(pts0, dtype=np.float32).reshape(-1, 1, 2)
    dst = np.asarray(pts1, dtype=np.float32).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, thresh)
    if mask is None:
        return 0, 0.0, False, None

    mask_bool = mask.ravel().astype(bool)
    inliers = int(mask_bool.sum())
    ratio = float(inliers / max(1, len(mask_bool)))
    ok = bool(H is not None and inliers >= 4)

    return inliers, ratio, ok, mask_bool


def grid_coverage(points: np.ndarray, shape_hw: Tuple[int, int], grid: int = 4) -> float:
    if points is None or len(points) == 0:
        return 0.0

    h, w = shape_hw
    if h <= 0 or w <= 0:
        return 0.0

    pts = np.asarray(points, dtype=np.float32)
    xs = np.clip(pts[:, 0], 0, w - 1)
    ys = np.clip(pts[:, 1], 0, h - 1)

    gx = np.floor(xs / max(1e-6, w / grid)).astype(int)
    gy = np.floor(ys / max(1e-6, h / grid)).astype(int)
    gx = np.clip(gx, 0, grid - 1)
    gy = np.clip(gy, 0, grid - 1)

    occupied = set(zip(gx.tolist(), gy.tolist()))
    return float(len(occupied) / float(grid * grid))


def draw_matches_canvas(
    rgb0: np.ndarray,
    rgb1: np.ndarray,
    pts0: np.ndarray,
    pts1: np.ndarray,
    inlier_mask: Optional[np.ndarray],
    max_draw: int,
) -> np.ndarray:
    h0, w0 = rgb0.shape[:2]
    h1, w1 = rgb1.shape[:2]

    canvas = np.zeros((max(h0, h1), w0 + w1, 3), dtype=np.uint8)
    canvas[:h0, :w0] = rgb0
    canvas[:h1, w0:w0 + w1] = rgb1

    if pts0 is None or pts1 is None or len(pts0) == 0:
        return canvas

    pts0 = np.asarray(pts0, dtype=np.float32)
    pts1 = np.asarray(pts1, dtype=np.float32)

    if inlier_mask is not None and len(inlier_mask) == len(pts0):
        inlier_mask = inlier_mask.astype(bool)
        order = np.concatenate([np.where(inlier_mask)[0], np.where(~inlier_mask)[0]])
    else:
        inlier_mask = None
        order = np.arange(len(pts0))

    order = order[:max_draw]

    for idx in order:
        x0, y0 = int(round(pts0[idx, 0])), int(round(pts0[idx, 1]))
        x1, y1 = int(round(pts1[idx, 0] + w0)), int(round(pts1[idx, 1]))

        is_inlier = True if inlier_mask is None else bool(inlier_mask[idx])
        color = (0, 255, 0) if is_inlier else (255, 60, 60)

        cv2.circle(canvas, (x0, y0), 3, color, -1)
        cv2.circle(canvas, (x1, y1), 3, color, -1)
        cv2.line(canvas, (x0, y0), (x1, y1), color, 1)

    return canvas


class LightGlueRunner:
    def __init__(self, device: str, max_keypoints: int, resize_long: int):
        from lightglue import LightGlue, SuperPoint

        self.device = device
        self.max_keypoints = max_keypoints
        self.resize_long = resize_long

        self.extractor = SuperPoint(max_num_keypoints=max_keypoints).eval().to(device)
        self.matcher = LightGlue(features="superpoint").eval().to(device)

        self.image_cache: Dict[str, np.ndarray] = {}
        self.feature_cache: Dict[str, Dict[str, Any]] = {}

    def load_image(self, path: Path) -> np.ndarray:
        key = str(path)
        if key in self.image_cache:
            return self.image_cache[key]

        rgb = read_rgb(path)
        if rgb is None:
            raise RuntimeError(f"Could not read image: {path}")

        rgb = resize_longest(rgb, self.resize_long)
        self.image_cache[key] = rgb
        return rgb

    def extract(self, path: Path) -> Dict[str, Any]:
        import torch

        key = str(path)
        if key in self.feature_cache:
            return self.feature_cache[key]

        rgb = self.load_image(path)
        tensor = torch_tensor_from_rgb(rgb, self.device)

        with torch.no_grad():
            try:
                feats = self.extractor.extract(tensor, resize=None)
            except TypeError:
                feats = self.extractor.extract(tensor)

        self.feature_cache[key] = feats
        return feats

    def match(self, path0: Path, path1: Path) -> Dict[str, Any]:
        import torch

        start = time.time()

        rgb0 = self.load_image(path0)
        rgb1 = self.load_image(path1)
        feats0 = self.extract(path0)
        feats1 = self.extract(path1)

        with torch.no_grad():
            matches01 = self.matcher({"image0": feats0, "image1": feats1})

        pts0, pts1 = extract_lightglue_matches(matches01, feats0, feats1)

        return {
            "status": "ok",
            "error": "",
            "rgb0": rgb0,
            "rgb1": rgb1,
            "pts0": pts0,
            "pts1": pts1,
            "matches": int(len(pts0)),
            "runtime_s": float(time.time() - start),
        }


def candidate_image_path(row: pd.Series) -> Optional[Path]:
    for col in [
        "candidate_image_path",
        "sat_image_path",
        "satellite_image_path",
        "tile_image_path",
        "image_path",
        "tile_path",
    ]:
        if col in row.index:
            p = resolve_path(row.get(col))
            if p is not None:
                return p
    return None


def merge_uav_paths(qdiag: pd.DataFrame, qsum: pd.DataFrame) -> pd.DataFrame:
    q = qdiag.copy()
    q["token_str"] = q["token"].astype(str)
    qsum = qsum.copy()
    qsum["token_str"] = qsum["token"].astype(str)

    if "uav_image_path" not in q.columns:
        q = q.merge(qsum[["token_str", "uav_image_path"]], on="token_str", how="left")
    else:
        q = q.merge(
            qsum[["token_str", "uav_image_path"]],
            on="token_str",
            how="left",
            suffixes=("", "_from_summary"),
        )
        if "uav_image_path_from_summary" in q.columns:
            q["uav_image_path"] = q["uav_image_path"].fillna(q["uav_image_path_from_summary"])

    return q


def choose_tokens(qdiag: pd.DataFrame, args: argparse.Namespace) -> List[str]:
    q = qdiag.copy()
    q["token_str"] = q["token"].astype(str)

    if args.tokens.strip():
        toks = [t.strip() for t in args.tokens.split(",") if t.strip()]
        return toks[: args.max_tokens] if args.max_tokens > 0 else toks

    selected: List[str] = []
    classes = [c.strip() for c in args.classes.split(",") if c.strip()]

    for cls in classes:
        sub = q[q["local_decision_class"].astype(str) == cls].copy()
        if len(sub) == 0:
            continue

        if "local_minus_phog_error_m" in sub.columns:
            sub["abs_delta"] = pd.to_numeric(sub["local_minus_phog_error_m"], errors="coerce").abs()
            sub = sub.sort_values("abs_delta", ascending=False, kind="mergesort")

        selected.extend(sub["token_str"].head(args.tokens_per_class).tolist())

    seen = set()
    out = []
    for t in selected:
        if t not in seen:
            seen.add(t)
            out.append(t)

    if args.max_tokens > 0:
        out = out[: args.max_tokens]

    return out


def lightglue_score(
    matches: int,
    inliers: int,
    inlier_ratio: float,
    uav_cov: float,
    sat_cov: float,
    homography_success: bool,
) -> float:
    """
    Unsupervised candidate score.
    No reference/error/GT is used here.

    Rationale:
    - inliers are the main evidence.
    - raw matches help slightly.
    - inlier ratio penalizes noisy matches.
    - min coverage penalizes matches clustered in one vegetation patch.
    - failed homography is allowed but receives no homography bonus.
    """
    spread = min(float(uav_cov), float(sat_cov))
    score = (
        float(inliers)
        + 0.04 * float(matches)
        + 4.0 * float(inlier_ratio)
        + 6.0 * float(spread)
    )
    if homography_success:
        score += 2.0
    return float(score)


def process_candidate(
    runner: LightGlueRunner,
    token: str,
    qrow: pd.Series,
    crow: pd.Series,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    uav_path = resolve_path(qrow.get("uav_image_path"))
    cand_path = candidate_image_path(crow)

    base: Dict[str, Any] = {
        "token": token,
        "failure_group": safe_str(qrow.get("failure_group")),
        "local_decision_class": safe_str(qrow.get("local_decision_class")),
        "uav_image_path": str(uav_path) if uav_path else "",
        "candidate_image_path": str(cand_path) if cand_path else "",
        "tile_id": safe_str(crow.get("tile_id")),
        "candidate_pool_rank": safe_float(crow.get("candidate_pool_rank")),
        "local_verifier_rank": safe_float(crow.get("local_verifier_rank")),
        "phog_score": safe_float(crow.get("phog_score")),
        "akaze_score": safe_float(crow.get("local_score")),
        "akaze_good_matches": safe_float(crow.get("good_matches")),
        "akaze_ransac_inliers": safe_float(crow.get("ransac_inliers")),
        "eval_error_m": safe_float(crow.get("eval_error_m")),
        "lightglue_status": "not_started",
        "lightglue_error": "",
        "lightglue_matches": 0,
        "lightglue_ransac_inliers": 0,
        "lightglue_inlier_ratio": 0.0,
        "lightglue_homography_success": False,
        "lightglue_uav_coverage": 0.0,
        "lightglue_sat_coverage": 0.0,
        "lightglue_score": -1.0,
        "runtime_s": 0.0,
    }

    if uav_path is None or cand_path is None:
        base["lightglue_status"] = "missing_path"
        return base

    try:
        result = runner.match(uav_path, cand_path)
        pts0 = result["pts0"]
        pts1 = result["pts1"]

        inliers, inlier_ratio, h_ok, mask = homography_stats(
            pts0,
            pts1,
            args.ransac_thresh,
        )

        if mask is not None and len(mask) == len(pts0):
            inlier_pts0 = pts0[mask]
            inlier_pts1 = pts1[mask]
        else:
            inlier_pts0 = pts0
            inlier_pts1 = pts1

        rgb0 = result["rgb0"]
        rgb1 = result["rgb1"]

        uav_cov = grid_coverage(inlier_pts0, rgb0.shape[:2], grid=4)
        sat_cov = grid_coverage(inlier_pts1, rgb1.shape[:2], grid=4)

        score = lightglue_score(
            matches=int(result["matches"]),
            inliers=int(inliers),
            inlier_ratio=float(inlier_ratio),
            uav_cov=float(uav_cov),
            sat_cov=float(sat_cov),
            homography_success=bool(h_ok),
        )

        base.update(
            {
                "lightglue_status": "ok",
                "lightglue_matches": int(result["matches"]),
                "lightglue_ransac_inliers": int(inliers),
                "lightglue_inlier_ratio": float(inlier_ratio),
                "lightglue_homography_success": bool(h_ok),
                "lightglue_uav_coverage": float(uav_cov),
                "lightglue_sat_coverage": float(sat_cov),
                "lightglue_score": float(score),
                "runtime_s": float(result["runtime_s"]),
            }
        )
        return base

    except Exception as exc:
        base["lightglue_status"] = "failed"
        base["lightglue_error"] = repr(exc)
        base["traceback"] = traceback.format_exc(limit=2)
        return base


def rank_candidates(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) == 0:
        return df

    out = df.copy()
    out["lightglue_score_num"] = numeric_col(out, "lightglue_score")
    out["lightglue_ransac_inliers_num"] = numeric_col(out, "lightglue_ransac_inliers")
    out["lightglue_matches_num"] = numeric_col(out, "lightglue_matches")
    out["lightglue_uav_coverage_num"] = numeric_col(out, "lightglue_uav_coverage")
    out["candidate_pool_rank_num"] = numeric_col(out, "candidate_pool_rank")

    ranked_parts = []
    for token, g in out.groupby("token", dropna=False):
        gg = g.sort_values(
            [
                "lightglue_score_num",
                "lightglue_ransac_inliers_num",
                "lightglue_matches_num",
                "lightglue_uav_coverage_num",
                "candidate_pool_rank_num",
            ],
            ascending=[False, False, False, False, True],
            kind="mergesort",
        ).copy()
        gg["lightglue_rank"] = np.arange(1, len(gg) + 1)
        ranked_parts.append(gg)

    return pd.concat(ranked_parts, ignore_index=True)


def summarize_queries(ranked: pd.DataFrame, threshold_m: float) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    if len(ranked) == 0:
        return pd.DataFrame()

    ranked = ranked.copy()
    ranked["eval_error_num"] = numeric_col(ranked, "eval_error_m")
    ranked["candidate_pool_rank_num"] = numeric_col(ranked, "candidate_pool_rank")
    ranked["local_verifier_rank_num"] = numeric_col(ranked, "local_verifier_rank")
    ranked["lightglue_rank_num"] = numeric_col(ranked, "lightglue_rank")

    for token, g in ranked.groupby("token", dropna=False):
        g = g.copy()

        phog_top = g.sort_values("candidate_pool_rank_num", kind="mergesort").iloc[0]
        lg_top = g.sort_values("lightglue_rank_num", kind="mergesort").iloc[0]

        if g["local_verifier_rank_num"].notna().any():
            local_top = g.sort_values("local_verifier_rank_num", kind="mergesort").iloc[0]
        else:
            local_top = None

        valid = g.dropna(subset=["eval_error_num"])
        oracle = valid.sort_values("eval_error_num", kind="mergesort").iloc[0] if len(valid) else None

        phog_err = safe_float(phog_top.get("eval_error_m"))
        lg_err = safe_float(lg_top.get("eval_error_m"))
        local_err = safe_float(local_top.get("eval_error_m")) if local_top is not None else None
        oracle_err = safe_float(oracle.get("eval_error_m")) if oracle is not None else None

        rows.append(
            {
                "token": token,
                "failure_group": safe_str(phog_top.get("failure_group")),
                "local_decision_class": safe_str(phog_top.get("local_decision_class")),
                "processed_candidates": int(len(g)),

                "phog_top1_tile_id": safe_str(phog_top.get("tile_id")),
                "phog_top1_error_m": phog_err,
                "phog_hit_le_threshold": bool(phog_err is not None and phog_err <= threshold_m),

                "akaze_top1_tile_id": safe_str(local_top.get("tile_id")) if local_top is not None else "",
                "akaze_top1_error_m": local_err,
                "akaze_hit_le_threshold": bool(local_err is not None and local_err <= threshold_m),

                "lightglue_top1_tile_id": safe_str(lg_top.get("tile_id")),
                "lightglue_top1_error_m": lg_err,
                "lightglue_hit_le_threshold": bool(lg_err is not None and lg_err <= threshold_m),
                "lightglue_top1_score": safe_float(lg_top.get("lightglue_score")),
                "lightglue_top1_matches": safe_float(lg_top.get("lightglue_matches")),
                "lightglue_top1_inliers": safe_float(lg_top.get("lightglue_ransac_inliers")),
                "lightglue_top1_inlier_ratio": safe_float(lg_top.get("lightglue_inlier_ratio")),
                "lightglue_top1_uav_coverage": safe_float(lg_top.get("lightglue_uav_coverage")),
                "lightglue_top1_sat_coverage": safe_float(lg_top.get("lightglue_sat_coverage")),

                "oracle_topk_tile_id": safe_str(oracle.get("tile_id")) if oracle is not None else "",
                "oracle_topk_error_m": oracle_err,
                "oracle_topk_hit_le_threshold": bool(oracle_err is not None and oracle_err <= threshold_m),
                "oracle_lightglue_rank": safe_float(oracle.get("lightglue_rank")) if oracle is not None else None,
                "oracle_lightglue_score": safe_float(oracle.get("lightglue_score")) if oracle is not None else None,
            }
        )

    return pd.DataFrame(rows)


def group_summary(query_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if len(query_summary) == 0:
        return pd.DataFrame()

    for group, g in query_summary.groupby("failure_group", dropna=False):
        rows.append(
            {
                "failure_group": safe_str(group),
                "count": int(len(g)),
                "phog_hit_rate": float(g["phog_hit_le_threshold"].mean()),
                "akaze_hit_rate": float(g["akaze_hit_le_threshold"].mean()),
                "lightglue_hit_rate": float(g["lightglue_hit_le_threshold"].mean()),
                "oracle_topk_hit_rate": float(g["oracle_topk_hit_le_threshold"].mean()),
                "phog_median_error_m": safe_float(pd.to_numeric(g["phog_top1_error_m"], errors="coerce").median()),
                "akaze_median_error_m": safe_float(pd.to_numeric(g["akaze_top1_error_m"], errors="coerce").median()),
                "lightglue_median_error_m": safe_float(pd.to_numeric(g["lightglue_top1_error_m"], errors="coerce").median()),
                "oracle_topk_median_error_m": safe_float(pd.to_numeric(g["oracle_topk_error_m"], errors="coerce").median()),
                "oracle_lightglue_rank_median": safe_float(pd.to_numeric(g["oracle_lightglue_rank"], errors="coerce").median()),
            }
        )

    return pd.DataFrame(rows).sort_values("failure_group")


def plot_hit_rates(group_df: pd.DataFrame, out_path: Path) -> None:
    if len(group_df) == 0:
        return

    x = np.arange(len(group_df))
    width = 0.20

    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.bar(x - 1.5 * width, group_df["phog_hit_rate"], width, label="PHOG")
    ax.bar(x - 0.5 * width, group_df["akaze_hit_rate"], width, label="AKAZE")
    ax.bar(x + 0.5 * width, group_df["lightglue_hit_rate"], width, label="LightGlue")
    ax.bar(x + 1.5 * width, group_df["oracle_topk_hit_rate"], width, label="Oracle in processed top-K")

    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Hit rate <= threshold")
    ax.set_title("S5A.3 hit-rate comparison by failure group")
    ax.set_xticks(x)
    ax.set_xticklabels(group_df["failure_group"].tolist(), rotation=35, ha="right")
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_median_errors(group_df: pd.DataFrame, out_path: Path) -> None:
    if len(group_df) == 0:
        return

    x = np.arange(len(group_df))
    width = 0.20

    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.bar(x - 1.5 * width, group_df["phog_median_error_m"], width, label="PHOG")
    ax.bar(x - 0.5 * width, group_df["akaze_median_error_m"], width, label="AKAZE")
    ax.bar(x + 0.5 * width, group_df["lightglue_median_error_m"], width, label="LightGlue")
    ax.bar(x + 1.5 * width, group_df["oracle_topk_median_error_m"], width, label="Oracle")

    ax.set_ylabel("Median top-1 error [m]")
    ax.set_title("S5A.3 median error comparison by failure group")
    ax.set_xticks(x)
    ax.set_xticklabels(group_df["failure_group"].tolist(), rotation=35, ha="right")
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_score_vs_error(ranked: pd.DataFrame, out_path: Path) -> None:
    if len(ranked) == 0:
        return

    df = ranked.copy()
    df["eval_error_num"] = numeric_col(df, "eval_error_m")
    df["score_num"] = numeric_col(df, "lightglue_score")
    df = df.dropna(subset=["eval_error_num", "score_num"])

    if len(df) == 0:
        return

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.scatter(df["score_num"], df["eval_error_num"], s=18, alpha=0.55)
    ax.set_xlabel("LightGlue verifier score")
    ax.set_ylabel("Evaluation error [m]")
    ax.set_title("S5A.3 LightGlue score vs post-ranking error")
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def add_image(ax, img: Optional[np.ndarray], title: str) -> None:
    ax.axis("off")
    ax.set_title(title, fontsize=9)
    if img is None:
        ax.text(0.5, 0.5, "missing", ha="center", va="center")
    else:
        ax.imshow(img)


def role_rows_for_panel(ranked_token: pd.DataFrame) -> List[Tuple[str, pd.Series]]:
    g = ranked_token.copy()
    g["candidate_pool_rank_num"] = numeric_col(g, "candidate_pool_rank")
    g["local_verifier_rank_num"] = numeric_col(g, "local_verifier_rank")
    g["lightglue_rank_num"] = numeric_col(g, "lightglue_rank")
    g["eval_error_num"] = numeric_col(g, "eval_error_m")

    rows: List[Tuple[str, pd.Series]] = []

    rows.append(("PHOG top1", g.sort_values("candidate_pool_rank_num", kind="mergesort").iloc[0]))

    if g["local_verifier_rank_num"].notna().any():
        rows.append(("AKAZE top1", g.sort_values("local_verifier_rank_num", kind="mergesort").iloc[0]))

    rows.append(("LightGlue top1", g.sort_values("lightglue_rank_num", kind="mergesort").iloc[0]))

    valid = g.dropna(subset=["eval_error_num"])
    if len(valid):
        rows.append(("Oracle best in processed top-K", valid.sort_values("eval_error_num", kind="mergesort").iloc[0]))

    # Remove duplicate tile roles while preserving role label by allowing same image if it is important.
    return rows[:4]


def save_token_panel(
    runner: LightGlueRunner,
    ranked_token: pd.DataFrame,
    qrow: pd.Series,
    out_path: Path,
    args: argparse.Namespace,
) -> None:
    token = safe_str(qrow.get("token"))
    group = safe_str(qrow.get("failure_group"))
    decision = safe_str(qrow.get("local_decision_class"))
    uav_path = resolve_path(qrow.get("uav_image_path"))
    if uav_path is None:
        return

    rows = role_rows_for_panel(ranked_token)

    fig, axes = plt.subplots(len(rows), 3, figsize=(17, 4.2 * len(rows)))
    if len(rows) == 1:
        axes = np.asarray([axes])

    fig.suptitle(
        f"S5A.3 LightGlue top-K verifier panel | token={token} | group={group} | class={decision}\n"
        "Green = RANSAC inlier match, red = rejected/outlier. GT/error shown only after ranking.",
        fontsize=12,
    )

    for i, (role, row) in enumerate(rows):
        cand_path = resolve_path(row.get("candidate_image_path"))
        if cand_path is None:
            cand_path = candidate_image_path(row)

        match_img = None
        uav_rgb = None
        cand_rgb = None

        if cand_path is not None:
            result = runner.match(uav_path, cand_path)
            pts0, pts1 = result["pts0"], result["pts1"]
            inliers, ratio, h_ok, mask = homography_stats(pts0, pts1, args.ransac_thresh)
            uav_rgb, cand_rgb = result["rgb0"], result["rgb1"]
            match_img = draw_matches_canvas(
                uav_rgb,
                cand_rgb,
                pts0,
                pts1,
                mask,
                args.max_draw_matches,
            )

        role_title = (
            f"{role}\n"
            f"tile={safe_str(row.get('tile_id'))} | "
            f"PHOG-rank={fmt(row.get('candidate_pool_rank'), 0)} | "
            f"AKAZE-rank={fmt(row.get('local_verifier_rank'), 0)} | "
            f"LG-rank={fmt(row.get('lightglue_rank'), 0)} | "
            f"err={fmt(row.get('eval_error_m'), 1)}m"
        )

        metrics_title = (
            f"LG score={fmt(row.get('lightglue_score'), 2)} | "
            f"matches={fmt(row.get('lightglue_matches'), 0)} | "
            f"inliers={fmt(row.get('lightglue_ransac_inliers'), 0)} | "
            f"ratio={fmt(row.get('lightglue_inlier_ratio'), 2)} | "
            f"covU={fmt(row.get('lightglue_uav_coverage'), 2)} | "
            f"covS={fmt(row.get('lightglue_sat_coverage'), 2)}"
        )

        add_image(axes[i, 0], uav_rgb, f"UAV input\n{token}")
        add_image(axes[i, 1], cand_rgb, role_title)
        add_image(axes[i, 2], match_img, metrics_title)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    dirs = ensure_dirs(args.out_base)
    run_suffix = f"_{args.run_name}" if args.run_name else ""

    env = detect_device(args.device)
    if not env.get("torch_available") or not env.get("lightglue_available"):
        raise RuntimeError(
            "Torch/LightGlue not available. Run S5A.2B install/verify before S5A.3."
        )

    qdiag = read_csv(args.query_diagnostics, "S5A.1B query diagnostics")
    cand = read_csv(args.candidate_scores, "S5A.1 candidate scores")
    qsum = read_csv(args.query_summary, "S5A.1 query summary")

    qdiag = merge_uav_paths(qdiag, qsum)
    qdiag["token_str"] = qdiag["token"].astype(str)

    cand = cand.copy()
    cand["token_str"] = cand["token"].astype(str)
    cand["candidate_pool_rank_num"] = numeric_col(cand, "candidate_pool_rank")

    tokens = choose_tokens(qdiag, args)

    runner = LightGlueRunner(
        device=safe_str(env.get("selected_device", "cpu")),
        max_keypoints=args.max_keypoints,
        resize_long=args.resize_long,
    )

    processed_rows: List[Dict[str, Any]] = []

    for token in tokens:
        qsub = qdiag[qdiag["token_str"] == str(token)]
        if len(qsub) == 0:
            print(f"[WARN] token {token}: missing query diagnostics")
            continue
        qrow = qsub.iloc[0]

        csub = cand[cand["token_str"] == str(token)].copy()
        csub = csub.sort_values("candidate_pool_rank_num", kind="mergesort").head(args.top_k)

        print(f"[S5A.3] token={token} candidates={len(csub)} group={safe_str(qrow.get('failure_group'))}")

        for _, crow in csub.iterrows():
            row = process_candidate(runner, str(token), qrow, crow, args)
            processed_rows.append(row)

    candidate_df = pd.DataFrame(processed_rows)
    ranked_df = rank_candidates(candidate_df)
    query_df = summarize_queries(ranked_df, args.threshold_m)
    group_df = group_summary(query_df)

    candidate_out = dirs["metadata"] / f"s5a3_lightglue_candidate_scores{run_suffix}.csv"
    query_out = dirs["metadata"] / f"s5a3_lightglue_query_summary{run_suffix}.csv"
    group_out = dirs["metadata"] / f"s5a3_lightglue_group_summary{run_suffix}.csv"
    summary_out = dirs["reports"] / f"s5a3_lightglue_topk_verifier_summary{run_suffix}.json"
    env_out = dirs["reports"] / f"s5a3_environment{run_suffix}.json"

    ranked_df.to_csv(candidate_out, index=False)
    query_df.to_csv(query_out, index=False)
    group_df.to_csv(group_out, index=False)

    fig_hit = dirs["figures"] / f"s5a3_hit_rate_by_group{run_suffix}.png"
    fig_err = dirs["figures"] / f"s5a3_median_error_by_group{run_suffix}.png"
    fig_score = dirs["figures"] / f"s5a3_lightglue_score_vs_error{run_suffix}.png"

    plot_hit_rates(group_df, fig_hit)
    plot_median_errors(group_df, fig_err)
    plot_score_vs_error(ranked_df, fig_score)

    panel_paths: List[str] = []
    if args.save_panels:
        for token in tokens:
            qsub = qdiag[qdiag["token_str"] == str(token)]
            rsub = ranked_df[ranked_df["token"].astype(str) == str(token)]
            if len(qsub) == 0 or len(rsub) == 0:
                continue

            qrow = qsub.iloc[0]
            group = safe_str(qrow.get("failure_group")).replace("/", "_").replace(" ", "_")
            decision = safe_str(qrow.get("local_decision_class")).replace("/", "_").replace(" ", "_")
            panel_base_dir = dirs["panels"] / args.run_name if args.run_name else dirs["panels"]
            out_path = panel_base_dir / group / decision / f"s5a3_token{int(float(token)):04d}_lightglue_ranking_panel.png"
            save_token_panel(runner, rsub, qrow, out_path, args)
            panel_paths.append(str(out_path))

    with open(env_out, "w", encoding="utf-8") as f:
        json.dump(env, f, indent=2)

    summary: Dict[str, Any] = {
        "stage": "S5A.3_lightglue_topk_verifier",
        "locked_rule": "reference/error columns are evaluation-only; LightGlue score uses only matches, inliers, inlier ratio, coverage, and homography success",
        "top_k": args.top_k,
        "threshold_m": args.threshold_m,
        "tokens": tokens,
        "num_tokens": int(len(query_df)),
        "candidate_rows": int(len(ranked_df)),
        "status_counts": ranked_df["lightglue_status"].value_counts(dropna=False).to_dict() if len(ranked_df) else {},
        "phog_hit_rate": float(query_df["phog_hit_le_threshold"].mean()) if len(query_df) else 0.0,
        "akaze_hit_rate": float(query_df["akaze_hit_le_threshold"].mean()) if len(query_df) else 0.0,
        "lightglue_hit_rate": float(query_df["lightglue_hit_le_threshold"].mean()) if len(query_df) else 0.0,
        "oracle_topk_hit_rate": float(query_df["oracle_topk_hit_le_threshold"].mean()) if len(query_df) else 0.0,
        "phog_median_error_m": safe_float(pd.to_numeric(query_df["phog_top1_error_m"], errors="coerce").median()) if len(query_df) else None,
        "akaze_median_error_m": safe_float(pd.to_numeric(query_df["akaze_top1_error_m"], errors="coerce").median()) if len(query_df) else None,
        "lightglue_median_error_m": safe_float(pd.to_numeric(query_df["lightglue_top1_error_m"], errors="coerce").median()) if len(query_df) else None,
        "oracle_topk_median_error_m": safe_float(pd.to_numeric(query_df["oracle_topk_error_m"], errors="coerce").median()) if len(query_df) else None,
        "mean_runtime_per_candidate_s": safe_float(pd.to_numeric(ranked_df["runtime_s"], errors="coerce").mean()) if len(ranked_df) else None,
        "candidate_scores_csv": str(candidate_out),
        "query_summary_csv": str(query_out),
        "group_summary_csv": str(group_out),
        "environment_json": str(env_out),
        "hit_rate_figure": str(fig_hit),
        "median_error_figure": str(fig_err),
        "score_vs_error_figure": str(fig_score),
        "panel_paths": panel_paths,
    }

    with open(summary_out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("S5A.3 LightGlue top-K verifier complete")
    print("---------------------------------------")
    print(f"Tokens processed:          {summary['num_tokens']} -> {tokens}")
    print(f"Top-K per token:           {args.top_k}")
    print(f"Candidate rows:            {summary['candidate_rows']}")
    print(f"Status counts:             {summary['status_counts']}")
    print(f"PHOG hit <=thr:            {summary['phog_hit_rate']:.3f}")
    print(f"AKAZE hit <=thr:           {summary['akaze_hit_rate']:.3f}")
    print(f"LightGlue hit <=thr:       {summary['lightglue_hit_rate']:.3f}")
    print(f"Oracle processed-K <=thr:  {summary['oracle_topk_hit_rate']:.3f}")
    print(f"PHOG median error:         {summary['phog_median_error_m']} m")
    print(f"AKAZE median error:        {summary['akaze_median_error_m']} m")
    print(f"LightGlue median error:    {summary['lightglue_median_error_m']} m")
    print(f"Oracle median error:       {summary['oracle_topk_median_error_m']} m")
    print(f"Mean runtime/candidate:    {summary['mean_runtime_per_candidate_s']} s")
    print(f"Candidate scores CSV:      {candidate_out}")
    print(f"Query summary CSV:         {query_out}")
    print(f"Group summary CSV:         {group_out}")
    print(f"Summary JSON:              {summary_out}")
    print(f"Figures:                   {fig_hit}")
    print(f"                           {fig_err}")
    print(f"                           {fig_score}")
    if args.save_panels:
        print(f"Panels saved:              {len(panel_paths)}")
        print(f"Panel dir:                 {dirs['panels']}")
    print()
#    print("Locked rule: reference/error columns were used only after LightGlue ranking for evaluation and diagnostic labels.")


if __name__ == "__main__":
    main()
