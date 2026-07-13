#!/usr/bin/env python3
"""
S5A.2 — Learned verifier preflight inside PHOG top-K

Purpose
-------
Preflight learned/local matchers before running a full benchmark.

This script:
1. Checks environment support for torch, LightGlue/SuperPoint, and Kornia LoFTR.
2. Builds a small smoke-test pair manifest from S5A.1B failure diagnostics.
3. For selected tokens, compares:
   - PHOG top1 candidate
   - OpenCV local top1 candidate
   - Oracle-best candidate inside PHOG top-K
4. Optionally runs a learned matcher if available.
5. Saves inner-working panels for learning/debugging.

Locked rule
-----------
Reference/error columns are used only after ranking for smoke-test labels and
oracle visualization. They are not used for retrieval/ranking/scoring.

Code Used:
export PYTHONPATH=$PWD/src

python scripts/satloc/s5a/s5a_2_learned_verifier_preflight.py \
  --backend auto \
  --tokens-per-class 1 \
  --max-pairs 12 \
  --resize-long 768


command ran after installing Loftr, lightglue:
export PYTHONPATH=$PWD/src

python scripts/satloc/s5a/s5a_2_learned_verifier_preflight.py \
  --backend lightglue_superpoint \
  --tokens-per-class 1 \
  --max-pairs 12 \
  --resize-long 768 \
  --max-keypoints 2048
"""

from __future__ import annotations

import argparse
import importlib.util
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
    parser = argparse.ArgumentParser(description="S5A.2 learned verifier preflight")

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
    parser.add_argument(
        "--out-base",
        type=Path,
        default=Path("outputs/satloc"),
    )

    parser.add_argument(
        "--backend",
        choices=["auto", "lightglue_superpoint", "loftr"],
        default="auto",
        help="Learned matcher backend. auto tries LightGlue first, then LoFTR.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
    )
    parser.add_argument("--resize-long", type=int, default=768)
    parser.add_argument("--max-keypoints", type=int, default=2048)
    parser.add_argument("--ransac-thresh", type=float, default=5.0)
    parser.add_argument("--max-draw-matches", type=int, default=120)

    parser.add_argument(
        "--classes",
        type=str,
        default="correct_available_but_local_missed,local_destroyed_phog,local_rescue,candidate_pool_not_good_enough",
        help="Comma-separated local_decision_class values to sample.",
    )
    parser.add_argument("--tokens-per-class", type=int, default=1)
    parser.add_argument(
        "--tokens",
        type=str,
        default="",
        help="Optional explicit comma-separated token list. Overrides class sampling.",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=12,
        help="Maximum image pairs to run through learned matcher.",
    )
    parser.add_argument(
        "--skip-model-run",
        action="store_true",
        help="Only build manifest and visual input panels; do not run learned backend.",
    )

    return parser.parse_args()


def ensure_dirs(out_base: Path) -> Dict[str, Path]:
    paths = {
        "metadata": out_base / "metadata" / "s5a_learned_local_verifier",
        "reports": out_base / "reports" / "s5a_learned_local_verifier",
        "figures": out_base / "figures" / "s5a_learned_local_verifier",
        "panels": out_base / "figures" / "s5a_learned_local_verifier" / "s5a2_learned_preflight_panels",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def read_csv(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")
    return pd.read_csv(path)


def package_exists(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


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


def fmt_float(value: Any, digits: int = 1) -> str:
    value = safe_float(value)
    if value is None:
        return "NA"
    return f"{value:.{digits}f}"


def resolve_path(value: Any) -> Optional[Path]:
    text = safe_str(value)
    if not text:
        return None

    p = Path(text)
    if p.exists():
        return p

    root_p = Path.cwd() / p
    if root_p.exists():
        return root_p

    return None


def numeric_col(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series([np.nan] * len(df), index=df.index)
    return pd.to_numeric(df[col], errors="coerce")


def detect_environment(requested_device: str) -> Dict[str, Any]:
    env: Dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "opencv_version": cv2.__version__,
        "torch_available": package_exists("torch"),
        "lightglue_available": package_exists("lightglue"),
        "kornia_available": package_exists("kornia"),
        "transformers_available": package_exists("transformers"),
        "timm_available": package_exists("timm"),
        "requested_device": requested_device,
        "selected_device": "cpu",
        "torch_version": None,
        "cuda_available": False,
        "mps_available": False,
    }

    if not env["torch_available"]:
        return env

    try:
        import torch

        env["torch_version"] = torch.__version__
        env["cuda_available"] = bool(torch.cuda.is_available())
        env["mps_available"] = bool(
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        )

        if requested_device == "cpu":
            env["selected_device"] = "cpu"
        elif requested_device == "cuda":
            env["selected_device"] = "cuda" if env["cuda_available"] else "cpu"
        elif requested_device == "mps":
            env["selected_device"] = "mps" if env["mps_available"] else "cpu"
        else:
            if env["cuda_available"]:
                env["selected_device"] = "cuda"
            elif env["mps_available"]:
                env["selected_device"] = "mps"
            else:
                env["selected_device"] = "cpu"

    except Exception as exc:
        env["torch_error"] = repr(exc)

    return env


def select_backend(args: argparse.Namespace, env: Dict[str, Any]) -> str:
    if args.backend != "auto":
        return args.backend

    if env.get("lightglue_available", False) and env.get("torch_available", False):
        return "lightglue_superpoint"

    if env.get("kornia_available", False) and env.get("torch_available", False):
        return "loftr"

    return "none"


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
    max_side = max(h, w)
    if max_side <= longest:
        return img

    scale = longest / float(max_side)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def resize_longest_multiple(img: np.ndarray, longest: int, multiple: int = 8) -> np.ndarray:
    out = resize_longest(img, longest)
    h, w = out.shape[:2]

    h2 = max(multiple, (h // multiple) * multiple)
    w2 = max(multiple, (w // multiple) * multiple)

    if h2 != h or w2 != w:
        out = cv2.resize(out, (w2, h2), interpolation=cv2.INTER_AREA)

    return out


def rgb_to_gray(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def draw_points_matches(
    rgb0: np.ndarray,
    rgb1: np.ndarray,
    pts0: np.ndarray,
    pts1: np.ndarray,
    inlier_mask: Optional[np.ndarray],
    max_draw: int,
) -> np.ndarray:
    img0 = rgb0.copy()
    img1 = rgb1.copy()

    h0, w0 = img0.shape[:2]
    h1, w1 = img1.shape[:2]
    canvas_h = max(h0, h1)
    canvas_w = w0 + w1

    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    canvas[:h0, :w0] = img0
    canvas[:h1, w0:w0 + w1] = img1

    if pts0 is None or pts1 is None or len(pts0) == 0:
        return canvas

    pts0 = np.asarray(pts0, dtype=np.float32)
    pts1 = np.asarray(pts1, dtype=np.float32)

    if inlier_mask is not None and len(inlier_mask) == len(pts0):
        order = np.argsort(~inlier_mask.astype(bool))
    else:
        order = np.arange(len(pts0))

    order = order[:max_draw]

    for idx in order:
        p0 = pts0[idx]
        p1 = pts1[idx]

        x0, y0 = int(round(p0[0])), int(round(p0[1]))
        x1, y1 = int(round(p1[0] + w0)), int(round(p1[1]))

        if inlier_mask is not None and len(inlier_mask) == len(pts0):
            is_inlier = bool(inlier_mask[idx])
        else:
            is_inlier = True

        color = (0, 255, 0) if is_inlier else (255, 80, 80)

        cv2.circle(canvas, (x0, y0), 3, color, -1)
        cv2.circle(canvas, (x1, y1), 3, color, -1)
        cv2.line(canvas, (x0, y0), (x1, y1), color, 1)

    return canvas


def homography_inliers(
    pts0: np.ndarray,
    pts1: np.ndarray,
    ransac_thresh: float,
) -> Tuple[int, float, bool, Optional[np.ndarray]]:
    if pts0 is None or pts1 is None or len(pts0) < 4 or len(pts1) < 4:
        return 0, 0.0, False, None

    src = np.asarray(pts0, dtype=np.float32).reshape(-1, 1, 2)
    dst = np.asarray(pts1, dtype=np.float32).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, ransac_thresh)
    if mask is None:
        return 0, 0.0, False, None

    mask_bool = mask.ravel().astype(bool)
    inliers = int(mask_bool.sum())
    ratio = float(inliers / max(1, len(mask_bool)))
    success = bool(H is not None and inliers >= 4)

    return inliers, ratio, success, mask_bool


def torch_tensor_from_rgb(rgb: np.ndarray, device: str):
    import torch

    arr = rgb.astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(device)


def torch_tensor_from_gray(gray: np.ndarray, device: str):
    import torch

    arr = gray.astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
    return tensor.to(device)


def strip_batch_tensor(x: Any):
    try:
        import torch

        if torch.is_tensor(x):
            while x.ndim >= 3 and x.shape[0] == 1:
                x = x[0]
            return x
    except Exception:
        pass
    return x


def _to_numpy_no_batch(x):
    """
    Convert torch/list/tuple/numpy LightGlue outputs into a clean numpy object.
    Handles common LightGlue version differences:
      tensor [1,K,2] -> [K,2]
      list length 1  -> first item
      tuple length 1 -> first item
    """
    import numpy as np

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
            return np.zeros((0, 2), dtype=np.int64)
        if len(x) == 1:
            return _to_numpy_no_batch(x[0])
        try:
            return np.asarray(x)
        except Exception:
            return np.array(list(x), dtype=object)

    return x


def _extract_lightglue_matches(matches01, feats0, feats1):
    """
    Return matched point arrays pts0, pts1 from different LightGlue output formats.

    Supported formats:
      matches01["matches"]  -> Kx2 tensor/array/list
      matches01["matches0"] -> N vector where value is matched index in image1 or -1
    """
    import numpy as np

    kpts0 = _to_numpy_no_batch(feats0["keypoints"]).astype(np.float32)
    kpts1 = _to_numpy_no_batch(feats1["keypoints"]).astype(np.float32)

    if "matches" in matches01:
        matches = _to_numpy_no_batch(matches01["matches"])

        if isinstance(matches, list):
            matches = np.asarray(matches)

        matches = np.asarray(matches)

        # Some versions may return shape [1,K,2] or object-like list.
        while matches.ndim >= 3 and matches.shape[0] == 1:
            matches = matches[0]

        if matches.size == 0:
            return (
                np.zeros((0, 2), dtype=np.float32),
                np.zeros((0, 2), dtype=np.float32),
            )

        matches = matches.astype(np.int64)

        if matches.ndim != 2 or matches.shape[1] < 2:
            return (
                np.zeros((0, 2), dtype=np.float32),
                np.zeros((0, 2), dtype=np.float32),
            )

        valid = (
            (matches[:, 0] >= 0)
            & (matches[:, 1] >= 0)
            & (matches[:, 0] < len(kpts0))
            & (matches[:, 1] < len(kpts1))
        )
        matches = matches[valid]

        if len(matches) == 0:
            return (
                np.zeros((0, 2), dtype=np.float32),
                np.zeros((0, 2), dtype=np.float32),
            )

        pts0 = kpts0[matches[:, 0]]
        pts1 = kpts1[matches[:, 1]]
        return pts0.astype(np.float32), pts1.astype(np.float32)

    if "matches0" in matches01:
        matches0 = _to_numpy_no_batch(matches01["matches0"])
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
            return (
                np.zeros((0, 2), dtype=np.float32),
                np.zeros((0, 2), dtype=np.float32),
            )

        return kpts0[idx0].astype(np.float32), kpts1[idx1].astype(np.float32)

    return (
        np.zeros((0, 2), dtype=np.float32),
        np.zeros((0, 2), dtype=np.float32),
    )


def run_lightglue_superpoint_pair(
    rgb0: np.ndarray,
    rgb1: np.ndarray,
    device: str,
    max_keypoints: int,
) -> Dict[str, Any]:
    import torch
    from lightglue import LightGlue, SuperPoint

    result: Dict[str, Any] = {
        "backend": "lightglue_superpoint",
        "status": "failed",
        "error": "",
        "matches": 0,
        "pts0": np.zeros((0, 2), dtype=np.float32),
        "pts1": np.zeros((0, 2), dtype=np.float32),
    }

    extractor = SuperPoint(max_num_keypoints=max_keypoints).eval().to(device)
    matcher = LightGlue(features="superpoint").eval().to(device)

    img0 = torch_tensor_from_rgb(rgb0, device)
    img1 = torch_tensor_from_rgb(rgb1, device)

    with torch.no_grad():
        try:
            feats0 = extractor.extract(img0, resize=None)
            feats1 = extractor.extract(img1, resize=None)
        except TypeError:
            feats0 = extractor.extract(img0)
            feats1 = extractor.extract(img1)

        matches01 = matcher({"image0": feats0, "image1": feats1})

    pts0, pts1 = _extract_lightglue_matches(matches01, feats0, feats1)

    result["status"] = "ok"
    result["matches"] = int(len(pts0))
    result["pts0"] = pts0.astype(np.float32)
    result["pts1"] = pts1.astype(np.float32)

    return result


def run_loftr_pair(
    rgb0: np.ndarray,
    rgb1: np.ndarray,
    device: str,
) -> Dict[str, Any]:
    import torch
    from kornia.feature import LoFTR

    result: Dict[str, Any] = {
        "backend": "loftr",
        "status": "failed",
        "error": "",
        "matches": 0,
        "pts0": np.zeros((0, 2), dtype=np.float32),
        "pts1": np.zeros((0, 2), dtype=np.float32),
    }

    gray0 = rgb_to_gray(rgb0)
    gray1 = rgb_to_gray(rgb1)

    t0 = torch_tensor_from_gray(gray0, device)
    t1 = torch_tensor_from_gray(gray1, device)

    matcher = LoFTR(pretrained="outdoor").eval().to(device)

    with torch.no_grad():
        out = matcher({"image0": t0, "image1": t1})

    pts0 = out["keypoints0"].detach().cpu().numpy().astype(np.float32)
    pts1 = out["keypoints1"].detach().cpu().numpy().astype(np.float32)

    result["status"] = "ok"
    result["matches"] = int(len(pts0))
    result["pts0"] = pts0
    result["pts1"] = pts1

    return result


def run_learned_pair(
    backend: str,
    rgb0: np.ndarray,
    rgb1: np.ndarray,
    device: str,
    max_keypoints: int,
) -> Dict[str, Any]:
    start = time.time()

    try:
        if backend == "lightglue_superpoint":
            result = run_lightglue_superpoint_pair(rgb0, rgb1, device, max_keypoints)
        elif backend == "loftr":
            result = run_loftr_pair(rgb0, rgb1, device)
        else:
            result = {
                "backend": backend,
                "status": "skipped",
                "error": "No learned backend selected or available.",
                "matches": 0,
                "pts0": np.zeros((0, 2), dtype=np.float32),
                "pts1": np.zeros((0, 2), dtype=np.float32),
            }
    except Exception as exc:
        result = {
            "backend": backend,
            "status": "failed",
            "error": repr(exc),
            "traceback": traceback.format_exc(limit=3),
            "matches": 0,
            "pts0": np.zeros((0, 2), dtype=np.float32),
            "pts1": np.zeros((0, 2), dtype=np.float32),
        }

    result["runtime_s"] = float(time.time() - start)
    return result


def choose_tokens(qdiag: pd.DataFrame, args: argparse.Namespace) -> List[str]:
    qdiag = qdiag.copy()
    qdiag["token_str"] = qdiag["token"].astype(str)

    if args.tokens.strip():
        return [t.strip() for t in args.tokens.split(",") if t.strip()]

    selected: List[str] = []
    classes = [c.strip() for c in args.classes.split(",") if c.strip()]

    for cls in classes:
        sub = qdiag[qdiag["local_decision_class"].astype(str) == cls].copy()
        if len(sub) == 0:
            continue

        if "local_minus_phog_error_m" in sub.columns:
            sub["abs_delta"] = pd.to_numeric(sub["local_minus_phog_error_m"], errors="coerce").abs()
            sub = sub.sort_values("abs_delta", ascending=False, kind="mergesort")

        selected.extend(sub["token_str"].head(args.tokens_per_class).tolist())

    seen = set()
    out = []
    for token in selected:
        if token not in seen:
            seen.add(token)
            out.append(token)

    return out


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


def select_candidate_rows(cand: pd.DataFrame, token: str) -> List[Tuple[str, Optional[pd.Series]]]:
    g = cand[cand["token"].astype(str) == str(token)].copy()
    if len(g) == 0:
        return [
            ("phog_top1", None),
            ("local_top1", None),
            ("oracle_best", None),
        ]

    g["candidate_pool_rank_num"] = numeric_col(g, "candidate_pool_rank")
    g["local_rank_num"] = numeric_col(g, "local_verifier_rank")
    g["eval_error_num"] = numeric_col(g, "eval_error_m")
    g["local_score_num"] = numeric_col(g, "local_score")

    phog_top1 = g.sort_values("candidate_pool_rank_num", kind="mergesort").iloc[0]

    if g["local_rank_num"].notna().any():
        local_top1 = g.sort_values("local_rank_num", kind="mergesort").iloc[0]
    else:
        local_top1 = g.sort_values("local_score_num", ascending=False, kind="mergesort").iloc[0]

    oracle_best = None
    valid = g.dropna(subset=["eval_error_num"])
    if len(valid):
        oracle_best = valid.sort_values("eval_error_num", kind="mergesort").iloc[0]

    return [
        ("phog_top1", phog_top1),
        ("local_top1", local_top1),
        ("oracle_best", oracle_best),
    ]


def candidate_image_path(row: Optional[pd.Series]) -> Optional[Path]:
    if row is None:
        return None

    candidate_cols = [
        "candidate_image_path",
        "sat_image_path",
        "satellite_image_path",
        "tile_image_path",
        "image_path",
        "tile_path",
    ]

    for col in candidate_cols:
        if col in row.index:
            p = resolve_path(row.get(col))
            if p is not None:
                return p

    return None


def candidate_label(pair_role: str, row: Optional[pd.Series]) -> str:
    if row is None:
        return f"{pair_role}: missing candidate"

    parts = [pair_role]

    tile_id = safe_str(row.get("tile_id"))
    if tile_id:
        parts.append(f"tile={tile_id}")

    if "candidate_pool_rank" in row.index:
        parts.append(f"PHOG-rank={fmt_float(row.get('candidate_pool_rank'), 0)}")

    if "local_verifier_rank" in row.index:
        parts.append(f"AKAZE-rank={fmt_float(row.get('local_verifier_rank'), 0)}")

    if "eval_error_m" in row.index:
        parts.append(f"eval-error={fmt_float(row.get('eval_error_m'), 1)}m")

    if "local_score" in row.index:
        parts.append(f"AKAZE-score={fmt_float(row.get('local_score'), 2)}")

    if "good_matches" in row.index and "ransac_inliers" in row.index:
        parts.append(
            f"AKAZE good/inliers={fmt_float(row.get('good_matches'), 0)}/{fmt_float(row.get('ransac_inliers'), 0)}"
        )

    return " | ".join(parts)


def make_pair_manifest(
    qdiag: pd.DataFrame,
    cand: pd.DataFrame,
    tokens: List[str],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for token in tokens:
        qsub = qdiag[qdiag["token_str"] == str(token)]
        if len(qsub) == 0:
            continue

        qrow = qsub.iloc[0]
        uav_path = resolve_path(qrow.get("uav_image_path"))

        for pair_role, crow in select_candidate_rows(cand, token):
            cpath = candidate_image_path(crow)

            rows.append(
                {
                    "token": token,
                    "failure_group": safe_str(qrow.get("failure_group")),
                    "local_decision_class": safe_str(qrow.get("local_decision_class")),
                    "pair_role": pair_role,
                    "uav_image_path": str(uav_path) if uav_path else "",
                    "candidate_image_path": str(cpath) if cpath else "",
                    "tile_id": safe_str(crow.get("tile_id")) if crow is not None else "",
                    "candidate_pool_rank": safe_float(crow.get("candidate_pool_rank")) if crow is not None else None,
                    "local_verifier_rank": safe_float(crow.get("local_verifier_rank")) if crow is not None else None,
                    "eval_error_m": safe_float(crow.get("eval_error_m")) if crow is not None else None,
                    "local_score": safe_float(crow.get("local_score")) if crow is not None else None,
                }
            )

    return pd.DataFrame(rows)


def add_image(ax, img: Optional[np.ndarray], title: str, cmap: Optional[str] = None) -> None:
    ax.axis("off")
    ax.set_title(title, fontsize=9)
    if img is None:
        ax.text(0.5, 0.5, "missing", ha="center", va="center", fontsize=11)
    else:
        ax.imshow(img, cmap=cmap)


def save_pair_panel(
    out_path: Path,
    token: str,
    failure_group: str,
    decision_class: str,
    pair_role: str,
    uav_rgb: Optional[np.ndarray],
    cand_rgb: Optional[np.ndarray],
    match_vis: Optional[np.ndarray],
    learned_result: Dict[str, Any],
    row: pd.Series,
    backend: str,
    device: str,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    fig.suptitle(
        f"S5A.2 learned verifier preflight | token {token} | {pair_role}\n"
        f"group={failure_group} | class={decision_class} | backend={backend} | device={device}",
        fontsize=12,
    )

    add_image(axes[0, 0], uav_rgb, "UAV input to learned matcher")
    add_image(axes[0, 1], cand_rgb, "Satellite candidate input")
    add_image(axes[0, 2], match_vis, "Learned matches / inliers")

    uav_gray = rgb_to_gray(uav_rgb) if uav_rgb is not None else None
    cand_gray = rgb_to_gray(cand_rgb) if cand_rgb is not None else None

    add_image(axes[1, 0], uav_gray, "UAV grayscale model view", cmap="gray")
    add_image(axes[1, 1], cand_gray, "Satellite grayscale model view", cmap="gray")

    axes[1, 2].axis("off")

    status = safe_str(learned_result.get("status"))
    error = safe_str(learned_result.get("error"))
    if len(error) > 220:
        error = error[:220] + "..."

    text = (
        f"Candidate label:\n"
        f"{pair_role}\n"
        f"tile={safe_str(row.get('tile_id'))}\n"
        f"PHOG-rank={fmt_float(row.get('candidate_pool_rank'), 0)}\n"
        f"AKAZE-rank={fmt_float(row.get('local_verifier_rank'), 0)}\n"
        f"eval-error={fmt_float(row.get('eval_error_m'), 1)} m\n"
        f"AKAZE-score={fmt_float(row.get('local_score'), 2)}\n\n"
        f"Learned matcher result:\n"
        f"status={status}\n"
        f"matches={learned_result.get('matches', 0)}\n"
        f"RANSAC inliers={learned_result.get('ransac_inliers', 0)}\n"
        f"inlier ratio={learned_result.get('inlier_ratio', 0.0):.3f}\n"
        f"homography success={learned_result.get('homography_success', False)}\n"
        f"runtime={learned_result.get('runtime_s', 0.0):.3f}s\n"
        f"error={error}"
    )

    axes[1, 2].text(0.02, 0.98, text, ha="left", va="top", fontsize=9)

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def process_pair(
    row: pd.Series,
    pair_idx: int,
    dirs: Dict[str, Path],
    backend: str,
    env: Dict[str, Any],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    token = safe_str(row.get("token"))
    pair_role = safe_str(row.get("pair_role"))
    failure_group = safe_str(row.get("failure_group"))
    decision_class = safe_str(row.get("local_decision_class"))

    uav_path = resolve_path(row.get("uav_image_path"))
    candidate_path = resolve_path(row.get("candidate_image_path"))

    result_row: Dict[str, Any] = row.to_dict()
    result_row.update(
        {
            "selected_backend": backend,
            "selected_device": env.get("selected_device", "cpu"),
            "learned_status": "not_started",
            "learned_error": "",
            "learned_matches": 0,
            "learned_ransac_inliers": 0,
            "learned_inlier_ratio": 0.0,
            "learned_homography_success": False,
            "learned_runtime_s": 0.0,
            "panel_path": "",
        }
    )

    if uav_path is None or candidate_path is None:
        result_row["learned_status"] = "missing_image_path"
        return result_row

    uav_rgb_raw = read_rgb(uav_path)
    cand_rgb_raw = read_rgb(candidate_path)

    if uav_rgb_raw is None or cand_rgb_raw is None:
        result_row["learned_status"] = "image_read_failed"
        return result_row

    # Learned matchers receive the resized RGB inputs. These are also what we display.
    if backend == "loftr":
        uav_rgb = resize_longest_multiple(uav_rgb_raw, args.resize_long, multiple=8)
        cand_rgb = resize_longest_multiple(cand_rgb_raw, args.resize_long, multiple=8)
    else:
        uav_rgb = resize_longest(uav_rgb_raw, args.resize_long)
        cand_rgb = resize_longest(cand_rgb_raw, args.resize_long)

    learned_result: Dict[str, Any]

    if args.skip_model_run:
        learned_result = {
            "backend": backend,
            "status": "skipped",
            "error": "--skip-model-run was set",
            "matches": 0,
            "pts0": np.zeros((0, 2), dtype=np.float32),
            "pts1": np.zeros((0, 2), dtype=np.float32),
            "runtime_s": 0.0,
        }
    elif backend == "none":
        learned_result = {
            "backend": "none",
            "status": "skipped",
            "error": "No learned backend available. Install/cache LightGlue or Kornia LoFTR before learned run.",
            "matches": 0,
            "pts0": np.zeros((0, 2), dtype=np.float32),
            "pts1": np.zeros((0, 2), dtype=np.float32),
            "runtime_s": 0.0,
        }
    else:
        learned_result = run_learned_pair(
            backend=backend,
            rgb0=uav_rgb,
            rgb1=cand_rgb,
            device=safe_str(env.get("selected_device", "cpu")),
            max_keypoints=args.max_keypoints,
        )

    pts0 = learned_result.get("pts0", np.zeros((0, 2), dtype=np.float32))
    pts1 = learned_result.get("pts1", np.zeros((0, 2), dtype=np.float32))

    inliers, inlier_ratio, h_success, inlier_mask = homography_inliers(
        pts0,
        pts1,
        args.ransac_thresh,
    )

    learned_result["ransac_inliers"] = int(inliers)
    learned_result["inlier_ratio"] = float(inlier_ratio)
    learned_result["homography_success"] = bool(h_success)

    match_vis = draw_points_matches(
        uav_rgb,
        cand_rgb,
        pts0,
        pts1,
        inlier_mask,
        args.max_draw_matches,
    )

    safe_cls = decision_class.replace("/", "_").replace(" ", "_")
    safe_group = failure_group.replace("/", "_").replace(" ", "_")
    safe_role = pair_role.replace("/", "_").replace(" ", "_")

    panel_path = (
        dirs["panels"]
        / safe_group
        / safe_cls
        / f"s5a2_token{int(float(token)):04d}_{safe_role}_{backend}.png"
    )

    save_pair_panel(
        out_path=panel_path,
        token=token,
        failure_group=failure_group,
        decision_class=decision_class,
        pair_role=pair_role,
        uav_rgb=uav_rgb,
        cand_rgb=cand_rgb,
        match_vis=match_vis,
        learned_result=learned_result,
        row=row,
        backend=backend,
        device=safe_str(env.get("selected_device", "cpu")),
    )

    result_row.update(
        {
            "learned_status": safe_str(learned_result.get("status")),
            "learned_error": safe_str(learned_result.get("error")),
            "learned_matches": int(learned_result.get("matches", 0)),
            "learned_ransac_inliers": int(inliers),
            "learned_inlier_ratio": float(inlier_ratio),
            "learned_homography_success": bool(h_success),
            "learned_runtime_s": float(learned_result.get("runtime_s", 0.0)),
            "panel_path": str(panel_path),
        }
    )

    return result_row


def main() -> None:
    args = parse_args()
    dirs = ensure_dirs(args.out_base)

    env = detect_environment(args.device)
    selected_backend = select_backend(args, env)

    qdiag = read_csv(args.query_diagnostics, "S5A.1B query diagnostics")
    cand = read_csv(args.candidate_scores, "S5A.1 candidate scores")
    qsum = read_csv(args.query_summary, "S5A.1 query summary")

    qdiag = merge_uav_paths(qdiag, qsum)
    tokens = choose_tokens(qdiag, args)

    pair_manifest = make_pair_manifest(qdiag, cand, tokens)

    pair_manifest_path = dirs["metadata"] / "s5a2_smoke_pair_manifest.csv"
    pair_results_path = dirs["metadata"] / "s5a2_learned_preflight_pair_results.csv"
    summary_path = dirs["reports"] / "s5a2_learned_verifier_preflight_summary.json"
    env_path = dirs["reports"] / "s5a2_environment_check.json"

    pair_manifest.to_csv(pair_manifest_path, index=False)

    with open(env_path, "w", encoding="utf-8") as f:
        json.dump(env, f, indent=2)

    process_df = pair_manifest.head(args.max_pairs).copy()

    result_rows: List[Dict[str, Any]] = []
    for idx, row in process_df.iterrows():
        result_row = process_pair(
            row=row,
            pair_idx=int(idx),
            dirs=dirs,
            backend=selected_backend,
            env=env,
            args=args,
        )
        result_rows.append(result_row)

    results_df = pd.DataFrame(result_rows)
    results_df.to_csv(pair_results_path, index=False)

    ok_runs = results_df[results_df["learned_status"].astype(str) == "ok"] if len(results_df) else results_df

    summary: Dict[str, Any] = {
        "stage": "S5A.2_learned_verifier_preflight",
        "locked_rule": "reference/error columns are evaluation-only; used only after ranking for smoke-test labels/oracle panels",
        "requested_backend": args.backend,
        "selected_backend": selected_backend,
        "selected_device": env.get("selected_device", "cpu"),
        "environment": env,
        "tokens_selected": tokens,
        "num_tokens_selected": int(len(tokens)),
        "num_pairs_in_manifest": int(len(pair_manifest)),
        "num_pairs_processed": int(len(results_df)),
        "num_ok_model_runs": int(len(ok_runs)),
        "mean_matches_ok_runs": float(ok_runs["learned_matches"].mean()) if len(ok_runs) else None,
        "median_matches_ok_runs": float(ok_runs["learned_matches"].median()) if len(ok_runs) else None,
        "mean_ransac_inliers_ok_runs": float(ok_runs["learned_ransac_inliers"].mean()) if len(ok_runs) else None,
        "median_ransac_inliers_ok_runs": float(ok_runs["learned_ransac_inliers"].median()) if len(ok_runs) else None,
        "pair_manifest_csv": str(pair_manifest_path),
        "pair_results_csv": str(pair_results_path),
        "environment_json": str(env_path),
        "panel_dir": str(dirs["panels"]),
    }

    if len(results_df):
        summary["status_counts"] = results_df["learned_status"].value_counts(dropna=False).to_dict()
    else:
        summary["status_counts"] = {}

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("S5A.2 learned verifier preflight complete")
    print("----------------------------------------")
    print(f"Requested backend:       {args.backend}")
    print(f"Selected backend:        {selected_backend}")
    print(f"Selected device:         {env.get('selected_device', 'cpu')}")
    print(f"Torch available:         {env.get('torch_available')}")
    print(f"LightGlue available:     {env.get('lightglue_available')}")
    print(f"Kornia available:        {env.get('kornia_available')}")
    print(f"Tokens selected:         {len(tokens)} -> {tokens}")
    print(f"Pairs in manifest:       {len(pair_manifest)}")
    print(f"Pairs processed:         {len(results_df)}")
    print(f"OK model runs:           {summary['num_ok_model_runs']}")
    print(f"Status counts:           {summary['status_counts']}")
    print(f"Pair manifest CSV:       {pair_manifest_path}")
    print(f"Pair results CSV:        {pair_results_path}")
    print(f"Environment JSON:        {env_path}")
    print(f"Summary JSON:            {summary_path}")
    print(f"Working panels dir:      {dirs['panels']}")
    print()
#    print("Locked rule: reference/error columns were used only after ranking for diagnostic labels/oracle display.")


if __name__ == "__main__":
    main()
