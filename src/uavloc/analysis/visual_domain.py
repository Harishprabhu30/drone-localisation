from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import pandas as pd


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_image_bgr(path: str | Path) -> Optional[np.ndarray]:
    path = Path(path)
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    return img


def resize_for_analysis(img_bgr: np.ndarray, max_dim: int = 900) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    scale = max_dim / max(h, w)

    if scale >= 1.0:
        return img_bgr

    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    return cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)


def shannon_entropy(gray: np.ndarray) -> float:
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    prob = hist / max(hist.sum(), 1.0)
    prob = prob[prob > 0]
    return float(-(prob * np.log2(prob)).sum())


def colorfulness_metric(img_bgr: np.ndarray) -> float:
    b, g, r = cv2.split(img_bgr.astype(np.float32))

    rg = np.abs(r - g)
    yb = np.abs(0.5 * (r + g) - b)

    std_rg = np.std(rg)
    std_yb = np.std(yb)
    mean_rg = np.mean(rg)
    mean_yb = np.mean(yb)

    return float(math.sqrt(std_rg**2 + std_yb**2) + 0.3 * math.sqrt(mean_rg**2 + mean_yb**2))


def gradient_orientation_entropy(gray: np.ndarray) -> float:
    gray_f = gray.astype(np.float32) / 255.0

    gx = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3)

    mag = np.sqrt(gx * gx + gy * gy)
    ang = np.arctan2(gy, gx)

    valid = mag > np.percentile(mag, 70)
    if valid.sum() < 20:
        return 0.0

    hist, _ = np.histogram(ang[valid], bins=18, range=(-np.pi, np.pi))
    prob = hist.astype(np.float64) / max(hist.sum(), 1)
    prob = prob[prob > 0]

    return float(-(prob * np.log2(prob)).sum())


def hough_line_count(edges: np.ndarray) -> int:
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=60,
        minLineLength=40,
        maxLineGap=8,
    )

    if lines is None:
        return 0

    return int(len(lines))


def compute_orb_count(gray: np.ndarray) -> int:
    orb = cv2.ORB_create(nfeatures=2500)
    kps = orb.detect(gray, None)
    return int(len(kps))


def compute_akaze_count(gray: np.ndarray) -> int:
    try:
        akaze = cv2.AKAZE_create()
        kps = akaze.detect(gray, None)
        return int(len(kps))
    except Exception:
        return -1


def compute_image_stats(image_path: str | Path, max_dim: int = 900) -> Dict[str, Any]:
    image_path = Path(image_path)

    img_bgr_raw = read_image_bgr(image_path)

    base: Dict[str, Any] = {
        "image_path": str(image_path),
        "filename": image_path.name,
        "file_size_bytes": image_path.stat().st_size if image_path.exists() else None,
        "read_ok": img_bgr_raw is not None,
    }

    if img_bgr_raw is None:
        base["error"] = "cv2_read_failed"
        return base

    raw_h, raw_w = img_bgr_raw.shape[:2]
    img_bgr = resize_for_analysis(img_bgr_raw, max_dim=max_dim)
    h, w = img_bgr.shape[:2]

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Perceptual luma from RGB.
    r = img_rgb[:, :, 0].astype(np.float32)
    g = img_rgb[:, :, 1].astype(np.float32)
    b = img_rgb[:, :, 2].astype(np.float32)
    luma = 0.299 * r + 0.587 * g + 0.114 * b
    luma_u8 = np.clip(luma, 0, 255).astype(np.uint8)

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    lap = cv2.Laplacian(gray, cv2.CV_64F)
    lap_var = float(lap.var())

    edges = cv2.Canny(gray, 80, 160)
    edge_density = float((edges > 0).mean())

    gray_f = gray.astype(np.float32) / 255.0
    gx = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx * gx + gy * gy)

    stats: Dict[str, Any] = {
        **base,
        "raw_width": int(raw_w),
        "raw_height": int(raw_h),
        "analysis_width": int(w),
        "analysis_height": int(h),
        "aspect_ratio": float(raw_w / raw_h),
        "rgb_mean_r": float(r.mean()),
        "rgb_mean_g": float(g.mean()),
        "rgb_mean_b": float(b.mean()),
        "rgb_std_r": float(r.std()),
        "rgb_std_g": float(g.std()),
        "rgb_std_b": float(b.std()),
        "luma_mean": float(luma.mean()),
        "luma_std": float(luma.std()),
        "luma_p05": float(np.percentile(luma, 5)),
        "luma_p50": float(np.percentile(luma, 50)),
        "luma_p95": float(np.percentile(luma, 95)),
        "dark_pixel_ratio": float((luma < 30).mean()),
        "bright_pixel_ratio": float((luma > 225).mean()),
        "saturation_mean": float(sat.mean()),
        "saturation_std": float(sat.std()),
        "value_mean": float(val.mean()),
        "value_std": float(val.std()),
        "colorfulness": colorfulness_metric(img_bgr),
        "laplacian_variance": lap_var,
        "edge_density": edge_density,
        "hough_line_count": hough_line_count(edges),
        "entropy_gray": shannon_entropy(gray),
        "entropy_luma": shannon_entropy(luma_u8),
        "gradient_mag_mean": float(grad_mag.mean()),
        "gradient_mag_std": float(grad_mag.std()),
        "gradient_mag_p95": float(np.percentile(grad_mag, 95)),
        "gradient_orientation_entropy": gradient_orientation_entropy(gray),
        "orb_keypoint_count": compute_orb_count(gray),
        "akaze_keypoint_count": compute_akaze_count(gray),
    }

    # Simple vegetation / greenness diagnostic, not a classifier.
    rgb_sum = r + g + b + 1e-6
    excess_green = 2.0 * g - r - b
    stats["green_ratio_mean"] = float((g / rgb_sum).mean())
    stats["excess_green_mean"] = float(excess_green.mean())
    stats["excess_green_std"] = float(excess_green.std())

    return stats


def add_relative_quality_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    numeric_cols = [
        "luma_mean",
        "luma_std",
        "laplacian_variance",
        "edge_density",
        "entropy_gray",
        "orb_keypoint_count",
        "akaze_keypoint_count",
        "hough_line_count",
        "gradient_mag_mean",
        "colorfulness",
        "green_ratio_mean",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[f"{col}_percentile"] = df[col].rank(pct=True)

    if "laplacian_variance_percentile" in df.columns:
        df["flag_blurry_relative"] = df["laplacian_variance_percentile"] <= 0.15

    if "edge_density_percentile" in df.columns:
        df["flag_low_edge_relative"] = df["edge_density_percentile"] <= 0.15
        df["flag_high_edge_relative"] = df["edge_density_percentile"] >= 0.85

    if "orb_keypoint_count_percentile" in df.columns:
        df["flag_low_orb_relative"] = df["orb_keypoint_count_percentile"] <= 0.15
        df["flag_high_orb_relative"] = df["orb_keypoint_count_percentile"] >= 0.85

    if "green_ratio_mean_percentile" in df.columns:
        df["flag_green_dominant_relative"] = df["green_ratio_mean_percentile"] >= 0.80

    return df


def summarize_stats(df: pd.DataFrame) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "rows": int(len(df)),
        "read_ok": int(df["read_ok"].sum()) if "read_ok" in df.columns else None,
        "columns": list(df.columns),
        "metrics": {},
    }

    metric_cols = [
        "luma_mean",
        "luma_std",
        "laplacian_variance",
        "edge_density",
        "entropy_gray",
        "orb_keypoint_count",
        "akaze_keypoint_count",
        "hough_line_count",
        "gradient_mag_mean",
        "colorfulness",
        "green_ratio_mean",
        "excess_green_mean",
    ]

    for col in metric_cols:
        if col not in df.columns:
            continue

        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if s.empty:
            continue

        summary["metrics"][col] = {
            "min": float(s.min()),
            "p05": float(s.quantile(0.05)),
            "p25": float(s.quantile(0.25)),
            "median": float(s.median()),
            "mean": float(s.mean()),
            "p75": float(s.quantile(0.75)),
            "p95": float(s.quantile(0.95)),
            "max": float(s.max()),
        }

    flag_cols = [c for c in df.columns if c.startswith("flag_")]
    summary["flags"] = {}
    for col in flag_cols:
        summary["flags"][col] = int(df[col].fillna(False).sum())

    return summary


def save_json(data: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)