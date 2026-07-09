#!/usr/bin/env python3
"""
S5A.1C — Visualize local-verifier internals

Purpose
-------
Create visual panels showing how the OpenCV local verifier processes UAV and
satellite candidates inside the PHOG top-K pool.

For selected representative tokens, the script saves panels containing:
1. UAV original image
2. UAV preprocessed image
3. UAV AKAZE/SIFT/ORB keypoints
4. PHOG top-1 candidate and matches
5. Local-verifier top-1 candidate and matches
6. Oracle-best candidate inside PHOG top-K and matches

Locked rule
-----------
Reference/error columns are used only to choose diagnostic labels and oracle panels
after ranking. They are not used for retrieval/ranking/scoring.

Code used:
export PYTHONPATH=$PWD/src

python scripts/satloc/s5a/s5a_1c_visualize_local_verifier_workings.py
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="S5A.1C visualize local verifier workings")
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
        "--out-dir",
        type=Path,
        default=Path("outputs/satloc/figures/s5a_learned_local_verifier/s5a1c_working_panels"),
    )
    parser.add_argument(
        "--backend",
        choices=["akaze", "orb", "sift"],
        default="akaze",
        help="Detector used for visualization. Should match S5A.1 if possible.",
    )
    parser.add_argument(
        "--preprocess",
        choices=["gray", "clahe_gray", "luma", "clahe_luma"],
        default="clahe_gray",
    )
    parser.add_argument("--resize-long", type=int, default=768)
    parser.add_argument("--ratio", type=float, default=0.75)
    parser.add_argument("--ransac-thresh", type=float, default=5.0)
    parser.add_argument("--max-match-draw", type=int, default=80)
    parser.add_argument(
        "--classes",
        type=str,
        default="local_destroyed_phog,correct_available_but_local_missed,local_rescue,candidate_pool_not_good_enough",
        help="Comma-separated local_decision_class values to sample.",
    )
    parser.add_argument(
        "--tokens-per-class",
        type=int,
        default=2,
        help="How many representative tokens to visualize from each class.",
    )
    parser.add_argument(
        "--tokens",
        type=str,
        default="",
        help="Optional comma-separated explicit tokens. If set, overrides class sampling.",
    )
    return parser.parse_args()


def read_csv(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")
    return pd.read_csv(path)


def safe_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value).strip()


def safe_float(value: Any) -> Optional[float]:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


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


def read_rgb(path: Path) -> Optional[np.ndarray]:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None

    if img.ndim == 2:
        rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    else:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return rgb


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


def to_gray_or_luma(rgb: np.ndarray, mode: str) -> np.ndarray:
    if mode in {"luma", "clahe_luma"}:
        r = rgb[:, :, 0].astype(np.float32)
        g = rgb[:, :, 1].astype(np.float32)
        b = rgb[:, :, 2].astype(np.float32)
        gray = 0.299 * r + 0.587 * g + 0.114 * b
        return np.clip(gray, 0, 255).astype(np.uint8)

    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def preprocess_rgb(rgb: np.ndarray, mode: str, resize_long: int) -> np.ndarray:
    rgb_resized = resize_longest(rgb, resize_long)
    gray = to_gray_or_luma(rgb_resized, mode)

    if mode in {"clahe_gray", "clahe_luma"}:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

    return gray


def create_detector_and_norm(backend: str):
    backend = backend.lower()
    if backend == "akaze":
        return cv2.AKAZE_create(), cv2.NORM_HAMMING
    if backend == "orb":
        return cv2.ORB_create(nfeatures=2500), cv2.NORM_HAMMING
    if backend == "sift":
        if not hasattr(cv2, "SIFT_create"):
            raise RuntimeError("This OpenCV build does not provide SIFT.")
        return cv2.SIFT_create(nfeatures=2500), cv2.NORM_L2
    raise ValueError(f"Unknown backend: {backend}")


def detect_keypoints(gray: np.ndarray, backend: str):
    detector, _norm = create_detector_and_norm(backend)
    kp, desc = detector.detectAndCompute(gray, None)
    if kp is None:
        kp = []
    return kp, desc


def keypoint_overlay(rgb: np.ndarray, gray: np.ndarray, backend: str) -> Tuple[np.ndarray, int]:
    kp, _desc = detect_keypoints(gray, backend)
    base = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    overlay = cv2.drawKeypoints(
        base,
        kp,
        None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )
    overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB) if overlay.shape[2] == 3 else overlay
    return overlay, len(kp)


def match_pair(
    q_gray: np.ndarray,
    c_gray: np.ndarray,
    backend: str,
    ratio: float,
    ransac_thresh: float,
    max_match_draw: int,
) -> Dict[str, Any]:
    detector, norm = create_detector_and_norm(backend)

    q_kp, q_desc = detector.detectAndCompute(q_gray, None)
    c_kp, c_desc = detector.detectAndCompute(c_gray, None)

    q_kp = [] if q_kp is None else q_kp
    c_kp = [] if c_kp is None else c_kp

    result: Dict[str, Any] = {
        "q_keypoints": len(q_kp),
        "c_keypoints": len(c_kp),
        "raw_matches": 0,
        "good_matches": 0,
        "ransac_inliers": 0,
        "inlier_ratio": 0.0,
        "homography_success": False,
        "matches_vis": None,
    }

    if q_desc is None or c_desc is None or len(q_kp) < 4 or len(c_kp) < 4:
        return result

    matcher = cv2.BFMatcher(norm, crossCheck=False)
    knn = matcher.knnMatch(q_desc, c_desc, k=2)
    result["raw_matches"] = len(knn)

    good = []
    for m_n in knn:
        if len(m_n) != 2:
            continue
        m, n = m_n
        if m.distance < ratio * n.distance:
            good.append(m)

    result["good_matches"] = len(good)

    inlier_mask = None
    if len(good) >= 4:
        src = np.float32([q_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([c_kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, ransac_thresh)
        if mask is not None:
            inlier_mask = mask.ravel().astype(bool)
            inliers = int(inlier_mask.sum())
            result["ransac_inliers"] = inliers
            result["inlier_ratio"] = float(inliers / max(1, len(good)))
            result["homography_success"] = bool(H is not None and inliers >= 4)

    # Draw only inlier matches if available; otherwise draw good matches.
    draw_matches = good
    draw_mask = None
    if inlier_mask is not None and len(inlier_mask) == len(good) and int(inlier_mask.sum()) > 0:
        draw_matches = [m for m, keep in zip(good, inlier_mask) if keep]
        draw_mask = None

    draw_matches = sorted(draw_matches, key=lambda m: m.distance)[:max_match_draw]

    q_vis = cv2.cvtColor(q_gray, cv2.COLOR_GRAY2BGR)
    c_vis = cv2.cvtColor(c_gray, cv2.COLOR_GRAY2BGR)
    matches_vis_bgr = cv2.drawMatches(
        q_vis,
        q_kp,
        c_vis,
        c_kp,
        draw_matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        matchesMask=draw_mask,
    )
    result["matches_vis"] = cv2.cvtColor(matches_vis_bgr, cv2.COLOR_BGR2RGB)
    return result


def numeric_col(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series([np.nan] * len(df), index=df.index)
    return pd.to_numeric(df[col], errors="coerce")


def choose_tokens(qdiag: pd.DataFrame, args: argparse.Namespace) -> List[str]:
    if args.tokens.strip():
        return [t.strip() for t in args.tokens.split(",") if t.strip()]

    classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    selected: List[str] = []

    work = qdiag.copy()
    work["token_str"] = work["token"].astype(str)

    for cls in classes:
        subset = work[work["local_decision_class"].astype(str) == cls].copy()
        if len(subset) == 0:
            continue

        # Prefer examples where local and PHOG differ strongly.
        if "local_minus_phog_error_m" in subset.columns:
            subset["abs_delta"] = pd.to_numeric(subset["local_minus_phog_error_m"], errors="coerce").abs()
            subset = subset.sort_values("abs_delta", ascending=False, kind="mergesort")

        selected.extend(subset["token_str"].head(args.tokens_per_class).tolist())

    # Preserve order and remove duplicates.
    seen = set()
    unique = []
    for t in selected:
        if t not in seen:
            seen.add(t)
            unique.append(t)
    return unique


def select_candidate_rows(cand: pd.DataFrame, token: str) -> Dict[str, Optional[pd.Series]]:
    g = cand[cand["token"].astype(str) == str(token)].copy()
    if len(g) == 0:
        return {"phog_top1": None, "local_top1": None, "oracle_best": None}

    g["candidate_pool_rank_num"] = numeric_col(g, "candidate_pool_rank")
    g["local_rank_num"] = numeric_col(g, "local_verifier_rank")
    g["eval_error_num"] = numeric_col(g, "eval_error_m")
    g["local_score_num"] = numeric_col(g, "local_score")

    phog_top1 = g.sort_values("candidate_pool_rank_num", kind="mergesort").iloc[0]

    if g["local_rank_num"].notna().any():
        local_top1 = g.sort_values("local_rank_num", kind="mergesort").iloc[0]
    else:
        local_top1 = g.sort_values("local_score_num", ascending=False, kind="mergesort").iloc[0]

    valid_error = g.dropna(subset=["eval_error_num"])
    oracle_best = None
    if len(valid_error) > 0:
        oracle_best = valid_error.sort_values("eval_error_num", kind="mergesort").iloc[0]

    return {
        "phog_top1": phog_top1,
        "local_top1": local_top1,
        "oracle_best": oracle_best,
    }


def row_label(name: str, row: pd.Series) -> str:
    if row is None:
        return f"{name}: missing"

    tile_id = safe_str(row.get("tile_id"))
    pool_rank = safe_float(row.get("candidate_pool_rank"))
    local_rank = safe_float(row.get("local_verifier_rank"))
    err = safe_float(row.get("eval_error_m"))
    score = safe_float(row.get("local_score"))
    good = safe_float(row.get("good_matches"))
    inl = safe_float(row.get("ransac_inliers"))

    parts = [name]
    if tile_id:
        parts.append(f"tile={tile_id}")
    if pool_rank is not None:
        parts.append(f"PHOG-rank={int(pool_rank)}")
    if local_rank is not None:
        parts.append(f"local-rank={int(local_rank)}")
    if err is not None:
        parts.append(f"err={err:.1f}m")
    if score is not None:
        parts.append(f"score={score:.2f}")
    if good is not None and inl is not None:
        parts.append(f"good/inliers={int(good)}/{int(inl)}")
    return " | ".join(parts)


def add_image(ax, img: Optional[np.ndarray], title: str, cmap: Optional[str] = None) -> None:
    ax.axis("off")
    ax.set_title(title, fontsize=9)
    if img is None:
        ax.text(0.5, 0.5, "missing", ha="center", va="center")
        return
    ax.imshow(img, cmap=cmap)


def candidate_path(row: Optional[pd.Series]) -> Optional[Path]:
    if row is None:
        return None
    return resolve_path(row.get("candidate_image_path"))


def build_panel_for_token(
    token: str,
    qrow: pd.Series,
    cand: pd.DataFrame,
    args: argparse.Namespace,
) -> Optional[Path]:
    uav_path = resolve_path(qrow.get("uav_image_path"))
    if uav_path is None:
        print(f"[WARN] token {token}: missing UAV path")
        return None

    uav_rgb = read_rgb(uav_path)
    if uav_rgb is None:
        print(f"[WARN] token {token}: cannot read UAV image")
        return None

    uav_gray = preprocess_rgb(uav_rgb, args.preprocess, args.resize_long)
    uav_kp_img, uav_kp_count = keypoint_overlay(uav_rgb, uav_gray, args.backend)

    chosen = select_candidate_rows(cand, token)

    panel_items = [
        ("PHOG top1", chosen["phog_top1"]),
        ("Local top1", chosen["local_top1"]),
        ("Oracle best in PHOG topK", chosen["oracle_best"]),
    ]

    fig = plt.figure(figsize=(18, 12))
    grid = fig.add_gridspec(4, 4, height_ratios=[1.0, 1.15, 1.15, 1.15])

    cls = safe_str(qrow.get("local_decision_class"))
    group = safe_str(qrow.get("failure_group"))
    phog_err = safe_float(qrow.get("phog_error_m"))
    local_err = safe_float(qrow.get("local_error_m"))
    oracle_err = safe_float(qrow.get("oracle_error_m"))

    title = (
        f"S5A.1C local-verifier workings | token {token} | group={group} | class={cls}\n"
        f"PHOG err={phog_err:.1f}m | local err={local_err:.1f}m | oracle topK err={oracle_err:.1f}m | "
        f"backend={args.backend}, preprocess={args.preprocess}"
    )
    fig.suptitle(title, fontsize=13)

    ax00 = fig.add_subplot(grid[0, 0])
    add_image(ax00, resize_longest(uav_rgb, args.resize_long), "UAV original")

    ax01 = fig.add_subplot(grid[0, 1])
    add_image(ax01, uav_gray, "UAV preprocessed", cmap="gray")

    ax02 = fig.add_subplot(grid[0, 2])
    add_image(ax02, uav_kp_img, f"UAV {args.backend.upper()} keypoints: {uav_kp_count}")

    ax03 = fig.add_subplot(grid[0, 3])
    ax03.axis("off")
    explanation = (
        "How to read:\n"
        "PHOG top1 = old structural rank-1\n"
        "Local top1 = AKAZE/OpenCV rerank winner\n"
        "Oracle = nearest candidate in PHOG top-K after evaluation\n\n"
        "If Local top1 has strong matches but high error,\n"
        "the local verifier is being fooled by repeated texture."
    )
    ax03.text(0.02, 0.98, explanation, ha="left", va="top", fontsize=10)

    for row_idx, (name, crow) in enumerate(panel_items, start=1):
        cpath = candidate_path(crow)
        c_rgb = read_rgb(cpath) if cpath else None

        c_gray = None
        c_kp_img = None
        c_kp_count = 0
        match_vis = None
        match_stats = ""

        if c_rgb is not None:
            c_gray = preprocess_rgb(c_rgb, args.preprocess, args.resize_long)
            c_kp_img, c_kp_count = keypoint_overlay(c_rgb, c_gray, args.backend)
            match_result = match_pair(
                uav_gray,
                c_gray,
                args.backend,
                args.ratio,
                args.ransac_thresh,
                args.max_match_draw,
            )
            match_vis = match_result["matches_vis"]
            match_stats = (
                f"computed: qkp={match_result['q_keypoints']}, ckp={match_result['c_keypoints']}, "
                f"raw={match_result['raw_matches']}, good={match_result['good_matches']}, "
                f"inliers={match_result['ransac_inliers']}, ratio={match_result['inlier_ratio']:.2f}"
            )

        ax_a = fig.add_subplot(grid[row_idx, 0])
        add_image(ax_a, resize_longest(c_rgb, args.resize_long) if c_rgb is not None else None, f"{name} original")

        ax_b = fig.add_subplot(grid[row_idx, 1])
        add_image(ax_b, c_gray, f"{name} preprocessed", cmap="gray")

        ax_c = fig.add_subplot(grid[row_idx, 2])
        add_image(ax_c, c_kp_img, f"{name} {args.backend.upper()} keypoints: {c_kp_count}")

        ax_d = fig.add_subplot(grid[row_idx, 3])
        label = row_label(name, crow)
        if match_stats:
            label = label + "\n" + match_stats
        add_image(ax_d, match_vis, label)

    safe_group = group.replace("/", "_").replace(" ", "_")
    safe_cls = cls.replace("/", "_").replace(" ", "_")
    out_path = args.out_dir / safe_group / safe_cls / f"s5a1c_token{int(float(token)):04d}_{safe_cls}_workings.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=160)
    plt.close(fig)

    return out_path


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    qdiag = read_csv(args.query_diagnostics, "S5A.1B query diagnostics")
    cand = read_csv(args.candidate_scores, "S5A.1 candidate scores")
    qsum = read_csv(args.query_summary, "S5A.1 query summary")

    qdiag["token_str"] = qdiag["token"].astype(str)
    qsum["token_str"] = qsum["token"].astype(str)

    # Add UAV image path from query summary if missing in diagnostics.
    if "uav_image_path" not in qdiag.columns:
        qdiag = qdiag.merge(
            qsum[["token_str", "uav_image_path"]],
            on="token_str",
            how="left",
        )
    else:
        qdiag = qdiag.merge(
            qsum[["token_str", "uav_image_path"]],
            on="token_str",
            how="left",
            suffixes=("", "_from_summary"),
        )
        qdiag["uav_image_path"] = qdiag["uav_image_path"].fillna(qdiag.get("uav_image_path_from_summary", ""))

    tokens = choose_tokens(qdiag, args)

    manifest_rows: List[Dict[str, Any]] = []
    saved_paths: List[Path] = []

    for token in tokens:
        subset = qdiag[qdiag["token_str"] == str(token)]
        if len(subset) == 0:
            print(f"[WARN] token {token}: not found in query diagnostics")
            continue

        qrow = subset.iloc[0]
        out_path = build_panel_for_token(str(token), qrow, cand, args)

        if out_path is not None:
            saved_paths.append(out_path)
            manifest_rows.append(
                {
                    "token": token,
                    "failure_group": safe_str(qrow.get("failure_group")),
                    "local_decision_class": safe_str(qrow.get("local_decision_class")),
                    "panel_path": str(out_path),
                    "phog_error_m": safe_float(qrow.get("phog_error_m")),
                    "local_error_m": safe_float(qrow.get("local_error_m")),
                    "oracle_error_m": safe_float(qrow.get("oracle_error_m")),
                }
            )

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_path = args.out_dir / "s5a1c_working_panel_manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)

    print("S5A.1C local-verifier working panels complete")
    print("------------------------------------------------")
    print(f"Backend:              {args.backend}")
    print(f"Preprocess:           {args.preprocess}")
    print(f"Tokens requested:     {len(tokens)}")
    print(f"Panels saved:         {len(saved_paths)}")
    print(f"Output dir:           {args.out_dir}")
    print(f"Manifest CSV:         {manifest_path}")
    print()
    print("Saved panels:")
    for p in saved_paths:
        print(f"  {p}")
    print()
    print("Locked rule: reference/error columns were used only after ranking for diagnostic labels/oracle display.")


if __name__ == "__main__":
    main()
