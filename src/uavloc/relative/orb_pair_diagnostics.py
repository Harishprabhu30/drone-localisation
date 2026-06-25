from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import json

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def read_gray(image_path: Path) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    return image


def compute_orb_pair_match(
    image_path_1: Path,
    image_path_2: Path,
    nfeatures: int = 2000,
    ratio_thresh: float = 0.75,
    ransac_reproj_thresh: float = 3.0,
) -> Tuple[Dict[str, Any], Optional[np.ndarray], Optional[List[cv2.DMatch]], Optional[List[cv2.KeyPoint]], Optional[List[cv2.KeyPoint]], Optional[np.ndarray], Optional[np.ndarray]]:
    img1 = read_gray(image_path_1)
    img2 = read_gray(image_path_2)

    orb = cv2.ORB_create(nfeatures=nfeatures)

    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    result: Dict[str, Any] = {
        "kp1": int(len(kp1)),
        "kp2": int(len(kp2)),
        "raw_matches": 0,
        "good_matches": 0,
        "homography_found": False,
        "ransac_inliers": 0,
        "inlier_ratio": 0.0,
        "mean_match_distance": None,
        "median_match_distance": None,
        "status": "ok",
        "error": "",
    }

    if des1 is None or des2 is None or len(kp1) == 0 or len(kp2) == 0:
        result["status"] = "no_descriptors"
        return result, None, None, kp1, kp2, img1, img2

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn_matches = matcher.knnMatch(des1, des2, k=2)

    good_matches: List[cv2.DMatch] = []

    for pair in knn_matches:
        if len(pair) < 2:
            continue

        m, n = pair

        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)

    result["raw_matches"] = int(len(knn_matches))
    result["good_matches"] = int(len(good_matches))

    if good_matches:
        distances = np.array([m.distance for m in good_matches], dtype=float)
        result["mean_match_distance"] = float(np.mean(distances))
        result["median_match_distance"] = float(np.median(distances))

    homography = None
    inlier_mask = None

    if len(good_matches) >= 4:
        pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        homography, inlier_mask = cv2.findHomography(
            pts1,
            pts2,
            cv2.RANSAC,
            ransac_reproj_thresh,
        )

        if homography is not None and inlier_mask is not None:
            inliers = int(inlier_mask.ravel().sum())
            result["homography_found"] = True
            result["ransac_inliers"] = inliers
            result["inlier_ratio"] = float(inliers / max(len(good_matches), 1))
        else:
            result["status"] = "homography_failed"
    else:
        result["status"] = "too_few_matches"

    return result, homography, good_matches, kp1, kp2, img1, img2


def run_orb_pair_diagnostics(
    synced_df: pd.DataFrame,
    max_pairs: Optional[int] = None,
    nfeatures: int = 2000,
    ratio_thresh: float = 0.75,
    ransac_reproj_thresh: float = 3.0,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    if max_pairs is None:
        max_pairs = len(synced_df) - 1

    max_pairs = min(max_pairs, len(synced_df) - 1)

    for i in range(max_pairs):
        row1 = synced_df.iloc[i]
        row2 = synced_df.iloc[i + 1]

        image_path_1 = Path(row1["image_path"])
        image_path_2 = Path(row2["image_path"])

        try:
            result, _, _, _, _, _, _ = compute_orb_pair_match(
                image_path_1=image_path_1,
                image_path_2=image_path_2,
                nfeatures=nfeatures,
                ratio_thresh=ratio_thresh,
                ransac_reproj_thresh=ransac_reproj_thresh,
            )
        except Exception as exc:
            result = {
                "kp1": None,
                "kp2": None,
                "raw_matches": None,
                "good_matches": None,
                "homography_found": False,
                "ransac_inliers": None,
                "inlier_ratio": None,
                "mean_match_distance": None,
                "median_match_distance": None,
                "status": "error",
                "error": str(exc),
            }

        dx_ref = float(row2["x_enu_m"] - row1["x_enu_m"]) if "x_enu_m" in synced_df.columns else None
        dy_ref = float(row2["y_enu_m"] - row1["y_enu_m"]) if "y_enu_m" in synced_df.columns else None
        dz_ref = float(row2["z_enu_m"] - row1["z_enu_m"]) if "z_enu_m" in synced_df.columns else None

        if dx_ref is not None and dy_ref is not None:
            ref_step_distance_m = float(np.sqrt(dx_ref**2 + dy_ref**2))
        else:
            ref_step_distance_m = None

        output_row = {
            "pair_index": i,
            "imgid_1": int(row1["imgid"]),
            "imgid_2": int(row2["imgid"]),
            "image_1": row1["image_path"],
            "image_2": row2["image_path"],
            "timestamp_1": row1.get("timestamp", None),
            "timestamp_2": row2.get("timestamp", None),
            "ref_dx_m": dx_ref,
            "ref_dy_m": dy_ref,
            "ref_dz_m": dz_ref,
            "ref_step_distance_m": ref_step_distance_m,
        }

        output_row.update(result)

        rows.append(output_row)

    df = pd.DataFrame(rows)

    df["pair_quality"] = "weak"

    valid = df[
        (df["status"] == "ok")
        & (df["homography_found"] == True)
        & df["ransac_inliers"].notna()
        & df["inlier_ratio"].notna()
    ].copy()

    good_mask = (
        (df["status"] == "ok")
        & (df["homography_found"] == True)
        & (df["good_matches"] >= 80)
        & (df["ransac_inliers"] >= 40)
        & (df["inlier_ratio"] >= 0.30)
    )

    medium_mask = (
        (df["status"] == "ok")
        & (df["homography_found"] == True)
        & (df["good_matches"] >= 40)
        & (df["ransac_inliers"] >= 20)
        & (df["inlier_ratio"] >= 0.20)
        & (~good_mask)
    )

    df.loc[good_mask, "pair_quality"] = "good"
    df.loc[medium_mask, "pair_quality"] = "medium"

    return df


def summarize_orb_pairs(pair_df: pd.DataFrame) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "total_pairs": int(len(pair_df)),
        "good_pairs": int((pair_df["pair_quality"] == "good").sum()),
        "medium_pairs": int((pair_df["pair_quality"] == "medium").sum()),
        "weak_pairs": int((pair_df["pair_quality"] == "weak").sum()),
        "homography_success_count": int(pair_df["homography_found"].sum()),
        "status_counts": pair_df["status"].value_counts(dropna=False).to_dict(),
        "quality_counts": pair_df["pair_quality"].value_counts(dropna=False).to_dict(),
    }

    numeric_cols = [
        "kp1",
        "kp2",
        "raw_matches",
        "good_matches",
        "ransac_inliers",
        "inlier_ratio",
        "mean_match_distance",
        "median_match_distance",
        "ref_step_distance_m",
    ]

    for col in numeric_cols:
        if col in pair_df.columns:
            valid = pd.to_numeric(pair_df[col], errors="coerce").dropna()
            if not valid.empty:
                summary[col] = {
                    "min": float(valid.min()),
                    "mean": float(valid.mean()),
                    "median": float(valid.median()),
                    "max": float(valid.max()),
                }

    return summary


def save_json(data: Dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w") as f:
        json.dump(data, f, indent=2)


def plot_pair_quality(pair_df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)

    axes[0].plot(pair_df["imgid_1"], pair_df["good_matches"], linewidth=1.2)
    axes[0].set_ylabel("Good matches")
    axes[0].set_title("ORB Consecutive Frame-Pair Diagnostics")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(pair_df["imgid_1"], pair_df["ransac_inliers"], linewidth=1.2)
    axes[1].set_ylabel("RANSAC inliers")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(pair_df["imgid_1"], pair_df["inlier_ratio"], linewidth=1.2)
    axes[2].set_xlabel("Image ID")
    axes[2].set_ylabel("Inlier ratio")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def draw_pair_matches(
    synced_df: pd.DataFrame,
    pair_row: pd.Series,
    output_path: Path,
    nfeatures: int = 500,
    ratio_thresh: float = 0.75,
    ransac_reproj_thresh: float = 3.0,
    max_draw_matches: int = 80,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    image_path_1 = Path(pair_row["image_1"])
    image_path_2 = Path(pair_row["image_2"])

    result, _, good_matches, kp1, kp2, img1, img2 = compute_orb_pair_match(
        image_path_1=image_path_1,
        image_path_2=image_path_2,
        nfeatures=nfeatures,
        ratio_thresh=ratio_thresh,
        ransac_reproj_thresh=ransac_reproj_thresh,
    )

    if good_matches is None or kp1 is None or kp2 is None or img1 is None or img2 is None:
        raise ValueError(f"Could not draw matches for pair {pair_row['imgid_1']} -> {pair_row['imgid_2']}")

    draw_matches = sorted(good_matches, key=lambda m: m.distance)[:max_draw_matches]

    match_image = cv2.drawMatches(
        img1,
        kp1,
        img2,
        kp2,
        draw_matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    title = (
        f"ORB matches: imgid {int(pair_row['imgid_1'])} -> {int(pair_row['imgid_2'])} | "
        f"good={int(pair_row['good_matches'])}, "
        f"inliers={int(pair_row['ransac_inliers']) if pd.notna(pair_row['ransac_inliers']) else 0}, "
        f"ratio={float(pair_row['inlier_ratio']) if pd.notna(pair_row['inlier_ratio']) else 0:.2f}"
    )

    plt.figure(figsize=(14, 7))
    plt.imshow(match_image, cmap="gray")
    plt.axis("off")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()