from pathlib import Path
from typing import Dict, Any, List
import json
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def read_image_gray(image_path: Path) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError("Could not read image: {image_path}")
    
    return image

def compute_image_quality(image_path: Path) -> Dict[str, Any]:
    gray = read_image_gray(image_path)

    height, width = gray.shape[:2]

    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))

    # Blur Score: Variance of Laplacian
    # Lower value means blurrier image
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    orb = cv2.ORB_create(nfeatures=1000)
    keypoints = orb.detect(gray, None)
    orb_keypoints = int(len(keypoints))

    return {
        "image_width": int(width),
        "image_height": int(height),
        "brightness_mean": brightness,
        "contrast_std": contrast,
        "blur_score_laplacian_var": blur_score,
        "orb_keypoints": orb_keypoints,
    }

def build_frame_quality_table(synced_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for _, row in synced_df.iterrows():
        image_path = Path(row["image_path"])

        try:
            quality = compute_image_quality(image_path)
            status = "ok"
            error = ""
        except Exception as exc:
            quality = {
                "image_width": None,
                "image_height": None,
                "brightness_mean": None,
                "contrast_std": None,
                "blur_score_laplacian_var": None,
                "orb_keypoints": None,
            }
            status = "error"
            error = str(exc)

        output_row = {
            "imgid": int(row["imgid"]),
            "image_path": row["image_path"],
            "image_filename": row["image_filename"],
            "timestamp": row.get("timestamp", None),
            "x_enu_m": row.get("x_enu_m", None),
            "y_enu_m": row.get("y_enu_m", None),
            "z_enu_m": row.get("z_enu_m", None),
            "lat": row.get("lat", None),
            "lon": row.get("lon", None),
            "alt": row.get("alt", None),
            "quality_status": status,
            "quality_error": error,
        }

        output_row.update(quality)
        rows.append(output_row)

    quality_df = pd.DataFrame(rows)

    valid = quality_df[quality_df["quality_status"] == "ok"].copy()

    if not valid.empty:
        blur_q25 = valid["blur_score_laplacian_var"].quantile(0.25)
        kp_q25 = valid["orb_keypoints"].quantile(0.25)
        contrast_q25 = valid["contrast_std"].quantile(0.25)

        quality_df["is_low_blur_score"] = quality_df["blur_score_laplacian_var"] < blur_q25
        quality_df["is_low_feature_count"] = quality_df["orb_keypoints"] < kp_q25
        quality_df["is_low_contrast"] = quality_df["contrast_std"] < contrast_q25

        quality_df["weak_frame_candidate"] = (
            quality_df["is_low_blur_score"]
            | quality_df["is_low_feature_count"]
            | quality_df["is_low_contrast"]
        )
    else:
        quality_df["is_low_blur_score"] = False
        quality_df["is_low_feature_count"] = False
        quality_df["is_low_contrast"] = False
        quality_df["weak_frame_candidate"] = False

    return quality_df


def create_quality_summary(quality_df: pd.DataFrame) -> Dict[str, Any]:
    valid = quality_df[quality_df["quality_status"] == "ok"].copy()

    summary: Dict[str, Any] = {
        "total_frames": int(len(quality_df)),
        "valid_images": int(len(valid)),
        "failed_images": int((quality_df["quality_status"] != "ok").sum()),
    }

    numeric_cols = [
        "image_width",
        "image_height",
        "brightness_mean",
        "contrast_std",
        "blur_score_laplacian_var",
        "orb_keypoints",
    ]

    for col in numeric_cols:
        if col in valid.columns and not valid.empty:
            summary[col] = {
                "min": float(valid[col].min()),
                "mean": float(valid[col].mean()),
                "median": float(valid[col].median()),
                "max": float(valid[col].max()),
            }

    if "weak_frame_candidate" in quality_df.columns:
        summary["weak_frame_candidates"] = int(quality_df["weak_frame_candidate"].sum())

    if "is_low_blur_score" in quality_df.columns:
        summary["low_blur_score_frames"] = int(quality_df["is_low_blur_score"].sum())

    if "is_low_feature_count" in quality_df.columns:
        summary["low_feature_count_frames"] = int(quality_df["is_low_feature_count"].sum())

    if "is_low_contrast" in quality_df.columns:
        summary["low_contrast_frames"] = int(quality_df["is_low_contrast"].sum())

    return summary


def save_json(data: Dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w") as f:
        json.dump(data, f, indent=2)


def plot_sample_frames(quality_df: pd.DataFrame, output_path: Path, max_frames: int = 12) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if quality_df.empty:
        raise ValueError("quality_df is empty.")

    # Select frames evenly across the sequence.
    indices = np.linspace(0, len(quality_df) - 1, min(max_frames, len(quality_df))).astype(int)
    sample_df = quality_df.iloc[indices].copy()

    cols = 4
    rows = int(np.ceil(len(sample_df) / cols))

    plt.figure(figsize=(14, 3.5 * rows))

    for i, (_, row) in enumerate(sample_df.iterrows(), start=1):
        image = cv2.imread(str(row["image_path"]), cv2.IMREAD_GRAYSCALE)

        if image is None:
            continue

        plt.subplot(rows, cols, i)
        plt.imshow(image, cmap="gray")
        plt.axis("off")
        plt.title(
            f"imgid {int(row['imgid'])}\n"
            f"blur={row['blur_score_laplacian_var']:.1f}, "
            f"kp={int(row['orb_keypoints'])}"
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_quality_metrics(quality_df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    valid = quality_df[quality_df["quality_status"] == "ok"].copy()

    if valid.empty:
        raise ValueError("No valid images available for quality plots.")

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    axes[0].plot(valid["imgid"], valid["blur_score_laplacian_var"], linewidth=1.2)
    axes[0].set_ylabel("Blur score")
    axes[0].set_title("Frame Quality Metrics")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(valid["imgid"], valid["orb_keypoints"], linewidth=1.2)
    axes[1].set_ylabel("ORB keypoints")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(valid["imgid"], valid["brightness_mean"], linewidth=1.2, label="Brightness")
    axes[2].plot(valid["imgid"], valid["contrast_std"], linewidth=1.2, label="Contrast")
    axes[2].set_xlabel("Image ID")
    axes[2].set_ylabel("Pixel statistic")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()