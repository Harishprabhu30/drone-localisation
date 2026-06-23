from pathlib import Path
from typing import Dict, List, Optional, Any
import re
import json

import pandas as pd


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def find_image_files(image_dir: Path) -> List[Path]:
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    return sorted(
        [
            p for p in image_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ]
    )


def extract_imgid_from_filename(image_path: Path) -> Optional[int]:
    """
    Extract image id from filename.

    Examples supported:
        1.jpg
        img_000001.jpg
        frame_123.png
    """
    matches = re.findall(r"\d+", image_path.stem)

    if not matches:
        return None

    return int(matches[-1])


def build_image_index(image_dir: Path) -> pd.DataFrame:
    image_files = find_image_files(image_dir)

    rows = []
    for image_path in image_files:
        imgid = extract_imgid_from_filename(image_path)

        rows.append(
            {
                "imgid": imgid,
                "image_path": str(image_path),
                "image_filename": image_path.name,
            }
        )

    df = pd.DataFrame(rows)

    if df.empty:
        raise ValueError(f"No images found in: {image_dir}")

    return df.sort_values("imgid").reset_index(drop=True)


def synchronize_images_with_reference(
    image_df: pd.DataFrame,
    reference_df: pd.DataFrame,
) -> pd.DataFrame:
    if "imgid" not in image_df.columns:
        raise ValueError("image_df must contain imgid column.")

    if "imgid" not in reference_df.columns:
        raise ValueError("reference_df must contain imgid column.")

    ref_cols = [
        "imgid",
        "timestamp",
        "lat",
        "lon",
        "alt",
        "x_enu_m",
        "y_enu_m",
        "z_enu_m",
        "reference_type",
        "fix_type",
        "num_sat",
        "eph_m",
        "epv_m",
        "vel_n_m_s",
        "vel_e_m_s",
        "vel_d_m_s",
    ]

    ref_cols = [c for c in ref_cols if c in reference_df.columns]

    reference_subset = reference_df[ref_cols].copy()

    synced = image_df.merge(
        reference_subset,
        on="imgid",
        how="left",
        validate="one_to_one",
    )

    synced["has_reference"] = synced["timestamp"].notna()

    return synced


def summarize_frame_sync(synced: pd.DataFrame, image_df: pd.DataFrame, reference_df: pd.DataFrame) -> Dict[str, Any]:
    missing_reference = synced[~synced["has_reference"]]

    summary = {
        "image_count": int(len(image_df)),
        "reference_rows": int(len(reference_df)),
        "synced_rows": int(len(synced)),
        "rows_with_reference": int(synced["has_reference"].sum()),
        "rows_missing_reference": int((~synced["has_reference"]).sum()),
        "image_imgid_min": int(image_df["imgid"].min()) if image_df["imgid"].notna().any() else None,
        "image_imgid_max": int(image_df["imgid"].max()) if image_df["imgid"].notna().any() else None,
        "reference_imgid_min": int(reference_df["imgid"].min()) if "imgid" in reference_df.columns else None,
        "reference_imgid_max": int(reference_df["imgid"].max()) if "imgid" in reference_df.columns else None,
        "missing_reference_imgids": missing_reference["imgid"].dropna().astype(int).tolist()[:50],
    }

    return summary


def save_json(data: Dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w") as f:
        json.dump(data, f, indent=2)