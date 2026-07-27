#!/usr/bin/env python3
"""
S8.12C.1 — Spatial multi-crop sweep.

Tests whether LightGlue improves when a satellite candidate is shown at:
- a reduced geographic footprint; and
- multiple spatial translations within the original tile.

Frozen:
- 3 smoke queries
- 20 DINOv2 candidates/query
- original tile IDs
- original candidate ranks
- oracle membership
- evaluation coordinates
- LightGlue configuration

Sweep:
- scales: 60%, 45%
- positions: north, south, east, west, centre

Total:
3 × 20 × 2 × 5 = 600 LightGlue pairs

Aggregation:
All crop instances belonging to one original tile are fused back into one tile.
The primary diagnostic fusion is the maximum LightGlue score across crops.

Outputs:
- crop candidate pools
- directional crop contact sheets
- all crop-level verifier outputs
- tile-level fused rankings
- query-level summaries
- position/scale summaries
- oracle versus non-oracle summaries
- directional winner visualizations
- JSON closeout report

Command to execute:

1. Prepare the crops first created csv, figures, folders

python scripts/villoc/s8_12c1_spatial_multicrop_sweep.py \
  --prepare-only \
  2>&1 | tee \
  outputs/villoc/90_deg/logs/s8_12c1_prepare.log

 2. Once the above is run successfully, execute this: 

python scripts/villoc/s8_12c1_spatial_multicrop_sweep.py \
  --device auto \
  --resize-long 512 \
  --max-keypoints 2048 \
  --ransac-thresh 5.0 \
  2>&1 | tee \
  outputs/villoc/90_deg/logs/s8_12c1_spatial_multicrop_full.log

3. To resume the interrupted run:

python scripts/villoc/s8_12c1_spatial_multicrop_sweep.py \
  --device auto \
  --resize-long 512 \
  --max-keypoints 2048 \
  --ransac-thresh 5.0 \
  --resume \
  2>&1 | tee -a \
  outputs/villoc/90_deg/logs/s8_12c1_spatial_multicrop_full.log

"""

from __future__ import annotations

import argparse
import ast
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path.cwd().resolve()

OUT_BASE = ROOT / "outputs/villoc/90_deg"

BASE_RANKED_CSV = (
    OUT_BASE
    / "metadata/s7d_lightglue"
    / (
        "s7d1_lightglue_candidate_scores_ranked_"
        "s8_12b0_villoc_1024_s512_top20_smoke3.csv"
    )
)

ORACLE_CSV = (
    OUT_BASE
    / "metadata"
    / "s8_10b_uav_tile_oracle_1024_s512.csv"
)

SELECTION_JSON = (
    OUT_BASE
    / "reports/s8_12b"
    / "s8_12b0_smoke_query_selection.json"
)

VERIFIER_SCRIPT = (
    ROOT
    / "scripts/satloc/s7d"
    / "s7d_1_lightglue_merged_candidate_verifier.py"
)

META_DIR = (
    OUT_BASE
    / "metadata/s8_12c1_spatial_multicrop"
)

REPORT_DIR = (
    OUT_BASE
    / "reports/s8_12c1_spatial_multicrop"
)

FIG_DIR = (
    OUT_BASE
    / "figures/s8_12c1_spatial_multicrop"
)

CROP_ROOT = (
    ROOT
    / "data/processed/villoc/90_deg"
    / "s8_12c1_spatial_multicrop_tiles"
)

S7D_META_DIR = (
    OUT_BASE
    / "metadata/s7d_lightglue"
)

S7D_PANEL_DIR = (
    OUT_BASE
    / "figures/s7d_lightglue/panels"
)

QUERY_IDS = ["23", "33", "52"]

SCALES = [0.60, 0.45]

POSITIONS = [
    "north",
    "south",
    "east",
    "west",
    "centre",
]

EXPECTED_POSITION = {
    "23": "east",
    "33": "west and south",
    "52": "centre",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="S8.12C.1 spatial multi-crop sweep."
    )

    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Generate crops and visual sheets without running LightGlue.",
    )

    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Aggregate existing LightGlue outputs only.",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume verifier outputs when supported.",
    )

    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
    )

    parser.add_argument(
        "--resize-long",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--max-keypoints",
        type=int,
        default=2048,
    )

    parser.add_argument(
        "--ransac-thresh",
        type=float,
        default=5.0,
    )

    parser.add_argument(
        "--max-draw-matches",
        type=int,
        default=120,
    )

    return parser.parse_args()


def ensure_directories() -> None:
    for path in [
        META_DIR,
        REPORT_DIR,
        FIG_DIR,
        CROP_ROOT,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def normalize_id(value: Any) -> str:
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass

    text = str(value).strip()

    try:
        number = float(text)

        if np.isfinite(number) and number.is_integer():
            return str(int(number))
    except Exception:
        pass

    return text


def safe_float(value: Any) -> float:
    try:
        result = float(value)

        if np.isfinite(result):
            return result
    except Exception:
        pass

    return float("nan")


def truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value

    if value is None:
        return False

    try:
        if pd.isna(value):
            return False
    except Exception:
        pass

    return str(value).strip().lower() in {
        "true",
        "1",
        "yes",
        "y",
        "t",
    }


def find_column(
    frame: pd.DataFrame,
    candidates: Iterable[str],
    label: str,
    required: bool = True,
) -> str | None:
    mapping = {
        str(column).lower(): column
        for column in frame.columns
    }

    for candidate in candidates:
        if candidate.lower() in mapping:
            return mapping[candidate.lower()]

    if required:
        raise KeyError(
            f"Could not infer {label}. "
            f"Tried: {list(candidates)}. "
            f"Available: {list(frame.columns)}"
        )

    return None


def resolve_path(value: Any) -> Path:
    path = Path(str(value)).expanduser()

    candidates = [
        path,
        ROOT / path,
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    raise FileNotFoundError(
        f"Could not resolve image path: {value}"
    )


def parse_tile_set(value: Any) -> set[str]:
    if value is None:
        return set()

    try:
        if pd.isna(value):
            return set()
    except Exception:
        pass

    if isinstance(
        value,
        (list, tuple, set, np.ndarray),
    ):
        return {
            normalize_id(item)
            for item in value
            if normalize_id(item)
        }

    text = str(value).strip()

    if text.lower() in {
        "",
        "nan",
        "none",
        "null",
        "[]",
    }:
        return set()

    try:
        parsed = ast.literal_eval(text)

        if isinstance(
            parsed,
            (list, tuple, set),
        ):
            return {
                normalize_id(item)
                for item in parsed
                if normalize_id(item)
            }
    except Exception:
        pass

    text = re.sub(
        r"[\[\]\(\)\{\}\"']",
        "",
        text,
    )

    return {
        normalize_id(item)
        for item in re.split(r"[,;|\s]+", text)
        if normalize_id(item)
    }


def load_roles() -> dict[str, str]:
    fallback = {
        "23": "hard",
        "33": "middle",
        "52": "easy",
    }

    if not SELECTION_JSON.exists():
        return fallback

    try:
        data = json.loads(
            SELECTION_JSON.read_text(
                encoding="utf-8"
            )
        )
    except Exception:
        return fallback

    rows = data.get("queries", data)

    if isinstance(rows, dict):
        rows = rows.get("queries", [])

    roles = fallback.copy()

    if isinstance(rows, list):
        for row in rows:
            query_id = normalize_id(
                row.get("query_id")
            )

            role = str(
                row.get("role", "unknown")
            )

            if query_id:
                roles[query_id] = role

    return roles


def load_oracle_sets() -> dict[str, set[str]]:
    if not ORACLE_CSV.exists():
        raise FileNotFoundError(ORACLE_CSV)

    frame = pd.read_csv(ORACLE_CSV)

    query_col = find_column(
        frame,
        ["query_id", "token0_id", "sample_id"],
        "oracle query ID",
    )

    packed_col = find_column(
        frame,
        ["oracle_tile_ids"],
        "packed oracle tile IDs",
        required=False,
    )

    single_col = find_column(
        frame,
        [
            "oracle_tile_id",
            "tile_id",
            "sat_tile_id",
        ],
        "single oracle tile ID",
        required=False,
    )

    oracle_sets: dict[str, set[str]] = {}

    for _, row in frame.iterrows():
        query_id = normalize_id(row[query_col])

        if not query_id:
            continue

        oracle_sets.setdefault(query_id, set())

        if packed_col is not None:
            oracle_sets[query_id].update(
                parse_tile_set(row[packed_col])
            )

        if single_col is not None:
            tile_id = normalize_id(
                row[single_col]
            )

            if tile_id:
                oracle_sets[query_id].add(
                    tile_id
                )

    return oracle_sets


def load_base_pool() -> pd.DataFrame:
    if not BASE_RANKED_CSV.exists():
        raise FileNotFoundError(BASE_RANKED_CSV)

    frame = pd.read_csv(BASE_RANKED_CSV)

    query_col = find_column(
        frame,
        ["query_id", "token0_id"],
        "query ID",
    )

    tile_col = find_column(
        frame,
        ["tile_id", "sat_tile_id"],
        "tile ID",
    )

    rank_col = find_column(
        frame,
        [
            "candidate_rank",
            "retrieval_rank",
            "rank",
        ],
        "candidate rank",
    )

    uav_col = find_column(
        frame,
        [
            "uav_image_path",
            "query_image_path",
        ],
        "UAV image path",
    )

    sat_col = find_column(
        frame,
        [
            "sat_image_path",
            "satellite_image_path",
            "candidate_image_path",
            "tile_path",
        ],
        "satellite image path",
    )

    eval_col = find_column(
        frame,
        [
            "eval_error_m",
            "center_error_m",
            "tile_center_error_m",
        ],
        "evaluation error",
        required=False,
    )

    output = pd.DataFrame(
        {
            "query_id": frame[query_col].map(
                normalize_id
            ),
            "tile_id": frame[tile_col].map(
                normalize_id
            ),
            "candidate_rank": pd.to_numeric(
                frame[rank_col],
                errors="coerce",
            ),
            "uav_image_path": frame[
                uav_col
            ].astype(str),
            "sat_image_path_original": frame[
                sat_col
            ].astype(str),
            "eval_error_m": (
                pd.to_numeric(
                    frame[eval_col],
                    errors="coerce",
                )
                if eval_col is not None
                else np.nan
            ),
        }
    )

    for column in [
        "policy",
        "budget",
        "center_rank",
        "resize_rank",
        "source_count",
        "sources",
        "resize_unique",
        "merge_score",
    ]:
        if column in frame.columns:
            output[column] = frame[column]

    output = output[
        output["query_id"].isin(QUERY_IDS)
    ].copy()

    output = output.sort_values(
        [
            "query_id",
            "candidate_rank",
            "tile_id",
        ],
        kind="mergesort",
    )

    output = output.drop_duplicates(
        ["query_id", "tile_id"],
        keep="first",
    )

    output = (
        output.groupby(
            "query_id",
            group_keys=False,
        )
        .head(20)
        .reset_index(drop=True)
    )

    counts = (
        output.groupby("query_id")
        .size()
        .to_dict()
    )

    for query_id in QUERY_IDS:
        count = counts.get(query_id, 0)

        if count != 20:
            raise RuntimeError(
                f"Query {query_id}: expected "
                f"20 candidates, found {count}."
            )

    return output


def crop_anchor(position: str) -> tuple[float, float]:
    """
    Returns normalized placement parameters.

    x_alpha and y_alpha define the crop's top-left position
    inside the available movement range:

        x0 = x_alpha * (W - crop_width)
        y0 = y_alpha * (H - crop_height)

    Therefore:
    - 0.0 anchors to the left/top edge
    - 0.5 centres the crop
    - 1.0 anchors to the right/bottom edge
    """

    anchors = {
        "north": (0.5, 0.0),
        "south": (0.5, 1.0),
        "east": (1.0, 0.5),
        "west": (0.0, 0.5),
        "centre": (0.5, 0.5),
    }

    if position not in anchors:
        raise ValueError(
            f"Unsupported position: {position}"
        )

    return anchors[position]


def spatial_crop(
    image: np.ndarray,
    scale: float,
    position: str,
) -> tuple[
    np.ndarray,
    tuple[int, int, int, int],
]:
    if image is None or image.size == 0:
        raise ValueError(
            "Cannot crop an empty image."
        )

    height, width = image.shape[:2]

    crop_width = max(
        32,
        min(
            width,
            int(round(width * scale)),
        ),
    )

    crop_height = max(
        32,
        min(
            height,
            int(round(height * scale)),
        ),
    )

    x_alpha, y_alpha = crop_anchor(
        position
    )

    max_x0 = width - crop_width
    max_y0 = height - crop_height

    x0 = int(round(x_alpha * max_x0))
    y0 = int(round(y_alpha * max_y0))

    x0 = max(0, min(x0, max_x0))
    y0 = max(0, min(y0, max_y0))

    x1 = x0 + crop_width
    y1 = y0 + crop_height

    crop = image[y0:y1, x0:x1].copy()

    return crop, (x0, y0, x1, y1)


def crop_output_path(
    scale_percent: int,
    position: str,
    query_id: str,
    tile_id: str,
) -> Path:
    return (
        CROP_ROOT
        / f"scale_{scale_percent}"
        / position
        / f"query_{query_id}"
        / (
            f"tile_{tile_id}_"
            f"s{scale_percent}_{position}.png"
        )
    )


def candidate_pool_path(
    scale_percent: int,
    position: str,
) -> Path:
    return (
        META_DIR
        / (
            "s8_12c1_candidate_pool_"
            f"s{scale_percent}_{position}.csv"
        )
    )


def run_name(
    scale_percent: int,
    position: str,
) -> str:
    return (
        "s8_12c1_villoc_1024_s512_"
        "top20_smoke3_"
        f"s{scale_percent}_{position}"
    )


def verifier_ranked_path(
    scale_percent: int,
    position: str,
) -> Path:
    name = run_name(
        scale_percent,
        position,
    )

    return (
        S7D_META_DIR
        / (
            "s7d1_lightglue_candidate_"
            f"scores_ranked_{name}.csv"
        )
    )


def create_crop_pool(
    base: pd.DataFrame,
    scale: float,
    position: str,
) -> pd.DataFrame:
    scale_percent = int(
        round(scale * 100)
    )

    rows: list[dict[str, Any]] = []

    for index, row in base.iterrows():
        query_id = normalize_id(
            row["query_id"]
        )

        tile_id = normalize_id(
            row["tile_id"]
        )

        source_path = resolve_path(
            row["sat_image_path_original"]
        )

        image = cv2.imread(
            str(source_path),
            cv2.IMREAD_COLOR,
        )

        if image is None:
            raise RuntimeError(
                f"Failed to read {source_path}"
            )

        crop, bounds = spatial_crop(
            image,
            scale,
            position,
        )

        output_path = crop_output_path(
            scale_percent,
            position,
            query_id,
            tile_id,
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        success = cv2.imwrite(
            str(output_path),
            crop,
            [
                cv2.IMWRITE_PNG_COMPRESSION,
                3,
            ],
        )

        if not success:
            raise RuntimeError(
                f"Failed to save {output_path}"
            )

        output = row.to_dict()

        output.update(
            {
                "sat_image_path": str(
                    output_path
                ),
                "context_scale_fraction": float(
                    scale
                ),
                "context_scale_percent": int(
                    scale_percent
                ),
                "crop_position": position,
                "crop_x0_px": int(bounds[0]),
                "crop_y0_px": int(bounds[1]),
                "crop_x1_px": int(bounds[2]),
                "crop_y1_px": int(bounds[3]),
                "crop_width_px": int(
                    crop.shape[1]
                ),
                "crop_height_px": int(
                    crop.shape[0]
                ),
                "original_width_px": int(
                    image.shape[1]
                ),
                "original_height_px": int(
                    image.shape[0]
                ),
                "crop_instance_id": (
                    f"q{query_id}_"
                    f"tile{tile_id}_"
                    f"s{scale_percent}_"
                    f"{position}"
                ),
            }
        )

        rows.append(output)

        if (index + 1) % 20 == 0:
            print(
                f"  s{scale_percent} "
                f"{position}: "
                f"{index + 1}/{len(base)}"
            )

    result = pd.DataFrame(rows)

    result.to_csv(
        candidate_pool_path(
            scale_percent,
            position,
        ),
        index=False,
    )

    return result


def select_representative_tile(
    query_group: pd.DataFrame,
    oracle_tiles: set[str],
) -> pd.Series:
    oracle_group = query_group[
        query_group["tile_id"].isin(
            oracle_tiles
        )
    ].sort_values(
        "candidate_rank",
        kind="mergesort",
    )

    if not oracle_group.empty:
        return oracle_group.iloc[0]

    return query_group.sort_values(
        "candidate_rank",
        kind="mergesort",
    ).iloc[0]


def load_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(
        str(path),
        cv2.IMREAD_COLOR,
    )

    if image is None:
        raise RuntimeError(
            f"Failed to read image: {path}"
        )

    return cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB,
    )


def save_directional_crop_sheets(
    base: pd.DataFrame,
    oracle_sets: dict[str, set[str]],
    roles: dict[str, str],
) -> None:
    """
    One figure per query and scale.

    Layout:
    UAV | north | centre
        | west  | east
        | south | full tile
    """

    for query_id in QUERY_IDS:
        query_group = base[
            base["query_id"] == query_id
        ]

        representative = (
            select_representative_tile(
                query_group,
                oracle_sets.get(
                    query_id,
                    set(),
                ),
            )
        )

        tile_id = normalize_id(
            representative["tile_id"]
        )

        original_path = resolve_path(
            representative[
                "sat_image_path_original"
            ]
        )

        uav_path = resolve_path(
            representative[
                "uav_image_path"
            ]
        )

        original_bgr = cv2.imread(
            str(original_path),
            cv2.IMREAD_COLOR,
        )

        uav_rgb = load_rgb(uav_path)
        original_rgb = load_rgb(
            original_path
        )

        for scale in SCALES:
            scale_percent = int(
                round(scale * 100)
            )

            fig, axes = plt.subplots(
                2,
                4,
                figsize=(18, 9),
            )

            axes = np.asarray(
                axes
            ).reshape(-1)

            axes[0].imshow(uav_rgb)
            axes[0].set_title(
                f"UAV query {query_id}\n"
                f"role={roles.get(query_id)}"
            )
            axes[0].axis("off")

            display_positions = [
                "north",
                "centre",
                "east",
                "west",
                "south",
            ]

            for axis_index, position in enumerate(
                display_positions,
                start=1,
            ):
                crop, bounds = spatial_crop(
                    original_bgr,
                    scale,
                    position,
                )

                crop_rgb = cv2.cvtColor(
                    crop,
                    cv2.COLOR_BGR2RGB,
                )

                marker = (
                    "EXPECTED"
                    if position
                    == EXPECTED_POSITION[
                        query_id
                    ]
                    else ""
                )

                axes[axis_index].imshow(
                    crop_rgb
                )

                axes[axis_index].set_title(
                    f"{position.upper()} "
                    f"{marker}\n"
                    f"x={bounds[0]}:{bounds[2]}, "
                    f"y={bounds[1]}:{bounds[3]}"
                )

                axes[axis_index].axis("off")

            axes[6].imshow(original_rgb)
            axes[6].set_title(
                "Original full tile\n"
                f"tile={tile_id}, "
                f"DINO rank="
                f"{int(representative['candidate_rank'])}"
            )
            axes[6].axis("off")

            axes[7].axis("off")
            axes[7].text(
                0.02,
                0.95,
                (
                    "Pre-run hypothesis\n\n"
                    f"Expected best position: "
                    f"{EXPECTED_POSITION[query_id]}\n\n"
                    f"Scale: {scale_percent}%\n"
                    f"Crop size: "
                    f"{int(original_bgr.shape[1] * scale)}"
                    f" × "
                    f"{int(original_bgr.shape[0] * scale)} px\n\n"
                    "Expectation is diagnostic only.\n"
                    "It does not affect ranking."
                ),
                va="top",
                fontsize=12,
            )

            fig.suptitle(
                "S8.12C.1 — Spatial crop bank\n"
                f"query={query_id}, "
                f"scale={scale_percent}%",
                fontsize=14,
            )

            fig.tight_layout(
                rect=[0, 0, 1, 0.93]
            )

            output_path = (
                FIG_DIR
                / (
                    f"s8_12c1_query_{query_id}_"
                    f"s{scale_percent}_"
                    "directional_crop_sheet.png"
                )
            )

            fig.savefig(
                output_path,
                dpi=180,
            )

            plt.close(fig)


def run_verifier(
    scale_percent: int,
    position: str,
    args: argparse.Namespace,
) -> None:
    pool_path = candidate_pool_path(
        scale_percent,
        position,
    )

    name = run_name(
        scale_percent,
        position,
    )

    command = [
        sys.executable,
        str(VERIFIER_SCRIPT),
        "--candidate-pool",
        str(pool_path),
        "--out-base",
        str(OUT_BASE),
        "--run-name",
        name,
        "--query-ids",
        ",".join(QUERY_IDS),
        "--max-candidates",
        "20",
        "--device",
        args.device,
        "--resize-long",
        str(args.resize_long),
        "--max-keypoints",
        str(args.max_keypoints),
        "--ransac-thresh",
        str(args.ransac_thresh),
        "--checkpoint-every-candidates",
        "10",
        "--status-every-candidates",
        "5",
        "--save-panels",
        "--panel-query-ids",
        ",".join(QUERY_IDS),
        "--max-draw-matches",
        str(args.max_draw_matches),
    ]

    if args.resume:
        command.append("--resume")

    print()
    print(
        f"LightGlue: scale={scale_percent}% "
        f"position={position}"
    )
    print("-" * 55)

    subprocess.run(
        command,
        cwd=ROOT,
        check=True,
    )


def standardize_verifier_output(
    scale_percent: int,
    position: str,
    oracle_sets: dict[str, set[str]],
    roles: dict[str, str],
) -> pd.DataFrame:
    path = verifier_ranked_path(
        scale_percent,
        position,
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Missing verifier output: {path}"
        )

    frame = pd.read_csv(path)

    query_col = find_column(
        frame,
        ["query_id", "token0_id"],
        "query ID",
    )

    tile_col = find_column(
        frame,
        ["tile_id", "sat_tile_id"],
        "tile ID",
    )

    candidate_rank_col = find_column(
        frame,
        [
            "candidate_rank",
            "retrieval_rank",
            "rank",
        ],
        "candidate rank",
    )

    lg_rank_col = find_column(
        frame,
        ["lightglue_rank"],
        "LightGlue rank",
    )

    frame["query_id_norm"] = frame[
        query_col
    ].map(normalize_id)

    frame["tile_id_norm"] = frame[
        tile_col
    ].map(normalize_id)

    frame["candidate_rank_num"] = (
        pd.to_numeric(
            frame[candidate_rank_col],
            errors="coerce",
        )
    )

    frame["lightglue_rank_num"] = (
        pd.to_numeric(
            frame[lg_rank_col],
            errors="coerce",
        )
    )

    metric_candidates = {
        "lightglue_score_num": [
            "lightglue_score",
            "lg_score_num",
        ],
        "matches_num": [
            "lightglue_matches",
            "lg_matches_num",
        ],
        "inliers_num": [
            "lightglue_ransac_inliers",
            "lg_inliers_num",
        ],
        "inlier_ratio_num": [
            "lightglue_inlier_ratio",
        ],
        "uav_coverage_num": [
            "lightglue_uav_coverage",
        ],
        "sat_coverage_num": [
            "lightglue_sat_coverage",
        ],
        "eval_error_num": [
            "eval_error_m",
            "center_error_m",
            "tile_center_error_m",
        ],
        "runtime_num": [
            "runtime_s",
        ],
    }

    for output_column, candidates in (
        metric_candidates.items()
    ):
        source = find_column(
            frame,
            candidates,
            output_column,
            required=False,
        )

        frame[output_column] = (
            pd.to_numeric(
                frame[source],
                errors="coerce",
            )
            if source is not None
            else np.nan
        )

    homography_col = find_column(
        frame,
        ["lightglue_homography_success"],
        "homography success",
        required=False,
    )

    frame["homography_success"] = (
        frame[homography_col].map(
            truthy
        )
        if homography_col is not None
        else False
    )

    frame["min_coverage_num"] = np.minimum(
        frame[
            "uav_coverage_num"
        ].fillna(0),
        frame[
            "sat_coverage_num"
        ].fillna(0),
    )

    frame["context_scale_percent"] = (
        scale_percent
    )

    frame["crop_position"] = position

    frame["crop_instance_id"] = [
        (
            f"q{query_id}_"
            f"tile{tile_id}_"
            f"s{scale_percent}_"
            f"{position}"
        )
        for query_id, tile_id in zip(
            frame["query_id_norm"],
            frame["tile_id_norm"],
        )
    ]

    frame["is_geometric_oracle"] = [
        (
            tile_id
            in oracle_sets.get(
                query_id,
                set(),
            )
        )
        for query_id, tile_id in zip(
            frame["query_id_norm"],
            frame["tile_id_norm"],
        )
    ]

    frame["role"] = (
        frame["query_id_norm"]
        .map(roles)
        .fillna("unknown")
    )

    frame["geometry_gate"] = (
        frame["homography_success"]
        & (
            frame["inliers_num"]
            >= 10
        )
        & (
            frame["inlier_ratio_num"]
            >= 0.15
        )
        & (
            frame["min_coverage_num"]
            >= 0.10
        )
    )

    sat_path_col = find_column(
        frame,
        [
            "sat_image_path",
            "satellite_image_path",
            "candidate_image_path",
            "tile_path",
        ],
        "satellite image path",
        required=False,
    )

    uav_path_col = find_column(
        frame,
        [
            "uav_image_path",
            "query_image_path",
        ],
        "UAV image path",
        required=False,
    )

    frame["sat_crop_path"] = (
        frame[sat_path_col].astype(str)
        if sat_path_col is not None
        else ""
    )

    frame["uav_image_path_norm"] = (
        frame[uav_path_col].astype(str)
        if uav_path_col is not None
        else ""
    )

    return frame


def choose_best_crop(
    group: pd.DataFrame,
) -> pd.Series:
    """
    Diagnostic max fusion.

    Tie-breaking:
    1. LightGlue score
    2. RANSAC inliers
    3. inlier ratio
    4. minimum spatial coverage
    5. matches
    6. larger crop scale
    7. deterministic position order
    """

    position_order = {
        "centre": 0,
        "north": 1,
        "east": 2,
        "south": 3,
        "west": 4,
    }

    ranked = group.copy()

    ranked["_position_order"] = (
        ranked["crop_position"]
        .map(position_order)
        .fillna(99)
    )

    ranked = ranked.sort_values(
        [
            "lightglue_score_num",
            "inliers_num",
            "inlier_ratio_num",
            "min_coverage_num",
            "matches_num",
            "context_scale_percent",
            "_position_order",
        ],
        ascending=[
            False,
            False,
            False,
            False,
            False,
            False,
            True,
        ],
        kind="mergesort",
    )

    return ranked.iloc[0]


def build_fused_tile_table(
    crop_results: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for (
        query_id,
        tile_id,
    ), group in crop_results.groupby(
        [
            "query_id_norm",
            "tile_id_norm",
        ],
        sort=True,
    ):
        best = choose_best_crop(group)

        valid_group = group[
            group["geometry_gate"]
        ]

        ordered_scores = (
            pd.to_numeric(
                group["lightglue_score_num"],
                errors="coerce",
            )
            .dropna()
            .sort_values(
                ascending=False
            )
            .to_numpy()
        )

        best_score = (
            float(ordered_scores[0])
            if len(ordered_scores) >= 1
            else float("nan")
        )

        second_best_score = (
            float(ordered_scores[1])
            if len(ordered_scores) >= 2
            else float("nan")
        )

        score_gap = (
            best_score - second_best_score
            if (
                np.isfinite(best_score)
                and np.isfinite(
                    second_best_score
                )
            )
            else float("nan")
        )

        rows.append(
            {
                "query_id": query_id,
                "role": str(best["role"]),
                "tile_id": tile_id,
                "candidate_rank": safe_float(
                    best["candidate_rank_num"]
                ),
                "eval_error_m": safe_float(
                    best["eval_error_num"]
                ),
                "is_geometric_oracle": bool(
                    best["is_geometric_oracle"]
                ),
                "fused_lightglue_score": (
                    best_score
                ),
                "second_best_crop_score": (
                    second_best_score
                ),
                "best_second_score_gap": (
                    score_gap
                ),
                "best_crop_scale_percent": int(
                    best[
                        "context_scale_percent"
                    ]
                ),
                "best_crop_position": str(
                    best["crop_position"]
                ),
                "best_crop_matches": safe_float(
                    best["matches_num"]
                ),
                "best_crop_inliers": safe_float(
                    best["inliers_num"]
                ),
                "best_crop_inlier_ratio": (
                    safe_float(
                        best[
                            "inlier_ratio_num"
                        ]
                    )
                ),
                "best_crop_min_coverage": (
                    safe_float(
                        best[
                            "min_coverage_num"
                        ]
                    )
                ),
                "best_crop_homography_success": (
                    bool(
                        best[
                            "homography_success"
                        ]
                    )
                ),
                "best_crop_geometry_gate": bool(
                    best["geometry_gate"]
                ),
                "valid_crop_count": int(
                    len(valid_group)
                ),
                "positive_inlier_crop_count": int(
                    (
                        group["inliers_num"] > 0
                    ).sum()
                ),
                "crop_trials": int(
                    len(group)
                ),
                "best_crop_instance_id": str(
                    best["crop_instance_id"]
                ),
                "best_crop_image_path": str(
                    best["sat_crop_path"]
                ),
                "uav_image_path": str(
                    best[
                        "uav_image_path_norm"
                    ]
                ),
                "expected_position": (
                    EXPECTED_POSITION.get(
                        query_id,
                        "",
                    )
                ),
                "position_matches_visual_hypothesis": (
                    str(
                        best["crop_position"]
                    )
                    == EXPECTED_POSITION.get(
                        query_id,
                        "",
                    )
                ),
            }
        )

    fused = pd.DataFrame(rows)

    fused = fused.sort_values(
        [
            "query_id",
            "fused_lightglue_score",
            "best_crop_inliers",
            "best_crop_inlier_ratio",
            "best_crop_min_coverage",
            "best_crop_matches",
            "candidate_rank",
        ],
        ascending=[
            True,
            False,
            False,
            False,
            False,
            False,
            True,
        ],
        kind="mergesort",
    )

    fused["fused_lightglue_rank"] = (
        fused.groupby(
            "query_id"
        ).cumcount()
        + 1
    )

    return fused


def build_query_summary(
    fused: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for query_id, group in fused.groupby(
        "query_id",
        sort=True,
    ):
        ranked = group.sort_values(
            "fused_lightglue_rank",
            kind="mergesort",
        )

        top = ranked.iloc[0]

        oracle = ranked[
            ranked[
                "is_geometric_oracle"
            ]
        ]

        nonoracle = ranked[
            ~ranked[
                "is_geometric_oracle"
            ]
        ]

        best_oracle = (
            oracle.iloc[0]
            if not oracle.empty
            else None
        )

        best_nonoracle = (
            nonoracle.iloc[0]
            if not nonoracle.empty
            else None
        )

        oracle_score = (
            safe_float(
                best_oracle[
                    "fused_lightglue_score"
                ]
            )
            if best_oracle is not None
            else float("nan")
        )

        nonoracle_score = (
            safe_float(
                best_nonoracle[
                    "fused_lightglue_score"
                ]
            )
            if best_nonoracle is not None
            else float("nan")
        )

        rows.append(
            {
                "role": str(top["role"]),
                "query_id": query_id,
                "fused_top_tile_id": str(
                    top["tile_id"]
                ),
                "fused_top_is_oracle": bool(
                    top[
                        "is_geometric_oracle"
                    ]
                ),
                "fused_top_candidate_rank": (
                    safe_float(
                        top[
                            "candidate_rank"
                        ]
                    )
                ),
                "fused_top_eval_error_m": (
                    safe_float(
                        top["eval_error_m"]
                    )
                ),
                "fused_top_score": safe_float(
                    top[
                        "fused_lightglue_score"
                    ]
                ),
                "fused_top_scale_percent": int(
                    top[
                        "best_crop_scale_percent"
                    ]
                ),
                "fused_top_position": str(
                    top[
                        "best_crop_position"
                    ]
                ),
                "fused_top_matches": safe_float(
                    top[
                        "best_crop_matches"
                    ]
                ),
                "fused_top_inliers": safe_float(
                    top[
                        "best_crop_inliers"
                    ]
                ),
                "fused_top_inlier_ratio": (
                    safe_float(
                        top[
                            "best_crop_inlier_ratio"
                        ]
                    )
                ),
                "fused_top_min_coverage": (
                    safe_float(
                        top[
                            "best_crop_min_coverage"
                        ]
                    )
                ),
                "fused_top_valid_crop_count": (
                    int(
                        top[
                            "valid_crop_count"
                        ]
                    )
                ),
                "expected_position": (
                    EXPECTED_POSITION.get(
                        query_id,
                        "",
                    )
                ),
                "top_position_matches_hypothesis": (
                    str(
                        top[
                            "best_crop_position"
                        ]
                    )
                    == EXPECTED_POSITION.get(
                        query_id,
                        "",
                    )
                ),
                "best_oracle_tile_id": (
                    str(
                        best_oracle[
                            "tile_id"
                        ]
                    )
                    if best_oracle is not None
                    else ""
                ),
                "best_oracle_fused_rank": (
                    safe_float(
                        best_oracle[
                            "fused_lightglue_rank"
                        ]
                    )
                    if best_oracle is not None
                    else float("nan")
                ),
                "best_oracle_candidate_rank": (
                    safe_float(
                        best_oracle[
                            "candidate_rank"
                        ]
                    )
                    if best_oracle is not None
                    else float("nan")
                ),
                "best_oracle_score": (
                    oracle_score
                ),
                "best_oracle_scale_percent": (
                    int(
                        best_oracle[
                            "best_crop_scale_percent"
                        ]
                    )
                    if best_oracle is not None
                    else np.nan
                ),
                "best_oracle_position": (
                    str(
                        best_oracle[
                            "best_crop_position"
                        ]
                    )
                    if best_oracle is not None
                    else ""
                ),
                "best_oracle_inliers": (
                    safe_float(
                        best_oracle[
                            "best_crop_inliers"
                        ]
                    )
                    if best_oracle is not None
                    else float("nan")
                ),
                "best_oracle_inlier_ratio": (
                    safe_float(
                        best_oracle[
                            "best_crop_inlier_ratio"
                        ]
                    )
                    if best_oracle is not None
                    else float("nan")
                ),
                "best_oracle_min_coverage": (
                    safe_float(
                        best_oracle[
                            "best_crop_min_coverage"
                        ]
                    )
                    if best_oracle is not None
                    else float("nan")
                ),
                "best_nonoracle_score": (
                    nonoracle_score
                ),
                "oracle_score_margin": (
                    oracle_score
                    - nonoracle_score
                    if (
                        np.isfinite(
                            oracle_score
                        )
                        and np.isfinite(
                            nonoracle_score
                        )
                    )
                    else float("nan")
                ),
            }
        )

    return pd.DataFrame(rows)


def build_position_scale_summary(
    crop_results: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for (
        scale_percent,
        position,
        is_oracle,
    ), group in crop_results.groupby(
        [
            "context_scale_percent",
            "crop_position",
            "is_geometric_oracle",
        ],
        sort=True,
    ):
        rows.append(
            {
                "context_scale_percent": int(
                    scale_percent
                ),
                "crop_position": position,
                "group": (
                    "oracle"
                    if bool(is_oracle)
                    else "non_oracle"
                ),
                "pairs": int(len(group)),
                "homography_success_rate": (
                    float(
                        group[
                            "homography_success"
                        ].mean()
                    )
                ),
                "geometry_gate_rate": float(
                    group[
                        "geometry_gate"
                    ].mean()
                ),
                "median_matches": float(
                    pd.to_numeric(
                        group["matches_num"],
                        errors="coerce",
                    ).median()
                ),
                "median_inliers": float(
                    pd.to_numeric(
                        group["inliers_num"],
                        errors="coerce",
                    ).median()
                ),
                "median_inlier_ratio": float(
                    pd.to_numeric(
                        group[
                            "inlier_ratio_num"
                        ],
                        errors="coerce",
                    ).median()
                ),
                "median_min_coverage": float(
                    pd.to_numeric(
                        group[
                            "min_coverage_num"
                        ],
                        errors="coerce",
                    ).median()
                ),
                "median_lightglue_score": (
                    float(
                        pd.to_numeric(
                            group[
                                "lightglue_score_num"
                            ],
                            errors="coerce",
                        ).median()
                    )
                ),
            }
        )

    return pd.DataFrame(rows)


def build_query_direction_table(
    crop_results: pd.DataFrame,
) -> pd.DataFrame:
    """
    Best oracle crop at every query × scale × position.

    This directly tests:
    - q23 north
    - q33 east
    - q52 centre
    """

    rows: list[dict[str, Any]] = []

    oracle = crop_results[
        crop_results[
            "is_geometric_oracle"
        ]
    ]

    for (
        query_id,
        scale_percent,
        position,
    ), group in oracle.groupby(
        [
            "query_id_norm",
            "context_scale_percent",
            "crop_position",
        ],
        sort=True,
    ):
        best = choose_best_crop(group)

        rows.append(
            {
                "query_id": query_id,
                "role": str(best["role"]),
                "context_scale_percent": int(
                    scale_percent
                ),
                "crop_position": position,
                "expected_position": (
                    EXPECTED_POSITION.get(
                        query_id,
                        "",
                    )
                ),
                "is_expected_position": (
                    position
                    == EXPECTED_POSITION.get(
                        query_id,
                        "",
                    )
                ),
                "best_oracle_tile_id": str(
                    best["tile_id_norm"]
                ),
                "best_oracle_candidate_rank": (
                    safe_float(
                        best[
                            "candidate_rank_num"
                        ]
                    )
                ),
                "best_oracle_score": safe_float(
                    best[
                        "lightglue_score_num"
                    ]
                ),
                "best_oracle_matches": safe_float(
                    best["matches_num"]
                ),
                "best_oracle_inliers": safe_float(
                    best["inliers_num"]
                ),
                "best_oracle_inlier_ratio": (
                    safe_float(
                        best[
                            "inlier_ratio_num"
                        ]
                    )
                ),
                "best_oracle_min_coverage": (
                    safe_float(
                        best[
                            "min_coverage_num"
                        ]
                    )
                ),
                "best_oracle_geometry_gate": (
                    bool(
                        best[
                            "geometry_gate"
                        ]
                    )
                ),
            }
        )

    return pd.DataFrame(rows)


def save_query_direction_plots(
    direction_table: pd.DataFrame,
) -> None:
    position_order = [
        "north",
        "east",
        "centre",
        "west",
        "south",
    ]

    for query_id in QUERY_IDS:
        query = direction_table[
            direction_table[
                "query_id"
            ] == query_id
        ].copy()

        if query.empty:
            continue

        fig, axes = plt.subplots(
            1,
            2,
            figsize=(14, 5.5),
            sharey=True,
        )

        for axis, scale_percent in zip(
            axes,
            [60, 45],
        ):
            subset = query[
                query[
                    "context_scale_percent"
                ] == scale_percent
            ].copy()

            subset[
                "crop_position"
            ] = pd.Categorical(
                subset[
                    "crop_position"
                ],
                categories=position_order,
                ordered=True,
            )

            subset = subset.sort_values(
                "crop_position"
            )

            positions = (
                subset[
                    "crop_position"
                ].astype(str)
            )

            scores = subset[
                "best_oracle_score"
            ]

            axis.bar(
                positions,
                scores,
            )

            expected = EXPECTED_POSITION[
                query_id
            ]

            for index, row in subset.reset_index(
                drop=True
            ).iterrows():
                label = (
                    "expected"
                    if row[
                        "crop_position"
                    ] == expected
                    else ""
                )

                axis.text(
                    index,
                    row[
                        "best_oracle_score"
                    ],
                    (
                        f"{row['best_oracle_inliers']:.0f} inl\n"
                        f"{label}"
                    ),
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

            axis.set_title(
                f"{scale_percent}% context"
            )

            axis.set_xlabel(
                "Crop position"
            )

            axis.grid(
                axis="y",
                alpha=0.3,
            )

        axes[0].set_ylabel(
            "Best oracle LightGlue score"
        )

        fig.suptitle(
            "S8.12C.1 — Directional oracle response\n"
            f"query={query_id}, "
            f"expected={EXPECTED_POSITION[query_id]}",
            fontsize=13,
        )

        fig.tight_layout(
            rect=[0, 0, 1, 0.91]
        )

        fig.savefig(
            FIG_DIR
            / (
                f"s8_12c1_query_{query_id}_"
                "directional_oracle_response.png"
            ),
            dpi=180,
        )

        plt.close(fig)


def save_fused_rank_plot(
    query_summary: pd.DataFrame,
) -> None:
    ordered = query_summary.sort_values(
        "query_id"
    )

    labels = [
        (
            f"Q{row.query_id}\n"
            f"{row.role}"
        )
        for row in ordered.itertuples()
    ]

    values = ordered[
        "best_oracle_fused_rank"
    ].to_numpy()

    plt.figure(figsize=(8, 5.5))

    bars = plt.bar(
        labels,
        values,
    )

    for bar, value in zip(
        bars,
        values,
    ):
        plt.text(
            bar.get_x()
            + bar.get_width() / 2,
            value,
            f"{value:.0f}",
            ha="center",
            va="bottom",
        )

    plt.ylabel(
        "Best oracle fused rank"
    )

    plt.title(
        "S8.12C.1 — Oracle rank after "
        "scale-position fusion"
    )

    plt.ylim(
        0,
        max(5, np.nanmax(values) + 1),
    )

    plt.grid(
        axis="y",
        alpha=0.3,
    )

    plt.tight_layout()

    plt.savefig(
        FIG_DIR
        / (
            "s8_12c1_best_oracle_"
            "fused_rank.png"
        ),
        dpi=180,
    )

    plt.close()


def save_fused_winner_sheets(
    query_summary: pd.DataFrame,
    fused: pd.DataFrame,
) -> None:
    for _, summary in query_summary.iterrows():
        query_id = normalize_id(
            summary["query_id"]
        )

        query = fused[
            fused["query_id"] == query_id
        ].sort_values(
            "fused_lightglue_rank"
        )

        if query.empty:
            continue

        top_rows = query.head(5)

        uav_path = resolve_path(
            query.iloc[0][
                "uav_image_path"
            ]
        )

        uav_rgb = load_rgb(uav_path)

        fig, axes = plt.subplots(
            2,
            3,
            figsize=(16, 10),
        )

        axes = np.asarray(
            axes
        ).reshape(-1)

        axes[0].imshow(uav_rgb)

        axes[0].set_title(
            f"UAV query {query_id}\n"
            f"expected direction="
            f"{EXPECTED_POSITION[query_id]}"
        )

        axes[0].axis("off")

        for axis_index, (_, row) in enumerate(
            top_rows.iterrows(),
            start=1,
        ):
            axis = axes[axis_index]

            try:
                crop_path = resolve_path(
                    row[
                        "best_crop_image_path"
                    ]
                )

                crop_rgb = load_rgb(
                    crop_path
                )

                axis.imshow(crop_rgb)
            except Exception:
                pass

            oracle_label = (
                "ORACLE"
                if bool(
                    row[
                        "is_geometric_oracle"
                    ]
                )
                else "NON-ORACLE"
            )

            axis.set_title(
                f"Fused rank "
                f"{int(row['fused_lightglue_rank'])} "
                f"— {oracle_label}\n"
                f"tile={row['tile_id']}, "
                f"DINO-r="
                f"{row['candidate_rank']:.0f}\n"
                f"{row['best_crop_scale_percent']}% "
                f"{row['best_crop_position']}\n"
                f"score="
                f"{row['fused_lightglue_score']:.2f}, "
                f"inliers="
                f"{row['best_crop_inliers']:.0f}, "
                f"ratio="
                f"{row['best_crop_inlier_ratio']:.2f}\n"
                f"valid crops="
                f"{row['valid_crop_count']}/10"
            )

            axis.axis("off")

        fig.suptitle(
            "S8.12C.1 — Fused tile ranking "
            "and winning spatial crops",
            fontsize=14,
        )

        fig.tight_layout(
            rect=[0, 0, 1, 0.94]
        )

        fig.savefig(
            FIG_DIR
            / (
                f"s8_12c1_query_{query_id}_"
                "fused_top5_winning_crops.png"
            ),
            dpi=180,
        )

        plt.close(fig)


def save_panel_inventory() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for scale in SCALES:
        scale_percent = int(
            round(scale * 100)
        )

        for position in POSITIONS:
            name = run_name(
                scale_percent,
                position,
            )

            found = sorted(
                S7D_PANEL_DIR.glob(
                    f"*{name}*.png"
                )
            )

            rows.append(
                {
                    "context_scale_percent": (
                        scale_percent
                    ),
                    "crop_position": position,
                    "panel_count": len(found),
                    "panel_paths": "|".join(
                        str(path)
                        for path in found
                    ),
                }
            )

    inventory = pd.DataFrame(rows)

    inventory.to_csv(
        REPORT_DIR
        / (
            "s8_12c1_lightglue_"
            "panel_inventory.csv"
        ),
        index=False,
    )

    return inventory


def aggregate(
    oracle_sets: dict[str, set[str]],
    roles: dict[str, str],
) -> None:
    crop_frames: list[pd.DataFrame] = []

    for scale in SCALES:
        scale_percent = int(
            round(scale * 100)
        )

        for position in POSITIONS:
            crop_frames.append(
                standardize_verifier_output(
                    scale_percent,
                    position,
                    oracle_sets,
                    roles,
                )
            )

    crop_results = pd.concat(
        crop_frames,
        ignore_index=True,
    )

    crop_results.to_csv(
        META_DIR
        / (
            "s8_12c1_all_crop_level_"
            "lightglue_results.csv"
        ),
        index=False,
    )

    expected_pairs = (
        len(QUERY_IDS)
        * 20
        * len(SCALES)
        * len(POSITIONS)
    )

    if len(crop_results) != expected_pairs:
        raise RuntimeError(
            f"Expected {expected_pairs} crop pairs, "
            f"found {len(crop_results)}."
        )

    fused = build_fused_tile_table(
        crop_results
    )

    fused.to_csv(
        META_DIR
        / (
            "s8_12c1_fused_tile_"
            "rankings.csv"
        ),
        index=False,
    )

    query_summary = build_query_summary(
        fused
    )

    query_summary.to_csv(
        REPORT_DIR
        / (
            "s8_12c1_query_fused_"
            "summary.csv"
        ),
        index=False,
    )

    position_scale_summary = (
        build_position_scale_summary(
            crop_results
        )
    )

    position_scale_summary.to_csv(
        REPORT_DIR
        / (
            "s8_12c1_position_scale_"
            "oracle_nonoracle_summary.csv"
        ),
        index=False,
    )

    direction_table = (
        build_query_direction_table(
            crop_results
        )
    )

    direction_table.to_csv(
        REPORT_DIR
        / (
            "s8_12c1_query_direction_"
            "oracle_response.csv"
        ),
        index=False,
    )

    save_query_direction_plots(
        direction_table
    )

    save_fused_rank_plot(
        query_summary
    )

    save_fused_winner_sheets(
        query_summary,
        fused,
    )

    save_panel_inventory()

    top1_hits = int(
        query_summary[
            "fused_top_is_oracle"
        ].sum()
    )

    oracle_ranks = pd.to_numeric(
        query_summary[
            "best_oracle_fused_rank"
        ],
        errors="coerce",
    )

    hypothesis_matches = int(
        query_summary[
            "top_position_matches_hypothesis"
        ].sum()
    )

    all_top1 = top1_hits == len(
        QUERY_IDS
    )

    all_expected_positions = (
        hypothesis_matches
        == len(QUERY_IDS)
    )

    if all_top1:
        status = (
            "PASS_SPATIAL_MULTICROP_"
            "ORACLE_TOP1_3_OF_3"
        )

        decision = (
            "Scale-position search recovered an "
            "oracle Top-1 tile for all smoke queries. "
            "Proceed to crop-fusion confidence analysis "
            "and efficiency reduction."
        )
    elif top1_hits > 2:
        status = (
            "PASS_SPATIAL_MULTICROP_"
            "PARTIAL_STRONG"
        )

        decision = (
            "Spatial multi-crop improved the verifier "
            "but did not fully resolve all queries. "
            "Inspect directional winners and false-positive "
            "crop multiplicity before expanding."
        )
    elif oracle_ranks.median() <= 2:
        status = (
            "PASS_SPATIAL_MULTICROP_"
            "ORACLE_NEAR_TOP"
        )

        decision = (
            "Oracle tiles are consistently near the top, "
            "but max-score fusion still permits strong "
            "non-oracle crops. Proceed to consistency-aware "
            "fusion rather than a larger benchmark."
        )
    else:
        status = (
            "PASS_SPATIAL_MULTICROP_"
            "NO_CLEAR_GAIN"
        )

        decision = (
            "Spatial translation did not provide a clear "
            "oracle-ranking gain. Proceed to structural "
            "reranking or attention-guided crop proposal."
        )

    report = {
        "stage": "S8.12C.1",
        "title": (
            "Spatial multi-crop sweep"
        ),
        "queries": QUERY_IDS,
        "scales_percent": [60, 45],
        "positions": POSITIONS,
        "expected_directions": (
            EXPECTED_POSITION
        ),
        "candidates_per_query": 20,
        "crop_trials_per_tile": 10,
        "total_pairs": int(
            len(crop_results)
        ),
        "fused_tiles": int(
            len(fused)
        ),
        "top1_oracle_hits": top1_hits,
        "top1_oracle_rate": (
            top1_hits / len(QUERY_IDS)
        ),
        "median_best_oracle_fused_rank": (
            float(
                oracle_ranks.median()
            )
        ),
        "direction_hypothesis_matches": (
            hypothesis_matches
        ),
        "all_expected_directions_confirmed": (
            all_expected_positions
        ),
        "fusion_policy": (
            "Maximum LightGlue score across "
            "the 10 crops belonging to each tile"
        ),
        "fusion_warning": (
            "Max fusion is diagnostic. Ten crop trials "
            "increase false-positive opportunity and must "
            "not yet be treated as a calibrated production "
            "confidence score."
        ),
        "status": status,
        "decision": decision,
        "outputs": {
            "crop_results": str(
                META_DIR
                / (
                    "s8_12c1_all_crop_level_"
                    "lightglue_results.csv"
                )
            ),
            "fused_rankings": str(
                META_DIR
                / (
                    "s8_12c1_fused_tile_"
                    "rankings.csv"
                )
            ),
            "query_summary": str(
                REPORT_DIR
                / (
                    "s8_12c1_query_fused_"
                    "summary.csv"
                )
            ),
            "direction_table": str(
                REPORT_DIR
                / (
                    "s8_12c1_query_direction_"
                    "oracle_response.csv"
                )
            ),
            "position_scale_summary": str(
                REPORT_DIR
                / (
                    "s8_12c1_position_scale_"
                    "oracle_nonoracle_summary.csv"
                )
            ),
            "figures": str(FIG_DIR),
        },
    }

    report_path = (
        REPORT_DIR
        / (
            "s8_12c1_spatial_multicrop_"
            "summary.json"
        )
    )

    report_path.write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("Query-level fused ranking")
    print("-------------------------")

    display_columns = [
        "role",
        "query_id",
        "fused_top_is_oracle",
        "fused_top_candidate_rank",
        "fused_top_score",
        "fused_top_scale_percent",
        "fused_top_position",
        "fused_top_inliers",
        "fused_top_inlier_ratio",
        "fused_top_min_coverage",
        "expected_position",
        "top_position_matches_hypothesis",
        "best_oracle_fused_rank",
        "best_oracle_scale_percent",
        "best_oracle_position",
        "best_oracle_inliers",
        "oracle_score_margin",
    ]

    print(
        query_summary[
            display_columns
        ].to_string(index=False)
    )

    print()
    print("Directional oracle response")
    print("---------------------------")

    print(
        direction_table.sort_values(
            [
                "query_id",
                "context_scale_percent",
                "best_oracle_score",
            ],
            ascending=[
                True,
                False,
                False,
            ],
        ).to_string(index=False)
    )

    print()
    print("----------------------------------------")
    print("S8.12C.1 COMPLETE")
    print("STATUS:", status)
    print(
        "FUSED TOP-1 ORACLE:",
        f"{top1_hits}/{len(QUERY_IDS)}",
    )
    print(
        "DIRECTION HYPOTHESIS:",
        f"{hypothesis_matches}/{len(QUERY_IDS)}",
    )
    print("DECISION:", decision)
    print("Report:", report_path)


def main() -> int:
    args = parse_args()

    ensure_directories()

    print(
        "S8.12C.1 — Spatial Multi-Crop Sweep"
    )
    print(
        "-----------------------------------"
    )
    print("Queries:", QUERY_IDS)
    print("Candidates/query: 20")
    print("Scales: 60%, 45%")
    print(
        "Positions:",
        ", ".join(POSITIONS),
    )
    print("Crop trials/tile: 10")
    print("Total pairs: 600")
    print(
        "Expected directions:",
        EXPECTED_POSITION,
    )

    if not VERIFIER_SCRIPT.exists():
        raise FileNotFoundError(
            VERIFIER_SCRIPT
        )

    oracle_sets = load_oracle_sets()
    roles = load_roles()

    if not args.aggregate_only:
        base = load_base_pool()

        print()
        print("Preparing spatial crop banks")
        print("----------------------------")

        for scale in SCALES:
            scale_percent = int(
                round(scale * 100)
            )

            for position in POSITIONS:
                pool_path = (
                    candidate_pool_path(
                        scale_percent,
                        position,
                    )
                )

                if (
                    args.resume
                    and pool_path.exists()
                ):
                    print(
                        f"  existing: "
                        f"s{scale_percent} "
                        f"{position}"
                    )
                else:
                    create_crop_pool(
                        base,
                        scale,
                        position,
                    )

        save_directional_crop_sheets(
            base,
            oracle_sets,
            roles,
        )

        if args.prepare_only:
            print()
            print(
                "S8.12C.1 PREPARATION COMPLETE"
            )
            print(
                "Inspect directional crop sheets "
                "before starting LightGlue."
            )
            return 0

        for scale in SCALES:
            scale_percent = int(
                round(scale * 100)
            )

            for position in POSITIONS:
                ranked_path = (
                    verifier_ranked_path(
                        scale_percent,
                        position,
                    )
                )

                if (
                    args.resume
                    and ranked_path.exists()
                ):
                    print(
                        "Skipping completed run:",
                        ranked_path,
                    )
                    continue

                run_verifier(
                    scale_percent,
                    position,
                    args,
                )

    aggregate(
        oracle_sets,
        roles,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
