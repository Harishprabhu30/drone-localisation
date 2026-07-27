#!/usr/bin/env python3
"""
S8.12C.0 — Multi-scale satellite context sweep.

Controlled variable
-------------------
Only the visible satellite footprint changes.

Frozen:
- same 3 smoke queries
- same DINOv2 Top-20 candidate pool
- same tile IDs
- same candidate ranks
- same evaluation coordinates
- same oracle definitions
- same LightGlue configuration

Scales:
- 100%
- 75%
- 60%
- 45%
- 30%

Each crop is centred in the original satellite tile. The cropped image is
stored losslessly as PNG and passed to the existing S7D.1 LightGlue verifier.
The verifier performs its normal resize-long preprocessing.

Outputs:
- per-scale candidate pools
- cropped satellite images
- crop contact sheets
- per-scale LightGlue candidate/ranking outputs
- per-query scale comparison
- aggregate scale comparison
- ranked Top-1 visual summaries
- final recommendation JSON

command to run prepare only:

1. First run preparation only, preps csv, figures

Do this before starting the 300 LightGlue pairs:

python scripts/villoc/s8_12c0_multiscale_satellite_context_sweep.py \
  --prepare-only \
  2>&1 | tee \
  outputs/villoc/90_deg/logs/s8_12c0_multiscale_prepare.log

2. Once the above is run, use this command:

python scripts/villoc/s8_12c0_multiscale_satellite_context_sweep.py \
  --device auto \
  --resize-long 512 \
  --max-keypoints 2048 \
  --ransac-thresh 5.0 \
  2>&1 | tee \
  outputs/villoc/90_deg/logs/s8_12c0_multiscale_full.log

3. IF to resume an interrupted run:

python scripts/villoc/s8_12c0_multiscale_satellite_context_sweep.py \
  --device auto \
  --resize-long 512 \
  --max-keypoints 2048 \
  --ransac-thresh 5.0 \
  --resume \
  2>&1 | tee -a \
  outputs/villoc/90_deg/logs/s8_12c0_multiscale_full.log

## Important limitation

This first experiment uses centre crops.

That assumes the UAV-relevant portion lies reasonably near the tile centre. But a geometric oracle tile only guarantees 
that the UAV coordinate lies somewhere inside the tile—not necessarily at its centre.

Therefore:

centre crop failure
≠ proof that multi-scale is wrong

A reduced crop could accidentally remove the actual GT position when the UAV lies near a tile edge.

For this controlled first sweep, that is acceptable because we are isolating simple context reduction. 
Later, a georeferenced GT-centred diagnostic crop can be used strictly as an oracle ceiling, not as an online method.

"""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path.cwd().resolve()

BASE_RUN_NAME = "s8_12b0_villoc_1024_s512_top20_smoke3"

BASE_RANKED_CSV = (
    ROOT
    / "outputs/villoc/90_deg/metadata/s7d_lightglue"
    / f"s7d1_lightglue_candidate_scores_ranked_{BASE_RUN_NAME}.csv"
)

ORACLE_CSV = (
    ROOT
    / "outputs/villoc/90_deg/metadata"
    / "s8_10b_uav_tile_oracle_1024_s512.csv"
)

SELECTION_JSON = (
    ROOT
    / "outputs/villoc/90_deg/reports/s8_12b"
    / "s8_12b0_smoke_query_selection.json"
)

VERIFIER_SCRIPT = (
    ROOT
    / "scripts/satloc/s7d"
    / "s7d_1_lightglue_merged_candidate_verifier.py"
)

OUT_BASE = ROOT / "outputs/villoc/90_deg"

META_DIR = OUT_BASE / "metadata/s8_12c_multiscale"
REPORT_DIR = OUT_BASE / "reports/s8_12c_multiscale"
FIG_DIR = OUT_BASE / "figures/s8_12c_multiscale"
CROP_ROOT = (
    ROOT
    / "data/processed/villoc/90_deg"
    / "s8_12c_multiscale_tiles"
)

S7D_META_DIR = OUT_BASE / "metadata/s7d_lightglue"
S7D_REPORT_DIR = OUT_BASE / "reports/s7d_lightglue"
S7D_PANEL_DIR = OUT_BASE / "figures/s7d_lightglue/panels"

SCALES = [1.00, 0.75, 0.60, 0.45, 0.30]
SCALE_LABELS = {
    1.00: "100",
    0.75: "75",
    0.60: "60",
    0.45: "45",
    0.30: "30",
}

QUERY_IDS = ["23", "33", "52"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="S8.12C.0 multi-scale satellite context sweep."
    )

    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Create crops and candidate pools, but do not run LightGlue.",
    )

    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Skip crop generation and matching; aggregate existing outputs.",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Pass --resume to each LightGlue verifier run.",
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
        number = float(value)
        return number if np.isfinite(number) else float("nan")
    except Exception:
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


def resolve_path(value: Any) -> Path:
    raw = Path(str(value)).expanduser()

    candidates = [
        raw,
        ROOT / raw,
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    raise FileNotFoundError(
        f"Image path could not be resolved: {value}"
    )


def find_column(
    frame: pd.DataFrame,
    candidates: Iterable[str],
    label: str,
    required: bool = True,
) -> str | None:
    lower = {
        str(column).lower(): column
        for column in frame.columns
    }

    for candidate in candidates:
        if candidate.lower() in lower:
            return lower[candidate.lower()]

    if required:
        raise KeyError(
            f"Could not infer {label}. "
            f"Tried {list(candidates)}. "
            f"Available columns: {list(frame.columns)}"
        )

    return None


def parse_tile_set(value: Any) -> set[str]:
    if value is None:
        return set()

    try:
        if pd.isna(value):
            return set()
    except Exception:
        pass

    if isinstance(value, (list, tuple, set, np.ndarray)):
        return {
            normalize_id(item)
            for item in value
            if normalize_id(item)
        }

    text = str(value).strip()

    if not text or text.lower() in {
        "nan",
        "none",
        "null",
        "[]",
    }:
        return set()

    try:
        parsed = ast.literal_eval(text)

        if isinstance(parsed, (list, tuple, set)):
            return {
                normalize_id(item)
                for item in parsed
                if normalize_id(item)
            }
    except Exception:
        pass

    cleaned = re.sub(
        r"[\[\]\(\)\{\}\"']",
        "",
        text,
    )

    return {
        normalize_id(part)
        for part in re.split(r"[,;|\s]+", cleaned)
        if normalize_id(part)
    }


def load_roles() -> dict[str, str]:
    if not SELECTION_JSON.exists():
        return {
            "23": "hard",
            "33": "middle",
            "52": "easy",
        }

    data = json.loads(
        SELECTION_JSON.read_text(encoding="utf-8")
    )

    rows = data.get("queries", data)

    if isinstance(rows, dict):
        rows = rows.get("queries", [])

    roles: dict[str, str] = {}

    for row in rows:
        query_id = normalize_id(row.get("query_id"))
        role = str(row.get("role", "unknown"))

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

    result: dict[str, set[str]] = {}

    for _, row in frame.iterrows():
        query_id = normalize_id(row[query_col])

        if not query_id:
            continue

        result.setdefault(query_id, set())

        if packed_col is not None:
            result[query_id].update(
                parse_tile_set(row[packed_col])
            )

        if single_col is not None:
            tile_id = normalize_id(row[single_col])

            if tile_id:
                result[query_id].add(tile_id)

    return result


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

    result = pd.DataFrame(
        {
            "query_id": frame[query_col].map(normalize_id),
            "tile_id": frame[tile_col].map(normalize_id),
            "candidate_rank": pd.to_numeric(
                frame[rank_col],
                errors="coerce",
            ),
            "uav_image_path": frame[uav_col].astype(str),
            "sat_image_path_original": frame[sat_col].astype(str),
            "eval_error_m": (
                pd.to_numeric(frame[eval_col], errors="coerce")
                if eval_col is not None
                else np.nan
            ),
        }
    )

    for optional in [
        "policy",
        "budget",
        "center_rank",
        "resize_rank",
        "source_count",
        "sources",
        "resize_unique",
        "merge_score",
    ]:
        if optional in frame.columns:
            result[optional] = frame[optional]

    result = result[
        result["query_id"].isin(QUERY_IDS)
    ].copy()

    result = result.sort_values(
        ["query_id", "candidate_rank", "tile_id"],
        kind="mergesort",
    )

    result = result.drop_duplicates(
        ["query_id", "tile_id"],
        keep="first",
    )

    result = (
        result.groupby(
            "query_id",
            group_keys=False,
            sort=True,
        )
        .head(20)
        .reset_index(drop=True)
    )

    counts = result.groupby("query_id").size().to_dict()

    for query_id in QUERY_IDS:
        if counts.get(query_id, 0) != 20:
            raise RuntimeError(
                f"Expected 20 candidates for query {query_id}, "
                f"found {counts.get(query_id, 0)}."
            )

    return result


def centre_crop(
    image: np.ndarray,
    fraction: float,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    if image is None or image.size == 0:
        raise ValueError("Cannot crop an empty image.")

    height, width = image.shape[:2]

    if fraction >= 0.999999:
        return image.copy(), (0, 0, width, height)

    crop_width = max(
        32,
        min(width, int(round(width * fraction))),
    )

    crop_height = max(
        32,
        min(height, int(round(height * fraction))),
    )

    x0 = max(0, (width - crop_width) // 2)
    y0 = max(0, (height - crop_height) // 2)

    x1 = min(width, x0 + crop_width)
    y1 = min(height, y0 + crop_height)

    cropped = image[y0:y1, x0:x1].copy()

    return cropped, (x0, y0, x1, y1)


def crop_output_path(
    scale_label: str,
    query_id: str,
    tile_id: str,
) -> Path:
    return (
        CROP_ROOT
        / f"scale_{scale_label}"
        / f"query_{query_id}"
        / f"tile_{tile_id}_crop_{scale_label}.png"
    )


def create_scaled_pool(
    base: pd.DataFrame,
    scale: float,
) -> pd.DataFrame:
    scale_label = SCALE_LABELS[scale]

    rows: list[dict[str, Any]] = []

    for index, row in base.iterrows():
        query_id = normalize_id(row["query_id"])
        tile_id = normalize_id(row["tile_id"])

        source_path = resolve_path(
            row["sat_image_path_original"]
        )

        image = cv2.imread(
            str(source_path),
            cv2.IMREAD_COLOR,
        )

        if image is None:
            raise RuntimeError(
                f"OpenCV failed to read: {source_path}"
            )

        crop, bounds = centre_crop(image, scale)

        output_path = crop_output_path(
            scale_label,
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
            [cv2.IMWRITE_PNG_COMPRESSION, 3],
        )

        if not success:
            raise RuntimeError(
                f"Failed to save crop: {output_path}"
            )

        output = row.to_dict()

        output.update(
            {
                "sat_image_path": str(output_path),
                "context_scale_fraction": float(scale),
                "context_scale_percent": int(
                    round(scale * 100)
                ),
                "crop_x0_px": int(bounds[0]),
                "crop_y0_px": int(bounds[1]),
                "crop_x1_px": int(bounds[2]),
                "crop_y1_px": int(bounds[3]),
                "crop_width_px": int(crop.shape[1]),
                "crop_height_px": int(crop.shape[0]),
                "original_width_px": int(image.shape[1]),
                "original_height_px": int(image.shape[0]),
            }
        )

        rows.append(output)

        if (index + 1) % 20 == 0:
            print(
                f"  scale {scale_label}%: "
                f"prepared {index + 1}/{len(base)}"
            )

    result = pd.DataFrame(rows)

    pool_path = (
        META_DIR
        / f"s8_12c0_candidate_pool_scale_{scale_label}.csv"
    )

    result.to_csv(pool_path, index=False)

    return result


def save_crop_contact_sheets(
    base: pd.DataFrame,
    oracle_sets: dict[str, set[str]],
    roles: dict[str, str],
) -> None:
    for query_id in QUERY_IDS:
        group = base[
            base["query_id"] == query_id
        ].sort_values("candidate_rank")

        oracle_group = group[
            group["tile_id"].isin(
                oracle_sets.get(query_id, set())
            )
        ]

        if not oracle_group.empty:
            representative = oracle_group.iloc[0]
            tile_role = "first DINO oracle"
        else:
            representative = group.iloc[0]
            tile_role = "DINO Top-1"

        tile_id = normalize_id(
            representative["tile_id"]
        )

        uav_path = resolve_path(
            representative["uav_image_path"]
        )

        uav_bgr = cv2.imread(
            str(uav_path),
            cv2.IMREAD_COLOR,
        )

        if uav_bgr is None:
            raise RuntimeError(
                f"Failed to read UAV image: {uav_path}"
            )

        uav_rgb = cv2.cvtColor(
            uav_bgr,
            cv2.COLOR_BGR2RGB,
        )

        fig, axes = plt.subplots(
            2,
            3,
            figsize=(15, 9),
        )

        axes = np.asarray(axes).reshape(-1)

        axes[0].imshow(uav_rgb)
        axes[0].set_title(
            f"UAV query {query_id}\n"
            f"role={roles.get(query_id, 'unknown')}"
        )
        axes[0].axis("off")

        original_path = resolve_path(
            representative["sat_image_path_original"]
        )

        original_bgr = cv2.imread(
            str(original_path),
            cv2.IMREAD_COLOR,
        )

        for axis_index, scale in enumerate(
            SCALES,
            start=1,
        ):
            scale_label = SCALE_LABELS[scale]

            crop, bounds = centre_crop(
                original_bgr,
                scale,
            )

            crop_rgb = cv2.cvtColor(
                crop,
                cv2.COLOR_BGR2RGB,
            )

            axes[axis_index].imshow(crop_rgb)
            axes[axis_index].set_title(
                f"{scale_label}% context\n"
                f"{crop.shape[1]}×{crop.shape[0]} px"
            )
            axes[axis_index].axis("off")

        fig.suptitle(
            "S8.12C.0 — Satellite context reduction\n"
            f"query={query_id}, tile={tile_id}, "
            f"{tile_role}, DINO rank="
            f"{int(representative['candidate_rank'])}",
            fontsize=13,
        )

        fig.tight_layout(
            rect=[0, 0, 1, 0.93]
        )

        output_path = (
            FIG_DIR
            / f"s8_12c0_query_{query_id}"
            f"_satellite_context_contact_sheet.png"
        )

        fig.savefig(
            output_path,
            dpi=180,
        )

        plt.close(fig)


def verifier_run_name(scale_label: str) -> str:
    return (
        f"s8_12c0_villoc_1024_s512_top20_smoke3"
        f"_context{scale_label}"
    )


def run_verifier(
    scale_label: str,
    args: argparse.Namespace,
) -> None:
    pool_path = (
        META_DIR
        / f"s8_12c0_candidate_pool_scale_{scale_label}.csv"
    )

    run_name = verifier_run_name(scale_label)

    command = [
        sys.executable,
        str(VERIFIER_SCRIPT),
        "--candidate-pool",
        str(pool_path),
        "--out-base",
        str(OUT_BASE),
        "--run-name",
        run_name,
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
        f"Running LightGlue scale {scale_label}%"
    )
    print("------------------------------------")
    print(" ".join(command))

    subprocess.run(
        command,
        cwd=ROOT,
        check=True,
    )


def ranked_output_path(scale_label: str) -> Path:
    run_name = verifier_run_name(scale_label)

    return (
        S7D_META_DIR
        / f"s7d1_lightglue_candidate_scores_ranked_"
        f"{run_name}.csv"
    )


def candidate_output_path(scale_label: str) -> Path:
    run_name = verifier_run_name(scale_label)

    return (
        S7D_META_DIR
        / f"s7d1_lightglue_candidate_scores_"
        f"{run_name}.csv"
    )


def load_ranked_scale(
    scale: float,
    oracle_sets: dict[str, set[str]],
    roles: dict[str, str],
) -> pd.DataFrame:
    scale_label = SCALE_LABELS[scale]
    path = ranked_output_path(scale_label)

    if not path.exists():
        raise FileNotFoundError(
            f"Missing ranked output for scale "
            f"{scale_label}%: {path}"
        )

    frame = pd.read_csv(path)

    frame["query_id"] = frame[
        find_column(
            frame,
            ["query_id", "token0_id"],
            "ranked query ID",
        )
    ].map(normalize_id)

    frame["tile_id"] = frame[
        find_column(
            frame,
            ["tile_id", "sat_tile_id"],
            "ranked tile ID",
        )
    ].map(normalize_id)

    frame["candidate_rank_num"] = pd.to_numeric(
        frame[
            find_column(
                frame,
                [
                    "candidate_rank",
                    "retrieval_rank",
                    "rank",
                ],
                "candidate rank",
            )
        ],
        errors="coerce",
    )

    frame["lightglue_rank_num"] = pd.to_numeric(
        frame[
            find_column(
                frame,
                ["lightglue_rank"],
                "LightGlue rank",
            )
        ],
        errors="coerce",
    )

    for target, candidates in {
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
        ],
        "runtime_num": [
            "runtime_s",
        ],
    }.items():
        source = find_column(
            frame,
            candidates,
            target,
            required=False,
        )

        frame[target] = (
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
        frame[homography_col].map(truthy)
        if homography_col is not None
        else False
    )

    frame["min_coverage_num"] = np.minimum(
        frame["uav_coverage_num"].fillna(0),
        frame["sat_coverage_num"].fillna(0),
    )

    frame["is_geometric_oracle"] = [
        tile_id in oracle_sets.get(query_id, set())
        for query_id, tile_id in zip(
            frame["query_id"],
            frame["tile_id"],
        )
    ]

    frame["role"] = (
        frame["query_id"]
        .map(roles)
        .fillna("unknown")
    )

    frame["context_scale_fraction"] = scale
    frame["context_scale_percent"] = int(
        round(scale * 100)
    )

    return frame


def build_query_scale_summary(
    all_ranked: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for (
        scale_percent,
        query_id,
    ), group in all_ranked.groupby(
        [
            "context_scale_percent",
            "query_id",
        ],
        sort=True,
    ):
        g = group.copy()

        top = g.sort_values(
            [
                "lightglue_rank_num",
                "candidate_rank_num",
            ],
            kind="mergesort",
        ).iloc[0]

        oracle = g[
            g["is_geometric_oracle"]
        ].copy()

        best_oracle = (
            oracle.sort_values(
                [
                    "lightglue_rank_num",
                    "candidate_rank_num",
                ],
                kind="mergesort",
            ).iloc[0]
            if not oracle.empty
            else None
        )

        nonoracle = g[
            ~g["is_geometric_oracle"]
        ].copy()

        best_nonoracle = (
            nonoracle.sort_values(
                [
                    "lightglue_rank_num",
                    "candidate_rank_num",
                ],
                kind="mergesort",
            ).iloc[0]
            if not nonoracle.empty
            else None
        )

        top_passes_gate = bool(
            top["homography_success"]
            and top["inliers_num"] >= 10
            and top["inlier_ratio_num"] >= 0.15
            and top["min_coverage_num"] >= 0.10
        )

        oracle_score = (
            safe_float(
                best_oracle["lightglue_score_num"]
            )
            if best_oracle is not None
            else float("nan")
        )

        nonoracle_score = (
            safe_float(
                best_nonoracle["lightglue_score_num"]
            )
            if best_nonoracle is not None
            else float("nan")
        )

        rows.append(
            {
                "role": str(top["role"]),
                "query_id": query_id,
                "context_scale_percent": int(
                    scale_percent
                ),
                "processed_candidates": int(len(g)),
                "lightglue_top_tile_id": str(
                    top["tile_id"]
                ),
                "lightglue_top_candidate_rank": safe_float(
                    top["candidate_rank_num"]
                ),
                "lightglue_top_is_oracle": bool(
                    top["is_geometric_oracle"]
                ),
                "lightglue_top_eval_error_m": safe_float(
                    top["eval_error_num"]
                ),
                "lightglue_top_score": safe_float(
                    top["lightglue_score_num"]
                ),
                "lightglue_top_matches": safe_float(
                    top["matches_num"]
                ),
                "lightglue_top_inliers": safe_float(
                    top["inliers_num"]
                ),
                "lightglue_top_inlier_ratio": safe_float(
                    top["inlier_ratio_num"]
                ),
                "lightglue_top_min_coverage": safe_float(
                    top["min_coverage_num"]
                ),
                "lightglue_top_geometry_gate": top_passes_gate,
                "best_oracle_tile_id": (
                    str(best_oracle["tile_id"])
                    if best_oracle is not None
                    else ""
                ),
                "best_oracle_candidate_rank": (
                    safe_float(
                        best_oracle[
                            "candidate_rank_num"
                        ]
                    )
                    if best_oracle is not None
                    else float("nan")
                ),
                "best_oracle_lightglue_rank": (
                    safe_float(
                        best_oracle[
                            "lightglue_rank_num"
                        ]
                    )
                    if best_oracle is not None
                    else float("nan")
                ),
                "best_oracle_score": oracle_score,
                "best_oracle_inliers": (
                    safe_float(
                        best_oracle["inliers_num"]
                    )
                    if best_oracle is not None
                    else float("nan")
                ),
                "best_oracle_inlier_ratio": (
                    safe_float(
                        best_oracle[
                            "inlier_ratio_num"
                        ]
                    )
                    if best_oracle is not None
                    else float("nan")
                ),
                "best_oracle_min_coverage": (
                    safe_float(
                        best_oracle[
                            "min_coverage_num"
                        ]
                    )
                    if best_oracle is not None
                    else float("nan")
                ),
                "best_nonoracle_score": nonoracle_score,
                "oracle_score_margin": (
                    oracle_score - nonoracle_score
                    if (
                        np.isfinite(oracle_score)
                        and np.isfinite(nonoracle_score)
                    )
                    else float("nan")
                ),
                "runtime_total_s": float(
                    pd.to_numeric(
                        g["runtime_num"],
                        errors="coerce",
                    ).sum()
                ),
            }
        )

    return pd.DataFrame(rows)


def build_scale_summary(
    query_summary: pd.DataFrame,
    all_ranked: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for scale_percent, group in query_summary.groupby(
        "context_scale_percent",
        sort=True,
    ):
        candidates = all_ranked[
            all_ranked[
                "context_scale_percent"
            ] == scale_percent
        ]

        oracle_candidates = candidates[
            candidates["is_geometric_oracle"]
        ]

        nonoracle_candidates = candidates[
            ~candidates["is_geometric_oracle"]
        ]

        rows.append(
            {
                "context_scale_percent": int(
                    scale_percent
                ),
                "queries": int(len(group)),
                "pairs": int(len(candidates)),
                "top1_oracle_hits": int(
                    group[
                        "lightglue_top_is_oracle"
                    ].sum()
                ),
                "top1_oracle_rate": float(
                    group[
                        "lightglue_top_is_oracle"
                    ].mean()
                ),
                "top1_geometry_gate_passes": int(
                    group[
                        "lightglue_top_geometry_gate"
                    ].sum()
                ),
                "median_best_oracle_lg_rank": float(
                    pd.to_numeric(
                        group[
                            "best_oracle_lightglue_rank"
                        ],
                        errors="coerce",
                    ).median()
                ),
                "mean_best_oracle_lg_rank": float(
                    pd.to_numeric(
                        group[
                            "best_oracle_lightglue_rank"
                        ],
                        errors="coerce",
                    ).mean()
                ),
                "median_oracle_score_margin": float(
                    pd.to_numeric(
                        group["oracle_score_margin"],
                        errors="coerce",
                    ).median()
                ),
                "median_oracle_inliers": float(
                    pd.to_numeric(
                        oracle_candidates[
                            "inliers_num"
                        ],
                        errors="coerce",
                    ).median()
                ),
                "median_nonoracle_inliers": float(
                    pd.to_numeric(
                        nonoracle_candidates[
                            "inliers_num"
                        ],
                        errors="coerce",
                    ).median()
                ),
                "median_oracle_score": float(
                    pd.to_numeric(
                        oracle_candidates[
                            "lightglue_score_num"
                        ],
                        errors="coerce",
                    ).median()
                ),
                "median_nonoracle_score": float(
                    pd.to_numeric(
                        nonoracle_candidates[
                            "lightglue_score_num"
                        ],
                        errors="coerce",
                    ).median()
                ),
                "runtime_total_s": float(
                    pd.to_numeric(
                        candidates["runtime_num"],
                        errors="coerce",
                    ).sum()
                ),
                "runtime_median_pair_s": float(
                    pd.to_numeric(
                        candidates["runtime_num"],
                        errors="coerce",
                    ).median()
                ),
            }
        )

    summary = pd.DataFrame(rows)

    # Higher is better, except oracle rank.
    summary["selection_score"] = (
        100.0 * summary["top1_oracle_rate"]
        + 10.0 * summary[
            "top1_geometry_gate_passes"
        ]
        + 2.0 * summary[
            "median_oracle_score_margin"
        ].fillna(-10.0)
        - summary[
            "median_best_oracle_lg_rank"
        ].fillna(100.0)
    )

    return summary.sort_values(
        [
            "selection_score",
            "top1_oracle_rate",
            "median_best_oracle_lg_rank",
        ],
        ascending=[False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)


def save_scale_metric_figures(
    query_summary: pd.DataFrame,
    scale_summary: pd.DataFrame,
) -> None:
    ordered = scale_summary.sort_values(
        "context_scale_percent",
        ascending=False,
    )

    plt.figure(figsize=(9, 5.5))
    plt.plot(
        ordered["context_scale_percent"],
        ordered["top1_oracle_rate"],
        marker="o",
    )
    plt.xlabel("Satellite context retained (%)")
    plt.ylabel("LightGlue Top-1 oracle rate")
    plt.ylim(-0.05, 1.05)
    plt.title(
        "S8.12C.0 — Top-1 oracle success versus context"
    )
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        FIG_DIR
        / "s8_12c0_top1_oracle_rate_vs_context.png",
        dpi=180,
    )
    plt.close()

    plt.figure(figsize=(9, 5.5))

    for query_id, group in query_summary.groupby(
        "query_id",
        sort=True,
    ):
        ordered_query = group.sort_values(
            "context_scale_percent",
            ascending=False,
        )

        plt.plot(
            ordered_query[
                "context_scale_percent"
            ],
            ordered_query[
                "best_oracle_lightglue_rank"
            ],
            marker="o",
            label=f"Query {query_id}",
        )

    plt.gca().invert_yaxis()
    plt.xlabel("Satellite context retained (%)")
    plt.ylabel(
        "Best geometric-oracle LightGlue rank"
    )
    plt.title(
        "S8.12C.0 — Oracle rank across satellite scales"
    )
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        FIG_DIR
        / "s8_12c0_oracle_rank_vs_context.png",
        dpi=180,
    )
    plt.close()

    plt.figure(figsize=(9, 5.5))

    plt.plot(
        ordered["context_scale_percent"],
        ordered["median_oracle_inliers"],
        marker="o",
        label="Oracle candidates",
    )

    plt.plot(
        ordered["context_scale_percent"],
        ordered["median_nonoracle_inliers"],
        marker="o",
        label="Non-oracle candidates",
    )

    plt.xlabel("Satellite context retained (%)")
    plt.ylabel("Median RANSAC inliers")
    plt.title(
        "S8.12C.0 — Geometric support across scales"
    )
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        FIG_DIR
        / "s8_12c0_oracle_nonoracle_inliers_vs_context.png",
        dpi=180,
    )
    plt.close()

    plt.figure(figsize=(9, 5.5))

    for query_id, group in query_summary.groupby(
        "query_id",
        sort=True,
    ):
        ordered_query = group.sort_values(
            "context_scale_percent",
            ascending=False,
        )

        plt.plot(
            ordered_query[
                "context_scale_percent"
            ],
            ordered_query[
                "oracle_score_margin"
            ],
            marker="o",
            label=f"Query {query_id}",
        )

    plt.axhline(0.0, linestyle="--")
    plt.xlabel("Satellite context retained (%)")
    plt.ylabel(
        "Best oracle score − best non-oracle score"
    )
    plt.title(
        "S8.12C.0 — Oracle score separation"
    )
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        FIG_DIR
        / "s8_12c0_oracle_score_margin_vs_context.png",
        dpi=180,
    )
    plt.close()


def load_rgb(path_value: Any) -> np.ndarray | None:
    try:
        path = resolve_path(path_value)
    except Exception:
        return None

    image = cv2.imread(
        str(path),
        cv2.IMREAD_COLOR,
    )

    if image is None:
        return None

    return cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB,
    )


def save_ranked_top1_contact_sheets(
    query_summary: pd.DataFrame,
    all_ranked: pd.DataFrame,
    roles: dict[str, str],
) -> None:
    for query_id in QUERY_IDS:
        summary = query_summary[
            query_summary["query_id"] == query_id
        ].sort_values(
            "context_scale_percent",
            ascending=False,
        )

        if summary.empty:
            continue

        query_rows = all_ranked[
            all_ranked["query_id"] == query_id
        ]

        first = query_rows.iloc[0]
        uav_rgb = load_rgb(
            first["uav_image_path"]
        )

        fig, axes = plt.subplots(
            2,
            3,
            figsize=(16, 10),
        )

        axes = np.asarray(axes).reshape(-1)

        axes[0].axis("off")

        if uav_rgb is not None:
            axes[0].imshow(uav_rgb)

        axes[0].set_title(
            f"UAV query {query_id}\n"
            f"role={roles.get(query_id, 'unknown')}"
        )

        for axis_index, (_, row) in enumerate(
            summary.iterrows(),
            start=1,
        ):
            scale_percent = int(
                row["context_scale_percent"]
            )

            ranked_group = query_rows[
                query_rows[
                    "context_scale_percent"
                ] == scale_percent
            ]

            top = ranked_group.sort_values(
                "lightglue_rank_num",
                kind="mergesort",
            ).iloc[0]

            sat_rgb = load_rgb(
                top["sat_image_path"]
            )

            axes[axis_index].axis("off")

            if sat_rgb is not None:
                axes[axis_index].imshow(sat_rgb)

            oracle_label = (
                "ORACLE"
                if bool(
                    row["lightglue_top_is_oracle"]
                )
                else "NON-ORACLE"
            )

            axes[axis_index].set_title(
                f"{scale_percent}% — {oracle_label}\n"
                f"tile={row['lightglue_top_tile_id']} "
                f"DINO-r={row['lightglue_top_candidate_rank']:.0f}\n"
                f"score={row['lightglue_top_score']:.2f}, "
                f"inliers={row['lightglue_top_inliers']:.0f}, "
                f"ratio={row['lightglue_top_inlier_ratio']:.2f}\n"
                f"oracle LG rank="
                f"{row['best_oracle_lightglue_rank']:.0f}"
            )

        fig.suptitle(
            "S8.12C.0 — LightGlue Top-1 candidate "
            "at each satellite context scale",
            fontsize=13,
        )

        fig.tight_layout(
            rect=[0, 0, 1, 0.94]
        )

        output_path = (
            FIG_DIR
            / f"s8_12c0_query_{query_id}"
            f"_ranked_top1_scale_comparison.png"
        )

        fig.savefig(
            output_path,
            dpi=180,
        )

        plt.close(fig)


def copy_panel_inventory() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for scale in SCALES:
        scale_label = SCALE_LABELS[scale]
        run_name = verifier_run_name(scale_label)

        for query_id in QUERY_IDS:
            patterns = [
                f"*{run_name}*{query_id}*.png",
                f"*{query_id}*{run_name}*.png",
            ]

            found: list[Path] = []

            for pattern in patterns:
                found.extend(
                    S7D_PANEL_DIR.glob(pattern)
                )

            found = sorted(set(found))

            rows.append(
                {
                    "context_scale_percent": int(
                        round(scale * 100)
                    ),
                    "query_id": query_id,
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
        / "s8_12c0_lightglue_match_panel_inventory.csv",
        index=False,
    )

    return inventory


def aggregate(
    oracle_sets: dict[str, set[str]],
    roles: dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ranked_frames: list[pd.DataFrame] = []

    for scale in SCALES:
        ranked_frames.append(
            load_ranked_scale(
                scale,
                oracle_sets,
                roles,
            )
        )

    all_ranked = pd.concat(
        ranked_frames,
        ignore_index=True,
    )

    all_ranked.to_csv(
        META_DIR
        / "s8_12c0_all_scales_ranked_candidates.csv",
        index=False,
    )

    query_summary = build_query_scale_summary(
        all_ranked
    )

    query_summary.to_csv(
        REPORT_DIR
        / "s8_12c0_query_scale_summary.csv",
        index=False,
    )

    scale_summary = build_scale_summary(
        query_summary,
        all_ranked,
    )

    scale_summary.to_csv(
        REPORT_DIR
        / "s8_12c0_scale_summary.csv",
        index=False,
    )

    save_scale_metric_figures(
        query_summary,
        scale_summary,
    )

    save_ranked_top1_contact_sheets(
        query_summary,
        all_ranked,
        roles,
    )

    panel_inventory = copy_panel_inventory()

    winner = scale_summary.iloc[0]

    baseline = scale_summary[
        scale_summary[
            "context_scale_percent"
        ] == 100
    ].iloc[0]

    winning_scale = int(
        winner["context_scale_percent"]
    )

    baseline_rate = float(
        baseline["top1_oracle_rate"]
    )

    winning_rate = float(
        winner["top1_oracle_rate"]
    )

    winning_rank = float(
        winner["median_best_oracle_lg_rank"]
    )

    baseline_rank = float(
        baseline["median_best_oracle_lg_rank"]
    )

    materially_better = bool(
        winning_scale != 100
        and (
            winning_rate > baseline_rate
            or winning_rank < baseline_rank
            or (
                winner["median_oracle_score_margin"]
                >
                baseline[
                    "median_oracle_score_margin"
                ]
                + 1.0
            )
        )
    )

    if materially_better:
        status = (
            "PASS_MULTISCALE_IMPROVEMENT_"
            "FREEZE_BEST_CONTEXT"
        )
        decision = (
            f"Freeze {winning_scale}% satellite context "
            "for the next verifier benchmark."
        )
    else:
        status = (
            "PASS_MULTISCALE_NO_CLEAR_IMPROVEMENT_"
            "PROCEED_PHOG_LSD"
        )
        decision = (
            "No reduced context scale clearly outperformed "
            "the 100% baseline. Proceed to PHOG+LSD "
            "structural reranking."
        )

    report = {
        "stage": "S8.12C.0",
        "title": (
            "Multi-scale satellite context sweep"
        ),
        "controlled_variable": (
            "Central satellite crop fraction only"
        ),
        "scales_percent": [
            int(round(scale * 100))
            for scale in SCALES
        ],
        "queries": QUERY_IDS,
        "queries_count": len(QUERY_IDS),
        "candidates_per_query": 20,
        "pairs_per_scale": 60,
        "total_pairs": 300,
        "winning_scale_percent": winning_scale,
        "baseline_scale_percent": 100,
        "winning_top1_oracle_rate": winning_rate,
        "baseline_top1_oracle_rate": baseline_rate,
        "winning_median_best_oracle_rank": (
            winning_rank
        ),
        "baseline_median_best_oracle_rank": (
            baseline_rank
        ),
        "status": status,
        "decision": decision,
        "important_note": (
            "Tile IDs, oracle membership and evaluation "
            "coordinates remain unchanged. This experiment "
            "tests visual footprint only."
        ),
        "outputs": {
            "all_ranked_candidates": str(
                META_DIR
                / "s8_12c0_all_scales_ranked_candidates.csv"
            ),
            "query_scale_summary": str(
                REPORT_DIR
                / "s8_12c0_query_scale_summary.csv"
            ),
            "scale_summary": str(
                REPORT_DIR
                / "s8_12c0_scale_summary.csv"
            ),
            "match_panel_inventory": str(
                REPORT_DIR
                / "s8_12c0_lightglue_match_panel_inventory.csv"
            ),
            "figure_directory": str(FIG_DIR),
            "crop_directory": str(CROP_ROOT),
        },
    }

    report_path = (
        REPORT_DIR
        / "s8_12c0_multiscale_context_summary.json"
    )

    report_path.write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("Query-scale summary")
    print("-------------------")

    display_columns = [
        "role",
        "query_id",
        "context_scale_percent",
        "lightglue_top_is_oracle",
        "lightglue_top_candidate_rank",
        "lightglue_top_score",
        "lightglue_top_inliers",
        "lightglue_top_inlier_ratio",
        "lightglue_top_min_coverage",
        "best_oracle_lightglue_rank",
        "best_oracle_inliers",
        "oracle_score_margin",
    ]

    print(
        query_summary[
            display_columns
        ].sort_values(
            [
                "query_id",
                "context_scale_percent",
            ],
            ascending=[True, False],
        ).to_string(index=False)
    )

    print()
    print("Scale summary")
    print("-------------")

    print(
        scale_summary.to_string(index=False)
    )

    print()
    print("----------------------------------------")
    print("S8.12C.0 COMPLETE")
    print("STATUS:", status)
    print("WINNING SCALE:", f"{winning_scale}%")
    print("DECISION:", decision)
    print("Report:", report_path)

    return query_summary, scale_summary


def main() -> int:
    args = parse_args()

    ensure_directories()

    print(
        "S8.12C.0 — Multi-Scale Satellite Context Sweep"
    )
    print(
        "----------------------------------------------"
    )
    print("Queries:", QUERY_IDS)
    print("Candidates/query: 20")
    print(
        "Scales:",
        ", ".join(
            f"{int(scale * 100)}%"
            for scale in SCALES
        ),
    )
    print("Total LightGlue pairs: 300")
    print(
        "Controlled variable: satellite visible footprint"
    )

    if not VERIFIER_SCRIPT.exists():
        raise FileNotFoundError(
            f"Missing verifier: {VERIFIER_SCRIPT}"
        )

    roles = load_roles()
    oracle_sets = load_oracle_sets()

    if not args.aggregate_only:
        base = load_base_pool()

        print()
        print("Preparing centre-cropped satellite pools")
        print("----------------------------------------")

        for scale in SCALES:
            scale_label = SCALE_LABELS[scale]

            pool_path = (
                META_DIR
                / f"s8_12c0_candidate_pool_scale_"
                f"{scale_label}.csv"
            )

            if (
                args.resume
                and pool_path.exists()
            ):
                print(
                    f"  scale {scale_label}%: "
                    "candidate pool already exists"
                )
            else:
                create_scaled_pool(
                    base,
                    scale,
                )

        save_crop_contact_sheets(
            base,
            oracle_sets,
            roles,
        )

        if args.prepare_only:
            print()
            print(
                "S8.12C.0 PREPARATION COMPLETE"
            )
            print(
                "LightGlue was not started "
                "because --prepare-only was used."
            )
            return 0

        for scale in SCALES:
            scale_label = SCALE_LABELS[scale]

            ranked_path = ranked_output_path(
                scale_label
            )

            if (
                args.resume
                and ranked_path.exists()
            ):
                print(
                    f"Skipping completed scale "
                    f"{scale_label}%: {ranked_path}"
                )
                continue

            run_verifier(
                scale_label,
                args,
            )

    aggregate(
        oracle_sets,
        roles,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
