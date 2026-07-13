from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd


INDEX_PATH = Path(
    "outputs/satloc/metadata/uav_frames_index_enriched.csv"
)
SEQUENCE = "traj01"

OUTPUT_CSV = Path(
    "outputs/satloc/metadata/s6a_relative_motion/"
    "s6a0b_sequence_order_candidates.csv"
)
OUTPUT_JSON = Path(
    "outputs/satloc/reports/s6a_relative_motion/"
    "s6a0b_sequence_order_audit.json"
)

ORDER_CANDIDATES = [
    "token0_id",
    "frame_index_in_sequence",
    "global_frame_index",
    "token1_order",
]

MAX_VISUAL_PAIRS = 80
ORB_FEATURES = 1200
RATIO_THRESHOLD = 0.75
RANSAC_THRESHOLD_PX = 3.0
RESIZE_LONG_SIDE = 640


def resolve_image_path(value: Any) -> Path:
    path = Path(str(value)).expanduser()

    candidates = [
        path,
        Path.cwd() / path,
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    return path


def resize_long_side(
    image: np.ndarray,
    target_long_side: int,
) -> np.ndarray:
    height, width = image.shape[:2]
    long_side = max(height, width)

    if long_side <= target_long_side:
        return image

    scale = target_long_side / float(long_side)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))

    return cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA,
    )


def read_gray(path: Path) -> np.ndarray | None:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        return None

    return resize_long_side(image, RESIZE_LONG_SIDE)


def evaluate_visual_pair(
    path_a: Path,
    path_b: Path,
    orb: cv2.ORB,
    matcher: cv2.BFMatcher,
) -> dict[str, Any]:
    image_a = read_gray(path_a)
    image_b = read_gray(path_b)

    if image_a is None or image_b is None:
        return {
            "read_ok": False,
            "good_matches": 0,
            "inliers": 0,
            "inlier_ratio": 0.0,
            "affine_ok": False,
        }

    keypoints_a, descriptors_a = orb.detectAndCompute(image_a, None)
    keypoints_b, descriptors_b = orb.detectAndCompute(image_b, None)

    if (
        descriptors_a is None
        or descriptors_b is None
        or len(keypoints_a) < 3
        or len(keypoints_b) < 3
    ):
        return {
            "read_ok": True,
            "good_matches": 0,
            "inliers": 0,
            "inlier_ratio": 0.0,
            "affine_ok": False,
        }

    knn_matches = matcher.knnMatch(
        descriptors_a,
        descriptors_b,
        k=2,
    )

    good_matches = []
    for pair in knn_matches:
        if len(pair) != 2:
            continue

        best, second = pair
        if best.distance < RATIO_THRESHOLD * second.distance:
            good_matches.append(best)

    if len(good_matches) < 3:
        return {
            "read_ok": True,
            "good_matches": len(good_matches),
            "inliers": 0,
            "inlier_ratio": 0.0,
            "affine_ok": False,
        }

    points_a = np.float32(
        [keypoints_a[match.queryIdx].pt for match in good_matches]
    )
    points_b = np.float32(
        [keypoints_b[match.trainIdx].pt for match in good_matches]
    )

    affine, inlier_mask = cv2.estimateAffinePartial2D(
        points_a,
        points_b,
        method=cv2.RANSAC,
        ransacReprojThreshold=RANSAC_THRESHOLD_PX,
        maxIters=2000,
        confidence=0.995,
        refineIters=10,
    )

    if affine is None or inlier_mask is None:
        return {
            "read_ok": True,
            "good_matches": len(good_matches),
            "inliers": 0,
            "inlier_ratio": 0.0,
            "affine_ok": False,
        }

    inliers = int(np.asarray(inlier_mask).ravel().sum())

    return {
        "read_ok": True,
        "good_matches": len(good_matches),
        "inliers": inliers,
        "inlier_ratio": inliers / max(len(good_matches), 1),
        "affine_ok": True,
    }


def safe_stat(
    values: np.ndarray,
    function,
    default: float = float("nan"),
) -> float:
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return default
    return float(function(finite))


def audit_order_candidate(
    traj: pd.DataFrame,
    order_column: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    work = traj.copy()

    work["_candidate_order"] = pd.to_numeric(
        work[order_column],
        errors="coerce",
    )

    invalid_order_values = int(work["_candidate_order"].isna().sum())
    work = work.dropna(subset=["_candidate_order"])

    # Stable sort makes duplicate handling reproducible.
    work = work.sort_values(
        ["_candidate_order", "filename"],
        kind="mergesort",
    ).reset_index(drop=True)

    order_values = work["_candidate_order"].to_numpy(dtype=float)
    order_differences = np.diff(order_values)

    duplicate_orders = int(np.sum(order_differences == 0))
    unit_steps = int(np.sum(order_differences == 1))
    nonunit_positive_steps = int(np.sum(order_differences > 1))

    result: dict[str, Any] = {
        "order_column": order_column,
        "rows": int(len(work)),
        "invalid_order_values": invalid_order_values,
        "unique_order_values": int(
            work["_candidate_order"].nunique()
        ),
        "duplicate_adjacent_orders": duplicate_orders,
        "unit_order_steps": unit_steps,
        "nonunit_positive_steps": nonunit_positive_steps,
        "first_order": (
            float(order_values[0]) if len(order_values) else None
        ),
        "last_order": (
            float(order_values[-1]) if len(order_values) else None
        ),
    }

    # Reference values are used only for dataset-order validation.
    if {"x_enu_m", "y_enu_m"}.issubset(work.columns):
        x = pd.to_numeric(
            work["x_enu_m"],
            errors="coerce",
        ).to_numpy(dtype=float)

        y = pd.to_numeric(
            work["y_enu_m"],
            errors="coerce",
        ).to_numpy(dtype=float)

        reference_steps = np.hypot(
            np.diff(x),
            np.diff(y),
        )

        result.update(
            {
                "reference_total_path_m": safe_stat(
                    reference_steps,
                    np.sum,
                ),
                "reference_median_step_m": safe_stat(
                    reference_steps,
                    np.median,
                ),
                "reference_p95_step_m": safe_stat(
                    reference_steps,
                    lambda values: np.percentile(values, 95),
                ),
                "reference_max_step_m": safe_stat(
                    reference_steps,
                    np.max,
                ),
                "reference_zero_steps": int(
                    np.sum(
                        np.isfinite(reference_steps)
                        & (reference_steps == 0)
                    )
                ),
            }
        )

    token0_values = pd.to_numeric(
        work["token0_id"],
        errors="coerce",
    )

    result["first_15_token0_ids"] = [
        int(value)
        for value in token0_values.head(15)
        if pd.notna(value)
    ]

    result["first_5_filenames"] = (
        work["filename"].head(5).astype(str).tolist()
    )

    pair_count = max(len(work) - 1, 0)

    if pair_count == 0:
        return result, []

    sample_count = min(MAX_VISUAL_PAIRS, pair_count)
    pair_indices = np.linspace(
        0,
        pair_count - 1,
        num=sample_count,
        dtype=int,
    )
    pair_indices = np.unique(pair_indices)

    orb = cv2.ORB_create(
        nfeatures=ORB_FEATURES,
        scaleFactor=1.2,
        nlevels=8,
        edgeThreshold=31,
        patchSize=31,
        fastThreshold=20,
    )
    matcher = cv2.BFMatcher(
        cv2.NORM_HAMMING,
        crossCheck=False,
    )

    pair_rows: list[dict[str, Any]] = []

    for pair_index in pair_indices:
        row_a = work.iloc[int(pair_index)]
        row_b = work.iloc[int(pair_index) + 1]

        path_a = resolve_image_path(row_a["image_path"])
        path_b = resolve_image_path(row_b["image_path"])

        pair_result = evaluate_visual_pair(
            path_a,
            path_b,
            orb,
            matcher,
        )

        pair_rows.append(
            {
                "order_column": order_column,
                "pair_index": int(pair_index),
                "token0_a": int(row_a["token0_id"]),
                "token0_b": int(row_b["token0_id"]),
                "order_a": float(row_a["_candidate_order"]),
                "order_b": float(row_b["_candidate_order"]),
                **pair_result,
            }
        )

    pair_df = pd.DataFrame(pair_rows)

    good_matches = pair_df["good_matches"].to_numpy(dtype=float)
    inliers = pair_df["inliers"].to_numpy(dtype=float)
    inlier_ratios = pair_df["inlier_ratio"].to_numpy(dtype=float)

    result.update(
        {
            "visual_pairs_sampled": int(len(pair_df)),
            "visual_affine_successes": int(
                pair_df["affine_ok"].sum()
            ),
            "visual_affine_success_rate": float(
                pair_df["affine_ok"].mean()
            ),
            "visual_good_matches_median": safe_stat(
                good_matches,
                np.median,
            ),
            "visual_inliers_median": safe_stat(
                inliers,
                np.median,
            ),
            "visual_inlier_ratio_median": safe_stat(
                inlier_ratios,
                np.median,
            ),
        }
    )

    return result, pair_rows


def main() -> None:
    if not INDEX_PATH.exists():
        raise FileNotFoundError(
            f"Missing SatLoc UAV index: {INDEX_PATH}"
        )

    index_df = pd.read_csv(INDEX_PATH)

    if "sequence" not in index_df.columns:
        raise RuntimeError(
            "'sequence' column is missing from the UAV index."
        )

    traj = index_df[
        index_df["sequence"].astype(str) == SEQUENCE
    ].copy()

    if traj.empty:
        raise RuntimeError(
            f"No frames found for sequence {SEQUENCE!r}."
        )

    missing_columns = [
        column
        for column in ORDER_CANDIDATES
        if column not in traj.columns
    ]

    available_candidates = [
        column
        for column in ORDER_CANDIDATES
        if column in traj.columns
    ]

    if not available_candidates:
        raise RuntimeError(
            "None of the expected order candidates are available."
        )

    summaries = []
    all_pair_rows = []

    for order_column in available_candidates:
        summary, pair_rows = audit_order_candidate(
            traj,
            order_column,
        )
        summaries.append(summary)
        all_pair_rows.extend(pair_rows)

    summary_df = pd.DataFrame(summaries)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    summary_df.to_csv(OUTPUT_CSV, index=False)

    json_payload = {
        "index_path": str(INDEX_PATH),
        "sequence": SEQUENCE,
        "rows": int(len(traj)),
        "missing_candidate_columns": missing_columns,
        "candidate_summaries": summaries,
        "sampled_visual_pairs": all_pair_rows,
        "important_rule": (
            "Reference coordinates were used only for dataset-order "
            "validation, not visual-motion estimation."
        ),
    }

    with OUTPUT_JSON.open("w", encoding="utf-8") as file:
        json.dump(json_payload, file, indent=2)

    print("\nS6A.0B sequence-order audit")
    print("---------------------------")
    print(f"Sequence: {SEQUENCE}")
    print(f"Rows:     {len(traj)}")

    for summary in summaries:
        print("\nCandidate:", summary["order_column"])
        print("-" * (11 + len(summary["order_column"])))
        print(
            "Unique order values:       "
            f"{summary['unique_order_values']}"
        )
        print(
            "Duplicate adjacent values: "
            f"{summary['duplicate_adjacent_orders']}"
        )
        print(
            "First 15 token0 IDs:        "
            f"{summary['first_15_token0_ids']}"
        )

        if "reference_total_path_m" in summary:
            print(
                "Reference total path [m]: "
                f"{summary['reference_total_path_m']:.3f}"
            )
            print(
                "Reference median step [m]: "
                f"{summary['reference_median_step_m']:.3f}"
            )
            print(
                "Reference p95 step [m]:    "
                f"{summary['reference_p95_step_m']:.3f}"
            )
            print(
                "Reference maximum step [m]: "
                f"{summary['reference_max_step_m']:.3f}"
            )

        print(
            "Visual affine success:     "
            f"{summary.get('visual_affine_success_rate', float('nan')):.3f}"
        )
        print(
            "Visual good median:        "
            f"{summary.get('visual_good_matches_median', float('nan')):.1f}"
        )
        print(
            "Visual inliers median:     "
            f"{summary.get('visual_inliers_median', float('nan')):.1f}"
        )
        print(
            "Visual inlier ratio median:"
            f" {summary.get('visual_inlier_ratio_median', float('nan')):.3f}"
        )

    print("\nSaved:")
    print(OUTPUT_CSV)
    print(OUTPUT_JSON)

    print(
        "\nDo not select an official order solely because its numeric "
        "values are consecutive. Confirm that token0 IDs, reference "
        "continuity, and visual overlap all agree."
    )


if __name__ == "__main__":
    main()
