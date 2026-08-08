'''
python -m py_compile \
  scripts/villoc/blind_demo/stage10b2_blind_map_bootstrap_audit.py

ROOT=outputs/villoc/traj01_90deg_stable120m
RUN="$ROOT/blind_demo_addons/eval_traj01_known_gt"

python scripts/villoc/blind_demo/stage10b2_blind_map_bootstrap_audit.py \
  --root "$ROOT" \
  --run-root "$RUN" \
  --min-visual-baseline-px 100 \
  --min-map-baseline-m 50 \
  --tile-margin-m 15 \
  --min-lock-support 3 \
  --min-lock-fraction 0.60 \
  2>&1 | tee \
  "$RUN/logs/stage10b2_blind_map_bootstrap_audit.log"
'''

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


def bool_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False)

    return (
        s.astype(str)
        .str.strip()
        .str.lower()
        .isin({
            "true",
            "1",
            "yes",
            "y",
            "t",
        })
    )


def fit_similarity_complex(
    visual_xy: np.ndarray,
    map_xy: np.ndarray,
) -> dict:
    """
    Fit:
        map_complex = a * visual_complex + b

    where complex coefficient a contains scale+rotation.
    No reflection is allowed.
    """

    if len(visual_xy) < 2:
        raise ValueError(
            "Need at least two correspondences."
        )

    z = (
        visual_xy[:, 0]
        + 1j * visual_xy[:, 1]
    )

    w = (
        map_xy[:, 0]
        + 1j * map_xy[:, 1]
    )

    z_mean = np.mean(z)
    w_mean = np.mean(w)

    z0 = z - z_mean
    w0 = w - w_mean

    denom = float(
        np.sum(
            np.abs(z0) ** 2
        )
    )

    if denom <= 1e-12:
        raise ValueError(
            "Degenerate visual baseline."
        )

    a = (
        np.sum(
            np.conj(z0) * w0
        )
        / denom
    )

    b = (
        w_mean
        - a * z_mean
    )

    scale = float(
        abs(a)
    )

    rotation_deg = float(
        np.degrees(
            np.angle(a)
        )
    )

    return {
        "a_real": float(a.real),
        "a_imag": float(a.imag),
        "b_real": float(b.real),
        "b_imag": float(b.imag),
        "scale_m_per_visual_px": scale,
        "rotation_deg": rotation_deg,
    }


def apply_similarity(
    visual_xy: np.ndarray,
    model: dict,
) -> np.ndarray:
    a = complex(
        model["a_real"],
        model["a_imag"],
    )

    b = complex(
        model["b_real"],
        model["b_imag"],
    )

    z = (
        visual_xy[:, 0]
        + 1j * visual_xy[:, 1]
    )

    w = (
        a * z
        + b
    )

    return np.column_stack([
        w.real,
        w.imag,
    ])


def point_to_rect_distance(
    x: np.ndarray,
    y: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    bottom: np.ndarray,
    top: np.ndarray,
) -> np.ndarray:
    dx = np.maximum.reduce([
        left - x,
        np.zeros_like(x),
        x - right,
    ])

    dy = np.maximum.reduce([
        bottom - y,
        np.zeros_like(y),
        y - top,
    ])

    return np.hypot(
        dx,
        dy,
    )


def choose_unique_representatives(
    rows: pd.DataFrame,
) -> pd.DataFrame:
    """
    Repeated observations of one map tile/center are not
    independent geometric anchors.

    Prefer:
      1. strict-B
      2. higher hybrid score
      3. more inliers
    """

    x = rows.copy()

    x["_center_key"] = (
        x["center_easting"]
        .round(3)
        .astype(str)
        + "_"
        + x["center_northing"]
        .round(3)
        .astype(str)
    )

    x = x.sort_values(
        [
            "_center_key",
            "strict_b_blind",
            "reranked_top1_hybrid_score",
            "reranked_top1_inliers",
        ],
        ascending=[
            True,
            False,
            False,
            False,
        ],
        kind="mergesort",
    )

    x = (
        x.drop_duplicates(
            "_center_key",
            keep="first",
        )
        .sort_values(
            "timestamp_s",
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    return x


def max_pairwise_span(
    xy: np.ndarray,
) -> float:
    if len(xy) < 2:
        return 0.0

    best = 0.0

    for i, j in combinations(
        range(len(xy)),
        2,
    ):
        d = float(
            np.linalg.norm(
                xy[j] - xy[i]
            )
        )

        best = max(
            best,
            d,
        )

    return best


def robust_pair_bootstrap(
    reps: pd.DataFrame,
    *,
    min_visual_baseline_px: float,
    min_map_baseline_m: float,
    tile_margin_m: float,
) -> dict | None:

    if len(reps) < 3:
        return None

    visual = reps[
        [
            "visual_x_px",
            "visual_y_px",
        ]
    ].to_numpy(float)

    map_xy = reps[
        [
            "center_easting",
            "center_northing",
        ]
    ].to_numpy(float)

    left = reps[
        "left_easting"
    ].to_numpy(float)

    right = reps[
        "right_easting"
    ].to_numpy(float)

    bottom = reps[
        "bottom_northing"
    ].to_numpy(float)

    top = reps[
        "top_northing"
    ].to_numpy(float)

    strict_b = reps[
        "strict_b_blind"
    ].to_numpy(bool)

    models = []

    for i, j in combinations(
        range(len(reps)),
        2,
    ):
        visual_baseline = float(
            np.linalg.norm(
                visual[j]
                - visual[i]
            )
        )

        map_baseline = float(
            np.linalg.norm(
                map_xy[j]
                - map_xy[i]
            )
        )

        if (
            visual_baseline
            < min_visual_baseline_px
        ):
            continue

        if (
            map_baseline
            < min_map_baseline_m
        ):
            continue

        try:
            model = fit_similarity_complex(
                visual[[i, j]],
                map_xy[[i, j]],
            )
        except ValueError:
            continue

        pred = apply_similarity(
            visual,
            model,
        )

        rect_dist = (
            point_to_rect_distance(
                pred[:, 0],
                pred[:, 1],
                left,
                right,
                bottom,
                top,
            )
        )

        center_dist = (
            np.linalg.norm(
                pred - map_xy,
                axis=1,
            )
        )

        inlier = (
            rect_dist
            <= tile_margin_m
        )

        support = int(
            inlier.sum()
        )

        strict_b_support = int(
            (
                inlier
                & strict_b
            ).sum()
        )

        models.append({
            **model,
            "seed_i": int(i),
            "seed_j": int(j),
            "seed_query_i": int(
                reps.iloc[i]["query_id"]
            ),
            "seed_query_j": int(
                reps.iloc[j]["query_id"]
            ),
            "visual_baseline_px": (
                visual_baseline
            ),
            "map_baseline_m": (
                map_baseline
            ),
            "support": support,
            "strict_b_support": (
                strict_b_support
            ),
            "support_fraction": float(
                support
                / len(reps)
            ),
            "median_rect_residual_m": float(
                np.median(
                    rect_dist
                )
            ),
            "median_center_residual_m": float(
                np.median(
                    center_dist
                )
            ),
            "_inlier_mask": inlier,
        })

    if not models:
        return None

    models.sort(
        key=lambda x: (
            -x["support"],
            -x["strict_b_support"],
            x[
                "median_rect_residual_m"
            ],
            x[
                "median_center_residual_m"
            ],
        )
    )

    best = models[0]

    inlier = best.pop(
        "_inlier_mask"
    )

    # Refine using all model-consistent unique anchors.
    if int(inlier.sum()) >= 2:
        refined = fit_similarity_complex(
            visual[inlier],
            map_xy[inlier],
        )

        pred = apply_similarity(
            visual,
            refined,
        )

        rect_dist = (
            point_to_rect_distance(
                pred[:, 0],
                pred[:, 1],
                left,
                right,
                bottom,
                top,
            )
        )

        center_dist = np.linalg.norm(
            pred - map_xy,
            axis=1,
        )

        refined_inlier = (
            rect_dist
            <= tile_margin_m
        )

        best.update(
            refined
        )

        best[
            "refined_support"
        ] = int(
            refined_inlier.sum()
        )

        best[
            "refined_strict_b_support"
        ] = int(
            (
                refined_inlier
                & strict_b
            ).sum()
        )

        best[
            "refined_support_fraction"
        ] = float(
            refined_inlier.mean()
        )

        best[
            "refined_median_rect_residual_m"
        ] = float(
            np.median(
                rect_dist
            )
        )

        best[
            "refined_median_center_residual_m"
        ] = float(
            np.median(
                center_dist
            )
        )

        best[
            "_refined_inlier"
        ] = refined_inlier

    return best


def first_distinct_time(
    df: pd.DataFrame,
    count: int,
) -> float | None:

    seen = set()

    for _, row in df.sort_values(
        "timestamp_s"
    ).iterrows():

        key = (
            round(
                float(
                    row["center_easting"]
                ),
                3,
            ),
            round(
                float(
                    row["center_northing"]
                ),
                3,
            ),
        )

        seen.add(key)

        if len(seen) >= count:
            return float(
                row["timestamp_s"]
            )

    return None


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--run-root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--min-visual-baseline-px",
        type=float,
        default=100.0,
    )

    parser.add_argument(
        "--min-map-baseline-m",
        type=float,
        default=50.0,
    )

    parser.add_argument(
        "--tile-margin-m",
        type=float,
        default=15.0,
    )

    parser.add_argument(
        "--min-lock-support",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--min-lock-fraction",
        type=float,
        default=0.60,
    )

    args = parser.parse_args()

    started = time.perf_counter()

    root = args.root.resolve()
    run_root = args.run_root.resolve()

    blind_path = (
        run_root
        / "metadata/blind_query_manifest.csv"
    )

    visual_path = (
        run_root
        / "metadata/s8_xfeat_relative_frontend/"
          "s8r4_xfeat_relative_trajectory_blind_raw.csv"
    )

    qsum_path = (
        root
        / "reports/s8_12e1_top20_verifier_reranker/"
          "512_s256_orb_hybrid_top20_img518/"
          "s8_12e1_query_summary.csv"
    )

    cand_path = (
        root
        / "reports/s8_12e1_top20_verifier_reranker/"
          "512_s256_orb_hybrid_top20_img518/"
          "s8_12e1_all_candidate_verifier_scores.csv"
    )

    for path in [
        blind_path,
        visual_path,
        qsum_path,
        cand_path,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                path
            )

    # -------------------------------------------------
    # Read ONLY blind-safe columns.
    # -------------------------------------------------

    blind = pd.read_csv(
        blind_path,
        usecols=[
            "sequence_frame_id",
            "query_id",
            "token0_id",
            "timestamp_s",
        ],
    )

    visual = pd.read_csv(
        visual_path,
        usecols=[
            "sequence_frame_id",
            "token0_id",
            "visual_x_px",
            "visual_y_px",
        ],
    )

    qsum = pd.read_csv(
        qsum_path,
        usecols=[
            "query_id",
            "reranked_top1_tile_id",
            "reranked_top1_original_rank",
            "reranked_top1_inliers",
            "reranked_top1_inlier_ratio",
            "reranked_top1_query_inlier_coverage",
            "reranked_top1_verifier_score",
            "reranked_top1_hybrid_score",
        ],
    )

    cand = pd.read_csv(
        cand_path,
        usecols=[
            "query_id",
            "tile_id",
            "center_easting",
            "center_northing",
            "left_easting",
            "right_easting",
            "bottom_northing",
            "top_northing",
            "homography_ok",
        ],
    )

    for df in [
        blind,
        qsum,
        cand,
    ]:
        if "query_id" in df.columns:
            df["query_id"] = (
                pd.to_numeric(
                    df["query_id"],
                    errors="raise",
                )
                .astype(int)
            )

    selected = qsum.merge(
        cand,
        left_on=[
            "query_id",
            "reranked_top1_tile_id",
        ],
        right_on=[
            "query_id",
            "tile_id",
        ],
        how="left",
        validate="one_to_one",
    )

    if (
        selected["tile_id"]
        .isna()
        .any()
    ):
        raise RuntimeError(
            "Missing selected ORB candidate."
        )

    homography = bool_series(
        selected[
            "homography_ok"
        ]
    )

    inliers = pd.to_numeric(
        selected[
            "reranked_top1_inliers"
        ],
        errors="coerce",
    )

    ratio = pd.to_numeric(
        selected[
            "reranked_top1_inlier_ratio"
        ],
        errors="coerce",
    )

    coverage = pd.to_numeric(
        selected[
            "reranked_top1_query_inlier_coverage"
        ],
        errors="coerce",
    )

    rank = pd.to_numeric(
        selected[
            "reranked_top1_original_rank"
        ],
        errors="coerce",
    )

    selected[
        "strict_a_blind"
    ] = (
        homography
        & (inliers >= 25)
        & (ratio >= 0.30)
        & (coverage >= 0.10)
        & (rank <= 15)
    )

    selected[
        "strict_b_blind"
    ] = (
        homography
        & (inliers >= 40)
        & (ratio >= 0.35)
        & (coverage >= 0.12)
        & (rank <= 12)
    )

    safe = (
        blind.merge(
            visual,
            on=[
                "sequence_frame_id",
                "token0_id",
            ],
            how="inner",
            validate="one_to_one",
        )
        .merge(
            selected,
            on="query_id",
            how="inner",
            validate="one_to_one",
        )
        .sort_values(
            "timestamp_s",
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    if len(safe) != 403:
        raise RuntimeError(
            f"Expected 403 rows, got {len(safe)}."
        )

    strict_a = safe[
        safe["strict_a_blind"]
    ].copy()

    strict_b = safe[
        safe["strict_b_blind"]
    ].copy()

    if strict_a.empty:
        raise RuntimeError(
            "No strict-A blind anchors."
        )

    first_a = strict_a.iloc[0]
    first_b = (
        strict_b.iloc[0]
        if len(strict_b)
        else None
    )

    two_distinct_time = (
        first_distinct_time(
            strict_a,
            2,
        )
    )

    three_distinct_time = (
        first_distinct_time(
            strict_a,
            3,
        )
    )

    # -------------------------------------------------
    # Incremental online map-lock search.
    # -------------------------------------------------

    lock = None
    lock_reps = None

    event_times = sorted(
        strict_a[
            "timestamp_s"
        ]
        .unique()
        .tolist()
    )

    for t in event_times:
        seen = strict_a[
            strict_a["timestamp_s"]
            <= t
        ].copy()

        reps = (
            choose_unique_representatives(
                seen
            )
        )

        if len(reps) < 3:
            continue

        if not bool(
            reps[
                "strict_b_blind"
            ].any()
        ):
            continue

        visual_span = (
            max_pairwise_span(
                reps[
                    [
                        "visual_x_px",
                        "visual_y_px",
                    ]
                ].to_numpy(float)
            )
        )

        map_span = (
            max_pairwise_span(
                reps[
                    [
                        "center_easting",
                        "center_northing",
                    ]
                ].to_numpy(float)
            )
        )

        if (
            visual_span
            < args.min_visual_baseline_px
        ):
            continue

        if (
            map_span
            < args.min_map_baseline_m
        ):
            continue

        model = robust_pair_bootstrap(
            reps,
            min_visual_baseline_px=(
                args.min_visual_baseline_px
            ),
            min_map_baseline_m=(
                args.min_map_baseline_m
            ),
            tile_margin_m=(
                args.tile_margin_m
            ),
        )

        if model is None:
            continue

        support = int(
            model.get(
                "refined_support",
                model["support"],
            )
        )

        fraction = float(
            model.get(
                "refined_support_fraction",
                model[
                    "support_fraction"
                ],
            )
        )

        strict_b_support = int(
            model.get(
                "refined_strict_b_support",
                model[
                    "strict_b_support"
                ],
            )
        )

        if (
            support
            >= args.min_lock_support
            and fraction
            >= args.min_lock_fraction
            and strict_b_support
            >= 1
        ):
            lock = {
                **{
                    k: v
                    for k, v in model.items()
                    if not k.startswith("_")
                },
                "lock_timestamp_s": float(t),
                "lock_sequence_frame_id": int(
                    safe.loc[
                        safe["timestamp_s"]
                        == t,
                        "sequence_frame_id",
                    ].iloc[-1]
                ),
                "unique_anchor_count": int(
                    len(reps)
                ),
                "visual_span_px": float(
                    visual_span
                ),
                "map_span_m": float(
                    map_span
                ),
            }

            lock_reps = reps.copy()
            break

    if lock is None:
        status = (
            "REVIEW_NO_BLIND_MAP_LOCK"
        )
    else:
        status = (
            "PASS_BLIND_MAP_BOOTSTRAP"
        )

    metadata_dir = (
        run_root
        / "metadata/blind_map_bootstrap"
    )

    report_dir = (
        run_root
        / "reports/blind_map_bootstrap"
    )

    metadata_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    safe_anchor_path = (
        metadata_dir
        / "blind_anchor_candidates.csv"
    )

    report_path = (
        report_dir
        / "blind_map_bootstrap_report.json"
    )

    safe.to_csv(
        safe_anchor_path,
        index=False,
    )

    lock_anchor_path = None

    if lock_reps is not None:
        lock_anchor_path = (
            metadata_dir
            / "blind_map_lock_unique_anchors.csv"
        )

        lock_reps.drop(
            columns=[
                "_center_key",
            ],
            errors="ignore",
        ).to_csv(
            lock_anchor_path,
            index=False,
        )

    finished = time.perf_counter()

    report = {
        "stage": (
            "STAGE_10B2_BLIND_MAP_BOOTSTRAP"
        ),
        "status": status,
        "blind_contract": {
            "gps_used": False,
            "srt_used": False,
            "reference_used": False,
            "oracle_used": False,
            "evaluation_error_used": False,
            "map_coordinates_allowed": True,
        },
        "counts": {
            "frames": int(
                len(safe)
            ),
            "strict_a_candidates": int(
                len(strict_a)
            ),
            "strict_b_candidates": int(
                len(strict_b)
            ),
        },
        "acquisition": {
            "first_strict_a_time_s": float(
                first_a["timestamp_s"]
            ),
            "first_strict_a_query_id": int(
                first_a["query_id"]
            ),
            "first_strict_b_time_s": (
                float(
                    first_b["timestamp_s"]
                )
                if first_b is not None
                else None
            ),
            "first_strict_b_query_id": (
                int(
                    first_b["query_id"]
                )
                if first_b is not None
                else None
            ),
            "first_two_distinct_map_centers_time_s": (
                two_distinct_time
            ),
            "first_three_distinct_map_centers_time_s": (
                three_distinct_time
            ),
        },
        "lock_configuration": {
            "min_visual_baseline_px": (
                args.min_visual_baseline_px
            ),
            "min_map_baseline_m": (
                args.min_map_baseline_m
            ),
            "tile_margin_m": (
                args.tile_margin_m
            ),
            "min_lock_support": (
                args.min_lock_support
            ),
            "min_lock_fraction": (
                args.min_lock_fraction
            ),
            "requires_strict_b_support": True,
        },
        "map_lock": lock,
        "runtime": {
            "total_stage_wall_s": float(
                finished
                - started
            )
        },
        "outputs": {
            "anchor_candidates": str(
                safe_anchor_path
            ),
            "lock_unique_anchors": (
                str(lock_anchor_path)
                if lock_anchor_path
                else None
            ),
        },
    }

    report_path.write_text(
        json.dumps(
            report,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    print("=" * 80)
    print(
        "STAGE 10B.2 — BLIND MAP "
        "BOOTSTRAP AUDIT"
    )
    print("=" * 80)

    print()
    print("Blind contract")
    print("-" * 80)
    print("GPS used       : false")
    print("SRT used       : false")
    print("reference used : false")
    print("oracle used    : false")

    print()
    print("Acquisition")
    print("-" * 80)

    print(
        "first strict-A fix        :",
        f"{float(first_a['timestamp_s']):.3f} s",
        f"(query {int(first_a['query_id'])})",
    )

    if first_b is not None:
        print(
            "first strict-B fix        :",
            f"{float(first_b['timestamp_s']):.3f} s",
            f"(query {int(first_b['query_id'])})",
        )

    print(
        "two distinct map centers  :",
        two_distinct_time,
        "s",
    )

    print(
        "three distinct map centers:",
        three_distinct_time,
        "s",
    )

    print()
    print("Map lock")
    print("-" * 80)

    if lock is None:
        print("LOCK NOT FOUND")

    else:
        print(
            "lock time                 :",
            f"{lock['lock_timestamp_s']:.3f} s",
        )

        print(
            "lock frame                :",
            lock[
                "lock_sequence_frame_id"
            ],
        )

        print(
            "unique anchors            :",
            lock[
                "unique_anchor_count"
            ],
        )

        print(
            "support                   :",
            lock.get(
                "refined_support",
                lock["support"],
            ),
        )

        print(
            "strict-B support          :",
            lock.get(
                "refined_strict_b_support",
                lock["strict_b_support"],
            ),
        )

        print(
            "support fraction          :",
            f"{lock.get('refined_support_fraction', lock['support_fraction']):.3f}",
        )

        print(
            "visual span               :",
            f"{lock['visual_span_px']:.3f} px",
        )

        print(
            "map span                  :",
            f"{lock['map_span_m']:.3f} m",
        )

        print(
            "scale                     :",
            f"{lock['scale_m_per_visual_px']:.6f} m/px",
        )

        print(
            "rotation                  :",
            f"{lock['rotation_deg']:.3f} deg",
        )

        print(
            "median center residual    :",
            f"{lock.get('refined_median_center_residual_m', lock['median_center_residual_m']):.3f} m",
        )

    print()
    print("Saved")
    print("-" * 80)
    print(safe_anchor_path)
    print(lock_anchor_path)
    print(report_path)

    print()
    print("status:", status)


if __name__ == "__main__":
    main()
