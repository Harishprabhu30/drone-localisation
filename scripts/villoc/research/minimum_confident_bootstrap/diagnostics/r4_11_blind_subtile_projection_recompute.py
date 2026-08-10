#!/usr/bin/env python3
"""
R4.11 — blind sub-tile projection recomputation.

Purpose
-------
Test whether ORB query→satellite homography can replace coarse satellite
tile-centre measurements with continuous within-tile map observations.

PHASE A — BLIND
----------------
For existing q1..q38 Top-20 candidate pairs:

  1. reproduce the exact historical ORB verifier;
  2. verify recomputed statistics against frozen stored statistics;
  3. retain query→tile homography H;
  4. project the processed UAV image centre into tile pixels;
  5. convert projected tile pixels to EPSG:3346 using tile bounds;
  6. freeze and hash all blind outputs.

No GT/reference/SRT/GPS is read.

PHASE B — POST-FREEZE
------------------------
Only after Phase-A freeze:

  * attach reference trajectory;
  * compare tile-centre position error vs sub-tile projected-point error;
  * stratify by existing candidate/reranker evidence.

No R3 algorithm is modified.

Locked R4.10 contract
---------------------
preprocess      = clahe_luma
resize_long     = 1024
ORB nfeatures   = 1800
fastThreshold   = 7
edgeThreshold   = 15
patchSize       = 31
Lowe ratio      = 0.80
RANSAC threshold= 5.0 px

H direction:
    query image pixel -> satellite tile pixel

Command:

SCRIPT=scripts/villoc/research/minimum_confident_bootstrap/diagnostics/r4_11_blind_subtile_projection_recompute.py
RUN=outputs/demo_runs/traj01_blind_regression_001

R3=outputs/research_runs/minimum_confident_bootstrap/traj01_blind_r3_001


python "$SCRIPT" \
  --repo-root "$PWD" \
  --run-root "$RUN" \
  --research-root "$R3" \
  --prefix-max-query 38 \
  2>&1 | tee \
  "$R3/postfreeze_eval/r4_11_blind_subtile_projection_recompute.log"

"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from pyproj import Transformer


# ============================================================
# Locked verifier configuration
# ============================================================

PREPROCESS = "clahe_luma"
RESIZE_LONG = 1024

NFEATURES = 1800
FAST_THRESHOLD = 7
EDGE_THRESHOLD = 15
PATCH_SIZE = 31

LOWE_RATIO = 0.80
RANSAC_THRESHOLD = 5.0


# ============================================================
# Helpers
# ============================================================

def sha256(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def bool_series(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin(
            [
                "true",
                "1",
                "yes",
            ]
        )
    )


def bbox_coverage(
    points: np.ndarray,
    image_hw,
) -> float:

    h, w = image_hw

    if (
        h <= 0
        or w <= 0
        or len(points) == 0
    ):
        return 0.0

    xs = points[:, 0]
    ys = points[:, 1]

    area = (
        max(
            0.0,
            float(
                xs.max()
                - xs.min()
            ),
        )
        *
        max(
            0.0,
            float(
                ys.max()
                - ys.min()
            ),
        )
    )

    return float(
        np.clip(
            area
            / float(
                w * h
            ),
            0.0,
            1.0,
        )
    )


# ============================================================
# Exact historical preprocessing
# ============================================================

@dataclass
class FeaturePack:
    ok: bool
    image_shape: tuple[int, int]
    keypoints: list
    descriptors: np.ndarray | None
    error: str | None


def read_image_for_verifier(
    path: Path,
) -> tuple[np.ndarray | None, str | None]:

    img = cv2.imread(
        str(path),
        cv2.IMREAD_COLOR,
    )

    if img is None:
        return (
            None,
            f"cv2.imread failed: {path}",
        )

    h, w = img.shape[:2]

    scale = (
        float(
            RESIZE_LONG
        )
        / float(
            max(
                h,
                w,
            )
        )
    )

    # Historical verifier ONLY downsizes.
    if scale < 1.0:

        img = cv2.resize(
            img,
            (
                int(
                    round(
                        w * scale
                    )
                ),
                int(
                    round(
                        h * scale
                    )
                ),
            ),
            interpolation=cv2.INTER_AREA,
        )

    ycrcb = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2YCrCb,
    )

    y = ycrcb[
        :,
        :,
        0,
    ]

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(
            8,
            8,
        ),
    )

    gray = clahe.apply(
        y
    )

    return (
        gray,
        None,
    )


def create_detector():
    return cv2.ORB_create(
        nfeatures=NFEATURES,
        fastThreshold=FAST_THRESHOLD,
        edgeThreshold=EDGE_THRESHOLD,
        patchSize=PATCH_SIZE,
    )


def compute_features(
    path: Path,
    detector,
) -> FeaturePack:

    gray, error = (
        read_image_for_verifier(
            path
        )
    )

    if gray is None:

        return FeaturePack(
            False,
            (
                0,
                0,
            ),
            [],
            None,
            error,
        )

    keypoints, descriptors = (
        detector.detectAndCompute(
            gray,
            None,
        )
    )

    if (
        descriptors is None
        or len(
            keypoints
        )
        == 0
    ):

        return FeaturePack(
            False,
            gray.shape[:2],
            keypoints or [],
            None,
            "no descriptors",
        )

    return FeaturePack(
        True,
        gray.shape[:2],
        keypoints,
        descriptors,
        None,
    )


# ============================================================
# Exact verifier + retained H
# ============================================================

def verify_pair_with_geometry(
    q: FeaturePack,
    s: FeaturePack,
):

    result = {
        "good_matches": 0,
        "inliers": 0,
        "inlier_ratio": 0.0,
        "query_inlier_coverage": 0.0,
        "sat_inlier_coverage": 0.0,
        "homography_ok": False,
        "verifier_score": 0.0,

        "H": None,

        "query_center_u_px": math.nan,
        "query_center_v_px": math.nan,

        "projected_tile_u_px": math.nan,
        "projected_tile_v_px": math.nan,

        "projective_denominator": math.nan,

        "inlier_reprojection_rmse_px": math.nan,

        "error": None,
    }

    if (
        not q.ok
        or not s.ok
        or q.descriptors is None
        or s.descriptors is None
    ):

        result[
            "error"
        ] = (
            q.error
            or s.error
            or "features missing"
        )

        return result


    matcher = cv2.BFMatcher(
        cv2.NORM_HAMMING,
        crossCheck=False,
    )


    try:

        knn = matcher.knnMatch(
            q.descriptors,
            s.descriptors,
            k=2,
        )

    except cv2.error as exc:

        result[
            "error"
        ] = (
            f"knnMatch failed: {exc}"
        )

        return result


    good = []

    for pair in knn:

        if len(
            pair
        ) < 2:
            continue

        m, n = pair

        if (
            m.distance
            <
            LOWE_RATIO
            * n.distance
        ):

            good.append(
                m
            )


    result[
        "good_matches"
    ] = int(
        len(
            good
        )
    )


    if len(
        good
    ) < 4:

        result[
            "verifier_score"
        ] = float(
            0.03
            * len(
                good
            )
        )

        return result


    q_pts = np.float32(
        [
            q.keypoints[
                m.queryIdx
            ].pt
            for m in good
        ]
    )


    s_pts = np.float32(
        [
            s.keypoints[
                m.trainIdx
            ].pt
            for m in good
        ]
    )


    H, mask = cv2.findHomography(
        q_pts,
        s_pts,
        cv2.RANSAC,
        RANSAC_THRESHOLD,
    )


    if mask is None:

        inlier_mask = np.zeros(
            len(
                good
            ),
            dtype=bool,
        )

    else:

        inlier_mask = (
            mask.ravel()
            .astype(bool)
        )


    inliers = int(
        inlier_mask.sum()
    )


    q_cov = bbox_coverage(
        q_pts[
            inlier_mask
        ],
        q.image_shape,
    )


    s_cov = bbox_coverage(
        s_pts[
            inlier_mask
        ],
        s.image_shape,
    )


    inlier_ratio = float(
        inliers
        / max(
            1,
            len(
                good
            ),
        )
    )


    homography_ok = bool(
        H is not None
        and inliers >= 4
    )


    verifier_score = (
        float(
            inliers
        )
        +
        0.04
        * float(
            len(
                good
            )
        )
        +
        8.0
        * float(
            inlier_ratio
        )
        +
        4.0
        * float(
            math.sqrt(
                max(
                    q_cov,
                    0.0,
                )
            )
        )
        +
        2.0
        * float(
            math.sqrt(
                max(
                    s_cov,
                    0.0,
                )
            )
        )
    )


    result.update(
        {
            "inliers":
                inliers,

            "inlier_ratio":
                inlier_ratio,

            "query_inlier_coverage":
                q_cov,

            "sat_inlier_coverage":
                s_cov,

            "homography_ok":
                homography_ok,

            "verifier_score":
                verifier_score,

            "H":
                H,
        }
    )


    if not homography_ok:

        return result


    # --------------------------------------------------------
    # Reprojection quality of the actual RANSAC inliers.
    # --------------------------------------------------------

    if inliers > 0:

        projected_inliers = (
            cv2.perspectiveTransform(
                q_pts[
                    inlier_mask
                ]
                .reshape(
                    -1,
                    1,
                    2,
                ),
                H,
            )
            .reshape(
                -1,
                2,
            )
        )


        residual = (
            projected_inliers
            - s_pts[
                inlier_mask
            ]
        )


        rmse = float(
            np.sqrt(
                np.mean(
                    np.sum(
                        residual
                        ** 2,
                        axis=1,
                    )
                )
            )
        )


        result[
            "inlier_reprojection_rmse_px"
        ] = rmse


    # --------------------------------------------------------
    # Query optical/image centre in processed-image coordinates.
    #
    # Continuous image coordinates:
    #   centre = (W/2, H/2)
    #
    # q1 expected:
    #   1024 × 576 -> (512, 288)
    # --------------------------------------------------------

    qh, qw = (
        q.image_shape
    )


    center_u = (
        float(
            qw
        )
        / 2.0
    )

    center_v = (
        float(
            qh
        )
        / 2.0
    )


    result[
        "query_center_u_px"
    ] = center_u

    result[
        "query_center_v_px"
    ] = center_v


    # Projective denominator, useful as blind numerical audit.
    denominator = float(
        H[
            2,
            0
        ]
        * center_u
        +
        H[
            2,
            1
        ]
        * center_v
        +
        H[
            2,
            2
        ]
    )


    result[
        "projective_denominator"
    ] = denominator


    point = np.float32(
        [
            [
                [
                    center_u,
                    center_v,
                ]
            ]
        ]
    )


    projected = (
        cv2.perspectiveTransform(
            point,
            H,
        )
        .reshape(
            2,
        )
    )


    result[
        "projected_tile_u_px"
    ] = float(
        projected[
            0
        ]
    )


    result[
        "projected_tile_v_px"
    ] = float(
        projected[
            1
        ]
    )


    return result


# ============================================================
# Tile pixel -> EPSG:3346
# ============================================================

def tile_pixel_to_map(
    u: float,
    v: float,
    tile_hw,
    left_easting: float,
    right_easting: float,
    bottom_northing: float,
    top_northing: float,
):

    h, w = tile_hw


    # --------------------------------------------------------
    # Raster outer-edge convention.
    #
    # OpenCV integer coordinates refer to pixel centres.
    # Raster bounds describe tile outer edges.
    #
    # Thus pixel centre u=0 corresponds to:
    #     (0.5 / W)
    #
    # of the tile's map extent.
    # --------------------------------------------------------

    easting = (
        left_easting
        +
        (
            (
                u
                + 0.5
            )
            / float(
                w
            )
        )
        *
        (
            right_easting
            - left_easting
        )
    )


    northing = (
        top_northing
        -
        (
            (
                v
                + 0.5
            )
            / float(
                h
            )
        )
        *
        (
            top_northing
            - bottom_northing
        )
    )


    return (
        float(
            easting
        ),
        float(
            northing
        ),
    )


# ============================================================
# Evaluation helpers
# ============================================================

def error_summary(
    group: pd.DataFrame,
):

    valid = group[
        np.isfinite(
            group[
                "projected_error_m"
            ]
        )
    ].copy()


    if len(
        valid
    ) == 0:

        return {
            "pairs":
                0,
        }


    delta = (
        valid[
            "projected_error_m"
        ]
        -
        valid[
            "tile_center_error_m"
        ]
    )


    return {
        "pairs":
            int(
                len(
                    valid
                )
            ),

        "tile_center_error_median":
            float(
                valid[
                    "tile_center_error_m"
                ].median()
            ),

        "projected_error_median":
            float(
                valid[
                    "projected_error_m"
                ].median()
            ),

        "tile_center_error_p90":
            float(
                valid[
                    "tile_center_error_m"
                ].quantile(
                    0.90
                )
            ),

        "projected_error_p90":
            float(
                valid[
                    "projected_error_m"
                ].quantile(
                    0.90
                )
            ),

        "median_projected_minus_center_m":
            float(
                delta.median()
            ),

        "projected_better_count":
            int(
                (
                    delta
                    < 0.0
                ).sum()
            ),

        "projected_better_fraction":
            float(
                (
                    delta
                    < 0.0
                ).mean()
            ),
    }


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()


    parser.add_argument(
        "--repo-root",
        type=Path,
        required=True,
    )


    parser.add_argument(
        "--run-root",
        type=Path,
        required=True,
    )


    parser.add_argument(
        "--research-root",
        type=Path,
        required=True,
    )


    parser.add_argument(
        "--prefix-max-query",
        type=int,
        default=38,
    )


    args = parser.parse_args()


    repo = (
        args.repo_root
        .resolve()
    )


    run = (
        args.run_root
        .resolve()
    )


    research = (
        args.research_root
        .resolve()
    )


    max_query = int(
        args.prefix_max_query
    )


    out_dir = (
        research
        / "postfreeze_eval"
    )


    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    candidate_csv = (
        run
        / "reports/"
          "s8_12e1_top20_verifier_reranker/"
          "512_s256_orb_hybrid_top20_img518/"
          "s8_12e1_all_candidate_verifier_scores.csv"
    )


    verifier_source = (
        repo
        / "scripts/"
          "villoc/"
          "s8_12e1_top20_verifier_reranker.py"
    )


    if not candidate_csv.exists():
        raise RuntimeError(
            f"Missing candidate CSV: {candidate_csv}"
        )


    # ========================================================
    #
    # PHASE A — BLIND
    #
    # ========================================================

    candidates = pd.read_csv(
        candidate_csv
    )


    candidates[
        "query_id"
    ] = pd.to_numeric(
        candidates[
            "query_id"
        ],
        errors="raise",
    ).astype(int)


    prefix = (
        candidates[
            candidates[
                "query_id"
            ]
            <= max_query
        ]
        .copy()
        .sort_values(
            [
                "query_id",
                "hybrid_rank",
            ]
        )
        .reset_index(
            drop=True
        )
    )


    detector = create_detector()


    feature_cache: dict[
        str,
        FeaturePack
    ] = {}


    def features(path_text):
        path = Path(
            str(
                path_text
            )
        )

        if not path.is_absolute():
            path = (
                repo
                / path
            )

        key = str(
            path.resolve()
        )

        if key not in feature_cache:

            feature_cache[
                key
            ] = compute_features(
                Path(
                    key
                ),
                detector,
            )

        return feature_cache[
            key
        ]


    rows = []


    for pair_index, row in (
        prefix.iterrows()
    ):

        q = features(
            row[
                "query_image_resolved"
            ]
        )


        s = features(
            row[
                "tile_image_resolved"
            ]
        )


        result = (
            verify_pair_with_geometry(
                q,
                s,
            )
        )


        H = result[
            "H"
        ]


        H_values = {
            f"H{r}{c}":
                math.nan
            for r in range(
                3
            )
            for c in range(
                3
            )
        }


        projected_easting = (
            math.nan
        )

        projected_northing = (
            math.nan
        )

        projected_inside_tile = (
            False
        )


        if (
            H is not None
            and result[
                "homography_ok"
            ]
            and np.isfinite(
                result[
                    "projected_tile_u_px"
                ]
            )
            and np.isfinite(
                result[
                    "projected_tile_v_px"
                ]
            )
        ):

            Hn = np.asarray(
                H,
                dtype=float,
            )


            if (
                abs(
                    Hn[
                        2,
                        2
                    ]
                )
                > 1e-12
            ):

                Hn = (
                    Hn
                    / Hn[
                        2,
                        2
                    ]
                )


            for r in range(
                3
            ):

                for c in range(
                    3
                ):

                    H_values[
                        f"H{r}{c}"
                    ] = float(
                        Hn[
                            r,
                            c
                        ]
                    )


            u = float(
                result[
                    "projected_tile_u_px"
                ]
            )

            v = float(
                result[
                    "projected_tile_v_px"
                ]
            )


            sh, sw = (
                s.image_shape
            )


            projected_inside_tile = bool(
                0.0
                <= u
                < float(
                    sw
                )
                and
                0.0
                <= v
                < float(
                    sh
                )
            )


            (
                projected_easting,
                projected_northing,
            ) = tile_pixel_to_map(
                u,
                v,
                s.image_shape,
                float(
                    row[
                        "left_easting"
                    ]
                ),
                float(
                    row[
                        "right_easting"
                    ]
                ),
                float(
                    row[
                        "bottom_northing"
                    ]
                ),
                float(
                    row[
                        "top_northing"
                    ]
                ),
            )


        rows.append(
            {
                "query_id":
                    int(
                        row[
                            "query_id"
                        ]
                    ),

                "tile_id":
                    str(
                        row[
                            "tile_id"
                        ]
                    ),

                "dino_rank":
                    int(
                        row[
                            "rank"
                        ]
                    ),

                "hybrid_rank":
                    int(
                        row[
                            "hybrid_rank"
                        ]
                    ),

                "dino_score":
                    float(
                        row[
                            "score"
                        ]
                    ),

                "stored_good_matches":
                    int(
                        row[
                            "good_matches"
                        ]
                    ),

                "recomputed_good_matches":
                    int(
                        result[
                            "good_matches"
                        ]
                    ),

                "stored_inliers":
                    int(
                        row[
                            "inliers"
                        ]
                    ),

                "recomputed_inliers":
                    int(
                        result[
                            "inliers"
                        ]
                    ),

                "stored_inlier_ratio":
                    float(
                        row[
                            "inlier_ratio"
                        ]
                    ),

                "recomputed_inlier_ratio":
                    float(
                        result[
                            "inlier_ratio"
                        ]
                    ),

                "stored_query_inlier_coverage":
                    float(
                        row[
                            "query_inlier_coverage"
                        ]
                    ),

                "recomputed_query_inlier_coverage":
                    float(
                        result[
                            "query_inlier_coverage"
                        ]
                    ),

                "stored_sat_inlier_coverage":
                    float(
                        row[
                            "sat_inlier_coverage"
                        ]
                    ),

                "recomputed_sat_inlier_coverage":
                    float(
                        result[
                            "sat_inlier_coverage"
                        ]
                    ),

                "stored_homography_ok":
                    bool(
                        str(
                            row[
                                "homography_ok"
                            ]
                        )
                        .lower()
                        in {
                            "true",
                            "1",
                            "yes",
                        }
                    ),

                "recomputed_homography_ok":
                    bool(
                        result[
                            "homography_ok"
                        ]
                    ),

                "stored_verifier_score":
                    float(
                        row[
                            "verifier_score"
                        ]
                    ),

                "recomputed_verifier_score":
                    float(
                        result[
                            "verifier_score"
                        ]
                    ),

                "query_processed_height":
                    int(
                        q.image_shape[
                            0
                        ]
                    ),

                "query_processed_width":
                    int(
                        q.image_shape[
                            1
                        ]
                    ),

                "tile_processed_height":
                    int(
                        s.image_shape[
                            0
                        ]
                    ),

                "tile_processed_width":
                    int(
                        s.image_shape[
                            1
                        ]
                    ),

                "query_center_u_px":
                    result[
                        "query_center_u_px"
                    ],

                "query_center_v_px":
                    result[
                        "query_center_v_px"
                    ],

                "projected_tile_u_px":
                    result[
                        "projected_tile_u_px"
                    ],

                "projected_tile_v_px":
                    result[
                        "projected_tile_v_px"
                    ],

                "projected_inside_tile":
                    projected_inside_tile,

                "projective_denominator":
                    result[
                        "projective_denominator"
                    ],

                "inlier_reprojection_rmse_px":
                    result[
                        "inlier_reprojection_rmse_px"
                    ],

                "projected_easting":
                    projected_easting,

                "projected_northing":
                    projected_northing,

                "center_easting":
                    float(
                        row[
                            "center_easting"
                        ]
                    ),

                "center_northing":
                    float(
                        row[
                            "center_northing"
                        ]
                    ),

                "left_easting":
                    float(
                        row[
                            "left_easting"
                        ]
                    ),

                "right_easting":
                    float(
                        row[
                            "right_easting"
                        ]
                    ),

                "bottom_northing":
                    float(
                        row[
                            "bottom_northing"
                        ]
                    ),

                "top_northing":
                    float(
                        row[
                            "top_northing"
                        ]
                    ),

                **H_values,

                "verifier_error":
                    result[
                        "error"
                    ],
            }
        )


        if (
            (
                pair_index
                + 1
            )
            % 100
            == 0
        ):

            print(
                "processed pairs:",
                pair_index
                + 1,
                "/",
                len(
                    prefix
                ),
            )


    blind = pd.DataFrame(
        rows
    )


    # ========================================================
    # Exact reproduction gate
    # ========================================================

    good_match_mismatch = int(
        (
            blind[
                "stored_good_matches"
            ]
            !=
            blind[
                "recomputed_good_matches"
            ]
        ).sum()
    )


    inlier_mismatch = int(
        (
            blind[
                "stored_inliers"
            ]
            !=
            blind[
                "recomputed_inliers"
            ]
        ).sum()
    )


    homography_mismatch = int(
        (
            blind[
                "stored_homography_ok"
            ]
            !=
            blind[
                "recomputed_homography_ok"
            ]
        ).sum()
    )


    max_ratio_diff = float(
        np.nanmax(
            np.abs(
                blind[
                    "stored_inlier_ratio"
                ]
                -
                blind[
                    "recomputed_inlier_ratio"
                ]
            )
        )
    )


    max_query_cov_diff = float(
        np.nanmax(
            np.abs(
                blind[
                    "stored_query_inlier_coverage"
                ]
                -
                blind[
                    "recomputed_query_inlier_coverage"
                ]
            )
        )
    )


    max_sat_cov_diff = float(
        np.nanmax(
            np.abs(
                blind[
                    "stored_sat_inlier_coverage"
                ]
                -
                blind[
                    "recomputed_sat_inlier_coverage"
                ]
            )
        )
    )


    max_verifier_score_diff = float(
        np.nanmax(
            np.abs(
                blind[
                    "stored_verifier_score"
                ]
                -
                blind[
                    "recomputed_verifier_score"
                ]
            )
        )
    )


    reproduction_pass = bool(
        good_match_mismatch
        == 0
        and
        inlier_mismatch
        == 0
        and
        homography_mismatch
        == 0
        and
        max_ratio_diff
        < 1e-9
        and
        max_query_cov_diff
        < 1e-9
        and
        max_sat_cov_diff
        < 1e-9
        and
        max_verifier_score_diff
        < 1e-9
    )


    blind_path = (
        out_dir
        / "r4_11_blind_subtile_projection_pairs.csv"
    )


    freeze_manifest_path = (
        out_dir
        / "r4_11_blind_subtile_projection_freeze_manifest.json"
    )


    blind.to_csv(
        blind_path,
        index=False,
    )


    freeze_manifest = {
        "stage":
            "R4.11_BLIND_SUBTILE_PROJECTION_FREEZE",

        "configuration": {
            "preprocess":
                PREPROCESS,

            "resize_long":
                RESIZE_LONG,

            "nfeatures":
                NFEATURES,

            "fast_threshold":
                FAST_THRESHOLD,

            "edge_threshold":
                EDGE_THRESHOLD,

            "patch_size":
                PATCH_SIZE,

            "lowe_ratio":
                LOWE_RATIO,

            "ransac_threshold_px":
                RANSAC_THRESHOLD,

            "homography_direction":
                (
                    "query pixel -> "
                    "satellite tile pixel"
                ),

            "query_center_definition":
                (
                    "(processed_width/2, "
                    "processed_height/2)"
                ),

            "tile_pixel_to_map_convention":
                (
                    "pixel-center coordinates "
                    "with tile bounds interpreted "
                    "as raster outer edges"
                ),
        },

        "input": {
            "candidate_csv":
                str(
                    candidate_csv
                ),

            "candidate_csv_sha256":
                sha256(
                    candidate_csv
                ),

            "verifier_source":
                str(
                    verifier_source
                ),

            "verifier_source_sha256":
                (
                    sha256(
                        verifier_source
                    )
                    if verifier_source.exists()
                    else None
                ),

            "prefix_max_query":
                max_query,
        },

        "counts": {
            "pairs":
                int(
                    len(
                        blind
                    )
                ),

            "unique_queries":
                int(
                    blind[
                        "query_id"
                    ].nunique()
                ),

            "unique_tiles":
                int(
                    blind[
                        "tile_id"
                    ].nunique()
                ),

            "recomputed_homography_ok":
                int(
                    blind[
                        "recomputed_homography_ok"
                    ].sum()
                ),

            "projected_inside_tile":
                int(
                    blind[
                        "projected_inside_tile"
                    ].sum()
                ),
        },

        "reproduction_gate": {
            "pass":
                reproduction_pass,

            "good_match_mismatch_rows":
                good_match_mismatch,

            "inlier_mismatch_rows":
                inlier_mismatch,

            "homography_mismatch_rows":
                homography_mismatch,

            "max_inlier_ratio_abs_diff":
                max_ratio_diff,

            "max_query_coverage_abs_diff":
                max_query_cov_diff,

            "max_sat_coverage_abs_diff":
                max_sat_cov_diff,

            "max_verifier_score_abs_diff":
                max_verifier_score_diff,
        },

        "blind_contract": {
            "gt_used":
                False,

            "reference_used":
                False,

            "srt_used":
                False,

            "gps_used":
                False,

            "orb_rerun":
                True,

            "r3_modified":
                False,
        },

        "blind_output": {
            "csv":
                str(
                    blind_path
                ),

            "sha256":
                sha256(
                    blind_path
                ),
        },
    }


    freeze_manifest_path.write_text(
        json.dumps(
            freeze_manifest,
            indent=2,
        )
    )


    freeze_sha = sha256(
        freeze_manifest_path
    )


    print()
    print("=" * 112)
    print(
        "R4.11 PHASE A — "
        "BLIND SUB-TILE PROJECTION FROZEN"
    )
    print("=" * 112)


    print(
        "pairs:",
        len(
            blind
        ),
    )


    print(
        "queries:",
        blind[
            "query_id"
        ].nunique(),
    )


    print(
        "unique satellite tiles:",
        blind[
            "tile_id"
        ].nunique(),
    )


    print()

    print(
        "query processed shapes:"
    )

    print(
        blind[
            [
                "query_processed_width",
                "query_processed_height",
            ]
        ]
        .drop_duplicates()
        .to_string(
            index=False
        )
    )


    print()

    print(
        "tile processed shapes:"
    )

    print(
        blind[
            [
                "tile_processed_width",
                "tile_processed_height",
            ]
        ]
        .drop_duplicates()
        .to_string(
            index=False
        )
    )


    print()
    print("-" * 112)
    print("ORB REPRODUCTION GATE")
    print("-" * 112)


    print(
        "good-match mismatches:",
        good_match_mismatch,
    )


    print(
        "inlier mismatches:",
        inlier_mismatch,
    )


    print(
        "homography-ok mismatches:",
        homography_mismatch,
    )


    print(
        "max inlier-ratio diff:",
        max_ratio_diff,
    )


    print(
        "max query-coverage diff:",
        max_query_cov_diff,
    )


    print(
        "max satellite-coverage diff:",
        max_sat_cov_diff,
    )


    print(
        "max verifier-score diff:",
        max_verifier_score_diff,
    )


    print(
        "REPRODUCTION:",
        (
            "PASS"
            if reproduction_pass
            else "FAIL"
        ),
    )


    print()
    print("-" * 112)
    print("BLIND GEOMETRIC OUTPUT")
    print("-" * 112)


    print(
        "homography OK:",
        int(
            blind[
                "recomputed_homography_ok"
            ].sum()
        ),
        "/",
        len(
            blind
        ),
    )


    print(
        "projected centre inside tile:",
        int(
            blind[
                "projected_inside_tile"
            ].sum()
        ),
        "/",
        len(
            blind
        ),
    )


    print(
        "median inlier reprojection RMSE:",
        f"{blind['inlier_reprojection_rmse_px'].median():.3f} px",
    )


    print(
        "blind freeze SHA256:",
        freeze_sha,
    )


    if not reproduction_pass:

        print()
        print(
            "STATUS: "
            "FAIL_R4_11_ORB_REPRODUCTION"
        )

        print(
            "GT attachment intentionally skipped."
        )

        return


    # ========================================================
    #
    # PHASE B — GT FIRST READ OCCURS HERE
    #
    # ========================================================

    reference_path = (
        run
        / "evaluation/"
          "reference_attachment.csv"
    )


    reference = pd.read_csv(
        reference_path
    )


    required_reference = {
        "query_id",
        "eval_ref_lat",
        "eval_ref_lon",
    }


    missing = (
        required_reference
        - set(
            reference.columns
        )
    )


    if missing:

        raise RuntimeError(
            "Reference attachment missing: "
            + str(
                sorted(
                    missing
                )
            )
        )


    transformer = (
        Transformer.from_crs(
            "EPSG:4326",
            "EPSG:3346",
            always_xy=True,
        )
    )


    gt_e, gt_n = transformer.transform(
        reference[
            "eval_ref_lon"
        ].to_numpy(float),

        reference[
            "eval_ref_lat"
        ].to_numpy(float),
    )


    reference[
        "gt_easting"
    ] = gt_e


    reference[
        "gt_northing"
    ] = gt_n


    reference[
        "query_id"
    ] = pd.to_numeric(
        reference[
            "query_id"
        ],
        errors="raise",
    ).astype(int)


    evaluated = (
        blind
        .merge(
            reference[
                [
                    "query_id",
                    "gt_easting",
                    "gt_northing",
                ]
            ],
            on="query_id",
            how="left",
            validate="many_to_one",
        )
    )


    evaluated[
        "tile_center_error_m"
    ] = np.hypot(
        evaluated[
            "center_easting"
        ]
        -
        evaluated[
            "gt_easting"
        ],

        evaluated[
            "center_northing"
        ]
        -
        evaluated[
            "gt_northing"
        ],
    )


    evaluated[
        "projected_error_m"
    ] = np.hypot(
        evaluated[
            "projected_easting"
        ]
        -
        evaluated[
            "gt_easting"
        ],

        evaluated[
            "projected_northing"
        ]
        -
        evaluated[
            "gt_northing"
        ],
    )


    evaluated[
        "projected_minus_center_error_m"
    ] = (
        evaluated[
            "projected_error_m"
        ]
        -
        evaluated[
            "tile_center_error_m"
        ]
    )


    evaluated[
        "gt_inside_tile"
    ] = (
        (
            evaluated[
                "gt_easting"
            ]
            >=
            evaluated[
                "left_easting"
            ]
        )
        &
        (
            evaluated[
                "gt_easting"
            ]
            <=
            evaluated[
                "right_easting"
            ]
        )
        &
        (
            evaluated[
                "gt_northing"
            ]
            >=
            evaluated[
                "bottom_northing"
            ]
        )
        &
        (
            evaluated[
                "gt_northing"
            ]
            <=
            evaluated[
                "top_northing"
            ]
        )
    )


    # ========================================================
    # Evaluation-only groups
    # ========================================================

    groups = {
        "all_homography_ok":
            evaluated[
                evaluated[
                    "recomputed_homography_ok"
                ]
            ],

        "projected_inside_tile":
            evaluated[
                evaluated[
                    "projected_inside_tile"
                ]
            ],

        "gt_inside_tile_eval_only":
            evaluated[
                evaluated[
                    "gt_inside_tile"
                ]
            ],

        "gt_inside_and_projection_inside_eval_only":
            evaluated[
                evaluated[
                    "gt_inside_tile"
                ]
                &
                evaluated[
                    "projected_inside_tile"
                ]
            ],

        "hybrid_top4":
            evaluated[
                evaluated[
                    "hybrid_rank"
                ]
                <= 4
            ],

        "hybrid_top5":
            evaluated[
                evaluated[
                    "hybrid_rank"
                ]
                <= 5
            ],

        "dino_top5":
            evaluated[
                evaluated[
                    "dino_rank"
                ]
                <= 5
            ],

        "center_error_le_51_2m_eval_only":
            evaluated[
                evaluated[
                    "tile_center_error_m"
                ]
                <= 51.2
            ],
    }


    summary_rows = []


    for name, group in (
        groups.items()
    ):

        stats = error_summary(
            group
        )


        summary_rows.append(
            {
                "group":
                    name,

                **stats,
            }
        )


    summaries = pd.DataFrame(
        summary_rows
    )


    # ========================================================
    # Query-level hybrid Top-1 comparison
    # ========================================================

    hybrid_top1 = (
        evaluated
        .sort_values(
            [
                "query_id",
                "hybrid_rank",
            ]
        )
        .groupby(
            "query_id",
            as_index=False,
        )
        .first()
    )


    hybrid_top1_summary = (
        error_summary(
            hybrid_top1
        )
    )


    # ========================================================
    # Save
    # ========================================================

    evaluated_path = (
        out_dir
        / "r4_11_postfreeze_subtile_projection_eval.csv"
    )


    summary_path = (
        out_dir
        / "r4_11_subtile_projection_error_summary.csv"
    )


    report_path = (
        out_dir
        / "r4_11_subtile_projection_recompute.json"
    )


    evaluated.to_csv(
        evaluated_path,
        index=False,
    )


    summaries.to_csv(
        summary_path,
        index=False,
    )


    report = {
        "stage":
            "R4.11_SUBTILE_PROJECTION_RECOMPUTE",

        "status":
            "PASS_R4_11_SUBTILE_PROJECTION_RECOMPUTE_EXECUTION",

        "blind_freeze_manifest_sha256":
            freeze_sha,

        "reproduction_gate":
            freeze_manifest[
                "reproduction_gate"
            ],

        "hybrid_top1_postfreeze_comparison":
            hybrid_top1_summary,

        "contract": {
            "phase_a_used_gt":
                False,

            "phase_a_used_reference":
                False,

            "gt_loaded_after_blind_projection_freeze":
                True,

            "r3_modified":
                False,

            "existing_candidate_ranking_modified":
                False,
        },

        "outputs": {
            "blind_projection_pairs":
                str(
                    blind_path
                ),

            "blind_freeze_manifest":
                str(
                    freeze_manifest_path
                ),

            "postfreeze_pair_evaluation":
                str(
                    evaluated_path
                ),

            "error_summary":
                str(
                    summary_path
                ),
        },
    }


    report_path.write_text(
        json.dumps(
            report,
            indent=2,
        )
    )


    # ========================================================
    # Print post-freeze evidence
    # ========================================================

    print()
    print("=" * 112)
    print(
        "R4.11 PHASE B — "
        "POST-FREEZE SUB-TILE PROJECTION EVALUATION"
    )
    print("=" * 112)


    print(
        summaries.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.3f}",
        )
    )


    print()
    print("-" * 112)
    print("HYBRID TOP-1 QUERY-LEVEL COMPARISON")
    print("-" * 112)


    for key, value in (
        hybrid_top1_summary.items()
    ):

        if isinstance(
            value,
            float,
        ):

            print(
                f"{key}: "
                f"{value:.3f}"
            )

        else:

            print(
                f"{key}: "
                f"{value}"
            )


    print()
    print("=" * 112)
    print("R4.11 OUTPUT")
    print("=" * 112)


    print(
        "blind projection pairs:",
        blind_path,
    )


    print(
        "blind freeze manifest:",
        freeze_manifest_path,
    )


    print(
        "postfreeze pair evaluation:",
        evaluated_path,
    )


    print(
        "summary:",
        summary_path,
    )


    print(
        "report:",
        report_path,
    )


    print()


    print(
        "STATUS: "
        "PASS_R4_11_SUBTILE_PROJECTION_RECOMPUTE_EXECUTION"
    )


if __name__ == "__main__":
    main()
