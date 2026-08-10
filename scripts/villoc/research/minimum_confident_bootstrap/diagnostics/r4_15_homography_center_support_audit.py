#!/usr/bin/env python3
"""
R4.15 — homography centre-support reliability audit.

Motivation
----------
R4.14 showed that increasing Top-M does not recover q38.
The correct geographic tile exists, but its homography-derived query-centre
projection can still be inaccurate.

Hypothesis
----------
RANSAC reprojection error alone does not characterize whether the QUERY IMAGE
CENTRE is geometrically supported by the inlier correspondences.

PHASE A — BLIND
----------------
Recompute the exact frozen ORB verifier for q1..q38 Top-20 and measure:

Query-side:
    centre inside inlier convex hull
    signed centre-to-hull distance
    nearest inlier distance
    centre-to-inlier-centroid distance
    convex hull area fraction
    angular coverage around query centre

Satellite-side:
    projected centre inside satellite inlier convex hull
    signed projected-centre-to-hull distance
    nearest satellite inlier distance
    satellite hull area fraction

Homography local geometry:
    local Jacobian singular values
    Jacobian anisotropy / condition number
    local area-scale determinant
    projective denominator

No GT/reference/SRT/GPS is read.

Freeze + hash.

PHASE B — POST-FREEZE
------------------------
Attach R4.11 evaluation labels and determine whether any BLIND support
descriptor correlates with continuous projected-point accuracy.

No thresholds or production policy are created.

Command:

SCRIPT=scripts/villoc/research/minimum_confident_bootstrap/diagnostics/r4_15_homography_center_support_audit.py
RUN=outputs/demo_runs/traj01_blind_regression_001
R3=outputs/research_runs/minimum_confident_bootstrap/traj01_blind_r3_001
python "$SCRIPT" \
  --repo-root "$PWD" \
  --run-root "$RUN" \
  --research-root "$R3" \
  --prefix-max-query 38 \
  2>&1 | tee \
  "$R3/postfreeze_eval/r4_15_homography_center_support_audit.log"

"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


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


def load_r411_module(repo: Path):

    source = (
        repo
        / "scripts/villoc/research/"
          "minimum_confident_bootstrap/"
          "diagnostics/"
          "r4_11_blind_subtile_projection_recompute.py"
    )

    if not source.exists():

        raise RuntimeError(
            f"Missing R4.11 source: {source}"
        )

    spec = importlib.util.spec_from_file_location(
        "r411_module",
        source,
    )

    if (
        spec is None
        or spec.loader is None
    ):

        raise RuntimeError(
            "Could not import R4.11."
        )

    module = importlib.util.module_from_spec(
        spec
    )

    # Register the dynamically loaded module before execution.
    #
    # @dataclass inside R4.11 resolves type information through
    # sys.modules[cls.__module__]. Without this registration,
    # Python 3.10 raises:
    #
    #   AttributeError: 'NoneType' object has no attribute '__dict__'
    #
    sys.modules[
        spec.name
    ] = module

    spec.loader.exec_module(
        module
    )

    return (
        module,
        source,
    )


def convex_hull_metrics(
    points: np.ndarray,
    target_xy,
    image_hw,
):
    """
    All metrics are BLIND image-plane geometry.

    signed_hull_distance_px:
        positive  -> target inside hull
        zero      -> on hull
        negative  -> outside hull
    """

    points = np.asarray(
        points,
        dtype=np.float32,
    ).reshape(
        -1,
        2,
    )

    target = np.asarray(
        target_xy,
        dtype=float,
    ).reshape(
        2,
    )

    h, w = image_hw

    diag = float(
        np.hypot(
            w,
            h,
        )
    )

    if len(points) == 0:

        return {
            "hull_area_fraction":
                0.0,

            "target_inside_hull":
                False,

            "signed_hull_distance_px":
                math.nan,

            "signed_hull_distance_norm":
                math.nan,

            "nearest_point_distance_px":
                math.nan,

            "nearest_point_distance_norm":
                math.nan,

            "centroid_distance_px":
                math.nan,

            "centroid_distance_norm":
                math.nan,

            "angular_coverage_fraction":
                0.0,

            "max_angular_gap_deg":
                360.0,
        }


    # --------------------------------------------------------
    # Point-cloud distances
    # --------------------------------------------------------

    distances = np.linalg.norm(
        points.astype(float)
        - target[
            None,
            :
        ],
        axis=1,
    )


    nearest = float(
        distances.min()
    )


    centroid = points.mean(
        axis=0
    ).astype(float)


    centroid_distance = float(
        np.linalg.norm(
            centroid
            - target
        )
    )


    # --------------------------------------------------------
    # Angular support around target.
    #
    # Full surrounding support => small maximum angular gap.
    # One-sided support         => large maximum angular gap.
    # --------------------------------------------------------

    vectors = (
        points.astype(float)
        - target[
            None,
            :
        ]
    )


    nonzero = (
        np.linalg.norm(
            vectors,
            axis=1,
        )
        > 1e-9
    )


    vectors = vectors[
        nonzero
    ]


    if len(vectors) >= 2:

        angles = np.mod(
            np.arctan2(
                vectors[:, 1],
                vectors[:, 0],
            ),
            2.0
            * np.pi,
        )


        angles = np.sort(
            angles
        )


        wrapped = np.concatenate(
            [
                angles,
                [
                    angles[0]
                    + 2.0
                    * np.pi
                ],
            ]
        )


        gaps = np.diff(
            wrapped
        )


        max_gap = float(
            gaps.max()
        )


        angular_coverage = float(
            (
                2.0
                * np.pi
                - max_gap
            )
            /
            (
                2.0
                * np.pi
            )
        )


        max_gap_deg = float(
            np.degrees(
                max_gap
            )
        )

    else:

        angular_coverage = 0.0
        max_gap_deg = 360.0


    # --------------------------------------------------------
    # Convex hull
    # --------------------------------------------------------

    if len(points) >= 3:

        hull = cv2.convexHull(
            points
        )


        hull_area = float(
            cv2.contourArea(
                hull
            )
        )


        hull_fraction = float(
            hull_area
            / float(
                w * h
            )
        )


        signed_distance = float(
            cv2.pointPolygonTest(
                hull,
                (
                    float(
                        target[0]
                    ),
                    float(
                        target[1]
                    ),
                ),
                True,
            )
        )


        inside = bool(
            signed_distance
            >= 0.0
        )

    else:

        hull_fraction = 0.0
        signed_distance = math.nan
        inside = False


    return {
        "hull_area_fraction":
            hull_fraction,

        "target_inside_hull":
            inside,

        "signed_hull_distance_px":
            signed_distance,

        "signed_hull_distance_norm":
            (
                float(
                    signed_distance
                    / diag
                )
                if np.isfinite(
                    signed_distance
                )
                else math.nan
            ),

        "nearest_point_distance_px":
            nearest,

        "nearest_point_distance_norm":
            float(
                nearest
                / diag
            ),

        "centroid_distance_px":
            centroid_distance,

        "centroid_distance_norm":
            float(
                centroid_distance
                / diag
            ),

        "angular_coverage_fraction":
            angular_coverage,

        "max_angular_gap_deg":
            max_gap_deg,
    }


def homography_jacobian(
    H: np.ndarray,
    u: float,
    v: float,
):

    H = np.asarray(
        H,
        dtype=float,
    )


    a, b, c = H[0]
    d, e, f = H[1]
    g, h, i = H[2]


    denominator = (
        g * u
        + h * v
        + i
    )


    numerator_u = (
        a * u
        + b * v
        + c
    )


    numerator_v = (
        d * u
        + e * v
        + f
    )


    if (
        not np.isfinite(
            denominator
        )
        or abs(
            denominator
        )
        <= 1e-12
    ):

        return {
            "jacobian_smax":
                math.nan,

            "jacobian_smin":
                math.nan,

            "jacobian_condition":
                math.inf,

            "jacobian_det_abs":
                math.nan,

            "jacobian_local_area_scale":
                math.nan,

            "projective_denominator_abs":
                abs(
                    float(
                        denominator
                    )
                ),
        }


    denominator_sq = (
        denominator
        ** 2
    )


    J = np.asarray(
        [
            [
                (
                    a * denominator
                    - numerator_u * g
                )
                / denominator_sq,

                (
                    b * denominator
                    - numerator_u * h
                )
                / denominator_sq,
            ],

            [
                (
                    d * denominator
                    - numerator_v * g
                )
                / denominator_sq,

                (
                    e * denominator
                    - numerator_v * h
                )
                / denominator_sq,
            ],
        ],
        dtype=float,
    )


    singular = np.linalg.svd(
        J,
        compute_uv=False,
    )


    smax = float(
        singular.max()
    )


    smin = float(
        singular.min()
    )


    condition = (
        float(
            smax / smin
        )
        if smin > 1e-12
        else math.inf
    )


    determinant = float(
        abs(
            np.linalg.det(
                J
            )
        )
    )


    return {
        "jacobian_smax":
            smax,

        "jacobian_smin":
            smin,

        "jacobian_condition":
            condition,

        "jacobian_det_abs":
            determinant,

        "jacobian_local_area_scale":
            float(
                math.sqrt(
                    determinant
                )
            ),

        "projective_denominator_abs":
            abs(
                float(
                    denominator
                )
            ),
    }


# ============================================================
# Exact match reconstruction
# ============================================================

def pair_geometry(
    q,
    s,
    module,
):
    """
    Recreate exact R4.11 matching while retaining inlier point geometry.
    """

    if (
        not q.ok
        or not s.ok
        or q.descriptors is None
        or s.descriptors is None
    ):

        return None


    matcher = cv2.BFMatcher(
        cv2.NORM_HAMMING,
        crossCheck=False,
    )


    knn = matcher.knnMatch(
        q.descriptors,
        s.descriptors,
        k=2,
    )


    good = []


    for pair in knn:

        if len(pair) < 2:
            continue

        m, n = pair

        if (
            m.distance
            <
            module.LOWE_RATIO
            * n.distance
        ):

            good.append(
                m
            )


    if len(good) < 4:

        return None


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
        module.RANSAC_THRESHOLD,
    )


    if (
        H is None
        or mask is None
    ):

        return None


    inlier_mask = (
        mask.ravel()
        .astype(bool)
    )


    inliers = int(
        inlier_mask.sum()
    )


    if inliers < 4:

        return None


    q_inliers = q_pts[
        inlier_mask
    ]


    s_inliers = s_pts[
        inlier_mask
    ]


    qh, qw = (
        q.image_shape
    )


    query_center = np.asarray(
        [
            float(
                qw
            )
            / 2.0,

            float(
                qh
            )
            / 2.0,
        ],
        dtype=float,
    )


    projected_center = (
        cv2.perspectiveTransform(
            np.float32(
                [
                    [
                        query_center
                    ]
                ]
            ),
            H,
        )
        .reshape(
            2,
        )
        .astype(float)
    )


    q_support = convex_hull_metrics(
        q_inliers,
        query_center,
        q.image_shape,
    )


    s_support = convex_hull_metrics(
        s_inliers,
        projected_center,
        s.image_shape,
    )


    jacobian = homography_jacobian(
        H,
        float(
            query_center[0]
        ),
        float(
            query_center[1]
        ),
    )


    projected_inliers = (
        cv2.perspectiveTransform(
            q_inliers.reshape(
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


    reprojection_residual = (
        projected_inliers
        - s_inliers
    )


    reprojection_rmse = float(
        np.sqrt(
            np.mean(
                np.sum(
                    reprojection_residual
                    ** 2,
                    axis=1,
                )
            )
        )
    )


    return {
        "good_matches":
            int(
                len(
                    good
                )
            ),

        "inliers":
            inliers,

        "query_center_u_px":
            float(
                query_center[0]
            ),

        "query_center_v_px":
            float(
                query_center[1]
            ),

        "projected_tile_u_px":
            float(
                projected_center[0]
            ),

        "projected_tile_v_px":
            float(
                projected_center[1]
            ),

        "reprojection_rmse_px":
            reprojection_rmse,

        # Query-side support.
        "q_center_inside_inlier_hull":
            q_support[
                "target_inside_hull"
            ],

        "q_hull_area_fraction":
            q_support[
                "hull_area_fraction"
            ],

        "q_center_hull_signed_distance_px":
            q_support[
                "signed_hull_distance_px"
            ],

        "q_center_hull_signed_distance_norm":
            q_support[
                "signed_hull_distance_norm"
            ],

        "q_center_nearest_inlier_distance_px":
            q_support[
                "nearest_point_distance_px"
            ],

        "q_center_nearest_inlier_distance_norm":
            q_support[
                "nearest_point_distance_norm"
            ],

        "q_center_centroid_distance_px":
            q_support[
                "centroid_distance_px"
            ],

        "q_center_centroid_distance_norm":
            q_support[
                "centroid_distance_norm"
            ],

        "q_center_angular_coverage_fraction":
            q_support[
                "angular_coverage_fraction"
            ],

        "q_center_max_angular_gap_deg":
            q_support[
                "max_angular_gap_deg"
            ],

        # Satellite-side support.
        "s_projected_inside_inlier_hull":
            s_support[
                "target_inside_hull"
            ],

        "s_hull_area_fraction":
            s_support[
                "hull_area_fraction"
            ],

        "s_projected_hull_signed_distance_px":
            s_support[
                "signed_hull_distance_px"
            ],

        "s_projected_hull_signed_distance_norm":
            s_support[
                "signed_hull_distance_norm"
            ],

        "s_projected_nearest_inlier_distance_px":
            s_support[
                "nearest_point_distance_px"
            ],

        "s_projected_nearest_inlier_distance_norm":
            s_support[
                "nearest_point_distance_norm"
            ],

        "s_projected_centroid_distance_px":
            s_support[
                "centroid_distance_px"
            ],

        "s_projected_centroid_distance_norm":
            s_support[
                "centroid_distance_norm"
            ],

        "s_projected_angular_coverage_fraction":
            s_support[
                "angular_coverage_fraction"
            ],

        "s_projected_max_angular_gap_deg":
            s_support[
                "max_angular_gap_deg"
            ],

        **jacobian,
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


    out = (
        research
        / "postfreeze_eval"
    )


    module, r411_source = (
        load_r411_module(
            repo
        )
    )


    candidate_path = (
        run
        / "reports/"
          "s8_12e1_top20_verifier_reranker/"
          "512_s256_orb_hybrid_top20_img518/"
          "s8_12e1_all_candidate_verifier_scores.csv"
    )


    r411_blind_path = (
        out
        / "r4_11_blind_subtile_projection_pairs.csv"
    )


    r411_eval_path = (
        out
        / "r4_11_postfreeze_subtile_projection_eval.csv"
    )


    candidate = pd.read_csv(
        candidate_path
    )


    r411_blind = pd.read_csv(
        r411_blind_path
    )


    candidate[
        "query_id"
    ] = pd.to_numeric(
        candidate[
            "query_id"
        ],
        errors="raise",
    ).astype(int)


    r411_blind[
        "query_id"
    ] = pd.to_numeric(
        r411_blind[
            "query_id"
        ],
        errors="raise",
    ).astype(int)


    prefix = (
        candidate[
            candidate[
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


    # ========================================================
    #
    # PHASE A — BLIND
    #
    # ========================================================

    detector = (
        module.create_detector()
    )


    feature_cache = {}


    def features(path_value):

        path = Path(
            str(
                path_value
            )
        )


        if not path.is_absolute():

            path = (
                repo
                / path
            )


        path = path.resolve()

        key = str(
            path
        )


        if key not in feature_cache:

            feature_cache[
                key
            ] = (
                module.compute_features(
                    path,
                    detector,
                )
            )


        return feature_cache[
            key
        ]


    rows = []


    for index, row in prefix.iterrows():

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


        geom = pair_geometry(
            q,
            s,
            module,
        )


        if geom is None:

            raise RuntimeError(
                "Unexpected failed geometry for "
                f"q{row['query_id']} "
                f"{row['tile_id']}"
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

                "hybrid_rank":
                    int(
                        row[
                            "hybrid_rank"
                        ]
                    ),

                "dino_rank":
                    int(
                        row[
                            "rank"
                        ]
                    ),

                "stored_inliers":
                    int(
                        row[
                            "inliers"
                        ]
                    ),

                **geom,
            }
        )


        if (
            index
            + 1
        ) % 100 == 0:

            print(
                "processed:",
                index + 1,
                "/",
                len(
                    prefix
                ),
            )


    blind = pd.DataFrame(
        rows
    )


    # ========================================================
    # Reproduction check against R4.11
    # ========================================================

    check = blind.merge(
        r411_blind[
            [
                "query_id",
                "tile_id",
                "recomputed_inliers",
                "projected_tile_u_px",
                "projected_tile_v_px",
            ]
        ],
        on=[
            "query_id",
            "tile_id",
        ],
        validate="one_to_one",
    )


    inlier_mismatch = int(
        (
            check[
                "inliers"
            ]
            !=
            check[
                "recomputed_inliers"
            ]
        ).sum()
    )


    max_u_diff = float(
        np.nanmax(
            np.abs(
                check[
                    "projected_tile_u_px_x"
                ]
                -
                check[
                    "projected_tile_u_px_y"
                ]
            )
        )
    )


    max_v_diff = float(
        np.nanmax(
            np.abs(
                check[
                    "projected_tile_v_px_x"
                ]
                -
                check[
                    "projected_tile_v_px_y"
                ]
            )
        )
    )


    reproduction_pass = bool(
        inlier_mismatch == 0
        and max_u_diff < 1e-5
        and max_v_diff < 1e-5
    )


    blind_path = (
        out
        / "r4_15_blind_homography_center_support.csv"
    )


    freeze_path = (
        out
        / "r4_15_blind_homography_center_support_freeze_manifest.json"
    )


    blind.to_csv(
        blind_path,
        index=False,
    )


    freeze = {
        "stage":
            "R4.15_BLIND_HOMOGRAPHY_CENTER_SUPPORT_FREEZE",

        "input": {
            "candidate_csv_sha256":
                sha256(
                    candidate_path
                ),

            "r4_11_source_sha256":
                sha256(
                    r411_source
                ),

            "r4_11_blind_projection_sha256":
                sha256(
                    r411_blind_path
                ),
        },

        "counts": {
            "pairs":
                int(
                    len(
                        blind
                    )
                ),

            "queries":
                int(
                    blind[
                        "query_id"
                    ].nunique()
                ),
        },

        "reproduction_gate": {
            "pass":
                reproduction_pass,

            "inlier_mismatch":
                inlier_mismatch,

            "max_projected_u_diff_px":
                max_u_diff,

            "max_projected_v_diff_px":
                max_v_diff,
        },

        "blind_contract": {
            "gt_used":
                False,

            "reference_used":
                False,

            "threshold_created":
                False,

            "r3_modified":
                False,
        },

        "output": {
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


    freeze_path.write_text(
        json.dumps(
            freeze,
            indent=2,
        )
    )


    freeze_sha = sha256(
        freeze_path
    )


    print()
    print("=" * 116)
    print(
        "R4.15 PHASE A — "
        "BLIND HOMOGRAPHY CENTRE SUPPORT FROZEN"
    )
    print("=" * 116)


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
        "inlier mismatches vs R4.11:",
        inlier_mismatch,
    )


    print(
        "max projected-u difference:",
        max_u_diff,
    )


    print(
        "max projected-v difference:",
        max_v_diff,
    )


    print(
        "REPRODUCTION:",
        (
            "PASS"
            if reproduction_pass
            else "FAIL"
        ),
    )


    print(
        "query centre inside inlier hull:",
        int(
            blind[
                "q_center_inside_inlier_hull"
            ].sum()
        ),
        "/",
        len(
            blind
        ),
    )


    print(
        "projected satellite point "
        "inside satellite inlier hull:",
        int(
            blind[
                "s_projected_inside_inlier_hull"
            ].sum()
        ),
        "/",
        len(
            blind
        ),
    )


    print(
        "blind freeze SHA256:",
        freeze_sha,
    )


    if not reproduction_pass:

        print(
            "STATUS: "
            "FAIL_R4_15_REPRODUCTION"
        )

        return


    # ========================================================
    #
    # PHASE B — GT FIRST READ
    #
    # ========================================================

    evaluation = pd.read_csv(
        r411_eval_path
    )


    evaluation[
        "query_id"
    ] = pd.to_numeric(
        evaluation[
            "query_id"
        ],
        errors="raise",
    ).astype(int)


    labels = evaluation[
        [
            "query_id",
            "tile_id",
            "projected_error_m",
            "tile_center_error_m",
            "gt_inside_tile",
            "projected_inside_tile",
        ]
    ].copy()


    annotated = blind.merge(
        labels,
        on=[
            "query_id",
            "tile_id",
        ],
        validate="one_to_one",
    )


    # ========================================================
    # Feature audit restricted to candidates whose tile
    # actually contains GT. This isolates projection reliability
    # from retrieval/geographic-aliasing failure.
    # ========================================================

    correct_region = annotated[
        annotated[
            "gt_inside_tile"
        ]
        .astype(bool)
    ].copy()


    features = [
        "inliers",

        "q_hull_area_fraction",
        "q_center_hull_signed_distance_norm",
        "q_center_nearest_inlier_distance_norm",
        "q_center_centroid_distance_norm",
        "q_center_angular_coverage_fraction",
        "q_center_max_angular_gap_deg",

        "s_hull_area_fraction",
        "s_projected_hull_signed_distance_norm",
        "s_projected_nearest_inlier_distance_norm",
        "s_projected_centroid_distance_norm",
        "s_projected_angular_coverage_fraction",
        "s_projected_max_angular_gap_deg",

        "jacobian_smax",
        "jacobian_smin",
        "jacobian_condition",
        "jacobian_det_abs",
        "jacobian_local_area_scale",
        "projective_denominator_abs",

        "reprojection_rmse_px",
    ]


    audit_rows = []


    for feature in features:

        pair_data = (
            correct_region[
                [
                    feature,
                    "projected_error_m",
                ]
            ]
            .replace(
                [
                    np.inf,
                    -np.inf,
                ],
                np.nan,
            )
            .dropna()
        )


        if len(
            pair_data
        ) >= 3:

            spearman = float(
                pair_data[
                    feature
                ].corr(
                    pair_data[
                        "projected_error_m"
                    ],
                    method="spearman",
                )
            )

        else:

            spearman = math.nan


        audit_rows.append(
            {
                "feature":
                    feature,

                "correct_region_pairs":
                    int(
                        len(
                            pair_data
                        )
                    ),

                "spearman_vs_projected_error":
                    spearman,

                "abs_spearman":
                    (
                        abs(
                            spearman
                        )
                        if np.isfinite(
                            spearman
                        )
                        else math.nan
                    ),
            }
        )


    feature_audit = (
        pd.DataFrame(
            audit_rows
        )
        .sort_values(
            "abs_spearman",
            ascending=False,
        )
    )


    # ========================================================
    # q36 / q37 / q38 current-frame inspection
    # ========================================================

    late = (
        annotated[
            annotated[
                "query_id"
            ].isin(
                [
                    36,
                    37,
                    38,
                ]
            )
        ]
        .sort_values(
            [
                "query_id",
                "projected_error_m",
            ]
        )
    )


    late_correct = (
        late[
            late[
                "gt_inside_tile"
            ]
            .astype(bool)
        ]
        .copy()
    )


    # ========================================================
    # Evaluation-only descriptive groups.
    #
    # Not thresholds proposed for online use.
    # ========================================================

    region_good = correct_region[
        correct_region[
            "projected_error_m"
        ]
        <= 10.0
    ]


    region_bad = correct_region[
        correct_region[
            "projected_error_m"
        ]
        >= 25.0
    ]


    comparison_features = [
        "inliers",
        "q_hull_area_fraction",
        "q_center_hull_signed_distance_norm",
        "q_center_nearest_inlier_distance_norm",
        "q_center_centroid_distance_norm",
        "q_center_angular_coverage_fraction",
        "s_hull_area_fraction",
        "s_projected_hull_signed_distance_norm",
        "jacobian_condition",
        "jacobian_local_area_scale",
        "reprojection_rmse_px",
    ]


    comparison_rows = []


    for feature in comparison_features:

        comparison_rows.append(
            {
                "feature":
                    feature,

                "good_le10_count":
                    int(
                        region_good[
                            feature
                        ]
                        .replace(
                            [
                                np.inf,
                                -np.inf,
                            ],
                            np.nan,
                        )
                        .notna()
                        .sum()
                    ),

                "good_le10_median":
                    float(
                        region_good[
                            feature
                        ]
                        .replace(
                            [
                                np.inf,
                                -np.inf,
                            ],
                            np.nan,
                        )
                        .median()
                    ),

                "bad_ge25_count":
                    int(
                        region_bad[
                            feature
                        ]
                        .replace(
                            [
                                np.inf,
                                -np.inf,
                            ],
                            np.nan,
                        )
                        .notna()
                        .sum()
                    ),

                "bad_ge25_median":
                    float(
                        region_bad[
                            feature
                        ]
                        .replace(
                            [
                                np.inf,
                                -np.inf,
                            ],
                            np.nan,
                        )
                        .median()
                    ),
            }
        )


    comparison = pd.DataFrame(
        comparison_rows
    )


    # ========================================================
    # Save
    # ========================================================

    annotated_path = (
        out
        / "r4_15_gt_annotated_homography_center_support.csv"
    )


    feature_path = (
        out
        / "r4_15_center_support_feature_audit.csv"
    )


    comparison_path = (
        out
        / "r4_15_good_bad_projection_support_comparison.csv"
    )


    report_path = (
        out
        / "r4_15_homography_center_support_audit.json"
    )


    annotated.to_csv(
        annotated_path,
        index=False,
    )


    feature_audit.to_csv(
        feature_path,
        index=False,
    )


    comparison.to_csv(
        comparison_path,
        index=False,
    )


    report = {
        "stage":
            "R4.15_HOMOGRAPHY_CENTER_SUPPORT_AUDIT",

        "status":
            "PASS_R4_15_HOMOGRAPHY_CENTER_SUPPORT_AUDIT_EXECUTION",

        "blind_freeze_manifest_sha256":
            freeze_sha,

        "counts": {
            "all_pairs":
                int(
                    len(
                        annotated
                    )
                ),

            "gt_inside_tile_eval_only_pairs":
                int(
                    len(
                        correct_region
                    )
                ),

            "eval_good_le10m":
                int(
                    len(
                        region_good
                    )
                ),

            "eval_bad_ge25m":
                int(
                    len(
                        region_bad
                    )
                ),
        },

        "contract": {
            "phase_a_used_gt":
                False,

            "gt_loaded_after_blind_freeze":
                True,

            "threshold_selected":
                False,

            "online_policy_created":
                False,

            "r3_modified":
                False,
        },
    }


    report_path.write_text(
        json.dumps(
            report,
            indent=2,
        )
    )


    # ========================================================
    # Print
    # ========================================================

    print()
    print("=" * 116)
    print(
        "R4.15 PHASE B — "
        "POST-FREEZE HOMOGRAPHY RELIABILITY AUDIT"
    )
    print("=" * 116)


    print(
        "correct-region candidate pairs:",
        len(
            correct_region
        ),
    )


    print()
    print("=" * 116)
    print(
        "BLIND FEATURE CORRELATION "
        "WITH SUB-TILE PROJECTED ERROR"
    )
    print("=" * 116)


    print(
        feature_audit.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.3f}",
        )
    )


    print()
    print("=" * 116)
    print(
        "EVAL-ONLY GOOD <=10 m "
        "VS BAD >=25 m"
    )
    print("=" * 116)


    print(
        comparison.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        )
    )


    print()
    print("=" * 116)
    print(
        "q36/q37/q38 CORRECT-REGION "
        "HOMOGRAPHIES"
    )
    print("=" * 116)


    show = [
        "query_id",
        "tile_id",
        "hybrid_rank",
        "dino_rank",
        "projected_error_m",
        "inliers",

        "q_center_inside_inlier_hull",
        "q_hull_area_fraction",
        "q_center_hull_signed_distance_norm",
        "q_center_nearest_inlier_distance_norm",
        "q_center_centroid_distance_norm",
        "q_center_angular_coverage_fraction",

        "s_projected_inside_inlier_hull",
        "s_hull_area_fraction",
        "s_projected_hull_signed_distance_norm",

        "jacobian_smax",
        "jacobian_smin",
        "jacobian_condition",
        "jacobian_local_area_scale",

        "reprojection_rmse_px",
    ]


    print(
        late_correct[
            show
        ].to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        )
    )


    print()
    print("=" * 116)
    print("R4.15 OUTPUT")
    print("=" * 116)


    print(
        "blind support features:",
        blind_path,
    )


    print(
        "blind freeze manifest:",
        freeze_path,
    )


    print(
        "GT annotated support:",
        annotated_path,
    )


    print(
        "feature audit:",
        feature_path,
    )


    print(
        "good/bad comparison:",
        comparison_path,
    )


    print(
        "report:",
        report_path,
    )


    print()


    print(
        "STATUS: "
        "PASS_R4_15_HOMOGRAPHY_CENTER_SUPPORT_AUDIT_EXECUTION"
    )


if __name__ == "__main__":
    main()
