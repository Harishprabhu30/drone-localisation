#!/usr/bin/env python3
"""
R5.2C — Villoc 90-deg post-freeze R3-v2 validation.

The R5.2B blind run is already frozen.

R5.2C may therefore attach evaluation/reference information.

Questions
---------
1. Was the frozen NO_PROVISIONAL_LOCK safe?
2. Were Top-4 sub-tile observations geographically useful?
3. Were blind leaders geographically useful despite failure to mature?
4. Was the failure caused by candidate availability, leader selection,
   or a non-stationary visual->map similarity?
5. Does local similarity scale vary with altitude?

IMPORTANT
---------
This stage:
    * verifies the R5.2B blind freeze and every frozen output hash;
    * only then reads the old evaluation-only Villoc 90-deg reference index;
    * aligns by timestamp, not row order;
    * does not modify R3-v2;
    * does not tune thresholds;
    * does not select a production policy.

Command:

SCRIPT=scripts/villoc/research/minimum_confident_bootstrap/diagnostics/r5_2c_90deg_postfreeze_validation.py
RUN=outputs/research_runs/minimum_confident_bootstrap/validation_90deg_blind_r5_2_001

python "$SCRIPT" \
  --repo-root "$PWD" \
  --run-root "$RUN" \
  --expected-blind-freeze-sha256 \
    9fe09e2c3187fe4294fb8a79d1fd659d4f0c06693304c31806a3b8a56223a635 \
  2>&1 | tee \
  "$RUN/r5_v2/r5_2c_90deg_postfreeze_validation.log"
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer


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


def bool_value(value) -> bool:

    if isinstance(value, bool):
        return value

    return str(value).strip().lower() in {
        "true",
        "1",
        "1.0",
        "yes",
    }


def apply_similarity(
    xy: np.ndarray,
    model,
) -> np.ndarray:

    xy = np.asarray(
        xy,
        dtype=float,
    )

    z = (
        xy[:, 0]
        + 1j * xy[:, 1]
    )

    a = complex(
        float(model["a_real"]),
        float(model["a_imag"]),
    )

    b = complex(
        float(model["b_real"]),
        float(model["b_imag"]),
    )

    w = a * z + b

    return np.column_stack(
        [
            w.real,
            w.imag,
        ]
    )


def fit_similarity(
    visual_xy: np.ndarray,
    map_xy: np.ndarray,
):

    visual_xy = np.asarray(
        visual_xy,
        dtype=float,
    )

    map_xy = np.asarray(
        map_xy,
        dtype=float,
    )

    if (
        len(visual_xy) < 2
        or len(visual_xy) != len(map_xy)
    ):
        raise ValueError(
            "Need >=2 paired points."
        )

    x = visual_xy[:, 0]
    y = visual_xy[:, 1]

    e = map_xy[:, 0]
    n = map_xy[:, 1]

    # E = ar*x - ai*y + br
    # N = ai*x + ar*y + bi

    A = np.zeros(
        (
            2 * len(x),
            4,
        ),
        dtype=float,
    )

    bvec = np.zeros(
        2 * len(x),
        dtype=float,
    )

    A[0::2, 0] = x
    A[0::2, 1] = -y
    A[0::2, 2] = 1.0

    A[1::2, 0] = y
    A[1::2, 1] = x
    A[1::2, 3] = 1.0

    bvec[0::2] = e
    bvec[1::2] = n

    sol, _, _, _ = np.linalg.lstsq(
        A,
        bvec,
        rcond=None,
    )

    ar, ai, br, bi = sol

    scale = float(
        math.hypot(
            ar,
            ai,
        )
    )

    rotation = float(
        math.degrees(
            math.atan2(
                ai,
                ar,
            )
        )
    )

    model = {
        "a_real":
            float(ar),

        "a_imag":
            float(ai),

        "b_real":
            float(br),

        "b_imag":
            float(bi),

        "scale_m_per_visual_px":
            scale,

        "rotation_deg":
            rotation,
    }

    pred = apply_similarity(
        visual_xy,
        model,
    )

    residual = np.linalg.norm(
        pred - map_xy,
        axis=1,
    )

    return model, residual


def longest_true_run(values) -> int:

    best = 0
    current = 0

    for value in values:

        if bool(value):
            current += 1
            best = max(
                best,
                current,
            )

        else:
            current = 0

    return best


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
        "--expected-blind-freeze-sha256",
        required=True,
    )

    args = parser.parse_args()

    repo = args.repo_root.resolve()
    run = args.run_root.resolve()

    v2 = (
        run
        / "r5_v2"
    )

    freeze_path = (
        v2
        / "r5_1_blind_implementation_freeze_manifest.json"
    )

    observations_path = (
        v2
        / "r5_1_blind_top4_subtile_observations.csv"
    )

    hypotheses_path = (
        v2
        / "r5_1_blind_subtile_hypotheses.csv"
    )

    updates_path = (
        v2
        / "r5_1_blind_leader_updates.csv"
    )

    timeline_path = (
        v2
        / "r5_1_blind_policy_timeline.csv"
    )

    policy_results_path = (
        v2
        / "r5_1_blind_policy_results.json"
    )

    relative_path = (
        run
        / "metadata/"
          "s8_xfeat_relative_frontend/"
          "s8r4_xfeat_relative_trajectory_blind_raw.csv"
    )

    manifest_path = (
        run
        / "metadata/"
          "blind_query_manifest.csv"
    )

    reference_index_path = (
        repo
        / "outputs/villoc/90_deg/metadata/"
          "s8_5_uav_frames_index_v_1fps.csv"
    )

    tile_index_path = (
        repo
        / "outputs/villoc/90_deg/metadata/"
          "s8_9_satellite_tile_index_512_s256.csv"
    )


    required = [
        freeze_path,
        observations_path,
        hypotheses_path,
        updates_path,
        timeline_path,
        policy_results_path,
        relative_path,
        manifest_path,
        reference_index_path,
        tile_index_path,
    ]

    missing = [
        str(p)
        for p in required
        if not p.exists()
    ]

    if missing:
        raise RuntimeError(
            "Missing input(s):\n"
            + "\n".join(
                missing
            )
        )


    # ========================================================
    #
    # PHASE 0 — VERIFY FROZEN BLIND RUN BEFORE GT READ
    #
    # ========================================================

    actual_freeze_sha = sha256(
        freeze_path
    )

    if (
        actual_freeze_sha
        != args.expected_blind_freeze_sha256
    ):
        raise RuntimeError(
            "R5.2B blind freeze SHA mismatch.\n"
            f"expected: {args.expected_blind_freeze_sha256}\n"
            f"actual:   {actual_freeze_sha}"
        )


    freeze = json.loads(
        freeze_path.read_text()
    )


    blind_contract = freeze.get(
        "blind_contract",
        {}
    )


    for key in [
        "gps_used",
        "srt_used",
        "reference_used",
        "oracle_used",
        "evaluation_error_used",
    ]:

        if bool(
            blind_contract.get(
                key,
                False,
            )
        ):
            raise RuntimeError(
                f"Frozen blind contract violation: {key}=True"
            )


    expected_outputs = freeze.get(
        "outputs",
        {}
    )


    output_map = {
        "observations":
            observations_path,

        "hypotheses":
            hypotheses_path,

        "leader_updates":
            updates_path,

        "policy_timeline":
            timeline_path,

        "policy_results":
            policy_results_path,
    }


    output_hash_rows = []


    for name, path in output_map.items():

        expected = (
            expected_outputs
            .get(
                name,
                {}
            )
            .get(
                "sha256"
            )
        )

        actual = sha256(
            path
        )

        ok = bool(
            expected == actual
        )

        output_hash_rows.append(
            {
                "artifact":
                    name,

                "expected_sha256":
                    expected,

                "actual_sha256":
                    actual,

                "pass":
                    ok,
            }
        )

        if not ok:
            raise RuntimeError(
                f"Frozen output hash mismatch: {name}"
            )


    print()
    print("=" * 120)
    print(
        "R5.2C PHASE 0 — "
        "FROZEN BLIND RUN VERIFIED"
    )
    print("=" * 120)

    print(
        "blind freeze SHA256:",
        actual_freeze_sha,
    )

    print(
        "frozen outputs verified:",
        len(
            output_hash_rows
        ),
    )

    print(
        "GT/reference read so far:",
        False,
    )


    # ========================================================
    #
    # PHASE A — FIRST REFERENCE READ
    #
    # ========================================================

    manifest = pd.read_csv(
        manifest_path
    )

    relative = pd.read_csv(
        relative_path
    )

    observations = pd.read_csv(
        observations_path
    )

    hypotheses = pd.read_csv(
        hypotheses_path
    )

    updates = pd.read_csv(
        updates_path
    )

    timeline = pd.read_csv(
        timeline_path
    )


    # --------------------------------------------------------
    # Reference is intentionally loaded only after verification.
    # --------------------------------------------------------

    reference = pd.read_csv(
        reference_index_path
    )


    # ========================================================
    # Resolve reference schema
    # ========================================================

    def choose_column(
        df,
        candidates,
        label,
    ):

        for column in candidates:

            if column in df.columns:
                return column

        raise RuntimeError(
            f"Could not resolve {label}; "
            f"tried {candidates}"
        )


    ref_time_col = choose_column(
        reference,
        [
            "timestamp_s",
            "video_time_s",
            "sample_time_s",
        ],
        "reference timestamp",
    )

    lat_col = choose_column(
        reference,
        [
            "latitude",
            "lat",
        ],
        "reference latitude",
    )

    lon_col = choose_column(
        reference,
        [
            "longitude",
            "lon",
        ],
        "reference longitude",
    )

    alt_col = choose_column(
        reference,
        [
            "rel_alt_m",
            "relative_altitude_m",
            "relative_alt_m",
        ],
        "relative altitude",
    )


    reference_small = reference[
        [
            ref_time_col,
            lat_col,
            lon_col,
            alt_col,
        ]
    ].copy()


    reference_small = reference_small.rename(
        columns={
            ref_time_col:
                "reference_timestamp_s",

            lat_col:
                "gt_latitude",

            lon_col:
                "gt_longitude",

            alt_col:
                "gt_rel_alt_m",
        }
    )


    for column in [
        "reference_timestamp_s",
        "gt_latitude",
        "gt_longitude",
        "gt_rel_alt_m",
    ]:

        reference_small[
            column
        ] = pd.to_numeric(
            reference_small[
                column
            ],
            errors="raise",
        )


    reference_small = (
        reference_small
        .sort_values(
            "reference_timestamp_s"
        )
        .reset_index(
            drop=True
        )
    )


    manifest[
        "query_id"
    ] = pd.to_numeric(
        manifest[
            "query_id"
        ],
        errors="raise",
    ).astype(int)


    manifest[
        "timestamp_s"
    ] = pd.to_numeric(
        manifest[
            "timestamp_s"
        ],
        errors="raise",
    )


    query_time = (
        manifest[
            [
                "query_id",
                "timestamp_s",
            ]
        ]
        .sort_values(
            "timestamp_s"
        )
        .reset_index(
            drop=True
        )
    )


    aligned = pd.merge_asof(
        query_time,
        reference_small,
        left_on="timestamp_s",
        right_on="reference_timestamp_s",
        direction="nearest",
        tolerance=0.050,
    )


    aligned[
        "reference_alignment_error_ms"
    ] = (
        (
            aligned[
                "timestamp_s"
            ]
            -
            aligned[
                "reference_timestamp_s"
            ]
        )
        .abs()
        * 1000.0
    )


    missing_reference = int(
        aligned[
            "gt_latitude"
        ].isna().sum()
    )


    if missing_reference:
        raise RuntimeError(
            "Reference alignment failed for "
            f"{missing_reference} blind queries."
        )


    # ========================================================
    # WGS84 -> EPSG:3346 evaluation coordinates
    # ========================================================

    transformer = Transformer.from_crs(
        "EPSG:4326",
        "EPSG:3346",
        always_xy=True,
    )


    gt_e, gt_n = transformer.transform(
        aligned[
            "gt_longitude"
        ].to_numpy(float),
        aligned[
            "gt_latitude"
        ].to_numpy(float),
    )


    aligned[
        "gt_easting"
    ] = gt_e

    aligned[
        "gt_northing"
    ] = gt_n


    gt_by_q = (
        aligned
        .set_index(
            "query_id"
        )
    )


    print()
    print("=" * 120)
    print(
        "R5.2C PHASE A — "
        "POST-FREEZE REFERENCE ATTACHMENT"
    )
    print("=" * 120)

    print(
        "blind queries:",
        len(
            query_time
        ),
    )

    print(
        "reference rows available:",
        len(
            reference
        ),
    )

    print(
        "reference aligned queries:",
        len(
            aligned
        )
        - missing_reference,
    )

    print(
        "reference alignment median ms:",
        float(
            aligned[
                "reference_alignment_error_ms"
            ].median()
        ),
    )

    print(
        "reference alignment max ms:",
        float(
            aligned[
                "reference_alignment_error_ms"
            ].max()
        ),
    )

    print(
        "relative altitude range m:",
        float(
            aligned[
                "gt_rel_alt_m"
            ].min()
        ),
        "->",
        float(
            aligned[
                "gt_rel_alt_m"
            ].max()
        ),
    )


    # ========================================================
    #
    # PHASE B — TOP-4 CONTINUOUS OBSERVATION AVAILABILITY
    #
    # ========================================================

    observations[
        "query_id"
    ] = pd.to_numeric(
        observations[
            "query_id"
        ],
        errors="raise",
    ).astype(int)


    obs = observations.merge(
        aligned[
            [
                "query_id",
                "gt_easting",
                "gt_northing",
                "gt_rel_alt_m",
            ]
        ],
        on="query_id",
        how="left",
        validate="many_to_one",
    )


    obs[
        "projected_error_m"
    ] = np.hypot(
        obs[
            "projected_easting"
        ]
        -
        obs[
            "gt_easting"
        ],

        obs[
            "projected_northing"
        ]
        -
        obs[
            "gt_northing"
        ],
    )


    tile_index = pd.read_csv(
        tile_index_path
    )


    tile_geometry = tile_index[
        [
            "tile_id",
            "left_easting",
            "right_easting",
            "bottom_northing",
            "top_northing",
        ]
    ].copy()


    obs = obs.merge(
        tile_geometry,
        on="tile_id",
        how="left",
        validate="many_to_one",
    )


    obs[
        "gt_inside_tile"
    ] = (
        (
            obs[
                "gt_easting"
            ]
            >= obs[
                "left_easting"
            ]
        )
        &
        (
            obs[
                "gt_easting"
            ]
            <= obs[
                "right_easting"
            ]
        )
        &
        (
            obs[
                "gt_northing"
            ]
            >= obs[
                "bottom_northing"
            ]
        )
        &
        (
            obs[
                "gt_northing"
            ]
            <= obs[
                "top_northing"
            ]
        )
    )


    candidate_rows = []


    for q, group in obs.groupby(
        "query_id",
        sort=True,
    ):

        group = group.sort_values(
            "projected_error_m"
        )

        best = group.iloc[
            0
        ]


        candidate_rows.append(
            {
                "query_id":
                    int(q),

                "gt_rel_alt_m":
                    float(
                        best[
                            "gt_rel_alt_m"
                        ]
                    ),

                "best_top4_projected_error_m":
                    float(
                        best[
                            "projected_error_m"
                        ]
                    ),

                "best_top4_tile_id":
                    str(
                        best[
                            "tile_id"
                        ]
                    ),

                "best_top4_choice_rank":
                    int(
                        best[
                            "candidate_choice_rank"
                        ]
                    ),

                "best_top4_dino_rank":
                    int(
                        best[
                            "rank"
                        ]
                    ),

                "best_top4_hybrid_rank":
                    int(
                        best[
                            "hybrid_rank"
                        ]
                    ),

                "any_gt_inside_tile":
                    bool(
                        group[
                            "gt_inside_tile"
                        ].any()
                    ),

                "best_error_le10":
                    bool(
                        best[
                            "projected_error_m"
                        ]
                        <= 10.0
                    ),

                "best_error_le20":
                    bool(
                        best[
                            "projected_error_m"
                        ]
                        <= 20.0
                    ),

                "best_error_le40":
                    bool(
                        best[
                            "projected_error_m"
                        ]
                        <= 40.0
                    ),

                "best_error_le80":
                    bool(
                        best[
                            "projected_error_m"
                        ]
                        <= 80.0
                    ),
            }
        )


    candidate_eval = pd.DataFrame(
        candidate_rows
    )


    print()
    print("=" * 120)
    print(
        "TOP-4 SUB-TILE OBSERVATION AVAILABILITY"
    )
    print("=" * 120)

    print(
        "queries:",
        len(
            candidate_eval
        ),
    )

    print(
        "GT-containing tile available:",
        int(
            candidate_eval[
                "any_gt_inside_tile"
            ].sum()
        ),
        "/",
        len(
            candidate_eval
        ),
    )

    for threshold in [
        10,
        20,
        40,
        80,
    ]:

        column = (
            f"best_error_le{threshold}"
        )

        print(
            f"Top-4 projected <= {threshold:2d} m:",
            int(
                candidate_eval[
                    column
                ].sum()
            ),
            "/",
            len(
                candidate_eval
            ),
        )


    print(
        "best Top-4 projected error median:",
        float(
            candidate_eval[
                "best_top4_projected_error_m"
            ].median()
        ),
    )

    print(
        "best Top-4 projected error p90:",
        float(
            candidate_eval[
                "best_top4_projected_error_m"
            ].quantile(
                0.90
            )
        ),
    )


    # ========================================================
    #
    # PHASE C — BLIND LEADER QUALITY
    #
    # ========================================================

    relative[
        "query_id"
    ] = pd.to_numeric(
        relative[
            "token0_id"
            if "query_id" not in relative.columns
            else "query_id"
        ],
        errors="raise",
    ).astype(int)


    if "query_id" not in relative.columns:
        relative[
            "query_id"
        ] = pd.to_numeric(
            relative[
                "token0_id"
            ],
            errors="raise",
        ).astype(int)


    rel_by_q = (
        relative
        .set_index(
            "query_id"
        )
    )


    hypotheses[
        "hypothesis_id"
    ] = pd.to_numeric(
        hypotheses[
            "hypothesis_id"
        ],
        errors="raise",
    ).astype(int)


    hypotheses[
        "update_query_id"
    ] = pd.to_numeric(
        hypotheses[
            "update_query_id"
        ],
        errors="raise",
    ).astype(int)


    hyp_by_id = (
        hypotheses
        .set_index(
            "hypothesis_id"
        )
    )


    updates[
        "update_query_id"
    ] = pd.to_numeric(
        updates[
            "update_query_id"
        ],
        errors="raise",
    ).astype(int)


    leader_rows = []


    for _, update in updates.iterrows():

        q = int(
            update[
                "update_query_id"
            ]
        )

        leader_value = update[
            "blind_leader_hypothesis_id"
        ]


        if pd.isna(
            leader_value
        ):
            continue


        leader_id = int(
            float(
                leader_value
            )
        )


        leader = hyp_by_id.loc[
            leader_id
        ]


        visual = (
            rel_by_q.loc[
                [
                    q
                ],
                [
                    "visual_x_px",
                    "visual_y_px",
                ],
            ]
            .to_numpy(float)
        )


        gt_row = gt_by_q.loc[
            q
        ]


        gt_xy = np.asarray(
            [
                float(
                    gt_row[
                        "gt_easting"
                    ]
                ),
                float(
                    gt_row[
                        "gt_northing"
                    ]
                ),
            ]
        )


        leader_prediction = (
            apply_similarity(
                visual,
                leader,
            )[0]
        )


        leader_error = float(
            np.linalg.norm(
                leader_prediction
                - gt_xy
            )
        )


        q_hyp = hypotheses[
            hypotheses[
                "update_query_id"
            ]
            == q
        ].copy()


        gt_errors = []


        for _, h in q_hyp.iterrows():

            pred = apply_similarity(
                visual,
                h,
            )[0]

            gt_errors.append(
                float(
                    np.linalg.norm(
                        pred
                        - gt_xy
                    )
                )
            )


        q_hyp[
            "gt_current_prediction_error_m"
        ] = gt_errors


        q_hyp = q_hyp.sort_values(
            [
                "gt_current_prediction_error_m",
                "median_projected_residual_m",
            ]
        )


        oracle_best = q_hyp.iloc[
            0
        ]


        leader_rank = int(
            np.flatnonzero(
                q_hyp[
                    "hypothesis_id"
                ].to_numpy(int)
                == leader_id
            )[0]
            + 1
        )


        leader_rows.append(
            {
                "update_query_id":
                    q,

                "gt_rel_alt_m":
                    float(
                        gt_row[
                            "gt_rel_alt_m"
                        ]
                    ),

                "blind_leader_hypothesis_id":
                    leader_id,

                "blind_leader_scale":
                    float(
                        leader[
                            "scale_m_per_visual_px"
                        ]
                    ),

                "blind_leader_rotation_deg":
                    float(
                        leader[
                            "rotation_deg"
                        ]
                    ),

                "blind_leader_current_gt_error_m":
                    leader_error,

                "postfreeze_best_admissible_error_m":
                    float(
                        oracle_best[
                            "gt_current_prediction_error_m"
                        ]
                    ),

                "blind_leader_gt_rank":
                    leader_rank,

                "admissible_hypotheses":
                    int(
                        len(
                            q_hyp
                        )
                    ),
            }
        )


    leader_eval = pd.DataFrame(
        leader_rows
    )


    print()
    print("=" * 120)
    print(
        "BLIND LEADER POST-FREEZE QUALITY"
    )
    print("=" * 120)

    print(
        "leader updates:",
        len(
            leader_eval
        ),
    )

    print(
        "leader GT error median:",
        float(
            leader_eval[
                "blind_leader_current_gt_error_m"
            ].median()
        ),
    )

    print(
        "leader GT error p90:",
        float(
            leader_eval[
                "blind_leader_current_gt_error_m"
            ].quantile(
                0.90
            )
        ),
    )

    print(
        "postfreeze best-admissible error median:",
        float(
            leader_eval[
                "postfreeze_best_admissible_error_m"
            ].median()
        ),
    )

    for threshold in [
        10,
        20,
        40,
        80,
    ]:

        print(
            f"leader <= {threshold:2d} m:",
            int(
                (
                    leader_eval[
                        "blind_leader_current_gt_error_m"
                    ]
                    <= threshold
                ).sum()
            ),
            "/",
            len(
                leader_eval
            ),
        )


    # ========================================================
    #
    # PHASE D — WHY MATURITY NEVER OCCURRED
    #
    # ========================================================

    timeline[
        "update_query_id"
    ] = pd.to_numeric(
        timeline[
            "update_query_id"
        ],
        errors="raise",
    ).astype(int)


    timeline_eval = timeline.merge(
        aligned[
            [
                "query_id",
                "gt_rel_alt_m",
            ]
        ],
        left_on="update_query_id",
        right_on="query_id",
        how="left",
        validate="many_to_one",
    )


    timeline_eval[
        "minimum_innovation_m"
    ] = pd.to_numeric(
        timeline_eval[
            "minimum_innovation_m"
        ],
        errors="coerce",
    )


    maturity_rows = []


    for policy, group in (
        timeline_eval.groupby(
            "policy",
            sort=True,
        )
    ):

        group = group.sort_values(
            "update_query_id"
        ).copy()


        activation = float(
            group[
                "activation_threshold_m"
            ].iloc[0]
        )


        finite = group[
            np.isfinite(
                group[
                    "minimum_innovation_m"
                ]
            )
        ].copy()


        within = (
            finite[
                "minimum_innovation_m"
            ]
            <= activation
        )


        maturity_rows.append(
            {
                "policy":
                    policy,

                "activation_threshold_m":
                    activation,

                "finite_innovation_updates":
                    int(
                        len(
                            finite
                        )
                    ),

                "innovation_within_activation_count":
                    int(
                        within.sum()
                    ),

                "longest_consecutive_within_activation":
                    int(
                        longest_true_run(
                            within.tolist()
                        )
                    ),

                "innovation_median_m":
                    (
                        float(
                            finite[
                                "minimum_innovation_m"
                            ].median()
                        )
                        if len(
                            finite
                        )
                        else math.nan
                    ),

                "innovation_p90_m":
                    (
                        float(
                            finite[
                                "minimum_innovation_m"
                            ].quantile(
                                0.90
                            )
                        )
                        if len(
                            finite
                        )
                        else math.nan
                    ),

                "maximum_recorded_streak":
                    int(
                        pd.to_numeric(
                            group[
                                "consistency_streak_after"
                            ],
                            errors="coerce",
                        )
                        .fillna(0)
                        .max()
                    ),
            }
        )


    maturity_eval = pd.DataFrame(
        maturity_rows
    )


    print()
    print("=" * 120)
    print(
        "MATURITY FAILURE AUDIT"
    )
    print("=" * 120)

    print(
        maturity_eval.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.3f}",
        )
    )


    # ========================================================
    #
    # PHASE E — CONSTANT-SIMILARITY MODEL AUDIT
    #
    # GT is evaluation-only here.
    #
    # ========================================================

    relative_gt = relative.merge(
        aligned[
            [
                "query_id",
                "gt_easting",
                "gt_northing",
                "gt_rel_alt_m",
            ]
        ],
        on="query_id",
        how="inner",
        validate="one_to_one",
    )


    visual_xy_all = relative_gt[
        [
            "visual_x_px",
            "visual_y_px",
        ]
    ].to_numpy(float)


    gt_xy_all = relative_gt[
        [
            "gt_easting",
            "gt_northing",
        ]
    ].to_numpy(float)


    global_model, global_residual = (
        fit_similarity(
            visual_xy_all,
            gt_xy_all,
        )
    )


    window_rows = []

    window_size = 15
    step = 5


    for start in range(
        0,
        len(
            relative_gt
        )
        - window_size
        + 1,
        step,
    ):

        stop = (
            start
            + window_size
        )


        w = relative_gt.iloc[
            start:stop
        ]


        visual_xy = w[
            [
                "visual_x_px",
                "visual_y_px",
            ]
        ].to_numpy(float)


        gt_xy = w[
            [
                "gt_easting",
                "gt_northing",
            ]
        ].to_numpy(float)


        model, residual = fit_similarity(
            visual_xy,
            gt_xy,
        )


        window_rows.append(
            {
                "start_query_id":
                    int(
                        w[
                            "query_id"
                        ].iloc[0]
                    ),

                "end_query_id":
                    int(
                        w[
                            "query_id"
                        ].iloc[-1]
                    ),

                "altitude_median_m":
                    float(
                        w[
                            "gt_rel_alt_m"
                        ].median()
                    ),

                "altitude_min_m":
                    float(
                        w[
                            "gt_rel_alt_m"
                        ].min()
                    ),

                "altitude_max_m":
                    float(
                        w[
                            "gt_rel_alt_m"
                        ].max()
                    ),

                "similarity_scale":
                    float(
                        model[
                            "scale_m_per_visual_px"
                        ]
                    ),

                "similarity_rotation_deg":
                    float(
                        model[
                            "rotation_deg"
                        ]
                    ),

                "residual_median_m":
                    float(
                        np.median(
                            residual
                        )
                    ),

                "residual_p90_m":
                    float(
                        np.quantile(
                            residual,
                            0.90,
                        )
                    ),

                "residual_max_m":
                    float(
                        np.max(
                            residual
                        )
                    ),
            }
        )


    window_eval = pd.DataFrame(
        window_rows
    )


    scale_altitude_spearman = (
        float(
            window_eval[
                "similarity_scale"
            ].corr(
                window_eval[
                    "altitude_median_m"
                ],
                method="spearman",
            )
        )
        if len(
            window_eval
        )
        >= 3
        else math.nan
    )


    print()
    print("=" * 120)
    print(
        "XFEAT -> GT CONSTANT-SIMILARITY AUDIT"
    )
    print("=" * 120)

    print(
        "global similarity scale:",
        global_model[
            "scale_m_per_visual_px"
        ],
    )

    print(
        "global similarity rotation deg:",
        global_model[
            "rotation_deg"
        ],
    )

    print(
        "global residual median m:",
        float(
            np.median(
                global_residual
            )
        ),
    )

    print(
        "global residual p90 m:",
        float(
            np.quantile(
                global_residual,
                0.90,
            )
        ),
    )

    print(
        "global residual max m:",
        float(
            np.max(
                global_residual
            )
        ),
    )

    print()
    print(
        "15-frame local similarity windows:"
    )

    print(
        window_eval.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.3f}",
        )
    )

    print()
    print(
        "local similarity scale vs "
        "median altitude Spearman:",
        scale_altitude_spearman,
    )


    # ========================================================
    # Save all post-freeze evidence
    # ========================================================

    out = (
        v2
        / "postfreeze_eval"
    )

    out.mkdir(
        parents=True,
        exist_ok=True,
    )


    alignment_path = (
        out
        / "r5_2c_reference_alignment.csv"
    )

    obs_path = (
        out
        / "r5_2c_top4_projection_evaluation.csv"
    )

    candidate_path = (
        out
        / "r5_2c_top4_availability_summary.csv"
    )

    leader_path = (
        out
        / "r5_2c_blind_leader_evaluation.csv"
    )

    maturity_path = (
        out
        / "r5_2c_maturity_failure_summary.csv"
    )

    timeline_eval_path = (
        out
        / "r5_2c_policy_timeline_gt_annotated.csv"
    )

    window_path = (
        out
        / "r5_2c_local_similarity_windows.csv"
    )

    report_path = (
        out
        / "r5_2c_postfreeze_validation_report.json"
    )


    aligned.to_csv(
        alignment_path,
        index=False,
    )

    obs.to_csv(
        obs_path,
        index=False,
    )

    candidate_eval.to_csv(
        candidate_path,
        index=False,
    )

    leader_eval.to_csv(
        leader_path,
        index=False,
    )

    maturity_eval.to_csv(
        maturity_path,
        index=False,
    )

    timeline_eval.to_csv(
        timeline_eval_path,
        index=False,
    )

    window_eval.to_csv(
        window_path,
        index=False,
    )


    report = {
        "stage":
            "R5.2C_90DEG_POSTFREEZE_VALIDATION",

        "status":
            "PASS_R5_2C_POSTFREEZE_VALIDATION_EXECUTION",

        "blind_freeze_sha256":
            actual_freeze_sha,

        "blind_freeze_verified_before_reference":
            True,

        "reference": {
            "path":
                str(
                    reference_index_path
                ),

            "role":
                "evaluation_only",

            "blind_queries":
                int(
                    len(
                        aligned
                    )
                ),

            "reference_alignment_median_ms":
                float(
                    aligned[
                        "reference_alignment_error_ms"
                    ].median()
                ),

            "reference_alignment_max_ms":
                float(
                    aligned[
                        "reference_alignment_error_ms"
                    ].max()
                ),

            "rel_alt_min_m":
                float(
                    aligned[
                        "gt_rel_alt_m"
                    ].min()
                ),

            "rel_alt_max_m":
                float(
                    aligned[
                        "gt_rel_alt_m"
                    ].max()
                ),
        },

        "top4_observation_evaluation": {
            "queries":
                int(
                    len(
                        candidate_eval
                    )
                ),

            "best_projected_error_median_m":
                float(
                    candidate_eval[
                        "best_top4_projected_error_m"
                    ].median()
                ),

            "best_projected_error_p90_m":
                float(
                    candidate_eval[
                        "best_top4_projected_error_m"
                    ].quantile(
                        0.90
                    )
                ),

            "gt_inside_tile_queries":
                int(
                    candidate_eval[
                        "any_gt_inside_tile"
                    ].sum()
                ),

            "best_le10":
                int(
                    candidate_eval[
                        "best_error_le10"
                    ].sum()
                ),

            "best_le20":
                int(
                    candidate_eval[
                        "best_error_le20"
                    ].sum()
                ),

            "best_le40":
                int(
                    candidate_eval[
                        "best_error_le40"
                    ].sum()
                ),
        },

        "leader_evaluation": {
            "leader_updates":
                int(
                    len(
                        leader_eval
                    )
                ),

            "median_gt_error_m":
                float(
                    leader_eval[
                        "blind_leader_current_gt_error_m"
                    ].median()
                ),

            "p90_gt_error_m":
                float(
                    leader_eval[
                        "blind_leader_current_gt_error_m"
                    ].quantile(
                        0.90
                    )
                ),

            "best_admissible_median_gt_error_m":
                float(
                    leader_eval[
                        "postfreeze_best_admissible_error_m"
                    ].median()
                ),
        },

        "global_similarity_evaluation": {
            **{
                k:
                    float(v)
                for k, v
                in global_model.items()
            },

            "residual_median_m":
                float(
                    np.median(
                        global_residual
                    )
                ),

            "residual_p90_m":
                float(
                    np.quantile(
                        global_residual,
                        0.90,
                    )
                ),
        },

        "local_similarity": {
            "window_size_queries":
                window_size,

            "step_queries":
                step,

            "windows":
                int(
                    len(
                        window_eval
                    )
                ),

            "scale_altitude_spearman":
                scale_altitude_spearman,
        },

        "contract": {
            "blind_outputs_modified":
                False,

            "production_policy_selected":
                False,

            "threshold_tuned":
                False,

            "r3v2_modified":
                False,
        },
    }


    report_path.write_text(
        json.dumps(
            report,
            indent=2,
        )
    )


    print()
    print("=" * 120)
    print(
        "R5.2C OUTPUT"
    )
    print("=" * 120)

    print(
        "reference alignment:",
        alignment_path,
    )

    print(
        "Top-4 projection evaluation:",
        obs_path,
    )

    print(
        "Top-4 summary:",
        candidate_path,
    )

    print(
        "leader evaluation:",
        leader_path,
    )

    print(
        "maturity summary:",
        maturity_path,
    )

    print(
        "annotated policy timeline:",
        timeline_eval_path,
    )

    print(
        "local similarity windows:",
        window_path,
    )

    print(
        "report:",
        report_path,
    )

    print()
    print(
        "production policy selected:",
        False,
    )

    print(
        "R3-v2 modified:",
        False,
    )

    print()

    print(
        "STATUS: "
        "PASS_R5_2C_POSTFREEZE_VALIDATION_EXECUTION"
    )


if __name__ == "__main__":
    main()
