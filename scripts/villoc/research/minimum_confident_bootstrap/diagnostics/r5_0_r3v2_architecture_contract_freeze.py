#!/usr/bin/env python3
"""
R5.0 — R3-v2 architecture contract freeze.

Purpose
-------
Freeze the architecture justified by the completed R4 diagnostic series
BEFORE implementing the clean R3-v2 blind bootstrap.

This stage does NOT:
    * read GT/reference/SRT/GPS;
    * select a final tracking threshold;
    * rerun localization;
    * modify R3;
    * modify the demo pipeline.

It verifies the relevant blind freeze manifests and writes a single
machine-readable contract for R5.1.

R3-v2 architectural contract
----------------------------

ABSOLUTE OBSERVATION
    DINO Top-4 geographic candidate shortlist
        ->
    exact historical ORB verifier
        ->
    retain query->tile homography
        ->
    project processed query image centre
        ->
    continuous EPSG:3346 sub-tile observation

RELATIVE GEOMETRY
    XFeat blind relative trajectory

ACQUISITION
    R3-style 4-frame evidence geometry
    explicit Top-4 candidate combinations
    similarity transform
    current blind leader may replace provisional state

MATURITY
    three consecutive state-consistent transitions

TRACKING
    predict current map position using last accepted transform + XFeat
    rank current Top-4 observations by innovation
    accept or HOLD depending on predeclared tracking policy

OUTPUT SAFETY
    research output may only become:
        PROVISIONAL_ABSOLUTE_LOCK
        NO_PROVISIONAL_LOCK

    Never ABSOLUTE_LOCKED.

POLICY STATUS
-------------
traj01 does NOT select the final production policy.

Candidate geometry-relative policy family remains frozen:

    activation:
        0.25 grid spacing = 12.8 m
        0.50 grid spacing = 25.6 m

    tracking:
        0.25 grid spacing = 12.8 m
        0.50 grid spacing = 25.6 m

    maturity support:
        3 consecutive transitions

All candidate policies must remain available for independent validation.

Command:

SCRIPT=scripts/villoc/research/minimum_confident_bootstrap/diagnostics/r5_0_r3v2_architecture_contract_freeze.py
R3=outputs/research_runs/minimum_confident_bootstrap/traj01_blind_r3_001

python "$SCRIPT" \
  --research-root "$R3" \
  2>&1 | tee \
  "$R3/postfreeze_eval/r5_0_r3v2_architecture_contract_freeze.log"
  
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


MAP_SPACING_M = 51.2
SUPPORT_REQUIRED = 3


def sha256(path: Path) -> str:

    h = hashlib.sha256()

    with path.open("rb") as f:

        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):

            h.update(block)

    return h.hexdigest()


def load_json(path: Path):

    if not path.exists():

        raise RuntimeError(
            f"Missing required freeze manifest: {path}"
        )

    return json.loads(
        path.read_text()
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--research-root",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    research = (
        args.research_root
        .resolve()
    )

    post = (
        research
        / "postfreeze_eval"
    )


    # ========================================================
    # Blind evidence provenance
    # ========================================================

    manifests = {
        "r4_11_subtile_projection":
            post
            / "r4_11_blind_subtile_projection_freeze_manifest.json",

        "r4_12_subtile_bootstrap_geometry":
            post
            / "r4_12_blind_subtile_geometry_freeze_manifest.json",

        "r4_15_homography_support":
            post
            / "r4_15_blind_homography_center_support_freeze_manifest.json",

        "r4_16_measurement_innovation":
            post
            / "r4_16_blind_measurement_innovation_freeze_manifest.json",

        "r4_18_acquisition_tracking":
            post
            / "r4_18_blind_acquisition_tracking_freeze_manifest.json",
    }


    loaded = {
        name:
            load_json(
                path
            )
        for name, path
        in manifests.items()
    }


    # ========================================================
    # Verify all upstream Phase-A contracts remained blind.
    # ========================================================

    violations = []


    for name, payload in (
        loaded.items()
    ):

        contract = (
            payload.get(
                "blind_contract",
                {}
            )
        )


        for forbidden_key in [
            "gt_used",
            "reference_used",
            "oracle_used",
        ]:

            if bool(
                contract.get(
                    forbidden_key,
                    False,
                )
            ):

                violations.append(
                    f"{name}: "
                    f"{forbidden_key}=True"
                )


    if violations:

        raise RuntimeError(
            "Blind contract violation:\n"
            + "\n".join(
                violations
            )
        )


    r411 = loaded[
        "r4_11_subtile_projection"
    ]


    reproduction_pass = bool(
        r411[
            "reproduction_gate"
        ][
            "pass"
        ]
    )


    if not reproduction_pass:

        raise RuntimeError(
            "R4.11 ORB reproduction gate was not PASS."
        )


    # ========================================================
    # Freeze R3-v2 architecture.
    # ========================================================

    contract = {
        "stage":
            "R5.0_R3V2_ARCHITECTURE_CONTRACT_FREEZE",

        "status":
            "PASS_R5_0_R3V2_ARCHITECTURE_CONTRACT_FREEZE",

        "scope":
            (
                "research implementation candidate; "
                "not production/demo policy"
            ),

        "absolute_observation": {
            "candidate_source":
                "DINO retrieval",

            "candidate_count":
                4,

            "verifier":
                "ORB",

            "orb_preprocess":
                "clahe_luma",

            "orb_resize_long":
                1024,

            "orb_nfeatures":
                1800,

            "orb_lowe_ratio":
                0.80,

            "orb_ransac_threshold_px":
                5.0,

            "homography_direction":
                "query pixel -> satellite tile pixel",

            "map_measurement":
                (
                    "processed query image centre projected "
                    "through ORB homography into satellite "
                    "tile and converted continuously to EPSG:3346"
                ),

            "tile_center_as_primary_measurement":
                False,
        },

        "relative_geometry": {
            "source":
                "XFeat blind relative trajectory",

            "gt_alignment":
                False,
        },

        "bootstrap_hypothesis": {
            "evidence_queries":
                4,

            "candidate_combinations":
                "explicit Top-4 Cartesian enumeration",

            "transform":
                "least-squares 2D similarity",

            "minimum_unique_tile_ids":
                3,

            "minimum_visual_span_px":
                100.0,

            "minimum_map_span_m":
                50.0,

            "median_measurement_residual_limit_m":
                51.2,

            "maximum_measurement_residual_limit_m":
                102.4,
        },

        "state_machine": {
            "initial_mode":
                "ACQUISITION",

            "acquisition_behavior":
                (
                    "available current blind leader may replace "
                    "provisional state; innovation does not block "
                    "recovery before maturity"
                ),

            "maturity_support_required":
                SUPPORT_REQUIRED,

            "maturity_signal":
                (
                    "consecutive minimum current Top-4 "
                    "sub-tile innovations within activation gate"
                ),

            "mature_mode":
                "TRACKING",

            "tracking_behavior":
                (
                    "predict current map position from last "
                    "accepted similarity transform and XFeat; "
                    "rank current sub-tile measurements by "
                    "innovation; accept compatible update or HOLD"
                ),

            "state_hold_allowed":
                True,
        },

        "candidate_policy_family": {
            "map_spacing_m":
                MAP_SPACING_M,

            "activation_thresholds": {
                "quarter_spacing":
                    0.25
                    * MAP_SPACING_M,

                "half_spacing":
                    0.50
                    * MAP_SPACING_M,
            },

            "tracking_thresholds": {
                "quarter_spacing":
                    0.25
                    * MAP_SPACING_M,

                "half_spacing":
                    0.50
                    * MAP_SPACING_M,
            },

            "maturity_support_required":
                SUPPORT_REQUIRED,

            "final_policy_selected":
                False,

            "selection_rule":
                (
                    "must not be selected using traj01 GT; "
                    "requires independent blind validation"
                ),
        },

        "homography_reliability": {
            "jacobian_condition":
                (
                    "retain as diagnostic / candidate quality "
                    "signal; no hard acceptance threshold frozen"
                ),

            "reprojection_rmse":
                (
                    "diagnostic only; not sufficient as "
                    "projection trust criterion"
                ),

            "convex_hull_containment":
                (
                    "diagnostic only; not required"
                ),
        },

        "output_contract": {
            "allowed_positive_state":
                "PROVISIONAL_ABSOLUTE_LOCK",

            "allowed_negative_state":
                "NO_PROVISIONAL_LOCK",

            "forbidden_state":
                "ABSOLUTE_LOCKED",

            "gt_before_output_freeze":
                False,
        },

        "upstream_blind_freezes": {
            name: {
                "path":
                    str(
                        path
                    ),

                "sha256":
                    sha256(
                        path
                    ),
            }
            for name, path
            in manifests.items()
        },

        "implementation_next":
            (
                "R5.1 clean R3-v2 blind implementation "
                "using this frozen contract"
            ),
    }


    out_path = (
        research
        / "r5_0_r3v2_architecture_contract.json"
    )


    out_path.write_text(
        json.dumps(
            contract,
            indent=2,
        )
    )


    digest = sha256(
        out_path
    )


    print()
    print("=" * 116)
    print(
        "R5.0 — R3-v2 ARCHITECTURE CONTRACT FREEZE"
    )
    print("=" * 116)

    print(
        "upstream blind freezes verified:",
        len(
            manifests
        ),
    )

    print(
        "R4.11 ORB reproduction:",
        "PASS",
    )

    print(
        "Top-M:",
        4,
    )

    print(
        "absolute measurement:",
        "ORB continuous sub-tile projection",
    )

    print(
        "state modes:",
        "ACQUISITION -> TRACKING",
    )

    print(
        "maturity support:",
        SUPPORT_REQUIRED,
    )

    print(
        "activation candidate gates:",
        "12.8 m, 25.6 m",
    )

    print(
        "tracking candidate gates:",
        "12.8 m, 25.6 m",
    )

    print(
        "final policy selected:",
        False,
    )

    print(
        "GT/reference used:",
        False,
    )

    print(
        "contract:",
        out_path,
    )

    print(
        "contract SHA256:",
        digest,
    )

    print()

    print(
        "STATUS: "
        "PASS_R5_0_R3V2_ARCHITECTURE_CONTRACT_FREEZE"
    )


if __name__ == "__main__":
    main()
