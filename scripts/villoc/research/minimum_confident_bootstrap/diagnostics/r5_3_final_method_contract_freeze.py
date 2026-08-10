#!/usr/bin/env python3
"""
R5.3 — final R3-v2 research method / operating-envelope freeze.

This stage closes the minimum-confident-bootstrap research cycle.

Policy selection rule
---------------------
The R5.0 architecture froze four geometry-relative candidate policies.

The secondary Villoc 90_deg validation did not provide evidence that any
one candidate had superior localization accuracy because none matured.

All candidates safely produced NO_PROVISIONAL_LOCK.

Therefore the final candidate is selected using a safety-first tie-break:

    choose the strictest member of the already-frozen policy family.

Final policy:
    activation threshold = 0.25 map spacing = 12.8 m
    tracking threshold   = 0.25 map spacing = 12.8 m
    maturity support     = 3 consecutive consistent transitions

This is not GT optimization.

Operating envelope
------------------
Validated/intended:
    * near-nadir monocular RGB
    * prepared georeferenced orthophoto
    * 512_s256 map representation
    * approximately stationary visual-to-map metric scale
    * stable / slowly varying flight altitude
    * safe NO_PROVISIONAL_LOCK when confidence is insufficient

Known limitation:
    * strong altitude variation can produce strongly varying monocular
      visual scale, violating the single-global-similarity assumption.

That limitation is frozen as future research rather than patched into
the final demo method.

Command:

SCRIPT=scripts/villoc/research/minimum_confident_bootstrap/diagnostics/r5_3_final_method_contract_freeze.py

R3=outputs/research_runs/minimum_confident_bootstrap/traj01_blind_r3_001

VAL=outputs/research_runs/minimum_confident_bootstrap/validation_90deg_blind_r5_2_001

python "$SCRIPT" \
  --repo-root "$PWD" \
  --research-root "$R3" \
  --validation-run-root "$VAL" \
  2>&1 | tee \
  "$R3/postfreeze_eval/r5_3_final_method_contract_freeze.log"
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


EXPECTED_R5_0_SHA = (
    "c47acc6313cd8d32a59b81d9457e2904"
    "7192e7b9d6ef56524f39c4ce3208f93e"
)

EXPECTED_R5_2B_FREEZE_SHA = (
    "9fe09e2c3187fe4294fb8a79d1fd659d"
    "4f0c06693304c31806a3b8a56223a635"
)

MAP_SPACING_M = 51.2

FINAL_POLICY = (
    "activate_quarter_track_quarter"
)


def sha256(path: Path) -> str:

    h = hashlib.sha256()

    with path.open("rb") as f:

        for block in iter(
            lambda: f.read(
                1024 * 1024
            ),
            b"",
        ):

            h.update(block)

    return h.hexdigest()


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--repo-root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--research-root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--validation-run-root",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    repo = args.repo_root.resolve()
    research = args.research_root.resolve()
    validation = (
        args.validation_run_root
        .resolve()
    )


    # ========================================================
    # Frozen upstream evidence
    # ========================================================

    r5_0_contract = (
        research
        / "r5_0_r3v2_architecture_contract.json"
    )

    v2_script = (
        repo
        / "scripts/villoc/research/"
          "minimum_confident_bootstrap/"
          "minimum_confident_bootstrap_v2.py"
    )

    validation_freeze = (
        validation
        / "r5_v2/"
          "r5_1_blind_implementation_freeze_manifest.json"
    )

    validation_report = (
        validation
        / "r5_v2/postfreeze_eval/"
          "r5_2c_postfreeze_validation_report.json"
    )


    required = [
        r5_0_contract,
        v2_script,
        validation_freeze,
        validation_report,
    ]


    for path in required:

        if not path.exists():

            raise RuntimeError(
                f"Missing required evidence: {path}"
            )


    # ========================================================
    # Integrity checks
    # ========================================================

    r5_0_sha = sha256(
        r5_0_contract
    )


    if (
        r5_0_sha
        != EXPECTED_R5_0_SHA
    ):

        raise RuntimeError(
            "R5.0 architecture contract SHA mismatch.\n"
            f"expected: {EXPECTED_R5_0_SHA}\n"
            f"actual:   {r5_0_sha}"
        )


    r5_2b_sha = sha256(
        validation_freeze
    )


    if (
        r5_2b_sha
        != EXPECTED_R5_2B_FREEZE_SHA
    ):

        raise RuntimeError(
            "R5.2B blind validation freeze SHA mismatch.\n"
            f"expected: {EXPECTED_R5_2B_FREEZE_SHA}\n"
            f"actual:   {r5_2b_sha}"
        )


    architecture = json.loads(
        r5_0_contract.read_text()
    )


    validation_eval = json.loads(
        validation_report.read_text()
    )


    # ========================================================
    # Confirm candidate family has not changed.
    # ========================================================

    family = architecture[
        "candidate_policy_family"
    ]


    activation = family[
        "activation_thresholds"
    ]


    tracking = family[
        "tracking_thresholds"
    ]


    support = int(
        family[
            "maturity_support_required"
        ]
    )


    quarter_activation = float(
        activation[
            "quarter_spacing"
        ]
    )


    quarter_tracking = float(
        tracking[
            "quarter_spacing"
        ]
    )


    if abs(
        quarter_activation
        -
        0.25 * MAP_SPACING_M
    ) > 1e-9:

        raise RuntimeError(
            "Frozen quarter activation threshold changed."
        )


    if abs(
        quarter_tracking
        -
        0.25 * MAP_SPACING_M
    ) > 1e-9:

        raise RuntimeError(
            "Frozen quarter tracking threshold changed."
        )


    if support != 3:

        raise RuntimeError(
            "Frozen maturity support changed."
        )


    # ========================================================
    # Validation result classification
    # ========================================================

    top4 = validation_eval[
        "top4_observation_evaluation"
    ]


    leader = validation_eval[
        "leader_evaluation"
    ]


    global_similarity = validation_eval[
        "global_similarity_evaluation"
    ]


    local_similarity = validation_eval[
        "local_similarity"
    ]


    # ========================================================
    # Final research contract
    # ========================================================

    final_contract = {
        "stage":
            "R5.3_FINAL_MINIMUM_CONFIDENT_BOOTSTRAP_FREEZE",

        "status":
            "PASS_R5_3_FINAL_METHOD_CONTRACT_FREEZE",

        "method_name":
            "minimum_confident_bootstrap_v2",

        "research_status":
            "METHOD_FROZEN_FOR_IN_ENVELOPE_BLIND_DEMO",

        "architecture_contract": {
            "path":
                str(
                    r5_0_contract
                ),

            "sha256":
                r5_0_sha,
        },

        "implementation": {
            "path":
                str(
                    v2_script
                ),

            "sha256":
                sha256(
                    v2_script
                ),
        },

        "final_policy": {
            "name":
                FINAL_POLICY,

            "selection_basis":
                (
                    "safety-first tie-break among the already "
                    "frozen candidate family after secondary "
                    "validation; not GT accuracy optimization"
                ),

            "activation_threshold_fraction_of_map_spacing":
                0.25,

            "activation_threshold_m":
                quarter_activation,

            "tracking_threshold_fraction_of_map_spacing":
                0.25,

            "tracking_threshold_m":
                quarter_tracking,

            "maturity_support_required":
                support,

            "performance_optimality_claimed":
                False,

            "safety_priority":
                True,
        },

        "absolute_frontend": {
            "candidate_depth_dino":
                20,

            "bootstrap_top_m":
                4,

            "verifier":
                "ORB",

            "map_measurement":
                "continuous ORB sub-tile projection",

            "tile_variant":
                "512_s256",
        },

        "state_machine": {
            "initial":
                "ACQUISITION",

            "mature":
                "TRACKING",

            "tracking_update":
                "minimum-innovation accept-or-hold",

            "safe_fallback":
                "NO_PROVISIONAL_LOCK",
        },

        "operating_envelope": {
            "camera_view":
                "near-nadir",

            "map_requirement":
                (
                    "prepared georeferenced orthophoto with "
                    "matching 512_s256 descriptor cache"
                ),

            "relative_frontend":
                "XFeat",

            "visual_metric_scale":
                (
                    "approximately stationary over the "
                    "bootstrap/tracking interval"
                ),

            "altitude_behavior":
                (
                    "stable or slowly varying; no numerical "
                    "altitude tolerance is claimed from current data"
                ),

            "large_altitude_excursions_validated":
                False,

            "failure_behavior":
                (
                    "if evidence does not mature consistently, "
                    "return NO_PROVISIONAL_LOCK rather than force "
                    "absolute initialization"
                ),
        },

        "secondary_validation": {
            "dataset":
                "Villoc 90_deg",

            "blind_freeze_path":
                str(
                    validation_freeze
                ),

            "blind_freeze_sha256":
                r5_2b_sha,

            "blind_result":
                "NO_PROVISIONAL_LOCK",

            "top4_gt_inside_tile_queries":
                int(
                    top4[
                        "gt_inside_tile_queries"
                    ]
                ),

            "top4_queries":
                int(
                    top4[
                        "queries"
                    ]
                ),

            "top4_best_le40":
                int(
                    top4[
                        "best_le40"
                    ]
                ),

            "best_top4_projected_error_median_m":
                float(
                    top4[
                        "best_projected_error_median_m"
                    ]
                ),

            "blind_leader_median_gt_error_m":
                float(
                    leader[
                        "median_gt_error_m"
                    ]
                ),

            "best_admissible_median_gt_error_m":
                float(
                    leader[
                        "best_admissible_median_gt_error_m"
                    ]
                ),

            "global_similarity_residual_median_m":
                float(
                    global_similarity[
                        "residual_median_m"
                    ]
                ),

            "global_similarity_residual_p90_m":
                float(
                    global_similarity[
                        "residual_p90_m"
                    ]
                ),

            "local_scale_vs_altitude_spearman":
                float(
                    local_similarity[
                        "scale_altitude_spearman"
                    ]
                ),

            "interpretation":
                (
                    "safe rejection; weak absolute candidate support "
                    "plus strongly non-stationary monocular visual scale "
                    "under large altitude variation"
                ),
        },

        "known_limitations": [
            (
                "single global visual-to-map similarity is not "
                "validated for large altitude-induced scale changes"
            ),
            (
                "Top-4 geographic candidate availability can remain "
                "insufficient in repetitive/low-context scenes"
            ),
            (
                "NO_PROVISIONAL_LOCK is a valid safe output and "
                "must not be treated as an execution failure"
            ),
        ],

        "future_research_not_in_current_demo": {
            "scale_adaptive_similarity":
                True,

            "sliding_or_piecewise_similarity":
                True,

            "altitude_conditioned_scale":
                True,

            "current_method_must_not_be_patched_for_demo":
                True,
        },

        "output_contract": {
            "positive":
                "PROVISIONAL_ABSOLUTE_LOCK",

            "negative":
                "NO_PROVISIONAL_LOCK",

            "forbidden":
                "ABSOLUTE_LOCKED",
        },

        "next_stage":
            (
                "R5.4 production adapter + blind recorded-flight "
                "demo integration without changing mathematical policy"
            ),
    }


    out_path = (
        research
        / "r5_3_final_minimum_confident_bootstrap_contract.json"
    )


    out_path.write_text(
        json.dumps(
            final_contract,
            indent=2,
        )
    )


    digest = sha256(
        out_path
    )


    print()
    print("=" * 118)
    print(
        "R5.3 — FINAL MINIMUM-CONFIDENT-BOOTSTRAP CONTRACT FREEZE"
    )
    print("=" * 118)

    print(
        "R5.0 architecture SHA:",
        r5_0_sha,
    )

    print(
        "R5.2B validation freeze SHA:",
        r5_2b_sha,
    )

    print(
        "final policy:",
        FINAL_POLICY,
    )

    print(
        "activation gate:",
        quarter_activation,
        "m",
    )

    print(
        "tracking gate:",
        quarter_tracking,
        "m",
    )

    print(
        "maturity support:",
        support,
    )

    print(
        "policy selection:",
        "SAFETY_FIRST_NOT_GT_OPTIMIZED",
    )

    print(
        "operating view:",
        "near-nadir",
    )

    print(
        "large altitude variation:",
        "OUTSIDE CURRENT VALIDATED ENVELOPE",
    )

    print(
        "safe no-lock fallback:",
        True,
    )

    print(
        "final contract:",
        out_path,
    )

    print(
        "final contract SHA256:",
        digest,
    )

    print()

    print(
        "STATUS: "
        "PASS_R5_3_FINAL_METHOD_CONTRACT_FREEZE"
    )


if __name__ == "__main__":
    main()
