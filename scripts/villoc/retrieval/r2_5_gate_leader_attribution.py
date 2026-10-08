#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path.cwd().resolve()

DEFAULT_R23_ROOT = Path(
    "outputs/research_runs/retrieval_candidate_pool_v2/"
    "r2_cross_scale_fusion/r2_3_state_replay/run"
)
DEFAULT_R14_768_ROOT = Path(
    "outputs/research_runs/retrieval_candidate_pool_v2/"
    "r1_4_downstream_replay/runs/768_s256"
)
DEFAULT_R24_REPORT = Path(
    "outputs/research_runs/retrieval_candidate_pool_v2/"
    "r2_cross_scale_fusion/r2_4_failure_attribution/"
    "r2_4_state_failure_attribution_report.json"
)
DEFAULT_OUTPUT_ROOT = Path(
    "outputs/research_runs/retrieval_candidate_pool_v2/"
    "r2_cross_scale_fusion/r2_5_gate_leader_diagnostic"
)

POLICY = "activate_quarter_track_quarter"


def resolve(path_like: str | Path) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else ROOT / path


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_int_list(value) -> list[int]:
    if pd.isna(value) or str(value).strip() == "":
        return []
    return [
        int(float(part.strip()))
        for part in str(value).split(",")
        if part.strip()
    ]


def parse_str_list(value) -> list[str]:
    if pd.isna(value) or str(value).strip() == "":
        return []
    return [
        part.strip()
        for part in str(value).split(",")
        if part.strip()
    ]


def variant_from_tile_id(tile_id: str) -> str:
    value = str(tile_id)
    if "::" not in value:
        return "single_scale"
    return value.split("::", 1)[0]


def row_json(row: pd.Series | None) -> dict | None:
    if row is None:
        return None

    result = {}
    for key, value in row.to_dict().items():
        if isinstance(value, np.generic):
            value = value.item()
        if pd.isna(value):
            value = None
        result[key] = value
    return result


def get_policy_row(
    timeline: pd.DataFrame,
    query_id: int,
) -> pd.Series:
    sub = timeline[
        (timeline["policy"].astype(str) == POLICY)
        & (
            pd.to_numeric(
                timeline["update_query_id"],
                errors="coerce",
            ) == int(query_id)
        )
    ]

    if len(sub) != 1:
        raise RuntimeError(
            f"Expected exactly one {POLICY} row at q{query_id}, got {len(sub)}."
        )

    return sub.iloc[0]


def get_leader_row(
    leaders: pd.DataFrame,
    query_id: int,
) -> pd.Series:
    sub = leaders[
        pd.to_numeric(
            leaders["update_query_id"],
            errors="coerce",
        ) == int(query_id)
    ]

    if len(sub) != 1:
        raise RuntimeError(
            f"Expected exactly one leader-update row at q{query_id}, got {len(sub)}."
        )

    return sub.iloc[0]


def get_hypothesis(
    hypotheses: pd.DataFrame,
    hypothesis_id: int,
) -> pd.Series:
    sub = hypotheses[
        pd.to_numeric(
            hypotheses["hypothesis_id"],
            errors="coerce",
        ) == int(hypothesis_id)
    ]

    if len(sub) != 1:
        raise RuntimeError(
            f"Expected exactly one hypothesis {hypothesis_id}, got {len(sub)}."
        )

    return sub.iloc[0]


def supporting_observations(
    hypothesis: pd.Series,
    observations: pd.DataFrame,
) -> list[dict]:
    qids = parse_int_list(
        hypothesis["evidence_query_ids"]
    )
    tile_ids = parse_str_list(
        hypothesis["tile_ids"]
    )
    choice_ranks = parse_int_list(
        hypothesis["candidate_choice_ranks"]
    )

    if not (
        len(qids)
        == len(tile_ids)
        == len(choice_ranks)
    ):
        raise RuntimeError(
            "Hypothesis evidence lists have inconsistent lengths."
        )

    rows = []

    for qid, tile_id, choice_rank in zip(
        qids,
        tile_ids,
        choice_ranks,
    ):
        sub = observations[
            (
                pd.to_numeric(
                    observations["query_id"],
                    errors="coerce",
                )
                == int(qid)
            )
            & (
                observations["tile_id"].astype(str)
                == str(tile_id)
            )
            & (
                pd.to_numeric(
                    observations["candidate_choice_rank"],
                    errors="coerce",
                )
                == int(choice_rank)
            )
        ]

        if len(sub) != 1:
            raise RuntimeError(
                "Could not uniquely recover supporting observation "
                f"q={qid} tile={tile_id} choice={choice_rank}; rows={len(sub)}"
            )

        row = row_json(sub.iloc[0])
        row["tile_variant"] = variant_from_tile_id(
            str(tile_id)
        )
        rows.append(row)

    return rows


def diagnose_run(
    *,
    label: str,
    backend_root: Path,
    query_id: int,
    forced_hypothesis_id: int | None = None,
) -> dict:
    paths = {
        "policy_timeline": (
            backend_root
            / "r5_1_blind_policy_timeline.csv"
        ),
        "leader_updates": (
            backend_root
            / "r5_1_blind_leader_updates.csv"
        ),
        "hypotheses": (
            backend_root
            / "r5_1_blind_subtile_hypotheses.csv"
        ),
        "observations": (
            backend_root
            / "r5_1_blind_top4_subtile_observations.csv"
        ),
    }

    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)

    timeline = pd.read_csv(paths["policy_timeline"])
    leaders = pd.read_csv(paths["leader_updates"])
    hypotheses = pd.read_csv(paths["hypotheses"])
    observations = pd.read_csv(paths["observations"])

    policy_row = get_policy_row(
        timeline,
        query_id,
    )
    leader_row = get_leader_row(
        leaders,
        query_id,
    )

    leader_hypothesis_id = (
        int(
            float(
                leader_row["blind_leader_hypothesis_id"]
            )
        )
        if pd.notna(
            leader_row["blind_leader_hypothesis_id"]
        )
        else None
    )

    active_hypothesis_id_after = (
        int(
            float(
                policy_row["active_hypothesis_id_after"]
            )
        )
        if pd.notna(
            policy_row["active_hypothesis_id_after"]
        )
        else None
    )

    hypothesis_id = (
        int(forced_hypothesis_id)
        if forced_hypothesis_id is not None
        else leader_hypothesis_id
    )

    if hypothesis_id is None:
        hypothesis = None
        support = []
    else:
        hypothesis = get_hypothesis(
            hypotheses,
            hypothesis_id,
        )
        support = supporting_observations(
            hypothesis,
            observations,
        )

    innovation_best_tile_id = str(
        policy_row.get(
            "innovation_best_tile_id",
            "",
        )
    )
    innovation_best_choice_rank = (
        int(
            float(
                policy_row[
                    "innovation_best_choice_rank"
                ]
            )
        )
        if pd.notna(
            policy_row[
                "innovation_best_choice_rank"
            ]
        )
        else None
    )

    current_q_support = [
        row
        for row in support
        if int(row["query_id"]) == int(query_id)
    ]

    leader_tile_ids = (
        parse_str_list(
            hypothesis["tile_ids"]
        )
        if hypothesis is not None
        else []
    )

    gate_tile_in_leader = (
        innovation_best_tile_id in leader_tile_ids
        if innovation_best_tile_id
        else False
    )

    gate_choice_matches_leader_at_current_q = False
    if (
        innovation_best_choice_rank is not None
        and current_q_support
    ):
        gate_choice_matches_leader_at_current_q = any(
            str(row["tile_id"])
            == innovation_best_tile_id
            and int(row["candidate_choice_rank"])
            == innovation_best_choice_rank
            for row in current_q_support
        )

    support_variants = [
        variant_from_tile_id(tile_id)
        for tile_id in leader_tile_ids
    ]

    return {
        "label": label,
        "query_id": int(query_id),
        "policy": POLICY,
        "policy_row": row_json(policy_row),
        "leader_update_row": row_json(leader_row),
        "leader_hypothesis_id": leader_hypothesis_id,
        "active_hypothesis_id_after": active_hypothesis_id_after,
        "diagnosed_hypothesis_id": hypothesis_id,
        "accepted_hypothesis_is_current_leader": (
            active_hypothesis_id_after
            == leader_hypothesis_id
            if leader_hypothesis_id is not None
            else False
        ),
        "hypothesis": (
            row_json(hypothesis)
            if hypothesis is not None
            else None
        ),
        "supporting_observations": support,
        "support_variants": support_variants,
        "support_variant_count": int(
            len(set(support_variants))
        ),
        "innovation_gate": {
            "minimum_innovation_m": (
                float(
                    policy_row[
                        "minimum_innovation_m"
                    ]
                )
                if pd.notna(
                    policy_row[
                        "minimum_innovation_m"
                    ]
                )
                else None
            ),
            "tracking_threshold_m": (
                float(
                    policy_row[
                        "tracking_threshold_m"
                    ]
                )
                if pd.notna(
                    policy_row[
                        "tracking_threshold_m"
                    ]
                )
                else None
            ),
            "innovation_best_tile_id": innovation_best_tile_id,
            "innovation_best_choice_rank": innovation_best_choice_rank,
            "gate_tile_in_accepted_leader_hypothesis": gate_tile_in_leader,
            "gate_choice_matches_accepted_leader_current_q_observation": (
                gate_choice_matches_leader_at_current_q
            ),
        },
        "input_hashes": {
            name: {
                "path": str(path),
                "sha256": sha256_file(path),
            }
            for name, path in paths.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "R2.5 post-freeze diagnostic: determine whether the current "
            "minimum-innovation gate observation actually supports the blind "
            "leader hypothesis installed by TRACKING_ACCEPT."
        )
    )
    parser.add_argument(
        "--r2-3-root",
        type=Path,
        default=DEFAULT_R23_ROOT,
    )
    parser.add_argument(
        "--r1-4-768-root",
        type=Path,
        default=DEFAULT_R14_768_ROOT,
    )
    parser.add_argument(
        "--r2-4-report",
        type=Path,
        default=DEFAULT_R24_REPORT,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    args = parser.parse_args()

    r23_root = resolve(args.r2_3_root)
    r14_root = resolve(args.r1_4_768_root)
    r24_report_path = resolve(args.r2_4_report)
    output_root = resolve(args.output_root)

    if not r24_report_path.exists():
        raise FileNotFoundError(r24_report_path)

    r24 = load_json(r24_report_path)

    if (
        r24.get("status")
        != "PASS_R2_STATE_FAILURE_ATTRIBUTION"
    ):
        raise RuntimeError(
            "R2.4 attribution gate has not passed."
        )

    poisoning = r24["poisoning_event_candidate"]
    query_id = int(
        poisoning["effective_query_id"]
    )
    poisoning_hypothesis_id = int(
        poisoning["source_hypothesis_id"]
    )

    r23_backend = (
        r23_root
        / "reports/blind_map_bootstrap/"
          "minimum_confident_v2"
    )
    r14_backend = (
        r14_root
        / "reports/blind_map_bootstrap/"
          "minimum_confident_v2"
    )

    r23 = diagnose_run(
        label="r2_triple_fused_orb",
        backend_root=r23_backend,
        query_id=query_id,
        forced_hypothesis_id=poisoning_hypothesis_id,
    )
    r14 = diagnose_run(
        label="r1_4_768_single_scale",
        backend_root=r14_backend,
        query_id=query_id,
    )

    comparison = {
        "same_query_id": int(query_id),
        "r2_action": r23["policy_row"]["action"],
        "r14_action": r14["policy_row"]["action"],
        "r2_minimum_innovation_m": (
            r23["innovation_gate"][
                "minimum_innovation_m"
            ]
        ),
        "r14_minimum_innovation_m": (
            r14["innovation_gate"][
                "minimum_innovation_m"
            ]
        ),
        "r2_gate_tile_in_accepted_leader": (
            r23["innovation_gate"][
                "gate_tile_in_accepted_leader_hypothesis"
            ]
        ),
        "r14_gate_tile_in_current_leader": (
            r14["innovation_gate"][
                "gate_tile_in_accepted_leader_hypothesis"
            ]
        ),
        "r2_gate_choice_matches_leader_current_q": (
            r23["innovation_gate"][
                "gate_choice_matches_accepted_leader_current_q_observation"
            ]
        ),
        "r14_gate_choice_matches_leader_current_q": (
            r14["innovation_gate"][
                "gate_choice_matches_accepted_leader_current_q_observation"
            ]
        ),
        "r2_support_variants": r23[
            "support_variants"
        ],
        "r14_support_variants": r14[
            "support_variants"
        ],
        "r2_leader_rotation_deg": (
            float(
                r23["hypothesis"]["rotation_deg"]
            )
            if r23["hypothesis"] is not None
            else None
        ),
        "r14_leader_rotation_deg": (
            float(
                r14["hypothesis"]["rotation_deg"]
            )
            if r14["hypothesis"] is not None
            else None
        ),
        "r2_leader_scale": (
            float(
                r23["hypothesis"][
                    "scale_m_per_visual_px"
                ]
            )
            if r23["hypothesis"] is not None
            else None
        ),
        "r14_leader_scale": (
            float(
                r14["hypothesis"][
                    "scale_m_per_visual_px"
                ]
            )
            if r14["hypothesis"] is not None
            else None
        ),
    }

    report = {
        "stage": "R2.5",
        "status": "PASS_R2_GATE_LEADER_ATTRIBUTION",
        "created_at_utc": now_utc(),
        "purpose": (
            "Post-freeze diagnostic only. Test whether the current "
            "minimum-innovation observation that opens the TRACKING gate is "
            "actually part of the blind leader hypothesis that becomes active."
        ),
        "poisoning_query_id": query_id,
        "r2_triple": r23,
        "stable_768_control": r14,
        "comparison": comparison,
        "code_contract_observation": {
            "tracking_gate": (
                "minimum innovation over any current valid Top-4 observation"
            ),
            "accepted_object": (
                "independently selected blind Pareto leader hypothesis"
            ),
            "same_object_required_by_frozen_code": False,
        },
        "scope_guarantees": {
            "localization_rerun": False,
            "orb_rerun": False,
            "state_policy_changed": False,
            "reference_used_for_localization": False,
            "postfreeze_analysis_only": True,
        },
        "next_decision": (
            "Close R2 candidate-fusion study and open a separate "
            "bootstrap/state-safety stage if the gate observation and accepted "
            "leader are shown to be decoupled or mixed-scale leader formation "
            "is otherwise incompatible with the frozen 512-oriented method."
        ),
    }

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )
    report_path = (
        output_root
        / "r2_5_gate_leader_attribution_report.json"
    )
    report_path.write_text(
        json.dumps(
            report,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    print("=" * 100)
    print("R2.5 — INNOVATION GATE / LEADER HYPOTHESIS ATTRIBUTION")
    print("=" * 100)
    print("poisoning query:", query_id)

    print()
    print("R2 triple")
    print("-" * 100)
    print(
        json.dumps(
            {
                "action": r23["policy_row"]["action"],
                "minimum_innovation_m": r23[
                    "innovation_gate"
                ]["minimum_innovation_m"],
                "tracking_threshold_m": r23[
                    "innovation_gate"
                ]["tracking_threshold_m"],
                "innovation_best_tile_id": r23[
                    "innovation_gate"
                ]["innovation_best_tile_id"],
                "innovation_best_choice_rank": r23[
                    "innovation_gate"
                ]["innovation_best_choice_rank"],
                "leader_hypothesis_id": r23[
                    "leader_hypothesis_id"
                ],
                "leader_tile_ids": (
                    parse_str_list(
                        r23["hypothesis"]["tile_ids"]
                    )
                    if r23["hypothesis"] is not None
                    else []
                ),
                "leader_candidate_choice_ranks": (
                    parse_int_list(
                        r23["hypothesis"][
                            "candidate_choice_ranks"
                        ]
                    )
                    if r23["hypothesis"] is not None
                    else []
                ),
                "leader_support_variants": r23[
                    "support_variants"
                ],
                "gate_tile_in_leader": r23[
                    "innovation_gate"
                ][
                    "gate_tile_in_accepted_leader_hypothesis"
                ],
                "gate_choice_matches_leader_current_q": r23[
                    "innovation_gate"
                ][
                    "gate_choice_matches_accepted_leader_current_q_observation"
                ],
                "leader_scale": (
                    r23["hypothesis"][
                        "scale_m_per_visual_px"
                    ]
                    if r23["hypothesis"] is not None
                    else None
                ),
                "leader_rotation_deg": (
                    r23["hypothesis"]["rotation_deg"]
                    if r23["hypothesis"] is not None
                    else None
                ),
                "leader_median_residual_m": (
                    r23["hypothesis"][
                        "median_projected_residual_m"
                    ]
                    if r23["hypothesis"] is not None
                    else None
                ),
                "leader_max_residual_m": (
                    r23["hypothesis"][
                        "max_projected_residual_m"
                    ]
                    if r23["hypothesis"] is not None
                    else None
                ),
            },
            indent=2,
            default=str,
        )
    )

    print()
    print("R2 accepted-leader supporting observations")
    print("-" * 100)
    for row in r23["supporting_observations"]:
        print(
            json.dumps(
                {
                    "query_id": row.get("query_id"),
                    "tile_id": row.get("tile_id"),
                    "tile_variant": row.get(
                        "tile_variant"
                    ),
                    "candidate_choice_rank": row.get(
                        "candidate_choice_rank"
                    ),
                    "projected_easting": row.get(
                        "projected_easting"
                    ),
                    "projected_northing": row.get(
                        "projected_northing"
                    ),
                    "inliers": row.get("inliers"),
                    "recomputed_homography_ok": row.get(
                        "recomputed_homography_ok"
                    ),
                },
                default=str,
            )
        )

    print()
    print("Stable 768 control at q99")
    print("-" * 100)
    print(
        json.dumps(
            {
                "action": r14["policy_row"]["action"],
                "minimum_innovation_m": r14[
                    "innovation_gate"
                ]["minimum_innovation_m"],
                "tracking_threshold_m": r14[
                    "innovation_gate"
                ]["tracking_threshold_m"],
                "innovation_best_tile_id": r14[
                    "innovation_gate"
                ]["innovation_best_tile_id"],
                "innovation_best_choice_rank": r14[
                    "innovation_gate"
                ]["innovation_best_choice_rank"],
                "leader_hypothesis_id": r14[
                    "leader_hypothesis_id"
                ],
                "leader_tile_ids": (
                    parse_str_list(
                        r14["hypothesis"]["tile_ids"]
                    )
                    if r14["hypothesis"] is not None
                    else []
                ),
                "leader_candidate_choice_ranks": (
                    parse_int_list(
                        r14["hypothesis"][
                            "candidate_choice_ranks"
                        ]
                    )
                    if r14["hypothesis"] is not None
                    else []
                ),
                "gate_tile_in_leader": r14[
                    "innovation_gate"
                ][
                    "gate_tile_in_accepted_leader_hypothesis"
                ],
                "gate_choice_matches_leader_current_q": r14[
                    "innovation_gate"
                ][
                    "gate_choice_matches_accepted_leader_current_q_observation"
                ],
                "leader_scale": (
                    r14["hypothesis"][
                        "scale_m_per_visual_px"
                    ]
                    if r14["hypothesis"] is not None
                    else None
                ),
                "leader_rotation_deg": (
                    r14["hypothesis"]["rotation_deg"]
                    if r14["hypothesis"] is not None
                    else None
                ),
            },
            indent=2,
            default=str,
        )
    )

    print()
    print("STATUS: PASS_R2_GATE_LEADER_ATTRIBUTION")
    print("report:", report_path)


if __name__ == "__main__":
    main()
