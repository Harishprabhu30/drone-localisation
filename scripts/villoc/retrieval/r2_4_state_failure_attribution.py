#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer


ROOT = Path.cwd().resolve()

DEFAULT_R23_ROOT = Path(
    "outputs/research_runs/retrieval_candidate_pool_v2/"
    "r2_cross_scale_fusion/r2_3_state_replay"
)
DEFAULT_R22_ROOT = Path(
    "outputs/research_runs/retrieval_candidate_pool_v2/"
    "r2_cross_scale_fusion/r2_2_blind_orb"
)
DEFAULT_R14_768_ROOT = Path(
    "outputs/research_runs/retrieval_candidate_pool_v2/"
    "r1_4_downstream_replay/runs/768_s256"
)
DEFAULT_OUTPUT_ROOT = Path(
    "outputs/research_runs/retrieval_candidate_pool_v2/"
    "r2_cross_scale_fusion/r2_4_failure_attribution"
)

THRESHOLDS_M = (100.0, 200.0, 300.0)


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


def wrap_degrees(value: float) -> float:
    return float((value + 180.0) % 360.0 - 180.0)


def transform_delta(previous: dict | None, current: dict) -> dict:
    if previous is None:
        return {
            "scale_ratio": None,
            "scale_change_pct": None,
            "rotation_delta_deg": None,
            "translation_delta_m": None,
        }

    prev_scale = float(previous["scale_m_per_visual_px"])
    curr_scale = float(current["scale_m_per_visual_px"])

    scale_ratio = curr_scale / prev_scale
    rotation_delta = wrap_degrees(
        float(current["rotation_deg"])
        - float(previous["rotation_deg"])
    )
    translation_delta = math.hypot(
        float(current["b_real"]) - float(previous["b_real"]),
        float(current["b_imag"]) - float(previous["b_imag"]),
    )

    return {
        "scale_ratio": float(scale_ratio),
        "scale_change_pct": float(
            100.0 * (scale_ratio - 1.0)
        ),
        "rotation_delta_deg": float(rotation_delta),
        "translation_delta_m": float(translation_delta),
    }


def event_table(timeline: dict) -> pd.DataFrame:
    rows = []
    previous = None

    for event in timeline.get("events", []):
        transform = event["transform"]
        delta = transform_delta(previous, transform)

        rows.append(
            {
                "event_index": int(event["event_index"]),
                "effective_query_id": int(
                    event["effective_from_query_id"]
                ),
                "source_query_id": int(
                    event["source_update_query_id"]
                ),
                "source_hypothesis_id": event.get(
                    "source_hypothesis_id"
                ),
                "action": str(event.get("action", "")),
                "scale_m_per_visual_px": float(
                    transform["scale_m_per_visual_px"]
                ),
                "rotation_deg": float(
                    transform["rotation_deg"]
                ),
                "a_real": float(transform["a_real"]),
                "a_imag": float(transform["a_imag"]),
                "b_real": float(transform["b_real"]),
                "b_imag": float(transform["b_imag"]),
                **delta,
            }
        )

        previous = transform

    return pd.DataFrame(rows)


def attach_error_context(
    events: pd.DataFrame,
    attachment: pd.DataFrame,
) -> pd.DataFrame:
    out = events.copy()

    errors = (
        attachment[
            ["query_id", "eval_position_error_m"]
        ]
        .copy()
    )
    errors["query_id"] = pd.to_numeric(
        errors["query_id"],
        errors="raise",
    ).astype(int)
    errors["eval_position_error_m"] = pd.to_numeric(
        errors["eval_position_error_m"],
        errors="coerce",
    )
    error_by_q = errors.set_index("query_id")[
        "eval_position_error_m"
    ]

    before = []
    at = []
    after = []

    for qid in out["effective_query_id"].astype(int):
        before.append(
            float(error_by_q.get(qid - 1, np.nan))
        )
        at.append(
            float(error_by_q.get(qid, np.nan))
        )
        after.append(
            float(error_by_q.get(qid + 1, np.nan))
        )

    out["trajectory_error_before_m"] = before
    out["trajectory_error_at_m"] = at
    out["trajectory_error_after_m"] = after
    out["trajectory_error_jump_m"] = (
        out["trajectory_error_at_m"]
        - out["trajectory_error_before_m"]
    )

    return out


def first_threshold_crossings(
    attachment: pd.DataFrame,
) -> dict[str, dict | None]:
    frame = attachment.copy()
    frame["query_id"] = pd.to_numeric(
        frame["query_id"],
        errors="raise",
    ).astype(int)
    frame["eval_position_error_m"] = pd.to_numeric(
        frame["eval_position_error_m"],
        errors="coerce",
    )

    result = {}

    for threshold in THRESHOLDS_M:
        hit = frame[
            frame["eval_position_error_m"] > threshold
        ]

        key = str(int(threshold))

        if hit.empty:
            result[key] = None
        else:
            row = hit.iloc[0]
            result[key] = {
                "query_id": int(row["query_id"]),
                "error_m": float(
                    row["eval_position_error_m"]
                ),
            }

    return result


def last_event_at_or_before(
    events: pd.DataFrame,
    query_id: int,
) -> dict | None:
    sub = events[
        events["effective_query_id"] <= int(query_id)
    ]

    if sub.empty:
        return None

    row = sub.iloc[-1]
    return {
        key: (
            value.item()
            if isinstance(value, np.generic)
            else value
        )
        for key, value in row.to_dict().items()
    }


def next_event_after(
    events: pd.DataFrame,
    query_id: int,
) -> dict | None:
    sub = events[
        events["effective_query_id"] > int(query_id)
    ]

    if sub.empty:
        return None

    row = sub.iloc[0]
    return {
        key: (
            value.item()
            if isinstance(value, np.generic)
            else value
        )
        for key, value in row.to_dict().items()
    }


def load_reference_xy(
    attachment: pd.DataFrame,
) -> pd.DataFrame:
    required = {
        "query_id",
        "eval_ref_lon",
        "eval_ref_lat",
    }
    missing = required - set(attachment.columns)
    if missing:
        raise RuntimeError(
            f"Evaluation attachment missing {sorted(missing)}"
        )

    frame = attachment[
        ["query_id", "eval_ref_lon", "eval_ref_lat"]
    ].copy()
    frame["query_id"] = pd.to_numeric(
        frame["query_id"],
        errors="raise",
    ).astype(int)

    transformer = Transformer.from_crs(
        "EPSG:4326",
        "EPSG:3346",
        always_xy=True,
    )
    x, y = transformer.transform(
        pd.to_numeric(
            frame["eval_ref_lon"],
            errors="raise",
        ).to_numpy(float),
        pd.to_numeric(
            frame["eval_ref_lat"],
            errors="raise",
        ).to_numpy(float),
    )

    frame["gt_x"] = x
    frame["gt_y"] = y

    return frame.set_index("query_id")


def evaluate_candidate_pool(
    *,
    query_id: int,
    scored: pd.DataFrame,
    tile_index: pd.DataFrame,
    reference_xy: pd.DataFrame,
    label: str,
) -> dict:
    query = scored[
        pd.to_numeric(
            scored["query_id"],
            errors="coerce",
        ) == int(query_id)
    ].copy()

    if query.empty:
        return {
            "label": label,
            "query_id": int(query_id),
            "present": False,
        }

    tile_lookup = (
        tile_index.assign(
            tile_id=tile_index["tile_id"].astype(str)
        )
        .set_index("tile_id")
    )

    if int(query_id) not in reference_xy.index:
        raise RuntimeError(
            f"Reference missing q{query_id}"
        )

    ref = reference_xy.loc[int(query_id)]
    gt_x = float(ref["gt_x"])
    gt_y = float(ref["gt_y"])

    query["tile_id"] = query["tile_id"].astype(str)
    query["hybrid_rank"] = pd.to_numeric(
        query["hybrid_rank"],
        errors="coerce",
    )
    query["verifier_rank"] = pd.to_numeric(
        query["verifier_rank"],
        errors="coerce",
    )
    query["rank"] = pd.to_numeric(
        query["rank"],
        errors="coerce",
    )

    errors = []
    contains = []

    for row in query.itertuples(index=False):
        tid = str(row.tile_id)

        if tid not in tile_lookup.index:
            errors.append(np.nan)
            contains.append(False)
            continue

        tile = tile_lookup.loc[tid]
        errors.append(
            math.hypot(
                float(tile["center_easting"]) - gt_x,
                float(tile["center_northing"]) - gt_y,
            )
        )
        contains.append(
            bool(
                float(tile["left_easting"]) <= gt_x
                <= float(tile["right_easting"])
                and float(tile["bottom_northing"]) <= gt_y
                <= float(tile["top_northing"])
            )
        )

    query["postfreeze_center_error_m"] = errors
    query["postfreeze_contains_query"] = contains

    query = query.sort_values(
        [
            "hybrid_rank",
            "verifier_rank",
            "rank",
            "tile_id",
        ],
        kind="mergesort",
    ).reset_index(drop=True)
    query["bootstrap_choice_rank"] = np.arange(
        1,
        len(query) + 1,
    )

    valid = query[
        np.isfinite(
            query["postfreeze_center_error_m"]
        )
    ]

    best_error_row = (
        valid.sort_values(
            "postfreeze_center_error_m"
        ).iloc[0]
        if not valid.empty
        else None
    )
    containing = query[
        query["postfreeze_contains_query"]
    ]

    selected = query.iloc[0]
    top4 = query.head(4)

    columns = [
        col
        for col in (
            "tile_id",
            "rank",
            "hybrid_rank",
            "verifier_rank",
            "bootstrap_choice_rank",
            "inliers",
            "inlier_ratio",
            "homography_ok",
            "verifier_score",
            "hybrid_score",
            "representative_variant",
            "representative_original_tile_id",
            "support_scales",
            "postfreeze_center_error_m",
            "postfreeze_contains_query",
        )
        if col in query.columns
    ]

    def row_dict(row: pd.Series | None):
        if row is None:
            return None
        data = {}
        for col in columns:
            value = row[col]
            if isinstance(value, np.generic):
                value = value.item()
            if pd.isna(value):
                value = None
            data[col] = value
        return data

    return {
        "label": label,
        "query_id": int(query_id),
        "present": True,
        "candidate_count": int(len(query)),
        "selected_hybrid_top1": row_dict(selected),
        "best_center_error_candidate": row_dict(
            best_error_row
        ),
        "first_containing_candidate": row_dict(
            containing.iloc[0]
            if not containing.empty
            else None
        ),
        "containing_candidate_count": int(
            query["postfreeze_contains_query"].sum()
        ),
        "le40_candidate_count": int(
            (
                query["postfreeze_center_error_m"]
                <= 40.0
            ).sum()
        ),
        "le80_candidate_count": int(
            (
                query["postfreeze_center_error_m"]
                <= 80.0
            ).sum()
        ),
        "top4_contains_count": int(
            top4["postfreeze_contains_query"].sum()
        ),
        "top4_le40_count": int(
            (
                top4["postfreeze_center_error_m"]
                <= 40.0
            ).sum()
        ),
        "top4_le80_count": int(
            (
                top4["postfreeze_center_error_m"]
                <= 80.0
            ).sum()
        ),
        "top4": [
            row_dict(row)
            for _, row in top4.iterrows()
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "R2.4 post-freeze failure attribution for the failed R2.3 state "
            "replay. No localization decisions are changed."
        )
    )
    parser.add_argument(
        "--r2-3-root",
        type=Path,
        default=DEFAULT_R23_ROOT,
    )
    parser.add_argument(
        "--r2-2-root",
        type=Path,
        default=DEFAULT_R22_ROOT,
    )
    parser.add_argument(
        "--r1-4-768-root",
        type=Path,
        default=DEFAULT_R14_768_ROOT,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    args = parser.parse_args()

    r23_root = resolve(args.r2_3_root)
    r22_root = resolve(args.r2_2_root)
    r14_768_root = resolve(args.r1_4_768_root)
    output_root = resolve(args.output_root)

    r23_run = r23_root / "run"

    paths = {
        "r23_timeline": (
            r23_run
            / "reports/blind_map_bootstrap/"
              "canonical_map_state_timeline.json"
        ),
        "r23_attachment": (
            r23_run
            / "evaluation/reference_attachment.csv"
        ),
        "r22_scores": (
            r22_root
            / "reports/r2_2_triple_orb_hybrid_top20/"
              "s8_12e1_all_candidate_verifier_scores.csv"
        ),
        "r22_mixed_index": (
            r22_root
            / "metadata/r2_2_mixed_representative_tile_index.csv"
        ),
        "r22_report": (
            r22_root
            / "reports/r2_2_triple_fusion_orb_report.json"
        ),
        "r14_timeline": (
            r14_768_root
            / "reports/blind_map_bootstrap/"
              "canonical_map_state_timeline.json"
        ),
        "r14_attachment": (
            r14_768_root
            / "evaluation/reference_attachment.csv"
        ),
        "r14_scores": (
            r14_768_root
            / "reports/s8_12e1_top20_verifier_reranker/"
              "768_s256_orb_hybrid_top20_img518/"
              "s8_12e1_all_candidate_verifier_scores.csv"
        ),
        "r14_index": (
            ROOT
            / "outputs/research_runs/retrieval_candidate_pool_v2/"
              "r1_map_pyramid/metadata/"
              "s8_9_satellite_tile_index_768_s256.csv"
        ),
    }

    missing = [
        f"{name}: {path}"
        for name, path in paths.items()
        if not path.exists()
    ]
    if missing:
        raise RuntimeError(
            "R2.4 required evidence missing:\n"
            + "\n".join(missing)
        )

    hashes_before = {
        name: sha256_file(path)
        for name, path in paths.items()
    }

    r23_timeline = load_json(paths["r23_timeline"])
    r14_timeline = load_json(paths["r14_timeline"])
    r22_report = load_json(paths["r22_report"])

    r23_attachment = pd.read_csv(
        paths["r23_attachment"]
    )
    r14_attachment = pd.read_csv(
        paths["r14_attachment"]
    )
    r22_scores = pd.read_csv(
        paths["r22_scores"]
    )
    r14_scores = pd.read_csv(
        paths["r14_scores"]
    )
    r22_index = pd.read_csv(
        paths["r22_mixed_index"]
    )
    r14_index = pd.read_csv(
        paths["r14_index"]
    )

    r23_events = attach_error_context(
        event_table(r23_timeline),
        r23_attachment,
    )
    r14_events = attach_error_context(
        event_table(r14_timeline),
        r14_attachment,
    )

    if r23_events.empty:
        raise RuntimeError("R2.3 timeline has no state events.")

    crossings = first_threshold_crossings(
        r23_attachment
    )

    q100 = (
        int(crossings["100"]["query_id"])
        if crossings["100"] is not None
        else None
    )

    if q100 is None:
        raise RuntimeError(
            "R2.3 never exceeded 100 m; failure-attribution trigger absent."
        )

    poisoning_event = last_event_at_or_before(
        r23_events,
        q100,
    )
    next_r23_event = next_event_after(
        r23_events,
        q100,
    )

    if poisoning_event is None:
        raise RuntimeError(
            "No R2.3 map-state event exists before first >100 m error."
        )

    source_q = int(
        poisoning_event["source_query_id"]
    )
    effective_q = int(
        poisoning_event["effective_query_id"]
    )

    reference_xy = load_reference_xy(
        r23_attachment
    )

    r22_pool = evaluate_candidate_pool(
        query_id=source_q,
        scored=r22_scores,
        tile_index=r22_index,
        reference_xy=reference_xy,
        label="r2_triple_fused_orb",
    )

    r14_pool = evaluate_candidate_pool(
        query_id=source_q,
        scored=r14_scores,
        tile_index=r14_index,
        reference_xy=reference_xy,
        label="r1_4_768_single_scale",
    )

    stable_768_event = last_event_at_or_before(
        r14_events,
        effective_q,
    )

    # Rank suspicious R2.3 events using only post-freeze diagnostics.
    ranked_events = r23_events.copy()
    ranked_events["abs_rotation_delta_deg"] = (
        pd.to_numeric(
            ranked_events["rotation_delta_deg"],
            errors="coerce",
        ).abs()
    )
    ranked_events["abs_scale_change_pct"] = (
        pd.to_numeric(
            ranked_events["scale_change_pct"],
            errors="coerce",
        ).abs()
    )
    ranked_events["abs_error_jump_m"] = (
        pd.to_numeric(
            ranked_events["trajectory_error_jump_m"],
            errors="coerce",
        ).abs()
    )

    ranked_events = ranked_events.sort_values(
        [
            "abs_error_jump_m",
            "abs_rotation_delta_deg",
            "abs_scale_change_pct",
        ],
        ascending=False,
        na_position="last",
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    events_csv = (
        output_root
        / "r2_4_r23_state_events_with_postfreeze_error.csv"
    )
    r23_events.to_csv(
        events_csv,
        index=False,
    )

    comparison_csv = (
        output_root
        / "r2_4_768_state_events_with_postfreeze_error.csv"
    )
    r14_events.to_csv(
        comparison_csv,
        index=False,
    )

    top_suspicious = []
    for _, row in ranked_events.head(10).iterrows():
        item = {}
        for key, value in row.to_dict().items():
            if isinstance(value, np.generic):
                value = value.item()
            if pd.isna(value):
                value = None
            item[key] = value
        top_suspicious.append(item)

    report = {
        "stage": "R2.4",
        "status": "PASS_R2_STATE_FAILURE_ATTRIBUTION",
        "created_at_utc": now_utc(),
        "purpose": (
            "Post-freeze diagnostic only. Attribute the failed R2.3 trajectory "
            "to accepted state transitions and determine whether useful "
            "fused+ORB candidates existed at the poisoning source query."
        ),
        "r2_3_threshold_crossings": crossings,
        "poisoning_event_candidate": poisoning_event,
        "next_r2_3_event_after_first_gt100": next_r23_event,
        "poisoning_source_query_id": source_q,
        "poisoning_effective_query_id": effective_q,
        "candidate_pool_at_poisoning_source_query": {
            "r2_triple_fused_orb": r22_pool,
            "stable_768_control": r14_pool,
        },
        "stable_768_state_at_same_effective_query": (
            stable_768_event
        ),
        "top_suspicious_r2_3_state_events": top_suspicious,
        "r2_2_named_diagnostics": {
            key: r22_report.get(
                "named_diagnostics",
                {},
            ).get(key)
            for key in ("q57", "q228", "q390")
        },
        "interpretation_contract": {
            "candidate_available": (
                "A low-error or containing candidate exists in the full "
                "R2.2 ORB-scored Top-20."
            ),
            "bootstrap_visible": (
                "A candidate is in bootstrap choice ranks 1..4 after the "
                "unchanged hybrid/verifier ordering."
            ),
            "state_poisoning": (
                "A map-state event is considered a poisoning candidate when "
                "it is the last accepted event at/before the first >100 m "
                "post-freeze trajectory error. This is attribution evidence, "
                "not a causal proof by intervention."
            ),
        },
        "scope_guarantees": {
            "localization_rerun": False,
            "orb_rerun": False,
            "state_policy_changed": False,
            "reference_used_for_localization": False,
            "postfreeze_analysis_only": True,
        },
        "inputs": {
            name: {
                "path": str(path),
                "sha256": hashes_before[name],
            }
            for name, path in paths.items()
        },
        "outputs": {
            "r23_events_csv": str(events_csv),
            "r14_768_events_csv": str(comparison_csv),
        },
        "next_decision": (
            "Use the poisoning-event and candidate-pool evidence to decide "
            "whether R2 should close with 768 as the promoted baseline, or "
            "whether a separately scoped verifier/reranker stage is warranted."
        ),
    }

    report_path = (
        output_root
        / "r2_4_state_failure_attribution_report.json"
    )
    report_path.write_text(
        json.dumps(
            report,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    hashes_after = {
        name: sha256_file(path)
        for name, path in paths.items()
    }

    if hashes_before != hashes_after:
        raise RuntimeError(
            "Diagnostic source evidence changed during R2.4."
        )

    print("=" * 100)
    print("R2.4 — STATE FAILURE ATTRIBUTION")
    print("=" * 100)
    print("first >100 m:", crossings["100"])
    print("first >200 m:", crossings["200"])
    print("first >300 m:", crossings["300"])

    print()
    print("Poisoning-event candidate")
    print("-" * 100)
    print(
        json.dumps(
            poisoning_event,
            indent=2,
            default=str,
        )
    )

    print()
    print("R2 fused+ORB candidate pool at poisoning source query")
    print("-" * 100)
    print(
        json.dumps(
            r22_pool,
            indent=2,
            default=str,
        )
    )

    print()
    print("Stable 768 candidate pool at same query")
    print("-" * 100)
    print(
        json.dumps(
            r14_pool,
            indent=2,
            default=str,
        )
    )

    print()
    print("Stable 768 active state at same effective query")
    print("-" * 100)
    print(
        json.dumps(
            stable_768_event,
            indent=2,
            default=str,
        )
    )

    print()
    print("STATUS: PASS_R2_STATE_FAILURE_ATTRIBUTION")
    print("report:", report_path)


if __name__ == "__main__":
    main()
