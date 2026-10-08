#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from pyproj import Transformer


ROOT = Path.cwd().resolve()
DEFAULT_CONFIG = Path(
    "configs/research/absolute_frontend_decoupling_v1.yaml"
)
DEFAULT_OUTPUT_ROOT = Path(
    "outputs/research_runs/absolute_frontend_decoupling_v1/r3_2"
)


def resolve(path_like: str | Path) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else ROOT / path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def composite_tile_id(variant: str, tile_id: str) -> str:
    return f"{variant}::{tile_id}"


def parse_members(value: str) -> list[dict[str, Any]]:
    members = json.loads(str(value))
    if not isinstance(members, list):
        raise RuntimeError("members_json must decode to a list")
    return [dict(item) for item in members]


def load_reference_xy(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"query_id", "eval_ref_lon", "eval_ref_lat"}
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(
            f"Reference attachment missing {sorted(missing)}"
        )

    frame = frame.copy()
    frame["query_id"] = pd.to_numeric(
        frame["query_id"], errors="raise"
    ).astype(int)

    transformer = Transformer.from_crs(
        "EPSG:4326",
        "EPSG:3346",
        always_xy=True,
    )
    x, y = transformer.transform(
        pd.to_numeric(
            frame["eval_ref_lon"], errors="raise"
        ).to_numpy(float),
        pd.to_numeric(
            frame["eval_ref_lat"], errors="raise"
        ).to_numpy(float),
    )
    frame["gt_x"] = x
    frame["gt_y"] = y
    return frame.set_index("query_id")


def prepare_composite_index(
    specs: list[tuple[str, Path]],
) -> pd.DataFrame:
    frames = []

    for variant, path in specs:
        frame = pd.read_csv(path).copy()
        required = {
            "tile_id",
            "left_easting",
            "bottom_northing",
            "right_easting",
            "top_northing",
            "center_easting",
            "center_northing",
        }
        missing = required - set(frame.columns)
        if missing:
            raise RuntimeError(
                f"{variant} tile index missing {sorted(missing)}"
            )

        frame["source_tile_id"] = frame["tile_id"].astype(str)
        frame["source_variant"] = variant
        frame["tile_id"] = [
            composite_tile_id(variant, tid)
            for tid in frame["source_tile_id"]
        ]
        frames.append(frame)

    mixed = pd.concat(frames, ignore_index=True)
    if mixed["tile_id"].duplicated().any():
        raise RuntimeError("Composite tile IDs are not unique")

    return mixed.set_index("tile_id")


def load_regions(
    fused_path: Path,
    provenance_path: Path,
) -> dict[int, list[dict[str, Any]]]:
    fused = pd.read_csv(fused_path).copy()
    provenance = pd.read_csv(provenance_path).copy()

    fused["query_id"] = pd.to_numeric(
        fused["query_id"], errors="raise"
    ).astype(int)
    fused["rank"] = pd.to_numeric(
        fused["rank"], errors="raise"
    ).astype(int)
    provenance["query_id"] = pd.to_numeric(
        provenance["query_id"], errors="raise"
    ).astype(int)
    provenance["fused_rank"] = pd.to_numeric(
        provenance["fused_rank"], errors="raise"
    ).astype(int)

    merged = fused.merge(
        provenance,
        left_on=["query_id", "rank", "tile_id"],
        right_on=[
            "query_id",
            "fused_rank",
            "composite_tile_id",
        ],
        how="left",
        validate="one_to_one",
    )

    if merged["members_json"].isna().any():
        bad = merged[merged["members_json"].isna()].head(5)
        raise RuntimeError(
            "Missing provenance for fused rows: "
            + bad[["query_id", "rank", "tile_id"]].to_json(
                orient="records"
            )
        )

    by_query: dict[int, list[dict[str, Any]]] = {}

    for qid, group in merged.groupby("query_id", sort=True):
        regions = []
        for row in group.sort_values("rank").itertuples(index=False):
            members = parse_members(row.members_json)
            regions.append(
                {
                    "query_id": int(qid),
                    "fused_rank": int(row.rank),
                    "representative_composite_tile_id": str(
                        row.tile_id
                    ),
                    "representative_variant": str(
                        row.representative_variant
                    ),
                    "representative_original_tile_id": str(
                        row.representative_original_tile_id
                    ),
                    "fused_rrf_score": float(
                        row.fused_rrf_score
                    ),
                    "support_scale_count": int(
                        row.support_scale_count
                    ),
                    "support_scales": str(row.support_scales),
                    "members": members,
                }
            )
        by_query[int(qid)] = regions

    return by_query


def region_priority_key(region: dict[str, Any]) -> tuple:
    return (
        -int(region["support_scale_count"]),
        -float(region["fused_rrf_score"]),
        int(region["fused_rank"]),
    )


def best_support_region(
    regions: list[dict[str, Any]],
) -> dict[str, Any]:
    if not regions:
        raise RuntimeError("No fused regions available")
    return min(regions, key=region_priority_key)


def find_768_anchor_region(
    regions: list[dict[str, Any]],
    tile_id_768: str,
) -> dict[str, Any] | None:
    matches = []
    for region in regions:
        for member in region["members"]:
            if (
                str(member["variant"]) == "768_s256"
                and str(member["tile_id"]) == str(tile_id_768)
            ):
                matches.append(region)
                break

    if len(matches) > 1:
        raise RuntimeError(
            f"768 anchor {tile_id_768} appears in multiple regions"
        )
    return matches[0] if matches else None


def select_region_member(
    region: dict[str, Any],
    member_policy: str,
) -> tuple[str, str]:
    if member_policy == "representative":
        return (
            str(region["representative_composite_tile_id"]),
            "region_representative",
        )

    if member_policy != "prefer_768":
        raise ValueError(
            f"Unknown member policy: {member_policy}"
        )

    members_768 = [
        member
        for member in region["members"]
        if str(member["variant"]) == "768_s256"
    ]

    if len(members_768) > 1:
        raise RuntimeError(
            "A fused region contains more than one 768 member"
        )

    if len(members_768) == 1:
        member = members_768[0]
        return (
            composite_tile_id(
                "768_s256",
                str(member["tile_id"]),
            ),
            "preferred_768_member",
        )

    return (
        str(region["representative_composite_tile_id"]),
        "fallback_region_representative",
    )


def select_for_query(
    *,
    qid: int,
    tile_id_768: str,
    regions: list[dict[str, Any]],
    policy_name: str,
    missing_anchor_support_count: int,
) -> dict[str, Any]:
    baseline_composite = composite_tile_id(
        "768_s256",
        tile_id_768,
    )
    anchor = find_768_anchor_region(
        regions,
        tile_id_768,
    )
    anchor_support = (
        int(anchor["support_scale_count"])
        if anchor is not None
        else int(missing_anchor_support_count)
    )
    best = best_support_region(regions)

    if policy_name == "768_dino_top1":
        chosen = baseline_composite
        chosen_region = anchor
        member_reason = "direct_768_top1"
        reason = "baseline"
    elif policy_name == "triple_support_first_representative":
        chosen, member_reason = select_region_member(
            best,
            "representative",
        )
        chosen_region = best
        reason = "support_first"
    elif policy_name == "triple_support_first_768_preferred":
        chosen, member_reason = select_region_member(
            best,
            "prefer_768",
        )
        chosen_region = best
        reason = "support_first"
    elif policy_name == "768_strict_support_rescue":
        if int(best["support_scale_count"]) > anchor_support:
            chosen, member_reason = select_region_member(
                best,
                "prefer_768",
            )
            chosen_region = best
            reason = "rescue_strictly_more_support"
        else:
            chosen = baseline_composite
            chosen_region = anchor
            member_reason = "direct_768_top1"
            reason = "hold_768_no_stronger_support"
    elif policy_name == "768_three_scale_rescue":
        if (
            int(best["support_scale_count"]) == 3
            and anchor_support < 3
        ):
            chosen, member_reason = select_region_member(
                best,
                "prefer_768",
            )
            chosen_region = best
            reason = "rescue_three_scale_consensus"
        else:
            chosen = baseline_composite
            chosen_region = anchor
            member_reason = "direct_768_top1"
            reason = "hold_768_no_three_scale_advantage"
    else:
        raise ValueError(
            f"Unknown policy: {policy_name}"
        )

    if "::" not in chosen:
        raise RuntimeError(
            f"Expected composite chosen tile ID, got {chosen}"
        )
    chosen_variant, chosen_original = chosen.split("::", 1)

    return {
        "selection": policy_name,
        "query_id": int(qid),
        "tile_id": chosen,
        "chosen_variant": chosen_variant,
        "chosen_original_tile_id": chosen_original,
        "selection_reason": reason,
        "member_selection_reason": member_reason,
        "switched_from_768_top1": bool(
            chosen != baseline_composite
        ),
        "anchor_region_found": bool(
            anchor is not None
        ),
        "anchor_support_scale_count": int(
            anchor_support
        ),
        "anchor_fused_rank": (
            int(anchor["fused_rank"])
            if anchor is not None
            else None
        ),
        "best_region_fused_rank": int(
            best["fused_rank"]
        ),
        "best_region_support_scale_count": int(
            best["support_scale_count"]
        ),
        "best_region_support_scales": str(
            best["support_scales"]
        ),
        "best_region_fused_rrf_score": float(
            best["fused_rrf_score"]
        ),
        "chosen_region_fused_rank": (
            int(chosen_region["fused_rank"])
            if chosen_region is not None
            else None
        ),
        "chosen_region_support_scale_count": (
            int(chosen_region["support_scale_count"])
            if chosen_region is not None
            else 1
        ),
        "chosen_region_support_scales": (
            str(chosen_region["support_scales"])
            if chosen_region is not None
            else "768_s256"
        ),
    }


def evaluate_selection(
    selected: pd.DataFrame,
    tile_index: pd.DataFrame,
    reference_xy: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for row in selected.itertuples(index=False):
        qid = int(row.query_id)
        tile_id = str(row.tile_id)

        if qid not in reference_xy.index:
            raise RuntimeError(
                f"Reference missing q{qid}"
            )
        if tile_id not in tile_index.index:
            raise RuntimeError(
                f"Composite tile index missing {tile_id}"
            )

        ref = reference_xy.loc[qid]
        tile = tile_index.loc[tile_id]

        x = float(ref["gt_x"])
        y = float(ref["gt_y"])
        error = float(
            math.hypot(
                float(tile["center_easting"]) - x,
                float(tile["center_northing"]) - y,
            )
        )
        contains = bool(
            float(tile["left_easting"]) <= x
            <= float(tile["right_easting"])
            and float(tile["bottom_northing"]) <= y
            <= float(tile["top_northing"])
        )

        base = row._asdict()
        base.update(
            {
                "center_error_m": error,
                "contains_query": contains,
                "le40": bool(error <= 40.0),
                "le80": bool(error <= 80.0),
            }
        )
        rows.append(base)

    return pd.DataFrame(rows)


def summarize(frame: pd.DataFrame) -> dict[str, Any]:
    errors = frame["center_error_m"].to_numpy(float)
    return {
        "query_count": int(len(frame)),
        "switches_from_768": int(
            frame["switched_from_768_top1"].sum()
        ),
        "contains_hits": int(
            frame["contains_query"].sum()
        ),
        "contains_rate": float(
            frame["contains_query"].mean()
        ),
        "le40_hits": int(frame["le40"].sum()),
        "le40_rate": float(frame["le40"].mean()),
        "le80_hits": int(frame["le80"].sum()),
        "le80_rate": float(frame["le80"].mean()),
        "center_error_mean_m": float(
            np.mean(errors)
        ),
        "center_error_median_m": float(
            np.median(errors)
        ),
        "center_error_p95_m": float(
            np.percentile(errors, 95)
        ),
        "center_error_max_m": float(
            np.max(errors)
        ),
    }


def switch_diagnostics(
    frame: pd.DataFrame,
    baseline: pd.DataFrame,
) -> dict[str, Any]:
    left = frame[
        [
            "query_id",
            "switched_from_768_top1",
            "center_error_m",
            "contains_query",
            "le40",
            "le80",
        ]
    ].copy()
    right = baseline[
        [
            "query_id",
            "center_error_m",
            "contains_query",
            "le40",
            "le80",
        ]
    ].copy()

    merged = left.merge(
        right,
        on="query_id",
        suffixes=("_policy", "_768"),
        validate="one_to_one",
    )
    switched = merged[
        merged["switched_from_768_top1"]
    ].copy()

    if switched.empty:
        return {
            "switch_count": 0,
            "center_error_improved": 0,
            "center_error_worsened": 0,
            "center_error_tied": 0,
            "containment_gain": 0,
            "containment_loss": 0,
            "le40_gain": 0,
            "le40_loss": 0,
            "le80_gain": 0,
            "le80_loss": 0,
            "median_error_delta_m": None,
        }

    delta = (
        switched["center_error_m_policy"]
        - switched["center_error_m_768"]
    )
    eps = 1e-9

    return {
        "switch_count": int(len(switched)),
        "center_error_improved": int(
            (delta < -eps).sum()
        ),
        "center_error_worsened": int(
            (delta > eps).sum()
        ),
        "center_error_tied": int(
            (delta.abs() <= eps).sum()
        ),
        "containment_gain": int(
            (
                switched["contains_query_policy"]
                & ~switched["contains_query_768"]
            ).sum()
        ),
        "containment_loss": int(
            (
                ~switched["contains_query_policy"]
                & switched["contains_query_768"]
            ).sum()
        ),
        "le40_gain": int(
            (
                switched["le40_policy"]
                & ~switched["le40_768"]
            ).sum()
        ),
        "le40_loss": int(
            (
                ~switched["le40_policy"]
                & switched["le40_768"]
            ).sum()
        ),
        "le80_gain": int(
            (
                switched["le80_policy"]
                & ~switched["le80_768"]
            ).sum()
        ),
        "le80_loss": int(
            (
                ~switched["le80_policy"]
                & switched["le80_768"]
            ).sum()
        ),
        "median_error_delta_m": float(
            np.median(delta.to_numpy(float))
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    args = parser.parse_args()

    config_path = resolve(args.config)
    output_root = resolve(args.output_root)
    cfg = yaml.safe_load(config_path.read_text())
    r32 = cfg["r3_2"]

    artifacts = {
        name: resolve(value)
        for name, value in cfg["artifacts"].items()
    }

    required = (
        "stable_768_topk",
        "triple_fused_topk",
        "triple_region_provenance",
        "tile_index_512",
        "tile_index_768",
        "tile_index_1024",
        "reference_attachment",
        "r3_1_report",
    )
    for name in required:
        if not artifacts[name].exists():
            raise FileNotFoundError(
                f"{name}: {artifacts[name]}"
            )

    # ==============================================================
    # PHASE 1 — BLIND SELECTION ONLY.
    # No reference, ORB result, or state result is loaded here.
    # ==============================================================
    topk_768 = pd.read_csv(
        artifacts["stable_768_topk"]
    ).copy()
    topk_768["query_id"] = pd.to_numeric(
        topk_768["query_id"], errors="raise"
    ).astype(int)
    topk_768["rank"] = pd.to_numeric(
        topk_768["rank"], errors="raise"
    ).astype(int)
    top1_768 = topk_768[
        topk_768["rank"] == 1
    ][["query_id", "tile_id"]].copy()

    if len(top1_768) != 403:
        raise RuntimeError(
            f"Expected 403 768 Top-1 rows, got {len(top1_768)}"
        )

    regions_by_q = load_regions(
        artifacts["triple_fused_topk"],
        artifacts["triple_region_provenance"],
    )

    policy_names = [
        item["name"]
        for item in r32["policies"]
    ]

    blind_rows = []

    for row in top1_768.itertuples(index=False):
        qid = int(row.query_id)
        tile_id_768 = str(row.tile_id)
        regions = regions_by_q.get(qid, [])

        if len(regions) != 20:
            raise RuntimeError(
                f"q{qid}: expected 20 fused regions, got {len(regions)}"
            )

        for policy_name in policy_names:
            blind_rows.append(
                select_for_query(
                    qid=qid,
                    tile_id_768=tile_id_768,
                    regions=regions,
                    policy_name=policy_name,
                    missing_anchor_support_count=int(
                        r32["missing_anchor_support_count"]
                    ),
                )
            )

    blind = pd.DataFrame(blind_rows)

    for policy_name in policy_names:
        count = int(
            (blind["selection"] == policy_name).sum()
        )
        if count != 403:
            raise RuntimeError(
                f"{policy_name}: expected 403 rows, got {count}"
            )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )
    freeze_path = (
        output_root
        / "r3_2_blind_consensus_selections.csv"
    )
    blind.to_csv(
        freeze_path,
        index=False,
    )
    freeze_sha = sha256_file(freeze_path)

    # ==============================================================
    # PHASE 2 — POST-SELECTION EVALUATION ONLY.
    # ==============================================================
    reference_xy = load_reference_xy(
        artifacts["reference_attachment"]
    )
    full_index = prepare_composite_index(
        [
            ("512_s256", artifacts["tile_index_512"]),
            ("768_s256", artifacts["tile_index_768"]),
            ("1024_s256", artifacts["tile_index_1024"]),
        ]
    )

    evaluated_parts = []
    summaries = {}

    for policy_name in policy_names:
        selected = blind[
            blind["selection"] == policy_name
        ].copy()
        evaluated = evaluate_selection(
            selected,
            full_index,
            reference_xy,
        )
        evaluated_parts.append(evaluated)
        summaries[policy_name] = summarize(
            evaluated
        )

    evaluated_all = pd.concat(
        evaluated_parts,
        ignore_index=True,
    )

    if sha256_file(freeze_path) != freeze_sha:
        raise RuntimeError(
            "Blind R3.2 selection artifact changed during evaluation"
        )

    baseline_eval = evaluated_all[
        evaluated_all["selection"] == "768_dino_top1"
    ].copy()

    switch_stats = {}
    for policy_name in policy_names:
        frame = evaluated_all[
            evaluated_all["selection"] == policy_name
        ].copy()
        switch_stats[policy_name] = switch_diagnostics(
            frame,
            baseline_eval,
        )

    # R3.1 parity is checked after the blind selection freeze.
    r31 = json.loads(
        artifacts["r3_1_report"].read_text()
    )
    r31_baseline = r31["summary"]["768_dino_top1"]
    current_baseline = summaries["768_dino_top1"]

    parity_fields = (
        "contains_hits",
        "le40_hits",
        "le80_hits",
    )
    for field in parity_fields:
        if int(r31_baseline[field]) != int(
            current_baseline[field]
        ):
            raise RuntimeError(
                f"R3.1 parity failed for {field}: "
                f"{r31_baseline[field]} vs {current_baseline[field]}"
            )

    named = {}
    for qid in r32["named_diagnostics"]:
        qid = int(qid)
        named[f"q{qid}"] = {}

        for policy_name in policy_names:
            row = evaluated_all[
                (evaluated_all["selection"] == policy_name)
                & (evaluated_all["query_id"] == qid)
            ]
            named[f"q{qid}"][policy_name] = (
                row.iloc[0].to_dict()
                if len(row) == 1
                else None
            )

    eval_path = (
        output_root
        / "r3_2_consensus_selection_evaluation.csv"
    )
    evaluated_all.to_csv(
        eval_path,
        index=False,
    )

    report = {
        "stage": "R3.2",
        "status": "PASS_R3_RETRIEVAL_NATIVE_CONSENSUS_SELECTOR",
        "created_at_utc": now_utc(),
        "scientific_question": (
            "Can cross-scale physical-region support act as a blind rescue "
            "signal around the strong 768 Top-1 baseline without ORB or state?"
        ),
        "policy_contract": r32,
        "blind_selection_freeze": {
            "path": str(freeze_path),
            "sha256": freeze_sha,
            "reference_used_for_selection": False,
            "orb_used_for_selection": False,
            "state_used_for_selection": False,
        },
        "summaries": summaries,
        "switch_diagnostics_postfreeze": switch_stats,
        "r3_1_baseline_parity": {
            "pass": True,
            "fields": {
                field: current_baseline[field]
                for field in parity_fields
            },
        },
        "named_diagnostics": named,
        "input_hashes": {
            name: {
                "path": str(path),
                "sha256": sha256_file(path),
            }
            for name, path in artifacts.items()
            if path.exists()
        },
        "outputs": {
            "evaluation_csv": str(eval_path),
        },
        "scope_guarantees": {
            "learned_selector_used": False,
            "gt_tuned_threshold_used": False,
            "orb_reranking_run": False,
            "orb_projection_run": False,
            "bootstrap_run": False,
            "state_policy_changed": False,
            "reference_used_before_selection_freeze": False,
        },
        "next_decision": (
            "Promote only a selector that improves or preserves 768 direct "
            "selection. Otherwise retain 768 and move to confidence/temporal "
            "retrieval evidence without tuning thresholds to this trajectory."
        ),
    }

    report_path = (
        output_root
        / "r3_2_consensus_selector_report.json"
    )
    report_path.write_text(
        json.dumps(
            report,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    print("=" * 112)
    print("R3.2 — RETRIEVAL-NATIVE CONSENSUS / RESCUE SELECTOR")
    print("=" * 112)

    for policy_name in policy_names:
        item = summaries[policy_name]
        sw = switch_stats[policy_name]
        print(
            f"{policy_name:42s} "
            f"contains={item['contains_hits']:3d}/403 "
            f"<=40={item['le40_hits']:3d}/403 "
            f"<=80={item['le80_hits']:3d}/403 "
            f"median={item['center_error_median_m']:8.3f} "
            f"p95={item['center_error_p95_m']:8.3f} "
            f"switches={item['switches_from_768']:3d}"
        )
        if sw["switch_count"]:
            print(
                " " * 4
                + "switch diagnostic: "
                + f"improved={sw['center_error_improved']} "
                + f"worsened={sw['center_error_worsened']} "
                + f"contain +{sw['containment_gain']}"
                + f"/-{sw['containment_loss']} "
                + f"<=40 +{sw['le40_gain']}/-{sw['le40_loss']} "
                + f"<=80 +{sw['le80_gain']}/-{sw['le80_loss']}"
            )

    for qid in r32["named_diagnostics"]:
        print()
        print(f"q{int(qid)}:")
        print(
            json.dumps(
                named[f"q{int(qid)}"],
                indent=2,
                default=str,
            )
        )

    print()
    print("STATUS: PASS_R3_RETRIEVAL_NATIVE_CONSENSUS_SELECTOR")
    print("report:", report_path)


if __name__ == "__main__":
    main()
