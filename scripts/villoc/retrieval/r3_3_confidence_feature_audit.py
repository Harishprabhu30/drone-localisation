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
    "outputs/research_runs/absolute_frontend_decoupling_v1/r3_3"
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
    decoded = json.loads(str(value))
    if not isinstance(decoded, list):
        raise RuntimeError("members_json must decode to a list")
    return [dict(item) for item in decoded]


def member_rank(
    members: list[dict[str, Any]],
    variant: str,
) -> float:
    rows = [
        int(member["per_scale_rank"])
        for member in members
        if str(member["variant"]) == variant
    ]
    if len(rows) > 1:
        raise RuntimeError(
            f"Region has more than one member for {variant}"
        )
    return float(rows[0]) if rows else float("nan")


def rank_stats(
    members: list[dict[str, Any]],
) -> dict[str, float]:
    ranks = np.asarray(
        [
            float(member["per_scale_rank"])
            for member in members
        ],
        dtype=float,
    )
    if len(ranks) == 0:
        return {
            "mean": float("nan"),
            "max": float("nan"),
            "std": float("nan"),
        }
    return {
        "mean": float(np.mean(ranks)),
        "max": float(np.max(ranks)),
        "std": float(np.std(ranks)),
    }


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
                f"{variant}: tile index missing {sorted(missing)}"
            )

        frame["source_tile_id"] = frame["tile_id"].astype(str)
        frame["source_variant"] = variant
        frame["tile_id"] = [
            composite_tile_id(variant, value)
            for value in frame["source_tile_id"]
        ]
        frames.append(frame)

    merged = pd.concat(frames, ignore_index=True)
    if merged["tile_id"].duplicated().any():
        raise RuntimeError("Composite tile IDs are not unique")

    return merged.set_index("tile_id")


def load_fused_regions(
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
        raise RuntimeError(
            "Missing fused-region provenance rows"
        )

    result: dict[int, list[dict[str, Any]]] = {}

    for qid, group in merged.groupby("query_id", sort=True):
        rows = []
        for row in group.sort_values("rank").itertuples(index=False):
            rows.append(
                {
                    "fused_rank": int(row.rank),
                    "representative_tile_id": str(row.tile_id),
                    "fused_rrf_score": float(row.fused_rrf_score),
                    "support_scale_count": int(
                        row.support_scale_count
                    ),
                    "support_scales": str(row.support_scales),
                    "members": parse_members(row.members_json),
                }
            )
        result[int(qid)] = rows

    return result


def find_anchor_region(
    regions: list[dict[str, Any]],
    tile_id_768: str,
) -> dict[str, Any] | None:
    found = []

    for region in regions:
        for member in region["members"]:
            if (
                str(member["variant"]) == "768_s256"
                and str(member["tile_id"]) == str(tile_id_768)
            ):
                found.append(region)
                break

    if len(found) > 1:
        raise RuntimeError(
            f"768 anchor {tile_id_768} occurs in multiple fused regions"
        )

    return found[0] if found else None


def center_distance(
    index: pd.DataFrame,
    a: str,
    b: str,
) -> float:
    if a not in index.index or b not in index.index:
        return float("nan")

    ra = index.loc[a]
    rb = index.loc[b]

    return float(
        math.hypot(
            float(ra["center_easting"])
            - float(rb["center_easting"]),
            float(ra["center_northing"])
            - float(rb["center_northing"]),
        )
    )


def evaluate_tile(
    index: pd.DataFrame,
    reference: pd.DataFrame,
    qid: int,
    tile_id: str,
) -> dict[str, Any]:
    if tile_id not in index.index:
        raise RuntimeError(
            f"Tile index missing {tile_id}"
        )
    if qid not in reference.index:
        raise RuntimeError(
            f"Reference missing q{qid}"
        )

    tile = index.loc[tile_id]
    ref = reference.loc[qid]

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

    return {
        "center_error_m": error,
        "contains": contains,
        "le80": bool(error <= 80.0),
    }


def evaluate_region_members(
    *,
    region: dict[str, Any],
    index: pd.DataFrame,
    reference: pd.DataFrame,
    qid: int,
) -> dict[str, Any]:
    results = []

    for member in region["members"]:
        cid = composite_tile_id(
            str(member["variant"]),
            str(member["tile_id"]),
        )
        result = evaluate_tile(
            index,
            reference,
            qid,
            cid,
        )
        results.append(
            {
                "tile_id": cid,
                **result,
            }
        )

    best = min(
        results,
        key=lambda item: item["center_error_m"],
    )

    return {
        "any_member_contains": bool(
            any(item["contains"] for item in results)
        ),
        "any_member_le80": bool(
            any(item["le80"] for item in results)
        ),
        "best_member_center_error_m": float(
            best["center_error_m"]
        ),
        "best_member_tile_id": str(
            best["tile_id"]
        ),
    }


def category(
    anchor_good: bool,
    fused_good: bool,
) -> str:
    if anchor_good and fused_good:
        return "both_good"
    if anchor_good and not fused_good:
        return "anchor_only_good"
    if not anchor_good and fused_good:
        return "fused_only_good"
    return "both_bad"


def binary_auc(
    positive: pd.Series,
    score: pd.Series,
) -> dict[str, Any] | None:
    frame = pd.DataFrame(
        {
            "positive": positive.astype(bool),
            "score": pd.to_numeric(
                score,
                errors="coerce",
            ),
        }
    ).dropna()

    n_pos = int(frame["positive"].sum())
    n_neg = int((~frame["positive"]).sum())

    if n_pos == 0 or n_neg == 0:
        return None

    ranks = frame["score"].rank(
        method="average",
        ascending=True,
    )
    rank_sum_pos = float(
        ranks[frame["positive"]].sum()
    )
    auc = (
        rank_sum_pos
        - n_pos * (n_pos + 1) / 2.0
    ) / float(n_pos * n_neg)

    if auc >= 0.5:
        direction = "higher_favors_fused"
        diagnostic_auc = auc
    else:
        direction = "lower_favors_fused"
        diagnostic_auc = 1.0 - auc

    return {
        "positive_count": n_pos,
        "negative_count": n_neg,
        "raw_auc_higher_favors_fused": float(auc),
        "diagnostic_auc_best_direction": float(
            diagnostic_auc
        ),
        "direction": direction,
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
    r33 = cfg["r3_3"]

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
        "r3_2_report",
    )
    for name in required:
        if not artifacts[name].exists():
            raise FileNotFoundError(
                f"{name}: {artifacts[name]}"
            )

    r32 = json.loads(
        artifacts["r3_2_report"].read_text()
    )
    if (
        r32.get("status")
        != "PASS_R3_RETRIEVAL_NATIVE_CONSENSUS_SELECTOR"
    ):
        raise RuntimeError(
            "R3.2 gate has not passed"
        )

    # ==============================================================
    # PHASE 1 — BLIND FEATURE FREEZE.
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
    topk_768["score"] = pd.to_numeric(
        topk_768["score"], errors="raise"
    )

    regions_by_q = load_fused_regions(
        artifacts["triple_fused_topk"],
        artifacts["triple_region_provenance"],
    )

    composite_index = prepare_composite_index(
        [
            ("512_s256", artifacts["tile_index_512"]),
            ("768_s256", artifacts["tile_index_768"]),
            ("1024_s256", artifacts["tile_index_1024"]),
        ]
    )

    blind_rows = []

    for qid, group in topk_768.groupby("query_id", sort=True):
        ordered = group.sort_values("rank")
        if len(ordered) < 2:
            raise RuntimeError(
                f"q{qid}: need at least Top-2 768 retrieval rows"
            )

        top1_768 = ordered.iloc[0]
        top2_768 = ordered.iloc[1]
        if int(top1_768["rank"]) != 1:
            raise RuntimeError(
                f"q{qid}: 768 ranking does not start at 1"
            )

        regions = regions_by_q.get(int(qid), [])
        if len(regions) != 20:
            raise RuntimeError(
                f"q{qid}: expected 20 fused regions, got {len(regions)}"
            )

        fused_top1 = regions[0]
        fused_top2 = regions[1]
        anchor = find_anchor_region(
            regions,
            str(top1_768["tile_id"]),
        )

        fused_stats = rank_stats(
            fused_top1["members"]
        )

        anchor_cid = composite_tile_id(
            "768_s256",
            str(top1_768["tile_id"]),
        )
        fused_cid = str(
            fused_top1["representative_tile_id"]
        )

        blind_rows.append(
            {
                "query_id": int(qid),
                "anchor_tile_id": anchor_cid,
                "fused_top1_tile_id": fused_cid,
                "dino768_top1_score": float(
                    top1_768["score"]
                ),
                "dino768_top2_score": float(
                    top2_768["score"]
                ),
                "dino768_top1_margin": float(
                    top1_768["score"]
                    - top2_768["score"]
                ),
                "anchor_found": bool(
                    anchor is not None
                ),
                "anchor_fused_rank": (
                    float(anchor["fused_rank"])
                    if anchor is not None
                    else float("nan")
                ),
                "anchor_support_scale_count": (
                    float(
                        anchor["support_scale_count"]
                    )
                    if anchor is not None
                    else float("nan")
                ),
                "anchor_fused_rrf_score": (
                    float(anchor["fused_rrf_score"])
                    if anchor is not None
                    else float("nan")
                ),
                "fused_top1_support_scale_count": int(
                    fused_top1["support_scale_count"]
                ),
                "fused_top1_rrf_score": float(
                    fused_top1["fused_rrf_score"]
                ),
                "fused_top2_rrf_score": float(
                    fused_top2["fused_rrf_score"]
                ),
                "fused_top1_rrf_margin": float(
                    fused_top1["fused_rrf_score"]
                    - fused_top2["fused_rrf_score"]
                ),
                "fused_top1_member_rank_mean": fused_stats[
                    "mean"
                ],
                "fused_top1_member_rank_max": fused_stats[
                    "max"
                ],
                "fused_top1_member_rank_std": fused_stats[
                    "std"
                ],
                "fused_top1_512_member_rank": member_rank(
                    fused_top1["members"],
                    "512_s256",
                ),
                "fused_top1_768_member_rank": member_rank(
                    fused_top1["members"],
                    "768_s256",
                ),
                "fused_top1_1024_member_rank": member_rank(
                    fused_top1["members"],
                    "1024_s256",
                ),
                "anchor_to_fused_top1_center_distance_m": (
                    center_distance(
                        composite_index,
                        anchor_cid,
                        fused_cid,
                    )
                ),
                "anchor_and_fused_top1_same_region": bool(
                    anchor is not None
                    and int(anchor["fused_rank"]) == 1
                ),
                "three_scale_region_count_top5": int(
                    sum(
                        int(
                            region["support_scale_count"] == 3
                        )
                        for region in regions[:5]
                    )
                ),
                "three_scale_region_count_top20": int(
                    sum(
                        int(
                            region["support_scale_count"] == 3
                        )
                        for region in regions
                    )
                ),
                "multiscale_region_count_top20": int(
                    sum(
                        int(
                            region["support_scale_count"] > 1
                        )
                        for region in regions
                    )
                ),
            }
        )

    blind = pd.DataFrame(blind_rows)
    if len(blind) != 403:
        raise RuntimeError(
            f"Expected 403 feature rows, got {len(blind)}"
        )

    output_root.mkdir(parents=True, exist_ok=True)
    blind_path = (
        output_root
        / "r3_3_blind_retrieval_confidence_features.csv"
    )
    blind.to_csv(blind_path, index=False)
    blind_sha = sha256_file(blind_path)

    # ==============================================================
    # PHASE 2 — POST-FREEZE LABEL ATTACHMENT / DIAGNOSTIC ONLY.
    # ==============================================================
    reference = load_reference_xy(
        artifacts["reference_attachment"]
    )

    labelled_rows = []
    for row in blind.itertuples(index=False):
        qid = int(row.query_id)
        anchor_eval = evaluate_tile(
            composite_index,
            reference,
            qid,
            str(row.anchor_tile_id),
        )
        fused_eval = evaluate_tile(
            composite_index,
            reference,
            qid,
            str(row.fused_top1_tile_id),
        )

        fused_region = regions_by_q[qid][0]
        region_eval = evaluate_region_members(
            region=fused_region,
            index=composite_index,
            reference=reference,
            qid=qid,
        )

        base = row._asdict()
        base.update(
            {
                "anchor_center_error_m": anchor_eval["center_error_m"],
                "anchor_contains": anchor_eval["contains"],
                "anchor_le80": anchor_eval["le80"],
                "fused_top1_center_error_m": fused_eval["center_error_m"],
                "fused_top1_contains": fused_eval["contains"],
                "fused_top1_le80": fused_eval["le80"],
                "fused_top1_region_any_member_contains": region_eval[
                    "any_member_contains"
                ],
                "fused_top1_region_any_member_le80": region_eval[
                    "any_member_le80"
                ],
                "fused_top1_region_best_member_center_error_m": region_eval[
                    "best_member_center_error_m"
                ],
                "fused_top1_region_best_member_tile_id": region_eval[
                    "best_member_tile_id"
                ],
                "fused_minus_anchor_error_m": float(
                    fused_eval["center_error_m"]
                    - anchor_eval["center_error_m"]
                ),
                "containment_category": category(
                    anchor_eval["contains"],
                    fused_eval["contains"],
                ),
                "le80_category": category(
                    anchor_eval["le80"],
                    fused_eval["le80"],
                ),
            }
        )
        labelled_rows.append(base)

    labelled = pd.DataFrame(labelled_rows)

    if sha256_file(blind_path) != blind_sha:
        raise RuntimeError(
            "Blind confidence feature table changed during evaluation"
        )

    feature_names = list(r33["numeric_features"])
    audit = {}

    for label_name, category_col in (
        ("containment", "containment_category"),
        ("le80", "le80_category"),
    ):
        discordant = labelled[
            labelled[category_col].isin(
                ["anchor_only_good", "fused_only_good"]
            )
        ].copy()
        positive = (
            discordant[category_col]
            == "fused_only_good"
        )

        ranked = []
        for feature in feature_names:
            if feature not in discordant.columns:
                raise RuntimeError(
                    f"Configured feature missing: {feature}"
                )
            result = binary_auc(
                positive,
                discordant[feature],
            )
            if result is not None:
                ranked.append(
                    {
                        "feature": feature,
                        **result,
                    }
                )

        ranked.sort(
            key=lambda item: (
                -item["diagnostic_auc_best_direction"],
                item["feature"],
            )
        )

        counts = (
            labelled[category_col]
            .value_counts()
            .to_dict()
        )

        audit[label_name] = {
            "category_counts": {
                str(k): int(v)
                for k, v in counts.items()
            },
            "discordant_query_count": int(
                len(discordant)
            ),
            "fused_only_good_count": int(
                positive.sum()
            ),
            "anchor_only_good_count": int(
                (~positive).sum()
            ),
            "feature_auc_on_discordant_queries": ranked,
        }

    group_medians = {}
    for category_col in (
        "containment_category",
        "le80_category",
    ):
        group_medians[category_col] = {}
        for group_name, group in labelled.groupby(
            category_col,
            sort=True,
        ):
            group_medians[category_col][str(group_name)] = {
                feature: (
                    float(
                        pd.to_numeric(
                            group[feature],
                            errors="coerce",
                        ).median()
                    )
                    if pd.to_numeric(
                        group[feature],
                        errors="coerce",
                    ).notna().any()
                    else None
                )
                for feature in feature_names
            }

    named = {}
    for qid in r33["named_diagnostics"]:
        qid = int(qid)
        row = labelled[
            labelled["query_id"] == qid
        ]
        named[f"q{qid}"] = (
            row.iloc[0].to_dict()
            if len(row) == 1
            else None
        )

    labelled_path = (
        output_root
        / "r3_3_postfreeze_confidence_feature_labels.csv"
    )
    labelled.to_csv(labelled_path, index=False)

    report = {
        "stage": "R3.3",
        "status": "PASS_R3_RETRIEVAL_CONFIDENCE_AUDIT",
        "created_at_utc": now_utc(),
        "scientific_question": (
            "Which static blind retrieval signals, if any, distinguish "
            "queries where fused rank-1 should rescue 768 from queries where "
            "768 should be held?"
        ),
        "blind_feature_freeze": {
            "path": str(blind_path),
            "sha256": blind_sha,
            "reference_used": False,
            "orb_used": False,
            "state_used": False,
        },
        "audit": audit,
        "category_feature_medians": group_medians,
        "named_diagnostics": named,
        "interpretation_contract": {
            "diagnostic_auc_only": True,
            "thresholds_fitted": False,
            "selector_changed": False,
            "generalization_claimed": False,
            "warning": (
                "AUC is descriptive on this trajectory only. Any feature "
                "promoted into a selector requires a separately frozen policy "
                "and independent validation."
            ),
        },
        "input_hashes": {
            name: {
                "path": str(path),
                "sha256": sha256_file(path),
            }
            for name, path in artifacts.items()
            if path.exists()
        },
        "outputs": {
            "labelled_feature_csv": str(labelled_path),
        },
        "scope_guarantees": {
            "new_selector_run": False,
            "gt_tuned_threshold_used": False,
            "orb_reranking_run": False,
            "orb_projection_run": False,
            "bootstrap_run": False,
            "state_policy_changed": False,
            "reference_used_before_feature_freeze": False,
        },
        "next_decision": (
            "If no static feature has credible separation on discordant cases, "
            "move to causal temporal retrieval consistency. If one signal is "
            "strong and mechanistically defensible, freeze a simple selector "
            "before testing it; do not tune a threshold on this trajectory."
        ),
    }

    report_path = (
        output_root
        / "r3_3_retrieval_confidence_audit_report.json"
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
    print("R3.3 — RETRIEVAL CONFIDENCE FEATURE AUDIT")
    print("=" * 112)

    for label_name in ("containment", "le80"):
        block = audit[label_name]
        print()
        print(label_name.upper())
        print("categories:", block["category_counts"])
        print(
            "discordant:",
            block["discordant_query_count"],
            "fused-only:",
            block["fused_only_good_count"],
            "anchor-only:",
            block["anchor_only_good_count"],
        )
        print("top diagnostic features:")
        for item in block[
            "feature_auc_on_discordant_queries"
        ][:10]:
            print(
                f"  {item['feature']:42s} "
                f"AUC*={item['diagnostic_auc_best_direction']:.3f} "
                f"{item['direction']}"
            )

    for qid in r33["named_diagnostics"]:
        print()
        print(f"q{int(qid)}:")
        row = named[f"q{int(qid)}"]
        if row is None:
            print("  missing")
            continue
        fields = [
            "anchor_tile_id",
            "fused_top1_tile_id",
            "dino768_top1_margin",
            "anchor_fused_rank",
            "anchor_support_scale_count",
            "anchor_fused_rrf_score",
            "fused_top1_support_scale_count",
            "fused_top1_rrf_score",
            "fused_top1_rrf_margin",
            "fused_top1_768_member_rank",
            "anchor_to_fused_top1_center_distance_m",
            "three_scale_region_count_top5",
            "containment_category",
            "le80_category",
            "anchor_center_error_m",
            "fused_top1_center_error_m",
            "fused_top1_region_any_member_contains",
            "fused_top1_region_best_member_center_error_m",
        ]
        print(
            json.dumps(
                {
                    field: row.get(field)
                    for field in fields
                },
                indent=2,
                default=str,
            )
        )

    print()
    print("STATUS: PASS_R3_RETRIEVAL_CONFIDENCE_AUDIT")
    print("report:", report_path)


if __name__ == "__main__":
    main()
