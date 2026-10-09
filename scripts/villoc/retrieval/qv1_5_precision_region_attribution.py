#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

try:
    from scripts.villoc.retrieval.qv1_3_fixed_budget_multiview_pool import (
        load_reference_xy,
        tile_centers,
        evaluate_pool,
    )
except ModuleNotFoundError:
    from qv1_3_fixed_budget_multiview_pool import (
        load_reference_xy,
        tile_centers,
        evaluate_pool,
    )

ROOT = Path.cwd().resolve()
DEFAULT_CONFIG = Path(
    "configs/research/query_view_candidate_generation_v1.yaml"
)
DEFAULT_OUTPUT_ROOT = Path(
    "outputs/research_runs/query_view_candidate_generation_v1/qv1_5"
)


def resolve(path_like):
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


def evaluate_tiles_for_query(
    *,
    qid: int,
    pool: pd.DataFrame,
    map_index: pd.DataFrame,
    reference: pd.DataFrame,
) -> pd.DataFrame:
    lookup = map_index.set_index("tile_id")
    ref = reference.loc[int(qid)]
    x = float(ref["gt_x"])
    y = float(ref["gt_y"])

    rows = []
    for row in pool[pool["query_id"] == int(qid)].sort_values("pool_rank").itertuples(index=False):
        tile = lookup.loc[str(row.tile_id)]
        err = float(
            math.hypot(
                float(tile["center_easting"]) - x,
                float(tile["center_northing"]) - y,
            )
        )
        contains = bool(
            float(tile["left_easting"]) <= x <= float(tile["right_easting"])
            and float(tile["bottom_northing"]) <= y <= float(tile["top_northing"])
        )
        rows.append(
            {
                "pool_rank": int(row.pool_rank),
                "tile_id": str(row.tile_id),
                "source_view": str(row.source_view),
                "source_rank": int(row.source_rank),
                "center_error_m": err,
                "contains": contains,
                "le40": bool(err <= 40.0),
                "le80": bool(err <= 80.0),
            }
        )
    return pd.DataFrame(rows)


def classify_loss(policy_eval: pd.Series) -> str:
    if bool(policy_eval["contain_r20"]) and bool(policy_eval["le80_r20"]):
        return "precision_only_region_safe"
    if (not bool(policy_eval["contain_r20"])) and bool(policy_eval["le80_r20"]):
        return "precision_only_le80_safe"
    return "true_region_regression"


def nearest_selected_to_tile(
    *,
    target_tile_id: str,
    selected: pd.DataFrame,
    centers: dict[str, tuple[float, float]],
) -> dict:
    tx, ty = centers[target_tile_id]
    best = None

    for row in selected.itertuples(index=False):
        sx, sy = centers[str(row.tile_id)]
        d = float(math.hypot(tx - sx, ty - sy))
        item = {
            "tile_id": str(row.tile_id),
            "pool_rank": int(row.pool_rank),
            "source_view": str(row.source_view),
            "source_rank": int(row.source_rank),
            "distance_m": d,
        }
        if best is None or (d, item["pool_rank"]) < (best["distance_m"], best["pool_rank"]):
            best = item

    return best


def rescue_loss_table(candidate: pd.DataFrame, baseline: pd.DataFrame) -> dict:
    merged = candidate.merge(
        baseline,
        on="query_id",
        suffixes=("_candidate", "_baseline"),
        validate="one_to_one",
    )
    out = {}
    for metric in ("contain_r20", "le40_r20", "le80_r20"):
        c = merged[f"{metric}_candidate"].astype(bool)
        b = merged[f"{metric}_baseline"].astype(bool)
        out[metric] = {
            "rescues": int((c & ~b).sum()),
            "losses": int((~c & b).sum()),
            "net": int(c.sum() - b.sum()),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()

    cfg = yaml.safe_load(resolve(args.config).read_text())
    qv = cfg["qv1_5"]
    control = cfg["control"]

    output_root = resolve(args.output_root)
    qv13_root = resolve(qv["qv1_3_root"])
    qv14_root = resolve(qv["qv1_4_root"])

    map_index = pd.read_csv(resolve(control["map_index"])).copy()
    map_index["tile_id"] = map_index["tile_id"].astype(str)
    centers = tile_centers(map_index)
    reference = load_reference_xy(resolve(control["reference_attachment"]))

    paths = {
        "center_top20": qv14_root / "blind_pools/qv1_4_center_top20.csv",
        "center10_allview_diverse20": qv13_root / "blind_pools/qv1_3_center10_allview_diverse20.csv",
        "center_unique_allview_fill20": qv14_root / "blind_pools/qv1_4_center_unique_allview_fill20.csv",
    }

    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)

    pools = {}
    evals = {}

    for name, path in paths.items():
        pool = pd.read_csv(path).copy()
        pool["query_id"] = pd.to_numeric(pool["query_id"], errors="raise").astype(int)
        pool["pool_rank"] = pd.to_numeric(pool["pool_rank"], errors="raise").astype(int)
        pool["source_rank"] = pd.to_numeric(pool["source_rank"], errors="raise").astype(int)
        pool["tile_id"] = pool["tile_id"].astype(str)
        pools[name] = pool

        query_eval, _ = evaluate_pool(
            pool,
            map_index,
            reference,
        )
        evals[name] = query_eval

    baseline = evals["center_top20"]
    report_policies = {}
    loss_rows = []

    for policy in qv["policies"]:
        policy = str(policy)
        candidate = evals[policy]
        summary = rescue_loss_table(candidate, baseline)

        merged = candidate.merge(
            baseline,
            on="query_id",
            suffixes=("_policy", "_center"),
            validate="one_to_one",
        )

        lost = merged[
            (~merged["le40_r20_policy"].astype(bool))
            & merged["le40_r20_center"].astype(bool)
        ].copy()

        category_counts = {
            "precision_only_region_safe": 0,
            "precision_only_le80_safe": 0,
            "true_region_regression": 0,
        }

        for row in lost.itertuples(index=False):
            qid = int(row.query_id)
            policy_eval = pd.Series(
                {
                    "contain_r20": row.contain_r20_policy,
                    "le80_r20": row.le80_r20_policy,
                }
            )
            category = classify_loss(policy_eval)
            category_counts[category] += 1

            center_tiles = evaluate_tiles_for_query(
                qid=qid,
                pool=pools["center_top20"],
                map_index=map_index,
                reference=reference,
            )
            precise = center_tiles[center_tiles["le40"]].sort_values(
                ["center_error_m", "pool_rank"]
            ).iloc[0]

            policy_tiles = evaluate_tiles_for_query(
                qid=qid,
                pool=pools[policy],
                map_index=map_index,
                reference=reference,
            )
            selected_q = pools[policy][pools[policy]["query_id"] == qid]
            nearest = nearest_selected_to_tile(
                target_tile_id=str(precise["tile_id"]),
                selected=selected_q,
                centers=centers,
            )
            nearest_eval = policy_tiles[
                policy_tiles["tile_id"] == nearest["tile_id"]
            ].iloc[0]

            loss_rows.append(
                {
                    "policy": policy,
                    "query_id": qid,
                    "category": category,
                    "lost_center_precise_tile_id": str(precise["tile_id"]),
                    "lost_center_precise_rank": int(precise["pool_rank"]),
                    "lost_center_precise_error_m": float(precise["center_error_m"]),
                    "nearest_selected_tile_id": str(nearest["tile_id"]),
                    "nearest_selected_distance_to_precise_tile_m": float(nearest["distance_m"]),
                    "nearest_selected_pool_rank": int(nearest["pool_rank"]),
                    "nearest_selected_source_view": str(nearest["source_view"]),
                    "nearest_selected_source_rank": int(nearest["source_rank"]),
                    "nearest_selected_error_m": float(nearest_eval["center_error_m"]),
                    "nearest_selected_contains": bool(nearest_eval["contains"]),
                    "nearest_selected_le80": bool(nearest_eval["le80"]),
                    "policy_contains_r20": bool(row.contain_r20_policy),
                    "policy_le80_r20": bool(row.le80_r20_policy),
                }
            )

        report_policies[policy] = {
            "rescue_loss_vs_center": summary,
            "le40_loss_count": int(len(lost)),
            "le40_loss_categories": category_counts,
        }

    losses = pd.DataFrame(loss_rows)

    distance_summary = {}
    for policy in qv["policies"]:
        policy = str(policy)
        subset = losses[losses["policy"] == policy]
        if subset.empty:
            distance_summary[policy] = {}
            continue

        distance_summary[policy] = {
            "median_nearest_distance_m": float(
                subset["nearest_selected_distance_to_precise_tile_m"].median()
            ),
            "p95_nearest_distance_m": float(
                subset["nearest_selected_distance_to_precise_tile_m"].quantile(0.95)
            ),
            "within_51p2m_count": int(
                (subset["nearest_selected_distance_to_precise_tile_m"] <= 51.2).sum()
            ),
            "within_102p4m_count": int(
                (subset["nearest_selected_distance_to_precise_tile_m"] <= 102.4).sum()
            ),
        }

    named = {}
    for qid in qv["named_diagnostics"]:
        qid = int(qid)
        named[f"q{qid}"] = {}
        for policy in qv["policies"]:
            policy = str(policy)
            erow = evals[policy][evals[policy]["query_id"] == qid]
            lrow = losses[
                (losses["policy"] == policy)
                & (losses["query_id"] == qid)
            ]
            named[f"q{qid}"][policy] = {
                "evaluation": (
                    erow.iloc[0].to_dict()
                    if len(erow) == 1
                    else None
                ),
                "le40_loss_attribution": (
                    lrow.iloc[0].to_dict()
                    if len(lrow) == 1
                    else None
                ),
            }

    output_root.mkdir(parents=True, exist_ok=True)
    losses_path = output_root / "qv1_5_le40_loss_attribution.csv"
    losses.to_csv(losses_path, index=False)

    report = {
        "stage": "QV1.5",
        "status": "PASS_QV1_PRECISION_REGION_ATTRIBUTION",
        "created_at_utc": now_utc(),
        "scientific_question": (
            "Are <=40 losses in fixed-budget multi-view pools true region-level "
            "regressions or precision-only losses inside otherwise useful regions?"
        ),
        "policies": report_policies,
        "nearest_tile_distance_summary": distance_summary,
        "named_diagnostics": named,
        "input_hashes": {
            name: {
                "path": str(path),
                "sha256": sha256_file(path),
            }
            for name, path in paths.items()
        },
        "outputs": {
            "le40_loss_attribution_csv": str(losses_path),
        },
        "interpretation_contract": {
            "new_pool_built": False,
            "new_descriptor_inference_run": False,
            "postfreeze_diagnostic_only": True,
            "warning": (
                "Containment and <=80 indicate region-level candidate utility. "
                "<=40 is a stricter tile-center precision diagnostic and should "
                "not be conflated with whether geometric refinement can operate "
                "inside a containing tile."
            ),
        },
    }

    report_path = output_root / "qv1_5_precision_region_tradeoff_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str))

    print("=" * 118)
    print("QV1.5 — PRECISION / REGION TRADEOFF ATTRIBUTION")
    print("=" * 118)

    for policy in qv["policies"]:
        policy = str(policy)
        block = report_policies[policy]
        dist_block = distance_summary[policy]

        print()
        print(policy)
        print("  rescue/loss:", block["rescue_loss_vs_center"])
        print("  <=40 losses:", block["le40_loss_count"])
        print("  categories:", block["le40_loss_categories"])
        if dist_block:
            print("  nearest selected to lost precise tile:", dist_block)

    for qid in qv["named_diagnostics"]:
        print()
        print(f"q{int(qid)}:")
        print(json.dumps(named[f"q{int(qid)}"], indent=2, default=str))

    print()
    print("STATUS: PASS_QV1_PRECISION_REGION_ATTRIBUTION")
    print("report:", report_path)


if __name__ == "__main__":
    main()
