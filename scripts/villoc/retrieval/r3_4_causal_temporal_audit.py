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

from scripts.villoc.retrieval.r3_3_confidence_feature_audit import (
    binary_auc,
    composite_tile_id,
    load_fused_regions,
    prepare_composite_index,
)

ROOT = Path.cwd().resolve()
DEFAULT_CONFIG = Path("configs/research/absolute_frontend_decoupling_v1.yaml")
DEFAULT_OUTPUT_ROOT = Path("outputs/research_runs/absolute_frontend_decoupling_v1/r3_4")


def resolve(path_like):
    path = Path(path_like)
    return path if path.is_absolute() else ROOT / path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def center(index: pd.DataFrame, tile_id: str) -> tuple[float, float]:
    row = index.loc[tile_id]
    return float(row["center_easting"]), float(row["center_northing"])


def dist(a, b) -> float:
    return float(math.hypot(a[0] - b[0], a[1] - b[1]))


def temporal_support(
    current_tile_id: str,
    history_qids: list[int],
    regions_by_q: dict[int, list[dict[str, Any]]],
    index: pd.DataFrame,
    radius_m: float,
    rank_weight_k: float,
) -> dict[str, float]:
    c = center(index, current_tile_id)
    hit_count = 0
    weighted = 0.0
    nearest_distances = []
    hit_ranks = []

    for qid in history_qids:
        best = None
        for region in regions_by_q[int(qid)]:
            rcenter = center(index, str(region["representative_tile_id"]))
            d = dist(c, rcenter)
            rank = int(region["fused_rank"])
            candidate = (d, rank)
            if best is None or candidate < best:
                best = candidate

        if best is None:
            continue

        d, rank = best
        nearest_distances.append(float(d))
        if d <= float(radius_m):
            hit_count += 1
            hit_ranks.append(rank)
            weighted += 1.0 / (float(rank_weight_k) + float(rank))

    n = len(history_qids)
    return {
        "history_count": float(n),
        "hit_count": float(hit_count),
        "hit_fraction": float(hit_count / n) if n else float("nan"),
        "rank_weighted_support": float(weighted),
        "nearest_distance_mean_m": float(np.mean(nearest_distances)) if nearest_distances else float("nan"),
        "nearest_distance_median_m": float(np.median(nearest_distances)) if nearest_distances else float("nan"),
        "matched_rank_mean": float(np.mean(hit_ranks)) if hit_ranks else float("nan"),
        "matched_rank_best": float(np.min(hit_ranks)) if hit_ranks else float("nan"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()

    cfg = yaml.safe_load(resolve(args.config).read_text())
    r34 = cfg["r3_4"]
    artifacts = {k: resolve(v) for k, v in cfg["artifacts"].items()}

    for name in (
        "stable_768_topk",
        "triple_fused_topk",
        "triple_region_provenance",
        "tile_index_512",
        "tile_index_768",
        "tile_index_1024",
        "r3_3_report",
        "r3_3_labels",
    ):
        if not artifacts[name].exists():
            raise FileNotFoundError(f"{name}: {artifacts[name]}")

    r33 = json.loads(artifacts["r3_3_report"].read_text())
    if r33.get("status") != "PASS_R3_RETRIEVAL_CONFIDENCE_AUDIT":
        raise RuntimeError("R3.3 gate has not passed")

    topk = pd.read_csv(artifacts["stable_768_topk"]).copy()
    topk["query_id"] = pd.to_numeric(topk["query_id"], errors="raise").astype(int)
    topk["rank"] = pd.to_numeric(topk["rank"], errors="raise").astype(int)
    top1 = topk[topk["rank"] == 1][["query_id", "tile_id"]].copy()
    if len(top1) != 403:
        raise RuntimeError(f"Expected 403 768 Top1 rows, got {len(top1)}")

    regions = load_fused_regions(
        artifacts["triple_fused_topk"],
        artifacts["triple_region_provenance"],
    )
    index = prepare_composite_index([
        ("512_s256", artifacts["tile_index_512"]),
        ("768_s256", artifacts["tile_index_768"]),
        ("1024_s256", artifacts["tile_index_1024"]),
    ])

    qids = sorted(top1["query_id"].astype(int).tolist())
    top1_lookup = top1.set_index("query_id")["tile_id"].astype(str).to_dict()
    pos = {qid: i for i, qid in enumerate(qids)}

    rows = []
    for qid in qids:
        current_regions = regions[int(qid)]
        if len(current_regions) != 20:
            raise RuntimeError(f"q{qid}: expected 20 fused regions, got {len(current_regions)}")

        anchor_id = composite_tile_id("768_s256", str(top1_lookup[qid]))
        fused_id = str(current_regions[0]["representative_tile_id"])
        row = {
            "query_id": int(qid),
            "anchor_tile_id": anchor_id,
            "fused_top1_tile_id": fused_id,
        }

        for window in r34["history_windows"]:
            history = qids[max(0, pos[qid] - int(window)):pos[qid]]
            for radius in r34["radii_m"]:
                suffix = f"w{int(window)}_r{int(round(float(radius)))}"
                a = temporal_support(
                    anchor_id, history, regions, index,
                    float(radius), float(r34["rank_weight_k"])
                )
                f = temporal_support(
                    fused_id, history, regions, index,
                    float(radius), float(r34["rank_weight_k"])
                )
                for k, v in a.items():
                    row[f"anchor_{k}_{suffix}"] = v
                for k, v in f.items():
                    row[f"fused_{k}_{suffix}"] = v
                for metric in (
                    "hit_count",
                    "hit_fraction",
                    "rank_weighted_support",
                    "nearest_distance_mean_m",
                    "nearest_distance_median_m",
                ):
                    row[f"delta_fused_minus_anchor_{metric}_{suffix}"] = float(f[metric]) - float(a[metric])

        if pos[qid] > 0:
            prev_qid = qids[pos[qid] - 1]
            prev_anchor = composite_tile_id("768_s256", str(top1_lookup[prev_qid]))
            prev_fused = str(regions[prev_qid][0]["representative_tile_id"])
            row["anchor_displacement_from_prev_anchor_m"] = dist(center(index, anchor_id), center(index, prev_anchor))
            row["fused_displacement_from_prev_fused_m"] = dist(center(index, fused_id), center(index, prev_fused))
        else:
            row["anchor_displacement_from_prev_anchor_m"] = float("nan")
            row["fused_displacement_from_prev_fused_m"] = float("nan")

        rows.append(row)

    blind = pd.DataFrame(rows)
    output_root = resolve(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    blind_path = output_root / "r3_4_blind_causal_temporal_features.csv"
    blind.to_csv(blind_path, index=False)
    blind_sha = sha256_file(blind_path)

    labels = pd.read_csv(artifacts["r3_3_labels"]).copy()
    labels["query_id"] = pd.to_numeric(labels["query_id"], errors="raise").astype(int)
    labelled = blind.merge(
        labels[[
            "query_id",
            "containment_category",
            "le80_category",
            "anchor_center_error_m",
            "fused_top1_center_error_m",
        ]],
        on="query_id",
        how="left",
        validate="one_to_one",
    )

    if sha256_file(blind_path) != blind_sha:
        raise RuntimeError("Blind temporal feature table changed after label attachment")

    feature_cols = [
        c for c in blind.columns
        if c not in {"query_id", "anchor_tile_id", "fused_top1_tile_id"}
    ]

    audit = {}
    for label_name, category_col in (
        ("containment", "containment_category"),
        ("le80", "le80_category"),
    ):
        discordant = labelled[
            labelled[category_col].isin(["anchor_only_good", "fused_only_good"])
        ].copy()
        positive = discordant[category_col] == "fused_only_good"
        ranked = []
        for feature in feature_cols:
            result = binary_auc(positive, discordant[feature])
            if result is not None:
                ranked.append({"feature": feature, **result})
        ranked.sort(key=lambda x: (-x["diagnostic_auc_best_direction"], x["feature"]))
        audit[label_name] = {
            "discordant_query_count": int(len(discordant)),
            "fused_only_good_count": int(positive.sum()),
            "anchor_only_good_count": int((~positive).sum()),
            "feature_auc_on_discordant_queries": ranked,
        }

    named = {}
    for qid in r34["named_diagnostics"]:
        row = labelled[labelled["query_id"] == int(qid)]
        named[f"q{int(qid)}"] = row.iloc[0].to_dict() if len(row) == 1 else None

    labelled_path = output_root / "r3_4_postfreeze_temporal_feature_labels.csv"
    labelled.to_csv(labelled_path, index=False)

    report = {
        "stage": "R3.4",
        "status": "PASS_R3_CAUSAL_TEMPORAL_RETRIEVAL_AUDIT",
        "created_at_utc": now_utc(),
        "blind_feature_freeze": {
            "path": str(blind_path),
            "sha256": blind_sha,
            "future_queries_used": False,
            "reference_used": False,
            "orb_used": False,
            "state_used": False,
        },
        "audit": audit,
        "r3_3_best_static_diagnostic": {
            label: (
                r33["audit"][label]["feature_auc_on_discordant_queries"][0]
                if r33["audit"][label]["feature_auc_on_discordant_queries"]
                else None
            )
            for label in ("containment", "le80")
        },
        "named_diagnostics": named,
        "scope_guarantees": {
            "selector_changed": False,
            "thresholds_fitted": False,
            "future_query_information_used": False,
            "orb_reranking_run": False,
            "bootstrap_run": False,
            "state_policy_changed": False,
            "reference_used_before_feature_freeze": False,
        },
    }

    report_path = output_root / "r3_4_causal_temporal_retrieval_audit_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str))

    print("=" * 116)
    print("R3.4 — CAUSAL TEMPORAL RETRIEVAL AUDIT")
    print("=" * 116)
    for label in ("containment", "le80"):
        block = audit[label]
        static_best = report["r3_3_best_static_diagnostic"][label]
        print()
        print(label.upper())
        print(
            "discordant:", block["discordant_query_count"],
            "fused-only:", block["fused_only_good_count"],
            "anchor-only:", block["anchor_only_good_count"],
        )
        print("R3.3 best static:", static_best)
        print("top temporal features:")
        for item in block["feature_auc_on_discordant_queries"][:12]:
            print(
                f"  {item['feature']:58s} "
                f"AUC*={item['diagnostic_auc_best_direction']:.3f} "
                f"{item['direction']}"
            )

    for qid in r34["named_diagnostics"]:
        print()
        print(f"q{int(qid)}:")
        print(json.dumps(named[f"q{int(qid)}"], indent=2, default=str))

    print()
    print("STATUS: PASS_R3_CAUSAL_TEMPORAL_RETRIEVAL_AUDIT")
    print("report:", report_path)


if __name__ == "__main__":
    main()
