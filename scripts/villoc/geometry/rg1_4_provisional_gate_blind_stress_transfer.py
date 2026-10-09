#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from uavloc.data.trajectory_adapter import load_trajectory_spec  # noqa: E402
from scripts.villoc.research.minimum_confident_bootstrap.diagnostics import (  # noqa: E402
    r4_11_blind_subtile_projection_recompute as r411,
)


ROOT = REPO_ROOT
DEFAULT_CONFIG = Path("configs/research/region_candidate_geometry_v1.yaml")


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def numeric_summary(series: pd.Series) -> dict[str, Any]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if not len(values):
        return {"count": 0, "mean": None, "median": None, "p95": None, "max": None}
    return {
        "count": int(len(values)),
        "mean": float(values.mean()),
        "median": float(values.median()),
        "p95": float(values.quantile(0.95)),
        "max": float(values.max()),
    }


def choose_threshold(
    table: pd.DataFrame,
    *,
    family: str,
    minimum_recall: dict[str, float],
) -> dict[str, Any]:
    rows = table[table["family"].astype(str) == family].copy()
    if rows.empty:
        raise RuntimeError(f"No RG1.3 rows for family={family!r}")

    mask = pd.Series(True, index=rows.index)
    checks = {}

    for label, minimum in minimum_recall.items():
        column = f"{label}_recall"
        if column not in rows.columns:
            raise RuntimeError(f"RG1.3 table missing {column}")
        values = pd.to_numeric(rows[column], errors="coerce")
        mask &= values >= float(minimum)
        checks[label] = {
            "column": column,
            "minimum": float(minimum),
        }

    eligible = rows[mask].copy()
    if eligible.empty:
        raise RuntimeError(
            "No inliers-only RG1.3 gate satisfies all predeclared recall constraints."
        )

    eligible["inlier_threshold"] = pd.to_numeric(
        eligible["inlier_threshold"], errors="raise"
    ).astype(int)

    selected = eligible.sort_values(
        ["inlier_threshold", "accepted_query_fraction"],
        ascending=[False, True],
        kind="mergesort",
    ).iloc[0]

    threshold = int(selected["inlier_threshold"])

    return {
        "family": family,
        "inlier_threshold": threshold,
        "selection_rule": "highest_inlier_threshold_satisfying_all_recall_constraints",
        "minimum_recall": minimum_recall,
        "development_metrics": {
            "accepted_query_fraction": float(selected["accepted_query_fraction"]),
            "contains_precision": float(selected["contains_query_precision"]),
            "contains_recall": float(selected["contains_query_recall"]),
            "le80_precision": float(selected["le80_precision"]),
            "le80_recall": float(selected["le80_recall"]),
            "le40_precision": float(selected["le40_precision"]),
            "le40_recall": float(selected["le40_recall"]),
            "accepted_projected_error_median": (
                float(selected["accepted_projected_error_median"])
                if pd.notna(selected["accepted_projected_error_median"])
                else None
            ),
            "projection_improves_rate": (
                float(selected["projection_improves_rate"])
                if pd.notna(selected["projection_improves_rate"])
                else None
            ),
        },
        "eligible_gate_count": int(len(eligible)),
        "recall_checks": checks,
    }


def resolve_image(value: Any, extra_roots: list[Path] | None = None) -> Path:
    raw = str(value).strip()
    path = Path(raw).expanduser()
    candidates = [path] if path.is_absolute() else [ROOT / path]
    if not path.is_absolute():
        for root in extra_roots or []:
            candidates.append(root / path)

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate.resolve()

    raise FileNotFoundError(
        "Could not resolve image. Tried: " + ", ".join(str(x) for x in candidates)
    )


def compute_anchor_geometry(
    *,
    manifest: pd.DataFrame,
    pool: pd.DataFrame,
    tile_index: pd.DataFrame,
    tile_index_path: Path,
) -> pd.DataFrame:
    anchor = pool[pd.to_numeric(pool["pool_rank"], errors="raise").astype(int) == 1].copy()

    if len(anchor) != len(manifest):
        raise RuntimeError("Expected exactly one frozen retrieval anchor per blind query.")

    if not (
        (anchor["source_view"].astype(str) == "center_square")
        & (pd.to_numeric(anchor["source_rank"], errors="raise").astype(int) == 1)
    ).all():
        raise RuntimeError("Blind-stress anchor is not frozen center-square Top1.")

    manifest_lookup = manifest.set_index("query_id")
    tile_index = tile_index.copy()
    tile_index["tile_id"] = tile_index["tile_id"].astype(str)
    tile_lookup = tile_index.set_index("tile_id")

    detector = r411.create_detector()
    query_cache: dict[int, Any] = {}
    tile_cache: dict[str, Any] = {}
    rows = []

    for idx, row in anchor.sort_values("query_id").reset_index(drop=True).iterrows():
        qid = int(row["query_id"])
        tile_id = str(row["tile_id"])

        if qid not in query_cache:
            qrow = manifest_lookup.loc[qid]
            qpath = resolve_image(qrow["image_path"])
            query_cache[qid] = (qpath, r411.compute_features(qpath, detector))
        qpath, qfeat = query_cache[qid]

        if tile_id not in tile_cache:
            trow = tile_lookup.loc[tile_id]
            tpath = resolve_image(
                trow["tile_path"],
                extra_roots=[tile_index_path.parent],
            )
            tile_cache[tile_id] = (tpath, r411.compute_features(tpath, detector))
        tpath, tfeat = tile_cache[tile_id]
        trow = tile_lookup.loc[tile_id]

        result = r411.verify_pair_with_geometry(qfeat, tfeat)

        easting = math.nan
        northing = math.nan
        inside = False

        if bool(result["homography_ok"]):
            u = float(result["projected_tile_u_px"])
            v = float(result["projected_tile_v_px"])
            easting, northing = r411.tile_pixel_to_map(
                u,
                v,
                tfeat.image_shape,
                float(trow["left_easting"]),
                float(trow["right_easting"]),
                float(trow["bottom_northing"]),
                float(trow["top_northing"]),
            )
            th, tw = tfeat.image_shape
            inside = bool(0.0 <= u < float(tw) and 0.0 <= v < float(th))

        rows.append(
            {
                "query_id": qid,
                "pool_rank": 1,
                "tile_id": tile_id,
                "source_view": str(row["source_view"]),
                "source_rank": int(row["source_rank"]),
                "good_matches": int(result["good_matches"]),
                "inliers": int(result["inliers"]),
                "inlier_ratio": float(result["inlier_ratio"]),
                "query_inlier_coverage": float(result["query_inlier_coverage"]),
                "tile_inlier_coverage": float(result["sat_inlier_coverage"]),
                "homography_ok": bool(result["homography_ok"]),
                "projected_tile_u_px": float(result["projected_tile_u_px"]),
                "projected_tile_v_px": float(result["projected_tile_v_px"]),
                "projected_easting": float(easting),
                "projected_northing": float(northing),
                "projected_inside_tile": inside,
                "inlier_reprojection_rmse_px": float(
                    result["inlier_reprojection_rmse_px"]
                ),
                "query_image_resolved": str(qpath),
                "tile_image_resolved": str(tpath),
            }
        )

    return pd.DataFrame(rows)


def accepted_temporal_stats(frame: pd.DataFrame) -> dict[str, Any]:
    accepted = frame[frame["accepted"].astype(bool)].sort_values("query_id").copy()

    if len(accepted) < 2:
        return {
            "accepted_query_count": int(len(accepted)),
            "consecutive_accepted_projection_jump_m": numeric_summary(
                pd.Series(dtype=float)
            ),
            "accepted_query_gap": numeric_summary(pd.Series(dtype=float)),
        }

    jumps = []
    gaps = []

    rows = list(accepted.itertuples(index=False))
    for previous, current in zip(rows[:-1], rows[1:]):
        if all(
            math.isfinite(float(v))
            for v in (
                previous.projected_easting,
                previous.projected_northing,
                current.projected_easting,
                current.projected_northing,
            )
        ):
            jumps.append(
                math.hypot(
                    float(current.projected_easting) - float(previous.projected_easting),
                    float(current.projected_northing) - float(previous.projected_northing),
                )
            )
        gaps.append(int(current.query_id) - int(previous.query_id))

    return {
        "accepted_query_count": int(len(accepted)),
        "consecutive_accepted_projection_jump_m": numeric_summary(pd.Series(jumps)),
        "accepted_query_gap": numeric_summary(pd.Series(gaps)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()

    cfg = yaml.safe_load(resolve(args.config).read_text(encoding="utf-8"))
    rg = cfg["rg1_4"]

    dev_table_path = resolve(rg["development"]["rg1_3_table"])
    dev_report_path = resolve(rg["development"]["rg1_3_report"])

    if not dev_table_path.exists() or not dev_report_path.exists():
        raise FileNotFoundError("RG1.3 artifacts are missing.")

    dev_report = json.loads(dev_report_path.read_text(encoding="utf-8"))
    if dev_report.get("status") != "PASS_RG1_3_RETRIEVAL_ANCHOR_TRUST_REFINEMENT_AUDIT":
        raise RuntimeError("RG1.3 has not passed.")

    dev_table = pd.read_csv(dev_table_path)

    freeze = choose_threshold(
        dev_table,
        family=str(rg["freeze_rule"]["family"]),
        minimum_recall={
            str(k): float(v)
            for k, v in rg["freeze_rule"]["minimum_recall"].items()
        },
    )

    threshold = int(freeze["inlier_threshold"])

    blind_cfg = rg["blind_stress"]
    spec_path = resolve(blind_cfg["trajectory_spec"])
    manifest_path = resolve(blind_cfg["canonical_manifest"])
    pool_path = resolve(blind_cfg["frozen_candidate_pool"])
    spec = load_trajectory_spec(spec_path)

    if spec.role != "blind_stress":
        raise RuntimeError("RG1.4 blind transfer requires role='blind_stress'.")
    if spec.reference.available:
        raise RuntimeError("RG1.4 blind transfer refuses a trajectory with reference.")

    manifest = pd.read_csv(manifest_path).copy()
    pool = pd.read_csv(pool_path).copy()

    manifest["query_id"] = pd.to_numeric(
        manifest["query_id"], errors="raise"
    ).astype(int)
    pool["query_id"] = pd.to_numeric(pool["query_id"], errors="raise").astype(int)

    if len(manifest) != int(blind_cfg["expected_queries"]):
        raise RuntimeError(
            f"Expected {blind_cfg['expected_queries']} blind queries, got {len(manifest)}."
        )

    variant_name = str(blind_cfg["map_variant"])
    variant = spec.raw["map"]["variants"][variant_name]
    tile_index_path = resolve(variant["tile_index"])
    tile_index = pd.read_csv(tile_index_path).copy()

    evidence = compute_anchor_geometry(
        manifest=manifest,
        pool=pool,
        tile_index=tile_index,
        tile_index_path=tile_index_path,
    )

    evidence["accepted"] = (
        pd.to_numeric(evidence["inliers"], errors="raise")
        >= threshold
    )

    output_root = resolve(rg["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)

    evidence_path = output_root / "rg1_4_blind_anchor_geometry_evidence.csv"
    freeze_path = output_root / "rg1_4_frozen_gate.json"
    report_path = output_root / "rg1_4_blind_stress_transfer_report.json"

    freeze_payload = {
        "stage": "RG1.4_FREEZE",
        "created_at_utc": now_utc(),
        "source": {
            "rg1_3_table": str(dev_table_path),
            "rg1_3_table_sha256": sha256_file(dev_table_path),
            "rg1_3_report": str(dev_report_path),
            "rg1_3_report_sha256": sha256_file(dev_report_path),
        },
        "frozen_gate": freeze,
        "blind_data_seen_during_threshold_selection": False,
    }
    freeze_path.write_text(
        json.dumps(freeze_payload, indent=2),
        encoding="utf-8",
    )

    evidence.to_csv(evidence_path, index=False)

    accepted = evidence["accepted"].astype(bool)
    acceptance_count = int(accepted.sum())
    acceptance_fraction = float(accepted.mean())

    accepted_rows = evidence[accepted].copy()
    rejected_rows = evidence[~accepted].copy()

    report = {
        "stage": "RG1.4",
        "status": "PASS_RG1_4_PROVISIONAL_GATE_BLIND_STRESS_TRANSFER",
        "created_at_utc": now_utc(),
        "frozen_gate": freeze,
        "blind_stress": {
            "trajectory_id": spec.trajectory_id,
            "query_count": int(len(evidence)),
            "reference_available": False,
            "accuracy_metrics_computed": False,
            "accepted_query_count": acceptance_count,
            "accepted_query_fraction": acceptance_fraction,
            "rejected_query_count": int((~accepted).sum()),
            "inliers_all": numeric_summary(evidence["inliers"]),
            "inliers_accepted": numeric_summary(accepted_rows["inliers"]),
            "inliers_rejected": numeric_summary(rejected_rows["inliers"]),
            "inlier_ratio_all": numeric_summary(evidence["inlier_ratio"]),
            "inlier_ratio_accepted": numeric_summary(accepted_rows["inlier_ratio"]),
            "projected_inside_tile_rate_all": float(
                evidence["projected_inside_tile"].astype(bool).mean()
            ),
            "projected_inside_tile_rate_accepted": (
                float(accepted_rows["projected_inside_tile"].astype(bool).mean())
                if len(accepted_rows)
                else None
            ),
            "projection_reprojection_rmse_px_accepted": numeric_summary(
                accepted_rows["inlier_reprojection_rmse_px"]
            ),
            "temporal_behavior": accepted_temporal_stats(evidence),
        },
        "development_reference_for_behavior_only": {
            "accepted_query_fraction": freeze["development_metrics"][
                "accepted_query_fraction"
            ],
            "contains_precision": freeze["development_metrics"][
                "contains_precision"
            ],
            "contains_recall": freeze["development_metrics"]["contains_recall"],
            "le80_precision": freeze["development_metrics"]["le80_precision"],
            "le80_recall": freeze["development_metrics"]["le80_recall"],
            "le40_precision": freeze["development_metrics"]["le40_precision"],
            "le40_recall": freeze["development_metrics"]["le40_recall"],
        },
        "scope_guarantees": {
            "threshold_retuned_on_blind_stress": False,
            "blind_reference_used": False,
            "blind_accuracy_claim_allowed": False,
            "candidate_switching_enabled": False,
            "top20_reranked": False,
            "bootstrap_enabled": False,
            "state_enabled": False,
            "temporal_fusion_enabled": False,
        },
        "promotion": rg["promotion"],
        "outputs": {
            "gate_freeze": str(freeze_path),
            "blind_anchor_evidence": str(evidence_path),
        },
    }

    report_path.write_text(
        json.dumps(report, indent=2, default=str),
        encoding="utf-8",
    )

    print("=" * 116)
    print("RG1.4 — PROVISIONAL ANCHOR-TRUST GATE BLIND-STRESS TRANSFER")
    print("=" * 116)
    print(
        "frozen development-derived gate:",
        f"inliers >= {threshold}",
    )
    print(
        "development accepted fraction:",
        freeze["development_metrics"]["accepted_query_fraction"],
    )
    print(
        "development recalls:",
        {
            "contains": freeze["development_metrics"]["contains_recall"],
            "le80": freeze["development_metrics"]["le80_recall"],
            "le40": freeze["development_metrics"]["le40_recall"],
        },
    )
    print()
    print("blind trajectory:", spec.trajectory_id)
    print("blind queries:", len(evidence))
    print("blind accepted:", acceptance_count)
    print("blind acceptance fraction:", acceptance_fraction)
    print("blind median inliers:", numeric_summary(evidence["inliers"])["median"])
    print(
        "blind accepted median inliers:",
        numeric_summary(accepted_rows["inliers"])["median"],
    )
    print(
        "blind accepted inside-tile rate:",
        (
            float(accepted_rows["projected_inside_tile"].astype(bool).mean())
            if len(accepted_rows)
            else None
        ),
    )
    print(
        "blind accepted projection-jump median m:",
        report["blind_stress"]["temporal_behavior"][
            "consecutive_accepted_projection_jump_m"
        ]["median"],
    )
    print("blind reference used:", False)
    print("blind accuracy computed:", False)
    print()
    print("STATUS: PASS_RG1_4_PROVISIONAL_GATE_BLIND_STRESS_TRANSFER")
    print("report:", report_path)


if __name__ == "__main__":
    main()
