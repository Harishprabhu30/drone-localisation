#!/usr/bin/env python3
"""GD1.1: development-only post-freeze geometric discrimination audit."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import numpy as np
import pandas as pd
import yaml

from uavloc.evaluation.reference import load_reference_xy

ROOT = REPO_ROOT
DEFAULT_CONFIG = ROOT / "configs/research/top20_geometric_discrimination_gd1_1.yaml"

BASE_REQUIRED = {
    "query_id", "pool_rank", "tile_id", "source_view", "source_rank",
    "homography_ok", "finite_projection", "projected_easting",
    "projected_northing",
}
TILE_REQUIRED = {
    "tile_id", "left_easting", "right_easting", "bottom_northing",
    "top_northing", "center_easting", "center_northing",
}


def resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"{label} missing columns: {missing}")


def bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    mapping = {
        "true": True, "false": False, "1": True, "0": False,
        "yes": True, "no": False, "y": True, "n": False,
    }
    out = series.astype(str).str.strip().str.lower().map(mapping)
    return out.astype("boolean")


def numeric_feature(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(float)
    text = series.astype(str).str.strip().str.lower()
    boolish = text.map({
        "true": 1.0, "false": 0.0,
        "yes": 1.0, "no": 0.0,
        "y": 1.0, "n": 0.0,
    })
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.where(numeric.notna(), boolish).astype(float)


def verify_config(cfg: dict[str, Any]) -> None:
    if cfg.get("stage") != "GD1.1":
        raise RuntimeError("Expected GD1.1 config")
    if cfg.get("map_variant") != "768_s256" or cfg.get("map_crs") != "EPSG:3346":
        raise RuntimeError("Map contract changed")
    if cfg.get("labels") != ["contains_query", "center_le40", "center_le80"]:
        raise RuntimeError("GD1.1 label contract changed")
    boundary = cfg["scientific_boundary"]
    if boundary.get("development_reference_allowed") is not True:
        raise RuntimeError("Development reference must be explicitly allowed")
    if boundary.get("reference_after_gd1_0_freeze_only") is not True:
        raise RuntimeError("Reference-after-freeze contract missing")
    for key in (
        "blind_stress_loaded", "blind_stress_reference_allowed", "evidence_mutable",
        "reranking_promoted", "selector_promoted", "threshold_tuning_enabled",
        "learned_model_enabled", "learned_frontend_enabled", "bootstrap_enabled",
        "state_enabled",
    ):
        if boundary.get(key) is not False:
            raise RuntimeError(f"{key} must be false")


def verify_gd10(cfg: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    block = cfg["gd1_0"]
    report_path = resolve(block["report"])
    evidence_path = resolve(block["development_evidence"])
    if not report_path.is_file() or not evidence_path.is_file():
        raise FileNotFoundError("Missing GD1.0 freeze input")

    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "PASS_GD1_0_REFERENCE_FREE_TOP20_GEOMETRIC_EVIDENCE_FREEZE":
        raise RuntimeError("GD1.0 report not PASS")
    boundary = report.get("scientific_boundary", {})
    if boundary.get("reference_loaded") is not False:
        raise RuntimeError("GD1.0 reference boundary violated")
    if boundary.get("candidate_selected") is not False:
        raise RuntimeError("GD1.0 candidate-selection boundary violated")
    if boundary.get("reranking_executed") is not False:
        raise RuntimeError("GD1.0 reranking boundary violated")

    dev = report["trajectories"]["development"]["output"]["sha256"]
    blind = report["trajectories"]["blind_stress"]["output"]["sha256"]
    if dev != block["expected_development_sha256"]:
        raise RuntimeError("GD1.0 development report hash mismatch")
    if blind != block["expected_blind_sha256"]:
        raise RuntimeError("GD1.0 blind report hash mismatch")
    if sha256_file(evidence_path) != block["expected_development_sha256"]:
        raise RuntimeError("GD1.0 development evidence bytes changed")
    return evidence_path, report


def verify_tile_geometry(cfg: dict[str, Any]) -> Path:
    block = cfg["rg2_0"]
    report_path = resolve(block["report"])
    tile_path = resolve(block["tile_geometry"])
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "PASS_RG2_0_MAP_TOPOLOGY_PREFLIGHT":
        raise RuntimeError("RG2.0 map report not PASS")
    if sha256_file(tile_path) != report["provenance"]["tile_geometry_sha256"]:
        raise RuntimeError("RG2.0 tile geometry changed")
    return tile_path


def attach_reference_labels(
    evidence: pd.DataFrame,
    tile: pd.DataFrame,
    reference: pd.DataFrame,
) -> pd.DataFrame:
    require_columns(evidence, BASE_REQUIRED, "GD1.0 evidence")
    require_columns(tile, TILE_REQUIRED, "RG2.0 tile geometry")

    e = evidence.copy()
    e["query_id"] = pd.to_numeric(e["query_id"], errors="raise").astype(int)
    e["pool_rank"] = pd.to_numeric(e["pool_rank"], errors="raise").astype(int)
    e["tile_id"] = e["tile_id"].astype(str)

    t = tile[list(TILE_REQUIRED)].copy()
    t["tile_id"] = t["tile_id"].astype(str)
    if t["tile_id"].duplicated().any():
        raise RuntimeError("Duplicate tile geometry IDs")

    ref = reference[["gt_x", "gt_y"]].reset_index()
    out = (
        e.merge(t, on="tile_id", how="left", validate="many_to_one")
        .merge(ref, on="query_id", how="left", validate="many_to_one")
    )
    if out[
        [
            "left_easting", "right_easting", "bottom_northing", "top_northing",
            "center_easting", "center_northing", "gt_x", "gt_y",
        ]
    ].isna().any().any():
        raise RuntimeError("Reference/tile join produced missing values")

    out["contains_query"] = (
        (out["gt_x"] >= out["left_easting"])
        & (out["gt_x"] <= out["right_easting"])
        & (out["gt_y"] >= out["bottom_northing"])
        & (out["gt_y"] <= out["top_northing"])
    )
    out["center_error_m"] = np.hypot(
        out["center_easting"] - out["gt_x"],
        out["center_northing"] - out["gt_y"],
    )
    out["center_le40"] = out["center_error_m"] <= 40.0
    out["center_le80"] = out["center_error_m"] <= 80.0

    px = pd.to_numeric(out["projected_easting"], errors="coerce")
    py = pd.to_numeric(out["projected_northing"], errors="coerce")
    finite = np.isfinite(px) & np.isfinite(py)
    out["projection_error_m"] = np.nan
    out.loc[finite, "projection_error_m"] = np.hypot(
        px[finite] - out.loc[finite, "gt_x"],
        py[finite] - out.loc[finite, "gt_y"],
    )
    return out


def binary_auc(labels: pd.Series, scores: pd.Series) -> dict[str, Any]:
    frame = pd.DataFrame({
        "label": labels.astype(bool),
        "score": pd.to_numeric(scores, errors="coerce"),
    }).replace([np.inf, -np.inf], np.nan).dropna()

    positives = int(frame["label"].sum())
    negatives = int((~frame["label"]).sum())
    if positives == 0 or negatives == 0:
        return {
            "count": int(len(frame)),
            "positives": positives,
            "negatives": negatives,
            "auc": None,
            "auc_star": None,
            "direction": None,
        }

    ranks = frame["score"].rank(method="average", ascending=True)
    pos_rank_sum = float(ranks[frame["label"]].sum())
    auc = (
        pos_rank_sum - positives * (positives + 1) / 2.0
    ) / float(positives * negatives)
    if auc >= 0.5:
        direction = "higher"
        auc_star = auc
    else:
        direction = "lower"
        auc_star = 1.0 - auc
    return {
        "count": int(len(frame)),
        "positives": positives,
        "negatives": negatives,
        "auc": float(auc),
        "auc_star": float(auc_star),
        "direction": direction,
    }


def correlation(x: pd.Series, y: pd.Series) -> dict[str, float | None]:
    xv = numeric_feature(x)
    yv = pd.to_numeric(y, errors="coerce").astype(float)
    mask = np.isfinite(xv) & np.isfinite(yv)
    xv, yv = xv[mask], yv[mask]
    if len(xv) < 3 or xv.nunique() < 2 or yv.nunique() < 2:
        return {"count": int(len(xv)), "pearson": None, "spearman": None}
    pearson = float(np.corrcoef(xv.to_numpy(), yv.to_numpy())[0, 1])
    xr = xv.rank(method="average").to_numpy()
    yr = yv.rank(method="average").to_numpy()
    spearman = float(np.corrcoef(xr, yr)[0, 1])
    return {
        "count": int(len(xv)),
        "pearson": pearson,
        "spearman": spearman,
    }


def rank_hits(
    frame: pd.DataFrame,
    feature: str,
    label: str,
    direction: str,
    topks: list[int],
) -> dict[str, Any]:
    rows = []
    for qid, group in frame.groupby("query_id", sort=True):
        g = group.copy()
        g["_feature"] = numeric_feature(g[feature])
        g = g.replace([np.inf, -np.inf], np.nan).dropna(subset=["_feature"])
        if g.empty:
            rows.append((qid, 0, False, {}))
            continue
        ascending = direction == "lower"
        g = g.sort_values(
            ["_feature", "pool_rank", "tile_id"],
            ascending=[ascending, True, True],
            kind="mergesort",
        )
        availability = bool(group[label].astype(bool).any())
        hits = {
            int(k): bool(g.head(int(k))[label].astype(bool).any())
            for k in topks
        }
        rows.append((qid, len(g), availability, hits))

    return {
        "queries": int(len(rows)),
        "queries_with_finite_feature": int(sum(n > 0 for _, n, _, _ in rows)),
        "label_available_in_original_top20": int(sum(av for _, _, av, _ in rows)),
        "hits": {
            f"top{k}": int(sum(h.get(int(k), False) for _, _, _, h in rows))
            for k in topks
        },
    }


def retrieval_baseline(
    frame: pd.DataFrame,
    label: str,
    topks: list[int],
) -> dict[str, int]:
    return {
        f"top{k}": int(
            sum(
                group.sort_values("pool_rank").head(int(k))[label].astype(bool).any()
                for _, group in frame.groupby("query_id", sort=True)
            )
        )
        for k in topks
    }


def deterministic_csv(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, lineterminator="\n", float_format="%.9f").encode("utf-8")


def deterministic_json(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def write_immutable(files: dict[Path, bytes]) -> None:
    changed = [str(p) for p, data in files.items() if p.exists() and p.read_bytes() != data]
    if changed:
        raise RuntimeError("Refusing to overwrite differing GD1.1 outputs: " + ", ".join(changed))
    for path, data in files.items():
        if path.exists():
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as handle:
            handle.write(data)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GD1.1 post-freeze Top20 geometric discrimination audit"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()

    cfg_path = resolve(args.config)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    verify_config(cfg)

    evidence_path, gd10 = verify_gd10(cfg)
    tile_path = verify_tile_geometry(cfg)

    evidence = pd.read_csv(evidence_path).copy()
    tile = pd.read_csv(tile_path).copy()
    if len(evidence) != int(cfg["gd1_0"]["expected_development_rows"]):
        raise RuntimeError("GD1.0 development row count mismatch")
    if evidence["query_id"].nunique() != int(cfg["gd1_0"]["expected_development_queries"]):
        raise RuntimeError("GD1.0 development query count mismatch")

    # Critical boundary: freeze lineage verified before reference is opened.
    reference_path = resolve(cfg["development"]["reference_attachment"])
    print("Frozen GD1.0 evidence verified before reference attachment.")
    print("Attaching development reference for GD1.1 diagnostics only.")
    reference = load_reference_xy(reference_path)
    labeled = attach_reference_labels(evidence, tile, reference)

    feature_groups = gd10["evidence_groups"]
    features: list[str] = []
    for group_name in sorted(feature_groups):
        for feature in feature_groups[group_name]:
            if feature not in features:
                features.append(feature)
    missing = [f for f in features if f not in labeled.columns]
    if missing:
        raise RuntimeError(f"Configured GD1.0 evidence columns missing: {missing}")

    labels = list(cfg["labels"])
    topks = [int(v) for v in cfg["diagnostics"]["query_ranking_topk"]]
    rows: list[dict[str, Any]] = []
    analysis: dict[str, Any] = {
        "retrieval_baseline": {},
        "features": {},
        "label_counts": {},
    }

    for label in labels:
        analysis["retrieval_baseline"][label] = retrieval_baseline(
            labeled, label, topks
        )
        analysis["label_counts"][label] = {
            "positive_pairs": int(labeled[label].astype(bool).sum()),
            "queries_with_positive_in_top20": int(
                sum(
                    group[label].astype(bool).any()
                    for _, group in labeled.groupby("query_id", sort=True)
                )
            ),
        }

    for feature in features:
        fblock: dict[str, Any] = {
            "group": next(
                group_name
                for group_name, names in feature_groups.items()
                if feature in names
            ),
            "projection_error_correlation": correlation(
                labeled[feature], labeled["projection_error_m"]
            ),
            "labels": {},
        }
        values = numeric_feature(labeled[feature])
        finite_count = int(np.isfinite(values).sum())
        unique_count = int(values[np.isfinite(values)].nunique())
        fblock["finite_count"] = finite_count
        fblock["unique_finite_values"] = unique_count

        for label in labels:
            auc = binary_auc(labeled[label], values)
            ranking = None
            if auc["direction"] is not None:
                ranking = rank_hits(
                    labeled, feature, label, str(auc["direction"]), topks
                )
            fblock["labels"][label] = {
                "pair_auc": auc,
                "diagnostic_feature_ranking": ranking,
            }
            rows.append({
                "feature": feature,
                "group": fblock["group"],
                "label": label,
                "finite_count": finite_count,
                "unique_finite_values": unique_count,
                "auc": auc["auc"],
                "auc_star": auc["auc_star"],
                "direction": auc["direction"],
                "feature_top1_hits": (
                    ranking["hits"]["top1"] if ranking is not None else None
                ),
                "feature_top3_hits": (
                    ranking["hits"].get("top3") if ranking is not None else None
                ),
                "feature_top5_hits": (
                    ranking["hits"].get("top5") if ranking is not None else None
                ),
                "feature_top10_hits": (
                    ranking["hits"].get("top10") if ranking is not None else None
                ),
                "feature_top20_hits": (
                    ranking["hits"].get("top20") if ranking is not None else None
                ),
                "projection_error_pearson": fblock["projection_error_correlation"]["pearson"],
                "projection_error_spearman": fblock["projection_error_correlation"]["spearman"],
            })
        analysis["features"][feature] = fblock

    diagnostics = pd.DataFrame(rows).sort_values(
        ["label", "auc_star", "feature"],
        ascending=[True, False, True],
        kind="mergesort",
    ).reset_index(drop=True)

    report = {
        "stage": "GD1.1",
        "status": "PASS_GD1_1_POSTFREEZE_GEOMETRIC_DISCRIMINATION_AUDIT",
        "scope": "development_only_postfreeze",
        "analysis": analysis,
        "scientific_boundary": {
            "gd1_0_generated_before_reference": True,
            "gd1_0_evidence_changed_after_reference": False,
            "blind_stress_loaded": False,
            "blind_stress_reference_used": False,
            "reranker_promoted": False,
            "selector_promoted": False,
            "threshold_tuning_executed": False,
            "learned_model_trained": False,
            "learned_frontend_executed": False,
            "bootstrap_state_modified": False,
        },
        "interpretation": {
            "auc_direction_is_development_diagnostic_only": True,
            "feature_ranking_is_not_a_deployable_reranker": True,
            "no_multivariate_selector_fit": True,
            "no_frontend_winner_selected": True,
        },
        "provenance": {
            "config_sha256": sha256_file(cfg_path),
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "gd1_0_report_sha256": sha256_file(resolve(cfg["gd1_0"]["report"])),
            "development_evidence_sha256": sha256_file(evidence_path),
            "tile_geometry_sha256": sha256_file(tile_path),
            "reference_attachment_sha256": sha256_file(reference_path),
        },
        "outputs": {},
    }

    out = resolve(cfg["output_root"])
    labeled_path = out / "gd1_1_labeled_top20_geometric_evidence.csv"
    diag_path = out / "gd1_1_feature_discrimination.csv"
    files = {
        labeled_path: deterministic_csv(labeled),
        diag_path: deterministic_csv(diagnostics),
    }
    for label, path in (
        ("labeled_evidence", labeled_path),
        ("feature_discrimination", diag_path),
    ):
        report["outputs"][label] = {
            "path": str(path.relative_to(ROOT)),
            "sha256": hashlib.sha256(files[path]).hexdigest(),
            "rows": int(len(labeled) if label == "labeled_evidence" else len(diagnostics)),
        }

    report_path = out / "gd1_1_discrimination_report.json"
    files[report_path] = deterministic_json(report)
    write_immutable(files)

    print("Status:", report["status"])
    print("Retrieval baseline:")
    for label in labels:
        print(f"  {label}: {analysis['retrieval_baseline'][label]}")
    topn = int(cfg["diagnostics"]["top_features_to_print"])
    for label in labels:
        subset = diagnostics[diagnostics["label"] == label].head(topn)
        print(f"Top {len(subset)} features for {label}:")
        for row in subset.itertuples(index=False):
            print(
                f"  {row.feature}: auc*={row.auc_star:.4f} dir={row.direction} "
                f"feature_top1={row.feature_top1_hits}/403 "
                f"feature_top5={row.feature_top5_hits}/403"
            )
    print("Blind stress loaded: False")
    print("Reranker/selector promoted: False")
    print("Report:", report_path)
    print(
        "Feature discrimination SHA256:",
        report["outputs"]["feature_discrimination"]["sha256"],
    )


if __name__ == "__main__":
    main()
