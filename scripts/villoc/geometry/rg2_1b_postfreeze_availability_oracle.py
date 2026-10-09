#!/usr/bin/env python3
"""RG2.1-B: development-only post-freeze local-region availability oracle audit.

Candidate sets are verified byte-for-byte against RG2.1-A hashes before reference
is opened. No blind-stress table, image, descriptor, ORB, projection, bootstrap,
or temporal state is loaded or executed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

# Standalone scripts must be runnable directly from any working directory.
# Bootstrap the repository's src/ package path before importing uavloc.
REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import pandas as pd
import yaml

from uavloc.evaluation.reference import load_reference_xy


ROOT = REPO_ROOT
DEFAULT_CONFIG = ROOT / "configs/research/local_map_neighborhood_rg2_1b.yaml"


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def normalize_bool(series: pd.Series, *, label: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    mapping = {
        "true": True, "false": False,
        "1": True, "0": False,
        "yes": True, "no": False,
    }
    result = series.astype(str).str.strip().str.lower().map(mapping)
    if result.isna().any():
        bad = sorted(series[result.isna()].astype(str).unique().tolist())
        raise ValueError(f"{label} has non-boolean values: {bad[:10]}")
    return result.astype(bool)


def require_columns(frame: pd.DataFrame, required: set[str], *, label: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{label} missing columns: {missing}")


def verify_config(cfg: dict[str, Any]) -> None:
    if cfg.get("stage") != "RG2.1-B":
        raise ValueError("Expected RG2.1-B configuration")
    if cfg.get("map_variant") != "768_s256" or cfg.get("map_crs") != "EPSG:3346":
        raise ValueError("Frozen RG2 map contract changed")
    if cfg.get("sets") != ["anchor", "immediate", "full_overlap", "frozen_top20"]:
        raise ValueError("Candidate-set contract changed without versioning")
    boundary = cfg.get("scientific_boundary", {})
    expected_false = (
        "blind_stress_loaded", "blind_stress_reference_allowed",
        "candidate_sets_mutable", "qv_top20_reranking_allowed",
        "orb_enabled", "projection_enabled", "bootstrap_enabled",
        "state_enabled", "learned_model_enabled",
    )
    if boundary.get("development_reference_allowed") is not True:
        raise ValueError("Development reference must be explicitly allowed post-freeze")
    if boundary.get("reference_after_freeze_only") is not True:
        raise ValueError("Reference-after-freeze boundary not enabled")
    for key in expected_false:
        if boundary.get(key) is not False:
            raise ValueError(f"{key} must be false")
    interp = cfg.get("interpretation", {})
    if interp.get("same_budget_claim_allowed") is not False:
        raise ValueError("RG2.1-B must forbid same-budget superiority claims")


def verify_rg2_0(cfg: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    block = cfg["rg2_0"]
    report_path = resolve(block["report"])
    tile_path = resolve(block["tile_geometry"])
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "PASS_RG2_0_MAP_TOPOLOGY_PREFLIGHT":
        raise RuntimeError("RG2.0 topology preflight is not PASS")
    if report.get("map_variant") != "768_s256" or report.get("map_crs") != "EPSG:3346":
        raise RuntimeError("RG2.0 map contract mismatch")
    if report.get("topology", {}).get("tile_count") != 432:
        raise RuntimeError("Expected 432 RG2.0 tiles")
    if report["provenance"]["relationships_sha256"] != block["expected_relationships_sha256"]:
        raise RuntimeError("RG2.0 topology hash differs from frozen RG2.1-A topology")
    if sha256_file(tile_path) != report["provenance"]["tile_geometry_sha256"]:
        raise RuntimeError("RG2.0 tile geometry file changed after preflight")
    return tile_path, report


def verify_rg2_1a(cfg: dict[str, Any]) -> tuple[Path, Path, Path, dict[str, Any]]:
    block = cfg["rg2_1a"]
    dev = block["development"]
    report_path = resolve(block["report"])
    local_path = resolve(dev["local_neighborhoods"])
    summaries_path = resolve(dev["query_summaries"])
    top20_path = resolve(dev["frozen_qv_top20"])
    for path in (report_path, local_path, summaries_path, top20_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    actual = {
        "local_neighborhoods": sha256_file(local_path),
        "query_summaries": sha256_file(summaries_path),
        "frozen_qv_top20": sha256_file(top20_path),
    }
    expected = {
        "local_neighborhoods": dev["expected_local_neighborhoods_sha256"],
        "query_summaries": dev["expected_query_summaries_sha256"],
        "frozen_qv_top20": dev["expected_frozen_qv_top20_sha256"],
    }
    if actual != expected:
        raise RuntimeError(
            "RG2.1-A development freeze hashes changed: "
            + json.dumps({"expected": expected, "actual": actual}, sort_keys=True)
        )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "PASS_RG2_1A_REFERENCE_FREE_NEIGHBORHOOD_FREEZE":
        raise RuntimeError("RG2.1-A freeze report is not PASS")
    boundaries = report.get("boundaries", {})
    for key in (
        "reference_loaded", "srt_gps_loaded", "image_or_descriptor_loaded",
        "orb_executed", "candidate_generator_executed", "top20_reranked",
        "bootstrap_state_modified", "oracle_evaluation_executed",
    ):
        if boundaries.get(key) is not False:
            raise RuntimeError(f"RG2.1-A boundary violation in freeze report: {key}")

    rep_dev = report.get("trajectories", {}).get("development", {})
    if rep_dev.get("queries") != cfg["development"]["expected_queries"]:
        raise RuntimeError("RG2.1-A development query count mismatch")
    for label, expected_sha in expected.items():
        got = rep_dev.get("outputs", {}).get(label, {}).get("sha256")
        if got != expected_sha:
            raise RuntimeError(f"Freeze report hash mismatch for {label}")

    return local_path, summaries_path, top20_path, report


def load_and_validate_frozen_sets(
    local_path: Path,
    summaries_path: Path,
    top20_path: Path,
    *,
    expected_queries: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    local = pd.read_csv(local_path).copy()
    summaries = pd.read_csv(summaries_path).copy()
    top20 = pd.read_csv(top20_path).copy()

    require_columns(local, {
        "trajectory_id", "query_id", "anchor_tile_id", "tile_id",
        "union_rank", "is_anchor", "in_immediate", "in_full_overlap",
        "in_frozen_qv_top20", "frozen_qv_rank",
    }, label="RG2.1-A local neighborhoods")
    require_columns(summaries, {
        "trajectory_id", "query_id", "anchor_tile_id", "anchor_count",
        "immediate_count", "full_overlap_count", "union_count", "top20_count",
    }, label="RG2.1-A query summaries")
    require_columns(top20, {
        "trajectory_id", "query_id", "pool_rank", "tile_id",
        "source_view", "source_rank",
    }, label="RG2.1-A frozen Top20")

    for frame in (local, summaries, top20):
        frame["query_id"] = pd.to_numeric(frame["query_id"], errors="raise").astype(int)
        frame["tile_id" if "tile_id" in frame.columns else "anchor_tile_id"] = (
            frame["tile_id" if "tile_id" in frame.columns else "anchor_tile_id"].astype(str)
        )

    for col in ("is_anchor", "in_immediate", "in_full_overlap", "in_frozen_qv_top20"):
        local[col] = normalize_bool(local[col], label=col)
    local["union_rank"] = pd.to_numeric(local["union_rank"], errors="raise").astype(int)
    top20["pool_rank"] = pd.to_numeric(top20["pool_rank"], errors="raise").astype(int)
    top20["source_rank"] = pd.to_numeric(top20["source_rank"], errors="raise").astype(int)

    qids = sorted(summaries["query_id"].unique().tolist())
    if len(qids) != expected_queries or qids != sorted(local["query_id"].unique().tolist()):
        raise RuntimeError("RG2.1-A query IDs/count differ across local tables")
    if qids != sorted(top20["query_id"].unique().tolist()):
        raise RuntimeError("RG2.1-A query IDs differ between local and Top20 tables")
    if summaries["query_id"].duplicated().any():
        raise RuntimeError("Duplicate query summaries")

    for qid in qids:
        lg = local[local["query_id"] == qid].copy()
        tg = top20[top20["query_id"] == qid].copy()
        sg = summaries[summaries["query_id"] == qid].iloc[0]
        if lg["tile_id"].duplicated().any():
            raise RuntimeError(f"Duplicate local tile at query {qid}")
        if tg["tile_id"].duplicated().any():
            raise RuntimeError(f"Duplicate Top20 tile at query {qid}")
        if len(tg) != 20 or sorted(tg["pool_rank"].tolist()) != list(range(1, 21)):
            raise RuntimeError(f"Frozen Top20 budget/ranks changed at query {qid}")
        anchor = lg[lg["is_anchor"]]
        if len(anchor) != 1:
            raise RuntimeError(f"Expected one local anchor at query {qid}")
        top1 = tg[tg["pool_rank"] == 1]
        if len(top1) != 1 or str(top1.iloc[0]["tile_id"]) != str(anchor.iloc[0]["tile_id"]):
            raise RuntimeError(f"Anchor/Top20 Top1 mismatch at query {qid}")
        if str(top1.iloc[0]["source_view"]) != "center_square" or int(top1.iloc[0]["source_rank"]) != 1:
            raise RuntimeError(f"Frozen center-square Top1 contract changed at query {qid}")
        immediate = lg[lg["in_immediate"]]
        overlap = lg[lg["in_full_overlap"]]
        if not set(anchor["tile_id"]).issubset(set(immediate["tile_id"])):
            raise RuntimeError(f"Anchor not in immediate set at query {qid}")
        if not set(immediate["tile_id"]).issubset(set(overlap["tile_id"])):
            raise RuntimeError(f"Immediate not subset of full overlap at query {qid}")
        if int(sg["anchor_count"]) != 1:
            raise RuntimeError(f"Summary anchor count changed at query {qid}")
        if int(sg["immediate_count"]) != len(immediate):
            raise RuntimeError(f"Summary immediate count mismatch at query {qid}")
        if int(sg["full_overlap_count"]) != len(overlap):
            raise RuntimeError(f"Summary overlap count mismatch at query {qid}")
        if int(sg["union_count"]) != len(lg):
            raise RuntimeError(f"Summary union count mismatch at query {qid}")
        if int(sg["top20_count"]) != len(tg):
            raise RuntimeError(f"Summary Top20 count mismatch at query {qid}")
    return local, summaries, top20


def load_tile_geometry(path: Path) -> pd.DataFrame:
    tile = pd.read_csv(path).copy()
    require_columns(tile, {
        "tile_id", "left_easting", "right_easting", "bottom_northing",
        "top_northing", "center_easting", "center_northing",
    }, label="RG2.0 tile geometry")
    tile["tile_id"] = tile["tile_id"].astype(str)
    if tile["tile_id"].duplicated().any() or len(tile) != 432:
        raise RuntimeError("RG2.0 tile geometry uniqueness/count failure")
    for col in (
        "left_easting", "right_easting", "bottom_northing", "top_northing",
        "center_easting", "center_northing",
    ):
        tile[col] = pd.to_numeric(tile[col], errors="raise").astype(float)
    return tile


def attach_reference(
    candidates: pd.DataFrame,
    *,
    tile: pd.DataFrame,
    reference: pd.DataFrame,
) -> pd.DataFrame:
    qids = sorted(candidates["query_id"].astype(int).unique().tolist())
    missing_ref = sorted(set(qids) - set(reference.index.astype(int).tolist()))
    if missing_ref:
        raise RuntimeError(f"Reference missing query IDs: {missing_ref[:20]}")
    out = candidates.merge(
        tile[[
            "tile_id", "left_easting", "right_easting",
            "bottom_northing", "top_northing",
            "center_easting", "center_northing",
        ]],
        on="tile_id", how="left", validate="many_to_one",
    )
    if out[["center_easting", "center_northing"]].isna().any().any():
        raise RuntimeError("Candidate references unknown tile IDs")
    ref = reference[["gt_x", "gt_y"]].reset_index()
    out = out.merge(ref, on="query_id", how="left", validate="many_to_one")
    if out[["gt_x", "gt_y"]].isna().any().any():
        raise RuntimeError("Reference attachment failed")
    dx = out["center_easting"].astype(float) - out["gt_x"].astype(float)
    dy = out["center_northing"].astype(float) - out["gt_y"].astype(float)
    out["center_error_m"] = (dx * dx + dy * dy) ** 0.5
    out["contains_query"] = (
        (out["left_easting"] <= out["gt_x"])
        & (out["gt_x"] <= out["right_easting"])
        & (out["bottom_northing"] <= out["gt_y"])
        & (out["gt_y"] <= out["top_northing"])
    )
    out["le40"] = out["center_error_m"] <= 40.0
    out["le80"] = out["center_error_m"] <= 80.0
    return out


def budget_summary(counts: pd.Series) -> dict[str, Any]:
    values = pd.to_numeric(counts, errors="raise").astype(int)
    return {
        "min": int(values.min()),
        "median": float(values.median()),
        "mean": float(values.mean()),
        "p95": float(values.quantile(0.95)),
        "max": int(values.max()),
        "histogram": {
            str(int(k)): int(v)
            for k, v in sorted(values.value_counts().to_dict().items())
        },
    }


def set_hits(eval_rows: pd.DataFrame, membership: pd.Series) -> pd.DataFrame:
    frame = eval_rows[membership].copy()
    rows = []
    for qid, group in frame.groupby("query_id", sort=True):
        rows.append({
            "query_id": int(qid),
            "candidate_count": int(len(group)),
            "contain": bool(group["contains_query"].any()),
            "le40": bool(group["le40"].any()),
            "le80": bool(group["le80"].any()),
            "best_center_error_m": float(group["center_error_m"].min()),
        })
    return pd.DataFrame(rows).sort_values("query_id").reset_index(drop=True)


def pair_block(a: pd.Series, b: pd.Series) -> dict[str, int]:
    a = a.astype(bool)
    b = b.astype(bool)
    return {
        "both_hit": int((a & b).sum()),
        "a_only": int((a & ~b).sum()),
        "b_only": int((~a & b).sum()),
        "neither": int((~a & ~b).sum()),
    }


def summarize_set(name: str, frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "name": name,
        "queries": int(len(frame)),
        "candidate_budget": budget_summary(frame["candidate_count"]),
        "contain_hits": int(frame["contain"].sum()),
        "contain_rate": float(frame["contain"].mean()),
        "le40_hits": int(frame["le40"].sum()),
        "le40_rate": float(frame["le40"].mean()),
        "le80_hits": int(frame["le80"].sum()),
        "le80_rate": float(frame["le80"].mean()),
        "best_center_error_m": {
            "median": float(frame["best_center_error_m"].median()),
            "mean": float(frame["best_center_error_m"].mean()),
            "p95": float(frame["best_center_error_m"].quantile(0.95)),
        },
    }


def evaluate(
    local: pd.DataFrame,
    top20: pd.DataFrame,
    tile: pd.DataFrame,
    reference: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    local_eval = attach_reference(local, tile=tile, reference=reference)
    top20_eval = attach_reference(top20, tile=tile, reference=reference)

    sets = {
        "anchor": set_hits(local_eval, local_eval["is_anchor"]),
        "immediate": set_hits(local_eval, local_eval["in_immediate"]),
        "full_overlap": set_hits(local_eval, local_eval["in_full_overlap"]),
        "frozen_top20": set_hits(top20_eval, pd.Series(True, index=top20_eval.index)),
    }
    qids = sets["anchor"]["query_id"].tolist()
    for name, frame in sets.items():
        if frame["query_id"].tolist() != qids:
            raise RuntimeError(f"Query mismatch in evaluated set {name}")

    query = pd.DataFrame({"query_id": qids})
    for name, frame in sets.items():
        suffix = name
        for col in ("candidate_count", "contain", "le40", "le80", "best_center_error_m"):
            query[f"{suffix}_{col}"] = frame[col].to_numpy()

    # Nested geometry sets must be monotonic by construction.
    for label in ("contain", "le40", "le80"):
        if (
            query[f"anchor_{label}"] & ~query[f"immediate_{label}"]
        ).any():
            raise RuntimeError(f"Anchor->{label} immediate monotonicity violated")
        if (
            query[f"immediate_{label}"] & ~query[f"full_overlap_{label}"]
        ).any():
            raise RuntimeError(f"Immediate->{label} overlap monotonicity violated")
        if (
            query[f"anchor_{label}"] & ~query[f"frozen_top20_{label}"]
        ).any():
            raise RuntimeError(f"Anchor->{label} Top20 monotonicity violated")

    summaries = {name: summarize_set(name, frame) for name, frame in sets.items()}
    pairs: dict[str, dict[str, dict[str, int]]] = {}
    comparisons = [
        ("anchor", "immediate"),
        ("immediate", "full_overlap"),
        ("anchor", "full_overlap"),
        ("anchor", "frozen_top20"),
        ("immediate", "frozen_top20"),
        ("full_overlap", "frozen_top20"),
    ]
    for a, b in comparisons:
        pairs[f"{a}_vs_{b}"] = {
            label: pair_block(query[f"{a}_{label}"], query[f"{b}_{label}"])
            for label in ("contain", "le40", "le80")
        }
    return local_eval, top20_eval, query, {
        "sets": summaries,
        "paired_outcomes": pairs,
    }


def verify_historical_parity(cfg: dict[str, Any], query: pd.DataFrame) -> None:
    expected_top20 = cfg["historical_parity"]["frozen_top20"]
    expected_anchor = cfg["historical_parity"]["anchor"]
    if len(query) != expected_top20["query_count"]:
        raise RuntimeError("Historical query-count parity failed")
    actual_top20 = {
        "contain_hits": int(query["frozen_top20_contain"].sum()),
        "le40_hits": int(query["frozen_top20_le40"].sum()),
        "le80_hits": int(query["frozen_top20_le80"].sum()),
    }
    actual_anchor = {
        "contain_hits": int(query["anchor_contain"].sum()),
        "le40_hits": int(query["anchor_le40"].sum()),
        "le80_hits": int(query["anchor_le80"].sum()),
    }
    exp_top20 = {k: int(expected_top20[k]) for k in actual_top20}
    exp_anchor = {k: int(expected_anchor[k]) for k in actual_anchor}
    if actual_top20 != exp_top20:
        raise RuntimeError(
            "Frozen Top20 historical parity failed: "
            + json.dumps({"expected": exp_top20, "actual": actual_top20}, sort_keys=True)
        )
    if actual_anchor != exp_anchor:
        raise RuntimeError(
            "Anchor historical parity failed: "
            + json.dumps({"expected": exp_anchor, "actual": actual_anchor}, sort_keys=True)
        )


def write_csv_deterministic(frame: pd.DataFrame, path: Path) -> None:
    data = frame.to_csv(index=False, lineterminator="\n", float_format="%.6f")
    payload = data.encode("utf-8")
    if path.exists() and path.read_bytes() != payload:
        raise RuntimeError(f"Refusing to overwrite differing RG2.1-B output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with path.open("xb") as handle:
            handle.write(payload)


def write_json_deterministic(value: dict[str, Any], path: Path) -> None:
    payload = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False)
               + "\n").encode("utf-8")
    if path.exists() and path.read_bytes() != payload:
        raise RuntimeError(f"Refusing to overwrite differing RG2.1-B output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with path.open("xb") as handle:
            handle.write(payload)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RG2.1-B post-freeze local-region availability oracle audit"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    cfg_path = resolve(args.config)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    verify_config(cfg)

    tile_path, rg20_report = verify_rg2_0(cfg)
    local_path, summaries_path, top20_path, rg21a_report = verify_rg2_1a(cfg)
    local, summaries, top20 = load_and_validate_frozen_sets(
        local_path, summaries_path, top20_path,
        expected_queries=int(cfg["development"]["expected_queries"]),
    )
    tile = load_tile_geometry(tile_path)

    # Critical boundary: all candidate/topology validation above precedes opening
    # any reference attachment.
    reference_path = resolve(cfg["development"]["reference_attachment"])
    if not reference_path.is_file():
        raise FileNotFoundError(reference_path)
    print("Frozen candidate/topology inputs verified before reference attachment.")
    print("Attaching development reference for RG2.1-B oracle audit only.")
    reference = load_reference_xy(reference_path)
    if reference.index.duplicated().any():
        raise RuntimeError("Duplicate query IDs in reference attachment")
    qids = sorted(summaries["query_id"].astype(int).tolist())
    if sorted(reference.index.astype(int).tolist()) != qids:
        raise RuntimeError("Reference query set does not exactly match frozen development queries")

    local_eval, top20_eval, query_eval, analysis = evaluate(
        local, top20, tile, reference
    )
    verify_historical_parity(cfg, query_eval)

    out = resolve(cfg["output_root"])
    local_out = out / "rg2_1b_local_candidate_oracle_labels.csv"
    top20_out = out / "rg2_1b_frozen_top20_oracle_labels.csv"
    query_out = out / "rg2_1b_query_availability_comparison.csv"
    write_csv_deterministic(local_eval, local_out)
    write_csv_deterministic(top20_eval, top20_out)
    write_csv_deterministic(query_eval, query_out)

    report = {
        "stage": "RG2.1-B",
        "status": "PASS_RG2_1B_POSTFREEZE_AVAILABILITY_ORACLE_AUDIT",
        "scope": "development_only_postfreeze",
        "map_variant": "768_s256",
        "map_crs": "EPSG:3346",
        "query_count": int(len(query_eval)),
        "historical_parity": "PASS",
        "analysis": analysis,
        "provenance": {
            "config_sha256": sha256_file(cfg_path),
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "reference_attachment_sha256": sha256_file(reference_path),
            "rg2_0_report_sha256": sha256_file(resolve(cfg["rg2_0"]["report"])),
            "rg2_0_tile_geometry_sha256": sha256_file(tile_path),
            "rg2_0_relationships_sha256": rg20_report["provenance"]["relationships_sha256"],
            "rg2_1a_report_sha256": sha256_file(resolve(cfg["rg2_1a"]["report"])),
            "rg2_1a_local_neighborhoods_sha256": sha256_file(local_path),
            "rg2_1a_query_summaries_sha256": sha256_file(summaries_path),
            "rg2_1a_frozen_top20_sha256": sha256_file(top20_path),
        },
        "outputs": {
            "local_candidate_oracle_labels": {
                "path": str(local_out.relative_to(ROOT)),
                "sha256": sha256_file(local_out),
                "rows": int(len(local_eval)),
            },
            "frozen_top20_oracle_labels": {
                "path": str(top20_out.relative_to(ROOT)),
                "sha256": sha256_file(top20_out),
                "rows": int(len(top20_eval)),
            },
            "query_availability_comparison": {
                "path": str(query_out.relative_to(ROOT)),
                "sha256": sha256_file(query_out),
                "rows": int(len(query_eval)),
            },
        },
        "scientific_boundary": {
            "candidate_sets_generated_before_reference": True,
            "candidate_sets_changed_after_reference": False,
            "blind_stress_loaded": False,
            "blind_stress_reference_used": False,
            "orb_executed": False,
            "projection_executed": False,
            "bootstrap_state_modified": False,
            "same_budget_superiority_claim": False,
        },
        "interpretation": {
            "availability_not_selector": True,
            "full_overlap_budget_varies_and_often_exceeds_20": True,
            "overlapping_tiles_not_independent_evidence": True,
            "comparison_with_top20_is_coverage_cost_tradeoff_not_same_budget": True,
        },
    }
    report_out = out / "rg2_1b_availability_oracle_report.json"
    write_json_deterministic(report, report_out)

    # Report is written last; print the main scientific quantities explicitly.
    print("Status:", report["status"])
    for name in ("anchor", "immediate", "full_overlap", "frozen_top20"):
        s = analysis["sets"][name]
        print(
            f"{name}: candidates mean={s['candidate_budget']['mean']:.3f} "
            f"median={s['candidate_budget']['median']:.1f} "
            f"range={s['candidate_budget']['min']}..{s['candidate_budget']['max']} | "
            f"contain={s['contain_hits']}/{s['queries']} "
            f"le40={s['le40_hits']}/{s['queries']} "
            f"le80={s['le80_hits']}/{s['queries']}"
        )
    print("Paired outcomes (full_overlap vs frozen_top20):")
    print(json.dumps(
        analysis["paired_outcomes"]["full_overlap_vs_frozen_top20"],
        indent=2, sort_keys=True,
    ))
    print("Historical QV1.4 parity: PASS")
    print("Blind stress loaded: False")
    print("Same-budget superiority claim: False")
    print("Report:", report_out)
    print("Query comparison SHA256:", sha256_file(query_out))


if __name__ == "__main__":
    main()
