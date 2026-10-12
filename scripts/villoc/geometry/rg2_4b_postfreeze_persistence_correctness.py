#!/usr/bin/env python3
"""RG2.4-B: development-only post-freeze persistence correctness audit."""
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
DEFAULT_CONFIG = ROOT / "configs/research/local_map_neighborhood_rg2_4b.yaml"

TRANSITION_REQUIRED = {
    "track_id","subset","method","threshold_name","link_threshold_m",
    "query_id","timestamp_s","consensus_easting","consensus_northing",
    "locality_index","anchor_tile_id",
}
RUN_REQUIRED = {
    "track_id","subset","method","threshold_name","link_threshold_m",
    "locality_index","start_query_id","end_query_id","query_count","duration_s",
    "consensus_path_length_m","consensus_max_pairwise_span_m",
    "start_end_displacement_m","unique_anchor_tiles","anchor_tile_changes",
    "distinct_supporting_tile_windows","support_redundancy_ratio",
    "radial_median_median_m","pairwise_p90_median_m","bbox_diagonal_median_m",
}


def resolve(v: str | Path) -> Path:
    p = Path(v).expanduser()
    return p if p.is_absolute() else ROOT / p


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"{label} missing columns: {missing}")


def verify_config(cfg: dict[str, Any]) -> None:
    if cfg.get("stage") != "RG2.4-B":
        raise RuntimeError("Expected RG2.4-B")
    if cfg.get("map_variant") != "768_s256" or cfg.get("map_crs") != "EPSG:3346":
        raise RuntimeError("Map contract changed")
    ac = cfg["audit_contract"]
    if ac.get("deployment_rule_selection_allowed") is not False:
        raise RuntimeError("Deployment rule selection must remain disabled")
    if ac.get("threshold_tuning_allowed") is not False:
        raise RuntimeError("Threshold tuning must remain disabled")
    boundary = cfg["scientific_boundary"]
    if boundary.get("development_reference_allowed") is not True:
        raise RuntimeError("Development reference must be explicitly allowed")
    if boundary.get("reference_after_rg2_4a_freeze_only") is not True:
        raise RuntimeError("Reference-after-freeze contract missing")
    for key in (
        "blind_stress_loaded","blind_stress_reference_allowed","locality_ids_mutable",
        "threshold_tuning_enabled","method_selection_enabled","bootstrap_enabled",
        "state_enabled","relative_frontend_fusion_enabled",
    ):
        if boundary.get(key) is not False:
            raise RuntimeError(f"{key} must be false")


def verify_rg24a(cfg: dict[str, Any]) -> tuple[Path, Path, dict[str, Any]]:
    block = cfg["rg2_4a"]
    report_path = resolve(block["report"])
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "PASS_RG2_4A_REFERENCE_FREE_LOCALITY_PERSISTENCE_FREEZE":
        raise RuntimeError("RG2.4-A report not PASS")
    boundary = report.get("scientific_boundary", {})
    for key in ("reference_loaded","rg2_3b_accuracy_loaded","bootstrap_modified","state_modified"):
        if boundary.get(key) is not False:
            raise RuntimeError(f"RG2.4-A boundary violation: {key}")

    dev = block["development"]
    transitions = resolve(dev["transitions"])
    runs = resolve(dev["runs"])
    if sha256_file(transitions) != dev["expected_transitions_sha256"]:
        raise RuntimeError("RG2.4-A development transition bytes changed")
    if sha256_file(runs) != dev["expected_runs_sha256"]:
        raise RuntimeError("RG2.4-A development run bytes changed")

    rep_dev = report["trajectories"]["development"]["outputs"]
    rep_blind = report["trajectories"]["blind_stress"]["outputs"]
    if rep_dev["transitions"]["sha256"] != dev["expected_transitions_sha256"]:
        raise RuntimeError("RG2.4-A development transition report hash mismatch")
    if rep_dev["runs"]["sha256"] != dev["expected_runs_sha256"]:
        raise RuntimeError("RG2.4-A development run report hash mismatch")
    blind = block["blind_stress"]
    if rep_blind["transitions"]["sha256"] != blind["expected_transitions_sha256"]:
        raise RuntimeError("RG2.4-A blind transition report hash mismatch")
    if rep_blind["runs"]["sha256"] != blind["expected_runs_sha256"]:
        raise RuntimeError("RG2.4-A blind run report hash mismatch")
    return transitions, runs, report


def corr(x: pd.Series, y: pd.Series) -> dict[str, float | None]:
    xv = pd.to_numeric(x, errors="coerce").astype(float)
    yv = pd.to_numeric(y, errors="coerce").astype(float)
    mask = np.isfinite(xv) & np.isfinite(yv)
    xv, yv = xv[mask], yv[mask]
    if len(xv) < 3 or xv.nunique() < 2 or yv.nunique() < 2:
        return {"pearson": None, "spearman": None}
    pearson = float(np.corrcoef(xv.to_numpy(), yv.to_numpy())[0, 1])
    xr = xv.rank(method="average").to_numpy()
    yr = yv.rank(method="average").to_numpy()
    spearman = float(np.corrcoef(xr, yr)[0, 1])
    return {"pearson": pearson, "spearman": spearman}


def error_summary(errors: pd.Series) -> dict[str, float | int]:
    e = pd.to_numeric(errors, errors="raise").astype(float)
    if len(e) == 0 or not np.isfinite(e.to_numpy()).all():
        raise RuntimeError("Finite non-empty errors required")
    return {
        "count": int(len(e)),
        "rmse_m": float(np.sqrt(np.mean(np.square(e)))),
        "mean_m": float(e.mean()),
        "median_m": float(e.median()),
        "p90_m": float(e.quantile(0.90)),
        "p95_m": float(e.quantile(0.95)),
        "le40": int((e <= 40.0).sum()),
        "le80": int((e <= 80.0).sum()),
    }


def evaluate(
    transitions: pd.DataFrame,
    runs: pd.DataFrame,
    reference: pd.DataFrame,
    cfg: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    ref = reference[["gt_x","gt_y"]].reset_index()
    qids = sorted(transitions["query_id"].astype(int).unique().tolist())
    if sorted(reference.index.astype(int).tolist()) != qids:
        raise RuntimeError("Reference query set differs from RG2.4-A development queries")

    t = transitions.merge(ref, on="query_id", how="left", validate="many_to_one")
    t["error_m"] = np.hypot(
        t["consensus_easting"].astype(float) - t["gt_x"].astype(float),
        t["consensus_northing"].astype(float) - t["gt_y"].astype(float),
    )

    keys = ["track_id","locality_index"]
    agg = t.groupby(keys, sort=True).agg(
        run_error_median_m=("error_m","median"),
        run_error_mean_m=("error_m","mean"),
        run_error_max_m=("error_m","max"),
        run_le40_fraction=("error_m", lambda s: float((s <= 40.0).mean())),
        run_le80_fraction=("error_m", lambda s: float((s <= 80.0).mean())),
    ).reset_index()

    r = runs.merge(agg, on=keys, how="left", validate="one_to_one")
    if r["run_error_median_m"].isna().any():
        raise RuntimeError("Run-error merge failed")

    min_wrong = int(cfg["audit_contract"]["persistent_wrong_definition"]["minimum_run_queries"])
    wrong_threshold = float(cfg["audit_contract"]["persistent_wrong_definition"]["median_error_gt_m"])
    r["persistent_wrong"] = (
        (pd.to_numeric(r["query_count"], errors="raise").astype(int) >= min_wrong)
        & (r["run_error_median_m"].astype(float) > wrong_threshold)
    )

    t = t.merge(
        r[["track_id","locality_index","query_count","persistent_wrong"]],
        on=keys, how="left", validate="many_to_one",
    )

    report: dict[str, Any] = {"tracks": {}}
    run_thresholds = [int(v) for v in cfg["audit_contract"]["run_length_thresholds"]]
    features = list(cfg["audit_contract"]["analyze_run_features"])

    for track_id, tg in t.groupby("track_id", sort=True):
        rg = r[r["track_id"] == track_id].copy()
        tg = tg.sort_values("query_id", kind="mergesort")
        block: dict[str, Any] = {
            "all_queries": error_summary(tg["error_m"]),
            "run_count": int(len(rg)),
            "singleton_run_count": int((rg["query_count"].astype(int) == 1).sum()),
            "persistent_wrong_run_count": int(rg["persistent_wrong"].astype(bool).sum()),
            "persistent_wrong_query_count": int(
                tg["persistent_wrong"].astype(bool).sum()
            ),
            "longest_persistent_wrong_run_queries": int(
                rg.loc[rg["persistent_wrong"], "query_count"].max()
            ) if rg["persistent_wrong"].any() else 0,
            "by_minimum_run_length": {},
            "run_feature_error_association": {},
        }
        for minimum in run_thresholds:
            qmask = tg["query_count"].astype(int) >= minimum
            rmask = rg["query_count"].astype(int) >= minimum
            block["by_minimum_run_length"][str(minimum)] = {
                "query_count": int(qmask.sum()),
                "query_fraction": float(qmask.mean()),
                "query_errors": (
                    error_summary(tg.loc[qmask, "error_m"])
                    if qmask.any() else None
                ),
                "run_count": int(rmask.sum()),
                "runs_median_error_le40": int(
                    (rg.loc[rmask, "run_error_median_m"] <= 40.0).sum()
                ),
                "runs_median_error_le80": int(
                    (rg.loc[rmask, "run_error_median_m"] <= 80.0).sum()
                ),
                "runs_median_error_gt80": int(
                    (rg.loc[rmask, "run_error_median_m"] > 80.0).sum()
                ),
            }

        for feature in features:
            if feature not in rg.columns:
                raise RuntimeError(f"Configured run feature missing: {feature}")
            block["run_feature_error_association"][feature] = corr(
                rg[feature], rg["run_error_median_m"]
            )
        report["tracks"][track_id] = block

    return t, r, report


def deterministic_csv(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, lineterminator="\n", float_format="%.9f").encode("utf-8")


def deterministic_json(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def write_immutable(files: dict[Path, bytes]) -> None:
    bad = [str(p) for p, data in files.items() if p.exists() and p.read_bytes() != data]
    if bad:
        raise RuntimeError("Refusing to overwrite differing RG2.4-B outputs: " + ", ".join(bad))
    for path, data in files.items():
        if path.exists():
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as f:
            f.write(data)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RG2.4-B post-freeze locality persistence correctness audit"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    cfg_path = resolve(args.config)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    verify_config(cfg)

    transition_path, run_path, rg24a_report = verify_rg24a(cfg)
    transitions = pd.read_csv(transition_path).copy()
    runs = pd.read_csv(run_path).copy()
    require_columns(transitions, TRANSITION_REQUIRED, "RG2.4-A transitions")
    require_columns(runs, RUN_REQUIRED, "RG2.4-A runs")
    transitions["query_id"] = pd.to_numeric(transitions["query_id"], errors="raise").astype(int)
    transitions["locality_index"] = pd.to_numeric(
        transitions["locality_index"], errors="raise"
    ).astype(int)
    runs["locality_index"] = pd.to_numeric(runs["locality_index"], errors="raise").astype(int)

    dev = cfg["rg2_4a"]["development"]
    if len(transitions) != int(dev["expected_transition_rows"]):
        raise RuntimeError("Development transition row count mismatch")
    if len(runs) != int(dev["expected_run_rows"]):
        raise RuntimeError("Development run row count mismatch")

    reference_path = resolve(cfg["development"]["reference_attachment"])
    print("Frozen RG2.4-A persistence tracks verified before reference attachment.")
    print("Attaching development reference for persistence correctness audit only.")
    reference = load_reference_xy(reference_path)

    query_eval, run_eval, analysis = evaluate(transitions, runs, reference, cfg)

    out = resolve(cfg["output_root"])
    qpath = out / "rg2_4b_query_persistence_accuracy.csv"
    rpath = out / "rg2_4b_locality_run_accuracy.csv"
    files = {
        qpath: deterministic_csv(query_eval),
        rpath: deterministic_csv(run_eval),
    }

    report = {
        "stage": "RG2.4-B",
        "status": "PASS_RG2_4B_POSTFREEZE_PERSISTENCE_CORRECTNESS_AUDIT",
        "scope": "development_only_postfreeze",
        "analysis": analysis,
        "scientific_boundary": {
            "rg2_4a_generated_before_reference": True,
            "locality_ids_changed_after_reference": False,
            "blind_stress_loaded": False,
            "blind_stress_reference_used": False,
            "threshold_tuning_executed": False,
            "method_selected": False,
            "bootstrap_modified": False,
            "state_modified": False,
            "relative_frontend_fused": False,
        },
        "provenance": {
            "config_sha256": sha256_file(cfg_path),
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "reference_attachment_sha256": sha256_file(reference_path),
            "rg2_4a_report_sha256": sha256_file(resolve(cfg["rg2_4a"]["report"])),
            "development_transitions_sha256": sha256_file(transition_path),
            "development_runs_sha256": sha256_file(run_path),
        },
        "outputs": {},
    }
    for label, path in (("query_accuracy", qpath), ("run_accuracy", rpath)):
        report["outputs"][label] = {
            "path": str(path.relative_to(ROOT)),
            "sha256": hashlib.sha256(files[path]).hexdigest(),
        }

    report_path = out / "rg2_4b_persistence_correctness_report.json"
    files[report_path] = deterministic_json(report)
    write_immutable(files)

    print("Status:", report["status"])
    for track_id in sorted(analysis["tracks"]):
        b = analysis["tracks"][track_id]
        p3 = b["by_minimum_run_length"]["3"]
        qerr = p3["query_errors"]
        print(
            f"{track_id}: runs={b['run_count']} persistent_wrong_runs={b['persistent_wrong_run_count']} "
            f"longest_wrong={b['longest_persistent_wrong_run_queries']} "
            f"q_in_run>=3={p3['query_count']} "
            + (
                f"run>=3 le40={qerr['le40']}/{qerr['count']} "
                f"le80={qerr['le80']}/{qerr['count']} median={qerr['median_m']:.3f}"
                if qerr else "run>=3 none"
            )
        )
    print("Blind stress loaded: False")
    print("Threshold tuning executed: False")
    print("Bootstrap/state modified: False")
    print("Report:", report_path)
    print("Run accuracy SHA256:", report["outputs"]["run_accuracy"]["sha256"])


if __name__ == "__main__":
    main()
